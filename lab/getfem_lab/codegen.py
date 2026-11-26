# Copyright (C) 2026 Yves Renard, Konstantinos Poulios.
#
# This file is a part of GetFEM
#
# GetFEM  is  free software;  you  can  redistribute  it  and/or modify it
# under  the  terms  of the  GNU  Lesser General Public License as published
# by  the  Free Software Foundation;  either version 3 of the License,  or
# (at your option) any later version along with the GCC Runtime Library
# Exception either version 3.1 or (at your option) any later version.
# This program  is  distributed  in  the  hope  that it will be useful,  but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or  FITNESS  FOR  A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License and GCC Runtime Library Exception for more details.
# You  should  have received a copy of the GNU Lesser General Public License
# along  with  this program. If not, see https://www.gnu.org/licenses/.

"""Translation of a model file into a readable, standalone GetFEM Python script.

The meshes (with their regions) are prepared separately by `mesher`; the
solve script only loads them. Each model of the file becomes one gf.Model:

    meshes -> finite element methods (shared) -> models (variables, data,
    macros, modules, terms, conditions) -> global solver -> post-processing

The script does not depend on getfem_lab: it can be read, modified and run
again by hand. Each module is a set of functions registered in MODULES,
which write its macros and terms in GWFL; its declarative description is
model.MODULE_SPECS (properties, options, conditions) used for validation.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import model as M, parameters as P


class ScriptWriter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.indent = ""

    def __call__(self, line: str = "") -> None:
        self.lines.append(self.indent + line if line else "")

    def section(self, title: str) -> None:
        self("")
        self(f"# --- {title} " + "-" * max(0, 70 - len(title)))

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def gwfl(value) -> str:
    """Model value -> piece of GWFL expression."""
    if isinstance(value, (int, float)):
        return repr(float(value))
    return f"({value.strip()})"


def gwfl_vec(value, dim: int) -> str:
    """Vector model value (list of components or GWFL expression) -> GWFL vector."""
    if isinstance(value, list):
        return "[" + ", ".join(gwfl(c) for c in value) + "]"
    return gwfl(value)


def gwfl_mat(value) -> str:
    """Matrix model value -> GWFL matrix: a list of rows gives [a, b; c, d]
    (rows, as in GWFL), a number its multiple of the identity."""
    if isinstance(value, list):
        return "[" + "; ".join(", ".join(gwfl(c) for c in row) for row in value) + "]"
    if isinstance(value, (int, float)):
        return f"({float(value)!r}*Id(meshdim))"
    return gwfl(value)


def gwfl_kind(value, kind: str) -> str:
    return gwfl_mat(value) if kind == M.MATRIX else gwfl_vec(value, 0) if kind == M.VECTOR else gwfl(value)


def ident(name: str) -> str:
    return re.sub(r"\W", "_", name)


# ---------------------------------------------------------------- context

@dataclass
class Ctx:
    """What the writing functions of a model need."""
    model: dict                   # the whole file (resolved, with defaults)
    md: dict                      # the model
    dim: int
    mesh: str                     # mesh name
    py: str                       # Python name of the gf.Model ("md_thermal")
    mfs: dict = field(default_factory=dict)      # variable/import name -> Python name of its MeshFem
    transient: bool = False       # solved in time (by the main solver)
    schemes: dict = field(default_factory=dict)  # variable in time -> its time scheme
    zero_time: list = field(default_factory=list)  # stationary here, in time in another global solver
    partial: bool = False         # solved by partial steps (some of its variables at a time)
    t0: float = 0.0               # the time at the start
    timed: list = field(default_factory=list)    # data depending on t: (name, mf, GWFL expression)
    dirichlet: dict = field(default_factory=dict)  # variable -> its Dirichlet conditions by simplification:
                                                   # [(region, data name or None: zero)]

    @property
    def explicit(self) -> dict:
        """The variables of an explicit scheme (Newmark, beta = 0): their schemes."""
        return {u: sc for u, sc in self.schemes.items() if M.is_explicit(sc)}

    @property
    def m(self) -> str:
        return f"m_{ident(self.mesh)}"

    @property
    def mim(self) -> str:
        return f"mim_{ident(self.mesh)}"

    @property
    def mf_data(self) -> str:
        return f"mf_data_{ident(self.md['name'])}"

    def rg(self, selection: str) -> str:
        return f"RG_{ident(self.mesh)}[{selection!r}]"

    def var(self, name: str) -> dict | None:
        return M.by_name(self.md["variables"], name)

    def kind(self, name: str) -> str:
        v = self.var(name)
        if v is not None:
            return M.variable_kind(v)
        im = M.by_name(self.md["imports"], name)
        src = M.model_by_name(self.model, im["model"])
        return M.variable_kind(M.by_name(src["variables"], im["variable"]))

    def value(self, value, kind: str) -> str:
        return gwfl_vec(value, self.dim) if kind == M.VECTOR else gwfl(value)

    def fem_data(self, w, name: str, mf: str, expr: str) -> None:
        """Data `name` on mf, interpolation of a GWFL expression; updated at
        each time step if it depends on the time t."""
        w(f"{self.py}.add_initialized_fem_data({name!r}, {mf}, {self.py}.interpolation({expr!r}, {mf}))")
        if uses_time(expr):
            self.timed.append((name, mf, expr))


def uses_time(expr: str) -> bool:
    """Whether a GWFL expression depends on the time t."""
    return re.search(r"(?<![\w.])t(?![\w(])", str(expr)) is not None


def fem_key(model: dict, var: dict, dim: int) -> tuple:
    c = var.get("components", 1)
    qdim = dim if c == "dim" else int(c)
    return (var["mesh"], var.get("fem", "lagrange"), int(var.get("degree", 1)), qdim)


def fem_name(key: tuple) -> str:
    mesh, fem, degree, qdim = key
    kind = "P" if fem == "lagrange" else "DP"
    return f"mf_{ident(mesh)}_{kind}{degree}" + ("" if qdim == 1 else f"_q{qdim}")


def used_meshes(model: dict) -> list[str]:
    """Names of the meshes the models use (each built or loaded once)."""
    out = []
    for md in model.get("models", []):
        for v in md.get("variables", []):
            if v.get("mesh") not in out:
                out.append(v.get("mesh"))
    return out


# ---------------------------------------------------------------- data

def coordinates(w: ScriptWriter, md_py: str, dim: int) -> None:
    """Macros x, y, z for the coordinates (see model.COORDINATES)."""
    for i, c in enumerate(M.COORDINATES[:dim], start=1):
        w(f"{md_py}.add_macro({c!r}, 'X({i})')")


def parameters(model: dict, md: dict, w: ScriptWriter, md_py: str) -> None:
    """Global then local parameters, as GetFEM data usable in every expression."""
    params = M.scope_parameters(model, md)
    if not params:
        return
    values, errors = M.parameter_values(model, md)
    if errors:
        raise ValueError("invalid parameters:\n  - " + "\n  - ".join(errors))
    w("# Parameters")
    for p in params:
        name, expr = p["name"], p["value"]
        notes = ([f"= {expr}"] if isinstance(expr, str) else []) + \
            ([p["description"]] if p.get("description") else [])
        comment = f"  # {', '.join(notes)}" if notes else ""
        v = values[name]
        sh = P.shape_of(v)
        if not sh:
            w(f"{md_py}.add_initialized_data({name!r}, [{float(v)!r}]){comment}")
        elif len(sh) == 1:
            w(f"{md_py}.add_initialized_data({name!r}, {[float(x) for x in v]!r}){comment}")
        else:                       # a matrix: its components by columns, and its sizes
            cols = [float(x) for x in v.flatten(order="F")]
            w(f"{md_py}.add_initialized_data({name!r}, {cols!r}, {list(sh)!r})"
              f"  # {P.format_value(v)}{comment.replace('  # ', ', ')}")


def _piecewise(c: Ctx, name: str, pieces: list[tuple[str, object, str]], what: str, w: ScriptWriter,
               kind: str = M.SCALAR) -> None:
    """`name` defined by (selection, value, comment) pieces; the last one wins.

    Numbers give finite element data (discontinuous, per element). With an
    expression, `name` is a macro instead, sum of the expressions times the
    indicators of where each one applies: it may then depend on the
    coordinates and on the variables (a conductivity depending on T).
    Vectors and matrices (kind) are always macros."""
    w("")
    py = f"{c.py}_{ident(name)}"
    numeric = kind == M.SCALAR and all(isinstance(v, (int, float)) for _, v, _ in pieces)
    w(f"covered = np.zeros({c.mf_data}.nbdof(), bool)")
    if numeric:
        w(f"{py} = np.zeros({c.mf_data}.nbdof())")
    parts = []
    for j, (selection, value, comment) in enumerate(reversed(pieces)):
        k = len(pieces) - j
        w(f"dofs = np.setdiff1d(on_region({c.mf_data}, {c.rg(selection)}), np.flatnonzero(covered))"
          f"  # {comment}")
        if numeric:
            w(f"{py}[dofs] = {float(value)!r}")
        else:
            chi = f"chi_{ident(name)}_{k}"
            w(f"{chi} = np.zeros({c.mf_data}.nbdof())")
            w(f"{chi}[dofs] = 1.0")
            w(f"{c.py}.add_initialized_fem_data({chi!r}, {c.mf_data}, {chi})")
            text = gwfl_kind(value, kind)
            parts.insert(0, text if len(pieces) == 1 and selection == M.ALL else f"{chi}*{text}")
        w("covered[dofs] = True")
    w("if not covered.all():")
    w(f"    raise RuntimeError(\"{what} '{name}' of model {c.md['name']} is undefined on part of the domain\")")
    if numeric:
        w(f"{c.py}.add_initialized_fem_data({name!r}, {c.mf_data}, {py})")
    else:
        w(f"{c.py}.add_macro({name!r}, {' + '.join(parts)!r})")


def _fields(c: Ctx, w: ScriptWriter) -> None:
    for f in M.model_fields(c.model, c.md):
        pieces = [(v["selection"], v["value"], v["selection"]) for v in f.get("values", [])]
        _piecewise(c, f["name"], pieces, "field", w, M.field_kind(f))


def _properties(c: Ctx, w: ScriptWriter) -> None:
    """Each property of the modules becomes discontinuous finite element data,
    defined per selection: plain 'k' can then be used in any expression. A
    property is a data of the model, shared by its modules (rho for the heat
    and the elasticity): declared once, by a module that uses it if any."""
    done = set()
    used = [(mod, prop) for mod in c.md["modules"] for prop in M.used_properties(c.model, c.md, mod)]
    given = [(mod, prop) for mod in c.md["modules"] for prop in M.spec_of(mod)["properties"]
             if mod["properties"].get(prop)]
    for mod, prop in used + given:
        name = M.data_name(mod, prop)
        if name not in done:
            done.add(name)
            pieces = []
            for pc in mod["properties"].get(prop, []):
                if "material" in pc:
                    mat = M.material(c.model, pc["material"])
                    pieces.append((pc["selection"], mat["properties"][prop],
                                   f"{pc['selection']}: {pc['material']}"))
                else:
                    pieces.append((pc["selection"], pc["value"], pc["selection"]))
            _piecewise(c, name, pieces, "property", w, M.property_kind(mod, prop))


# ---------------------------------------------------------------- modules
#
# Each module provides functions writing script lines (c: Ctx, mod: dict,
# i: index of the module, w):
#   declare : macros (also used by gwfl.py to check the expressions)
#   setup   : terms and conditions
#   post    : results entries and fields exported to VTU (list of
#             (mf, values, name) Python expressions)

def _heat_declare(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    u = mod["variable"]
    w(f"{c.py}.add_macro('q', '-k*Grad_{u}')  # heat flux vector")


def _term(c: Ctx, w: ScriptWriter, expr: str, rg: str | None = None) -> None:
    """A term of the weak formulation (linear if GetFEM accepts it as such)."""
    w(f"LINEAR[{c.md['name']!r}] &= add_term({c.py}, {c.mim}, {expr!r}" + (f", {rg})" if rg else ")"))


def _source(c: Ctx, w: ScriptWriter, expr: str, rg: str | None = None) -> None:
    """A source (a right-hand side): a term, not a source term of GetFEM, so
    that it may depend on the unknowns (f(T), a coupling: its tangent; still
    linear when it is affine)."""
    _term(c, w, "-(" + expr + ")", rg)


def _heat_setup(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    u, md = mod["variable"], c.py
    w(f"# -div(k grad {u}) = f")
    w(f"LINEAR[{c.md['name']!r}] &= add_term({md}, {c.mim}, 'k*Grad_{u}.Grad_Test_{u}')")
    for src in mod["sources"]:
        _source(c, w, gwfl(src["value"]) + "*Test_" + u, c.rg(src["selection"]))
    for j, bc in enumerate(mod["conditions"], start=1):
        rg = c.rg(bc["selection"])
        w("")
        if bc["type"] == "temperature":
            data = f"{u}_D{i}_{j}"
            mf = c.mfs[u]
            w(f"# Imposed temperature on {bc['selection']}")
            c.fem_data(w, data, mf, gwfl(bc["value"]))
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg}, {data!r})")
            c.dirichlet.setdefault(u, []).append((rg, data))
        elif bc["type"] == "heat_flux":
            w(f"# Imposed incoming flux on {bc['selection']}: k grad {u}.n = q")
            _source(c, w, gwfl(bc["value"]) + "*Test_" + u, rg)
        elif bc["type"] == "convection":
            h, text = gwfl(bc["h"]), gwfl(bc["T_ext"])
            w(f"# Convection on {bc['selection']}: -k grad {u}.n = h ({u} - T_ext)")
            _term(c, w, h + "*" + u + "*Test_" + u, rg)
            _source(c, w, h + "*" + text + "*Test_" + u, rg)


def _heat_post(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> list[str]:
    u, md = mod["variable"], c.py
    w(f"R['min_{u}'] = float({md}.variable({u!r}).min())")
    w(f"R['max_{u}'] = float({md}.variable({u!r}).max())")
    deg = max(int(c.var(u).get("degree", 1)) - 1, 0)
    w(f"mf_flux = gf.MeshFem({c.m}, {c.m}.dim())")
    w(f"mf_flux.set_classical_discontinuous_fem({deg})")
    w(f"flux = {md}.interpolation('q', mf_flux)")
    return ["mf_flux", "flux", "'heat_flux'"]


EPS0 = 8.8541878128e-12     # the permittivity of vacuum (F/m, CODATA 2018)


def _electrostatics_declare(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    u, md = mod["variable"], c.py
    w(f"{md}.add_initialized_data('eps0', [{EPS0!r}])  # the permittivity of vacuum (F/m)")
    w(f"{md}.add_macro('E_field', '-Grad_{u}')  # the electric field")
    w(f"{md}.add_macro('D_field', 'eps0*eps_r*E_field')  # the electric displacement")


def _electrostatics_setup(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    u, md = mod["variable"], c.py
    w(f"# -div(eps_r grad {u}) = rho_v/eps0 (the equation divided by eps0: a matrix of order 1)")
    _term(c, w, f"(eps_r*Grad_{u}).Grad_Test_{u}")
    for src in mod["sources"]:
        _source(c, w, "(" + gwfl(src["value"]) + ")/eps0*Test_" + u, c.rg(src["selection"]))
    for j, bc in enumerate(mod["conditions"], start=1):
        rg = c.rg(bc["selection"])
        w("")
        if bc["type"] == "potential":
            data = f"{u}_D{i}_{j}"
            w(f"# Imposed potential on {bc['selection']}")
            c.fem_data(w, data, c.mfs[u], gwfl(bc["value"]))
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg}, {data!r})")
            c.dirichlet.setdefault(u, []).append((rg, data))
        elif bc["type"] == "ground":
            w(f"# Ground on {bc['selection']}: {u} = 0")
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg})")
            c.dirichlet.setdefault(u, []).append((rg, None))
        elif bc["type"] == "surface_charge":
            w(f"# Surface charge on {bc['selection']}: -D.n = sigma_s")
            _source(c, w, "(" + gwfl(bc["value"]) + ")/eps0*Test_" + u, rg)
        elif bc["type"] == "floating":
            vf, lam = f"{u}_float{i}_{j}", f"{u}_lambda{i}_{j}"
            w(f"# Floating conductor on {bc['selection']}: {u} = {vf} (one unknown constant, by a multiplier "
              f"{lam} = D.n/eps0), its total charge -int D.n = Q")
            w(f"{md}.add_variable({vf!r}, 1)")
            w(f"{md}.add_filtered_fem_variable({lam!r}, {c.mfs[u]}, {rg})")
            w(f"area = gf.asm_generic({c.mim}, 0, '1', {rg}, {md})")
            _term(c, w, f"{lam}*Test_{u} + ({u} - {vf})*Test_{lam} - {lam}*Test_{vf}", rg)
            q = gwfl(bc["charge"])
            w(f"LINEAR[{c.md['name']!r}] &= add_term({md}, {c.mim}, '-(({q})/eps0/' + repr(area) + '*Test_{vf})', "
              f"{rg})   # (the charge Q spread on the boundary)")


def _electrostatics_post(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> list[str]:
    u, md = mod["variable"], c.py
    w(f"R['min_{u}'] = float({md}.variable({u!r}).min())")
    w(f"R['max_{u}'] = float({md}.variable({u!r}).max())")
    w(f"R['electric_energy'] = float(gf.asm_generic({c.mim}, 0, '0.5*D_field.E_field', -1, {md}))")
    conductors = [(j, bc) for j, bc in enumerate(mod["conditions"], start=1)
                  if bc["type"] in ("potential", "ground", "floating")]
    if conductors:
        w("# The charge of each conductor, Q = -int_Gamma D.n (n: the outward normal of the domain)")
        sources = [(gwfl(src["value"]), c.rg(src["selection"])) for src in mod["sources"]]
        sources += [(gwfl(bc["value"]), c.rg(bc["selection"])) for bc in mod["conditions"]
                    if bc["type"] == "surface_charge"]
        _conductor_charges(c, mod, i, w, conductors, f"eps0*eps_r*Grad_{u}", sources, "charges", "floating")
    deg = max(int(c.var(u).get("degree", 1)) - 1, 0)
    w(f"mf_E = gf.MeshFem({c.m}, {c.m}.dim())")
    w(f"mf_E.set_classical_discontinuous_fem({deg})")
    w(f"E_field = {md}.interpolation('E_field', mf_E)")
    w(f"D_field = {md}.interpolation('D_field', mf_E)")
    return ["mf_E", "E_field", "'electric_field'", "mf_E", "D_field", "'electric_displacement'"]


def _conductor_charges(c: Ctx, mod: dict, i: int, w: ScriptWriter, conductors, flux: str, sources,
                       key: str, unknown: str) -> None:
    """The charge (current) of each conductor, from the equation tested by
    phi, 1 on its nodes and 0 elsewhere: int flux.grad phi - int sources phi
    (as accurate as the energy; the flux on the boundary less)."""
    u, md, mf = mod["variable"], c.py, c.mfs[mod["variable"]]
    w(f"R[{key!r}], R['potentials'] = {{}}, {{}}")
    w(f"{md}.add_initialized_fem_data('{u}__phi', {mf}, np.zeros({mf}.nbdof()))")
    for j, bc in conductors:
        sel = bc["selection"]
        w(f"phi = np.zeros({mf}.nbdof())")
        w(f"phi[{mf}.basic_dof_on_region({c.rg(sel)})] = 1.0")
        w(f"{md}.set_variable('{u}__phi', phi)")
        w(f"Q = gf.asm_generic({c.mim}, 0, '({flux}).Grad_{u}__phi', -1, {md})")
        for expr, rg in sources:
            w(f"Q -= gf.asm_generic({c.mim}, 0, '({expr})*{u}__phi', {rg}, {md})")
        w(f"R[{key!r}][{sel!r}] = float(Q)")
        if bc["type"] == unknown:
            w(f"R['potentials'][{sel!r}] = float({md}.variable('{u}_float{i}_{j}')[0])")


def _currents_declare(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    u, md = mod["variable"], c.py
    w(f"{md}.add_macro('E_field', '-Grad_{u}')  # the electric field")
    w(f"{md}.add_macro('J_field', 'sigma_e*E_field')  # the current density")
    w(f"{md}.add_macro('Q_joule', 'J_field.E_field')  # the Joule heat (W/m3)")


def _heat_of(c: Ctx) -> dict | None:
    return next((m for m in c.md["modules"] if m["type"] == "heat_conduction"), None)


def _currents_setup(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    u, md = mod["variable"], c.py
    w(f"# -div(sigma_e grad {u}) = Q_j, divided by a mean conductivity sigma_ref (a matrix of order 1: the")
    w("# conductivities of metals, 1e7 S/m, would bring the residual to the round-off)")
    w(f"sigma_ref = gf.asm_generic({c.mim}, 0, 'Trace(sigma_e*Id(meshdim))/meshdim', -1, {md}) / "
      f"gf.asm_generic({c.mim}, 0, '1', -1, {md})")
    w("sigma_ref = sigma_ref if np.isfinite(sigma_ref) and sigma_ref > 0 else 1.0")
    w(f"{md}.add_initialized_data('sigma_ref', [sigma_ref])")
    _term(c, w, f"(sigma_e/sigma_ref*Grad_{u}).Grad_Test_{u}")
    for src in mod["sources"]:
        _source(c, w, "(" + gwfl(src["value"]) + ")/sigma_ref*Test_" + u, c.rg(src["selection"]))
    if mod.get("joule_heating") == "heat_conduction" and _heat_of(c) is not None:
        T = _heat_of(c)["variable"]
        w(f"# the Joule heat, a source of the heat conduction: J.E")
        _source(c, w, f"Q_joule*Test_{T}")
    for j, bc in enumerate(mod["conditions"], start=1):
        rg = c.rg(bc["selection"])
        w("")
        if bc["type"] == "potential":
            data = f"{u}_D{i}_{j}"
            w(f"# Imposed potential on {bc['selection']}")
            c.fem_data(w, data, c.mfs[u], gwfl(bc["value"]))
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg}, {data!r})")
            c.dirichlet.setdefault(u, []).append((rg, data))
        elif bc["type"] == "ground":
            w(f"# Ground on {bc['selection']}: {u} = 0")
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg})")
            c.dirichlet.setdefault(u, []).append((rg, None))
        elif bc["type"] == "current_density":
            w(f"# Normal current density entering on {bc['selection']}: -J.n = j_n")
            _source(c, w, "(" + gwfl(bc["value"]) + ")/sigma_ref*Test_" + u, rg)
        elif bc["type"] == "terminal":
            vt, lam = f"{u}_float{i}_{j}", f"{u}_lambda{i}_{j}"
            w(f"# Terminal on {bc['selection']}: {u} = {vt} (one unknown constant, by a multiplier "
              f"{lam}), its total current entering -int J.n = I")
            w(f"{md}.add_variable({vt!r}, 1)")
            w(f"{md}.add_filtered_fem_variable({lam!r}, {c.mfs[u]}, {rg})")
            w(f"area = gf.asm_generic({c.mim}, 0, '1', {rg}, {md})")
            _term(c, w, f"{lam}*Test_{u} + ({u} - {vt})*Test_{lam} - {lam}*Test_{vt}", rg)
            q = gwfl(bc["current"])
            w(f"LINEAR[{c.md['name']!r}] &= add_term({md}, {c.mim}, '-(({q})/sigma_ref/' + repr(area) + "
              f"'*Test_{vt})', "
              f"{rg})   # (the current I spread on the boundary)")


def _currents_post(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> list[str]:
    u, md = mod["variable"], c.py
    w(f"R['min_{u}'] = float({md}.variable({u!r}).min())")
    w(f"R['max_{u}'] = float({md}.variable({u!r}).max())")
    w(f"R['joule_power'] = float(gf.asm_generic({c.mim}, 0, 'Q_joule', -1, {md}))")
    conductors = [(j, bc) for j, bc in enumerate(mod["conditions"], start=1)
                  if bc["type"] in ("potential", "ground", "terminal")]
    if conductors:
        w("# The current entering by each electrode, I = -int_Gamma J.n (n: the outward normal of the domain)")
        sources = [(gwfl(src["value"]), c.rg(src["selection"])) for src in mod["sources"]]
        sources += [(gwfl(bc["value"]), c.rg(bc["selection"])) for bc in mod["conditions"]
                    if bc["type"] == "current_density"]
        _conductor_charges(c, mod, i, w, conductors, f"sigma_e*Grad_{u}", sources, "currents", "terminal")
    deg = max(int(c.var(u).get("degree", 1)) - 1, 0)
    w(f"mf_E = gf.MeshFem({c.m}, {c.m}.dim())")
    w(f"mf_E.set_classical_discontinuous_fem({deg})")
    w(f"mf_Q = gf.MeshFem({c.m}, 1)")
    w(f"mf_Q.set_classical_discontinuous_fem({deg})")
    w(f"E_field = {md}.interpolation('E_field', mf_E)")
    w(f"J_field = {md}.interpolation('J_field', mf_E)")
    w(f"Q_joule = {md}.interpolation('Q_joule', mf_Q)")
    return ["mf_E", "E_field", "'electric_field'", "mf_E", "J_field", "'current_density'",
            "mf_Q", "Q_joule", "'joule_heat'"]


def _expansion_of(c: Ctx, u: str) -> dict | None:
    return next((m for m in c.md["modules"] if m["type"] == "thermal_expansion"
                 and m["displacement"] == u), None)


def _elasticity_declare(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    u, md, dim = mod["variable"], c.py, c.dim
    plane = mod.get("plane", "strain")
    w("# Lame coefficients from E and nu")
    w(f"{md}.add_macro('mu_L', 'E/(2*(1+nu))')")
    if dim == 2 and plane == "stress":
        w(f"{md}.add_macro('lambda_L', 'E*nu/(1-sqr(nu))')  # plane stress")
    else:
        w(f"{md}.add_macro('lambda_L', 'E*nu/((1+nu)*(1-2*nu))')"
          f"{'  # plane strain' if dim == 2 else ''}")
    w(f"{md}.add_macro('eps', 'Sym(Grad_{u})')")
    exp = _expansion_of(c, u)
    if exp is not None:
        beta = "2*lambda_L+2*mu_L" if dim == 2 and plane == "stress" else \
            "lambda_L+2*mu_L" if dim == 1 else "3*lambda_L+2*mu_L"
        w("# Thermal expansion: sigma = lambda tr(eps) I + 2 mu eps - beta alpha (T - T_ref) I")
        w(f"{md}.add_macro('beta_L', {beta!r})")
        w(f"{md}.add_macro('theta_L', {'alpha*(' + exp['temperature'] + ' - ' + gwfl(exp['T_ref']) + ')'!r})")
        w(f"{md}.add_macro('sigma', 'lambda_L*Trace(eps)*Id(meshdim) + 2*mu_L*eps "
          f"- beta_L*theta_L*Id(meshdim)')")
    else:
        w(f"{md}.add_macro('sigma', 'lambda_L*Trace(eps)*Id(meshdim) + 2*mu_L*eps')")
    if dim == 3:
        w(f"{md}.add_macro('von_mises', 'sqrt(3/2)*Norm(Deviator(sigma))')")
    elif dim == 2:
        szz = "0" if plane == "stress" else \
            ("lambda_L*Trace(eps) - beta_L*theta_L" if exp is not None else "lambda_L*Trace(eps)")
        w(f"{md}.add_macro('sigma_zz', {szz!r})  # out-of-plane stress")
        w(f"{md}.add_macro('von_mises', 'sqrt(0.5*(sqr(sigma(1,1)-sigma(2,2)) + sqr(sigma(2,2)-sigma_zz)"
          " + sqr(sigma_zz-sigma(1,1))) + 3*sqr(sigma(1,2)))')")
    else:
        w(f"{md}.add_macro('von_mises', 'abs(sigma(1,1))')")
    if any(bc["type"] == "roller" for bc in mod["conditions"]):
        w("# Roller conditions use Nitsche's method: no extra unknown, so they can share")
        w("# nodes with other displacement conditions. GetFEM's penalty is 1/(gamma0*h).")
        w(f"{md}.add_initialized_data('nitsche_gamma0', "
          f"[1e-2 / np.max(np.abs({md}.interpolation('E', {c.mf_data})))])")


def _elasticity_setup(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    u, md, dim = mod["variable"], c.py, c.dim
    w(f"# -div sigma({u}) = f")
    w(f"LINEAR[{c.md['name']!r}] &= add_term({md}, {c.mim}, 'sigma:Grad_Test_{u}')")
    for src in mod["sources"]:
        _source(c, w, gwfl_vec(src["value"], dim) + ".Test_" + u, c.rg(src["selection"]))
    for j, bc in enumerate(mod["conditions"], start=1):
        rg = c.rg(bc["selection"])
        w("")
        if bc["type"] == "fixed":
            w(f"# Fixed on {bc['selection']}: {u} = 0")
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg})")
            c.dirichlet.setdefault(u, []).append((rg, None))
        elif bc["type"] == "displacement":
            data = f"{u}_D{i}_{j}"
            mf = c.mfs[u]
            w(f"# Imposed displacement on {bc['selection']}")
            c.fem_data(w, data, mf, gwfl_vec(bc["value"], dim))
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg}, {data!r})")
            c.dirichlet.setdefault(u, []).append((rg, data))
        elif bc["type"] == "roller":
            w(f"# Roller on {bc['selection']}: {u}.n = 0, free tangential displacement")
            w(f"{md}.add_normal_Dirichlet_condition_with_Nitsche_method({c.mim}, {u!r}, 'sigma*Normal', "
              f"'nitsche_gamma0', {rg}, 1.0)")
        elif bc["type"] == "traction":
            w(f"# Surface traction on {bc['selection']}: sigma.n = g")
            _source(c, w, gwfl_vec(bc["value"], dim) + ".Test_" + u, rg)
        elif bc["type"] == "pressure":
            w(f"# Pressure on {bc['selection']}: sigma.n = -p n")
            _source(c, w, "-" + gwfl(bc["value"]) + "*(Normal.Test_" + u + ")", rg)


def _elasticity_post(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> list[str]:
    u, md = mod["variable"], c.py
    w(f"R['max_displacement'] = float(np.max(np.linalg.norm({md}.variable({u!r}).reshape(-1, {c.m}.dim()), axis=1)))")
    deg = max(int(c.var(u).get("degree", 1)) - 1, 0)
    w(f"mf_vm = gf.MeshFem({c.m}, 1)")
    w(f"mf_vm.set_classical_discontinuous_fem({deg})")
    w(f"von_mises = {md}.interpolation('von_mises', mf_vm)")
    w("R['max_von_mises'] = float(von_mises.max())")
    w(f"R['warp'] = {u!r}")
    return ["mf_vm", "von_mises", "'von_mises'"]


def _pde_parts(c: Ctx, mod: dict) -> dict:
    """The pieces of a PDE module: its unknown, whether it is a vector, the
    nonzero coefficients among its properties."""
    spec = M.spec_of(mod)
    props = mod.get("properties", {})
    nonzero = {p for p in spec["properties"] if not _is_zero(props.get(p))}
    u = mod["variable"]
    n = {q: f"{q}_{u}" for q in ("K", "b", "j", "v", "r", "d", "m", "F", "tau")}
    return {"u": u, "n": n, "vector": mod.get("unknown") == "vector", "spec": spec, "nz": nonzero,
            "general": mod.get("form") == "general",
            "aniso": mod.get("diffusion") == "anisotropic"}


def _pde_declare(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    p = _pde_parts(c, mod)
    u, md, n, nz = p["u"], c.py, p["n"], p["nz"]
    if p["general"]:
        return                                      # (F_u: a property)
    diff = (f"Grad_{u}*{n['K']}" if p["vector"] else f"{n['K']}*Grad_{u}") if "K" in nz else "0"
    flux = diff + (f" + {u}*{n['b']}" if "b" in nz else "") + (f" - {n['j']}" if "j" in nz else "")
    w(f"{md}.add_macro({n['F']!r}, {flux!r})  # the flux: K grad {u} + b {u} - j")
    if mod.get("stabilization") == "supg" and "stabilization" in p["spec"]["options"]:
        cc = f"Trace({n['K']})/meshdim" if p["aniso"] else n["K"]
        k = int(c.var(u).get("degree", 1))
        h = "element_size" if k == 1 else f"(element_size/{k})"
        w(f"# SUPG: tau = (|2 v/h|^2 + 9 |4 K/h^2|^2)^(-1/2), h the size of the element"
          f"{'' if k == 1 else ' over the degree'}")
        w(f"{md}.add_macro({n['tau']!r}, '1/sqrt(sqr(2*Norm({n['v']})/{h}) "
          f"+ 9*sqr(4*({cc})/sqr({h})) + 1e-300)')")


def _pde_setup(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    p = _pde_parts(c, mod)
    u, md, nz, vec, n = p["u"], c.py, p["nz"], p["vector"], p["n"]
    lin = f"LINEAR[{c.md['name']!r}] &= add_term({md}, {c.mim}, "
    dot, colon = (".", ":") if vec else ("*", ".")
    test = f"Test_{u}"

    def source(expr: str, rg: str | None = None) -> None:
        _source(c, w, expr, rg)
    if p["general"]:
        w("# -div F = f")
        w(f"{lin}{n['F'] + colon + 'Grad_' + test!r})")
    else:
        w(f"# -div(K grad {u} + b {u} - j) + v.grad {u} + r {u} = f")
        K, b, v, r = n["K"], n["b"], n["v"], n["r"]
        if "K" in nz:
            w(f"{lin}{(f'(Grad_{u}*{K}):Grad_{test}' if vec else f'({K}*Grad_{u}).Grad_{test}')!r})")
        if "b" in nz:
            w(f"{lin}{f'{u}*{b}.Grad_{test}'!r})")
        if "v" in nz:
            w(f"{lin}{(f'(Grad_{u}*{v}).{test}' if vec else f'({v}.Grad_{u})*{test}')!r})")
        if "r" in nz:
            w(f"{lin}{(f'({r}*{u}).{test}' if vec else f'{r}*{u}*{test}')!r})")
        if "j" in nz:
            source(n["j"] + colon + "Grad_" + test)
    for src in mod["sources"]:
        f = gwfl_vec(src["value"], c.dim) if vec else gwfl(src["value"])
        source(f + dot + test, c.rg(src["selection"]))
    if mod.get("stabilization") == "supg" and "stabilization" in p["spec"]["options"] and "v" in nz:
        _supg(c, mod, p, w)
    for j, bc in enumerate(mod["conditions"], start=1):
        rg = c.rg(bc["selection"])
        value = (gwfl_vec(bc["value"], c.dim) if vec else gwfl(bc["value"])) if "value" in bc else None
        w("")
        if bc["type"] == "imposed_value":
            data = f"{u}_D{i}_{j}"
            w(f"# Imposed value on {bc['selection']}")
            c.fem_data(w, data, c.mfs[u], value)
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg}, {data!r})")
            c.dirichlet.setdefault(u, []).append((rg, data))
        elif bc["type"] == "flux":
            w(f"# Imposed flux on {bc['selection']}: F.n = g")
            source(value + dot + test, rg)
        elif bc["type"] == "robin":
            q = gwfl_mat(bc["q"]) if vec else gwfl(bc["q"])
            w(f"# Robin condition on {bc['selection']}: F.n = g - q {u}")
            w(f"{lin}{(f'({q}*{u}).{test}' if vec else f'{q}*{u}*{test}')!r}, {rg})")
            source(value + dot + test, rg)


def _supg(c: Ctx, mod: dict, p: dict, w: ScriptWriter) -> None:
    """SUPG: tau (v.grad w) times the residual of the equation (w the test
    function), added on each element (the second derivatives of u from
    degree 2: K Laplacian u, the gradient of K neglected; the time
    derivatives in time)."""
    u, nz, md, n = p["u"], p["nz"], c.py, p["n"]
    res = [f"{n['v']}.Grad_{u}"]
    if "b" in nz:
        res.append(f"{n['b']}.Grad_{u}")
    if "r" in nz:
        res.append(f"{n['r']}*{u}")
    if "K" in nz and int(c.var(u).get("degree", 1)) >= 2:
        res.append(f"-Trace({n['K']}*Hess_{u})" if p["aniso"] else f"-{n['K']}*Trace(Hess_{u})")
    inertia = p["spec"].get("inertia")
    if inertia and c.transient and M.in_time(c.model, c.md, mod):
        if inertia["order"] == 1:
            res.append(f"{n['d']}*Dot_{u}")
        else:
            res.append(f"{n['m']}*{u}__inertia")
            if "d" in nz:
                res.append(f"{n['d']}*Dot_{u}")
    weight = f"{n['tau']}*({n['v']}.Grad_Test_{u})"
    w("# SUPG stabilization: tau (v.grad w) (residual of the equation)")
    w(f"LINEAR[{c.md['name']!r}] &= add_term({md}, {c.mim}, {weight + '*(' + ' + '.join(res) + ')'!r})")
    for src in mod["sources"]:
        w(f"LINEAR[{c.md['name']!r}] &= add_term({md}, {c.mim}, {'-' + weight + '*(' + gwfl(src['value']) + ')'!r}, "
          f"{c.rg(src['selection'])})")


def _pde_post(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> list[str]:
    p = _pde_parts(c, mod)
    u, md = p["u"], c.py
    if p["vector"]:
        w(f"R['max_{u}'] = float(np.max(np.linalg.norm({md}.variable({u!r}).reshape(-1, {c.m}.dim()), axis=1)))")
        return []
    w(f"R['min_{u}'] = float({md}.variable({u!r}).min())")
    w(f"R['max_{u}'] = float({md}.variable({u!r}).max())")
    deg = max(int(c.var(u).get("degree", 1)) - 1, 0)
    w(f"mf_flux_{u} = gf.MeshFem({c.m}, {c.m}.dim())")
    w(f"mf_flux_{u}.set_classical_discontinuous_fem({deg})")
    w(f"flux_{u} = {md}.interpolation({p['n']['F']!r}, mf_flux_{u})")
    return [f"mf_flux_{u}", f"flux_{u}", repr(f"flux_{u}")]


def _nothing(c: Ctx, mod: dict, i: int, w: ScriptWriter):
    return []


MODULES = {
    "heat_conduction": (_heat_declare, _heat_setup, _heat_post),
    "electrostatics": (_electrostatics_declare, _electrostatics_setup, _electrostatics_post),
    "electric_currents": (_currents_declare, _currents_setup, _currents_post),
    "linear_elasticity": (_elasticity_declare, _elasticity_setup, _elasticity_post),
    "thermal_expansion": (_nothing, _nothing, _nothing),   # written by the elasticity module
    "pde": (_pde_declare, _pde_setup, _pde_post),
}


# ---------------------------------------------------------------- terms and generic conditions

def _terms(c: Ctx, w: ScriptWriter) -> None:
    for t in c.md["terms"]:
        w(f"# Term {t['name']}")
        w(f"LINEAR[{c.md['name']!r}] &= add_term({c.py}, {c.mim}, {str(t['expr'])!r}, "
          f"{c.rg(t['selection'])}, {t['linear']!r})")


def _conditions(c: Ctx, w: ScriptWriter) -> None:
    md = c.py
    for i, cd in enumerate(c.md["conditions"], start=1):
        u, rg, kind = cd["variable"], c.rg(cd["selection"]), c.kind(cd["variable"])
        test = ".Test_" if kind == M.VECTOR else "*Test_"
        w("")
        t = cd["type"]
        if t in ("dirichlet", "normal_dirichlet"):
            data = f"{u}_C{i}"
            normal = t == "normal_dirichlet"
            w(f"# {'Normal component' if normal else 'Dirichlet condition'} on {cd['selection']} "
              f"({cd['method']})")
            if normal:
                c.fem_data(w, data, f"mf_data_scalar_{ident(c.md['name'])}", gwfl(cd["value"]))
            else:
                c.fem_data(w, data, c.mfs[u], c.value(cd["value"], kind))
            deg = int(c.var(u).get("degree", 1))
            name = "add_normal_Dirichlet_condition" if normal else "add_Dirichlet_condition"
            if cd["method"] == "simplification":
                w(f"{md}.{name}_with_simplification({u!r}, {rg}, {data!r})")
                if not normal:
                    c.dirichlet.setdefault(u, []).append((rg, data))
            elif cd["method"] == "multiplier" and c.partial:     # (its multiplier disabled with u)
                w(f"ib = {md}.{name}_with_multipliers({c.mim}, {u!r}, {deg}, {rg}, {data!r})")
                w(f"MULTIPLIERS[{c.md['name']!r}].setdefault({u!r}, []).append({md}.mult_varname_Dirichlet(ib))")
            elif cd["method"] == "multiplier":
                w(f"{md}.{name}_with_multipliers({c.mim}, {u!r}, {deg}, {rg}, {data!r})")
            else:
                w(f"{md}.{name}_with_penalization({c.mim}, {u!r}, {float(cd.get('penalty', 1e10))!r}, "
                  f"{rg}, {data!r})")
        elif t == "neumann":
            w(f"# Neumann condition on {cd['selection']}: incoming flux")
            _source(c, w, c.value(cd["value"], kind) + test + u, rg)
        elif t == "robin":
            h, g = gwfl(cd["h"]), c.value(cd["value"], kind)
            w(f"# Robin condition on {cd['selection']}: flux = h ({u} - g)")
            _term(c, w, h + "*" + u + test + u, rg)
            _source(c, w, h + "*" + g + test + u, rg)


# ---------------------------------------------------------------- models

def contexts(model: dict) -> list[Ctx]:
    out = []
    for md in model["models"]:
        mesh = M.model_mesh(model, md)
        dim = M.geometry_dim(model, M.geometry_of(model, mesh))
        c = Ctx(model, md, int(dim), mesh["name"], f"md_{ident(md['name'])}")
        c.partial = md["name"] in M.partial_models(M.main_solver(model))
        if M.is_transient(model, md):
            solver = M.main_solver(model)
            c.transient = True
            c.schemes = {u: M.scheme_of(model, md, u) for u in M.time_variables(model, md)}
            c.t0 = M.time_values(model, solver)[0]
        if len(model.get("solvers", [])) > 1:      # (its time derivatives zero here)
            c.zero_time = [u for u in M.timeable_variables(md)
                           if u not in c.schemes and M.scheme_anywhere(model, md, u) is not None]
        for v in md["variables"]:
            c.mfs[v["name"]] = fem_name(fem_key(model, v, c.dim))
        for im in md["imports"]:
            src = M.model_by_name(model, im["model"])
            c.mfs[im["name"]] = fem_name(fem_key(model, M.by_name(src["variables"], im["variable"]), c.dim))
        out.append(c)
    return out


def time_declarations(c: Ctx, w: ScriptWriter) -> None:
    """In time: the time t of the GWFL expressions (its start value; 0 for
    a stationary model), the time step and the scheme of each variable in
    time: Dot_u from u and the previous values (Previous_u, Previous_Dot_u,
    u__prev2 for BDF2), Dot_u = (c0 u + c1 u_n + c2 u_{n-1})/dt + c3 Dot_u_n,
    the coefficients in the data u__scheme (those of the first step, set
    again after it by shift())."""
    md = c.py
    w(f"{md}.set_time({c.t0!r})  # the time t of the expressions")
    for u in c.zero_time:
        w(f"# {u}: stationary for this solver (in time for another one): its time derivatives are zero")
        w(f"{md}.add_macro('Previous_{u}', '{u}')")
        for d in ("Dot", "Previous_Dot") + (("Dot2", "Previous_Dot2") if M.variable_order(c.md, u) == 2 else ()):
            w(f"{md}.add_macro('{d}_{u}', '0*{u}')")
    if not c.transient:
        return
    w(f"{md}.set_time_step(1.0)  # the time step timestep (set by the solver)")
    w("try:      # (timestep: a data of the model, assembled again when it changes)")
    w(f"    {md}.variable('timestep')")
    w("except RuntimeError:")
    w("    raise RuntimeError('this GetFEM is too old for the transient solver (its time step must be a "
      "data of the model)') from None")
    for u, sc in c.schemes.items():
        mf = c.mfs[u]
        if M.is_explicit(sc):
            _explicit_declarations(c, u, sc, w)
            continue
        if M.SCHEME_ORDER[sc["type"]] == 2:
            _second_order_declarations(c, u, sc, w)
            continue
        first = list(M.scheme_coefficients(sc, first=True))
        w(f"# Time scheme of {u}: {M.scheme_label(sc)} (the equation written at t_n+1),")
        w(f"#   Dot_{u} = (c0*{u} + c1*Previous_{u} + c2*{u}__prev2)/timestep + c3*Previous_Dot_{u}")
        w(f"{md}.add_initialized_data('{u}__scheme', {first!r})  # c0, c1, c2, c3 (first step)")
        w(f"{md}.add_initialized_fem_data('{u}__prev', {mf}, np.zeros({mf}.nbdof()))")
        w(f"{md}.add_initialized_fem_data('{u}__rate', {mf}, np.zeros({mf}.nbdof()))")
        two = sc.get("type") == "bdf2"
        if two:
            w(f"{md}.add_initialized_fem_data('{u}__prev2', {mf}, np.zeros({mf}.nbdof()))")
        w(f"{md}.add_macro('Previous_{u}', '{u}__prev')")
        w(f"{md}.add_macro('Previous_Dot_{u}', '{u}__rate')")
        prev2 = f" + {u}__scheme(3)*{u}__prev2" if two else ""
        w(f"{md}.add_macro('Dot_{u}', '({u}__scheme(1)*{u} + {u}__scheme(2)*{u}__prev{prev2})/timestep "
          f"+ {u}__scheme(4)*{u}__rate')")


def _second_order_declarations(c: Ctx, u: str, sc: dict, w: ScriptWriter) -> None:
    """A second order scheme: the previous displacement, velocity,
    acceleration (and memory) of u, the macros Dot2_u, Dot_u and
    u__inertia (the acceleration of the inertia term), their coefficients
    in the data u__acc, u__vel, u__inertia_c (second_order_coefficients)."""
    md, mf = c.py, c.mfs[u]
    k = M.second_order_coefficients(sc)
    b, g, am, af = M.second_order_parameters(sc)
    mem = k["inertia"][3] != 0.0
    w(f"# Time scheme of {u}: {M.scheme_label(sc)} (beta = {b:.6g}, gamma = {g:.6g}, alpha_m = {am:.6g}, "
      f"alpha_f = {af:.6g});")
    w("#   the equation written at t_n+1, the inertia with the acceleration")
    w(f"#   ((1 - alpha_m) Dot2_{u} + alpha_m Previous_Dot2_{u} - alpha_f {u}__mem)/(1 - alpha_f), {u}__mem its "
      "value at the previous step")
    w(f"{md}.add_initialized_data('{u}__acc', {k['acc']!r})  # Dot2_{u}: A0, A1, A2")
    w(f"{md}.add_initialized_data('{u}__vel', {k['vel']!r})  # Dot_{u}: V0, V1, V2")
    w(f"{md}.add_initialized_data('{u}__inertia_c', {k['inertia']!r})  # {u}__inertia: I0, I1, I2, I3")
    for name in ("prev", "rate", "rate2") + (("mem",) if mem else ()):
        w(f"{md}.add_initialized_fem_data('{u}__{name}', {mf}, np.zeros({mf}.nbdof()))")
    w(f"{md}.add_macro('Previous_{u}', '{u}__prev')")
    w(f"{md}.add_macro('Previous_Dot_{u}', '{u}__rate')")
    w(f"{md}.add_macro('Previous_Dot2_{u}', '{u}__rate2')")
    d = f"({u} - {u}__prev)"
    w(f"{md}.add_macro('Dot2_{u}', '{u}__acc(1)*{d}/sqr(timestep) + {u}__acc(2)*{u}__rate/timestep "
      f"+ {u}__acc(3)*{u}__rate2')")
    w(f"{md}.add_macro('Dot_{u}', '{u}__vel(1)*{d}/timestep + {u}__vel(2)*{u}__rate "
      f"+ {u}__vel(3)*timestep*{u}__rate2')")
    m = f" + {u}__inertia_c(4)*{u}__mem" if mem else ""
    w(f"{md}.add_macro('{u}__inertia', '{u}__inertia_c(1)*{d}/sqr(timestep) + {u}__inertia_c(2)*{u}__rate/timestep "
      f"+ {u}__inertia_c(3)*{u}__rate2{m}')")


def _explicit_declarations(c: Ctx, u: str, sc: dict, w: ScriptWriter) -> None:
    """An explicit scheme: Newmark beta = 0 (u_n+1 = u_n + dt v_n + dt^2/2
    a_n known before the step, M a_n+1 = -G(u_n+1, v*, t_n+1), v* = v_n +
    (1 - gamma) dt a_n, v_n+1 = v* + gamma dt a_n+1), or the explicit
    generalized-alpha (M atilde_n = -G(u_n, v_n, t_n), atilde_n the memory
    u__mem: a_n+1 = (atilde_n - alpha_m a_n)/(1 - alpha_m), then u_n+1,
    v_n+1, all known before the step). explicit_predict_<model>() sets them
    at the new time, the implicit variables of the model are solved with
    them given, explicit_solve_<model>() solves the mass system with all the
    variables at the new time. Dot2_u, Dot_u: the data u__a, u__v; the
    inertia term of the model: zero (the mass solved apart)."""
    md, mf = c.py, c.mfs[u]
    am, b, g = M.explicit_parameters(sc)
    w(f"# Time scheme of {u}: {M.scheme_label(sc)} (alpha_m = {am:.6g}, beta = {b:.6g}, gamma = {g:.6g}):")
    w(f"#   {u} known before the step (explicit_predict_{ident(c.md['name'])}), the mass system solved "
      f"after it (explicit_solve_{ident(c.md['name'])})")
    names = ("prev", "rate", "rate2", "a", "v") + (("mem",) if sc["type"] != "explicit_newmark" else ())
    for name in names:
        w(f"{md}.add_initialized_fem_data('{u}__{name}', {mf}, np.zeros({mf}.nbdof()))")
    w(f"{md}.add_macro('Previous_{u}', '{u}__prev')")
    w(f"{md}.add_macro('Previous_Dot_{u}', '{u}__rate')")
    w(f"{md}.add_macro('Previous_Dot2_{u}', '{u}__rate2')")
    w(f"{md}.add_macro('Dot2_{u}', '{u}__a')")
    w(f"{md}.add_macro('Dot_{u}', '{u}__v')")
    w(f"{md}.add_macro('{u}__inertia', '0*{u}__a')   # (the inertia: the mass system, apart)")


def _explicit_code(c: Ctx, w: ScriptWriter) -> None:
    """explicit_setup_<model>(): the mass matrix of each explicit variable
    (its free dofs: those of no Dirichlet condition), factorized once or
    lumped, the stability limit of the time step (the largest eigenvalue of
    M^-1 K by power iterations, dt omega_max below the limit of the scheme);
    explicit_predict_<model>(predict) and explicit_solve_<model>(predict):
    a time step (predict False: the initial acceleration, at the current
    values). On the Dirichlet dofs, the imposed values and their time
    derivatives (central differences in time of their expression)."""
    md, name = c.py, ident(c.md["name"])
    info = {}
    for u, sc in c.explicit.items():
        mass = None
        for mod in c.md["modules"]:
            inertia = M.spec_of(mod).get("inertia")
            if inertia and mod.get(inertia["variable"]) == u:
                mass = inertia["expr"].replace("{u}", u).replace(f"{u}__inertia", u)
        timed = {n: e for n, _mf, e in c.timed}
        bcs = [(rg, d, timed.get(d)) for rg, d in c.dirichlet.get(u, [])]
        info[u] = (sc, M.explicit_parameters(sc), sc.get("mass", "consistent") == "lumped", mass, bcs)
    w("")
    w(f"EXPLICIT_{name} = {{}}   # variable: (i0, n, free dofs, Dirichlet dofs, M solve, M_ID)")
    for u, (sc, (am, b, g), lumped, mass, bcs) in info.items():
        mf = c.mfs[u]
        w("")
        w(f"def bc_{name}_{ident(u)}(t, dt):")
        w(f'    """{u} imposed at t on its Dirichlet dofs, its velocity, its acceleration (zero elsewhere)."""')
        w.indent = "    "
        w(f"n = {md}.interval_of_variable({u!r})[1]")
        w("out = [np.zeros(n), np.zeros(n), np.zeros(n)]")
        for rg, d, expr in bcs:
            if d is None:
                continue
            w(f"dofs = np.asarray({mf}.basic_dof_on_region({rg}), dtype=int)")
            if expr is None:
                w(f"out[0][dofs] = {md}.variable({d!r})[dofs]")
            else:
                w(f"g = time_derivatives({md}, {expr!r}, {mf}, t, dt)")
                w("for k in range(3):")
                w("    out[k][dofs] = g[k][dofs]")
        w("return out")
        w.indent = ""
    w("")
    w(f"def explicit_setup_{name}():")
    w(f'    """The mass matrices of the explicit variables of {c.md["name"]!r}, the stability limit."""')
    w.indent = "    "
    w(f"{md}.assembly('build_matrix')")
    w(f"K = {md}.tangent_matrix()")
    w("dt = (T_END - T_START) / N_STEPS")
    for u, (sc, (am, b, g), lumped, mass, bcs) in info.items():
        mf = c.mfs[u]
        limit = M.explicit_limit(sc)
        w(f"i0, n = {md}.interval_of_variable({u!r})")
        w(f"Mfull = gf.asm_generic({c.mim}, 2, {mass!r}, -1, {md})   # (the inertia term: its mass)")
        w("D = np.unique(np.concatenate([np.zeros(0, dtype=int)] + ["
          + ", ".join(f"np.asarray({mf}.basic_dof_on_region({rg}), dtype=int)" for rg, _d, _e in bcs) + "]))")
        w("I = np.setdiff1d(np.arange(n), D)")
        w("Muu = gf.Spmat('copy', Mfull, range(i0, i0 + n), range(i0, i0 + n))")
        w(f"solve_M, M_ID = mass_solver(Muu, I, D, {lumped!r}, {u!r})")
        w(f"EXPLICIT_{name}[{u!r}] = (i0, n, I, D, solve_M, M_ID)")
        w("KII = gf.Spmat('copy', gf.Spmat('copy', K, range(i0, i0 + n), range(i0, i0 + n)), list(I), list(I))")
        w("omega = np.sqrt(max(power_iteration(lambda x: solve_M(np.asarray(KII.mult(x))), len(I)), 0.0))")
        w(f"limit = {limit!r} / omega if omega > 0 else np.inf   # (dt omega_max <= {limit:.4g})")
        w(f"results['models'][{c.md['name']!r}].setdefault('stable_time_step', {{}})[{u!r}] = float(limit)")
        w("if dt > limit:")
        w(f"    raise RuntimeError(f'explicit scheme of {u}: the time step {{dt:.4g}} exceeds its stability "
          f"limit {{limit:.4g}} (dt omega_max <= {limit:.4g}): at least "
          "{int(np.ceil((T_END - T_START) / limit))} time steps')")
        w("if dt > 0.9 * limit:")
        w(f"    results.setdefault('warnings', []).append(f'explicit scheme of {u}: the time step {{dt:.4g}} "
          "close to its stability limit {limit:.4g}')")
    w.indent = ""
    w("")
    w(f"def explicit_predict_{name}(predict=True):")
    w('    """The explicit variables at the new time (known before the step: from the previous values;')
    w('    predict False: the current ones, for the initial acceleration), their velocity and')
    w('    acceleration (Newmark beta = 0: v_n + dt a_n and a_n meanwhile, for the implicit variables),')
    w('    on the Dirichlet dofs the imposed values and their derivatives."""')
    w.indent = "    "
    w(f"dt = float({md}.variable('timestep')[0])")
    w("t = T_NOW[0]")
    for u, (sc, (am, b, g), lumped, mass, bcs) in info.items():
        w(f"i0, n, I, D, solve_M, M_ID = EXPLICIT_{name}[{u!r}]")
        w(f"un, vn, an = ({md}.variable('{u}__prev'), {md}.variable('{u}__rate'), {md}.variable('{u}__rate2'))")
        w("if predict:")
        if sc["type"] == "explicit_newmark":
            w("    u1, v1, a1 = un + dt * vn + 0.5 * dt * dt * an, vn + dt * an, an")
        else:
            w(f"    a1 = ({md}.variable('{u}__mem') - {am!r} * an) / {1 - am!r}")
            w(f"    u1 = un + dt * vn + dt * dt * ({0.5 - b!r} * an + {b!r} * a1)")
            w(f"    v1 = vn + dt * ({1 - g!r} * an + {g!r} * a1)")
        w("else:")
        w(f"    u1, v1, a1 = {md}.variable({u!r}), vn, 0 * an")
        w("u1, v1, a1 = np.array(u1), np.array(v1), np.array(a1)")
        if bcs:
            w(f"g = bc_{name}_{ident(u)}(t, dt)")
            if sc["type"] == "explicit_newmark":
                w("u1[D], v1[D], a1[D] = g[0][D], g[1][D], g[2][D]")
            else:           # (a1: the acceleration of the scheme, from the memory, also there)
                w("u1[D], v1[D] = g[0][D], g[1][D]")
                w("if not predict:")
                w("    a1[D] = g[2][D]")
        w(f"{md}.set_variable({u!r}, u1)")
        w(f"{md}.set_variable('{u}__v', v1)")
        w(f"{md}.set_variable('{u}__a', a1)")
    w.indent = ""
    w("")
    w(f"def explicit_solve_{name}(predict=True):")
    w('    """The mass system, the other variables at the new time: M a = -G (G: the residual without')
    w('    the inertia). Newmark beta = 0: a_n+1 (the velocity v* + gamma dt a_n+1; v*, the damping, at the')
    w('    predicted velocity); generalized-alpha: the memory atilde_n+1, the acceleration at t_n+1 (on the')
    w('    dofs of a Dirichlet condition, that of the imposed value). predict False: the initial')
    w('    acceleration (and memory)."""')
    w.indent = "    "
    w(f"dt = float({md}.variable('timestep')[0])")
    w("t = T_NOW[0]")
    for u, (sc, (am, b, g), lumped, mass, bcs) in info.items():
        if sc["type"] == "explicit_newmark":
            w(f"vn, an = {md}.variable('{u}__rate'), {md}.variable('{u}__rate2')")
            w(f"v = np.array(vn + {1 - g!r} * dt * an if predict else vn)   # (v*: the damping)")
            if bcs:
                w(f"D = EXPLICIT_{name}[{u!r}][3]")
                w(f"v[D] = np.asarray({md}.variable('{u}__v'))[D]")
            w(f"{md}.set_variable('{u}__v', v)")
    w(f"{md}.assembly('build_all')")
    w(f"r = np.array({md}.rhs())")
    w(f"if LINEAR[{c.md['name']!r}]:      # (a linear model: its right-hand side F, not its residual)")
    w(f"    r = r - {md}.tangent_matrix().mult({md}.from_variables())")
    w("")
    w("def mass_system(i0, n, I, D, solve_M, M_ID, aD):")
    w("    a = np.array(aD)")
    w("    rI = r[i0:i0 + n][I]")
    w("    if M_ID is not None and len(D):")
    w("        rI = rI - np.asarray(M_ID.mult(a[D]))")
    w("    a[I] = solve_M(rI)")
    w("    return a")
    w("")
    for u, (sc, (am, b, g), lumped, mass, bcs) in info.items():
        w(f"i0, n, I, D, solve_M, M_ID = EXPLICIT_{name}[{u!r}]")
        acc_now = f"bc_{name}_{ident(u)}(t, dt)[2]" if bcs else "np.zeros(n)"
        if sc["type"] == "explicit_newmark":
            w(f"a = mass_system(i0, n, I, D, solve_M, M_ID, {acc_now})")
            w(f"{md}.set_variable('{u}__a', a)")
            w("if predict:              # (on the Dirichlet dofs: the derivative of the imposed value)")
            w(f"    v = np.array({md}.variable('{u}__v'))")
            w(f"    v[I] += {g!r} * dt * a[I]")
            w(f"    {md}.set_variable('{u}__v', v)")
        else:
            w(f"{md}.set_variable('{u}__mem', mass_system(i0, n, I, D, solve_M, M_ID, {acc_now}))")
            w("if not predict:          # (the initial acceleration: M a_0 = -G_0)")
            w(f"    {md}.set_variable('{u}__a', mass_system(i0, n, I, D, solve_M, M_ID, {acc_now}))")
    w.indent = ""


def _is_zero(pieces) -> bool:
    """A property zero everywhere (its values all the number 0)."""
    def zero(v):
        if isinstance(v, list):
            return all(zero(x) for x in v)
        try:
            return not isinstance(v, bool) and float(v) == 0.0
        except (TypeError, ValueError):
            return False
    return bool(pieces) and all("material" not in pc and zero(pc.get("value")) for pc in pieces)


def declare_model(c: Ctx, w: ScriptWriter) -> None:
    """The gf.Model of a model, its variables, data and macros (no term): also
    the context in which gwfl.py checks the expressions."""
    md = c.py
    w(f"{md} = gf.Model('real')")
    for v in c.md["variables"]:
        if v.get("selection"):
            w(f"{md}.add_filtered_fem_variable({v['name']!r}, {c.mfs[v['name']]}, {c.rg(v['selection'])})")
        else:
            w(f"{md}.add_fem_variable({v['name']!r}, {c.mfs[v['name']]})")
    for im in c.md["imports"]:
        w(f"# Imported: {im['variable']} of model {im['model']} (updated before each solve)")
        w(f"{md}.add_initialized_fem_data({im['name']!r}, {c.mfs[im['name']]}, "
          f"np.zeros({c.mfs[im['name']]}.nbdof()))")
    coordinates(w, md, c.dim)
    time_declarations(c, w)
    parameters(c.model, c.md, w, md)
    deg = max([int(v.get("degree", 1)) for v in c.md["variables"]] + [1])
    w("")
    w("# Data defined per selection (fields, then module properties): discontinuous")
    w(f"{c.mf_data} = gf.MeshFem({c.m}, 1)")
    w(f"{c.mf_data}.set_classical_discontinuous_fem({deg})")
    if any(cd["type"] == "normal_dirichlet" for cd in c.md["conditions"]):
        w(f"mf_data_scalar_{ident(c.md['name'])} = gf.MeshFem({c.m}, 1)")
        w(f"mf_data_scalar_{ident(c.md['name'])}.set_classical_fem({deg})")
    _fields(c, w)
    _properties(c, w)
    for i, mod in enumerate(c.md["modules"], start=1):
        w("")
        w(f"# Module {i}: {M.MODULE_SPECS[mod['type']]['label']}")
        MODULES[mod["type"]][0](c, mod, i, w)


def setup_model(c: Ctx, w: ScriptWriter) -> None:
    for i, mod in enumerate(c.md["modules"], start=1):
        w("")
        MODULES[mod["type"]][1](c, mod, i, w)
        inertia = M.spec_of(mod).get("inertia")
        if inertia and c.transient and M.in_time(c.model, c.md, mod):
            u = mod[inertia["variable"]]
            expr = inertia["expr"].replace("{u}", u)
            w(f"# Time derivative: {expr}")
            w(f"LINEAR[{c.md['name']!r}] &= add_term({c.py}, {c.mim}, {expr!r})")
            for prop, dexpr in inertia.get("damping", {}).items():
                if not _is_zero(mod.get("properties", {}).get(prop)):
                    dexpr = dexpr.replace("{u}", u)
                    w(f"# Damping ({prop}): {dexpr}")
                    w(f"LINEAR[{c.md['name']!r}] &= add_term({c.py}, {c.mim}, {dexpr!r})")
    if c.md["terms"]:
        w("")
        _terms(c, w)
    _conditions(c, w)


HELPERS = '''
def on_region(mf, rg):
    """Degrees of freedom of mf on a region (-1: all)."""
    return np.arange(mf.nbdof()) if rg < 0 else mf.basic_dof_on_region(rg)


def add_term(md, mim, expr, region=-1, linear='auto'):
    """A term of the weak formulation (the terms sum to zero); 'auto': linear
    if GetFEM accepts it as such. Returns whether it is linear."""
    if linear == 'auto':
        try:
            md.add_linear_term(mim, expr, region)
            return True
        except RuntimeError:
            linear = False
    if linear:
        md.add_linear_term(mim, expr, region)
    else:
        md.add_nonlinear_term(mim, expr, region)
    return bool(linear)


SOLVED = set()


def mass_solver(Muu, I, D, lumped, name):
    """M a = r on the free dofs I of an explicit variable (D: its Dirichlet
    dofs): lumped (the sums of the rows of M), or factorized once (SciPy's
    SuperLU if available, GetFEM's otherwise); and M_ID (None if lumped)."""
    if lumped:
        d = np.asarray(Muu.mult(np.ones(Muu.size()[1])))[I]
        if np.any(d <= 0):
            raise RuntimeError(f'the lumped mass of {name} is not positive (P2 elements on simplices?): '
                               'take the consistent mass')
        return (lambda r: r / d), None
    MII = gf.Spmat('copy', Muu, list(I), list(I))
    MID = gf.Spmat('copy', Muu, list(I), list(D)) if len(D) else None
    try:
        import scipy.sparse
        import scipy.sparse.linalg
        jc, ir = MII.csc_ind()
        lu = scipy.sparse.linalg.splu(scipy.sparse.csc_matrix((MII.csc_val(), ir, jc), shape=MII.size()))
        return lu.solve, MID
    except ImportError:
        return (lambda r: np.asarray(gf.linsolve_superlu(MII, r)[0]).ravel()), MID


def power_iteration(op, n, iterations=100):
    """The largest eigenvalue of op (M^-1 K: real, positive), by power iterations."""
    if n == 0:
        return 0.0
    x, lam = np.random.default_rng(0).standard_normal(n), 0.0
    for _ in range(iterations):
        y = op(x)
        lam = np.linalg.norm(y) / np.linalg.norm(x)
        if lam == 0.0:
            break
        x = y / np.linalg.norm(y)
    return float(lam)


def scipy_csc(A):
    """A GetFEM sparse matrix as a SciPy one (CSC)."""
    import scipy.sparse
    jc, ir = A.csc_ind()
    return scipy.sparse.csc_matrix((A.csc_val(), ir, jc), shape=A.size())


def eigen_modes(K, Mm, free, k, sigma, native=None):
    """The k eigenvalues of K x = lambda M x nearest to sigma, on the free
    dofs (K, M: SciPy matrices), sorted, their vectors (M-normalized when M
    is positive on them), the eigensolver used: ARPACK in shift-invert mode,
    a factorization of K - sigma M; GetFEM's gf.eigsh if it has one (that of
    getfem-rs: ARPACK-ng and MUMPS; native: K and M as GetFEM matrices) for
    a symmetric problem, SciPy's otherwise (eigsh; eigs if nonsymmetric,
    the real parts); a small system: dense (SciPy's LAPACK)."""
    import scipy.linalg
    import scipy.sparse.linalg as sla
    K, Mm = K[free][:, free].tocsc(), Mm[free][:, free].tocsc()
    n = len(free)
    symmetric = abs(K - K.T).max() <= 1e-10 * max(abs(K).max(), 1e-300)
    if symmetric and native is not None and hasattr(gf, 'eigsh') and n > max(4 * k, 200):
        f = [int(i) for i in free]
        vals, modes = gf.eigsh(gf.Spmat('copy', native[0], f, f), gf.Spmat('copy', native[1], f, f),
                               nev=k, sigma=sigma)
        used = 'GetFEM eigsh (ARPACK)'
        vals, vecs = np.asarray(vals), np.asarray(modes).T
    elif n <= max(4 * k, 200):
        used = 'dense (LAPACK)'
        A, B = K.toarray(), Mm.toarray()
        vals, vecs = scipy.linalg.eig(A, B)
        ok = np.isfinite(vals) & (np.abs(vals) < 1e15 * max(np.abs(np.diag(A)).max(), 1e-300)
                                  / max(np.abs(np.diag(B)).max(), 1e-300))
        vals, vecs = vals[ok], vecs[:, ok]
        keep = np.argsort(np.abs(vals - sigma))[:k]
        vals, vecs = vals[keep], vecs[:, keep]
    else:
        used = 'SciPy ' + ('eigsh' if symmetric else 'eigs') + ' (ARPACK)'
        lu = sla.splu((K - sigma * Mm).tocsc())
        op = sla.LinearOperator((n, n), matvec=lu.solve, dtype=float)
        if symmetric:
            vals, vecs = sla.eigsh(K, k, Mm, sigma=sigma, OPinv=op, which='LM')
        else:
            vals, vecs = sla.eigs(K, k, Mm, sigma=sigma, OPinv=op, which='LM')
    if np.abs(np.imag(vals)).max(initial=0.0) > 1e-8 * max(np.abs(vals).max(initial=0.0), 1e-300):
        raise RuntimeError('complex eigenvalues (a nonsymmetric problem, a convection?): its modes are '
                           'not computed here')
    vals, vecs = np.real(vals), np.real(vecs)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    for j in range(vecs.shape[1]):
        m = float(vecs[:, j] @ (Mm @ vecs[:, j]))
        if m > 0:
            vecs[:, j] /= np.sqrt(m)
    return vals, vecs, used


def time_derivatives(md, expr, mf, t, dt):
    """An expression of the time t interpolated on mf, its first and second
    derivatives in time (central differences, h = dt/10)."""
    h, g = 0.1 * dt, []
    for s in (t - h, t, t + h):
        md.set_time(s)
        g.append(np.asarray(md.interpolation(expr, mf)))
    md.set_time(t)
    return g[1], (g[2] - g[0]) / (2 * h), (g[2] - 2 * g[1] + g[0]) / h**2


def first_solve(name):
    """Whether a model is solved for the first time."""
    first = name not in SOLVED
    SOLVED.add(name)
    return first


def write_pvd(name, series):
    """A ParaView collection of the VTU files of a series (their times)."""
    with open(name, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0"?>\\n<VTKFile type="Collection" version="0.1">\\n<Collection>\\n')
        for t, vtu in zip(series['t'], series['vtu']):
            f.write(f'  <DataSet timestep="{t!r}" part="0" file="{vtu}"/>\\n')
        f.write('</Collection>\\n</VTKFile>\\n')


def solve(md, R, max_iter=50, max_res=1e-10, check_linear=True, lsolver='auto'):
    """Solve a model and check the result (check_linear: the linear system);
    R: its results entries; lsolver: the solver of the linear systems."""
    t = time.perf_counter()
    options = ['max_iter', max_iter, 'max_res', max_res]
    if lsolver != 'auto':
        if lsolver == 'mumps' and not gf.util_mumps_linked():
            R.setdefault('warnings', []).append('MUMPS is not available in this GetFEM: SuperLU used')
            lsolver = 'superlu'
        options += ['lsolver', lsolver]
        R['linear_solver'] = lsolver
        if check_linear and lsolver.startswith(('cg/', 'gmres/')):
            # a linear model: max_iter bounds the iterations of the linear solver itself
            options[1] = max(max_iter, 10000)
    info = md.solve(*options)
    R['nb_dofs'] = int(md.nbdof())
    R['iterations'] = int(info[0]) if isinstance(info, tuple) else None
    converged = bool(info[1]) if isinstance(info, tuple) and len(info) > 1 else True
    if check_linear:
        # Check that the system was actually solved: on a singular system some
        # GetFEM versions only emit a warning and leave a zero solution.
        md.assembly('build_all')
        K, F, U = md.tangent_matrix(), md.rhs(), md.from_variables()
        residual = np.linalg.norm(K.mult(U) - F) / max(np.linalg.norm(F), 1e-300)
        R['relative_residual'] = float(residual)
        if not residual < 1e-6:
            if lsolver.startswith(('cg/', 'gmres/')):
                raise RuntimeError(f'the iterative linear solver {lsolver} did not converge (relative '
                                   f'residual {residual:.1e}): choose a direct one or another '
                                   'preconditioner (CG: a symmetric positive definite system only)')
            raise RuntimeError(f'solve failed (relative residual {residual:.1e}): the system is '
                               'probably singular, check the boundary conditions')
    elif not converged:
        raise RuntimeError('Newton iterations did not converge')
    if check_linear and not converged:
        R.setdefault('warnings', []).append(
            'the linear system is singular: the solution is not unique '
            '(for instance a rigid body motion is not prevented)')
    R['time_s'] = time.perf_counter() - t
'''


def _solver_code(model: dict, ctxs: list[Ctx], w: ScriptWriter, previous: bool = False) -> None:
    """solve_<model>() for each model, and the main solver."""
    by = {c.md["name"]: c for c in ctxs}
    for c in ctxs:
        s = c.md["solver"]
        exp = list(c.explicit) if c.transient else []
        if exp:
            _explicit_code(c, w)
        w("")
        if c.partial:
            names = [v["name"] for v in c.md["variables"]]
            w(f"def solve_{ident(c.md['name'])}(only=None):")
            w(f'    """only: the variables solved (a partial step), the others of the model given."""')
        else:
            w(f"def solve_{ident(c.md['name'])}():")
        w.indent = "    "
        for im in c.md["imports"]:
            src = by[im["model"]]
            w(f"{c.py}.set_variable({im['name']!r}, {src.py}.variable({im['variable']!r}))")
        if c.partial:
            w(f"given = [v for v in {names!r} if only is not None and v not in only]")
            w(f"given += [mult for v in given for mult in MULTIPLIERS[{c.md['name']!r}].get(v, [])]")
            w("for v in given:")
            w(f"    {c.py}.disable_variable(v)")
        args = ""
        if s.get("max_iter"):
            args += f", max_iter={int(s['max_iter'])}"
        if s.get("max_res"):
            args += f", max_res={float(s['max_res'])!r}"
        if s.get("linear_solver", "auto") != "auto":
            args += f", lsolver={s['linear_solver']!r}"
        linear = "True" if s["type"] == "linear" else "False" if s["type"] == "newton" else f"LINEAR[{c.md['name']!r}]"
        if c.transient:                  # (the linear system checked at the first time step only)
            linear = f"{linear} and first_solve({c.md['name']!r})"
        if exp:                          # the explicit variables: their step, then given to the others
            if not c.partial:
                w("given = []")
            n = ident(c.md["name"])
            if c.partial:
                w(f"step = only is None or any(v in only for v in {exp!r})")
                w("if step:")
                w(f"    explicit_predict_{n}()   # (known before the step: given to the implicit variables)")
            else:
                w(f"explicit_predict_{n}()   # (known before the step: given to the implicit variables)")
            w(f"given += [v for v in {exp!r} if v not in given]")
            w("for v in given:")
            w(f"    {c.py}.disable_variable(v)")
            rest = [v["name"] for v in c.md["variables"] if v["name"] not in exp]
            w("try:")
            w(f"    if [v for v in {rest!r} if v not in given]:      # (the other variables, implicit)")
            w(f"        solve({c.py}, results['models'][{c.md['name']!r}]{args}, check_linear={linear})")
            w("finally:")
            w("    for v in given:")
            w(f"        {c.py}.enable_variable(v)")
            if c.partial:
                w("if step:")
                w(f"    explicit_solve_{n}()     # (their acceleration, the others at the new time)")
            else:
                w(f"explicit_solve_{n}()     # (their acceleration, the others at the new time)")
        elif c.partial:
            w("try:")
            w(f"    solve({c.py}, results['models'][{c.md['name']!r}]{args}, check_linear={linear})")
            w("finally:")
            w("    for v in given:")
            w(f"        {c.py}.enable_variable(v)")
        else:
            w(f"solve({c.py}, results['models'][{c.md['name']!r}]{args}, check_linear={linear})")
        w.indent = ""
    solver = M.main_solver(model)
    w("")
    w(f"# Solver {solver['name']!r}: {solver['type']} of {', '.join(map(str, solver['steps']))}")
    if solver["type"] == "transient":
        _transient_code(model, solver, ctxs, w, previous)
        return
    if solver["type"] == "modal":
        _modal_code(model, solver, by[M.step_models(solver)[0]], w)
        return
    _coupled_code(solver, solver["type"], by, w)


def _modal_code(model: dict, solver: dict, c: Ctx, w: ScriptWriter) -> None:
    """The eigenmodes of a model: K its tangent matrix at the current
    values, M the assembly of the time derivatives of its modules (their
    inertia term, with u for its derivative); the dofs of the Dirichlet
    conditions by simplification removed (those recorded, and the rows of
    K reduced to their diagonal); the modes nearest to the shift
    (eigen_modes). Each mode written to a VTU file (post-processed as a
    solution: its fields), scaled to a largest value 1 of the variables with
    a time derivative; the results as a series of the mode numbers; the
    values of the variables restored."""
    md, name = c.py, c.md["name"]
    order = M.modal_order(c.md)
    vals, _ = P.values(model)
    shift = P.scalar(solver.get("shift", 0.0), vals)
    sigma = (2 * math.pi * shift) ** 2 if order == 2 else shift
    k = int(solver.get("modes", 6))
    masses = []
    timed = M.modal_variables(c.md)          # (those of a lower order: quasi-static, no mass)
    for mod in c.md["modules"]:
        inertia = M.spec_of(mod).get("inertia")
        u = M.inertia_variable(mod)
        if inertia and u in timed:
            e = inertia["expr"].replace("{u}", u).replace(f"{u}__inertia", u)
            masses.append(re.sub(rf"\bDot_{re.escape(u)}\b", u, e))
    stem = Path(c.md["vtu"]).stem if c.md.get("vtu") else None
    what = "omega^2 (rad/s)^2" if order == 2 else "decay rate (1/s)"
    w(f"# The eigenmodes of {name!r}: K phi = lambda M phi, lambda = {what}, the {k} nearest to "
      f"{sigma:g}")
    w("t_modal = time.perf_counter()")
    w(f"R = results['models'][{name!r}]")
    w(f"kept = {md}.from_variables()          # (the values, restored after)")
    w(f"{md}.assembly('build_matrix')")
    w(f"Kg = {md}.tangent_matrix()")
    w(f"Mg = gf.asm_generic({c.mim}, 2, {' + '.join(masses)!r}, -1, {md})")
    w("K, Mm = scipy_csc(Kg), scipy_csc(Mg)")
    w("N = K.shape[0]")
    w("D = [np.zeros(0, dtype=int)]       # the dofs of the Dirichlet conditions by simplification")
    for u, bcs in c.dirichlet.items():
        mf = c.mfs[u]
        for rg, _d in bcs:
            w(f"D.append({md}.interval_of_variable({u!r})[0] + np.asarray({mf}.basic_dof_on_region({rg}), "
              "dtype=int))")
    w("Kr = K.tocsr()                     # (and the rows of K reduced to their diagonal)")
    w("nnz = np.diff(Kr.indptr)")
    w("single = np.where(nnz == 1)[0]")
    w("D.append(single[Kr.indices[Kr.indptr[single]] == single])")
    w("free = np.setdiff1d(np.arange(N), np.concatenate(D))")
    w(f"sigma = {sigma!r}")
    w("if sigma == 0.0:                    # (a little below 0: a free body, its rigid modes at 0)")
    w("    dK, dM = np.abs(K.diagonal()[free]), np.abs(Mm.diagonal()[free])")
    w("    sigma = -1e-8 * dK.max(initial=1.0) / max(dM.max(initial=1.0), 1e-300)")
    w(f"k = min({k}, len(free) - 2)")
    w("if k < 1:")
    w(f"    raise RuntimeError('modal analysis of {name}: too few free degrees of freedom')")
    w("lam, phi, R['eigensolver'] = eigen_modes(K, Mm, free, k, sigma, (Kg, Mg))")
    w("R['nb_dofs'] = int(N)")
    w("R['eigenvalues'] = [float(x) for x in lam]")
    if order == 2:
        w("R['frequencies'] = [float(np.sqrt(max(x, 0.0)) / (2 * np.pi)) for x in lam]   # (Hz)")
        series = "frequency"
        label = "f'mode {j + 1}: f = {np.sqrt(max(lam[j], 0.0)) / (2 * np.pi):.6g} Hz'"
    else:
        w("R['time_constants'] = [float(1 / x) if x > 0 else None for x in lam]   # (s)")
        series = "rate"
        label = "f'mode {j + 1}: rate {lam[j]:.6g} 1/s'"
    # the effective masses: the vector variables of the dimension of the mesh (displacements)
    rigid = [v["name"] for v in c.md["variables"] if v["name"] in timed and M.variable_kind(v) == M.VECTOR]
    if order == 2 and rigid:
        w("# The effective mass of each mode along each axis, (phi^T M r)^2 (phi M-normalized, r the unit")
        w("# translation of the displacements, on the free dofs); the mass moved, r^T M r")
        w("R['effective_mass'], R['mass'] = {}, {}")
        for d, ax in enumerate("xyz"[:c.dim]):
            vec = "[" + ", ".join("1" if j == d else "0" for j in range(c.dim)) + "]"
            w("r = np.zeros(N)")
            for u in rigid:
                w(f"i0, n = {md}.interval_of_variable({u!r})")
                w(f"r[i0:i0 + n] = {md}.interpolation({vec!r}, {c.mfs[u]})")
            w("r = r[free]")
            w("Mr = Mm[free][:, free] @ r")
            w(f"R['effective_mass'][{ax!r}] = [float((phi[:, j] @ Mr) ** 2) for j in range(len(lam))]")
            w(f"R['mass'][{ax!r}] = float(r @ Mr)")
    w(f"S = {{'t': [], 'vtu': [], 'labels': [], {series!r}: []}}")
    w("for j in range(len(lam)):")
    w.indent = "    "
    w("x = np.zeros(N)")
    w("x[free] = phi[:, j]")
    w(f"{md}.to_variables(x)")
    w(f"big = max(np.abs({md}.variable(u)).max(initial=0.0) for u in {timed!r})")
    w("i = max(range(N), key=lambda i: abs(x[i]))")
    w("x *= (1.0 if x[i] >= 0 else -1.0) / (big if big > 0 else 1.0)   # (largest value 1, positive)")
    w(f"{md}.to_variables(x)")
    w(f"results['models'][{name!r}] = {{}}")
    vtu = f"f'{stem}_mode{{j + 1:03d}}.vtu'" if stem else "None"
    w(f"post_{ident(name)}({vtu})")
    w(f"Rj = results['models'][{name!r}]")
    w("S['t'].append(j + 1)")
    if stem:
        w("S['vtu'].append(Rj['vtu'])")
    w(f"S['labels'].append({label})")
    if order == 2:
        w(f"S[{series!r}].append(R['frequencies'][j])")
    else:
        w(f"S[{series!r}].append(float(lam[j]))")
    w("for key, value in list(Rj.items()) + [(q, v) for q, v in Rj.get('quantities', {}).items()]:")
    w("    if isinstance(value, float) and key not in ('time_s', 'relative_residual'):")
    w("        S.setdefault(key, []).append(value)")
    w.indent = ""
    w(f"results['models'][{name!r}] = R")
    w("R['series'] = S")
    w("if len(lam):")
    w("    R.update({key: v for key, v in Rj.items() if isinstance(v, str) and key != 'vtu'})   # (warp)")
    w(f"R['mesh'] = {{'name': {c.mesh!r}, 'nb_elements': int({c.m}.nbcvs())}}")
    if stem:
        w("R['vtu'] = S['vtu'][0] if S['vtu'] else None")
    w(f"{md}.to_variables(kept)")
    w("R['time_s'] = time.perf_counter() - t_modal")
    w(f"results['solver'] = {{'name': {solver['name']!r}, 'type': 'modal', 'modes': len(lam), "
      "'shift': float(sigma)}")


def _coupled_code(solver: dict, coupling: str, by: dict, w: ScriptWriter) -> None:
    """The models of a solver solved in sequence, or by staggered iterations
    (results['solver'] set; inside the time loop: its indentation)."""
    parsed = [M.parse_step(st) for st in solver["steps"]]
    calls = [f"solve_{ident(n)}({'' if v is None else repr(v)})" for n, v in parsed]
    if coupling == "sequence":
        for call in calls:
            w(call)
        w(f"results['solver'] = {{'name': {solver['name']!r}, 'type': {solver['type']!r}}}")
        return
    names = M.step_models(solver)
    last = {n: max(j for j, (m, _) in enumerate(parsed) if m == n) for n in names}   # its last step
    outer = w.indent
    w(f"history = []")
    w(f"for iteration in range(1, {int(solver['max_iter'])} + 1):")
    w.indent = outer + "    "
    w("before = {" + ", ".join(f"{n!r}: {by[n].py}.from_variables().copy()" for n in names) + "}")
    for j, ((n, _), call) in enumerate(zip(parsed, calls)):
        w(call)
        if float(solver["relaxation"]) != 1.0 and last[n] == j:
            r = float(solver["relaxation"])
            w(f"if iteration > 1:  # relaxation")
            w(f"    {by[n].py}.to_variables({r!r} * {by[n].py}.from_variables() + {1 - r!r} * before[{n!r}])")
    w("change = max(np.linalg.norm(" + "{0}.from_variables() - before[n]) / max(np.linalg.norm({0}.from_variables()), 1e-300)"
      .format("MODELS[n]") + f" for n in {names!r})")
    w("history.append(float(change))")
    w(f"if change < {float(solver['tolerance'])!r}:")
    w("    break")
    w.indent = outer
    w(f"results['solver'] = {{'name': {solver['name']!r}, 'type': 'staggered', 'iterations': iteration, "
      f"'history': history, 'converged': bool(change < {float(solver['tolerance'])!r})}}")
    w("if not results['solver']['converged']:")
    w(f"    results.setdefault('warnings', []).append('staggered solver {solver['name']}: not converged "
      f"after {int(solver['max_iter'])} iterations')")


def _transient_code(model: dict, solver: dict, ctxs: list[Ctx], w: ScriptWriter, previous: bool = False) -> None:
    """The time loop: the initial values, then at each time step the time
    and the data depending on it set, the models solved (in sequence or by
    staggered iterations), the previous values shifted; the results saved
    every save.every steps (VTU files and a .pvd collection, the scalar
    results as series)."""
    t0, t1, dt = M.time_values(model, solver)
    n = max(1, int(round((t1 - t0) / dt)))
    every = int(solver["save"]["every"])
    cs = [c for c in ctxs if c.md["name"] in M.step_models(solver)]
    by = {c.md["name"]: c for c in ctxs}
    w(f"T_START, T_END, N_STEPS = {t0!r}, {t1!r}, {n}   # time step {(t1 - t0) / n!r}")
    w("")
    w("T_NOW = [T_START]   # the current time")
    w("")
    w("def set_time(t, dt):")
    w('    """The time and the time step of the models, the data depending on t."""')
    w("    T_NOW[0] = t")
    for c in cs:
        w(f"    {c.py}.set_time(t)")
        w(f"    {c.py}.set_time_step(dt)")
        for name, mf, expr in c.timed:
            w(f"    {c.py}.set_variable({name!r}, {c.py}.interpolation({expr!r}, {mf}))")
    w("")
    w("def shift(k):")
    w('    """After the time step k: the previous values (Previous_u, Previous_Dot_u, u__prev2),')
    w('    the coefficients of the schemes after their first step."""')
    lines = 0
    for c in cs:
        for u, sc in c.schemes.items():
            if M.is_explicit(sc):                        # (the step done: u__v, u__a at the new time)
                w(f"    {c.py}.set_variable('{u}__rate', {c.py}.variable('{u}__v'))")
                w(f"    {c.py}.set_variable('{u}__rate2', {c.py}.variable('{u}__a'))")
                w(f"    {c.py}.set_variable('{u}__prev', {c.py}.variable({u!r}))")
                lines += 1
                continue
            if M.SCHEME_ORDER[sc["type"]] == 2:          # (all from the values before the shift)
                mem = M.second_order_coefficients(sc)["inertia"][3] != 0.0
                names = [("rate", f"Dot_{u}"), ("rate2", f"Dot2_{u}")] + ([("mem", f"{u}__inertia")] if mem else [])
                w(f"    new = [{', '.join(f'{c.py}.interpolation({e!r}, {c.mfs[u]})' for _, e in names)}]")
                for j, (n, _) in enumerate(names):
                    w(f"    {c.py}.set_variable('{u}__{n}', new[{j}])")
                w(f"    {c.py}.set_variable('{u}__prev', {c.py}.variable({u!r}))")
                lines += 1
                continue
            w(f"    {c.py}.set_variable('{u}__rate', {c.py}.interpolation('Dot_{u}', {c.mfs[u]}))")
            if sc.get("type") == "bdf2":
                w(f"    {c.py}.set_variable('{u}__prev2', {c.py}.variable('{u}__prev'))")
            w(f"    {c.py}.set_variable('{u}__prev', {c.py}.variable({u!r}))")
            after = list(M.scheme_coefficients(sc))
            if after != list(M.scheme_coefficients(sc, first=True)):
                w("    if k == 1:")
                w(f"        {c.py}.set_variable('{u}__scheme', {after!r})  # {M.scheme_label(sc)} from now on")
            lines += 1
    if not lines:
        w("    pass")
    w("")
    w("SERIES = {" + ", ".join(f"{c.md['name']!r}: {{'t': [], 'vtu': []}}" for c in cs) + "}")
    post = []                    # the coefficients giving the derivatives at the saved time (after shift)
    for c in cs:
        for u, sc in c.schemes.items():
            if M.is_explicit(sc):                        # (u__v, u__a: those of the saved time)
                continue
            if M.SCHEME_ORDER[sc["type"]] == 2:
                post += [(c.py, f"{u}__acc", [0.0, 0.0, 1.0]), (c.py, f"{u}__vel", [0.0, 1.0, 0.0]),
                         (c.py, f"{u}__inertia_c", [0.0, 0.0, 1.0, 0.0])]
            else:
                post.append((c.py, f"{u}__scheme", [0.0, 0.0, 0.0, 1.0]))
    if post:
        w("# at a saved time (after shift), Dot_u (Dot2_u) are the derivatives kept: Previous_Dot_u...")
        w("AT_SAVE = [" + ", ".join(f"({py}, {n!r}, {v!r})" for py, n, v in post) + "]")
    mems = [(c.py, u) for c in cs for u, sc in c.explicit.items() if sc["type"] != "explicit_newmark"]
    if mems:     # (the acceleration of the scheme: at first order; its memory, M^-1 G: at second order)
        w("# at a saved time, Dot2_u of an explicit generalized-alpha: its memory (the acceleration at t)")
        w("MEMORIES = [" + ", ".join(f"({py}, {u!r})" for py, u in mems) + "]")
    w("def save(k, t):")
    w('    """The results at a saved time: the fields (a VTU file each), the scalars."""')
    if post:
        w("    kept = [(md, name, md.variable(name)) for md, name, _ in AT_SAVE]")
        w("    for md, name, value in AT_SAVE:")
        w("        md.set_variable(name, value)")
    if mems:
        w("    algorithmic = [(md, u, md.variable(u + '__a')) for md, u in MEMORIES]")
        w("    for md, u in MEMORIES:")
        w("        md.set_variable(u + '__a', md.variable(u + '__mem'))")
    for c in cs:
        name = c.md["name"]
        stem = Path(c.md["vtu"]).stem if c.md.get("vtu") else None
        vtu = f"f'{stem}_{{k:05d}}.vtu'" if stem else "None"
        w(f"    post_{ident(name)}({vtu})")
        w(f"    R, S = results['models'][{name!r}], SERIES[{name!r}]")
        w("    S['t'].append(t)")
        if stem:
            w("    S['vtu'].append(R['vtu'])")
        w("    for key, value in list(R.items()) + [(q, v) for q, v in R.get('quantities', {}).items()]:")
        w("        if isinstance(value, float) and key not in ('time_s', 'relative_residual'):")
        w("            S.setdefault(key, []).append(value)")
    if post:
        w("    for md, name, value in kept:")
        w("        md.set_variable(name, value)")
    if mems:
        w("    for md, u, value in algorithmic:")
        w("        md.set_variable(u + '__a', value)")
    w("")
    w("# The initial values" + (": the state left by the previous solver" if previous else ""))
    for c in cs:
        for v in c.md["variables"]:
            u, mf = v["name"], c.mfs[v["name"]]
            kind = M.variable_kind(v)
            zero = [0.0] * c.dim if kind == M.VECTOR else 0.0
            init = c.value(v.get("initial", zero), kind)
            if previous:
                w(f"{c.py}.set_variable({u!r}, state[{c.md['name']!r}][{u!r}])")
            else:
                w(f"{c.py}.set_variable({u!r}, {c.py}.interpolation({init!r}, {mf}))")
            if u in c.schemes:
                w(f"{c.py}.set_variable('{u}__prev', {c.py}.variable({u!r}))")
                if c.schemes[u].get("type") == "bdf2":
                    w(f"{c.py}.set_variable('{u}__prev2', {c.py}.variable({u!r}))")
                if M.SCHEME_ORDER[c.schemes[u]["type"]] == 2 and previous:   # its velocity, if in time before
                    vel = c.value(v.get("initial_velocity", zero), kind)
                    w(f"{c.py}.set_variable('{u}__rate', state[{c.md['name']!r}].get('{u}__rate', "
                      f"{c.py}.interpolation({vel!r}, {mf})))")
                elif M.SCHEME_ORDER[c.schemes[u]["type"]] == 2:
                    vel = c.value(v.get("initial_velocity", zero), kind)
                    w(f"{c.py}.set_variable('{u}__rate', {c.py}.interpolation({vel!r}, {mf}))  # initial velocity")
    w("set_time(T_START, (T_END - T_START) / N_STEPS)")
    for c in cs:
        _initial_accelerations(c, by, w)
    for c in cs:
        tv = [u for u, sc in c.schemes.items() if M.SCHEME_ORDER[sc["type"]] == 1]
        if any(M.scheme_needs_rate(c.schemes[u]) for u in tv):
            w(f"# The initial time derivative of {', '.join(tv)} (needed by the first step of "
              + ", ".join(sorted({M.scheme_label(c.schemes[u]) for u in tv if M.scheme_needs_rate(c.schemes[u])}))
              + "):")
            w("# an implicit Euler step of a tiny time step from the initial values")
            eps = "1e-6 * (T_END - T_START) / N_STEPS"
            w(f"set_time(T_START + {eps}, {eps})")
            for u in tv:
                w(f"{c.py}.set_variable('{u}__scheme', [1.0, -1.0, 0.0, 0.0])")
            w(f"solve_{ident(c.md['name'])}()")
            for u in tv:
                w(f"{c.py}.set_variable('{u}__rate', ({c.py}.variable({u!r}) - {c.py}.variable('{u}__prev'))"
                  f" / ({eps}))")
                w(f"{c.py}.set_variable({u!r}, {c.py}.variable('{u}__prev'))")
                w(f"{c.py}.set_variable('{u}__scheme', {list(M.scheme_coefficients(c.schemes[u], first=True))!r})")
            w("set_time(T_START, (T_END - T_START) / N_STEPS)")
    w("save(0, T_START)")
    w("")
    w("# The time steps")
    w("for k in range(1, N_STEPS + 1):")
    w.indent = "    "
    w("t = T_START + k * (T_END - T_START) / N_STEPS")
    w("set_time(t, (T_END - T_START) / N_STEPS)")
    _coupled_code(solver, solver["coupling"], by, w)
    if solver["coupling"] == "staggered":
        w("if not results['solver']['converged']:")
        w("    results['warnings'][-1] += f' (at t = {t:g})'")
    w("shift(k)")
    w(f"if k % {every} == 0 or k == N_STEPS:")
    w("    save(k, t)")
    w.indent = ""
    w(f"results['solver'] = {{'name': {solver['name']!r}, 'type': 'transient', 'steps': N_STEPS, "
      f"'start': T_START, 'end': T_END}}")
    w("for name, S in SERIES.items():")
    w("    results['models'][name]['series'] = S")
    for c in cs:
        if c.md.get("vtu"):
            pvd = Path(c.md["vtu"]).with_suffix(".pvd").name
            w(f"write_pvd({pvd!r}, SERIES[{c.md['name']!r}])")
            w(f"results['models'][{c.md['name']!r}]['pvd'] = {pvd!r}")


def _initial_accelerations(c: Ctx, by: dict, w: ScriptWriter) -> None:
    """The initial acceleration of the second order variables of a model:
    M a0 = -G(u0, v0, t0) (G: all the terms but the inertia): r = -G
    assembled without inertia at the initial values (the residual; F - K U0
    for a linear model, whose right-hand side is F), then (M/eps^2 + K) dU =
    r (eps = 1e-6 dt): dU = eps^2 a0 + O(eps^4), with no difference of
    values (no cancellation); the memory of HHT and generalized-alpha starts
    from it. Or zero ("initial_acceleration": "zero")."""
    two = {u: sc for u, sc in c.schemes.items() if M.SCHEME_ORDER[sc["type"]] == 2 and not M.is_explicit(sc)}
    if c.explicit:
        name = ident(c.md["name"])
        w(f"# The explicit variables of {c.md['name']!r}: their mass matrices, the stability limit; their initial")
        w("# acceleration: M a0 = -G(u0, v0, t0)")
        for im in c.md["imports"]:
            w(f"{c.py}.set_variable({im['name']!r}, {by[im['model']].py}.variable({im['variable']!r}))")
        w(f"explicit_setup_{name}()")
        w(f"explicit_predict_{name}(predict=False)")
        w(f"explicit_solve_{name}(predict=False)")
        for u, sc in c.explicit.items():
            if sc.get("initial_acceleration", "equation") == "zero":
                w(f"{c.py}.set_variable('{u}__a', 0 * {c.py}.variable('{u}__a'))")
            w(f"{c.py}.set_variable('{u}__rate2', {c.py}.variable('{u}__a'))")
    if not two:
        return
    md = c.py
    mems = [u for u, sc in two.items() if M.second_order_coefficients(sc)["inertia"][3] != 0.0]
    solved = [u for u, sc in two.items() if sc.get("initial_acceleration", "equation") == "equation"]
    if solved:
        w(f"# The initial acceleration of {', '.join(solved)}: M a0 = -G(u0, v0, t0) (all the terms but the")
        w("# inertia, at the initial values): r = -G assembled without inertia (F - K U0 if linear), then")
        w("# (M/eps^2 + K) dU = r: dU = eps^2 a0 + O(eps^4), no difference of values")
        for im in c.md["imports"]:
            w(f"{md}.set_variable({im['name']!r}, {by[im['model']].py}.variable({im['variable']!r}))")
        w("EPS = 1e-6 * (T_END - T_START) / N_STEPS")
        w("set_time(T_START, EPS)")
        for u in solved:                      # no inertia, Dot_u = v0
            w(f"{md}.set_variable('{u}__acc', [0.0, 0.0, 0.0])")
            w(f"{md}.set_variable('{u}__vel', [0.0, 1.0, 0.0])")
            w(f"{md}.set_variable('{u}__inertia_c', [0.0, 0.0, 0.0, 0.0])")
        w(f"{md}.assembly('build_all')")
        w(f"r = {md}.rhs()")
        w(f"if LINEAR[{c.md['name']!r}]:      # (a linear model: its right-hand side F, not its residual)")
        w(f"    r = r - {md}.tangent_matrix().mult({md}.from_variables())")
        for u in solved:                      # the inertia (u - u_n)/eps^2
            w(f"{md}.set_variable('{u}__acc', [1.0, 0.0, 0.0])")
            w(f"{md}.set_variable('{u}__inertia_c', [1.0, 0.0, 0.0, 0.0])")
        w(f"{md}.assembly('build_matrix')")
        w(f"dU = gf.linsolve_superlu({md}.tangent_matrix(), r)[0]")
        for u in solved:
            k = M.second_order_coefficients(two[u])
            w(f"i0, n = {md}.interval_of_variable({u!r})")
            w(f"{md}.set_variable('{u}__rate2', dU[i0:i0 + n] / EPS**2)")
            w(f"{md}.set_variable('{u}__acc', {k['acc']!r})")
            w(f"{md}.set_variable('{u}__vel', {k['vel']!r})")
            w(f"{md}.set_variable('{u}__inertia_c', {k['inertia']!r})")
        w("set_time(T_START, (T_END - T_START) / N_STEPS)")
    for u in mems:
        w(f"{md}.set_variable('{u}__mem', {md}.variable('{u}__rate2'))  # the memory: G_0 = -M a0")


def _post(c: Ctx, w: ScriptWriter) -> None:
    """post_<model>(vtu): its results entries, its fields written to the
    VTU file vtu (None: not written); called after the solve, or at each
    saved time."""
    md, name = c.py, c.md["name"]
    w(f"def post_{ident(name)}(vtu):")
    w.indent = "    "
    w(f"R = results['models'][{name!r}]")
    w(f"R['mesh'] = {{'name': {c.mesh!r}, 'nb_elements': int({c.m}.nbcvs())}}")
    exports = []
    for v in c.md["variables"]:
        exports += [c.mfs[v["name"]], f"{md}.variable({v['name']!r})", repr(v["name"])]
    for i, mod in enumerate(c.md["modules"], start=1):
        exports += MODULES[mod["type"]][2](c, mod, i, w) or []
    if c.md["outputs"]:
        w("R['quantities'] = {}")
        for q in c.md["outputs"]:
            w(f"R['quantities'][{q['name']!r}] = float(gf.asm('generic', {c.mim}, 0, {str(q['expr'])!r}, "
              f"{c.rg(q['selection'])}, {md}))")
    for f in M.model_fields(c.model, c.md):
        kind = M.field_kind(f)
        if kind == M.SCALAR:
            exports += [c.mf_data, f"{md}.interpolation({f['name']!r}, {c.mf_data})", repr(f["name"])]
        elif kind == M.VECTOR:                    # (matrices are not written to the VTU files)
            mfv = f"mf_vdata_{ident(c.md['name'])}"
            w(f"{mfv} = gf.MeshFem({c.m}, {c.m}.dim())")
            w(f"{mfv}.set_classical_discontinuous_fem(0)")
            exports += [mfv, f"{md}.interpolation({f['name']!r}, {mfv})", repr(f["name"])]
    w("if vtu:")
    w(f"    {exports[0]}.export_to_vtu(vtu, " + ", ".join(exports) + ")")
    w("    R['vtu'] = vtu")
    w.indent = ""


def generate(model: dict, meshes: dict[str, tuple[str | Path | None, dict]] | None = None,
             standalone: bool = False) -> str:
    """Solve script for already prepared meshes: meshes = {name: (file, info)}
    (see mesher.ensure_mesh). standalone: the script follows the meshing
    script (runner.export_script), which defines m_<mesh> and RG_<mesh>.
    With several global solvers, each one is a function solver_<name>(state)
    and the script runs them in order, or one (--solver name) after those it
    starts from; each one writes its files in its folder.
    Raises ValueError if the model is invalid."""
    errors = M.solve_errors(model)
    if errors:
        raise ValueError("invalid model:\n  - " + "\n  - ".join(errors))
    model = P.resolved(M.with_defaults(model))
    meshes = meshes or {}
    ctxs = contexts(model)
    several = len(model.get("solvers", [])) > 1
    w = ScriptWriter()
    w("#!/usr/bin/env python3")
    w(f"# GetFEM script generated by getfem_lab on {_dt.datetime.now():%Y-%m-%d %H:%M}")
    w(f"# File: {model.get('name', '')}; models: {', '.join(c.md['name'] for c in ctxs)}")
    w("# This script is standalone: it can be read, modified and run again by hand.")
    if several:
        names = [s["name"] for s in model["solvers"]]
        w(f"# Global solvers: {', '.join(names)}, run in this order; one of them only (after those it")
        w("# starts from): python this_script.py --solver NAME. Each one writes its files in its folder.")
    w("import json")
    if several:
        w("import os")
        w("import sys")
    w("import time")
    w("")
    w("import numpy as np")
    w("import getfem as gf")
    for line in HELPERS.splitlines():
        w(line)
    w("")
    if not several:
        w("t0 = time.perf_counter()")
        w(f"results = {{'name': {model.get('name', '')!r}, 'status': 'error', 'models': {{"
          + ", ".join(f"{c.md['name']!r}: {{}}" for c in ctxs) + "}}")
    _shared(model, meshes, standalone, ctxs, w)
    if not several:
        _body(model, ctxs, w)
        w("with open('results.json', 'w', encoding='utf-8') as f:")
        w("    json.dump(results, f, indent=2, ensure_ascii=False)")
        w("dofs = sum(r.get('nb_dofs', 0) for r in results['models'].values())")
        w("print(f\"solved: {dofs} unknowns, {results['time_s']:.2f} s (results in results.json)\")")
        return w.text()
    for s in model["solvers"]:
        view = M.solver_first(model, s["name"])
        vctxs = contexts(view)
        start = s.get("start", "initial")
        w.section(f"Global solver {s['name']!r} ({s['type']}, from "
                  + ("the state left by the previous solver)" if start == "previous" else "the initial values)"))
        inner = ScriptWriter()
        inner("t0 = time.perf_counter()")
        inner("SOLVED.clear()")
        inner(f"results = {{'name': {model.get('name', '')!r}, 'status': 'error', 'models': {{"
              + ", ".join(f"{c.md['name']!r}: {{}}" for c in vctxs) + "}}")
        _body(view, vctxs, inner, previous=start == "previous")
        inner("state = {}   # the state left: the values of the variables, the velocities of those in time")
        for c in vctxs:
            inner(f"state[{c.md['name']!r}] = {{v: {c.py}.variable(v) for v in "
                  f"{[v['name'] for v in c.md['variables']]!r}}}")
            for u in c.schemes:
                inner(f"state[{c.md['name']!r}]['{u}__rate'] = {c.py}.variable('{u}__rate')")
        inner("return results, state")
        w(f"def solver_{ident(s['name'])}(state=None):")
        w(f'    """The global solver {s["name"]!r}: its models declared, solved, post-processed; returns its')
        w('    results and the state it leaves."""')
        for line in inner.lines:
            w(("    " + line) if line else "")
        w("")
    w.section("Run the global solvers")
    w("SOLVERS = {" + ", ".join(f"{s['name']!r}: (solver_{ident(s['name'])}, {s.get('start', 'initial')!r})"
                                for s in model["solvers"]) + "}")
    w(f"ORDER = {names!r}")
    w("def chain(name):")
    w('    """The solver named, after those it starts from."""')
    w("    i = j = ORDER.index(name)")
    w("    while j > 0 and SOLVERS[ORDER[j]][1] == 'previous':")
    w("        j -= 1")
    w("    return ORDER[j:i + 1]")
    w("run = chain(sys.argv[sys.argv.index('--solver') + 1]) if '--solver' in sys.argv else ORDER")
    w("t_all = time.perf_counter()")
    w("everything, state = {}, None")
    w("for name in run:")
    w("    os.makedirs(name, exist_ok=True)")
    w("    here = os.getcwd()")
    w("    os.chdir(name)")
    w("    try:")
    w("        r, state = SOLVERS[name][0](state if SOLVERS[name][1] == 'previous' else None)")
    w("    finally:")
    w("        os.chdir(here)")
    w("    for mr in r['models'].values():           # (the files: in the folder of the solver)")
    w("        if mr.get('vtu'):")
    w("            mr['vtu'] = f\"{name}/{mr['vtu']}\"")
    w("        if mr.get('series', {}).get('vtu'):")
    w("            mr['series']['vtu'] = [f'{name}/{f}' for f in mr['series']['vtu']]")
    w("    r['solver_name'] = name")
    w("    everything[name] = r")
    w("results = dict(everything[run[-1]])     # the last one run; each one in 'solvers'")
    w("results['solvers'] = everything")
    w("results['run'] = run")
    w("results['time_s'] = time.perf_counter() - t_all")
    w("results['status'] = 'ok' if all(r['status'] == 'ok' for r in everything.values()) else 'error'")
    w("with open('results.json', 'w', encoding='utf-8') as f:")
    w("    json.dump(results, f, indent=2, ensure_ascii=False)")
    w("print(f\"solved: {', '.join(run)}, {results['time_s']:.2f} s (results in results.json)\")")
    return w.text()


def _shared(model: dict, meshes: dict, standalone: bool, ctxs: list[Ctx], w: ScriptWriter) -> None:
    """The meshes, their integration methods, the finite element methods
    (shared by all the global solvers)."""
    w.section("Meshes, prepared by the meshing step" if not standalone else "Meshes, built above")
    integ: dict[str, int] = {}
    for c in ctxs:
        mesh = M.find_mesh(model, c.mesh)
        deg = max(int(v.get("degree", 1)) for v in c.md["variables"])
        integ[c.mesh] = max(integ.get(c.mesh, 0), int(mesh.get("integration", 2 * deg)))
    for name in used_meshes(model):
        mid = ident(name)
        if not standalone:
            path, info = meshes[name]
            w(f"m_{mid} = gf.Mesh('load', {str(Path(path).resolve())!r})")
            w(f"RG_{mid} = {info['regions']!r}")
        w(f"mim_{mid} = gf.MeshIm(m_{mid}, {integ[name]})")

    w.section("Finite element methods (shared by the variables with the same one)")
    done = set()
    for c in ctxs:
        for v in c.md["variables"]:
            key = fem_key(model, v, c.dim)
            if key in done:
                continue
            done.add(key)
            mesh, fem, degree, qdim = key
            py = fem_name(key)
            w(f"{py} = gf.MeshFem(m_{ident(mesh)}, {qdim})")
            w(f"{py}.set_classical_{'discontinuous_' if fem == 'lagrange_discontinuous' else ''}fem({degree})")


def _body(model: dict, ctxs: list[Ctx], w: ScriptWriter, previous: bool = False) -> None:
    """The models (for the main global solver of model), their
    post-processing, the resolution. previous: from the state left by the
    previous solver (state: {model: {variable: values}})."""
    w("LINEAR = {}")
    if any(c.partial for c in ctxs):
        w("MULTIPLIERS = {}   # the multipliers of the variables (disabled with them in a partial step)")
    for c in ctxs:
        w.section(f"Model {c.md['name']!r}")
        declare_model(c, w)
        w("")
        w(f"# Terms and conditions of {c.md['name']!r}")
        w(f"LINEAR[{c.md['name']!r}] = True   # False once a term is not")
        if c.partial:
            w(f"MULTIPLIERS[{c.md['name']!r}] = {{}}")
        setup_model(c, w)
    w("MODELS = {" + ", ".join(f"{c.md['name']!r}: {c.py}" for c in ctxs) + "}")

    w.section("Post-processing (functions)")
    for c in ctxs:
        _post(c, w)
        w("")

    w.section("Solve")
    if previous and not M.main_solver(model).get("type") == "transient":
        w("# from the state left by the previous solver")
        for c in ctxs:
            for v in c.md["variables"]:
                w(f"{c.py}.set_variable({v['name']!r}, state[{c.md['name']!r}][{v['name']!r}])")
    _solver_code(model, ctxs, w, previous)
    w("")
    for c in ctxs:
        if not c.transient and not M.is_modal(model, c.md):
            w(f"post_{ident(c.md['name'])}({c.md.get('vtu')!r})")
    w("results['status'] = 'ok'")
    w("results['time_s'] = time.perf_counter() - t0")
