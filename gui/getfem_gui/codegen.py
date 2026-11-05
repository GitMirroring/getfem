# Copyright (C) 2026 Yves Renard.
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

The script does not depend on getfem_gui: it can be read, modified and run
again by hand. Each module is a set of functions registered in MODULES,
which write its macros and terms in GWFL; its declarative description is
model.MODULE_SPECS (properties, options, conditions) used for validation.
"""

from __future__ import annotations

import datetime as _dt
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
        w(f"{md_py}.add_initialized_data({name!r}, [{values[name]!r}]){comment}")


def _piecewise(c: Ctx, name: str, pieces: list[tuple[str, object, str]], what: str, w: ScriptWriter) -> None:
    """`name` defined by (selection, value, comment) pieces; the last one wins.

    Numbers give finite element data (discontinuous, per element). With an
    expression, `name` is a macro instead, sum of the expressions times the
    indicators of where each one applies: it may then depend on the
    coordinates and on the variables (a conductivity depending on T)."""
    w("")
    py = f"{c.py}_{ident(name)}"
    numeric = all(isinstance(v, (int, float)) for _, v, _ in pieces)
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
            parts.insert(0, gwfl(value) if len(pieces) == 1 and selection == M.ALL else f"{chi}*{gwfl(value)}")
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
        _piecewise(c, f["name"], pieces, "field", w)


def _properties(c: Ctx, w: ScriptWriter) -> None:
    """Each property of the modules becomes discontinuous finite element data,
    defined per selection: plain 'k' can then be used in any expression."""
    for mod in c.md["modules"]:
        for prop in M.MODULE_SPECS[mod["type"]]["properties"]:
            pieces = []
            for pc in mod["properties"].get(prop, []):
                if "material" in pc:
                    mat = M.material(c.model, pc["material"])
                    pieces.append((pc["selection"], mat["properties"][prop],
                                   f"{pc['selection']}: {pc['material']}"))
                else:
                    pieces.append((pc["selection"], pc["value"], pc["selection"]))
            _piecewise(c, prop, pieces, "property", w)


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


def _heat_setup(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> None:
    u, md = mod["variable"], c.py
    w(f"# -div(k grad {u}) = f")
    w(f"LINEAR[{c.md['name']!r}] &= add_term({md}, {c.mim}, 'k*Grad_{u}.Grad_Test_{u}')")
    for src in mod["sources"]:
        w(f"{md}.add_source_term({c.mim}, {gwfl(src['value']) + '*Test_' + u!r}, {c.rg(src['selection'])})")
    for j, bc in enumerate(mod["conditions"], start=1):
        rg = c.rg(bc["selection"])
        w("")
        if bc["type"] == "temperature":
            data = f"{u}_D{i}_{j}"
            mf = c.mfs[u]
            w(f"# Imposed temperature on {bc['selection']}")
            w(f"{md}.add_initialized_fem_data({data!r}, {mf}, {md}.interpolation({gwfl(bc['value'])!r}, {mf}))")
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg}, {data!r})")
        elif bc["type"] == "heat_flux":
            w(f"# Imposed incoming flux on {bc['selection']}: k grad {u}.n = q")
            w(f"{md}.add_source_term({c.mim}, {gwfl(bc['value']) + '*Test_' + u!r}, {rg})")
        elif bc["type"] == "convection":
            h, text = gwfl(bc["h"]), gwfl(bc["T_ext"])
            w(f"# Convection on {bc['selection']}: -k grad {u}.n = h ({u} - T_ext)")
            w(f"{md}.add_linear_term({c.mim}, {h + '*' + u + '*Test_' + u!r}, {rg})")
            w(f"{md}.add_source_term({c.mim}, {h + '*' + text + '*Test_' + u!r}, {rg})")


def _heat_post(c: Ctx, mod: dict, i: int, w: ScriptWriter) -> list[str]:
    u, md = mod["variable"], c.py
    w(f"R['min_{u}'] = float({md}.variable({u!r}).min())")
    w(f"R['max_{u}'] = float({md}.variable({u!r}).max())")
    deg = max(int(c.var(u).get("degree", 1)) - 1, 0)
    w(f"mf_flux = gf.MeshFem({c.m}, {c.m}.dim())")
    w(f"mf_flux.set_classical_discontinuous_fem({deg})")
    w(f"flux = {md}.interpolation('q', mf_flux)")
    return ["mf_flux", "flux", "'heat_flux'"]


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
        w(f"{md}.add_source_term({c.mim}, {gwfl_vec(src['value'], dim) + '.Test_' + u!r}, "
          f"{c.rg(src['selection'])})")
    for j, bc in enumerate(mod["conditions"], start=1):
        rg = c.rg(bc["selection"])
        w("")
        if bc["type"] == "fixed":
            w(f"# Fixed on {bc['selection']}: {u} = 0")
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg})")
        elif bc["type"] == "displacement":
            data = f"{u}_D{i}_{j}"
            mf = c.mfs[u]
            w(f"# Imposed displacement on {bc['selection']}")
            w(f"{md}.add_initialized_fem_data({data!r}, {mf}, "
              f"{md}.interpolation({gwfl_vec(bc['value'], dim)!r}, {mf}))")
            w(f"{md}.add_Dirichlet_condition_with_simplification({u!r}, {rg}, {data!r})")
        elif bc["type"] == "roller":
            w(f"# Roller on {bc['selection']}: {u}.n = 0, free tangential displacement")
            w(f"{md}.add_normal_Dirichlet_condition_with_Nitsche_method({c.mim}, {u!r}, 'sigma*Normal', "
              f"'nitsche_gamma0', {rg}, 1.0)")
        elif bc["type"] == "traction":
            w(f"# Surface traction on {bc['selection']}: sigma.n = g")
            w(f"{md}.add_source_term({c.mim}, {gwfl_vec(bc['value'], dim) + '.Test_' + u!r}, {rg})")
        elif bc["type"] == "pressure":
            w(f"# Pressure on {bc['selection']}: sigma.n = -p n")
            w(f"{md}.add_source_term({c.mim}, {'-' + gwfl(bc['value']) + '*(Normal.Test_' + u + ')'!r}, {rg})")


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


def _nothing(c: Ctx, mod: dict, i: int, w: ScriptWriter):
    return []


MODULES = {
    "heat_conduction": (_heat_declare, _heat_setup, _heat_post),
    "linear_elasticity": (_elasticity_declare, _elasticity_setup, _elasticity_post),
    "thermal_expansion": (_nothing, _nothing, _nothing),   # written by the elasticity module
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
                w(f"{md}.add_initialized_fem_data({data!r}, mf_data_scalar_{ident(c.md['name'])}, "
                  f"{md}.interpolation({gwfl(cd['value'])!r}, mf_data_scalar_{ident(c.md['name'])}))")
            else:
                mf = c.mfs[u]
                w(f"{md}.add_initialized_fem_data({data!r}, {mf}, "
                  f"{md}.interpolation({c.value(cd['value'], kind)!r}, {mf}))")
            deg = int(c.var(u).get("degree", 1))
            name = "add_normal_Dirichlet_condition" if normal else "add_Dirichlet_condition"
            if cd["method"] == "simplification":
                w(f"{md}.{name}_with_simplification({u!r}, {rg}, {data!r})")
            elif cd["method"] == "multiplier":
                w(f"{md}.{name}_with_multipliers({c.mim}, {u!r}, {deg}, {rg}, {data!r})")
            else:
                w(f"{md}.{name}_with_penalization({c.mim}, {u!r}, {float(cd.get('penalty', 1e10))!r}, "
                  f"{rg}, {data!r})")
        elif t == "neumann":
            w(f"# Neumann condition on {cd['selection']}: incoming flux")
            w(f"{md}.add_source_term({c.mim}, {c.value(cd['value'], kind) + test + u!r}, {rg})")
        elif t == "robin":
            h, g = gwfl(cd["h"]), c.value(cd["value"], kind)
            w(f"# Robin condition on {cd['selection']}: flux = h ({u} - g)")
            w(f"{md}.add_linear_term({c.mim}, {h + '*' + u + test + u!r}, {rg})")
            w(f"{md}.add_source_term({c.mim}, {h + '*' + g + test + u!r}, {rg})")


# ---------------------------------------------------------------- models

def contexts(model: dict) -> list[Ctx]:
    out = []
    for md in model["models"]:
        mesh = M.model_mesh(model, md)
        dim = M.geometry_dim(model, M.geometry_of(model, mesh))
        c = Ctx(model, md, int(dim), mesh["name"], f"md_{ident(md['name'])}")
        for v in md["variables"]:
            c.mfs[v["name"]] = fem_name(fem_key(model, v, c.dim))
        for im in md["imports"]:
            src = M.model_by_name(model, im["model"])
            c.mfs[im["name"]] = fem_name(fem_key(model, M.by_name(src["variables"], im["variable"]), c.dim))
        out.append(c)
    return out


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


def solve(md, R, max_iter=50, max_res=1e-10, check_linear=True):
    """Solve a model and check the result (check_linear: the linear system);
    R: its results entries."""
    t = time.perf_counter()
    info = md.solve('max_iter', max_iter, 'max_res', max_res)
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


def _solver_code(model: dict, ctxs: list[Ctx], w: ScriptWriter) -> None:
    """solve_<model>() for each model, and the main solver."""
    by = {c.md["name"]: c for c in ctxs}
    for c in ctxs:
        s = c.md["solver"]
        w("")
        w(f"def solve_{ident(c.md['name'])}():")
        w.indent = "    "
        for im in c.md["imports"]:
            src = by[im["model"]]
            w(f"{c.py}.set_variable({im['name']!r}, {src.py}.variable({im['variable']!r}))")
        args = ""
        if s.get("max_iter"):
            args += f", max_iter={int(s['max_iter'])}"
        if s.get("max_res"):
            args += f", max_res={float(s['max_res'])!r}"
        linear = "True" if s["type"] == "linear" else "False" if s["type"] == "newton" else f"LINEAR[{c.md['name']!r}]"
        w(f"solve({c.py}, results['models'][{c.md['name']!r}]{args}, check_linear={linear})")
        w.indent = ""
    solver = M.main_solver(model)
    w("")
    w(f"# Solver {solver['name']!r}: {solver['type']} of {', '.join(solver['steps'])}")
    steps = [f"solve_{ident(n)}" for n in solver["steps"]]
    if solver["type"] == "sequence":
        for st in steps:
            w(f"{st}()")
        w(f"results['solver'] = {{'name': {solver['name']!r}, 'type': 'sequence'}}")
        return
    names = solver["steps"]
    w(f"history = []")
    w(f"for iteration in range(1, {int(solver['max_iter'])} + 1):")
    w.indent = "    "
    w("before = {" + ", ".join(f"{n!r}: {by[n].py}.from_variables().copy()" for n in names) + "}")
    for n, st in zip(names, steps):
        w(f"{st}()")
        if float(solver["relaxation"]) != 1.0:
            r = float(solver["relaxation"])
            w(f"if iteration > 1:  # relaxation")
            w(f"    {by[n].py}.to_variables({r!r} * {by[n].py}.from_variables() + {1 - r!r} * before[{n!r}])")
    w("change = max(np.linalg.norm(" + "{0}.from_variables() - before[n]) / max(np.linalg.norm({0}.from_variables()), 1e-300)"
      .format("MODELS[n]") + f" for n in {names!r})")
    w("history.append(float(change))")
    w(f"if change < {float(solver['tolerance'])!r}:")
    w("    break")
    w.indent = ""
    w(f"results['solver'] = {{'name': {solver['name']!r}, 'type': 'staggered', 'iterations': iteration, "
      f"'history': history, 'converged': bool(change < {float(solver['tolerance'])!r})}}")
    w("if not results['solver']['converged']:")
    w(f"    results.setdefault('warnings', []).append('staggered solver {solver['name']}: not converged "
      f"after {int(solver['max_iter'])} iterations')")


def _post(c: Ctx, w: ScriptWriter) -> None:
    md, name = c.py, c.md["name"]
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
        exports += [c.mf_data, f"{md}.interpolation({f['name']!r}, {c.mf_data})", repr(f["name"])]
    if c.md.get("vtu"):
        first = exports[0]
        w(f"{first}.export_to_vtu({c.md['vtu']!r}, " + ", ".join(exports) + ")")
        w(f"R['vtu'] = {c.md['vtu']!r}")


def generate(model: dict, meshes: dict[str, tuple[str | Path | None, dict]] | None = None,
             standalone: bool = False) -> str:
    """Solve script for already prepared meshes: meshes = {name: (file, info)}
    (see mesher.ensure_mesh). standalone: the script follows the meshing
    script (runner.export_script), which defines m_<mesh> and RG_<mesh>.
    Raises ValueError if the model is invalid."""
    errors = M.validate(model)
    if errors:
        raise ValueError("invalid model:\n  - " + "\n  - ".join(errors))
    model = P.resolved(M.with_defaults(model))
    meshes = meshes or {}
    ctxs = contexts(model)
    w = ScriptWriter()
    w("#!/usr/bin/env python3")
    w(f"# GetFEM script generated by getfem_gui on {_dt.datetime.now():%Y-%m-%d %H:%M}")
    w(f"# File: {model.get('name', '')}; models: {', '.join(c.md['name'] for c in ctxs)}")
    w("# This script is standalone: it can be read, modified and run again by hand.")
    w("import json")
    w("import time")
    w("")
    w("import numpy as np")
    w("import getfem as gf")
    for line in HELPERS.splitlines():
        w(line)
    w("")
    w("t0 = time.perf_counter()")
    w(f"results = {{'name': {model.get('name', '')!r}, 'status': 'error', 'models': {{"
      + ", ".join(f"{c.md['name']!r}: {{}}" for c in ctxs) + "}}")

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

    w("LINEAR = {}")
    for c in ctxs:
        w.section(f"Model {c.md['name']!r}")
        declare_model(c, w)
        w("")
        w(f"# Terms and conditions of {c.md['name']!r}")
        w(f"LINEAR[{c.md['name']!r}] = True   # False once a term is not")
        setup_model(c, w)
    w("MODELS = {" + ", ".join(f"{c.md['name']!r}: {c.py}" for c in ctxs) + "}")

    w.section("Solve")
    _solver_code(model, ctxs, w)

    w.section("Post-processing")
    for c in ctxs:
        w(f"# Model {c.md['name']!r}")
        _post(c, w)
        w("")
    w("results['status'] = 'ok'")
    w("results['time_s'] = time.perf_counter() - t0")
    w("with open('results.json', 'w', encoding='utf-8') as f:")
    w("    json.dump(results, f, indent=2, ensure_ascii=False)")
    w("dofs = sum(r.get('nb_dofs', 0) for r in results['models'].values())")
    w("print(f\"solved: {dofs} unknowns, {results['time_s']:.2f} s (results in results.json)\")")
    return w.text()
