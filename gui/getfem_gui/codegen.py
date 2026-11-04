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

"""Translation of a model into a readable, standalone GetFEM Python script.

The mesh (with its regions) is prepared separately by `mesher`; the solve
script only loads it. The script does not depend on getfem_gui: it can be
read, modified and run again by hand.

Each physics is a pair of functions registered in PHYSICS, which write their
terms in GWFL, plus a declarative entry in model.PHYSICS_SPECS (properties,
options, boundary conditions) used for validation.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from . import model as M


class ScriptWriter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        self.lines.append(line)

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


def _rg(name: str) -> str:
    return f"REGIONS[{name!r}]"


# ---------------------------------------------------------------- materials

def _materials(model: dict, w: ScriptWriter) -> None:
    """Each property becomes discontinuous finite element data, defined per selection.

    Plain 'k' can then be used in any GWFL expression (terms, fluxes, derived
    quantities), even when k varies from one material to another.
    """
    materials = model["materials"]
    props = sorted({p for mat in materials for p in mat["properties"]})
    w("mf_mat = gf.MeshFem(m, 1)")
    w("mf_mat.set_classical_discontinuous_fem(degree)")
    w("")
    w("def on_region(rg):")
    w("    return np.arange(mf_mat.nbdof()) if rg < 0 else mf_mat.basic_dof_on_region(rg)")
    for prop in props:
        w("")
        w(f"{prop} = np.full(mf_mat.nbdof(), np.nan)")
        for mat in materials:
            if prop not in mat["properties"]:
                continue
            value = mat["properties"][prop]
            dofs = f"on_region({_rg(mat['selection'])})"
            if isinstance(value, (int, float)):
                w(f"{prop}[{dofs}] = {float(value)!r}  # {mat['name']}")
            else:
                w(f"{prop}[{dofs}] = md.interpolation({gwfl(value)!r}, mf_mat)[{dofs}]  # {mat['name']}")
        w(f"if np.isnan({prop}).any():")
        w(f"    raise RuntimeError(\"property '{prop}' is undefined on part of the domain\")")
        w(f"md.add_initialized_fem_data({prop!r}, mf_mat, {prop})")


# ---------------------------------------------------------------- physics
#
# Each physics provides two functions writing script lines:
#   setup(model, w, dim) : variable, macros, terms and boundary conditions
#   post(model, w, dim)  : results entries and the `fields` exported to VTU
# The finite element method `mf` is created beforehand, scalar or vector
# according to M.PHYSICS_SPECS[type]["field"].

def _heat_stationary(model: dict, w: ScriptWriter, dim: int) -> None:
    phys = model["physics"]
    u = phys["variable"]
    w(f"md.add_fem_variable({u!r}, mf)")
    w(f"md.add_macro('q', '-k*Grad_{u}')  # heat flux vector")
    w(f"# -div(k grad {u}) = f")
    w(f"md.add_linear_term(mim, 'k*Grad_{u}.Grad_Test_{u}')")
    for src in phys["sources"]:
        w(f"md.add_source_term(mim, {gwfl(src['value']) + '*Test_' + u!r}, {_rg(src['selection'])})")

    for i, bc in enumerate(phys["boundary_conditions"], start=1):
        rg = _rg(bc["selection"])
        w("")
        if bc["type"] == "temperature":
            data = f"{u}_D{i}"
            w(f"# Imposed temperature on {bc['selection']}")
            w(f"md.add_initialized_fem_data({data!r}, mf, md.interpolation({gwfl(bc['value'])!r}, mf))")
            w(f"md.add_Dirichlet_condition_with_simplification({u!r}, {rg}, {data!r})")
        elif bc["type"] == "heat_flux":
            w(f"# Imposed incoming flux on {bc['selection']}: k grad {u}.n = q")
            w(f"md.add_source_term(mim, {gwfl(bc['value']) + '*Test_' + u!r}, {rg})")
        elif bc["type"] == "convection":
            h, text = gwfl(bc["h"]), gwfl(bc["T_ext"])
            w(f"# Convection on {bc['selection']}: -k grad {u}.n = h ({u} - T_ext)")
            w(f"md.add_linear_term(mim, {h + '*' + u + '*Test_' + u!r}, {rg})")
            w(f"md.add_source_term(mim, {h + '*' + text + '*Test_' + u!r}, {rg})")


def _heat_post(model: dict, w: ScriptWriter, dim: int) -> None:
    u = model["physics"]["variable"]
    w(f"{u} = md.variable({u!r})")
    w(f"results['min_{u}'] = float({u}.min())")
    w(f"results['max_{u}'] = float({u}.max())")
    w("mf_flux = gf.MeshFem(m, m.dim())")
    w("mf_flux.set_classical_discontinuous_fem(max(degree - 1, 0))")
    w("flux = md.interpolation('q', mf_flux)")
    w(f"fields = [mf, {u}, {u!r}, mf_flux, flux, 'heat_flux']")


def _linear_elasticity(model: dict, w: ScriptWriter, dim: int) -> None:
    phys = model["physics"]
    u = phys["variable"]
    plane = phys.get("plane", "strain")
    w(f"md.add_fem_variable({u!r}, mf)")
    w("")
    w("# Lame coefficients from E and nu")
    w("md.add_macro('mu_L', 'E/(2*(1+nu))')")
    if dim == 2 and plane == "stress":
        w("md.add_macro('lambda_L', 'E*nu/(1-sqr(nu))')  # plane stress")
    else:
        w(f"md.add_macro('lambda_L', 'E*nu/((1+nu)*(1-2*nu))')"
          f"{'  # plane strain' if dim == 2 else ''}")
    w(f"md.add_macro('eps', 'Sym(Grad_{u})')")
    w("md.add_macro('sigma', 'lambda_L*Trace(eps)*Id(meshdim) + 2*mu_L*eps')")
    if dim == 3:
        w("md.add_macro('von_mises', 'sqrt(3/2)*Norm(Deviator(sigma))')")
    elif dim == 2:
        szz = "0" if plane == "stress" else "lambda_L*Trace(eps)"
        w(f"md.add_macro('sigma_zz', {szz!r})  # out-of-plane stress")
        w("md.add_macro('von_mises', 'sqrt(0.5*(sqr(sigma(1,1)-sigma(2,2)) + sqr(sigma(2,2)-sigma_zz)"
          " + sqr(sigma_zz-sigma(1,1))) + 3*sqr(sigma(1,2)))')")
    else:
        w("md.add_macro('von_mises', 'abs(sigma(1,1))')")
    w("")
    w(f"# -div sigma({u}) = f")
    w(f"md.add_linear_term(mim, 'sigma:Grad_Test_{u}')")
    for src in phys["sources"]:
        w(f"md.add_source_term(mim, {gwfl_vec(src['value'], dim) + '.Test_' + u!r}, "
          f"{_rg(src['selection'])})")

    if any(bc["type"] == "roller" for bc in phys["boundary_conditions"]):
        w("# Roller conditions use Nitsche's method: no extra unknown, so they can share")
        w("# nodes with other displacement conditions. GetFEM's penalty is 1/(gamma0*h).")
        w("md.add_initialized_data('nitsche_gamma0', [1e-2 / np.max(np.abs(E))])")
    for i, bc in enumerate(phys["boundary_conditions"], start=1):
        rg = _rg(bc["selection"])
        w("")
        if bc["type"] == "fixed":
            w(f"# Fixed on {bc['selection']}: {u} = 0")
            w(f"md.add_Dirichlet_condition_with_simplification({u!r}, {rg})")
        elif bc["type"] == "displacement":
            data = f"{u}_D{i}"
            w(f"# Imposed displacement on {bc['selection']}")
            w(f"md.add_initialized_fem_data({data!r}, mf, "
              f"md.interpolation({gwfl_vec(bc['value'], dim)!r}, mf))")
            w(f"md.add_Dirichlet_condition_with_simplification({u!r}, {rg}, {data!r})")
        elif bc["type"] == "roller":
            w(f"# Roller on {bc['selection']}: {u}.n = 0, free tangential displacement")
            w(f"md.add_normal_Dirichlet_condition_with_Nitsche_method(mim, {u!r}, 'sigma*Normal', "
              f"'nitsche_gamma0', {rg}, 1.0)")
        elif bc["type"] == "traction":
            w(f"# Surface traction on {bc['selection']}: sigma.n = g")
            w(f"md.add_source_term(mim, {gwfl_vec(bc['value'], dim) + '.Test_' + u!r}, {rg})")
        elif bc["type"] == "pressure":
            w(f"# Pressure on {bc['selection']}: sigma.n = -p n")
            w(f"md.add_source_term(mim, {'-' + gwfl(bc['value']) + '*(Normal.Test_' + u + ')'!r}, {rg})")


def _elasticity_post(model: dict, w: ScriptWriter, dim: int) -> None:
    u = model["physics"]["variable"]
    w(f"{u} = md.variable({u!r})")
    w(f"results['max_displacement'] = float(np.max(np.linalg.norm({u}.reshape(-1, m.dim()), axis=1)))")
    w("mf_vm = gf.MeshFem(m, 1)")
    w("mf_vm.set_classical_discontinuous_fem(max(degree - 1, 0))")
    w("von_mises = md.interpolation('von_mises', mf_vm)")
    w("results['max_von_mises'] = float(von_mises.max())")
    w(f"fields = [mf, {u}, 'displacement', mf_vm, von_mises, 'von_mises']")


PHYSICS = {
    "heat_stationary": (_heat_stationary, _heat_post),
    "linear_elasticity": (_linear_elasticity, _elasticity_post),
}


# ---------------------------------------------------------------- script

def generate(model: dict, mesh_path: str | Path, mesh_info: dict) -> str:
    """Solve script for an already prepared mesh (see mesher.ensure_mesh).

    Raises ValueError if the model is invalid.
    """
    errors = M.validate(model)
    if errors:
        raise ValueError("invalid model:\n  - " + "\n  - ".join(errors))
    model = M.with_defaults(model)
    out = model["outputs"]
    ptype = model["physics"]["type"]
    physics, post = PHYSICS[ptype]
    dim = int(mesh_info["dim"])
    vector = M.PHYSICS_SPECS[ptype]["field"] == M.VECTOR

    w = ScriptWriter()
    w("#!/usr/bin/env python3")
    w(f"# GetFEM script generated by getfem_gui on {_dt.datetime.now():%Y-%m-%d %H:%M}")
    w(f"# Model: {model.get('name', '')}")
    w("# This script is standalone: it can be read, modified and run again by hand.")
    w("import json")
    w("import time")
    w("")
    w("import numpy as np")
    w("import getfem as gf")
    w("")
    w("t0 = time.perf_counter()")
    w(f"results = {{'model': {model.get('name', '')!r}, 'status': 'error'}}")
    w(f"degree = {model['discretization']['degree']}")

    w.section(f"Mesh {mesh_info.get('mesh', '')!r} (prepared by the meshing step)")
    w(f"m = gf.Mesh('load', {str(Path(mesh_path).resolve())!r})")
    w(f"REGIONS = {mesh_info['regions']!r}")
    w("results['mesh'] = {'name': %r, 'nb_elements': int(m.nbcvs())}" % mesh_info.get("mesh", ""))

    w.section("Discretization")
    w(f"mf = gf.MeshFem(m, {'m.dim()' if vector else '1'})")
    w("mf.set_classical_fem(degree)")
    w("mim = gf.MeshIm(m, 2 * degree)")
    w("md = gf.Model('real')")

    w.section("Materials")
    _materials(model, w)

    w.section(f"Physics: {ptype}")
    physics(model, w, dim)

    w.section("Solve")
    w("solve_info = md.solve()  # (number of iterations, converged)")
    w("converged = bool(solve_info[1]) if isinstance(solve_info, tuple) and len(solve_info) > 1 else True")
    w("results['nb_dofs'] = int(md.nbdof())")
    w("# Check that the linear system was actually solved: on a singular system some")
    w("# GetFEM versions only emit a warning and leave a zero solution.")
    w("md.assembly('build_all')")
    w("K, F, U = md.tangent_matrix(), md.rhs(), md.from_variables()")
    w("residual = np.linalg.norm(K.mult(U) - F) / max(np.linalg.norm(F), 1e-300)")
    w("results['relative_residual'] = float(residual)")
    w("if not residual < 1e-6:")
    w("    raise RuntimeError(f'linear solve failed (relative residual {residual:.1e}): '")
    w("                       'the system is probably singular, check the boundary conditions')")
    w("if not converged:  # solved, but the solver met a singular matrix")
    w("    results.setdefault('warnings', []).append(")
    w("        'the linear system is singular: the solution is not unique '")
    w("        '(for instance a rigid body motion is not prevented)')")

    w.section("Post-processing")
    post(model, w, dim)
    if out["quantities"]:
        w("quantities = {}")
        for q in out["quantities"]:
            w(f"quantities[{q['name']!r}] = float(gf.asm('generic', mim, 0, {str(q['expr'])!r}, "
              f"{_rg(q['selection'])}, md))")
        w("results['quantities'] = quantities")
    if out.get("vtu"):
        w(f"mf.export_to_vtu({out['vtu']!r}, *fields)")
        w(f"results['vtu'] = {out['vtu']!r}")

    w("")
    w("results['status'] = 'ok'")
    w("results['time_s'] = time.perf_counter() - t0")
    w("with open('results.json', 'w', encoding='utf-8') as f:")
    w("    json.dump(results, f, indent=2, ensure_ascii=False)")
    w("print(json.dumps(results, indent=2, ensure_ascii=False))")
    return w.text()
