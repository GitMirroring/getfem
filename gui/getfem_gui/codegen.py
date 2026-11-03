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

"""Traduction d'un modèle en script Python GetFEM de calcul, lisible et autonome.

Le maillage (et ses régions) est préparé à part par `mesher` ; le script de
calcul ne fait que le charger. Il ne dépend pas de getfem_gui : on peut le
lire, le modifier et le relancer à la main.

Chaque physique est une paire de fonctions enregistrée dans PHYSICS, qui
écrit ses termes en GWFL. Ajouter une physique revient à ajouter une paire.
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
    """Valeur du modèle -> morceau d'expression GWFL."""
    if isinstance(value, (int, float)):
        return repr(float(value))
    return f"({value.strip()})"


def _rg(name: str) -> str:
    return f"REGIONS[{name!r}]"


# ---------------------------------------------------------------- matériaux

def _materials(model: dict, w: ScriptWriter) -> None:
    """Chaque propriété devient une donnée EF discontinue, définie par sélection.

    On peut ainsi écrire simplement 'k' dans n'importe quelle expression GWFL
    (termes, flux, grandeurs dérivées), même si k varie d'un matériau à l'autre.
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
        w(f"    raise RuntimeError(\"propriété '{prop}' non définie sur une partie du domaine\")")
        w(f"md.add_initialized_fem_data({prop!r}, mf_mat, {prop})")


# ---------------------------------------------------------------- physiques

def _heat_stationary(model: dict, w: ScriptWriter) -> None:
    phys = model["physics"]
    u = phys["variable"]
    w(f"md.add_fem_variable({u!r}, mf)")
    w(f"# -div(k grad {u}) = f")
    w(f"md.add_linear_term(mim, 'k*Grad_{u}.Grad_Test_{u}')")
    for src in phys["sources"]:
        w(f"md.add_source_term(mim, {gwfl(src['value']) + '*Test_' + u!r}, {_rg(src['selection'])})")

    for i, bc in enumerate(phys["boundary_conditions"], start=1):
        rg = _rg(bc["selection"])
        w("")
        if bc["type"] == "temperature":
            data = f"{u}_D{i}"
            w(f"# Température imposée sur {bc['selection']}")
            w(f"md.add_initialized_fem_data({data!r}, mf, md.interpolation({gwfl(bc['value'])!r}, mf))")
            w(f"md.add_Dirichlet_condition_with_simplification({u!r}, {rg}, {data!r})")
        elif bc["type"] == "heat_flux":
            w(f"# Flux entrant imposé sur {bc['selection']} : k grad {u}.n = q")
            w(f"md.add_source_term(mim, {gwfl(bc['value']) + '*Test_' + u!r}, {rg})")
        elif bc["type"] == "convection":
            h, text = gwfl(bc["h"]), gwfl(bc["T_ext"])
            w(f"# Convection sur {bc['selection']} : -k grad {u}.n = h ({u} - T_ext)")
            w(f"md.add_linear_term(mim, {h + '*' + u + '*Test_' + u!r}, {rg})")
            w(f"md.add_source_term(mim, {h + '*' + text + '*Test_' + u!r}, {rg})")


def _heat_post(model: dict, w: ScriptWriter) -> None:
    u = model["physics"]["variable"]
    w(f"{u} = md.variable({u!r})")
    w(f"results['min_{u}'] = float({u}.min())")
    w(f"results['max_{u}'] = float({u}.max())")
    w("mf_flux = gf.MeshFem(m, m.dim())")
    w("mf_flux.set_classical_discontinuous_fem(max(degree - 1, 0))")
    w(f"flux = md.interpolation('-k*Grad_{u}', mf_flux)")
    w(f"fields = [mf, {u}, {u!r}, mf_flux, flux, 'flux_thermique']")


PHYSICS = {"heat_stationary": (_heat_stationary, _heat_post)}


# ---------------------------------------------------------------- script

def generate(model: dict, mesh_path: str | Path, mesh_info: dict) -> str:
    """Script de calcul pour un maillage déjà préparé (voir mesher.ensure_mesh).

    Lève ValueError si le modèle est invalide.
    """
    errors = M.validate(model)
    if errors:
        raise ValueError("modèle invalide :\n  - " + "\n  - ".join(errors))
    model = M.with_defaults(model)
    out = model["outputs"]
    physics, post = PHYSICS[model["physics"]["type"]]

    w = ScriptWriter()
    w("#!/usr/bin/env python3")
    w(f"# Script GetFEM généré par getfem_gui le {_dt.datetime.now():%Y-%m-%d %H:%M}")
    w(f"# Modèle : {model.get('name', '')}")
    w("# Ce script est autonome : il peut être lu, modifié et relancé à la main.")
    w("import json")
    w("import time")
    w("")
    w("import numpy as np")
    w("import getfem as gf")
    w("")
    w("t0 = time.perf_counter()")
    w(f"results = {{'model': {model.get('name', '')!r}, 'status': 'error'}}")
    w(f"degree = {model['discretization']['degree']}")

    w.section(f"Maillage {mesh_info.get('mesh', '')!r} (préparé par l'étape de maillage)")
    w(f"m = gf.Mesh('load', {str(Path(mesh_path).resolve())!r})")
    w(f"REGIONS = {mesh_info['regions']!r}")
    w("results['mesh'] = {'name': %r, 'nb_elements': int(m.nbcvs())}" % mesh_info.get("mesh", ""))

    w.section("Discrétisation")
    w("mf = gf.MeshFem(m, 1)")
    w("mf.set_classical_fem(degree)")
    w("mim = gf.MeshIm(m, 2 * degree)")
    w("md = gf.Model('real')")

    w.section("Matériaux")
    _materials(model, w)

    w.section(f"Physique : {model['physics']['type']}")
    physics(model, w)

    w.section("Résolution")
    w("md.solve()")
    w("results['nb_dofs'] = int(md.nbdof())")

    w.section("Post-traitement")
    post(model, w)
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
