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

"""Tests de bout en bout contre des solutions exactes.

Chaque cas maille, génère le script GetFEM, l'exécute dans un processus séparé
et compare la solution à la solution analytique via une grandeur dérivée
(l'intégrale de l'erreur au carré), calculée par le script lui-même.

    python -m pytest tests/       (ou : python tests/test_heat.py)
"""

import copy
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import mesher, model as M, runner  # noqa: E402

try:
    import gmsh  # noqa: F401
    HAVE_GMSH = True
except ImportError:
    HAVE_GMSH = False


class Skip(Exception):
    pass


def skip(reason):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(reason)
    raise Skip(reason)


# ------------------------------------------------------------------ outils

def base(dim=2, n=8, element="simplex", degree=1):
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "box", "origin": [0.0] * dim, "size": [1.0] * dim}
    m["meshes"][0].update(divisions=[n] * dim, element=element)
    m["discretization"]["degree"] = degree
    return m


def error_quantity(model, exact):
    model["outputs"]["quantities"].append(
        {"name": "err2", "expr": f"sqr(T - ({exact}))", "selection": "all"})
    return model


def solve(model, workdir=None):
    if workdir is not None:
        return _solve(model, workdir)
    with tempfile.TemporaryDirectory() as tmp:
        return _solve(model, tmp)


def _solve(model, workdir):
    res = runner.run(model, workdir)
    assert res.ok, f"échec : {res.error}\n{res.stderr[-2000:]}"
    return res.results


def l2_error(results):
    return math.sqrt(abs(results["quantities"]["err2"]))


# ------------------------------------------------------------------ cas de l'étape 0

def test_lineaire_et_flux():
    m = error_quantity(base(), "X(1)")
    m["outputs"]["quantities"] += [
        {"name": "flux_xmin", "expr": "-k*Grad_T.Normal", "selection": "xmin"},
        {"name": "flux_xmax", "expr": "-k*Grad_T.Normal", "selection": "xmax"},
    ]
    r = solve(m)
    assert l2_error(r) < 1e-10
    assert abs(r["quantities"]["flux_xmin"] - 1.0) < 1e-10
    assert abs(r["quantities"]["flux_xmax"] + 1.0) < 1e-10


def test_source_quadratique_p2():
    m = base(degree=2)
    m["physics"]["sources"] = [{"selection": "all", "value": 2.0}]
    m["physics"]["boundary_conditions"][1]["value"] = 0.0
    r = solve(error_quantity(m, "X(1)*(1-X(1))"))
    assert l2_error(r) < 1e-10
    assert abs(r["max_T"] - 0.25) < 1e-10


def test_convection_quadrangles():
    m = base(element="quad")
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "xmin", "value": 1.0},
        {"type": "convection", "selection": "xmax", "h": 1.0, "T_ext": 0.0},
    ]
    assert l2_error(solve(error_quantity(m, "1 - X(1)/2"))) < 1e-10


def test_flux_impose():
    m = base()
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "xmin", "value": 0.0},
        {"type": "heat_flux", "selection": "xmax", "value": 2.0},
    ]
    assert l2_error(solve(error_quantity(m, "2*X(1)"))) < 1e-10


def test_3d_expression_sur_tout_le_bord():
    m = base(dim=3, n=3)
    exact = "X(1) + 2*X(2) - X(3)"
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "boundary", "value": exact}]
    assert l2_error(solve(error_quantity(m, exact))) < 1e-10


def test_conductivite_variable():
    m = base(n=10, degree=2)
    m["materials"][0]["properties"]["k"] = "1 + X(1)"
    assert l2_error(solve(error_quantity(m, "log(1 + X(1)) / log(2)"))) < 1e-5


# ------------------------------------------------------------------ géométrie / maillage

TWO_MATERIALS_EXACT = "1.5*X(1) - Heaviside(X(1) - 0.5) * (X(1) - 0.5)"


def test_selections_par_criteres():
    """Domaine 'droite' (boîte), frontière partielle (boîte), réunion de frontières.
    Le matériau défini en dernier l'emporte sur sa sélection."""
    m = base(n=8)
    m["geometries"][0]["selections"] = {
        "droite": {"kind": "domain", "criteria": {"type": "in_box", "min": [0.5 - 1e-9, -1], "max": [2, 2]}},
        "milieu_xmax": {"kind": "boundary", "criteria": {"type": "in_box", "min": [0.99, 0.25 - 1e-9], "max": [1.01, 0.75 + 1e-9]}},
        "parois": {"kind": "boundary", "criteria": {"type": "union", "of": ["ymin", "ymax"]}},
    }
    m["materials"].append({"name": "metal", "selection": "droite", "properties": {"k": 3.0}})
    error_quantity(m, TWO_MATERIALS_EXACT)
    m["outputs"]["quantities"] += [
        {"name": "longueur_milieu", "expr": "1", "selection": "milieu_xmax"},
        {"name": "longueur_parois", "expr": "1", "selection": "parois"},
        {"name": "aire_droite", "expr": "1", "selection": "droite"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        r = solve(m, tmp)
        info = json.loads((Path(tmp) / "maillages" / "maillage.json").read_text())
    q = r["quantities"]
    assert l2_error(r) < 1e-10
    assert abs(q["longueur_milieu"] - 0.5) < 1e-12
    assert abs(q["longueur_parois"] - 2.0) < 1e-12
    assert abs(q["aire_droite"] - 0.5) < 1e-12
    assert info["kinds"]["droite"] == "domain" and info["kinds"]["parois"] == "boundary"


def annulus_model(mesher_spec):
    """Couronne R1 < r < R2 (différence de deux disques), solution exacte en log r."""
    R1, R2 = 0.3, 1.0
    m = M.default_model()
    m["geometries"][0] = {
        "name": "couronne",
        "source": {"type": "csg", "dim": 2, "shape": {"type": "difference", "of": [
            {"type": "ball", "center": [0.0, 0.0], "radius": R2},
            {"type": "ball", "center": [0.0, 0.0], "radius": R1}]}},
        "selections": {"interieur": {"kind": "boundary",
                                     "criteria": {"type": "in_ball", "center": [0, 0], "radius": R1 + 0.05}}},
    }
    m["meshes"] = [dict(name="maillage", geometry="couronne", **mesher_spec)]
    m["discretization"]["degree"] = 2
    exact = f"log(sqrt(sqr(X(1)) + sqr(X(2))) / {R2}) / log({R1 / R2})"
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "boundary", "value": exact}]
    error_quantity(m, exact)
    m["outputs"]["quantities"].append(
        {"name": "flux_interieur", "expr": "-k*Grad_T.Normal", "selection": "interieur"})
    return m, 2 * math.pi / math.log(R1 / R2)


def test_mailleur_getfem_csg():
    m, flux_exact = annulus_model({"mesher": "getfem", "h": 0.1, "order": 2})
    r = solve(m)
    assert l2_error(r) < 1e-3, l2_error(r)
    assert abs(r["quantities"]["flux_interieur"] / flux_exact - 1) < 2e-2, r["quantities"]


def test_mailleur_gmsh_csg():
    if not HAVE_GMSH:
        skip("module gmsh non installé")
    m, flux_exact = annulus_model({"mesher": "gmsh", "size": 0.05, "order": 2})
    r = solve(m)
    assert l2_error(r) < 1e-3
    assert abs(r["quantities"]["flux_interieur"] / flux_exact - 1) < 2e-2


def test_plan_et_normale_sur_plaque_trouee():
    """'normal' prend aussi les faces du trou orientées pareil ; 'plane' non."""
    m = base()
    m["geometries"][0] = {
        "name": "geometrie",
        "source": {"type": "csg", "dim": 2, "shape": {"type": "difference", "of": [
            {"type": "rectangle", "min": [0.0, 0.0], "max": [0.2, 0.1]},
            {"type": "ball", "center": [0.07, 0.05], "radius": 0.02}]}},
        "selections": {
            "gauche_plan": {"kind": "boundary", "criteria": {"type": "plane", "point": [0, 0], "normal": [1, 0]}},
            "gauche_normale": {"kind": "boundary", "criteria": {"type": "normal", "direction": [-1, 0]}},
            "trou": {"kind": "boundary", "criteria": {"type": "in_ball", "center": [0.07, 0.05], "radius": 0.025}},
        }}
    m["meshes"][0] = {"name": "maillage", "geometry": "geometrie", "mesher": "getfem", "h": 0.01, "order": 2}
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "gauche_plan", "value": 0.0}]
    m["outputs"]["quantities"] = [
        {"name": n, "expr": "1", "selection": n} for n in ("gauche_plan", "gauche_normale", "trou")]
    q = solve(m)["quantities"]
    assert abs(q["gauche_plan"] - 0.1) < 1e-12
    assert q["gauche_normale"] > 0.1 + 1e-3          # le piège : une face du trou en plus
    assert abs(q["trou"] - 2 * math.pi * 0.02) < 1e-4


def test_deux_maillages_une_geometrie_et_cache():
    m = base(degree=1)
    m["meshes"] = [
        {"name": "grossier", "geometry": "geometrie", "mesher": "structured", "divisions": [4, 4]},
        {"name": "fin", "geometry": "geometrie", "mesher": "structured", "divisions": [16, 16]},
    ]
    m["physics"]["sources"] = [{"selection": "all", "value": 2.0}]
    m["physics"]["boundary_conditions"][1]["value"] = 0.0
    error_quantity(m, "X(1)*(1-X(1))")
    with tempfile.TemporaryDirectory() as tmp:
        errs = {}
        for name in ("grossier", "fin"):
            m["physics"]["mesh"] = name
            r = solve(m, tmp)
            assert r["mesh"]["name"] == name
            errs[name] = l2_error(r)
        # P1, h divisé par 4 : erreur L2 divisée par ~16
        assert 10 < errs["grossier"] / errs["fin"] < 20, errs
        # Changer la physique ne remaille pas ; changer le maillage, si.
        m["physics"]["sources"][0]["value"] = 3.0
        assert mesher.ensure_mesh(m, "fin", tmp).cached
        m["meshes"][1]["divisions"] = [8, 8]
        assert not mesher.ensure_mesh(m, "fin", tmp).cached


def write_two_material_msh(path, n=4):
    """Carré unité en triangles, format Gmsh 2.2, deux surfaces physiques
    (x < 0.5 et x > 0.5) et quatre bords nommés."""
    N = 2 * n
    nid = lambda i, j: j * (N + 1) + i + 1
    elems = []
    for j in range(N):
        elems += [(1, 1, nid(0, j), nid(0, j + 1)), (1, 2, nid(N, j), nid(N, j + 1))]
    for i in range(N):
        elems += [(1, 3, nid(i, 0), nid(i + 1, 0)), (1, 4, nid(i, N), nid(i + 1, N))]
    for j in range(N):
        for i in range(N):
            tag = 10 if i < n else 11
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            elems += [(2, tag, a, b, c), (2, tag, a, c, d)]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "6",
             '1 1 "gauche"', '1 2 "droite"', '1 3 "bas"', '1 4 "haut"',
             '2 10 "isolant"', '2 11 "metal"', "$EndPhysicalNames",
             "$Nodes", str((N + 1) ** 2)]
    lines += [f"{nid(i, j)} {i / N!r} {j / N!r} 0" for j in range(N + 1) for i in range(N + 1)]
    lines += ["$EndNodes", "$Elements", str(len(elems))]
    for k, (etype, tag, *conn) in enumerate(elems, start=1):
        lines.append(f"{k} {etype} 2 {tag} {tag} " + " ".join(map(str, conn)))
    lines.append("$EndElements")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_geometrie_par_import_de_maillage():
    with tempfile.TemporaryDirectory() as tmp:
        write_two_material_msh(Path(tmp) / "plaque.msh")
        m = M.default_model()
        m["_source_dir"] = tmp
        m["geometries"][0]["source"] = {"type": "mesh_file", "file": "plaque.msh"}
        m["geometries"][0]["selections"] = {
            "bords_horizontaux": {"kind": "boundary", "criteria": {"type": "union", "of": ["bas", "haut"]}}}
        m["meshes"][0] = {"name": "maillage", "geometry": "geometrie", "mesher": "import"}
        m["materials"] = [
            {"name": "isolant", "selection": "isolant", "properties": {"k": 1.0}},
            {"name": "metal", "selection": "metal", "properties": {"k": 3.0}},
        ]
        m["physics"]["boundary_conditions"] = [
            {"type": "temperature", "selection": "gauche", "value": 0.0},
            {"type": "temperature", "selection": "droite", "value": 1.0},
        ]
        error_quantity(m, TWO_MATERIALS_EXACT)
        m["outputs"]["quantities"] += [
            {"name": "flux_gauche", "expr": "-k*Grad_T.Normal", "selection": "gauche"},
            {"name": "longueur_horiz", "expr": "1", "selection": "bords_horizontaux"},
        ]
        assert M.validate(m) == []
        r = solve(m, Path(tmp) / "run")
        assert l2_error(r) < 1e-10
        assert abs(r["quantities"]["flux_gauche"] - 1.5) < 1e-10
        assert abs(r["quantities"]["longueur_horiz"] - 2.0) < 1e-12


def test_migration_v1():
    v1 = {
        "format_version": 1, "name": "ancien",
        "mesh": {"type": "box", "origin": [0, 0], "size": [1, 1], "divisions": [6, 6]},
        "discretization": {"degree": 2},
        "materials": [{"name": "mat", "region": "all", "properties": {"k": 1}}],
        "physics": {"type": "heat_stationary", "variable": "T",
                    "sources": [{"region": "all", "value": 2}],
                    "boundary_conditions": [
                        {"type": "temperature", "region": "xmin", "value": 0},
                        {"type": "temperature", "region": "xmax", "value": 0}]},
        "outputs": {"vtu": "", "quantities": [
            {"name": "err2", "expr": "sqr(T - X(1)*(1-X(1)))", "region": "all"}]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ancien.json"
        p.write_text(json.dumps(v1), encoding="utf-8")
        m = M.load(p)
    assert m["format_version"] == 2 and M.validate(m) == []
    assert m["physics"]["boundary_conditions"][0]["selection"] == "xmin"
    assert l2_error(solve(m)) < 1e-10


# ------------------------------------------------------------------ validation

def test_validation_messages():
    def errs(mutate):
        m = base()
        mutate(m)
        return M.validate(m)

    def bc(sel):
        def f(m):
            m["physics"]["boundary_conditions"] = [{"type": "temperature", "selection": sel, "value": 0}]
        return f

    assert any("sélection inconnue 'nord'" in e for e in errs(bc("nord")))
    assert any("n'est pas une frontière" in e for e in errs(bc("all")))
    assert any("constante près" in e for e in errs(lambda m: m["physics"].update(
        boundary_conditions=[{"type": "heat_flux", "selection": "xmin", "value": 1}])))
    assert any("conductivité manquante" in e for e in errs(
        lambda m: m["materials"][0].update(properties={})))
    assert any("ne sait pas mailler" in e for e in errs(lambda m: m["geometries"][0].update(
        source={"type": "csg", "dim": 2, "shape": {"type": "ball", "center": [0, 0], "radius": 1}})))
    assert any("maillage inconnu" in e for e in errs(lambda m: m["physics"].update(mesh="autre")))
    assert any("ne s'applique qu'à" in e for e in errs(lambda m: m["geometries"][0].update(
        selections={"s": {"kind": "domain", "criteria": {"type": "normal", "direction": [1, 0]}}})))


def test_selection_vide_signalee():
    m = base()
    m["geometries"][0]["selections"] = {
        "rien": {"kind": "boundary", "criteria": {"type": "in_box", "min": [5, 5], "max": [6, 6]}}}
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
    assert not res.ok and "'rien' est vide" in res.error


def test_aller_retour_json():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.json"
        m = base()
        M.save(m, p)
        m2 = M.load(p)
        m2.pop("_source_dir")
        assert m2 == m


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok      {name}")
            except Skip as exc:
                print(f"ignoré  {name} : {exc}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"ÉCHEC   {name} : {exc}")
    sys.exit(1 if failed else 0)
