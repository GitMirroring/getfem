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

"""End-to-end tests against exact solutions.

Each case meshes, generates the GetFEM script, runs it in a separate process
and compares the solution with the analytical one through a derived quantity
(the integral of the squared error), computed by the script itself.

    python -m pytest tests/       (or: python tests/test_heat.py)
"""

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


# ------------------------------------------------------------------ helpers

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
    assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
    return res.results


def l2_error(results):
    return math.sqrt(abs(results["quantities"]["err2"]))


# ------------------------------------------------------------------ physics on boxes

def test_linear_solution_and_fluxes():
    m = error_quantity(base(), "X(1)")
    m["outputs"]["quantities"] += [
        {"name": "flux_xmin", "expr": "-k*Grad_T.Normal", "selection": "xmin"},
        {"name": "flux_xmax", "expr": "-k*Grad_T.Normal", "selection": "xmax"},
    ]
    r = solve(m)
    assert l2_error(r) < 1e-10
    assert abs(r["quantities"]["flux_xmin"] - 1.0) < 1e-10
    assert abs(r["quantities"]["flux_xmax"] + 1.0) < 1e-10


def test_quadratic_source_p2():
    m = base(degree=2)
    m["physics"]["sources"] = [{"selection": "all", "value": 2.0}]
    m["physics"]["boundary_conditions"][1]["value"] = 0.0
    r = solve(error_quantity(m, "X(1)*(1-X(1))"))
    assert l2_error(r) < 1e-10
    assert abs(r["max_T"] - 0.25) < 1e-10


def test_convection_on_quadrilaterals():
    m = base(element="quad")
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "xmin", "value": 1.0},
        {"type": "convection", "selection": "xmax", "h": 1.0, "T_ext": 0.0},
    ]
    assert l2_error(solve(error_quantity(m, "1 - X(1)/2"))) < 1e-10


def test_imposed_flux():
    m = base()
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "xmin", "value": 0.0},
        {"type": "heat_flux", "selection": "xmax", "value": 2.0},
    ]
    assert l2_error(solve(error_quantity(m, "2*X(1)"))) < 1e-10


def test_3d_expression_on_whole_boundary():
    m = base(dim=3, n=3)
    exact = "X(1) + 2*X(2) - X(3)"
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "boundary", "value": exact}]
    assert l2_error(solve(error_quantity(m, exact))) < 1e-10


def test_variable_conductivity():
    m = base(n=10, degree=2)
    m["materials"][0]["properties"]["k"] = "1 + X(1)"
    assert l2_error(solve(error_quantity(m, "log(1 + X(1)) / log(2)"))) < 1e-5


# ------------------------------------------------------------------ geometry / mesh

TWO_MATERIALS_EXACT = "1.5*X(1) - Heaviside(X(1) - 0.5) * (X(1) - 0.5)"


def test_selections_by_criteria():
    """Domain 'right' (box), partial boundary (box), union of boundaries.
    The material listed last wins on its selection."""
    m = base(n=8)
    m["geometries"][0]["selections"] = {
        "right": {"kind": "domain", "criteria": {"type": "in_box", "min": [0.5 - 1e-9, -1], "max": [2, 2]}},
        "middle_xmax": {"kind": "boundary", "criteria": {"type": "in_box", "min": [0.99, 0.25 - 1e-9], "max": [1.01, 0.75 + 1e-9]}},
        "walls": {"kind": "boundary", "criteria": {"type": "union", "of": ["ymin", "ymax"]}},
    }
    m["materials"].append({"name": "metal", "selection": "right", "properties": {"k": 3.0}})
    error_quantity(m, TWO_MATERIALS_EXACT)
    m["outputs"]["quantities"] += [
        {"name": "length_middle", "expr": "1", "selection": "middle_xmax"},
        {"name": "length_walls", "expr": "1", "selection": "walls"},
        {"name": "area_right", "expr": "1", "selection": "right"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        r = solve(m, tmp)
        info = json.loads((Path(tmp) / mesher.MESH_DIR / "mesh.json").read_text())
    q = r["quantities"]
    assert l2_error(r) < 1e-10
    assert abs(q["length_middle"] - 0.5) < 1e-12
    assert abs(q["length_walls"] - 2.0) < 1e-12
    assert abs(q["area_right"] - 0.5) < 1e-12
    assert info["kinds"]["right"] == "domain" and info["kinds"]["walls"] == "boundary"


def annulus_model(mesher_spec):
    """Annulus R1 < r < R2 (difference of two disks), exact solution in log r."""
    R1, R2 = 0.3, 1.0
    m = M.default_model()
    m["geometries"][0] = {
        "name": "annulus",
        "source": {"type": "csg", "dim": 2, "shape": {"type": "difference", "of": [
            {"type": "ball", "center": [0.0, 0.0], "radius": R2},
            {"type": "ball", "center": [0.0, 0.0], "radius": R1}]}},
        "selections": {"inner": {"kind": "boundary",
                                 "criteria": {"type": "in_ball", "center": [0, 0], "radius": R1 + 0.05}}},
    }
    m["meshes"] = [dict(name="mesh", geometry="annulus", **mesher_spec)]
    m["discretization"]["degree"] = 2
    exact = f"log(sqrt(sqr(X(1)) + sqr(X(2))) / {R2}) / log({R1 / R2})"
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "boundary", "value": exact}]
    error_quantity(m, exact)
    m["outputs"]["quantities"].append(
        {"name": "inner_flux", "expr": "-k*Grad_T.Normal", "selection": "inner"})
    return m, 2 * math.pi / math.log(R1 / R2)


def test_getfem_mesher_csg():
    m, flux_exact = annulus_model({"mesher": "getfem", "h": 0.1, "order": 2})
    r = solve(m)
    assert l2_error(r) < 1e-3, l2_error(r)
    assert abs(r["quantities"]["inner_flux"] / flux_exact - 1) < 2e-2, r["quantities"]


def test_gmsh_mesher_csg():
    if not HAVE_GMSH:
        skip("gmsh module not installed")
    m, flux_exact = annulus_model({"mesher": "gmsh", "size": 0.05, "order": 2})
    r = solve(m)
    assert l2_error(r) < 1e-3
    assert abs(r["quantities"]["inner_flux"] / flux_exact - 1) < 2e-2


def test_plane_versus_normal_on_plate_with_hole():
    """'normal' also picks hole faces with the same orientation; 'plane' does not."""
    m = base()
    m["geometries"][0] = {
        "name": "geometry",
        "source": {"type": "csg", "dim": 2, "shape": {"type": "difference", "of": [
            {"type": "rectangle", "min": [0.0, 0.0], "max": [0.2, 0.1]},
            {"type": "ball", "center": [0.07, 0.05], "radius": 0.02}]}},
        "selections": {
            "left_plane": {"kind": "boundary", "criteria": {"type": "plane", "point": [0, 0], "normal": [1, 0]}},
            "left_normal": {"kind": "boundary", "criteria": {"type": "normal", "direction": [-1, 0]}},
            "hole": {"kind": "boundary", "criteria": {"type": "in_ball", "center": [0.07, 0.05], "radius": 0.025}},
        }}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "getfem", "h": 0.01, "order": 2}
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "left_plane", "value": 0.0}]
    m["outputs"]["quantities"] = [
        {"name": n, "expr": "1", "selection": n} for n in ("left_plane", "left_normal", "hole")]
    q = solve(m)["quantities"]
    assert abs(q["left_plane"] - 0.1) < 1e-12
    assert q["left_normal"] > 0.1 + 1e-3          # the pitfall: one hole face too many
    assert abs(q["hole"] - 2 * math.pi * 0.02) < 1e-4


def test_two_meshes_one_geometry_and_cache():
    m = base(degree=1)
    m["meshes"] = [
        {"name": "coarse", "geometry": "geometry", "mesher": "structured", "divisions": [4, 4]},
        {"name": "fine", "geometry": "geometry", "mesher": "structured", "divisions": [16, 16]},
    ]
    m["physics"]["sources"] = [{"selection": "all", "value": 2.0}]
    m["physics"]["boundary_conditions"][1]["value"] = 0.0
    error_quantity(m, "X(1)*(1-X(1))")
    with tempfile.TemporaryDirectory() as tmp:
        errs = {}
        for name in ("coarse", "fine"):
            m["physics"]["mesh"] = name
            r = solve(m, tmp)
            assert r["mesh"]["name"] == name
            errs[name] = l2_error(r)
        # P1, h divided by 4: L2 error divided by about 16
        assert 10 < errs["coarse"] / errs["fine"] < 20, errs
        # Changing the physics does not remesh; changing the mesh does.
        m["physics"]["sources"][0]["value"] = 3.0
        assert mesher.ensure_mesh(m, "fine", tmp).cached
        m["meshes"][1]["divisions"] = [8, 8]
        assert not mesher.ensure_mesh(m, "fine", tmp).cached


def write_two_material_msh(path, n=4):
    """Unit square in triangles, Gmsh 2.2 format, two physical surfaces
    (x < 0.5 and x > 0.5) and four named edges."""
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
             '1 1 "left"', '1 2 "right"', '1 3 "bottom"', '1 4 "top"',
             '2 10 "insulator"', '2 11 "metal"', "$EndPhysicalNames",
             "$Nodes", str((N + 1) ** 2)]
    lines += [f"{nid(i, j)} {i / N!r} {j / N!r} 0" for j in range(N + 1) for i in range(N + 1)]
    lines += ["$EndNodes", "$Elements", str(len(elems))]
    for k, (etype, tag, *conn) in enumerate(elems, start=1):
        lines.append(f"{k} {etype} 2 {tag} {tag} " + " ".join(map(str, conn)))
    lines.append("$EndElements")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_geometry_from_imported_mesh():
    with tempfile.TemporaryDirectory() as tmp:
        write_two_material_msh(Path(tmp) / "plate.msh")
        m = M.default_model()
        m["_source_dir"] = tmp
        m["geometries"][0]["source"] = {"type": "mesh_file", "file": "plate.msh"}
        m["geometries"][0]["selections"] = {
            "horizontal_edges": {"kind": "boundary", "criteria": {"type": "union", "of": ["bottom", "top"]}}}
        m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "import"}
        m["materials"] = [
            {"name": "insulator", "selection": "insulator", "properties": {"k": 1.0}},
            {"name": "metal", "selection": "metal", "properties": {"k": 3.0}},
        ]
        m["physics"]["boundary_conditions"] = [
            {"type": "temperature", "selection": "left", "value": 0.0},
            {"type": "temperature", "selection": "right", "value": 1.0},
        ]
        error_quantity(m, TWO_MATERIALS_EXACT)
        m["outputs"]["quantities"] += [
            {"name": "flux_left", "expr": "-k*Grad_T.Normal", "selection": "left"},
            {"name": "length_horizontal", "expr": "1", "selection": "horizontal_edges"},
        ]
        assert M.validate(m) == []
        r = solve(m, Path(tmp) / "run")
        assert l2_error(r) < 1e-10
        assert abs(r["quantities"]["flux_left"] - 1.5) < 1e-10
        assert abs(r["quantities"]["length_horizontal"] - 2.0) < 1e-12


def test_migration_from_v1():
    v1 = {
        "format_version": 1, "name": "old",
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
        p = Path(tmp) / "old.json"
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

    assert any("unknown selection 'north'" in e for e in errs(bc("north")))
    assert any("is not a boundary" in e for e in errs(bc("all")))
    assert any("up to a constant" in e for e in errs(lambda m: m["physics"].update(
        boundary_conditions=[{"type": "heat_flux", "selection": "xmin", "value": 1}])))
    assert any("missing thermal conductivity" in e for e in errs(
        lambda m: m["materials"][0].update(properties={})))
    assert any("cannot mesh" in e for e in errs(lambda m: m["geometries"][0].update(
        source={"type": "csg", "dim": 2, "shape": {"type": "ball", "center": [0, 0], "radius": 1}})))
    assert any("unknown mesh" in e for e in errs(lambda m: m["physics"].update(mesh="other")))
    assert any("only applies to" in e for e in errs(lambda m: m["geometries"][0].update(
        selections={"s": {"kind": "domain", "criteria": {"type": "normal", "direction": [1, 0]}}})))


def test_empty_selection_is_reported():
    m = base()
    m["geometries"][0]["selections"] = {
        "nothing": {"kind": "boundary", "criteria": {"type": "in_box", "min": [5, 5], "max": [6, 6]}}}
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
    assert not res.ok and "'nothing' is empty" in res.error


def test_json_round_trip():
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
                print(f"ok       {name}")
            except Skip as exc:
                print(f"skipped  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAILED   {name}: {exc}")
    sys.exit(1 if failed else 0)


# ------------------------------------------------------------------ regions (subdomains)

def layered_model(h=0.1):
    """Bar [0, 1] x [0, 0.5] made of two materials, k = 1 for x < 0.5 and k = 3
    beyond (a second rectangle): T = 0 at x = 0, T = 1 at x = 1. The exact
    solution is piecewise linear, with a kink at the interface x = 0.5."""
    m = M.default_model()
    m["geometries"][0] = {
        "name": "bar",
        "source": {"type": "csg", "dim": 2, "shape": {"type": "union", "of": [
            {"type": "rectangle", "min": [0.0, 0.0], "max": [1.0, 0.5]},
            {"type": "rectangle", "min": [0.5, 0.0], "max": [1.0, 0.5]}]}},
        "selections": {
            "left": {"kind": "domain", "criteria": {"type": "regions", "regions": [[[0]]]}},
            "right": {"kind": "domain", "criteria": {"type": "regions", "regions": [[[0], [1]]]}},
            "x0": {"kind": "boundary", "criteria": {"type": "plane", "point": [0, 0], "normal": [1, 0]}},
            "x1": {"kind": "boundary", "criteria": {"type": "plane", "point": [1, 0], "normal": [1, 0]}},
        },
    }
    m["meshes"] = [{"name": "mesh", "geometry": "bar", "mesher": "getfem", "h": h, "order": 1}]
    m["discretization"]["degree"] = 1
    m["materials"] = [{"name": "soft", "selection": "left", "properties": {"k": 1.0}},
                      {"name": "hard", "selection": "right", "properties": {"k": 3.0}}]
    m["physics"]["sources"] = []
    m["physics"]["boundary_conditions"] = [
        {"type": "temperature", "selection": "x0", "value": 0.0},
        {"type": "temperature", "selection": "x1", "value": 1.0}]
    q = 1.0 / (0.5 / 1.0 + 0.5 / 3.0)
    error_quantity(m, f"{q}*X(1) - {2 * q / 3}*pos_part(X(1) - 0.5)")
    m["outputs"]["quantities"] += [
        {"name": "area_left", "expr": "1", "selection": "left"},
        {"name": "area_right", "expr": "1", "selection": "right"}]
    return m


def test_regions_follow_the_interface():
    """The GetFEM mesher follows the side of the inner rectangle: P1 elements
    give the exact (piecewise linear) solution, and the areas are exact."""
    r = solve(layered_model())
    assert abs(r["quantities"]["area_left"] - 0.25) < 1e-12, r["quantities"]
    assert abs(r["quantities"]["area_right"] - 0.25) < 1e-12, r["quantities"]
    assert l2_error(r) < 1e-8, l2_error(r)


def test_disk_inclusion_regions():
    """A disk inside a square: the elements of the two regions are on either
    side of the circle (no element crosses it), and they tile the square."""
    import numpy as np
    import getfem as gf
    m = M.default_model()
    m["geometries"][0] = {
        "name": "g",
        "source": {"type": "csg", "dim": 2, "shape": {"type": "union", "of": [
            {"type": "rectangle", "min": [0.0, 0.0], "max": [1.0, 1.0]},
            {"type": "ball", "center": [0.5, 0.5], "radius": 0.25}]}},
        "selections": {
            "matrix": {"kind": "domain", "criteria": {"type": "regions", "regions": [[[0]]]}},
            "inclusion": {"kind": "domain", "criteria": {"type": "regions", "regions": [[[0], [1]]]}},
        },
    }
    m["meshes"] = [{"name": "mesh", "geometry": "g", "mesher": "getfem", "h": 0.08, "order": 1}]
    m["physics"]["sources"] = []
    m["physics"]["boundary_conditions"] = [{"type": "temperature", "selection": "boundary", "value": 0.0}]
    with tempfile.TemporaryDirectory() as tmp:
        res = mesher.ensure_mesh(m, "mesh", tmp)
        assert res.ok, res.error
        mesh = gf.Mesh("load", str(res.path))
        regions = res.info["regions"]
        cv_in = mesh.region(regions["inclusion"])[0]
        cv_out = mesh.region(regions["matrix"])[0]
        assert len(set(cv_in) & set(cv_out)) == 0
        assert len(cv_in) + len(cv_out) == mesh.nbcvs()
        P = mesh.pts()
        for cvs, sign in ((cv_in, -1), (cv_out, 1)):
            for cv in cvs:
                d = np.linalg.norm(P[:, mesh.pid_from_cvid(cv)[0]].T - 0.5, axis=1) - 0.25
                assert np.all(sign * d > -1e-9), (cv, d)
