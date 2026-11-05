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

"""Polygons: geometry, format, editing (table, handles, drawing), mesh and
solve, extrusion.

    python -m pytest tests/       (or: python tests/test_polygon.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, geometry2d as G, mesher, model as M, parameters as P, runner  # noqa: E402

L_SHAPE = [[0, 0], [1, 0], [1, 0.5], ["a", 0.5], ["a", 1], [0, 1]]       # a = 0.5
ROOT = ("geometries", 0, "source", "shape")


def model_with(shape, dim=2, h=0.1):
    m = M.default_model()
    m["parameters"] = [{"name": "a", "value": 0.5}]
    m["geometries"][0]["source"] = {"type": "csg", "dim": dim, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = {}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "getfem", "h": h, "order": 1}
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": "x + 2*y"}]
    md["outputs"] = [{"name": "area", "expr": "1", "selection": "all"},
                     {"name": "length", "expr": "1", "selection": "boundary"},
                     {"name": "err2", "expr": "sqr(T - x - 2*y)", "selection": "all"}]
    return m


def test_geometry():
    V = np.array([[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]], float)
    d = G.polygon_sdf(V, np.array([[0.25, 0.25], [0.75, 0.75], [0.6, 0.6], [2, 0], [0.5, 0.75]]))
    assert np.allclose(d, [-0.25, 0.25, 0.1, 1.0, 0.0])
    assert abs(G.polygon_area(V) - 0.75) < 1e-12 and G.polygon_area(V[::-1]) < 0
    assert G.polygon_problem(V) is None and G.polygon_problem(V[::-1]) is None
    assert "cross" in G.polygon_problem([[0, 0], [1, 1], [1, 0], [0, 1]])
    assert "coincide" in G.polygon_problem([[0, 0], [1, 0], [1, 0], [0, 1]])
    assert "area" in G.polygon_problem([[0, 0], [1, 0], [2, 0]])
    # boundary pieces of a polygon cut by a disk
    shape = {"type": "difference", "of": [{"type": "polygon", "points": V.tolist()},
                                         {"type": "ball", "center": [0, 0], "radius": 0.25}]}
    pieces = G.pieces(shape)
    length = sum(np.linalg.norm(np.diff(p.points(512), axis=0), axis=1).sum() for p in pieces)
    assert abs(length - (4 - 0.5 + 2 * math.pi * 0.25 / 4)) < 1e-4
    side = next(p for p in pieces if p.side.side == "side1")
    assert abs(side.t0 - 0.25) < 1e-9 and side.t1 == 1.0
    assert G.side_label(shape, side.side.ref) == "side 1 of polygon 1"


def test_format():
    m = model_with({"type": "polygon", "points": L_SHAPE})
    assert M.validate(m) == []
    assert P.resolved(m)["geometries"][0]["source"]["shape"]["points"][3] == [0.5, 0.5]
    bad = copy.deepcopy(m)
    E.get_at(bad, ROOT)["points"][3] = [0.5, -0.5]                    # crosses itself
    assert any("not a simple polygon" in e for e in M.validate(bad))
    E.get_at(bad, ROOT)["points"] = [[0, 0], [1, 0]]
    assert any("at least three vertices" in e for e in M.validate(bad))
    bad = model_with({"type": "polygon", "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}, dim=3)
    assert any("polygon only exists in 2D" in e for e in M.validate(bad))
    script = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert "gf.MesherObject('polygon', np.array([[0.0, 0.0], [1.0, 0.0]," in script
    gm = copy.deepcopy(m)
    gm["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "gmsh", "size": 0.1}
    code = mesher.mesh_script(P.resolved(gm), gm["meshes"][0], "mesh")
    assert "addCurveLoop" in code and "addPlaneSurface" in code
    compile(code, "mesh.py", "exec")


def test_editing():
    m = model_with({"type": "union", "of": [{"type": "rectangle", "min": [0, 0], "max": [1, "a"]},
                                            {"type": "ball", "center": [1, 0.25], "radius": 0.25}]})
    # a rectangle changed into a polygon keeps its corners, a disk gives an octagon
    f = next(f for f in E.fields(m, ROOT + ("of", 0)) if f.op == "shape_type")
    assert "polygon" in [v for v, _ in f.choices]
    E.apply(m, f, "polygon")
    assert E.get_at(m, ROOT + ("of", 0)) == {"type": "polygon", "points": [[0, 0], [1, 0], [1, 0.5], [0, 0.5]]}
    E.apply(m, next(f for f in E.fields(m, ROOT + ("of", 1)) if f.op == "shape_type"), "polygon")
    assert len(E.get_at(m, ROOT + ("of", 1))["points"]) == 8 and M.validate(m) == []
    poly = ROOT + ("of", 0)
    table = next(f for f in E.fields(m, poly) if f.kind == "vertex_table")
    assert table.key == poly + ("points",) and E.vertex_rows(m, table.key)[2] == (1, 0.5, 1, 0.5)
    # a vertex in the middle of side 2, then removed; at least three vertices
    assert E.do_action(m, poly, "insert_vertex:1") == poly
    assert E.get_at(m, poly)["points"][2] == [1.0, 0.25]
    E.do_action(m, poly, "remove_vertex:2")
    E.do_action(m, poly, "remove_vertex:0")
    try:
        E.do_action(m, poly, "remove_vertex:0")
        raise AssertionError("a triangle lost a vertex")
    except ValueError:
        pass
    # handles: vertices, middles of the sides (a new vertex), center
    m = model_with({"type": "polygon", "points": L_SHAPE})
    names = [n for n, _ in E.shape_handles(m, ROOT)]
    assert names[:6] == [f"v{i}" for i in range(6)] and names[6:12] == [f"m{i}" for i in range(6)]
    notes = E.move_handle(m, ROOT, "v3", [0.4, 0.5])          # 'a' is a parameter: it is changed
    assert m["parameters"][0]["value"] == 0.4 and E.get_at(m, ROOT)["points"][3] == ["a", 0.5] and notes
    E.move_handle(m, ROOT, "m0", [0.5, -0.1])
    assert E.get_at(m, ROOT)["points"][1] == [0.5, -0.1] and len(E.get_at(m, ROOT)["points"]) == 7
    try:
        E.move_handle(m, ROOT, "v2", [0.2, 2.0])               # would cross itself
        raise AssertionError("a crossing polygon was accepted")
    except ValueError as exc:
        assert "simple polygon" in str(exc) and len(E.get_at(m, ROOT)["points"]) == 7
    m = model_with({"type": "polygon", "points": [[0, 0], [1, 0], [0, 1]]})
    E.move_handle(m, ROOT, "center", [1, 1])
    assert np.allclose(E.get_at(m, ROOT)["points"], [[2 / 3, 2 / 3], [5 / 3, 2 / 3], [2 / 3, 5 / 3]])
    # a polygon in the profile of an extrusion
    ext = model_with({"type": "extrusion", "z": [0, 1], "profile": {"type": "rectangle", "min": [0, 0],
                                                                    "max": [1, 1]}}, dim=3)
    E.apply(ext, next(f for f in E.fields(ext, ROOT + ("profile",)) if f.op == "shape_type"), "polygon")
    assert E.get_at(ext, ROOT + ("profile",))["type"] == "polygon" and M.validate(ext) == []


def test_side_references_follow_the_edits():
    """Edge selections follow a rectangle changed into a polygon, a vertex
    inserted (the side split) or removed (two sides merged)."""
    m = model_with({"type": "union", "of": [{"type": "rectangle", "min": [0, 0], "max": [1, 1]},
                                            {"type": "ball", "center": [1, 1], "radius": 0.3}]})
    m["geometries"][0]["selections"] = {"left": {"kind": "boundary", "criteria": {"type": "edges", "edges": [
        {"shape": [0], "side": "xmin"}, {"shape": [0], "side": "ymax"}]}}}
    rect = ROOT + ("of", 0)
    E.apply(m, next(f for f in E.fields(m, rect) if f.op == "shape_type"), "polygon")
    edges = lambda: [r["side"] for r in m["geometries"][0]["selections"]["left"]["criteria"]["edges"]]  # noqa: E731
    assert edges() == ["side4", "side3"] and M.validate(m) == []
    E.do_action(m, rect, "insert_vertex:2")              # in the side 3 (top)
    assert edges() == ["side5", "side3", "side4"] and M.validate(m) == []
    E.do_action(m, rect, "remove_vertex:3")              # the new vertex again
    assert edges() == ["side4", "side3"] and M.validate(m) == []
    E.do_action(m, rect, "remove_vertex:0")              # (0, 0): sides 4 and 1 merged
    assert edges() == ["side3", "side2"] and M.validate(m) == []


def _solve(m):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
        return res.results["models"]["heat"]["quantities"]


def test_mesh_and_solve():
    """Needs a GetFEM with the 'polygon' mesher object."""
    m = model_with({"type": "polygon", "points": L_SHAPE})
    m["geometries"][0]["selections"] = {
        "notch": {"kind": "boundary", "criteria": {"type": "edges", "edges": [
            {"shape": [], "side": "side3"}, {"shape": [], "side": "side4"}]}}}
    m["models"][0]["outputs"].append({"name": "notch", "expr": "1", "selection": "notch"})
    q = _solve(m)
    assert abs(q["area"] - 0.75) < 1e-9 and abs(q["length"] - 4) < 1e-9 and abs(q["notch"] - 1) < 1e-9
    assert q["err2"] < 1e-20                              # a linear solution, P2 elements
    # regions of a polygon and a disk (conforming: fixed points on the interfaces)
    m = model_with({"type": "union", "of": [{"type": "polygon", "points": [[0, 0], [1, 0], [0, 1]]},
                                            {"type": "ball", "center": [0, 0], "radius": 0.4}]}, h=0.05)
    m["geometries"][0]["selections"] = {"tri": {"kind": "domain", "criteria": {"type": "regions",
                                                                               "regions": [[[0]]]}}}
    m["models"][0]["outputs"].append({"name": "tri", "expr": "1", "selection": "tri"})
    q = _solve(m)
    disk_out = math.pi * 0.16 / 4                          # the quarter of the disk in the triangle
    assert abs(q["tri"] - (0.5 - disk_out)) < 2e-3


def test_extruded_polygon_preview():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return
    from getfem_gui.app import scene
    U = [[0, 0], [1, 0], [1, 1], [0.6, 1], [0.6, 0.3], [0.4, 0.3], [0.4, 1], [0, 1]]
    shape = {"type": "extrusion", "z": [0, 0.3], "profile": {"type": "polygon", "points": U}}
    surf = scene.geometry_surface_3d(shape)
    a = np.asarray(surf.compute_cell_sizes().cell_data["Area"])
    assert abs(a.sum() - (2 * 0.86 + 0.3 * 5.4)) < 1e-9


def test_drawing_a_polygon_in_the_window():
    import os
    import shutil
    try:
        from PySide6.QtWidgets import QApplication, QToolButton
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    examples = Path(__file__).resolve().parents[1] / "examples"
    with tempfile.TemporaryDirectory() as tmp:
        path = shutil.copy(examples / "plate.json", tmp)
        win = MainWindow(path)
        try:
            v = win.viewer
            win.select(("geometries", 0))
            v.findChild(QToolButton, "draw:polygon").setChecked(True)
            for p in ([0.02, 0.02], [0.06, 0.02], [0.04, 0.05]):
                v.draw_begin(p)
                v.draw_end(p)
            v.draw_move([0.03, 0.04])
            assert "rubber" in v.plotter.actors and len(v._poly) == 3
            v.polygon_back()
            assert len(v._poly) == 2
            v.draw_begin([0.04, 0.05])
            v.draw_end([0.04, 0.05])
            v.draw_begin([0.02, 0.02])                                # the first vertex: closed
            v.draw_end([0.02, 0.02])
            src = win.model["geometries"][0]["source"]
            assert src["type"] == "csg" and src["shape"]["of"][1]["type"] == "polygon"
            assert len(src["shape"]["of"][1]["points"]) == 3 and "Polygon added" in win.state.text()
            assert win.errors == []
            # its handles once selected
            win.select(("geometries", 0, "source", "shape", "of", 1))
            assert len(v._handles) == 7
        finally:
            win.setWindowModified(False)
            win.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
