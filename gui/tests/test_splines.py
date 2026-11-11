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

"""Curved sides of polygons: splines through points, smooth vertices; the
exact geometry, its sides (selections), the meshers (GetFEM's polygons with
cubic Bezier sides, Gmsh, Netgen), 3D profiles, export, editing (the table
of the vertices, the handles, the vertices inserted and removed).

    python -m pytest tests/       (or: python tests/test_splines.py)
"""

import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import (editing as E, export, geometry2d as G, geometry3d as G3, mesher,  # noqa: E402
                        model as M, parameters as P, runner, transforms as T)

# a square: its right side through two points, its top side through one,
# smooth at the vertex 3 where they meet, the vertex 1 rounded
SQUARE = {"type": "polygon", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
          "splines": [{"side": 2, "points": [[1.2, 0.35], [0.9, 0.7]]}, {"side": 3, "points": [[0.5, "yt"]]}],
          "smooth": [3], "fillets": [{"vertex": 1, "radius": 0.2}]}
ROOT = ("geometries", 0, "source", "shape")
CURVED = {"kind": "boundary", "criteria": {"type": "edges", "edges": [{"shape": [], "side": "side2"}]}}


def model_with(shape, dim=2, mesh=None, selections=None, outputs=()):
    m = M.default_model()
    m["parameters"] = [{"name": "yt", "value": 1.3}]
    m["geometries"][0]["source"] = {"type": "csg", "dim": dim, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = dict(selections or {})
    m["meshes"][0] = dict({"name": "mesh", "geometry": "geometry"},
                          **(mesh or {"mesher": "getfem", "h": 0.05, "order": 2}))
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary",
                                       "value": "x + 2*y" + (" + 3*z" if dim == 3 else "")}]
    md["outputs"] = [{"name": "area", "expr": "1", "selection": "all"}] + list(outputs)
    return m


def resolved(shape):
    return P.resolved(model_with(shape))["geometries"][0]["source"]["shape"]


def exact(shape):
    """Area of the shape (a fine polyline), length of the side 2."""
    fine = G.boundary_polyline(shape, per_span=400, per_arc=400)
    side = next(s for s in G.sides(shape) if s.side == "side2")
    return abs(G.polygon_area(fine)), side.length()


def test_geometry():
    """The splines pass through their points, tangent at the smooth vertex;
    the signed distance, the sides, the problems."""
    sh = resolved(SQUARE)
    C = G.spline_controls(sh)
    assert sorted(C) == [1, 2] and len(C[1]) == 10 and len(C[2]) == 7       # 3 and 2 cubic curves
    assert np.allclose(C[1][[0, 3, 6, 9]], [[1, 0], [1.2, 0.35], [0.9, 0.7], [1, 1]])
    d1, d2 = G.bezier_tangents(C[1], [1.0])[0], G.bezier_tangents(C[2], [0.0])[0]
    assert abs(d1[0] * d2[1] - d1[1] * d2[0]) < 1e-9 * np.linalg.norm(d1) * np.linalg.norm(d2)   # smooth at 3
    assert np.allclose(G.bezier_part(C[2], 0.25, 0.75)[[0, -1]], G.bezier_points(C[2], [0.25, 0.75]))
    X = np.random.default_rng(0).random((4000, 2)) * 1.6 - 0.2
    fine = G.boundary_polyline(sh, per_span=300, per_arc=300)
    assert np.max(np.abs(G.sdf(sh, X) - G.polygon_sdf(fine, X))) < 1e-5
    names = sorted({pc.side.side for pc in G.pieces(sh)})
    assert names == ["fillet1", "side1", "side2", "side3", "side4"]
    lo, hi = G.bounds(sh)
    assert hi[1] > 1.29 and hi[0] > 1.19
    assert G.side_label(sh, {"shape": [], "side": "side2"}) == "side 2 of polygon 1"
    assert len(G.tangent_points(sh)) == 2 + 3 + 1          # the fillet, the points, the smooth vertex
    assert G.spline_problem(dict(sh, fillets=[{"vertex": 2, "radius": 0.1}])).startswith("vertex 2: a fillet")
    assert G.spline_problem(dict(sh, smooth=[1])).startswith("vertex 1: smooth between two straight sides")
    assert "cross" in G.spline_problem(dict(sh, splines=[{"side": 2, "points": [[-0.5, 0.5]]}]))


def test_validation():
    m = model_with(SQUARE)
    assert M.validate(m) == []
    for change, msg in [({"splines": [{"side": 7, "points": []}]}, "side number from 1 to 4"),
                        ({"splines": [{"side": 2, "points": [[1, 2, 3]]}]}, "splines[0].points[0]"),
                        ({"splines": [{"side": 2, "points": []}, {"side": 2, "points": []}]}, "already a spline"),
                        ({"smooth": [9]}, "smooth: a list of vertex numbers"),
                        ({"smooth": [1, 3]}, "smooth between two straight sides")]:
        bad = copy.deepcopy(m)
        bad["geometries"][0]["source"]["shape"].update(change)
        assert any(msg in e for e in M.validate(bad)), (change, M.validate(bad))


def test_meshed_getfem():
    """The GetFEM mesher (a polygon with cubic Bezier sides): the area and
    the length of the curved side 2 (a selection)."""
    sh = resolved(SQUARE)
    area, length = exact(sh)
    m = model_with(SQUARE, selections={"curved": CURVED},
                   outputs=[{"name": "length", "expr": "1", "selection": "curved"}])
    code = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert "curved sides: cubic Bezier curves" in code
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"{res.error}\n{res.stderr[-2000:]}"
        q = res.results["models"]["heat"]["quantities"]
    assert abs(q["area"] - area) < 1e-5 and abs(q["length"] - length) < 1e-4


def test_other_meshers_and_export():
    """The Gmsh and Netgen scripts (Bezier curves), the SVG and DXF outlines."""
    for name, word in (("gmsh", "addBezier"), ("netgen", "BezierCurve")):
        for shape, dim in ((SQUARE, 2), ({"type": "extrusion", "z": [0, 0.5], "profile": SQUARE}, 3)):
            m = model_with(shape, dim, mesh={"mesher": name, "size": 0.1})
            code = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
            compile(code, "mesh.py", "exec")
            assert word in code
    sh = resolved(SQUARE)
    assert export.svg(sh).count(" C ") == 5 and export.dxf(sh).count("\nPOLYLINE\n") == 2


def test_profile_in_3d():
    """A curved profile extruded: the volume, the area of the face of the
    curved side, meshed by GetFEM (its curves as one constraint)."""
    prof = {"type": "polygon", "points": [[0.2, 0], [1, 0], [1, 1], [0.2, 1]],
            "splines": [{"side": 2, "points": [[1.2, 0.35], [0.9, 0.7]]}]}
    area, length = exact(prof)
    ref = {"shape": [], "face": "side", "profile": {"shape": [], "side": "side2"}}
    m = model_with({"type": "extrusion", "z": [0, 0.5], "profile": prof}, 3,
                   mesh={"mesher": "getfem", "h": 0.12, "order": 2},
                   selections={"curved": {"kind": "boundary", "criteria": {"type": "faces", "faces": [ref]}}},
                   outputs=[{"name": "face", "expr": "1", "selection": "curved"}])
    assert M.validate(m) == []
    spec = G3.face_spec(G3.shape_of(P.resolved(m)["geometries"][0]["source"]), ref)
    assert spec["curve"]["kind"] == "bezier"
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"{res.error}\n{res.stderr[-2000:]}"
        q = res.results["models"]["heat"]["quantities"]
    # (the meshes of the GetFEM mesher differ a little from one platform to
    # another: 0.03 % here, 0.4 % on macOS for the volume; the curves as
    # constraints for each of their parts gave 2 % on the face)
    assert abs(q["area"] / (0.5 * area) - 1) < 1e-2, (q["area"], 0.5 * area)
    assert abs(q["face"] / (0.5 * length) - 1) < 1e-2, (q["face"], 0.5 * length)


def test_transforms():
    """A polygon with a spline turned: its points too."""
    tr = {"type": "transform", "angle": 90, "center": [0, 0], "of": [copy.deepcopy(SQUARE)]}
    flat = T.flatten(resolved(tr), 2)
    poly = flat["of"][0] if "of" in flat else flat
    assert np.allclose(poly["splines"][0]["points"], [[-0.35, 1.2], [-0.7, 0.9]])


def test_editing():
    """A side made curved, its points inserted and removed, smooth vertices,
    the vertices inserted and removed, the handles."""
    m = model_with({"type": "union", "of": [{"type": "polygon", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                                            "fillets": [{"vertex": 2, "radius": 0.1}]}]})
    poly = ROOT + ("of", 0)
    E.do_action(m, poly, "spline:side:1")             # outward: below the bottom side, no more fillet at 2
    shape = E.get_at(m, poly)
    assert shape["splines"] == [{"side": 1, "points": [[0.5, -0.15]]}] and "fillets" not in shape
    assert M.validate(m) == []
    E.do_action(m, poly, "spline:insert:1:0")         # a point after the first one, on the curve
    assert len(shape["splines"][0]["points"]) == 2 and shape["splines"][0]["points"][1][0] > 0.5
    E.do_action(m, poly, "spline:remove:1:1")
    E.do_action(m, poly, "smooth:2")
    assert shape["smooth"] == [2] and M.validate(m) == []
    rows = E.polygon_rows(m, poly)
    assert [(r[0], r[1]) for r in rows] == [("vertex", 0), ("spline", 1), ("vertex", 1), ("vertex", 2),
                                           ("vertex", 3)]
    assert E.near_spline(m, poly, 1) and E.near_spline(m, poly, 2) and not E.near_spline(m, poly, 3)
    # the handles: the point of the spline (no middle of the curved side); dragged
    names = [n for n, _ in E.shape_handles(m, poly)]
    assert "s0_0" in names and "m0" not in names and "m1" in names
    acts = dict((a, label) for label, a in E.handle_actions(m, poly, "s0_0"))
    assert "spline:straight:1" in acts and "smooth:2" in dict((a, l) for l, a in E.handle_actions(m, poly, "v1"))
    E.move_handle(m, poly, "s0_0", [0.5, -0.3])
    assert shape["splines"][0]["points"][0] == [0.5, -0.3]
    E.move_handle(m, poly, "center", [0.6, 0.4])          # the whole polygon, its spline with it
    assert np.allclose(shape["splines"][0]["points"][0], [0.6, -0.4], atol=1e-12)
    # a vertex inserted in the curved side: a point of its spline; elsewhere, the sides renumbered
    E.do_action(m, poly, "insert_vertex:0")
    assert len(shape["points"]) == 4 and len(shape["splines"][0]["points"]) == 2
    E.do_action(m, poly, "insert_vertex:2")
    assert len(shape["points"]) == 5 and shape["splines"][0]["side"] == 1 and shape["smooth"] == [2]
    # the vertex 2 removed: its two sides one curve, through it
    E.do_action(m, poly, "remove_vertex:1")
    assert shape["splines"] == [{"side": 1, "points": shape["splines"][0]["points"]}]
    assert len(shape["splines"][0]["points"]) == 3 and "smooth" not in shape and M.validate(m) == []
    E.do_action(m, poly, "spline:straight:1")
    assert "splines" not in shape


def test_vertex_table():
    """The table of the vertices: the points of the splines after their
    vertex, smooth instead of a fillet at the ends of a curve; the Curved
    button."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QToolButton
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "square.json"
        M.save(model_with(SQUARE), path)
        win = MainWindow(path)
        try:
            win.select(ROOT)
            table = win.props.findChild(object, "vertices_table")
            labels = [table.verticalHeaderItem(i).text() for i in range(table.rowCount())]
            assert labels == ["1", "2", "2.1", "2.2", "3", "3.1", "4"]
            assert table.item(4, 2).text() == "smooth" and table.item(4, 2).checkState() == Qt.CheckState.Checked
            assert table.item(1, 2).checkState() == Qt.CheckState.Unchecked and table.item(0, 2).text() == "0.2"
            assert table.item(5, 1).text() == "yt"
            table.item(1, 2).setCheckState(Qt.CheckState.Checked)        # smooth at the vertex 2
            assert E.get_at(win.model, ROOT)["smooth"] == [2, 3]
            table = win.props.findChild(object, "vertices_table")
            table.setCurrentCell(6, 0)                                    # the vertex 4: its side curved
            win.props.findChild(QToolButton, "table:curve").click()
            assert [s["side"] for s in E.get_at(win.model, ROOT)["splines"]] == [2, 3, 4]
        finally:
            win.setWindowModified(False)
            win.close()


def test_constraints():
    """A constrained polygon (a square of side 2, drawn roughly): the
    points of its spline follow the side, by the similarity of its chord;
    the vertex dragged, the points written to the model with it."""
    from getfem_gui import sketch as SK
    drawn = [[0, 0], [1.9, 0.1], [2.1, 2.0], [0.1, 1.8]]
    poly = {"type": "polygon", "points": drawn, "splines": [{"side": 2, "points": [[2.4, 1.0]]}],
            "constraints": [{"type": "horizontal", "side": 1}, {"type": "vertical", "side": 2},
                            {"type": "horizontal", "side": 3}, {"type": "vertical", "side": 4},
                            {"type": "length", "side": 1, "value": 2}, {"type": "equal", "side": 1, "other": 2},
                            {"type": "fixed", "vertex": 1, "at": [0, 0]}]}
    m = model_with({"type": "union", "of": [poly]})
    assert M.validate(m) == []
    r = P.resolved(m)["geometries"][0]["source"]["shape"]["of"][0]
    assert np.allclose(r["points"], [[0, 0], [2, 0], [2, 2], [0, 2]], atol=1e-8)
    f = SK.chord_map(drawn[1], drawn[2], [2, 0], [2, 2])
    assert np.allclose(r["splines"][0]["points"], f([[2.4, 1.0]]), atol=1e-8)
    path = ROOT + ("of", 0)
    E.move_handle(m, path, "v2", [2.5, 2.5])          # a vertex of the curved side: still a square, larger
    r2 = P.resolved(m)["geometries"][0]["source"]["shape"]["of"][0]
    V = np.asarray(r2["points"], float)
    g = SK.chord_map(r["points"][1], r["points"][2], V[1], V[2])
    assert np.allclose(r2["splines"][0]["points"], g(r["splines"][0]["points"]), atol=1e-6)
    assert np.allclose(E.get_at(m, path)["splines"][0]["points"], r2["splines"][0]["points"], atol=1e-6)


def test_drag_preview():
    """A point of a spline dragged in the view: the handle follows the
    mouse, and the outline with its curves (not only the straight
    polygon)."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "square.json"
        M.save(model_with(SQUARE), path)
        win = MainWindow(path)
        try:
            win.select(ROOT)
            v = win.viewer
            if v.plotter is None:
                return
            handles = dict(v._handles)
            assert "s1_0" in handles and "m1" not in handles
            assert v.handle_press(handles["s1_0"])
            v._draw_step = 1e-3                               # (the snapping, fine)
            v.handle_move([1.4, 0.35])
            active = v.plotter.actors["handle_active"].GetMapper().GetInput()
            assert np.allclose(np.asarray(active.GetPoint(0))[:2], [1.4, 0.35])
            rubber = v.plotter.actors["rubber"].GetMapper().GetInput()
            assert rubber.GetNumberOfPoints() > 50 and 1.4 - 1e-6 < rubber.GetBounds()[1] < 1.45   # through it
            v.handle_release([1.4, 0.35])
            assert E.get_at(win.model, ROOT)["splines"][0]["points"][0] == [1.4, 0.35]
        finally:
            win.setWindowModified(False)
            win.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
