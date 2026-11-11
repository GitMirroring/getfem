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


def test_removing_shapes():
    """Any part of a boolean can be removed; a boolean left with one part is
    replaced by it; the edge and region references follow."""
    m = model_with({"type": "difference", "of": [{"type": "polygon", "points": L_SHAPE},
                                                 {"type": "ball", "center": [0.25, 0.25], "radius": 0.1}]})
    sels = m["geometries"][0]["selections"]
    sels["notch"] = {"kind": "boundary", "criteria": {"type": "edges", "edges": [
        {"shape": [0], "side": "side3"}, {"shape": [1], "side": "circle"}]}}
    sels["plate"] = {"kind": "domain", "criteria": {"type": "regions", "regions": [[[0]]]}}
    # a polygon drawn (union): the root moves under a boolean of two shapes
    E.add_shape_2d(m, 0, {"type": "polygon", "points": [[1, 0], [1.5, 0], [1.5, 0.5]]}, "union")
    assert E.get_at(m, ROOT)["type"] == "union" and M.validate(m) == []
    assert ("remove", "Remove") in E.actions(m, ROOT + ("of", 1))
    assert E.do_action(m, ROOT + ("of", 1), "remove") == ROOT
    assert E.get_at(m, ROOT)["type"] == "difference"
    assert sels["notch"]["criteria"]["edges"] == [{"shape": [0], "side": "side3"}, {"shape": [1], "side": "circle"}]
    assert sels["plate"]["criteria"]["regions"] == [[[0]]] and M.validate(m) == []
    # the hole removed: the difference becomes the polygon; its references with it
    E.do_action(m, ROOT + ("of", 1), "remove")
    assert E.get_at(m, ROOT)["type"] == "polygon"
    assert sels["notch"]["criteria"]["edges"] == [{"shape": [], "side": "side3"}]
    assert sels["plate"]["criteria"]["regions"] == [[[]]] and M.validate(m) == []
    # in a union of three, the references after the removed part shift
    m = model_with({"type": "union", "of": [{"type": "rectangle", "min": [0, 0], "max": [1, 1]},
                                            {"type": "ball", "center": [1, 0.5], "radius": 0.3},
                                            {"type": "polygon", "points": [[0, 1], [1, 1], [0.5, 1.5]]}]})
    m["geometries"][0]["selections"] = {"top": {"kind": "boundary", "criteria": {"type": "edges", "edges": [
        {"shape": [1], "side": "circle"}, {"shape": [2], "side": "side2"}]}}}
    E.do_action(m, ROOT + ("of", 1), "remove")
    assert m["geometries"][0]["selections"]["top"]["criteria"]["edges"] == [{"shape": [1], "side": "side2"}]
    assert M.validate(m) == []


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


def test_middle_and_right_clicks():
    """Drawing a polygon: a middle click puts its last vertex and closes it;
    a right click proposes to remove a vertex. A right click on a handle of
    a polygon: remove the vertex, round it (a fillet), insert a vertex in a
    side. The clicks are told from the drags (which pan and zoom the view)."""
    import os
    import shutil
    try:
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QApplication, QToolButton, QWidget
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.viewer import ClickFilter
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])
    # the filter: clicks of the middle and right buttons, not their drags
    got = []
    w = QWidget()
    f = ClickFilter(lambda *a: None, w, other=lambda wid, x, y, b: got.append((round(x), round(y), b)))
    w.installEventFilter(f)

    def send(kind, x, button):
        QApplication.sendEvent(w, QMouseEvent(kind, QPointF(x, 20), QPointF(x, 20), button, button,
                                              Qt.KeyboardModifier.NoModifier))
    for button, end in ((Qt.MouseButton.MiddleButton, 12), (Qt.MouseButton.RightButton, 11),
                        (Qt.MouseButton.MiddleButton, 60)):
        send(QEvent.Type.MouseButtonPress, 10, button)
        send(QEvent.Type.MouseButtonRelease, end, button)
    assert got == [(12, 20, Qt.MouseButton.MiddleButton), (11, 20, Qt.MouseButton.RightButton)]
    app.processEvents()
    examples = Path(__file__).resolve().parents[1] / "examples"
    with tempfile.TemporaryDirectory() as tmp:
        win = MainWindow(shutil.copy(examples / "plate.json", tmp))
        try:
            v = win.viewer
            win.select(("geometries", 0))
            v.findChild(QToolButton, "draw:polygon").setChecked(True)
            for p in ([0.02, 0.02], [0.06, 0.02], [0.07, 0.04]):
                v.draw_begin(p)
                v.draw_end(p)
            items = dict(v.context_items([0.06, 0.02]))
            assert set(items) == {"Remove this vertex (2)", "Remove the last vertex (Backspace)",
                                  "Close the polygon (Enter, middle click)", "Start the polygon again"}
            items["Remove this vertex (2)"]()
            assert v._poly == [[0.02, 0.02], [0.07, 0.04]]
            v.middle_click([0.03, 0.05])                         # the last vertex: closed
            src = win.model["geometries"][0]["source"]
            poly = ("geometries", 0, "source", "shape", "of", 1)
            assert E.get_at(win.model, poly)["points"] == [[0.02, 0.02], [0.07, 0.04], [0.03, 0.05]]
            assert src["shape"]["of"][1]["type"] == "polygon" and not v._poly
            # the menu of the handles of the polygon selected
            v.stop_drawing()
            win.select(poly)
            menu = [label for label, _ in v.context_items([0.07, 0.04])]
            assert menu == ["Round vertex 2 (fillet)"]           # three vertices: none removed
            assert E.handle_actions(win.model, poly, "m0") == [("Insert a vertex in side 1", "insert_vertex:0"),
                                                                ("Make side 1 curved (a spline)", "spline:side:1")]
            dict(v.context_items([0.07, 0.04]))["Round vertex 2 (fillet)"]()
            fl = E.get_at(win.model, poly)["fillets"]
            assert len(fl) == 1 and fl[0]["vertex"] == 2 and 0 < fl[0]["radius"] < 0.01 and win.errors == []
            win.select(poly)
            assert "f2" in dict(v._handles)
            win._handle_action("insert_vertex:0")
            assert len(E.get_at(win.model, poly)["points"]) == 4
            win.select(poly)
            assert "Remove vertex 1" in [label for label, _ in v.context_items([0.02, 0.02])]
            win._handle_action("fillet:sharp:3")                 # the fillet followed its vertex (now 3)
            assert "fillets" not in E.get_at(win.model, poly)
            win._handle_action("remove_vertex:1")
            win._handle_action("remove_vertex:1")                # a triangle: not removed, said so
            assert len(E.get_at(win.model, poly)["points"]) == 3 and "three vertices" in win.state.text()
        finally:
            win.setWindowModified(False)
            win.close()


def test_adding_a_polygon_enters_its_vertices():
    """"Add a polygon" gives an empty polygon: in a 2D geometry its vertices
    are clicked in the view; in a profile they are typed (Add a vertex)."""
    model = M.default_model()
    path = E.do_action(model, ("geometries",), "add_geometry:csg2")
    root = path + ("source", "shape")
    poly = E.do_action(model, root, "add_shape:polygon")
    assert E.get_at(model, poly) == {"type": "polygon", "points": []}
    assert E.polygon_to_enter(model, poly) and E.shape_handles(model, poly) == []
    assert any("at least three vertices" in e for e in M.validate(model))
    for _ in range(3):
        E.do_action(model, poly, "add_vertex")
    pts = E.get_at(model, poly)["points"]
    assert len(pts) == 3 and not E.polygon_to_enter(model, poly)
    assert not [e for e in M.validate(model) if e.startswith(f"geometries[{path[1]}]")], M.validate(model)
    # an incomplete polygon is nothing in the previews
    shape = {"type": "union", "of": [{"type": "rectangle", "min": [0, 0], "max": [1, 1]},
                                     {"type": "polygon", "points": [[0, 0], [2, 0]]}]}
    assert [p for p, _ in G.primitives(shape)] == [(0,)]
    lo, hi = G.bounds(shape)
    assert list(lo) == [0, 0] and list(hi) == [1, 1]
    assert G.sdf(shape, np.array([[1.5, 0.5]]))[0] > 0
    # in a profile too: it is shown in its plane, its vertices clicked
    model = M.default_model()
    path = E.do_action(model, ("geometries",), "add_geometry:extrusion")
    prof = path + ("source", "shape", "profile")
    poly = E.do_action(model, prof, "add_shape:polygon")
    assert E.polygon_to_enter(model, poly) and E.draw_root(model, poly) == prof
    assert E.can_draw(model, path + ("source", "shape")) is None          # the extrusion: 3D


def test_drawing_in_a_profile():
    """A shape drawn in the profile of a revolution: added to the profile;
    when it is wrapped in a new boolean, the faces swept by its sides follow."""
    model = M.load(Path(__file__).resolve().parents[1] / "examples" / "pulley_3d.json")
    prof = ("geometries", 0, "source", "shape", "profile")
    rect = prof + ("of", 0)
    assert E.draw_root(model, rect) == prof and E.primitive_node(model, rect)["type"] == "rectangle"
    assert len(E.shape_handles(model, rect)) == 5                       # corners and center
    new, _ = E.add_shape_2d(model, 0, {"type": "ball", "center": [0.03, 0.01], "radius": 0.002},
                            "difference", prof)
    assert new == prof + ("of", 2) and E.get_at(model, prof)["type"] == "difference"
    new, _ = E.add_shape_2d(model, 0, {"type": "rectangle", "min": [0.02, 0.0], "max": [0.03, 0.002]},
                            "union", prof)
    assert new == prof + ("of", 1) and E.get_at(model, prof + ("of", 0))["type"] == "difference"
    face = model["geometries"][0]["selections"]["bore"]["criteria"]["faces"][0]
    assert face["profile"]["shape"] == [0, 0] and face["shape"] == []
    assert M.validate(model) == []


def test_entering_a_polygon_in_the_window():
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
        path = shutil.copy(examples / "plate_hole_tension.json", tmp)
        win = MainWindow(path)
        try:
            v = win.viewer
            root = ("geometries", 0, "source", "shape")
            win.select(root)
            win._panel_action("add_shape:polygon")
            poly = win.current_path()
            assert poly == root + ("of", 2) and v._entering
            assert v.findChild(QToolButton, "draw:polygon").isChecked()
            for p in ([0.02, 0.02], [0.06, 0.02], [0.04, 0.04], [0.02, 0.02]):
                v.draw_begin(p)
                v.draw_end(p)
            pts = E.get_at(win.model, poly)["points"]
            assert len(pts) == 3 and len(win.model["geometries"][0]["source"]["shape"]["of"]) == 3
            assert not v._entering and not v.findChild(QToolButton, "draw:polygon").isChecked()
            assert "3 vertices entered" in win.state.text() and win.errors == []
            assert len(v._handles) == 7                       # its handles, once entered
            win.undo()
            assert E.get_at(win.model, poly)["points"] == []
            # in a new, empty geometry: an empty frame to draw in, then the vertices clicked
            win.select(("geometries",))
            win._panel_action("add_geometry:csg2")
            g = win.current_path()
            win.select(g + ("source", "shape"))
            assert not v.draw_bar.isHidden() and "frame" in v.plotter.actors
            win._panel_action("add_shape:polygon")
            poly = win.current_path()
            assert v._entering and "draw_status" in v.plotter.actors
            for p in ([0.2, 0.2], [0.6, 0.2], [0.4, 0.5], [0.2, 0.2]):
                v.draw_begin(p)
                v.draw_end(p)
            assert len(E.get_at(win.model, poly)["points"]) == 3
            assert not [e for e in win.errors if e.startswith(f"geometries[{g[1]}]")], win.errors
            assert "fill" in v.plotter.actors
            # in the profile of a new revolution: shown in the (r, z) plane, vertices clicked
            win.select(("geometries",))
            win._panel_action("add_geometry:revolution")
            prof = win.current_path()                              # its profile selected: the 2D view
            g = prof[:2]
            assert prof == g + ("source", "shape", "profile")
            assert not v.draw_bar.isHidden() and "(r, z)" in v.status.text()
            win._panel_action("add_shape:polygon")
            poly = win.current_path()
            assert v._entering and "grid" in v.plotter.actors
            for p in ([0.4, 0.2], [0.7, 0.2], [0.55, 0.6], [0.4, 0.2]):
                v.draw_begin(p)
                v.draw_end(p)
            assert E.get_at(win.model, poly)["points"] == [[0.4, 0.2], [0.7, 0.2], [0.55, 0.6]]
            assert not [e for e in win.errors if e.startswith(f"geometries[{g[1]}]")], win.errors
            win.select(prof)                                       # a rectangle drawn in it
            v.findChild(QToolButton, "draw:rectangle").setChecked(True)
            v.draw_begin([0.8, 0.0])
            v.draw_end([0.9, 0.3])
            assert len(E.get_at(win.model, prof)["of"]) == 2 and win.current_path() == prof
            assert not v.findChild(QToolButton, "draw:rectangle").isChecked() and v._tool is None
            # a polygon drawn: the tool released, a click then grabs a handle (not a new polygon)
            win.select(prof)
            v.findChild(QToolButton, "draw:polygon").setChecked(True)
            for p in ([0.1, 0.5], [0.3, 0.5], [0.2, 0.8], [0.1, 0.5]):
                v.draw_begin(p)
                v.draw_end(p)
            assert len(E.get_at(win.model, prof)["of"]) == 3 and v._tool is None
            assert not v.findChild(QToolButton, "draw:polygon").isChecked()
        finally:
            win.setWindowModified(False)
            win.close()


def test_grid_and_axes_while_drawing():
    """The axes and their coordinates on a 2D geometry; while drawing, the grid step set by
    -, +, typed, or back to automatic; the points snap to it."""
    import os
    import shutil
    try:
        from PySide6.QtWidgets import QApplication, QCheckBox, QToolButton
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    examples = Path(__file__).resolve().parents[1] / "examples"
    assert G.next_step(0.01) == 0.02 and G.next_step(0.02) == 0.05 and G.next_step(0.05) == 0.1
    assert G.next_step(0.01, finer=True) == 0.005 and G.next_step(0.2, finer=True) == 0.1
    assert G.major_step(0.01, 0.3) == 0.05
    with tempfile.TemporaryDirectory() as tmp:
        path = shutil.copy(examples / "plate.json", tmp)
        win = MainWindow(path)
        try:
            v = win.viewer
            win.select(("geometries", 0))
            v.findChild(QToolButton, "draw:polygon").setChecked(True)
            auto = v._draw_step
            assert "axes" in v.plotter.actors and v.grid_edit.text() == f"{auto:g}"
            assert v.grid_label.text() == "auto"
            v.findChild(QToolButton, "draw:grid_coarser").click()
            assert v._draw_step == G.next_step(auto) and v.grid_label.text() == "fixed"
            v.findChild(QToolButton, "draw:grid_finer").click()
            v.findChild(QToolButton, "draw:grid_finer").click()
            assert v._draw_step == G.next_step(auto, finer=True)
            v.grid_edit.setText("0.025")
            v._grid_typed()
            assert v._draw_step == 0.025
            v.draw_begin([0.031, 0.012])
            v.draw_end([0.031, 0.012])
            assert v._poly == [[0.025, 0.0]]                        # snapped to the grid
            v.refresh_grid()                                        # (after a zoom): kept
            assert v._draw_step == 0.025
            v.findChild(QToolButton, "draw:grid_auto").click()
            assert v._draw_step == auto and v.grid_label.text() == "auto"
            # a primitive selected: its handles snap to the grid, which is shown and set
            win.select(("geometries", 0, "source", "shape"))
            v.stop_drawing()
            ball = ("geometries", 0, "source", "shape", "of", 1)
            E.add_shape_2d(win.model, 0, {"type": "ball", "center": [0.1, 0.05], "radius": 0.02}, "union")
            win._changed(select=ball)
            assert v._handles and "grid" in v.plotter.actors and v.grid_edit.text() != ""
            v.grid_edit.setText("0.03")
            v._grid_typed()
            assert v._draw_step == 0.03
            assert v.handle_press([0.1, 0.05])
            v.handle_release([0.13, 0.08])
            assert E.get_at(win.model, ball)["center"] == [0.12, 0.09]           # snapped to 0.03
            v.findChild(QCheckBox, "draw:snap").setChecked(False)
            assert v.handle_press([0.12, 0.09])
            v.handle_release([0.1234, 0.0876])
            assert E.get_at(win.model, ball)["center"] == [0.1234, 0.0876]       # anywhere
            v.findChild(QCheckBox, "draw:snap").setChecked(True)
            v.findChild(QToolButton, "draw:grid_auto").click()
            win.select(("geometries", 0))
            v.stop_drawing()                    # the axes stay on the geometry, not the grid
            assert "axes" in v.plotter.actors and "grid" not in v.plotter.actors and v.grid_edit.text() == ""
            win.select(("models", 0))
            v.refresh_grid()
            win.select(("geometries", 0, "selections"))
            assert "axes" in v.plotter.actors
            v.show_message("nothing")
            v.refresh_grid()
            assert "axes" not in v.plotter.actors
        finally:
            win.setWindowModified(False)
            win.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
