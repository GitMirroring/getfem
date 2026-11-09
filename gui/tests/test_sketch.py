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

"""Constraints on the vertices of polygons (sketch.py): the solver, the
resolution of the model (parameters), editing, handles, the view.

    python -m pytest tests/       (or: python tests/test_sketch.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, model as M, parameters as P, runner, sketch as SK  # noqa: E402

QUAD = [[0.0, 0.0], [1.0, 0.1], [1.1, 1.0], [0.1, 0.9]]
# a square of side a: horizontal and vertical sides, equal, its first vertex fixed at the origin
SQUARE = [{"type": "horizontal", "side": 1}, {"type": "vertical", "side": 2}, {"type": "horizontal", "side": 3},
          {"type": "vertical", "side": 4}, {"type": "length", "side": 1, "value": "a"},
          {"type": "equal", "side": 1, "other": 2}, {"type": "fixed", "vertex": 1, "at": [0, 0]}]
ROOT = ("geometries", 0, "source", "shape")


def model_with(shape, params=()):
    m = M.default_model()
    m["parameters"] = [{"name": "a", "value": 0.5}] + list(params)
    m["geometries"][0]["source"] = {"type": "csg", "dim": 2, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = {}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "getfem", "h": 0.05, "order": 1}
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": "x + 2*y"}]
    md["outputs"] = [{"name": "area", "expr": "1", "selection": "all"}]
    return m


def resolved_points(m, path=ROOT):
    return np.asarray(E.get_at(P.resolved(m), path)["points"], float)


def test_solver():
    sq = [dict(c, value=2.0) if "value" in c else c for c in SQUARE]
    P_ = SK.solve({"q": (np.array(QUAD), sq)})["q"]
    assert np.allclose(P_, [[0, 0], [2, 0], [2, 2], [0, 2]], atol=1e-9)
    # an equilateral triangle: an angle of 60 degrees between two equal sides
    tri = SK.solve({"t": (np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]]),
                          [{"type": "angle", "vertex": 2, "value": 60}, {"type": "equal", "side": 1, "other": 2}])})["t"]
    L = np.linalg.norm(np.roll(tri, -1, axis=0) - tri, axis=1)
    assert np.allclose(L, L[0], rtol=1e-8)
    # parallel and perpendicular sides; as little moved as possible: nothing when they already hold
    rect = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    out = SK.solve({"r": (rect, [{"type": "parallel", "side": 1, "other": 3},
                                 {"type": "perpendicular", "side": 1, "other": 2}])})["r"]
    assert np.allclose(out, rect)
    # coordinates that must not move; weights (a dragged vertex)
    out = SK.solve({"q": (np.array(QUAD), [{"type": "horizontal", "side": 1}])},
                   fixed={"q": np.array([[True, True], [False, False], [False, False], [False, False]])})["q"]
    assert np.allclose(out[0], [0, 0]) and abs(out[1, 1]) < 1e-12
    out = SK.solve({"q": (np.array(QUAD), [{"type": "horizontal", "side": 1}])},
                   weights={"q": np.array([1.0, 1e3, 1.0, 1.0])})["q"]
    assert abs(out[1, 1] - 0.1) < 1e-3 and abs(out[0, 1] - out[1, 1]) < 1e-12
    # contradictory constraints
    try:
        SK.solve({"q": (np.array(QUAD), [{"type": "length", "side": 1, "value": 1.0},
                                         {"type": "length", "side": 1, "value": 2.0}])})
        raise AssertionError("contradictory constraints accepted")
    except SK.SketchError as exc:
        assert "cannot hold" in str(exc)
    assert SK.measure({"type": "angle", "vertex": 1}, rect) == 90.0


def test_resolved_model():
    """The model as resolved: the polygon a square of side a (a parameter);
    a coordinate given by an expression stays; errors reported."""
    m = model_with({"type": "polygon", "points": QUAD, "constraints": SQUARE})
    assert M.validate(m) == []
    assert np.allclose(resolved_points(m), [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]], atol=1e-9)
    m["parameters"][0]["value"] = 0.8
    assert np.allclose(resolved_points(m)[2], [0.8, 0.8])
    assert E.get_at(m, ROOT)["points"] == QUAD               # the drawing itself is kept
    # a vertex given by an expression: not moved, the others follow
    m2 = model_with({"type": "polygon", "points": [[0, 0], [1, 0.1], ["b", 1.0], [0.1, 0.9]],
                     "constraints": [{"type": "horizontal", "side": 2}]}, [{"name": "b", "value": 1.2}])
    pts = resolved_points(m2)
    assert pts[2, 0] == 1.2 and abs(pts[1, 1] - pts[2, 1]) < 1e-12
    for bad, msg in [({"type": "length", "side": 7, "value": 1}, "side 7"),
                     ({"type": "spiral", "side": 1}, "unknown constraint"),
                     ({"type": "parallel", "side": 1, "other": 1}, "two different sides"),
                     ({"type": "length", "side": 1, "value": 1}, "cannot hold")]:
        m3 = model_with({"type": "polygon", "points": QUAD,
                         "constraints": [{"type": "length", "side": 1, "value": 2}, bad]})
        errs = M.validate(m3)
        assert any(msg in e and e.startswith("geometries[0].source.shape.constraints") for e in errs), (bad, errs)


def test_meshed():
    """The constrained square meshed by GetFEM: its area."""
    m = model_with({"type": "difference", "of": [{"type": "polygon", "points": QUAD, "constraints": SQUARE},
                                                {"type": "ball", "center": [0.25, 0.25], "radius": 0.1}]})
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"{res.error}\n{res.stderr[-2000:]}"
        q = res.results["models"]["heat"]["quantities"]
    assert abs(q["area"] - (0.25 - math.pi * 0.01)) < 2e-3


def test_editing():
    """Constraints added from the polygon (holding as drawn), their tree
    nodes and forms, removal, the vertices renumbered."""
    m = model_with({"type": "union", "of": [{"type": "polygon", "points": QUAD}]})
    poly = ROOT + ("of", 0)
    acts = dict(E.actions(m, poly))
    assert "constrain:length" in acts and "constrain:coincident" not in acts     # no other polygon
    paths = []
    for t in ("horizontal", "length", "angle", "parallel", "fixed"):
        before = resolved_points(m, poly)
        paths.append(E.do_action(m, poly, f"constrain:{t}"))
        if t in ("length", "angle", "fixed"):               # measured: holds as drawn (rounded)
            assert np.allclose(resolved_points(m, poly), before, atol=1e-3), t
    assert paths[-1] == poly + ("constraints", 4) and M.validate(m) == []
    cons = E.get_at(m, poly)["constraints"]
    assert cons[0]["type"] == "horizontal" and cons[3]["type"] == "parallel"
    labels = [n.label for n in E.tree(m).walk() if n.kind == "constraint"]
    assert labels[0] == "Side 3 horizontal" and labels[1].startswith("Length of side 1 = ")
    fields = {f.key[-1]: f for f in E.fields(m, paths[1])}
    assert set(fields) >= {"type", "side", "value"}
    E.apply(m, fields["value"], "a")
    assert abs(np.linalg.norm(np.subtract(*resolved_points(m, poly)[:2])) - 0.5) < 1e-9
    E.apply(m, fields["type"], "vertical")                   # another type: its own defaults
    assert E.get_at(m, paths[1])["type"] == "vertical" and "value" not in E.get_at(m, paths[1])
    assert ("remove", "Remove") in E.actions(m, paths[1])
    assert E.do_action(m, paths[1], "remove") == poly and len(E.get_at(m, poly)["constraints"]) == 4
    # a vertex inserted in side 1: the constraints of side 1 dropped, the numbers after it shifted
    E.get_at(m, poly)["constraints"] = [{"type": "horizontal", "side": 1}, {"type": "vertical", "side": 2},
                                        {"type": "angle", "vertex": 3, "value": 90}]
    E._vertex_inserted(m, poly, 0)
    assert E.get_at(m, poly)["constraints"] == [{"type": "vertical", "side": 3},
                                                {"type": "angle", "vertex": 4, "value": 90}]
    E._vertex_removed(m, poly, 3, 5)                          # the vertex 4: sides 3 and 4 merged
    assert E.get_at(m, poly)["constraints"] == []


def test_coincident():
    """A vertex on a vertex of another polygon: solved together; the
    reference follows the tree."""
    tri = {"type": "polygon", "points": [[1.2, 0.0], [2.0, 0.0], [1.6, 0.7]]}
    m = model_with({"type": "union", "of": [{"type": "polygon", "points": QUAD}, tri]})
    poly = ROOT + ("of", 1)
    assert "constrain:coincident" in dict(E.actions(m, poly))
    c = E.do_action(m, poly, "constrain:coincident")
    assert E.get_at(m, c) == {"type": "coincident", "vertex": 1, "shape": [0], "other": 2}
    pts = resolved_points(m, poly)
    assert np.allclose(pts[0], resolved_points(m, ROOT + ("of", 0))[1]) and M.validate(m) == []
    field = next(f for f in E.fields(m, c) if f.op == "constraint_shape")
    assert E.field_value(m, field) == "0" and [k for k, _ in field.choices] == ["0"]
    E.do_action(m, ROOT + ("of", 0), "wrap:difference")       # the other polygon moves in the tree
    assert E.get_at(m, c)["shape"] == [0, 0] and M.validate(m) == []
    assert np.allclose(resolved_points(m, poly)[0], resolved_points(m, ROOT + ("of", 0, "of", 0))[1])


def test_handles():
    """Dragging a vertex of a constrained polygon: as close as the
    constraints let it go, the others moved as little as possible; written
    to the model."""
    m = model_with({"type": "polygon", "points": QUAD,
                    "constraints": [{"type": "horizontal", "side": 1}, {"type": "vertical", "side": 2},
                                    {"type": "fixed", "vertex": 1, "at": [0, 0]}]})
    E.move_handle(m, ROOT, "v2", [1.5, 1.4])
    pts = np.asarray(E.get_at(m, ROOT)["points"], float)
    assert np.allclose(pts[0], [0, 0]) and abs(pts[1, 1]) < 1e-9 and abs(pts[1, 0] - pts[2, 0]) < 1e-9
    assert np.allclose(pts[2], [1.5, 1.4], atol=1e-3)
    assert np.allclose(resolved_points(m), pts, atol=1e-8)       # the model holds them already
    marks = dict((t, p) for p, t in E.constraint_marks(m, ROOT))
    assert set(marks) == {"H", "V", "fixed"} and np.allclose(marks["H"], (pts[0] + pts[1]) / 2)


def test_window():
    """A constraint selected: its polygon with its handles, the sides it
    refers to highlighted, the marks of the constraints in the view."""
    import os
    import shutil
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        M.save(model_with({"type": "polygon", "points": QUAD, "constraints": SQUARE}), path)
        win = MainWindow(shutil.copy(path, tmp + "/sketch.json"))
        try:
            v = win.viewer
            win.select(ROOT + ("constraints", 4))
            title, shape, pieces, highlighted, _, _ = win._preview(win.current_path())
            assert "constraint: length of side 1" in title and [p.side.side for p in highlighted] == ["side1"]
            assert any(k.startswith("marks") for k in v.plotter.actors) and len(v._handles) == 9
            win._handle_moved("v3", [0.9, 0.7])                   # its size is the parameter a: kept
            assert win.model["parameters"][0]["value"] == 0.5     # a is a parameter: the square keeps its size
            assert np.allclose(resolved_points(win.model)[2], [0.5, 0.5], atol=1e-6)
        finally:
            win.setWindowModified(False)
            win.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
