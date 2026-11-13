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

"""Curves drawn as polygons (open or closed, with fillets, splines and
constraints; in 3D in a plane) and parametric curves: the geometry of open
polygons, validation, the pieces and the elements (lengths), solves on
them, editing (conversions, actions, handles, dimensions), drawing in the
view (open, closed).

    python -m pytest tests/test_curves.py      (or: python tests/test_curves.py)
"""

import copy
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, geometry2d as G, manifold as F, model as M, parameters as P  # noqa: E402
from getfem_gui import runner  # noqa: E402

HOOK = {"type": "polygon", "open": True, "points": [[0, 0], [1, 0], [1, 1], [2, 1.5], [3, 1]],
        "fillets": [{"vertex": 2, "radius": 0.2}], "splines": [{"side": 3, "points": [[1.5, 1.6]]}],
        "smooth": [4], "constraints": [{"type": "horizontal", "side": 1}]}
SQUARE = {"type": "polygon", "open": False, "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
          "fillets": [{"vertex": k, "radius": 0.25} for k in (1, 2, 3, 4)]}
POLY = "geometries", 0, "source", "shape"


def model_with(src, conditions=None, order=2, h=0.05):
    m = M.default_model()
    m["parameters"] = [{"name": "R", "value": 1.0}, {"name": "a", "value": 1.0}]
    m["geometries"][0]["source"] = copy.deepcopy(src)
    m["geometries"][0]["selections"] = {}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "getfem", "h": h, "order": order}
    md = m["models"][0]
    md["modules"][0]["conditions"] = conditions if conditions is not None else [
        {"type": "temperature", "selection": "start", "value": 0},
        {"type": "temperature", "selection": "end", "value": 0}]
    md["modules"][0]["sources"] = [{"selection": "all", "value": 1}]
    md["outputs"] = [{"name": "length", "expr": "1", "selection": "all"},
                     {"name": "T_mean", "expr": "T", "selection": "all"}]
    return m


def length_of(m):
    """The length of the mesh of the curve (the integral of 1 on it)."""
    import getfem as gf
    from getfem_gui import mesher
    assert not [e for e in M.validate(m) if e.startswith(("geometries", "meshes"))], M.validate(m)
    with tempfile.TemporaryDirectory() as tmp:
        res = mesher.ensure_mesh(m, "mesh", tmp)
        assert res.ok, res.error
        msh = gf.Mesh("load", str(res.path))
        mim = gf.MeshIm(msh, gf.Integ("IM_GAUSS1D(10)"))
        return gf.asm_generic(mim, 0, "1", -1)


def solve(m):
    assert M.validate(m) == [], M.validate(m)
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"{res.error}\n{res.stderr[-2000:]}"
        return res.results["models"]["heat"]["quantities"]


def test_open_polygons():
    """The sides of an open polygon (not the closing one), no inside (the
    distance to it), its parts from the first vertex to the last."""
    sides = G.sides(HOOK)
    assert [s.side for s in sides] == ["side1", "side2", "side3", "side4", "fillet2"]
    assert G.side_names(HOOK) == ("side1", "side2", "side3", "side4", "fillet2")
    d = G.sdf(HOOK, np.array([[0.5, 0.0], [0.5, 0.5], [3.0, 1.0]]))
    assert abs(d[0]) < 1e-12 and abs(d[1] - 0.5) < 1e-9 and abs(d[2]) < 1e-12 and np.all(d >= 0)
    parts = G.outline_parts(HOOK)
    assert [p[0] for p in parts] == ["line", "arc", "line", "bezier", "bezier", "line"]
    assert parts[0][1] == [0.0, 0.0] and parts[-1][2] == [3.0, 1.0]
    assert G.incomplete({"type": "polygon", "open": True, "points": [[0, 0]]})
    assert not G.incomplete({"type": "polygon", "open": True, "points": [[0, 0], [1, 0]]})
    assert G.open_chain_problem([[0, 0], [0, 0]]) == "vertices 1 and 2 coincide"


def test_validation():
    ok = {"type": "curve", "dim": 2, "shape": HOOK}
    assert M.validate(model_with(ok)) == []
    for shape, msg in [
            (dict(HOOK, fillets=[{"vertex": 1, "radius": 0.1}]), "no fillet at an end"),
            (dict(HOOK, splines=[{"side": 5, "points": [[1, 1]]}]), "side 5 does not exist"),
            (dict(HOOK, smooth=[5]), "no smooth vertex at an end"),
            (dict(HOOK, points=[[0, 0]]), "at least two vertices"),
            (dict(HOOK, constraints=[{"type": "length", "side": 5, "value": 1}]), "side 5 does not exist"),
            (dict(HOOK, open=False, points=[[0, 0], [1, 0]], fillets=[], splines=[], smooth=[]),
             "at least three vertices")]:
        errs = M.validate(model_with({"type": "curve", "dim": 2, "shape": shape}))
        assert any(msg in e for e in errs), (msg, errs)
    csg = M.default_model()
    csg["geometries"][0]["source"] = {"type": "csg", "dim": 2, "shape": dict(SQUARE, open=True)}
    assert any("an open polygon is a curve" in e for e in M.validate(csg))
    placed = {"type": "curve", "dim": 3, "shape": HOOK, "origin": [0, 0, 0], "axes": [[1, 0, 0], [1, 0, 0]]}
    assert any("independent" in e for e in M.validate(model_with(placed)))
    for par, t, msg in [({"x": "cos(t)", "y": "foo(t)", "z": "t"}, [0, 1], "unknown function 'foo'"),
                        ({"x": "cos(t)", "y": "sin(t)", "z": "t"}, [1, 0], "t0 < t1"),
                        ({"x": "zz*t", "y": "t", "z": "t"}, [0, 1], "unknown name 'zz'")]:
        errs = M.validate(model_with({"type": "curve", "dim": 3, "parametric": par, "t": t}))
        assert any(msg in e for e in errs), (msg, errs)


def test_pieces_and_meshes():
    """The elements follow the curve: a closed square with its four fillets
    (its length exact, no end); an open polygon with an arc and a spline
    (the nodes on them); the constraints solved first."""
    sq = {"type": "curve", "dim": 2, "shape": SQUARE}
    assert F.chain_ends(sq) == [] and F.is_closed(sq)
    assert abs(length_of(model_with(sq)) - (4 * 0.5 + 2 * math.pi * 0.25)) < 1e-5      # (arcs of order 2)
    r = P.resolved(model_with({"type": "curve", "dim": 2, "shape": HOOK}))["geometries"][0]["source"]
    pieces = F.chain_pieces(r)
    X = F.chain_elements(pieces, 0.05, 2)
    arc = pieces[1]
    C, rad, *_ = F.arc_frame(arc[1], arc[2], arc[3])
    on = np.abs(np.linalg.norm(X - C[:, None, None], axis=0) - rad) < 1e-12
    assert on.sum() >= 3 and abs(rad - 0.2) < 1e-12
    assert np.allclose(X[:, 0, 0], [0, 0]) and np.allclose(X[:, -1, -1], [3, 1])
    q = solve(model_with({"type": "curve", "dim": 2, "shape": HOOK}))
    L = q["length"]
    assert abs(q["T_mean"] / L - L * L / 12) < 1e-4          # -T'' = 1, T = 0 at both ends


def test_placed_and_parametric():
    """A polygon in a plane of 3D with a height; a helix (its length); a
    closed ellipse (no end)."""
    placed = {"type": "curve", "dim": 3, "shape": {"type": "polygon", "open": True, "points": [[0, 0], [2, 0]]},
              "origin": [0, 0, 1], "axes": [[0, 1, 0], [0, 0, 1]], "height": "0.5*y"}
    assert F.chain_ends(placed) == [[0.0, 0.0, 1.0], [0.0, 2.0, 1.0]]
    q = solve(model_with(placed))
    assert abs(q["length"] - 2) < 1e-9
    helix = {"type": "curve", "dim": 3, "parametric": {"x": "R*cos(t)", "y": "R*sin(t)", "z": "a*t/(2*pi)"},
             "t": [0, "4*pi"]}
    q = solve(model_with(helix))
    assert abs(q["length"] - 2 * math.sqrt((2 * math.pi) ** 2 + 1)) < 1e-4
    vals, _ = P.values(model_with(helix))
    assert np.allclose(F.chain_ends(P.resolved(model_with(helix))["geometries"][0]["source"], vals)[1], [1, 0, 2])
    ellipse = {"type": "curve", "dim": 2, "parametric": {"x": "2*cos(t)", "y": "sin(t)"}, "t": [0, "2*pi"]}
    r = P.resolved(model_with(ellipse))["geometries"][0]["source"]
    assert F.is_closed(r, vals) and F.chain_ends(r, vals) == []
    assert abs(length_of(model_with(ellipse)) - 9.688448220547675) < 1e-4       # the perimeter of the ellipse


def test_editing():
    """A new 2D curve: an open polygon (its node in the tree, its fields);
    the actions of its vertices (no fillet at an end: extend it); the
    conversions between polygon, parametric curve and points; the dimension
    of the space (3D: the polygon in a plane)."""
    m = M.default_model()
    m["geometries"][0]["source"] = E.new_geometry_source("curve2")
    m["geometries"][0]["selections"] = {}
    assert m["geometries"][0]["source"]["shape"]["open"] is True
    node = E.find_node(E.tree(m), POLY)
    assert node is not None and node.label == "Curve: open polygon (3 vertices)"
    assert [f.key[-1] for f in E.fields(m, POLY)][:2] == ["open", "points"]
    assert E.can_draw(m, POLY) == 0 and E.can_draw(m, ("geometries", 0)) == 0
    assert [n for n, _ in E.shape_handles(m, POLY)] == ["v0", "v1", "v2", "m0", "m1", "center"]
    acts = dict((a, l) for l, a in E.handle_actions(m, POLY, "v2"))
    assert "extend:2" in acts and not any(a.startswith("fillet") for a in acts)
    E.do_action(m, POLY, "extend:2")
    E.do_action(m, POLY, "extend:0")
    assert E.get_at(m, POLY)["points"] == [[-0.5, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.5]]
    corner = copy.deepcopy(m)
    E.get_at(corner, POLY)["points"][1] = [0.0, 0.5]
    E.do_action(corner, POLY, "fillet:all")                  # every corner but the ends
    assert [f["vertex"] for f in E.get_at(corner, POLY)["fillets"]] == [2, 3, 4]
    E.do_action(m, POLY, "fillet:vertex:3")
    assert [f["vertex"] for f in E.get_at(m, POLY)["fillets"]] == [3]
    assert not [e for e in M.validate(m) if e.startswith("geometries")]
    dims = E.polygon_dimensions(m, POLY)
    assert [d["k"] for d in dims if d["kind"] == "side"] == [1, 2, 3, 4]
    gacts = [a for a, _ in E.actions(m, ("geometries", 0))]
    assert "curve:points" in gacts and "curve:parametric" in gacts and "curve:polygon" not in gacts
    E.do_action(m, ("geometries", 0), "curve:points")
    src = m["geometries"][0]["source"]
    assert "shape" not in src and src["points"][0] == [-0.5, 0.0] and sum(t is not None for t in src["through"]) == 1
    E.do_action(m, ("geometries", 0), "curve:polygon")
    assert m["geometries"][0]["source"]["shape"]["splines"][0]["side"] == 3     # the arc: a spline through it
    dim = next(f for f in E.fields(m, ("geometries", 0)) if f.op == "curve_dim")
    E.apply(m, dim, 3)
    src = m["geometries"][0]["source"]
    assert src["dim"] == 3 and src["origin"] == [0.0, 0.0, 0.0] and "shape" in src
    assert "Height" in [f.label for f in E.fields(m, ("geometries", 0))]
    assert E.can_draw(m, POLY) == 0 and E.can_draw(m, ("geometries", 0)) is None     # 3D: from its node
    E.do_action(m, ("geometries", 0), "curve:parametric")
    src = m["geometries"][0]["source"]
    assert set(src["parametric"]) == {"x", "y", "z"} and not [e for e in M.validate(m) if e.startswith("geometries")]
    E.apply(m, dim, 2)
    assert set(m["geometries"][0]["source"]["parametric"]) == {"x", "y"}
    E.do_action(m, ("geometries", 0), "curve:polygon")
    E.do_action(m, POLY, "clear_curve")
    assert E.get_at(m, POLY) == {"type": "polygon", "points": [], "open": True}
    assert E.polygon_to_enter(m, POLY)


def test_drawing():
    """In the view: the polygon tool alone for a curve; Enter: an open
    curve (from two vertices); a click on the first vertex: a closed one;
    the preview without inside."""
    try:
        from PySide6.QtWidgets import QApplication
        import pyvista  # noqa: F401
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"), GETFEM_GUI_NO_3D="",
                      GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        m = model_with({"type": "curve", "dim": 2, "shape": HOOK})
        path = Path(tmp) / "curve.json"
        M.save(m, path)
        win = MainWindow(str(path))
        try:
            v = win.viewer
            if v.plotter is None:
                return
            win.select(POLY)
            assert v.curve_mode and not v.draw_buttons["rectangle"].isVisibleTo(v.draw_bar)
            assert "fill" not in v.plotter.actors and "piece0" in v.plotter.actors
            assert "selected0" in v.plotter.actors                  # the polygon of its node
            v.draw_buttons["polygon"].setChecked(True)
            for p in ([0, 0], [1, 1]):
                v._polygon_click(p)
            v.polygon_close()
            assert E.get_at(win.model, POLY) == {"type": "polygon", "points": [[0, 0], [1, 1]], "open": True}
            assert win.state.text() == "Curve drawn (open, 2 vertices)"
            v.draw_buttons["polygon"].setChecked(True)
            for p in ([0, 0], [1, 0], [1, 1], [0, 0]):
                v._polygon_click(p)
            assert E.get_at(win.model, POLY)["open"] is False and len(E.get_at(win.model, POLY)["points"]) == 3
            win.undo()
            assert E.get_at(win.model, POLY)["open"] is True
            win.select(("geometries", 0))
            assert "fill" not in v.plotter.actors
        finally:
            win.setWindowModified(False)
            win.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
