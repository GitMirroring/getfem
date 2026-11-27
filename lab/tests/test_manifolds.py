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

"""Curves and surfaces (geometries of a lower dimension than their space):
format, meshes, solves on them, editing, preview.

    python -m pytest tests/       (or: python tests/test_manifolds.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, manifold as F, mesher, model as M, parameters as P, runner  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

CURVE = {"type": "curve", "dim": 3, "points": [[1, 0, 0], [-1, 0, 0.5], [0, 0, 2]],
         "through": [[0, 1, 0.25], None], "closed": False}
DISH = {"type": "surface", "shape": {"type": "ball", "center": [0, 0], "radius": "R"},
        "origin": [0, 0, 0], "axes": [[1, 0, 0], [0, 1, 0]], "height": "0.5*(sqr(x) + sqr(y))"}
TUBE = {"type": "swept_surface", "sweep": "revolution", "points": [[1, 0], [1, 2]], "angle": [0, 360]}
ROOF = {"type": "swept_surface", "sweep": "extrusion", "points": [[1, 0], [-1, 0]], "through": [[0, 1]],
        "z": [0, 2]}


def model_with(src, mesh=None, conditions=None, selections=None):
    m = M.default_model()
    m["parameters"] = [{"name": "R", "value": 1.0}]
    m["geometries"][0]["source"] = copy.deepcopy(src)
    m["geometries"][0]["selections"] = copy.deepcopy(selections or {})
    m["meshes"][0] = dict({"name": "mesh", "geometry": "geometry"},
                          **(mesh or {"mesher": "getfem", "h": 0.1, "order": 2}))
    md = m["models"][0]
    md["modules"][0]["conditions"] = conditions or [{"type": "temperature", "selection": "boundary",
                                                     "value": 0}]
    md["modules"][0]["sources"] = [{"selection": "all", "value": 1}]
    md["outputs"] = [{"name": "measure", "expr": "1", "selection": "all"},
                     {"name": "boundary", "expr": "1", "selection": "boundary"},
                     {"name": "T_mean", "expr": "T", "selection": "all"}]
    return m


def _solve(m):
    assert M.validate(m) == [], M.validate(m)
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
        return res.results["models"]["heat"]["quantities"], res


def test_chain():
    C, r, e1, e2, angle = F.arc_frame([1, 0, 0], [0, 1, 0], [-1, 0, 0])
    assert np.allclose(C, 0) and abs(r - 1) < 1e-12 and abs(angle - math.pi) < 1e-12
    C, r, e1, e2, angle = F.arc_frame([1, 0], [0, -1], [-1, 0])          # clockwise
    assert abs(angle - math.pi) < 1e-12 and np.allclose(e2, [0, -1])
    X = F.chain_elements(F.chain_pieces(CURVE), 0.1, 2)
    assert X.shape[:2] == (3, 3)
    assert np.allclose(X[:, 0, 0], [1, 0, 0]) and np.allclose(X[:, -1, -1], [0, 0, 2])
    C, r, *_ = F.arc_frame(*CURVE["points"][:1], CURVE["through"][0], CURVE["points"][1])
    assert np.allclose(np.linalg.norm(X[:, :, :20] - C[:, None, None], axis=0), r)   # the arc nodes
    assert F.chain_problem({"points": [[0, 0], [0, 0]]}) is not None
    assert "aligned" in F.chain_problem({"points": [[0, 0], [2, 0]], "through": [[1, 0]]})
    assert F.chain_ends(CURVE) == [[1, 0, 0], [0, 0, 2]] and F.chain_ends(dict(CURVE, closed=True)) == []
    code = F.height_code("0.5*(sqr(x) + sqr(y)) + a*sin(pi*x)", {"a": 2.0})
    assert "np.square" in code and "np.sin" in code and "2.0" in code
    x, y = np.array([0.5]), np.array([0.0])
    assert abs(eval(code, {"np": np}, {"x": x, "y": y})[0] - (0.125 + 2.0)) < 1e-12


def test_validation():
    for src in (CURVE, DISH, TUBE, ROOF):
        assert M.validate(model_with(src)) == [], src["type"]
    bad = model_with(dict(DISH, height="sqr(x) + zz"))
    assert any("height" in e for e in M.validate(bad))
    bad = model_with(dict(DISH, axes=[[1, 0, 0], [2, 0, 0]]))
    assert any("independent" in e for e in M.validate(bad))
    bad = model_with(dict(TUBE, points=[[0, 0], [1, 2]]))
    assert any("must not touch the axis" in e for e in M.validate(bad))
    bad = model_with(dict(CURVE, through=[[0, 1, 0], None, None]))
    assert any("at most one point per piece" in e for e in M.validate(bad))
    bad = model_with(CURVE, mesh={"mesher": "gmsh", "size": 0.1})
    assert any("mesher" in e for e in M.validate(bad))
    # the selections of a surface are on its flat domain (2D criteria, edges)
    ok = model_with(DISH, selections={"half": {"kind": "domain", "criteria": {"type": "in_box", "min": [0, -2],
                                                                              "max": [2, 2]}}})
    assert M.validate(ok) == []
    assert M.geometry_dim(ok, ok["geometries"][0]) == 3 and M.topological_dim(ok, ok["geometries"][0]) == 2
    table = M.selection_table(model_with(CURVE), model_with(CURVE)["geometries"][0])
    assert table["start"]["builtin"] and table["end"]["criteria"]["center"] == [0, 0, 2]


def test_curve():
    """A wire in 3D (an arc then a segment): length, ends, a solve."""
    m = model_with(CURVE, conditions=[{"type": "temperature", "selection": "start", "value": 0},
                                      {"type": "temperature", "selection": "end", "value": 0}])
    m["models"][0]["outputs"].append({"name": "start", "expr": "1", "selection": "start"})
    q, res = _solve(m)
    C, r, e1, e2, angle = F.arc_frame([1, 0, 0], [0, 1, 0.25], [-1, 0, 0.5])
    L = r * angle + math.sqrt(1 + 2.25)
    assert abs(q["measure"] - L) < 1e-6 and q["boundary"] == 2 and q["start"] == 1
    # -T'' = 1, T = 0 at both ends: mean value L^2 / 12
    assert abs(q["T_mean"] / L - L * L / 12) < 1e-6
    assert "TDIM = 1" in mesher.mesh_script(P.resolved(m), P.resolved(m)["meshes"][0], "mesh")


def test_surfaces():
    """A paraboloid dish (flat domain + height), a tube and a roof (swept)."""
    sels = {"rim": {"kind": "boundary", "criteria": {"type": "edges",
                                                    "edges": [{"shape": [], "side": "circle"}]}}}
    m = model_with(DISH, selections=sels, conditions=[{"type": "temperature", "selection": "rim", "value": 0}])
    q, _ = _solve(m)
    assert abs(q["measure"] - 2 * math.pi / 3 * (2 ** 1.5 - 1)) < 1e-5
    assert abs(q["boundary"] - 2 * math.pi) < 1e-5
    q, _ = _solve(model_with(TUBE))
    assert abs(q["measure"] - 4 * math.pi) < 1e-5 and abs(q["boundary"] - 4 * math.pi) < 1e-5
    # the tube: -lap T = 1 along the axis only (T = 0 on both circles): T = z (2 - z) / 2
    assert abs(q["T_mean"] / q["measure"] - 1 / 3) < 1e-4
    q, _ = _solve(model_with(ROOF))
    assert abs(q["measure"] - 2 * math.pi) < 1e-5 and abs(q["boundary"] - (2 * math.pi + 4)) < 1e-5
    # quadrangles of the swept surfaces; the planar domain meshed by the GetFEM mesher in triangles
    code = mesher.mesh_script(P.resolved(model_with(ROOF)), P.resolved(model_with(ROOF))["meshes"][0], "mesh")
    assert "gf.Mesh('prismatic', m, 20, 2)" in code and "remesh(m, Q, 3)" in code


def test_export():
    """A surface and a curve exported to Gmsh and Abaqus (shell and beam elements)."""
    try:
        import getfem as gf
    except ImportError:
        return
    from getfem_lab import export as X
    for src, etype, measure in [(TUBE, "S8R", 4 * math.pi), (CURVE, "B32", None)]:
        m = model_with(src)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "w"
            msh = X.export_mesh(m, "mesh", work, Path(tmp) / "x.msh", build=True)
            assert '2 1 "domain"' in msh.read_text() if src is TUBE else '1 1 "domain"' in msh.read_text()
            inp = X.export_mesh(m, "mesh", work, Path(tmp) / "x.inp").read_text()
            assert f"TYPE={etype}," in inp
            if measure:
                mm = gf.Mesh("import", "gmsh", str(msh))
                assert mm.dim() == 3
                assert abs(gf.asm("generic", gf.MeshIm(mm, 4), 0, "1", -1) / measure - 1) < 1e-3


def test_editing():
    m = model_with(CURVE)
    f = next(f for f in E.fields(m, ("geometries", 0)) if f.op == "source_type")
    assert {"curve", "surface", "swept_surface"} <= {v for v, _ in f.choices}
    dim, rows = E.chain_rows(m, 0)
    assert dim == 3 and rows[0] == (1, 0, 0, "0, 1, 0.25") and rows[1][-1] == ""
    table = next(f for f in E.fields(m, ("geometries", 0)) if f.kind == "chain_table")
    E.apply(m, E.Field(table.key[:-1] + ("through", 1), "", "text", op="chain_through"), "0.5, 0.5, 1.5")
    assert m["geometries"][0]["source"]["through"] == [[0, 1, 0.25], [0.5, 0.5, 1.5]] and M.validate(m) == []
    E.do_action(m, ("geometries", 0), "insert_point:0")             # splits the arc into two segments
    src = m["geometries"][0]["source"]
    assert len(src["points"]) == 4 and src["through"][0] is None and src["through"][2] == [0.5, 0.5, 1.5]
    E.do_action(m, ("geometries", 0), "remove_point:3")
    assert len(src["points"]) == 3 and len(src["through"]) == 2 and M.validate(m) == []
    E.apply(m, next(f for f in E.fields(m, ("geometries", 0)) if f.op == "curve_dim"), 2)
    assert src["dim"] == 2 and all(len(p) == 2 for p in src["points"])
    # a source changed into a surface: the flat domain in the tree, edited in 2D; the mesher follows
    E.apply(m, f, "surface")
    assert m["geometries"][0]["source"]["type"] == "surface" and m["meshes"][0]["mesher"] in ("gmsh", "getfem")
    node = E.find_node(E.tree(m), ("geometries", 0, "source", "shape"))
    assert node.label.startswith("Flat domain")
    assert E.shape_dim(m, ("geometries", 0, "source", "shape")) == 2
    types = next(f for f in E.fields(m, ("geometries", 0, "source", "shape")) if f.op == "shape_type").choices
    assert "polygon" in [v for v, _ in types] and "cylinder" not in [v for v, _ in types]
    keys = [f.key[-1] for f in E.fields(m, ("geometries", 0))]
    assert {"origin", "height"} <= set(keys)
    E.apply(m, f, "swept_surface")
    assert m["meshes"][0]["mesher"] == "getfem" and M.validate(m) == []


def test_preview():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return
    from getfem_lab.app import scene
    s = scene.manifold_preview(P.resolved(model_with(DISH))["geometries"][0]["source"], {"R": 1.0})
    assert abs(s.area / (2 * math.pi / 3 * (2 ** 1.5 - 1)) - 1) < 5e-3
    s = scene.manifold_preview(TUBE, {})
    assert abs(s.area / (4 * math.pi) - 1) < 2e-3
    s = scene.manifold_preview(CURVE, {})
    assert s.n_lines >= 1


def test_example_and_window():
    import os
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_LAB_NO_3D="", GETFEM_LAB_OFFSCREEN_3D="1")
    from getfem_lab.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    model = M.load(EXAMPLES / "heated_dish.json")
    assert M.validate(model) == []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.json"
        M.save(model, path)
        win = MainWindow(path)
        try:
            win.select(("geometries", 0))
            assert win.viewer._last[0] == "manifold" and "surface" in win.viewer.status.text()
            win.select(("geometries", 0, "source", "shape"))
            assert win.viewer._last[0] == "geometry"              # the flat domain, in 2D
            win.select(("models", 0, "modules", 0, "conditions", 0))
            assert win.viewer._last[0] == "geometry" and win.viewer._pieces    # edges to pick
        finally:
            win.setWindowModified(False)
            win.close()
    q, _ = _solve(model)
    assert q["heat_flow"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
