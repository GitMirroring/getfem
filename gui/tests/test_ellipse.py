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

"""Ellipses: geometry, format, editing (handles, drawing), mesh and solve.

    python -m pytest tests/       (or: python tests/test_ellipse.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, geometry2d as G, mesher, model as M, parameters as P, runner  # noqa: E402

ELLIPSE = {"type": "ellipse", "center": [0.2, -0.1], "radii": ["A", 0.5], "angle": 30}      # A = 1
ROOT = ("geometries", 0, "source", "shape")


def perimeter(a, b):
    t = np.linspace(0, 2 * np.pi, 200001)
    return float(np.trapezoid(np.hypot(a * np.sin(t), b * np.cos(t)), t))


def model_with(shape, h=0.08):
    m = M.default_model()
    m["parameters"] = [{"name": "A", "value": 1.0}]
    m["geometries"][0]["source"] = {"type": "csg", "dim": 2, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = {}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "getfem", "h": h, "order": 3}
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": 0.0}]
    md["outputs"] = [{"name": "area", "expr": "1", "selection": "all"},
                     {"name": "length", "expr": "1", "selection": "boundary"}]
    return m


def test_geometry():
    e = P.resolved(model_with(ELLIPSE))["geometries"][0]["source"]["shape"]
    side = G.sides(e)[0]
    assert side.kind == "ellipse" and side.closed and abs(side.length() - perimeter(1, 0.5)) < 1e-5
    P0 = side.points(np.linspace(0, 1, 7))
    assert np.allclose(G.sdf(e, P0), 0, atol=1e-12)
    N = side.normals([0.0, 0.25])
    th = math.radians(30)
    assert np.allclose(N, [[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    lo, hi = G.bounds(e)
    assert np.allclose(hi - [0.2, -0.1], [math.hypot(math.cos(th), 0.5 * math.sin(th)),
                                          math.hypot(math.sin(th), 0.5 * math.cos(th))])
    assert abs(G.sdf(e, np.array([[0.2, -0.1]]))[0] + 0.5) < 1e-12
    # an ellipse minus a disk on its boundary: one piece of the ellipse, a piece of the circle
    shape = {"type": "difference", "of": [e, {"type": "ball", "center": side.points([0.0])[0].tolist(),
                                              "radius": 0.2}]}
    pieces = G.pieces(shape)
    pe = [p for p in pieces if p.side.side == "ellipse"]
    assert len(pe) == 1 and 0 < pe[0].t0 < 0.1 and 0.9 < pe[0].t1 < 1.0
    assert G.side_label(shape, pe[0].side.ref) == "ellipse of ellipse 1"


def test_format_and_scripts():
    m = model_with(ELLIPSE)
    assert M.validate(m) == []
    bad = copy.deepcopy(m)
    E.get_at(bad, ROOT)["radii"] = [1, -0.5]
    assert any("radii" in e for e in M.validate(bad))
    flat = model_with(ELLIPSE)
    flat["geometries"][0]["source"]["dim"] = 3
    assert any("ellipse only exists in 2D" in e for e in M.validate(flat))
    script = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert f"gf.MesherObject('ellipse', [0.2, -0.1], 1.0, 0.5, {math.radians(30)!r})" in script
    gm = copy.deepcopy(m)
    E.get_at(gm, ROOT)["radii"] = [0.3, 0.6]                        # minor axis first: turned by 90 more
    gm["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "gmsh", "size": 0.1}
    code = mesher.mesh_script(P.resolved(gm), gm["meshes"][0], "mesh")
    assert "addDisk(0.2, -0.1, 0.0, 0.6, 0.3)" in code and repr(math.radians(120)) in code
    compile(code, "mesh.py", "exec")


def test_editing():
    m = model_with({"type": "difference", "of": [{"type": "rectangle", "min": [-1, -1], "max": [1, 1]},
                                                 {"type": "ball", "center": [0, 0], "radius": "A/4"}]})
    m["geometries"][0]["selections"] = {"hole": {"kind": "boundary", "criteria": {"type": "edges", "edges": [
        {"shape": [1], "side": "circle"}]}}}
    disk = ROOT + ("of", 1)
    E.apply(m, next(f for f in E.fields(m, disk) if f.op == "shape_type"), "ellipse")
    assert E.get_at(m, disk) == {"type": "ellipse", "center": [0, 0], "radii": ["A/4", "A/4"], "angle": 0.0}
    sel = m["geometries"][0]["selections"]["hole"]["criteria"]["edges"]
    assert sel == [{"shape": [1], "side": "ellipse"}] and M.validate(m) == []
    E.get_at(m, disk)["radii"] = [0.25, 0.25]
    names = dict(E.shape_handles(m, disk))
    assert names["axis1"] == [0.25, 0.0] and np.allclose(names["axis2"], [0.0, 0.25])
    E.move_handle(m, disk, "axis1", [0.3, 0.3])                     # longer and turned by 45 degrees
    e = E.get_at(m, disk)
    assert abs(e["radii"][0] - math.hypot(0.3, 0.3)) < 1e-12 and abs(e["angle"] - 45) < 1e-9
    E.move_handle(m, disk, "axis2", [-0.1, 0.1])                    # the second semi-axis, same angle
    e = E.get_at(m, disk)
    assert abs(e["radii"][1] - math.hypot(0.1, 0.1)) < 1e-12 and abs(e["angle"] - 45) < 1e-9
    E.move_handle(m, disk, "center", [0.1, 0.0])
    assert E.get_at(m, disk)["center"] == [0.1, 0.0] and M.validate(m) == []
    # back to a disk; to a polygon
    E.apply(m, next(f for f in E.fields(m, disk) if f.op == "shape_type"), "polygon")
    assert len(E.get_at(m, disk)["points"]) == 12 and len(sel) == 1
    assert len(m["geometries"][0]["selections"]["hole"]["criteria"]["edges"]) == 12 and M.validate(m) == []
    # drawn in the rectangle of two corners
    d = G.drawn_shape("ellipse", [0, 0], [2, 1])
    assert d == {"type": "ellipse", "center": [1.0, 0.5], "radii": [1.0, 0.5], "angle": 0.0}
    assert G.drawn_shape("ellipse", [0, 0], [2, 0]) is None


def _solve(m):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
        return res.results["models"]["heat"]["quantities"]


def test_mesh_and_solve():
    """Needs a GetFEM with the 'ellipse' mesher object."""
    hole = {"type": "ellipse", "center": [0.2, -0.1], "radii": [0.2, 0.3], "angle": -20}
    m = model_with({"type": "difference", "of": [ELLIPSE, hole]})
    m["geometries"][0]["selections"] = {"inner": {"kind": "boundary", "criteria": {"type": "edges", "edges": [
        {"shape": [1], "side": "ellipse"}]}}}
    m["models"][0]["outputs"].append({"name": "inner", "expr": "1", "selection": "inner"})
    q = _solve(m)
    assert abs(q["area"] - math.pi * (0.5 - 0.06)) < 5e-4
    assert abs(q["length"] - perimeter(1, 0.5) - perimeter(0.2, 0.3)) < 5e-4
    assert abs(q["inner"] - perimeter(0.2, 0.3)) < 5e-4
    # regions: an ellipse across a rectangle (fixed points along the interfaces)
    m = model_with({"type": "union", "of": [{"type": "rectangle", "min": [0, -1], "max": [2, 1]},
                                            {"type": "ellipse", "center": [0, 0], "radii": [0.5, 0.8],
                                             "angle": 0}]}, h=0.06)
    m["meshes"][0]["order"] = 1
    m["geometries"][0]["selections"] = {"left": {"kind": "domain", "criteria": {"type": "regions",
                                                                                "regions": [[[1]]]}}}
    m["models"][0]["outputs"].append({"name": "left", "expr": "1", "selection": "left"})
    q = _solve(m)
    assert abs(q["left"] - math.pi * 0.4 / 2) < 5e-3                # the half ellipse outside


def test_drawing_an_ellipse_in_the_window():
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
            v.findChild(QToolButton, "draw:ellipse").setChecked(True)
            v.draw_begin([0.02, 0.02])
            v.draw_move([0.05, 0.03])
            assert "rubber" in v.plotter.actors
            v.draw_end([0.06, 0.04])
            src = win.model["geometries"][0]["source"]
            e = src["shape"]["of"][1]
            assert e["type"] == "ellipse" and np.allclose(e["center"], [0.04, 0.03])
            assert np.allclose(e["radii"], [0.02, 0.01]) and "Ellipse added" in win.state.text()
            assert win.errors == []
            win.select(("geometries", 0, "source", "shape", "of", 1))
            assert [n for n, _ in v._handles] == ["center", "axis1", "axis2"]
        finally:
            win.setWindowModified(False)
            win.close()


def test_extruded_ellipse_preview():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return
    from getfem_gui.app import scene
    shape = {"type": "extrusion", "z": [0, 0.5], "profile": {"type": "ellipse", "center": [0, 0],
                                                             "radii": [1, 0.5], "angle": 10}}
    surf = scene.geometry_surface_3d(shape)
    a = np.asarray(surf.compute_cell_sizes().cell_data["Area"]).sum()
    assert abs(a / (2 * math.pi * 0.5 + 0.5 * perimeter(1, 0.5)) - 1) < 1e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
