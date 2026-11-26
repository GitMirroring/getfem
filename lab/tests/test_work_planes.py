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

"""Work planes: extrusions and revolutions in any plane (an origin and two
axes), an extrusion put on a face clicked in the 3D view (a boss or a
pocket), the section of the rest of the geometry under a profile, the
snapping to vertices.

    python -m pytest tests/       (or: python tests/test_work_planes.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, geometry3d as G3, mesher, model as M, parameters as P, runner  # noqa: E402

C = 1 / math.sqrt(2)
# a 1 x 0.5 rectangle extruded by 0.2, its plane through (1, 2, 3), x along (1, 1, 0)/sqrt(2), y along z
PLACED = {"type": "extrusion", "z": [0.0, 0.2], "origin": [1.0, 2.0, 3.0], "axes": [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
          "profile": {"type": "rectangle", "min": [0.0, 0.0], "max": [1.0, 0.5]}}


def model_with(shape, mesh=None, selections=None, outputs=()):
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "csg", "dim": 3, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = dict(selections or {})
    m["meshes"][0] = dict({"name": "mesh", "geometry": "geometry"},
                          **(mesh or {"mesher": "getfem", "h": 0.08, "order": 1}))
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": "x + 2*y + 3*z"}]
    md["outputs"] = [{"name": "volume", "expr": "1", "selection": "all"},
                     {"name": "err", "expr": "sqr(T - x - 2*y - 3*z)", "selection": "all"}] + list(outputs)
    return m


def solve(m):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"{res.error}\n{res.stderr[-2000:]}"
        return res.results["models"]["heat"]["quantities"]


def test_frame():
    O, R = G3.frame(PLACED)
    assert np.allclose(R.T @ R, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-12
    assert np.allclose(R[:, 2], [C, -C, 0])                       # e1 x e2
    assert G3.frame({k: v for k, v in PLACED.items() if k not in ("origin", "axes")}) is None
    for R0 in (R, np.eye(3), np.diag([1.0, -1.0, -1.0]), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])):
        axis, angle = G3.rotation_vector(R0)
        a = np.asarray(axis)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        Rr = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
        assert np.allclose(Rr, R0), (R0, axis, angle)
    # signed distance and bounds: those of the shape in its plane, moved
    Q = np.random.default_rng(0).uniform(-0.5, 1.5, (200, 3))
    plain = {k: v for k, v in PLACED.items() if k not in ("origin", "axes")}
    assert np.allclose(G3.sdf(PLACED, G3.to_global(PLACED, Q)), G3.sdf(plain, Q))
    lo, hi = G3.bounds(PLACED)
    assert np.allclose(lo, [1.0, 2 - 0.2 * C, 3.0]) and np.allclose(hi, [1 + C + 0.2 * C, 2 + C, 3.5])


def test_validation_and_form():
    m = model_with(PLACED)
    assert M.validate(m) == []
    for change, msg in [({"axes": [[1, 0, 0], [2, 0, 0]]}, "must not be parallel"),
                        ({"origin": [0, 0]}, "origin: 3 numbers expected"),
                        ({"axes": [[1, 0, 0]]}, "two axes")]:
        bad = model_with(dict(PLACED, **change))
        assert any(msg in e for e in M.validate(bad)), (msg, M.validate(bad))
    m["parameters"] = [{"name": "O", "value": [1.0, 2.0, 3.0]}, {"name": "d", "value": "[1, 1, 0]"}]
    m["geometries"][0]["source"]["shape"].update(origin="O", axes=["d", [0, 0, 1]])
    assert M.validate(m) == [] and P.resolved(m)["geometries"][0]["source"]["shape"]["axes"][0] == [1, 1, 0]
    path = ("geometries", 0, "source", "shape")
    keys = {f.key[len(path):]: f for f in E.fields(m, path)}
    assert E.field_value(m, keys[("placement",)]) == "custom" and ("axes", 1) in keys
    E.apply(m, keys[("placement",)], "xy")
    assert "origin" not in E.get_at(m, path) and "axes" not in E.get_at(m, path)
    keys = {f.key[len(path):]: f for f in E.fields(m, path)}
    assert E.field_value(m, keys[("placement",)]) == "xy" and ("origin",) not in keys
    E.apply(m, keys[("placement",)], "custom")
    assert E.get_at(m, path)["axes"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_meshed_in_its_plane():
    """The GetFEM mesher ('placed'): the volume, a linear solution exact, the
    faces of the placed extrusion found on the mesh."""
    top = {"kind": "boundary", "criteria": {"type": "faces", "faces": [{"shape": [], "face": "top"}]}}
    side = {"kind": "boundary", "criteria": {"type": "faces", "faces": [
        {"shape": [], "face": "side", "profile": {"shape": [], "side": "ymax"}}]}}
    m = model_with(PLACED, {"mesher": "getfem", "h": 0.05, "order": 1}, {"top": top, "side": side},
                   [{"name": "top_area", "expr": "1", "selection": "top"},
                    {"name": "side_area", "expr": "1", "selection": "side"}])
    code = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert "gf.MesherObject('placed'" in code
    q = solve(m)
    assert abs(q["volume"] - 0.1) < 5e-3 and q["err"] < 1e-16
    assert abs(q["top_area"] - 0.5) < 2e-2 and abs(q["side_area"] - 0.2) < 1e-2


def test_layers_in_its_plane():
    """In layers: exact volume and faces (prisms), the mesh in its plane."""
    top = {"kind": "boundary", "criteria": {"type": "faces", "faces": [{"shape": [], "face": "top"}]}}
    m = model_with(PLACED, {"mesher": "getfem", "h": 0.1, "order": 1, "layers": 3}, {"top": top},
                   [{"name": "top_area", "expr": "1", "selection": "top"},
                    {"name": "zmax", "expr": "z", "selection": "all"}])
    code = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert "m.transform([[" in code
    q = solve(m)
    assert abs(q["volume"] - 0.1) < 1e-12 and abs(q["top_area"] - 0.5) < 1e-12 and q["err"] < 1e-20
    assert abs(q["zmax"] / q["volume"] - 3.25) < 1e-12              # centroid z: 3 + 0.25


def test_gmsh_and_netgen_scripts():
    for mesher_name in ("gmsh", "netgen"):
        m = model_with(PLACED, {"mesher": mesher_name, "size": 0.1})
        code = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
        assert "placed in its frame" in code
        assert ("occ.rotate(" in code) if mesher_name == "gmsh" else (".Rotate(Axis((0, 0, 0), Dir(" in code)
        compile(code, "mesh.py", "exec")
    rev = dict(copy.deepcopy(PLACED), type="revolution")
    del rev["z"]
    rev["profile"] = {"type": "rectangle", "min": [0.5, 0.0], "max": [1.0, 0.5]}
    m = model_with(rev, {"mesher": "gmsh", "size": 0.1})
    assert M.validate(m) == []
    compile(mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh"), "mesh.py", "exec")


def test_extrusion_on_a_face():
    """A boss on the top face of the block with a hole (its root, a
    difference, put in a union) and a pocket (subtracted): the plane of the
    face, the section of the rest under the profile."""
    m = M.load(Path(__file__).resolve().parents[1] / "examples" / "block_with_hole_3d.json")
    root = ("geometries", 0, "source", "shape")
    assert ("pick_plane:boss" in dict(E.actions(m, root))) and ("pick_plane:pocket" in dict(E.actions(m, root)))
    lo, hi = G3.bounds(G3.shape_of(P.resolved(m)["geometries"][0]["source"]))
    top = [float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2, float(hi[2])]
    origin, axes = E.work_plane(top, [0, 0, 1])
    assert axes == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    origin, axes = E.work_plane(top, [0, -1, 0])
    assert np.allclose(np.cross(*axes), [0, -1, 0])
    prof = E.add_extrusion_on_plane(m, root, top, [0, 0, 1], 0.02, "boss")
    assert E.get_at(m, root)["type"] == "union" and prof == root + ("of", 1, "profile")
    ext = E.get_at(m, prof[:-1])
    assert ext["z"] == [0.0, 0.02] and ext["origin"] == top
    assert M.validate(m) != [] and E.get_at(m, prof) == {"type": "union", "of": []}
    E.do_action(m, prof, "add_shape:rectangle")
    rect = E.get_at(m, prof + ("of", 0))
    assert max(abs(x) for x in rect["max"]) < 0.05            # at the scale of the block, around the origin
    assert M.validate(m) == []
    shape = G3.shape_of(P.resolved(m)["geometries"][0]["source"])
    assert G3.sdf(shape, np.array([[top[0], top[1], top[2] + 0.01]]))[0] < 0      # the boss
    # a pocket: subtracted from a new difference
    p2 = E.add_extrusion_on_plane(m, root, top, [0, 0, 1], 0.02, "pocket")
    assert E.get_at(m, root)["type"] == "difference" and E.get_at(m, p2[:-1])["z"] == [-0.02, 0.0]
    E.do_action(m, p2, "add_shape:ball")
    shape = G3.shape_of(P.resolved(m)["geometries"][0]["source"])
    assert G3.sdf(shape, np.array([[top[0], top[1], top[2] - 0.005]]))[0] > 0     # cut
    assert M.validate(m) == []
    # the section of the rest of the geometry in the plane of the pocket
    r = P.resolved(m)
    rest = G3.without(r["geometries"][0]["source"]["shape"], (1,))
    elo, ehi = G3.plane_extent(rest, E.get_at(r, p2[:-1]))
    xs, ys, d = G3.section_grid(rest, E.get_at(r, p2[:-1]), elo, ehi, 60)
    assert d.min() < 0 < d.max()


def test_section_and_snapping_in_the_window():
    import os
    import shutil
    try:
        from PySide6.QtWidgets import QApplication, QPushButton
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_LAB_NO_3D="", GETFEM_LAB_OFFSCREEN_3D="1")
    from getfem_lab.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    examples = Path(__file__).resolve().parents[1] / "examples"
    with tempfile.TemporaryDirectory() as tmp:
        win = MainWindow(shutil.copy(examples / "block_with_hole_3d.json", tmp))
        try:
            v = win.viewer
            root = ("geometries", 0, "source", "shape")
            win.select(root)
            b = win.props.findChild(QPushButton, "action:pick_plane")
            assert b is not None and [a.data() for a in b.menu().actions()] == ["pick_plane:boss",
                                                                                  "pick_plane:pocket"]
            next(a for a in b.menu().actions() if a.data() == "pick_plane:boss").trigger()
            assert v._faces is not None
            surf, refs = v._faces
            side = next(i for i, r in enumerate(refs) if r.get("face") == "side")       # the hole
            v.last_face = side
            win._face_clicked(refs[side])
            assert "not plane" in win.state.text() and win.current_path() == root
            top = next(i for i, r in enumerate(refs) if r.get("face") == "zmax")
            v.last_face = top
            win._face_clicked(refs[top])
            prof = win.current_path()
            assert prof[-1] == "profile" and "reference" in v.plotter.actors
            assert "(x, y) plane" in v.status.text()
            # a corner of the section (a corner of the block) snaps a drawn point
            corner = v._snap_points[np.argmin(np.sum(v._snap_points, axis=1))]
            px = __import__("getfem_lab.app.scene", fromlist=["x"]).pixel_size(v.plotter)
            assert v._snapped([corner[0] + 2 * px, corner[1] - 2 * px]) == [float(corner[0]), float(corner[1])]
            win.undo()
            assert win.model["geometries"][0]["source"]["shape"]["type"] == "difference"
            win.select(root)
            win._panel_action("pick_plane:pocket")
            v._escape()
            assert getattr(win, "_plane_target", None) is None
        finally:
            win.setWindowModified(False)
            win.close()


def test_drawing_on_the_plane_in_3d():
    """The 3D button on a profile: the profile on its plane, in the 3D
    geometry seen from aside (translucent); a rectangle drawn there goes to
    the profile, in the coordinates of the plane; XY: flat again. For a
    revolution, the geometry in its (r, z) half plane."""
    import os
    import shutil
    try:
        from PySide6.QtWidgets import QApplication, QToolButton
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_LAB_NO_3D="", GETFEM_LAB_OFFSCREEN_3D="1")
    from getfem_lab import docshots as D
    from getfem_lab.app import scene
    from getfem_lab.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    examples = Path(__file__).resolve().parents[1] / "examples"
    with tempfile.TemporaryDirectory() as tmp:
        win = MainWindow(shutil.copy(examples / "block_with_hole_3d.json", tmp))
        try:
            v = win.viewer
            D.work_plane(win)                          # a boss with a disk on the top face
            prof = win.current_path()
            assert not v.plane_3d and "context" not in v.plotter.actors
            v._view("view_isometric")
            assert v.plane_3d and "context" in v.plotter.actors and "in 3D" in v.status.text()
            cam = np.asarray(v.plotter.camera.direction)
            assert abs(cam[2]) < 0.99                  # seen from aside
            ctx = v._context.bounds                    # the block, in the frame of its top face
            assert abs(ctx[4] + 0.05) < 1e-3 and ctx[5] > 0      # under the face: the block; above: the boss
            w, h = v.plotter.renderer.GetSize()
            for x, y in ((0.3 * w, 0.4 * h), (0.6 * w, 0.7 * h)):
                p = scene.display_to_world(v.plotter, x, y)
                assert abs(p[2]) < 1e-12               # on the plane
            v.findChild(QToolButton, "draw:rectangle").setChecked(True)
            v.draw_begin([0.011, 0.012, 0.0])
            v.draw_move([0.031, 0.029, 0.0])
            v.draw_end([0.031, 0.029, 0.0])
            rect = E.get_at(win.model, prof)["of"][-1]
            assert rect["type"] == "rectangle" and np.allclose(rect["min"], [0.01, 0.01]) \
                and np.allclose(rect["max"], [0.03, 0.03])
            assert v.plane_3d and win.current_path() == prof
            lo, hi = v._view_box()                     # the grid covers the plane around the block
            assert lo[0] < -0.1 and hi[0] > 0.1
            v._view("view_xy")
            assert not v.plane_3d and "context" not in v.plotter.actors
            assert abs(np.asarray(v.plotter.camera.direction)[2]) > 0.99
            # a revolution: the geometry in its (r, z) half plane
            win.select(("geometries",))
            win._panel_action("add_geometry:revolution")
            g = win.current_path()
            win.select(g + ("source", "shape", "profile"))
            win._panel_action("add_shape:rectangle")
            win.select(g + ("source", "shape", "profile"))
            v._view("view_isometric")
            b = v._context.bounds
            assert b[0] < 0 < b[1] and b[4] < 0 < b[5]       # around the axis, both sides of the plane
        finally:
            win.setWindowModified(False)
            win.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
