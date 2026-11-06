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

"""Faces of 3D geometries: references, 'faces' selections (format, preview,
mesh), picking with the mouse in the window.

    python -m pytest tests/       (or: python tests/test_faces3d.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, geometry3d as G3, mesher, model as M, parameters as P, runner  # noqa: E402

BLOCK = {"type": "difference", "of": [
    {"type": "rectangle", "min": [0, 0, 0], "max": [2, 1, 0.5]},
    {"type": "cylinder", "origin": [1, 0.5, -0.5], "axis": [0, 0, 1], "length": 1.5, "radius": 0.25}]}
SINK = {"type": "extrusion", "z": [0, 0.4], "profile": {"type": "union", "of": [
    {"type": "rectangle", "min": [0, 0], "max": [1, 0.2]},
    {"type": "polygon", "points": [[0.2, 0.2], [0.8, 0.2], [0.5, 0.4]]}]}}
TUBE = {"type": "revolution", "profile": {"type": "difference", "of": [
    {"type": "rectangle", "min": [0.5, 0], "max": [1, 1]},
    {"type": "ball", "center": [1, 0.5], "radius": 0.2}]}}

HOLE = {"shape": [1], "face": "side"}
TOP = {"shape": [0], "face": "zmax"}


def model_with(shape, faces, h=0.15):
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "csg", "dim": 3, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = {"picked": {"kind": "boundary",
                                                   "criteria": {"type": "faces", "faces": faces}}}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "getfem", "h": h, "order": 2}
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "picked", "value": 1.0},
                                      {"type": "convection", "selection": "boundary", "h": 1.0, "T_ext": 0.0}]
    md["outputs"] = [{"name": "picked", "expr": "1", "selection": "picked"},
                     {"name": "area", "expr": "1", "selection": "boundary"}]
    return m


def test_references():
    assert [f["face"] for f in G3.primitive_faces(BLOCK["of"][0])] == list(G3.BOX_FACES)
    faces = G3.primitive_faces(SINK)
    assert faces[:2] == [{"face": "bottom"}, {"face": "top"}] and len(faces) == 2 + 4 + 3
    assert faces[6] == {"face": "side", "profile": {"shape": [1], "side": "side1"}}
    assert G3.face_label(BLOCK, HOLE) == "side of cylinder 2"
    assert G3.face_label(SINK, dict(faces[6], shape=[])) == "side (side 1 of polygon 2) of extrusion 1"
    assert G3.face_exists(BLOCK, TOP) and not G3.face_exists(BLOCK, {"shape": [1], "face": "zmax"})
    c = G3.toggle_face({"type": "plane"}, TOP)
    assert c == {"type": "faces", "faces": [TOP]}
    c = G3.toggle_face(c, HOLE)
    assert G3.toggle_face(c, TOP) == {"type": "faces", "faces": [HOLE]}
    # validation
    m = model_with(BLOCK, [TOP, HOLE])
    assert M.validate(m) == []
    E.get_at(m, ("geometries", 0, "selections", "picked", "criteria"))["faces"].append({"shape": [3], "face": "x"})
    assert any("no such face" in e for e in M.validate(m))
    flat = M.default_model()
    flat["geometries"][0]["selections"] = {"f": {"kind": "boundary", "criteria": {"type": "faces", "faces": [TOP]}}}
    assert any("requires a 3D box or csg" in e for e in M.validate(flat))
    # the criterion is offered for 3D geometries only
    f = next(f for f in E.fields(model_with(BLOCK, [TOP]), ("geometries", 0, "selections", "picked"))
             if f.op == "criteria_type")
    assert "faces" in [v for v, _ in f.choices]
    f = next(f for f in E.fields(flat, ("geometries", 0, "selections", "f")) if f.op == "criteria_type")
    assert "faces" not in [v for v, _ in f.choices]


def test_face_tests():
    """on_face on points of the faces (as the meshing script uses it)."""
    t = 1e-9
    spec = G3.face_spec(BLOCK, HOLE)
    th = np.array([0.0, 0.1, 0.05])
    Q = np.column_stack([1 + 0.25 * np.cos(th), 0.5 + 0.25 * np.sin(th), [0.1, 0.2, 0.3]])
    n = np.array([np.cos(th[2]), np.sin(th[2]), 0.0])
    assert G3.on_face(spec, Q, n, t) and not G3.on_face(spec, Q, np.array([0, 0, 1.0]), t)
    top = G3.face_spec(BLOCK, TOP)
    assert G3.on_face(top, np.array([[0.1, 0.1, 0.5], [0.2, 0.1, 0.5], [0.1, 0.3, 0.5]]), np.array([0, 0, 1.0]), t)
    assert not G3.on_face(top, np.array([[0.1, 0.1, 0.5], [0.2, 0.1, 0.4], [0.1, 0.3, 0.5]]),
                          np.array([0, 0, 1.0]), t)
    cap = G3.face_spec(SINK, {"shape": [], "face": "top"})
    assert G3.on_face(cap, np.array([[0.5, 0.3, 0.4], [0.45, 0.25, 0.4], [0.55, 0.25, 0.4]]),
                      np.array([0, 0, 1.0]), t)
    assert not G3.on_face(cap, np.array([[0.9, 0.5, 0.4], [0.45, 0.25, 0.4], [0.55, 0.25, 0.4]]),
                          np.array([0, 0, 1.0]), t)                  # outside the profile
    slope = G3.face_spec(SINK, {"shape": [], "face": "side", "profile": {"shape": [1], "side": "side2"}})
    P0, P1 = np.array([0.8, 0.2]), np.array([0.5, 0.4])
    Q = np.array([[*(P0 + s * (P1 - P0)), z] for s, z in ((0.2, 0.1), (0.6, 0.1), (0.4, 0.3))])
    d = P1 - P0
    nrm = np.array([d[1], -d[0], 0]) / np.linalg.norm(d)
    assert G3.on_face(slope, Q, nrm, t)
    ring = G3.face_spec(TUBE, {"shape": [], "face": "side", "profile": {"shape": [1], "side": "circle"}})
    a = np.array([0.3, 1.0, 2.0])
    r, z = 1 - 0.2 * np.cos(0.5), 0.5 + 0.2 * np.sin(0.5)
    Q = np.column_stack([r * np.cos(a), r * np.sin(a), [z, z, z]])
    assert G3.on_face(ring, Q, np.array([-np.cos(1.0), -np.sin(1.0), 0.0]), 1e-9)


def test_preview_matching():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return
    from getfem_gui.app import scene
    for shape, faces, area in [
            (BLOCK, [TOP, HOLE], 2 - math.pi * 0.0625 + 2 * math.pi * 0.25 * 0.5),
            (SINK, [{"shape": [], "face": "side", "profile": {"shape": [1], "side": "side2"}},
                    {"shape": [], "face": "bottom"}], 0.4 * math.hypot(0.3, 0.2) + 0.26),
            (TUBE, [{"shape": [], "face": "side", "profile": {"shape": [1], "side": "circle"}}],
             math.pi * 0.2 * 2 * math.pi * (1 - 0.4 / math.pi))]:          # Pappus
        m = model_with(shape, faces)
        geom = m["geometries"][0]
        surf = scene.geometry_surface_3d(shape)
        C, N = np.asarray(surf.cell_centers().points), np.asarray(surf.cell_data["outward"])
        found = G3.faces_matching(shape, C, N, "picked", M.selection_table(m, geom), scene.cell_faces(surf))
        a = np.asarray(surf.compute_cell_sizes().cell_data["Area"])
        assert abs(a[found].sum() / area - 1) < 2e-3, (shape["type"], a[found].sum(), area)
    assert G3.faces_matching(BLOCK, C, N, "picked", M.selection_table(m, geom)) is None  # without tags


def test_mesh_script_and_mesh():
    """Needs a GetFEM with the extrusion, revolution and polygon mesher objects."""
    m = model_with(BLOCK, [TOP, HOLE])
    script = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert "def on_face(spec, Q, N, tol):" in script and "faces_on_faces([{'kind': 'box_face'" in script
    compile(script, "mesh.py", "exec")
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
        q = res.results["models"]["heat"]["quantities"]
    exact = 2 - math.pi * 0.0625 + 2 * math.pi * 0.25 * 0.5
    assert abs(q["picked"] / exact - 1) < 2e-3
    U = {"type": "extrusion", "z": [0, 0.4], "profile": {"type": "polygon", "points": [
        [0, 0], [1, 0], [1, 1], [0.6, 1], [0.6, 0.3], [0.4, 0.3], [0.4, 1], [0, 1]]}}
    side = lambda k: {"shape": [], "face": "side", "profile": {"shape": [], "side": f"side{k}"}}  # noqa: E731
    m = model_with(U, [side(4), side(5), {"shape": [], "face": "top"}], h=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
        q = res.results["models"]["heat"]["quantities"]
    assert abs(q["picked"] - ((0.7 + 0.2) * 0.4 + 0.86)) < 1e-9
    # a revolution: the groove (a circle of the profile turned)
    m = model_with(TUBE, [{"shape": [], "face": "side", "profile": {"shape": [1], "side": "circle"}}], h=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
        q = res.results["models"]["heat"]["quantities"]
    # half a torus surface (Pappus: length pi r, centroid at radius R - 2r/pi)
    r, R = 0.2, 1.0
    assert abs(q["picked"] / (math.pi * r * 2 * math.pi * (R - 2 * r / math.pi)) - 1) < 5e-3


def test_picking_in_the_window():
    import os
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        model = model_with(BLOCK, [TOP])
        path = Path(tmp) / "m.json"
        M.save(model, path)
        win = MainWindow(path)
        try:
            v = win.viewer

            def display(point):                 # a world point -> display coordinates
                ren = v.plotter.renderer
                ren.SetWorldPoint(*point, 1.0)
                ren.WorldToDisplay()
                x, y, _ = ren.GetDisplayPoint()
                return x, y

            def click(point):
                v.plotter.render()
                v.pick_face_display(*display(point))

            # on a condition: a new selection for it (it used the whole boundary)
            cond = ("models", 0, "modules", 0, "conditions", 1)
            win.select(cond)
            assert v._faces is not None and "click faces" in v.status.text()
            fid, ref = v.face_at_display(*display([0.3, 0.2, 0.5]))
            assert ref == {"face": "zmax", "shape": [0]}
            v.hover_face_display(*display([0.3, 0.2, 0.5]))
            assert "hover" in v.plotter.actors
            click([0.3, 0.2, 0.5])
            name = E.get_at(win.model, cond)["selection"]
            sel = win.model["geometries"][0]["selections"][name]
            assert name != "boundary" and sel["criteria"] == {"type": "faces", "faces": [ref]}
            assert "added to" in win.state.text() and win.errors == []
            click([0.3, 0.2, 0.5])                                   # again: removed
            assert sel["criteria"]["faces"] == [] and "removed from" in win.state.text()
            # on a selection node: the picked faces
            win.select(("geometries", 0, "selections", "picked"))
            assert v._faces is not None
            click([2.0, 0.5, 0.25])                                  # the xmax face
            crit = win.model["geometries"][0]["selections"]["picked"]["criteria"]
            assert [r["face"] for r in crit["faces"]] == ["zmax", "xmax"]
            assert np.any(v._last[2])                                # highlighted on the preview
            win.undo()
            assert [r["face"] for r in win.model["geometries"][0]["selections"]["picked"]["criteria"]["faces"]] \
                == ["zmax"]
            # nothing to pick on a module
            win.select(("models", 0, "modules", 0))
            assert v._faces is None
        finally:
            win.setWindowModified(False)
            win.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
