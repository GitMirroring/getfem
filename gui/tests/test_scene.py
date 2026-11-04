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

"""3D scenes (mesh, highlighted selection, result field) rendered off screen.

Skipped when PyVista is not installed.

    python -m pytest tests/       (or: python tests/test_scene.py)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pyvista as pv
except ImportError:  # pragma: no cover
    pv = None
    if "pytest" in sys.modules:
        import pytest
        pytest.skip("PyVista is not installed", allow_module_level=True)

from getfem_gui import mesher, model as M, runner  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def plotter():
    p = pv.Plotter(off_screen=True, window_size=(400, 300))
    p.set_background("white")
    return p


def highlight_pixels(img):
    """Number of pixels with the highlight color (orange-red)."""
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return int(np.sum((r > 180) & (g < 120) & (b < 90)))


def test_mesh_and_selection():
    from getfem_gui.app import scene
    model = M.load(EXAMPLES / "plate_with_hole.json")
    with tempfile.TemporaryDirectory() as tmp:
        name = M.physics_mesh(model)["name"]
        mres = mesher.ensure_mesh(model, name, tmp)
        assert mres.ok, mres.error
        mesh = scene.read(mres.view_file())
        assert mesh.n_cells == mres.info["nb_elements"]
        p = plotter()
        scene.show_mesh(p, mesh)
        img = p.screenshot(return_img=True)
        assert img.std() > 5 and highlight_pixels(img) == 0
        for sel in mres.info["view"]["selections"]:
            scene.show_mesh(p, mesh, scene.read(mres.view_file(sel)), reset=False)
            assert highlight_pixels(p.screenshot(return_img=True)) > 20, sel
        p.close()


def test_result_field_and_deformed_shape():
    from getfem_gui.app import scene
    model = M.load(EXAMPLES / "plate_hole_tension.json")
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(model, tmp)
        assert res.ok, res.error
        ds = scene.read(Path(tmp) / res.results["vtu"])
        names = scene.field_names(ds)
        assert {"displacement", "von_mises"} <= set(names)
        p = plotter()
        scale = scene.deformation_scale(ds, "displacement")
        assert scale > 1
        scene.show_field(p, ds, "von_mises", warp="displacement", scale=scale)
        img = p.screenshot(return_img=True)
        assert img.std() > 5
        scene.show_field(p, ds, "displacement", component=0)
        assert p.scalar_bars
        p.close()


def test_preview_and_picking_in_the_window():
    """The exact 2D preview, and edges picked with clicks (simulated at world
    coordinates, with an off-screen plotter in place of the embedded one)."""
    import os
    import shutil
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        path = shutil.copy(EXAMPLES / "plate_hole_tension.json", tmp)
        win = MainWindow(path)
        try:
            assert win.viewer.available
            win.select(("geometries", 0))
            assert win.viewer._last[0] == "geometry"                   # no mesh needed
            win.select(("geometries", 0, "selections"))
            win._panel_action("add_selection")
            path_sel = win.current_path()
            assert win.viewer._pieces, "the preview of a selection is pickable"
            win.viewer.pick_world((0.0, 0.03))                        # left side, above the hole
            win.viewer.pick_world((0.007, 0.007))                     # on the hole
            sel = E_get(win, path_sel)
            assert sel["criteria"] == {"type": "edges", "edges": [
                {"shape": [0], "side": "xmin"}, {"shape": [1], "side": "circle"}]}
            win.viewer.pick_world((0.007, 0.007))                     # again: removed
            win.viewer.pick_world((0.05, 0.025))                      # inside: nothing
            assert E_get(win, path_sel)["criteria"]["edges"] == [{"shape": [0], "side": "xmin"}]
            win.undo()
            assert len(E_get(win, path_sel)["criteria"]["edges"]) == 2
            win.viewer.hover_world((0.1, 0.02))
            assert win.viewer._hover.side.side == "xmax"
            assert win.errors == []
            # a boundary condition shows its support on the preview, without mesh
            win.select(("physics", "boundary_conditions", 2))
            last = win.viewer._last
            assert last[0] == "geometry" and [p.side.side for p in last[3]] == ["xmax"]
            img = win.viewer.plotter.screenshot(return_img=True)
            assert highlight_pixels(img) > 20
            # clicking edges from the condition edits its support: a new selection,
            # since 'loaded_edge' is also used by an output
            assert win.viewer._pieces
            win.viewer.pick_world((0.05, 0.05))                       # top edge
            bc = win.model["physics"]["boundary_conditions"][2]
            assert bc["selection"] == "traction_3"
            sels = win.model["geometries"][0]["selections"]
            assert sels["traction_3"]["criteria"]["edges"] == [
                {"shape": [0], "side": "xmax"}, {"shape": [0], "side": "ymax"}]
            assert "loaded_edge" in sels and win.current_path() == ("physics", "boundary_conditions", 2)
            assert "traction_3" in win.tree.currentItem().text(0)
            win.viewer.pick_world((0.1, 0.02))                        # xmax: removed, in place
            assert sels["traction_3"]["criteria"]["edges"] == [{"shape": [0], "side": "ymax"}]
            win.undo()
            win.undo()
            assert win.model["physics"]["boundary_conditions"][2]["selection"] == "loaded_edge"
            assert "traction_3" not in win.model["geometries"][0]["selections"]
            win.select(("materials", 0))
            assert not win.viewer._pieces                              # a domain: not pickable
        finally:
            win.setWindowModified(False)
            win.close()


def test_regions_picked_in_the_window():
    """Domain selections drawn on the preview, and regions picked with clicks
    (world coordinates, off-screen plotter), for a selection and a material."""
    import os
    import shutil
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        path = shutil.copy(EXAMPLES / "plate_inclusion.json", tmp)
        win = MainWindow(path)
        try:
            # the material of the inclusion: its region is highlighted
            win.select(("materials", 1))
            last = win.viewer._last
            assert last[0] == "geometry" and callable(last[4]) and last[6] == "regions"
            img = win.viewer.plotter.screenshot(return_img=True)
            assert highlight_pixels(img) > 200
            win.viewer.hover_world((0.05, 0.05))
            assert win.viewer._hover == [[0], [1]]
            # a click in the matrix: 'inclusion' is also used by outputs, so the
            # material gets a new selection, starting from 'inclusion'
            win.viewer.pick_world((0.01, 0.01))
            sels = win.model["geometries"][0]["selections"]
            assert win.model["materials"][1]["selection"] == "copper"
            assert sels["copper"]["criteria"]["regions"] == [[[0], [1]], [[0]]]
            assert sels["inclusion"]["criteria"]["regions"] == [[[0], [1]]]
            win.viewer.pick_world((0.05, 0.05))                       # used once: edited in place
            assert sels["copper"]["criteria"]["regions"] == [[[0]]]
            win.viewer.pick_world((0.2, 0.2))                         # outside: nothing
            win.undo()
            win.undo()
            assert win.model["materials"][1]["selection"] == "inclusion"
            assert "copper" not in win.model["geometries"][0]["selections"]
            # a material on the whole domain: a new selection, from empty
            win.model["materials"][0]["selection"] = "all"
            win.select(("materials", 0))
            win.viewer.pick_world((0.05, 0.05))
            mat = win.model["materials"][0]
            assert mat["selection"] == "epoxy"
            assert win.model["geometries"][0]["selections"]["epoxy"] == {
                "kind": "domain", "criteria": {"type": "regions", "regions": [[[0], [1]]]}}
            # a domain selection replaced by picked regions
            win.select(("geometries", 0, "selections"))
            win._panel_action("add_selection")
            sp = win.current_path()
            sel = win.model["geometries"][0]["selections"][sp[3]]
            sel["kind"] = "domain"
            sel["criteria"] = {"type": "in_box", "min": [0, 0], "max": [0.05, 0.1]}
            win.update_view()
            assert callable(win.viewer._last[4]) and win.viewer._last[6] == "regions"
            win.viewer.pick_world((0.09, 0.09))
            assert win.model["geometries"][0]["selections"][sp[3]]["criteria"] == {
                "type": "regions", "regions": [[[0]]]}
            # boundary conditions keep picking edges
            win.select(("physics", "boundary_conditions", 0))
            assert win.viewer._last[6] is True and win.viewer._pieces
        finally:
            win.setWindowModified(False)
            win.close()


def test_drawing_in_the_window():
    """Drawing a disk to subtract it from a box geometry (mouse simulated at
    world coordinates, off-screen plotter)."""
    import os
    import shutil
    try:
        from PySide6.QtWidgets import QApplication, QComboBox, QToolButton
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        path = shutil.copy(EXAMPLES / "plate.json", tmp)
        win = MainWindow(path)
        try:
            v = win.viewer
            win.select(("physics",))
            assert v.draw_bar.isHidden()                               # only for a 2D geometry
            win.select(("geometries", 0))
            assert not v.draw_bar.isHidden()
            op = v.findChild(QComboBox, "draw:operation")
            op.setCurrentIndex(op.findData("difference"))
            v.findChild(QToolButton, "draw:ball").setChecked(True)
            assert v._draw_step is not None
            step = v._draw_step
            v.draw_begin((0.1 + 0.3 * step, 0.05))                     # snapped to the grid
            v.draw_move((0.12, 0.05))
            assert "rubber" in v.plotter.actors
            v.draw_end((0.12 + 0.2 * step, 0.05))
            src = win.model["geometries"][0]["source"]
            assert src["type"] == "csg" and src["shape"]["type"] == "difference"
            disk = src["shape"]["of"][1]
            assert disk["center"] == [0.1, 0.05] and abs(disk["radius"] - 0.02) < 1e-12
            assert win.errors == [] and win.current_path() == ("geometries", 0)
            assert "subtracted" in win.state.text()
            v.draw_begin((0.0, 0.0))
            v.stop_drawing()                                           # Escape
            assert v._tool is None and "rubber" not in v.plotter.actors
            win.undo()
            assert win.model["geometries"][0]["source"]["type"] == "box"
        finally:
            win.setWindowModified(False)
            win.close()


def test_handles_in_the_window():
    import os
    import shutil
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        path = shutil.copy(EXAMPLES / "plate_hole_tension.json", tmp)
        win = MainWindow(path)
        try:
            v = win.viewer
            ball = ("geometries", 0, "source", "shape", "of", 1)
            win.select(ball)
            assert dict(v._handles)["radius"] == [0.01, 0.0] and "handles" in v.plotter.actors
            assert not v.handle_press((0.05, 0.03))                     # nothing there: pans
            assert v.handle_press((0.0101, 0.0001))                     # the radius handle
            v.handle_move((0.0149, 0.0))
            assert "rubber" in v.plotter.actors
            v.handle_release((0.0149, 0.0))                             # snapped to the grid
            a = next(p for p in win.model["parameters"] if p["name"] == "a")
            assert abs(a["value"] - 0.015) < 1e-12 and "parameter a" in win.state.text()
            assert win.current_path() == ball and dict(v._handles)["radius"][0] == a["value"]
            win.undo()
            assert next(p for p in win.model["parameters"] if p["name"] == "a")["value"] == 0.01
            win.select(("geometries", 0))
            assert v._handles == []                                     # only on a shape node
        finally:
            win.setWindowModified(False)
            win.close()


def E_get(win, path):
    from getfem_gui import editing as E
    return E.get_at(win.model, path)


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok       {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAILED   {name}: {exc!r}")
    sys.exit(1 if failed else 0)
