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
        name = model["models"][0]["variables"][0]["mesh"]
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
        ds = scene.read(Path(tmp) / res.results["models"]["mechanics"]["vtu"])
        names = scene.field_names(ds)
        assert {"u", "von_mises"} <= set(names)
        p = plotter()
        scale = scene.deformation_scale(ds, "u")
        assert scale > 1
        scene.show_field(p, ds, "von_mises", warp="u", scale=scale)
        img = p.screenshot(return_img=True)
        assert img.std() > 5
        scene.show_field(p, ds, "u", component=0)
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
            win.select(("models", 0, "modules", 0, "conditions", 2))
            last = win.viewer._last
            assert last[0] == "geometry" and [p.side.side for p in last[3]] == ["xmax"]
            img = win.viewer.plotter.screenshot(return_img=True)
            assert highlight_pixels(img) > 20
            # clicking edges from the condition edits its support: a new selection,
            # since 'loaded_edge' is also used by an output
            assert win.viewer._pieces
            win.viewer.pick_world((0.05, 0.05))                       # top edge
            bc = win.model["models"][0]["modules"][0]["conditions"][2]
            assert bc["selection"] == "traction_3"
            sels = win.model["geometries"][0]["selections"]
            assert sels["traction_3"]["criteria"]["edges"] == [
                {"shape": [0], "side": "xmax"}, {"shape": [0], "side": "ymax"}]
            assert "loaded_edge" in sels and win.current_path() == ("models", 0, "modules", 0, "conditions", 2)
            assert "traction_3" in win.tree.currentItem().text(0)
            win.viewer.pick_world((0.1, 0.02))                        # xmax: removed, in place
            assert sels["traction_3"]["criteria"]["edges"] == [{"shape": [0], "side": "ymax"}]
            win.undo()
            win.undo()
            assert win.model["models"][0]["modules"][0]["conditions"][2]["selection"] == "loaded_edge"
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
            win.select(("models", 0, "modules", 0, "properties", "k", 1))
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
            assert E_get(win, ("models", 0, "modules", 0, "properties", "k", 1))["selection"] == "copper"
            assert sels["copper"]["criteria"]["regions"] == [[[0], [1]], [[0]]]
            assert sels["inclusion"]["criteria"]["regions"] == [[[0], [1]]]
            win.viewer.pick_world((0.05, 0.05))                       # used once: edited in place
            assert sels["copper"]["criteria"]["regions"] == [[[0]]]
            win.viewer.pick_world((0.2, 0.2))                         # outside: nothing
            win.undo()
            win.undo()
            assert E_get(win, ("models", 0, "modules", 0, "properties", "k", 1))["selection"] == "inclusion"
            assert "copper" not in win.model["geometries"][0]["selections"]
            # a material on the whole domain: a new selection, from empty
            E_get(win, ("models", 0, "modules", 0, "properties", "k", 0))["selection"] = "all"
            win.select(("models", 0, "modules", 0, "properties", "k", 0))
            win.viewer.pick_world((0.05, 0.05))
            mat = E_get(win, ("models", 0, "modules", 0, "properties", "k", 0))
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
            win.select(("models", 0, "modules", 0, "conditions", 0))
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
            win.select(("models", 0))
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


def test_field_display_options():
    """Isolines, bands, color range, arrows, values along a cut line (drawn
    with the mouse, simulated at world coordinates) and probing, after a solve."""
    import os
    import shutil
    import time
    try:
        from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QSpinBox
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui import editing as E
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        path = shutil.copy(EXAMPLES / "plate_inclusion.json", tmp)
        win = MainWindow(path)
        try:
            win.compute()
            t0 = time.time()
            while (win.results is None or win.jobs.running) and time.time() - t0 < 120:
                app.processEvents()
                time.sleep(0.01)
            assert win.results is not None, win.log.toPlainText()
            win.select((E.RESULTS, "heat", "T"))
            p = win.props
            assert {"cmap", "isolines", "bands", "arrows", "cut", "range_min"} <= {
                w.objectName() for w in p.findChildren(object)}
            p.findChild(QSpinBox, "isolines").setValue(6)
            arrows = p.findChild(QComboBox, "arrows")
            arrows.setCurrentIndex(arrows.findData("heat_flux"))
            arrows.activated.emit(arrows.currentIndex())
            lo = p.findChild(QLineEdit, "range_min")
            lo.setText("30")
            lo.editingFinished.emit()
            s = win.view_settings[("heat", "T")]
            assert s["isolines"] == 6 and s["arrows"] == "heat_flux" and s["range"] == [30.0, None]
            plotter = win.viewer.plotter
            assert {"isolines", "arrows", "field"} <= set(plotter.actors)
            assert plotter.actors["field"].mapper.scalar_range[0] == 30
            # probe
            v = win.viewer.probe_world((0.05, 0.05, 0.0))
            assert abs(v - 50) < 0.5 and "T = " in win.viewer.probe_label.text()
            assert win.viewer.probe_world((0.5, 0.5, 0.0)) is None
            # cut line drawn with the tool
            assert win.viewer.result_bar.isVisibleTo(win.viewer)
            win.viewer.cut_button.setChecked(True)
            win.viewer.draw_begin((0.0, 0.05))
            win.viewer.draw_move((0.05, 0.05))
            win.viewer.draw_end((0.1, 0.05))
            assert win.cut_line is not None and not win.viewer.cut_button.isChecked()
            assert "cut" in plotter.actors and win.viewer.cut_panel.isVisibleTo(win.viewer)
            d, values, a, b, label = win.viewer._cut
            assert abs(values[0] - 20) < 1e-6 and abs(values[-1] - 80) < 1e-6
            assert abs(np.nanmax(values[(d > 0.03) & (d < 0.07)]) - np.nanmin(values[(d > 0.03) & (d < 0.07)])) < 1.0
            csv = Path(tmp) / "cut.csv"
            win.viewer.save_cut(csv)
            lines = csv.read_text().splitlines()
            assert lines[0] == "distance,x,y,T" and len(lines) == len(d) + 1
            # the same cut line for another field; removed with the panel button
            win.select((E.RESULTS, "heat", "heat_flux"))
            assert "cut" in plotter.actors and "magnitude" in [
                p.findChild(QComboBox, "component").itemText(i) for i in range(4)]
            assert p.findChild(QLineEdit, "cut").text() == "0, 0.05; 0.1, 0.05"
            win.viewer.cutCleared.emit()
            assert win.cut_line is None and "cut" not in plotter.actors
            assert not win.viewer.cut_panel.isVisibleTo(win.viewer)
            # typed in the form
            line = p.findChild(QLineEdit, "cut")
            line.setText("0.05, 0; 0.05, 0.1")
            line.editingFinished.emit()
            assert win.cut_line == [[0.05, 0.0], [0.05, 0.1]]
            # settings kept when computing again, the view leaves the results
            win.select(("materials", 0))
            assert not win.viewer.result_bar.isVisibleTo(win.viewer)
            assert not win.viewer.cut_panel.isVisibleTo(win.viewer)
        finally:
            win.setWindowModified(False)
            win.close()
