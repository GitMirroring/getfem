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

"""Graphical interface, driven without a screen (Qt "offscreen" platform).

Skipped when PySide6 is not installed. The 3D view is replaced by a
placeholder here (GETFEM_GUI_NO_3D); see test_scene.py for the 3D scenes.

    python -m pytest tests/       (or: python tests/test_app.py)
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GETFEM_GUI_NO_3D", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QPushButton
except ImportError:  # pragma: no cover
    QApplication = None
    if "pytest" in sys.modules:
        import pytest
        pytest.skip("PySide6 is not installed", allow_module_level=True)

from getfem_gui import editing as E, model as M  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def app():
    return QApplication.instance() or QApplication([])


def window(example="plate.json", tmp=None):
    from getfem_gui.app.window import MainWindow
    app()
    path = EXAMPLES / example
    if tmp is not None:   # a copy, so that the working folder goes to tmp
        path = Path(shutil.copy(path, Path(tmp) / example))
    win = MainWindow(path)
    win.show()
    return win


def close(win):
    win.setWindowModified(False)
    win.close()


def wait(condition, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def edit(widget: QLineEdit, text: str):
    widget.setFocus()
    widget.setText(text)
    widget.setModified(True)
    widget.editingFinished.emit()


def choose(combo: QComboBox, value):
    i = combo.findData(value)
    assert i >= 0, (value, [combo.itemData(k) for k in range(combo.count())])
    combo.setCurrentIndex(i)
    combo.activated.emit(i)


def button(win, ident):
    b = win.props.findChild(QPushButton, f"action:{ident}")
    assert b is not None, ident
    return b


# ------------------------------------------------------------------ tests

def test_tree_shows_the_model():
    win = window()
    labels = [it.text(0) for it in win._iter_items()]
    assert any("Boundary conditions" in label for label in labels)
    assert any("xmin (built-in)" in label for label in labels)
    assert win.select(("physics", "boundary_conditions", 0))
    assert win.props.editor(("physics", "boundary_conditions", 0, "value")) is not None
    close(win)


def test_edit_a_value_then_undo():
    win = window()
    key = ("materials", 0, "properties", "k")
    win.select(("materials", 0))
    old = E.get_at(win.model, key)
    edit(win.props.editor(key), "2.5")
    assert E.get_at(win.model, key) == 2.5 and win.isWindowModified()
    edit(win.props.editor(key), "2.5*(1+X(1))")
    assert E.get_at(win.model, key) == "2.5*(1+X(1))"
    win.undo()
    win.undo()
    assert E.get_at(win.model, key) == old
    win.redo()
    assert E.get_at(win.model, key) == 2.5
    close(win)


def test_errors_are_shown_on_fields_tree_and_problems():
    win = window()
    key = ("materials", 0, "properties", "k")
    win.select(("materials", 0))
    edit(win.props.editor(key), "")
    assert win.errors and win.problems.count() == len(win.errors)
    assert "background" in win.props.editor(key).styleSheet()
    item = next(it for it in win._iter_items() if it.data(0, 256) == ("materials", 0))
    assert "empty expression" in item.toolTip(0)
    # activating a problem selects the node
    win.select(())
    win._problem_activated(win.problems.item(0))
    assert win.current_path() == ("materials", 0)
    # computing is refused
    win.compute()
    assert not win.jobs.running and win.bottom.currentWidget() is win.problems
    close(win)


def test_change_of_type_rebuilds_the_form():
    win = window()
    path = ("physics", "boundary_conditions", 0)
    win.select(path)
    choose(win.props.editor(path + ("type",)), "convection")
    assert win.props.editor(path + ("h",)) is not None
    assert win.props.editor(path + ("value",)) is None
    assert "Convection" in win.tree.currentItem().text(0)
    close(win)


def test_add_and_remove_with_buttons():
    win = window()
    n = len(win.model["physics"]["boundary_conditions"])
    win.select(("physics", "boundary_conditions"))
    button(win, "add_bc").click()
    assert len(win.model["physics"]["boundary_conditions"]) == n + 1
    assert win.current_path() == ("physics", "boundary_conditions", n)
    button(win, "remove").click()
    assert len(win.model["physics"]["boundary_conditions"]) == n
    close(win)


def test_rename_a_selection_from_the_form():
    win = window("plate_with_hole.json")
    gi = next(i for i, g in enumerate(win.model["geometries"]) if "hole" in g["selections"])
    path = ("geometries", gi, "selections", "hole")
    win.select(path)
    edit(win.props.editor(path), "circle")
    assert win.current_path() == ("geometries", gi, "selections", "circle")
    assert any(bc["selection"] == "circle" for bc in win.model["physics"]["boundary_conditions"])
    assert win.errors == []
    close(win)


def test_compute_shows_results_and_reloads_them():
    with tempfile.TemporaryDirectory() as tmp:
        win = window("plate.json", tmp)
        win.compute()
        assert win.jobs.running
        assert wait(lambda: win.results is not None and not win.jobs.running), win.log.toPlainText()
        assert win.results["status"] == "ok"
        assert win.current_path()[0] == E.RESULTS
        assert "T" in win.result_fields
        win.select((E.RESULTS,))
        assert win.props.findChild(object, "results_table").rowCount() > 3
        assert "Solved" in win.log.toPlainText()
        # an edit makes the results out of date, undoing it makes them current again
        win.select(("materials", 0))
        edit(win.props.editor(("materials", 0, "properties", "k")), "3")
        assert win.results_stale
        win.undo()
        assert not win.results_stale
        # results are found again when the model is reopened
        path = win.path
        close(win)
        from getfem_gui.app.window import MainWindow
        win2 = MainWindow(path)
        assert win2.results is not None and not win2.results_stale
        close(win2)


def test_elasticity_results_and_display_settings():
    with tempfile.TemporaryDirectory() as tmp:
        win = window("plate_hole_tension.json", tmp)
        win.compute()
        assert wait(lambda: win.results is not None and not win.jobs.running), win.log.toPlainText()
        assert win.current_path() == (E.RESULTS, "von_mises")
        assert {"displacement", "von_mises"} <= set(win.result_fields)
        from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox
        win.props.findChild(QCheckBox, "warp").setChecked(False)
        assert win.view_settings["von_mises"]["warp"] is False
        win.select((E.RESULTS, "displacement"))
        combo = win.props.findChild(QComboBox, "component")
        choose(combo, 1)
        assert win.view_settings["displacement"]["component"] == 1
        spin = win.props.findChild(QDoubleSpinBox, "scale")
        spin.setValue(5.0)
        spin.editingFinished.emit()
        assert win.view_settings["displacement"]["scale"] == 5.0
        close(win)


def test_mesh_then_view_status():
    with tempfile.TemporaryDirectory() as tmp:
        win = window("plate_with_hole.json", tmp)
        win.select(("meshes", 0))
        button(win, "cmd:build_mesh").click()
        assert wait(lambda: not win.jobs.running and "elements" in win.state.text() + win.log.toPlainText())
        assert "Mesh " in win.log.toPlainText()
        close(win)


def test_stop_a_computation():
    with tempfile.TemporaryDirectory() as tmp:
        win = window("plate.json", tmp)
        win.model["meshes"][0]["divisions"] = [400, 400]   # long enough to be interrupted
        win.compute()
        assert wait(lambda: "Solving" in win.log.toPlainText() or not win.jobs.running, 60)
        win.jobs.stop()
        assert wait(lambda: not win.jobs.running, 30)
        assert "Interrupted" in win.log.toPlainText() or win.results is not None
        close(win)


def test_save_as_and_new():
    with tempfile.TemporaryDirectory() as tmp:
        win = window()
        win.path = Path(tmp) / "copy.json"
        assert win.save()
        assert M.load(win.path)["name"] == win.model["name"]
        win.new_model()
        assert win.path is None and win.model["physics"]["type"] == "heat_stationary"
        close(win)


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok       {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"FAILED   {name}: {exc!r}")
    sys.exit(1 if failed else 0)
