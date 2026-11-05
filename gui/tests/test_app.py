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

def test_missing_pyside_message():
    """Without PySide6, the message names the interpreter and the real error."""
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "PySide6"
        fake.mkdir()
        (fake / "__init__.py").write_text("raise ImportError('broken for the test')\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([tmp, str(EXAMPLES.parent),
                                                           os.environ.get("PYTHONPATH", "")]))
        proc = subprocess.run([sys.executable, "-m", "getfem_gui", "gui", "--no-splash"],
                              capture_output=True, text=True, env=env, cwd=tmp)
    assert proc.returncode == 1
    assert "broken for the test" in proc.stderr and sys.executable in proc.stderr


def test_tree_shows_the_model():
    win = window()
    assert not (EXAMPLES / "plate_work").exists()       # opening creates nothing
    labels = [it.text(0) for it in win._iter_items()]
    assert any("Conditions" in label for label in labels) and any(l.startswith("heat") for l in labels)
    assert any("xmin (built-in)" in label for label in labels)
    assert win.select(("models", 0, "modules", 0, "conditions", 0))
    assert win.props.editor(("models", 0, "modules", 0, "conditions", 0, "value")) is not None
    close(win)


def test_reason_of_a_missing_3d_view_stays_visible():
    win = window()
    if not win.viewer.available:
        assert "GETFEM_GUI_NO_3D" in win.log.toPlainText()
        win.select(("meshes", 0))
        text = win.viewer.placeholder.text()
        assert "3D view unavailable" in text and "not built yet" in text
    close(win)


def test_main_with_splash_screen():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMessageBox
    import getfem_gui.app as A
    from getfem_gui.app.splash import Splash
    seen = {}

    from getfem_gui.app.window import MainWindow

    def check():
        app = QApplication.instance()
        try:
            seen["splash"] = [w.isVisible() for w in app.topLevelWidgets()
                              if isinstance(w, Splash) and w not in before]
            win = next(w for w in app.topLevelWidgets() if isinstance(w, MainWindow) and w.isVisible())
            seen["version"] = win.versions.version
            original = QMessageBox.about
            QMessageBox.about = lambda parent, title, text: seen.setdefault("about", text)
            shown = QMessageBox.exec
            QMessageBox.exec = lambda box: seen.setdefault("license", box.text())
            try:
                win.about()
                win.license_info()
            finally:
                QMessageBox.about = original
                QMessageBox.exec = shown
            win.setWindowModified(False)
        finally:
            app.quit()
    app()
    hidden = {}
    before = set(QApplication.instance().topLevelWidgets())   # windows of other tests

    def early():   # the main window waits for the start-up window
        hidden["main"] = [w.isVisible() for w in QApplication.instance().topLevelWidgets()
                          if isinstance(w, MainWindow) and w not in before]
    QTimer.singleShot(1500, early)
    QTimer.singleShot(4500, check)
    assert A.main([str(EXAMPLES / "plate.json")]) == 0
    assert hidden["main"] == [False] and seen["splash"] == [False]   # closed after about 3 s
    import getfem
    assert seen["version"] == getfem.util_version()
    assert f"GetFEM GUI 0.1 on GetFEM {seen['version']}" in seen["about"]
    assert "rapid prototyping of complex multiphysics" in seen["about"]
    assert "Help ▸ License" in seen["about"]
    assert "GNU Lesser General Public License" in seen["license"] and "GCC" in seen["license"]
    assert "https://www.gnu.org/licenses/" in seen["license"]
    for w in QApplication.instance().topLevelWidgets():
        w.close()


def test_click_filter():
    """Clicks are seen at the Qt level: press and release without moving."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QWidget
    from getfem_gui.app.viewer import ClickFilter
    app()
    clicks = []
    w = QWidget()
    f = ClickFilter(lambda widget, x, y: clicks.append((x, y)))
    w.installEventFilter(f)

    def send(kind, x, y):
        QApplication.sendEvent(w, QMouseEvent(kind, QPointF(x, y), QPointF(x, y),
                                              Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                              Qt.KeyboardModifier.NoModifier))
    send(QEvent.Type.MouseButtonPress, 10, 20)
    send(QEvent.Type.MouseButtonRelease, 11, 21)          # a click
    send(QEvent.Type.MouseButtonPress, 10, 20)
    send(QEvent.Type.MouseButtonRelease, 60, 20)          # a drag (panning)
    assert clicks == [(11.0, 21.0)]

    # while drawing (or dragging a handle), the hooks take the events: the view
    # must not pan; when they do not take them, clicks work as before
    class Hooks:
        on, seen = True, []

        def active(self):
            return self.on

        def press(self, w, x, y):
            self.seen.append("press")
            return self.on

        def move(self, w, x, y):
            self.seen.append("move")
            return self.on

        def release(self, w, x, y):
            self.seen.append("release")
            return self.on
    hooks = Hooks()
    f.draw = hooks
    taken = []
    for kind, x in ((QEvent.Type.MouseButtonPress, 10), (QEvent.Type.MouseMove, 30),
                    (QEvent.Type.MouseButtonRelease, 30)):
        taken.append(f.eventFilter(w, QMouseEvent(kind, QPointF(x, 20), QPointF(x, 20),
                                                  Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                                  Qt.KeyboardModifier.NoModifier)))
    assert taken == [True, True, True] and hooks.seen == ["press", "move", "release"]
    assert clicks == [(11.0, 21.0)]                       # no click while drawing
    hooks.on = False                                      # e.g. no handle under the mouse
    send(QEvent.Type.MouseButtonPress, 10, 20)
    send(QEvent.Type.MouseButtonRelease, 10, 20)
    assert clicks == [(11.0, 21.0), (10.0, 20.0)]


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


def test_invalid_expressions_are_found_in_the_background():
    win = window()
    key = ("materials", 0, "properties", "k")
    win.select(("materials", 0))
    edit(win.props.editor(key), "1+(X(1)")
    assert M.validate(win.model) == []          # syntactically valid JSON model...
    assert wait(lambda: any("Unbalanced parenthesis" in e for e in win.errors), 60)
    assert win.problems.count() == 1            # ...but GetFEM finds the error
    assert "background" in win.props.editor(key).styleSheet()
    win.compute()
    assert not win.jobs.running and win.bottom.currentWidget() is win.problems
    edit(win.props.editor(key), "1+X(1)")
    assert wait(lambda: win.errors == [] and win._gwfl_key == win._model_key(), 60)
    assert "background" not in win.props.editor(key).styleSheet()
    close(win)


def test_parameters_and_fields_in_the_window():
    win = window("plate_hole_tension.json")
    ball = ("geometries", 0, "source", "shape", "of", 1)
    win.select(ball)
    edit(win.props.editor(ball + ("radius",)), "a/2")
    assert E.get_at(win.model, ball + ("radius",)) == "a/2" and win.errors == []
    # a parameter making the geometry invalid: the error shows on the geometry
    i = next(k for k, p in enumerate(win.model["parameters"]) if p["name"] == "a")
    win.select(("parameters", i))
    edit(win.props.editor(("parameters", i, "value")), "-0.01")
    assert any(e.startswith("geometries[0].source.shape.of[1].radius") for e in win.errors)
    win.undo()
    assert win.errors == []
    # a field with two values, then one removed with its button
    win.select(("fields",))
    button(win, "add_field").click()
    path = win.current_path()
    button(win, "add_field_value").click()
    assert len(win.model["fields"][0]["values"]) == 2
    from PySide6.QtWidgets import QToolButton
    rm = win.props.findChild(QToolButton, "remove:" + ".".join(map(str, path + ("values", 1))))
    rm.click()
    assert len(win.model["fields"][0]["values"]) == 1 and win.errors == []
    close(win)


def test_change_of_type_rebuilds_the_form():
    win = window()
    path = ("models", 0, "modules", 0, "conditions", 0)
    win.select(path)
    choose(win.props.editor(path + ("type",)), "convection")
    assert win.props.editor(path + ("h",)) is not None
    assert win.props.editor(path + ("value",)) is None
    assert "Convection" in win.tree.currentItem().text(0)
    close(win)


def test_add_and_remove_with_buttons():
    win = window()
    n = len(win.model["models"][0]["modules"][0]["conditions"])
    win.select(("models", 0, "modules", 0, "conditions"))
    button(win, "add_bc").click()
    assert len(win.model["models"][0]["modules"][0]["conditions"]) == n + 1
    assert win.current_path() == ("models", 0, "modules", 0, "conditions", n)
    button(win, "remove").click()
    assert len(win.model["models"][0]["modules"][0]["conditions"]) == n
    close(win)


def test_rename_a_selection_from_the_form():
    win = window("plate_with_hole.json")
    gi = next(i for i, g in enumerate(win.model["geometries"]) if "hole" in g["selections"])
    path = ("geometries", gi, "selections", "hole")
    win.select(path)
    edit(win.props.editor(path), "circle")
    assert win.current_path() == ("geometries", gi, "selections", "circle")
    assert any(bc["selection"] == "circle" for bc in win.model["models"][0]["modules"][0]["conditions"])
    assert win.errors == []
    close(win)


def test_compute_shows_results_and_reloads_them():
    with tempfile.TemporaryDirectory() as tmp:
        win = window("plate.json", tmp)
        win.compute()                       # after the check of the expressions
        assert wait(lambda: win.jobs.running or win.results is not None, 60)
        assert wait(lambda: win.results is not None and not win.jobs.running), win.log.toPlainText()
        assert win.results["status"] == "ok"
        assert win.current_path()[0] == E.RESULTS
        assert "T" in win.result_fields["heat"]
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
        assert win.current_path() == (E.RESULTS, "mechanics", "von_mises")
        assert {"u", "von_mises"} <= set(win.result_fields["mechanics"])
        from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox
        win.props.findChild(QCheckBox, "warp").setChecked(False)
        assert win.graphic_settings("mechanics", "von_mises")["warp"] is False
        win.select((E.RESULTS, "mechanics", "u"))
        combo = win.props.findChild(QComboBox, "component")
        choose(combo, 1)
        assert win.graphic_settings("mechanics", "u")["component"] == 1
        spin = win.props.findChild(QDoubleSpinBox, "scale")
        spin.setValue(5.0)
        spin.editingFinished.emit()
        assert win.graphic_settings("mechanics", "u")["scale"] == 5.0
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
        assert wait(lambda: win.jobs.running, 60)
        assert wait(lambda: "Solving" in win.log.toPlainText() or not win.jobs.running, 60)
        win.jobs.stop()
        assert wait(lambda: not win.jobs.running, 30)
        assert "Interrupted" in win.log.toPlainText() or win.results is not None
        close(win)


def test_export_python_script():
    with tempfile.TemporaryDirectory() as tmp:
        win = window()
        out = Path(tmp) / "script.py"
        assert win.export_script(out) and "Part 2: solve" in out.read_text()
        win.select(("materials", 0))
        edit(win.props.editor(("materials", 0, "properties", "k")), "")
        assert not win.export_script(Path(tmp) / "bad.py")        # invalid model: refused
        win.undo()
        # through the menu action, as the user does (the dialog is replaced)
        import getfem_gui.app.window as W
        chosen = Path(tmp) / "from_menu.py"
        original = W.QFileDialog.getSaveFileName
        W.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(chosen), ""))
        try:
            assert wait(lambda: win._gwfl_key == win._model_key(), 60)
            win.a_export.trigger()
        finally:
            W.QFileDialog.getSaveFileName = original
        assert chosen.exists() and "Part 1: mesh" in chosen.read_text()
        close(win)


def test_save_as_and_new():
    with tempfile.TemporaryDirectory() as tmp:
        win = window()
        win.path = Path(tmp) / "copy.json"
        assert win.save()
        assert M.load(win.path)["name"] == win.model["name"]
        win.new_model()
        assert win.path is None and win.model["models"][0]["modules"][0]["type"] == "heat_conduction"
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


def test_changing_the_mesher_from_its_combo_box():
    """The form is rebuilt from the signal of its own combo box: it must not
    be deleted while the signal runs (this crashed now and then), and the
    element size is kept."""
    win = window("plate_with_hole.json")
    i = next(k for k, m in enumerate(win.model["meshes"]) if m["mesher"] == "getfem")
    win.select(("meshes", i))
    h = win.model["meshes"][i]["h"]
    for mesher in ("gmsh", "getfem") * 5:
        combo = win.props.editor(("meshes", i, "mesher"))
        choose(combo, mesher)
        QApplication.processEvents()
        assert win.model["meshes"][i]["mesher"] == mesher
    assert win.model["meshes"][i]["h"] == h
    close(win)
