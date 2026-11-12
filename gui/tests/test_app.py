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
    from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QMenu, QPushButton
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
    assert True not in hidden["main"] and seen["splash"] == [False]   # closed after about 3 s
    import getfem
    assert seen["version"] == getfem.util_version()
    assert f"GetFEM GUI 0.1 on GetFEM {seen['version']}" in seen["about"]
    assert "rapid prototyping of complex multiphysics" in seen["about"]
    assert "Help ▸ License" in seen["about"]
    assert "GNU Lesser General Public License" in seen["license"] and "GCC" in seen["license"]
    assert "https://www.gnu.org/licenses/" in seen["license"]
    for w in QApplication.instance().topLevelWidgets():
        w.close()


def test_command_line_default():
    """No command, or a model file: the window (getfem_gui [model.json],
    installed by pip from pyproject.toml); the other commands unchanged."""
    from getfem_gui import cli
    assert cli.with_command([]) == ["gui"]
    assert cli.with_command(["m.json", "--no-splash"]) == ["gui", "m.json", "--no-splash"]
    assert cli.with_command(["run", "m.json"]) == ["run", "m.json"]
    assert cli.with_command(["--help"]) == ["--help"]
    args = cli.parser().parse_args(cli.with_command(["m.json"]))
    assert args.cmd == "gui" and args.model == "m.json"
    try:
        import tomllib
    except ImportError:                     # Python < 3.11
        return
    with open(Path(__file__).resolve().parents[1] / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)
    assert project["project"]["scripts"]["getfem_gui"] == "getfem_gui.cli:main"


def test_main_window_failing():
    """An error while the main window is built: the start-up window is
    closed and main returns (it was left on screen forever)."""
    from PySide6.QtCore import QTimer
    import getfem_gui.app as A
    from getfem_gui.app import window
    from getfem_gui.app.splash import Splash
    app().processEvents()                             # (events left by the other tests)
    before = set(QApplication.instance().topLevelWidgets())

    def failing(*args, **kwargs):
        raise RuntimeError("no window")
    stop = QTimer()
    stop.timeout.connect(QApplication.instance().quit)
    stop.start(20000)                                 # not forever, if it fails
    original, window.MainWindow = window.MainWindow, failing
    try:
        assert A.main([str(EXAMPLES / "plate.json")]) == 1
    finally:
        window.MainWindow = original
        stop.stop()
    assert not [w for w in QApplication.instance().topLevelWidgets()
                if isinstance(w, Splash) and w not in before and w.isVisible()]


_QTPLOTTER_CHECK = """
import pyvista as pv
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from getfem_gui.app.qtplotter import BasePlotter, QtPlotter
app = QApplication([])
w = QtPlotter()
assert isinstance(w, BasePlotter) and w.interactor is w and w.iren is not None
w.resize(400, 300)
w.add_mesh(pv.Sphere(), name="sphere")
w.view_isometric()
renders, render = [], BasePlotter.render
BasePlotter.render = lambda self: (renders.append(1), render(self))[1]
for _ in range(3):
    w.render()
app.processEvents()
assert "sphere" in w.actors and renders == []     # not shown: no rendering (it blocks on macOS)
w.show()
app.processEvents()
def count(n):
    renders.clear()
    for _ in range(n):
        w.render()
    assert renders == []                          # from the event loop,
    app.processEvents()
    return len(renders)
assert count(1) == 1 and count(3) == 1           # once for all
BasePlotter.render = render
# macOS: paint events render only after a change (Qt >= 6.10 repaints the
# widget after each rendering: an endless loop otherwise)
paints = []
w.ren_win.AddObserver("StartEvent", lambda *a: paints.append(1))
w.paint_only_when_changed = True
w.repaint()
assert paints == []                               # unchanged: kept as drawn
w.render()
w.repaint()                                       # changed: rendered once,
w.repaint()
assert len(paints) == 1, paints
app.processEvents()                               # and not again by render()
assert len(paints) == 1, paints
w.resize(420, 300)
w.repaint()
assert len(paints) == 2, paints                   # resized: rendered
cam = w.camera_position

def send(kind, x, buttons):
    QApplication.sendEvent(w, QMouseEvent(kind, QPointF(x, 150), QPointF(x, 150), Qt.MouseButton.LeftButton,
                                          buttons, Qt.KeyboardModifier.NoModifier))
    app.processEvents()
send(QEvent.Type.MouseButtonPress, 150, Qt.MouseButton.LeftButton)
for x in range(160, 260, 20):
    send(QEvent.Type.MouseMove, x, Qt.MouseButton.LeftButton)
send(QEvent.Type.MouseButtonRelease, 260, Qt.MouseButton.NoButton)
assert w.camera_position != cam                   # turned by VTK's trackball style
w.close()
w.render()                                        # closed: nothing happens
app.processEvents()
print("QTPLOTTER OK")
"""


def test_qtplotter_without_pyvistaqt():
    """The 3D view is VTK's own Qt widget made a PyVista plotter: no
    pyvistaqt anywhere; the widget is a plotter (meshes, camera) and its
    mouse events go to VTK's interactor. Run apart, with the Qt offscreen
    platform and without X display: VTK must then render without window
    (EGL or OSMesa); otherwise (X, macOS, Windows) the check is skipped. It
    renders from the event loop, once for several requests, and not before
    it is shown (that blocks on macOS)."""
    import re
    import subprocess
    src = Path(__file__).resolve().parents[1] / "getfem_gui"
    imports = [p.name for p in src.rglob("*.py")
               if re.search(r"^\s*(from|import) pyvistaqt", p.read_text(encoding="utf-8"), re.M)]
    assert not imports, imports
    try:
        import pyvista  # noqa: F401
        import PySide6  # noqa: F401
    except ImportError:
        return
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env.update(QT_QPA_PLATFORM="offscreen", PYTHONPATH=os.pathsep.join(
        [str(src.parent)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    kind = subprocess.run([sys.executable, "-c", "import vtk; print(vtk.vtkRenderWindow().GetClassName())"],
                          env=env, capture_output=True, text=True).stdout.strip()
    if not any(k in kind for k in ("EGL", "OSMesa")):   # X, Cocoa, Win32: a real window needed
        return
    r = subprocess.run([sys.executable, "-c", _QTPLOTTER_CHECK], env=env, capture_output=True, text=True,
                       timeout=120)
    assert "QTPLOTTER OK" in r.stdout, r.stderr[-3000:]


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
    win.select(("parameters",))
    table = win.props.editor(("parameters",))
    assert table.rowCount() == len(win.model["parameters"]) and table.item(i, 0).text() == "a"
    assert table.item(2, 2).text() == "0.01"                        # the value of the expression
    table.item(i, 1).setText("-0.01")
    assert win.model["parameters"][i]["value"] == -0.01
    assert any(e.startswith("geometries[0].source.shape.of[1].radius") for e in win.errors)
    win.undo()
    assert win.errors == []
    # renamed in the table: its uses follow; moved, removed
    table = win.props.editor(("parameters",))
    table.item(i, 0).setText("radius")
    assert E.get_at(win.model, ball + ("radius",)) == "radius/2"
    table = win.props.editor(("parameters",))
    assert table.currentRow() == i                                  # the edited cell keeps the focus
    n = len(win.model["parameters"])
    win._panel_action("add_parameter")
    assert win.current_path() == ("parameters",) and len(win.model["parameters"]) == n + 1
    win._panel_action(f"move_up:{n}")
    assert win.model["parameters"][n - 1]["name"] == "p"
    win._panel_action(f"remove_row:{n - 1}")
    assert len(win.model["parameters"]) == n and win.errors == []
    labels = [it.text(0) for it in win._iter_items()]
    assert f"Parameters ({n})" in labels and not any(l.startswith("L = ") for l in labels)
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


def test_vector_parameters_in_the_window():
    """A matrix typed in the table; a point given by a vector parameter."""
    win = window("plate_hole_tension.json")
    n = len(win.model["parameters"])
    win.select(("parameters",))
    win._panel_action("add_parameter")
    table = win.props.editor(("parameters",))
    table.item(n, 1).setText("[1, 0; 0, 2]")
    table = win.props.editor(("parameters",))
    assert win.model["parameters"][n]["value"] == "[1, 0; 0, 2]"
    assert table.item(n, 2).text() == "[1, 0; 0, 2]"
    table.item(n, 1).setText("[0, 0]")
    assert win.model["parameters"][n]["value"] == [0, 0] and win.errors == []
    name = win.model["parameters"][n]["name"]
    ball = ("geometries", 0, "source", "shape", "of", 1)
    win.select(ball)
    holder = win.props.editor(ball + ("center",))
    edit(holder.edits[0], name)                                    # the whole vector
    assert E.get_at(win.model, ball + ("center",)) == name and win.errors == []
    line = win.props.editor(ball + ("center",))
    assert isinstance(line, QLineEdit) and line.text() == name
    edit(line, "0, 0.001")
    assert E.get_at(win.model, ball + ("center",)) == [0.0, 0.001]
    close(win)


def test_vector_parameter_for_a_vertex_in_the_window():
    win = window("l_bracket.json")
    win.model["parameters"].append({"name": "V", "value": [0.0, 0.0]})
    poly = ("geometries", 0, "source", "shape", "of", 0)
    win.select(poly)
    table = win.props.editor(poly + ("points",))
    table.item(0, 0).setText("V")
    assert E.get_at(win.model, poly + ("points", 0)) == "V" and win.errors == []
    table = win.props.editor(poly + ("points",))
    assert table.item(0, 0).text() == "V" and table.item(0, 1).text() == ""
    from PySide6.QtCore import Qt
    assert not table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    table.item(0, 0).setText("0.01")
    assert E.get_at(win.model, poly + ("points", 0)) == [0.01, 0.0]
    close(win)


def test_add_a_geometry_from_its_menu():
    """+ on Geometries: a submenu of the kinds of geometry; so does the button
    of the panel."""
    win = window("plate.json")
    win.select(("geometries",))
    win._fill_add_menu()
    sub = next(a.menu() for a in win.add_button.menu().actions() if a.menu() is not None)
    kinds = [a.data() for a in sub.actions() if a.data()]
    assert kinds == [f"add_geometry:{k}" for _, k, _, _ in E.GEOMETRY_KINDS]
    next(a for a in sub.actions() if a.data() == "add_geometry:extrusion").trigger()
    geom = win.model["geometries"][-1]
    assert geom["source"]["shape"]["type"] == "extrusion"
    assert win.current_path() == ("geometries", 1, "source", "shape", "profile")     # its profile, to draw
    win.select(("geometries",))
    from PySide6.QtWidgets import QPushButton
    b = win.props.findChild(QPushButton, "action:add_geometry")
    assert b is not None and b.menu() is not None
    next(a for a in b.menu().actions() if a.data() == "add_geometry:curve3").trigger()
    assert win.model["geometries"][-1]["source"]["type"] == "curve"
    assert win.errors == ["geometries[1].source.shape.profile.of: empty union: add shapes to it"]
    close(win)


def test_constructive_geometry_from_nothing_in_the_window():
    """An empty 2D and 3D constructive geometry, an extrusion of an empty
    profile: selected without error; shapes and booleans added."""
    win = window("plate.json")
    for kind in ("csg2", "csg3", "extrusion"):
        win.select(("geometries",))
        win._panel_action(f"add_geometry:{kind}")
        path = win.current_path()
        win.select(path)
        root = path + ("source", "shape")
        win.select(root)
        assert any("empty union" in e for e in win.errors)
        if kind == "extrusion":
            root = root + ("profile",)
            win.select(root)
        win._panel_action("add_shape:rectangle")
        win.select(root)
        win._panel_action("add_shape:difference")
        inner = win.current_path()
        win._panel_action("add_shape:ball")
        win.select(inner)
        win._panel_action("add_shape:ball")
        assert not [e for e in win.errors if e.startswith(f"geometries[{path[1]}]")], win.errors
        win.select(path)
    close(win)


def test_variants_of_an_action_in_one_menu():
    """The actions of a boolean: "Add…" and "Put in…" buttons with menus
    (not a button each), and submenus in the menu of the tree."""
    from PySide6.QtWidgets import QPushButton
    from getfem_gui.app.properties import group_label
    assert group_label(["Add a rectangle", "Add an ellipse"]) == "Add…"
    assert group_label(["Put in a union", "Put in an intersection"]) == "Put in…"
    win = window("plate_hole_tension.json")
    root = ("geometries", 0, "source", "shape")
    win.select(root)
    buttons = [b.objectName() for b in win.props.findChildren(QPushButton)]
    assert "action:add_shape" in buttons and "action:wrap" in buttons
    assert not any(n.startswith("action:add_shape:") for n in buttons)
    b = win.props.findChild(QPushButton, "action:add_shape")
    kinds = [a.data() for a in b.menu().actions()]
    assert "add_shape:rectangle" in kinds and "add_shape:difference" in kinds
    next(a for a in b.menu().actions() if a.data() == "add_shape:ball").trigger()
    assert win.current_path() == root + ("of", 2)
    menu = win._populate(QMenu(), root, False)
    titles = [a.text() for a in menu.actions()]
    assert "Add" in titles and "Put in" in titles and not any(t.startswith("Add a") for t in titles)
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
        # the edges of the elements over the field: a setting (the check box, or the Edges button)
        assert win.graphic_settings("mechanics", "u").get("edges", False) is False
        win.props.findChild(QCheckBox, "edges").setChecked(True)
        assert win.graphic_settings("mechanics", "u")["edges"] is True and win.viewer.edges["field"]
        win.viewer.toggle_edges()                           # the button of the view: the same setting
        assert win.graphic_settings("mechanics", "u")["edges"] is False and not win.viewer.edges["field"]
        assert not win.props.findChild(QCheckBox, "edges").isChecked()
        assert M.validate(win.model) == []
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


def test_materials_from_the_library_in_the_window():
    win = window("plate_inclusion.json")
    win.select(("materials",))
    menu = button(win, "add_from_library").menu()
    metals = next(a.menu() for a in menu.actions() if a.text() == "Metals")
    next(a for a in metals.actions() if a.text() == "Copper").trigger()
    assert win.model["materials"][-1]["from"] == "library:Copper"
    assert win.current_path() == ("materials", len(win.model["materials"]) - 1)
    win.select(("materials",))
    table = win.props.editor(("materials",))
    assert table.columnCount() == 2 and table.horizontalHeaderItem(1).text() == "k (W/(m·K))"
    table.item(0, 1).setText("0.25")
    # the same menu from the + button above the tree
    win.select(("materials",))
    labels = [a.text() for a in win._populate(win.add_button.menu(), ("materials",), True).actions()]
    assert "Add from a library" in labels
    assert win.model["materials"][0]["properties"]["k"] == 0.25 and win.errors == []
    close(win)


def test_action_buttons_wrapped():
    """The action buttons of a node on several lines when the panel is
    narrow, none beyond its right side, not overlapping."""
    from PySide6.QtWidgets import QPushButton
    from getfem_gui.app.window import MainWindow
    a = app()
    win = MainWindow(EXAMPLES / "l_bracket.json")
    try:
        win.resize(1200, 800)
        win.show()
        win.select(("geometries", 0, "source", "shape", "of", 0))
        a.processEvents()
        bs = sorted((b for b in win.props.findChildren(QPushButton) if b.objectName().startswith("action:")),
                    key=lambda b: (b.geometry().y(), b.geometry().x()))
        rows = sorted({b.geometry().y() for b in bs})
        assert len(bs) >= 4 and len(rows) >= 2
        width = bs[0].parentWidget().width()
        assert all(b.geometry().right() < width for b in bs)
        for b1, b2 in zip(bs, bs[1:]):
            if b1.geometry().y() == b2.geometry().y():
                assert b2.geometry().x() > b1.geometry().right()
    finally:
        win.setWindowModified(False)
        win.close()


def test_platform_under_wayland():
    """Under Wayland, Qt's X11 platform (XWayland) when it works: the 3D
    view (VTK's widget, an X window) inside the window; Qt's choice
    otherwise, or when the user chose."""
    from getfem_gui.app import choose_platform
    if not sys.platform.startswith("linux"):
        return
    wl = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
    env = dict(wl)
    assert choose_platform(env, probe=lambda: True) == "xcb" and env["QT_QPA_PLATFORM"] == "xcb"
    for env in ({"DISPLAY": ":0"}, {"WAYLAND_DISPLAY": "wayland-0"}, dict(wl, QT_QPA_PLATFORM="wayland"),
                dict(wl, GETFEM_GUI_WAYLAND="1")):
        before = dict(env)
        assert choose_platform(env, probe=lambda: True) is None and env == before
    env = dict(wl)
    assert choose_platform(env, probe=lambda: False) is None and "QT_QPA_PLATFORM" not in env
