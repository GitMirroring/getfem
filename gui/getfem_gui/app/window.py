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


"""Main window: model tree, properties of the selected node, 3D view,
messages. The model is a plain dictionary (see model.py); every change goes
through `change`, which keeps an undo history and validates the model."""

from __future__ import annotations

import copy
import html
import json
import tempfile
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QSplitter, QStyle, QTabWidget,
                               QTreeWidget, QTreeWidgetItem)

from .. import editing as E, mesher, model as M
from .jobs import GwflChecker, JobRunner
from .properties import PropertiesPanel
from .viewer import Viewer

APP_NAME = "GetFEM GUI"
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
PATH_ROLE = Qt.ItemDataRole.UserRole
UNDO_LIMIT = 200

# Result field shown first, and the field used for the deformed shape
DEFAULT_FIELD = {"linear_elasticity": ("von_mises", "displacement")}


class MainWindow(QMainWindow):
    def __init__(self, path: str | Path | None = None, python: str | None = None):
        super().__init__()
        self.jobs = JobRunner(python, self)
        self.jobs.output.connect(self._job_output)
        self.jobs.meshFinished.connect(self._mesh_finished)
        self.jobs.solveFinished.connect(self._solve_finished)
        self.jobs.runningChanged.connect(self._running_changed)
        self._tmpdir = None
        self._discard_job = False
        # GWFL check in the background, a moment after the last change
        self.checker = GwflChecker(python, self)
        self.checker.finished.connect(self._gwfl_checked)
        self.gwfl_errors: list[str] = []
        self._gwfl_key = None              # model for which gwfl_errors hold
        self._gwfl_failure_logged = False
        self._compute_after_check = False
        self._check_timer = QTimer(self)
        self._check_timer.setSingleShot(True)
        self._check_timer.setInterval(600)
        self._check_timer.timeout.connect(self._start_gwfl_check)
        self._build_ui()
        self._build_actions()
        self.model: dict = {}
        self.path: Path | None = None
        if path:
            self.open_file(path)
        else:
            self.new_model()
        if not self.viewer.available:
            self._log(self.viewer.error.strip() or "3D view unavailable", error=True)

    # ================================================================ interface

    def _build_ui(self):
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 860)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(220)
        self.tree.currentItemChanged.connect(lambda cur, _prev: self._node_selected())
        self.props = PropertiesPanel()
        self.props.changed.connect(self.change)
        self.props.action.connect(self._panel_action)
        self.props.removeProperty.connect(self._remove_property)
        self.props.viewChanged.connect(self._result_view_changed)
        self.viewer = Viewer()
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.tree)
        split.addWidget(self.props)
        split.addWidget(self.viewer)
        split.setStretchFactor(2, 1)
        split.setSizes([250, 340, 800])
        self.setCentralWidget(split)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setFont(self.props.font())
        self.problems = QListWidget()
        self.problems.itemActivated.connect(self._problem_activated)
        self.bottom = QTabWidget()
        self.bottom.addTab(self.log, "Messages")
        self.bottom.addTab(self.problems, "Problems")
        dock = QDockWidget("Messages", self)
        dock.setObjectName("messages")
        dock.setWidget(self.bottom)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dock.setTitleBarWidget(QLabel(""))
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [180], Qt.Orientation.Vertical)

        self.state = QLabel("")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(120)
        self.progress.hide()
        self.statusBar().addWidget(self.state, 1)
        self.statusBar().addPermanentWidget(self.progress)

    def _action(self, text, slot, shortcut=None, icon=None, tip=None) -> QAction:
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if icon is not None:
            a.setIcon(self.style().standardIcon(icon))
        if tip:
            a.setStatusTip(tip)
            a.setToolTip(tip)
        a.triggered.connect(slot)
        return a

    def _build_actions(self):
        SP = QStyle.StandardPixmap
        self.a_new = self._action("&New", self.new_model, QKeySequence.StandardKey.New, SP.SP_FileIcon)
        self.a_open = self._action("&Open...", self.open_dialog, QKeySequence.StandardKey.Open,
                                   SP.SP_DialogOpenButton)
        self.a_save = self._action("&Save", self.save, QKeySequence.StandardKey.Save,
                                   SP.SP_DialogSaveButton)
        self.a_save_as = self._action("Save &as...", self.save_as, QKeySequence.StandardKey.SaveAs)
        self.a_quit = self._action("&Quit", self.close, QKeySequence.StandardKey.Quit)
        self.a_undo = self._action("&Undo", self.undo, QKeySequence.StandardKey.Undo, SP.SP_ArrowBack)
        self.a_redo = self._action("&Redo", self.redo, QKeySequence.StandardKey.Redo, SP.SP_ArrowForward)
        self.a_mesh = self._action("Build &mesh", lambda: self.build_mesh(), "F7", SP.SP_FileDialogDetailedView,
                                   "Build the mesh of the selected node (or the one used by the physics)")
        self.a_compute = self._action("&Compute", self.compute, "F8", SP.SP_MediaPlay,
                                      "Mesh if needed, then solve")
        self.a_stop = self._action("S&top", self.jobs.stop, None, SP.SP_MediaStop,
                                   "Interrupt the running computation")
        self.a_stop.setEnabled(False)
        self.a_script = self._action("Show the solve &script", self.show_script)
        self.a_folder = self._action("Show the &working folder", self.show_workdir)

        bar = self.menuBar()
        m = bar.addMenu("&File")
        for a in (self.a_new, self.a_open):
            m.addAction(a)
        ex = m.addMenu("Open an &example")
        for p in sorted(EXAMPLES_DIR.glob("*.json")):
            ex.addAction(self._action(p.stem, lambda _=False, p=p: self.open_example(p)))
        m.addSeparator()
        for a in (self.a_save, self.a_save_as):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.a_quit)
        m = bar.addMenu("&Edit")
        m.addAction(self.a_undo)
        m.addAction(self.a_redo)
        m = bar.addMenu("&Compute")
        for a in (self.a_mesh, self.a_compute, self.a_stop):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.a_script)
        m.addAction(self.a_folder)
        m = bar.addMenu("&Help")
        m.addAction(self._action("&About", self.about))

        tb = self.addToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        for a in (self.a_new, self.a_open, self.a_save, None, self.a_undo, self.a_redo, None,
                  self.a_mesh, self.a_compute, self.a_stop):
            tb.addSeparator() if a is None else tb.addAction(a)

    # ================================================================ files

    def _confirm_discard(self) -> bool:
        if not self.isWindowModified():
            return True
        r = QMessageBox.question(self, APP_NAME, "The model has been modified. Save the changes?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                                 | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return r == QMessageBox.StandardButton.Discard

    def _set_model(self, model: dict, path: Path | None):
        if self.jobs.running:          # its result belongs to the previous model
            self._discard_job = True
            self.jobs.stop()
        self.model = model
        self.path = path
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self.view_settings: dict[str, dict] = {}
        self.setWindowModified(False)
        self._load_previous_results()
        self.refresh(select=())
        self._schedule_gwfl_check()
        self.tree.expandToDepth(1)
        self._update_title()
        self._update_undo()

    def new_model(self):
        if self._confirm_discard():
            model = M.default_model()
            model["_source_dir"] = str(Path.cwd())
            self._set_model(model, None)

    def open_dialog(self):
        if not self._confirm_discard():
            return
        start = str(self.path.parent if self.path else Path.cwd())
        name, _ = QFileDialog.getOpenFileName(self, "Open a model", start, "Models (*.json)")
        if name:
            self.open_file(name, confirm=False)

    def open_file(self, path, confirm=True) -> bool:
        if confirm and not self._confirm_discard():
            return False
        path = Path(path).resolve()
        try:
            model = M.load(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, APP_NAME, f"Cannot open {path}:\n{exc}")
            return False
        self._set_model(model, path)
        self._log(f"Opened {path}")
        return True

    def open_example(self, path: Path):
        """An example is opened as a new, unsaved model (the original is kept)."""
        if self._confirm_discard() and self.open_file(path, confirm=False):
            self.path = None
            self._load_previous_results()
            self._update_title()

    def save(self) -> bool:
        if self.path is None:
            return self.save_as()
        try:
            M.save(self.model, self.path)
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Cannot save {self.path}:\n{exc}")
            return False
        self.setWindowModified(False)
        self._log(f"Saved {self.path}")
        return True

    def save_as(self) -> bool:
        start = str(self.path or Path(self.model.get("_source_dir", Path.cwd())) / "model.json")
        name, _ = QFileDialog.getSaveFileName(self, "Save the model", start, "Models (*.json)")
        if not name:
            return False
        path = Path(name)
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        old_dir = Path(self.model.get("_source_dir", "."))
        self._rebase_files(old_dir, path.parent)
        self.path = path.resolve()
        self.model["_source_dir"] = str(self.path.parent)
        self._update_title()
        return self.save()

    def _rebase_files(self, old_dir: Path, new_dir: Path):
        """Keep relative file references (CAD, meshes) valid after a 'save as'."""
        for geom in self.model.get("geometries", []):
            src = geom.get("source", {})
            f = src.get("file")
            if f and not Path(f).is_absolute():
                p = (old_dir / f).resolve()
                try:
                    src["file"] = str(p.relative_to(new_dir.resolve()))
                except ValueError:
                    src["file"] = str(p)

    @property
    def workdir(self) -> Path:
        """Meshes and results go next to the model file, in <name>_work/."""
        if self.path is not None:
            return self.path.with_name(self.path.stem + "_work")
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="getfem_gui_")
        return Path(self._tmpdir.name)

    def _update_title(self):
        name = self.path.name if self.path else "unsaved"
        self.setWindowTitle(f"{self.model.get('name', 'Model')} — {name}[*] — {APP_NAME}")

    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        self.jobs.stop()
        self.checker.cancel()
        self.viewer.close_plotter()
        event.accept()

    # ================================================================ tree

    def current_path(self) -> tuple:
        item = self.tree.currentItem()
        return item.data(0, PATH_ROLE) if item is not None else ()

    def refresh(self, select=None, rebuild_panel=True):
        """Rebuild the tree (keeping the expanded nodes), validate, and show the
        node `select` (by default the current one)."""
        if select is None:
            select = self.current_path()
        expanded = set()
        it = self._iter_items()
        for item in it:
            if item.isExpanded():
                expanded.add(item.data(0, PATH_ROLE))
        self.errors = self._all_errors()
        root = E.tree(self.model, list(self.result_fields), has_results=self.results is not None)
        if self.results is not None and self.results_stale:
            res = E.find_node(root, (E.RESULTS,))
            res.label += " (out of date)"
        self.node_errors = E.errors_by_node(root, self.errors)
        warn = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)

        self.tree.blockSignals(True)
        self.tree.clear()
        items = {}

        def add(node, parent):
            item = QTreeWidgetItem([node.label])
            item.setData(0, PATH_ROLE, node.path)
            errs = self.node_errors.get(node.path, [])
            if errs:
                item.setIcon(0, warn)
                item.setToolTip(0, "\n".join(errs))
            (parent.addChild(item) if parent is not None else self.tree.addTopLevelItem(item))
            items[node.path] = item
            for c in node.children:
                add(c, item)
        add(root, None)
        for p, item in items.items():
            item.setExpanded(p in expanded or p == ())
        target = select
        while target not in items and target:
            target = target[:-1]
        item = items.get(target)
        self.tree.setCurrentItem(item)
        p = item.parent() if item else None
        while p is not None:
            p.setExpanded(True)
            p = p.parent()
        self.tree.blockSignals(False)

        self._update_problems()
        if rebuild_panel:
            self._node_selected()

    # ================================================================ GWFL check

    def _model_key(self) -> str:
        return json.dumps(_results_key(self.model), sort_keys=True)

    def _all_errors(self) -> list[str]:
        errors = M.validate(self.model)
        if not errors and self._gwfl_key == self._model_key():
            errors = list(self.gwfl_errors)
        return errors

    def _schedule_gwfl_check(self):
        self._check_timer.start()

    def _start_gwfl_check(self):
        if M.validate(self.model):        # structural errors first
            self._compute_after_check = False
            return
        key = self._model_key()
        if key == self._gwfl_key:
            self._gwfl_checked(key, self.gwfl_errors)
            return
        if not self.jobs.running:
            self.state.setText("Checking the expressions...")
        self.checker.check(copy.deepcopy(self.model), key, self.workdir)

    def _gwfl_checked(self, key, result):
        if key != self._model_key():      # the model changed meanwhile
            return
        if isinstance(result, str):       # the check could not run: do not block
            if not self._gwfl_failure_logged:
                self._log(f"Warning: {result}", error=True)
                self._gwfl_failure_logged = True
            result = []
        self.gwfl_errors, self._gwfl_key = result, key
        self.refresh(rebuild_panel=False)
        self._update_marks()
        if self._compute_after_check:
            self._compute_after_check = False
            self.compute()

    def _iter_items(self):
        stack = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def select(self, path):
        for item in self._iter_items():
            if item.data(0, PATH_ROLE) == tuple(path):
                self.tree.setCurrentItem(item)
                return True
        return False

    def _update_problems(self):
        self.problems.clear()
        for path, errs in self.node_errors.items():
            for e in errs:
                it = QListWidgetItem(e)
                it.setData(PATH_ROLE, path)
                self.problems.addItem(it)
        n = len(self.errors)
        self.bottom.setTabText(1, f"Problems ({n})" if n else "Problems")
        if not self.jobs.running:
            self.state.setText("Model valid" if not n else f"{n} problem(s): see the Problems tab")

    def _problem_activated(self, item):
        self.select(item.data(PATH_ROLE))

    # ================================================================ node display

    def _node_selected(self):
        path = self.current_path()
        if path is None:
            return
        if path[:1] == (E.RESULTS,):
            self._show_result_node(path)
            return
        item = self.tree.currentItem()
        title = item.text(0) if item else ""
        actions = self._node_actions(path)
        self._panel_fields = E.fields(self.model, path)
        self.props.show_node(self.model, title, path, self._panel_fields, self.errors,
                             self.node_errors.get(path, []), actions,
                             Path(self.model.get("_source_dir", ".")))
        self.update_view()

    def _show_result_node(self, path):
        if self.results is None:
            return
        if path == (E.RESULTS,):
            self.props.show_results(self.results, self.results_stale,
                                    [("cmd:compute", "Compute again")] if self.results_stale else [])
            self._show_field(self._default_field())
            return
        name = path[1]
        warp = self._warp_field()
        settings = self._settings(name)
        self.props.show_result_field(name, self.result_fields.get(name, 1), settings,
                                     can_warp=warp is not None)
        self._show_field(name)

    def _default_field(self):
        ptype = self.model.get("physics", {}).get("type")
        name = DEFAULT_FIELD.get(ptype, (self.model.get("physics", {}).get("variable"),))[0]
        if name in self.result_fields or not self.result_fields:
            return name
        return next(iter(self.result_fields))

    def _warp_field(self):
        ptype = self.model.get("physics", {}).get("type")
        w = DEFAULT_FIELD.get(ptype, (None, None))[1]
        return w if w in self.result_fields else None

    def _settings(self, name):
        s = self.view_settings.setdefault(name, {"component": None, "warp": True, "scale": None})
        if s["scale"] is None and self._warp_field() and self.viewer.available and self.result_file:
            from . import scene
            s["scale"] = round(scene.deformation_scale(scene.read(self.result_file), self._warp_field()), 6)
        return s

    def _result_view_changed(self, change: dict):
        path = self.current_path()
        if path[:1] == (E.RESULTS,) and len(path) == 2:
            self._settings(path[1]).update(change)
            self._show_field(path[1])

    def _show_field(self, name):
        if self.result_file is None or not self.result_file.exists():
            self.viewer.show_message("No result file")
            return
        s = self._settings(name)
        warp = self._warp_field() if s.get("warp", True) else None
        status = f"Result: {name}" + (" (out of date)" if self.results_stale else "")
        if warp:
            status += f" — deformation × {s.get('scale') or 1:g}"
        try:
            self.viewer.show_field(self.result_file, name, s.get("component"), warp,
                                   s.get("scale") or 1.0, status=status)
        except Exception as exc:  # noqa: BLE001 (unreadable or unexpected file)
            self.viewer.show_message(f"Cannot display {name}: {exc}")

    def update_view(self):
        path = self.current_path()
        if path[:1] == (E.RESULTS,):
            return
        mesh_name = E.mesh_to_show(self.model, path)
        mesh_name2, sel = E.highlighted_selection(self.model, path)
        if mesh_name is None:
            self.viewer.show_message("This geometry has no mesh: add one to display it.")
            return
        try:
            mres = mesher.cached_mesh(self.model, mesh_name, self.workdir, up_to_date=False)
        except Exception:  # noqa: BLE001 (incomplete model being edited)
            mres = None
        if mres is None:
            self.viewer.show_message(f"Mesh {mesh_name!r} not built yet: Compute ▸ Build mesh (F7)")
            return
        status = f"Mesh {mesh_name}: {mres.info.get('nb_elements', '?')} elements, " \
                 f"{mres.info.get('nb_points', '?')} points"
        if mres.stale:
            status += " — out of date (F7 to rebuild)"
        sel_file = None
        if sel is not None and mesh_name2 == mesh_name:
            sel_file = mres.view_file(sel) if sel != M.ALL else mres.view_file()
            status += f" — selection {sel!r}" + ("" if sel_file else " (not in this mesh, rebuild it)")
        try:
            self.viewer.show_mesh(mres.view_file(), sel_file, status=status)
        except Exception as exc:  # noqa: BLE001
            self.viewer.show_message(f"Cannot display the mesh: {exc}")

    # ================================================================ editing

    def _snapshot(self):
        self._undo.append(copy.deepcopy(self.model))
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        self._update_undo()

    def _changed(self, select=None, rebuild_panel=True):
        self.setWindowModified(True)
        self._schedule_gwfl_check()
        if self.results is not None:
            self.results_stale = _results_key(self.model) != self.results_model
        self._update_title()
        self.refresh(select=select, rebuild_panel=rebuild_panel)

    def change(self, f: E.Field, value):
        """Apply a field change coming from the properties panel."""
        before = copy.deepcopy(self.model)
        old_fields = [(x.key, x.kind, x.choices, x.size) for x in self._panel_fields]
        try:
            new_path = E.apply(self.model, f, value)
        except (ValueError, KeyError, TypeError) as exc:
            self.model = before
            self.state.setText(f"Not applied: {exc}")
            self._log(f"Not applied: {exc}", error=True)
            QTimer.singleShot(0, self._node_selected)
            return
        if self.model == before:
            return
        self._undo.append(before)
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        self._update_undo()
        path = new_path or self.current_path()
        new_fields = [(x.key, x.kind, x.choices, x.size) for x in E.fields(self.model, path)]
        # Rebuilding the form would lose the keyboard focus: only do it when the
        # form itself changes, otherwise just update the error marks.
        same = new_fields == old_fields and new_path is None and not self._focus_outside_panel()
        self._changed(select=path, rebuild_panel=not same)
        if same:
            self._update_marks()
            self.update_view()

    def _node_actions(self, path):
        actions = E.actions(self.model, path)
        if path[:1] == ("meshes",) and len(path) == 2:
            actions = [("cmd:build_mesh", "Build mesh")] + actions
        if path == ("physics",):
            actions = actions + [("cmd:compute", "Compute")]
        return actions

    def _focus_outside_panel(self) -> bool:
        w = QApplication.focusWidget()
        while w is not None:
            if w is self.props:
                return False
            w = w.parentWidget()
        return True

    def _update_marks(self):
        from .properties import ERROR_STYLE
        for f in self._panel_fields:
            w = self.props.editor(f.key)
            if w is None or f.kind == "info":
                continue
            msgs = E.errors_at(self.errors, f.key) if f.op != "rename_selection" else []
            w.setStyleSheet(ERROR_STYLE if msgs else "")
            w.setToolTip("\n".join(msgs) if msgs else f.help)

    def _panel_action(self, ident: str):
        path = self.current_path()
        if ident == "cmd:build_mesh":
            self.build_mesh(E.get_at(self.model, path + ("name",)))
            return
        if ident == "cmd:compute":
            self.compute()
            return
        name = None
        if E.needs_name(ident):
            name, ok = QInputDialog.getText(self, APP_NAME, "Name of the property (e.g. rho):")
            if not ok or not name.strip():
                return
            name = name.strip()
        before = copy.deepcopy(self.model)
        try:
            new_path = E.do_action(self.model, path, ident, name=name)
        except (ValueError, KeyError, IndexError) as exc:
            self.model = before
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=new_path if new_path is not None else path)

    def _remove_property(self, f: E.Field):
        self._snapshot()
        E.get_at(self.model, f.key[:-1]).pop(f.key[-1], None)
        self._changed()

    def undo(self):
        if self._undo:
            self._redo.append(copy.deepcopy(self.model))
            self.model = self._undo.pop()
            self._update_undo()
            self._changed()

    def redo(self):
        if self._redo:
            self._undo.append(copy.deepcopy(self.model))
            self.model = self._redo.pop()
            self._update_undo()
            self._changed()

    def _update_undo(self):
        self.a_undo.setEnabled(bool(getattr(self, "_undo", None)))
        self.a_redo.setEnabled(bool(getattr(self, "_redo", None)))

    # ================================================================ computations

    def _log(self, text: str, error: bool = False, color: str | None = None):
        text = text.rstrip("\n")
        if not text:
            return
        color = "#b3261e" if error else color
        if color:
            body = html.escape(text).replace("\n", "<br>").replace("  ", "&nbsp; ")
            self.log.appendHtml(f'<span style="color:{color}">{body}</span>')
        else:
            self.log.appendPlainText(text)

    def _job_output(self, text: str, is_err: bool):
        # Detailed output of the processes (GetFEM warnings, tracebacks) in grey;
        # the summary of a failure is then given in red by the finish handlers.
        self._log(text, color="#7a7f87" if is_err else None)

    def _running_changed(self, running: bool):
        self.progress.setVisible(running)
        self.a_stop.setEnabled(running)
        self.a_mesh.setEnabled(not running)
        self.a_compute.setEnabled(not running)
        if running:
            self.state.setText("Computing...")

    def _check_gwfl(self) -> bool:
        if self.gwfl_errors:
            self.bottom.setCurrentWidget(self.problems)
            self.state.setText("Some expressions are invalid: see the Problems tab")
            self._log("Cannot compute, invalid expressions:\n  " + "\n  ".join(self.gwfl_errors),
                      error=True)
            return False
        return True

    def _check(self, prefixes=None) -> bool:
        errors = M.validate(self.model)
        if prefixes:
            errors = [e for e in errors if e.startswith(prefixes)]
        if errors:
            self.bottom.setCurrentWidget(self.problems)
            self.state.setText("The model has problems: see the Problems tab")
            self._log("Cannot compute, the model has problems:\n  " + "\n  ".join(errors), error=True)
            return False
        return True

    def build_mesh(self, name: str | None = None):
        if self.jobs.running:
            return
        if name is None:
            name = E.mesh_to_show(self.model, self.current_path()) or \
                (M.physics_mesh(self.model) or {}).get("name")
        if name is None or not self._check(("geometries", "meshes")):
            return
        self.bottom.setCurrentWidget(self.log)
        self.jobs.build_mesh(self.model, name, self.workdir, force=False)

    def compute(self):
        if self.jobs.running or not self._check():
            return
        if self._gwfl_key != self._model_key():   # expressions not checked yet
            self._compute_after_check = True
            self._check_timer.stop()
            self._start_gwfl_check()
            return
        if not self._check_gwfl():
            return
        self.bottom.setCurrentWidget(self.log)
        self.jobs.solve(self.model, self.workdir)

    def _mesh_finished(self, mres):
        if self._discard_job:
            self._discard_job = self.jobs.running
            return
        if not mres.ok:
            self._log(f"Meshing failed: {mres.error}", error=True)
            self.state.setText(f"Meshing failed: {mres.error}")
            return
        info = mres.info
        if not mres.cached:
            sizes = ", ".join(f"{k}: {v}" for k, v in info.get("sizes", {}).items())
            self._log(f"Mesh {info.get('mesh')!r}: {info.get('nb_elements')} elements, "
                      f"{info.get('nb_points')} points" + (f" (selections — {sizes})" if sizes else ""))
        self.state.setText(f"Mesh {info.get('mesh')!r} ready")
        self.update_view()

    def _solve_finished(self, res):
        if self._discard_job:
            self._discard_job = False
            return
        if not res.ok:
            msg = res.error or "the solve failed"
            self._log(f"Solve failed: {msg}", error=True)
            self.state.setText(f"Solve failed: {msg}")
            return
        r = res.results
        self.results = r
        self.result_file = res.workdir / r["vtu"] if r.get("vtu") else None
        self.result_fields = self._read_fields()
        self.view_settings = {}
        self.results_model = _results_key(self.jobs.model)
        self.results_stale = _results_key(self.model) != self.results_model
        (res.workdir / "model_used.json").write_text(json.dumps(self.results_model, indent=1),
                                                     encoding="utf-8")
        self._log(f"Solved: {r.get('nb_dofs')} unknowns, {r.get('time_s', 0):.2f} s, "
                  f"relative residual {r.get('relative_residual', 0):.1e}")
        for w in r.get("warnings", []):
            self._log(f"Warning: {w}", error=True)
        for k, v in r.get("quantities", {}).items():
            self._log(f"  {k} = {v:.6g}")
        self.state.setText("Solved")
        self.refresh(select=(E.RESULTS, self._default_field()) if self.result_fields else (E.RESULTS,))

    def _read_fields(self) -> dict[str, int]:
        if self.result_file is None or not self.result_file.exists():
            return {}
        if self.viewer.available:
            from . import scene
            try:
                return scene.field_names(scene.read(self.result_file))
            except Exception:  # noqa: BLE001
                return {}
        return _vtu_field_names(self.result_file)

    def _load_previous_results(self):
        """Results of an earlier session (marked out of date if the model changed since)."""
        self.results, self.result_file, self.result_fields = None, None, {}
        self.results_model, self.results_stale = None, False
        f = self.workdir / "results.json"
        used = self.workdir / "model_used.json"
        if not (f.exists() and used.exists()):
            return
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            used_model = json.loads(used.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if r.get("status") != "ok":
            return
        self.results = r
        self.results_model = used_model
        self.results_stale = _results_key(self.model) != used_model
        self.result_file = self.workdir / r["vtu"] if r.get("vtu") else None
        self.result_fields = self._read_fields()

    def show_script(self):
        script = self.workdir / "run_getfem.py"
        if not script.exists():
            QMessageBox.information(self, APP_NAME, "No solve script yet: compute first (F8).")
            return
        dlg = QPlainTextEdit()
        dlg.setWindowTitle(f"{script} — {APP_NAME}")
        dlg.setPlainText(script.read_text(encoding="utf-8"))
        dlg.setReadOnly(True)
        dlg.resize(800, 700)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.show()
        self._script_window = dlg

    def show_workdir(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        self.workdir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.workdir)))

    def about(self):
        QMessageBox.about(self, APP_NAME,
                          f"<b>{APP_NAME}</b><p>A graphical interface for "
                          "<a href='https://getfem.org'>GetFEM</a>, the finite element library."
                          "<p>Distributed under the GNU LGPL, like GetFEM.")


def _results_key(model: dict) -> dict:
    """The part of the model that determines the results (for staleness checks),
    as it reads back from JSON."""
    return json.loads(json.dumps({k: v for k, v in model.items()
                                  if not k.startswith("_") and k != "name"}))


def _vtu_field_names(path: Path) -> dict[str, int]:
    """Array names of a VTU file without VTK (used when PyVista is missing)."""
    import re
    text = path.read_text(encoding="utf-8", errors="replace")
    piece = text.split("<PointData", 1)[-1]
    out = {}
    for tag in re.findall(r"<DataArray[^>]*>", piece):
        name = re.search(r'Name="([^"]+)"', tag)
        ncomp = re.search(r'NumberOfComponents="(\d+)"', tag)
        if name and name.group(1) != "Points":
            out[name.group(1)] = int(ncomp.group(1)) if ncomp else 1
    return out
