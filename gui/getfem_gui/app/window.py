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


"""Main window: model tree, properties of the selected node, 3D view,
messages. The model is a plain dictionary (see model.py); every change goes
through `change`, which keeps an undo history and validates the model."""

from __future__ import annotations

import copy
import html
import re
import json
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                               QPlainTextEdit, QProgressBar, QSplitter, QStyle, QTabWidget,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import editing as E, geometry2d as G, mesher, model as M, parameters as P
from .jobs import GwflChecker, JobRunner
from .properties import PropertiesPanel
from .viewer import Viewer

APP_NAME = "GetFEM GUI"
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
PATH_ROLE = Qt.ItemDataRole.UserRole
UNDO_LIMIT = 200

# Result field shown first, and the field used for the deformed shape


class MainWindow(QMainWindow):
    def __init__(self, path: str | Path | None = None, python: str | None = None,
                 versions=None):
        super().__init__()
        self.versions = versions        # splash.GetfemVersion, or None
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
        # right click: the actions of the node; the + button: what can be added to it
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.add_button = QToolButton()
        self.add_button.setText("+")
        self.add_button.setObjectName("add_button")
        self.add_button.setToolTip("Add an item to the selected node")
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_button.setMenu(QMenu(self.add_button))
        self.add_button.menu().aboutToShow.connect(self._fill_add_menu)
        tree_box = QWidget()
        tree_col = QVBoxLayout(tree_box)
        tree_col.setContentsMargins(0, 0, 0, 0)
        tree_col.setSpacing(2)
        tree_col.addWidget(self.add_button, 0, Qt.AlignmentFlag.AlignLeft)
        tree_col.addWidget(self.tree, 1)
        self.props = PropertiesPanel()
        self.props.changed.connect(self.change)
        self.props.action.connect(self._panel_action)
        self.props.removeProperty.connect(self._remove_property)
        self.props.viewChanged.connect(self._result_view_changed)
        self.viewer = Viewer()
        self.viewer.edgeClicked.connect(self._edge_clicked)
        self.viewer.handleClicked.connect(self._handle_clicked)
        self.viewer.fieldEdgesToggled.connect(self._field_edges_toggled)
        self.viewer.exportRequested.connect(lambda: self.export_view())
        self.viewer.field_edges_saved = True
        self.viewer.regionClicked.connect(self._region_clicked)
        self.viewer.faceClicked.connect(self._face_clicked)
        self.viewer.cutDrawn.connect(lambda ab: self._set_cut(ab))
        self.viewer.cutCleared.connect(lambda: self._set_cut(None))
        self.viewer.shapeDrawn.connect(self._shape_drawn)
        self.viewer.polygonEntered.connect(self._polygon_entered)
        self.viewer.escaped.connect(self._cancel_plane_pick)
        self.viewer.handleMoved.connect(self._handle_moved)
        self.viewer.handleAction.connect(self._handle_action)
        self.viewer.handle_menu = lambda name: E.handle_actions(
            self.model, E.constraint_owner(self.current_path()) or self.current_path(), name)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(tree_box)
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
        # triggered(bool checked): the slots are called without it (it would
        # otherwise be taken for their first optional argument)
        a.triggered.connect(lambda checked=False: slot())
        return a

    def _build_actions(self):
        SP = QStyle.StandardPixmap
        self.a_new = self._action("&New", self.new_model, QKeySequence.StandardKey.New, SP.SP_FileIcon)
        self.a_open = self._action("&Open...", self.open_dialog, QKeySequence.StandardKey.Open,
                                   SP.SP_DialogOpenButton)
        self.a_save = self._action("&Save", self.save, QKeySequence.StandardKey.Save,
                                   SP.SP_DialogSaveButton)
        self.a_save_as = self._action("Save &as...", self.save_as, QKeySequence.StandardKey.SaveAs)
        self.a_export = self._action("&Export as a Python script...", self.export_script, None, None,
                                     "A standalone GetFEM script that meshes and solves the model")
        self.a_export_geometry = self._action("Export the &geometry...", lambda: self.export_geometry(), None,
                                              None, "The geometry of the selected node as SVG, DXF, STL, "
                                                    "STEP...")
        self.a_export_mesh = self._action("Export the &mesh...", lambda: self.export_mesh(), None, None,
                                          "The mesh of the selected node as GetFEM, VTK, Gmsh, Abaqus...")
        self.a_export_view = self._action("Export the &view...", lambda: self.export_view(), None, None,
                                          "The view (mesh, field, geometry) as PDF, SVG or EPS (vector) or "
                                          "as an image (PNG...)")
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
        for a in (self.a_save, self.a_save_as, self.a_export, self.a_export_geometry, self.a_export_mesh,
                  self.a_export_view):
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
        m.addAction(self._action("&License", self.license_info))

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
        root = E.tree(self.model, {m: list(f) for m, f in self.result_fields.items()},
                      has_results=self.results is not None)
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
        # in the temporary folder: opening a model must not create its work folder
        self.checker.check(copy.deepcopy(self.model), key, None)

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
        if path[:1] == ("models",) and len(path) == 4 and path[2] == "graphics":
            self._show_graphic_node(path)
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
        """A result node: the summary of all the models or of one, or a field,
        shown with its graphic (the one named after it, if any)."""
        if self.results is None:
            return
        target = self._result_target(path)
        if len(path) < 3:
            data = self._results_summary(path[1] if len(path) == 2 else None)
            self.props.show_results(data, self.results_stale,
                                    [("cmd:compute", "Compute again")] if self.results_stale else [])
            if target is not None:
                self._show_field(*target)
            return
        mname, name = target
        md = M.model_by_name(self.model, mname)
        gi = E.find_graphic(md, name) if md else None
        self._show_settings_form(mname, name, md["graphics"][gi] if gi is not None else None)
        self._show_field(mname, name)

    def _show_graphic_node(self, path):
        """A saved graphic: its form, and its field if the results have it."""
        md = E.get_at(self.model, path[:2])
        g = E.get_at(self.model, path)
        self._show_settings_form(md["name"], g.get("field"), g, graphic_path=path)
        fields = self.result_fields.get(md["name"], {})
        if g.get("field") in fields:
            self._show_field(md["name"], g["field"], g)
        elif self.results is None:
            self.viewer.show_message(f"Graphic {g.get('name')!r}: no results yet (Compute, F8)")
        else:
            self.viewer.show_message(f"Graphic {g.get('name')!r}: the results of {md['name']!r} have no "
                                     f"field {g.get('field')!r} ({', '.join(fields) or 'none'})")

    def _show_settings_form(self, mname, name, graphic, graphic_path=None):
        fields = self.result_fields.get(mname, {})
        md = M.model_by_name(self.model, mname)
        planar = (M.model_dim(self.model, md) if md else 2) == 2
        known = list(fields) or [v.get("name") for v in (md or {}).get("variables", [])]
        self.props.show_result_field(name, fields.get(name, 1), self._settings(mname, name, graphic),
                                     can_warp=self._warp_field(mname) is not None,
                                     vectors=tuple(n for n, c in fields.items() if c > 1),
                                     planar=planar,
                                     graphic=graphic if graphic_path is not None else None,
                                     field_names=tuple(known))

    def _results_summary(self, mname=None) -> dict:
        """What the results table shows: one model, or all of them (its
        quantities prefixed by the model name)."""
        r = self.results or {}
        models = r.get("models", {})
        if mname is not None:
            out = dict(models.get(mname, {}))
            out["status"] = r.get("status")
            return out
        out = {"status": r.get("status"), "time_s": r.get("time_s")}
        solver = r.get("solver", {})
        if solver.get("type") == "staggered":
            out["iterations"] = solver.get("iterations")
        out["nb_dofs"] = sum(m.get("nb_dofs", 0) for m in models.values())
        out["quantities"] = {f"{m}.{q}" if len(models) > 1 else q: v for m, mr in models.items()
                             for q, v in mr.get("quantities", {}).items()}
        out["warnings"] = list(r.get("warnings", [])) + [w for mr in models.values() for w in mr.get("warnings", [])]
        for m, mr in models.items():
            for k, v in mr.items():
                if k.startswith(("max_", "min_")):
                    out[f"{k}" if len(models) == 1 else f"{k} ({m})"] = v
        return out

    def _result_target(self, path):
        """(model name, field name) shown for a result node."""
        if not self.result_fields:
            return None
        mname = path[1] if len(path) >= 2 else next(iter(self.result_fields))
        if len(path) >= 3:
            return mname, path[2]
        return mname, self._default_field(mname)

    def _default_field(self, mname):
        fields = self.result_fields.get(mname, {})
        md = M.model_by_name(self.model, mname) or {}
        for name in ["von_mises"] + [v.get("name") for v in md.get("variables", [])]:
            if name in fields:
                return name
        return next(iter(fields), None)

    def _warp_field(self, mname):
        w = (self.results or {}).get("models", {}).get(mname, {}).get("warp")
        return w if w in self.result_fields.get(mname, {}) else None

    def graphic_settings(self, mname, name):
        """The display settings of a result field (its graphic, or the defaults)."""
        md = M.model_by_name(self.model, mname) or {}
        gi = E.find_graphic(md, name)
        return self._settings(mname, name, md["graphics"][gi] if gi is not None else None)

    def _settings(self, mname, name, graphic):
        s = E.graphic_settings(graphic)
        file = self.result_files.get(mname)
        if s["scale"] is None and self._warp_field(mname) and self.viewer.available and file and file.exists():
            from . import scene
            s["scale"] = round(scene.deformation_scale(scene.read(file), self._warp_field(mname)), 6)
        return s

    def _current_graphic(self, create: bool):
        """(path, graphic) changed by the display settings: the selected
        graphic, or the one of the selected result field (created if needed)."""
        path = self.current_path()
        if path[:1] == ("models",) and len(path) == 4 and path[2] == "graphics":
            return path, E.get_at(self.model, path)
        if path[:1] == (E.RESULTS,) and len(path) == 3:
            k = next((i for i, m in enumerate(self.model.get("models", [])) if m.get("name") == path[1]), None)
            if k is None:
                return None, None
            md = self.model["models"][k]
            gi = E.find_graphic(md, path[2])
            if gi is None:
                if not create:
                    return None, None
                gp = E.add_graphic(md, path[2], ("models", k))
                gi = gp[-1]
            return ("models", k, "graphics", gi), md["graphics"][gi]
        return None, None

    def _field_edges_toggled(self, on: bool) -> None:
        """The Edges button of the view on a result field: its setting, and
        the check box of the form."""
        from PySide6.QtWidgets import QCheckBox
        box = self.props.findChild(QCheckBox, "edges")
        if box is not None:
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)
        self._result_view_changed({"edges": on})

    def _result_view_changed(self, change: dict):
        """A display setting changed: saved in the graphic (an edit of the
        model, undoable, which does not make the results out of date)."""
        before = copy.deepcopy(self.model)
        gpath, graphic = self._current_graphic(create=True)
        if graphic is None:
            return
        change = dict(change)
        if "name" in change:
            name = str(change["name"]).strip()
            others = [g.get("name") for g in E.get_at(self.model, gpath[:3]) if g is not graphic]
            if not name or name in others:
                self.model = before
                self.state.setText(f"Not applied: the name {name!r} is empty or already used")
                return
            change["name"] = name
        E.set_graphic(graphic, change)
        if self.model == before:
            return
        self._undo.append(before)
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        self._update_undo()
        self.setWindowModified(True)
        self._update_title()
        path = self.current_path()
        rebuild = bool({"name", "field"} & set(change)) or len(E.get_at(before, gpath[:3], [])) != \
            len(E.get_at(self.model, gpath[:3], []))
        self.refresh(select=path, rebuild_panel=False)
        if rebuild or path[0] != E.RESULTS:
            self._node_selected()
        else:
            self._show_field(path[1], path[2])

    def _set_cut(self, ab):
        """A cut line drawn in the view (or removed): saved in the graphic."""
        path = self.current_path()
        if path[:1] == (E.RESULTS,) and len(path) == 3 or path[:1] == ("models",) and \
                len(path) == 4 and path[2] == "graphics":
            self._result_view_changed({"cut": ab})

    def _show_field(self, mname, name, graphic=None):
        file = self.result_files.get(mname)
        if file is None or not file.exists() or name is None:
            self.viewer.show_message("No result file")
            return
        if graphic is None:
            md = M.model_by_name(self.model, mname) or {}
            gi = E.find_graphic(md, name)
            graphic = md["graphics"][gi] if gi is not None else None
        s = self._settings(mname, name, graphic)
        warp = self._warp_field(mname) if s.get("warp", True) else None
        status = f"Result of {mname}: {name}" + (" (out of date)" if self.results_stale else "")
        if graphic is not None and graphic.get("name") != name:
            status = f"Graphic {graphic.get('name')!r} — " + status
        if warp:
            status += f" — deformation × {s.get('scale') or 1:g}"
        try:
            view = {"isolines": s.get("isolines", 0), "bands": s.get("bands", False),
                    "cmap": s.get("cmap"), "clim": s.get("range"), "arrows": s.get("arrows"),
                    "cut": s.get("cut"), "edges": bool(s.get("edges", False))}
            self.viewer.show_field(file, name, s.get("component"), warp,
                                   s.get("scale") or 1.0, status=status, view=view)
        except Exception as exc:  # noqa: BLE001 (unreadable or unexpected file)
            self.viewer.show_message(f"Cannot display {name}: {exc}")

    def _preview(self, path):
        """What the exact 2D preview shows for a node, or None when it does not
        apply (mesh and result nodes, 3D or imported geometries):
        (title, shape, pieces, highlighted pieces, domain highlighted, pickable).
        A constraint of a polygon: the polygon, the sides it refers to highlighted."""
        poly = E.constraint_owner(path)
        if poly is None:
            return self._preview_of(path)
        out = self._preview_of(poly)
        if out is None:
            return None
        title, shape, pieces, _, domain, pickable = out
        from .. import sketch
        c = E.get_at(self.model, path, {})
        root = E.profile_root(poly) or tuple(poly[:4])
        src = tuple(poly[len(root) + 1::2])
        sides = {f"side{k}" for kind, k in sketch.references(c) if kind == "side"}
        if c.get("type") in ("angle",):          # the two sides at the vertex
            n = len(E.get_at(self.model, poly, {}).get("points") or [])
            k = c.get("vertex", 1)
            sides = {f"side{(k - 2) % n + 1}", f"side{k}"} if n else set()
        highlighted = [p for p in pieces if p.side.shape == src and p.side.side in sides]
        hint = E.constraint_pick_hint(self.model, path)
        return title.split(" — ")[0] + f" — constraint: {sketch.label(c)}" + (f" ({hint})" if hint else ""), \
            shape, pieces, highlighted, domain, "edges"

    def _preview_of(self, path):
        if path[:1] in (("meshes",), (E.RESULTS,)):
            return None
        model = P.resolved(self.model)
        geoms = model.get("geometries", [])
        gi = E.preview_geometry(self.model, path)
        if gi is None or gi >= len(geoms):
            return None
        geom = geoms[gi]
        prof = E.profile_root(path)
        if prof is not None:                  # the profile of an extrusion or a revolution, in its plane
            shape = E.get_at(model, prof)
            if not isinstance(shape, dict):
                return None
            try:
                pieces = G.pieces(shape)
            except (KeyError, TypeError, ValueError, IndexError):
                return None
            swept = E.get_at(model, prof[:-1], {})
            plane = "(r, z)" if swept.get("type") == "revolution" else "(x, y)"
            self._reference = self._section_under(model, gi, prof)
            self._context = self._plane_context(model, gi, prof)
            csg = tuple(path[len(prof) + 1::2])
            highlighted = [p for p in pieces if p.side.shape[:len(csg)] == csg] if len(path) > len(prof) else []
            return (f"Profile of the {swept.get('type')} of {geom.get('name')!r}, in the {plane} plane",
                    shape, pieces, highlighted, False, False)
        shape = G.shape_of(geom.get("source", {}))
        if shape is None:
            return None
        try:
            pieces = G.pieces(shape)
            table = M.selection_table(model, geom)
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        title = f"Geometry {geom.get('name')!r}"
        highlighted, domain, pickable = [], False, False
        if path[:1] == ("geometries",) and len(path) >= 5 and path[2:4] == ("source", "shape"):
            csg = tuple(path[5::2])             # ("of", j, "of", k) -> (j, k)
            highlighted = [p for p in pieces if p.side.shape[:len(csg)] == csg]
            title += " — shape highlighted"
            return title, shape, pieces, highlighted, False, False
        if path[:1] == ("geometries",) and len(path) == 4 and path[2] == "selections":
            sel = path[3]
            pickable = not table.get(sel, {}).get("builtin") and table.get(sel, {}).get("kind") == M.BOUNDARY
        else:
            _, sel = E.highlighted_selection(self.model, path)
            pickable = E.can_pick_edges(self.model, path)
        if E.can_pick_regions(self.model, path):
            pickable = "regions"
        if sel is not None and sel in table:
            if table[sel]["kind"] == M.DOMAIN:
                level = G.domain_level(shape, sel, table)
                domain = True if sel == M.ALL else level
                title += f" — domain {sel!r}" + ("" if level is not None else " (shown on the mesh only)")
            else:
                found = G.pieces_matching(shape, sel, table, pieces)
                highlighted = found or []
                title += f" — boundary {sel!r}" + ("" if found is not None else " (shown on the mesh only)")
        if pickable == "regions":
            title += ": click regions to add or remove them"
        elif pickable:
            title += ": click edges to add or remove them"
        return title, shape, pieces, highlighted, domain, pickable

    def _show_manifold(self, path) -> bool:
        """The preview of a curve or a surface in 3D for the node of its
        geometry (its other nodes show the flat domain of a surface, in 2D,
        or the mesh): False if none applies."""
        from .. import manifold
        if path[:1] != ("geometries",) or len(path) != 2 or not self.viewer.available:
            return False
        model = P.resolved(self.model)
        geom = model["geometries"][path[1]]
        src = geom.get("source", {})
        if src.get("type") not in manifold.SOURCES:
            return False
        if any(e.startswith(f"geometries[{path[1]}]") or e.startswith("parameters") for e in self.errors):
            self.viewer.show_message("The geometry has errors: see the panel.")
            return True
        vals, _ = P.values(model)
        what = {"curve": "curve", "surface": "surface (the flat domain below: its selections)",
                "swept_surface": "swept surface"}[src["type"]]
        try:
            self.viewer.show_manifold(src, vals, status=f"{geom['name']}: {what} (preview, no mesh)",
                                      key=geom.get("name"))
        except Exception as exc:  # noqa: BLE001
            self.viewer.show_message(f"Cannot display the geometry: {exc}")
        return True

    def _show_preview_3d(self, path) -> bool:
        """The preview of a 3D box or CSG geometry for a node (False if none
        applies): its surface, the selection or the support of the node,
        or the selected shape."""
        from .. import geometry3d as G3
        if path[:1] in (("meshes",), (E.RESULTS,)):
            return False
        model = P.resolved(self.model)
        gi = E.preview_geometry(self.model, path)
        geoms = model.get("geometries", [])
        if gi is None or gi >= len(geoms):
            return False
        geom = geoms[gi]
        shape = G3.shape_of(geom.get("source", {}))
        if shape is None or not self.viewer.available:
            return False
        try:
            G3.bounds(shape)
            table = M.selection_table(model, geom)
        except (KeyError, TypeError, ValueError, IndexError):
            return False
        title = f"Geometry {geom.get('name')!r}"
        highlighted, part = None, None
        if path[:1] == ("geometries",) and len(path) >= 5 and path[2:4] == ("source", "shape"):
            part = E.part_3d(model, path)
            title += " — shape highlighted"
        else:
            if path[:1] == ("geometries",) and len(path) == 4 and path[2] == "selections":
                sel = path[3]
            else:
                _, sel = E.highlighted_selection(self.model, path)
            if sel is not None and sel in table:
                if table[sel]["kind"] == M.DOMAIN:
                    highlighted = True if sel == M.ALL else None
                    title += f" — domain {sel!r}" + ("" if sel == M.ALL else " (shown on the mesh only)")
                else:
                    from . import scene
                    surf = scene.geometry_surface_3d(shape)
                    C = np.asarray(surf.cell_centers().points) if surf.n_cells else np.zeros((0, 3))
                    N = np.asarray(surf.cell_data["outward"]) if surf.n_cells else np.zeros((0, 3))
                    found = G3.faces_matching(shape, C, N, sel, table, scene.cell_faces(surf))
                    highlighted = found
                    title += f" — boundary {sel!r}" + ("" if found is not None else " (shown on the mesh only)")
        pickable = E.can_pick_faces(self.model, path) or E.can_pick_faces_of_selection(self.model, path)
        if pickable:
            title += ": click faces to add or remove them"
        if getattr(self, "_plane_target", None) == path:
            pickable = True
            title += ": click the plane face to put the extrusion on (Escape: cancel)"
        try:
            self.viewer.show_geometry_3d(shape, highlighted, part, status=title + " (preview, no mesh)",
                                         key=geom.get("name"), pickable=pickable)
        except Exception as exc:  # noqa: BLE001
            self.viewer.show_message(f"Cannot display the geometry: {exc}")
        return True

    def _handle_preview(self, path, name, point):
        """The primitive as it would be with the handle at point (for the view)."""
        m = copy.deepcopy(self.model)
        try:
            E.move_handle(m, path, name, point)
        except ValueError:
            return None
        from .. import transforms as T
        A, b = E.handle_frame(m, path)            # where the transforms above it put it
        return T.flatten(E.get_at(P.resolved(m), path), 2, A, b)

    def _handle_action(self, ident: str) -> None:
        """An action chosen in the menu of a handle (right click in the view):
        on the polygon (also when one of its constraints is selected); an
        error goes to the status bar."""
        path = E.constraint_owner(self.current_path()) or self.current_path()
        before = copy.deepcopy(self.model)
        try:
            new_path = E.do_action(self.model, path, ident)
        except (ValueError, KeyError, IndexError) as exc:
            self.model = before
            self.state.setText(str(exc))
            return
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=new_path if new_path is not None else path)

    def _handle_moved(self, name: str, point):
        path = E.constraint_owner(self.current_path()) or self.current_path()
        before = copy.deepcopy(self.model)
        try:
            notes = E.move_handle(self.model, path, name, point)
        except ValueError as exc:          # LockedValue included: the model is unchanged
            self.state.setText(f"Not moved: {exc}")
            self.update_view()
            return
        if self.model == before:
            return
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=path)
        self.state.setText("Shape modified" + (f" ({'; '.join(notes)})" if notes else ""))

    def _polygon_entered(self, points):
        """The vertices of the empty polygon selected in the tree, clicked in the view."""
        path = self.current_path()
        if not E.polygon_to_enter(self.model, path):
            return
        before = copy.deepcopy(self.model)
        E.get_at(self.model, path)["points"] = [list(p) for p in points]
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=path)
        self.state.setText(f"Polygon of {len(points)} vertices entered")

    def _cancel_plane_pick(self):
        if getattr(self, "_plane_target", None) is not None:
            self._plane_target = None
            self.state.setText("Cancelled")
            self.update_view()

    def _extrusion_on_face(self, path):
        from .. import geometry3d as G3
        target, self._plane_target = self._plane_target, None
        if target != path:
            return
        plane = self.viewer.face_plane(getattr(self.viewer, "last_face", -1))
        if plane is None:
            self._plane_target = target
            self.state.setText("This face is not plane: click a plane face")
            return
        C, n = plane
        shape = G3.shape_of(P.resolved(self.model)["geometries"][path[1]]["source"])
        lo, hi = G3.bounds(shape)
        before = copy.deepcopy(self.model)
        prof = E.add_extrusion_on_plane(self.model, path, C, n, 0.2 * float(np.max(hi - lo)),
                                        getattr(self, "_plane_kind", "boss"))
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=prof)
        z = E.get_at(self.model, prof[:-1])["z"]
        self.state.setText("Extrusion " + ("out of" if z[0] >= 0 else "into") + " the face: draw its profile "
                           "(the section of the solid is shown under it)")

    def _plane_context(self, model: dict, gi: int, prof):
        """The 3D geometry in the coordinates of the plane of a profile (the
        plane z = 0; for a revolution, its (r, z) half plane), for the
        profile drawn in the 3D view; None without 3D view."""
        if not self.viewer.available:
            return None
        from .. import geometry3d as G3
        from . import scene
        try:
            root = G3.shape_of(model["geometries"][gi]["source"])
            G3.bounds(root)
            surf = scene.geometry_surface_3d(root)
            swept = E.placed_swept(model, prof)
            O, R = G3.frame(swept) or (np.zeros(3), np.eye(3))
            Q = (np.asarray(surf.points, float) - O) @ R
            if swept["type"] == "revolution":           # (x, z, -y): the (r, z) half plane at z = 0
                Q = np.column_stack([Q[:, 0], Q[:, 2], -Q[:, 1]])
            out = surf.copy()
            out.points = Q
            return out
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            return None

    def _section_under(self, model: dict, gi: int, prof) -> tuple | None:
        """The section of the rest of the 3D geometry in the plane of a
        profile (geometry3d.section_grid), drawn under it."""
        from .. import geometry2d as G
        from .. import geometry3d as G3
        try:
            root = model["geometries"][gi]["source"]["shape"]
            swept = E.placed_swept(model, prof)
            rest = G3.without(root, tuple(prof[5:-1:2]))
            if next(G3.primitives(rest), None) is None:
                return None
            lo, hi = G3.plane_extent(rest, swept)
            if not G.is_empty(swept["profile"]):
                plo, phi = G.bounds(swept["profile"])
                lo, hi = np.minimum(lo, plo), np.maximum(hi, phi)
            m = 0.08 * float(np.max(hi - lo) or 1.0)
            return G3.section_grid(rest, swept, lo - m, hi + m)
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    def _shape_drawn(self, shape: dict, op: str):
        """A rectangle, a disk or a polygon drawn in the view: combined with the geometry."""
        path = self.current_path()
        gi = E.can_draw(self.model, path)
        if gi is None:
            return
        root = E.draw_root(self.model, path)
        before = copy.deepcopy(self.model)
        try:
            _, notes = E.add_shape_2d(self.model, gi, shape, op, root)
        except (ValueError, KeyError, TypeError) as exc:
            self.model = before
            self.state.setText(f"Not added: {exc}")
            return
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=E.profile_root(path) or ("geometries", gi))
        what = {"rectangle": "Rectangle", "ball": "Disk", "ellipse": "Ellipse", "polygon": "Polygon"}.get(shape["type"], "Shape")
        verb = {"union": "added", "difference": "subtracted", "intersection": "intersected"}[op]
        self.state.setText(f"{what} {verb}" + (": " + "; ".join(notes) if notes else ""))
        for n in notes:
            self._log(n)

    def _constraint_pick(self, kind: str, k: int) -> bool:
        """A side or a vertex of the polygon clicked while one of its
        constraints is selected: set in it (for a pair of sides, the clicks
        set the side, then the other side, in turn)."""
        path = self.current_path()
        if E.constraint_owner(path) is None:
            return False
        nxt = getattr(self, "_pick_next", None)
        other = nxt is not None and nxt[0] == path and nxt[1] == "other"
        before = copy.deepcopy(self.model)
        what = E.pick_for_constraint(self.model, path, kind, k, other)
        if what is None:
            self.model = before
            self.state.setText(f"Not for this constraint: {E.constraint_pick_hint(self.model, path)}")
            return True
        self._pick_next = (path, "side" if other else "other")
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=path)
        self.state.setText(f"Constraint: {what}")
        return True

    def _handle_clicked(self, name: str) -> None:
        """A handle clicked: a vertex (or a side, by its middle or a point of
        its spline) for the selected constraint."""
        if re.fullmatch(r"v\d+", name):
            self._constraint_pick("vertex", int(name[1:]) + 1)
        elif re.fullmatch(r"m\d+", name):
            self._constraint_pick("side", int(name[1:]) + 1)
        elif re.fullmatch(r"s\d+_\d+", name):
            self._constraint_pick("side", int(name[1:].split("_")[0]) + 1)

    def _edge_clicked(self, piece):
        """A click on an edge of the preview: toggles it in the selected
        selection, or in the support of the selected condition / output; a
        side of the polygon of the selected constraint: set in it."""
        path = self.current_path()
        poly = E.constraint_owner(path)
        if poly is not None:
            root = E.profile_root(poly) or tuple(poly[:4])
            if piece.side.shape == tuple(poly[len(root) + 1::2]) and re.fullmatch(r"side\d+", piece.side.side):
                self._constraint_pick("side", int(piece.side.side[4:]))
            return
        if E.can_pick_edges(self.model, path):
            before = copy.deepcopy(self.model)
            name, created = E.pick_edge_for_item(self.model, path, piece.side.ref)
            self._undo.append(before)
            self._redo.clear()
            self._update_undo()
            self._changed(select=path)
            gi = E.preview_geometry(self.model, path)
            shape = G.shape_of(P.resolved(self.model)["geometries"][gi]["source"])
            crit = self.model["geometries"][gi]["selections"][name]["criteria"]
            on = any((tuple(r["shape"]), r["side"]) == (piece.side.shape, piece.side.side)
                     for r in crit["edges"])
            self.state.setText(f"{G.side_label(shape, piece.side.ref)} {'added to' if on else 'removed from'} "
                               f"{name!r}" + (" (new selection for this item)" if created else ""))
            return
        if not (path[:1] == ("geometries",) and len(path) == 4 and path[2] == "selections"):
            return
        sel = self.model["geometries"][path[1]]["selections"].get(path[3])
        if sel is None:
            return
        before = copy.deepcopy(self.model)
        replaced = sel.get("criteria", {}).get("type") != "edges"
        sel["criteria"] = G.toggle_edge(sel.get("criteria", {}), piece.side.ref)
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=path)
        shape = G.shape_of(P.resolved(self.model)["geometries"][path[1]]["source"])
        what = G.side_label(shape, piece.side.ref)
        self.state.setText(f"{what}: " + ("added" if any(
            (tuple(r["shape"]), r["side"]) == (piece.side.shape, piece.side.side)
            for r in sel["criteria"]["edges"]) else "removed")
            + (" (the previous criterion was replaced by picked edges)" if replaced else ""))

    def _current_faces(self, gi: int, selection: str) -> list[dict]:
        """The faces of the 3D preview entirely in a boundary selection."""
        from .. import geometry3d as G3
        from . import scene
        model = P.resolved(self.model)
        geom = model["geometries"][gi]
        shape = G3.shape_of(geom["source"])
        surf = scene.geometry_surface_3d(shape)
        refs = scene.face_refs(surf)
        if refs is None or not surf.n_cells:
            return []
        C, N = np.asarray(surf.cell_centers().points), np.asarray(surf.cell_data["outward"])
        found = G3.faces_matching(shape, C, N, selection, M.selection_table(model, geom), scene.cell_faces(surf))
        if found is None:
            return []
        ids = np.asarray(surf.cell_data["face_id"])
        return [refs[i] for i in np.unique(ids) if np.all(found[ids == i])]

    def _face_clicked(self, ref):
        """A click on a face of the 3D preview: toggles it in the selected
        boundary selection, or in the support of the selected item; or gives
        the work plane of a new extrusion ("Add an extrusion on a face")."""
        from .. import geometry3d as G3
        path = self.current_path()
        if getattr(self, "_plane_target", None) is not None:
            self._extrusion_on_face(path)
            return
        gi = E.preview_geometry(self.model, path)
        if gi is None:
            return
        before = copy.deepcopy(self.model)
        created, replaced = False, False
        if E.can_pick_faces(self.model, path):
            item = E.get_at(self.model, path)
            start = self._current_faces(gi, item.get("selection", M.ALL))
            name, created = E.pick_face_for_item(self.model, path, ref, start)
        elif E.can_pick_faces_of_selection(self.model, path):
            name = path[3]
            sel = self.model["geometries"][gi]["selections"][name]
            replaced = sel.get("criteria", {}).get("type") != "faces"
            sel["criteria"] = G3.toggle_face(sel.get("criteria", {}), ref)
        else:
            return
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=path)
        shape = G3.shape_of(P.resolved(self.model)["geometries"][gi]["source"])
        faces = self.model["geometries"][gi]["selections"][name]["criteria"]["faces"]
        on = any(G3.face_key(r) == G3.face_key(ref) for r in faces)
        self.state.setText(f"{G3.face_label(shape, ref)} {'added to' if on else 'removed from'} {name!r}"
                           + (" (new selection for this item)" if created else "")
                           + (" (the previous criterion was replaced by picked faces)" if replaced else ""))

    def _region_clicked(self, region):
        """A click in a region of the preview: toggles it in the selected
        domain selection, or in the support of the selected item."""
        path = self.current_path()
        if not E.can_pick_regions(self.model, path):
            return
        before = copy.deepcopy(self.model)
        if path[:1] == ("geometries",):
            replaced = self.model["geometries"][path[1]]["selections"][path[3]] \
                .get("criteria", {}).get("type") != "regions"
            gi = path[1]
        else:
            replaced = False
            gi = E.preview_geometry(self.model, path)
        name, created = E.pick_region_for_item(self.model, path, region)
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=path)
        shape = G.shape_of(P.resolved(self.model)["geometries"][gi]["source"])
        crit = self.model["geometries"][gi]["selections"][name]["criteria"]
        key = sorted(tuple(p) for p in region)
        on = any(sorted(tuple(p) for p in r) == key for r in crit.get("regions", []))
        self.state.setText(f"Region {G.region_label(shape, region)} {'added to' if on else 'removed from'} "
                           f"{name!r}" + (" (new selection for this item)" if created else "")
                           + (" (the previous criterion was replaced by picked regions)" if replaced else ""))

    def update_view(self):
        path = self.current_path()
        if getattr(self, "_plane_target", None) not in (None, path):    # another node: no more picking
            self._plane_target = None
        if path[:1] == (E.RESULTS,):
            return
        if self._show_manifold(path):
            self.viewer.set_drawing(False)
            self.viewer.set_handles([])
            return
        preview = self._preview(path)
        if preview is None and self._show_preview_3d(path):
            self.viewer.set_drawing(False)
            self.viewer.set_handles([])
            return
        self.viewer.set_drawing(preview is not None and E.can_draw(self.model, path) is not None)
        spath = E.constraint_owner(path) or path          # a constraint: the handles of its polygon
        handles = E.shape_handles(self.model, spath) if preview is not None else []
        self.viewer.set_handles([])
        if preview is not None:
            title, shape, pieces, highlighted, domain, pickable = preview
            if handles:
                title += " — drag the handles to move or resize"
            reference, self._reference = getattr(self, "_reference", None), None
            context, self._context = getattr(self, "_context", None), None
            try:
                self.viewer.show_geometry(shape, pieces, highlighted, domain, status=title,
                                          pickable=pickable, key=title.split(" — ")[0].split(":")[0],
                                          reference=reference, context=context)
                self.viewer.set_handles(handles, lambda name, point, p=spath: self._handle_preview(p, name, point),
                                        marks=E.constraint_marks(self.model, path))
                if E.polygon_to_enter(self.model, path):
                    self.viewer.enter_polygon()
            except Exception as exc:  # noqa: BLE001
                self.viewer.show_message(f"Cannot display the geometry: {exc}")
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
        same = new_fields == old_fields and new_path is None and not self._focus_outside_panel() \
            and not any(x.kind in ("parameter_table", "material_table") for x in self._panel_fields)
        self._changed(select=path, rebuild_panel=not same)
        if same:
            self._update_marks()
            self.update_view()

    def _node_actions(self, path):
        actions = E.actions(self.model, path)
        if path[:1] == ("meshes",) and len(path) == 2:
            actions = [("cmd:build_mesh", "Build mesh"), ("cmd:export_mesh", "Export...")] + actions
        if path[:1] == ("geometries",) and len(path) == 2 \
                and self.model["geometries"][path[1]].get("source", {}).get("type") in ("box", "csg"):
            actions = actions + [("cmd:export_geometry", "Export...")]
        if path[:1] == ("models",) and len(path) == 2 or path == ("solvers",):
            actions = actions + [("cmd:compute", "Compute")]
        return actions

    def menu_actions(self, path, adding_only=False) -> list[tuple[str, str]]:
        """Actions of a node for its menus; "add a module" lists the module types."""
        out = []
        for ident, label in self._node_actions(path):
            if ident == "add_module":
                out += [(i, l) for i, l in E.actions(self.model, path[:2] + ("modules",))]
            elif not adding_only or ident.startswith("add"):
                out.append((ident, label))
        return out

    def _populate(self, menu, path, adding_only):
        from .properties import geometry_menu, group_label, library_menu
        menu.clear()
        actions = self.menu_actions(path, adding_only)
        groups: dict[str, list] = {}
        for ident, label in actions:              # the variants of an action: a submenu
            if ":" in ident and not ident.startswith("cmd:"):
                groups.setdefault(ident.split(":", 1)[0], []).append((ident, label))
        subs: dict[str, QMenu] = {}
        for ident, label in actions:
            if ident in ("add_from_library", "add_geometry"):     # submenus: materials, kinds of geometry
                make = library_menu if ident == "add_from_library" else geometry_menu
                sub = make(menu, lambda i, p=path: self._run_action(p, i))
                sub.setTitle(label.rstrip("…"))
                menu.addMenu(sub)
                continue
            prefix = ident.split(":", 1)[0]
            target = menu
            if len(groups.get(prefix, [])) > 1:
                if prefix not in subs:
                    subs[prefix] = menu.addMenu(group_label([lab for _, lab in groups[prefix]]).rstrip("…"))
                target = subs[prefix]
            act = target.addAction(label)
            act.setData(ident)
            act.triggered.connect(lambda checked=False, i=ident, p=path: self._run_action(p, i))
        return menu

    def _fill_add_menu(self):
        path = self.current_path()
        self._populate(self.add_button.menu(), path if path is not None else (), True)
        if not self.add_button.menu().actions():
            self.add_button.menu().addAction("(nothing to add here)").setEnabled(False)

    def _tree_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        path = item.data(0, PATH_ROLE)
        self.tree.setCurrentItem(item)
        menu = self._populate(QMenu(self.tree), path, False)
        if menu.actions():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _run_action(self, path, ident):
        if self.current_path() != path:
            self.select(path)
        self._panel_action(ident)

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
            if w is None or f.kind in ("info", "parameter_table", "material_table", "vertex_table", "chain_table"):
                continue
            msgs = E.errors_at(self.errors, f.key) if f.op != "rename_selection" else []
            w.setStyleSheet(ERROR_STYLE if msgs else "")
            w.setToolTip("\n".join(msgs) if msgs else f.help)

    def _panel_action(self, ident: str):
        path = self.current_path()
        if ident.startswith("pick_plane:"):   # the next click on a face gives the plane
            self._plane_target = path
            self._plane_kind = ident.split(":", 1)[1]
            self.update_view()
            self.state.setText("Click the plane face to put the extrusion on (Escape: cancel)")
            return
        if ident == "cmd:build_mesh":
            self.build_mesh(E.get_at(self.model, path + ("name",)))
            return
        if ident == "cmd:compute":
            self.compute()
            return
        if ident == "cmd:export_mesh":
            self.export_mesh(E.get_at(self.model, path + ("name",)))
            return
        if ident == "cmd:export_geometry":
            self.export_geometry(E.get_at(self.model, path + ("name",)))
            return
        if ident == "cmd:save_material":
            from .. import library
            mat = E.get_at(self.model, path)
            where = library.save_to_user(mat)
            self._log(f"Material {mat.get('name')!r} saved in {where}")
            self.state.setText(f"Saved in your library ({where})")
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
        new_path = E.selected_after(ident, new_path)   # (a new union, a new profile: there, to fill)
        self._undo.append(before)
        self._redo.clear()
        self._update_undo()
        self._changed(select=new_path if new_path is not None else path)

    def _remove_property(self, f: E.Field):
        self._snapshot()
        E.remove_at(self.model, f.remove)
        self._changed(rebuild_panel=True)

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
            name = E.mesh_to_show(self.model, self.current_path())
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
        self._read_result_files(res.workdir)
        self.results_model = _results_key(self.jobs.model)
        self.results_stale = _results_key(self.model) != self.results_model
        (res.workdir / "model_used.json").write_text(json.dumps(self.results_model, indent=1),
                                                     encoding="utf-8")
        solver = r.get("solver", {})
        self._log(f"Solved in {r.get('time_s', 0):.2f} s ({solver.get('type', '')}"
                  + (f", {solver.get('iterations')} iterations" if solver.get("iterations") else "") + ")")
        for w in r.get("warnings", []):
            self._log(f"Warning: {w}", error=True)
        for mname, mr in r.get("models", {}).items():
            res_ = mr.get("relative_residual")
            self._log(f"Model {mname}: {mr.get('nb_dofs')} unknowns"
                      + (f", relative residual {res_:.1e}" if res_ is not None else "")
                      + (f", {mr.get('iterations')} Newton iterations" if res_ is None else ""))
            for w in mr.get("warnings", []):
                self._log(f"Warning ({mname}): {w}", error=True)
            for k, v in mr.get("quantities", {}).items():
                self._log(f"  {k} = {v:.6g}")
        self.state.setText("Solved")
        target = self._result_target((E.RESULTS,))
        self.refresh(select=(E.RESULTS,) + target if target and target[1] else (E.RESULTS,))

    def _read_result_files(self, workdir: Path) -> None:
        """The VTU file of each model and its fields."""
        self.result_files, self.result_fields = {}, {}
        for mname, mr in (self.results or {}).get("models", {}).items():
            if not mr.get("vtu"):
                continue
            file = workdir / mr["vtu"]
            self.result_files[mname] = file
            self.result_fields[mname] = self._read_fields(file)

    def _read_fields(self, file: Path) -> dict[str, int]:
        if not file.exists():
            return {}
        if self.viewer.available:
            from . import scene
            try:
                return scene.field_names(scene.read(file))
            except Exception:  # noqa: BLE001
                return {}
        return _vtu_field_names(file)

    def _load_previous_results(self):
        """Results of an earlier session (marked out of date if the model changed since)."""
        self.results, self.result_files, self.result_fields = None, {}, {}
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
        self._read_result_files(self.workdir)

    def export_script(self, path: str | Path | None = None) -> bool:
        """Write the standalone script (meshing then solving) of the model."""
        from .. import runner
        if not self._check() or not self._check_gwfl():
            return False
        if path is None:
            stem = self.path.stem if self.path else "model"
            start = str((self.path.parent if self.path else Path.cwd()) / f"{stem}_getfem.py")
            path, _ = QFileDialog.getSaveFileName(self, "Export as a Python script", start,
                                                  "Python scripts (*.py)")
            if not path:
                return False
        try:
            Path(path).write_text(runner.export_script(self.model), encoding="utf-8")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, APP_NAME, f"Cannot export the script:\n{exc}")
            return False
        self._log(f"Script written to {path} (run it with a Python that provides GetFEM)")
        self.state.setText(f"Script written to {path}")
        return True

    def _selected_geometry(self) -> str | None:
        path = self.current_path()
        gi = E.preview_geometry(self.model, path) if path else None
        if gi is None and path[:1] == ("meshes",) and len(path) >= 2:
            g = M.find_geometry(self.model, self.model["meshes"][path[1]].get("geometry"))
            return g["name"] if g else None
        geoms = self.model.get("geometries", [])
        if gi is None:
            return geoms[0]["name"] if geoms else None
        return geoms[gi]["name"]

    def _selected_mesh(self) -> str | None:
        path = self.current_path() or ()
        meshes = self.model.get("meshes", [])
        if path[:1] == ("meshes",) and len(path) >= 2:
            return meshes[path[1]]["name"]
        md = E.owner(self.model, path)
        if md is not None:
            mesh = M.model_mesh(self.model, md)
            if mesh:
                return mesh.get("name")
        if path[:1] == ("geometries",) and len(path) >= 2:
            gname = self.model["geometries"][path[1]]["name"]
            for m in meshes:
                if m.get("geometry") == gname:
                    return m["name"]
        return meshes[0]["name"] if meshes else None

    def _export_target(self, title: str, stem: str, formats: dict, path) -> str | None:
        from .. import export
        if path is not None:
            return str(path)
        start = str((self.path.parent if self.path else Path.cwd()) / f"{stem}{next(iter(formats))}")
        path, _ = QFileDialog.getSaveFileName(self, title, start, export.file_filters(formats))
        return path or None

    def export_geometry(self, name: str | None = None, path=None) -> bool:
        """Write a geometry (that of the selected node by default) in another format."""
        from .. import export
        name = name or self._selected_geometry()
        geom = M.find_geometry(self.model, name) if name else None
        if geom is None or geom.get("source", {}).get("type") not in ("box", "csg"):
            QMessageBox.information(self, APP_NAME, "Select a box or constructive (csg) geometry to export.")
            return False
        src = geom["source"]
        dim = len(src["size"]) if src["type"] == "box" else src.get("dim", 2)
        formats = export.geometry_formats(dim)
        path = self._export_target(f"Export the geometry {name!r}", name, formats, path)
        if not path:
            return False
        if Path(path).suffix.lower() not in formats:
            path = str(Path(path).with_suffix(next(iter(formats))))
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            export.export_geometry(self.model, name, path, python=self.jobs.python)
        except export.ExportError as exc:
            QMessageBox.critical(self, APP_NAME, f"Cannot export the geometry:\n{exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._log(f"Geometry {name!r} written to {path}")
        self.state.setText(f"Geometry {name!r} written to {path}")
        return True

    def export_mesh(self, name: str | None = None, path=None, build: bool | None = None) -> bool:
        """Write a mesh (that of the selected node by default) in another format;
        a mesh not built (or not up to date) is built first, if accepted."""
        from .. import export
        name = name or self._selected_mesh()
        if name is None or M.find_mesh(self.model, name) is None:
            QMessageBox.information(self, APP_NAME, "Select a mesh to export.")
            return False
        if not self._check():
            return False
        if mesher.cached_mesh(self.model, name, self.workdir) is None:
            if build is None:
                build = QMessageBox.question(
                    self, APP_NAME, f"The mesh {name!r} is not built or not up to date. Build it now?") \
                    == QMessageBox.StandardButton.Yes
            if not build:
                return False
        path = self._export_target(f"Export the mesh {name!r}", name, export.MESH_FORMATS, path)
        if not path:
            return False
        if Path(path).suffix.lower() not in export.MESH_FORMATS:
            path = str(Path(path).with_suffix(".msh"))
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            export.export_mesh(self.model, name, self.workdir, path, python=self.jobs.python, build=True)
        except export.ExportError as exc:
            QMessageBox.critical(self, APP_NAME, f"Cannot export the mesh:\n{exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._log(f"Mesh {name!r} written to {path}")
        self.state.setText(f"Mesh {name!r} written to {path}")
        return True

    def export_view(self, path=None, mode: str | None = None, dpi: int | None = None) -> bool:
        """Write the view as vector graphics (PDF, SVG, EPS) or as an image
        (PNG, JPEG, TIFF, or PDF with the image): without path, the format,
        the kind and the density chosen in a dialog, then the file."""
        v = self.viewer
        if v.plotter is None:
            QMessageBox.information(self, APP_NAME, "No 3D view to export.")
            return False
        if path is None:
            from .export_dialog import FORMATS, ExportDialog
            last = getattr(self, "_export_choice", (".pdf", "vector", 300))
            dlg = ExportDialog(self, tuple(v.plotter.window_size), v.vector_problem(), *last)
            if not self.run_export_dialog(dlg):
                return False
            ext, mode, dpi = dlg.ext(), dlg.mode(), dlg.dpi()
            self._export_choice = (ext, mode, dpi)
            name = next(n for e, n, _ in FORMATS if e == ext)
            stem = re.sub(r"[^\w.-]+", "_", v.status.text().split(" — ")[0])[:60].strip("_") or "view"
            start = str((self.path.parent if self.path else Path.cwd()) / f"{stem}{ext}")
            path, _ = QFileDialog.getSaveFileName(self, "Export the view", start, f"{name} (*{ext})")
            if not path:
                return False
            if Path(path).suffix.lower() != ext:
                path += ext
        ext = Path(path).suffix.lower()
        if mode is None:
            mode = "vector" if ext in v.VECTOR_FORMATS else "image"
        dpi = int(dpi or 300)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            what = v.export_view(path, mode=mode, dpi=dpi)
        except (ValueError, OSError, RuntimeError) as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, APP_NAME, f"Cannot export the view:\n{exc}")
            return False
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
        note = f" (an image at {dpi} dpi)" if what == "image" else ""
        self._log(f"View written to {path}{note}")
        self.state.setText(f"View written to {path}{note}")
        return True

    def run_export_dialog(self, dlg) -> bool:
        """The dialog of the export run (replaced by the tests)."""
        return dlg.exec() == dlg.DialogCode.Accepted

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
        from .. import DESCRIPTION
        from .splash import title
        version = self.versions.version if self.versions is not None else None
        QMessageBox.about(self, APP_NAME,
                          f"<b>{title(version)}</b><p>A graphical interface for "
                          f"<a href='https://getfem.org'>GetFEM</a>, {DESCRIPTION}."
                          "<p>Distributed under the GNU LGPL, like GetFEM "
                          "(<i>Help ▸ License</i>).")

    def license_info(self) -> str:
        """The license notice, in a dialog; returns its text."""
        from .. import LICENSE
        import html
        text = html.escape(LICENSE).replace("\n\n", "<p>").replace("\n", " ")
        text = text.replace("https://www.gnu.org/licenses/",
                            "<a href='https://www.gnu.org/licenses/'>https://www.gnu.org/licenses/</a>")
        box = QMessageBox(QMessageBox.Icon.NoIcon, f"{APP_NAME}: license", text, QMessageBox.StandardButton.Close,
                          self)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setObjectName("license")
        box.exec()
        return LICENSE


def _results_key(model: dict) -> dict:
    """The part of the model that determines the results (for staleness checks),
    as it reads back from JSON."""
    m = {k: v for k, v in model.items() if not k.startswith("_") and k not in ("name", "graphics")}
    m["models"] = [{k: v for k, v in md.items() if k != "graphics"} for md in m.get("models", [])]
    return json.loads(json.dumps(m))


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
