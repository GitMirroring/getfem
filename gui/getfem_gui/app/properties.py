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


"""Properties panel: the form of the selected node, built from the field
descriptions of `editing.fields`. Every change is reported through the
`changed(field, value)` signal; the window applies it to the model."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
                               QToolButton, QVBoxLayout, QWidget)

from .. import editing as E, model as M

ERROR_STYLE = "QLineEdit, QComboBox { background: #fde8e4; }"
AXES = "xyz"


def _mono() -> QFont:
    f = QFont("monospace")
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


class PropertiesPanel(QScrollArea):
    changed = Signal(object, object)          # Field, value
    action = Signal(str)                      # editing action identifier, or a window command
    removeProperty = Signal(object)           # Field
    viewChanged = Signal(dict)                # settings of a result field display

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(280)
        self._fields: dict[tuple, QWidget] = {}
        self._build_empty()

    def _build_empty(self):
        # The old form is deleted later, not now: the change being applied may
        # come from one of its widgets (a combo box whose signal is still
        # running), and deleting it inside its own signal crashes Qt.
        old = self.takeWidget()
        if old is not None:
            old.hide()
            old.deleteLater()
        self.body = QWidget()
        self.layout_ = QVBoxLayout(self.body)
        self.layout_.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidget(self.body)
        self._fields = {}

    def _title(self, text: str):
        t = QLabel(text)
        f = t.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        t.setFont(f)
        t.setWordWrap(True)
        self.layout_.addWidget(t)

    def _messages(self, errors: list[str]):
        for e in errors:
            lab = QLabel("⚠ " + e)
            lab.setWordWrap(True)
            lab.setStyleSheet("color: #b3261e;")
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.layout_.addWidget(lab)

    def _buttons(self, actions: list[tuple[str, str]]):
        if not actions:
            return
        row = QHBoxLayout()
        row.setContentsMargins(0, 8, 0, 0)
        for ident, label in actions:
            b = QPushButton(label)
            b.setObjectName(f"action:{ident}")
            b.clicked.connect(lambda _=False, i=ident: self.action.emit(i))
            row.addWidget(b)
        row.addStretch(1)
        w = QWidget()
        w.setLayout(row)
        self.layout_.addWidget(w)

    # -------------------------------------------------------------- model nodes

    def show_node(self, model: dict, title: str, path, fields: list[E.Field], errors: list[str],
                  node_errors: list[str], actions: list[tuple[str, str]], base_dir: Path | None):
        self._build_empty()
        self._base_dir = base_dir
        self._title(title)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for f in fields:
            if f.kind == "info":
                lab = QLabel(f.help)
                lab.setWordWrap(True)
                lab.setStyleSheet("color: #555;")
                form.addRow(lab)
                continue
            if f.kind == "parameter_table":
                form.addRow(self._parameter_table(model, f, errors))
                continue
            widget = self._editor(model, f)
            msgs = E.errors_at(errors, f.key) if f.op != "rename_selection" else []
            if msgs:
                widget.setStyleSheet(ERROR_STYLE)
                widget.setToolTip("\n".join(msgs))
            elif f.help:
                widget.setToolTip(f.help)
            if f.removable:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(widget, 1)
                rm = QToolButton()
                rm.setText("✕")
                rm.setToolTip(f"Remove {f.label}")
                rm.setObjectName(f"remove:{'.'.join(map(str, f.remove))}")
                rm.clicked.connect(lambda _=False, fld=f: self.removeProperty.emit(fld))
                row.addWidget(rm)
                holder = QWidget()
                holder.setLayout(row)
                form.addRow(f.label, holder)
            else:
                form.addRow(f.label, widget)
            self._fields[f.key] = widget
        holder = QWidget()
        holder.setLayout(form)
        self.layout_.addWidget(holder)
        self._messages([e.partition(": ")[2] or e for e in node_errors])
        self._buttons(actions)

    def _parameter_table(self, model, f: E.Field, errors) -> QWidget:
        """All the parameters of a list in one table: name, value (number or
        expression), its number, description; rows added, removed, moved."""
        rows = E.parameter_rows(model, f.key)
        table = QTableWidget(len(rows), 4)
        table.setObjectName("parameters_table")
        table.setHorizontalHeaderLabels(["Name", "Value", "=", "Description"])
        table.verticalHeader().hide()
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        for i, r in enumerate(rows):
            texts = (r["name"] or "", E.format_value(r["value"]),
                     "" if r["number"] is None else E.format_number(float(r["number"])), r["description"] or "")
            for j, t in enumerate(texts):
                item = QTableWidgetItem(t)
                if j == 1:
                    item.setFont(_mono())
                if j == 2:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setForeground(Qt.GlobalColor.darkGray)
                    item.setToolTip("value of the expression")
                if j != 2:
                    fld = r["fields"][0 if j == 0 else 1 if j == 1 else 2]
                    msgs = E.errors_at(errors, fld.key)
                    if msgs:
                        item.setBackground(Qt.GlobalColor.yellow)
                        item.setToolTip("\n".join(msgs))
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

        def edited(item):
            i, j = item.row(), item.column()
            if j == 2 or i >= len(rows):
                return
            fld = rows[i]["fields"][0 if j == 0 else 1 if j == 1 else 2]
            text = item.text().strip()
            value = E.parse_value(text) if j == 1 else text
            if value == (rows[i]["name"], rows[i]["value"], None, rows[i]["description"])[j]:
                return
            self._last_cell = (f.key, i, j)
            self._commit(fld, value)
        table.itemChanged.connect(edited)
        # the cell edited last keeps the focus when the form is rebuilt
        last = getattr(self, "_last_cell", None)
        if last and last[0] == f.key and last[1] < len(rows):
            table.setCurrentCell(last[1], last[2])
            QTimer.singleShot(0, table.setFocus)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        for ident, text, tip in (("remove_row", "Remove", "Remove the selected parameter"),
                                 ("move_up", "Up", "Move up (a parameter may only use the ones above it)"),
                                 ("move_down", "Down", "Move down")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setObjectName(f"table:{ident}")
            b.clicked.connect(lambda _=False, v=ident: table.currentRow() >= 0
                              and self._row_action(f.key, v, table.currentRow()))
            bar.addWidget(b)
        bar.addStretch(1)
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(table)
        col.addLayout(bar)
        table.setMinimumHeight(min(80 + 30 * len(rows), 420))
        self._fields[f.key] = table
        return holder

    def _row_action(self, key, verb, row):
        target = {"move_up": row - 1, "move_down": row + 1}.get(verb, row)
        self._last_cell = (key, max(target, 0), 0)
        self.action.emit(f"{verb}:{row}")

    def editor(self, key) -> QWidget | None:
        """Editing widget of a field (for tests)."""
        return self._fields.get(tuple(key))

    def _commit(self, f: E.Field, value):
        self.changed.emit(f, value)

    def _line(self, text: str, f: E.Field, parse, mono=False) -> QLineEdit:
        edit = QLineEdit(text)
        if mono:
            edit.setFont(_mono())

        def done():
            if edit.isModified():
                edit.setModified(False)
                try:
                    value = parse(edit.text())
                except ValueError as exc:
                    edit.setStyleSheet(ERROR_STYLE)
                    edit.setToolTip(str(exc))
                    return
                self._commit(f, value)
        edit.editingFinished.connect(done)
        return edit

    def _components(self, values: list, f: E.Field, parse) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        edits = []

        def done():
            if not any(e.isModified() for e in edits):
                return
            for e in edits:
                e.setModified(False)
            try:
                value = [parse(e.text()) for e in edits]
            except ValueError as exc:
                holder.setStyleSheet(ERROR_STYLE)
                holder.setToolTip(str(exc))
                return
            self._commit(f, value)
        for i, v in enumerate(values):
            e = QLineEdit(E.format_value(v))
            e.setPlaceholderText(AXES[i] if i < 3 else str(i))
            e.setToolTip(AXES[i] if i < 3 else str(i))
            e.editingFinished.connect(done)
            row.addWidget(e)
            edits.append(e)
        holder = QWidget()
        holder.setLayout(row)
        holder.edits = edits
        return holder

    def _editor(self, model: dict, f: E.Field) -> QWidget:
        value = E.field_value(model, f)
        k = f.kind
        if f.op == "rename_selection":
            value = f.key[-1]
        if k in ("text", "ident"):
            return self._line("" if value is None else str(value), f, lambda s: s.strip())
        if k == "value":
            return self._line(E.format_value(value), f, E.parse_value, mono=True)
        if k == "number":
            return self._line(E.format_value(value), f, E.parse_number)
        if k == "int":
            return self._line(E.format_value(value), f, lambda s: E.parse_number(s, True))
        if k in ("numbers", "ints"):
            vals = list(value) if isinstance(value, list) else []
            vals = (vals + [0] * (f.size or len(vals)))[:f.size or len(vals)]
            return self._components(vals, f, lambda s, i=(k == "ints"): E.parse_number(s, i))
        if k == "vector":
            parts = E.split_vector(value, f.size or 1)
            if parts is None:
                return self._line(E.format_value(value), f, E.parse_value, mono=True)
            return self._components(parts, f, E.parse_value)
        if k == "choice":
            combo = QComboBox()
            values = [c[0] for c in f.choices]
            for v, label in f.choices:
                combo.addItem(label, v)
            if value not in values:
                combo.addItem(f"{value} (invalid)" if value is not None else "", value)
            combo.setCurrentIndex(values.index(value) if value in values else combo.count() - 1)
            combo.activated.connect(lambda i, c=combo: c.itemData(i) != value
                                    and self._commit(f, c.itemData(i)))
            return combo
        if k == "names":
            lst = QListWidget()
            chosen = list(value or [])
            for v, label in f.choices:
                it = QListWidgetItem(label)
                it.setData(Qt.ItemDataRole.UserRole, v)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if v in chosen else Qt.CheckState.Unchecked)
                lst.addItem(it)
            lst.setMaximumHeight(22 * max(3, min(len(f.choices), 8)))

            def changed(_item):
                self._commit(f, [lst.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lst.count())
                                 if lst.item(i).checkState() == Qt.CheckState.Checked])
            lst.itemChanged.connect(changed)
            return lst
        if k == "file":
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            edit = self._line("" if value is None else str(value), f, lambda s: s.strip())
            row.addWidget(edit, 1)
            b = QToolButton()
            b.setText("…")

            def browse():
                start = str(self._base_dir or Path.cwd())
                name, _ = QFileDialog.getOpenFileName(self, "Choose a file", start)
                if name:
                    p = Path(name)
                    if self._base_dir is not None:
                        try:
                            p = p.relative_to(self._base_dir)
                        except ValueError:
                            pass
                    self._commit(f, str(p))
            b.clicked.connect(browse)
            row.addWidget(b)
            holder = QWidget()
            holder.setLayout(row)
            return holder
        return QLabel(E.format_value(value))

    # -------------------------------------------------------------- results

    def show_results(self, results: dict, stale: bool, actions):
        self._build_empty()
        self._title("Results" + (" (out of date)" if stale else ""))
        rows = [(k, results[k]) for k in ("status", "nb_dofs", "time_s", "relative_residual")
                if k in results]
        rows += [(k, v) for k, v in results.items()
                 if (k.startswith("max_") or k.startswith("min_")) and isinstance(v, (int, float))]
        rows += [(f"∫ {k}", v) for k, v in results.get("quantities", {}).items()]
        table = QTableWidget(len(rows), 2)
        table.setObjectName("results_table")
        table.setHorizontalHeaderLabels(["Quantity", "Value"])
        table.verticalHeader().hide()
        for i, (k, v) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(k)))
            text = f"{v:.6g}" if isinstance(v, float) else str(v)
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(i, 1, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        self.layout_.addWidget(table)
        self._messages(results.get("warnings", []))
        self._buttons(actions)

    def show_result_field(self, name: str, ncomp: int, settings: dict, can_warp: bool,
                          vectors: tuple = (), planar: bool = True, graphic: dict | None = None,
                          field_names: tuple = ()):
        """Display settings of a result field: component, deformed shape,
        colors, isolines, arrows (of one of the vector fields), cut line.
        graphic: a saved graphic, whose name and field are also edited."""
        self._build_empty()
        self._title(f"Graphic: {graphic.get('name')}" if graphic is not None else f"Result: {name}")
        form = QFormLayout()
        if graphic is not None:
            gname = QLineEdit(graphic.get("name", ""))
            gname.setObjectName("graphic_name")
            gname.editingFinished.connect(
                lambda: gname.text().strip() != graphic.get("name")
                and self.viewChanged.emit({"name": gname.text().strip()}))
            form.addRow("Name", gname)
            fcombo = QComboBox()
            fcombo.setObjectName("graphic_field")
            for n in dict.fromkeys(list(field_names) + [graphic.get("field")]):
                fcombo.addItem(n, n)
            fcombo.setCurrentIndex(fcombo.findData(graphic.get("field")))
            fcombo.activated.connect(lambda i: fcombo.itemData(i) != graphic.get("field")
                                     and self.viewChanged.emit({"field": fcombo.itemData(i)}))
            form.addRow("Field", fcombo)
        if ncomp > 1:
            combo = QComboBox()
            combo.setObjectName("component")
            combo.addItem("magnitude", None)
            for i in range(ncomp):
                combo.addItem(f"component {AXES[i] if i < 3 else i}", i)
            combo.setCurrentIndex(combo.findData(settings.get("component")))
            combo.activated.connect(lambda i: self.viewChanged.emit({"component": combo.itemData(i)}))
            form.addRow("Show", combo)
        if can_warp:
            check = QCheckBox("deformed shape")
            check.setObjectName("warp")
            check.setChecked(settings.get("warp", True))
            check.toggled.connect(lambda on: self.viewChanged.emit({"warp": on}))
            form.addRow("", check)
            spin = QDoubleSpinBox()
            spin.setObjectName("scale")
            spin.setDecimals(3)
            spin.setRange(0.0, 1e12)
            spin.setValue(settings.get("scale") or 1.0)
            spin.setToolTip("Amplification of the displacement (the default makes the largest "
                            "displacement 10% of the model size)")
            spin.editingFinished.connect(lambda: self.viewChanged.emit({"scale": spin.value()}))
            form.addRow("Scale factor", spin)
        self._view_fields(form, settings, vectors, planar)
        holder = QWidget()
        holder.setLayout(form)
        self.layout_.addWidget(holder)
        self.layout_.addWidget(self._hint(
            ("Click on the field to read its value; the Cut line tool of the view (2D) "
             "draws a segment along which the field is plotted. " if planar else "")
            + ("These settings are saved in the graphic." if graphic is not None else
               "Changing these settings saves them in a graphic of the model, named after the field.")))

    def _hint(self, text):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #666;")
        return label

    def _view_fields(self, form, settings, vectors, planar):
        cmap = QComboBox()
        cmap.setObjectName("cmap")
        for c in M.COLORMAPS:
            cmap.addItem(c, c)
        cmap.setCurrentIndex(max(cmap.findData(settings.get("cmap") or M.COLORMAPS[0]), 0))
        cmap.activated.connect(lambda i: self.viewChanged.emit({"cmap": cmap.itemData(i)}))
        form.addRow("Colormap", cmap)
        rng = settings.get("range") or [None, None]
        row = QHBoxLayout()
        edits = []
        for j, tip in enumerate(("minimum", "maximum")):
            e = QLineEdit("" if rng[j] is None else f"{rng[j]:.6g}")
            e.setObjectName(f"range_{tip[:3]}")
            e.setPlaceholderText("auto")
            e.setToolTip(f"Color range {tip} (empty: the {tip} of the values)")
            row.addWidget(e)
            edits.append(e)

        def range_changed():
            vals = []
            for e in edits:
                try:
                    vals.append(float(e.text()) if e.text().strip() else None)
                except ValueError:
                    e.setText("")
                    vals.append(None)
            self.viewChanged.emit({"range": None if vals == [None, None] else vals})
        for e in edits:
            e.editingFinished.connect(range_changed)
        holder = QWidget()
        holder.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Range", holder)
        iso = QSpinBox()
        iso.setObjectName("isolines")
        iso.setRange(0, 100)
        iso.setValue(int(settings.get("isolines") or 0))
        iso.setSpecialValueText("none")
        iso.valueChanged.connect(lambda v: self.viewChanged.emit({"isolines": v}))
        form.addRow("Isolines", iso)
        bands = QCheckBox("color bands")
        bands.setObjectName("bands")
        bands.setChecked(bool(settings.get("bands")))
        bands.toggled.connect(lambda on: self.viewChanged.emit({"bands": on}))
        form.addRow("", bands)
        if vectors:
            arrows = QComboBox()
            arrows.setObjectName("arrows")
            arrows.addItem("none", None)
            for v in vectors:
                arrows.addItem(v, v)
            arrows.setCurrentIndex(max(arrows.findData(settings.get("arrows")), 0))
            arrows.activated.connect(lambda i: self.viewChanged.emit({"arrows": arrows.itemData(i)}))
            form.addRow("Arrows", arrows)
        if planar:
            cut = settings.get("cut")
            line = QLineEdit("; ".join(", ".join(f"{c:.6g}" for c in p) for p in cut) if cut else "")
            line.setObjectName("cut")
            line.setPlaceholderText("xA, yA; xB, yB")
            line.setToolTip("Segment along which the field is plotted (or draw it with the "
                            "Cut line tool of the view); empty: none")

            def cut_changed():
                text = line.text().strip()
                if not text:
                    self.viewChanged.emit({"cut": None})
                    return
                try:
                    pts = [[float(c) for c in part.split(",")] for part in text.split(";")]
                    if len(pts) != 2 or any(len(p) != 2 for p in pts) or pts[0] == pts[1]:
                        raise ValueError
                except ValueError:
                    line.setStyleSheet("background: #fde2e1;")
                    return
                line.setStyleSheet("")
                self.viewChanged.emit({"cut": pts})
            line.editingFinished.connect(cut_changed)
            form.addRow("Cut line", line)
