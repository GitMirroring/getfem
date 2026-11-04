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

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QScrollArea, QTableWidget, QTableWidgetItem,
                               QToolButton, QVBoxLayout, QWidget)

from .. import editing as E

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
        value = E.get_at(model, f.key)
        k = f.kind
        if f.op == "rename_selection":
            value = f.key[-1]
        if f.op == "box_dim":
            value = len(value or [])
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

    def show_result_field(self, name: str, ncomp: int, settings: dict, can_warp: bool):
        """Display settings of a result field: component, deformed shape."""
        self._build_empty()
        self._title(f"Result: {name}")
        form = QFormLayout()
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
        holder = QWidget()
        holder.setLayout(form)
        self.layout_.addWidget(holder)
