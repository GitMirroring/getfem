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

"""The dialog of File > Export the view...: the format, vector graphics or
an image, the pixel density of an image (with the size it gives), before
the choice of the file."""

from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QRadioButton, QSpinBox, QVBoxLayout, QWidget)

# (extension, name, vector possible)
FORMATS = [(".pdf", "PDF", True), (".svg", "SVG", True), (".eps", "EPS", True), (".ps", "PostScript", True),
           (".png", "PNG", False), (".jpg", "JPEG", False), (".tif", "TIFF", False)]


class ExportDialog(QDialog):
    """The choices of an export of the view: ext(), mode() ("vector" or
    "image"), dpi(). view_size: the size of the view in pixels; vector: why
    vector graphics are not possible for this view ("" if they are)."""

    def __init__(self, parent=None, view_size=(800, 600), vector_problem: str = "",
                 ext: str = ".pdf", mode: str = "vector", dpi: int = 300):
        super().__init__(parent)
        self.setWindowTitle("Export the view")
        self.view_size = view_size
        self.vector_problem = vector_problem
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        self.format = QComboBox()
        self.format.setObjectName("export_format")
        for e, name, _ in FORMATS:
            self.format.addItem(f"{name} (*{e})", e)
        form.addRow("Format:", self.format)
        kind = QWidget()
        row = QHBoxLayout(kind)
        row.setContentsMargins(0, 0, 0, 0)
        self.vector = QRadioButton("Vector graphics")
        self.vector.setObjectName("export_vector")
        self.image = QRadioButton("Image")
        self.image.setObjectName("export_image")
        group = QButtonGroup(self)
        group.addButton(self.vector)
        group.addButton(self.image)
        row.addWidget(self.vector)
        row.addWidget(self.image)
        row.addStretch(1)
        form.addRow("Content:", kind)
        self.density = QSpinBox()
        self.density.setObjectName("export_dpi")
        self.density.setRange(24, 2400)
        self.density.setSingleStep(50)
        self.density.setSuffix(" dpi")
        self.density.setToolTip("Pixels per inch, a pixel of the view counting for 1/96 inch")
        form.addRow("Pixel density:", self.density)
        self.size = QLabel()
        self.size.setObjectName("export_size")
        form.addRow("", self.size)
        self.note = QLabel()
        self.note.setObjectName("export_note")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #666666")
        lay.addWidget(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save...")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        i = self.format.findData(ext)
        self.format.setCurrentIndex(i if i >= 0 else 0)
        (self.vector if mode == "vector" else self.image).setChecked(True)
        self.density.setValue(int(dpi))
        self.format.currentIndexChanged.connect(self._update)
        self.vector.toggled.connect(self._update)
        self.density.valueChanged.connect(self._update)
        self._update()

    def ext(self) -> str:
        return self.format.currentData()

    def mode(self) -> str:
        return "vector" if self.vector.isChecked() and self.vector.isEnabled() else "image"

    def dpi(self) -> int:
        return int(self.density.value())

    def _update(self, *_):
        vector_format = next(v for e, _, v in FORMATS if e == self.ext())
        self.vector.setEnabled(vector_format and not self.vector_problem)
        if not self.vector.isEnabled() and self.vector.isChecked():
            self.image.setChecked(True)
        image = self.mode() == "image"
        self.density.setEnabled(image)
        w, h = self.view_size
        scale = max(1, round(self.dpi() / 96))
        if image:
            inches = f"{w / 96:.1f} × {h / 96:.1f} in" if vector_format else ""
            self.size.setText(f"{w * scale} × {h * scale} pixels" + (f", a page of {inches}" if inches else ""))
        else:
            self.size.setText("")
        if not vector_format:
            self.note.setText("")
        elif self.vector_problem:
            self.note.setText(f"Vector graphics: not possible here ({self.vector_problem}).")
        elif image:
            self.note.setText(f"The image of the view in a {self.format.currentText().split(' ')[0]} file.")
        else:
            self.note.setText("The polygons and lines of the view, sharp at any zoom.")
