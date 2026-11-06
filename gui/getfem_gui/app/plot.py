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


"""A small line plot drawn with QPainter (values of a field along a cut
line): no plotting library, so nothing more to install or to configure."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round values (1, 2 or 5 x 10^k steps) covering [lo, hi], about n of them."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    e = math.floor(math.log10(raw))
    step = next(m * 10.0 ** e for m in (1, 2, 5, 10) if m * 10.0 ** e >= raw)
    first = math.ceil(lo / step - 1e-9) * step
    return [round(first + i * step, 12) for i in range(int((hi - first) / step + 1e-9) + 1)]


class LinePlot(QWidget):
    COLOR = QColor("#d62728")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.xlabel = self.ylabel = ""

    def set_data(self, x, y, xlabel: str = "", ylabel: str = "") -> None:
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.xlabel, self.ylabel = xlabel, ylabel
        self.update()

    def ranges(self) -> tuple[float, float, float, float]:
        ok = np.isfinite(self.y)
        if not ok.any():
            return 0.0, 1.0, 0.0, 1.0
        x0, x1 = float(self.x.min()), float(self.x.max())
        y0, y1 = float(self.y[ok].min()), float(self.y[ok].max())
        if y1 - y0 <= 1e-12 * max(abs(y0), abs(y1), 1e-300):
            pad = 0.5 * (abs(y0) or 1.0)
        else:
            pad = 0.05 * (y1 - y0)
        return x0, x1 if x1 > x0 else x0 + 1.0, y0 - pad, y1 + pad

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("white"))
        font = QFont(self.font())
        font.setPointSizeF(max(font.pointSizeF() - 1, 7))
        p.setFont(font)
        fm = p.fontMetrics()
        x0, x1, y0, y1 = self.ranges()
        yt = nice_ticks(y0, y1, 4)
        labels = [f"{v:.4g}" for v in yt]
        left = max([fm.horizontalAdvance(t) for t in labels] + [20]) + 12 + fm.height()
        area = QRectF(left, 10, max(self.width() - left - 16, 10),
                      max(self.height() - 18 - 2 * fm.height(), 10))

        def X(v):
            return area.left() + (v - x0) / (x1 - x0) * area.width()

        def Y(v):
            return area.bottom() - (v - y0) / (y1 - y0) * area.height()
        grid, axis = QPen(QColor("#e3e6ea")), QPen(QColor("#555555"))
        for v, t in zip(yt, labels):
            p.setPen(grid)
            p.drawLine(QPointF(area.left(), Y(v)), QPointF(area.right(), Y(v)))
            p.setPen(axis)
            p.drawText(QRectF(0, Y(v) - fm.height() / 2, left - 6, fm.height()),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, t)
        for v in nice_ticks(x0, x1, 6):
            p.setPen(grid)
            p.drawLine(QPointF(X(v), area.top()), QPointF(X(v), area.bottom()))
            p.setPen(axis)
            p.drawText(QRectF(X(v) - 40, area.bottom() + 2, 80, fm.height()),
                       Qt.AlignmentFlag.AlignHCenter, f"{v:.4g}")
        p.setPen(axis)
        p.drawRect(area)
        p.drawText(QRectF(area.left(), area.bottom() + 2 + fm.height(), area.width(), fm.height()),
                   Qt.AlignmentFlag.AlignHCenter, self.xlabel)
        p.save()
        p.translate(fm.height() / 2 + 2, area.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-area.height() / 2, -fm.height() / 2, area.height(), fm.height()),
                   Qt.AlignmentFlag.AlignHCenter, self.ylabel)
        p.restore()
        path, drawing = QPainterPath(), False
        for xv, yv in zip(self.x, self.y):          # NaN (outside the domain): a gap
            if not math.isfinite(yv):
                drawing = False
                continue
            pt = QPointF(X(xv), Y(yv))
            path.lineTo(pt) if drawing else path.moveTo(pt)
            drawing = True
        p.setPen(QPen(self.COLOR, 2))
        p.drawPath(path)
        p.end()
