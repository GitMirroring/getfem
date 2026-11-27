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


# the colors of the curves (Tableau 10), in turn
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
          "#bcbd22", "#17becf")


def as_floats(values) -> np.ndarray:
    """Values of a curve (None: NaN, a gap)."""
    return np.array([np.nan if v is None else v for v in values], float)


class LinePlot(QWidget):
    COLOR = QColor("#d62728")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.series = []                 # [(x, y, label)]
        self.xlabel = self.ylabel = self.title = ""

    @property
    def x(self):
        return self.series[0][0] if self.series else np.zeros(0)

    @property
    def y(self):
        return self.series[0][1] if self.series else np.zeros(0)

    def set_data(self, x, y, xlabel: str = "", ylabel: str = "") -> None:
        """One curve."""
        self.set_series([(x, y, "")], xlabel, ylabel)

    def set_series(self, series, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
        """Several curves [(x, y, label)] (a legend if labeled); y: None or
        NaN for a gap."""
        self.series = [(np.asarray(x, float), as_floats(y), label) for x, y, label in series]
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title
        self.update()

    def ranges(self) -> tuple[float, float, float, float]:
        xs = [x for x, y, _ in self.series if len(x)]
        ys = [y[np.isfinite(y)] for _, y, _ in self.series]
        ys = [y for y in ys if len(y)]
        if not xs or not ys:
            return 0.0, 1.0, 0.0, 1.0
        x0, x1 = float(min(x.min() for x in xs)), float(max(x.max() for x in xs))
        y0, y1 = float(min(y.min() for y in ys)), float(max(y.max() for y in ys))
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
        top = 10 + (fm.height() + 4 if self.title else 0)
        area = QRectF(left, top, max(self.width() - left - 16, 10),
                      max(self.height() - top - 8 - 2 * fm.height(), 10))
        if self.title:
            p.setPen(QPen(QColor("#333333")))
            p.drawText(QRectF(area.left(), 4, area.width(), fm.height() + 2), Qt.AlignmentFlag.AlignHCenter,
                       self.title)

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
        p.setClipRect(area.adjusted(-2, -2, 2, 2))
        single = len(self.series) == 1
        for k, (xs, ys, _label) in enumerate(self.series):
            color = self.COLOR if single else QColor(COLORS[k % len(COLORS)])
            path, drawing, alone = QPainterPath(), False, []
            for i, (xv, yv) in enumerate(zip(xs, ys)):        # NaN (outside the domain): a gap
                if not math.isfinite(yv):
                    drawing = False
                    continue
                pt = QPointF(X(xv), Y(yv))
                path.lineTo(pt) if drawing else path.moveTo(pt)
                isolated = not drawing and (i + 1 >= len(ys) or not math.isfinite(ys[i + 1]))
                if isolated:                          # (a single value: a dot)
                    alone.append(pt)
                drawing = True
            p.setPen(QPen(color, 2))
            p.drawPath(path)
            p.setBrush(color)
            for pt in alone:
                p.drawEllipse(pt, 3, 3)
            p.setBrush(Qt.BrushStyle.NoBrush)
        p.setClipping(False)
        labels = [label for _, _, label in self.series]
        if len(self.series) > 1 or any(labels):          # the legend, top right
            w = max(fm.horizontalAdvance(t or " ") for t in labels) + 34
            h = fm.height() * len(labels) + 6
            corners = [QRectF(area.right() - w - 6, area.top() + 6, w, h),
                       QRectF(area.left() + 6, area.top() + 6, w, h),
                       QRectF(area.right() - w - 6, area.bottom() - h - 6, w, h),
                       QRectF(area.left() + 6, area.bottom() - h - 6, w, h)]

            def covered(box):            # the points of the curves under a box (the fewest: its corner)
                return sum(int(box.contains(QPointF(X(xv), Y(yv)))) for xs, ys, _ in self.series
                           for xv, yv in zip(xs, ys) if math.isfinite(yv))
            box = min(corners, key=covered)
            p.setPen(QPen(QColor("#bbbbbb")))
            p.setBrush(QColor(255, 255, 255, 220))
            p.drawRect(box)
            for k, label in enumerate(labels):
                color = self.COLOR if single else QColor(COLORS[k % len(COLORS)])
                yk = box.top() + 3 + (k + 0.5) * fm.height()
                p.setPen(QPen(color, 2))
                p.drawLine(QPointF(box.left() + 5, yk), QPointF(box.left() + 25, yk))
                p.setPen(QPen(QColor("#333333")))
                p.drawText(QRectF(box.left() + 30, yk - fm.height() / 2, w - 32, fm.height()),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
        p.end()


class PlotPanel(QWidget):
    """A plot of the results, in place of the view: its curves, a legend,
    the values saved as a CSV file."""

    def __init__(self, parent=None):
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(4, 2, 4, 2)
        head = QHBoxLayout()
        self.caption = QLabel("")
        self.caption.setStyleSheet("color: #555;")
        head.addWidget(self.caption, 1)
        self.csv_button = QToolButton()
        self.csv_button.setText("CSV…")
        self.csv_button.setToolTip("Save the values of the curves (CSV file)")
        self.csv_button.clicked.connect(self._save_dialog)
        head.addWidget(self.csv_button)
        col.addLayout(head)
        self.plot = LinePlot()
        self.plot.setObjectName("results_plot")
        col.addWidget(self.plot, 1)
        self.data = None

    def show_plot(self, name: str, data: dict) -> None:
        """data: {'type', 'title', 'xlabel', 'curves': [{'label', 'x', 'y'}]}."""
        self.data = data
        curves = data.get("curves") or []
        self.plot.set_series([(c.get("x", []), c.get("y", []), c.get("label", "")) for c in curves if c],
                             data.get("xlabel", ""), "", data.get("title") or name)
        self.caption.setText(f"Plot {name}: {len([c for c in curves if c])} curve(s)"
                             + ("" if curves else " (none computed)"))

    def save_csv(self, path) -> None:
        """The curves, one after the other: label, x, y (empty outside the domain)."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"curve,{self.data.get('xlabel', 'x')},value\n")
            for c in self.data.get("curves") or []:
                if not c:
                    continue
                label = '"' + str(c.get("label", "")).replace('"', '""') + '"'
                for x, y in zip(c.get("x", []), c.get("y", [])):
                    f.write(f"{label},{x:.10g}," + ("" if y is None else f"{y:.10g}") + "\n")

    def _save_dialog(self):
        from PySide6.QtWidgets import QFileDialog
        if not self.data:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the curves", "plot.csv", "CSV files (*.csv)")
        if path:
            self.save_csv(path)
