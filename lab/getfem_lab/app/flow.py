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


"""A layout that puts its widgets in a row, wrapped to the next lines when
the row is too narrow for them (the flow layout of the Qt examples)."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy


class FlowLayout(QLayout):
    """Widgets left to right, then on the next lines."""

    def __init__(self, parent=None, spacing: int = -1):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing

    def __del__(self):
        while self.count():
            self.takeAt(0)

    def addItem(self, item):  # noqa: N802 (Qt)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802 (Qt)
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802 (Qt)
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 (Qt)
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802 (Qt)
        return True

    def heightForWidth(self, width):  # noqa: N802 (Qt)
        return self._place(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):  # noqa: N802 (Qt)
        super().setGeometry(rect)
        self._place(rect, apply=True)

    def sizeHint(self):  # noqa: N802 (Qt)
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802 (Qt)
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _gap(self, item, orientation) -> int:
        if self._spacing >= 0:
            return self._spacing
        w = item.widget()
        style = w.style() if w is not None else None
        if style is None:
            return 6
        gap = style.layoutSpacing(QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton,
                                  orientation)
        return gap if gap >= 0 else 6             # (some styles give none)

    def _place(self, rect: QRect, apply: bool) -> int:
        """Put the items in rect (if apply); the height they take."""
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            dx = self._gap(item, Qt.Orientation.Horizontal)
            dy = self._gap(item, Qt.Orientation.Vertical)
            if x + hint.width() > area.right() + 1 and line > 0:     # the next line
                x, y, line = area.x(), y + line + dy, 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + dx
            line = max(line, hint.height())
        return y + line - rect.y() + m.bottom()


__all__ = ["FlowLayout"]
