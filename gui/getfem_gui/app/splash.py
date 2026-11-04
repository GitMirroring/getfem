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


"""Start-up window, shown for about two seconds while the main window loads,
and the version of GetFEM (asked to the interpreter that runs the
computations, which may not be the one running the interface)."""

from __future__ import annotations

import sys

from PySide6.QtCore import QObject, QProcess, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import QSplashScreen

from .. import DESCRIPTION, __version__

DURATION_MS = 2000

# Version of GetFEM: util_version() since GetFEM 5.5; package metadata otherwise
_VERSION_SCRIPT = """
import getfem
try:
    print(getfem.util_version())
except AttributeError:
    from importlib.metadata import version
    print(version("getfem"))
"""


class GetfemVersion(QObject):
    """Asks the version of GetFEM in the background; `found(text)` ("" if unknown)."""
    found = Signal(str)

    def __init__(self, python: str | None = None, parent=None):
        super().__init__(parent)
        self.version: str | None = None
        self._proc = QProcess(self)
        self._proc.finished.connect(self._done)
        self._proc.errorOccurred.connect(
            lambda err: err == QProcess.ProcessError.FailedToStart and self._done())
        self._proc.start(python or sys.executable, ["-c", _VERSION_SCRIPT])

    def _done(self, *args):
        if self.version is not None:
            return
        try:
            out = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace").strip()
            ok = self._proc.exitStatus() == QProcess.ExitStatus.NormalExit and self._proc.exitCode() == 0
        except RuntimeError:
            return
        self.version = out.splitlines()[-1] if ok and out else ""
        self.found.emit(self.version)


def title(getfem_version: str | None) -> str:
    text = f"GetFEM GUI {__version__}"
    if getfem_version:
        text += f" on GetFEM {getfem_version}"
    return text


class Splash(QSplashScreen):
    W, H = 560, 300

    def __init__(self, versions: GetfemVersion | None = None):
        super().__init__(self._render(None))
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        if versions is not None:
            versions.found.connect(lambda v: self.setPixmap(self._render(v)))
            if versions.version is not None:
                self.setPixmap(self._render(versions.version))

    @classmethod
    def _render(cls, getfem_version: str | None) -> QPixmap:
        ratio = 2.0       # drawn at twice the size: sharp on high density screens
        pix = QPixmap(int(cls.W * ratio), int(cls.H * ratio))
        pix.setDevicePixelRatio(ratio)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        g = QLinearGradient(0, 0, cls.W, cls.H)
        g.setColorAt(0.0, QColor("#16324f"))
        g.setColorAt(1.0, QColor("#1f6f78"))
        p.fillRect(0, 0, cls.W, cls.H, g)
        # a few mesh-like triangles, as a discreet decoration
        p.setPen(QColor(255, 255, 255, 28))
        step = 40
        for i in range(-cls.H // step, cls.W // step + 2):
            x = i * step
            p.drawLine(x, cls.H, x + cls.H, 0)
            p.drawLine(x, 0, x + cls.H, cls.H)
        for y in range(0, cls.H + 1, step):
            p.drawLine(0, y, cls.W, y)

        def text(rect, size, string, bold=False, color="#ffffff", flags=Qt.AlignmentFlag.AlignLeft):
            f = QFont()
            f.setPointSizeF(size)
            f.setBold(bold)
            p.setFont(f)
            p.setPen(QColor(color))
            p.drawText(QRectF(*rect), int(flags | Qt.TextFlag.TextWordWrap), string)

        text((36, 48, cls.W - 72, 50), 26, "GetFEM GUI", bold=True)
        text((38, 100, cls.W - 72, 30), 12, title(getfem_version), color="#cfe8ea")
        text((38, 145, cls.W - 76, 70), 11.5,
             f"A graphical interface for GetFEM, {DESCRIPTION}.", color="#ffffff")
        text((38, cls.H - 48, cls.W - 76, 20), 9, "Distributed under the GNU LGPL  ·  getfem.org",
             color="#a9c7cc")
        p.end()
        return pix
