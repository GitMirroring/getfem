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


"""Graphical interface (PySide6 for the window, PyVista for the 3D view).

    python -m getfem_gui gui [model.json]      or    python -m getfem_gui.app [model.json]
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None, python: str | None = None, splash: bool = True) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        from PySide6.QtCore import QElapsedTimer, QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        # say which interpreter, and why: PySide6 may be installed for another
        # Python, or installed but not loadable (architecture, Python version)
        print(f"error: the graphical interface requires PySide6, which cannot be imported by\n"
              f"    {sys.executable} (Python {sys.version.split()[0]})\n"
              f"    {type(exc).__name__}: {exc}\n"
              f"Install it for this interpreter: {sys.executable} -m pip install pyside6 pyvista pyvistaqt",
              file=sys.stderr)
        return 1
    from .viewer import prepare_opengl

    prepare_opengl()
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("GetFEM GUI")
    app.setOrganizationName("GetFEM")
    app.setStyle("Fusion")

    from .splash import DURATION_MS, GetfemVersion, Splash
    versions = GetfemVersion(python)
    windows = []                        # keeps the main window alive

    def build():
        from .window import MainWindow  # loading the 3D view takes a moment
        windows.append(MainWindow(argv[0] if argv else None, python=python, versions=versions))
        return windows[-1]

    if not splash:
        build().show()
        return app.exec()

    # The start-up window is shown by the event loop: the main window is built
    # from the loop, once the start-up window is on screen (some window
    # systems, macOS in particular, need several turns of the loop to show a
    # window). It is shown when the start-up window closes, after
    # DURATION_MS at least: shown earlier, it would cover the start-up window
    # (a window cannot stay on top on Wayland).
    screen = Splash(versions)
    clock = QElapsedTimer()

    def start():
        win = build()

        def done():
            if not win.isVisible():
                win.show()
                screen.close()
        screen.clicked.connect(done)
        QTimer.singleShot(max(0, DURATION_MS - clock.elapsed()), done)

    screen.show()
    clock.start()                       # the DURATION_MS count from now
    QTimer.singleShot(150, start)       # after the first paints of the start-up window
    return app.exec()
