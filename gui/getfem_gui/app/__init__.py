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

    # The start-up window must be on screen before the long part of the
    # loading: the 3D libraries are imported in a background thread (they do
    # not touch Qt) while the event loop shows the start-up window, and the
    # main window is built from the loop when both are done. DURATION_MS counts
    # from the first paint of the start-up window (some window systems, macOS
    # in particular, show a window only after several turns of the loop). The
    # main window is shown when the start-up window closes: shown earlier, it
    # would cover it (a window cannot stay on top on Wayland).
    import threading
    screen = Splash(versions)
    clock = QElapsedTimer()
    loader = threading.Thread(target=_preload_3d, daemon=True)
    state = {"painted": False, "built": False}

    def painted():
        if not state["painted"]:
            state["painted"] = True
            clock.start()

    def try_build():
        if state["built"]:
            return
        if not state["painted"] or loader.is_alive():
            QTimer.singleShot(30, try_build)
            return
        state["built"] = True
        win = build()

        def done():
            if not win.isVisible():
                win.show()
                screen.close()
        screen.clicked.connect(done)
        QTimer.singleShot(max(0, DURATION_MS - clock.elapsed()), done)

    screen.painted.connect(painted)
    screen.show()
    screen.raise_()
    loader.start()
    QTimer.singleShot(30, try_build)
    return app.exec()


def _preload_3d() -> None:
    """Import the 3D libraries (the slow part of the start); errors are left to
    the 3D view, which reports them."""
    import os
    if os.environ.get("GETFEM_GUI_NO_3D"):
        return
    try:
        from .viewer import _quiet_matplotlib
        _quiet_matplotlib()
        import pyvista  # noqa: F401
        import pyvista.plotting  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
