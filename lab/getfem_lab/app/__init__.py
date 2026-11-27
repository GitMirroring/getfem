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

    getfem_lab [model.json]    or    python -m getfem_lab [model.json]    or    python -m getfem_lab.app
"""

from __future__ import annotations

import os
import sys


def _xcb_works(timeout: float = 10.0) -> bool:
    """Whether Qt can open its X11 (xcb) platform here (its plugin may need
    system libraries, libxcb-cursor0 for Qt >= 6.5): a short subprocess."""
    import subprocess
    code = "from PySide6.QtWidgets import QApplication; QApplication(['probe'])"
    env = dict(os.environ, QT_QPA_PLATFORM="xcb")
    try:
        return subprocess.run([sys.executable, "-c", code], env=env, timeout=timeout,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def choose_platform(env=None, probe=_xcb_works) -> str | None:
    """The Qt platform for the window: on Linux under Wayland, X11 (xcb,
    through XWayland) when it works. The 3D view is VTK's Qt widget, which
    renders in a window of the X server: under the wayland platform of Qt,
    VTK opens it apart and the frame of the interface stays empty.
    QT_QPA_PLATFORM chosen by the user, or GETFEM_LAB_WAYLAND set, keep
    Qt's choice. Sets QT_QPA_PLATFORM; returns the platform chosen (None:
    Qt's)."""
    env = os.environ if env is None else env
    if not sys.platform.startswith("linux") or env.get("QT_QPA_PLATFORM") or env.get("GETFEM_LAB_WAYLAND"):
        return None
    if not env.get("WAYLAND_DISPLAY") or not env.get("DISPLAY"):     # not Wayland, or no XWayland
        return None
    if not probe():
        print("warning: under Wayland, the 3D view needs Qt's X11 platform (xcb), which cannot be "
              "opened (on Ubuntu: sudo apt install libxcb-cursor0); it may open in a separate window",
              file=sys.stderr)
        return None
    env["QT_QPA_PLATFORM"] = "xcb"
    return "xcb"


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
              f"Install it for this interpreter: {sys.executable} -m pip install pyside6 pyvista",
              file=sys.stderr)
        return 1
    from .viewer import prepare_opengl

    if QApplication.instance() is None:
        choose_platform()               # Wayland: X11 (XWayland), for the 3D view inside the window
    from . import icon
    prepare_opengl()
    if QApplication.instance() is None:
        icon.set_macos_name()           # the name in the menu bar and the Dock (with pyobjc)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName(icon.NAME)   # also the class of the windows on X11 (StartupWMClass)
    app.setOrganizationName("GetFEM")
    app.setDesktopFileName(icon.DESKTOP_ID)     # the app_id of the windows on Wayland
    app.setWindowIcon(icon.app_icon())  # the Dock (macOS), the task bar, the windows
    app.setStyle("Fusion")

    from .splash import DURATION_MS, GetfemVersion, Splash
    versions = GetfemVersion(python)
    windows = []                        # keeps the main window alive

    def build():
        from .window import MainWindow  # loading the 3D view takes a moment
        windows.append(MainWindow(argv[0] if argv else None, python=python, versions=versions))
        return windows[-1]

    if not splash or os.environ.get("GETFEM_LAB_NO_SPLASH"):
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
        try:
            win = build()
        except BaseException:
            state["failed"] = True
            screen.close()                       # not left on screen: the error is reported
            app.exit(1)
            raise

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
    code = app.exec()
    return 1 if state.get("failed") else code


def _preload_3d() -> None:
    """Import the 3D libraries (the slow part of the start); errors are left to
    the 3D view, which reports them."""
    import os
    if os.environ.get("GETFEM_LAB_NO_3D"):
        return
    try:
        from .viewer import _quiet_matplotlib
        _quiet_matplotlib()
        import pyvista  # noqa: F401
        import pyvista.plotting  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
