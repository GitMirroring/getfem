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

"""The icon and the name of the application, for the bar of the running
applications (the Dock of macOS, the dash of GNOME, the task bar).

The icon (resources/icon_*.png, built by lab/tools/make_icon.py) is the icon
of every window: Qt gives it to the Dock on macOS and to the window manager
on X11. The dash of GNOME (Ubuntu) shows, for a window, the icon of the
desktop file whose StartupWMClass is the class of the window (X11) or whose
name is the app_id of the window (Wayland): install_desktop() writes one
(getfem_lab install-desktop), which also puts the interface among the
applications. This module does not import Qt (except app_icon()).
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent / "resources"
SIZES = (32, 64, 128, 256, 512)
NAME = "GetFEM Lab"            # the application name: the class of the windows on X11
DESKTOP_ID = "getfem-lab"      # the desktop file, and the app_id of the windows on Wayland


def icon_file(size: int = 512) -> Path:
    return RESOURCES / f"icon_{size}.png"


def app_icon():
    """The icon of the application, at all its sizes (a QIcon)."""
    from PySide6.QtGui import QIcon
    icon = QIcon()
    for s in SIZES:
        icon.addFile(str(icon_file(s)))
    return icon


def set_macos_name(name: str = NAME) -> bool:
    """On macOS, the name of the application in the menu bar and the Dock
    (otherwise "Python"), when pyobjc is installed; to be called before the
    QApplication is created. Returns whether it was set."""
    if sys.platform != "darwin":
        return False
    try:
        from Foundation import NSBundle     # pyobjc
    except ImportError:
        return False
    try:
        info = NSBundle.mainBundle().localizedInfoDictionary() or NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = name
        return True
    except Exception:  # noqa: BLE001
        return False


def _data_home(env) -> Path:
    return Path(env.get("XDG_DATA_HOME") or Path(env.get("HOME", "~")).expanduser() / ".local" / "share")


def launch_command(python: str | None = None, env=None) -> str:
    """The command which opens the interface (Exec of the desktop file): the
    present interpreter running this package, with the PYTHONPATH of now
    (getfem and the package itself, when they are not installed), and
    --python when given."""
    env = os.environ if env is None else env
    package_parent = str(Path(__file__).resolve().parents[2])
    paths = [package_parent] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != package_parent]
    cmd = ["env", "PYTHONPATH=" + os.pathsep.join(paths), sys.executable, "-m", "getfem_lab"]
    if python:
        cmd += ["--python", python]
    return " ".join(shlex.quote(c) for c in cmd) + " %f"


def desktop_entry(python: str | None = None, env=None) -> str:
    """The content of the desktop file (freedesktop.org) of the interface."""
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        f"Name={NAME}",
        "GenericName=Finite element modelling",
        "Comment=Graphical interface of GetFEM: geometries, meshes, models, results",
        f"Exec={launch_command(python, env)}",
        f"Icon={DESKTOP_ID}",
        "Terminal=false",
        "Categories=Science;Engineering;Education;",
        "MimeType=application/json;",
        f"StartupWMClass={NAME}",
        "StartupNotify=true",
        ""])


def install_desktop(python: str | None = None, env=None, uninstall: bool = False) -> list[Path]:
    """Writes (or removes) the desktop file of the interface and its icons in
    the user's data directory (~/.local/share/applications,
    ~/.local/share/icons/hicolor). Returns the files written (or removed)."""
    env = os.environ if env is None else env
    home = _data_home(env)
    desktop = home / "applications" / f"{DESKTOP_ID}.desktop"
    icons = [home / "icons" / "hicolor" / f"{s}x{s}" / "apps" / f"{DESKTOP_ID}.png" for s in SIZES]
    files = [desktop] + icons
    if uninstall:
        removed = [f for f in files if f.exists()]
        for f in removed:
            f.unlink()
        return removed
    for s, f in zip(SIZES, icons):
        f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(icon_file(s), f)
    desktop.parent.mkdir(parents=True, exist_ok=True)
    desktop.write_text(desktop_entry(python, env), encoding="utf-8")
    desktop.chmod(0o755)
    return files
