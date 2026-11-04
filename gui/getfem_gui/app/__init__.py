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


def main(argv: list[str] | None = None, python: str | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("error: the graphical interface requires PySide6 (pip install pyside6 pyvistaqt)",
              file=sys.stderr)
        return 1
    from .window import MainWindow

    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("GetFEM GUI")
    app.setOrganizationName("GetFEM")
    app.setStyle("Fusion")
    win = MainWindow(argv[0] if argv else None, python=python)
    win.show()
    return app.exec()
