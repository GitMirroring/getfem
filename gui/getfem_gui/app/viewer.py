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


"""The 3D view: a PyVista plotter embedded in Qt (pyvistaqt).

Without PyVista (or when GETFEM_GUI_NO_3D is set, e.g. for tests), a plain
text placeholder is shown instead, and the rest of the window still works.
With GETFEM_GUI_OFFSCREEN_3D set, the scenes are drawn by an off-screen
PyVista plotter instead of the embedded one (tests, screenshots without a
display server).
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget


def prepare_opengl() -> None:
    """Process-wide OpenGL settings, to be applied before any window exists.

    Recent pyvistaqt (>= 0.13) renders through a QOpenGLWidget; on Wayland its
    context must match the default format of the top-level window, otherwise
    the view stays black. This is the format of VTK's QVTKRenderWindowAdapter:
    desktop OpenGL 3.2 core (not GLES, whose GLSL rejects VTK's shaders); VTK
    renders into its own framebuffers, hence the minimal buffers.
    """
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setVersion(3, 2)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    for size in (fmt.setRedBufferSize, fmt.setGreenBufferSize, fmt.setBlueBufferSize,
                 fmt.setAlphaBufferSize, fmt.setDepthBufferSize):
        size(8)
    fmt.setStencilBufferSize(0)
    fmt.setSamples(0)
    QSurfaceFormat.setDefaultFormat(fmt)
    if QCoreApplication.instance() is None:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)


def _quiet_matplotlib() -> None:
    """PyVista imports matplotlib.pyplot (for its colors). With PySide6 already
    loaded, matplotlib would pick its Qt backend, which fails with versions of
    matplotlib older than the installed PySide6 (TypeError on KeyboardModifier
    with PySide6 6.12). The window needs no matplotlib backend: use Agg,
    unless the user chose one."""
    if "matplotlib.pyplot" not in sys.modules:
        os.environ.setdefault("MPLBACKEND", "Agg")


def _load_plotter(parent):
    """(plotter, widget to insert in the layout)."""
    if os.environ.get("GETFEM_GUI_NO_3D"):
        raise ImportError("3D view disabled (GETFEM_GUI_NO_3D)")
    _quiet_matplotlib()
    if os.environ.get("GETFEM_GUI_OFFSCREEN_3D"):
        import pyvista as pv  # noqa: PLC0415
        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        plotter.set_background("white", top="#dfe6ee")
        return plotter, QLabel("(off-screen 3D view)")
    # The window is PySide6: pyvistaqt (through qtpy) must use the same binding,
    # whatever QT_API says in the environment (e.g. pyqt5 set by another tool).
    os.environ["QT_API"] = "pyside6"
    from pyvistaqt import QtInteractor  # noqa: PLC0415
    from . import scene  # noqa: F401, PLC0415  (fails early if pyvista is missing)
    plotter = QtInteractor(parent)
    plotter.set_background("white", top="#dfe6ee")
    plotter.enable_anti_aliasing("fxaa")
    return plotter, plotter


class Viewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        self.status = QLabel("")
        self.status.setStyleSheet("color: #555;")
        bar.addWidget(self.status, 1)
        self._buttons = []
        for text, tip, slot in (("Fit", "Zoom to fit (F)", self.fit),
                                ("XY", "View the XY plane", lambda: self._view("view_xy")),
                                ("3D", "Isometric view", lambda: self._view("view_isometric")),
                                ("Edges", "Show or hide the element edges", self.toggle_edges)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
            self._buttons.append(b)
        layout.addLayout(bar)
        self.edges = {"mesh": True, "field": False}   # element edges, per kind of display
        self._last = None
        self._shown_key = None
        self.error = ""            # why the 3D view is unavailable (with the traceback)
        try:
            self.plotter, self.view_widget = _load_plotter(self)
            layout.addWidget(self.view_widget, 1)
            self.placeholder = None
        except Exception as exc:  # noqa: BLE001 (missing package, no OpenGL, ...)
            self.plotter = None
            self.error = traceback.format_exc()
            self._reason = f"3D view unavailable: {type(exc).__name__}: {exc}"
            self.placeholder = QLabel()
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.placeholder.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._placeholder("")
            self.placeholder.setWordWrap(True)
            layout.addWidget(self.placeholder, 1)
            for b in self._buttons:
                b.setEnabled(False)

    def _placeholder(self, text: str) -> None:
        """Without 3D view: the reason stays visible, with the current message."""
        hint = ("Install pyvista and pyvistaqt (pip install pyvista pyvistaqt); "
                "the details are in the Messages tab.")
        self.placeholder.setText(f"{self._reason}\n{hint}" + (f"\n\n{text}" if text else ""))

    @property
    def available(self) -> bool:
        return self.plotter is not None

    # -------------------------------------------------------------- display

    def show_message(self, text: str) -> None:
        self.status.setText(text)
        self._last = ("message", text)
        self._shown_key = None
        if self.plotter is not None:
            from . import scene
            scene.message(self.plotter, "")
            self.plotter.render()
        else:
            self._placeholder(text)

    def show_mesh(self, mesh_file: Path, selection_file: Path | None = None,
                  status: str = "") -> None:
        self.status.setText(status)
        self._last = ("mesh", mesh_file, selection_file, status)
        if self.plotter is None:
            self._placeholder(status or str(mesh_file))
            return
        from . import scene
        key = ("mesh", str(mesh_file))
        mesh = scene.read(mesh_file)
        sel = scene.read(selection_file) if selection_file else None
        scene.show_mesh(self.plotter, mesh, sel, reset=key != self._shown_key, edges=self.edges["mesh"])
        self._shown_key = key
        self.plotter.render()

    def show_field(self, result_file: Path, name: str, component=None, warp=None,
                   scale=1.0, status: str = "") -> None:
        self.status.setText(status)
        self._last = ("field", result_file, name, component, warp, scale, status)
        if self.plotter is None:
            self._placeholder(status or name)
            return
        from . import scene
        key = ("result", str(result_file))
        ds = scene.read(result_file)
        scene.show_field(self.plotter, ds, name, component, warp, scale, edges=self.edges["field"],
                         reset=key != self._shown_key)
        self._shown_key = key
        self.plotter.render()

    # -------------------------------------------------------------- tools

    def fit(self):
        if self.plotter is not None:
            self.plotter.reset_camera()
            self.plotter.render()

    def _view(self, method):
        if self.plotter is not None:
            getattr(self.plotter, method)()
            if method == "view_isometric":
                self.plotter.disable_parallel_projection()
            self.plotter.reset_camera()
            self.plotter.render()

    def toggle_edges(self):
        kind = "field" if self._last and self._last[0] == "field" else "mesh"
        self.edges[kind] = not self.edges[kind]
        self.refresh()

    def refresh(self):
        last, self._last = self._last, None
        if last is None:
            return
        if last[0] == "mesh":
            self.show_mesh(*last[1:])
        elif last[0] == "field":
            self.show_field(*last[1:])
        else:
            self.show_message(last[1])

    def screenshot(self, path: str | Path) -> None:
        if self.plotter is not None:
            self.plotter.screenshot(str(path))

    def close_plotter(self):
        if self.plotter is not None:
            self.plotter.close()
