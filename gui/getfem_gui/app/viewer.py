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

import numpy as np

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget


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
    # Multi-sampling renders thin lines (element edges) much better than FXAA,
    # which is only a fallback for drivers without multi-sampled framebuffers.
    try:
        plotter.enable_anti_aliasing("msaa", multi_samples=8)
    except Exception:  # noqa: BLE001
        plotter.enable_anti_aliasing("fxaa")
    return plotter, plotter


class _DrawHooks:
    """Mouse events of the view, while a drawing tool is active."""

    def __init__(self, viewer):
        self.viewer = viewer

    def active(self) -> bool:
        return self.viewer._tool is not None

    def _world(self, widget, x, y):
        from . import scene
        s = widget.devicePixelRatioF()
        return scene.display_to_world(self.viewer.plotter, round(x * s), round((widget.height() - y - 1) * s))

    def press(self, widget, x, y):
        self.viewer.draw_begin(self._world(widget, x, y))

    def move(self, widget, x, y):
        self.viewer.draw_move(self._world(widget, x, y))

    def release(self, widget, x, y):
        self.viewer.draw_end(self._world(widget, x, y))


class ClickFilter(QObject):
    """Reports left clicks (press and release without moving) on a widget.

    Done at the Qt level: VTK interactors swallow button release events, and
    PyVista then listens to them on the interactor style, which is replaced
    whenever the interaction style changes (2D / 3D)."""

    def __init__(self, callback, parent=None, draw=None):
        super().__init__(parent)
        self.callback = callback          # callback(widget, x, y), Qt logical coordinates
        self.draw = draw                  # object with active(), press/move/release(widget, x, y)
        self._press = None

    def eventFilter(self, obj, ev):
        t = ev.type()
        if self.draw is not None and self.draw.active() and t in (
                QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease):
            # drawing a shape: the view must not pan, the events are taken here
            if t == QEvent.Type.MouseMove:
                self.draw.move(obj, ev.position().x(), ev.position().y())
            elif ev.button() == Qt.MouseButton.LeftButton:
                (self.draw.press if t == QEvent.Type.MouseButtonPress else self.draw.release)(
                    obj, ev.position().x(), ev.position().y())
            return True
        if t == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
            self._press = ev.position()
        elif t == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
            press, self._press = self._press, None
            pos = ev.position()
            if press is not None and abs(pos.x() - press.x()) + abs(pos.y() - press.y()) <= 4:
                self.callback(obj, pos.x(), pos.y())
        return False                      # the view still receives the events (panning)


class Viewer(QWidget):
    edgeClicked = Signal(object)          # geometry2d.Piece clicked in a pickable preview
    shapeDrawn = Signal(object, str)      # primitive (model format), operation

    PICK_PIXELS = 7                       # click tolerance
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
                                ("XY", "View the XY plane (pan and zoom; Ctrl + drag to rotate)", lambda: self._view("view_xy")),
                                ("3D", "Isometric view, free rotation", lambda: self._view("view_isometric")),
                                ("Edges", "Show or hide the element edges", self.toggle_edges)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
            self._buttons.append(b)
        layout.addLayout(bar)
        self._build_draw_bar(layout)
        self.edges = {"mesh": True, "field": False}   # element edges, per kind of display
        self._last = None
        self._shown_key = None
        self.error = ""            # why the 3D view is unavailable (with the traceback)
        self._pieces, self._hover = [], None     # picking state
        try:
            self.plotter, self.view_widget = _load_plotter(self)
            layout.addWidget(self.view_widget, 1)
            self.placeholder = None
            self._install_picking()
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
        self._pieces = []
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
        self._pieces = []
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

    def show_geometry(self, shape: dict, pieces: list, highlighted=(), domain=False,
                      status: str = "", pickable: bool = False, key: str | None = None) -> None:
        """Exact preview of a 2D geometry; with pickable, clicks on its boundary
        pieces are reported by edgeClicked."""
        self.status.setText(status)
        self._last = ("geometry", shape, pieces, highlighted, domain, status, pickable, key)
        self._pieces = pieces if pickable else []
        self._hover = None
        if self.plotter is None:
            self._placeholder(status)
            return
        from . import scene
        # the camera is kept while the same geometry is edited (key: its name)
        key = ("geometry", key if key is not None else repr(shape))
        scene.show_geometry_2d(self.plotter, shape, pieces, highlighted, domain,
                               reset=key != self._shown_key)
        self._shown_key = key
        if self._tool is not None:
            self._show_grid(True)
        self.plotter.render()

    # -------------------------------------------------------------- picking

    def _install_picking(self):
        iren = getattr(self.plotter, "iren", None)
        if iren is None:              # off-screen plotter: see pick_world / hover_world
            return
        iren.add_observer("MouseMoveEvent", self._on_move)
        self._click_filter = ClickFilter(self._on_click, self, draw=_DrawHooks(self))
        self.view_widget.installEventFilter(self._click_filter)

    def _event_world(self):
        from . import scene
        x, y = self.plotter.iren.interactor.GetEventPosition()
        return (x, y), scene.display_to_world(self.plotter, x, y)

    def _on_click(self, widget, x, y):
        """A click on the view (Qt logical coordinates), converted to VTK display
        coordinates as the render window interactor does (origin at the bottom)."""
        if not self._pieces:
            return
        from . import scene
        scale = widget.devicePixelRatioF()
        world = scene.display_to_world(self.plotter, round(x * scale),
                                       round((widget.height() - y - 1) * scale))
        self.pick_world(world)

    def _on_move(self, *args):
        if self._pieces:
            self.hover_world(self._event_world()[1])

    def _nearest(self, world):
        from .. import geometry2d as G
        from . import scene
        return G.nearest(self._pieces, world, self.PICK_PIXELS * scene.pixel_size(self.plotter))

    def pick_world(self, world) -> None:
        """A click at a point of the 2D view (world coordinates)."""
        piece = self._nearest(world) if self._pieces else None
        if piece is not None:
            self.edgeClicked.emit(piece)

    def hover_world(self, world) -> None:
        from . import scene
        piece = self._nearest(world) if self._pieces else None
        if piece != self._hover:
            self._hover = piece
            scene.show_hover(self.plotter, piece)
            self.plotter.render()

    # -------------------------------------------------------------- drawing

    OPERATIONS = (("union", "add"), ("difference", "subtract"), ("intersection", "intersect"))

    def _build_draw_bar(self, layout):
        self.draw_bar = QWidget()
        row = QHBoxLayout(self.draw_bar)
        row.setContentsMargins(4, 0, 4, 2)
        row.addWidget(QLabel("Draw:"))
        self.draw_buttons = {}
        for kind, text, tip in (("rectangle", "Rectangle", "Drag from a corner to the opposite one"),
                                ("ball", "Disk", "Drag from the center to the circle")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip + " (Escape: stop drawing)")
            b.setCheckable(True)
            b.setObjectName(f"draw:{kind}")
            b.toggled.connect(lambda on, k=kind: self._tool_toggled(k, on))
            row.addWidget(b)
            self.draw_buttons[kind] = b
        self.draw_op = QComboBox()
        self.draw_op.setObjectName("draw:operation")
        for op, label in self.OPERATIONS:
            self.draw_op.addItem(label, op)
        self.draw_op.setToolTip("How the drawn shape is combined with the geometry")
        row.addWidget(self.draw_op)
        self.grid_label = QLabel("")
        self.grid_label.setStyleSheet("color: #555;")
        row.addWidget(self.grid_label)
        row.addStretch(1)
        layout.addWidget(self.draw_bar)
        self.draw_bar.hide()
        self._tool = None
        self._draw_start = None
        self._draw_step = None
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.stop_drawing)

    def set_drawing(self, enabled: bool) -> None:
        """Show the drawing tools (a 2D geometry is selected) or hide them."""
        if not enabled:
            self.stop_drawing()
        self.draw_bar.setVisible(enabled and self.plotter is not None)

    def stop_drawing(self) -> None:
        for b in self.draw_buttons.values():
            b.setChecked(False)

    def _tool_toggled(self, kind, on):
        if on:
            for k, b in self.draw_buttons.items():
                if k != kind:
                    b.setChecked(False)
            self._tool = kind
        elif self._tool == kind:
            self._tool = None
        self.draw_cancel()
        self._show_grid(self._tool is not None)

    def _show_grid(self, on: bool):
        if self.plotter is None:
            return
        from .. import geometry2d as G
        from . import scene
        if not on:
            self._draw_step = None
            self.grid_label.setText("")
            scene.show_grid(self.plotter, None, None, None)
            self.plotter.render()
            return
        width = scene.pixel_size(self.plotter) * max(self.view_widget.width(), 100)
        self._draw_step = G.grid_step(width)
        self.grid_label.setText(f"grid {self._draw_step:g}")
        lo = scene.display_to_world(self.plotter, 0, 0)
        hi = scene.display_to_world(self.plotter, self.view_widget.width() * self.view_widget.devicePixelRatioF(),
                                    self.view_widget.height() * self.view_widget.devicePixelRatioF())
        scene.show_grid(self.plotter, np.minimum(lo, hi), np.maximum(lo, hi), self._draw_step)
        self.plotter.render()

    def _snapped(self, world):
        from .. import geometry2d as G
        return G.snap(world, self._draw_step) if self._draw_step else [float(world[0]), float(world[1])]

    def draw_begin(self, world) -> None:
        if self._tool is not None:
            self._draw_start = self._snapped(world)

    def draw_move(self, world) -> None:
        if self._tool is None or self._draw_start is None or self.plotter is None:
            return
        from .. import geometry2d as G
        from . import scene
        scene.show_rubber(self.plotter, G.drawn_shape(self._tool, self._draw_start, self._snapped(world)))
        self.plotter.render()

    def draw_end(self, world) -> None:
        if self._tool is None or self._draw_start is None:
            return
        from .. import geometry2d as G
        shape = G.drawn_shape(self._tool, self._draw_start, self._snapped(world))
        self.draw_cancel()
        if shape is not None:
            self.shapeDrawn.emit(shape, self.draw_op.currentData())

    def draw_cancel(self) -> None:
        self._draw_start = None
        if self.plotter is not None:
            from . import scene
            scene.show_rubber(self.plotter, None)
            self.plotter.render()

    # -------------------------------------------------------------- results

    def show_field(self, result_file: Path, name: str, component=None, warp=None,
                   scale=1.0, status: str = "") -> None:
        self.status.setText(status)
        self._last = ("field", result_file, name, component, warp, scale, status)
        self._pieces = []
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
            from . import scene
            getattr(self.plotter, method)()
            if method == "view_isometric":
                self.plotter.disable_parallel_projection()
            scene.set_style(self.plotter, planar=method == "view_xy")
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
        elif last[0] == "geometry":
            self.show_geometry(*last[1:])
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
