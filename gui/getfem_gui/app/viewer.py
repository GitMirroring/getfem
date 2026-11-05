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
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QToolButton, QVBoxLayout,
                               QWidget)


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

    def press(self, widget, x, y) -> bool:
        v = self.viewer
        if v._tool is not None:
            v.draw_begin(self._world(widget, x, y))
            return True
        return v.handle_press(self._world(widget, x, y))

    def move(self, widget, x, y) -> bool:
        v = self.viewer
        if v._tool is not None:
            v.draw_move(self._world(widget, x, y))
            return True
        if v._grabbed is not None:
            v.handle_move(self._world(widget, x, y))
            return True
        return False

    def release(self, widget, x, y) -> bool:
        v = self.viewer
        if v._tool is not None:
            v.draw_end(self._world(widget, x, y))
            return True
        if v._grabbed is not None:
            v.handle_release(self._world(widget, x, y))
            return True
        return False


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
        if self.draw is not None and t in (
                QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease):
            # drawing a shape or dragging a handle: the events are taken here (the
            # view must not pan); the hooks say whether they take each event
            x, y = ev.position().x(), ev.position().y()
            if t == QEvent.Type.MouseMove:
                taken = self.draw.move(obj, x, y)
            elif ev.button() != Qt.MouseButton.LeftButton:
                taken = False
            elif t == QEvent.Type.MouseButtonPress:
                taken = self.draw.press(obj, x, y)
            else:
                taken = self.draw.release(obj, x, y)
            if taken:
                self._press = None
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
    regionClicked = Signal(object)        # region (geometry2d.region_at) clicked in the preview
    faceClicked = Signal(object)          # face (geometry3d reference) clicked in a 3D preview
    shapeDrawn = Signal(object, str)      # primitive (model format), operation
    handleMoved = Signal(str, object)     # handle name, new position (snapped)
    cutDrawn = Signal(object)             # cut line [a, b] drawn on a field
    cutCleared = Signal()

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
        self._build_result_bar(layout)
        self.edges = {"mesh": True, "field": False}   # element edges, per kind of display
        self._last = None
        self._shown_key = None
        self.error = ""            # why the 3D view is unavailable (with the traceback)
        self._pieces, self._hover = [], None     # picking state (edges)
        self._faces = None                       # picking state (3D faces): (surface, refs)
        self._regions_of = None                  # shape whose regions are picked
        try:
            self.plotter, self.view_widget = _load_plotter(self)
            layout.addWidget(self.view_widget, 1)
            self._build_cut_panel(layout)
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

    def _leave_field(self) -> None:
        """Another kind of display: the tools of the results go away."""
        if self._tool == "line":
            self.cut_button.setChecked(False)
        self.result_bar.hide()
        if self.cut_panel is not None:
            self.cut_panel.hide()
        self._cut = None

    def show_message(self, text: str) -> None:
        self._leave_field()
        self.status.setText(text)
        self._last = ("message", text)
        self._pieces = []
        self._faces = None
        self._shown_key = None
        if self.plotter is not None:
            from . import scene
            scene.message(self.plotter, "")
            self.plotter.render()
        else:
            self._placeholder(text)

    def show_mesh(self, mesh_file: Path, selection_file: Path | None = None,
                  status: str = "") -> None:
        self._leave_field()
        self.status.setText(status)
        self._last = ("mesh", mesh_file, selection_file, status)
        self._pieces = []
        self._faces = None
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
        """Exact preview of a 2D geometry; domain: True (the whole domain) or a
        function negative in the highlighted part (geometry2d.domain_level).
        pickable: True or "edges", clicks on boundary pieces are reported by
        edgeClicked; "regions", clicks in the domain by regionClicked."""
        self._leave_field()
        self.status.setText(status)
        self._last = ("geometry", shape, pieces, highlighted, domain, status, pickable, key)
        self._pieces = pieces if pickable in (True, "edges") else []
        self._faces = None
        self._regions_of = shape if pickable == "regions" else None
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
        if self._handles:
            from . import scene as _s
            _s.show_handles(self.plotter, [p for _, p in self._handles], minor=self._minor_handles())
        self.plotter.render()

    def show_geometry_3d(self, shape: dict, highlighted=None, part=None, status: str = "",
                         key: str | None = None, pickable: bool = False) -> None:
        """Preview of a 3D geometry (its implicit surface), without meshing;
        pickable: clicks on its faces are reported by faceClicked."""
        self._leave_field()
        self.status.setText(status)
        self._last = ("geometry3d", shape, highlighted, part, status, key, pickable)
        self._pieces, self._regions_of, self._hover, self._faces = [], None, None, None
        if self.plotter is None:
            self._placeholder(status)
            return
        from . import scene
        key = ("geometry3d", key if key is not None else repr(shape))
        scene.show_geometry_3d(self.plotter, shape, highlighted, part, reset=key != self._shown_key)
        self._shown_key = key
        if pickable:
            surf = scene.geometry_surface_3d(shape)
            refs = scene.face_refs(surf)
            if refs is not None:
                self._faces = (surf, refs)
        self.plotter.render()

    def face_at_display(self, x: float, y: float):
        """(face id, reference) of the face of the 3D preview at the display
        point (VTK display coordinates), or None."""
        if self._faces is None or self.plotter is None:
            return None
        import vtk
        actor = self.plotter.actors.get("surface")
        if actor is None:
            return None
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()
        picker.AddPickList(actor)
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        cell = picker.GetCellId()
        surf, refs = self._faces
        if cell < 0 or cell >= surf.n_cells:
            return None
        fid = int(surf.cell_data["face_id"][cell])
        return fid, refs[fid]

    def pick_face_display(self, x: float, y: float) -> None:
        found = self.face_at_display(x, y)
        if found is not None:
            self.faceClicked.emit(found[1])

    def hover_face_display(self, x: float, y: float) -> None:
        from . import scene
        found = self.face_at_display(x, y)
        fid = None if found is None else found[0]
        if fid != self._hover:
            self._hover = fid
            scene.show_face_hover(self.plotter, self._faces[0] if self._faces else None, fid)
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
        field = self._last is not None and self._last[0] == "field"
        if self._faces is not None:
            scale = widget.devicePixelRatioF()
            self.pick_face_display(round(x * scale), round((widget.height() - y - 1) * scale))
            return
        if not self._pieces and self._regions_of is None and not field:
            return
        from . import scene
        scale = widget.devicePixelRatioF()
        world = scene.display_to_world(self.plotter, round(x * scale),
                                       round((widget.height() - y - 1) * scale))
        if field:
            self.probe_world(world)
        else:
            self.pick_world(world)

    def _on_move(self, *args):
        if self._faces is not None:
            x, y = self.plotter.iren.interactor.GetEventPosition()
            self.hover_face_display(x, y)
            return
        if self._pieces or self._regions_of is not None:
            self.hover_world(self._event_world()[1])

    def _nearest(self, world):
        from .. import geometry2d as G
        from . import scene
        return G.nearest(self._pieces, world, self.PICK_PIXELS * scene.pixel_size(self.plotter))

    def pick_world(self, world) -> None:
        """A click at a point of the 2D view (world coordinates)."""
        if self._regions_of is not None:
            from .. import geometry2d as G
            region = G.region_at(self._regions_of, world)
            if region is not None:
                self.regionClicked.emit(region)
            return
        piece = self._nearest(world) if self._pieces else None
        if piece is not None:
            self.edgeClicked.emit(piece)

    def hover_world(self, world) -> None:
        from . import scene
        if self._regions_of is not None:
            from .. import geometry2d as G
            region = G.region_at(self._regions_of, world)
            if region != self._hover:
                self._hover = region
                scene.show_region_hover(self.plotter, self._regions_of, region)
                self.plotter.render()
            return
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
                                ("ball", "Disk", "Drag from the center to the circle"),
                                ("ellipse", "Ellipse", "Drag from a corner to the opposite one of the "
                                                       "rectangle around it (turn it with its handles)"),
                                ("polygon", "Polygon", "Click the vertices; click the first one again, "
                                                       "double-click or press Enter to close; Backspace "
                                                       "removes the last one")):
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
        self._handles, self._handle_preview, self._grabbed = [], None, None
        self._poly = []                   # vertices of the polygon being drawn
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.stop_drawing)
        for key, slot in ((Qt.Key.Key_Return, self.polygon_close), (Qt.Key.Key_Enter, self.polygon_close),
                          (Qt.Key.Key_Backspace, self.polygon_back)):
            sc = QShortcut(QKeySequence(key), self, slot)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)   # not in the forms

    # -------------------------------------------------------------- handles

    def set_handles(self, handles, preview=None) -> None:
        """Handles [(name, position)] of the selected primitive; preview(name,
        point) gives the shape while a handle is dragged."""
        self._handles, self._handle_preview, self._grabbed = list(handles), preview, None
        self._draw_handles()

    def _draw_handles(self, active=None):
        if self.plotter is None:
            return
        from . import scene
        scene.show_handles(self.plotter, [p for _, p in self._handles], active, minor=self._minor_handles())
        self.plotter.render()

    def _minor_handles(self) -> list[bool]:
        return [n[:1] == "m" and n[1:].isdigit() for n, _ in self._handles]

    def handle_press(self, world) -> bool:
        """Grab the handle under the mouse, if any (world coordinates)."""
        if not self._handles or self.plotter is None:
            return False
        from .. import geometry2d as G
        from . import scene
        px = scene.pixel_size(self.plotter)
        d = [np.hypot(world[0] - p[0], world[1] - p[1]) for _, p in self._handles]
        i = int(np.argmin(d))
        if d[i] > 9 * px:
            return False
        self._grabbed = i
        self._draw_step = G.grid_step(px * max(self.view_widget.width(), 100))
        self._draw_handles(active=i)
        return True

    def handle_move(self, world) -> None:
        if self._grabbed is None or self._handle_preview is None:
            return
        from . import scene
        name = self._handles[self._grabbed][0]
        scene.show_rubber(self.plotter, self._handle_preview(name, self._snapped(world)))
        self.plotter.render()

    def handle_release(self, world) -> None:
        if self._grabbed is None:
            return
        name = self._handles[self._grabbed][0]
        self._grabbed = None
        point = self._snapped(world)
        self._draw_step = None
        self.draw_cancel()
        self._draw_handles()
        self.handleMoved.emit(name, point)

    def set_drawing(self, enabled: bool) -> None:
        """Show the drawing tools (a 2D geometry is selected) or hide them."""
        if not enabled:
            self.stop_drawing()
        self.draw_bar.setVisible(enabled and self.plotter is not None)

    def stop_drawing(self) -> None:
        for b in self.draw_buttons.values():
            b.setChecked(False)
        if getattr(self, "cut_button", None) is not None:
            self.cut_button.setChecked(False)

    def _tool_toggled(self, kind, on):
        if on:
            for k, b in self.draw_buttons.items():
                if k != kind:
                    b.setChecked(False)
            self._tool = kind
        elif self._tool == kind:
            self._tool = None
        self._poly = []
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

    # polygon: a vertex per click (release); closed on the first vertex
    def _polygon_rubber(self, current=None) -> None:
        from . import scene
        pts = self._poly + ([current] if current is not None else [])
        scene.show_rubber(self.plotter, {"type": "polyline", "points": pts} if pts else None)
        self.plotter.render()

    def _polygon_click(self, world) -> None:
        from . import scene
        self.plotter.remove_actor("draw_status")
        p = self._snapped(world)
        tol = 8 * scene.pixel_size(self.plotter)
        if len(self._poly) >= 3 and np.hypot(p[0] - self._poly[0][0], p[1] - self._poly[0][1]) < tol:
            self.polygon_close()
            return
        if self._poly and np.hypot(p[0] - self._poly[-1][0], p[1] - self._poly[-1][1]) < tol:
            if len(self._poly) >= 3:       # a double click on the last vertex closes
                self.polygon_close()
            return
        self._poly.append(p)
        self._polygon_rubber()

    def polygon_close(self) -> None:
        """Finish the polygon being drawn (at least three vertices)."""
        if self._tool != "polygon" or len(self._poly) < 3:
            return
        from .. import geometry2d as G
        pts, self._poly = self._poly, []
        self.draw_cancel()
        problem = G.polygon_problem(pts)
        if problem:
            self.status_message(f"Polygon not added: {problem}")
            return
        self.shapeDrawn.emit({"type": "polygon", "points": pts}, self.draw_op.currentData())

    def polygon_back(self) -> None:
        if self._tool == "polygon" and self._poly:
            self._poly.pop()
            self._polygon_rubber()

    def status_message(self, text: str) -> None:
        if self.plotter is not None:
            self.plotter.add_text(text, position="lower_left", font_size=9, color="#aa3300", name="draw_status")
            self.plotter.render()

    def draw_move(self, world) -> None:
        if self._tool == "polygon" and self.plotter is not None:
            if self._poly:
                self._polygon_rubber(self._snapped(world))
            return
        if self._tool is None or self._draw_start is None or self.plotter is None:
            return
        from .. import geometry2d as G
        from . import scene
        if self._tool == "line":
            scene.show_rubber(self.plotter, {"type": "segment", "a": self._draw_start,
                                             "b": self._snapped(world)})
            self.plotter.render()
            return
        scene.show_rubber(self.plotter, G.drawn_shape(self._tool, self._draw_start, self._snapped(world)))
        self.plotter.render()

    def draw_end(self, world) -> None:
        if self._tool == "polygon" and self.plotter is not None:
            self._draw_start = None
            self._polygon_click(world)
            return
        if self._tool is None or self._draw_start is None:
            return
        from .. import geometry2d as G
        if self._tool == "line":
            a, b = self._draw_start, self._snapped(world)
            self.draw_cancel()
            if a != b:
                self.cut_button.setChecked(False)
                self.cutDrawn.emit([a, b])
            return
        shape = G.drawn_shape(self._tool, self._draw_start, self._snapped(world))
        self.draw_cancel()
        if shape is not None:
            self.shapeDrawn.emit(shape, self.draw_op.currentData())

    def draw_cancel(self) -> None:
        self._draw_start = None
        if self._tool != "polygon":
            self._poly = []
        if self.plotter is not None:
            from . import scene
            scene.show_rubber(self.plotter, None)
            self.plotter.render()

    # -------------------------------------------------------------- results

    def show_field(self, result_file: Path, name: str, component=None, warp=None,
                   scale=1.0, status: str = "", view: dict | None = None) -> None:
        """A result field; view: display options of scene.show_field (isolines,
        bands, cmap, clim, arrows, cut). A cut line also shows the values
        along it under the view; a click on a 2D field shows its value."""
        view = dict(view or {})
        self.status.setText(status)
        self._last = ("field", result_file, name, component, warp, scale, status, view)
        self._pieces, self._regions_of = [], None
        self._faces = None
        self.probe_label.setText("")
        if self.plotter is None:
            self._placeholder(status or name)
            return
        from . import scene
        key = ("result", str(result_file))
        ds = scene.read(result_file)
        scene.show_field(self.plotter, ds, name, component, warp, scale, edges=self.edges["field"],
                         reset=key != self._shown_key, **view)
        self._shown_key = key
        self.result_bar.setVisible(scene.is_planar(ds))
        self._show_cut(ds, name, component, view.get("cut"))
        if self._tool == "line":
            self._show_grid(True)
        self.plotter.render()

    # -------------------------------------------------------------- results: cut line, probe

    def _build_result_bar(self, layout):
        self.result_bar = QWidget()
        row = QHBoxLayout(self.result_bar)
        row.setContentsMargins(4, 0, 4, 2)
        self.cut_button = QToolButton()
        self.cut_button.setText("Cut line")
        self.cut_button.setObjectName("cut_line")
        self.cut_button.setCheckable(True)
        self.cut_button.setToolTip("Drag from A to B to plot the field along the segment "
                                   "(Escape: stop). A click on the field shows its value.")
        self.cut_button.toggled.connect(lambda on: self._tool_toggled("line", on))
        row.addWidget(self.cut_button)
        self.probe_label = QLabel("")
        self.probe_label.setObjectName("probe")
        self.probe_label.setStyleSheet("color: #555;")
        row.addWidget(self.probe_label, 1)
        layout.addWidget(self.result_bar)
        self.result_bar.hide()
        self.cut_panel = None
        self._cut = None                     # (distance, values, a, b, label)

    def _build_cut_panel(self, layout):
        from .plot import LinePlot
        self.cut_panel = QWidget()
        col = QVBoxLayout(self.cut_panel)
        col.setContentsMargins(4, 2, 4, 2)
        head = QHBoxLayout()
        self.cut_title = QLabel("")
        self.cut_title.setStyleSheet("color: #555;")
        head.addWidget(self.cut_title, 1)
        for text, tip, slot in (("CSV…", "Save the values along the line (CSV file)", self._save_cut_dialog),
                                ("Remove", "Remove the cut line", self.cutCleared.emit)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda checked=False, f=slot: f())
            head.addWidget(b)
        col.addLayout(head)
        self.cut_plot = LinePlot()
        col.addWidget(self.cut_plot)
        layout.addWidget(self.cut_panel)
        self.cut_panel.hide()

    def _show_cut(self, ds, name, component, cut):
        if self.cut_panel is None:
            return
        if not cut:
            self._cut = None
            self.cut_panel.hide()
            return
        from . import scene
        d, v = scene.line_values(ds, name, component, cut[0], cut[1])
        _, label = scene.values(ds, name, component)
        self._cut = (d, v, np.asarray(cut[0], float), np.asarray(cut[1], float), label)
        fmt = lambda p: "(" + ", ".join(f"{c:.4g}" for c in p) + ")"      # noqa: E731
        self.cut_title.setText(f"{label} along A {fmt(cut[0])} → B {fmt(cut[1])}")
        self.cut_plot.set_data(d, v, "distance from A", label)
        self.cut_panel.show()

    def save_cut(self, path) -> None:
        """The values along the cut line, as a CSV file (NaN outside the domain)."""
        d, v, a, b, label = self._cut
        u = (b - a) / (np.linalg.norm(b - a) or 1.0)
        with open(path, "w", encoding="utf-8") as f:
            f.write("distance," + ",".join("xyz"[:len(a)]) + f",{label}\n")
            for di, vi in zip(d, v):
                p = a + di * u
                f.write(f"{di:.10g}," + ",".join(f"{c:.10g}" for c in p) + f",{vi:.10g}\n")

    def _save_cut_dialog(self):
        if self._cut is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the values along the line", "cut.csv",
                                              "CSV files (*.csv)")
        if path:
            self.save_cut(path)

    def probe_world(self, world) -> float | None:
        """Value of the shown field at a point of a 2D view (world coordinates)."""
        if self._last is None or self._last[0] != "field":
            return None
        from . import scene
        _, result_file, name, component = self._last[:4]
        ds = scene.read(result_file)
        value = scene.probe(ds, name, component, world)
        _, label = scene.values(ds, name, component)
        where = f"({world[0]:.4g}, {world[1]:.4g})"
        self.probe_label.setText(f"{label} = {value:.6g} at {where}" if value is not None
                                 else f"{where}: outside the domain")
        return value

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
        elif last[0] == "geometry3d":
            self.show_geometry_3d(*last[1:])
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
