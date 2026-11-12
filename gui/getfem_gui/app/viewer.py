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


"""The 3D view: a PyVista plotter embedded in Qt (qtplotter.py, VTK's Qt widget).

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

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMenu,
                               QSpinBox, QToolButton, QVBoxLayout, QWidget)


def prepare_opengl() -> None:
    """Process-wide OpenGL settings, to be applied before any window exists.

    When VTK's Qt widget renders through a QOpenGLWidget
    (vtkmodules.qt.QVTKRWIBase), on Wayland its context must match the
    default format of the top-level window, otherwise the view stays black.
    This is the format of VTK's QVTKRenderWindowAdapter: desktop OpenGL 3.2
    core (not GLES, whose GLSL rejects VTK's shaders). The 3D view (app/
    qtplotter.py) renders in the native window of its widget, whose visual
    follows this default format: a depth buffer of 24 bits (with 8 bits,
    the faces and the edges of the elements at nearly the same depth could
    not be told apart: edges hidden or showing through the faces).
    """
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setVersion(3, 2)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    for size in (fmt.setRedBufferSize, fmt.setGreenBufferSize, fmt.setBlueBufferSize,
                 fmt.setAlphaBufferSize):
        size(8)
    fmt.setDepthBufferSize(24)
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
    from . import scene  # noqa: F401, PLC0415  (fails early if pyvista is missing)
    from .qtplotter import QtPlotter  # noqa: PLC0415  (VTK's own Qt widget: no pyvistaqt)
    plotter = QtPlotter(parent)
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

    def view_changed(self) -> None:
        v = self.viewer
        if v._tool is not None or (v._last and v._last[0] == "geometry"):
            v._grid_timer.start()

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

    def __init__(self, callback, parent=None, draw=None, other=None):
        super().__init__(parent)
        self.callback = callback          # callback(widget, x, y), Qt logical coordinates
        self.draw = draw                  # object with active(), press/move/release(widget, x, y)
        self.other = other                # other(widget, x, y, button): middle and right clicks
        self._press = None
        self._other_press = {}            # button: position (a click of the middle or right button)

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
        if self.draw is not None and (t == QEvent.Type.Wheel or t == QEvent.Type.MouseButtonRelease) \
                and hasattr(self.draw, "view_changed"):
            self.draw.view_changed()          # a zoom or a pan: the grid follows
        if self.other is not None and t in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease) \
                and ev.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            # a click (no move: the drags still pan and zoom the view)
            if t == QEvent.Type.MouseButtonPress:
                self._other_press[ev.button()] = ev.position()
            else:
                press, pos = self._other_press.pop(ev.button(), None), ev.position()
                if press is not None and abs(pos.x() - press.x()) + abs(pos.y() - press.y()) <= 4:
                    self.other(obj, pos.x(), pos.y(), ev.button())
            return False
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
    polygonEntered = Signal(object)       # vertices of a polygon of the tree being entered
    handleMoved = Signal(str, object)     # handle name, new position (snapped)
    cutDrawn = Signal(object)             # cut line [a, b] drawn on a field
    cutCleared = Signal()
    escaped = Signal()                    # Escape pressed in the view (drawing stopped)
    handleAction = Signal(str)            # an editing action chosen in the menu of a handle
    handleClicked = Signal(str)           # a handle clicked (pressed and released without moving)
    fieldEdgesToggled = Signal(bool)      # the Edges button on a result field: its setting

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
                                ("3D", "Isometric view, free rotation (a profile: drawn on its plane, in the "
                                       "3D geometry)", lambda: self._view("view_isometric")),
                                ("Edges", "Show or hide the element edges (on a result field: a setting of its graphic)",
                                 self.toggle_edges)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
            self._buttons.append(b)
        # the display of quadratic elements (meshes of geometric order 2): subdivisions
        self.curved_label = QLabel("Curved:")
        self.curved_box = QSpinBox()
        self.curved_box.setObjectName("curved")
        self.curved_box.setRange(0, 4)
        self.curved_box.setValue(2)
        tip = ("Elements of order 2: each face is drawn in 4^n triangles and each edge in 2^n segments, "
               "placed and colored by the quadratic interpolation (0: straight, through the corners)")
        self.curved_label.setToolTip(tip)
        self.curved_box.setToolTip(tip)
        self.curved_box.valueChanged.connect(lambda _v: self.refresh())
        for w in (self.curved_label, self.curved_box):
            w.setVisible(False)
            bar.insertWidget(bar.count() - 1, w)
        layout.addLayout(bar)
        self._build_draw_bar(layout)
        self._build_result_bar(layout)
        self.edges = {"mesh": True, "field": False}   # element edges, per kind of display
        self._last = None
        self._shown_key = None
        self.plane_3d = False      # a profile drawn on its plane in the 3D view (else flat)
        self._context = None       # the 3D geometry around the plane of the profile shown
        self._plane_box = None     # the part of the plane shown in 3D (grid, axes)
        self.error = ""            # why the 3D view is unavailable (with the traceback)
        self._pieces, self._hover = [], None     # picking state (edges)
        self._faces = None                       # picking state (3D faces): (surface, refs)
        self._regions_of = None                  # shape whose regions are picked
        self.handle_menu = None                  # handle name -> [(label, editing action)] (the window)
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
        hint = ("Install pyvista (pip install pyvista; Debian: python3-pyvista); "
                "the details are in the Messages tab.")
        self.placeholder.setText(f"{self._reason}\n{hint}" + (f"\n\n{text}" if text else ""))

    @property
    def available(self) -> bool:
        return self.plotter is not None

    # -------------------------------------------------------------- display

    def _leave_field(self) -> None:
        """Another kind of display: the tools of the results go away."""
        self._snap_points = None
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
        self._show_curved(scene.is_curved(mesh))
        scene.show_mesh(self.plotter, mesh, sel, reset=key != self._shown_key, edges=self.edges["mesh"],
                        level=self.curved_box.value())
        self._shown_key = key
        self.plotter.render()

    def show_geometry(self, shape: dict, pieces: list, highlighted=(), domain=False,
                      status: str = "", pickable: bool = False, key: str | None = None,
                      reference=None, context=None) -> None:
        """Exact preview of a 2D geometry; domain: True (the whole domain) or a
        function negative in the highlighted part (geometry2d.domain_level).
        pickable: True or "edges", clicks on boundary pieces are reported by
        edgeClicked; "regions", clicks in the domain by regionClicked.
        context: for a profile, the 3D geometry in the coordinates of its
        plane (a surface): with plane_3d (the 3D button), the profile is
        shown and drawn on its plane in the 3D view, with the geometry
        around it."""
        self._leave_field()
        self._show_curved(False)
        self._context = context
        in_3d = self.plane_3d and context is not None
        self.status.setText(status + (" — in 3D: rotate with the mouse, draw on the plane (XY: flat)"
                                      if in_3d else ""))
        self._last = ("geometry", shape, pieces, highlighted, domain, status, pickable, key, reference, context)
        self._pieces = pieces if pickable in (True, "edges") else []
        self._faces = None
        self._regions_of = shape if pickable == "regions" else None
        self._hover = None
        if self.plotter is None:
            self._placeholder(status)
            return
        from . import scene
        # the camera is kept while the same geometry is edited (key: its name)
        key = ("geometry3" if in_3d else "geometry", key if key is not None else repr(shape))
        from .. import geometry2d as G
        lines = scene.reference_lines(reference) if reference is not None else None
        self._plane_box = self._plane_extent(shape, context, lines) if in_3d else None
        reset = key != self._shown_key or (not in_3d and self._outside_view(shape))
        scene.show_geometry_2d(self.plotter, shape, pieces, highlighted, domain,
                               reset=reset and not in_3d, reference=lines)
        if in_3d:
            scene.show_context(self.plotter, context)
            if reset:
                scene.view_plane_3d(self.plotter)
        corners = scene.reference_corners(lines)
        self._snap_points = np.concatenate([G.key_points(shape), corners]) if len(corners) else G.key_points(shape)
        self._shown_key = key
        if self._grid_wanted():
            self._show_grid(True)
        else:
            self._show_axes()
        if self._handles:
            from . import scene as _s
            _s.show_handles(self.plotter, [p for _, p in self._handles], minor=self._minor_handles())
            _s.show_marks(self.plotter, getattr(self, "_marks", []))
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

    def show_manifold(self, src: dict, values: dict, status: str = "", key: str | None = None) -> None:
        """Preview of a curve or of a surface in 3D, without meshing."""
        self._leave_field()
        self.status.setText(status)
        self._last = ("manifold", src, values, status, key)
        self._pieces, self._regions_of, self._hover, self._faces = [], None, None, None
        if self.plotter is None:
            self._placeholder(status)
            return
        from . import scene
        key = ("manifold", key if key is not None else repr(src))
        scene.show_manifold(self.plotter, src, values, reset=key != self._shown_key)
        self._shown_key = key
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
            self.last_face = found[0]
            self.faceClicked.emit(found[1])

    def face_plane(self, fid: int):
        """(center, outward normal) of a face of the 3D preview, None if it is
        not plane."""
        from . import scene
        if self._faces is None or self._last is None or self._last[0] != "geometry3d":
            return None
        return scene.face_plane(self._faces[0], fid, self._last[1])

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
        self._click_filter = ClickFilter(self._on_click, self, draw=_DrawHooks(self), other=self._on_other_click)
        self.view_widget.installEventFilter(self._click_filter)

    def _on_other_click(self, widget, x, y, button) -> None:
        """A click of the middle button (the last vertex of a polygon being
        drawn) or of the right one (a menu: the vertices)."""
        world = _DrawHooks(self)._world(widget, x, y)
        if button == Qt.MouseButton.MiddleButton:
            self.middle_click(world)
            return
        items = self.context_items(world)
        if items:
            menu = QMenu(self)
            for label, fn in items:
                menu.addAction(label).triggered.connect(lambda _c=False, f=fn: f())
            menu.exec(widget.mapToGlobal(QPoint(round(x), round(y))))

    def middle_click(self, world) -> None:
        """Drawing a polygon: the point is its last vertex (it is closed)."""
        if self._tool != "polygon" or self.plotter is None:
            return
        from . import scene
        p = self._snapped(world)
        tol = 8 * scene.pixel_size(self.plotter)
        near = [q for q in (self._poly[:1] + self._poly[-1:]) if np.hypot(p[0] - q[0], p[1] - q[1]) < tol]
        if not near:
            self._poly.append(p)
        if len(self._poly) >= 3:
            self.polygon_close()
        else:
            self._polygon_rubber()
            self.status_message("A polygon needs at least three vertices")

    def context_items(self, world) -> list:
        """The entries of the menu of a right click at a point: [(label,
        function)]. Drawing a polygon: remove the vertex clicked, the last one,
        close, cancel. Otherwise, on a handle of the selected primitive: what
        handle_menu (set by the window) gives for it."""
        if self.plotter is None:
            return []
        from . import scene
        tol = 8 * scene.pixel_size(self.plotter)
        if self._tool == "polygon":
            out = []
            d = [np.hypot(world[0] - q[0], world[1] - q[1]) for q in self._poly]
            if d and min(d) < tol:
                j = int(np.argmin(d))
                out.append((f"Remove this vertex ({j + 1})", lambda j=j: self._polygon_remove(j)))
            if self._poly:
                out.append(("Remove the last vertex (Backspace)", self.polygon_back))
            if len(self._poly) >= 3:
                out.append(("Close the polygon (Enter, middle click)", self.polygon_close))
            if self._poly:
                out.append(("Start the polygon again", lambda: self._polygon_remove(None)))
            return out
        if not self._handles or self.handle_menu is None:
            return []
        d = [np.hypot(world[0] - p[0], world[1] - p[1]) for _, p in self._handles]
        i = int(np.argmin(d))
        if d[i] > 9 * scene.pixel_size(self.plotter):
            return []
        return [(label, lambda ident=ident: self.handleAction.emit(ident))
                for label, ident in self.handle_menu(self._handles[i][0])]

    def _polygon_remove(self, j) -> None:
        """A vertex of the polygon being drawn removed (None: all of them)."""
        if j is None:
            self._poly = []
        elif 0 <= j < len(self._poly):
            del self._poly[j]
        self._polygon_rubber()

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
                                                       "double-click, press Enter or click the middle button "
                                                       "(the last vertex) to close; Backspace removes the "
                                                       "last vertex, a right click a vertex")):
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
        row.addWidget(QLabel("Grid:"))
        self.grid_edit = QLineEdit()
        self.grid_edit.setObjectName("draw:grid")
        self.grid_edit.setFixedWidth(70)
        self.grid_edit.setToolTip("Step of the grid (the drawn points snap to it); empty: it follows the zoom")
        self.grid_edit.editingFinished.connect(self._grid_typed)
        row.addWidget(self.grid_edit)
        for ident, text, tip in (("finer", "−", "A finer grid"), ("coarser", "+", "A coarser grid"),
                                 ("auto", "Auto", "The grid follows the zoom")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setObjectName(f"draw:grid_{ident}")
            b.clicked.connect(lambda _=False, i=ident: self._grid_button(i))
            row.addWidget(b)
        self.grid_label = QLabel("")
        self.grid_label.setStyleSheet("color: #555;")
        row.addWidget(self.grid_label)
        self.snap_box = QCheckBox("snap")
        self.snap_box.setObjectName("draw:snap")
        self.snap_box.setChecked(True)
        self.snap_box.setToolTip("The drawn points and the dragged handles snap to the grid "
                                 "(unchecked: anywhere)")
        row.addWidget(self.snap_box)
        self._grid_fixed = None           # a step chosen by the user (None: it follows the zoom)
        self._grid_timer = QTimer(self)
        self._grid_timer.setSingleShot(True)
        self._grid_timer.setInterval(150)
        self._grid_timer.timeout.connect(self.refresh_grid)
        row.addStretch(1)
        layout.addWidget(self.draw_bar)
        self.draw_bar.hide()
        self._tool = None
        self._draw_start = None
        self._draw_step = None
        self._handles, self._handle_preview, self._grabbed = [], None, None
        self._poly = []                   # vertices of the polygon being drawn
        self._entering = False            # the polygon is the one of the tree being entered
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._escape)
        for key, slot in ((Qt.Key.Key_Return, self.polygon_close), (Qt.Key.Key_Enter, self.polygon_close),
                          (Qt.Key.Key_Backspace, self.polygon_back)):
            sc = QShortcut(QKeySequence(key), self, slot)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)   # not in the forms

    # -------------------------------------------------------------- handles

    def set_handles(self, handles, preview=None, marks=()) -> None:
        """Handles [(name, position)] of the selected primitive; preview(name,
        point) gives the shape while a handle is dragged; marks [(position,
        text)]: its constraints, drawn by its sides and vertices."""
        self._handles, self._handle_preview, self._grabbed = list(handles), preview, None
        self._marks = list(marks)
        if self.plotter is not None and self._tool is None:
            self._show_grid(self._grid_wanted())
        self._draw_handles()

    def _draw_handles(self, active=None):
        if self.plotter is None:
            return
        from . import scene as _scene
        _scene.show_marks(self.plotter, getattr(self, "_marks", []))
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
        self._grabbed_at = (world[0], world[1], px)
        if not self._draw_step:
            self._draw_step = self._grid_fixed or G.grid_step(px * max(self.view_widget.width(), 100))
        self._draw_handles(active=i)
        return True

    def handle_move(self, world) -> None:
        if self._grabbed is None or self._handle_preview is None:
            return
        from . import scene
        name = self._handles[self._grabbed][0]
        point = self._snapped(world)
        scene.move_active_handle(self.plotter, point)          # the handle follows the mouse,
        scene.show_rubber(self.plotter, self._handle_preview(name, point))   # the outline too, if valid
        self.plotter.render()

    def handle_release(self, world) -> None:
        if self._grabbed is None:
            return
        name = self._handles[self._grabbed][0]
        self._grabbed = None
        point = self._snapped(world)
        self.draw_cancel()
        self._draw_handles()
        x0, y0, px = getattr(self, "_grabbed_at", (None, None, 0))
        if x0 is not None and np.hypot(world[0] - x0, world[1] - y0) <= 3 * px:   # not moved: a click
            self.handleClicked.emit(name)
            return
        self.handleMoved.emit(name, point)

    def set_drawing(self, enabled: bool) -> None:
        """Show the drawing tools (a 2D geometry is selected) or hide them."""
        if not enabled:
            self.stop_drawing()
        self.draw_bar.setVisible(enabled and self.plotter is not None)

    def _escape(self) -> None:
        self.stop_drawing()
        self.escaped.emit()

    def stop_drawing(self) -> None:
        for b in self.draw_buttons.values():
            b.setChecked(False)
        if getattr(self, "cut_button", None) is not None:
            self.cut_button.setChecked(False)

    def enter_polygon(self) -> None:
        """Enter the vertices of the (empty) polygon selected in the tree: the
        polygon tool, whose result goes to that polygon."""
        if self.plotter is None or (self._entering and self._tool == "polygon"):
            return
        self.draw_buttons["polygon"].setChecked(True)
        self._entering = True
        self.status_message("Click the first vertex of the polygon (then the others; click the first "
                            "one again, double-click, Enter or a middle click (the last vertex) to close; "
                            "right click: remove a vertex)")

    def _tool_toggled(self, kind, on):
        if not on and kind == "polygon":
            self._entering = False
        if on:
            for k, b in self.draw_buttons.items():
                if k != kind:
                    b.setChecked(False)
            self._tool = kind
        elif self._tool == kind:
            self._tool = None
        self._poly = []
        self.draw_cancel()
        self._show_grid(self._grid_wanted())

    def _grid_wanted(self) -> bool:
        """The grid is shown while drawing, and while a primitive with handles
        is selected (its handles snap to it)."""
        return self._tool is not None or (bool(self._handles) and self.draw_bar.isVisibleTo(self)
                                          and bool(self._last) and self._last[0] == "geometry")

    def _show_grid(self, on: bool):
        if self.plotter is None:
            return
        from .. import geometry2d as G
        from . import scene
        if not on:
            self._draw_step = None
            self.grid_label.setText("")
            self.grid_edit.setText("")
            scene.show_grid(self.plotter, None, None, None)
            self._show_axes()                 # kept on a 2D geometry
            self.plotter.render()
            return
        lo, hi = self._view_box()
        width = float(hi[0] - lo[0]) if self._plane_box is not None \
            else scene.pixel_size(self.plotter) * max(self.view_widget.width(), 100)
        self._draw_step = self._grid_fixed or G.grid_step(width)
        if not self.grid_edit.hasFocus():
            self.grid_edit.setText(f"{self._draw_step:g}")
        shown = (np.ceil(hi[0] / self._draw_step) - np.floor(lo[0] / self._draw_step) + 1) * \
            (np.ceil(hi[1] / self._draw_step) - np.floor(lo[1] / self._draw_step) + 1) <= 40000
        self.grid_label.setText(("auto" if self._grid_fixed is None else "fixed")
                                + ("" if shown else ", too fine to be shown"))
        scene.show_grid(self.plotter, lo, hi, self._draw_step)
        scene.show_axes_2d(self.plotter, lo, hi, G.major_step(self._draw_step, float(hi[0] - lo[0])))
        self.plotter.render()

    @staticmethod
    def _plane_extent(shape, context, lines):
        """The part of the plane of a profile shown in 3D: around the profile,
        the section under it and the geometry seen in the plane."""
        from .. import geometry2d as G
        boxes = []
        if not G.is_empty(shape):
            try:
                boxes.append(G.bounds(shape))
            except ValueError:
                pass
        for data in (context, lines):
            if data is not None and data.n_points:
                b = data.bounds
                boxes.append((np.array([b[0], b[2]]), np.array([b[1], b[3]])))
        if not boxes:
            return np.zeros(3), np.ones(3)
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        m = 0.1 * float(np.max(hi - lo) or 1.0)
        return np.append(lo - m, 0.0), np.append(hi + m, 0.0)

    def _outside_view(self, shape: dict) -> bool:
        """The geometry goes out of the view (moved or copied by a transform,
        a value typed in a form): the view is then fitted to it again."""
        from .. import geometry2d as G
        if self.plotter is None or G.is_empty(shape):
            return False
        try:
            lo, hi = G.bounds(shape)
            vlo, vhi = self._view_box()
        except Exception:  # noqa: BLE001 -- no view yet: nothing to refit
            return False
        return bool(np.any(lo[:2] < vlo[:2]) or np.any(hi[:2] > vhi[:2]))

    def _view_box(self):
        if self._plane_box is not None and self._last and self._last[0] == "geometry":
            return self._plane_box               # a profile in 3D: the part of its plane shown
        from . import scene
        w, h = self.plotter.renderer.GetSize()           # in pixels of the render window
        lo = scene.display_to_world(self.plotter, 0, 0)
        hi = scene.display_to_world(self.plotter, w, h)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def _show_axes(self) -> None:
        """The axes and their coordinates on a 2D geometry (also when not
        drawing); removed on the other views."""
        from .. import geometry2d as G
        from . import scene
        if self.plotter is None:
            return
        if not (self._last and self._last[0] == "geometry"):
            scene.show_axes_2d(self.plotter, None, None, None)
            return
        lo, hi = self._view_box()
        width = float(hi[0] - lo[0])
        step = self._draw_step or G.grid_step(width)
        scene.show_axes_2d(self.plotter, lo, hi, G.major_step(step, width))

    def refresh_grid(self) -> None:
        """The grid and the axes again (after a zoom or a pan)."""
        if self.plotter is None:
            return
        if self._grid_wanted():
            self._show_grid(True)
        elif self._last and self._last[0] == "geometry":
            self._show_axes()
            self.plotter.render()

    def _grid_button(self, ident: str) -> None:
        from .. import geometry2d as G
        if ident == "auto":
            self._grid_fixed = None
        elif self._draw_step:
            self._grid_fixed = G.next_step(self._draw_step, finer=ident == "finer")
        self.refresh_grid()

    def _grid_typed(self) -> None:
        text = self.grid_edit.text().strip()
        if not text:
            self._grid_fixed = None
        else:
            try:
                step = float(text)
            except ValueError:
                step = 0.0
            if step <= 0 or not np.isfinite(step):
                self.grid_edit.setText(f"{self._draw_step:g}" if self._draw_step else "")
                return
            self._grid_fixed = step
        self.grid_edit.clearFocus()
        self.refresh_grid()

    SNAP_PIXELS = 8                       # the distance at which a point snaps to a vertex

    def _snapped(self, world):
        """The point snapped: to a vertex of the geometry (or of the section
        under a profile) within SNAP_PIXELS, else to the grid."""
        from .. import geometry2d as G
        if not self.snap_box.isChecked():
            return [float(world[0]), float(world[1])]
        pts = getattr(self, "_snap_points", None)
        if pts is not None and len(pts) and self.plotter is not None:
            from . import scene
            d = np.linalg.norm(pts - np.asarray(world[:2], float), axis=1)
            k = int(np.argmin(d))
            if d[k] <= self.SNAP_PIXELS * scene.pixel_size(self.plotter):
                return [float(pts[k][0]), float(pts[k][1])]
        if self._draw_step:
            return G.snap(world, self._draw_step)
        return [float(world[0]), float(world[1])]

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
        if self._entering:
            self._entering = False
            self.draw_buttons["polygon"].setChecked(False)
            self.polygonEntered.emit(pts)
            return
        op = self.draw_op.currentData()
        self.draw_buttons["polygon"].setChecked(False)     # done: a click now grabs a handle
        self.shapeDrawn.emit({"type": "polygon", "points": pts}, op)

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
            op = self.draw_op.currentData()
            self.draw_buttons[self._tool].setChecked(False)   # done: a click now grabs a handle
            self.shapeDrawn.emit(shape, op)

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
        bands, cmap, clim, arrows, cut, edges: the edges of the elements). A
        cut line also shows the values along it under the view; a click on a
        2D field shows its value."""
        view = dict(view or {})
        if "edges" in view:                    # the setting of the graphic
            self.edges["field"] = bool(view["edges"])
        view.pop("edges", None)
        self.status.setText(status)
        self._last = ("field", result_file, name, component, warp, scale, status, view)
        self._snap_points = None
        self._pieces, self._regions_of = [], None
        self._faces = None
        self.probe_label.setText("")
        if self.plotter is None:
            self._placeholder(status or name)
            return
        from . import scene
        key = ("result", str(result_file))
        ds = scene.read(result_file)
        self._show_curved(scene.is_curved(ds))
        scene.show_field(self.plotter, ds, name, component, warp, scale, edges=self.edges["field"],
                         reset=key != self._shown_key, level=self.curved_box.value(), **view)
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
        if self.plotter is not None and self._last and self._last[0] == "geometry" \
                and self._context is not None:   # a profile: flat, or on its plane in the 3D geometry
            self.plane_3d = method == "view_isometric"
            self._shown_key = None
            self.refresh()
            return
        if self.plotter is not None:
            from . import scene
            getattr(self.plotter, method)()
            if method == "view_isometric":
                self.plotter.disable_parallel_projection()
            scene.set_style(self.plotter, planar=method == "view_xy")
            self.plotter.reset_camera()
            self.plotter.render()

    def _show_curved(self, on: bool) -> None:
        """The subdivision control, for a display of quadratic elements."""
        self.curved_label.setVisible(on)
        self.curved_box.setVisible(on)

    def toggle_edges(self):
        """The edges of the elements shown or hidden: on a result field, its
        display setting (saved in its graphic, by the window)."""
        kind = "field" if self._last and self._last[0] == "field" else "mesh"
        self.edges[kind] = not self.edges[kind]
        if kind == "field" and getattr(self, "field_edges_saved", False):
            self.fieldEdgesToggled.emit(self.edges[kind])     # the window shows the field again
            return
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
