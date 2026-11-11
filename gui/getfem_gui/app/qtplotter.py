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

"""A PyVista plotter embedded in a Qt widget.

The widget is VTK's own Qt render window interactor (vtkmodules.qt, part of
the Python wrapping of VTK), which is also a pyvista.BasePlotter: all the
plotting methods of PyVista (add_mesh, camera, picking...) are those of the
widget. This is what pyvistaqt's QtInteractor does; the interface does not
use the rest of pyvistaqt (its windows, menus, render timer), and does not
depend on it, since it is not packaged by every distribution.

The plotter renders when asked (render()): the view calls it after each
change. The rendering is not done at once but from the Qt event loop, once
for all the changes made meanwhile, and only when the widget is on screen
(a hidden widget is drawn when shown, by its paint event). Rendering a
widget not yet shown blocks on macOS, where the window of the widget is
created by the window system only when shown: the main window would never
open (pyvistaqt also renders from the event loop on macOS).
"""

from __future__ import annotations

import pyvista
import vtkmodules.qt

vtkmodules.qt.PyQtImpl = "PySide6"      # the binding of the window, whatever is imported first

from pyvista.plotting.render_window_interactor import RenderWindowInteractor  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: E402

try:
    from pyvista.plotting import BasePlotter
except ImportError:                       # PyVista < 0.40
    from pyvista.plotting.plotting import BasePlotter


class _NoEvents:
    """What receives the Qt events of the widget once the plotter is closed
    (Qt may still send some: leave, resize)."""

    def _ignore(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return self._ignore


class QtPlotter(QVTKRenderWindowInteractor, BasePlotter):
    """A Qt widget that is a PyVista plotter."""

    def __init__(self, parent=None, multi_samples: int | None = None, **kwargs):
        # QVTKRenderWindowInteractor.__init__ initializes the Qt widget, whose
        # initialization would call BasePlotter.__init__ (without arguments)
        # through the method resolution order: called here, once, instead.
        init = BasePlotter.__init__
        BasePlotter.__init__ = lambda *args, **kw: None
        try:
            QVTKRenderWindowInteractor.__init__(self, parent=parent)
        finally:
            BasePlotter.__init__ = init
        BasePlotter.__init__(self, **kwargs)
        self.interactor = self
        self.ren_win = self.GetRenderWindow()
        self.ren_win.SetMultiSamples(pyvista.global_theme.multi_samples if multi_samples is None
                                     else multi_samples)
        for renderer in self.renderers:
            renderer.view_isometric()
            self.ren_win.AddRenderer(renderer)
        self.background_color = pyvista.global_theme.background
        self.iren = RenderWindowInteractor(self, interactor=self.ren_win.GetInteractor())
        self.iren.interactor.RemoveObservers("MouseMoveEvent")       # VTK's picking on motion: not used
        self.iren.initialize()
        self.enable_trackball_style()
        self._observers = {}
        self.iren.add_observer("KeyPressEvent", self.key_press_event)
        self.reset_key_events()
        self._first_time = False                 # BasePlotter: the window is set up, it may render

    def render(self) -> None:
        """Render soon, from the event loop (the view calls it after each
        change; PyVista also calls it from many of its methods)."""
        if getattr(self, "_closed", False):
            return
        self._rendered = True
        if not getattr(self, "_render_pending", False):
            self._render_pending = True
            QTimer.singleShot(0, self._render_now)

    def _render_now(self) -> None:
        self._render_pending = False
        if getattr(self, "_closed", False) or not self.isVisible():
            return                               # drawn by its paint event when shown
        BasePlotter.render(self)

    def paintEvent(self, ev):  # noqa: N802 (Qt)
        if not getattr(self, "_closed", False):
            super().paintEvent(ev)

    def resizeEvent(self, ev):  # noqa: N802 (Qt)
        if not getattr(self, "_closed", False):
            super().resizeEvent(ev)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        BasePlotter.close(self)
        QVTKRenderWindowInteractor.close(self)
        self._Iren = _NoEvents()
