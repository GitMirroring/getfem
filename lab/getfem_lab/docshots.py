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

"""Screenshots of the user documentation.

The window is drawn without a screen (Qt "offscreen" platform) and its 3D
view by an off-screen PyVista plotter, pasted where the view is. Requires
PySide6 and PyVista, and an OpenGL context for VTK (with a server such as
Xvfb when there is no display):

    xvfb-run -a python -m getfem_lab.docshots [directory] [name ...]

writes the PNG files (by default into doc/sphinx/source/lab/images of the
GetFEM sources); names select some of the shots.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["GETFEM_LAB_OFFSCREEN_3D"] = "1"
os.environ.pop("GETFEM_LAB_NO_3D", None)

from . import editing as E  # noqa: E402

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "doc" / "sphinx" / "source" / "lab" / "images"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SIZE = (1280, 760)


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _settle(seconds: float = 0.5):
    app = _app()
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)


def _wait(condition, timeout: float = 300.0) -> bool:
    app = _app()
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


def _window(example: str, tmp: str):
    from .app.window import MainWindow
    _app()
    path = Path(shutil.copy(EXAMPLES / example, Path(tmp) / example))
    win = MainWindow(path)
    win.resize(*SIZE)
    win.show()
    _settle()
    return win


def _grab(win, out: Path):
    """The window, with the off-screen 3D view pasted on its placeholder."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QLabel
    _settle()
    pix = win.grab()
    label = next((w for w in win.viewer.findChildren(QLabel) if w.text() == "(off-screen 3D view)"), None)
    plotter = win.viewer.plotter
    if label is not None and plotter is not None:
        ratio = pix.devicePixelRatio()
        w, h = label.width(), label.height()
        plotter.window_size = [int(w * ratio), int(h * ratio)]
        plotter.render()
        img = plotter.screenshot(None, return_img=True)
        img = img.copy(order="C")
        qimg = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
        qimg.setDevicePixelRatio(ratio)
        painter = QPainter(pix)
        painter.drawImage(label.mapTo(win, QPoint(0, 0)), qimg)
        painter.end()
    pix.save(str(out))


def _condition_path(model, ctype):
    for k, md in enumerate(model["models"]):
        for i, mod in enumerate(md.get("modules", [])):
            for j, c in enumerate(mod.get("conditions", [])):
                if c["type"] == ctype:
                    return ("models", k, "modules", i, "conditions", j)
    return None


def window_2d(win):
    """The window: a boundary condition selected, its support highlighted."""
    win.select(_condition_path(win.model, "traction"))


def polygon_handles(win):
    """A polygon selected in the tree: its handles."""
    shape = win.model["geometries"][0]["source"]["shape"]
    path = next((p for p, s in _shapes(shape, ("geometries", 0, "source", "shape")) if s["type"] == "polygon"))
    win.select(path)
    _wide_form(win)


def _wide_form(win):
    """The form wide enough to be seen whole."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter
    split = next(w for w in win.findChildren(QSplitter) if w.orientation() == Qt.Orientation.Horizontal)
    split.setSizes([230, 470, 580])


def _shapes(shape, path):
    yield path, shape
    for i, s in enumerate(shape.get("of", [])):
        yield from _shapes(s, path + ("of", i))


def preview_3d(win):
    """A 3D CSG geometry previewed without meshing."""
    win.select(("geometries", 0))


def faces_3d(win):
    """A selection of picked faces of a 3D geometry."""
    win.select(("geometries", 0, "selections", "hole"))


def surface(win):
    """A surface: its flat domain and height."""
    win.select(("geometries", 0))


def mesh_layers(win):
    """A mesh in layers (prisms) of an extrusion."""
    win.select(("meshes", 0))
    win.build_mesh()
    _wait(lambda: not win.jobs.running and win.viewer is not None)
    _settle(1.0)
    win.select(("meshes", 0))


def curve(win):
    """The wire of heated_wire.json, its polygon selected, with its dimensions."""
    win.select(("geometries", 0, "source", "shape"))
    win.viewer.dims_button.setChecked(True)


def mesh_sizes(win):
    """The plate with a hole meshed finer around the hole and in a band."""
    geom = win.model["geometries"][0]
    geom["selections"]["band"] = {"kind": "domain", "criteria": {"type": "in_box", "min": [0.14, 0.0],
                                                                 "max": [0.17, 0.1]}}
    mesh = win.model["meshes"][0]
    mesh["order"] = 1
    mesh["sizes"] = [{"selection": "hole", "size": 0.002}, {"selection": "band", "size": 0.004}]
    win._changed(select=("meshes", 0, "sizes", 0))
    win.build_mesh()
    _wait(lambda: not win.jobs.running and win.viewer is not None)
    _settle(1.0)
    win.select(("meshes", 0, "sizes", 0))


def mesh_quality(win):
    """The worst tetrahedra of a 3D mesh, in the mesh made transparent."""
    mesh_layers(win)
    win.viewer.quality_box.setValue(0.5)
    win.viewer.quality_button.setChecked(True)


def results(win):
    """The von Mises stress of the plate with a hole."""
    win.compute()
    _wait(lambda: win.results is not None and not win.jobs.running)
    win.log.clear()                     # the traces of GetFEM name the build folders
    win.select((E.RESULTS, "mechanics", "von_mises"))


def anisotropic(win):
    """The temperature in a fibre composite plate."""
    win.compute()
    _wait(lambda: win.results is not None and not win.jobs.running)
    win.log.clear()
    win.select((E.RESULTS, "heat", "T"))


def work_plane(win):
    """A boss put on the top face of the block: its profile drawn on the
    section of the block."""
    root = ("geometries", 0, "source", "shape")
    win.select(root)
    win._panel_action("pick_plane:boss")
    _settle()
    surf, refs = win.viewer._faces
    top = next(i for i, r in enumerate(refs) if r.get("face") == "zmax")
    win.viewer.last_face = top
    win._face_clicked(refs[top])
    prof = win.current_path()
    win._panel_action("add_shape:ball")
    win.select(prof)


def work_plane_3d(win):
    """The profile of the boss drawn on its plane, in the 3D view."""
    work_plane(win)
    win.viewer._view("view_isometric")
    from PySide6.QtWidgets import QToolButton
    v = win.viewer
    v.findChild(QToolButton, "draw:rectangle").setChecked(True)
    v.draw_begin([-0.06, -0.01, 0.0])
    v.draw_move([0.0, 0.01, 0.0])
    v.draw_end([0.0, 0.01, 0.0])


def constraints(win):
    """A polygon drawn roughly, then constrained: a horizontal base, parallel
    sides, a length and an angle; the angle selected."""
    v = win.viewer
    win.select(("geometries",))
    win._panel_action("add_geometry:csg2")
    root = win.current_path() + ("source", "shape")
    win.select(root)
    win._panel_action("add_shape:polygon")
    poly = win.current_path()
    for p in ([0.0, 0.0], [0.42, 0.03], [0.5, 0.31], [0.12, 0.27], [0.0, 0.0]):
        v.draw_begin(p)
        v.draw_end(p)
    for t in ("horizontal", "parallel", "length", "angle"):
        win.select(poly)
        win._panel_action("constrain:" + t)
    c = E.get_at(win.model, poly)["constraints"]
    c[0]["side"] = 1
    c[2]["value"] = 0.4
    c[3]["value"] = 60
    win._changed(select=poly + ("constraints", 3))


def dimensions(win):
    """The constrained polygon with its dimensions shown: the base imposed,
    the other sides measured."""
    constraints(win)
    win.select(E.constraint_owner(win.current_path()))
    win.viewer.dims_button.setChecked(True)


def array_3d(win):
    """The hole of the block copied in a linear array: its selection holds
    the three holes."""
    hole = ("geometries", 0, "source", "shape", "of", 1)
    win.select(hole)
    win._panel_action("transform:array")
    _settle()
    arr = E.get_at(win.model, hole)
    arr.update(count=3, step=["Lx/4", 0, 0])
    arr["of"][0]["origin"][0] = "Lx/4"
    win._changed(select=hole)
    _wide_form(win)


SHOTS = {
    "window.png": ("plate_hole_tension.json", window_2d),
    "polygon_handles.png": ("l_bracket.json", polygon_handles),
    "preview_3d.png": ("heat_sink_3d.json", preview_3d),
    "faces_3d.png": ("block_with_hole_3d.json", faces_3d),
    "work_plane.png": ("block_with_hole_3d.json", work_plane),
    "work_plane_3d.png": ("block_with_hole_3d.json", work_plane_3d),
    "constraints.png": ("plate_with_hole.json", constraints),
    "dimensions.png": ("plate_with_hole.json", dimensions),
    "array_3d.png": ("block_with_hole_3d.json", array_3d),
    "surface.png": ("heated_dish.json", surface),
    "mesh_layers.png": ("heat_sink_3d.json", mesh_layers),
    "mesh_quality.png": ("block_with_hole_3d.json", mesh_quality),
    "mesh_sizes.png": ("plate_with_hole.json", mesh_sizes),
    "curve.png": ("heated_wire.json", curve),
    "results.png": ("plate_hole_tension.json", results),
    "anisotropic.png": ("anisotropic_plate.json", anisotropic),
}


def make(directory: str | Path = DEFAULT_DIR, names=None) -> list[Path]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for name, (example, action) in SHOTS.items():
        if names and name not in names and name.removesuffix(".png") not in names:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            win = _window(example, tmp)
            action(win)
            _grab(win, d / name)
            win.setWindowModified(False)
            win.close()
            out.append(d / name)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    directory = args.pop(0) if args and not args[0].endswith(".png") and "/" in args[0] else DEFAULT_DIR
    for p in make(directory, args):
        print("written:", p)
