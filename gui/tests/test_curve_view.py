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

"""The view of the meshes of curves (elements of dimension 1, in 2D or 3D):
the mesh (thick lines, the ends of the elements as dots, a selection
highlighted), a field on it, its vector export; no quality controls.

    python -m pytest tests/test_curve_view.py      (or: python tests/test_curve_view.py)
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, mesher, model as M, runner  # noqa: E402


def curve_model(kind, order=1):
    m = M.default_model()
    m["geometries"][0]["source"] = E._default_source("curve", int(kind[-1]))
    m["geometries"][0]["selections"] = {}
    m["meshes"][0] = {"name": "mesh", "geometry": m["geometries"][0]["name"], "mesher": "getfem", "h": 0.1,
                      "order": order}
    m["models"][0]["modules"][0]["conditions"] = [{"type": "temperature", "selection": "start", "value": "0"},
                                                  {"type": "temperature", "selection": "end", "value": "1"}]
    m["models"][0]["outputs"] = []
    return m


def test_scene():
    """Each curve (2D, 3D; linear, quadratic): its elements drawn as lines,
    their ends as dots (one more than the elements of an open curve), the
    selection of an end highlighted; the temperature on it."""
    try:
        import pyvista as pv
        import getfem  # noqa: F401
    except ImportError:
        return
    from getfem_gui.app import scene
    for kind, order in (("curve2", 1), ("curve2", 2), ("curve3", 1), ("curve3", 2)):
        m = curve_model(kind, order)
        assert M.validate(m) == []
        with tempfile.TemporaryDirectory() as tmp:
            res = mesher.ensure_mesh(m, "mesh", tmp)
            assert res.ok, res.error
            mesh = scene.read(res.view_file())
            assert scene.is_curve(mesh) and not scene.is_curve(pv.Sphere().cast_to_unstructured_grid())
            pl = pv.Plotter(off_screen=True, window_size=(300, 200))
            scene.show_mesh(pl, mesh, scene.read(res.view_file("end")))
            assert pl.actors["mesh"].mapper.dataset.n_points >= mesh.n_points
            n = res.info["nb_elements"]
            assert pl.actors["edges"].mapper.dataset.n_points == n + 1
            assert pl.actors["selection"].mapper.dataset.n_points == 1
            result = runner.run(m, tmp)
            assert result.ok, result.error
            ds = scene.read(Path(tmp) / "heat.vtu")
            scene.show_field(pl, ds, "T", None, edges=True)
            assert "field" in pl.actors and pl.actors["edges"].mapper.dataset.n_points == n + 1
            pl.close()


def test_viewer():
    """In the viewer: the mesh of a curve shown (no quality controls), and
    exported as vector graphics."""
    try:
        from PySide6.QtWidgets import QApplication
        import pyvista  # noqa: F401
        import getfem  # noqa: F401
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"), GETFEM_GUI_NO_3D="",
                      GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.viewer import Viewer
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        res = mesher.ensure_mesh(curve_model("curve2"), "mesh", tmp)
        v = Viewer()
        try:
            if v.plotter is None:
                return
            v.show_mesh(res.view_file(), status="Mesh")
            assert "mesh" in v.plotter.actors and v.quality_button.isHidden()
            assert v.status.text() == "Mesh"
            svg = Path(tmp) / "curve.svg"
            assert v.export_view(svg) == "vector" and b"<path" in svg.read_bytes()
        finally:
            v.close_plotter()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
