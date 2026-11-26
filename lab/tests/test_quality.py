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

"""The quality of the elements of a mesh (quality.py) and its display:
the measures on regular, distorted and flat elements of each kind; the
view (all the elements colored in 2D, only the worst ones in 3D); the
Quality button of the viewer; the vector export of the 2D display.

    python -m pytest tests/test_quality.py      (or: python tests/test_quality.py)
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import quality as Q  # noqa: E402

S3 = np.sqrt(3) / 2


def one(points, vtk_type, n=None):
    pts = np.asarray(points, float)
    return float(Q.cell_quality(pts, [vtk_type], [list(range(n or len(pts)))])[0])


def test_regular_elements():
    """1 for the regular element of each kind, whatever its size, position
    and the orientation of its numbering."""
    R = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])          # a rotation
    for kind, t in (("quad", 9), ("hex", 12), ("wedge", 13), ("pyramid", 14)):
        X = Q._REGULAR[kind][0]
        assert abs(one(3.0 * X @ R.T + [1, 2, 3], t) - 1) < 1e-12, kind
        assert abs(one(X * [1, 1, -1], t) - 1) < 1e-12, kind          # mirrored: the other orientation
    assert abs(one([[0, 0], [1, 0], [0.5, S3]], 5) - 1) < 1e-12
    assert abs(one([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], 10) - 1) < 1e-12
    assert one([[0, 0], [1, 0]], 3) == 1                               # a segment: no measure


def test_distorted_elements():
    """Below 1 for a distorted element, 0 for a flat or folded one."""
    assert abs(one([[0, 0], [1, 0], [0, 1]], 5) - 2 * (np.sqrt(2) - 1)) < 1e-12   # right isosceles: 0.828
    assert one([[0, 0], [1, 0], [2, 0]], 5) == 0                       # flat
    assert one([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], 10) == 0  # flat tetrahedron
    assert 0 < one([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 10) < 1
    assert one([[0, 0], [2, 0], [0.4, 0.4], [0, 2]], 9) == 0           # a quadrangle not convex
    assert abs(one([[0, 0], [2, 0], [2, 1], [0, 1]], 9) - 1) < 1e-12   # a rectangle: right angles
    X = Q._REGULAR["hex"][0].copy()
    X[4:, 0] += 1                                                      # sheared by 45 degrees
    assert abs(one(X, 12) - np.sqrt(0.5)) < 1e-12
    # quadratic cells: measured by their corners
    tri6 = [[0, 0], [1, 0], [0.5, S3], [0.5, 0], [0.75, S3 / 2], [0.25, S3 / 2]]
    assert abs(one(tri6, 22) - 1) < 1e-12
    # a quadrangle in 3D (a surface mesh), and a cell of a kind not measured
    q = Q.cell_quality([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [5, 5, 5]], [9, 1], [[0, 1, 2, 3], [4]])
    assert abs(q[0] - 1) < 1e-12 and np.isnan(q[1])
    assert Q.summary(np.array([0.2, 0.9, np.nan]), 0.3) == "quality min 0.2, mean 0.55; 1 of 2 elements below 0.3"


def _meshes(tmp):
    """A 2D mesh of triangles with one flat triangle, a 3D mesh of
    tetrahedra (a cube in 6) with a distorted one, written as VTU files."""
    import pyvista as pv
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.02, 0]], float)
    tris = [[0, 4, 3], [4, 2, 3], [4, 1, 2], [0, 1, 4]]                # the last one nearly flat
    m2 = pv.UnstructuredGrid(np.hstack([[3] + t for t in tris]), np.full(4, 5, np.uint8), P)
    C = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    tets = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
    C = np.vstack([C, [[0.5, 0.5, 0.02]]])
    tets.append([0, 1, 2, 8])                                         # nearly flat
    m3 = pv.UnstructuredGrid(np.hstack([[4] + t for t in tets]), np.full(len(tets), 10, np.uint8), C)
    f2, f3 = Path(tmp) / "m2.vtu", Path(tmp) / "m3.vtu"
    m2.save(str(f2))
    m3.save(str(f3))
    return f2, f3


def test_scene():
    """2D: all the elements colored by their quality, the worst outlined;
    3D: the mesh translucent, only the worst elements drawn."""
    try:
        import pyvista as pv
    except ImportError:
        return
    from getfem_lab.app import scene
    with tempfile.TemporaryDirectory() as tmp:
        f2, f3 = _meshes(tmp)
        pl = pv.Plotter(off_screen=True, window_size=(400, 300))
        m2 = scene.read(f2)
        q = scene.mesh_quality(m2)
        assert q.shape == (4,) and q[3] < 0.1 < 0.5 < q[:3].min()
        text = scene.show_quality(pl, m2, 0.3)
        assert text.endswith("1 of 4 elements below 0.3") and "quality_worst" in pl.actors
        assert np.allclose(pl.actors["mesh"].mapper.dataset.cell_data["quality"], q)
        assert scene.show_quality(pl, m2, 0.001).endswith("0 of 4 elements below 0.001")
        assert "quality_worst" not in pl.actors
        m3 = scene.read(f3)
        text = scene.show_quality(pl, m3, 0.3)
        assert text.endswith("1 of 7 elements below 0.3")
        worst = pl.actors["quality_worst"].mapper.dataset
        assert worst.n_cells == 1 and pl.actors["mesh"].prop.opacity < 0.5
        pl.close()


def test_viewer():
    """The Quality button of the view: shown with a mesh, the threshold
    with it; the summary in the status line; the 2D display exported as
    vector graphics (one color per element)."""
    try:
        from PySide6.QtWidgets import QApplication
        import pyvista  # noqa: F401
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"), GETFEM_LAB_NO_3D="",
                      GETFEM_LAB_OFFSCREEN_3D="1")
    from getfem_lab.app.viewer import Viewer
    from getfem_lab.app import vector2d
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        f2, f3 = _meshes(tmp)
        v = Viewer()
        try:
            if v.plotter is None:
                return
            v.show_mesh(f2, status="Mesh m2")
            assert not v.quality_button.isHidden() and v.quality_box.isHidden()
            assert "quality_worst" not in v.plotter.actors
            v.quality_button.setChecked(True)
            assert not v.quality_box.isHidden() and "quality_worst" in v.plotter.actors
            assert v.status.text() == "Mesh m2 — quality min 0.00319, mean 0.598; 1 of 4 elements below 0.3"
            v.quality_box.setValue(0.0)
            assert v.status.text().endswith("0 of 4 elements below 0")
            parts = vector2d.planar_parts(v.plotter)
            faces = next(p for p in parts if p[0] == "faces")
            assert faces[3] is not None and len(faces[3]) == 3 * len(faces[1])     # a value per corner
            svg = Path(tmp) / "q.svg"
            assert v.export_view(svg) == "vector" and svg.stat().st_size > 0
            v.show_mesh(f3, status="Mesh m3")
            assert v.status.text().startswith("Mesh m3 — quality") and "quality_worst" not in v.plotter.actors
            v.quality_box.setValue(0.3)
            assert "quality_worst" in v.plotter.actors
            v.show_message("nothing")
            assert v.quality_button.isHidden() and v.quality_box.isHidden()
            v.show_mesh(f3, status="Mesh m3")
            assert not v.quality_button.isHidden()
            import pyvista as pv
            field = Path(tmp) / "field.vtu"                       # a result field: no quality controls
            ds = pv.read(str(f2))
            ds.point_data["u"] = ds.points[:, 0]
            ds.save(str(field))
            assert v.edges_button.isHidden()                      # a mesh: always its edges
            v.show_field(field, "u")
            assert v.quality_button.isHidden() and v.quality_box.isHidden()
            assert not v.edges_button.isHidden()                  # a field: edges or not
        finally:
            v.close_plotter()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
