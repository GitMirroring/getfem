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

"""The display of curved (quadratic) elements: faces subdivided and placed
by the quadratic interpolation, edges drawn on their parabolas, the
selections of a curved mesh refined; the subdivision control of the view.

    python -m pytest tests/       (or: python tests/test_curved.py)
"""

import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import mesher, model as M  # noqa: E402

DISK = {"type": "ball", "center": [0, 0], "radius": 1.0}


def disk_model(order=2, h=0.5):
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "csg", "dim": 2, "shape": copy.deepcopy(DISK)}
    m["geometries"][0]["selections"] = {"rim": {"kind": "boundary", "criteria": {
        "type": "edges", "edges": [{"shape": [], "side": "circle"}]}}}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "getfem", "h": h, "order": order}
    return m


def _pv():
    try:
        import pyvista as pv
        from getfem_gui.app import scene
    except ImportError:
        return None, None
    return pv, scene


def test_quadratic_display():
    """A coarse disk of order 2: the faces and the edges drawn closer to the
    circle as the subdivision level grows; level 0, the corners only."""
    pv, scene = _pv()
    if pv is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        res = mesher.ensure_mesh(disk_model(), "mesh", tmp)
        assert res.ok, res.error
        ds = pv.read(str(res.view_file()))
        rim = pv.read(str(res.view_file("rim")))
    assert scene.is_curved(ds)
    gaps, edges = [], []
    for level in (0, 1, 2, 3):
        surf = scene.surface(ds, level)
        out = surf.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        gaps.append(float(np.max(1 - np.linalg.norm(out.cell_centers().points[:, :2], axis=1))))
        wire = scene._wire(ds, level)
        edges.append(wire.n_cells)
        k = 2 ** level if level else 1
        assert wire.n_points == wire.n_cells * (k + 1)           # each edge: 2^level segments
    assert gaps[0] > 0.05 and gaps[0] > gaps[1] > gaps[2] > gaps[3] and gaps[3] < 5e-3    # (order 2: not exact)
    assert len(set(edges)) == 1                                  # the same edges, more or less finely
    # level 0: from corner to corner (the points of the faces drawn)
    corners = np.asarray(scene.surface(ds, 0).points, float)
    W = np.asarray(scene._wire(ds, 0).points, float)
    assert np.max(np.min(np.linalg.norm(W[:, None] - corners[None], axis=2), axis=1)) < 1e-6
    # a mesh of solid elements: the edges of its boundary only (the inner ones would show
    # through the faces near the surface); a tetrahedron cut in four at an inner point
    import pyvista as pv
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, 0.25]], float)
    C = np.array([[0, 1, 2, 4], [0, 2, 3, 4], [0, 3, 1, 4], [1, 3, 2, 4]])
    lin = pv.UnstructuredGrid({pv.CellType.TETRA: C}, X)
    assert lin.extract_all_edges().n_cells == 10 and scene._wire(lin, 0).n_cells == 6
    pairs = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]          # the edges of a quadratic tetrahedron
    mids, Q = {}, list(X)
    conn = []
    for c in C:
        row = list(c)
        for i, j in pairs:
            key = tuple(sorted((c[i], c[j])))
            if key not in mids:
                mids[key] = len(Q)
                Q.append((X[c[i]] + X[c[j]]) / 2)
            row.append(mids[key])
        conn.append(row)
    quad = pv.UnstructuredGrid({pv.CellType.QUADRATIC_TETRA: np.array(conn)}, np.array(Q))
    assert scene.is_curved(quad) and scene._wire(quad, 2).n_cells == 6
    # the selection of a curved mesh: refined (4 segments for each face), on the circle
    assert rim.n_cells > 30 and np.max(np.abs(np.linalg.norm(rim.points[:, :2], axis=1) - 1)) < 1e-2
    # a linear mesh: unchanged
    with tempfile.TemporaryDirectory() as tmp:
        lin = pv.read(str(mesher.ensure_mesh(disk_model(order=1), "mesh", tmp).view_file()))
    assert not scene.is_curved(lin) and scene.surface(lin, 3) is lin


def test_view_control():
    """The Curved control of the view: shown for a curved mesh, it redraws
    with its level; hidden for a geometry."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="", GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "disk.json"
        M.save(disk_model(), path)
        win = MainWindow(path)
        try:
            v = win.viewer
            win.select(("meshes", 0))
            win.build_mesh()
            while win.jobs.running:
                app.processEvents()
            win.select(("meshes", 0))
            assert not v.curved_box.isHidden() and v.curved_box.value() == 2
            n2 = v.plotter.actors["edges"].GetMapper().GetInput().GetNumberOfPoints()
            v.curved_box.setValue(4)
            n4 = v.plotter.actors["edges"].GetMapper().GetInput().GetNumberOfPoints()
            assert n4 > 3 * n2
            # the faces drawn behind the edges (no edges hidden by the faces as the view turns)
            mapper = v.plotter.actors["mesh"].GetMapper()
            assert mapper.GetResolveCoincidentTopology() == 1                       # polygon offset
            from vtkmodules.vtkCommonCore import reference
            factor, units = reference(0.0), reference(0.0)
            mapper.GetRelativeCoincidentTopologyPolygonOffsetParameters(factor, units)
            assert float(factor) > 0 and float(units) > 0
            win.select(("geometries", 0))
            assert v.curved_box.isHidden()
        finally:
            win.setWindowModified(False)
            win.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
