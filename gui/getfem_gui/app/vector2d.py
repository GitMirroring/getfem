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

"""The vector export of a 2D view without the GL2PS module of VTK: what the
plotter shows (its actors: faces with their colors or a colormap, lines,
glyphs) drawn again by matplotlib, in the plane of the view, as PDF, SVG or
EPS: Gouraud triangles for the fields, segments for the edges and the
isolines, the color bar and the title."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# the actors not drawn: the text, the handles and helpers of the editing
_SKIP_PREFIX = ("handle", "rubber", "draw", "grid", "axes", "title", "cut_ends", "hover")


def _rgb(prop) -> tuple:
    return tuple(float(c) for c in prop.GetColor())


def _lut_cmap(lut):
    """The colors of a lookup table of VTK as a matplotlib colormap."""
    from matplotlib.colors import ListedColormap
    n = lut.GetNumberOfTableValues()
    return ListedColormap([lut.GetTableValue(i)[:3] for i in range(n)])


def cell_rows(cells) -> list:
    """The point indices of each cell of a vtkCellArray (polygons, lines),
    read through VTK (the same in all versions of PyVista)."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    if cells is None or not cells.GetNumberOfCells():
        return []
    off = vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int64)
    conn = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64)
    return [conn[a:b] for a, b in zip(off[:-1], off[1:])]


def _input(mapper):
    """The dataset drawn by a mapper, given as data or through a pipeline
    (recent PyVista: the mesh with its scalars through a filter), updated."""
    if mapper.GetNumberOfInputConnections(0):
        alg = mapper.GetInputAlgorithm()
        if alg is not None:
            alg.Update()
        return mapper.GetInputDataObject(0, 0)
    return mapper.GetInput() if hasattr(mapper, "GetInput") else None


def planar_parts(plotter) -> list:
    """The parts of a 2D view, back to front: ("faces", triangles (m x 3),
    points (n x 2), values or None, cmap, clim, color, opacity, name),
    ("lines", segments (k x 2 x 2), color, width, opacity, name). Raises
    ValueError for a view that is not planar."""
    import pyvista as pv
    out = []
    for name, actor in plotter.actors.items():
        if name.startswith(_SKIP_PREFIX) or not actor.GetVisibility():
            continue
        mapper = actor.GetMapper()
        data = _input(mapper) if mapper is not None else None
        if data is None:
            continue
        ds = pv.wrap(data)
        if not getattr(ds, "n_points", 0):
            continue
        P = np.asarray(ds.points, float)
        if np.ptp(P[:, 2]) > 1e-9 * max(np.ptp(P[:, :2]), 1e-300):
            raise ValueError("a 3D view: its vector export needs the GL2PS module of VTK")
        prop = actor.GetProperty()
        poly = ds if isinstance(ds, pv.PolyData) else ds.extract_surface()
        if poly.GetNumberOfPolys() or poly.GetNumberOfStrips():
            tri = poly.triangulate()
            rows = [r for r in cell_rows(tri.GetPolys()) if len(r) == 3]
            faces = np.array(rows, np.int64).reshape(-1, 3)
            values, cmap, clim = None, None, None
            if mapper.GetScalarVisibility() and mapper.GetLookupTable() is not None:
                arr = mapper.GetArrayName() or "_shown"
                if arr in tri.point_data:
                    values = np.asarray(tri.point_data[arr], float)
                    if values.ndim > 1:
                        values = np.linalg.norm(values, axis=1)
                    cmap, clim = _lut_cmap(mapper.GetLookupTable()), tuple(mapper.GetScalarRange())
            if len(faces):
                out.append(("faces", faces, np.asarray(tri.points)[:, :2], values, cmap, clim, _rgb(prop),
                            float(prop.GetOpacity()), name))
        if poly.GetNumberOfLines():
            Q = np.asarray(poly.points)[:, :2]
            segs = [np.stack([Q[ids[:-1]], Q[ids[1:]]], axis=1) for ids in cell_rows(poly.GetLines()) if len(ids) > 1]
            if segs:
                out.append(("lines", np.concatenate(segs), _rgb(prop), float(prop.GetLineWidth()),
                            float(prop.GetOpacity()), name))
    order = {"mesh": 0, "field": 0, "domain": 0, "undeformed": 1, "selection": 2}
    out.sort(key=lambda p: (p[0] == "lines", order.get(p[-1], 3)))
    return out


def export_planar(plotter, path: str | Path, title: str = "", label: str = "", size=(8.0, 6.0)) -> None:
    """The 2D view drawn by matplotlib into path (PDF, SVG, EPS, PS: the
    format by its extension), its field with a color bar."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.tri import Triangulation
    parts = planar_parts(plotter)
    if not parts:
        raise ValueError("nothing to export in the view")
    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    mappable = None
    for p in parts:
        if p[0] == "faces":
            _, faces, X, values, cmap, clim, color, alpha, name = p
            tri = Triangulation(X[:, 0], X[:, 1], faces)
            if values is not None:
                mappable = ax.tripcolor(tri, values, cmap=cmap, vmin=clim[0], vmax=clim[1], shading="gouraud",
                                        alpha=alpha, rasterized=False)
            else:
                ax.tripcolor(tri, np.zeros(len(X)), cmap=matplotlib.colors.ListedColormap([color]),
                             shading="flat", alpha=alpha, edgecolors="none")
        else:
            _, segs, color, width, alpha, name = p
            ax.add_collection(LineCollection(segs, colors=[color], linewidths=max(0.3, 0.6 * width), alpha=alpha))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
    if mappable is not None:
        bar = fig.colorbar(mappable, ax=ax, shrink=0.8)
        if label:
            bar.set_label(label)
    if title:
        ax.set_title(title, fontsize=10)
    fig.savefig(str(path), bbox_inches="tight")
