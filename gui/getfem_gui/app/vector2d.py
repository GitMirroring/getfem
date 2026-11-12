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

"""The vector export of the view: what the plotter shows (its actors: faces
with their colors or a colormap, lines, glyphs) drawn again by matplotlib
as PDF, SVG or EPS. A 2D view in its plane: Gouraud triangles for the
fields, segments for the edges and the isolines, the color bar and the
title. A 3D view as projected by its camera, by the painter's algorithm:
the faces shaded and the lines sorted from the farthest, the back faces of
closed surfaces and the hidden edges left out. (The GL2PS exporter of VTK
only gives an image of the 3D props with its OpenGL 3 renderer.)"""

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
            raise ValueError("not a 2D view")
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


# ------------------------------------------------------------------ 3D views

def _matrix(m) -> np.ndarray:
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def _apply(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Homogeneous transform of points (n x 3) -> (n x 4)."""
    return np.hstack([X, np.ones((len(X), 1))]) @ M.T


def _lut_colors(lut, values: np.ndarray) -> np.ndarray:
    """RGBA of values through a lookup table of VTK (its table and range)."""
    n = lut.GetNumberOfTableValues()
    table = np.array([lut.GetTableValue(i) for i in range(n)], float)
    lo, hi = lut.GetRange()
    t = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
    return table[np.clip((t * n).astype(int), 0, n - 1)]


def _closed(F: np.ndarray) -> bool:
    """Triangles forming closed surfaces (each edge shared by two)."""
    if not len(F):
        return False
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1).astype(np.int64)
    code = E[:, 0] * (int(E.max()) + 1) + E[:, 1]
    _, count = np.unique(code, return_counts=True)
    return bool(np.all(count >= 2))


def _outward(poly):
    """The triangles of a closed surface ordered as seen from outside."""
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
    f = vtkPolyDataNormals()
    f.SetInputData(poly)
    f.ConsistencyOn()
    f.AutoOrientNormalsOn()
    f.SplittingOff()
    f.ComputePointNormalsOff()
    f.ComputeCellNormalsOn()
    f.Update()
    return f.GetOutput()


def view_items(plotter) -> dict:
    """What a 3D view shows, as items for a painter: polygons on the screen
    (pixels, n x 3 x 2: triangles, or segments as [a, b, b]), their depth
    (distance along the view), face and edge colors (RGBA), line widths;
    the back faces of closed surfaces and the edges of facing_wire turned
    away are left out; the faces shaded as by the headlight of VTK."""
    import pyvista as pv
    ren = plotter.renderer
    w, h = (int(v) for v in plotter.window_size)
    cam = ren.GetActiveCamera()
    P = _matrix(cam.GetCompositeProjectionTransformMatrix(w / max(h, 1), -1, 1))
    V = _matrix(cam.GetViewTransformMatrix())
    eye, parallel = np.array(cam.GetPosition()), bool(cam.GetParallelProjection())
    dop = np.array(cam.GetDirectionOfProjection())
    polys, depth, face, edge, width = [], [], [], [], []

    def screen(X):
        H = _apply(P, X)
        ok = H[:, 3] > 1e-12
        S = np.zeros((len(X), 2))
        S[ok] = (H[ok, :2] / H[ok, 3:4] + 1) * 0.5 * np.array([w, h])
        return S, ok

    def area(S, T):
        u, v = S[T[:, 1]] - S[T[:, 0]], S[T[:, 2]] - S[T[:, 0]]
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    sizes = []
    for name, actor in plotter.actors.items():
        if name.startswith(_SKIP_PREFIX) or not actor.GetVisibility() or not hasattr(actor, "GetProperty"):
            continue
        mapper = actor.GetMapper()
        data = _input(mapper) if mapper is not None else None
        if data is None:
            continue
        ds = pv.wrap(data)
        if not getattr(ds, "n_points", 0):
            continue
        poly = ds if isinstance(ds, pv.PolyData) else ds.extract_surface()
        M = _matrix(actor.GetMatrix())
        prop = actor.GetProperty()
        rgb = np.array(prop.GetColor(), float)
        alpha = float(prop.GetOpacity())
        if poly.GetNumberOfPolys() or poly.GetNumberOfStrips():
            tri = pv.wrap(poly.triangulate())
            F = np.array([r for r in cell_rows(tri.GetPolys()) if len(r) == 3], np.int64).reshape(-1, 3)
            if len(F):
                closed = _closed(F)
                if closed:
                    tri = pv.wrap(_outward(tri))
                    F = np.array([r for r in cell_rows(tri.GetPolys()) if len(r) == 3], np.int64).reshape(-1, 3)
                X = _apply(M, np.asarray(tri.points, float))[:, :3]
                S, ok = screen(X)
                keep = ok[F].all(axis=1)
                if closed:
                    keep &= area(S, F) > 0
                n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
                nn = np.linalg.norm(n, axis=1)
                keep &= nn > 0
                F, n, nn = F[keep], n[keep], nn[keep]
                C = X[F].mean(axis=1)
                colors = np.tile(np.append(rgb, alpha), (len(F), 1))
                if mapper.GetScalarVisibility() and mapper.GetLookupTable() is not None:
                    arr = mapper.GetArrayName() or "_shown"
                    vals = None
                    if arr in tri.point_data:
                        vals = np.asarray(tri.point_data[arr], float)
                        vals = (np.linalg.norm(vals, axis=1) if vals.ndim > 1 else vals)[F].mean(axis=1)
                    elif arr in tri.cell_data:
                        vals = np.asarray(tri.cell_data[arr], float)
                        vals = (np.linalg.norm(vals, axis=1) if vals.ndim > 1 else vals)[keep]
                    if vals is not None:
                        lut = mapper.GetLookupTable()
                        lut.SetRange(*mapper.GetScalarRange())
                        colors = _lut_colors(lut, vals)
                        colors[:, 3] = alpha
                if prop.GetLighting():
                    view = dop[None, :] if parallel else (C - eye)
                    view = view / np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-300)
                    cos = np.abs(np.einsum("ij,ij->i", n / nn[:, None], view))
                    # (the light kit of PyVista: nearly even, a little darker edge on)
                    shade = np.clip(prop.GetAmbient() + prop.GetDiffuse() * (0.72 + 0.28 * cos), 0, 1)
                    colors[:, :3] *= shade[:, None]
                polys.append(S[F])
                depth.append(-_apply(V, C)[:, 2])
                face.append(colors)
                edge.append(colors)
                width.append(np.full(len(F), 0.25))
                sizes.append(np.median(np.linalg.norm(X[F[:, 1]] - X[F[:, 0]], axis=1)) if len(F) else 0)
        if poly.GetNumberOfLines():
            X = _apply(M, np.asarray(poly.points, float))[:, :3]
            S, ok = screen(X)
            rows = cell_rows(poly.GetLines())
            seg = np.array([(a, b) for ids in rows for a, b in zip(ids[:-1], ids[1:])], np.int64).reshape(-1, 2)
            keep = ok[seg].all(axis=1)
            own = None                                   # facing_wire: the depth of its triangle
            if all(k in poly.point_data for k in ("tri0", "tri1", "tri2")):
                T3 = [_apply(M, np.asarray(poly.point_data[k], float)[seg[:, 0]])[:, :3]
                      for k in ("tri0", "tri1", "tri2")]
                T = [screen(t)[0] for t in T3]
                u, v = T[1] - T[0], T[2] - T[0]
                keep &= u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0] >= 0
                own = -_apply(V, (T3[0] + T3[1] + T3[2]) / 3)[:, 2]
            seg = seg[keep]
            if len(seg):
                polys.append(S[seg[:, [0, 1, 1]]])
                if own is not None:                      # just after its triangle
                    d = own[keep]
                    depth.append(d - 1e-6 * max(float(np.ptp(d)), 1e-300))
                    sizes.append(0.0)
                else:                                    # (biased below)
                    depth.append(-_apply(V, X[seg].mean(axis=1))[:, 2])
                    sizes.append(-1.0)
                face.append(np.zeros((len(seg), 4)))
                edge.append(np.tile(np.append(rgb, alpha), (len(seg), 1)))
                width.append(np.full(len(seg), max(0.3, 0.6 * float(prop.GetLineWidth()))))
    if not polys:
        raise ValueError("nothing to export in the view")
    # the other lines drawn after the faces about them: depth less a typical element
    bias = float(np.median([s for s in sizes if s > 0])) if any(s > 0 for s in sizes) else 0.0
    is_line = [np.full(len(d), s < 0) for d, s in zip(depth, sizes)]
    D = np.concatenate(depth) - bias * np.concatenate(is_line)
    order = np.argsort(-D, kind="stable")                 # far first
    return {"polys": np.concatenate(polys)[order], "face": np.concatenate(face)[order],
            "edge": np.concatenate(edge)[order], "width": np.concatenate(width)[order], "size": (w, h)}


def export_3d(plotter, path: str | Path, title: str = "", label: str = "") -> None:
    """A 3D view drawn by matplotlib as vector graphics, as on the screen
    (the same framing, a white background), by the painter's algorithm:
    the faces and lines sorted from the farthest; the color bar of the
    field and the title."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.figure import Figure
    items = view_items(plotter)
    w, h = items["size"]
    fig = Figure(figsize=(w / 96, h / 96))
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.add_collection(PolyCollection(items["polys"], closed=True, facecolors=items["face"],
                                     edgecolors=items["edge"], linewidths=items["width"]))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_axis_off()
    field = plotter.actors.get("field")
    if field is not None and field.GetMapper().GetLookupTable() is not None:
        lut = field.GetMapper().GetLookupTable()
        n = lut.GetNumberOfTableValues()
        cmap = ListedColormap([lut.GetTableValue(i)[:3] for i in range(n)])
        lo, hi = field.GetMapper().GetScalarRange()
        cax = fig.add_axes((0.87, 0.12, 0.025, 0.6))
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(lo, hi), cmap=cmap), cax=cax)
        bar.ax.tick_params(labelsize=8)
        if label:
            bar.set_label(label, fontsize=9)
    if title:
        fig.text(0.01, 0.98, title, ha="left", va="top", fontsize=9, color="#444444")
    fig.savefig(str(path), facecolor="white")
