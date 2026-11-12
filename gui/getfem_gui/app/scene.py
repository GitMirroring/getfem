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


"""3D scenes drawn with PyVista: mesh, highlighted selection, result field.

These functions only need a PyVista plotter, so that they can be used (and
tested) off screen as well as in the Qt window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv

SURFACE = "#dfe3e8"
EDGES = "#5b6472"
HIGHLIGHT = "#e4572e"
REFERENCE = "#5b9bd5"         # the section of the rest of a 3D geometry under a profile
FADED = "#eef0f3"
CMAP = "viridis"

_cache: dict[tuple[str, float], pv.DataSet] = {}


def read(path: str | Path) -> pv.DataSet:
    """Read a VTU file, cached as long as the file is not modified."""
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime)
    if key not in _cache:
        for k in [k for k in _cache if k[0] == key[0]]:
            del _cache[k]
        _cache[key] = pv.read(str(path))
    return _cache[key]


def is_planar(ds: pv.DataSet) -> bool:
    b = ds.bounds
    return abs(b[5] - b[4]) <= 1e-12 * max(abs(b[1] - b[0]), abs(b[3] - b[2]), 1e-300)


def size(ds: pv.DataSet) -> float:
    b = np.asarray(ds.bounds)
    return float(np.linalg.norm(b[1::2] - b[0::2])) or 1.0


def field_names(ds: pv.DataSet) -> dict[str, int]:
    """{name: number of components} of the point and cell arrays."""
    out = {}
    for data in (ds.point_data, ds.cell_data):
        for name in data.keys():
            if not name.startswith("vtk"):
                arr = data[name]
                out[name] = 1 if arr.ndim == 1 else arr.shape[1]
    return out


def set_style(plotter, planar: bool) -> None:
    """Mouse interaction: pan and zoom for a planar model (rotation only with
    Ctrl), free rotation in 3D. Nothing to do for an off-screen plotter."""
    try:
        plotter.enable_2d_style() if planar else plotter.enable_trackball_style()
    except (AttributeError, TypeError):
        pass


def _camera(plotter, ds, reset: bool) -> None:
    if not reset:
        return
    planar = is_planar(ds)
    if planar:
        plotter.view_xy()
        plotter.enable_parallel_projection()
    else:
        plotter.disable_parallel_projection()
        plotter.view_isometric()
    set_style(plotter, planar)
    plotter.reset_camera()


# ------------------------------------------------------------------ curved (quadratic) elements

CURVED = 2              # default subdivision level of the display of quadratic elements

# the edges of the quadratic cells of VTK: (corner, middle node, corner)
_QUADRATIC_EDGES = {
    21: [(0, 2, 1)],                                                          # edge
    22: [(0, 3, 1), (1, 4, 2), (2, 5, 0)],                                    # triangle
    23: [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)],                         # quadrangle
    24: [(0, 4, 1), (1, 5, 2), (2, 6, 0), (0, 7, 3), (1, 8, 3), (2, 9, 3)],   # tetrahedron
    25: [(0, 8, 1), (1, 9, 2), (2, 10, 3), (3, 11, 0), (4, 12, 5), (5, 13, 6), (6, 14, 7), (7, 15, 4),
         (0, 16, 4), (1, 17, 5), (2, 18, 6), (3, 19, 7)],                     # hexahedron
    26: [(0, 6, 1), (1, 7, 2), (2, 8, 0), (3, 9, 4), (4, 10, 5), (5, 11, 3),
         (0, 12, 3), (1, 13, 4), (2, 14, 5)],                                 # wedge
}
_QUADRATIC_EDGES[28] = _QUADRATIC_EDGES[23]          # biquadratic quadrangle
_QUADRATIC_EDGES[29] = _QUADRATIC_EDGES[25]          # triquadratic hexahedron
_QUADRATIC_EDGES[32] = _QUADRATIC_EDGES[26]          # biquadratic wedge


def is_curved(ds: pv.DataSet) -> bool:
    """The dataset has quadratic cells (a mesh of geometric order 2)."""
    types = getattr(ds, "celltypes", None)
    return types is not None and len(types) > 0 and bool(np.isin(types, list(_QUADRATIC_EDGES)).any())


def surface(ds: pv.DataSet, level: int = CURVED) -> pv.DataSet:
    """What is drawn of a dataset: for quadratic cells, their outer surface,
    each face subdivided `level` times (its points placed by the quadratic
    interpolation, and the point data interpolated: curved faces, curved
    color maps); level 0, the faces through their corners."""
    if not is_curved(ds):
        return ds
    return ds.extract_surface(nonlinear_subdivision=max(0, int(level)))


def _wire(ds: pv.DataSet, level: int = CURVED) -> pv.DataSet:
    """Edges of the elements; for quadratic elements, each edge drawn by
    2^level segments on its parabola (straight between the corners for
    level 0, as the faces of surface())."""
    if not ds.n_cells:
        return ds
    if not is_curved(ds):
        return ds.extract_all_edges()
    triples = []
    types = np.asarray(ds.celltypes)
    for t, edges in _QUADRATIC_EDGES.items():
        if not np.any(types == t):
            continue
        conn = ds.cells_dict.get(t)
        if conn is None:
            continue
        conn = np.asarray(conn)
        triples += [conn[:, list(e)] for e in edges]
    if not triples:
        return ds.extract_all_edges()
    T = np.vstack(triples)
    key = np.sort(T[:, [0, 2]], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)          # each edge once
    T = T[np.sort(first)]
    k = 2 ** int(level) if level > 0 else 1          # level 0: from corner to corner, as the faces drawn
    t = np.linspace(0.0, 1.0, k + 1)
    w = np.stack([(1 - t) * (1 - 2 * t), 4 * t * (1 - t), t * (2 * t - 1)])   # quadratic Lagrange
    X = np.asarray(ds.points, float)
    P = np.einsum("jk,ejd->ekd", w, X[T])                          # edges x samples x 3
    n = len(T)
    ids = np.arange(n * (k + 1)).reshape(n, k + 1)
    lines = np.hstack([np.full((n, 1), k + 1), ids]).ravel()
    return pv.PolyData(P.reshape(-1, 3), lines=lines)


def _outline(ds: pv.DataSet, level: int = CURVED) -> pv.DataSet:
    """Contour of the domain: boundary edges in 2D, feature edges in 3D."""
    surf = ds.extract_surface(nonlinear_subdivision=level if is_curved(ds) else 0)
    if is_planar(ds):
        return surf.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                          manifold_edges=False, non_manifold_edges=False)
    return surf.extract_feature_edges(feature_angle=30)


def clear(plotter) -> None:
    plotter.clear()
    plotter.remove_scalar_bar() if plotter.scalar_bars else None


def message(plotter, text: str) -> None:
    clear(plotter)
    plotter.add_text(text, position="upper_left", font_size=10, color="#444444")


def behind_lines(actor):
    """The faces of an actor drawn a little behind the lines at the same
    place (the edges of the elements, the isolines): without it, the depth
    test lets the faces hide parts of the edges, which appear and
    disappear as the view turns. An offset of the depth of the polygons in
    the depth buffer (OpenGL polygon offset): no cost at drawing."""
    mapper = actor.GetMapper() if actor is not None else None
    if mapper is not None:
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(2.0, 4.0)
    return actor


def show_mesh(plotter, mesh: pv.DataSet, selection: pv.DataSet | None = None,
              title: str = "", reset: bool = True, edges: bool = True, level: int = CURVED) -> None:
    """The mesh, and optionally a selection (faces or elements) highlighted;
    level: the subdivisions of quadratic elements (see surface)."""
    clear(plotter)
    planar = is_planar(mesh)
    faded = selection is not None
    behind_lines(plotter.add_mesh(surface(mesh, level), color=FADED if faded else SURFACE,
                                  opacity=1.0 if planar or not faded else 0.35,
                                  show_edges=False, lighting=not planar, name="mesh"))
    if edges:
        plotter.add_mesh(_wire(mesh, level), color=EDGES, opacity=0.35 if faded else 1.0,
                         line_width=1, lighting=False, name="edges")
    if selection is not None and selection.n_points:
        dim = selection.get_cell(0).dimension if selection.n_cells else 0
        if dim == 1:   # boundary of a 2D domain
            plotter.add_mesh(selection, color=HIGHLIGHT, line_width=5, lighting=False,
                             render_lines_as_tubes=False, name="selection")
        else:
            behind_lines(plotter.add_mesh(surface(selection, level), color=HIGHLIGHT, show_edges=False,
                                          lighting=not planar, name="selection"))
            if edges:
                plotter.add_mesh(_wire(selection, level), color="#7a2a14", line_width=1,
                                 lighting=False, name="selection_edges")
    if title:
        plotter.add_text(title, position="upper_left", font_size=10, color="#444444", name="title")
    _camera(plotter, mesh, reset)


def deformation_scale(ds: pv.DataSet, name: str) -> float:
    """Factor such that the largest displacement is 10% of the model size."""
    u = np.asarray(ds.point_data[name])
    umax = float(np.max(np.linalg.norm(u.reshape(len(u), -1), axis=1))) if len(u) else 0.0
    return 0.1 * size(ds) / umax if umax > 0 else 1.0


from ..model import COLORMAPS as CMAPS  # noqa: E402
CUT = "#d62728"


def values(ds: pv.DataSet, name: str, component: int | None = None) -> tuple[np.ndarray, str]:
    """The scalar shown for a field (a component or the magnitude of a vector),
    at the points (cell values are averaged to the points), and its label."""
    if name in ds.point_data:
        arr = np.asarray(ds.point_data[name])
    else:
        arr = np.asarray(ds.cell_data_to_point_data().point_data[name])
    if arr.ndim == 1:
        return arr.astype(float), name
    if component is None:
        return np.linalg.norm(arr, axis=1), f"|{name}|"
    return arr[:, component].astype(float), f"{name} ({'xyz'[component] if component < 3 else component})"


def _with_values(ds: pv.DataSet, name: str, component, warp, scale) -> tuple[pv.DataSet, str]:
    """A copy of ds holding the shown scalar "_shown", deformed by `warp`."""
    shown = ds.copy()
    shown.point_data["_shown"], label = values(ds, name, component)
    if warp and warp in ds.point_data:
        vec = np.asarray(ds.point_data[warp])
        if vec.ndim == 2 and vec.shape[1] < 3:
            vec = np.hstack([vec, np.zeros((len(vec), 3 - vec.shape[1]))])
        shown.points = ds.points + scale * vec
    return shown, label


def isoline_levels(lo: float, hi: float, n: int) -> np.ndarray:
    """n levels strictly between lo and hi, evenly spaced."""
    if n <= 0 or not hi > lo:
        return np.array([])
    return np.linspace(lo, hi, n + 2)[1:-1]


def _arrows(shown: pv.DataSet, ds: pv.DataSet, field: str, count: int = 24) -> pv.PolyData | None:
    """Arrows of a vector field at the points of a regular grid (about `count`
    along the longest side) inside the shown domain."""
    if field not in ds.point_data and field not in ds.cell_data:
        return None
    src = shown.copy()
    vec = np.asarray(ds.point_data[field] if field in ds.point_data
                     else ds.cell_data_to_point_data().point_data[field], float)
    if vec.ndim != 2:
        return None
    if vec.shape[1] < 3:
        vec = np.hstack([vec, np.zeros((len(vec), 3 - vec.shape[1]))])
    src.point_data["_vec"] = vec
    b = np.asarray(src.bounds, float)
    lo, hi = b[0::2], b[1::2]
    step = (hi - lo).max() / count
    dims = [max(int(np.floor((h - l) / step)) + 1, 1) for l, h in zip(lo, hi)]
    origin = lo + 0.5 * ((hi - lo) - (np.array(dims) - 1) * step)
    grid = pv.ImageData(dimensions=dims, spacing=(step, step, step), origin=origin)
    probe = grid.sample(src)
    mask = np.asarray(probe.point_data["vtkValidPointMask"]).astype(bool)
    pts = pv.PolyData(np.asarray(probe.points)[mask])
    v = np.asarray(probe.point_data["_vec"])[mask]
    vmax = float(np.max(np.linalg.norm(v, axis=1))) if len(v) else 0.0
    if not len(v) or vmax <= 0:
        return None
    pts.point_data["_vec"] = v
    return pts.glyph(orient="_vec", scale="_vec", factor=0.9 * step / vmax,
                     geom=pv.Arrow(tip_length=0.3, tip_radius=0.12, shaft_radius=0.04))


def show_field(plotter, ds: pv.DataSet, name: str, component: int | None = None,
               warp: str | None = None, scale: float = 1.0, edges: bool = False,
               title: str = "", reset: bool = True, isolines: int = 0, bands: bool = False,
               cmap: str | None = None, clim=None, arrows: str | None = None, cut=None,
               level: int = CURVED) -> None:
    """A result field, possibly on the shape deformed by the field `warp`.

    isolines: number of isolines (black, evenly spaced in the range); bands:
    a banded colormap (one band between isolines, 10 without them); cmap: a
    colormap name (CMAPS); clim: (min, max) of the color range (None: the
    range of the values); arrows: a vector field drawn with arrows; cut: a
    segment (two points) drawn on the view (see line_values); level: the
    subdivisions of quadratic elements (see surface)."""
    clear(plotter)
    shown, label = _with_values(ds, name, component, warp, scale)
    curved = is_curved(shown)
    drawn = surface(shown, level)           # quadratic elements: subdivided, values interpolated
    if warp and warp in ds.point_data:
        plotter.add_mesh(_outline(ds, level), color="#7d8690", line_width=1, lighting=False,
                         name="undeformed")
    data = np.asarray(shown.point_data["_shown"])
    auto = (float(np.min(data)), float(np.max(data))) if data.size else (0.0, 1.0)
    clim = auto if clim is None else tuple(a if c is None else float(c) for c, a in zip(clim, auto))
    if not clim[1] > clim[0]:
        clim = (clim[0] - 0.5 * (abs(clim[0]) or 1.0), clim[0] + 0.5 * (abs(clim[0]) or 1.0))
    planar = is_planar(ds)
    field = plotter.add_mesh(drawn, scalars="_shown", cmap=cmap or CMAP, clim=clim,
                         n_colors=(isolines + 1 if isolines > 0 else 10) if bands else 256,
                         show_edges=edges and not curved, edge_color=EDGES, lighting=not planar,
                         scalar_bar_args={"title": label, "vertical": True, "position_x": 0.86,
                                          "position_y": 0.1, "height": 0.65, "width": 0.05,
                                          "fmt": "%.3g", "color": "#333333",
                                          "n_labels": (isolines + 2 if isolines + 2 <= 12 else 5) if bands and isolines
                                          else 11 if bands else 5,
                                          "title_font_size": 14, "label_font_size": 12},
                         name="field")
    behind_lines(field)                     # (the edges, the isolines in front of it)
    if edges and curved:                    # the edges of the elements, not of their subdivisions
        plotter.add_mesh(_wire(shown, level), color=EDGES, line_width=1, lighting=False, name="edges")
    levels = isoline_levels(*clim, isolines)
    if levels.size:
        surf = drawn.extract_surface(nonlinear_subdivision=0)
        lines = surf.contour(isosurfaces=levels, scalars="_shown")
        if lines.n_points:
            plotter.add_mesh(lines, color="#1a1a1a", line_width=1.5, lighting=False, name="isolines")
    if arrows:
        glyphs = _arrows(shown, ds, arrows)
        if glyphs is not None:
            plotter.add_mesh(glyphs, color="#222222", lighting=not planar, name="arrows")
    if cut is not None:
        a, b = (np.pad(np.asarray(p, float)[:3], (0, 3 - len(p[:3]))) for p in cut)
        plotter.add_mesh(pv.Line(a, b), color=CUT, line_width=3, lighting=False, name="cut")
        plotter.add_point_labels(np.array([a, b]), ["A", "B"], font_size=12, text_color=CUT,
                                 shape=None, show_points=True, point_color=CUT, point_size=8,
                                 always_visible=True, name="cut_ends")
    if title:
        plotter.add_text(title, position="upper_left", font_size=10, color="#444444", name="title")
    _camera(plotter, ds, reset)


def line_values(ds: pv.DataSet, name: str, component: int | None, a, b,
                n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """The shown scalar along the segment [a, b] (in the undeformed
    configuration): (distance from a, value), NaN outside the domain."""
    a, b = (np.pad(np.asarray(p, float)[:3], (0, 3 - len(p[:3]))) for p in (a, b))
    src = ds.copy()
    src.point_data["_shown"], _ = values(ds, name, component)
    line = src.sample_over_line(a, b, resolution=n)
    dist = np.linalg.norm(np.asarray(line.points) - a, axis=1)
    v = np.asarray(line.point_data["_shown"], float).copy()
    v[~np.asarray(line.point_data["vtkValidPointMask"]).astype(bool)] = np.nan
    return dist, v


def probe(ds: pv.DataSet, name: str, component: int | None, point) -> float | None:
    """The shown scalar at a point (None outside the domain)."""
    p = np.pad(np.asarray(point, float)[:3], (0, 3 - len(point[:3])))
    src = ds.copy()
    src.point_data["_shown"], _ = values(ds, name, component)
    out = pv.PolyData(p[None, :]).sample(src)
    if not bool(np.asarray(out.point_data["vtkValidPointMask"])[0]):
        return None
    return float(np.asarray(out.point_data["_shown"])[0])


# ------------------------------------------------------------------ 2D geometry preview

HOVER = "#f2a65a"
OUTLINE = "#33414f"
FILL = "#cfe0ea"
INTERFACE = "#7d8a96"


def _polyline(points: np.ndarray) -> pv.PolyData:
    pts = np.column_stack([points, np.zeros(len(points))])
    return pv.lines_from_points(pts)


def _fill(shape: dict, n: int = 240, level=None) -> pv.DataSet:
    """The domain (or where level < 0), as the part of a fine grid where the
    distance is negative (the boundary itself is drawn exactly, on top)."""
    from .. import geometry2d as G
    lo, hi = G.bounds(shape)
    margin = 0.02 * (hi - lo).max()
    lo, hi = lo - margin, hi + margin
    h = (hi - lo).max() / n
    dims = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    grid = pv.ImageData(dimensions=(int(dims[0]), int(dims[1]), 1), spacing=(h, h, 1.0),
                        origin=(lo[0], lo[1], 0.0))
    grid.point_data["d"] = (level or (lambda P: G.sdf(shape, P)))(grid.points[:, :2])
    return grid.clip_scalar(scalars="d", value=0.0, invert=True)


def reference_lines(reference) -> pv.PolyData | None:
    """The zero lines of a section grid (xs, ys, values) (geometry3d.section_grid),
    joined into polylines."""
    xs, ys, values = reference
    if len(xs) < 2 or len(ys) < 2:
        return None
    img = pv.ImageData(dimensions=(len(xs), len(ys), 1), spacing=(xs[1] - xs[0], ys[1] - ys[0], 1.0),
                       origin=(xs[0], ys[0], 0.0))
    img.point_data["d"] = np.asarray(values, float).ravel(order="F")
    lines = img.contour([0.0], scalars="d")
    if lines.n_points == 0:
        return None
    return lines.strip()


def reference_corners(lines: pv.PolyData | None, angle: float = 35.0) -> np.ndarray:
    """The ends and the corners (turn larger than angle) of reference lines."""
    if lines is None or lines.n_cells == 0:
        return np.zeros((0, 2))
    out = []
    P = np.asarray(lines.points)[:, :2]
    cells = np.asarray(lines.lines)
    i, cos = 0, np.cos(np.radians(angle))
    while i < len(cells):
        n = cells[i]
        ids = cells[i + 1:i + 1 + n]
        i += n + 1
        Q = P[ids]
        closed = len(Q) > 2 and np.allclose(Q[0], Q[-1])
        if not closed:
            out += [Q[0], Q[-1]]
        for k in range(1, len(Q) - 1):
            a, b = Q[k] - Q[k - 1], Q[k + 1] - Q[k]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0 and a @ b < cos * na * nb:
                out.append(Q[k])
    return np.array(out) if out else np.zeros((0, 2))


def show_geometry_2d(plotter, shape: dict, pieces: list, highlighted=(), domain: bool = False,
                     title: str = "", reset: bool = True, reference=None) -> None:
    """Exact preview of a 2D geometry: the domain, the interfaces between its
    regions, its boundary pieces, some of them highlighted (a boundary
    selection); domain=True highlights the domain, a function the part where
    it is negative (geometry2d.domain_level). reference: lines drawn under
    it (the section of the rest of a 3D geometry, for a profile)."""
    from .. import geometry2d as G
    clear(plotter)
    if reference is not None and reference.n_points:
        plotter.add_mesh(reference, color=REFERENCE, line_width=2, lighting=False, name="reference",
                         pickable=False)
    if G.is_empty(shape):               # nothing yet: a frame to draw in
        if reference is None or not reference.n_points:
            plotter.add_mesh(_polyline(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float)),
                             color="#c8ced4", line_width=1, lighting=False, name="frame")
        plotter.add_text("Empty geometry: draw shapes, or add them in the tree", position="upper_left",
                         font_size=10, color="#444444", name="title")
        if reset:
            plotter.view_xy()
            plotter.enable_parallel_projection()
            set_style(plotter, True)
            plotter.reset_camera()
        return
    whole = domain is True
    plotter.add_mesh(_fill(shape), color=HIGHLIGHT if whole else FILL, opacity=0.55 if whole else 1.0,
                     lighting=False, name="fill")
    if callable(domain):
        part = _fill(shape, level=domain)
        if part.n_cells:
            plotter.add_mesh(part, color=HIGHLIGHT, opacity=0.55, lighting=False, name="part")
    for i, p in enumerate(G.interfaces(shape)):
        plotter.add_mesh(_polyline(p.points()), color=INTERFACE, line_width=1.5, lighting=False,
                         name=f"interface{i}")
    for i, p in enumerate(pieces):
        plotter.add_mesh(_polyline(p.points()), color=OUTLINE, line_width=2, lighting=False,
                         name=f"piece{i}")
    outline = G.pieces(shape, level=domain) if callable(domain) else []   # when made of sides
    for i, p in enumerate(list(highlighted) + outline):
        plotter.add_mesh(_polyline(p.points()), color=HIGHLIGHT, line_width=5, lighting=False,
                         name=f"selected{i}")
    if title:
        plotter.add_text(title, position="upper_left", font_size=10, color="#444444", name="title")
    if reset:
        plotter.view_xy()
        plotter.enable_parallel_projection()
        set_style(plotter, True)
        plotter.reset_camera()


_surfaces: dict[str, pv.PolyData] = {}


def geometry_surface_3d(shape: dict, n: int = 90) -> pv.PolyData:
    """The boundary of a 3D CSG shape: the zero level set of its distance
    function on a grid of about n points along the longest side, with
    outward normals (cell data "outward"). The last ones are cached."""
    key = repr((shape, n))
    if key not in _surfaces:
        if len(_surfaces) > 8:
            _surfaces.clear()
        _surfaces[key] = _surface_3d(shape, n)
    return _surfaces[key]


def _tagged(surf: pv.PolyData, face: int) -> pv.PolyData:
    surf.cell_data["face"] = np.full(surf.n_cells, face, dtype=np.int32)
    return surf


def _rebuilt(surf: pv.PolyData, tri: np.ndarray, keep=None) -> pv.PolyData:
    """A triangulated surface with new triangles (n x 4), its face tags kept."""
    face = np.asarray(surf.cell_data["face"]) if "face" in surf.cell_data else None
    if keep is not None:
        tri = tri[keep]
        face = face[keep] if face is not None else None
    out = pv.PolyData(np.asarray(surf.points), tri.ravel())
    if face is not None:
        out.cell_data["face"] = face
    return out


def _primitive_surface(prim: dict, h: float) -> pv.PolyData:
    """Triangulated surface of a primitive, outward normals, triangles of
    size about h (fine enough to clip it along the other shapes); the cell
    data "face" is the index of the face in geometry3d.primitive_faces."""
    t = prim["type"]
    if "_frame" in prim and t not in ("extrusion", "revolution"):     # moved (transforms): built, then placed
        b, A = (np.asarray(x, float) for x in prim["_frame"])
        surf = _primitive_surface({k: v for k, v in prim.items() if k != "_frame"}, h)
        surf.points = b + np.asarray(surf.points) @ A.T
        return surf
    if t == "rectangle":
        lo, hi = np.asarray(prim["min"], float), np.asarray(prim["max"], float)
        L = hi - lo
        faces = []
        for axis in range(3):
            i, j = [a for a in range(3) if a != axis]
            u = np.linspace(lo[i], hi[i], max(1, int(np.ceil(L[i] / h))) + 1)
            v = np.linspace(lo[j], hi[j], max(1, int(np.ceil(L[j] / h))) + 1)
            U, V = np.meshgrid(u, v, indexing="ij")
            for value in (lo[axis], hi[axis]):
                X = [None, None, None]
                X[i], X[j], X[axis] = U, V, np.full_like(U, value)
                grid = pv.StructuredGrid(X[0][:, :, None], X[1][:, :, None], X[2][:, :, None])
                faces.append(_tagged(grid.extract_surface(), len(faces)))     # xmin, xmax, ymin...
        surf = faces[0].merge(faces[1:])
    elif t == "ball":
        r = float(prim["radius"])
        nt = max(48, int(np.ceil(2 * np.pi * r / h)))
        surf = _tagged(pv.Sphere(radius=r, center=prim["center"], theta_resolution=nt,
                                 phi_resolution=max(24, nt // 2)), 0)
    elif t == "cylinder":
        o = np.asarray(prim["origin"], float)
        a = np.asarray(prim["axis"], float)
        a = a / (np.linalg.norm(a) or 1.0)
        L, r = float(prim["length"]), float(prim["radius"])
        nt = max(48, int(np.ceil(2 * np.pi * r / h)))
        side = pv.CylinderStructured(radius=r, height=L, center=o + L / 2 * a, direction=a,
                                     theta_resolution=nt, z_resolution=max(2, int(np.ceil(L / h)))).extract_surface()
        nr = max(2, int(np.ceil(r / h)))
        caps = [_tagged(pv.Disc(center=o, inner=0.0, outer=r, normal=-a, r_res=nr, c_res=nt), 1),
                _tagged(pv.Disc(center=o + L * a, inner=0.0, outer=r, normal=a, r_res=nr, c_res=nt), 2)]
        surf = _tagged(side, 0).merge(caps)
    elif t in ("extrusion", "revolution"):
        surf = _swept_surface(prim, h)
    else:
        raise ValueError(f"no 3D surface for {t!r}")
    surf = surf.triangulate().clean()
    # outward normals, whatever the orientation pyvista gives
    from .. import geometry3d as G3
    surf = surf.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=False,
                                consistent_normals=True, split_vertices=False)
    C = np.asarray(surf.cell_centers().points)
    N = np.asarray(surf.cell_data["Normals"])
    eps = 1e-3 * h
    wrong = G3.sdf(prim, C + eps * N) < G3.sdf(prim, C - eps * N)
    if wrong.any():
        tri = np.array(surf.faces).reshape(-1, 4)
        tri[wrong] = tri[wrong][:, [0, 1, 3, 2]]
        surf = _rebuilt(surf, tri)
    surf.cell_data.pop("Normals", None)
    return surf


def _profile_face(profile: dict, curves, h: float) -> pv.PolyData:
    """The 2D domain of a profile (holes included) in the plane z = 0,
    triangles of size about h: Delaunay triangulation of the points of its
    boundary (curves) and of a grid of interior points far enough from the
    boundary for the sides to be edges; the triangles outside are removed."""
    from .. import geometry2d as G2
    B = np.unique(np.round(np.concatenate(curves), 12), axis=0)
    lo, hi = B.min(axis=0), B.max(axis=0)
    gx = np.arange(lo[0] + h / 2, hi[0], h)
    gy = np.arange(lo[1] + h / 2, hi[1], h)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    Q = np.column_stack([X.ravel(), Y.ravel()])
    if len(Q):
        Q = Q[G2.sdf(profile, Q) < -0.6 * h]
    pts = np.concatenate([B, Q])
    cloud = pv.PolyData(np.column_stack([pts, np.zeros(len(pts))]))
    face = cloud.delaunay_2d()
    if face.n_cells == 0:
        return face
    C = np.asarray(face.cell_centers().points)[:, :2]
    inside = np.flatnonzero(G2.sdf(profile, C) < 0)
    return face.extract_cells(inside).extract_surface().triangulate()


def _swept_surface(prim: dict, h: float) -> pv.PolyData:
    """Surface of an extrusion (along z: sides and two caps) or of a
    revolution (about z: the boundary of the profile turned around)."""
    from .. import geometry3d as G3
    tagged = G3.profile_curves(prim, h, refs=True)
    if not tagged:
        raise ValueError("empty profile")
    curves = [c for c, _ in tagged]
    index = {G3.face_key(dict(f, shape=[])): i for i, f in enumerate(G3.primitive_faces(prim))}
    tags = [index[G3.face_key({"shape": [], "face": "side", "profile": ref})] for _, ref in tagged]
    parts = []
    if prim["type"] == "extrusion":
        z0, z1 = sorted(float(v) for v in prim["z"])
        z = np.linspace(z0, z1, max(1, int(np.ceil((z1 - z0) / h))) + 1)
        for c, tag in zip(curves, tags):
            X = np.repeat(c[:, 0:1], len(z), axis=1)
            Y = np.repeat(c[:, 1:2], len(z), axis=1)
            Z = np.repeat(z[None, :], len(c), axis=0)
            parts.append(_tagged(pv.StructuredGrid(X[:, :, None], Y[:, :, None], Z[:, :, None]).extract_surface(),
                                 tag))
        face = _profile_face(G3.effective_profile(prim), curves, h)
        for tag, value in enumerate((z0, z1)):                 # bottom, top
            cap = face.copy()
            cap.points[:, 2] = value
            parts.append(_tagged(cap, tag))
    else:
        R = max(float(np.max(c[:, 0])) for c in curves)
        nt = max(48, int(np.ceil(2 * np.pi * R / h)))
        th = np.linspace(0.0, 2 * np.pi, nt + 1)
        for c, tag in zip(curves, tags):
            r = np.maximum(c[:, 0], 0.0)[:, None]
            X, Y = r * np.cos(th)[None, :], r * np.sin(th)[None, :]
            Z = np.repeat(c[:, 1:2], len(th), axis=1)
            parts.append(_tagged(pv.StructuredGrid(X[:, :, None], Y[:, :, None], Z[:, :, None]).extract_surface(),
                                 tag))
    surf = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
    surf = surf.triangulate().clean()
    # degenerate triangles (on the axis of a revolution) removed
    if surf.n_cells:
        tri = np.array(surf.faces).reshape(-1, 4)
        good = (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 3]) & (tri[:, 1] != tri[:, 3])
        surf = _rebuilt(surf, tri, good)
    if G3.frame(prim) is not None:        # built in its work plane, then placed
        surf.points = G3.to_global(prim, np.asarray(surf.points))
    return surf


def _surface_3d(shape: dict, n: int) -> pv.PolyData:
    """The surfaces of the primitives, each clipped to its part on the
    boundary of the shape; outward normals."""
    from .. import geometry3d as G3
    lo, hi = G3.bounds(shape)
    size = float(np.linalg.norm(hi - lo)) or 1.0
    h = size / n
    pieces, refs = [], []
    for path, prim in G3.primitives(shape):
        try:
            surf = _primitive_surface(prim, h)
        except (KeyError, TypeError, ValueError):
            return _surface_3d_grid(shape, n)
        # the faces of all the primitives numbered together (cell data "face_id")
        surf.cell_data["face_id"] = np.asarray(surf.cell_data["face"]) + len(refs)
        refs += [dict(f, shape=list(prim.get("_src", path))) for f in G3.primitive_faces(prim)]
        surf.point_data["c"] = G3.boundary_level(shape, path, np.asarray(surf.points), 10 * size)
        part = surf.clip_scalar(scalars="c", value=0.0, invert=True)
        if part.n_cells == 0:
            continue
        part = part.extract_surface().triangulate()
        if G3.orientation(shape, path) < 0:          # a hole: its normals point inwards
            tri = np.array(part.faces).reshape(-1, 4)[:, [0, 1, 3, 2]]
            ids = np.asarray(part.cell_data["face_id"])
            part = pv.PolyData(np.asarray(part.points), tri.ravel())
            part.cell_data["face_id"] = ids
        part.cell_data.pop("face", None)
        pieces.append(part)
    if not pieces:
        return pv.PolyData()
    surf = pieces[0].merge(pieces[1:]) if len(pieces) > 1 else pieces[0]
    surf = surf.clean(tolerance=1e-9 * size, absolute=False)
    surf = surf.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=False,
                                consistent_normals=False, split_vertices=False)
    # outward normals kept under another name: a "Normals" array of the cells
    # makes VTK shade every face alike
    surf.cell_data["outward"] = surf.cell_data.pop("Normals")
    # the edges: between faces at an angle, and the seams between the pieces
    # (intersections of the primitives: the boundaries of the clipped parts)
    surf.field_data["edges_kind"] = ["exact"]
    surf.field_data["face_refs"] = [json.dumps(refs)]
    return surf


def face_refs(surf) -> list[dict] | None:
    """The faces (geometry3d references) of the preview surface of a 3D
    shape, indexed by its cell data "face_id" (None for a grid surface)."""
    if "face_refs" not in surf.field_data or "face_id" not in surf.cell_data:
        return None
    return json.loads(str(surf.field_data["face_refs"][0]))


def face_plane(surf, fid: int, shape: dict):
    """The plane of a face of a preview surface (its visible part): (center,
    outward unit normal), or None if the face is not plane."""
    from .. import geometry3d as G3
    ids = np.flatnonzero(np.asarray(surf.cell_data["face_id"]) == fid)
    if not len(ids):
        return None
    tri = np.array(surf.faces).reshape(-1, 4)[ids, 1:]
    X = np.asarray(surf.points)[tri]                       # cells x 3 x 3
    cross = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    if area.sum() <= 0:
        return None
    C = (X.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()
    n = np.linalg.svd(X.reshape(-1, 3) - C)[2][-1]
    lo, hi = G3.bounds(shape)
    size = float(np.linalg.norm(hi - lo)) or 1.0
    if np.max(np.abs((X.reshape(-1, 3) - C) @ n)) > 1e-6 * size:
        return None
    eps = 1e-3 * size
    if G3.sdf(shape, (C + eps * n)[None])[0] < G3.sdf(shape, (C - eps * n)[None])[0]:
        n = -n
    return C, n


def cell_faces(surf) -> list | None:
    """The face (geometry3d.face_key) of each cell of a preview surface."""
    from .. import geometry3d as G3
    refs = face_refs(surf)
    if refs is None:
        return None
    keys = [G3.face_key(r) for r in refs]
    return [keys[i] for i in np.asarray(surf.cell_data["face_id"])]


def _surface_3d_grid(shape: dict, n: int) -> pv.PolyData:
    """Fallback: the zero level set on a grid (marching cubes)."""
    from .. import geometry3d as G3
    lo, hi = G3.bounds(shape)
    margin = 0.03 * (hi - lo).max()
    lo, hi = lo - margin, hi + margin
    h = (hi - lo).max() / n
    dims = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    grid = pv.ImageData(dimensions=tuple(int(d) for d in dims), spacing=(h, h, h), origin=tuple(lo))
    grid.point_data["d"] = G3.sdf(shape, np.asarray(grid.points))
    surf = grid.contour(isosurfaces=[0.0], scalars="d").triangulate().clean()
    if surf.n_cells == 0:
        return surf
    surf = surf.compute_normals(cell_normals=True, point_normals=True, auto_orient_normals=False,
                                consistent_normals=True, split_vertices=False)
    # outward: along the gradient of the distance function
    C = np.asarray(surf.cell_centers().points)
    N = np.asarray(surf.cell_data["Normals"])
    eps = 0.5 * h
    if np.mean(G3.sdf(shape, C + eps * N) - G3.sdf(shape, C - eps * N)) < 0:
        N = -N
        surf.point_data["Normals"] = -np.asarray(surf.point_data["Normals"])
    surf.cell_data.pop("Normals")
    surf.cell_data["outward"] = N
    return surf


def show_geometry_3d(plotter, shape: dict, highlighted=None, part: dict | None = None,
                     title: str = "", reset: bool = True) -> None:
    """Preview of a 3D geometry: its surface; highlighted: a boolean per face
    (a boundary selection) or True (the whole domain); part: a primitive
    drawn as a translucent overlay (the shape selected in the tree)."""
    clear(plotter)
    try:                                    # lights from several directions: faces shaded apart
        plotter.remove_all_lights()
        plotter.enable_lightkit()
    except (AttributeError, TypeError):
        pass
    surf = geometry_surface_3d(shape)
    if surf.n_cells == 0:
        message(plotter, "empty geometry")
        return
    exact = "edges_kind" in surf.field_data
    behind_lines(plotter.add_mesh(surf, color=HIGHLIGHT if highlighted is True else SURFACE,
                                  smooth_shading=not exact, opacity=0.55 if highlighted is True else 1.0,
                                  ambient=0.3, diffuse=0.7, specular=0.1, name="surface"))
    edges = surf.extract_feature_edges(feature_angle=30, boundary_edges=exact, manifold_edges=False,
                                       non_manifold_edges=False)
    if edges.n_points:
        plotter.add_mesh(edges, color=OUTLINE, line_width=2, lighting=False, name="feature_edges")
    if highlighted is not None and highlighted is not True and np.any(highlighted):
        sel = surf.extract_cells(np.flatnonzero(highlighted))
        plotter.add_mesh(sel, color=HIGHLIGHT, smooth_shading=True, name="selected")
    if part is not None:
        try:
            p = geometry_surface_3d(part, 60)
            if p.n_cells:
                plotter.add_mesh(p, color=HOVER, opacity=0.45, smooth_shading=True, name="part")
        except (KeyError, TypeError, ValueError):
            pass
    if title:
        plotter.add_text(title, position="upper_left", font_size=10, color="#444444", name="title")
    if reset:
        plotter.disable_parallel_projection()
        plotter.view_isometric()
        plotter.camera.azimuth = 20         # not exactly isometric: the faces are shaded differently
        plotter.camera.elevation = 10
        set_style(plotter, False)
        plotter.reset_camera()


def manifold_preview(src: dict, values: dict, n: int = 120) -> pv.PolyData:
    """The preview of a curve (lines) or of a surface in 3D (triangles or
    quadrangles) of a resolved source."""
    from .. import geometry3d as G3, manifold
    t = src["type"]
    if t == "curve":
        X = manifold.chain_polyline(src)
        X = np.column_stack([X, np.zeros((len(X), 3 - X.shape[1]))]) if X.shape[1] < 3 else X
        return pv.lines_from_points(X)
    if t == "surface":
        prim = {"type": "extrusion", "profile": src["shape"], "z": [0.0, 1.0]}
        from .. import geometry2d as G2
        h = G2.size(src["shape"]) / n
        curves = G3.profile_curves(prim, h)
        face = _profile_face(src["shape"], curves, h)
        P = manifold.map_surface(np.asarray(face.points)[:, :2].T, src, values).T
        return pv.PolyData(P, np.asarray(face.faces))
    # a swept surface: the polyline of its curve times the sweep
    X = manifold.chain_polyline(src)
    m = max(24, int(np.ceil(n * manifold.sweep_length(src) / max(float(np.ptp(X, axis=0).max()), 1e-300))))
    m = min(m, 4 * n)
    t_ = np.linspace(0.0, 1.0, m + 1)
    A = np.repeat(X[:, 0:1], len(t_), axis=1)
    B = np.repeat(X[:, 1:2], len(t_), axis=1)
    T = np.repeat(t_[None, :], len(X), axis=0)
    Q = manifold.map_swept(np.vstack([A.ravel(), B.ravel(), T.ravel()]), src)
    grid = pv.StructuredGrid(*(Q[k].reshape(A.shape)[:, :, None] for k in range(3)))
    return grid.extract_surface()


def show_manifold(plotter, src: dict, values: dict, title: str = "", reset: bool = True) -> None:
    """Preview of a curve or of a surface in 3D (its flat domain for the
    selections, as a 2D geometry)."""
    clear(plotter)
    surf = manifold_preview(src, values)
    if src["type"] == "curve":
        plotter.add_mesh(surf, color=OUTLINE, line_width=4, lighting=False, name="surface")
        from .. import manifold
        ends = np.asarray([list(p) + [0.0] * (3 - len(p)) for p in manifold.chain_ends(src)])
        if len(ends):
            plotter.add_mesh(pv.PolyData(ends), color=HIGHLIGHT, point_size=12, style="points",
                             render_points_as_spheres=True, lighting=False, name="ends")
    else:
        try:
            plotter.remove_all_lights()
            plotter.enable_lightkit()
        except (AttributeError, TypeError):
            pass
        behind_lines(plotter.add_mesh(surf, color=SURFACE, smooth_shading=True, ambient=0.3, diffuse=0.7,
                                      specular=0.1, name="surface"))
        edges = surf.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                           non_manifold_edges=False)
        if edges.n_points:
            plotter.add_mesh(edges, color=OUTLINE, line_width=2, lighting=False, name="feature_edges")
    if title:
        plotter.add_text(title, position="upper_left", font_size=10, color="#444444", name="title")
    if reset:
        flat = src["type"] == "curve" and src.get("dim") == 2
        if flat:
            plotter.enable_parallel_projection()
            plotter.view_xy()
        else:
            plotter.disable_parallel_projection()
            plotter.view_isometric()
            plotter.camera.azimuth = 20
            plotter.camera.elevation = 10
        set_style(plotter, flat)
        plotter.reset_camera()


def show_face_hover(plotter, surf, face_id) -> None:
    """Highlight a face of a 3D preview surface (None: remove)."""
    plotter.remove_actor("hover")
    if surf is None or face_id is None:
        return
    cells = np.flatnonzero(np.asarray(surf.cell_data["face_id"]) == face_id)
    if len(cells):
        # opaque, as the selected faces: a translucent face over the surface may be hidden by it
        plotter.add_mesh(surf.extract_cells(cells), color=HOVER, smooth_shading=True,
                         name="hover", pickable=False)


def show_region_hover(plotter, shape: dict, region) -> None:
    """Highlight the region under the mouse (None: remove)."""
    from .. import geometry2d as G
    plotter.remove_actor("hover")
    if region is not None:
        part = _fill(shape, level=G.region_level(shape, [region]))
        if part.n_cells:
            plotter.add_mesh(part, color=HOVER, opacity=0.45, lighting=False, name="hover")


def show_hover(plotter, piece) -> None:
    """Highlight the piece under the mouse (None: remove)."""
    plotter.remove_actor("hover")
    if piece is not None:
        plotter.add_mesh(_polyline(piece.points()), color=HOVER, line_width=6, lighting=False,
                         name="hover")


def _display_point(ren, x: float, y: float, depth: float) -> np.ndarray:
    ren.SetDisplayPoint(float(x), float(y), depth)
    ren.DisplayToWorld()
    w = np.asarray(ren.GetWorldPoint(), float)
    return w[:3] / (w[3] if w[3] else 1.0)


def display_to_world(plotter, x: float, y: float) -> np.ndarray:
    """World point of the z = 0 plane under display pixel (x, y): where the
    ray of the pixel meets that plane (a 2D view, or a work plane seen in 3D;
    the point of the focal plane when the ray is along the plane)."""
    ren = plotter.renderer
    a, b = _display_point(ren, x, y, 0.0), _display_point(ren, x, y, 1.0)
    d = b - a
    if abs(d[2]) <= 1e-9 * (np.linalg.norm(d) or 1.0):
        return a
    t = -a[2] / d[2]
    return a + t * d


def pixel_size(plotter) -> float:
    """World length of one pixel in the z = 0 plane, at the center of the view."""
    w, h = plotter.renderer.GetSize()
    cx, cy = max(w, 2) / 2, max(h, 2) / 2
    a, b = display_to_world(plotter, cx, cy), display_to_world(plotter, cx + 1, cy)
    return float(np.linalg.norm(b - a)) or 1e-9


PROFILE_3D = "#e8a060"


def show_context(plotter, surface) -> None:
    """The 3D geometry around a work plane (in the coordinates of the plane,
    z along its normal), translucent, when the profile is drawn in 3D."""
    plotter.remove_actor("context")
    if surface is not None and surface.n_points:
        plotter.add_mesh(surface, color="#9aa7b2", opacity=0.25, smooth_shading=False, name="context",
                         pickable=False)
        fill = plotter.actors.get("fill")
        if fill is not None:              # the profile stands out of the geometry around it
            fill.prop.color = PROFILE_3D
            fill.prop.opacity = 0.85


def view_plane_3d(plotter) -> None:
    """An oblique view of a work plane (the z = 0 plane), free rotation."""
    plotter.view_isometric()
    plotter.disable_parallel_projection()
    set_style(plotter, False)
    plotter.reset_camera()


def show_rubber(plotter, shape: dict | None) -> None:
    """Outline of the shape being drawn (None: remove)."""
    plotter.remove_actor("rubber")
    if shape is None:
        return
    if shape["type"] == "segment":
        pts = np.array([shape["a"][:2], shape["b"][:2]], float)
    elif shape["type"] == "rectangle":
        (x0, y0), (x1, y1) = shape["min"], shape["max"]
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    elif shape["type"] == "ellipse":
        from .. import geometry2d as G
        pts = next(s for s in G.sides(shape)).points(np.linspace(0, 1, 129))
    elif shape["type"] == "polygon" and (shape.get("splines") or shape.get("fillets")):
        from .. import geometry2d as G         # with its curved sides and its arcs
        try:
            pts = G.boundary_polyline(shape, per_span=24, per_arc=16)
        except (TypeError, ValueError, KeyError, IndexError, np.linalg.LinAlgError):
            pts = np.array([p[:2] for p in shape["points"]], float)
        pts = np.vstack([pts, pts[:1]])
    elif shape["type"] in ("polygon", "polyline"):
        pts = np.array([p[:2] for p in shape["points"]], float)
        if shape["type"] == "polygon":
            pts = np.vstack([pts, pts[:1]])
        if len(pts) == 1:
            pts = np.vstack([pts, pts])
    else:
        c, r = np.asarray(shape["center"]), shape["radius"]
        th = np.linspace(0, 2 * np.pi, 97)
        pts = c + r * np.column_stack([np.cos(th), np.sin(th)])
    plotter.add_mesh(_polyline(pts), color=HOVER, line_width=3, lighting=False, name="rubber")


def show_grid(plotter, lo, hi, step: float) -> None:
    """Light grid of the snapping step over [lo, hi] (None step: remove)."""
    plotter.remove_actor("grid")
    if step is None:
        return
    xs = np.arange(np.floor(lo[0] / step), np.ceil(hi[0] / step) + 1) * step
    ys = np.arange(np.floor(lo[1] / step), np.ceil(hi[1] / step) + 1) * step
    if len(xs) * len(ys) > 40000:          # too fine to be useful on screen
        return
    lines = []
    for x in xs:
        lines.append([[x, ys[0], 0.0], [x, ys[-1], 0.0]])
    for y in ys:
        lines.append([[xs[0], y, 0.0], [xs[-1], y, 0.0]])
    pts = np.array(lines, float).reshape(-1, 3)
    cells = np.hstack([[2, 2 * i, 2 * i + 1] for i in range(len(lines))])
    grid = pv.PolyData(pts, lines=cells)
    # opaque: a translucent grid is hidden behind the (opaque) domain, in its
    # plane, by some OpenGL drivers
    plotter.add_mesh(grid, color="#aebfcc", line_width=1, lighting=False, name="grid", pickable=False)


def show_axes_2d(plotter, lo, hi, major: float | None) -> None:
    """The x and y axes (through the origin, when in view) and the coordinates
    of the major lines along the lower and left edges of [lo, hi] (None: remove)."""
    for name in ("axes", "axes_labels"):
        plotter.remove_actor(name)
    if major is None:
        return
    lines = []
    if lo[1] <= 0 <= hi[1]:
        lines.append([[lo[0], 0.0, 0.0], [hi[0], 0.0, 0.0]])
    if lo[0] <= 0 <= hi[0]:
        lines.append([[0.0, lo[1], 0.0], [0.0, hi[1], 0.0]])
    if lines:
        pts = np.array(lines, float).reshape(-1, 3)
        cells = np.hstack([[2, 2 * i, 2 * i + 1] for i in range(len(lines))])
        plotter.add_mesh(pv.PolyData(pts, lines=cells), color="#5b6b78", line_width=1.5, lighting=False,
                         name="axes", pickable=False)
    dx, dy = hi[0] - lo[0], hi[1] - lo[1]
    xs = np.arange(np.ceil(lo[0] / major), np.floor(hi[0] / major) + 1) * major
    ys = np.arange(np.ceil(lo[1] / major), np.floor(hi[1] / major) + 1) * major
    pts, labels = [], []
    for x in xs:
        if x > lo[0] + 0.04 * dx:
            pts.append([x, lo[1] + 0.015 * dy, 0.0])
            labels.append(f"{round(x, 12) + 0.0:g}")
    for y in ys:
        if y > lo[1] + 0.05 * dy:
            pts.append([lo[0] + 0.01 * dx, y, 0.0])
            labels.append(f"{round(y, 12) + 0.0:g}")
    if pts:
        plotter.add_point_labels(np.array(pts), labels, font_size=10, text_color="#33414f", shape=None,
                                 show_points=False, always_visible=True, name="axes_labels", pickable=False)


def show_marks(plotter, marks) -> None:
    """The constraints of a polygon: their symbols by its sides and vertices
    [(position, text)] (empty: remove)."""
    plotter.remove_actor("marks")
    if not marks:
        return
    pts = np.array([[p[0], p[1], 0.0] for p, _ in marks], float)
    plotter.add_point_labels(pts, [t for _, t in marks], font_size=12, text_color="#8a3b12", shape="rounded_rect",
                             shape_color="#fff3e8", shape_opacity=0.7, point_size=1, show_points=False,
                             always_visible=True, name="marks")


def move_active_handle(plotter, point) -> None:
    """The grabbed handle drawn where the mouse is (while it is dragged)."""
    plotter.add_mesh(pv.PolyData(np.array([[float(point[0]), float(point[1]), 0.0]])), color=HOVER,
                     point_size=16, style="points", lighting=False, name="handle_active")


def show_handles(plotter, positions, active: int | None = None, minor=None) -> None:
    """Handles of the selected primitive (empty: remove); minor: True for the
    secondary handles (middles of the sides of a polygon), drawn smaller."""
    plotter.remove_actor("handles")
    plotter.remove_actor("handles_minor")
    plotter.remove_actor("handle_active")
    if not len(positions):
        return
    pts = np.column_stack([np.asarray(positions, float), np.zeros(len(positions))])
    small = np.zeros(len(pts), bool) if minor is None else np.asarray(minor, bool)
    if (~small).any():
        plotter.add_mesh(pv.PolyData(pts[~small]), color="#1f6f78", point_size=12, style="points",
                         lighting=False, name="handles")
    if small.any():
        plotter.add_mesh(pv.PolyData(pts[small]), color="#6fa8ad", point_size=8, style="points",
                         lighting=False, name="handles_minor")
    if active is not None:
        plotter.add_mesh(pv.PolyData(pts[active:active + 1]), color=HOVER, point_size=16,
                         style="points", lighting=False, name="handle_active")
