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


"""3D scenes drawn with PyVista: mesh, highlighted selection, result field.

These functions only need a PyVista plotter, so that they can be used (and
tested) off screen as well as in the Qt window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

SURFACE = "#dfe3e8"
EDGES = "#5b6472"
HIGHLIGHT = "#e4572e"
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


def _wire(ds: pv.DataSet) -> pv.DataSet:
    """Edges of the elements (works for linear and quadratic elements)."""
    return ds.extract_all_edges() if ds.n_cells else ds


def _outline(ds: pv.DataSet) -> pv.DataSet:
    """Contour of the domain: boundary edges in 2D, feature edges in 3D."""
    surf = ds.extract_surface(nonlinear_subdivision=0)
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


def show_mesh(plotter, mesh: pv.DataSet, selection: pv.DataSet | None = None,
              title: str = "", reset: bool = True, edges: bool = True) -> None:
    """The mesh, and optionally a selection (faces or elements) highlighted."""
    clear(plotter)
    planar = is_planar(mesh)
    faded = selection is not None
    plotter.add_mesh(mesh, color=FADED if faded else SURFACE, opacity=1.0 if planar or not faded else 0.35,
                     show_edges=False, lighting=not planar, name="mesh")
    if edges:
        plotter.add_mesh(_wire(mesh), color=EDGES, opacity=0.35 if faded else 1.0,
                         line_width=1, lighting=False, name="edges")
    if selection is not None and selection.n_points:
        dim = selection.get_cell(0).dimension if selection.n_cells else 0
        if dim == 1:   # boundary of a 2D domain
            plotter.add_mesh(selection, color=HIGHLIGHT, line_width=5, lighting=False,
                             render_lines_as_tubes=False, name="selection")
        else:
            plotter.add_mesh(selection, color=HIGHLIGHT, show_edges=False, lighting=not planar,
                             name="selection")
            if edges:
                plotter.add_mesh(_wire(selection), color="#7a2a14", line_width=1,
                                 lighting=False, name="selection_edges")
    if title:
        plotter.add_text(title, position="upper_left", font_size=10, color="#444444", name="title")
    _camera(plotter, mesh, reset)


def deformation_scale(ds: pv.DataSet, name: str) -> float:
    """Factor such that the largest displacement is 10% of the model size."""
    u = np.asarray(ds.point_data[name])
    umax = float(np.max(np.linalg.norm(u.reshape(len(u), -1), axis=1))) if len(u) else 0.0
    return 0.1 * size(ds) / umax if umax > 0 else 1.0


def show_field(plotter, ds: pv.DataSet, name: str, component: int | None = None,
               warp: str | None = None, scale: float = 1.0, edges: bool = False,
               title: str = "", reset: bool = True) -> None:
    """A result field, possibly on the shape deformed by the field `warp`."""
    clear(plotter)
    shown = ds
    if warp and warp in ds.point_data:
        vec = np.asarray(ds.point_data[warp])
        if vec.ndim == 2 and vec.shape[1] < 3:
            vec = np.hstack([vec, np.zeros((len(vec), 3 - vec.shape[1]))])
        shown = ds.copy()
        shown.points = ds.points + scale * vec
        plotter.add_mesh(_outline(ds), color="#7d8690", line_width=1, lighting=False,
                         name="undeformed")
    ncomp = field_names(ds).get(name, 1)
    label = name if component is None or ncomp == 1 else f"{name} ({'xyz'[component]})"
    if ncomp > 1 and component is None:
        label = f"|{name}|"
    plotter.add_mesh(shown, scalars=name, component=component if ncomp > 1 else None,
                     cmap=CMAP, show_edges=edges, edge_color=EDGES, lighting=not is_planar(ds),
                     scalar_bar_args={"title": label, "vertical": True, "position_x": 0.86,
                                      "position_y": 0.1, "height": 0.65, "width": 0.05,
                                      "fmt": "%.3g", "color": "#333333",
                                      "title_font_size": 14, "label_font_size": 12},
                     name="field")
    if title:
        plotter.add_text(title, position="upper_left", font_size=10, color="#444444", name="title")
    _camera(plotter, ds, reset)


# ------------------------------------------------------------------ 2D geometry preview

HOVER = "#f2a65a"
OUTLINE = "#33414f"
FILL = "#cfe0ea"


def _polyline(points: np.ndarray) -> pv.PolyData:
    pts = np.column_stack([points, np.zeros(len(points))])
    return pv.lines_from_points(pts)


def _fill(shape: dict, n: int = 240) -> pv.DataSet:
    """The domain, as the part of a fine grid where the distance is negative
    (the boundary itself is drawn exactly, on top)."""
    from .. import geometry2d as G
    lo, hi = G.bounds(shape)
    margin = 0.02 * (hi - lo).max()
    lo, hi = lo - margin, hi + margin
    h = (hi - lo).max() / n
    dims = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    grid = pv.ImageData(dimensions=(int(dims[0]), int(dims[1]), 1), spacing=(h, h, 1.0),
                        origin=(lo[0], lo[1], 0.0))
    grid.point_data["d"] = G.sdf(shape, grid.points[:, :2])
    return grid.clip_scalar(scalars="d", value=0.0, invert=True)


def show_geometry_2d(plotter, shape: dict, pieces: list, highlighted=(), domain: bool = False,
                     title: str = "", reset: bool = True) -> None:
    """Exact preview of a 2D geometry: the domain, its boundary pieces, some of
    them highlighted (a boundary selection); domain=True highlights the domain."""
    clear(plotter)
    plotter.add_mesh(_fill(shape), color=HIGHLIGHT if domain else FILL, opacity=0.55 if domain else 1.0,
                     lighting=False, name="fill")
    for i, p in enumerate(pieces):
        plotter.add_mesh(_polyline(p.points()), color=OUTLINE, line_width=2, lighting=False,
                         name=f"piece{i}")
    for i, p in enumerate(highlighted):
        plotter.add_mesh(_polyline(p.points()), color=HIGHLIGHT, line_width=5, lighting=False,
                         name=f"selected{i}")
    if title:
        plotter.add_text(title, position="upper_left", font_size=10, color="#444444", name="title")
    if reset:
        plotter.view_xy()
        plotter.enable_parallel_projection()
        set_style(plotter, True)
        plotter.reset_camera()


def show_hover(plotter, piece) -> None:
    """Highlight the piece under the mouse (None: remove)."""
    plotter.remove_actor("hover")
    if piece is not None:
        plotter.add_mesh(_polyline(piece.points()), color=HOVER, line_width=6, lighting=False,
                         name="hover")


def display_to_world(plotter, x: float, y: float) -> np.ndarray:
    """World point (in the z = 0 plane for a 2D view) under display pixel (x, y)."""
    ren = plotter.renderer
    ren.SetDisplayPoint(float(x), float(y), 0.0)
    ren.DisplayToWorld()
    w = np.asarray(ren.GetWorldPoint(), float)
    return w[:3] / (w[3] if w[3] else 1.0)


def pixel_size(plotter) -> float:
    """World length of one pixel (parallel projection)."""
    a, b = display_to_world(plotter, 0, 0), display_to_world(plotter, 1, 0)
    return float(np.linalg.norm(b - a)) or 1e-9


def show_rubber(plotter, shape: dict | None) -> None:
    """Outline of the shape being drawn (None: remove)."""
    plotter.remove_actor("rubber")
    if shape is None:
        return
    if shape["type"] == "rectangle":
        (x0, y0), (x1, y1) = shape["min"], shape["max"]
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
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
    plotter.add_mesh(grid, color="#9fb3c2", opacity=0.35, line_width=1, lighting=False, name="grid",
                     pickable=False)
