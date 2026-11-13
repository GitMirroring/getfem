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


"""Exact description of 2D geometries (box and CSG), without meshing.

The boundary of a CSG geometry is made of parts of the sides of its
primitives (the four sides of a rectangle, the circle of a disk, the sides
of a polygon). Each side
is cut where it actually borders the domain: a side shared by two united
rectangles is not on the boundary, a circle in an intersection only partly.
This gives

- an exact preview of the geometry, before any meshing;
- boundary *pieces* that the user can pick with the mouse. A picked piece is
  stored by the side it belongs to, e.g. {"shape": [1], "side": "circle"}
  ("the circle of the second primitive"), not by coordinates: the selection
  then follows the changes of the geometry (a parameter for the radius).

Only numpy is needed. Shapes use the format of model.py (numbers: the
model is resolved by parameters.resolve first).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import transforms as T

RECTANGLE_SIDES = ("xmin", "xmax", "ymin", "ymax")
BOOLEANS = ("union", "intersection", "difference")


# ------------------------------------------------------------------ shapes

def shape_of(source: dict) -> dict | None:
    """The CSG tree of a 2D box or csg source (None for other sources)."""
    t = source.get("type")
    if t == "box" and len(source.get("size", [])) == 2:
        o, s = source["origin"], source["size"]
        return {"type": "rectangle", "min": list(o), "max": [o[0] + s[0], o[1] + s[1]]}
    if (t == "csg" and source.get("dim") == 2) or t == "surface":     # surface: its flat domain
        shape = source.get("shape")
        return T.flatten(shape, 2) if isinstance(shape, dict) else shape
    return None


def primitive_at(shape: dict, path) -> dict | None:
    """The node at path (indices in "of") of the tree of the model; in a
    flattened tree (transforms.flatten), the (first copy of the) primitive
    of that path, with the transformations applied."""
    if shape.get("_flat"):
        return next((p for q, p in primitives(shape) if q == tuple(path)), None)
    for i in path:
        try:
            shape = shape["of"][i]
        except (KeyError, IndexError, TypeError):
            return None
    return shape


def primitives(shape: dict, path: tuple = ()):
    """(path, primitive) of the leaves of a CSG tree; path = indices in "of"
    in the tree of the model. A tree with transforms or arrays is flattened
    first (transforms.flatten): the primitives are then transformed, and a
    primitive copied by an array comes once per copy, with the same path."""
    if not path and not shape.get("_flat") and T.has_transforms(shape):
        shape = T.flatten(shape, 2)
    if shape.get("type") in BOOLEANS:
        for i, s in enumerate(shape.get("of", [])):
            yield from primitives(s, path + (i,))
    elif not incomplete(shape):
        yield T.source_path(shape, path), shape


def source_paths(shape: dict) -> list[tuple]:
    """The paths of the primitives in the tree of the model, each once."""
    out = []
    for q, _ in primitives(shape):
        if q not in out:
            out.append(q)
    return out


def incomplete(shape: dict) -> bool:
    """A polygon being entered (fewer than three vertices): not a shape yet."""
    return shape.get("type") == "polygon" and len(shape.get("points") or []) < 3


def sdf(shape: dict, P: np.ndarray) -> np.ndarray:
    """Signed distance like function (negative inside) at the points P (n x 2)."""
    t = shape["type"]
    if t in T.TRANSFORMS:
        return sdf(T.flatten(shape, 2), P)
    if t == "rectangle":
        lo, hi = np.asarray(shape["min"], float), np.asarray(shape["max"], float)
        return np.max(np.maximum(lo - P, P - hi), axis=1)
    if t == "ball":
        return np.linalg.norm(P - np.asarray(shape["center"], float), axis=1) - float(shape["radius"])
    if t == "polygon":
        if incomplete(shape):
            return np.full(len(P), np.inf)
        if spline_points(shape):
            return curved_polygon_sdf(shape, P)
        rounded = _fillets_of(shape)
        if rounded is not None:
            return rounded_polygon_sdf(rounded[0], rounded[1], P)
        return polygon_sdf(np.asarray(shape["points"], float), P)
    if t == "ellipse":
        return ellipse_sdf(shape["center"], shape["radii"], float(shape.get("angle", 0.0)), P)
    parts = [sdf(s, P) for s in shape["of"]]
    if not parts:                          # an empty boolean: nothing
        return np.full(len(P), np.inf)
    if t == "union":
        return np.min(parts, axis=0)
    if t == "intersection":
        return np.max(parts, axis=0)
    if t == "difference":
        return parts[0] if len(parts) == 1 else np.maximum(parts[0], -np.min(parts[1:], axis=0))
    raise ValueError(f"shape {t!r} is not 2D")


def bounds(shape: dict) -> tuple[np.ndarray, np.ndarray]:
    t = shape["type"]
    if t in T.TRANSFORMS:
        return bounds(T.flatten(shape, 2))
    if t == "rectangle":
        return np.asarray(shape["min"], float), np.asarray(shape["max"], float)
    if t == "ball":
        c, r = np.asarray(shape["center"], float), float(shape["radius"])
        return c - r, c + r
    if t == "polygon":
        if incomplete(shape):
            raise ValueError("a polygon being entered")
        V = np.asarray(shape["points"], float)
        rounded = _fillets_of(shape)
        if rounded is not None:
            V = np.vstack([rounded[0]] + [f.points(65) for f in rounded[1]])
        if spline_points(shape):
            V = np.vstack([V, boundary_polyline(shape, per_span=64)])
        return V.min(axis=0), V.max(axis=0)
    if t == "ellipse":
        c = np.asarray(shape["center"], float)
        (r1, r2), th = (float(r) for r in shape["radii"]), math.radians(float(shape.get("angle", 0.0)))
        e = np.array([math.hypot(r1 * math.cos(th), r2 * math.sin(th)),
                      math.hypot(r1 * math.sin(th), r2 * math.cos(th))])
        return c - e, c + e
    b = []
    for s in shape["of"]:
        try:
            b.append(bounds(s))
        except ValueError:                 # an empty boolean in it
            if t == "difference" and not b:
                raise
    if not b:
        raise ValueError("empty shape")
    if t == "union":
        return np.min([x[0] for x in b], axis=0), np.max([x[1] for x in b], axis=0)
    if t == "intersection":
        return np.max([x[0] for x in b], axis=0), np.min([x[1] for x in b], axis=0)
    return b[0]


def polygon_sdf(V: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Signed euclidean distance to the polygon of vertices V (n x 2) at the
    points P (m x 2), negative inside."""
    A, B = V, np.roll(V, -1, axis=0)
    D = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", D, D), 1e-300)
    W = P[:, None, :] - A[None, :, :]                       # m x n x 2
    t = np.clip(np.einsum("mnk,nk->mn", W, D) / L2, 0.0, 1.0)
    dist = np.min(np.linalg.norm(W - t[:, :, None] * D[None], axis=2), axis=1)
    # crossing number
    y, x = P[:, 1:2], P[:, 0:1]
    ya, yb = A[None, :, 1], B[None, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = A[None, :, 0] + (y - ya) * (B[None, :, 0] - A[None, :, 0]) / (yb - ya)
    cross = ((ya > y) != (yb > y)) & (x < xc)
    inside = np.sum(cross, axis=1) % 2 == 1
    return np.where(inside, -dist, dist)


def ellipse_sdf(center, radii, angle_deg: float, P: np.ndarray) -> np.ndarray:
    """Signed euclidean distance to an ellipse (semi-axes radii, the first
    one at angle_deg degrees from the x axis), negative inside; the nearest
    point is found as in D. Eberly, "Distance from a point to an ellipse"."""
    th = math.radians(angle_deg)
    D = np.asarray(P, float) - np.asarray(center, float)
    a = D[:, 0] * math.cos(th) + D[:, 1] * math.sin(th)
    b = -D[:, 0] * math.sin(th) + D[:, 1] * math.cos(th)
    r1, r2 = float(radii[0]), float(radii[1])
    swap = r1 < r2
    e0, e1 = max(r1, r2), min(r1, r2)
    y0, y1 = (np.abs(b), np.abs(a)) if swap else (np.abs(a), np.abs(b))
    q0, q1 = np.empty_like(y0), np.empty_like(y1)
    # y1 > 0 and y0 > 0: the root of F(s) by bisection
    gen = (y0 > 0) & (y1 > 0)
    z0, z1 = y0[gen] / e0, y1[gen] / e1
    g = z0 ** 2 + z1 ** 2 - 1
    r0 = (e0 / e1) ** 2
    n0 = r0 * z0
    s0, s1 = z1 - 1, np.where(g < 0, 0.0, np.hypot(n0, z1) - 1)
    for _ in range(100):
        s = (s0 + s1) / 2
        gs = (n0 / (s + r0)) ** 2 + (z1 / (s + 1)) ** 2 - 1
        s0, s1 = np.where(gs > 0, s, s0), np.where(gs > 0, s1, s)
    s = (s0 + s1) / 2
    q0[gen], q1[gen] = r0 * y0[gen] / (s + r0), y1[gen] / (s + 1)
    # on the minor axis
    m = (y0 <= 0) & (y1 > 0)
    q0[m], q1[m] = 0.0, e1
    # on the major axis
    m = y1 <= 0
    x = e0 * y0[m] / (e0 * e0 - e1 * e1) if e0 > e1 else np.full(m.sum(), np.inf)
    inner = x < 1
    q0[m] = np.where(inner, e0 * np.minimum(x, 1), e0)
    q1[m] = np.where(inner, e1 * np.sqrt(np.maximum(0.0, 1 - np.minimum(x, 1) ** 2)), 0.0)
    d = np.hypot(y0 - q0, y1 - q1)
    inside = (a / r1) ** 2 + (b / r2) ** 2 < 1
    return np.where(inside, -d, d)


def polygon_area(points) -> float:
    """Signed area (positive counterclockwise)."""
    V = np.asarray(points, float)
    x, y = V[:, 0], V[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_problem(points) -> str | None:
    """Why a list of vertices is not a simple polygon (None if it is)."""
    V = np.asarray(points, float)
    n = len(V)
    if n < 3:
        return "at least three vertices expected"
    scale = float(np.ptp(V, axis=0).max()) or 1.0
    for i in range(n):
        if np.linalg.norm(V[(i + 1) % n] - V[i]) <= 1e-12 * scale:
            return f"vertices {i + 1} and {(i + 1) % n + 1} coincide"

    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) <= 1e-12 * scale * scale else (1 if v > 0 else -1)

    def on(a, b, c):        # c on the segment [a, b], knowing they are aligned
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        for j in range(i + 1, n):
            if (j + 1) % n == i or j == (i + 1) % n:
                continue                 # adjacent sides
            c, d = V[j], V[(j + 1) % n]
            o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
            if (o1 != o2 and o3 != o4) or (o1 == 0 and on(a, b, c)) or (o2 == 0 and on(a, b, d)) \
                    or (o3 == 0 and on(c, d, a)) or (o4 == 0 and on(c, d, b)):
                return f"sides {i + 1} and {j + 1} cross"
    if abs(polygon_area(V)) <= 1e-12 * scale * scale:
        return "the polygon has no area"
    return None


# ------------------------------------------------------------------ fillets of polygons

@dataclass(frozen=True)
class Fillet:
    """The arc of circle that rounds the vertex k (1-based) of a polygon,
    tangent to its two sides: from t1 (on the side before the vertex) to t2
    (on the side after it); convex: the polygon is cut (a convex corner),
    else filled (a reflex corner)."""
    vertex: int
    center: tuple
    radius: float
    t1: tuple
    t2: tuple
    theta0: float           # angle of t1 seen from the center (radians)
    sweep: float            # signed angle from t1 to t2 (|sweep| < pi)
    convex: bool

    def points(self, n: int = 33) -> np.ndarray:
        th = self.theta0 + self.sweep * np.linspace(0.0, 1.0, n)
        return np.asarray(self.center) + self.radius * np.column_stack([np.cos(th), np.sin(th)])


def fillet_radii(prim: dict, n: int) -> dict:
    """{vertex (1-based): radius} of the (resolved) fillets of a polygon of
    n vertices; radii that are not positive numbers are left out (errors of
    the validation)."""
    out = {}
    for f in prim.get("fillets") or []:
        k, r = (f.get("vertex"), f.get("radius")) if isinstance(f, dict) else (None, None)
        if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= n \
                and isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0:
            out[k] = float(r)
    return out


def fillets(points, radii: dict) -> tuple[list, list[Fillet], str | None]:
    """The outline of a polygon whose vertices are rounded: (the vertices
    of the polygon with each rounded vertex replaced by its two tangent
    points, the fillets, why it cannot be done or None)."""
    V = np.asarray(points, float)
    n = len(V)
    orient = 1.0 if polygon_area(V) > 0 else -1.0
    found, d = {}, np.zeros(n)
    for k, r in radii.items():
        i = k - 1
        v, p, q = V[i], V[i - 1], V[(i + 1) % n]
        lu, lw = np.linalg.norm(p - v), np.linalg.norm(q - v)
        if lu <= 0 or lw <= 0:
            return [], [], f"vertex {k}: its sides have no length"
        u, w = (p - v) / lu, (q - v) / lw
        theta = math.acos(max(-1.0, min(1.0, float(u @ w))))       # the angle between the two sides
        if theta < 1e-9 or math.pi - theta < 1e-9:
            return [], [], f"vertex {k}: no corner to round (the sides are aligned or folded)"
        d[i] = r / math.tan(theta / 2)
        b = (u + w) / np.linalg.norm(u + w)
        c = v + r / math.sin(theta / 2) * b
        t1, t2 = v + d[i] * u, v + d[i] * w
        a1, a2 = math.atan2(*(t1 - c)[::-1]), math.atan2(*(t2 - c)[::-1])
        sweep = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        e1, e2 = v - p, q - v
        convex = float(e1[0] * e2[1] - e1[1] * e2[0]) * orient > 0
        found[k] = Fillet(k, tuple(map(float, c)), r, tuple(map(float, t1)), tuple(map(float, t2)), a1, sweep,
                          convex)
    for i in range(n):                       # the side i + 1, from vertex i + 1 to i + 2
        L = float(np.linalg.norm(V[(i + 1) % n] - V[i]))
        if d[i] + d[(i + 1) % n] > L * (1 + 1e-9):
            which = [str(k + 1) for k in (i, (i + 1) % n) if d[k] > 0]
            return [], [], (f"side {i + 1}: the fillet{'s' if len(which) > 1 else ''} of vertex "
                            f"{' and '.join(which)} too large for it")
    outline = []
    for i in range(n):
        f = found.get(i + 1)
        outline += [list(f.t1), list(f.t2)] if f else [[float(x) for x in V[i]]]
    return outline, [found[k] for k in sorted(found)], None


def _fillets_of(prim: dict):
    """(outline, fillets) of a polygon with fillets, None without (or if
    they cannot be made: the polygon is then left sharp, the validation
    says why)."""
    pts = prim.get("points") or []
    radii = fillet_radii(prim, len(pts))
    if not radii or len(pts) < 3:
        return None
    outline, found, problem = fillets(pts, radii)
    return None if problem else (np.asarray(outline, float), found)


def rounded_outline(prim: dict, per_arc: int = 16) -> list:
    """The vertices of a polygon, its fillets drawn as per_arc segments
    each (a polygon close to it: the inside tests of the meshing scripts)."""
    rounded = _fillets_of(prim)
    if rounded is None:
        return [[float(x) for x in v] for v in prim["points"]]
    outline, found = rounded
    by = {f.t1: f for f in found}
    out = []
    for v in outline:
        f = by.get(tuple(v))
        out += f.points(per_arc + 1)[:-1].tolist() if f else [list(v)]
    return out


def fillet_ends(shape: dict) -> np.ndarray:
    """The ends of the fillets on the boundary of a 2D shape (n x 2): where
    a side meets an arc tangentially (no corner for the GetFEM mesher, which
    is given them as vertices)."""
    out = []
    try:
        for pc in pieces(shape):
            if pc.side.kind == "fillet":
                out += pc.side.points([pc.t0, pc.t1]).tolist()
    except (ValueError, KeyError, TypeError, IndexError):
        return np.zeros((0, 2))
    return np.unique(np.round(np.asarray(out, float), 14), axis=0) if out else np.zeros((0, 2))


def tangent_points(shape: dict) -> np.ndarray:
    """The points of the boundary of a 2D shape where two of its curves meet
    tangentially (n x 2): the ends of the fillets, the points of the splines
    and their ends at smooth vertices (the GetFEM mesher, given them as
    vertices, meshes well there and each face of the mesh is on one curve;
    not the corners, which it finds itself)."""
    out = [fillet_ends(shape)]
    try:
        for _, prim in primitives(shape):
            if prim.get("type") != "polygon" or not spline_points(prim):
                continue
            V = np.asarray(prim["points"], float)
            n = len(V)
            through = spline_points(prim)
            out += list(through.values())
            out += [V[k][None] for k in smooth_vertices(prim) if k in through or (k - 1) % n in through]
        P = np.vstack([p.reshape(-1, 2) for p in out])
        if len(P) and len(out) > 1:          # those on the boundary of the shape
            P = P[np.abs(sdf(shape, P)) <= 1e-9 * size(shape)]
    except (ValueError, KeyError, TypeError, IndexError):
        P = fillet_ends(shape)
    return np.unique(np.round(P, 14), axis=0) if len(P) else np.zeros((0, 2))


def fillet_problem(prim: dict) -> str | None:
    pts = prim.get("points") or []
    radii = fillet_radii(prim, len(pts))
    return fillets(pts, radii)[2] if radii and len(pts) >= 3 else None


def _in_segment(f: Fillet, P: np.ndarray) -> np.ndarray:
    """In the circular segment between the chord t1 t2 and the arc."""
    c, t1, t2 = (np.asarray(x, float) for x in (f.center, f.t1, f.t2))
    m = (t1 + t2) / 2
    toward = m - c                                  # from the center to the chord, then the arc
    return (np.linalg.norm(P - c, axis=1) < f.radius) & ((P - m) @ toward > 0)


def _arc_distance(f: Fillet, P: np.ndarray) -> np.ndarray:
    c = np.asarray(f.center, float)
    D = P - c
    ang = np.arctan2(D[:, 1], D[:, 0])
    s = ((ang - f.theta0) * np.sign(f.sweep)) % (2 * math.pi)        # the angle along the arc
    on = s <= abs(f.sweep)
    ends = np.minimum(np.linalg.norm(P - np.asarray(f.t1), axis=1), np.linalg.norm(P - np.asarray(f.t2), axis=1))
    return np.where(on, np.abs(np.linalg.norm(D, axis=1) - f.radius), ends)


def rounded_polygon_sdf(outline: np.ndarray, found: list, P: np.ndarray) -> np.ndarray:
    """Signed euclidean distance to a polygon with rounded vertices (outline,
    fillets from fillets())."""
    base = polygon_sdf(outline, P)                  # the polygon with its corners cut (or filled)
    inside = base < 0
    dist = np.abs(base)
    corner = {tuple(f.t1): f for f in found}
    n = len(outline)
    chords = [i for i in range(n) if tuple(outline[i]) in corner and tuple(outline[(i + 1) % n]) ==
              corner[tuple(outline[i])].t2]
    if chords:                                       # the chords are not on the boundary: the arcs are
        keep = [i for i in range(n) if i not in chords]
        A, B = outline[keep], outline[[(i + 1) % n for i in keep]]
        D = B - A
        L2 = np.maximum(np.einsum("ij,ij->i", D, D), 1e-300)
        W = P[:, None, :] - A[None, :, :]
        t = np.clip(np.einsum("mnk,nk->mn", W, D) / L2, 0.0, 1.0)
        dist = np.min(np.linalg.norm(W - t[:, :, None] * D[None], axis=2), axis=1) if keep else \
            np.full(len(P), np.inf)
    for f in found:
        dist = np.minimum(dist, _arc_distance(f, P))
        seg = _in_segment(f, P)
        inside = (inside | seg) if f.convex else (inside & ~seg)
    return np.where(inside, -dist, dist)


# ------------------------------------------------------------------ splines: curved sides of polygons
#
# "splines": [{"side": k, "points": [[x, y], ...]}] makes the side k (from
# vertex k to vertex k + 1, 1-based) a cubic spline through these points
# (C2, parametrized by the length of its chords); "smooth": [k, ...] makes
# the boundary tangent at the vertex k: the end of a spline there takes the
# direction of the straight side on the other side of the vertex (or, two
# splines meeting there, the direction from the point before the vertex to
# the point after it); otherwise the end is free (natural: no curvature).
# Each spline is a chain of cubic Bezier curves (one per interval between
# its points): their control points are what the meshers are given.

def spline_points(prim: dict) -> dict:
    """{side (0-based): its points (m x 2)} of the (resolved) splines of a
    polygon; the entries that are not valid are left out (errors of the
    validation)."""
    n = len(prim.get("points") or [])
    out = {}
    for s in prim.get("splines") or []:
        if not isinstance(s, dict):
            continue
        k, pts = s.get("side"), s.get("points")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n or k - 1 in out \
                or not isinstance(pts, list):
            continue
        try:
            P = np.asarray(pts, float).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(P)):
            out[k - 1] = P
    return out


def smooth_vertices(prim: dict) -> set:
    """The vertices (0-based) where the boundary of a polygon is smooth."""
    n = len(prim.get("points") or [])
    return {k - 1 for k in prim.get("smooth") or [] if isinstance(k, int) and not isinstance(k, bool)
            and 1 <= k <= n}


def _natural_or_clamped(Q: np.ndarray, T0, T1) -> np.ndarray:
    """The derivatives at the points Q (n x 2, n >= 2) of the C2 cubic
    spline through them, parametrized by the length of its chords; T0, T1:
    the unit tangents at the ends, or None (no curvature there)."""
    n = len(Q)
    h = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    h = np.maximum(h, 1e-300)
    A = np.zeros((n, n))
    R = np.zeros((n, 2))
    for j in range(1, n - 1):
        A[j, j - 1], A[j, j], A[j, j + 1] = h[j], 2 * (h[j - 1] + h[j]), h[j - 1]
        R[j] = 3 * (h[j] * (Q[j] - Q[j - 1]) / h[j - 1] + h[j - 1] * (Q[j + 1] - Q[j]) / h[j])
    if T0 is None:
        A[0, 0], A[0, 1], R[0] = 2.0, 1.0, 3 * (Q[1] - Q[0]) / h[0]
    else:
        A[0, 0], R[0] = 1.0, T0
    if T1 is None:
        A[-1, -2], A[-1, -1], R[-1] = 1.0, 2.0, 3 * (Q[-1] - Q[-2]) / h[-1]
    else:
        A[-1, -1], R[-1] = 1.0, T1
    return np.linalg.solve(A, R)


def spline_controls(prim: dict) -> dict:
    """{side (0-based): the control points ((3 m + 1) x 2) of the chain of
    cubic Bezier curves of its spline} for the splines of a polygon."""
    V = np.asarray(prim.get("points") or [], float).reshape(-1, 2)
    n = len(V)
    through = spline_points(prim)
    if not through or n < 2:
        return {}
    smooth = smooth_vertices(prim)

    def chain(i):           # the points of the side i, its ends included
        return np.vstack([V[i], through.get(i, np.zeros((0, 2))), V[(i + 1) % n]])

    def unit(d):
        L = float(np.linalg.norm(d))
        return d / L if L > 0 else None

    def tangent(k):         # the direction of the boundary at the smooth vertex k, along it
        before, after = chain((k - 1) % n), chain(k)
        return unit(after[1] - before[-2])

    out = {}
    for i, inner in through.items():
        Q = chain(i)
        T0 = tangent(i) if i in smooth else None
        T1 = tangent((i + 1) % n) if (i + 1) % n in smooth else None
        D = _natural_or_clamped(Q, T0, T1)
        C = [Q[0]]
        for j in range(len(Q) - 1):
            hj = float(np.linalg.norm(Q[j + 1] - Q[j]))
            C += [Q[j] + hj * D[j] / 3, Q[j + 1] - hj * D[j + 1] / 3, Q[j + 1]]
        out[i] = np.asarray(C, float)
    return out


def bezier_points(C: np.ndarray, t) -> np.ndarray:
    """Points of a chain of cubic Bezier curves (control points C, (3 m + 1)
    x 2) at t in [0, 1] (the curve j for t in [j / m, (j + 1) / m])."""
    t = np.atleast_1d(np.asarray(t, float))
    m = (len(C) - 1) // 3
    j = np.clip(np.floor(t * m).astype(int), 0, m - 1)
    u = (t * m - j)[:, None]
    s = 1 - u
    P0, P1, P2, P3 = C[3 * j], C[3 * j + 1], C[3 * j + 2], C[3 * j + 3]
    return s ** 3 * P0 + 3 * s * s * u * P1 + 3 * s * u * u * P2 + u ** 3 * P3


def bezier_tangents(C: np.ndarray, t) -> np.ndarray:
    """Derivatives (with respect to t) of a chain of cubic Bezier curves."""
    t = np.atleast_1d(np.asarray(t, float))
    m = (len(C) - 1) // 3
    j = np.clip(np.floor(t * m).astype(int), 0, m - 1)
    u = (t * m - j)[:, None]
    s = 1 - u
    P0, P1, P2, P3 = C[3 * j], C[3 * j + 1], C[3 * j + 2], C[3 * j + 3]
    return 3 * m * (s * s * (P1 - P0) + 2 * s * u * (P2 - P1) + u * u * (P3 - P2))


def bezier_part(C: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """The control points of the part t0 <= t <= t1 of a chain of cubic
    Bezier curves (each curve cut by de Casteljau's algorithm)."""
    m = (len(C) - 1) // 3
    out = []
    for j in range(m):
        a, b = max(t0, j / m), min(t1, (j + 1) / m)
        if b <= a + 1e-15:
            continue
        P = C[3 * j:3 * j + 4]
        ua, ub = a * m - j, b * m - j

        def split(P, u):     # the part [u, 1] of a cubic curve
            p01, p12, p23 = P[0] + u * (P[1] - P[0]), P[1] + u * (P[2] - P[1]), P[2] + u * (P[3] - P[2])
            p012, p123 = p01 + u * (p12 - p01), p12 + u * (p23 - p12)
            q = p012 + u * (p123 - p012)
            return np.array([q, p123, p23, P[3]])
        if ua > 0:
            P = split(P, ua)
            ub = (ub - ua) / (1 - ua)
        if ub < 1:                       # the part [0, ub]: the reversed curve cut at 1 - ub
            P = split(P[::-1], 1 - ub)[::-1]
        out += ([P[0]] if not out else []) + [P[1], P[2], P[3]]
    return np.asarray(out, float)


def boundary_polyline(prim: dict, per_span: int = 24, per_arc: int = 16) -> np.ndarray:
    """The boundary of a polygon with its fillets and splines as a closed
    polyline (n x 2, the last point not repeated): the inside tests and the
    distances of the curved polygons (an approximation, close for these
    counts)."""
    V = np.asarray(prim["points"], float)
    n = len(V)
    controls = spline_controls(prim)
    rounded = _fillets_of(prim)
    by = {f.vertex - 1: f for f in rounded[1]} if rounded is not None else {}
    out = []
    for i in range(n):
        f = by.get(i)
        out += f.points(per_arc + 1).tolist() if f is not None else [V[i].tolist()]
        if i in controls:                 # the side i, after the vertex i
            C = controls[i]
            m = (len(C) - 1) // 3
            out += bezier_points(C, np.linspace(0, 1, m * per_span + 1)[1:-1]).tolist()
    return np.asarray(out, float)


def _side_tangents(side, t) -> np.ndarray:
    """Derivatives along a side of a polygon (segment, fillet or spline), in
    the direction of the boundary."""
    t = np.atleast_1d(np.asarray(t, float))
    if side.kind == "segment":
        return np.tile(np.subtract(side.b, side.a), (len(t), 1))
    if side.kind == "spline":
        return bezier_tangents(np.asarray(side.a, float), t)
    r, th0, sw = side.b
    th = th0 + sw * t
    return r * sw * np.column_stack([-np.sin(th), np.cos(th)])


def _side_seconds(side, t) -> np.ndarray:
    """Second derivatives along a side of a polygon."""
    t = np.atleast_1d(np.asarray(t, float))
    if side.kind == "segment":
        return np.zeros((len(t), 2))
    if side.kind == "spline":
        C = np.asarray(side.a, float)
        m = (len(C) - 1) // 3
        j = np.clip(np.floor(t * m).astype(int), 0, m - 1)
        u = (t * m - j)[:, None]
        P0, P1, P2, P3 = C[3 * j], C[3 * j + 1], C[3 * j + 2], C[3 * j + 3]
        return 6 * m * m * ((1 - u) * (P2 - 2 * P1 + P0) + u * (P3 - 2 * P2 + P1))
    r, th0, sw = side.b
    th = th0 + sw * t
    return -r * sw * sw * np.column_stack([np.cos(th), np.sin(th)])


def _side_samples(side) -> tuple[np.ndarray, np.ndarray]:
    """Parameters and points sampling a curved side (16 per curve)."""
    m = 16 * ((len(side.a) - 1) // 3 if side.kind == "spline" else 1)
    ts = np.linspace(0.0, 1.0, m + 1)
    return ts, side.points(ts)


def _side_nearest(side, P: np.ndarray, samples=None, start=None) -> tuple[np.ndarray, np.ndarray]:
    """(distance, parameter) of the nearest points of a side of a polygon
    (start: the parameters to begin from, near the nearest points)."""
    if side.kind == "segment":
        a, d = np.asarray(side.a, float), np.subtract(side.b, side.a)
        t = np.clip((P - a) @ d / max(float(d @ d), 1e-300), 0.0, 1.0)
        return np.linalg.norm(a + t[:, None] * d - P, axis=1), t
    ts, S = samples if samples is not None else _side_samples(side)
    if start is None:
        t = ts[np.argmin(np.einsum("pkj,pkj->pk", P[:, None, :] - S[None], P[:, None, :] - S[None]), axis=1)]
    else:
        t = np.asarray(start, float).copy()
    dist = np.linalg.norm(side.points(t) - P, axis=1)
    h = 0.5 / (len(ts) - 1)             # a Newton step stays near its sample
    todo = np.arange(len(P))
    for _ in range(8):                  # Newton on (C(t) - P).C'(t) = 0, the steps that get closer
        tt, Q = t[todo], P[todo]
        R, D, D2 = side.points(tt) - Q, _side_tangents(side, tt), _side_seconds(side, tt)
        f = np.einsum("ij,ij->i", R, D)
        df = np.einsum("ij,ij->i", D, D) + np.einsum("ij,ij->i", R, D2)
        step = np.clip(np.where(df > 0, -f / np.where(df > 0, df, 1.0), -np.sign(f) * h), -h, h)
        tn = np.clip(tt + step, 0.0, 1.0)
        dn = np.linalg.norm(side.points(tn) - Q, axis=1)
        better = dn < dist[todo]
        t[todo[better]], dist[todo[better]] = tn[better], dn[better]
        todo = todo[better & (np.abs(step) > 1e-13)]   # the converged ones stop
        if not len(todo):
            break
    return np.linalg.norm(side.points(t) - P, axis=1), t


def _crossings_inside(V: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Inside the polygon of vertices V, by the parity of the crossings of
    a half line (no distances; matplotlib's Path when there, in C++)."""
    try:
        from matplotlib.path import Path
    except ImportError:
        Path = None
    if Path is not None and len(P) > 64:
        return Path(np.asarray(V, float)).contains_points(np.asarray(P, float))
    A, B = V, np.roll(V, -1, axis=0)
    y, x = P[:, 1:2], P[:, 0:1]
    ya, yb = A[None, :, 1], B[None, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = A[None, :, 0] + (y - ya) * (B[None, :, 0] - A[None, :, 0]) / (yb - ya)
    return np.sum(((ya > y) != (yb > y)) & (x < xc), axis=1) % 2 == 1


_CURVED_CACHE: dict = {}


def _curved_parts(prim: dict):
    """The sides of a curved polygon, a fine polyline of its boundary and
    its orientation, kept for the last few polygons (the same polygon is
    asked many times for one view: fill, pieces, interfaces)."""
    import json
    try:
        key = json.dumps(prim, sort_keys=True, default=str)
    except (TypeError, ValueError):
        key = None
    hit = _CURVED_CACHE.get(key) if key is not None else None
    if hit is not None:
        return hit
    all_sides = sides(prim)
    samples = [_side_samples(side) if side.kind != "segment" else None for side in all_sides]
    A, B, which, ta, tb = [], [], [], [], []      # the chords of the samples, by side
    for i, (side, smp) in enumerate(zip(all_sides, samples)):
        ts, S = smp if smp is not None else (np.array([0.0, 1.0]), np.array([side.a, side.b], float))
        A.append(S[:-1]), B.append(S[1:]), which.append(np.full(len(ts) - 1, i)), ta.append(ts[:-1]), tb.append(ts[1:])
    soup = tuple(np.concatenate(x) for x in (A, B, which, ta, tb))
    poly = boundary_polyline(prim, per_span=16, per_arc=16)    # (the crossings)
    out = (all_sides, samples, soup, poly, polygon_area(poly) > 0)
    if key is not None:
        if len(_CURVED_CACHE) > 16:
            _CURVED_CACHE.clear()
        _CURVED_CACHE[key] = out
    return out


def _segments_nearest(A: np.ndarray, B: np.ndarray, P: np.ndarray):
    """(distance, index, position along it) of the nearest of the segments
    A-B for each point P."""
    D = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", D, D), 1e-300)
    R = P[:, None, :] - A[None]
    u = np.clip(np.einsum("pkj,kj->pk", R, D) / L2[None], 0.0, 1.0)
    R -= u[..., None] * D[None]
    d2 = np.einsum("pkj,pkj->pk", R, R)
    j = np.argmin(d2, axis=1)
    rows = np.arange(len(P))
    return np.sqrt(d2[rows, j]), j, u[rows, j]


def _soup_nearest(A: np.ndarray, B: np.ndarray, P: np.ndarray, group: int = 16):
    """_segments_nearest for many segments: in groups of a few consecutive
    ones, each group tried only for the points that its bounding box may
    hold nearer than the nearest group start."""
    if len(A) <= 2 * group:
        return _segments_nearest(A, B, P)
    starts = list(range(0, len(A), group))
    lo = np.array([np.minimum(A[i:i + group].min(axis=0), B[i:i + group].min(axis=0)) for i in starts])
    hi = np.array([np.maximum(A[i:i + group].max(axis=0), B[i:i + group].max(axis=0)) for i in starts])
    W = A[::group]
    ub = np.sqrt(np.min(((P[:, None, :] - W[None]) ** 2).sum(axis=2), axis=1)) * (1 + 1e-12)
    dist, idx, pos = np.full(len(P), np.inf), np.zeros(len(P), int), np.zeros(len(P))
    for g, i in enumerate(starts):
        gap = np.maximum(np.maximum(lo[g] - P, P - hi[g]), 0.0)
        k = np.flatnonzero(np.einsum("ij,ij->i", gap, gap) <= ub * ub)
        if len(k):
            d, j, u = _segments_nearest(A[i:i + group], B[i:i + group], P[k])
            better = d < dist[k]
            kb = k[better]
            dist[kb], idx[kb], pos[kb] = d[better], i + j[better], u[better]
    return dist, idx, pos


def curved_polygon_sdf(prim: dict, P: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Signed euclidean distance to a polygon with splines (and fillets):
    the nearest chord of samples of the sides gives the nearest side and a
    start on it, Newton's method the nearest point (all the sides for the
    points whose nearest point is an end of a side); inside by the normal
    there, or, at a vertex, by the crossings with a polyline."""
    P = np.asarray(P, float)
    if len(P) > chunk:
        return np.concatenate([curved_polygon_sdf(prim, P[i:i + chunk], chunk) for i in range(0, len(P), chunk)])
    all_sides, samples, (A, B, which, ta, tb), poly, ccw = _curved_parts(prim)
    _, j, u = _soup_nearest(A, B, P)               # the nearest chord: its side, a start on it
    side_of, start = which[j].copy(), ta[j] + u * (tb[j] - ta[j])
    best, tpar = np.zeros(len(P)), np.zeros(len(P))
    for i, side in enumerate(all_sides):
        k = np.flatnonzero(side_of == i)
        if len(k):
            best[k], tpar[k] = _side_nearest(side, P[k], samples[i], start[k])
    # at an end of its side, the nearest point may be on the next side: all the sides then
    ends = np.flatnonzero((tpar <= 1e-12) | (tpar >= 1 - 1e-12))
    if len(ends):
        for i, side in enumerate(all_sides):
            d, t = _side_nearest(side, P[ends], samples[i])
            closer = d < best[ends]
            best[ends[closer]], tpar[ends[closer]], side_of[ends[closer]] = d[closer], t[closer], i
    inside = _crossings_inside(poly, P)
    sgn = np.full(len(P), np.nan)                  # inside by the normal at the nearest point
    for i, side in enumerate(all_sides):
        k = np.flatnonzero(side_of == i)
        if len(k):
            t = tpar[k]
            T = _side_tangents(side, t)
            left = np.einsum("ij,ij->i", P[k] - side.points(t), np.column_stack([-T[:, 1], T[:, 0]]))
            sgn[k] = np.where((t > 1e-9) & (t < 1 - 1e-9), np.sign(left if ccw else -left), np.nan)
    decided = ~(np.isnan(sgn) | (sgn == 0))
    inside[decided] = sgn[decided] > 0
    return np.where(inside, -best, best)


def polyline_profile(shape: dict) -> dict:
    """The 2D tree with its polygons with fillets or splines replaced by
    polygons close to them (boundary_polyline): for the tests of the meshing
    scripts that only know straight polygons."""
    if not isinstance(shape, dict):
        return shape
    t = shape.get("type")
    if t == "polygon" and (shape.get("fillets") or shape.get("splines")) and not incomplete(shape):
        try:
            return {"type": "polygon", "points": boundary_polyline(shape, per_span=64, per_arc=64).tolist()}
        except (TypeError, ValueError, KeyError, np.linalg.LinAlgError):
            return shape
    if "of" in shape:
        return dict(shape, of=[polyline_profile(s) for s in shape["of"]])
    return shape


def outline_parts(prim: dict) -> list:
    """The boundary of a polygon with fillets or splines, in order, as
    parts for the CAD meshers: ("line", a, b), ("arc", a, b, center) (the
    smaller arc from a to b) and ("bezier", [p0, p1, p2, p3]) (a cubic
    Bezier curve), each starting where the one before ends."""
    V = [[float(x) for x in v] for v in prim["points"]]
    n = len(V)
    controls = spline_controls(prim)
    rounded = _fillets_of(prim)
    by = {f.vertex - 1: f for f in rounded[1]} if rounded is not None else {}
    out = []
    for i in range(n):
        f = by.get(i)
        if f is not None:
            out.append(("arc", list(f.t1), list(f.t2), list(f.center)))
        if i in controls:
            C = controls[i].tolist()
            out += [("bezier", C[3 * j:3 * j + 4]) for j in range((len(C) - 1) // 3)]
            continue
        a = list(f.t2) if f is not None else V[i]
        g = by.get((i + 1) % n)
        b = list(g.t1) if g is not None else V[(i + 1) % n]
        out.append(("line", a, b))
    return out


def spline_problem(prim: dict) -> str | None:
    """Why the splines of a polygon cannot be made (None if they can): a
    fillet at an end of a spline, a smooth vertex between two straight
    sides, a boundary that crosses itself."""
    V = prim.get("points") or []
    n = len(V)
    through = spline_points(prim)
    for k in sorted(fillet_radii(prim, n)):
        if (k - 1) in through or (k - 2) % n in through:
            return f"vertex {k}: a fillet at an end of a spline (make the vertex smooth instead)"
    for k in sorted(smooth_vertices(prim)):
        if k not in through and (k - 1) % n not in through:
            return f"vertex {k + 1}: smooth between two straight sides (no spline there)"
    if through:
        P = boundary_polyline(prim, per_span=12)
        problem = polygon_problem(P)
        if problem:
            return "the curved sides cross the boundary" if "cross" in problem or "coincide" in problem \
                else problem
    return None


def size(shape: dict) -> float:
    try:
        lo, hi = bounds(shape)
    except ValueError:                     # nothing yet (an empty geometry)
        return 1.0
    return float(np.linalg.norm(hi - lo)) or 1.0


def is_empty(shape: dict) -> bool:
    """Nothing to draw yet: no complete primitive."""
    return next(primitives(shape), None) is None


# ------------------------------------------------------------------ sides and pieces

@dataclass(frozen=True)
class Side:
    """A side of a primitive, parametrized by t in [0, 1]."""
    shape: tuple            # path of the primitive
    side: str               # xmin, xmax, ymin, ymax (rectangle) or circle (disk)
    kind: str               # "segment", "arc", "ellipse" (closed curves: t = 0 and 1 meet), "fillet" or "spline"
    a: tuple                # segment: first point; arc, ellipse, fillet: center
    b: tuple                # segment: second point; arc: (radius, 0); ellipse: (r1, r2, angle in radians);
    #                         fillet (an arc from angle theta0, by sweep, in radians): (radius, theta0, sweep);
    #                         spline: a = the control points of its chain of cubic Bezier curves, b = ()

    @property
    def ref(self) -> dict:
        return {"shape": list(self.shape), "side": self.side}

    @property
    def closed(self) -> bool:
        return self.kind not in ("segment", "fillet", "spline")

    def length(self, t0: float = 0.0, t1: float = 1.0) -> float:
        if self.kind == "segment":
            return float(np.linalg.norm(np.subtract(self.b, self.a))) * (t1 - t0)
        if self.kind == "arc":
            return 2 * math.pi * self.b[0] * (t1 - t0)
        if self.kind == "fillet":
            return abs(self.b[0] * self.b[2]) * (t1 - t0)
        P = self.points(np.linspace(t0, t1, 2049))
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    def points(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        if self.kind == "segment":
            a, b = np.asarray(self.a), np.asarray(self.b)
            return a + t[:, None] * (b - a)
        if self.kind == "spline":
            return bezier_points(np.asarray(self.a, float), t)
        if self.kind == "ellipse":
            (r1, r2, th), c = self.b, np.asarray(self.a)
            u, v = r1 * np.cos(2 * math.pi * t), r2 * np.sin(2 * math.pi * t)
            return c + np.column_stack([u * math.cos(th) - v * math.sin(th), u * math.sin(th) + v * math.cos(th)])
        c, r = np.asarray(self.a), self.b[0]
        th = self.b[1] + self.b[2] * t if self.kind == "fillet" else 2 * math.pi * t
        return c + r * np.column_stack([np.cos(th), np.sin(th)])

    def normals(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        if self.kind == "segment":
            d = np.asarray(self.b) - np.asarray(self.a)
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)
            return np.tile(n, (len(t), 1))
        if self.kind == "spline":
            D = bezier_tangents(np.asarray(self.a, float), t)
            N = np.column_stack([-D[:, 1], D[:, 0]])
            return N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
        if self.kind == "ellipse":
            r1, r2, th = self.b
            nu, nv = np.cos(2 * math.pi * t) / r1, np.sin(2 * math.pi * t) / r2
            N = np.column_stack([nu * math.cos(th) - nv * math.sin(th), nu * math.sin(th) + nv * math.cos(th)])
            return N / np.linalg.norm(N, axis=1)[:, None]
        th = self.b[1] + self.b[2] * t if self.kind == "fillet" else 2 * math.pi * t
        return np.column_stack([np.cos(th), np.sin(th)])


def side_names(prim: dict) -> tuple:
    """Names of the sides of a primitive."""
    t = prim.get("type")
    if t == "rectangle":
        return RECTANGLE_SIDES
    if t == "polygon":
        if prim.get("side_names"):            # a turned rectangle (transforms.flatten)
            return tuple(prim["side_names"])
        n = len(prim.get("points", []))
        return tuple(f"side{i + 1}" for i in range(n)) + tuple(f"fillet{k}" for k in sorted(fillet_radii(prim, n)))
    if t == "ellipse":
        return ("ellipse",)
    return ("circle",)


def sides(shape: dict) -> list[Side]:
    out = []
    for path, p in primitives(shape):
        if p["type"] == "rectangle":
            (x0, y0), (x1, y1) = p["min"], p["max"]
            out += [Side(path, "xmin", "segment", (x0, y0), (x0, y1)),
                    Side(path, "xmax", "segment", (x1, y0), (x1, y1)),
                    Side(path, "ymin", "segment", (x0, y0), (x1, y0)),
                    Side(path, "ymax", "segment", (x0, y1), (x1, y1))]
        elif p["type"] == "ball":
            out.append(Side(path, "circle", "arc", tuple(p["center"]), (float(p["radius"]), 0.0)))
        elif p["type"] == "ellipse":
            out.append(Side(path, "ellipse", "ellipse", tuple(float(x) for x in p["center"]),
                            (float(p["radii"][0]), float(p["radii"][1]), math.radians(float(p.get("angle", 0.0))))))
        elif p["type"] == "polygon":
            V = [tuple(float(x) for x in v) for v in p["points"]]
            names = side_names(p)
            rounded = _fillets_of(p)
            controls = spline_controls(p)
            if rounded is None:
                out += [Side(path, names[i], "spline", tuple(map(tuple, controls[i].tolist())), ()) if i in controls
                        else Side(path, names[i], "segment", V[i], V[(i + 1) % len(V)]) for i in range(len(V))]
                continue
            n, by = len(V), {f.vertex: f for f in rounded[1]}
            for i in range(n):                 # the sides, between the tangent points of the fillets
                if i in controls:
                    out.append(Side(path, names[i], "spline", tuple(map(tuple, controls[i].tolist())), ()))
                    continue
                a = by[i + 1].t2 if i + 1 in by else V[i]
                b = by[(i + 1) % n + 1].t1 if (i + 1) % n + 1 in by else V[(i + 1) % n]
                out.append(Side(path, names[i], "segment", tuple(a), tuple(b)))
            out += [Side(path, f"fillet{f.vertex}", "fillet", f.center, (f.radius, f.theta0, f.sweep))
                    for f in rounded[1]]
    return out


@dataclass(frozen=True)
class Piece:
    """The part t0 <= t <= t1 of a side that lies on the boundary of the domain
    (for a circle, t1 may exceed 1: the piece goes through the angle 0)."""
    side: Side
    t0: float
    t1: float

    def points(self, n: int = 64) -> np.ndarray:
        if self.side.kind == "fillet":
            m = max(2, int(n * abs(self.side.b[2]) / (2 * math.pi) * (self.t1 - self.t0)) + 4)
        elif self.side.kind == "spline":
            m = max(2, int(n * (len(self.side.a) - 1) // 3 * (self.t1 - self.t0)) + 2)
        else:
            m = max(2, int(n * (self.t1 - self.t0)) + 2) if self.side.closed else 2
        return self.side.points(np.linspace(self.t0, self.t1, m))

    def criterion(self) -> dict:
        """Concrete description, used to find the faces of a mesh."""
        if self.side.kind == "segment":
            p = self.side.points([self.t0, self.t1])
            return {"segment": [p[0].tolist(), p[1].tolist()]}
        if self.side.kind == "ellipse":
            r1, r2, th = self.side.b
            return {"ellipse": {"center": [float(c) for c in self.side.a], "radii": [float(r1), float(r2)],
                                "angle": float(th), "from": float(self.t0), "to": float(self.t1)}}
        if self.side.kind == "spline":
            return {"bezier": bezier_part(np.asarray(self.side.a, float), self.t0, self.t1).tolist()}
        if self.side.kind == "fillet":
            r, th0, sw = self.side.b
            a, b = th0 + sw * self.t0, th0 + sw * self.t1
            lo = min(a, b)
            return {"arc": {"center": [float(c) for c in self.side.a], "radius": float(r),
                            "from": float(math.degrees(lo) % 360.0), "sweep": float(math.degrees(abs(b - a)))}}
        return {"arc": {"center": [float(c) for c in self.side.a], "radius": float(self.side.b[0]),
                        "from": float(360.0 * self.t0), "sweep": float(360.0 * (self.t1 - self.t0))}}

    def distance(self, point) -> float:
        pts = self.points(256)
        q = np.asarray(point, float)[:2]
        a, b = pts[:-1], pts[1:]
        d = b - a
        L2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-300)
        s = np.clip(np.einsum("ij,ij->i", q - a, d) / L2, 0, 1)
        return float(np.min(np.linalg.norm(a + s[:, None] * d - q, axis=1)))


def _separates(level, side: Side, t, eps) -> np.ndarray:
    """True where the side separates level < 0 from level >= 0."""
    P, N = side.points(t), side.normals(t)
    return (level(P + eps * N) < 0) != (level(P - eps * N) < 0)


def _runs(side: Side, test, samples: int) -> list[Piece]:
    """The parts of a side where test(t) (boolean array) is true, as pieces
    whose ends are refined by bisection."""
    t = (np.arange(samples) + 0.5) / samples
    on = test(t)
    if not on.any():
        return []
    if on.all():
        return [Piece(side, 0.0, 1.0)]
    changes = np.flatnonzero(on[1:] != on[:-1])
    # the transitions between the samples i and i + 1, all bisected together
    lo, hi, state = t[changes].copy(), t[changes + 1].copy(), on[changes]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        same = test(mid) == state
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    ends = (0.5 * (lo + hi)).tolist()
    starts = ends[0::2] if not on[0] else [0.0] + ends[1::2]
    stops = ends[1::2] if not on[0] else ends[0::2]
    if on[-1]:
        stops = list(stops) + [1.0]
    runs = list(zip(starts, stops))
    if side.closed and on[0] and on[-1] and len(runs) > 1:   # through angle 0
        (s0, e0), (sl, el) = runs[0], runs[-1]
        runs = [(sl, 1.0 + e0)] + runs[1:-1]
    return [Piece(side, a, b) for a, b in runs if b > a]


def pieces(shape: dict, samples: int = 720, level=None) -> list[Piece]:
    """The boundary of the domain, as parts of the sides of the primitives.

    A point of a side is on the boundary if it separates inside and outside,
    tested at a distance eps on each side; where two sides are tangent, this
    overestimates a piece by about sqrt(2 eps), hence a small eps.
    level: another function negative inside (e.g. region_level) whose
    boundary is wanted instead (it must be made of sides of the primitives)."""
    if level is None and _single(shape):          # all its sides, whole
        return [Piece(side, 0.0, 1.0) for side in sides(shape)]
    eps = 1e-9 * size(shape)
    level = level or (lambda P: sdf(shape, P))
    out = []
    for side in sides(shape):
        out += _runs(side, lambda t, side=side: _separates(level, side, t, eps), samples)
    return out


def _single(shape: dict) -> bool:
    """A single primitive (no boolean operation, no copies)."""
    prims = list(primitives(shape))
    return len(prims) == 1 and prims[0][1].get("type") not in BOOLEANS and (
        shape.get("type") not in BOOLEANS and not T.has_transforms(shape))


def key_points(shape: dict) -> np.ndarray:
    """The points that the drawing snaps to: corners and centers of the
    primitives, vertices of the polygons, ends of the boundary pieces and of
    the interfaces (where sides cross)."""
    out = []
    for _, s in primitives(shape):
        t = s["type"]
        if t == "rectangle":
            (x0, y0), (x1, y1) = s["min"], s["max"]
            out += [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        elif t in ("ball", "ellipse"):
            out.append(list(s["center"]))
        elif t == "polygon":
            out += [list(p) for p in s["points"]]
    try:
        for p in pieces(shape) + interfaces(shape):
            if not (p.side.closed and p.t1 - p.t0 >= 1 - 1e-9):
                out += p.side.points([p.t0, p.t1]).tolist()
    except (ValueError, KeyError, TypeError, IndexError):
        pass
    if not out:
        return np.zeros((0, 2))
    return np.unique(np.round(np.asarray(out, float), 12), axis=0)


def interfaces(shape: dict, samples: int = 720) -> list[Piece]:
    """The parts of the sides of the primitives that lie inside the domain:
    the interfaces between its regions (e.g. the circle of an inclusion)."""
    if _single(shape):                            # no interfaces in a single primitive
        return []
    eps = 1e-9 * size(shape)
    out = []
    for side in sides(shape):
        def inside(t, side=side):
            P, N = side.points(t), side.normals(t)
            return (sdf(shape, P + eps * N) < 0) & (sdf(shape, P - eps * N) < 0)
        out += _runs(side, inside, samples)
    return out


def interface_points(shape: dict, h: float) -> np.ndarray:
    """Points every h or so along the interfaces, ends included (2 x n): given
    to the GetFEM mesher as fixed vertices, they make the mesh follow the
    interfaces, so that each element is in one region."""
    pts = []
    for pc in interfaces(shape):
        side = pc.side
        length = side.length(pc.t0, pc.t1)
        n = max(1, int(math.ceil(length / h)))
        closed = side.closed and pc.t1 - pc.t0 >= 1.0 - 1e-12
        t = np.linspace(pc.t0, pc.t1, n + 1)[:-1 if closed else None]
        if closed and n < 8:
            t = np.linspace(0.0, 1.0, 8, endpoint=False)
        pts += list(side.points(t))
    kept: list[np.ndarray] = []
    for p in pts:                      # ends shared by two interfaces: once
        if all(np.linalg.norm(p - q) > 0.25 * h for q in kept):
            kept.append(p)
    return np.array(kept, float).T.reshape(2, -1)


# ------------------------------------------------------------------ regions

def _paths(shape: dict) -> list[tuple]:
    return source_paths(shape)


def region_at(shape: dict, point) -> list[list[int]] | None:
    """The region of a point of the domain: the paths of the primitives that
    contain it (None outside the domain). A region is stored this way, e.g.
    [[0], [1]] for "in rectangle 1 and in disk 2" (and in no other primitive)."""
    P = np.asarray(point, float)[None, :2]
    if sdf(shape, P)[0] >= 0:
        return None
    inside = []
    for path, prim in primitives(shape):
        if sdf(prim, P)[0] < 0 and list(path) not in inside:
            inside.append(list(path))
    return inside


def region_level(shape: dict, regions: list) -> callable:
    """Function of the points (n x 2), negative exactly in the given regions
    (each one a list of paths, see region_at)."""
    prims = list(primitives(shape))

    def level(P):
        d = sdf(shape, P)
        each = {}
        for path, prim in prims:            # the copies of a primitive (arrays): together
            v = sdf(prim, P)
            each[path] = np.minimum(each[path], v) if path in each else v
        out = np.full(len(P), np.inf)
        for reg in regions:
            inside = {tuple(p) for p in reg}
            r = d.copy()
            for path, v in each.items():
                r = np.maximum(r, v if path in inside else -v)
            out = np.minimum(out, r)
        return out
    return level


def region_label(shape: dict, region: list) -> str:
    """'rectangle 1 + disk 2': the primitives containing the region."""
    from .model import shape_word
    names = _paths(shape)
    parts = []
    for p in region:
        prim = primitive_at(shape, p)
        if tuple(p) in names and prim is not None:
            parts.append(f"{shape_word(prim['type'], 2)} {names.index(tuple(p)) + 1}")
        else:
            parts.append("a missing shape")
    return " + ".join(parts) or "no shape"


def toggle_region(criteria: dict, region: list) -> dict:
    """Criteria after a click in a region: added, or removed if already there
    (another kind of criterion is replaced by the clicked region)."""
    key = sorted(tuple(p) for p in region)
    if criteria.get("type") != "regions":
        return {"type": "regions", "regions": [[list(p) for p in key]]}
    regs = [[list(p) for p in r] for r in criteria.get("regions", [])]
    kept = [r for r in regs if sorted(tuple(p) for p in r) != key]
    if len(kept) == len(regs):
        kept.append([list(p) for p in key])
    return {"type": "regions", "regions": kept}


def domain_level(shape: dict, selection: str, table: dict):
    """Function negative in a domain selection (table: model.selection_table),
    for the preview; None if it cannot be evaluated on the geometry."""
    sel = table.get(selection)
    if sel is None or sel.get("kind") != "domain":
        return None
    if selection == "all":
        return lambda P: sdf(shape, P)
    crit = sel.get("criteria", {})
    t = crit.get("type")
    try:
        if t == "regions":
            return region_level(shape, crit.get("regions", []))
        if t == "in_box":
            box = {"type": "rectangle", "min": crit["min"][:2], "max": crit["max"][:2]}
            return lambda P: np.maximum(sdf(shape, P), sdf(box, P))
        if t == "union":
            parts = [domain_level(shape, o, table) for o in crit.get("of", [])]
            if not parts or any(p is None for p in parts):
                return None
            return lambda P: np.min([p(P) for p in parts], axis=0)
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return None


def pieces_of(shape: dict, ref: dict, all_pieces: list[Piece] | None = None) -> list[Piece]:
    """Boundary pieces of a side reference {"shape": [...], "side": ...}."""
    path, name = tuple(ref.get("shape", [])), ref.get("side")
    return [p for p in (all_pieces if all_pieces is not None else pieces(shape))
            if p.side.shape == path and p.side.side == name]


def nearest(all_pieces: list[Piece], point, tolerance: float) -> Piece | None:
    """The piece closest to a point, if closer than tolerance."""
    best = min(all_pieces, key=lambda p: p.distance(point), default=None)
    if best is None or best.distance(point) > tolerance:
        return None
    return best


def side_label(shape: dict, ref: dict) -> str:
    """Readable name of a side reference: 'circle of ball 2', 'xmin of rectangle 1',
    'side 3 of polygon 2'."""
    prim = primitive_at(shape, ref.get("shape", []))
    names = source_paths(shape)
    try:
        index = names.index(tuple(ref.get("shape", []))) + 1
    except ValueError:
        return f"{ref.get('side')} of a missing shape"
    side = str(ref.get("side"))
    for word in ("side", "fillet"):
        if side.startswith(word) and side[len(word):].isdigit():
            side = f"{word} {side[len(word):]}"
    return f"{side} of {prim['type'] if prim else '?'} {index}"


# ------------------------------------------------------------------ selections on the pieces

def outward_normals(shape: dict, piece: Piece, t) -> np.ndarray:
    side = piece.side
    N = side.normals(t)
    eps = 1e-9 * size(shape)
    outside = sdf(shape, side.points(t) + eps * N) > 0
    return np.where(outside[:, None], N, -N)


def _matches(shape, piece: Piece, crit: dict, table: dict, tol: float) -> bool:
    t = crit.get("type")
    ts = np.linspace(piece.t0, piece.t1, 9)
    P = piece.side.points(ts)
    if t == "edges":
        return any(piece.side.shape == tuple(r.get("shape", [])) and piece.side.side == r.get("side")
                   for r in crit.get("edges", []))
    if t == "plane":
        n = np.asarray(crit["normal"], float)
        n = n / (np.linalg.norm(n) or 1.0)
        return bool(np.all(np.abs((P - np.asarray(crit["point"], float)) @ n) < tol))
    if t == "in_box":
        lo, hi = np.asarray(crit["min"], float), np.asarray(crit["max"], float)
        return bool(np.all((P >= lo - tol) & (P <= hi + tol)))
    if t == "in_ball":
        return bool(np.all(np.linalg.norm(P - np.asarray(crit["center"], float), axis=1)
                           <= float(crit["radius"]) + tol))
    if t == "normal":
        d = np.asarray(crit["direction"], float)
        d = d / (np.linalg.norm(d) or 1.0)
        cos = outward_normals(shape, piece, ts) @ d
        return bool(np.all(cos >= math.cos(math.radians(float(crit.get("angle", 1.0))))))
    if t == "union":
        return any(_matches(shape, piece, table[o].get("criteria", {}), table, tol)
                   for o in crit.get("of", []) if o in table)
    return False


def pieces_matching(shape: dict, selection: str, table: dict,
                    all_pieces: list[Piece] | None = None) -> list[Piece] | None:
    """Boundary pieces of a boundary selection (table: model.selection_table),
    or None if it cannot be evaluated on the geometry (e.g. physical groups)."""
    all_pieces = pieces(shape) if all_pieces is None else all_pieces
    sel = table.get(selection)
    if sel is None or sel.get("kind") != "boundary":
        return None
    if selection == "boundary":
        return list(all_pieces)
    crit = sel.get("criteria", {})
    if crit.get("type") not in ("edges", "plane", "in_box", "in_ball", "normal", "union"):
        return None
    tol = 1e-6 * size(shape)
    try:
        return [p for p in all_pieces if _matches(shape, p, crit, table, tol)]
    except (KeyError, TypeError, ValueError):
        return None


def toggle_edge(criteria: dict, ref: dict) -> dict:
    """Criteria after a click on a side: added, or removed if already there
    (another kind of criterion is replaced by the clicked side)."""
    if criteria.get("type") != "edges":
        return {"type": "edges", "edges": [dict(ref)]}
    edges = [dict(r) for r in criteria.get("edges", [])]
    key = (tuple(ref["shape"]), ref["side"])
    kept = [r for r in edges if (tuple(r.get("shape", [])), r.get("side")) != key]
    if len(kept) == len(edges):
        kept.append(dict(ref))
    return {"type": "edges", "edges": kept}


# ------------------------------------------------------------------ drawing

def grid_step(width: float, divisions: int = 40) -> float:
    """A round step (1, 2 or 5 x 10^n) giving about `divisions` steps over width."""
    if width <= 0:
        return 1.0
    raw = width / divisions
    e = math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * 10 ** e >= raw:
            return m * 10.0 ** e
    return 10.0 ** (e + 1)


def next_step(step: float, finer: bool = False) -> float:
    """The next round step (1, 2, 5 x 10^n) above step, or below it."""
    e = math.floor(math.log10(step) + 1e-9)
    seq = [m * 10.0 ** k for k in (e - 1, e, e + 1) for m in (1, 2, 5)]
    if finer:
        return max(s for s in seq if s < step * (1 - 1e-9))
    return min(s for s in seq if s > step * (1 + 1e-9))


def major_step(step: float, width: float, count: int = 8) -> float:
    """A multiple of the grid step (1, 2, 5 x 10^n) for about `count` labels over width."""
    s = step
    while width / s > count:
        s = next_step(s)
    return s


def snap(point, step: float) -> list[float]:
    return [round(round(float(c) / step) * step, 12) for c in point[:2]]


def drawn_shape(kind: str, p0, p1) -> dict | None:
    """The primitive drawn from p0 to p1: a rectangle (two corners), a disk
    (center, point of the circle) or an ellipse (inscribed in the rectangle
    of two corners); None if degenerate."""
    p0, p1 = np.asarray(p0, float)[:2], np.asarray(p1, float)[:2]
    if kind == "rectangle":
        lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
        if np.any(hi - lo <= 0):
            return None
        return {"type": "rectangle", "min": [float(x) for x in lo], "max": [float(x) for x in hi]}
    if kind == "ellipse":
        lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
        if np.any(hi - lo <= 0):
            return None
        return {"type": "ellipse", "center": [float(x) for x in (lo + hi) / 2],
                "radii": [float(x) for x in (hi - lo) / 2], "angle": 0.0}
    if kind == "ball":
        r = float(np.linalg.norm(p1 - p0))
        if r <= 0:
            return None
        return {"type": "ball", "center": [float(x) for x in p0], "radius": r}
    raise ValueError(f"unknown shape {kind!r}")
