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

RECTANGLE_SIDES = ("xmin", "xmax", "ymin", "ymax")
BOOLEANS = ("union", "intersection", "difference")


# ------------------------------------------------------------------ shapes

def shape_of(source: dict) -> dict | None:
    """The CSG tree of a 2D box or csg source (None for other sources)."""
    t = source.get("type")
    if t == "box" and len(source.get("size", [])) == 2:
        o, s = source["origin"], source["size"]
        return {"type": "rectangle", "min": list(o), "max": [o[0] + s[0], o[1] + s[1]]}
    if t == "csg" and source.get("dim") == 2:
        return source.get("shape")
    return None


def primitive_at(shape: dict, path) -> dict | None:
    for i in path:
        try:
            shape = shape["of"][i]
        except (KeyError, IndexError, TypeError):
            return None
    return shape


def primitives(shape: dict, path: tuple = ()):
    """(path, primitive) of the leaves of a CSG tree; path = indices in "of"."""
    if shape.get("type") in BOOLEANS:
        for i, s in enumerate(shape.get("of", [])):
            yield from primitives(s, path + (i,))
    else:
        yield path, shape


def sdf(shape: dict, P: np.ndarray) -> np.ndarray:
    """Signed distance like function (negative inside) at the points P (n x 2)."""
    t = shape["type"]
    if t == "rectangle":
        lo, hi = np.asarray(shape["min"], float), np.asarray(shape["max"], float)
        return np.max(np.maximum(lo - P, P - hi), axis=1)
    if t == "ball":
        return np.linalg.norm(P - np.asarray(shape["center"], float), axis=1) - float(shape["radius"])
    if t == "polygon":
        return polygon_sdf(np.asarray(shape["points"], float), P)
    if t == "ellipse":
        return ellipse_sdf(shape["center"], shape["radii"], float(shape.get("angle", 0.0)), P)
    parts = [sdf(s, P) for s in shape["of"]]
    if t == "union":
        return np.min(parts, axis=0)
    if t == "intersection":
        return np.max(parts, axis=0)
    if t == "difference":
        return np.maximum(parts[0], -np.min(parts[1:], axis=0))
    raise ValueError(f"shape {t!r} is not 2D")


def bounds(shape: dict) -> tuple[np.ndarray, np.ndarray]:
    t = shape["type"]
    if t == "rectangle":
        return np.asarray(shape["min"], float), np.asarray(shape["max"], float)
    if t == "ball":
        c, r = np.asarray(shape["center"], float), float(shape["radius"])
        return c - r, c + r
    if t == "polygon":
        V = np.asarray(shape["points"], float)
        return V.min(axis=0), V.max(axis=0)
    if t == "ellipse":
        c = np.asarray(shape["center"], float)
        (r1, r2), th = (float(r) for r in shape["radii"]), math.radians(float(shape.get("angle", 0.0)))
        e = np.array([math.hypot(r1 * math.cos(th), r2 * math.sin(th)),
                      math.hypot(r1 * math.sin(th), r2 * math.cos(th))])
        return c - e, c + e
    b = [bounds(s) for s in shape["of"]]
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


def size(shape: dict) -> float:
    lo, hi = bounds(shape)
    return float(np.linalg.norm(hi - lo)) or 1.0


# ------------------------------------------------------------------ sides and pieces

@dataclass(frozen=True)
class Side:
    """A side of a primitive, parametrized by t in [0, 1]."""
    shape: tuple            # path of the primitive
    side: str               # xmin, xmax, ymin, ymax (rectangle) or circle (disk)
    kind: str               # "segment", "arc" or "ellipse" (closed curves: t = 0 and 1 meet)
    a: tuple                # segment: first point; arc, ellipse: center
    b: tuple                # segment: second point; arc: (radius, 0); ellipse: (r1, r2, angle in radians)

    @property
    def ref(self) -> dict:
        return {"shape": list(self.shape), "side": self.side}

    @property
    def closed(self) -> bool:
        return self.kind != "segment"

    def length(self, t0: float = 0.0, t1: float = 1.0) -> float:
        if self.kind == "segment":
            return float(np.linalg.norm(np.subtract(self.b, self.a))) * (t1 - t0)
        if self.kind == "arc":
            return 2 * math.pi * self.b[0] * (t1 - t0)
        P = self.points(np.linspace(t0, t1, 2049))
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    def points(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        if self.kind == "segment":
            a, b = np.asarray(self.a), np.asarray(self.b)
            return a + t[:, None] * (b - a)
        if self.kind == "ellipse":
            (r1, r2, th), c = self.b, np.asarray(self.a)
            u, v = r1 * np.cos(2 * math.pi * t), r2 * np.sin(2 * math.pi * t)
            return c + np.column_stack([u * math.cos(th) - v * math.sin(th), u * math.sin(th) + v * math.cos(th)])
        c, r = np.asarray(self.a), self.b[0]
        th = 2 * math.pi * t
        return c + r * np.column_stack([np.cos(th), np.sin(th)])

    def normals(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        if self.kind == "segment":
            d = np.asarray(self.b) - np.asarray(self.a)
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)
            return np.tile(n, (len(t), 1))
        if self.kind == "ellipse":
            r1, r2, th = self.b
            nu, nv = np.cos(2 * math.pi * t) / r1, np.sin(2 * math.pi * t) / r2
            N = np.column_stack([nu * math.cos(th) - nv * math.sin(th), nu * math.sin(th) + nv * math.cos(th)])
            return N / np.linalg.norm(N, axis=1)[:, None]
        th = 2 * math.pi * t
        return np.column_stack([np.cos(th), np.sin(th)])


def side_names(prim: dict) -> tuple:
    """Names of the sides of a primitive."""
    t = prim.get("type")
    if t == "rectangle":
        return RECTANGLE_SIDES
    if t == "polygon":
        return tuple(f"side{i + 1}" for i in range(len(prim.get("points", []))))
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
            out += [Side(path, f"side{i + 1}", "segment", V[i], V[(i + 1) % len(V)]) for i in range(len(V))]
    return out


@dataclass(frozen=True)
class Piece:
    """The part t0 <= t <= t1 of a side that lies on the boundary of the domain
    (for a circle, t1 may exceed 1: the piece goes through the angle 0)."""
    side: Side
    t0: float
    t1: float

    def points(self, n: int = 64) -> np.ndarray:
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

    def refine(i):              # transition between samples i and i + 1
        lo, hi = t[i], t[i + 1]
        state = on[i]
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if test(np.array([mid]))[0] == state:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    ends = [refine(i) for i in changes]
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
    eps = 1e-9 * size(shape)
    level = level or (lambda P: sdf(shape, P))
    out = []
    for side in sides(shape):
        out += _runs(side, lambda t, side=side: _separates(level, side, t, eps), samples)
    return out


def interfaces(shape: dict, samples: int = 720) -> list[Piece]:
    """The parts of the sides of the primitives that lie inside the domain:
    the interfaces between its regions (e.g. the circle of an inclusion)."""
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
    return [p for p, _ in primitives(shape)]


def region_at(shape: dict, point) -> list[list[int]] | None:
    """The region of a point of the domain: the paths of the primitives that
    contain it (None outside the domain). A region is stored this way, e.g.
    [[0], [1]] for "in rectangle 1 and in disk 2" (and in no other primitive)."""
    P = np.asarray(point, float)[None, :2]
    if sdf(shape, P)[0] >= 0:
        return None
    return [list(path) for path, prim in primitives(shape) if sdf(prim, P)[0] < 0]


def region_level(shape: dict, regions: list) -> callable:
    """Function of the points (n x 2), negative exactly in the given regions
    (each one a list of paths, see region_at)."""
    prims = list(primitives(shape))

    def level(P):
        d = sdf(shape, P)
        each = {path: sdf(prim, P) for path, prim in prims}
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
    """'rectangle 1 + ball 2': the primitives containing the region."""
    names = _paths(shape)
    parts = []
    for p in region:
        prim = primitive_at(shape, p)
        if tuple(p) in names and prim is not None:
            parts.append(f"{prim['type']} {names.index(tuple(p)) + 1}")
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
    names = [p for p, _ in primitives(shape)]
    try:
        index = names.index(tuple(ref.get("shape", []))) + 1
    except ValueError:
        return f"{ref.get('side')} of a missing shape"
    side = str(ref.get("side"))
    if side.startswith("side") and side[4:].isdigit():
        side = f"side {side[4:]}"
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
