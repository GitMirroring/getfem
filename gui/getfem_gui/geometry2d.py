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


"""Exact description of 2D geometries (box and CSG), without meshing.

The boundary of a CSG geometry is made of parts of the sides of its
primitives (the four sides of a rectangle, the circle of a disk). Each side
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
    b = [bounds(s) for s in shape["of"]]
    if t == "union":
        return np.min([x[0] for x in b], axis=0), np.max([x[1] for x in b], axis=0)
    if t == "intersection":
        return np.max([x[0] for x in b], axis=0), np.min([x[1] for x in b], axis=0)
    return b[0]


def size(shape: dict) -> float:
    lo, hi = bounds(shape)
    return float(np.linalg.norm(hi - lo)) or 1.0


# ------------------------------------------------------------------ sides and pieces

@dataclass(frozen=True)
class Side:
    """A side of a primitive, parametrized by t in [0, 1]."""
    shape: tuple            # path of the primitive
    side: str               # xmin, xmax, ymin, ymax (rectangle) or circle (disk)
    kind: str               # "segment" or "arc"
    a: tuple                # segment: first point; arc: center
    b: tuple                # segment: second point; arc: (radius, 0)

    @property
    def ref(self) -> dict:
        return {"shape": list(self.shape), "side": self.side}

    def points(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        if self.kind == "segment":
            a, b = np.asarray(self.a), np.asarray(self.b)
            return a + t[:, None] * (b - a)
        c, r = np.asarray(self.a), self.b[0]
        th = 2 * math.pi * t
        return c + r * np.column_stack([np.cos(th), np.sin(th)])

    def normals(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        if self.kind == "segment":
            d = np.asarray(self.b) - np.asarray(self.a)
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)
            return np.tile(n, (len(t), 1))
        th = 2 * math.pi * t
        return np.column_stack([np.cos(th), np.sin(th)])


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
    return out


@dataclass(frozen=True)
class Piece:
    """The part t0 <= t <= t1 of a side that lies on the boundary of the domain
    (for a circle, t1 may exceed 1: the piece goes through the angle 0)."""
    side: Side
    t0: float
    t1: float

    def points(self, n: int = 64) -> np.ndarray:
        m = max(2, int(n * (self.t1 - self.t0)) + 2) if self.side.kind == "arc" else 2
        return self.side.points(np.linspace(self.t0, self.t1, m))

    def criterion(self) -> dict:
        """Concrete description, used to find the faces of a mesh."""
        if self.side.kind == "segment":
            p = self.side.points([self.t0, self.t1])
            return {"segment": [p[0].tolist(), p[1].tolist()]}
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


def _on_boundary(shape, side: Side, t, eps) -> np.ndarray:
    """True where the side separates the inside from the outside."""
    P, N = side.points(t), side.normals(t)
    return (sdf(shape, P + eps * N) < 0) != (sdf(shape, P - eps * N) < 0)


def pieces(shape: dict, samples: int = 720) -> list[Piece]:
    """The boundary of the domain, as parts of the sides of the primitives.

    A point of a side is on the boundary if it separates inside and outside,
    tested at a distance eps on each side; where two sides are tangent, this
    overestimates a piece by about sqrt(2 eps), hence a small eps."""
    eps = 1e-9 * size(shape)
    out = []
    for side in sides(shape):
        t = (np.arange(samples) + 0.5) / samples
        on = _on_boundary(shape, side, t, eps)
        if not on.any():
            continue
        if on.all():
            out.append(Piece(side, 0.0, 1.0))
            continue
        # runs of boundary samples; their ends are refined by bisection
        changes = np.flatnonzero(on[1:] != on[:-1])

        def refine(i):              # transition between samples i and i + 1
            lo, hi = t[i], t[i + 1]
            state = on[i]
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if _on_boundary(shape, side, [mid], eps)[0] == state:
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
        if side.kind == "arc" and on[0] and on[-1] and len(runs) > 1:   # through angle 0
            (s0, e0), (sl, el) = runs[0], runs[-1]
            runs = [(sl, 1.0 + e0)] + runs[1:-1]
        out += [Piece(side, a, b) for a, b in runs if b > a]
    return out


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
    """Readable name of a side reference: 'circle of ball 2', 'xmin of rectangle 1'."""
    prim = primitive_at(shape, ref.get("shape", []))
    names = [p for p, _ in primitives(shape)]
    try:
        index = names.index(tuple(ref.get("shape", []))) + 1
    except ValueError:
        return f"{ref.get('side')} of a missing shape"
    return f"{ref.get('side')} of {prim['type'] if prim else '?'} {index}"


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
