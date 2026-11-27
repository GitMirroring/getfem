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

"""Constraints on the vertices of polygons (a sketch).

A polygon may have a list of constraints; sides and vertices are numbered
from 1 (side k goes from vertex k to vertex k + 1, as side1, side2... in
the selections):

    {"type": "horizontal" | "vertical", "side": k}
    {"type": "length", "side": k, "value": L}
    {"type": "angle", "vertex": k, "value": degrees}     the angle between the
                                                         two sides at the vertex
    {"type": "parallel" | "perpendicular" | "equal", "side": k, "other": l}
    {"type": "fixed", "vertex": k, "at": [x, y]}
    {"type": "coincident", "vertex": k, "shape": [path], "other": l}
                     the vertex at the vertex l of another polygon of the same
                     2D tree (path: indices in "of" in that tree)

The values are numbers or expressions of parameters. The vertices given in
the model are where the shape was drawn: when the model is resolved
(parameters.resolved), the polygons of a 2D tree that have constraints are
moved as little as possible (least squares) for their constraints to hold;
a coordinate given by an expression is not moved. The constraints of the
polygons of a tree are solved together (a coincidence links two of them).
"""

from __future__ import annotations

import math

import numpy as np

SIDE_TYPES = ("horizontal", "vertical", "length")
PAIR_TYPES = ("parallel", "perpendicular", "equal")
VERTEX_TYPES = ("angle", "fixed", "coincident")
TYPES = SIDE_TYPES + ("angle",) + PAIR_TYPES + ("fixed", "coincident")
VALUED = ("length", "angle")
LABELS = {"horizontal": "horizontal side", "vertical": "vertical side", "length": "length of a side",
          "angle": "angle at a vertex", "parallel": "parallel sides", "perpendicular": "perpendicular sides",
          "equal": "sides of equal length", "fixed": "fixed vertex",
          "coincident": "vertex on a vertex of another polygon"}
# the marks of the constraints in the view (the fonts of VTK have few symbols)
SYMBOLS = {"horizontal": "H", "vertical": "V", "parallel": "//", "perpendicular": "perp", "equal": "=",
           "fixed": "fixed", "coincident": "on"}


class SketchError(ValueError):
    pass


def references(c: dict) -> list[tuple[str, int]]:
    """The sides and vertices (kind, 1-based index) of the polygon that a
    constraint refers to."""
    t = c.get("type")
    if t in SIDE_TYPES:
        return [("side", c.get("side"))]
    if t in PAIR_TYPES:
        return [("side", c.get("side")), ("side", c.get("other"))]
    return [("vertex", c.get("vertex"))]


def _open_check(c: dict, poly: dict) -> str | None:
    """An open polygon (a curve): no closing side, no angle at its ends."""
    if not poly.get("open"):
        return None
    n = len(poly.get("points") or [])
    for kind, k in references(c):
        if kind == "side" and k == n:
            return f"side {n} does not exist (the polygon is open: {n - 1} sides)"
        if kind == "vertex" and c.get("type") == "angle" and k in (1, n):
            return f"no angle at an end (vertex {k}) of an open polygon"
    return None


def check(c: dict, n: int, tree: dict | None = None) -> str | None:
    """What is wrong with a constraint of a polygon of n vertices (None if
    nothing); tree: the 2D tree, for a coincidence."""
    t = c.get("type") if isinstance(c, dict) else None
    if t not in TYPES:
        return f"unknown constraint {t!r} ({', '.join(TYPES)})"
    for kind, k in references(c):
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n:
            return f"{kind} {k!r}: a {kind} number from 1 to {n} expected"
    if t in PAIR_TYPES and c.get("side") == c.get("other"):
        return "two different sides expected"
    if t in VALUED:
        v = c.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return "value: number expected"
        if t == "length" and v <= 0:
            return "value: a positive length expected"
        if t == "angle" and not 0 < v < 360:
            return "value: an angle between 0 and 360 degrees expected"
    if t == "fixed":
        at = c.get("at")
        if not isinstance(at, list) or len(at) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                          for x in at):
            return "at: 2 numbers expected"
    if t == "coincident" and tree is not None:
        other = node_at(tree, c.get("shape", []))
        if not isinstance(other, dict) or other.get("type") != "polygon":
            return f"shape {c.get('shape')!r}: not a polygon of this tree"
        m = len(other.get("points") or [])
        k = c.get("other")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= m:
            return f"other: a vertex number from 1 to {m} expected"
    return None


def node_at(tree: dict, path) -> dict | None:
    node = tree
    for i in path or []:
        try:
            node = node["of"][i]
        except (KeyError, IndexError, TypeError):
            return None
    return node


def polygons(tree: dict, path: tuple = ()):
    """(path, polygon) of the polygons of a 2D tree (through booleans and
    transforms)."""
    if not isinstance(tree, dict):
        return
    if tree.get("type") == "polygon":
        yield path, tree
    for i, s in enumerate(tree.get("of", []) or []):
        yield from polygons(s, path + (i,))


# ------------------------------------------------------------------ residuals

def _side(P, k):
    n = len(P)
    return P[(k - 1) % n], P[k % n]


def measure(c: dict, P: np.ndarray) -> float | None:
    """The current value of a valued constraint (length, angle in degrees)
    on the vertices P (n x 2); None for the other ones."""
    t = c["type"]
    if t == "length":
        a, b = _side(P, c["side"])
        return float(np.linalg.norm(b - a))
    if t == "angle":
        k, n = c["vertex"] - 1, len(P)
        u, v = P[(k - 1) % n] - P[k], P[(k + 1) % n] - P[k]
        cos = float(u @ v) / ((np.linalg.norm(u) * np.linalg.norm(v)) or 1.0)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return None


def _residuals(c: dict, P: np.ndarray, scale: float, others) -> list[float]:
    t = c["type"]
    if t in ("horizontal", "vertical"):
        a, b = _side(P, c["side"])
        return [float(b[1] - a[1]) if t == "horizontal" else float(b[0] - a[0])]
    if t == "length":
        a, b = _side(P, c["side"])
        return [float(np.linalg.norm(b - a)) - float(c["value"])]
    if t == "angle":
        k, n = c["vertex"] - 1, len(P)
        u, v = P[(k - 1) % n] - P[k], P[(k + 1) % n] - P[k]
        # the angle through (cos, sin): smooth also at 0 and 180 degrees
        nu, nv = (np.linalg.norm(u) or 1.0), (np.linalg.norm(v) or 1.0)
        cos, sin = float(u @ v) / (nu * nv), abs(float(u[0] * v[1] - u[1] * v[0])) / (nu * nv)
        th = math.radians(float(c["value"]))
        return [scale * (cos * math.sin(th) - sin * math.cos(th))]
    if t in PAIR_TYPES:
        a, b = _side(P, c["side"])
        p, q = _side(P, c["other"])
        d, e = b - a, q - p
        if t == "equal":
            return [float(np.linalg.norm(d) - np.linalg.norm(e))]
        norm = (np.linalg.norm(d) * np.linalg.norm(e)) or 1.0
        value = float(d[0] * e[1] - d[1] * e[0]) if t == "parallel" else float(d @ e)
        return [scale * value / norm]
    if t == "fixed":
        x = P[c["vertex"] - 1]
        return [float(x[0] - c["at"][0]), float(x[1] - c["at"][1])]
    if t == "coincident":
        x = P[c["vertex"] - 1]
        y = others(tuple(c["shape"]))[c["other"] - 1]
        return [float(x[0] - y[0]), float(x[1] - y[1])]
    raise SketchError(f"unknown constraint {t!r}")


def solve(polys: dict, fixed: dict | None = None, weights: dict | None = None,
          tol: float = 1e-13, iterations: int = 50) -> dict:
    """Vertices satisfying the constraints, as close as possible to the
    given ones. polys: {key: (points (n x 2), constraints)}; fixed: {key:
    boolean mask (n x 2) of the coordinates that must not move}; weights:
    {key: weights of the vertices (n)} for the distance (a dragged vertex
    weighs more). Returns {key: points}; raises SketchError if the
    constraints cannot hold together."""
    keys = list(polys)
    X0 = {k: np.asarray(polys[k][0], float).reshape(-1, 2) for k in keys}
    sizes = [len(X0[k]) for k in keys]
    offsets = np.cumsum([0] + [2 * s for s in sizes])
    x0 = np.concatenate([X0[k].ravel() for k in keys]) if keys else np.zeros(0)
    allp = np.concatenate([X0[k] for k in keys]) if keys else np.zeros((0, 2))
    scale = float(np.max(np.ptp(allp, axis=0))) if len(allp) else 1.0
    scale = scale or 1.0
    free = np.ones(len(x0), bool)
    w = np.ones(len(x0))
    for i, k in enumerate(keys):
        if fixed and k in fixed:
            free[offsets[i]:offsets[i + 1]] = ~np.asarray(fixed[k], bool).ravel()
        if weights and k in weights:
            w[offsets[i]:offsets[i + 1]] = np.repeat(np.asarray(weights[k], float), 2)

    def unpack(x):
        return {k: x[offsets[i]:offsets[i + 1]].reshape(-1, 2) for i, k in enumerate(keys)}

    def g(x):
        Ps = unpack(x)
        out = []
        for k in keys:
            for c in polys[k][1]:
                out += _residuals(c, Ps[k], scale, lambda q: Ps[q])
        return np.array(out, float)

    if not len(g(x0)):
        return unpack(x0)
    x = x0.copy()
    idx = np.flatnonzero(free)
    Winv = 1.0 / w[idx]
    h = 1e-7 * scale
    for _ in range(iterations):
        r = g(x)
        if np.max(np.abs(r)) <= tol * scale:
            return unpack(x)
        J = np.empty((len(r), len(idx)))
        for j, i in enumerate(idx):
            e = np.zeros(len(x))
            e[i] = h
            J[:, j] = (g(x + e) - g(x - e)) / (2 * h)
        # the smallest weighted move from x0 that satisfies the linearized
        # constraints: dx = (x0 - x) + W^-1 J^T lambda
        d0 = (x0 - x)[idx]
        A = (J * Winv) @ J.T
        lam = np.linalg.lstsq(A, -r - J @ d0, rcond=1e-12)[0]
        dx = d0 + Winv * (J.T @ lam)
        # a damped step when the constraints are far from holding
        step = 1.0
        n0 = np.linalg.norm(r)
        for _ in range(20):
            xt = x.copy()
            xt[idx] += step * dx
            if np.linalg.norm(g(xt)) < n0 or step < 1e-3:
                break
            step /= 2
        x = xt
    r = g(x)
    if np.max(np.abs(r)) > 1e-7 * scale:
        raise SketchError("the constraints cannot hold together (contradictory, or too far from the drawing)")
    return unpack(x)


def _where(path) -> str:
    return "".join(f".of[{i}]" for i in path)


def solve_tree(tree: dict, weights: dict | None = None) -> list[str]:
    """Move the vertices of the constrained polygons of a resolved 2D tree
    (in place) for their constraints to hold. The coordinates that were
    given by expressions are in the "_fixed" mask of each polygon (set by
    parameters.resolved). weights: {path: weights of its vertices}.
    Returns the errors, each starting with the path of its polygon in the
    tree (".of[1].constraints[0]: ...")."""
    if not isinstance(tree, dict):
        return []
    polys, fixed, errors = {}, {}, []
    linked = set()
    for path, poly in polygons(tree):
        cons = poly.get("constraints") or []
        if not cons:
            continue
        pts = poly.get("points") or []
        if len(pts) < (2 if poly.get("open") else 3) \
                or any(not isinstance(v, (int, float)) or isinstance(v, bool) for p in pts for v in p):
            continue
        bad = [f"{_where(path)}.constraints[{j}]: {msg}" for j, c in enumerate(cons)
               if (msg := check(c, len(pts), tree) or _open_check(c, poly)) is not None]
        if bad:
            errors += bad
            continue
        polys[path] = (np.asarray(pts, float), cons)
        fixed[path] = np.asarray(poly.get("_fixed", np.zeros((len(pts), 2), bool)), bool)
        linked |= {tuple(c["shape"]) for c in cons if c["type"] == "coincident"}
    for q in linked:                  # a polygon a vertex is put on: in the system (unmoved if no constraint)
        if q not in polys:
            pts = np.asarray(node_at(tree, q).get("points"), float)
            polys[q] = (pts, [])
            fixed[q] = np.ones(pts.shape, bool)
    if not polys:
        return errors
    try:
        out = solve(polys, fixed, weights)
    except SketchError as exc:
        return errors + [f"{_where(sorted(polys)[0])}.constraints: {exc}"]
    for path, P in out.items():
        node = node_at(tree, path)
        follow_splines(node, polys[path][0], P)
        node["points"] = [[float(x), float(y)] for x, y in P]
    return errors


def chord_map(a0, b0, a1, b1):
    """The similarity of the plane (rotation, scaling, translation) that
    takes the segment a0 b0 to a1 b1 (a translation if a0 = b0), as a
    function of points (n x 2)."""
    za0, zb0, za1, zb1 = (complex(float(p[0]), float(p[1])) for p in (a0, b0, a1, b1))
    r = (zb1 - za1) / (zb0 - za0) if abs(zb0 - za0) > 0 else 1.0

    def f(Q):
        z = za1 + r * (np.asarray(Q, float) @ np.array([1, 1j]) - za0)
        return np.column_stack([z.real, z.imag])
    return f


def follow_splines(poly: dict, old, new) -> None:
    """The points of the splines of a polygon whose vertices go from old to
    new: each moved by the similarity that takes the old chord of its side
    to the new one (its shape kept relative to its ends)."""
    n = len(old)
    for sp in poly.get("splines") or []:
        k = sp.get("side") if isinstance(sp, dict) else None
        pts = sp.get("points") if isinstance(sp, dict) else None
        if not isinstance(k, int) or not 1 <= k <= n or not pts:
            continue
        try:
            Q = np.asarray(pts, float).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        f = chord_map(old[k - 1], old[k % n], new[k - 1], new[k % n])
        sp["points"] = f(Q).tolist()


def label(c: dict, value_text: str | None = None) -> str:
    """'length of side 2 = L', 'sides 1 and 3 parallel'..."""
    t = c.get("type")
    v = value_text if value_text is not None else str(c.get("value"))
    if t in ("horizontal", "vertical"):
        return f"side {c.get('side')} {t}"
    if t == "length":
        return f"length of side {c.get('side')} = {v}"
    if t == "angle":
        return f"angle at vertex {c.get('vertex')} = {v}°"
    if t in PAIR_TYPES:
        what = {"parallel": "parallel", "perpendicular": "perpendicular", "equal": "of equal length"}[t]
        return f"sides {c.get('side')} and {c.get('other')} {what}"
    if t == "fixed":
        return f"vertex {c.get('vertex')} fixed"
    if t == "coincident":
        return f"vertex {c.get('vertex')} on vertex {c.get('other')} of another polygon"
    return str(t)
