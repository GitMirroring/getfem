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

"""Curves and surfaces: geometries of a lower dimension than their space
(beams, cables, shells, membranes).

Sources:

- {"type": "curve", "dim": 2, "shape": <polygon>}: a 2D curve drawn as a
  polygon, open ("open": true: from its first vertex to its last one) or
  closed, with its fillets (arcs of circle), splines (cubic Bezier curves)
  and constraints, as the polygons of the 2D geometries.
- {"type": "curve", "dim": 3, "shape": <polygon>, "origin": [x, y, z],
  "axes": [e1, e2], "height": h}: a curve drawn as a polygon in local
  coordinates (x, y), placed in 3D as the flat domain of a surface: at
  origin + x e1 + y e2 + h(x, y) n (a plane curve for a constant height).
- {"type": "curve", "dim": 2 or 3, "parametric": {"x": ..., "y": ...[, "z":
  ...]}, "t": [t0, t1]}: a parametric curve, its coordinates expressions of
  t and the parameters (a helix: "cos(t)", "sin(t)", "0.1*t"); closed when
  its two ends meet.
- {"type": "curve", "dim": 2 or 3, "points": [[...], ...], "through": [...],
  "closed": false}: a chain of segments and arcs of circle; through[i] is
  null (the segment from points[i] to points[i+1]) or a point the arc goes
  through. Closed: the last point is joined to the first one.
- {"type": "surface", "shape": <2D CSG shape>, "origin": [x, y, z],
  "axes": [e1, e2], "height": h}: a flat domain (the 2D shape, in local
  coordinates x, y) placed in 3D at origin + x e1 + y e2 + h(x, y) n, n the
  unit normal e1 x e2; the height h is a number or an expression of x, y
  and the parameters. The selections are defined on the flat domain (local
  coordinates), as for a 2D geometry.
- {"type": "surface", "shape": <2D CSG shape>, "parametric": {"x": ..., "y":
  ..., "z": ...}}: a parametric surface, its points (x, y, z) expressions of
  the coordinates (u, v) of the flat domain (the 2D shape) and of the
  parameters (a cylinder: "cos(u)", "sin(u)", "v" on [0, 2 pi] x [0, 1]);
  the points of its edges that meet are merged (the seam of a cylinder, of
  a torus), the map must not degenerate (no pole).
- {"type": "swept_surface", "sweep": "extrusion" or "revolution", "shape"
  (an open or closed polygon, as a 2D curve) or "points", "through",
  "closed" (a 2D chain, as a curve), "z": [z0, z1] (extrusion
  along z of a curve of the (x, y) plane) or "angle": [a0, a1] (revolution
  about the z axis, in degrees, of a curve of the (r, z) half plane, which
  must not touch the axis)}.

Their meshes are built by the meshing script: a curve is divided into
elements of about the element size (the nodes of an order 2 or 3 element
on the arcs); a surface is the mesh of its flat domain (by any 2D mesher)
mapped to 3D; a swept surface is a 1D mesh of its curve extruded or turned
in layers (quadrangles, gf.Mesh('prismatic')). The functions marked so are
copied into the scripts: they use numpy and math only.
"""

from __future__ import annotations

import ast
import inspect
import math

import numpy as np

SOURCES = ("curve", "surface", "swept_surface")
TOPOLOGICAL_DIM = {"curve": 1, "surface": 2, "swept_surface": 2, "boundary": 2}


def space_dim(src: dict) -> int | None:
    t = src.get("type")
    if t == "curve":
        return src.get("dim")
    if t in ("surface", "swept_surface"):
        return 3
    return None


def chain_dim(src: dict) -> int:
    """Dimension of the points of the chain of a curve or a swept surface."""
    return src.get("dim", 2) if src.get("type") == "curve" else 2


# ------------------------------------------------------------- the chain

def curve_polygon(src: dict) -> dict | None:
    """The polygon of a curve or of the curve of a swept surface drawn as a
    polygon (None for a chain of points)."""
    shape = src.get("shape")
    return shape if src.get("type") in ("curve", "swept_surface") and isinstance(shape, dict) else None


def is_placed(src: dict) -> bool:
    """A 3D curve drawn as a polygon in its plane (origin, axes, height)."""
    return src.get("type") == "curve" and curve_polygon(src) is not None and src.get("dim") == 3


def is_parametric(src: dict) -> bool:
    return src.get("type") == "curve" and isinstance(src.get("parametric"), dict)


AXES = ("x", "y", "z")


def param_code(expr, values: dict, var: str = "t") -> str:
    """Python (numpy) expression of a coordinate of a parametric curve, an
    array function of t (the parameters replaced by their values)."""
    code = height_code(expr, values, (var,))
    return code.replace("0 * x)", f"0 * {var})")


def param_points(src: dict, values: dict, t) -> np.ndarray:
    """Points (dim x n) of a parametric curve at the values t."""
    t = np.asarray(t, float)
    par = src["parametric"]
    out = []
    for k in AXES[:int(src.get("dim", 3))]:
        out.append(eval(param_code(par.get(k, 0.0), values), {"np": np, "__builtins__": {}},  # noqa: S307
                        {"t": t}) + 0.0 * t)
    return np.vstack(out)


def _param_samples(src: dict, values: dict, n: int = 2001):
    t0, t1 = (float(v) for v in src.get("t", [0.0, 1.0]))
    t = np.linspace(t0, t1, n)
    return t, param_points(src, values, t)


def param_closed(src: dict, values: dict) -> bool:
    _, X = _param_samples(src, values, 401)
    scale = float(np.ptp(X, axis=1).max()) or 1.0
    return float(np.linalg.norm(X[:, -1] - X[:, 0])) <= 1e-9 * scale


def is_closed(src: dict, values: dict | None = None) -> bool:
    poly = curve_polygon(src)
    if poly is not None:
        return not poly.get("open")
    if is_parametric(src):
        try:
            return param_closed(src, values or {})
        except (ValueError, TypeError, KeyError, SyntaxError, ZeroDivisionError):
            return False
    return bool(src.get("closed"))


def map_curve(P2: np.ndarray, src: dict, values: dict | None = None) -> np.ndarray:
    """Points of a placed curve (3 x n) for points of its polygon (2 x n)."""
    return map_surface(P2, src, values or {})


def _polygon_pieces(poly: dict) -> list:
    """The pieces of the polygon of a curve: its sides (segments, splines as
    ["bezier", P0, P1, P2, P3] each), its fillets (arcs)."""
    from . import geometry2d as G
    out = []
    for part in G.outline_parts(poly):
        if part[0] == "line":
            out.append(["segment", list(map(float, part[1])), list(map(float, part[2]))])
        elif part[0] == "arc":                        # the smaller arc from a to b about c: through its middle
            a, b, c = (np.asarray(x, float) for x in part[1:4])
            m = 0.5 * (a + b) - c
            r = float(np.linalg.norm(a - c))
            t = c + r * m / (np.linalg.norm(m) or 1.0)
            out.append(["arc", a.tolist(), t.tolist(), b.tolist()])
        else:
            out.append(["bezier"] + [list(map(float, q)) for q in part[1]])
    return out


def chain_pieces(src: dict) -> list:
    """The pieces of a chain: ["segment", A, B], ["arc", A, T, B] (the arc
    from A to B through T) or ["bezier", A, C1, C2, B] (a cubic Bezier
    curve); numbers, as lists."""
    poly = curve_polygon(src)
    if poly is not None:
        return _polygon_pieces(poly)
    pts = [list(map(float, p)) for p in src.get("points", [])]
    through = list(src.get("through") or [])
    n = len(pts) if src.get("closed") else len(pts) - 1
    out = []
    for i in range(max(n, 0)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        t = through[i] if i < len(through) else None
        out.append(["arc", a, list(map(float, t)), b] if t is not None else ["segment", a, b])
    return out


def chain_problem(src: dict) -> str | None:
    """Why the chain of a curve or a swept surface is not valid (None if it is)."""
    poly = curve_polygon(src)
    if poly is not None:                         # (its vertices checked as those of a polygon)
        from . import geometry2d as G
        return G.fillet_problem(poly) or G.spline_problem(poly)
    pts = np.asarray(src.get("points", []), float)
    if len(pts) < 2:
        return "at least two points expected"
    if src.get("closed") and len(pts) < 3 and not any(t is not None for t in (src.get("through") or [])):
        return "a closed chain needs three points (or an arc)"
    scale = float(np.ptp(pts, axis=0).max()) or 1.0
    for k, piece in enumerate(chain_pieces(src)):
        a, b = np.asarray(piece[1]), np.asarray(piece[-1])
        if np.linalg.norm(b - a) <= 1e-12 * scale:
            return f"piece {k + 1}: its two ends coincide"
        if piece[0] == "arc":
            t = np.asarray(piece[2])
            u, v = t - a, b - a
            cross = np.linalg.norm(np.cross(np.append(u, 0)[:3], np.append(v, 0)[:3]))
            if cross <= 1e-9 * np.linalg.norm(u) * np.linalg.norm(v):
                return f"piece {k + 1}: the point of the arc is aligned with its ends"
    return None


def arc_frame(a, t, b):
    """(center, radius, e1, e2, angle) of the arc from a to b through t, in
    2D or 3D: the arc is center + radius (cos s e1 + sin s e2), s in
    [0, angle] (marked: copied into the scripts)."""
    d = len(a)
    A, T, B = (np.append(np.asarray(p, float), [0.0] * (3 - d)) for p in (a, t, b))
    u, v = T - A, B - A
    w = np.cross(u, v)
    ww = float(w @ w)
    C = A + (np.cross(w, u) * float(v @ v) + np.cross(v, w) * float(u @ u)) / (2.0 * ww)
    r = float(np.linalg.norm(A - C))
    e1 = (A - C) / r
    n = w / math.sqrt(ww)
    e2 = np.cross(n, e1)
    ang = lambda P: math.atan2(float((P - C) @ e2), float((P - C) @ e1)) % (2 * math.pi)  # noqa: E731
    angle = ang(B)
    if angle <= 1e-14:
        angle = 2 * math.pi
    return C[:d], r, e1[:d], e2[:d], angle


def bezier_by_length(C, h, order):
    """Points of the cubic Bezier curve of control points C (4 x dim) at
    lengths along it equally spaced, about h apart (order + 1 points per
    element): dim x (n order + 1) (marked: copied into the scripts)."""
    C = np.asarray(C, float)
    u = np.linspace(0.0, 1.0, 2049)
    w = np.stack([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3])
    P = w.T @ C
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    n = max(1, int(math.ceil(s[-1] / h)))
    t = np.interp(np.linspace(0.0, s[-1], n * order + 1), s, u)
    w = np.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3])
    return (w.T @ C).T, n


def curve_elements(f, t0, t1, h, order, closed):
    """Nodes of the elements of the parametric curve t -> f(t) (dim x n for
    an array t), from t0 to t1, equally spaced along its length, about h
    apart: dim x (order + 1) x number of elements (marked: copied into the
    scripts)."""
    t = np.linspace(t0, t1, 8001)
    P = f(t)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=1), axis=0))])
    n = max(3 if closed else 1, int(math.ceil(s[-1] / h)))
    X = f(np.interp(np.linspace(0.0, s[-1], n * order + 1), s, t))
    if closed:
        X[:, -1] = X[:, 0]                      # the same node: the curve closed
    return np.stack([X[:, k * order:k * order + order + 1] for k in range(n)], axis=2)


def chain_elements(pieces, h, order):
    """Nodes of the elements of a chain (GetFEM order of GT_PK(1, order):
    equally spaced from the first to the last one; along a Bezier curve, by
    its length): an array (dim x (order + 1) x number of elements) (marked:
    copied into the scripts)."""
    out = []
    for piece in pieces:
        if piece[0] == "segment":
            A, B = np.asarray(piece[1], float), np.asarray(piece[2], float)
            n = max(1, int(math.ceil(float(np.linalg.norm(B - A)) / h)))
            s = np.linspace(0.0, 1.0, n * order + 1)
            X = A[:, None] + (B - A)[:, None] * s[None, :]
        elif piece[0] == "bezier":
            X, n = bezier_by_length(piece[1:], h, order)
            X[:, -1] = np.asarray(piece[4], float)
        else:
            C, r, e1, e2, angle = arc_frame(piece[1], piece[2], piece[3])
            n = max(1, int(math.ceil(r * angle / h)))
            s = np.linspace(0.0, angle, n * order + 1)
            X = C[:, None] + r * (e1[:, None] * np.cos(s)[None, :] + e2[:, None] * np.sin(s)[None, :])
            X[:, -1] = np.asarray(piece[3], float)          # the end exactly
        X[:, 0] = np.asarray(piece[1], float)
        out.append(np.stack([X[:, k * order:k * order + order + 1] for k in range(n)], axis=2))
    return np.concatenate(out, axis=2)


def chain_polyline(src: dict, n_arc: int = 64, values: dict | None = None) -> np.ndarray:
    """Points along the chain (for the preview): an array (number x dim);
    a placed curve, in 3D (values: those of the parameters, for its
    height); a parametric curve, sampled."""
    if is_parametric(src):
        return _param_samples(src, values or {}, 401)[1].T
    if is_placed(src):
        P2 = chain_polyline(dict(src, dim=2), n_arc)
        return map_curve(P2.T, src, values).T
    pts = []
    for piece in chain_pieces(src):
        if piece[0] == "segment":
            seg = np.array([piece[1], piece[2]], float)
        elif piece[0] == "bezier":
            C = np.asarray(piece[1:], float)
            u = np.linspace(0.0, 1.0, 33)[:, None]
            seg = (1 - u) ** 3 * C[0] + 3 * u * (1 - u) ** 2 * C[1] + 3 * u ** 2 * (1 - u) * C[2] + u ** 3 * C[3]
        else:
            C, r, e1, e2, angle = arc_frame(piece[1], piece[2], piece[3])
            s = np.linspace(0.0, angle, max(8, int(n_arc * angle / (2 * math.pi)) + 2))
            seg = C[None, :] + r * (np.cos(s)[:, None] * e1[None, :] + np.sin(s)[:, None] * e2[None, :])
        pts.append(seg if not pts else seg[1:])
    return np.concatenate(pts) if pts else np.zeros((0, chain_dim(src)))


def chain_ends(src: dict, values: dict | None = None) -> list:
    """The two ends of an open chain (no end for a closed one); in 3D for a
    placed curve (values: those of the parameters, for its height)."""
    if is_parametric(src):
        if is_closed(src, values):
            return []
        t0, t1 = (float(v) for v in src.get("t", [0.0, 1.0]))
        X = param_points(src, values or {}, [t0, t1])
        return [X[:, 0].tolist(), X[:, 1].tolist()]
    if is_placed(src):
        ends = chain_ends(dict(src, dim=2))
        return [] if not ends else map_curve(np.asarray(ends, float).T, src, values).T.tolist()
    poly = curve_polygon(src)
    pts = (poly or {}).get("points", []) if poly is not None else src.get("points", [])
    if is_closed(src) or len(pts) < 2:
        return []
    return [list(map(float, pts[0])), list(map(float, pts[-1]))]


# ------------------------------------------------------------- surfaces

NUMPY_FUNCTIONS = {
    "sqr": "np.square", "sqrt": "np.sqrt", "pow": "np.power", "exp": "np.exp", "log": "np.log",
    "log10": "np.log10", "sin": "np.sin", "cos": "np.cos", "tan": "np.tan", "asin": "np.arcsin",
    "acos": "np.arccos", "atan": "np.arctan", "atan2": "np.arctan2", "sinh": "np.sinh", "cosh": "np.cosh",
    "tanh": "np.tanh", "abs": "np.abs", "min": "np.minimum", "max": "np.maximum",
}


def height_code(height, values: dict, variables: tuple = ("x", "y")) -> str:
    """Python (numpy) expression of the height of a surface, a function of
    the arrays x and y (variables): the parameters replaced by their values,
    the functions by those of numpy."""
    if isinstance(height, (int, float)) and not isinstance(height, bool):
        return f"({float(height)!r} + 0 * x)"

    class Sub(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in variables:
                return node
            if node.id == "pi":
                return ast.copy_location(ast.Constant(math.pi), node)
            if node.id in values:
                if np.shape(values[node.id]):
                    raise ValueError(f"{node.id} is not a number: use its components, {node.id}(1)...")
                return ast.copy_location(ast.Constant(float(values[node.id])), node)
            raise ValueError(f"unknown name {node.id!r} in the "
                             + ("height" if variables == ("x", "y") else "expression"))

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in values:     # a component: C(1)
                from . import parameters as P
                value = P.evaluate(ast.unparse(node), values)
                if np.shape(value):
                    raise ValueError(f"{ast.unparse(node)} is not a number")
                return ast.copy_location(ast.Constant(float(value)), node)
            if not isinstance(node.func, ast.Name) or node.func.id not in NUMPY_FUNCTIONS:
                raise ValueError(f"unknown function {ast.unparse(node.func)!r} in the "
                                 + ("height" if variables == ("x", "y") else "expression"))
            node.args = [self.visit(a) for a in node.args]
            node.func = ast.parse(NUMPY_FUNCTIONS[node.func.id], mode="eval").body
            return node

    tree = ast.parse(str(height).strip(), mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            raise ValueError("'**' is not available: use pow(a, b) or sqr(a)")
    expr = ast.unparse(Sub().visit(tree).body)
    return f"(({expr}) + 0 * x)"


def frame(src: dict):
    """(origin, e1, e2, n) of a surface: e1 unit, e2 orthonormalized, n = e1 x e2."""
    o = np.asarray(src.get("origin", [0.0, 0.0, 0.0]), float)
    axes = src.get("axes") or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    e1, e2 = np.asarray(axes[0], float), np.asarray(axes[1], float)
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 - (e2 @ e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    return o, e1, e2, np.cross(e1, e2)


def is_param_surface(src: dict) -> bool:
    return src.get("type") == "surface" and isinstance(src.get("parametric"), dict)


def surface_code(expr, values: dict) -> str:
    """Python (numpy) expression of a coordinate of a parametric surface, an
    array function of u and v (the parameters replaced by their values)."""
    return height_code(expr, values, ("u", "v")).replace("0 * x)", "0 * u)")


def param_surface_points(src: dict, values: dict, u, v) -> np.ndarray:
    """Points (3 x n) of a parametric surface at the coordinates (u, v)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    par = src["parametric"]
    return np.vstack([eval(surface_code(par.get(k, 0.0), values), {"np": np, "__builtins__": {}},  # noqa: S307
                           {"u": u, "v": v}) + 0.0 * u for k in AXES])


def surface_tangents(src: dict, values: dict, u, v, eps: float):
    """The tangents (X_u, X_v) of a parametric surface (3 x n each), by
    centered differences of step eps."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    f = lambda a, b: param_surface_points(src, values, a, b)     # noqa: E731
    return (f(u + eps, v) - f(u - eps, v)) / (2 * eps), (f(u, v + eps) - f(u, v - eps)) / (2 * eps)


def surface_samples(src: dict, n: int = 41):
    """Points (u, v) of the flat domain of a surface (2 x k), on a grid of its
    bounding box (its edges included), and the step of the grid."""
    from . import geometry2d as G
    lo, hi = G.bounds(src["shape"])
    U, V = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    P = np.vstack([U.ravel(), V.ravel()])
    step = float(max(hi[0] - lo[0], hi[1] - lo[1])) / (n - 1)
    inside = G.sdf(src["shape"], P.T) <= 1e-9 * max(step, 1e-300)
    return P[:, inside], step


def param_surface_problem(src: dict, values: dict) -> str | None:
    """Why a parametric surface cannot be meshed (not finite, a pole or a
    fold: its tangents not independent at a point of its flat domain)."""
    P, step = surface_samples(src)
    if P.shape[1] == 0:
        return "the flat domain is empty"
    with np.errstate(all="ignore"):
        X = param_surface_points(src, values, P[0], P[1])
        if not np.all(np.isfinite(X)):
            j = int(np.flatnonzero(~np.all(np.isfinite(X), axis=0))[0])
            return f"not finite at (u, v) = ({P[0, j]:.4g}, {P[1, j]:.4g})"
        Xu, Xv = surface_tangents(src, values, P[0], P[1], 1e-6 * max(step, 1e-300))
    nu, nv = np.linalg.norm(Xu, axis=0), np.linalg.norm(Xv, axis=0)
    J = np.linalg.norm(np.cross(Xu.T, Xv.T), axis=1)
    if not np.all(np.isfinite(J)):
        return "the map is not differentiable everywhere"
    bad = J <= 1e-6 * nu.max() * nv.max()
    if np.any(bad):
        j = int(np.flatnonzero(bad)[0])
        return (f"the map degenerates at (u, v) = ({P[0, j]:.4g}, {P[1, j]:.4g}) (a pole, or a fold: its "
                "elements would be flat); take a flat domain that avoids it")
    return None


def _side_range(src: dict, axis: int, at: float, lo, hi, step: float):
    """The range of the other coordinate where the line {axis = at} is on
    the boundary of the flat domain (None if it is not)."""
    from . import geometry2d as G
    other = 1 - axis
    t = np.linspace(lo[other], hi[other], 401)
    P = np.zeros((401, 2))
    P[:, axis], P[:, other] = at, t
    on = np.abs(G.sdf(src["shape"], P)) <= 1e-6 * max(step, 1e-300)
    if not on.any():
        return None
    return float(t[on].min()), float(t[on].max())


def seams(src: dict, values: dict) -> list[dict]:
    """The seams of a parametric surface: two opposite sides of the box of
    its flat domain (u = u0 and u = u1, or v = v0 and v = v1), on its
    boundary, that the map sends to the same curve, point by point, in the
    same direction or reversed (a Moebius strip): [{"axis": 0 (u) or 1 (v),
    "at": (a0, a1), "range": (t0, t1), "reversed": bool}]."""
    from . import geometry2d as G
    lo, hi = (np.asarray(x, float) for x in G.bounds(src["shape"]))
    step = float(max(hi - lo)) / 40
    out = []
    for axis in (0, 1):
        r0 = _side_range(src, axis, lo[axis], lo, hi, step)
        r1 = _side_range(src, axis, hi[axis], lo, hi, step)
        if r0 is None or r1 is None or abs(r0[0] - r1[0]) > 1e-9 * step or abs(r0[1] - r1[1]) > 1e-9 * step:
            continue
        t = np.linspace(r0[0], r0[1], 101)
        A = np.zeros((2, t.size))
        A[axis], A[1 - axis] = lo[axis], t
        B = A.copy()
        B[axis] = hi[axis]
        with np.errstate(all="ignore"):
            XA = param_surface_points(src, values, A[0], A[1])
            scale = float(np.ptp(XA, axis=1).max()) or 1.0
            for rev in (False, True):
                Bt = B.copy()
                if rev:
                    Bt[1 - axis] = r0[0] + r0[1] - t
                XB = param_surface_points(src, values, Bt[0], Bt[1])
                if np.all(np.linalg.norm(XA - XB, axis=0) <= 1e-9 * scale):
                    out.append({"axis": axis, "at": (float(lo[axis]), float(hi[axis])), "range": r0,
                                "reversed": rev})
                    break
    return out


def seam_vertices(src: dict, values: dict, seam: dict, size) -> np.ndarray:
    """Points (2 x n) on the two sides of a seam, about size(points) apart
    (a function of the points 2 x k of the flat domain), the same ones on
    both sides (mirrored if reversed): the vertices of the flat mesh there,
    so that the sides match once mapped."""
    axis, (a0, a1), (t0, t1) = seam["axis"], seam["at"], seam["range"]
    t = np.linspace(t0, t1, 401)
    p = np.zeros((2, t.size))
    p[axis], p[1 - axis] = a0, t
    density = 1.0 / (0.8 * np.asarray(size(p), float))              # points per unit of t
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(t))])
    n = max(2, int(np.ceil(cum[-1])))
    t = np.interp(np.linspace(0.0, cum[-1], n + 1), cum, t)        # evenly in the local size
    t[0], t[-1] = t0, t1
    A = np.zeros((2, t.size))
    A[axis], A[1 - axis] = a0, t
    B = np.zeros((2, t.size))
    B[axis], B[1 - axis] = a1, (t0 + t1 - t) if seam["reversed"] else t
    return np.hstack([A, B])


def scaled_shape(shape: dict, su: float, sv: float) -> dict | None:
    """The 2D shape scaled by (su, sv) along its axes (None if it cannot be:
    fillets, curved sides, turned ellipses)."""
    t = shape.get("type")
    if t in ("union", "intersection", "difference"):
        parts = [scaled_shape(c, su, sv) for c in shape.get("of", [])]
        return None if any(p is None for p in parts) else dict(shape, of=parts)
    S = np.array([su, sv])
    if t == "rectangle":
        return dict(shape, min=(S * np.asarray(shape["min"], float)).tolist(),
                    max=(S * np.asarray(shape["max"], float)).tolist())
    if t == "polygon" and not shape.get("fillets") and not shape.get("splines"):
        return dict(shape, points=[(S * np.asarray(p, float)).tolist() for p in shape["points"]])
    if t == "ball":
        return {"type": "ellipse", "center": (S * np.asarray(shape["center"], float)).tolist(),
                "radii": [float(shape["radius"]) * su, float(shape["radius"]) * sv], "angle": 0.0}
    if t == "ellipse" and float(shape.get("angle", 0.0)) % 180.0 == 0.0:
        r = np.asarray(shape["radii"], float)
        return dict(shape, center=(S * np.asarray(shape["center"], float)).tolist(), radii=(S * r).tolist())
    return None


def scaled_surface(src: dict, su: float, sv: float) -> dict | None:
    """The parametric surface with its flat domain scaled by (su, sv), the
    map then of the scaled coordinates (None if the shape cannot be)."""
    shape = scaled_shape(src["shape"], su, sv)
    if shape is None:
        return None

    class Sub(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in ("u", "v"):
                k = su if node.id == "u" else sv
                return ast.copy_location(ast.BinOp(ast.Name(node.id, ast.Load()), ast.Div(), ast.Constant(k)), node)
            return node
    par = {k: (ast.unparse(Sub().visit(ast.parse(str(e), mode="eval")).body) if isinstance(e, str) else e)
           for k, e in src["parametric"].items()}
    return dict(src, shape=shape, parametric=par)


def map_surface(P2: np.ndarray, src: dict, values: dict) -> np.ndarray:
    """Points of a surface (3 x n) for points of its flat domain (2 x n)."""
    if is_param_surface(src):
        return param_surface_points(src, values, P2[0], P2[1])
    o, e1, e2, n = frame(src)
    x, y = P2[0], P2[1]
    hgt = eval(height_code(src.get("height", 0.0), values), {"np": np, "__builtins__": {}},  # noqa: S307
               {"x": x, "y": y})
    return o[:, None] + e1[:, None] * x + e2[:, None] * y + n[:, None] * hgt


def map_swept(P: np.ndarray, src: dict) -> np.ndarray:
    """Points of a swept surface (3 x n) for the points (a, b, t) of the
    layered mesh of its curve (t in [0, 1] along the sweep)."""
    a, b, t = P
    if src.get("sweep") == "revolution":
        a0, a1 = (math.radians(float(v)) for v in src.get("angle", [0.0, 360.0]))
        th = a0 + (a1 - a0) * t
        return np.vstack([a * np.cos(th), a * np.sin(th), b])
    z0, z1 = (float(v) for v in src.get("z", [0.0, 1.0]))
    return np.vstack([a, b, z0 + (z1 - z0) * t])


def sweep_length(src: dict) -> float:
    """Length of the sweep (for the number of layers): along z, or along the
    largest circle."""
    if src.get("sweep") == "revolution":
        a0, a1 = (math.radians(float(v)) for v in src.get("angle", [0.0, 360.0]))
        r = float(np.max(chain_polyline(src)[:, 0]))
        return abs(a1 - a0) * r
    z0, z1 = (float(v) for v in src.get("z", [0.0, 1.0]))
    return abs(z1 - z0)


def full_turn(src: dict) -> bool:
    if src.get("sweep") != "revolution":
        return False
    a0, a1 = (float(v) for v in src.get("angle", [0.0, 360.0]))
    return abs(abs(a1 - a0) - 360.0) < 1e-9


# ------------------------------------------------------------- scripts

SCRIPT_FUNCTIONS = (arc_frame, bezier_by_length, chain_elements, curve_elements)


def script_functions() -> list[str]:
    lines = ["# Curves (the functions of manifold.py in the interface)"]
    for fn in SCRIPT_FUNCTIONS:
        lines += inspect.getsource(fn).rstrip().splitlines() + [""]
    return lines


SEAM_CODE = '''
def fix_seams(m, seams, top, K):
    """A flat mesh of triangles (degree 1) whose seams match: on each seam,
    the nodes of a side missing on the other side are inserted there (the
    triangle along it split), the nearly equal ones made equal; then of
    degree K (its boundary nodes projected on the domain top)."""
    P = [p for p in m.pts().T]
    pid, idx = m.pid_from_cvid(m.cvid())
    T = [list(pid[idx[i]:idx[i + 1]]) for i in range(len(idx) - 1)]
    X = np.array(P)
    tol = 1e-7 * np.ptp(X, axis=0).max()
    for axis, a0, a1, t0, t1, rev in seams:
        def side(a):
            X = np.array(P)
            return [i for i in np.flatnonzero(np.abs(X[:, axis] - a) < tol)
                    if t0 - tol < X[i, 1 - axis] < t1 + tol]
        def t_of(i, b):              # the coordinate along the side, that of side A
            t = P[i][1 - axis]
            return t0 + t1 - t if (rev and b) else t
        def point(t, b):
            p = np.zeros(2)
            p[axis] = a1 if b else a0
            p[1 - axis] = t0 + t1 - t if (rev and b) else t
            return p
        A = sorted(side(a0), key=lambda i: t_of(i, 0))
        B = sorted(side(a1), key=lambda i: t_of(i, 1))
        ta, tb = [t_of(i, 0) for i in A], [t_of(i, 1) for i in B]
        tau = 0.25 * min(np.diff(ta).min(), np.diff(tb).min())
        # merged along the side: a node of A and one of B closer than tau are one, made equal
        events = sorted([(t, 0, i) for t, i in zip(ta, A)] + [(t, 1, i) for t, i in zip(tb, B)])
        k, pairs = 0, []
        while k < len(events):
            if k + 1 < len(events) and events[k + 1][0] - events[k][0] < tau and events[k + 1][1] != events[k][1]:
                pairs.append((events[k], events[k + 1]))
                k += 2
            else:
                pairs.append((events[k],))
                k += 1
        sides = {0: A, 1: B}
        for group in pairs:
            if len(group) == 2:                           # the node of B moved to that of A
                ea, eb = sorted(group, key=lambda e: e[1])
                P[eb[2]] = point(ea[0], 1)
                continue
            t, s, i = group[0]                            # missing on the other side: inserted there
            b = 1 - s
            mine = sides[b]
            tm = np.array([t_of(j, b) for j in mine])
            k = int(np.searchsorted(tm, t))
            i0, i1 = mine[k - 1], mine[k]                 # the edge (i0, i1): split
            n = len(P)
            P.append(point(t, b))
            for c, tri in enumerate(list(T)):
                if i0 in tri and i1 in tri:
                    T[c] = [n if q == i1 else q for q in tri]
                    T.append([n if q == i0 else q for q in tri])
                    break
            mine.insert(k, n)
    X = np.array(P).T
    T = np.array(T).T
    gt = gf.GeoTrans(f'GT_PK(2,{K})')
    out = gf.Mesh('empty', 2)
    R = gt.pts()
    A, B, C = X[:, T[0]], X[:, T[1]], X[:, T[2]]
    nodes = A[:, None, :] + (B - A)[:, None, :] * R[0][None, :, None] + (C - A)[:, None, :] * R[1][None, :, None]
    out.add_convex(gt, nodes)
    if K > 1:                        # the nodes of the boundary on the domain
        Q = out.pts()
        ids = out.pid_in_faces(out.outer_faces())
        Q[:, ids] = top.project(Q[:, ids])
        out.set_pts(Q)
    return out
'''


REMESH_CODE = '''
def remesh(m, Q, dim):
    """The elements of m with their points moved to Q (dim x nbpts), in a
    mesh of dimension dim: the points that then coincide are merged (seam
    of a revolution); returns (new mesh, new number of each element)."""
    out = gf.Mesh('empty', dim)
    gts, gidx = m.geotrans(m.cvid())
    gts = list(gts) if isinstance(gts, (list, tuple)) else [gts]
    gidx = np.atleast_1d(gidx)
    cvs = m.cvid()
    pids, idx = m.pid_from_cvid(cvs)
    new = np.zeros(int(cvs.max()) + 1, int)
    for k, gt in enumerate(gts):
        sel = np.flatnonzero(gidx == k)
        ids = np.array([pids[idx[c]:idx[c + 1]] for c in sel]).T
        new[cvs[sel]] = out.add_convex(gt, Q[:, ids])
    return out, new


def remap_selections(SEL, new):
    """The selections (computed on the flat domain) on the new elements."""
    out = {}
    for name, (kind, ids) in SEL.items():
        ids = np.array(ids)
        if ids.size:
            ids = new[ids] if kind == 'domain' else np.vstack([new[ids[0]], ids[1]])
        out[name] = (kind, ids)
    return out
'''
