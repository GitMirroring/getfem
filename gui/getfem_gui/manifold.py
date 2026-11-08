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
- {"type": "swept_surface", "sweep": "extrusion" or "revolution", "points",
  "through", "closed" (a 2D chain, as a curve), "z": [z0, z1] (extrusion
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
TOPOLOGICAL_DIM = {"curve": 1, "surface": 2, "swept_surface": 2}


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

def chain_pieces(src: dict) -> list:
    """The pieces of a chain: ["segment", A, B] or ["arc", A, T, B] (the arc
    from A to B through T); numbers, as lists."""
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


def chain_elements(pieces, h, order):
    """Nodes of the elements of a chain (GetFEM order of GT_PK(1, order):
    equally spaced from the first to the last one): an array (dim x
    (order + 1) x number of elements) (marked: copied into the scripts)."""
    out = []
    for piece in pieces:
        if piece[0] == "segment":
            A, B = np.asarray(piece[1], float), np.asarray(piece[2], float)
            n = max(1, int(math.ceil(float(np.linalg.norm(B - A)) / h)))
            s = np.linspace(0.0, 1.0, n * order + 1)
            X = A[:, None] + (B - A)[:, None] * s[None, :]
        else:
            C, r, e1, e2, angle = arc_frame(piece[1], piece[2], piece[3])
            n = max(1, int(math.ceil(r * angle / h)))
            s = np.linspace(0.0, angle, n * order + 1)
            X = C[:, None] + r * (e1[:, None] * np.cos(s)[None, :] + e2[:, None] * np.sin(s)[None, :])
            X[:, -1] = np.asarray(piece[3], float)          # the end exactly
        X[:, 0] = np.asarray(piece[1], float)
        out.append(np.stack([X[:, k * order:k * order + order + 1] for k in range(n)], axis=2))
    return np.concatenate(out, axis=2)


def chain_polyline(src: dict, n_arc: int = 64) -> np.ndarray:
    """Points along the chain (for the preview): an array (number x dim)."""
    pts = []
    for piece in chain_pieces(src):
        if piece[0] == "segment":
            seg = np.array([piece[1], piece[2]], float)
        else:
            C, r, e1, e2, angle = arc_frame(piece[1], piece[2], piece[3])
            s = np.linspace(0.0, angle, max(8, int(n_arc * angle / (2 * math.pi)) + 2))
            seg = C[None, :] + r * (np.cos(s)[:, None] * e1[None, :] + np.sin(s)[:, None] * e2[None, :])
        pts.append(seg if not pts else seg[1:])
    return np.concatenate(pts) if pts else np.zeros((0, chain_dim(src)))


def chain_ends(src: dict) -> list:
    """The two ends of an open chain (no end for a closed one)."""
    if src.get("closed") or len(src.get("points", [])) < 2:
        return []
    return [list(map(float, src["points"][0])), list(map(float, src["points"][-1]))]


# ------------------------------------------------------------- surfaces

NUMPY_FUNCTIONS = {
    "sqr": "np.square", "sqrt": "np.sqrt", "pow": "np.power", "exp": "np.exp", "log": "np.log",
    "log10": "np.log10", "sin": "np.sin", "cos": "np.cos", "tan": "np.tan", "asin": "np.arcsin",
    "acos": "np.arccos", "atan": "np.arctan", "atan2": "np.arctan2", "sinh": "np.sinh", "cosh": "np.cosh",
    "tanh": "np.tanh", "abs": "np.abs", "min": "np.minimum", "max": "np.maximum",
}


def height_code(height, values: dict) -> str:
    """Python (numpy) expression of the height of a surface, a function of
    the arrays x and y: the parameters replaced by their values, the
    functions by those of numpy."""
    if isinstance(height, (int, float)) and not isinstance(height, bool):
        return f"({float(height)!r} + 0 * x)"

    class Sub(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in ("x", "y"):
                return node
            if node.id == "pi":
                return ast.copy_location(ast.Constant(math.pi), node)
            if node.id in values:
                if np.shape(values[node.id]):
                    raise ValueError(f"{node.id} is not a number: use its components, {node.id}(1)...")
                return ast.copy_location(ast.Constant(float(values[node.id])), node)
            raise ValueError(f"unknown name {node.id!r} in the height")

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in values:     # a component: C(1)
                from . import parameters as P
                value = P.evaluate(ast.unparse(node), values)
                if np.shape(value):
                    raise ValueError(f"{ast.unparse(node)} is not a number")
                return ast.copy_location(ast.Constant(float(value)), node)
            if not isinstance(node.func, ast.Name) or node.func.id not in NUMPY_FUNCTIONS:
                raise ValueError("unknown function in the height")
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


def map_surface(P2: np.ndarray, src: dict, values: dict) -> np.ndarray:
    """Points of a surface (3 x n) for points of its flat domain (2 x n)."""
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
        r = max(float(p[0]) for p in src.get("points", [[1.0, 0.0]]))
        return abs(a1 - a0) * r
    z0, z1 = (float(v) for v in src.get("z", [0.0, 1.0]))
    return abs(z1 - z0)


def full_turn(src: dict) -> bool:
    if src.get("sweep") != "revolution":
        return False
    a0, a1 = (float(v) for v in src.get("angle", [0.0, 360.0]))
    return abs(abs(a1 - a0) - 360.0) < 1e-9


# ------------------------------------------------------------- scripts

SCRIPT_FUNCTIONS = (arc_frame, chain_elements)


def script_functions() -> list[str]:
    lines = ["# Curves (the functions of manifold.py in the interface)"]
    for fn in SCRIPT_FUNCTIONS:
        lines += inspect.getsource(fn).rstrip().splitlines() + [""]
    return lines


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
