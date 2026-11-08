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

"""Transformations of the shapes of a constructive geometry.

Two nodes of the CSG tree hold one shape ("of": [shape]) and move it:

    {"type": "transform", "of": [shape], "translate": [tx, ty(, tz)],
     "angle": degrees, "center": [x, y(, z)], "axis": [ax, ay, az] (3D)}

        the shape turned by angle about center (about the axis through
        center, in 3D), then translated: x -> c + Rot (x - c) + t;

    {"type": "array", "of": [shape], "count": n,
     "kind": "linear", "step": [dx, dy(, dz)]}                copies k = 0..n-1,
     "kind": "circular", "center": [...], "angle": degrees, "axis": [...] (3D)}

        the union of n copies: translated by k step, or turned by k angle
        about center (and axis).

The rest of the interface works on a flattened tree (flatten): the
transforms are applied to the primitives, the arrays replaced by unions of
copies. In 2D, the transformation is written in the parameters of each
primitive (a turned rectangle becomes a polygon whose sides keep their
names xmin...); in 3D, a primitive gets a "_frame" [origin, R] (x ->
origin + R x). Each primitive keeps the path of its node in the tree of the
model ("_src"), so that the references to sides and faces (selections)
are those of the model: a reference to a side of a copied shape is that
side on every copy.
"""

from __future__ import annotations

import copy
import math

import numpy as np

TRANSFORMS = ("transform", "array")
BOOLEANS = ("union", "intersection", "difference")


def has_transforms(shape) -> bool:
    if not isinstance(shape, dict):
        return False
    if shape.get("type") in TRANSFORMS:
        return True
    return any(has_transforms(s) for s in shape.get("of", []) if isinstance(s, dict)) \
        or has_transforms(shape.get("profile"))


def rotation(angle_deg: float, axis=None, dim: int = 2) -> np.ndarray:
    a = math.radians(float(angle_deg))
    if dim == 2:
        return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    u = np.asarray(axis if axis is not None else [0.0, 0.0, 1.0], float)
    u = u / (np.linalg.norm(u) or 1.0)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def maps(node: dict, dim: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The rigid maps (A, b), x -> A x + b, of a transform (one) or an array
    (one per copy)."""
    zero = np.zeros(dim)
    if node["type"] == "transform":
        A = rotation(node.get("angle", 0.0), node.get("axis"), dim)
        c = np.asarray(node.get("center", zero), float)
        t = np.asarray(node.get("translate", zero), float)
        return [(A, c - A @ c + t)]
    n = max(int(node.get("count", 1)), 1)
    if node.get("kind", "linear") == "linear":
        step = np.asarray(node.get("step", zero), float)
        return [(np.eye(dim), k * step) for k in range(n)]
    c = np.asarray(node.get("center", zero), float)
    out = []
    for k in range(n):
        A = rotation(k * float(node.get("angle", 0.0)), node.get("axis"), dim)
        out.append((A, c - A @ c))
    return out


def _bake_2d(prim: dict, A: np.ndarray, b: np.ndarray) -> dict:
    """A 2D primitive moved by x -> A x + b (A a rotation)."""
    t = prim["type"]
    move = lambda p: (A @ np.asarray(p, float) + b).tolist()       # noqa: E731
    turn = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    out = dict(prim)
    if t == "ball":
        out["center"] = move(prim["center"])
    elif t == "ellipse":
        out["center"] = move(prim["center"])
        out["angle"] = float(prim.get("angle", 0.0)) + turn
    elif t == "polygon":
        out["points"] = [move(p) for p in prim["points"]]
    elif t == "rectangle":
        if np.allclose(A, np.eye(2)):
            out["min"], out["max"] = move(prim["min"]), move(prim["max"])
        else:                                 # turned: a polygon, its sides named as the rectangle's
            (x0, y0), (x1, y1) = prim["min"], prim["max"]
            out = {"type": "polygon", "points": [move(p) for p in ([x0, y0], [x1, y0], [x1, y1], [x0, y1])],
                   "side_names": ["ymin", "xmax", "ymax", "xmin"]}
            for k in ("_src",):
                if k in prim:
                    out[k] = prim[k]
    return out


def flatten(shape: dict, dim: int, A=None, b=None) -> dict:
    """The tree without its transforms and arrays (see the module): marked
    "_flat"; the shape itself when it has none. A, b: a map x -> A x + b
    applied to the whole tree (the transforms above it, frame_along)."""
    moved = A is not None and not (np.allclose(A, np.eye(dim)) and np.allclose(b, 0))
    if not has_transforms(shape) and not moved:
        return shape

    def rec(node, src, A, b):
        t = node.get("type")
        if t in TRANSFORMS:
            child = node.get("of", [None])[0]
            if not isinstance(child, dict):
                return {"type": "union", "of": []}
            copies = [rec(child, src + (0,), A @ A2, A @ b2 + b) for A2, b2 in maps(node, dim)]
            return copies[0] if t == "transform" else {"type": "union", "of": copies}
        if t in BOOLEANS:
            return {"type": t, "of": [rec(s, src + (i,), A, b) for i, s in enumerate(node.get("of", []))]}
        prim = copy.deepcopy(node)
        prim["_src"] = list(src)
        if t in ("extrusion", "revolution") and has_transforms(prim.get("profile")):
            prim["profile"] = flatten(prim["profile"], 2)
        if np.allclose(A, np.eye(dim)) and np.allclose(b, 0):
            return prim
        if dim == 2:
            return _bake_2d(prim, A, b)
        prim["_frame"] = [b.tolist(), A.tolist()]
        return prim

    out = rec(shape, (), np.asarray(A, float) if moved else np.eye(dim),
              np.asarray(b, float) if moved else np.zeros(dim))
    out = dict(out) if isinstance(out, dict) else out
    out["_flat"] = True
    return out


def source_path(prim: dict, path) -> tuple:
    """The path in the tree of the model of a primitive of a (flattened) tree."""
    return tuple(prim["_src"]) if "_src" in prim else tuple(path)


def frame_along(shape: dict, path, dim: int):
    """The rigid map (A, b) of the transforms on the way to the node at path
    (in the tree of the model; the first copy of an array), x -> A x + b."""
    A, b = np.eye(dim), np.zeros(dim)
    node = shape
    for i in path:
        if not isinstance(node, dict):
            break
        if node.get("type") in TRANSFORMS:
            A2, b2 = maps(node, dim)[0]
            A, b = A @ A2, A @ b2 + b
        try:
            node = node["of"][i]
        except (KeyError, IndexError, TypeError):
            break
    return A, b
