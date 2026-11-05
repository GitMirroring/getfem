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


"""3D geometries (box and CSG) without meshing, for the preview.

The shapes are described implicitly, by a signed distance like function
(negative inside), as the GetFEM mesher does; the preview is its zero level
set on a regular grid (marching cubes), and boundary selections are
evaluated on that surface. Only numpy is needed.
"""

from __future__ import annotations

import numpy as np

BOOLEANS = ("union", "intersection", "difference")


def shape_of(source: dict) -> dict | None:
    """The CSG tree of a 3D box or csg source (None for other sources)."""
    t = source.get("type")
    if t == "box" and len(source.get("size", [])) == 3:
        o, s = source["origin"], source["size"]
        return {"type": "rectangle", "min": list(o), "max": [a + b for a, b in zip(o, s)]}
    if t == "csg" and source.get("dim") == 3:
        return source.get("shape")
    return None


def sdf(shape: dict, P: np.ndarray) -> np.ndarray:
    """Signed distance like function at the points P (n x 3)."""
    t = shape["type"]
    if t == "rectangle":
        lo, hi = np.asarray(shape["min"], float), np.asarray(shape["max"], float)
        q = np.maximum(lo - P, P - hi)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        return np.where(np.all(q <= 0, axis=1), np.max(q, axis=1), outside)
    if t == "ball":
        return np.linalg.norm(P - np.asarray(shape["center"], float), axis=1) - float(shape["radius"])
    if t == "cylinder":
        o = np.asarray(shape["origin"], float)
        a = np.asarray(shape["axis"], float)
        a = a / (np.linalg.norm(a) or 1.0)
        L, r = float(shape["length"]), float(shape["radius"])
        s = (P - o) @ a
        radial = np.linalg.norm((P - o) - s[:, None] * a, axis=1) - r
        axial = np.maximum(-s, s - L)
        q = np.stack([radial, axial], axis=1)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        return np.where(np.all(q <= 0, axis=1), np.max(q, axis=1), outside)
    parts = [sdf(s, P) for s in shape["of"]]
    if t == "union":
        return np.min(parts, axis=0)
    if t == "intersection":
        return np.max(parts, axis=0)
    if t == "difference":
        return np.maximum(parts[0], -np.min(parts[1:], axis=0))
    raise ValueError(f"shape {t!r} is not 3D")


def bounds(shape: dict) -> tuple[np.ndarray, np.ndarray]:
    t = shape["type"]
    if t == "rectangle":
        return np.asarray(shape["min"], float), np.asarray(shape["max"], float)
    if t == "ball":
        c, r = np.asarray(shape["center"], float), float(shape["radius"])
        return c - r, c + r
    if t == "cylinder":
        o = np.asarray(shape["origin"], float)
        a = np.asarray(shape["axis"], float)
        a = a / (np.linalg.norm(a) or 1.0)
        L, r = float(shape["length"]), float(shape["radius"])
        e = r * np.sqrt(np.maximum(1.0 - a * a, 0.0))        # extent of the end disks
        ends = np.array([o, o + L * a])
        return ends.min(axis=0) - e, ends.max(axis=0) + e
    b = [bounds(s) for s in shape["of"]]
    if t == "union":
        return np.min([x[0] for x in b], axis=0), np.max([x[1] for x in b], axis=0)
    if t == "intersection":
        return np.max([x[0] for x in b], axis=0), np.min([x[1] for x in b], axis=0)
    return b[0]


def primitives(shape: dict, path: tuple = ()):
    if shape.get("type") in BOOLEANS:
        for i, s in enumerate(shape.get("of", [])):
            yield from primitives(s, path + (i,))
    else:
        yield path, shape


def faces_matching(shape: dict, centers: np.ndarray, normals: np.ndarray, selection: str,
                   table: dict) -> np.ndarray | None:
    """Which faces of the preview surface (centers, outward normals) are in a
    boundary selection (None if it cannot be evaluated without a mesh)."""
    sel = table.get(selection)
    if sel is None or sel.get("kind") != "boundary":
        return None
    if selection == "boundary":
        return np.ones(len(centers), bool)
    lo, hi = bounds(shape)
    size = float(np.linalg.norm(hi - lo)) or 1.0
    return _match(sel.get("criteria", {}), centers, normals, table, size)


def _match(crit, C, N, table, size):
    t = crit.get("type")
    tol = 0.012 * size             # the surface is the one of a grid: about one grid step
    if t == "plane":
        n = np.asarray(crit["normal"], float)
        n = n / (np.linalg.norm(n) or 1.0)
        on = np.abs((C - np.asarray(crit["point"], float)) @ n) < tol
        return on & (np.abs(N @ n) > 0.9)
    if t == "normal":
        d = np.asarray(crit["direction"], float)
        d = d / (np.linalg.norm(d) or 1.0)
        return N @ d >= np.cos(np.radians(max(float(crit.get("angle", 1.0)), 8.0)))
    if t == "in_box":
        lo, hi = np.asarray(crit["min"], float), np.asarray(crit["max"], float)
        return np.all((C >= lo - tol) & (C <= hi + tol), axis=1)
    if t == "in_ball":
        return np.linalg.norm(C - np.asarray(crit["center"], float), axis=1) <= float(crit["radius"]) + tol
    if t == "union":
        out = np.zeros(len(C), bool)
        for o in crit.get("of", []):
            sub = _match(table.get(o, {}).get("criteria", {}), C, N, table, size) \
                if o != "boundary" else np.ones(len(C), bool)
            if sub is None:
                return None
            out |= sub
        return out
    return None
