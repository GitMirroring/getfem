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
(negative inside), as the GetFEM mesher does. The preview (app/scene.py) is
made of the surfaces of the primitives, each one clipped to the part that
is on the boundary of the whole shape (boundary_level): the faces stay flat
and the edges sharp. Boundary selections are evaluated on that surface.
Only numpy is needed here.
"""

from __future__ import annotations

import numpy as np

from . import geometry2d as G2

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
    if t == "extrusion":
        z0, z1 = (float(v) for v in shape["z"])
        q = np.stack([G2.sdf(shape["profile"], P[:, :2]), np.maximum(z0 - P[:, 2], P[:, 2] - z1)], axis=1)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        return np.where(np.all(q <= 0, axis=1), np.max(q, axis=1), outside)
    if t == "revolution":
        return G2.sdf(shape["profile"], meridian(P))
    parts = [sdf(s, P) for s in shape["of"]]
    if t == "union":
        return np.min(parts, axis=0)
    if t == "intersection":
        return np.max(parts, axis=0)
    if t == "difference":
        return np.maximum(parts[0], -np.min(parts[1:], axis=0))
    raise ValueError(f"shape {t!r} is not 3D")


def meridian(P: np.ndarray) -> np.ndarray:
    """(r, z) of the points P (n x 3), for a revolution about the z axis."""
    return np.column_stack([np.hypot(P[:, 0], P[:, 1]), P[:, 2]])


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
    if t == "extrusion":
        lo, hi = G2.bounds(shape["profile"])
        z = sorted(float(v) for v in shape["z"])
        return np.array([lo[0], lo[1], z[0]]), np.array([hi[0], hi[1], z[1]])
    if t == "revolution":
        lo, hi = G2.bounds(shape["profile"])
        R = max(float(hi[0]), 0.0)
        return np.array([-R, -R, lo[1]]), np.array([R, R, hi[1]])
    b = [bounds(s) for s in shape["of"]]
    if t == "union":
        return np.min([x[0] for x in b], axis=0), np.max([x[1] for x in b], axis=0)
    if t == "intersection":
        return np.max([x[0] for x in b], axis=0), np.min([x[1] for x in b], axis=0)
    return b[0]


def effective_profile(shape: dict) -> dict:
    """The profile of an extrusion or a revolution; for a revolution, its
    part with r >= 0 (the rest is ignored, as by the GetFEM mesher)."""
    prof = shape["profile"]
    if shape["type"] == "revolution":
        lo, hi = G2.bounds(prof)
        if lo[0] < 0:
            m = 1.0 + float(np.max(np.abs(np.concatenate([lo, hi]))))
            prof = {"type": "intersection",
                    "of": [prof, {"type": "rectangle", "min": [0.0, lo[1] - m], "max": [hi[0] + m, hi[1] + m]}]}
    return prof


def profile_curves(shape: dict, h: float) -> list[np.ndarray]:
    """The boundary of the profile of an extrusion or a revolution, as
    polylines (n x 2) with segments of length about h; for a revolution,
    without the parts on the axis."""
    prof = effective_profile(shape)
    size = G2.size(prof)
    out = []
    for p in G2.pieces(prof):
        side = p.side
        length = side.length(p.t0, p.t1)
        n = max(1, int(np.ceil(length / h)))
        pts = side.points(np.linspace(p.t0, p.t1, n + 1))
        if shape["type"] == "revolution" and np.max(np.abs(pts[:, 0])) < 1e-9 * size:
            continue
        out.append(pts)
    return out


def primitives(shape: dict, path: tuple = ()):
    if shape.get("type") in BOOLEANS:
        for i, s in enumerate(shape.get("of", [])):
            yield from primitives(s, path + (i,))
    else:
        yield path, shape


def _sdf_with(shape: dict, P: np.ndarray, path: tuple, value: float, here: tuple = ()) -> np.ndarray:
    """sdf where the primitive at `path` is replaced by a constant value."""
    if here == path:
        return np.full(len(P), value)
    t = shape["type"]
    if t not in BOOLEANS:
        return sdf(shape, P)
    parts = [_sdf_with(s, P, path, value, here + (i,)) for i, s in enumerate(shape["of"])]
    if t == "union":
        return np.min(parts, axis=0)
    if t == "intersection":
        return np.max(parts, axis=0)
    return np.maximum(parts[0], -np.min(parts[1:], axis=0))


def orientation(shape: dict, path: tuple) -> int:
    """+1 if the result grows with the primitive at path (its outward normal
    is outward of the result), -1 if it is subtracted (an odd number of
    times)."""
    sign, node = 1, shape
    for i in path:
        if node["type"] == "difference" and i > 0:
            sign = -sign
        node = node["of"][i]
    return sign


def boundary_level(shape: dict, path: tuple, P: np.ndarray, big: float) -> np.ndarray:
    """On the surface of the primitive at path: negative where it is on the
    boundary of the whole shape, positive elsewhere. The primitive is
    replaced by "everywhere inside" (-big) then "everywhere outside" (+big):
    the point is on the boundary if the rest of the shape then gives inside,
    then outside. With a large `big`, the result is a combination of the
    distances to the other primitives: smooth, so that clipping the surface
    of the primitive along it (linear interpolation) is accurate."""
    eps = big
    lo, hi = _sdf_with(shape, P, path, -eps), _sdf_with(shape, P, path, eps)
    if orientation(shape, path) > 0:
        return np.maximum(lo, -hi)
    return np.maximum(hi, -lo)


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
    tol = 1e-4 * size             # the preview surface is exact, but for the curves
    if t == "plane":
        n = np.asarray(crit["normal"], float)
        n = n / (np.linalg.norm(n) or 1.0)
        on = np.abs((C - np.asarray(crit["point"], float)) @ n) < tol
        return on & (np.abs(N @ n) > 0.99)
    if t == "normal":
        d = np.asarray(crit["direction"], float)
        d = d / (np.linalg.norm(d) or 1.0)
        return N @ d >= np.cos(np.radians(max(float(crit.get("angle", 1.0)), 1.0)))
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
