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

import math

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


def profile_curves(shape: dict, h: float, refs: bool = False) -> list:
    """The boundary of the profile of an extrusion or a revolution, as
    polylines (n x 2) with segments of length about h; for a revolution,
    without the parts on the axis. refs: (polyline, side reference in the
    profile) pairs instead."""
    prof = effective_profile(shape)
    wrapped = prof is not shape["profile"]       # the profile is then prof["of"][0]
    size = G2.size(prof)
    out = []
    for p in G2.pieces(prof):
        side = p.side
        length = side.length(p.t0, p.t1)
        n = max(1, int(np.ceil(length / h)))
        pts = side.points(np.linspace(p.t0, p.t1, n + 1))
        if shape["type"] == "revolution" and np.max(np.abs(pts[:, 0])) < 1e-9 * size:
            continue
        if not refs:
            out.append(pts)
        elif not wrapped or side.shape[:1] == (0,):
            ref = {"shape": list(side.shape[1:] if wrapped else side.shape), "side": side.side}
            out.append((pts, ref))
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
                   table: dict, cell_faces=None) -> np.ndarray | None:
    """Which faces of the preview surface (centers, outward normals) are in a
    boundary selection (None if it cannot be evaluated without a mesh).
    cell_faces: the face (face_key) of each cell, for 'faces' criteria."""
    sel = table.get(selection)
    if sel is None or sel.get("kind") != "boundary":
        return None
    if selection == "boundary":
        return np.ones(len(centers), bool)
    lo, hi = bounds(shape)
    size = float(np.linalg.norm(hi - lo)) or 1.0
    return _match(sel.get("criteria", {}), centers, normals, table, size, cell_faces)


def _match(crit, C, N, table, size, cell_faces=None):
    t = crit.get("type")
    if t == "faces":
        if cell_faces is None:
            return None
        keys = {face_key(r) for r in crit.get("faces", [])}
        return np.array([k in keys for k in cell_faces], bool)
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
            sub = _match(table.get(o, {}).get("criteria", {}), C, N, table, size, cell_faces) \
                if o != "boundary" else np.ones(len(C), bool)
            if sub is None:
                return None
            out |= sub
        return out
    return None


# ------------------------------------------------------------------ faces

# A face of a 3D CSG shape is a face of one of its primitives, referenced as
# {"shape": [path], "face": name}: xmin... zmax of a box (rectangle), "sphere"
# of a ball, "side", "start" and "end" of a cylinder, "bottom" and "top" of
# an extrusion; the side of an extrusion or of a revolution swept by a side
# of the profile is {"shape": [path], "face": "side", "profile": <side
# reference in the profile, as in geometry2d>}.

BOX_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def primitive_at(shape: dict, path) -> dict | None:
    for i in path:
        try:
            shape = shape["of"][i]
        except (KeyError, IndexError, TypeError):
            return None
    return shape


def face_key(ref: dict) -> tuple:
    prof = ref.get("profile") or {}
    return (tuple(ref.get("shape", [])), ref.get("face"), tuple(prof.get("shape", [])), prof.get("side"))


def primitive_faces(prim: dict) -> list[dict]:
    """The faces of a primitive (without their "shape" key)."""
    t = prim.get("type")
    if t == "rectangle":
        return [{"face": f} for f in BOX_FACES]
    if t == "ball":
        return [{"face": "sphere"}]
    if t == "cylinder":
        return [{"face": f} for f in ("side", "start", "end")]
    if t in ("extrusion", "revolution"):
        sides = [{"face": "side", "profile": dict(side.ref)} for side in G2.sides(prim["profile"])]
        return ([{"face": "bottom"}, {"face": "top"}] if t == "extrusion" else []) + sides
    return []


def face_exists(shape: dict, ref: dict) -> bool:
    prim = primitive_at(shape, ref.get("shape", []))
    if prim is None or prim.get("type") in BOOLEANS:
        return False
    key = face_key(dict(ref, shape=[]))
    return any(face_key(dict(f, shape=[])) == key for f in primitive_faces(prim))


def face_label(shape: dict, ref: dict) -> str:
    """'xmin of rectangle 1', 'top of extrusion 2', 'side (circle of ball 1)
    of revolution 1'."""
    names = [p for p, _ in primitives(shape)]
    path = tuple(ref.get("shape", []))
    prim = primitive_at(shape, path)
    if prim is None or path not in names:
        return f"{ref.get('face')} of a missing shape"
    what = f"{prim['type']} {names.index(path) + 1}"
    if ref.get("profile"):
        return f"side ({G2.side_label(prim['profile'], ref['profile'])}) of {what}"
    return f"{ref.get('face')} of {what}"


def toggle_face(criteria: dict, ref: dict) -> dict:
    """Criteria after a click on a face: added, or removed if already there
    (another kind of criterion is replaced by the clicked face)."""
    ref = {k: v for k, v in ref.items() if k in ("shape", "face", "profile")}
    if criteria.get("type") != "faces":
        return {"type": "faces", "faces": [ref]}
    faces = [dict(r) for r in criteria.get("faces", [])]
    kept = [r for r in faces if face_key(r) != face_key(ref)]
    if len(kept) == len(faces):
        kept.append(ref)
    return {"type": "faces", "faces": kept}


def face_spec(shape: dict, ref: dict) -> dict:
    """Concrete description of a face (numbers), for on_face."""
    prim = primitive_at(shape, ref.get("shape", []))
    t, f = prim["type"], ref.get("face")
    if t == "rectangle":
        k = BOX_FACES.index(f)
        axis, hi = k // 2, k % 2 == 1
        return {"kind": "box_face", "axis": axis, "value": float((prim["max"] if hi else prim["min"])[axis]),
                "min": [float(x) for x in prim["min"]], "max": [float(x) for x in prim["max"]]}
    if t == "ball":
        return {"kind": "sphere", "center": [float(x) for x in prim["center"]], "radius": float(prim["radius"])}
    if t == "cylinder":
        a = np.asarray(prim["axis"], float)
        a = (a / (np.linalg.norm(a) or 1.0)).tolist()
        o, L, r = [float(x) for x in prim["origin"]], float(prim["length"]), float(prim["radius"])
        if f == "side":
            return {"kind": "cylinder_side", "origin": o, "axis": a, "length": L, "radius": r}
        return {"kind": "disk", "center": o if f == "start" else [x + L * y for x, y in zip(o, a)],
                "normal": a, "radius": r}
    if t == "extrusion" and f in ("bottom", "top"):
        z = sorted(float(v) for v in prim["z"])
        return {"kind": "cap", "z": z[0] if f == "bottom" else z[1], "profile": prim["profile"]}
    side = next(sd for sd in G2.sides(prim["profile"])
                if (list(sd.shape), sd.side) == (list(ref["profile"]["shape"]), ref["profile"]["side"]))
    curve = {"kind": side.kind, "a": [float(x) for x in side.a], "b": [float(x) for x in side.b]}
    if t == "extrusion":
        return {"kind": "extruded_side", "curve": curve, "z": sorted(float(v) for v in prim["z"])}
    return {"kind": "revolved_side", "curve": curve}


# The functions below are also copied into the meshing scripts (they use
# numpy and math only).

def curve_distance(curve, Q):
    """Distance of the 2D points Q (n x 2) to a side (segment, circle or
    ellipse; for an ellipse, a first order estimate)."""
    a, b = np.asarray(curve["a"], float), curve["b"]
    if curve["kind"] == "segment":
        d = np.asarray(b, float) - a
        s = np.clip((Q - a) @ d / max(float(d @ d), 1e-300), 0.0, 1.0)
        return np.linalg.norm(Q - a - s[:, None] * d, axis=1)
    if curve["kind"] == "arc":
        return np.abs(np.linalg.norm(Q - a, axis=1) - b[0])
    r1, r2, th = b
    D = Q - a
    u = D[:, 0] * math.cos(th) + D[:, 1] * math.sin(th)
    v = -D[:, 0] * math.sin(th) + D[:, 1] * math.cos(th)
    g = np.hypot(u / r1, v / r2)
    grad = np.hypot(u / r1 ** 2, v / r2 ** 2) / np.maximum(g, 1e-300)
    return np.abs(g - 1.0) / np.maximum(grad, 1e-300)


def curve_normal(curve, Q):
    """Unit normal of a side at the 2D points Q (near it)."""
    a, b = np.asarray(curve["a"], float), curve["b"]
    if curve["kind"] == "segment":
        d = np.asarray(b, float) - a
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)
        return np.tile(n, (len(Q), 1))
    D = Q - a
    if curve["kind"] == "arc":
        return D / np.maximum(np.linalg.norm(D, axis=1), 1e-300)[:, None]
    r1, r2, th = b
    c, s = math.cos(th), math.sin(th)
    u, v = D[:, 0] * c + D[:, 1] * s, -D[:, 0] * s + D[:, 1] * c
    nu, nv = u / r1 ** 2, v / r2 ** 2
    N = np.column_stack([nu * c - nv * s, nu * s + nv * c])
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]


def profile_level(shape, P):
    """A distance like function (negative inside) of a 2D profile, close to
    the distance near its boundary."""
    t = shape["type"]
    if t == "rectangle":
        lo, hi = np.asarray(shape["min"], float), np.asarray(shape["max"], float)
        return np.max(np.maximum(lo - P, P - hi), axis=1)
    if t == "ball":
        return np.linalg.norm(P - np.asarray(shape["center"], float), axis=1) - float(shape["radius"])
    if t == "ellipse":
        r1, r2 = (float(r) for r in shape["radii"])
        curve = {"kind": "ellipse", "a": shape["center"], "b": [r1, r2, math.radians(float(shape.get("angle", 0)))]}
        D = P - np.asarray(shape["center"], float)
        th = curve["b"][2]
        u = D[:, 0] * math.cos(th) + D[:, 1] * math.sin(th)
        v = -D[:, 0] * math.sin(th) + D[:, 1] * math.cos(th)
        d = curve_distance(curve, P)
        return np.where(np.hypot(u / r1, v / r2) < 1, -d, d)
    if t == "polygon":
        V = np.asarray(shape["points"], float)
        W = np.roll(V, -1, axis=0)
        d = np.min([curve_distance({"kind": "segment", "a": V[i], "b": W[i]}, P) for i in range(len(V))], axis=0)
        x, y = P[:, 0:1], P[:, 1:2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = V[:, 0] + (y - V[:, 1]) * (W[:, 0] - V[:, 0]) / (W[:, 1] - V[:, 1])
        inside = np.sum(((V[:, 1] > y) != (W[:, 1] > y)) & (x < xc), axis=1) % 2 == 1
        return np.where(inside, -d, d)
    parts = [profile_level(s, P) for s in shape["of"]]
    if t == "union":
        return np.min(parts, axis=0)
    if t == "intersection":
        return np.max(parts, axis=0)
    return np.maximum(parts[0], -np.min(parts[1:], axis=0))


def on_face(spec, Q, N, tol):
    """Whether a face of a mesh, of vertices Q (n x 3) and unit normal N (at
    its center), lies on the face described by spec (face_spec): all its
    vertices on the surface (tolerance tol), its normal along the normal of
    the surface (which tells apart a face at the edge of two surfaces)."""
    k = spec["kind"]
    C = Q.mean(axis=0)
    if k == "box_face":
        i = spec["axis"]
        lo, hi = np.asarray(spec["min"], float), np.asarray(spec["max"], float)
        ok = np.all(np.abs(Q[:, i] - spec["value"]) < tol) and np.all((Q >= lo - tol) & (Q <= hi + tol))
        return bool(ok and abs(N[i]) > 0.5)
    if k == "sphere":
        c = np.asarray(spec["center"], float)
        D = Q - c
        if not np.all(np.abs(np.linalg.norm(D, axis=1) - spec["radius"]) < tol):
            return False
        n = (C - c) / (np.linalg.norm(C - c) or 1.0)
        return bool(abs(n @ N) > 0.5)
    if k in ("cylinder_side", "disk"):
        a = np.asarray(spec["axis" if k == "cylinder_side" else "normal"], float)
        o = np.asarray(spec["origin" if k == "cylinder_side" else "center"], float)
        s = (Q - o) @ a
        R = np.linalg.norm((Q - o) - s[:, None] * a, axis=1)
        if k == "disk":
            return bool(np.all(np.abs(s) < tol) and np.all(R < spec["radius"] + tol) and abs(a @ N) > 0.5)
        if not (np.all(np.abs(R - spec["radius"]) < tol) and np.all(s > -tol) and np.all(s < spec["length"] + tol)):
            return False
        rc = (C - o) - ((C - o) @ a) * a
        return bool(abs(rc @ N) > 0.5 * (np.linalg.norm(rc) or 1.0))
    if k == "cap":
        return bool(np.all(np.abs(Q[:, 2] - spec["z"]) < tol) and abs(N[2]) > 0.5
                    and np.all(profile_level(spec["profile"], Q[:, :2]) < tol))
    if k == "extruded_side":
        z0, z1 = spec["z"]
        if not (np.all(curve_distance(spec["curve"], Q[:, :2]) < tol)
                and np.all(Q[:, 2] > z0 - tol) and np.all(Q[:, 2] < z1 + tol)):
            return False
        n = curve_normal(spec["curve"], C[None, :2])[0]
        return bool(abs(n[0] * N[0] + n[1] * N[1]) > 0.5)
    if k == "revolved_side":
        rz = np.column_stack([np.hypot(Q[:, 0], Q[:, 1]), Q[:, 2]])
        if not np.all(curve_distance(spec["curve"], rz) < tol):
            return False
        rc = math.hypot(C[0], C[1])
        n = curve_normal(spec["curve"], np.array([[rc, C[2]]]))[0]
        e = np.array([C[0], C[1]]) / (rc or 1.0)
        return bool(abs(n[0] * (e[0] * N[0] + e[1] * N[1]) + n[1] * N[2]) > 0.5)
    return False


FACE_FUNCTIONS = (curve_distance, curve_normal, profile_level, on_face)
