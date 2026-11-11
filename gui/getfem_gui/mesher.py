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

"""Meshing step: geometry + parameters -> GetFEM mesh with regions.

For each mesh of the model, a readable script is generated that:
  1. builds the mesh (structured, GetFEM mesher, Gmsh) or imports it;
  2. evaluates the selections of the geometry on this mesh;
  3. renumbers the regions 1, 2, ... in the order of the selections;
  4. saves the mesh in the native GetFEM format (regions included)
     together with a JSON information file (regions, number of elements, ...);
  5. exports VTU files for display: the mesh and each selection.

The result is cached: remeshing only happens when the geometry, the mesh
parameters or the source file have changed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry2d as G, geometry3d as G3, model as M, parameters as P, transforms as T

# To be incremented whenever the generated script changes, to invalidate caches.
SCRIPT_VERSION = 14

# Subdirectory of the working directory where meshes are stored.
MESH_DIR = "meshes"

GF_SHAPES = {"union": "union", "intersection": "intersect", "difference": "set minus"}
OCC_BOOLEANS = {"union": "fuse", "intersection": "intersect", "difference": "cut"}


class _W:
    def __init__(self):
        self.lines: list[str] = []
        self.counter = 0

    def __call__(self, line=""):
        self.lines.append(line)

    def section(self, title):
        self("")
        self(f"# --- {title} " + "-" * max(0, 70 - len(title)))

    def var(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter}"


def _floats(v):
    return [float(x) for x in v]


def _points(shape):
    return [_floats(p) for p in shape["points"]]


# ------------------------------------------------------------------ construction

def _build_structured(model, geom, mesh, w):
    src = geom["source"]
    axes = [f"np.linspace({float(o)!r}, {float(o) + float(s)!r}, {int(n) + 1})"
            for o, s, n in zip(src["origin"], src["size"], mesh["divisions"])]
    kind = "regular_simplices" if mesh.get("element", "simplex") == "simplex" else "cartesian"
    w(f"m = gf.Mesh({kind!r}, {', '.join(axes)})")


def _gf_shape(shape, w) -> str:
    """Write the lines creating a GetFEM MesherObject and return its variable name."""
    t = shape["type"]
    parts = [_gf_shape(s, w) for s in shape["of"]] if t in GF_SHAPES else []
    v = w.var("o")
    if t == "rectangle":
        w(f"{v} = gf.MesherObject('rectangle', {_floats(shape['min'])}, {_floats(shape['max'])})")
    elif t == "ball":
        w(f"{v} = gf.MesherObject('ball', {_floats(shape['center'])}, {float(shape['radius'])!r})")
    elif t == "polygon" and G.fillet_radii(shape, len(shape["points"])):
        _gf_rounded_polygon(shape, v, w)
    elif t == "polygon":
        w(f"{v} = gf.MesherObject('polygon', np.array({_points(shape)}).T)")
    elif t == "ellipse":
        r1, r2 = _floats(shape["radii"])
        w(f"{v} = gf.MesherObject('ellipse', {_floats(shape['center'])}, {r1!r}, {r2!r}, "
          f"{math.radians(float(shape.get('angle', 0.0)))!r})")
    elif t == "cylinder":
        w(f"{v} = gf.MesherObject('cylinder', {_floats(shape['origin'])}, {_floats(shape['axis'])}, "
          f"{float(shape['length'])!r}, {float(shape['radius'])!r})")
    elif t in ("extrusion", "revolution"):
        p = _gf_shape(shape["profile"], w)
        if t == "extrusion":
            z0, z1 = _floats(shape["z"])
            w(f"{v} = gf.MesherObject('extrusion', {p}, {z0!r}, {z1!r})")
        else:
            w(f"{v} = gf.MesherObject('revolution', {p})   # profile in the (r, z) plane, axis z")
    elif t in GF_SHAPES and len(parts) == 1:        # a boolean of one shape: that shape
        return parts[0]
    elif t == "difference" and len(parts) > 2:
        # a minus b minus c...: a minus the union of the others (some versions of
        # GetFEM only use the first two objects of 'set minus')
        w(f"{v} = gf.MesherObject('set minus', {parts[0]}, "
          f"gf.MesherObject('union', {', '.join(parts[1:])}))")
    else:
        w(f"{v} = gf.MesherObject({GF_SHAPES[t]!r}, {', '.join(parts)})")
    fr = G3.frame(shape) if t not in GF_SHAPES else None
    if fr is not None:                    # in its work plane, or moved: placed (origin, axes)
        O, R = fr
        w(f"{v} = gf.MesherObject('placed', {v}, {O.tolist()!r}, {R[:, 0].tolist()!r}, "
          f"{R[:, 1].tolist()!r})")
    return v


def _fillet_lines_3d(shape, h: float) -> np.ndarray:
    """Points every h or so along the lines where the arcs of the fillets of
    the profiles of extrusions and revolutions meet the sides (3 x n), on
    the boundary of the solid: fixed vertices for the GetFEM mesher, so that
    each face of the mesh is on a side or on an arc."""
    pts = []
    for _, prim in G3.primitives(shape):
        if prim.get("type") not in ("extrusion", "revolution"):
            continue
        J = G.fillet_ends(prim["profile"])
        if not len(J):
            continue
        if prim["type"] == "extrusion":
            z0, z1 = sorted(float(z) for z in prim["z"])
            z = np.linspace(z0, z1, max(1, int(math.ceil((z1 - z0) / h))) + 1)
            local = [[x, y, zz] for x, y in J for zz in z]
        else:
            local = []
            for r, zz in J:
                if r <= 0:
                    continue
                n = max(8, int(math.ceil(2 * math.pi * r / h)))
                local += [[r * math.cos(a), r * math.sin(a), zz] for a in np.linspace(0, 2 * math.pi, n,
                                                                                        endpoint=False)]
        if not local:
            continue
        X = np.asarray(local, float)
        if G3.frame(prim) is not None:
            X = G3.to_global(prim, X)
        on = np.abs(G3.sdf(shape, X)) <= 1e-9 * max(1.0, float(np.max(np.abs(X))))
        pts += X[on].tolist()
    if not pts:
        return np.zeros((3, 0))
    return np.unique(np.round(np.asarray(pts, float), 14), axis=0).T


def _gf_rounded_polygon(shape, v, w):
    """A polygon with fillets for the GetFEM mesher. A reflex corner: the
    quadrilateral K (center of the arc, tangent points pushed a little into
    the polygon, the vertex pushed as much) less the disk of the arc is added
    to the polygon. A convex corner: the polygon is intersected with the
    disk united with the three half planes that leave out the corner: beyond
    the radii from the center to the tangent points, or beyond the vertex
    (the corner region of the disk: between these radii, before the vertex).
    No boundary of the parts coincides with another one (the GetFEM mesher
    projects badly on such unions)."""
    pts = _points(shape)
    _, found, _ = G.fillets(pts, G.fillet_radii(shape, len(pts)))
    V = np.asarray(pts, float)
    w(f"# polygon with rounded vertices {[f.vertex for f in found]}")
    w(f"{v} = gf.MesherObject('polygon', np.array({pts}).T)")
    for f in found:
        c, t1, t2 = (np.asarray(x, float) for x in (f.center, f.t1, f.t2))
        vx = V[f.vertex - 1]
        e1, e2 = (t1 - c) / f.radius, (t2 - c) / f.radius
        ball = f"gf.MesherObject('ball', {list(f.center)!r}, {f.radius!r})"
        if f.convex:
            keep = [f"gf.MesherObject('half space', {c.tolist()!r}, {(-e1).tolist()!r})",
                    f"gf.MesherObject('half space', {c.tolist()!r}, {(-e2).tolist()!r})",
                    f"gf.MesherObject('half space', {vx.tolist()!r}, {(vx - c).tolist()!r})", ball]
            w(f"{v} = gf.MesherObject('intersect', {v}, gf.MesherObject('union', {', '.join(keep)}))")
        else:
            delta = 0.1 * f.radius
            half = math.acos(max(-1.0, min(1.0, float(e1 @ e2)))) / 2      # half the angle at the center
            out = vx + delta / max(math.cos(half), 1e-9) * (vx - c) / np.linalg.norm(vx - c)
            K = [c.tolist(), (t1 + delta * e1).tolist(), out.tolist(), (t2 + delta * e2).tolist()]
            w(f"{v} = gf.MesherObject('union', {v}, gf.MesherObject('set minus', "
              f"gf.MesherObject('polygon', np.array({K!r}).T), {ball}))")


def _box_as_shape(src):
    return {"type": "rectangle", "min": src["origin"],
            "max": [o + s for o, s in zip(src["origin"], src["size"])]}


def _source_shape(src):
    """The CSG tree of a box or csg source, its transforms and arrays applied
    (transforms.flatten)."""
    if src["type"] == "box":
        return _box_as_shape(src)
    return T.flatten(src["shape"], src.get("dim", 2))


def _build_getfem(model, geom, mesh, w):
    src = geom["source"]
    shape = _source_shape(src)
    w("# Implicit description of the geometry (MesherObject)")
    top = _gf_shape(shape, w)
    h, K = float(mesh['h']), int(mesh.get('order', 1))
    V = G.interface_points(shape, h) if src["type"] == "csg" and src.get("dim") == 2 else np.zeros((2, 0))
    if src["type"] == "csg" and src.get("dim") == 2:     # the ends of the fillets: vertices, so that
        J = G.fillet_ends(shape)                          # each face is on a side or on an arc
        if len(J):
            V = np.hstack([V, np.array([p for p in J if all(np.linalg.norm(p - q) > 1e-9 * h for q in V.T)],
                                       float).T.reshape(2, -1)])
    elif src["type"] == "csg" and src.get("dim") == 3:
        V = _fillet_lines_3d(shape, h)
    w("# GetFEM experimental mesher: element size h, geometric degree K")
    if V.shape[1] == 0:
        w(f"m = gf.Mesh('generate', {top}, {h!r}, {K})")
        return
    w("# Fixed vertices along the interfaces between regions (sides of shapes")
    w("# inside the domain), and at the ends of the fillets: the elements follow")
    w("# them, each one is in one region, each boundary face on one side")
    rows = ", ".join("[" + ", ".join(f"{x:.12g}" for x in row) + "]" for row in V)
    w(f"interface_vertices = np.array([{rows}])")
    w(f"m = gf.Mesh('generate', {top}, {h!r}, {K}, interface_vertices)")


def _occ_shape(shape, dim, w) -> str:
    """Lines creating an OpenCASCADE entity in Gmsh; return a list of (dim, tag)."""
    t = shape["type"]
    parts = [_occ_shape(s, dim, w) for s in shape["of"]] if t in OCC_BOOLEANS else []
    v = w.var("e")
    occ = "gmsh.model.occ"
    if t == "rectangle":
        lo, hi = _floats(shape["min"]), _floats(shape["max"])
        d = [b - a for a, b in zip(lo, hi)]
        if dim == 2:
            w(f"{v} = [(2, {occ}.addRectangle({lo[0]!r}, {lo[1]!r}, 0.0, {d[0]!r}, {d[1]!r}))]")
        else:
            w(f"{v} = [(3, {occ}.addBox({lo[0]!r}, {lo[1]!r}, {lo[2]!r}, {d[0]!r}, {d[1]!r}, {d[2]!r}))]")
    elif t == "ball":
        c, r = _floats(shape["center"]), float(shape["radius"])
        if dim == 2:
            w(f"{v} = [(2, {occ}.addDisk({c[0]!r}, {c[1]!r}, 0.0, {r!r}, {r!r}))]")
        else:
            w(f"{v} = [(3, {occ}.addSphere({c[0]!r}, {c[1]!r}, {c[2]!r}, {r!r}))]")
    elif t == "ellipse":
        (cx, cy), (r1, r2) = _floats(shape["center"]), _floats(shape["radii"])
        th = math.radians(float(shape.get("angle", 0.0)))
        if r1 < r2:                    # OpenCASCADE wants the major semi-axis along x first
            r1, r2, th = r2, r1, th + math.pi / 2
        w(f"{v} = [(2, {occ}.addDisk({cx!r}, {cy!r}, 0.0, {r1!r}, {r2!r}))]")
        if th:
            w(f"{occ}.rotate({v}, {cx!r}, {cy!r}, 0.0, 0.0, 0.0, 1.0, {th!r})")
    elif t == "polygon" and G.fillet_radii(shape, len(shape["points"])):
        outline, found, _ = G.fillets(_points(shape), G.fillet_radii(shape, len(shape["points"])))
        centers = {tuple(f.t1): list(f.center) for f in found}
        w(f"# polygon with rounded vertices {[f.vertex for f in found]}: lines and arcs of circle")
        w(f"_p = [{occ}.addPoint(x, y, 0.0) for x, y in {outline}]")
        w(f"_arcs = {{{', '.join(f'{outline.index(list(t))}: {c!r}' for t, c in centers.items())}}}")
        w("_l = []")
        w("for i in range(len(_p)):")
        w("    j = (i + 1) % len(_p)")
        w("    if i in _arcs:")
        w(f"        _c = {occ}.addPoint(_arcs[i][0], _arcs[i][1], 0.0)")
        w(f"        _l.append({occ}.addCircleArc(_p[i], _c, _p[j]))")
        w("    else:")
        w(f"        _l.append({occ}.addLine(_p[i], _p[j]))")
        w(f"{v} = [(2, {occ}.addPlaneSurface([{occ}.addCurveLoop(_l)]))]")
    elif t == "polygon":
        w(f"_p = [{occ}.addPoint(x, y, 0.0) for x, y in {_points(shape)}]")
        w(f"_l = [{occ}.addLine(_p[i], _p[(i + 1) % len(_p)]) for i in range(len(_p))]")
        w(f"{v} = [(2, {occ}.addPlaneSurface([{occ}.addCurveLoop(_l)]))]")
    elif t == "cylinder":
        o, a = _floats(shape["origin"]), _floats(shape["axis"])
        n = sum(x * x for x in a) ** 0.5
        L = float(shape["length"])
        d = [x / n * L for x in a]
        w(f"{v} = [(3, {occ}.addCylinder({o[0]!r}, {o[1]!r}, {o[2]!r}, "
          f"{d[0]!r}, {d[1]!r}, {d[2]!r}, {float(shape['radius'])!r}))]")
    elif t == "extrusion":
        p = _occ_shape(shape["profile"], 2, w)
        z0, z1 = _floats(shape["z"])
        w(f"{occ}.translate({p}, 0.0, 0.0, {z0!r})")
        w(f"{v} = [e for e in {occ}.extrude({p}, 0.0, 0.0, {z1 - z0!r}) if e[0] == 3]")
    elif t == "revolution":
        # the profile, drawn in the (x, y) plane for (r, z), is turned into
        # the (x, z) plane, then about the z axis
        p = _occ_shape(G3.effective_profile(shape), 2, w)
        w(f"{occ}.rotate({p}, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, math.pi / 2)")
        w(f"{v} = [e for e in {occ}.revolve({p}, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2 * math.pi) if e[0] == 3]")
    elif len(parts) == 1:                 # a boolean of one shape: that shape
        w(f"{v} = {parts[0]}")
    elif t == "intersection":
        # OpenCASCADE intersects the object with the union of the tools:
        # intersect the shapes two by two
        w(f"{v} = {parts[0]}")
        for q in parts[1:]:
            w(f"{v}, _ = {occ}.intersect({v}, {q})")
    else:          # fuse(a, b + c), cut(a, b + c): a with or without b and c
        w(f"{v}, _ = {occ}.{OCC_BOOLEANS[t]}({parts[0]}, {' + '.join(parts[1:])})")
    if t not in OCC_BOOLEANS and dim == 3:
        _occ_place(shape, v, w)
    return v


def _occ_place(shape, v, w):
    """A 3D primitive in its frame (the work plane of an extrusion or a
    revolution, the transforms and arrays above it): built in the frame,
    then turned and moved there."""
    fr = G3.frame(shape)
    if fr is None:
        return
    occ = "gmsh.model.occ"
    O, R = fr
    axis, angle = G3.rotation_vector(R)
    w("# placed in its frame")
    if angle:
        w(f"{occ}.rotate({v}, 0.0, 0.0, 0.0, {axis[0]!r}, {axis[1]!r}, {axis[2]!r}, {angle!r})")
    w(f"{occ}.translate({v}, {O[0]!r}, {O[1]!r}, {O[2]!r})")


def _ng_place(shape, v, w):
    fr = G3.frame(shape)
    if fr is None:
        return
    O, R = fr
    axis, angle = G3.rotation_vector(R)
    w("# placed in its frame")
    if angle:
        w(f"{v} = {v}.Rotate(Axis((0, 0, 0), Dir({axis[0]!r}, {axis[1]!r}, {axis[2]!r})), "
          f"{math.degrees(angle)!r})")
    w(f"{v} = {v}.Move(Vec({O[0]!r}, {O[1]!r}, {O[2]!r}))")


def _build_gmsh(model, geom, mesh, w, msh_name):
    src = geom["source"]
    w("import math")
    w("import gmsh")
    w("")
    w("gmsh.initialize()")
    w("gmsh.option.setNumber('General.Terminal', 0)")
    w("gmsh.model.add('geometry')")
    if src["type"] == "cad":
        w(f"entities = gmsh.model.occ.importShapes({str(M.resolve_file(model, src['file']))!r})")
        w("gmsh.model.occ.synchronize()")
        w("dim = max(d for d, _ in entities)")
    else:
        dim = len(src["size"]) if src["type"] == "box" else src["dim"]
        shape = _source_shape(src)
        _occ_shape(shape, dim, w)
        w("gmsh.model.occ.synchronize()")
        w(f"dim = {dim}")
    if mesh.get("element") == "quad":
        w("# Quadrangles: Frontal-Delaunay for quads, Blossom recombination, then one")
        w("# subdivision, which guarantees quadrangles only (and halves the size:")
        w("# the size asked for is doubled before)")
        w("gmsh.option.setNumber('Mesh.Algorithm', 8)")
        w("gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)")
        w("gmsh.option.setNumber('Mesh.RecombineAll', 1)")
        w("gmsh.option.setNumber('Mesh.SubdivisionAlgorithm', 1)")
    size = float(mesh["size"]) * (2.0 if mesh.get("element") == "quad" else 1.0)
    w("# A single physical group (the domain): Gmsh then only writes elements of")
    w("# maximal dimension; boundaries come from the selections, further down.")
    w("gmsh.model.addPhysicalGroup(dim, [t for _, t in gmsh.model.getEntities(dim)], 1)")
    w(f"gmsh.option.setNumber('Mesh.MeshSizeMin', {size!r})")
    w(f"gmsh.option.setNumber('Mesh.MeshSizeMax', {size!r})")
    w("gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)")
    w("gmsh.model.mesh.generate(dim)")
    w(f"gmsh.model.mesh.setOrder({int(mesh.get('order', 1))})")
    w(f"gmsh.write({msh_name!r})")
    w("gmsh.finalize()")
    w(f"m = gf.Mesh('import', 'gmsh', {msh_name!r})")
    _drop_z(w)


def _ng_shape(shape, dim, w) -> str:
    """Lines creating a shape with the OpenCASCADE interface of Netgen
    (netgen.occ); return its variable name."""
    t = shape["type"]
    parts = [_ng_shape(s, dim, w) for s in shape["of"]] if t in OCC_BOOLEANS else []
    v = w.var("s")
    if t == "rectangle":
        lo, hi = _floats(shape["min"]), _floats(shape["max"])
        if dim == 2:
            w(f"{v} = WorkPlane().MoveTo({lo[0]!r}, {lo[1]!r}).Rectangle({hi[0] - lo[0]!r}, "
              f"{hi[1] - lo[1]!r}).Face()")
        else:
            w(f"{v} = Box(Pnt({lo[0]!r}, {lo[1]!r}, {lo[2]!r}), Pnt({hi[0]!r}, {hi[1]!r}, {hi[2]!r}))")
    elif t == "ball":
        c, r = _floats(shape["center"]), float(shape["radius"])
        if dim == 2:
            w(f"{v} = WorkPlane().Circle({c[0]!r}, {c[1]!r}, {r!r}).Face()")
        else:
            w(f"{v} = Sphere(Pnt({c[0]!r}, {c[1]!r}, {c[2]!r}), {r!r})")
    elif t == "polygon":
        pts = _points(shape)
        radii = G.fillet_radii(shape, len(pts))
        if radii:                          # the fillets: arcs tangent to the sides, through their ends
            outline, found, _ = G.fillets(pts, radii)
            arcs = {tuple(f.t1): f for f in found}
            w(f"_wp = WorkPlane().MoveTo({outline[0][0]!r}, {outline[0][1]!r})")
            n = len(outline)
            for i in range(n):
                q = outline[(i + 1) % n]
                f = arcs.get(tuple(outline[i]))
                if f is not None:
                    w(f"_wp.ArcTo({q[0]!r}, {q[1]!r}, ({-math.sin(f.theta0) * math.copysign(1, f.sweep)!r}, "
                      f"{math.cos(f.theta0) * math.copysign(1, f.sweep)!r}))")
                elif i < n - 1:
                    w(f"_wp.LineTo({q[0]!r}, {q[1]!r})")
            w(f"{v} = _wp.Close().Face()")
        else:
            w(f"_wp = WorkPlane().MoveTo({pts[0][0]!r}, {pts[0][1]!r})")
            for x, y in pts[1:]:
                w(f"_wp.LineTo({x!r}, {y!r})")
            w(f"{v} = _wp.Close().Face()")
    elif t == "cylinder":
        o, a = _floats(shape["origin"]), np.asarray(_floats(shape["axis"]))
        a = (a / np.linalg.norm(a)).tolist()
        w(f"{v} = Cylinder(Pnt({o[0]!r}, {o[1]!r}, {o[2]!r}), Vec({a[0]!r}, {a[1]!r}, {a[2]!r}), "
          f"{float(shape['radius'])!r}, {float(shape['length'])!r})")
    elif t == "extrusion":
        p = _ng_shape(shape["profile"], 2, w)
        z0, z1 = _floats(shape["z"])
        w(f"{v} = {p}.Extrude({z1 - z0!r}).Move(Vec(0, 0, {z0!r}))")
    elif t == "revolution":
        p = _ng_shape(G3.effective_profile(shape), 2, w)
        w("# the profile, drawn in the (x, y) plane for (r, z), turned into the (x, z) plane")
        w(f"{v} = {p}.Rotate(Axis((0, 0, 0), X), 90).Revolve(Axis((0, 0, 0), Z), 360)")
    elif t in OCC_BOOLEANS:
        op = {"union": "+", "intersection": "*", "difference": "-"}[t]
        w(f"{v} = " + f" {op} ".join(parts))
    else:
        raise ValueError(f"{t!r} shapes cannot be meshed by Netgen yet (use Gmsh)")
    if t not in OCC_BOOLEANS and dim == 3:
        _ng_place(shape, v, w)
    return v


def _build_netgen(model, geom, mesh, w, msh_name):
    src = geom["source"]
    w("from netgen.occ import *")
    if src["type"] == "cad":
        w(f"geo = OCCGeometry({str(M.resolve_file(model, src['file']))!r})")
    else:
        dim = len(src["size"]) if src["type"] == "box" else src["dim"]
        shape = _source_shape(src)
        top = _ng_shape(shape, dim, w)
        w(f"geo = OCCGeometry({top}, dim={dim})")
    quad = mesh.get("element") == "quad"
    w("# Netgen: element size, quadrangles (quad dominated) if asked")
    w(f"ngmesh = geo.GenerateMesh(maxh={float(mesh['size'])!r}" + (", quad_dominated=True)" if quad else ")"))
    if int(mesh.get("order", 1)) == 2:
        w("ngmesh.SecondOrder()")
    w(f"ngmesh.Export({msh_name!r}, 'Gmsh2 Format')")
    w(f"m = gf.Mesh('import', 'gmsh', {msh_name!r})")
    _drop_z(w)


def _build_layered(model, geom, mesh, w, basename):
    """A 3D extrusion or revolution meshed in layers: the 2D profile is
    meshed (triangles or quadrangles), then extruded or turned in layers of
    prisms or hexahedra (gf.Mesh('prismatic'), its points then mapped)."""
    shape = M.swept_root(T.flatten(geom["source"]["shape"], 3))
    n, K = int(mesh["layers"]), int(mesh.get("order", 1))
    profile = G3.effective_profile(shape)
    g2 = {"name": geom["name"], "source": {"type": "csg", "dim": 2, "shape": profile}, "selections": {}}
    m2 = {k: v for k, v in mesh.items() if k != "layers"}
    w("# The profile (2D)")
    if mesh["mesher"] == "getfem":
        _build_getfem(model, g2, m2, w)
    elif mesh["mesher"] == "gmsh":
        _build_gmsh(model, g2, m2, w, f"{basename}_profile.msh")
    else:
        _build_netgen(model, g2, m2, w, f"{basename}_profile.msh")
    w(f"# {n} layers in [0, 1] along a third coordinate t (geometric degree {K} along t)")
    w(f"m = gf.Mesh('prismatic', m, {n}, {K})")
    fr = G3.frame(shape)

    def place():
        if fr is not None:                # in its work plane
            w("# placed in its frame: x -> O + R x")
            w(f"m.transform({fr[1].tolist()!r})")
            w(f"m.translate({fr[0].tolist()!r})")
    if shape["type"] == "extrusion":
        z0, z1 = _floats(shape["z"])
        w(f"# extrusion: z = {z0!r} + {z1 - z0!r} t")
        w(f"m.transform([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, {z1 - z0!r}]])")
        w(f"m.translate([0.0, 0.0, {z0!r}])")
        place()
        return
    w("# revolution: (r, z, t) -> (r cos a, r sin a, z), a = -2 pi t (a direct basis);")
    w("# the elements are added again, which merges the points of the seam (t = 0, 1)")
    w("P = m.pts()")
    w("a = -2 * np.pi * P[2]")
    w("Q = np.vstack([P[0] * np.cos(a), P[0] * np.sin(a), P[1]])")
    w("gts, gidx = m.geotrans(m.cvid())")
    w("gts = list(gts) if isinstance(gts, (list, tuple)) else [gts]")
    w("gidx = np.atleast_1d(gidx)")
    w("pids, idx = m.pid_from_cvid(m.cvid())")
    w("mr = gf.Mesh('empty', 3)")
    w("for k, gt in enumerate(gts):")
    w("    cvs = np.flatnonzero(gidx == k)")
    w("    ids = np.array([pids[idx[c]:idx[c + 1]] for c in cvs]).T")
    w("    mr.add_convex(gt, Q[:, ids])")
    w("m = mr")
    place()


def _build_curve(model, geom, mesh, w):
    """A curve: its chain divided into elements of the element size."""
    from . import manifold
    src = geom["source"]
    K = int(mesh.get("order", 1))
    for line in manifold.script_functions():
        w(line)
    w(f"pieces = {manifold.chain_pieces(src)!r}")
    w(f"X = chain_elements(pieces, {float(mesh['h'])!r}, {K})")
    w(f"m = gf.Mesh('empty', {int(src['dim'])})")
    w(f"m.add_convex(gf.GeoTrans('GT_PK(1,{K})'), X)")


def _build_swept(model, geom, mesh, w):
    """A swept surface: the 1D mesh of its curve (in the plane (a, b)), in
    layers along t (gf.Mesh('prismatic')), mapped to the extrusion or the
    revolution (the seam of a whole turn merged)."""
    from . import manifold
    src = geom["source"]
    K, h = int(mesh.get("order", 1)), float(mesh["h"])
    n = max(3 if manifold.full_turn(src) else 1, int(math.ceil(manifold.sweep_length(src) / h)))
    _build_curve(model, {"source": dict(src, type="curve", dim=2)}, mesh, w)
    w(manifold.REMESH_CODE)
    w(f"m = gf.Mesh('prismatic', m, {n}, {K})")
    w("P = m.pts()")
    w("a, b, t = P")
    if src["sweep"] == "revolution":
        a0, a1 = (math.radians(float(v)) for v in src.get("angle", [0.0, 360.0]))
        w(f"th = {a0!r} + {a1 - a0!r} * t")
        w("Q = np.vstack([a * np.cos(th), a * np.sin(th), b])")
    else:
        z0, z1 = (float(v) for v in src.get("z", [0.0, 1.0]))
        w(f"Q = np.vstack([a, b, {z0!r} + {z1 - z0!r} * t])")
    w("m, _ = remesh(m, Q, 3)")


def _build_flat(model, geom, mesh, w, basename):
    """The flat domain of a surface, meshed in 2D (mapped to 3D after the
    selections, which are defined on it)."""
    g2 = {"name": geom["name"], "source": {"type": "csg", "dim": 2, "shape": T.flatten(geom["source"]["shape"], 2)},
          "selections": geom.get("selections", {})}
    w("# The flat domain (local coordinates x, y)")
    if mesh["mesher"] == "getfem":
        _build_getfem(model, g2, mesh, w)
    elif mesh["mesher"] == "gmsh":
        _build_gmsh(model, g2, mesh, w, f"{basename}_gmsh.msh")
    else:
        _build_netgen(model, g2, mesh, w, f"{basename}_netgen.msh")


def _map_flat(model, geom, w):
    from . import manifold
    src = geom["source"]
    vals, _ = P.values(model)
    o, e1, e2, n = manifold.frame(src)
    w("")
    w("# The flat domain mapped to the surface: origin + x e1 + y e2 + height(x, y) n")
    w(manifold.REMESH_CODE)
    w("x, y = m.pts()")
    w(f"height = {manifold.height_code(src.get('height', 0.0), vals)}")
    w(f"o, e1, e2, n = (np.array(v)[:, None] for v in ({o.tolist()!r}, {e1.tolist()!r}, {e2.tolist()!r}, "
      f"{n.tolist()!r}))")
    w("m, new = remesh(m, o + e1 * x + e2 * y + n * height, 3)")
    w("SEL = remap_selections(SEL, new)")


def _build_import(model, geom, mesh, w):
    w(f"m = gf.Mesh('import', 'gmsh', {str(M.resolve_file(model, geom['source']['file']))!r})")
    _drop_z(w)


def _drop_z(w):
    w("# Gmsh always writes 3D points: drop z for a planar mesh")
    w("if m.pts().shape[0] == 3 and np.allclose(m.pts()[2], 0.0):")
    w("    m.transform([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])")


# ------------------------------------------------------------------ selections

def _faces_code(name, crit, w, shape):
    """A 'faces' selection: the boundary faces on faces of 3D primitives."""
    specs = [G3.face_spec(shape, ref) for ref in crit["faces"]]
    w(f"# {name}: " + ", ".join(G3.face_label(shape, ref) for ref in crit["faces"]))
    w(f"SEL[{name!r}] = ('boundary', faces_on_faces({specs!r}))")


def _selection_code(name, sel, w, shape=None):
    kind, c = sel["kind"], sel.get("criteria", {})
    t = c.get("type")
    if name == M.ALL:
        return
    if name == M.BOUNDARY:
        w(f"SEL[{name!r}] = ('boundary', m.outer_faces(TDIM))")
    elif t == "plane":
        w(f"SEL[{name!r}] = ('boundary', faces_in_plane({_floats(c['point'])}, {_floats(c['normal'])}))")
    elif t == "normal":
        angle = float(c.get("angle", 1.0))
        w(f"SEL[{name!r}] = ('boundary', m.outer_faces_with_direction("
          f"{_floats(c['direction'])}, np.radians({angle!r}), TDIM))")
    elif t == "in_box" and kind == M.BOUNDARY:
        w(f"SEL[{name!r}] = ('boundary', m.outer_faces_in_box({_floats(c['min'])}, {_floats(c['max'])}, TDIM))")
    elif t == "in_box":
        w(f"SEL[{name!r}] = ('domain', cvs(m.convexes_in_box({_floats(c['min'])}, {_floats(c['max'])})))")
    elif t == "in_ball":
        w(f"SEL[{name!r}] = ('boundary', m.outer_faces_in_ball({_floats(c['center'])}, "
          f"{float(c['radius'])!r}, TDIM))")
    elif t == "physical":
        tag = int(c["tag"])
        w(f"if {tag} not in imported:")
        w(f"    raise RuntimeError('physical group {tag} missing from the file (selection {name})')")
        if kind == M.BOUNDARY:
            w(f"SEL[{name!r}] = ('boundary', imported[{tag}])")
        else:
            w(f"SEL[{name!r}] = ('domain', cvs(imported[{tag}]))")
    elif t == "edges":
        # sides of primitives, resolved here into the parts on the boundary
        all_pieces = G.pieces(shape)
        found = [pc.criterion() for ref in c["edges"] for pc in G.pieces_of(shape, ref, all_pieces)]
        labels = ", ".join(G.side_label(shape, ref) for ref in c["edges"])
        w(f"# {name}: {labels}")
        w(f"SEL[{name!r}] = ('boundary', faces_on_pieces({found!r}))")
    elif t == "regions":
        prims = [(path, p) for path, p in G.primitives(shape)]
        w(f"# {name}: " + "; ".join(G.region_label(shape, r) for r in c["regions"]))
        w("PRIMITIVES = [")
        for path, p in prims:
            if p["type"] == "rectangle":
                w(f"    ('rectangle', {_floats(p['min'])}, {_floats(p['max'])}),")
            elif p["type"] == "polygon":
                w(f"    ('polygon', {G.rounded_outline(p) if p.get('fillets') else _points(p)}, None),")
            elif p["type"] == "ellipse":
                w(f"    ('ellipse', {_floats(p['center'])}, ({_floats(p['radii'])}, "
                  f"{math.radians(float(p.get('angle', 0.0)))!r})),")
            else:
                w(f"    ('ball', {_floats(p['center'])}, {float(p['radius'])!r}),")
        w("]")
        paths = G.source_paths(shape)
        groups = [paths.index(tuple(path)) for path, _ in prims]
        bits = [[tuple(q) in {tuple(x) for x in reg} for q in paths] for reg in c["regions"]]
        if groups == list(range(len(prims))):
            w(f"SEL[{name!r}] = ('domain', cvs_in_regions(PRIMITIVES, {bits!r}))")
        else:                             # copies of a shape (arrays): one group of primitives
            w(f"SEL[{name!r}] = ('domain', cvs_in_regions(PRIMITIVES, {bits!r}, {groups!r}))")
    elif t == "union":
        parts = ", ".join(f"SEL[{o!r}][1]" for o in c["of"])
        if kind == M.BOUNDARY:
            w(f"SEL[{name!r}] = ('boundary', np.unique(np.hstack([{parts}]), axis=1))")
        else:
            w(f"SEL[{name!r}] = ('domain', np.unique(np.hstack([{parts}])))")


def mesh_script(model: dict, mesh: dict, basename: str, raw_key: str | None = None,
                reuse_raw: bool = False, standalone: bool = False) -> str:
    """Text of the meshing script; it writes <basename>.mesh and <basename>.json.

    The mesh before the selections ("raw" mesh, <basename>_raw.mesh) depends
    on the geometry and the mesh parameters only: with reuse_raw, it is loaded
    instead of being generated again, so that changing a selection does not
    run the mesher again. Otherwise it is saved, with raw_key.

    standalone: the meshing part of a script that goes on with the solve
    (runner.export_script): nothing is written, `m` and REGIONS are used."""
    geom = M.geometry_of(model, mesh)
    table = M.selection_table(model, geom)
    w = _W()
    w("#!/usr/bin/env python3")
    w(f"# Meshing script generated by getfem_gui: mesh {mesh['name']!r}, "
      f"geometry {geom['name']!r}")
    w("import json")
    w("import math")
    w("import os")
    w("")
    w("import numpy as np")
    w("import getfem as gf")
    w("")
    w("def cvs(a):")
    w("    \"\"\"Element numbers (first row if given an elements/faces array).\"\"\"")
    w("    a = np.asarray(a)")
    w("    return np.unique(a[0] if a.ndim == 2 else a)")
    w("")
    w("def faces_in_plane(point, normal):")
    w("    \"\"\"Boundary faces whose nodes all lie in the plane (point, normal).\"\"\"")
    w("    n = np.asarray(normal, float) / np.linalg.norm(normal)")
    w("    P = m.pts()")
    w("    tol = 1e-6 * max(np.ptp(P, axis=1).max(), 1e-300)")
    w("    faces = m.outer_faces_with_direction(n, np.radians(1.0), TDIM)")
    w("    faces = np.hstack([faces, m.outer_faces_with_direction(-n, np.radians(1.0), TDIM)])")
    w("    keep = [j for j in range(faces.shape[1])")
    w("            if np.all(np.abs((P[:, m.pid_in_faces(faces[:, j:j+1])].T - point) @ n) < tol)]")
    w("    return faces[:, keep]")

    w("")
    w("def on_piece(Q, piece, tol):")
    w("    \"\"\"Whether all the points Q (n x 2) lie on a segment or on an arc.\"\"\"")
    w("    if 'segment' in piece:")
    w("        a, b = (np.asarray(x, float) for x in piece['segment'])")
    w("        d = b - a")
    w("        L = np.linalg.norm(d)")
    w("        s = (Q - a) @ d / L**2")
    w("        h = np.abs((Q - a) @ np.array([-d[1], d[0]])) / L")
    w("        return bool(np.all(h < tol) and np.all(s > -tol / L) and np.all(s < 1 + tol / L))")
    w("    if 'ellipse' in piece:")
    w("        e = piece['ellipse']")
    w("        (r1, r2), th = e['radii'], e['angle']")
    w("        D = Q - np.asarray(e['center'], float)")
    w("        u = D[:, 0] * np.cos(th) + D[:, 1] * np.sin(th)")
    w("        v = -D[:, 0] * np.sin(th) + D[:, 1] * np.cos(th)")
    w("        if not np.all(np.abs(np.hypot(u / r1, v / r2) - 1) * min(r1, r2) < tol):")
    w("            return False")
    w("        t = (np.arctan2(v / r2, u / r1) / (2 * np.pi) - e['from']) % 1.0")
    w("        slack = tol / min(r1, r2)")
    w("        return bool(np.all((t <= e['to'] - e['from'] + slack) | (t >= 1.0 - slack)))")
    w("    arc = piece['arc']")
    w("    D = Q - np.asarray(arc['center'], float)")
    w("    if not np.all(np.abs(np.linalg.norm(D, axis=1) - arc['radius']) < tol):")
    w("        return False")
    w("    angle = (np.degrees(np.arctan2(D[:, 1], D[:, 0])) - arc['from']) % 360.0")
    w("    slack = np.degrees(tol / arc['radius'])")
    w("    return bool(np.all((angle <= arc['sweep'] + slack) | (angle >= 360.0 - slack)))")
    w("")
    w("# Faces of 3D primitives (the functions of geometry3d.py in the interface)")
    for fn in G3.FACE_FUNCTIONS:
        for line in inspect.getsource(fn).rstrip().splitlines():
            w(line)
        w("")
    w("")
    w("def faces_on_faces(specs):")
    w("    \"\"\"Boundary faces lying on one of the faces described by specs.\"\"\"")
    w("    P = m.pts()")
    w("    tol = 1e-6 * max(np.ptp(P, axis=1).max(), 1e-300)")
    w("    faces = m.outer_faces(TDIM)")
    w("    if faces.size == 0:")
    w("        return faces")
    w("    normals = m.normal_of_faces(faces)")
    w("    keep = [j for j in range(faces.shape[1])")
    w("            if any(on_face(s, P[:, m.pid_in_faces(faces[:, j:j+1])].T, normals[:, j], tol) for s in specs)]")
    w("    return faces[:, keep]")
    w("")
    w("def faces_on_pieces(pieces):")
    w("    \"\"\"Boundary faces whose nodes all lie on one of the pieces.\"\"\"")
    w("    P = m.pts()")
    w("    tol = 1e-6 * max(np.ptp(P, axis=1).max(), 1e-300)")
    w("    faces = m.outer_faces(TDIM)")
    w("    keep = [j for j in range(faces.shape[1])")
    w("            if any(on_piece(P[:, m.pid_in_faces(faces[:, j:j+1])].T, pc, tol) for pc in pieces)]")
    w("    return faces[:, keep]")

    w("")
    w("def cvs_in_regions(primitives, regions, groups=None):")
    w("    \"\"\"Elements whose center is in one of the regions; a region is given by")
    w("    which primitives contain it (one boolean per primitive, or per group of")
    w("    primitives: the copies of a shape).\"\"\"")
    w("    cv = m.cvid()")
    w("    pid, idx = m.pid_from_cvid(cv)")
    w("    C = np.add.reduceat(m.pts()[:2, pid], idx[:-1], axis=1) / np.diff(idx)")
    w("    inside = []")
    w("    for kind, a, b in primitives:")
    w("        if kind == 'rectangle':")
    w("            inside.append(np.all((C > np.c_[a]) & (C < np.c_[b]), axis=0))")
    w("        elif kind == 'ellipse':")
    w("            (r1, r2), th = b")
    w("            D = C - np.c_[a]")
    w("            u = D[0] * np.cos(th) + D[1] * np.sin(th)")
    w("            v = -D[0] * np.sin(th) + D[1] * np.cos(th)")
    w("            inside.append((u / r1)**2 + (v / r2)**2 < 1)")
    w("        elif kind == 'polygon':                 # crossing number")
    w("            V = np.array(a)")
    w("            W = np.roll(V, -1, axis=0)")
    w("            x, y = C[0][:, None], C[1][:, None]")
    w("            with np.errstate(divide='ignore', invalid='ignore'):")
    w("                xc = V[:, 0] + (y - V[:, 1]) * (W[:, 0] - V[:, 0]) / (W[:, 1] - V[:, 1])")
    w("            cross = ((V[:, 1] > y) != (W[:, 1] > y)) & (x < xc)")
    w("            inside.append(np.sum(cross, axis=1) % 2 == 1)")
    w("        else:")
    w("            inside.append(np.linalg.norm(C - np.c_[a], axis=0) < b)")
    w("    inside = np.array(inside).reshape(len(primitives), -1)")
    w("    if groups is not None:")
    w("        inside = np.array([np.any(inside[[i for i, g in enumerate(groups) if g == k]], axis=0)")
    w("                           for k in range(max(groups) + 1)])")
    w("    keep = np.zeros(cv.size, bool)")
    w("    for bits in regions:")
    w("        keep |= np.all(inside == np.c_[bits], axis=0)")
    w("    return cv[keep]")

    raw = f"{basename}_raw.mesh"
    if reuse_raw:
        w.section("Mesh (unchanged since the last meshing: only the selections changed)")
        w(f"m = gf.Mesh('load', {raw!r})")
    else:
        w.section(f"Mesh ({mesh['mesher']})")
        mesher = mesh["mesher"]
        stype = geom.get("source", {}).get("type")
        if stype == "curve":
            _build_curve(model, geom, mesh, w)
        elif stype == "swept_surface":
            _build_swept(model, geom, mesh, w)
        elif stype == "surface":
            _build_flat(model, geom, mesh, w, basename)
        elif mesh.get("layers"):
            _build_layered(model, geom, mesh, w, basename)
        elif mesher == "structured":
            _build_structured(model, geom, mesh, w)
        elif mesher == "getfem":
            _build_getfem(model, geom, mesh, w)
        elif mesher == "gmsh":
            _build_gmsh(model, geom, mesh, w, f"{basename}_gmsh.msh")
        elif mesher == "netgen":
            _build_netgen(model, geom, mesh, w, f"{basename}_netgen.msh")
        else:
            _build_import(model, geom, mesh, w)
        if raw_key is not None:
            w("# Kept to evaluate other selections without meshing again")
            w(f"m.save({raw!r})")
            w(f"with open({basename + '_raw.json'!r}, 'w', encoding='utf-8') as f:")
            w(f"    json.dump({{'key': {raw_key!r}}}, f)")

    w.section("Selections -> regions")
    from . import manifold
    tdim = manifold.TOPOLOGICAL_DIM.get(geom.get("source", {}).get("type"))
    w("TDIM = " + (f"{tdim}   # the dimension of the elements (lower than that of the space)"
                   if tdim and geom["source"]["type"] != "surface" else "m.dim()"))
    w("# Regions possibly present after an import (physical groups)")
    w("imported = {int(rg): m.region(int(rg)) for rg in np.atleast_1d(m.regions())}")
    w("SEL = {}")
    ordered = [n for n, s in table.items() if s.get("criteria", {}).get("type") != "union"]
    ordered += [n for n, s in table.items() if s.get("criteria", {}).get("type") == "union"]
    shape = G.shape_of(geom.get("source", {}))
    solid = G3.shape_of(geom.get("source", {}))
    for name in ordered:
        if table[name].get("criteria", {}).get("type") == "faces":
            _faces_code(name, table[name]["criteria"], w, solid)
        else:
            _selection_code(name, table[name], w, shape)
    user = [n for n, s in table.items() if not s["builtin"]]
    if geom.get("source", {}).get("type") == "surface":
        _map_flat(model, geom, w)
    w("")
    w(f"for name in {user!r}:")
    w("    if SEL[name][1].size == 0:")
    w("        raise RuntimeError(f\"selection '{name}' is empty on this mesh\")")
    w("for rg in imported:")
    w("    m.delete_region(rg)")
    w("REGIONS = {'all': -1}")
    w("KINDS = {'all': 'domain'}")
    w("for i, (name, (kind, ids)) in enumerate(SEL.items(), start=1):")
    w("    m.set_region(i, ids)")
    w("    REGIONS[name] = i")
    w("    KINDS[name] = kind")

    if standalone:
        w("print(f'mesh: {m.nbcvs()} elements, {m.nbpts()} points')")
        return "\n".join(w.lines) + "\n"

    w.section("Save")
    w(f"m.save({basename + '.mesh'!r})")
    w("")
    w("# Files for display: the mesh, and each selection (faces or elements)")
    view = f"{basename}_view"
    w(f"os.makedirs({view!r}, exist_ok=True)")
    w(f"m.export_to_vtu({view + '/mesh.vtu'!r})")
    w("VIEW = {'mesh': " + repr(view + "/mesh.vtu") + ", 'selections': {}}")
    w("# a curved mesh (geometric order 2): the selections refined to follow it")
    w("def _linear(gt):")
    w("    name = gt.char()")
    w("    return 'LINEAR' in name or name.rstrip(')').endswith(',1')")
    w("try:")
    w("    _gts = m.geotrans()")
    w("    _gts = _gts[0] if isinstance(_gts, tuple) else _gts")
    w("    NREF = 1 if all(_linear(g) for g in (_gts if isinstance(_gts, list) else [_gts])) else 4")
    w("except Exception:   # noqa: BLE001")
    w("    NREF = 1")
    w("for name, (kind, ids) in SEL.items():")
    w("    if ids.size:")
    w("        ids = ids if kind == 'boundary' else np.atleast_2d(ids)")
    w(f"        file = f'{view}/{{name}}.vtu'")
    w("        gf.Slice(('none',), m, NREF, ids).export_to_vtu(file)")
    w("        VIEW['selections'][name] = file")
    w("info = {")
    w(f"    'mesh': {mesh['name']!r},")
    w("    'dim': int(m.dim()),")
    w("    'nb_points': int(m.nbpts()),")
    w("    'nb_elements': int(m.nbcvs()),")
    w("    'regions': REGIONS,")
    w("    'kinds': KINDS,")
    w("    'sizes': {n: int(ids.shape[-1]) for n, (_, ids) in SEL.items()},")
    w("    'view': VIEW,")
    w("}")
    w(f"with open({basename + '.json'!r}, 'w', encoding='utf-8') as f:")
    w("    json.dump(info, f, indent=2)")
    w("print(f\"mesh {info['mesh']}: {info['nb_elements']} elements, {info['nb_points']} points\")")
    return "\n".join(w.lines) + "\n"


# ------------------------------------------------------------------ cache and execution

def raw_key(model: dict, mesh: dict) -> str:
    """Hash of what the mesh before the selections depends on: the geometry
    without its selections, and the mesh parameters."""
    geom = dict(M.geometry_of(model, mesh))
    geom.pop("selections", None)
    geom.pop("name", None)
    m = {k: v for k, v in mesh.items() if k not in ("name", "geometry")}
    payload = {"version": SCRIPT_VERSION, "source": geom, "mesh": m}
    src = geom.get("source", {})
    if "file" in src:
        path = M.resolve_file(model, src["file"])
        payload["file"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(model: dict, mesh: dict) -> str:
    """Hash of what the mesh depends on (geometry and mesh, with the values of
    the parameters: a parameter used elsewhere does not invalidate it)."""
    geom = M.geometry_of(model, mesh)
    payload = {"version": SCRIPT_VERSION, "geometry": geom, "mesh": mesh}
    src = geom.get("source", {})
    if "file" in src:
        path = M.resolve_file(model, src["file"])
        payload["file"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def error_line(stderr: str) -> str:
    """Most useful line of a subprocess error output, to be shown to the user.

    That is the Python exception line (e.g. "RuntimeError: ..."), which is not
    always the last line: GetFEM may print warnings to stderr after it.
    """
    lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
    line = lines[-1] if lines else ""
    for i in range(len(lines) - 1, -1, -1):
        if _EXCEPTION_LINE.match(lines[i]):
            # keep the continuation of multi-line messages, not GetFEM's noise
            rest = [l for l in lines[i + 1:] if not _NOISE_LINE.match(l)]
            line = " ".join([lines[i]] + rest)
            break
    if "No module named 'netgen'" in line:
        line += (" — Netgen is missing in the Python that runs GetFEM: install it "
                 "(pip install netgen-mesher) or choose another mesher for this mesh")
    if "No module named 'gmsh'" in line:
        line += (" — the Gmsh Python module is missing in the Python that runs GetFEM: install it "
                 "(pip install gmsh) or choose the GetFEM mesher for this mesh")
    missing = _MISSING_OBJECT.search(line)
    if missing and missing.group(1) in RECENT_MESHER_OBJECTS:
        line += (f" — this GetFEM has no '{missing.group(1)}' mesher object: it needs the patch "
                 f"\"{RECENT_MESHER_OBJECTS[missing.group(1)]}\" (rebuild and reinstall GetFEM, "
                 "its Python interface included), or mesh with Gmsh")
    return line


# Mesher objects added to GetFEM for the interface, and the patch that adds each one
RECENT_MESHER_OBJECTS = {
    "extrusion": "mesher: extrusion and revolution of 2D objects",
    "placed": "mesher: objects placed in a frame (rigid motion)",
    "revolution": "mesher: extrusion and revolution of 2D objects",
    "polygon": "mesher: polygons (not necessarily convex)",
    "ellipse": "mesher: a working ellipse",
}
_MISSING_OBJECT = re.compile(r"Bad command name: (\w+)")


_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt)\b.*:")
_NOISE_LINE = re.compile(r"^(Level \d+ Warning|Trace \d+ in)\b")


@dataclass
class MeshResult:
    ok: bool
    path: Path
    info: dict = field(default_factory=dict)
    cached: bool = False
    stderr: str = ""
    stale: bool = False      # built for an older version of the geometry or mesh

    @property
    def error(self) -> str:
        return error_line(self.stderr)

    def view_file(self, selection: str | None = None) -> Path | None:
        """VTU file for display: the mesh, or a selection (None if unavailable)."""
        view = self.info.get("view", {})
        rel = view.get("mesh") if selection is None else view.get("selections", {}).get(selection)
        return self.path.parent / rel if rel else None


@dataclass
class MeshJob:
    """A mesh to build: run `command` in `folder`, then call `finish`.

    Lets a caller (the GUI) run the process itself, e.g. to stream its output
    or to interrupt it; `ensure_mesh` is the blocking version.
    """
    mesh_name: str
    folder: Path
    command: list[str]
    key: str
    origin: dict | None = None   # what an out of date mesh may still show (see origin())

    @property
    def mesh_file(self) -> Path:
        return self.folder / f"{self.mesh_name}.mesh"

    @property
    def info_file(self) -> Path:
        return self.folder / f"{self.mesh_name}.json"

    def finish(self, returncode: int, stderr: str = "") -> MeshResult:
        if returncode != 0 or not self.info_file.exists():
            return MeshResult(False, self.mesh_file, stderr=stderr or f"exit code {returncode}")
        info = json.loads(self.info_file.read_text(encoding="utf-8"))
        info["key"] = self.key
        if self.origin is not None:
            info["origin"] = self.origin
        self.info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
        return MeshResult(True, self.mesh_file, info, stderr=stderr)


def mesh_folder(workdir: str | Path) -> Path:
    return Path(workdir).resolve() / MESH_DIR


def origin(model: dict, mesh: dict) -> dict:
    """What a mesh is built on, apart from the details: its geometry (name,
    kind, dimensions). A mesh out of date is still shown for the same origin
    (the geometry was edited), not for another one: a new mesh, or a new
    geometry, which takes the name of a deleted one."""
    geom = M.geometry_of(model, mesh) or {}
    try:
        dims = [M.geometry_dim(model, geom), M.topological_dim(model, geom)]
    except Exception:  # noqa: BLE001 (a file being edited)
        dims = None
    return {"geometry": mesh.get("geometry"), "type": geom.get("source", {}).get("type"), "dims": dims}


def cached_mesh(model: dict, mesh_name: str, workdir: str | Path,
                up_to_date: bool = True) -> MeshResult | None:
    """The mesh already built in workdir, if any (only if up to date by
    default; out of date, only if built on the same origin)."""
    model = P.resolved(M.with_defaults(model))
    mesh = M.find_mesh(model, mesh_name)
    if mesh is None:
        return None
    base = mesh_folder(workdir) / mesh_name
    info_file, mesh_file = base.with_suffix(".json"), base.with_suffix(".mesh")
    if not (info_file.exists() and mesh_file.exists()):
        return None
    try:
        info = json.loads(info_file.read_text(encoding="utf-8"))
        current = info.get("key") == cache_key(model, mesh)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not current and (up_to_date or info.get("origin") != origin(model, mesh)):
        return None
    return MeshResult(True, mesh_file, info, cached=True, stale=not current)


def prepare_mesh(model: dict, mesh_name: str, workdir: str | Path,
                 python: str | None = None, force: bool = False) -> MeshResult | MeshJob:
    """Return the cached mesh if it is up to date, otherwise write the meshing
    script and return the job to run."""
    model = P.resolved(M.with_defaults(model))
    mesh = M.find_mesh(model, mesh_name)
    if mesh is None:
        raise ValueError(f"unknown mesh: {mesh_name!r}")
    if not force:
        cached = cached_mesh(model, mesh_name, workdir)
        if cached is not None:
            return cached
    folder = mesh_folder(workdir)
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / f"{mesh_name}.py"
    rkey = raw_key(model, mesh)
    reuse = False
    raw_info = folder / f"{mesh_name}_raw.json"
    if not force and raw_info.exists() and (folder / f"{mesh_name}_raw.mesh").exists():
        try:
            reuse = json.loads(raw_info.read_text(encoding="utf-8")).get("key") == rkey
        except (OSError, ValueError):
            reuse = False
    if not reuse:
        raw_info.unlink(missing_ok=True)
    script.write_text(mesh_script(model, mesh, mesh_name, rkey, reuse), encoding="utf-8")
    (folder / f"{mesh_name}.json").unlink(missing_ok=True)
    return MeshJob(mesh_name, folder, [python or sys.executable, script.name],
                   cache_key(model, mesh), origin(model, mesh))


def ensure_mesh(model: dict, mesh_name: str, workdir: str | Path,
                python: str | None = None, force: bool = False) -> MeshResult:
    """Build the mesh in <workdir>/meshes/ unless it is already up to date."""
    job = prepare_mesh(model, mesh_name, workdir, python, force)
    if isinstance(job, MeshResult):
        return job
    proc = subprocess.run(job.command, cwd=job.folder, capture_output=True, text=True)
    return job.finish(proc.returncode, proc.stderr)
