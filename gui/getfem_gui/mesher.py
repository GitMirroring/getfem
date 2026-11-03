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

"""Meshing step: geometry + parameters -> GetFEM mesh with regions.

For each mesh of the model, a readable script is generated that:
  1. builds the mesh (structured, GetFEM mesher, Gmsh) or imports it;
  2. evaluates the selections of the geometry on this mesh;
  3. renumbers the regions 1, 2, ... in the order of the selections;
  4. saves the mesh in the native GetFEM format (regions included)
     together with a JSON information file (regions, number of elements, ...).

The result is cached: remeshing only happens when the geometry, the mesh
parameters or the source file have changed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import model as M

# To be incremented whenever the generated script changes, to invalidate caches.
SCRIPT_VERSION = 3

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
    elif t == "cylinder":
        w(f"{v} = gf.MesherObject('cylinder', {_floats(shape['origin'])}, {_floats(shape['axis'])}, "
          f"{float(shape['length'])!r}, {float(shape['radius'])!r})")
    else:
        w(f"{v} = gf.MesherObject({GF_SHAPES[t]!r}, {', '.join(parts)})")
    return v


def _box_as_shape(src):
    return {"type": "rectangle", "min": src["origin"],
            "max": [o + s for o, s in zip(src["origin"], src["size"])]}


def _build_getfem(model, geom, mesh, w):
    src = geom["source"]
    shape = _box_as_shape(src) if src["type"] == "box" else src["shape"]
    w("# Implicit description of the geometry (MesherObject)")
    top = _gf_shape(shape, w)
    w("# GetFEM experimental mesher: element size h, geometric degree K")
    w(f"m = gf.Mesh('generate', {top}, {float(mesh['h'])!r}, {int(mesh.get('order', 1))})")


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
    elif t == "cylinder":
        o, a = _floats(shape["origin"]), _floats(shape["axis"])
        n = sum(x * x for x in a) ** 0.5
        L = float(shape["length"])
        d = [x / n * L for x in a]
        w(f"{v} = [(3, {occ}.addCylinder({o[0]!r}, {o[1]!r}, {o[2]!r}, "
          f"{d[0]!r}, {d[1]!r}, {d[2]!r}, {float(shape['radius'])!r}))]")
    else:
        w(f"{v}, _ = {occ}.{OCC_BOOLEANS[t]}({parts[0]}, {' + '.join(parts[1:])})")
    return v


def _build_gmsh(model, geom, mesh, w, msh_name):
    src = geom["source"]
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
        shape = _box_as_shape(src) if src["type"] == "box" else src["shape"]
        _occ_shape(shape, dim, w)
        w("gmsh.model.occ.synchronize()")
        w(f"dim = {dim}")
    w("# A single physical group (the domain): Gmsh then only writes elements of")
    w("# maximal dimension; boundaries come from the selections, further down.")
    w("gmsh.model.addPhysicalGroup(dim, [t for _, t in gmsh.model.getEntities(dim)], 1)")
    w(f"gmsh.option.setNumber('Mesh.MeshSizeMin', {float(mesh['size'])!r})")
    w(f"gmsh.option.setNumber('Mesh.MeshSizeMax', {float(mesh['size'])!r})")
    w("gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)")
    w("gmsh.model.mesh.generate(dim)")
    w(f"gmsh.model.mesh.setOrder({int(mesh.get('order', 1))})")
    w(f"gmsh.write({msh_name!r})")
    w("gmsh.finalize()")
    w(f"m = gf.Mesh('import', 'gmsh', {msh_name!r})")
    _drop_z(w)


def _build_import(model, geom, mesh, w):
    w(f"m = gf.Mesh('import', 'gmsh', {str(M.resolve_file(model, geom['source']['file']))!r})")
    _drop_z(w)


def _drop_z(w):
    w("# Gmsh always writes 3D points: drop z for a planar mesh")
    w("if m.pts().shape[0] == 3 and np.allclose(m.pts()[2], 0.0):")
    w("    m.transform([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])")


# ------------------------------------------------------------------ selections

def _selection_code(name, sel, w):
    kind, c = sel["kind"], sel.get("criteria", {})
    t = c.get("type")
    if name == M.ALL:
        return
    if name == M.BOUNDARY:
        w(f"SEL[{name!r}] = ('boundary', m.outer_faces())")
    elif t == "plane":
        w(f"SEL[{name!r}] = ('boundary', faces_in_plane({_floats(c['point'])}, {_floats(c['normal'])}))")
    elif t == "normal":
        angle = float(c.get("angle", 1.0))
        w(f"SEL[{name!r}] = ('boundary', m.outer_faces_with_direction("
          f"{_floats(c['direction'])}, np.radians({angle!r})))")
    elif t == "in_box" and kind == M.BOUNDARY:
        w(f"SEL[{name!r}] = ('boundary', m.outer_faces_in_box({_floats(c['min'])}, {_floats(c['max'])}))")
    elif t == "in_box":
        w(f"SEL[{name!r}] = ('domain', cvs(m.convexes_in_box({_floats(c['min'])}, {_floats(c['max'])})))")
    elif t == "in_ball":
        w(f"SEL[{name!r}] = ('boundary', m.outer_faces_in_ball({_floats(c['center'])}, "
          f"{float(c['radius'])!r}))")
    elif t == "physical":
        tag = int(c["tag"])
        w(f"if {tag} not in imported:")
        w(f"    raise RuntimeError('physical group {tag} missing from the file (selection {name})')")
        if kind == M.BOUNDARY:
            w(f"SEL[{name!r}] = ('boundary', imported[{tag}])")
        else:
            w(f"SEL[{name!r}] = ('domain', cvs(imported[{tag}]))")
    elif t == "union":
        parts = ", ".join(f"SEL[{o!r}][1]" for o in c["of"])
        if kind == M.BOUNDARY:
            w(f"SEL[{name!r}] = ('boundary', np.unique(np.hstack([{parts}]), axis=1))")
        else:
            w(f"SEL[{name!r}] = ('domain', np.unique(np.hstack([{parts}])))")


def mesh_script(model: dict, mesh: dict, basename: str) -> str:
    """Text of the meshing script; it writes <basename>.mesh and <basename>.json."""
    geom = M.geometry_of(model, mesh)
    table = M.selection_table(model, geom)
    w = _W()
    w("#!/usr/bin/env python3")
    w(f"# Meshing script generated by getfem_gui: mesh {mesh['name']!r}, "
      f"geometry {geom['name']!r}")
    w("import json")
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
    w("    faces = m.outer_faces_with_direction(n, np.radians(1.0))")
    w("    faces = np.hstack([faces, m.outer_faces_with_direction(-n, np.radians(1.0))])")
    w("    keep = [j for j in range(faces.shape[1])")
    w("            if np.all(np.abs((P[:, m.pid_in_faces(faces[:, j:j+1])].T - point) @ n) < tol)]")
    w("    return faces[:, keep]")

    w.section(f"Mesh ({mesh['mesher']})")
    mesher = mesh["mesher"]
    if mesher == "structured":
        _build_structured(model, geom, mesh, w)
    elif mesher == "getfem":
        _build_getfem(model, geom, mesh, w)
    elif mesher == "gmsh":
        _build_gmsh(model, geom, mesh, w, f"{basename}_gmsh.msh")
    else:
        _build_import(model, geom, mesh, w)

    w.section("Selections -> regions")
    w("# Regions possibly present after an import (physical groups)")
    w("imported = {int(rg): m.region(int(rg)) for rg in np.atleast_1d(m.regions())}")
    w("SEL = {}")
    ordered = [n for n, s in table.items() if s.get("criteria", {}).get("type") != "union"]
    ordered += [n for n, s in table.items() if s.get("criteria", {}).get("type") == "union"]
    for name in ordered:
        _selection_code(name, table[name], w)
    user = [n for n, s in table.items() if not s["builtin"]]
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

    w.section("Save")
    w(f"m.save({basename + '.mesh'!r})")
    w("info = {")
    w(f"    'mesh': {mesh['name']!r},")
    w("    'dim': int(m.dim()),")
    w("    'nb_points': int(m.nbpts()),")
    w("    'nb_elements': int(m.nbcvs()),")
    w("    'regions': REGIONS,")
    w("    'kinds': KINDS,")
    w("    'sizes': {n: int(ids.shape[-1]) for n, (_, ids) in SEL.items()},")
    w("}")
    w(f"with open({basename + '.json'!r}, 'w', encoding='utf-8') as f:")
    w("    json.dump(info, f, indent=2)")
    w("print(json.dumps(info))")
    return "\n".join(w.lines) + "\n"


# ------------------------------------------------------------------ cache and execution

def cache_key(model: dict, mesh: dict) -> str:
    geom = M.geometry_of(model, mesh)
    payload = {"version": SCRIPT_VERSION, "geometry": geom, "mesh": mesh}
    src = geom.get("source", {})
    if "file" in src:
        path = M.resolve_file(model, src["file"])
        payload["file"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MeshResult:
    ok: bool
    path: Path
    info: dict = field(default_factory=dict)
    cached: bool = False
    stderr: str = ""

    @property
    def error(self) -> str:
        lines = [l for l in self.stderr.strip().splitlines() if l.strip()]
        return lines[-1] if lines else ""


def ensure_mesh(model: dict, mesh_name: str, workdir: str | Path,
                python: str | None = None, force: bool = False) -> MeshResult:
    """Build the mesh in <workdir>/meshes/ unless it is already up to date."""
    model = M.with_defaults(model)
    mesh = M.by_name(model["meshes"], mesh_name)
    if mesh is None:
        raise ValueError(f"unknown mesh: {mesh_name!r}")
    folder = Path(workdir).resolve() / MESH_DIR
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / mesh_name
    info_file, mesh_file = base.with_suffix(".json"), base.with_suffix(".mesh")
    key = cache_key(model, mesh)

    if not force and info_file.exists() and mesh_file.exists():
        info = json.loads(info_file.read_text(encoding="utf-8"))
        if info.get("key") == key:
            return MeshResult(True, mesh_file, info, cached=True)

    script = base.with_suffix(".py")
    script.write_text(mesh_script(model, mesh, mesh_name), encoding="utf-8")
    info_file.unlink(missing_ok=True)
    proc = subprocess.run([python or sys.executable, script.name], cwd=folder,
                          capture_output=True, text=True)
    if proc.returncode != 0 or not info_file.exists():
        return MeshResult(False, mesh_file, stderr=proc.stderr)
    info = json.loads(info_file.read_text(encoding="utf-8"))
    info["key"] = key
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return MeshResult(True, mesh_file, info)
