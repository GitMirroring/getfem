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

"""Export of geometries and meshes to other formats.

Geometries (box and CSG):
- 2D: SVG and DXF outlines (exact segments, arcs and ellipses);
- 3D: STL, OBJ, PLY, VTP surfaces (the exact preview surface, PyVista);
- 2D and 3D: STEP, IGES, BREP solids, built with OpenCASCADE by the Gmsh
  Python module (in a subprocess, as the meshing).

Meshes (built in the working folder): GetFEM (.mesh), VTK (.vtu, .vtk),
Gmsh (.msh, version 2.2, with the selections as physical groups) and Abaqus
(.inp, domain selections as element sets, boundary selections as node
sets). The mesh is read by a GetFEM subprocess, which writes its points,
elements and selections in a neutral JSON file; the files are then
written here, without GetFEM.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import geometry2d as G2, geometry3d as G3, mesher, model as M, parameters as P

GEOMETRY_FORMATS_2D = {".svg": "SVG outline", ".dxf": "DXF outline"}
GEOMETRY_FORMATS_3D = {".stl": "STL surface", ".obj": "OBJ surface", ".ply": "PLY surface",
                       ".vtp": "VTK surface"}
CAD_FORMATS = {".step": "STEP (Gmsh/OpenCASCADE)", ".stp": "STEP (Gmsh/OpenCASCADE)",
               ".iges": "IGES (Gmsh/OpenCASCADE)", ".igs": "IGES (Gmsh/OpenCASCADE)",
               ".brep": "BREP (Gmsh/OpenCASCADE)"}
MESH_FORMATS = {".mesh": "GetFEM mesh", ".vtu": "VTK unstructured grid", ".vtk": "VTK legacy",
                ".msh": "Gmsh 2.2", ".inp": "Abaqus input"}


class ExportError(Exception):
    pass


def geometry_formats(dim: int) -> dict[str, str]:
    return {**(GEOMETRY_FORMATS_2D if dim == 2 else GEOMETRY_FORMATS_3D if dim == 3 else {}),
            **(CAD_FORMATS if dim in (2, 3) else {})}


def file_filters(formats: dict[str, str]) -> str:
    """Qt file dialog filters, one per format."""
    seen, out = set(), []
    for ext, label in formats.items():
        if label in seen:
            continue
        seen.add(label)
        exts = " ".join(f"*{e}" for e, lab in formats.items() if lab == label)
        out.append(f"{label} ({exts})")
    return ";;".join(out)


def _geometry(model: dict, name: str) -> tuple[dict, dict, int]:
    """(resolved geometry, shape, dimension) of a box or CSG geometry."""
    r = P.resolved(M.with_defaults(model))
    geom = M.find_geometry(r, name)
    if geom is None:
        raise ExportError(f"unknown geometry {name!r}")
    gi = next(i for i, g in enumerate(r.get("geometries", [])) if g is geom) if geom in r.get("geometries", []) \
        else None
    errors = [e for e in M.validate(model)
              if e.startswith("parameters") or (gi is not None and e.startswith(f"geometries[{gi}]"))]
    if errors:
        raise ExportError("the geometry has errors:\n" + "\n".join(errors[:5]))
    src = geom["source"]
    if src.get("type") == "box":
        dim = len(src["size"])
        shape = {"type": "rectangle", "min": list(src["origin"]),
                 "max": [o + s for o, s in zip(src["origin"], src["size"])]}
    elif src.get("type") == "csg":
        dim, shape = src["dim"], src["shape"]
    else:
        raise ExportError(f"a geometry {src.get('type')!r} cannot be exported (only box and csg)")
    return geom, shape, dim


def export_geometry(model: dict, name: str, out: str | Path, python: str | None = None) -> Path:
    """Write the geometry `name` of the model; the format is given by the suffix."""
    out = Path(out)
    ext = out.suffix.lower()
    geom, shape, dim = _geometry(model, name)
    if ext not in geometry_formats(dim):
        raise ExportError(f"{ext or 'no suffix'}: not a format for a {dim}D geometry "
                          f"({', '.join(geometry_formats(dim))})")
    if ext in CAD_FORMATS:
        _export_cad(shape, dim, out, python)
    elif ext == ".svg":
        out.write_text(svg(shape), encoding="utf-8")
    elif ext == ".dxf":
        out.write_text(dxf(shape), encoding="utf-8")
    else:
        try:
            from .app import scene
        except ImportError as exc:
            raise ExportError(f"PyVista is needed for {ext} files ({exc})") from exc
        surf = scene.geometry_surface_3d(shape)
        P = np.asarray(surf.points, float)
        T = np.asarray(surf.faces).reshape(-1, 4)[:, 1:]
        if ext == ".stl":
            write_stl(out, P, T)
        elif ext == ".obj":
            out.write_text("# exported by the GetFEM GUI\n" + "".join(f"v {x:.12g} {y:.12g} {z:.12g}\n" for x, y, z in P)
                           + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in T), encoding="utf-8")
        elif ext == ".ply":
            out.write_text(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\n"
                           f"property float z\nelement face {len(T)}\nproperty list uchar int vertex_indices\n"
                           "end_header\n" + "".join(f"{x:.12g} {y:.12g} {z:.12g}\n" for x, y, z in P)
                           + "".join(f"3 {a} {b} {c}\n" for a, b, c in T), encoding="utf-8")
        else:
            clean = surf.copy()
            for k in list(clean.cell_data):
                del clean.cell_data[k]
            clean.field_data.clear()
            clean.save(str(out))
    return out


def write_stl(out: Path, P: np.ndarray, T: np.ndarray) -> None:
    """Binary STL file of a triangulated surface (outward normals)."""
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    N = np.cross(B - A, C - A)
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
    rec = np.zeros(len(T), dtype=[("n", "<f4", 3), ("a", "<f4", 3), ("b", "<f4", 3), ("c", "<f4", 3),
                                  ("attr", "<u2")])
    rec["n"], rec["a"], rec["b"], rec["c"] = N, A, B, C
    with open(out, "wb") as f:
        f.write(b"GetFEM GUI geometry".ljust(80, b" "))
        f.write(np.uint32(len(T)).tobytes())
        f.write(rec.tobytes())


# ------------------------------------------------------------- 2D outlines

def _outline(shape: dict) -> list:
    """The boundary pieces of a 2D shape: ("segment", a, b), ("arc", center,
    radius, angle0, angle1) or ("ellipse", center, r1, r2, angle, t0, t1)."""
    out = []
    for p in G2.pieces(shape):
        s = p.side
        if s.kind == "segment":
            a, b = s.points([p.t0, p.t1])
            out.append(("segment", a, b))
        elif s.kind == "arc":
            out.append(("arc", np.asarray(s.a, float), s.b[0], 2 * math.pi * p.t0, 2 * math.pi * p.t1))
        else:
            r1, r2, th = s.b
            out.append(("ellipse", np.asarray(s.a, float), r1, r2, th, 2 * math.pi * p.t0, 2 * math.pi * p.t1))
    return out


def svg(shape: dict) -> str:
    """SVG drawing of the outline of a 2D shape (y up, as in the model)."""
    lo, hi = G2.bounds(shape)
    size = float(max(hi - lo)) or 1.0
    m = 0.02 * size
    w, h = float(hi[0] - lo[0]) + 2 * m, float(hi[1] - lo[1]) + 2 * m
    fy = lambda y: float(hi[1] + m - y)          # noqa: E731  (SVG has y down)
    fx = lambda x: float(x - lo[0] + m)          # noqa: E731
    paths = []
    for piece in _outline(shape):
        if piece[0] == "segment":
            _, a, b = piece
            paths.append(f"M {fx(a[0]):.9g} {fy(a[1]):.9g} L {fx(b[0]):.9g} {fy(b[1]):.9g}")
            continue
        if piece[0] == "arc":
            _, c, r, a0, a1 = piece
            r1 = r2 = r
            th = 0.0
        else:
            _, c, r1, r2, th, a0, a1 = piece
        pt = lambda a: (c[0] + r1 * math.cos(a) * math.cos(th) - r2 * math.sin(a) * math.sin(th),  # noqa: E731
                        c[1] + r1 * math.cos(a) * math.sin(th) + r2 * math.sin(a) * math.cos(th))
        # in two halves at most (an SVG arc cannot be a whole ellipse)
        n = 2 if a1 - a0 > math.pi else 1
        cuts = [a0 + (a1 - a0) * k / n for k in range(n + 1)]
        x, y = pt(cuts[0])
        d = f"M {fx(x):.9g} {fy(y):.9g}"
        for k in range(n):
            x, y = pt(cuts[k + 1])
            large = 1 if cuts[k + 1] - cuts[k] > math.pi else 0
            # counterclockwise in the model: clockwise (sweep 0) with y down
            d += f" A {r1:.9g} {r2:.9g} {-math.degrees(th):.9g} {large} 0 {fx(x):.9g} {fy(y):.9g}"
        paths.append(d)
    body = "\n".join(f'  <path d="{d}"/>' for d in paths)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.9g}" height="{h:.9g}" '
            f'viewBox="0 0 {w:.9g} {h:.9g}">\n'
            f'<g fill="none" stroke="black" stroke-width="{size / 500:.9g}">\n{body}\n</g>\n</svg>\n')


def dxf(shape: dict) -> str:
    """DXF (R12) drawing of the outline of a 2D shape: LINE and ARC entities,
    the pieces of ellipses as polylines."""
    out = ["0", "SECTION", "2", "ENTITIES"]

    def entity(kind, **codes):
        out.extend(["0", kind, "8", "0"])
        for code, value in codes.items():
            out.extend([code[1:], f"{value:.12g}" if isinstance(value, float) else str(value)])

    for piece in _outline(shape):
        if piece[0] == "segment":
            _, a, b = piece
            entity("LINE", c10=float(a[0]), c20=float(a[1]), c30=0.0, c11=float(b[0]), c21=float(b[1]), c31=0.0)
        elif piece[0] == "arc" and piece[4] - piece[3] < 2 * math.pi - 1e-12:
            _, c, r, a0, a1 = piece
            entity("ARC", c10=float(c[0]), c20=float(c[1]), c30=0.0, c40=float(r),
                   c50=math.degrees(a0) % 360.0, c51=math.degrees(a1) % 360.0)
        elif piece[0] == "arc":
            _, c, r, _, _ = piece
            entity("CIRCLE", c10=float(c[0]), c20=float(c[1]), c30=0.0, c40=float(r))
        else:
            _, c, r1, r2, th, a0, a1 = piece
            n = max(16, int(math.ceil(96 * (a1 - a0) / (2 * math.pi))))
            out.extend(["0", "POLYLINE", "8", "0", "66", "1", "10", "0", "20", "0", "30", "0"])
            for a in np.linspace(a0, a1, n + 1):
                x = c[0] + r1 * math.cos(a) * math.cos(th) - r2 * math.sin(a) * math.sin(th)
                y = c[1] + r1 * math.cos(a) * math.sin(th) + r2 * math.sin(a) * math.cos(th)
                out.extend(["0", "VERTEX", "8", "0", "10", f"{x:.12g}", "20", f"{y:.12g}", "30", "0"])
            out.extend(["0", "SEQEND", "8", "0"])
    out.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(out) + "\n"


# -------------------------------------------------------------- CAD files

def cad_script(shape: dict, dim: int, out: str) -> str:
    """Gmsh script building the shape with OpenCASCADE and writing it."""
    w = mesher._W()
    w("import math")
    w("import gmsh")
    w("")
    w("gmsh.initialize()")
    w("gmsh.option.setNumber('General.Terminal', 0)")
    w("gmsh.model.add('geometry')")
    mesher._occ_shape(shape, dim, w)
    w("gmsh.model.occ.synchronize()")
    w(f"gmsh.write({out!r})")
    w("gmsh.finalize()")
    return "\n".join(w.lines) + "\n"


def _export_cad(shape, dim, out: Path, python):
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "export_cad.py"
        target = Path(tmp) / f"geometry{out.suffix.lower()}"
        script.write_text(cad_script(shape, dim, str(target)), encoding="utf-8")
        proc = subprocess.run([python or sys.executable, str(script)], capture_output=True, text=True, cwd=tmp)
        if proc.returncode != 0 or not target.exists():
            err = mesher.error_line(proc.stderr)
            if "No module named 'gmsh'" in proc.stderr:
                err = "the Gmsh Python module is needed for this format (pip install gmsh)"
            raise ExportError(err or f"exit code {proc.returncode}")
        shutil.copyfile(target, out)


# ------------------------------------------------------------------ meshes

# GetFEM geometric transformations: (Gmsh element type, order of the nodes
# for Gmsh, Abaqus element type, order for Abaqus, indices of the vertices)
ELEMENTS = {
    "GT_PK(1,1)": (1, [0, 1], "T2D2", [0, 1], [0, 1]),
    "GT_PK(1,2)": (8, [0, 2, 1], "T2D3", [0, 2, 1], [0, 2]),
    "GT_PK(2,1)": (2, [0, 1, 2], "CPS3", [0, 1, 2], [0, 1, 2]),
    "GT_PK(2,2)": (9, [0, 2, 5, 1, 4, 3], "CPS6", [0, 2, 5, 1, 4, 3], [0, 2, 5]),
    "GT_QK(2,1)": (3, [0, 1, 3, 2], "CPS4", [0, 1, 3, 2], [0, 1, 2, 3]),
    "GT_PK(3,1)": (4, [0, 1, 2, 3], "C3D4", [0, 1, 2, 3], [0, 1, 2, 3]),
    "GT_PK(3,2)": (11, [0, 2, 5, 9, 1, 4, 3, 6, 8, 7], "C3D10", [0, 2, 5, 9, 1, 4, 3, 6, 7, 8], [0, 2, 5, 9]),
    "GT_QK(3,1)": (5, [0, 1, 3, 2, 4, 5, 7, 6], "C3D8", [0, 1, 3, 2, 4, 5, 7, 6], list(range(8))),
}
ELEMENTS["GT_LINEAR_PRODUCT(GT_PK(1,1),GT_PK(1,1))"] = ELEMENTS["GT_QK(2,1)"]
ELEMENTS["GT_LINEAR_PRODUCT(GT_LINEAR_PRODUCT(GT_PK(1,1),GT_PK(1,1)),GT_PK(1,1))"] = ELEMENTS["GT_QK(3,1)"]
# Gmsh element types of the faces: (number of vertices, order) -> type
FACE_TYPES = {(1, 1): 15, (2, 1): 1, (2, 2): 8, (3, 1): 2, (3, 2): 9, (4, 1): 3}

MESH_DATA_SCRIPT = r'''
import json, sys
import numpy as np
import getfem as gf

mesh_file, info_file, out = sys.argv[1:4]
m = gf.Mesh('load', mesh_file)
info = json.load(open(info_file))
cvs = m.cvid()
gts, gidx = m.geotrans(cvs)          # the distinct transformations, the one of each element
if not isinstance(gts, (list, tuple)):
    gts = [gts]
names = [str(g.char()) for g in gts]
gidx = np.atleast_1d(gidx)
pids, idx = m.pid_from_cvid(cvs)
cells = [[names[int(gidx[k])], [int(p) for p in pids[idx[k]:idx[k + 1]]]] for k in range(len(cvs))]
index = {int(cv): k for k, cv in enumerate(cvs)}
regions = {}
for name, rid in info.get("regions", {}).items():
    if rid < 0 or rid not in m.regions():
        continue
    cvf = m.region(rid)
    kind = info.get("kinds", {}).get(name, "domain")
    if kind == "domain":
        regions[name] = {"kind": kind, "cells": sorted({index[int(c)] for c in cvf[0]})}
    else:
        faces = []
        for c, f in zip(cvf[0], cvf[1]):
            faces.append([index[int(c)], [int(p) for p in m.pid_in_faces(np.array([[c], [f]]))]])
        regions[name] = {"kind": kind, "faces": faces}
json.dump({"dim": int(m.dim()), "points": m.pts().T.tolist(), "cells": cells, "regions": regions},
          open(out, "w"))
'''

MESH_VTK_SCRIPT = r'''
import sys
import getfem as gf
m = gf.Mesh('load', sys.argv[1])
if sys.argv[2].endswith('.vtu'):
    m.export_to_vtu(sys.argv[2], 'ascii')
else:
    m.export_to_vtk(sys.argv[2], 'ascii')
'''


def _run(script: str, args, python=None):
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "export_mesh.py"
        f.write_text(script, encoding="utf-8")
        proc = subprocess.run([python or sys.executable, str(f), *map(str, args)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise ExportError(mesher.error_line(proc.stderr) or f"exit code {proc.returncode}")


def export_mesh(model: dict, name: str, workdir: str | Path, out: str | Path, python: str | None = None,
                build: bool = False) -> Path:
    """Write the mesh `name` (built in workdir; built there first if build)."""
    out = Path(out)
    ext = out.suffix.lower()
    if ext not in MESH_FORMATS:
        raise ExportError(f"{ext or 'no suffix'}: not a mesh format ({', '.join(MESH_FORMATS)})")
    res = mesher.ensure_mesh(model, name, workdir, python) if build else mesher.cached_mesh(model, name, workdir)
    if res is None:
        raise ExportError(f"the mesh {name!r} is not built (or not up to date): build it first")
    if not res.ok:
        raise ExportError(f"meshing failed: {res.error}")
    mesh_file, info_file = res.path, res.path.with_suffix(".json")
    if ext == ".mesh":
        shutil.copyfile(mesh_file, out)
    elif ext in (".vtu", ".vtk"):
        _run(MESH_VTK_SCRIPT, [mesh_file, out], python)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "mesh.json"
            _run(MESH_DATA_SCRIPT, [mesh_file, info_file, data_file], python)
            data = json.loads(data_file.read_text(encoding="utf-8"))
        out.write_text(msh(data) if ext == ".msh" else inp(data), encoding="utf-8")
    return out


def _element(gt: str) -> tuple:
    gt = gt.replace(" ", "")
    if gt not in ELEMENTS:
        raise ExportError(f"elements {gt} cannot be written in this format")
    return ELEMENTS[gt]


def _face_nodes(data: dict, cell: int, nodes: list[int]) -> tuple[list[int], int]:
    """Nodes of a face of a cell in the Gmsh order (vertices, then the
    middles of the edges 0-1, 1-2, 2-0 for a quadratic face), and its number
    of vertices."""
    gt, cn = data["cells"][cell]
    corners = {cn[i] for i in _element(gt)[4]}
    vert = [p for p in nodes if p in corners]
    others = [p for p in nodes if p not in corners]
    P = np.asarray(data["points"], float)
    if len(vert) == 4:                     # a quadrangle: its vertices in turn
        X = P[vert] - P[vert].mean(axis=0)
        n = np.cross(X[1] - X[0], X[2] - X[0]) if X.shape[1] == 3 else None
        if n is not None and np.linalg.norm(n) > 0:
            e1 = X[0] / (np.linalg.norm(X[0]) or 1.0)
            e2 = np.cross(n / np.linalg.norm(n), e1)
            ang = np.arctan2(X @ e2, X @ e1)
            vert = [vert[i] for i in np.argsort(ang)]
    if not others:
        return vert, len(vert)
    k = len(vert)
    mids = []
    for i in range(k):
        target = (P[vert[i]] + P[vert[(i + 1) % k]]) / 2 if k > 2 else (P[vert[0]] + P[vert[1]]) / 2
        j = int(np.argmin([np.linalg.norm(P[o] - target) for o in others]))
        mids.append(others[j])
        if k == 2:
            break
    return vert + mids, k


def msh(data: dict) -> str:
    """Gmsh 2.2 ASCII file: the elements of the mesh (physical group 1,
    "domain", or the first domain selection that contains them), then the
    boundary faces of each boundary selection (a physical group each)."""
    dim, P = data["dim"], np.asarray(data["points"], float)
    names = [n for n in data["regions"] if n not in ("all", "boundary") or data["regions"][n]["kind"] == "boundary"]
    groups = {name: k + 2 for k, name in enumerate(names)}
    cell_group = [1] * len(data["cells"])
    for name in names:
        reg = data["regions"][name]
        if reg["kind"] == "domain":
            for c in reg["cells"]:
                if cell_group[c] == 1:
                    cell_group[c] = groups[name]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames",
             str(1 + len(groups)), f'{dim} 1 "domain"']
    for name, tag in groups.items():
        d = dim if data["regions"][name]["kind"] == "domain" else dim - 1
        lines.append(f'{d} {tag} "{name}"')
    lines += ["$EndPhysicalNames", "$Nodes", str(len(P))]
    for i, p in enumerate(P):
        x = list(p) + [0.0] * (3 - len(p))
        lines.append(f"{i + 1} {x[0]:.16g} {x[1]:.16g} {x[2]:.16g}")
    lines.append("$EndNodes")
    elems = []
    for c, (gt, nodes) in enumerate(data["cells"]):
        etype, order = _element(gt)[:2]
        elems.append(f"{etype} 2 {cell_group[c]} {cell_group[c]} " + " ".join(str(nodes[i] + 1) for i in order))
    for name in names:
        reg = data["regions"][name]
        if reg["kind"] != "boundary":
            continue
        for cell, nodes in reg["faces"]:
            fn, nv = _face_nodes(data, cell, nodes)
            ftype = FACE_TYPES.get((nv, 2 if len(fn) > nv else 1))
            if ftype is None:
                raise ExportError(f"faces of {len(fn)} nodes cannot be written in this format")
            elems.append(f"{ftype} 2 {groups[name]} {groups[name]} " + " ".join(str(p + 1) for p in fn))
    lines += ["$Elements", str(len(elems))]
    lines += [f"{i + 1} {e}" for i, e in enumerate(elems)]
    lines.append("$EndElements")
    return "\n".join(lines) + "\n"


def inp(data: dict) -> str:
    """Abaqus input file: nodes, elements (one *ELEMENT block per type),
    an element set per domain selection, a node set per boundary selection."""
    dim, P = data["dim"], np.asarray(data["points"], float)
    if dim == 1:
        raise ExportError("1D meshes cannot be written in the Abaqus format")
    lines = ["*HEADING", "Mesh exported by the GetFEM GUI", "*NODE"]
    for i, p in enumerate(P):
        lines.append(f"{i + 1}, " + ", ".join(f"{x:.16g}" for x in p))
    by_type: dict[str, list[int]] = {}
    for c, (gt, _) in enumerate(data["cells"]):
        by_type.setdefault(_element(gt)[2], []).append(c)
    for etype, cells in by_type.items():
        lines.append(f"*ELEMENT, TYPE={etype}, ELSET=domain")
        for c in cells:
            gt, nodes = data["cells"][c]
            order = _element(gt)[3]
            lines.append(f"{c + 1}, " + ", ".join(str(nodes[i] + 1) for i in order))

    def chunks(ids):
        ids = [str(i) for i in ids]
        return [", ".join(ids[k:k + 16]) for k in range(0, len(ids), 16)]

    for name, reg in data["regions"].items():
        label = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
        if reg["kind"] == "domain":
            if name == "all":
                continue
            lines.append(f"*ELSET, ELSET={label}")
            lines += chunks(c + 1 for c in reg["cells"])
        else:
            lines.append(f"*NSET, NSET={label}")
            lines += chunks(sorted({p + 1 for _, nodes in reg["faces"] for p in nodes}))
    return "\n".join(lines) + "\n"
