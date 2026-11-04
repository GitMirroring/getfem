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

"""Application data model, independent of any graphical interface.

A model is a Python dictionary serialized as JSON. This module provides
loading (with migration of older format versions), saving, selection tables
and validation. Validation returns readable messages rather than raising an
exception, so that a GUI can display them next to the relevant fields.

Format version 2
================

Geometry and mesh are separate. A geometry carries *named selections*; every
mesh of that geometry inherits them. The physics only knows selection names
and the name of the mesh it uses.

{
  "format_version": 2,
  "name": "Plate with a hole",
  "geometries": [
    {
      "name": "plate",
      "source": {...},                      # see GEOMETRY_SOURCES
      "selections": {
        "hole":  {"kind": "boundary", "criteria": {"type": "in_ball", "center": [0.5, 0.5], "radius": 0.21}},
        "walls": {"kind": "boundary", "criteria": {"type": "union", "of": ["ymin", "ymax"]}}
      }
    }
  ],
  "meshes": [
    {"name": "coarse", "geometry": "plate", "mesher": "getfem", "h": 0.1},
    {"name": "fine",   "geometry": "plate", "mesher": "gmsh",   "size": 0.02, "order": 2}
  ],
  "discretization": {"degree": 2},
  "materials": [{"name": "steel", "selection": "all", "properties": {"k": 50}}],
  "physics": {
    "type": "heat_stationary", "mesh": "fine", "variable": "T",
    "sources": [{"selection": "all", "value": 0}],
    "boundary_conditions": [
      {"type": "temperature", "selection": "xmin", "value": 20},
      {"type": "heat_flux",   "selection": "hole", "value": 100},
      {"type": "convection",  "selection": "xmax", "h": 10, "T_ext": 20}
    ]
  },
  "study": {"type": "stationary"},
  "outputs": {"vtu": "result.vtu",
              "quantities": [{"name": "flux", "expr": "-k*Grad_T.Normal", "selection": "xmin"}]}
}

Geometry sources
----------------
- box       : {"type": "box", "origin": [...], "size": [...]}         (1D, 2D, 3D)
- csg       : {"type": "csg", "dim": 2, "shape": <shape>}            primitives + booleans
              shape = {"type": "rectangle", "min": [...], "max": [...]}
                    | {"type": "ball", "center": [...], "radius": r}
                    | {"type": "cylinder", "origin": [...], "axis": [...], "length": l, "radius": r}  (3D)
                    | {"type": "union" | "intersection" | "difference", "of": [<shape>, ...]}
- cad       : {"type": "cad", "file": "part.step"}                   STEP, IGES, BREP (through Gmsh)
- mesh_file : {"type": "mesh_file", "file": "part.msh"}              the geometry is the mesh

Meshers and compatible sources
------------------------------
- structured : box               {"divisions": [...], "element": "simplex" | "quad"}
- getfem     : box, csg          {"h": size, "order": 1}   (GetFEM experimental mesher)
- gmsh       : box, csg, cad     {"size": size, "order": 1}
- import     : mesh_file         (no parameter)

Selections
----------
Always available: "all" (the whole domain) and "boundary" (the whole boundary).
box adds xmin, xmax, ymin, ...; mesh_file adds its named physical groups.
User selections: {"kind": "domain" | "boundary", "criteria": {...}} with
- {"type": "plane", "point": [...], "normal": [...]}   boundary faces lying in a plane
- {"type": "normal", "direction": [...], "angle": 1}   boundary faces with a close normal (degrees);
                                                         beware, this includes inner boundaries (holes)
- {"type": "in_box", "min": [...], "max": [...]}        boundary faces / elements inside the box
- {"type": "in_ball", "center": [...], "radius": r}     boundary faces inside the ball
- {"type": "physical", "tag": n}                         physical group of an imported mesh
- {"type": "union", "of": ["name1", "name2", ...]}       union of selections of the same kind
Criteria are evaluated on the mesh, so they survive remeshing.

Physics
-------
The available physics, their material properties, options and boundary
conditions are described declaratively in PHYSICS_SPECS (a GUI can build its
forms from it):

- heat_stationary   : -div(k grad T) = f
                      properties k; sources: heat source (scalar)
                      BCs: temperature {value}, heat_flux {value} (incoming),
                           convection {h, T_ext}
                      GWFL macro: q = -k*Grad_T (heat flux vector)
- linear_elasticity : -div sigma(u) = f, small strains, isotropic material
                      properties E, nu; option "plane": "strain" | "stress" (2D only)
                      sources: body force (vector)
                      BCs: fixed {}, displacement {value: vector}, roller {} (u.n = 0),
                           traction {value: vector}, pressure {value} (on the outward normal)
                      GWFL macros: eps, sigma, von_mises, mu_L, lambda_L

Numerical values may be numbers or GWFL expressions, in which the space
coordinates are x, y, z (up to the dimension; X(1), X(2), X(3) also work).
Vector values are a list
of such values (one per space dimension) or a GWFL vector expression such as
"[0, -9.81*rho]". The macros of the physics may be used in the expressions of
the derived quantities, e.g. "(sigma*Normal)(1)" for a reaction force.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

FORMAT_VERSION = 2

ALL = "all"
BOUNDARY = "boundary"
DOMAIN = "domain"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AXES = "xyz"

GEOMETRY_SOURCES = ("box", "csg", "cad", "mesh_file")
MESHERS = {
    "structured": ("box",),
    "getfem": ("box", "csg"),
    "gmsh": ("box", "csg", "cad"),
    "import": ("mesh_file",),
}
CRITERIA = {
    "plane": (BOUNDARY,),
    "normal": (BOUNDARY,),
    "in_box": (BOUNDARY, DOMAIN),
    "in_ball": (BOUNDARY,),
    "physical": (BOUNDARY, DOMAIN),
    "union": (BOUNDARY, DOMAIN),
}
SHAPES_BOOLEAN = ("union", "intersection", "difference")

SCALAR = "scalar"
VECTOR = "vector"

# Space coordinates, defined in every expression (GWFL macros x = X(1), ...,
# up to the dimension of the mesh); these names cannot be used for anything else.
COORDINATES = ("x", "y", "z")

# Declarative description of the physics. Value kinds are SCALAR or VECTOR.
PHYSICS_SPECS = {
    "heat_stationary": {
        "label": "Heat transfer (stationary)",
        "field": SCALAR,
        "variable": "T",
        "properties": {"k": "thermal conductivity"},
        "property_defaults": {"k": 1.0},
        "options": {},
        "source": SCALAR,
        "boundary_conditions": {
            "temperature": {"value": SCALAR},
            "heat_flux": {"value": SCALAR},
            "convection": {"h": SCALAR, "T_ext": SCALAR},
        },
        "anchoring": ("temperature", "convection"),
        "anchoring_message": "without imposed temperature or convection, "
                             "the temperature is only defined up to a constant",
        "macros": ("q",),
    },
    "linear_elasticity": {
        "label": "Linear elasticity",
        "field": VECTOR,
        "variable": "u",
        "properties": {"E": "Young's modulus", "nu": "Poisson's ratio"},
        "property_defaults": {"E": 1.0, "nu": 0.3},
        "options": {"plane": ("strain", "stress")},
        "source": VECTOR,
        "boundary_conditions": {
            "fixed": {},
            "displacement": {"value": VECTOR},
            "roller": {},
            "traction": {"value": VECTOR},
            "pressure": {"value": SCALAR},
        },
        "anchoring": ("fixed", "displacement", "roller"),
        "anchoring_message": "without any displacement constraint, "
                             "rigid body motions are not prevented",
        "macros": ("eps", "sigma", "sigma_zz", "von_mises", "mu_L", "lambda_L", "nitsche_gamma0"),
    },
}


# ------------------------------------------------------------------ utilities

def box_boundary_names(dim: int) -> list[str]:
    """Names of the faces of a box: xmin, xmax, ymin, ..."""
    return [f"{_AXES[i]}{side}" for i in range(dim) for side in ("min", "max")]


def default_model() -> dict:
    """Starting model: unit square, conduction, left and right edges fixed."""
    return {
        "format_version": FORMAT_VERSION,
        "name": "New model",
        "geometries": [{
            "name": "geometry",
            "source": {"type": "box", "origin": [0.0, 0.0], "size": [1.0, 1.0]},
            "selections": {},
        }],
        "meshes": [{
            "name": "mesh", "geometry": "geometry", "mesher": "structured",
            "divisions": [10, 10], "element": "simplex",
        }],
        "discretization": {"degree": 2},
        "materials": [{"name": "material", "selection": ALL, "properties": {"k": 1.0}}],
        "physics": {
            "type": "heat_stationary",
            "mesh": "mesh",
            "variable": "T",
            "sources": [],
            "boundary_conditions": [
                {"type": "temperature", "selection": "xmin", "value": 0.0},
                {"type": "temperature", "selection": "xmax", "value": 1.0},
            ],
        },
        "study": {"type": "stationary"},
        "outputs": {"vtu": "result.vtu", "quantities": []},
    }


def load(path: str | Path) -> dict:
    """Load a model; a file in an older format version is migrated in memory."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        model = json.load(f)
    model = migrate(model)
    model["_source_dir"] = str(path.resolve().parent)
    return model


_TOP_ORDER = ("format_version", "name", "geometries", "meshes", "discretization",
              "materials", "physics", "study", "outputs")


def save(model: dict, path: str | Path) -> None:
    keys = [k for k in _TOP_ORDER if k in model] + [k for k in model if k not in _TOP_ORDER]
    data = {k: model[k] for k in keys if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_file(model: dict, file: str) -> Path:
    p = Path(file)
    if not p.is_absolute():
        p = Path(model.get("_source_dir", ".")) / p
    return p.resolve()


def read_gmsh_physical_names(path: str | Path) -> dict[str, tuple[int, int]]:
    """$PhysicalNames section of an ASCII .msh file (formats 2 and 4): {name: (dim, tag)}."""
    names: dict[str, tuple[int, int]] = {}
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as f:
            lines = iter(f)
            for line in lines:
                if line.strip() == "$PhysicalNames":
                    for _ in range(int(next(lines))):
                        parts = next(lines).split(maxsplit=2)
                        names[parts[2].strip().strip('"')] = (int(parts[0]), int(parts[1]))
                    break
    except (OSError, ValueError, StopIteration, IndexError):
        return {}
    return names


# ------------------------------------------------------------------ access

def by_name(items: list[dict], name: str) -> dict | None:
    return next((it for it in items if it.get("name") == name), None)


def physics_mesh(model: dict) -> dict | None:
    """Mesh used by the physics (the first one if not specified)."""
    meshes = model.get("meshes", [])
    name = model.get("physics", {}).get("mesh")
    if name is None:
        return meshes[0] if meshes else None
    return by_name(meshes, name)


def geometry_of(model: dict, mesh: dict) -> dict | None:
    return by_name(model.get("geometries", []), mesh.get("geometry"))


def geometry_dim(model: dict, geom: dict) -> int | None:
    src = geom.get("source", {})
    if src.get("type") == "box":
        return len(src.get("size", []))
    if src.get("type") in ("csg", "cad"):
        return src.get("dim", 3 if src.get("type") == "cad" else None)
    if src.get("type") == "mesh_file" and "file" in src:
        dims = [d for d, _ in read_gmsh_physical_names(resolve_file(model, src["file"])).values()]
        return max(dims) if dims else None
    return None


def selection_table(model: dict, geom: dict) -> dict[str, dict]:
    """All selections of a geometry: {name: {"kind": ..., "builtin": bool, ...}}.

    The order matters: it fixes the numbering of the GetFEM regions.
    """
    table: dict[str, dict] = {
        ALL: {"kind": DOMAIN, "builtin": True},
        BOUNDARY: {"kind": BOUNDARY, "builtin": True},
    }
    src = geom.get("source", {})
    if src.get("type") == "box":
        dim = len(src.get("size", []))
        origin, size = src.get("origin", []), src.get("size", [])
        for i, name in enumerate(box_boundary_names(dim)):
            axis, sign = divmod(i, 2)
            normal = [0.0] * dim
            normal[axis] = 1.0
            point = list(origin)
            if sign and axis < len(point) and axis < len(size):
                point[axis] = origin[axis] + size[axis]
            table[name] = {"kind": BOUNDARY, "builtin": True,
                           "criteria": {"type": "plane", "point": point, "normal": normal}}
    elif src.get("type") == "mesh_file" and "file" in src:
        names = read_gmsh_physical_names(resolve_file(model, src["file"]))
        top = max((d for d, _ in names.values()), default=0)
        for name, (d, tag) in names.items():
            table[name] = {"kind": DOMAIN if d == top else BOUNDARY, "builtin": True,
                           "criteria": {"type": "physical", "tag": tag}}
    for name, sel in geom.get("selections", {}).items():
        table[name] = {"kind": sel.get("kind"), "builtin": False, "criteria": sel.get("criteria", {})}
    return table


# ------------------------------------------------------------------ validation

def _check_value(v, where: str, errors: list[str], kind: str = SCALAR,
                 dim: int | None = None) -> None:
    """Check a scalar value (number or GWFL expression) or a vector value
    (list of scalar values, one per dimension, or GWFL vector expression)."""
    if kind == VECTOR and isinstance(v, list):
        if dim is not None and len(v) != dim:
            errors.append(f"{where}: {dim} components expected, got {len(v)}")
        for i, c in enumerate(v):
            _check_value(c, f"{where}[{i}]", errors)
        return
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        what = "list of components or GWFL expression" if kind == VECTOR \
            else "number or GWFL expression"
        errors.append(f"{where}: value expected ({what}), got {v!r}")
    elif isinstance(v, str) and not v.strip():
        errors.append(f"{where}: empty expression")
    elif kind == VECTOR and not isinstance(v, str):
        errors.append(f"{where}: vector expected, got the number {v!r}")


def _check_vec(v, n, where, errors, positive=False):
    if not isinstance(v, list) or (n is not None and len(v) != n) \
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v):
        errors.append(f"{where}: {n or ''} numbers expected".replace("  ", " "))
        return False
    if positive and any(x <= 0 for x in v):
        errors.append(f"{where}: strictly positive values expected")
    return True


def _check_positive(v, where, errors):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        errors.append(f"{where}: strictly positive number expected")


def _check_shape(shape, dim, where, errors):
    t = shape.get("type") if isinstance(shape, dict) else None
    if t == "rectangle":
        _check_vec(shape.get("min"), dim, f"{where}.min", errors)
        _check_vec(shape.get("max"), dim, f"{where}.max", errors)
    elif t == "ball":
        _check_vec(shape.get("center"), dim, f"{where}.center", errors)
        _check_positive(shape.get("radius"), f"{where}.radius", errors)
    elif t == "cylinder":
        if dim != 3:
            errors.append(f"{where}: cylinder only exists in 3D")
        _check_vec(shape.get("origin"), dim, f"{where}.origin", errors)
        _check_vec(shape.get("axis"), dim, f"{where}.axis", errors)
        _check_positive(shape.get("length"), f"{where}.length", errors)
        _check_positive(shape.get("radius"), f"{where}.radius", errors)
    elif t in SHAPES_BOOLEAN:
        parts = shape.get("of", [])
        if len(parts) < 2:
            errors.append(f"{where}.of: at least two shapes expected")
        for i, s in enumerate(parts):
            _check_shape(s, dim, f"{where}.of[{i}]", errors)
    else:
        errors.append(f"{where}.type: unknown shape {t!r} "
                      "(rectangle, ball, cylinder, union, intersection, difference)")


def _validate_geometry(model, geom, where, errors):
    src = geom.get("source", {})
    t = src.get("type")
    if t == "box":
        dim = len(src.get("size", []))
        if dim not in (1, 2, 3):
            errors.append(f"{where}.source.size: 1, 2 or 3 values expected")
        else:
            _check_vec(src.get("size"), dim, f"{where}.source.size", errors, positive=True)
            _check_vec(src.get("origin"), dim, f"{where}.source.origin", errors)
    elif t == "csg":
        dim = src.get("dim")
        if dim not in (2, 3):
            errors.append(f"{where}.source.dim: 2 or 3 expected")
        else:
            _check_shape(src.get("shape"), dim, f"{where}.source.shape", errors)
    elif t in ("cad", "mesh_file"):
        if "file" not in src:
            errors.append(f"{where}.source.file: missing file path")
        elif not resolve_file(model, src["file"]).exists():
            errors.append(f"{where}.source.file: file not found ({resolve_file(model, src['file'])})")
        elif t == "mesh_file":
            by_tag: dict[int, list[str]] = {}
            for name, (_, tag) in read_gmsh_physical_names(resolve_file(model, src["file"])).items():
                by_tag.setdefault(tag, []).append(name)
            for tag, group in by_tag.items():
                if len(group) > 1:
                    errors.append(f"{where}.source.file: physical groups {', '.join(group)} "
                                  f"share the number {tag}; GetFEM needs distinct numbers")
        if t == "cad" and Path(src.get("file", "")).suffix.lower() not in (".step", ".stp", ".iges", ".igs", ".brep"):
            errors.append(f"{where}.source.file: accepted CAD formats are STEP, IGES, BREP")
    else:
        errors.append(f"{where}.source.type: {', '.join(GEOMETRY_SOURCES)} expected, got {t!r}")
        return

    table = selection_table(model, geom)
    dim = geometry_dim(model, geom)
    for name, sel in geom.get("selections", {}).items():
        w = f"{where}.selections.{name}"
        if not _IDENT.match(name):
            errors.append(f"{w}: invalid name (letters, digits, _)")
        if name in table and table[name]["builtin"]:
            errors.append(f"{w}: {name!r} is already a predefined selection")
        kind = sel.get("kind")
        if kind not in (DOMAIN, BOUNDARY):
            errors.append(f"{w}.kind: 'domain' or 'boundary' expected")
            continue
        crit = sel.get("criteria", {})
        ct = crit.get("type")
        if ct not in CRITERIA:
            errors.append(f"{w}.criteria.type: {', '.join(CRITERIA)} expected, got {ct!r}")
            continue
        if kind not in CRITERIA[ct]:
            errors.append(f"{w}: criterion {ct!r} only applies to {', '.join(CRITERIA[ct])}")
        if ct == "normal":
            _check_vec(crit.get("direction"), dim, f"{w}.criteria.direction", errors)
        elif ct == "plane":
            _check_vec(crit.get("point"), dim, f"{w}.criteria.point", errors)
            _check_vec(crit.get("normal"), dim, f"{w}.criteria.normal", errors)
        elif ct == "in_box":
            _check_vec(crit.get("min"), dim, f"{w}.criteria.min", errors)
            _check_vec(crit.get("max"), dim, f"{w}.criteria.max", errors)
        elif ct == "in_ball":
            _check_vec(crit.get("center"), dim, f"{w}.criteria.center", errors)
            _check_positive(crit.get("radius"), f"{w}.criteria.radius", errors)
        elif ct == "physical":
            if t != "mesh_file":
                errors.append(f"{w}: criterion 'physical' requires a mesh_file geometry")
            if not isinstance(crit.get("tag"), int):
                errors.append(f"{w}.criteria.tag: integer expected")
        elif ct == "union":
            for other in crit.get("of", []):
                if other not in table or other == name:
                    errors.append(f"{w}.criteria.of: unknown selection {other!r}")
                elif table[other]["kind"] != kind:
                    errors.append(f"{w}.criteria.of: {other!r} is not of kind {kind!r}")
                elif table[other].get("criteria", {}).get("type") == "union":
                    errors.append(f"{w}.criteria.of: union of unions not supported ({other!r})")
            if not crit.get("of"):
                errors.append(f"{w}.criteria.of: list of selections expected")


def _validate_mesh(model, mesh, where, errors):
    geom = geometry_of(model, mesh)
    if geom is None:
        errors.append(f"{where}.geometry: unknown geometry {mesh.get('geometry')!r}")
        return
    mesher = mesh.get("mesher")
    if mesher not in MESHERS:
        errors.append(f"{where}.mesher: {', '.join(MESHERS)} expected, got {mesher!r}")
        return
    stype = geom.get("source", {}).get("type")
    if stype not in MESHERS[mesher]:
        errors.append(f"{where}.mesher: {mesher!r} cannot mesh a {stype!r} geometry "
                      f"(compatible: {', '.join(MESHERS[mesher])})")
    if mesher == "structured" and stype == "box":
        dim = len(geom["source"].get("size", []))
        div = mesh.get("divisions")
        if not isinstance(div, list) or len(div) != dim or any(
                not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in div):
            errors.append(f"{where}.divisions: {dim} integers >= 1 expected")
        if mesh.get("element", "simplex") not in ("simplex", "quad"):
            errors.append(f"{where}.element: 'simplex' or 'quad' expected")
    elif mesher == "getfem":
        _check_positive(mesh.get("h"), f"{where}.h", errors)
    elif mesher == "gmsh":
        _check_positive(mesh.get("size"), f"{where}.size", errors)
    if mesher in ("getfem", "gmsh") and mesh.get("order", 1) not in (1, 2, 3):
        errors.append(f"{where}.order: 1, 2 or 3 expected")


def validate(model: dict) -> list[str]:
    """Return the list of errors (empty if the model is valid)."""
    errors: list[str] = []
    if model.get("format_version") != FORMAT_VERSION:
        errors.append(f"format_version: {FORMAT_VERSION} expected, got {model.get('format_version')!r}")

    # --- Geometries and meshes
    for key in ("geometries", "meshes"):
        items = model.get(key, [])
        if not items:
            errors.append(f"{key}: at least one item is required")
        names = [it.get("name") for it in items]
        for i, name in enumerate(names):
            if not isinstance(name, str) or not _IDENT.match(name):
                errors.append(f"{key}[{i}].name: identifier expected, got {name!r}")
            elif names.count(name) > 1:
                errors.append(f"{key}[{i}].name: name {name!r} used more than once")
    for i, geom in enumerate(model.get("geometries", [])):
        _validate_geometry(model, geom, f"geometries[{i}]", errors)
    for i, mesh in enumerate(model.get("meshes", [])):
        _validate_mesh(model, mesh, f"meshes[{i}]", errors)

    degree = model.get("discretization", {}).get("degree", 1)
    if not isinstance(degree, int) or isinstance(degree, bool) or not 1 <= degree <= 6:
        errors.append("discretization.degree: integer between 1 and 6 expected")

    phys = model.get("physics", {})
    mesh = physics_mesh(model)
    if mesh is None:
        errors.append(f"physics.mesh: unknown mesh {phys.get('mesh')!r}")
        return errors
    geom = geometry_of(model, mesh)
    if geom is None:
        return errors
    table = selection_table(model, geom)

    def check_sel(name, kind, where):
        if name not in table:
            errors.append(f"{where}: unknown selection {name!r} (geometry {geom['name']!r}: "
                          f"{', '.join(table)})")
        elif table[name]["kind"] != kind:
            what = "a domain" if kind == DOMAIN else "a boundary"
            errors.append(f"{where}: {name!r} is not {what}")

    # --- Physics type
    ptype = phys.get("type")
    spec = PHYSICS_SPECS.get(ptype)
    if spec is None:
        errors.append(f"physics.type: {', '.join(PHYSICS_SPECS)} expected, got {ptype!r}")
        return errors
    dim = geometry_dim(model, geom)
    var = phys.get("variable", spec["variable"])
    if not _IDENT.match(str(var)):
        errors.append(f"physics.variable: identifier expected, got {var!r}")
    elif var in COORDINATES:
        errors.append(f"physics.variable: {var!r} is reserved for the coordinates")
    elif var in spec["macros"]:
        errors.append(f"physics.variable: {var!r} is a reserved name for this physics")
    reserved = set(spec["macros"]) | {str(var)}

    # --- Materials
    materials = model.get("materials", [])
    if not materials:
        errors.append("materials: at least one material is required")
    seen = set()
    for i, mat in enumerate(materials):
        where = f"materials[{i}]"
        name = mat.get("name", "")
        if not _IDENT.match(str(name)):
            errors.append(f"{where}.name: identifier expected, got {name!r}")
        if name in seen:
            errors.append(f"{where}.name: name {name!r} already used")
        seen.add(name)
        check_sel(mat.get("selection", ALL), DOMAIN, f"{where}.selection")
        props = mat.get("properties", {})
        for prop, label in spec["properties"].items():
            if prop not in props:
                errors.append(f"{where}.properties.{prop}: missing {label}")
        for prop, v in props.items():
            if not _IDENT.match(prop):
                errors.append(f"{where}.properties: invalid name {prop!r}")
            elif prop in COORDINATES:
                errors.append(f"{where}.properties: {prop!r} is reserved for the coordinates")
            elif prop in reserved:
                errors.append(f"{where}.properties: {prop!r} is a reserved name for this physics")
            _check_value(v, f"{where}.properties.{prop}", errors)

    # --- Options
    for opt, value in phys.items():
        if opt in ("type", "mesh", "variable", "sources", "boundary_conditions"):
            continue
        if opt not in spec["options"]:
            errors.append(f"physics.{opt}: unknown option for {ptype!r}")
        elif value not in spec["options"][opt]:
            errors.append(f"physics.{opt}: {', '.join(spec['options'][opt])} expected, got {value!r}")
        elif opt == "plane" and dim is not None and dim != 2:
            errors.append("physics.plane: only meaningful in 2D")

    # --- Sources and boundary conditions
    for i, src in enumerate(phys.get("sources", [])):
        check_sel(src.get("selection", ALL), DOMAIN, f"physics.sources[{i}].selection")
        _check_value(src.get("value"), f"physics.sources[{i}].value", errors, spec["source"], dim)

    bc_specs = spec["boundary_conditions"]
    anchored = False
    for i, bc in enumerate(phys.get("boundary_conditions", [])):
        where = f"physics.boundary_conditions[{i}]"
        btype = bc.get("type")
        if btype not in bc_specs:
            errors.append(f"{where}.type: {', '.join(bc_specs)} expected, got {btype!r}")
            continue
        check_sel(bc.get("selection"), BOUNDARY, f"{where}.selection")
        for key, kind in bc_specs[btype].items():
            _check_value(bc.get(key), f"{where}.{key}", errors, kind, dim)
        for key in bc:
            if key not in ("type", "selection") and key not in bc_specs[btype]:
                errors.append(f"{where}.{key}: unknown field for a {btype!r} condition")
        anchored |= btype in spec["anchoring"]
    if not anchored and not errors:
        errors.append(f"physics.boundary_conditions: {spec['anchoring_message']}")

    # --- Outputs
    for i, q in enumerate(model.get("outputs", {}).get("quantities", [])):
        where = f"outputs.quantities[{i}]"
        if not _IDENT.match(str(q.get("name", ""))):
            errors.append(f"{where}.name: identifier expected")
        _check_value(q.get("expr"), f"{where}.expr", errors)
        sel = q.get("selection", ALL)
        if sel not in table:
            errors.append(f"{where}.selection: unknown selection {sel!r}")

    return errors


def with_defaults(model: dict) -> dict:
    """Copy of the model completed with the optional default values."""
    m = copy.deepcopy(model)
    m.setdefault("discretization", {}).setdefault("degree", 1)
    for geom in m.get("geometries", []):
        geom.setdefault("selections", {})
    for mesh in m.get("meshes", []):
        if mesh.get("mesher") == "structured":
            mesh.setdefault("element", "simplex")
        if mesh.get("mesher") in ("getfem", "gmsh"):
            mesh.setdefault("order", 1)
    phys = m.setdefault("physics", {})
    phys.setdefault("mesh", m["meshes"][0]["name"] if m.get("meshes") else None)
    spec = PHYSICS_SPECS.get(phys.get("type"), {})
    phys.setdefault("variable", spec.get("variable", "u"))
    phys.setdefault("sources", [])
    phys.setdefault("boundary_conditions", [])
    for mat in m.get("materials", []):
        mat.setdefault("selection", ALL)
    for src in phys["sources"]:
        src.setdefault("selection", ALL)
    out = m.setdefault("outputs", {})
    out.setdefault("vtu", "result.vtu")
    out.setdefault("quantities", [])
    for q in out["quantities"]:
        q.setdefault("selection", ALL)
    return m


# ------------------------------------------------------------------ migration

def migrate(model: dict) -> dict:
    """Bring a model from an older format version to the current one (copy)."""
    version = model.get("format_version")
    if version == FORMAT_VERSION:
        return model
    if version == 1:
        return _migrate_v1(model)
    raise ValueError(f"unknown format_version {version!r}")


def _migrate_v1(old: dict) -> dict:
    m = copy.deepcopy(old)
    mesh = m.pop("mesh", {})
    extra_regions = m.pop("regions", {})

    if mesh.get("type") == "box":
        geom = {"name": "geometry", "source": {"type": "box", "origin": mesh.get("origin"),
                                                "size": mesh.get("size")}, "selections": {}}
        new_mesh = {"name": "mesh", "geometry": "geometry", "mesher": "structured",
                    "divisions": mesh.get("divisions"), "element": mesh.get("element", "simplex")}
    else:
        geom = {"name": "geometry", "source": {"type": "mesh_file", "file": mesh.get("file")},
                "selections": {}}
        new_mesh = {"name": "mesh", "geometry": "geometry", "mesher": "import"}

    # region -> selection everywhere, keeping the key order
    def rename(items):
        for it in items:
            if "region" in it:
                renamed = {("selection" if k == "region" else k): v for k, v in it.items()}
                it.clear()
                it.update(renamed)
    phys = m.setdefault("physics", {})
    rename(m.get("materials", []))
    rename(phys.get("sources", []))
    rename(phys.get("boundary_conditions", []))
    rename(m.get("outputs", {}).get("quantities", []))

    # Regions declared by hand become selections by physical group number.
    domain_uses = {it.get("selection") for it in m.get("materials", []) + phys.get("sources", [])}
    for name, tag in extra_regions.items():
        kind = DOMAIN if name in domain_uses else BOUNDARY
        geom["selections"][name] = {"kind": kind, "criteria": {"type": "physical", "tag": int(tag)}}

    m["geometries"] = [geom]
    m["meshes"] = [new_mesh]
    phys["mesh"] = "mesh"
    m["format_version"] = FORMAT_VERSION
    return m
