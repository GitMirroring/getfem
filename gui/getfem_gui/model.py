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

A model file is a Python dictionary serialized as JSON. This module provides
loading, saving, the scopes of names, selection tables and validation.
Validation returns readable messages rather than raising an exception, so
that a GUI can display them next to the relevant fields. The design is
described in doc/model_format_v3.md.

Format version 3
================

The file follows the logic of GetFEM: shared definitions, then *models*,
each of them becoming one `gf.Model`, then global solvers chaining them.

{
  "format_version": 3,
  "name": "Plate with an inclusion",
  "parameters": [{"name": "L", "value": 0.1}],
  "materials":  [{"name": "copper", "properties": {"k": 400}}],
  "geometries": [{"name": "plate", "source": {...}, "selections": {...}}],
  "meshes":     [{"name": "mesh", "geometry": "plate", "mesher": "getfem", "h": "L/40"}],
  "fields":     [{"name": "porosity", "geometry": "plate",
                  "values": [{"selection": "all", "value": 0.1}]}],
  "models": [{
    "name": "thermal",
    "parameters": [], "geometries": [], "meshes": [], "fields": [],   # local ones
    "imports": [],                                  # results of other models
    "variables": [{"name": "T", "mesh": "mesh", "fem": "lagrange", "degree": 2}],
    "modules": [{"type": "heat_conduction", "variable": "T",
                 "properties": {"k": [{"selection": "matrix", "value": 0.2},
                                      {"selection": "inclusion", "material": "copper"}]},
                 "sources": [{"selection": "all", "value": 0}],
                 "conditions": [{"type": "temperature", "selection": "cold", "value": 20}]}],
    "terms": [{"name": "extra", "expr": "-q0*Test_T", "selection": "inclusion"}],
    "conditions": [{"type": "dirichlet", "variable": "T", "selection": "hot", "value": 80}],
    "solver": {"type": "auto"},
    "outputs": [{"name": "heat_flow", "expr": "-k*Grad_T.Normal", "selection": "cold"}],
    "vtu": "thermal.vtu"
  }],
  "solvers": [{"name": "main", "type": "sequence", "steps": ["thermal"]}],
  "outputs": [],
  "graphics": []
}

Names and scopes
----------------
Parameters, materials and fields of the top level are visible in every
model; the local parameters, fields, variables, imports, module properties
and macros of a model only in it. A local name may not hide a global one.
Geometry and mesh names are unique in the whole file. A model takes the
result of another one through an import, {"name": "T", "model": "thermal",
"variable": "T"} (for now on the same mesh), and is then solved after it.

Geometry sources
----------------
- box       : {"type": "box", "origin": [...], "size": [...]}         (1D, 2D, 3D)
- csg       : {"type": "csg", "dim": 2, "shape": <shape>}            primitives + booleans
              shape = {"type": "rectangle", "min": [...], "max": [...]}
                    | {"type": "ball", "center": [...], "radius": r}
                    | {"type": "cylinder", "origin": [...], "axis": [...], "length": l, "radius": r}  (3D)
                    | {"type": "extrusion", "profile": <2D shape>, "z": [z0, z1]}  (3D, along z)
                    | {"type": "revolution", "profile": <2D shape>}  (3D, about the z axis; the
                      profile is given in the (r, z) half plane, its part with r < 0 is ignored)
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
- {"type": "edges", "edges": [{"shape": [1], "side": "circle"}, ...]}
                                                         2D box or csg: sides of primitives (the
                                                         parts on the boundary); "shape" is the path
                                                         of the primitive in the CSG tree (indices in
                                                         "of"), "side" xmin, xmax, ymin, ymax or circle
- {"type": "regions", "regions": [[[0], [1]], [[0]]]}
                                                         2D box or csg: elements in regions of the
                                                         domain; a region is given by the paths of the
                                                         primitives containing it (here "in rectangle 1
                                                         and in ball 2", then "in rectangle 1 only").
                                                         The GetFEM mesher follows the interfaces between
                                                         regions (the sides of primitives inside the
                                                         domain); Gmsh does not yet (staircase)
Criteria are evaluated on the mesh, so they survive remeshing.

Models
------
- variables : {"name", "mesh", "fem": "lagrange" | "lagrange_discontinuous",
               "degree": 1..6, "components": 1 | "dim" | n}
- modules   : predefined physics, see MODULE_SPECS (properties per domain
              selection, given by a value or a material; sources;
              conditions of the module)
- terms     : GWFL weak forms, {"name", "expr", "selection", "linear": "auto"
              | true | false}; the terms of a model sum to zero (a source
              is written "-f*Test_u")
- conditions: generic ones on a variable, see CONDITIONS
- solver    : {"type": "auto" | "linear" | "newton", "max_iter", "max_res"}
- outputs   : integrals {"name", "expr", "selection"}; "vtu": file name of
              the export (variables, module fields, fields), null for none
- graphics  : saved views of the result fields, {"name", "field", and the
              display settings of GRAPHIC_DEFAULTS: component, warp, scale,
              isolines, bands, cmap, range, arrows, cut}
Global solvers: {"name", "type": "sequence" | "staggered", "steps": [model
names], "max_iter", "tolerance", "relaxation"}; without any, the models are
solved in an order compatible with their imports.

Numerical values may be numbers or GWFL expressions, in which the space
coordinates are x, y, z (up to the dimension; X(1), X(2), X(3) also work).
Vector values are a list of such values (one per space dimension) or a GWFL
vector expression such as "[0, -9.81*rho]".
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from . import parameters as P

FORMAT_VERSION = 3

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
    "edges": (BOUNDARY,),
    "regions": (DOMAIN,),
}
SHAPES_BOOLEAN = ("union", "intersection", "difference")
SHAPES_SWEPT = ("extrusion", "revolution")        # 3D shapes made of a 2D profile

SCALAR = "scalar"
VECTOR = "vector"

# Space coordinates, defined in every expression (GWFL macros x = X(1), ...,
# up to the dimension of the mesh); these names cannot be used for anything else.
COORDINATES = ("x", "y", "z")

FEMS = ("lagrange", "lagrange_discontinuous")
SOLVER_TYPES = ("auto", "linear", "newton")
GLOBAL_SOLVERS = ("sequence", "staggered")
DIRICHLET_METHODS = ("simplification", "multiplier", "penalization")
COLORMAPS = ("viridis", "coolwarm", "jet", "plasma", "gray")     # the first one: default

# Catalogue of the material properties: name -> (label, SI unit). The
# modules and the material libraries use these names.
PROPERTIES = {
    "k": ("thermal conductivity", "W/(m·K)"),
    "E": ("Young's modulus", "Pa"),
    "nu": ("Poisson's ratio", "1"),
    "alpha": ("thermal expansion coefficient", "1/K"),
    "rho": ("density", "kg/m³"),
    "cp": ("specific heat capacity", "J/(kg·K)"),
}

# Saved views of a result field (models[k].graphics): their display settings;
# a missing key takes the default value.
GRAPHIC_DEFAULTS = {"component": None, "warp": True, "scale": None, "isolines": 0, "bands": False,
                    "cmap": None, "range": None, "arrows": None, "cut": None}

# Generic conditions on a variable: their values ("var": the kind of the
# variable, scalar or vector)
CONDITIONS = {
    "dirichlet": {"value": "var"},
    "normal_dirichlet": {"value": SCALAR},
    "neumann": {"value": "var"},
    "robin": {"h": SCALAR, "value": "var"},
}
ANCHORING_CONDITIONS = ("dirichlet", "normal_dirichlet", "robin")

# Declarative description of the modules (predefined physics). "variables":
# the roles of the variables a module acts on (key of the module holding the
# name), with the defaults of a variable it creates; value kinds are SCALAR
# or VECTOR.
MODULE_SPECS = {
    "heat_conduction": {
        "label": "Heat conduction (stationary)",
        "variables": {"variable": {"name": "T", "kind": SCALAR, "degree": 2, "label": "temperature"}},
        "properties": {"k": "thermal conductivity"},
        "property_defaults": {"k": 1.0},
        "options": {},
        "values": {},
        "source": SCALAR,
        "conditions": {
            "temperature": {"value": SCALAR},
            "heat_flux": {"value": SCALAR},
            "convection": {"h": SCALAR, "T_ext": SCALAR},
        },
        "anchoring": ("temperature", "convection"),
        "anchoring_message": "without imposed temperature or convection, "
                             "the temperature is only defined up to a constant",
        "macros": ("q",),
        "exports": {"heat_flux": ("q", VECTOR)},
    },
    "linear_elasticity": {
        "label": "Linear elasticity",
        "variables": {"variable": {"name": "u", "kind": VECTOR, "degree": 2, "label": "displacement"}},
        "properties": {"E": "Young's modulus", "nu": "Poisson's ratio"},
        "property_defaults": {"E": 1.0, "nu": 0.3},
        "options": {"plane": ("strain", "stress")},
        "values": {},
        "source": VECTOR,
        "conditions": {
            "fixed": {},
            "displacement": {"value": VECTOR},
            "roller": {},
            "traction": {"value": VECTOR},
            "pressure": {"value": SCALAR},
        },
        "anchoring": ("fixed", "displacement", "roller"),
        "anchoring_message": "without any displacement constraint, "
                             "rigid body motions are not prevented",
        "macros": ("eps", "sigma", "sigma_zz", "von_mises", "mu_L", "lambda_L", "beta_L", "theta_L",
                   "nitsche_gamma0"),
        "exports": {"von_mises": ("von_mises", SCALAR)},
    },
    "thermal_expansion": {
        "label": "Thermal expansion",
        "variables": {"temperature": {"name": "T", "kind": SCALAR, "degree": 2, "label": "temperature",
                                      "import": True},
                      "displacement": {"name": "u", "kind": VECTOR, "degree": 2, "label": "displacement"}},
        "properties": {"alpha": "thermal expansion coefficient"},
        "property_defaults": {"alpha": 1e-5},
        "options": {},
        "values": {"T_ref": SCALAR},
        "value_defaults": {"T_ref": 0.0},
        "source": None,
        "conditions": {},
        "requires": ("linear_elasticity", "displacement"),
        "anchoring": (),
        "macros": (),
        "exports": {},
    },
}


# ------------------------------------------------------------------ utilities

def box_boundary_names(dim: int) -> list[str]:
    """Names of the faces of a box: xmin, xmax, ymin, ..."""
    return [f"{_AXES[i]}{side}" for i in range(dim) for side in ("min", "max")]


def new_module(mtype: str, dim: int = 2, names: dict | None = None) -> dict:
    """A module of type mtype with default properties, acting on the variables
    of the given names (roles -> names; the defaults of the spec otherwise)."""
    spec = MODULE_SPECS[mtype]
    mod = {"type": mtype}
    for role, v in spec["variables"].items():
        mod[role] = (names or {}).get(role, v["name"])
    for opt, choices in spec["options"].items():
        if opt != "plane" or dim == 2:
            mod[opt] = choices[0]
    for key, value in spec.get("value_defaults", {}).items():
        mod[key] = value
    mod["properties"] = {p: [{"selection": ALL, "value": v}]
                         for p, v in spec["property_defaults"].items()}
    if spec["source"] is not None:
        mod["sources"] = []
    if spec["conditions"]:
        mod["conditions"] = []
    return mod


def new_variable(name: str, mesh: str, kind: str = SCALAR, degree: int = 2) -> dict:
    var = {"name": name, "mesh": mesh, "fem": "lagrange", "degree": degree}
    if kind == VECTOR:
        var["components"] = "dim"
    return var


def new_model(name: str, mesh: str, mtype: str | None = None, dim: int = 2) -> dict:
    """An empty model, or one with a module of type mtype and its variables."""
    md = {"name": name, "variables": [], "modules": [], "terms": [], "conditions": [],
          "solver": {"type": "auto"}, "outputs": [], "vtu": f"{name}.vtu"}
    if mtype is not None:
        spec = MODULE_SPECS[mtype]
        for role, v in spec["variables"].items():
            md["variables"].append(new_variable(v["name"], mesh, v["kind"], v["degree"]))
        md["modules"].append(new_module(mtype, dim))
    return md


def default_model() -> dict:
    """Starting model: unit square, heat conduction, left and right edges fixed."""
    md = new_model("heat", "mesh", "heat_conduction")
    md["modules"][0]["conditions"] = [
        {"type": "temperature", "selection": "xmin", "value": 0.0},
        {"type": "temperature", "selection": "xmax", "value": 1.0},
    ]
    return {
        "format_version": FORMAT_VERSION,
        "name": "New model",
        "parameters": [],
        "materials": [],
        "geometries": [{
            "name": "geometry",
            "source": {"type": "box", "origin": [0.0, 0.0], "size": [1.0, 1.0]},
            "selections": {},
        }],
        "meshes": [{
            "name": "mesh", "geometry": "geometry", "mesher": "structured",
            "divisions": [10, 10], "element": "simplex",
        }],
        "fields": [],
        "models": [md],
        "solvers": [],
        "outputs": [],
        "graphics": [],
    }


def load(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        model = json.load(f)
    if model.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"format_version {FORMAT_VERSION} expected, got "
                         f"{model.get('format_version')!r} (older formats are not supported)")
    model["_source_dir"] = str(path.resolve().parent)
    return model


_TOP_ORDER = ("format_version", "name", "parameters", "materials", "geometries", "meshes", "fields",
              "models", "solvers", "outputs", "graphics")
_MODEL_ORDER = ("name", "description", "parameters", "geometries", "meshes", "fields", "imports",
                "variables", "modules", "terms", "conditions", "solver", "outputs", "vtu", "graphics")


def _ordered(d: dict, order) -> dict:
    keys = [k for k in order if k in d] + [k for k in d if k not in order]
    return {k: d[k] for k in keys if not k.startswith("_")}


def save(model: dict, path: str | Path) -> None:
    data = _ordered(model, _TOP_ORDER)
    data["models"] = [_ordered(md, _MODEL_ORDER) for md in data.get("models", [])]
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



# ------------------------------------------------------------------ access and scopes

def by_name(items: list[dict], name: str) -> dict | None:
    return next((it for it in items if isinstance(it, dict) and it.get("name") == name), None)


def all_geometries(model: dict) -> list[tuple[dict, dict | None]]:
    """(geometry, model owning it or None for the top level), in file order."""
    out = [(g, None) for g in model.get("geometries", [])]
    for md in model.get("models", []):
        out += [(g, md) for g in md.get("geometries", [])]
    return out


def all_meshes(model: dict) -> list[tuple[dict, dict | None]]:
    out = [(m, None) for m in model.get("meshes", [])]
    for md in model.get("models", []):
        out += [(m, md) for m in md.get("meshes", [])]
    return out


def find_geometry(model: dict, name: str) -> dict | None:
    return next((g for g, _ in all_geometries(model) if g.get("name") == name), None)


def find_mesh(model: dict, name: str) -> dict | None:
    return next((m for m, _ in all_meshes(model) if m.get("name") == name), None)


def geometry_of(model: dict, mesh: dict) -> dict | None:
    return find_geometry(model, mesh.get("geometry"))


def model_by_name(model: dict, name: str) -> dict | None:
    return by_name(model.get("models", []), name)


def model_mesh(model: dict, md: dict) -> dict | None:
    """The mesh of a model: the one of its variables (all on the same mesh for now)."""
    for v in md.get("variables", []):
        mesh = find_mesh(model, v.get("mesh"))
        if mesh is not None:
            return mesh
    return None


def model_geometry(model: dict, md: dict) -> dict | None:
    mesh = model_mesh(model, md)
    return geometry_of(model, mesh) if mesh else None


def model_dim(model: dict, md: dict) -> int | None:
    geom = model_geometry(model, md)
    return geometry_dim(model, geom) if geom else None


def variable_kind(var: dict) -> str:
    c = var.get("components", 1)
    return SCALAR if c == 1 else VECTOR


def module_variables(md: dict) -> list[tuple[dict, str, str]]:
    """(module, role, variable name) for the variables the modules act on."""
    out = []
    for mod in md.get("modules", []):
        spec = MODULE_SPECS.get(mod.get("type"), {})
        for role, v in spec.get("variables", {}).items():
            out.append((mod, role, mod.get(role, v["name"])))
    return out


def model_fields(model: dict, md: dict) -> list[dict]:
    """The fields usable in a model: the top level ones of its geometry, then
    its own ones."""
    geom = model_geometry(model, md)
    gname = geom.get("name") if geom else None
    glob = [f for f in model.get("fields", []) if f.get("geometry", gname) == gname]
    return glob + list(md.get("fields", []))


def model_properties(md: dict) -> list[str]:
    out = []
    for mod in md.get("modules", []):
        for p in MODULE_SPECS.get(mod.get("type"), {}).get("properties", {}):
            if p not in out:
                out.append(p)
    return out


def model_macros(md: dict) -> list[str]:
    out = []
    for mod in md.get("modules", []):
        for m in MODULE_SPECS.get(mod.get("type"), {}).get("macros", ()):
            if m not in out:
                out.append(m)
    return out


def global_names(model: dict) -> dict[str, str]:
    """GWFL names of the top level: {name: what}."""
    out = {}
    for p in model.get("parameters", []):
        out.setdefault(p.get("name"), "parameter")
    for f in model.get("fields", []):
        out.setdefault(f.get("name"), "field")
    return out


def local_names(md: dict) -> dict[str, str]:
    """GWFL names defined by a model: {name: what}, in definition order."""
    out = {}
    for p in md.get("parameters", []):
        out.setdefault(p.get("name"), "parameter")
    for f in md.get("fields", []):
        out.setdefault(f.get("name"), "field")
    for im in md.get("imports", []):
        out.setdefault(im.get("name"), "import")
    for v in md.get("variables", []):
        out.setdefault(v.get("name"), "variable")
    for p in model_properties(md):
        out.setdefault(p, "property")
    for m in model_macros(md):
        out.setdefault(m, "macro")
    return out


def scope_parameters(model: dict, md: dict | None) -> list[dict]:
    """The parameters visible in a model (top level, then local), in order."""
    return list(model.get("parameters", [])) + (list(md.get("parameters", [])) if md else [])


def parameter_values(model: dict, md: dict | None = None) -> tuple[dict[str, float], list[str]]:
    return P.values({"parameters": scope_parameters(model, md)})


def material(model: dict, name: str) -> dict | None:
    return by_name(model.get("materials", []), name)


def solve_order(model: dict) -> list[str]:
    """Model names in an order compatible with the imports (file order otherwise);
    a cycle leaves the remaining models in file order."""
    names = [md.get("name") for md in model.get("models", [])]
    deps = {md.get("name"): {im.get("model") for im in md.get("imports", [])} & set(names)
            for md in model.get("models", [])}
    order: list[str] = []
    while len(order) < len(names):
        ready = [n for n in names if n not in order and deps[n] <= set(order)]
        if not ready:
            order += [n for n in names if n not in order]
            break
        order.append(ready[0])
    return order


def main_solver(model: dict) -> dict:
    """The solver run by Compute: the first global one, or a sequence of all
    the models in the order of their imports."""
    if model.get("solvers"):
        return model["solvers"][0]
    return {"name": "default", "type": "sequence", "steps": solve_order(model)}


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
        origin, size = _numbers(model, src.get("origin", [])), _numbers(model, src.get("size", []))
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


def _numbers(model: dict, values: list) -> list:
    """Numbers of a list that may hold expressions of parameters (0 if invalid:
    validation reports it)."""
    if all(isinstance(v, (int, float)) for v in values):
        return list(values)
    vals, _ = P.values(model)
    out = []
    for v in values:
        try:
            out.append(P.evaluate(v, vals))
        except P.ParameterError:
            out.append(0.0)
    return out


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
    elif t in SHAPES_SWEPT:
        if dim != 3:
            errors.append(f"{where}: {t} only exists in 3D")
        if t == "extrusion" and _check_vec(shape.get("z"), 2, f"{where}.z", errors):
            if shape["z"][0] >= shape["z"][1]:
                errors.append(f"{where}.z: z0 < z1 expected")
        profile = shape.get("profile")
        if not isinstance(profile, dict):
            errors.append(f"{where}.profile: 2D shape expected")
        else:
            _check_shape(profile, 2, f"{where}.profile", errors)
    elif t in SHAPES_BOOLEAN:
        parts = shape.get("of", [])
        if len(parts) < 2:
            errors.append(f"{where}.of: at least two shapes expected")
        for i, s in enumerate(parts):
            _check_shape(s, dim, f"{where}.of[{i}]", errors)
    else:
        errors.append(f"{where}.type: unknown shape {t!r} "
                      "(rectangle, ball, cylinder, extrusion, revolution, union, intersection, "
                      "difference)")


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
        elif ct == "edges":
            _check_edges(src, crit, w, errors)
        elif ct == "regions":
            _check_regions(src, crit, w, errors)
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


def _check_edges(src, crit, where, errors):
    from . import geometry2d as G
    shape = G.shape_of(src)
    if shape is None:
        errors.append(f"{where}: criterion 'edges' requires a 2D box or csg geometry")
        return
    edges = crit.get("edges")
    if not isinstance(edges, list) or not edges:
        errors.append(f"{where}.criteria.edges: no edge: click edges of the geometry in the view")
        return
    try:
        all_pieces = G.pieces(shape)
    except (KeyError, TypeError, ValueError):
        return                       # invalid shape: reported elsewhere
    for j, ref in enumerate(edges):
        w = f"{where}.criteria.edges[{j}]"
        prim = G.primitive_at(shape, ref.get("shape", []) if isinstance(ref, dict) else None)
        if not isinstance(ref, dict) or prim is None or prim.get("type") in G.BOOLEANS:
            errors.append(f"{w}: no such primitive shape {ref.get('shape') if isinstance(ref, dict) else ref!r}")
        elif ref.get("side") not in (G.RECTANGLE_SIDES if prim["type"] == "rectangle" else ("circle",)):
            errors.append(f"{w}: a {prim['type']} has no side {ref.get('side')!r}")
        elif not G.pieces_of(shape, ref, all_pieces):
            errors.append(f"{w}: the {G.side_label(shape, ref)} is not on the boundary")


def _check_regions(src, crit, where, errors):
    from . import geometry2d as G
    shape = G.shape_of(src)
    if shape is None:
        errors.append(f"{where}: criterion 'regions' requires a 2D box or csg geometry")
        return
    regions = crit.get("regions")
    if not isinstance(regions, list) or not regions:
        errors.append(f"{where}.criteria.regions: no region: click in the geometry in the view")
        return
    for j, reg in enumerate(regions):
        w = f"{where}.criteria.regions[{j}]"
        if not isinstance(reg, list) or not reg:
            errors.append(f"{w}: list of primitive paths expected")
            continue
        for path in reg:
            prim = G.primitive_at(shape, path) if isinstance(path, list) else None
            if prim is None or prim.get("type") in G.BOOLEANS:
                errors.append(f"{w}: no such primitive shape {path!r}")


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



def _check_name(name, where, errors, taken: dict[str, str] | None = None, what="name") -> bool:
    """An identifier, not reserved, not in `taken` ({name: what it is})."""
    if not isinstance(name, str) or not _IDENT.match(name):
        errors.append(f"{where}: identifier expected, got {name!r}")
    elif name in COORDINATES:
        errors.append(f"{where}: {name!r} is reserved for the coordinates")
    elif name in P.GWFL_NAMES:
        errors.append(f"{where}: {name!r} is a reserved GWFL name")
    elif taken is not None and name in taken:
        errors.append(f"{where}: {name!r} is already a {taken[name]}")
    else:
        return True
    return False


def _validate_parameter_list(params, prefix, errors, taken: dict[str, str]) -> None:
    for i, p in enumerate(params):
        name = p.get("name") if isinstance(p, dict) else None
        if _check_name(name, f"{prefix}[{i}].name", errors, taken):
            taken[name] = "parameter"


def _validate_fields(fields, prefix, table_of, errors, taken: dict[str, str]) -> None:
    """Fields defined per domain selection; table_of(field) gives the
    selection table, or None (an error was reported)."""
    for i, f in enumerate(fields):
        where = f"{prefix}[{i}]"
        if _check_name(f.get("name"), f"{where}.name", errors, taken):
            taken[f["name"]] = "field"
        table = table_of(f, where)
        values = f.get("values", [])
        if not values:
            errors.append(f"{where}.values: at least one value (on a domain selection) is required")
        for j, v in enumerate(values):
            if table is not None:
                _check_selection(table, v.get("selection", ALL), DOMAIN, f"{where}.values[{j}].selection",
                                 errors)
            _check_value(v.get("value"), f"{where}.values[{j}].value", errors)


def _check_selection(table, name, kind, where, errors) -> None:
    if name not in table:
        errors.append(f"{where}: unknown selection {name!r} (available: {', '.join(table)})")
    elif kind is not None and table[name]["kind"] != kind:
        what = "a domain" if kind == DOMAIN else "a boundary"
        errors.append(f"{where}: {name!r} is not {what}")


def _check_int(v, lo, hi, where, errors) -> bool:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        errors.append(f"{where}: integer between {lo} and {hi} expected, got {v!r}")
        return False
    return True


def validate(model: dict) -> list[str]:
    """Return the list of errors (empty if the model is valid)."""
    errors: list[str] = []
    if model.get("format_version") != FORMAT_VERSION:
        errors.append(f"format_version: {FORMAT_VERSION} expected, got {model.get('format_version')!r}")

    # --- Parameters; the numerical fields of geometries and meshes may be
    # expressions of them: the rest of the validation works on their values.
    taken: dict[str, str] = {}
    _validate_parameter_list(model.get("parameters", []), "parameters", errors, taken)
    for k, md in enumerate(model.get("models", [])):
        _validate_parameter_list(md.get("parameters", []), f"models[{k}].parameters", errors, dict(taken))
    resolved, perrors = P.resolve(model)
    if perrors:
        return errors + perrors
    model = resolved

    # --- Materials
    seen: set[str] = set()
    for i, mat in enumerate(model.get("materials", [])):
        where = f"materials[{i}]"
        name = mat.get("name")
        if not isinstance(name, str) or not _IDENT.match(name):
            errors.append(f"{where}.name: identifier expected, got {name!r}")
        elif name in seen:
            errors.append(f"{where}.name: name {name!r} used more than once")
        seen.add(name)
        for prop, v in (mat.get("properties") or {}).items():
            if not _IDENT.match(str(prop)):
                errors.append(f"{where}.properties: invalid name {prop!r}")
            _check_value(v, f"{where}.properties.{prop}", errors)

    # --- Geometries and meshes (names unique in the whole file)
    for key, items in (("geometries", all_geometries(model)), ("meshes", all_meshes(model))):
        if key == "geometries" and not items:
            errors.append("geometries: at least one geometry is required")
        names = [it.get("name") for it, _ in items]
        for (it, owner), name in zip(items, names):
            where = _where(model, key, it, owner)
            if not isinstance(name, str) or not _IDENT.match(name):
                errors.append(f"{where}.name: identifier expected, got {name!r}")
            elif names.count(name) > 1:
                errors.append(f"{where}.name: name {name!r} used more than once in the file")
    for geom, owner in all_geometries(model):
        _validate_geometry(model, geom, _where(model, "geometries", geom, owner), errors)
    for mesh, owner in all_meshes(model):
        where = _where(model, "meshes", mesh, owner)
        visible = [g.get("name") for g in model.get("geometries", [])] + \
            ([g.get("name") for g in owner.get("geometries", [])] if owner else [])
        if mesh.get("geometry") not in visible and find_geometry(model, mesh.get("geometry")) is not None:
            errors.append(f"{where}.geometry: {mesh.get('geometry')!r} belongs to another model")
            continue
        _validate_mesh(model, mesh, where, errors)

    # --- Fields of the top level
    def global_table(f, where):
        geoms = model.get("geometries", [])
        gname = f.get("geometry", geoms[0].get("name") if len(geoms) == 1 else None)
        geom = by_name(geoms, gname)
        if geom is None:
            errors.append(f"{where}.geometry: unknown geometry {f.get('geometry')!r}")
            return None
        return selection_table(model, geom)
    _validate_fields(model.get("fields", []), "fields", global_table, errors, taken)

    # --- Models
    models = model.get("models", [])
    if not models:
        errors.append("models: at least one model is required")
    mnames = [md.get("name") for md in models]
    for k, md in enumerate(models):
        name = md.get("name")
        if not isinstance(name, str) or not _IDENT.match(name):
            errors.append(f"models[{k}].name: identifier expected, got {name!r}")
        elif mnames.count(name) > 1:
            errors.append(f"models[{k}].name: name {name!r} used more than once")
        _validate_model(model, md, k, errors, taken)

    # --- Global solvers, and the order of the imports
    snames = [s.get("name") for s in model.get("solvers", [])]
    for i, s in enumerate(model.get("solvers", [])):
        where = f"solvers[{i}]"
        if not isinstance(s.get("name"), str) or not _IDENT.match(s.get("name", "")):
            errors.append(f"{where}.name: identifier expected, got {s.get('name')!r}")
        elif snames.count(s["name"]) > 1:
            errors.append(f"{where}.name: name {s['name']!r} used more than once")
        if s.get("type") not in GLOBAL_SOLVERS:
            errors.append(f"{where}.type: {', '.join(GLOBAL_SOLVERS)} expected, got {s.get('type')!r}")
        steps = s.get("steps", [])
        if not isinstance(steps, list) or not steps:
            errors.append(f"{where}.steps: list of model names expected")
            steps = []
        for j, st in enumerate(steps):
            if st not in mnames:
                errors.append(f"{where}.steps[{j}]: unknown model {st!r}")
        if s.get("type") == "staggered":
            _check_int(s.get("max_iter", 20), 1, 10000, f"{where}.max_iter", errors)
            tol = s.get("tolerance", 1e-6)
            if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol <= 0:
                errors.append(f"{where}.tolerance: strictly positive number expected")
            rel = s.get("relaxation", 1.0)
            if isinstance(rel, bool) or not isinstance(rel, (int, float)) or not 0 < rel <= 1:
                errors.append(f"{where}.relaxation: number in ]0, 1] expected")
    if not errors and models:
        _check_import_order(model, errors)

    for key in ("outputs", "graphics"):
        if not isinstance(model.get(key, []), list):
            errors.append(f"{key}: list expected")
    return errors


def _where(model, key, item, owner) -> str:
    """Path of a geometry or a mesh, for the messages."""
    if owner is None:
        return f"{key}[{next(i for i, it in enumerate(model[key]) if it is item)}]"
    k = next(i for i, md in enumerate(model["models"]) if md is owner)
    return f"models[{k}].{key}[{next(i for i, it in enumerate(owner[key]) if it is item)}]"


def _check_import_order(model, errors) -> None:
    """Each import must be computed before: by the order of the main solver,
    or inside a staggered loop."""
    solver = main_solver(model)
    steps = solver.get("steps", [])
    for k, md in enumerate(model.get("models", [])):
        for j, im in enumerate(md.get("imports", [])):
            src = im.get("model")
            where = f"models[{k}].imports[{j}].model"
            if md.get("name") not in steps:
                continue
            if src not in steps:
                errors.append(f"{where}: model {src!r} is not solved by the solver "
                              f"{solver.get('name')!r}")
            elif solver.get("type") == "sequence" and steps.index(src) > steps.index(md["name"]):
                errors.append(f"{where}: model {src!r} is solved after {md['name']!r}: "
                              "change the order of the steps, or use a staggered solver")


def _validate_model(model: dict, md: dict, k: int, errors: list[str], gtaken: dict[str, str]) -> None:
    w = f"models[{k}]"
    taken = dict(gtaken)
    for p in md.get("parameters", []):
        if isinstance(p, dict) and isinstance(p.get("name"), str):
            taken.setdefault(p["name"], "parameter")

    # --- Variables (all on the same mesh for now)
    variables = md.get("variables", [])
    if not variables:
        errors.append(f"{w}.variables: at least one variable is required")
    visible = {m.get("name") for m in model.get("meshes", [])} | {m.get("name") for m in md.get("meshes", [])}
    mesh = None
    for i, v in enumerate(variables):
        where = f"{w}.variables[{i}]"
        if _check_name(v.get("name"), f"{where}.name", errors, taken):
            taken[v["name"]] = "variable"
        if v.get("mesh") not in visible:
            errors.append(f"{where}.mesh: unknown mesh {v.get('mesh')!r}")
        elif mesh is None:
            mesh = find_mesh(model, v["mesh"])
        elif v["mesh"] != mesh.get("name"):
            errors.append(f"{where}.mesh: all the variables of a model are on the same mesh "
                          f"for now ({mesh.get('name')!r})")
        fem = v.get("fem", "lagrange")
        if fem not in FEMS:
            errors.append(f"{where}.fem: {', '.join(FEMS)} expected, got {fem!r}")
        _check_int(v.get("degree", 1), 0 if fem == "lagrange_discontinuous" else 1, 6,
                   f"{where}.degree", errors)
        c = v.get("components", 1)
        if c != "dim" and (isinstance(c, bool) or not isinstance(c, int) or c < 1):
            errors.append(f"{where}.components: 1, \"dim\" or a positive integer expected, got {c!r}")
    if mesh is None:
        return
    geom = geometry_of(model, mesh)
    if geom is None:
        return
    table = selection_table(model, geom)
    dim = geometry_dim(model, geom)
    var_kinds = {v.get("name"): variable_kind(v) for v in variables}

    # --- Imports
    known_kinds = dict(var_kinds)
    for i, im in enumerate(md.get("imports", [])):
        where = f"{w}.imports[{i}]"
        if _check_name(im.get("name"), f"{where}.name", errors, taken):
            taken[im["name"]] = "import"
        src = model_by_name(model, im.get("model"))
        if src is None or src is md:
            errors.append(f"{where}.model: unknown model {im.get('model')!r}")
            continue
        var = by_name(src.get("variables", []), im.get("variable"))
        if var is None:
            errors.append(f"{where}.variable: model {src.get('name')!r} has no variable {im.get('variable')!r}")
            continue
        if var.get("mesh") != mesh.get("name"):
            errors.append(f"{where}: {im.get('variable')!r} is on the mesh {var.get('mesh')!r}, "
                          f"not on {mesh.get('name')!r} (imports on the same mesh only, for now)")
        known_kinds[im.get("name")] = variable_kind(var)

    # --- Local fields
    _validate_fields(md.get("fields", []), f"{w}.fields", lambda f, where: table, errors, taken)

    # --- Modules: their properties and macros join the names of the model
    macro_owner: dict[str, int] = {}
    for i, mod in enumerate(md.get("modules", [])):
        spec = MODULE_SPECS.get(mod.get("type"))
        if spec is None:
            continue
        for p in spec["properties"]:
            if p not in taken:
                taken[p] = "property"
            elif taken[p] != "property":
                errors.append(f"{w}.modules[{i}]: its property {p!r} is already a {taken[p]}")
        for m in spec["macros"]:
            if m in macro_owner:
                errors.append(f"{w}.modules[{i}]: macro {m!r} is already defined by module "
                              f"{macro_owner[m] + 1} (one module of this type per model)")
            elif m in taken:
                errors.append(f"{w}.modules[{i}]: its macro {m!r} is already a {taken[m]}")
            else:
                macro_owner[m] = i
                taken[m] = "macro"
    anchored: set[str] = {c.get("variable") for c in md.get("conditions", [])
                          if c.get("type") in ANCHORING_CONDITIONS}
    for i, mod in enumerate(md.get("modules", [])):
        _validate_module(model, md, mod, f"{w}.modules[{i}]", table, dim, var_kinds, known_kinds,
                         errors, anchored)

    # --- Terms
    tnames = [t.get("name") for t in md.get("terms", [])]
    for i, t in enumerate(md.get("terms", [])):
        where = f"{w}.terms[{i}]"
        name = t.get("name")
        if not isinstance(name, str) or not _IDENT.match(name):
            errors.append(f"{where}.name: identifier expected, got {name!r}")
        elif tnames.count(name) > 1:
            errors.append(f"{where}.name: name {name!r} used more than once")
        expr = t.get("expr")
        if not isinstance(expr, str) or not expr.strip():
            errors.append(f"{where}.expr: GWFL expression expected")
        _check_selection(table, t.get("selection", ALL), None, f"{where}.selection", errors)
        if t.get("linear", "auto") not in ("auto", True, False):
            errors.append(f"{where}.linear: \"auto\", true or false expected")

    # --- Generic conditions
    for i, c in enumerate(md.get("conditions", [])):
        where = f"{w}.conditions[{i}]"
        ctype = c.get("type")
        if ctype not in CONDITIONS:
            errors.append(f"{where}.type: {', '.join(CONDITIONS)} expected, got {ctype!r}")
            continue
        var = c.get("variable")
        if var not in var_kinds:
            errors.append(f"{where}.variable: unknown variable {var!r}")
            continue
        _check_selection(table, c.get("selection"), BOUNDARY, f"{where}.selection", errors)
        for key, kind in CONDITIONS[ctype].items():
            _check_value(c.get(key), f"{where}.{key}", errors, var_kinds[var] if kind == "var" else kind, dim)
        if ctype == "normal_dirichlet" and var_kinds[var] != VECTOR:
            errors.append(f"{where}.variable: a normal condition needs a vector variable")
        if ctype in ("dirichlet", "normal_dirichlet"):
            method = c.get("method", "simplification" if ctype == "dirichlet" else "multiplier")
            allowed = DIRICHLET_METHODS if ctype == "dirichlet" else DIRICHLET_METHODS[1:]
            if method not in allowed:
                errors.append(f"{where}.method: {', '.join(allowed)} expected, got {method!r}")
            elif method == "penalization":
                _check_positive(c.get("penalty", 1e10), f"{where}.penalty", errors)
        for key in c:
            if key not in ("type", "variable", "selection", "method", "penalty") and key not in CONDITIONS[ctype]:
                errors.append(f"{where}.{key}: unknown field for a {ctype!r} condition")

    # --- Solver
    s = md.get("solver", {"type": "auto"})
    if s.get("type", "auto") not in SOLVER_TYPES:
        errors.append(f"{w}.solver.type: {', '.join(SOLVER_TYPES)} expected, got {s.get('type')!r}")
    if "max_iter" in s:
        _check_int(s["max_iter"], 1, 10000, f"{w}.solver.max_iter", errors)
    if "max_res" in s:
        _check_positive(s["max_res"], f"{w}.solver.max_res", errors)

    # --- Outputs
    onames = [q.get("name") for q in md.get("outputs", [])]
    for i, q in enumerate(md.get("outputs", [])):
        where = f"{w}.outputs[{i}]"
        if not _IDENT.match(str(q.get("name", ""))):
            errors.append(f"{where}.name: identifier expected")
        elif onames.count(q["name"]) > 1:
            errors.append(f"{where}.name: name {q['name']!r} used more than once")
        _check_value(q.get("expr"), f"{where}.expr", errors)
        _check_selection(table, q.get("selection", ALL), None, f"{where}.selection", errors)
    vtu = md.get("vtu", f"{md.get('name')}.vtu")
    if vtu is not None and (not isinstance(vtu, str) or not vtu.endswith(".vtu") or "/" in vtu):
        errors.append(f"{w}.vtu: file name ending with .vtu expected (or null)")
    _validate_graphics(md.get("graphics", []), f"{w}.graphics", errors)


def _validate_graphics(graphics, where, errors) -> None:
    if not isinstance(graphics, list):
        errors.append(f"{where}: list expected")
        return
    names = [g.get("name") for g in graphics if isinstance(g, dict)]
    for i, g in enumerate(graphics):
        w = f"{where}[{i}]"
        if not isinstance(g, dict):
            errors.append(f"{w}: object expected")
            continue
        if not isinstance(g.get("name"), str) or not g["name"].strip():
            errors.append(f"{w}.name: name expected")
        elif names.count(g["name"]) > 1:
            errors.append(f"{w}.name: name {g['name']!r} used more than once")
        if not isinstance(g.get("field"), str) or not g["field"]:
            errors.append(f"{w}.field: name of a result field expected")
        c = g.get("component")
        if c is not None and (isinstance(c, bool) or not isinstance(c, int) or c < 0):
            errors.append(f"{w}.component: component number or null expected")
        for key in ("warp", "bands"):
            if key in g and not isinstance(g[key], bool):
                errors.append(f"{w}.{key}: true or false expected")
        if "isolines" in g:
            _check_int(g["isolines"], 0, 100, f"{w}.isolines", errors)
        sc = g.get("scale")
        if sc is not None and (isinstance(sc, bool) or not isinstance(sc, (int, float)) or sc < 0):
            errors.append(f"{w}.scale: positive number or null expected")
        if g.get("cmap") is not None and g["cmap"] not in COLORMAPS:
            errors.append(f"{w}.cmap: {', '.join(COLORMAPS)} expected, got {g['cmap']!r}")
        r = g.get("range")
        if r is not None and (not isinstance(r, list) or len(r) != 2 or any(
                x is not None and (isinstance(x, bool) or not isinstance(x, (int, float))) for x in r)):
            errors.append(f"{w}.range: [min, max] expected (null for a bound taken from the values)")
        if g.get("arrows") is not None and not isinstance(g["arrows"], str):
            errors.append(f"{w}.arrows: name of a vector field or null expected")
        cut = g.get("cut")
        if cut is not None and (not isinstance(cut, list) or len(cut) != 2 or any(
                not isinstance(p, list) or not 2 <= len(p) <= 3 or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) for x in p) for p in cut)):
            errors.append(f"{w}.cut: two points [[xA, yA], [xB, yB]] expected")
        for key in g:
            if key not in ("name", "field") and key not in GRAPHIC_DEFAULTS:
                errors.append(f"{w}.{key}: unknown setting")


def _validate_module(model, md, mod, w, table, dim, var_kinds, known_kinds, errors, anchored) -> None:
    mtype = mod.get("type")
    spec = MODULE_SPECS.get(mtype)
    if spec is None:
        errors.append(f"{w}.type: {', '.join(MODULE_SPECS)} expected, got {mtype!r}")
        return
    # variables of the module
    for role, v in spec["variables"].items():
        name = mod.get(role, v["name"])
        kinds = known_kinds if v.get("import") else var_kinds
        if name not in kinds:
            what = "variable or import" if v.get("import") else "variable"
            errors.append(f"{w}.{role}: unknown {what} {name!r} (the {v['label']})")
        elif kinds[name] != v["kind"]:
            errors.append(f"{w}.{role}: the {v['label']} {name!r} should be a {v['kind']} variable")
    if "requires" in spec:
        other, role = spec["requires"]
        name = mod.get(role, spec["variables"][role]["name"])
        if not any(m.get("type") == other and m.get("variable", MODULE_SPECS[other]["variables"]["variable"]["name"])
                   == name for m in md.get("modules", [])):
            errors.append(f"{w}: requires a {MODULE_SPECS[other]['label'].lower()} module on {name!r}")
    # options and values
    known = {"type", "properties", "sources", "conditions"} | set(spec["variables"])
    for opt, value in mod.items():
        if opt in known:
            continue
        if opt in spec["options"]:
            if value not in spec["options"][opt]:
                errors.append(f"{w}.{opt}: {', '.join(spec['options'][opt])} expected, got {value!r}")
            elif opt == "plane" and dim is not None and dim != 2:
                errors.append(f"{w}.plane: only meaningful in 2D")
        elif opt in spec["values"]:
            _check_value(value, f"{w}.{opt}", errors, spec["values"][opt], dim)
        else:
            errors.append(f"{w}.{opt}: unknown option for {mtype!r}")
    # properties, per domain selection
    props = mod.get("properties", {})
    for prop, label in spec["properties"].items():
        pieces = props.get(prop)
        where = f"{w}.properties.{prop}"
        if not isinstance(pieces, list) or not pieces:
            errors.append(f"{where}: missing {label} (a value per domain selection)")
            continue
        for j, pc in enumerate(pieces):
            _check_selection(table, pc.get("selection", ALL), DOMAIN, f"{where}[{j}].selection", errors)
            if "material" in pc:
                mat = material(model, pc["material"])
                if mat is None:
                    errors.append(f"{where}[{j}].material: unknown material {pc['material']!r}")
                elif prop not in (mat.get("properties") or {}):
                    errors.append(f"{where}[{j}].material: {pc['material']!r} has no {prop!r}")
            else:
                _check_value(pc.get("value"), f"{where}[{j}].value", errors)
    for prop in props:
        if prop not in spec["properties"]:
            errors.append(f"{w}.properties.{prop}: not a property of {mtype!r}")
    # sources and conditions
    for i, src in enumerate(mod.get("sources", [])):
        _check_selection(table, src.get("selection", ALL), DOMAIN, f"{w}.sources[{i}].selection", errors)
        _check_value(src.get("value"), f"{w}.sources[{i}].value", errors, spec["source"], dim)
    bc_specs = spec["conditions"]
    is_anchored = False
    for i, bc in enumerate(mod.get("conditions", [])):
        where = f"{w}.conditions[{i}]"
        btype = bc.get("type")
        if btype not in bc_specs:
            errors.append(f"{where}.type: {', '.join(bc_specs)} expected, got {btype!r}")
            continue
        _check_selection(table, bc.get("selection"), BOUNDARY, f"{where}.selection", errors)
        for key, kind in bc_specs[btype].items():
            _check_value(bc.get(key), f"{where}.{key}", errors, kind, dim)
        for key in bc:
            if key not in ("type", "selection") and key not in bc_specs[btype]:
                errors.append(f"{where}.{key}: unknown field for a {btype!r} condition")
        is_anchored |= btype in spec["anchoring"]
    var = mod.get("variable", spec["variables"].get("variable", {}).get("name"))
    if spec["anchoring"] and not is_anchored and var not in anchored and not errors:
        errors.append(f"{w}.conditions: {spec['anchoring_message']}")


def with_defaults(model: dict) -> dict:
    """Copy of the model completed with the optional default values."""
    m = copy.deepcopy(model)
    for key in ("parameters", "materials", "fields", "models", "solvers", "outputs", "graphics"):
        m.setdefault(key, [])
    for geom, _ in all_geometries(m):
        geom.setdefault("selections", {})
    for mesh, _ in all_meshes(m):
        if mesh.get("mesher") == "structured":
            mesh.setdefault("element", "simplex")
        if mesh.get("mesher") in ("getfem", "gmsh"):
            mesh.setdefault("order", 1)
    geoms = m.get("geometries", [])
    for f in m["fields"]:
        if len(geoms) == 1:
            f.setdefault("geometry", geoms[0].get("name"))
        for v in f.get("values", []):
            v.setdefault("selection", ALL)
    for md in m["models"]:
        for key in ("parameters", "geometries", "meshes", "fields", "imports", "variables",
                    "modules", "terms", "conditions", "outputs", "graphics"):
            md.setdefault(key, [])
        md.setdefault("solver", {"type": "auto"})
        md["solver"].setdefault("type", "auto")
        md.setdefault("vtu", f"{md.get('name')}.vtu")
        for f in md["fields"]:
            for v in f.get("values", []):
                v.setdefault("selection", ALL)
        for v in md["variables"]:
            v.setdefault("fem", "lagrange")
            v.setdefault("degree", 1)
            v.setdefault("components", 1)
        for mod in md["modules"]:
            spec = MODULE_SPECS.get(mod.get("type"), {})
            for role, v in spec.get("variables", {}).items():
                mod.setdefault(role, v["name"])
            for key, value in spec.get("value_defaults", {}).items():
                mod.setdefault(key, value)
            mod.setdefault("properties", {})
            for pieces in mod["properties"].values():
                for pc in pieces:
                    pc.setdefault("selection", ALL)
            mod.setdefault("sources", [])
            mod.setdefault("conditions", [])
            for src in mod["sources"]:
                src.setdefault("selection", ALL)
        for t in md["terms"]:
            t.setdefault("selection", ALL)
            t.setdefault("linear", "auto")
        for c in md["conditions"]:
            if c.get("type") == "dirichlet":
                c.setdefault("method", "simplification")
            elif c.get("type") == "normal_dirichlet":
                c.setdefault("method", "multiplier")
        for q in md["outputs"]:
            q.setdefault("selection", ALL)
    for s in m["solvers"]:
        if s.get("type") == "staggered":
            s.setdefault("max_iter", 20)
            s.setdefault("tolerance", 1e-6)
            s.setdefault("relaxation", 1.0)
    return m
