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


"""Editing a model from a user interface, independently of any toolkit.

- `tree(model)`: the tree of the model, as shown in the model browser;
- `fields(model, path)`: the form fields of a node;
- `apply(model, field, value)`: change a field, keeping the model consistent
  (renaming a selection updates its uses, changing the type of a boundary
  condition resets its values, ...);
- `actions(model, path)` / `do_action(...)`: add and remove items;
- `errors_by_node` / `errors_at`: place the messages of `model.validate`.

A node or a field is designated by its path in the model dictionary, e.g.
("physics", "boundary_conditions", 0) or ("materials", 0, "properties", "E").
Functions modify the model in place.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field

from . import model as M, parameters as P

Path = tuple

PRIMITIVES = ("rectangle", "ball", "cylinder")
RESULTS = "_results"          # first element of the path of the result nodes

_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")


@dataclass
class Node:
    path: Path
    label: str
    kind: str
    children: list["Node"] = field(default_factory=list)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass
class Field:
    key: Path                     # where the value lives in the model
    label: str
    kind: str                     # text, ident, value, vector, numbers, ints, int, number,
                                  # choice, names, file, info
    choices: tuple = ()           # ((value, label), ...) for choice and names
    size: int | None = None       # number of components (vector, numbers, ints)
    op: str | None = None         # special operation instead of a plain assignment
    help: str = ""
    remove: Path | None = None    # what a "remove" button next to the field deletes

    @property
    def removable(self) -> bool:
        return self.remove is not None


# ------------------------------------------------------------------ access

def get_at(model: dict, key: Path, default=None):
    obj = model
    for k in key:
        try:
            obj = obj[k]
        except (KeyError, IndexError, TypeError):
            return default
    return obj


def set_at(model: dict, key: Path, value) -> None:
    obj = model
    for k, nxt in zip(key[:-1], key[1:]):
        if isinstance(obj, dict) and k not in obj:
            obj[k] = [] if isinstance(nxt, int) else {}
        obj = obj[k]
    obj[key[-1]] = value


def physics_geometry(model: dict) -> dict | None:
    mesh = M.physics_mesh(model)
    return M.geometry_of(model, mesh) if mesh else None


def physics_dim(model: dict) -> int:
    geom = physics_geometry(model)
    return (M.geometry_dim(model, geom) if geom else None) or 2


def geometry_dim(model: dict, gi: int) -> int:
    geoms = model.get("geometries", [])
    return (M.geometry_dim(model, geoms[gi]) if gi < len(geoms) else None) or 2


def selections_of_kind(model: dict, kind: str | None, geom: dict | None = None) -> list[str]:
    geom = geom if geom is not None else physics_geometry(model)
    if geom is None:
        return []
    return [n for n, s in M.selection_table(model, geom).items() if kind is None or s["kind"] == kind]


def unique_name(base: str, used) -> str:
    used = set(used)
    if base not in used:
        return base
    i = 2
    while f"{base}{i}" in used:
        i += 1
    return f"{base}{i}"


# ------------------------------------------------------------------ values

def format_number(x) -> str:
    """Shortest exact text: 200000, 0.3, 1e8, 2.1e-5."""
    if isinstance(x, bool) or not isinstance(x, float):
        return str(x)
    if x != 0 and (abs(x) >= 1e6 or abs(x) < 1e-4):
        for digits in range(17):
            mant, exp = f"{x:.{digits}e}".split("e")
            if float(f"{mant}e{exp}") == x:
                return f"{mant}e{int(exp)}"
    if x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    return repr(x)


def format_value(v) -> str:
    if isinstance(v, (int, float)):
        return format_number(v)
    if isinstance(v, list):
        return "[" + ", ".join(format_value(c) for c in v) + "]"
    return "" if v is None else str(v)


def parse_value(text: str):
    """A number if the text is one, otherwise a GWFL expression (string)."""
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_number(text: str, integer: bool = False):
    """A number, or an expression of the parameters (kept as text, evaluated
    when meshing; validation reports the errors)."""
    text = text.strip()
    try:
        return int(text) if integer else float(text)
    except ValueError:
        pass
    if not text:
        raise ValueError(f"{'integer' if integer else 'number'} expected")
    return text


def split_vector(v, size: int) -> list | None:
    """Components of a vector value for per-component editing, or None if the
    value is a GWFL expression that is not written as [a, b, ...]."""
    if isinstance(v, list):
        return (list(v) + [0.0] * size)[:size] if len(v) != size else list(v)
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            parts, depth, cur = [], 0, ""
            for ch in s[1:-1]:
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append(cur)
                    cur = ""
                else:
                    cur += ch
            parts.append(cur)
            if len(parts) == size:
                return [parse_value(p) for p in parts]
        return None
    return [0.0] * size


# ------------------------------------------------------------------ tree

def _shape_label(shape: dict) -> str:
    t = shape.get("type", "?")
    return t.capitalize() if t in PRIMITIVES else t.capitalize()


def _shape_node(shape: dict, path: Path) -> Node:
    node = Node(path, _shape_label(shape), "shape")
    if shape.get("type") in M.SHAPES_BOOLEAN:
        for j, s in enumerate(shape.get("of", [])):
            node.children.append(_shape_node(s, path + ("of", j)))
    return node


def _bc_label(bc: dict) -> str:
    return f"{bc.get('type', '?').replace('_', ' ').capitalize()} — {bc.get('selection', '?')}"


def tree(model: dict, result_fields=(), has_results: bool = False) -> Node:
    root = Node((), model.get("name", "Model"), "model")

    params = Node(("parameters",), "Parameters", "parameters")
    vals, _ = P.values(model)
    for i, p in enumerate(model.get("parameters", [])):
        name = p.get("name", "?")
        v = p.get("value")
        label = f"{name} = {format_value(v)}"
        if isinstance(v, str) and name in vals:
            label += f"  ({format_number(vals[name])})"
        params.children.append(Node(("parameters", i), label, "parameter"))
    root.children.append(params)

    geoms = Node(("geometries",), "Geometries", "geometries")
    for i, g in enumerate(model.get("geometries", [])):
        src = g.get("source", {})
        gn = Node(("geometries", i), f"{g.get('name', '?')} ({src.get('type', '?')})", "geometry")
        if src.get("type") == "csg" and isinstance(src.get("shape"), dict):
            gn.children.append(_shape_node(src["shape"], ("geometries", i, "source", "shape")))
        sels = Node(("geometries", i, "selections"), "Selections", "selections")
        for name, s in M.selection_table(model, g).items():
            kind = "selection" if not s["builtin"] else "builtin_selection"
            label = name if not s["builtin"] else f"{name} (built-in)"
            sels.children.append(Node(("geometries", i, "selections", name), label, kind))
        gn.children.append(sels)
        geoms.children.append(gn)
    root.children.append(geoms)

    meshes = Node(("meshes",), "Meshes", "meshes")
    used = (M.physics_mesh(model) or {}).get("name")
    for i, m in enumerate(model.get("meshes", [])):
        label = f"{m.get('name', '?')} ({m.get('mesher', '?')})"
        if m.get("name") == used:
            label += " — used"
        meshes.children.append(Node(("meshes", i), label, "mesh"))
    root.children.append(meshes)

    root.children.append(Node(("discretization",), "Discretization", "discretization"))

    fields_node = Node(("fields",), "Fields", "fields")
    for i, f in enumerate(model.get("fields", [])):
        fields_node.children.append(Node(("fields", i), f.get("name", "?"), "field"))
    root.children.append(fields_node)

    mats = Node(("materials",), "Materials", "materials")
    for i, mat in enumerate(model.get("materials", [])):
        mats.children.append(Node(("materials", i),
                                  f"{mat.get('name', '?')} — {mat.get('selection', M.ALL)}", "material"))
    root.children.append(mats)

    phys = model.get("physics", {})
    spec = M.PHYSICS_SPECS.get(phys.get("type"), {})
    pn = Node(("physics",), spec.get("label", "Physics"), "physics")
    srcs = Node(("physics", "sources"), "Sources", "sources")
    for i, s in enumerate(phys.get("sources", [])):
        srcs.children.append(Node(("physics", "sources", i), f"Source — {s.get('selection', M.ALL)}", "source"))
    bcs = Node(("physics", "boundary_conditions"), "Boundary conditions", "bcs")
    for i, bc in enumerate(phys.get("boundary_conditions", [])):
        bcs.children.append(Node(("physics", "boundary_conditions", i), _bc_label(bc), "bc"))
    pn.children += [srcs, bcs]
    root.children.append(pn)

    root.children.append(Node(("study",), "Study (stationary)", "study"))

    outs = Node(("outputs",), "Outputs", "outputs")
    for i, q in enumerate(model.get("outputs", {}).get("quantities", [])):
        outs.children.append(Node(("outputs", "quantities", i), q.get("name", "?"), "quantity"))
    root.children.append(outs)

    if has_results or result_fields:
        res = Node((RESULTS,), "Results", "results")
        for name in result_fields:
            res.children.append(Node((RESULTS, name), name, "result_field"))
        root.children.append(res)
    return root


def find_node(root: Node, path: Path) -> Node | None:
    return next((n for n in root.walk() if n.path == path), None)


# ------------------------------------------------------------------ errors

def error_path(message: str) -> tuple[Path, str]:
    """('materials', 0, 'properties', 'E'), 'missing ...' from a validation message."""
    head, sep, text = message.partition(": ")
    if not sep or " " in head:
        return (), message
    tokens = []
    pos = 0
    for m in _TOKEN.finditer(head):
        if m.start() != pos and head[pos:m.start()] != ".":
            return (), message
        tokens.append(m.group(1) if m.group(1) is not None else int(m.group(2)))
        pos = m.end()
    return tuple(tokens), text


def errors_by_node(root: Node, errors: list[str]) -> dict[Path, list[str]]:
    """Each message goes to the deepest node whose path prefixes the error path."""
    paths = {n.path for n in root.walk()}
    out: dict[Path, list[str]] = {}
    for e in errors:
        p, _ = error_path(e)
        while p and p not in paths:
            p = p[:-1]
        out.setdefault(p, []).append(e)
    return out


def errors_at(errors: list[str], key: Path) -> list[str]:
    """Messages about a field (its path, or a sub-path such as a vector component)."""
    out = []
    for e in errors:
        p, text = error_path(e)
        if p[:len(key)] == key and p:
            out.append(text)
    return out


# ------------------------------------------------------------------ fields

def _vec_field(key, label, kind, dim, help=""):
    return Field(key, label, kind, size=dim, help=help)


def _shape_fields(model, path, dim) -> list[Field]:
    shape = get_at(model, path, {})
    t = shape.get("type")
    types = PRIMITIVES + M.SHAPES_BOOLEAN
    out = [Field(path + ("type",), "Type", "choice", tuple((x, x) for x in types), op="shape_type")]
    if t == "rectangle":
        out += [_vec_field(path + ("min",), "Lower corner", "numbers", dim),
                _vec_field(path + ("max",), "Upper corner", "numbers", dim)]
    elif t == "ball":
        out += [_vec_field(path + ("center",), "Center", "numbers", dim),
                Field(path + ("radius",), "Radius", "number")]
    elif t == "cylinder":
        out += [_vec_field(path + ("origin",), "Origin", "numbers", dim),
                _vec_field(path + ("axis",), "Axis", "numbers", dim),
                Field(path + ("length",), "Length", "number"),
                Field(path + ("radius",), "Radius", "number")]
    else:
        out.append(Field(path, "", "info", help="The first shape is combined with the following ones."))
    return out


def _criteria_fields(model, gi, name, dim) -> list[Field]:
    base = ("geometries", gi, "selections", name)
    sel = get_at(model, base, {})
    kind = sel.get("kind", M.BOUNDARY)
    crit = sel.get("criteria", {})
    ct = crit.get("type")
    c = base + ("criteria",)
    from . import geometry2d as G
    planar = G.shape_of(P.resolved(model)["geometries"][gi].get("source", {})) is not None
    possible = [t for t, kinds in M.CRITERIA.items() if kind in kinds and (t not in ("edges", "regions") or planar)]
    out = [Field(c + ("type",), "Criterion", "choice", tuple((t, t) for t in possible),
                 op="criteria_type")]
    if ct == "plane":
        out += [_vec_field(c + ("point",), "Point", "numbers", dim),
                _vec_field(c + ("normal",), "Normal", "numbers", dim)]
    elif ct == "normal":
        out += [_vec_field(c + ("direction",), "Direction", "numbers", dim),
                Field(c + ("angle",), "Tolerance (degrees)", "number")]
    elif ct == "in_box":
        out += [_vec_field(c + ("min",), "Lower corner", "numbers", dim),
                _vec_field(c + ("max",), "Upper corner", "numbers", dim)]
    elif ct == "in_ball":
        out += [_vec_field(c + ("center",), "Center", "numbers", dim),
                Field(c + ("radius",), "Radius", "number")]
    elif ct == "physical":
        out.append(Field(c + ("tag",), "Physical group number", "int"))
    elif ct == "edges":
        shape = G.shape_of(P.resolved(model)["geometries"][gi].get("source", {}))
        labels = [G.side_label(shape, r) for r in crit.get("edges", [])] if shape else []
        out.append(Field(c + ("edges",), "", "info",
                         help=("Edges: " + ", ".join(labels) if labels else "No edge yet.")
                         + "\nClick edges in the view to add or remove them."))
    elif ct == "regions":
        shape = G.shape_of(P.resolved(model)["geometries"][gi].get("source", {}))
        labels = [G.region_label(shape, r) for r in crit.get("regions", [])] if shape else []
        out.append(Field(c + ("regions",), "", "info",
                         help=("Regions (in the shapes): " + "; ".join(labels) if labels else "No region yet.")
                         + "\nClick in the view to add or remove the region under the mouse."))
    elif ct == "union":
        geom = model["geometries"][gi]
        table = M.selection_table(model, geom)
        names = [n for n, s in table.items() if s["kind"] == kind and n != name
                 and s.get("criteria", {}).get("type") != "union"]
        out.append(Field(c + ("of",), "Selections", "names", tuple((n, n) for n in names)))
    return out


def fields(model: dict, path: Path) -> list[Field]:
    """Form fields of the node at `path`."""
    n = len(path)
    if path == ():
        return [Field(("name",), "Model name", "text")]

    if path[0] == "geometries" and n >= 2:
        gi = path[1]
        geom = get_at(model, ("geometries", gi), {})
        dim = geometry_dim(model, gi)
        if n == 2:
            src = geom.get("source", {})
            base = ("geometries", gi, "source")
            out = [Field(("geometries", gi, "name"), "Name", "ident", op="rename_geometry"),
                   Field(base + ("type",), "Source", "choice",
                         (("box", "box"), ("csg", "constructive geometry (csg)"),
                          ("cad", "CAD file (STEP, IGES, BREP)"), ("mesh_file", "mesh file (.msh)")),
                         op="source_type")]
            t = src.get("type")
            if t == "box":
                out += [Field(base + ("size",), "Dimension", "choice",
                              ((1, "1D"), (2, "2D"), (3, "3D")), op="box_dim"),
                        _vec_field(base + ("origin",), "Origin", "numbers", dim),
                        _vec_field(base + ("size",), "Size", "numbers", dim)]
            elif t == "csg":
                out.append(Field(base + ("dim",), "Dimension", "choice", ((2, "2D"), (3, "3D")),
                                 op="csg_dim"))
            elif t in ("cad", "mesh_file"):
                out.append(Field(base + ("file",), "File", "file"))
            return out
        if path[2] == "source":
            return _shape_fields(model, path, dim)
        if path[2] == "selections" and n == 3:
            return [Field(path, "", "info", help="Named parts of the geometry, used by materials, "
                                                 "conditions and outputs.")]
        if path[2] == "selections" and n == 4:
            name = path[3]
            table = M.selection_table(model, geom)
            if table.get(name, {}).get("builtin"):
                return [Field(path, "", "info",
                              help=f"Built-in {table[name]['kind']} selection.")]
            base = ("geometries", gi, "selections", name)
            return [Field(base, "Name", "ident", op="rename_selection"),
                    Field(base + ("kind",), "Kind", "choice",
                          ((M.BOUNDARY, "boundary"), (M.DOMAIN, "domain")), op="selection_kind")
                    ] + _criteria_fields(model, gi, name, dim)

    if path == ("parameters",):
        return [Field(path, "", "info", help="Named constants, usable in every numerical field "
                      "(\"L/2\") and expression. A parameter may use the ones listed before it.")]
    if path[0] == "parameters" and n == 2:
        vals, _ = P.values(model)
        name = get_at(model, path + ("name",))
        out = [Field(path + ("name",), "Name", "ident", op="rename_parameter"),
               Field(path + ("value",), "Value", "value",
                     help="number or expression: + - * /, pi, " + ", ".join(sorted(P.FUNCTIONS))),
               Field(path + ("description",), "Description", "text")]
        if isinstance(get_at(model, path + ("value",)), str) and name in vals:
            out.append(Field(path, "", "info", help=f"= {format_number(vals[name])}"))
        return out
    if path == ("fields",):
        return [Field(path, "", "info", help="Fields defined by a value per domain selection (the "
                      "last one wins), usable in every expression, including material properties.")]
    if path[0] == "fields" and n == 2:
        out = [Field(path + ("name",), "Name", "ident", op="rename_field"),
               Field(path + ("description",), "Description", "text")]
        domains = tuple((s, s) for s in selections_of_kind(model, M.DOMAIN))
        for j, _ in enumerate(get_at(model, path + ("values",), [])):
            out += [Field(path + ("values", j, "selection"), f"Domain {j + 1}", "choice", domains),
                    Field(path + ("values", j, "value"), f"Value {j + 1}", "value",
                          remove=path + ("values", j))]
        return out

    if path == ("geometries",):
        return [Field(path, "", "info", help="Geometries and their named selections.")]
    if path == ("meshes",):
        return [Field(path, "", "info", help="Several meshes may be defined for one geometry; "
                                             "the physics uses one of them.")]

    if path[0] == "meshes" and n == 2:
        mi = path[1]
        mesh = get_at(model, path, {})
        geom = M.geometry_of(model, mesh) or {}
        stype = geom.get("source", {}).get("type")
        dim = (M.geometry_dim(model, geom) if geom else None) or 2
        meshers = [x for x, srcs in M.MESHERS.items() if stype in srcs] or list(M.MESHERS)
        out = [Field(path + ("name",), "Name", "ident", op="rename_mesh"),
               Field(path + ("geometry",), "Geometry", "choice",
                     tuple((g["name"], g["name"]) for g in model.get("geometries", []))),
               Field(path + ("mesher",), "Mesher", "choice", tuple((x, x) for x in meshers),
                     op="mesher")]
        mesher = mesh.get("mesher")
        if mesher == "structured":
            out += [_vec_field(path + ("divisions",), "Divisions", "ints", dim),
                    Field(path + ("element",), "Elements", "choice",
                          (("simplex", "triangles / tetrahedra"),
                           ("quad", "quadrilaterals / hexahedra")))]
        elif mesher in ("getfem", "gmsh"):
            key = "h" if mesher == "getfem" else "size"
            out += [Field(path + (key,), "Element size", "number"),
                    Field(path + ("order",), "Geometric order", "choice",
                          ((1, "1 (straight)"), (2, "2 (curved)"), (3, "3")))]
        return out

    if path == ("discretization",):
        return [Field(("discretization", "degree"), "Finite element degree", "choice",
                      tuple((d, f"P{d}") for d in range(1, 7)))]

    if path == ("materials",):
        return [Field(path, "", "info", help="The last material listed wins where several overlap.")]
    if path[0] == "materials" and n == 2:
        spec = M.PHYSICS_SPECS.get(model.get("physics", {}).get("type"), {})
        mat = get_at(model, path, {})
        out = [Field(path + ("name",), "Name", "ident"),
               Field(path + ("selection",), "Domain", "choice",
                     tuple((s, s) for s in selections_of_kind(model, M.DOMAIN)))]
        props = mat.get("properties", {})
        for p, label in spec.get("properties", {}).items():
            out.append(Field(path + ("properties", p), f"{p} ({label})", "value"))
        for p in props:
            if p not in spec.get("properties", {}):
                out.append(Field(path + ("properties", p), p, "value",
                                 remove=path + ("properties", p)))
        return out

    if path == ("physics",):
        phys = model.get("physics", {})
        spec = M.PHYSICS_SPECS.get(phys.get("type"), {})
        out = [Field(("physics", "type"), "Physics", "choice",
                     tuple((k, s["label"]) for k, s in M.PHYSICS_SPECS.items()), op="physics_type"),
               Field(("physics", "mesh"), "Mesh", "choice",
                     tuple((m["name"], m["name"]) for m in model.get("meshes", []))),
               Field(("physics", "variable"), "Unknown", "ident")]
        for opt, values in spec.get("options", {}).items():
            if opt == "plane" and physics_dim(model) != 2:
                continue
            out.append(Field(("physics", opt), "2D model" if opt == "plane" else opt, "choice",
                             tuple((v, f"plane {v}" if opt == "plane" else v) for v in values)))
        return out

    spec = M.PHYSICS_SPECS.get(model.get("physics", {}).get("type"), {})
    dim = physics_dim(model)
    if path == ("physics", "sources"):
        return [Field(path, "", "info", help="Volume sources (right-hand side).")]
    if path[:2] == ("physics", "sources") and n == 3:
        kind = "vector" if spec.get("source") == M.VECTOR else "value"
        return [Field(path + ("selection",), "Domain", "choice",
                      tuple((s, s) for s in selections_of_kind(model, M.DOMAIN))),
                Field(path + ("value",), "Value", kind, size=dim if kind == "vector" else None)]
    if path == ("physics", "boundary_conditions"):
        return [Field(path, "", "info", help="Boundaries without condition are free "
                                             "(zero flux, zero traction).")]
    if path[:2] == ("physics", "boundary_conditions") and n == 3:
        bc = get_at(model, path, {})
        specs = spec.get("boundary_conditions", {})
        out = [Field(path + ("type",), "Type", "choice",
                     tuple((t, t.replace("_", " ")) for t in specs), op="bc_type"),
               Field(path + ("selection",), "Boundary", "choice",
                     tuple((s, s) for s in selections_of_kind(model, M.BOUNDARY)))]
        for key, kind in specs.get(bc.get("type"), {}).items():
            label = {"value": "Value", "h": "Exchange coefficient h", "T_ext": "External temperature"}.get(key, key)
            if kind == M.VECTOR:
                out.append(Field(path + (key,), label, "vector", size=dim))
            else:
                out.append(Field(path + (key,), label, "value"))
        return out

    if path == ("study",):
        return [Field(path, "", "info", help="Stationary study (the only one for now).")]
    if path == ("outputs",):
        return [Field(("outputs", "vtu"), "VTU file", "text")]
    if path[:2] == ("outputs", "quantities") and n == 3:
        return [Field(path + ("name",), "Name", "ident"),
                Field(path + ("expr",), "Integrand (GWFL)", "value"),
                Field(path + ("selection",), "Integrated on", "choice",
                      tuple((s, s) for s in selections_of_kind(model, None)))]
    return []


# ------------------------------------------------------------------ changes

def _default_shape(t: str, dim: int) -> dict:
    if t == "rectangle":
        return {"type": "rectangle", "min": [0.0] * dim, "max": [1.0] * dim}
    if t == "ball":
        return {"type": "ball", "center": [0.0] * dim, "radius": 0.5}
    if t == "cylinder":
        return {"type": "cylinder", "origin": [0.0] * dim, "axis": [0.0] * (dim - 1) + [1.0],
                "length": 1.0, "radius": 0.5}
    return {"type": t, "of": [_default_shape("rectangle", dim), _default_shape("ball", dim)]}


def _shape_in_geometry(model: dict, gi: int, t: str) -> dict:
    """A new primitive for a 2D geometry: in its middle, at its scale (the
    default shape otherwise)."""
    import numpy as np
    from . import geometry2d as G
    dim = geometry_dim(model, gi)
    try:
        lo, hi = G.bounds(G.shape_of(P.resolved(model)["geometries"][gi]["source"]))
    except (KeyError, TypeError, ValueError, AttributeError, P.ParameterError):
        return _default_shape(t, dim)
    if dim != 2 or not np.all(np.isfinite([*lo, *hi])) or np.any(hi <= lo):
        return _default_shape(t, dim)
    r = lambda v: float(f"{v:.3g}")          # noqa: E731
    c, e = (lo + hi) / 2, float((hi - lo).min())
    if t == "rectangle":
        return {"type": "rectangle", "min": [r(x - e / 8) for x in c], "max": [r(x + e / 8) for x in c]}
    if t == "ball":
        return {"type": "ball", "center": [r(x) for x in c], "radius": r(e / 8)}
    return _default_shape(t, dim)


def _default_source(t: str, dim: int) -> dict:
    if t == "box":
        return {"type": "box", "origin": [0.0] * dim, "size": [1.0] * dim}
    if t == "csg":
        return {"type": "csg", "dim": max(dim, 2), "shape": _default_shape("difference", max(dim, 2))}
    return {"type": t, "file": ""}


def _default_criteria(t: str, dim: int, kind: str) -> dict:
    e = [1.0] + [0.0] * (dim - 1)
    return {
        "plane": {"type": "plane", "point": [0.0] * dim, "normal": e},
        "normal": {"type": "normal", "direction": e, "angle": 1.0},
        "in_box": {"type": "in_box", "min": [0.0] * dim, "max": [1.0] * dim},
        "in_ball": {"type": "in_ball", "center": [0.0] * dim, "radius": 0.5},
        "physical": {"type": "physical", "tag": 1},
        "union": {"type": "union", "of": []},
        "edges": {"type": "edges", "edges": []},
        "regions": {"type": "regions", "regions": []},
    }[t]


def _default_mesh_params(mesher: str, dim: int) -> dict:
    return {"structured": {"divisions": [10] * dim, "element": "simplex"},
            "getfem": {"h": 0.1, "order": 1},
            "gmsh": {"size": 0.1, "order": 1},
            "import": {}}[mesher]


def _zero(kind: str, dim: int):
    return [0.0] * dim if kind == M.VECTOR else 0.0


def _default_bc(btype: str, spec: dict, selection: str, dim: int) -> dict:
    bc = {"type": btype, "selection": selection}
    for key, kind in spec["boundary_conditions"][btype].items():
        bc[key] = 20.0 if key == "T_ext" else (1.0 if key == "h" else _zero(kind, dim))
    return bc


_WORD = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])"


def rename_identifier(model: dict, old: str, new: str) -> None:
    """Rename a parameter or a field, and its uses in every expression (whole
    words only: Grad_L is not a use of L)."""
    pattern = re.compile(_WORD.format(re.escape(old)))

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "name" and v == old and obj in named:
                    obj[k] = new
                elif isinstance(v, str) and k not in ("name", "selection", "geometry", "mesh",
                                                       "type", "file", "description", "vtu",
                                                       "kind", "element", "mesher"):
                    obj[k] = pattern.sub(new, v)
                elif k != "of" or not all(isinstance(x, str) for x in v):  # union: names
                    walk(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    obj[i] = pattern.sub(new, v)
                else:
                    walk(v)
    named = [p for p in model.get("parameters", []) if p.get("name") == old] + \
        [f for f in model.get("fields", []) if f.get("name") == old]
    walk({k: v for k, v in model.items() if not k.startswith("_")})


def _rename_selection(model, gi, old, new) -> None:
    geom = model["geometries"][gi]
    geom["selections"] = {(new if k == old else k): v for k, v in geom.get("selections", {}).items()}
    for sel in geom["selections"].values():
        crit = sel.get("criteria", {})
        if crit.get("type") == "union":
            crit["of"] = [new if o == old else o for o in crit.get("of", [])]
    if physics_geometry(model) is not geom:
        return
    phys = model.get("physics", {})
    items = model.get("materials", []) + phys.get("sources", []) + phys.get("boundary_conditions", []) \
        + model.get("outputs", {}).get("quantities", [])
    for it in items:
        if it.get("selection") == old:
            it["selection"] = new


def apply(model: dict, f: Field, value) -> Path | None:
    """Set a field. Returns the new path of the node when it changed (renaming
    a selection changes its path), None otherwise."""
    key = f.key
    if f.op is None:
        set_at(model, key, value)
        return None
    if f.op == "rename_geometry":
        old = get_at(model, key)
        set_at(model, key, value)
        for mesh in model.get("meshes", []):
            if mesh.get("geometry") == old:
                mesh["geometry"] = value
    elif f.op in ("rename_parameter", "rename_field"):
        old = get_at(model, key)
        if value != old:
            rename_identifier(model, old, value)
    elif f.op == "rename_mesh":
        old = get_at(model, key)
        set_at(model, key, value)
        if model.get("physics", {}).get("mesh") == old:
            model["physics"]["mesh"] = value
    elif f.op == "rename_selection":
        gi, old = key[1], key[3]
        if value == old:
            return None
        if value in M.selection_table(model, model["geometries"][gi]):
            raise ValueError(f"the name {value!r} is already used")
        _rename_selection(model, gi, old, value)
        return ("geometries", gi, "selections", value)
    elif f.op == "source_type":
        gi = key[1]
        if get_at(model, key) != value:
            model["geometries"][gi]["source"] = _default_source(value, geometry_dim(model, gi))
    elif f.op == "box_dim":
        src = get_at(model, key[:-1])
        dim = int(value)
        src["origin"] = (list(src.get("origin", [])) + [0.0] * dim)[:dim]
        src["size"] = (list(src.get("size", [])) + [1.0] * dim)[:dim]
        _fit_dimension(model, key[1], dim)
    elif f.op == "csg_dim":
        dim = int(value)
        src = get_at(model, key[:-1])
        if src.get("dim") != dim:
            src["dim"] = dim
            src["shape"] = _default_shape("difference", dim)
            _fit_dimension(model, key[1], dim)
    elif f.op == "shape_type":
        path = key[:-1]
        shape = get_at(model, path)
        old = shape.get("type")
        dim = geometry_dim(model, key[1])
        if value == old:
            return None
        if value in M.SHAPES_BOOLEAN and old in M.SHAPES_BOOLEAN:
            shape["type"] = value
        elif value in M.SHAPES_BOOLEAN:
            set_at(model, path, {"type": value, "of": [shape, _default_shape("ball", dim)]})
        else:
            set_at(model, path, _default_shape(value, dim))
    elif f.op == "selection_kind":
        sel = get_at(model, key[:-1])
        sel["kind"] = value
        if value not in M.CRITERIA.get(sel.get("criteria", {}).get("type"), ()):
            dim = geometry_dim(model, key[1])
            sel["criteria"] = _default_criteria("in_box", dim, value)
    elif f.op == "criteria_type":
        sel = get_at(model, key[:-2])
        if sel.get("criteria", {}).get("type") != value:
            sel["criteria"] = _default_criteria(value, geometry_dim(model, key[1]), sel.get("kind"))
    elif f.op == "mesher":
        mesh = get_at(model, key[:-1])
        if mesh.get("mesher") != value:
            geom = M.geometry_of(model, mesh) or {}
            dim = (M.geometry_dim(model, geom) if geom else None) or 2
            for k in ("divisions", "element", "h", "size", "order"):
                mesh.pop(k, None)
            mesh["mesher"] = value
            mesh.update(_default_mesh_params(value, dim))
    elif f.op == "bc_type":
        bc = get_at(model, key[:-1])
        if bc.get("type") != value:
            spec = M.PHYSICS_SPECS[model["physics"]["type"]]
            set_at(model, key[:-1], _default_bc(value, spec, bc.get("selection", M.BOUNDARY),
                                                physics_dim(model)))
    elif f.op == "physics_type":
        set_physics_type(model, value)
    else:
        raise ValueError(f"unknown operation {f.op!r}")
    return None


def _fit_dimension(model: dict, gi: int, dim: int) -> None:
    """After a change of dimension: adapt the meshes of the geometry, drop the
    selections whose criteria no longer fit, and the plane option if not 2D."""
    geom = model["geometries"][gi]
    for mesh in model.get("meshes", []):
        if mesh.get("geometry") == geom["name"] and mesh.get("mesher") == "structured":
            div = list(mesh.get("divisions", []))
            mesh["divisions"] = (div + [10] * dim)[:dim]
    for sel in geom.get("selections", {}).values():
        crit = sel.get("criteria", {})
        for k, v in list(crit.items()):
            if isinstance(v, list) and k != "of":
                crit[k] = (list(v) + [0.0] * dim)[:dim]
    phys = model.get("physics", {})
    if physics_geometry(model) is geom:
        if dim != 2:
            phys.pop("plane", None)
        spec = M.PHYSICS_SPECS.get(phys.get("type"), {})
        if spec.get("source") == M.VECTOR:
            for s in phys.get("sources", []):
                if isinstance(s.get("value"), list):
                    s["value"] = (s["value"] + [0.0] * dim)[:dim]
        for bc in phys.get("boundary_conditions", []):
            for k, kind in spec.get("boundary_conditions", {}).get(bc.get("type"), {}).items():
                if kind == M.VECTOR and isinstance(bc.get(k), list):
                    bc[k] = (bc[k] + [0.0] * dim)[:dim]


def set_physics_type(model: dict, ptype: str) -> None:
    """Change the physics: conditions and sources are dropped (they do not
    translate), materials keep their name and domain, and get the properties
    of the new physics (existing values are kept when the name matches)."""
    spec = M.PHYSICS_SPECS[ptype]
    phys = model.setdefault("physics", {})
    if phys.get("type") == ptype:
        return
    old_spec = M.PHYSICS_SPECS.get(phys.get("type"), {})
    model["physics"] = {"type": ptype, "mesh": phys.get("mesh"), "variable": spec["variable"],
                        "sources": [], "boundary_conditions": []}
    if model["physics"]["mesh"] is None:
        model["physics"].pop("mesh")
    for mat in model.get("materials", []):
        props = mat.get("properties", {})
        new = {p: props.get(p, spec["property_defaults"][p]) for p in spec["properties"]}
        for p, v in props.items():   # user properties (density, ...) are kept
            if p not in old_spec.get("properties", {}) and p not in new:
                new[p] = v
        mat["properties"] = new


# ------------------------------------------------------------------ actions

def actions(model: dict, path: Path) -> list[tuple[str, str]]:
    """Possible operations on a node: [(identifier, label)]."""
    n = len(path)
    out: list[tuple[str, str]] = []
    if path == ("parameters",):
        out.append(("add_parameter", "Add a parameter"))
    elif path[:1] == ("parameters",) and n == 2:
        out.append(("remove", "Remove"))
    elif path == ("fields",):
        out.append(("add_field", "Add a field"))
    elif path[:1] == ("fields",) and n == 2:
        out += [("add_field_value", "Add a value on a domain"), ("remove", "Remove")]
    elif path == ("geometries",):
        out.append(("add_geometry", "Add a geometry"))
    elif path[:1] == ("geometries",) and n == 2:
        out += [("add_selection", "Add a selection"), ("add_mesh", "Add a mesh of this geometry")]
        if len(model.get("geometries", [])) > 1:
            out.append(("remove", "Remove"))
    elif path[:1] == ("geometries",) and n > 3 and path[2] == "source":
        shape = get_at(model, path, {})
        if shape.get("type") in M.SHAPES_BOOLEAN:
            out += [(f"add_shape:{t}", f"Add a {t}") for t in PRIMITIVES
                    if t != "cylinder" or geometry_dim(model, path[1]) == 3]
        parent = get_at(model, path[:-2], {})
        if path[-2] == "of" and len(parent.get("of", [])) > 2:
            out.append(("remove", "Remove"))
    elif path[:1] == ("geometries",) and n == 3 and path[2] == "selections":
        out.append(("add_selection", "Add a selection"))
    elif path[:1] == ("geometries",) and n == 4:
        geom = model["geometries"][path[1]]
        if path[3] in geom.get("selections", {}):
            out.append(("remove", "Remove"))
    elif path == ("meshes",):
        out.append(("add_mesh", "Add a mesh"))
    elif path[:1] == ("meshes",) and n == 2:
        out.append(("use_mesh", "Use for the physics"))
        if len(model.get("meshes", [])) > 1:
            out.append(("remove", "Remove"))
    elif path == ("materials",):
        out.append(("add_material", "Add a material"))
    elif path[:1] == ("materials",) and n == 2:
        out.append(("add_property", "Add a property"))
        out.append(("remove", "Remove"))
    elif path == ("physics",) or path == ("physics", "boundary_conditions"):
        out.append(("add_bc", "Add a boundary condition"))
        if path == ("physics",):
            out.append(("add_source", "Add a source"))
    elif path == ("physics", "sources"):
        out.append(("add_source", "Add a source"))
    elif path[:2] in (("physics", "sources"), ("physics", "boundary_conditions")) and n == 3:
        out.append(("remove", "Remove"))
    elif path == ("outputs",):
        out.append(("add_quantity", "Add an integrated quantity"))
    elif path[:2] == ("outputs", "quantities") and n == 3:
        out.append(("remove", "Remove"))
    return out


def do_action(model: dict, path: Path, action: str, name: str | None = None) -> Path | None:
    """Perform an action; returns the path of the node to select afterwards."""
    dim = physics_dim(model)
    phys = model.setdefault("physics", {})
    spec = M.PHYSICS_SPECS.get(phys.get("type"), {})
    if action == "remove":
        parent = get_at(model, path[:-1])
        if isinstance(parent, list):
            del parent[path[-1]]
            if parent:
                return path[:-1] + (min(path[-1], len(parent) - 1),)
        else:
            del parent[path[-1]]
        return path[:-2] if path[-2] == "of" else path[:-1]
    if action == "add_parameter":
        params = model.setdefault("parameters", [])
        taken = {p.get("name") for p in params} | {f.get("name") for f in model.get("fields", [])}
        params.append({"name": unique_name(name or "p", taken), "value": 1.0, "description": ""})
        return ("parameters", len(params) - 1)
    if action == "add_field":
        fields_ = model.setdefault("fields", [])
        taken = {p.get("name") for p in model.get("parameters", [])} | {f.get("name") for f in fields_}
        fields_.append({"name": unique_name(name or "f", taken), "description": "",
                        "values": [{"selection": M.ALL, "value": 0.0}]})
        return ("fields", len(fields_) - 1)
    if action == "add_field_value":
        model["fields"][path[1]].setdefault("values", []).append({"selection": M.ALL, "value": 0.0})
        return path
    if action == "add_geometry":
        geoms = model.setdefault("geometries", [])
        gname = unique_name(name or "geometry", [g.get("name") for g in geoms])
        geoms.append({"name": gname, "source": _default_source("box", dim), "selections": {}})
        return ("geometries", len(geoms) - 1)
    if action == "add_selection":
        gi = path[1]
        geom = model["geometries"][gi]
        table = M.selection_table(model, geom)
        sname = unique_name(name or "selection", table)
        gdim = geometry_dim(model, gi)
        geom.setdefault("selections", {})[sname] = {
            "kind": M.BOUNDARY, "criteria": _default_criteria("plane", gdim, M.BOUNDARY)}
        return ("geometries", gi, "selections", sname)
    if action.startswith("add_shape:"):
        shape = get_at(model, path)
        shape.setdefault("of", []).append(_shape_in_geometry(model, path[1], action.split(":", 1)[1]))
        return path + ("of", len(shape["of"]) - 1)
    if action == "add_mesh":
        meshes = model.setdefault("meshes", [])
        geom = model["geometries"][path[1]] if path[:1] == ("geometries",) else \
            (physics_geometry(model) or model["geometries"][0])
        stype = geom.get("source", {}).get("type")
        mesher = next(x for x, srcs in M.MESHERS.items() if stype in srcs)
        gdim = M.geometry_dim(model, geom) or 2
        mesh = {"name": unique_name(name or "mesh", [m.get("name") for m in meshes]),
                "geometry": geom["name"], "mesher": mesher}
        mesh.update(_default_mesh_params(mesher, gdim))
        meshes.append(mesh)
        return ("meshes", len(meshes) - 1)
    if action == "use_mesh":
        phys["mesh"] = model["meshes"][path[1]]["name"]
        return path
    if action == "add_material":
        mats = model.setdefault("materials", [])
        mats.append({"name": unique_name(name or "material", [m.get("name") for m in mats]),
                     "selection": M.ALL, "properties": dict(spec.get("property_defaults", {}))})
        return ("materials", len(mats) - 1)
    if action == "add_property":
        if not name or not M._IDENT.match(name):
            raise ValueError("a property name (letters, digits, _) is required")
        props = model["materials"][path[1]].setdefault("properties", {})
        if name in props:
            raise ValueError(f"the property {name!r} already exists")
        props[name] = 1.0
        return path
    if action == "add_bc":
        bcs = phys.setdefault("boundary_conditions", [])
        boundaries = selections_of_kind(model, M.BOUNDARY)
        used = {bc.get("selection") for bc in bcs}
        sel = next((s for s in boundaries if s != M.BOUNDARY and s not in used),
                   boundaries[0] if boundaries else M.BOUNDARY)
        bcs.append(_default_bc(next(iter(spec["boundary_conditions"])), spec, sel, dim))
        return ("physics", "boundary_conditions", len(bcs) - 1)
    if action == "add_source":
        srcs = phys.setdefault("sources", [])
        srcs.append({"selection": M.ALL, "value": _zero(spec.get("source", M.SCALAR), dim)})
        return ("physics", "sources", len(srcs) - 1)
    if action == "add_quantity":
        qs = model.setdefault("outputs", {}).setdefault("quantities", [])
        qs.append({"name": unique_name(name or "quantity", [q.get("name") for q in qs]),
                   "expr": "1", "selection": M.ALL})
        return ("outputs", "quantities", len(qs) - 1)
    raise ValueError(f"unknown action {action!r}")


def remove_at(model: dict, path: Path) -> None:
    """Delete the item at path (dictionary key or list element)."""
    parent = get_at(model, path[:-1])
    if isinstance(parent, list):
        del parent[path[-1]]
    else:
        parent.pop(path[-1], None)


def needs_name(action: str) -> bool:
    """Actions for which the interface asks for a name first."""
    return action == "add_property"


def highlighted_selection(model: dict, path: Path) -> tuple[str | None, str | None]:
    """(mesh name, selection name) to highlight when a node is selected:
    the selection itself, the domain of a material or a source, the boundary
    of a condition, the support of an output quantity."""
    mesh = M.physics_mesh(model) or {}
    if path[:1] == ("geometries",) and len(path) == 4 and path[2] == "selections":
        geom = model["geometries"][path[1]]
        meshes = [m for m in model.get("meshes", []) if m.get("geometry") == geom.get("name")]
        target = mesh if mesh in meshes else (meshes[0] if meshes else {})
        return target.get("name"), path[3]
    if path[:1] == ("geometries",):
        return None, None
    item = get_at(model, path)
    if isinstance(item, dict) and "selection" in item:
        return mesh.get("name"), item.get("selection", M.ALL)
    return None, None


def mesh_to_show(model: dict, path: Path) -> str | None:
    """Name of the mesh the viewer should show for a node."""
    if path[:1] == ("meshes",) and len(path) == 2:
        return get_at(model, path + ("name",))
    if path[:1] == ("geometries",) and len(path) >= 2:
        geom = model["geometries"][path[1]]
        mesh = M.physics_mesh(model) or {}
        if mesh.get("geometry") == geom.get("name"):
            return mesh.get("name")
        other = [m for m in model.get("meshes", []) if m.get("geometry") == geom.get("name")]
        return other[0]["name"] if other else None
    m, _ = highlighted_selection(model, path)
    return m or (M.physics_mesh(model) or {}).get("name")


# ------------------------------------------------------------------ picking edges for an item

def selection_uses(model: dict, name: str) -> int:
    """How many items (materials, sources, conditions, outputs, unions) use a
    selection of the geometry of the physics."""
    phys = model.get("physics", {})
    items = model.get("materials", []) + phys.get("sources", []) + phys.get("boundary_conditions", []) \
        + model.get("outputs", {}).get("quantities", []) \
        + [v for f in model.get("fields", []) for v in f.get("values", [])]
    n = sum(1 for it in items if it.get("selection", M.ALL) == name)
    geom = physics_geometry(model) or {}
    for sel in geom.get("selections", {}).values():
        crit = sel.get("criteria", {})
        if crit.get("type") == "union" and name in crit.get("of", []):
            n += 1
    return n


def can_pick_edges(model: dict, path: Path) -> bool:
    """Whether clicks on edges edit the support of the item at path (a boundary
    condition or an output integrated on a boundary)."""
    from . import geometry2d as G
    if not (path[:2] in (("physics", "boundary_conditions"), ("outputs", "quantities")) and len(path) == 3):
        return False
    geom = physics_geometry(model)
    if geom is None or G.shape_of(P.resolved(model)["geometries"][model["geometries"].index(geom)]
                                  .get("source", {})) is None:
        return False
    sel = get_at(model, path + ("selection",), M.ALL)
    return M.selection_table(model, geom).get(sel, {}).get("kind") == M.BOUNDARY


def pick_edge_for_item(model: dict, path: Path, ref: dict) -> tuple[str, bool]:
    """Add or remove an edge (side reference) in the support of the item at
    path. Its selection is edited in place when it is an 'edges' selection
    used by this item only; otherwise a new selection is created, starting
    from the current support (empty if it is the whole boundary), named
    after the item. Returns (selection name,
    created)."""
    from . import geometry2d as G
    item = get_at(model, path)
    geom = physics_geometry(model)
    gi = model["geometries"].index(geom)
    resolved = P.resolved(model)
    rgeom = resolved["geometries"][gi]
    table = M.selection_table(resolved, rgeom)
    name = item.get("selection", M.ALL)
    sel = geom.get("selections", {}).get(name)
    if sel is not None and sel.get("criteria", {}).get("type") == "edges" \
            and selection_uses(model, name) == 1:
        sel["criteria"] = G.toggle_edge(sel["criteria"], ref)
        return name, False
    shape = G.shape_of(rgeom["source"])
    start: list[dict] = []
    # from the whole boundary (the default of a new condition), a click means
    # "this edge", not "everything but this edge"
    for p in (G.pieces_matching(shape, name, table) or []) if name != M.BOUNDARY else []:
        if p.side.ref not in start:
            start.append(p.side.ref)
    criteria = G.toggle_edge({"type": "edges", "edges": start}, ref)
    if path[0] == "physics":
        base = f"{item.get('type', 'condition')}_{path[2] + 1}"
    else:
        base = f"{item.get('name', 'output')}_support"
    new = unique_name(base, table)
    geom.setdefault("selections", {})[new] = {"kind": M.BOUNDARY, "criteria": criteria}
    item["selection"] = new
    return new, True


# ------------------------------------------------------------------ picking regions for an item

def can_pick_regions(model: dict, path: Path) -> bool:
    """Whether clicks in the domain edit the support of the item at path (a
    material, a source or an output on a domain), or the selection at path
    (a domain selection of a 2D box or csg geometry)."""
    from . import geometry2d as G
    if path[:1] == ("geometries",) and len(path) == 4 and path[2] == "selections":
        geom = model["geometries"][path[1]]
        sel = geom.get("selections", {}).get(path[3], {})
        gi = path[1]
    elif (path[:1] == ("materials",) and len(path) == 2) or \
            (path[:2] in (("physics", "sources"), ("outputs", "quantities")) and len(path) == 3):
        geom = physics_geometry(model)
        if geom is None:
            return False
        gi = model["geometries"].index(geom)
        name = get_at(model, path + ("selection",), M.ALL)
        sel = M.selection_table(model, geom).get(name, {})
    else:
        return False
    if G.shape_of(P.resolved(model)["geometries"][gi].get("source", {})) is None:
        return False
    return sel.get("kind") == M.DOMAIN


def pick_region_for_item(model: dict, path: Path, region: list) -> tuple[str, bool]:
    """Add or remove a region (see geometry2d.region_at) in the selection at
    path, or in the support of the item at path. The selection of an item is
    edited in place when it is a 'regions' selection used by this item only;
    otherwise a new selection is created, starting from the current support
    when it is made of regions (empty if it is the whole domain), named after
    the item. Returns (selection name, created)."""
    from . import geometry2d as G
    if path[:1] == ("geometries",):
        sel = model["geometries"][path[1]]["selections"][path[3]]
        sel["criteria"] = G.toggle_region(sel.get("criteria", {}), region)
        return path[3], False
    item = get_at(model, path)
    geom = physics_geometry(model)
    name = item.get("selection", M.ALL)
    sel = geom.get("selections", {}).get(name)
    if sel is not None and sel.get("criteria", {}).get("type") == "regions" \
            and selection_uses(model, name) == 1:
        sel["criteria"] = G.toggle_region(sel["criteria"], region)
        return name, False
    start = copy.deepcopy(sel["criteria"]) if sel is not None and \
        sel.get("criteria", {}).get("type") == "regions" else {"type": "regions", "regions": []}
    criteria = G.toggle_region(start, region)
    if path[0] == "materials":
        base = item.get("name", "material")
    elif path[0] == "physics":
        base = f"source_{path[2] + 1}"
    else:
        base = f"{item.get('name', 'output')}_support"
    new = unique_name(base, M.selection_table(model, geom))
    geom.setdefault("selections", {})[new] = {"kind": M.DOMAIN, "criteria": criteria}
    item["selection"] = new
    return new, True


# ------------------------------------------------------------------ drawing shapes

def can_draw(model: dict, path: Path) -> int | None:
    """Index of the 2D geometry (box or csg) that a drawn shape goes to, when
    the node at path is that geometry or one of its shapes; None otherwise."""
    if path[:1] != ("geometries",) or len(path) < 2 or (len(path) > 2 and path[2] != "source"):
        return None
    gi = path[1]
    src = get_at(model, ("geometries", gi, "source"), {})
    if src.get("type") == "box" and len(src.get("size", [])) == 2:
        return gi
    if src.get("type") == "csg" and src.get("dim") == 2:
        return gi
    return None


def _prefix_edge_refs(geom: dict, prefix: tuple) -> None:
    """The CSG root moved under a boolean at `prefix`: update the edges refs
    and the regions (paths of primitives)."""
    for sel in geom.get("selections", {}).values():
        crit = sel.get("criteria", {})
        if crit.get("type") == "edges":
            for ref in crit.get("edges", []):
                ref["shape"] = list(prefix) + list(ref.get("shape", []))
        elif crit.get("type") == "regions":
            crit["regions"] = [[list(prefix) + list(p) for p in reg] for reg in crit.get("regions", [])]


def _box_to_csg(model: dict, gi: int) -> list[str]:
    """Turn a 2D box into a CSG rectangle; its built-in sides (xmin...) become
    selections of the same name, so that their uses stay valid; structured
    meshes of the geometry switch to the GetFEM mesher. Returns notes."""
    geom = model["geometries"][gi]
    src = geom["source"]
    o, s = src["origin"], src["size"]
    shape = {"type": "rectangle", "min": list(o),
             "max": [f"{format_value(a)} + {format_value(b)}" if isinstance(a, str) or isinstance(b, str)
                     else a + b for a, b in zip(o, s)]}
    geom["source"] = {"type": "csg", "dim": 2, "shape": shape}
    sides = {"xmin": "xmin", "xmax": "xmax", "ymin": "ymin", "ymax": "ymax"}
    old = geom.get("selections", {})
    new = {name: {"kind": M.BOUNDARY, "criteria": {"type": "edges", "edges": [{"shape": [], "side": side}]}}
           for name, side in sides.items() if name not in old}
    new.update(old)          # edges refs of a box already designate the root rectangle
    geom["selections"] = new
    notes = ["the box became a constructive geometry; its sides xmin... are now selections"]
    for mesh in model.get("meshes", []):
        if mesh.get("geometry") == geom.get("name") and mesh.get("mesher") == "structured":
            vals, _ = P.values(model)
            try:
                sizes = [P.evaluate(x, vals) for x in s]
                h = min(si / max(int(P.evaluate(d, vals)), 1) for si, d in zip(sizes, mesh["divisions"]))
            except (P.ParameterError, KeyError, TypeError):
                h = 0.1
            for k in ("divisions", "element"):
                mesh.pop(k, None)
            mesh.update({"mesher": "getfem", "h": h, "order": 1})
            notes.append(f"mesh {mesh.get('name')!r}: structured mesher replaced by the GetFEM mesher (h = {h:g})")
    return notes


def add_shape_2d(model: dict, gi: int, shape: dict, op: str) -> tuple[Path, list[str]]:
    """Combine a drawn primitive with a 2D geometry: op is union, difference
    (the shape is removed) or intersection. Returns the path of the new shape
    node and notes about what was changed."""
    if op not in M.SHAPES_BOOLEAN:
        raise ValueError(f"unknown operation {op!r}")
    notes: list[str] = []
    geom = model["geometries"][gi]
    if geom["source"].get("type") == "box":
        notes += _box_to_csg(model, gi)
    src = geom["source"]
    root = src["shape"]
    base = ("geometries", gi, "source", "shape")
    if root.get("type") == op:            # a \ b \ c, a u b u c...: one more operand
        root.setdefault("of", []).append(shape)
        return base + ("of", len(root["of"]) - 1), notes
    src["shape"] = {"type": op, "of": [root, shape]}
    _prefix_edge_refs(geom, (0,))
    return base + ("of", 1), notes


# ------------------------------------------------------------------ handles of 2D primitives

class LockedValue(ValueError):
    """A value given by an expression of parameters cannot be dragged."""


def primitive_node(model: dict, path: Path) -> dict | None:
    """The 2D primitive (rectangle or disk) of a shape node, if path is one."""
    if can_draw(model, path) is None or len(path) < 4 or path[2:4] != ("source", "shape"):
        return None
    shape = get_at(model, path)
    return shape if isinstance(shape, dict) and shape.get("type") in ("rectangle", "ball") else None


def shape_handles(model: dict, path: Path) -> list[tuple[str, list[float]]]:
    """Handles of the primitive at path: [(name, position)], in model values."""
    shape = primitive_node(model, path)
    if shape is None:
        return []
    r = P.resolved(model)
    s = get_at(r, path)
    if s is None or any(isinstance(v, str) for k in ("min", "max", "center") for v in s.get(k, [])) \
            or isinstance(s.get("radius"), str):
        return []                          # invalid expressions: validation reports them
    if s["type"] == "rectangle":
        (x0, y0), (x1, y1) = s["min"], s["max"]
        return [("min", [x0, y0]), ("xmax_ymin", [x1, y0]), ("max", [x1, y1]), ("xmin_ymax", [x0, y1]),
                ("center", [(x0 + x1) / 2, (y0 + y1) / 2])]
    (cx, cy), rad = s["center"], s["radius"]
    return [("center", [cx, cy]), ("radius", [cx + rad, cy])]


def _set_number(model: dict, key: Path, value: float, notes: list[str]) -> None:
    """Set a numerical value: a number is replaced; a bare parameter name sets
    that parameter (when it is a number itself); an expression is locked."""
    current = get_at(model, key)
    if current is None or isinstance(current, (int, float)) and not isinstance(current, bool):
        set_at(model, key, float(value))
        return
    name = current.strip() if isinstance(current, str) else None
    for p in model.get("parameters", []):
        if p.get("name") == name and isinstance(p.get("value"), (int, float)):
            p["value"] = float(value)
            notes.append(f"parameter {name} = {format_number(float(value))}")
            return
    raise LockedValue(f"{current!r} is an expression of parameters: change it in the properties")


def move_handle(model: dict, path: Path, handle: str, point) -> list[str]:
    """Move a handle of the primitive at path to point. Returns notes (e.g.
    the parameters changed); raises LockedValue, leaving the model unchanged,
    when a value to change is an expression."""
    shape = primitive_node(model, path)
    if shape is None:
        raise ValueError("not a rectangle or a disk")
    before = copy.deepcopy(model)
    notes: list[str] = []
    x, y = float(point[0]), float(point[1])
    r = get_at(P.resolved(model), path)
    try:
        if shape["type"] == "rectangle":
            (x0, y0), (x1, y1) = r["min"], r["max"]
            if handle == "center":
                dx, dy = x - (x0 + x1) / 2, y - (y0 + y1) / 2
                new = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
            else:
                nx0, ny0, nx1, ny1 = x0, y0, x1, y1
                if handle in ("min", "xmin_ymax"):
                    nx0 = x
                else:
                    nx1 = x
                if handle in ("min", "xmax_ymin"):
                    ny0 = y
                else:
                    ny1 = y
                new = (min(nx0, nx1), min(ny0, ny1), max(nx0, nx1), max(ny0, ny1))
                if new[2] - new[0] <= 0 or new[3] - new[1] <= 0:
                    raise ValueError("the rectangle would be empty")
            for k, i, v, old in (("min", 0, new[0], x0), ("min", 1, new[1], y0),
                                 ("max", 0, new[2], x1), ("max", 1, new[3], y1)):
                if abs(v - old) > 1e-15 * max(1.0, abs(old)):
                    _set_number(model, path + (k, i), v, notes)
        elif handle == "center":
            for i, v in enumerate((x, y)):
                if abs(v - r["center"][i]) > 1e-15 * max(1.0, abs(v)):
                    _set_number(model, path + ("center", i), v, notes)
        else:
            rad = math.hypot(x - r["center"][0], y - r["center"][1])
            if rad <= 0:
                raise ValueError("the radius would be zero")
            _set_number(model, path + ("radius",), rad, notes)
    except ValueError:
        model.clear()
        model.update(before)
        raise
    return notes
