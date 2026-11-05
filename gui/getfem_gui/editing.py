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


"""Editing a model file from a user interface, independently of any toolkit.

- `tree(model)`: the tree of the file, as shown in the model browser;
- `fields(model, path)`: the form fields of a node;
- `apply(model, field, value)`: change a field, keeping the file consistent
  (renaming a selection or a variable updates its uses, changing the type of
  a condition resets its values, ...);
- `actions(model, path)` / `do_action(...)`: add and remove items;
- `errors_by_node` / `errors_at`: place the messages of `model.validate`.

A node or a field is designated by its path in the model dictionary, e.g.
("models", 0, "modules", 0, "conditions", 1) or ("materials", 0, "properties", "E").
Functions modify the model in place.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field

from . import model as M, parameters as P

Path = tuple

PRIMITIVES_2D = ("rectangle", "ball", "ellipse", "polygon")
PRIMITIVES = ("rectangle", "ball", "cylinder") + M.SHAPES_SWEPT


def primitives_of_dim(dim: int) -> tuple:
    """The primitives of a shape of dimension dim (2: a 2D geometry or a profile)."""
    return PRIMITIVES_2D if dim == 2 else PRIMITIVES
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
                                  # choice, names, file, info, parameter_table (a whole list),
                                  # material_table, vertex_table (the points of a polygon)
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


def owner(model: dict, path: Path) -> dict | None:
    """The model (gf.Model) a node belongs to, if any."""
    if path[:1] == ("models",) and len(path) >= 2:
        return get_at(model, path[:2])
    return None


def item_geometry(model: dict, path: Path) -> dict | None:
    """The geometry whose selections an item uses: its model's one, a field's
    one, or the geometry itself."""
    md = owner(model, path)
    if md is not None:
        return M.model_geometry(model, md)
    if path[:1] == ("fields",) and len(path) >= 2:
        f = get_at(model, path[:2], {})
        geoms = model.get("geometries", [])
        name = f.get("geometry", geoms[0].get("name") if len(geoms) == 1 else None)
        return M.by_name(geoms, name)
    if path[:1] == ("geometries",) and len(path) >= 2:
        return get_at(model, path[:2])
    return None


def geometry_dim(model: dict, gi: int) -> int:
    geoms = model.get("geometries", [])
    return (M.geometry_dim(model, geoms[gi]) if gi < len(geoms) else None) or 2


def shape_dim(model: dict, path: Path) -> int:
    """Dimension of the shape at path (a node under geometries/i/source/shape):
    2 in the profile of an extrusion or a revolution."""
    return 2 if "profile" in path[4:] else geometry_dim(model, path[1])


def item_dim(model: dict, path: Path) -> int:
    geom = item_geometry(model, path)
    return (M.geometry_dim(model, geom) if geom else None) or 2


def main_model(model: dict) -> dict | None:
    """The model shown by default (the first one)."""
    models = model.get("models", [])
    return models[0] if models else None


def selections_of_kind(model: dict, kind: str | None, geom: dict | None) -> list[str]:
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


def _choices(names) -> tuple:
    return tuple((n, n) for n in names)


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
    elif shape.get("type") in M.SHAPES_SWEPT and isinstance(shape.get("profile"), dict):
        child = _shape_node(shape["profile"], path + ("profile",))
        child.label = f"Profile: {child.label}"
        node.children.append(child)
    return node


def _cond_label(c: dict) -> str:
    return f"{str(c.get('type', '?')).replace('_', ' ').capitalize()} — {c.get('selection', '?')}"


def _params_node(path: Path, params: list, vals: dict) -> Node:
    """The parameters are edited together, in a table (no node each)."""
    return Node(path, f"Parameters ({len(params)})" if params else "Parameters", "parameters")


def _fields_node(path: Path, fields_: list) -> Node:
    node = Node(path, "Fields", "fields")
    for i, f in enumerate(fields_):
        node.children.append(Node(path + (i,), f.get("name", "?"), "field"))
    return node


def _module_node(md_path: Path, i: int, mod: dict) -> Node:
    spec = M.MODULE_SPECS.get(mod.get("type"), {})
    path = md_path + ("modules", i)
    node = Node(path, spec.get("label", str(mod.get("type"))), "module")
    for p in spec.get("properties", {}):
        for j, pc in enumerate(mod.get("properties", {}).get(p, [])):
            what = pc["material"] if "material" in pc else format_value(pc.get("value"))
            node.children.append(Node(path + ("properties", p, j),
                                      f"{p} = {what} — {pc.get('selection', M.ALL)}", "property"))
    if spec.get("source") is not None:
        srcs = Node(path + ("sources",), "Sources", "sources")
        for j, s in enumerate(mod.get("sources", [])):
            srcs.children.append(Node(path + ("sources", j), f"Source — {s.get('selection', M.ALL)}", "source"))
        node.children.append(srcs)
    if spec.get("conditions"):
        bcs = Node(path + ("conditions",), "Conditions", "conditions")
        for j, c in enumerate(mod.get("conditions", [])):
            bcs.children.append(Node(path + ("conditions", j), _cond_label(c), "condition"))
        node.children.append(bcs)
    return node


def _model_node(model: dict, k: int, md: dict) -> Node:
    path = ("models", k)
    kinds = ", ".join(M.MODULE_SPECS.get(m.get("type"), {}).get("label", "?").split(" (")[0]
                      for m in md.get("modules", []))
    node = Node(path, f"{md.get('name', '?')}" + (f" ({kinds})" if kinds else ""), "model")
    vals, _ = M.parameter_values(model, md)
    if md.get("parameters"):
        node.children.append(_params_node(path + ("parameters",), md["parameters"], vals))
    if md.get("fields"):
        node.children.append(_fields_node(path + ("fields",), md["fields"]))
    if md.get("imports"):
        imps = Node(path + ("imports",), "Imports", "imports")
        for i, im in enumerate(md["imports"]):
            imps.children.append(Node(path + ("imports", i),
                                      f"{im.get('name')} ← {im.get('model')}.{im.get('variable')}", "import"))
        node.children.append(imps)
    vars_ = Node(path + ("variables",), "Variables", "variables")
    for i, v in enumerate(md.get("variables", [])):
        fem = "P" if v.get("fem", "lagrange") == "lagrange" else "discontinuous P"
        comp = "" if v.get("components", 1) == 1 else " (vector)"
        vars_.children.append(Node(path + ("variables", i),
                                   f"{v.get('name')}{comp}: {fem}{v.get('degree', 1)} on {v.get('mesh')}",
                                   "variable"))
    node.children.append(vars_)
    mods = Node(path + ("modules",), "Modules", "modules")
    for i, mod in enumerate(md.get("modules", [])):
        mods.children.append(_module_node(path, i, mod))
    node.children.append(mods)
    terms = Node(path + ("terms",), "Terms", "terms")
    for i, t in enumerate(md.get("terms", [])):
        terms.children.append(Node(path + ("terms", i), f"{t.get('name')} — {t.get('selection', M.ALL)}", "term"))
    node.children.append(terms)
    conds = Node(path + ("conditions",), "Conditions", "conditions")
    for i, c in enumerate(md.get("conditions", [])):
        conds.children.append(Node(path + ("conditions", i), f"{_cond_label(c)} ({c.get('variable')})",
                                   "condition"))
    node.children.append(conds)
    s = md.get("solver", {})
    node.children.append(Node(path + ("solver",), f"Solver ({s.get('type', 'auto')})", "solver"))
    outs = Node(path + ("outputs",), "Outputs", "outputs")
    for i, q in enumerate(md.get("outputs", [])):
        outs.children.append(Node(path + ("outputs", i), q.get("name", "?"), "quantity"))
    node.children.append(outs)
    graphics = Node(path + ("graphics",), "Graphics", "graphics")
    for i, g in enumerate(md.get("graphics", [])):
        label = g.get("name", "?") + ("" if g.get("name") == g.get("field") else f" ({g.get('field')})")
        graphics.children.append(Node(path + ("graphics", i), label, "graphic"))
    node.children.append(graphics)
    return node


def tree(model: dict, result_fields: dict | None = None, has_results: bool = False) -> Node:
    """result_fields: {model name: [field names]} of the results, if any."""
    root = Node((), model.get("name", "Model"), "file")
    vals, _ = P.values(model)
    root.children.append(_params_node(("parameters",), model.get("parameters", []), vals))

    mats = Node(("materials",), "Materials", "materials")
    for i, mat in enumerate(model.get("materials", [])):
        props = ", ".join(mat.get("properties", {}))
        mats.children.append(Node(("materials", i), f"{mat.get('name', '?')}" + (f" ({props})" if props else ""),
                                  "material"))
    root.children.append(mats)

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
    for i, m in enumerate(model.get("meshes", [])):
        users = [md.get("name") for md in model.get("models", [])
                 if any(v.get("mesh") == m.get("name") for v in md.get("variables", []))]
        label = f"{m.get('name', '?')} ({m.get('mesher', '?')})" + (f" — {', '.join(users)}" if users else "")
        meshes.children.append(Node(("meshes", i), label, "mesh"))
    root.children.append(meshes)

    root.children.append(_fields_node(("fields",), model.get("fields", [])))

    models = Node(("models",), "Models", "models")
    for k, md in enumerate(model.get("models", [])):
        models.children.append(_model_node(model, k, md))
    root.children.append(models)

    solvers = Node(("solvers",), "Solvers", "solvers")
    for i, s in enumerate(model.get("solvers", [])):
        solvers.children.append(Node(("solvers", i), f"{s.get('name')} ({s.get('type')}: "
                                     f"{', '.join(s.get('steps', []))})", "global_solver"))
    root.children.append(solvers)

    if has_results or result_fields:
        res = Node((RESULTS,), "Results", "results")
        for mname, names in (result_fields or {}).items():
            mn = Node((RESULTS, mname), mname, "result_model")
            for name in names:
                mn.children.append(Node((RESULTS, mname, name), name, "result_field"))
            res.children.append(mn)
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
    types = primitives_of_dim(dim) + M.SHAPES_BOOLEAN
    out = [Field(path + ("type",), "Type", "choice", tuple((x, x) for x in types), op="shape_type")]
    if path[-1] == "profile":
        out.append(Field(path, "", "info", help="2D profile: (x, y) for an extrusion, (r, z) for a "
                                                "revolution (the part with r < 0 is ignored)."))
    if t == "rectangle":
        out += [_vec_field(path + ("min",), "Lower corner", "numbers", dim),
                _vec_field(path + ("max",), "Upper corner", "numbers", dim)]
    elif t == "ball":
        out += [_vec_field(path + ("center",), "Center", "numbers", dim),
                Field(path + ("radius",), "Radius", "number")]
    elif t == "ellipse":
        out += [_vec_field(path + ("center",), "Center", "numbers", 2),
                _vec_field(path + ("radii",), "Semi-axes", "numbers", 2),
                Field(path + ("angle",), "Angle (degrees)", "number",
                      help="Angle of the first semi-axis with the x axis")]
    elif t == "polygon":
        out.append(Field(path + ("points",), "Vertices", "vertex_table",
                         help="Vertices in order (either direction); values or expressions. In the "
                              "view, drag a vertex, the middle of a side (adds a vertex) or the center."))
    elif t == "cylinder":
        out += [_vec_field(path + ("origin",), "Origin", "numbers", dim),
                _vec_field(path + ("axis",), "Axis", "numbers", dim),
                Field(path + ("length",), "Length", "number"),
                Field(path + ("radius",), "Radius", "number")]
    elif t == "extrusion":
        out += [_vec_field(path + ("z",), "From z to z", "numbers", 2),
                Field(path, "", "info", help="The profile (2D shape below) is extruded along z.")]
    elif t == "revolution":
        out.append(Field(path, "", "info", help="The profile (2D shape below, in the (r, z) plane) "
                                                "turns about the z axis."))
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
    from . import geometry3d as G3
    solid = G3.shape_of(P.resolved(model)["geometries"][gi].get("source", {})) is not None
    possible = [t for t, kinds in M.CRITERIA.items() if kind in kinds
                and (t not in ("edges", "regions") or planar) and (t != "faces" or solid)]
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
    elif ct == "faces":
        shape = G3.shape_of(P.resolved(model)["geometries"][gi].get("source", {}))
        labels = []
        for r in crit.get("faces", []):
            try:
                labels.append(G3.face_label(shape, r))
            except (KeyError, TypeError, ValueError, AttributeError):
                labels.append("?")
        out.append(Field(c + ("faces",), "", "info",
                         help=("Faces: " + ", ".join(labels) if labels else "No face yet.")
                         + "\nClick faces in the view to add or remove them."))
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


def _param_fields(model, path) -> list[Field]:
    md = owner(model, path)
    vals, _ = M.parameter_values(model, md)
    name = get_at(model, path + ("name",))
    out = [Field(path + ("name",), "Name", "ident", op="rename_parameter"),
           Field(path + ("value",), "Value", "value",
                 help="number or expression: + - * /, pi, " + ", ".join(sorted(P.FUNCTIONS))),
           Field(path + ("description",), "Description", "text")]
    if isinstance(get_at(model, path + ("value",)), str) and name in vals:
        out.append(Field(path, "", "info", help=f"= {format_number(vals[name])}"))
    return out


def material_rows(model: dict) -> tuple[list[str], list[dict]]:
    """The table of materials: the properties asked by the modules (columns),
    and per material its name, origin and a field per cell."""
    from . import library
    props = library.used_properties(model)
    rows = []
    for i, mat in enumerate(model.get("materials", [])):
        values = mat.get("properties", {})
        rows.append({"name": mat.get("name"), "from": mat.get("from", ""),
                     "values": [values.get(p) for p in props],
                     "fields": [Field(("materials", i, "properties", p), p, "value") for p in props]})
    return props, rows


def parameter_rows(model: dict, path: Path) -> list[dict]:
    """The rows of a table of parameters (path: ("parameters",) or
    ("models", k, "parameters")): name, value as written, its number (None
    if it cannot be evaluated), description, and the keys of the cells."""
    md = owner(model, path)
    vals, _ = M.parameter_values(model, md)
    rows = []
    for i, p in enumerate(get_at(model, path, [])):
        base = path + (i,)
        name = p.get("name")
        rows.append({"name": name, "value": p.get("value"), "number": vals.get(name),
                     "description": p.get("description", ""),
                     "fields": (Field(base + ("name",), "Name", "ident", op="rename_parameter"),
                                Field(base + ("value",), "Value", "value"),
                                Field(base + ("description",), "Description", "text"))})
    return rows


def _field_fields(model, path) -> list[Field]:
    out = [Field(path + ("name",), "Name", "ident", op="rename_field"),
           Field(path + ("description",), "Description", "text")]
    if path[:1] == ("fields",):
        out.append(Field(path + ("geometry",), "Geometry", "choice",
                         _choices(g.get("name") for g in model.get("geometries", []))))
    domains = _choices(selections_of_kind(model, M.DOMAIN, item_geometry(model, path)))
    for j, _ in enumerate(get_at(model, path + ("values",), [])):
        out += [Field(path + ("values", j, "selection"), f"Domain {j + 1}", "choice", domains),
                Field(path + ("values", j, "value"), f"Value {j + 1}", "value",
                      remove=path + ("values", j))]
    return out


def _value_field(key, label, kind, dim) -> Field:
    if kind == M.VECTOR:
        return Field(key, label, "vector", size=dim)
    return Field(key, label, "value")


_LABELS = {"value": "Value", "h": "Exchange coefficient h", "T_ext": "External temperature",
           "T_ref": "Reference temperature"}


def _model_fields(model, path) -> list[Field]:
    """Fields of the nodes inside a model (path starts with ("models", k))."""
    k, rest = path[1], path[2:]
    md = model["models"][k]
    base = ("models", k)
    n = len(rest)
    geom = M.model_geometry(model, md)
    dim = item_dim(model, path)
    domains = _choices(selections_of_kind(model, M.DOMAIN, geom))
    boundaries = _choices(selections_of_kind(model, M.BOUNDARY, geom))
    anywhere = _choices(selections_of_kind(model, None, geom))
    if n == 0:
        return [Field(base + ("name",), "Name", "ident", op="rename_model"),
                Field(base + ("description",), "Description", "text"),
                Field(base + ("vtu",), "VTU file", "text", help="file of the exported fields")]
    head = rest[0]
    if head == "parameters":
        if n == 1:
            return [Field(path, "", "info", help="Parameters of this model only (a parameter may "
                          "use the global ones and the ones listed before it)."),
                    Field(path, "", "parameter_table")]
        return _param_fields(model, path)
    if head == "fields":
        if n == 1:
            return [Field(path, "", "info", help="Fields of this model only.")]
        return _field_fields(model, path)
    if head == "imports":
        if n == 1:
            return [Field(path, "", "info", help="Variables of other models, used here as data "
                          "(the other model is solved first).")]
        im = get_at(model, path, {})
        src = M.model_by_name(model, im.get("model")) or {}
        return [Field(path + ("name",), "Name here", "ident"),
                Field(path + ("model",), "From the model", "choice",
                      _choices(m.get("name") for m in model.get("models", []) if m is not md)),
                Field(path + ("variable",), "Variable", "choice",
                      _choices(v.get("name") for v in src.get("variables", [])))]
    if head == "variables":
        if n == 1:
            return [Field(path, "", "info", help="The unknowns, with their finite element method.")]
        v = get_at(model, path, {})
        lo = 0 if v.get("fem") == "lagrange_discontinuous" else 1
        meshes = [m.get("name") for m in model.get("meshes", [])] + [m.get("name") for m in md.get("meshes", [])]
        return [Field(path + ("name",), "Name", "ident", op="rename_variable"),
                Field(path + ("mesh",), "Mesh", "choice", _choices(meshes)),
                Field(path + ("fem",), "Finite element", "choice",
                      (("lagrange", "Lagrange (continuous)"), ("lagrange_discontinuous", "discontinuous Lagrange"))),
                Field(path + ("degree",), "Degree", "choice", tuple((d, str(d)) for d in range(lo, 7))),
                Field(path + ("components",), "Components", "choice", ((1, "scalar"), ("dim", "vector")))]
    if head == "modules":
        if n == 1:
            return [Field(path, "", "info", help="Predefined physics, with their properties per domain, "
                          "sources and conditions.")]
        mod = model["models"][k]["modules"][rest[1]]
        spec = M.MODULE_SPECS.get(mod.get("type"), {})
        mpath = base + ("modules", rest[1])
        if n == 2:
            out = []
            kinds = {v.get("name"): M.variable_kind(v) for v in md.get("variables", [])}
            for role, v in spec.get("variables", {}).items():
                names = [x for x, kd in kinds.items() if kd == v["kind"]]
                if v.get("import"):
                    names += [im.get("name") for im in md.get("imports", [])]
                out.append(Field(mpath + (role,), v["label"].capitalize(), "choice", _choices(names)))
            for opt, values in spec.get("options", {}).items():
                if opt == "plane" and dim != 2:
                    continue
                out.append(Field(mpath + (opt,), "2D model" if opt == "plane" else opt, "choice",
                                 tuple((x, f"plane {x}" if opt == "plane" else x) for x in values)))
            for key, kind in spec.get("values", {}).items():
                out.append(_value_field(mpath + (key,), _LABELS.get(key, key), kind, dim))
            props = ", ".join(f"{p} ({label})" for p, label in spec.get("properties", {}).items())
            out.append(Field(mpath, "", "info", help=f"Properties: {props}, each given per domain "
                             "(the last one listed wins)."))
            return out
        if rest[2] == "properties" and n == 5:
            p, j = rest[3], rest[4]
            pc = get_at(model, path, {})
            mats = [m.get("name") for m in model.get("materials", []) if p in (m.get("properties") or {})]
            out = [Field(path + ("selection",), "Domain", "choice", domains),
                   Field(path, "Given by", "choice",
                         (("value", "a value"),) + tuple((f"material:{x}", f"material {x}") for x in mats),
                         op="piece_source")]
            if "material" not in pc:
                out.append(Field(path + ("value",), f"{p} ({spec['properties'].get(p, p)})", "value"))
            return out
        if rest[2] == "sources":
            if n == 3:
                return [Field(path, "", "info", help="Volume sources (right-hand side).")]
            kind = spec.get("source", M.SCALAR)
            return [Field(path + ("selection",), "Domain", "choice", domains),
                    _value_field(path + ("value",), "Value", kind, dim)]
        if rest[2] == "conditions":
            if n == 3:
                return [Field(path, "", "info", help="Boundaries without condition are free "
                              "(zero flux, zero traction).")]
            bc = get_at(model, path, {})
            specs = spec.get("conditions", {})
            out = [Field(path + ("type",), "Type", "choice",
                         tuple((t, t.replace("_", " ")) for t in specs), op="mcond_type"),
                   Field(path + ("selection",), "Boundary", "choice", boundaries)]
            for key, kind in specs.get(bc.get("type"), {}).items():
                out.append(_value_field(path + (key,), _LABELS.get(key, key), kind, dim))
            return out
        return []
    if head == "terms":
        if n == 1:
            return [Field(path, "", "info", help="Terms of the weak formulation in GWFL; the terms "
                          "of a model sum to zero (a source is written -f*Test_u).")]
        return [Field(path + ("name",), "Name", "ident"),
                Field(path + ("expr",), "Expression (GWFL)", "value"),
                Field(path + ("selection",), "Integrated on", "choice", anywhere),
                Field(path + ("linear",), "Linear", "choice",
                      (("auto", "detected"), (True, "yes"), (False, "no")))]
    if head == "conditions":
        if n == 1:
            return [Field(path, "", "info", help="Generic conditions on a variable (the modules "
                          "have their own ones).")]
        c = get_at(model, path, {})
        kinds = {v.get("name"): M.variable_kind(v) for v in md.get("variables", [])}
        vkind = kinds.get(c.get("variable"), M.SCALAR)
        out = [Field(path + ("type",), "Type", "choice",
                     tuple((t, t.replace("_", " ")) for t in M.CONDITIONS), op="cond_type"),
               Field(path + ("variable",), "Variable", "choice", _choices(kinds)),
               Field(path + ("selection",), "Boundary", "choice", boundaries)]
        for key, kind in M.CONDITIONS.get(c.get("type"), {}).items():
            out.append(_value_field(path + (key,), _LABELS.get(key, key),
                                    vkind if kind == "var" else kind, dim))
        if c.get("type") in ("dirichlet", "normal_dirichlet"):
            methods = M.DIRICHLET_METHODS if c.get("type") == "dirichlet" else M.DIRICHLET_METHODS[1:]
            out.append(Field(path + ("method",), "Method", "choice", _choices(methods)))
            if c.get("method") == "penalization":
                out.append(Field(path + ("penalty",), "Penalty coefficient", "number"))
        return out
    if head == "solver":
        out = [Field(path + ("type",), "Type", "choice",
                     (("auto", "automatic (linear or Newton)"), ("linear", "linear"), ("newton", "Newton")))]
        if get_at(model, path + ("type",), "auto") != "linear":
            out += [Field(path + ("max_iter",), "Maximum iterations", "int"),
                    Field(path + ("max_res",), "Residual tolerance", "number")]
        return out
    if head == "graphics":
        return [Field(path, "", "info", help="Saved views of the result fields, with their "
                      "display settings (changing the display of a result field saves it here).")] \
            if n == 1 else []
    if head == "outputs":
        if n == 1:
            return [Field(base + ("vtu",), "VTU file", "text")]
        return [Field(path + ("name",), "Name", "ident"),
                Field(path + ("expr",), "Integrand (GWFL)", "value"),
                Field(path + ("selection",), "Integrated on", "choice", anywhere)]
    return []


def fields(model: dict, path: Path) -> list[Field]:
    """Form fields of the node at `path`."""
    n = len(path)
    if path == ():
        return [Field(("name",), "Name", "text")]
    if path[0] == "models":
        if n == 1:
            return [Field(path, "", "info", help="Each model is a GetFEM model (gf.Model): variables, "
                          "modules, terms and conditions solved together.")]
        return _model_fields(model, path)

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
            return _shape_fields(model, path, shape_dim(model, path))
        if path[2] == "selections" and n == 3:
            return [Field(path, "", "info", help="Named parts of the geometry, used by properties, "
                                                 "conditions, terms and outputs.")]
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
                      "(\"L/2\") and expression. A parameter may use the ones listed before it."),
                Field(path, "", "parameter_table")]
    if path[0] == "parameters" and n == 2:
        return _param_fields(model, path)
    if path == ("materials",):
        return [Field(path, "", "info", help="Materials of the file: copies of library materials "
                      "or your own ones, used by the properties of the modules (\"Given by\"). "
                      "The table shows the properties asked by the modules of the file."),
                Field(path, "", "material_table")]
    if path[0] == "materials" and n == 2:
        mat = get_at(model, path, {})
        out = [Field(path + ("name",), "Name", "ident", op="rename_material"),
               Field(path + ("description",), "Description", "text")]
        if mat.get("from"):
            out.append(Field(path, "", "info", help=f"Copied from the {mat['from'].replace(':', ' ', 1)} "
                             "(the copy may be changed: the library is not)."))
        for p in mat.get("properties", {}):
            label, unit = M.PROPERTIES.get(p, (p, ""))
            out.append(Field(path + ("properties", p), f"{p} ({unit})" if unit and unit != "1" else p,
                             "value", help=label, remove=path + ("properties", p)))
        return out
    if path == ("fields",):
        return [Field(path, "", "info", help="Fields defined by a value per domain selection of a "
                      "geometry (the last one wins), usable in every expression.")]
    if path[0] == "fields" and n == 2:
        return _field_fields(model, path)

    if path == ("geometries",):
        return [Field(path, "", "info", help="Geometries and their named selections.")]
    if path == ("meshes",):
        return [Field(path, "", "info", help="Several meshes may be defined for one geometry; "
                                             "the variables of the models say which one they use.")]
    if path[0] == "meshes" and n == 2:
        mesh = get_at(model, path, {})
        geom = M.geometry_of(model, mesh) or {}
        stype = geom.get("source", {}).get("type")
        dim = (M.geometry_dim(model, geom) if geom else None) or 2
        meshers = [x for x, srcs in M.MESHERS.items() if stype in srcs] or list(M.MESHERS)
        out = [Field(path + ("name",), "Name", "ident", op="rename_mesh"),
               Field(path + ("geometry",), "Geometry", "choice",
                     _choices(g["name"] for g in model.get("geometries", []))),
               Field(path + ("mesher",), "Mesher", "choice", _choices(meshers), op="mesher")]
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

    if path == ("solvers",):
        return [Field(path, "", "info", help="Global solvers chain the models (the first one is run "
                      "by Compute). Without any, the models are solved in the order of their imports.")]
    if path[0] == "solvers" and n == 2:
        s = get_at(model, path, {})
        out = [Field(path + ("name",), "Name", "ident"),
               Field(path + ("type",), "Type", "choice",
                     (("sequence", "sequence"), ("staggered", "staggered iterations"))),
               Field(path + ("steps",), "Models, in order", "names",
                     _choices(m.get("name") for m in model.get("models", [])))]
        if s.get("type") == "staggered":
            out += [Field(path + ("max_iter",), "Maximum iterations", "int"),
                    Field(path + ("tolerance",), "Tolerance (relative change)", "number"),
                    Field(path + ("relaxation",), "Relaxation", "number")]
        return out
    return []


# ------------------------------------------------------------------ changes

def _default_shape(t: str, dim: int) -> dict:
    if t == "rectangle":
        return {"type": "rectangle", "min": [0.0] * dim, "max": [1.0] * dim}
    if t == "ball":
        return {"type": "ball", "center": [0.0] * dim, "radius": 0.5}
    if t == "polygon":
        return {"type": "polygon", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.6], [0.5, 1.0], [0.0, 0.6]]}
    if t == "ellipse":
        return {"type": "ellipse", "center": [0.0, 0.0], "radii": [0.6, 0.3], "angle": 0.0}
    if t == "cylinder":
        return {"type": "cylinder", "origin": [0.0] * dim, "axis": [0.0] * (dim - 1) + [1.0],
                "length": 1.0, "radius": 0.5}
    if t == "extrusion":
        return {"type": "extrusion", "z": [0.0, 0.5],
                "profile": {"type": "difference", "of": [
                    {"type": "rectangle", "min": [0.0, 0.0], "max": [2.0, 1.0]},
                    {"type": "ball", "center": [1.0, 0.5], "radius": 0.25}]}}
    if t == "revolution":
        return {"type": "revolution",
                "profile": {"type": "rectangle", "min": [0.5, 0.0], "max": [1.0, 1.0]}}
    return {"type": t, "of": [_default_shape("rectangle", dim), _default_shape("ball", dim)]}


def _shape_in_profile(model: dict, path: Path, t: str) -> dict:
    """A new primitive in the profile of an extrusion or a revolution, at its scale."""
    from . import geometry2d as G
    k = len(path) - 1 - path[::-1].index("profile")
    try:
        lo, hi = G.bounds(get_at(P.resolved(model), path[:k + 1]))
    except (KeyError, TypeError, ValueError, AttributeError, P.ParameterError):
        return _default_shape(t, 2)
    return _scaled_shape(t, lo, hi, 2)


def _scaled_shape(t: str, lo, hi, dim: int) -> dict:
    import numpy as np
    if not np.all(np.isfinite([*lo, *hi])) or np.any(hi <= lo):
        return _default_shape(t, dim)
    r = lambda v: float(f"{v:.3g}")          # noqa: E731
    c, e = (lo + hi) / 2, float((hi - lo).min())
    if t == "rectangle":
        return {"type": "rectangle", "min": [r(x - e / 8) for x in c], "max": [r(x + e / 8) for x in c]}
    if t == "ball":
        return {"type": "ball", "center": [r(x) for x in c], "radius": r(e / 8)}
    if t == "ellipse":
        return {"type": "ellipse", "center": [r(x) for x in c], "radii": [r(e / 6), r(e / 12)], "angle": 0.0}
    if t == "polygon":
        house = _default_shape("polygon", 2)["points"]
        return {"type": "polygon", "points": [[r(c[0] + (x - 0.5) * e / 4), r(c[1] + (y - 0.5) * e / 4)]
                                              for x, y in house]}
    return _default_shape(t, dim)


def _polygon_from(model: dict, path: Path, shape: dict) -> dict:
    """A polygon replacing a 2D primitive: the corners of a rectangle, an
    octagon in a disk, 12 points of an ellipse (the default polygon otherwise)."""
    try:
        s = get_at(P.resolved(model), path)
        if s.get("type") == "rectangle":
            (x0, y0), (x1, y1) = (float(v) for v in s["min"]), (float(v) for v in s["max"])
            return {"type": "polygon", "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}
        if s.get("type") == "ball":
            (cx, cy), rad = (float(v) for v in s["center"]), float(s["radius"])
            return {"type": "polygon", "points": [
                [float(f"{cx + rad * math.cos(a):.6g}"), float(f"{cy + rad * math.sin(a):.6g}")]
                for a in (k * math.pi / 4 for k in range(8))]}
        if s.get("type") == "ellipse":
            from . import geometry2d as G
            import numpy as np
            pts = G.sides(s)[0].points(np.arange(12) / 12.0)
            return {"type": "polygon", "points": [[float(f"{x:.6g}"), float(f"{y:.6g}")] for x, y in pts]}
    except (KeyError, TypeError, ValueError, AttributeError, P.ParameterError):
        pass
    return _default_shape("polygon", 2)


def _map_side_refs(model: dict, path: Path, mapping) -> None:
    """The sides of the 2D primitive at path were renamed or split: update
    the edge references of the selections (mapping(old side) -> new sides)."""
    if path[:4] != ("geometries", path[1], "source", "shape") or "profile" in path[4:] \
            or any(not isinstance(k, int) for k in path[5::2]):
        return
    prim = list(path[5::2])
    for sel in model["geometries"][path[1]].get("selections", {}).values():
        crit = sel.get("criteria", {})
        if crit.get("type") != "edges":
            continue
        out = []
        for ref in crit.get("edges", []):
            if list(ref.get("shape", [])) != prim:
                out.append(ref)
                continue
            for new in mapping(ref.get("side")):
                if not any(list(r.get("shape", [])) == prim and r.get("side") == new for r in out):
                    out.append({"shape": prim, "side": new})
        crit["edges"] = out


def _side_number(name) -> int | None:
    return int(name[4:]) if isinstance(name, str) and name.startswith("side") and name[4:].isdigit() else None


def _vertex_inserted(model: dict, path: Path, i: int) -> None:
    """A vertex inserted after the vertex i (0-based): the side i + 1 is split."""
    def mapping(name):
        k = _side_number(name)
        if k is None:
            return [name]
        return [f"side{k}", f"side{k + 1}"] if k == i + 1 else [f"side{k + 1 if k > i + 1 else k}"]
    _map_side_refs(model, path, mapping)


def _vertex_removed(model: dict, path: Path, i: int, n: int) -> None:
    """The vertex i (0-based) of a polygon of n vertices removed: its two
    sides become one."""
    def mapping(name):
        k = _side_number(name)
        if k is None:
            return [name]
        if i == 0:                     # sides n and 1 -> n - 1; k -> k - 1
            return [f"side{n - 1}"] if k in (1, n) else [f"side{k - 1}"]
        if k in (i, i + 1):
            return [f"side{i}"]
        return [f"side{k - 1 if k > i + 1 else k}"]
    _map_side_refs(model, path, mapping)


def vertex_rows(model: dict, path: Path) -> list[tuple]:
    """Rows of the vertex table of a polygon at path (the points list):
    (x, y) as given (numbers or expressions) and their values."""
    pts = get_at(model, path, []) or []
    try:
        vals = get_at(P.resolved(model), path, []) or []
    except P.ParameterError:
        vals = []
    out = []
    for i, p in enumerate(pts):
        v = vals[i] if i < len(vals) and isinstance(vals[i], list) else [None, None]
        out.append((p[0] if len(p) > 0 else 0.0, p[1] if len(p) > 1 else 0.0, v[0], v[1]))
    return out


def _shape_in_geometry(model: dict, gi: int, t: str) -> dict:
    """A new primitive for a 2D geometry: in its middle, at its scale (the
    default shape otherwise)."""
    from . import geometry2d as G
    dim = geometry_dim(model, gi)
    if dim != 2:
        return _default_shape(t, dim)
    try:
        lo, hi = G.bounds(G.shape_of(P.resolved(model)["geometries"][gi]["source"]))
    except (KeyError, TypeError, ValueError, AttributeError, P.ParameterError):
        return _default_shape(t, dim)
    return _scaled_shape(t, lo, hi, dim)


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
        "faces": {"type": "faces", "faces": []},
        "regions": {"type": "regions", "regions": []},
    }[t]


def _default_mesh_params(mesher: str, dim: int) -> dict:
    return {"structured": {"divisions": [10] * dim, "element": "simplex"},
            "getfem": {"h": 0.1, "order": 1},
            "gmsh": {"size": 0.1, "order": 1},
            "import": {}}[mesher]


def _zero(kind: str, dim: int):
    return [0.0] * dim if kind == M.VECTOR else 0.0


def _zero(kind: str, dim: int):
    return [0.0] * dim if kind == M.VECTOR else 0.0


def _default_condition(btype: str, values: dict, selection: str, dim: int) -> dict:
    bc = {"type": btype, "selection": selection}
    for key, kind in values.items():
        bc[key] = 20.0 if key == "T_ext" else (1.0 if key == "h" else _zero(kind, dim))
    return bc


def field_value(model: dict, f: Field):
    """The current value shown by a field (the model value, but for a few
    special fields)."""
    if f.op == "piece_source":
        pc = get_at(model, f.key, {})
        return f"material:{pc['material']}" if "material" in pc else "value"
    if f.op == "box_dim":
        return len(get_at(model, f.key, []))
    return get_at(model, f.key)


_WORD = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])"
_NOT_EXPRESSIONS = ("name", "selection", "geometry", "mesh", "type", "file", "description", "vtu",
                    "kind", "element", "mesher", "fem", "method", "model", "material", "linear",
                    "components", "side", "plane")


def rename_identifier(scope: dict, old: str, new: str) -> None:
    """Rename a parameter, a field or a variable, and its uses in the
    expressions of `scope` (the whole file or one model), whole words only
    (Grad_L is not a use of L, but Grad_T and Test_T are uses of the variable
    T). Variable references of the modules and conditions ("variable",
    "temperature"...) follow."""
    pattern = re.compile(r"(?<![A-Za-z0-9_])((?:Grad_|Hess_|Div_|Test_|Test2_|Dot_|Dot2_|Previous_|Previous2_)*)"
                         + re.escape(old) + r"(?![A-Za-z0-9_])")
    sub = pattern.sub

    class _P:      # pattern.sub(new, text), keeping the prefixes
        @staticmethod
        def sub(repl, text):
            return sub(lambda mt: mt.group(1) + repl, text)
    pattern = _P

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "name" and v == old and any(obj is x for x in named):
                    obj[k] = new
                elif k == "variable" and "model" in obj:
                    continue                 # an import: a variable of the other model
                elif isinstance(v, str) and k not in _NOT_EXPRESSIONS:
                    obj[k] = pattern.sub(new, v)
                elif isinstance(v, (dict, list)) and not (k in ("of", "steps")
                                                          and all(isinstance(x, str) for x in v)):
                    walk(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    obj[i] = pattern.sub(new, v)
                else:
                    walk(v)
    named = [x for key in ("parameters", "fields", "variables", "imports") for x in scope.get(key, [])
             if isinstance(x, dict) and x.get("name") == old]
    walk({k: v for k, v in scope.items() if not k.startswith("_")})
    # the imports of other models referring to a renamed variable keep their local name


def _items_with_selection(model: dict, geom: dict) -> list[dict]:
    """The items using a selection of a geometry: in the models on that
    geometry (properties, sources, conditions, terms, outputs, filtered
    variables, local fields) and the fields of the top level on it."""
    out = []

    def collect(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("selection"), str):
                out.append(obj)
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)
    for md in model.get("models", []):
        if M.model_geometry(model, md) is geom:
            collect({k: v for k, v in md.items() if k not in ("geometries", "meshes")})
    geoms = model.get("geometries", [])
    for f in model.get("fields", []):
        if M.by_name(geoms, f.get("geometry", geoms[0].get("name") if len(geoms) == 1 else None)) is geom:
            collect(f)
    return out


def _rename_selection(model, gi, old, new) -> None:
    geom = model["geometries"][gi]
    geom["selections"] = {(new if k == old else k): v for k, v in geom.get("selections", {}).items()}
    for sel in geom["selections"].values():
        crit = sel.get("criteria", {})
        if crit.get("type") == "union":
            crit["of"] = [new if o == old else o for o in crit.get("of", [])]
    for it in _items_with_selection(model, geom):
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
        for mesh, _ in M.all_meshes(model):
            if mesh.get("geometry") == old:
                mesh["geometry"] = value
        for fld in model.get("fields", []):
            if fld.get("geometry") == old:
                fld["geometry"] = value
    elif f.op in ("rename_parameter", "rename_field", "rename_variable"):
        old = get_at(model, key)
        if value != old:
            md = owner(model, key)
            rename_identifier(md if md is not None else model, old, value)
            if f.op == "rename_variable":
                for other in model.get("models", []):
                    for im in other.get("imports", []):
                        if im.get("model") == md.get("name") and im.get("variable") == old:
                            im["variable"] = value
    elif f.op == "rename_mesh":
        old = get_at(model, key)
        set_at(model, key, value)
        for md in model.get("models", []):
            for v in md.get("variables", []):
                if v.get("mesh") == old:
                    v["mesh"] = value
    elif f.op == "rename_model":
        old = get_at(model, key)
        set_at(model, key, value)
        for md in model.get("models", []):
            for im in md.get("imports", []):
                if im.get("model") == old:
                    im["model"] = value
        for s in model.get("solvers", []):
            s["steps"] = [value if x == old else x for x in s.get("steps", [])]
    elif f.op == "rename_material":
        old = get_at(model, key)
        set_at(model, key, value)
        for md in model.get("models", []):
            for mod in md.get("modules", []):
                for pieces in mod.get("properties", {}).values():
                    for pc in pieces:
                        if pc.get("material") == old:
                            pc["material"] = value
    elif f.op == "rename_selection":
        gi, old = key[1], key[3]
        if value == old:
            return None
        if value in M.selection_table(model, model["geometries"][gi]):
            raise ValueError(f"the name {value!r} is already used")
        _rename_selection(model, gi, old, value)
        return ("geometries", gi, "selections", value)
    elif f.op == "piece_source":
        pc = get_at(model, key)
        if value == "value":
            if "material" in pc:
                mat = M.material(model, pc.pop("material")) or {}
                p = key[-2]
                pc["value"] = (mat.get("properties") or {}).get(p, 1.0)
        else:
            pc.pop("value", None)
            pc["material"] = value.split(":", 1)[1]
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
        dim = shape_dim(model, path)
        if value == old:
            return None
        if value in M.SHAPES_BOOLEAN and old in M.SHAPES_BOOLEAN:
            shape["type"] = value
        elif value in M.SHAPES_BOOLEAN:
            set_at(model, path, {"type": value, "of": [shape, _default_shape("ball", dim)]})
        elif value == "ellipse" and old == "ball":       # same center, both semi-axes the radius
            set_at(model, path, {"type": "ellipse", "center": copy.deepcopy(shape.get("center", [0.0, 0.0])),
                                 "radii": [copy.deepcopy(shape.get("radius", 0.5))] * 2, "angle": 0.0})
            _map_side_refs(model, path, lambda name: ["ellipse" if name == "circle" else name])
        elif value == "ball" and old == "ellipse":
            set_at(model, path, {"type": "ball", "center": copy.deepcopy(shape.get("center", [0.0, 0.0])),
                                 "radius": copy.deepcopy((shape.get("radii") or [0.5])[0])})
            _map_side_refs(model, path, lambda name: ["circle" if name == "ellipse" else name])
        elif value == "polygon" and old in ("rectangle", "ball", "ellipse"):
            poly = _polygon_from(model, path, shape)
            set_at(model, path, poly)
            n = len(poly["points"])
            if old == "rectangle" and n == 4:
                names = {"ymin": "side1", "xmax": "side2", "ymax": "side3", "xmin": "side4"}
                _map_side_refs(model, path, lambda name: [names.get(name, name)])
            else:
                _map_side_refs(model, path, lambda name: [f"side{k + 1}" for k in range(n)]
                               if name in ("circle", "ellipse") else [name])
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
            _change_mesher(model, mesh, value)
    elif f.op == "mcond_type":
        bc = get_at(model, key[:-1])
        if bc.get("type") != value:
            mod = get_at(model, key[:-3])
            spec = M.MODULE_SPECS[mod["type"]]
            set_at(model, key[:-1], _default_condition(value, spec["conditions"][value],
                                                       bc.get("selection", M.BOUNDARY), item_dim(model, key)))
    elif f.op == "cond_type":
        c = get_at(model, key[:-1])
        if c.get("type") != value:
            md = owner(model, key)
            var = M.by_name(md.get("variables", []), c.get("variable")) or {}
            vk = M.variable_kind(var)
            values = {k2: (vk if kd == "var" else kd) for k2, kd in M.CONDITIONS[value].items()}
            new = _default_condition(value, values, c.get("selection", M.BOUNDARY), item_dim(model, key))
            new = {"type": value, "variable": c.get("variable"), **{k2: v for k2, v in new.items() if k2 != "type"}}
            if value in ("dirichlet", "normal_dirichlet"):
                new["method"] = "simplification" if value == "dirichlet" else "multiplier"
            set_at(model, key[:-1], new)
    else:
        raise ValueError(f"unknown operation {f.op!r}")
    return None


def _change_mesher(model: dict, mesh: dict, mesher: str) -> None:
    """Change the mesher of a mesh, keeping its fineness: the element size is
    kept between the GetFEM mesher and Gmsh (an expression of parameters
    too), and converted from or to the divisions of a structured mesh; the
    geometric order is kept."""
    geom = M.geometry_of(model, mesh) or {}
    dim = (M.geometry_dim(model, geom) if geom else None) or 2
    old = mesh.get("mesher")
    size = mesh.get("h" if old == "getfem" else "size") if old in ("getfem", "gmsh") else None
    order = mesh.get("order", 1)
    rmesh = M.find_mesh(P.resolved(model), mesh.get("name")) or {}
    rgeom = M.geometry_of(P.resolved(model), rmesh) or {}
    src = rgeom.get("source", {})
    lengths = None
    if src.get("type") == "box" and all(isinstance(x, (int, float)) for x in src.get("size", [])):
        lengths = [float(x) for x in src["size"]]
    if old == "structured" and lengths:
        div = rmesh.get("divisions", [])
        if len(div) == len(lengths) and all(isinstance(d, int) and d > 0 for d in div):
            size = float(f"{min(l / d for l, d in zip(lengths, div)):.6g}")
    for k in ("divisions", "element", "h", "size", "order"):
        mesh.pop(k, None)
    mesh["mesher"] = mesher
    mesh.update(_default_mesh_params(mesher, dim))
    if mesher in ("getfem", "gmsh"):
        if size is not None:
            mesh["h" if mesher == "getfem" else "size"] = size
        mesh["order"] = order
    elif mesher == "structured" and lengths:
        h = rmesh.get("h" if old == "getfem" else "size")
        if isinstance(h, (int, float)) and h > 0:
            mesh["divisions"] = [max(1, int(math.ceil(l / h - 1e-9))) for l in lengths]


def _fit_dimension(model: dict, gi: int, dim: int) -> None:
    """After a change of dimension: adapt the meshes of the geometry, the
    criteria of its selections, and the models using it (the plane option,
    the vector values)."""
    geom = model["geometries"][gi]
    for mesh in model.get("meshes", []):
        if mesh.get("geometry") == geom["name"] and mesh.get("mesher") == "structured":
            div = list(mesh.get("divisions", []))
            mesh["divisions"] = (div + [10] * dim)[:dim]
    for sel in geom.get("selections", {}).values():
        crit = sel.get("criteria", {})
        for k, v in list(crit.items()):
            if isinstance(v, list) and k not in ("of", "edges", "regions"):
                crit[k] = (list(v) + [0.0] * dim)[:dim]
    for md in model.get("models", []):
        if M.model_geometry(model, md) is not geom:
            continue
        for mod in md.get("modules", []):
            spec = M.MODULE_SPECS.get(mod.get("type"), {})
            if dim != 2:
                mod.pop("plane", None)
            elif "plane" in spec.get("options", {}):
                mod.setdefault("plane", spec["options"]["plane"][0])
            if spec.get("source") == M.VECTOR:
                for s in mod.get("sources", []):
                    if isinstance(s.get("value"), list):
                        s["value"] = (s["value"] + [0.0] * dim)[:dim]
            for bc in mod.get("conditions", []):
                for k, kind in spec.get("conditions", {}).get(bc.get("type"), {}).items():
                    if kind == M.VECTOR and isinstance(bc.get(k), list):
                        bc[k] = (bc[k] + [0.0] * dim)[:dim]
        for c in md.get("conditions", []):
            for k in ("value",):
                if isinstance(c.get(k), list):
                    c[k] = (c[k] + [0.0] * dim)[:dim]


# ------------------------------------------------------------------ actions

def actions(model: dict, path: Path) -> list[tuple[str, str]]:
    """Possible operations on a node: [(identifier, label)]. Headers offer to
    add their items (the interface shows them in a context menu and as a +
    button)."""
    n = len(path)
    out: list[tuple[str, str]] = []
    md = owner(model, path)
    rest = path[2:] if md is not None else ()
    if path == ():
        out += [("add_parameter", "Add a parameter"), ("add_material", "Add a material"),
                ("add_geometry", "Add a geometry"), ("add_field", "Add a field")]
        out += [(f"add_model:{t}", f"Add a model: {s['label'].lower()}") for t, s in M.MODULE_SPECS.items()
                if "requires" not in s]
        out += [("add_model:", "Add an empty model"), ("add_solver", "Add a global solver")]
    elif path == ("parameters",):
        out.append(("add_parameter", "Add a parameter"))
    elif path == ("materials",):
        out += [("add_from_library", "Add from a library…"), ("add_material", "Add an empty material")]
    elif path[:1] == ("materials",) and n == 2:
        out += [("add_property", "Add a property"), ("cmd:save_material", "Save in my library"),
                ("remove", "Remove")]
    elif path == ("fields",):
        out.append(("add_field", "Add a field"))
    elif path[:1] in (("fields",), ("parameters",)) and n == 2:
        if path[0] == "fields":
            out.append(("add_field_value", "Add a value on a domain"))
        out.append(("remove", "Remove"))
    elif path == ("geometries",):
        out.append(("add_geometry", "Add a geometry"))
    elif path[:1] == ("geometries",) and n == 2:
        out += [("add_selection", "Add a selection"), ("add_mesh", "Add a mesh of this geometry")]
        if len(model.get("geometries", [])) > 1:
            out.append(("remove", "Remove"))
    elif path[:1] == ("geometries",) and n > 3 and path[2] == "source":
        shape = get_at(model, path, {})
        if shape.get("type") in M.SHAPES_BOOLEAN:
            out += [(f"add_shape:{t}", f"Add a{'n' if t[0] in 'aeiou' else ''} {t}")
                    for t in primitives_of_dim(shape_dim(model, path))]
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
        if len(model.get("meshes", [])) > 1:
            out.append(("remove", "Remove"))
    elif path == ("models",):
        out += [(f"add_model:{t}", f"Add a model: {s['label'].lower()}") for t, s in M.MODULE_SPECS.items()
                if "requires" not in s]
        out.append(("add_model:", "Add an empty model"))
    elif path == ("solvers",):
        out.append(("add_solver", "Add a global solver"))
    elif path[:1] == ("solvers",) and n == 2:
        out.append(("remove", "Remove"))
    elif md is not None and not rest:
        out += [("add_variable", "Add a variable"), ("add_module", "Add a module…"),
                ("add_term", "Add a term"), ("add_condition", "Add a condition"),
                ("add_quantity", "Add an integrated quantity"),
                ("add_local_parameter", "Add a parameter"), ("add_local_field", "Add a field"),
                ("add_import", "Import a variable of another model")]
        if len(model.get("models", [])) > 1:
            out.append(("remove", "Remove"))
    elif rest == ("parameters",):
        out.append(("add_local_parameter", "Add a parameter"))
    elif rest == ("fields",):
        out.append(("add_local_field", "Add a field"))
    elif rest == ("imports",):
        out.append(("add_import", "Import a variable of another model"))
    elif rest == ("variables",):
        out.append(("add_variable", "Add a variable"))
    elif rest == ("modules",):
        present = {m.get("type") for m in md.get("modules", [])}
        out += [(f"add_module:{t}", f"Add {s['label'].lower()}") for t, s in M.MODULE_SPECS.items()
                if t not in present]      # one module of each type per model
    elif rest == ("terms",):
        out.append(("add_term", "Add a term"))
    elif rest == ("conditions",):
        out.append(("add_condition", "Add a condition"))
    elif rest == ("outputs",):
        out.append(("add_quantity", "Add an integrated quantity"))
    elif rest == ("graphics",):
        out.append(("add_graphic", "Add a graphic"))
    elif rest[:1] == ("modules",) and len(rest) == 2:
        spec = M.MODULE_SPECS.get(get_at(model, path + ("type",)), {})
        out += [(f"add_piece:{p}", f"Add a domain for {p}") for p in spec.get("properties", {})]
        if spec.get("source") is not None:
            out.append(("add_source", "Add a source"))
        if spec.get("conditions"):
            out.append(("add_bc", "Add a condition"))
        out += [("unfold", "Unfold into terms and conditions")] if _can_unfold(get_at(model, path)) else []
        out.append(("remove", "Remove"))
    elif rest[:1] == ("modules",) and len(rest) == 3 and rest[2] == "sources":
        out.append(("add_source", "Add a source"))
    elif rest[:1] == ("modules",) and len(rest) == 3 and rest[2] == "conditions":
        out.append(("add_bc", "Add a condition"))
    elif rest[:1] == ("modules",) and len(rest) == 5 and rest[2] == "properties":
        if len(get_at(model, path[:-1], [])) > 1:
            out.append(("remove", "Remove"))
    elif rest[:1] in (("parameters",), ("fields",), ("imports",), ("variables",), ("terms",),
                      ("conditions",), ("outputs",), ("graphics",)) and len(rest) == 2:
        if rest[0] == "fields":
            out.append(("add_field_value", "Add a value on a domain"))
        out.append(("remove", "Remove"))
    elif rest[:1] == ("modules",) and len(rest) == 4 and rest[2] in ("sources", "conditions"):
        out.append(("remove", "Remove"))
    return out


def needs_name(action: str) -> bool:
    """Actions for which the interface asks for a name first."""
    return action == "add_property"


def _names_in(model: dict, md: dict | None) -> set:
    names = set(M.global_names(model))
    for m in model.get("models", []):
        if md is None or m is md:
            names |= set(M.local_names(m))
    return names


def _first_boundary(model, md, used=()) -> str:
    boundaries = selections_of_kind(model, M.BOUNDARY, M.model_geometry(model, md))
    return next((s for s in boundaries if s != M.BOUNDARY and s not in used),
                boundaries[0] if boundaries else M.BOUNDARY)


def do_action(model: dict, path: Path, action: str, name: str | None = None) -> Path | None:
    """Perform an action; returns the path of the node to select afterwards."""
    md = owner(model, path)
    dim = item_dim(model, path) if md is not None else \
        ((M.geometry_dim(model, model["geometries"][0]) if model.get("geometries") else None) or 2)
    if action.split(":")[0] in ("insert_vertex", "remove_vertex"):
        verb, i = action.split(":")
        pts, i = get_at(model, path + ("points",)), int(i)
        if verb == "remove_vertex":
            if len(pts) <= 3:
                raise ValueError("a polygon keeps at least three vertices")
            del pts[i]
            _vertex_removed(model, path, i, len(pts) + 1)
        else:                              # a new vertex in the middle of the side i
            r = get_at(P.resolved(model), path + ("points",))
            a, b = r[i], r[(i + 1) % len(r)]
            pts.insert(i + 1, [float(f"{(a[k] + b[k]) / 2:.6g}") for k in range(2)])
            _vertex_inserted(model, path, i)
        return path
    if action.split(":")[0] in ("remove_row", "move_up", "move_down"):
        verb, i = action.split(":")
        items, i = get_at(model, path), int(i)
        if verb == "remove_row":
            del items[i]
        elif verb == "move_up" and i > 0:
            items[i - 1], items[i] = items[i], items[i - 1]
        elif verb == "move_down" and i + 1 < len(items):
            items[i + 1], items[i] = items[i], items[i + 1]
        return path
    if action == "remove":
        parent = get_at(model, path[:-1])
        if isinstance(parent, list):
            del parent[path[-1]]
            if parent:
                return path[:-1] + (min(path[-1], len(parent) - 1),)
        else:
            del parent[path[-1]]
        return path[:-2] if path[-2] == "of" else path[:-1]
    if action in ("add_parameter", "add_local_parameter"):
        target = model if action == "add_parameter" else md
        params = target.setdefault("parameters", [])
        params.append({"name": unique_name(name or "p", _names_in(model, md)), "value": 1.0, "description": ""})
        return (("models", path[1]) if md is not None else ()) + ("parameters",)
    if action in ("add_field", "add_local_field"):
        target = model if action == "add_field" else md
        fields_ = target.setdefault("fields", [])
        f = {"name": unique_name(name or "f", _names_in(model, md)), "description": "",
             "values": [{"selection": M.ALL, "value": 0.0}]}
        if action == "add_field" and model.get("geometries"):
            f = {"name": f["name"], "geometry": model["geometries"][0]["name"], **{k: v for k, v in f.items()
                                                                                 if k != "name"}}
        fields_.append(f)
        return (("models", path[1]) if md is not None else ()) + ("fields", len(fields_) - 1)
    if action == "add_field_value":
        fpath = path if path[-2] == "fields" else path
        get_at(model, fpath).setdefault("values", []).append({"selection": M.ALL, "value": 0.0})
        return path
    if action.startswith("add_from_library:"):
        from . import library
        return ("materials", library.copy_into(model, action.split(":", 1)[1]))
    if action == "add_material":
        mats = model.setdefault("materials", [])
        mats.append({"name": unique_name(name or "material", [m.get("name") for m in mats]),
                     "description": "", "properties": {}})
        return ("materials", len(mats) - 1)
    if action == "add_property":
        if not name or not M._IDENT.match(name):
            raise ValueError("a property name (letters, digits, _) is required")
        props = model["materials"][path[1]].setdefault("properties", {})
        if name in props:
            raise ValueError(f"the property {name!r} already exists")
        props[name] = 1.0
        return path
    if action == "add_geometry":
        geoms = model.setdefault("geometries", [])
        gname = unique_name(name or "geometry", [g.get("name") for g, _ in M.all_geometries(model)])
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
        t = action.split(":", 1)[1]
        new = _shape_in_geometry(model, path[1], t) if shape_dim(model, path) == geometry_dim(model, path[1]) \
            else _shape_in_profile(model, path, t)
        shape.setdefault("of", []).append(new)
        return path + ("of", len(shape["of"]) - 1)
    if action == "add_mesh":
        meshes = model.setdefault("meshes", [])
        geom = model["geometries"][path[1]] if path[:1] == ("geometries",) else model["geometries"][0]
        stype = geom.get("source", {}).get("type")
        mesher = next(x for x, srcs in M.MESHERS.items() if stype in srcs)
        gdim = M.geometry_dim(model, geom) or 2
        mesh = {"name": unique_name(name or "mesh", [m.get("name") for m, _ in M.all_meshes(model)]),
                "geometry": geom["name"], "mesher": mesher}
        mesh.update(_default_mesh_params(mesher, gdim))
        meshes.append(mesh)
        return ("meshes", len(meshes) - 1)
    if action.startswith("add_model:"):
        mtype = action.split(":", 1)[1] or None
        models = model.setdefault("models", [])
        base = {"heat_conduction": "heat", "linear_elasticity": "mechanics"}.get(mtype, "model")
        mname = unique_name(name or base, [m.get("name") for m in models])
        mesh = (model.get("meshes") or [{}])[0].get("name", "mesh")
        new = M.new_model(mname, mesh, mtype, dim)
        if mtype is None:                       # one variable to start with
            new["variables"].append(M.new_variable(unique_name("u", _names_in(model, None)), mesh))
        taken = _names_in(model, None)
        for v in new["variables"]:          # names already used in the file: the next ones
            if v["name"] in M.global_names(model):
                old = v["name"]
                v["name"] = unique_name(old, taken)
                for mod in new["modules"]:
                    for role in M.MODULE_SPECS[mod["type"]]["variables"]:
                        if mod.get(role) == old:
                            mod[role] = v["name"]
        models.append(new)
        return ("models", len(models) - 1)
    if action == "add_solver":
        solvers = model.setdefault("solvers", [])
        solvers.append({"name": unique_name(name or "solver", [s.get("name") for s in solvers]),
                        "type": "sequence", "steps": M.solve_order(model)})
        return ("solvers", len(solvers) - 1)
    base = ("models", path[1]) if md is not None else ()
    mesh_name = (M.model_mesh(model, md) or (model.get("meshes") or [{}])[0]).get("name", "mesh") \
        if md is not None else None
    if action == "add_variable":
        vars_ = md.setdefault("variables", [])
        vars_.append(M.new_variable(unique_name(name or "v", _names_in(model, md)), mesh_name, M.SCALAR, 1))
        return base + ("variables", len(vars_) - 1)
    if action == "add_import":
        others = [m for m in model.get("models", []) if m is not md and m.get("variables")]
        if not others:
            raise ValueError("no other model to import from")
        src = others[0]
        var = src["variables"][0]["name"]
        imps = md.setdefault("imports", [])
        imps.append({"name": unique_name(var, _names_in(model, md)), "model": src["name"], "variable": var})
        return base + ("imports", len(imps) - 1)
    if action == "add_module":
        return base + ("modules",)
    if action.startswith("add_module:"):
        mtype = action.split(":", 1)[1]
        spec = M.MODULE_SPECS[mtype]
        names = {}
        existing = {v["name"]: M.variable_kind(v) for v in md.get("variables", [])}
        imports = [im["name"] for im in md.get("imports", [])]
        for role, v in spec["variables"].items():
            same = [n for n, k in existing.items() if k == v["kind"]] + (imports if v.get("import") else [])
            if v["name"] in same or (v["name"] in imports and v.get("import")):
                names[role] = v["name"]
            elif same and role != "variable":
                names[role] = same[0]
            else:                            # a new variable
                vname = unique_name(v["name"], _names_in(model, md))
                md.setdefault("variables", []).append(M.new_variable(vname, mesh_name, v["kind"], v["degree"]))
                names[role] = vname
                existing[vname] = v["kind"]
        mods = md.setdefault("modules", [])
        mods.append(M.new_module(mtype, dim, names))
        if "requires" in spec:
            other, role = spec["requires"]
            if not any(m.get("type") == other and m.get("variable") == names[role] for m in mods):
                mods.insert(len(mods) - 1, M.new_module(other, dim, {"variable": names[role]}))
        return base + ("modules", len(mods) - 1)
    if action.startswith("add_piece:"):
        p = action.split(":", 1)[1]
        pieces = get_at(model, path).setdefault("properties", {}).setdefault(p, [])
        default = M.MODULE_SPECS[get_at(model, path + ("type",))]["property_defaults"].get(p, 1.0)
        pieces.append({"selection": M.ALL, "value": default})
        return path + ("properties", p, len(pieces) - 1)
    if action == "add_source":
        mpath = path[:4]
        mod = get_at(model, mpath)
        spec = M.MODULE_SPECS[mod["type"]]
        srcs = mod.setdefault("sources", [])
        srcs.append({"selection": M.ALL, "value": _zero(spec.get("source", M.SCALAR), dim)})
        return mpath + ("sources", len(srcs) - 1)
    if action == "add_bc":
        mpath = path[:4]
        mod = get_at(model, mpath)
        spec = M.MODULE_SPECS[mod["type"]]
        bcs = mod.setdefault("conditions", [])
        btype = next(iter(spec["conditions"]))
        sel = _first_boundary(model, md, {bc.get("selection") for bc in bcs})
        bcs.append(_default_condition(btype, spec["conditions"][btype], sel, dim))
        return mpath + ("conditions", len(bcs) - 1)
    if action == "add_term":
        terms = md.setdefault("terms", [])
        var = (md.get("variables") or [{"name": "u"}])[0]["name"]
        terms.append({"name": unique_name(name or "term", [t.get("name") for t in terms]),
                      "expr": f"Grad_{var}.Grad_Test_{var}", "selection": M.ALL})
        return base + ("terms", len(terms) - 1)
    if action == "add_condition":
        conds = md.setdefault("conditions", [])
        var = (md.get("variables") or [{"name": "u"}])[0]
        kind = M.variable_kind(var)
        conds.append({"type": "dirichlet", "variable": var["name"],
                      "selection": _first_boundary(model, md, {c.get("selection") for c in conds}),
                      "value": _zero(kind, dim), "method": "simplification"})
        return base + ("conditions", len(conds) - 1)
    if action == "add_quantity":
        qs = md.setdefault("outputs", [])
        qs.append({"name": unique_name(name or "quantity", [q.get("name") for q in qs]),
                   "expr": "1", "selection": M.ALL})
        return base + ("outputs", len(qs) - 1)
    if action == "add_graphic":
        return add_graphic(md, None, path[:2])
    if action == "unfold":
        return unfold_module(model, path)
    raise ValueError(f"unknown action {action!r}")


def remove_at(model: dict, path: Path) -> None:
    """Delete the item at path (dictionary key or list element)."""
    parent = get_at(model, path[:-1])
    if isinstance(parent, list):
        del parent[path[-1]]
    else:
        parent.pop(path[-1], None)


# ------------------------------------------------------------------ graphics

def default_graphic_field(md: dict) -> str:
    """The field shown by a new graphic: von Mises for an elasticity model,
    otherwise the first variable."""
    if any(m.get("type") == "linear_elasticity" for m in md.get("modules", [])):
        return "von_mises"
    return (md.get("variables") or [{"name": "u"}])[0]["name"]


def find_graphic(md: dict, field: str) -> int | None:
    """Index of the graphic named after a field (the one a result node shows)."""
    return next((i for i, g in enumerate(md.get("graphics", [])) if g.get("name") == field), None)


def add_graphic(md: dict, field: str | None, base: Path = ()) -> Path:
    """A graphic of a field with the default settings, named after it."""
    field = field or default_graphic_field(md)
    graphics = md.setdefault("graphics", [])
    graphics.append({"name": unique_name(field, [g.get("name") for g in graphics]), "field": field})
    return base + ("graphics", len(graphics) - 1)


def graphic_settings(graphic: dict | None) -> dict:
    """The display settings of a graphic, the missing ones by default."""
    out = copy.deepcopy(M.GRAPHIC_DEFAULTS)
    out.update({k: copy.deepcopy(v) for k, v in (graphic or {}).items() if k in M.GRAPHIC_DEFAULTS})
    return out


def set_graphic(graphic: dict, change: dict) -> None:
    """Change settings of a graphic; a default value is not stored."""
    for k, v in change.items():
        if k in ("name", "field"):
            graphic[k] = v
        elif v == M.GRAPHIC_DEFAULTS.get(k, object()):
            graphic.pop(k, None)
        else:
            graphic[k] = v


# ------------------------------------------------------------------ unfolding a module

def _can_unfold(mod: dict) -> bool:
    return isinstance(mod, dict) and mod.get("type") == "heat_conduction"


def unfold_module(model: dict, path: Path) -> Path:
    """Replace a heat conduction module by its GWFL terms and generic
    conditions (the conductivity becomes a local field). The other modules
    are not unfoldable yet (their macros: sigma, von_mises...)."""
    md = owner(model, path)
    mod = get_at(model, path)
    if not _can_unfold(mod):
        raise ValueError("only the heat conduction module can be unfolded for now")
    u = mod["variable"]
    pieces = mod.get("properties", {}).get("k", [])
    values = []
    for pc in pieces:
        v = pc.get("value")
        if "material" in pc:
            v = (M.material(model, pc["material"]) or {}).get("properties", {}).get("k", 1.0)
        values.append({"selection": pc.get("selection", M.ALL), "value": v})
    md.setdefault("fields", []).append({"name": "k", "description": "thermal conductivity",
                                        "values": values})
    terms = md.setdefault("terms", [])
    taken = [t.get("name") for t in terms]
    terms.append({"name": unique_name("conduction", taken), "expr": f"k*Grad_{u}.Grad_Test_{u}",
                  "selection": M.ALL})
    for j, s in enumerate(mod.get("sources", [])):
        terms.append({"name": unique_name("source", [t["name"] for t in terms]),
                      "expr": f"-({format_value(s['value'])})*Test_{u}", "selection": s.get("selection", M.ALL)})
    conds = md.setdefault("conditions", [])
    for bc in mod.get("conditions", []):
        sel = bc.get("selection")
        if bc["type"] == "temperature":
            conds.append({"type": "dirichlet", "variable": u, "selection": sel, "value": bc["value"],
                          "method": "simplification"})
        elif bc["type"] == "heat_flux":
            conds.append({"type": "neumann", "variable": u, "selection": sel, "value": bc["value"]})
        elif bc["type"] == "convection":
            conds.append({"type": "robin", "variable": u, "selection": sel, "h": bc["h"], "value": bc["T_ext"]})
    md["modules"].remove(mod)
    for q in md.get("outputs", []):          # the macro q of the module
        q["expr"] = re.sub(_WORD.format("q"), f"(-k*Grad_{u})", str(q.get("expr", "")))
    return path[:2] + ("terms",)


# ------------------------------------------------------------------ what the view shows

def geometry_mesh(model: dict, geom: dict) -> dict | None:
    """The mesh shown for a geometry: one used by a model, else the first one."""
    meshes = [m for m in model.get("meshes", []) if m.get("geometry") == geom.get("name")]
    for md in model.get("models", []):
        mesh = M.model_mesh(model, md)
        if mesh in meshes:
            return mesh
    return meshes[0] if meshes else None


def highlighted_selection(model: dict, path: Path) -> tuple[str | None, str | None]:
    """(mesh name, selection name) to highlight when a node is selected:
    the selection itself, the domain of a property or a source, the boundary
    of a condition, the support of a term or an output."""
    if path[:1] == ("geometries",) and len(path) == 4 and path[2] == "selections":
        mesh = geometry_mesh(model, model["geometries"][path[1]]) or {}
        return mesh.get("name"), path[3]
    md = owner(model, path)
    item = get_at(model, path)
    if md is not None and isinstance(item, dict) and "selection" in item and len(path) > 2:
        return (M.model_mesh(model, md) or {}).get("name"), item.get("selection", M.ALL)
    return None, None


def mesh_to_show(model: dict, path: Path) -> str | None:
    """Name of the mesh the viewer should show for a node."""
    if path[:1] == ("meshes",) and len(path) == 2:
        return get_at(model, path + ("name",))
    if path[:1] == ("geometries",) and len(path) >= 2:
        return (geometry_mesh(model, model["geometries"][path[1]]) or {}).get("name")
    md = owner(model, path) or main_model(model)
    return (M.model_mesh(model, md) or {}).get("name") if md else None


def part_3d(model: dict, path: Path) -> dict | None:
    """The 3D shape drawn over the preview for the shape node at path; in
    the profile of an extrusion or a revolution, the part of the profile
    extruded or turned likewise."""
    shape = get_at(model, path)
    if "profile" not in path[4:] or not isinstance(shape, dict):
        return shape
    k = path.index("profile", 4)
    swept = get_at(model, path[:k])
    return dict(swept, profile=shape) if isinstance(swept, dict) else None


def preview_geometry(model: dict, path: Path) -> int | None:
    """Index of the (top level) geometry previewed for a node."""
    if path[:1] == ("geometries",) and len(path) >= 2:
        return path[1]
    md = owner(model, path) or (main_model(model) if path[:1] not in (("meshes",),) else None)
    geom = M.model_geometry(model, md) if md else None
    geoms = model.get("geometries", [])
    return next((i for i, g in enumerate(geoms) if g is geom), None)


# ------------------------------------------------------------------ picking edges and regions for an item

def selection_uses(model: dict, geom: dict, name: str) -> int:
    """How many items (properties, sources, conditions, terms, outputs,
    fields, unions) use a selection of a geometry."""
    n = sum(1 for it in _items_with_selection(model, geom) if it.get("selection", M.ALL) == name)
    for sel in geom.get("selections", {}).values():
        crit = sel.get("criteria", {})
        if crit.get("type") == "union" and name in crit.get("of", []):
            n += 1
    return n


def _item_base_name(model: dict, path: Path, item: dict) -> str:
    rest = path[2:]
    if rest[:1] == ("modules",) and len(rest) == 5 and rest[2] == "properties":
        return item.get("material") or f"{rest[3]}_{rest[4] + 1}"
    if rest[:1] == ("modules",) and len(rest) == 4:
        return f"{item.get('type', 'source')}_{rest[3] + 1}" if rest[2] == "conditions" \
            else f"source_{rest[3] + 1}"
    if rest[:1] == ("conditions",):
        return f"{item.get('type', 'condition')}_{rest[1] + 1}"
    if rest[:1] == ("terms",):
        return f"{item.get('name', 'term')}_support"
    return f"{item.get('name', 'output')}_support"


def _pickable(model: dict, path: Path, kind: str, dim: int = 2) -> tuple[int, dict] | None:
    """(geometry index, item) when the item at path has a selection of that
    kind on a box or csg geometry of the top level, of dimension dim (2 or 3)."""
    from . import geometry2d as G2, geometry3d as G3
    G = G2 if dim == 2 else G3
    md = owner(model, path)
    item = get_at(model, path)
    if md is None or len(path) < 4 or not isinstance(item, dict) or "selection" not in item:
        return None
    geom = M.model_geometry(model, md)
    geoms = model.get("geometries", [])
    gi = next((i for i, g in enumerate(geoms) if g is geom), None)
    if gi is None or G.shape_of(P.resolved(model)["geometries"][gi].get("source", {})) is None:
        return None
    if M.selection_table(model, geom).get(item.get("selection", M.ALL), {}).get("kind") != kind:
        return None
    return gi, item


def can_pick_edges(model: dict, path: Path) -> bool:
    """Whether clicks on edges edit the support of the item at path (a
    condition, a term or an output on a boundary)."""
    return _pickable(model, path, M.BOUNDARY) is not None


def pick_edge_for_item(model: dict, path: Path, ref: dict) -> tuple[str, bool]:
    """Add or remove an edge (side reference) in the support of the item at
    path. Its selection is edited in place when it is an 'edges' selection
    used by this item only; otherwise a new selection is created, starting
    from the current support (empty if it is the whole boundary), named
    after the item. Returns (selection name, created)."""
    from . import geometry2d as G
    gi, item = _pickable(model, path, M.BOUNDARY)
    geom = model["geometries"][gi]
    resolved = P.resolved(model)
    rgeom = resolved["geometries"][gi]
    table = M.selection_table(resolved, rgeom)
    name = item.get("selection", M.ALL)
    sel = geom.get("selections", {}).get(name)
    if sel is not None and sel.get("criteria", {}).get("type") == "edges" \
            and selection_uses(model, geom, name) == 1:
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
    new = unique_name(_item_base_name(model, path, item), table)
    geom.setdefault("selections", {})[new] = {"kind": M.BOUNDARY, "criteria": criteria}
    item["selection"] = new
    return new, True


def can_pick_faces(model: dict, path: Path) -> bool:
    """Whether clicks on faces of a 3D preview edit the support of the item at
    path (a condition, a term or an output on a boundary)."""
    return _pickable(model, path, M.BOUNDARY, dim=3) is not None


def can_pick_faces_of_selection(model: dict, path: Path) -> bool:
    """Whether the node at path is a user boundary selection of a 3D box or
    csg geometry (clicks on faces then edit it)."""
    from . import geometry3d as G3
    if not (path[:1] == ("geometries",) and len(path) == 4 and path[2] == "selections"):
        return False
    sel = model["geometries"][path[1]].get("selections", {}).get(path[3])
    return sel is not None and sel.get("kind") == M.BOUNDARY and \
        G3.shape_of(P.resolved(model)["geometries"][path[1]].get("source", {})) is not None


def pick_face_for_item(model: dict, path: Path, ref: dict, start=()) -> tuple[str, bool]:
    """Add or remove a face (geometry3d reference) in the support of the item
    at path, as pick_edge_for_item does for edges. start: the faces of the
    current support (computed on the preview), for a new selection."""
    from . import geometry3d as G3
    gi, item = _pickable(model, path, M.BOUNDARY, dim=3)
    geom = model["geometries"][gi]
    table = M.selection_table(model, geom)
    name = item.get("selection", M.ALL)
    sel = geom.get("selections", {}).get(name)
    if sel is not None and sel.get("criteria", {}).get("type") == "faces" \
            and selection_uses(model, geom, name) == 1:
        sel["criteria"] = G3.toggle_face(sel["criteria"], ref)
        return name, False
    begin = [] if name == M.BOUNDARY else [dict(r) for r in start]
    criteria = G3.toggle_face({"type": "faces", "faces": begin}, ref)
    new = unique_name(_item_base_name(model, path, item), table)
    geom.setdefault("selections", {})[new] = {"kind": M.BOUNDARY, "criteria": criteria}
    item["selection"] = new
    return new, True


def can_pick_regions(model: dict, path: Path) -> bool:
    """Whether clicks in the domain edit the support of the item at path (a
    property, a source, a term or an output on a domain), or the selection at
    path (a domain selection of a 2D box or csg geometry)."""
    from . import geometry2d as G
    if path[:1] == ("geometries",) and len(path) == 4 and path[2] == "selections":
        sel = model["geometries"][path[1]].get("selections", {}).get(path[3], {})
        return sel.get("kind") == M.DOMAIN and \
            G.shape_of(P.resolved(model)["geometries"][path[1]].get("source", {})) is not None
    return _pickable(model, path, M.DOMAIN) is not None


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
    gi, item = _pickable(model, path, M.DOMAIN)
    geom = model["geometries"][gi]
    name = item.get("selection", M.ALL)
    sel = geom.get("selections", {}).get(name)
    if sel is not None and sel.get("criteria", {}).get("type") == "regions" \
            and selection_uses(model, geom, name) == 1:
        sel["criteria"] = G.toggle_region(sel["criteria"], region)
        return name, False
    start = copy.deepcopy(sel["criteria"]) if sel is not None and \
        sel.get("criteria", {}).get("type") == "regions" else {"type": "regions", "regions": []}
    criteria = G.toggle_region(start, region)
    new = unique_name(_item_base_name(model, path, item), M.selection_table(model, geom))
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
    return shape if isinstance(shape, dict) and shape.get("type") in PRIMITIVES_2D else None


def shape_handles(model: dict, path: Path) -> list[tuple[str, list[float]]]:
    """Handles of the primitive at path: [(name, position)], in model values."""
    shape = primitive_node(model, path)
    if shape is None:
        return []
    r = P.resolved(model)
    s = get_at(r, path)
    if s is None or any(isinstance(v, str) for k in ("min", "max", "center") for v in s.get(k, [])) \
            or isinstance(s.get("radius"), str) \
            or any(not isinstance(v, (int, float)) for p in s.get("points", []) for v in p):
        return []                          # invalid expressions: validation reports them
    if s["type"] == "ellipse":
        if any(isinstance(v, str) for v in s.get("radii", [])) or isinstance(s.get("angle", 0.0), str):
            return []
        (cx, cy), (r1, r2) = s["center"], s["radii"]
        th = math.radians(float(s.get("angle", 0.0)))
        return [("center", [cx, cy]),
                ("axis1", [cx + r1 * math.cos(th), cy + r1 * math.sin(th)]),
                ("axis2", [cx - r2 * math.sin(th), cy + r2 * math.cos(th)])]
    if s["type"] == "polygon":
        V = [[float(x), float(y)] for x, y in s["points"]]
        n = len(V)
        mids = [[(V[i][0] + V[(i + 1) % n][0]) / 2, (V[i][1] + V[(i + 1) % n][1]) / 2] for i in range(n)]
        center = [sum(v[0] for v in V) / n, sum(v[1] for v in V) / n]
        return [(f"v{i}", v) for i, v in enumerate(V)] + [(f"m{i}", m) for i, m in enumerate(mids)] \
            + [("center", center)]
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
        raise ValueError("not a rectangle, a disk or a polygon")
    before = copy.deepcopy(model)
    notes: list[str] = []
    x, y = float(point[0]), float(point[1])
    r = get_at(P.resolved(model), path)
    try:
        if shape["type"] == "ellipse":
            (cx, cy), (r1, r2) = (float(v) for v in r["center"]), (float(v) for v in r["radii"])
            if handle == "center":
                for i, v in enumerate((x, y)):
                    if abs(v - r["center"][i]) > 1e-15 * max(1.0, abs(v)):
                        _set_number(model, path + ("center", i), v, notes)
                return notes
            dist = math.hypot(x - cx, y - cy)
            if dist <= 0:
                raise ValueError("the semi-axis would be zero")
            # the first axis turns the ellipse; the second one only sets its length
            if handle == "axis1":
                angle = math.degrees(math.atan2(y - cy, x - cx))
                _set_number(model, path + ("radii", 0), dist, notes)
            else:
                angle = math.degrees(math.atan2(y - cy, x - cx)) - 90.0
                _set_number(model, path + ("radii", 1), dist, notes)
            angle = (angle + 180.0) % 360.0 - 180.0
            if abs(angle - float(r.get("angle", 0.0))) > 1e-9:
                _set_number(model, path + ("angle",), float(f"{angle:.10g}"), notes)
            return notes
        if shape["type"] == "polygon":
            V = [list(map(float, v)) for v in r["points"]]
            if handle == "center":
                cx, cy = sum(v[0] for v in V) / len(V), sum(v[1] for v in V) / len(V)
                new = [[v[0] + x - cx, v[1] + y - cy] for v in V]
                for i, (v, old) in enumerate(zip(new, V)):
                    for k in range(2):
                        if abs(v[k] - old[k]) > 1e-15 * max(1.0, abs(old[k])):
                            _set_number(model, path + ("points", i, k), v[k], notes)
                return notes
            if handle.startswith("m"):
                i = int(handle[1:])
                V.insert(i + 1, [x, y])
                from . import geometry2d as G
                problem = G.polygon_problem(V)
                if problem:
                    raise ValueError(f"not a simple polygon ({problem})")
                shape["points"].insert(i + 1, [x, y])
                _vertex_inserted(model, path, i)
                return notes
            i = int(handle[1:])
            V[i] = [x, y]
            from . import geometry2d as G
            problem = G.polygon_problem(V)
            if problem:
                raise ValueError(f"not a simple polygon ({problem})")
            for k, v in enumerate((x, y)):
                if abs(v - r["points"][i][k]) > 1e-15 * max(1.0, abs(v)):
                    _set_number(model, path + ("points", i, k), v, notes)
        elif shape["type"] == "rectangle":
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
