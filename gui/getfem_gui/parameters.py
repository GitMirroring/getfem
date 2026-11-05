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


"""Global parameters: named constants usable everywhere in a model.

    "parameters": [
      {"name": "L", "value": 0.1, "description": "plate length"},
      {"name": "a", "value": "L/10", "description": "hole radius"},
      {"name": "p", "value": "1e8"}
    ]

A parameter is a number or an expression of the parameters listed before
it. They are used in two ways:

- in the numerical fields of the geometries and meshes (sizes, positions,
  radii, element sizes, divisions...), which then hold an expression such as
  "L/2". These are evaluated here, in Python, before meshing: a change of
  parameter remeshes only if the geometry or the mesh actually changes;
- in the GWFL expressions of the physics, where they are GetFEM data.

The expression syntax is therefore the part common to Python and GWFL:
numbers, + - * /, parentheses, pi and the functions of FUNCTIONS. It is
evaluated without `eval` (through the syntax tree of the expression).
"""

from __future__ import annotations

import ast
import copy
import math
import operator

FUNCTIONS = {
    "sqr": lambda a: a * a, "sqrt": math.sqrt, "pow": math.pow, "exp": math.exp,
    "log": math.log, "log10": math.log10, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "abs": abs, "min": min, "max": max,
}
CONSTANTS = {"pi": math.pi}

# Names that a parameter (or a field) cannot take: GWFL functions and keywords.
GWFL_NAMES = set(FUNCTIONS) | set(CONSTANTS) | {
    "X", "Normal", "Id", "meshdim", "element_size", "element_K", "element_B", "timestep", "t",
    "Grad", "Hess", "Div", "Test", "Test2", "Def", "Trace", "Deviator", "Sym", "Skew", "Norm",
    "Norm_sqr", "Det", "Inv", "Transpose", "Contract", "Reshape", "Index_move_last", "Swap_indices",
    "Heaviside", "sign", "pos_part", "neg_part", "erf", "erfc", "sinc", "asinh", "acosh", "atanh",
}

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class ParameterError(ValueError):
    pass


def evaluate(expr, values: dict[str, float]) -> float:
    """Value of a number or of an expression of the parameters `values`."""
    if isinstance(expr, bool):
        raise ParameterError(f"number or expression expected, got {expr!r}")
    if isinstance(expr, (int, float)):
        return float(expr)
    if not isinstance(expr, str) or not expr.strip():
        raise ParameterError("number or expression expected")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ParameterError(f"syntax error in {expr!r} (at character {exc.offset or 0})") from None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            raise ParameterError("'**' is not available (GWFL has no power operator): "
                                 "use pow(a, b) or sqr(a)")
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name):
            if node.id in values:
                return values[node.id]
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            raise ParameterError(f"unknown parameter {node.id!r}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            f = FUNCTIONS.get(node.func.id)
            if f is None:
                raise ParameterError(f"unknown function {node.func.id!r} "
                                     f"(available: {', '.join(sorted(FUNCTIONS))})")
            return f(*[ev(a) for a in node.args])
        raise ParameterError(f"not allowed in a parameter expression: {ast.unparse(node)!r}")

    try:
        value = ev(tree)
    except ParameterError:
        raise
    except ZeroDivisionError:
        raise ParameterError("division by zero") from None
    except (ValueError, OverflowError, TypeError) as exc:
        raise ParameterError(f"cannot evaluate {expr!r}: {exc}") from None
    if not math.isfinite(value):
        raise ParameterError(f"{expr!r} is not finite")
    return value


def values(model: dict) -> tuple[dict[str, float], list[str]]:
    """Values of the parameters, evaluated in order, and the errors."""
    out: dict[str, float] = {}
    errors: list[str] = []
    for i, p in enumerate(model.get("parameters", [])):
        name = p.get("name")
        try:
            v = evaluate(p.get("value"), out)
        except ParameterError as exc:
            errors.append(f"parameters[{i}].value: {exc}")
            continue
        if isinstance(name, str):
            out[name] = v
    return out, errors


# ------------------------------------------------------------------ resolution

# Numerical fields of geometries and meshes that may hold an expression
SHAPE_KEYS = ("min", "max", "center", "radius", "origin", "axis", "length", "z")
CRITERIA_KEYS = ("point", "normal", "direction", "angle", "min", "max", "center", "radius")
BOX_KEYS = ("origin", "size")
MESH_KEYS = ("h", "size")
INTEGER_KEYS = ("divisions",)


def _resolve(obj: dict, keys, where: str, vals, errors, integer=False):
    for k in keys:
        if k not in obj:
            continue
        v = obj[k]
        items = v if isinstance(v, list) else [v]
        res = []
        for j, x in enumerate(items):
            w = f"{where}.{k}[{j}]" if isinstance(v, list) else f"{where}.{k}"
            if isinstance(x, str):
                try:
                    x = evaluate(x, vals)
                except ParameterError as exc:
                    errors.append(f"{w}: {exc}")
                    return
                if integer:
                    if abs(x - round(x)) > 1e-9 * max(1.0, abs(x)):
                        errors.append(f"{w}: integer expected, got {x:g}")
                        return
                    x = int(round(x))
            res.append(x)
        obj[k] = res if isinstance(v, list) else res[0]


def _resolve_shape(shape, where, vals, errors):
    if not isinstance(shape, dict):
        return
    _resolve(shape, SHAPE_KEYS, where, vals, errors)
    for j, s in enumerate(shape.get("of", [])):
        _resolve_shape(s, f"{where}.of[{j}]", vals, errors)
    _resolve_shape(shape.get("profile"), f"{where}.profile", vals, errors)


def _resolve_geometries(geoms, meshes, prefix, vals, errors):
    for i, g in enumerate(geoms):
        src = g.get("source", {})
        where = f"{prefix}geometries[{i}].source"
        if src.get("type") == "box":
            _resolve(src, BOX_KEYS, where, vals, errors)
        elif src.get("type") == "csg":
            _resolve_shape(src.get("shape"), f"{where}.shape", vals, errors)
        for name, sel in (g.get("selections") or {}).items():
            if isinstance(sel, dict) and isinstance(sel.get("criteria"), dict):
                _resolve(sel["criteria"], CRITERIA_KEYS, f"{prefix}geometries[{i}].selections.{name}.criteria",
                         vals, errors)
    for i, mesh in enumerate(meshes):
        _resolve(mesh, MESH_KEYS, f"{prefix}meshes[{i}]", vals, errors)
        _resolve(mesh, INTEGER_KEYS, f"{prefix}meshes[{i}]", vals, errors, integer=True)


def resolve(model: dict) -> tuple[dict, list[str]]:
    """Copy of the model where the numerical fields of the geometries and
    meshes are numbers (expressions of parameters evaluated), and the errors.
    The geometries and meshes of a model also see its local parameters."""
    m = copy.deepcopy(model)
    vals, errors = values(m)
    _resolve_geometries(m.get("geometries", []), m.get("meshes", []), "", vals, errors)
    for k, md in enumerate(m.get("models", [])):
        if not isinstance(md, dict):
            continue
        local, lerr = values({"parameters": list(m.get("parameters", [])) + list(md.get("parameters", []))})
        n = len(m.get("parameters", []))
        for e in lerr:
            i = int(e.split("[", 1)[1].split("]", 1)[0])
            if i >= n:
                errors.append(f"models[{k}].parameters[{i - n}]" + e.split("]", 1)[1])
        _resolve_geometries(md.get("geometries", []), md.get("meshes", []), f"models[{k}].",
                            local, errors)
    return m, errors


def resolved(model: dict) -> dict:
    """Resolved copy of the model, leaving the unresolvable values as they are
    (for display and naming; validation reports them)."""
    return resolve(model)[0]
