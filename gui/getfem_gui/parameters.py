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


"""Global parameters: named constants usable everywhere in a model.

    "parameters": [
      {"name": "L", "value": 0.1, "description": "plate length"},
      {"name": "a", "value": "L/10", "description": "hole radius"},
      {"name": "p", "value": "1e8"}
    ]

A parameter is a number, a vector or a matrix, given by a number or an
expression of the parameters listed before it:

    {"name": "g", "value": [0, 0, -9.81]}             a vector
    {"name": "C", "value": "[L/2, H/2]"}              an expression
    {"name": "K", "value": "[1, 0; 0, k2]"}           a matrix (GWFL syntax, by rows)
    {"name": "K2", "value": [[1, 0], [0, "k2"]]}      a matrix in JSON: a list of rows

In a text, a matrix is written [a, b; c, d] only: GWFL reads [[a, b], [c, d]]
as a list of columns, so that this form is refused.

They are used in two ways:

- in the numerical fields of the geometries and meshes (sizes, positions,
  radii, element sizes, divisions...), which then hold an expression such as
  "L/2" or "C(1)", or a vector parameter where a point or a direction is
  expected ("center": "C"). These are evaluated here, in Python, before
  meshing: a change of parameter remeshes only if the geometry or the mesh
  actually changes;
- in the GWFL expressions of the physics, where they are GetFEM data
  (vectors and matrices with their sizes: "K*Grad_T", "g(3)").

The expression syntax is therefore the part common to Python and GWFL:
numbers, + - * /, parentheses, pi, the functions of FUNCTIONS, vectors
[a, b, c], matrices [a, b; c, d] (rows separated by ";"), the components
of a vector or a matrix C(i), K(i, j) (numbered from 1, as in GWFL), the
dot product of vectors u.v, the matrix-vector and matrix-matrix products
K*u, and the functions of TENSOR_FUNCTIONS. It is evaluated without `eval`
(through the syntax tree of the expression).
"""

from __future__ import annotations

import ast
import copy
import math
import operator

import numpy as np

FUNCTIONS = {
    "sqr": lambda a: a * a, "sqrt": math.sqrt, "pow": math.pow, "exp": math.exp,
    "log": math.log, "log10": math.log10, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "abs": abs, "min": min, "max": max,
}
CONSTANTS = {"pi": math.pi}
TENSOR_FUNCTIONS = {
    "Norm": lambda a: float(np.linalg.norm(a)), "Norm_sqr": lambda a: float(np.sum(a * a)),
    "Trace": lambda a: float(np.trace(a)), "Det": lambda a: float(np.linalg.det(a)),
    "Inv": np.linalg.inv, "Transpose": lambda a: a.T, "Id": lambda n: np.eye(int(n)),
}

# Names that a parameter (or a field) cannot take: GWFL functions and keywords.
GWFL_NAMES = set(FUNCTIONS) | set(CONSTANTS) | set(TENSOR_FUNCTIONS) | {
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


def shape_of(value) -> tuple:
    """() for a number, (n,) for a vector, (m, n) for a matrix."""
    return np.shape(value)


def describe(value) -> str:
    """'a number', 'a vector of 3 components', 'a 2x2 matrix'."""
    sh = shape_of(value)
    return "a number" if not sh else f"a vector of {sh[0]} components" if len(sh) == 1 \
        else f"a {sh[0]}x{sh[1]} matrix"


def _semicolons(text: str) -> str:
    """The GWFL matrices [a, b; c, d] written as lists of rows [[a, b], [c, d]]."""
    if ";" not in text:
        return text
    out, i = [], 0
    while i < len(text):
        if text[i] != "[":
            out.append(text[i])
            i += 1
            continue
        depth, j = 0, i
        while j < len(text):
            depth += {"[": 1, "]": -1}.get(text[j], 0)
            if depth == 0:
                break
            j += 1
        inner = _semicolons(text[i + 1:j])
        rows, depth, cur = [], 0, ""
        for ch in inner:
            depth += {"[": 1, "(": 1, "]": -1, ")": -1}.get(ch, 0)
            if ch == ";" and depth == 0:
                rows.append(cur)
                cur = ""
            else:
                cur += ch
        rows.append(cur)
        out.append("[" + ", ".join(f"[{r.strip()}]" for r in rows) + "]" if len(rows) > 1 else f"[{inner}]")
        i = j + 1
    return "".join(out)


def _array(items: list):
    """A vector of numbers, or a matrix from its rows."""
    if all(not shape_of(x) for x in items):
        return np.array([float(x) for x in items])
    if all(len(shape_of(x)) == 1 for x in items) and len({len(x) for x in items}) == 1:
        return np.array([[float(c) for c in x] for x in items])
    raise ParameterError("a vector holds numbers, a matrix rows of the same length")


def _mul(a, b):
    sa, sb = shape_of(a), shape_of(b)
    if not sa or not sb:
        return a * b
    if len(sa) == 2 and sa[1] == sb[0]:
        return a @ b
    if len(sa) == 1 and len(sb) == 1:
        raise ParameterError("the product of two vectors: use the dot product u.v")
    raise ParameterError(f"cannot multiply {describe(a)} by {describe(b)}")


def _add(op):
    def f(a, b):
        if shape_of(a) != shape_of(b):
            raise ParameterError(f"cannot add or subtract {describe(a)} and {describe(b)}")
        return op(a, b)
    return f


def _div(a, b):
    if shape_of(b):
        raise ParameterError(f"cannot divide by {describe(b)}")
    return a / b


_TBINOPS = {ast.Add: _add(operator.add), ast.Sub: _add(operator.sub), ast.Mult: _mul, ast.Div: _div}


def evaluate(expr, values: dict):
    """Value of a number, of a list (a vector, or a matrix by rows) or of an
    expression of the parameters `values`: a float or a numpy array."""
    if isinstance(expr, bool):
        raise ParameterError(f"number or expression expected, got {expr!r}")
    if isinstance(expr, (int, float)):
        return float(expr)
    if isinstance(expr, list):
        if not expr:
            raise ParameterError("empty vector")
        return _check_finite(_array([evaluate(x, values) for x in expr]), expr)
    if not isinstance(expr, str) or not expr.strip():
        raise ParameterError("number or expression expected")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.List) and any(isinstance(e, ast.List) for e in node.elts):
                # in GWFL, [[a, b], [c, d]] lists the columns: ambiguous, so refused
                raise ParameterError(f"{expr!r}: write a matrix [a, b; c, d] (rows separated by ';')")
    except SyntaxError:
        pass
    try:
        tree = ast.parse(_semicolons(expr.strip()), mode="eval")
    except SyntaxError as exc:
        raise ParameterError(f"syntax error in {expr!r} (at character {exc.offset or 0})") from None

    def index(value, args, name):
        sh = shape_of(value)
        if not sh:
            raise ParameterError(f"{name} is a number: it has no components")
        if len(args) != len(sh):
            raise ParameterError(f"{name} is {describe(value)}: {len(sh)} indices expected")
        idx = []
        for a, n in zip(args, sh):
            i = ev(a)
            if shape_of(i) or abs(i - round(i)) > 1e-12 or not 1 <= round(i) <= n:
                raise ParameterError(f"index of {name} out of range (1 to {n}, numbered from 1)")
            idx.append(int(round(i)) - 1)
        return float(value[tuple(idx)])

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.List):
            if not node.elts:
                raise ParameterError("empty vector")
            return _array([ev(e) for e in node.elts])
        if isinstance(node, ast.BinOp) and type(node.op) in _TBINOPS:
            return _TBINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            raise ParameterError("'**' is not available (GWFL has no power operator): "
                                 "use pow(a, b) or sqr(a)")
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Attribute):          # u.v: the dot product of two vectors
            a, b = ev(node.value), ev(ast.Name(node.attr, ast.Load()))
            if len(shape_of(a)) != 1 or shape_of(a) != shape_of(b):
                raise ParameterError(f"the dot product needs two vectors of the same size, "
                                     f"got {describe(a)} and {describe(b)}")
            return float(a @ b)
        if isinstance(node, ast.Subscript):
            raise ParameterError(f"{ast.unparse(node)!r}: the components are written C(1), "
                                 "K(1, 2), numbered from 1")
        if isinstance(node, ast.Name):
            if node.id in values:
                return values[node.id]
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            raise ParameterError(f"unknown parameter {node.id!r}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name in values:
                return index(values[name], node.args, name)
            if name in TENSOR_FUNCTIONS:
                return TENSOR_FUNCTIONS[name](*[ev(a) for a in node.args])
            f = FUNCTIONS.get(name)
            if f is None:
                raise ParameterError(f"unknown function {name!r} "
                                     f"(available: {', '.join(sorted(set(FUNCTIONS) | set(TENSOR_FUNCTIONS)))})")
            args = [ev(a) for a in node.args]
            for a in args:
                if shape_of(a):
                    raise ParameterError(f"{name} takes numbers, got {describe(a)}")
            return f(*args)
        raise ParameterError(f"not allowed in a parameter expression: {ast.unparse(node)!r}")

    try:
        value = ev(tree)
    except ParameterError:
        raise
    except ZeroDivisionError:
        raise ParameterError("division by zero") from None
    except np.linalg.LinAlgError as exc:
        raise ParameterError(f"cannot evaluate {expr!r}: {exc}") from None
    except (ValueError, OverflowError, TypeError) as exc:
        raise ParameterError(f"cannot evaluate {expr!r}: {exc}") from None
    return _check_finite(value, expr)


def _check_finite(value, expr):
    if not np.all(np.isfinite(value)):
        raise ParameterError(f"{expr!r} is not finite")
    if isinstance(value, np.ndarray) and value.ndim > 2:
        raise ParameterError(f"{expr!r}: only numbers, vectors and matrices")
    return value


def scalar(expr, values: dict) -> float:
    """The value of a number or an expression that must be a number."""
    v = evaluate(expr, values)
    if shape_of(v):
        raise ParameterError(f"a number is expected, {_text(expr)} is {describe(v)}")
    return float(v)


def vector(expr, values: dict, size: int | None = None) -> list[float]:
    """The components of a vector expression (of `size` components)."""
    v = evaluate(expr, values)
    sh = shape_of(v)
    if len(sh) != 1 or (size is not None and sh[0] != size):
        what = "a vector" if size is None else f"a vector of {size} components"
        raise ParameterError(f"{what} is expected, {_text(expr)} is {describe(v)}")
    return [float(x) for x in v]


def _text(expr) -> str:
    return repr(expr) if isinstance(expr, str) else "the value"


def values(model: dict) -> tuple[dict, list[str]]:
    """Values of the parameters (floats, or numpy arrays for the vectors and
    matrices), evaluated in order, and the errors."""
    out: dict = {}
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


def format_value(v) -> str:
    """A value as text: 0.5, [1, 0, 0], [1, 0; 0, 2] (GWFL syntax)."""
    from .editing import format_number
    sh = shape_of(v)
    if not sh:
        return format_number(float(v))
    if len(sh) == 1:
        return "[" + ", ".join(format_number(float(x)) for x in v) + "]"
    return "[" + "; ".join(", ".join(format_number(float(x)) for x in row) for row in v) + "]"


# ------------------------------------------------------------------ resolution

# Numerical fields of geometries and meshes that may hold an expression
SHAPE_KEYS = ("min", "max", "center", "radius", "radii", "angle", "origin", "axis", "length", "z",
              "translate", "step")
CRITERIA_KEYS = ("point", "normal", "direction", "angle", "min", "max", "center", "radius")
BOX_KEYS = ("origin", "size")
MESH_KEYS = ("h", "size")
INTEGER_KEYS = ("divisions",)


SCALAR_KEYS = ("radius", "length", "h")


def _resolve(obj: dict, keys, where: str, vals, errors, integer=False, scalars=SCALAR_KEYS):
    """The values of the keys evaluated: each component of a list must be a
    number; a whole value may be a vector expression (a point given by a
    vector parameter), which becomes the list of its components."""
    for k in keys:
        if k not in obj:
            continue
        v = obj[k]
        if isinstance(v, str):
            try:
                x = evaluate(v, vals)
            except ParameterError as exc:
                errors.append(f"{where}.{k}: {exc}")
                return
            if len(shape_of(x)) > 1 or (shape_of(x) and k in scalars):
                what = "a number" if k in scalars else "a number or a vector"
                errors.append(f"{where}.{k}: {what} is expected, {v!r} is {describe(x)}")
                return
            v = [float(c) for c in x] if shape_of(x) else float(x)
        items = v if isinstance(v, list) else [v]
        res = []
        for j, x in enumerate(items):
            w = f"{where}.{k}[{j}]" if isinstance(v, list) else f"{where}.{k}"
            if isinstance(x, str):
                try:
                    x = scalar(x, vals)
                except ParameterError as exc:
                    errors.append(f"{w}: {exc}")
                    return
            if integer and isinstance(x, (int, float)):
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
    _resolve(shape, ("count",), where, vals, errors, integer=True, scalars=("count",))
    if shape.get("constraints") and isinstance(shape.get("points"), list):
        # the coordinates given by expressions stay where they are (sketch.py)
        shape["_fixed"] = [[True, True] if isinstance(v, str) else
                           [isinstance(v, list) and len(v) > k and isinstance(v[k], str) for k in (0, 1)]
                           for v in shape["points"]]
        for j, c in enumerate(shape["constraints"]):
            if isinstance(c, dict):
                _resolve(c, ("value", "at"), f"{where}.constraints[{j}]", vals, errors, scalars=("value",))
    for j, f in enumerate(shape.get("fillets") or []):
        if isinstance(f, dict):
            _resolve(f, ("radius",), f"{where}.fillets[{j}]", vals, errors, scalars=("radius",))
    for j, sp in enumerate(shape.get("splines") or []):       # the points of the curved sides
        if isinstance(sp, dict) and isinstance(sp.get("points"), list):
            for i, v in enumerate(sp["points"]):
                if isinstance(v, (list, str)):
                    holder = {"v": v}
                    _resolve(holder, ("v",), f"{where}.splines[{j}].points[{i}]", vals, errors)
                    sp["points"][i] = holder["v"]
    if isinstance(shape.get("points"), list):
        for j, v in enumerate(shape["points"]):
            if isinstance(v, (list, str)):
                holder = {"v": v}
                _resolve(holder, ("v",), f"{where}.points[{j}]", vals, errors)
                shape["points"][j] = holder["v"]
    if isinstance(shape.get("axes"), list):          # the work plane of an extrusion or a revolution
        for j, v in enumerate(shape["axes"]):
            if isinstance(v, (list, str)):
                holder = {"v": v}
                _resolve(holder, ("v",), f"{where}.axes[{j}]", vals, errors)
                shape["axes"][j] = holder["v"]
    for j, s in enumerate(shape.get("of", [])):
        _resolve_shape(s, f"{where}.of[{j}]", vals, errors)
    _resolve_shape(shape.get("profile"), f"{where}.profile", vals, errors)


def _solve_sketches(shape, dim, where, errors):
    """The constraints of the polygons (sketch.py), solved in each 2D tree:
    the tree itself in 2D, the profiles of the extrusions and revolutions in
    3D."""
    from . import sketch
    if not isinstance(shape, dict):
        return
    if dim == 2:
        errors += [f"{where}{e}" for e in sketch.solve_tree(shape)]
        return
    if isinstance(shape.get("profile"), dict):
        _solve_sketches(shape["profile"], 2, f"{where}.profile", errors)
    for j, s in enumerate(shape.get("of", []) or []):
        _solve_sketches(s, 3, f"{where}.of[{j}]", errors)


def _resolve_geometries(geoms, meshes, prefix, vals, errors):
    for i, g in enumerate(geoms):
        src = g.get("source", {})
        where = f"{prefix}geometries[{i}].source"
        if src.get("type") == "box":
            _resolve(src, BOX_KEYS, where, vals, errors)
        elif src.get("type") == "csg":
            _resolve_shape(src.get("shape"), f"{where}.shape", vals, errors)
            _solve_sketches(src.get("shape"), 3 if src.get("dim") == 3 else 2, f"{where}.shape", errors)
        elif src.get("type") == "surface":
            _resolve_shape(src.get("shape"), f"{where}.shape", vals, errors)
            _solve_sketches(src.get("shape"), 2, f"{where}.shape", errors)
            _resolve(src, ("origin",), where, vals, errors)
            for j, ax in enumerate(src.get("axes") or []):
                if isinstance(ax, (list, str)):
                    holder = {"v": ax}
                    _resolve(holder, ("v",), f"{where}.axes[{j}]", vals, errors)
                    src["axes"][j] = holder["v"]
        elif src.get("type") in ("curve", "swept_surface"):
            _resolve(src, ("z", "angle"), where, vals, errors)
            for key in ("points", "through"):
                for j, pt in enumerate(src.get(key) or []):
                    if isinstance(pt, (list, str)):
                        holder = {"v": pt}
                        _resolve(holder, ("v",), f"{where}.{key}[{j}]", vals, errors)
                        src[key][j] = holder["v"]
        for name, sel in (g.get("selections") or {}).items():
            if isinstance(sel, dict) and isinstance(sel.get("criteria"), dict):
                _resolve(sel["criteria"], CRITERIA_KEYS, f"{prefix}geometries[{i}].selections.{name}.criteria",
                         vals, errors)
    for i, mesh in enumerate(meshes):
        _resolve(mesh, MESH_KEYS, f"{prefix}meshes[{i}]", vals, errors, scalars=MESH_KEYS)
        _resolve(mesh, INTEGER_KEYS, f"{prefix}meshes[{i}]", vals, errors, integer=True)
        _resolve(mesh, ("growth",), f"{prefix}meshes[{i}]", vals, errors, scalars=("growth",))
        for j, rule in enumerate(mesh.get("refine") or []):       # the refinements
            if isinstance(rule, dict):
                _resolve(rule, ("times",), f"{prefix}meshes[{i}].refine[{j}]", vals, errors, integer=True,
                         scalars=("times",))
        for j, rule in enumerate(mesh.get("sizes") or []):        # the local sizes
            if isinstance(rule, dict):
                _resolve(rule, ("size",), f"{prefix}meshes[{i}].sizes[{j}]", vals, errors, scalars=("size",))


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
