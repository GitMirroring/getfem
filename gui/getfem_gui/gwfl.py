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


"""Checking the GWFL expressions of a model before meshing and solving.

Every expression written by the user (properties, sources, condition
values, terms, integrated quantities) is evaluated by GetFEM itself, in a
separate process, on a mesh of one or two elements, in the context of its
model in the solve script: same variables, imports, property names and
macros of the modules. This catches what only GetFEM knows: syntax errors, unknown
names, X(3) in 2D, Normal outside of a boundary, a scalar where a vector is
expected, a test function in a value, ...

Errors are returned in the format of model.validate ("path: message"), so
that an interface can show them on the corresponding fields.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from . import codegen, model as M, parameters as P

SCALAR, VECTOR = M.SCALAR, M.VECTOR


def expressions(model: dict) -> list[dict]:
    """The GWFL expressions of the file (numbers are not listed): [{"model":
    index of the model giving the context, "where": path, "expr": text,
    "kind": scalar|vector|term, "on": domain|boundary}]."""
    model = M.with_defaults(model)
    out: list[dict] = []

    def add(k, where, value, kind, on):
        if isinstance(value, list):              # vector given per component
            for j, c in enumerate(value):
                add(k, f"{where}[{j}]", c, SCALAR, on)
        elif isinstance(value, str) and value.strip():
            out.append({"model": k, "where": where, "expr": value, "kind": kind, "on": on})

    users: dict[str, int] = {}
    for k, md in enumerate(model["models"]):
        w = f"models[{k}]"
        geom = M.model_geometry(model, md) or {}
        table = M.selection_table(model, geom) if geom else {}
        for i, f in enumerate(md["fields"]):
            for j, v in enumerate(f.get("values", [])):
                add(k, f"{w}.fields[{i}].values[{j}].value", v.get("value"), SCALAR, M.DOMAIN)
        for i, f in enumerate(model["fields"]):
            if f in M.model_fields(model, md):
                users.setdefault(f"fields[{i}]", k)
        for i, mod in enumerate(md["modules"]):
            spec = M.MODULE_SPECS.get(mod.get("type"), {})
            mw = f"{w}.modules[{i}]"
            for p, pieces in mod.get("properties", {}).items():
                for j, pc in enumerate(pieces):
                    if "material" in pc:
                        users.setdefault(pc["material"], k)
                    add(k, f"{mw}.properties.{p}[{j}].value", pc.get("value"), SCALAR, M.DOMAIN)
            for key, kind in spec.get("values", {}).items():
                add(k, f"{mw}.{key}", mod.get(key), kind, M.DOMAIN)
            for j, src in enumerate(mod.get("sources", [])):
                add(k, f"{mw}.sources[{j}].value", src.get("value"), spec.get("source", SCALAR), M.DOMAIN)
            for j, bc in enumerate(mod.get("conditions", [])):
                for key, kind in spec.get("conditions", {}).get(bc.get("type"), {}).items():
                    add(k, f"{mw}.conditions[{j}].{key}", bc.get(key), kind, M.BOUNDARY)
        kinds = {v["name"]: M.variable_kind(v) for v in md["variables"]}
        for i, t in enumerate(md["terms"]):
            on = table.get(t.get("selection", M.ALL), {}).get("kind", M.DOMAIN)
            add(k, f"{w}.terms[{i}].expr", t.get("expr"), "term", on)
        for i, c in enumerate(md["conditions"]):
            for key, kind in M.CONDITIONS.get(c.get("type"), {}).items():
                kind = kinds.get(c.get("variable"), SCALAR) if kind == "var" else kind
                add(k, f"{w}.conditions[{i}].{key}", c.get(key), kind, M.BOUNDARY)
        for i, q in enumerate(md["outputs"]):
            on = table.get(q.get("selection", M.ALL), {}).get("kind", M.DOMAIN)
            add(k, f"{w}.outputs[{i}].expr", q.get("expr"), SCALAR, on)
    for i, f in enumerate(model["fields"]):
        for j, v in enumerate(f.get("values", [])):
            add(users.get(f"fields[{i}]", 0), f"fields[{i}].values[{j}].value", v.get("value"), SCALAR,
                M.DOMAIN)
    for i, mat in enumerate(model["materials"]):
        for p, v in (mat.get("properties") or {}).items():
            add(users.get(mat.get("name"), 0), f"materials[{i}].properties.{p}", v, SCALAR, M.DOMAIN)
    return out


def _sample(model, md, name) -> float:
    """A representative value of a property or a field: its first numeric
    value (a neutral 1 could make a macro singular, e.g. nu = 1)."""
    for mod in md["modules"]:
        for pc in mod.get("properties", {}).get(name, []):
            v = M.material(model, pc["material"])["properties"].get(name) if "material" in pc \
                and M.material(model, pc["material"]) else pc.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        default = M.MODULE_SPECS.get(mod.get("type"), {}).get("property_defaults", {}).get(name)
        if default is not None:
            return float(default)
    for f in M.model_fields(model, md):
        if f.get("name") == name:
            nums = [v["value"] for v in f.get("values", []) if isinstance(v.get("value"), (int, float))]
            return float(nums[0]) if nums else 1.0
    return 1.0


def check_script(model: dict, items: list[dict]) -> str:
    """Script evaluating each expression in the context of its model (same
    names as in the solve script: variables, imports, parameters, fields,
    properties, macros); it prints a JSON list of errors."""
    model = P.resolved(M.with_defaults(model))
    w = codegen.ScriptWriter()
    w("# GWFL check generated by getfem_gui: evaluates each expression of the file")
    w("import json, os, sys, tempfile")
    w("import numpy as np")
    w("import getfem as gf")
    w("CONTEXTS = {}")
    for k, md in enumerate(model["models"]):
        if not any(it["model"] == k for it in items):
            continue
        mesh = M.model_mesh(model, md)
        dim = int(M.model_dim(model, md) or 2)
        w("")
        w(f"# Model {md['name']!r}")
        axes = ", ".join(["np.array([0.0, 1.0])"] * dim)
        w(f"m = gf.Mesh('regular_simplices', {axes})")
        w("m.set_region(1, m.outer_faces())")
        c = codegen.Ctx(model, md, dim, mesh["name"] if mesh else "m", "md")
        deg = max([int(v.get("degree", 1)) for v in md["variables"]] + [1])
        w(f"mim = gf.MeshIm(m, {2 * deg})")
        w("md = gf.Model('real')")
        names = [(v["name"], v) for v in md["variables"]]
        for im in md["imports"]:
            src = M.model_by_name(model, im["model"])
            names.append((im["name"], M.by_name(src["variables"], im["variable"])))
        for i, (name, v) in enumerate(names):
            key = codegen.fem_key(model, v, dim)
            w(f"mf{i} = gf.MeshFem(m, {key[3]})")
            w(f"mf{i}.set_classical_fem({max(key[2], 1)})")
            c.mfs[name] = f"mf{i}"
            if any(name == vv["name"] for vv in md["variables"]):
                w(f"md.add_fem_variable({name!r}, mf{i})")
            else:
                w(f"md.add_initialized_fem_data({name!r}, mf{i}, np.zeros(mf{i}.nbdof()))")
        codegen.coordinates(w, "md", dim)
        codegen.parameters(model, md, w, "md")
        w("# fields and properties: their names, with a representative value")
        for f in M.model_fields(model, md):
            w(f"md.add_initialized_data({f['name']!r}, [{_sample(model, md, f['name'])!r}])")
        for p in M.model_properties(md):
            w(f"md.add_initialized_data({p!r}, [{_sample(model, md, p)!r}])")
        w(f"{c.m} = m")
        w(f"{c.mf_data} = gf.MeshFem(m, 1)")
        w(f"{c.mf_data}.set_classical_discontinuous_fem(0)")
        for i, mod in enumerate(md["modules"], start=1):
            codegen.MODULES[mod["type"]][0](c, mod, i, w)
        w(f"CONTEXTS[{k}] = (m, mim, md, {dim})")
    w(f"ITEMS = {json.dumps(items)}")
    w('''
def evaluate(ctx, expr, region, order):
    """Evaluate, capturing what GetFEM writes on the C++ error stream."""
    m, mim, md, dim = ctx
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as log:
        os.dup2(log.fileno(), 2)
        try:
            value, error = gf.asm('generic', mim, order, expr, region, md), None
        except Exception as exc:
            value, error = None, str(exc)
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
        log.seek(0)
        text = log.read().decode("utf-8", "replace")
    return value, error, text

def is_caret(line):
    return "^" in line and set(line.strip()) <= set("-^")

def explain(expr, error, text):
    """GetFEM prints the expression, a line with a caret under the error, and
    the message. The expression may be a macro, rather than the user's one."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and "exception caught" not in l]
    where = ""
    col = None
    i = next((k for k, l in enumerate(lines) if is_caret(l)), None)
    if i is not None:
        echoed = lines[i - 1].strip() if i > 0 else ""
        if echoed == expr.strip():
            col = lines[i].index("^")
        elif echoed.startswith("Def ") and ":=" in echoed:
            where = f"in the macro {echoed[4:echoed.index(':=')].strip()}: "
        lines = lines[i + 1:]
    msg = " ".join(l.strip() for l in lines).strip()
    if not msg and error:
        msg = error.split(" : ", 1)[-1].strip().splitlines()[-1].strip()
    msg = (msg or "invalid expression").rstrip(".")
    if col is not None:
        msg += f" (at character {col + 1})"
    return where + msg

errors = []
for it in ITEMS:
    ctx = CONTEXTS[it["model"]]
    term = it["kind"] == "term"
    value, error, text = evaluate(ctx, it["expr"], 1 if it["on"] == "boundary" else -1, 1 if term else 0)
    if error is not None:
        errors.append({"where": it["where"], "message": explain(it["expr"], error, text)})
        continue
    if term:
        continue
    size = int(np.size(value))
    DIM = ctx[3]
    expected = 1 if it["kind"] == "scalar" else DIM
    if size == 0:
        errors.append({"where": it["where"],
                       "message": "a value is expected, not an expression with a test function"})
    elif size != expected:
        what = "a scalar" if expected == 1 else f"a vector of {DIM} components"
        got = "a scalar" if size == 1 else f"{size} components"
        errors.append({"where": it["where"], "message": f"{what} is expected, got {got}"})
print("GWFL_CHECK " + json.dumps(errors))
''')
    return w.text()


def check(model: dict, python: str | None = None, timeout: float = 120) -> list[str]:
    """GWFL errors of a model, as "path: message" (empty if all is well).

    The model must pass model.validate first. Raises RuntimeError if the
    check itself cannot run (e.g. GetFEM cannot be imported)."""
    job = prepare(model, python)
    if job is None:
        return []
    proc = subprocess.run(job["command"], cwd=job["cwd"], capture_output=True, text=True,
                          timeout=timeout)
    try:
        return parse_output(proc.stdout, proc.stderr)
    finally:
        Path(job["script"]).unlink(missing_ok=True)


def prepare(model: dict, python: str | None = None, folder: str | Path | None = None) -> dict | None:
    """Write the check script; return {"command", "cwd", "script"}, or None if
    there is nothing to check. Lets an interface run the process itself."""
    items = expressions(model)
    if not items:
        return None
    folder = Path(folder) if folder else Path(tempfile.gettempdir())
    folder.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="gwfl_check_", suffix=".py", dir=folder)
    with open(fd, "w", encoding="utf-8") as f:
        f.write(check_script(model, items))
    return {"command": [python or sys.executable, name], "cwd": str(folder), "script": name}


def parse_output(stdout: str, stderr: str) -> list[str]:
    for line in stdout.splitlines():
        if line.startswith("GWFL_CHECK "):
            return [f"{e['where']}: {e['message']}" for e in json.loads(line[len("GWFL_CHECK "):])]
    from .mesher import error_line
    raise RuntimeError(f"the GWFL check could not run: {error_line(stderr) or 'no output'}")
