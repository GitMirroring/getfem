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

"""Global parameters and user fields.

    python -m pytest tests/       (or: python tests/test_parameters.py)
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, mesher, model as M, parameters as P, runner  # noqa: E402


def load(name):
    return M.load(Path(__file__).resolve().parents[1] / "examples" / name)


def solve(model):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(model, tmp)
        assert res.ok, f"{res.error}\n{res.stderr[-1500:]}"
        return res.results["models"][model["models"][0]["name"]]   # the first model


def test_evaluation():
    assert P.evaluate("2*L + sqr(a) - pow(2, 3)", {"L": 1.5, "a": 3.0}) == 4.0
    assert abs(P.evaluate("sin(pi/2) + max(1, 2) + abs(-1)", {}) - 4.0) < 1e-15
    for bad, msg in [("L**2", "pow(a, b)"), ("2*(3", "syntax error"), ("b", "unknown parameter 'b'"),
                     ("open('f')", "unknown function"), ("x.y", "unknown parameter 'x'"), ("L.L", "two vectors"),
                     ("lambda: 1", "not allowed"), ("1/0", "division"),
                     ("log(-1)", "cannot evaluate"), ("", "expected")]:
        try:
            P.evaluate(bad, {"L": 1.0})
            raise AssertionError(f"{bad!r} accepted")
        except P.ParameterError as exc:
            assert msg in str(exc), (bad, str(exc))


def test_parameters_in_order_and_validation():
    m = load("plate_hole_tension.json")
    vals, errors = P.values(m)
    assert errors == [] and vals["a"] == 0.01 and M.validate(m) == []
    m["parameters"] += [{"name": "b", "value": "c + 1"}, {"name": "c", "value": 1},
                        {"name": "x", "value": 1}, {"name": "sqrt", "value": 1},
                        {"name": "u", "value": 1}, {"name": "L", "value": 2}]
    errors = M.validate(m)
    assert any("parameters[5].value: unknown parameter 'c'" in e for e in errors)
    assert any("parameters[7].name: 'x' is reserved for the coordinates" in e for e in errors)
    assert any("parameters[8].name: 'sqrt' is a reserved GWFL name" in e for e in errors)
    assert any("parameters[10].name: 'L' is already a parameter" in e for e in errors)
    m = load("plate_hole_tension.json")
    m["parameters"].append({"name": "u", "value": 1})
    assert any("variables[0].name: 'u' is already a parameter" in e for e in M.validate(m))
    # an expression in a geometry field is checked when resolving
    m = load("plate_hole_tension.json")
    m["geometries"][0]["source"]["shape"]["of"][1]["radius"] = "a*2 +"
    assert any(e.startswith("geometries[0].source.shape.of[1].radius: syntax error")
               for e in M.validate(m))
    m = load("plate_hole_tension.json")
    m["fields"] = [{"name": "S", "values": [{"selection": "all", "value": 1.0}]}]
    assert any("'S' is already a parameter" in e for e in M.validate(m))
    m = load("plate_hole_tension.json")
    m["models"][0]["parameters"] = [{"name": "a", "value": 1.0}]       # no shadowing
    assert any("models[0].parameters[0].name: 'a' is already a parameter" in e for e in M.validate(m))


def test_divisions_from_a_parameter():
    m = load("plate.json")
    m["parameters"] = [{"name": "n", "value": 8}]
    m["meshes"][0]["divisions"] = ["2*n", "n"]
    assert M.validate(m) == []
    m["meshes"][0]["divisions"] = ["n/3", "n"]
    assert any("integer expected" in e for e in M.validate(m))


def test_a_parameter_remeshes_only_when_used_by_the_mesh():
    m = load("plate_hole_tension.json")
    name = m["models"][0]["variables"][0]["mesh"]
    with tempfile.TemporaryDirectory() as tmp:
        assert mesher.ensure_mesh(m, name, tmp).ok
        m["parameters"][4]["value"] = 2e8                  # S: physics only
        assert isinstance(mesher.prepare_mesh(m, name, tmp), mesher.MeshResult)
        m["parameters"][2]["value"] = 0.012                 # a: radius of the hole
        assert isinstance(mesher.prepare_mesh(m, name, tmp), mesher.MeshJob)


def test_parameters_in_the_physics():
    m = load("plate_hole_tension.json")
    r1 = solve(m)
    m["parameters"][4]["value"] = "2*1e8"
    r2 = solve(m)
    f1, f2 = r1["quantities"]["applied_force"], r2["quantities"]["applied_force"]
    assert abs(f2 / f1 - 2) < 1e-10
    assert abs(f1 / (1e8 * 0.05) - 1) < 1e-3                # S times the height H


def test_user_field_used_by_a_property():
    base = load("plate.json")
    direct = load("plate.json")
    direct["materials"][0]["properties"]["k"] = 2 * base["materials"][0]["properties"]["k"]
    m = load("plate.json")
    m["fields"] = [{"name": "kfac", "values": [{"selection": "all", "value": 3.0},
                                              {"selection": "all", "value": "1 + 0*x"}]}]
    m["materials"] = []
    m["models"][0]["modules"][0]["properties"]["k"] = [{"selection": "all", "value": "?"}]
    m["models"][0]["modules"][0]["properties"]["k"][0]["value"] = \
        f"{base['materials'][0]['properties']['k']}*2*kfac"
    assert M.validate(m) == []
    assert abs(solve(m)["max_T"] - solve(direct)["max_T"]) < 1e-9   # the last value wins
    m["fields"][0]["name"] = "k"
    assert any("its property 'k' is already a field" in e for e in M.validate(m))
    m["fields"][0]["name"] = "pi"
    assert any("reserved GWFL name" in e for e in M.validate(m))


def test_user_field_is_exported():
    m = load("plate.json")
    m["fields"] = [{"name": "dist", "values": [{"selection": "all", "value": "sqrt(sqr(x)+sqr(y))"}]}]
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, res.error
        vtu = res.results["models"]["heat"]["vtu"]
        assert 'Name="dist"' in (Path(tmp) / vtu).read_text(errors="replace")


def test_rename_a_parameter_updates_its_uses():
    m = load("plate_hole_tension.json")
    i = next(k for k, p in enumerate(m["parameters"]) if p["name"] == "a")
    E.apply(m, E.Field(("parameters", i, "name"), "", "ident", op="rename_parameter"), "radius")
    shape = m["geometries"][0]["source"]["shape"]
    assert shape["of"][1]["radius"] == "radius" and m["parameters"][i]["name"] == "radius"
    assert m["meshes"][0]["h"] == "h"                                  # whole words only
    assert m["models"][0]["outputs"][0]["expr"] == "(sigma*Normal)(1)"
    assert M.validate(m) == []


def test_standalone_script():
    """The exported script meshes and solves alone, with the same results."""
    import subprocess
    m = load("plate_hole_tension.json")
    reference = solve(m)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "model.py"
        script.write_text(runner.export_script(m), encoding="utf-8")
        text = script.read_text()
        assert "import getfem_gui" not in text and "from getfem_gui" not in text
        proc = subprocess.run([sys.executable, script.name], cwd=tmp, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-2000:]
        import json
        r = json.loads((Path(tmp) / "results.json").read_text())["models"]["mechanics"]
        for k, v in reference["quantities"].items():
            assert abs(r["quantities"][k] - v) <= 1e-9 * max(1.0, abs(v)), k
        assert (Path(tmp) / r["vtu"]).exists()


def test_editing_nodes_and_actions():
    m = load("plate.json")
    root = E.tree(m)
    assert E.find_node(root, ("parameters",)) and E.find_node(root, ("fields",))
    p = E.do_action(m, ("parameters",), "add_parameter")
    f = E.do_action(m, ("fields",), "add_field")
    E.do_action(m, f, "add_field_value")
    assert M.validate(m) == []
    fields = E.fields(m, f)
    removable = [x for x in fields if x.removable]
    assert len(removable) == 2 and removable[1].remove == f + ("values", 1)
    E.remove_at(m, removable[1].remove)
    assert len(m["fields"][0]["values"]) == 1
    assert E.parse_number("L/2") == "L/2" and E.parse_number("3") == 3.0
    assert p == ("parameters",) and m["parameters"][0]["name"] == "p"
    assert E.do_action(m, p, "add_parameter") == p and len(m["parameters"]) == 2
    E.do_action(m, p, "move_down:0")
    assert [x["name"] for x in m["parameters"]] == ["p2", "p"]
    E.do_action(m, p, "remove_row:1")
    assert [x["name"] for x in m["parameters"]] == ["p2"]
    rows = E.parameter_rows(m, p)
    assert rows[0]["number"] == 1.0 and rows[0]["fields"][0].op == "rename_parameter"


# ------------------------------------------------------------------ vectors and matrices

def test_vector_and_matrix_evaluation():
    vals, errors = P.values({"parameters": [
        {"name": "L", "value": 2}, {"name": "g", "value": [0, 0, -9.81]},
        {"name": "K", "value": [[1, 0], [0, "L"]]}, {"name": "K2", "value": "[1, 2; 3, L]"},
        {"name": "C", "value": "[L/2, 1]"}, {"name": "u", "value": "K2*C"},
        {"name": "d", "value": "C.C + Norm(g) + K(2, 2) + Det(K2) + Trace(Transpose(K2)*Inv(K))"}]})
    assert errors == []
    assert P.format_value(vals["g"]) == "[0, 0, -9.81]" and P.format_value(vals["K"]) == "[1, 0; 0, 2]"
    assert P.format_value(vals["K2"]) == "[1, 2; 3, 2]" and P.format_value(vals["u"]) == "[3, 5]"
    assert abs(vals["d"] - (2 + 9.81 + 2 - 4 + (1 + 1))) < 1e-12
    assert P.describe(vals["K"]) == "a 2x2 matrix" and P.describe(vals["g"]) == "a vector of 3 components"
    assert P.scalar("C(2) + K(1, 1)", vals) == 2.0 and P.vector("2*C", vals, 2) == [2.0, 2.0]
    for bad, msg in [("C*C", "dot product u.v"), ("C[0]", "numbered from 1"), ("C(3)", "out of range"),
                     ("K(1)", "2 indices expected"), ("L(1)", "is a number"), ("sqrt(C)", "takes numbers"),
                     ("C + g", "cannot add"), ("1/C", "cannot divide"), ("g.C", "two vectors of the same size"),
                     ("[1, [2, 3]]", "a vector holds numbers"), ("[[1, 2], [3]]", "rows of the same length"),
                     ("Inv([1, 1; 1, 1])", "cannot evaluate")]:
        try:
            P.evaluate(bad, vals)
            raise AssertionError(f"{bad!r} accepted")
        except P.ParameterError as exc:
            assert msg in str(exc), (bad, str(exc))
    for expr, msg in [("C", "a number is expected"), ("K", "a 2x2 matrix")]:
        try:
            P.scalar(expr, vals)
            raise AssertionError(expr)
        except P.ParameterError as exc:
            assert msg in str(exc)


def test_vector_parameters_in_the_geometry():
    """A point given by a vector parameter, or by its components; the
    handles change the parameter."""
    m = load("plate_hole_tension.json")
    m["parameters"].append({"name": "C", "value": [0.0, 0.0]})
    hole = m["geometries"][0]["source"]["shape"]["of"][1]
    hole["center"] = "C"
    m["geometries"][0]["source"]["shape"]["of"][0]["max"] = ["L", "H + C(2)"]
    assert M.validate(m) == []
    r = P.resolved(m)["geometries"][0]["source"]["shape"]
    assert r["of"][1]["center"] == [0.0, 0.0] and r["of"][0]["max"] == [0.1, 0.05]
    path = ("geometries", 0, "source", "shape", "of", 1)
    notes = E.move_handle(m, path, "center", (0.001, 0.0))
    assert m["parameters"][-1]["value"] == [0.001, 0.0] and hole["center"] == "C"
    assert notes == ["parameter C(1) = 0.001"]
    for value, msg in [([1, 2, 3], "2 numbers expected"), ([[1, 0], [0, 1]], "a 2x2 matrix")]:
        bad = load("plate_hole_tension.json")
        bad["parameters"].append({"name": "C", "value": value})
        bad["geometries"][0]["source"]["shape"]["of"][1]["center"] = "C"
        errors = M.validate(bad)
        assert any(msg in e for e in errors), (msg, errors)
    bad = load("plate_hole_tension.json")
    bad["parameters"].append({"name": "C", "value": [1, 2]})
    bad["geometries"][0]["source"]["shape"]["of"][1]["radius"] = "C"
    bad["meshes"][0]["h"] = "C"
    errors = M.validate(bad)
    assert "geometries[0].source.shape.of[1].radius: a number is expected, 'C' is a vector of 2 components" \
        in errors and any(e.startswith("meshes[0].h: a number is expected") for e in errors)
    # the form: a vector parameter typed in the first box gives the whole vector
    f = next(x for x in E.fields(m, path) if x.key[-1] == "center")
    assert f.op == "numbers"
    E.apply(m, f, ["C(1)", "0"])
    assert hole["center"] == ["C(1)", 0.0]
    E.apply(m, f, ["C", 0.0])
    assert hole["center"] == "C"
    E.apply(m, f, "0.01, 0.02")
    assert hole["center"] == [0.01, 0.02]
    # the height of a surface uses the components
    from getfem_gui import manifold
    vals = {"C": P.evaluate([1.0, 2.0], {})}
    assert "2.0" in manifold.height_code("C(2)*x", vals)
    try:
        manifold.height_code("C*x", vals)
        raise AssertionError("a vector accepted in a height")
    except ValueError as exc:
        assert "C(1)" in str(exc)


def test_vector_and_matrix_parameters_in_the_physics():
    """GetFEM data with their sizes: components, products, the solve script."""
    m = load("plate_hole_tension.json")
    m["parameters"] += [{"name": "g", "value": "[1, 2]"}, {"name": "K", "value": "[1, 2; 3, 4]"}]
    m["models"][0]["outputs"] += [{"name": "k12", "expr": "K(1, 2)", "selection": "all"},
                                  {"name": "kg2", "expr": "(K*g)(2)", "selection": "all"},
                                  {"name": "gg", "expr": "g.g", "selection": "all"}]
    assert M.validate(m) == []
    q = solve(m)["quantities"]
    area = q["area"]
    assert abs(q["k12"] / area - 2) < 1e-10 and abs(q["kg2"] / area - 11) < 1e-10
    assert abs(q["gg"] / area - 5) < 1e-10
    from getfem_gui import codegen
    w = codegen.ScriptWriter()
    codegen.parameters(m, m["models"][0], w, "md")
    script = w.text()
    assert "add_initialized_data('K', [1.0, 3.0, 2.0, 4.0], [2, 2])  # [1, 2; 3, 4]" in script


def test_editing_vector_and_matrix_parameters():
    assert E.parse_parameter("[1, 0, 0]") == [1, 0, 0]
    assert E.parse_parameter("[[1, 0], [0, 2]]") == [[1, 0], [0, 2]]
    assert E.parse_parameter("[1, 0; 0, 2]") == "[1, 0; 0, 2]"
    assert E.parse_parameter("[L, 0]") == "[L, 0]" and E.parse_parameter("2.5") == 2.5
    m = load("plate.json")
    p = E.do_action(m, ("parameters",), "add_parameter")
    rows = E.parameter_rows(m, p)
    E.apply(m, rows[0]["fields"][1], "[1, 0; 0, 2]")
    E.apply(m, E.do_action(m, p, "add_parameter") and E.parameter_rows(m, p)[1]["fields"][1], "[3, 4]")
    rows = E.parameter_rows(m, p)
    assert m["parameters"][1]["value"] == [3, 4]
    assert P.format_value(rows[0]["number"]) == "[1, 0; 0, 2]" and P.format_value(rows[1]["number"]) == "[3, 4]"
    assert M.validate(m) == []


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok       {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAILED   {name}: {exc!r}")
    sys.exit(1 if failed else 0)
