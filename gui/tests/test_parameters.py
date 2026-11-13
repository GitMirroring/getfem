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
                     ("[1, [2, 3]]", "write a matrix [a, b; c, d]"), ("[[1, 2], [3, 4]]", "rows separated by"),
                     ("[1, 2; 3]", "rows of the same length"),
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
    assert E.parse_parameter("[[1, 0], [0, 2]]") == "[[1, 0], [0, 2]]"         # refused when evaluated
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


def test_anisotropic_conduction_and_tensor_fields():
    """A conductivity tensor: k Id gives the isotropic solution; the fibres
    of the example carry more heat along them; vector and matrix fields."""
    m = load("anisotropic_plate.json")
    assert M.validate(m) == []
    q = solve(m)["quantities"]
    assert abs(q["outflow"] / q["injected_power"] - 1) < 1e-3
    assert q["flux_along_fibres"] > 3 * q["flux_across_fibres"]
    iso = load("anisotropic_plate.json")
    iso["parameters"][1]["value"] = 10.0                                   # k2 = k1: k1 Id
    aniso = solve(iso)["quantities"]
    del iso["models"][0]["modules"][0]["conductivity"]
    iso["models"][0]["modules"][0]["properties"]["k"] = [{"selection": "all", "value": 10.0}]
    plain = solve(iso)["quantities"]
    assert abs(aniso["outflow"] - plain["outflow"]) < 1e-9 * abs(plain["outflow"])
    assert abs(aniso["flux_along_fibres"] - plain["flux_along_fibres"]) < 1e-8 * plain["flux_along_fibres"]
    # a number for a tensor is that number times the identity; a list of rows; a matrix field
    m = load("anisotropic_plate.json")
    mod = m["models"][0]["modules"][0]
    mod["properties"]["k"] = [{"selection": "all", "value": [[10.0, 0.0], [0.0, 10.0]]}]
    m["models"][0]["fields"].append({"name": "Kf", "kind": "matrix", "values": [{"selection": "all", "value": 2.0}]})
    m["models"][0]["outputs"].append({"name": "kf12", "expr": "Kf(1, 1) + Kf(1, 2)", "selection": "all"})
    q = solve(m)["quantities"]
    assert abs(q["outflow"] - plain["outflow"]) < 1e-9 * abs(plain["outflow"])
    assert abs(q["kf12"] - 2 * 0.01) < 1e-12
    # errors
    bad = load("anisotropic_plate.json")
    bad["models"][0]["modules"][0]["properties"]["k"] = [{"selection": "all", "value": [[1, 0, 0], [0, 1, 0]]}]
    bad["models"][0]["fields"][0]["kind"] = "tensor"
    errors = M.validate(bad)
    assert any("properties.k[0].value: a 2x2 matrix expected" in e for e in errors), errors
    assert any("fields[0].kind: scalar, vector, matrix expected" in e for e in errors)
    bad = load("anisotropic_plate.json")
    del bad["models"][0]["modules"][0]["conductivity"]
    bad["models"][0]["modules"][0]["properties"]["k"] = [{"selection": "all", "value": [[1, 0], [0, 1]]}]
    assert any("anisotropic option" in e for e in M.validate(bad))
    bad = load("anisotropic_plate.json")
    bad["materials"] = [{"name": "composite", "properties": {"k": [[10, 0], [0, 1]]}}]
    bad["models"][0]["modules"][0]["properties"]["k"] = [{"selection": "all", "material": "composite"}]
    assert M.validate(bad) == []
    del bad["models"][0]["modules"][0]["conductivity"]
    assert any("is a matrix: the module needs its anisotropic option" in e for e in M.validate(bad))


def test_tensor_values_checked_by_getfem():
    from getfem_gui import gwfl
    m = load("anisotropic_plate.json")
    m["models"][0]["modules"][0]["properties"]["k"] = [{"selection": "all", "value": "d"}]
    m["models"][0]["fields"][0]["values"][0]["value"] = "K"
    errors = gwfl.check(m)
    assert "models[0].modules[0].properties.k[0].value: a 2x2 matrix is expected, got 2 components" in errors
    assert "models[0].fields[0].values[0].value: a vector of 2 components is expected, got 4 components" in errors


def test_editing_tensor_fields():
    m = load("anisotropic_plate.json")
    path = ("models", 0, "fields", 0)
    fields = {f.key[-1]: f for f in E.fields(m, path)}
    assert fields["kind"].op == "field_kind" and fields["value"].kind == "vector" and fields["value"].size == 2
    E.apply(m, fields["kind"], "matrix")
    assert m["models"][0]["fields"][0]["kind"] == "matrix"
    value = next(f for f in E.fields(m, path) if f.key[-1] == "value")
    assert value.kind == "value" and "by rows" in value.help
    E.apply(m, value, "[1, 0; 0, 2]")
    E.apply(m, next(f for f in E.fields(m, path) if f.key[-1] == "kind"), "scalar")
    assert "kind" not in m["models"][0]["fields"][0]
    assert E.format_value([[1, 0], [0, 2.5]]) == "[1, 0; 0, 2.5]"
    piece = next(f for f in E.fields(m, ("models", 0, "modules", 0, "properties", "k", 0)) if f.key[-1] == "value")
    assert "by rows" in piece.help
    opt = next(f for f in E.fields(m, ("models", 0, "modules", 0)) if f.key[-1] == "conductivity")
    assert [v for v, _ in opt.choices] == ["isotropic", "anisotropic"]


def test_vector_parameters_for_vertices_and_curve_points():
    m = load("l_bracket.json")
    m["parameters"].append({"name": "V", "value": [0.0, 0.0]})
    pts_path = ("geometries", 0, "source", "shape", "of", 0, "points")
    assert E.get_at(m, pts_path)[0] == [0, 0]
    cell = lambda i, j: E.Field(pts_path + (i, j), "v", "value", size=2, op="point_component")  # noqa: E731
    E.apply(m, cell(0, 0), "V")
    assert E.get_at(m, pts_path)[0] == "V" and M.validate(m) == []
    assert E.vertex_rows(m, pts_path)[0] == ("V", "", 0.0, 0.0)
    assert E.move_handle(m, pts_path[:-1], "v0", (0.001, 0.002)) == ["parameter V(1) = 0.001",
                                                                       "parameter V(2) = 0.002"]
    E.do_action(m, pts_path[:-1], "insert_vertex:0")
    assert E.get_at(m, pts_path)[0] == "V" and M.validate(m) == []
    E.apply(m, cell(0, 1), 0.5)                        # a number: the components again
    assert E.get_at(m, pts_path)[0] == [0.001, 0.5]
    # the points of a curve, and the point an arc goes through
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "curve", "dim": 2, "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                                    "through": [None, None], "closed": False}      # (given by points)
    m["parameters"] = [{"name": "P", "value": [1.0, 0.0]}, {"name": "Q", "value": "[1.2, 0.5]"}]
    E.apply(m, E.Field(("geometries", 0, "source", "points", 1, 0), "p", "value", size=2,
                       op="point_component"), "P")
    E.apply(m, E.Field(("geometries", 0, "source", "through", 1), "a", "text", op="chain_through"), "Q")
    src = m["geometries"][0]["source"]
    assert src["points"][1] == "P" and src["through"][1] == "Q"
    assert E.chain_rows(m, 0)[1][1] == ("P", "", "Q")
    r = P.resolved(m)["geometries"][0]["source"]
    assert r["points"][1] == [1.0, 0.0] and r["through"][1] == [1.2, 0.5]


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
