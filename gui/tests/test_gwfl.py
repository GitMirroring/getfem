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

"""Checking the GWFL expressions of a model (requires GetFEM).

    python -m pytest tests/       (or: python tests/test_gwfl.py)
"""

import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import gwfl, model as M, runner  # noqa: E402

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.json"))


def load(name):
    return M.load(Path(__file__).resolve().parents[1] / "examples" / name)


def by_path(errors):
    return {e.partition(": ")[0]: e.partition(": ")[2] for e in errors}


def test_examples_are_valid():
    for ex in EXAMPLES:
        assert gwfl.check(M.load(ex)) == [], ex.name


def test_numbers_only_need_no_process():
    m = M.default_model()
    assert gwfl.expressions(m) == [] and gwfl.prepare(m) is None and gwfl.check(m) == []


def test_errors_are_located_and_explained():
    m = load("plate_hole_tension.json")
    m["materials"][0]["properties"]["E"] = "2e11*(1+X(1)"
    m["materials"][0]["properties"]["rho"] = "foo*2"
    m["physics"]["sources"] = [{"selection": "all", "value": "[0, -9.81*rho, 1]"}]
    m["physics"]["boundary_conditions"][2]["value"] = ["1e8*Normal(1)", "X(3)"]
    m["outputs"]["quantities"] += [
        {"name": "n", "expr": "Normal(1)", "selection": "all"},
        {"name": "t", "expr": "Test_u(1)", "selection": "all"},
        {"name": "v", "expr": "von_mises*sigma_zz + Norm(u)", "selection": "all"},
        {"name": "s", "expr": "Grad_u", "selection": "all"},
    ]
    assert M.validate(m) == []
    e = by_path(gwfl.check(m))
    assert e["materials[0].properties.E"] == "Unbalanced parenthesis (at character 12)"
    assert e["materials[0].properties.rho"].startswith('Unknown variable, function, operator or data "foo"')
    assert e["physics.sources[0].value"] == "a vector of 2 components is expected, got 3 components"
    assert "Index for X" in e["physics.boundary_conditions[2].value[1]"]
    assert "physics.boundary_conditions[2].value[0]" not in e     # Normal is fine on a boundary
    assert "normal" in e["outputs.quantities[3].expr"]              # but not in the domain
    assert "test function" in e["outputs.quantities[4].expr"]
    assert "outputs.quantities[5].expr" not in e                    # macros and the unknown are known
    assert e["outputs.quantities[6].expr"] == "a scalar is expected, got 4 components"
    assert len(e) == 7


def test_error_inside_a_macro():
    m = load("plate_hole_tension.json")
    m["materials"][0]["properties"]["nu"] = 1.0     # plane stress: 1 - nu^2 = 0
    e = by_path(gwfl.check(m))
    assert e and all(v.startswith("in the macro lambda_L:") for v in e.values())


def test_heat_expressions_and_3d():
    m = load("plate.json")
    m["geometries"][0]["source"] = {"type": "box", "origin": [0, 0, 0], "size": [1, 1, 1]}
    m["meshes"][0]["divisions"] = [2, 2, 2]
    m["materials"][0]["properties"]["k"] = "1 + X(3)"
    m["physics"]["boundary_conditions"][0]["value"] = "20 + q(1)"
    m["outputs"]["quantities"] = [{"name": "f", "expr": "q.Normal", "selection": "xmin"}]
    assert M.validate(m) == [] and gwfl.check(m) == []


def test_coordinates_x_y_z():
    m = load("plate.json")                       # 2D
    m["materials"][0]["properties"]["k"] = "50*(1 + 0.1*x*y)"
    m["outputs"]["quantities"] = [{"name": "a", "expr": "x + X(2)", "selection": "all"},
                                  {"name": "b", "expr": "z", "selection": "all"}]
    e = by_path(gwfl.check(m))
    assert list(e) == ["outputs.quantities[1].expr"] and '"z"' in e["outputs.quantities[1].expr"]
    # reserved names
    m = load("plate.json")
    m["materials"][0]["properties"]["y"] = 1.0
    m["physics"]["variable"] = "x"
    errors = M.validate(m)
    assert any("'y' is reserved for the coordinates" in e for e in errors)
    assert any("physics.variable: 'x' is reserved" in e for e in errors)


def test_coordinates_give_the_same_solution_as_X():
    results = []
    for value in ("20 + 100*X(1)*X(2)", "20 + 100*x*y"):
        m = load("plate.json")
        m["physics"]["boundary_conditions"][0]["value"] = value
        with tempfile.TemporaryDirectory() as tmp:
            res = runner.run(m, tmp)
            assert res.ok, res.error
            results.append(res.results)
    assert results[0]["quantities"] == results[1]["quantities"]
    assert results[0]["max_T"] == results[1]["max_T"]


def test_run_refuses_an_invalid_expression():
    m = load("plate.json")
    m["materials"][0]["properties"]["k"] = "1+"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            runner.run(m, tmp)
            raise AssertionError("not refused")
        except ValueError as exc:
            assert "materials[0].properties.k" in str(exc)
        assert not (Path(tmp) / "meshes").exists()                   # refused before meshing


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
