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

"""Several modules and several models: thermal expansion (monolithic or
through an import), global solvers (sequence, staggered), user terms and
generic conditions, local parameters.

    python -m pytest tests/
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import model as M, runner  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
E, NU, ALPHA, DT = 70e9, 0.33, 23e-6, 100.0


def run(model):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(model, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-3000:]}"
        return res.results


def monolithic(dim=2, plane="stress", uniform=True):
    """One model: heat conduction, elasticity, thermal expansion. uniform: T
    imposed equal to T_ref + DT on the whole boundary (free expansion, with
    rollers); otherwise the thermo-elastic plate of the example."""
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "box", "origin": [0.0] * dim, "size": [1.0] * dim}
    m["meshes"][0]["divisions"] = [4] * dim
    md = M.new_model("coupled", "mesh", "heat_conduction", dim)
    for v in M.new_model("x", "mesh", "linear_elasticity", dim)["variables"]:
        md["variables"].append(v)
    md["modules"].append(M.new_module("linear_elasticity", dim))
    md["modules"].append(M.new_module("thermal_expansion", dim))
    heat, elas, exp = md["modules"]
    elas["properties"] = {"E": [{"selection": "all", "value": E}], "nu": [{"selection": "all", "value": NU}]}
    exp["properties"] = {"alpha": [{"selection": "all", "value": ALPHA}]}
    exp["T_ref"] = 20.0
    if dim == 2:
        elas["plane"] = plane
    heat["conditions"] = [{"type": "temperature", "selection": "boundary", "value": 20.0 + DT}]
    elas["conditions"] = [{"type": "roller", "selection": s} for s in ("xmin", "ymin", "zmin")[:dim]]
    md["outputs"] = [{"name": "ux", "expr": "u(1)", "selection": "xmax"},
                     {"name": "vm", "expr": "von_mises", "selection": "all"}]
    m["models"] = [md]
    return m


def test_free_thermal_expansion():
    """Uniform heating with rollers: no stress in plane stress and 3D, the
    strain is alpha DT; in plane strain the in-plane strain is (1+nu) alpha DT
    and sigma_zz = -E alpha DT."""
    for dim, plane, strain, vm in ((2, "stress", ALPHA * DT, 0.0),
                                   (2, "strain", (1 + NU) * ALPHA * DT, E * ALPHA * DT),
                                   (3, None, ALPHA * DT, 0.0)):
        r = run(monolithic(dim, plane))["models"]["coupled"]["quantities"]
        assert abs(r["ux"] - strain) < 1e-9 * strain, (dim, plane, r)    # faces of area 1
        assert abs(r["vm"] - vm) < 1e-6 * E * ALPHA * DT, (dim, plane, r)


def test_monolithic_and_sequence_agree():
    """The thermo-elastic plate of the example (two models, a sequence) and
    the same problem in one model give the same displacement."""
    two = M.load(EXAMPLES / "thermoelastic_plate.json")
    one = copy.deepcopy(two)
    thermal, mech = one["models"]
    mech.pop("imports")
    thermal["variables"] += mech["variables"]
    thermal["modules"] += mech["modules"]
    thermal["outputs"] += mech["outputs"]
    thermal["name"] = "coupled"
    one["models"] = [thermal]
    one["solvers"] = []
    assert M.validate(one) == []
    r2, r1 = run(two)["models"], run(one)["models"]["coupled"]
    assert abs(r2["mechanical"]["max_displacement"] / r1["max_displacement"] - 1) < 1e-9
    assert abs(r2["mechanical"]["quantities"]["elongation"] / r1["quantities"]["elongation"] - 1) < 1e-9
    # free expansion along x: mean elongation of the heated side ~ alpha * mean(T - T_ref) * L
    assert abs(r1["quantities"]["elongation"] / (23e-6 * 50 * 0.2) - 1) < 2e-2


def test_staggered_two_way_coupling():
    """A conductivity that depends on the dilatation (two-way coupling): the
    staggered iterations between two models converge to the solution of the
    monolithic, nonlinear model (Newton)."""
    two = M.load(EXAMPLES / "thermoelastic_plate.json")
    thermal, mech = two["models"]
    thermal["imports"] = [{"name": "u", "model": "mechanical", "variable": "u"}]
    thermal["modules"][0]["properties"]["k"] = [{"selection": "all", "value": "237*(1 + 2000*Div_u)"}]
    two["solvers"] = [{"name": "main", "type": "staggered", "steps": ["thermal", "mechanical"],
                       "max_iter": 50, "tolerance": 1e-10}]
    assert M.validate(two) == []
    one = copy.deepcopy(two)
    t1, m1 = one["models"]
    t1.pop("imports")
    m1.pop("imports")
    t1["variables"] += m1["variables"]
    t1["modules"] += m1["modules"]
    t1["outputs"] += m1["outputs"]
    t1["name"] = "coupled"
    one["models"] = [t1]
    one["solvers"] = []
    res2 = run(two)
    assert res2["solver"]["converged"] and res2["solver"]["iterations"] > 2, res2["solver"]
    r1 = run(one)["models"]["coupled"]
    r2 = res2["models"]
    assert r1["iterations"] >= 2                                     # Newton: nonlinear
    assert abs(r2["thermal"]["quantities"]["heat_flow"] / r1["quantities"]["heat_flow"] - 1) < 1e-7
    assert abs(r2["mechanical"]["max_displacement"] / r1["max_displacement"] - 1) < 1e-7
    # without the coupling the flow would be 5925 W/m: the dilatation raises k
    assert r1["quantities"]["heat_flow"] > 5925 * 1.05


def test_terms_and_generic_conditions():
    """-Laplace T = 2 written as user terms, with Dirichlet (three methods),
    Neumann and Robin conditions: exact with P2."""
    for method in ("simplification", "multiplier", "penalization"):
        m = M.default_model()
        md = m["models"][0]
        md["modules"] = []
        md["parameters"] = [{"name": "f", "value": 2.0, "description": "a local parameter"}]
        md["terms"] = [{"name": "diffusion", "expr": "Grad_T.Grad_Test_T", "selection": "all"},
                       {"name": "source", "expr": "-f*Test_T", "selection": "all"}]
        # exact: T = x (1 - x) + y; on ymin dT/dn = -1 (Neumann); on ymax
        # dT/dn = 1 = -h (T - g) with h = 1: g = T + 1 (Robin)
        md["conditions"] = [
            {"type": "dirichlet", "variable": "T", "selection": "xmin", "value": "y", "method": method},
            {"type": "dirichlet", "variable": "T", "selection": "xmax", "value": "y", "method": method},
            {"type": "neumann", "variable": "T", "selection": "ymin", "value": -1.0},
            {"type": "robin", "variable": "T", "selection": "ymax", "h": 1.0, "value": "x*(1-x) + 2"},
        ]
        md["outputs"] = [{"name": "err2", "expr": "sqr(T - (x*(1-x) + y))", "selection": "all"}]
        assert M.validate(m) == [], M.validate(m)
        r = run(m)["models"]["heat"]
        tol = 1e-6 if method == "penalization" else 1e-10
        assert math.sqrt(abs(r["quantities"]["err2"])) < tol, (method, r["quantities"])


def test_nonlinear_user_term():
    """A nonlinear term ('auto' falls back to add_nonlinear_term): -div((1 + T^2) grad T) = 0,
    T = 0 and 1 on the sides: (T + T^3/3) is linear in x."""
    m = M.default_model()
    md = m["models"][0]
    md["modules"] = []
    md["terms"] = [{"name": "diffusion", "expr": "(1 + sqr(T))*Grad_T.Grad_Test_T", "selection": "all"}]
    md["conditions"] = [{"type": "dirichlet", "variable": "T", "selection": "xmin", "value": 0.0},
                        {"type": "dirichlet", "variable": "T", "selection": "xmax", "value": 1.0}]
    m["meshes"][0]["divisions"] = [20, 2]
    md["outputs"] = [{"name": "err2", "expr": "sqr(T + pow(T,3)/3 - 4/3*x)", "selection": "all"}]
    r = run(m)["models"]["heat"]
    assert r["iterations"] >= 2 and "relative_residual" not in r
    assert math.sqrt(r["quantities"]["err2"]) < 1e-4


def test_coupling_validation():
    two = M.load(EXAMPLES / "thermoelastic_plate.json")
    assert M.validate(two) == []

    def errs(mutate):
        m = copy.deepcopy(two)
        mutate(m)
        return M.validate(m)
    assert any("solved after 'mechanical'" in e for e in errs(
        lambda m: m["solvers"][0].update(steps=["mechanical", "thermal"])))
    assert any("is not solved by the solver" in e for e in errs(
        lambda m: m["solvers"][0].update(steps=["mechanical"])))
    assert any("has no variable 'Q'" in e for e in errs(
        lambda m: m["models"][1]["imports"][0].update(variable="Q")))
    assert any("requires a linear elasticity module" in e for e in errs(
        lambda m: m["models"][1]["modules"].pop(0)))
    assert any("'L' is already a parameter" in e for e in errs(
        lambda m: m["models"][0].update(parameters=[{"name": "L", "value": 1}])))
    assert any("same mesh" in e for e in errs(lambda m: (
        m["meshes"].append({"name": "other", "geometry": "plate", "mesher": "structured", "divisions": [4, 4]}),
        m["models"][1]["variables"][0].update(mesh="other"))))
    # without global solver, the models are solved in the order of their imports
    m = copy.deepcopy(two)
    m["solvers"] = []
    m["models"].reverse()
    assert M.validate(m) == [] and M.solve_order(m) == ["thermal", "mechanical"]
