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

"""Partial resolutions: a step of a global solver solving some variables of
a model ("model/u,v"), the others given (disabled, with their
multipliers). In sequence and by staggered iterations, linear and
nonlinear, stationary and in time: the same solution as the monolithic
resolution; validation, editing.

    python -m pytest tests/test_partial_steps.py      (or: python tests/test_partial_steps.py)
"""

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, model as M, runner  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def run(model):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(model, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-3000:]}"
        return res.results


def _getfem():
    try:
        import getfem  # noqa: F401
        return True
    except ImportError:
        return False


def one_model(k="237"):
    """The thermo-elastic plate of the example in one model (T and u); k: its
    conductivity (237*(1 + 2000*Div_u): a two-way, nonlinear coupling)."""
    m = M.load(EXAMPLES / "thermoelastic_plate.json")
    thermal, mech = m["models"]
    mech.pop("imports")
    thermal["variables"] += mech["variables"]
    thermal["modules"] += mech["modules"]
    thermal["outputs"] += mech["outputs"]
    thermal["name"] = "coupled"
    thermal["modules"][0]["properties"]["k"] = [{"selection": "all", "value": k}]
    m["models"] = [thermal]
    m["solvers"] = []
    return m


def coupled_terms(method="multiplier"):
    """-div grad T = 1 + S/2, -div grad S = T/2 (terms, linear), T = 0 by a
    Dirichlet condition (method), S = 0: a two-way linear coupling."""
    m = M.default_model()
    md = m["models"][0]
    md["modules"] = []
    md["variables"].append(dict(md["variables"][0], name="S"))
    md["terms"] = [{"name": "T", "expr": "Grad_T.Grad_Test_T - (1 + S/2)*Test_T", "selection": "all"},
                   {"name": "S", "expr": "Grad_S.Grad_Test_S - T/2*Test_S", "selection": "all"}]
    md["conditions"] = [{"type": "dirichlet", "variable": "T", "selection": "boundary", "value": 0,
                         "method": method},
                        {"type": "dirichlet", "variable": "S", "selection": "boundary", "value": 0,
                         "method": "simplification"}]
    md["outputs"] = [{"name": "T_mean", "expr": "T", "selection": "all"},
                     {"name": "S_mean", "expr": "S", "selection": "all"}]
    return m


def test_parse_and_validation():
    assert M.parse_step("heat") == ("heat", None) and M.parse_step("heat/T, S") == ("heat", ["T", "S"])
    assert M.make_step("heat", ["T", "S"]) == "heat/T,S" and M.make_step("heat") == "heat"
    m = one_model()
    m["solvers"] = [{"name": "main", "type": "sequence", "steps": ["coupled/T", "coupled/u"]}]
    assert M.validate(m) == [] and M.step_models(m["solvers"][0]) == ["coupled"]
    assert M.partial_models(m["solvers"][0]) == ["coupled"]
    for steps, msg in ((["coupled/T"], "the variable 'u' of 'coupled' solved by no step"),
                       (["coupled/T", "coupled/v"], "model 'coupled' has no variable 'v'"),
                       (["coupled/T,T", "coupled/u"], "each variable once"),
                       (["other/T"], "unknown model 'other'")):
        m["solvers"][0]["steps"] = steps
        assert any(msg in e for e in M.validate(m)), (steps, M.validate(m))
    m["solvers"][0]["steps"] = ["coupled/T", "coupled"]                 # (a variable solved twice: fine)
    assert M.validate(m) == []


def test_sequence_one_way():
    """T then u in the same (linear) model: u takes the T just solved, as
    the monolithic resolution (a one-way coupling)."""
    if not _getfem():
        return
    m = one_model()
    ref = run(m)["models"]["coupled"]
    m["solvers"] = [{"name": "main", "type": "sequence", "steps": ["coupled/T", "coupled/u"]}]
    r = run(m)["models"]["coupled"]
    assert abs(r["quantities"]["elongation"] / ref["quantities"]["elongation"] - 1) < 1e-9
    assert abs(r["max_displacement"] / ref["max_displacement"] - 1) < 1e-9
    m["solvers"][0]["steps"] = ["coupled/u", "coupled/T"]   # u first: with T = 0 (not yet solved),
    r = run(m)["models"]["coupled"]                          # below T_ref = 20: the plate contracts
    assert r["quantities"]["elongation"] < 0 < ref["quantities"]["elongation"]


def test_staggered_two_way():
    """A two-way coupling inside one model: staggered iterations on T and u
    converge to the monolithic solution, nonlinear (k depends on the
    dilatation: Newton) and linear (terms; a Dirichlet condition by a
    multiplier, disabled with its variable)."""
    if not _getfem():
        return
    m = one_model("237*(1 + 2000*Div_u)")
    ref = run(m)["models"]["coupled"]
    m["solvers"] = [{"name": "main", "type": "staggered", "steps": ["coupled/T", "coupled/u"],
                     "max_iter": 50, "tolerance": 1e-10}]
    res = run(m)
    assert res["solver"]["converged"] and res["solver"]["iterations"] > 2, res["solver"]
    r = res["models"]["coupled"]
    assert abs(r["quantities"]["heat_flow"] / ref["quantities"]["heat_flow"] - 1) < 1e-7
    assert abs(r["max_displacement"] / ref["max_displacement"] - 1) < 1e-7
    for method in ("multiplier", "penalization", "simplification"):
        m = coupled_terms(method)
        ref = run(m)["models"]["heat"]["quantities"]
        m["solvers"] = [{"name": "main", "type": "staggered", "steps": ["heat/T", "heat/S"],
                         "max_iter": 50, "tolerance": 1e-12, "relaxation": 0.9}]
        assert M.validate(m) == []
        res = run(m)
        q = res["models"]["heat"]["quantities"]
        assert res["solver"]["converged"] and res["solver"]["iterations"] > 2
        assert abs(q["S_mean"] / ref["S_mean"] - 1) < 1e-8 and abs(q["T_mean"] / ref["T_mean"] - 1) < 1e-8
        assert abs(ref["S_mean"]) > 1e-3 * abs(ref["T_mean"])             # (a real coupling)
    text = runner.export_script(coupled_terms())
    assert "MULTIPLIERS" not in text and "disable_variable" not in text  # (only with partial steps)


def test_in_time():
    """The thermal shock in one model: T then u at each time step gives
    the same as the monolithic resolution (T does not depend on u)."""
    if not _getfem():
        return
    m = M.load(EXAMPLES / "thermal_shock_strip.json")
    thermal, mech = m["models"]
    mech.pop("imports")
    thermal["variables"] += mech["variables"]
    thermal["modules"] += mech["modules"]
    thermal["outputs"] += mech["outputs"]
    thermal["name"] = "coupled"
    m["models"] = [thermal]
    s = m["solvers"][0]
    s["time"]["end"] = 0.005
    s["steps"] = ["coupled"]
    s["schemes"] = {"coupled/T": {"type": "bdf2"}, "coupled/u": {"type": "generalized_alpha", "rho_inf": 0.9}}
    assert M.validate(m) == []
    ref = np.array(run(m)["models"]["coupled"]["series"]["tip_deflection"])
    s["steps"] = ["coupled/T", "coupled/u"]
    tip = np.array(run(m)["models"]["coupled"]["series"]["tip_deflection"])
    assert np.abs(tip - ref).max() < 1e-7 * np.abs(ref).max() and np.abs(ref).max() > 0
    # rho, a property of the heat and of the elasticity: one data of the model, defined once
    heat, elas = thermal["modules"][:2]
    heat["properties"]["rho"] = [{"selection": "all", "value": 1000}]
    assert any("properties.rho: defined otherwise by module 1" in e for e in M.validate(m))
    s["schemes"]["coupled/u"] = {"type": "stationary"}                 # (rho not used by the elasticity)
    assert M.validate(m) == []


def test_editing():
    """The steps offered (each model, each variable of a model with
    several), the renames followed."""
    m = one_model()
    m["solvers"] = [{"name": "main", "type": "sequence", "steps": ["coupled/T", "coupled/u"]}]
    f = next(x for x in E.fields(m, ("solvers", 0)) if x.key[-1] == "steps")
    assert [c for c, _ in f.choices] == ["coupled", "coupled/T", "coupled/u"]
    E.apply(m, f, ["coupled/u", "coupled/T"])
    assert m["solvers"][0]["steps"] == ["coupled/u", "coupled/T"]
    v = next(x for x in E.fields(m, ("models", 0, "variables", 0)) if x.key[-1] == "name")
    E.apply(m, v, "Temp")
    assert m["solvers"][0]["steps"] == ["coupled/u", "coupled/Temp"]
    n = next(x for x in E.fields(m, ("models", 0)) if x.key[-1] == "name")
    E.apply(m, n, "plate")
    assert m["solvers"][0]["steps"] == ["plate/u", "plate/Temp"] and M.validate(m) == []
    m["solvers"][0]["steps"] = ["plate/u,Temp"]                          # (a step of several variables)
    f = next(x for x in E.fields(m, ("solvers", 0)) if x.key[-1] == "steps")
    assert ("plate/u,Temp", "plate: u, Temp") in f.choices


def test_shared_property():
    """A module added takes the properties already given by the others (rho
    of the elasticity for a heat conduction); a property edited in one
    module is edited in the others."""
    m = M.load(EXAMPLES / "beam_vibration.json")
    E.do_action(m, ("models", 0, "modules"), "add_module:heat_conduction")
    elas, heat = m["models"][0]["modules"]
    assert heat["properties"]["rho"] == elas["properties"]["rho"] == [{"selection": "all", "material": "steel"}]
    assert M.validate(m) == []
    f = next(x for x in E.fields(m, ("models", 0, "modules", 1)) if x.label.startswith("rho"))
    E.apply(m, f, f.choices[0][0] if f.choices else 7000.0)
    assert heat["properties"]["rho"] == elas["properties"]["rho"] and M.validate(m) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
