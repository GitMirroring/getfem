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

"""Transient solvers ({"type": "transient", "time", "coupling", "save",
"schemes"}): validation, the theta-method (orders of convergence on an
exact solution), the data depending on the time, the initial values, the
couplings at each time step, the results (series, VTU files, .pvd),
editing.

    python -m pytest tests/test_transient.py      (or: python tests/test_transient.py)
"""

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import codegen, editing as E, model as M, runner  # noqa: E402


def bar(end=0.1, step=0.01, theta=1.0, n=40, every=1):
    """A bar [0, 1], its ends at 0, from sin(pi x): T = exp(-pi^2 t) sin(pi x)."""
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "box", "origin": [0], "size": [1]}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "structured", "divisions": [n]}
    md = m["models"][0]
    md["variables"][0]["initial"] = "sin(pi*x)"
    mod = md["modules"][0]
    mod["conditions"] = [{"type": "temperature", "selection": "boundary", "value": 0}]
    mod["properties"].update(rho=[{"selection": "all", "value": 2.0}], cp=[{"selection": "all", "value": 0.5}])
    md["outputs"] = [{"name": "mean", "expr": "T", "selection": "all"}]
    m["solvers"] = [{"name": "main", "type": "transient", "steps": [md["name"]],
                     "time": {"end": end, "step": step}, "save": {"every": every},
                     "schemes": {md["name"]: {"type": "theta", "theta": theta}}}]
    return m


def run(m):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, res.error
        files = sorted(p.name for p in Path(tmp).iterdir())
    return res.results, files


def test_validation():
    m = bar()
    assert M.validate(m) == []
    s = m["solvers"][0]
    for change, msg in [({"time": {"end": 0.1}}, "{'start', 'end', 'step'} expected"),
                        ({"time": {"end": -1, "step": 0.1}}, "after the start expected"),
                        ({"time": {"end": 1, "step": 0}}, "strictly positive expected"),
                        ({"time": {"end": 1, "step": "dt"}}, "unknown parameter 'dt'"),
                        ({"coupling": "monolithic"}, "sequence, staggered expected"),
                        ({"save": {"every": 0}}, "save.every"),
                        ({"schemes": {"heat": {"theta": 0}}}, "number in ]0, 1] expected"),
                        ({"schemes": {"heat": {"type": "bdf"}}}, "theta expected")]:
        m2 = bar()
        m2["solvers"][0].update(change)
        assert any(msg in e for e in M.validate(m2)), (msg, M.validate(m2))
    m2 = bar()                                   # the density is needed in time
    del m2["models"][0]["modules"][0]["properties"]["rho"]
    assert any("missing density" in e and "needed in time" in e for e in M.validate(m2))
    m2["solvers"] = []                           # not in a stationary model
    assert M.validate(m2) == []
    m2 = bar()
    m2["parameters"] = [{"name": "dt", "value": 0.02}]
    s = m2["solvers"][0]
    s["time"]["step"] = "dt"
    assert M.validate(m2) == [] and M.time_values(m2, s) == (0.0, 0.1, 0.02)
    m2["models"][0]["variables"][0]["initial"] = [1, 2]
    assert any("initial" in e for e in M.validate(m2))


def _error(theta, step):
    results, _ = run(bar(end=0.1, step=step, theta=theta, n=60))
    S = results["models"]["heat"]["series"]
    exact = 2 / np.pi * np.exp(-np.pi ** 2 * 0.1 * 1.0)                       # rho*cp = 1
    assert abs(S["t"][-1] - 0.1) < 1e-12
    return abs(S["mean"][-1] - exact)


def test_theta_method():
    """Implicit Euler: first order; Crank-Nicolson: second order (much
    smaller errors)."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    e1, e2 = _error(1.0, 0.02), _error(1.0, 0.01)
    assert 1.8 < e1 / e2 < 2.2
    c1, c2 = _error(0.5, 0.02), _error(0.5, 0.01)
    assert 3.5 < c1 / c2 < 4.5 and c2 < e2 / 20


def test_series_and_files():
    """Saved every 2 steps (and at the end): the times, the scalars, a VTU
    file each, the .pvd collection; the final values as for a stationary
    model; the initial value."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    results, files = run(bar(end=0.1, step=0.02, every=2))
    R = results["models"]["heat"]
    S = R["series"]
    assert np.allclose(S["t"], [0, 0.04, 0.08, 0.1])
    assert abs(S["mean"][0] - 2 / np.pi) < 1e-4 and S["min_T"][0] == min(S["min_T"])
    assert S["vtu"] == ["heat_00000.vtu", "heat_00002.vtu", "heat_00004.vtu", "heat_00005.vtu"]
    assert R["vtu"] == "heat_00005.vtu" and R["pvd"] == "heat.pvd" and R["quantities"]["mean"] == S["mean"][-1]
    assert set(S["vtu"]) | {"heat.pvd"} <= set(files)
    assert results["solver"] == {"name": "main", "type": "transient", "steps": 5, "start": 0.0, "end": 0.1}


def test_time_dependent_data():
    """A temperature imposed as a function of t: updated at each time step
    (the ends of the bar follow it)."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    m = bar(end=1.0, step=0.1)
    md = m["models"][0]
    md["variables"][0]["initial"] = 0
    md["modules"][0]["conditions"][0]["value"] = "2*t"
    md["outputs"] = [{"name": "ends", "expr": "T", "selection": "boundary"}]
    results, _ = run(m)
    S = results["models"]["heat"]["series"]
    assert np.allclose(S["ends"], 2 * 2 * np.array(S["t"]), atol=1e-9) and S["ends"][0] == 0
    text = runner.export_script(m)
    assert "md_heat.set_variable('T_D1_1', md_heat.interpolation('(2*t)', mf_mesh_P2))" in text
    assert codegen.uses_time("2*t") and codegen.uses_time("sin(t)") and not codegen.uses_time("sqrt(x)") \
        and not codegen.uses_time("t_ext + T")


def test_coupling():
    """Two models in time: the second one heated by the temperature of the
    first (imported at each step); in turn, or by staggered iterations
    (the same here: the first does not depend on the second)."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    m = bar(end=0.1, step=0.02)
    second = copy.deepcopy(m["models"][0])
    second.update(name="second", imports=[{"name": "Ta", "model": "heat", "variable": "T"}], vtu="second.vtu")
    second["variables"][0]["initial"] = 0
    second["modules"][0]["sources"] = [{"selection": "all", "value": "Ta"}]
    m["models"].append(second)
    s = m["solvers"][0]
    s["steps"].append("second")
    assert M.validate(m) == []
    results, files = run(m)
    a, b = (results["models"][n]["series"] for n in ("heat", "second"))
    assert a["t"] == b["t"] and b["mean"][0] == 0 and b["mean"][-1] > 0.001 and "second.pvd" in files
    s["coupling"] = "staggered"
    results2, _ = run(m)
    assert np.allclose(results2["models"]["second"]["series"]["mean"], b["mean"], rtol=1e-6)
    s["coupling"], s["steps"] = "sequence", ["second", "heat"]
    assert any("solved after" in e for e in M.validate(m))


def test_editing():
    """The type of a solver set to transient: its settings and fields; the
    rename of a model followed; the initial value of a variable."""
    m = bar()
    m["solvers"] = []
    path = E.do_action(m, ("solvers",), "add_solver")
    f = next(x for x in E.fields(m, path) if x.key[-1] == "type")
    E.apply(m, f, "transient")
    s = E.get_at(m, path)
    assert s["time"] == {"start": 0.0, "end": 1.0, "step": 0.1} and s["schemes"] == {"heat": {"type": "theta",
                                                                                           "theta": 1.0}}
    assert M.validate(m) == []
    labels = [x.label for x in E.fields(m, path)]
    assert {"Start time", "End time", "Time step", "Save every (steps)", "Coupling at each step",
            "Theta of heat"} <= set(labels)
    f = next(x for x in E.fields(m, path) if x.key[-1] == "coupling")
    E.apply(m, f, "staggered")
    assert s["max_iter"] == 20 and "Maximum iterations" in [x.label for x in E.fields(m, path)]
    f = next(x for x in E.fields(m, ("models", 0)) if x.key[-1] == "name")
    E.apply(m, f, "thermal")
    assert s["steps"] == ["thermal"] and "thermal" in s["schemes"] and M.validate(m) == []
    f = next(x for x in E.fields(m, path) if x.key[-1] == "type")
    E.apply(m, f, "sequence")
    assert "time" not in s and "schemes" not in s
    init = next(x for x in E.fields(m, ("models", 0, "variables", 0)) if x.key[-1] == "initial")
    assert init.label == "Initial value (in time)" and E.field_value(m, init) == "sin(pi*x)"


def test_example():
    """The example: the plate heated from an edge; it reaches the
    temperature of the edge."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    m = M.load(Path(__file__).resolve().parents[1] / "examples" / "plate_heating.json")
    results, files = run(m)
    S = results["models"]["heat"]["series"]
    assert len(S["t"]) == 13 and abs(S["mean_T"][0] - 20) < 1e-9 and 95 < S["mean_T"][-1] < 100
    assert all(b > a for a, b in zip(S["mean_T"], S["mean_T"][1:]))
    assert max(S["heat_in"]) > 10000 and "heat.pvd" in files


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
