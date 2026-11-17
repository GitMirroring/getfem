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


def bar(end=0.1, step=0.01, theta=1.0, n=40, every=1, scheme=None):
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
                     "schemes": {"heat/T": scheme or {"type": "theta", "theta": theta}}}]
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
                        ({"schemes": {"heat/T": {"theta": 0}}}, "number in ]0, 1] expected"),
                        ({"schemes": {"heat/T": {"type": "bdf"}}}, "stationary, theta, bdf2 expected")]:
        m2 = bar()
        m2["solvers"][0].update(change)
        assert any(msg in e for e in M.validate(m2)), (msg, M.validate(m2))
    m2 = bar()                                   # the density is needed in time
    del m2["models"][0]["modules"][0]["properties"]["rho"]
    assert any("missing density" in e and "needed in time" in e for e in M.validate(m2))
    m2["solvers"] = []                           # no transient solver: stationary, no density needed
    assert M.validate(m2) == [] and M.time_variables(m2, m2["models"][0]) == []
    m2 = bar(scheme={"type": "stationary"})      # stationary in a transient solver: quasi-static
    del m2["models"][0]["modules"][0]["properties"]["rho"]
    assert M.validate(m2) == [] and M.time_variables(m2, m2["models"][0]) == []
    m2 = bar()                                   # no scheme given: implicit Euler
    m2["solvers"][0]["schemes"] = {"other/T": {"type": "bdf"}}       # (an unknown model: ignored)
    md = m2["models"][0]
    assert M.validate(m2) == [] and M.scheme_of(m2, md, "T") == M.DEFAULT_SCHEME
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


def test_bdf2():
    """BDF2: second order with both first steps (implicit Euler, the default,
    or Crank-Nicolson: a smaller error); its coefficients after the first
    step; started by implicit Euler, monotone for initial values that do
    not satisfy the conditions (Crank-Nicolson overshoots)."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    for sc in ({"type": "bdf2"}, {"type": "bdf2", "start": "crank_nicolson"}):
        errors = []
        for step in (0.01, 0.005):
            results, _ = run(bar(end=0.1, step=step, n=60, scheme=sc))
            S = results["models"]["heat"]["series"]
            errors.append(abs(S["mean"][-1] - 2 / np.pi * np.exp(-np.pi ** 2 * 0.1)))
        assert 3.5 < errors[0] / errors[1] < 4.6 and errors[1] < _error(1.0, 0.005) / 10, (sc, errors)
    text = runner.export_script(bar(scheme={"type": "bdf2"}))
    assert "md_heat.add_initialized_data('T__scheme', [1.0, -1.0, 0.0, 0.0])" in text       # implicit Euler
    assert "md_heat.set_variable('T__scheme', [1.5, -2.0, 0.5, 0.0])" in text               # then BDF2
    assert "md_heat.add_macro('Dot_T', '(T__scheme(1)*T + T__scheme(2)*T__prev + T__scheme(3)*T__prev2)" \
        "/timestep + T__scheme(4)*T__rate')" in text
    assert "initial time derivative" not in text                                          # (not needed)
    text = runner.export_script(bar(scheme={"type": "bdf2", "start": "crank_nicolson"}))
    assert "md_heat.add_initialized_data('T__scheme', [2.0, -2.0, 0.0, -1.0])" in text     # Crank-Nicolson
    assert M.scheme_coefficients({"type": "theta", "theta": 0.5}) == \
        M.scheme_coefficients({"type": "bdf2", "start": "crank_nicolson"}, True)
    assert M.validate(bar(scheme={"type": "bdf2", "start": "euler"})) != []
    m = M.default_model()                         # T = 0 at first, 1 imposed on a side: the maximum principle
    for sc, monotone in (({"type": "bdf2"}, True), ({"type": "bdf2", "start": "crank_nicolson"}, False)):
        m["solvers"] = [{"name": "main", "type": "transient", "steps": ["heat"],
                         "time": {"end": 0.5, "step": 0.05}, "schemes": {"heat/T": sc}}]
        top = max(run(m)[0]["models"]["heat"]["series"]["max_T"])
        assert (top <= 1 + 1e-9) == monotone, (sc, top)


def test_derivative_in_outputs():
    """Dot_T in an output is the time derivative at the saved time (the one
    kept by the scheme): here d(mean)/dt = -pi^2 mean."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    for sc in ({"type": "theta", "theta": 0.5}, {"type": "bdf2"}):
        m = bar(end=0.1, step=0.0025, n=60, scheme=sc)
        m["models"][0]["outputs"].append({"name": "rate", "expr": "Dot_T", "selection": "all"})
        S = run(m)[0]["models"]["heat"]["series"]
        ratio = np.array(S["rate"]) / np.array(S["mean"])
        first = 0 if M.scheme_needs_rate(sc) else 1         # (the initial derivative: computed if needed)
        assert np.allclose(ratio[first:], -np.pi ** 2, rtol=2e-3), (sc, ratio)


def test_schemes_per_variable():
    """A scheme per variable: T in time (BDF2), S stationary at each time
    (-div(grad S) = T, T imported: quasi-static, following T)."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    m = bar(end=0.1, step=0.02, scheme={"type": "bdf2"})
    second = copy.deepcopy(m["models"][0])
    second.update(name="second", imports=[{"name": "Ta", "model": "heat", "variable": "T"}])
    second["variables"][0].update(name="S", initial=0)
    mod = second["modules"][0]
    mod.update(variable="S", sources=[{"selection": "all", "value": "Ta"}])
    mod["conditions"][0]["selection"] = "boundary"
    for p in ("rho", "cp"):
        del mod["properties"][p]
    second["outputs"] = [{"name": "mean_S", "expr": "S", "selection": "all"}]
    m["models"].append(second)
    s = m["solvers"][0]
    s["steps"].append("second")
    s["schemes"]["second/S"] = {"type": "stationary"}
    assert M.validate(m) == [] and M.time_variables(m, second) == [] and not M.in_time(m, second, mod)
    assert M.time_variables(m, m["models"][0]) == ["T"]
    results, _ = run(m)
    T, S = results["models"]["heat"]["series"], results["models"]["second"]["series"]
    ratio = np.array(S["mean_S"]) / np.array(T["mean"])         # S = T / pi^2 at each time
    assert np.allclose(ratio[1:], 1 / np.pi ** 2, rtol=1e-3), ratio
    text = runner.export_script(m)
    assert "S__scheme" not in text and "Dot_S" not in text and "rho*cp*Dot_T*Test_T" in text


def test_upgrade():
    """The earlier form: one scheme per model, the modules "transient" or
    not; made one scheme per variable on loading."""
    m = bar(theta=0.5)
    s = m["solvers"][0]
    s["schemes"] = {"heat": {"type": "theta", "theta": 0.5}}
    m["models"][0]["modules"][0]["transient"] = True
    other = copy.deepcopy(m["models"][0])
    other["name"] = "other"
    other["modules"][0]["transient"] = False
    m["models"].append(other)
    s["steps"].append("other")
    assert M.upgrade_schemes(m)
    assert s["schemes"] == {"heat/T": {"type": "theta", "theta": 0.5}, "other/T": {"type": "stationary"}}
    assert all("transient" not in mod for md in m["models"] for mod in md["modules"])
    assert not M.upgrade_schemes(m) and M.validate(m) == []
    with tempfile.TemporaryDirectory() as tmp:                     # on loading
        old = bar()
        old["solvers"][0]["schemes"] = {"heat": {"type": "theta", "theta": 1.0}}
        old["models"][0]["modules"][0]["transient"] = True
        M.save(old, Path(tmp) / "old.json")
        new = M.load(Path(tmp) / "old.json")
        assert new["solvers"][0]["schemes"] == {"heat/T": {"type": "theta", "theta": 1.0}}


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
    assert s["time"] == {"start": 0.0, "end": 1.0, "step": 0.1} and s["schemes"] == {"heat/T": {"type": "theta",
                                                                                             "theta": 1.0}}
    assert M.validate(m) == []
    labels = [x.label for x in E.fields(m, path)]
    assert {"Start time", "End time", "Time step", "Save every (steps)", "Coupling at each step",
            "Scheme of T (heat)", "Theta of T"} <= set(labels)
    f = next(x for x in E.fields(m, path) if x.label == "Scheme of T (heat)")
    assert E.field_value(m, f) == "theta" and dict(f.choices)["bdf2"].startswith("BDF2")
    E.apply(m, f, "bdf2")
    assert s["schemes"]["heat/T"] == {"type": "bdf2", "start": "implicit_euler"}
    labels = [x.label for x in E.fields(m, path)]
    assert "Theta of T" not in labels and "First step of T" in labels
    E.apply(m, f, "stationary")                  # (no density needed then)
    assert M.time_variables(m, m["models"][0]) == [] and M.validate(m) == []
    E.apply(m, f, "theta")
    assert s["schemes"]["heat/T"] == {"type": "theta", "theta": 1.0}
    f = next(x for x in E.fields(m, path) if x.label == "Theta of T")
    E.apply(m, f, 0.5)
    assert s["schemes"]["heat/T"]["theta"] == 0.5 and E.field_value(m, f) == 0.5
    f = next(x for x in E.fields(m, path) if x.key[-1] == "coupling")
    E.apply(m, f, "staggered")
    assert s["max_iter"] == 20 and "Maximum iterations" in [x.label for x in E.fields(m, path)]
    f = next(x for x in E.fields(m, ("models", 0)) if x.key[-1] == "name")
    E.apply(m, f, "thermal")
    assert s["steps"] == ["thermal"] and s["schemes"] == {"thermal/T": {"type": "theta", "theta": 0.5}}
    f = next(x for x in E.fields(m, ("models", 0, "variables", 0)) if x.key[-1] == "name")
    E.apply(m, f, "Temp")                        # the rename of a variable followed
    assert s["schemes"] == {"thermal/Temp": {"type": "theta", "theta": 0.5}} and M.validate(m) == []
    m["models"].append(M.new_model("other", "mesh", "heat_conduction"))
    f = next(x for x in E.fields(m, path) if x.key[-1] == "steps")
    E.apply(m, f, ["thermal", "other"])          # a model added: its variables in time (rho, cp given)
    assert s["schemes"]["other/T"] == {"type": "theta", "theta": 1.0} and M.validate(m) == []
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


def test_window():
    """The results in time in the window: the time bar of the view (the
    last time first; another time: its file, its values in the table), the
    curves of the scalar results; hidden elsewhere."""
    try:
        import getfem  # noqa: F401
        from PySide6.QtWidgets import QApplication, QComboBox
    except ImportError:
        return
    import os
    import shutil
    import time
    os.environ.update(QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"), GETFEM_GUI_NO_3D="",
                      GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.plot import LinePlot
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])

    def wait(condition, timeout=120.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            app.processEvents()
            if condition():
                return True
            time.sleep(0.01)
        return False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plate_heating.json"
        shutil.copy(Path(__file__).resolve().parents[1] / "examples" / "plate_heating.json", path)
        win = MainWindow(str(path))
        try:
            v = win.viewer
            win.compute()
            assert wait(lambda: win.results is not None and not win.jobs.running), win.log.toPlainText()
            assert win.current_path() == (E.RESULTS, "heat", "T")
            assert win.result_times("heat") == [5.0 * k for k in range(13)]
            assert not v.time_bar.isHidden() and v.time_label.text() == "t = 60  (13/13)"
            assert v.status.text().endswith("t = 60") and win.result_files["heat"].name == "heat_00060.vtu"
            lo, hi = v.plotter.actors["field"].mapper.scalar_range
            assert abs(lo - 20) < 1e-6 and 99 < hi <= 100.01                  # over all the times
            v.time_slider.setValue(2)
            assert win.result_files["heat"].name == "heat_00010.vtu" and v.status.text().endswith("t = 10")
            assert tuple(v.plotter.actors["field"].mapper.scalar_range) == (lo, hi)    # the same colors
            win.select((E.RESULTS,))
            table = win.props.findChild(object, "results_table")
            rows = {table.item(i, 0).text(): table.item(i, 1).text() for i in range(table.rowCount())}
            assert rows["time"] == "10" and 40 < float(rows["∫ mean_T"]) < 50
            combo = win.props.findChild(QComboBox, "curve")
            assert combo.currentText() == "mean_T" and len(win.props.findChild(LinePlot, "curve_plot").x) == 13
            v.play_button.setChecked(True)
            assert wait(lambda: not v.play_button.isChecked(), 30) and v.time_slider.value() == 12
            win.select(("models", 0))
            assert v.time_bar.isHidden()
        finally:
            win.setWindowModified(False)
            win.close()


def test_module_form():
    """The form of a module: no In time box (the scheme is chosen in the
    transient solver), the equation solved and where it stands in time
    (stationary, in time and its scheme, quasi-static), its data, the GWFL
    terms of the script."""
    from getfem_gui import equations
    m = M.default_model()
    path = ("models", 0, "modules", 0)
    fields = {f.label: f for f in E.fields(m, path)}
    assert "In time" not in fields
    eq = fields["Equation"].help
    assert "<b>Stationary</b> (a transient global solver" in eq and "−div(k ∇T) = f" in eq and "xmin" not in eq
    assert "rho (density)" not in fields and E.field_value(m, fields["k (thermal conductivity)"]) == 1.0
    terms = fields["GWFL terms of the script"].help.splitlines()
    assert "k*Grad_T.Grad_Test_T" in terms and not any("xmin" in x for x in terms)    # (no conditions)
    path_s = E.do_action(m, ("solvers",), "add_solver")         # in time: by a transient solver
    E.apply(m, next(x for x in E.fields(m, path_s) if x.key[-1] == "type"), "transient")
    assert M.validate(m) == []
    fields = {f.label: f for f in E.fields(m, path)}
    assert "ρ c<sub>p</sub> ∂T/∂t − div(k ∇T) = f" in fields["Equation"].help
    assert "from t = 0 to 1, step 0.1, implicit Euler" in fields["Equation"].help
    assert {"rho (density)", "cp (specific heat capacity)"} <= set(fields)
    assert fields["GWFL terms of the script"].help.splitlines()[-1] == "rho*cp*Dot_T*Test_T"
    f = next(x for x in E.fields(m, path_s) if x.label == "Scheme of T (heat)")
    E.apply(m, f, "bdf2")
    assert "step 0.1, BDF2" in equations.equation(m, m["models"][0], m["models"][0]["modules"][0])
    E.apply(m, f, "stationary")
    assert M.validate(m) == [] and "quasi-static" in equations.equation(m, m["models"][0],
                                                                        m["models"][0]["modules"][0])
    assert "rho (density)" not in {f.label for f in E.fields(m, path)}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
