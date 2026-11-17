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

"""Several global solvers: run one after the other, each one from the
initial values or from the state left by the previous one ("start":
"previous"), each one with its results (its folder); one of them only
(--solver, after those it starts from); the time derivatives of a variable
in time for another solver are zero in a stationary one; validation,
editing, the window.

    python -m pytest tests/test_solvers.py      (or: python tests/test_solvers.py)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from getfem_gui import editing as E, model as M, runner  # noqa: E402
import test_elastodynamics as T  # noqa: E402


def three_solvers():
    """The cantilever under a tip load: "static" (its equilibrium);
    "preloaded" (in time, from the state left by static: it stays at
    rest); "sudden" (in time, from the initial values: the load applied at
    once)."""
    m = T.beam(end=0.4, step=0.02)
    md = m["models"][0]
    md["variables"][0]["initial"] = ["0", "0"]
    md["modules"][0]["conditions"].append({"type": "traction", "selection": "xmax", "value": [0, -1e-4]})
    dyn = m["solvers"][0]
    m["solvers"] = [{"name": "static", "type": "sequence", "steps": ["beam"]},
                    dict(dyn, name="preloaded", start="previous"),
                    dict(dyn, name="sudden")]
    return m


def _getfem():
    try:
        import getfem  # noqa: F401
        return True
    except ImportError:
        return False


def test_validation_and_chain():
    m = three_solvers()
    assert M.validate(m) == []
    assert M.solver_chain(m) == ["static", "preloaded", "sudden"]
    assert M.solver_chain(m, "preloaded") == ["static", "preloaded"]
    assert M.solver_chain(m, "sudden") == ["sudden"]
    m["solvers"][0]["start"] = "previous"
    assert any("the first solver starts from the initial values" in e for e in M.validate(m))
    m["solvers"][0]["start"] = "elsewhere"
    assert any("solvers[0].start: initial, previous expected" in e for e in M.validate(m))
    m = three_solvers()                 # rho: needed by the solvers in time only, checked for each one
    m["solvers"] = [m["solvers"][0], m["solvers"][2]]
    del m["models"][0]["modules"][0]["properties"]["rho"]
    errors = M.validate(m)
    assert errors and all(e.endswith("(solver sudden)") for e in errors), errors
    view = M.solver_first(three_solvers(), "sudden")
    assert [s["name"] for s in view["solvers"]] == ["sudden", "static", "preloaded"]


def test_scripts():
    """One solver: the script as before; several: a function each, run in
    order, the time derivatives zero in a stationary one."""
    one = T.beam()
    text = runner.export_script(one)
    assert "def solver_" not in text and "--solver" not in text
    text = runner.export_script(three_solvers())
    for line in ("def solver_static(state=None):", "def solver_preloaded(state=None):",
                 "SOLVERS = {'static': (solver_static, 'initial'), 'preloaded': (solver_preloaded, 'previous'), "
                 "'sudden': (solver_sudden, 'initial')}",
                 "md_beam.set_variable('u', state['beam']['u'])",
                 "md_beam.add_macro('Dot_u', '0*u')"):
        assert line in text, line


def test_run():
    """All the solvers, in order, each one its folder and its results; the
    preloaded beam stays at the static equilibrium; one solver only."""
    if not _getfem():
        return
    m = three_solvers()
    with tempfile.TemporaryDirectory() as tmp:
        r = runner.run(m, tmp)
        assert r.ok, r.stderr[-2000:]
        R = r.results
        assert R["run"] == ["static", "preloaded", "sudden"] and R["solver_name"] == "sudden"
        assert set(R["solvers"]) == {"static", "preloaded", "sudden"}
        assert R["models"] == R["solvers"]["sudden"]["models"]                  # (the last one)
        for name in R["run"]:
            mr = R["solvers"][name]["models"]["beam"]
            assert mr["vtu"].startswith(f"{name}/") and (Path(tmp) / mr["vtu"]).exists()
        static = R["solvers"]["static"]["models"]["beam"]
        tip = static["quantities"]["tip"]
        assert tip < 0 and static["quantities"]["kinetic"] == 0               # (Dot_u: zero there)
        pre = np.array(R["solvers"]["preloaded"]["models"]["beam"]["series"]["tip"])
        assert np.abs(pre - tip).max() < 1e-9 * abs(tip)                       # at rest at the equilibrium
        sud = np.array(R["solvers"]["sudden"]["models"]["beam"]["series"]["tip"])
        assert sud[0] == 0 and sud[-1] < 0                                     # from rest, set in motion
        r = runner.run(m, tmp, solver="preloaded")                            # with the one it starts from
        assert r.ok and r.results["run"] == ["static", "preloaded"]
        r = runner.run(m, tmp, solver="sudden")
        assert r.ok and r.results["run"] == ["sudden"]
        assert json.loads((Path(tmp) / "results.json").read_text())["run"] == ["sudden"]


def test_editing():
    m = three_solvers()
    labels = lambda i: [f.label for f in E.fields(m, ("solvers", i))]          # noqa: E731
    assert "Start from" not in labels(0) and "Start from" in labels(1)
    f = next(x for x in E.fields(m, ("solvers", 2)) if x.label == "Start from")
    assert E.field_value(m, f) == "initial"
    E.apply(m, f, "previous")
    assert m["solvers"][2]["start"] == "previous"
    E.apply(m, f, "initial")
    assert "start" not in m["solvers"][2]
    path = E.do_action(m, ("solvers",), "add_solver")  # a copy of the last one, from the initial values
    new = E.get_at(m, path)
    assert new["name"] == "solver" and new["type"] == "transient" and "start" not in new
    assert new["schemes"] == m["solvers"][2]["schemes"] and new["schemes"] is not m["solvers"][2]["schemes"]
    E.remove_at(m, path)
    E.remove_at(m, ("solvers", 0))                     # the new first one: from the initial values
    assert m["solvers"][0]["name"] == "preloaded" and "start" not in m["solvers"][0] and M.validate(m) == []


def test_window():
    """Compute this solver; the results of each solver computed."""
    if not _getfem():
        return
    try:
        import test_app as A
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except ImportError:
        return
    from getfem_gui.app.window import MainWindow
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "three.json"
        M.save(three_solvers(), path)
        A.app()
        win = MainWindow(path)
        win.show()
        assert ("cmd:compute_solver", "Compute this solver") in win._node_actions(("solvers", 1))
        assert ("cmd:compute_solver", "Compute this solver") not in win._node_actions(("solvers",))
        win._node_action(("solvers", 1), "cmd:compute_solver") if hasattr(win, "_node_action") \
            else win.compute("preloaded")
        assert A.wait(lambda: win.results is not None and not win.jobs.running), win.log.toPlainText()
        assert win.results["run"] == ["static", "preloaded"] and win.results["solver_name"] == "preloaded"
        assert win._results_summary()["global solver"] == "preloaded"
        assert ("cmd:results_of:static", "Results of the solver static") in win._results_actions()
        win.show_solver_results("static")
        assert win.results["solver_name"] == "static" and win.result_files["beam"].parent.name == "static"
        assert win.result_series == {}                                          # (stationary)
        A.close(win)


def test_linear_solvers():
    """The solver of the linear systems of a model: each one gives the same
    solution (the plate, its temperature by simplification, non symmetric:
    the direct ones and GMRES, also for the elasticity; with convection only,
    symmetric positive definite: CG too); MUMPS: SuperLU if not available;
    editing."""
    m = M.load(T.Path(__file__).resolve().parents[1] / "examples" / "plate.json")
    md = m["models"][0]
    f = next(x for x in E.fields(m, ("models", 0, "solver")) if x.label == "Linear solver")
    assert E.field_value(m, f) == "auto" and "gmres/ilut" in dict(f.choices)
    E.apply(m, f, "gmres/ilu")
    assert md["solver"]["linear_solver"] == "gmres/ilu" and M.validate(m) == []
    E.apply(m, f, "auto")
    assert "linear_solver" not in md["solver"]
    md["solver"]["linear_solver"] = "pardiso"
    assert any("linear_solver: auto, mumps" in e for e in M.validate(m))
    assert "lsolver" not in runner.export_script(M.load(T.Path(__file__).resolve().parents[1] / "examples"
                                                        / "plate.json")).split("def _solver")[0].split(
        "info = md.solve(*options)")[1][:200]          # (auto: GetFEM's choice, as before)
    if not _getfem():
        return
    import getfem
    examples = T.Path(__file__).resolve().parents[1] / "examples"
    direct_gmres = [ls for ls in M.LINEAR_SOLVERS if ls != "cg/ildlt"]
    for name, solvers in (("plate.json", [ls for ls in direct_gmres if ls != "gmres/ilutp"]),
                          ("plate_hole_tension.json", direct_gmres),
                          ("convection only", ["superlu", "cg/ildlt"])):
        ref = None
        for ls in solvers:
            m = M.load(examples / ("plate.json" if name == "convection only" else name))
            if name == "convection only":
                m["models"][0]["modules"][0]["conditions"].pop(0)        # (the temperature imposed)
            for mdl in m["models"]:
                mdl.setdefault("solver", {})["linear_solver"] = ls
            with tempfile.TemporaryDirectory() as tmp:
                r = runner.run(m, tmp)
            assert r.ok, (name, ls, r.error)
            mr = next(iter(r.results["models"].values()))
            q = {k: v for k, v in mr.items() if k.startswith("max_")}
            if ref is None:
                ref = q
            assert all(abs(q[k] - ref[k]) <= 1e-6 * max(abs(ref[k]), 1e-12) for k in ref), (name, ls, q, ref)
            if ls == "mumps" and not getfem.util_mumps_linked():
                assert mr["linear_solver"] == "superlu" and any("MUMPS" in w for w in mr["warnings"])
            elif ls != "auto":
                assert mr["linear_solver"] == ls
    m = M.load(examples / "plate.json")             # an iterative solver which does not converge: said so
    m["models"][0]["solver"] = {"linear_solver": "cg/ildlt"}    # (a non symmetric system)
    with tempfile.TemporaryDirectory() as tmp:
        r = runner.run(m, tmp)
    assert not r.ok and "the iterative linear solver cg/ildlt did not converge" in r.error, r.error


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
