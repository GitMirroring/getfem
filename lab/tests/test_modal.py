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

"""The modal solver: the eigenmodes K phi = lambda M phi of a model (ARPACK
by SciPy). The decay rates of the heat in a square, pi^2 (m^2 + n^2); the
frequencies of a membrane (PDE of order 2); a cantilever beam against
Euler-Bernoulli, its effective masses; a free beam (its rigid modes at 0);
a modal solver after a static one (the tangent at the state left); the
tuning fork; editing, validation.

    python -m pytest tests/test_modal.py      (or: python tests/test_modal.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, equations, model as M, runner  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
PI2 = math.pi ** 2


def _getfem():
    try:
        import getfem  # noqa: F401
        return True
    except ImportError:
        return False


def run(m, name):
    assert M.validate(m) == [], M.validate(m)
    with tempfile.TemporaryDirectory() as tmp:
        r = runner.run(m, tmp)
        assert r.ok, f"{r.error}\n{r.stderr[-3000:]}"
        for f in r.results["models"][name].get("series", {}).get("vtu", []):
            assert (Path(tmp) / f).exists()
    return r.results


def modal(m, modes=4, **kw):
    m["solvers"] = [dict({"name": "modes", "type": "modal", "steps": [m["models"][0]["name"]],
                          "modes": modes}, **kw)]
    return m


def heat(n=16):
    """The unit square, T = 0 on its boundary, rho cp = k = 1 (P2)."""
    m = M.default_model()
    m["meshes"][0]["divisions"] = [n, n]
    md = m["models"][0]
    md["variables"][0]["degree"] = 2
    mod = md["modules"][0]
    mod["conditions"] = [{"type": "temperature", "selection": "boundary", "value": 0.0}]
    for p in ("rho", "cp"):
        mod["properties"][p] = [{"selection": "all", "value": 1.0}]
    return m


def beam(fixed=True, nx=100, ny=5, L=1.0, h=0.05):
    """A steel beam (plane stress, P2), clamped at xmin or free."""
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "box", "origin": [0, 0], "size": [L, h]}
    m["meshes"][0]["divisions"] = [nx, ny]
    md = m["models"][0]
    md.update(name="beam", vtu="beam.vtu")
    md["variables"] = [M.new_variable("u", "mesh", M.VECTOR, 2)]
    el = M.new_module("linear_elasticity", 2)
    el["plane"] = "stress"
    for p, v in (("E", 2e11), ("nu", 0.3), ("rho", 7800.0)):
        el["properties"][p] = [{"selection": "all", "value": v}]
    el["conditions"] = [{"type": "fixed", "selection": "xmin"}] if fixed else []
    md["modules"], md["outputs"] = [el], []
    return modal(m, 6)


def test_editing_and_validation():
    """The solver type modal: its settings, the transient properties given;
    its errors; the variables of a lower order quasi-static."""
    m = M.default_model()
    path = E.do_action(m, ("solvers",), "add_solver")
    f = next(x for x in E.fields(m, path) if x.key[-1] == "type")
    E.apply(m, f, "modal")
    s = m["solvers"][0]
    assert (s["modes"], s["shift"], s["steps"]) == (6, 0.0, ["heat"])
    mod = m["models"][0]["modules"][0]
    assert "rho" in mod["properties"] and "cp" in mod["properties"] and M.validate(m) == []
    labels = [x.label for x in E.fields(m, path)]
    assert "Number of modes" in labels and "Shift: decay rate (1/s)" in labels and "Model" in labels
    assert "<b>Eigenmodes</b>" in equations.equation(m, m["models"][0], mod)
    for key, value, error in (("modes", 0, "modes"), ("shift", -1.0, "shift"),
                              ("steps", ["heat", "heat2"], "unknown model")):
        bad = copy.deepcopy(m)
        bad["solvers"][0][key] = value
        assert any(error in e for e in M.validate(bad)), (key, M.validate(bad))
    m2 = copy.deepcopy(m)
    del mod["properties"]["rho"]
    assert any("rho" in e for e in M.validate(m))
    E.apply(m2, f, "sequence")
    assert "modes" not in m2["solvers"][0] and "shift" not in m2["solvers"][0]
    # no time derivative: no eigenmodes
    es = M.default_model()
    es["models"][0]["modules"] = [M.new_module("electrostatics", 2)]
    es["models"][0]["modules"][0]["conditions"] = [{"type": "ground", "selection": "xmin"}]
    es["models"][0]["variables"] = [M.new_variable("V", "mesh", M.SCALAR, 1)]
    assert any("no time derivative" in e for e in M.validate(modal(es)))
    b = beam()                                 # (a temperature with it: quasi-static, cp not needed)
    b["models"][0]["variables"].append(M.new_variable("T", "mesh", M.SCALAR, 1))
    b["models"][0]["modules"].append(M.new_module("heat_conduction", 2))
    b["models"][0]["modules"][1]["conditions"] = [{"type": "temperature", "selection": "xmin", "value": 0}]
    assert M.validate(b) == [] and M.modal_variables(b["models"][0]) == ["u"]
    assert "<b>Stationary</b> for the eigenmodes" in equations.equation(b, b["models"][0],
                                                                        b["models"][0]["modules"][1])


def test_heat_decay_rates():
    """The decay rates pi^2 (m^2 + n^2): 2, 5, 5, 8 times pi^2; the time
    constants their inverses; the modes in a series (labels, VTU files)."""
    if not _getfem():
        return
    r = run(modal(heat()), "heat")
    R = r["models"]["heat"]
    for x, k in zip(R["eigenvalues"], (2, 5, 5, 8)):
        assert abs(x / (k * PI2) - 1) < 1e-3
    assert abs(R["time_constants"][0] * 2 * PI2 - 1) < 1e-3
    S = R["series"]
    assert S["t"] == [1, 2, 3, 4] and S["labels"][0].startswith("mode 1: rate 19.7")
    assert len(S["vtu"]) == 4 and R["vtu"] == S["vtu"][0] and r["solver"]["type"] == "modal"
    assert "ARPACK" in R["eigensolver"]
    assert max(S["max_T"]) <= 1 + 1e-12                   # (scaled: a largest value 1)


def test_membrane():
    """The PDE of order 2, m u'' - div(grad u) = 0 on the unit square, u = 0
    on its boundary: the frequencies sqrt(m^2 + n^2)/2."""
    if not _getfem():
        return
    m = M.default_model()
    m["meshes"][0]["divisions"] = [16, 16]
    md = m["models"][0]
    md.update(name="pde", vtu="pde.vtu")
    md["variables"] = [M.new_variable("u", "mesh", M.SCALAR, 2)]
    mod = M.new_module("pde", 2)
    md["modules"] = [mod]
    E.apply(m, next(x for x in E.fields(m, ("models", 0, "modules", 0)) if x.key[-1] == "time_order"), "2")
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    R = run(modal(m, 5), "pde")["models"]["pde"]
    for f, k in zip(R["frequencies"], (2, 5, 5, 8, 10)):
        assert abs(2 * f / math.sqrt(k) - 1) < 2e-4


def test_cantilever():
    """The clamped beam (L = 1, h = 0.05: slender): its bending frequencies
    those of Euler-Bernoulli, (beta L)^2/(2 pi) sqrt(E h^2/(12 rho L^4)),
    within 1.5 % then 3 % (shear, rotary inertia); its first axial mode
    sqrt(E/rho)/(4 L); the effective masses of the bending modes along y,
    61 %, 19 %, 6.5 % of the mass; with a temperature (first order:
    quasi-static), the same; a free beam: three rigid modes at 0."""
    if not _getfem():
        return
    R = run(beam(), "beam")["models"]["beam"]
    c = math.sqrt(2e11 * 0.05 ** 2 / (12 * 7800)) / (2 * math.pi)
    f = R["frequencies"]
    assert abs(f[0] / (1.8751 ** 2 * c) - 1) < 0.015 and abs(f[1] / (4.6941 ** 2 * c) - 1) < 0.03
    assert abs(f[2] / (7.8548 ** 2 * c) - 1) < 0.04
    assert abs(f[3] / (math.sqrt(2e11 / 7800) / 4) - 1) < 0.01                    # (axial)
    my = [x / R["mass"]["y"] for x in R["effective_mass"]["y"]]
    assert abs(my[0] - 0.613) < 0.005 and abs(my[1] - 0.188) < 0.005 and abs(my[2] - 0.065) < 0.005
    assert R["effective_mass"]["x"][3] / R["mass"]["x"] > 0.8 and R["effective_mass"]["x"][0] < 1e-8
    b = beam()                               # (with a temperature: quasi-static, the same modes)
    b["models"][0]["variables"].append(M.new_variable("T", "mesh", M.SCALAR, 1))
    hm = M.new_module("heat_conduction", 2)
    hm["conditions"] = [{"type": "temperature", "selection": "xmin", "value": 0.0}]
    b["models"][0]["modules"].append(hm)
    Rt = run(b, "beam")["models"]["beam"]
    assert max(abs(x / y - 1) for x, y in zip(Rt["frequencies"], f)) < 1e-9
    R = run(beam(False, 40, 2), "beam")["models"]["beam"]
    assert R["frequencies"][:3] == [0.0, 0.0, 0.0] and R["frequencies"][3] > 200
    assert all(abs(x) < 1e-6 * R["eigenvalues"][3] for x in R["eigenvalues"][:3])


def test_after_a_static_solver():
    """k = 1 + T, T = 1 on the boundary: from the initial values (T = 0)
    the rates 2 pi^2 k = 2 pi^2; after the static solver (T = 1: k = 2,
    grad T = 0) 4 pi^2; the state left unchanged by the modal solver."""
    if not _getfem():
        return
    m = heat()
    mod = m["models"][0]["modules"][0]
    mod["properties"]["k"] = [{"selection": "all", "value": "1 + T"}]
    mod["conditions"][0]["value"] = 1.0
    m["solvers"] = [{"name": "static", "type": "sequence", "steps": ["heat"]},
                    {"name": "modes", "type": "modal", "steps": ["heat"], "modes": 2, "start": "previous"},
                    {"name": "again", "type": "sequence", "steps": ["heat"], "start": "previous"},
                    {"name": "cold", "type": "modal", "steps": ["heat"], "modes": 2}]
    r = run(m, "heat")["solvers"]
    assert abs(r["modes"]["models"]["heat"]["eigenvalues"][0] / (4 * PI2) - 1) < 1e-3
    assert abs(r["cold"]["models"]["heat"]["eigenvalues"][0] / (2 * PI2) - 1) < 1e-3
    assert abs(r["again"]["models"]["heat"]["min_T"] - 1) < 1e-8


def test_tuning_fork():
    """The example: its fork mode (the second one, the prongs in
    opposition: no effective mass) at 440 Hz; the first one sideways."""
    if not _getfem():
        return
    R = run(M.load(EXAMPLES / "tuning_fork.json"), "fork")["models"]["fork"]
    assert abs(R["frequencies"][1] - 440) < 2 and R["frequencies"][0] < 300
    mx = [x / R["mass"]["x"] for x in R["effective_mass"]["x"]]
    assert mx[1] < 1e-6 and mx[0] > 0.5 and R["warp"] == "u"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
