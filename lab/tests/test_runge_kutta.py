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

"""The explicit Runge-Kutta schemes (RK2 at first order, RK4 at first and
second order): their stability limits, the order of convergence (a value
imposed on the boundary depending on the time, a source), a wave and a
diffusion coupled in one model (both RK4) as with implicit schemes, a
temperature RK with a quasi-static displacement in its model; validation
and editing.

    python -m pytest tests/test_runge_kutta.py      (or: python tests/test_runge_kutta.py)
"""

import copy
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from getfem_lab import editing as E, equations, model as M  # noqa: E402
import test_elastodynamics as B  # noqa: E402
import test_explicit as X  # noqa: E402
import test_pde as P  # noqa: E402

RK2, RK4 = {"type": "rk2"}, {"type": "rk4"}


def diffusion(dt, scheme, end=0.5):
    """u' - div(grad u) = f, u = (x + y) sin(2t) (P1: exact in space): the
    errors on u, Dot_u."""
    m, md, mod = P.pde(4, 1)
    P.option(m, "time_order", "1")
    mod["sources"] = [{"selection": "all", "value": "2*(x+y)*cos(2*t)"}]
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": "(x+y)*sin(2*t)"}]
    md["outputs"] = [{"name": "eu", "expr": "sqr(u - (x+y)*sin(2*t))", "selection": "all"},
                     {"name": "ev", "expr": "sqr(Dot_u - 2*(x+y)*cos(2*t))", "selection": "all"}]
    m["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"],
                     "time": {"start": 0, "end": end, "step": dt}, "schemes": {"pde/u": dict(scheme)}}]
    assert M.validate(m) == [], M.validate(m)
    r, R = X.run(m, "pde")
    assert r.ok, r.error
    return [max(np.sqrt(R["series"][k])) for k in ("eu", "ev")], R


def test_validation_and_editing():
    """RK2 and RK4 offered at first order, RK4 only at second order; their
    stability limits (the real axis: 2, 2.785; the imaginary one: 2
    sqrt(2)); the other variables in time of the model with the same
    scheme, the model not solved by partial steps; the label, the equation."""
    assert M.schemes_for(1)[-2:] == ["rk2", "rk4"] and "rk2" not in M.schemes_for(2)
    assert abs(M.rk_limit(RK2, 1) - 2) < 1e-9 and abs(M.rk_limit(RK4, 1) - 2.7852935634) < 1e-9
    assert abs(M.rk_limit(RK4, 2) - 2 * math.sqrt(2)) < 1e-9
    assert M.is_explicit(RK4) and M.is_runge_kutta(RK2) and not M.is_runge_kutta(X.CENTRAL)
    assert M.scheme_label(RK2) == "RK2 (Heun)" and M.scheme_label(dict(RK4, mass="lumped")) == "RK4, lumped mass"
    m = B.beam(dict(RK4))
    assert M.validate(m) == []
    assert "RK4; from the initial value and velocity" in equations.equation(
        m, m["models"][0], m["models"][0]["modules"][0])
    f = next(x for x in E.fields(m, ("solvers", 0)) if x.key[-1] == "type" and "schemes" in x.key)
    E.apply(m, f, "rk4")
    assert m["solvers"][0]["schemes"]["beam/u"] == {"type": "rk4", "mass": "consistent"}
    assert any(x.key[-1] == "mass" for x in E.fields(m, ("solvers", 0)))
    assert not any(x.key[-1] == "initial_acceleration" for x in E.fields(m, ("solvers", 0)))
    m, md, mod = P.pde(4)
    P.option(m, "time_order", "2")
    second = E.get_at(m, E.do_action(m, ("models", 0, "modules"), "add_module:pde"))
    P.option(m, "time_order", "1", 1)
    second["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    m["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"],
                     "time": {"start": 0, "end": 0.1, "step": 0.01},
                     "schemes": {"pde/u": dict(RK4), "pde/u2": {"type": "theta", "theta": 1.0}}}]
    assert any("other variables in time of 'pde' (u2) with the same scheme" in e for e in M.validate(m))
    m["solvers"][0]["schemes"]["pde/u2"] = {"type": "stationary"}
    assert M.validate(m) == []
    m["solvers"][0]["schemes"]["pde/u2"] = dict(RK4)
    assert M.validate(m) == []
    m["solvers"][0]["steps"] = ["pde/u", "pde/u2"]
    assert any("solved as a whole" in e for e in M.validate(m))


def test_first_order():
    """RK2 converges at the second order, RK4 at the fourth (u; Dot_u, the
    derivative from the equation at the saved time); above the stability
    limit, an error with the number of steps needed."""
    if not X._getfem():
        return
    (a1, v1), R = diffusion(0.004, RK2)
    (a2, v2), _ = diffusion(0.002, RK2)
    assert 3.6 < a1 / a2 < 4.6 and 3.0 < v1 / v2 < 6.0 and a1 < 2e-5
    assert abs(R["stable_time_step"]["u"] - 0.004884) < 1e-5
    (b1, w1), R = diffusion(0.004, RK4)
    (b2, w2), _ = diffusion(0.002, RK4)
    assert 13 < b1 / b2 < 26 and w1 / w2 > 13 and b1 < 1e-7
    assert abs(R["stable_time_step"]["u"] / 0.004884 - 2.7853 / 2) < 1e-3   # (the same lambda_max)
    m, md, mod = P.pde(4, 1)
    P.option(m, "time_order", "1")
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    m["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"],
                     "time": {"start": 0, "end": 0.5, "step": 0.02}, "schemes": {"pde/u": dict(RK4)}}]
    r, _ = X.run(m, "pde")
    assert not r.ok and "exceeds its stability limit" in r.error and "at least 74 time steps" in r.error


def test_second_order():
    """RK4 on the wave equation with a moving boundary: u, Dot_u, Dot2_u at
    the fourth order; more accurate than the central difference."""
    if not X._getfem():
        return
    e1, R = X.moving_boundary(0.02, RK4)
    e2, _ = X.moving_boundary(0.01, RK4)
    assert 13 < e1[0] / e2[0] < 19 and e1[1] / e2[1] > 13 and e1[2] / e2[2] > 12, (e1, e2)
    assert e2[0] < 2e-7 and 0.02 < R["stable_time_step"]["u"] < 0.04
    central, _ = X.moving_boundary(0.01)
    assert e2[0] < 1e-2 * central[0]


def test_coupled_wave_and_diffusion():
    """A wave u (second order) and a diffusion u2 heated by its velocity,
    coupled both ways in one model, both RK4: as both implicit."""
    if not X._getfem():
        return
    m, md, mod = P.pde(8)
    P.option(m, "time_order", "2")
    md["variables"][0]["initial"] = P.EXACT
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    second = E.get_at(m, E.do_action(m, ("models", 0, "modules"), "add_module:pde"))
    P.option(m, "time_order", "1", 1)
    second["sources"] = [{"selection": "all", "value": "sqr(Dot_u)"}]
    second["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    mod["sources"] = [{"selection": "all", "value": "-u2"}]
    md["outputs"] = [{"name": "U", "expr": "sqr(u)", "selection": "all"},
                     {"name": "W", "expr": "u2", "selection": "all"}]
    out = []
    for schemes in ((X.AVERAGE, {"type": "theta", "theta": 0.5}), (RK4, RK4)):
        mm = copy.deepcopy(m)
        mm["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"],
                          "time": {"start": 0, "end": 0.1, "step": 0.00025},
                          "schemes": {"pde/u": dict(schemes[0]), "pde/u2": dict(schemes[1])}}]
        assert M.validate(mm) == []
        r, R = X.run(mm, "pde")
        assert r.ok, r.error
        out.append((np.array(R["series"]["U"]), np.array(R["series"]["W"])))
    (u1, w1), (u2, w2) = out
    assert np.abs(u1 - u2).max() < 1e-5 * u1.max() and np.abs(w1 - w2).max() < 1e-3 * w1.max() and w1[-1] > 0.01


def test_quasi_static_displacement():
    """The thermal shock of the strip in one model, its temperature RK2 or
    RK4, its displacement stationary (solved at each stage): as with BDF2."""
    if not X._getfem():
        return

    def strip(scheme):
        m = M.load(Path(__file__).resolve().parents[1] / "examples" / "thermal_shock_strip.json")
        s = m["solvers"][0]
        s["time"], s["save"] = {"start": 0, "end": 3e-6, "step": 1.5e-8}, {"every": 20}
        thermal, mech = m["models"]
        mech.pop("imports")
        for k in ("variables", "modules", "outputs"):
            thermal[k] += mech[k]
        m["models"], s["steps"] = [thermal], ["thermal"]
        s["schemes"] = {"thermal/T": dict(scheme), "thermal/u": {"type": "stationary"}}
        assert M.validate(m) == [], M.validate(m)
        r, R = X.run(m, "thermal")
        assert r.ok, r.error
        return np.array(R["series"]["tip_deflection"])
    ref = strip({"type": "bdf2"})
    assert np.abs(ref).max() > 0
    for sc in (RK2, RK4):
        assert np.abs(strip(sc) - ref).max() < 1e-5 * np.abs(ref).max(), sc


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
