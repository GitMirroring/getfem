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

"""The explicit scheme (Newmark, beta = 0): the acceleration solved with
the mass matrix (consistent or lumped), the stability limit checked, the
Dirichlet values and their time derivatives, the second order, an explicit
variable with implicit ones in the same model; validation and editing.

    python -m pytest tests/test_explicit.py      (or: python tests/test_explicit.py)
"""

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from getfem_gui import editing as E, model as M, runner  # noqa: E402
import test_elastodynamics as B  # noqa: E402
import test_pde as P  # noqa: E402

CENTRAL = {"type": "newmark", "beta": 0.0, "gamma": 0.5}
AVERAGE = {"type": "newmark", "beta": 0.25, "gamma": 0.5}


def _getfem():
    try:
        import getfem  # noqa: F401
        return True
    except ImportError:
        return False


def run(m, name):
    with tempfile.TemporaryDirectory() as tmp:
        r = runner.run(m, tmp)
    return r, (r.results["models"][name] if r.ok else None)


def moving_boundary(dt, scheme=CENTRAL):
    """The wave equation, u = (x + y) sin(2t) (a source, the value imposed
    on the boundary, the initial velocity): the errors on u, Dot_u, Dot2_u."""
    m, md, mod = P.pde(8, 2)
    P.option(m, "time_order", "2")
    md["variables"][0]["initial_velocity"] = "2*(x+y)"
    mod["sources"] = [{"selection": "all", "value": "-4*(x+y)*sin(2*t)"}]
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": "(x+y)*sin(2*t)"}]
    md["outputs"] = [{"name": "eu", "expr": "sqr(u - (x+y)*sin(2*t))", "selection": "all"},
                     {"name": "ev", "expr": "sqr(Dot_u - 2*(x+y)*cos(2*t))", "selection": "all"},
                     {"name": "ea", "expr": "sqr(Dot2_u + 4*(x+y)*sin(2*t))", "selection": "all"}]
    m["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"],
                     "time": {"start": 0, "end": 1.0, "step": dt}, "schemes": {"pde/u": dict(scheme)}}]
    assert M.validate(m) == [], M.validate(m)
    r, R = run(m, "pde")
    assert r.ok, r.error
    return [max(np.sqrt(R["series"][k])) for k in ("eu", "ev", "ea")], R


def test_validation_and_editing():
    """beta = 0 accepted (explicit), its mass matrix offered; no multiplier
    nor normal condition on an explicit variable."""
    m = B.beam(dict(CENTRAL))
    assert M.validate(m) == [] and M.is_explicit(m["solvers"][0]["schemes"]["beam/u"])
    assert M.scheme_label(CENTRAL) == "explicit Newmark (central difference)"
    f = next(x for x in E.fields(m, ("solvers", 0)) if x.key[-1] == "mass")
    E.apply(m, f, "lumped")
    assert m["solvers"][0]["schemes"]["beam/u"]["mass"] == "lumped" and M.validate(m) == []
    f = next(x for x in E.fields(m, ("solvers", 0)) if x.key[-1] == "beta")
    E.apply(m, f, 0.25)
    assert "mass" not in m["solvers"][0]["schemes"]["beam/u"]
    assert not any(x.key[-1] == "mass" for x in E.fields(m, ("solvers", 0)))
    m = B.beam(dict(CENTRAL, mass="diagonal"))
    assert any("mass: consistent, lumped expected" in e for e in M.validate(m))
    m = B.beam(dict(CENTRAL))
    m["models"][0]["conditions"] = [{"type": "dirichlet", "variable": "u", "selection": "xmax",
                                     "value": [0, 0], "method": "multiplier"}]
    assert any("explicit, u cannot have the condition 1" in e for e in M.validate(m))
    m["models"][0]["conditions"][0]["method"] = "simplification"
    assert M.validate(m) == []


def test_stability_limit():
    """The time step checked at the start: too large, an error with the
    number of steps needed; below, the stable time step reported."""
    if not _getfem():
        return
    r, _ = run(B.beam(dict(CENTRAL), end=0.2, step=0.05), "beam")
    assert not r.ok and "exceeds its stability limit" in r.error and "time steps" in r.error
    r, R = run(B.beam(dict(CENTRAL), end=0.02, step=0.002), "beam")
    assert r.ok and 0.002 < R["stable_time_step"]["u"] < 0.05


def test_beam():
    """The cantilever released from a bent shape: the explicit scheme
    follows the implicit one (a small time step), its energy oscillates
    without drift."""
    if not _getfem():
        return
    _, ref = run(B.beam(dict(AVERAGE), end=1.0, step=0.001), "beam")
    r, R = run(B.beam(dict(CENTRAL), end=1.0, step=0.002), "beam")
    assert r.ok, r.error
    tip, rt = np.array(R["series"]["tip"]), np.array(ref["series"]["tip"])[::2]
    assert np.abs(tip - rt).max() < 3e-3 * np.abs(rt).max()
    e = np.array(R["series"]["kinetic"]) + np.array(R["series"]["potential"])
    assert np.abs(e / e[0] - 1).max() < 3e-3


def test_moving_boundary_second_order():
    """u, Dot_u, Dot2_u converge at the second order, a value imposed on
    the boundary depending on the time; as close as the implicit scheme."""
    if not _getfem():
        return
    e1, R = moving_boundary(0.01)
    e2, _ = moving_boundary(0.005)
    for a, b in zip(e1, e2):
        assert 3.5 < a / b < 4.5, (e1, e2)
    assert e1[0] < 1e-4 and e1[1] < 5e-4
    implicit, _ = moving_boundary(0.01, AVERAGE)
    assert e1[0] < 2 * implicit[0]


def test_lumped_mass():
    """Lumped (P1): the stable time step larger, the same accuracy; with P2
    on simplices, an error."""
    if not _getfem():
        return

    def standing(sc, degree, dt):
        m, md, mod = P.pde(16, degree)
        P.option(m, "time_order", "2")
        md["variables"][0]["initial"] = P.EXACT
        mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
        md["outputs"] = [{"name": "err", "expr": f"sqr(u - {P.EXACT}*cos(sqrt(2)*pi*t))", "selection": "all"}]
        m["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"],
                         "time": {"start": 0, "end": 0.3, "step": dt}, "schemes": {"pde/u": sc}}]
        return run(m, "pde")
    (r1, consistent), (r2, lumped) = standing(dict(CENTRAL), 1, 0.004), standing(dict(CENTRAL, mass="lumped"), 1, 0.004)
    assert r1.ok and r2.ok
    assert lumped["stable_time_step"]["u"] > 1.5 * consistent["stable_time_step"]["u"]
    assert max(np.sqrt(lumped["series"]["err"])) < 1e-2 and max(np.sqrt(consistent["series"]["err"])) < 1e-2
    r, _ = standing(dict(CENTRAL, mass="lumped"), 2, 0.002)
    assert not r.ok and "lumped mass of u is not positive" in r.error


def test_with_implicit_variables():
    """An explicit wave u and an implicit diffusion w (Crank-Nicolson) in
    the same model, coupled both ways (nonlinear): the same as with both
    implicit."""
    if not _getfem():
        return
    m, md, mod = P.pde(8)
    P.option(m, "time_order", "2")
    md["variables"][0]["initial"] = P.EXACT
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    path = E.do_action(m, ("models", 0, "modules"), "add_module:pde")
    second = E.get_at(m, path)
    P.option(m, "time_order", "1", 1)
    second["sources"] = [{"selection": "all", "value": "sqr(Dot_u)"}]      # heated by the velocity of u
    second["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    mod["sources"] = [{"selection": "all", "value": "-u2"}]
    md["outputs"] = [{"name": "U", "expr": "sqr(u)", "selection": "all"},
                     {"name": "W", "expr": "u2", "selection": "all"}]
    out = []
    for sc in (CENTRAL, AVERAGE):
        mm = copy.deepcopy(m)
        mm["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"],
                          "time": {"start": 0, "end": 0.5, "step": 0.005},
                          "schemes": {"pde/u": dict(sc), "pde/u2": {"type": "theta", "theta": 0.5}}}]
        assert M.validate(mm) == []
        r, R = run(mm, "pde")
        assert r.ok, r.error
        out.append((np.array(R["series"]["U"]), np.array(R["series"]["W"])))
    (u1, w1), (u2, w2) = out
    assert np.abs(u1 - u2).max() < 3e-3 * u2.max() and np.abs(w1 - w2).max() < 3e-3 * w2.max() and w2[-1] > 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
