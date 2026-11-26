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

"""The electric currents module: -div(sigma grad V) = Q_j; potential,
ground, current density, terminal; the currents of the electrodes, the
Joule power; the Joule heating of a heat conduction module (the exact
parabola, a conductivity depending on the temperature); the busbar
example; editing, validation (sigma_e: the name sigma is the stress of the elasticity).

    python -m pytest tests/test_electric_currents.py      (or: python tests/test_electric_currents.py)
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, equations, library, model as M, runner  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _getfem():
    try:
        import getfem  # noqa: F401
        return True
    except ImportError:
        return False


def strip(conditions, sigma=2.0):
    """The unit square, a conductor between two electrodes (xmin, xmax)."""
    m = M.default_model()
    md = m["models"][0]
    md.update(name="ec", vtu="ec.vtu")
    md["variables"] = [M.new_variable("V", "mesh", M.SCALAR, 2)]
    mod = M.new_module("electric_currents", 2)
    mod["properties"]["sigma_e"] = [{"selection": "all", "value": sigma}]
    mod["conditions"] = conditions
    md["modules"], md["outputs"] = [mod], []
    return m


def joule(sigma=3.0):
    """Heat conduction (k = 4, T = 0 on xmin and xmax) heated by the current
    of the electric currents module (V = 0 and 2 on xmin and xmax)."""
    m = M.default_model()
    md = m["models"][0]
    md.update(name="jh", vtu="jh.vtu")
    heat = md["modules"][0]
    heat["conditions"] = [{"type": "temperature", "selection": "xmin", "value": 0.0},
                          {"type": "temperature", "selection": "xmax", "value": 0.0}]
    heat["sources"] = []
    heat["properties"]["k"] = [{"selection": "all", "value": 4.0}]
    ec = E.get_at(m, E.do_action(m, ("models", 0, "modules"), "add_module:electric_currents"))
    ec["properties"]["sigma_e"] = [{"selection": "all", "value": sigma}]
    ec["conditions"] = [{"type": "ground", "selection": "xmin"},
                        {"type": "potential", "selection": "xmax", "value": 2.0}]
    ec["joule_heating"] = "heat_conduction"
    md["outputs"] = [{"name": "heat_out", "expr": "q.Normal", "selection": "boundary"}]
    return m


def run(m, name):
    assert M.validate(m) == [], M.validate(m)
    with tempfile.TemporaryDirectory() as tmp:
        r = runner.run(m, tmp)
    assert r.ok, f"{r.error}\n{r.stderr[-3000:]}"
    return r.results["models"][name]


def test_editing_and_validation():
    m = strip([{"type": "ground", "selection": "xmin"}])
    path = ("models", 0, "modules", 0)
    assert "−div(σ ∇V) = Q<sub>j</sub>" in equations.equation(m, m["models"][0], E.get_at(m, path))
    labels = [f.label for f in E.fields(m, path)]
    assert "sigma_e (electrical conductivity)" in labels and "Joule heating" in labels
    bc = E.do_action(m, path, "add_bc")
    f = next(x for x in E.fields(m, bc) if x.key[-1] == "type")
    assert [k for k, _ in f.choices] == ["potential", "ground", "current_density", "terminal"]
    E.apply(m, f, "terminal")
    assert E.get_at(m, bc)["current"] == 0.0
    assert "Total current I (A, entering)" in [x.label for x in E.fields(m, bc)]
    m["models"][0]["modules"][0]["conditions"] = [{"type": "terminal", "selection": "xmax", "current": 1.0}]
    assert any("only defined up to a constant" in e for e in M.validate(m))
    m = strip([{"type": "ground", "selection": "xmin"}])
    m["models"][0]["modules"][0]["joule_heating"] = "heat_conduction"      # (no heat module)
    assert any("heat" in e for e in M.validate(m))
    m = joule()                                                            # (no electrostatics with it)
    assert "add_module:electrostatics" not in dict(E.actions(m, ("models", 0, "modules")))
    m["models"][0]["modules"].append(M.new_module("electrostatics", 2))
    assert any("exclude each other" in e for e in M.validate(m))
    m = joule()
    assert M.validate(m) == [] and "heat conduction (T)" in equations.equation(
        m, m["models"][0], m["models"][0]["modules"][1])
    sigma = {m["name"]: m["properties"].get("sigma_e") for _, m in library.materials(library.LIBRARY)}
    assert sigma["Copper"] == 5.8e7 and sigma["Aluminium (pure)"] == 3.774e7 and sigma["Epoxy"] is None


def test_strip():
    """The current sigma V per unit depth through the strip (its two
    electrodes: -I and I), the Joule power sigma V^2; a terminal of current
    I: its potential I/sigma, with a conductivity of a metal."""
    if not _getfem():
        return
    r = run(strip([{"type": "ground", "selection": "xmin"},
                   {"type": "potential", "selection": "xmax", "value": 1.0}]), "ec")
    assert abs(r["currents"]["xmax"] - 2) < 1e-10 and abs(r["currents"]["xmin"] + 2) < 1e-10
    assert abs(r["joule_power"] - 2) < 1e-10 and abs(r["min_V"]) < 1e-12 and abs(r["max_V"] - 1) < 1e-12
    r = run(strip([{"type": "ground", "selection": "xmin"},
                   {"type": "terminal", "selection": "xmax", "current": 3.0}], 2e7), "ec")
    assert abs(r["potentials"]["xmax"] / 1.5e-7 - 1) < 1e-10 and abs(r["currents"]["xmax"] / 3 - 1) < 1e-10
    assert abs(r["joule_power"] / (3 * 1.5e-7) - 1) < 1e-10
    m = strip([{"type": "ground", "selection": "xmin"},                    # (a current density: J.n = j)
               {"type": "current_density", "selection": "xmax", "value": 4.0}])
    assert abs(run(m, "ec")["max_V"] - 2) < 1e-10


def test_joule_heating():
    """The Joule heat sigma E^2 = 12, uniform: the temperature the parabola
    of maximum sigma E^2/(8 k) = 0.375 (P2: exact), all the heat out; with
    sigma(T) = 3/(1 + T/2), the heat out the Joule power still (Newton)."""
    if not _getfem():
        return
    r = run(joule(), "jh")
    assert abs(r["max_T"] - 0.375) < 1e-10 and abs(r["joule_power"] - 12) < 1e-10
    assert abs(r["quantities"]["heat_out"] - 12) < 1e-9
    r = run(joule("3/(1 + 0.5*T)"), "jh")
    assert 0.3 < r["max_T"] < 0.375 and abs(r["quantities"]["heat_out"] / r["joule_power"] - 1) < 2e-3


def test_busbar():
    """The copper busbar (sigma 5.8e7 S/m, the residual scaled): its
    resistance per unit depth near L/(sigma H) (the hole and the heat add some 15 %),
    larger hot; the heat out by the convection small, most through the
    ends."""
    if not _getfem():
        return
    m = M.load(EXAMPLES / "busbar_joule.json")
    r = run(m, "busbar")
    R = 0.03 / r["currents"]["right"]
    assert 0.1 / (5.8e7 * 0.02) < R < 1.25 * 0.1 / (5.8e7 * 0.02)
    assert 30 < r["max_T"] < 40 and 0 < r["quantities"]["heat_to_air"] < 0.01 * r["joule_power"]
    next(p for p in m["parameters"] if p["name"] == "alpha_R")["value"] = 0.0
    r0 = run(m, "busbar")
    assert r0["currents"]["right"] > r["currents"]["right"] and r0["max_T"] > r["max_T"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
