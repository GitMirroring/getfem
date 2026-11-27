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

"""The electrostatics module: -div(eps0 eps_r grad V) = rho_v; potential,
ground, surface charge, floating conductor; the charges of the
conductors, the energy, the fields; the coaxial cable (its capacitance
per unit length, 2 pi eps0 eps_r / ln(b/a)); editing, validation.

    python -m pytest tests/test_electrostatics.py      (or: python tests/test_electrostatics.py)
"""

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, equations, library, model as M, runner  # noqa: E402

EPS0 = 8.8541878128e-12
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _getfem():
    try:
        import getfem  # noqa: F401
        return True
    except ImportError:
        return False


def plates(eps_r=2.0, conditions=None, source=None):
    """The unit square between two plates (xmin, xmax)."""
    m = M.default_model()
    md = m["models"][0]
    md.update(name="es", vtu="es.vtu")
    md["variables"] = [M.new_variable("V", "mesh", M.SCALAR, 2)]
    mod = M.new_module("electrostatics", 2)
    mod["properties"]["eps_r"] = [{"selection": "all", "value": eps_r}]
    mod["conditions"] = conditions or [{"type": "ground", "selection": "xmin"},
                                       {"type": "potential", "selection": "xmax", "value": 1.0}]
    if source is not None:
        mod["sources"] = [{"selection": "all", "value": source}]
    md["modules"], md["outputs"] = [mod], []
    return m


def run(m, name="es"):
    assert M.validate(m) == [], M.validate(m)
    with tempfile.TemporaryDirectory() as tmp:
        r = runner.run(m, tmp)
    assert r.ok, f"{r.error}\n{r.stderr[-3000:]}"
    return r.results["models"][name]


def test_editing_and_validation():
    m = plates()
    md, mod = m["models"][0], m["models"][0]["modules"][0]
    assert "−div(ε<sub>0</sub> ε<sub>r</sub> ∇V) = ρ<sub>v</sub>" in equations.equation(m, md, mod)
    labels = [f.label for f in E.fields(m, ("models", 0, "modules", 0))]
    assert "eps_r (relative permittivity)" in labels and "rho_v (volume charge density, C/m³)" in labels
    assert ("add_source", "Add a domain for rho_v") in E.actions(m, ("models", 0, "modules", 0))
    path = E.do_action(m, ("models", 0, "modules", 0), "add_bc")
    f = next(x for x in E.fields(m, path) if x.key[-1] == "type")
    assert [k for k, _ in f.choices] == ["potential", "ground", "surface_charge", "floating"]
    E.apply(m, f, "floating")
    assert E.get_at(m, path)["charge"] == 0.0
    mod["conditions"] = [{"type": "floating", "selection": "xmax", "charge": 0.0}]
    assert any("only defined up to a constant" in e for e in M.validate(m))
    m["parameters"] = [{"name": "eps0", "value": 1.0}]               # (the permittivity of vacuum: taken)
    assert M.validate(m)
    eps = {m["name"]: m["properties"].get("eps_r") for _, m in library.materials(library.LIBRARY)}
    assert eps["Water (20 °C)"] == 80.0 and eps["Polyethylene (HDPE)"] == 2.3 and eps["Copper"] is None


def test_plates():
    """Two plates at 0 and 1 V: the capacitance eps0 eps_r per unit area
    and depth (the charges +-C V, the energy C V^2/2, E = -1 along x);
    anisotropic, the same along x."""
    if not _getfem():
        return
    r = run(plates())
    C = 2 * EPS0
    assert abs(r["charges"]["xmax"] / C - 1) < 1e-10 and abs(r["charges"]["xmin"] / C + 1) < 1e-10
    assert abs(r["electric_energy"] / (C / 2) - 1) < 1e-10 and (r["min_V"], r["max_V"]) == (0.0, 1.0)
    m = plates()
    E.apply(m, next(x for x in E.fields(m, ("models", 0, "modules", 0)) if x.key[-1] == "permittivity"),
            "anisotropic")
    m["models"][0]["modules"][0]["properties"]["eps_r"][0]["value"] = "[2, 0; 0, 7]"
    assert abs(run(m)["charges"]["xmax"] / C - 1) < 1e-10


def test_charges():
    """A volume charge (V = rho x (1 - x) / (2 eps0 eps_r)), a surface charge
    (E = sigma/(eps0 eps_r) along the plate), a floating conductor of charge
    Q (V = Q/C) and of charge 0 (its potential that of the conditions)."""
    if not _getfem():
        return
    rho = 1e-9
    r = run(plates(source=rho, conditions=[{"type": "ground", "selection": "xmin"},
                                            {"type": "ground", "selection": "xmax"}]))
    assert abs(r["max_V"] / (rho / (8 * 2 * EPS0)) - 1) < 1e-8
    assert abs(r["charges"]["xmin"] + rho / 2) < 1e-12 * rho          # (each plate: half of -rho)
    sigma = 1e-11
    r = run(plates(conditions=[{"type": "ground", "selection": "xmin"},
                               {"type": "surface_charge", "selection": "xmax", "value": sigma}]))
    assert abs(r["max_V"] / (sigma / (2 * EPS0)) - 1) < 1e-8 and abs(r["charges"]["xmin"] + sigma) < 1e-8 * sigma
    Q = 1e-11
    r = run(plates(conditions=[{"type": "ground", "selection": "xmin"},
                               {"type": "floating", "selection": "xmax", "charge": Q}]))
    assert abs(r["potentials"]["xmax"] / (Q / (2 * EPS0)) - 1) < 1e-10 and abs(r["charges"]["xmax"] / Q - 1) < 1e-10
    r = run(plates(conditions=[{"type": "potential", "selection": "xmin", "value": 3.0},
                               {"type": "floating", "selection": "xmax", "charge": 0.0}]))
    assert abs(r["potentials"]["xmax"] - 3.0) < 1e-10 and abs(r["min_V"] - 3.0) < 1e-10


def test_coax_cable():
    """The coaxial cable (polyethylene, a = 1 mm, b = 3.5 mm): its capacitance
    per unit length, from the energy and from the charges (the equation
    tested by 1 on the conductor: as accurate), 2 pi eps0 eps_r / ln(b/a)."""
    if not _getfem():
        return
    r = run(M.load(EXAMPLES / "coax_cable.json"), "cable")
    exact = 2 * math.pi * EPS0 * 2.3 / math.log(3.5)
    assert abs(r["quantities"]["capacitance"] / exact - 1) < 1e-4
    assert abs(r["charges"]["inner"] / exact - 1) < 1e-4 and abs(r["charges"]["outer"] / exact + 1) < 1e-4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
