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

"""Linear elasticity: end-to-end tests against exact solutions.

    python -m pytest tests/       (or: python tests/test_elasticity.py)
"""

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import model as M, runner  # noqa: E402

E, NU, S = 200e3, 0.3, 50.0      # Young's modulus, Poisson's ratio, applied stress


def elastic(dim=2, size=None, n=4, degree=2, plane=None, element="simplex"):
    size = size or [2.0, 1.0, 1.0][:dim]
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "box", "origin": [0.0] * dim, "size": size}
    m["meshes"][0].update(divisions=[n] * dim, element=element)
    m["discretization"]["degree"] = degree
    m["materials"] = [{"name": "steel", "selection": "all", "properties": {"E": E, "nu": NU}}]
    m["physics"] = {"type": "linear_elasticity", "mesh": "mesh", "variable": "u",
                    "sources": [], "boundary_conditions": []}
    if plane:
        m["physics"]["plane"] = plane
    return m


def bc(model, *conds):
    model["physics"]["boundary_conditions"] += [dict(c) for c in conds]
    return model


def error_quantity(model, exact):
    model["outputs"]["quantities"].append(
        {"name": "err2", "expr": f"Norm_sqr(u - {exact})", "selection": "all"})
    model["outputs"]["quantities"].append(
        {"name": "ref2", "expr": f"Norm_sqr({exact})", "selection": "all"})
    return model


def solve(model):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(model, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
        return res.results


def rel_error(r):
    q = r["quantities"]
    return math.sqrt(abs(q["err2"]) / q["ref2"])


def rollers(*names):
    return [{"type": "roller", "selection": n} for n in names]


# ------------------------------------------------------------------ uniaxial tension

def uniaxial(dim, plane):
    m = elastic(dim=dim, plane=plane)
    bc(m, *rollers(*["xmin", "ymin", "zmin"][:dim]))
    traction = [S] + [0.0] * (dim - 1)
    bc(m, {"type": "traction", "selection": "xmax", "value": traction})
    m["outputs"]["quantities"].append(
        {"name": "reaction_x", "expr": "(sigma*Normal)(1)", "selection": "xmin"})
    return m


def test_uniaxial_plane_stress():
    m = error_quantity(uniaxial(2, "stress"), f"[{S}/{E}*X(1), -{NU}*{S}/{E}*X(2)]")
    m["outputs"]["quantities"].append({"name": "vm", "expr": "von_mises", "selection": "all"})
    r = solve(m)
    assert rel_error(r) < 1e-10
    assert abs(r["quantities"]["reaction_x"] + S * 1.0) < 1e-8      # height 1
    assert abs(r["quantities"]["vm"] / 2.0 - S) < 1e-8             # area 2
    assert abs(r["max_von_mises"] - S) < 1e-8


def test_uniaxial_plane_strain():
    a = S * (1 - NU ** 2) / E
    b = -S * NU * (1 + NU) / E
    m = error_quantity(uniaxial(2, "strain"), f"[{a}*X(1), {b}*X(2)]")
    # sigma_zz = nu*S, so von Mises = S*sqrt(1 - nu + nu^2)
    m["outputs"]["quantities"].append({"name": "vm", "expr": "von_mises", "selection": "all"})
    r = solve(m)
    assert rel_error(r) < 1e-10
    assert abs(r["quantities"]["vm"] / 2.0 - S * math.sqrt(1 - NU + NU ** 2)) < 1e-8


def test_uniaxial_3d():
    m = error_quantity(uniaxial(3, None),
                       f"[{S}/{E}*X(1), -{NU}*{S}/{E}*X(2), -{NU}*{S}/{E}*X(3)]")
    m["meshes"][0]["divisions"] = [2, 2, 2]
    r = solve(m)
    assert rel_error(r) < 1e-10
    assert abs(r["max_von_mises"] - S) < 1e-8


def test_uniaxial_quadrilaterals_p1():
    m = error_quantity(uniaxial(2, "stress"), f"[{S}/{E}*X(1), -{NU}*{S}/{E}*X(2)]")
    m["meshes"][0]["element"] = "quad"
    m["discretization"]["degree"] = 1
    assert rel_error(solve(m)) < 1e-10


# ------------------------------------------------------------------ other conditions

def test_hydrostatic_pressure():
    """Pressure p on two faces, rollers on the two others: sigma = -p I."""
    p = 30.0
    m = elastic(plane="stress")
    bc(m, *rollers("xmin", "ymin"),
       {"type": "pressure", "selection": "xmax", "value": p},
       {"type": "pressure", "selection": "ymax", "value": p})
    c = -p * (1 - NU) / E
    r = solve(error_quantity(m, f"[{c}*X(1), {c}*X(2)]"))
    assert rel_error(r) < 1e-10


def test_manufactured_solution_with_body_force():
    """u = (x^2, x y), plane strain: f = -div sigma = (-(3 lambda + 5 mu), 0)."""
    m = elastic(plane="strain", n=4)
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    exact = ["sqr(X(1))", "X(1)*X(2)"]
    m["physics"]["sources"] = [{"selection": "all", "value": [-(3 * lam + 5 * mu), 0.0]}]
    bc(m, {"type": "displacement", "selection": "boundary", "value": exact})
    r = solve(error_quantity(m, "[sqr(X(1)), X(1)*X(2)]"))
    assert rel_error(r) < 1e-10


def test_gravity_as_expression_with_density():
    """Body force given as a GWFL vector expression using a material property.
    Column of height H fixed at the bottom, rollers on the sides (1D state):
    u_y = rho g / (lambda + 2 mu) * (y^2/2 - H y)   (plane strain)."""
    H, rho, g = 1.0, 8e-3, 9.81
    m = elastic(size=[0.2, H], n=4, plane="strain")
    m["materials"][0]["properties"]["rho"] = rho
    m["physics"]["sources"] = [{"selection": "all", "value": f"[0, -{g}*rho]"}]
    bc(m, {"type": "displacement", "selection": "ymin", "value": [0.0, 0.0]},
       *rollers("xmin", "xmax"))
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    c = rho * g / (lam + 2 * mu)
    m["outputs"]["quantities"].append(
        {"name": "reaction_y", "expr": "(sigma*Normal)(2)", "selection": "ymin"})
    r = solve(error_quantity(m, f"[0, {c}*(sqr(X(2))/2 - {H}*X(2))]"))
    assert rel_error(r) < 1e-10
    # The base carries the weight: reaction = rho g * width * H
    assert abs(r["quantities"]["reaction_y"] / (rho * g * 0.2 * H) - 1) < 1e-8


def test_fixed_and_roller_sharing_a_corner():
    """Clamped left edge, roller on the bottom edge (they share a node),
    tension on the right: the system must be solvable and in equilibrium."""
    m = elastic(plane="stress", n=8)
    bc(m, {"type": "fixed", "selection": "xmin"}, *rollers("ymin"),
       {"type": "traction", "selection": "xmax", "value": [S, 0.0]})
    m["outputs"]["quantities"].append(
        {"name": "reaction_x", "expr": "(sigma*Normal)(1)", "selection": "xmin"})
    r = solve(m)
    # reaction computed from the stress near the singular clamped corners: approximate
    assert abs(r["quantities"]["reaction_x"] / (-S) - 1) < 5e-2, r["quantities"]


def test_singular_systems():
    """A roller on the left edge only leaves the vertical translation free;
    validation cannot see it. With a horizontal load the system is singular
    but compatible: solved, with a warning. With a vertical load component it
    has no solution: the run must fail instead of returning u = 0."""
    for load, solvable in (([S, 0.0], True), ([S, 1.0], False)):
        m = elastic(plane="stress")
        bc(m, *rollers("xmin"), {"type": "traction", "selection": "xmax", "value": load})
        assert M.validate(m) == []
        with tempfile.TemporaryDirectory() as tmp:
            res = runner.run(m, tmp)
        if solvable:
            assert res.ok and "not unique" in res.results["warnings"][0]
            assert abs(res.results["max_von_mises"] - S) < 1e-8
        else:
            assert not res.ok and "linear solve failed" in res.error, res.error


def test_stress_concentration_around_a_hole():
    """Quarter of a plate with a small hole under remote tension (Kirsch):
    the stress concentration factor is close to 3."""
    a = 0.1
    m = elastic(plane="stress")
    m["geometries"][0] = {
        "name": "geometry",
        "source": {"type": "csg", "dim": 2, "shape": {"type": "difference", "of": [
            {"type": "rectangle", "min": [0.0, 0.0], "max": [1.0, 1.0]},
            {"type": "ball", "center": [0.0, 0.0], "radius": a}]}},
        "selections": {
            "sym_x": {"kind": "boundary", "criteria": {"type": "plane", "point": [0, 0], "normal": [1, 0]}},
            "sym_y": {"kind": "boundary", "criteria": {"type": "plane", "point": [0, 0], "normal": [0, 1]}},
            "right": {"kind": "boundary", "criteria": {"type": "plane", "point": [1, 0], "normal": [1, 0]}},
        }}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "getfem", "h": 0.05, "order": 2}
    bc(m, *rollers("sym_x", "sym_y"), {"type": "traction", "selection": "right", "value": [S, 0.0]})
    kt = solve(m)["max_von_mises"] / S
    assert 2.8 < kt < 3.4, kt


# ------------------------------------------------------------------ validation

def test_elasticity_validation_messages():
    def errs(mutate, dim=2):
        m = elastic(dim=dim)
        bc(m, {"type": "fixed", "selection": "xmin"})
        mutate(m)
        return M.validate(m)

    assert M.validate(bc(elastic(), {"type": "fixed", "selection": "xmin"})) == []
    assert any("missing Poisson's ratio" in e for e in errs(
        lambda m: m["materials"][0]["properties"].pop("nu")))
    assert any("2 components expected" in e for e in errs(lambda m: bc(
        m, {"type": "traction", "selection": "xmax", "value": [1, 2, 3]})))
    assert any("only meaningful in 2D" in e for e in errs(
        lambda m: m["physics"].update(plane="stress"), dim=3))
    assert any("strain, stress expected" in e for e in errs(
        lambda m: m["physics"].update(plane="axisymmetric")))
    assert any("expected, got 'temperature'" in e for e in errs(lambda m: bc(
        m, {"type": "temperature", "selection": "xmax", "value": 1})))
    assert any("unknown field" in e for e in errs(lambda m: bc(
        m, {"type": "roller", "selection": "xmax", "value": 1})))
    assert any("reserved name" in e for e in errs(
        lambda m: m["materials"][0]["properties"].update(sigma=1.0)))
    assert any("rigid body motions" in e for e in M.validate(
        bc(elastic(), {"type": "traction", "selection": "xmax", "value": [1, 0]})))


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok       {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAILED   {name}: {exc}")
    sys.exit(1 if failed else 0)
