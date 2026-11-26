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

"""The PDE module: coefficient and general forms, scalar and vector
unknowns, its conditions, SUPG, the orders in time (by the schemes of the
global solver); manufactured solutions; its options in the form (the
properties follow), the names of its coefficients (K_u: those of its
unknown), the equation shown (all the terms of its form).

    python -m pytest tests/test_pde.py      (or: python tests/test_pde.py)
"""

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, equations, model as M, runner  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXACT = "sin(pi*x)*sin(pi*y)"


def _getfem():
    try:
        import getfem  # noqa: F401
        return True
    except ImportError:
        return False


def pde(n=16, degree=2):
    """The unit square, a PDE module on u (its defaults: -div(grad u) = f)."""
    m = M.default_model()
    m["meshes"][0]["divisions"] = [n, n]
    md = m["models"][0]
    md.update(name="pde", vtu="pde.vtu")
    md["variables"] = [M.new_variable("u", "mesh", M.SCALAR, degree)]
    md["modules"] = [M.new_module("pde", 2)]
    return m, md, md["modules"][0]


def option(m, key, value, k=0):
    f = next(x for x in E.fields(m, ("models", 0, "modules", k)) if x.key[-1] == key)
    E.apply(m, f, value)


def run(m):
    assert M.validate(m) == [], M.validate(m)
    with tempfile.TemporaryDirectory() as tmp:
        r = runner.run(m, tmp)
        assert r.ok, f"{r.error}\n{r.stderr[-3000:]}"
        return r.results["models"]["pde"]


def test_options_and_properties():
    """The properties follow the options: those of the form, of the order
    in time; a vector unknown: its variable, matrices; d at order 2 a
    damping (0); the options shown."""
    m, md, mod = pde()
    assert list(mod["properties"]) == ["K", "b", "j", "v", "r"] and M.validate(m) == []
    assert mod["properties"]["v"][0]["value"] == [0.0, 0.0]
    shown = [f.key[-1] for f in E.fields(m, ("models", 0, "modules", 0)) if f.op == "pde_option"]
    assert shown == ["form", "unknown", "time_order", "diffusion", "stabilization"]
    option(m, "time_order", "1")
    assert mod["properties"]["d"] == [{"selection": "all", "value": 1.0}]
    option(m, "time_order", "2")
    assert mod["properties"]["d"][0]["value"] == 0.0 and mod["properties"]["m"][0]["value"] == 1.0
    option(m, "time_order", "0")
    assert "d" not in mod["properties"] and "m" not in mod["properties"]
    mod["properties"]["K"][0]["value"] = 2.0
    option(m, "diffusion", "anisotropic")
    assert mod["properties"]["K"][0]["value"] == 2.0 and M.property_kind(mod, "K") == M.MATRIX
    mod["properties"]["K"][0]["value"] = "[2, 0; 0, 2]"
    option(m, "diffusion", "isotropic")
    assert mod["properties"]["K"][0]["value"] == 2.0
    option(m, "unknown", "vector")                       # its variable, its coefficients, its values
    assert md["variables"][0]["components"] == "dim" and "b" not in mod["properties"]
    assert M.property_kind(mod, "r") == M.MATRIX and M.property_kind(mod, "j") == M.MATRIX
    shown = [f.key[-1] for f in E.fields(m, ("models", 0, "modules", 0)) if f.op == "pde_option"]
    assert "stabilization" not in shown and M.validate(m) == []
    option(m, "form", "general")
    assert list(mod["properties"]) == ["F"] and mod["properties"]["F"][0]["value"] == "Grad_u"
    assert M.validate(m) == []
    option(m, "unknown", "scalar")
    assert md["variables"][0].get("components", 1) == 1 and M.validate(m) == []
    md["variables"][0]["components"] = 3                   # (a vector: of the dimension)
    mod["unknown"] = "vector"
    assert any("number of components" in e for e in M.validate(m))


def test_names_and_editing():
    """The coefficients are named after the unknown: no clash with a
    parameter r; renaming the unknown renames them in the expressions; the
    equation shows all the terms of the form (not whether it is solved in
    time: the solvers), each field the name of its coefficient (help); the
    GWFL terms of the script (the nonzero ones)."""
    m, md, mod = pde()
    m["parameters"] = [{"name": "r", "value": 2.0}, {"name": "K", "value": 3.0}]
    mod["properties"]["r"] = [{"selection": "all", "value": "r"}]           # (the parameter r)
    md["outputs"] = [{"name": "q", "expr": "r_u*u + K_u", "selection": "all"}]
    assert M.validate(m) == []
    eq = equations.equation(m, md, mod)
    assert "−div(K ∇u + b u − j) + v·∇u + r u = f" in eq and "∂u/∂t" not in eq and "olve" not in eq
    tip = next(x for x in E.fields(m, ("models", 0, "modules", 0)) if x.label == "K (diffusion)").help
    assert tip == "diffusion; K_u in the expressions"
    f = next(x for x in E.fields(m, ("models", 0, "variables", 0)) if x.key[-1] == "name")
    E.apply(m, f, "w")
    assert mod["variable"] == "w" and md["outputs"][0]["expr"] == "r_w*w + K_w" and M.validate(m) == []
    terms = next(x for x in E.fields(m, ("models", 0, "modules", 0)) if x.kind == "code").help
    assert "(K_w*Grad_w).Grad_Test_w" in terms and "r_w*w*Test_w" in terms and "v_w" not in terms
    mod["properties"]["v"][0]["value"] = [1, 0]
    option(m, "stabilization", "supg")
    terms = next(x for x in E.fields(m, ("models", 0, "modules", 0)) if x.kind == "code").help
    assert "tau_w := " in terms and "SUPG" in equations.equation(m, md, mod)
    # a second PDE in the model: its own coefficients
    path = E.do_action(m, ("models", 0, "modules"), "add_module:pde")
    second = E.get_at(m, path)
    assert second["variable"] == "u" and [v["name"] for v in md["variables"]] == ["w", "u"]
    second["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    assert M.validate(m) == [] and M.model_properties(md)[:2] == ["K_w", "b_w"] and "K_u" in M.model_properties(md)


def test_several_pdes():
    """Several PDEs in one model: a new one on a new unknown (the menu still
    offers it); a PDE per unknown; their own coefficients, macros, fields;
    a coupled system (-Lap u + u - w = f1, -Lap w + 2 w - u = f2,
    manufactured; the coupling by a source depending on w, a term in u)
    solved together, and in time with an order per unknown."""
    m, md, mod = pde(12)
    assert ("add_module:pde", "Add pde (coefficient or general form)") in E.actions(m, ("models", 0, "modules"))
    path = E.do_action(m, ("models", 0, "modules"), "add_module:pde")
    second = E.get_at(m, path)
    assert second["variable"] == "u2" and [v["name"] for v in md["variables"]] == ["u", "u2"]
    assert [n.label for n in E.find_node(E.tree(m), ("models", 0, "modules")).children] == ["PDE on u", "PDE on u2"]
    second["variable"] = "u"
    assert any("'u' is already the unknown of the PDE module 1" in e for e in M.validate(m))
    f = next(x for x in E.fields(m, ("models", 0, "variables", 1)) if x.key[-1] == "name")
    E.apply(m, f, "w")                                    # (the second unknown renamed: its PDE follows)
    second["variable"] = "w"
    exact_u, exact_w = EXACT, f"sin(2*pi*x)*sin(pi*y)"
    mod["properties"]["r"] = [{"selection": "all", "value": 1.0}]
    second["properties"]["r"] = [{"selection": "all", "value": 2.0}]
    mod["sources"] = [{"selection": "all", "value": f"(2*sqr(pi) + 1)*{exact_u} - {exact_w} + w"}]
    second["sources"] = [{"selection": "all", "value": f"(5*sqr(pi) + 2)*{exact_w} - {exact_u}"}]
    md["terms"] = [{"name": "coupling", "expr": "-u*Test_w", "selection": "all"}]   # (or by a term)
    for mm in (mod, second):
        mm["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    md["outputs"] = [{"name": "err_u", "expr": f"sqr(u - {exact_u})", "selection": "all"},
                     {"name": "err_w", "expr": f"sqr(w - {exact_w})", "selection": "all"}]
    assert M.validate(m) == [], M.validate(m)
    text = runner.export_script(m)
    for line in ("md_pde.add_macro('F_u', 'K_u*Grad_u')", "md_pde.add_macro('F_w', 'K_w*Grad_w')",
                 "'flux_u'", "'flux_w'"):
        assert line in text, line
    if not _getfem():
        return
    r = run(m)
    assert r["quantities"]["err_u"] < 1e-6 and r["quantities"]["err_w"] < 1e-5, r["quantities"]
    assert r["max_u"] > 0.9 and r["max_w"] > 0.9
    # in time: u of order 1 (a diffusion from a unit source), w of order 0 (-Lap w + 2 w = u, the
    # coupling term: at each time step, from u)
    option(m, "time_order", "1", 0)
    mod["sources"] = [{"selection": "all", "value": 1.0}]
    second["sources"] = [{"selection": "all", "value": 0.0}]
    md["outputs"] = [{"name": "mean_u", "expr": "u", "selection": "all"},
                     {"name": "mean_w", "expr": "w", "selection": "all"}]
    m["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"],
                     "time": {"start": 0, "end": 0.2, "step": 0.02}}]
    assert list(M.with_defaults(m)["solvers"][0]["schemes"]) == ["pde/u"]
    s = run(m)["series"]
    assert s["mean_u"][0] == 0 and 0 < s["mean_u"][-1] and s["mean_w"][-1] > s["mean_w"][1] > 0


def test_coefficient_form():
    """Manufactured solutions: -div(c grad u) + a u = f (P2, its order);
    anisotropic c; al, ga, be (1D-like); flux and Robin conditions."""
    if not _getfem():
        return
    errs = []
    for n in (8, 16):
        m, md, mod = pde(n)
        mod["properties"]["r"] = [{"selection": "all", "value": 3.0}]
        mod["sources"] = [{"selection": "all", "value": f"(2*sqr(pi) + 3)*{EXACT}"}]
        mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
        md["outputs"] = [{"name": "err", "expr": f"sqr(u - {EXACT})", "selection": "all"}]
        errs.append(np.sqrt(run(m)["quantities"]["err"]))
    assert errs[1] < 2e-4 and 6 < errs[0] / errs[1] < 10, errs                 # (order 3 in L2)
    m, md, mod = pde(12)                                                        # anisotropic
    option(m, "diffusion", "anisotropic")
    mod["properties"]["K"] = [{"selection": "all", "value": "[2, 0.5; 0.5, 1]"}]
    mod["sources"] = [{"selection": "all", "value": f"3*sqr(pi)*{EXACT} - sqr(pi)*cos(pi*x)*cos(pi*y)"}]
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    md["outputs"] = [{"name": "err", "expr": f"sqr(u - {EXACT})", "selection": "all"}]
    assert run(m)["quantities"]["err"] < 1e-7
    # u = x^2: -div(grad u + al u - ga) + be.grad u = f with al = (1, 0), ga = (2, 0), be = (0, 1)
    m, md, mod = pde(6)
    for p, v in (("b", [1, 0]), ("j", [2, 0]), ("v", [0, 1])):
        mod["properties"][p] = [{"selection": "all", "value": v}]
    mod["sources"] = [{"selection": "all", "value": "-2 - 2*x"}]
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": "sqr(x)"}]
    md["outputs"] = [{"name": "err", "expr": "sqr(u - sqr(x))", "selection": "all"}]
    assert run(m)["quantities"]["err"] < 1e-20
    # -u'' = 0, Robin on xmin (F.n = -u' = g - q u: u(0) = 1/2 with q = 2, g = 0), flux 1 on xmax
    m, md, mod = pde(4)
    mod["conditions"] = [{"type": "robin", "selection": "xmin", "q": 2.0, "value": 0.0},
                         {"type": "flux", "selection": "xmax", "value": 1.0}]
    md["outputs"] = [{"name": "err", "expr": "sqr(u - x - 0.5)", "selection": "all"}]
    r = run(m)
    assert r["quantities"]["err"] < 1e-20 and abs(r["max_u"] - 1.5) < 1e-10


def test_vector_and_general_form():
    """A vector unknown coupled by a matrix a; the general form, nonlinear
    (Newton)."""
    if not _getfem():
        return
    m, md, mod = pde(12)
    option(m, "unknown", "vector")
    mod["properties"]["r"] = [{"selection": "all", "value": "[2, 1; 1, 2]"}]
    mod["sources"] = [{"selection": "all", "value": [f"(2*sqr(pi) + 2)*{EXACT}", EXACT]}]
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": [0, 0]}]
    md["outputs"] = [{"name": "err", "expr": f"sqr(u(1) - {EXACT}) + sqr(u(2))", "selection": "all"}]
    assert run(m)["quantities"]["err"] < 1e-7
    m, md, mod = pde(8)
    option(m, "form", "general")
    mod["properties"]["F"] = [{"selection": "all", "value": "(1 + sqr(u))*Grad_u"}]
    mod["sources"] = [{"selection": "all", "value": "-2*x"}]
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": "x"}]
    md["outputs"] = [{"name": "err", "expr": "sqr(u - x)", "selection": "all"}]
    r = run(m)
    assert r["quantities"]["err"] < 1e-20 and r["iterations"] > 1


def test_supg():
    """A dominant convection, -k u'' + u' = 0 (u(0) = 0, u(1) = 1): the
    Galerkin solution oscillates, SUPG's does not; with a moderate one and
    P2, SUPG (consistent) is as close to the exact solution as Galerkin."""
    if not _getfem():
        return

    def layer(k, stab, degree):
        m, md, mod = pde(20, degree)
        mod["properties"]["K"] = [{"selection": "all", "value": k}]
        mod["properties"]["v"] = [{"selection": "all", "value": [1, 0]}]
        option(m, "stabilization", stab)
        mod["conditions"] = [{"type": "imposed_value", "selection": "xmin", "value": 0},
                             {"type": "imposed_value", "selection": "xmax", "value": 1}]
        exact = f"(exp((x - 1)/{k}) - exp(-1/{k}))/(1 - exp(-1/{k}))"
        md["outputs"] = [{"name": "err", "expr": f"sqr(u - {exact})", "selection": "all"}]
        return run(m)
    galerkin, supg = layer(0.005, "none", 1), layer(0.005, "supg", 1)
    assert galerkin["min_u"] < -0.3 and supg["min_u"] > -1e-6 and supg["max_u"] < 1 + 1e-6
    err = [layer(0.1, stab, 2)["quantities"]["err"] for stab in ("none", "supg")]   # (tau: h over the degree)
    assert err[1] < 3 * err[0] < 1e-7, err


def test_in_time():
    """Order 1: the same as the heat conduction; order 2: the wave
    equation (a standing wave, Newmark)."""
    if not _getfem():
        return
    m, md, mod = pde(8)
    option(m, "time_order", "1")
    mod["properties"]["d"] = [{"selection": "all", "value": 2.0}]
    mod["sources"] = [{"selection": "all", "value": 1.0}]
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    md["outputs"] = [{"name": "mean", "expr": "u", "selection": "all"}]
    m["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"],
                     "time": {"start": 0, "end": 0.05, "step": 0.01}}]
    assert "d ∂u/∂t − div(K ∇u + b u − j) + v·∇u + r u = f" in equations.equation(m, md, mod)
    s1 = run(m)["series"]["mean"]
    h = copy.deepcopy(m)
    hm = h["models"][0]
    hm["variables"] = [M.new_variable("T", "mesh", M.SCALAR, 2)]
    hm["modules"] = [M.new_module("heat_conduction", 2)]
    hm["modules"][0]["properties"]["rho"] = [{"selection": "all", "value": 2.0}]
    hm["modules"][0]["sources"] = [{"selection": "all", "value": 1.0}]
    hm["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": 0}]
    hm["outputs"] = [{"name": "mean", "expr": "T", "selection": "all"}]
    h["solvers"][0]["schemes"] = {}
    s2 = run(h)["series"]["mean"]
    assert np.allclose(s1, s2, rtol=1e-12) and s1[-1] > 0
    m, md, mod = pde(12)
    option(m, "time_order", "2")
    md["variables"][0]["initial"] = EXACT
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": 0}]
    md["outputs"] = [{"name": "err", "expr": f"sqr(u - {EXACT}*cos(sqrt(2)*pi*t))", "selection": "all"}]
    m["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"],
                     "time": {"start": 0, "end": 0.3, "step": 0.01}}]
    assert M.with_defaults(m)["solvers"][0]["schemes"]["pde/u"]["type"] == "newmark"
    assert "m ∂²u/∂t² + d ∂u/∂t − div(K ∇u" in equations.equation(m, md, mod)
    assert max(run(m)["series"]["err"]) < 1e-6
    m["solvers"] = []                                     # stationary: no time derivative
    assert run(m)["max_u"] < 1e-12


def test_examples():
    """The convection-diffusion (SUPG: no overshoot, Galerkin: one), the
    membrane (Newmark: its energy, pi/2, conserved)."""
    if not _getfem():
        return
    m = M.load(EXAMPLES / "convection_diffusion.json")
    m["models"][0]["name"] = "pde"
    m["models"][0]["vtu"] = "pde.vtu"
    r = run(m)
    assert r["min_u"] > -1e-8 and r["max_u"] < 1
    m["models"][0]["modules"][0]["stabilization"] = "none"
    assert run(m)["max_u"] > 2
    with tempfile.TemporaryDirectory() as tmp:
        r = runner.run(M.load(EXAMPLES / "membrane_wave.json"), tmp)
    assert r.ok, r.error
    s = r.results["models"]["membrane"]["series"]
    e = np.array(s["energy"])
    assert np.abs(e / (np.pi / 2) - 1).max() < 1e-3 and np.abs(e / e[0] - 1).max() < 1e-6
    assert min(s["min_w"]) < -0.2                         # (reflected waves)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
