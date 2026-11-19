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

"""Elastodynamics: the linear elasticity in time, with the second order
schemes (Newmark, HHT-alpha, generalized-alpha): validation, the equality
with the standard schemes (on M and K assembled apart), the special cases,
the order 2, the energy (conserved, dissipated), the Rayleigh damping, the
initial acceleration, editing.

    python -m pytest tests/test_elastodynamics.py      (or: python tests/test_elastodynamics.py)
"""

import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, equations, model as M, runner  # noqa: E402


def beam(scheme=None, end=1.0, step=0.05, n=(20, 2), every=1, load=False):
    """A cantilever [0, 1] x [0, 0.1] (E = 1, nu = 0.3, rho = 1, plane strain),
    fixed at x = 0, released from u = (0, 0.01 x^2) at rest; or (load) from
    u = 0 at rest under the smooth load f = (0, -sin(t)^2/1000)."""
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "box", "origin": [0, 0], "size": [1, 0.1]}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "structured", "divisions": list(n)}
    md = M.new_model("beam", "mesh", "linear_elasticity", 2)
    md["variables"][0]["initial"] = ["0", "0.01*sqr(x)"]
    md["modules"][0]["conditions"] = [{"type": "fixed", "selection": "xmin"}]
    if load:
        md["variables"][0]["initial"] = ["0", "0"]
        md["modules"][0]["sources"] = [{"selection": "all", "value": ["0", "-sqr(sin(t))/1000"]}]
    md["outputs"] = [{"name": "tip", "expr": "u(2)", "selection": "xmax"},
                     {"name": "kinetic", "expr": "0.5*rho*Dot_u.Dot_u", "selection": "all"},
                     {"name": "potential", "expr": "0.5*sigma:eps", "selection": "all"}]
    m["models"] = [md]
    m["solvers"] = [{"name": "main", "type": "transient", "steps": ["beam"],
                     "time": {"end": end, "step": step}, "save": {"every": every},
                     "schemes": {"beam/u": scheme or {"type": "newmark"}}}]
    return m


def run(m):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, res.error
    return res.results["models"]["beam"]["series"]


def _getfem():
    try:
        import getfem  # noqa: F401
        return True
    except ImportError:
        return False


def test_validation():
    m = beam()
    assert M.validate(m) == []
    md = m["models"][0]
    assert M.variable_order(md, "u") == 2 and M.time_variables(m, md) == ["u"]
    assert M.schemes_for(2) == ["stationary", "newmark", "hht", "generalized_alpha",
                                          "explicit_newmark", "explicit_generalized_alpha"]
    for sc, msg in [({"type": "theta"}, "stationary, newmark, hht, generalized_alpha, explicit_newmark, "
                                       "explicit_generalized_alpha expected"),
                    ({"type": "newmark", "beta": 0}, "beta: number in ]0, 0.5] expected"),
                    ({"type": "newmark", "gamma": 0.4}, "gamma: number in [0.5, 1] expected"),
                    ({"type": "hht", "alpha": 0.1}, "alpha: number in [-0.3333, 0] expected"),
                    ({"type": "generalized_alpha", "rho_inf": 2}, "rho_inf: number in [0, 1] expected"),
                    ({"type": "newmark", "initial_acceleration": "rest"}, "equation, zero expected")]:
        m2 = beam(sc)
        assert any(msg in e for e in M.validate(m2)), (sc, M.validate(m2))
    m2 = beam()                                      # no scheme given: Newmark, average acceleration
    m2["solvers"][0]["schemes"] = {}
    assert M.scheme_of(m2, m2["models"][0], "u") == {"type": "newmark", "beta": 0.25, "gamma": 0.5,
                                                      "initial_acceleration": "equation"}
    m2 = beam({"type": "stationary"})               # quasi-static: no density needed
    del m2["models"][0]["modules"][0]["properties"]["rho"]
    assert M.validate(m2) == [] and M.time_variables(m2, m2["models"][0]) == []
    m2 = beam()                                      # in time, no constraint: allowed (the inertia)
    m2["models"][0]["modules"][0]["conditions"] = []
    assert M.validate(m2) == []


def test_parameters():
    """beta, gamma, alpha_m, alpha_f of the schemes; the coefficients."""
    assert M.second_order_parameters({"type": "newmark"}) == (0.25, 0.5, 0.0, 0.0)
    b, g, am, af = M.second_order_parameters({"type": "hht", "alpha": -0.1})
    assert np.allclose([b, g, am, af], [1.1 ** 2 / 4, 0.6, 0, 0.1])
    b, g, am, af = M.second_order_parameters({"type": "generalized_alpha", "rho_inf": 0.5})
    assert np.allclose([am, af], [0, 1 / 3]) and np.isclose(g, 0.5 - am + af) and np.isclose(b, (1 - am + af) ** 2 / 4)
    k = M.second_order_coefficients({"type": "newmark"})
    assert k == {"acc": [4.0, -4.0, -1.0], "vel": [2.0, -1.0, 0.0], "inertia": [4.0, -4.0, -1.0, -0.0]}
    k = M.second_order_coefficients({"type": "generalized_alpha", "rho_inf": 1.0})     # = Newmark
    assert np.allclose(k["acc"], [4, -4, -1]) and np.allclose(k["inertia"], [4, -4, 0, -1])


def _reference(sc, end, step):
    """The standard generalized-alpha scheme (Chung and Hulbert: (1 - am) M
    a1 + am M a0 + (1 - af) K u1 + af K u0 = 0, Newmark's relations) on M
    and K assembled apart on the mesh of the script, a0 = -M^-1 K u0: the
    tip displacement at each step."""
    import getfem as gf
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    b, g, am, af = M.second_order_parameters(sc)
    ns = {"__name__": "reference"}
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            exec(runner.export_script(beam(sc, end=step, step=step)), ns)
        finally:
            os.chdir(cwd)
    m, mf, mim = ns["m_mesh"], ns["mf_mesh_P2_q2"], ns["mim_mesh"]
    lam, mu = 0.3 / (1.3 * 0.4), 1 / 2.6
    md = gf.Model("real")
    md.add_fem_variable("u", mf)
    K = gf.asm_generic(mim, 2, f"({lam}*Trace(Sym(Grad_u))*Id(2) + 2*{mu}*Sym(Grad_u)):Grad_Test_u", -1, md)
    Mm = gf.asm_generic(mim, 2, "u.Test_u", -1, md)
    K, Mm = sp.csr_matrix(K.full()), sp.csr_matrix(Mm.full())
    P = mf.basic_dof_nodes()
    free = np.where(P[0] > 1e-9)[0]
    u0 = np.zeros(mf.nbdof())
    u0[1::2] = 0.01 * P[0, 1::2] ** 2
    m.set_region(9, m.outer_faces_with_direction([1, 0], 0.01))
    tip = gf.asm_generic(mim, 1, "Test_u(2)", 9, md)[free]
    Kf, Mf = K[free][:, free], Mm[free][:, free]
    u, v = u0[free], np.zeros(len(free))
    a = sla.spsolve(Mf.tocsc(), -(Kf @ u))
    n = int(round(end / step))
    lu = sla.splu(((1 - am) / (b * step ** 2) * Mf + (1 - af) * Kf).tocsc())
    out = [tip @ u]
    for _ in range(n):
        u1 = lu.solve(-af * (Kf @ u) - am * (Mf @ a)
                      + (1 - am) * (Mf @ ((u + step * v) / (b * step ** 2) + (0.5 - b) / b * a)))
        a1 = (u1 - u - step * v) / (b * step ** 2) - (0.5 - b) / b * a
        v, u, a = v + step * ((1 - g) * a + g * a1), u1, a1
        out.append(tip @ u)
    return np.array(out)


def test_standard_schemes():
    """The schemes written at t_{n+1} (the modified acceleration, its memory)
    are the standard ones: the same displacements, to rounding errors."""
    if not _getfem():
        return
    for sc in ({"type": "newmark", "beta": 0.3, "gamma": 0.6}, {"type": "hht", "alpha": -0.3},
               {"type": "generalized_alpha", "rho_inf": 0.2}):
        ours = np.array(run(beam(sc))["tip"])
        ref = _reference(sc, 1.0, 0.05)
        assert np.abs(ours - ref).max() < 1e-9 * np.abs(ref).max(), sc


def test_special_cases():
    """HHT with alpha = 0 and generalized-alpha with rho_inf = 1 are
    Newmark's average acceleration."""
    if not _getfem():
        return
    newmark = np.array(run(beam())["tip"])
    for sc in ({"type": "hht", "alpha": 0.0}, {"type": "generalized_alpha", "rho_inf": 1.0}):
        assert np.allclose(run(beam(sc))["tip"], newmark, rtol=0, atol=1e-12), sc


def test_order_two():
    """Second order in time: under a smooth load from rest, the error at t =
    2 (against a fine time step) divided by about 4 when the time step is
    halved."""
    if not _getfem():
        return
    for sc in ({"type": "newmark"}, {"type": "hht", "alpha": -0.1}, {"type": "generalized_alpha", "rho_inf": 0.8}):
        fine = run(beam(sc, end=2.0, step=2.0 / 256, n=(10, 1), load=True))["tip"][-1]
        e = [abs(run(beam(sc, end=2.0, step=2.0 / k, n=(10, 1), load=True))["tip"][-1] - fine) for k in (32, 64)]
        assert 3.6 < e[0] / e[1] < 4.6, (sc, e)


def test_energy():
    """Newmark's average acceleration conserves the energy (no load); HHT and
    the generalized-alpha scheme dissipate (the high frequencies), as the
    Rayleigh damping; the derivatives in the outputs are those at the saved
    times."""
    if not _getfem():
        return
    def energy(m):
        S = run(m)
        return np.array(S["kinetic"]) + np.array(S["potential"])
    e = energy(beam())
    assert e[0] > 0 and np.abs(e - e[0]).max() < 1e-9 * e[0]
    d = energy(beam({"type": "hht", "alpha": -0.3}))
    assert np.all(np.diff(d) <= 1e-12 * d[0]) and d[-1] < 0.9 * d[0]
    d = energy(beam({"type": "generalized_alpha", "rho_inf": 0.0}))   # (a modified energy decreases)
    assert d[-1] < 0.1 * d[0]
    m = beam()
    m["models"][0]["modules"][0]["properties"]["alpha_M"] = [{"selection": "all", "value": 1.0}]
    d = energy(m)
    assert np.all(np.diff(d) <= 1e-12 * d[0]) and d[-1] < 0.6 * d[0]
    m = beam()
    m["models"][0]["modules"][0]["properties"]["beta_K"] = [{"selection": "all", "value": 0.01}]
    d = energy(m)
    assert np.all(np.diff(d) <= 1e-12 * d[0]) and d[-1] < 0.9 * d[0]


def test_initial_acceleration():
    """From the equation (a0 = -M^-1 K u0: the reference), or zero; the
    initial velocity."""
    if not _getfem():
        return
    sc = {"type": "newmark", "initial_acceleration": "zero"}
    zero = np.array(run(beam(sc))["tip"])
    ref = _reference({"type": "newmark"}, 1.0, 0.05)
    assert np.abs(zero - ref).max() > 1e-4 * np.abs(ref).max()
    m = beam()                                       # from the rest shape, an initial velocity
    md = m["models"][0]
    md["variables"][0]["initial"] = ["0", "0"]
    md["variables"][0]["initial_velocity"] = ["0", "0.01*sqr(x)"]
    assert M.validate(m) == []
    S = run(m)
    assert S["tip"][0] == 0 and S["tip"][1] > 0 and np.isclose(S["kinetic"][0], 0.5 * 1e-4 / 5 * 0.1, rtol=1e-6)


def test_script():
    text = runner.export_script(beam({"type": "hht", "alpha": -0.1}))
    for line in ("md_beam.add_macro('Dot2_u', 'u__acc(1)*(u - u__prev)/sqr(timestep) + u__acc(2)*u__rate/timestep "
                 "+ u__acc(3)*u__rate2')",
                 "md_beam.add_macro('u__inertia', 'u__inertia_c(1)*(u - u__prev)/sqr(timestep) "
                 "+ u__inertia_c(2)*u__rate/timestep + u__inertia_c(3)*u__rate2 + u__inertia_c(4)*u__mem')",
                 "add_term(md_beam, mim_mesh, 'rho*u__inertia.Test_u')",
                 "md_beam.set_variable('u__mem', md_beam.variable('u__rate2'))  # the memory: G_0 = -M a0"):
        assert line in text, line
    assert "alpha_M" not in text.split("# Time derivative")[1].split("\n")[1]          # (damping zero: no term)
    assert "'u__mem'" not in runner.export_script(beam())                             # (Newmark: no memory)


def test_editing():
    """The solver offers the second order schemes to u, their parameters,
    the initial acceleration; the variable its initial velocity; the module
    its density and damping, the equation in time."""
    m = beam()
    path = ("solvers", 0)
    fields = {f.label: f for f in E.fields(m, path)}
    f = fields["Scheme of u (beam)"]
    assert [k for k, _ in f.choices] == ["stationary", "newmark", "hht", "generalized_alpha",
                                          "explicit_newmark", "explicit_generalized_alpha"]
    assert {"Beta of u", "Gamma of u", "Initial acceleration of u"} <= set(fields)
    assert E.field_value(m, fields["Beta of u"]) == 0.25
    E.apply(m, f, "hht")
    s = m["solvers"][0]
    assert s["schemes"]["beam/u"] == {"type": "hht", "alpha": -0.05, "initial_acceleration": "equation"}
    fields = {f.label: f for f in E.fields(m, path)}
    assert "Alpha of u" in fields and "Beta of u" not in fields
    E.apply(m, fields["Alpha of u"], -0.2)
    E.apply(m, fields["Initial acceleration of u"], "zero")
    assert s["schemes"]["beam/u"] == {"type": "hht", "alpha": -0.2, "initial_acceleration": "zero"}
    E.apply(m, fields["Scheme of u (beam)"], "generalized_alpha")
    labels = [f.label for f in E.fields(m, path)]
    assert "Spectral radius at infinity (rho_inf) of u" in labels and M.validate(m) == []
    assert "Initial velocity" in [f.label for f in E.fields(m, ("models", 0, "variables", 0))]
    mpath = ("models", 0, "modules", 0)
    labels = {f.label: f for f in E.fields(m, mpath)}
    assert {"rho (density)", "alpha_M (mass damping (Rayleigh, 1/s))"} <= set(labels)
    md, mod = m["models"][0], m["models"][0]["modules"][0]
    eq = equations.equation(m, md, mod)
    assert "ρ ∂²u/∂t² − div σ(u) = f" in eq and "generalized-α ρ∞ = 0.8" in eq and "initial value and velocity" in eq
    mod["properties"]["alpha_M"] = [{"selection": "all", "value": 0.5}]
    mod["properties"]["beta_K"] = [{"selection": "all", "value": 0.01}]
    eq = equations.equation(m, md, mod)
    assert "ρ ∂²u/∂t² + α<sub>M</sub> ρ ∂u/∂t − div (σ(u) + β<sub>K</sub> σ<sub>e</sub>(∂u/∂t))" in eq
    terms = equations.gwfl_terms(m, 0, 0)
    assert "rho*u__inertia.Test_u" in terms and "alpha_M*rho*Dot_u.Test_u" in terms
    E.apply(m, fields["Scheme of u (beam)"], "stationary")
    assert "rho (density)" not in {f.label for f in E.fields(m, mpath)}
    assert "<b>Stationary</b>" in equations.equation(m, md, mod)
    m2 = copy.deepcopy(m)
    m2["solvers"] = []
    assert "Initial velocity" not in [f.label for f in E.fields(m2, ("models", 0, "variables", 0))]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
