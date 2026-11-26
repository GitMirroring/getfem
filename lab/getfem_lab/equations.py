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

"""What a module solves, for the form of its node: the equation (HTML; its
boundary conditions and data are in the form and the tree) and the GWFL
terms that the generated script adds for it (read from the code generated
for the module, so that they are those of the script).
"""

from __future__ import annotations

import html
import re

from . import model as M

NABLA, DIV = "∇", "div"


def _v(value) -> str:
    """A value of the model, as text."""
    if isinstance(value, list):
        return "(" + ", ".join(_v(x) for x in value) + ")"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return html.escape(str(value))


def equation(model: dict, md: dict, mod: dict) -> str:
    """The equation solved by a module and its data, as HTML."""
    fn = {"heat_conduction": _heat, "linear_elasticity": _elasticity,
          "thermal_expansion": _expansion, "pde": _pde}.get(mod.get("type"))
    return fn(model, md, mod) if fn else ""


def _time(model: dict, md: dict, mod: dict) -> str:
    """Where the module stands in time."""
    if not M.spec_of(mod).get("inertia"):
        return ""
    s = M.main_solver(model)
    where = " (its time scheme is chosen in the solver)"
    if M.in_time(model, md, mod):
        t = s.get("time", {})
        sc = M.scheme_of(model, md, M.inertia_variable(mod))
        return (f"<p><b>In time</b>, by the solver {html.escape(s['name'])}: from t = {_v(t.get('start', 0))} "
                f"to {_v(t.get('end'))}, step {_v(t.get('step'))}, {html.escape(M.scheme_label(sc))}; "
                f"from the initial value{' and velocity' if M.SCHEME_ORDER.get(sc['type']) == 2 else ''} "
                f"of the variable{where}.</p>")
    if M.is_transient(model, md):
        return (f"<p><b>Stationary</b> at each time step of the solver {html.escape(s['name'])} "
                f"(quasi-static: no time derivative){where}.</p>")
    return "<p><b>Stationary</b> (a transient global solver solves in time).</p>"


def _heat(model: dict, md: dict, mod: dict) -> str:
    T = html.escape(mod.get("variable", "T"))
    timed = M.in_time(model, md, mod)
    lhs = f"ρ c<sub>p</sub> ∂{T}/∂t − " if timed else "−"
    k = "k" if mod.get("conductivity", "isotropic") == "isotropic" else "K"
    return _time(model, md, mod) + _block([f"{lhs}{DIV}({k} {NABLA}{T}) = f &nbsp; in Ω"])


def _elasticity(model: dict, md: dict, mod: dict) -> str:
    u = html.escape(mod.get("variable", "u"))
    expansion = any(m.get("type") == "thermal_expansion" and m.get("displacement") == mod.get("variable")
                    for m in md.get("modules", []))
    dim = M.model_dim(model, md) or 2
    plane = f" (plane {mod.get('plane', 'strain')})" if dim == 2 else ""
    if M.in_time(model, md, mod):
        from .codegen import _is_zero
        props = mod.get("properties", {})
        mass = not _is_zero(props.get("alpha_M"))
        stiff = not _is_zero(props.get("beta_K"))
        first = f"ρ ∂²{u}/∂t²" + (f" + α<sub>M</sub> ρ ∂{u}/∂t" if mass else "") + f" − {DIV} " \
            + (f"(σ({u}) + β<sub>K</sub> σ<sub>e</sub>(∂{u}/∂t))" if stiff else f"σ({u})")
        damping = mass or stiff
        lines = [first + " = f &nbsp; in Ω"]
        if damping:
            lines.append("Rayleigh damping α<sub>M</sub> M + β<sub>K</sub> K" + (" (σ<sub>e</sub>: the elastic "
                         "stress, without thermal expansion)" if stiff else ""))
        return _time(model, md, mod) + _block(lines + [
            "σ = λ tr(ε) I + 2μ ε" + (" − β α (T − T<sub>ref</sub>) I" if expansion else "")
            + f", &nbsp; ε = ({NABLA}{u} + {NABLA}{u}<sup>T</sup>)/2", f"λ, μ: from E and ν{plane}"])
    return _time(model, md, mod) + _block([f"−{DIV} σ({u}) = f &nbsp; in Ω",
                   "σ = λ tr(ε) I + 2μ ε" + (" − β α (T − T<sub>ref</sub>) I" if expansion else "")
                   + f", &nbsp; ε = ({NABLA}{u} + {NABLA}{u}<sup>T</sup>)/2",
                   f"λ, μ: from E and ν{plane}"])


def _expansion(model: dict, md: dict, mod: dict) -> str:
    T, u = html.escape(str(mod.get("temperature"))), html.escape(str(mod.get("displacement")))
    return _block([f"σ({u}) gets − β α ({T} − T<sub>ref</sub>) I, &nbsp; β = 3λ + 2μ (in 3D)"])


def _pde(model: dict, md: dict, mod: dict) -> str:
    """All the terms of its form, of its order in time (whatever the values
    of the coefficients: each one has its place), the names of the
    coefficients in the expressions."""
    spec = M.spec_of(mod)
    u = html.escape(mod.get("variable", "u"))
    vector = mod.get("unknown") == "vector"
    order = spec.get("inertia", {}).get("order", 0)
    lhs = ([f"m ∂²{u}/∂t²"] if order == 2 else []) + ([f"d ∂{u}/∂t"] if order else [])
    if mod.get("form") == "general":
        lhs.append(f"{DIV} F")
    else:
        lhs.append(f"{DIV}(K {NABLA}{u}" + ("" if vector else f" + b {u}") + " − j)")
        lhs += [f"({NABLA}{u}) v" if vector else f"v·{NABLA}{u}", f"r {u}"]
    lines = [_sum(lhs) + " = f &nbsp; in Ω"]
    if mod.get("form") == "general":
        lines.append(f"F: the flux, an expression of {u} and {NABLA}{u}")
    if mod.get("stabilization") == "supg" and "stabilization" in spec["options"]:
        lines.append(f"SUPG: + τ (v·{NABLA}w) (residual), &nbsp; τ = ((2|v|/h)² + 9 (4K/h²)²)<sup>−1/2</sup>")
    if vector:
        lines.append(f"{u}: a vector; r, d, m: matrices")
    return _block(lines)                 # (whether it is solved in time: the global solvers)


def _sum(terms) -> str:
    """Terms added, a divergence subtracted (-div ...)."""
    eq = ""
    for j, term in enumerate(terms):
        neg = term.startswith(DIV)
        eq += (("−" if neg else "") + term) if j == 0 else ((" − " if neg else " + ") + term)
    return eq or "0"


def _block(lines) -> str:
    return "<p style='margin-left: 8px'>" + "<br>".join(lines) + "</p>"


# ---------------------------------------------------------------- the GWFL terms of the script

_CALL = re.compile(r"(add_term|add_linear_term|add_source_term|add_nonlinear_term|add_macro|"
                   r"add_Dirichlet_condition_with_simplification|add_Dirichlet_condition_with_multipliers|"
                   r"add_normal_Dirichlet_condition_with_multipliers|add_initialized_fem_data)\((.*)\)")
_STR = re.compile(r"'((?:[^'\\]|\\.)*)'")
_RG = re.compile(r"RG_\w+\[('[^']*')\]")


def gwfl_terms(model: dict, md_index: int, mod_index: int) -> list[str]:
    """The GWFL terms that the script adds for the equation of a module (its
    macros, its terms, its sources; not its boundary conditions), one per
    line: 'expr', 'expr  on sel  (right-hand side)', 'name := expr'. [] if
    the model cannot be generated now."""
    from . import codegen, parameters as P
    try:
        r = P.resolved(M.with_defaults(model))
        ctxs = codegen.contexts(r)
        c = ctxs[md_index]
        mod = dict(r["models"][md_index]["modules"][mod_index], conditions=[])     # (the equation only)
        w = codegen.ScriptWriter()
        declare, setup, _ = codegen.MODULES[mod["type"]]
        declare(c, mod, mod_index + 1, w)
        setup(c, mod, mod_index + 1, w)
        inertia = M.spec_of(mod).get("inertia")
        if inertia and c.transient and M.in_time(r, r["models"][md_index], mod):
            u = mod[inertia["variable"]]
            w(f"# Time derivative (Dot_{u}: by the scheme)")
            w(f"add_term(md, mim, {inertia['expr'].replace('{u}', u)!r})")
            for prop, dexpr in inertia.get("damping", {}).items():
                if not codegen._is_zero(mod.get("properties", {}).get(prop)):
                    w(f"add_term(md, mim, {dexpr.replace('{u}', u)!r})")
    except Exception:          # noqa: BLE001 (the model is not complete: no terms to show)
        return []
    out = []
    for line in w.lines:
        s = line.strip()
        if s.startswith("#"):
            out.append(s)
            continue
        m = _CALL.search(s)
        if not m:
            continue
        call, args = m.groups()
        rg = _RG.search(args)
        strs = _STR.findall(_RG.sub("", args))
        on = f"  on {rg.group(1)[1:-1]}" if rg else ""
        if call == "add_macro" and len(strs) >= 2:
            out.append(f"{strs[0]} := {strs[1]}")
        elif call == "add_initialized_fem_data" and len(strs) >= 2:
            out.append(f"{strs[0]} := {strs[1]}")
        elif call.startswith(("add_Dirichlet", "add_normal_Dirichlet")) and strs:
            data = strs[-1] if len(strs) > 1 else "0"
            out.append(f"{strs[0]} = {data}{on}  (Dirichlet)" if call.startswith("add_Dirichlet")
                       else f"{strs[0]}.n = {data}{on}  (Dirichlet)")
        elif strs:
            expr = next((x for x in strs if "Test_" in x), strs[0])
            out.append(f"{expr}{on}" + ("  (right-hand side)" if call == "add_source_term" else ""))
    return out
