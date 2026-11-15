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
          "thermal_expansion": _expansion}.get(mod.get("type"))
    return fn(model, md, mod) if fn else ""


def _time(model: dict, md: dict, mod: dict) -> str:
    """Where the module stands in time."""
    if not M.MODULE_SPECS.get(mod.get("type"), {}).get("inertia"):
        return ""
    s = M.main_solver(model)
    if M.in_time(mod) and M.is_transient(model, md):
        t = s.get("time", {})
        theta = s.get("schemes", {}).get(md.get("name"), {}).get("theta", 1)
        return (f"<p><b>In time</b>, by the solver {html.escape(s['name'])}: from t = {_v(t.get('start', 0))} "
                f"to {_v(t.get('end'))}, step {_v(t.get('step'))}, θ-method θ = {_v(theta)}; "
                f"from the initial value of the variable.</p>")
    if M.is_transient(model, md):
        return (f"<p><b>Stationary</b> at each time step of the solver {html.escape(s['name'])} "
                "(quasi-static: no time derivative).</p>")
    return "<p><b>Stationary.</b></p>"


def _heat(model: dict, md: dict, mod: dict) -> str:
    T = html.escape(mod.get("variable", "T"))
    timed = M.in_time(mod) and M.is_transient(model, md)
    lhs = f"ρ c<sub>p</sub> ∂{T}/∂t − " if timed else "−"
    k = "k" if mod.get("conductivity", "isotropic") == "isotropic" else "K"
    return _time(model, md, mod) + _block([f"{lhs}{DIV}({k} {NABLA}{T}) = f &nbsp; in Ω"])


def _elasticity(model: dict, md: dict, mod: dict) -> str:
    u = html.escape(mod.get("variable", "u"))
    expansion = any(m.get("type") == "thermal_expansion" and m.get("displacement") == mod.get("variable")
                    for m in md.get("modules", []))
    dim = M.model_dim(model, md) or 2
    plane = f" (plane {mod.get('plane', 'strain')})" if dim == 2 else ""
    return _block([f"−{DIV} σ({u}) = f &nbsp; in Ω",
                   "σ = λ tr(ε) I + 2μ ε" + (" − β α (T − T<sub>ref</sub>) I" if expansion else "")
                   + f", &nbsp; ε = ({NABLA}{u} + {NABLA}{u}<sup>T</sup>)/2",
                   f"λ, μ: from E and ν{plane}"])


def _expansion(model: dict, md: dict, mod: dict) -> str:
    T, u = html.escape(str(mod.get("temperature"))), html.escape(str(mod.get("displacement")))
    return _block([f"σ({u}) gets − β α ({T} − T<sub>ref</sub>) I, &nbsp; β = 3λ + 2μ (in 3D)"])


def _block(lines) -> str:
    return "<p style='margin-left: 8px'>" + "<br>".join(lines) + "</p>"


# ---------------------------------------------------------------- the GWFL terms of the script

_CALL = re.compile(r"(add_term|add_linear_term|add_source_term|add_nonlinear_term|add_macro|"
                   r"add_Dirichlet_condition_with_simplification|add_Dirichlet_condition_with_multipliers|"
                   r"add_normal_Dirichlet_condition_with_multipliers|add_initialized_fem_data)\((.*)\)")
_STR = re.compile(r"'((?:[^'\\]|\\.)*)'")
_RG = re.compile(r"RG_\w+\[('[^']*')\]")


def gwfl_terms(model: dict, md_index: int, mod_index: int) -> list[str]:
    """The GWFL terms that the script adds for a module (and its macros,
    its imposed values), one per line: 'expr', 'expr  (rhs) on sel',
    'name := expr', 'u = data on sel  (Dirichlet)'. [] if the model cannot
    be generated now."""
    from . import codegen, parameters as P
    try:
        r = P.resolved(M.with_defaults(model))
        ctxs = codegen.contexts(r)
        c = ctxs[md_index]
        mod = r["models"][md_index]["modules"][mod_index]
        w = codegen.ScriptWriter()
        declare, setup, _ = codegen.MODULES[mod["type"]]
        declare(c, mod, mod_index + 1, w)
        setup(c, mod, mod_index + 1, w)
        inertia = M.MODULE_SPECS[mod["type"]].get("inertia")
        if inertia and c.transient and M.in_time(mod):
            u = mod[inertia["variable"]]
            w(f"# Time derivative (Dot_{u}: by the scheme)")
            w(f"add_term(md, mim, {inertia['expr'].replace('{u}', u)!r})")
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
