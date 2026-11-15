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

"""What a module solves, for the form of its node: the equation (HTML,
with its data: properties per domain, sources, conditions) and the GWFL
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


def _pieces(mod: dict, prop: str) -> str:
    out = []
    for pc in mod.get("properties", {}).get(prop, []):
        what = f"material {html.escape(pc['material'])}" if "material" in pc else _v(pc.get("value"))
        out.append(f"{what} on {html.escape(str(pc.get('selection')))}")
    return "; ".join(out) or "<i>not given</i>"


def _sources(mod: dict, name: str) -> str:
    out = [f"{_v(src.get('value'))} on {html.escape(str(src.get('selection')))}" for src in mod.get("sources", [])]
    return f"{name} = " + ("; ".join(out) if out else "0")


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
    lines = [f"{lhs}{DIV}({k} {NABLA}{T}) = f &nbsp; in Ω"]
    for bc in mod.get("conditions", []):
        sel = html.escape(str(bc.get("selection")))
        t = bc.get("type")
        if t == "temperature":
            lines.append(f"{T} = {_v(bc.get('value'))} &nbsp; on {sel}")
        elif t == "heat_flux":
            lines.append(f"{k} {NABLA}{T}·n = {_v(bc.get('value'))} &nbsp; on {sel} (incoming flux)")
        elif t == "convection":
            lines.append(f"−{k} {NABLA}{T}·n = h ({T} − T<sub>ext</sub>), h = {_v(bc.get('h'))}, "
                         f"T<sub>ext</sub> = {_v(bc.get('T_ext'))} &nbsp; on {sel}")
    lines.append(f"{k} {NABLA}{T}·n = 0 &nbsp; on the rest of the boundary")
    data = [f"{k} ({'conductivity' if k == 'k' else 'conductivity tensor'}): {_pieces(mod, 'k')}",
            _sources(mod, "f")]
    if timed:
        data += [f"ρ (density): {_pieces(mod, 'rho')}", f"c<sub>p</sub> (specific heat): {_pieces(mod, 'cp')}"]
    return _time(model, md, mod) + _block(lines, data)


def _elasticity(model: dict, md: dict, mod: dict) -> str:
    u = html.escape(mod.get("variable", "u"))
    dim = M.model_dim(model, md) or 2
    expansion = any(m.get("type") == "thermal_expansion" and m.get("displacement") == mod.get("variable")
                    for m in md.get("modules", []))
    lines = [f"−{DIV} σ({u}) = f &nbsp; in Ω",
             "σ = λ tr(ε) I + 2μ ε" + (" − β α (T − T<sub>ref</sub>) I" if expansion else "")
             + f", &nbsp; ε = ({NABLA}{u} + {NABLA}{u}<sup>T</sup>)/2"]
    for bc in mod.get("conditions", []):
        sel = html.escape(str(bc.get("selection")))
        t = bc.get("type")
        if t == "fixed":
            lines.append(f"{u} = 0 &nbsp; on {sel}")
        elif t == "displacement":
            lines.append(f"{u} = {_v(bc.get('value'))} &nbsp; on {sel}")
        elif t == "roller":
            lines.append(f"{u}·n = 0, (σn)<sub>t</sub> = 0 &nbsp; on {sel}")
        elif t == "traction":
            lines.append(f"σn = {_v(bc.get('value'))} &nbsp; on {sel}")
        elif t == "pressure":
            lines.append(f"σn = −p n, p = {_v(bc.get('value'))} &nbsp; on {sel}")
    lines.append("σn = 0 &nbsp; on the rest of the boundary")
    plane = (f", plane {mod.get('plane', 'strain')}" if dim == 2 else "")
    data = [f"λ, μ from E (Young's modulus): {_pieces(mod, 'E')} and ν (Poisson's ratio): "
            f"{_pieces(mod, 'nu')}{plane}", _sources(mod, "f")]
    return _block(lines, data)


def _expansion(model: dict, md: dict, mod: dict) -> str:
    T, u = html.escape(str(mod.get("temperature"))), html.escape(str(mod.get("displacement")))
    lines = [f"σ({u}) gets − β α ({T} − T<sub>ref</sub>) I, &nbsp; β = 3λ + 2μ (in 3D)"]
    data = [f"α (expansion coefficient): {_pieces(mod, 'alpha')}", f"T<sub>ref</sub> = {_v(mod.get('T_ref'))}"]
    return _block(lines, data)


def _block(lines, data) -> str:
    eq = "<br>".join(lines)
    return (f"<p style='margin-left: 8px'>{eq}</p><p>with " + "<br>".join(data) + "</p>")


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
