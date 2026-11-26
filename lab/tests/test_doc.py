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

"""The user documentation (doc/sphinx/source/lab of the GetFEM sources).

- the reference pages are up to date with the code (otherwise: run
  ``python -m getfem_lab.docref``);
- every page is in a table of contents, every image exists, every example
  has its section, the documentation is linked from the GetFEM one;
- the results quoted by the page of the examples hold;
- with Sphinx installed, the pages build without warnings.

    python -m pytest tests/       (or: python tests/test_doc.py)
"""

import importlib.util
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import docref, model as M, runner  # noqa: E402

DOC = docref.DEFAULT_DIR.parent
SOURCE = DOC.parent
EXAMPLES = docref.EXAMPLES


def _pages():
    return sorted(DOC.rglob("*.rst"))


def test_reference_is_up_to_date():
    stale = [name for name, text in docref.pages().items()
             if not (docref.DEFAULT_DIR / name).exists()
             or (docref.DEFAULT_DIR / name).read_text(encoding="utf-8") != text]
    assert not stale, f"run  python -m getfem_lab.docref  (out of date: {', '.join(stale)})"
    generated = set(docref.PAGES) | {"index.rst"}
    extra = [p.name for p in docref.DEFAULT_DIR.iterdir() if p.name not in generated]
    assert not extra, f"files of the reference that are not generated: {extra}"


def _toctree_entries(page: Path) -> list[Path]:
    out, inside = [], False
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.startswith(".. toctree::"):
            inside = True
            continue
        if inside:
            if line and not line.startswith(" "):
                inside = False
            elif line.strip() and not line.strip().startswith(":"):
                out.append((page.parent / line.strip()).with_suffix(".rst"))
    return out


def test_structure():
    listed = {DOC / "index.rst"}
    todo = [DOC / "index.rst"]
    while todo:
        for p in _toctree_entries(todo.pop()):
            assert p.exists(), p
            if p not in listed:
                listed.add(p)
                todo.append(p)
    missing = [str(p.relative_to(DOC)) for p in _pages() if p not in listed]
    assert not missing, f"pages in no table of contents: {missing}"
    # images, labels
    labels = set()
    for p in _pages():
        text = p.read_text(encoding="utf-8")
        for img in re.findall(r"^\.\. (?:image|figure):: (\S+)", text, re.M):
            assert (p.parent / img).exists(), (p, img)
        labels |= set(re.findall(r"^\.\. _([\w-]+):", text, re.M))
    for p in _pages():
        for ref in re.findall(r":ref:`(?:[^`<]*<)?(lab-[\w-]+)>?`", p.read_text(encoding="utf-8")):
            assert ref in labels, (p, ref)
    for ex in EXAMPLES.glob("*.json"):
        assert f"lab-example-{ex.stem}" in labels, f"examples.rst: no section for {ex.name}"
    # the documentation of the library links to it
    assert "lab/index.rst" in (SOURCE / "contents.rst").read_text(encoding="utf-8")
    assert 'pathto("lab/index")' in (SOURCE / ".templates" / "indexcontent.html").read_text(encoding="utf-8")


def _solve(name):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(M.load(EXAMPLES / name), tmp)
        assert res.ok, f"{name}: {res.error}\n{res.stderr[-2000:]}"
        return {k: v.get("quantities", {}) for k, v in res.results["models"].items()}


def _close(a, b, rel):
    return abs(a - b) <= rel * max(abs(a), abs(b))


def test_examples_quoted_results():
    """Every example solves; the balances stated in examples.rst hold."""
    q = _solve("plate.json")["heat"]
    assert _close(q["injected_power"], q["outflow_xmin"] + q["outflow_convection"], 1e-3)
    q = _solve("plate_hole_tension.json")["mechanics"]
    assert _close(q["applied_force"], 1e8 * 0.05, 1e-3)               # S times the half height
    assert _close(q["applied_force"], -q["reaction_symmetry_x"], 2e-3)
    q = _solve("l_bracket.json")["mechanics"]
    assert _close(q["applied_force"], -2e6 * 0.1, 2e-3)                # F times W
    assert _close(q["applied_force"], -q["reaction"], 2e-2)
    q = _solve("heat_sink_3d.json")["heat"]
    assert _close(q["injected_power"], 1e4 * 0.06 * 0.05, 1e-4)        # qin times the base: 30 W
    assert _close(q["injected_power"], q["convected_power"], 1e-4)
    q = _solve("heated_dish.json")["heat"]
    f, r0, r1 = 0.4, 0.2, 0.5                                           # area of the paraboloid
    area = 2 * math.pi * (2 * f) ** 2 / 3 * ((1 + (r1 / (2 * f)) ** 2) ** 1.5 - (1 + (r0 / (2 * f)) ** 2) ** 1.5)
    assert _close(q["area"], area, 1e-3), (q["area"], area)
    q = _solve("heated_wire.json")["heat"]
    L, R = 0.3, 0.05                                                    # the wire: its sides less the
    length = 4 * L - 4 * (2 * R - math.pi * R / 2)                      # corners, plus its four arcs
    assert _close(q["length"], length, 1e-6), (q["length"], length)
    assert _close(q["heat_out"], 2000 * length, 1e-6)                  # the heat generated leaves by the ends
    q = _solve("heated_tray.json")["heat"]
    L, W, H = 0.4, 0.3, 0.1                                             # the faces of the box but its top
    assert _close(q["area"], L * W + 2 * H * (L + W), 1e-8) and _close(q["bottom"], L * W, 1e-8)
    assert _close(q["heat_out"], 500 * L * W, 1e-3)                    # the heat received leaves by the rim
    q = _solve("plate_heating.json")["heat"]                          # in time: the final values
    assert 95 < q["mean_T"] < 100 and q["heat_in"] > 0
    with tempfile.TemporaryDirectory() as tmp:                        # elastodynamics
        res = runner.run(M.load(EXAMPLES / "beam_vibration.json"), tmp)
    assert res.ok, res.error
    S = res.results["models"]["beam"]["series"]
    t, d, e = (np.array(S[k]) for k in ("t", "tip_deflection", "energy"))
    after = t > 0.0051                                                  # after the pulse
    cross = t[after][1:][np.sign(d[after][1:]) != np.sign(d[after][:-1])]
    E, nu, rho, h, L = 210e9 / (1 - 0.3 ** 2), 0.3, 7800, 0.05, 1.0    # the first bending mode
    w1 = 3.516 * math.sqrt(E * h ** 3 / 12 / (rho * h * L ** 4))
    assert _close(2 * np.mean(np.diff(cross)), 2 * math.pi / w1, 3e-2)
    assert abs(e[-1] / e[after][0] - 1) < 1e-3
    m = M.load(EXAMPLES / "thermal_shock_strip.json")                 # thermally induced vibrations
    series = []
    for sc in (None, {"type": "stationary"}):
        if sc:
            m["solvers"][0]["schemes"]["mechanical/u"] = sc
        with tempfile.TemporaryDirectory() as tmp:
            res = runner.run(m, tmp)
        assert res.ok, res.error
        series.append(res.results["models"])
    t = np.array(series[0]["thermal"]["series"]["t"])
    Tm = np.array(series[0]["thermal"]["series"]["T_mean"])
    assert np.allclose(Tm - 20, 1e6 * t / (2700 * 897 * 1e-3), rtol=1e-6, atol=1e-9)   # the heat received
    dyn, qs = (np.array(r["mechanical"]["series"]["tip_deflection"]) for r in series)
    assert 1.3 < dyn.min() / qs[-1] < 1.45 and np.all(np.diff(qs) < 1e-4 * abs(qs[-1]))   # overshoot; monotone
    x = dyn - qs                                                         # the vibration about it
    k = np.where(np.sign(x[2:]) != np.sign(x[1:-1]))[0] + 1
    cross = t[k] - x[k] * (t[k + 1] - t[k]) / (x[k + 1] - x[k])
    E, rho, h, L = 70e9 / (1 - 0.33 ** 2), 2700, 1e-3, 0.05
    w1 = 3.516 * math.sqrt(E * h ** 2 / 12 / (rho * L ** 4))
    assert _close(2 * np.mean(np.diff(cross)), 2 * math.pi / w1, 3e-2)
    q = _solve("heated_pipe.json")["heat"]
    R, L = 0.05, 0.6                                                    # the tube; heated on a half
    assert _close(q["area"], 2 * math.pi * R * L, 1e-4) and _close(q["heated_area"], math.pi * R * L, 1e-4)
    assert _close(q["heat_out_inlet"] + q["heat_out_outlet"], 1000 * math.pi * R * L, 1e-3)
    q = _solve("anisotropic_plate.json")["heat"]
    assert _close(q["injected_power"], q["outflow"], 1e-3)
    assert q["flux_along_fibres"] > 3 * q["flux_across_fibres"]
    q = _solve("flange_2d.json")["heat"]
    assert _close(q["bolt_holes_length"], 6 * 2 * math.pi * 0.004, 1e-3)    # the n holes of the array
    assert _close(q["power_in"], q["power_bolts"] + q["power_rim"], 1.5e-2)
    q = _solve("pin_fin_heat_sink_3d.json")["heat"]
    assert _close(q["injected_power"], 2e4 * 0.03 ** 2, 1e-2)              # qin times the base: 18 W
    assert _close(q["injected_power"], q["convected_power"], 1e-2)
    r, F = 0.0025, 0.02                                                    # the 9 pins in the air
    assert _close(q["pins_area"], 9 * (2 * math.pi * r * F + math.pi * r ** 2), 6e-2)
    for name in ("plate_with_hole.json", "plate_inclusion.json", "thermoelastic_plate.json",
                 "block_with_hole_3d.json", "pulley_3d.json"):
        assert all(v for v in _solve(name).values())


def test_make_check_lists_the_tests():
    """`make check` (lab/Makefile.am) runs every test file of the interface."""
    here = Path(__file__).resolve().parent
    listed = re.findall(r"tests/(test_\w+\.py)", (here.parent / "Makefile.am").read_text(encoding="utf-8"))
    assert sorted(listed) == sorted(p.name for p in here.glob("test_*.py"))


def test_sphinx_build():
    """The pages, with stubs for the labels of the rest of the GetFEM
    documentation, build without any warning."""
    if importlib.util.find_spec("sphinx") is None:
        if "pytest" in sys.modules:
            import pytest
            pytest.skip("Sphinx is not installed")
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "source"
        shutil.copytree(DOC, src / "lab")
        shutil.copy(SOURCE / "replaces.txt", src / "replaces.txt")
        own, outside, docs = set(), set(), set()
        for p in _pages():
            text = p.read_text(encoding="utf-8")
            own |= set(re.findall(r"^\.\. _([\w-]+):", text, re.M))
            outside |= set(re.findall(r":ref:`(?:[^`<]*<)?([\w-]+)>?`", text))
            docs |= {(p.parent / d).resolve() for d in re.findall(r":doc:`(?:[^`<]*<)?([^`>]+)>?`", text)}
        stubs = ["Stubs", "=====", ""] + [f".. _{x}:\n\n{x}\n{'-' * len(x)}\n" for x in sorted(outside - own)]
        (src / "stubs.rst").write_text("\n".join(stubs), encoding="utf-8")
        entries = ["lab/index", "stubs"]
        for d in docs:
            rel = d.relative_to(SOURCE.resolve())
            (src / rel.parent).mkdir(parents=True, exist_ok=True)
            (src / rel).with_suffix(".rst").write_text(f"{rel}\n{'=' * len(str(rel))}\n", encoding="utf-8")
            entries.append(str(rel))
        (src / "index.rst").write_text("Doc\n===\n\n.. toctree::\n\n" + "".join(f"   {e}\n" for e in entries),
                                       encoding="utf-8")
        (src / "conf.py").write_text("extensions = ['sphinx.ext.mathjax']\nproject = 'GetFEM'\n"
                                     "master_doc = 'index'\n", encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "sphinx", "-b", "html", "-q", "-W", "--keep-going",
                            str(src), str(Path(tmp) / "html")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-4000:]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
