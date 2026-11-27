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

"""The plots of a model (1D graphs of curves, computed with the solution):
along a segment (None outside the domain, a curve per time asked) or in
time (a value at a point, an integral, an average); validation, editing
(the tree, the forms, the actions), the examples; the plot panel of the
window (its curves, its CSV file).

    python -m pytest tests/test_plots.py      (or: python tests/test_plots.py)
"""

import copy
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from getfem_lab import editing as E, model as M, runner  # noqa: E402
import test_pde as P  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _getfem():
    try:
        import getfem  # noqa: F401
        return True
    except ImportError:
        return False


def run(m, name):
    assert M.validate(m) == [], M.validate(m)
    with tempfile.TemporaryDirectory() as tmp:
        r = runner.run(m, tmp)
    assert r.ok, f"{r.error}\n{r.stderr[-3000:]}"
    return r.results["models"][name]


def linear():
    """The default model (T = x on the unit square: 0 and 1 on xmin, xmax),
    a plot along y = 1/2 (from outside), one in time."""
    m = M.default_model()
    m["models"][0]["plots"] = [
        {"name": "profile", "type": "line", "title": "T", "curves": [
            {"label": "T", "expr": "T", "from": [-0.5, 0.5], "to": [1.5, 0.5], "points": 11},
            {"expr": "Grad_T(1)", "from": [0, "h"], "to": [1, "h"], "points": 5}]},
        {"name": "probe", "type": "time", "curves": [
            {"label": "center", "expr": "T", "point": [0.5, 0.5]},
            {"expr": "T", "at": "average"}, {"expr": "T", "at": "integral", "selection": "xmax"}]}]
    m["parameters"] = [{"name": "h", "value": 0.25}]
    return m


def test_validation_and_editing():
    """The errors of a plot; the tree (Plots, a plot, its curves; in the
    results), its form, adding a plot (a segment across the geometry at
    mid-height) and curves, duplicating, the times, the abscissa changed
    (the curves made over), the value of a time curve."""
    m = linear()
    assert M.validate(m) == []
    bad = copy.deepcopy(m)
    p = bad["models"][0]["plots"]
    p[0]["curves"][0]["to"] = [-0.5, 0.5]
    p[0]["curves"][1]["times"] = "first"
    p[0]["curves"][1]["points"] = 1
    p[1]["curves"][0]["point"] = [1, 2, 3]
    p[1]["curves"][1]["at"] = "maximum"
    p[1]["curves"][2]["selection"] = "nowhere"
    p.append({"name": "profile", "type": "surface", "curves": []})
    errors = M.validate(bad)
    for text in ("curves[0].to: a point other than its start", "times: 'last', 'all' or a list of times",
                 "points: ", "point: a point of 2 coordinates", "at: point, integral, average expected",
                 "unknown selection 'nowhere'", "name 'profile' used more than once", "type: line, time"):
        assert any(text in e for e in errors), (text, errors)
    root = E.tree(m, {"heat": ["T"]}, True, {"heat": ["profile"]})
    plots = E.find_node(root, ("models", 0, "plots"))
    assert [c.label for c in plots.children] == ["profile (along a line)", "probe (in time)"]
    assert [c.label for c in plots.children[1].children] == ["center at [0.5, 0.5]", "T (average on all)",
                                                             "T (integral on xmax)"]
    assert E.find_node(root, (E.RESULTS, "heat", "plots", "profile")).kind == "result_plot"
    assert [f.label for f in E.fields(m, ("models", 0, "plots", 0, "curves", 0))] == \
        ["Label", "Expression (GWFL)", "From A", "To B", "Number of points", "Times"]
    path = E.do_action(m, ("models", 0, "plots"), "add_plot")
    new = E.get_at(m, path)
    assert new["name"] == "plot" and new["curves"] == [
        {"label": "", "expr": "T", "from": [0.0, 0.5], "to": [1.0, 0.5], "points": M.PLOT_POINTS, "times": "last"}]
    cpath = E.do_action(m, path, "add_curve")
    assert cpath == path + ("curves", 1) and E.get_at(m, cpath)["from"] == [0.0, 0.5]
    times = next(f for f in E.fields(m, cpath) if f.key[-1] == "times")
    E.apply(m, times, "0.5, 1")
    assert E.get_at(m, cpath)["times"] == [0.5, 1.0] and E.field_value(m, times) == "0.5, 1"
    E.apply(m, times, "all")
    assert E.get_at(m, cpath)["times"] == "all"
    assert E.do_action(m, cpath, "duplicate_curve") == path + ("curves", 2) and len(new["curves"]) == 3
    E.do_action(m, path + ("curves", 2), "remove")
    E.apply(m, next(f for f in E.fields(m, path) if f.key[-1] == "type"), "time")
    assert new["curves"][0] == {"label": "", "expr": "T", "at": "point", "point": [0.5, 0.5]}
    at = next(f for f in E.fields(m, path + ("curves", 0)) if f.key[-1] == "at")
    E.apply(m, at, "average")
    assert new["curves"][0]["selection"] == M.ALL and "point" not in new["curves"][0]
    assert [f.label for f in E.fields(m, path + ("curves", 0))][-2:] == ["Value", "Selection"]
    assert M.validate(m) == []
    assert ("add_curve", "Add a curve") in E.actions(m, path)
    assert ("add_plot", "Add a plot") in E.actions(m, ("models", 0, "plots"))


def test_stationary():
    """T = x: along the line, None outside the domain, x inside (from its
    start at -0.5), the derivative 1 (a point given by a parameter); a
    single value in time at t = 0: T at the center, its average, its
    integral on xmax (the side at T = 1: 1)."""
    if not _getfem():
        return
    R = run(linear(), "heat")
    prof, probe = R["plots"]["profile"], R["plots"]["probe"]
    assert prof["xlabel"] == "distance from A" and prof["title"] == "T"
    c = prof["curves"][0]
    assert c["label"] == "T" and abs(c["x"][-1] - 2.0) < 1e-12 and len(c["y"]) == 11
    for x, y in zip(c["x"], c["y"]):
        if x < 0.5 - 1e-9 or x > 1.5 + 1e-9:
            assert y is None
        else:
            assert abs(y - (x - 0.5)) < 1e-9
    assert prof["curves"][1]["label"] == "Grad_T(1)" and max(abs(y - 1) for y in prof["curves"][1]["y"]) < 1e-9
    vals = [cv["y"] for cv in probe["curves"]]
    assert probe["xlabel"] == "t" and probe["curves"][0]["x"] == [0.0]
    assert abs(vals[0][0] - 0.5) < 1e-9 and abs(vals[1][0] - 0.5) < 1e-9 and abs(vals[2][0] - 1) < 1e-9


def test_transient():
    """u = (x + y) sin(2t) (BDF2): the profile at the times asked (the
    nearest saved ones) and at the end, the error at a point in time (each
    saved time)."""
    if not _getfem():
        return
    m, md, mod = P.pde(4, 1)
    P.option(m, "time_order", "1")
    mod["sources"] = [{"selection": "all", "value": "2*(x+y)*cos(2*t)"}]
    mod["conditions"] = [{"type": "imposed_value", "selection": "boundary", "value": "(x+y)*sin(2*t)"}]
    m["solvers"] = [{"name": "main", "type": "transient", "steps": ["pde"], "time": {"start": 0, "end": 0.5,
                     "step": 0.05}, "save": {"every": 2}, "schemes": {"pde/u": {"type": "bdf2"}}}]
    md["plots"] = [{"name": "prof", "type": "line", "curves": [
        {"expr": "u", "from": [0, 0.5], "to": [1, 0.5], "points": 3, "times": [0.21, 0.5]},
        {"label": "end", "expr": "u", "from": [0, 0.5], "to": [1, 0.5], "points": 3}]},
        {"name": "probe", "type": "time", "curves": [
            {"label": "error", "expr": "u - (x+y)*sin(2*t)", "point": [0.5, 0.5]}]}]
    R = run(m, "pde")
    curves = R["plots"]["prof"]["curves"]
    assert [c["label"] for c in curves] == ["u, t = 0.2", "u, t = 0.5", "end"]
    assert curves[1]["y"] == curves[2]["y"] and abs(curves[2]["y"][1] - math.sin(1.0)) < 2e-3
    err = R["plots"]["probe"]["curves"][0]
    assert err["x"] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] and max(abs(e) for e in err["y"]) < 1e-3


def test_examples():
    """The plots of the examples: the heating of the plate (the profiles at
    four times, the temperatures in time rising to 100), the busbar (the
    axis interrupted by the hole, as the edge elsewhere)."""
    if not _getfem():
        return
    R = run(M.load(EXAMPLES / "plate_heating.json"), "heat")
    assert [c["label"] for c in R["plots"]["profiles"]["curves"]] == [f"T, t = {t}" for t in (10, 20, 30, 60)]
    hist = R["plots"]["history"]["curves"]
    assert [c["label"] for c in hist] == ["center", "far edge", "mean"] and len(hist[0]["x"]) == 13
    assert all(c["y"][0] == 20.0 or abs(c["y"][0] - 20) < 1e-9 for c in hist)
    assert all(95 < c["y"][-1] < 100 for c in hist) and hist[1]["y"][3] < hist[0]["y"][3]
    R = run(M.load(EXAMPLES / "busbar_joule.json"), "busbar")
    axis, edge = R["plots"]["temperature"]["curves"]
    gap = [x for x, y in zip(axis["x"], axis["y"]) if y is None]
    assert abs(min(gap) - 0.045) < 1e-3 and abs(max(gap) - 0.055) < 1e-3
    assert 30 < max(edge["y"]) < 40 and abs(axis["y"][0] - 20) < 1e-6
    J = [y for y in R["plots"]["current"]["curves"][0]["y"] if y is not None]
    assert max(J) > 1.3 * J[0]                              # (the current concentrated by the hole)


def test_panel():
    """The plot panel: its curves (gaps), the legend, the CSV file."""
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from getfem_lab.app.plot import PlotPanel
    except ImportError:
        return
    QApplication.instance() or QApplication([])
    panel = PlotPanel()
    panel.resize(600, 400)
    data = {"type": "line", "title": "T", "xlabel": "distance from A",
            "curves": [{"label": "a", "x": [0, 1, 2], "y": [1.0, None, 3.0]}, {"label": "b", "x": [0, 2], "y": [0, 1]}]}
    panel.show_plot("profile", data)
    assert len(panel.plot.series) == 2 and panel.plot.title == "T" and "2 curve(s)" in panel.caption.text()
    assert panel.plot.ranges()[2] < 0 < 3 < panel.plot.ranges()[3]
    panel.grab()                                              # (painted: the gap, the legend)
    with tempfile.TemporaryDirectory() as tmp:
        panel.save_csv(Path(tmp) / "p.csv")
        lines = (Path(tmp) / "p.csv").read_text().splitlines()
    assert lines[0] == "curve,distance from A,value" and lines[2] == '"a",1,' and lines[4] == '"b",0,0'


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
