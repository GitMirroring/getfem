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

"""The layers of a graphic, drawn over its field: arrows of a vector field
(their number, their length), isolines of another field (its component, the
number of levels; on the boundary surface in 3D), in a color; their
validation, the settings panel (Add arrows, Add isolines, a setting, Remove),
the example of the busbar.

    python -m pytest tests/test_layers.py      (or: python tests/test_layers.py)
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pyvista as pv
except ImportError:  # pragma: no cover
    pv = None

from getfem_lab import editing as E, model as M  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def graphic(layers):
    m = M.default_model()
    m["models"][0]["graphics"] = [{"name": "T", "field": "T", "layers": layers}]
    return m


def test_validation():
    """The errors of the layers; a default setting not stored."""
    assert M.validate(graphic([{"type": "arrows", "field": "heat_flux", "count": 20, "scale": 2.0,
                                "color": "red"},
                               {"type": "isolines", "field": "T", "component": None, "count": 5}])) == []
    errors = M.validate(graphic([{"type": "streamlines", "field": "q"},
                                 {"type": "arrows", "field": "", "count": 0, "scale": -1, "color": "pink"},
                                 {"type": "isolines", "field": "T", "scale": 1.0, "component": -1}]))
    for text in ("layers[0]: a layer of type arrows, isolines", "layers[1].field: name of a result field",
                 "layers[1].count", "layers[1].scale: positive", "layers[1].color: black, white",
                 "layers[2].scale: unknown setting of isolines", "layers[2].component"):
        assert any(text in e for e in errors), (text, errors)
    assert any("layers: list expected" in e for e in M.validate(graphic({"type": "arrows"})))
    g = {"name": "T", "field": "T", "layers": [{"type": "arrows", "field": "q"}]}
    E.set_graphic(g, {"layers": []})
    assert "layers" not in g
    m = M.load(EXAMPLES / "busbar_joule.json")
    assert M.validate(m) == [] and [lay["type"] for lay in m["models"][0]["graphics"][0]["layers"]] == \
        ["arrows", "isolines"]


def square(three_d=False):
    """The unit square (cube) of tetrahedra/triangles: T = x, V = y, J = (1, 0)."""
    if three_d:
        ds = pv.ImageData(dimensions=(6, 6, 6), spacing=(0.2, 0.2, 0.2)).triangulate()
    else:
        ds = pv.Plane(center=(0.5, 0.5, 0), i_size=1, j_size=1, i_resolution=8,
                      j_resolution=8).triangulate().cast_to_unstructured_grid()
    p = np.asarray(ds.points)
    ds.point_data["T"] = p[:, 0]
    ds.point_data["V"] = p[:, 1]
    ds.point_data["J"] = np.c_[np.ones(len(p)), np.zeros(len(p))]
    return ds


def test_scene():
    """Over T: arrows of J (about count along the side, all along x, their
    length scaled), isolines of V (horizontal, count levels), a missing
    field skipped; in 3D, the isolines on the boundary."""
    if pv is None:
        return
    from getfem_lab.app import scene
    ds = square()
    pl = pv.Plotter(off_screen=True, window_size=(300, 200))
    scene.show_field(pl, ds, "T", layers=[
        {"type": "arrows", "field": "J", "count": 5, "scale": 2.0, "color": "red"},
        {"type": "isolines", "field": "V", "count": 3, "color": "white"},
        {"type": "isolines", "field": "nowhere"}])
    assert "layer0" in pl.actors and "layer1" in pl.actors and "layer2" not in pl.actors
    shown, _ = scene._with_values(ds, "T", None, None, 1.0)
    one = scene._arrows(shown, ds, "J", 5, 1.0)
    two = scene._arrows(shown, ds, "J", 5, 2.0)
    assert one.n_points and np.ptp(np.asarray(two.points)[:, 0]) > np.ptp(np.asarray(one.points)[:, 0])
    lines = pl.actors["layer1"].mapper.dataset
    y = np.round(np.asarray(lines.points)[:, 1], 6)
    assert sorted(set(y)) == [0.25, 0.5, 0.75]
    names = scene.add_layers(pl, shown, ds, [{"type": "isolines", "field": "J", "component": 0, "count": 2}])
    assert names == []                                        # (a constant: no level)
    ds3 = square(True)
    pl3 = pv.Plotter(off_screen=True, window_size=(300, 200))
    scene.show_field(pl3, ds3, "T", layers=[{"type": "isolines", "field": "V", "count": 4}])
    lines = pl3.actors["layer0"].mapper.dataset
    pts = np.asarray(lines.points)
    assert lines.n_points and np.all(np.min(np.minimum(pts, 1 - pts), axis=1) < 1e-9)   # (on the faces)


def test_panel():
    """The settings panel: a layer row each; Add arrows, Add isolines (of
    another field than the shown one), a setting changed, Remove: the
    layers emitted."""
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QComboBox, QPushButton, QSpinBox, QToolButton
        from getfem_lab.app.properties import PropertiesPanel
    except ImportError:
        return
    QApplication.instance() or QApplication([])
    panel = PropertiesPanel()
    emitted = []
    panel.viewChanged.connect(emitted.append)
    comps = {"T": 1, "V": 1, "J": 2}
    layers = [{"type": "arrows", "field": "J"}, {"type": "isolines", "field": "J", "count": 4}]
    panel.show_result_field("T", 1, {"layers": layers}, False, vectors=("J",), field_comps=comps)
    find = lambda cls, name: panel.findChild(cls, name)          # noqa: E731
    assert find(QComboBox, "layer0_field").currentData() == "J" and find(QSpinBox, "layer1_count").value() == 4
    assert find(QComboBox, "layer1_component") is not None and find(QComboBox, "layer0_component") is None
    find(QPushButton, "add_isolines").click()
    assert emitted[-1] == {"layers": layers + [{"type": "isolines", "field": "V"}]}
    find(QPushButton, "add_arrows").click()
    assert emitted[-1]["layers"][-1] == {"type": "arrows", "field": "J"}
    color = find(QComboBox, "layer1_color")
    color.activated.emit(color.findData("red"))
    assert emitted[-1]["layers"][1] == {"type": "isolines", "field": "J", "count": 4, "color": "red"}
    find(QToolButton, "layer0_remove").click()
    assert emitted[-1] == {"layers": [layers[1]]}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
