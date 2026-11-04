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

"""Exact 2D geometry: boundary pieces, 'edges' selections, evaluation of
selections on the geometry (the meshing tests require GetFEM).

    python -m pytest tests/       (or: python tests/test_geometry2d.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import geometry2d as G, mesher, model as M, parameters as P  # noqa: E402


def load(name):
    return M.load(Path(__file__).resolve().parents[1] / "examples" / name)


def rect(x0, y0, x1, y1):
    return {"type": "rectangle", "min": [x0, y0], "max": [x1, y1]}


def ball(x, y, r):
    return {"type": "ball", "center": [x, y], "radius": r}


def sides_of(pieces):
    return sorted((p.side.shape, p.side.side) for p in pieces)


def test_pieces_of_a_quarter_plate():
    shape = {"type": "difference", "of": [rect(0, 0, 0.1, 0.05), ball(0, 0, 0.01)]}
    pieces = {(p.side.shape, p.side.side): p for p in G.pieces(shape)}
    assert set(pieces) == {((0,), "xmin"), ((0,), "xmax"), ((0,), "ymin"), ((0,), "ymax"), ((1,), "circle")}
    left = pieces[((0,), "xmin")].criterion()["segment"]
    assert abs(left[0][1] - 0.01) < 1e-12 and abs(left[1][1] - 0.05) < 1e-12   # from the hole up
    arc = pieces[((1,), "circle")].criterion()["arc"]
    assert abs(arc["from"]) < 1e-9 and abs(arc["sweep"] - 90) < 1e-9


def test_shared_and_inner_sides_are_not_boundary():
    two = {"type": "union", "of": [rect(0, 0, 1, 1), rect(1, 0, 2, 1)]}
    assert ((0,), "xmax") not in sides_of(G.pieces(two)) and ((1,), "xmin") not in sides_of(G.pieces(two))
    lens = {"type": "intersection", "of": [ball(0, 0, 1), ball(1, 0, 1)]}
    arcs = G.pieces(lens)
    assert len(arcs) == 2 and all(abs(p.t1 - p.t0 - 1 / 3) < 1e-7 for p in arcs)   # 120 degrees each
    hole_inside = {"type": "difference", "of": [rect(0, 0, 1, 1), ball(0.5, 0.5, 0.2)]}
    circle = [p for p in G.pieces(hole_inside) if p.side.side == "circle"]
    assert len(circle) == 1 and circle[0].t0 == 0.0 and circle[0].t1 == 1.0


def test_arc_through_angle_zero():
    shape = {"type": "union", "of": [rect(-1, -1, 0, 1), ball(0, 0, 1)]}   # right half of the disk
    arc = [p for p in G.pieces(shape) if p.side.side == "circle"]
    assert len(arc) == 1
    a = arc[0].criterion()["arc"]
    # tangent to the sides of the rectangle at both ends: accurate to about 0.003 degree
    assert abs(a["from"] - 270) < 1e-2 and abs(a["sweep"] - 180) < 1e-2


def test_picking_and_toggling():
    shape = {"type": "difference", "of": [rect(0, 0, 0.1, 0.05), ball(0, 0, 0.01)]}
    pieces = G.pieces(shape)
    hit = G.nearest(pieces, (0.01 * math.cos(0.7), 0.01 * math.sin(0.7)), 1e-4)
    assert (hit.side.shape, hit.side.side) == ((1,), "circle")
    assert G.nearest(pieces, (0.05, 0.025), 1e-3) is None
    crit = G.toggle_edge({"type": "plane"}, hit.side.ref)
    assert crit == {"type": "edges", "edges": [{"shape": [1], "side": "circle"}]}
    assert G.toggle_edge(crit, hit.side.ref)["edges"] == []


def test_selections_evaluated_on_the_geometry():
    m = P.resolved(load("plate_with_hole.json"))
    geom = m["geometries"][0]
    shape, table = G.shape_of(geom["source"]), M.selection_table(m, geom)
    assert sides_of(G.pieces_matching(shape, "hole", table)) == [((1,), "circle")]     # in_ball
    assert sides_of(G.pieces_matching(shape, "left", table)) == [((0,), "xmin")]       # plane
    assert len(G.pieces_matching(shape, "boundary", table)) == 5
    assert G.pieces_matching(shape, "all", table) is None                               # a domain


def edges(*refs):
    return {"kind": "boundary", "criteria": {"type": "edges", "edges": [
        {"shape": list(s), "side": side} for s, side in refs]}}


def test_validation_of_edges():
    m = load("plate_hole_tension.json")
    sels = m["geometries"][0]["selections"]
    sels["e1"] = edges(((3,), "xmin"))
    sels["e2"] = edges(((1,), "ymin"))
    sels["e3"] = {"kind": "boundary", "criteria": {"type": "edges", "edges": []}}
    errors = M.validate(m)
    assert any("e1.criteria.edges[0]: no such primitive" in e for e in errors)
    assert any("e2.criteria.edges[0]: a ball has no side 'ymin'" in e for e in errors)
    assert any("e3.criteria.edges: no edge" in e for e in errors)
    m = load("plate.json")                                    # a box: its sides
    m["geometries"][0]["selections"]["west"] = edges(((), "xmin"))
    assert M.validate(m) == []


def test_edges_select_the_same_faces_and_follow_the_parameters():
    m = load("plate_hole_tension.json")
    sels = m["geometries"][0]["selections"]
    sels["left_e"] = edges(((0,), "xmin"))
    sels["right_e"] = edges(((0,), "xmax"))
    sels["hole_e"] = edges(((1,), "circle"))
    sels["two"] = edges(((0,), "xmin"), ((1,), "circle"))
    assert M.validate(m) == []
    with tempfile.TemporaryDirectory() as tmp:
        for a in (0.01, 0.02):
            m2 = copy.deepcopy(m)
            m2["parameters"][2]["value"] = a
            res = mesher.ensure_mesh(m2, "coarse", tmp)
            assert res.ok, res.error
            n = res.info["sizes"]
            assert n["left_e"] == n["symmetry_x"] and n["right_e"] == n["loaded_edge"]
            assert n["hole_e"] > 0 and n["two"] == n["left_e"] + n["hole_e"]
            ymax = n["boundary"] - n["symmetry_x"] - n["symmetry_y"] - n["loaded_edge"] - n["hole_e"]
            assert ymax > 0                       # the remaining side


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok       {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAILED   {name}: {exc!r}")
    sys.exit(1 if failed else 0)
