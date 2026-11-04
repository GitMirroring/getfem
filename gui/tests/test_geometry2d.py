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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, geometry2d as G, mesher, model as M, parameters as P  # noqa: E402


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


def test_changing_a_selection_does_not_mesh_again():
    m = load("plate_hole_tension.json")
    with tempfile.TemporaryDirectory() as tmp:
        first = mesher.ensure_mesh(m, "coarse", tmp)
        script = Path(tmp) / mesher.MESH_DIR / "coarse.py"
        assert "Mesh('generate'" in script.read_text()
        m["geometries"][0]["selections"]["hole_e"] = edges(((1,), "circle"))
        second = mesher.ensure_mesh(m, "coarse", tmp)
        assert second.ok and not second.cached
        text = script.read_text()
        assert "gf.Mesh('load', 'coarse_raw.mesh')" in text and "Mesh('generate'" not in text
        assert second.info["nb_points"] == first.info["nb_points"] and second.info["sizes"]["hole_e"] > 0
        m["parameters"][2]["value"] = 0.012                 # the geometry changes: meshed again
        assert mesher.ensure_mesh(m, "coarse", tmp).ok
        assert "Mesh('generate'" in script.read_text()


def test_drawn_shapes_and_grid():
    assert G.grid_step(1.0) == 0.05 and G.grid_step(0.25) == 0.01 and G.grid_step(47) == 2.0
    assert G.snap([0.01234, 0.0049], 0.005) == [0.01, 0.005]
    assert G.drawn_shape("rectangle", [1, 2], [0, 0]) == {"type": "rectangle", "min": [0, 0], "max": [1, 2]}
    assert G.drawn_shape("ball", [0, 0], [3, 4])["radius"] == 5.0
    assert G.drawn_shape("rectangle", [0, 0], [1, 0]) is None


def test_adding_a_shape_keeps_the_selections():
    m = load("plate_hole_tension.json")
    geom = m["geometries"][0]
    geom["selections"]["hole_e"] = edges(((1,), "circle"))
    def targets():
        shape = G.shape_of(P.resolved(m)["geometries"][0]["source"])
        return {n: [G.primitive_at(shape, r["shape"]) for r in s["criteria"]["edges"]]
                for n, s in geom["selections"].items() if s["criteria"]["type"] == "edges"}
    before = targets()
    path, _ = E.add_shape_2d(m, 0, ball(0.08, 0.025, 0.005), "difference")   # same operation
    assert path[-2:] == ("of", 2) and geom["source"]["shape"]["type"] == "difference"
    path, _ = E.add_shape_2d(m, 0, rect(0.1, 0.0, 0.12, 0.02), "union")      # the root is wrapped
    assert geom["source"]["shape"]["type"] == "union" and path[-2:] == ("of", 1)
    assert geom["selections"]["hole_e"]["criteria"]["edges"][0]["shape"] == [0, 1]
    assert targets() == before and M.validate(m) == []


def test_drawing_on_a_box_and_solving():
    from getfem_gui import runner
    m = load("plate.json")
    m["geometries"][0]["selections"]["west"] = edges(((), "xmin"))
    uses = {bc["selection"] for bc in m["physics"]["boundary_conditions"]}
    path, notes = E.add_shape_2d(m, 0, ball(0.1, 0.05, 0.02), "difference")
    geom = m["geometries"][0]
    assert geom["source"]["type"] == "csg" and len(notes) == 2
    assert uses <= set(geom["selections"])                       # xmin... are now selections
    assert geom["selections"]["west"]["criteria"]["edges"] == [{"shape": [0], "side": "xmin"}]
    assert m["meshes"][0]["mesher"] == "getfem" and m["meshes"][0]["h"] > 0
    assert M.validate(m) == []
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, res.error
        assert res.mesh.info["sizes"]["west"] == res.mesh.info["sizes"]["xmin"] > 0


def test_handles_numbers_parameters_and_locked_values():
    m = load("plate_hole_tension.json")
    ball_p = ("geometries", 0, "source", "shape", "of", 1)
    rect_p = ("geometries", 0, "source", "shape", "of", 0)
    assert dict(E.shape_handles(m, ball_p)) == {"center": [0.0, 0.0], "radius": [0.01, 0.0]}
    assert len(E.shape_handles(m, rect_p)) == 5 and E.shape_handles(m, ("geometries", 0)) == []
    # a bare parameter name: the parameter changes, the geometry follows
    assert E.move_handle(m, ball_p, "radius", [0.0, 0.015]) == ["parameter a = 0.015"]
    assert E.get_at(m, ball_p + ("radius",)) == "a" and m["parameters"][2]["value"] == 0.015
    # an expression is locked, and nothing changes
    m["geometries"][0]["source"]["shape"]["of"][0]["max"] = ["2*H", "H"]
    before = copy.deepcopy(m)
    try:
        E.move_handle(m, rect_p, "center", [0.06, 0.03])
        raise AssertionError("locked value moved")
    except E.LockedValue as exc:
        assert "'2*H'" in str(exc)
    assert m == before
    # numbers; a corner dragged beyond the opposite one
    m = load("plate_with_hole.json")
    r = ("geometries", 0, "source", "shape", "of", 0)
    E.move_handle(m, r, "min", [0.3, 0.2])
    assert E.get_at(m, r) == {"type": "rectangle", "min": [0.2, 0.1], "max": [0.3, 0.2]}
    try:
        E.move_handle(m, r, "max", [0.2, 0.3])                  # zero width
        raise AssertionError("empty rectangle accepted")
    except ValueError:
        pass
    b = ("geometries", 0, "source", "shape", "of", 1)
    E.move_handle(m, b, "center", [0.25, 0.15])
    assert E.get_at(m, b + ("center",)) == [0.25, 0.15]


def test_a_hole_drawn_in_a_difference_is_meshed():
    """Subtracting a disk from a geometry that is already a difference gives
    a third operand: all of them must reach the mesher (and a condition on
    the new hole, picked from a new condition, must select its faces)."""
    from getfem_gui import runner
    m = load("plate_hole_tension.json")
    with tempfile.TemporaryDirectory() as tmp:
        before = mesher.ensure_mesh(m, "coarse", tmp).info["nb_elements"]
        E.add_shape_2d(m, 0, ball(0.06, 0.025, 0.01), "difference")
        assert len(m["geometries"][0]["source"]["shape"]["of"]) == 3
        E.do_action(m, ("physics",), "add_bc")
        bc = ("physics", "boundary_conditions", len(m["physics"]["boundary_conditions"]) - 1)
        E.apply(m, E.Field(bc + ("type",), "", "choice", op="bc_type"), "fixed")
        assert E.get_at(m, bc + ("selection",)) == "boundary"      # the default
        name, _ = E.pick_edge_for_item(m, bc, {"shape": [2], "side": "circle"})
        assert m["geometries"][0]["selections"][name]["criteria"]["edges"] == [
            {"shape": [2], "side": "circle"}]                         # this edge only
        res = runner.run(m, tmp)
        assert res.ok, res.error
        n = res.mesh.info
        assert n["nb_elements"] < before                             # the hole is there
        assert 0 < n["sizes"][name] < n["sizes"]["loaded_edge"] * 3   # its circle only


def test_picking_from_a_condition_edits_its_support():
    m = load("plate_hole_tension.json")
    bc = ("physics", "boundary_conditions", 2)
    assert E.can_pick_edges(m, bc) and not E.can_pick_edges(m, ("materials", 0))
    assert E.selection_uses(m, "loaded_edge") == 2          # the condition and an output
    name, created = E.pick_edge_for_item(m, bc, {"shape": [0], "side": "ymax"})
    assert (name, created) == ("traction_3", True)
    assert m["geometries"][0]["selections"]["loaded_edge"]["criteria"]["type"] == "plane"   # kept
    assert m["outputs"]["quantities"][0]["selection"] == "loaded_edge"
    assert E.pick_edge_for_item(m, bc, {"shape": [1], "side": "circle"}) == ("traction_3", False)
    assert E.pick_edge_for_item(m, bc, {"shape": [1], "side": "circle"}) == ("traction_3", False)
    assert M.validate(m) == []
    with tempfile.TemporaryDirectory() as tmp:
        n = mesher.ensure_mesh(m, "coarse", tmp).info["sizes"]
        assert n["traction_3"] > n["loaded_edge"]          # the loaded edge and the top edge
    # an output integrated on a boundary: same mechanism
    q = ("outputs", "quantities", 1)                           # reaction on symmetry_x
    name, created = E.pick_edge_for_item(m, q, {"shape": [1], "side": "circle"})
    assert created and m["outputs"]["quantities"][1]["selection"] == name
    assert m["geometries"][0]["selections"][name]["criteria"]["edges"] == [
        {"shape": [0], "side": "xmin"}, {"shape": [1], "side": "circle"}]
    # a built-in selection (box): starts from its side
    m = load("plate.json")
    bc = ("physics", "boundary_conditions", 0)
    side = m["physics"]["boundary_conditions"][0]["selection"]
    name, created = E.pick_edge_for_item(m, bc, {"shape": [], "side": "ymin"})
    assert created and m["geometries"][0]["selections"][name]["criteria"]["edges"] == [
        {"shape": [], "side": side}, {"shape": [], "side": "ymin"}]
    assert M.validate(m) == []


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


# ------------------------------------------------------------------ regions

def test_regions_and_interfaces():
    plate = {"type": "difference", "of": [
        {"type": "union", "of": [rect(0, 0, 1, 1), ball(0.5, 0.5, 0.2), ball(1.0, 0.5, 0.2)]},
        ball(0.5, 0.5, 0.05)]}
    assert G.region_at(plate, (0.1, 0.1)) == [[0, 0]]
    assert G.region_at(plate, (0.5, 0.62)) == [[0, 0], [0, 1]]
    assert G.region_at(plate, (1.1, 0.5)) == [[0, 2]]                # outside the rectangle
    assert G.region_at(plate, (0.5, 0.5)) is None                    # in the hole
    assert G.region_at(plate, (2, 2)) is None
    # interfaces: the circle of the inclusion, the arc of the half disk, the
    # part of xmax inside it; not the hole (a boundary)
    found = {(p.side.shape, p.side.side): (p.t0, p.t1) for p in G.interfaces(plate)}
    assert set(found) == {((0, 1), "circle"), ((0, 0), "xmax"), ((0, 2), "circle")}
    t0, t1 = found[((0, 0), "xmax")]
    assert abs(t0 - 0.3) < 1e-9 and abs(t1 - 0.7) < 1e-9
    V = G.interface_points(plate, 0.05)
    assert V.shape[0] == 2
    d = np.linalg.norm(V[:, :, None] - V[:, None, :], axis=0) + np.eye(V.shape[1])
    assert d.min() > 0.0125                                          # shared ends once
    level = G.region_level(plate, [[[0, 0], [0, 1]], [[0, 2]]])
    v = level(np.array([[0.5, 0.62], [1.1, 0.5], [0.1, 0.1], [0.5, 0.5]]))
    assert v[0] < 0 and v[1] < 0 and v[2] > 0 and v[3] > 0
    outline = sides_of(G.pieces(plate, level=G.region_level(plate, [[[0, 0], [0, 1]]])))
    assert outline == [((0, 1), "circle"), ((1,), "circle")]
    assert G.region_label(plate, [[0, 0], [0, 1]]) == "rectangle 1 + ball 2"


def test_toggle_region_and_domain_level():
    c = G.toggle_region({"type": "in_box"}, [[1], [0]])
    assert c == {"type": "regions", "regions": [[[0], [1]]]}
    c = G.toggle_region(c, [[0]])
    assert c["regions"] == [[[0], [1]], [[0]]]
    assert G.toggle_region(c, [[1], [0]])["regions"] == [[[0]]]
    m = load("plate_inclusion.json")
    r = P.resolved(m)
    geom = r["geometries"][0]
    shape, table = G.shape_of(geom["source"]), M.selection_table(r, geom)
    P_ = np.array([[0.05, 0.05], [0.01, 0.01]])
    assert list(G.domain_level(shape, "inclusion", table)(P_) < 0) == [True, False]
    assert list(G.domain_level(shape, "matrix", table)(P_) < 0) == [False, True]
    assert list(G.domain_level(shape, "all", table)(P_) < 0) == [True, True]
    assert G.domain_level(shape, "cold", table) is None


def test_regions_follow_the_geometry_edits():
    """Regions are paths of primitives: they follow when the root of the
    geometry is wrapped by a drawn shape, and validation checks them."""
    m = load("plate_inclusion.json")
    sels = m["geometries"][0]["selections"]
    E.add_shape_2d(m, 0, ball(0.0, 0.0, 0.01), "difference")
    assert sels["inclusion"]["criteria"]["regions"] == [[[0, 0], [0, 1]]]
    assert sels["matrix"]["criteria"]["regions"] == [[[0, 0]]]
    assert M.validate(m) == []
    sels["matrix"]["criteria"]["regions"] = [[[5]]]
    assert any("matrix" in e and "no such primitive" in e for e in M.validate(m))
    sels["matrix"]["criteria"]["regions"] = []
    assert any("no region" in e for e in M.validate(m))
    sels["matrix"]["kind"] = "boundary"
    assert any("only applies to domain" in e for e in M.validate(m))


def test_mesh_script_has_interface_vertices():
    m = load("plate_inclusion.json")
    r = P.resolved(m)
    script = mesher.mesh_script(r, r["meshes"][0], "mesh")
    assert "interface_vertices = np.array(" in script
    assert "cvs_in_regions(PRIMITIVES, [[True, True]])" in script
    m = load("plate_with_hole.json")
    r = P.resolved(m)
    assert "interface_vertices" not in mesher.mesh_script(r, r["meshes"][0], "mesh")
    m = load("plate_inclusion.json")
    new = E.do_action(m, ("geometries", 0, "source", "shape"), "add_shape:ball")
    assert E.get_at(m, new) == {"type": "ball", "center": [0.05, 0.05], "radius": 0.0125}
