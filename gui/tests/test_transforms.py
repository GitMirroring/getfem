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

"""Transformations of shapes: a shape moved and turned (transform), copied
in a linear or circular array; the selections of its sides and faces hold
on every copy; editing (wrap, unwrap, handles).

    python -m pytest tests/       (or: python tests/test_transforms.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import (editing as E, geometry2d as G, geometry3d as G3, mesher, model as M,  # noqa: E402
                        parameters as P, runner, transforms as T)

R = 0.1
# a 4 x 1 plate with three holes: an array of a disk
PLATE = {"type": "difference", "of": [
    {"type": "rectangle", "min": [0.0, 0.0], "max": [4.0, 1.0]},
    {"type": "array", "kind": "linear", "count": 3, "step": [1.0, 0.0],
     "of": [{"type": "ball", "center": [1.0, 0.5], "radius": R}]}]}
HOLES = {"kind": "boundary", "criteria": {"type": "edges", "edges": [{"shape": [1, 0], "side": "circle"}]}}

# a unit cube with four pins around it: a circular array of a cylinder about the vertical axis of the cube
PINS = {"type": "union", "of": [
    {"type": "rectangle", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
    {"type": "array", "kind": "circular", "count": 4, "angle": 90.0, "center": [0.5, 0.5, 0.0],
     "axis": [0.0, 0.0, 1.0],
     "of": [{"type": "cylinder", "origin": [1.0, 0.5, 0.5], "axis": [1.0, 0.0, 0.0], "length": 0.3,
             "radius": 0.2}]}]}
ENDS = {"kind": "boundary", "criteria": {"type": "faces", "faces": [{"shape": [1, 0], "face": "end"}]}}


def model_with(shape, dim, mesh=None, selections=None, outputs=()):
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "csg", "dim": dim, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = dict(selections or {})
    m["meshes"][0] = dict({"name": "mesh", "geometry": "geometry"},
                          **(mesh or {"mesher": "getfem", "h": 0.08, "order": 1}))
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary",
                                       "value": "x + 2*y" + (" + 3*z" if dim == 3 else "")}]
    md["outputs"] = [{"name": "volume", "expr": "1", "selection": "all"}] + list(outputs)
    return m


def solve(m):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"{res.error}\n{res.stderr[-2000:]}"
        return res.results["models"]["heat"]["quantities"]


def test_maps():
    """A transform: turned about its center, then translated; an array: the
    copies k = 0..n-1."""
    (A, b), = T.maps({"type": "transform", "angle": 90, "center": [1, 0], "translate": [0, 2]}, 2)
    assert np.allclose(A @ [2, 0] + b, [1, 3])                     # (2, 0) about (1, 0): (1, 1), then + (0, 2)
    lin = T.maps({"type": "array", "kind": "linear", "count": 3, "step": [1, 2]}, 2)
    assert [list(b) for _, b in lin] == [[0, 0], [1, 2], [2, 4]]
    circ = T.maps({"type": "array", "kind": "circular", "count": 4, "angle": 90, "center": [0.5, 0.5, 0],
                   "axis": [0, 0, 1]}, 3)
    assert np.allclose([A @ [1, 0.5, 0.2] + b for A, b in circ], [[1, 0.5, 0.2], [0.5, 1, 0.2],
                                                                   [0, 0.5, 0.2], [0.5, 0, 0.2]])
    # composed along a path; flattened tree without transforms: the tree itself
    shape = {"type": "transform", "translate": [1, 0], "of": [
        {"type": "transform", "angle": 90, "center": [0, 0], "of": [{"type": "ball", "center": [1, 0], "radius": 1}]}]}
    A, b = T.frame_along(shape, (0, 0), 2)
    assert np.allclose(A @ [1, 0] + b, [1, 1])
    assert np.allclose(G.primitive_at(G.shape_of({"type": "csg", "dim": 2, "shape": shape}), (0, 0))["center"],
                       [1, 1])
    assert T.flatten(PLATE["of"][0], 2) is PLATE["of"][0]


def test_2d_flattened():
    """The copies of a disk: three holes, one path in the tree of the model;
    a turned rectangle keeps the names of its sides."""
    shape = G.shape_of({"type": "csg", "dim": 2, "shape": PLATE})
    prims = list(G.primitives(shape))
    assert [p for p, _ in prims] == [(0,), (1, 0), (1, 0), (1, 0)]
    assert [q["center"] for _, q in prims[1:]] == [[1.0, 0.5], [2.0, 0.5], [3.0, 0.5]]
    assert np.allclose(G.sdf(shape, np.array([[2.0, 0.5], [2.5, 0.5]])), [R, -0.4])
    assert len(G.pieces_of(shape, HOLES["criteria"]["edges"][0])) == 3
    assert G.side_label(shape, {"shape": [1, 0], "side": "circle"}) == "circle of ball 2"
    assert G.region_at(shape, [0.5, 0.5]) == [[0]]
    level = G.region_level(shape, [[[0]]])
    assert (level(np.array([[0.5, 0.5], [2.0, 0.5]])) < 0).tolist() == [True, False]
    turned = {"type": "transform", "angle": 30, "center": [0, 0], "translate": [1, 0],
              "of": [{"type": "rectangle", "min": [0, 0], "max": [2, 1]}]}
    flat = G.shape_of({"type": "csg", "dim": 2, "shape": turned})
    assert flat["type"] == "polygon" and [s.side for s in G.sides(flat)] == ["ymin", "xmax", "ymax", "xmin"]
    th = math.radians(30)
    ymax = next(s for s in G.sides(flat) if s.side == "ymax")
    assert np.allclose(ymax.b, [1 - math.sin(th), math.cos(th)])                # the corner (0, 1), moved
    # a turned ellipse: its angle grows
    ell = {"type": "transform", "angle": 30, "of": [{"type": "ellipse", "center": [0, 0], "radii": [2, 1]}]}
    assert abs(G.shape_of({"type": "csg", "dim": 2, "shape": ell})["angle"] - 30) < 1e-12


def test_2d_meshed():
    """The GetFEM mesher: the area of the plate with three holes, the
    selection of the circle of the disk is the three circles."""
    m = model_with(PLATE, 2, {"mesher": "getfem", "h": 0.05, "order": 2}, {"holes": HOLES},
                   [{"name": "holes_length", "expr": "1", "selection": "holes"}])
    assert M.validate(m) == []
    q = solve(m)
    assert abs(q["volume"] - (4 - 3 * math.pi * R ** 2)) < 1e-3
    assert abs(q["holes_length"] - 3 * 2 * math.pi * R) < 1e-2


def test_regions_of_copies():
    """A domain selection of a region of the copies: the three disks of an
    inclusion (union instead of difference), meshed with the region."""
    shape = dict(copy.deepcopy(PLATE), type="union")
    sel = {"kind": "domain", "criteria": {"type": "regions", "regions": [[[0], [1, 0]]]}}
    m = model_with(shape, 2, {"mesher": "getfem", "h": 0.04, "order": 1}, {"disks": sel},
                   [{"name": "disks_area", "expr": "1", "selection": "disks"}])
    code = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert "[0, 1, 1, 1]" in code                # the copies: one group
    q = solve(m)
    assert abs(q["disks_area"] - 3 * math.pi * R ** 2) < 3e-3


def test_3d_flattened():
    """A circular array of a cylinder: four pins, their faces selected
    together (face_spec 'any')."""
    shape = G3.shape_of({"type": "csg", "dim": 3, "shape": PINS})
    lo, hi = G3.bounds(shape)
    assert np.allclose(lo, [-0.3, -0.3, 0.0]) and np.allclose(hi, [1.3, 1.3, 1.0])
    P_ = np.array([[1.2, 0.5, 0.5], [0.5, 1.2, 0.5], [-0.2, 0.5, 0.5], [0.5, -0.2, 0.5], [1.2, 1.2, 0.5]])
    assert (G3.sdf(shape, P_) < 0).tolist() == [True, True, True, True, False]
    spec = G3.face_spec(shape, ENDS["criteria"]["faces"][0])
    assert spec["kind"] == "any" and len(spec["of"]) == 4
    for C, N in (([1.3, 0.5, 0.5], [1, 0, 0]), ([0.5, -0.3, 0.5], [0, -1, 0])):
        Q = np.asarray(C) + np.array([[0, 0, 0], [0, 0, 0.05], [0, 0.05, 0], [0.05, 0, 0]]) * (1 - np.abs(N))
        assert G3.on_face(spec, Q, np.asarray(N, float), 1e-6)
    assert G3.face_label(shape, {"shape": [1, 0], "face": "end"}) == "end of cylinder 2"
    assert G3.face_exists(shape, {"shape": [1, 0], "face": "side"})


def test_3d_meshed():
    """The GetFEM mesher ('placed' cylinders): the volume, the area of the
    four ends; the Gmsh and Netgen scripts."""
    m = model_with(PINS, 3, {"mesher": "getfem", "h": 0.06, "order": 1}, {"ends": ENDS},
                   [{"name": "ends_area", "expr": "1", "selection": "ends"}])
    assert M.validate(m) == []
    code = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert code.count("gf.MesherObject('placed'") == 3                   # the first copy is not moved
    q = solve(m)
    assert abs(q["volume"] - (1 + 4 * 0.3 * math.pi * 0.04)) < 2e-2
    assert abs(q["ends_area"] - 4 * math.pi * 0.04) < 2e-2
    for name in ("gmsh", "netgen"):
        m2 = model_with(PINS, 3, {"mesher": name, "size": 0.1})
        code = mesher.mesh_script(P.resolved(m2), m2["meshes"][0], "mesh")
        assert "placed in its frame" in code
        compile(code, "mesh.py", "exec")


def test_validation_and_parameters():
    m = model_with(PLATE, 2)
    m["parameters"] = [{"name": "n", "value": 3.0}, {"name": "d", "value": [1.0, 0.0]}]
    arr = m["geometries"][0]["source"]["shape"]["of"][1]
    arr.update(count="n", step="d")
    assert M.validate(m) == []
    assert P.resolved(m)["geometries"][0]["source"]["shape"]["of"][1]["count"] == 3
    for change, msg in [({"count": 0}, "positive integer"), ({"count": "n/2"}, "integer expected"),
                        ({"kind": "spiral"}, "'linear' or 'circular'"), ({"step": [1]}, "step: 2 numbers"),
                        ({"of": []}, "one shape expected"), ({"axis": [0, 0, 1]}, "only in 3D")]:
        bad = copy.deepcopy(m)
        bad["geometries"][0]["source"]["shape"]["of"][1].update(change)
        assert any(msg in e for e in M.validate(bad)), (change, M.validate(bad))
    bad = model_with({"type": "transform", "angle": "a", "of": [PINS]}, 3)
    assert any("angle" in e for e in M.validate(bad))


def test_editing():
    """Wrap a shape in a transform or an array (sensible parameters), the
    form, unwrap, remove; the references to the sides follow."""
    m = model_with(PLATE["of"][0], 2, selections={"right": {"kind": "boundary", "criteria": {
        "type": "edges", "edges": [{"shape": [], "side": "xmax"}]}}})
    root = ("geometries", 0, "source", "shape")
    acts = dict(E.actions(m, root))
    assert "transform:transform" in acts and "transform:array" in acts and "unwrap" not in acts
    E.do_action(m, root, "transform:transform")
    node = E.get_at(m, root)
    assert node["type"] == "transform" and node["center"] == [2.0, 0.5] and node["translate"] == [0.0, 0.0]
    assert E.get_at(m, root + ("of", 0))["type"] == "rectangle"
    sel = m["geometries"][0]["selections"]["right"]["criteria"]
    assert sel["edges"] == [{"shape": [0], "side": "xmax"}] and M.validate(m) == []
    labels = [f.label for f in E.fields(m, root)]
    assert "Translation" in labels and "Rotation (degrees)" in labels and "Axis of rotation" not in labels
    assert "unwrap" in dict(E.actions(m, root))
    assert E.tree(m).children  # the tree builds
    # the handles of the rectangle are moved and turned with it
    node.update(angle=90.0, translate=[1.0, 0.0])
    handles = dict(E.shape_handles(m, root + ("of", 0)))
    assert np.allclose(handles["min"], [3.5, -1.5])       # (0, 0) turned about (2, 0.5), then + (1, 0)
    E.move_handle(m, root + ("of", 0), "min", [3.5, -1.0])
    assert np.allclose(E.get_at(m, root + ("of", 0))["min"], [0.5, 0.0])
    E.do_action(m, root, "unwrap")
    assert E.get_at(m, root)["type"] == "rectangle" and sel["edges"] == [{"shape": [], "side": "xmax"}]
    # an array: three copies side by side; circular; removing its shape removes it
    E.do_action(m, root, "wrap:union")
    E.do_action(m, root, "add_shape:ball")
    disk = root + ("of", 1)
    E.do_action(m, disk, "transform:array")
    arr = E.get_at(m, disk)
    assert arr["count"] == 3 and arr["kind"] == "linear" and arr["step"][1] == 0 and arr["step"][0] > 0
    assert E._shape_label(arr) == "Array (linear, 3 copies)" and M.validate(m) == []
    kind = next(f for f in E.fields(m, disk) if f.op == "array_kind")
    E.apply(m, kind, "circular")
    assert arr["kind"] == "circular" and "step" not in arr and arr["angle"] == 90.0 and M.validate(m) == []
    assert "Angle between copies (degrees)" in [f.label for f in E.fields(m, disk)]
    E.do_action(m, disk + ("of", 0), "remove")
    assert E.get_at(m, root)["type"] == "rectangle"          # the union of one shape left: that shape


def test_layers_dropped():
    """An extrusion meshed in layers copied in an array: meshed as a whole."""
    m = M.load(Path(__file__).resolve().parents[1] / "examples" / "heat_sink_3d.json")
    assert m["meshes"][0].get("layers")
    E.do_action(m, ("geometries", 0, "source", "shape"), "wrap:array")
    assert "layers" not in m["meshes"][0] and M.validate(m) == []


def test_preview_surfaces():
    """The preview of the pins: the faces of the copies refer to the
    cylinder of the model; the part shown for the selected cylinder is
    placed."""
    try:
        from getfem_gui.app import scene
    except ImportError:
        return
    shape = G3.shape_of({"type": "csg", "dim": 3, "shape": PINS})
    surf = scene._surface_3d(shape, 30)
    refs = scene.face_refs(surf)
    ends = [i for i, r in enumerate(refs) if r == {"shape": [1, 0], "face": "end"}]
    assert len(ends) == 4
    ids = np.asarray(surf.cell_data["face_id"])
    C = np.asarray(surf.cell_centers().points)
    xs = sorted(round(float(C[ids == i][:, 0].mean()), 2) for i in ends)
    assert xs == [-0.3, 0.5, 0.5, 1.3]
    m = model_with({"type": "transform", "translate": [0, 0, 2.0], "of": [PINS]}, 3)
    part = E.part_3d(P.resolved(m), ("geometries", 0, "source", "shape", "of", 0, "of", 0))
    assert np.allclose(G3.bounds(part)[0], [0.0, 0.0, 2.0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
