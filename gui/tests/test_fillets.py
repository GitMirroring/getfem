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

"""Fillets of polygons: vertices rounded by arcs of circle tangent to their
two sides; the exact geometry, its sides (selections), the meshers,
editing (the vertex table, handles, renumbering), 3D profiles, export.

    python -m pytest tests/       (or: python tests/test_fillets.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import (editing as E, export, geometry2d as G, geometry3d as G3, mesher,  # noqa: E402
                        model as M, parameters as P, runner)

# an L: its vertex 1 (convex) and 4 (reflex) rounded
L_SHAPE = {"type": "polygon", "points": [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]],
           "fillets": [{"vertex": 1, "radius": "rf"}, {"vertex": 4, "radius": 0.3}, {"vertex": 5, "radius": 0.1}]}
CUT = 1 - math.pi / 4                         # the area of a corner of a square of side 1, out of the disk
L_AREA = 3 - CUT * 0.04 + CUT * 0.09 - CUT * 0.01
ROUND = {"kind": "boundary", "criteria": {"type": "edges", "edges": [{"shape": [], "side": "fillet1"},
                                                                     {"shape": [], "side": "fillet4"}]}}
ROOT = ("geometries", 0, "source", "shape")


def model_with(shape, dim=2, mesh=None, selections=None, outputs=()):
    m = M.default_model()
    m["parameters"] = [{"name": "rf", "value": 0.2}]
    m["geometries"][0]["source"] = {"type": "csg", "dim": dim, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = dict(selections or {})
    m["meshes"][0] = dict({"name": "mesh", "geometry": "geometry"},
                          **(mesh or {"mesher": "getfem", "h": 0.05, "order": 2}))
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": "x + 2*y"}]
    md["outputs"] = [{"name": "area", "expr": "1", "selection": "all"}] + list(outputs)
    return m


def resolved(shape):
    return P.resolved(model_with(shape))["geometries"][0]["source"]["shape"]


def test_geometry():
    """The exact shape: area (Monte Carlo), the sides and arcs on the
    boundary, their names and lengths; the problems."""
    sh = resolved(L_SHAPE)
    rng = np.random.default_rng(0)
    X = rng.random((1_000_000, 2)) * 2
    assert abs((G.sdf(sh, X) < 0).mean() * 4 - L_AREA) < 5e-3
    d = G.sdf(sh, np.array([[0.05, 0.05], [1.05, 1.05], [1.2, 1.2], [0.5, 0.5]]))
    assert d[0] > 0 and d[1] < 0 and d[2] > 0 and abs(d[3] + 0.5) < 1e-12   # the corners cut, filled
    pcs = G.pieces(sh)
    names = sorted({p.side.side for p in pcs})
    assert names == ["fillet1", "fillet4", "fillet5"] + [f"side{k}" for k in range(1, 7)]
    length = {p.side.side: p.side.length(p.t0, p.t1) for p in pcs}
    assert abs(length["fillet4"] - math.pi / 2 * 0.3) < 1e-12 and abs(length["side3"] - 0.7) < 1e-12
    assert G.side_label(sh, {"shape": [], "side": "fillet4"}) == "fillet 4 of polygon 1"
    assert G.side_names(sh)[-3:] == ("fillet1", "fillet4", "fillet5")
    assert len(G.fillet_ends(sh)) == 6
    outline = np.asarray(G.rounded_outline(sh))
    assert abs(abs(G.polygon_area(outline)) - L_AREA) < 2e-3
    big = dict(sh, fillets=[{"vertex": 4, "radius": 2.0}])
    assert G.fillet_problem(big) == "side 3: the fillet of vertex 4 too large for it"
    assert G.fillet_problem(dict(sh, points=[[0, 0], [1, 0], [2, 0], [1, 1]],
                                 fillets=[{"vertex": 2, "radius": 0.1}])).startswith("vertex 2: no corner")


def test_validation():
    m = model_with(L_SHAPE)
    assert M.validate(m) == []
    for change, msg in [([{"vertex": 9, "radius": 0.1}], "vertex number from 1 to 6"),
                        ([{"vertex": 2, "radius": -1}], "positive number"),
                        ([{"vertex": 2, "radius": 0.1}, {"vertex": 2, "radius": 0.2}], "already rounded"),
                        ([{"vertex": 4, "radius": 5}], "too large")]:
        bad = copy.deepcopy(m)
        bad["geometries"][0]["source"]["shape"]["fillets"] = change
        assert any(msg in e for e in M.validate(bad)), (change, M.validate(bad))


def test_meshed_getfem():
    """The GetFEM mesher: the area, the length of two arcs selected (the
    ends of the arcs are vertices of the mesh)."""
    m = model_with(L_SHAPE, selections={"round": ROUND},
                   outputs=[{"name": "round_length", "expr": "1", "selection": "round"}])
    code = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert "'half space'" in code and "interface_vertices" in code
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"{res.error}\n{res.stderr[-2000:]}"
        q = res.results["models"]["heat"]["quantities"]
    assert abs(q["area"] - L_AREA) < 1e-4
    assert abs(q["round_length"] - math.pi / 2 * 0.5) < 1e-4


def test_other_meshers_and_export():
    """The Gmsh and Netgen scripts (lines and arcs), the region test of the
    scripts, the SVG and DXF outlines."""
    for name in ("gmsh", "netgen"):
        m = model_with(L_SHAPE, mesh={"mesher": name, "size": 0.1})
        code = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
        compile(code, "mesh.py", "exec")
        assert ("addCircleArc" if name == "gmsh" else "ArcTo") in code
    sh = resolved(L_SHAPE)
    assert export.svg(sh).count(" A ") == 3
    dxf = export.dxf(sh)
    assert dxf.count("\nARC\n") == 3 and dxf.count("\nLINE\n") == 6


def test_editing():
    """Round all the corners, the vertex table, the handles, the vertices
    inserted and removed, removed fillets."""
    poly = ROOT + ("of", 0)
    m = model_with({"type": "union", "of": [{"type": "polygon",
                                            "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}]})
    acts = dict(E.actions(m, poly))
    assert "fillet:all" in acts and "fillet:remove" not in acts
    E.do_action(m, poly, "fillet:all")
    assert E.get_at(m, poly)["fillets"] == [{"vertex": k, "radius": 0.2} for k in (1, 2, 3, 4)]
    assert M.validate(m) == [] and E.fillet_cells(m, poly) == [0.2] * 4
    f = E.Field(poly + ("fillets", 1), "fillet of vertex 2", "value", op="vertex_fillet")
    E.apply(m, f, None)                                   # vertex 2 sharp
    E.apply(m, E.Field(poly + ("fillets", 2), "", "value", op="vertex_fillet"), "rf")
    assert E.fillet_cells(m, poly) == [0.2, None, "rf", 0.2]
    # the handle in the middle of the arc of vertex 1: its radius
    handles = dict(E.shape_handles(m, poly))
    assert "f1" in handles and "f2" not in handles
    mid = np.asarray(handles["f1"])
    assert abs(np.linalg.norm(mid - [0.2, 0.2]) - 0.2) < 1e-12
    E.move_handle(m, poly, "f1", [0.1 * (1 - 1 / math.sqrt(2)), 0.1 * (1 - 1 / math.sqrt(2))])
    assert abs(E.get_at(m, poly)["fillets"][0]["radius"] - 0.1) < 1e-6
    E.move_handle(m, poly, "f3", [1 - 0.05, 1 - 0.05])    # its radius is the parameter rf: changed
    assert m["parameters"][0]["value"] != 0.2
    # a vertex inserted after vertex 2, then vertex 1 removed: the fillets follow
    E._vertex_inserted(m, poly, 1)
    assert [f["vertex"] for f in E.get_at(m, poly)["fillets"]] == [1, 4, 5]
    E._vertex_removed(m, poly, 0, 5)
    assert [f["vertex"] for f in E.get_at(m, poly)["fillets"]] == [3, 4]
    E.do_action(m, poly, "fillet:remove")
    assert "fillets" not in E.get_at(m, poly)


def test_profile_in_3d():
    """A rounded profile extruded: its arcs are faces of the extrusion
    (preview, face selection), meshed by GetFEM (the volume)."""
    prof = {"type": "polygon", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "fillets": [{"vertex": 3, "radius": 0.4}]}
    ext = {"type": "extrusion", "z": [0, 0.5], "profile": prof}
    ref = {"shape": [], "face": "side", "profile": {"shape": [], "side": "fillet3"}}
    m = model_with(ext, 3, mesh={"mesher": "getfem", "h": 0.08, "order": 2},
                   selections={"round": {"kind": "boundary", "criteria": {"type": "faces", "faces": [ref]}}},
                   outputs=[{"name": "round_area", "expr": "1", "selection": "round"}])
    m["models"][0]["modules"][0]["conditions"][0]["value"] = "x + 2*y + 3*z"
    assert M.validate(m) == []
    shape = G3.shape_of(P.resolved(m)["geometries"][0]["source"])
    assert G3.face_exists(shape, ref) and G3.face_label(shape, ref) == "side (fillet 3 of polygon 1) of extrusion 1"
    spec = G3.face_spec(shape, ref)
    assert spec["kind"] == "extruded_side" and spec["curve"]["kind"] == "arc"
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"{res.error}\n{res.stderr[-2000:]}"
        q = res.results["models"]["heat"]["quantities"]
    area = 1 - (1 - math.pi / 4) * 0.16
    assert abs(q["area"] - 0.5 * area) < 5e-3
    assert abs(q["round_area"] - 0.5 * math.pi / 2 * 0.4) < 2e-3
    try:
        from getfem_gui.app import scene
    except ImportError:
        return
    surf = scene._surface_3d(shape, 40)
    assert ref in [dict(r, shape=[]) for r in scene.face_refs(surf)]


def test_reflex_fillet_in_3d():
    """A reflex fillet in the profile of an extrusion, meshed by GetFEM (its
    parts are not closer than h inside the shape: the mesher used to fail);
    the volume."""
    prof = {"type": "polygon", "points": [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]],
            "fillets": [{"vertex": 4, "radius": 0.3}]}
    m = model_with({"type": "extrusion", "z": [0, 0.3], "profile": prof}, 3,
                   mesh={"mesher": "getfem", "h": 0.1, "order": 1})
    m["models"][0]["modules"][0]["conditions"][0]["value"] = "x + 2*y + 3*z"
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"{res.error}\n{res.stderr[-2000:]}"
        q = res.results["models"]["heat"]["quantities"]
    assert abs(q["area"] - 0.3 * (3 + CUT * 0.09)) < 2e-3


def test_no_fillet_on_the_axis():
    """A vertex of a profile of revolution on the axis: no fillet (its arc
    would be tangent to the axis); "round all the corners" leaves it."""
    prof = {"type": "polygon", "points": [[0, 0.5], [1, 0.3], [0.35, 0.75], [0.8, 0.95], [0, 1.1]]}
    m = model_with({"type": "revolution", "profile": prof}, 3)
    path = ROOT + ("profile",)
    E.do_action(m, path, "fillet:all")
    assert [f["vertex"] for f in E.get_at(m, path)["fillets"]] == [2, 3, 4] and M.validate(m) == []
    E.get_at(m, path)["fillets"].append({"vertex": 5, "radius": 0.1})
    assert any("vertex 5 is on the axis of the revolution" in e for e in M.validate(m))
    m["geometries"][0]["source"]["shape"]["type"] = "extrusion"     # an extrusion: allowed
    m["geometries"][0]["source"]["shape"]["z"] = [0, 1]
    assert M.validate(m) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
