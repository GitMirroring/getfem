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

"""3D shapes made of a 2D profile: extrusion and revolution (format,
editing, preview, mesh and solve).

    python -m pytest tests/       (or: python tests/test_geometry3d.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, geometry3d as G3, mesher, model as M, parameters as P, runner  # noqa: E402

PROFILE = {"type": "difference", "of": [{"type": "rectangle", "min": [0, 0], "max": [2, 1]},
                                         {"type": "ball", "center": [1, 0.5], "radius": "R"}]}
EXTRUSION = {"type": "extrusion", "z": [0, "H"], "profile": PROFILE}
TUBE = {"type": "revolution", "profile": {"type": "rectangle", "min": [1, 0], "max": [1.5, 1]}}


def model_with(shape, h=0.25):
    m = M.default_model()
    m["parameters"] = [{"name": "R", "value": 0.3}, {"name": "H", "value": 0.5}]
    m["geometries"][0]["source"] = {"type": "csg", "dim": 3, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = {
        "bottom": {"kind": "boundary", "criteria": {"type": "plane", "point": [0, 0, 0], "normal": [0, 0, 1]}}}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "getfem", "h": h, "order": 2}
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "bottom", "value": 0.0}]
    md["outputs"] = [{"name": "volume", "expr": "1", "selection": "all"},
                     {"name": "area", "expr": "1", "selection": "boundary"}]
    return m


def test_format():
    m = model_with(EXTRUSION)
    assert M.validate(m) == []
    shape = P.resolved(m)["geometries"][0]["source"]["shape"]
    assert shape["z"] == [0, 0.5] and shape["profile"]["of"][1]["radius"] == 0.3
    bad = copy.deepcopy(m)
    bad["geometries"][0]["source"]["shape"]["z"] = [1, 0]
    assert any("z0 < z1" in e for e in M.validate(bad))
    bad["geometries"][0]["source"]["shape"] = {"type": "extrusion", "z": [0, 1], "profile": {
        "type": "ball", "center": [0, 0, 0], "radius": 1}}
    assert any("profile.center" in e for e in M.validate(bad))        # a 2D profile
    flat = M.default_model()
    flat["geometries"][0]["source"] = {"type": "csg", "dim": 2, "shape": copy.deepcopy(TUBE)}
    assert any("revolution only exists in 3D" in e for e in M.validate(flat))


def test_distance_and_bounds():
    ext = P.resolved(model_with(EXTRUSION))["geometries"][0]["source"]["shape"]
    d = G3.sdf(ext, np.array([[0.1, 0.1, 0.1], [1, 0.5, 0.25], [0.1, 0.1, 0.7], [3, 0.5, 0.25]]))
    assert d[0] < 0 and d[1] > 0 and abs(d[2] - 0.2) < 1e-12 and abs(d[3] - 1) < 1e-12
    lo, hi = G3.bounds(ext)
    assert np.allclose(lo, [0, 0, 0]) and np.allclose(hi, [2, 1, 0.5])
    d = G3.sdf(TUBE, np.array([[1.25, 0, 0.5], [0, 1.25, 0.5], [0, 0, 0.5], [0, 2, 0.5]]))
    assert d[0] < 0 and d[1] < 0 and abs(d[2] - 1) < 1e-12 and abs(d[3] - 0.5) < 1e-12
    lo, hi = G3.bounds(TUBE)
    assert np.allclose(lo, [-1.5, -1.5, 0]) and np.allclose(hi, [1.5, 1.5, 1])
    # the part of a profile with r < 0 is ignored
    cyl = {"type": "revolution", "profile": {"type": "rectangle", "min": [-1, 0], "max": [1, 1]}}
    curves = G3.profile_curves(cyl, 0.1)
    assert all(np.all(c[:, 0] > -1e-12) for c in curves)
    length = sum(np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in curves)
    assert abs(length - 3.0) < 1e-9                     # top, outer side, bottom (not the axis)


def test_editing():
    m = model_with(EXTRUSION)
    root = ("geometries", 0, "source", "shape")
    tree = E.tree(m)
    node = E.find_node(tree, root)
    assert node.label == "Extrusion" and node.children[0].label == "Profile: Difference"
    assert [c.path for c in node.children[0].children] == [root + ("profile", "of", 0), root + ("profile", "of", 1)]
    assert E.shape_dim(m, root) == 3 and E.shape_dim(m, root + ("profile", "of", 1)) == 2
    keys = [f.key for f in E.fields(m, root)]
    assert root + ("z",) in keys
    types = [v for v, _ in next(f for f in E.fields(m, root + ("profile",)) if f.op == "shape_type").choices]
    assert "cylinder" not in types and "extrusion" not in types and "rectangle" in types
    actions = [a for a, _ in E.actions(m, root + ("profile",))]
    assert "add_shape:ball" in actions and "add_shape:cylinder" not in actions
    new = E.do_action(m, root + ("profile",), "add_shape:rectangle")
    assert new == root + ("profile", "of", 2) and len(E.get_at(m, new)["min"]) == 2
    assert M.validate(m) == []
    # the profile part drawn over the preview: extruded likewise
    part = E.part_3d(P.resolved(m), root + ("profile", "of", 1))
    assert part["type"] == "extrusion" and part["profile"]["type"] == "ball" and part["z"] == [0, 0.5]
    # a 3D primitive changed into a revolution, and back into a boolean
    m = model_with({"type": "union", "of": [{"type": "rectangle", "min": [0, 0, 0], "max": [1, 1, 1]},
                                            {"type": "ball", "center": [0, 0, 0], "radius": 0.5}]})
    f = next(f for f in E.fields(m, root + ("of", 1)) if f.op == "shape_type")
    E.apply(m, f, "revolution")
    assert E.get_at(m, root + ("of", 1))["type"] == "revolution" and M.validate(m) == []
    f = next(f for f in E.fields(m, root + ("of", 1, "profile")) if f.op == "shape_type")
    E.apply(m, f, "union")
    prof = E.get_at(m, root + ("of", 1, "profile"))
    assert prof["type"] == "union" and len(prof["of"][1]["center"]) == 2 and M.validate(m) == []


def test_mesh_script():
    m = model_with(EXTRUSION)
    script = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert "gf.MesherObject('extrusion', " in script and ", 0.0, 0.5)" in script
    m = model_with(TUBE)
    assert "gf.MesherObject('revolution', " in mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    gm = copy.deepcopy(m)
    gm["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "gmsh", "size": 0.2}
    code = mesher.mesh_script(P.resolved(gm), gm["meshes"][0], "mesh")
    assert "occ.revolve(" in code and "occ.rotate(" in code
    compile(code, "mesh.py", "exec")


def _measures(shape, h):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(model_with(shape, h), tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
        return res.results["models"]["heat"]["quantities"]


def test_meshed_extrusion_and_revolution():
    """Needs a GetFEM with the 'extrusion' and 'revolution' mesher objects."""
    q = _measures(EXTRUSION, 0.2)
    A, L = 2 - math.pi * 0.09, 6 + 2 * math.pi * 0.3
    assert abs(q["volume"] - 0.5 * A) < 2e-3 and abs(q["area"] - (2 * A + 0.5 * L)) < 2e-2
    q = _measures(TUBE, 0.2)
    vol = math.pi * (1.5 ** 2 - 1)
    assert abs(q["volume"] - vol) < 1e-2 and abs(q["area"] - (2 * math.pi * 2.5 + 2 * vol)) < 5e-2


def test_preview():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return
    from getfem_gui.app import scene
    ext = P.resolved(model_with(EXTRUSION))["geometries"][0]["source"]["shape"]
    shape = {"type": "difference", "of": [ext, {"type": "cylinder", "origin": [0.3, 0.5, -1],
                                                "axis": [0, 0, 1], "length": 3, "radius": 0.15}]}
    A, L = 2 - math.pi * 0.09, 6 + 2 * math.pi * 0.3
    for s, area, vol in [
            (ext, 2 * A + 0.5 * L, 0.5 * A),
            (shape, 2 * (A - math.pi * 0.15 ** 2) + 0.5 * (L + 2 * math.pi * 0.15), 0.5 * (A - math.pi * 0.15 ** 2)),
            (TUBE, 2 * math.pi * 2.5 + 2 * math.pi * 1.25, math.pi * 1.25)]:
        surf = scene.geometry_surface_3d(s)
        N, C = np.asarray(surf.cell_data["outward"]), np.asarray(surf.cell_centers().points)
        a = np.asarray(surf.compute_cell_sizes().cell_data["Area"])
        assert abs(a.sum() / area - 1) < 2e-3
        assert abs(np.sum(np.einsum("ij,ij->i", C, N) * a) / 3 / vol - 1) < 2e-3   # outward normals
        assert surf.field_data["edges_kind"][0] == "exact"
    # the bottom of the tube: a plane selection on the exact surface
    surf = scene.geometry_surface_3d(TUBE)
    C, N = np.asarray(surf.cell_centers().points), np.asarray(surf.cell_data["outward"])
    table = M.selection_table(model_with(TUBE), model_with(TUBE)["geometries"][0])
    found = G3.faces_matching(TUBE, C, N, "bottom", table)
    a = np.asarray(surf.compute_cell_sizes().cell_data["Area"])
    assert abs(a[found].sum() / (math.pi * 1.25) - 1) < 2e-3


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
