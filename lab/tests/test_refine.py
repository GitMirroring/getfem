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

"""Refinements of the meshes ({"refine": [{"selection", "times"}]}): format
and validation, editing, the meshing scripts (the whole mesh by its mesher,
the selections by GetFEM), the meshes made, the cache.

    python -m pytest tests/test_refine.py      (or: python tests/test_refine.py)
"""

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, mesher, model as M, parameters as P  # noqa: E402

PLATE = {"type": "difference", "of": [{"type": "rectangle", "min": [0, 0], "max": [2, 1]},
                                      {"type": "ball", "center": [1, 0.5], "radius": 0.25}]}
HOLE = {"kind": "boundary", "criteria": {"type": "edges", "edges": [{"shape": [1], "side": "circle"}]}}
RIGHT = {"kind": "domain", "criteria": {"type": "in_box", "min": [1.6, 0], "max": [2, 1]}}


def model_with(mesh, source=None):
    m = M.default_model()
    m["parameters"] = [{"name": "n", "value": 2}]
    m["geometries"][0]["source"] = source or {"type": "csg", "dim": 2, "shape": copy.deepcopy(PLATE)}
    m["geometries"][0]["selections"] = {"hole": copy.deepcopy(HOLE), "right": copy.deepcopy(RIGHT)} \
        if source is None else {}
    m["meshes"][0] = dict({"name": "mesh", "geometry": "geometry"}, **mesh)
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": "x + 2*y"}]
    md["outputs"] = [{"name": "area", "expr": "1", "selection": "all"}]
    return m


def test_validation():
    ok = model_with({"mesher": "getfem", "h": 0.2, "refine": [{"selection": "all", "times": 1},
                                                              {"selection": "hole", "times": "n"}]})
    assert M.validate(ok) == []
    assert P.resolved(ok)["meshes"][0]["refine"][1] == {"selection": "hole", "times": 2}
    for refine, extra, msg in [
            ([{"selection": "nothing", "times": 1}], {}, "unknown selection 'nothing'"),
            ([{"selection": "all", "times": 5}], {}, "an integer from 1 to 4"),
            ([{"selection": "hole", "times": 1}], {"mesher": "gmsh", "size": 0.2, "element": "quad"},
             "triangles and tetrahedra only"),
            ({"selection": "all"}, {}, "a list of")]:
        mesh = dict({"mesher": "getfem", "h": 0.2, "refine": refine}, **extra)
        if "size" in extra:
            mesh.pop("h")
        m = model_with(mesh)
        assert any(msg in e for e in M.validate(m)), (msg, M.validate(m))


def test_editing():
    """The first refinement: of the whole mesh; the next ones: of the user
    selections; their form and nodes; renamed with the selection; kept by
    another mesher; removed."""
    m = model_with({"mesher": "getfem", "h": 0.2})
    assert ("add_refinement", "Add a refinement") in E.actions(m, ("meshes", 0))
    path = E.do_action(m, ("meshes", 0), "add_refinement")
    assert path == ("meshes", 0, "refine", 0) and m["meshes"][0]["refine"] == [{"selection": "all", "times": 1}]
    E.do_action(m, ("meshes", 0), "add_refinement")
    assert m["meshes"][0]["refine"][1] == {"selection": "hole", "times": 1}
    keys = {f.key[-1]: f for f in E.fields(m, ("meshes", 0, "refine", 1))}
    assert [v for v, _ in keys["selection"].choices] == ["all", "boundary", "hole", "right"]
    assert keys["times"].kind == "int"
    labels = [c.label for c in E.find_node(E.tree(m), ("meshes", 0)).children]
    assert labels == ["Refined 1× everywhere", "Refined 1× on hole"]
    assert E.highlighted_selection(m, ("meshes", 0, "refine", 1)) == ("mesh", "hole")
    E._rename_selection(m, 0, "hole", "circle")
    assert m["meshes"][0]["refine"][1]["selection"] == "circle"
    E._change_mesher(m, m["meshes"][0], "gmsh")
    assert len(m["meshes"][0]["refine"]) == 2 and M.validate(m) == []
    E.do_action(m, ("meshes", 0, "refine", 0), "remove")
    assert E.do_action(m, ("meshes", 0, "refine", 0), "remove") == ("meshes", 0)
    assert "refine" not in m["meshes"][0]


def script(m):
    r = P.resolved(m)
    return mesher.mesh_script(r, r["meshes"][0], "mesh")


def test_scripts():
    """The whole mesh refined by its mesher (Gmsh, Netgen: following the
    geometry; structured: more divisions), by GetFEM for the GetFEM mesher
    (the new boundary nodes projected on the geometry); the selections by
    GetFEM, located again each time."""
    refine = [{"selection": "all", "times": 2}, {"selection": "hole", "times": 1}]
    for mesher_, key, native in (("gmsh", "size", "gmsh.model.mesh.refine()"),
                                 ("netgen", "size", "ngmesh.Refine()")):
        text = script(model_with({"mesher": mesher_, key: 0.2, "refine": refine}))
        compile(text, f"{mesher_}.py", "exec")
        assert f"for _ in range(2):   # refined" in text and native in text
        assert "REFINE = [('hole', 1)]" in text and "m.refine(cv)" in text and ".project(" not in text
        assert "STORED = {" in text
    m = model_with({"mesher": "getfem", "h": 0.2, "refine": refine})
    text = script(m)
    compile(text, "getfem.py", "exec")
    r = P.resolved(m)                      # the mesh before the selections reused: their regions kept
    again = mesher.mesh_script(r, r["meshes"][0], "mesh", reuse_raw=True)
    assert "m = gf.Mesh('load', 'mesh_raw.mesh')" in again and "m.refine" not in again
    assert "STORED = {" in again
    assert "for _ in range(2):   # the whole mesh" in text and ".project(_P[:, _B])" in text
    box = {"type": "box", "origin": [0, 0], "size": [2, 1]}
    text = script(model_with({"mesher": "structured", "divisions": [4, 2], "element": "quad",
                              "refine": [{"selection": "all", "times": 2}]}, box))
    assert "np.linspace(0.0, 2.0, 17), np.linspace(0.0, 1.0, 9)" in text and "m.refine" not in text
    text = script(model_with({"mesher": "getfem", "h": 0.2}))
    assert "Refinements" not in text and ".project(" not in text


def _mesh(m):
    import getfem as gf
    with tempfile.TemporaryDirectory() as tmp:
        res = mesher.ensure_mesh(m, "mesh", tmp)
        assert res.ok, res.error
        return gf.Mesh("load", str(res.path)), res.info


def test_getfem_meshes():
    """Refined everywhere: four times the triangles, the boundary nodes on
    the circle; refined on the hole: its faces split, the rest less; on a
    part of the domain: its elements split."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    m0, info0 = _mesh(model_with({"mesher": "getfem", "h": 0.2}))
    m1, info1 = _mesh(model_with({"mesher": "getfem", "h": 0.2, "refine": [{"selection": "all", "times": 1}]}))
    assert info1["nb_elements"] == 4 * info0["nb_elements"]
    assert info1["sizes"]["hole"] == 2 * info0["sizes"]["hole"]
    B = m1.pid_in_faces(m1.outer_faces())
    X = m1.pts()[:, B]
    r = np.hypot(X[0] - 1, X[1] - 0.5)
    assert np.abs(r[r < 0.3] - 0.25).max() < 1e-10          # on the circle (projected)
    _, info2 = _mesh(model_with({"mesher": "getfem", "h": 0.2, "refine": [{"selection": "hole", "times": 2}]}))
    assert info2["sizes"]["hole"] == 4 * info0["sizes"]["hole"]
    assert info0["nb_elements"] < info2["nb_elements"] < 16 * info0["nb_elements"]
    assert info2["sizes"]["right"] == info0["sizes"]["right"] or info2["sizes"]["right"] < 2 * info0["sizes"]["right"]
    _, info3 = _mesh(model_with({"mesher": "getfem", "h": 0.2, "refine": [{"selection": "right", "times": 1}]}))
    assert info3["sizes"]["right"] >= 4 * info0["sizes"]["right"]


def test_structured():
    box = {"type": "box", "origin": [0, 0], "size": [2, 1]}
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    m, info = _mesh(model_with({"mesher": "structured", "divisions": [4, 2], "element": "quad",
                                "refine": [{"selection": "all", "times": 1}]}, box))
    assert info["nb_elements"] == 32


def test_cache():
    """The mesh before the selections depends on them with a local
    refinement, not with a refinement of the whole mesh only."""
    m = model_with({"mesher": "getfem", "h": 0.2, "refine": [{"selection": "hole", "times": 1}]})
    key = mesher.raw_key(m, m["meshes"][0])
    m["geometries"][0]["selections"]["right"]["criteria"]["min"] = [1.5, 0]
    assert mesher.raw_key(m, m["meshes"][0]) != key
    m["meshes"][0]["refine"] = [{"selection": "all", "times": 1}]
    key = mesher.raw_key(m, m["meshes"][0])
    m["geometries"][0]["selections"]["right"]["criteria"]["min"] = [1.4, 0]
    assert mesher.raw_key(m, m["meshes"][0]) == key


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
