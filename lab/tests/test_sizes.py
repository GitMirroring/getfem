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

"""Local sizes of the meshes ({"sizes": [{"selection", "size"}], "growth"}):
format and validation, editing, the meshing scripts of the three meshers,
the meshes made (finer on the selections), the cache.

    python -m pytest tests/test_sizes.py      (or: python tests/test_sizes.py)
"""

import copy
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, mesher, model as M, parameters as P, runner  # noqa: E402

HAS_GMSH = importlib.util.find_spec("gmsh") is not None
PLATE = {"type": "difference", "of": [{"type": "rectangle", "min": [0, 0], "max": [2, 1]},
                                      {"type": "ball", "center": [1, 0.5], "radius": 0.25}]}
HOLE = {"kind": "boundary", "criteria": {"type": "edges", "edges": [{"shape": [1], "side": "circle"}]}}
INNER = {"kind": "domain", "criteria": {"type": "in_box", "min": [1.4, 0], "max": [2, 1]}}


def model_with(mesh, shape=PLATE, dim=2, selections=None):
    m = M.default_model()
    m["parameters"] = [{"name": "hh", "value": 0.1}]
    m["geometries"][0]["source"] = {"type": "csg", "dim": dim, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = copy.deepcopy(selections if selections is not None
                                                     else {"hole": HOLE, "right": INNER})
    m["meshes"][0] = dict({"name": "mesh", "geometry": "geometry"}, **mesh)
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": "x + 2*y"}]
    md["outputs"] = [{"name": "area", "expr": "1", "selection": "all"}]
    return m


def test_validation():
    ok = model_with({"mesher": "getfem", "h": "hh", "sizes": [{"selection": "hole", "size": "hh/5"}]})
    assert M.validate(ok) == []
    assert P.resolved(ok)["meshes"][0]["sizes"] == [{"selection": "hole", "size": 0.02}]
    for sizes, extra, msg in [
            ([{"selection": "nothing", "size": 0.01}], {}, "unknown selection 'nothing'"),
            ([{"selection": "all", "size": 0.01}], {}, "a part of the domain expected"),
            ([{"selection": "hole", "size": 0.2}], {}, "smaller than the size of the mesh (0.1)"),
            ([{"selection": "hole", "size": -1}], {}, "strictly positive"),
            ([{"selection": "hole", "size": 0.01}], {"growth": 1.0}, "a rate above 1"),
            ("hole", {}, "a list of")]:
        m = model_with(dict({"mesher": "gmsh", "size": 0.1, "sizes": sizes}, **extra))
        assert any(msg in e for e in M.validate(m)), (msg, M.validate(m))
    swept = {"type": "extrusion", "z": [0, 0.4], "profile": PLATE}
    m = model_with({"mesher": "gmsh", "size": 0.1, "layers": 3, "sizes": [{"selection": "boundary", "size": 0.01}]},
                   swept, 3, {})
    assert any("not in layers" in e for e in M.validate(m))


def test_editing():
    """Adding a local size (on the first user selection of edges, a quarter
    of the size), its form, the tree, the growth of the mesh; renamed with
    its selection; kept by another free mesher, dropped by a structured
    one; removed."""
    m = model_with({"mesher": "getfem", "h": "hh"})
    assert ("add_size", "Add a local size") in E.actions(m, ("meshes", 0))
    path = E.do_action(m, ("meshes", 0), "add_size")
    assert path == ("meshes", 0, "sizes", 0)
    assert m["meshes"][0]["sizes"] == [{"selection": "hole", "size": 0.025}]
    keys = {f.key[-1]: f for f in E.fields(m, path)}
    assert [v for v, _ in keys["selection"].choices] == ["boundary", "hole", "right"]
    assert "growth" in {f.key[-1] for f in E.fields(m, ("meshes", 0))}
    tree = E.tree(m)
    node = E.find_node(tree, ("meshes", 0, "sizes", 0))
    assert node is not None and node.label == "Size 0.025 on hole"
    assert E.mesh_to_show(m, path) == "mesh" and E.highlighted_selection(m, path) == ("mesh", "hole")
    E.do_action(m, ("meshes", 0), "add_size")                 # the next one: another selection
    assert m["meshes"][0]["sizes"][1]["selection"] == "right"
    E._rename_selection(m, 0, "hole", "circle")
    assert m["meshes"][0]["sizes"][0]["selection"] == "circle"
    E._change_mesher(m, m["meshes"][0], "gmsh")
    assert len(m["meshes"][0]["sizes"]) == 2 and M.validate(m) == []
    assert E.do_action(m, ("meshes", 0, "sizes", 1), "remove") == ("meshes", 0, "sizes", 0)
    assert E.do_action(m, ("meshes", 0, "sizes", 0), "remove") == ("meshes", 0)
    assert "sizes" not in m["meshes"][0] and "growth" not in m["meshes"][0]
    E.do_action(m, ("meshes", 0), "add_size")
    box = model_with({"mesher": "getfem", "h": 0.1, "sizes": [{"selection": "boundary", "size": 0.05}]})
    box["geometries"][0]["source"] = {"type": "box", "origin": [0, 0], "size": [1, 1]}
    E._change_mesher(box, box["meshes"][0], "structured")
    assert "sizes" not in box["meshes"][0]
    assert ("add_size", "Add a local size") not in E.actions(box, ("meshes", 0))


def test_scripts():
    """The three meshers: a first mesh of the global size, the selections of
    the local sizes sampled on it, the final mesh with them (the GetFEM
    mesher: its sizes; Gmsh: a size callback; Netgen: sizes restricted at
    points). The scripts compile; without local sizes, one mesh."""
    for mesher_, key, call in (("getfem", "h", "SIZE_POINTS, SIZE_VALUES, GROWTH)"),
                               ("gmsh", "size", "setSizeCallback("), ("netgen", "size", "mp.RestrictH(")):
        m = model_with({"mesher": mesher_, key: 0.1, "growth": 1.2,
                        "sizes": [{"selection": "hole", "size": 0.02}, {"selection": "right", "size": 0.05}]})
        r = P.resolved(m)
        text = mesher.mesh_script(r, r["meshes"][0], "mesh")
        compile(text, f"{mesher_}.py", "exec")
        assert "size_points('hole', 0.02)" in text and "size_points('right', 0.05)" in text
        assert "H_GLOBAL, GROWTH = 0.1, 1.2" in text and call in text, mesher_
        assert text.index("SEL = {}") < text.index("SIZE_SETS") < text.rindex("SEL = {}")
        m["meshes"][0].pop("sizes")
        r = P.resolved(m)
        text = mesher.mesh_script(r, r["meshes"][0], "mesh")
        assert "SIZE_SETS" not in text and "local_size" not in text


def edge_lengths(mesh_file):
    import getfem as gf
    msh = gf.Mesh("load", str(mesh_file))
    X = msh.pts()
    E_ = set()
    for cv in msh.cvid():
        p = msh.pid_from_cvid(cv)[0][:3]
        E_ |= {tuple(sorted((int(p[i]), int(p[j])))) for i in range(3) for j in range(i + 1, 3)}
    E_ = np.array(sorted(E_)).T
    return 0.5 * (X[:2, E_[0]] + X[:2, E_[1]]), np.linalg.norm(X[:, E_[0]] - X[:, E_[1]], axis=0)


def test_getfem_mesh():
    """The GetFEM mesher with a local size on the hole and another one on a
    part of the domain: the elements have these sizes there, the global one
    far from them; the selections hold on the final mesh; solved."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    m = model_with({"mesher": "getfem", "h": 0.1,
                    "sizes": [{"selection": "hole", "size": 0.02}, {"selection": "right", "size": 0.04}]})
    with tempfile.TemporaryDirectory() as tmp:
        res = mesher.ensure_mesh(m, "mesh", tmp)
        assert res.ok, res.error
        C, L = edge_lengths(res.path)
        d_hole = np.abs(np.hypot(C[0] - 1, C[1] - 0.5) - 0.25)
        assert 0.012 < L[d_hole < 0.005].mean() < 0.03
        right = C[0] > 1.45
        assert 0.025 < L[right].mean() < 0.055
        far = (d_hole > 0.5) & (C[0] < 1.0)
        assert far.any() and 0.075 < L[far].mean() < 0.125
        assert res.info["sizes"]["hole"] > 50 and res.info["nb_elements"] > 1000
        result = runner.run(m, tmp)
        assert result.ok, result.error
        area = result.results["models"]["heat"]["quantities"]["area"]
        assert abs(area - (2 - np.pi * 0.25 ** 2)) < 1e-3


def test_cache():
    """The mesh before the selections is meshed again when a selection of
    a local size changes (it is located by it), not for another one."""
    m = model_with({"mesher": "getfem", "h": 0.1, "sizes": [{"selection": "hole", "size": 0.02}]})
    m["geometries"][0]["selections"]["other"] = copy.deepcopy(INNER)
    key = mesher.raw_key(m, m["meshes"][0])
    m["geometries"][0]["selections"]["other"]["criteria"]["min"] = [1.5, 0]
    assert mesher.raw_key(m, m["meshes"][0]) != key       # (any selection: they may be in a union)
    m["meshes"][0].pop("sizes")
    key = mesher.raw_key(m, m["meshes"][0])
    m["geometries"][0]["selections"]["other"]["criteria"]["min"] = [1.6, 0]
    assert mesher.raw_key(m, m["meshes"][0]) == key


def test_gmsh_mesh():
    if not HAS_GMSH:
        return
    m = model_with({"mesher": "gmsh", "size": 0.1, "sizes": [{"selection": "hole", "size": 0.02}]})
    with tempfile.TemporaryDirectory() as tmp:
        res = mesher.ensure_mesh(m, "mesh", tmp)
        assert res.ok, res.error
        C, L = edge_lengths(res.path)
        d_hole = np.abs(np.hypot(C[0] - 1, C[1] - 0.5) - 0.25)
        assert L[d_hole < 0.005].mean() < 0.03 and L[d_hole > 0.5].mean() > 0.07


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
