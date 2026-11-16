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

"""The boundary of a 3D geometry as a surface ({"type": "boundary",
"geometry", "selection"}): validation, the selections of the solid as parts
of the surface, editing (the form, the renames, the mesher), the meshing
scripts, the meshes made (the whole boundary, some faces, curved), the
cache, the preview.

    python -m pytest tests/test_boundary.py      (or: python tests/test_boundary.py)
"""

import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, mesher, model as M, parameters as P  # noqa: E402

R = 0.3
SOLID = {"name": "solid", "source": {"type": "csg", "dim": 3, "shape": {"type": "difference", "of": [
    {"type": "rectangle", "min": [0, 0, 0], "max": [2, 1, 1]}, {"type": "ball", "center": [1, 0.5, 0.5], "radius": R}]}},
    "selections": {"hole": {"kind": "boundary", "criteria": {"type": "in_ball", "center": [1, 0.5, 0.5],
                                                              "radius": 0.35}},
                   "top": {"kind": "boundary", "criteria": {"type": "plane", "point": [0, 0, 1],
                                                             "normal": [0, 0, 1]}},
                   "core": {"kind": "domain", "criteria": {"type": "in_box", "min": [0, 0, 0], "max": [1, 1, 1]}}}}


def model_with(mesh, selection=None):
    m = M.default_model()
    skin = {"name": "skin", "source": {"type": "boundary", "geometry": "solid"}, "selections": {}}
    if selection:
        skin["source"]["selection"] = selection
    m["geometries"] = [copy.deepcopy(SOLID), skin]
    m["meshes"] = [dict({"name": "mesh", "geometry": "skin"}, **mesh)]
    md = m["models"][0]
    md["variables"][0]["mesh"] = "mesh"
    md["modules"][0]["conditions"] = []
    md["outputs"] = [{"name": "area", "expr": "1", "selection": "all"},
                     {"name": "hole_area", "expr": "1", "selection": "hole"}]
    return m


def errors(m):
    return [e for e in M.validate(m) if not e.startswith("models")]


def test_validation():
    """The solid: another geometry, in 3D, not a surface; the faces: one of
    its boundary selections; the mesh: checked as a mesh of the solid,
    without local sizes or refinements."""
    m = model_with({"mesher": "getfem", "h": 0.2}, "top")
    assert errors(m) == []
    assert M.geometry_dim(m, m["geometries"][1]) == 3 and M.topological_dim(m, m["geometries"][1]) == 2
    table = M.selection_table(m, m["geometries"][1])
    assert list(table) == ["all", "boundary", "hole", "top"]                     # its boundary selections
    assert table["hole"]["kind"] == "domain" and table["hole"]["builtin"] and table["hole"]["inherited"]
    for src, msg in [({"geometry": "skin"}, "another geometry of the model expected"),
                     ({"geometry": "nothing"}, "another geometry of the model expected"),
                     ({"geometry": "solid", "selection": "core"}, "a boundary selection of 'solid' expected"),
                     ({"geometry": "solid", "selection": "nothing"}, "a boundary selection of 'solid' expected")]:
        m["geometries"][1]["source"] = dict({"type": "boundary"}, **src)
        assert any(msg in e for e in M.validate(m)), (msg, M.validate(m))
    m = model_with({"mesher": "getfem", "h": 0.2})
    m["geometries"].append({"name": "flat", "source": E._default_source("surface", 3), "selections": {}})
    m["geometries"][1]["source"]["geometry"] = "flat"
    assert any("'flat' is not a 3D solid" in e for e in M.validate(m))
    for extra, msg in [({"refine": [{"selection": "all", "times": 1}]}, "no local sizes or refinements"),
                       ({"order": 5}, "order: 1, 2 or 3 expected"),
                       ({"mesher": "structured", "divisions": [2, 2, 2]}, "'structured' cannot mesh a 'csg'")]:
        m = model_with(dict({"mesher": "getfem", "h": 0.2}, **extra))
        assert any(msg in e for e in M.validate(m)), (msg, M.validate(m))


def test_editing():
    """Added from the menu: the boundary of the first solid; its form (the
    solid, its faces); its built-in selections in the tree; its mesh by a
    mesher of the solid; the renames of the solid and of its selections
    followed; another solid: the faces kept if it has them."""
    m = M.default_model()
    assert ("Curves and surfaces", "boundary", "Boundary of a 3D geometry",
            "its faces (or some of them) as a surface: shells, membranes") in E.GEOMETRY_KINDS
    m["geometries"].append(copy.deepcopy(SOLID))
    path = E.do_action(m, ("geometries",), "add_geometry:boundary")
    geom = E.get_at(m, path)
    assert geom["source"] == {"type": "boundary", "geometry": "solid"} and errors(m) == []
    keys = {f.key[-1]: f for f in E.fields(m, path)}
    assert keys["geometry"].choices == (("solid", "solid"),)                     # the 3D solids, itself excepted
    assert keys["selection"].choices == ((None, "all (its whole boundary)"), ("hole", "hole"), ("top", "top"))
    E.apply(m, keys["selection"], "top")
    assert geom["source"]["selection"] == "top"
    node = E.find_node(E.tree(m), path)
    assert node.label == "geometry2 (boundary of solid: top)"
    assert [c.label for c in node.children[0].children] == ["all (built-in)", "boundary (built-in)",
                                                             "hole (from solid)", "top (from solid)"]
    assert "faces of the boundary selection 'hole'" in E.fields(m, path + ("selections", "hole"))[0].help
    mesh = E.do_action(m, path, "add_mesh")
    assert E.get_at(m, mesh)["mesher"] in M.FREE_MESHERS and errors(m) == []
    assert "Layers" not in [f.label for f in E.fields(m, mesh)]
    assert [v for v, _ in next(f for f in E.fields(m, mesh) if f.key[-1] == "mesher").choices] \
        == [x for x, srcs in M.MESHERS.items() if "csg" in srcs]
    E.apply(m, next(f for f in E.fields(m, ("geometries", 1)) if f.key[-1] == "name"), "block")
    assert geom["source"]["geometry"] == "block"
    E._rename_selection(m, 1, "top", "lid")
    assert geom["source"]["selection"] == "lid" and "lid" in M.selection_table(m, geom)
    E.apply(m, keys["selection"], None)
    assert "selection" not in geom["source"]
    m["geometries"].append({"name": "cube", "source": E._default_source("box", 3), "selections": {}})
    geom["source"]["selection"] = "lid"
    E.apply(m, keys["geometry"], "cube")                     # no 'lid' on the cube: the whole boundary
    assert geom["source"] == {"type": "boundary", "geometry": "cube"}
    E._change_mesher(m, E.get_at(m, mesh), "gmsh")
    m["geometries"][1]["source"] = E._default_source("box", 3)            # the solid changed: its mesher
    E._fit_meshes(m, 1)
    m["geometries"][-1]["source"] = E._default_source("csg", 3)
    E._fit_meshes(m, len(m["geometries"]) - 1)
    assert errors(m) == []
    f = next(f for f in E.fields(m, path) if f.key[-1] == "type")         # the source type changed to it
    E.apply(m, f, "box")
    E.apply(m, next(f for f in E.fields(m, path) if f.key[-1] == "type"), "boundary")
    assert E.get_at(m, path)["source"]["geometry"] in ("block", "cube")


def script(m):
    r = P.resolved(m)
    return mesher.mesh_script(r, r["meshes"][0], "mesh")


def test_scripts():
    """The solid meshed by each mesher, its selections located, its faces
    taken; the scripts compile."""
    for mesh in ({"mesher": "getfem", "h": 0.2}, {"mesher": "gmsh", "size": 0.2, "order": 2},
                 {"mesher": "netgen", "size": 0.2}):
        text = script(model_with(mesh, "top"))
        compile(text, "boundary.py", "exec")
        assert "# The volume 'solid'" in text and "FACES = SEL['top'][1]" in text
        assert "surface = gf.Mesh('faces', m, FACES)" in text and "for k, name in enumerate(['hole', 'top'])" in text
        assert text.index("surface.set_region(1000 + k, cv)") < text.index("TDIM = 2")
    text = script(model_with({"mesher": "getfem", "h": 0.2}))
    assert "FACES = m.outer_faces(3)" in text


def test_cache():
    """The mesh before the selections depends on the solid and its
    selections (regions of the surface), not on those of the surface."""
    m = model_with({"mesher": "getfem", "h": 0.2})
    key = mesher.raw_key(m, m["meshes"][0])
    m["geometries"][1]["selections"]["part"] = {"kind": "domain", "criteria": {"type": "in_box", "min": [0, 0, 0],
                                                                                "max": [1, 1, 1]}}
    assert mesher.raw_key(m, m["meshes"][0]) == key
    m["geometries"][0]["selections"]["top"]["criteria"]["point"] = [0, 0, 0]
    assert mesher.raw_key(m, m["meshes"][0]) != key
    key = mesher.cache_key(m, m["meshes"][0])
    m["geometries"][0]["source"]["shape"]["of"][1]["radius"] = 0.2
    assert mesher.cache_key(m, m["meshes"][0]) != key


def _mesh(m):
    import getfem as gf
    with tempfile.TemporaryDirectory() as tmp:
        res = mesher.ensure_mesh(m, "mesh", tmp)
        assert res.ok, res.error
        return gf.Mesh("load", str(res.path)), res.info


def _area(msh, region=-1):
    import getfem as gf
    mim = gf.MeshIm(msh, gf.Integ("IM_TRIANGLE(6)"))
    return gf.asm_generic(mim, 0, "1", region)


def test_meshes():
    """The whole boundary: a closed surface of triangles, the hole a part of
    it, the elements those of the faces of the volume mesh; curved (order
    2): closer to the sphere; some faces: an open surface, its edge."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    sphere = 4 * np.pi * R ** 2
    msh, info = _mesh(model_with({"mesher": "getfem", "h": 0.2}))
    assert info["dim"] == 3 and info["sizes"]["boundary"] == 0
    assert msh.geotrans()[0].char() == "GT_PK(2,1)"
    hole1 = _area(msh, info["regions"]["hole"])
    assert abs(_area(msh) - (10 + hole1)) < 1e-10 and 0.85 * sphere < hole1 < sphere
    msh, info = _mesh(model_with({"mesher": "getfem", "h": 0.2, "order": 2}))
    hole2 = _area(msh, info["regions"]["hole"])
    assert msh.geotrans()[0].char() == "GT_PK(2,2)" and abs(hole2 - sphere) < abs(hole1 - sphere)
    msh, info = _mesh(model_with({"mesher": "getfem", "h": 0.2}, "top"))
    assert abs(_area(msh) - 2) < 1e-10 and info["sizes"]["hole"] == 0 and info["sizes"]["top"] == msh.nbcvs()
    assert info["sizes"]["boundary"] > 0                                       # the edge of the top face
    m = model_with({"mesher": "structured", "divisions": [4, 2, 2], "element": "quad"}, "xmax")
    m["geometries"][0]["source"] = {"type": "box", "origin": [0, 0, 0], "size": [2, 1, 1]}
    m["geometries"][0]["selections"] = {}
    msh, info = _mesh(m)
    assert msh.nbcvs() == 4 and msh.geotrans()[0].char() == "GT_QK(2,1)"
    assert np.allclose(msh.pts()[0], 2.0) and set(info["sizes"]) >= {"xmin", "xmax", "zmax"}


def test_solve():
    """A heat problem on some faces (an open surface): heated, its edge at
    a fixed temperature; the heat balance, the area of the faces."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    from getfem_gui import runner
    m = model_with({"mesher": "getfem", "h": 0.2, "order": 2}, "top")
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": 0}]
    md["modules"][0]["sources"] = [{"selection": "top", "value": 3.0}]
    md["outputs"] = [{"name": "top_area", "expr": "1", "selection": "top"},
                     {"name": "out", "expr": "-k*Grad_T.Normal", "selection": "boundary"}]
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, res.error
    q = res.results["models"][md["name"]]["quantities"]
    assert abs(q["top_area"] - 2) < 1e-8 and abs(q["out"] / (3.0 * 2) - 1) < 0.05     # (a coarse mesh)


def test_preview():
    """The node of the geometry: the solid, the faces taken highlighted."""
    try:
        from PySide6.QtWidgets import QApplication
        import pyvista  # noqa: F401
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"), GETFEM_GUI_NO_3D="",
                      GETFEM_GUI_OFFSCREEN_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "skin.json"
        M.save(model_with({"mesher": "getfem", "h": 0.2}, "top"), path)
        win = MainWindow(str(path))
        try:
            if win.viewer.plotter is None:
                return
            win.select(("geometries", 1))
            text = win.viewer.status.text()
            assert "the boundary of 'solid' ('top')" in text and "preview" in text, text
            win.select(("geometries", 1, "selections", "hole"))
            assert "boundary 'hole'" in win.viewer.status.text()
        finally:
            win.setWindowModified(False)
            win.close()


def test_element_size():
    """A new mesh: an element size of the scale of the geometry (the volume
    of a boundary: a tenth of its extent, a round number); a size larger
    than the geometry: an error (the meshers fail on it)."""
    m = M.load(Path(__file__).resolve().parents[1] / "examples" / "pin_fin_heat_sink_3d.json")
    assert M.geometry_extent(m, m["geometries"][0]) == 0.03 and M.default_element_size(m, m["geometries"][0]) == 0.002
    path = E.do_action(m, ("geometries",), "add_geometry:boundary")
    mesh = E.get_at(m, E.do_action(m, path, "add_mesh"))
    assert mesh["h" if mesh["mesher"] == "getfem" else "size"] == 0.002
    E._change_mesher(m, mesh, "getfem")
    mesh["h"] = 0.1
    assert any("0.1 is too large for the geometry (of size 0.03): about 0.002 expected" in e for e in M.validate(m))
    m2 = M.default_model()                           # 2D: a twentieth
    assert M.default_element_size(m2, m2["geometries"][0]) == 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
