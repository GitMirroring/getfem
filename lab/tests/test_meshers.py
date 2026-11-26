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

"""Meshers: Netgen, quadrangles with Gmsh and Netgen, meshes in layers of
extrusions and revolutions (prisms, hexahedra).

Gmsh and Netgen are optional: without them, the scripts are only checked
(they are compiled); the meshes in layers are built with the GetFEM mesher.

    python -m pytest tests/       (or: python tests/test_meshers.py)
"""

import copy
import importlib.util
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, export as X, mesher, model as M, parameters as P, runner  # noqa: E402

HAS_GMSH = importlib.util.find_spec("gmsh") is not None
HAS_NETGEN = importlib.util.find_spec("netgen") is not None

U = [[0, 0], [1, 0], [1, 1], [0.6, 1], [0.6, 0.3], [0.4, 0.3], [0.4, 1], [0, 1]]
EXTRUDED = {"type": "extrusion", "z": [0, 0.4], "profile": {"type": "polygon", "points": U}}
TUBE = {"type": "revolution", "profile": {"type": "difference", "of": [
    {"type": "rectangle", "min": [0.5, 0], "max": [1, 1]},
    {"type": "ball", "center": [1, 0.5], "radius": 0.2}]}}
PLATE = {"type": "difference", "of": [{"type": "rectangle", "min": [0, 0], "max": [2, 1]},
                                      {"type": "ball", "center": [1, 0.5], "radius": 0.25}]}


def model_with(shape, dim, mesh):
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "csg", "dim": dim, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = {}
    m["meshes"][0] = dict({"name": "mesh", "geometry": "geometry"}, **mesh)
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": "x + 2*y"}]
    md["outputs"] = [{"name": "volume", "expr": "1", "selection": "all"},
                     {"name": "area", "expr": "1", "selection": "boundary"},
                     {"name": "err2", "expr": "sqr(T - x - 2*y)", "selection": "all"}]
    return m


def script(m):
    r = P.resolved(m)
    return mesher.mesh_script(r, r["meshes"][0], "mesh")


def _solve(m):
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
        return res.results["models"]["heat"]["quantities"], res


def test_validation():
    ok = model_with(EXTRUDED, 3, {"mesher": "getfem", "h": 0.1, "order": 1, "layers": 4})
    assert M.validate(ok) == []
    q = model_with(EXTRUDED, 3, {"mesher": "gmsh", "size": 0.1, "layers": 4, "element": "quad"})
    assert M.validate(q) == []
    for mesh, msg in [
            ({"mesher": "getfem", "h": 0.1, "layers": 4, "element": "quad"}, "GetFEM mesher makes triangles"),
            ({"mesher": "gmsh", "size": 0.1, "element": "quad"}, "hexahedra by layers only"),
            ({"mesher": "gmsh", "size": 0.1, "layers": -1}, "integer >= 1"),
            ({"mesher": "netgen", "size": 0.1, "order": 3}, "1 or 2 expected")]:
        assert any(msg in e for e in M.validate(model_with(EXTRUDED, 3, mesh))), msg
    bad = model_with(PLATE, 2, {"mesher": "gmsh", "size": 0.1, "layers": 3})
    assert any("only for a 3D geometry that is an extrusion" in e for e in M.validate(bad))
    axis = {"type": "revolution", "profile": {"type": "rectangle", "min": [0, 0], "max": [1, 1]}}
    errs = M.validate(model_with(axis, 3, {"mesher": "getfem", "h": 0.1, "layers": 2}))
    assert any("at least 3 layers" in e for e in errs) and any("must not touch the axis" in e for e in errs)
    ell = {"type": "ellipse", "center": [0, 0], "radii": [1, 0.5], "angle": 0}
    assert any("ellipses are not meshed by Netgen" in e
               for e in M.validate(model_with(ell, 2, {"mesher": "netgen", "size": 0.1})))
    assert M.validate(model_with(PLATE, 2, {"mesher": "gmsh", "size": 0.1, "element": "quad"})) == []


def test_editing():
    m = model_with(EXTRUDED, 3, {"mesher": "gmsh", "size": 0.1, "order": 2})
    keys = {f.key[-1]: f for f in E.fields(m, ("meshes", 0))}
    assert "layers" in keys and "element" not in keys
    assert "netgen" in [v for v, _ in keys["mesher"].choices]
    m["meshes"][0]["layers"] = 4
    keys = {f.key[-1]: f for f in E.fields(m, ("meshes", 0))}
    assert [v for v, _ in keys["element"].choices] == ["simplex", "quad"]
    m["meshes"][0]["element"] = "quad"
    E.apply(m, keys["mesher"], "netgen")              # size, order, layers and quadrangles kept
    assert m["meshes"][0] == {"name": "mesh", "geometry": "geometry", "mesher": "netgen", "size": 0.1,
                              "order": 2, "layers": 4, "element": "quad"}
    E.apply(m, next(f for f in E.fields(m, ("meshes", 0)) if f.key[-1] == "mesher"), "getfem")
    assert m["meshes"][0]["h"] == 0.1 and "element" not in m["meshes"][0] and m["meshes"][0]["layers"] == 4
    assert M.validate(m) == []
    flat = model_with(PLATE, 2, {"mesher": "netgen", "size": 0.1})
    keys = {f.key[-1]: f for f in E.fields(flat, ("meshes", 0))}
    assert "element" in keys and "layers" not in keys


def test_scripts():
    """Gmsh quadrangles, Netgen (2D, 3D, quadrangles, layers): the scripts compile."""
    code = script(model_with(PLATE, 2, {"mesher": "gmsh", "size": 0.1, "element": "quad"}))
    assert "Mesh.RecombineAll', 1" in code and "Mesh.SubdivisionAlgorithm', 1" in code
    assert "Mesh.MeshSizeMax', 0.2" in code                # doubled: the subdivision halves it
    compile(code, "mesh.py", "exec")
    code = script(model_with(PLATE, 2, {"mesher": "netgen", "size": 0.1, "element": "quad", "order": 2}))
    assert "from netgen.occ import *" in code and "Circle(1.0, 0.5, 0.25)" in code
    assert "GenerateMesh(maxh=0.1, quad_dominated=True)" in code and "SecondOrder()" in code
    assert "OCCGeometry(s3, dim=2)" in code and "s1 - s2" in code
    compile(code, "mesh.py", "exec")
    code = script(model_with({"type": "difference", "of": [TUBE, {"type": "cylinder", "origin": [0, 0, -1],
                                                                  "axis": [1, 0, 0], "length": 3, "radius": 0.1}]},
                             3, {"mesher": "netgen", "size": 0.1}))
    assert ".Revolve(Axis((0, 0, 0), Z), 360)" in code and "Cylinder(Pnt(0.0, 0.0, -1.0)" in code
    compile(code, "mesh.py", "exec")
    code = script(model_with(EXTRUDED, 3, {"mesher": "netgen", "size": 0.1, "layers": 4, "element": "quad"}))
    assert "OCCGeometry(s1, dim=2)" in code and "gf.Mesh('prismatic', m, 4, 1)" in code
    compile(code, "mesh.py", "exec")
    code = script(model_with(EXTRUDED, 3, {"mesher": "netgen", "size": 0.1}))
    assert ".Extrude(0.4).Move(Vec(0, 0, 0.0))" in code
    err = mesher.error_line("ModuleNotFoundError: No module named 'netgen'")
    assert "pip install netgen-mesher" in err


def test_layers_extrusion():
    """A U shaped profile extruded in layers of prisms: exact volume and
    area, a linear solution exact; the mesh exported to Gmsh reads back."""
    m = model_with(EXTRUDED, 3, {"mesher": "getfem", "h": 0.1, "order": 1, "layers": 4})
    q, res = _solve(m)
    assert abs(q["volume"] - 0.86 * 0.4) < 1e-12 and abs(q["area"] - (2 * 0.86 + 0.4 * 5.4)) < 1e-12
    assert q["err2"] < 1e-20
    info = res.mesh.info if hasattr(res, "mesh") else None
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "w"
        text = X.export_mesh(m, "mesh", work, Path(tmp) / "p.msh", build=True).read_text()
        elements = text.split("$Elements")[1].splitlines()[2:]
        assert any(line.split()[1] == "6" for line in elements)            # prisms
    assert info is None or info["nb_elements"] > 0


def test_layers_revolution():
    """A grooved ring turned in 24 layers (quadratic along the angle)."""
    m = model_with(TUBE, 3, {"mesher": "getfem", "h": 0.1, "order": 2, "layers": 24})
    q, _ = _solve(m)
    r, R = 0.2, 1.0
    # volume: Pappus (the half disk removed has its centroid at R - 4r/(3 pi))
    vol = math.pi * (1 - 0.25) - math.pi * r * r / 2 * 2 * math.pi * (R - 4 * r / (3 * math.pi))
    assert abs(q["volume"] / vol - 1) < 2e-3, (q["volume"], vol)
    assert q["err2"] < 1e-8


def test_quadrangles_and_netgen_meshes():
    """With Gmsh or Netgen installed: quadrangles in 2D, hexahedra in layers."""
    meshers = [x for x, ok in (("gmsh", HAS_GMSH), ("netgen", HAS_NETGEN)) if ok]
    for name in meshers:
        q, _ = _solve(model_with(PLATE, 2, {"mesher": name, "size": 0.1, "order": 2, "element": "quad"}))
        assert abs(q["volume"] - (2 - math.pi / 16)) < 1e-3
        q, _ = _solve(model_with(EXTRUDED, 3, {"mesher": name, "size": 0.1, "layers": 4, "element": "quad"}))
        assert abs(q["volume"] - 0.86 * 0.4) < 1e-9
        q, _ = _solve(model_with(TUBE, 3, {"mesher": name, "size": 0.15}))
        assert q["volume"] > 0


def test_out_of_date_mesh_shown_only_for_its_geometry():
    """A mesh out of date is still shown when its geometry was edited, not
    when its name is that of a mesh of another geometry (a new mesh taking
    the name of a deleted one): a 3D mesh shown for a 2D one."""
    m = model_with(PLATE, 2, {"mesher": "getfem", "h": 0.3, "order": 1})
    with tempfile.TemporaryDirectory() as tmp:
        res = mesher.ensure_mesh(m, "mesh", tmp)
        assert res.ok, res.error
        assert res.info["origin"] == {"geometry": "geometry", "type": "csg", "dims": [2, 2]}
        edited = copy.deepcopy(m)                         # the geometry edited: shown, out of date
        edited["geometries"][0]["source"]["shape"]["of"][1]["radius"] = 0.3
        assert mesher.cached_mesh(edited, "mesh", tmp) is None
        assert mesher.cached_mesh(edited, "mesh", tmp, up_to_date=False).stale
        other = copy.deepcopy(m)                          # the mesh on another (3D) geometry
        other["geometries"].append({"name": "block", "source": {"type": "box", "size": [1, 1, 1]},
                                    "selections": {}})
        other["meshes"][0]["geometry"] = "block"
        assert mesher.cached_mesh(other, "mesh", tmp, up_to_date=False) is None
        same_name = copy.deepcopy(m)                      # same name, but now 3D
        same_name["geometries"][0]["source"] = {"type": "box", "size": [1, 1, 1]}
        assert mesher.cached_mesh(same_name, "mesh", tmp, up_to_date=False) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
