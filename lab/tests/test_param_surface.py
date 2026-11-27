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

"""Parametric surfaces ({"type": "surface", "shape": <flat domain in (u,
v)>, "parametric": {"x", "y", "z"}}): validation (expressions, poles), the
seams found (a cylinder, a torus, a Moebius strip), the meshes made (their
seams closed, curved, sized on the surface), editing, the preview, a heat
problem.

    python -m pytest tests/test_param_surface.py      (or: python tests/test_param_surface.py)
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, manifold, mesher, model as M, parameters as P  # noqa: E402

CYLINDER = {"x": "R*cos(u)", "y": "R*sin(u)", "z": "v"}
TORUS = {"x": "(2 + cos(v))*cos(u)", "y": "(2 + cos(v))*sin(u)", "z": "sin(v)"}
MOEBIUS = {"x": "(1 + v*cos(u/2))*cos(u)", "y": "(1 + v*cos(u/2))*sin(u)", "z": "v*sin(u/2)"}


def model_with(par, lo, hi, mesh=None, shape=None):
    m = M.default_model()
    m["parameters"] = [{"name": "R", "value": 0.5}]
    m["geometries"][0] = {"name": "s", "selections": {}, "source": {
        "type": "surface", "shape": shape or {"type": "rectangle", "min": lo, "max": hi}, "parametric": dict(par)}}
    m["meshes"][0] = dict({"name": "mesh", "geometry": "s"}, **(mesh or {"mesher": "getfem", "h": 0.2}))
    m["models"][0]["modules"][0]["conditions"] = []
    m["models"][0]["outputs"] = [{"name": "area", "expr": "1", "selection": "all"}]
    return m


def errors(m):
    return [e for e in M.validate(m) if not e.startswith("models")]


def resolved_source(m):
    return P.resolved(m)["geometries"][0]["source"]


def test_validation():
    """Expressions of u, v and the parameters; finite; no pole (the
    tangents independent everywhere on the flat domain)."""
    assert errors(model_with(CYLINDER, [0, 0], ["2*pi", 1])) == []
    for par, msg in [(dict(CYLINDER, z="w"), "unknown name 'w'"),
                     (dict(CYLINDER, z="v**2"), "'**' is not available"),
                     (dict(CYLINDER, z="log(v)"), "not finite at (u, v) = (0"),
                     ({"x": "cos(u)*sin(v)", "y": "sin(u)*sin(v)", "z": "cos(v)"}, "degenerates at (u, v) = ("),
                     ({"x": "u", "y": "u", "z": "0"}, "degenerates")]:
        lo = [0, 0]
        m = model_with(par, lo, ["2*pi", "pi"] if "sin(v)" in str(par) else ["2*pi", 1])
        assert any(msg in e for e in errors(m)), (msg, errors(m))
    m = model_with({"x": "cos(u)*sin(v)", "y": "sin(u)*sin(v)", "z": "cos(v)"}, [0, 0.2], ["2*pi", 2.9])
    assert errors(m) == []                                     # a sphere without its poles
    m = model_with(CYLINDER, [0, 0], ["2*pi", 1], {"mesher": "netgen", "size": 0.2})
    assert any("not meshed by Netgen" in e for e in errors(m))


def test_seams():
    """The sides that the map sends to the same curve: in the same
    direction (a cylinder: u; a torus: u and v) or reversed (a Moebius
    strip); none for a part of a cylinder."""
    vals = {"R": 0.5}
    found = manifold.seams(resolved_source(model_with(CYLINDER, [0, 0], ["2*pi", 1])), vals)
    assert [(s["axis"], s["reversed"]) for s in found] == [(0, False)]
    assert np.allclose(found[0]["at"], (0, 2 * np.pi)) and np.allclose(found[0]["range"], (0, 1))
    found = manifold.seams(resolved_source(model_with(TORUS, [0, 0], ["2*pi", "2*pi"])), vals)
    assert [(s["axis"], s["reversed"]) for s in found] == [(0, False), (1, False)]
    found = manifold.seams(resolved_source(model_with(MOEBIUS, [0, -0.3], ["2*pi", 0.3])), vals)
    assert [(s["axis"], s["reversed"]) for s in found] == [(0, True)]
    assert manifold.seams(resolved_source(model_with(CYLINDER, [0, 0], ["pi", 1])), vals) == []
    src = resolved_source(model_with(CYLINDER, [0, 0], ["2*pi", 1]))
    V = manifold.seam_vertices(src, vals, manifold.seams(src, vals)[0], lambda p: np.full(p.shape[1], 0.1))
    n = V.shape[1] // 2
    assert np.allclose(V[1, :n], V[1, n:]) and np.allclose(V[0, :n], 0) and np.allclose(V[0, n:], 2 * np.pi)
    sc = manifold.scaled_surface(src, 0.5, 1.0)
    assert sc["shape"]["max"] == [np.pi, 1.0] and sc["parametric"]["x"] == "R * cos(u / 0.5)"
    assert manifold.scaled_shape({"type": "polygon", "points": [[0, 0], [1, 0], [0, 1]],
                                  "fillets": [{"vertex": 1, "radius": 0.1}]}, 2, 1) is None


def _mesh(m):
    import getfem as gf
    with tempfile.TemporaryDirectory() as tmp:
        res = mesher.ensure_mesh(m, "mesh", tmp)
        assert res.ok, res.error
        msh = gf.Mesh("load", str(res.path))
        mim = gf.MeshIm(msh, gf.Integ("IM_TRIANGLE(6)"))
        rg = res.info["regions"]
        return msh, gf.asm_generic(mim, 0, "1", -1), gf.asm_generic(mim, 0, "1", rg["boundary"]), res.info


def test_meshes():
    """A cylinder: its seam closed (the boundary: its two circles), of degree
    2 closer to the area; elements of about the size on the surface (the
    flat domain scaled); a torus: closed; a Moebius strip: one edge."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    R = 0.5
    msh, area1, edge, info = _mesh(model_with(CYLINDER, [0, 0], ["2*pi", 1]))
    assert abs(edge - 4 * np.pi * R) < 0.05 and info["dim"] == 3              # (polygons)
    n = msh.nbcvs()
    assert 2 * np.pi * R / (0.2 ** 2 * np.sqrt(3) / 4) / 2 < n < 2 * 2 * np.pi * R / (0.2 ** 2 * np.sqrt(3) / 4)
    msh, area2, edge, _ = _mesh(model_with(CYLINDER, [0, 0], ["2*pi", 1], {"mesher": "getfem", "h": 0.2,
                                                                           "order": 2}))
    assert msh.geotrans()[0].char() == "GT_PK(2,2)"
    assert abs(area2 - 2 * np.pi * R) < 1e-4 < abs(area1 - 2 * np.pi * R)
    assert abs(edge - 4 * np.pi * R) < 1e-3
    _, area, edge, info = _mesh(model_with(TORUS, [0, 0], ["2*pi", "2*pi"], {"mesher": "getfem", "h": 0.4,
                                                                               "order": 2}))
    assert info["sizes"]["boundary"] == 0 and abs(area / (8 * np.pi ** 2) - 1) < 1e-4
    _, area, edge, info = _mesh(model_with(MOEBIUS, [0, -0.3], ["2*pi", 0.3], {"mesher": "getfem", "h": 0.1}))
    assert 12.5 < edge < 12.8                                 # one edge (no seam left), about 4 pi long
    _, area, edge, info = _mesh(model_with({"x": "v*cos(u)", "y": "v*sin(u)", "z": "v"}, [0, 0.5], ["pi", 2],
                                           {"mesher": "getfem", "h": 0.15}))
    assert abs(area - np.pi * np.sqrt(2) * (4 - 0.25) / 2) < 0.02            # half a cone: sized along v


def test_scripts():
    """Gmsh: the seams periodic; local sizes where the stretch varies; the
    scripts compile."""
    m = model_with(CYLINDER, [0, 0], ["2*pi", 1], {"mesher": "gmsh", "size": 0.2, "order": 2})
    r = P.resolved(m)
    text = mesher.mesh_script(r, r["meshes"][0], "mesh")
    compile(text, "gmsh.py", "exec")
    assert "gmsh.model.mesh.setPeriodic(1, [_slave[0][1]], [_master[0][1]]" in text
    assert "meshed scaled by (0.5, 1)" in text and "m.transform([[2.0, 0.0], [0.0, 1.0]])" in text
    assert "setSizeCallback" not in text                     # a cylinder: one size
    m = model_with({"x": "v*cos(u)", "y": "v*sin(u)", "z": "v"}, [0, 0.5], ["pi", 2], {"mesher": "gmsh", "size": 0.2})
    r = P.resolved(m)
    text = mesher.mesh_script(r, r["meshes"][0], "mesh")
    compile(text, "gmsh.py", "exec")
    assert "setSizeCallback" in text and "setPeriodic" not in text
    m = model_with(CYLINDER, [0, 0], ["pi", 1], {"mesher": "netgen", "size": 0.2})
    r = P.resolved(m)
    compile(mesher.mesh_script(r, r["meshes"][0], "mesh"), "netgen.py", "exec")


def test_editing():
    """The kind 'Parametric surface' (a cylinder), its form, the map
    switched to a plane and a height and back (the flat domain kept)."""
    m = M.default_model()
    path = E.do_action(m, ("geometries",), "add_geometry:surface_param")
    assert errors(m) == []
    fields = E.fields(m, path)
    assert [f.label for f in fields][2:6] == ["Map", "x(u, v)", "y(u, v)", "z(u, v)"]
    f = fields[2]
    assert E.field_value(m, f) is True
    assert E.find_node(E.tree(m), path).children[0].label.startswith("Flat domain (u, v): ")
    E.apply(m, f, False)
    src = E.get_at(m, path)["source"]
    assert "parametric" not in src and src["height"] == 0.0 and src["shape"]["max"] == ["2*pi", 1.0]
    assert errors(m) == [] and E.field_value(m, f) is False
    E.apply(m, f, True)
    assert src["parametric"] == {"x": "u", "y": "v", "z": "0"} and "origin" not in src and errors(m) == []
    mesh = E.do_action(m, path, "add_mesh")
    assert errors(m) == [] and E.get_at(m, mesh)["mesher"] in M.FREE_MESHERS


def test_solve():
    """The example: a pipe heated on half its wall, its ends at a fixed
    temperature: the heat leaves by them."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    from getfem_lab import runner
    m = M.load(Path(__file__).resolve().parents[1] / "examples" / "heated_pipe.json")
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, res.error
    q = res.results["models"]["heat"]["quantities"]
    R, L = 0.05, 0.6
    assert abs(q["area"] / (2 * np.pi * R * L) - 1) < 1e-4 and abs(q["heated_area"] / (np.pi * R * L) - 1) < 1e-4
    assert abs((q["heat_out_inlet"] + q["heat_out_outlet"]) / (1000 * np.pi * R * L) - 1) < 1e-3
    assert abs(q["heat_out_inlet"] - q["heat_out_outlet"]) < 1e-6 * q["heat_out_inlet"]


def test_preview():
    """The node of the geometry: the surface, its seam closed."""
    try:
        from PySide6.QtWidgets import QApplication
        import pyvista  # noqa: F401
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"), GETFEM_LAB_NO_3D="",
                      GETFEM_LAB_OFFSCREEN_3D="1")
    from getfem_lab.app import scene
    m = model_with(CYLINDER, [0, 0], ["2*pi", 1])
    surf = scene.manifold_preview(resolved_source(m), {"R": 0.5})
    edges = surf.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                       non_manifold_edges=False)
    assert abs(edges.compute_cell_sizes()["Length"].sum() - 4 * np.pi * 0.5) < 0.01     # no seam
    from PySide6.QtWidgets import QApplication
    from getfem_lab.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cyl.json"
        M.save(m, path)
        win = MainWindow(str(path))
        try:
            if win.viewer.plotter is None:
                return
            win.select(("geometries", 0))
            assert "s: parametric surface" in win.viewer.status.text()
        finally:
            win.setWindowModified(False)
            win.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
