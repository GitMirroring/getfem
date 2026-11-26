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

"""Ellipsoids: distance, format, editing, preview, mesh and solve.

    python -m pytest tests/       (or: python tests/test_ellipsoid.py)
"""

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_lab import editing as E, geometry3d as G3, mesher, model as M, parameters as P, runner  # noqa: E402

ELLIPSOID = {"type": "ellipsoid", "center": [0.1, 0.2, -0.1], "radii": ["A", 0.6, 0.35]}      # A = 1
HOLE = {"type": "ball", "center": [0.3, 0.3, -0.1], "radius": 0.15}
ROOT = ("geometries", 0, "source", "shape")


def area(a, b, c, n=1200):
    """The area of an ellipsoid (quadrature on its parametrization)."""
    u, v = np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n)
    U, V = np.meshgrid(u, v, indexing="ij")
    Xu = np.stack([a * np.cos(U) * np.cos(V), b * np.cos(U) * np.sin(V), -c * np.sin(U)])
    Xv = np.stack([-a * np.sin(U) * np.sin(V), b * np.sin(U) * np.cos(V), 0 * U])
    J = np.linalg.norm(np.cross(Xu, Xv, axis=0), axis=0)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz          # (NumPy < 2: trapz)
    return float(trapezoid(trapezoid(J, v, axis=1), u))


def model_with(shape, mesher_name="getfem"):
    m = M.default_model()
    m["parameters"] = [{"name": "A", "value": 1.0}]
    m["geometries"][0]["source"] = {"type": "csg", "dim": 3, "shape": copy.deepcopy(shape)}
    m["geometries"][0]["selections"] = {}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": mesher_name, "h": 0.12, "order": 2} \
        if mesher_name == "getfem" else {"name": "mesh", "geometry": "geometry", "mesher": mesher_name, "size": 0.12}
    md = m["models"][0]
    md["modules"][0]["conditions"] = [{"type": "temperature", "selection": "boundary", "value": 0.0}]
    md["outputs"] = [{"name": "volume", "expr": "1", "selection": "all"},
                     {"name": "area", "expr": "1", "selection": "boundary"}]
    return m


def test_distance():
    """The distance is euclidean: the nearest point on the surface, the
    segment to it along the normal there; on the planes of symmetry and for
    a sphere too; the sign; zero on the surface."""
    rng = np.random.default_rng(0)
    c = np.array([0.1, -0.2, 0.3])
    for r in ([1.0, 0.6, 0.35], [0.3, 0.9, 0.5], [0.5, 0.5, 0.8], [0.7, 0.7, 0.7]):
        r = np.asarray(r)
        P0 = c + rng.normal(size=(600, 3)) * 0.6
        P0[:100, 2], P0[100:200, 0], P0[200:250, :2] = c[2], c[0], c[:2]        # on the planes, an axis
        d = G3.ellipsoid_sdf(c, r, P0)
        D = P0 - c
        order = np.argsort(-r, kind="stable")
        Q = G3.ellipsoid_nearest(r[order], np.abs(D[:, order]))
        Qg = np.empty_like(Q)
        Qg[:, order] = Q * np.sign(D[:, order] + (D[:, order] == 0))         # back in place
        assert np.allclose(np.sum((Qg / r) ** 2, axis=1), 1, atol=1e-12)    # on the surface
        assert np.allclose(np.abs(d), np.linalg.norm(D - Qg, axis=1), atol=1e-12)
        n = Qg / r ** 2                                                     # the normal there
        far = np.abs(d) > 1e-6
        cross = np.linalg.norm(np.cross(D[far] - Qg[far], n[far]), axis=1)
        assert np.all(cross < 1e-8 * np.linalg.norm(n[far], axis=1) * np.abs(d[far]) + 1e-12)
        assert np.all((d < 0) == (np.sum((D / r) ** 2, axis=1) < 1))
        if np.ptp(r) == 0:
            assert np.allclose(d, np.linalg.norm(D, axis=1) - r[0], atol=1e-12)
    shape = {"type": "ellipsoid", "center": c.tolist(), "radii": [1.0, 0.6, 0.35]}
    lo, hi = G3.bounds(shape)
    assert np.allclose(lo, c - [1.0, 0.6, 0.35]) and np.allclose(hi, c + [1.0, 0.6, 0.35])
    assert G3.primitive_faces(shape) == [{"face": "surface"}]


def test_format_scripts_and_editing():
    m = model_with({"type": "difference", "of": [ELLIPSOID, HOLE]})
    assert M.validate(m) == []
    bad = copy.deepcopy(m)
    E.get_at(bad, ROOT + ("of", 0))["radii"] = [1, -0.5, 0.3]
    assert any("radii[1]" in e for e in M.validate(bad))
    flat = model_with(ELLIPSOID)
    flat["geometries"][0]["source"]["dim"] = 2
    assert any("ellipsoid only exists in 3D" in e for e in M.validate(flat))
    script = mesher.mesh_script(P.resolved(m), m["meshes"][0], "mesh")
    assert "gf.MesherObject('ellipsoid', [0.1, 0.2, -0.1], [1.0, 0.6, 0.35])" in script
    gm = model_with(ELLIPSOID, "gmsh")
    code = mesher.mesh_script(P.resolved(gm), gm["meshes"][0], "mesh")
    assert "addSphere(0.1, 0.2, -0.1, 1.0)" in code and "dilate(" in code and "1.0, 0.6, 0.35)" in code
    compile(code, "mesh.py", "exec")
    ng = model_with(ELLIPSOID, "netgen")
    assert any("ellipsoids are not meshed by Netgen" in e for e in M.validate(ng))
    # added to a 3D geometry; a ball turned into an ellipsoid (its face renamed in the selections)
    labels = [label for _, label in E.actions(m, ROOT)]
    assert "Add an ellipsoid" in labels
    m["geometries"][0]["selections"] = {"hole": {"kind": "boundary", "criteria": {"type": "faces", "faces": [
        {"shape": [1], "face": "sphere"}]}}}
    hole = ROOT + ("of", 1)
    E.apply(m, next(f for f in E.fields(m, hole) if f.op == "shape_type"), "ellipsoid")
    assert E.get_at(m, hole) == {"type": "ellipsoid", "center": [0.3, 0.3, -0.1], "radii": [0.15] * 3}
    faces = m["geometries"][0]["selections"]["hole"]["criteria"]["faces"]
    assert faces == [{"shape": [1], "face": "surface"}] and M.validate(m) == []
    assert [f.label for f in E.fields(m, hole)][1:] == ["Center", "Semi-axes"]
    E.apply(m, next(f for f in E.fields(m, hole) if f.op == "shape_type"), "ball")
    assert E.get_at(m, hole) == HOLE and faces == [{"shape": [1], "face": "sphere"}]


def test_face_of_a_mesh():
    """on_face (the scripts' selections of faces): the faces of a mesh on the
    surface of a turned ellipsoid, not those off it."""
    c, r = np.array([0.1, 0.2, -0.1]), np.array([1.0, 0.6, 0.35])
    th = math.radians(30)
    R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
    spec = {"kind": "ellipsoid", "center": c.tolist(), "radii": r.tolist(),
            "frame": {"origin": [0.5, 0.0, 0.0], "R": R.tolist()}}
    u = np.array([0.3, 0.31, 0.32]), np.array([1.0, 1.02, 1.05])
    local = c + np.column_stack([r[0] * np.sin(u[0]) * np.cos(u[1]), r[1] * np.sin(u[0]) * np.sin(u[1]),
                                 r[2] * np.cos(u[0])])
    Q = np.array([0.5, 0.0, 0.0]) + local @ R.T
    n = ((local.mean(axis=0) - c) / r ** 2) @ R.T
    n /= np.linalg.norm(n)
    assert G3.on_face(spec, Q, n, 1e-6) and not G3.on_face(spec, Q * 1.01, n, 1e-6)
    assert not G3.on_face(spec, Q, np.cross(n, [0.0, 0.0, 1.0]) / np.linalg.norm(np.cross(n, [0, 0, 1.0])), 1e-6)


def test_preview():
    pv = __import__("pytest").importorskip("pyvista")  # noqa: F841
    from getfem_lab.app import scene
    m = model_with({"type": "difference", "of": [
        {"type": "transform", "of": [ELLIPSOID], "angle": 30.0, "axis": [0, 0, 1.0], "center": [0.1, 0.2, -0.1],
         "translate": [0.0, 0.0, 0.0]}, HOLE]})
    shape = G3.shape_of(P.resolved(m)["geometries"][0]["source"])
    surf = scene.geometry_surface_3d(shape)
    assert abs(surf.area / (area(1.0, 0.6, 0.35) + 4 * math.pi * 0.15 ** 2) - 1) < 2e-3
    assert scene.face_refs(surf) == [{"face": "surface", "shape": [0, 0]}, {"face": "sphere", "shape": [1]}]


def test_mesh_and_solve():
    """Needs a GetFEM with the 'ellipsoid' mesher object."""
    m = model_with({"type": "difference", "of": [ELLIPSOID, HOLE]})
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(m, tmp)
        assert res.ok, f"failed: {res.error}\n{res.stderr[-2000:]}"
        q = res.results["models"]["heat"]["quantities"]
    assert abs(q["volume"] - 4 / 3 * math.pi * (1.0 * 0.6 * 0.35 - 0.15 ** 3)) < 3e-4
    assert abs(q["area"] / (area(1.0, 0.6, 0.35) + 4 * math.pi * 0.15 ** 2) - 1) < 3e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
