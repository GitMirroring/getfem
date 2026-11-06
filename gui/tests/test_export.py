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

"""Export of geometries (SVG, DXF, STL, OBJ, PLY, VTP, STEP...) and meshes
(GetFEM, VTK, Gmsh, Abaqus).

    python -m pytest tests/       (or: python tests/test_export.py)
"""

import copy
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import cli, export as X, model as M  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def bracket(order=2, h=0.02):
    m = M.load(EXAMPLES / "l_bracket.json")
    m["meshes"][0].update(h=h, order=order)
    return m


def block3d(order=2):
    m = M.default_model()
    m["geometries"][0]["source"] = {"type": "csg", "dim": 3, "shape": {"type": "difference", "of": [
        {"type": "rectangle", "min": [0, 0, 0], "max": [1, 1, 0.5]},
        {"type": "cylinder", "origin": [0.5, 0.5, -1], "axis": [0, 0, 1], "length": 3, "radius": 0.2}]}}
    m["geometries"][0]["selections"] = {
        "hole": {"kind": "boundary", "criteria": {"type": "faces", "faces": [{"shape": [1], "face": "side"}]}},
        "half": {"kind": "domain", "criteria": {"type": "in_box", "min": [0, 0, 0], "max": [0.5, 1, 0.5]}}}
    m["meshes"][0] = {"name": "mesh", "geometry": "geometry", "mesher": "getfem", "h": 0.25, "order": order}
    m["models"][0]["modules"][0]["conditions"] = [{"type": "temperature", "selection": "hole", "value": 0}]
    return m


def test_outlines():
    m = bracket()
    with tempfile.TemporaryDirectory() as tmp:
        svg = X.export_geometry(m, "bracket", Path(tmp) / "b.svg").read_text()
        dxf = X.export_geometry(m, "bracket", Path(tmp) / "b.dxf").read_text()
    assert svg.startswith("<?xml") and svg.count("<path") == 7          # 6 sides, the hole
    hole = [d for d in re.findall(r'd="([^"]+)"', svg) if " A " in d][0]
    assert hole.count(" A ") == 2                                        # a whole circle: two arcs
    lines = dxf.splitlines()
    assert lines.count("LINE") == 6 and lines.count("CIRCLE") == 1 and lines[-1] == "EOF"
    # an ellipse cut by a rectangle: an arc of ellipse
    e = M.default_model()
    e["geometries"][0]["source"] = {"type": "csg", "dim": 2, "shape": {"type": "intersection", "of": [
        {"type": "ellipse", "center": [0, 0], "radii": [2, 1], "angle": 30},
        {"type": "rectangle", "min": [0, -3], "max": [3, 3]}]}}
    with tempfile.TemporaryDirectory() as tmp:
        svg = X.export_geometry(e, "geometry", Path(tmp) / "e.svg").read_text()
        dxf = X.export_geometry(e, "geometry", Path(tmp) / "e.dxf").read_text()
    assert " A 2 1 -30 " in svg and "POLYLINE" in dxf
    try:
        X.export_geometry(e, "geometry", "e.stl")
        raise AssertionError("a 3D format accepted for a 2D geometry")
    except X.ExportError as exc:
        assert "not a format for a 2D geometry" in str(exc)


def test_cad_script():
    code = X.cad_script(block3d()["geometries"][0]["source"]["shape"], 3, "block.step")
    assert "gmsh.model.occ.addBox" in code and "gmsh.write('block.step')" in code
    compile(code, "export_cad.py", "exec")
    assert set(X.geometry_formats(2)) >= {".svg", ".dxf", ".step", ".iges", ".brep"}


def test_surfaces():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return
    m = block3d()
    with tempfile.TemporaryDirectory() as tmp:
        stl = X.export_geometry(m, "geometry", Path(tmp) / "b.stl")
        n = int(np.frombuffer(stl.read_bytes()[80:84], "<u4")[0])
        assert stl.stat().st_size == 84 + 50 * n and n > 100
        rec = np.frombuffer(stl.read_bytes()[84:], dtype=[("n", "<f4", 3), ("a", "<f4", 3), ("b", "<f4", 3),
                                                          ("c", "<f4", 3), ("attr", "<u2")])
        area = 0.5 * np.linalg.norm(np.cross(rec["b"] - rec["a"], rec["c"] - rec["a"]), axis=1).sum()
        exact = 2 * (1 - math.pi * 0.04) + 4 * 0.5 + 2 * math.pi * 0.2 * 0.5
        assert abs(area / exact - 1) < 2e-3
        obj = X.export_geometry(m, "geometry", Path(tmp) / "b.obj").read_text().splitlines()
        assert sum(1 for line in obj if line.startswith("f ")) == n
        ply = X.export_geometry(m, "geometry", Path(tmp) / "b.ply").read_text()
        assert f"element face {n}" in ply
        assert X.export_geometry(m, "geometry", Path(tmp) / "b.vtp").stat().st_size > 0


def _measures(m):
    import getfem as gf
    deg = 6 if m.dim() == 2 else 5
    mim = gf.MeshIm(m, gf.Integ(f"IM_{'TRIANGLE' if m.dim() == 2 else 'TETRAHEDRON'}({deg})"))
    return gf.asm("generic", mim, 0, "1", -1), {int(r): gf.asm("generic", mim, 0, "1", int(r)) for r in m.regions()}


def _read_back(path):
    """Measures of a Gmsh file read by GetFEM: (volume, {physical tag: measure})."""
    import getfem as gf
    return _measures(gf.Mesh("import", "gmsh", str(path)))


def _original(work):
    """Measures of the mesh built in work: (volume, {selection: measure})."""
    import json
    import getfem as gf
    info = json.loads((work / "meshes" / "mesh.json").read_text())
    v, regs = _measures(gf.Mesh("load", str(work / "meshes" / "mesh.mesh")))
    return v, {n: regs[r] for n, r in info["regions"].items() if r in regs}


def test_meshes():
    """Needs GetFEM (with the polygon mesher object); P2 triangles and tetrahedra."""
    try:
        import getfem  # noqa: F401
    except ImportError:
        return
    for model, mesh_name, vol, sels in [
            (bracket(), "mesh", 0.1 * 0.3 + 0.15 * 0.1 - math.pi * 0.0004, {"top": 0.1, "end": 0.1}),
            (block3d(), "mesh", 0.5 * (1 - math.pi * 0.04), {"hole": 2 * math.pi * 0.2 * 0.5})]:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            try:
                X.export_mesh(model, mesh_name, work, Path(tmp) / "x.msh")
                raise AssertionError("a mesh not built was exported")
            except X.ExportError as exc:
                assert "build it first" in str(exc)
            msh = X.export_mesh(model, mesh_name, work, Path(tmp) / "x.msh", build=True)
            text = msh.read_text()
            v, regs = _read_back(msh)
            v0, regs0 = _original(work)
            assert abs(v - v0) < 1e-12 * v0 and abs(v / vol - 1) < 5e-3, (v, v0, vol)
            names = dict(re.findall(r'^\d (\d+) "(\w+)"$', text, re.M))
            for name, size in sels.items():          # the same measures as GetFEM, close to the exact ones
                tag = int(next(t for t, n in names.items() if n == name))
                assert abs(regs[tag] - regs0[name]) < 1e-12 and abs(regs[tag] / size - 1) < 0.1, \
                    (name, regs[tag], regs0[name], size)
            inp = X.export_mesh(model, mesh_name, work, Path(tmp) / "x.inp").read_text()
            etype = "CPS6" if "top" in sels else "C3D10"
            assert f"*ELEMENT, TYPE={etype}, ELSET=domain" in inp
            assert all(f"*NSET, NSET={n}" in inp for n in sels)
            for ext in (".vtu", ".vtk", ".mesh"):
                assert X.export_mesh(model, mesh_name, work, Path(tmp) / f"x{ext}").stat().st_size > 1000
    with tempfile.TemporaryDirectory() as tmp:     # a domain selection: a physical group, an element set
        model = block3d(order=1)
        work = Path(tmp) / "work"
        text = X.export_mesh(model, "mesh", work, Path(tmp) / "b.msh", build=True).read_text()
        assert '3 4 "half"' in text
        assert "*ELSET, ELSET=half" in X.export_mesh(model, "mesh", work, Path(tmp) / "b.inp").read_text()


def test_command_line():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "b.json"
        M.save(bracket(), f)
        assert cli.main(["export-geometry", str(f), "bracket", str(Path(tmp) / "b.svg")]) == 0
        assert (Path(tmp) / "b.svg").exists()
        assert cli.main(["export-geometry", str(f), "nothing", str(Path(tmp) / "b.svg")]) == 1


def test_window():
    import os
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    os.environ.update(QT_QPA_PLATFORM="offscreen", GETFEM_GUI_NO_3D="1")
    from getfem_gui.app.window import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "b.json"
        M.save(bracket(order=1, h=0.03), f)
        win = MainWindow(f)
        try:
            win.select(("geometries", 0))
            assert ("cmd:export_geometry", "Export...") in win.menu_actions(("geometries", 0))
            assert win.export_geometry(path=str(Path(tmp) / "g.dxf")) and (Path(tmp) / "g.dxf").exists()
            win.select(("meshes", 0))
            assert ("cmd:export_mesh", "Export...") in win.menu_actions(("meshes", 0))
            assert not win.export_mesh(path=str(Path(tmp) / "m.msh"), build=False)
            assert win.export_mesh(path=str(Path(tmp) / "m.msh"), build=True)
            assert "$MeshFormat" in (Path(tmp) / "m.msh").read_text() and "written to" in win.state.text()
        finally:
            win.setWindowModified(False)
            win.close()


def test_missing_mesher_object_message():
    """A GetFEM without the new mesher objects: the error says which patch."""
    from getfem_gui import mesher
    err = mesher.error_line("Traceback (most recent call last):\n  File \"mesh.py\"\n"
                            "RuntimeError: (Getfem::InterfaceError) -- Bad command name: polygon\n")
    assert "Bad command name: polygon" in err and "mesher: polygons" in err and "Gmsh" in err


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
