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

"""3D scenes (mesh, highlighted selection, result field) rendered off screen.

Skipped when PyVista is not installed.

    python -m pytest tests/       (or: python tests/test_scene.py)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pyvista as pv
except ImportError:  # pragma: no cover
    pv = None
    if "pytest" in sys.modules:
        import pytest
        pytest.skip("PyVista is not installed", allow_module_level=True)

from getfem_gui import mesher, model as M, runner  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def plotter():
    p = pv.Plotter(off_screen=True, window_size=(400, 300))
    p.set_background("white")
    return p


def highlight_pixels(img):
    """Number of pixels with the highlight color (orange-red)."""
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return int(np.sum((r > 180) & (g < 120) & (b < 90)))


def test_mesh_and_selection():
    from getfem_gui.app import scene
    model = M.load(EXAMPLES / "plate_with_hole.json")
    with tempfile.TemporaryDirectory() as tmp:
        name = M.physics_mesh(model)["name"]
        mres = mesher.ensure_mesh(model, name, tmp)
        assert mres.ok, mres.error
        mesh = scene.read(mres.view_file())
        assert mesh.n_cells == mres.info["nb_elements"]
        p = plotter()
        scene.show_mesh(p, mesh)
        img = p.screenshot(return_img=True)
        assert img.std() > 5 and highlight_pixels(img) == 0
        for sel in mres.info["view"]["selections"]:
            scene.show_mesh(p, mesh, scene.read(mres.view_file(sel)), reset=False)
            assert highlight_pixels(p.screenshot(return_img=True)) > 20, sel
        p.close()


def test_result_field_and_deformed_shape():
    from getfem_gui.app import scene
    model = M.load(EXAMPLES / "plate_hole_tension.json")
    with tempfile.TemporaryDirectory() as tmp:
        res = runner.run(model, tmp)
        assert res.ok, res.error
        ds = scene.read(Path(tmp) / res.results["vtu"])
        names = scene.field_names(ds)
        assert {"displacement", "von_mises"} <= set(names)
        p = plotter()
        scale = scene.deformation_scale(ds, "displacement")
        assert scale > 1
        scene.show_field(p, ds, "von_mises", warp="displacement", scale=scale)
        img = p.screenshot(return_img=True)
        assert img.std() > 5
        scene.show_field(p, ds, "displacement", component=0)
        assert p.scalar_bars
        p.close()


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok       {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAILED   {name}: {exc!r}")
    sys.exit(1 if failed else 0)
