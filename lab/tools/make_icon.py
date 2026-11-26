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

"""Builds the icon of the application (getfem_lab/app/resources/icon_*.png):
a tripod in plane elasticity, its three ends fixed, its hole pulled down,
computed by the interface itself (GetFEM mesher, P2) and colored by the
displacement (viridis), on a light blue disc.

    cd lab && python tools/make_icon.py        (needs getfem, pyvista, Pillow)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from getfem_lab import model as M, runner  # noqa: E402

RESOURCES = LAB / "getfem_lab" / "app" / "resources"
SIZES = (512, 256, 128, 64, 32)


def tripod_model():
    arms, ends = [], {}
    for k in range(3):
        a = np.pi / 2 + 2 * np.pi * k / 3
        d = np.array([np.cos(a), np.sin(a)])
        n = np.array([-d[1], d[0]])
        w = 0.14
        P = [0.1 * d + w * n, 1.0 * d + w * n, 1.0 * d - w * n, 0.1 * d - w * n]
        arms.append({"type": "polygon", "points": [list(map(float, p)) for p in P]})
        ends[f"end{k}"] = {"kind": "boundary", "criteria": {"type": "in_ball", "center": list(map(float, d)),
                                                            "radius": 0.16}}
    shape = {"type": "difference",
             "of": [{"type": "union", "of": arms + [{"type": "ball", "center": [0, 0], "radius": 0.34}]},
                    {"type": "ball", "center": [0, 0], "radius": 0.17}]}
    sel = dict(ends, ends={"kind": "boundary", "criteria": {"type": "union", "of": list(ends)}},
               hole={"kind": "boundary", "criteria": {"type": "in_ball", "center": [0, 0], "radius": 0.2}})
    m = M.default_model()
    m["geometries"][0] = {"name": "tripod", "source": {"type": "csg", "dim": 2, "shape": shape}, "selections": sel}
    m["meshes"][0] = {"name": "mesh", "geometry": "tripod", "mesher": "getfem", "h": 0.03, "order": 2}
    m["models"] = [M.new_model("mech", "mesh", "linear_elasticity", 2)]
    m["models"][0]["modules"][0]["conditions"] = [{"type": "fixed", "selection": "ends"},
                                                  {"type": "traction", "selection": "hole", "value": [0.0, -1.0]}]
    m["models"][0]["vtu"] = "tripod.vtu"
    return m


def render(vtu, N=1024):
    import pyvista as pv
    from PIL import Image, ImageDraw, ImageFilter
    g = pv.read(vtu).extract_surface()
    g.point_data["s"] = np.linalg.norm(np.asarray(g.point_data["u"]), axis=1) ** 0.7
    pl = pv.Plotter(off_screen=True, window_size=(N, N))
    pl.set_background("white")
    pl.add_mesh(g, scalars="s", cmap="viridis", show_scalar_bar=False, lighting=False)
    pl.add_mesh(g.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False),
                color="#1c2b44", line_width=4)
    pl.view_xy()
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 1.22
    tri = Image.fromarray(pl.screenshot(transparent_background=True, return_img=True)).convert("RGBA")
    pl.close()
    # the background: a disc, a radial gradient of light blues, a thin rim, a soft shadow
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(xx - N / 2, yy - N / 2) / (N / 2)
    c0, c1 = np.array([245, 249, 255]), np.array([160, 190, 235])
    t = np.clip(r, 0, 1)[..., None] ** 1.6
    rgb = (c0 * (1 - t) + c1 * t).astype(np.uint8)
    alpha = (np.clip((0.985 - r) * N / 3, 0, 1) * 255).astype(np.uint8)
    bg = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    ImageDraw.Draw(bg).ellipse([N * 0.0075, N * 0.0075, N * 0.9925, N * 0.9925],
                               outline=(90, 120, 170, 255), width=int(N * 0.012))
    shadow = tri.split()[3].filter(ImageFilter.GaussianBlur(N * 0.012))
    sh = Image.new("RGBA", (N, N), (20, 40, 80, 0))
    sh.putalpha(shadow.point(lambda a: int(a * 0.35)))
    bg.alpha_composite(sh, (int(N * 0.012), int(N * 0.018)))
    bg.alpha_composite(tri)
    return bg


def main():
    m = tripod_model()
    errors = M.validate(m)
    if errors:
        raise SystemExit("\n".join(errors))
    with tempfile.TemporaryDirectory() as d:
        r = runner.run(m, d)
        if not r.ok:
            raise SystemExit(r.error)
        img = render(f"{d}/tripod.vtu")
    from PIL import Image
    for size in SIZES:
        img.resize((size, size), Image.LANCZOS).save(RESOURCES / f"icon_{size}.png")
        print(RESOURCES / f"icon_{size}.png")


if __name__ == "__main__":
    main()
