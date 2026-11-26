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

"""Builds the logo of GetFEM: the tripod of the icon (make_icon.py), "Get",
and "FEM" in meshed letters (F in quadrangles, E in triangles, M mixed:
quadrangles in its stems, triangles in its V), colored (viridis) by a
smooth field. The letters are those of Inter Display (SIL Open Font
License), as outlines: the SVG needs no font.

Writes the logo as SVG (doc/sphinx/source/.static/logo_getfem.svg) and its
PNG renderings: the documentation (index page, LaTeX, sidebar, favicon),
the Scilab help, the start-up window of the interface.

    cd lab && python tools/make_logo.py

Needs fontTools, numpy, scipy, matplotlib (the colormap, point in polygon),
Pillow, rsvg-convert (librsvg), and the Inter Display font (fc-match).
"""

import base64
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from matplotlib.path import Path as MPath
from scipy.spatial import Delaunay

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent
ICON = LAB / "getfem_lab" / "app" / "resources" / "icon_512.png"
NAVY, TEAL = "#16324f", "#1f6f78"
H_MESH = 150.0                      # the size of the elements (font units; cap height 1490)


def font_file(style):
    """The file of Inter Display in this style (fontconfig)."""
    f = subprocess.run(["fc-match", "-f", "%{file}", f"Inter Display:style={style}"],
                       capture_output=True, text=True, check=True).stdout
    if "InterDisplay" not in Path(f).name:
        raise SystemExit(f"the font Inter Display ({style}) is not installed (fc-match gives {f})")
    return f


# ---------------------------------------------------------------- the text

def kern_table(font):
    """Pair kerning of the GPOS table (pair adjustments, formats 1 and 2)."""
    pairs, classes = {}, []
    if "GPOS" not in font:
        return lambda a, b: 0
    for lookup in font["GPOS"].table.LookupList.Lookup:
        for st in lookup.SubTable:
            if st.LookupType == 9:
                st = st.ExtSubTable
            if getattr(st, "LookupType", None) != 2:
                continue
            if st.Format == 1:
                cov = st.Coverage.glyphs
                for i, ps in enumerate(st.PairSet):
                    for r in ps.PairValueRecord:
                        pairs.setdefault((cov[i], r.SecondGlyph), getattr(r.Value1, "XAdvance", 0) or 0)
            else:
                classes.append(st)

    def k(a, b):
        if (a, b) in pairs:
            return pairs[(a, b)]
        for st in classes:
            if a in st.Coverage.glyphs:
                c1, c2 = st.ClassDef1.classDefs.get(a, 0), st.ClassDef2.classDefs.get(b, 0)
                v = getattr(st.Class1Record[c1].Class2Record[c2].Value1, "XAdvance", 0) or 0
                if v:
                    return v
        return 0
    return k


def text_path(text, style, size, x=0.0, y=0.0, tracking=0.0):
    """SVG path data of the text (baseline at y), and its width."""
    font = TTFont(font_file(style))
    gs, cmap = font.getGlyphSet(), font.getBestCmap()
    s = size / font["head"].unitsPerEm
    kern = kern_table(font)
    names = [cmap[ord(c)] for c in text]
    d, pen_x = [], 0.0
    for i, n in enumerate(names):
        pen = SVGPathPen(gs)
        gs[n].draw(TransformPen(pen, (s, 0, 0, -s, x + pen_x, y)))
        d.append(pen.getCommands())
        pen_x += gs[n].width * s + tracking
        if i + 1 < len(names):
            pen_x += kern(n, names[i + 1]) * s
    return " ".join(d), pen_x - tracking


# ------------------------------------------------- the meshes of the letters

def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def outline(ch):
    """The outline of a letter of Inter Display Black (curves flattened),
    font units, and its advance."""
    f = TTFont(font_file("Black"))
    g = f.getGlyphSet()[f.getBestCmap()[ord(ch)]]
    p = RecordingPen()
    g.draw(p)
    pts, cur = [], None
    for op, a in p.value:
        if op in ("moveTo", "lineTo"):
            cur = np.array(a[0], float)
            pts.append(cur)
        elif op == "curveTo":
            P0, P1, P2, P3 = cur, *(np.array(q, float) for q in a)
            for t in np.linspace(0, 1, 9)[1:]:
                pts.append((1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3)
            cur = pts[-1]
    poly = np.array(pts)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    return poly, g.width


def nseg(length, h):
    """The number of segments of a side (one rule everywhere: conforming)."""
    return max(1, int(round(length / h + 0.2)))


def _split(a, b, h):
    return np.linspace(a, b, nseg(abs(b - a), h) + 1)


def sample_boundary(poly, h, corner=20.0, keep=()):
    """Boundary nodes about every h: the corners of the outline (a turn of
    more than `corner` degrees, and the points `keep`), each chain between
    two of them (flattened curves) resampled uniformly by arc length."""
    n = len(poly)
    turn = []
    for k in range(n):
        u, v = poly[k] - poly[k - 1], poly[(k + 1) % n] - poly[k]
        turn.append(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
    kept = [np.array(p, float) for p in keep]
    corners = [k for k in range(n) if turn[k] > corner
               or any(np.linalg.norm(poly[k] - p) < 1e-6 for p in kept)] or [0]
    out = []
    for a, b in zip(corners, corners[1:] + [corners[0] + n]):
        chain = np.array([poly[k % n] for k in range(a, b + 1)])
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))])
        m = nseg(s[-1], h)
        t = np.arange(m) / m * s[-1]
        out.append(np.c_[np.interp(t, s, chain[:, 0]), np.interp(t, s, chain[:, 1])])
    return np.vstack(out)


def _dist_to_boundary(P, poly):
    d = np.full(len(P), np.inf)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    return d


def _ccw(P, T):
    T = np.array(T)
    a = _cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    T[a < 0] = T[a < 0][:, ::-1]
    return T


def _inside_tris(P, T, poly, h):
    """The triangles inside (centroid and edge midpoints), not flat."""
    path = MPath(poly)
    ok = path.contains_points(P[T].mean(1))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        M = 0.5 * (P[T[:, a]] + P[T[:, b]])
        ok &= path.contains_points(M) | (_dist_to_boundary(M, poly) < 1e-6 * h)
    A = np.abs(_cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))
    return T[ok & (A > 1e-3 * h * h)]


def tri_mesh(poly, h, keep=(), iters=400):
    """Triangles of good shape (as Persson and Strang's DistMesh, the
    boundary nodes fixed): the interior nodes, from a hexagonal lattice,
    pushed apart by springs between neighbours and retriangulated (Delaunay)
    as they move, kept at least 0.4 h from the boundary."""
    path = MPath(poly)
    B = sample_boundary(poly, h, keep=keep)
    lo, hi = poly.min(0), poly.max(0)
    dy = h * np.sqrt(3) / 2
    rows = []
    for j, y in enumerate(np.arange(lo[1], hi[1] + dy, dy)):
        xs = np.arange(lo[0] + (h / 2 if j % 2 else 0), hi[0] + h, h)
        rows.append(np.c_[xs, np.full_like(xs, y)])
    I = np.vstack(rows)
    I = I[path.contains_points(I)]
    I = I[_dist_to_boundary(I, poly) > 0.5 * h]
    nb = len(B)
    P = np.vstack([B, I])
    rng = np.random.default_rng(0)
    old = np.full_like(P, np.inf)
    for _ in range(iters):
        if np.max(np.linalg.norm(P - old, axis=1)) > 0.1 * h:
            old = P.copy()
            T = _inside_tris(P, Delaunay(P + rng.normal(scale=1e-7 * h, size=P.shape)).simplices, poly, h)
            bars = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
        v = P[bars[:, 0]] - P[bars[:, 1]]
        L = np.linalg.norm(v, axis=1)
        F = np.maximum(1.2 * np.sqrt(np.mean(L ** 2)) - L, 0)
        Fv = (F / L)[:, None] * v
        Ft = np.zeros_like(P)
        np.add.at(Ft, bars[:, 0], Fv)
        np.add.at(Ft, bars[:, 1], -Fv)
        Ft[:nb] = 0
        new = P + 0.2 * Ft
        bad = ~path.contains_points(new[nb:]) | (_dist_to_boundary(new[nb:], poly) < 0.4 * h)
        new[nb:][bad] = P[nb:][bad]
        move = np.max(np.linalg.norm((new - P)[nb:], axis=1)) if len(P) > nb else 0
        P = new
        if move < 1e-4 * h:
            break
    T = _inside_tris(P, Delaunay(P + rng.normal(scale=1e-7 * h, size=P.shape)).simplices, poly, h)
    return P, [list(t) for t in _ccw(P, T)]


def quad_mesh(rects, h):
    """Quadrangles: a structured mesh of each rectangle (x0, x1, y0, y1),
    the nodes shared by coordinates (conforming where the sides match)."""
    nodes, cells = {}, []

    def node(x, y):
        return nodes.setdefault((round(x, 6), round(y, 6)), len(nodes))
    for x0, x1, y0, y1 in rects:
        gx, gy = _split(x0, x1, h), _split(y0, y1, h)
        for i in range(len(gx) - 1):
            for j in range(len(gy) - 1):
                cells.append([node(gx[i], gy[j]), node(gx[i + 1], gy[j]),
                              node(gx[i + 1], gy[j + 1]), node(gx[i], gy[j + 1])])
    return np.array(list(nodes), float), cells


def rectangles(poly):
    """A rectilinear letter as rectangles: the bands between the ordinates
    of its outline, each cut into its stem (first column) and the rest."""
    xs, ys = sorted(set(poly[:, 0])), sorted(set(poly[:, 1]))
    path = MPath(poly)
    R = []
    for c, d in zip(ys[:-1], ys[1:]):
        cols = [(a, b) for a, b in zip(xs[:-1], xs[1:]) if path.contains_point(((a + b) / 2, (c + d) / 2))]
        R.append((cols[0][0], cols[0][1], c, d))
        if len(cols) > 1:
            R.append((cols[1][0], cols[-1][1], c, d))
    return R


def straight_m(poly, stems=((488, 576, 1047), (1406, 576, 1065))):
    """The outline of M with the inner sides of its stems vertical up to the
    tips of the notches (in Inter, a slight curve: 13 units of 1490)."""
    P = poly.copy()
    for x, y0, y1 in stems:
        sel = (np.abs(P[:, 0] - x) < 20) & (P[:, 1] > y0) & (P[:, 1] <= y1 + 1)
        P[sel, 0] = x
    return P


def _area(V):
    return 0.5 * np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1])


def mixed_m(P, h, xl=488.0, xr=1406.0):
    """M: structured quadrangles in its stems, triangles in the V between
    them, the nodes of the sides shared."""
    x0, x1, top = P[:, 0].min(), P[:, 0].max(), P[:, 1].max()
    yl = P[np.isclose(P[:, 0], xl), 1].max()           # the tips of the notches
    yr = P[np.isclose(P[:, 0], xr), 1].max()
    Pq, Cq = quad_mesh([(x0, xl, 0, yl), (x0, xl, yl, top), (xr, x1, 0, yr), (xr, x1, yr, top)], h)
    inner = np.array([p for p in P if xl - 1e-9 <= p[0] <= xr + 1e-9
                      and not (np.isclose(p[0], xl) and p[1] < yl - 1e-9)
                      and not (np.isclose(p[0], xr) and p[1] < yr - 1e-9)])
    inner = np.roll(inner, -int(np.argmin(np.hypot(inner[:, 0] - xl, inner[:, 1] - yl))), axis=0)
    kr = int(np.argmin(np.hypot(inner[:, 0] - xr, inner[:, 1] - yr)))
    V = np.vstack([inner[:kr + 1], [[xr, top]], inner[kr + 1:], [[xl, top]]])
    if _area(V) < 0:
        V = V[::-1]
    Pt, Ct = tri_mesh(V, h, keep=[(xl, yl), (xl, top), (xr, yr), (xr, top)])
    nodes, cells = {}, []
    for Pm, Cm in ((Pq, Cq), (Pt, Ct)):
        for c in Cm:
            cells.append([nodes.setdefault((round(Pm[k, 0], 4), round(Pm[k, 1], 4)), len(nodes)) for k in c])
    return np.array(list(nodes), float), cells


def letter_meshes(h=H_MESH):
    """{letter: (nodes, cells, outline, advance)}, font units."""
    pF, wF = outline("F")
    pE, wE = outline("E")
    pM, wM = outline("M")
    pM = straight_m(pM)
    return {"F": (*quad_mesh(rectangles(pF), h), pF, wF),
            "E": (*tri_mesh(pE, 0.95 * h), pE, wE),
            "M": (*mixed_m(pM, h), pM, wM)}


def quality(P, cells):
    """Of the triangles: 2 r_inscribed / r_circumscribed (1: equilateral)."""
    q = []
    for c in (c for c in cells if len(c) == 3):
        a, b, d = (np.linalg.norm(P[c[i]] - P[c[(i + 1) % 3]]) for i in range(3))
        s = (a + b + d) / 2
        A = max(s * (s - a) * (s - b) * (s - d), 0) ** 0.5
        q.append(0.0 if A == 0 else 8 * A * A / (s * a * b * d))
    return np.array(q)


def check_meshes(meshes):
    """Conforming (each edge in at most two cells, the free edges as long as
    the outline) and of good shape."""
    for ch, (P, C, poly, _) in meshes.items():
        E = {}
        for c in C:
            for i in range(len(c)):
                e = tuple(sorted((c[i], c[(i + 1) % len(c)])))
                E[e] = E.get(e, 0) + 1
        free = sum(np.linalg.norm(P[a] - P[b]) for (a, b), n in E.items() if n == 1)
        perimeter = sum(np.linalg.norm(poly[i] - poly[i - 1]) for i in range(len(poly)))
        q = quality(P, C)
        assert max(E.values()) <= 2 and abs(free - perimeter) < 1e-3 * perimeter, ch
        assert not len(q) or q.min() > 0.6, (ch, q.min())
        print(f"{ch}: {len(C)} elements ({sum(len(c) == 4 for c in C)} quadrangles), "
              f"triangles of quality >= {q.min():.2f}" if len(q) else f"{ch}: {len(C)} quadrangles")


# ----------------------------------------------------------------- the logo

def fem_svg(x, base, size, meshes, tracking=2.0, edge_opacity=0.6):
    """The meshed letters FEM colored (flat, viridis) by a smooth field over
    the word; their width."""
    import matplotlib
    cmap = matplotlib.colormaps["viridis"]
    font = TTFont(font_file("Black"))
    s = size / font["head"].unitsPerEm
    kern = kern_table(font)
    pen, placed = 0.0, []
    for i, ch in enumerate("FEM"):
        P, C, poly, adv = meshes[ch]
        placed.append((np.c_[x + (pen + P[:, 0]) * s, base - P[:, 1] * s], C,
                       np.c_[x + (pen + poly[:, 0]) * s, base - poly[:, 1] * s]))
        pen += adv + tracking / s + (kern(ch, "FEM"[i + 1]) if i < 2 else 0)
    allX = np.vstack([p[0] for p in placed])
    lo, hi = allX.min(0), allX.max(0)

    def field(Z):                      # yellow at the foot of F and the top of M
        u, v = ((Z - lo) / (hi - lo)).T
        return np.exp(-((u - 0.02) ** 2 / 0.10 + (v - 1.0) ** 2 / 0.7)) \
            + np.exp(-((u - 1.0) ** 2 / 0.07 + v ** 2 / 0.8)) \
            + 0.35 * np.exp(-((u - 0.5) ** 2 / 0.03 + (v - 0.5) ** 2 / 0.3))
    centers = [np.array([X[c].mean(0) for c in C]) for X, C, _ in placed]
    vals = np.concatenate([field(Z) for Z in centers])
    vmin, vmax = vals.min(), vals.max()
    out = []
    for (X, C, O), Z in zip(placed, centers):
        for c, f in zip(C, (field(Z) - vmin) / (vmax - vmin)):
            r, g, b, _ = cmap(0.04 + 0.92 * f)
            pts = " ".join(f"{X[k, 0]:.2f},{X[k, 1]:.2f}" for k in c)
            out.append(f'<polygon points="{pts}" fill="#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" '
                       f'stroke="#ffffff" stroke-opacity="{edge_opacity}" stroke-width="0.9" '
                       'stroke-linejoin="round"/>')
        pts = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in O)
        out.append(f'<polygon points="{pts}" fill="none" stroke="{NAVY}" stroke-width="2.2"/>')
    return "\n".join(out), (pen - tracking / s) * s


def logo_svg(meshes, tagline=True, edge_opacity=0.6, background=None):
    """The logo (SVG): the tripod, "Get", the meshed "FEM", and under them
    "FINITE ELEMENT LIBRARY"."""
    size, base, x0 = 132, 140, 214
    get_d, get_w = text_path("Get", "Medium", size, x0, base, tracking=-2)
    fem_x = x0 + get_w + 6
    fem, fem_w = fem_svg(fem_x, base, size, meshes, edge_opacity=edge_opacity)
    W, H = int(fem_x + fem_w + 24), (200 if tagline else 168)
    icon = "data:image/png;base64," + base64.b64encode(ICON.read_bytes()).decode()
    cap = size * 0.727
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
             "<title>GetFEM</title>"]
    if background:
        parts.append(f'<rect width="{W}" height="{H}" fill="{background}"/>')
    parts += [f'<image x="14" y="{base - cap / 2 - 92:.2f}" width="184" height="184" xlink:href="{icon}"/>',
              f'<path d="{get_d}" fill="{NAVY}"/>', fem]
    if tagline:
        _, tw = text_path("FINITE ELEMENT LIBRARY", "Medium", 19, tracking=4.2)
        tx = fem_x + fem_w - tw
        tg, _ = text_path("FINITE ELEMENT LIBRARY", "Medium", 19, tx, base + 38, tracking=4.2)
        parts += [f'<path d="{tg}" fill="{TEAL}"/>',
                  f'<rect x="{x0 + 4}" y="{base + 24}" width="{tx - x0 - 20:.2f}" height="3" rx="1.5" '
                  f'fill="{TEAL}" opacity="0.55"/>']
    parts.append("</svg>\n")
    return "\n".join(parts)


def render(svg, width, out, rgb=False):
    """The SVG as a PNG of this width (rsvg-convert); on white when rgb."""
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(svg)
    png = subprocess.run(["rsvg-convert", "-w", str(width), f.name], capture_output=True, check=True).stdout
    Path(f.name).unlink()
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    if rgb:
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[3])
        im = bg
    im.save(out, optimize=True)
    print(out.relative_to(ROOT), im.size)


def main():
    from PIL import Image
    meshes = letter_meshes()
    check_meshes(meshes)
    static = ROOT / "doc" / "sphinx" / "source" / ".static"
    scilab = ROOT / "interface" / "src" / "scilab" / "help" / "latex"
    full = logo_svg(meshes)
    (static / "logo_getfem.svg").write_text(full)
    print((static / "logo_getfem.svg").relative_to(ROOT))
    small = logo_svg(meshes, tagline=False, edge_opacity=0.35)       # the mesh lighter at small sizes
    render(full, 700, static / "logogetfem.png")                       # index page, LaTeX
    render(small, 400, static / "logo_getfem_small.png")              # sidebar (shown 200 wide)
    render(full, 700, scilab / "logogetfem.png")
    render(full, 700, scilab / "logogetfemwhitebg.png", rgb=True)
    render(small, 200, scilab / "logo_getfem_small.png")
    render(logo_svg(meshes, tagline=False), 440, LAB / "getfem_lab" / "app" / "resources" / "logo_getfem.png")
    icon = Image.open(ICON)
    icon.resize((48, 48), Image.LANCZOS).save(static / "icon.png", optimize=True)
    icon.save(static / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    print((static / "icon.png").relative_to(ROOT), (static / "favicon.ico").relative_to(ROOT))


if __name__ == "__main__":
    main()
