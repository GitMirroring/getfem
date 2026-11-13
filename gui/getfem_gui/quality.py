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

"""The quality of the elements of a mesh, from 0 (degenerate) to 1 (the
regular element of its kind), computed on their corners (the cells of a
VTK file: VTK numbering of the points; quadratic cells by their corners).

- triangles and tetrahedra: the radius ratio, d r / R (r the radius of the
  inscribed circle or sphere, R that of the circumscribed one, d the
  dimension): 1 for the equilateral triangle and the regular tetrahedron;
- quadrangles, hexahedra, prisms, pyramids: the smallest scaled Jacobian at
  the corners (the volume spanned by the unit vectors of the edges from the
  corner), divided by its value on the regular element (a square, a cube, a
  prism of equilateral triangles and square sides, a pyramid of
  equilateral triangles): 1 for it, 0 or less (shown as 0) for a flat or
  folded element;
- segments: 1 (no measure).

Numpy only (no GUI toolkit): the view (app/scene.py) colors the elements
with it and shows the worst ones.
"""

from __future__ import annotations

import numpy as np

# VTK cell types: (number of corners, kind)
TRIANGLES = (5, 22, 34)            # triangle, quadratic, biquadratic
QUADS = (9, 23, 28)                # quad, quadratic (8 nodes), biquadratic (9)
TETS = (10, 24)
HEXES = (12, 25, 29)
WEDGES = (13, 26, 32)
PYRAMIDS = (14, 27)
SEGMENTS = (3, 21)
CORNERS = {**{t: 3 for t in TRIANGLES}, **{t: 4 for t in QUADS}, **{t: 4 for t in TETS},
           **{t: 8 for t in HEXES}, **{t: 6 for t in WEDGES}, **{t: 5 for t in PYRAMIDS},
           **{t: 2 for t in SEGMENTS}}

_S3 = np.sqrt(3.0) / 2
# the regular elements, in the VTK numbering, and the edges at their corners
_REGULAR = {
    "quad": (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float),
             [(0, 1, 3), (1, 2, 0), (2, 3, 1), (3, 0, 2)]),
    "hex": (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float),
            [(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
             (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)]),
    "wedge": (np.array([[0, 0, 0], [1, 0, 0], [0.5, _S3, 0], [0, 0, 1], [1, 0, 1], [0.5, _S3, 1]], float),
              [(0, 1, 2, 3), (1, 2, 0, 4), (2, 0, 1, 5), (3, 4, 5, 0), (4, 5, 3, 1), (5, 3, 4, 2)]),
    "pyramid": (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, np.sqrt(0.5)]], float),
                [(0, 1, 3, 4), (1, 2, 0, 4), (2, 3, 1, 4), (3, 0, 2, 4)]),
}


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def _corner_values(X: np.ndarray, corners) -> np.ndarray:
    """The scaled Jacobian at each corner (cells x corners): the
    determinant of the unit vectors of its edges; for a 2D element (two
    edges), the cross product along the mean normal of the element."""
    out = []
    if len(corners[0]) == 3:                        # quadrangles, possibly in 3D
        crosses = [np.cross(_unit(X[:, b] - X[:, a]), _unit(X[:, c] - X[:, a])) for a, b, c in corners]
        normal = _unit(sum(crosses))
        return np.stack([np.sum(c * normal, axis=1) for c in crosses], axis=1)
    for a, b, c, d in corners:
        out.append(np.linalg.det(np.stack([_unit(X[:, b] - X[:, a]), _unit(X[:, c] - X[:, a]),
                                           _unit(X[:, d] - X[:, a])], axis=1)))
    return np.stack(out, axis=1)


def _oriented_corners():
    """The edges at the corners of each kind, in the order that makes the
    regular element positive, and the scaled Jacobian of the latter."""
    out = {}
    for kind, (X, corners) in _REGULAR.items():
        v = _corner_values(X[None], corners)[0]
        if len(corners[0]) == 4:
            corners = [c if s > 0 else (c[0], c[2], c[1], c[3]) for c, s in zip(corners, v)]
        out[kind] = (corners, float(np.min(np.abs(v))))
    return out


_CORNERS = _oriented_corners()


def _jacobian_quality(X: np.ndarray, kind: str) -> np.ndarray:
    corners, ideal = _CORNERS[kind]
    v = _corner_values(X, corners)
    if len(corners[0]) == 4:
        # the orientation of the numbering of the file: that of most of the corners
        v = v * np.where(np.sum(v, axis=1) < 0, -1.0, 1.0)[:, None]
    return np.clip(np.min(v, axis=1) / ideal, 0.0, 1.0)


def _triangle_quality(X: np.ndarray) -> np.ndarray:
    """2 r / R = 8 A^2 / (s a b c) (A the area, s the half perimeter)."""
    a = np.linalg.norm(X[:, 1] - X[:, 2], axis=1)
    b = np.linalg.norm(X[:, 2] - X[:, 0], axis=1)
    c = np.linalg.norm(X[:, 0] - X[:, 1], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1)
    den = 0.5 * (a + b + c) * a * b * c
    return np.clip(np.divide(8 * area ** 2, den, out=np.zeros_like(den), where=den > 0), 0.0, 1.0)


def _tetra_quality(X: np.ndarray) -> np.ndarray:
    """3 r / R, r = 3 V / S (S the area of the faces), 24 V R = sqrt of the
    product of the sums of the products of opposite edges (Crelle)."""
    p0, p1, p2, p3 = (X[:, i] for i in range(4))
    vol = np.abs(np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6
    area = sum(0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=1)
               for p, q, r in ((p0, p1, p2), (p0, p1, p3), (p0, p2, p3), (p1, p2, p3)))
    ln = lambda p, q: np.linalg.norm(q - p, axis=1)          # noqa: E731
    aa, bb, cc = ln(p0, p1) * ln(p2, p3), ln(p0, p2) * ln(p1, p3), ln(p0, p3) * ln(p1, p2)
    prod = (aa + bb + cc) * (aa + bb - cc) * (aa - bb + cc) * (-aa + bb + cc)
    den = area * np.sqrt(np.maximum(prod, 0.0))
    # 3 r / R = 3 (3 V / S) / (sqrt(prod) / (24 V)) = 216 V^2 / (S sqrt(prod))
    return np.clip(np.divide(216 * vol ** 2, den, out=np.zeros_like(den), where=den > 0), 0.0, 1.0)


def cell_quality(points: np.ndarray, celltypes, cells) -> np.ndarray:
    """The quality of each cell (VTK types, the point indices of each cell
    in the VTK numbering): from 0 to 1, NaN for a kind of cell not
    measured."""
    pts = np.asarray(points, float)
    if pts.ndim == 2 and pts.shape[1] < 3:
        pts = np.hstack([pts, np.zeros((len(pts), 3 - pts.shape[1]))])
    types = np.asarray(celltypes, int)
    q = np.full(len(types), np.nan)
    for group, fn in ((TRIANGLES, _triangle_quality), (TETS, _tetra_quality),
                      (QUADS, lambda X: _jacobian_quality(X, "quad")),
                      (HEXES, lambda X: _jacobian_quality(X, "hex")),
                      (WEDGES, lambda X: _jacobian_quality(X, "wedge")),
                      (PYRAMIDS, lambda X: _jacobian_quality(X, "pyramid"))):
        ids = np.flatnonzero(np.isin(types, group))
        if len(ids):
            k = CORNERS[group[0]]
            X = pts[np.array([np.asarray(cells[i])[:k] for i in ids], int)]
            q[ids] = fn(X)
    q[np.isin(types, SEGMENTS)] = 1.0
    return q


def summary(q: np.ndarray, threshold: float) -> str:
    """The minimum and the mean of the qualities, and how many elements are
    below the threshold."""
    v = q[np.isfinite(q)]
    if not len(v):
        return "quality: no element measured"
    bad = int(np.sum(v < threshold))
    return (f"quality min {v.min():.3g}, mean {v.mean():.3g}; {bad} of {len(v)} element"
            f"{'s' if len(v) > 1 else ''} below {threshold:g}")
