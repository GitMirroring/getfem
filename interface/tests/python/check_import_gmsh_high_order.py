#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python GetFEM interface
#
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
# along  with  this program.  If not, see https://www.gnu.org/licenses/.
#
############################################################################
"""  Test of the import of high-order (2 and 3) Gmsh elements: quadrangles,
     tetrahedra, hexahedra and prisms. Curved meshes built by GetFEM are
     written in the Gmsh format (2.2) with the nodes in the order of Gmsh,
     imported again, and their points and volume compared.

     The node orderings of Gmsh: the vertices, the nodes of the edges (in
     the order of the edges of Gmsh, each from its first vertex), of the
     faces (in the order of the faces of Gmsh, as an element of lower
     order), then of the interior (recursively), as described by the
     documentation of Gmsh and implemented in mesh/gmsh.cpp of MFEM.
"""
import os
import tempfile

import numpy as np

import getfem as gf

gf.util_trace_level(0)


def quad(i, j, r):
    if r == 0:
        return 0
    corners = {(0, 0): 0, (r, 0): 1, (r, r): 2, (0, r): 3}
    if (i, j) in corners:
        return corners[(i, j)]
    if j == 0:
        return 4 + (i - 1)
    if i == r:
        return 4 + (r - 1) + (j - 1)
    if j == r:
        return 4 + 2 * (r - 1) + (r - 1 - i)
    if i == 0:
        return 4 + 3 * (r - 1) + (r - 1 - j)
    return 4 * r + quad(i - 1, j - 1, r - 2)


def hexa(i, j, k, r):
    if r == 0:
        return 0
    corners = {(0, 0, 0): 0, (r, 0, 0): 1, (r, r, 0): 2, (0, r, 0): 3,
               (0, 0, r): 4, (r, 0, r): 5, (r, r, r): 6, (0, r, r): 7}
    if (i, j, k) in corners:
        return corners[(i, j, k)]
    e = r - 1
    if j in (0, r) and k in (0, r):      # the edges along x
        return 8 + {(0, 0): i - 1, (r, 0): 6 * r - 6 - i,
                    (0, r): 8 * r - 9 + i, (r, r): 12 * r - 12 - i}[(j, k)]
    if i in (0, r) and k in (0, r):      # along y
        return 8 + {(0, 0): r - 2 + j, (r, 0): 3 * r - 4 + j,
                    (0, r): 9 * r - 10 + j, (r, r): 10 * r - 11 + j}[(i, k)]
    if i in (0, r) and j in (0, r):      # along z
        return 8 + {(0, 0): 2 * r - 3 + k, (r, 0): 4 * r - 5 + k,
                    (0, r): 7 * r - 8 + k, (r, r): 6 * r - 7 + k}[(i, j)]
    f0 = 8 + 12 * e
    if k == 0:
        return f0 + quad(j - 1, i - 1, r - 2)
    if j == 0:
        return f0 + e * e + quad(i - 1, k - 1, r - 2)
    if i == 0:
        return f0 + 2 * e * e + quad(k - 1, j - 1, r - 2)
    if i == r:
        return f0 + 3 * e * e + quad(j - 1, k - 1, r - 2)
    if j == r:
        return f0 + 4 * e * e + quad(r - i - 1, k - 1, r - 2)
    if k == r:
        return f0 + 5 * e * e + quad(i - 1, j - 1, r - 2)
    return f0 + 6 * e * e + hexa(i - 1, j - 1, k - 1, r - 2)


def tet(i, j, k, r):                     # orders <= 3
    l = r - i - j - k
    corners = {(0, 0, 0): 0, (r, 0, 0): 1, (0, r, 0): 2, (0, 0, r): 3}
    if (i, j, k) in corners:
        return corners[(i, j, k)]
    e = r - 1
    if [i, j, k, l].count(0) == 2:       # the edges
        if j == 0 and k == 0:
            return 4 + (i - 1)                       # 0-1, from 0
        if k == 0 and l == 0:
            return 4 + e + (j - 1)                   # 1-2, from 1
        if i == 0 and k == 0:
            return 4 + 2 * e + (r - j - 1)           # 2-0, from 2
        if i == 0 and j == 0:
            return 4 + 3 * e + (r - k - 1)           # 3-0, from 3
        if i == 0 and l == 0:
            return 4 + 4 * e + (r - k - 1)           # 3-2, from 3
        return 4 + 5 * e + (r - k - 1)               # 3-1, from 3
    h = (r - 1) * (r - 2) // 2                       # one node per face at order 3
    f0 = 4 + 6 * e
    return f0 + h * [k == 0, j == 0, i == 0, l == 0].index(True)


def prism(i, j, k, r):                   # orders <= 3
    l = r - i - j
    corners = {(0, 0, 0): 0, (r, 0, 0): 1, (0, r, 0): 2, (0, 0, r): 3, (r, 0, r): 4, (0, r, r): 5}
    if (i, j, k) in corners:
        return corners[(i, j, k)]
    e = r - 1
    if (i, j) in ((0, 0), (r, 0), (0, r)):           # the vertical edges
        return 6 + {(0, 0): 2, (r, 0): 4, (0, r): 5}[(i, j)] * e + (k - 1)
    if k in (0, r) and [i, j, l].count(0) == 1:      # the edges of the triangles
        top = k == r
        if j == 0:
            return 6 + (6 if top else 0) * e + (i - 1)
        if i == 0:
            return 6 + (7 if top else 1) * e + (j - 1)
        return 6 + (8 if top else 3) * e + (j - 1)
    h = (r - 1) * (r - 2) // 2
    f0 = 6 + 9 * e
    if k == 0:
        return f0
    if k == r:
        return f0 + h
    q0 = f0 + 2 * h
    if j == 0:
        return q0 + quad(i - 1, k - 1, r - 2)
    if i == 0:
        return q0 + e * e + quad(k - 1, j - 1, r - 2)
    if l == 0:
        return q0 + 2 * e * e + quad(j - 1, k - 1, r - 2)
    return q0 + 3 * e * e + (0 if k == 1 else 1)


# (geometric transformation, Gmsh type, ordering, dimension, order)
CASES = [("GT_QK(2,2)", 10, quad, 2, 2), ("GT_QK(2,3)", 36, quad, 2, 3),
         ("GT_PK(3,2)", 11, tet, 3, 2), ("GT_PK(3,3)", 29, tet, 3, 3),
         ("GT_QK(3,2)", 12, hexa, 3, 2), ("GT_QK(3,3)", 92, hexa, 3, 3),
         ("GT_PRISM(3,2)", 13, prism, 3, 2), ("GT_PRISM(3,3)", 90, prism, 3, 3)]

for gt, gtype, order_of, dim, r in CASES:
    m = gf.Mesh('import', 'structured',
                'GT="%s";SIZES=[%s];NSUBDIV=[%s]' % (gt, ",".join(["1"] * dim), ",".join(["2"] * dim)))
    # curved: the last coordinate times 1 + 0.3 x^(r-1), exact with this order:
    # the volume is 1 + 0.3 / r
    P = m.pts()
    P[dim - 1] *= 1 + 0.3 * P[0] ** (r - 1)
    m.set_pts(P)
    ref = gf.GeoTrans(gt).pts()                     # GetFEM's reference nodes, in its order
    idx = np.rint(ref * r).astype(int)
    perm = [order_of(*idx[:, g], r) for g in range(idx.shape[1])]
    assert sorted(perm) == list(range(len(perm))), gt
    with tempfile.TemporaryDirectory() as tmp:
        name = os.path.join(tmp, "m.msh")
        with open(name, "w") as f:
            f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n%d\n" % m.nbpts())
            for p in range(m.nbpts()):
                x = list(P[:, p]) + [0.0] * (3 - dim)
                f.write("%d %r %r %r\n" % (p + 1, float(x[0]), float(x[1]), float(x[2])))
            f.write("$EndNodes\n$Elements\n%d\n" % m.nbcvs())
            for n, cv in enumerate(m.cvid()):
                pid = m.pid_from_cvid(cv)[0]
                gmsh_nodes = [0] * len(pid)
                for g, q in enumerate(pid):             # GetFEM's node g is Gmsh's node perm[g]
                    gmsh_nodes[perm[g]] = q + 1
                f.write("%d %d 2 1 1 %s\n" % (n + 1, gtype, " ".join(map(str, gmsh_nodes))))
            f.write("$EndElements\n")
        m2 = gf.Mesh('import', 'gmsh', name)
    assert m2.nbcvs() == m.nbcvs(), gt
    volume = gf.asm('generic', gf.MeshIm(m2, 8), 0, '1', -1)
    assert abs(volume - (1 + 0.3 / r)) < 1e-10, (gt, volume)

    def element_volumes(mesh):              # (sorted points, volume) of each element
        out = []
        for cv in mesh.cvid():
            pts = mesh.pts(mesh.pid_from_cvid(cv)[0])[:dim]
            key = tuple(sorted(tuple(np.round(x, 10)) for x in pts.T))
            mesh.set_region(100 + cv, np.atleast_2d(cv))
            out.append((key, gf.asm('generic', gf.MeshIm(mesh, 8), 0, '1', 100 + cv)))
        return sorted(out)
    # the same elements (their points, maybe in another symmetric order) with the same volumes
    for (k1, v1), (k2, v2) in zip(element_volumes(m), element_volumes(m2)):
        assert k1 == k2 and abs(v1 - v2) < 1e-12, (gt, v1, v2)
    print("%-14s Gmsh type %2d: %d elements, volume %.12f" % (gt, gtype, m2.nbcvs(), volume))
