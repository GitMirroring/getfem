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
"""  Test of gf.Mesh('faces'): the boundary of 3D meshes (tetrahedra of
     degree 1 and 2, hexahedra, prisms) as surface meshes; their area, the
     points shared, the faces given in an order kept.
"""
import numpy as np

import getfem as gf

gf.util_trace_level(0)


def area(m):
  mim = gf.MeshIm(m, 0)
  gts, cv2gt = m.geotrans()
  gts = gts if isinstance(gts, list) else [gts]
  for i, gt in enumerate(gts):
    integ = 'IM_TRIANGLE(6)' if 'PK' in gt.char() else 'IM_GAUSS_PARALLELEPIPED(2,6)'
    mim.set_integ(gf.Integ(integ), m.cvid()[np.asarray(cv2gt).ravel() == i])
  return gf.asm_generic(mim, 0, '1', -1)


def gt_names(m):
  gts = m.geotrans()[0]
  return [g.char() for g in (gts if isinstance(gts, list) else [gts])]


# a cube of tetrahedra, degree 1: 12 triangles on its boundary, area 6
m = gf.Mesh('regular simplices', np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3))
outer = m.outer_faces()
s = gf.Mesh('faces', m)
assert s.dim() == 3 and s.nbcvs() == outer.shape[1] == 6 * 2 * 4
assert s.nbpts() == 26                                           # the boundary points only, shared
assert abs(area(s) - 6) < 1e-12
assert gt_names(s) == ['GT_PK(2,1)']
# faces given: in their order (the i-th element is the i-th face)
top = m.outer_faces_with_direction([0, 0, 1], 0.1)
s2 = gf.Mesh('faces', m, top[:, ::-1])
assert s2.nbcvs() == top.shape[1] and abs(area(s2) - 1) < 1e-12
for i in range(top.shape[1]):
  P = s2.pts(s2.pid_from_cvid(i)[0])
  Q = m.pts(m.pid_in_faces(top[:, ::-1][:, i:i + 1]))
  assert np.allclose(np.sort(P, axis=1), np.sort(Q, axis=1))

# degree 2 (a ball): curved faces, closer to the sphere
mo = gf.MesherObject('ball', [0, 0, 0], 1.0)
for K in (1, 2):
  mb = gf.Mesh('generate', mo, 0.4, K)
  sb = gf.Mesh('faces', mb)
  assert gt_names(sb) == ['GT_PK(2,%d)' % K]
  A = area(sb)
  if K == 1:
    A1 = A
assert abs(A - 4 * np.pi) < abs(A1 - 4 * np.pi) and abs(A - 4 * np.pi) < 0.02

# hexahedra (Q1, Q2 and its incomplete version), prisms
for gt in ('GT_QK(3,1)', 'GT_QK(3,2)', 'GT_Q2_INCOMPLETE(3)', 'GT_PRISM(3,1)'):
  if gt == 'GT_PRISM(3,1)':
    mq = gf.Mesh('prismatic', gf.Mesh('triangles grid', np.linspace(0, 2, 3), np.linspace(0, 1, 2)), 2)
    expected = 2 * (2 + 2 + 1)
  else:
    mq = gf.Mesh('cartesian', np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    if gt != 'GT_QK(3,1)':
      mq = gf.Mesh('empty', 3)
      g = gf.GeoTrans(gt)
      mq.add_convex(g, g.pts())
    expected = 6
  sq = gf.Mesh('faces', mq)
  assert abs(area(sq) - expected) < 1e-12, (gt, area(sq))

print('mesh faces: ok')
