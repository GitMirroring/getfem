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
"""  Test of the 'distance' and 'project' of a mesher object: the nodes of
     the boundary of a refined mesh projected on the geometry (a square
     with a hole, a ball), the area and volume then closer to the exact ones.
"""
import numpy as np

import getfem as gf

gf.util_trace_level(0)

mo = gf.MesherObject('set minus', gf.MesherObject('rectangle', [0, 0], [1, 1]),
                     gf.MesherObject('ball', [0.5, 0.5], 0.2))
d = mo.distance(np.array([[0.5, 0.5, 0.0, 0.5], [0.5, 0.71, 0.0, 1.5]]))
assert np.allclose(d, [0.2, -0.01, 0.0, 0.5])
Q = mo.project(np.array([[0.69, 0.0, 0.5], [0.29, 0.5, 1.2]]))   # on the hole, on a side, on the top
u = np.array([0.19, -0.21]) / np.hypot(0.19, 0.21)
assert np.allclose(Q, [[0.5 + 0.2*u[0], 0.0, 0.5], [0.5 + 0.2*u[1], 0.5, 1.0]])


def measure(m):
  mim = gf.MeshIm(m, gf.Integ('IM_TRIANGLE(6)' if m.dim() == 2 else 'IM_TETRAHEDRON(5)'))
  mf = gf.MeshFem(m, 1)
  mf.set_fem(gf.Fem('FEM_PK(%d,1)' % m.dim()))
  return gf.asm_generic(mim, 0, '1', -1, 'u', 1, mf, np.zeros(mf.nbdof()))


def refined_and_projected(mo, h):
  m = gf.Mesh('generate', mo, h, 1)
  m.refine()
  before = measure(m)
  P = m.pts()
  B = m.pid_in_faces(m.outer_faces())
  P[:, B] = mo.project(P[:, B])
  m.set_pts(P)
  assert np.abs(mo.distance(m.pts()[:, B])).max() < 1e-10
  return before, measure(m)


exact = 1 - np.pi * 0.2**2
before, after = refined_and_projected(mo, 0.1)
print("square with a hole: area %g refined, %g projected (exact %g)" % (before, after, exact))
assert abs(after - exact) < 0.3 * abs(before - exact)

ball = gf.MesherObject('ball', [0, 0, 0], 1)
before, after = refined_and_projected(ball, 0.4)
print("ball: volume %g refined, %g projected (exact %g)" % (before, after, 4 * np.pi / 3))
assert abs(after - 4 * np.pi / 3) < 0.5 * abs(before - 4 * np.pi / 3)
