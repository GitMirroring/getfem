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
"""  Test of the local sizes of the GetFEM mesher (Mesh('generate', ...,
     size_points, sizes, growth)): a square with a hole, finer around the
     hole; the elements there have the size asked for, those far from it
     the global size, and their quality stays good.
"""
import numpy as np

import getfem as gf

gf.util_trace_level(0)

h, s, growth = 0.1, 0.01, 1.3
mo = gf.MesherObject('set minus', gf.MesherObject('rectangle', [0, 0], [1, 1]),
                     gf.MesherObject('ball', [0.5, 0.5], 0.15))
th = np.linspace(0, 2*np.pi, 200, endpoint=False)
P = np.array([0.5 + 0.15*np.cos(th), 0.5 + 0.15*np.sin(th)])     # the hole, sampled
m = gf.Mesh('generate', mo, h, 1, np.zeros((2, 0)), P, np.full(P.shape[1], s), growth)
uniform = gf.Mesh('generate', mo, h, 1)


def edge_lengths(m):
  """The middles and lengths of the edges of the mesh."""
  X = m.pts()
  E = set()
  for cv in m.cvid():
    p = m.pid_from_cvid(cv)[0]
    E |= {tuple(sorted((p[i], p[j]))) for i in range(3) for j in range(i + 1, 3)}
  E = np.array(sorted(E)).T
  return 0.5*(X[:, E[0]] + X[:, E[1]]), np.linalg.norm(X[:, E[0]] - X[:, E[1]], axis=0)


C, L = edge_lengths(m)
d = np.abs(np.hypot(C[0] - 0.5, C[1] - 0.5) - 0.15)        # distance to the hole
near, far = L[d < 0.005], L[d > (h - s)/(growth - 1) + h]
print("edges near the hole: %d, mean length %g; far from it: %d, mean %g"
      % (len(near), near.mean(), len(far), far.mean()))
assert 0.6*s < near.mean() < 1.6*s
assert 0.75*h < far.mean() < 1.25*h
assert m.nbcvs() > 2*uniform.nbcvs()
q = m.quality()
print("quality: min %g, mean %g" % (q.min(), q.mean()))
assert q.min() > 0.3 and q.mean() > 0.7

# a size not smaller than the global one: the same as without sizes
m2 = gf.Mesh('generate', mo, h, 1, np.zeros((2, 0)), P, np.full(P.shape[1], 2*h))
assert abs(m2.nbcvs() - uniform.nbcvs()) <= 0.1*uniform.nbcvs()
