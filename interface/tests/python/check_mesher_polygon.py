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
"""  Test of the 'polygon' mesher object: measures of the generated meshes
     (non convex polygons, one of them extruded).
"""
import numpy as np

import getfem as gf

gf.util_trace_level(0)


def measures(m, integ):
  mim = gf.MeshIm(m, gf.Integ(integ))
  m.set_region(1, m.outer_faces())
  return gf.asm('generic', mim, 0, '1', -1), gf.asm('generic', mim, 0, '1', 1)


# L shaped domain, vertices given clockwise
L = np.array([[0, 0], [0, 1], [0.5, 1], [0.5, 0.5], [1, 0.5], [1, 0]]).T
m = gf.Mesh('generate', gf.MesherObject('polygon', L), 0.1, 1)
area, perimeter = measures(m, 'IM_TRIANGLE(6)')
print('L: area %g (exact 0.75), perimeter %g (exact 4)' % (area, perimeter))
assert abs(area - 0.75) < 1e-8 and abs(perimeter - 4) < 1e-8

# Star with ten vertices (counterclockwise)
a = 2*np.pi*np.arange(10)/10
r = np.where(np.arange(10) % 2, 0.45, 1.)
S = np.array([r*np.cos(a), r*np.sin(a)])
m = gf.Mesh('generate', gf.MesherObject('polygon', S), 0.1, 1)
area, perimeter = measures(m, 'IM_TRIANGLE(6)')
exact_area = 5*np.sin(2*np.pi/10)*0.45
exact_perimeter = 10*np.linalg.norm(S[:, 1] - S[:, 0])
print('star: area %g (exact %g), perimeter %g (exact %g)'
      % (area, exact_area, perimeter, exact_perimeter))
assert abs(area - exact_area) < 1e-8 and abs(perimeter - exact_perimeter) < 1e-8

# Extruded U shaped profile
U = np.array([[0, 0], [1, 0], [1, 1], [0.6, 1], [0.6, 0.3], [0.4, 0.3],
              [0.4, 1], [0, 1]]).T
m = gf.Mesh('generate', gf.MesherObject('extrusion', gf.MesherObject('polygon', U),
                                        0, 0.3), 0.12, 1)
vol, area = measures(m, 'IM_TETRAHEDRON(5)')
exact = 2*0.86 + 0.3*5.4
print('extruded U: volume %g (exact 0.258), area %g (exact %g)' % (vol, area, exact))
assert abs(vol - 0.258) < 1e-8 and abs(area - exact) < 1e-8

# Curved sides: cubic Bezier curves (a column of NaN: a straight side).
# A unit square whose right side is an S and whose top side bulges.
def bezier_outline(V, C, n=4000):
  """Points along the boundary (exact area and length by Green's formula
  and the sum of the chords, for n large)."""
  pts = []
  for j in range(V.shape[1]):
    A, B = V[:, j], V[:, (j+1) % V.shape[1]]
    if np.isnan(C[0, j]):
      pts.append(A[None, :])
    else:
      t = np.linspace(0, 1, n, endpoint=False)[:, None]; s = 1 - t
      pts.append(s**3*A + 3*s*s*t*C[0:2, j] + 3*s*t*t*C[2:4, j] + t**3*B)
  Q = np.vstack(pts); R = np.roll(Q, -1, axis=0)
  return (0.5*np.sum(Q[:, 0]*R[:, 1] - R[:, 0]*Q[:, 1]),
          np.sum(np.linalg.norm(R - Q, axis=1)))

V = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float).T
C = np.full((4, 4), np.nan)
C[:, 1] = [1.3, 0.3, 0.7, 0.7]       # right side: an S
C[:, 2] = [0.7, 1.4, 0.3, 1.4]       # top side: a bulge
exact_area, exact_perimeter = bezier_outline(V, C)
m = gf.Mesh('generate', gf.MesherObject('polygon', V, C), 0.1, 2)
area, perimeter = measures(m, 'IM_TRIANGLE(6)')
print('curved square: area %g (exact %g), perimeter %g (exact %g)'
      % (area, exact_area, perimeter, exact_perimeter))
assert abs(area - exact_area) < 1e-5 and abs(perimeter - exact_perimeter) < 1e-4

# the same, without curved side: the polygon
m = gf.Mesh('generate', gf.MesherObject('polygon', V, np.full((4, 4), np.nan)), 0.2, 1)
area, perimeter = measures(m, 'IM_TRIANGLE(6)')
assert abs(area - 1) < 1e-8 and abs(perimeter - 4) < 1e-8

# extruded
m = gf.Mesh('generate', gf.MesherObject('extrusion', gf.MesherObject('polygon', V, C),
                                        0, 0.5), 0.15, 2)
vol, area = measures(m, 'IM_TETRAHEDRON(5)')
print('extruded curved square: volume %g (exact %g)' % (vol, 0.5*exact_area))
assert abs(vol - 0.5*exact_area) < 1e-3
