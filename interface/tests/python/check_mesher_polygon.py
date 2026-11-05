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
