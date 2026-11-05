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
"""  Test of the 'ellipse' mesher object: area and perimeter of a mesh of
     an inclined ellipse with an elliptic hole.
"""
import numpy as np

import getfem as gf

gf.util_trace_level(0)

a, b, angle = 1.0, 0.5, 0.4
E = gf.MesherObject('ellipse', [0.2, -0.1], a, b, angle)
H = gf.MesherObject('ellipse', [0.2, -0.1], 0.2, 0.3)
m = gf.Mesh('generate', gf.MesherObject('set minus', E, H), 0.08, 3)
mim = gf.MeshIm(m, gf.Integ('IM_TRIANGLE(10)'))
m.set_region(1, m.outer_faces())
area = gf.asm('generic', mim, 0, '1', -1)
perimeter = gf.asm('generic', mim, 0, '1', 1)


def ellipse_perimeter(a, b):
  t = np.linspace(0, 2*np.pi, 200001)
  return np.trapezoid(np.hypot(a*np.sin(t), b*np.cos(t)), t)


exact_area = np.pi*(a*b - 0.2*0.3)
exact_perimeter = ellipse_perimeter(a, b) + ellipse_perimeter(0.2, 0.3)
print('ellipse: area %g (exact %g), perimeter %g (exact %g)'
      % (area, exact_area, perimeter, exact_perimeter))
assert abs(area - exact_area) < 1e-3 and abs(perimeter - exact_perimeter) < 1e-3
