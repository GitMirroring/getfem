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
"""  Test of the 'extrusion' and 'revolution' mesher objects: volume and
     area of the generated meshes.
"""
import numpy as np

import getfem as gf

gf.util_trace_level(0)


def measures(m):
  mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)'))
  m.set_region(1, m.outer_faces())
  return gf.asm('generic', mim, 0, '1', -1), gf.asm('generic', mim, 0, '1', 1)


# Extrusion along z of a rectangle with a circular hole
r = 0.3
profile = gf.MesherObject('set minus',
                          gf.MesherObject('rectangle', [0., 0.], [2., 1.]),
                          gf.MesherObject('ball', [1., 0.5], r))
m = gf.Mesh('generate', gf.MesherObject('extrusion', profile, 0., 0.5), 0.2, 2)
vol, area = measures(m)
A, P = 2. - np.pi*r*r, 6. + 2.*np.pi*r
print('extrusion: volume %g (exact %g), area %g (exact %g)'
      % (vol, 0.5*A, area, 2*A + 0.5*P))
assert abs(vol - 0.5*A) < 1e-3 and abs(area - (2*A + 0.5*P)) < 1e-2

# Revolution about z of a rectangle in the (r, z) plane: a thick tube
tube = gf.MesherObject('revolution',
                       gf.MesherObject('rectangle', [1., 0.], [1.5, 1.]))
m = gf.Mesh('generate', tube, 0.2, 2)
vol, area = measures(m)
exact_vol = np.pi*(1.5**2 - 1.)
exact_area = 2.*np.pi*(1.5 + 1.) + 2.*exact_vol
print('revolution: volume %g (exact %g), area %g (exact %g)'
      % (vol, exact_vol, area, exact_area))
assert abs(vol - exact_vol) < 1e-2 and abs(area - exact_area) < 5e-2
