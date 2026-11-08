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
"""  Test of the 'placed' mesher object (a rigid motion): an extrusion
     placed in an oblique plane, a 2D rectangle turned; volume, area and
     position of the generated meshes.
"""
import numpy as np

import getfem as gf

gf.util_trace_level(0)

c = 1. / np.sqrt(2.)

# A 1 x 0.5 rectangle extruded by 0.2, placed with its origin at (1, 2, 3),
# its x axis along (1, 1, 0) and its y axis along z: the extrusion goes
# along x ^ y = (1, -1, 0) / sqrt(2)
e = gf.MesherObject('extrusion', gf.MesherObject('rectangle', [0., 0.], [1., 0.5]), 0., 0.2)
p = gf.MesherObject('placed', e, [1., 2., 3.], [1., 1., 0.], [0., 0., 1.])
m = gf.Mesh('generate', p, 0.05, 1)
mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(2)'))
m.set_region(1, m.outer_faces())
vol = gf.asm('generic', mim, 0, '1', -1)
area = gf.asm('generic', mim, 0, '1', 1)
P = m.pts()
print('placed extrusion: volume %g (exact 0.1), area %g (exact 1.6)' % (vol, area))
print('  points in [%s], [%s]' % (P.min(axis=1), P.max(axis=1)))
assert abs(vol - 0.1) < 2e-3 and abs(area - 1.6) < 2e-2
assert np.allclose(P.min(axis=1), [1., 2. - 0.2*c, 3.], atol=1e-6)
assert np.allclose(P.max(axis=1), [1. + 1.2*c, 2. + c, 3.5], atol=1e-6)

# In 2D: a rectangle turned by 30 degrees about (1, 1)
t = np.pi / 6.
r2 = gf.MesherObject('placed', gf.MesherObject('rectangle', [0., 0.], [2., 1.]),
                     [1., 1.], [np.cos(t), np.sin(t)])
m = gf.Mesh('generate', r2, 0.1, 1)
mim = gf.MeshIm(m, gf.Integ('IM_TRIANGLE(2)'))
area = gf.asm('generic', mim, 0, '1', -1)
print('placed rectangle: area %g (exact 2)' % area)
assert abs(area - 2.) < 1e-6
