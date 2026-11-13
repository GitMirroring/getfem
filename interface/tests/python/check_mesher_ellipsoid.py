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
"""  Test of the 'ellipsoid' mesher object: volume and area of a mesh of an
     inclined ellipsoid with a spherical hole, and of a spheroid.
"""
import numpy as np

import getfem as gf

gf.util_trace_level(0)
trapezoid = getattr(np, "trapezoid", None) or np.trapz    # (NumPy < 2: trapz)


def ellipsoid_area(a, b, c, n=1200):
  u = np.linspace(0, np.pi, n)
  v = np.linspace(0, 2*np.pi, 2*n)
  U, V = np.meshgrid(u, v, indexing="ij")
  Xu = np.stack([a*np.cos(U)*np.cos(V), b*np.cos(U)*np.sin(V), -c*np.sin(U)])
  Xv = np.stack([-a*np.sin(U)*np.sin(V), b*np.sin(U)*np.cos(V), 0*U])
  J = np.linalg.norm(np.cross(Xu, Xv, axis=0), axis=0)
  return trapezoid(trapezoid(J, v, axis=1), u)


def measures(obj, h):
  m = gf.Mesh('generate', obj, h, 2)
  mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(6)'))
  m.set_region(1, m.outer_faces())
  return gf.asm('generic', mim, 0, '1', -1), gf.asm('generic', mim, 0, '1', 1)


# an ellipsoid turned about z, less a ball
a, b, c, th, r = 1.0, 0.6, 0.35, 0.5, 0.15
axes = np.array([[np.cos(th), -np.sin(th), 0.], [np.sin(th), np.cos(th), 0.], [0., 0., 1.]])
E = gf.MesherObject('ellipsoid', [0.1, 0.2, -0.1], [a, b, c], axes)
H = gf.MesherObject('ball', [0.3, 0.3, -0.1], r)
volume, area = measures(gf.MesherObject('set minus', E, H), 0.12)
exact_volume = 4/3*np.pi*(a*b*c - r**3)
exact_area = ellipsoid_area(a, b, c) + 4*np.pi*r**2
print('ellipsoid: volume %g (exact %g), area %g (exact %g)' % (volume, exact_volume, area, exact_area))
assert abs(volume - exact_volume) < 3e-4 and abs(area - exact_area) < 2e-3*exact_area

# a spheroid (two equal semi-axes), along the axes
volume, area = measures(gf.MesherObject('ellipsoid', [0., 0., 0.], [0.5, 0.5, 0.8]), 0.12)
exact_volume, exact_area = 4/3*np.pi*0.5*0.5*0.8, ellipsoid_area(0.5, 0.5, 0.8)
print('spheroid: volume %g (exact %g), area %g (exact %g)' % (volume, exact_volume, area, exact_area))
assert abs(volume - exact_volume) < 3e-4 and abs(area - exact_area) < 2e-3*exact_area
