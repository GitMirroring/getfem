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
"""  Test of the time step of a model in the assembly language: timestep is
     a data of the model (of size 1, a reserved name), so that a term using
     it is assembled again when the time step changes (by set_time_step or
     by setting the data).
"""
import numpy as np

import getfem as gf

gf.util_trace_level(0)

m = gf.Mesh('cartesian', np.linspace(0, 1, 5), np.linspace(0, 1, 5))
mf = gf.MeshFem(m, 1)
mf.set_classical_fem(1)
mim = gf.MeshIm(m, 2)

md = gf.Model('real')
md.add_fem_variable('u', mf)
md.add_linear_term(mim, '(u - 2*timestep)*Test_u')       # a linear term: assembled once
for dt in (1.0, 0.1, 0.01):
  md.set_time_step(dt)
  md.solve()
  assert np.allclose(md.variable('u'), 2 * dt), (dt, md.variable('u'))
md.set_variable('timestep', [3.0])                        # the data itself
md.solve()
assert np.allclose(md.variable('u'), 6.0)

try:                                                      # a reserved name
  md.add_initialized_data('timestep', [1.0])
except RuntimeError:
  pass
else:
  assert False, 'timestep: a reserved name expected'

md2 = gf.Model('real')                                    # 1 by default
md2.add_fem_variable('u', mf)
md2.add_linear_term(mim, '(u - timestep)*Test_u')
md2.solve()
assert np.allclose(md2.variable('u'), 1.0)

print('timestep: ok')
