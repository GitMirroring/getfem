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
"""  Test of disabled variables in a linear model: a disabled variable is
     a data, its coupling terms with the enabled variables go to the right
     hand side (they were dropped: the variable was taken as zero). A
     staggered resolution of a coupled linear problem, one variable at a
     time, converges to its monolithic solution, for a coupling term of a
     single variable pair (non symmetric) and for a symmetric one.
"""
import numpy as np

import getfem as gf

gf.util_trace_level(0)

m = gf.Mesh('cartesian', np.linspace(0, 1, 11))
m.set_region(1, m.outer_faces())
mf = gf.MeshFem(m, 1)
mf.set_classical_fem(2)
mim = gf.MeshIm(m, 4)


def model(symmetric):
  md = gf.Model('real')
  md.add_fem_variable('u', mf)
  md.add_fem_variable('p', mf)
  # -u'' = 1 + p/2, -p'' = u/2 (symmetric: the same coupling written once)
  if symmetric:
    md.add_linear_term(mim, 'Grad_u.Grad_Test_u + Grad_p.Grad_Test_p - 0.5*(p*Test_u + u*Test_p)')
  else:
    md.add_linear_term(mim, 'Grad_u.Grad_Test_u - 0.5*p*Test_u')
    md.add_linear_term(mim, 'Grad_p.Grad_Test_p - 0.5*u*Test_p')
  md.add_source_term(mim, 'Test_u')
  ib = md.add_Dirichlet_condition_with_multipliers(mim, 'u', mf, 1)
  md.add_Dirichlet_condition_with_simplification('p', 1)
  return md, md.mult_varname_Dirichlet(ib)


for symmetric in (False, True):
  md, mult = model(symmetric)
  md.solve()
  u, p = md.variable('u'), md.variable('p')
  md, mult = model(symmetric)
  for it in range(20):
    md.disable_variable('p')                     # u (and its multiplier), p given
    md.solve()
    md.enable_variable('p')
    md.disable_variable('u'); md.disable_variable(mult)   # p, u given
    md.solve()
    md.enable_variable('u'); md.enable_variable(mult)
  err = max(np.abs(md.variable('u') - u).max(), np.abs(md.variable('p') - p).max())
  assert err < 1e-10 * np.abs(u).max(), (symmetric, err)
  assert np.abs(p).max() > 1e-3                  # (a real coupling)
