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
# along  with  this program. If not, see https://www.gnu.org/licenses/.
#
############################################################################
"""  Test of gf.eigsh (gf.linsolve('eigsh', ...)): the ARPACK-ng eigensolver.

  The modes of the Laplacian on the unit square, u = 0 on its boundary
  (P2 elements): the eigenvalues pi^2 (m^2 + n^2); the matrices restricted
  to the inner dofs; M-orthonormal modes; the standard problem (M omitted)
  on a tridiagonal matrix; a shift; the wrong arguments.

  $Id$
"""

import numpy as np
import getfem as gf

if not gf.util_arpack_linked():
  print('ARPACK-ng not linked with GetFEM, aborting the test'); exit(0)

m = gf.Mesh('cartesian', np.linspace(0., 1., 17), np.linspace(0., 1., 17))
mf = gf.MeshFem(m, 1)
mf.set_classical_fem(2)
mim = gf.MeshIm(m, 4)
md = gf.Model('real')
md.add_fem_variable('u', mf)
K = gf.asm_generic(mim, 2, 'Grad_u.Grad_Test_u', -1, md)
M = gf.asm_generic(mim, 2, 'u*Test_u', -1, md)
m.set_region(1, m.outer_faces())
bd = set(mf.basic_dof_on_region(1))
inner = [i for i in range(mf.nbdof()) if i not in bd]
Ki, Mi = gf.Spmat('copy', K, inner, inner), gf.Spmat('copy', M, inner, inner)

lam, modes = gf.eigsh(Ki, Mi, nev=4, sigma=0.)
print('eigenvalues / pi^2:', lam / np.pi**2)
assert lam.shape == (4,) and modes.shape == (4, len(inner))
assert np.all(np.diff(lam) >= 0)
for l, k in zip(lam, (2., 5., 5., 8.)):
  assert abs(l / (k * np.pi**2) - 1) < 1e-3
Kd, Md = Ki.full(), Mi.full()
for l, phi in zip(lam, modes):
  assert np.linalg.norm(Kd @ phi - l * (Md @ phi)) < 1e-8 * l * np.linalg.norm(Md @ phi)
G = modes @ Md @ modes.T
assert np.abs(G - np.eye(4)).max() < 1e-10

# a shift: the two nearest to 30 pi^2 (m^2 + n^2 = 29 twice: (2, 5), (5, 2))
lam, modes = gf.eigsh(Ki, Mi, 2, 30 * np.pi**2)
print('near 30 pi^2:', lam / np.pi**2)
assert np.all(np.abs(lam / (29 * np.pi**2) - 1) < 1e-2)

# the standard problem: tridiag(-1, 2, -1), lambda_k = 2 - 2 cos(k pi / (n + 1))
n = 50
T = gf.Spmat('empty', n, n)
for i in range(n):
  T.add(i, i, 2.)
  if i + 1 < n:
    T.add(i, i + 1, -1.); T.add(i + 1, i, -1.)
lam, modes = gf.linsolve('eigsh', T, 3)
exact = 2 - 2 * np.cos(np.arange(1, 4) * np.pi / (n + 1))
assert np.abs(lam - exact).max() < 1e-12
assert np.abs(modes @ modes.T - np.eye(3)).max() < 1e-10

for args in ((T, 0), (T, n), (T, Mi, 2)):
  try:
    gf.linsolve('eigsh', *args)
  except Exception:
    continue
  raise AssertionError('wrong arguments accepted: %s' % (args[1:],))
print('ok')
