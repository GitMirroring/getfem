/*===========================================================================

 Copyright (C) 2026 Yves Renard.

 This file is a part of GetFEM

 GetFEM  is  free software;  you  can  redistribute  it  and/or modify it
 under  the  terms  of the  GNU  Lesser General Public License as published
 by  the  Free Software Foundation;  either version 3 of the License,  or
 (at your option) any later version along with the GCC Runtime Library
 Exception either version 3.1 or (at your option) any later version.
 This program  is  distributed  in  the  hope  that it will be useful,  but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 or  FITNESS  FOR  A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 License and GCC Runtime Library Exception for more details.
 You  should  have received a copy of the GNU Lesser General Public License
 along  with  this program. If not, see https://www.gnu.org/licenses/.

===========================================================================*/

/* gmm::ARPACK_eigsh on the 1D finite differences and P1 elements of
   (0, 1) with Dirichlet conditions, whose eigenvalues are known in closed
   form; a singular M (a dof without mass); the wrong arguments. */

#include "gmm/gmm.h"
#include "gmm/gmm_arpack_interface.h"

using gmm::size_type;
using std::cout;
using std::endl;

#if defined(GMM_USES_ARPACK)

typedef gmm::csc_matrix<double> CSC;

/* n x n tridiagonal (b, a, b) */
static CSC tridiag(size_type n, double a, double b) {
  gmm::row_matrix<gmm::wsvector<double> > K(n, n);
  for (size_type i = 0; i < n; ++i) {
    K(i, i) = a;
    if (i + 1 < n) K(i, i+1) = K(i+1, i) = b;
  }
  CSC C(n, n);
  gmm::copy(K, C);
  return C;
}

/* max_k |K phi_k - lambda_k M phi_k| / |lambda_k|, and
   max |phi_j' M phi_k - delta_jk| */
static void check(const CSC &K, const CSC &M, const std::vector<double> &lambda,
                  const gmm::dense_matrix<double> &modes,
                  double &res, double &orth) {
  size_type n = gmm::mat_nrows(K), nev = lambda.size();
  std::vector<double> kp(n), mp(n), phi(n);
  res = orth = 0.0;
  for (size_type k = 0; k < nev; ++k) {
    gmm::copy(gmm::mat_col(modes, k), phi);
    gmm::mult(K, phi, kp);
    gmm::mult(M, phi, mp);
    gmm::add(gmm::scaled(mp, -lambda[k]), kp);
    res = std::max(res, gmm::vect_norm2(kp) / gmm::abs(lambda[k]));
    for (size_type j = 0; j < nev; ++j)
      orth = std::max(orth, gmm::abs(gmm::vect_sp(gmm::mat_col(modes, j), mp)
                                     - (j == k ? 1.0 : 0.0)));
  }
}

static void laplacian_standard() {
  // K = tridiag(-1, 2, -1): lambda_k = 2 - 2 cos(k pi / (n + 1))
  size_type n = 200;
  CSC K = tridiag(n, 2.0, -1.0);
  std::vector<double> lambda;
  gmm::dense_matrix<double> modes;
  size_type nconv = gmm::ARPACK_eigsh(K, 5, 0.0, lambda, modes);
  GMM_ASSERT1(nconv == 5 && lambda.size() == 5 && gmm::mat_ncols(modes) == 5
              && gmm::mat_nrows(modes) == n, "wrong sizes");
  for (size_type i = 0; i < 5; ++i) {
    double th = double(i + 1) * M_PI / double(n + 1);
    GMM_ASSERT1(gmm::abs(lambda[i] - (2.0 - 2.0 * cos(th))) < 1e-12,
                "eigenvalue " << i << ": " << lambda[i]);
  }
  double res, orth;
  check(K, tridiag(n, 1.0, 0.0), lambda, modes, res, orth);
  GMM_ASSERT1(res < 1e-9 && orth < 1e-10, "residual " << res
              << ", orthonormality " << orth);
  cout << "standard eigenproblem: ok, " << gmm::vref(lambda) << endl;
}

static void p1_generalized_shifted() {
  // P1 on n + 1 elements of size h: K = tridiag(-1, 2, -1) / h,
  // M = tridiag(1, 4, 1) h / 6: lambda_k = 6 / h^2 (1 - cos t) / (2 + cos t),
  // t = k pi h; the 4 nearest to a shift between lambda_20 and lambda_21
  size_type n = 300;
  double h = 1.0 / double(n + 1);
  CSC K = tridiag(n, 2.0 / h, -1.0 / h), M = tridiag(n, 4.0 * h / 6.0, h / 6.0);
  auto lam = [h](size_type k) {
    double t = double(k) * M_PI * h;
    return 6.0 / (h * h) * (1.0 - cos(t)) / (2.0 + cos(t));
  };
  double sigma = 0.5 * (lam(20) + lam(21)) + 1.0;
  std::vector<double> lambda;
  gmm::dense_matrix<double> modes;
  gmm::ARPACK_eigsh(K, M, 4, sigma, lambda, modes);
  GMM_ASSERT1(lambda.size() == 4, "4 eigenvalues expected");
  for (size_type i = 0; i < 4; ++i)
    GMM_ASSERT1(gmm::abs(lambda[i] - lam(19 + i)) < 1e-9 * lam(19 + i),
                "eigenvalue " << i << ": " << lambda[i] << " " << lam(19 + i));
  double res, orth;
  check(K, M, lambda, modes, res, orth);
  GMM_ASSERT1(res < 1e-9 && orth < 1e-10, "residual " << res
              << ", orthonormality " << orth);
  cout << "generalized eigenproblem, shifted: ok, " << gmm::vref(lambda) << endl;
}

static void singular_mass() {
  // the P1 problem and a dof without mass (K: 1, coupled to the first dof
  // by 1/2): its eigenvalue infinite, the lowest ones those of the
  // condensed problem (K(0, 0) - 1/4), the dof -phi_0/2 in their modes
  size_type n = 100;
  double h = 1.0 / double(n + 1);
  CSC K0 = tridiag(n, 2.0 / h, -1.0 / h), M0 = tridiag(n, 4.0 * h / 6.0, h / 6.0);
  gmm::col_matrix<gmm::wsvector<double> > K(n + 1, n + 1), M(n + 1, n + 1);
  gmm::copy(K0, gmm::sub_matrix(K, gmm::sub_interval(0, n)));
  gmm::copy(M0, gmm::sub_matrix(M, gmm::sub_interval(0, n)));
  K(n, n) = 1.0; K(0, n) = K(n, 0) = 0.5;
  std::vector<double> lambda;
  gmm::dense_matrix<double> modes;
  gmm::ARPACK_eigsh(K, M, 3, 0.0, lambda, modes);
  for (size_type i = 0; i < 3; ++i)
    GMM_ASSERT1(gmm::abs(modes(n, i) + 0.5 * modes(0, i)) < 1e-10,
                "mode " << i << ": " << modes(n, i) << " " << modes(0, i));
  gmm::col_matrix<gmm::wsvector<double> > Kc(n, n);
  gmm::copy(K0, Kc);
  Kc(0, 0) = Kc(0, 0) - 0.25;
  std::vector<double> lambda0;
  gmm::ARPACK_eigsh(Kc, M0, 3, 0.0, lambda0, modes);
  for (size_type i = 0; i < 3; ++i)
    GMM_ASSERT1(gmm::abs(lambda[i] - lambda0[i]) < 1e-9 * lambda0[i],
                "eigenvalue " << i << ": " << lambda[i] << " " << lambda0[i]);
  cout << "singular M: ok, " << gmm::vref(lambda) << endl;
}

static void bad_arguments() {
  CSC K = tridiag(10, 2.0, -1.0), M = tridiag(9, 1.0, 0.0);
  std::vector<double> lambda;
  gmm::dense_matrix<double> modes;
  size_type nerr = 0;
  try { gmm::ARPACK_eigsh(K, 0, 0.0, lambda, modes); }
  catch (const gmm::gmm_error &) { ++nerr; }
  try { gmm::ARPACK_eigsh(K, 10, 0.0, lambda, modes); }
  catch (const gmm::gmm_error &) { ++nerr; }
  try { gmm::ARPACK_eigsh(K, M, 2, 0.0, lambda, modes); }
  catch (const gmm::gmm_error &) { ++nerr; }
  GMM_ASSERT1(nerr == 3, "the wrong arguments should be refused");
  cout << "wrong arguments refused: ok" << endl;
}

#endif

int main(void) {
  try {
#if defined(GMM_USES_ARPACK)
    laplacian_standard();
    p1_generalized_shifted();
    singular_mass();
    bad_arguments();
#else
    cout << "GetFEM built without ARPACK: test skipped" << endl;
#endif
  }
  GMM_STANDARD_CATCH_ERROR;
  return 0;
}
