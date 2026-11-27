/* -*- c++ -*- (enables emacs c++ mode) */
/*===========================================================================

 Copyright (C) 2026 Yves Renard

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

 As a special exception, you  may use  this file  as it is a part of a free
 software  library  without  restriction.  Specifically,  if   other  files
 instantiate  templates  or  use macros or inline functions from this file,
 or  you compile this  file  and  link  it  with other files  to produce an
 executable, this file  does  not  by itself cause the resulting executable
 to be covered  by the GNU Lesser General Public License.  This   exception
 does not  however  invalidate  any  other  reasons why the executable file
 might be covered by the GNU Lesser General Public License.

===========================================================================*/

/**@file gmm_arpack_interface.h
   @author  Yves Renard <Yves.Renard@insa-lyon.fr>
   @date 2026.
   @brief Interface with ARPACK-ng: the eigenpairs of a symmetric
   generalized eigenproblem nearest to a shift.

   ARPACK_eigsh computes the nev eigenvalues nearest to a shift sigma of
   K phi = lambda M phi (K symmetric, M symmetric positive semi-definite;
   M omitted: the identity), and their modes (M-orthonormal), by the
   implicitly restarted Lanczos method of ARPACK (dsaupd, dseupd, through
   the ISO C binding of ARPACK-ng: dsaupd_c, dseupd_c) in shift-invert
   mode: the largest eigenvalues of (K - sigma M)^{-1} M, K - sigma M
   factorized once (by MUMPS, symmetric, if GetFEM is built with it; by
   SuperLU otherwise). A singular M is allowed (the multipliers of a mixed
   problem: their eigenvalues are infinite, never found). The same
   method and defaults as scipy.sparse.linalg.eigsh(K, nev, M, sigma).

   Sequential (ARPACK keeps its state between calls: one call at a time,
   serialized by a mutex).
*/
#ifndef GMM_ARPACK_INTERFACE_H
#define GMM_ARPACK_INTERFACE_H

#include "gmm_kernel.h"

#if defined(GMM_USES_ARPACK)

#include <algorithm>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>

#if defined(GMM_USES_MUMPS)
# include "gmm_MUMPS_interface.h"
#elif defined(GMM_USES_SUPERLU)
# include "gmm_superlu_interface.h"
#else
# error "gmm_arpack_interface.h needs MUMPS or SuperLU (K - sigma M is factorized)"
#endif

/* ARPACK-ng's ISO C binding (arpack.h, a_int = int: the LP64 build) */
extern "C" {
  void dsaupd_c(int *ido, char const *bmat, int n, char const *which, int nev,
                double tol, double *resid, int ncv, double *v, int ldv,
                int *iparam, int *ipntr, double *workd, double *workl,
                int lworkl, int *info);
  void dseupd_c(int rvec, char const *howmny, int const *select, double *d,
                double *z, int ldz, double sigma, char const *bmat, int n,
                char const *which, int nev, double tol, double *resid,
                int ncv, double *v, int ldv, int *iparam, int *ipntr,
                double *workd, double *workl, int lworkl, int *info);
}

namespace gmm {

  namespace arpack_detail {

    /* ARPACK keeps state between calls (Fortran SAVE): one at a time. */
    inline std::mutex &arpack_mutex() { static std::mutex m; return m; }

    inline std::string error_message(const char *routine, int info) {
      std::stringstream s;
      switch (info) {
      case 1: s << "maximum number of iterations taken"; break;
      case 3: s << "no shifts could be applied (increase ncv)"; break;
      case -9999: s << "could not build an Arnoldi factorization"; break;
      default: s << routine << " info = " << info
                 << " (see the ARPACK documentation)";
      }
      return s.str();
    }

#if defined(GMM_USES_MUMPS)
    /* K - sigma M, factorized by MUMPS (general symmetric: indefinite) */
    class shifted_factor {
      mumps_context<double> ctx;
    public:
      template <typename MAT> explicit shifted_factor(const MAT &A)
        : ctx(2) {
        csc_matrix<double> C(mat_nrows(A), mat_ncols(A));
        copy(A, C);
        ctx.set_matrix(C, false);
        ctx.ICNTL(1) = -1; ctx.ICNTL(2) = -1;  // no output
        ctx.ICNTL(3) = -1; ctx.ICNTL(4) = 0;
        ctx.ICNTL(14) += 80;                   // as MUMPS_solve
        ctx.analyze_and_factorize();
        GMM_ASSERT1(ctx.error_check(), "ARPACK_eigsh: the factorization of "
                    "K - sigma M failed (sigma an eigenvalue?)");
      }
      void solve(std::vector<double> &x) {
        ctx.set_vector(x);
        ctx.solve();
        GMM_ASSERT1(ctx.error_check(), "ARPACK_eigsh: MUMPS solve failed");
        ctx.mpi_broadcast();
        copy(ctx.vector(), x);
      }
    };
#else
    /* K - sigma M, factorized by SuperLU */
    class shifted_factor {
      SuperLU_factor<double> lu;
      std::vector<double> y;
    public:
      template <typename MAT> explicit shifted_factor(const MAT &A)
        : y(mat_nrows(A)) { lu.build_with(A); }
      void solve(std::vector<double> &x) { lu.solve(y, x); copy(y, x); }
    };
#endif

    /* y = M x; M null: the identity */
    template <typename MAT>
    inline void mult_M(const MAT *M, const double *x, double *y,
                       size_type n) {
      if (M) {
        std::vector<double> xx(x, x + n), yy(n);
        mult(*M, xx, yy);
        std::copy(yy.begin(), yy.end(), y);
      } else
        std::copy(x, x + n, y);
    }

    template <typename MAT1, typename MAT2>
    size_type eigsh(const MAT1 &K, const MAT2 *M, size_type nev,
                    double sigma, std::vector<double> &lambda,
                    dense_matrix<double> &modes) {
      size_type n = mat_nrows(K);
      GMM_ASSERT1(mat_ncols(K) == n, "ARPACK_eigsh: K is " << n << "x"
                  << mat_ncols(K) << ", it should be square");
      GMM_ASSERT1(!M || (mat_nrows(*M) == n && mat_ncols(*M) == n),
                  "ARPACK_eigsh: K is " << n << "x" << n
                  << ", M must be the same size");
      GMM_ASSERT1(nev > 0 && nev < n, "ARPACK_eigsh: nev = " << nev
                  << ", needs 0 < nev < n = " << n);
      GMM_ASSERT1(n < size_type(std::numeric_limits<int>::max() / 32),
                  "ARPACK_eigsh: too large a system for the LP64 ARPACK");

      // K - sigma M, factorized
      col_matrix<wsvector<double> > A(n, n);
      copy(K, A);
      if (M) add(scaled(*M, -sigma), A);
      else for (size_type i = 0; i < n; ++i) A(i, i) = A(i, i) - sigma;
      shifted_factor lu(A);

      // scipy's defaults: ncv, 10 n iterations, the machine precision
      size_type ncv = std::min(n, std::max<size_type>(2 * nev + 1, 20));
      size_type lworkl = ncv * (ncv + 8);
      const char *bmat = M ? "G" : "I", *which = "LM";
      int n_ = int(n), nev_ = int(nev), ncv_ = int(ncv);
      std::vector<double> resid(n), v(n * ncv), workd(3 * n),
        workl(lworkl), tmp(n);
      std::mt19937 gen(1);   // a reproducible starting vector
      std::uniform_real_distribution<double> unif(-1.0, 1.0);
      for (double &r : resid) r = unif(gen);
      int iparam[11] = {0}, ipntr[14] = {0};
      iparam[0] = 1;                              // exact shifts
      iparam[2] = int(std::min<size_type>(10 * n, size_type(1) << 30));
      iparam[6] = 3;                              // shift-invert
      int ido = 0, info = 1;                      // info 1: resid given

      std::lock_guard<std::mutex> guard(arpack_mutex());
      for (;;) {
        dsaupd_c(&ido, bmat, n_, which, nev_, 0.0, resid.data(), ncv_,
                 v.data(), n_, iparam, ipntr, workd.data(), workl.data(),
                 int(lworkl), &info);
        // ARPACK's 1-based offsets into workd: x, y, and M x
        auto at = [&](int i) { return size_type(std::max(ipntr[i], 1) - 1); };
        double *x = &workd[at(0)], *y = &workd[at(1)];
        if (ido == -1 || (ido == 1 && !M)) {   // y = (K - sigma M)^-1 M x
          mult_M(M, x, tmp.data(), n);
          lu.solve(tmp);
          std::copy(tmp.begin(), tmp.end(), y);
        } else if (ido == 1) {        // the same, M x given (at ipntr[2])
          std::copy(&workd[at(2)], &workd[at(2)] + n, tmp.begin());
          lu.solve(tmp);
          std::copy(tmp.begin(), tmp.end(), y);
        } else if (ido == 2) {                    // y = M x
          std::copy(x, x + n, tmp.begin());
          mult_M(M, tmp.data(), y, n);
        } else if (ido == 99)
          break;
        else
          GMM_ASSERT1(false, "ARPACK_eigsh: "
                      << error_message("dsaupd", std::min(info, -1)));
      }
      GMM_ASSERT1(info == 0, "ARPACK_eigsh: " << error_message("dsaupd",
                                                               info));

      std::vector<int> select(ncv);        // workspace with howmny "A"
      std::vector<double> d(nev), z(n * nev);
      dseupd_c(1, "A", select.data(), d.data(), z.data(), n_, sigma, bmat,
               n_, which, nev_, 0.0, resid.data(), ncv_, v.data(), n_,
               iparam, ipntr, workd.data(), workl.data(), int(lworkl),
               &info);
      GMM_ASSERT1(info == 0, "ARPACK_eigsh: " << error_message("dseupd",
                                                               info));
      size_type nconv = std::min(size_type(iparam[4]), nev);
      std::vector<size_type> order(nconv);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](size_type a, size_type b) { return d[a] < d[b]; });
      lambda.resize(nconv);
      resize(modes, n, nconv);
      for (size_type j = 0; j < nconv; ++j) {
        lambda[j] = d[order[j]];
        for (size_type i = 0; i < n; ++i)
          modes(i, j) = z[order[j] * n + i];
      }
      return nconv;
    }

  } // namespace arpack_detail

  /** The nev eigenvalues nearest to sigma of K phi = lambda M phi (K
      symmetric, M symmetric positive semi-definite), ascending, in
      lambda, and their modes, the columns of modes (n x nconv,
      M-orthonormal). Returns the number of converged eigenvalues nconv
      (nev, normally). ARPACK-ng in shift-invert mode, K - sigma M
      factorized by MUMPS or SuperLU. */
  template <typename MAT1, typename MAT2>
  size_type ARPACK_eigsh(const MAT1 &K, const MAT2 &M, size_type nev,
                         double sigma, std::vector<double> &lambda,
                         dense_matrix<double> &modes)
  { return arpack_detail::eigsh(K, &M, nev, sigma, lambda, modes); }

  /** The same for K phi = lambda phi (M the identity; the modes
      orthonormal). */
  template <typename MAT1>
  size_type ARPACK_eigsh(const MAT1 &K, size_type nev, double sigma,
                         std::vector<double> &lambda,
                         dense_matrix<double> &modes) {
    return arpack_detail::eigsh(K, static_cast<const MAT1 *>(nullptr), nev,
                                sigma, lambda, modes);
  }

} // namespace gmm

#endif // GMM_USES_ARPACK

#endif // GMM_ARPACK_INTERFACE_H
