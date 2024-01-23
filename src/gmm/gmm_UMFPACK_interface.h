/* -*- c++ -*- (enables emacs c++ mode) */
/*===========================================================================

 Copyright (C) 2024-2024 Konstantinos Poulios

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
 along  with  this program;  if not, write to the Free Software Foundation,
 Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301, USA.

 As a special exception, you  may use  this file  as it is a part of a free
 software  library  without  restriction.  Specifically,  if   other  files
 instantiate  templates  or  use macros or inline functions from this file,
 or  you compile this  file  and  link  it  with other files  to produce an
 executable, this file  does  not  by itself cause the resulting executable
 to be covered  by the GNU Lesser General Public License.  This   exception
 does not  however  invalidate  any  other  reasons why the executable file
 might be covered by the GNU Lesser General Public License.

===========================================================================*/

/**@file gmm_UMFPACK_interface.h
   @author Konstantinos Poulios <logari81@gmail.com>,
   @date January 11, 2024.
   @brief Interface with UMFPACK (direct solver for sparse matrices).
*/
#if defined(GMM_USES_UMFPACK)

#ifndef GMM_UMFPACK_INTERFACE_H
#define GMM_UMFPACK_INTERFACE_H

#include "gmm_kernel.h"


extern "C" {
#include <umfpack.h>
}

namespace gmm {

  /* ********************************************************************* */
  /*   UMFPACK solve interface                                             */
  /* ********************************************************************* */

  //
  template <typename T>
  struct umfpack_interf {
    int n;
    std::vector<int> c;
    std::vector<int> i;
    std::vector<double> ar;
    std::vector<double> xr;
    std::vector<double> br;

    inline int
    umfpack_symbolic(void **output) {
      return umfpack_di_symbolic(n, n, &(c[0]), &(i[0]), &(ar[0]),
                                 output, NULL, NULL);
    }
    inline int
    umfpack_numeric(void *symbolic, void **output) {
      return umfpack_di_numeric(&(c[0]), &(i[0]), &(ar[0]),
                                symbolic, output, NULL, NULL);
    }
    static inline void
    umfpack_free_symbolic(void **symbolic) {
      umfpack_di_free_symbolic(symbolic);
    }
    inline int
    umfpack_solve(void *numeric) {
      return umfpack_di_solve(UMFPACK_A, &(c[0]), &(i[0]), &(ar[0]),
                              &(xr[0]), &(br[0]), numeric, NULL, NULL);
    }
    static inline void
    umfpack_free_numeric(void **numeric) {
      umfpack_di_free_numeric(numeric);
    }
    template<typename MAT, typename VECTX, typename VECTB>
    umfpack_interf(MAT &A, VECTX &X, VECTB &B)
    {
      n = int(gmm::vect_size(B));
      GMM_ASSERT2(gmm::mat_nrows(A) == size_type(n), "Incompatible matrix-vector sizes");
      GMM_ASSERT2(gmm::mat_ncols(A) == size_type(n), "System matrix needs to be square");
      xr.resize(n);
      br.resize(n);
      gmm::copy(X, xr);
      gmm::copy(B, br);
      csc_matrix<double> csc_A(n, n);
      gmm::copy(A, csc_A);
      c.resize(csc_A.jc.size());
      i.resize(csc_A.ir.size());
      ar.resize(csc_A.pr.size());
      gmm::copy(csc_A.jc, c);
      gmm::copy(csc_A.ir, i);
      gmm::copy(csc_A.pr, ar);
    }
  };

  template <typename T>
  struct umfpack_interf<std::complex<T>> {
    int n;
    std::vector<int> c;
    std::vector<int> i;
    std::vector<double> ar, ai;
    std::vector<double> xr, xi;
    std::vector<double> br, bi;
  
    inline int
    umfpack_symbolic(void **output) {
      return umfpack_zi_symbolic(n, n, &(c[0]), &(i[0]), &(ar[0]), &(ai[0]),
                                 output, NULL, NULL);
    }
    inline int
    umfpack_numeric(void *symbolic, void **output) {
      return umfpack_zi_numeric(&(c[0]), &(i[0]), &(ar[0]), &(ai[0]),
                                symbolic, output, NULL, NULL);
    }
    static inline void
    umfpack_free_symbolic(void **symbolic) {
      umfpack_zi_free_symbolic(symbolic);
    }
    inline int
    umfpack_solve(void *numeric) {
      return umfpack_zi_solve(UMFPACK_A, &(c[0]), &(i[0]), &(ar[0]), &(ai[0]),
                              &(xr[0]), &(xi[0]), &(br[0]), &(bi[0]), numeric, NULL, NULL);
    }
    static inline void
    umfpack_free_numeric(void **numeric) {
      umfpack_zi_free_numeric(numeric);
    }
    template<typename MAT, typename VECTX, typename VECTB>
    umfpack_interf(MAT &A, VECTX &X, VECTB &B)
    {
      n = int(gmm::vect_size(B));
      GMM_ASSERT2(gmm::mat_nrows(A) == size_type(n), "Incompatible matrix-vector sizes");
      GMM_ASSERT2(gmm::mat_ncols(A) == size_type(n), "System matrix needs to be square");
      xr.resize(n);
      xi.resize(n);
      br.resize(n);
      bi.resize(n);
      gmm::copy(gmm::real_part(X), xr);
      gmm::copy(gmm::imag_part(X), xi);
      gmm::copy(gmm::real_part(B), br);
      gmm::copy(gmm::imag_part(B), bi);
      csc_matrix<std::complex<double>> csc_A(n, n);
      gmm::copy(A, csc_A);
      c.resize(csc_A.jc.size());
      i.resize(csc_A.ir.size());
      ar.resize(csc_A.pr.size());
      ai.resize(csc_A.pr.size());
      gmm::copy(csc_A.jc, c);
      gmm::copy(csc_A.ir, i);
      gmm::copy(gmm::real_part(csc_A.pr), ar);
      gmm::copy(gmm::imag_part(csc_A.pr), ai);
    }
  };

  // UMFPACK solve interface, returns "true" if successful
  template <typename MAT, typename VECTX, typename VECTB>
  bool UMFPACK_solve(const MAT &A, VECTX &X, const VECTB &B) {
    typedef typename linalg_traits<MAT>::value_type T;
    umfpack_interf<T> intrf(A, B, X); // possibly convert float to double
                                      // umfpack supports only double precision
    int status;
    void *symbolic, *numeric;
    status = intrf.umfpack_symbolic(&symbolic);
    status = intrf.umfpack_numeric(symbolic, &numeric);
    intrf.umfpack_free_symbolic(&symbolic);
    status = intrf.umfpack_solve(numeric);
    intrf.umfpack_free_numeric(&numeric);
    return (status == UMFPACK_OK);
  }

}
  
#endif // GMM_UMFPACK_INTERFACE_H

#endif // GMM_USES_UMFPACK
