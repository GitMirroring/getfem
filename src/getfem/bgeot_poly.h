/* -*- c++ -*- (enables emacs c++ mode) */
/*===========================================================================

 Copyright (C) 2000-2020 Yves Renard

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


#ifndef BGEOT_POLY_H__
#define BGEOT_POLY_H__

/** @file bgeot_poly.h
    @author  Yves Renard <Yves.Renard@insa-lyon.fr>
    @date December 01, 2000.
    @brief Multivariate polynomials.
*/

#include "bgeot_config.h"
#include "dal_static_stored_objects.h"
#include <vector>

namespace bgeot
{
  /// used as the common size type in the library
  typedef size_t size_type;
  ///
  /// used as the common short type integer in the library
  typedef gmm::uint16_type short_type;
  ///

  /** Return the value of @f$ \frac{(n+p)!}{n!p!} @f$ which
   * is the number of monomials of a polynomial of @f$n@f$
   * variables and degree @f$d@f$.
   */
  size_type alpha(short_type n, short_type d);

  /** Vector of integer (16 bits type) which represent the powers
   *  of a monomial
   */
  class power_index {
    std::vector<short_type> v;
    mutable short_type degree_;
    mutable size_type global_index_;
    void dirty() const
    { degree_ = short_type(-1); global_index_ = size_type(-1); }
  public :
    typedef std::vector<short_type>::iterator iterator;
    typedef std::vector<short_type>::const_iterator const_iterator;
    typedef std::vector<short_type>::reverse_iterator reverse_iterator;
    typedef std::vector<short_type>::const_reverse_iterator const_reverse_iterator;
    short_type operator[](size_type idx) const { return v[idx]; }
    short_type& operator[](size_type idx) { dirty(); return v[idx]; }

    iterator begin() { dirty(); return v.begin(); }
    const_iterator begin() const { return v.begin(); }
    iterator end() { dirty(); return v.end(); }
    const_iterator end() const { return v.end(); }

    reverse_iterator rbegin() { dirty(); return v.rbegin(); }
    const_reverse_iterator rbegin() const { return v.rbegin(); }
    reverse_iterator rend() { dirty(); return v.rend(); }
    const_reverse_iterator rend() const { return v.rend(); }

    size_type size() const { return v.size(); }
    /// Gives the next power index
    const power_index &operator ++();
    /// Gives the next power index
    const power_index operator ++(int)
      { power_index res = *this; ++(*this); return res; }
    /// Gives the previous power index
    const power_index &operator --();
    /// Gives the previous power index
    const power_index operator --(int)
      { power_index res = *this; --(*this); return res; }
    /**  Gives the global number of the index (i.e. the position of
     *   the corresponding monomial
     */
    size_type global_index() const;
    /// Gives the degree.
    short_type degree() const;
    /// Constructor
    power_index(short_type nn);
    /// Constructor
    power_index() { dirty(); }
  };

  /**
   * This class deals with plain polynomials with
   * several variables.
   *
   * A polynomial of @f$n@f$ variables and degree @f$d@f$ is stored in a vector
   * of @f$\alpha_d^n@f$ components.
   *
   * <h3>Example of code</h3>
   *
   *   the following code is valid :
   *   @code
   *   #include<bgeot_poly.h>
   *   bgeot::polynomial<double> P, Q;
   *   P = bgeot::polynomial<double>(2,2,0); // P = x
   *   Q = bgeot::polynomial<double>(2,2,1); // Q = y
   *   P += Q; // P is equal to x+y.
   *   P *= Q; // P is equal to xy + y^2
   *   bgeot::power_index pi(P.dim());
   *   bgeot::polynomial<double>::const_iterator ite = Q.end();
   *   bgeot::polynomial<double>::const_iterator itb = Q.begin();
   *   for ( ; itb != ite; ++itb, ++pi)
   *     if (*itb != double(0))
   *       cout "there is x to the power " << pi[0]
   *             << " and y to the power "
   *             << pi[1] << " and a coefficient " << *itb << endl;
   *  @endcode
   *
   *  <h3>Monomials ordering.</h3>
   *
   *       The constant coefficient is placed first with the index 0.
   *       Two monomials of different degrees are ordered following
   *       there respective degree.
   *
   *       If two monomials have the same degree, they are ordered with the
   *       degree of the mononomials without the n first variables which
   *       have the same degree. The index of the monomial
   *       @f$ x_0^{i_0}x_1^{i_1} ... x_{n-1}^{i_{n-1}} @f$
   *       is then
   *       @f$ \alpha_{d-1}^{n} + \alpha_{d-i_0-1}^{n-1}
   *          + \alpha_{d-i_0-i_1-1}^{n-2} + ... + \alpha_{i_{n-1}-1}^{1}, @f$
   *       where @f$d = \sum_{l=0}^{n-1} i_l@f$ is the degree of the monomial.
   *       (by convention @f$\alpha_{-1}^{n} = 0@f$).
   *
   *  <h3>Dealing with the vector of power.</h3>
   *
   *        The answer to the question : what is the next and previous
   *        monomial of @f$x_0^{i_0}x_1^{i_1} ... x_{n-1}^{i_{n-1}}@f$ in the
   *        vector is the following :
   *
   *        To take the next coefficient, let @f$l@f$ be the last index between 0
   *        and @f$n-2@f$ such that @f$i_l \ne 0@f$ (@f$l = -1@f$ if there is not), then
   *        make the operations @f$a = i_{n-1}; i_{n-1} = 0; i_{l+1} = a+1;
   *        \mbox{ if } l \ge 0 \mbox{ then } i_l = i_l - 1@f$.
   *
   *        To take the previous coefficient, let @f$l@f$ be the last index
   *        between 0 and @f$n-1@f$ such that @f$i_l \ne 0@f$ (if there is not, there
   *        is no previous monomial) then make the operations @f$a = i_l;
   *        i_l = 0; i_{n-1} = a - 1; \mbox{ if } l \ge 1 \mbox{ then }
   *        i_{l-1} = i_{l-1} + 1@f$.
   *
   *  <h3>Direct product multiplication.</h3>
   *
   *        This direct product multiplication of P and Q is the
   *        multiplication considering that the variables of Q follow the
   *        variables of P. The result is a polynomial with the number of
   *        variables of P plus the number of variables of Q.
   *        The resulting polynomials have a smaller degree.
   *
   */
  template<typename T> class polynomial : public std::vector<T> {
  protected :

    short_type n, d;

  public :

    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::reverse_iterator reverse_iterator;
    typedef typename std::vector<T>::const_reverse_iterator const_reverse_iterator;

    /// Gives the degree of the polynomial
    short_type degree() const { return d; }
    /**  gives the degree of the polynomial, considering only non-zero
     * coefficients
     */
    short_type real_degree() const;
    ///     Gives the dimension (number of variables)
    short_type dim() const { return n; }
    /// Change the degree of the polynomial to d.
    void change_degree(short_type dd);
    /** Add to the polynomial a monomial of coefficient a and
     * correpsonding to the power index pi.
     */
    void add_monomial(const T &coeff, const power_index &power);
    ///  Add Q to P. P contains the result.
    polynomial &operator +=(const polynomial &Q);
    /// Subtract Q from P. P contains the result.
    polynomial &operator -=(const polynomial &Q);
    /// Add Q to P.
    polynomial operator +(const polynomial &Q) const
      { polynomial R = *this; R += Q; return R; }
    /// Subtract Q from P.
    polynomial operator -(const polynomial &Q) const
      { polynomial R = *this; R -= Q; return R; }
    polynomial operator -() const;
    /// Multiply P with Q. P contains the result.
    polynomial &operator *=(const polynomial &Q);
    /// Multiply P with Q.
    polynomial operator *(const polynomial &Q) const;
    /** Product of P and Q considering that variables of Q come after
     * variables of P. P contains the result
     */
    void direct_product(const polynomial &Q);
    /// Multiply P with the scalar a. P contains the result.
    polynomial &operator *=(const T &e);
    /// Multiply P with the scalar a.
    polynomial operator *(const T &e) const;
    /// Divide P with the scalar a. P contains the result.
    polynomial &operator /=(const T &e);
    /// Divide P with the scalar a.
    polynomial operator /(const T &e) const
      { polynomial res = *this; res /= e; return res; }
    /// operator ==.
    bool operator ==(const polynomial &Q) const;
    /// operator !=.
    bool operator !=(const polynomial &Q) const
    { return !(operator ==(*this,Q)); }
    /// Derivative of P with respect to the variable k. P contains the result.
    void derivative(short_type k);
    /// Makes P = 1.
    void one() { change_degree(0); (*this)[0] = T(1); }
    void clear() { change_degree(0); (*this)[0] = T(0); }
    bool is_zero()
    { return(this->real_degree()==0) && (this->size()==0 || (*this)[0]==T(0)); }
    template <typename ITER> T horner(power_index &mi, short_type k,
                                      short_type de, const ITER &it) const;
    /** Evaluate the polynomial. "it" is an iterator pointing to the list
     * of variables. A Horner scheme is used.
     */
    template <typename ITER> T eval(const ITER &it) const;

    /// Constructor.
    polynomial() : std::vector<T>(1)
      { n = 0; d = 0; (*this)[0] = 0.0; }
    /// Constructor.
    polynomial(short_type dim_, short_type degree_);
    /// Constructor for the polynomial 'x' (k=0), 'y' (k=1), 'z' (k=2) etc.
    polynomial(short_type dim_, short_type degree_, short_type k);
  };


  template<typename T> polynomial<T>::polynomial(short_type nn, short_type dd)
    : std::vector<T>(alpha(nn,dd))
  { n = nn; d = dd; std::fill(this->begin(), this->end(), T(0)); }

  template<typename T> polynomial<T>::polynomial(short_type nn,
                                                 short_type dd, short_type k)
    : std::vector<T>(alpha(nn,dd)) {
    n = nn; d = std::max(short_type(1), dd);
    std::fill(this->begin(), this->end(), T(0));
    (*this)[k+1] = T(1);
  }

  template<typename T>
  polynomial<T> polynomial<T>::operator *(const polynomial &Q) const
  { polynomial res = *this; res *= Q; return res; }

  template<typename T>
  bool polynomial<T>::operator ==(const polynomial &Q) const {
    if (dim() != Q.dim()) return false;
    const_iterator it1 = this->begin(), ite1 = this->end();
    const_iterator it2 = Q.begin(), ite2 = Q.end();
    for ( ; it1 != ite1 && it2 != ite2; ++it1, ++it2)
      if (*it1 != *it2) return false;
    for ( ; it1 != ite1; ++it1) if (*it1 != T(0)) return false;
    for ( ; it2 != ite2; ++it2) if (*it2 != T(0)) return false;
    return true;
  }

  template<typename T>
  polynomial<T> polynomial<T>::operator *(const T &e) const
  { polynomial res = *this; res *= e; return res; }

  template<typename T> short_type polynomial<T>::real_degree() const {
    const_reverse_iterator it = this->rbegin(), ite = this->rend();
    size_type l = this->size();
    for ( ; it != ite; ++it, --l) { if (*it != T(0)) break; }
    short_type dd = degree();
    while (dd > 0 && alpha(n, short_type(dd-1)) > l) --dd;
    return dd;
  }

  template<typename T> void polynomial<T>::change_degree(short_type dd) {
    this->resize(alpha(n,dd));
    if (dd > d) std::fill(this->begin() + alpha(n,d), this->end(), T(0));
    d = dd;
  }

  template<typename T>
  void polynomial<T>::add_monomial(const T &coeff, const power_index &power) {
    size_type i = power.global_index();
    GMM_ASSERT2(n == power.size(), "dimensions mismatch");
    if (i >= this->size()) { change_degree(power.degree()); }
    ((*this)[i]) += coeff;
  }

  template<typename T>
  polynomial<T> &polynomial<T>::operator +=(const polynomial &Q) {
    GMM_ASSERT2(Q.dim() == dim(), "dimensions mismatch");

    if (Q.degree() > degree()) change_degree(Q.degree());
    iterator it = this->begin();
    const_iterator itq = Q.begin(), ite = Q.end();
    for ( ; itq != ite; ++itq, ++it) *it += *itq;
    return *this;
  }

  template<typename T>
  polynomial<T> &polynomial<T>::operator -=(const polynomial &Q) {
    GMM_ASSERT2(Q.dim() == dim() && dim() != 0, "dimensions mismatch");

    if (Q.degree() > degree()) change_degree(Q.degree());
    iterator it = this->begin();
    const_iterator itq = Q.begin(), ite = Q.end();
    for ( ; itq != ite; ++itq, ++it) *it -= *itq;
    return *this;
  }

  template<typename T>
  polynomial<T> polynomial<T>::operator -() const {
    polynomial<T> Q = *this;
    iterator itq = Q.begin(), ite = Q.end();
    for ( ; itq != ite; ++itq) *itq = -(*itq);
    return Q;
  }

  template<typename T>
  polynomial<T> &polynomial<T>::operator *=(const polynomial &Q) {
    GMM_ASSERT2(Q.dim() == dim(), "dimensions mismatch");

    polynomial aux = *this;
    change_degree(0); (*this)[0] = T(0);

    power_index miq(Q.dim()), mia(dim()), mitot(dim());
    if (dim() > 0) miq[dim()-1] = Q.degree();
    const_reverse_iterator itq = Q.rbegin(), ite = Q.rend();
    for ( ; itq != ite; ++itq, --miq) {
      if (*itq != T(0)) {
        reverse_iterator ita = aux.rbegin(), itae = aux.rend();
        std::fill(mia.begin(), mia.end(), 0);
        if (dim() > 0) mia[dim()-1] = aux.degree();
        for ( ; ita != itae; ++ita, --mia)
          if (*ita != T(0)) {
            power_index::iterator mita = mia.begin(), mitq = miq.begin();
            power_index::iterator mit = mitot.begin(), mite = mia.end();
            for ( ; mita != mite; ++mita, ++mitq, ++mit)
              *mit = short_type((*mita) + (*mitq)); /* on pourrait calculer
                                           directement l'index global. */
            //             cerr << "*= : " << *this << ", itq*ita="
            //            << (*itq) * (*ita) << endl;
            //             cerr << " itq = " << *itq << endl;
            //             cerr << " ita = " << *ita << endl;
            add_monomial((*itq) * (*ita), mitot);

          }
      }
    }
    return *this;
  }

  template<typename T>
    void polynomial<T>::direct_product(const polynomial &Q) {
    polynomial aux = *this;

    change_degree(0); n = short_type(n+Q.dim()); (*this)[0] = T(0);

    power_index miq(Q.dim()), mia(aux.dim()), mitot(dim());
    if (Q.dim() > 0) miq[Q.dim()-1] = Q.degree();
    const_reverse_iterator itq = Q.rbegin(), ite = Q.rend();
    for ( ; itq != ite; ++itq, --miq)
      if (*itq != T(0)) {
        reverse_iterator ita = aux.rbegin(), itae = aux.rend();
        std::fill(mia.begin(), mia.end(), 0);
        if (aux.dim() > 0) mia[aux.dim()-1] = aux.degree();
        for ( ; ita != itae; ++ita, --mia)
          if (*ita != T(0)) {
            std::copy(mia.begin(), mia.end(), mitot.begin());
            std::copy(miq.begin(), miq.end(), mitot.begin() + aux.dim());
            add_monomial((*itq) * (*ita), mitot); /* on pourrait calculer
                                           directement l'index global. */
          }
      }
  }

  template<typename T>
    polynomial<T> &polynomial<T>::operator *=(const T &e) {
    iterator it = this->begin(), ite = this->end();
    for ( ; it != ite; ++it) (*it) *= e;
    return *this;
  }

  template<typename T>
    polynomial<T> &polynomial<T>::operator /=(const T &e) {
    iterator it = this->begin(), ite = this->end();
    for ( ; it != ite; ++it) (*it) /= e;
    return *this;
  }

  template<typename T>
  inline void polynomial<T>::derivative(short_type k) {
    GMM_ASSERT2(k < n, "index out of range");

     iterator it = this->begin(), ite = this->end();
     power_index mi(dim());
     for ( ; it != ite; ++it) {
       if ((*it) != T(0) && mi[k] > 0)
         { mi[k]--; (*this)[mi.global_index()] = (*it) * T(mi[k] + 1); mi[k]++; }
       *it = T(0);
       ++mi;
     }
     if (d > 0) change_degree(short_type(d-1));
  }

  template<typename T> template<typename ITER>
  inline T polynomial<T>::horner(power_index &mi, short_type k,
                                 short_type de, const ITER &it) const {
    if (k == 0)
      return (*this)[mi.global_index()];
    else {
      //const std::string vars="xyzuvw ";
      //std::string v{vars[k-1]}, res="";
      T v = (*(it+k-1)), res = T(0);
      for (mi[k-1] = short_type(degree() - de); mi[k-1] != short_type(-1);
           (mi[k-1])--)
        res = horner(mi, short_type(k-1), short_type(de + mi[k-1]), it)
          + v * res;
        //res = horner(mi, short_type(k-1), short_type(de + mi[k-1]), it)
        //     + (res.empty() ? "" : ("+" + v + "*(" + res + ")"));
      mi[k-1] = 0;
      return res;
    }
  }


  template<typename T> template<typename ITER>
  T polynomial<T>::eval(const ITER &it) const {
    /* direct evaluation for common low degree polynomials */
    unsigned deg = degree();
    const_iterator P = this->begin();
    if (deg == 0) return P[0];
    else if (deg == 1) {
      T s = P[0];
      for (size_type i=0; i < dim(); ++i) s += it[i]*P[i+1];
      return s;
    }

    switch (dim()) {
      case 1: {
        T x = it[0];
        if (deg == 2)     return P[0] + x*(P[1] + x*(P[2]));
        if (deg == 3)     return P[0] + x*(P[1] + x*(P[2] + x*(P[3])));
        if (deg == 4)     return P[0] + x*(P[1] + x*(P[2] + x*(P[3] + x*(P[4]))));
        if (deg == 5)     return P[0] + x*(P[1] + x*(P[2] + x*(P[3] + x*(P[4] + x*(P[5])))));
        if (deg == 6)     return P[0] + x*(P[1] + x*(P[2] + x*(P[3] + x*(P[4] + x*(P[5] + x*(P[6]))))));
      } break;
      case 2: {
        T x = it[0];
        T y = it[1];
        if (deg == 2)     return P[0] + x*(P[1] + x*(P[3])) + y*(P[2] + x*(P[4]) + y*(P[5]));
        if (deg == 3)     return P[0] + x*(P[1] + x*(P[3] + x*(P[6]))) + y*(P[2] + x*(P[4] + x*(P[7])) + y*(P[5] + x*(P[8]) + y*(P[9])));
        if (deg == 4)     return P[0] + x*(P[1] + x*(P[3] + x*(P[6] + x*(P[10])))) + y*(P[2] + x*(P[4] + x*(P[7] + x*(P[11]))) + y*(P[5] + x*(P[8] + x*(P[12])) + y*(P[9] + x*(P[13]) + y*(P[14]))));
        if (deg == 5)     return P[0] + x*(P[1] + x*(P[3] + x*(P[6] + x*(P[10] + x*(P[15]))))) + y*(P[2] + x*(P[4] + x*(P[7] + x*(P[11] + x*(P[16])))) + y*(P[5] + x*(P[8] + x*(P[12] + x*(P[17]))) + y*(P[9] + x*(P[13] + x*(P[18])) + y*(P[14] + x*(P[19]) + y*(P[20])))));
        if (deg == 6)     return P[0] + x*(P[1] + x*(P[3] + x*(P[6] + x*(P[10] + x*(P[15] + x*(P[21])))))) + y*(P[2] + x*(P[4] + x*(P[7] + x*(P[11] + x*(P[16] + x*(P[22]))))) + y*(P[5] + x*(P[8] + x*(P[12] + x*(P[17] + x*(P[23])))) + y*(P[9] + x*(P[13] + x*(P[18] + x*(P[24]))) + y*(P[14] + x*(P[19] + x*(P[25])) + y*(P[20] + x*(P[26]) + y*(P[27]))))));
      } break;
      case 3: {
        T x = it[0];
        T y = it[1];
        T z = it[2];
        if (deg == 2)     return P[0] + x*(P[1] + x*(P[4])) + y*(P[2] + x*(P[5]) + y*(P[7])) + z*(P[3] + x*(P[6]) + y*(P[8]) + z*(P[9]));
        if (deg == 3)     return P[0] + x*(P[1] + x*(P[4] + x*(P[10]))) + y*(P[2] + x*(P[5] + x*(P[11])) + y*(P[7] + x*(P[13]) + y*(P[16]))) + z*(P[3] + x*(P[6] + x*(P[12])) + y*(P[8] + x*(P[14]) + y*(P[17])) + z*(P[9] + x*(P[15]) + y*(P[18]) + z*(P[19])));
        if (deg == 4)     return P[0] + x*(P[1] + x*(P[4] + x*(P[10] + x*(P[20])))) + y*(P[2] + x*(P[5] + x*(P[11] + x*(P[21]))) + y*(P[7] + x*(P[13] + x*(P[23])) + y*(P[16] + x*(P[26]) + y*(P[30])))) + z*(P[3] + x*(P[6] + x*(P[12] + x*(P[22]))) + y*(P[8] + x*(P[14] + x*(P[24])) + y*(P[17] + x*(P[27]) + y*(P[31]))) + z*(P[9] + x*(P[15] + x*(P[25])) + y*(P[18] + x*(P[28]) + y*(P[32])) + z*(P[19] + x*(P[29]) + y*(P[33]) + z*(P[34]))));
        if (deg == 5)     return P[0] + x*(P[1] + x*(P[4] + x*(P[10] + x*(P[20] + x*(P[35]))))) + y*(P[2] + x*(P[5] + x*(P[11] + x*(P[21] + x*(P[36])))) + y*(P[7] + x*(P[13] + x*(P[23] + x*(P[38]))) + y*(P[16] + x*(P[26] + x*(P[41])) + y*(P[30] + x*(P[45]) + y*(P[50]))))) + z*(P[3] + x*(P[6] + x*(P[12] + x*(P[22] + x*(P[37])))) + y*(P[8] + x*(P[14] + x*(P[24] + x*(P[39]))) + y*(P[17] + x*(P[27] + x*(P[42])) + y*(P[31] + x*(P[46]) + y*(P[51])))) + z*(P[9] + x*(P[15] + x*(P[25] + x*(P[40]))) + y*(P[18] + x*(P[28] + x*(P[43])) + y*(P[32] + x*(P[47]) + y*(P[52]))) + z*(P[19] + x*(P[29] + x*(P[44])) + y*(P[33] + x*(P[48]) + y*(P[53])) + z*(P[34] + x*(P[49]) + y*(P[54]) + z*(P[55])))));
        if (deg == 6)     return P[0] + x*(P[1] + x*(P[4] + x*(P[10] + x*(P[20] + x*(P[35] + x*(P[56])))))) + y*(P[2] + x*(P[5] + x*(P[11] + x*(P[21] + x*(P[36] + x*(P[57]))))) + y*(P[7] + x*(P[13] + x*(P[23] + x*(P[38] + x*(P[59])))) + y*(P[16] + x*(P[26] + x*(P[41] + x*(P[62]))) + y*(P[30] + x*(P[45] + x*(P[66])) + y*(P[50] + x*(P[71]) + y*(P[77])))))) + z*(P[3] + x*(P[6] + x*(P[12] + x*(P[22] + x*(P[37] + x*(P[58]))))) + y*(P[8] + x*(P[14] + x*(P[24] + x*(P[39] + x*(P[60])))) + y*(P[17] + x*(P[27] + x*(P[42] + x*(P[63]))) + y*(P[31] + x*(P[46] + x*(P[67])) + y*(P[51] + x*(P[72]) + y*(P[78]))))) + z*(P[9] + x*(P[15] + x*(P[25] + x*(P[40] + x*(P[61])))) + y*(P[18] + x*(P[28] + x*(P[43] + x*(P[64]))) + y*(P[32] + x*(P[47] + x*(P[68])) + y*(P[52] + x*(P[73]) + y*(P[79])))) + z*(P[19] + x*(P[29] + x*(P[44] + x*(P[65]))) + y*(P[33] + x*(P[48] + x*(P[69])) + y*(P[53] + x*(P[74]) + y*(P[80]))) + z*(P[34] + x*(P[49] + x*(P[70])) + y*(P[54] + x*(P[75]) + y*(P[81])) + z*(P[55] + x*(P[76]) + y*(P[82]) + z*(P[83]))))));
      } break;
      case 4: {
        T x = it[0];
        T y = it[1];
        T z = it[2];
        T u = it[3];
        if (deg == 3)
          return P[0]+x*(P[1]+x*(P[5]+x*(P[15])))
                     +y*(P[2]+x*(P[6]+x*(P[16]))
                             +y*(P[9]+x*(P[19])+y*(P[25])))
                     +z*(P[3]+x*(P[7]+x*(P[17]))
                             +y*(P[10]+x*(P[20])+y*(P[26]))
                             +z*(P[12]+x*(P[22])+y*(P[28])+z*(P[31])))
                     +u*(P[4]+x*(P[8]+x*(P[18]))
                             +y*(P[11]+x*(P[21])+y*(P[27]))
                             +z*(P[13]+x*(P[23])+y*(P[29])+z*(P[32]))
                             +u*(P[14]+x*(P[24])+y*(P[30])+z*(P[33])+u*(P[34])));
        if (deg == 4)
          return P[0]+x*(P[1]+x*(P[5]+x*(P[15]+x*(P[35]))))
                     +y*(P[2]+x*(P[6]+x*(P[16]+x*(P[36])))
                             +y*(P[9]+x*(P[19]+x*(P[39]))
                                     +y*(P[25]+x*(P[45])+y*(P[55]))))
                     +z*(P[3]+x*(P[7]+x*(P[17]+x*(P[37])))
                             +y*(P[10]+x*(P[20]+x*(P[40]))
                                      +y*(P[26]+x*(P[46])+y*(P[56])))
                             +z*(P[12]+x*(P[22]+x*(P[42]))
                                      +y*(P[28]+x*(P[48])+y*(P[58]))
                                      +z*(P[31]+x*(P[51])+y*(P[61])+z*(P[65]))))
                     +u*(P[4]+x*(P[8]+x*(P[18]+x*(P[38])))
                             +y*(P[11]+x*(P[21]+x*(P[41]))
                                      +y*(P[27]+x*(P[47])+y*(P[57])))
                             +z*(P[13]+x*(P[23]+x*(P[43]))
                                      +y*(P[29]+x*(P[49])+y*(P[59]))
                                      +z*(P[32]+x*(P[52])+y*(P[62])+z*(P[66])))
                             +u*(P[14]+x*(P[24]+x*(P[44]))
                                      +y*(P[30]+x*(P[50])+y*(P[60]))
                                      +z*(P[33]+x*(P[53])+y*(P[63])+z*(P[67]))
                                      +u*(P[34]+x*(P[54])+y*(P[64])+z*(P[68])+u*(P[69]))));
      } break;
      case 5: {
        T x = it[0];
        T y = it[1];
        T z = it[2];
        T u = it[3];
        T v = it[4];
        if (deg == 4)
          return P[0]+x*(P[1]+x*(P[6]+x*(P[21]+x*(P[56]))))
                     +y*(P[2]+x*(P[7]+x*(P[22]+x*(P[57])))
                             +y*(P[11]+x*(P[26]+x*(P[61]))
                                      +y*(P[36]+x*(P[71])+y*(P[91]))))
                     +z*(P[3]+x*(P[8]+x*(P[23]+x*(P[58])))
                             +y*(P[12]+x*(P[27]+x*(P[62]))
                                      +y*(P[37]+x*(P[72])+y*(P[92])))
                             +z*(P[15]+x*(P[30]+x*(P[65]))
                                      +y*(P[40]+x*(P[75])+y*(P[95]))
                                      +z*(P[46]+x*(P[81])+y*(P[101])+z*(P[111]))))
                     +u*(P[4]+x*(P[9]+x*(P[24]+x*(P[59])))
                             +y*(P[13]+x*(P[28]+x*(P[63]))+y*(P[38]+x*(P[73])+y*(P[93])))
                             +z*(P[16]+x*(P[31]+x*(P[66]))
                                      +y*(P[41]+x*(P[76])+y*(P[96]))
                                      +z*(P[47]+x*(P[82])+y*(P[102])+z*(P[112])))
                             +u*(P[18]+x*(P[33]+x*(P[68]))
                                      +y*(P[43]+x*(P[78])+y*(P[98]))
                                      +z*(P[49]+x*(P[84])+y*(P[104])+z*(P[114]))
                                      +u*(P[52]+x*(P[87])+y*(P[107])+z*(P[117])+u*(P[121]))))
                     +v*(P[5]+x*(P[10]+x*(P[25]+x*(P[60])))
                             +y*(P[14]+x*(P[29]+x*(P[64]))+y*(P[39]+x*(P[74])+y*(P[94])))
                             +z*(P[17]+x*(P[32]+x*(P[67]))
                                      +y*(P[42]+x*(P[77])+y*(P[97]))
                                      +z*(P[48]+x*(P[83])+y*(P[103])+z*(P[113])))
                             +u*(P[19]+x*(P[34]+x*(P[69]))
                                      +y*(P[44]+x*(P[79])+y*(P[99]))
                                      +z*(P[50]+x*(P[85])+y*(P[105])+z*(P[115]))
                                      +u*(P[53]+x*(P[88])+y*(P[108])+z*(P[118])+u*(P[122])))
                             +v*(P[20]+x*(P[35]+x*(P[70]))
                                      +y*(P[45]+x*(P[80])+y*(P[100]))
                                      +z*(P[51]+x*(P[86])+y*(P[106])+z*(P[116]))
                                      +u*(P[54]+x*(P[89])+y*(P[109])+z*(P[119])+u*(P[123]))
                                      +v*(P[55]+x*(P[90])+y*(P[110])+z*(P[120])+u*(P[124])+v*(P[125]))));
        if (deg == 5)
          return P[0]+x*(P[1]+x*(P[6]+x*(P[21]+x*(P[56]+x*(P[126])))))
                     +y*(P[2]+x*(P[7]+x*(P[22]+x*(P[57]+x*(P[127]))))
                             +y*(P[11]+x*(P[26]+x*(P[61]+x*(P[131])))
                                      +y*(P[36]+x*(P[71]+x*(P[141]))
                                               +y*(P[91]+x*(P[161])+y*(P[196])))))
                     +z*(P[3]+x*(P[8]+x*(P[23]+x*(P[58]+x*(P[128]))))
                             +y*(P[12]+x*(P[27]+x*(P[62]+x*(P[132])))
                                      +y*(P[37]+x*(P[72]+x*(P[142]))
                                               +y*(P[92]+x*(P[162])+y*(P[197]))))
                             +z*(P[15]+x*(P[30]+x*(P[65]+x*(P[135])))
                                      +y*(P[40]+x*(P[75]+x*(P[145]))
                                               +y*(P[95]+x*(P[165])+y*(P[200])))
                                      +z*(P[46]+x*(P[81]+x*(P[151]))
                                               +y*(P[101]+x*(P[171])+y*(P[206]))
                                               +z*(P[111]+x*(P[181])+y*(P[216])+z*(P[231])))))
                     +u*(P[4]+x*(P[9]+x*(P[24]+x*(P[59]+x*(P[129]))))
                             +y*(P[13]+x*(P[28]+x*(P[63]+x*(P[133])))
                                      +y*(P[38]+x*(P[73]+x*(P[143]))
                                               +y*(P[93]+x*(P[163])+y*(P[198]))))
                             +z*(P[16]+x*(P[31]+x*(P[66]+x*(P[136])))
                                      +y*(P[41]+x*(P[76]+x*(P[146]))
                                               +y*(P[96]+x*(P[166])+y*(P[201])))
                                      +z*(P[47]+x*(P[82]+x*(P[152]))
                                               +y*(P[102]+x*(P[172])+y*(P[207]))
                                               +z*(P[112]+x*(P[182])+y*(P[217])+z*(P[232]))))
                             +u*(P[18]+x*(P[33]+x*(P[68]+x*(P[138])))
                                      +y*(P[43]+x*(P[78]+x*(P[148]))
                                               +y*(P[98]+x*(P[168])+y*(P[203])))
                                      +z*(P[49]+x*(P[84]+x*(P[154]))
                                               +y*(P[104]+x*(P[174])+y*(P[209]))
                                               +z*(P[114]+x*(P[184])+y*(P[219])+z*(P[234])))
                                      +u*(P[52]+x*(P[87]+x*(P[157]))
                                               +y*(P[107]+x*(P[177])+y*(P[212]))
                                               +z*(P[117]+x*(P[187])+y*(P[222])+z*(P[237]))
                                               +u*(P[121]+x*(P[191])+y*(P[226])+z*(P[241])+u*(P[246])))))
                     +v*(P[5]+x*(P[10]+x*(P[25]+x*(P[60]+x*(P[130]))))
                             +y*(P[14]+x*(P[29]+x*(P[64]+x*(P[134])))
                                      +y*(P[39]+x*(P[74]+x*(P[144]))
                                               +y*(P[94]+x*(P[164])+y*(P[199]))))
                             +z*(P[17]+x*(P[32]+x*(P[67]+x*(P[137])))
                                      +y*(P[42]+x*(P[77]+x*(P[147]))
                                               +y*(P[97]+x*(P[167])+y*(P[202])))
                                      +z*(P[48]+x*(P[83]+x*(P[153]))
                                               +y*(P[103]+x*(P[173])+y*(P[208]))
                                               +z*(P[113]+x*(P[183])+y*(P[218])+z*(P[233]))))
                             +u*(P[19]+x*(P[34]+x*(P[69]+x*(P[139])))
                                      +y*(P[44]+x*(P[79]+x*(P[149]))
                                               +y*(P[99]+x*(P[169])+y*(P[204])))
                                      +z*(P[50]+x*(P[85]+x*(P[155]))
                                               +y*(P[105]+x*(P[175])+y*(P[210]))
                                               +z*(P[115]+x*(P[185])+y*(P[220])+z*(P[235])))
                                      +u*(P[53]+x*(P[88]+x*(P[158]))
                                               +y*(P[108]+x*(P[178])+y*(P[213]))
                                               +z*(P[118]+x*(P[188])+y*(P[223])+z*(P[238]))
                                               +u*(P[122]+x*(P[192])+y*(P[227])+z*(P[242])+u*(P[247]))))
                             +v*(P[20]+x*(P[35]+x*(P[70]+x*(P[140])))
                                      +y*(P[45]+x*(P[80]+x*(P[150]))
                                               +y*(P[100]+x*(P[170])+y*(P[205])))
                                      +z*(P[51]+x*(P[86]+x*(P[156]))
                                               +y*(P[106]+x*(P[176])+y*(P[211]))
                                               +z*(P[116]+x*(P[186])+y*(P[221])+z*(P[236])))
                                      +u*(P[54]+x*(P[89]+x*(P[159]))
                                               +y*(P[109]+x*(P[179])+y*(P[214]))
                                               +z*(P[119]+x*(P[189])+y*(P[224])+z*(P[239]))
                                               +u*(P[123]+x*(P[193])+y*(P[228])+z*(P[243])+u*(P[248])))
                                      +v*(P[55]+x*(P[90]+x*(P[160]))
                                               +y*(P[110]+x*(P[180])+y*(P[215]))
                                               +z*(P[120]+x*(P[190])+y*(P[225])+z*(P[240]))
                                               +u*(P[124]+x*(P[194])+y*(P[229])+z*(P[244])+u*(P[249]))
                                               +v*(P[125]+x*(P[195])+y*(P[230])+z*(P[245])+u*(P[250])+v*(P[251])))));
      } break;
      case 6: {
        T x = it[0];
        T y = it[1];
        T z = it[2];
        T u = it[3];
        T v = it[4];
        T w = it[5];
        if (deg == 5)
          return P[0]+x*(P[1]+x*(P[7]+x*(P[28]+x*(P[84]+x*(P[210])))))
                     +y*(P[2]+x*(P[8]+x*(P[29]+x*(P[85]+x*(P[211]))))
                             +y*(P[13]+x*(P[34]+x*(P[90]+x*(P[216])))
                                      +y*(P[49]+x*(P[105]+x*(P[231]))
                                               +y*(P[140]+x*(P[266])+y*(P[336])))))
                     +z*(P[3]+x*(P[9]+x*(P[30]+x*(P[86]+x*(P[212]))))
                             +y*(P[14]+x*(P[35]+x*(P[91]+x*(P[217])))
                                      +y*(P[50]+x*(P[106]+x*(P[232]))
                                               +y*(P[141]+x*(P[267])+y*(P[337]))))
                             +z*(P[18]+x*(P[39]+x*(P[95]+x*(P[221])))
                                      +y*(P[54]+x*(P[110]+x*(P[236]))
                                               +y*(P[145]+x*(P[271])+y*(P[341])))
                                      +z*(P[64]+x*(P[120]+x*(P[246]))
                                               +y*(P[155]+x*(P[281])+y*(P[351]))
                                               +z*(P[175]+x*(P[301])+y*(P[371])+z*(P[406])))))
                     +u*(P[4]+x*(P[10]+x*(P[31]+x*(P[87]+x*(P[213]))))
                             +y*(P[15]+x*(P[36]+x*(P[92]+x*(P[218])))
                                      +y*(P[51]+x*(P[107]+x*(P[233]))+y*(P[142]+x*(P[268])+y*(P[338]))))
                             +z*(P[19]+x*(P[40]+x*(P[96]+x*(P[222])))
                                      +y*(P[55]+x*(P[111]+x*(P[237]))
                                               +y*(P[146]+x*(P[272])+y*(P[342])))
                                      +z*(P[65]+x*(P[121]+x*(P[247]))
                                               +y*(P[156]+x*(P[282])+y*(P[352]))
                                               +z*(P[176]+x*(P[302])+y*(P[372])+z*(P[407]))))
                             +u*(P[22]+x*(P[43]+x*(P[99]+x*(P[225])))
                                      +y*(P[58]+x*(P[114]+x*(P[240]))+y*(P[149]+x*(P[275])+y*(P[345])))
                                      +z*(P[68]+x*(P[124]+x*(P[250]))
                                               +y*(P[159]+x*(P[285])+y*(P[355]))
                                               +z*(P[179]+x*(P[305])+y*(P[375])+z*(P[410])))
                                      +u*(P[74]+x*(P[130]+x*(P[256]))
                                               +y*(P[165]+x*(P[291])+y*(P[361]))
                                               +z*(P[185]+x*(P[311])+y*(P[381])+z*(P[416]))
                                               +u*(P[195]+x*(P[321])+y*(P[391])+z*(P[426])+u*(P[441])))))
                     +v*(P[5]+x*(P[11]+x*(P[32]+x*(P[88]+x*(P[214]))))
                             +y*(P[16]+x*(P[37]+x*(P[93]+x*(P[219])))
                                      +y*(P[52]+x*(P[108]+x*(P[234]))
                                               +y*(P[143]+x*(P[269])+y*(P[339]))))
                             +z*(P[20]+x*(P[41]+x*(P[97]+x*(P[223])))
                                      +y*(P[56]+x*(P[112]+x*(P[238]))
                                               +y*(P[147]+x*(P[273])+y*(P[343])))
                                      +z*(P[66]+x*(P[122]+x*(P[248]))
                                               +y*(P[157]+x*(P[283])+y*(P[353]))
                                               +z*(P[177]+x*(P[303])+y*(P[373])+z*(P[408]))))
                             +u*(P[23]+x*(P[44]+x*(P[100]+x*(P[226])))
                                      +y*(P[59]+x*(P[115]+x*(P[241]))
                                               +y*(P[150]+x*(P[276])+y*(P[346])))
                                      +z*(P[69]+x*(P[125]+x*(P[251]))
                                               +y*(P[160]+x*(P[286])+y*(P[356]))
                                               +z*(P[180]+x*(P[306])+y*(P[376])+z*(P[411])))
                                      +u*(P[75]+x*(P[131]+x*(P[257]))
                                               +y*(P[166]+x*(P[292])+y*(P[362]))
                                               +z*(P[186]+x*(P[312])+y*(P[382])+z*(P[417]))
                                               +u*(P[196]+x*(P[322])+y*(P[392])+z*(P[427])+u*(P[442]))))
                             +v*(P[25]+x*(P[46]+x*(P[102]+x*(P[228])))
                                      +y*(P[61]+x*(P[117]+x*(P[243]))
                                               +y*(P[152]+x*(P[278])+y*(P[348])))
                                      +z*(P[71]+x*(P[127]+x*(P[253]))
                                               +y*(P[162]+x*(P[288])+y*(P[358]))
                                               +z*(P[182]+x*(P[308])+y*(P[378])+z*(P[413])))
                                      +u*(P[77]+x*(P[133]+x*(P[259]))
                                               +y*(P[168]+x*(P[294])+y*(P[364]))
                                               +z*(P[188]+x*(P[314])+y*(P[384])+z*(P[419]))
                                               +u*(P[198]+x*(P[324])+y*(P[394])+z*(P[429])+u*(P[444])))
                                      +v*(P[80]+x*(P[136]+x*(P[262]))
                                               +y*(P[171]+x*(P[297])+y*(P[367]))
                                               +z*(P[191]+x*(P[317])+y*(P[387])+z*(P[422]))
                                               +u*(P[201]+x*(P[327])+y*(P[397])+z*(P[432])+u*(P[447]))
                                               +v*(P[205]+x*(P[331])+y*(P[401])+z*(P[436])+u*(P[451])+v*(P[456])))))
                     +w*(P[6]+x*(P[12]+x*(P[33]+x*(P[89]+x*(P[215]))))
                             +y*(P[17]+x*(P[38]+x*(P[94]+x*(P[220])))
                                      +y*(P[53]+x*(P[109]+x*(P[235]))
                                               +y*(P[144]+x*(P[270])+y*(P[340]))))
                             +z*(P[21]+x*(P[42]+x*(P[98]+x*(P[224])))
                                      +y*(P[57]+x*(P[113]+x*(P[239]))
                                               +y*(P[148]+x*(P[274])+y*(P[344])))
                                      +z*(P[67]+x*(P[123]+x*(P[249]))
                                               +y*(P[158]+x*(P[284])+y*(P[354]))
                                               +z*(P[178]+x*(P[304])+y*(P[374])+z*(P[409]))))
                             +u*(P[24]+x*(P[45]+x*(P[101]+x*(P[227])))
                                      +y*(P[60]+x*(P[116]+x*(P[242]))
                                               +y*(P[151]+x*(P[277])+y*(P[347])))
                                      +z*(P[70]+x*(P[126]+x*(P[252]))
                                               +y*(P[161]+x*(P[287])+y*(P[357]))
                                               +z*(P[181]+x*(P[307])+y*(P[377])+z*(P[412])))
                                      +u*(P[76]+x*(P[132]+x*(P[258]))
                                               +y*(P[167]+x*(P[293])+y*(P[363]))
                                               +z*(P[187]+x*(P[313])+y*(P[383])+z*(P[418]))
                                               +u*(P[197]+x*(P[323])+y*(P[393])+z*(P[428])+u*(P[443]))))
                             +v*(P[26]+x*(P[47]+x*(P[103]+x*(P[229])))
                                      +y*(P[62]+x*(P[118]+x*(P[244]))
                                               +y*(P[153]+x*(P[279])+y*(P[349])))
                                      +z*(P[72]+x*(P[128]+x*(P[254]))
                                               +y*(P[163]+x*(P[289])+y*(P[359]))
                                               +z*(P[183]+x*(P[309])+y*(P[379])+z*(P[414])))
                                      +u*(P[78]+x*(P[134]+x*(P[260]))
                                               +y*(P[169]+x*(P[295])+y*(P[365]))
                                               +z*(P[189]+x*(P[315])+y*(P[385])+z*(P[420]))
                                               +u*(P[199]+x*(P[325])+y*(P[395])+z*(P[430])+u*(P[445])))
                                      +v*(P[81]+x*(P[137]+x*(P[263]))
                                               +y*(P[172]+x*(P[298])+y*(P[368]))
                                               +z*(P[192]+x*(P[318])+y*(P[388])+z*(P[423]))
                                               +u*(P[202]+x*(P[328])+y*(P[398])+z*(P[433])+u*(P[448]))
                                               +v*(P[206]+x*(P[332])+y*(P[402])+z*(P[437])+u*(P[452])+v*(P[457]))))
                             +w*(P[27]+x*(P[48]+x*(P[104]+x*(P[230])))
                                      +y*(P[63]+x*(P[119]+x*(P[245]))
                                               +y*(P[154]+x*(P[280])+y*(P[350])))
                                      +z*(P[73]+x*(P[129]+x*(P[255]))
                                               +y*(P[164]+x*(P[290])+y*(P[360]))
                                               +z*(P[184]+x*(P[310])+y*(P[380])+z*(P[415])))
                                      +u*(P[79]+x*(P[135]+x*(P[261]))
                                               +y*(P[170]+x*(P[296])+y*(P[366]))
                                               +z*(P[190]+x*(P[316])+y*(P[386])+z*(P[421]))
                                               +u*(P[200]+x*(P[326])+y*(P[396])+z*(P[431])+u*(P[446])))
                                      +v*(P[82]+x*(P[138]+x*(P[264]))
                                               +y*(P[173]+x*(P[299])+y*(P[369]))
                                               +z*(P[193]+x*(P[319])+y*(P[389])+z*(P[424]))
                                               +u*(P[203]+x*(P[329])+y*(P[399])+z*(P[434])+u*(P[449]))
                                               +v*(P[207]+x*(P[333])+y*(P[403])+z*(P[438])+u*(P[453])+v*(P[458])))
                                      +w*(P[83]+x*(P[139]+x*(P[265]))
                                               +y*(P[174]+x*(P[300])+y*(P[370]))
                                               +z*(P[194]+x*(P[320])+y*(P[390])+z*(P[425]))
                                               +u*(P[204]+x*(P[330])+y*(P[400])+z*(P[435])+u*(P[450]))
                                               +v*(P[208]+x*(P[334])+y*(P[404])+z*(P[439])+u*(P[454])+v*(P[459]))
                                               +w*(P[209]+x*(P[335])+y*(P[405])+z*(P[440])+u*(P[455])+v*(P[460])+w*(P[461])))));
        if (deg == 6)
          return P[0]+x*(P[1]+x*(P[7]+x*(P[28]+x*(P[84]+x*(P[210]+x*(P[462]))))))
                     +y*(P[2]+x*(P[8]+x*(P[29]+x*(P[85]+x*(P[211]+x*(P[463])))))
                             +y*(P[13]+x*(P[34]+x*(P[90]+x*(P[216]+x*(P[468]))))
                                      +y*(P[49]+x*(P[105]+x*(P[231]+x*(P[483])))
                                               +y*(P[140]+x*(P[266]+x*(P[518]))
                                                         +y*(P[336]+x*(P[588])+y*(P[714]))))))
                     +z*(P[3]+x*(P[9]+x*(P[30]+x*(P[86]+x*(P[212]+x*(P[464])))))
                             +y*(P[14]+x*(P[35]+x*(P[91]+x*(P[217]+x*(P[469]))))
                                      +y*(P[50]+x*(P[106]+x*(P[232]+x*(P[484])))
                                               +y*(P[141]+x*(P[267]+x*(P[519]))
                                                         +y*(P[337]+x*(P[589])+y*(P[715])))))
                             +z*(P[18]+x*(P[39]+x*(P[95]+x*(P[221]+x*(P[473]))))
                                      +y*(P[54]+x*(P[110]+x*(P[236]+x*(P[488])))
                                               +y*(P[145]+x*(P[271]+x*(P[523]))
                                                         +y*(P[341]+x*(P[593])+y*(P[719]))))
                                      +z*(P[64]+x*(P[120]+x*(P[246]+x*(P[498])))
                                               +y*(P[155]+x*(P[281]+x*(P[533]))
                                                         +y*(P[351]+x*(P[603])+y*(P[729])))
                                               +z*(P[175]+x*(P[301]+x*(P[553]))
                                                         +y*(P[371]+x*(P[623])+y*(P[749]))
                                                         +z*(P[406]+x*(P[658])+y*(P[784])+z*(P[840]))))))
                     +u*(P[4]+x*(P[10]+x*(P[31]+x*(P[87]+x*(P[213]+x*(P[465])))))
                             +y*(P[15]+x*(P[36]+x*(P[92]+x*(P[218]+x*(P[470]))))
                                      +y*(P[51]+x*(P[107]+x*(P[233]+x*(P[485])))
                                               +y*(P[142]+x*(P[268]+x*(P[520]))
                                                         +y*(P[338]+x*(P[590])+y*(P[716])))))
                             +z*(P[19]+x*(P[40]+x*(P[96]+x*(P[222]+x*(P[474]))))
                                      +y*(P[55]+x*(P[111]+x*(P[237]+x*(P[489])))
                                               +y*(P[146]+x*(P[272]+x*(P[524]))
                                                         +y*(P[342]+x*(P[594])+y*(P[720]))))
                                      +z*(P[65]+x*(P[121]+x*(P[247]+x*(P[499])))
                                               +y*(P[156]+x*(P[282]+x*(P[534]))
                                                         +y*(P[352]+x*(P[604])+y*(P[730])))
                                               +z*(P[176]+x*(P[302]+x*(P[554]))
                                                         +y*(P[372]+x*(P[624])+y*(P[750]))
                                                         +z*(P[407]+x*(P[659])+y*(P[785])+z*(P[841])))))
                             +u*(P[22]+x*(P[43]+x*(P[99]+x*(P[225]+x*(P[477]))))
                                      +y*(P[58]+x*(P[114]+x*(P[240]+x*(P[492])))
                                               +y*(P[149]+x*(P[275]+x*(P[527]))
                                                         +y*(P[345]+x*(P[597])+y*(P[723]))))
                                      +z*(P[68]+x*(P[124]+x*(P[250]+x*(P[502])))
                                               +y*(P[159]+x*(P[285]+x*(P[537]))
                                                         +y*(P[355]+x*(P[607])+y*(P[733])))
                                               +z*(P[179]+x*(P[305]+x*(P[557]))
                                                         +y*(P[375]+x*(P[627])+y*(P[753]))
                                                         +z*(P[410]+x*(P[662])+y*(P[788])+z*(P[844]))))
                                      +u*(P[74]+x*(P[130]+x*(P[256]+x*(P[508])))
                                               +y*(P[165]+x*(P[291]+x*(P[543]))
                                                         +y*(P[361]+x*(P[613])+y*(P[739])))
                                               +z*(P[185]+x*(P[311]+x*(P[563]))
                                                         +y*(P[381]+x*(P[633])+y*(P[759]))
                                                         +z*(P[416]+x*(P[668])+y*(P[794])+z*(P[850])))
                                               +u*(P[195]+x*(P[321]+x*(P[573]))
                                                         +y*(P[391]+x*(P[643])+y*(P[769]))
                                                         +z*(P[426]+x*(P[678])+y*(P[804])+z*(P[860]))
                                                         +u*(P[441]+x*(P[693])+y*(P[819])+z*(P[875])+u*(P[896]))))))
                     +v*(P[5]+x*(P[11]+x*(P[32]+x*(P[88]+x*(P[214]+x*(P[466])))))
                             +y*(P[16]+x*(P[37]+x*(P[93]+x*(P[219]+x*(P[471]))))
                                      +y*(P[52]+x*(P[108]+x*(P[234]+x*(P[486])))
                                               +y*(P[143]+x*(P[269]+x*(P[521]))
                                                         +y*(P[339]+x*(P[591])+y*(P[717])))))
                             +z*(P[20]+x*(P[41]+x*(P[97]+x*(P[223]+x*(P[475]))))
                                      +y*(P[56]+x*(P[112]+x*(P[238]+x*(P[490])))
                                               +y*(P[147]+x*(P[273]+x*(P[525]))
                                                         +y*(P[343]+x*(P[595])+y*(P[721]))))
                                      +z*(P[66]+x*(P[122]+x*(P[248]+x*(P[500])))
                                               +y*(P[157]+x*(P[283]+x*(P[535]))
                                                         +y*(P[353]+x*(P[605])+y*(P[731])))
                                               +z*(P[177]+x*(P[303]+x*(P[555]))
                                                         +y*(P[373]+x*(P[625])+y*(P[751]))
                                                         +z*(P[408]+x*(P[660])+y*(P[786])+z*(P[842])))))
                             +u*(P[23]+x*(P[44]+x*(P[100]+x*(P[226]+x*(P[478]))))
                                      +y*(P[59]+x*(P[115]+x*(P[241]+x*(P[493])))
                                               +y*(P[150]+x*(P[276]+x*(P[528]))
                                                         +y*(P[346]+x*(P[598])+y*(P[724]))))
                                      +z*(P[69]+x*(P[125]+x*(P[251]+x*(P[503])))
                                               +y*(P[160]+x*(P[286]+x*(P[538]))
                                                         +y*(P[356]+x*(P[608])+y*(P[734])))
                                               +z*(P[180]+x*(P[306]+x*(P[558]))
                                                         +y*(P[376]+x*(P[628])+y*(P[754]))
                                                         +z*(P[411]+x*(P[663])+y*(P[789])+z*(P[845]))))
                                      +u*(P[75]+x*(P[131]+x*(P[257]+x*(P[509])))
                                               +y*(P[166]+x*(P[292]+x*(P[544]))
                                                         +y*(P[362]+x*(P[614])+y*(P[740])))
                                               +z*(P[186]+x*(P[312]+x*(P[564]))
                                                         +y*(P[382]+x*(P[634])+y*(P[760]))
                                                         +z*(P[417]+x*(P[669])+y*(P[795])+z*(P[851])))
                                               +u*(P[196]+x*(P[322]+x*(P[574]))
                                                         +y*(P[392]+x*(P[644])+y*(P[770]))
                                                         +z*(P[427]+x*(P[679])+y*(P[805])+z*(P[861]))
                                                         +u*(P[442]+x*(P[694])+y*(P[820])+z*(P[876])+u*(P[897])))))
                             +v*(P[25]+x*(P[46]+x*(P[102]+x*(P[228]+x*(P[480]))))
                                      +y*(P[61]+x*(P[117]+x*(P[243]+x*(P[495])))
                                               +y*(P[152]+x*(P[278]+x*(P[530]))
                                                         +y*(P[348]+x*(P[600])+y*(P[726]))))
                                      +z*(P[71]+x*(P[127]+x*(P[253]+x*(P[505])))
                                               +y*(P[162]+x*(P[288]+x*(P[540]))
                                                         +y*(P[358]+x*(P[610])+y*(P[736])))
                                               +z*(P[182]+x*(P[308]+x*(P[560]))
                                                         +y*(P[378]+x*(P[630])+y*(P[756]))
                                                         +z*(P[413]+x*(P[665])+y*(P[791])+z*(P[847]))))
                                      +u*(P[77]+x*(P[133]+x*(P[259]+x*(P[511])))
                                               +y*(P[168]+x*(P[294]+x*(P[546]))
                                                         +y*(P[364]+x*(P[616])+y*(P[742])))
                                               +z*(P[188]+x*(P[314]+x*(P[566]))
                                                         +y*(P[384]+x*(P[636])+y*(P[762]))
                                                         +z*(P[419]+x*(P[671])+y*(P[797])+z*(P[853])))
                                               +u*(P[198]+x*(P[324]+x*(P[576]))
                                                         +y*(P[394]+x*(P[646])+y*(P[772]))
                                                         +z*(P[429]+x*(P[681])+y*(P[807])+z*(P[863]))
                                                         +u*(P[444]+x*(P[696])+y*(P[822])+z*(P[878])+u*(P[899]))))
                                      +v*(P[80]+x*(P[136]+x*(P[262]+x*(P[514])))
                                               +y*(P[171]+x*(P[297]+x*(P[549]))
                                                         +y*(P[367]+x*(P[619])+y*(P[745])))
                                               +z*(P[191]+x*(P[317]+x*(P[569]))
                                                         +y*(P[387]+x*(P[639])+y*(P[765]))
                                                         +z*(P[422]+x*(P[674])+y*(P[800])+z*(P[856])))
                                               +u*(P[201]+x*(P[327]+x*(P[579]))
                                                         +y*(P[397]+x*(P[649])+y*(P[775]))
                                                         +z*(P[432]+x*(P[684])+y*(P[810])+z*(P[866]))
                                                         +u*(P[447]+x*(P[699])+y*(P[825])+z*(P[881])+u*(P[902])))
                                               +v*(P[205]+x*(P[331]+x*(P[583]))
                                                         +y*(P[401]+x*(P[653])+y*(P[779]))
                                                         +z*(P[436]+x*(P[688])+y*(P[814])+z*(P[870]))
                                                         +u*(P[451]+x*(P[703])+y*(P[829])+z*(P[885])+u*(P[906]))
                                                         +v*(P[456]+x*(P[708])+y*(P[834])+z*(P[890])+u*(P[911])+v*(P[917]))))))
                     +w*(P[6]+x*(P[12]+x*(P[33]+x*(P[89]+x*(P[215]+x*(P[467])))))
                             +y*(P[17]+x*(P[38]+x*(P[94]+x*(P[220]+x*(P[472]))))
                                      +y*(P[53]+x*(P[109]+x*(P[235]+x*(P[487])))
                                               +y*(P[144]+x*(P[270]+x*(P[522]))
                                                         +y*(P[340]+x*(P[592])+y*(P[718])))))
                             +z*(P[21]+x*(P[42]+x*(P[98]+x*(P[224]+x*(P[476]))))
                                      +y*(P[57]+x*(P[113]+x*(P[239]+x*(P[491])))
                                               +y*(P[148]+x*(P[274]+x*(P[526]))
                                                         +y*(P[344]+x*(P[596])+y*(P[722]))))
                                      +z*(P[67]+x*(P[123]+x*(P[249]+x*(P[501])))
                                               +y*(P[158]+x*(P[284]+x*(P[536]))
                                                         +y*(P[354]+x*(P[606])+y*(P[732])))
                                               +z*(P[178]+x*(P[304]+x*(P[556]))
                                                         +y*(P[374]+x*(P[626])+y*(P[752]))
                                                         +z*(P[409]+x*(P[661])+y*(P[787])+z*(P[843])))))
                             +u*(P[24]+x*(P[45]+x*(P[101]+x*(P[227]+x*(P[479]))))
                                      +y*(P[60]+x*(P[116]+x*(P[242]+x*(P[494])))
                                               +y*(P[151]+x*(P[277]+x*(P[529]))
                                                         +y*(P[347]+x*(P[599])+y*(P[725]))))
                                      +z*(P[70]+x*(P[126]+x*(P[252]+x*(P[504])))
                                               +y*(P[161]+x*(P[287]+x*(P[539]))
                                                         +y*(P[357]+x*(P[609])+y*(P[735])))
                                               +z*(P[181]+x*(P[307]+x*(P[559]))
                                                         +y*(P[377]+x*(P[629])+y*(P[755]))
                                                         +z*(P[412]+x*(P[664])+y*(P[790])+z*(P[846]))))
                                      +u*(P[76]+x*(P[132]+x*(P[258]+x*(P[510])))
                                               +y*(P[167]+x*(P[293]+x*(P[545]))
                                                         +y*(P[363]+x*(P[615])+y*(P[741])))
                                               +z*(P[187]+x*(P[313]+x*(P[565]))
                                                         +y*(P[383]+x*(P[635])+y*(P[761]))
                                                         +z*(P[418]+x*(P[670])+y*(P[796])+z*(P[852])))
                                               +u*(P[197]+x*(P[323]+x*(P[575]))
                                                         +y*(P[393]+x*(P[645])+y*(P[771]))
                                                         +z*(P[428]+x*(P[680])+y*(P[806])+z*(P[862]))
                                                         +u*(P[443]+x*(P[695])+y*(P[821])+z*(P[877])+u*(P[898])))))
                             +v*(P[26]+x*(P[47]+x*(P[103]+x*(P[229]+x*(P[481]))))
                                      +y*(P[62]+x*(P[118]+x*(P[244]+x*(P[496])))
                                               +y*(P[153]+x*(P[279]+x*(P[531]))
                                                         +y*(P[349]+x*(P[601])+y*(P[727]))))
                                      +z*(P[72]+x*(P[128]+x*(P[254]+x*(P[506])))
                                               +y*(P[163]+x*(P[289]+x*(P[541]))
                                                         +y*(P[359]+x*(P[611])+y*(P[737])))
                                               +z*(P[183]+x*(P[309]+x*(P[561]))
                                                         +y*(P[379]+x*(P[631])+y*(P[757]))
                                                         +z*(P[414]+x*(P[666])+y*(P[792])+z*(P[848]))))
                                      +u*(P[78]+x*(P[134]+x*(P[260]+x*(P[512])))
                                               +y*(P[169]+x*(P[295]+x*(P[547]))
                                                         +y*(P[365]+x*(P[617])+y*(P[743])))
                                               +z*(P[189]+x*(P[315]+x*(P[567]))
                                                         +y*(P[385]+x*(P[637])+y*(P[763]))
                                                         +z*(P[420]+x*(P[672])+y*(P[798])+z*(P[854])))
                                               +u*(P[199]+x*(P[325]+x*(P[577]))
                                                         +y*(P[395]+x*(P[647])+y*(P[773]))
                                                         +z*(P[430]+x*(P[682])+y*(P[808])+z*(P[864]))
                                                         +u*(P[445]+x*(P[697])+y*(P[823])+z*(P[879])+u*(P[900]))))
                                      +v*(P[81]+x*(P[137]+x*(P[263]+x*(P[515])))
                                               +y*(P[172]+x*(P[298]+x*(P[550]))
                                                         +y*(P[368]+x*(P[620])+y*(P[746])))
                                               +z*(P[192]+x*(P[318]+x*(P[570]))
                                                         +y*(P[388]+x*(P[640])+y*(P[766]))
                                                         +z*(P[423]+x*(P[675])+y*(P[801])+z*(P[857])))
                                               +u*(P[202]+x*(P[328]+x*(P[580]))
                                                         +y*(P[398]+x*(P[650])+y*(P[776]))
                                                         +z*(P[433]+x*(P[685])+y*(P[811])+z*(P[867]))
                                                         +u*(P[448]+x*(P[700])+y*(P[826])+z*(P[882])+u*(P[903])))
                                               +v*(P[206]+x*(P[332]+x*(P[584]))
                                                         +y*(P[402]+x*(P[654])+y*(P[780]))
                                                         +z*(P[437]+x*(P[689])+y*(P[815])+z*(P[871]))
                                                         +u*(P[452]+x*(P[704])+y*(P[830])+z*(P[886])+u*(P[907]))
                                                         +v*(P[457]+x*(P[709])+y*(P[835])+z*(P[891])+u*(P[912])+v*(P[918])))))
                             +w*(P[27]+x*(P[48]+x*(P[104]+x*(P[230]+x*(P[482]))))
                                      +y*(P[63]+x*(P[119]+x*(P[245]+x*(P[497])))
                                               +y*(P[154]+x*(P[280]+x*(P[532]))
                                                         +y*(P[350]+x*(P[602])+y*(P[728]))))
                                      +z*(P[73]+x*(P[129]+x*(P[255]+x*(P[507])))
                                               +y*(P[164]+x*(P[290]+x*(P[542]))
                                                         +y*(P[360]+x*(P[612])+y*(P[738])))
                                               +z*(P[184]+x*(P[310]+x*(P[562]))
                                                         +y*(P[380]+x*(P[632])+y*(P[758]))
                                                         +z*(P[415]+x*(P[667])+y*(P[793])+z*(P[849]))))
                                      +u*(P[79]+x*(P[135]+x*(P[261]+x*(P[513])))
                                               +y*(P[170]+x*(P[296]+x*(P[548]))
                                                         +y*(P[366]+x*(P[618])+y*(P[744])))
                                               +z*(P[190]+x*(P[316]+x*(P[568]))
                                                         +y*(P[386]+x*(P[638])+y*(P[764]))
                                                         +z*(P[421]+x*(P[673])+y*(P[799])+z*(P[855])))
                                               +u*(P[200]+x*(P[326]+x*(P[578]))
                                                         +y*(P[396]+x*(P[648])+y*(P[774]))
                                                         +z*(P[431]+x*(P[683])+y*(P[809])+z*(P[865]))
                                                         +u*(P[446]+x*(P[698])+y*(P[824])+z*(P[880])+u*(P[901]))))
                                      +v*(P[82]+x*(P[138]+x*(P[264]+x*(P[516])))
                                               +y*(P[173]+x*(P[299]+x*(P[551]))
                                                         +y*(P[369]+x*(P[621])+y*(P[747])))
                                               +z*(P[193]+x*(P[319]+x*(P[571]))
                                                         +y*(P[389]+x*(P[641])+y*(P[767]))
                                                         +z*(P[424]+x*(P[676])+y*(P[802])+z*(P[858])))
                                               +u*(P[203]+x*(P[329]+x*(P[581]))
                                                         +y*(P[399]+x*(P[651])+y*(P[777]))
                                                         +z*(P[434]+x*(P[686])+y*(P[812])+z*(P[868]))
                                                         +u*(P[449]+x*(P[701])+y*(P[827])+z*(P[883])+u*(P[904])))
                                               +v*(P[207]+x*(P[333]+x*(P[585]))
                                                         +y*(P[403]+x*(P[655])+y*(P[781]))
                                                         +z*(P[438]+x*(P[690])+y*(P[816])+z*(P[872]))
                                                         +u*(P[453]+x*(P[705])+y*(P[831])+z*(P[887])+u*(P[908]))
                                                         +v*(P[458]+x*(P[710])+y*(P[836])+z*(P[892])+u*(P[913])+v*(P[919]))))
                                      +w*(P[83]+x*(P[139]+x*(P[265]+x*(P[517])))
                                               +y*(P[174]+x*(P[300]+x*(P[552]))
                                                         +y*(P[370]+x*(P[622])+y*(P[748])))
                                               +z*(P[194]+x*(P[320]+x*(P[572]))
                                                         +y*(P[390]+x*(P[642])+y*(P[768]))
                                                         +z*(P[425]+x*(P[677])+y*(P[803])+z*(P[859])))
                                               +u*(P[204]+x*(P[330]+x*(P[582]))
                                                         +y*(P[400]+x*(P[652])+y*(P[778]))
                                                         +z*(P[435]+x*(P[687])+y*(P[813])+z*(P[869]))
                                                         +u*(P[450]+x*(P[702])+y*(P[828])+z*(P[884])+u*(P[905])))
                                               +v*(P[208]+x*(P[334]+x*(P[586]))
                                                         +y*(P[404]+x*(P[656])+y*(P[782]))
                                                         +z*(P[439]+x*(P[691])+y*(P[817])+z*(P[873]))
                                                         +u*(P[454]+x*(P[706])+y*(P[832])+z*(P[888])+u*(P[909]))
                                                         +v*(P[459]+x*(P[711])+y*(P[837])+z*(P[893])+u*(P[914])+v*(P[920])))
                                               +w*(P[209]+x*(P[335]+x*(P[587]))
                                                         +y*(P[405]+x*(P[657])+y*(P[783]))
                                                         +z*(P[440]+x*(P[692])+y*(P[818])+z*(P[874]))
                                                         +u*(P[455]+x*(P[707])+y*(P[833])+z*(P[889])+u*(P[910]))
                                                         +v*(P[460]+x*(P[712])+y*(P[838])+z*(P[894])+u*(P[915])+v*(P[921]))
                                                         +w*(P[461]+x*(P[713])+y*(P[839])+z*(P[895])+u*(P[916])+v*(P[922])+w*(P[923]))))));
      } break;
    }

    /*
    switch (deg) {
      case 0: return (*this)[0];
      case 1: {
        T s = (*this)[0];
        for (size_type i=0; i < dim(); ++i) s += it[i]*(*this)[i+1];
        return s;
      }
      case 2:
      case 3: {
        if (dim() == 1) {
          const T &x = it[0];
          if      (deg == 2) return p[0] + x*(p[1] + x*p[2]);
          else if (deg == 3) return p[0] + x*(p[1] + x*(p[2]+x*p[3]));
        } else if (dim() == 2) {
          const T &x = it[0];
          const T &y = it[1];
          if      (deg == 2)
            return p[0] + p[1]*x + p[2]*y + p[3]*x*x + p[4]*x*y + p[5]*y*y;
          else if (deg == 3)
            return p[0] + p[1]*x + p[2]*y + p[3]*x*x + p[4]*x*y + p[5]*y*y +
              p[6]*x*x*x + p[7]*x*x*y + p[8]*x*y*y + p[9]*y*y*y;
        } else if (dim() == 3) {
          const T &x = it[0];
          const T &y = it[1];
          const T &z = it[2];
          if (deg == 2)
            return p[0] + p[1]*x + p[2]*y + p[3]*z + p[4]*x*x + p[5]*x*y + p[6]*x*z +
              p[7]*y*y + p[8]*y*z + p[9]*z*z;
          else if (deg == 3)
            return p[0] + p[1]*x + p[2]*y + p[3]*z + p[4]*x*x + p[5]*x*y + p[6]*x*z +
              p[7]*y*y + p[8]*y*z + p[9]*z*z +
              p[10]*x*x*x + p[11]*x*x*y + p[12]*x*x*z + p[13]*x*y*y + p[14]*x*y*z + p[15]*x*z*z +
              p[16]*y*y*y + p[17]*y*y*z + p[18]*y*z*z +
              p[19]*z*z*z;
        }
      }
      }*/
    /* for other polynomials, Horner evaluation (quite slow..) */
    power_index mi(dim());
    return horner(mi, dim(), 0, it);
  }

  template<typename ITER>
    typename std::iterator_traits<ITER>::value_type
        eval_monomial(const power_index &mi, ITER it) {
    typename std::iterator_traits<ITER>::value_type res
      = typename std::iterator_traits<ITER>::value_type(1);
    power_index::const_iterator mit = mi.begin(), mite = mi.end();
    for ( ; mit != mite; ++mit, ++it)
      for (short_type l = 0; l < *mit; ++l)
        res *= *it;
    return res;
  }


  /// Print P to the output stream o. for instance cout << P;
  template<typename T>  std::ostream &operator <<(std::ostream &o,
                                                  const polynomial<T>& P) {
    bool first = true; size_type n = 0;
    typename polynomial<T>::const_iterator it = P.begin(), ite = P.end();
    power_index mi(P.dim());
    if (it != ite && *it != T(0))
      { o << *it; first = false; ++it; ++n; ++mi; }
    for ( ; it != ite ; ++it, ++mi ) {
      if (*it != T(0)) {
        bool first_var = true;
        if (!first) { if (*it < T(0)) o << " - "; else o << " + "; }
        else if (*it < T(0)) o << "-";
        if (gmm::abs(*it)!=T(1)) { o << gmm::abs(*it); first_var = false; }
        for (short_type j = 0; j < P.dim(); ++j)
          if (mi[j] != 0) {
            if (!first_var) o << "*";
            first_var = false;
            if (P.dim() <= 7) o << "xyzwvut"[j];
            else o << "x_" << j;
            if (mi[j] > 1) o << "^" << mi[j];
          }
        first = false; ++n;
      }
    }
    if (n == 0) o << "0";
    return o;
  }

  /**
     polynomial variable substitution
     @param P the original polynomial
     @param S the substitution poly (not a multivariate one)
     @param subs_dim : which variable is substituted
     example: poly_subs(x+y*x^2, x+1, 0) = x+1 + y*(x+1)^2
  */
  template<typename T>
  polynomial<T> poly_substitute_var(const polynomial<T>& P,
                                    const polynomial<T>& S,
                                    size_type subs_dim) {
    GMM_ASSERT2(S.dim() == 1 && subs_dim < P.dim(),
                "wrong arguments for polynomial substitution");
    polynomial<T> res(P.dim(),0);
    bgeot::power_index pi(P.dim());
    std::vector< polynomial<T> > Spow(1);
    Spow[0] = polynomial<T>(1, 0); Spow[0].one(); // Spow stores powers of S
    for (size_type k=0; k < P.size(); ++k, ++pi) {
      if (P[k] == T(0)) continue;
      while (pi[subs_dim] >= Spow.size())
        Spow.push_back(S*Spow.back());
      const polynomial<T>& p = Spow[pi[subs_dim]];
      bgeot::power_index pi2(pi);
      for (short_type i=0; i < p.size(); ++i) {
        pi2[subs_dim] = i;
        res.add_monomial(p[i]*P[k],pi2);
      }
    }
    return res;
  }

  template<typename U, typename T>
  polynomial<T> operator *(T c, const polynomial<T> &p)
  { polynomial<T> q = p; q *= c; return q; }

  typedef polynomial<opt_long_scalar_type> base_poly;

  /* usual constant polynomials  */

  inline base_poly null_poly(short_type n) { return base_poly(n, 0); }
  inline base_poly one_poly(short_type n)
  { base_poly res=base_poly(n, 0); res.one(); return res;  }
  inline base_poly one_var_poly(short_type n, short_type k)
  { return base_poly(n, 1, k); }

  /** read a base_poly on the stream ist. */
  base_poly read_base_poly(short_type n, std::istream &f);

  /** read a base_poly on the string s. */
  base_poly read_base_poly(short_type n, const std::string &s);


  /**********************************************************************/
  /* A class for rational fractions                                     */
  /**********************************************************************/

  template<typename T> class rational_fraction : public std::vector<T> {
  protected :

    polynomial<T> numerator_, denominator_;

  public :

    const polynomial<T> &numerator() const { return numerator_; }
    const polynomial<T> &denominator() const { return denominator_; }

    short_type dim() const { return numerator_.dim(); }

    /// Add Q to P. P contains the result.
    rational_fraction &operator +=(const rational_fraction &Q) {
      numerator_ = numerator_*Q.denominator() + Q.numerator()*denominator_;
      denominator_ *= Q.denominator();
      return *this;
    }
    /// Subtract Q from P. P contains the result.
    rational_fraction &operator -=(const rational_fraction &Q) {
      numerator_ = numerator_*Q.denominator() - Q.numerator()*denominator_;
      denominator_ *= Q.denominator();
      return *this;
    }
    /// Add Q to P.
    rational_fraction operator +(const rational_fraction &Q) const
    { rational_fraction R = *this; R += Q; return R; }
    /// Subtract Q from P.
    rational_fraction operator -(const rational_fraction &Q) const
    { rational_fraction R = *this; R -= Q; return R; }
    /// Add Q to P.
    rational_fraction operator +(const polynomial<T> &Q) const
    { rational_fraction R(numerator_+Q*denominator_, denominator_); return R; }
    /// Subtract Q from P.
    rational_fraction operator -(const polynomial<T> &Q) const
    { rational_fraction R(numerator_-Q*denominator_, denominator_); return R; }
    rational_fraction operator -() const
    { rational_fraction R(-numerator_, denominator_); return R; }
    /// Multiply P with Q. P contains the result.
    rational_fraction &operator *=(const rational_fraction &Q)
    { numerator_*=Q.numerator(); denominator_*=Q.denominator(); return *this; }
    /// Divide P by Q. P contains the result.
    rational_fraction &operator /=(const rational_fraction &Q)
    { numerator_*=Q.denominator(); denominator_*=Q.numerator(); return *this; }
    /// Multiply P with Q.
    rational_fraction operator *(const rational_fraction &Q) const
    { rational_fraction R = *this; R *= Q; return R; }
    /// Divide P by Q.
    rational_fraction operator /(const rational_fraction &Q) const
    { rational_fraction R = *this; R /= Q; return R; }
    /// Multiply P with the scalar a. P contains the result.
    rational_fraction &operator *=(const T &e)
    { numerator_ *= e; return *this; }
    /// Multiply P with the scalar a.
    rational_fraction operator *(const T &e) const
    { rational_fraction R = *this; R *= e; return R; }
    /// Divide P with the scalar a. P contains the result.
    rational_fraction &operator /=(const T &e)
    { denominator_ *= e; return *this; }
    /// Divide P with the scalar a.
    rational_fraction operator /(const T &e) const
    { rational_fraction res = *this; res /= e; return res; }
    /// operator ==.
    bool operator ==(const rational_fraction &Q) const
    { return  (numerator_==Q.numerator() && denominator_==Q.denominator()); }
    /// operator !=.
    bool operator !=(const rational_fraction  &Q) const
    { return !(operator ==(*this,Q)); }
    /// Derivative of P with respect to the variable k. P contains the result.
    void derivative(short_type k) {
      polynomial<T> der_num = numerator_;   der_num.derivative(k);
      polynomial<T> der_den = denominator_; der_den.derivative(k);
      if (der_den.is_zero()) {
        if (der_num.is_zero()) this->clear();
        else numerator_ = der_num;
      } else {
        numerator_ = der_num * denominator_ - der_den * numerator_;
        denominator_ =  denominator_ * denominator_;
      }
    }
    /// Makes P = 1.
    void one() { numerator_.one(); denominator_.one(); }
    void clear() { numerator_.clear(); denominator_.one(); }
    template <typename ITER> T eval(const ITER &it) const {
      typedef typename gmm::number_traits<T>::magnitude_type R;
      T a = numerator_.eval(it), b = denominator_.eval(it);
      if (b == T(0)) { // The better should be to evaluate the derivatives ...
        std::vector<T> p(it, it+dim());
	R no = gmm::vect_norm2(p);
	if (no == R(0)) { gmm::fill_random(p); gmm::scale(p, R(1E-35)); }
	else gmm::scale(p, R(0.9999999));
	a = numerator_.eval(p.begin());
        b = denominator_.eval(p.begin());
      }
      if (a != T(0)) a /= b;
      return a;
    }
    /// Constructor.
    rational_fraction()
      : numerator_(1,0), denominator_(1,0) { clear(); }
    /// Constructor.
    rational_fraction(short_type dim_)
      : numerator_(dim_,0), denominator_(dim_,0)  { clear(); }
    /// Constructor
    rational_fraction(const polynomial<T> &numer)
      : numerator_(numer), denominator_(numer.dim(),0) { denominator_.one(); }
    /// Constructor
    rational_fraction(const polynomial<T> &numer, const polynomial<T> &denom)
      : numerator_(numer), denominator_(denom)
    { GMM_ASSERT1(numer.dim() == denom.dim(), "Dimensions mismatch"); }
  };

  /// Add Q to P.
  template<typename T>
  rational_fraction<T> operator +(const polynomial<T> &P,
                                  const rational_fraction<T> &Q) {
    rational_fraction<T> R(P*Q.denominator()+Q.numerator(), Q.denominator());
    return R;
  }
  /// Subtract Q from P.
  template<typename T>
  rational_fraction<T> operator -(const polynomial<T> &P,
                                  const rational_fraction<T> &Q) {
    rational_fraction<T> R(P*Q.denominator()-Q.numerator(), Q.denominator());
    return R;
  }

  template<typename T>  std::ostream &operator <<
  (std::ostream &o, const rational_fraction<T>& P) {
    o << "[" << P.numerator() << "]/[" << P.denominator() << "]";
    return o;
  }

  typedef rational_fraction<opt_long_scalar_type> base_rational_fraction;

}  /* end of namespace bgeot.                                           */


#endif  /* BGEOT_POLY_H__ */
