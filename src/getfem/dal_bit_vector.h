/* -*- c++ -*- (enables emacs c++ mode) */
/*===========================================================================

 Copyright (C) 1995-2026 Yves Renard

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


#ifndef DAL_BIT_VECTOR_H__
#define DAL_BIT_VECTOR_H__


/** @file dal_bit_vector.h
   @author  Yves Renard <Yves.Renard@insa-lyon.fr>
   @date June 01, 1995.
    @brief Provide a dynamic bit container.
    
    Provide a dynamic bit container, which can also be considered as a
    set of integers.
    
    As a convention, the default value of a bit is false.  The main
    member functions are dal::bit_vector::is_in,
    dal::bit_vector::add, dal::bit_vector::sup. Iterate over the
    bit_vector with dal::bv_visitor
*/

#include "dal_basic.h"
#include <limits.h>
#include <atomic>
#include <bitset>
#include <cstdint>
#include <iterator>
#include <algorithm>
#include <vector>

namespace dal {

  typedef std::uint64_t bit_support;
  static const bit_support WD_BIT = bit_support(CHAR_BIT*sizeof(bit_support));
  static const bit_support WD_MASK = WD_BIT - 1;

  /* Word level operations (std::popcount and friends need c++20). */
#if defined(__GNUC__) || defined(__clang__)
  /// number of non-zero bits of x
  inline unsigned bit_count(bit_support x)
  { return unsigned(__builtin_popcountll(x)); }
  /// index of the lowest non-zero bit of x (x should not be zero)
  inline unsigned bit_lowest(bit_support x)
  { return unsigned(__builtin_ctzll(x)); }
  /// index of the highest non-zero bit of x (x should not be zero)
  inline unsigned bit_highest(bit_support x)
  { return unsigned(WD_MASK) - unsigned(__builtin_clzll(x)); }
#else
  inline unsigned bit_count(bit_support x) {
    x -= (x >> 1) & 0x5555555555555555ULL;
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return unsigned((x * 0x0101010101010101ULL) >> 56);
  }
  inline unsigned bit_lowest(bit_support x)
  { return bit_count((x & (~x + 1)) - 1); }
  inline unsigned bit_highest(bit_support x) {
    x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16; x |= x >> 32;
    return bit_count(x) - 1;
  }
#endif

  class bit_vector;
  class bv_visitor;

  /* Bound on the true or false bits, tightened by the const member functions
     of bit_vector. Relaxed atomic operations are sufficient: any value left
     by concurrent readers is a valid bound. */
  class cached_index {
    std::atomic<size_t> v;
  public :
    cached_index(size_t i = 0) : v(i) {}
    cached_index(const cached_index &o) : v(size_t(o)) {}
    cached_index &operator =(const cached_index &o)
    { return *this = size_t(o); }
    cached_index &operator =(size_t i)
    { v.store(i, std::memory_order_relaxed); return *this; }
    operator size_t(void) const { return v.load(std::memory_order_relaxed); }
  };

  struct APIDECL bit_reference {
    typedef size_t  size_type;

    bit_vector* bv;
    size_type ind;

    bit_reference(bit_vector* b, size_type i) : bv(b), ind(i) {}
    bit_reference(void) {}
    inline operator bool(void) const;
    inline bit_reference& operator = (bool x);
    bit_reference& operator=(const bit_reference& x)
    { return *this = bool(x); }
    bool operator==(const bit_reference& x) const
    { return bool(*this) == bool(x); }
    bool operator<(const bit_reference& x) const
      { return bool(*this) < bool(x); }
    bool operator>(const bit_reference& x) const
      { return bool(*this) > bool(x); }
    bool operator>=(const bit_reference& x) const
      { return bool(*this) >= bool(x); }
    void flip(void) { if (bool(*this)) *this = false; else *this = true; }
  };

  struct APIDECL bit_iterator {
    typedef bool             value_type;
    typedef bit_reference    reference;
    typedef bit_reference*   pointer;
    typedef size_t           size_type;
    typedef ptrdiff_t        difference_type;
    typedef std::random_access_iterator_tag iterator_category;

    size_type ind;
    bit_vector* bv;

    bit_iterator(void) {}
    bit_iterator(bit_vector &b, size_type i) : ind(i), bv(&b) {}
    reference operator*() const { return reference(bv, ind); }
    bit_iterator& operator++() { ++ind; return *this; }
    bit_iterator operator++(int)
    { bit_iterator tmp=*this; ++ind; return tmp; }
    bit_iterator& operator--() { --ind; return *this; }
    bit_iterator operator--(int)
    { bit_iterator tmp = *this; --ind; return tmp; }
    bit_iterator& operator+=(difference_type i) { ind += i; return *this; }
    bit_iterator& operator-=(difference_type i) { ind -= i; return *this; }
    bit_iterator operator+(difference_type i) const
    { bit_iterator tmp = *this; return tmp += i; }
    bit_iterator operator-(difference_type i) const
    { bit_iterator tmp = *this; return tmp -= i; }
    difference_type operator-(bit_iterator x) const { return ind - x.ind; }
    reference operator[](difference_type i) { return *(*this + i); }
    size_type index(void) const { return ind; }
    bool operator==(const bit_iterator& x) const { return ind == x.ind; }
    bool operator!=(const bit_iterator& x) const { return ind != x.ind; }
    bool operator<(bit_iterator x) const { return ind < x.ind; }
    bool operator<=(bit_iterator x) const { return ind <= x.ind; }
    bool operator>(bit_iterator x) const { return ind > x.ind; }
    bool operator>=(bit_iterator x) const { return ind >= x.ind; }
  };

  struct APIDECL bit_const_iterator {
    typedef bool             value_type;
    typedef bool             reference;
    typedef const bool*      pointer;
    typedef size_t           size_type;
    typedef ptrdiff_t        difference_type;
    typedef std::random_access_iterator_tag iterator_category;

    size_type ind;
    const bit_vector* bv;

    bit_const_iterator() {}
    bit_const_iterator(const bit_vector &b, size_type i) : ind(i), bv(&b) {}
    bit_const_iterator(const bit_iterator& x) : ind(x.ind), bv(x.bv) {}
    inline reference operator*() const;
    bit_const_iterator& operator++() { ++ind;  return *this; }
    bit_const_iterator operator++(int)
    { bit_const_iterator tmp = *this; ++ind; return tmp; }
    bit_const_iterator& operator--() { --ind; return *this; }
    bit_const_iterator operator--(int)
    { bit_const_iterator tmp = *this; --ind; return tmp; }
    bit_const_iterator& operator+=(difference_type i)
    { ind += i; return *this; }
    bit_const_iterator& operator-=(difference_type i)
    { ind -= i; return *this; }
    bit_const_iterator operator+(difference_type i) const
    { bit_const_iterator tmp = *this; return tmp += i; }
    bit_const_iterator operator-(difference_type i) const
    { bit_const_iterator tmp = *this; return tmp -= i; }
    difference_type operator-(bit_const_iterator x) const { return ind-x.ind; }
    reference operator[](difference_type i) { return *(*this + i); }
    size_type index(void) const { return ind; }
    bool operator==(const bit_const_iterator& x) const { return ind == x.ind; }
    bool operator!=(const bit_const_iterator& x) const { return ind != x.ind; }
    bool operator<(bit_const_iterator x) const { return ind < x.ind; }
    bool operator<=(bit_const_iterator x) const { return ind <= x.ind; }
    bool operator>(bit_const_iterator x) const { return ind > x.ind; }
    bool operator>=(bit_const_iterator x) const { return ind >= x.ind; }
  };

  /** Dynamic bit container. Concurrent calls of const member functions are
      safe, provided that no thread modifies the bit_vector. */
  class APIDECL bit_vector {
  public :
    
    typedef bool         value_type;
    typedef size_t       size_type;
    typedef ptrdiff_t    difference_type;
    typedef bool         const_reference;
    typedef const bool*  const_pointer;
    typedef bit_reference reference;
    typedef bit_reference*   pointer;
    typedef bit_iterator iterator;
    typedef bit_const_iterator const_iterator;
    
  protected :
    
    friend class bv_visitor;
    friend struct bit_reference;
    friend struct bit_const_iterator;

    /* Never empty. All the bits of index >= size() are false. */
    std::vector<bit_support> words;
    mutable cached_index ifirst_true, ilast_true;
    mutable cached_index ifirst_false, ilast_false;
    size_type icard;
    
    void fill_false(size_type i1, size_type i2);
    /* index of the first bit >= i (resp. of the last bit <= i) such that
       bit ^ flip is true, size_type(-1) if there is no such bit. */
    size_type next_bit(size_type i, bit_support flip) const;
    size_type prev_bit(size_type i, bit_support flip) const;
    void update_card(void);
 
  public : 
    
    void change_for_true(size_type i) {
      ifirst_true = std::min(size_type(ifirst_true), i);
      ilast_true = std::max(size_type(ilast_true), i);
      ++icard;
    }
    void change_for_false(size_type i) {
      ifirst_false = std::min(size_type(ifirst_false), i);
      ilast_false = std::max(size_type(ilast_false), i);
      --icard;
    }
    
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    size_type size(void) const
    { return std::max(size_type(ilast_true), size_type(ilast_false))+1;}
    
    iterator begin(void) { return iterator(*this, 0); }
    const_iterator begin(void) const { return const_iterator(*this, 0); }
    iterator end(void) { return iterator(*this, size()); }
    const_iterator end(void) const { return const_iterator(*this, size()); }

    reverse_iterator rbegin(void) { return reverse_iterator(end()); }
    const_reverse_iterator rbegin(void) const
    { return const_reverse_iterator(end()); }
    reverse_iterator rend(void) { return reverse_iterator(begin()); }
    const_reverse_iterator rend(void) const
    { return const_reverse_iterator(begin()); }
    
    size_type capacity(void) const { return words.capacity() * WD_BIT; }
    size_type max_size(void) const { return (size_type(-1)); }
    /// true if no index is in the set
    bool empty(void) const { return icard == 0; }
    reference front(void) { return *begin(); }
    const_reference front(void) const { return *begin(); }
    reference back(void) { return *(end() - 1); }
    const_reference back(void) const { return *(end() - 1); }
    
    
    const_reference operator [](size_type ii) const { return is_in(ii); }
    reference operator [](size_type ii)
    { if (ii >= size()) fill_false(size(),ii); return reference(this, ii); }
    
    void swap(bit_vector &da);
    
    void clear(void) {
      icard = 0;
      ifirst_false = 0; ilast_false = 0; ifirst_true = 0; ilast_true = 0;
      words.assign(1, bit_support(0));
    }
    void swap(size_type i1, size_type i2) {
      if (i1 != i2) {
	reference r1 = (*this)[i1], r2 = (*this)[i2];
	bool tmp = r1; r1 = r2; r2 = tmp;
      }
    }
    size_type memsize(void) const
    { return sizeof(bit_vector) + words.capacity() * sizeof(bit_support); }
    size_type card(void) const { return icard; }
    /// index of first non-zero entry (size_type(-1) for an empty bit_vector)
    size_type first_true(void) const;
    /// index of first zero entry (size_type(0) for an empty bit_vector)
    size_type first_false(void) const;
    /// index of last non-zero entry (size_type(-1) for an empty bit_vector)
    size_type last_true(void) const;
    /// index of last zero entry (size_type(0) for an empty bit_vector)
    size_type last_false(void) const;
    /// remove all elements found in bv
    bit_vector &setminus(const bit_vector &bv);
    bit_vector &operator |=(const bit_vector &bv);
    bit_vector &operator &=(const bit_vector &bv);
    
    bit_vector operator |(const bit_vector &bv) const
    { bit_vector r(*this); r |= bv; return r; }
    bit_vector operator &(const bit_vector &bv) const
    { bit_vector r(*this); r &= bv; return r; }
    bool operator ==(const bit_vector &bv) const;
    bool operator !=(const bit_vector &bv) const
    { return !((*this) == bv); }
    
    bit_vector(void) : words(1, bit_support(0)), icard(0) {}
    bit_vector(const bit_vector &) = default;
    bit_vector &operator =(const bit_vector &) = default;
    /* The moved-from object has to remain a valid (empty) bit_vector. */
    bit_vector(bit_vector &&bv) : bit_vector() { swap(bv); }
    bit_vector &operator =(bit_vector &&bv)
    { if (this != &bv) { clear(); swap(bv); } return *this; }
    template <size_t N> bit_vector(const std::bitset<N> &bs) : bit_vector() {
      for (size_type i=0; i < bs.size(); ++i) { if (bs[i]) add(i); }
    }
    
    /// merges the integer values of the supplied container into the bit_vector
    template <typename ICONT> dal::bit_vector& merge_from(const ICONT& c) {
      for (typename ICONT::const_iterator it = c.begin(); it != c.end(); ++it)
	add(*it);
      return *this;
    }
    /** merges the integer values of the supplied iterator range into
     * the bit_vector */
    template <typename IT> dal::bit_vector& merge_from(IT b, IT e) {
      while (b != e) { add(*b++); }
      return *this;
    }
    /** return true if the supplied bit_vector is a subset of the current
     * bit_vector */
    bool contains(const dal::bit_vector &other) const;
    
  public : 
    /// return true if (*this)[i] == true
    bool is_in(size_type i) const {
      size_type iw = i / WD_BIT;
      return iw < words.size() && ((words[iw] >> (i & WD_MASK)) & 1);
    }
    void add(size_type i) { (*this)[i] = true; }
    /** set the interval [i .. i+nb-1] to true */
    void add(size_type i, size_type nb);
    void sup(size_type i) { (*this)[i] = false; } /* deprecated ...*/
    void del(size_type i) { (*this)[i] = false; }
    /** set the interval [i .. i+nb-1] to false */
    void sup(size_type i, size_type nb); /* deprecated ...*/
    void del(size_type i, size_type nb);
    int first(void) const { return (card() == 0) ? -1 : int(first_true()); }
    int last(void) const { return (card() == 0) ? -1 : int(last_true()); }
    inline int take_first(void)
    { int res = first(); if (res >= 0) sup(res); return res; }
    inline int take_last(void)
    { int res = last(); if (res >= 0) sup(res); return res; }
  };

  inline bit_reference::operator bool(void) const { return bv->is_in(ind); }

  inline bit_reference& bit_reference::operator = (bool x) {
    bit_support &w = bv->words[ind / WD_BIT];
    bit_support mask = bit_support(1) << (ind & WD_MASK);
    if (x) { if (!(w & mask)) { w |= mask; bv->change_for_true(ind); } }
    else   { if (w & mask) { w &= ~mask; bv->change_for_false(ind); } }
    return *this;
  }

  inline bit_const_iterator::reference
  bit_const_iterator::operator*() const { return bv->is_in(ind); }

  /**
     if you are only interested in indexes of true values of a bit_vector
     (i.e. if you use it as an int set), use bv_visitor instead of
     bit_vector::const_iterator (much faster)

     example:
     @code
     for (bv_visitor i(v); !i.finished(); ++i) {
       .... (use i as an unsigned int)
     }
     @endcode
     CAUTION: use bv_visitor_c instead of bv_visitor if the class bv_visitor
     need to store a copy of the bit_vector 
     (if the original is destroyed just after the creation...)
  */
  class APIDECL bv_visitor {
    typedef dal::bit_vector::size_type size_type;
    const bit_vector *bv;
    size_type iw, ilast, ind;
    bit_support v; // bits of the word iw which remain to be visited
  public:
    bv_visitor(const dal::bit_vector& b)
      : bv(&b), iw(0), ilast(0), ind(0), v(0) {
      if (!b.empty()) {
        ind = b.first_true(); ilast = b.last_true() + 1; iw = ind / WD_BIT;
        v = b.words[iw] & (~bit_support(0) << (ind & WD_MASK));
        v &= v - 1;
      }
    }
    bool finished() const { return ind >= ilast; }
    bool operator++() {
      while (!v) {
        if (++iw >= bv->words.size() || iw * WD_BIT >= ilast)
          { ind = ilast; return false; }
        v = bv->words[iw];
      }
      ind = iw * WD_BIT + bit_lowest(v);
      v &= v - 1;
      return ind < ilast;
    }
    operator size_type() const { return ind; }
  };

  /**
    bv_visitor with local copy of the bit_vector
  */
  class APIDECL bv_visitor_c {
    bit_vector bv;
    bv_visitor v; // no inheritance since v must be init after bv
  public:
    bv_visitor_c(const dal::bit_vector& b) : bv(b), v(bv) {}
    bool finished() const { return v.finished(); }
    bool operator++() { return ++v; }
    operator dal::bit_vector::size_type() const
    { return dal::bit_vector::size_type(v); }
  };

  /// extract index of first entry in the bit_vector
  inline int APIDECL &operator << (int &i, bit_vector &s)
  { i = s.take_first(); return i; }
  inline const int APIDECL &operator >> (const int &i, bit_vector &s)
  { s.add(i); return i; }

  inline size_t APIDECL &operator << (size_t &i, bit_vector &s)
  { i = s.take_first(); return i; }
  inline const size_t &operator >> (const size_t &i, bit_vector &s)
  { s.add(i); return i; }

  std::ostream APIDECL &operator <<(std::ostream &o, const bit_vector &s);
  
  /**Iterator class for bv_iterable and bv_iterable_c.*/
  template<typename ITERABLE_BV>
  class const_bv_iterator
  {
    typedef dal::bit_vector::size_type size_type;

  public:

    typedef std::forward_iterator_tag iterator_category;
    typedef size_type value_type;
    typedef size_type difference_type;
    typedef size_type* pointer;
    typedef size_type& reference;

    const_bv_iterator(const ITERABLE_BV* p_iterable, size_type pos)
      : p_iterable_(const_cast<ITERABLE_BV*>(p_iterable)), pos_(pos)
    {}

    bool operator!= (const const_bv_iterator &other) const{
      return pos_ < other.pos_;
    }

    size_type operator *() const{
      return size_type(*p_iterable_);
    }

    //difference of iterators
    size_type operator-(const const_bv_iterator &other) const{
      if (pos_ == other.pos_) return 0;

      auto &vector_this  = p_iterable_->v_;
      auto &vector_other = other.p_iterable_->v_;
      bv_visitor v_this(vector_this), v_other(vector_other);
      while (v_this < pos_) ++v_this;
      while (v_other < other.pos_) ++v_other;
      auto &v     = (pos_ < other.pos_)  ? v_this : v_other;
      auto &v_end = (pos_ >= other.pos_) ? v_this : v_other;

      size_type count = 0;
      while(v < v_end) { ++v; ++count;}
      return count;
    }

    const const_bv_iterator &operator++(){
      ++*p_iterable_;
      pos_ = *p_iterable_;
      return *this;
    }

  private:    
    ITERABLE_BV *p_iterable_;
    size_type pos_;
  };

    
  /**
  Wrapper class to make bit_vector iterable on true values.
  It enables the use of c++11 range-based loop feature.
  example:
  @code
  for (auto i : bv_iterable(bit_vector)) {
  .... //(use i as an unsigned int)
  }
  */
  class bv_iterable
  {
  public:
    bv_iterable(const bit_vector &v) : v_(v), visitor_(v){}

    const_bv_iterator<bv_iterable> begin() const;
    const_bv_iterator<bv_iterable> end() const;
    inline bool operator++(){return ++visitor_;};
    inline operator dal::bit_vector::size_type() const{return visitor_;};

  private:
    const bit_vector& v_;
    bv_visitor visitor_;
    friend class const_bv_iterator<bv_iterable>;
  };

  /***Same as bv_iterable class except the bit_vector is copied locally.*/
  class bv_iterable_c
  {
  public:
    bv_iterable_c(const bit_vector &v) : v_(v), visitor_(v_){}
    const_bv_iterator<bv_iterable_c> begin() const;
    const_bv_iterator<bv_iterable_c> end() const;
    inline bool operator++(){return ++visitor_;};
    inline operator dal::bit_vector::size_type() const{return visitor_;};

  private:
    bit_vector v_;
    bv_visitor visitor_;
    friend class const_bv_iterator<bv_iterable_c>;
  };

}

#endif /* DAL_BIT_VECTOR_H__ */
