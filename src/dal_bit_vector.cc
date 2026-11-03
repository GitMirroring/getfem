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

===========================================================================*/


#include "getfem/dal_bit_vector.h"

namespace dal {

  void bit_vector::fill_false(size_type, size_type i2) {
    size_type nw = i2 / WD_BIT + 1;
    if (nw > words.size()) words.resize(nw, bit_support(0));
    ilast_false = i2;
  }

  void bit_vector::update_card(void) {
    icard = 0;
    for (bit_support w : words) icard += bit_count(w);
  }

  void bit_vector::swap(bit_vector &da) {
    words.swap(da.words);
    std::swap(ifirst_true, da.ifirst_true);
    std::swap(ifirst_false, da.ifirst_false);
    std::swap(ilast_true, da.ilast_true);
    std::swap(ilast_false, da.ilast_false);
    std::swap(icard, da.icard);
  }

  bit_vector::size_type
  bit_vector::next_bit(size_type i, bit_support flip) const {
    size_type iw = i / WD_BIT;
    if (iw >= words.size()) return size_type(-1);
    bit_support x = (words[iw] ^ flip) & (~bit_support(0) << (i & WD_MASK));
    while (!x) {
      if (++iw == words.size()) return size_type(-1);
      x = words[iw] ^ flip;
    }
    return iw * WD_BIT + bit_lowest(x);
  }

  bit_vector::size_type
  bit_vector::prev_bit(size_type i, bit_support flip) const {
    size_type iw = i / WD_BIT;
    bit_support x = (words[iw] ^ flip)
      & (~bit_support(0) >> (WD_MASK - (i & WD_MASK)));
    while (!x) {
      if (iw == 0) return size_type(-1);
      x = words[--iw] ^ flip;
    }
    return iw * WD_BIT + bit_highest(x);
  }

  bit_vector::size_type bit_vector::first_true(void) const {
    if (empty()) { ifirst_true = 0; ilast_true = 0; return size_type(-1); }
    size_type i = next_bit(ifirst_true, bit_support(0));
    ifirst_true = i;
    return i;
  }

  bit_vector::size_type bit_vector::first_false(void) const {
    size_type i = next_bit(ifirst_false, ~bit_support(0));
    if (i == size_type(-1)) // all the stored bits are true
      i = std::max(size_type(ifirst_false), words.size() * WD_BIT);
    ifirst_false = i;
    return i;
  }

  bit_vector::size_type bit_vector::last_true(void) const {
    if (empty()) { ilast_true = 0; return size_type(-1); }
    size_type i = prev_bit(ilast_true, bit_support(0));
    ilast_true = i;
    return i;
  }

  bit_vector::size_type bit_vector::last_false(void) const {
    size_type i = prev_bit(ilast_false, ~bit_support(0));
    if (i == size_type(-1)) i = 0;
    ilast_false = i;
    return i;
  }

  bit_vector &bit_vector::setminus(const bit_vector& b) {
    if (b.empty()) return *this;
    size_type last = b.last_true(), first = b.first_true();
    if (last >= size()) fill_false(size(), last);
    size_type nw = std::min(words.size(), b.words.size());
    for (size_type i = 0; i < nw; ++i) words[i] &= ~b.words[i];
    update_card();
    ifirst_false = std::min(size_type(ifirst_false), first);
    ilast_false = std::max(size_type(ilast_false), last);
    return *this;
  }

  bit_vector &bit_vector::operator |=(const bit_vector &b) {
    if (b.empty()) return *this;
    size_type last = b.last_true(), first = b.first_true();
    if (last >= size()) fill_false(size(), last);
    size_type nw = std::min(words.size(), b.words.size());
    for (size_type i = 0; i < nw; ++i) words[i] |= b.words[i];
    update_card();
    ifirst_true = std::min(size_type(ifirst_true), first);
    ilast_true = std::max(size_type(ilast_true), last);
    return *this;
  }

  bit_vector &bit_vector::operator &=(const bit_vector &b) {
    size_type nw = std::min(words.size(), b.words.size());
    for (size_type i = 0; i < nw; ++i) words[i] &= b.words[i];
    std::fill(words.begin() + nw, words.end(), bit_support(0));
    update_card();
    ifirst_true = std::max(size_type(ifirst_true), size_type(b.ifirst_true));
    ilast_true = std::min(size_type(ilast_true), size_type(b.ilast_true));
    if (ifirst_true > ilast_true) { clear(); return *this; }
    ilast_false = std::min(size()-1, std::max(size_type(ilast_false),
                                              size_type(b.ilast_false)));
    ifirst_false = std::min(size_type(ifirst_false),
                            size_type(b.ifirst_false));
    return *this;
  }

  bool bit_vector::operator ==(const bit_vector &b) const {
    size_type nw = std::min(words.size(), b.words.size());
    for (size_type i = 0; i < nw; ++i)
      if (words[i] != b.words[i]) return false;
    for (size_type i = nw; i < words.size(); ++i)
      if (words[i]) return false;
    for (size_type i = nw; i < b.words.size(); ++i)
      if (b.words[i]) return false;
    return true;
  }

  void bit_vector::add(size_type i, size_type nb) {
    if (!nb) return;
    size_type l = i + nb - 1, il = l / WD_BIT, ifw = i / WD_BIT;
    if (l >= size()) fill_false(size(), l);
    for (size_type iw = ifw; iw <= il; ++iw) {
      bit_support m = ~bit_support(0);
      if (iw == ifw) m &= ~bit_support(0) << (i & WD_MASK);
      if (iw == il) m &= ~bit_support(0) >> (WD_MASK - (l & WD_MASK));
      icard += bit_count(m & ~words[iw]);
      words[iw] |= m;
    }
    ifirst_true = std::min(size_type(ifirst_true), i);
    ilast_true = std::max(size_type(ilast_true), l);
  }

  void bit_vector::del(size_type i, size_type nb) {
    if (!nb) return;
    size_type l = i + nb - 1, il = l / WD_BIT, ifw = i / WD_BIT;
    if (l >= size()) fill_false(size(), l);
    for (size_type iw = ifw; iw <= il; ++iw) {
      bit_support m = ~bit_support(0);
      if (iw == ifw) m &= ~bit_support(0) << (i & WD_MASK);
      if (iw == il) m &= ~bit_support(0) >> (WD_MASK - (l & WD_MASK));
      icard -= bit_count(m & words[iw]);
      words[iw] &= ~m;
    }
    ifirst_false = std::min(size_type(ifirst_false), i);
    ilast_false = std::max(size_type(ilast_false), l);
  }

  void bit_vector::sup(size_type i, size_type nb) { del(i, nb); }

  bool bit_vector::contains(const dal::bit_vector& other) const {
    size_type nw = std::min(words.size(), other.words.size());
    for (size_type i = 0; i < nw; ++i)
      if (other.words[i] & ~words[i]) return false;
    for (size_type i = nw; i < other.words.size(); ++i)
      if (other.words[i]) return false;
    return true;
  }

  std::ostream &operator <<(std::ostream &o, const bit_vector &s) {
    bool first = true;
    o << "[";
    for (bv_visitor i(s); !i.finished(); ++i) {
      if (!first) o << " ";
      o << bit_vector::size_type(i);
      first = false;
    }
    o << "]";
    return o;
  }

  const_bv_iterator<bv_iterable> bv_iterable::begin() const{
    return const_bv_iterator<bv_iterable>(this, v_.first_true());
  }
  const_bv_iterator<bv_iterable> bv_iterable::end() const{
    return const_bv_iterator<bv_iterable>(this, v_.last_true() + 1);
  }

  const_bv_iterator<bv_iterable_c> bv_iterable_c::begin() const{
    return const_bv_iterator<bv_iterable_c>(this, v_.first_true());
  }
  const_bv_iterator<bv_iterable_c> bv_iterable_c::end() const{
    return const_bv_iterator<bv_iterable_c>(this, v_.last_true() + 1);
  }
}

