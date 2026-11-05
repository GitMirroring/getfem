/* -*- c++ -*- (enables emacs c++ mode) */
/*===========================================================================

 Copyright (C) 2004-2026 Julien Pommier, Yves Renard

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

/**@file getfem_mesher.h
   @author  Julien Pommier <Julien.Pommier@insa-toulouse.fr>, Yves Renard <Yves.Renard@insa-lyon.fr>
   @date May 1, 2004.
   @brief An experimental mesher.
*/

#ifndef GETFEM_MESHER_H__
#define GETFEM_MESHER_H__


#include "getfem_mesh.h"
#include "gmm/gmm_condition_number.h"
#include "bgeot_comma_init.h"
#include "gmm/gmm_solver_bfgs.h"
#include "getfem_export.h"
#include "bgeot_kdtree.h"
#include <typeinfo>

namespace getfem {

  class mesher_virtual_function : virtual public dal::static_stored_object {
  public:
    virtual scalar_type operator()(const base_node &P) const = 0;
    virtual ~mesher_virtual_function() {}
  };

  class mvf_constant : public mesher_virtual_function {
    scalar_type c;
  public: 
    mvf_constant(scalar_type c_) : c(c_) {}
    scalar_type operator()(const base_node &) const { return c; }
  };

  // Signed distance definition. Should not be a real signed distance but
  // should satisfy that dist(P) is less than the euclidean distance from
  // P to the boundary and more than 1/sqrt(N) time this distance.

#define SEPS 1e-8

  class mesher_signed_distance : public mesher_virtual_function {
  protected:
    mutable size_type id;
  public:
    mesher_signed_distance() : id(size_type(-1)) {}
    virtual ~mesher_signed_distance() {}
    virtual bool bounding_box(base_node &bmin, base_node &bmax) const = 0;
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector &bv) const = 0;
    virtual scalar_type grad(const base_node &P,
			     base_small_vector &G) const = 0;
    virtual void hess(const base_node &P, base_matrix &H) const = 0;
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const=0;
    virtual scalar_type operator()(const base_node &P) const  = 0;
  };

  typedef std::shared_ptr<const mesher_signed_distance> pmesher_signed_distance;

  class mesher_half_space : public mesher_signed_distance {
    base_node x0; base_small_vector n; scalar_type xon;
  public:
    mesher_half_space() = default;
    mesher_half_space(const base_node &x0_, const base_small_vector &n_)
      : x0(x0_), n(n_)
    { n /= gmm::vect_norm2(n); xon = gmm::vect_sp(x0, n); }
    bool bounding_box(base_node &, base_node &) const
    { return false; }
    virtual scalar_type operator()(const base_node &P) const
    { return xon - gmm::vect_sp(P,n); }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector &bv) const {
      scalar_type d = xon - gmm::vect_sp(P,n);
      bv[id] = (gmm::abs(d) < SEPS);
      return d;
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const {
      id = list.size(); list.push_back(this);
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      G = n; G *= scalar_type(-1); 
      return xon - gmm::vect_sp(P,n);
    }
    void hess(const base_node &P, base_matrix &H) const {
      gmm::resize(H, P.size(), P.size()); gmm::clear(H);
    }

  };

  inline pmesher_signed_distance  new_mesher_half_space
  (const base_node &x0, const base_small_vector &n)
  { return std::make_shared<mesher_half_space>(x0, n); }

  class mesher_level_set : public mesher_signed_distance {
    bgeot::base_poly base;
    mutable std::vector<base_poly> gradient;
    mutable std::vector<base_poly> hessian;
    const fem<base_poly> *pf;
    mutable int initialized;
    scalar_type shift_ls;     // for the computation of a gap on a level_set.
  public:
    bool is_initialized(void) const { return initialized; }
    mesher_level_set() : initialized(0) {}
    template <typename VECT>
    mesher_level_set(pfem pf_, const VECT &coeff_,
		     scalar_type shift_ls_ = scalar_type(0)) {
      init_base(pf_, coeff_);
      set_shift(shift_ls_);
    }
    void set_shift(scalar_type shift_ls_) {
      shift_ls = shift_ls_; 
      if (shift_ls != scalar_type(0)) {
	base_node P(pf->dim()); base_small_vector G(pf->dim());
	grad(P, G);
	shift_ls *= gmm::vect_norm2(G);
      }
    }
    template <typename VECT> void init_base(pfem pf_, const VECT &coeff_);
    void init_grad(void) const;
    void init_hess(void) const;

    bool bounding_box(base_node &, base_node &) const
    { return false; }
    virtual scalar_type operator()(const base_node &P) const
    {  return bgeot::to_scalar(base.eval(P.begin())) + shift_ls; }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector &bv) const
    { scalar_type d = (*this)(P); bv[id] = (gmm::abs(d) < SEPS); return d; }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const {
      id = list.size(); list.push_back(this);
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const;
    void hess(const base_node &P, base_matrix &H) const;
  };

  template <typename VECT> 
  void mesher_level_set::init_base(pfem pf_, const VECT &coeff_) {
    std::vector<scalar_type> coeff(coeff_.begin(), coeff_.end());
    GMM_ASSERT1(gmm::vect_norm2(coeff) != 0, "level is zero!");
    pf = dynamic_cast<const fem<base_poly>*> (pf_.get());
    GMM_ASSERT1(pf, "A polynomial fem is required for level set (got "
		<< typeid(pf_).name() << ")");
    base = base_poly(pf->base()[0].dim(), pf->base()[0].degree());
    for (unsigned i=0; i < pf->nb_base(0); ++i) {
      base += pf->base()[i] * coeff[i];
    }
    initialized = 0; 
  }

  template <typename VECT> 
  inline pmesher_signed_distance new_mesher_level_set
  (pfem pf, const VECT &coeff, scalar_type shift_ls = scalar_type(0))
  { return std::make_shared<mesher_level_set>(pf, coeff, shift_ls); }


  class mesher_ball : public mesher_signed_distance {
    base_node x0; scalar_type R;
  public:
    mesher_ball(base_node x0_, scalar_type R_) : x0(x0_), R(R_) {}
    bool bounding_box(base_node &bmin, base_node &bmax) const { 
      bmin = bmax = x0; 
      for (size_type i=0; i < x0.size(); ++i) { bmin[i] -= R; bmax[i] += R; }
      return true;
    }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector &bv) const {
      scalar_type d = gmm::vect_dist2(P,x0)-R;
      bv[id] = (gmm::abs(d) < SEPS);
      return d;
    }
    virtual scalar_type operator()(const base_node &P) const
    { return gmm::vect_dist2(P,x0)-R; }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const {
      id = list.size(); list.push_back(this);
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      G = P; G -= x0;
      scalar_type e= gmm::vect_norm2(G), d = e - R;
      while (e == scalar_type(0))
	{ gmm::fill_random(G); e = gmm::vect_norm2(G); }
      G /= e;
      return d;
    }
    void hess(const base_node &, base_matrix &) const {
      GMM_ASSERT1(false, "Sorry, to be done");
    }
  };

  inline pmesher_signed_distance new_mesher_ball(base_node x0, scalar_type R)
  { return std::make_shared<mesher_ball>(x0, R); }

  class mesher_rectangle : public mesher_signed_distance {
    // ajouter une rotation rigide et translation ..
    base_node rmin, rmax;
    std::vector<mesher_half_space> hfs;
  public:
    mesher_rectangle(base_node rmin_, base_node rmax_)
      : rmin(rmin_), rmax(rmax_), hfs(rmin.size()*2) {
      base_node n(rmin_.size());
      for (unsigned k = 0; k < rmin.size(); ++k) {
        n[k] = 1.0;
        hfs[k*2] = mesher_half_space(rmin, n);
        n[k] = -1.0;
        hfs[k*2+1] = mesher_half_space(rmax, n);
        n[k] = 0.0;
      }
    }
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      bmin = rmin; bmax = rmax;
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const {
      size_type N = rmin.size();
      scalar_type d = rmin[0] - P[0];
      for (size_type i=0; i < N; ++i) {
	d = std::max(d, rmin[i] - P[i]);
	d = std::max(d, P[i] - rmax[i]);
      }
      return d;
    }

    virtual scalar_type operator()(const base_node &P, dal::bit_vector &bv)
      const {
      scalar_type d = this->operator()(P);
      if (gmm::abs(d) < SEPS)
	for (int k = 0; k < 2*rmin.size(); ++k) hfs[k](P, bv);
      return d;
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      unsigned i = 0; scalar_type di = hfs[i](P);
      for (int k = 1; k < 2*rmin.size(); ++k) {
	scalar_type dk = hfs[k](P);
	if (dk > di) { i = k; di = dk; }
      }
      return hfs[i].grad(P, G);
    }
    void hess(const base_node &P, base_matrix &H) const {
      gmm::resize(H, P.size(), P.size()); gmm::clear(H);
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const {
      for (int k = 0; k < 2*rmin.size(); ++k)
	hfs[k].register_constraints(list);
    }
  };

  inline pmesher_signed_distance new_mesher_rectangle(base_node rmin,
						      base_node rmax)
  { return std::make_shared<mesher_rectangle>(rmin, rmax); }

  class mesher_simplex_ref : public mesher_signed_distance {
    // To be added : rigid motion, dilatation ...
    std::vector<mesher_half_space> hfs;
    unsigned N;
    base_node org;
  public:
    mesher_simplex_ref(unsigned N_) : hfs(N_+1), N(N_), org(N_) {
      base_node no(N);
      for (unsigned k = 0; k < N; ++k) {
        no[k] = 1.0;
        hfs[k] = mesher_half_space(org, no);
        no[k] = 0.0;
      }
      std::fill(org.begin(), org.end(), 1.0/N);
      no = -org;
      hfs[N] = mesher_half_space(org, no);
    }
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      bmin.resize(N); bmax.resize(N);
      std::fill(bmin.begin(), bmin.end(), scalar_type(0));
      std::fill(bmax.begin(), bmax.end(), scalar_type(1));
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const {
      scalar_type d = - P[0];
      for (size_type i=1; i < N; ++i) d = std::max(d, - P[i]);
      d = std::max(d, gmm::vect_sp(P - org, org) / gmm::vect_norm2(org));
      return d;
    }

    virtual scalar_type operator()(const base_node &P, dal::bit_vector &bv)
      const {
      scalar_type d = this->operator()(P);
      if (gmm::abs(d) < SEPS) for (unsigned k = 0; k < N+1; ++k) hfs[k](P, bv);
      return d;
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      unsigned i = 0; scalar_type di = hfs[i](P);
      for (unsigned k = 1; k < N+1; ++k) {
	scalar_type dk = hfs[k](P);
	if (dk > di) { i = k; di = dk; }
      }
      return hfs[i].grad(P, G);
    }
    void hess(const base_node &P, base_matrix &H) const {
      gmm::resize(H, P.size(), P.size()); gmm::clear(H);
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const
    { for (unsigned k = 0; k < N+1; ++k) hfs[k].register_constraints(list); }
  };

  inline pmesher_signed_distance new_mesher_simplex_ref(unsigned N)
  { return std::make_shared<mesher_simplex_ref>(N); }


  class mesher_prism_ref : public mesher_signed_distance {
    // To be added : rigid motion, dilatation ...
    std::vector<mesher_half_space> hfs;
    unsigned N;
    base_node org;
  public:
    mesher_prism_ref(unsigned N_) : hfs(N_+2), N(N_) {  
      base_node no(N);
      org = no;
      for (unsigned k = 0; k < N; ++k) {
        no[k] = 1.0;
        hfs[k] = mesher_half_space(org, no);
        no[k] = 0.0;
      }
      no[N-1] = -1.0;
      org[N-1] = 1.0;
      hfs[N] = mesher_half_space(org, no);
      std::fill(org.begin(), org.end(), 1.0/N);
      org[N-1] = 0.0;
      no = -org;
      hfs[N+1] = mesher_half_space(org, no);    
    }
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      bmin.resize(N); bmax.resize(N);
      std::fill(bmin.begin(), bmin.end(), scalar_type(0));
      std::fill(bmax.begin(), bmax.end(), scalar_type(1));
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const {
      scalar_type d = - P[0];
      for (size_type i=1; i < N; ++i) d = std::max(d, - P[i]);
      d = std::max(d, P[N-1] - scalar_type(1));
      d = std::max(d, gmm::vect_sp(P - org, org) / gmm::vect_norm2(org));
      return d;
    }

    virtual scalar_type operator()(const base_node &P, dal::bit_vector &bv)
      const {
      scalar_type d = this->operator()(P);
      if (gmm::abs(d) < SEPS) for (unsigned k = 0; k < N+2; ++k) hfs[k](P, bv);
      return d;
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      unsigned i = 0; scalar_type di = hfs[i](P);
      for (unsigned k = 1; k < N+2; ++k) {
	scalar_type dk = hfs[k](P);
	if (dk > di) { i = k; di = dk; }
      }
      return hfs[i].grad(P, G);
    }
    void hess(const base_node &P, base_matrix &H) const {
      gmm::resize(H, P.size(), P.size()); gmm::clear(H);
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const
    { for (unsigned k = 0; k < N+2; ++k) hfs[k].register_constraints(list); }
  };

  inline pmesher_signed_distance new_mesher_prism_ref(unsigned N)
  { return std::make_shared<mesher_prism_ref>(N); }



  // Rem : It would be better for the convergence of Newton's methods to
  //       take somthing more smooth than min. for instance tthe square root
  //       of positive parts or negative parts depending on the position of
  //       point (in or out the domain).

  class mesher_union : public mesher_signed_distance {
    std::vector<pmesher_signed_distance> dists;
    mutable std::vector<scalar_type> vd;
    mutable bool isin;
    bool with_min;
  public:
    mesher_union(const std::vector<pmesher_signed_distance>
			&dists_) : dists(dists_) 
    { vd.resize(dists.size()); with_min = true; }

    mesher_union
    (const pmesher_signed_distance &a,
     const pmesher_signed_distance &b,
     const pmesher_signed_distance &c = pmesher_signed_distance(),
     const pmesher_signed_distance &d = pmesher_signed_distance(),
     const pmesher_signed_distance &e = pmesher_signed_distance(),
     const pmesher_signed_distance &f = pmesher_signed_distance(),
     const pmesher_signed_distance &g = pmesher_signed_distance(),
     const pmesher_signed_distance &h = pmesher_signed_distance(),
     const pmesher_signed_distance &i = pmesher_signed_distance(),
     const pmesher_signed_distance &j = pmesher_signed_distance(),
     const pmesher_signed_distance &k = pmesher_signed_distance(),
     const pmesher_signed_distance &l = pmesher_signed_distance(),
     const pmesher_signed_distance &m = pmesher_signed_distance(),
     const pmesher_signed_distance &n = pmesher_signed_distance(),
     const pmesher_signed_distance &o = pmesher_signed_distance(),
     const pmesher_signed_distance &p = pmesher_signed_distance(),
     const pmesher_signed_distance &q = pmesher_signed_distance(),
     const pmesher_signed_distance &r = pmesher_signed_distance(),
     const pmesher_signed_distance &s = pmesher_signed_distance(),
     const pmesher_signed_distance &t = pmesher_signed_distance()) {
      dists.push_back(a); dists.push_back(b);
      with_min = true;
      if (c) dists.push_back(c);
      if (d) dists.push_back(d);
      if (e) dists.push_back(e);
      if (f) dists.push_back(f);
      if (g) dists.push_back(g);
      if (h) dists.push_back(h);
      if (i) dists.push_back(i);
      if (j) dists.push_back(j);
      if (k) dists.push_back(k);
      if (l) dists.push_back(l);
      if (m) dists.push_back(m);
      if (n) dists.push_back(n);
      if (o) dists.push_back(o);
      if (p) dists.push_back(p);
      if (q) dists.push_back(q);
      if (r) dists.push_back(r);
      if (s) dists.push_back(s);
      if (t) dists.push_back(t);
      vd.resize(dists.size());
    }
    
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      base_node bmin2, bmax2;
      bool b = dists[0]->bounding_box(bmin, bmax);
      if (!b) return false;
      for (size_type k = 1; k < dists.size(); ++k) {
	b = dists[k]->bounding_box(bmin2, bmax2);
	if (!b) return false;
	for (unsigned i=0; i < bmin.size(); ++i) { 
	  bmin[i] = std::min(bmin[i],bmin2[i]);
	  bmax[i] = std::max(bmax[i],bmax2[i]);
	}
      }
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const {
      scalar_type d, f(0), g(1);
      if (with_min) {
	d = (*(dists[0]))(P);
	for (size_type k = 1; k < dists.size(); ++k)
	  d = std::min(d, (*(dists[k]))(P));
      }
      else { // essai raté ...
	isin = false;
	for (size_type k = 0; k < dists.size(); ++k) {
	  vd[k] = (*(dists[k]))(P);
	  if (vd[k] <= scalar_type(0)) isin = true;
	  f += gmm::sqr(gmm::neg(vd[k]));
	  g *= gmm::pos(vd[k]);
	}
	d = isin ? -gmm::sqrt(f)
	  : pow(g, scalar_type(1) / scalar_type(dists.size()));
      }
      return d;
    }
    scalar_type operator()(const base_node &P, dal::bit_vector &bv) const {
      if (with_min) {
	scalar_type d = vd[0] = (*(dists[0]))(P);
	bool ok = (d > -SEPS);
	for (size_type k = 1; k < dists.size(); ++k) {
	  vd[k] = (*(dists[k]))(P); if (vd[k] <= -SEPS) ok = false;
	  d = std::min(d,vd[k]);
	}
	for (size_type k = 0; ok && k < dists.size(); ++k) {
	  if (vd[k] < SEPS) (*(dists[k]))(P, bv);
	}
	return d;
      }
      else { // essai raté ...
	vd[0] = (*(dists[0]))(P);
	bool ok = (vd[0] > -SEPS);
	for (size_type k = 1; k < dists.size(); ++k) {
	  vd[k] = (*(dists[k]))(P); if (vd[k] <= -SEPS) ok = false;
	}
	for (size_type k = 0; ok && k < dists.size(); ++k) {
	  if (vd[k] < SEPS) (*(dists[k]))(P, bv);
	}
	return operator()(P);
      }
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const {
      for (size_type k = 0; k < dists.size(); ++k)
	dists[k]->register_constraints(list); 
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      scalar_type d;
      if (with_min /* || gmm::abs(d) < SEPS*/) {
	d = (*(dists[0]))(P);
	size_type i = 0;
	for (size_type k = 1; k < dists.size(); ++k) {
	  scalar_type d2 = (*(dists[k]))(P);
	  if (d2 < d) { d = d2; i = k; }
	}
	return dists[i]->grad(P, G);
      }
      else { // essai raté ...
	d = operator()(P);
	base_small_vector Gloc;
	for (size_type k = 0; k < dists.size(); ++k) {
	  dists[k]->grad(P, Gloc);
	  if (isin) 
	    Gloc *= -gmm::neg(vd[k]);
	  else 
	    Gloc *= pow(d, scalar_type(dists.size())) / vd[k];
	  if (!k) G = Gloc; else G += Gloc;
	}
	if (isin)
	  G *= scalar_type(1) / d;
	else
	  G /= pow(d, scalar_type(dists.size()-1)) * scalar_type(dists.size());
	return d;
      }
    }
    void hess(const base_node &P, base_matrix &H) const {
      scalar_type d = (*(dists[0]))(P);
      if (with_min || gmm::abs(d) < SEPS) {
	size_type i = 0;
	for (size_type k = 1; k < dists.size(); ++k) {
	  scalar_type d2 = (*(dists[k]))(P);
	  if (d2 < d) { d = d2; i = k; }
	}
	dists[i]->hess(P, H);
      }
      else {
	GMM_ASSERT1(false, "Sorry, to be done");
      }
    }
  };

  inline pmesher_signed_distance new_mesher_union
  (const std::vector<pmesher_signed_distance> &dists)
  { return std::make_shared<mesher_union>(dists); }

  inline pmesher_signed_distance new_mesher_union
  (const pmesher_signed_distance &a,
   const pmesher_signed_distance &b,
   const pmesher_signed_distance &c = pmesher_signed_distance(),
   const pmesher_signed_distance &d = pmesher_signed_distance(),
   const pmesher_signed_distance &e = pmesher_signed_distance(),
   const pmesher_signed_distance &f = pmesher_signed_distance(),
   const pmesher_signed_distance &g = pmesher_signed_distance(),
   const pmesher_signed_distance &h = pmesher_signed_distance(),
   const pmesher_signed_distance &i = pmesher_signed_distance(),
   const pmesher_signed_distance &j = pmesher_signed_distance(),
   const pmesher_signed_distance &k = pmesher_signed_distance(),
   const pmesher_signed_distance &l = pmesher_signed_distance(),
   const pmesher_signed_distance &m = pmesher_signed_distance(),
   const pmesher_signed_distance &n = pmesher_signed_distance(),
   const pmesher_signed_distance &o = pmesher_signed_distance(),
   const pmesher_signed_distance &p = pmesher_signed_distance(),
   const pmesher_signed_distance &q = pmesher_signed_distance(),
   const pmesher_signed_distance &r = pmesher_signed_distance(),
   const pmesher_signed_distance &s = pmesher_signed_distance(),
   const pmesher_signed_distance &t = pmesher_signed_distance()) {
    return std::make_shared<mesher_union>
      (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t);
  }
 


  class mesher_intersection : public mesher_signed_distance {
    std::vector<pmesher_signed_distance> dists;
    mutable std::vector<scalar_type> vd;

    // const mesher_signed_distance &a, &b;
  public:
    
    mesher_intersection(const std::vector<pmesher_signed_distance>
			&dists_) : dists(dists_) 
    { vd.resize(dists.size()); }
    
    mesher_intersection
    (const pmesher_signed_distance &a,
     const pmesher_signed_distance &b,
     const pmesher_signed_distance &c = pmesher_signed_distance(),
     const pmesher_signed_distance &d = pmesher_signed_distance(),
     const pmesher_signed_distance &e = pmesher_signed_distance(),
     const pmesher_signed_distance &f = pmesher_signed_distance(),
     const pmesher_signed_distance &g = pmesher_signed_distance(),
     const pmesher_signed_distance &h = pmesher_signed_distance(),
     const pmesher_signed_distance &i = pmesher_signed_distance(),
     const pmesher_signed_distance &j = pmesher_signed_distance(),
     const pmesher_signed_distance &k = pmesher_signed_distance(),
     const pmesher_signed_distance &l = pmesher_signed_distance(),
     const pmesher_signed_distance &m = pmesher_signed_distance(),
     const pmesher_signed_distance &n = pmesher_signed_distance(),
     const pmesher_signed_distance &o = pmesher_signed_distance(),
     const pmesher_signed_distance &p = pmesher_signed_distance(),
     const pmesher_signed_distance &q = pmesher_signed_distance(),
     const pmesher_signed_distance &r = pmesher_signed_distance(),
     const pmesher_signed_distance &s = pmesher_signed_distance(),
     const pmesher_signed_distance &t = pmesher_signed_distance()) {
      dists.push_back(a); dists.push_back(b);
      if (c) dists.push_back(c);
      if (d) dists.push_back(d);
      if (e) dists.push_back(e);
      if (f) dists.push_back(f);
      if (g) dists.push_back(g);
      if (h) dists.push_back(h);
      if (i) dists.push_back(i);
      if (j) dists.push_back(j);
      if (k) dists.push_back(k);
      if (l) dists.push_back(l);
      if (m) dists.push_back(m);
      if (n) dists.push_back(n);
      if (o) dists.push_back(o);
      if (p) dists.push_back(p);
      if (q) dists.push_back(q);
      if (r) dists.push_back(r);
      if (s) dists.push_back(s);
      if (t) dists.push_back(t);
      vd.resize(dists.size());
    }
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      base_node bmin2, bmax2;
      bool first;
      bool b = dists[0]->bounding_box(bmin, bmax); first = !b;
      for (size_type k = 1; k < dists.size(); ++k) {
	bool bb = dists[k]->bounding_box(bmin2, bmax2);
	for (unsigned i=0; i < bmin.size() && bb && !first; ++i) { 
	  bmin[i] = std::max(bmin[i],bmin2[i]);
	  bmax[i] = std::max(std::min(bmax[i],bmax2[i]), bmin[i]);
	}
	if (first && bb) { bmin = bmin2; bmax = bmax2; first = false; }
	b = b || bb;
      }
      return b;
    }
    virtual scalar_type operator()(const base_node &P) const {
      scalar_type d = (*(dists[0]))(P);
      for (size_type k = 1; k < dists.size(); ++k)
	d = std::max(d, (*(dists[k]))(P));
      return d;

    }
    scalar_type operator()(const base_node &P, dal::bit_vector &bv) const {
      scalar_type d = vd[0] = (*(dists[0]))(P);
      bool ok = (d < SEPS);
      for (size_type k = 1; k < dists.size(); ++k) {
	vd[k] = (*(dists[k]))(P); if (vd[k] >= SEPS) ok = false;
	d = std::min(d,vd[k]);
      }
      for (size_type k = 0; ok && k < dists.size(); ++k) {
	if (vd[k] > -SEPS) (*(dists[k]))(P, bv);
      }
      return d;
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const {
      for (size_type k = 0; k < dists.size(); ++k) {
	dists[k]->register_constraints(list); 
      }
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      scalar_type d = (*(dists[0]))(P);
      size_type i = 0;
      for (size_type k = 1; k < dists.size(); ++k) {
	scalar_type d2 = (*(dists[k]))(P);
	if (d2 > d) { d = d2; i = k; }
      }
      return dists[i]->grad(P, G);
    }
    void hess(const base_node &P, base_matrix &H) const {
      scalar_type d = (*(dists[0]))(P);
      size_type i = 0;
      for (size_type k = 1; k < dists.size(); ++k) {
	scalar_type d2 = (*(dists[k]))(P);
	if (d2 > d) { d = d2; i = k; }
      }
      dists[i]->hess(P, H);
    }
  };

  inline pmesher_signed_distance new_mesher_intersection
  (const std::vector<pmesher_signed_distance> &dists)
  { return std::make_shared<mesher_intersection>(dists); }

  inline pmesher_signed_distance new_mesher_intersection
  (const pmesher_signed_distance &a,
   const pmesher_signed_distance &b,
   const pmesher_signed_distance &c = pmesher_signed_distance(),
   const pmesher_signed_distance &d = pmesher_signed_distance(),
   const pmesher_signed_distance &e = pmesher_signed_distance(),
   const pmesher_signed_distance &f = pmesher_signed_distance(),
   const pmesher_signed_distance &g = pmesher_signed_distance(),
   const pmesher_signed_distance &h = pmesher_signed_distance(),
   const pmesher_signed_distance &i = pmesher_signed_distance(),
   const pmesher_signed_distance &j = pmesher_signed_distance(),
   const pmesher_signed_distance &k = pmesher_signed_distance(),
   const pmesher_signed_distance &l = pmesher_signed_distance(),
   const pmesher_signed_distance &m = pmesher_signed_distance(),
   const pmesher_signed_distance &n = pmesher_signed_distance(),
   const pmesher_signed_distance &o = pmesher_signed_distance(),
   const pmesher_signed_distance &p = pmesher_signed_distance(),
   const pmesher_signed_distance &q = pmesher_signed_distance(),
   const pmesher_signed_distance &r = pmesher_signed_distance(),
   const pmesher_signed_distance &s = pmesher_signed_distance(),
   const pmesher_signed_distance &t = pmesher_signed_distance()) {
    return std::make_shared<mesher_intersection>
      (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t);
  }
 


  class mesher_setminus : public mesher_signed_distance {
    pmesher_signed_distance a, b;
  public:
    mesher_setminus(const pmesher_signed_distance& a_,
		    const pmesher_signed_distance &b_) : a(a_), b(b_) {}
    bool bounding_box(base_node &bmin, base_node &bmax) const
    { return a->bounding_box(bmin,bmax); }
    scalar_type operator()(const base_node &P, dal::bit_vector &bv) const {
      scalar_type da = (*a)(P), db = -(*b)(P);
      if (da < SEPS && db < SEPS) {
	if (da > -SEPS) (*a)(P, bv);
	if (db > -SEPS) (*b)(P, bv);
      }
      return std::max(da, db);
    }
    scalar_type operator()(const base_node &P) const
    { return std::max((*a)(P),-(*b)(P)); }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const {
      a->register_constraints(list); b->register_constraints(list);
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      scalar_type da = (*a)(P), db = -(*b)(P);
      if (da > db) return a->grad(P, G);
      else { b->grad(P, G); G *= scalar_type(-1); return db; }
    }
    void hess(const base_node &P, base_matrix &H) const {
      scalar_type da = (*a)(P), db = -(*b)(P);
      if (da > db) a->hess(P, H);
      else { b->hess(P, H); gmm::scale(H, scalar_type(-1)); }
    }

  };
  
  inline pmesher_signed_distance new_mesher_setminus
  (const pmesher_signed_distance &a, const pmesher_signed_distance &b)
  { return std::make_shared<mesher_setminus>(a, b); }


  // An object whose boundary is not a constraint of the mesher (used for the
  // lids of the pockets of a polygon).
  class mesher_unconstrained : public mesher_signed_distance {
    pmesher_signed_distance d;
  public:
    mesher_unconstrained(const pmesher_signed_distance &d_) : d(d_) {}
    bool bounding_box(base_node &bmin, base_node &bmax) const
    { return d->bounding_box(bmin, bmax); }
    virtual scalar_type operator()(const base_node &P) const
    { return (*d)(P); }
    virtual scalar_type operator()(const base_node &P, dal::bit_vector &)
      const { return (*d)(P); }
    scalar_type grad(const base_node &P, base_small_vector &G) const
    { return d->grad(P, G); }
    void hess(const base_node &P, base_matrix &H) const { d->hess(P, H); }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>&) const {}
  };

  // A simple polygon in 2D (not necessarily convex), given by its vertices
  // in either order. It is built from half planes, as the other objects:
  // its convex hull (intersection of half planes) minus its pockets (the
  // parts of the hull outside the polygon, polygons themselves, treated
  // likewise). The sides are thus constraints of the mesher, which keeps
  // the corners.
  class mesher_polygon : public mesher_signed_distance {
    std::vector<base_node> V;
    pmesher_signed_distance d;

    static scalar_type cross(const base_node &O, const base_node &A,
			     const base_node &B)
    { return (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0]); }

    // lid: for a pocket, the side shared with the hull of the enclosing
    // polygon, whose half plane is not repeated (the hull has it)
    static pmesher_signed_distance build(std::vector<base_node> P,
					 const base_node *lid = 0) {
      size_type n = P.size();
      scalar_type area(0), scale(0);
      for (size_type i = 0; i < n; ++i) {
	area += cross(P[0], P[i], P[(i+1) % n]);
	scale = std::max(scale, gmm::vect_dist2(P[i], P[0]));
      }
      if (area < 0) std::reverse(P.begin(), P.end());
      // convex hull (indices, counterclockwise), monotone chain
      std::vector<size_type> idx(n), H(2*n);
      for (size_type i = 0; i < n; ++i) idx[i] = i;
      std::sort(idx.begin(), idx.end(), [&P](size_type a, size_type b)
		{ return P[a][0] < P[b][0]
		    || (P[a][0] == P[b][0] && P[a][1] < P[b][1]); });
      scalar_type eps = 1e-12 * scale * scale;
      size_type k = 0;
      for (size_type i = 0; i < n; ++i) {
	while (k >= 2 && cross(P[H[k-2]], P[H[k-1]], P[idx[i]]) <= eps) --k;
	H[k++] = idx[i];
      }
      for (size_type i = n-1, t = k+1; i-- > 0; ) {
	while (k >= t && cross(P[H[k-2]], P[H[k-1]], P[idx[i]]) <= eps) --k;
	H[k++] = idx[i];
      }
      H.resize(k-1);
      // the hull follows the polygon in the same direction: start at the
      // smallest index, the indices then increase along the hull
      std::rotate(H.begin(), std::min_element(H.begin(), H.end()), H.end());
      std::vector<pmesher_signed_distance> sides, pockets;
      for (size_type h = 0; h < H.size(); ++h) {
	size_type a = H[h], b = H[(h+1) % H.size()];
	const base_node &A = P[a], &B = P[b];
	bool is_lid = lid
	  && ((gmm::vect_dist2(A, lid[0]) == 0 && gmm::vect_dist2(B, lid[1]) == 0)
	   || (gmm::vect_dist2(A, lid[1]) == 0 && gmm::vect_dist2(B, lid[0]) == 0));
	if (is_lid) continue;
	pmesher_signed_distance side = new_mesher_half_space
	  (A, base_small_vector(A[1]-B[1], B[0]-A[0]));
	if ((a + 1) % n != b) { // a pocket: the chain from a to b
	  std::vector<base_node> Q;
	  for (size_type i = a; ; i = (i + 1) % n)
	    { Q.push_back(P[i]); if (i == b) break; }
	  scalar_type qa(0);
	  for (size_type i = 0; i < Q.size(); ++i)
	    qa += cross(Q[0], Q[i], Q[(i+1) % Q.size()]);
	  base_node AB[2] = {A, B};
	  if (Q.size() >= 3 && gmm::abs(qa) > eps) {
	    pockets.push_back(build(Q, AB));
	    // the lid is not a side of the polygon: not a constraint either
	    // (an extra constraint at the vertices a and b disturbs the mesher)
	    side = std::make_shared<mesher_unconstrained>(side);
	  }
	}
	sides.push_back(side);
      }
      pmesher_signed_distance hull = (sides.size() == 1) ? sides[0]
	: new_mesher_intersection(sides);
      if (pockets.empty()) return hull;
      return new_mesher_setminus(hull, pockets.size() == 1 ? pockets[0]
				 : new_mesher_union(pockets));
    }

  public:
    mesher_polygon(const std::vector<base_node> &vertices) : V(vertices) {
      GMM_ASSERT1(V.size() >= 3, "A polygon needs at least three vertices");
      for (size_type i = 0; i < V.size(); ++i) {
	GMM_ASSERT1(V[i].size() == 2, "A polygon is defined in 2D");
	GMM_ASSERT1(gmm::vect_dist2(V[i], V[(i+1) % V.size()]) > 0,
		    "Two consecutive vertices of a polygon coincide");
      }
      d = build(V);
    }
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      bmin = bmax = V[0];
      for (const base_node &v : V)
	for (size_type k = 0; k < 2; ++k) {
	  bmin[k] = std::min(bmin[k], v[k]);
	  bmax[k] = std::max(bmax[k], v[k]);
	}
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const
    { return (*d)(P); }
    virtual scalar_type operator()(const base_node &P, dal::bit_vector &bv)
      const { return (*d)(P, bv); }
    scalar_type grad(const base_node &P, base_small_vector &G) const
    { return d->grad(P, G); }
    void hess(const base_node &P, base_matrix &H) const { d->hess(P, H); }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const
    { d->register_constraints(list); }
  };

  inline pmesher_signed_distance
  new_mesher_polygon(const std::vector<base_node> &vertices)
  { return std::make_shared<mesher_polygon>(vertices); }


  class mesher_tube : public mesher_signed_distance {
    base_node x0; base_node n; scalar_type R;
  public:
      mesher_tube(base_node x0_, base_node n_, scalar_type R_)
	: x0(x0_), n(n_), R(R_)
    { n /= gmm::vect_norm2(n); }
    bool bounding_box(base_node &, base_node &) const
    { return false; }
    virtual scalar_type operator()(const base_node &P) const {
      base_node v(P); v -= x0;
      gmm::add(gmm::scaled(n, -gmm::vect_sp(v, n)), v);
      return gmm::vect_norm2(v) - R;
    }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector &bv) const {
      scalar_type d = (*this)(P);
      bv[id] = (gmm::abs(d) < SEPS);
      return d;
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const {
      id = list.size(); list.push_back(this);
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      G = P; G -= x0;
      gmm::add(gmm::scaled(n, -gmm::vect_sp(G, n)), G);
      scalar_type e = gmm::vect_norm2(G), d = e - R;
      while (e == scalar_type(0)) {
	gmm::fill_random(G);
	gmm::add(gmm::scaled(n, -gmm::vect_sp(G, n)), G);
	e = gmm::vect_norm2(G);
      }
      G /= e;
      return d;
    }
    void hess(const base_node &, base_matrix &) const {
      GMM_ASSERT1(false, "Sorry, to be done");
    }
  };

  inline pmesher_signed_distance new_mesher_tube(base_node x0, base_node n,
						 scalar_type R)
  { return std::make_shared<mesher_tube>(x0,n,R); }


  class mesher_cylinder : public mesher_signed_distance {
    base_node x0; base_small_vector n;
    scalar_type L, R;
    pmesher_signed_distance t, p1, p2, i1;
  public:
    mesher_cylinder(const base_node &c, const base_small_vector &no,
		    scalar_type L_, scalar_type R_)
      : x0(c), n(no/gmm::vect_norm2(no)), L(L_), R(R_),
	t(new_mesher_tube(x0, n, R)),
	p1(new_mesher_half_space(x0, n)),
	p2(new_mesher_half_space(x0+n*L, -1.0 * n)),
	i1(new_mesher_intersection(p1, p2, t)) {}
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      base_node x1(x0+n*L);
      bmin = bmax = x0;
      for (unsigned i = 0; i < gmm::vect_size(x0); ++i) {
	bmin[i] = std::min(x0[i], x1[i]) - R;
	bmax[i] = std::max(x0[i], x1[i]) + R;
      }
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const {return (*i1)(P); }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector& bv) const
    { return (*i1)(P, bv); }
    scalar_type grad(const base_node &P, base_small_vector &G) const
      { return i1->grad(P, G); }
    void hess(const base_node &, base_matrix &) const {
      GMM_ASSERT1(false, "Sorry, to be done");
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const
    { i1->register_constraints(list); }
  };

  inline pmesher_signed_distance new_mesher_cylinder
  (const base_node &c, const base_small_vector &no,
   scalar_type L, scalar_type R)
  { return std::make_shared<mesher_cylinder>(c,no,L,R); }


  class mesher_infinite_cone : public mesher_signed_distance {
    // uses the true distance to a cone.
    base_node x0; base_node n; scalar_type alpha;
  public:
      mesher_infinite_cone(base_node x0_, base_node n_, scalar_type alpha_)
	: x0(x0_), n(n_), alpha(alpha_)
    { n /= gmm::vect_norm2(n); }
    bool bounding_box(base_node &, base_node &) const
    { return false; }
    virtual scalar_type operator()(const base_node &P) const {
      base_node v(P); v -= x0;
      scalar_type v_n = gmm::vect_sp(v, n);
      gmm::add(gmm::scaled(n, -v_n), v);
      return gmm::vect_norm2(v) * cos(alpha) - gmm::abs(v_n) * sin(alpha);
    }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector &bv) const {
      scalar_type d = (*this)(P);
      bv[id] = (gmm::abs(d) < SEPS);
      return d;
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const {
      id = list.size(); list.push_back(this);
    }
    scalar_type grad(const base_node &P, base_small_vector &v) const {
      v = P; v -= x0;
      scalar_type v_n = gmm::vect_sp(v, n);
      gmm::add(gmm::scaled(n, -v_n), v);
      scalar_type no = gmm::vect_norm2(v);
      scalar_type d = no * cos(alpha) - gmm::abs(v_n) * sin(alpha);
      while (no ==  scalar_type(0)) {
	gmm::fill_random(v);
	gmm::add(gmm::scaled(n, -gmm::vect_sp(v, n)), v);
	no = gmm::vect_norm2(v);
      }
      v *= cos(alpha) / no;
      v -= (sin(alpha) * gmm::sgn(v_n)) * n;
      return d;
    }
    void hess(const base_node &, base_matrix &) const {
      GMM_ASSERT1(false, "Sorry, to be done");
    }
  };

  inline pmesher_signed_distance new_mesher_infinite_cone
  (base_node x0, base_node n, scalar_type alpha)
  { return std::make_shared<mesher_infinite_cone>(x0,n,alpha); }


  class mesher_cone : public mesher_signed_distance {
    base_node x0; base_small_vector n;
    scalar_type L, alpha;
    pmesher_signed_distance t, p1, p2, i1;
  public:
    mesher_cone(const base_node &c, const base_small_vector &no,
		    scalar_type L_, scalar_type alpha_)
      : x0(c), n(no/gmm::vect_norm2(no)), L(L_), alpha(alpha_),
	t(new_mesher_infinite_cone(x0, n, alpha)),
	p1(new_mesher_half_space(x0, n)),
	p2(new_mesher_half_space(x0+n*L, -1.0 * n)),
	i1(new_mesher_intersection(p1, p2, t)) {}
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      base_node x1(x0+n*L);
      scalar_type R = L * sin(alpha);
      bmin = bmax = x0;
      for (unsigned i = 0; i < gmm::vect_size(x0); ++i) {
	bmin[i] = std::min(x0[i], x1[i]) - R;
	bmax[i] = std::max(x0[i], x1[i]) + R;
      }
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const {return (*i1)(P); }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector& bv) const
    { return (*i1)(P, bv); }
    scalar_type grad(const base_node &P, base_small_vector &G) const
      { return i1->grad(P, G); }
    void hess(const base_node &, base_matrix &) const {
      GMM_ASSERT1(false, "Sorry, to be done");
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const
    { i1->register_constraints(list); }
  };

  inline pmesher_signed_distance new_mesher_cone
  (const base_node &c, const base_small_vector &no,
   scalar_type L, scalar_type alpha)
  { return std::make_shared<mesher_cone>(c,no,L,alpha); }


  // Ellipse in 2D: center x0, semi-axis r along the direction no, semi-axis
  // R along the orthogonal direction. The signed distance is the euclidean
  // one (nearest point computed as in D. Eberly, "Distance from a point to
  // an ellipse, an ellipsoid, or a hyperellipsoid", 2013).
  class mesher_ellipse : public mesher_signed_distance {
    base_node x0; base_small_vector n, t;
    scalar_type r, R;

    static scalar_type root(scalar_type r0, scalar_type z0, scalar_type z1,
			    scalar_type g) {
      scalar_type n0 = r0*z0, s0 = z1 - 1.;
      scalar_type s1 = (g < 0.) ? 0. : sqrt(n0*n0 + z1*z1) - 1., s = 0.;
      for (int i = 0; i < 200; ++i) {
	s = (s0 + s1) / 2.;
	if (s == s0 || s == s1) break;
	scalar_type a = n0 / (s + r0), b = z1 / (s + 1.);
	g = a*a + b*b - 1.;
	if (g > 0.) s0 = s; else if (g < 0.) s1 = s; else break;
      }
      return s;
    }
    // e0 >= e1 > 0, y0, y1 >= 0: nearest point (q0, q1) of the ellipse
    static void nearest(scalar_type e0, scalar_type e1, scalar_type y0,
			scalar_type y1, scalar_type &q0, scalar_type &q1) {
      if (y1 > 0.) {
	if (y0 > 0.) {
	  scalar_type z0 = y0 / e0, z1 = y1 / e1, g = z0*z0 + z1*z1 - 1.;
	  if (g != 0.) {
	    scalar_type r0 = (e0/e1)*(e0/e1), sb = root(r0, z0, z1, g);
	    q0 = r0 * y0 / (sb + r0); q1 = y1 / (sb + 1.);
	  } else { q0 = y0; q1 = y1; }
	} else { q0 = 0.; q1 = e1; }
      } else {
	scalar_type numer0 = e0*y0, denom0 = e0*e0 - e1*e1;
	if (numer0 < denom0) {
	  scalar_type xde0 = numer0 / denom0;
	  q0 = e0 * xde0; q1 = e1 * sqrt(std::max(0., 1. - xde0*xde0));
	} else { q0 = e0; q1 = 0.; }
      }
    }
    // signed distance, local coordinates (a, b) of P and of its nearest
    // point (qa, qb) on the ellipse, along (n, t)
    scalar_type local(const base_node &P, scalar_type &a, scalar_type &b,
		      scalar_type &qa, scalar_type &qb) const {
      base_small_vector v(P); v -= x0;
      a = gmm::vect_sp(v, n); b = gmm::vect_sp(v, t);
      scalar_type q0, q1;
      if (r >= R) { nearest(r, R, gmm::abs(a), gmm::abs(b), q0, q1); qa = q0; qb = q1; }
      else        { nearest(R, r, gmm::abs(b), gmm::abs(a), q0, q1); qa = q1; qb = q0; }
      if (a < 0.) qa = -qa;
      if (b < 0.) qb = -qb;
      scalar_type d = sqrt((a-qa)*(a-qa) + (b-qb)*(b-qb));
      return ((a/r)*(a/r) + (b/R)*(b/R) < 1.) ? -d : d;
    }

  public:
    mesher_ellipse(const base_node &center, const base_small_vector &no,
		   scalar_type r_, scalar_type R_)
      : x0(center), n(no/gmm::vect_norm2(no)), t(-n[1], n[0]), r(r_), R(R_) {
      GMM_ASSERT1(x0.size() == 2 && no.size() == 2,
		  "An ellipse is defined in 2D");
      GMM_ASSERT1(r > 0. && R > 0., "The semi-axes should be positive");
    }
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      bmin = bmax = x0;
      for (unsigned i = 0; i < 2; ++i) {
	scalar_type e = sqrt(gmm::sqr(r*n[i]) + gmm::sqr(R*t[i]));
	bmin[i] -= e; bmax[i] += e;
      }
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const {
      scalar_type a, b, qa, qb;
      return local(P, a, b, qa, qb);
    }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector& bv) const {
      scalar_type d = this->operator()(P);
      bv[id] = (gmm::abs(d) < SEPS);
      return d;
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      scalar_type a, b, qa, qb, d = local(P, a, b, qa, qb), ga, gb;
      if (gmm::abs(d) > SEPS * std::min(r, R)) {
	ga = (a - qa) / gmm::abs(d); gb = (b - qb) / gmm::abs(d);
	if (d < 0.) { ga = -ga; gb = -gb; }
      } else {      // on the ellipse: its normal
	ga = qa / (r*r); gb = qb / (R*R);
	scalar_type e = sqrt(ga*ga + gb*gb); ga /= e; gb /= e;
      }
      G.resize(2);
      G[0] = ga*n[0] + gb*t[0]; G[1] = ga*n[1] + gb*t[1];
      return d;
    }
    void hess(const base_node &P, base_matrix &H) const {
      // finite differences of the gradient
      scalar_type eps = 1e-6 * std::min(r, R);
      base_small_vector G1, G2;
      gmm::resize(H, 2, 2);
      for (size_type j = 0; j < 2; ++j) {
	base_node P1(P), P2(P); P1[j] -= eps; P2[j] += eps;
	grad(P1, G1); grad(P2, G2);
	for (size_type i = 0; i < 2; ++i) H(i, j) = (G2[i] - G1[i]) / (2.*eps);
      }
      scalar_type h = (H(0, 1) + H(1, 0)) / 2.; H(0, 1) = H(1, 0) = h;
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const
    { id = list.size(); list.push_back(this); }
  };

  inline pmesher_signed_distance new_mesher_ellipse
  (const base_node &center, const base_small_vector &no,
   scalar_type r, scalar_type R)
  { return std::make_shared<mesher_ellipse>(center,no,r,R); }



  class mesher_torus : public mesher_signed_distance { // to be done
    // ajouter une rotation rigide et translation ..
    scalar_type R, r;
  public:
    mesher_torus(scalar_type RR, scalar_type rr) : R(RR), r(rr) {}
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      bmin = base_node(3); bmax = base_node(3);
      bmin[0] = bmin[1] = -R-r; bmin[2] = -r;
      bmax[0] = bmax[1] = +R+r; bmax[2] = +r;
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const {
      scalar_type x = P[0], y = P[1], z = P[2], c = sqrt(x*x + y*y);
      return (c == 0.) ? R - r : sqrt(gmm::sqr(c-R) + z*z) - r;
    }
    virtual scalar_type operator()(const base_node &P, dal::bit_vector&bv)
      const {
      scalar_type d = this->operator()(P);
      bv[id] = (gmm::abs(d) < SEPS);
      return d;
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      G.resize(3);
      scalar_type x = P[0], y = P[1], z = P[2], c = sqrt(x*x + y*y), d(0);
      if (c == 0.) {
	d = R - r;
	gmm::fill_random(G); G[2] = 0.0; G /= gmm::vect_norm2(G);
      }
      else {
	scalar_type w = 1. - R / c, e = sqrt(gmm::sqr(c-R) + z*z);
	d = e - r;
	if (e == 0.) {
	  gmm::fill_random(G); G[0] = x; G[1] = y; G /= gmm::vect_norm2(G);
	}
	else {
	  G[0] = x * w / e; G[1] = y * w / e; G[2] = z / e;
	}
      }
      return d;
    }
    void hess(const base_node &, base_matrix &) const {
      GMM_ASSERT1(false, "Sorry, to be done");
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>&list) const
    { id = list.size(); list.push_back(this); }
  };


  inline pmesher_signed_distance new_mesher_torus(scalar_type R, scalar_type r)
  { return std::make_shared<mesher_torus>(R,r); }


  // 3D objects built from a 2D one by a map (x, y, z) -> (a, b):
  // (x, y) for an extrusion along the z axis, (sqrt(x^2+y^2), z) for a
  // revolution about the z axis (the profile is then given in the (r, z)
  // half plane, r >= 0). Both maps preserve the distance to the boundary
  // (in the meridian plane for the revolution).

  inline scalar_type mesher_map_2d(const base_node &P, bool revolution,
				   base_node &Q, base_small_vector &e) {
    GMM_ASSERT1(P.size() == 3, "A 3D point is expected");
    Q.resize(2); e.resize(2);
    if (!revolution) { Q[0] = P[0]; Q[1] = P[1]; return scalar_type(0); }
    scalar_type r = sqrt(P[0]*P[0] + P[1]*P[1]);
    Q[0] = r; Q[1] = P[2];
    if (r > scalar_type(0)) { e[0] = P[0] / r; e[1] = P[1] / r; }
    else { e[0] = scalar_type(1); e[1] = scalar_type(0); }
    return r;
  }

  // Gradient (G2) and hessian (H2) of the 2D function brought back in 3D.
  inline void mesher_map_2d_grad(bool revolution, const base_small_vector &e,
				 const base_small_vector &G2,
				 base_small_vector &G) {
    G.resize(3);
    if (!revolution) { G[0] = G2[0]; G[1] = G2[1]; G[2] = scalar_type(0); }
    else { G[0] = G2[0]*e[0]; G[1] = G2[0]*e[1]; G[2] = G2[1]; }
  }

  inline void mesher_map_2d_hess(bool revolution, scalar_type r,
				 const base_small_vector &e,
				 const base_small_vector &G2,
				 const base_matrix &H2, base_matrix &H) {
    gmm::resize(H, 3, 3); gmm::clear(H);
    if (!revolution) {
      for (size_type i = 0; i < 2; ++i)
	for (size_type j = 0; j < 2; ++j) H(i, j) = H2(i, j);
      return;
    }
    // J = [e0 e1 0; 0 0 1], H = J^T H2 J + dd/dr * Hess(r)
    scalar_type J[2][3] = {{e[0], e[1], 0.}, {0., 0., 1.}};
    for (size_type i = 0; i < 3; ++i)
      for (size_type j = 0; j < 3; ++j)
	for (size_type k = 0; k < 2; ++k)
	  for (size_type l = 0; l < 2; ++l)
	    H(i, j) += J[k][i] * H2(k, l) * J[l][j];
    if (r > scalar_type(0))
      for (size_type i = 0; i < 2; ++i)
	for (size_type j = 0; j < 2; ++j)
	  H(i, j) += G2[0] * ((i == j ? 1. : 0.) - e[i]*e[j]) / r;
  }

  // A constraint (elementary boundary) of the 2D object, seen in 3D.
  class mesher_mapped_constraint : public mesher_signed_distance {
    const mesher_signed_distance *c;
    bool revolution;
  public:
    mesher_mapped_constraint(const mesher_signed_distance *c_, bool rev)
      : c(c_), revolution(rev) {}
    size_type constraint_id() const { return id; }
    bool bounding_box(base_node &, base_node &) const { return false; }
    virtual scalar_type operator()(const base_node &P) const {
      base_node Q; base_small_vector e;
      mesher_map_2d(P, revolution, Q, e);
      return (*c)(Q);
    }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector &bv) const {
      scalar_type d = (*this)(P);
      bv[id] = (gmm::abs(d) < SEPS);
      return d;
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      base_node Q; base_small_vector e, G2;
      mesher_map_2d(P, revolution, Q, e);
      scalar_type d = c->grad(Q, G2);
      mesher_map_2d_grad(revolution, e, G2, G);
      return d;
    }
    void hess(const base_node &P, base_matrix &H) const {
      base_node Q; base_small_vector e, G2; base_matrix H2;
      scalar_type r = mesher_map_2d(P, revolution, Q, e);
      c->grad(Q, G2); c->hess(Q, H2);
      mesher_map_2d_hess(revolution, r, e, G2, H2, H);
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const
    { id = list.size(); list.push_back(this); }
  };

  // A 2D object seen in 3D: an infinite prism along z (revolution = false)
  // or the solid of revolution about the z axis (revolution = true).
  // The constraints of the 2D object are registered in a list of their
  // own and replaced, in the 3D list, by mapped constraints.
  class mesher_mapped_2d : public mesher_signed_distance {
    pmesher_signed_distance d2;
    bool revolution;
    mutable std::vector<const mesher_signed_distance*> list2;
    mutable std::vector<std::shared_ptr<mesher_mapped_constraint>> mapped;
    mutable dal::bit_vector bv2;
  public:
    mesher_mapped_2d(const pmesher_signed_distance &d, bool rev)
      : d2(d), revolution(rev) {}
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      if (!revolution) return false;
      base_node b0, b1;
      if (!d2->bounding_box(b0, b1)) return false;
      scalar_type R = std::max(gmm::abs(b0[0]), gmm::abs(b1[0]));
      bmin = base_node(3); bmax = base_node(3);
      bmin[0] = bmin[1] = -R; bmax[0] = bmax[1] = R;
      bmin[2] = b0[1]; bmax[2] = b1[1];
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const {
      base_node Q; base_small_vector e;
      mesher_map_2d(P, revolution, Q, e);
      return (*d2)(Q);
    }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector &bv) const {
      base_node Q; base_small_vector e;
      mesher_map_2d(P, revolution, Q, e);
      bv2.clear();
      scalar_type d = (*d2)(Q, bv2);
      for (dal::bv_visitor i(bv2); !i.finished(); ++i)
	if (i < mapped.size()) bv[mapped[i]->constraint_id()] = true;
      return d;
    }
    scalar_type grad(const base_node &P, base_small_vector &G) const {
      base_node Q; base_small_vector e, G2;
      mesher_map_2d(P, revolution, Q, e);
      scalar_type d = d2->grad(Q, G2);
      mesher_map_2d_grad(revolution, e, G2, G);
      return d;
    }
    void hess(const base_node &P, base_matrix &H) const {
      base_node Q; base_small_vector e, G2; base_matrix H2;
      scalar_type r = mesher_map_2d(P, revolution, Q, e);
      d2->grad(Q, G2); d2->hess(Q, H2);
      mesher_map_2d_hess(revolution, r, e, G2, H2, H);
    }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const {
      list2.clear();
      d2->register_constraints(list2);
      mapped.clear();
      for (const mesher_signed_distance *c : list2) {
	mapped.push_back
	  (std::make_shared<mesher_mapped_constraint>(c, revolution));
	mapped.back()->register_constraints(list);
      }
    }
  };

  // Extrusion of a 2D object along the z axis, between z0 and z1.
  class mesher_extrusion : public mesher_signed_distance {
    pmesher_signed_distance d2, p, h1, h2, i1;
    scalar_type z0, z1;
  public:
    mesher_extrusion(const pmesher_signed_distance &d,
		     scalar_type z0_, scalar_type z1_)
      : d2(d), p(std::make_shared<mesher_mapped_2d>(d, false)),
	z0(std::min(z0_, z1_)), z1(std::max(z0_, z1_)) {
      base_node x0(0., 0., z0), x1(0., 0., z1);
      base_small_vector n(0., 0., 1.);
      h1 = new_mesher_half_space(x0, n);
      h2 = new_mesher_half_space(x1, -1.0 * n);
      i1 = new_mesher_intersection(p, h1, h2);
    }
    bool bounding_box(base_node &bmin, base_node &bmax) const {
      base_node b0, b1;
      if (!d2->bounding_box(b0, b1)) return false;
      bmin = base_node(b0[0], b0[1], z0);
      bmax = base_node(b1[0], b1[1], z1);
      return true;
    }
    virtual scalar_type operator()(const base_node &P) const
    { return (*i1)(P); }
    virtual scalar_type operator()(const base_node &P,
				   dal::bit_vector &bv) const
    { return (*i1)(P, bv); }
    scalar_type grad(const base_node &P, base_small_vector &G) const
    { return i1->grad(P, G); }
    void hess(const base_node &P, base_matrix &H) const { i1->hess(P, H); }
    virtual void register_constraints(std::vector<const
				      mesher_signed_distance*>& list) const
    { i1->register_constraints(list); }
  };

  inline pmesher_signed_distance new_mesher_extrusion
  (const pmesher_signed_distance &d2, scalar_type z0, scalar_type z1)
  { return std::make_shared<mesher_extrusion>(d2, z0, z1); }

  inline pmesher_signed_distance new_mesher_revolution
  (const pmesher_signed_distance &d2)
  { return std::make_shared<mesher_mapped_2d>(d2, true); }

  // mesher
  void build_mesh(mesh &m, const pmesher_signed_distance& dist_,
		  scalar_type h0, const std::vector<base_node> &fixed_points
		  = std::vector<base_node>(), size_type K = 1, int noise = -1,
		  size_type iter_max = 500, int prefind = 1,
		  scalar_type dist_point_hull = 4,
		  scalar_type boundary_threshold_flatness = 0.11);

  // exported functions
  bool try_projection(const mesher_signed_distance& dist, base_node &X,
		      bool on_surface = false);
  bool pure_multi_constraint_projection
  (const std::vector<const mesher_signed_distance*> &list_constraints,
   base_node &X, const dal::bit_vector &cts);
  scalar_type curvature_radius_estimate(const mesher_signed_distance &dist,
					base_node X, bool proj = false);
  scalar_type min_curvature_radius_estimate
  (const std::vector<const mesher_signed_distance*> &list_constraints,
   const base_node &X, const dal::bit_vector &cts, size_type hide_first = 0);

}

#endif /* GETFEM_MESHER_H__ */
