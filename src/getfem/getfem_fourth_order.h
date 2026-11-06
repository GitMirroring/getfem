/* -*- c++ -*- (enables emacs c++ mode) */
/*===========================================================================

 Copyright (C) 2006-2026 Yves Renard

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

/**@file getfem_fourth_order.h
   @author  Yves Renard <Yves.Renard@insa-lyon.fr>,
            Julien Pommier <Julien.Pommier@insa-toulouse.fr>
   @date January 6, 2006.
   @brief assembly procedures and bricks for fourth order pdes.
*/
#ifndef GETFEM_FOURTH_ORDER_H__
#define GETFEM_FOURTH_ORDER_H__

#include "getfem_models.h"
#include "getfem_assembling.h"

namespace getfem {
  
  /* ******************************************************************** */
  /*            Bilaplacian assembly routines.                            */
  /* ******************************************************************** */

  /**
     assembly of @f$\int_\Omega \Delta u \Delta v@f$.
     @ingroup asm
  */
  template<typename MAT, typename VECT>
  void asm_stiffness_matrix_for_bilaplacian
  (const MAT &M, const mesh_im &mim, const mesh_fem &mf,
   const mesh_fem &mf_data, const VECT &A,
   const mesh_region &rg = mesh_region::all_convexes()) {
    asm_real_or_complex_1_param_mat
      (const_cast<MAT &>(M), mim, mf, &mf_data, A, rg,
       "A*Trace(Hess_Test_u)*Trace(Hess_Test2_u)");
  }

  template<typename MAT, typename VECT>
  void asm_stiffness_matrix_for_homogeneous_bilaplacian
  (const MAT &M, const mesh_im &mim, const mesh_fem &mf,
   const VECT &A, const mesh_region &rg = mesh_region::all_convexes()) {
    asm_real_or_complex_1_param_mat
      (const_cast<MAT &>(M), mim, mf, 0, A, rg,
       "A*Trace(Hess_Test_u)*Trace(Hess_Test2_u)");
  }

  /* Common part of the Kirchhoff-Love stiffness matrices. */
  template<typename MAT, typename VECT>
  void asm_stiffness_matrix_for_bilaplacian_KL_
  (const MAT &M, const mesh_im &mim, const mesh_fem &mf,
   const mesh_fem *mf_data, const VECT &D_, const VECT &nu_,
   const mesh_region &rg) {
    ga_workspace workspace;
    gmm::sub_interval Iu(0, mf.nb_dof());
    base_vector u(mf.nb_dof()), D(gmm::vect_size(D_)), nu(gmm::vect_size(nu_));
    gmm::copy(D_, D);
    gmm::copy(nu_, nu);
    workspace.add_fem_variable("u", mf, Iu, u);
    if (mf_data) {
      workspace.add_fem_constant("D", *mf_data, D);
      workspace.add_fem_constant("nu", *mf_data, nu);
    } else {
      workspace.add_fixed_size_constant("D", D);
      workspace.add_fixed_size_constant("nu", nu);
    }
    workspace.add_expression("D*((1-nu)*(Hess_Test_u:Hess_Test2_u)"
                             "+nu*Trace(Hess_Test_u)*Trace(Hess_Test2_u))",
                             mim, rg);
    workspace.assembly(2);
    if (gmm::mat_nrows(workspace.assembled_matrix()))
      gmm::add(workspace.assembled_matrix(), const_cast<MAT &>(M));
  }

  template<typename MAT, typename VECT>
  void asm_stiffness_matrix_for_bilaplacian_KL
  (const MAT &M, const mesh_im &mim, const mesh_fem &mf,
   const mesh_fem &mf_data, const VECT &D_, const VECT &nu_,
   const mesh_region &rg = mesh_region::all_convexes()) {
    asm_stiffness_matrix_for_bilaplacian_KL_(M, mim, mf, &mf_data, D_, nu_, rg);
  }

  template<typename MAT, typename VECT>
  void asm_stiffness_matrix_for_homogeneous_bilaplacian_KL
  (const MAT &M, const mesh_im &mim, const mesh_fem &mf,
   const VECT &D_, const VECT &nu_,
   const mesh_region &rg = mesh_region::all_convexes()) {
    asm_stiffness_matrix_for_bilaplacian_KL_(M, mim, mf, 0, D_, nu_, rg);
  }

  /* ******************************************************************** */
  /*            Bilaplacian bricks.                                       */
  /* ******************************************************************** */

  
  /** Adds a bilaplacian brick on the variable
      `varname` and on the mesh region `region`. 
      This represent a term :math:`\Delta(D \Delta u)`. 
      where :math:`D(x)` is a coefficient determined by `dataname` which
      could be constant or described on a f.e.m. The corresponding weak form
      is :math:`\int D(x)\Delta u(x) \Delta v(x) dx`.
  */
  size_type add_bilaplacian_brick
  (model &md, const mesh_im &mim, const std::string &varname,
   const std::string &dataname, size_type region = size_type(-1));

  /** Adds a bilaplacian brick on the variable
      `varname` and on the mesh region `region`.
      This represent a term :math:`\Delta(D \Delta u)` where :math:`D(x)`
      is a the flexion modulus determined by `dataname1`. The term is
      integrated by part following a Kirchhoff-Love plate model
      with `dataname2` the poisson ratio.
  */
  size_type add_bilaplacian_brick_KL
  (model &md, const mesh_im &mim, const std::string &varname,
   const std::string &dataname1, const std::string &dataname2,
   size_type region = size_type(-1));


  /* ******************************************************************** */
  /*            Normale derivative source term assembly routines.         */
  /* ******************************************************************** */

  /**
     assembly of @f$\int_\Gamma{\partial_n u f}@f$.
     @ingroup asm
  */
  template<typename VECT1, typename VECT2>
  void asm_normal_derivative_source_term
  (VECT1 &B, const mesh_im &mim, const mesh_fem &mf, const mesh_fem &mf_data,
   const VECT2 &F, const mesh_region &rg) {
    GMM_ASSERT1(mf_data.get_qdim() == 1,
                "invalid data mesh fem (Qdim=1 required)");

    size_type Q = gmm::vect_size(F) / mf_data.nb_dof();

    // const char *s;
    // if (mf.get_qdim() == 1 && Q == 1)
    //   s = "F=data(#2);"
    //     "V(#1)+=comp(Grad(#1).Normal().Base(#2))(:,i,i,j).F(j);";
    // else if (mf.get_qdim() == 1 && Q == gmm::sqr(mf.linked_mesh().dim()))
    //   s = "F=data(mdim(#1),mdim(#1),#2);"
    //     "V(#1)+=comp(Grad(#1).Normal().Normal().Normal().Base(#2))"
    //     "(:,i,i,k,l,j).F(k,l,j);";
    // else if (mf.get_qdim() > size_type(1) && Q == mf.get_qdim())
    //   s = "F=data(qdim(#1),#2);"
    //     "V(#1)+=comp(vGrad(#1).Normal().Base(#2))(:,i,k,k,j).F(i,j);";
    // else if (mf.get_qdim() > size_type(1) &&
    //          Q == size_type(mf.get_qdim()*gmm::sqr(mf.linked_mesh().dim())))
    //   s = "F=data(qdim(#1),mdim(#1),mdim(#1),#2);"
    //     "V(#1)+=comp(vGrad(#1).Normal().Normal().Normal().Base(#2))"
    //     "(:,i,k,k,l,m,j).F(i,l,m,j);";
    // else
    //   GMM_ASSERT1(false, "invalid rhs vector");
    // asm_real_or_complex_1_param(B, mim, mf, mf_data, F, rg, s);

    const char *s;
    if (mf.get_qdim() == 1 && Q == 1)
      s = "Grad_Test_u.(A*Normal)";
    else if (mf.get_qdim() == 1 && Q == gmm::sqr(mf.linked_mesh().dim()))
      s = "Grad_Test_u.(((Reshape(A,meshdim,meshdim)*Normal).Normal)*Normal)";
    else if (mf.get_qdim() > size_type(1) && Q == mf.get_qdim())
      s = "((Grad_Test_u')*A).Normal";
    else if (mf.get_qdim() > size_type(1) &&
             Q == size_type(mf.get_qdim()*gmm::sqr(mf.linked_mesh().dim())))
      s = "((((Grad_Test_u').Reshape(A,qdim(u),meshdim,meshdim)).Normal).Normal).Normal";
    else
      GMM_ASSERT1(false, "invalid rhs vector");
    asm_real_or_complex_1_param_vec(B, mim, mf, &mf_data, F, rg, s);
  }

  template<typename VECT1, typename VECT2>
  void asm_homogeneous_normal_derivative_source_term
  (VECT1 &B, const mesh_im &mim, const mesh_fem &mf,
   const VECT2 &F, const mesh_region &rg) {
    
    size_type Q = gmm::vect_size(F);

    // const char *s;
    // if (mf.get_qdim() == 1 && Q == 1)
    //   s = "F=data(1);"
    //     "V(#1)+=comp(Grad(#1).Normal())(:,i,i).F(1);";
    // else if (mf.get_qdim() == 1 && Q == gmm::sqr(mf.linked_mesh().dim()))
    //   s = "F=data(mdim(#1),mdim(#1));"
    //     "V(#1)+=comp(Grad(#1).Normal().Normal().Normal())"
    //     "(:,i,i,l,j).F(l,j);";
    // else if (mf.get_qdim() > size_type(1) && Q == mf.get_qdim())
    //   s = "F=data(qdim(#1));"
    //     "V(#1)+=comp(vGrad(#1).Normal())(:,i,k,k).F(i);";
    // else if (mf.get_qdim() > size_type(1) &&
    //          Q == size_type(mf.get_qdim()*gmm::sqr(mf.linked_mesh().dim())))
    //   s = "F=data(qdim(#1),mdim(#1),mdim(#1));"
    //     "V(#1)+=comp(vGrad(#1).Normal().Normal().Normal())"
    //     "(:,i,k,k,l,m).F(i,l,m);";
    // else
    //   GMM_ASSERT1(false, "invalid rhs vector");
    // asm_real_or_complex_1_param(B, mim, mf, mf, F, rg, s);

    const char *s;
    if (mf.get_qdim() == 1 && Q == 1)
      s = "Test_Grad_u.(A*Normal)";
    else if (mf.get_qdim() == 1 && Q == gmm::sqr(mf.linked_mesh().dim()))
      s = "Test_Grad_u.(((Reshape(A,meshdim,meshdim)*Normal).Normal)*Normal)";
    else if (mf.get_qdim() > size_type(1) && Q == mf.get_qdim())
      s = "((Test_Grad_u')*A).Normal";
    else if (mf.get_qdim() > size_type(1) &&
             Q == size_type(mf.get_qdim()*gmm::sqr(mf.linked_mesh().dim())))
      s = "((((Test_Grad_u').Reshape(A,qdim(u),meshdim,meshdim)).Normal).Normal).Normal";
    else
      GMM_ASSERT1(false, "invalid rhs vector");
    asm_real_or_complex_1_param_vec(B, mim, mf, 0, F, rg, s);
  }


  /* ******************************************************************** */
  /*                Normale derivative source term brick.                 */
  /* ******************************************************************** */


  /** Adds a normal derivative source term brick
      :math:`F = \int b.\partial_n v` on the variable `varname` and the
      mesh region `region`.
     
      Update the right hand side of the linear system.
      `dataname` represents `b` and `varname` represents `v`.
  */
  size_type add_normal_derivative_source_term_brick
  (model &md, const mesh_im &mim, const std::string &varname,
   const std::string &dataname, size_type region);


  /* ******************************************************************** */
  /*           Special boundary condition for Kirchhoff-Love model.       */
  /* ******************************************************************** */

  /*
    assembly of the special boundary condition for Kirchhoff-Love model.
    @ingroup asm
  */
  /* Common part of the Kirchhoff-Love Neumann terms. */
  template<typename VECT1, typename VECT2>
  void asm_neumann_KL_term_
  (VECT1 &B, const mesh_im &mim, const mesh_fem &mf, const mesh_fem *mf_data,
   const VECT2 &M_, const VECT2 &divM_, const mesh_region &rg) {
    ga_workspace workspace;
    gmm::sub_interval Iu(0, mf.nb_dof());
    base_vector u(mf.nb_dof()), M(gmm::vect_size(M_));
    base_vector divM(gmm::vect_size(divM_));
    gmm::copy(M_, M);
    gmm::copy(divM_, divM);
    workspace.add_fem_variable("u", mf, Iu, u);
    if (mf_data) {
      workspace.add_fem_constant("M", *mf_data, M);
      workspace.add_fem_constant("divM", *mf_data, divM);
    } else {
      workspace.add_fixed_size_constant("M", M);
      workspace.add_fixed_size_constant("divM", divM);
    }
    workspace.add_expression
      ("(divM.Normal)*Test_u"
       "-Grad_Test_u.(Reshape(M,meshdim,meshdim)*Normal)"
       "+(Grad_Test_u.Normal)*(Normal.(Reshape(M,meshdim,meshdim)*Normal))",
       mim, rg);
    base_vector V(mf.nb_dof());
    workspace.set_assembled_vector(V);
    workspace.assembly(1);
    gmm::add(V, B);
  }

  template<typename VECT1, typename VECT2>
  void asm_neumann_KL_term
  (VECT1 &B, const mesh_im &mim, const mesh_fem &mf, const mesh_fem &mf_data,
   const VECT2 &M, const VECT2 &divM, const mesh_region &rg) {
    GMM_ASSERT1(mf_data.get_qdim() == 1,
                "invalid data mesh fem (Qdim=1 required)");
    asm_neumann_KL_term_(B, mim, mf, &mf_data, M, divM, rg);
  }

  template<typename VECT1, typename VECT2>
  void asm_neumann_KL_homogeneous_term
  (VECT1 &B, const mesh_im &mim, const mesh_fem &mf,
   const VECT2 &M, const VECT2 &divM, const mesh_region &rg) {
    asm_neumann_KL_term_(B, mim, mf, 0, M, divM, rg);
  }

  /* ******************************************************************** */
  /*                Kirchhoff Love Neumann term brick.                    */
  /* ******************************************************************** */


  /** Adds a Neumann term brick for Kirchhoff-Love model
      on the variable `varname` and the mesh region `region`.
      `dataname1` represents the bending moment tensor and  `dataname2`
      its divergence.
  */
  size_type add_Kirchhoff_Love_Neumann_term_brick
  (model &md, const mesh_im &mim, const std::string &varname,
   const std::string &dataname1, const std::string &dataname2,
   size_type region);


  /* ******************************************************************** */
  /*            Normal derivative Dirichlet assembly routines.            */
  /* ******************************************************************** */

  /**
     Assembly of normal derivative Dirichlet constraints
     @f$ \partial_n u(x) = r(x) @f$ in a weak form
     @f[ \int_{\Gamma} \partial_n u(x)v(x)=\int_{\Gamma} r(x)v(x) \forall v@f],
     where @f$ v @f$ is in
     the space of multipliers corresponding to mf_mult.

     size(r_data) = Q   * nb_dof(mf_rh);

     version = |ASMDIR_BUILDH : build H
     |ASMDIR_BUILDR : build R
     |ASMDIR_BUILDALL : do everything.

     @ingroup asm
  */

  template<typename MAT, typename VECT1, typename VECT2>
  void asm_normal_derivative_dirichlet_constraints
  (MAT &H, VECT1 &R, const mesh_im &mim, const mesh_fem &mf_u,
   const mesh_fem &mf_mult, const mesh_fem &mf_r,
   const VECT2 &r_data, const mesh_region &rg, bool R_must_be_derivated, 
   int version) {
    typedef typename gmm::linalg_traits<VECT1>::value_type value_type;
    typedef typename gmm::number_traits<value_type>::magnitude_type magn_type;
    
    rg.from_mesh(mim.linked_mesh()).error_if_not_faces();
    
    if (version & ASMDIR_BUILDH) {
      ga_workspace workspace;
      gmm::sub_interval Imult(0, mf_mult.nb_dof());
      gmm::sub_interval Iu(mf_mult.nb_dof(), mf_u.nb_dof());
      base_vector mult(mf_mult.nb_dof()), u(mf_u.nb_dof());
      workspace.add_fem_variable("mult", mf_mult, Imult, mult);
      workspace.add_fem_variable("u", mf_u, Iu, u);
      if (mf_u.get_qdim() == 1 && mf_mult.get_qdim() == 1)
        workspace.add_expression("Test_mult*(Grad_Test2_u.Normal)", mim, rg);
      else
        workspace.add_expression("Test_mult.(Grad_Test2_u*Normal)", mim, rg);
      workspace.assembly(2);
      if (gmm::mat_nrows(workspace.assembled_matrix()))
        gmm::add(gmm::sub_matrix(workspace.assembled_matrix(), Imult, Iu), H);
      gmm::clean(H, gmm::default_tol(magn_type())
                 * gmm::mat_maxnorm(H) * magn_type(1000));
    }
    if (version & ASMDIR_BUILDR) {
      GMM_ASSERT1(mf_r.get_qdim() == 1,
                "invalid data mesh fem (Qdim=1 required)");
      if (!R_must_be_derivated) {
        asm_normal_source_term(R, mim, mf_mult, mf_r, r_data, rg);
      } else {
	asm_real_or_complex_1_param_vec(R, mim, mf_mult, &mf_r, r_data, rg,
					"(Grad_A.Normal)*Test_u");
        // asm_real_or_complex_1_param
        //   (R, mim, mf_mult, mf_r, r_data, rg,
        //    "R=data(#2); V(#1)+=comp(Base(#1).Grad(#2).Normal())(:,i,j,j).R(i)");
      }
    }
  }

  /* ******************************************************************** */
  /*            Normal derivative Dirichlet condition bricks.             */
  /* ******************************************************************** */

  /** Adds a Dirichlet condition on the normal derivative of the variable
      `varname` and on the mesh region `region` (which should be a boundary). 
      The general form is
      :math:`\int \partial_n u(x)v(x) = \int r(x)v(x) \forall v`
      where :math:`r(x)` is
      the right hand side for the Dirichlet condition (0 for
      homogeneous conditions) and :math:`v` is in a space of multipliers
      defined by the variable `multname` on the part of boundary determined
      by `region`. `dataname` is an optional parameter which represents
      the right hand side of the Dirichlet condition.
      If `R_must_be_derivated` is set to `true` then the normal
      derivative of `dataname` is considered.
  */
  size_type add_normal_derivative_Dirichlet_condition_with_multipliers
  (model &md, const mesh_im &mim, const std::string &varname,
   const std::string &multname, size_type region,
   const std::string &dataname = std::string(),
   bool R_must_be_derivated = false);
  

  /** Adds a Dirichlet condition on the normal derivative of the variable
      `varname` and on the mesh region `region` (which should be a boundary). 
      The general form is
      :math:`\int \partial_n u(x)v(x) = \int r(x)v(x) \forall v`
      where :math:`r(x)` is
      the right hand side for the Dirichlet condition (0 for
      homogeneous conditions) and :math:`v` is in a space of multipliers
      defined by the trace of mf_mult on the part of boundary determined
      by `region`. `dataname` is an optional parameter which represents
      the right hand side of the Dirichlet condition.
      If `R_must_be_derivated` is set to `true` then the normal
      derivative of `dataname` is considered.
  */
  size_type add_normal_derivative_Dirichlet_condition_with_multipliers
  (model &md, const mesh_im &mim, const std::string &varname,
   const mesh_fem &mf_mult, size_type region,
   const std::string &dataname = std::string(),
   bool R_must_be_derivated = false);

  /** Adds a Dirichlet condition on the normal derivative of the variable
      `varname` and on the mesh region `region` (which should be a boundary). 
      The general form is
      :math:`\int \partial_n u(x)v(x) = \int r(x)v(x) \forall v`
      where :math:`r(x)` is
      the right hand side for the Dirichlet condition (0 for
      homogeneous conditions) and :math:`v` is in a space of multipliers
      defined by the trace of a Lagranfe finite element method of degree
      `degree` and on the boundary determined
      by `region`. `dataname` is an optional parameter which represents
      the right hand side of the Dirichlet condition.
      If `R_must_be_derivated` is set to `true` then the normal
      derivative of `dataname` is considered.
  */
  size_type add_normal_derivative_Dirichlet_condition_with_multipliers
  (model &md, const mesh_im &mim, const std::string &varname,
   dim_type degree, size_type region,
   const std::string &dataname = std::string(),
   bool R_must_be_derivated = false);

    /** Adds a Dirichlet condition on the normal derivative of the variable
      `varname` and on the mesh region `region` (which should be a boundary). 
      The general form is
      :math:`\int \partial_n u(x)v(x) = \int r(x)v(x) \forall v`
      where :math:`r(x)` is
      the right hand side for the Dirichlet condition (0 for
      homogeneous conditions). For this brick the condition is enforced with
      a penalisation with a penanalization parameter `penalization_coeff` on
      the boundary determined by `region`.
      `dataname` is an optional parameter which represents
      the right hand side of the Dirichlet condition.
      If `R_must_be_derivated` is set to `true` then the normal
      derivative of `dataname` is considered.
      Note that it is possible to change the penalization coefficient
      using the function `getfem::change_penalization_coeff` of the standard
      Dirichlet condition.
  */
  size_type add_normal_derivative_Dirichlet_condition_with_penalization
  (model &md, const mesh_im &mim, const std::string &varname,
   scalar_type penalisation_coeff, size_type region, 
   const std::string &dataname = std::string(),
   bool R_must_be_derivated = false);
  





}  /* end of namespace getfem.                                             */


#endif /* GETFEM_FOURTH_ORDER_H__ */
