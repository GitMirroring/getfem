/*===========================================================================

 Copyright (C) 2011-2026 Yves Renard.

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
// $Id: gf_mesher_object_get.cc 3468 2010-02-24 20:12:15Z renard $

#include <getfemint.h>
#include <getfem/getfem_mesher.h>

using namespace getfemint;

/*@GFDOC
    General function for querying information about mesher_object objects.
@*/


void gf_mesher_object_get(getfemint::mexargs_in& in,
                          getfemint::mexargs_out& out) {

  if (in.narg() < 2) THROW_BADARG("Wrong number of input arguments");

  getfem::pmesher_signed_distance paf = to_mesher_object(in.pop());
  std::string init_cmd                = in.pop().to_string();
  std::string cmd                     = cmd_normalize(init_cmd);

  if (check_cmd(cmd, "char", in, out, 0, 0, 0, 1)) {
    /*@GET s = ('char')
      Output a (unique) string representation of the @tmo.

      This can be used to perform comparisons between two
      different @tmo objects.
      This function is to be completed.@*/
   GMM_ASSERT1(false, "Sorry, function to be done");
   // std::string s = ...;
   // out.pop().from_string(s.c_str());
  } else if (check_cmd(cmd, "distance", in, out, 1, 1, 0, 1)) {
    /*@GET d = ('distance', @mat PTS)
      The signed distance of the @tmo at the points `PTS` (one point per
      column): negative inside, zero on its boundary (not exactly the
      euclidean distance for every object, see the mesher).@*/
    darray P = in.pop().to_darray(-1, -1);
    darray d = out.pop().create_darray_h(unsigned(P.getn()));
    getfem::base_node X(P.getm());
    for (unsigned j = 0; j < P.getn(); ++j) {
      gmm::copy(P.col_to_bn(j), X);
      d[j] = (*paf)(X);
    }
  } else if (check_cmd(cmd, "project", in, out, 1, 1, 0, 1)) {
    /*@GET Q = ('project', @mat PTS)
      The points `PTS` (one point per column) projected on the boundary
      of the @tmo (on its zero level, by Newton iterations along the
      gradient of its signed distance), e.g. the new boundary nodes of a
      refined mesh. A point already on it stays; a point whose projection
      fails stays too.@*/
    darray P = in.pop().to_darray(-1, -1);
    darray Q = out.pop().create_darray(unsigned(P.getm()), unsigned(P.getn()));
    getfem::base_node X(P.getm()), Y(P.getm());
    for (unsigned j = 0; j < P.getn(); ++j) {
      gmm::copy(P.col_to_bn(j), X);
      gmm::copy(X, Y);
      if (gmm::abs((*paf)(X)) > 1e-12 && !getfem::try_projection(*paf, Y, true))
        gmm::copy(X, Y);
      for (unsigned k = 0; k < P.getm(); ++k) Q(k, j) = Y[k];
    }
  } else if (check_cmd(cmd, "display", in, out, 0, 0, 0, 0)) {
    /*@GET ('display')
      displays a short summary for a @tmo object.@*/
    infomsg() << "gfMesherObject object\n";
  } else
    bad_cmd(init_cmd);

}
