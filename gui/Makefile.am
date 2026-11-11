#  Copyright (C) 2026 Yves Renard, Konstantinos Poulios
#
#  This file is a part of GetFEM
#
#  GetFEM is free software;  you can  redistribute it  and/or modify it under
#  the  terms  of the  GNU  Lesser General Public License as published by the
#  Free Software Foundation;  either version 3  of  the License,  or (at your
#  option) any  later  version  along with  the GCC Runtime Library Exception
#  either version 3.1 or (at your option) any later version.
#  This program  is  distributed  in  the  hope  that it will be useful,  but
#  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or  FITNESS  FOR  A PARTICULAR PURPOSE.  See the GNU Lesser General Public
#  License and GCC Runtime Library Exception for more details.
#  You  should  have received a copy of the GNU Lesser General Public License
#  along  with  this program;  if not, write to the Free Software Foundation,
#  Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301, USA.

# The graphical interface: a pure Python package, not built nor installed by
# make (see README.md). `make check` runs its tests, each test file as a test,
# with the Python interface of GetFEM built here; `make -j check` runs them in
# parallel. A test file whose tests all need what is missing (pytest, PySide6,
# PyVista, Gmsh...) is reported as skipped.

EXTRA_DIST = README.md pyproject.toml getfem_gui examples doc tests/check.py

GUI_TESTS =                                     \
    tests/test_app.py                           \
    tests/test_coupling.py                      \
    tests/test_curved.py                        \
    tests/test_doc.py                           \
    tests/test_editing.py                       \
    tests/test_elasticity.py                    \
    tests/test_ellipse.py                       \
    tests/test_export.py                        \
    tests/test_faces3d.py                       \
    tests/test_fillets.py                       \
    tests/test_geometry2d.py                    \
    tests/test_geometry3d.py                    \
    tests/test_gwfl.py                          \
    tests/test_heat.py                          \
    tests/test_manifolds.py                     \
    tests/test_meshers.py                       \
    tests/test_parameters.py                    \
    tests/test_polygon.py                       \
    tests/test_scene.py                         \
    tests/test_sketch.py                        \
    tests/test_splines.py                       \
    tests/test_transforms.py                    \
    tests/test_work_planes.py

EXTRA_DIST += $(GUI_TESTS)

if BUILDPYTHON
TESTS = $(GUI_TESTS)

AM_TESTS_ENVIRONMENT = \
    export PYTHONPATH=$(abs_top_builddir)/interface/src/python:$(abs_srcdir):$(PYTHONPATH); \
    export LD_LIBRARY_PATH=$(abs_top_builddir)/src/.libs:$(LD_LIBRARY_PATH); \
    export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1;
LOG_COMPILER = $(PYTHON)
AM_LOG_FLAGS = $(srcdir)/tests/check.py
endif

# no bytecode caches in the distribution
dist-hook:
	find $(distdir) -name __pycache__ -prune -exec rm -rf {} +
