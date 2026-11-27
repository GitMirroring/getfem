.. $Id$

.. include:: ../replaces.txt

.. highlight:: c++

.. _gmm-arpack:


Interface with ARPACK
============================


It is possible to call ARPACK-ng (https://github.com/opencollab/arpack-ng),
the eigensolver of large sparse problems by the implicitly restarted Lanczos
method, from |gmm|. The following functions, defined in the file
``gmm/gmm_arpack_interface.h``, are available::

  nconv = ARPACK_eigsh(K, M, nev, sigma, lambda, modes);
  nconv = ARPACK_eigsh(K, nev, sigma, lambda, modes);

They compute the ``nev`` eigenvalues nearest to the shift ``sigma`` of the
generalized eigenproblem :math:`K \varphi = \lambda M \varphi` (``K``
symmetric, ``M`` symmetric positive semi-definite; for the second one,
:math:`M` is the identity), and their modes. ``K`` and ``M`` are sparse
matrices of ``double`` (for instance ``gmm::csc_matrix<double>`` or
``gmm::col_matrix<gmm::wsvector<double> >``), ``lambda`` a
``std::vector<double>`` and ``modes`` a ``gmm::dense_matrix<double>``. On
return, ``lambda`` holds the ``nconv`` converged eigenvalues (``nev``,
normally) in ascending order and the columns of ``modes`` (``n x nconv``)
their modes, M-orthonormal. ``nev`` should be between 1 and ``n - 1``.

ARPACK is used in shift-invert mode: the largest eigenvalues of
:math:`(K - \sigma M)^{-1} M` are computed (``dsaupd``, ``dseupd``), the
matrix :math:`K - \sigma M` being factorized once, by |mumps| (symmetric
factorization) if |gmm| is linked with it, by SuperLU otherwise. ``M`` may be
singular (the multipliers of a mixed problem, without mass: their
eigenvalues are infinite and never found). The defaults are those of
``scipy.sparse.linalg.eigsh`` (``ncv = min(n, max(2 nev + 1, 20))``,
``10 n`` iterations at most, the machine precision). The starting vector is
a reproducible random one. Since ARPACK keeps a state between calls, the
calls are serialized.

To use these functions, you need ARPACK-ng with its ISO C binding (the
functions ``dsaupd_c`` and ``dseupd_c``, the 32 bits integer version, as in
the ``libarpack2-dev`` package of Debian and Ubuntu), and to compile your
code with the additional options::

  g++ ...  -DGMM_USES_ARPACK -DGMM_USES_SUPERLU (or -DGMM_USES_MUMPS) ... -larpack

(``GMM_USES_ARPACK`` is defined in ``gmm_arch_config.h`` when |gf| is built
with ARPACK-ng).
