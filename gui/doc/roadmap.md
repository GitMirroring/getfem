# Roadmap of the graphical interface

Directions kept for later, not planned yet. When one of them is started, it
leaves this list (its design goes to the documentation and to
`model_format_v3.md`).

## Linear solvers

- A PETSc linear solver in the GetFEM core first, then in the interface.
  The core: an `abstract_linear_solver` wrapping a PETSc `KSP`, optional at
  configure time (as MUMPS), selected by `lsolver` and set by a string of
  PETSc options. Sequential first, then parallel (the distribution of the
  rows consistent with the parallel assembly). It would bring algebraic
  multigrid (GAMG, hypre BoomerAMG) and block preconditioners for the systems
  with multipliers (`PCFIELDSPLIT`, by the variables of the model).
  - For the elasticity, GAMG needs the rigid body modes (near null space),
    from the coordinates of the dofs.
  - A Dirichlet condition by simplification makes the system non symmetric:
    a symmetric variant (the columns eliminated too, the imposed values on
    the right hand side) for CG with AMG.
  - The interface: `petsc` among the linear solvers, a few presets (CG + GAMG
    for a symmetric system, GMRES + hypre otherwise), a field for the PETSc
    options; SuperLU with a warning if this GetFEM has no PETSc.
- For a linear model, GetFEM bounds the iterations of the linear solver by
  the `max_iter` of the solve (the script passes at least 10000 for an
  iterative one): a separate limit in the core (`lin_model_pb`), as for
  Newton.

## Time schemes

- Explicit Runge-Kutta schemes (RK2, RK4) for the problems which allow
  them: first order in time (M du/dt = -G(u, t): a diffusion, a PDE of
  order 1, the mass system solved at each stage), and second order ones
  written as first order systems; their stability limits (the eigenvalues
  of M^-1 K in their stability region).
- A mass matrix depending on the time or on the unknowns for the explicit
  schemes (assembled once now).

## Models

- Several heat conduction modules in one model (on different domains, for
  instance).
- The PDE module: a vector unknown of any number of components (now: the
  dimension of the space; the values of the module are sized by it), its
  conservative convection b for a vector unknown, SUPG for a vector
  unknown and in general form.
- Polar and spherical coordinates.
- Thin structures (plates, shells, beams).
- Contact.
- More couplings between modules.
- Thermoelastic damping (the term T0 β tr(ε̇) in the heat equation), as an
  option of the coupled thermoelasticity.

## Interface

- The results of several global solvers merged (in one view, one series).
- A grid editor for the anisotropic properties (a matrix, its entries in
  cells).
