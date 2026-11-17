.. include:: ../replaces.txt

.. highlight:: json

.. _gui-models:

Models
======

A model is one ``gf.Model``: variables, modules, terms and conditions
solved together. A new model is created from a module (the **+** button on
the *Models* node), with its variables. The forms of its nodes are listed
in :ref:`gui-ref-nodes`.

Variables
---------

A variable has a name, a mesh, a finite element method (continuous or
discontinuous Lagrange), a degree, and is scalar or vector (as many
components as the dimension)::

  {"name": "u", "mesh": "mesh", "fem": "lagrange", "degree": 2, "components": "dim"}

Modules
-------

A module is a predefined physics acting on variables
(:ref:`gui-ref-modules`):

``heat_conduction``
   stationary heat conduction, of conductivity ``k``; conditions
   ``temperature``, ``heat_flux``, ``convection``; macro ``q`` (the heat
   flux). With its option ``"conductivity": "anisotropic"``, ``k`` is a
   matrix (see below).
``linear_elasticity``
   of Young's modulus ``E`` and Poisson's ratio ``nu``, in plane strain or
   plane stress in 2D; conditions ``fixed``, ``displacement``, ``roller``
   (``u.n = 0``, imposed by Nitsche's method, so that it may share nodes with
   a fixed condition), ``traction``, ``pressure``; macros ``eps``, ``sigma``,
   ``von_mises``...
``thermal_expansion``
   coupling of a temperature (a variable of the model, or imported from
   another model) with a linear elasticity module: it adds
   :math:`-\beta\alpha(T - T_{ref})I` to the stress.

The form of a module shows the equation it solves, its **properties** and
its **source** :math:`f`, edited there, and the GWFL terms that the script
adds for it. The properties are given per domain selection (*Add a domain
for k*: the last one listed wins), by a value (a number or an expression,
possibly of the variables: a conductivity ``k(T)`` makes the model
nonlinear) or by a material. The source is 0 by default, a value or an
expression of the coordinates, the time, the variables; *Add a domain for f*
gives sources on domain selections, which add up. Editing the domain or the
value of a property or of a source shows its domain in the view, where the
regions clicked are added to it (or removed). The **conditions** (nodes
below the module) apply on boundary selections. A boundary without condition is free (zero flux, zero
traction). The interface warns when nothing fixes the solution (no imposed
temperature, no displacement constraint).

Anisotropic properties
----------------------

.. image:: images/anisotropic.png
   :width: 100%
   :alt: The temperature in a fibre composite plate: isotherms along the fibres

Some properties are tensors for an option of their module: the
conductivity of a heat conduction module with ``"conductivity":
"anisotropic"`` is a matrix of the dimension of the space. Each of its
values is then ``[a, b; c, d]`` (by rows, as in GWFL), a matrix parameter, a
GWFL matrix expression, a list of rows in the file, or a number, standing
for that number times the identity; a material may give a matrix too. In
``examples/anisotropic_plate.json``, the conductivity tensor of a fibre
composite is built from the direction ``d`` of the fibres:

.. code-block:: text

  {"name": "K", "value": "[k1*sqr(d(1)) + k2*sqr(d(2)), (k1 - k2)*d(1)*d(2);
                           (k1 - k2)*d(1)*d(2), k1*sqr(d(2)) + k2*sqr(d(1))]"}

The heat conduction module can be *unfolded* (tree action) into the
equivalent terms and generic conditions, a starting point to write a
variant.

Materials
---------

*Add from a library* (on the *Materials* node, a menu by category) copies a
material of the built-in library (:ref:`gui-ref-materials`) or of your own
library (``~/.config/getfem_gui/materials.json``, or the file given by
``GETFEM_GUI_MATERIALS``); a material of the model is added to your library
by *Save in my library*. Only the copies are in the model file, with their
origin, and they may be changed. The *Materials* node shows a table of the
materials and of the properties asked by the modules of the file.

Terms
-----

A term is a GWFL weak form integrated on a selection. The terms of a model
sum to zero: a source is written ``"-f*Test_u"``::

  {"name": "extra", "expr": "-q0*Test_T", "selection": "inclusion"}

A term that |gf| does not accept as linear is added as a nonlinear one, and
the model is then solved by Newton's method.

Generic conditions
------------------

Conditions on any variable (:ref:`gui-ref-conditions`): ``dirichlet``
(by simplification, multipliers or penalization), ``normal_dirichlet``,
``neumann``, ``robin``.

Solvers
-------

The solver of a model is ``auto`` (linear or Newton, depending on the
terms), ``linear`` or ``newton`` (maximum number of iterations, residual
tolerance). For a linear model, the solve script checks the relative
residual :math:`|KU - F|/|F|` and stops with an explicit error when it
exceeds :math:`10^{-6}` (a singular system); when the system is singular but
compatible, it is solved, with a warning.

Several models are solved by a global **solver**: ``sequence`` (in order),
or ``staggered`` (iterated until the relative change of the variables is
below a tolerance, with relaxation). A model takes a variable of another one
by an **import**, on the same mesh for now::

  {"name": "T", "model": "thermal", "variable": "T"}

Without a global solver, the models are solved in an order compatible with
their imports. See ``examples/thermoelastic_plate.json``.

Transient problems
------------------

A ``transient`` global solver solves its models in time, from a start time
to an end time by a fixed time step::

  {"name": "heating", "type": "transient", "steps": ["heat"],
   "time": {"start": 0, "end": 60, "step": 1}, "save": {"every": 5},
   "coupling": "sequence", "schemes": {"heat/T": {"type": "bdf2"}}}

Whether a variable is solved in time, and how, is a choice of the solver,
not of the physics: the same model may be solved stationary by one solver
and in time by another. The solver lists the variables of its models which
may have a time derivative (those of the modules with one: the temperature
of a heat conduction) and gives each one its **time scheme**
(``schemes``, keyed by ``model/variable``; its form shows a *Scheme of T*
choice for each):

- ``{"type": "stationary"}``: no time derivative, the equation solved at
  each time step (quasi-static: a heat conduction following imposed values
  that depend on the time, a potential driven by a temperature in time);
- ``{"type": "theta", "theta": 1}``: the :math:`\theta`-method,
  :math:`\theta = 1` implicit Euler (the default when a model is added to a
  transient solver), :math:`\theta = 1/2` Crank-Nicolson;
- ``{"type": "bdf2"}``: BDF2, second order, its first step by
  Crank-Nicolson.

In time, the heat conduction adds :math:`\rho c_p \partial T/\partial t`
(the term ``rho*cp*Dot_T*Test_T``), and its properties ``rho`` and ``cp``
are then required (they are not needed otherwise; they are given a default
value when the variable goes into time). The form of the module shows where
it stands: stationary, or in time with its scheme. ``Dot_T`` and
``Previous_T`` may appear in the terms of the model. Since every equation is
written at the new time (see :ref:`gui-time-schemes`), variables with
different schemes and stationary variables are solved together without
difficulty, in one model or in coupled ones.

- The variables start from their **initial** value (a number or an
  expression of the coordinates and the parameters; 0 by default).
- The time ``t`` may appear in the expressions (sources, conditions,
  terms): the imposed values depending on it are interpolated again at
  each time step.
- At each time step, the models of the solver are solved in turn
  (``coupling``: ``sequence``) or by staggered iterations (``staggered``,
  with ``max_iter``, ``tolerance``, ``relaxation``), each one taking the
  current variables of the others by its imports; a model without time
  derivative (an elasticity) is then quasi-static.
- The time step is the same for all the variables of a solver.
- The results are saved every ``save.every`` steps and at the end: a |vtu|
  file each (``heat_00005.vtu``...), a ParaView collection ``heat.pvd``,
  and the scalar results (outputs, extrema) as series of the saved times
  (``series`` in the results); the final values are those shown as for a
  stationary model.

See ``examples/plate_heating.json``.

.. _gui-time-schemes:

Time schemes: principle and implementation
------------------------------------------

**The principle.** Once discretized in space, a model in time is a system

.. math::

   M \dot U + F(U, t) = 0,

where :math:`M` comes from the time derivative terms (the "mass": :math:`\rho
c_p` for the heat) and :math:`F` gathers all the other terms: conduction,
sources, boundary conditions, nonlinear terms, constraints, the equations
of the stationary variables. The scheme does not touch :math:`F`: at each
time step, the equation is written **at** :math:`t_{n+1}`, as a stationary
problem,

.. math::

   M \dot U_{n+1} + F(U_{n+1}, t_{n+1}) = 0,

and the scheme only gives :math:`\dot U_{n+1}` as a function of the unknown
:math:`U_{n+1}` and of the values of the previous steps. No term of
:math:`F` is weighted one by one (no "dispatcher" of terms): the conditions
are imposed at :math:`t_{n+1}`, a nonlinear model is solved by Newton's
method as in the stationary case, the right-hand side stays as it is. The
schemes of the different variables are thus independent: each variable in
time has its own expression of its derivative, the stationary ones none.

The schemes of first order equations all have the form

.. math::

   \dot U_{n+1} = \frac{c_0 U_{n+1} + c_1 U_n + c_2 U_{n-1}}{\Delta t}
   + c_3 \dot U_n .

- The :math:`\theta`-method, :math:`(U_{n+1} - U_n)/\Delta t = \theta \dot
  U_{n+1} + (1 - \theta) \dot U_n`: :math:`(c_0, c_1, c_2, c_3) = (1/\theta,
  -1/\theta, 0, -(1 - \theta)/\theta)`. Since the equation also held at
  :math:`t_n`, this is the usual :math:`\theta`-method :math:`M (U_{n+1} -
  U_n)/\Delta t + \theta F_{n+1} + (1 - \theta) F_n = 0`. :math:`\theta = 1`
  is the implicit Euler scheme (first order, robust: the default);
  :math:`\theta = 1/2` Crank-Nicolson (second order, but it may oscillate
  with steep initial data); :math:`\theta < 1/2` is only conditionally
  stable.
- BDF2, :math:`\dot U_{n+1} = (3 U_{n+1} - 4 U_n + U_{n-1})/(2 \Delta t)`:
  :math:`(3/2, -2, 1/2, 0)`. Second order and L-stable (no oscillation, it
  damps the fast modes), with an error constant larger than that of
  Crank-Nicolson. It needs two previous values: its first step is a
  Crank-Nicolson one, :math:`(2, -2, 0, -1)`, which keeps the second order
  (one step of local error :math:`O(\Delta t^3)`).

When the first step needs the initial derivative :math:`\dot U_0`
(:math:`c_3 \neq 0`: :math:`\theta < 1`, BDF2), it is computed first from the
equation at :math:`t_0` (an implicit Euler step of a tiny time step from the
initial values).

The same principle holds for second order equations (elastodynamics, to
come): :math:`M \ddot U_{n+1} + F(U_{n+1}, t_{n+1}) = 0`, the scheme
(Newmark, HHT, generalized-:math:`\alpha`) giving the acceleration from
:math:`U_{n+1}`, the previous displacement, velocity and acceleration.

**The implementation.** The scheme is written in the script with GWFL macros
and data of the model, updated by the time loop (they can be read in the
exported script):

- ``T__prev``, ``T__rate`` and, for BDF2, ``T__prev2`` (data on the finite
  element method of ``T``): :math:`U_n`, :math:`\dot U_n`, :math:`U_{n-1}`;
  ``T__scheme``: the coefficients :math:`(c_0, c_1, c_2, c_3)` of the
  current step.
- The macros ``Previous_T`` (``T__prev``), ``Previous_Dot_T`` (``T__rate``)
  and ``Dot_T``: ``(T__scheme(1)*T + T__scheme(2)*T__prev +
  T__scheme(3)*T__prev2)/timestep + T__scheme(4)*T__rate``. The time
  derivative term of the module, ``rho*cp*Dot_T*Test_T``, is then an
  ordinary term (linear: assembled with the others, again when the
  coefficients change).
- The time ``t`` and the time step ``timestep`` of the expressions are those
  of the ``gf.Model`` (``set_time``, ``set_time_step``: data of the model, so
  that the terms using them are assembled again when they change); the
  imposed values that depend on the time (Dirichlet data) are interpolated
  again at each step.
- The loop: ``set_time(t, dt)`` (the time, the time step, the data depending
  on the time), the models solved (in turn, or by staggered iterations),
  then ``shift(k)``: ``T__rate`` gets the interpolation of ``Dot_T``,
  ``T__prev2`` the value of ``T__prev``, ``T__prev`` the value of ``T``, and
  after the first step the coefficients of the scheme from then on (BDF2);
  ``save(k, t)`` every ``save.every`` steps and at the end (the
  post-processing of each model, its VTU file, its scalar results appended
  to its series).

A new scheme is thus new coefficients (or a new expression of ``Dot_u``,
``Dot2_u``), its history and its update after each step
(``scheme_coefficients`` in ``model.py``, ``time_declarations`` and the
time loop in ``codegen.py``); the modules do not change.

Outputs
-------

An output is an integral of a GWFL expression on a selection (a heat flow,
a reaction force, an area); its value is given in *Messages* and in the
table of the *Results* node. The variables, the exported fields of the
modules (the heat flux, the von Mises stress) and the fields are written to
a |vtu| file per model.
