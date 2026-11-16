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

The form of a module shows the equation it solves, its **properties**,
edited there, and the GWFL terms that the script adds for it. The
properties are given per domain selection (*Add a domain for k*: the last
one listed wins), by a value (a number or an expression, possibly of the
variables: a conductivity ``k(T)`` makes the model nonlinear) or by a
material. Editing the domain or the value of a property shows its domain in
the view, where the regions clicked are added to it (or removed). The
**sources** (nodes below the module) are
volume sources per domain selection, the **conditions** apply on boundary
selections. A boundary without condition is free (zero flux, zero
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
   "coupling": "sequence", "schemes": {"heat": {"type": "theta", "theta": 1}}}

A module whose equation has a time derivative has an **In time** box in its
form. Checked (``"transient": true``), the module
is solved in time: the heat conduction adds :math:`\rho c_p \partial
T/\partial t` (the term ``rho*cp*Dot_T*Test_T``), and its properties
``rho`` and ``cp`` are then required (they are not needed otherwise); its
model must be solved by a transient solver (checking the box sets one up:
the main solver made transient, or created). Unchecked, the module is
stationary: in a transient solver, it is solved at each time step without
time derivative (quasi-static: a heat conduction following imposed values
that depend on the time, an elasticity driven by a temperature in time).
The equation is written at :math:`t_{n+1}`, the scheme giving the time
derivative ``Dot_T`` from ``T`` and the previous values (see
:ref:`gui-time-schemes`; :math:`\theta = 1`, implicit Euler, by default).
``Dot_T`` and ``Previous_T`` may appear in the terms of the model.

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
sources, boundary conditions, nonlinear terms, constraints. The scheme does
not touch :math:`F`: at each time step, the equation is written **at**
:math:`t_{n+1}`, as a stationary problem,

.. math::

   M \dot U_{n+1} + F(U_{n+1}, t_{n+1}) = 0,

and the scheme only gives :math:`\dot U_{n+1}` as a function of the unknown
:math:`U_{n+1}` and of the values of the previous step. For the
:math:`\theta`-method, :math:`(U_{n+1} - U_n)/\Delta t = \theta \dot U_{n+1} +
(1 - \theta) \dot U_n`, that is

.. math::

   \dot U_{n+1} = \frac{U_{n+1} - U_n}{\theta \Delta t} - \frac{1 -
   \theta}{\theta} \dot U_n .

Since the equation also held at :math:`t_n`, this is the usual
:math:`\theta`-method :math:`M (U_{n+1} - U_n)/\Delta t + \theta F_{n+1} + (1 -
\theta) F_n = 0`, without weighting the terms of :math:`F` one by one (no
"dispatcher" of terms): the conditions are imposed at :math:`t_{n+1}`, a
nonlinear model is solved by Newton's method as in the stationary case, the
right-hand side stays as it is. :math:`\theta = 1` is the implicit Euler
scheme (first order, robust: the default); :math:`\theta = 1/2`
Crank-Nicolson (second order, but it may oscillate with steep initial data);
:math:`\theta < 1/2` is only conditionally stable. When :math:`\theta < 1`,
the initial derivative :math:`\dot U_0` is needed: it is computed first from
the equation at :math:`t_0` (an implicit Euler step of a tiny time step from
the initial values).

The same principle holds for second order equations (elastodynamics, to
come): :math:`M \ddot U_{n+1} + F(U_{n+1}, t_{n+1}) = 0`, the scheme
(Newmark, HHT, generalized-:math:`\alpha`) giving the acceleration from
:math:`U_{n+1}`, the previous displacement, velocity and acceleration.

**The implementation.** The scheme is written in the script with GWFL macros
and data of the model, updated by the time loop (they can be read in the
exported script):

- ``T__prev`` and ``T__rate`` (data on the finite element method of ``T``):
  :math:`U_n` and :math:`\dot U_n`; ``theta_T``.
- The macros ``Previous_T`` (``T__prev``), ``Previous_Dot_T`` (``T__rate``)
  and ``Dot_T``: ``(T - T__prev)/(theta_T*timestep) - (1 - theta_T)/theta_T*T__rate``.
  The time derivative term of the module, ``rho*cp*Dot_T*Test_T``, is then an
  ordinary term (linear: assembled with the others).
- The time ``t`` and the time step ``timestep`` of the expressions are those
  of the ``gf.Model`` (``set_time``, ``set_time_step``: data of the model, so
  that the terms using them are assembled again when they change); the
  imposed values that depend on the time (Dirichlet data) are interpolated
  again at each step.
- The loop: ``set_time(t, dt)`` (the time, the time step, the data depending
  on the time), the models solved (in turn, or by staggered iterations),
  then ``shift()``: ``T__rate`` gets the interpolation of ``Dot_T``,
  ``T__prev`` the value of ``T``; ``save(k, t)`` every ``save.every`` steps
  and at the end (the post-processing of each model, its VTU file, its
  scalar results appended to its series).

A new scheme is thus a new expression of ``Dot_u`` (or ``Dot2_u``) and its
update after each step; the modules do not change.

Outputs
-------

An output is an integral of a GWFL expression on a selection (a heat flow,
a reaction force, an area); its value is given in *Messages* and in the
table of the *Results* node. The variables, the exported fields of the
modules (the heat flux, the von Mises stress) and the fields are written to
a |vtu| file per model.
