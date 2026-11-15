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

The **properties** of a module are given per domain selection, by a value
(a number or an expression, possibly of the variables: a conductivity
``k(T)`` makes the model nonlinear) or by a material. The **sources** are
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

A module whose equation has a time derivative adds it in time: the heat
conduction adds :math:`\rho c_p \partial T/\partial t` (the term
``rho*cp*Dot_T*Test_T``), and its properties ``rho`` and ``cp`` are then
required (they are not needed by a stationary model). The equation is
written at :math:`t_{n+1}`, as for a stationary model, the scheme giving
the time derivative ``Dot_T`` from ``T`` and the previous values
``Previous_T``, ``Previous_Dot_T``: the :math:`\theta`-method,
:math:`\dot T_{n+1} = (T_{n+1} - T_n)/(\theta \Delta t) - (1 -
\theta)/\theta\, \dot T_n` (:math:`\theta = 1`: implicit Euler, robust,
first order, the default; :math:`\theta = 1/2`: Crank-Nicolson, second
order, may oscillate; then the initial time derivative is computed first).
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

Outputs
-------

An output is an integral of a GWFL expression on a selection (a heat flow,
a reaction force, an area); its value is given in *Messages* and in the
table of the *Results* node. The variables, the exported fields of the
modules (the heat flux, the von Mises stress) and the fields are written to
a |vtu| file per model.
