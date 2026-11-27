.. include:: ../replaces.txt

.. _lab-examples:

Examples
========

The examples are in ``lab/examples`` (and in *File ▸ Open an example*).
Each of them is solved by the tests of the interface
(``tests/test_doc.py``), which check the balances quoted below between the
outputs of the examples.

.. include:: reference/examples_table.inc

.. _lab-example-plate:

Heated plate (``plate.json``)
-----------------------------

A rectangular plate (a ``box``, structured mesh) heated by a Gaussian
source, at a fixed temperature on its left side and cooled by convection on
the others (a ``union`` selection). The outputs compare the injected power
with the heat flowing out through the sides: they balance.

.. _lab-example-plate_with_hole:

Plate with a heated hole (``plate_with_hole.json``)
---------------------------------------------------

A ``csg`` plate with a hole, heated by a flux on the hole, at a fixed
temperature on the left and cooled by convection on the right. Two meshes
of the same geometry (``getfem`` and ``gmsh``); the selections use the
``plane`` and ``in_ball`` criteria.

.. _lab-example-plate_hole_tension:

Plate with a hole in tension (``plate_hole_tension.json``)
----------------------------------------------------------

A quarter of a plate with a hole, in linear elasticity: ``roller``
conditions on the two symmetry planes, a traction ``S`` on the right side.
It is written with parameters (the radius ``a`` of the hole can be changed
by dragging the circle). The applied force (``S`` times the half height,
:math:`5\cdot 10^6` N/m) is balanced by the reaction on the symmetry plane;
the von Mises stress shows the concentration at the hole.

.. _lab-example-plate_inclusion:

Plate with a conductive inclusion (``plate_inclusion.json``)
------------------------------------------------------------

A plate with a disk of another material: the disk is united with the plate,
and the two materials are given on ``regions`` selections (picked in the
view). The |gf| mesher follows the circle, so that the jump of conductivity
is represented exactly.

.. _lab-example-thermoelastic_plate:

Heated plate that expands (``thermoelastic_plate.json``)
--------------------------------------------------------

Two models solved in sequence: a thermal model computes the temperature,
which a mechanical model imports for its thermal expansion module. The
output is the elongation of the plate.

.. _lab-example-l_bracket:

L-shaped bracket (``l_bracket.json``)
-------------------------------------

A ``polygon`` with a hole, fixed at its top and loaded by a vertical
traction at its end, the two sides being picked ``edges`` selections. Its
inner corner is rounded by a fillet of radius ``Rf``: the stress there
depends on it (without the fillet, it would be infinite). The reaction on
the top balances the applied force (the traction ``F`` times the width
``W``).

.. _lab-example-block_with_hole_3d:

Block with a hole (``block_with_hole_3d.json``)
-----------------------------------------------

A 3D block minus a cylinder, at a fixed temperature on the face of the hole
(a ``faces`` selection, picked in the view) and cooled by convection on the
whole boundary. The output is the power injected through the hole.

.. _lab-example-heat_sink_3d:

Heat sink (``heat_sink_3d.json``)
---------------------------------

An ``extrusion`` of a finned profile, meshed in 12 layers of prisms, heated
under its base (``qin``) and cooled by convection in the air. The injected
power (``qin`` times the area of the base, 30 W) is balanced by the
convected power.

.. _lab-example-pulley_3d:

Grooved pulley (``pulley_3d.json``)
-----------------------------------

A ``revolution`` of a profile with a groove, at a fixed temperature in its
bore (the face swept by a side of the profile) and cooled by convection.

.. _lab-example-heated_dish:

Heated dish (``heated_dish.json``)
----------------------------------

A ``surface``: an annulus mapped to a paraboloid by its height, at fixed
temperatures on the hole and on the rim (``edges`` selections of the flat
domain). The temperature is solved on the surface (tangential gradients);
the ``area`` output gives the area of the paraboloid.

.. _lab-example-heated_wire:

Heated wire (``heated_wire.json``)
----------------------------------

A ``curve`` drawn as an open polygon with four fillets: a wire bent three
times, heated along its length (Joule effect, ``power`` per meter), its ends
(the built-in selections ``start`` and ``end``) held at 20 °C. The
conductivity is that of copper times the section of the wire. The ``length``
output gives the length of the wire, arcs included; ``heat_out``, the heat
that leaves it by its ends, balances the heat generated.

.. _lab-example-plate_heating:

Plate heating (``plate_heating.json``)
--------------------------------------

A transient heat problem: an aluminium plate at 20 °C whose left edge is
brought to 100 °C (``20 + 80*(1 - exp(-t/tau))``, a function of the time),
the other edges cooled by the air (convection). The ``transient`` solver
goes from 0 to 60 s by steps of 1 s (BDF2, second order), saving every 5 steps:
``mean_T``, the mean temperature, rises to nearly 100 °C; ``heat_in``, the
heat entering by the edge, peaks then decreases. Its plots
(:ref:`lab-plots`): ``profiles``, the temperature across the plate at 10,
20, 30 and 60 s; ``history``, the temperature at the center, at the far edge
and its mean, in time.

.. _lab-example-beam_vibration:

Beam vibration (``beam_vibration.json``)
----------------------------------------

Elastodynamics: a steel cantilever (1 m x 5 cm, plane strain), fixed at its
left end, struck at its tip by a force pulse of 5 ms (the traction
``-q_tip*sqr(sin(pi*t/tp))*Heaviside(tp - t)``, a function of the time). The
``transient`` solver follows it for 0.1 s by steps of 0.5 ms with the
generalized-:math:`\alpha` scheme (:math:`\rho_\infty = 0.9`), from rest
(the initial acceleration from the equation: zero here). The tip then
oscillates at the first bending frequency of the beam: a period of about
22.8 ms (:math:`2\pi/\omega_1`, :math:`\omega_1 = 3.516 \sqrt{E' I/(\rho A
L^4)}`, :math:`E' = E/(1 - \nu^2)` in plane strain); after the pulse, the
energy (``kinetic_energy`` plus ``strain_energy``: ``energy``) stays
constant within 0.1 % (the scheme only damps the high frequencies). The
*In time* curves of the results show them; the VTU files, the deformed
beam.

.. _lab-example-thermal_shock_strip:

Thermal shock of a strip (``thermal_shock_strip.json``)
-------------------------------------------------------

Thermally induced vibrations (Boley's problem): an aluminium strip (5 cm x 1
mm, plane strain), clamped at its left end, receives from t = 0 a heat flux
of 1 MW/m² on its top face. Two models solved in turn at each time step:
the heat in time (BDF2), and the elasticity with thermal expansion, which
imports the temperature, in time too (generalized-:math:`\alpha`): each
variable its scheme. The top face, warmer, expands: the strip bends down.
As the heating (the diffusion time through the thickness,
:math:`h^2/\kappa \approx` 10 ms) is not much slower than the first
bending mode of the strip (a period of 2.87 ms), the strip vibrates about
the quasi-static deflection: its tip goes down to about 1.37 times the
quasi-static deflection. With the scheme of ``u`` set to *stationary* in
the solver, the same problem is solved quasi-statically (no vibration).
``T_mean`` rises as :math:`q t/(\rho c_p h)` (12.4 °C in 30 ms: all the
heat received stays in the strip).

.. _lab-example-heated_pipe:

Heated pipe (``heated_pipe.json``)
----------------------------------

A parametric ``surface``: the flat domain [0, 2 pi] x [0, L] in (u, v),
mapped to the tube (R cos(u), R sin(u), v). Its sides u = 0 and u = 2 pi
meet: they are merged (a seam), so the tube is closed. The flat domain is
the union of two rectangles, so that the heated half of the tube is the
region ``heated``; the ends of the tube (``plane`` criteria in (u, v)) are
held at 20 °C. ``heat_out_inlet`` and ``heat_out_outlet`` share the heat
received; ``area`` and ``heated_area`` are those of the tube and of its
heated half.

.. _lab-example-heated_tray:

Heated tray (``heated_tray.json``)
----------------------------------

A ``boundary`` geometry: the faces of a box but its top (the boundary
selection ``walls`` of the box, a union of its faces), a sheet-metal tray.
The box is meshed, then its faces are taken as a surface mesh; the faces of
the box (``zmin``...) are built-in selections of the tray, and its rim (the
edge of the open surface) is its selection ``boundary``. Heated at its
bottom, the rim held at 20 °C: ``heat_out``, the heat that leaves by the
rim, balances the heat received; ``area`` and ``bottom`` are the areas of
the tray and of its bottom.

.. _lab-example-anisotropic_plate:

Fibre composite plate (``anisotropic_plate.json``)
--------------------------------------------------

Anisotropic conduction: the conductivity is a matrix parameter ``K``, built
from the conductivities along and across the fibres and their direction
``d`` (a vector parameter), and the heat conduction module has the
``anisotropic`` option. A heat source in the middle, the boundary at a
fixed temperature: the isotherms are stretched along the fibres. The
direction of the fibres is also a vector field, written to the results.
The outflow balances the injected power, and the heat flows mostly along
the fibres (``flux_along_fibres`` and ``flux_across_fibres``).

.. _lab-example-flange_2d:

Flange (``flange_2d.json``)
---------------------------

A flange with a keyway, held by ``n`` bolts: the bolt holes are a
circular ``array`` of a disk (``"count": "n"``, ``"angle": "360/n"``), the
keyway a rectangle turned by ``ka`` degrees about the center (a
``transform``). The shaft is hot (the ``bore`` selection: the circle of
the bore and the sides of the keyway), the bolts cold (``bolts``: the
circle of the disk of the array, that is all the holes), the rim cooled by
convection. ``bolt_holes_length`` is the perimeter of the ``n`` holes, and
the power entering through the bore (``power_in``) leaves through the bolts
and the rim (``power_bolts`` and ``power_rim``). Change ``n`` or ``ka``:
the holes and the keyway follow, and so do the selections.

.. _lab-example-pin_fin_heat_sink_3d:

Pin fin heat sink (``pin_fin_heat_sink_3d.json``)
-------------------------------------------------

A plate with ``n`` × ``n`` pins: a linear ``array`` (along y) of a linear
``array`` (along x) of a cylinder. The ``pins`` selection is two faces of
the cylinder (its ``side`` and ``end``), so it holds all the pins. Heated
under the plate (``qin``), cooled by convection on the plate and the pins
(``air``): the injected power (``qin`` times the area of the base) is
balanced by the convected power, and ``pins_area`` is the area of the
pins in the air (``n``\ :sup:`2` times their side above the plate and
their end).

.. _lab-example-convection_diffusion:

Convection-diffusion (``convection_diffusion.json``)
----------------------------------------------------

The ``pde`` module, in coefficient form: :math:`-\mathrm{div}(\kappa
\nabla u) + v\cdot\nabla u = 1` on the unit square, ``u = 0`` on its
boundary, with a small diffusion ``kappa`` = 0.002 and a unit convection at
the angle ``theta``. The convection dominates (on a 32 x 32 mesh,
:math:`|v| h/\kappa` is about 15): u grows along the streamlines
and drops to 0 in thin boundary layers at the outflow. With its option
``stabilization`` set to ``none`` (Galerkin), the solution overshoots
(its maximum is above 2); with ``supg`` (the example), it stays
between 0 and 1 (``max_u`` 0.93).

.. _lab-example-membrane_wave:

Membrane waves (``membrane_wave.json``)
---------------------------------------

The ``pde`` module of order 2 in time: the wave equation
:math:`\partial^2 w/\partial t^2 - \mathrm{div}(c_0^2 \nabla w) = 0` on a
square membrane fixed on its edges, from a Gaussian bump at rest (the
initial value of ``w``). The transient solver uses Newmark's average
acceleration scheme, which conserves the energy of a linear problem: the
output ``energy`` stays at :math:`\pi/2` (that of the initial bump, the
integral of :math:`c_0^2 |\nabla w|^2/2`) to six digits, while the waves
reflect on the edges (``w`` changes sign). ``displaced_volume`` starts at
:math:`\pi r_0^2`.

.. _lab-example-coax_cable:

Coaxial cable (``coax_cable.json``)
-----------------------------------

The ``electrostatics`` module on the section of a coaxial cable: an inner
conductor of radius ``a`` = 1 mm at the potential ``V0`` = 1 V, the outer
conductor of radius ``b`` = 3.5 mm grounded, polyethylene between them
(:math:`\varepsilon_r` = 2.3). Its capacitance per unit length, the output
``capacitance`` (:math:`2W/V_0^2`), and the charges of the conductors
(``charges``, :math:`\pm C V_0`) agree within :math:`10^{-4}` with
:math:`2\pi\varepsilon_0\varepsilon_r/\ln(b/a)` = 102.1 pF/m.

.. _lab-example-busbar_joule:

Copper busbar heated by its current (``busbar_joule.json``)
-----------------------------------------------------------

A copper bar of length ``L`` = 10 cm and width ``H`` = 2 cm, with a bolt
hole of radius 5 mm, under a voltage ``V0`` = 30 mV between its ends. The
``electric_currents`` module computes the current and its Joule heat, a
source of the ``heat_conduction`` module of the same model (option
``joule_heating``); the ends are held at ``T0`` = 20 °C, the long sides
lose heat to the air by convection (``h_conv`` = 10 W/(m²·K)). The
conductivity of copper decreases with the temperature,
:math:`\sigma_0/(1 + \alpha_R(T - T_0))` (:math:`\sigma_0` = 5.8e7 S/m,
:math:`\alpha_R` = 0.00393 /K), so the two problems are coupled both ways
and solved together by Newton's method (6 iterations). The maximum
temperature is about 35.7 °C (36.2 °C with :math:`\alpha_R` = 0, a larger
current); the resistance per unit depth, ``V0`` over the current
(``currents``), is some 15 % above that of the bar without hole, cold,
:math:`L/(\sigma_0 H)`; nearly all the heat leaves through the ends (the
output ``heat_to_air``, about 20 W/m of 9.1 kW/m). Its plots
(:ref:`lab-plots`): the temperature along the axis of the bar, interrupted
by the hole, and along its edge; the current density across the bar through
the hole, concentrated at its rim.

.. _lab-example-tuning_fork:

Tuning fork (``tuning_fork.json``)
----------------------------------

The eigenmodes of a steel tuning fork (2D, plane stress, P2 elements on a
mesh of the GetFEM mesher), held by the end of its stem: prongs of length
``Lp`` = 94.4 mm and width ``w`` = 5 mm, 10 mm apart. The ``modal`` solver
computes its six lowest modes. The first one (261 Hz) sways the whole fork
sideways (59 % of the mass along :math:`x`); the second one, at 440 Hz,
is the **fork mode**: the prongs vibrate in opposition, the stem hardly
moves (its effective mass is zero): little energy leaves through the hand,
the fork rings long. The
prongs were sized for it: a prong alone, a clamped beam, would give
:math:`1.875^2/(2\pi L_p^2)\, w \sqrt{E/12\rho}` = 470 Hz, the bridge is
not rigid. The frequencies change by 0.2 % with elements twice smaller.
