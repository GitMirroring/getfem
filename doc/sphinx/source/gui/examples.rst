.. include:: ../replaces.txt

.. _gui-examples:

Examples
========

The examples are in ``gui/examples`` (and in *File ▸ Open an example*).
Each of them is solved by the tests of the interface
(``tests/test_doc.py``), which check the balances quoted below between the
outputs of the examples.

.. include:: reference/examples_table.inc

.. _gui-example-plate:

Heated plate (``plate.json``)
-----------------------------

A rectangular plate (a ``box``, structured mesh) heated by a Gaussian
source, at a fixed temperature on its left side and cooled by convection on
the others (a ``union`` selection). The outputs compare the injected power
with the heat flowing out through the sides: they balance.

.. _gui-example-plate_with_hole:

Plate with a heated hole (``plate_with_hole.json``)
---------------------------------------------------

A ``csg`` plate with a hole, heated by a flux on the hole, at a fixed
temperature on the left and cooled by convection on the right. Two meshes
of the same geometry (``getfem`` and ``gmsh``); the selections use the
``plane`` and ``in_ball`` criteria.

.. _gui-example-plate_hole_tension:

Plate with a hole in tension (``plate_hole_tension.json``)
----------------------------------------------------------

A quarter of a plate with a hole, in linear elasticity: ``roller``
conditions on the two symmetry planes, a traction ``S`` on the right side.
It is written with parameters (the radius ``a`` of the hole can be changed
by dragging the circle). The applied force (``S`` times the half height,
:math:`5\cdot 10^6` N/m) is balanced by the reaction on the symmetry plane;
the von Mises stress shows the concentration at the hole.

.. _gui-example-plate_inclusion:

Plate with a conductive inclusion (``plate_inclusion.json``)
------------------------------------------------------------

A plate with a disk of another material: the disk is united with the plate,
and the two materials are given on ``regions`` selections (picked in the
view). The |gf| mesher follows the circle, so that the jump of conductivity
is represented exactly.

.. _gui-example-thermoelastic_plate:

Heated plate that expands (``thermoelastic_plate.json``)
--------------------------------------------------------

Two models solved in sequence: a thermal model computes the temperature,
which a mechanical model imports for its thermal expansion module. The
output is the elongation of the plate.

.. _gui-example-l_bracket:

L-shaped bracket (``l_bracket.json``)
-------------------------------------

A ``polygon`` with a hole, fixed at its top and loaded by a vertical
traction at its end, the two sides being picked ``edges`` selections. Its
inner corner is rounded by a fillet of radius ``Rf``: the stress there
depends on it (without the fillet, it would be infinite). The reaction on
the top balances the applied force (the traction ``F`` times the width
``W``).

.. _gui-example-block_with_hole_3d:

Block with a hole (``block_with_hole_3d.json``)
-----------------------------------------------

A 3D block minus a cylinder, at a fixed temperature on the face of the hole
(a ``faces`` selection, picked in the view) and cooled by convection on the
whole boundary. The output is the power injected through the hole.

.. _gui-example-heat_sink_3d:

Heat sink (``heat_sink_3d.json``)
---------------------------------

An ``extrusion`` of a finned profile, meshed in 12 layers of prisms, heated
under its base (``qin``) and cooled by convection in the air. The injected
power (``qin`` times the area of the base, 30 W) is balanced by the
convected power.

.. _gui-example-pulley_3d:

Grooved pulley (``pulley_3d.json``)
-----------------------------------

A ``revolution`` of a profile with a groove, at a fixed temperature in its
bore (the face swept by a side of the profile) and cooled by convection.

.. _gui-example-heated_dish:

Heated dish (``heated_dish.json``)
----------------------------------

A ``surface``: an annulus mapped to a paraboloid by its height, at fixed
temperatures on the hole and on the rim (``edges`` selections of the flat
domain). The temperature is solved on the surface (tangential gradients);
the ``area`` output gives the area of the paraboloid.

.. _gui-example-heated_wire:

Heated wire (``heated_wire.json``)
----------------------------------

A ``curve`` drawn as an open polygon with four fillets: a wire bent three
times, heated along its length (Joule effect, ``power`` per meter), its ends
(the built-in selections ``start`` and ``end``) held at 20 °C. The
conductivity is that of copper times the section of the wire. The ``length``
output gives the length of the wire, arcs included; ``heat_out``, the heat
that leaves it by its ends, balances the heat generated.

.. _gui-example-plate_heating:

Plate heating (``plate_heating.json``)
--------------------------------------

A transient heat problem: an aluminium plate at 20 °C whose left edge is
brought to 100 °C (``20 + 80*(1 - exp(-t/tau))``, a function of the time),
the other edges cooled by the air (convection). The ``transient`` solver
goes from 0 to 60 s by steps of 1 s (BDF2, second order), saving every 5 steps:
``mean_T``, the mean temperature, rises to nearly 100 °C; ``heat_in``, the
heat entering by the edge, peaks then decreases.

.. _gui-example-beam_vibration:

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

.. _gui-example-heated_pipe:

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

.. _gui-example-heated_tray:

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

.. _gui-example-anisotropic_plate:

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

.. _gui-example-flange_2d:

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

.. _gui-example-pin_fin_heat_sink_3d:

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
