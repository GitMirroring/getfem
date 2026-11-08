.. include:: ../replaces.txt

.. highlight:: json

.. _gui-geometries:

Geometries
==========

*Add a geometry* (the **+** button or a right click on *Geometries*) opens
a menu of the kinds of geometry: a segment, a rectangle or a box; a 2D
domain or a 3D solid by constructive geometry, an extrusion or a revolution
of a 2D profile; a curve in 2D or 3D, a surface in 3D, an extruded or
revolved curve; a CAD or a mesh file. A new box, curve or surface is a simple one of that
kind, to be edited; a new constructive geometry (2D domain, 3D solid) starts
from nothing, as does the profile of a new extrusion or revolution (see
:ref:`gui-geometries-building`). *Add a mesh of this geometry* gives it a
mesh. Its source can still be changed in its form.

A geometry is given by its *source* (see :ref:`gui-ref-sources` for all
the fields):

``box``
   a segment, a rectangle or a box (1D, 2D, 3D), with the built-in
   selections of its sides ``xmin``, ``xmax``, ``ymin``... A box gets a
   structured mesh.
``csg``
   a constructive geometry: primitives (rectangles, disks, ellipses,
   polygons in 2D; boxes, balls, cylinders, extrusions and revolutions of 2D
   profiles in 3D) combined by unions, intersections and differences.
``curve``
   a chain of segments and arcs of circle in 2D or 3D (beams, cables).
``surface``
   a surface in 3D: a flat 2D domain placed in space, plus a height along
   its normal (shells, membranes).
``swept_surface``
   a 2D curve extruded along z or turned about the z axis.
``cad``
   a STEP, IGES or BREP file, read by |gmsh| (OpenCASCADE).
``mesh_file``
   a |gmsh| mesh (``.msh``), which is the geometry itself; its named
   physical groups become selections.

Constructive geometries in 2D
-----------------------------

A ``csg`` geometry is a tree of shapes, shown under the geometry in the
model tree. A boolean combines the first of its shapes with the following
ones (the difference subtracts them all from the first one)::

  {"type": "csg", "dim": 2, "shape":
    {"type": "difference", "of": [
      {"type": "rectangle", "min": [0, 0], "max": ["L", "H"]},
      {"type": "ball", "center": [0, 0], "radius": "a"}]}}

The primitives (:ref:`gui-ref-shapes`) are:

* ``rectangle``: lower and upper corners;
* ``ball`` (a disk in 2D): center and radius;
* ``ellipse``: center, semi-axes, angle of the first semi-axis in degrees;
* ``polygon``: its vertices, in either order, not necessarily convex (but
  it must not cross itself). Its vertices are edited in a table: *Add* adds
  one after the last, *Insert* adds a vertex in the middle of a side,
  *Remove* removes one. *Add a polygon* gives an empty polygon: in a 2D
  geometry, the polygon tool of the view starts, and its vertices are
  clicked (as for drawing, below); in a profile, they are typed in the
  table.

Every number may be a parameter or an expression of parameters. The type
of a primitive can be changed in its form: a rectangle or a disk changed
into a polygon keeps its corners (an octagon for a disk), a disk changed
into an ellipse keeps its center and radius.

The 2D geometries are drawn exactly, without meshing. Their boundary is
made of the *sides* of the primitives: ``xmin``, ``xmax``, ``ymin``,
``ymax`` for a rectangle, ``circle`` for a disk, ``ellipse``, and ``side1``,
``side2``... for a polygon (from its first vertex). The selections refer to
these sides (see :ref:`gui-selections`), so that they follow the changes of
the geometry.

Drawing
^^^^^^^

With a 2D geometry selected, the tools of the view draw a shape by
dragging, snapped to a grid whose step follows the zoom:

* *Rectangle*: from corner to corner;
* *Disk*: from the center to a point of the circle;
* *Ellipse*: the ellipse inscribed in the rectangle of two corners;
* *Polygon*: a vertex per click; clicking the first vertex again,
  double-clicking or *Enter* closes it, *Backspace* removes the last vertex.

The shape is added to, subtracted from or intersected with the geometry
(*Escape* stops drawing). It joins the root boolean of the tree when it is
the same operation; otherwise, the root is wrapped in a new boolean. A box
becomes a ``csg`` geometry on the first drawing: its sides become
selections of the same name, so that their uses stay valid.

Handles
^^^^^^^

.. image:: images/polygon_handles.png
   :width: 100%
   :alt: The handles of a polygon

With a primitive selected in the tree, its handles move or resize it,
snapped to the grid: corners and center of a rectangle; center and a point
of the circle of a disk; center and the ends of the two semi-axes of an
ellipse (the first one turns it); vertices, middles of the sides and center
of a polygon (dragging the middle of a side adds a vertex). A value that is
a number is changed; a value that is the name of a parameter changes the
parameter (dragging the circle of the hole of
``examples/plate_hole_tension.json`` changes ``a``); a value given by an
expression (``"L/2"``) is locked, and the status bar says so.

Any shape of a boolean can be removed (right click); a boolean left with a
single shape is replaced by it.

.. _gui-geometries-building:

Building the tree
^^^^^^^^^^^^^^^^^

A new constructive geometry starts from nothing: its root is an empty union
(*Union (empty: add shapes)* in the tree, an error until a shape is added).
The actions of a shape node (right click, the **+** button, or the buttons
of its form) build the tree:

* on a boolean: *Add a rectangle*, *a ball*... (the primitives of its
  dimension), and *Add a union / an intersection / a difference*, an empty
  boolean to fill in turn;
* on any shape: *Put in a union / an intersection / a difference*, which
  makes it the first part of a new boolean (the other parts are then
  added to it);
* on the root (or a profile): *Remove (start again from nothing)*.

The type of a boolean is changed in its form (a union of a rectangle and a
disk becomes their difference). A boolean of a single shape is that shape;
an empty one is an error. In 2D, the first shape drawn on an empty geometry
becomes its first part. The selections that refer to sides or faces of
primitives follow these changes.

Geometries in 3D
----------------

.. image:: images/preview_3d.png
   :width: 100%
   :alt: The preview of an extruded heat sink

The 3D primitives are the ``rectangle`` (a box), the ``ball``, the
``cylinder`` (origin, axis, length, radius), and two shapes made of a 2D
profile, itself a 2D constructive tree edited under the shape:

* ``extrusion``: the profile, in the (x, y) plane, extruded along z between
  two values of z;
* ``revolution``: the profile, in the (r, z) half plane, turned about the z
  axis (its part with r < 0 is ignored).

3D geometries are previewed without meshing: the surface of each
primitive is clipped to its part on the boundary of the whole shape, so
that the faces stay flat and the edges sharp. Their *faces* are those of
the primitives: ``xmin``... ``zmax`` of a box, ``sphere``, the ``side`` and the
ends ``start`` and ``end`` of a cylinder, ``bottom`` and ``top`` of an
extrusion, and the face swept by each side of the profile of an extrusion
or a revolution. They are picked in the view to define selections (see
:ref:`gui-selections-faces`). See ``examples/heat_sink_3d.json`` and
``examples/pulley_3d.json``.

Curves and surfaces
-------------------

.. image:: images/surface.png
   :width: 100%
   :alt: A paraboloid dish: a flat annulus plus a height

Beams, cables, shells and membranes are geometries of a lower dimension
than their space:

``curve`` (2D or 3D)
   The points of the curve are given in a table: from each point, a segment
   goes to the next one, or an arc of circle if a point it goes through is
   given in the last column. A closed curve joins its last point to the
   first one. The ends of an open curve are the built-in selections
   ``start`` and ``end``.
``surface`` (3D)
   A flat domain, a 2D constructive tree drawn and edited as a 2D geometry
   (the *Flat domain* node), placed by an origin and two axes, plus a height
   along the normal: a number or an expression of the local coordinates
   ``x``, ``y`` and of the parameters (``"(sqr(x) + sqr(y))/(4*f)"``). Its
   selections are those of the flat domain (edges, regions, criteria in
   the local coordinates).
``swept_surface`` (3D)
   A 2D curve given as for ``curve``, extruded along z between two values,
   or turned about the z axis between two angles (the curve must then not
   touch the axis).

Their meshes are built by the meshing script (see :ref:`gui-meshes`) and
the models use them as any mesh, the gradients being then tangential. The
modules for beams and shells are to come. See ``examples/heated_dish.json``.

CAD files and meshes
--------------------

A ``cad`` geometry reads a STEP, IGES or BREP file with |gmsh|; it is meshed
by |gmsh| or Netgen. |gmsh| keeps the units of the file (often millimeters).
A ``mesh_file`` geometry is a ``.msh`` mesh, imported as it is (the
``import`` mesher); its named physical groups are selections.
