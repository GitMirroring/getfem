.. include:: ../replaces.txt

.. highlight:: json

.. _gui-meshes:

Meshes
======

A mesh belongs to a geometry, and is built by a *mesher*; several meshes
may be defined for one geometry, and the variables of the models say which
one they use. The meshers, the sources they accept and their fields are
listed in :ref:`gui-ref-meshers`.

``structured``
   a regular grid of a box: a number of divisions per direction, simplices
   or quadrilaterals / hexahedra.
``getfem``
   the experimental mesher of |gf| (``gf.MesherObject``, which needs |gf|
   built with Qhull), for boxes, constructive geometries, curves and
   surfaces: element size ``h`` and geometric order. In 2D, it follows the
   interfaces between the regions of a constructive geometry. It is slow in
   3D, and may lose thin or sharp details.
``gmsh``
   |gmsh| with OpenCASCADE (``pip install gmsh``), for boxes, constructive
   and CAD geometries and the flat domains of surfaces: element size and
   order. It is the default mesher of the new meshes of constructive and CAD
   geometries.
``netgen``
   Netgen with its own OpenCASCADE (``pip install netgen-mesher``): element
   size, order 1 or 2. It does not mesh ellipses yet.
``import``
   a ``.msh`` file used as it is.

::

  {"name": "fine", "geometry": "plate", "mesher": "gmsh", "size": "L/40", "order": 2}

A new mesh of a box is structured; those of the other geometries use
|gmsh| (or the mesher chosen by ``GETFEM_GUI_DEFAULT_MESHER``). When a
geometry changes its kind of source, its meshes change their mesher if
needed, with an equivalent element size.

The **geometric order** 2 gives curved elements, which follow the circles
and the curved surfaces. They are drawn curved, the meshes as the results:
each face of an element is drawn in 4\ :sup:`n` triangles and each edge in
2\ :sup:`n` segments, placed (and colored, for a result) by the quadratic
interpolation, *n* being set by *Curved* in the bar of the view (shown for
such meshes; 2 by default, 0: straight elements through their corners).
The selections of a curved mesh are drawn refined as well.
The geometric order 3 (``getfem`` and ``gmsh`` meshers) follows the curved
boundaries more closely still; its elements are drawn by the same
quadratic interpolation.

Quadrangles and hexahedra
-------------------------

``"element": "quad"`` gives quadrangles in 2D with |gmsh| (Blossom
recombination then a subdivision: quadrangles only) or Netgen (quad
dominated), as well as for the flat domains of surfaces.

.. _gui-meshes-layers:

Meshes in layers
----------------

.. image:: images/mesh_layers.png
   :width: 100%
   :alt: A heat sink meshed in 12 layers of prisms

A 3D extrusion or revolution can be meshed **in layers**
(``"layers": n``, with any of the three meshers): its profile is meshed in
2D, then extruded or turned about the axis in *n* layers by |gf|
(``gf.Mesh('prismatic')``), which gives prisms, or hexahedra from
quadrangles. With a geometric order 2, the layers of a revolution are
curved. The profile of a revolution meshed in layers must not touch the
axis, and needs at least 3 layers.

Curves and surfaces
-------------------

A curve is meshed by elements of the element size, the nodes of the arcs
being on them; a surface by a mesh of its flat domain (with any 2D mesher),
mapped to 3D (curved with order 2); a swept surface by a mesh of its curve,
extruded or turned in layers (quadrangles).

The cache
---------

A mesh is only rebuilt when its geometry, its parameters or its source file
change (a SHA-256 hash of them): changing the physics does not remesh. The
meshes and their selections are kept in ``<model>_work/meshes``. The
messages of the mesher appear in *Messages*; the number of elements of each
selection is given at the end.

Status
------

The ``structured``, ``getfem`` and ``import`` meshers are tested against
exact solutions. The scripts of |gmsh| and Netgen are tested where these
modules are installed.
