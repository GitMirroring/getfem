.. include:: ../replaces.txt

.. highlight:: bash

.. _lab-export:

Export
======

Geometries and meshes are exported from the *File* menu, or by *Export...*
on a geometry or a mesh in the tree; the suffix of the file gives the
format (:ref:`lab-ref-formats`).

Geometries
----------

A box or ``csg`` geometry is written:

* in 2D, as SVG or DXF outlines;
* in 3D, as STL, OBJ, PLY or VTP surfaces: the exact preview surface;
* in 2D and 3D, as STEP, IGES or BREP solids, built with OpenCASCADE by the
  |gmsh| |py| module.

Meshes
------

A mesh is written as:

* a |gf| mesh (``.mesh``), read by ``gf.Mesh('load', ...)``;
* a |vtk| file (``.vtu``, ``.vtk``);
* a |gmsh| 2.2 file (``.msh``): each element is in the physical group of its
  first domain selection (``domain`` otherwise), and the faces of the
  boundary selections are in physical groups of their own;
* an Abaqus input file (``.inp``): element sets for the domain selections,
  node sets for the boundary ones.

Linear and quadratic elements are written, and the prisms and hexahedra of
the meshes in layers. A mesh not up to date is built first.

Python scripts
--------------

*File ▸ Export as a Python script* writes a single standalone script that
builds the meshes, then solves the models, with |gf| only: to read it, to
modify it, or to run it elsewhere.

Command line
------------

::

  python -m getfem_lab export-geometry model.json NAME out.step
  python -m getfem_lab export-mesh model.json NAME out.msh
  python -m getfem_lab export model.json -o model_getfem.py

See :ref:`lab-ref-cli`.
