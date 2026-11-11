.. include:: ../replaces.txt

.. highlight:: text

.. _gui-concepts:

Concepts
========

The model file
--------------

A model file contains, in this order:

* **parameters**: named constants (``L = 0.1``, ``a = L/10``), usable in
  every numerical field and expression;
* **materials**: named sets of properties (conductivity, Young's modulus...),
  copied from a library or written by hand;
* **geometries**: the domains, each with its named **selections** (parts of
  the boundary or of the domain);
* **meshes**: several meshes may be defined for one geometry (a coarse and a
  fine one, say);
* **fields**: quantities given by a value per domain selection, usable in
  every expression;
* **models**: each one becomes a ``gf.Model``: its variables (on a mesh, with
  a finite element method), its modules (predefined physics, with their
  properties, sources and conditions), its weak form terms in GWFL, its
  generic conditions, its solver and its outputs (integrated quantities);
* **solvers**: global solvers that chain the models (in sequence, or by
  staggered iterations).

::

  {
    "format_version": 3,
    "name": "Plate with an inclusion",
    "parameters": [{"name": "L", "value": 0.1}],
    "materials":  [{"name": "copper", "properties": {"k": 400}}],
    "geometries": [{"name": "plate", "source": {...}, "selections": {...}}],
    "meshes":     [{"name": "mesh", "geometry": "plate", "mesher": "gmsh", "size": "L/40"}],
    "models": [{
      "name": "thermal",
      "variables": [{"name": "T", "mesh": "mesh", "fem": "lagrange", "degree": 2}],
      "modules": [{"type": "heat_conduction", "variable": "T",
                   "properties": {"k": [{"selection": "all", "value": 0.2},
                                        {"selection": "inclusion", "material": "copper"}]},
                   "conditions": [{"type": "temperature", "selection": "cold", "value": 20}]}],
      "outputs": [{"name": "heat_flow", "expr": "-k*Grad_T.Normal", "selection": "cold"}]
    }]
  }

The window edits this file; it can also be written by hand. The complete
description of each object is in the :ref:`reference <gui-reference>`.

.. _gui-concepts-flow:

From the file to the results
----------------------------

::

                 geometry ──► named selections (boundaries, subdomains)
                     │
          ┌──────────┴──────────┐
       mesh A                 mesh B         selections evaluated on the mesh
          │                      │           -> GetFEM regions
          ▼                      ▼
   model "thermal"  ──────►  model "mechanical"   one gf.Model each; an import
                                                  passes a result to another model
          └────── solver: sequence or staggered iterations
                     ▼
            results.json + one .vtu file per model

**Build mesh** writes, for each mesh, a meshing script that builds the mesh
and evaluates the selections on it; the meshes are kept in a cache
(``<model>_work/meshes``), and a mesh is only rebuilt when its geometry, its
parameters or its source file change. **Compute** writes the solve script,
which loads the meshes, and runs it. *Compute ▸ Show the solve script* shows
it, and *File ▸ Export as a Python script* writes a single standalone script
(meshing, then solving) that runs without the interface.

Parameters, fields and expressions
----------------------------------

**Parameters** are evaluated in order: a parameter may use the ones listed
before it. Every numerical field of the geometries and meshes accepts an
expression of them (``"L/2"``, ``"2*n"`` for a number of divisions). They are
evaluated before meshing, so that changing a parameter remeshes only when
the geometry or the mesh actually changes. In the physics, they are |gf|
data, usable in any expression. Their syntax is the part common to |py| and
GWFL: numbers, ``+ - * /``, parentheses, ``pi``, ``sqr``, ``sqrt``, ``pow``,
``exp``, ``log``, the trigonometric and hyperbolic functions, ``abs``,
``min``, ``max``.

A parameter may also be a **vector** or a **matrix**::

  {"name": "g", "value": [0, 0, -9.81]}
  {"name": "C", "value": "[L/2, H/2]"}
  {"name": "K", "value": "[k1, 0; 0, k2]"}       (by rows, as in GWFL)
  {"name": "K", "value": [[1, 0], [0, 2]]}       (in the file: a list of rows)

In the table of the parameters, ``[1, 0, 0]`` gives a vector and
``[1, 0; 0, 2]`` a matrix, written as in GWFL; the column "=" shows their
values. A text ``[[1, 0], [0, 2]]`` is refused: GWFL reads it as a list of
*columns*, so that writing it would be ambiguous. Their components are written ``C(1)``, ``K(1, 2)``, numbered from 1
as in GWFL; the expressions of parameters also know the dot product of two
vectors ``u.v``, the products ``K*u`` and ``K*M``, the sum and difference
of values of the same size, and ``Norm``, ``Norm_sqr``, ``Trace``, ``Det``,
``Inv``, ``Transpose``, ``Id(n)``.

In the geometries and meshes, a component is a number like any other
(``"radius": "C(1)/4"``), and a vector parameter can give a point or a
direction as a whole (``"center": "C"``): typed in the first box of the
field, it replaces the components, and the field becomes one line (typing
``0.1, 0.2`` in it gives the components back). The same holds for the
vertices of a polygon and the points of a curve, in their tables: a vector
parameter typed in the first column gives the whole point (and, for a
curve, in the column of the point an arc goes through). Dragging the handle of a
point given by a vector parameter changes the parameter. In the physics,
vectors and matrices are |gf| data of their sizes: ``g(3)``, ``K*Grad_T``,
``g.Normal``.

**Fields** have a value per domain selection of a geometry (the last one
listed wins: a default on ``all``, then exceptions); they may use the
coordinates, the parameters and the fields defined before them. A field is
a scalar, a **vector** (one value per component, or a vector expression) or
a **matrix** of the dimension of the space (``[a, b; c, d]`` by rows, a
matrix parameter, or a number standing for its multiple of the identity),
by its ``kind``::

  {"name": "fibre", "kind": "vector", "values": [{"selection": "all", "value": "[cos(t), sin(t)]"}]}

Scalar and vector fields are written to the |vtu| files of the results.

**Expressions** of the physics (properties, sources, values of the
conditions, terms, outputs) are written in GWFL, the weak form language of
|gf| (see :ref:`ud-gasm-high`). The coordinates are ``x``, ``y``, ``z`` (or
``X(1)``...), and the macros of the modules (the heat flux ``q``, the stress
``sigma``...) are available. A vector value is a list of values, one per
component, or a GWFL vector expression such as ``"[0, -9.81*rho]"``.

The expressions are checked by |gf| itself in the background, a moment
after each change: each of them is evaluated in the context of its model on
a mesh of one or two elements, and the errors are shown on the fields, in
the tree and in the *Problems* tab::

  materials[0].properties.E: Unbalanced parenthesis (at character 12)
  models[0].modules[0].sources[0].value: a vector of 2 components is expected, got 3

Scopes of the names
-------------------

The parameters, materials and fields of the top level are visible
everywhere; a model may have its own parameters and fields (and geometries
and meshes), visible in it only. A local name may not hide a global one.
Geometry and mesh names are unique in the whole file. The coordinates and
the GWFL functions are reserved names.

When two values of a property or of a field cover the same part of the
domain, the last one listed wins. The |gf| region numbers are assigned by
the interface, never by the user.

The working folder
------------------

Meshes and results go to ``<model>_work/`` next to the model file (a
temporary folder for a model not saved yet), and are found again when the
model is reopened. Results are marked *out of date* as soon as the model
changes (the display settings excepted). A mesh whose geometry was edited
is still shown, marked out of date, until it is rebuilt; a new mesh (even
under the name of a deleted one) shows nothing before it is built.
