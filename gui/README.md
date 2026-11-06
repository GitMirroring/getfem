# getfem_gui — a graphical user interface for GetFEM

A model file described in JSON is meshed, then translated into a readable
GetFEM Python script, each step running in a separate process. The file
follows the logic of GetFEM: shared definitions (parameters, materials,
geometries, meshes, fields), then *models*, each one a `gf.Model` with its
variables, modules (predefined physics), weak form terms and conditions, then
global solvers chaining the models (see `doc/model_format_v3.md`). A first graphical
interface (PySide6 window, PyVista 3D view) edits the model, builds the
meshes, runs the computation and shows the results; everything it does is
also available from the command line.

This directory is a pure Python package, independent of the GetFEM build: it
is neither compiled nor installed by `make` / CMake. It requires Python 3
(tested with 3.13), NumPy and the `getfem` module; Gmsh (`pip install gmsh`)
is optional. The graphical interface needs PySide6, and PyVista with
pyvistaqt for the 3D view (`pip install pyside6 pyvista pyvistaqt`); the core
and the command line do not.

## Getting started from the repository

```bash
cd gui
# the getfem module must be importable, e.g. after a CMake build:
export PYTHONPATH=<prefix>/lib/python3.X/site-packages
python -m getfem_gui run examples/plate.json -w /tmp/plate
python -m getfem_gui gui examples/plate_hole_tension.json   # graphical interface
python -m pytest tests/        # or: python tests/test_heat.py
```

## Overview

```
                 geometry ──► named selections (boundaries, subdomains)
                     │
          ┌──────────┴──────────┐
       mesh A                 mesh B          (structured, GetFEM mesher, Gmsh, import)
          │                      │            selections evaluated -> GetFEM regions
          ▼                      ▼            cache: <workdir>/meshes/<name>.mesh/.json
   model "thermal"         model "mechanical"   one gf.Model each: variables on a mesh,
   (heat conduction)  ──►  (elasticity,          modules, terms, conditions; an import
                            expansion, imports T) passes a result from one to the other
          │                      │
          └────── solver: sequence or staggered iterations
                     ▼
   run_getfem.py ──► results.json + one .vtu per model
```

## Usage

```bash
python -m getfem_gui new model.json                         # starting model
python -m getfem_gui validate examples/plate_with_hole.json
python -m getfem_gui mesh examples/plate_with_hole.json     # all meshes, with selection summary
python -m getfem_gui run examples/plate_with_hole.json -w results/
python -m getfem_gui generate model.json -o solve.py        # solve script only
python -m getfem_gui export model.json -o model_getfem.py   # standalone script: mesh + solve
```

`--python /path/to/python` selects the interpreter that provides GetFEM (and Gmsh).

## Graphical interface

```bash
python -m getfem_gui gui [model.json]       # --no-splash: no start-up window
```

A start-up window shows the versions ("GetFEM GUI 0.1 on GetFEM 5.5") for
about three seconds while the interface loads. The version of GetFEM is asked
to the interpreter that runs the computations (`getfem.util_version()`).

```
┌ Model tree ─────────┬ Properties ───────────┬ 3D view ──────────────────────┐
│ Geometries          │ Type     [traction ▾] │                               │
│   plate (csg)       │ Boundary [right    ▾] │   mesh, highlighted selection │
│     Selections      │ Value    [50] [0]     │   or result field             │
│ Meshes              │                       │                               │
│ Models              │ [Remove]              │                               │
│   mechanics         │                       │                               │
│     Modules, Terms… │                       │                               │
│ Results             │                       │                               │
├─────────────────────┴───────────────────────┴───────────────────────────────┤
│ Messages | Problems (n)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

- The **model tree** shows every part of the model; nodes with a problem
  carry a warning icon (the message is in the tooltip and in the *Problems*
  tab, where a double click selects the node).
- The **properties** form of a node is built from the model description
  (`PHYSICS_SPECS`, meshers, criteria): adding a physics or a boundary
  condition type does not require any interface code. Values are numbers or
  GWFL expressions (with the coordinates x, y, z); vectors are edited per component.
- 2D box and CSG geometries are drawn exactly, without meshing: selecting a
  geometry, a shape, a selection, a boundary condition... shows the geometry
  with the support highlighted. With a selection node selected, **clicking an
  edge adds or removes it** (the left button still pans the view: a click is
  a press and release without moving). Picked edges are stored as sides of
  primitives (see `geometry2d.py`), so that a selection follows the changes
  of the geometry, e.g. a parameter for the radius of a hole. Other
  geometries (CAD, imported meshes, 3D) are shown through their mesh.
- **Drawing**: with a 2D geometry selected, the *Rectangle* and *Disk*
  tools of the view draw a shape by dragging (corner to corner, center to
  circle), snapped to a grid whose step follows the zoom; it is added to,
  subtracted from or intersected with the geometry (Escape stops drawing).
  The *Ellipse* tool draws the ellipse inscribed in the rectangle of two
  corners. The *Polygon* tool takes a vertex per click; clicking the first vertex
  again, double-clicking or Enter closes it, Backspace removes the last
  vertex (a polygon that crosses itself is refused).
  The shape joins the root boolean of the CSG tree when it is the same
  operation, otherwise the root is wrapped in a new boolean, and the `edges`
  selections are updated so that they still designate the same sides. A box
  becomes a CSG geometry on the first drawing: its sides `xmin`... become
  selections of the same name (their uses stay valid), and a structured mesh
  of it switches to the GetFEM mesher with an equivalent element size.
- **Handles**: with a rectangle, a disk, an ellipse or a polygon selected in
  the tree, its handles (corners and center; center and a point of the
  circle; center and the ends of the two semi-axes, the first one turning
  the ellipse; vertices, middles of the sides and center) move or resize it, snapped to
  the grid; dragging the middle of a side of a polygon adds a vertex. A value that is a number is changed; a value that is
  the name of a parameter changes the parameter (dragging the circle of the
  hole of `plate_hole_tension.json` changes `a`); a value given by an
  expression of parameters (`"L/2"`) is locked, and the status bar says so.
- With a boundary condition (or an output integrated on a boundary)
  selected, clicking edges edits its support directly: its selection is
  modified in place if it is an `edges` selection used by this item only;
  otherwise a new selection is created from the current support (named
  after the item, e.g. `traction_3`), leaving the previous one to the other
  items that use it.
- **Subdomains**: the sides of the primitives that lie inside the domain
  (the circle of an inclusion united with a plate) are drawn as thin lines:
  they cut the domain into *regions*, each one given by the primitives that
  contain it ("in rectangle 1 and in ball 2"). With a domain selection, a
  material, a source or an output on a domain selected, its support is
  highlighted, the region under the mouse too, and **clicking a region adds
  or removes it**, with the same rules as for edges (a material on `all`
  gets a new selection named after it). See `examples/plate_inclusion.json`.
- **Build mesh** (F7) and **Compute** (F8) run in separate processes, whose
  output appears in *Messages*; **Stop** interrupts them. Meshes and results
  go to `<model>_work/` next to the model file (a temporary folder for an
  unsaved model), and are found again when the model is reopened; results
  are marked *out of date* as soon as the model changes.
- **Tree**: it follows the file (Parameters, Materials, Geometries, Meshes,
  Fields, Models with their Variables, Modules, Terms, Conditions, Solver,
  Outputs, then Solvers and Results). A right click on a node lists its
  actions; the **+** button above the tree lists what can be added to the
  selected node (a model from a module, a module, a term, a condition...).
  The parameters (of the file or of a model) are edited together, in a
  table: name, value or expression, its number, description; rows are added,
  removed and moved (a parameter may only use the ones above it), and a
  renamed parameter is renamed in all its uses.
- **3D geometries** (box, CSG) are previewed without meshing too: the
  surface of each primitive is clipped to its part on the boundary of the
  whole shape (along a smooth combination of the distance functions of the
  other primitives), so that the faces stay flat and the edges sharp; the
  boundary selections (plane, normal, box, ball criteria) and the shape
  selected in the tree are shown on it. See `examples/block_with_hole_3d.json`.
- **Export** (File menu, or *Export...* on a geometry or a mesh in the
  tree): a box or CSG geometry as SVG or DXF outlines (2D), STL, OBJ, PLY or
  VTP surfaces (3D, the exact preview surface), STEP, IGES or BREP solids
  (2D and 3D, built with OpenCASCADE by the Gmsh Python module); a mesh as
  GetFEM (`.mesh`), VTK (`.vtu`, `.vtk`), Gmsh 2.2 (`.msh`: each element in
  the physical group of its first domain selection, `domain` otherwise, and
  the faces of the boundary selections in physical groups of their own) or
  Abaqus (`.inp`: element sets for the domain selections, node sets for the
  boundary ones), linear or quadratic elements. A mesh not up to date is
  built first. Command line: `export-geometry model.json NAME out.step`,
  `export-mesh model.json NAME out.msh`.
- **Picking 3D faces**: with a boundary selection of a 3D box or CSG
  geometry, or an item on a boundary (condition, term, output), selected,
  the face under the mouse is highlighted and a click adds or removes it
  (a `faces` selection, created for the item as for 2D edges). A face is a
  face of a primitive (`xmin`… `zmax` of a box, the `side` and the ends of
  a cylinder, `bottom` and `top` of an extrusion, the side swept by a side
  of a profile), stored by reference, so that it follows the parameters; on
  the mesh, the boundary faces whose vertices lie on it (and whose normal
  agrees) are taken.
- **Ellipses** (2D): center, semi-axes, angle of the first semi-axis in
  degrees; their boundary is the side `ellipse`. A disk changed into an
  ellipse keeps its center and radius (and its `edges` references). They
  need the `ellipse` mesher object of GetFEM (patch "mesher: a working
  ellipse"); with Gmsh, an OpenCASCADE disk turned by the angle.
- **Polygons** (2D, not necessarily convex, vertices in either order) are
  primitives like rectangles and disks: their vertices are edited in a
  table (values or expressions of parameters; *Insert* adds a vertex in the
  middle of a side, *Remove* removes one), their sides (`side1`, `side2`...)
  can be picked in `edges` selections and follow the insertions and
  removals, and a rectangle or a disk changed into a polygon keeps its
  corners (an octagon for a disk). Polygons need the `polygon` mesher object
  of GetFEM (patch "mesher: polygons"). See `examples/l_bracket.json`.
- **Extrusion and revolution**: a 3D CSG shape may be a 2D profile (itself a
  CSG tree of rectangles and disks, edited under the shape in the tree)
  extruded along z, or turned about the z axis (the profile is then given in
  the (r, z) plane). The GetFEM mesher needs the `extrusion` and `revolution`
  mesher objects (GetFEM patch "mesher: extrusion and revolution of 2D
  objects"); with Gmsh, OpenCASCADE extrudes or revolves the profile. See
  `examples/heat_sink_3d.json` and `examples/pulley_3d.json`.
- **Materials**: *Add from a library* (a menu by category) copies a material of the built-in
  library (`getfem_gui/data/materials.json`, 27 common materials, typical
  values near 20 °C in SI units, read only) or of your own library
  (`~/.config/getfem_gui/materials.json`, or `$GETFEM_GUI_MATERIALS`; a
  material of the model is added to it with *Save in my library*). Only the
  copies are in the model file, with their origin; the solve script only
  gets the properties the modules use. The Materials node shows a table of
  the materials and the properties asked by the modules of the file.
- **Results**: one entry per model, then its fields; with a result field selected, its form sets the component
  (or magnitude), the deformed shape, the colormap, the color range (empty
  bounds: the range of the values), isolines (evenly spaced in the range),
  color bands, and arrows of a vector field (on a regular grid, length
  proportional to the magnitude). A click on a 2D field shows its value
  under the view's toolbar. The **Cut line** tool draws a segment A → B
  (or it is typed in the form): the field is plotted along it under the view
  (NaN gaps outside the domain), and the values can be saved as CSV.
  These settings are saved in the model file, in the **Graphics** of the
  model (one per result field by default, named after it; *Add a graphic*
  gives other views of a field, e.g. with bands, with their own cut line);
  they come back when the file is reopened, and changing them does not make
  the results out of date.
- Undo / redo for every change; *Compute ▸ Show the solve script* shows the
  generated GetFEM script, and *File ▸ Export as a Python script* writes a
  single standalone script (meshing, then solving) that runs without
  getfem_gui, to read, modify or run elsewhere.

The interface is split so that most of it can be tested without a screen:
`editing.py` (tree, forms and consistent changes: renaming a selection
updates its uses, changing a type resets the fields, ...) has no Qt
dependency, `app/scene.py` only needs PyVista, and `app/` holds the Qt code.
`tests/test_app.py` drives the window with the Qt `offscreen` platform; the
3D view is then replaced by a placeholder, or, with
`GETFEM_GUI_OFFSCREEN_3D=1` (and an X server such as `xvfb-run`), by an
off-screen PyVista plotter, so that only the embedding of the pyvistaqt
widget itself needs a real display.

## Layout

| File | Role |
|---|---|
| `getfem_gui/model.py` | JSON format (v3), scopes of names, selections, validation |
| `getfem_gui/mesher.py` | meshing step: generated script, regions, hash-based cache |
| `getfem_gui/codegen.py` | GetFEM solve script (loads the prepared mesh) |
| `getfem_gui/runner.py` | runs meshing then solving in subprocesses |
| `getfem_gui/cli.py` | command line interface |
| `getfem_gui/export.py` | export of geometries (SVG, DXF, STL, STEP...) and meshes (VTK, Gmsh, Abaqus) |
| `getfem_gui/editing.py` | model tree, forms and editing operations (no GUI toolkit) |
| `getfem_gui/gwfl.py` | check of the GWFL expressions by GetFEM, in a subprocess |
| `getfem_gui/library.py`, `data/materials.json` | material libraries (built-in, user's), copies into the model |
| `getfem_gui/geometry3d.py` | 3D shapes as distance functions, selections on the preview surface |
| `getfem_gui/geometry2d.py` | exact 2D geometry: boundary pieces of CSG shapes, picked edges, regions and interfaces |
| `getfem_gui/parameters.py` | global parameters: safe evaluation, resolution of numerical fields |
| `getfem_gui/app/window.py` | main window (PySide6) |
| `getfem_gui/app/properties.py` | properties form of the selected node |
| `getfem_gui/app/viewer.py`, `scene.py` | 3D view (pyvistaqt) and the PyVista scenes (fields: isolines, arrows, cut lines) |
| `getfem_gui/app/plot.py` | small line plot (QPainter) for the values along a cut line |
| `getfem_gui/app/jobs.py` | meshing and solving through QProcess (output, interruption) |
| `tests/test_heat.py` | 16 heat transfer cases (one of them requires Gmsh) |
| `tests/test_elasticity.py` | 11 linear elasticity cases |
| `tests/test_gwfl.py` | GWFL check: errors located on the right field, explained |
| `tests/test_geometry2d.py` | boundary pieces, `edges` selections on the mesh, evaluation of selections |
| `tests/test_parameters.py` | parameters (evaluation, remeshing, physics) and user fields |
| `tests/test_coupling.py` | thermal expansion (monolithic, sequence), staggered two-way coupling, user terms and generic conditions |
| `tests/test_editing.py` | editing operations, on every node of every example |
| `tests/test_app.py` | the window driven without a screen (requires PySide6) |
| `tests/test_scene.py` | the 3D scenes rendered off screen (requires PyVista) |
| `tests/test_export.py` | export of geometries and meshes (read back by GetFEM) |
| `tests/test_faces3d.py` | faces of 3D geometries: references, selections, mesh, picking in the window |
| `tests/test_ellipse.py` | ellipses: geometry, format, handles, drawing, mesh, solve |
| `tests/test_polygon.py` | polygons: geometry, format, table, handles, drawing, mesh, solve |
| `tests/test_geometry3d.py` | extrusion and revolution: format, editing, preview, mesh |
| `examples/` | heated plate (box), plate with a heated hole (CSG, two meshes), plate with a hole in tension (elasticity, symmetry), plate with a conductive inclusion (two materials on regions), heated plate that expands (two models in sequence), L-shaped bracket (polygon), 3D block with a hole, extruded heat sink, pulley (revolution) |

## Geometry, selections, meshes

The complete format description is in the docstring of `model.py`.

**Geometry sources**: `box` (1D/2D/3D), `csg` (rectangle, disk/ball,
ellipse, polygon, cylinder, extrusion and revolution of a 2D profile, union, intersection,
difference), `cad` (STEP, IGES, BREP read by
Gmsh/OpenCASCADE), `mesh_file` (a `.msh` mesh used as the geometry).

**Meshers** and accepted sources:

| Mesher | box | csg | cad | mesh_file |
|---|---|---|---|---|
| `structured` (grid, simplices or quadrilaterals) | ✓ | | | |
| `getfem` (experimental mesher, `MesherObject`) | ✓ | ✓ | | |
| `gmsh` (OpenCASCADE) | ✓ | ✓ | ✓ | |
| `import` | | | | ✓ |

**Selections**: `all` and `boundary` always exist; `box` adds
`xmin … zmax`, a `.msh` file brings its named physical groups. Other
selections are defined by criteria evaluated *on the mesh*, which makes them
independent of the mesher and robust to remeshing: `plane`, `normal`,
`in_box`, `in_ball`, `physical`, `union`, `edges` (2D box or CSG: sides
of primitives, as picked in the view, e.g.
`{"type": "edges", "edges": [{"shape": [1], "side": "circle"}]}`) and
`regions` (2D box or CSG: elements in regions, as picked in the view, each
region being the list of the paths of the primitives that contain it, e.g.
`{"type": "regions", "regions": [[[0], [1]]]}` for "in rectangle 1 and in
ball 2"; an element belongs to the region of its center). An empty selection is an explicit
error.

**Conforming subdomains**: for a 2D CSG geometry, the GetFEM mesher is given
fixed vertices every `h` along the interfaces between regions, which makes the
elements follow them: no element crosses the circle of an inclusion, and a
material jump is represented exactly (`test_regions_follow_the_interface`
recovers the piecewise linear solution of a layered bar with P1 elements).
With Gmsh, the shapes are still fused: a region is then approximated by the
elements whose center is in it (a staircase); fragmenting the shapes
(`occ.fragment`) is left for later.

Beware: `normal` picks *all* boundary faces oriented in the given direction,
including those of a hole (the right edge of a hole has a −x outward normal,
like the left edge of the part). For "the left face", use `plane`; this is
what the predefined faces `xmin…zmax` do. In 3D, the `faces` criterion
(picked faces) designates the faces of the primitives themselves.

GetFEM region numbers are assigned by the application, never by the user.
When two values of a property (or of a field) cover the same area, the last
one listed wins: one can thus define a default on `all`, then exceptions.

**Cache**: a mesh is only rebuilt when its geometry, its parameters or its
source file change (SHA-256 hash). Changing the physics does not remesh.

## Status of the meshers

- `structured`, `getfem` and `import` are tested against exact solutions
  (the `getfem` mesher requires GetFEM built with Qhull).
- `gmsh` (including CAD import) has **not been tested yet**. The test
  `test_gmsh_mesher_csg` runs automatically wherever `import gmsh` succeeds.
- CAD units: Gmsh/OpenCASCADE keeps the units of the STEP file (often mm);
  a scaling option will be needed.

## Models, modules, terms

A model is one `gf.Model`. Its **variables** carry their discretization
(mesh, Lagrange element and degree, scalar or vector); its **modules** are
predefined physics acting on them:

| Module | Properties | Conditions | GWFL macros |
|---|---|---|---|
| `heat_conduction` | `k` | `temperature`, `heat_flux`, `convection` | `q` (heat flux) |
| `linear_elasticity` | `E`, `nu` (option `plane`: `strain` / `stress` in 2D) | `fixed`, `displacement`, `roller`, `traction`, `pressure` | `eps`, `sigma`, `von_mises`, `mu_L`, `lambda_L` |
| `thermal_expansion` | `alpha` (value `T_ref`) | | adds `- beta alpha (T - T_ref) I` to `sigma` |

Properties are given per domain selection, by a value (number or GWFL
expression, possibly depending on the variables: a conductivity `k(T)` makes
the model nonlinear) or by a **material** of the top level. **Terms** are
GWFL weak forms integrated on a selection, summing to zero (`"-f*Test_u"` for
a source); a term that GetFEM does not accept as linear is added as a
nonlinear one, and the model is then solved by Newton's method. **Generic
conditions** act on a variable: `dirichlet` (simplification, multipliers or
penalization), `normal_dirichlet`, `neumann`, `robin`. The heat conduction
module can be *unfolded* into its terms and conditions (tree action).

Several models are solved by a global **solver**: `sequence` (in order), or
`staggered` (iterated until the relative change of the variables is below a
tolerance, with relaxation). A model takes the variable of another one with
an import, `{"name": "T", "model": "thermal", "variable": "T"}`, on the same
mesh for now. Without global solver, the models are solved in the order of
their imports. `test_staggered_two_way_coupling` checks that staggered
iterations between two models converge to the monolithic Newton solution.

`roller` imposes `u.n = 0` with Nitsche's method (GetFEM's
`add_normal_Dirichlet_condition_with_Nitsche_method`), without additional
unknowns, so that it may share nodes with a `fixed` condition.

**Solve check.** On a singular system, GetFEM releases up to 5.4 only print
a warning and may leave a zero solution. For a linear model, the solve
script therefore checks the relative residual `|K U - F| / |F|` and stops
with an explicit error if it exceeds 1e-6. When `md.solve()` reports
`converged = 0` while the residual is small, the system is singular but
compatible: it is solved, with a warning.

## Adding a module

Add a declarative entry to `model.MODULE_SPECS` (variables and their roles,
properties, options, values, conditions and their fields, anchoring
conditions, macro names): validation uses it, and the interface builds its
forms from it. Then write three functions in `codegen.py` (macros, GWFL
terms and conditions, post-processing) and register them in `MODULES`; the
first one is also used to check the user expressions.

## Parameters, fields, coordinates

```json
"parameters": [{"name": "L", "value": 0.1, "description": "half length of the plate"},
               {"name": "a", "value": "L/10", "description": "radius of the hole"}],
"fields": [{"name": "porosity",
            "values": [{"selection": "all", "value": "0.1*(1 + x)"},
                       {"selection": "inclusion", "value": 0.3}]}]
```

- **Parameters** are named constants, evaluated in order (a parameter may use
  the ones listed before it). Every numerical field of the geometries and
  meshes accepts an expression of them (`"L/2"`, `"2*n"` for divisions); they
  are evaluated before meshing, so that changing a parameter remeshes only
  when the geometry or the mesh actually changes. In the physics, they are
  GetFEM data, usable in any expression. Their syntax is the part common to
  Python and GWFL: numbers, `+ - * /`, parentheses, `pi`, `sqr`, `sqrt`,
  `pow`, `exp`, `log`, trigonometric and hyperbolic functions, `abs`, `min`,
  `max`; it is evaluated without `eval`.
- **Fields** have a value per domain selection of a geometry (the last one
  listed wins); they may use the coordinates, the parameters and the fields
  defined before them, are usable in every expression of the models on that
  geometry, and are exported with the results.
- **Scopes**: the parameters, materials and fields of the top level are
  visible everywhere; a model may have its own parameters and fields (and
  geometries and meshes), visible in it only. A local name may not hide a
  global one.
- **Coordinates** are `x`, `y`, `z` in every expression (up to the dimension of
  the mesh; `X(1)`... also works). These names and the GWFL functions are
  reserved.

`examples/plate_hole_tension.json` is written with parameters.

## Checking the GWFL expressions

`model.validate` checks the structure of a file without GetFEM. The
expressions written by the user (properties, sources, condition values,
terms, integrated quantities) are then checked by GetFEM itself
(`gwfl.check`): a small script evaluates each of them in the context of its
model in the solve script (same variables, imports, property names and
macros), on a mesh of one or two elements, and reports the errors as
`path: message`, e.g.

```
materials[0].properties.E: Unbalanced parenthesis (at character 12)
models[0].modules[0].sources[0].value: a vector of 2 components is expected, got 3 components
models[0].outputs[3].expr: Invalid outward unit normal vector. Possible reasons: not on boundary ...
```

`validate` and `run` (command line, `runner.run`) perform this check before
meshing; the graphical interface runs it in the background a moment after
each change and shows the errors on the fields, in the tree and in the
*Problems* tab. A model made of numbers only needs no check.

## Open topics

- Editing the geometries and meshes local to a model in the interface (the
  format, the validation and the solve already support them).
- Imports between different meshes (interpolation), time dependent models
  and the `time_loop` and `sweep` solvers, global outputs and saved graphics.
- Unfolding the elasticity module (its macros), a library of terms.
- Local Gmsh refinement per selection, CAD face names; fragmenting the CSG
  shapes for conforming subdomains with Gmsh.
