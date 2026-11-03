# getfem_gui — the core, without an interface yet

Foundation of a future graphical user interface for GetFEM. There is no GUI
yet: a model described in JSON is meshed, then translated into a readable
GetFEM Python script, each step running in a separate process.

This directory is a pure Python package, independent of the GetFEM build: it
is neither compiled nor installed by `make` / CMake. It requires Python 3
(tested with 3.13), NumPy and the `getfem` module; Gmsh (`pip install gmsh`)
is optional.

## Getting started from the repository

```bash
cd gui
# the getfem module must be importable, e.g. after a CMake build:
export PYTHONPATH=<prefix>/lib/python3.X/site-packages
python -m getfem_gui run examples/plate.json -w /tmp/plate
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
   physics ── uses one mesh, only knows selection names
          │
          ▼
   run_getfem.py ──► results.json + .vtu
```

## Usage

```bash
python -m getfem_gui new model.json                         # starting model
python -m getfem_gui validate examples/plate_with_hole.json
python -m getfem_gui mesh examples/plate_with_hole.json     # all meshes, with selection summary
python -m getfem_gui run examples/plate_with_hole.json -w results/
python -m getfem_gui generate model.json -o solve.py        # solve script only
python -m getfem_gui migrate old_v1.json -o new.json
```

`--python /path/to/python` selects the interpreter that provides GetFEM (and Gmsh).

## Layout

| File | Role |
|---|---|
| `getfem_gui/model.py` | JSON format (v2), selections, validation, migration from v1 |
| `getfem_gui/mesher.py` | meshing step: generated script, regions, hash-based cache |
| `getfem_gui/codegen.py` | GetFEM solve script (loads the prepared mesh) |
| `getfem_gui/runner.py` | runs meshing then solving in subprocesses |
| `getfem_gui/cli.py` | command line interface |
| `tests/test_heat.py` | 16 heat transfer cases (one of them requires Gmsh) |
| `tests/test_elasticity.py` | 11 linear elasticity cases |
| `examples/` | heated plate (box), plate with a heated hole (CSG, two meshes), plate with a hole in tension (elasticity, symmetry) |

## Geometry, selections, meshes

The complete format description is in the docstring of `model.py`.

**Geometry sources**: `box` (1D/2D/3D), `csg` (rectangle, disk/ball,
cylinder, union, intersection, difference), `cad` (STEP, IGES, BREP read by
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
`in_box`, `in_ball`, `physical`, `union`. An empty selection is an explicit
error.

Beware: `normal` picks *all* boundary faces oriented in the given direction,
including those of a hole (the right edge of a hole has a −x outward normal,
like the left edge of the part). For "the left face", use `plane`; this is
what the predefined faces `xmin…zmax` do.

GetFEM region numbers are assigned by the application, never by the user.
When two materials cover the same area, the last one listed wins: one can
thus define a default material on `all`, then exceptions.

**Cache**: a mesh is only rebuilt when its geometry, its parameters or its
source file change (SHA-256 hash). Changing the physics does not remesh.

## Status of the meshers

- `structured`, `getfem` and `import` are tested against exact solutions
  (the `getfem` mesher requires GetFEM built with Qhull).
- `gmsh` (including CAD import) has **not been tested yet**. The test
  `test_gmsh_mesher_csg` runs automatically wherever `import gmsh` succeeds.
- CAD units: Gmsh/OpenCASCADE keeps the units of the STEP file (often mm);
  a scaling option will be needed.

## Physics

| Physics | Properties | Boundary conditions | GWFL macros |
|---|---|---|---|
| `heat_stationary` | `k` | `temperature`, `heat_flux`, `convection` | `q` (heat flux) |
| `linear_elasticity` | `E`, `nu` (option `plane`: `strain` / `stress` in 2D) | `fixed`, `displacement`, `roller`, `traction`, `pressure` | `eps`, `sigma`, `von_mises`, `mu_L`, `lambda_L` |

Values may be numbers or GWFL expressions; vector values (displacement,
traction, body force) are lists of components or a GWFL vector expression
such as `"[0, -9.81*rho]"`, where `rho` is any extra material property. The
macros can be used in derived quantities: `(sigma*Normal)(1)` integrated on a
boundary gives the x component of the reaction force.

`roller` imposes `u.n = 0` with Nitsche's method (GetFEM's
`add_normal_Dirichlet_condition_with_Nitsche_method`), without additional
unknowns. A first version with multipliers made the system singular as soon
as a roller shared a node with a `fixed` or `displacement` condition
(simplified Dirichlet), e.g. a clamped edge next to a symmetry plane.

**Solve check.** On a singular system, the direct solver only prints a
warning and may leave a zero solution. Every solve script therefore checks
the relative residual `|K U - F| / |F|` and stops with an explicit error if
it exceeds 1e-6. A singular but compatible system (e.g. a rigid motion left
free but not loaded) is solved, with a warning in `results["warnings"]`.

## Adding a physics

Add a declarative entry to `model.PHYSICS_SPECS` (field kind, properties,
options, boundary conditions and their fields, anchoring conditions, macro
names): validation uses it, and a GUI can build its forms from it. Then write
two functions in `codegen.py` (GWFL terms, post-processing) and register them
in `PHYSICS`.

## Open topics

- Check the syntax of GWFL expressions as early as validation.
- Local Gmsh refinement per selection, CAD face names.
- Several physics on different meshes in the same model (GetFEM allows it)
  and convergence studies over several meshes.
