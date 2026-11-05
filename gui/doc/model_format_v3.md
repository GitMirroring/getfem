# GetFEM GUI — model format v3 (design)

Status: agreed design, being implemented (November 2026). It replaces the
format v2 of `model.py`, without compatibility.

## Principles

- The format follows GetFEM: a file holds shared definitions, then one or
  several **models**; each model becomes one `gf.Model`, with its own
  parameters, data and variables. Strong (monolithic) couplings are written
  inside one model.
- Several models may coexist; weak couplings (sequence, staggered
  iterations) go through **global solvers**.
- Predefined physics are **modules** inserted in a model: they create
  variables, terms and conditions, and can be *unfolded* into plain GWFL
  terms.
- Discretization belongs to the **variables** (as `mesh_fem` in GetFEM).
- What is generated stays readable: the exported script is an ordinary
  GetFEM script with the structure of the file.

## Top level

| Key | Content | GetFEM |
| --- | --- | --- |
| `parameters` | named constants, evaluated in order | Python values; fixed data in the models that use them |
| `materials` | named sets of property values (`copper`: k, E, nu, rho...) | values used by the modules |
| `geometries` | sources (box, csg, cad, file) and named selections | `MesherObject`, Gmsh, or an imported mesh |
| `meshes` | meshes of a geometry; selections become regions | `gf.Mesh` + `set_region` |
| `fields` | fields of a geometry, per domain selection; or a variable of a model shared by all the models | GWFL macro, or FEM data |
| `models` | the models | one `gf.Model` each |
| `solvers` | sequence, staggered iterations, sweep, time loop | Python loop around `md.solve()` |
| `outputs` | quantities and exports combining several models | `asm_generic`, VTU |
| `graphics` | saved views (field, isolines, cut line, curves) | display settings only |

```json
{
  "format_version": 3,
  "name": "Thermo-elastic plate",
  "parameters": [{"name": "L", "value": 0.1}],
  "materials": [{"name": "aluminium", "properties": {"k": 237, "E": 70e9, "nu": 0.33}}],
  "geometries": [{"name": "plate", "source": {}, "selections": {}}],
  "meshes": [{"name": "mesh", "geometry": "plate", "mesher": "getfem", "h": "L/40"}],
  "fields": [{"name": "porosity", "geometry": "plate", "values": []}],
  "models": [{"name": "thermal"}, {"name": "mechanical"}],
  "solvers": [{"name": "main", "type": "sequence", "steps": ["thermal", "mechanical"]}],
  "outputs": [],
  "graphics": []
}
```

## Inside a model

```json
{
  "name": "thermal",
  "parameters": [{"name": "T0", "value": 20}],
  "geometries": [],
  "meshes": [],
  "fields": [],
  "imports": [],
  "variables": [{"name": "T", "mesh": "mesh", "fem": "lagrange", "degree": 2}],
  "modules": [{"type": "heat_conduction", "variable": "T",
               "properties": {"k": [{"selection": "matrix", "value": 0.2},
                                    {"selection": "inclusion", "material": "copper"}]},
               "sources": [],
               "conditions": [{"type": "convection", "selection": "right", "h": 25, "T_ext": "T0"}]}],
  "terms": [],
  "conditions": [{"type": "dirichlet", "variable": "T", "selection": "left", "value": "T0"}],
  "solver": {"type": "auto"},
  "outputs": [{"type": "integral", "name": "heat_flow", "expr": "-k*Grad_T.Normal", "selection": "left"}],
  "graphics": []
}
```

Local geometries and meshes have the structure of the top level ones: they
make a model self-contained.

## Variables

| Key | Values | GetFEM |
| --- | --- | --- |
| `mesh` | a mesh of the file or of the model | the `MeshFem` is built on it |
| `fem` | `lagrange` (PK or QK), `lagrange_discontinuous`, `hierarchical`, or a raw GetFEM name | `set_fem` / `set_classical_fem` |
| `degree` | integer | |
| `components` | 1 (default), `"dim"`, or an integer | `qdim` |
| `selection` | optional: a variable on a part of the domain | `add_filtered_fem_variable` |

Variables with the same mesh, element, degree and qdim share a `MeshFem`; one
`MeshIm` per mesh, its degree deduced from the variables and the geometric
order (`"integration"` in the mesh overrides it). The multipliers of
Dirichlet conditions are created automatically.

## Terms and conditions

A term is a GWFL weak form integrated on a selection; the terms of a model
sum to zero, as with `add_linear_term` (a source is written `-f*Test_u`).
Keys: `expr`, `selection` (a boundary gives a boundary term), `linear`
(`"auto"`, true, false: `add_linear_term` or `add_nonlinear_term`),
`integration` (optional degree).

Generic conditions on a variable:

| Type | Parameters | GetFEM |
| --- | --- | --- |
| `dirichlet` | `value`, `method` | `multiplier` (default), `penalization`, `nitsche` (needs the flux expression) |
| `normal_dirichlet` | `value`, `method` | `add_normal_Dirichlet_condition_with_*` |
| `neumann` | `value` (incoming flux) | boundary term `-g*Test_u` |
| `robin` | `h`, `value` | boundary term `h*(u - g)*Test_u` |

## Modules

A module is declared in Python (as `PHYSICS_SPECS` today) with its GWFL
terms:

```python
"heat_conduction": {
    "label": "Heat conduction",
    "variables": {"T": {"components": 1, "degree": 2}},   # created if absent
    "properties": {"k": "thermal conductivity"},
    "macros": {"q": "-k*Grad_T"},
    "terms": ["k*Grad_T.Grad_Test_T"],
    "sources": {"value": "-{value}*Test_T"},
    "conditions": {
        "temperature": {"dirichlet": {"value": "{value}"}},
        "heat_flux": {"neumann": {"value": "{value}"}},
        "convection": {"robin": {"h": "{h}", "value": "{T_ext}"}},
    },
}
```

- variable names are the module's by default, renamable when inserted;
- properties are given per domain selection, as values, expressions or
  materials;
- a module may use the variables of another module of the same model
  (`thermal_expansion` uses `T` and `u`; `T` may also be imported data);
- *Unfold* replaces a module by its variables, terms and generic conditions.

First modules: stationary heat conduction, linear elasticity, thermal
expansion.

## Names and scopes

- Top level parameters, materials and fields are visible everywhere; local
  names in their model only; x, y, z everywhere.
- No shadowing: a local name equal to a global one is a validation error.
- No qualified names (in GWFL the dot is the contraction): a model declares
  what it takes from another one, under an ordinary GWFL name:
  `"imports": [{"name": "T", "model": "thermal", "variable": "T"}]`.
  For now, an imported variable is defined on the same mesh.
- An import must be satisfied by a global solver that solves the source
  model first. Renaming updates expressions and imports.

## Solvers

Local (`model.solver`): `auto` (linear if all the terms are, Newton
otherwise), `newton` (`max_iter`, `max_res`, line search), `linear`
(linear solver), later `transient` (scheme, `dt`, `t_end`).

Global (`solvers`), nestable: `sequence` (`steps`), `staggered` (`steps`,
`max_iter`, `tolerance`, `relaxation`), `sweep` (`parameter`, `values`,
`steps`), `time_loop` (`dt`, `t_end`, `steps`). Without a global solver,
the models are solved in the order of their imports.

## Outputs and graphics

Outputs: `integral` (`expr`, `selection`), `extremum`, `export` (fields to
VTU), and globally `history` (one value per iteration, time step or
parameter value). Graphics store the display settings of the interface
(field, component, isolines, bands, colormap, range, arrows, cut line) and
curves of histories.

## Interface

The tree follows the file. Each header (Parameters, Materials, Fields,
Models, a model, Modules, Terms...) adds its items with a right click
(context menu) and with a + button of the panel. A model is added empty or
from a module.

## Implementation steps

1. Format and validation (`model.py`), examples rewritten.
2. Generation per model (`codegen.py`): variables, shared `MeshFem`, terms,
   conditions, heat and elasticity modules; the physics tests give the same
   results.
3. Several models: imports, `sequence` and `staggered`, thermal expansion;
   monolithic and staggered thermo-elastic plates compared.
4. Tree and forms: new hierarchy, context menu, + button, unfolding.
5. Graphics saved in the file.
6. Then: transient solver, time loop, sweep, a library of terms.
