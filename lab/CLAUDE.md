# Notes for working on getfem_lab

What a new contributor, or an AI assistant starting a new session, needs to
know before changing the graphical interface. `README.md` gives the
overview, the command line and the role of each file ("Layout"); the user
documentation is in `doc/sphinx/source/lab/`; the model format is in
`doc/model_format_v3.md`; the directions kept for later in
`doc/roadmap.md`.

## Principles

- **The model is a plain JSON dictionary** (`model.py`, format v3). Every
  operation of the interface is a function of `editing.py` on that
  dictionary, without any GUI toolkit; the window (`app/window.py`) only
  calls them, through `change`, which keeps the undo history and validates.
  New features go first into `editing.py` / `model.py` with tests that need
  no window.
- **Meshing and solving run in separate processes** (`runner.py`,
  `app/jobs.py`): generated GetFEM scripts (`mesher.py`, `codegen.py`) whose
  `results.json` is the authoritative output. The interface never imports
  GetFEM for a computation, so a crash of the solver cannot take it down.
  Meshes are cached by a hash of what they depend on.
- **Geometry without meshing**: `geometry2d.py` (exact 2D CSG: signed
  distances, boundary pieces, picked edges, regions and interfaces; curved
  polygons with fillets and splines as chains of cubic Bezier curves) and
  `geometry3d.py` (3D shapes as distance functions) serve the preview, the
  selections and the snapping. The view calls them many times at each
  change (a grid of about 58000 points for the fill of a 2D domain): keep
  them vectorized and avoid per-point Python loops (see the commit "faster
  view of polygons with splines" on `curved_polygon_sdf` for the kind of
  cost to avoid).
- **The 3D view** (`app/viewer.py`, `app/scene.py`, `app/qtplotter.py`) is
  a PyVista plotter on VTK's own Qt widget (not pyvistaqt): renders are
  deferred (`QTimer.singleShot`) and done only when visible; on macOS the
  widget paints only when something changed (Qt >= 6.10 otherwise loops
  paint -> render forever).
- **Readable generated scripts**: `codegen.py` and `export.py` write
  scripts that a user can read, modify and run without the interface.

## Working conventions (with the GetFEM maintainers)

- The repository, code, comments, commit messages and documentation are in
  English. (Yves Renard prefers to discuss in French.)
- Changes are delivered as `git format-patch` files, one topic per patch,
  committed one after the other and each checked on a clean clone of the
  repository with the previous patches applied. Changes of the GetFEM core
  (`src/`, `interface/`, e.g. the mesher objects used for curved polygons,
  extrusions, revolutions) go into separate patches from the GUI ones.
- Commit messages: a short title (`Lab: ...`, `mesher: ...`), then what was
  wrong or missing and what changed; ending with the trailer
  `Co-Authored-By: ...` when written with an assistant (no link to a
  session).
- The patch author runs the tests of what was changed; the maintainers run
  `make check` (all the test files, `make -j check` in parallel) on macOS
  and on Ubuntu, and send back the logs of the failures.

## Tests

```
cd lab
QT_QPA_PLATFORM=offscreen xvfb-run -a python3 -m pytest -q tests/test_splines.py
```

- `make check` runs each file of `LAB_TESTS` (`Makefile.am`) as one test
  through `tests/check.py`; a file whose requirements (pytest, PySide6,
  PyVista, Gmsh...) are missing is skipped. A new test file must be added
  to `LAB_TESTS`.
- Window tests set `GETFEM_LAB_NO_3D=""` and `GETFEM_LAB_OFFSCREEN_3D=1` to
  get an off-screen plotter (by default the tests run without a 3D view).
  Other variables: `GETFEM_LAB_NO_SPLASH`, `GETFEM_LAB_WAYLAND` (keep the
  Wayland platform instead of xcb), `GETFEM_LAB_GL_INFO` (print the OpenGL
  context), `GETFEM_LAB_DEFAULT_MESHER`, `GETFEM_LAB_MATERIALS`.
- To wait for a job in a test, sleep between calls of `processEvents()`
  (see `wait` in `tests/test_app.py`); never loop on it as fast as possible
  (see below).
- The documentation: `tests/test_doc.py` fails when the reference pages are
  out of date: regenerate them with `python -m getfem_lab.docref`; the
  screenshots with `xvfb-run -a python -m getfem_lab.docshots`.
- Gmsh and Netgen are optional: their meshing scripts must still be tested
  at least for compilation when they are not installed.

## Configurations to keep working

The interface runs on configurations that differ in ways that matter; most
failures seen so far came from one of these differences.

- **Python 3.10 to 3.13.** Before 3.12, `None` is not immortal: the
  PySide6 of Ubuntu's Python 3.10 loses a reference to `None` at each call
  of `processEvents()`, and millions of calls abort Python
  ("none_dealloc: deallocating None").
- **NumPy 1.x and 2.x.** No `np.trapezoid` in 1.x (`np.trapz`, removed in
  2.x: use `getattr(np, "trapezoid", None) or np.trapz`); `np.unique(...,
  axis=0, return_inverse=True)` gives a 1D or 2D inverse depending on the
  version: `.ravel()` it.
- **Old and recent PyVista.** Recent PyVista gives the mesh with its scalars
  to the mapper through a filter (`mapper.GetInput()` is then None: update
  the input algorithm, see `vector2d._input`), and deprecates `faces`,
  `n_faces_strict`...: read the cells through VTK (`vtkCellArray` offsets
  and connectivity, see `vector2d.cell_rows`).
- **VTK 9.x.** Its GL2PS exporter does not capture the primitives with the
  OpenGL 3 renderer (an image of the scene only): the vector export of the
  view is drawn by matplotlib (`app/vector2d.py`: 2D views in their plane,
  3D views by the painter's algorithm).
- **GetFEM with SuperLU or MUMPS.** MUMPS may solve a singular compatible
  system without noticing it (no "not unique" warning).
- **Qt and OpenGL.** macOS: Qt >= 6.10 and the paint loop above. Linux
  under Wayland: the xcb platform is chosen (`app.choose_platform`), else
  VTK's window opens apart. The OpenGL format asks a 24-bit depth buffer
  (`viewer.prepare_opengl`); GPUs differ a lot in polygon offset.

## Rendering notes

- Edges over faces: the faces are pushed back by a polygon offset
  (`scene.behind_lines`, relative factor 0.5); for solid meshes only the
  edges of the boundary are drawn.
- The edges of solid meshes are drawn by `scene.facing_wire` and
  `scene.cull_back_edges`: each segment once per boundary triangle along
  it, with the corners of that triangle as vertex attributes; the shader
  drops the copies whose triangle faces away (otherwise the edges behind a
  silhouette show through for a few pixels). The triangles are oriented on
  the closed boundary surface (`vtkPolyDataNormals`), not from
  `vtkOriginalCellIds`, which numbers the faces, not the cells, for
  quadratic elements.
- Quadratic elements are shown subdivided (VTK's nonlinear subdivision)
  with their edges sampled on their parabolas (`scene.surface`,
  `scene._wire`), at the level of the *Curved* control.
