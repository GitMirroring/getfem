.. include:: ../replaces.txt

.. highlight:: bash

.. _gui-intro:

Introduction and installation
=============================

What it is
----------

``getfem_gui`` is a pure |py| package. A model is a file in the JSON
format which follows the logic of |gf|: shared definitions (parameters,
materials, geometries, meshes, fields), then *models*, each of them a
``gf.Model`` with its variables, modules (predefined physics), weak form
terms and conditions, then global solvers chaining the models. The
interface edits this file; two steps, each in a separate process, compute
from it:

* the *meshing*: a generated script builds each mesh (with the |gf|
  mesher, |gmsh| or Netgen) and evaluates the named selections on it, which
  become |gf| regions;
* the *solve*: a generated |gf| script loads the meshes, builds the models,
  solves them and writes the results (integrated quantities, |vtu| files).

.. image:: images/window.png
   :width: 100%
   :alt: The window of the interface

Requirements
------------

* |py| 3 (tested with 3.13) and |np|;
* the ``getfem`` |py| module (see :doc:`../install/install_linux`).
  Some features need recent additions to the |gf| mesher: polygons, ellipses,
  extrusions and revolutions of 2D objects (an error message says so when
  the installed |gf| lacks them);
* for the window: PySide6, and PyVista with pyvistaqt for the 3D view
  (``pip install pyside6 pyvista pyvistaqt``). The command line does not
  need them;
* optional: |gmsh| (``pip install gmsh``), the default mesher of the
  constructive and CAD geometries, also used to read CAD files and to export
  geometries as STEP, IGES or BREP; Netgen (``pip install netgen-mesher``).

The package is neither compiled nor installed by the build of |gf|: it is
run from the ``gui`` directory of the sources.

Starting
--------

::

  cd gui
  # the getfem module must be importable, e.g. after a build:
  export PYTHONPATH=<prefix>/lib/python3.X/site-packages
  python -m getfem_gui gui                                  # the window
  python -m getfem_gui gui examples/plate_hole_tension.json # with a model
  python -m getfem_gui run examples/plate.json -w /tmp/plate  # without window

``--python /path/to/python`` selects the interpreter that runs the
computations (the one that provides |gf|, |gmsh| and Netgen), when it is not
the one of the interface. A start-up window shows the versions of the
interface and of |gf| (``--no-splash`` removes it).

The tests of the interface are in ``gui/tests`` (``python -m pytest tests``);
those of the window run without a screen.

Environment variables
---------------------

``GETFEM_GUI_DEFAULT_MESHER``
   mesher of the new meshes of the constructive and CAD geometries
   (default ``gmsh``; ``getfem`` without |gmsh|).
``GETFEM_GUI_MATERIALS``
   the file of your own material library (default
   ``~/.config/getfem_gui/materials.json``).
``GETFEM_GUI_NO_3D``
   replaces the 3D view by a placeholder (tests).
``GETFEM_GUI_OFFSCREEN_3D``
   draws the 3D view off screen (tests, screenshots).
