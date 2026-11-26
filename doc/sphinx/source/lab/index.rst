.. include:: ../replaces.txt

.. _lab:

User documentation of GetFEM Lab
################################

GetFEM Lab, the graphical interface of |gf| (``getfem_lab``, in the ``lab``
directory of the sources), builds a finite element model without writing a program:
geometries, meshes, physics, solve and results are edited in a window, then
translated into a readable |gf| |py| script, which is run in a separate
process. Everything it does is also available from the command line, and
the script it writes can be run, read and modified without it.

The interface is under development: this documentation follows it, and
describes the features as they are. The reference part is generated from
the code of the interface.

.. toctree::
   :maxdepth: 2

   intro
   concepts
   window
   geometries
   selections
   meshes
   models
   results
   export
   examples
   reference/index
