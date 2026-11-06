# Copyright (C) 2026 Yves Renard, Konstantinos Poulios.
#
# This file is a part of GetFEM
#
# GetFEM  is  free software;  you  can  redistribute  it  and/or modify it
# under  the  terms  of the  GNU  Lesser General Public License as published
# by  the  Free Software Foundation;  either version 3 of the License,  or
# (at your option) any later version along with the GCC Runtime Library
# Exception either version 3.1 or (at your option) any later version.
# This program  is  distributed  in  the  hope  that it will be useful,  but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or  FITNESS  FOR  A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License and GCC Runtime Library Exception for more details.
# You  should  have received a copy of the GNU Lesser General Public License
# along  with  this program. If not, see https://www.gnu.org/licenses/.

"""Running a model: meshing (cached) then solving, each in a separate process,
so that the interface neither freezes nor crashes with the solver. The solve
script writes results.json, which is the authoritative output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import codegen, gwfl, mesher, model as M, parameters as P


@dataclass
class RunResult:
    ok: bool
    workdir: Path
    script: Path | None
    results: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    mesh: mesher.MeshResult | None = None

    @property
    def error(self) -> str:
        """Most useful line of the error output, to be shown to the user."""
        return mesher.error_line(self.stderr)


@dataclass
class SolveJob:
    """A solve to run: `command` in `workdir`, then `finish` (see mesher.MeshJob)."""
    workdir: Path
    script: Path
    command: list[str]
    meshes: dict[str, mesher.MeshResult]

    @property
    def mesh(self) -> mesher.MeshResult | None:
        return next(iter(self.meshes.values()), None)

    @property
    def results_file(self) -> Path:
        return self.workdir / "results.json"

    def finish(self, returncode: int | None, stdout: str = "", stderr: str = "") -> RunResult:
        results = {}
        if self.results_file.exists():
            try:
                results = json.loads(self.results_file.read_text(encoding="utf-8"))
            except ValueError:
                pass
        ok = returncode == 0 and results.get("status") == "ok"
        return RunResult(ok, self.workdir, self.script, results, stdout, stderr, returncode, self.mesh)


def meshes_needed(model: dict) -> list[str]:
    """The meshes used by the variables of the models."""
    return codegen.used_meshes(model)


def prepare_solve(model: dict, workdir: str | Path, meshes: dict[str, mesher.MeshResult],
                  python: str | None = None) -> SolveJob:
    """Write the solve script for already built meshes ({name: MeshResult})."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "run_getfem.py"
    script.write_text(codegen.generate(model, {n: (r.path, r.info) for n, r in meshes.items()}),
                      encoding="utf-8")
    job = SolveJob(workdir, script, [python or sys.executable, script.name], meshes)
    job.results_file.unlink(missing_ok=True)
    return job


def run(model: dict, workdir: str | Path, python: str | None = None,
        timeout: float | None = None, check_gwfl: bool = True) -> RunResult:
    """Check the model file (including its GWFL expressions), prepare the
    meshes used by the models, generate the solve script, run it.

    `python` selects the interpreter that provides GetFEM
    (by default, the one running getfem_gui).
    """
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    errors = M.validate(model)
    if not errors and check_gwfl:
        errors = gwfl.check(model, python=python)
    if errors:
        raise ValueError("invalid model:\n  - " + "\n  - ".join(errors))

    meshes = {}
    for name in meshes_needed(model):
        mres = mesher.ensure_mesh(model, name, workdir, python=python)
        if not mres.ok:
            return RunResult(False, workdir, None, stderr=f"mesh {name!r}: {mres.error}", mesh=mres)
        meshes[name] = mres

    job = prepare_solve(model, workdir, meshes, python)
    try:
        proc = subprocess.run(job.command, cwd=workdir, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return RunResult(False, workdir, job.script, stdout=exc.stdout or "",
                         stderr=f"solve interrupted after {timeout} s", mesh=job.mesh)
    return job.finish(proc.returncode, proc.stdout, proc.stderr)


def export_script(model: dict) -> str:
    """A single standalone script, meshing then solving: it does not depend on
    getfem_gui nor on a working folder, and writes results.json and the VTU
    files in the folder where it runs."""
    errors = M.validate(model)
    if errors:
        raise ValueError("invalid model:\n  - " + "\n  - ".join(errors))
    resolved = P.resolved(M.with_defaults(model))
    parts = []
    for name in meshes_needed(resolved):
        mesh = M.find_mesh(resolved, name)
        text = mesher.mesh_script(resolved, mesh, mesh["name"], standalone=True)
        text = "\n".join(l for l in text.splitlines() if not l.startswith("#!"))
        mid = codegen.ident(name)
        parts.append(f"# Mesh {name!r}\n{text}\nm_{mid}, RG_{mid} = m, REGIONS\n")
    solving = codegen.generate(model, None, standalone=True)
    solving = "\n".join(l for l in solving.splitlines() if not l.startswith("#!")) + "\n"
    head = ("#!/usr/bin/env python3\n"
            f"# Standalone GetFEM script exported by getfem_gui: {model.get('name', '')!r}\n"
            "# Part 1 builds the meshes and their regions, part 2 solves; run it with a\n"
            "# Python that provides GetFEM. It writes results.json and the VTU files here.\n")
    return (head + "\n# " + "=" * 70 + "\n# Part 1: meshes\n" + "\n".join(parts)
            + "\n# " + "=" * 70 + "\n# Part 2: solve\n" + solving)
