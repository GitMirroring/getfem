# Copyright (C) 2026 Yves Renard.
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

from . import codegen, gwfl, mesher, model as M


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
    mesh: mesher.MeshResult

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


def prepare_solve(model: dict, workdir: str | Path, mesh: mesher.MeshResult,
                  python: str | None = None) -> SolveJob:
    """Write the solve script for an already built mesh."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "run_getfem.py"
    script.write_text(codegen.generate(model, mesh.path, mesh.info), encoding="utf-8")
    job = SolveJob(workdir, script, [python or sys.executable, script.name], mesh)
    job.results_file.unlink(missing_ok=True)
    return job


def run(model: dict, workdir: str | Path, python: str | None = None,
        timeout: float | None = None, check_gwfl: bool = True) -> RunResult:
    """Check the model (including its GWFL expressions), prepare the mesh used
    by the physics, generate the solve script, run it.

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

    mesh_name = M.physics_mesh(model)["name"]
    mres = mesher.ensure_mesh(model, mesh_name, workdir, python=python)
    if not mres.ok:
        return RunResult(False, workdir, None, stderr=f"mesh {mesh_name!r}: {mres.error}", mesh=mres)

    job = prepare_solve(model, workdir, mres, python)
    try:
        proc = subprocess.run(job.command, cwd=workdir, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return RunResult(False, workdir, job.script, stdout=exc.stdout or "",
                         stderr=f"solve interrupted after {timeout} s", mesh=mres)
    return job.finish(proc.returncode, proc.stdout, proc.stderr)
