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

from . import codegen, mesher, model as M


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


def run(model: dict, workdir: str | Path, python: str | None = None,
        timeout: float | None = None) -> RunResult:
    """Prepare the mesh used by the physics, generate the solve script, run it.

    `python` selects the interpreter that provides GetFEM
    (by default, the one running getfem_gui).
    """
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    errors = M.validate(model)
    if errors:
        raise ValueError("invalid model:\n  - " + "\n  - ".join(errors))

    mesh_name = M.physics_mesh(model)["name"]
    mres = mesher.ensure_mesh(model, mesh_name, workdir, python=python)
    if not mres.ok:
        return RunResult(False, workdir, None, stderr=f"mesh {mesh_name!r}: {mres.error}", mesh=mres)

    script = workdir / "run_getfem.py"
    script.write_text(codegen.generate(model, mres.path, mres.info), encoding="utf-8")
    results_file = workdir / "results.json"
    results_file.unlink(missing_ok=True)
    try:
        proc = subprocess.run([python or sys.executable, script.name], cwd=workdir,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return RunResult(False, workdir, script, stdout=exc.stdout or "",
                         stderr=f"solve interrupted after {timeout} s", mesh=mres)

    results = {}
    if results_file.exists():
        results = json.loads(results_file.read_text(encoding="utf-8"))
    ok = proc.returncode == 0 and results.get("status") == "ok"
    return RunResult(ok, workdir, script, results, proc.stdout, proc.stderr, proc.returncode, mres)
