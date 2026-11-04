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


"""Meshing and solving in separate processes, driven by QProcess so that the
window stays responsive, the output is shown as it comes, and a run can be
interrupted."""

from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from .. import mesher, model as M, runner


class JobRunner(QObject):
    output = Signal(str, bool)          # text, is_error_stream
    meshFinished = Signal(object)       # mesher.MeshResult
    solveFinished = Signal(object)      # runner.RunResult
    runningChanged = Signal(bool)

    def __init__(self, python: str | None = None, parent=None):
        super().__init__(parent)
        self.python = python
        self._proc: QProcess | None = None
        self._job = None
        self._stdout: list[str] = []
        self._stderr: list[str] = []
        self._then_solve = False
        self._model: dict | None = None
        self._workdir: Path | None = None
        self._stopped = False

    @property
    def model(self) -> dict | None:
        """The model of the last computation (a copy taken when it started)."""
        return self._model

    @property
    def running(self) -> bool:
        return self._proc is not None

    # -------------------------------------------------------------- public

    def build_mesh(self, model: dict, mesh_name: str, workdir: str | Path, force: bool = False) -> None:
        self._start_mesh(model, mesh_name, workdir, force, then_solve=False)

    def solve(self, model: dict, workdir: str | Path) -> None:
        """Mesh (if needed) then solve. The model must be valid."""
        mesh = M.physics_mesh(model)
        self._start_mesh(model, mesh["name"], workdir, False, then_solve=True)

    def stop(self) -> None:
        if self._proc is not None:
            self._stopped = True
            self._proc.kill()

    # -------------------------------------------------------------- internals

    def _start_mesh(self, model, mesh_name, workdir, force, then_solve):
        if self.running:
            raise RuntimeError("a computation is already running")
        self._model = copy.deepcopy(model)     # the user may go on editing
        self._workdir = Path(workdir)
        self._then_solve = then_solve
        try:
            job = mesher.prepare_mesh(self._model, mesh_name, workdir, self.python, force)
        except (ValueError, KeyError, OSError) as exc:
            self.meshFinished.emit(mesher.MeshResult(False, Path(), stderr=f"RuntimeError: {exc}"))
            return
        if isinstance(job, mesher.MeshResult):           # up to date
            self.output.emit(f"Mesh {mesh_name!r} is up to date.\n", False)
            self._mesh_done(job)
            return
        self.output.emit(f"Building mesh {mesh_name!r}...\n", False)
        self._launch(job, job.folder)

    def _mesh_done(self, mres):
        self.meshFinished.emit(mres)
        if not (self._then_solve and mres.ok):
            return
        try:
            job = runner.prepare_solve(self._model, self._workdir, mres, self.python)
        except (ValueError, KeyError, OSError) as exc:
            self.solveFinished.emit(runner.RunResult(False, self._workdir, None,
                                                     stderr=f"RuntimeError: {exc}", mesh=mres))
            return
        self.output.emit("Solving...\n", False)
        self._launch(job, job.workdir)

    def _launch(self, job, cwd):
        self._job = job
        self._stdout, self._stderr = [], []
        self._stopped = False
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(cwd))
        proc.readyReadStandardOutput.connect(self._read_stdout)
        proc.readyReadStandardError.connect(self._read_stderr)
        proc.finished.connect(self._finished)
        proc.errorOccurred.connect(self._error)
        self._proc = proc
        self.runningChanged.emit(True)
        proc.start(job.command[0], job.command[1:])

    def _read_stdout(self):
        text = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._stdout.append(text)
        self.output.emit(text, False)

    def _read_stderr(self):
        text = bytes(self._proc.readAllStandardError()).decode("utf-8", "replace")
        self._stderr.append(text)
        self.output.emit(text, True)

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._stderr.append(f"RuntimeError: cannot start {self._job.command[0]!r}")
            self._finished(-1, QProcess.ExitStatus.CrashExit)

    def _finished(self, code, status):
        if self._proc is None:
            return
        proc, job = self._proc, self._job
        self._proc = None
        proc.deleteLater()
        stderr = "".join(self._stderr)
        if self._stopped:
            stderr += "\nInterrupted: stopped by the user"
        elif status == QProcess.ExitStatus.CrashExit and code != -1:
            stderr += "\nRuntimeError: the process crashed"
        if status == QProcess.ExitStatus.CrashExit:
            code = code or -1
        self.runningChanged.emit(False)
        if isinstance(job, mesher.MeshJob):
            self._mesh_done(job.finish(code, stderr))
        else:
            self.solveFinished.emit(job.finish(code, "".join(self._stdout), stderr))
