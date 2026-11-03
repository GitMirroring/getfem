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

"""Command line interface.

    python -m getfem_gui new model.json
    python -m getfem_gui validate model.json
    python -m getfem_gui mesh model.json [-m fine] [-w dir]   # all meshes by default
    python -m getfem_gui generate model.json [-w dir] [-o solve.py]
    python -m getfem_gui run model.json [-w dir]
    python -m getfem_gui migrate old.json -o new.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import codegen, mesher, model as M, runner


def _workdir(args) -> Path:
    return Path(args.workdir or Path(args.model).with_suffix("").name + "_run")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="getfem_gui")
    sub = p.add_subparsers(dest="cmd", required=True)

    def cmd(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("model")
        return sp

    cmd("new", "create a starting model")
    cmd("validate", "check a model")
    pm = cmd("mesh", "build the meshes (cached)")
    pm.add_argument("-m", "--mesh", action="append", help="mesh name (repeatable)")
    pm.add_argument("--force", action="store_true", help="ignore the cache")
    pg = cmd("generate", "write the GetFEM solve script")
    pg.add_argument("-o", "--output", help="output file (default: standard output)")
    cmd("run", "mesh then solve")
    pmi = cmd("migrate", "convert a model to the current format")
    pmi.add_argument("-o", "--output", required=True)
    for sp in (pm, pg, sub.choices["run"]):
        sp.add_argument("-w", "--workdir", help="working directory (default: <model>_run/)")
        sp.add_argument("--python", help="Python interpreter that provides GetFEM")

    args = p.parse_args(argv)

    if args.cmd == "new":
        M.save(M.default_model(), args.model)
        print(f"model created: {args.model}")
        return 0

    try:
        model = M.load(args.model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.cmd == "migrate":
        M.save(model, args.output)
        print(f"model in format {M.FORMAT_VERSION} written to {args.output}")
        return 0

    errors = M.validate(model)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if errors:
        return 1
    if args.cmd == "validate":
        print("model is valid")
        return 0

    workdir = _workdir(args)

    if args.cmd == "mesh":
        names = args.mesh or [m["name"] for m in model["meshes"]]
        status = 0
        for name in names:
            res = mesher.ensure_mesh(model, name, workdir, python=args.python, force=args.force)
            if res.ok:
                i = res.info
                tag = " (cached)" if res.cached else ""
                print(f"{name}{tag}: {i['nb_elements']} elements, {i['nb_points']} nodes, "
                      f"dim {i['dim']} -> {res.path}")
                for sel, size in i["sizes"].items():
                    print(f"    {sel:<20} {i['kinds'][sel]:<9} {size}")
            else:
                status = 1
                print(f"{name}: meshing failed: {res.error}", file=sys.stderr)
        return status

    if args.cmd == "generate":
        name = M.physics_mesh(model)["name"]
        res = mesher.ensure_mesh(model, name, workdir, python=args.python)
        if not res.ok:
            print(f"meshing of {name!r} failed: {res.error}", file=sys.stderr)
            return 1
        script = codegen.generate(model, res.path, res.info)
        if args.output:
            Path(args.output).write_text(script, encoding="utf-8")
        else:
            sys.stdout.write(script)
        return 0

    res = runner.run(model, workdir, python=args.python)
    if not res.ok:
        print(f"failed: {res.error}", file=sys.stderr)
        if res.script:
            print(f"script: {res.script}", file=sys.stderr)
        return 1
    for warning in res.results.get("warnings", []):
        print(f"warning: {warning}", file=sys.stderr)
    print(json.dumps(res.results, indent=2, ensure_ascii=False))
    return 0
