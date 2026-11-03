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

"""Ligne de commande.

    python -m getfem_gui new modele.json
    python -m getfem_gui validate modele.json
    python -m getfem_gui mesh modele.json [-m fin] [-w dossier]   # tous les maillages par défaut
    python -m getfem_gui generate modele.json [-w dossier] [-o calcul.py]
    python -m getfem_gui run modele.json [-w dossier]
    python -m getfem_gui migrate ancien.json -o nouveau.json
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

    cmd("new", "créer un modèle de départ")
    cmd("validate", "vérifier un modèle")
    pm = cmd("mesh", "construire les maillages (avec cache)")
    pm.add_argument("-m", "--mesh", action="append", help="nom du maillage (répétable)")
    pm.add_argument("--force", action="store_true", help="ignorer le cache")
    pg = cmd("generate", "écrire le script de calcul GetFEM")
    pg.add_argument("-o", "--output", help="fichier de sortie (défaut : sortie standard)")
    cmd("run", "mailler puis calculer")
    pmi = cmd("migrate", "convertir un modèle au format courant")
    pmi.add_argument("-o", "--output", required=True)
    for sp in (pm, pg, sub.choices["run"]):
        sp.add_argument("-w", "--workdir", help="dossier de travail (défaut : <modèle>_run/)")
        sp.add_argument("--python", help="interpréteur Python qui possède GetFEM")

    args = p.parse_args(argv)

    if args.cmd == "new":
        M.save(M.default_model(), args.model)
        print(f"modèle créé : {args.model}")
        return 0

    try:
        model = M.load(args.model)
    except ValueError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 1

    if args.cmd == "migrate":
        M.save(model, args.output)
        print(f"modèle au format {M.FORMAT_VERSION} écrit dans {args.output}")
        return 0

    errors = M.validate(model)
    for e in errors:
        print(f"erreur : {e}", file=sys.stderr)
    if errors:
        return 1
    if args.cmd == "validate":
        print("modèle valide")
        return 0

    workdir = _workdir(args)

    if args.cmd == "mesh":
        names = args.mesh or [m["name"] for m in model["meshes"]]
        status = 0
        for name in names:
            res = mesher.ensure_mesh(model, name, workdir, python=args.python, force=args.force)
            if res.ok:
                i = res.info
                tag = " (cache)" if res.cached else ""
                print(f"{name}{tag} : {i['nb_elements']} éléments, {i['nb_points']} nœuds, "
                      f"dim {i['dim']} -> {res.path}")
                for sel, size in i["sizes"].items():
                    print(f"    {sel:<20} {i['kinds'][sel]:<9} {size}")
            else:
                status = 1
                print(f"{name} : échec du maillage : {res.error}", file=sys.stderr)
        return status

    if args.cmd == "generate":
        name = M.physics_mesh(model)["name"]
        res = mesher.ensure_mesh(model, name, workdir, python=args.python)
        if not res.ok:
            print(f"échec du maillage {name!r} : {res.error}", file=sys.stderr)
            return 1
        script = codegen.generate(model, res.path, res.info)
        if args.output:
            Path(args.output).write_text(script, encoding="utf-8")
        else:
            sys.stdout.write(script)
        return 0

    res = runner.run(model, workdir, python=args.python)
    if not res.ok:
        print(f"échec : {res.error}", file=sys.stderr)
        if res.script:
            print(f"script : {res.script}", file=sys.stderr)
        return 1
    print(json.dumps(res.results, indent=2, ensure_ascii=False))
    return 0
