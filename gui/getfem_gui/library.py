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


"""Material libraries: the built-in one (data/materials.json, read only) and
the user's one (a JSON file of the same format, by default
~/.config/getfem_gui/materials.json, or $GETFEM_GUI_MATERIALS).

A material taken from a library is *copied* into the model file, with its
origin ("from": "library:Copper"): the file stays self-contained and gives
the same results if a library changes. Only the properties that the modules
use reach the solve script.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from . import model as M

BUILTIN = Path(__file__).resolve().parent / "data" / "materials.json"
LIBRARY, USER = "library", "user"


def user_file() -> Path:
    env = os.environ.get("GETFEM_GUI_MATERIALS")
    return Path(env) if env else Path.home() / ".config" / "getfem_gui" / "materials.json"


def _read(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [m for m in data.get("materials", []) if isinstance(m, dict) and m.get("name")]


def materials(source: str | None = None) -> list[tuple[str, dict]]:
    """(source, material) of the libraries: the built-in one, then the user's."""
    out = []
    if source in (None, LIBRARY):
        out += [(LIBRARY, m) for m in _read(BUILTIN)]
    if source in (None, USER):
        out += [(USER, m) for m in _read(user_file())]
    return out


def find(origin: str) -> dict | None:
    """The material of an origin "library:Copper" or "user:My steel"."""
    source, _, name = origin.partition(":")
    return next((m for s, m in materials(source) if m.get("name") == name), None)


def identifier(name: str) -> str:
    """A name usable in the model: 'Stainless steel 304' -> 'stainless_steel_304'."""
    ident = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_").lower() or "material"
    return ident if not ident[0].isdigit() else f"m_{ident}"


def copy_into(model: dict, origin: str) -> int:
    """Copy a material of a library into the model; returns its index."""
    mat = find(origin)
    if mat is None:
        raise ValueError(f"no material {origin!r} in the libraries")
    mats = model.setdefault("materials", [])
    taken = {m.get("name") for m in mats}
    base = identifier(mat["name"])
    name, i = base, 2
    while name in taken:
        name, i = f"{base}{i}", i + 1
    mats.append({"name": name, "description": mat["name"], "from": origin,
                 "properties": copy.deepcopy(mat.get("properties", {}))})
    return len(mats) - 1


def save_to_user(material: dict) -> Path:
    """Add (or replace) a material of the model in the user's library."""
    path = user_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {"description": "materials of the user (getfem_gui)", "materials": []}
    name = material.get("description") or material.get("name")
    entry = {"name": name, "category": "user", "properties": copy.deepcopy(material.get("properties", {}))}
    data["materials"] = [m for m in data.get("materials", []) if m.get("name") != name] + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def used_properties(model: dict) -> list[str]:
    """The properties asked by the modules of the file (the columns of the
    table of materials), in the order of the catalogue."""
    used = {p for md in model.get("models", []) for mod in md.get("modules", [])
            for p in M.MODULE_SPECS.get(mod.get("type"), {}).get("properties", {})}
    return [p for p in M.PROPERTIES if p in used] + sorted(used - set(M.PROPERTIES))
