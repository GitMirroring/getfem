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

"""Modèle de données de l'application, indépendant de toute interface graphique.

Un modèle est un dictionnaire Python sérialisé en JSON. Ce module fournit le
chargement (avec migration des anciennes versions), la sauvegarde, les tables
de sélections et la validation. La validation renvoie des messages lisibles
plutôt que de lever une exception, pour qu'une GUI puisse les afficher.

Format version 2
================

La géométrie et le maillage sont séparés. Une géométrie porte des
*sélections nommées* ; chaque maillage de cette géométrie en hérite. La
physique ne connaît que des noms de sélections et le nom du maillage utilisé.

{
  "format_version": 2,
  "name": "Plaque trouée",
  "geometries": [
    {
      "name": "plaque",
      "source": {...},                      # voir GEOMETRY_SOURCES
      "selections": {
        "trou":  {"kind": "boundary", "criteria": {"type": "in_ball", "center": [0.5, 0.5], "radius": 0.21}},
        "parois": {"kind": "boundary", "criteria": {"type": "union", "of": ["ymin", "ymax"]}}
      }
    }
  ],
  "meshes": [
    {"name": "grossier", "geometry": "plaque", "mesher": "getfem", "h": 0.1},
    {"name": "fin",      "geometry": "plaque", "mesher": "gmsh",   "size": 0.02, "order": 2}
  ],
  "discretization": {"degree": 2},
  "materials": [{"name": "acier", "selection": "all", "properties": {"k": 50}}],
  "physics": {
    "type": "heat_stationary", "mesh": "fin", "variable": "T",
    "sources": [{"selection": "all", "value": 0}],
    "boundary_conditions": [
      {"type": "temperature", "selection": "xmin", "value": 20},
      {"type": "heat_flux",   "selection": "trou", "value": 100},
      {"type": "convection",  "selection": "xmax", "h": 10, "T_ext": 20}
    ]
  },
  "study": {"type": "stationary"},
  "outputs": {"vtu": "resultat.vtu",
              "quantities": [{"name": "flux", "expr": "-k*Grad_T.Normal", "selection": "xmin"}]}
}

Sources de géométrie
--------------------
- box       : {"type": "box", "origin": [...], "size": [...]}         (1D, 2D, 3D)
- csg       : {"type": "csg", "dim": 2, "shape": <forme>}            primitives + booléens
              forme = {"type": "rectangle", "min": [...], "max": [...]}
                    | {"type": "ball", "center": [...], "radius": r}
                    | {"type": "cylinder", "origin": [...], "axis": [...], "length": l, "radius": r}  (3D)
                    | {"type": "union" | "intersection" | "difference", "of": [<forme>, ...]}
- cad       : {"type": "cad", "file": "piece.step"}                  STEP, IGES, BREP (via Gmsh)
- mesh_file : {"type": "mesh_file", "file": "piece.msh"}             la géométrie est le maillage

Mailleurs et sources compatibles
--------------------------------
- structured : box               {"divisions": [...], "element": "simplex" | "quad"}
- getfem     : box, csg          {"h": taille, "order": 1}   (mailleur expérimental de GetFEM)
- gmsh       : box, csg, cad     {"size": taille, "order": 1}
- import     : mesh_file         (aucun paramètre)

Sélections
----------
Toujours disponibles : "all" (tout le domaine) et "boundary" (tout le bord).
box ajoute xmin, xmax, ymin, ... ; mesh_file ajoute ses groupes physiques nommés.
Sélections utilisateur : {"kind": "domain" | "boundary", "criteria": {...}} avec
- {"type": "plane", "point": [...], "normal": [...]}   faces de bord contenues dans un plan
- {"type": "normal", "direction": [...], "angle": 1}   faces de bord de normale proche (degrés) ;
                                                         attention, inclut les bords intérieurs (trous)
- {"type": "in_box", "min": [...], "max": [...]}        faces de bord / éléments dans la boîte
- {"type": "in_ball", "center": [...], "radius": r}     faces de bord dans la boule
- {"type": "physical", "tag": n}                         groupe physique d'un maillage importé
- {"type": "union", "of": ["nom1", "nom2", ...]}         réunion de sélections de même nature
Les critères sont évalués sur le maillage : ils survivent au remaillage.

Les valeurs numériques peuvent être des nombres ou des expressions GWFL.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

FORMAT_VERSION = 2

ALL = "all"
BOUNDARY = "boundary"
DOMAIN = "domain"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AXES = "xyz"

GEOMETRY_SOURCES = ("box", "csg", "cad", "mesh_file")
MESHERS = {
    "structured": ("box",),
    "getfem": ("box", "csg"),
    "gmsh": ("box", "csg", "cad"),
    "import": ("mesh_file",),
}
CRITERIA = {
    "plane": (BOUNDARY,),
    "normal": (BOUNDARY,),
    "in_box": (BOUNDARY, DOMAIN),
    "in_ball": (BOUNDARY,),
    "physical": (BOUNDARY, DOMAIN),
    "union": (BOUNDARY, DOMAIN),
}
SHAPES_2D_3D = ("rectangle", "ball")
SHAPES_BOOLEAN = ("union", "intersection", "difference")


# ------------------------------------------------------------------ utilitaires

def box_boundary_names(dim: int) -> list[str]:
    """Noms des faces d'une boîte : xmin, xmax, ymin, ..."""
    return [f"{_AXES[i]}{side}" for i in range(dim) for side in ("min", "max")]


def default_model() -> dict:
    """Modèle de départ : carré unité, conduction, bords gauche/droit fixés."""
    return {
        "format_version": FORMAT_VERSION,
        "name": "Nouveau modèle",
        "geometries": [{
            "name": "geometrie",
            "source": {"type": "box", "origin": [0.0, 0.0], "size": [1.0, 1.0]},
            "selections": {},
        }],
        "meshes": [{
            "name": "maillage", "geometry": "geometrie", "mesher": "structured",
            "divisions": [10, 10], "element": "simplex",
        }],
        "discretization": {"degree": 2},
        "materials": [{"name": "materiau", "selection": ALL, "properties": {"k": 1.0}}],
        "physics": {
            "type": "heat_stationary",
            "mesh": "maillage",
            "variable": "T",
            "sources": [],
            "boundary_conditions": [
                {"type": "temperature", "selection": "xmin", "value": 0.0},
                {"type": "temperature", "selection": "xmax", "value": 1.0},
            ],
        },
        "study": {"type": "stationary"},
        "outputs": {"vtu": "resultat.vtu", "quantities": []},
    }


def load(path: str | Path) -> dict:
    """Charge un modèle ; un fichier d'une ancienne version est migré en mémoire."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        model = json.load(f)
    model = migrate(model)
    model["_source_dir"] = str(path.resolve().parent)
    return model


_TOP_ORDER = ("format_version", "name", "geometries", "meshes", "discretization",
              "materials", "physics", "study", "outputs")


def save(model: dict, path: str | Path) -> None:
    keys = [k for k in _TOP_ORDER if k in model] + [k for k in model if k not in _TOP_ORDER]
    data = {k: model[k] for k in keys if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_file(model: dict, file: str) -> Path:
    p = Path(file)
    if not p.is_absolute():
        p = Path(model.get("_source_dir", ".")) / p
    return p.resolve()


def read_gmsh_physical_names(path: str | Path) -> dict[str, tuple[int, int]]:
    """Section $PhysicalNames d'un .msh ASCII (formats 2 et 4) : {nom: (dim, tag)}."""
    names: dict[str, tuple[int, int]] = {}
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as f:
            lines = iter(f)
            for line in lines:
                if line.strip() == "$PhysicalNames":
                    for _ in range(int(next(lines))):
                        parts = next(lines).split(maxsplit=2)
                        names[parts[2].strip().strip('"')] = (int(parts[0]), int(parts[1]))
                    break
    except (OSError, ValueError, StopIteration, IndexError):
        return {}
    return names


# ------------------------------------------------------------------ accès

def by_name(items: list[dict], name: str) -> dict | None:
    return next((it for it in items if it.get("name") == name), None)


def physics_mesh(model: dict) -> dict | None:
    """Maillage utilisé par la physique (le premier si non précisé)."""
    meshes = model.get("meshes", [])
    name = model.get("physics", {}).get("mesh")
    if name is None:
        return meshes[0] if meshes else None
    return by_name(meshes, name)


def geometry_of(model: dict, mesh: dict) -> dict | None:
    return by_name(model.get("geometries", []), mesh.get("geometry"))


def geometry_dim(model: dict, geom: dict) -> int | None:
    src = geom.get("source", {})
    if src.get("type") == "box":
        return len(src.get("size", []))
    if src.get("type") in ("csg", "cad"):
        return src.get("dim", 3 if src.get("type") == "cad" else None)
    if src.get("type") == "mesh_file" and "file" in src:
        dims = [d for d, _ in read_gmsh_physical_names(resolve_file(model, src["file"])).values()]
        return max(dims) if dims else None
    return None


def selection_table(model: dict, geom: dict) -> dict[str, dict]:
    """Toutes les sélections d'une géométrie : {nom: {"kind": ..., "builtin": bool, ...}}.

    L'ordre est significatif : il fixe la numérotation des régions GetFEM.
    """
    table: dict[str, dict] = {
        ALL: {"kind": DOMAIN, "builtin": True},
        BOUNDARY: {"kind": BOUNDARY, "builtin": True},
    }
    src = geom.get("source", {})
    if src.get("type") == "box":
        dim = len(src.get("size", []))
        origin, size = src.get("origin", []), src.get("size", [])
        for i, name in enumerate(box_boundary_names(dim)):
            axis, sign = divmod(i, 2)
            normal = [0.0] * dim
            normal[axis] = 1.0
            point = list(origin)
            if sign and axis < len(point) and axis < len(size):
                point[axis] = origin[axis] + size[axis]
            table[name] = {"kind": BOUNDARY, "builtin": True,
                           "criteria": {"type": "plane", "point": point, "normal": normal}}
    elif src.get("type") == "mesh_file" and "file" in src:
        names = read_gmsh_physical_names(resolve_file(model, src["file"]))
        top = max((d for d, _ in names.values()), default=0)
        for name, (d, tag) in names.items():
            table[name] = {"kind": DOMAIN if d == top else BOUNDARY, "builtin": True,
                           "criteria": {"type": "physical", "tag": tag}}
    for name, sel in geom.get("selections", {}).items():
        table[name] = {"kind": sel.get("kind"), "builtin": False, "criteria": sel.get("criteria", {})}
    return table


# ------------------------------------------------------------------ validation

def _check_value(v, where: str, errors: list[str]) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        errors.append(f"{where} : valeur attendue (nombre ou expression GWFL), reçu {v!r}")
    elif isinstance(v, str) and not v.strip():
        errors.append(f"{where} : expression vide")


def _check_vec(v, n, where, errors, positive=False):
    if not isinstance(v, list) or (n is not None and len(v) != n) \
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v):
        errors.append(f"{where} : {n or ''} nombres attendus".replace("  ", " "))
        return False
    if positive and any(x <= 0 for x in v):
        errors.append(f"{where} : valeurs strictement positives attendues")
    return True


def _check_positive(v, where, errors):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        errors.append(f"{where} : nombre strictement positif attendu")


def _check_shape(shape, dim, where, errors):
    t = shape.get("type") if isinstance(shape, dict) else None
    if t == "rectangle":
        _check_vec(shape.get("min"), dim, f"{where}.min", errors)
        _check_vec(shape.get("max"), dim, f"{where}.max", errors)
    elif t == "ball":
        _check_vec(shape.get("center"), dim, f"{where}.center", errors)
        _check_positive(shape.get("radius"), f"{where}.radius", errors)
    elif t == "cylinder":
        if dim != 3:
            errors.append(f"{where} : cylinder n'existe qu'en 3D")
        _check_vec(shape.get("origin"), dim, f"{where}.origin", errors)
        _check_vec(shape.get("axis"), dim, f"{where}.axis", errors)
        _check_positive(shape.get("length"), f"{where}.length", errors)
        _check_positive(shape.get("radius"), f"{where}.radius", errors)
    elif t in SHAPES_BOOLEAN:
        parts = shape.get("of", [])
        if len(parts) < 2:
            errors.append(f"{where}.of : au moins deux formes attendues")
        for i, s in enumerate(parts):
            _check_shape(s, dim, f"{where}.of[{i}]", errors)
    else:
        errors.append(f"{where}.type : forme inconnue {t!r} "
                      "(rectangle, ball, cylinder, union, intersection, difference)")


def _validate_geometry(model, geom, where, errors):
    src = geom.get("source", {})
    t = src.get("type")
    if t == "box":
        dim = len(src.get("size", []))
        if dim not in (1, 2, 3):
            errors.append(f"{where}.source.size : 1, 2 ou 3 valeurs attendues")
        else:
            _check_vec(src.get("size"), dim, f"{where}.source.size", errors, positive=True)
            _check_vec(src.get("origin"), dim, f"{where}.source.origin", errors)
    elif t == "csg":
        dim = src.get("dim")
        if dim not in (2, 3):
            errors.append(f"{where}.source.dim : 2 ou 3 attendu")
        else:
            _check_shape(src.get("shape"), dim, f"{where}.source.shape", errors)
    elif t in ("cad", "mesh_file"):
        if "file" not in src:
            errors.append(f"{where}.source.file : chemin du fichier manquant")
        elif not resolve_file(model, src["file"]).exists():
            errors.append(f"{where}.source.file : fichier introuvable ({resolve_file(model, src['file'])})")
        elif t == "mesh_file":
            by_tag: dict[int, list[str]] = {}
            for name, (_, tag) in read_gmsh_physical_names(resolve_file(model, src["file"])).items():
                by_tag.setdefault(tag, []).append(name)
            for tag, group in by_tag.items():
                if len(group) > 1:
                    errors.append(f"{where}.source.file : les groupes physiques {', '.join(group)} "
                                  f"partagent le numéro {tag} ; GetFEM a besoin de numéros distincts")
        if t == "cad" and Path(src.get("file", "")).suffix.lower() not in (".step", ".stp", ".iges", ".igs", ".brep"):
            errors.append(f"{where}.source.file : formats CAO acceptés : STEP, IGES, BREP")
    else:
        errors.append(f"{where}.source.type : {', '.join(GEOMETRY_SOURCES)} attendu, reçu {t!r}")
        return

    table = selection_table(model, geom)
    dim = geometry_dim(model, geom)
    for name, sel in geom.get("selections", {}).items():
        w = f"{where}.selections.{name}"
        if not _IDENT.match(name):
            errors.append(f"{w} : nom invalide (lettres, chiffres, _)")
        if name in table and table[name]["builtin"]:
            errors.append(f"{w} : le nom {name!r} est déjà une sélection prédéfinie")
        kind = sel.get("kind")
        if kind not in (DOMAIN, BOUNDARY):
            errors.append(f"{w}.kind : 'domain' ou 'boundary' attendu")
            continue
        crit = sel.get("criteria", {})
        ct = crit.get("type")
        if ct not in CRITERIA:
            errors.append(f"{w}.criteria.type : {', '.join(CRITERIA)} attendu, reçu {ct!r}")
            continue
        if kind not in CRITERIA[ct]:
            errors.append(f"{w} : le critère {ct!r} ne s'applique qu'à {', '.join(CRITERIA[ct])}")
        if ct == "normal":
            _check_vec(crit.get("direction"), dim, f"{w}.criteria.direction", errors)
        elif ct == "plane":
            _check_vec(crit.get("point"), dim, f"{w}.criteria.point", errors)
            _check_vec(crit.get("normal"), dim, f"{w}.criteria.normal", errors)
        elif ct == "in_box":
            _check_vec(crit.get("min"), dim, f"{w}.criteria.min", errors)
            _check_vec(crit.get("max"), dim, f"{w}.criteria.max", errors)
        elif ct == "in_ball":
            _check_vec(crit.get("center"), dim, f"{w}.criteria.center", errors)
            _check_positive(crit.get("radius"), f"{w}.criteria.radius", errors)
        elif ct == "physical":
            if t != "mesh_file":
                errors.append(f"{w} : le critère 'physical' demande une géométrie mesh_file")
            if not isinstance(crit.get("tag"), int):
                errors.append(f"{w}.criteria.tag : entier attendu")
        elif ct == "union":
            for other in crit.get("of", []):
                if other not in table or other == name:
                    errors.append(f"{w}.criteria.of : sélection inconnue {other!r}")
                elif table[other]["kind"] != kind:
                    errors.append(f"{w}.criteria.of : {other!r} n'est pas de nature {kind!r}")
                elif table[other].get("criteria", {}).get("type") == "union":
                    errors.append(f"{w}.criteria.of : réunion de réunions non prise en charge ({other!r})")
            if not crit.get("of"):
                errors.append(f"{w}.criteria.of : liste de sélections attendue")


def _validate_mesh(model, mesh, where, errors):
    geom = geometry_of(model, mesh)
    if geom is None:
        errors.append(f"{where}.geometry : géométrie inconnue {mesh.get('geometry')!r}")
        return
    mesher = mesh.get("mesher")
    if mesher not in MESHERS:
        errors.append(f"{where}.mesher : {', '.join(MESHERS)} attendu, reçu {mesher!r}")
        return
    stype = geom.get("source", {}).get("type")
    if stype not in MESHERS[mesher]:
        errors.append(f"{where}.mesher : {mesher!r} ne sait pas mailler une géométrie {stype!r} "
                      f"(compatible : {', '.join(MESHERS[mesher])})")
    if mesher == "structured" and stype == "box":
        dim = len(geom["source"].get("size", []))
        div = mesh.get("divisions")
        if not isinstance(div, list) or len(div) != dim or any(
                not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in div):
            errors.append(f"{where}.divisions : {dim} entiers >= 1 attendus")
        if mesh.get("element", "simplex") not in ("simplex", "quad"):
            errors.append(f"{where}.element : 'simplex' ou 'quad' attendu")
    elif mesher == "getfem":
        _check_positive(mesh.get("h"), f"{where}.h", errors)
    elif mesher == "gmsh":
        _check_positive(mesh.get("size"), f"{where}.size", errors)
    if mesher in ("getfem", "gmsh") and mesh.get("order", 1) not in (1, 2, 3):
        errors.append(f"{where}.order : 1, 2 ou 3 attendu")


def validate(model: dict) -> list[str]:
    """Renvoie la liste des erreurs (vide si le modèle est valide)."""
    errors: list[str] = []
    if model.get("format_version") != FORMAT_VERSION:
        errors.append(f"format_version : {FORMAT_VERSION} attendu, reçu {model.get('format_version')!r}")

    # --- Géométries et maillages
    for key in ("geometries", "meshes"):
        items = model.get(key, [])
        if not items:
            errors.append(f"{key} : au moins un élément est nécessaire")
        names = [it.get("name") for it in items]
        for i, name in enumerate(names):
            if not isinstance(name, str) or not _IDENT.match(name):
                errors.append(f"{key}[{i}].name : identifiant attendu, reçu {name!r}")
            elif names.count(name) > 1:
                errors.append(f"{key}[{i}].name : nom {name!r} utilisé plusieurs fois")
    for i, geom in enumerate(model.get("geometries", [])):
        _validate_geometry(model, geom, f"geometries[{i}]", errors)
    for i, mesh in enumerate(model.get("meshes", [])):
        _validate_mesh(model, mesh, f"meshes[{i}]", errors)

    degree = model.get("discretization", {}).get("degree", 1)
    if not isinstance(degree, int) or isinstance(degree, bool) or not 1 <= degree <= 6:
        errors.append("discretization.degree : entier entre 1 et 6 attendu")

    phys = model.get("physics", {})
    mesh = physics_mesh(model)
    if mesh is None:
        errors.append(f"physics.mesh : maillage inconnu {phys.get('mesh')!r}")
        return errors
    geom = geometry_of(model, mesh)
    if geom is None:
        return errors
    table = selection_table(model, geom)

    def check_sel(name, kind, where):
        if name not in table:
            errors.append(f"{where} : sélection inconnue {name!r} (géométrie {geom['name']!r} : "
                          f"{', '.join(table)})")
        elif table[name]["kind"] != kind:
            what = "un domaine" if kind == DOMAIN else "une frontière"
            errors.append(f"{where} : {name!r} n'est pas {what}")

    # --- Matériaux
    materials = model.get("materials", [])
    if not materials:
        errors.append("materials : au moins un matériau est nécessaire")
    seen = set()
    for i, mat in enumerate(materials):
        where = f"materials[{i}]"
        name = mat.get("name", "")
        if not _IDENT.match(str(name)):
            errors.append(f"{where}.name : identifiant attendu, reçu {name!r}")
        if name in seen:
            errors.append(f"{where}.name : nom {name!r} déjà utilisé")
        seen.add(name)
        check_sel(mat.get("selection", ALL), DOMAIN, f"{where}.selection")
        props = mat.get("properties", {})
        if "k" not in props:
            errors.append(f"{where}.properties.k : conductivité manquante")
        for prop, v in props.items():
            if not _IDENT.match(prop):
                errors.append(f"{where}.properties : nom invalide {prop!r}")
            _check_value(v, f"{where}.properties.{prop}", errors)

    # --- Physique
    if phys.get("type") != "heat_stationary":
        errors.append(f"physics.type : seul 'heat_stationary' est disponible, reçu {phys.get('type')!r}")
    var = phys.get("variable", "T")
    if not _IDENT.match(str(var)):
        errors.append(f"physics.variable : identifiant attendu, reçu {var!r}")
    for i, src in enumerate(phys.get("sources", [])):
        check_sel(src.get("selection", ALL), DOMAIN, f"physics.sources[{i}].selection")
        _check_value(src.get("value"), f"physics.sources[{i}].value", errors)

    required = {"temperature": ["value"], "heat_flux": ["value"], "convection": ["h", "T_ext"]}
    has_dirichlet = False
    for i, bc in enumerate(phys.get("boundary_conditions", [])):
        where = f"physics.boundary_conditions[{i}]"
        btype = bc.get("type")
        if btype not in required:
            errors.append(f"{where}.type : {', '.join(required)} attendu, reçu {btype!r}")
            continue
        check_sel(bc.get("selection"), BOUNDARY, f"{where}.selection")
        for key in required[btype]:
            _check_value(bc.get(key), f"{where}.{key}", errors)
        has_dirichlet |= btype in ("temperature", "convection")
    if not has_dirichlet and not errors:
        errors.append("physics.boundary_conditions : sans température imposée ni convection, "
                      "la température n'est définie qu'à une constante près")

    # --- Sorties
    for i, q in enumerate(model.get("outputs", {}).get("quantities", [])):
        where = f"outputs.quantities[{i}]"
        if not _IDENT.match(str(q.get("name", ""))):
            errors.append(f"{where}.name : identifiant attendu")
        _check_value(q.get("expr"), f"{where}.expr", errors)
        sel = q.get("selection", ALL)
        if sel not in table:
            errors.append(f"{where}.selection : sélection inconnue {sel!r}")

    return errors


def with_defaults(model: dict) -> dict:
    """Copie du modèle complétée par les valeurs par défaut optionnelles."""
    m = copy.deepcopy(model)
    m.setdefault("discretization", {}).setdefault("degree", 1)
    for geom in m.get("geometries", []):
        geom.setdefault("selections", {})
    for mesh in m.get("meshes", []):
        if mesh.get("mesher") == "structured":
            mesh.setdefault("element", "simplex")
        if mesh.get("mesher") in ("getfem", "gmsh"):
            mesh.setdefault("order", 1)
    phys = m.setdefault("physics", {})
    phys.setdefault("mesh", m["meshes"][0]["name"] if m.get("meshes") else None)
    phys.setdefault("variable", "T")
    phys.setdefault("sources", [])
    phys.setdefault("boundary_conditions", [])
    for mat in m.get("materials", []):
        mat.setdefault("selection", ALL)
    for src in phys["sources"]:
        src.setdefault("selection", ALL)
    out = m.setdefault("outputs", {})
    out.setdefault("vtu", "resultat.vtu")
    out.setdefault("quantities", [])
    for q in out["quantities"]:
        q.setdefault("selection", ALL)
    return m


# ------------------------------------------------------------------ migration

def migrate(model: dict) -> dict:
    """Met un modèle d'une version antérieure au format courant (copie)."""
    version = model.get("format_version")
    if version == FORMAT_VERSION:
        return model
    if version == 1:
        return _migrate_v1(model)
    raise ValueError(f"format_version {version!r} inconnu")


def _migrate_v1(old: dict) -> dict:
    m = copy.deepcopy(old)
    mesh = m.pop("mesh", {})
    extra_regions = m.pop("regions", {})

    if mesh.get("type") == "box":
        geom = {"name": "geometrie", "source": {"type": "box", "origin": mesh.get("origin"),
                                                 "size": mesh.get("size")}, "selections": {}}
        new_mesh = {"name": "maillage", "geometry": "geometrie", "mesher": "structured",
                    "divisions": mesh.get("divisions"), "element": mesh.get("element", "simplex")}
    else:
        geom = {"name": "geometrie", "source": {"type": "mesh_file", "file": mesh.get("file")},
                "selections": {}}
        new_mesh = {"name": "maillage", "geometry": "geometrie", "mesher": "import"}

    # region -> selection partout
    def rename(items):
        for it in items:
            if "region" in it:
                renamed = {("selection" if k == "region" else k): v for k, v in it.items()}
                it.clear()
                it.update(renamed)
    phys = m.setdefault("physics", {})
    rename(m.get("materials", []))
    rename(phys.get("sources", []))
    rename(phys.get("boundary_conditions", []))
    rename(m.get("outputs", {}).get("quantities", []))

    # Les régions déclarées à la main deviennent des sélections par numéro.
    domain_uses = {it.get("selection") for it in m.get("materials", []) + phys.get("sources", [])}
    for name, tag in extra_regions.items():
        kind = DOMAIN if name in domain_uses else BOUNDARY
        geom["selections"][name] = {"kind": kind, "criteria": {"type": "physical", "tag": int(tag)}}

    m["geometries"] = [geom]
    m["meshes"] = [new_mesh]
    phys["mesh"] = "maillage"
    m["format_version"] = FORMAT_VERSION
    return m
