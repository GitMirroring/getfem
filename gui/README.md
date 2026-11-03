# getfem_gui — le cœur sans interface

Socle d'une future interface graphique pour GetFEM. Il n'y a pas encore de
GUI : un modèle décrit en JSON est maillé puis traduit en un script Python
GetFEM lisible, chaque étape s'exécutant dans un processus séparé.

Ce répertoire est un paquet Python pur, indépendant du build de GetFEM : il
n'est ni compilé ni installé par `make` / CMake. Il demande Python 3 (testé avec 3.13),
NumPy et le module `getfem` ; Gmsh (`pip install gmsh`) est optionnel.

## Mise en route depuis le dépôt

```bash
cd gui
# le module getfem doit être importable, par exemple après un build CMake :
export PYTHONPATH=<prefixe>/lib/python3.X/site-packages
python -m getfem_gui run examples/plaque.json -w /tmp/plaque
python -m pytest tests/        # ou : python tests/test_heat.py
```

```
                 géométrie ──► sélections nommées (frontières, sous-domaines)
                     │
          ┌──────────┴──────────┐
     maillage A             maillage B          (structuré, mailleur GetFEM, Gmsh, import)
          │                      │               évalue les sélections -> régions GetFEM
          ▼                      ▼               cache : <travail>/maillages/<nom>.mesh/.json
   physique ── utilise un maillage, ne connaît que des noms de sélections
          │
          ▼
   run_getfem.py ──► results.json + .vtu
```

## Utilisation

```bash
python -m getfem_gui new modele.json                     # modèle de départ
python -m getfem_gui validate examples/plaque_trouee.json
python -m getfem_gui mesh examples/plaque_trouee.json    # tous les maillages, avec bilan des sélections
python -m getfem_gui run examples/plaque_trouee.json -w resultats/
python -m getfem_gui generate modele.json -o calcul.py   # script de calcul seul
python -m getfem_gui migrate ancien_v1.json -o nouveau.json
```

`--python /chemin/vers/python` choisit l'interpréteur qui possède GetFEM (et Gmsh).

Tests contre des solutions exactes : `python -m pytest tests/` ou `python tests/test_heat.py`.

## Organisation

| Fichier | Rôle |
|---|---|
| `getfem_gui/model.py` | format JSON (v2), sélections, validation, migration depuis v1 |
| `getfem_gui/mesher.py` | étape de maillage : script généré, régions, cache par hash |
| `getfem_gui/codegen.py` | script de calcul GetFEM (charge le maillage préparé) |
| `getfem_gui/runner.py` | enchaîne maillage et calcul en sous-processus |
| `getfem_gui/cli.py` | ligne de commande |
| `tests/test_heat.py` | 17 cas de validation (dont 1 qui demande Gmsh) |
| `examples/` | plaque chauffée (boîte), plaque trouée (CSG, deux maillages) |

## Géométrie, sélections, maillages

La description complète du format est dans la docstring de `model.py`.

**Sources de géométrie** : `box` (1D/2D/3D), `csg` (rectangle, disque/boule,
cylindre, réunion, intersection, différence), `cad` (STEP, IGES, BREP lus par
Gmsh/OpenCASCADE), `mesh_file` (un maillage `.msh` sert de géométrie).

**Mailleurs** et sources acceptées :

| Mailleur | box | csg | cad | mesh_file |
|---|---|---|---|---|
| `structured` (grille, simplexes ou quadrangles) | ✓ | | | |
| `getfem` (mailleur expérimental, `MesherObject`) | ✓ | ✓ | | |
| `gmsh` (OpenCASCADE) | ✓ | ✓ | ✓ | |
| `import` | | | | ✓ |

**Sélections** : `all` et `boundary` existent toujours ; `box` ajoute
`xmin … zmax`, un `.msh` apporte ses groupes physiques nommés. On définit les
autres par critères évalués *sur le maillage*, ce qui les rend indépendantes
du mailleur et robustes au remaillage : `plane`, `normal`, `in_box`, `in_ball`,
`physical`, `union`. Une sélection vide est une erreur explicite.

Attention : `normal` prend *toutes* les faces de bord orientées dans la
direction donnée, y compris celles d'un trou (le bord droit d'un trou a une
normale extérieure −x, comme le bord gauche de la pièce). Pour « la face
gauche », utiliser `plane` ; c'est ce que font les faces prédéfinies `xmin…zmax`.

Les numéros de régions GetFEM sont attribués par l'application, jamais par
l'utilisateur. Quand deux matériaux couvrent la même zone, le dernier listé
l'emporte : on peut donc écrire un matériau par défaut sur `all`, puis des
exceptions.

**Cache** : un maillage n'est refait que si sa géométrie, ses paramètres ou le
fichier source changent (hash SHA-256). Changer la physique ne remaille pas.

## État des mailleurs

- `structured`, `getfem` et `import` sont testés contre des solutions exactes
  (le mailleur `getfem` demande un GetFEM compilé avec Qhull).
- `gmsh` (y compris l'import CAO) n'a **pas encore été testé**. Le test
  `test_mailleur_gmsh_csg` s'exécute automatiquement là où `import gmsh` réussit.
- Unités CAO : Gmsh/OpenCASCADE conserve les unités du fichier STEP
  (souvent le mm) ; il faudra une option d'échelle.

## Ajouter une physique

Écrire deux fonctions dans `codegen.py` (termes GWFL, post-traitement), les
enregistrer dans `PHYSICS`, et ajouter les conditions aux limites autorisées
dans `model.validate`. L'élasticité linéaire est le candidat naturel suivant.

## Pistes ouvertes

- Vérifier la syntaxe des expressions GWFL dès la validation.
- Raffinement local Gmsh par sélection (`refinements`), noms de faces CAO.
- Plusieurs physiques sur des maillages différents dans un même modèle
  (GetFEM le permet) et études de convergence sur plusieurs maillages.
