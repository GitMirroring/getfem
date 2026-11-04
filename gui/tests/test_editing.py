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

"""Model editing (toolkit independent): tree, fields, consistent changes.

    python -m pytest tests/       (or: python tests/test_editing.py)
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from getfem_gui import editing as E, model as M  # noqa: E402

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.json"))


def load(name):
    return M.load(Path(__file__).resolve().parents[1] / "examples" / name)


def test_tree_and_fields_of_every_node():
    for ex in EXAMPLES:
        model = M.load(ex)
        root = E.tree(model, result_fields=["T"])
        kinds = {n.kind for n in root.walk()}
        assert {"geometry", "mesh", "material", "physics", "bc"} <= kinds, ex.name
        for node in root.walk():
            for f in E.fields(model, node.path):
                assert f.kind in ("text", "ident", "value", "vector", "numbers", "ints", "int",
                                  "number", "choice", "names", "file", "info"), (node.path, f)
                if f.kind == "choice" and f.op is None:
                    value = E.get_at(model, f.key)
                    assert value is None or value in [c[0] for c in f.choices], (node.path, f, value)


def test_every_action_keeps_a_valid_model():
    """Adding items with their default values must not introduce errors."""
    for ex in EXAMPLES:
        base = M.load(ex)
        assert M.validate(base) == [], ex.name
        for node in E.tree(base).walk():
            for action, _ in E.actions(base, node.path):
                if action == "remove":
                    continue
                model = copy.deepcopy(base)
                new = E.do_action(model, node.path, action, name="rho" if E.needs_name(action) else None)
                assert M.validate(model) == [], (ex.name, node.path, action, M.validate(model))
                assert new is None or E.find_node(E.tree(model), new) is not None, (action, new)


def test_rename_selection_updates_its_uses():
    model = load("plate_with_hole.json")
    geom = next(i for i, g in enumerate(model["geometries"]) if "hole" in g["selections"])
    path = ("geometries", geom, "selections", "hole")
    f = next(f for f in E.fields(model, path) if f.op == "rename_selection")
    order = list(model["geometries"][geom]["selections"])
    new = E.apply(model, f, "circle")
    assert new == ("geometries", geom, "selections", "circle")
    assert list(model["geometries"][geom]["selections"]) == [("circle" if n == "hole" else n) for n in order]
    uses = [bc["selection"] for bc in model["physics"]["boundary_conditions"]]
    assert "circle" in uses and "hole" not in uses
    assert M.validate(model) == []
    try:
        E.apply(model, E.Field(new, f.label, f.kind, op=f.op), "left")
        raise AssertionError("duplicate name accepted")
    except ValueError:
        pass


def test_rename_mesh_and_geometry():
    model = load("plate_with_hole.json")
    E.apply(model, E.Field(("meshes", 0, "name"), "", "ident", op="rename_mesh"), "m1")
    E.apply(model, E.Field(("geometries", 0, "name"), "", "ident", op="rename_geometry"), "g1")
    assert M.validate(model) == []
    assert model["meshes"][0]["geometry"] == "g1"


def test_type_changes_reset_fields():
    model = load("plate.json")
    bcs = ("physics", "boundary_conditions")
    E.apply(model, E.Field(bcs + (0, "type"), "", "choice", op="bc_type"), "convection")
    bc = model["physics"]["boundary_conditions"][0]
    assert set(bc) == {"type", "selection", "h", "T_ext"}
    E.set_physics_type(model, "linear_elasticity")
    assert model["physics"]["boundary_conditions"] == []
    assert set(model["materials"][0]["properties"]) >= {"E", "nu"}
    E.do_action(model, ("physics",), "add_bc")
    assert M.validate(model) == []
    # mesher change: parameters of the new mesher only
    E.apply(model, E.Field(("meshes", 0, "mesher"), "", "choice", op="mesher"), "getfem")
    assert "divisions" not in model["meshes"][0] and model["meshes"][0]["h"] > 0
    assert M.validate(model) == []


def test_dimension_change_adapts_the_model():
    model = load("plate.json")
    E.set_physics_type(model, "linear_elasticity")
    model["physics"]["plane"] = "stress"
    E.do_action(model, ("physics",), "add_bc")
    E.apply(model, E.Field(("geometries", 0, "source", "size"), "", "choice", op="box_dim"), 3)
    assert len(model["geometries"][0]["source"]["size"]) == 3
    assert "plane" not in model["physics"]
    assert M.validate(model) == [], M.validate(model)


def test_csg_shapes():
    model = load("plate_with_hole.json")
    gi = next(i for i, g in enumerate(model["geometries"]) if g["source"]["type"] == "csg")
    shape = ("geometries", gi, "source", "shape")
    E.apply(model, E.Field(shape + ("of", 1, "type"), "", "choice", op="shape_type"), "union")
    assert E.get_at(model, shape + ("of", 1, "type")) == "union"
    assert len(E.get_at(model, shape + ("of", 1, "of"))) == 2
    new = E.do_action(model, shape, "add_shape:rectangle")
    assert ("remove", "Remove") in E.actions(model, new)
    assert M.validate(model) == []


def test_errors_are_placed_on_nodes_and_fields():
    model = load("plate.json")
    model["materials"][0]["properties"]["k"] = ""
    model["physics"]["boundary_conditions"][0]["selection"] = "nowhere"
    errors = M.validate(model)
    by_node = E.errors_by_node(E.tree(model), errors)
    assert ("materials", 0) in by_node
    assert ("physics", "boundary_conditions", 0) in by_node
    assert E.errors_at(errors, ("materials", 0, "properties", "k")) == ["empty expression"]
    assert E.error_path("format_version: 2 expected")[0] == ("format_version",)
    assert E.error_path("something without a path")[0] == ()


def test_values():
    assert E.parse_value(" 2 ") == 2 and E.parse_value("1e3") == 1000.0
    assert E.parse_value("sin(X(1))") == "sin(X(1))"
    assert E.format_value(200000.0) == "200000" and E.format_value(0.3) == "0.3"
    assert E.split_vector("[0, -9.81*rho]", 2) == [0, "-9.81*rho"]
    assert E.split_vector("[f(1,2), 3]", 2) == ["f(1,2)", 3]
    assert E.split_vector("g_vec", 2) is None
    assert E.split_vector([1.0, 2.0], 2) == [1.0, 2.0]


def test_highlight():
    model = load("plate_with_hole.json")
    mesh = M.physics_mesh(model)["name"]
    assert E.highlighted_selection(model, ("physics", "boundary_conditions", 0))[0] == mesh
    assert E.highlighted_selection(model, ("materials", 0)) == (mesh, model["materials"][0]["selection"])
    assert E.highlighted_selection(model, ("physics",)) == (None, None)
    assert E.mesh_to_show(model, ("meshes", 0)) == model["meshes"][0]["name"]


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok       {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAILED   {name}: {exc!r}")
    sys.exit(1 if failed else 0)
