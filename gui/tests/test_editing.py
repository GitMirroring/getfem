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


def MB(k=0, i=0):
    """Path of the module i of the model k."""
    return ("models", k, "modules", i)


def test_tree_and_fields_of_every_node():
    for ex in EXAMPLES:
        model = M.load(ex)
        root = E.tree(model, result_fields={"heat": ["T"]})
        kinds = {n.kind for n in root.walk()}
        assert {"geometry", "mesh", "model", "module", "variable", "property"} <= kinds, ex.name
        for node in root.walk():
            for f in E.fields(model, node.path):
                assert f.kind in ("text", "ident", "value", "vector", "numbers", "ints", "int",
                                  "number", "choice", "names", "file", "info", "parameter_table"), (node.path, f)
                if f.kind == "choice":
                    value = E.field_value(model, f)
                    assert value is None or value in [c[0] for c in f.choices], (node.path, f, value)


def test_every_action_keeps_a_valid_model():
    """Adding items with their default values must not introduce errors."""
    for ex in EXAMPLES:
        base = M.load(ex)
        assert M.validate(base) == [], ex.name
        for node in E.tree(base).walk():
            for action, _ in E.actions(base, node.path):
                if action in ("remove", "add_import") or (action == "unfold"):
                    continue
                model = copy.deepcopy(base)
                new = E.do_action(model, node.path, action, name="rho" if E.needs_name(action) else None)
                errors = M.validate(model)
                if action.startswith(("add_model:", "add_module:")):   # new: conditions to be set
                    errors = [e for e in errors if "up to a constant" not in e and "rigid body" not in e]
                assert errors == [], (ex.name, node.path, action, errors)
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
    uses = [bc["selection"] for bc in E.get_at(model, MB() + ("conditions",))]
    assert "circle" in uses and "hole" not in uses
    assert M.validate(model) == []
    try:
        E.apply(model, E.Field(new, f.label, f.kind, op=f.op), "left")
        raise AssertionError("duplicate name accepted")
    except ValueError:
        pass


def test_rename_mesh_geometry_model_variable_material():
    model = load("thermoelastic_plate.json")
    E.apply(model, E.Field(("meshes", 0, "name"), "", "ident", op="rename_mesh"), "m1")
    E.apply(model, E.Field(("geometries", 0, "name"), "", "ident", op="rename_geometry"), "g1")
    E.apply(model, E.Field(("models", 0, "name"), "", "ident", op="rename_model"), "heat")
    E.apply(model, E.Field(("models", 0, "variables", 0, "name"), "", "ident", op="rename_variable"), "theta")
    E.apply(model, E.Field(("materials", 0, "name"), "", "ident", op="rename_material"), "alu")
    assert M.validate(model) == [], M.validate(model)
    assert model["meshes"][0]["geometry"] == "g1" and model["models"][1]["variables"][0]["mesh"] == "m1"
    assert model["solvers"][0]["steps"] == ["heat", "mechanical"]
    imp = model["models"][1]["imports"][0]
    assert imp == {"name": "T", "model": "heat", "variable": "theta"}      # the local name stays
    th = model["models"][0]
    assert th["modules"][0]["variable"] == "theta" and th["outputs"][0]["expr"] == "-k*Grad_theta.Normal"
    assert model["models"][1]["modules"][1]["temperature"] == "T"
    assert th["modules"][0]["properties"]["k"][0]["material"] == "alu"


def test_rename_a_local_parameter_stays_local():
    model = load("thermoelastic_plate.json")
    for md in model["models"]:
        md["parameters"] = [{"name": "c", "value": 1.0}]
        md["outputs"].append({"name": "q2", "expr": "c*2", "selection": "all"})
    E.apply(model, E.Field(("models", 0, "parameters", 0, "name"), "", "ident", op="rename_parameter"), "c0")
    assert model["models"][0]["outputs"][-1]["expr"] == "c0*2"
    assert model["models"][1]["outputs"][-1]["expr"] == "c*2"
    assert M.validate(model) == []


def test_type_changes_reset_fields():
    model = load("plate.json")
    bcs = MB() + ("conditions",)
    E.apply(model, E.Field(bcs + (0, "type"), "", "choice", op="mcond_type"), "convection")
    bc = E.get_at(model, bcs + (0,))
    assert set(bc) == {"type", "selection", "h", "T_ext"}
    # a generic condition
    p = E.do_action(model, ("models", 0), "add_condition")
    E.apply(model, E.Field(p + ("type",), "", "choice", op="cond_type"), "robin")
    assert set(E.get_at(model, p)) == {"type", "variable", "selection", "h", "value"}
    assert M.validate(model) == []
    # mesher change: parameters of the new mesher only, the fineness is kept
    f = E.Field(("meshes", 0, "mesher"), "", "choice", op="mesher")
    E.apply(model, f, "getfem")                              # plate 0.2 x 0.1, 40 x 20 divisions
    assert "divisions" not in model["meshes"][0] and model["meshes"][0]["h"] == 0.005
    model["meshes"][0].update(h="2*0.004", order=2)
    E.apply(model, f, "gmsh")
    assert model["meshes"][0] == {"name": "mesh", "geometry": "plate", "mesher": "gmsh",
                                  "size": "2*0.004", "order": 2}
    E.apply(model, f, "getfem")
    assert model["meshes"][0]["h"] == "2*0.004" and model["meshes"][0]["order"] == 2
    model["meshes"][0]["h"] = 0.01
    E.apply(model, f, "structured")
    assert model["meshes"][0]["divisions"] == [20, 10]
    assert M.validate(model) == []


def test_modules_and_their_variables():
    model = load("plate.json")
    md = model["models"][0]
    new = E.do_action(model, ("models", 0, "modules"), "add_module:thermal_expansion")
    # the expansion needs a displacement and an elasticity module: both added
    assert [m["type"] for m in md["modules"]] == ["heat_conduction", "linear_elasticity", "thermal_expansion"]
    assert [v["name"] for v in md["variables"]] == ["T", "u"] and md["variables"][1]["components"] == "dim"
    assert E.get_at(model, new)["temperature"] == "T"
    md["modules"][1]["conditions"] = [{"type": "fixed", "selection": "xmin"}]
    assert M.validate(model) == [], M.validate(model)
    # a property given by a material, then by a value
    piece = MB() + ("properties", "k", 0)
    f = next(f for f in E.fields(model, piece) if f.op == "piece_source")
    assert E.field_value(model, f) == "material:aluminium"
    E.apply(model, f, "value")
    assert E.get_at(model, piece) == {"selection": "all", "value": 237.0}
    E.apply(model, f, "material:aluminium")
    assert E.get_at(model, piece) == {"selection": "all", "material": "aluminium"}
    p2 = E.do_action(model, MB(), "add_piece:k")
    assert p2 == MB() + ("properties", "k", 1) and M.validate(model) == []


def test_new_models_from_the_root():
    model = load("plate.json")
    p = E.do_action(model, (), "add_model:linear_elasticity")
    md = E.get_at(model, p)
    assert md["name"] == "mechanics" and md["variables"][0]["mesh"] == "mesh"
    md["modules"][0]["conditions"] = [{"type": "fixed", "selection": "xmin"}]
    md["modules"][0]["properties"] = {"E": [{"selection": "all", "value": 1.0}],
                                      "nu": [{"selection": "all", "value": 0.3}]}
    assert M.validate(model) == []
    imp = E.do_action(model, p, "add_import")
    assert E.get_at(model, imp) == {"name": "T2", "model": "heat", "variable": "T"} or \
        E.get_at(model, imp)["model"] == "heat"
    s = E.do_action(model, (), "add_solver")
    assert model["solvers"][0]["steps"] == ["heat", "mechanics"] and M.validate(model) == []


def test_unfold_heat_module():
    from getfem_gui import runner
    import tempfile
    model = load("plate_with_hole.json")
    ref = model
    unfolded = copy.deepcopy(model)
    E.do_action(unfolded, MB(), "unfold")
    md = unfolded["models"][0]
    assert md["modules"] == [] and md["fields"][0]["name"] == "k"
    assert [t["name"] for t in md["terms"]][:1] == ["conduction"]
    assert M.validate(unfolded) == [], M.validate(unfolded)
    with tempfile.TemporaryDirectory() as tmp:
        r1 = runner.run(ref, tmp + "/a").results["models"]["heat"]["quantities"]
        r2 = runner.run(unfolded, tmp + "/b").results["models"]["heat"]["quantities"]
    for k, v in r1.items():
        assert abs(r2[k] - v) <= 1e-9 * max(1.0, abs(v)), (k, v, r2[k])


def test_dimension_change_adapts_the_model():
    model = load("plate.json")
    E.do_action(model, ("models", 0, "modules"), "add_module:linear_elasticity")
    md = model["models"][0]
    assert md["modules"][1]["plane"] == "strain"
    md["modules"][1]["conditions"] = [{"type": "fixed", "selection": "xmin"}]
    E.apply(model, E.Field(("geometries", 0, "source", "size"), "", "choice", op="box_dim"), 3)
    assert len(model["geometries"][0]["source"]["size"]) == 3
    assert "plane" not in md["modules"][1]
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
    E.get_at(model, MB() + ("conditions", 0))["selection"] = "nowhere"
    errors = M.validate(model)
    by_node = E.errors_by_node(E.tree(model), errors)
    assert ("materials", 0) in by_node
    assert MB() + ("conditions", 0) in by_node
    assert E.errors_at(errors, ("materials", 0, "properties", "k")) == ["empty expression"]
    assert E.error_path("format_version: 2 expected")[0] == ("format_version",)
    assert E.error_path("something without a path")[0] == ()


def test_graphics():
    model = load("plate_hole_tension.json")
    md = model["models"][0]
    p = E.do_action(model, ("models", 0, "graphics"), "add_graphic")
    assert p == ("models", 0, "graphics", 0) and md["graphics"][0] == {"name": "von_mises", "field": "von_mises"}
    g = md["graphics"][0]
    E.set_graphic(g, {"isolines": 8, "cmap": "jet", "warp": False, "cut": [[0, 0], [0.1, 0.05]]})
    E.set_graphic(g, {"warp": True})                          # back to the default: not stored
    assert g == {"name": "von_mises", "field": "von_mises", "isolines": 8, "cmap": "jet",
                 "cut": [[0, 0], [0.1, 0.05]]}
    assert E.graphic_settings(g)["warp"] is True and E.graphic_settings(g)["isolines"] == 8
    assert E.find_graphic(md, "von_mises") == 0 and M.validate(model) == []
    assert E.find_node(E.tree(model), p).kind == "graphic"
    for bad, msg in (({"isolines": -1}, "isolines"), ({"cmap": "rainbow"}, "cmap"),
                     ({"range": [1]}, "range"), ({"cut": [[0, 0]]}, "cut"), ({"foo": 1}, "unknown setting")):
        m = copy.deepcopy(model)
        m["models"][0]["graphics"][0].update(bad)
        assert any(f"graphics[0].{msg}" in e or msg in e for e in M.validate(m)), bad
    m = copy.deepcopy(model)
    m["models"][0]["graphics"].append({"name": "von_mises", "field": "u"})
    assert any("used more than once" in e for e in M.validate(m))


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
    mesh = model["models"][0]["variables"][0]["mesh"]
    assert E.highlighted_selection(model, MB() + ("conditions", 0))[0] == mesh
    piece = MB() + ("properties", "k", 0)
    assert E.highlighted_selection(model, piece) == (mesh, E.get_at(model, piece)["selection"])
    assert E.highlighted_selection(model, ("models", 0)) == (None, None)
    assert E.mesh_to_show(model, ("meshes", 0)) == model["meshes"][0]["name"]
    assert E.mesh_to_show(model, ("models", 0, "terms")) == mesh


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
