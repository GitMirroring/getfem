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
                                  "number", "choice", "names", "file", "info", "parameter_table", "material_table",
                                  "vertex_table", "chain_table"), (node.path, f)
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
                if action in ("remove", "clear_shape", "add_import", "unfold", "add_from_library") \
                        or action.startswith(("cmd:", "pick_plane:")):      # (pick_plane: a click in the view)
                    continue
                model = copy.deepcopy(base)
                new = E.do_action(model, node.path, action, name="zeta" if E.needs_name(action) else None)
                errors = M.validate(model)
                if action.startswith(("add_model:", "add_module:")):   # new: conditions to be set
                    errors = [e for e in errors if "up to a constant" not in e and "rigid body" not in e]
                if action in ("add_shape:union", "add_shape:intersection", "add_shape:difference",
                              "add_shape:extrusion", "add_shape:revolution"):
                    # an empty boolean (or profile), to be filled
                    errors = [e for e in errors if ": add shapes to it" not in e]
                if action == "add_shape:polygon":        # its vertices to enter
                    errors = [e for e in errors if "at least three vertices" not in e]
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


def test_material_libraries():
    import json
    import os
    import tempfile
    from getfem_gui import library
    names = [m["name"] for _, m in library.materials(library.LIBRARY)]
    assert "Copper" in names and len(names) > 20
    for _, m in library.materials(library.LIBRARY):
        assert set(m["properties"]) <= set(M.PROPERTIES), m["name"]
    model = load("plate_inclusion.json")
    p = E.do_action(model, ("materials",), "add_from_library:library:Stainless steel 304")
    mat = E.get_at(model, p)
    assert mat["name"] == "stainless_steel_304" and mat["from"] == "library:Stainless steel 304"
    assert mat["properties"]["k"] == 16.2 and M.validate(model) == []
    p2 = E.do_action(model, ("materials",), "add_from_library:library:Stainless steel 304")
    assert E.get_at(model, p2)["name"] == "stainless_steel_3042"        # a second copy
    # used by a property: the copy, not the library
    piece = MB() + ("properties", "k", 1)
    f = next(x for x in E.fields(model, piece) if x.op == "piece_source")
    assert ("material:stainless_steel_304", "material stainless_steel_304") in f.choices
    E.apply(model, f, "material:stainless_steel_304")
    mat["properties"]["k"] = 15.0
    assert M.validate(model) == [] and library.find("library:Stainless steel 304")["properties"]["k"] == 16.2
    # the table: the properties asked by the modules only (here k)
    props, rows = E.material_rows(model)
    assert props == ["k"] and [r["values"] for r in rows][-2] == [15.0]
    E.apply(model, rows[0]["fields"][0], 0.3)
    assert model["materials"][0]["properties"]["k"] == 0.3
    # the user's library
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["GETFEM_GUI_MATERIALS"] = str(Path(tmp) / "mine.json")
        try:
            library.save_to_user({"name": "x", "description": "My steel", "properties": {"k": 20.0}})
            assert json.loads((Path(tmp) / "mine.json").read_text())["materials"][0]["name"] == "My steel"
            assert ("user", library.find("user:My steel")) in library.materials()
            p3 = E.do_action(model, ("materials",), "add_from_library:user:My steel")
            assert E.get_at(model, p3)["name"] == "my_steel"
        finally:
            del os.environ["GETFEM_GUI_MATERIALS"]


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



def test_default_mesher():
    """New meshes of CSG and CAD geometries use Gmsh (a box: structured);
    a geometry that changes its kind of source changes the mesher of its
    meshes if needed."""
    m = M.default_model()
    assert E.default_mesher("box") == "structured" and E.default_mesher("cad") == "gmsh"
    assert E.default_mesher("csg") == E.DEFAULT_MESHER == "gmsh"
    E.apply(m, next(f for f in E.fields(m, ("geometries", 0)) if f.op == "source_type"), "csg")
    assert m["meshes"][0]["mesher"] == "gmsh" and "divisions" not in m["meshes"][0]
    path = E.do_action(m, ("geometries", 0), "add_mesh")
    assert E.get_at(m, path)["mesher"] == "gmsh" and E.get_at(m, path)["size"] > 0
    from getfem_gui import mesher
    err = mesher.error_line("ModuleNotFoundError: No module named 'gmsh'")
    assert "pip install gmsh" in err and "GetFEM mesher" in err


def test_add_a_geometry_of_each_kind():
    """The menu of the kinds of geometry: each gives a valid geometry (the
    files excepted, until one is chosen), with a mesh added on demand."""
    for _, kind, _, _ in E.GEOMETRY_KINDS:
        m = M.default_model()
        path = E.do_action(m, ("geometries",), f"add_geometry:{kind}")
        geom = E.get_at(m, path)
        assert geom["name"] == "geometry2"
        errors = [e for e in M.validate(m) if e.startswith(f"geometries[{path[1]}]")]
        if kind in ("cad", "mesh_file"):
            assert all("file" in e for e in errors), (kind, errors)
            continue
        if kind in ("csg2", "csg3", "extrusion", "revolution"):      # from nothing: a shape to add
            assert len(errors) == 1 and "empty union: add shapes to it" in errors[0], (kind, errors)
            root = path + ("source", "shape") + (("profile",) if kind in ("extrusion", "revolution") else ())
            E.do_action(m, root, "add_shape:rectangle")
            errors = [e for e in M.validate(m) if e.startswith(f"geometries[{path[1]}]")]
        assert errors == [], (kind, errors)
        mesh = E.do_action(m, path, "add_mesh")
        assert M.validate(m) == [], (kind, M.validate(m))
        dims = {"box1": 1, "box2": 2, "csg2": 2, "curve2": 2}
        assert M.geometry_dim(m, geom) == dims.get(kind, 3), kind
        assert E.get_at(m, mesh)["geometry"] == "geometry2"
    src = E.new_geometry_source("revolution")
    assert src["shape"]["type"] == "revolution" and src["shape"]["profile"] == {"type": "union", "of": []}
    assert E.new_geometry_source("swept_extrusion")["sweep"] == "extrusion"
    assert ("add_geometry", "Add a geometry") in E.actions(M.default_model(), ("geometries",))


def test_constructive_geometry_from_nothing():
    """A csg geometry starts empty: shapes and booleans are added, a shape is
    put in a boolean, the root emptied again; the references follow."""
    import tempfile
    from getfem_gui import mesher, parameters as P
    m = M.default_model()
    path = E.do_action(m, ("geometries",), "add_geometry:csg2")
    root = path + ("source", "shape")
    assert E.get_at(m, root) == {"type": "union", "of": []}
    assert any("empty union: add shapes to it" in e for e in M.validate(m))
    tree = E.find_node(E.tree(m), root)
    assert "empty" in tree.label
    labels = dict(E.actions(m, root))
    assert "add_shape:difference" in labels and "wrap:union" not in labels and "clear_shape" not in labels
    rect = E.do_action(m, root, "add_shape:rectangle")
    assert rect == root + ("of", 0) and M.validate(m) == []            # a union of one shape
    # meshed: a boolean of one shape is that shape
    m["meshes"].append({"name": "m2", "geometry": E.get_at(m, path)["name"], "mesher": "getfem", "h": 0.2})
    code = mesher.mesh_script(P.resolved(m), m["meshes"][-1], "m2")
    compile(code, "m.py", "exec")
    with tempfile.TemporaryDirectory() as tmp:
        assert mesher.ensure_mesh(m, "m2", tmp).ok
    gm = dict(m["meshes"][-1], mesher="gmsh", size=0.2)
    gm.pop("h")
    compile(mesher.mesh_script(P.resolved(m), gm, "m2"), "m.py", "exec")
    # a selection of a side of the rectangle follows the changes of the tree
    E.get_at(m, path)["selections"] = {"left": {"kind": "boundary", "criteria": {
        "type": "edges", "edges": [{"shape": [0], "side": "xmin"}]}}}
    E.do_action(m, rect, "wrap:difference")
    assert E.get_at(m, rect)["type"] == "difference"
    assert E.get_at(m, path)["selections"]["left"]["criteria"]["edges"] == [{"shape": [0, 0], "side": "xmin"}]
    hole = E.do_action(m, rect, "add_shape:ball")
    assert hole == rect + ("of", 1) and M.validate(m) == []
    inner = E.do_action(m, root, "add_shape:intersection")
    assert any("of[1].of: empty intersection" in e for e in M.validate(m))
    E.do_action(m, inner, "add_shape:rectangle")
    assert M.validate(m) == []
    assert "clear_shape" in dict(E.actions(m, root))
    E.do_action(m, root, "clear_shape")
    assert E.get_at(m, root) == {"type": "union", "of": []}
    assert E.get_at(m, path)["selections"]["left"]["criteria"]["edges"] == []
    # drawing on an empty geometry: the drawn shape is the first one
    new, _ = E.add_shape_2d(m, path[1], {"type": "ball", "center": [0, 0], "radius": 1.0}, "difference")
    assert new == root + ("of", 0) and E.get_at(m, root)["type"] == "union"
    # an extrusion of a profile from nothing
    p3 = E.do_action(m, ("geometries",), "add_geometry:extrusion")
    prof = p3 + ("source", "shape", "profile")
    assert E.get_at(m, prof) == {"type": "union", "of": []}
    poly = E.do_action(m, prof, "add_shape:polygon")         # empty: its vertices to type
    for _ in range(3):
        E.do_action(m, poly, "add_vertex")
    assert not [e for e in M.validate(m) if e.startswith(f"geometries[{p3[1]}]")]


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


def test_new_geometries_and_sweeps_start_empty():
    """A new constructive geometry: its root (an empty union) is where one
    goes, to add shapes; a new extrusion or revolution, as a geometry or in
    one, has an empty profile, where one goes to draw it (the extrusion
    between the heights of the rest of the geometry)."""
    m = M.default_model()
    for kind, where in (("csg2", ("source", "shape")), ("csg3", ("source", "shape")),
                        ("extrusion", ("source", "shape", "profile")),
                        ("revolution", ("source", "shape", "profile"))):
        g = E.do_action(m, ("geometries",), "add_geometry:" + kind)
        assert E.selected_after("add_geometry:" + kind, g) == tuple(g) + where
        assert E.get_at(m, tuple(g) + where) == E.EMPTY_SHAPE
    gi = len(m["geometries"]) - 4                     # the 3D solid
    root = ("geometries", gi + 1, "source", "shape")
    E.do_action(m, root, "add_shape:rectangle")        # a box
    m["geometries"][gi + 1]["source"]["shape"]["of"][0].update(min=[0, 0, -1], max=[2, 2, 3])
    for t in ("extrusion", "revolution"):
        p = E.do_action(m, root, "add_shape:" + t)
        assert E.selected_after("add_shape:" + t, p) == tuple(p) + ("profile",)
        new = E.get_at(m, p)
        assert new["type"] == t and new["profile"] == E.EMPTY_SHAPE
    assert E.get_at(m, root)["of"][1]["z"] == [-1.0, 3.0]              # the heights of the box
    assert E.selected_after("add_shape:rectangle", ("x",)) == ("x",)
