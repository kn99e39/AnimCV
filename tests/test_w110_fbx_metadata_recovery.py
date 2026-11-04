"""Focused tests for the W110 FBX-metadata recovery diagnostic (scripts/w110_*; evidence only)."""
from __future__ import annotations

import ast
import inspect
import json
import re
import struct
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))

import w110_fbx_consumer_adapter as AD  # noqa: E402
import w110_fbx_metadata_recovery_contract as C  # noqa: E402

E = namedtuple("E", ("id", "props", "props_type", "elems"))
W110_BASE = Path("C:/Users/dna10/animcv-w110")


# --- tiny FBX-like tree builder ------------------------------------------------------------------------------------
def _node(nid, props, types, elems=()):
    return E(nid, list(props), types, list(elems))


def _curve(uid, flag, n=3):
    return _node(b"AnimationCurve", [uid, b"\x00\x01AnimCurve", b""], b"LSS", [
        _node(b"KeyTime", [np.arange(n, dtype=np.int64)], b"l"),
        _node(b"KeyValueFloat", [np.zeros(n, dtype=np.float32)], b"f"),
        _node(b"KeyAttrFlags", [np.array([flag], dtype=np.int32)], b"i"),
        _node(b"KeyAttrRefCount", [np.array([n], dtype=np.int32)], b"i")])


def _model(uid, name, mtype, p70=()):
    props = [_node(b"P", [k.encode(), b"Vector3D", b"", b"A", *v], b"SSSSDDD") for k, v in p70]
    return _node(b"Model", [uid, name.encode() + b"\x00\x01Model", mtype.encode()], b"LSS",
                 [_node(b"Properties70", [], b"", props)])


def build_tree(flags, stack="root|take", extra_curves=(), p70=None, extra_layer=False, nontransform=False):
    """Models: root (Null), aa (LimbNode, parent root), bb (LimbNode, parent aa).  ``flags``: {(model, prop): flag}."""
    objs = [_node(b"AnimationStack", [1, stack.encode() + b"\x00\x01AnimStack", b""], b"LSS"),
            _node(b"AnimationLayer", [2, stack.encode() + b"\x00\x01AnimLayer", b""], b"LSS"),
            _model(10, "root", "Null"), _model(11, "aa", "LimbNode", (p70 or {}).get("aa", ())),
            _model(12, "bb", "LimbNode")]
    conns = [_node(b"C", [b"OO", 2, 1], b"SLL"), _node(b"C", [b"OO", 11, 10], b"SLL"), _node(b"C", [b"OO", 12, 11], b"SLL")]
    if extra_layer:
        objs.append(_node(b"AnimationLayer", [3, b"second\x00\x01AnimLayer", b""], b"LSS"))
        conns.append(_node(b"C", [b"OO", 3, 1], b"SLL"))
    uid = 100
    props = {"T": b"Lcl Translation", "R": b"Lcl Rotation", "S": b"Lcl Scaling"}
    if nontransform:
        props["V"] = b"Visibility"
    for mi, m in ((10, "root"), (11, "aa"), (12, "bb")):
        for p, pname in props.items():
            cn = uid
            uid += 1
            objs.append(_node(b"AnimationCurveNode", [cn, p.encode() + b"\x00\x01AnimCurveNode", b""], b"LSS"))
            conns += [_node(b"C", [b"OO", cn, 2], b"SLL"), _node(b"C", [b"OP", cn, mi, pname], b"SLLS")]
            for ax in ("X", "Y", "Z"):
                cu = uid
                uid += 1
                objs.append(_curve(cu, flags.get((m, p), 0x6104)))
                conns.append(_node(b"C", [b"OP", cu, cn, f"d|{ax}".encode()], b"SLLS"))
    for cu, cn, ax in extra_curves:
        objs.append(_curve(cu, 0x6104))
        conns.append(_node(b"C", [b"OP", cu, cn, ax.encode()], b"SLLS"))
    root = E(b"", [], b"", [_node(b"Objects", [], b"", objs), _node(b"Connections", [], b"", conns)])
    return root


def scene_desc(stack="root|take", bones=None, missing_fc=None, extra_fc=None, rot="QUATERNION"):
    bones = bones or {"aa": {"parent": None, "rotation_mode": rot}, "bb": {"parent": "aa", "rotation_mode": rot}}
    fl = []
    for item, pre in (("root", ""), *((b, f'pose.bones["{b}"].') for b in bones)):
        q = "rotation_quaternion" if pre and rot == "QUATERNION" else "rotation_euler"
        for p, n in (("location", 3), (q, 4 if q == "rotation_quaternion" else 3), ("scale", 3)):
            fl += [[pre + p, i] for i in range(n)]
    if missing_fc:
        fl.remove(missing_fc)
    if extra_fc:
        fl.append(extra_fc)
    return {"objects": {"root": {"type": "ARMATURE", "rotation_mode": "XYZ", "bones": bones}},
            "actions": {f"root|{stack}": {"slots": ["OBSlot"], "fcurves": {"OBSlot": fl}},
                        "root|other": {"slots": ["OBSlot"], "fcurves": {"OBSlot": []}}}}


def directives(pl):
    return {(g["item"], g["group"]): g["directive"] for g in pl["groups"]}


# --- flags --------------------------------------------------------------------------------------------------------
def test_flag_classes_and_producer_flag():
    assert AD.flag_class(0x6104) == "LINEAR"
    assert AD.flag_class(0x6002) == "CONSTANT"
    assert AD.flag_class(0x6102) == "CONSTANT_NEXT"     # bit 0x100 under CONSTANT is eConstantNext
    assert AD.flag_class(0x6108) == "CUBIC"
    assert AD.flag_class(0x6000).startswith("UNKNOWN")
    assert C.CONSTANT_FLAG == 0x6002 and C.SOURCE_FLAG == 0x6104


# --- FBX-side map ------------------------------------------------------------------------------------------------
def test_fbx_map_connection_graph():
    fm = AD.fbx_curve_map(build_tree({}), "root|take")
    assert fm["status"] == "OK" and len(fm["curves"]) == 27
    assert {r["status"] for r in fm["curves"]} == {"OK"}
    r = fm["curves"][0]
    assert r["model_name"] == "root" and r["model_type"] == "Null" and r["group"] == "location" and r["channel"] == 0
    assert fm["models"]["11"]["parent"] == "root" and fm["models"]["11"]["parent_type"] == "Null"
    assert all(m["separable"] for m in fm["models"].values())


def test_fbx_map_rejects_and_counts():
    assert AD.fbx_curve_map(build_tree({}), "nope")["status"].startswith("NO_UNIQUE_STACK")
    assert AD.fbx_curve_map(build_tree({}, extra_layer=True), "root|take")["status"].startswith("UNSUPPORTED_LAYER")
    fm = AD.fbx_curve_map(build_tree({}, extra_curves=[(999, 104, "d|X")]), "root|take")   # 2nd curve on aa T.X? no: cn 104
    dup = [r for r in fm["curves"] if r["status"] == "DUPLICATE_CHANNEL"]
    assert len(dup) == 2
    fm = AD.fbx_curve_map(build_tree({}, nontransform=True), "root|take")
    assert sum(r["status"] == "UNSUPPORTED_PROPERTY" for r in fm["curves"]) == 9


def test_separability_from_model_properties():
    fm = AD.fbx_curve_map(build_tree({}, p70={"aa": [("RotationPivot", (0.0, 0.1, 0.0))]}), "root|take")
    assert fm["models"]["11"]["separable"] is False
    fm = AD.fbx_curve_map(build_tree({}, p70={"aa": [("RotationPivot", (0.0, 0.0, 0.0)),
                                                     ("GeometricScaling", (1.0, 1.0, 1.0))]}), "root|take")
    assert fm["models"]["11"]["separable"] is True


# --- planning -----------------------------------------------------------------------------------------------------
def test_plan_property_level_mixed_item():
    fm = AD.fbx_curve_map(build_tree({("aa", "R"): 0x6002}), "root|take")
    pl = AD.plan(fm, scene_desc())
    d = directives(pl)
    assert d[("aa", "rotation")] == "CONSTANT"
    assert d[("aa", "location")] == "LINEAR" and d[("aa", "scale")] == "LINEAR" and d[("bb", "rotation")] == "LINEAR"
    acc = pl["accounting"]
    assert acc["fbx_curves"] == acc["fbx_curves_mapped"] == 27 and acc["fbx_curves_constant"] == 3
    assert acc["imported_fcurves_in_target_actions"] == acc["imported_fcurves_mapped"] == 29
    assert acc["imported_fcurves_unmapped"] == [] and acc["fcurves_claimed_twice"] == 0
    assert acc["axis_level_one_to_one_curves"] == 0
    g = [x for x in pl["groups"] if x["item"] == "aa" and x["group"] == "rotation"][0]
    assert [f[:2] for f in g["fcurves"]] == [['pose.bones["aa"].rotation_quaternion', i] for i in range(4)]


def test_plan_coupled_item_needs_item_consensus():
    fm = AD.fbx_curve_map(build_tree({("aa", "R"): 0x6002}, p70={"aa": [("RotationPivot", (0.0, 0.1, 0.0))]}),
                          "root|take")
    d = directives(AD.plan(fm, scene_desc()))
    assert d[("aa", "rotation")] == d[("aa", "location")] == "UNRESOLVED"
    fm = AD.fbx_curve_map(build_tree({("aa", p): 0x6002 for p in "TRS"},
                                     p70={"aa": [("RotationPivot", (0.0, 0.1, 0.0))]}), "root|take")
    assert directives(AD.plan(fm, scene_desc()))[("aa", "rotation")] == "CONSTANT"


def test_plan_unresolved_cases():
    tree = build_tree({("aa", "R"): 0x6102})                       # CONSTANT-next: no Blender equivalent
    assert directives(AD.plan(AD.fbx_curve_map(tree, "root|take"), scene_desc()))[("aa", "rotation")] == "UNRESOLVED"
    tree = build_tree({("aa", "R"): 0x6108})                       # cubic
    assert directives(AD.plan(AD.fbx_curve_map(tree, "root|take"), scene_desc()))[("aa", "rotation")] == "UNRESOLVED"
    # bone missing from the imported armature
    pl = AD.plan(AD.fbx_curve_map(build_tree({}), "root|take"),
                 scene_desc(bones={"aa": {"parent": None, "rotation_mode": "QUATERNION"}}))
    assert directives(pl)[("bb", "rotation")] == "UNRESOLVED" and pl["accounting"]["items_unresolved"] == 1
    # parent mismatch
    pl = AD.plan(AD.fbx_curve_map(build_tree({}), "root|take"),
                 scene_desc(bones={"aa": {"parent": None, "rotation_mode": "QUATERNION"},
                                   "bb": {"parent": None, "rotation_mode": "QUATERNION"}}))
    assert directives(pl)[("bb", "location")] == "UNRESOLVED"
    # missing / unmapped imported F-curves
    pl = AD.plan(AD.fbx_curve_map(build_tree({}), "root|take"),
                 scene_desc(missing_fc=['pose.bones["aa"].scale', 2], extra_fc=["custom", 0]))
    assert directives(pl)[("aa", "scale")] == "UNRESOLVED"
    assert pl["accounting"]["imported_fcurves_unmapped"] == ["root|root|take:custom[0]"]


def test_plan_disagreement_within_group():
    tree = build_tree({("aa", "R"): 0x6104})
    # make only one axis of aa R constant
    objs = tree.elems[0].elems
    fm0 = AD.fbx_curve_map(tree, "root|take")
    uid = [r["uid"] for r in fm0["curves"] if r["model_name"] == "aa" and r["group"] == "rotation"][1]
    for k, o in enumerate(objs):
        if o.id == b"AnimationCurve" and o.props[0] == uid:
            objs[k] = _curve(uid, 0x6002)
    d = directives(AD.plan(AD.fbx_curve_map(tree, "root|take"), scene_desc()))
    assert d[("aa", "rotation")] == "UNRESOLVED"


# --- producer rule ------------------------------------------------------------------------------------------------
def test_source_group_parse():
    assert C.source_group_of('pose.bones["aa"].rotation_quaternion') == ("bone", "aa", "rotation")
    assert C.source_group_of("location") == ("object", None, "location")
    assert C.source_group_of('pose.bones["a\\"b"].scale') == ("bone", 'a"b', "scale")
    assert C.source_group_of("delta_location") is None
    assert C.source_group_of('pose.bones["aa"]["prop"]') is None


def test_producer_directives():
    fm = AD.fbx_curve_map(build_tree({}), "root|take")
    census = {"armature_object": "root", "bones": ["aa", "bb"],
              "groups": {"object|rotation": ["CONSTANT"], "bone:aa|rotation": ["LINEAR"],
                         "bone:bb|location": ["BEZIER"]}}
    d = C.producer_directives(fm, census)
    by = {}
    for r in fm["curves"]:
        by.setdefault((r["model_name"], r["group"]), set()).add(d[r["uid"]][0])
    assert by[("root", "rotation")] == {"CONSTANT"} and by[("aa", "rotation")] == {"LINEAR"}
    assert by[("bb", "location")] == {"UNSUPPORTED"} and by[("aa", "scale")] == {"UNDRIVEN"}
    census["bones"] = ["aa"]
    assert {d2 for d2, _ in C.producer_directives(fm, census).values()} >= {"UNMATCHED"}


# --- digests, grids, metrics --------------------------------------------------------------------------------------
def test_tree_digest_skips_only_timestamp():
    def tree(sec):
        hdr = _node(b"FBXHeaderExtension", [], b"", [_node(b"CreationTimeStamp", [], b"", [
            _node(b"Second", [sec], b"I")])])
        return E(b"", [], b"", [hdr, _node(b"Objects", [], b"", [_curve(5, 0x6104)])])
    a, b = tree(1), tree(2)
    assert C.tree_digest(a) != C.tree_digest(b)
    assert C.tree_digest(a, skip_paths=C.TIMESTAMP_PATHS) == C.tree_digest(b, skip_paths=C.TIMESTAMP_PATHS)
    c = E(b"", [], b"", [a.elems[0], _node(b"Objects", [], b"", [_curve(5, 0x6002)])])
    assert C.tree_digest(a, skip_paths=C.TIMESTAMP_PATHS) != C.tree_digest(c, skip_paths=C.TIMESTAMP_PATHS)
    assert C.tree_digest(a, [5]) == C.tree_digest(c, [5])


def test_near_boundary_times():
    assert C.near_boundary_times({1: "quiet", 2: "quiet", 3: "transition", 4: "quiet"}) == [3.99]


def _samples(times, bones, f):
    T, B = len(times), len(bones)
    d = {"times": np.array(times), "bones": np.array(bones), "world_head": np.zeros((T, B, 3)),
         "world_tail": np.zeros((T, B, 3)), "local_quat": np.zeros((T, B, 4))}
    for i, t in enumerate(times):
        for j in range(B):
            d["world_tail"][i, j] = f(t, j)
            d["local_quat"][i, j] = [1.0, f(t, j)[0], 0, 0]
    return d


def test_channel_function_and_components():
    times = [1.0, 1.25, 1.5, 2.0]
    held = _samples(times, ["a"], lambda t, j: (float(np.floor(t)), 0, 0))
    lin = _samples(times, ["a"], lambda t, j: (t, 0, 0))
    assert C.channel_function_error(held, times, "a", "floor") == 0.0
    assert C.channel_function_error(lin, times, "a", "lerp") == pytest.approx(0.0)
    assert C.channel_function_error(lin, times, "a", "floor") == pytest.approx(0.5)
    spec = {"held_bones": ["base", "upperarm_l", "lowerarm_l", "hand_l"], "smooth_bones": ["sibling"]}
    bones = ["base", "upperarm_l", "lowerarm_l", "hand_l", "sibling"]
    src = _samples(times, bones, lambda t, j: (t if j == 4 else 0.0, 0, 0))
    imp = _samples(times, bones, lambda t, j: (float(np.floor(t)) if j == 4 else 0.0, 0, 0))
    e = C.component_errors(src, imp, spec)
    assert e["held"] == 0.0 and e["smooth"] == pytest.approx(0.5)


# --- CASE mapping -------------------------------------------------------------------------------------------------
def _pre(**kw):
    cases = C.ALL_CASES
    pre = {k: True for k in ("preflight_ok", "encoding_valid", "isolation_ok", "adapter_integrity", "sources_ok",
                             "deterministic", "originals_unchanged", "smooth_preserved", "mixed_ok", "mixed_held_ok",
                             "controls_discriminate", "attribution")}
    pre.update({"p0": {c: True for c in cases}, "coverage": {c: True for c in cases},
                "recovered": {c: True for c in cases}})
    pre.update(kw)
    return pre


def test_case_mapping():
    assert C.decide_case(_pre())[0] == "A"
    assert C.decide_case(_pre(isolation_ok=False))[0] == "D"
    assert C.decide_case(_pre(adapter_integrity=False, recovered={c: False for c in C.ALL_CASES}))[0] == "D"
    assert C.decide_case(_pre(p0={**{c: True for c in C.ALL_CASES}, "real_clip": False}))[0] == "D"
    assert C.decide_case(_pre(recovered={c: False for c in C.ALL_CASES}, mixed_held_ok=False, mixed_ok=False))[0] == "B"
    assert C.decide_case(_pre(coverage={c: False for c in C.ALL_CASES}))[0] == "B"
    assert C.decide_case(_pre(recovered={**{c: True for c in C.ALL_CASES}, "real_clip": False}))[0] == "C"
    assert C.decide_case(_pre(mixed_ok=False))[0] == "C"
    assert C.decide_case(_pre(controls_discriminate=False))[0] == "C"
    assert C.decide_case(_pre(attribution=False, recovered={c: False for c in C.ALL_CASES},
                              mixed_held_ok=False))[0] == "C"


# --- consumer isolation (static) -----------------------------------------------------------------------------------
ALLOWED_ADAPTER_IMPORTS = {"__future__", "hashlib", "json", "pathlib", "bpy", "bpy_extras", "numpy", "io_scene_fbx"}


def test_adapter_imports_and_writes_are_restricted():
    src = (ROOT/"scripts/w110_fbx_consumer_adapter.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert mods <= ALLOWED_ADAPTER_IMPORTS, mods - ALLOWED_ADAPTER_IMPORTS
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    body = code.split('"""', 2)[2]                      # skip the module docstring
    for banned in ("open_mainfile", "save_as_mainfile", "load(", ".blend", "census", "declaration", "source",
                   "keyframe_points.add", "keyframe_points.insert", "keyframe_points.remove", ".co[", "handle_",
                   "fcurves.new", "fcurves.remove", ".update()", "import_scene", "export_scene", "write_",
                   "w10", "w110_fbx_metadata", "open("):
        assert banned not in body, banned
    sets = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "foreach_set"]
    assert len(sets) == 1 and sets[0].args[0].value == "interpolation"
    sig = inspect.signature(AD.recover)
    assert list(sig.parameters) == ["fbx_path", "stack_name"]


def test_consume_stage_reads_only_consumer_input():
    import w110_fbx_metadata_recovery_blender as B  # noqa: F401  (module import only; no bpy needed at import)
    src = inspect.getsource(B.consume) + inspect.getsource(B.consume_one)
    for banned in ("declaration", "census", r"(?<![a-z])source", r"\.blend", "open_mainfile", "produce"):
        assert not re.search(banned, src), banned


def test_consumer_input_whitelist():
    import w110_fbx_metadata_recovery as H
    ok = {"schema": "x", "blender_version": "5.1.0",
          "cases": {"c": {"stack": "root|t", "times": [1.0], "nb_times": [], "files": {
              "control": {"path": "C:/x/encode/c/control.fbx", "sha256": "a"},
              "meta": {"path": "C:/x/encode/c/meta.fbx", "sha256": "b"}}}}}
    assert H._consumer_clean(ok)
    assert H._consumer_clean(ok, "C:/x/encode")
    assert not H._consumer_clean(ok, "C:/y")
    w108 = json.loads(json.dumps(ok))
    w108["cases"]["c"]["stack"] = "root|W108 SYN step CONSTANT"     # historical take names are legitimate
    assert H._consumer_clean(w108)
    bad = json.loads(json.dumps(ok))
    bad["cases"]["c"]["files"]["meta"]["path"] = "C:/x/encode/c/other.fbx"
    assert not H._consumer_clean(bad)
    bad = json.loads(json.dumps(ok))
    bad["cases"]["c"]["source_blend"] = "C:/x/input.blend"
    assert not H._consumer_clean(bad)
    bad = json.loads(json.dumps(ok))
    bad["cases"]["c"]["files"]["meta"]["path"] = "C:/x/source.npz"
    assert not H._consumer_clean(bad)


# --- real inputs (Main-local; skipped elsewhere) --------------------------------------------------------------------
W107_V1 = Path("C:/Users/dna10/animcv-w107/results/probe/complete_control/fbx/V1.fbx")


@pytest.mark.skipif(not W107_V1.exists(), reason="Main-local W107 artifact not present")
def test_real_v1_map_is_complete_and_patchable():
    import w108_fbx_subframe_contract as C8
    import w109_fbx_constant_import_contract as C9
    root, _ = C8.fbx_read(W107_V1)
    fm = AD.fbx_curve_map(root, "root|W91 complete_control FIXED")
    assert fm["status"] == "OK" and len(fm["curves"]) == 468
    assert {r["status"] for r in fm["curves"]} == {"OK"} and all(m["separable"] for m in fm["models"].values())
    data = W107_V1.read_bytes()
    uids = [r["uid"] for r in fm["curves"] if r["group"] == "rotation"][:3]
    new, recs = C9.patch_flags(data, uids, C.CONSTANT_FLAG)
    assert len(new) == len(data) and len(recs) == 3
    assert C9.patch_flags(data, uids, C.SOURCE_FLAG)[0] == data


@pytest.mark.skipif(not (W110_BASE/"results/analysis/analysis_report.json").exists(),
                    reason="W110 stored outputs not present")
def test_stored_declaration_and_analysis_consistent():
    decl_p = W110_BASE/"results/declare/declaration.json"
    sha = (W110_BASE/"results/declare/declaration.sha256").read_text().strip()
    assert C.sha256_file(decl_p) == sha
    decl = json.loads(decl_p.read_text())
    assert decl["outcomes_read"] is False and decl["tolerances"] == C.TOL
    an = json.loads((W110_BASE/"results/analysis/analysis_report.json").read_text())
    assert an["declaration_sha256"] == sha
    assert C.decide_case(an["decision_inputs"]) == (an["CASE"], an["CASE_reason"])
    assert decl["consumer"]["adapter_sha256"] == C.sha256_file(ROOT/"scripts/w110_fbx_consumer_adapter.py")


def test_struct_sanity():
    # the producer flag differs from the source flag in exactly two bytes (W109)
    a, b = struct.pack("<i", C.SOURCE_FLAG), struct.pack("<i", C.CONSTANT_FLAG)
    assert sum(x != y for x, y in zip(a, b)) == 2
