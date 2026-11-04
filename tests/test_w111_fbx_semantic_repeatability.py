"""Focused tests for the W111 FBX semantic-repeatability diagnostic (scripts/w111_*; evidence only)."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))

import w108_fbx_subframe_contract as C8  # noqa: E402
import w111_fbx_semantic_identity as S  # noqa: E402
import w111_fbx_semantic_repeatability_contract as C  # noqa: E402

E = namedtuple("E", ("id", "props", "props_type", "elems"))
W110_BASE = Path("C:/Users/dna10/animcv-w110")
W111_BASE = Path("C:/Users/dna10/animcv-w111")
STACKS = ("root|take one", "root|take two")


# --- tiny Blender-FBX-like tree ------------------------------------------------------------------------------------
def _n(nid, props, types, elems=()):
    return E(nid, list(props), types, list(elems))


def _curve(uid, flag, vals):
    n = len(vals)
    return _n(b"AnimationCurve", [uid, b"\x00\x01AnimCurve", b""], b"LSS", [
        _n(b"Default", [0.0], b"D"), _n(b"KeyVer", [4008], b"I"),
        _n(b"KeyTime", [np.arange(n, dtype=np.int64)*1539538600], b"l"),
        _n(b"KeyValueFloat", [np.asarray(vals, dtype=np.float32)], b"f"),
        _n(b"KeyAttrFlags", [np.array([flag], dtype=np.int32)], b"i"),
        _n(b"KeyAttrRefCount", [np.array([n], dtype=np.int32)], b"i")])


def build(path="C:\\scratch\\a\\source.blend", ts=(2026, 11, 4), uid_offset=0, models=None):
    u = lambda x: x + uid_offset  # noqa: E731
    models = models or [(10, "root", "Null", 0), (11, "aa", "LimbNode", 10), (12, "bb", "LimbNode", 11),
                        (13, "cc", "LimbNode", 12)]
    hdr = _n(b"FBXHeaderExtension", [], b"", [
        _n(b"FBXHeaderVersion", [1003], b"I"),
        _n(b"CreationTimeStamp", [], b"", [_n(k, [v], b"I") for k, v in zip((b"Year", b"Month", b"Day"), ts)]),
        _n(b"SceneInfo", [b"GlobalInfo\x00\x01SceneInfo", b"UserData"], b"SS", [
            _n(b"Properties70", [], b"", [
                _n(b"P", [b"Original|ApplicationName", b"KString", b"", b"", b"Blender"], b"SSSSS"),
                _n(b"P", [S.PATH_PROP, b"KString", b"", b"", path.encode()], b"SSSSS")])])])
    docs = _n(b"Documents", [], b"", [_n(b"Count", [1], b"I"), _n(b"Document", [u(5), b"Scene", b"Scene"], b"LSS", [
        _n(b"RootNode", [0], b"L")])])
    objs, conns = [], []
    for i, st in enumerate(STACKS):
        objs.append(_n(b"AnimationStack", [u(1+2*i), st.encode() + b"\x00\x01AnimStack", b""], b"LSS"))
        objs.append(_n(b"AnimationLayer", [u(2+2*i), st.encode() + b"\x00\x01AnimLayer", b""], b"LSS"))
        conns.append(_n(b"C", [b"OO", u(2+2*i), u(1+2*i)], b"SLL"))
    for uid, name, kind, parent in models:
        objs.append(_n(b"NodeAttribute", [u(uid+100), name.encode() + b"\x00\x01NodeAttribute", kind.encode()], b"LSS"))
        objs.append(_n(b"Model", [u(uid), name.encode() + b"\x00\x01Model", kind.encode()], b"LSS",
                       [_n(b"Version", [232], b"I")]))
        conns.append(_n(b"C", [b"OO", u(uid+100), u(uid)], b"SLL"))
        conns.append(_n(b"C", [b"OO", u(uid), u(parent) if parent else 0], b"SLL"))
    cu = 1000
    for li in range(len(STACKS)):
        for uid, name, kind, _ in models:
            cu += 10
            objs.append(_n(b"AnimationCurveNode", [u(cu), b"R\x00\x01AnimCurveNode", b""], b"LSS"))
            conns.append(_n(b"C", [b"OO", u(cu), u(2+2*li)], b"SLL"))
            conns.append(_n(b"C", [b"OP", u(cu), u(uid), b"Lcl Rotation"], b"SLLS"))
            for ax, ch in enumerate((b"d|X", b"d|Y", b"d|Z")):
                flag = 0x6002 if (li == 0 and name in ("aa", "bb")) else 0x6104
                objs.append(_curve(u(cu+1+ax), flag, [ax + li, 1.0 + uid, 2.0 + 0.5*ax]))
                conns.append(_n(b"C", [b"OP", u(cu+1+ax), u(cu), ch], b"SLLS"))
    takes = _n(b"Takes", [], b"", [_n(b"Current", [b""], b"S")] + [_n(b"Take", [s.encode()], b"S") for s in STACKS])
    return _n(b"", [], b"", [hdr, _n(b"FileId", [b"FooBar"], b"R"), docs, _n(b"Objects", [], b"", objs),
                             _n(b"Connections", [], b"", conns), takes])


def _roundtrip(tree, tmp_path, name="x.fbx"):
    p = tmp_path/name
    p.write_bytes(S.write_fbx(tree, 7400))
    return C8.fbx_read(p)[0]


# --- canonicaliser ------------------------------------------------------------------------------------------------
def test_canonicalize_accepts_and_accounts():
    can = S.canonicalize(build())
    acc = can["accounting"]
    assert acc["objects_by_class"]["AnimationCurve"] == 24 and acc["curves_with_identity"] == 24
    assert acc["curves_by_stack"] == {STACKS[0]: 12, STACKS[1]: 12}
    assert acc["excluded"]["source_path_values"] == 1 and acc["excluded"]["creation_timestamp_elements"] == 1
    assert acc["excluded"]["object_uid_values"] == acc["objects"]
    assert len(set(can["uid_of"])) == acc["objects"]
    assert can["source_path"].endswith("source.blend")


def test_writer_roundtrip_is_exact(tmp_path):
    t = build()
    r = _roundtrip(t, tmp_path)
    assert S.compare(S.canonicalize(t), S.canonicalize(r))["equal"]
    a, b = S.canonicalize(t), S.canonicalize(r)
    assert a["order"] == b["order"] and a["conn_order"] == b["conn_order"] and a["timestamp"] == b["timestamp"]


def test_invariant_to_uids_order_timestamp_and_path():
    a = S.canonicalize(build())
    b0 = build(path="D:\\other\\place\\renamed.blend", ts=(2027, 1, 2), uid_offset=99_000)
    b0.elems[3].elems.reverse()
    b0.elems[4].elems.reverse()
    b = S.canonicalize(b0)
    assert S.compare(a, b)["equal"]
    rv = S.raw_variation(a, b)
    assert rv["uid_changed"] == rv["identities_compared"] and rv["document_uid_changed"]
    assert not rv["object_order_equal"] and not rv["connection_order_equal"]
    assert rv["timestamp_differs"] and rv["source_path_differs"]
    same = S.raw_variation(a, S.canonicalize(build()))
    assert same["uid_changed"] == 0 and same["object_order_equal"] and not same["source_path_differs"]


def test_semantic_differences_are_detected_and_localised():
    a = S.canonicalize(build())
    t = build()
    curve = [e for e in t.elems[3].elems if e.id == b"AnimationCurve"][5]
    curve.elems[3].props[0] = curve.elems[3].props[0] + np.float32(1e-3)
    cmp = S.compare(a, S.canonicalize(t))
    assert not cmp["equal"] and cmp["n_objects_differ"] == 1 and cmp["differences"] == 1
    t = build()
    t.elems[0].elems[0].props[0] = 1004                  # a retained header value
    cmp = S.compare(a, S.canonicalize(t))
    assert cmp["header_sections_differ"] == [0] and cmp["differences"] == 1


@pytest.mark.parametrize("name", list(S.CONTROLS))
def test_every_control_is_discriminating_on_synthetic_tree(name, tmp_path):
    t = build()
    can = S.canonicalize(t)
    mt, rec = S.mutate(t, can, STACKS[0], name)
    r = _roundtrip(mt, tmp_path)
    mcan, rej = S.try_canonicalize(r)
    j = S.judge_control(rec, can, mcan, rej)
    assert j["ok"], (name, j)
    # the base tree is never modified by a mutation
    assert S.compare(can, S.canonicalize(t))["equal"]


def test_judge_flags_masked_and_non_local():
    t = build()
    can = S.canonicalize(t)
    _, rec = S.mutate(t, can, STACKS[0], "NC2_key_value")
    assert S.judge_control(rec, can, can, None)["outcome"] == "MASKED"
    t2 = build()
    for e in t2.elems[3].elems:          # change TWO curves: not the declared one-record change
        if e.id == b"AnimationCurve":
            e.elems[3].props[0] = e.elems[3].props[0] + np.float32(1.0)
    j = S.judge_control(rec, can, S.canonicalize(t2), None)
    assert j["outcome"] == "DETECTED_NOT_LOCAL" and not j["ok"]


@pytest.mark.parametrize("edit,code", [
    ("dup_model", "DUPLICATE_NAMED_IDENTITY"), ("missing_ref", "UNRESOLVED_CONNECTION"),
    ("uid_leak", "UNEXPECTED_UID_REFERENCE"), ("two_parents", "AMBIGUOUS_OR_MISSING_RELATION"),
    ("orphan_curve", "AMBIGUOUS_OR_MISSING_RELATION"), ("dup_curve_node", "NON_INJECTIVE_IDENTITY"),
    ("mesh", "UNEXPECTED_OBJECT_CLASS"), ("no_timestamp", "TIMESTAMP_COUNT"), ("bad_path", "SOURCE_PATH_UNEXPECTED_VALUE"),
    ("two_paths", "SOURCE_PATH_FIELD_COUNT")])
def test_rejections(edit, code):
    t = build()
    objs, conns = t.elems[3].elems, t.elems[4].elems
    if edit == "dup_model":
        [e for e in objs if e.id == b"Model" and e.props[0] == 13][0].props[1] = b"aa\x00\x01Model"
    elif edit == "missing_ref":
        conns.append(_n(b"C", [b"OO", 11, 987654], b"SLL"))
    elif edit == "uid_leak":
        t.elems[2].elems[1].elems[0].props[0] = 11
    elif edit == "two_parents":
        conns.append(_n(b"C", [b"OO", 13, 11], b"SLL"))
    elif edit == "orphan_curve":
        objs.append(_curve(5555, 0x6104, [1.0]))
    elif edit == "dup_curve_node":
        objs.append(_n(b"AnimationCurveNode", [7777, b"R\x00\x01AnimCurveNode", b""], b"LSS"))
        conns.append(_n(b"C", [b"OO", 7777, 2], b"SLL"))
        conns.append(_n(b"C", [b"OP", 7777, 11, b"Lcl Rotation"], b"SLLS"))
    elif edit == "mesh":
        objs.append(_n(b"Geometry", [8888, b"m\x00\x01Geometry", b"Mesh"], b"LSS"))
    elif edit == "no_timestamp":
        del t.elems[0].elems[1]
    elif edit == "bad_path":
        t.elems[0].elems[2].elems[0].elems[1].props[4] = b"relative/source.blend"
    elif edit == "two_paths":
        p70 = t.elems[0].elems[2].elems[0]
        p70.elems.append(_n(b"P", [S.PATH_PROP, b"KString", b"", b"", b"C:\\x.blend"], b"SSSSS"))
    can, rej = S.try_canonicalize(t)
    assert can is None and rej["code"] == code


def test_keytime_arrays_are_not_uid_leaks_but_other_int64_arrays_are():
    t = build()
    curve = [e for e in t.elems[3].elems if e.id == b"AnimationCurve"][0]
    curve.elems[2].props[0] = np.array([0, 11, 12], dtype=np.int64)        # KeyTime holding UID-like values
    assert S.try_canonicalize(t)[1] is None
    curve.elems.append(_n(b"Extra", [np.array([11], dtype=np.int64)], b"l"))
    assert S.try_canonicalize(t)[1]["code"] == "UNEXPECTED_UID_REFERENCE"


def test_normalize_plan_maps_uids_and_counts_unmapped():
    can = S.canonicalize(build())
    inv = {u: k for k, u in can["uid_of"].items()}
    plan = {"stack": STACKS[0], "status": "OK", "items": {"11": {"model": 11, "name": "aa"}},
            "groups": [{"model": 11, "group": "rotation", "fbx_curves": [1022, 1021], "directive": "CONSTANT"}],
            "curve_status": {"1021": "CONSTANT", "424242": "UNRESOLVED"}, "accounting": {"fbx_curves": 2}}
    n = S.normalize_plan(plan, can["uid_of"])
    assert list(n["items"]) == [inv[11]] and n["groups"][0]["fbx_curves"] == sorted([inv[1021], inv[1022]])
    assert n["_unmapped_uids"] == 1 and "<UNMAPPED:424242>" in n["curve_status"]
    plan2 = json.loads(json.dumps(plan))
    can2 = S.canonicalize(build(uid_offset=50_000))
    plan2["items"] = {"50011": {"model": 50011, "name": "aa"}}
    plan2["groups"][0].update({"model": 50011, "fbx_curves": [51021, 51022]})
    plan2["curve_status"] = {"51021": "CONSTANT", "424242": "UNRESOLVED"}
    assert S.normalize_plan(plan2, can2["uid_of"]) == n


# --- contract -------------------------------------------------------------------------------------------------------
def test_matrix_is_two_by_two_plus_same_cell_repeat():
    cells = {c: (s, p) for c, s, p in C.CELLS}
    assert len(C.CELLS) == 5 and len(set(cells.values())) == 4 and cells["s1_pA_r"] == cells["s1_pA"]
    assert len(set(C.SEEDS.values())) == 2 and len(set(C.PATHS.values())) == 2
    for a, b in C.PAIRS["seed_only"]:
        assert cells[a][0] != cells[b][0] and cells[a][1] == cells[b][1]
    for a, b in C.PAIRS["path_only"]:
        assert cells[a][0] == cells[b][0] and cells[a][1] != cells[b][1]
    assert set(C.PRIMARY_CASES) <= set(C.CASES) and set(C.CONTROL_CASES) <= set(C.CASES)
    assert C.TOL["recovered_m"] == 1e-3 and C.TOL["deterministic_m"] == 1e-9


def test_producer_env_sets_only_the_seed():
    env = C.producer_env({"PATH": "x", "PYTHONPATH": "evil", "pythonhome": "y", "OTHER": "z"}, 1110111)
    assert env == {"PATH": "x", "OTHER": "z", "PYTHONHASHSEED": "1110111", "PYTHONNOUSERSITE": "1"}
    assert "--python-use-system-env" in C.PRODUCER_LAUNCH and "--python-use-system-env" not in C.CONSUMER_LAUNCH
    assert "PYTHONHASHSEED" not in C.consumer_env({"PYTHONHASHSEED": "1", "PATH": "x"})


def _pre(**over):
    pre = {k: True for k, _ in C.D_CHECKS}
    cases = {c: True for c in C.CASES}
    pre.update({k: dict(cases) for k in ("semantic_identical", "plans_identical", "scene_identical", "motion_stable",
                                         "motion_gate", "coverage")})
    pre["motion_material"] = {c: False for c in C.CASES}
    pre.update(controls_discriminate=True, variation_realized=True, attribution_clean=True)
    for k, v in over.items():
        if isinstance(pre.get(k), dict) and not isinstance(v, dict):
            pre[k] = {**pre[k], C.MIXED: v}
        else:
            pre[k] = v
    return pre


def test_case_mapping():
    assert C.decide_case(_pre())[0] == "A"
    for k, _ in C.D_CHECKS:
        assert C.decide_case(_pre(**{k: False}))[0] == "D"
    assert C.decide_case(_pre(semantic_identical=False))[0] == "B"
    assert C.decide_case(_pre(plans_identical=False))[0] == "B"
    assert C.decide_case(_pre(motion_material=True, motion_stable=False))[0] == "B"
    assert C.decide_case(_pre(motion_stable=False))[0] == "C"
    assert C.decide_case(_pre(variation_realized=False))[0] == "C"
    assert C.decide_case(_pre(attribution_clean=False))[0] == "C"
    assert C.decide_case(_pre(motion_gate=False))[0] == "C"
    assert C.decide_case(_pre(semantic_identical=False, repeat_ok=False))[0] == "D"


# --- isolation / whitelist -------------------------------------------------------------------------------------------
def test_consumer_input_whitelist(tmp_path):
    import w111_fbx_semantic_repeatability as H
    root = tmp_path/"encode"
    f = root/"s1_pA"/"syn_step"/"meta.fbx"
    ci = {"schema": "x", "blender_version": "5.1.0", "variants": {"s1_pA:syn_step": {
        "cell": "s1_pA", "case": "syn_step", "stack": "root|W108 SYN step CONSTANT", "times": [1.0], "nb_times": [],
        "fbx": {"path": str(f), "sha256": "0"*64}}}}
    assert H.consumer_clean(ci, root)
    for bad in ("C:/x/source.blend", str(root/"s1_pA"/"syn_step"/"control.fbx"), str(tmp_path/"other"/"meta.fbx")):
        c2 = json.loads(json.dumps(ci))
        c2["variants"]["s1_pA:syn_step"]["fbx"]["path"] = bad
        assert not H.consumer_clean(c2, root)
    c3 = json.loads(json.dumps(ci))
    c3["variants"]["s1_pA:syn_step"]["census"] = {}
    assert not H.consumer_clean(c3, root)


def test_consume_stage_is_source_free_and_uses_the_unchanged_adapter():
    src = (ROOT/"scripts/w111_fbx_semantic_repeatability_blender.py").read_text(encoding="utf-8")
    ns = {}
    import ast
    tree = ast.parse(src)
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("consume", "consume_one", "scene_digests"):
            ns[fn.name] = ast.get_source_segment(src, fn)
    for name, body in ns.items():
        for tok in ("declaration", "census", ".blend", "source_record", "open_mainfile", "srcpaths", "source.npz"):
            assert tok not in body, (name, tok)
    assert "A.recover(str(fbx), stack)" in ns["consume_one"]
    assert "import w110_fbx_consumer_adapter as A" in src
    assert "w111" not in (ROOT/"scripts/w110_fbx_consumer_adapter.py").read_text(encoding="utf-8")


def test_w111_code_does_not_touch_production_modules():
    for p in ("w111_fbx_semantic_identity.py", "w111_fbx_semantic_repeatability_contract.py",
              "w111_fbx_semantic_repeatability_blender.py", "w111_fbx_semantic_repeatability.py"):
        body = (ROOT/"scripts"/p).read_text(encoding="utf-8")
        assert "from blender" not in body and "import blender" not in body
        assert "write_text" not in body or p.endswith(("_blender.py", "repeatability.py"))


# --- real data (skipped when the historical artifacts are absent) -------------------------------------------------
@pytest.mark.skipif(not (W110_BASE/"results/inputs/complete_control_V1.fbx").exists(), reason="W110 artifacts absent")
def test_real_historical_v1_canonicalises_with_full_curve_identity():
    root, _ = C8.fbx_read(W110_BASE/"results/inputs/complete_control_V1.fbx")
    can = S.canonicalize(root)
    acc = can["accounting"]
    assert acc["curves"] == acc["curves_with_identity"] == 7488
    assert acc["curves_by_stack"]["root|W91 complete_control FIXED"] == 468
    assert acc["excluded"]["source_path_values"] == 1


@pytest.mark.skipif(not (W111_BASE/"results/analysis/analysis_report.json").exists(), reason="no stored W111 output")
def test_stored_declaration_and_analysis_consistent():
    d = W111_BASE/"results/declare/declaration.json"
    decl = json.loads(d.read_text())
    sha = hashlib.sha256(d.read_bytes()).hexdigest()
    assert decl["outcomes_read"] is False
    assert (W111_BASE/"results/declare/declaration.sha256").read_text().strip() == sha
    rep = json.loads((W111_BASE/"results/analysis/analysis_report.json").read_text())
    assert rep["declaration_sha256"] == sha
    assert C.decide_case(rep["decision_inputs"]) == (rep["CASE"], rep["CASE_reason"])
    assert inspect.getsource(C.decide_case).strip()
