"""Focused tests for the W112 FBX within-property interpolation boundary diagnostic (scripts/w112_*; evidence only)."""
from __future__ import annotations

import ast
import inspect
import itertools
import json
import re
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))

import w108_fbx_subframe_contract as C8  # noqa: E402
import w111_fbx_semantic_identity as S  # noqa: E402
import w112_fbx_within_property_contract as C  # noqa: E402
import w112_fbx_within_property as H  # noqa: E402
import w112_fbx_perkey_candidate as CAND  # noqa: E402

E = namedtuple("E", ("id", "props", "props_type", "elems"))
W112_BASE = Path("C:/Users/dna10/animcv-w112")
TPF = 1539538600


# --- declared interventions ---------------------------------------------------------------------------------------
def test_target_interp_specs():
    k = range(9)
    assert {C.target_interp("UNI_C", "R", a, i) for a in range(4) for i in k} == {"CONSTANT"}
    assert {C.target_interp("UNI_L", "L", a, i) for a in range(3) for i in k} == {"LINEAR"}
    assert [C.target_interp("AXIS", "R", a, 0) for a in range(4)] == ["LINEAR", "CONSTANT", "LINEAR", "LINEAR"]
    assert [C.target_interp("AXIS", "L", a, 3) for a in range(3)] == ["CONSTANT", "LINEAR", "LINEAR"]
    assert [C.target_interp("INT", "RR", 2, i) for i in range(4)] == ["CONSTANT", "LINEAR", "CONSTANT", "LINEAR"]
    assert [C.target_interp("INT1", "R", 1, i) for i in range(3)] == ["CONSTANT", "LINEAR", "CONSTANT"]
    assert {C.target_interp("INT1", "R", 0, i) for i in k} == {"LINEAR"}


def test_case_matrix():
    assert set(C.WITHIN_CASES) == {"syn_R_axis", "syn_R_int", "syn_R_int1", "syn_L_axis", "syn_L_int", "syn_L_int1",
                                   "rr_axis", "rr_int"}
    for f in C.FAMILIES:
        assert C.CASES[C.UNIFORM_LINEAR_OF[f]]["spec"] == "UNI_L" and C.CASES[C.UNIFORM_HELD_OF[f]]["spec"] == "UNI_C"
        assert C.CASES[C.UNIFORM_LINEAR_OF[f]]["family"] == f == C.CASES[C.UNIFORM_HELD_OF[f]]["family"]
    assert all("synthetic-on-real-rig" in C.CASES[c]["regime"] for c in C.FAMILY_CASES["RR"] if c != "complete_control")
    assert C.CASES["complete_control"]["regime"].startswith("benchmark_detector_observation")
    assert C.NC_BASE_CASE in C.WITHIN_CASES and C.CASES[C.NC_BASE_CASE]["spec"] == "INT"
    vals = np.array([C.loc_value(k) for k in range(9)])
    assert np.all(np.abs(np.diff(vals, axis=0)) > 1e-3)          # every location axis changes at every key


def test_in_force_and_interval_class():
    fr = [1.0, 4.0, 7.0]
    assert C.in_force(fr, ["CONSTANT", "LINEAR", "CONSTANT"], 0.0) == "CONSTANT"     # constant extrapolation
    assert C.in_force(fr, ["CONSTANT", "LINEAR", "CONSTANT"], 3.0) == "CONSTANT"
    assert C.in_force(fr, ["CONSTANT", "LINEAR", "CONSTANT"], 4.0) == "LINEAR"
    assert C.in_force(fr, ["CONSTANT", "LINEAR", "CONSTANT"], 6.0) == "LINEAR"
    a = {"frames": fr, "interpolation": ["CONSTANT", "LINEAR", "CONSTANT"]}
    b = {"frames": fr, "interpolation": ["LINEAR", "LINEAR", "LINEAR"]}
    assert C.interval_class([a, a], 1) == "held" and C.interval_class([a, b], 5) == "smooth"
    assert C.interval_class([a, b], 2) == "axis_mixed"


# --- producer rule ------------------------------------------------------------------------------------------------
def _fmap(curves):
    return {"curves": [dict(uid=u, status="OK", model_type="LimbNode", model_name=m, group=g) for u, m, g in curves]}


def _census(curves):
    return {"armature_object": "root", "bones": ["b"], "curves": curves}


def _cv(path, idx, interps, frames=(1.0, 4.0, 7.0)):
    return {"data_path": path, "index": idx, "frames": list(frames), "interpolation": list(interps)}


def test_producer_perkey_consensus_obstruction_undriven():
    q = 'pose.bones["b"].rotation_quaternion'
    fm = _fmap([(1, "b", "rotation"), (2, "b", "location")])
    ticks = {1: np.arange(7, dtype=np.int64)*TPF, 2: np.arange(7, dtype=np.int64)*TPF}
    uni = _census([_cv(q, i, ["CONSTANT", "LINEAR", "CONSTANT"]) for i in range(4)])
    p = C.producer_perkey(fm, uni, ticks, 1, 30)
    assert p[1]["kind"] == "PERKEY" and p[1]["per_key"] == ["CONSTANT"]*3 + ["LINEAR"]*3 + ["CONSTANT"]
    assert p[2]["kind"] == "UNDRIVEN"
    mixed = _census([_cv(q, 0, ["LINEAR"]*3), _cv(q, 1, ["CONSTANT"]*3)] + [_cv(q, i, ["LINEAR"]*3) for i in (2, 3)])
    p = C.producer_perkey(fm, mixed, ticks, 1, 30)
    assert set(p[1]["per_key"][:6]) == {"OBSTRUCTED"}
    bez = _census([_cv(q, i, ["BEZIER"]*3) for i in range(4)])
    assert set(C.producer_perkey(fm, bez, ticks, 1, 30)[1]["per_key"]) == {"UNSUPPORTED"}
    bad = {1: np.array([0, TPF, TPF + TPF//2], dtype=np.int64), 2: ticks[2]}
    assert C.producer_perkey(fm, uni, bad, 1, 30)[1]["kind"] == "BAD_KEY_GRID"


def test_rle_expand_roundtrip():
    seq = [0x6002]*3 + [0x6104]*2 + [0x6002]
    runs = C.rle(seq)
    assert runs == [(0x6002, 3), (0x6104, 2), (0x6002, 1)]
    assert C.expand_flags([a for a, _ in runs], [b for _, b in runs]) == seq


# --- FBX-level instrument -----------------------------------------------------------------------------------------
def test_densify_hold_and_lerp():
    t = np.array([0, TPF, 2*TPF], dtype=np.int64)
    v = np.array([0.0, 10.0, 30.0], dtype=np.float32)
    tt, vv = C.densify(t, v, ["CONSTANT", "LINEAR", "LINEAR"])
    assert len(tt) == len(vv) == 2*len(C.PHASES) + 1
    assert list(vv[:5]) == [0.0]*5                                  # held interval
    assert np.allclose(vv[5:10], [10.0, 15.0, 20.0, 25.0, 29.8])     # lerp interval incl. f + 0.99
    assert tt[0] == 0 and tt[-1] == 2*TPF and tt[4] == round(0.99*TPF) and np.all(np.diff(tt) > 0)
    g = C.grid(1, 3)
    assert g == [1.0, 1.25, 1.5, 1.75, 1.99, 2.0, 2.25, 2.5, 2.75, 2.99, 3.0]
    assert np.allclose((tt/TPF) + 1, g)                              # every probe is a key of the evaluation file


def test_evaluation_parts_cover_grid_without_close_keys():
    g = C.grid(1, 25)
    a, b = C.part_times(g, "a"), C.part_times(g, "b")
    assert sorted(a + b) == g and not set(a) & set(b)
    assert min(np.diff(a)) >= 0.25 - 1e-9 and min(np.diff(b)) >= 1 - 1e-9     # never within Blender's 0.01 merge
    t = np.arange(25, dtype=np.int64)*TPF
    for pt, (ph, end) in C.EPARTS.items():
        assert np.allclose(C.dense_ticks(t, ph, end)/TPF + 1, C.part_times(g, pt))


def _fake(times, err):
    """Fake samples: one bone whose head x carries ``err``."""
    T = len(times)
    h = np.zeros((T, 1, 3))
    h[:, 0, 0] = err
    return {"times": np.asarray(times), "bones": np.array(["b"]), "world_head": h, "world_tail": np.zeros((T, 1, 3)),
            "world_rot": np.tile(np.eye(3), (T, 1, 1, 1)), "object_matrix": np.tile(np.eye(4), (T, 1, 1))}


def test_exact_bound_equals_brute_force_over_product_space():
    rng = np.random.default_rng(112)
    times = C.grid(1, 4)                                             # 3 intervals
    iv = C.interval_index(times)
    labels = C.COMBOS3
    # separable: error at a probe of interval j depends only on the label chosen for interval j
    table = {lab: rng.uniform(0, 1e-2, size=len(times)) for lab in labels}
    for lab in labels:
        table[lab][[i for i, f in enumerate(iv) if f is None]] = 1e-4  # integer samples assignment-independent
    src = _fake(times, np.zeros(len(times)))
    by = {lab: _fake(times, table[lab]) for lab in labels}
    b = C.exact_bound(src, by, labels)
    best = np.inf
    for assign in itertools.product(labels, repeat=3):
        e = [table[assign[f - 1]][i] if f is not None else 1e-4 for i, f in enumerate(iv)]
        best = min(best, max(e))
    assert abs(b["bound_m"] - best) < 1e-15 and b["integer_spread_m"] == 0.0


# --- CASE rule ----------------------------------------------------------------------------------------------------
def _pre(**kw):
    W = C.WITHIN_CASES
    p = {k: True for k, _ in C.D_CHECKS}
    p.update({"imp_ok": {c: True for c in W}, "fbx_ok": {c: True for c in W}, "encoded": {c: True for c in W},
              "sf_ok": {c: True for c in W}, "imp_ok_uniL": {f: True for f in C.FAMILIES},
              "fbx_ok_uniL": {f: True for f in C.FAMILIES}})
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k] = {**p[k], **v}
        else:
            p[k] = v
    return p


def test_decide_case_precedence():
    assert C.decide_case(_pre())[0] == "A"
    for k, _ in C.D_CHECKS:
        assert C.decide_case(_pre(**{k: False}))[0] == "D"
    assert C.decide_case(_pre(fbx_ok={"syn_L_axis": False}))[0] == "B"
    assert C.decide_case(_pre(imp_ok={"rr_axis": False}))[0] == "B"
    # attribution guard: the family's uniform-LINEAR control fails the same layer -> not a refutation
    assert C.decide_case(_pre(fbx_ok={"syn_R_int": False}, fbx_ok_uniL={"R": False}))[0] == "C"
    assert C.decide_case(_pre(encoded={"syn_R_axis": False}))[0] == "C"
    assert C.decide_case(_pre(sf_ok={"syn_L_int": False}))[0] == "C"
    assert C.decide_case(_pre(fbx_ok={"syn_L_axis": False}, repeat_ok=False))[0] == "D"


# --- source-free candidate ----------------------------------------------------------------------------------------
ALLOWED_CANDIDATE_IMPORTS = {"__future__", "hashlib", "json", "pathlib", "bpy", "bpy_extras", "numpy", "io_scene_fbx",
                             "w110_fbx_consumer_adapter"}


def test_candidate_imports_and_writes_are_restricted():
    src = (ROOT/"scripts/w112_fbx_perkey_candidate.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert mods <= ALLOWED_CANDIDATE_IMPORTS, mods - ALLOWED_CANDIDATE_IMPORTS
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    body = code.split('"""', 2)[2]
    for banned in ("open_mainfile", "save_as_mainfile", ".blend", "census", "declaration", "source", "w112_fbx_within",
                   "keyframe_points.add", "keyframe_points.insert", "keyframe_points.remove", "handle_", "fcurves.new",
                   "fcurves.remove", ".update()", "import_scene", "export_scene", "write_", "open(", "oracle", ".npz"):
        assert banned not in body, banned
    sets = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "foreach_set"]
    assert len(sets) == 1 and sets[0].args[0].value == "interpolation"
    assert list(inspect.signature(CAND.recover).parameters) == ["fbx_path", "stack_name"]


def _group(fc_names, reasons=("UNSUPPORTED_FLAG_MIXED_OR_MISSING",), coupled=False):
    return {"directive": "UNRESOLVED", "reasons": list(reasons), "coupled": coupled, "item": "b", "group": "rotation",
            "fbx_curves": [1, 2, 3], "action": "a", "slot": "s", "fcurves": [[n, i, 1] for i, n in enumerate(fc_names)]}


def _scene_frames(n, shift=0.0, count=None):
    fr = list(1.0 + np.arange(count or n) + shift)
    return {("a", "s", "q", i): fr for i in range(4)}


def test_candidate_perkey_plan_rules():
    n = 4
    t = np.arange(n, dtype=np.int64)*TPF
    per = ["CONSTANT", "CONSTANT", "LINEAR", "LINEAR"]
    keys = {u: (t, list(per), True) for u in (1, 2, 3)}
    pl = {"groups": [_group(["q"]*4)]}
    r = CAND.perkey_plan({}, keys, pl, _scene_frames(n), 30)[0]
    assert r["status"] == "OK" and r["per_key"] == per
    k2 = dict(keys)
    k2[2] = (t, ["CONSTANT", "LINEAR", "LINEAR", "LINEAR"], True)
    assert CAND.perkey_plan({}, k2, pl, _scene_frames(n), 30)[0]["status"].startswith("UNSUPPORTED_PER_KEY_AXIS")
    k3 = dict(keys)
    k3[3] = (t + np.array([0, 0, 1, 0]), list(per), True)
    assert CAND.perkey_plan({}, k3, pl, _scene_frames(n), 30)[0]["status"] == "KEYTIME_MISMATCH_ACROSS_AXES"
    assert CAND.perkey_plan({}, keys, pl, _scene_frames(n, count=n + 1), 30)[0]["status"] == \
        "IMPORTED_KEY_COUNT_MISMATCH"
    sf = _scene_frames(n)
    sf[("a", "s", "q", 1)] = [1.0, 2.0, 3.01, 4.0]
    assert CAND.perkey_plan({}, keys, pl, sf, 30)[0]["status"] == "IMPORTED_KEY_TIME_MISMATCH"
    assert CAND.perkey_plan({}, {u: (t, None, False) for u in (1, 2, 3)}, pl, sf, 30)[0]["status"] == "BAD_REFCOUNT"
    # only groups left unresolved solely by per-key flag variation are considered; coupled items never
    assert CAND.perkey_plan({}, keys, {"groups": [_group(["q"]*4, reasons=("FLAG_DISAGREEMENT",))]},
                            _scene_frames(n), 30) == []
    assert CAND.perkey_plan({}, keys, {"groups": [_group(["q"]*4, coupled=True)]}, _scene_frames(n), 30) == []


def test_candidate_curve_keys_decodes_runs():
    c = E(b"AnimationCurve", [7, b"\x00\x01AnimCurve", b""], b"LSS", [
        E(b"KeyTime", [np.arange(5, dtype=np.int64)*TPF], b"l", []),
        E(b"KeyAttrFlags", [np.array([0x6002, 0x6104], dtype=np.int32)], b"i", []),
        E(b"KeyAttrRefCount", [np.array([2, 3], dtype=np.int32)], b"i", [])])
    root = E(b"", [], b"", [E(b"Objects", [], b"", [c])])
    t, per, ok = CAND.curve_keys(root, {7})[7]
    assert ok and per == ["CONSTANT"]*2 + ["LINEAR"]*3


# --- producer encoding, instrument files and whitelist ------------------------------------------------------------
def _tree():
    def curve(uid, vals):
        n = len(vals)
        return E(b"AnimationCurve", [uid, b"\x00\x01AnimCurve", b""], b"LSS", [
            E(b"Default", [0.0], b"D", []), E(b"KeyVer", [4008], b"I", []),
            E(b"KeyTime", [np.arange(n, dtype=np.int64)*TPF], b"l", []),
            E(b"KeyValueFloat", [np.asarray(vals, dtype=np.float32)], b"f", []),
            E(b"KeyAttrFlags", [np.array([0x6104], dtype=np.int32)], b"i", []),
            E(b"KeyAttrDataFloat", [np.array([0, 0, 9.4e-30, 0], dtype=np.float32)], b"f", []),
            E(b"KeyAttrRefCount", [np.array([n], dtype=np.int32)], b"i", [])])
    objs = E(b"Objects", [], b"", [curve(11, [0, 1, 2, 3]), curve(12, [5, 5, 5, 5])])
    return E(b"", [], b"", [E(b"FBXHeaderExtension", [], b"", [E(b"FBXVersion", [7400], b"I", [])]), objs])


def test_encode_meta_and_masked_equality(tmp_path):
    root = _tree()
    fmap = {"curves": [{"uid": 11}, {"uid": 12}]}
    perkey = {11: {"kind": "PERKEY", "per_key": ["CONSTANT", "LINEAR", "CONSTANT", "CONSTANT"]},
              12: {"kind": "UNDRIVEN", "per_key": None}}
    m, changed, intended = H.encode_meta(root, fmap, perkey)
    assert changed == [11] and intended[11] == [0x6002, 0x6104, 0x6002, 0x6002] and intended[12] == [0x6104]*4
    p = tmp_path/"m.fbx"
    p.write_bytes(S.write_fbx(m, 7400))
    mr, _ = C8.fbx_read(p)
    assert H.masked_equal(root, mr, [11])[0] and not H.masked_equal(root, mr, [])[0]
    e = H._sel_curves(mr, fmap)[11]
    assert C.expand_flags(*[np.asarray(C8._child(e, n).props[0]) for n in (b"KeyAttrFlags", b"KeyAttrRefCount")]) \
        == intended[11]
    assert np.array_equal(C8._child(e, b"KeyValueFloat").props[0], np.array([0, 1, 2, 3], dtype=np.float32))


def test_build_efile_densifies_every_curve():
    root = _tree()
    fmap = {"curves": [{"uid": 11}, {"uid": 12}]}
    sem = {11: ["CONSTANT"]*4, 12: ["LINEAR"]*4}
    for pt, n in (("a", 3*4 + 1), ("b", 3)):
        e = H._sel_curves(H.build_efile(root, fmap, sem, pt), fmap)
        for u in (11, 12):
            assert len(C8._child(e[u], b"KeyTime").props[0]) == n
            assert int(C8._child(e[u], b"KeyAttrRefCount").props[0][0]) == n
    ea = H._sel_curves(H.build_efile(root, fmap, sem, "a"), fmap)
    eb = H._sel_curves(H.build_efile(root, fmap, sem, "b"), fmap)
    assert list(C8._child(ea[11], b"KeyValueFloat").props[0][:4]) == [0.0]*4                 # held
    assert np.allclose(C8._child(eb[12], b"KeyValueFloat").props[0], [5, 5, 5])
    assert len(C8._child(H._sel_curves(root, fmap)[11], b"KeyTime").props[0]) == 4       # original untouched


def test_consumer_and_candidate_whitelists(tmp_path):
    f = {"path": str(tmp_path/"c"/"meta.fbx"), "sha256": "x"}
    ok = {"schema": "s", "blender_version": "5.1.0", "cases": {"c": {"stack": "root|W112 SYN R int", "times": [1.0],
                                                                  "files": {"meta": f}}}}
    assert H.consumer_clean(ok, tmp_path)
    for bad in ("root|source take", "x.blend", "census"):
        b = json.loads(json.dumps(ok))
        b["cases"]["c"]["stack"] = bad
        assert not H.consumer_clean(b, tmp_path)
    b = json.loads(json.dumps(ok))
    b["cases"]["c"]["files"]["meta"]["path"] = str(tmp_path/"c"/"efbx"/"e_CCC.fbx")
    assert not H.consumer_clean(b, tmp_path)
    cand = {"schema": "s", "blender_version": "5.1.0", "cases": {}, "controls": {
        "NC2_key_time": {"stack": "root|t", "times": [1.0], "fbx": {"path": str(tmp_path/"controls"/"nc2_key_time.fbx"),
                                                                     "sha256": "x"}}}}
    assert H.consumer_clean(cand, tmp_path, H.CANDIDATE_KEYS, ("meta.fbx",))
    assert not H.consumer_clean(cand, tmp_path)                       # wrong key set for the consumer input


def test_compare_runs_ignores_fresh_export_bytes_only():
    a = {"x": {"v1_fbx": {"sha256": "aa", "bytes": 10}, "bound_m": 1e-3, "seconds": 3.0}}
    b = {"x": {"v1_fbx": {"sha256": "bb", "bytes": 10}, "bound_m": 1e-3 + 1e-12, "seconds": 4.0}}
    assert H.compare_runs(a, b) == []
    b["x"]["v1_fbx"]["bytes"] = 11
    assert H.compare_runs(a, b)


def test_negative_control_judgement():
    good_a = {"isolated": True, "integrity_ok": True, "target_curves_changed": 0}
    s1 = {"isolated": True, "integrity_ok": True, "perkey_diff_keys": [4], "diff_vs_base_outside_interval_m": 0.0,
          "err_inside_interval_m": 0.05}
    assert H._judge_nc("NC1_interval_flag", {"S": s1, "A": good_a}, 4)
    assert not H._judge_nc("NC1_interval_flag", {"S": {**s1, "perkey_diff_keys": [4, 5]}, "A": good_a}, 4)
    assert not H._judge_nc("NC1_interval_flag", {"S": {**s1, "diff_vs_base_outside_interval_m": 1e-6}, "A": good_a}, 4)
    s2 = {"isolated": True, "integrity_ok": True, "target_curves_changed": 0,
          "target_perkey_status": ["KEYTIME_MISMATCH_ACROSS_AXES"]}
    assert H._judge_nc("NC2_key_time", {"S": s2, "A": good_a}, 4)
    assert not H._judge_nc("NC2_key_time", {"S": {**s2, "target_perkey_status": ["OK"]}, "A": good_a}, 4)
    s3 = {"isolated": True, "integrity_ok": True, "perkey_equal_to_base": True, "err_at_changed_key_m": 0.004}
    assert H._judge_nc("NC3_key_value", {"S": s3, "A": good_a}, 4)
    assert not H._judge_nc("NC3_key_value", {"S": {**s3, "err_at_changed_key_m": 1e-5}, "A": good_a}, 4)
    assert H._judge_nc("NC4_ambiguous_name", {"S": {"status": "FAILED"}, "A": {"status": "FAILED"}}, 4)
    u = {"target_curves_changed": 0, "target_directives": [], "twin_directives": ["UNRESOLVED"]*3,
         "twin_curves_changed": 0}
    assert H._judge_nc("NC4_ambiguous_name", {"S": u, "A": u}, 4)
    assert not H._judge_nc("NC4_ambiguous_name", {"S": {**u, "target_curves_changed": 2}, "A": u}, 4)
    assert not H._judge_nc("NC4_ambiguous_name", {"S": {**u, "twin_directives": []}, "A": u}, 4)       # not vacuous
    assert not H._judge_nc("NC4_ambiguous_name", {"S": {**u, "twin_directives": ["CONSTANT"]*3}, "A": u}, 4)


def test_source_free_stages_read_only_their_inputs():
    import w112_fbx_within_property_blender as B
    src = inspect.getsource(B.consume) + inspect.getsource(B.candidate) + inspect.getsource(B.candidate_one)
    for banned in ("declaration", "census", r"(?<![a-z_])source", r"\.blend", "open_mainfile", "produce", "oracle_input",
                   r"\.npz\b(?!\")"):
        assert not re.search(banned, src), banned


def test_real_v1_target_curves_are_mappable():
    p = Path("C:/Users/dna10/animcv-w110/results/inputs/complete_control_V1.fbx")
    if not p.exists():
        pytest.skip("Main-local W110 input not present")
    import w110_fbx_consumer_adapter as AD
    root, _ = C8.fbx_read(p)
    fm = AD.fbx_curve_map(root, "root|" + C.REAL_ACTION)
    assert fm["status"] == "OK"
    for prop in ("Lcl Rotation", "Lcl Translation"):
        assert H._target_uids(fm, {"item": C.TARGET_BONE, "fbx_property": prop}) is not None


# --- stored outputs -----------------------------------------------------------------------------------------------
def test_stored_declaration_and_analysis_consistent():
    d = W112_BASE/"results/declare/declaration.json"
    a = W112_BASE/"results/analysis/analysis_report.json"
    if not (d.exists() and a.exists()):
        pytest.skip("Main-local W112 outputs not present")
    decl = json.loads(d.read_text())
    rep = json.loads(a.read_text())
    assert decl["outcomes_read"] is False
    assert rep["declaration_sha256"] == C.sha256_file(d)
    case, why, detail = C.decide_case(rep["decision_inputs"])
    assert case == rep["case"] and why == rep["reason"]
