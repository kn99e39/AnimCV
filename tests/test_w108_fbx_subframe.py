"""W108 FBX subframe interpolation ownership: contract, FBX decoding, metric, CASE and stored-output tests (no Blender)."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w108_fbx_subframe_contract as C  # noqa: E402

BASE = Path(os.environ.get("W108_BASE", "C:/Users/dna10/animcv-w108"))
DECL = BASE/"results/declare/declaration.json"
ANALYSIS = BASE/"results/analysis/analysis_report.json"
W107_FBX = Path(os.environ.get("W107_BASE", "C:/Users/dna10/animcv-w107"))/"results/probe/complete_control/fbx/V1.fbx"


# --- sample grid and interval classes ------------------------------------------------------------------------------
def test_sample_grid_quarters_inside_range_and_contains_w107_grid():
    t = C.sample_times(1, 4)
    assert t == [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
    assert len(C.sample_times(1, 70)) == 4*69 + 1
    w107 = sorted([float(f) for f in range(1, 71)] + [f + 0.5 for f in range(1, 70)])   # W106 sample_times
    assert set(w107) <= set(C.sample_times(1, 70))
    assert list(C.phase_of([3.0, 3.25, 3.5, 3.75])) == [0.0, 0.25, 0.5, 0.75]


def test_interval_classes_from_source_keys_only():
    cls = C.interval_classes(1, 10, [1.0, 4.0, 7.0, 10.0])
    assert cls == {1: "quiet", 2: "quiet", 3: "transition", 4: "quiet", 5: "quiet", 6: "transition", 7: "quiet",
                   8: "quiet", 9: "transition"}
    assert set(C.interval_classes(1, 5, [1.0, 2.0, 3.0, 4.0, 5.0]).values()) == {"transition"}
    assert C.time_classes([3.0, 3.5, 4.25], cls) == ["integer", "transition", "quiet"]


# --- FBX key attribute decoding ------------------------------------------------------------------------------------
def test_decode_interp_bits():
    blender_export_flag = 1 << 2 | 1 << 8 | 1 << 13 | 1 << 14     # literal written by io_scene_fbx 5.15 exporter
    assert C.decode_interp(blender_export_flag) == "LINEAR"
    assert C.decode_interp(0x2) == "CONSTANT" and C.decode_interp(0x2 | 0x100) == "CONSTANT"
    assert C.decode_interp(0x8 | 0x400) == "CUBIC"
    assert C.decode_interp(0x6).startswith("UNKNOWN")


# minimal binary FBX 7400 writer for the reader test (format: header, node records, typed props, null record)
def _prop(v):
    if isinstance(v, bytes):
        return b"S" + struct.pack("<I", len(v)) + v
    if isinstance(v, int):
        return b"L" + struct.pack("<q", v)
    if isinstance(v, tuple):                    # (typecode, ndarray, compress)
        t, a, comp = v
        raw = a.tobytes()
        data = zlib.compress(raw) if comp else raw
        return t + struct.pack("<III", a.size, 1 if comp else 0, len(data)) + data
    raise TypeError(v)


def _node(name, props=(), children=(), off=0):
    pb = b"".join(_prop(p) for p in props)
    head_len = 13 + len(name)
    body = b""
    cur = off + head_len + len(pb)
    for ch in children:
        enc = _node(*ch, off=cur)
        body += enc
        cur += len(enc)
    if children:
        body += b"\x00"*13
    end = off + head_len + len(pb) + len(body)
    return struct.pack("<III", end, len(props), len(pb)) + bytes([len(name)]) + name + pb + body


def _fbx_bytes(flags, times):
    tick = C.FBX_KTIME_PER_SECOND//30
    kt = (b"l", np.array(times, dtype=np.int64)*tick, True)
    curve = (b"AnimationCurve", (11, b"\x00\x01AnimCurve", b""), [
        (b"KeyTime", (kt,)), (b"KeyValueFloat", ((b"f", np.arange(len(times), dtype=np.float32), False),)),
        (b"KeyAttrFlags", ((b"i", np.array([flags], dtype=np.int32), False),)),
        (b"KeyAttrRefCount", ((b"i", np.array([len(times)], dtype=np.int32), False),))])
    objects = (b"Objects", (), [(b"AnimationStack", (1, b"root|ACT\x00\x01AnimStack", b"")),
                                (b"AnimationLayer", (2, b"L\x00\x01AnimLayer", b"")),
                                (b"AnimationCurveNode", (3, b"R\x00\x01AnimCurveNode", b"")), curve])
    conns = (b"Connections", (), [(b"C", (b"OO", 2, 1)), (b"C", (b"OO", 3, 2)), (b"C", (b"OP", 11, 3, b"d|X"))])
    out = C.FBX_MAGIC + struct.pack("<I", 7400)
    for n in (objects, conns):
        out += _node(*n, off=len(out))
    return out + b"\x00"*13


def test_independent_reader_and_stack_summary(tmp_path):
    p = tmp_path/"t.fbx"
    p.write_bytes(_fbx_bytes(1 << 2 | 1 << 8 | 1 << 13 | 1 << 14, [1, 2, 3, 4]))
    root, ver = C.fbx_read(p)
    assert ver == 7400
    s = C.fbx_anim_summary(root, 30.0)
    st = s["stacks"]["root|ACT"]
    assert st["curves"] == 1 and st["keys"] == 4 and st["interpolation"] == ["LINEAR"]
    assert st["refcount_covers_keys"] and st["values_match_times"] and st["nonint_key_times"] == 0
    assert (st["frame_min"], st["frame_max"], st["distinct_frames"]) == (1.0, 4.0, 4)
    assert C.select_stack(s["stacks"], "ACT") == "root|ACT"
    p.write_bytes(_fbx_bytes(0x2, [1, 2, 3]))
    assert C.fbx_anim_summary(C.fbx_read(p)[0], 30.0)["stacks"]["root|ACT"]["interpolation"] == ["CONSTANT"]


@pytest.mark.skipif(not W107_FBX.exists(), reason="W107 FBX absent")
def test_independent_reader_parses_a_real_blender_fbx():
    root, ver = C.fbx_read(W107_FBX)
    s = C.fbx_anim_summary(root, 30.0)
    assert s["status"] == "PARSED" and ver >= 7400 and len(s["stacks"]) >= 1
    assert all(st["refcount_covers_keys"] and st["values_match_times"] for st in s["stacks"].values())


# --- boundary rule ---------------------------------------------------------------------------------------------------
def _st(interp, **kw):
    s = {"curves": 10, "keys": 700, "interpolation": interp, "flags_hex": ["0x6104"], "nonint_key_times": 0,
         "refcount_covers_keys": True, "values_match_times": True}
    s.update(kw)
    return s


def test_boundary_rule():
    assert C.boundary(["CONSTANT"], _st(["LINEAR"]), _st(["LINEAR"]), ["LINEAR"])["boundary"] == "EXPORT_SERIALIZATION"
    assert C.boundary(["CONSTANT"], _st(["CONSTANT"]), _st(["CONSTANT"]), ["LINEAR"])["boundary"] == "IMPORT"
    assert C.boundary(["CONSTANT"], _st(["CONSTANT"]), _st(["CONSTANT"]), ["CONSTANT"])["boundary"] == "NONE_OBSERVED"
    assert C.boundary(["CONSTANT"], _st(["LINEAR"]), _st(["LINEAR"], keys=699), ["LINEAR"])["boundary"] == \
        "UNRESOLVED_FBX_FLAGS_UNRELIABLE"
    assert C.boundary(["CONSTANT"], _st(["LINEAR"], refcount_covers_keys=False), _st(["LINEAR"]),
                      ["LINEAR"])["boundary"] == "UNRESOLVED_FBX_FLAGS_UNRELIABLE"
    assert C.boundary(["CONSTANT"], None, _st(["LINEAR"]), ["LINEAR"])["boundary"] == "UNRESOLVED_FBX_FLAGS_UNRELIABLE"
    assert C.boundary(["LINEAR"], _st(["LINEAR"]), _st(["LINEAR"]), ["LINEAR"])["boundary"] == \
        "NOT_APPLICABLE_SOURCE_NOT_HELD"


# --- metrics ---------------------------------------------------------------------------------------------------------
def _fake(times, B=3, seed=0):
    rng = np.random.default_rng(seed)
    n = len(times)
    return {"times": np.array(times), "bones": np.array([f"b{i}" for i in range(B)]),
            "world_head": rng.normal(size=(n, B, 3)), "world_tail": rng.normal(size=(n, B, 3)),
            "world_rot": np.tile(np.eye(3), (n, B, 1, 1)), "object_matrix": np.tile(np.diag([.01, .01, .01, 1.]), (n, 1, 1)),
            "local_quat": np.tile([1.0, 0, 0, 0], (n, B, 1)), "local_loc": np.zeros((n, B, 3)),
            "local_scale": np.ones((n, B, 3))}


def _imp(a):
    b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
    b["bones"] = np.array(list(a["bones"]) + ["x_end"])
    for k in ("world_head", "world_tail", "world_rot", "local_quat", "local_loc", "local_scale"):
        b[k] = np.concatenate([a[k], np.zeros_like(a[k][:, :1])], axis=1)
    return b


def test_subframe_report_by_phase_and_interval_class():
    times = C.sample_times(1, 4)
    cls = C.interval_classes(1, 4, [1.0, 3.0, 4.0])   # [1,2) quiet, [2,3) transition, [3,4) transition
    a = _fake(times)
    b = _imp(a)
    i_int = times.index(2.0)
    i_q = times.index(1.25)
    i_t = times.index(2.5)
    b["world_tail"][i_int, 0, 0] += 2e-6
    b["world_head"][i_q, 1, 1] -= 3e-4
    b["world_tail"][i_t, 2, 2] += 0.4
    r = C.subframe_report(a, b, times, cls)
    assert r["bones_compared"] == 3
    assert r["integer"]["max_abs_coord_m"] == pytest.approx(2e-6)
    assert r["phase_0.25"]["max_abs_coord_m"] == pytest.approx(3e-4)
    assert r["phase_0.50"]["max_abs_coord_m"] == pytest.approx(0.4)
    assert r["phase_0.75"]["max_abs_coord_m"] == 0.0
    assert r["subframe_quiet"]["max_abs_coord_m"] == pytest.approx(3e-4)
    assert r["subframe_transition"]["max_abs_coord_m"] == pytest.approx(0.4)
    assert r["subframe"]["samples_gt_1mm"] == 1 and r["subframe"]["worst_bones"][0][0] == "b2"
    assert r["subframe_quiet"]["head_max_abs_coord_m"] == pytest.approx(3e-4)
    g = C.w107_gate(a, b)
    assert g["gate_max_abs_coord_m"] == pytest.approx(2e-6) and g["half_frame_max_abs_coord_m"] == pytest.approx(0.4)


def test_rotation_angle_and_common_masks():
    r = np.eye(3)
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    assert C.rot_angle_deg(r, rz) == pytest.approx(90.0)
    new = C.sample_times(1, 3)
    old = [1.0, 1.5, 2.0, 2.5, 3.0]
    ma, mb = C.common_time_masks(new, old)
    assert list(np.array(new)[ma]) == old and mb.all()
    a = _fake(new)
    o = {k: (v[ma] if isinstance(v, np.ndarray) and v.ndim and k != "bones" else v) for k, v in a.items()}
    assert C.arrays_max_diff(a, o, mask_a=ma, mask_b=mb) == 0.0


# --- CASE mapping ------------------------------------------------------------------------------------------------
def _pre(**kw):
    ok = {"testable": True, "keydata_ok": True, "integer_unchanged": True, "material": True, "sufficient": True}
    base = {"preflight_ok": True, "sources_ok": True, "deterministic": True, "originals_unchanged": True,
            "synthetic_valid": True, "p0": {c: True for c in C.REAL_CASES},
            "boundary": {c: {"boundary": "EXPORT_SERIALIZATION"} for c in C.REAL_CASES},
            "p2": {c: dict(ok) for c in C.REAL_CASES},
            "synthetic": {k: {"testable": True, "contrast_ok": True, "confounded": False} for k in C.SYN_CONTROLS}}
    base.update(kw)
    return base


def test_case_mapping():
    assert C.decide_case(_pre())[0] == "A"
    for k in ("preflight_ok", "sources_ok", "deterministic", "originals_unchanged", "synthetic_valid"):
        assert C.decide_case(_pre(**{k: False}))[0] == "D"
    assert C.decide_case(_pre(p0={**{c: True for c in C.REAL_CASES}, "real_clip": False}))[0] == "D"
    pre = _pre()
    pre["p2"]["real_clip"]["material"] = False
    assert C.decide_case(pre)[0] == "C"
    pre = _pre()
    for c in C.REAL_CASES:
        pre["p2"][c]["material"] = False
    assert C.decide_case(pre)[0] == "B"
    pre = _pre()
    pre["boundary"]["arm_natural_most"] = {"boundary": "UNRESOLVED_FBX_FLAGS_UNRELIABLE"}
    assert C.decide_case(pre)[0] == "C"
    pre = _pre()
    pre["synthetic"]["syn_smooth"]["confounded"] = True
    assert C.decide_case(pre)[0] == "C"
    pre = _pre()
    pre["p2"]["complete_control"]["integer_unchanged"] = False
    assert C.decide_case(pre)[0] == "C"
    pre = _pre(boundary={c: {"boundary": "IMPORT"} for c in C.REAL_CASES})
    assert C.decide_case(pre)[0] == "A"


def test_fixed_cases_controls_and_intervention():
    assert C.REAL_CASES == ("complete_control", "arm_natural_most", "real_clip")
    assert all(a.endswith(" FIXED") for a in C.REAL_ACTIONS.values())
    assert {k: v["source_interpolation"] for k, v in C.SYN_CONTROLS.items()} == \
        {"syn_step": "CONSTANT", "syn_smooth": "LINEAR"}
    assert C.V1_OVERRIDE == {"bake_anim_simplify_factor": 0.0}
    assert C.TOL["material_reduction_ratio"] == 0.5 and C.TOL["sufficiency_m"] == 1e-3


# --- code guards ---------------------------------------------------------------------------------------------------
def test_w108_scripts_do_not_patch_exporter_or_mutate_keys():
    names = ("w108_fbx_subframe_blender.py", "w108_fbx_subframe.py", "w108_fbx_subframe_contract.py")
    for name in names:
        src = (ROOT/"scripts"/name).read_text(encoding="utf-8")
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    assert not (isinstance(t, ast.Attribute) and t.attr in ("export_fbx", "fbx", "co", "interpolation",
                                                                            "frame_start", "frame_end", "fps")), name
        assert "save_mainfile(" not in src.replace("save_as_mainfile(", "")
        assert "import torch" not in src
        if name == "w108_fbx_subframe_contract.py":
            continue                              # declaration prose only; it never touches Blender data
        for bad in ("keyframe_points.insert", "keyframe_points.remove", "keyframe_points.add", "keyframe_insert",
                    ".update()"):
            assert bad not in src, (name, bad)
    bl = (ROOT/"scripts/w108_fbx_subframe_blender.py").read_text(encoding="utf-8")
    assert bl.count('foreach_set("interpolation"') == 1          # the single interpolation write (P2 and P3 build)
    assert 'set_interpolation(cp, "CONSTANT")' in bl


# --- stored outputs (skipped before the run) -----------------------------------------------------------------------
needs_decl = pytest.mark.skipif(not DECL.exists(), reason="declaration not written yet")
needs_analysis = pytest.mark.skipif(not ANALYSIS.exists(), reason="analysis not written yet")


@needs_decl
def test_stored_declaration_frozen_before_outcomes():
    d = json.loads(DECL.read_text())
    s = hashlib.sha256(DECL.read_bytes()).hexdigest()
    assert (DECL.parent/"declaration.sha256").read_text().strip() == s
    assert d["outcomes_read"] is False
    assert d["exporter"]["v0_kwargs"] == {"use_selection": False, "object_types": ["ARMATURE"], "bake_anim": True}
    assert set(d["cases"]) == set(C.REAL_CASES) and set(d["synthetic_controls"]) == set(C.SYN_CONTROLS)
    for c, case in d["cases"].items():
        tm = case["timing"]
        assert case["times"] == C.sample_times(tm["frame_start"], tm["frame_end"])
    assert d["environment"]["blender_version"] == "5.1.0" and d["environment"]["fbx_addon_version"] == [5, 15, 0]


@needs_analysis
def test_stored_analysis_consistent():
    a = json.loads(ANALYSIS.read_text())
    assert a["declaration_sha256"] == hashlib.sha256(DECL.read_bytes()).hexdigest()
    assert a["CASE"] in "ABCD"
    assert C.decide_case(a["decision_inputs"])[0] == a["CASE"]
    assert all(a["originals_unchanged"][k]["ok"] for k in ("w107", "w106"))
    assert a["shared_tracked_changes"] == []
