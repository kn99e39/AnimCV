"""W106: existing BaseRig Action editability contract (pure contract checks + stored-output checks).

The stored-output checks read the local Main evidence under ``C:/Users/dna10/animcv-w106`` (or ``ANIMCV_W106``) and
skip when it is absent; they never re-run Blender.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w106_action_contract as C  # noqa: E402

W106 = Path(os.environ.get("ANIMCV_W106", "C:/Users/dna10/animcv-w106/results"))


def _close(a, b, tol=1e-12):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# --- quaternion / edit math -------------------------------------------------------------------------------------
def test_offset_and_exact_inverse():
    q = C.qaxis_angle((0.3, -0.5, 0.8), 73.0)
    off = C.e1_offset()
    q2 = C.apply_local_offset(q, off)
    assert abs(C.qangle_deg(q, q2) - C.E1_ANGLE_DEG) < 1e-9
    assert _close(C.apply_inverse_offset(q2, off), q, 1e-12)


def test_post_multiplied_offset_moves_bone_direction_by_exact_angle():
    # rotate the bone axis Y by q then by q*off: the angle between the two directions is exactly the offset angle
    def rot(q, v):
        p = C.qmul(C.qmul(q, (0.0, *v)), C.qconj(q))
        return p[1:]
    for q in (C.qaxis_angle((1, 0, 0), 0.0), C.qaxis_angle((0, 1, 0), 120.0), C.qaxis_angle((0.2, 0.9, -0.4), 47.0)):
        a, b = rot(q, (0, 1, 0)), rot(C.apply_local_offset(q, C.e1_offset()), (0, 1, 0))
        ang = math.degrees(math.acos(max(-1, min(1, sum(x*y for x, y in zip(a, b))))))
        assert abs(ang - C.E1_ANGLE_DEG) < 1e-9


def test_quaternion_sign_equivalence_and_canonical_form():
    q = C.qaxis_angle((0, 0, 1), 30.0)
    nq = tuple(-v for v in q)
    assert C.qangle_deg(q, nq) < 1e-6
    assert C.qcanon(nq) == C.qcanon(q)
    assert C.expected_elbow_displacement(0.277712, 20.0) == pytest.approx(0.0964, abs=1e-4)


# --- windows, shift, step model ---------------------------------------------------------------------------------
def test_windows_and_offsets():
    f3 = [1.0 + 3*k for k in range(24)]
    assert C.e1_window(24) == [10, 11, 12, 13, 14]
    assert C.e2_group(24) == [11, 12, 13]
    assert C.e2_offset(f3, C.e2_group(24)) == 1.0
    dense = [float(k) for k in range(1, 200)]
    assert C.e2_group(199) == [98, 99, 100]
    assert C.e2_offset(dense, C.e2_group(199)) == 0.5
    with pytest.raises(C.ContractViolation):
        C.e1_window(2)


def test_shift_refuses_reorder_merge_and_domain():
    f = [1.0, 4.0, 7.0, 10.0, 13.0]
    assert C.shift_keys(f, [1, 2], 1.0, 1, 13) == [1.0, 5.0, 8.0, 10.0, 13.0]
    with pytest.raises(C.ContractViolation):
        C.shift_keys(f, [1, 2], 3.0, 1, 13)          # lands on key 3 (merge)
    with pytest.raises(C.ContractViolation):
        C.shift_keys(f, [0, 1], -1.0, 1, 13)         # leaves the frame domain (negative-key case)
    with pytest.raises(C.ContractViolation):
        C.shift_keys(f, [3, 4], 1.0, 1, 13)          # past frame_end


def test_constant_step_model_and_hold_interval():
    f, v = [1.0, 4.0, 7.0], [10.0, 20.0, 30.0]
    assert [C.step_value(f, v, t) for t in (0.0, 1.0, 3.5, 4.0, 6.9, 7.0, 99.0)] == [10, 10, 10, 20, 20, 30, 30]
    assert C.hold_interval(f, [1]) == (4.0, 7.0)
    assert C.hold_interval(f, [0]) == (-math.inf, 4.0)
    assert C.hold_interval(f, [2]) == (7.0, math.inf)
    # E2 +1 on the middle key delays its whole hold by one frame
    g = C.shift_keys(f, [1], 1.0, 1, 7)
    assert [C.step_value(g, v, t) for t in (4.0, 5.0, 6.0, 7.0)] == [10, 20, 20, 30]
    # half-frame shifts on dense keys: +0.5 delays by one integer frame, -0.5 is invisible at integer frames
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    dv = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = C.shift_keys(d, [1, 2, 3], 0.5, 1, 5)
    m = C.shift_keys(d, [1, 2, 3], -0.5, 1, 5)
    assert [C.step_value(p, dv, t) for t in (1, 2, 3, 4, 5)] == [1, 1, 2, 3, 5]
    assert [C.step_value(m, dv, t) for t in (1, 2, 3, 4, 5)] == [1, 2, 3, 4, 5]


def test_sample_and_render_frames():
    assert C.sample_times(1, 3) == [1.0, 1.5, 2.0, 2.5, 3.0]
    f = [1.0 + 3*k for k in range(24)]
    rf = C.render_frames(f, C.e1_window(24))
    assert rf[0] == f[8] and rf[-1] == f[16]


# --- selection & decision ---------------------------------------------------------------------------------------
def test_moving_window_selection_is_deterministic_and_metadata_only():
    w = {"b": {"identities": ["same_person"], "target_keyed_every_sample": True, "target_range_deg": 40.0},
         "a": {"identities": ["same_person"], "target_keyed_every_sample": True, "target_range_deg": 5.0},
         "complete_control": {"identities": ["same_person"], "target_keyed_every_sample": True, "target_range_deg": 50.0},
         "c": {"identities": ["other_person"], "target_keyed_every_sample": True, "target_range_deg": 90.0},
         "d": {"identities": ["same_person"], "target_keyed_every_sample": True, "target_range_deg": 90.0}}
    chosen, reasons = C.select_moving_window(w)
    assert chosen == "b"
    assert C.select_moving_window(dict(reversed(list(w.items()))))[0] == "b"
    assert "ELIGIBLE" in reasons["b"][0] and reasons["d"] == ["eligible, not first"]


def _pre(**kw):
    base = {"preflight_ok": True, "determinism_ok": True, "synthetic_ok": True,
            "n0_ok": {"x": True, "y": True}, "real": {"x": {"E1": True, "E2": True, "E3": True, "E4": True},
                                                     "y": {"E1": True, "E2": True, "E3": True, "E4": True}},
            "first_failing_layer": {"x": None, "y": None}}
    base.update(kw)
    return base


def test_case_mapping():
    assert C.decide_case(_pre())[0] == "A"
    assert C.decide_case(_pre(preflight_ok=False))[0] == "D"
    assert C.decide_case(_pre(n0_ok={"x": False, "y": True}))[0] == "D"
    assert C.decide_case(_pre(determinism_ok=False))[0] == "D"
    assert C.decide_case(_pre(real={"x": None, "y": None}))[0] == "D"
    bad = _pre(real={"x": {"E1": True, "E2": False, "E3": True, "E4": True}, "y": {"E1": True, "E2": True, "E3": True, "E4": True}},
               first_failing_layer={"x": "timebase", "y": None})
    case, why = C.decide_case(bad)
    assert case == "B" and "timebase" in why
    assert C.decide_case(_pre(synthetic_ok=False))[0] == "C"
    assert C.decide_case(_pre(real={"x": {"E1": True, "E2": True, "E3": True, "E4": True}, "y": None},
                              n0_ok={"x": True}))[0] == "C"


def test_digest_matches_w105_audit_convention():
    ch = [{"data_path": "p", "index": 0, "frames": [1.0], "values": [0.5]}]
    h = hashlib.sha256()
    h.update(b"p[0]")
    h.update(b"1.000000:0.500000000;")
    assert C.channel_digest(ch) == h.hexdigest()


def test_no_production_import_or_gt_in_w106_scripts():
    for name in ("w106_action_contract.py", "w106_action_blender.py", "w106_action_editability.py"):
        src = (ROOT/"scripts"/name).read_text(encoding="utf-8")
        for bad in ("gt_reference", "reference_legs_cam", "GT_REF", "torch", "save_mainfile(", "mmpose"):
            assert bad not in src, (name, bad)
    # the only production import is the unchanged exporter, used for the FBX round trip
    assert "from blender.export import export_fbx" in (ROOT/"scripts/w106_action_blender.py").read_text(encoding="utf-8")


# --- stored outputs (Main-local; skipped when absent) -----------------------------------------------------------
def _need(p):
    if not p.exists():
        pytest.skip(f"stored W106 output absent: {p}")
    return p


def test_stored_declaration_hash_and_predeclared_flag():
    d = _need(W106/"declare"/"declaration.json")
    sha = (W106/"declare"/"declaration.sha256").read_text().strip()
    assert hashlib.sha256(d.read_bytes()).hexdigest() == sha
    decl = json.loads(d.read_text())
    assert decl["edit_outcomes_read"] is False
    assert set(decl["cases"]) >= {"complete_control", "real_clip"} and len(decl["cases"]) >= 3
    assert all(v["ok"] for v in decl["w105_identity"].values())
    for c in decl["cases"].values():
        assert c["action"].endswith("FIXED")  # M_Main slot, never R11


def test_stored_analysis_consistent_with_declaration():
    a = _need(W106/"analysis"/"analysis_report.json")
    rep = json.loads(a.read_text())
    sha = (W106/"declare"/"declaration.sha256").read_text().strip()
    assert rep["declaration_sha256"] == sha
    assert rep["CASE"] in "ABCD"
    assert rep["CASE"] == C.decide_case(rep["decision_inputs"])[0]
    assert rep["determinism"]["compared_conditions"] > 0


def test_stored_originals_unchanged():
    m = _need(W106.parent/"evidence_manifest.json")
    assert json.loads(m.read_text())["all_originals_unchanged"] is True
