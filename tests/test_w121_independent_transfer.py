"""W121 focused tests: source screen / dedup / fingerprint, time mapping, segment and hold-window selection, frozen CASE rule,
negative-control expectation consistency, per-request isolation, zero budget, factor fairness, hard pins, scope, display truth,
nonconvergence, exact reversal (Python-only rigs) and stored-real provenance checks (skipped when local artefacts are absent)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]
import w116_semantic_pin_base as S  # noqa: E402
import w118_transition_edit_base as XB  # noqa: E402
import w119_endpoint_transition_base as YB  # noqa: E402
import w120_factor_attribution_base as FB  # noqa: E402
import w121_independent_transfer_base as AB  # noqa: E402

BASE = Path("C:/Users/dna10/animcv-w121")


# ------------------------------------------------------------------------------------------- acquisition screen ---
def cand(**kw):
    c = {"title": "File:Jumping jacks demo.webm", "license": "CC BY-SA 4.0", "mime": "video/webm", "duration": 10.0, "w": 1280, "h": 720,
         "desc": "", "bytes": 1000}
    c.update(kw)
    return c


def test_metadata_screen_reasons():
    assert AB.screen_metadata(cand()) == []
    assert "R1_LICENSE_NOT_FREE_OR_UNKNOWN" in AB.screen_metadata(cand(license="CC BY-NC 4.0"))
    assert "R1_LICENSE_NOT_FREE_OR_UNKNOWN" in AB.screen_metadata(cand(license=None))
    assert "R3_DURATION_OUTSIDE_4_60_S" in AB.screen_metadata(cand(duration=200.0))
    assert "R4_RESOLUTION_BELOW_360" in AB.screen_metadata(cand(w=320, h=240))
    assert "R5_TIME_SCALE_NOT_REAL_TIME" in AB.screen_metadata(cand(title="File:Jumping jack slow motion.webm"))
    assert "R5_TIME_SCALE_NOT_REAL_TIME" in AB.screen_metadata(cand(desc="a <b>time-lapse</b> of jumping jacks"))
    assert "R6_TITLE_NOT_DECLARED_HUMAN_ACTIVITY" in AB.screen_metadata(cand(title="File:Waves on the Thames.webm"))
    assert "R2_NOT_DECODABLE_VIDEO_CONTAINER" in AB.screen_metadata(cand(mime="image/gif"))
    for lic in ("CC0", "Public domain", "CC BY 3.0", "CC BY-SA 4.0"):
        assert AB.license_ok(lic)


def test_dedup_catches_copy_retime_and_shift_but_not_other_footage():
    rng = np.random.default_rng(0)
    a = [int(x) for x in rng.integers(0, 2 ** 63, 60, dtype=np.int64)]
    assert AB.longest_matched_run(a, a) == 60
    retimed = [h for i, h in enumerate(a[7:]) for _ in range(2 if i % 3 == 0 else 1)]       # time shift + frame repetition
    assert AB.longest_matched_run(retimed, a) >= AB.DUP_RUN_FRAMES
    noisy = [h ^ (1 << int(rng.integers(0, 63))) for h in a]                                 # 1-bit re-encode noise
    assert AB.longest_matched_run(noisy, a) >= AB.DUP_RUN_FRAMES
    other = [int(x) for x in rng.integers(0, 2 ** 63, 60, dtype=np.int64)]
    assert AB.longest_matched_run(other, a) < AB.DUP_RUN_FRAMES
    assert AB.longest_matched_run([], a) == 0


def test_shot_cut_detection():
    t = [np.full((36, 64), 100, np.uint8)] * 5 + [np.full((36, 64), 200, np.uint8)] * 5
    assert AB.shot_cuts(t) == [5]
    assert AB.shot_cuts([np.full((36, 64), 100, np.uint8)] * 4) == []


# ------------------------------------------------------------------------------------------- segment selection ---
def _seg_inputs(n=150, fps=30.0):
    present = np.ones(n, bool)
    arm = np.ones(n, bool)
    wrist = np.stack([200 + 80 * np.sin(np.arange(n) / 10), 300 + 0 * np.arange(n)], 1)
    torso = np.full(n, 100.0)
    return present, arm, wrist, torso, fps


def test_segment_select_basic_and_cut_split():
    p, arm, w, t, fps = _seg_inputs()
    runs, ch = AB.segment_select(p, arm, w, t, [], fps)
    assert ch is not None and ch["window"] == [0, 149] and ch["eligible"]
    runs, ch = AB.segment_select(p, arm, w, t, [60], fps)       # cut splits into 60 + 90 frames: only the 90-frame run is long enough
    assert ch["window"] == [60, 149]
    assert any("S4_SEGMENT_TOO_SHORT" in r["reasons"] for r in runs)


def test_segment_select_rejects_low_arm_validity_static_arm_and_gaps():
    p, arm, w, t, fps = _seg_inputs()
    arm2 = arm.copy()
    arm2[::3] = False                                           # 66 % valid
    assert AB.segment_select(p, arm2, w, t, [], fps)[1] is None
    still = np.stack([np.full(150, 200.0), np.full(150, 300.0)], 1)
    runs, ch = AB.segment_select(p, arm, still, t, [], fps)
    assert ch is None and "S5_LEFT_WRIST_ACTIVITY_BELOW_0.50_TORSO" in runs[0]["reasons"]
    p2 = p.copy()
    p2[70] = False                                              # subject gap -> two runs of 70 and 79 frames, both too short
    assert AB.segment_select(p2, arm, w, t, [], fps)[1] is None


def test_segment_select_caps_long_runs_at_max_window():
    p, arm, w, t, fps = _seg_inputs(n=400)
    arm = arm.copy()
    arm[:100] = False
    runs, ch = AB.segment_select(p, arm, w, t, [], fps)
    assert ch["frames"] == AB.SEG_MAX_FRAMES and ch["window"][0] >= 100


# ------------------------------------------------------------------------------------------- hold windows ---
def test_hold_windows_low_mid_high_non_overlapping_and_edge_free():
    fps = 30.0
    frames = list(range(100, 300))
    wrist = {k: [0.002 * k if k < 200 else 0.4 + 0.02 * (k - 200), 0.0, 1.0] for k in frames}
    h = AB.hold_windows(frames, wrist, fps)
    L = h["L"]
    assert L == 9
    ch = h["chosen"]
    assert set(ch) == {"LOW", "MID", "HIGH"}
    assert ch["LOW"]["D_m"] <= ch["MID"]["D_m"] <= ch["HIGH"]["D_m"]
    for c in ch.values():
        assert len(c["frames"]) == L and c["frames"][0] - frames[0] >= AB.HOLD_EDGE and frames[-1] - c["frames"][-1] >= AB.HOLD_EDGE
    cs = sorted(ch.values(), key=lambda c: c["frames"][0])
    for x, y in zip(cs, cs[1:]):
        assert y["frames"][0] > x["frames"][-1] + 2 * AB.HOLD_EDGE
    assert AB.hold_windows(frames, wrist, fps) == h                                         # deterministic


# ------------------------------------------------------------------------------------------- CASE rule ---
def _mech(pi=(), ci=(), nd=(), vis=False):
    return {"primary_improved": list(pi), "conditional_improved": list(ci), "primary_new_defects": list(nd), "visible": vis}


def test_case_rule_mapping():
    V, C = {"V1": True}, {"C1": True}
    assert AB.case_rule(V, C, {}, 0, True)["case"] == "DATA_BLOCKED"
    assert AB.case_rule({"V1": False}, C, {}, 1, False)["case"] == "D"
    assert AB.case_rule(V, {"C1": False}, {}, 1, False)["case"] == "C"
    two = [("S1", "LOW", "a"), ("S1", "MID", "b")]
    r = AB.case_rule(V, C, {"GUIDE": _mech(two, vis=True), "FLOOR": _mech()}, 1, False)
    assert r["case"] == "A" and r["scope"] == "ONE-SOURCE TRANSFER"
    # two sources accepted but improvements only in one -> not reproducible across sources -> B
    assert AB.case_rule(V, C, {"GUIDE": _mech(two, vis=True), "FLOOR": _mech()}, 2, False)["case"] == "B"
    # same window twice (edit variants) counts once
    same = [("S1", "LOW", "a"), ("S1", "LOW", "b")]
    assert AB.case_rule(V, C, {"GUIDE": _mech(same, vis=True), "FLOOR": _mech()}, 1, False)["case"] == "B"
    # articulation defect blocks A
    assert AB.case_rule(V, C, {"GUIDE": _mech(two, nd=["c"], vis=True), "FLOOR": _mech()}, 1, False)["case"] == "B"
    # not visible blocks A
    assert AB.case_rule(V, C, {"GUIDE": _mech(two, vis=False), "FLOOR": _mech()}, 1, False)["case"] == "B"
    # conditional-only improvement -> B ; nothing -> C
    assert AB.case_rule(V, C, {"GUIDE": _mech(ci=[("S1", "LOW", "a")]), "FLOOR": _mech()}, 1, False)["case"] == "B"
    assert AB.case_rule(V, C, {"GUIDE": _mech(), "FLOOR": _mech()}, 1, False)["case"] == "C"


def test_negative_expectations_equal_frozen_precedence_predictor():
    import w121_independent_transfer as W21
    r = W21.neg_expectation_consistency()
    assert r["all_ok"], {k: v for k, v in r["controls"].items() if not v["ok"]}


# ------------------------------------------------------------------------------- frozen mechanism on a test rig ---
def _w118_case():
    import test_w118_transition_edit as T8
    return T8, T8.case()


def test_instrument_preflight_reused_unchanged(tmp_path):
    import w120_factor_attribution as W20
    r = W20.fixture_preflight(tmp_path / "s")
    assert r["all_ok"] and r["precedence_exhaustive"]["n_mismatches"] == 0


def test_isolation_zero_budget_fairness_and_nonconvergence():
    T8, (r, P, U, pin, base) = _w118_case()
    FR = list(range(T8.N))
    F = {k: (base[k] if k in base else U[k]) for k in FR}
    assert YB.isolation_check_values(F, U, pin, FR)["ok"]
    F2 = dict(F)
    F2[2] = {c: np.array(v) for c, v in base[pin[0]].items()}
    assert not YB.isolation_check_values(F2, U, pin, FR)["ok"]
    x0, xi0 = XB.transition_edit(r, U, base, pin, FR, T8.AUTH, "lower", m_max=0)
    m0 = YB.margins_of(xi0)
    for f in FB.FACTORS:
        v, i = FB.factor_transition(r, U, base, pin, FR, T8.AUTH, "upper", "lower", m0, 0, f, x_vals={})
        assert i["budget_frames"] == sorted(pin) and i["planned_frames"] == sorted(pin)
    x8, xi8 = XB.transition_edit(r, U, base, pin, FR, T8.AUTH, "lower")
    m8 = YB.margins_of(xi8)
    xv = {k: x8[k] for k in xi8["planned_frames"] if k not in set(pin)}
    v, i = FB.factor_transition(r, U, base, pin, FR, T8.AUTH, "upper", "lower", m8, xi8["m_max"], "G1F0", x_vals=xv, fault="skip_lower_aim")
    if i["frames"]:
        assert i["request_status"] == "NON_CONVERGED"
    rec = {"parent_id": "p", "rig_id": "R", "U_state_id": "u", "constraint_digests": ["c"], "pin_frames": pin, "auth": T8.AUTH,
           "pin_values_digest": "v", "margins": m8, "budget_frames": xi8["budget_frames"]}
    assert FB.fairness_factor(dict(rec, margins=None, budget_frames=None), rec, dict(rec, solver_params_digest=FB.SOLVER_PARAMS_DIGEST))["ok"]
    bad = FB.fairness_factor(dict(rec, margins=None, budget_frames=None), rec, dict(rec, budget_frames=sorted(pin)))
    assert "SEARCH_BUDGET_MISMATCH:Y" in bad["errors"]


def test_display_truth_and_hidden_change_detection():
    T8, (r, P, U, pin, base) = _w118_case()
    FR = list(range(T8.N))
    xv, xi = XB.transition_edit(r, U, base, pin, FR, T8.AUTH, "lower")
    Xv = {k: (xv[k] if k in xv else U[k]) for k in FR}
    Xe = {k: T8.fake_eval(r, Xv[k]) for k in FR}
    Ue = {k: T8.fake_eval(r, U[k]) for k in FR}
    arm = S.domain_points(r, T8.AUTH)
    fp = XB.derive_footprint(Xe, Ue, FR, pin, arm, wrist=("lower", "tail"), elbow=("upper", "tail"))
    spec = XB.display_spec(fp, FR, pin, xi["budget_frames"], xi["planned_frames"], "sid", "pid", "PROPOSED", "r", "X")
    assert XB.check_display(spec, Xe, Ue, FR, pin, "sid")["ok"]
    auto = [k for k, c in spec["cells"].items() if c == "AUTO_CHANGED"]
    if auto:
        bad = json.loads(json.dumps(spec))
        bad["cells"][auto[0]] = "UNCHANGED"
        assert not XB.check_display(bad, Xe, Ue, FR, pin, "sid")["ok"]
    # hidden change outside the budget and on an unauthorised channel is caught by the footprint and decide_x
    Xv2 = dict(Xv)
    Xv2[1] = dict(U[1], base=S.q_mul(U[1]["base"], S.q_axis_angle([1, 0, 0], math.radians(1.0))))
    Xe2 = dict(Xe)
    Xe2[1] = T8.fake_eval(r, Xv2[1])
    fp2 = XB.derive_footprint(Xe2, Ue, FR, pin, arm, wrist=("lower", "tail"), elbow=("upper", "tail"))
    good = {"pins_satisfied": True, "scope_ok": True, "locality_ok": True, "lengths_ok": True, "root_unchanged": True, "joint_limit_ok": True}
    d = XB.decide_x([], ["SOLVED"], [good], fp2, xi["budget_frames"], xi["planned_frames"], T8.AUTH)
    assert d["decision"] == "REJECT" and {"CHANGE_OUTSIDE_AUTOMATIC_BUDGET", "UNAUTHORIZED_CHANNEL_CHANGE"} <= set(d["reasons"])


def test_hard_pin_failure_and_solver_failure_masked_are_rejected():
    good = {"pins_satisfied": True, "scope_ok": True, "locality_ok": True, "lengths_ok": True, "root_unchanged": True, "joint_limit_ok": True}
    fp0 = {"frames_changed": [11], "channel_frames": {"upper": [11]}, "outside_arm_points_changed": {}, "matrix_world_changed_frames": []}
    d = YB.decide_y([], ["SOLVED"], [dict(good, pins_satisfied=False)], fp0, [10, 11, 12], [10, 11, 12], ["upper", "lower"],
                    {"request_status": "SOLVED"}, fairness={"ok": True, "errors": []}, display_check={"ok": True}, isolation={"ok": True})
    assert d["decision"] == "REJECT" and "EVALUATED_PIN_UNSATISFIED" in d["reasons"]
    d = YB.decide_y([], ["UNREACHABLE"], [good], fp0, [10, 11, 12], [10, 11, 12], ["upper", "lower"], {"request_status": "SOLVED"},
                    fairness={"ok": True, "errors": []}, display_check={"ok": True}, isolation={"ok": True})
    assert d["decision"] == "REJECT" and d["failure_class"] == "INFEASIBLE_REQUEST"


def test_exact_reversal_and_inexact_rollback_detection(tmp_path):
    st = S.Store(tmp_path / "s")
    p0 = S.make_state(None, "T", [0, 1], ["a"], {"a": [[0, 0, 0, 1], [0, 0, 0, 1]]}, {"kind": "ROOT"})
    c1 = S.make_state(p0["state_id"], "T", [0, 1], ["a"], {"a": [[0, 0, 0.1, 0.995], [0, 0, 0, 1]]}, {"kind": "C"})
    st.put(p0)
    st.init_ref("m", p0["state_id"])
    assert st.commit("m", p0["state_id"], c1, {"decision": "ACCEPT", "failure_class": "NONE", "reasons": []})["head_after"] == c1["state_id"]
    rv, rec = st.revert("m")
    assert rec["values_equal_restored"] and S.canon(rv["values"]) == S.canon(p0["values"])
    bad = json.loads(json.dumps(rv["values"]))
    bad["a"][0][0] += 1e-6
    assert S.canon(bad) != S.canon(p0["values"])
    r2 = st.commit("m", p0["state_id"], c1, {"decision": "ACCEPT", "failure_class": "NONE", "reasons": []})
    assert r2["reasons"] == ["STALE_PARENT"]


def test_source_state_binds_provenance_fields():
    v = S.make_state(None, "BaseRig", [3, 4], ["a"], {"a": [[0, 0, 0, 1]] * 2}, {"kind": "ROOT"},
                     {"clip_sha256": "x", "time_s": [0.1, 4 / 30], "video_subject_identity": "UNKNOWN"})
    w = json.loads(json.dumps(v))
    w["source"]["video_subject_identity"] = "VERIFIED"
    assert S.verify_state(v) and not S.verify_state(w)


# ------------------------------------------------------------------------------------------- stored real ---
@pytest.mark.skipif(not (BASE / "results/perceive/perceive.json").exists(), reason="W121 local perception artefacts absent")
def test_stored_frozen_reproduction_and_adapter_fidelity():
    p = json.loads((BASE / "results/perceive/perceive.json").read_text(encoding="utf-8"))
    assert p["historical_reproduction"]["ok"]
    assert all(p["historical_reproduction"]["arrays_byte_equal"].values())
    assert p["rig_adapter_fidelity"]["ok"] and p["rig_adapter_fidelity"]["mismatch_frames"] == []


@pytest.mark.skipif(not (BASE / "results/inspect/inspect.json").exists(), reason="W121 local inspection artefacts absent")
def test_stored_inspection_rejects_bogus_cadence_and_finds_no_historical_duplicate():
    r = json.loads((BASE / "results/inspect/inspect.json").read_text(encoding="utf-8"))
    for t, c in r["candidates"].items():
        assert c["dup_vs_historical_longest_run"] < AB.DUP_RUN_FRAMES
        if c["fps"] > 1000:
            assert "S1_DECLARED_FPS_INCONSISTENT_WITH_CONTAINER_TIMESTAMPS" in c["reasons"]
