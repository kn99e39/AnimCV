"""W122 focused tests: held-out window enumeration / exclusion / deterministic selection, source-dominated stratification,
the REPAIRED V10 repeatability (declared-telemetry-only exclusion, semantic corruption detected, vacuous exclusions
refused), the REPAIRED V12 DG5 control (verified-REJECT precondition), real-time kinematics, blind mapping / rating
schema / unblinding, the frozen W122 CASE rule, and stored-artefact checks (skipped when the local W121/W122 artefacts
are absent)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]
import w122_realtime_utility_base as AB  # noqa: E402

W121R = Path("C:/Users/dna10/animcv-w121/results")
W122R = Path("C:/Users/dna10/animcv-w122/results")


# ------------------------------------------------------------------------------------------- window selection ---
def _line(n=80, speed=None):
    speed = speed or (lambda k: 0.01)
    p, x = {}, 0.0
    for k in range(n):
        x += speed(k)
        p[k] = [x, 0.0, 0.0]
    return p


def _feas_all(cands, bad=()):
    return {c["capture"]: {e: (["UNREACHABLE"] if c["capture"] in bad and e == "RBpos" else ["SOLVED"] * 9) for e in AB.EDITS}
            for c in cands}


def test_enumeration_excludes_w121_windows_plus_budget_edges_and_track_flags():
    fr = list(range(100))
    wr = _line(100)
    pre = AB.enumerate_windows(fr, wr, [[40, 48]], None, flag_frames=[90])
    for c in pre:
        lo, hi = c["domain"]
        assert c["frames"][0] >= AB.HOLD_EDGE and c["frames"][-1] <= 99 - AB.HOLD_EDGE
        if set(range(lo, hi + 1)) & set(range(32, 57)):
            assert "OVERLAPS_W121_WINDOW_OR_BUDGET" in c["reasons"]
        if lo <= 90 <= hi:
            assert "TRACK_PROXY_FLAG" in c["reasons"]
    ok = [c for c in pre if c["reasons"] == ["F_NOT_RESOLVED"]]
    assert ok and all(not (set(range(c["domain"][0], c["domain"][1] + 1)) & set(range(32, 57))) for c in ok)


def test_feasibility_requires_all_three_edits_and_selection_is_deterministic():
    fr = list(range(120))
    wr = _line(120, lambda k: 0.002 if k < 60 else 0.03)
    pre = AB.enumerate_windows(fr, wr, [], None)
    bad = {pre[0]["capture"]}
    cands = AB.enumerate_windows(fr, wr, [], _feas_all(pre, bad))
    assert any("F_INFEASIBLE" in r for c in cands if c["capture"] in bad for r in c["reasons"])
    ch = AB.select_windows(cands)
    feas = [c for c in cands if c["feasible"]]
    assert ch["LOWER"]["D_m"] == min(c["D_m"] for c in feas)
    assert ch["HIGHER"]["D_m"] > ch["median_D_m"]
    assert ch["HIGHER"]["frames"][0] > ch["LOWER"]["frames"][-1] + 2 * AB.HOLD_EDGE or \
        ch["HIGHER"]["frames"][-1] < ch["LOWER"]["frames"][0] - 2 * AB.HOLD_EDGE
    assert AB.select_windows(copy.deepcopy(cands))["LOWER"]["capture"] == ch["LOWER"]["capture"]


def test_higher_window_is_not_forced():
    fr = list(range(40))
    wr = _line(40)
    pre = AB.enumerate_windows(fr, wr, [], None)
    cands = AB.enumerate_windows(fr, wr, [], _feas_all(pre))
    ch = AB.select_windows(cands)
    assert ch["LOWER"] is not None and ch["HIGHER"] is None and ch["higher_reason"]
    assert AB.select_windows([])["higher_reason"] == "NO_CANDIDATES"


def test_stratum_threshold_is_declared_50mm():
    assert AB.stratum([0.01, 0.049])["stratum"] == "EDIT_COMPARABLE"
    assert AB.stratum([0.01, 0.050])["stratum"] == "SOURCE_DOMINATED"
    assert AB.stratum([])["stratum"] == "EDIT_COMPARABLE"


# ------------------------------------------------------------------------------- V10 repaired repeatability ---
def _records(wall):
    return {"R1": {"x_info": {"planned_frames": [1, 2, 3]},
                   "factors": {f: {"info": {"request_status": "SOLVED", "planned_frames": [1, 2, 3], "wall_s": wall, "frames": {"2": {"status": "REACHED"}}}}
                               for f in ("G0F0", "G1F0")}}}


def test_v10_excludes_only_declared_wall_s_and_detects_semantic_corruption():
    r1, r2 = _records(0.12), _records(0.57)
    assert AB.canon(r1) != AB.canon(r2)
    rr = AB.records_repeatable(r1, r2)
    assert rr["equal"] and rr["non_vacuous"] and rr["n_removed"] == 2 and rr["n_compared_leaves"] > 0
    cor, path = AB.semantic_corruption(r2)
    assert path.endswith("info.request_status")
    rc = AB.records_repeatable(r1, cor)
    assert not rc["equal"] and rc["n_differing"] == 1
    r3 = _records(0.12)
    r3["R1"]["factors"]["G1F0"]["info"]["frames"]["2"]["status"] = "GUIDE_UNREACHABLE_OUTER"
    assert not AB.records_repeatable(r1, r3)["equal"]


@pytest.mark.parametrize("pats", [("*",), ("*.factors.*.info",), ("*.x_info.planned_frames",), ("*.factors.*.info.request_status",),
                                  ("*.factors.*.info.wall_s", "*.factors.*")])
def test_v10_refuses_vacuous_or_semantic_exclusions(pats):
    with pytest.raises(AB.ExclusionError):
        AB.validate_exclusions(pats)
    with pytest.raises(AB.ExclusionError):
        AB.records_repeatable(_records(0.1), _records(0.2), pats)


# ------------------------------------------------------------------------------------ V12 repaired DG5 ---
def _corrupt(kind, spec):
    s = copy.deepcopy(spec)
    if kind == "DY5":
        s["status"] = "ACCEPTED"
    return s


def test_dg5_requires_verified_reject_and_a_status_change():
    with pytest.raises(AB.PreconditionError):
        AB.dg5_inject({"status": "ACCEPTED"}, "ACCEPT", _corrupt)          # the W121 V12 defect: ACCEPT relabelled ACCEPT
    with pytest.raises(AB.PreconditionError):
        AB.dg5_inject({"status": "ACCEPTED"}, "REJECT", _corrupt)          # display already wrong: not a valid precondition
    with pytest.raises(AB.PreconditionError):
        AB.dg5_inject({"status": "REJECTED"}, "REJECT", lambda k, s: dict(s))   # corruption that changes nothing
    s = AB.dg5_inject({"status": "REJECTED"}, "REJECT", _corrupt)
    assert s["status"] == "ACCEPTED"


def test_dg5_on_stored_genuine_reject_fails_truth_check():
    p = W122R / "instrument/instrument.json"
    if not p.exists():
        pytest.skip("local W122 instrument artefact absent")
    d = json.loads(p.read_text())["V12_DG5_repaired_fixture"]["per_variant"]
    assert d["S1_REACH"]["truth_check_fails"] and d["S1_REACH"]["status_before"] == "REJECTED"
    assert d["S1_MID_RBpos"]["executed"] is False and "PRECONDITION" in d["S1_MID_RBpos"]["refused"]


# ------------------------------------------------------------------------------------------ kinematics ---
def test_kinematics_at_30fps_are_unfiltered_finite_differences():
    P = {k: [0.01 * k * k, 0.0, 0.0] for k in range(6)}
    st = AB.steps(P, range(6))
    assert st[1] == pytest.approx(0.01) and st[5] == pytest.approx(0.09)
    kin = AB.kinematics(P, range(6), 30.0)
    assert kin["speed_m_s"][5] == pytest.approx(0.09 * 30)
    assert kin["accel_m_s2"][5] == pytest.approx(0.02 * 30 * 30)


# ----------------------------------------------------------------------------------- blinding / ratings ---
def test_blind_mapping_is_seeded_balanced_enough_and_order_fixed():
    ids = [f"S{i % 3 + 1}_W{i}" for i in range(24)]
    m1, o1 = AB.blind_mapping(ids, "seed-x")
    m2, o2 = AB.blind_mapping(list(reversed(ids)), "seed-x")
    assert m1 == m2 and o1 == o2 and sorted(o1) == sorted(ids)
    assert {x["A"] for x in m1.values()} == {"G0F0", "G1F0"}
    assert all({x["A"], x["B"]} == {"G0F0", "G1F0"} for x in m1.values())
    m3, _ = AB.blind_mapping(ids, "seed-y")
    assert m3 != m1


def test_rating_schema_and_unblind_translation():
    r = {"overall": "A_VISIBLY_BETTER", "dimensions": {d: "SAME" for d in AB.DIMENSIONS}, "evidence": "f12 elbow", "frames_inspected": "all"}
    r["dimensions"]["elbow_plausibility"] = "B_BETTER"
    assert AB.validate_rating(r) == []
    assert "evidence_missing" in AB.validate_rating(dict(r, evidence=" "))
    assert "overall" in AB.validate_rating(dict(r, overall="BETTER"))
    u = AB.unblind(r, {"A": "G1F0", "B": "G0F0"})
    assert u["overall"] == "G1_BETTER" and u["dimensions"]["elbow_plausibility"] == "G0_BETTER" and u["dimensions"]["pin_behavior"] == "SAME"
    u = AB.unblind(r, {"A": "G0F0", "B": "G1F0"})
    assert u["overall"] == "G0_BETTER" and u["dimensions"]["elbow_plausibility"] == "G1_BETTER"
    assert AB.unblind(dict(r, overall="TIE"), {"A": "G0F0", "B": "G1F0"})["overall"] == "TIE"


# ------------------------------------------------------------------------------------------- CASE rule ---
def _h(src, win, guide=True, vis="G1_BETTER", ident="VERIFIED", harm=False, dims=None):
    return {"source": src, "window": win, "identity": ident, "raw_harm": harm,
            "numeric": {"guide_primary": guide, "guide_conditional": False, "floor_primary": False, "floor_conditional": False},
            "visual": None if vis is None else {"overall": vis, "dimensions": dims or {}}}


def test_case_rule_mapping():
    V, Cg = {"V1": True}, {"C1": True}
    assert AB.case_rule({"V1": False}, Cg, {}, False)["case"] == "D"
    assert AB.case_rule(V, {"C1": False}, {}, False)["case"] == "C"
    assert AB.case_rule(V, Cg, {}, True)["case"] == "DATA_BLOCKED"
    assert AB.case_rule(V, Cg, {"a": _h("S1", "NL"), "b": _h("S2", "NL")}, False)["case"] == "A"
    # two windows of ONE source: not A
    assert AB.case_rule(V, Cg, {"a": _h("S1", "NL"), "b": _h("S1", "NH")}, False)["case"] == "B"
    # identity UNKNOWN cannot count toward A
    assert AB.case_rule(V, Cg, {"a": _h("S1", "NL"), "b": _h("S2", "NL", ident="UNKNOWN")}, False)["case"] == "B"
    # numeric-only (tie / unjudgeable / no rating) is at most B
    assert AB.case_rule(V, Cg, {"a": _h("S1", "NL", vis="TIE"), "b": _h("S2", "NL", vis="UNJUDGEABLE")}, False)["case"] == "B"
    # visual-only is at most B
    assert AB.case_rule(V, Cg, {"a": _h("S1", "NL", guide=False), "b": _h("S2", "NL", guide=False)}, False)["case"] == "B"
    # raw-motion harm or a worse elbow dimension blocks utility
    assert AB.case_rule(V, Cg, {"a": _h("S1", "NL", harm=True), "b": _h("S2", "NL")}, False)["case"] == "B"
    assert AB.case_rule(V, Cg, {"a": _h("S1", "NL", dims={"elbow_plausibility": "G0_BETTER"}), "b": _h("S2", "NL")}, False)["case"] == "B"
    # nothing at all
    assert AB.case_rule(V, Cg, {"a": _h("S1", "NL", guide=False, vis="TIE")}, False)["case"] == "C"
    assert AB.case_rule(V, Cg, {"a": _h("S1", "NL", guide=False, vis="G0_BETTER")}, False)["case"] == "C"


def test_raw_harm_rule():
    assert not AB.raw_harm(0.060, 0.050)          # +10 mm, within max(25 %, 10 mm) = 12.5 mm
    assert AB.raw_harm(0.064, 0.050)
    assert not AB.raw_harm(0.018, 0.010)          # +8 mm < 10 mm floor
    assert not AB.raw_harm(None, 0.01)


# ------------------------------------------------------------------------------------ stored artefacts ---
def test_stored_preflight_reexecution_and_authentication():
    p = W122R / "preflight/preflight.json"
    if not p.exists():
        pytest.skip("local W122 preflight artefact absent")
    d = json.loads(p.read_text())
    assert d["stop"] == []
    for k in ("historical", "S1", "S2", "S3"):
        assert d["perception_reexecution"][k]["ok"] and all(d["perception_reexecution"][k]["arrays_byte_equal"].values())
    for sid in ("S1", "S2", "S3"):
        s = d["sources"][sid]
        assert s["ok"] and s["decoded_frames_equal_w121"] and s["fps_is_30"] and s["cadence_ok"]
        assert d["parents"][sid]["ok"] and d["parents"][sid]["probe_frames_bit_equal_w121"]
        assert d["parents"][sid]["source_binding"]["video_subject_identity"].startswith("UNKNOWN")
    assert d["w121_decision"]["case"] == "D" and d["w121_manifest"]["n_mismatched"] == 0


def test_stored_windows_are_disjoint_from_w121_and_reselect_identically():
    p = W122R / "windows/windows.json"
    if not p.exists():
        pytest.skip("local W122 windows artefact absent")
    w = json.loads(p.read_text())
    for sid, s in w["sources"].items():
        ch = AB.select_windows(s["candidates"])
        for tag, key in (("NL", "LOWER"), ("NH", "HIGHER")):
            got = s["chosen"][tag]
            assert (ch[key] is None) == (got is None)
            if got:
                assert ch[key]["frames"] == got["frames"]
                dom = set(range(got["frames"][0] - 8, got["frames"][-1] + 9))
                for lo, hi in s["w121_windows"]:
                    assert not dom & set(range(lo - 8, hi + 9))


def test_stored_realtime_videos_are_30fps_one_frame_per_source_frame():
    p = W122R / "render/render_report.json"
    if not p.exists():
        pytest.skip("local W122 render artefact absent")
    r = json.loads(p.read_text())
    n = 0
    for rid, v in r["videos"].items():
        for kind, x in v.items():
            if isinstance(x, dict) and "fps" in x:
                rep = 4 if "slowed" in kind else 1
                assert x["fps"] == pytest.approx(30.0) and x["frames_decoded"] == rep * len(v["frames"])
                n += 1
    assert n > 0


def test_stored_decision_keeps_denominators_separate():
    p = W122R / "decision/decision.json"
    if not p.exists():
        pytest.skip("local W122 decision artefact absent")
    d = json.loads(p.read_text())
    assert all("_NL_" in r or "_NH_" in r for r in d["held_out"])
    assert not set(d["held_out"]) & set(d["anchors_replication"])
    assert d["w121_official_case"].startswith("D")
