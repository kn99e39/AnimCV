"""Focused W105 tests: rig contract invariants, GT isolation, rig metrics/events, selection, decision mapping, and stored-real
checks of the W105 outputs (source identity, alignment, rig equivalence, Action identity, renderer manifests).

Stored-real fixtures run only when the frozen W102 workspace and the W105 outputs exist.
"""
from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import w105_rig_utility as w  # noqa: E402

W102 = Path(os.environ.get("W102_DIR", "C:/Users/dna10/animcv-w102"))
W105 = Path(os.environ.get("W105_RESULTS", "C:/Users/dna10/animcv-w105/results"))
HAVE_W102 = (W102/"evidence_manifest.json").exists()


def _rx(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def skeleton(knee_l=10.0, knee_r=10.0, arm_l=(0.0, 0.0, -1.0)):
    """Canonical (X right, Y depth, Z up) upright skeleton, left side at -X as in the W95 fixtures."""
    j = np.zeros((17, 3))
    j[1], j[4] = (-0.1, 0, 0), (0.1, 0, 0)
    j[7] = (0, -0.02, 0.10); j[8] = j[7]+(0, 0, 0.35); j[9] = j[8]+(0, 0, 0.05); j[10] = j[8]+(0, 0, 0.2)
    j[11], j[14] = j[8]+(-0.18, 0, 0), j[8]+(0.18, 0, 0)
    for (h, k, a), bend in (((1, 2, 3), knee_l), ((4, 5, 6), knee_r)):
        j[k] = j[h]+0.45*np.array([0, 0, -1.0])
        j[a] = j[k]+0.42*(_rx(-bend) @ np.array([0, 0, -1.0]))
    j[12] = j[11]+0.28*np.asarray(arm_l)/np.linalg.norm(arm_l)
    j[13] = j[12]+0.25*np.array([0, 0, -1.0])
    j[15], j[16] = j[14]+(0, 0, -0.28), j[14]+(0, 0, -0.53)
    return j


@pytest.fixture(scope="module")
def rc():
    return w.rig_context()


# ------------------------------------------------------------------ rig contract invariants
def test_rig_measure_common_parent_and_only_legs_consume_s0(rc):
    I = np.eye(3)
    a, b = w.rig_measure(skeleton(), I, I, rc), w.rig_measure(skeleton(knee_l=70.0, arm_l=(0.3, -0.5, -0.3)), I, I, rc)
    assert a["export"]["object"] == b["export"]["object"]
    assert a["export"]["bones"]["spine_01"] == b["export"]["bones"]["spine_01"]
    assert set(a["export"]["bones"]) == {"spine_01", "thigh_l", "calf_l", "thigh_r", "calf_r"}  # no S0 arm bone
    assert a["export"]["bones"]["calf_l"] != b["export"]["bones"]["calf_l"]
    assert a["export"]["bones"]["thigh_r"] == b["export"]["bones"]["thigh_r"]
    # an arm-only source change leaves every W102 export bone identical
    c = w.rig_measure(skeleton(arm_l=(0.3, -0.5, -0.3)), I, I, rc)
    assert json.dumps(c["export"]) == json.dumps(a["export"])


def test_rig_measure_missing_parent_is_unavailable_not_filled(rc):
    assert w.rig_measure(skeleton(), None, np.eye(3), rc) is None
    assert w.rig_measure(skeleton(), np.eye(3), None, rc) is None
    assert w.rig_measure(None, np.eye(3), np.eye(3), rc) is None


def test_rig_measure_deterministic_and_length_ratio(rc):
    I = np.eye(3)
    a, b = w.rig_measure(skeleton(), I, I, rc), w.rig_measure(skeleton(), I, I, rc)
    assert json.dumps(a) == json.dumps(b)
    assert abs(a["legs"]["l"]["source_ratio"]-0.45/0.42) < 1e-9
    assert a["legs"]["l"]["target_ratio"] > 0 and 0 < a["legs"]["l"]["reach_fraction"] <= 1


def test_leg_metrics_identity_fold_and_branch(rc):
    I = np.eye(3)
    G = skeleton(knee_l=20.0)
    ref = w.rig_measure(G, I, I, rc)
    m = w.leg_metrics(ref, ref, G)
    assert max(m["l"]["dir_ref"]+m["r"]["dir_ref"]) < 1e-6 and not m["l"]["fail"] and not m["l"]["fold"]
    folded = w.rig_measure(skeleton(knee_l=130.0), I, I, rc)
    f = w.leg_metrics(folded, ref, G)
    assert f["l"]["fold"] and f["l"]["fail"] and not f["r"]["fail"]
    flip = w.rig_measure(skeleton(knee_l=-40.0), I, I, rc)  # bend to the other side
    ref40 = w.rig_measure(skeleton(knee_l=40.0), I, I, rc)
    k = w.leg_metrics(flip, ref40, skeleton(knee_l=40.0))
    assert k["l"]["knee_wrong"] is True and k["l"]["fail"]


def test_pair_events_require_gt_quiet(rc):
    I = np.eye(3)
    a, b = w.rig_measure(skeleton(knee_l=10.0), I, I, rc), w.rig_measure(skeleton(knee_l=100.0), I, I, rc)
    Gq = skeleton(knee_l=10.0)
    n1, n2 = w.pair_events(a, b, Gq, Gq)       # rig shin step ~90 deg while GT is still
    assert n1 >= 1 and n2 == 0                  # thigh unchanged -> no local thigh event
    n1g, _ = w.pair_events(a, b, Gq, skeleton(knee_l=100.0))  # GT moves the same way: not an event
    assert n1g == 0
    assert w.pair_events(a, a, Gq, Gq) == (0, 0)


def test_s0_left_arm_secondary_only_changes_arm(rc):
    I = np.eye(3)
    a, b = w.s0_left_arm(skeleton(), I, I, rc), w.s0_left_arm(skeleton(arm_l=(0.3, -0.5, -0.3)), I, I, rc)
    assert a is not None and b is not None and a["upper_w"] != b["upper_w"]
    assert a["export"]["object_rotation_armature_xyzw"] == b["export"]["object_rotation_armature_xyzw"]


# ------------------------------------------------------------------ GT isolation (static)
def test_gt_free_construction_reads_no_gt():
    for fn in (w.build_r11_inputs, w.rig_measure, w.s0_left_arm, w.window_case, w.frame_record, w.build_clip, w.cmd_build, w.hmr_frames):
        src = inspect.getsource(fn)
        for banned in ("target_3d", "raw_references", "gt_rig", "\"G\"", "jointPositions", "uv"):
            assert banned not in src, (fn.__name__, banned)
    assert "identity" not in inspect.getsource(w.build_r11_inputs).split('"""')[2]  # rule never consults association labels


# ------------------------------------------------------------------ selection, partition, decision mapping
def test_restoration_partition():
    new = np.zeros((4, 17), bool)
    new[0, 3] = True; new[1, 13] = True; new[2, [2, 12]] = True; new[3, 10] = True
    assert w.leg_restored(new).tolist() == [True, False, True, False]
    assert w.arm_only_restored(new).tolist() == [False, True, False, False]


def test_best_window_deterministic_and_disjoint():
    rows = [{"sid": s, "frame": f} for s in ("a", "b") for f in range(0, 30, 3)]
    g = w.windows_by_sid(rows)
    score = lambda win: sum(rows[i]["sid"] == "b" for i in win)  # noqa: E731
    w1 = w.best_window(g, score, set(), length=4)
    assert all(rows[i]["sid"] == "b" for i in w1) and rows[w1[0]]["frame"] == 0
    w2 = w.best_window(g, score, set(w1), length=4)
    assert not set(w1) & set(w2) and w2 == w.best_window(g, score, set(w1), length=4)
    assert w.centred_window(g, rows, w1[0], set(w1), length=4) is None


def _block(rb_hi=-1.0, fail=(20, 5, 1, 16), fold=(10, 2, 0, 8), rom=(3, 3, 0, 0), p99=(80, 60), split=(-2, -1), src_hi=-5.0):
    ci = lambda hi: {"mean": hi-1, "ci95": [hi-2, hi]}  # noqa: E731
    return {"rig_dir_ref": {"delta": ci(rb_hi), "p99": list(p99)},
            "rig_fail_rows": {"M": fail[0], "R11": fail[1], "new": fail[2], "recovered": fail[3], "rate_change": ci(-0.01 if rb_hi < 0 else 0.01)},
            "false_fold_rows": {"M": fold[0], "R11": fold[1], "new": fold[2], "removed": fold[3]},
            "leg_rom_rows": {"M": rom[0], "R11": rom[1], "new": rom[2], "removed": rom[3]},
            "rig_dir_ref_delta_by_split": {"validation": {"mean": split[0]}, "test": {"mean": split[1]}},
            "source_dMPJPE": ci(src_hi)}


def test_decide_case_mapping():
    quiet = {"dir_step_events": {"delta_per_pair": {"ci95": [-0.1, 0.0]}}, "thigh_step_events": {"delta_per_pair": {"ci95": [-0.1, 0.1]}}}
    assert w.decide_case(_block(), quiet)["CASE"] == "A"
    assert w.decide_case(_block(rb_hi=1.0), quiet)["CASE"] == "B"          # source gain, no rig gain
    assert w.decide_case(_block(fail=(20, 19, 9, 10)), quiet)["CASE"] == "B"  # harm offsets
    assert w.decide_case(_block(split=(1, -2)), quiet)["CASE"] == "C"
    assert w.decide_case(_block(fail=(100, 98, 0, 2)), quiet)["CASE"] == "C"  # not material
    noisy = {"dir_step_events": {"delta_per_pair": {"ci95": [0.01, 0.2]}}}
    g = w.decide_case(_block(), noisy)
    assert g["RIG_HARM"] and g["harm_components"]["temporal"] and g["CASE"] == "B"
    assert w.decide_case(_block(rb_hi=1.0, src_hi=1.0), quiet)["CASE"] == "C"


def test_action_identity_detects_changed_value_on_unchanged_sample():
    ch = lambda vals: {"channels": [{"data_path": "p", "index": 0, "frames": [1.0, 4.0], "values": vals}], "fcurves": 1,  # noqa: E731
                       "keyed_frames": [1.0, 4.0], "interpolation": ["CONSTANT"], "value_digest": str(vals)}
    cases = {"cases": {"x": {"frames": [{"frame_index": 10, "source_changed": False}, {"frame_index": 13, "source_changed": True}]}}}
    audit = {"files": {"x/x.blend": {"scene_fps": 30, "actions": {"W91 x FIXED": ch([0.1, 0.2]), "W91 x RUNTIME": ch([0.1, 0.9])}}}}
    assert w.action_identity(audit, cases)["pass"]
    audit["files"]["x/x.blend"]["actions"]["W91 x RUNTIME"] = ch([0.3, 0.9])
    assert not w.action_identity(audit, cases)["pass"]


# ------------------------------------------------------------------ stored-real checks (W105 outputs)
need_build = pytest.mark.skipif(not (HAVE_W102 and (W105/"build/build_report.json").exists()), reason="W105 build outputs absent")


@need_build
def test_stored_r11_reproduction_and_noop_identity():
    b = json.loads((W105/"build/build_report.json").read_text())
    assert b["r11_w104_rows_fresh_forward_byte_equal"] and b["GT_read"] is False
    n = b["noop_rows_rig_export_identical"]
    assert n["identical"] == n["rows"] > 5000 and n["R11_source_bytes_equal_M"]
    v = json.loads((W105/"verify/verify_report.json").read_text())["checks"]
    assert v["r11_reconstruction"]["R11_input_byte_equal"] and v["rows"]["frame_timestamp_equal_bank"]


@need_build
def test_stored_cases_alignment_and_shared_parent():
    cases = json.loads((W105/"build/cases.json").read_text())["cases"]
    rows = json.loads((W102/"results/evaluation/rows.json").read_text())
    for name, case in cases.items():
        prev = None
        for fr in case["frames"]:
            r = rows[fr["row"]]
            assert (r["p"], r["frame"], r["sample_id"], r["sid"]) == (fr["bank_position"], fr["frame_index"], fr["sample_id"], case["sequence_id"])
            assert abs(r["timestamp"]-fr["time_s"]) < 1e-12
            if prev is not None:
                assert fr["frame_index"] > prev
            prev = fr["frame_index"]
            if len(fr["object_rotation_xyzw"]) == 2:
                assert fr["object_rotation_xyzw"]["FIXED"] == fr["object_rotation_xyzw"]["RUNTIME"]
                assert fr["extra_bone_locals_xyzw"]["FIXED"]["spine_01"] == fr["extra_bone_locals_xyzw"]["RUNTIME"]["spine_01"]
                assert fr["modes"].get("FIXED") == fr["modes"].get("RUNTIME")
                if not fr["source_changed"]:
                    assert fr["extra_bone_locals_xyzw"]["FIXED"] == fr["extra_bone_locals_xyzw"]["RUNTIME"]
            for slot in ("FIXED", "RUNTIME"):  # unavailable slots carry no pose (blanked, never held)
                if fr["yaw_deg"].get(slot) is None:
                    assert slot not in fr["object_rotation_xyzw"]
    assert all(not f["source_changed"] for f in cases["complete_control"]["frames"])


@pytest.mark.skipif(not (W105/"render/blender_report.json").exists(), reason="W105 renders absent")
def test_stored_renderer_manifest_and_actions():
    br = json.loads((W105/"render/blender_report.json").read_text())
    cases = json.loads((W105/"build/cases.json").read_text())["cases"]
    assert br["equivalent_rig_max_rest_difference"] <= 1e-4 and br["interpolation"] == "CONSTANT"
    for name, case in cases.items():
        n = len(case["frames"])
        assert br["cases"][name]["frames"] == n
        for slot in ("FIXED", "RUNTIME"):
            for view in ("video", "side"):
                assert len(list((W105/"render"/name).glob(f"{slot}_{view}_*.png"))) == n
    audit = json.loads((W105/"render_audit.json").read_text())
    res = w.action_identity(audit, json.loads((W105/"build/cases.json").read_text()))
    assert res["pass"]
    for name, case in cases.items():
        acts = audit["files"][f"{name}/{name}.blend"]["actions"]
        f0 = case["frames"][0]["frame_index"]
        keyed = {round(f) for f in acts[f"W91 {name} FIXED"]["keyed_frames"]}
        assert keyed <= {fr["frame_index"]-f0+1 for fr in case["frames"]}  # keys only at original sampled frame indices


@pytest.mark.skipif(not (W105/"analysis/analysis_report.json").exists(), reason="W105 analysis absent")
def test_stored_analysis_denominators_and_labels():
    a = json.loads((W105/"analysis/analysis_report.json").read_text())
    d = a["denominators"]
    assert d["all_rows"] == sum(d["identity"].values()) == 10483
    assert d["partial"] == 1223 and d["partial_leg"]+d["partial_arm_only"] <= d["partial"]
    assert d["partial_rig_eligible"]+sum(d["partial_exclusions"].values()) == d["partial"]
    assert a["CASE"] in "ABCD" and "GT_RIG" in a["regimes"]
    assert a["secondary_S0_left_arm"]["label"].startswith("SECONDARY")
    gt = json.loads((W105/"analysis/gt_reference_cases.json").read_text())["cases"]
    assert all(set(fr["object_rotation_xyzw"]) <= {"RUNTIME"} for c in gt.values() for fr in c["frames"])
