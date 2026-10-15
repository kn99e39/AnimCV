import math
import re
from pathlib import Path

import numpy as np
import pytest

from framepose.mpi3dhp_test_transfer import (
    CANONICAL_FROM_TEST17, CANONICAL_TEST17, FPS, SHORT_PAIR_FRAMES, SMPLX_FROM_TEST17, TEST17_NAMES, association_uv, bend_side,
    camera_mm_to_animcv, canonical_target, image_name, motion_metrics, regime, select_frames, side_error, static_metrics,
    stride_for,
)
from pose.pose_lifter import H36M_NAMES

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(83)


def test_left_arm_mapping_is_exact_and_anatomical():
    for side in ("left", "right"):
        for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            assert TEST17_NAMES[CANONICAL_FROM_TEST17[f"{side}_{j}"]] == f"{side}_{j}"
    assert TEST17_NAMES[CANONICAL_FROM_TEST17["pelvis"]] == "pelvis"
    assert [TEST17_NAMES[CANONICAL_TEST17[H36M_NAMES.index(n)]] for n in ("left_shoulder", "left_elbow", "left_wrist")] == \
        ["left_shoulder", "left_elbow", "left_wrist"]
    # Worklog 73 association indexes GT by SMPL-X body joint (16 L shoulder, 18 L elbow, 20 L wrist, 17/19/21 right).
    smplx_names = {16: "left_shoulder", 17: "right_shoulder", 18: "left_elbow", 19: "right_elbow", 20: "left_wrist",
                   21: "right_wrist", 1: "left_hip", 2: "right_hip", 4: "left_knee", 5: "right_knee", 7: "left_ankle", 8: "right_ankle"}
    assert {k: TEST17_NAMES[v] for k, v in SMPLX_FROM_TEST17.items()} == smplx_names
    from framepose.bedlam_supervision import COCO_LIMB_TO_SMPLX
    assert set(COCO_LIMB_TO_SMPLX.values()) == set(SMPLX_FROM_TEST17)
    a2 = RNG.normal(size=(17, 2))
    uv = association_uv(a2)
    assert np.allclose(uv[16], a2[5]) and np.allclose(uv[20], a2[7]) and np.isnan(uv[0]).all()


def test_camera_conversion_axes_units_and_root():
    p = np.array([[100.0, -200.0, 3000.0]])
    assert np.allclose(camera_mm_to_animcv(p), [[0.1, 3.0, 0.2]])  # +X right, +Y depth, +Z up (image y is down)
    a3 = RNG.normal(size=(17, 3)) * 300 + np.array([0, 0, 4000.0])
    t, valid = canonical_target(a3)
    assert valid.all() and np.allclose(t[0], 0)
    li, pe = H36M_NAMES.index("left_wrist"), H36M_NAMES.index("pelvis")
    assert np.allclose(t[li], camera_mm_to_animcv(a3[7]) - camera_mm_to_animcv(a3[14]), atol=1e-6)
    a3[7, 2] = -5.0
    assert not canonical_target(a3)[1][li]


def test_population_and_pair_interval_and_frame_identity():
    assert FPS["TS1"] == 25 and FPS["TS6"] == 50 and stride_for("TS1") == 10 and stride_for("TS5") == 20
    assert stride_for("TS1") / FPS["TS1"] == pytest.approx(0.4) and stride_for("TS5") / FPS["TS5"] == pytest.approx(0.4)
    assert SHORT_PAIR_FRAMES == {"TS5": 8, "TS6": 8} and 8 / FPS["TS5"] == pytest.approx(0.16)
    valid = np.zeros((30, 1))
    valid[[0, 10, 20, 21]] = 1.0
    assert select_frames(valid).tolist() == [0, 10, 20, 21]
    assert image_name(0) == "img_000001.jpg" and image_name(1234) == "img_001235.jpg"


def test_pairs_never_cross_a_missing_partner():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_framepose_anchored_arm_transfer_eval import NAMES, collect

    class S:
        def __init__(self, k, seq):
            self.sequence_id, self.frame_index, self.sample_id = seq, k, f"{seq}#{k}"
    frames = [("3dhp:TS1", k) for k in (0, 10, 20, 40, 50)] + [("3dhp:TS5", k) for k in range(0, 25)]
    samples = [S(k, q) for q, k in frames]
    n = len(samples)
    e, w = _arm(30, 50, 0.3)
    pred = {"target_valid": np.ones(n, bool), "target_elbow": np.tile(e, (n, 1)), "target_wrist": np.tile(w, (n, 1)),
            "anchor_elbow": np.tile(e, (n, 1)), "anchor_wrist": np.tile(w, (n, 1)), "anchor_rho": np.full(n, 0.5)}
    for c in NAMES.values():
        pred[f"{c}_elbow"], pred[f"{c}_wrist"] = np.tile(e, (n, 1)), np.tile(w, (n, 1))
        pred[f"{c}_correction"], pred[f"{c}_rho"] = np.zeros((n, 7)), np.full(n, 0.5)
    pred["target_valid"][3] = False          # TS1 frame 40 not evaluated
    rows, pairs, gaps = collect(samples, range(n), pred, lambda sid: stride_for(sid.split(":")[1]))
    got = sorted((x["sequence_id"], x["frame_index"], x["next_frame_index"]) for x in pairs)
    assert got == [("3dhp:TS1", 0, 10), ("3dhp:TS1", 10, 20)] + [("3dhp:TS5", k, k + 20) for k in range(5)]
    assert len(rows) == n - 1 and all(x["candidates"]["r1"]["upper_direction_residual_degrees"] == 0 for x in pairs)


def _arm(theta, bend, fwd):
    t, b = math.radians(theta), math.radians(bend)
    e = np.array([0.3 * math.cos(t), 0.3 * fwd, -0.3 * math.sin(t)])
    l = np.array([math.cos(t + b), -fwd, -math.sin(t + b)])
    w = e + 0.27 * l / np.linalg.norm(l)
    total = np.linalg.norm(e) + np.linalg.norm(w - e)
    return e / total, w / total


def test_static_metrics_zero_at_gt_and_known_values():
    e, w = _arm(30, 60, 0.4)
    m = static_metrics(e, w, e, w)
    assert all(v == pytest.approx(0, abs=1e-6) for v in m.values())
    # A pure rotation of the whole arm about the shoulder: position error = chord, directions all equal the angle.
    c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    m = static_metrics(rz @ e, rz @ w, e, w)
    def expected(v):  # angle between v and Rz(10 deg) v: cos = c + (1 - c) * (v_z / |v|)^2
        z = v[2] / np.linalg.norm(v)
        return math.degrees(math.acos(c + (1 - c) * z * z))
    for k, v in (("upper_direction_error_degrees", e), ("forearm_direction_error_degrees", w - e), ("chain_direction_error_degrees", w)):
        assert m[k] == pytest.approx(expected(v), abs=1e-6)
    assert m["elbow_error"] == pytest.approx(100 * 2 * np.linalg.norm(e) * math.sin(math.radians(expected(e)) / 2), rel=1e-9)
    assert m["reach_error"] == pytest.approx(0, abs=1e-9) and m["abs_bend_angle_error_degrees"] == pytest.approx(0, abs=1e-6)
    # Swinging the elbow about the shoulder->wrist axis by 25 deg changes only the bend side, by exactly 25 deg.
    d = w / np.linalg.norm(w)
    a = math.radians(25)
    e_sw = e * math.cos(a) + np.cross(d, e) * math.sin(a) + d * np.dot(d, e) * (1 - math.cos(a))
    m2 = static_metrics(e_sw, w, e, w)
    assert m2["side_error_degrees"] == pytest.approx(25, abs=1e-6) and m2["chain_direction_error_degrees"] == pytest.approx(0, abs=1e-6)
    assert m2["abs_bend_angle_error_degrees"] == pytest.approx(0, abs=1e-6) and m2["wrist_error"] == pytest.approx(0, abs=1e-9)
    straight = np.array([0.5, 0, 0]), np.array([1.0, 0, 0])
    assert bend_side(*straight) is None and side_error(*straight, e, w) is None
    r = regime(e, w)
    assert r["bend_regime"] == "strong [45,90)" and r["f_regime"] is not None


def test_motion_metrics_follow_the_worklog77_principle():
    e0, w0 = _arm(20, 40, 0.2)
    e1, w1 = _arm(28, 55, 0.35)
    m = motion_metrics(e0, w0, e1, w1, e0, w0, e1, w1)
    for k in ("upper_direction_residual_degrees", "forearm_direction_residual_degrees", "chain_direction_residual_degrees",
              "elbow_motion_residual", "wrist_motion_residual"):
        assert m[k] == pytest.approx(0, abs=1e-9)
    # A constant offset is not penalized; a frozen candidate's residual equals the GT motion.
    off = np.array([0.05, 0.0, 0.0])
    assert motion_metrics(e0 + off, w0 + off, e1 + off, w1 + off, e0, w0, e1, w1)["elbow_motion_residual"] == pytest.approx(0, abs=1e-9)
    frozen = motion_metrics(e0, w0, e0, w0, e0, w0, e1, w1)
    gt = motion_metrics(e0, w0, e1, w1, e0, w0, e1, w1)
    assert frozen["wrist_motion_residual"] == pytest.approx(gt["wrist_displacement"])
    assert frozen["wrist_displacement"] == 0 and frozen["upper_direction_change_degrees"] == pytest.approx(0, abs=1e-6)


def test_clustered_contrast_and_decision_are_deterministic():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_framepose_anchored_arm_transfer_eval import _cmp, _distributed, _holds
    items = [{"sequence_id": f"s{k % 6}", "candidates": {"a": {"m": 1.0 + 0.1 * (k % 7)}, "b": {"m": 2.0}}} for k in range(120)]
    c1, c2 = _cmp(items, "a", "b", "m"), _cmp(items, "a", "b", "m")
    assert c1 == c2 and c1["mean"] < 0 and c1["sequences"] == 6 and _holds(c1) and _distributed(c1)
    flat = [{"sequence_id": it["sequence_id"], "candidates": {"a": {"m": 2.0}, "b": {"m": 2.0}}} for it in items]
    assert not _holds(_cmp(flat, "a", "b", "m"))


def test_runtime_inputs_are_detector_owned_and_static_contracts():
    build = (ROOT / "scripts/build_mpi3dhp_test_external_bank.py").read_text()
    body = build.split("def main():", 1)[1]
    # The model inputs come only from the detector; GT 2D enters only identity association / verification.
    assert 'kp = people[idx]["keypoints"]' in body and "inp, visible = to_canonical_input(kp, size)" in body
    for line in body.splitlines():
        if "arrays[" in line or "to_canonical_input" in line or "inp" in line.split("=")[0]:
            assert "annot2" not in line and "annot3" not in line, line
    assert "association_uv(annot2[k])" in body
    assert "annotation does not reproject" in build and "STOP: image size" in build and "identity is not one constant offset" in build and "first_half" in build
    pred = (ROOT / "scripts/predict_framepose_anchored_arm_transfer.py").read_text()
    for check in ("STOP: frozen artifact identity mismatch", "STOP: historical reproduction failed", "STOP: a frozen model changed",
                  "REPRODUCTION_TOLERANCE = 1e-6", "requires_grad_(False)"):
        assert check in pred
    assert pred.index("STOP: historical reproduction failed") < pred.index("ExternalBank.load(args.external_bank)")
    assert not re.search(r"\.backward\(|optim|train\(|smooth", pred.split('"""', 2)[2])
    ev = (ROOT / "scripts/run_framepose_anchored_arm_transfer_eval.py").read_text()
    assert ev.index("PROTOCOL = {") < ev.index("def main():") and ev.index("DECISION = {") < ev.index("def main():")
    assert "STOP: predictions did not pass the historical reproduction gate" in ev
    assert not re.search(r"import torch|\.backward\(|smooth", ev)
    module = (ROOT / "src/framepose/mpi3dhp_test_transfer.py").read_text()
    assert not re.search(r"^(from|import) (retarget|blender|framepose\.(fk|ik|writer))", module, re.MULTILINE)
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "mpi3dhp_test_transfer.py":
            assert "mpi3dhp_test_transfer" not in path.read_text(), path
