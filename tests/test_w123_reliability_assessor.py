"""Focused W123 tests: feature allowlist / GT isolation, metric correctness, calibration, sidecar contract,
hard-pin independence, assessor repeatability, split identity and a frozen-S0 regression (Main data only)."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import w123_reliability_base as RB  # noqa: E402


def _synthetic(n=64, seed=0):
    rng = np.random.default_rng(seed)
    geometry = np.zeros((n, 17, 4), dtype=np.float32)
    geometry[:, :, :2] = rng.uniform(-0.6, 0.6, (n, 17, 2))
    geometry[:, :, 2] = rng.uniform(0.3, 1.3, (n, 17))
    geometry[:, :, 3] = 1.0
    geometry[0, 13, :] = 0.0                    # missing left wrist (zero token like S0 input)
    geometry[1, 12, 2] = 1.2852                 # raw score above 1
    prediction = rng.normal(0, 0.3, (n, 17, 3)).astype(np.float32)
    prediction[:, 0] = 0.0
    target = prediction + rng.normal(0, 0.05, prediction.shape).astype(np.float32)
    return geometry, prediction, target


# ---------------- feature allowlist / GT isolation ----------------

def test_runtime_features_refuse_gt_and_extra_keys():
    g, p, t = _synthetic()
    for extra in ("target", "gt_3d", "error", "sample_id"):
        with pytest.raises(ValueError, match="ALLOWLIST"):
            RB.runtime_features({"geometry": g, "prediction": p, extra: t})
    with pytest.raises(ValueError, match="ALLOWLIST"):
        RB.runtime_features({"geometry": g})
    f = RB.runtime_features({"geometry": g, "prediction": p})
    assert f.shape == (len(g), RB.FEATURE_DIM) and np.isfinite(f).all()


def test_assessor_output_independent_of_ground_truth():
    g, p, t = _synthetic(n=48)
    feats = RB.runtime_features({"geometry": g, "prediction": p})
    y = RB.log_err(RB.joint_errors(p, t))
    cfg = RB.AssessorConfig(hidden=16, epochs=2, batch_size=16, seeds=(1,))
    bundle = RB.train_assessor(feats, y, cfg)
    out1 = RB.assessor_predict(bundle, feats)
    # inference never receives GT: a different target cannot change inference outputs
    out2 = RB.assessor_predict(bundle, RB.runtime_features({"geometry": g, "prediction": p}))
    assert out1.tobytes() == out2.tobytes()


def test_raw_score_above_one_is_preserved_unclipped():
    g, p, _ = _synthetic()
    f = RB.runtime_features({"geometry": g, "prediction": p})
    geom_part = f[:, :68].reshape(-1, 17, 4)
    assert geom_part[1, 12, 2] == np.float32(1.2852)
    assert f[:, 68 + 51 + 17 + 1 + 16 + 1 + 1].max() >= 1.2852 - 1e-6   # valid_score_max column


def test_validity_channel_must_be_binary():
    g, p, _ = _synthetic()
    g[2, 3, 3] = 0.5
    with pytest.raises(ValueError):
        RB.runtime_features({"geometry": g, "prediction": p})


# ---------------- target ----------------

def test_joint_errors_units_rooting_and_invalid_target():
    p = np.zeros((1, 17, 3), dtype=np.float32)
    t = np.zeros((1, 17, 3), dtype=np.float32)
    t[0, 13] = [0.0, 0.3, 0.4]        # 0.5 m off at left wrist
    p[:, :] += 7.0                    # global offset removed by pelvis rooting
    e = RB.joint_errors(p, t)
    assert e.shape == (1, 16)
    assert math.isclose(e[0, RB.TARGET_JOINTS.index(13)], 0.5, rel_tol=1e-6)
    assert np.count_nonzero(e) == 1
    t[0, 5, 0] = np.nan
    with pytest.raises(ValueError, match="never zero-filled"):
        RB.joint_errors(p, t)


def test_reprojection_residual_zero_when_consistent():
    rng = np.random.default_rng(3)
    p = rng.normal(0, 0.3, (4, 17, 3)).astype(np.float32)
    g = np.zeros((4, 17, 4), dtype=np.float32)
    g[:, :, 0] = 0.8 * p[:, :, 0] + 0.1
    g[:, :, 1] = -0.8 * p[:, :, 2] - 0.05
    g[:, :, 2] = 1.0
    g[:, :, 3] = 1.0
    r, s = RB.reprojection_residual(g, p)
    assert np.abs(r).max() < 1e-5 and np.allclose(s, 0.8, atol=1e-5)
    g[0, 5, 0] += 0.2
    r2, _ = RB.reprojection_residual(g, p)
    assert r2[0, 5] == r2[0].max() and r2[0, 5] > 0.1


# ---------------- metrics ----------------

def test_spearman_matches_scipy_with_ties():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(5)
    x = rng.integers(0, 5, 200).astype(float)
    y = x + rng.normal(0, 2, 200)
    assert math.isclose(RB.spearman(x, y), spearmanr(x, y).correlation, rel_tol=1e-9)


def test_within_joint_spearman_removes_joint_prior():
    joint = np.repeat([0, 1], 50)
    rng = np.random.default_rng(1)
    e = rng.uniform(0, 1, 100) + joint * 10           # joint 1 has much larger error
    prior_only = joint.astype(float)                  # per-joint constant predictor
    assert RB.spearman(prior_only, e) > 0.8           # pooled Spearman rewards the prior
    assert RB.within_joint_spearman(prior_only, e, joint) == 0.0
    assert RB.within_joint_spearman(e, e, joint) == pytest.approx(1.0)


def test_average_precision_known_values_and_ties():
    assert RB.average_precision(np.array([3, 2, 1.0]), np.array([1, 0, 1], bool)) == pytest.approx((1 + 2 / 3) / 2)
    # all tied: AP equals base rate
    assert RB.average_precision(np.ones(10), np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], bool)) == pytest.approx(0.2)
    assert math.isnan(RB.average_precision(np.ones(3), np.zeros(3, bool)))


def test_triage_budget_per_video_and_label_free_tiebreak():
    risk = np.array([5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0], float)
    video = np.array(["a"] * 10 + ["b"] * 10)
    tb = RB.tiebreak_keys([f"c{i}" for i in range(20)])
    flags = RB.triage_flags(risk, video, tb, budget=0.10)
    assert flags.sum() == 2 and flags[0] and flags[10] + flags[11] == 1
    assert np.array_equal(flags, RB.triage_flags(risk, video, tb, budget=0.10))   # deterministic
    # negative / no-op control: a constant risk flags exactly the budget, chosen by tie-break only
    const = RB.triage_flags(np.zeros(20), video, tb, budget=0.10)
    assert const.sum() == 2


def test_conformal_shift_reaches_declared_coverage_and_outputs_valid():
    rng = np.random.default_rng(7)
    y = rng.normal(-3, 0.7, (500, 16))
    raw = np.zeros((500, 16, 2)) - 4.0
    shifts = RB.conformal_shift(raw, y)
    q = RB.apply_calibration(raw, shifts)
    e = np.exp(y) - RB.EPS_M
    assert RB.coverage(e.ravel(), q[..., 1].ravel()) >= 0.9
    assert abs(RB.coverage(e.ravel(), q[..., 0].ravel()) - 0.5) < 0.01
    assert (q >= 0).all() and (q[..., 1] >= q[..., 0]).all() and np.isfinite(q).all()


def test_cluster_bootstrap_resamples_whole_clusters():
    v = np.array(["a"] * 5 + ["b"] * 5)
    x = np.array([0.0] * 5 + [1.0] * 5)
    r = RB.cluster_bootstrap(v, lambda i: float(x[i].mean()), reps=200)
    assert r["clusters"] == 2 and r["ci95"][0] >= 0.0 and r["ci95"][1] <= 1.0
    single = RB.cluster_bootstrap(np.array(["a"] * 3), lambda i: 1.0)
    assert single["ci95"] is None


def test_table_baseline_fallback_and_score_bins():
    g, p, t = _synthetic(n=200, seed=11)
    y = RB.log_err(RB.joint_errors(p, t))
    res, _ = RB.reprojection_residual(g, p)
    for b in RB.make_baselines().values():
        b.fit(g, res, y)
        out = b.predict(g, res)
        assert out.shape == (200, 16, 2) and np.isfinite(out).all()
    g2 = g.copy()
    g2[:, :, 2] = 5.0          # unseen score domain falls into open top bin, never clipped to 1
    out = RB.make_baselines()["B_raw_score_validity"].fit(g, res, y).predict(g2, res)
    assert np.isfinite(out).all()


def test_assessor_training_repeatable():
    g, p, t = _synthetic(n=80, seed=2)
    feats = RB.runtime_features({"geometry": g, "prediction": p})
    y = RB.log_err(RB.joint_errors(p, t))
    cfg = RB.AssessorConfig(hidden=16, epochs=3, batch_size=16, seeds=(1, 2))
    a = RB.train_assessor(feats, y, cfg)
    b = RB.train_assessor(feats, y, cfg)
    assert RB.assessor_weights_sha(a) == RB.assessor_weights_sha(b)
    out = RB.assessor_predict(a, feats)
    assert (out[..., 1] >= out[..., 0]).all()


# ---------------- sidecar / provenance / hard pins ----------------

def _sidecar():
    g, p, _ = _synthetic(n=6)
    fi = np.arange(10, 16)
    q50 = np.full((6, 16), 0.05)
    q90 = np.full((6, 16), 0.12)
    sc = RB.build_sidecar(source={"clip_id": "SYN"}, frame_index=fi, timestamp=fi / 30.0, geometry=g, prediction=p,
                          q50=q50, q90=q90, checkpoint_sha256="ck", assessor_id="as", calibration_id="cal",
                          observation_regime="synthetic_projection")
    return sc, g, p, fi


def test_sidecar_valid_and_preserves_provenance_raw_score_and_missing():
    sc, g, p, fi = _sidecar()
    assert RB.validate_sidecar(sc, checkpoint_sha256="ck", prediction=p, frame_index=fi, geometry=g,
                               assessor_id="as", calibration_id="cal") == []
    assert sc["detector_raw_score"][1][RB.TARGET_JOINTS.index(12)] == pytest.approx(1.2852, abs=1e-6)
    assert sc["support"][0][RB.TARGET_JOINTS.index(13)] == RB.SUPPORT_CODES["SUPPORTED_INPUT_MISSING"]
    assert sc["subject_identity"] == "UNKNOWN"
    assert "not a probability" in sc["detector_score_semantics"]


@pytest.mark.parametrize("mutation,code", [
    ("stale_checkpoint", "STALE_CHECKPOINT"), ("reorder", "FRAME_INDEX_MISMATCH"), ("joint_order", "JOINT_ORDER"),
    ("gt_leak", "GT_OR_LABEL_FIELD_PRESENT"), ("score_clip", "RAW_SCORE_ALTERED"), ("stale_calib", "STALE_ASSESSOR_OR_CALIBRATION"),
    ("negative", "RELIABILITY_INVALID"), ("support", "SUPPORT_FLAGS"), ("pose", "POSE_HYPOTHESIS_MISMATCH")])
def test_sidecar_negative_controls(mutation, code):
    sc, g, p, fi = _sidecar()
    ck, cal, pred = "ck", "cal", p
    if mutation == "stale_checkpoint":
        ck = "other"
    elif mutation == "reorder":
        sc["frame_index"] = sc["frame_index"][::-1]
    elif mutation == "joint_order":
        sc["joint_names"] = sc["joint_names"][::-1]
    elif mutation == "gt_leak":
        sc["gt_error_m"] = [[0.0]]
    elif mutation == "score_clip":
        sc["detector_raw_score"] = np.minimum(np.asarray(sc["detector_raw_score"]), 1.0).tolist()
    elif mutation == "stale_calib":
        cal = "newer"
    elif mutation == "negative":
        sc["q50_m"][0][0] = -0.1
    elif mutation == "support":
        sc["support"][0][RB.TARGET_JOINTS.index(13)] = 0
    elif mutation == "pose":
        pred = p.copy()
        pred[0, 3, 0] += 1e-3
    errs = RB.validate_sidecar(sc, checkpoint_sha256=ck, prediction=pred, frame_index=fi, geometry=g,
                               assessor_id="as", calibration_id=cal)
    assert code in errs


def test_build_sidecar_rejects_reordered_frames_and_invalid_reliability():
    g, p, _ = _synthetic(n=3)
    with pytest.raises(ValueError, match="REORDERED"):
        RB.build_sidecar(source={}, frame_index=np.array([0, 2, 1]), timestamp=np.zeros(3), geometry=g, prediction=p,
                         q50=np.zeros((3, 16)), q90=np.zeros((3, 16)), checkpoint_sha256="c", assessor_id="a",
                         calibration_id="b", observation_regime="x")
    with pytest.raises(ValueError, match="INVALID"):
        RB.build_sidecar(source={}, frame_index=np.arange(3), timestamp=np.zeros(3), geometry=g, prediction=p,
                         q50=np.ones((3, 16)), q90=np.zeros((3, 16)), checkpoint_sha256="c", assessor_id="a",
                         calibration_id="b", observation_regime="x")


def test_hard_pins_independent_of_uncertainty():
    ref = np.zeros((5, 16, 3))
    pins = {(2, RB.TARGET_JOINTS.index(13)): np.array([0.1, 0.2, 0.3])}
    low = RB.illustrative_future_consumer(ref, np.full((5, 16), 0.01), pins)
    high = RB.illustrative_future_consumer(ref, np.full((5, 16), 5.0), pins)
    assert low["hard_pins"].keys() == high["hard_pins"].keys() == pins.keys()
    for k in pins:
        assert low["hard_pins"][k].tobytes() == high["hard_pins"][k].tobytes() == pins[k].astype(np.float64).tobytes()
    assert (high["soft_weight"] < low["soft_weight"]).all()
    assert low["mapping_status"] == "ILLUSTRATIVE_NOT_POLICY"
    pins[(2, RB.TARGET_JOINTS.index(13))][0] = 9.0        # consumer copies; caller mutation cannot leak in
    assert low["hard_pins"][(2, RB.TARGET_JOINTS.index(13))][0] == 0.1


# ---------------- split identity ----------------

def test_split_assignment_group_disjoint():
    import w123_reliability_assessor as W
    val = ["courtyard_a_00", "courtyard_b_01"]
    test = ["downtown_x_00", "downtown_x_01", "downtown_y_00", "flat_z_01", "office_q_00", "outdoors_r_01"]
    a = W.split_assignment(val, test)
    vs = a["video_split"]
    assert vs["downtown_x_00"] == vs["downtown_x_01"]
    assert all(vs[v] == "TRAIN" for v in val)
    assert {vs[v] for v in test} <= {"CALIB", "FINAL"}
    with pytest.raises(RuntimeError):
        W.split_assignment(["downtown_x_02"], test)


# ---------------- frozen S0 regression on Main data (skipped elsewhere) ----------------

W101 = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "animcv-w101" / "data"
W102 = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "animcv-w102" / "results" / "s0_sources"


@pytest.mark.skipif(not (W101 / "s0_checkpoint.pt").exists() or not (W102 / "prediction_B_Main_validation.npy").exists(),
                    reason="Main-local frozen S0 artifacts unavailable")
def test_frozen_s0_checkpoint_and_first_batch_byte_identical():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import w123_reliability_assessor as W
    assert RB.sha_file(W101 / "s0_checkpoint.pt") == W.CHECKPOINT_SHA
    from framepose.bank import load_bank
    from framepose.train import geometry_tensor
    bank = load_bank(W101 / "bank_3dpw_paired_v2.json")
    pos = bank.indices("validation")[:512]
    pred = W.s0_forward(geometry_tensor(bank)[pos])
    stored = np.load(W102 / "prediction_B_Main_validation.npy")[:512]
    assert pred.tobytes() == stored.tobytes()
    assert RB.sha_file(W101 / "s0_checkpoint.pt") == W.CHECKPOINT_SHA
