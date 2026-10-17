import itertools
import re
from pathlib import Path

import numpy as np
import pytest

from framepose.visual_depth_motion_attribution import (
    HIGH_GT_MOTION, LOW_GT_MOTION, depth_change_terms, gt_motion_stratum, motion_terms, shapley,
)

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(88)
PLAYERS = ("geometry", "content", "position", "global")


def _coalitions():
    return [frozenset(c) for k in range(5) for c in itertools.combinations(PLAYERS, k)]


def test_shapley_is_exact_and_recovers_additive_and_interaction_structure():
    coef = dict(zip(PLAYERS, RNG.normal(size=4)))
    values = {S: sum(coef[p] for p in S) for S in _coalitions()}
    phi = shapley(values, PLAYERS)
    assert all(phi[p] == pytest.approx(coef[p]) for p in PLAYERS)
    # pure interaction of content x position is split equally, and efficiency holds for arbitrary games
    values = {S: (2.0 if {"content", "position"} <= S else 0.0) for S in _coalitions()}
    phi = shapley(values, PLAYERS)
    assert phi["content"] == pytest.approx(1.0) and phi["position"] == pytest.approx(1.0) and phi["geometry"] == pytest.approx(0.0)
    values = {S: float(RNG.normal()) for S in _coalitions()}
    phi = shapley(values, PLAYERS)
    assert sum(phi.values()) == pytest.approx(values[frozenset(PLAYERS)] - values[frozenset()])


def _chain(n):
    u = RNG.normal(size=(n, 3))
    v = RNG.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rho = RNG.uniform(0.3, 0.7, n)
    e = rho[:, None] * u
    return e, e + (1 - rho)[:, None] * v


def test_depth_change_terms_sum_exactly_to_the_distal_depth_change():
    ea, wa = _chain(300)
    eb, wb = _chain(300)
    t = depth_change_terms(ea, wa, eb, wb)
    assert np.allclose(t["proximal_direction"] + t["distal_direction"] + t["ratio"], 100 * (wb[:, 1] - wa[:, 1]))


def test_motion_terms_and_strata_definitions():
    ga, gb = np.zeros((3, 3)), np.array([[0, 0.01, 0], [0, 0.10, 0], [0, 0.20, 0]])
    pa, pb = np.zeros((3, 3)), np.array([[0, 0.08, 0], [0, 0.10, 0], [0, -0.05, 0]])
    m = motion_terms(pa, pb, ga, gb)
    assert np.allclose(m["pred_disp"], [8, 10, -5]) and np.allclose(m["gt_disp"], [1, 10, 20])
    assert np.allclose(m["abs_residual"], [7, 0, 25]) and np.allclose(m["over_response"], [7, 0, -15])
    assert gt_motion_stratum(ga, gb).tolist() == ["low", "mid", "high"]
    assert LOW_GT_MOTION == 4.0 and HIGH_GT_MOTION == 15.0


def test_runner_contracts():
    run = (ROOT / "scripts/run_visual_depth_motion_attribution.py").read_text()
    assert run.index("DIAGNOSIS = {") < run.index("def main():")
    for check in ("STOP: Worklog 87 depth-motion residuals are not reproduced", "STOP: checkpoint identity mismatch",
                  "grid readout differs from the historical input", "coalition identity broken", "EVALUATION-ONLY"):
        assert check in run
    assert not re.search(r"\.backward\(|optim|smooth|train_joint|requires_grad_\(True\)", run)
    mod = (ROOT / "src/framepose/visual_depth_motion_attribution.py").read_text()
    assert "EVALUATION-ONLY" in mod
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "visual_depth_motion_attribution.py":
            assert "visual_depth_motion_attribution" not in path.read_text(), path


def test_validation_cadence_is_declared_and_uses_available_pairs():
    run = (ROOT / "scripts/run_visual_depth_motion_attribution.py").read_text()
    assert '"validation_cadence"' in run and 'Population("3dpw_validation", fb, fb.indices("validation"), g3, b3, r3, limb["3dpw"], disc["3dpw_gt2d_px"], lambda s: 6' in run
    assert "3dpw_validation_0.1s" in run


def _script(name):
    import importlib.util
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pair(seq, frame, **kw):
    row = {"sequence_id": seq, "frame": frame, "a": 0, "b": 1, "dG": 0.0, "dP0": 0.0, "dP1": 0.0, "res0": 1.0, "res1": 1.0,
           "phi": {"geometry": 0.0, "content": 0.0, "position": 0.0, "global": 0.0, "visual": 0.0}, "jitter": 0.01,
           "det_disp_px": 5.0, "dconf": 0.0, "crop_scale": 0.0, "crop_shift": 0.0, "feat_joints": 0.5, "feat_global": 0.1,
           "stratum": "low"}
    row.update(kw)
    return row


def test_review_selection_rules_are_fixed_and_find_the_declared_oscillation():
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    review = _script("review_visual_depth_motion_attribution")
    L = [_pair("s", 5 * i, res1=float(i), feat_joints=0.1 * i) for i in range(20)]
    # frames 50, 55, 60: low GT motion, |dP1| >= 5 with alternating sign
    for f, dp in ((50, 6.0), (55, -7.0), (60, 8.0)):
        L[f // 5].update(dG=1.0, dP1=dp)
    R = [dict(r) for r in L]
    Lg = [dict(r, res1=0.5) for r in L]
    cases = review.select({"left_arm_3dpw_test": L, "right_arm_3dpw_test": R, "right_leg_3dpw_test": Lg})
    assert set(cases) == {"large_error_low_gt_motion", "large_legitimate_gt_motion", "correct_g1_depth_tracking",
                          "g1_false_depth_oscillation", "detector_instability", "crop_readout_change",
                          "high_feature_small_output", "small_feature_large_output", "similar_error_both_arms",
                          "right_leg_contrast", "contradicts_visual_hypothesis"}
    assert cases["g1_false_depth_oscillation"][1]["frame"] == 50
    assert cases["large_error_low_gt_motion"][1]["frame"] == 90  # 95th percentile of res1 over 20 rows -> index 18
    assert cases["right_leg_contrast"][0] == "right_leg"
    assert "torch" not in (ROOT / "scripts/review_visual_depth_motion_attribution.py").read_text()


def test_posthoc_chain_kinematics_are_consistent_and_flag_bend_side_flips():
    text = (ROOT / "scripts/summarize_visual_depth_motion_strata.py").read_text()
    assert "does not enter the CASE classification" in text and "torch" not in text
    ns = {}
    src = text.split("def kinematics", 1)[1].split("\ndef chain_kinematics", 1)[0]
    exec("import numpy as np\ndef kinematics" + src, ns)
    ea, wa = _chain(200)
    k = ns["kinematics"](ea, wa)
    assert np.allclose(k["distal_y"], k["middle_y"] + (1 - k["rho"]) * k["v_y"])
    u, v = np.array([[0.0, 0, -1]]), np.array([[1.0, 0, 0]])
    na = ns["kinematics"](0.5 * u, 0.5 * u + 0.5 * v)["_n"]
    nb = ns["kinematics"](0.5 * u, 0.5 * u - 0.5 * v)["_n"]
    assert (na * nb).sum() < 0
