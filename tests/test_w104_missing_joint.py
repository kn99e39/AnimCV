"""Focused W104 tests: counterfactual construction, GT isolation, restoration provenance, metrics and decision mapping.

Stored-real fixtures run only when the frozen W102/W103 workspaces (and, for result parity, the W104 outputs) exist.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import w104_missing_joint as w104  # noqa: E402
import w104_analysis as w104a  # noqa: E402
from framepose.crops import crop_box, geometry_in_crop  # noqa: E402

W102 = Path(os.environ.get("W102_DIR", "C:/Users/dna10/animcv-w102"))
W101 = Path(os.environ.get("W101_DATA", "C:/Users/dna10/animcv-w101/data"))
W103 = Path(os.environ.get("W103_DIR", "C:/Users/dna10/animcv-w103"))
W104 = Path(os.environ.get("W104_RESULTS", "C:/Users/dna10/animcv-w104/results"))
STORED = (W102/"evidence_manifest.json").exists() and (W103/"evidence_manifest.json").exists()


def synthetic_row(seed=0):
    rng = np.random.default_rng(seed)
    size = (1080, 1920)
    px = np.zeros((17, 2))
    base = {1: (500, 1000), 4: (580, 1000), 2: (490, 1300), 5: (600, 1310), 3: (485, 1600), 6: (610, 1620),
            11: (470, 600), 14: (610, 600), 12: (440, 800), 15: (650, 800), 13: (430, 980), 16: (660, 990), 10: (540, 450)}
    for j, p in base.items():
        px[j] = np.asarray(p) + rng.normal(0, 3, 2)
    px[0] = (px[1]+px[4])/2; px[8] = px[9] = (px[11]+px[14])/2; px[7] = (px[9]+px[0])/2
    x = np.zeros((17, 3), np.float32)
    x[:, :2] = px/np.asarray(size); x[:, 2] = rng.uniform(.4, .99, 17)
    return x, np.ones(17, bool), size


# ------------------------------------------------------------------ construction
def test_closure_follows_production_midpoint_rule():
    assert w104.closure((2,)) == [2]
    assert w104.closure((1,)) == [0, 1, 7]
    assert w104.closure((14,)) == [7, 8, 9, 14]
    assert w104.closure((1, 11)) == [0, 1, 7, 8, 9, 11]


def test_factorial_four_conditions_and_invariants():
    x, v, size = synthetic_row()
    for joints in ((3,), (2, 3), (1, 2, 3), (13, 16), (11,)):
        t, c0, c1 = w104.factorial_row(x, v, size, joints)
        R = w104.closure(joints)
        prod = geometry_in_crop(x, v, size, crop_box(x, v, size))
        assert t["F00"].tobytes() == prod.tobytes()
        v1 = v.copy(); v1[R] = False
        # F11 is exactly the production path on the modified observation
        assert t["F11"].tobytes() == geometry_in_crop(x, v1, size, crop_box(x, v1, size)).tobytes()
        keep = np.setdiff1d(np.arange(17), R)
        # F10: only removed tokens change, and they are zeroed
        assert np.array_equal(t["F10"][keep], t["F00"][keep]) and not t["F10"][R].any()
        # F01: validity/confidence unchanged; coordinates an affine (scale+shift) image of F00
        assert np.array_equal(t["F01"][:, 2:], t["F00"][:, 2:])
        s = c0.side/c1.side
        np.testing.assert_allclose(t["F01"][:, :2]+1, (t["F00"][:, :2]+1)*s + 2*np.array([c0.x-c1.x, c0.y-c1.y])/c1.side, atol=2e-6)
        # F11 tokens = F01 tokens on surviving joints
        assert np.array_equal(t["F11"][keep], t["F01"][keep]) and not t["F11"][R].any()


def test_noop_and_crop_invariant_equality():
    x, v, size = synthetic_row(1)
    t, c0, c1 = w104.factorial_row(x, v, size, ())
    assert all(t[c].tobytes() == t["F00"].tobytes() for c in w104.FACTORIAL)
    # removing an interior joint (elbow inside the bbox) leaves the crop unchanged -> F01 == F00 exactly
    t, c0, c1 = w104.factorial_row(x, v, size, (12,))
    assert c0 == c1 and t["F01"].tobytes() == t["F00"].tobytes() and t["F11"].tobytes() == t["F10"].tobytes()


def test_degenerate_and_nonapplicable_refusal():
    x, v, size = synthetic_row(2)
    v2 = v.copy(); v2[3] = False
    with pytest.raises(ValueError, match="not applicable"):
        w104.factorial_row(x, v2, size, (3,))
    v3 = np.zeros(17, bool); v3[[2, 3]] = True
    with pytest.raises(ValueError, match="degenerate"):
        w104.factorial_row(x, v3, size, (3,))


def test_construction_functions_take_no_gt():
    for fn in (w104.factorial_row, w104.restore_row, w104.restoration_inputs, w104.build_factorial, w104.build_restoration,
               w104.declared_patterns, w104.natural_sets, w104.cmd_declare):
        src = inspect.getsource(fn)
        for banned in ("target_3d", "raw_references", "uv", "pred_", "load_predictions", "mpjpe", "jointPositions"):
            assert banned not in src.replace("geometry_in_crop", ""), (fn.__name__, banned)


def test_patterns_deterministic_and_outcome_free():
    rng = np.random.default_rng(0)
    masks = rng.uniform(size=(500, 17)) > .05
    a = w104.declared_patterns(w104.natural_sets(masks))
    b = w104.declared_patterns(w104.natural_sets(masks[::-1]))
    assert a == b
    names = [w104.pattern_name(k, j) for k, j in a]
    assert len(names) == len(set(names))
    for k, j in a:
        m = w104.mirror(j)
        if m is not None:
            assert m in [jj for _, jj in a]
            assert len(m) == len(j) and not set(m) & set(j)


# ------------------------------------------------------------------ restoration
def fake_landmarks(x, v, size, drop_conf=.12):
    from eval_real_observation_estimator import CANON, SOURCE_OF
    lm = np.full((16, 4), np.nan)
    for j, name in enumerate(w104.JOINTS):
        c = CANON.index(SOURCE_OF[name])
        lm[c] = (x[j, 0]*size[0], x[j, 1]*size[1], x[j, 2], 1.)
    return lm


def test_restoration_rule_restores_finite_raw_only_and_preserves_valid_tokens():
    from w101_reference_estimator import to_input
    x, v, size = synthetic_row(3)
    lm = fake_landmarks(x, v, size)
    from eval_real_observation_estimator import CANON
    lm[CANON.index("left_ankle"), 2:] = (.12, 0.)       # below threshold, finite
    lm[CANON.index("right_wrist")] = np.nan             # absent
    xm, vm = to_input(lm, size)
    raw, have, new = w104.restore_row(lm, size, xm, vm)
    assert np.flatnonzero(new).tolist() == [3]
    assert not have[16] and np.array_equal(raw[vm], xm[vm])
    t, c0, c1 = w104.restoration_inputs(xm, vm, raw, have, size)
    assert t["Rc"][3, 2] == np.float32(.3) and t["R11"][3, 2] == np.float32(.12)
    assert np.array_equal(t["Rc"][:, :2], t["R11"][:, :2]) and np.array_equal(t["R01"][:, 2:], geometry_in_crop(xm, vm, size, c0)[:, 2:])
    assert t["R10"][3, 3] == 1 and (np.abs(t["R10"][vm, :2] - geometry_in_crop(xm, vm, size, c0)[vm, :2]) == 0).all()


def test_privileged_oracle_changes_only_restored_tokens_and_is_labelled():
    from w101_reference_estimator import to_input
    x, v, size = synthetic_row(4)
    lm = fake_landmarks(x, v, size)
    from eval_real_observation_estimator import CANON
    lm[CANON.index("left_hip"), 2:] = (.2, 0.); lm[CANON.index("pelvis"), 3] = 0.; lm[CANON.index("spine"), 3] = 0.
    xm, vm = to_input(lm, size)
    raw, have, new = w104.restore_row(lm, size, xm, vm)
    assert set(np.flatnonzero(new)) == {0, 1, 7}
    uv = np.zeros((24, 2)); uv[w104.SMPL_OF[1]] = (520., 1010.)
    g = w104.privileged_oracle_row(raw, have, new, size, uv)
    r11 = w104.restoration_inputs(xm, vm, raw, have, size)[0]["R11"]
    assert np.array_equal(g[:, 2:], r11[:, 2:])      # identical validity and confidence
    assert "oracle_geometry" in w104.REGIME["P_oracle"] and "NOT GT-free" in w104.REGIME["P_oracle"]
    assert "benchmark_detector_observation" in w104.REGIME["P_bench"]
    assert all("counterfactual" in w104.REGIME[c] for c in ("F10", "F01", "F11"))


# ------------------------------------------------------------------ metrics, statistics, decision
def test_outcome_flags_knee_depth_severe():
    G = np.zeros((1, 17, 3))
    for j, p in {1: (.1, 0, 0), 2: (.1, .2, -.4), 3: (.1, 0, -.8), 4: (-.1, 0, 0), 5: (-.1, .2, -.4), 6: (-.1, 0, -.8),
                 7: (0, 0, .2), 8: (0, 0, .4), 9: (0, 0, .45), 10: (0, 0, .6), 11: (.2, 0, .4), 12: (.3, .3, .2),
                 13: (.35, 0, 0), 14: (-.2, 0, .4), 15: (-.3, .3, .2), 16: (-.35, 0, 0)}.items():
        G[0, j] = p
    J = G.copy(); J[0, 2, 1] = -.15; J[0, 12, 1] = -.3   # flip the left knee bend and left upper-arm depth
    m = w104a.outcome(J, G)
    assert m["knee_wrong"][0, 0] == 1 and m["depth_err"][0, w104a.SEG_INDEX["upper_arm_l"]] == 1
    assert m["limb_fail"][0, 0]
    J2 = G.copy(); J2[0, 1:] += np.array([0, .5, 0])
    assert w104a.outcome(J2, G)["severe"][0]


def test_per_row_pooling_and_video_bootstrap():
    rows = np.array([0, 0, 0, 1, 2, 2]); vals = np.array([1., 2., 3., 10., 0., 2.])
    u, m = w104a.per_row_mean(rows, vals)
    assert u.tolist() == [0, 1, 2] and m.tolist() == [2., 10., 1.]
    vid = np.array(["a", "b", "b"])
    out = w104a.pooled(rows, vals, vid)
    assert out["n"] == 3 and out["videos"] == 2 and abs(out["mean"]-13/3) < 1e-12 and out["ci95"] is not None
    single = w104a.pooled(np.array([0]), np.array([1.]), np.array(["a"]))
    assert single["ci95"] is None


def test_decision_mapping():
    ok = {"CAUSAL": True}; cs = {"CROP_SECONDARY": True}; ex = {"EXPLAINED": True}
    assert w104a.decide(False, ok, cs, ex, {"BENEFIT": True, "HARM": False}) == "D"
    assert w104a.decide(True, ok, cs, ex, {"BENEFIT": True, "HARM": False}) == "A"
    assert w104a.decide(True, ok, cs, ex, {"BENEFIT": False, "HARM": False}) == "B"
    assert w104a.decide(True, ok, cs, ex, {"BENEFIT": False, "HARM": True}) == "B"
    assert w104a.decide(True, ok, cs, ex, {"BENEFIT": True, "HARM": True}) == "C"
    assert w104a.decide(True, {"CAUSAL": False}, cs, ex, {"BENEFIT": True, "HARM": False}) == "C"
    assert w104a.decide(True, ok, {"CROP_SECONDARY": False}, ex, {"BENEFIT": False, "HARM": False}) == "C"
    assert w104a.decide(True, ok, cs, {"EXPLAINED": False}, {"BENEFIT": False, "HARM": False}) == "C"


def test_temporal_step_events_require_gt_stability():
    P = np.zeros((1, 17, 3)); P[0, 2] = (0, 0, -1)
    Q = P.copy(); Q[0, 2] = (0, 1, 0)                      # 90 deg thigh step in S0
    G1 = P.copy(); G2 = P.copy()                          # GT stable
    seg, gs, _, _ = w104a.step_events(P, Q, G1, G2)
    assert seg[0, w104a.SEG_INDEX["thigh_l"]] and gs[0, w104a.SEG_INDEX["thigh_l"]]
    seg, gs, _, _ = w104a.step_events(P, Q, G1, Q)        # GT moves too -> not an event
    assert not seg[0, w104a.SEG_INDEX["thigh_l"]]


# ------------------------------------------------------------------ stored-real fixtures
@pytest.mark.skipif(not STORED, reason="frozen W102/W103 workspaces not present")
def test_stored_raw_landmarks_reproduce_m_input_and_retain_low_conf():
    from w103_observability import load_context
    bank, obs, side, positions = load_context(W102, W101)
    lm, audit = w104.raw_landmarks(W102, bank, positions, obs)
    assert audit["rows_reproducing_M_input"] == len(positions) == 10483
    same = side["identity"] == "same_person"
    inv = same[:, None] & ~obs["input_valid_M"]
    from eval_real_observation_estimator import CANON, SOURCE_OF
    src = [CANON.index(SOURCE_OF[n]) for n in w104.JOINTS]
    assert np.isfinite(lm[:, src, 0])[inv].all() and (lm[:, src, 2][inv] < .3).all()


@pytest.mark.skipif(not STORED, reason="frozen W102/W103 workspaces not present")
def test_stored_sequence_isolation_of_neighbours():
    from w103_observability import load_context
    bank, obs, side, positions = load_context(W102, W101)
    row_of = {str(s): i for i, s in enumerate(obs["sample_id"])}
    for i in range(0, len(positions), 37):
        s = bank.samples[int(positions[i])]
        for d in ("previous", "next"):
            n = s.neighbors.get(d)
            if n and n in row_of:
                t = bank.samples[int(positions[row_of[n]])]
                assert t.sequence_id == s.sequence_id and (t.timestamp > s.timestamp) == (d == "next")


@pytest.mark.skipif(not (W104/"infer/infer_report.json").exists(), reason="W104 inference outputs not present")
def test_stored_w104_reproduction_gate():
    rep = json.loads((W104/"infer/infer_report.json").read_text())
    assert rep["gate_reproduction_pass"]
    assert all(v["byte_equal_to_w102_M_Main"] for v in rep["reproduction_stored_layout"].values())
    assert rep["factorial"]["F00_byte_equal_to_M_Main_geometry"]
    assert rep["reproduction_counterfactual_layout"]["F00_max_abs_mm_vs_stored"] <= w104.F00_BUDGET_MM
    assert rep["lf_normalized_sources_equal_w102"]
    z = np.load(W104/"infer/factorial_inputs.npz")
    f = np.load(W104/"infer/factorial.npz")
    inv = f["crop_invariant"]
    assert np.array_equal(z["F01"][inv], z["F00"][np.searchsorted(f["substrate"], f["rec_row"][inv])])


@pytest.mark.skipif(not (W104/"analysis/analysis_report.json").exists(), reason="W104 analysis outputs not present")
def test_stored_analysis_labels_and_denominators():
    rep = json.loads((W104/"analysis/analysis_report.json").read_text())
    assert rep["REPRODUCTION"] and rep["CASE"] in "ABCD"
    pr = rep["results"]["primary"]
    for c in ("P_oracle", "P_bench"):
        assert pr["restoration"]["privileged"][c]["regime"] == w104.REGIME[c]
    assert rep["reference_validity"]["rows_root_behind_camera"] == 162
    assert "reference_valid_sensitivity" in rep["results"]
