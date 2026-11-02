"""W103 observability contract: synthetic metric/selection checks plus stored-artifact checks on Main (skipped elsewhere)."""
import inspect
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/"src"), str(Path(__file__).resolve().parents[1]/"scripts")]
import w103_observability as w
import w103_analysis as wa

W102 = Path(os.environ.get("W102_ROOT", r"C:\Users\dna10\animcv-w102"))
W101 = Path(os.environ.get("W101_DATA", r"C:\Users\dna10\animcv-w101\data"))
W103 = Path(os.environ.get("W103_ROOT", r"C:\Users\dna10\animcv-w103"))
real = pytest.mark.skipif(not (W102/"evidence_manifest.json").exists(), reason="frozen W102 artifacts not on this machine")


def body():
    return np.array([[0, 0, 0], [-.1, 0, 0], [-.1, .12, -.4], [-.1, -.1, -.7], [.1, 0, 0], [.1, -.1, -.4], [.1, .1, -.75],
                     [0, 0, .25], [0, 0, .5], [0, 0, .55], [0, 0, .75], [-.2, 0, .5], [-.35, .15, .3], [-.5, .2, .15],
                     [.2, 0, .5], [.35, -.1, .3], [.5, -.15, .15]], float)


def features(points, valid=None, size=(1000, 1000)):
    """Production crop path for a (17,2) pixel pose -> (17,4) model-visible tensor."""
    from framepose.crops import crop_box, geometry_in_crop
    valid = np.ones(17, bool) if valid is None else valid
    x = np.zeros((17, 3)); x[:, :2] = np.asarray(points)/np.asarray(size); x[:, 2] = .9
    return geometry_in_crop(x, valid, size, crop_box(x, valid, size))


# ------------------------------------------------------------------ 2D similarity / normalization
def test_model_visible_tensor_is_translation_and_scale_invariant():
    pts = body()[:, [0, 2]]*[400, -400] + [500, 500]
    a = features(pts)
    b = features((pts-500)*.5 + [300, 700])
    xa, va = w.base_xy(a[None]); xb, vb = w.base_xy(b[None])
    assert w.full_distance(xa, va, xb, vb)[0] < 1e-6


def test_full_distance_uses_only_jointly_valid_base_points():
    a = features(body()[:, [0, 2]]*[400, -400] + 500)
    b = a.copy(); b[13, :2] += .5; b[13, 3] = 0          # invalid wrist moved: ignored
    xa, va = w.base_xy(a[None]); xb, vb = w.base_xy(b[None])
    assert w.full_distance(xa, va, xb, vb)[0] < 1e-12
    c = a.copy(); c[13, :2] += .13                       # valid wrist moved: RMS over 13 points
    xc, vc = w.base_xy(c[None])
    # model-visible tensors are float32 (production dtype): compare at float32 resolution
    assert math.isclose(w.full_distance(xa, va, xc, vc)[0], math.sqrt(.13**2*2/13), rel_tol=1e-6)


def test_local_limb_distance_is_limb_relative_and_requires_context():
    F = features(body()[:, [0, 2]]*[400, -400] + 500)[None]
    vec, ok = w.local_vectors(F, w.CHAINS["arm_l"])
    G = F.copy(); G[0, :, :2] += .2                      # common translation in crop units
    vec2, _ = w.local_vectors(G, w.CHAINS["arm_l"])
    assert ok[0] and w.local_distance(vec, vec2)[0] < 1e-12
    H = F.copy(); H[0, 4, 3] = 0                         # arm_r context = right hip (4) + left shoulder (11)
    assert not w.local_vectors(H, w.CHAINS["arm_r"])[1][0]
    assert w.local_vectors(H, w.CHAINS["arm_l"])[1][0]   # arm_l context (right shoulder, left hip) unaffected


def test_projection_features_reuse_crop_formula():
    uv = np.zeros((1, 24, 2)); uv[0, :, 0] = np.arange(24)*10; uv[0, :, 1] = np.arange(24)*5
    p = w.projection_features(uv)
    assert p.shape == (1, 13, 2) and np.abs(p).max() <= 1/1.5 + 1e-9


# ------------------------------------------------------------------ 3D metrics
def test_ray_rotation_maps_root_ray_to_forward_and_preserves_geometry():
    R = w.ray_rotation(np.array([.6, 3., -.4]))
    assert np.allclose(R@w.unit(np.array([.6, 3., -.4])), [0, 1, 0]) and np.allclose(R@R.T, np.eye(3))
    G = body(); Gr = G@R.T
    assert np.allclose(np.linalg.norm(Gr[1:]-Gr[:1], axis=1), np.linalg.norm(G[1:]-G[:1], axis=1))


def test_template_pose_keeps_directions_and_fixes_lengths():
    G = body()[None]; lengths = np.full(17, .3)
    T = w.template_pose(G, lengths)
    for c, p in w.PARENT.items():
        assert math.isclose(np.linalg.norm(T[0, c]-T[0, p]), .3, rel_tol=1e-9)
        assert w.angle_deg(T[0, c]-T[0, p], G[0, c]-G[0, p]) < 1e-4   # arccos resolution near 0 deg is ~1e-6
    big = G*1.3
    assert np.allclose(w.template_pose(big, lengths), T)   # body scale removed


def test_depth_flip_knee_branch_and_degenerate_knee():
    a = body()[None]
    b = a.copy(); b[..., 1] *= -1                           # mirror depth
    m = w.pair_metrics(w.row_geometry(a), w.row_geometry(b))
    assert m["depth_flip"].any() and m["material"][0]
    straight = a.copy(); straight[0, 2] = [-.1, 0, -.4]; straight[0, 3] = [-.1, 0, -.8]
    m2 = w.pair_metrics(w.row_geometry(straight), w.row_geometry(straight*[1, -1, 1]))
    assert not m2["knee_eligible"][0, 0] and not m2["knee_branch"][0, 0]   # near-straight knee never assigned a branch
    same = w.pair_metrics(w.row_geometry(a), w.row_geometry(a))
    assert not same["material"][0]


def test_torso_forward_matches_historical_torso_frames():
    from motion.torso_frame import torso_frames
    rng = np.random.default_rng(3)
    G = body()[None] + rng.normal(0, .02, (20, 17, 3))
    f = w.torso_forward(G)
    for g, v in zip(G, f):
        assert np.allclose(torso_frames(g)[0][:, 1], v)


def test_vectorized_s0_metrics_match_w97_source_metrics():
    import eval_external_hmr2_baseline as w97
    import eval_real_observation_estimator as w100
    rng = np.random.default_rng(7)
    G = body()[None] + rng.normal(0, .03, (25, 17, 3)); J = G + rng.normal(0, .08, G.shape)
    m = wa.s0_metrics(J, G)
    for i in range(len(G)):
        ref = w97.source_metrics(J[i]-J[i, 0], G[i]-G[i, 0])
        assert math.isclose(m["mpjpe"][i], ref["mpjpe"], rel_tol=1e-9)
        for s, k in enumerate("lr"):
            if f"knee_wrong_{k}" in ref:
                assert m["knee_wrong"][i, s] == ref[f"knee_wrong_{k}"]
            else:
                assert np.isnan(m["knee_wrong"][i, s])
        e, n = w100.depth_sign_errors(J[i]-J[i, 0], G[i]-G[i, 0])
        assert np.nansum(m["depth_err"][i]) == e and np.isfinite(m["depth_err"][i]).sum() == n


# ------------------------------------------------------------------ retrieval / independence / missing data
def test_retrieval_excludes_same_video_and_returns_distinct_sequences_deterministically():
    X = np.array([[0.], [0.], [.01], [.01], [.5]])
    seq = np.array(["s:a:0", "s:b:0", "s:b:0", "s:c:0", "s:d:0"]); vid = np.array(["a", "b", "b", "c", "d"])
    idx, d = w.retrieve(X, X, seq, vid, vid, k=3, dist=w.euclid_rms(1))
    assert idx[0].tolist() == [1, 3, 4]                     # one row per other sequence, nearest first, earliest tie
    assert all(vid[j] != vid[i] for i in range(5) for j in idx[i] if j >= 0)
    assert (idx[1] != 2).all()                              # same-video neighbor never returned


def test_observation_quality_separates_partial_mislocalized_and_well():
    uv = np.zeros((3, 24, 2))
    for j, s in w.SMPL_OF.items():
        uv[:, s] = [100+10*j, 100+20*j]
    x = np.zeros((3, 17, 3)); v = np.ones((3, 17), bool)
    for j, s in w.SMPL_OF.items():
        x[:, j, :2] = uv[0, s]/1000
    v[1, 13] = False
    x[2, 16, :2] += .2
    q = wa.observation_quality(x, v, uv, np.full((3, 2), 1000.))
    assert q["status"].tolist() == ["well", "partial", "mislocalized"] and q["n_limb_valid"][1] == 11


# ------------------------------------------------------------------ statistics / decision
def test_video_bootstrap_is_seeded_and_video_clustered():
    a = w.video_bootstrap(["v1", "v1", "v2", "v3"], [1, 1, 0, 1])
    b = w.video_bootstrap(["v1", "v1", "v2", "v3"], [1, 1, 0, 1])
    assert a == b and a["videos"] == 3 and a["mean"] == .75
    assert w.video_bootstrap(["v1"]*5, [1]*5)["ci95"] is None


def test_decision_rule_mapping_is_deterministic():
    def block(amb_lo, mod_lo):
        return {"all_important": {"share_ambiguity": {"ci95": [amb_lo, 1]}, "share_model_specific": {"ci95": [mod_lo, 1]}}}
    pair = {"N": {"controlled": {"material": {"count": 4}, "material_video_pairs": 3}}}
    weak = {"N": {"controlled": {"material": {"count": 1}, "material_video_pairs": 1}}}
    assert wa.decide(block(.2, .0), pair, True)["case"] == "A"
    assert wa.decide(block(.2, .0), weak, True)["case"] == "C"   # ambiguity without controlled pairs is not CASE A
    assert wa.decide(block(.0, .2), pair, True)["case"] == "B"
    assert wa.decide(block(.2, .2), pair, True)["case"] == "C"
    assert wa.decide(block(.2, .0), pair, False)["case"] == "D"


def test_declaration_stage_has_no_gt_input():
    assert list(inspect.signature(w.resolution_scales).parameters) == ["FB", "FM", "same"]
    src = inspect.getsource(w.cmd_declare) + inspect.getsource(w.resolution_scales) + inspect.getsource(w.pair_counts)
    assert "target_3d" not in src and "raw_references" not in src and "uv" not in src


def test_analysis_refuses_without_declaration_hash(tmp_path):
    from types import SimpleNamespace
    args = SimpleNamespace(out_dir=str(tmp_path/"a"), declaration=None, declaration_sha="x")
    with pytest.raises(RuntimeError, match="declaration"):
        wa.cmd_analyze(args)


# ------------------------------------------------------------------ stored real cases (Main)
@real
def test_frozen_artifacts_match_manifest():
    assert w.sha(W102/"evidence_manifest.json") == w.W102_MANIFEST_SHA
    manifest = json.loads((W102/"evidence_manifest.json").read_text())["files"]
    for rel in ("results/handoff/observations.npz", "results/s0_sources/prediction_M_Main_test.npy"):
        assert w.sha(W102/rel) == manifest[rel]["sha256"]
    assert w.sha(W102/"results/handoff/observations.npz") == w.OBSERVATIONS_SHA


@real
def test_real_input_reproduction_and_identity():
    bank, obs, side, positions = w.load_context(W102, W101)
    from w101_reference_estimator import geometry_for_condition
    import hashlib
    s0 = json.loads((W102/"results/s0_sources/s0_report.json").read_text())
    F = geometry_for_condition(bank, obs, "M")
    assert hashlib.sha256(F.tobytes()).hexdigest() == s0["conditions"]["M_Main"]["geometry_sha256"]
    assert w.derived_token_audit(bank.arrays["input_2d"][positions], bank.arrays["input_valid"][positions])["pelvis"] < 1e-6
    assert int((np.asarray(side["identity"]) == "same_person").sum()) == 6988


@real
def test_real_stored_analysis_parity_and_decision():
    path = W103/"results/analysis_r3/analysis_report.json"
    if not path.exists():
        pytest.skip("W103 analysis not run on this machine")
    r = json.loads(path.read_text())
    assert r["w102_parity"]["pass"] and r["declaration_sha256"] == w.sha(W103/"results/declare/declaration.json")
    assert r["reference_validity"]["rows_root_behind_camera"] == 162 and not r["reference_validity"]["sensitivity_exclusion_applied"]
    assert r["decision"]["case"] in "ABCD"
    rr = json.loads((W103/"results/reproduce_lf_audit/reproduce_report.json").read_text())
    assert all(v["byte_equal_to_w102"] for c in rr["conditions"].values() for v in c.values())
    assert rr["lf_normalized_sources_equal_w102"]
