"""Focused tests for the W115 scratch measurement library (synthetic; no Production code involved)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w114_source_preserving_base as B  # noqa: E402
import w114_source_preserving_contract as C  # noqa: E402
import w115_validity_base as W  # noqa: E402
import w115_validity_contract as K  # noqa: E402

RQ = [1.0, -1.410649872468639e-07, 0.0, 0.0]


# ------------------------------------------------------------------------------------- numeric comparator ---

def test_blender_mul_identity_and_float32():
    q = np.array([0.5, 0.5, 0.5, 0.5], np.float32)
    assert np.array_equal(W.blender_mul_f32([1, 0, 0, 0], q), q)
    out = W.blender_mul_f32(RQ, q)
    assert out.dtype == np.float32


def _frame(i, obj, bones, mode=None):
    fr = {"frame_index": i, "modes": {}, "yaw_deg": {"S": 0.0}, "object_rotation_xyzw": {"S": obj},
          "extra_bone_locals_xyzw": {"S": dict(bones)}}
    if mode:
        fr["modes"]["S"] = mode
    return fr


def test_storage_key_order_mode_overrides_extra_spine():
    bones = {"spine_01": [0.1, 0.0, 0.0, 0.99], "upperarm_l": [0, 0, 0, 1], "lowerarm_l": [0, 0, 0, 1]}
    mode = {"proximal_local_xyzw": [0.2, 0, 0, 0.98], "distal_local_xyzw": [0.3, 0, 0, 0.95],
            "extra_bone_locals_xyzw": {"spine_01": [0.0, 0.1, 0.0, 0.99]}}
    st = W.stored_case_values([_frame(0, [0, 0, 0, 1], bones, mode)], "S", RQ)
    assert np.array_equal(st["spine_01"][1], np.array([0.99, 0.0, 0.1, 0.0], np.float32))
    assert np.array_equal(st["upperarm_l"][1], np.array([0.98, 0.2, 0, 0], np.float32))


def test_storage_cast_below_and_above_resolution():
    a = 0.7
    tiny = a + 1e-12          # same float32
    big = a + 1e-6            # different float32
    s = lambda v: W.stored_case_values([_frame(0, [0, 0, 0, 1], {"spine_01": [v, 0, 0, 0.7]})], "S", RQ)["spine_01"][1]  # noqa: E731
    assert np.array_equal(s(a), s(tiny))
    assert not np.array_equal(s(a), s(big))


def test_classify_layers():
    l1a = {c: np.array([[1.0, 0, 0, 0]] * 3) for c in W.CH}
    l1b = {c: v.copy() for c, v in l1a.items()}
    l1b["spine_01"][1, 0] = 1.0 - 1e-12  # below float32 resolution near 1 (a zero component would resolve 1e-12)
    l1b["spine_01"][2, 1] = 1e-3         # material
    l2a = {c: v.astype(np.float32).astype(np.float64) for c, v in l1a.items()}
    l2b = {c: v.astype(np.float32).astype(np.float64) for c, v in l1b.items()}
    r = W.classify(l1a, l1b, l2a, l2b)["spine_01"]
    assert (r["invariant"], r["below_output_resolution"], r["material_stored"]) == (1, 1, 1)
    assert r["missing_or_contradictory"] == 0


def test_classify_sign_flip_is_stored_change_not_rotation():
    a = {c: np.array([[0.6, 0.8, 0, 0]]) for c in W.CH}
    b = {c: -v for c, v in a.items()}
    r = W.classify(a, b, a, b)["thigh_l"]
    assert r["material_stored"] == 1 and r["material_stored_rotation"] == 0 and r["stored_sign_flip_only"] == 1


# ----------------------------------------------------------------------------------- lock semantics ---

def test_local_world_divergence():
    T = 1
    before = {c: np.array([[1.0, 0, 0, 0]]) for c in W.CH}
    after = {c: v.copy() for c, v in before.items()}
    after["spine_01"] = np.array([[0.9, 0.1, 0, 0]])  # spine_01 changes; depicts thorax + shoulders
    lock = np.zeros((T, B.J), bool)
    lock[0, B.JI["right_wrist"]] = True               # not depicted by any keyed channel
    loc = W.local_lock(before, after, lock)
    assert loc["verdict"] == "SATISFIED" and loc["no_keyed_channel_cells"] == 1
    p0 = [{p: [0.0, 0.0, 0.0] for p in W.WORLD_POINT.values()}]
    p1 = [dict(p0[0], hand_r=[0.0, 0.0, 0.01])]
    wl = W.world_lock(p0, p1, lock)
    assert wl["verdict"] == "VIOLATED" and wl["max_world_displacement_m_per_joint"]["right_wrist"] == pytest.approx(0.01)
    lock2 = np.zeros((T, B.J), bool)
    lock2[0, B.JI["thorax"]] = True
    assert W.local_lock(before, after, lock2)["verdict"] == "VIOLATED"


def test_world_dependencies_from_rest():
    deps = W.world_dependencies(ROOT / "examples/e2e_demo/baserig_blender_rest.json")
    assert deps["left_wrist"] == ["__object__", "spine_01", "upperarm_l", "lowerarm_l"]
    assert deps["left_hip"] == ["__object__"]
    assert "spine_01" in deps["right_wrist"] and "upperarm_l" not in deps["right_wrist"]


# ------------------------------------------------------------------------------ semantic canonicalization ---

def test_own_root_relative_guard():
    assert W.own_root_relative("C:/a/b/build/x/y.json", "C:\\a\\b\\build") == "x/y.json"
    assert W.own_root_relative("C:/a/b/build2/x/y.json", "C:/a/b/build") is None
    assert W.own_root_relative("C:/a/b/buildx.json", "C:/a/b/build") is None


def test_field_pattern_is_not_a_prefix():
    assert W.matches(["digests", "synthetic_programme_states", 3], ["digests", "synthetic_programme_states", "*"])
    assert not W.matches(["synthetic", "controls_summary", "passed"], ["synthetic", "evidence_digest"])
    assert not W.matches(["digests", "synthetic", "x"], ["digests", "synthetic"])
    assert not W.matches(["synthetic", "programme", "state_ids", "k"], ["synthetic", "programme", "state_ids", "*"])


def _synthetic_bundle(root: Path):
    root.mkdir(parents=True)
    ev, _ = C.synthetic_evidence(root)
    cs = C.synth_directives(ev)
    lin = B.Lineage(ev)
    B.apply_edit(lin, ev, cs["SS1"], B.make_mock_edit(lin, ev, cs["SS1"], "E1", [(3, "left_wrist")], C.DELTA))
    B.apply_revert(lin, ev, B.make_revert(lin, "E1"))
    W.save_bundle(root / "bundles" / "synthetic", ev, cs, lin)
    return ev


def _chain(root):
    bd = W.load_bundle(root / "bundles" / "synthetic")
    return bd, W.normalized_chain(bd, [["sources", "synthetic_source", "path"]], str(root), root)


def test_normalized_chain_equal_across_roots_and_raw_differs(tmp_path):
    e1 = _synthetic_bundle(tmp_path / "r1")
    e2 = _synthetic_bundle(tmp_path / "r2")
    assert e1["meta"]["evidence_digest"] != e2["meta"]["evidence_digest"]       # path-dependent raw digest
    (b1, (c1, ref1, p1)), (b2, (c2, ref2, p2)) = _chain(tmp_path / "r1"), _chain(tmp_path / "r2")
    assert W.bundle_integrity(b1)["ok"] and W.bundle_integrity(b2)["ok"]
    assert not p1 and not p2 and c1 == c2 and ref1 == ref2


def test_normalization_rejects_substantive_and_non_allowlisted_changes(tmp_path):
    _synthetic_bundle(tmp_path / "r1")
    _synthetic_bundle(tmp_path / "r2")
    c1 = _chain(tmp_path / "r1")[1][0]
    b = tmp_path / "r2" / "bundles" / "synthetic"
    K._rechain(b, lambda bd: bd["arrays"].__setitem__("obs2d_score_raw", np.array(bd["arrays"]["obs2d_score_raw"]) + 0.0
                                                                  + np.eye(1, bd["arrays"]["obs2d_score_raw"].size, 7).reshape(bd["arrays"]["obs2d_score_raw"].shape) * 0.01))
    bd2, (c2, _, _) = _chain(tmp_path / "r2")
    assert W.bundle_integrity(bd2)["ok"]           # a self-consistent forgery passes integrity ...
    assert c2 != c1                                # ... and is rejected by the semantic chain
    shutil.rmtree(tmp_path / "r2")
    _synthetic_bundle(tmp_path / "r2")
    K._rechain(b, lambda bd: bd["meta"]["sources"]["synthetic_source"].__setitem__("sha256", "0" * 64))
    assert _chain(tmp_path / "r2")[1][0] != c1     # the recorded source sha256 is never normalized


def test_allowlisted_path_outside_own_root_is_not_normalized(tmp_path):
    _synthetic_bundle(tmp_path / "r1")
    bd = W.load_bundle(tmp_path / "r1" / "bundles" / "synthetic")
    _, _, problems = W.normalized_chain(bd, [["sources", "synthetic_source", "path"]], str(tmp_path / "elsewhere"), tmp_path / "r1")
    assert problems


def test_stale_digest_detected_by_integrity(tmp_path):
    _synthetic_bundle(tmp_path / "r1")
    bd = W.load_bundle(tmp_path / "r1" / "bundles" / "synthetic")
    bd["arrays"]["obs2d_score_raw"] = np.array(bd["arrays"]["obs2d_score_raw"]) * 1.0
    bd["arrays"]["obs2d_score_raw"][0, 0] += 0.01
    assert not W.bundle_integrity(bd)["ok"]


def test_edit_lineage_rollback_in_bundle(tmp_path):
    _synthetic_bundle(tmp_path / "r1")
    bd = W.load_bundle(tmp_path / "r1" / "bundles" / "synthetic")
    la = bd["lineage_arrays"]
    assert len(bd["states"]) == 3 and la["values_000"].tobytes() == la["values_002"].tobytes()
    assert la["values_001"].tobytes() != la["values_000"].tobytes()


# --------------------------------------------------------------------------------- frozen-rig reproduction ---

W102 = Path("C:/Users/dna10/animcv-w102/results/historical_video/observations_predictions.npz")


@pytest.mark.skipif(not W102.exists(), reason="Main-local W102 artefacts not present")
def test_storage_rig_on_unperturbed_real_clip_matches_w105_case_values():
    import w114_rig_influence as R
    decl = json.loads(Path("C:/Users/dna10/animcv-w114/results/declare/declaration.json").read_text())
    z, rep, insp = C.real_inputs(decl)
    ev, _ = C.real_evidence(decl, z, rep, insp)
    rc = R.rig_context()
    v = np.array(ev["arrays"]["est3d_xyz_hypothesis"])
    frames = R.real_case_frames(v, ev["arrays"]["timestamp"], rc, "h")
    vc = json.loads(Path(decl["inputs"]["w105_video_cases"]["path"]).read_text())["cases"]["real_clip"]["frames"]
    mine = json.loads(json.dumps(C.clean(R.slot_view(frames, "SLOT"))))
    assert sum(x == y for x, y in zip(mine, R.slot_view(vc, "FIXED"))) == 199
    srig = W.storage_rig("s", K.real_frames_fn(ev["arrays"]["timestamp"], rc), RQ, R.DEPICTS, R.OWNERS["real_s0_only"])
    ch = srig.channels(v)
    assert all(np.isfinite(ch[c]).all() for c in W.CH) and srig.tol == 0.0
