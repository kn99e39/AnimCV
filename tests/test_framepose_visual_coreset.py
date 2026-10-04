import re
from pathlib import Path

import numpy as np
import pytest

from framepose.visual_coreset import arm_features, coreset_size, farthest_point_coreset, redundancy

ROOT = Path(__file__).resolve().parent.parent


def test_arm_features_use_only_endpoints_and_are_l2_normalized():
    r = np.random.default_rng(0).normal(size=(5, 4, 3)).astype(np.float16)
    f = arm_features(r)
    assert f.shape == (5, 9)
    assert np.linalg.norm(f, axis=1) == pytest.approx(np.ones(5))
    r2 = r.copy()
    r2[:, 3] = 99  # global token must not matter
    assert np.array_equal(arm_features(r2), f)
    assert np.array_equal(arm_features(r), f)  # deterministic


def test_k_is_minimum_group_size():
    groups = {"a": np.arange(10), "b": np.arange(10, 14), "c": np.arange(14, 30)}
    assert coreset_size(groups) == 4


def test_centroid_nearest_initialization_and_farthest_point_selection():
    # Three tight clusters plus one point at the centroid direction.
    feats = np.zeros((7, 3))
    feats[0] = [1, 0, 0]; feats[1] = [0.99, 0.14, 0]; feats[2] = [0, 1, 0]
    feats[3] = [0.1, 0.99, 0]; feats[4] = [0, 0, 1]; feats[5] = [0.577, 0.577, 0.577]; feats[6] = [0, 0.1, 0.99]
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    positions = np.arange(7)
    core = farthest_point_coreset(feats, positions, 1)
    assert core.tolist() == [5]  # closest to the centroid
    core4 = farthest_point_coreset(feats, positions, 4)
    assert 5 in core4 and len(set(core4.tolist())) == 4
    clusters = [{0, 1}, {2, 3}, {4, 6}]
    assert all(len(set(core4.tolist()) & c) == 1 for c in clusters)  # one per cluster
    assert np.array_equal(core4, farthest_point_coreset(feats, positions[::-1], 4))  # order-independent
    assert farthest_point_coreset(feats, positions, 10).tolist() == positions.tolist()


def test_tie_break_by_smallest_position():
    feats = np.array([[1.0, 0.0]] * 10)  # indexed by FrameBank position
    assert farthest_point_coreset(feats, np.array([7, 3, 9, 5]), 2).tolist() == [3, 5]


def test_coreset_reduces_measured_redundancy():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(5, 16))
    feats = np.concatenate([b + 0.05 * rng.normal(size=(20, 16)) for b in base])
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    pos = np.arange(len(feats))
    core = farthest_point_coreset(feats, pos, 5)
    full, sub = redundancy(feats, pos), redundancy(feats, core)
    assert sub["mean_nearest_neighbour_cosine"] < full["mean_nearest_neighbour_cosine"]
    assert sub["frames"] == 5


def test_s1_s2_identical_sequence_draws_and_coreset_membership():
    pytest.importorskip("torch")
    from framepose.visual_supervision_sampling import EpochSampler

    seq = ["a"] * 30 + ["b"] * 12 + ["c"] * 8
    train = np.arange(50)
    groups = {"a": np.arange(30), "b": np.arange(30, 42), "c": np.arange(42, 50)}
    cores = {s: v[:: max(1, len(v) // 8)][:8] for s, v in groups.items()}
    s1 = EpochSampler("S1_SEQUENCE_BALANCED", train, groups, 1337)
    s2 = EpochSampler("S1_SEQUENCE_BALANCED", train, cores, 1337)
    for _ in range(5):
        a, b = s1.epoch(), s2.epoch()
        assert len(a) == len(b) == 50
        assert [seq[p] for p in a] == [seq[p] for p in b]
        members = set(np.concatenate(list(cores.values())).tolist())
        assert set(b.tolist()) <= members


def test_no_target_gt_or_error_in_coreset_and_contracts_checked():
    module = (ROOT / "src/framepose/visual_coreset.py").read_text()
    code = module.split('"""', 2)[2]
    assert not re.search(r"target|f_gt|error|depth|loss|random", code)
    script = (ROOT / "scripts/run_visual_coreset_diversity.py").read_text()
    for check in ("Worklog 68 cache identity mismatch", "eligible TRAIN population differs from Worklog 71",
                  "S1 and S2 sequence draws differ", "S1 replay does not reproduce the Worklog 71 recorded contribution",
                  "S2 drew a frame outside the declared coreset", "S2 optimization budget differs from S1"):
        assert check in script
    assert "from run_visual_supervision_diversity import CONFIG, train" in script
    coreset_part = script[script.index("# --- coreset ---"):script.index("# --- contribution accounting")]
    assert not re.search(r"targets|masks\[|f_gt", coreset_part)
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name == "visual_coreset.py":
            continue
        assert "visual_coreset" not in path.read_text(), path
