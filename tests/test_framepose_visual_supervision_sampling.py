import re
from pathlib import Path

import numpy as np
import pytest

from framepose.visual_supervision_sampling import (
    SAMPLERS, contribution_counts, contribution_summary, cosine_by_lag, train_sequence_groups,
)

ROOT = Path(__file__).resolve().parent.parent


def _fixture():
    seq = ["a"] * 60 + ["b"] * 30 + ["c"] * 10 + ["v"] * 5
    train = np.arange(100)
    eligible = np.ones(105, bool)
    eligible[5] = False  # an ineligible TRAIN row in sequence a
    return seq, train, eligible


def test_train_sequence_grouping_excludes_ineligible_and_non_train():
    seq, train, eligible = _fixture()
    groups = train_sequence_groups(seq, train, eligible)
    assert list(groups) == ["a", "b", "c"]
    assert len(groups["a"]) == 59 and 5 not in groups["a"]
    assert "v" not in groups  # validation rows never enter a TRAIN sampler


def test_s0_is_the_worklog69_epoch_and_deterministic():
    torch = pytest.importorskip("torch")
    from framepose.visual_supervision_sampling import EpochSampler

    seq, train, eligible = _fixture()
    groups = train_sequence_groups(seq, train, eligible)
    s0 = EpochSampler("S0_FRAME_UNIFORM", train, groups, 1337)
    gen = torch.Generator(device="cpu").manual_seed(1337)
    reference = [train[torch.randperm(len(train), generator=gen).numpy()] for _ in range(3)]
    for ref in reference:
        assert np.array_equal(s0.epoch(), ref)  # identical to the Worklog 69 loop
    again = EpochSampler("S0_FRAME_UNIFORM", train, groups, 1337)
    assert np.array_equal(again.epoch(), reference[0])


def test_s1_sequence_balanced_same_draw_count_deterministic_and_uniform():
    pytest.importorskip("torch")
    from framepose.visual_supervision_sampling import EpochSampler

    seq, train, eligible = _fixture()
    groups = train_sequence_groups(seq, train, eligible)
    s1 = EpochSampler("S1_SEQUENCE_BALANCED", train, groups, 1337)
    s0 = EpochSampler("S0_FRAME_UNIFORM", train, groups, 1337)
    orders1 = [s1.epoch() for _ in range(300)]
    orders0 = [s0.epoch() for _ in range(300)]
    assert all(len(o) == len(train) for o in orders1 + orders0)  # identical draws per epoch
    assert all(eligible[o].all() and np.isin(o, train).all() for o in orders1)
    c1 = contribution_summary(contribution_counts(orders1, seq), groups)
    c0 = contribution_summary(contribution_counts(orders0, seq), groups)
    assert c1["max_abs_deviation_from_uniform"] < 0.01      # ~1/3 each
    assert c0["max_abs_deviation_from_frame_share"] < 0.01  # follows frame frequency
    assert c0["per_sequence"]["a"]["share"] > 0.55 and c1["per_sequence"]["a"]["share"] < 0.35
    repeat = EpochSampler("S1_SEQUENCE_BALANCED", train, groups, 1337)
    assert np.array_equal(repeat.epoch(), orders1[0])
    assert SAMPLERS == ("S0_FRAME_UNIFORM", "S1_SEQUENCE_BALANCED")


def test_cosine_by_lag():
    feats = np.array([[1, 0], [1, 0], [0, 1], [1, 1]], float)
    order = np.array([0, 1, 2, 3])
    assert cosine_by_lag(feats, order, 1) == pytest.approx([1.0, 0.0, np.sqrt(0.5)])
    assert cosine_by_lag(feats, order, 2) == pytest.approx([0.0, np.sqrt(0.5)])
    assert len(cosine_by_lag(feats, order, 8)) == 0


def test_sampler_is_not_target_depth_or_error_aware_and_eval_unchanged():
    module = (ROOT / "src/framepose/visual_supervision_sampling.py").read_text()
    code = module.split('"""', 2)[2]
    assert not re.search(r"target|f_gt|error|depth|loss", code)
    script = (ROOT / "scripts/run_visual_supervision_diversity.py").read_text()
    assert "EpochSampler(name, train_pos, groups, CONFIG[\"seed\"])" in script
    assert "mae(val_pos)" in script and "rows_heldout.json" in script
    assert "S0/S1 optimization budget differs" in script
    assert "Worklog 69 model identity mismatch" in script and "Worklog 68 cache identity mismatch" in script
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name == "visual_supervision_sampling.py":
            continue
        assert "visual_supervision_sampling" not in path.read_text(), path
