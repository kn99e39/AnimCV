from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from framepose_oof_contract import assign_folds, group_key, validate_fold, validate_predictions


@dataclass
class Sample:
    sample_id: str
    source: str
    sequence_id: str
    split: str = "train"


def population():
    rows = [Sample(f"a{g}_{a}_{r}", "3DPW", f"3dpw:clip{g}:actor{a}")
            for g in range(10) for a in range(2) for r in range(g + 1)]
    rows += [Sample(f"b{g}_{r}", "BEDLAM", f"bedlam:scene:seq{g}") for g in range(10) for r in range(3)]
    rows += [Sample("validation", "3DPW", "3dpw:validation:actor0", "validation"),
             Sample("test", "BEDLAM", "bedlam:test:seq", "test")]
    return rows, np.arange(len(rows) - 2)


def test_assignment_is_label_blind_deterministic_and_capture_isolated():
    samples, train = population()
    a = assign_folds(samples, train)
    b = assign_folds(samples, train[::-1])
    assert np.array_equal(a, b)
    assert list(a[-2:]) == [-1, -1]
    assert set(a[train]) == set(range(5))
    for group in {group_key(s) for s in samples[:-2]}:
        assert len({a[p] for p in train if group_key(samples[p]) == group}) == 1


def test_assignment_refuses_heldout_rows():
    samples, train = population()
    with pytest.raises(ValueError, match="held-out"):
        assign_folds(samples, np.arange(len(samples)))


def test_fold_validation_rejects_row_and_capture_leakage():
    samples, train = population()
    fold = assign_folds(samples, train)
    pred, fit = train[fold[train] == 0], train[fold[train] != 0]
    validate_fold(samples, train, pred, fit, fit[:1])
    with pytest.raises(ValueError, match="OOF row"):
        validate_fold(samples, train, pred, np.append(fit, pred[0]), fit[:1])
    # Distinct rows/actors from the same capture are leakage-related.
    peers = [p for p in pred[1:] if group_key(samples[p]) == group_key(samples[pred[0]])]
    assert peers
    with pytest.raises(ValueError, match="sequence/capture"):
        validate_fold(samples, train, [pred[0]], np.append(fit, peers[0]), fit[:1])
    with pytest.raises(ValueError, match="unauthorized"):
        validate_fold(samples, train, pred, np.append(fit, len(samples) - 1), fit[:1])


def test_every_row_has_one_prediction_and_feature_order_is_exact():
    samples, train = population()
    fold = assign_folds(samples, train)
    records = [{"fold": f, "prediction_positions": train[fold[train] == f].tolist(),
                "refit_positions": train[fold[train] != f].tolist(),
                "selection_positions": train[fold[train] != f][:1].tolist()} for f in range(5)]
    ids = [samples[p].sample_id for p in train]
    assert validate_predictions(samples, train, fold, ids, np.zeros(len(train)), records)
    with pytest.raises(ValueError, match="identity"):
        validate_predictions(samples, train, fold, ids[::-1], np.zeros(len(train)), records)
    with pytest.raises(ValueError, match="exactly one"):
        validate_predictions(samples, train, fold, ids, np.zeros(len(train)), records + records[:1])
    with pytest.raises(ValueError, match="incomplete"):
        validate_predictions(samples, train, fold, ids, np.full(len(train), np.nan), records)


def test_oof_code_is_not_imported_by_production():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "src").rglob("*.py"):
        assert "framepose_oof_contract" not in path.read_text()
        assert "audit_framepose_oof_conditioning" not in path.read_text()


def test_comparison_refuses_changed_inference_and_preserves_pair_order():
    from copy import deepcopy
    from compare_framepose_oof_owners import merge

    rows = [{"sequence_id": "a", "frame_index": i, "next_frame_index": i + 1,
             "reference": {"motion": 1.}, "f_e1": .2,
             "candidates": {n: {"error": i} for n in ("b0", "b1", "c111", "g0", "g1")}} for i in (1, 2)]
    changed = deepcopy(rows)
    changed[0]["candidates"]["g0"]["error"] = 10.
    result = merge(rows, changed)
    assert [r["frame_index"] for r in result] == [1, 2]
    assert result[0]["candidates"]["g0_in"]["error"] == 1
    assert result[0]["candidates"]["g0_oof"]["error"] == 10.
    changed[0]["f_e1"] = .3
    with pytest.raises(ValueError, match="frozen"):
        merge(rows, changed)
    with pytest.raises(ValueError, match="populations"):
        merge(rows, rows[:1])


def test_sequence_cluster_uncertainty_keeps_frame_and_sequence_weighting_distinct():
    from run_framepose_visual_arm_ownership import _contrast

    items = [{"sequence_id": "long", "v": -2.} for _ in range(10)]
    items += [{"sequence_id": "short", "v": 20.}]
    out = _contrast(items, lambda r: r["v"])
    assert out["mean"] == 0.
    assert out["sequence_weighted_mean"] == 9.
    assert out["sequences_negative"] == 1 and out["sequences_positive"] == 1
    assert out["bootstrap"]["sequence_cluster"]["frame_weighted_ci95"] == [-2., 20.]
    assert set(out["bootstrap"]) == {"sequence_cluster", "moving_block_10", "moving_block_25"}
