import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "attribute_yaw_tail.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("attribute_yaw_tail", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _row(action, yaw_error, shoulder=None, hip=None):
    return {
        "global_index": 0, "action": action, "view": None, "yaw_error_deg": yaw_error,
        "shoulder_error_deg": shoulder, "hip_error_deg": hip,
        "shoulder_screen_span": 0.1, "hip_screen_span": 0.1,
        "shoulder_confidence_mean": 1.0, "hip_confidence_mean": 1.0,
    }


def test_pair_yaw_errors_matches_official_root_yaw_error_on_a_single_pair():
    """Isolating one pair (hip invalid) must reproduce exactly what
    _root_yaw_error_degrees itself computes from the shoulder pair alone --
    otherwise this attribution wouldn't be trustworthy against the report."""
    module = _load_module()
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import _root_yaw_error_degrees

    n = len(H36M_NAMES)
    reference = np.zeros((n, 3), dtype=np.float32)
    reference[H36M_NAMES.index("left_shoulder"), :2] = [-1, 0]
    reference[H36M_NAMES.index("right_shoulder"), :2] = [1, 0]
    estimate = reference.copy()
    estimate[H36M_NAMES.index("left_shoulder"), :2] = [0, -1]
    estimate[H36M_NAMES.index("right_shoulder"), :2] = [0, 1]
    valid = np.zeros(n, dtype=bool)
    valid[[H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")]] = True

    pair_errors = module._pair_yaw_errors(estimate, reference, valid)
    official = _root_yaw_error_degrees(estimate, reference, valid)

    assert pair_errors["hip"] is None
    assert pair_errors["shoulder"] == pytest.approx(official)
    assert pair_errors["shoulder"] == pytest.approx(90.0)


def test_pair_yaw_errors_none_when_axis_degenerate_or_invalid():
    module = _load_module()
    from pose.pose_lifter import H36M_NAMES

    n = len(H36M_NAMES)
    reference = np.zeros((n, 3), dtype=np.float32)
    estimate = reference.copy()
    valid = np.zeros(n, dtype=bool)  # nothing valid: both pairs degenerate/invalid

    pair_errors = module._pair_yaw_errors(estimate, reference, valid)

    assert pair_errors == {"shoulder": None, "hip": None}


def test_error_bins_counts_cumulative_thresholds():
    module = _load_module()
    errors = [10.0, 32.0, 46.0, 91.0, 151.0, 151.0]

    bins = module._error_bins(errors, bins=(30.0, 45.0, 90.0, 150.0))

    assert bins[">=30deg"]["count"] == 5
    assert bins[">=45deg"]["count"] == 4
    assert bins[">=90deg"]["count"] == 3
    assert bins[">=150deg"]["count"] == 2
    assert bins[">=30deg"]["fraction"] == pytest.approx(5 / 6)


def test_tail_concentration_ranks_worst_actions_and_shares():
    module = _load_module()
    rows = (
        [_row("bad_seq", 40.0)] * 8
        + [_row("mild_seq", 35.0)] * 2
        + [_row("clean_seq", 5.0)] * 10
    )

    result = module._tail_concentration(rows, threshold=30.0)

    assert result["total_tail_frames"] == 10
    assert result["actions_touched"] == 2
    assert result["worst_actions"][0] == {"action": "bad_seq", "tail_frame_count": 8}
    top1 = next(item for item in result["top_n_concentration"] if item["top_n_actions"] == 1)
    assert top1["share"] == pytest.approx(0.8)


def test_temporal_runs_distinguishes_contiguous_streak_from_singletons():
    module = _load_module()
    # One action with a 4-frame contiguous tail streak, another with two
    # isolated tail frames far apart.
    streak = [_row("streak_seq", 40.0)] * 4 + [_row("streak_seq", 5.0)] * 4
    scattered = (
        [_row("scattered_seq", 40.0)] + [_row("scattered_seq", 5.0)] * 3
        + [_row("scattered_seq", 40.0)] + [_row("scattered_seq", 5.0)] * 3
    )
    rows = streak + scattered

    result = module._temporal_runs(rows, threshold=30.0)

    assert result["run_count"] == 3  # one streak-of-4 + two singletons
    assert result["max_run_length"] == 4
    assert result["singleton_runs"] == 2
    assert result["runs_of_5_or_more"] == 0


def test_pair_disagreement_flags_large_shoulder_hip_gaps_and_missing_pairs():
    module = _load_module()
    rows = [
        _row("a", 30.0, shoulder=10.0, hip=12.0),   # agree
        _row("a", 40.0, shoulder=5.0, hip=60.0),    # disagree >= 20
        _row("a", None, shoulder=8.0, hip=None),    # hip missing
        _row("a", None, shoulder=None, hip=9.0),    # shoulder missing
        _row("a", None, shoulder=None, hip=None),   # both missing
    ]

    result = module._pair_disagreement(rows, disagreement_deg=20.0)

    assert result["frames_with_both_pairs"] == 2
    assert result["frames_disagreeing_ge_20deg"] == 1
    assert result["frames_missing_hip_pair"] == 1
    assert result["frames_missing_shoulder_pair"] == 1
    assert result["frames_missing_both_pairs"] == 1


def test_build_attribution_reports_bins_and_frame_count_without_torch():
    """The full attribution builder only needs numpy -- confirms it can run
    (and be tested) without the optional torch training extra."""
    module = _load_module()
    from pose.pose_lifter import H36M_NAMES

    n = len(H36M_NAMES)
    frames = 3
    prediction = np.zeros((frames, n, 3), dtype=np.float32)
    targets = np.zeros((frames, n, 3), dtype=np.float32)
    valid = np.ones((frames, n), dtype=bool)
    inputs = np.zeros((frames, n, 3), dtype=np.float32)
    inputs[..., 2] = 1.0  # full confidence
    metadata = [{"action": "seq_a", "view": None} for _ in range(frames)]

    left, right = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    targets[:, left, :2] = [-1, 0]
    targets[:, right, :2] = [1, 0]
    prediction[:, left, :2] = [-1, 0]
    prediction[:, right, :2] = [1, 0]
    left_hip, right_hip = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    targets[:, left_hip, :2] = [-1, -1]
    targets[:, right_hip, :2] = [1, -1]
    prediction[:, left_hip, :2] = [-1, -1]
    prediction[:, right_hip, :2] = [1, -1]

    attribution = module._build_attribution(prediction, targets, valid, inputs, metadata)

    assert attribution["frame_count"] == frames
    assert attribution["yaw_valid_frame_count"] == frames
    assert attribution["yaw_error_mae_deg"] == pytest.approx(0.0)
    assert attribution["yaw_error_bins"][">=30deg"]["count"] == 0
    assert len(attribution["rows"]) == frames
