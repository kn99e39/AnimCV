import pytest
import numpy as np

from scripts.export_pose_reconciliation_visual_review import (
    METHOD_MINIMUM_NORM,
    METHOD_SWIVEL,
    _blind_mapping,
    _bounds_for_view,
    _median_representative,
    _p90_representative,
)
from scripts import replay_pose_reconciliation as replay


def test_blind_assignment_is_stable_and_mixes_method_order():
    first = _blind_mapping("Rabc")
    assert first == _blind_mapping("Rabc")
    assert set(first) == {"A", "B"}
    assert set(first.values()) == {METHOD_MINIMUM_NORM, METHOD_SWIVEL}
    assert any(_blind_mapping(f"R{i}") != first for i in range(16))


def test_median_selection_uses_sample_id_as_deterministic_tie_break():
    sample_ids = {0: "z", 1: "a", 2: "m"}

    assert _median_representative([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, sample_ids) == 1


def test_p90_selection_is_confined_to_positive_tail():
    values = {0: -4.0, 1: 0.5, 2: 1.0, 3: 2.0, 4: 8.0}
    sample_ids = {index: f"s{index}" for index in values}
    row, threshold = _p90_representative(list(values), values, sample_ids)

    assert threshold == pytest.approx(6.2)
    assert row == 4


def test_three_dimensional_bounds_are_shared_and_cover_the_whole_skeleton():
    first = np.zeros((len(replay.JOINT_NAMES), 3), dtype=np.float64)
    second = first.copy()
    first[replay.JOINT_INDEX["left_elbow"]] = [0.1, -0.2, 0.4]
    second[replay.JOINT_INDEX["left_elbow"]] = [0.1, 0.5, 0.4]
    second[replay.JOINT_INDEX["head"]] = [0.0, 0.0, 1.8]
    valid = np.ones(len(replay.JOINT_NAMES), dtype=bool)
    chain = tuple(replay.JOINT_INDEX[name] for name in
                  ("left_shoulder", "left_elbow", "left_wrist"))

    bounds = _bounds_for_view([first, second], chain, "fixed_oblique", [valid, valid])
    reversed_bounds = _bounds_for_view([second, first], chain, "fixed_oblique", [valid, valid])
    projected_head = np.asarray([np.cos(np.deg2rad(35.0)) * 0.0
                                 - np.sin(np.deg2rad(35.0)) * 0.0,
                                 np.cos(np.deg2rad(22.0)) * 1.8])

    assert bounds == reversed_bounds
    assert bounds[2] <= projected_head[1] <= bounds[3]
