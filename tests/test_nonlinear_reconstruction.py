from __future__ import annotations

import numpy as np
import pytest

from common.canonical_pose import JOINT_INDEX, JOINT_NAMES
from framepose.nonlinear_reconstruction import (
    cubic_hermite, evaluate_reconstruction, find_gaps, quaternion_rotate,
    quaternion_squad, robust_local_bone_length, sequence_order, squad_control,
    swing_quaternion, timestamp_aware_tangent, reconstruct_nonlinear,
)


def test_sequence_order_and_gaps_never_cross_sequence():
    sequences = ["A", "A", "A", "B", "B"]
    usable = np.ones((5, len(JOINT_NAMES)), dtype=bool)
    joint = JOINT_INDEX["left_ankle"]
    usable[2, joint] = False
    usable[4, joint] = False
    grouped = sequence_order(sequences)
    gaps = find_gaps(usable, sequences)

    assert grouped["A"] == [0, 1, 2]
    assert grouped["B"] == [3, 4]
    gap_a = next(gap for gap in gaps if gap.sequence_id == "A" and gap.joint == joint)
    gap_b = next(gap for gap in gaps if gap.sequence_id == "B" and gap.joint == joint)
    assert gap_a.left_anchor == 1 and gap_a.right_anchor == -1
    assert gap_b.left_anchor == 3 and gap_b.right_anchor == -1


def test_timestamp_aware_hermite_hits_endpoints_and_derivatives():
    times = np.asarray([0.0, 0.5, 1.5, 2.0, 3.0])
    values = np.column_stack((times ** 2, np.zeros(len(times)), np.zeros(len(times))))
    tangent = timestamp_aware_tangent(values, times, 2, (1, 0), side="left")
    assert tangent is not None
    assert tangent[0] == pytest.approx(3.0)
    p0, p1 = values[2], values[3]
    m0 = np.asarray([3.0, 0.0, 0.0])
    m1 = np.asarray([4.0, 0.0, 0.0])
    assert np.allclose(cubic_hermite(p0, p1, m0, m1, times[2], times[3], times[2]), p0)
    assert np.allclose(cubic_hermite(p0, p1, m0, m1, times[2], times[3], times[3]), p1)


def test_squad_has_hemisphere_continuity_endpoints_and_swing_only_rotation():
    q0 = swing_quaternion(np.asarray([0.0, 0.0, 1.0]))
    q1 = swing_quaternion(np.asarray([1.0, 0.0, 0.0]))
    qp = swing_quaternion(np.asarray([0.0, -1.0, 0.0]))
    qn = swing_quaternion(np.asarray([0.0, 1.0, 0.0]))
    a0 = squad_control(qp, q0, q1)
    a1 = squad_control(q0, q1, qn)
    assert np.allclose(quaternion_squad(q0, q1, a0, a1, 0.0), q0)
    assert np.allclose(quaternion_squad(q0, q1, a0, a1, 1.0), q1)
    midpoint = quaternion_squad(q0, q1, a0, a1, 0.5)
    direction = quaternion_rotate(midpoint, np.asarray([0.0, 0.0, 1.0]))
    assert np.isfinite(direction).all()
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def _synthetic_pose(frame_count: int = 6) -> np.ndarray:
    offsets = np.asarray([[0.13 * i, 0.03 * (i % 3), 1.0 + 0.07 * i]
                          for i in range(len(JOINT_NAMES))], dtype=np.float64)
    motion = np.asarray([0.01, -0.02, 0.03])
    return offsets[None, :, :] + np.arange(frame_count)[:, None, None] * motion


def test_reconstruction_is_bounded_no_linear_fallback_and_preserves_outside_gap():
    poses = _synthetic_pose()
    target = poses.copy()
    joint = JOINT_INDEX["left_ankle"]
    poses[2:4, joint] += np.asarray([0.7, -0.4, 0.5])
    usable = np.ones(poses.shape[:2], dtype=bool)
    usable[2:4, joint] = False
    result = reconstruct_nonlinear(poses, usable, np.arange(len(poses), dtype=float), ["A"] * len(poses))

    assert result.changed[2, joint] and result.changed[3, joint]
    assert np.array_equal(result.recovered[~result.changed], poses[~result.changed])
    assert np.isfinite(result.recovered[2:4, joint]).all()
    assert np.array_equal(result.linear_control[0], poses[0])
    metrics = evaluate_reconstruction(poses, result.recovered, result.linear_control,
                                      target, np.ones_like(usable), result.changed,
                                      result.gaps, np.arange(len(poses), dtype=float),
                                      bone_records=result.bone_records)
    assert metrics["no_change_outside_changed"]
    assert metrics["reconstructable_target_valid"]["count"] == 2
    assert metrics["bone_length"]["error_mm"]["count"] >= 2


def test_insufficient_support_is_unresolved_and_bone_length_uses_trusted_median():
    poses = _synthetic_pose(4)
    usable = np.ones(poses.shape[:2], dtype=bool)
    joint = JOINT_INDEX["left_ankle"]
    usable[1:3, joint] = False
    result = reconstruct_nonlinear(poses, usable, np.arange(4, dtype=float), ["A"] * 4)
    assert result.unresolved
    assert not result.changed[:, joint].any()

    parent = JOINT_INDEX["left_knee"]
    poses[1, joint] = poses[1, parent] + np.asarray([0.0, 0.0, 1.0])
    poses[2, joint] = poses[2, parent] + np.asarray([0.0, 0.0, 3.0])
    usable[1:3, joint] = False
    assert robust_local_bone_length(poses, usable, parent, joint, [0, 1, 2, 3]) is not None
