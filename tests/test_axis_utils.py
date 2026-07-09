import math

import pytest

from retarget.axis_utils import (
    apply_rest_pose_correction,
    apply_rest_pose_correction_to_vector,
    axis_hint_to_vector,
    quaternion_conjugate,
    quaternion_from_axis_angle,
    quaternion_from_matrix,
    quaternion_from_vectors,
    quaternion_multiply,
    rotate_vector_by_quaternion,
)

_IDENTITY_MATRIX = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

# 90 degree rotation around +Z: x'=-y, y'=x, z'=z.
_ROTATE_Z_90_MATRIX = (
    (0.0, -1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _quaternion_angle(q):
    _, _, _, w = q
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(w))))


def test_axis_hint_to_vector_known_hints():
    assert axis_hint_to_vector("+Y") == (0.0, 1.0, 0.0)
    assert axis_hint_to_vector("-x") == (-1.0, 0.0, 0.0)


def test_axis_hint_to_vector_none_uses_default_view_axis():
    assert axis_hint_to_vector(None) == (0.0, 0.0, 1.0)


def test_axis_hint_to_vector_unknown_hint_raises():
    with pytest.raises(ValueError):
        axis_hint_to_vector("+W")


def test_quaternion_from_axis_angle_zero_angle_is_identity():
    assert quaternion_from_axis_angle((0.0, 0.0, 1.0), 0.0) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_quaternion_from_axis_angle_quarter_turn_around_z():
    x, y, z, w = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)

    assert (x, y, z, w) == pytest.approx((0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)))


def test_quaternion_from_axis_angle_normalizes_non_unit_axis():
    a = quaternion_from_axis_angle((0.0, 0.0, 2.0), math.pi / 2)
    b = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)

    assert a == pytest.approx(b)


def test_quaternion_from_axis_angle_zero_axis_returns_identity():
    assert quaternion_from_axis_angle((0.0, 0.0, 0.0), math.pi) == (0.0, 0.0, 0.0, 1.0)


def test_quaternion_multiply_identity_is_noop():
    q = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 3)
    identity = (0.0, 0.0, 0.0, 1.0)

    assert quaternion_multiply(q, identity) == pytest.approx(q)
    assert quaternion_multiply(identity, q) == pytest.approx(q)


def test_quaternion_multiply_composes_same_axis_rotations():
    quarter = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
    half = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi)

    composed = quaternion_multiply(quarter, quarter)

    assert composed == pytest.approx(half)


def test_quaternion_conjugate_negates_vector_part():
    q = quaternion_from_axis_angle((0.0, 1.0, 0.0), math.pi / 4)

    x, y, z, w = quaternion_conjugate(q)

    assert (x, y, z, w) == pytest.approx((-q[0], -q[1], -q[2], q[3]))


def test_quaternion_times_its_conjugate_is_identity():
    q = quaternion_from_axis_angle((1.0, 1.0, 0.0), 1.234)

    result = quaternion_multiply(q, quaternion_conjugate(q))

    assert result == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_quaternion_from_matrix_identity():
    assert quaternion_from_matrix(_IDENTITY_MATRIX) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_quaternion_from_matrix_matches_axis_angle_for_known_rotation():
    expected = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)

    assert quaternion_from_matrix(_ROTATE_Z_90_MATRIX) == pytest.approx(expected)


def test_apply_rest_pose_correction_none_matrix_is_noop():
    q = quaternion_from_axis_angle((0.0, 0.0, 1.0), 0.7)

    assert apply_rest_pose_correction(q, None) == q


def test_apply_rest_pose_correction_identity_matrix_is_noop():
    q = quaternion_from_axis_angle((0.0, 0.0, 1.0), 0.7)

    assert apply_rest_pose_correction(q, _IDENTITY_MATRIX) == pytest.approx(q)


def test_apply_rest_pose_correction_preserves_rotation_angle():
    # Conjugating by the rest orientation re-expresses the rotation in a
    # different basis, but the rotation angle itself is a similarity
    # invariant -- it should be unchanged regardless of the rest matrix.
    delta = quaternion_from_axis_angle((0.0, 0.0, 1.0), 1.1)

    corrected = apply_rest_pose_correction(delta, _ROTATE_Z_90_MATRIX)

    assert _quaternion_angle(corrected) == pytest.approx(_quaternion_angle(delta))


def test_apply_rest_pose_correction_to_vector_none_matrix_is_noop():
    assert apply_rest_pose_correction_to_vector((1.0, 2.0, 3.0), None) == (1.0, 2.0, 3.0)


def test_apply_rest_pose_correction_to_vector_rotates_into_bone_local_space():
    # Rest pose is rotated +90 deg around Z; a world-space delta of
    # (1, 0, 0) should come out as (0, -1, 0) in the bone's local frame.
    corrected = apply_rest_pose_correction_to_vector((1.0, 0.0, 0.0), _ROTATE_Z_90_MATRIX)

    assert corrected == pytest.approx((0.0, -1.0, 0.0))


def test_quaternion_from_vectors_identical_vectors_is_identity():
    assert quaternion_from_vectors((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 0.0, 0.0, 1.0)
    )


def test_quaternion_from_vectors_perpendicular_matches_axis_angle():
    q = quaternion_from_vectors((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    expected = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)

    assert q == pytest.approx(expected)


def test_quaternion_from_vectors_opposite_vectors_is_half_turn():
    q = quaternion_from_vectors((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))

    assert _quaternion_angle(q) == pytest.approx(math.pi)


def test_rotate_vector_by_quaternion_quarter_turn():
    q = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)

    result = rotate_vector_by_quaternion((1.0, 0.0, 0.0), q)

    assert result == pytest.approx((0.0, 1.0, 0.0))
