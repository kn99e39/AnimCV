import math

import pytest

from retarget.axis_utils import axis_hint_to_vector, quaternion_from_axis_angle


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
