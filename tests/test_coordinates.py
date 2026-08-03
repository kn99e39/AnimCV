import math

import pytest

from common.coordinates import camera_to_character, image_delta_to_camera


def test_image_delta_to_camera_maps_down_to_negative_up_axis():
    assert image_delta_to_camera((3.0, 4.0), depth=2.0) == (3.0, 2.0, -4.0)


def test_camera_to_character_applies_explicit_root_yaw():
    assert camera_to_character((1.0, 0.0, 2.0), math.pi / 2) == pytest.approx(
        (0.0, -1.0, 2.0)
    )


def test_camera_to_character_zero_yaw_is_identity():
    assert camera_to_character((1.0, 2.0, 3.0), 0.0) == pytest.approx((1.0, 2.0, 3.0))
