import numpy as np
import pytest

from framepose.crops import (
    CROP_CONTRACT, CROP_MARGIN, CROP_RESOLUTION, CropBox, crop_box, geometry_in_crop, render_crop,
)


def _observation(points: dict[int, tuple[float, float]], size=(200, 100)):
    joints = np.zeros((17, 3))
    valid = np.zeros(17, dtype=bool)
    for index, (x, y) in points.items():
        joints[index] = (x / size[0], y / size[1], 1.0)
        valid[index] = True
    return joints, valid, size


def test_crop_is_square_centred_on_the_valid_joint_box():
    joints, valid, size = _observation({0: (60.0, 20.0), 5: (100.0, 60.0)})
    box = crop_box(joints, valid, size)
    assert box.side == pytest.approx(40.0 * (1 + 2 * CROP_MARGIN))
    assert box.x + box.side / 2 == pytest.approx(80.0)
    assert box.y + box.side / 2 == pytest.approx(40.0)


def test_crop_falls_back_to_a_centred_square_without_enough_observations():
    joints, valid, size = _observation({0: (60.0, 20.0)})
    box = crop_box(joints, valid, size)
    assert box.side == pytest.approx(100.0)
    assert (box.x, box.y) == pytest.approx((50.0, 0.0))


def test_geometry_mapping_is_the_documented_inverse_of_the_crop():
    joints, valid, size = _observation({0: (60.0, 20.0), 5: (100.0, 60.0), 9: (80.0, 40.0)})
    box = crop_box(joints, valid, size)
    features = geometry_in_crop(joints, valid, size, box)
    # The box centre maps to the origin of the normalized crop frame.
    assert features[9][:2] == pytest.approx((0.0, 0.0), abs=1e-9)
    # And the documented mapping inverts back to source pixels exactly.
    for index, (x, y) in ((0, (60.0, 20.0)), (5, (100.0, 60.0))):
        recovered = (features[index][:2] + 1.0) / 2.0 * box.side + np.asarray([box.x, box.y])
        assert recovered == pytest.approx((x, y))
    assert features[9][3] == 1.0
    assert features[1][3] == 0.0
    assert features[1][:3] == pytest.approx((0.0, 0.0, 0.0)), "invalid joints must carry no position"


def test_render_crop_is_an_exact_identity_for_a_full_frame_box():
    rng = np.random.default_rng(3)
    image = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    rendered = render_crop(image, CropBox(0.0, 0.0, 64.0), 64)
    assert np.array_equal(rendered, image)


def test_render_crop_pads_outside_the_source_with_the_declared_constant():
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    rendered = render_crop(image, CropBox(-16.0, -16.0, 16.0), 8)
    assert rendered.max() == 0, "the region outside the image must be constant padding"
    assert rendered.shape == (8, 8, 3)


def test_render_crop_is_deterministic_and_identical_for_every_rgb_candidate():
    rng = np.random.default_rng(11)
    image = (rng.random((80, 96, 3)) * 255).astype(np.uint8)
    joints, valid, size = _observation({0: (30.0, 20.0), 5: (60.0, 60.0)}, size=(96, 80))
    box = crop_box(joints, valid, size)
    first = render_crop(image, box, CROP_RESOLUTION)
    second = render_crop(image, box, CROP_RESOLUTION)
    assert np.array_equal(first, second)
    assert first.shape == (CROP_RESOLUTION, CROP_RESOLUTION, 3)


def test_crop_contract_is_recorded_for_reports():
    assert CROP_CONTRACT["resolution"] == CROP_RESOLUTION
    assert CROP_CONTRACT["margin"] == CROP_MARGIN
    assert "padding" in CROP_CONTRACT and "geometry_mapping" in CROP_CONTRACT
