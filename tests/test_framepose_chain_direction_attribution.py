import math
import re
from pathlib import Path

import numpy as np
import pytest

from pose.framepose_chain_direction_attribution import (
    ARM, JOINT_INDEX, angle_degrees, chain_direction_row, component_hybrids,
    detector_vs_oracle_2d, image_plane_depth, joint_coordinate_errors, oracle_input_2d,
    project_animcv_to_pixels, segment_hybrids, xz_vs_image_direction,
)

ROOT = Path(__file__).resolve().parent.parent
K = np.array([[1000.0, 0, 540.0], [0, 1000.0, 960.0], [0, 0, 1]])


def _rot_z(points, degrees):
    a = math.radians(degrees)
    r = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return np.asarray(points) @ r.T


def test_detector_vs_oracle_2d_joint_and_direction_errors():
    oracle = np.array([[100., 100.], [100., 200.], [100., 300.]])
    detector = oracle + np.array([[0., 0.], [10., 0.], [0., 0.]])
    out = detector_vs_oracle_2d(detector, oracle)
    assert out["left_elbow_2d_error_px"] == pytest.approx(10)
    assert out["left_shoulder_2d_error_px"] == out["left_wrist_2d_error_px"] == 0
    assert out["shoulder_wrist_2d_direction_error_degrees"] == pytest.approx(0)
    assert out["shoulder_elbow_2d_direction_error_degrees"] == pytest.approx(math.degrees(math.atan(0.1)))
    zero = detector_vs_oracle_2d(oracle, oracle)
    assert all(v == pytest.approx(0) for k, v in zero.items() if k != "oracle_chain_length_px")


def test_oracle_input_follows_bank_convention_and_keeps_confidence():
    smpl = np.zeros((24, 2))
    smpl[16], smpl[17], smpl[1], smpl[2] = (300, 400), (500, 400), (320, 900), (480, 900)
    smpl[18], smpl[20], smpl[15] = (280, 600), (260, 800), (400, 200)
    detector = np.full((17, 3), 0.5)
    out = oracle_input_2d(smpl, (1080, 1920), detector)
    assert out[JOINT_INDEX["left_shoulder"]][:2] == pytest.approx((300 / 1080, 400 / 1920))
    assert out[JOINT_INDEX["neck"]][:2] == pytest.approx((400 / 1080, 400 / 1920))
    assert out[JOINT_INDEX["thorax"]][:2] == pytest.approx(out[JOINT_INDEX["neck"]][:2])
    assert out[JOINT_INDEX["pelvis"]][:2] == pytest.approx((400 / 1080, 900 / 1920))
    assert out[JOINT_INDEX["spine"]][:2] == pytest.approx((400 / 1080, 650 / 1920))
    assert (out[:, 2] == 0.5).all()
    arm_only = oracle_input_2d(smpl, (1080, 1920), detector, joints=ARM)
    assert arm_only[JOINT_INDEX["left_wrist"]][:2] == pytest.approx((260 / 1080, 800 / 1920))
    assert arm_only[JOINT_INDEX["neck"]] == pytest.approx(detector[JOINT_INDEX["neck"]])


def test_projection_matches_image_plane_conventions():
    point = np.array([[0.5, 4.0, 1.0]])  # right, forward, up
    u, v = project_animcv_to_pixels(point, K)[0]
    assert u == pytest.approx(540 + 125) and v == pytest.approx(960 - 250)
    # XZ of a 3D vector is compared with an image direction (u right, v down).
    assert xz_vs_image_direction(np.array([1.0, 7.0, 1.0]), np.array([1.0, -1.0])) == pytest.approx(0)
    with pytest.raises(ValueError):
        project_animcv_to_pixels(np.array([[0, -1.0, 0]]), K)


def test_xz_depth_decomposition_separates_components():
    gt = np.array([0.3, 0.0, -0.4])
    depth_only = np.array([0.3, 0.2, -0.4])
    out = image_plane_depth(depth_only, gt)
    assert out["xz_image_plane_angle_degrees"] == pytest.approx(0, abs=1e-9)
    assert out["full_3d_angle_degrees"] > 20
    assert out["forward_fraction_error"] == pytest.approx(0.2 / math.hypot(0.5, 0.2))
    plane_only = image_plane_depth(np.array([0.4, 0.0, -0.3]), gt)
    assert plane_only["forward_fraction_error"] == pytest.approx(0)
    assert plane_only["full_3d_angle_degrees"] == pytest.approx(plane_only["xz_image_plane_angle_degrees"])


def test_component_hybrids_substitute_raw_components_before_normalization():
    gt = np.array([0.3, 0.1, -0.4])
    h0 = np.array([0.3, 0.35, -0.4])  # pure depth error
    out = component_hybrids(h0, gt)
    assert out["h0_image_plane_gt_depth"] == pytest.approx(0, abs=1e-9)
    assert out["gt_image_plane_h0_depth"] == pytest.approx(out["h0_full"])
    assert out["gt_full"] == pytest.approx(0, abs=1e-9)
    h0 = np.array([0.4, 0.1, -0.3])  # pure image-plane error
    out = component_hybrids(h0, gt)
    assert out["gt_image_plane_h0_depth"] == pytest.approx(0, abs=1e-9)
    assert out["h0_image_plane_gt_depth"] == pytest.approx(out["h0_full"])


def test_segment_hybrids_use_fixed_gt_lengths():
    gt_u, gt_l = np.array([0, 0.3, 0]), np.array([0, 0, -0.25])
    # H0 upper direction wrong, wrong length too; lower direction correct.
    h0_u, h0_l = np.array([0.1, 0.5, 0]), np.array([0, 0, -0.05])
    out = segment_hybrids(h0_u, h0_l, gt_u, gt_l)
    assert out["gt_upper_h0_lower"] == pytest.approx(0, abs=1e-9)
    assert out["gt_upper_gt_lower"] == pytest.approx(0, abs=1e-9)
    assert out["h0_upper_gt_lower"] == pytest.approx(out["h0_upper_h0_lower"])
    assert out["lower_direction_error_degrees"] == pytest.approx(0, abs=1e-9)
    assert out["upper_direction_error_degrees"] == pytest.approx(angle_degrees(h0_u, gt_u))


def test_identical_h0_and_gt_give_zero_errors():
    chain = np.array([[0.2, 0.1, 0.4], [0.25, 0.05, 0.15], [0.3, -0.1, 0.0]])
    row = chain_direction_row(chain, chain)
    for key, value in row.items():
        if key.endswith("degrees") or key.startswith(("component_", "segment_")):
            assert value == pytest.approx(0, abs=1e-6), key
    assert row["forward_fraction_error"] == pytest.approx(0)
    errors = joint_coordinate_errors(chain, chain, np.zeros(3), np.zeros(3))
    assert all(v == pytest.approx(0) for v in errors.values())
    rotated = chain_direction_row(_rot_z(chain, 15), chain)
    assert rotated["full_3d_angle_degrees"] > 1


def test_gt_masking_rejects_non_finite_or_degenerate_chains():
    chain = np.array([[0.2, 0.1, 0.4], [0.25, 0.05, 0.15], [0.3, -0.1, 0.0]])
    bad = chain.copy()
    bad[2, 0] = np.nan
    with pytest.raises(ValueError):
        chain_direction_row(chain, bad)
    with pytest.raises(ValueError):
        chain_direction_row(chain, np.array([chain[0], chain[0], chain[2]]))


def test_no_production_target_leakage():
    production = [*sorted((ROOT / "src/framepose").glob("*.py")),
                  ROOT / "src/pose/framepose_bridge.py",
                  *sorted((ROOT / "src/motion").glob("animation_semantics*.py")),
                  *sorted((ROOT / "src/retarget").glob("framepose_*.py"))]
    pattern = re.compile(r"framepose_chain_direction_attribution")
    for path in production:
        assert not pattern.search(path.read_text()), path
    module = (ROOT / "src/pose/framepose_chain_direction_attribution.py").read_text()
    imports = [line for line in module.splitlines() if line.startswith(("import ", "from "))]
    assert "EVALUATION ONLY" in module
    assert not any(re.search(r"pickle|three_dpw|bank|train", line) for line in imports)
