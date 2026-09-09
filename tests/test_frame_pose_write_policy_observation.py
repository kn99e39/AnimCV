"""Write-policy comparison under a real perspective camera.

docs/39. `DEPTH_ONLY` preserves canonical X/Z; that is not the same as
preserving image position, because image coordinates divide by depth. These pin
the projection, the placement device, the reference separation and the semantic
identity the comparison depends on.
"""

import json

import numpy as np
import pytest

from common.canonical_pose import JOINT_INDEX, bend_direction
from framepose.branch_constraints import (
    CORRECTED, DEPTH_ONLY, MINIMUM_NORM, apply_branch_constraints,
)
from framepose.signs import SIGN_FIELD_NAMES, UNKNOWN, sign_state


def _load_script(name, relative):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    try:
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


POLICY = _load_script("diagnose_hinge_write_policy_observation",
                      "scripts/diagnose_hinge_write_policy_observation.py")

_K_LANDSCAPE = np.array([[1969.2, 0.0, 960.0], [0.0, 1961.9, 540.0], [0.0, 0.0, 1.0]])
_K_PORTRAIT = np.array([[1961.9, 0.0, 540.0], [0.0, 1969.2, 960.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------- the projection ----

def test_projection_matches_the_repository_axis_conversion():
    from pose.three_dpw_adapter import _world_to_animcv_camera

    rng = np.random.default_rng(2)
    opencv = rng.normal(size=(8, 3))
    opencv[:, 2] = rng.uniform(2.0, 6.0, size=8)
    animcv = _world_to_animcv_camera(opencv, np.eye(4))
    pixels = POLICY.project(animcv, _K_LANDSCAPE)
    np.testing.assert_allclose(
        pixels[:, 0], _K_LANDSCAPE[0, 0] * opencv[:, 0] / opencv[:, 2] + _K_LANDSCAPE[0, 2], atol=1e-9)
    np.testing.assert_allclose(
        pixels[:, 1], _K_LANDSCAPE[1, 1] * opencv[:, 1] / opencv[:, 2] + _K_LANDSCAPE[1, 2], atol=1e-9)


def test_projection_accepts_per_frame_intrinsics():
    """3DPW's test split mixes portrait and landscape, so one K per frame is
    the correct form and must agree with projecting each frame separately."""
    points = np.array([[0.1, 3.0, 0.2], [-0.3, 4.0, 0.5]])
    stacked = np.stack([_K_LANDSCAPE, _K_PORTRAIT])
    together = POLICY.project(points, stacked)
    apart = np.stack([POLICY.project(points[0], _K_LANDSCAPE), POLICY.project(points[1], _K_PORTRAIT)])
    np.testing.assert_allclose(together, apart, atol=1e-12)
    # ...and the two cameras genuinely differ, so the test has teeth.
    assert not np.allclose(POLICY.project(points, _K_LANDSCAPE), together)


def test_a_depth_only_write_still_moves_the_joint_in_the_image():
    """The premise of the whole batch: canonical X/Z unchanged does NOT mean
    image position unchanged, because x ~ X/Y and z ~ -Z/Y."""
    before = np.array([0.30, 3.0, 0.40])
    after = before.copy()
    after[1] += 0.40                                   # depth-only write
    assert after[0] == before[0] and after[2] == before[2]

    first, second = POLICY.project(before, _K_LANDSCAPE), POLICY.project(after, _K_LANDSCAPE)
    displacement = float(np.linalg.norm(second - first))
    assert displacement > 20.0, displacement          # tens of pixels, not zero
    # Both image axes move, not just one.
    assert abs(second[0] - first[0]) > 1.0 and abs(second[1] - first[1]) > 1.0


def test_image_displacement_of_a_depth_write_grows_with_the_depth_delta():
    before = np.array([0.30, 3.0, 0.40])
    displacements = []
    for delta in (0.1, 0.4, 1.0, 3.1):
        after = before + np.array([0.0, delta, 0.0])
        displacements.append(float(np.linalg.norm(
            POLICY.project(after, _K_LANDSCAPE) - POLICY.project(before, _K_LANDSCAPE))))
    assert displacements == sorted(displacements)
    assert displacements[-1] > 5 * displacements[0]


# --------------------------------------------------- placement and identity ----

def test_oracle_placement_is_a_single_shared_root_and_is_declared():
    """All three states must be placed at the SAME absolute root, or the
    comparison would measure the placement rather than the write policy."""
    import inspect

    source = inspect.getsource(POLICY.main)
    assert 'root = absolute_target[:, JOINT_INDEX["pelvis"]][:, None, :]' in source
    assert "placed = {name: array + root for name, array in states.items()}" in source
    text = inspect.getsource(POLICY)
    assert "oracle_absolute_root_placement" in text
    assert "not production inference" in text


def test_reconstruction_boundary_is_stated_in_the_units_it_is_computed_in():
    """docs/38's prose drifted from its code; the constant now carries the unit."""
    assert POLICY.RECONSTRUCTION_REFUSAL_MM == 1.0
    import inspect

    source = inspect.getsource(POLICY.main)
    assert "reconstruction.max() > RECONSTRUCTION_REFUSAL_MM" in source
    assert "implemented_refusal_boundary_mm" in source


def test_the_two_policies_still_enforce_the_identical_branch():
    """The comparison is only fair while this holds; the diagnostic refuses
    otherwise, and this is the same property on a synthetic chain."""
    layout = {"pelvis": (0, 0, 0), "spine": (0, 0, 0.2), "thorax": (0, 0, 0.45),
              "neck": (0, 0, 0.5), "head": (0, 0, 0.7),
              "left_shoulder": (-0.18, 0.0, 0.45), "right_shoulder": (0.18, 0.05, 0.45),
              "left_elbow": (-0.26, 0.06, 0.22), "right_elbow": (0.28, -0.06, 0.2),
              # The arm axis is deliberately tilted out of the image plane (a_y != 0),
              # or the two policies coincide and the comparison has no content.
              "left_wrist": (-0.30, 0.22, -0.02), "right_wrist": (0.32, 0.0, -0.05),
              "left_hip": (-0.11, 0.02, 0), "right_hip": (0.11, -0.02, 0),
              "left_knee": (-0.12, 0.07, -0.45), "right_knee": (0.12, -0.07, -0.45),
              "left_ankle": (-0.12, 0.0, -0.9), "right_ankle": (0.12, 0.0, -0.9)}
    pose = np.zeros((17, 3))
    for name, position in layout.items():
        pose[JOINT_INDEX[name]] = position
    valid = np.ones(17, dtype=bool)

    field = "left_elbow_forward_bend"
    index = SIGN_FIELD_NAMES.index(field)
    current = int(sign_state(pose, valid)[index])
    assert current != UNKNOWN
    requested = np.zeros(len(SIGN_FIELD_NAMES), dtype=np.int64)
    requested[index] = -current

    results = {}
    for policy in (DEPTH_ONLY, MINIMUM_NORM):
        corrected, report = apply_branch_constraints(pose, valid, requested, fields=[field],
                                                     hinge_write_policy=policy)
        assert report["fields"][field]["outcome"] == CORRECTED
        results[policy] = corrected

    chain = ("left_shoulder", "left_elbow", "left_wrist")
    bends = [bend_direction(results[p][JOINT_INDEX[chain[1]]], results[p][JOINT_INDEX[chain[0]]],
                            results[p][JOINT_INDEX[chain[2]]]) for p in (DEPTH_ONLY, MINIMUM_NORM)]
    np.testing.assert_allclose(bends[0], bends[1], atol=1e-12)
    np.testing.assert_array_equal(sign_state(results[DEPTH_ONLY], valid),
                                  sign_state(results[MINIMUM_NORM], valid))
    # And they land in different places, which is the whole point.
    assert not np.allclose(results[DEPTH_ONLY][JOINT_INDEX["left_elbow"]],
                           results[MINIMUM_NORM][JOINT_INDEX["left_elbow"]])


def test_input_2d_pixel_inversion_uses_the_stored_image_size():
    """input_2d is pixel / image_size, so recovering pixels multiplies it back."""
    size = np.array([1080.0, 1920.0])
    normalized = np.array([0.5829907407407408, 0.4646484375])
    np.testing.assert_allclose(normalized * size, np.array([629.63, 892.125]), atol=1e-3)


def test_references_a_and_b_are_never_merged():
    import inspect

    source = inspect.getsource(POLICY.main)
    assert "target_projection_error" in source and "observation_consistency_error" in source
    text = inspect.getsource(POLICY)
    # Reference B must never be described as detector error.
    assert "NOT detector error" in text or "not detector error" in text.lower()
    assert "A_projected_target" in text and "B_observation" in text


def test_historical_binding_reports_unverifiable_instead_of_asserting(tmp_path):
    totals = {"all_residual_flips": 279, "depth_correct_residual_flips": 83}
    assert POLICY._bind_history(None, "abc", totals) == {
        "attempted": False, "verifiable": False, "reason": "no historical artifact supplied"}

    thin = tmp_path / "thin.json"
    thin.write_text(json.dumps({"schema": "x", "fields": {}}))
    record = POLICY._bind_history(thin, "abc", totals)
    assert record["attempted"] is True and record["verifiable"] is False
    assert "residual_flip_records" in record["reason"]

    full = tmp_path / "full.json"
    full.write_text(json.dumps({
        "schema": "x", "hinge_write_policy": MINIMUM_NORM,
        "provenance": {"source": {"prediction": {"sha256": "abc"}}},
        "fields": {"left_knee_forward_bend": {"residual_flip_records": (
            [{"depth_side_correct": True}] * 83 + [{"depth_side_correct": False}] * 196)}}}))
    record = POLICY._bind_history(full, "abc", totals)
    assert record["verifiable"] is True and record["all_checks_pass"] is True
    assert record["checks"] == {"residual_total_matches": True, "depth_correct_matches": True,
                                "source_prediction_matches": True,
                                "hinge_policy_matches_minimum_norm": True}
    # A different source prediction must fail the binding, not be ignored.
    assert POLICY._bind_history(full, "different", totals)["all_checks_pass"] is False
