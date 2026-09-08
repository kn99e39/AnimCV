"""The local hinge plane, its two axes, and who can observe each of them.

docs/37. These pin the analytic claims the ownership diagnostic rests on, so a
later result cannot quietly depend on a convention chosen to make it come out.
"""

import numpy as np
import pytest

from common.canonical_pose import FORWARD_DEPTH_AXIS, JOINT_INDEX, bend_direction
from framepose.hinge_plane import (
    IMAGE_TO_CANONICAL, axis_transport, bend_components, local_basis, observed_screen_side,
)
from framepose.signs import SIGN_FIELD_NAMES, UNKNOWN, sign_state

_CHAIN = ("left_shoulder", "left_elbow", "left_wrist")
_FIELD = "left_elbow_forward_bend"


def _pose(**overrides):
    layout = {"pelvis": (0, 0, 0), "spine": (0, 0, 0.2), "thorax": (0, 0, 0.45),
              "neck": (0, 0, 0.5), "head": (0, 0, 0.7),
              "left_shoulder": (-0.18, 0.0, 0.45), "right_shoulder": (0.18, 0.05, 0.45),
              "left_elbow": (-0.28, 0.06, 0.2), "right_elbow": (0.28, -0.06, 0.2),
              "left_wrist": (-0.32, 0.0, -0.05), "right_wrist": (0.32, 0.0, -0.05),
              "left_hip": (-0.11, 0.02, 0), "right_hip": (0.11, -0.02, 0),
              "left_knee": (-0.12, 0.07, -0.45), "right_knee": (0.12, -0.07, -0.45),
              "left_ankle": (-0.12, 0.0, -0.9), "right_ankle": (0.12, 0.0, -0.9)}
    layout.update(overrides)
    pose = np.zeros((17, 3))
    for name, position in layout.items():
        pose[JOINT_INDEX[name]] = position
    return pose


# ------------------------------------------------------------- the basis ----

def test_local_basis_is_orthonormal_and_spans_the_perpendicular_plane():
    rng = np.random.default_rng(4)
    for _ in range(40):
        pose = _pose() + rng.normal(scale=0.1, size=(17, 3))
        basis = local_basis(pose, _CHAIN)
        if basis is None:
            continue
        a, u, v = basis["a_hat"], basis["u_hat"], basis["v_hat"]
        for vector in (a, u, v):
            assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-12)
        assert a @ u == pytest.approx(0.0, abs=1e-12)
        assert a @ v == pytest.approx(0.0, abs=1e-12)
        assert u @ v == pytest.approx(0.0, abs=1e-12)
        # u and v really do span the plane: the offset reconstructs exactly.
        components = bend_components(pose, _CHAIN)
        rebuilt = components["c_depth"] * u + components["c_screen"] * v
        np.testing.assert_allclose(rebuilt, components["offset"], atol=1e-12)


def test_v_hat_lies_in_the_canonical_image_plane():
    """cross(a, e_y) has zero Y component for ANY a, so the complementary bend
    axis is always perpendicular to camera depth -- it is visible geometry."""
    rng = np.random.default_rng(9)
    for _ in range(60):
        pose = _pose() + rng.normal(scale=0.15, size=(17, 3))
        basis = local_basis(pose, _CHAIN)
        if basis is None:
            continue
        assert basis["v_hat"][FORWARD_DEPTH_AXIS] == pytest.approx(0.0, abs=1e-12)


def test_c_depth_sign_equals_the_existing_hinge_sign_wherever_readable():
    """The decomposition introduces no new sign: c_depth is the contract's own
    forward-bend quantity, rescaled by a positive factor."""
    rng = np.random.default_rng(17)
    index = SIGN_FIELD_NAMES.index(_FIELD)
    checked = 0
    for _ in range(300):
        pose = _pose() + rng.normal(scale=0.12, size=(17, 3))
        valid = np.ones(17, dtype=bool)
        contract = int(sign_state(pose, valid)[index])
        if contract == UNKNOWN:
            continue
        components = bend_components(pose, _CHAIN)
        assert components is not None
        assert np.sign(components["c_depth"]) == contract
        # ...and it is exactly o_y / sqrt(f).
        assert components["c_depth"] == pytest.approx(
            components["offset"][FORWARD_DEPTH_AXIS] / np.sqrt(components["in_plane_fraction"]),
            rel=1e-12)
        checked += 1
    assert checked > 100


# ----------------------------------------------------- the 2D side predicate ----

def test_observed_screen_side_matches_the_3d_screen_component_under_the_measured_mapping():
    """With (X, Z) = (+x, -y), the 2D line-side predicate is the sign of
    c_screen. This is derived, not fitted: a projection of the 3D pose through
    the measured mapping must agree every time."""
    assert IMAGE_TO_CANONICAL == {"canonical_x_from": "+input_2d_x", "canonical_z_from": "-input_2d_y"}
    rng = np.random.default_rng(23)
    agreed = 0
    for _ in range(400):
        pose = _pose() + rng.normal(scale=0.15, size=(17, 3))
        components = bend_components(pose, _CHAIN)
        if components is None or components["c_screen"] == 0.0:
            continue
        # Project the SAME pose into observation coordinates: x = X, y = -Z.
        observation = np.zeros((17, 3))
        observation[:, 0] = pose[:, 0]
        observation[:, 1] = -pose[:, 2]
        observation[:, 2] = 1.0
        observed = observed_screen_side(observation, np.ones(17, dtype=bool), _CHAIN)
        assert observed["resolved"]
        assert observed["side"] == int(np.sign(components["c_screen"]))
        agreed += 1
    assert agreed > 300


def test_observed_screen_side_reports_degeneracy_instead_of_guessing():
    observation = np.zeros((17, 3))
    observation[:, 2] = 1.0
    for offset, name in ((0.0, "left_shoulder"), (0.5, "left_elbow"), (1.0, "left_wrist")):
        observation[JOINT_INDEX[name]] = (offset, offset, 1.0)   # exactly collinear
    result = observed_screen_side(observation, np.ones(17, dtype=bool), _CHAIN)
    assert result["resolved"] is False and result["side"] == 0
    assert "collinear" in result["reason"]
    assert result["cross"] == 0.0

    invalid = np.ones(17, dtype=bool)
    invalid[JOINT_INDEX["left_wrist"]] = False
    missing = observed_screen_side(observation, invalid, _CHAIN)
    assert missing["resolved"] is False and missing["cross"] is None


def test_observed_screen_side_uses_no_threshold():
    """Only an exact zero is unresolved; no collinearity tolerance exists."""
    import inspect

    source = inspect.getsource(observed_screen_side)
    body = source.split('"""')[2]
    assert "cross == 0.0" in body
    for banned in ("abs(cross) <", "1e-", "EPS", "tol"):
        assert banned not in body, banned


# ------------------------------------------------------- the counterfactual ----

def _rotation(axis, degrees):
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    angle = np.radians(degrees)
    skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + skew * np.sin(angle) + skew @ skew * (1 - np.cos(angle))


def test_axis_transport_removes_a_pure_axis_swing():
    """A swing about an axis perpendicular to the limb moves the limb axis and
    the bend together. Transporting the target bend through the same minimal
    rotation leaves exactly nothing: the whole historical error was axis
    orientation, not bend side."""
    target = _pose()
    pivot = target[JOINT_INDEX["left_shoulder"]]
    limb = target[JOINT_INDEX["left_wrist"]] - pivot
    swing_axis = np.cross(limb, np.array([0.0, 0.0, 1.0]))
    rotation = _rotation(swing_axis, 35.0)

    predicted = target.copy()
    for name in ("left_elbow", "left_wrist"):
        predicted[JOINT_INDEX[name]] = pivot + rotation @ (target[JOINT_INDEX[name]] - pivot)

    result = axis_transport(predicted, target, _CHAIN)
    assert result["resolved"]
    assert result["axis_angle_degrees"] == pytest.approx(35.0, abs=1e-6)
    assert result["historical_error_degrees"] > 1.0
    # arccos loses precision near 1, so this is exact to the limit of the metric.
    assert result["axis_normalized_error_degrees"] == pytest.approx(0.0, abs=1e-5)
    assert result["axis_normalized_flipped"] is False


def test_axis_transport_does_not_explain_away_a_twist_about_the_limb():
    """A twist about the limb axis leaves the axis exactly where it was and
    genuinely moves the bend to a different side. The minimal rotation is the
    identity there, so the counterfactual must preserve that error in full --
    otherwise it would launder real side error into 'axis geometry'."""
    target = _pose()
    pivot = target[JOINT_INDEX["left_shoulder"]]
    limb = target[JOINT_INDEX["left_wrist"]] - pivot
    rotation = _rotation(limb, 40.0)

    predicted = target.copy()
    predicted[JOINT_INDEX["left_elbow"]] = pivot + rotation @ (target[JOINT_INDEX["left_elbow"]] - pivot)

    result = axis_transport(predicted, target, _CHAIN)
    assert result["resolved"]
    assert result["axis_angle_degrees"] == pytest.approx(0.0, abs=1e-6)
    assert result["axis_normalized_error_degrees"] == pytest.approx(
        result["historical_error_degrees"], abs=1e-9)
    assert result["axis_normalized_error_degrees"] == pytest.approx(40.0, abs=1e-6)


def test_axis_transport_keeps_a_genuine_side_error_after_normalizing():
    """A bend mirrored within its own plane keeps its error once the axis is
    normalized -- the counterfactual must not explain away real side error."""
    target = _pose()
    predicted = target.copy()
    components = bend_components(target, _CHAIN)
    elbow = JOINT_INDEX["left_elbow"]
    predicted[elbow] = target[elbow] - 2.0 * components["c_screen"] * components["v_hat"]

    result = axis_transport(predicted, target, _CHAIN)
    assert result["resolved"]
    assert result["axis_normalized_error_degrees"] > 1.0


def test_axis_transport_refuses_an_antiparallel_axis_pair():
    target = _pose()
    predicted = target.copy()
    shoulder = target[JOINT_INDEX["left_shoulder"]]
    predicted[JOINT_INDEX["left_wrist"]] = shoulder - (target[JOINT_INDEX["left_wrist"]] - shoulder)
    predicted[JOINT_INDEX["left_elbow"]] = shoulder - (target[JOINT_INDEX["left_elbow"]] - shoulder)
    result = axis_transport(predicted, target, _CHAIN)
    assert result["resolved"] is False
    assert "antiparallel" in result["reason"]
    assert result["axis_angle_degrees"] == pytest.approx(180.0, abs=1e-6)
