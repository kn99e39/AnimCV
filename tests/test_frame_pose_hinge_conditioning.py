"""Conditioning of the hinge branch constraint, and the minimum-norm candidate.

docs/36. docs/35 found that a CORRECT oracle hinge sign could demand a +3.1 m
depth correction. These pin the analytic explanation and the parameter-free
alternative, and pin that the historical operator is unchanged.
"""

import numpy as np
import pytest

from common.canonical_pose import FORWARD_DEPTH_AXIS, JOINT_INDEX, bend_direction
from framepose.branch_constraints import (
    CORRECTED, DEPTH_ONLY, HINGE_WRITE_POLICIES, MINIMUM_NORM, UNRESOLVED,
    apply_branch_constraints,
)
from framepose.signs import SIGN_FIELD_NAMES, UNIT_FORWARD_EPSILON, UNKNOWN, sign_state

_CHAIN = ("left_shoulder", "left_elbow", "left_wrist")
_FIELD = "left_elbow_forward_bend"


def _pose_with_axis(depth_alignment: float, offset=(0.0, 0.06, 0.10), length=0.40):
    """A left-arm chain whose axis is tilted toward the camera depth axis.

    `depth_alignment` 0 keeps the axis in the image plane; ->1 aligns it with
    +Y, driving f -> 0.
    """
    pose = np.zeros((17, 3))
    layout = {"pelvis": (0, 0, 0), "spine": (0, 0, 0.2), "thorax": (0, 0, 0.45),
              "neck": (0, 0, 0.5), "head": (0, 0, 0.7),
              "right_shoulder": (0.18, 0.0, 0.45), "right_elbow": (0.28, 0.05, 0.2),
              "right_wrist": (0.32, 0.0, -0.05),
              "left_hip": (-0.11, 0.0, 0), "right_hip": (0.11, 0.0, 0),
              "left_knee": (-0.12, 0.05, -0.45), "right_knee": (0.12, -0.05, -0.45),
              "left_ankle": (-0.12, 0.0, -0.9), "right_ankle": (0.12, 0.0, -0.9)}
    for name, position in layout.items():
        pose[JOINT_INDEX[name]] = position

    shoulder = np.array([-0.18, 0.0, 0.45])
    plane_direction = np.array([-1.0, 0.0, -0.6])
    plane_direction /= np.linalg.norm(plane_direction)
    direction = (1 - depth_alignment) * plane_direction + depth_alignment * np.array([0.0, 1.0, 0.0])
    direction /= np.linalg.norm(direction)
    wrist = shoulder + direction * length
    elbow = shoulder + direction * (length / 2) + np.asarray(offset, dtype=float)

    pose[JOINT_INDEX["left_shoulder"]] = shoulder
    pose[JOINT_INDEX["left_elbow"]] = elbow
    pose[JOINT_INDEX["left_wrist"]] = wrist
    return pose, np.ones(17, dtype=bool)


def _geometry(pose):
    p, m, d = (pose[JOINT_INDEX[name]] for name in _CHAIN)
    axis = d - p
    axis_squared = float(axis @ axis)
    relative = m - p
    offset = relative - axis * float(relative @ axis) / axis_squared
    f = float(axis[0] ** 2 + axis[2] ** 2) / axis_squared
    return axis, offset, f


def _request(field, value):
    requested = np.zeros(len(SIGN_FIELD_NAMES), dtype=np.int64)
    requested[SIGN_FIELD_NAMES.index(field)] = value
    return requested


def _readable(pose, valid):
    """Is the fixture's own branch readable? Below the contract's floors there
    is nothing to request, which is a property of the fixture, not a failure."""
    return int(sign_state(pose, valid)[SIGN_FIELD_NAMES.index(_FIELD)]) != UNKNOWN


def _apply(pose, valid, policy):
    index = SIGN_FIELD_NAMES.index(_FIELD)
    current = int(sign_state(pose, valid)[index])
    assert current != UNKNOWN
    requested = _request(_FIELD, -current)
    return (*apply_branch_constraints(pose, valid, requested, fields=[_FIELD],
                                      hinge_write_policy=policy), requested)


# --------------------------------------------------------- the explanation ----

def test_sqrt_f_is_exactly_the_largest_readable_depth_fraction():
    """sqrt(f) is not an arbitrary guard: it is the maximum |o_y|/|o| that ANY
    offset perpendicular to the limb axis can have, so below
    UNIT_FORWARD_EPSILON the requested branch is unobservable wherever the
    middle joint is put."""
    rng = np.random.default_rng(11)
    for alignment in (0.0, 0.3, 0.6, 0.9, 0.99):
        pose, _ = _pose_with_axis(alignment)
        axis, _, f = _geometry(pose)
        best = 0.0
        for _ in range(4000):
            w = rng.normal(size=3)
            w = w - axis * float(w @ axis) / float(axis @ axis)   # any offset perp to the axis
            best = max(best, abs(w[FORWARD_DEPTH_AXIS]) / np.linalg.norm(w))
        assert best <= np.sqrt(f) + 1e-9
        assert best > np.sqrt(f) * 0.99, "the bound must be attained, not merely respected"


def test_depth_only_displacement_is_exactly_one_over_sqrt_f_times_the_minimum():
    """The excess is a pure slide along the limb, which cannot change the bend."""
    checked_tilted = False
    for alignment in (0.0, 0.4, 0.7, 0.9):
        pose, valid = _pose_with_axis(alignment)
        axis, offset, f = _geometry(pose)
        if np.sqrt(f) < UNIT_FORWARD_EPSILON or not _readable(pose, valid):
            continue
        depth, depth_report, _ = _apply(pose, valid, DEPTH_ONLY)
        minimum, minimum_report, _ = _apply(pose, valid, MINIMUM_NORM)
        assert depth_report["fields"][_FIELD]["outcome"] == CORRECTED
        assert minimum_report["fields"][_FIELD]["outcome"] == CORRECTED

        elbow = JOINT_INDEX["left_elbow"]
        moved_depth = np.linalg.norm(depth[elbow] - pose[elbow])
        moved_min = np.linalg.norm(minimum[elbow] - pose[elbow])
        assert moved_depth / moved_min == pytest.approx(1.0 / np.sqrt(f), rel=1e-9)

        residual = depth[elbow] - minimum[elbow]
        if np.linalg.norm(residual) < 1e-12:
            # Limb axis in the image plane: the two operators coincide exactly.
            assert moved_depth == pytest.approx(moved_min, rel=1e-12)
            continue
        # Otherwise the whole difference is parallel to the limb axis, i.e.
        # geometrically inert for the branch being enforced.
        cosine = abs(float(residual @ axis) / (np.linalg.norm(residual) * np.linalg.norm(axis)))
        assert cosine == pytest.approx(1.0, abs=1e-9)
        checked_tilted = True
    assert checked_tilted, "the fixture must exercise at least one tilted axis"


def test_minimum_norm_is_bounded_by_twice_the_offset_however_singular_the_axis():
    """|correction| <= 2|o| for every admissible geometry, because
    |o_y| <= sqrt(f)|o| always. The depth-only correction has no such bound."""
    ratios = []
    for alignment in np.linspace(0.0, 0.985, 25):
        pose, valid = _pose_with_axis(float(alignment))
        _, offset, f = _geometry(pose)
        if np.sqrt(f) < UNIT_FORWARD_EPSILON or not _readable(pose, valid):
            continue
        minimum, _, _ = _apply(pose, valid, MINIMUM_NORM)
        depth, _, _ = _apply(pose, valid, DEPTH_ONLY)
        elbow = JOINT_INDEX["left_elbow"]
        moved_min = np.linalg.norm(minimum[elbow] - pose[elbow])
        moved_depth = np.linalg.norm(depth[elbow] - pose[elbow])
        assert moved_min <= 2 * np.linalg.norm(offset) + 1e-12
        ratios.append(moved_depth / moved_min)
    # The depth-only operator's excess really does grow as the axis tilts: it
    # is 1x when the limb lies in the image plane and several-fold by the time
    # the fixture reaches the edge of observability.
    assert min(ratios) == pytest.approx(1.0, abs=1e-9)
    assert max(ratios) > 4.0


# ------------------------------------------------- the candidate's contract ----

def test_minimum_norm_installs_the_identical_branch_and_bend_direction():
    for alignment in (0.0, 0.35, 0.7, 0.88):
        pose, valid = _pose_with_axis(alignment)
        _, _, f = _geometry(pose)
        if np.sqrt(f) < UNIT_FORWARD_EPSILON or not _readable(pose, valid):
            continue
        depth, _, requested = _apply(pose, valid, DEPTH_ONLY)
        minimum, _, _ = _apply(pose, valid, MINIMUM_NORM)
        index = SIGN_FIELD_NAMES.index(_FIELD)
        assert int(sign_state(minimum, valid)[index]) == int(requested[index])
        # Not merely the same sign: the same unit bend direction, to 1e-12.
        joints = [pose[JOINT_INDEX[name]] for name in _CHAIN]
        np.testing.assert_allclose(
            bend_direction(minimum[JOINT_INDEX["left_elbow"]], joints[0], joints[2]),
            bend_direction(depth[JOINT_INDEX["left_elbow"]], joints[0], joints[2]), atol=1e-12)


def test_minimum_norm_preserves_offset_magnitude_and_both_bone_lengths():
    """The invariants the two operators do and do not share, stated exactly."""
    pose, valid = _pose_with_axis(0.6)
    _, offset, _ = _geometry(pose)
    depth, _, _ = _apply(pose, valid, DEPTH_ONLY)
    minimum, _, _ = _apply(pose, valid, MINIMUM_NORM)
    elbow = JOINT_INDEX["left_elbow"]
    shoulder, wrist = JOINT_INDEX["left_shoulder"], JOINT_INDEX["left_wrist"]

    for corrected in (depth, minimum):
        # Both keep the endpoints and the perpendicular offset magnitude.
        np.testing.assert_array_equal(corrected[shoulder], pose[shoulder])
        np.testing.assert_array_equal(corrected[wrist], pose[wrist])
        _, moved_offset, _ = _geometry(corrected)
        assert np.linalg.norm(moved_offset) == pytest.approx(np.linalg.norm(offset), abs=1e-12)

    # Only the minimum-norm operator keeps the bone lengths, because its
    # displacement is perpendicular to the limb axis.
    for near, far in ((shoulder, elbow), (elbow, wrist)):
        original = np.linalg.norm(pose[far] - pose[near])
        assert np.linalg.norm(minimum[far] - minimum[near]) == pytest.approx(original, abs=1e-12)
        assert abs(np.linalg.norm(depth[far] - depth[near]) - original) > 1e-6

    # Only the depth-only operator keeps the middle joint's screen-space X/Z.
    assert depth[elbow, 0] == pose[elbow, 0] and depth[elbow, 2] == pose[elbow, 2]
    assert not (minimum[elbow, 0] == pose[elbow, 0] and minimum[elbow, 2] == pose[elbow, 2])


def test_below_the_guard_the_branch_is_unreadable_so_both_policies_refuse():
    """Because sqrt(f) bounds |o_y|/|o|, a geometry under the guard cannot read
    ANY branch. So the guard is never a rejection of a readable request: it
    fires exactly when the oracle asks for a branch the prediction's own
    geometry cannot express, whichever write policy is used."""
    pose, valid = _pose_with_axis(0.995)
    _, _, f = _geometry(pose)
    assert np.sqrt(f) < UNIT_FORWARD_EPSILON
    index = SIGN_FIELD_NAMES.index(_FIELD)
    # The prediction's own branch is necessarily unreadable there.
    assert int(sign_state(pose, valid)[index]) == UNKNOWN

    # The oracle can still request one, which is how the path is reached.
    for wanted in (1, -1):
        for policy in HINGE_WRITE_POLICIES:
            _, report = apply_branch_constraints(pose, valid, _request(_FIELD, wanted),
                                                 fields=[_FIELD], hinge_write_policy=policy)
            entry = report["fields"][_FIELD]
            assert entry["outcome"] == UNRESOLVED
            assert "not observable" in entry["reason"] or "singular" in entry["reason"]
            assert entry["axis_in_plane_fraction"] == pytest.approx(f, rel=1e-12)


def test_minimum_norm_is_deterministic_and_introduces_no_threshold():
    import inspect

    from framepose.branch_constraints import _hinge_min_norm_correction

    pose, valid = _pose_with_axis(0.5)
    first, _, _ = _apply(pose, valid, MINIMUM_NORM)
    second, _, _ = _apply(pose, valid, MINIMUM_NORM)
    np.testing.assert_array_equal(first, second)

    # No cap, clamp, step size or new tolerance: the only literals are the
    # existing degeneracy guards the historical operator already used.
    # Check the executable body, not the prose that explains what it avoids.
    source = inspect.getsource(_hinge_min_norm_correction)
    body = source.split('"""')[2]
    for banned in ("clip", "clamp", "np.minimum", "np.maximum", "cap"):
        assert banned not in body, banned
    assert body.count("1e-12") == 1 and "UNIT_FORWARD_EPSILON" in body
    # Every numeric literal in the body must already exist in the contract or
    # be a structural constant, never a tuned magnitude.
    import re

    literals = {token for token in re.findall(r"(?<![\w.])\d+\.\d+(?:e-?\d+)?", body)}
    assert literals <= {"2.0", "1.0", "0.0"}, literals


def test_the_historical_operator_is_unmodified():
    """docs/33's operator must still be reachable and still be the default."""
    import inspect

    from framepose.branch_constraints import _hinge_correction

    source = inspect.getsource(_hinge_correction)
    assert "delta = -2.0 * float(offset[FORWARD_DEPTH_AXIS]) / in_plane_fraction" in source
    assert "corrected[middle_index, FORWARD_DEPTH_AXIS] = pose[middle_index, FORWARD_DEPTH_AXIS] + delta" in source

    pose, valid = _pose_with_axis(0.5)
    index = SIGN_FIELD_NAMES.index(_FIELD)
    requested = _request(_FIELD, -int(sign_state(pose, valid)[index]))
    default, report = apply_branch_constraints(pose, valid, requested, fields=[_FIELD])
    explicit, _ = apply_branch_constraints(pose, valid, requested, fields=[_FIELD],
                                           hinge_write_policy=DEPTH_ONLY)
    np.testing.assert_array_equal(default, explicit)
    assert report["hinge_write_policy"] == DEPTH_ONLY


def test_unknown_hinge_write_policy_is_refused():
    pose, valid = _pose_with_axis(0.4)
    with pytest.raises(ValueError, match="unknown hinge write policy"):
        apply_branch_constraints(pose, valid, _request(_FIELD, 1), fields=[_FIELD],
                                 hinge_write_policy="nope")
