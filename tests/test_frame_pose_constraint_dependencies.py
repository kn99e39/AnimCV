"""Constraint dependency graph and the dependency-aware bilateral operator.

docs/34 asks whether docs/33's bilateral failure belongs to the hard-constraint
abstraction or to an operator that moved the anchor joints and left the limb
geometry that depends on them behind. That question only has a clean answer if
the read/write/dependent relation is machine-readable and pinned against the
Sign Contract's own derivation, and if the new operator's preservation claims
are exact rather than approximate.
"""

import numpy as np
import pytest

from common.canonical_pose import FORWARD_DEPTH_AXIS, JOINT_INDEX, bend_direction
from framepose.branch_constraints import (
    ALREADY_SATISFIED, CORRECTED, apply_branch_constraints,
)
from framepose.constraint_graph import (
    ANCHOR_ONLY, BILATERAL_DEPENDENT_CHAINS, DEPENDENCY_AWARE, READ_JOINTS, WRITE_POLICIES,
    dependency_graph, field_entry, write_groups,
)
from framepose.signs import SIGN_FIELD_NAMES, UNKNOWN, sign_state


_LAYOUT = {
    "pelvis": (0, 0, 0), "spine": (0, 0, 0.2), "thorax": (0, 0, 0.45),
    "neck": (0, 0, 0.5), "head": (0, 0, 0.7),
    "left_shoulder": (-0.18, 0.0, 0.45), "right_shoulder": (0.18, 0.05, 0.45),
    "left_elbow": (-0.28, 0.06, 0.2), "right_elbow": (0.28, -0.06, 0.2),
    "left_wrist": (-0.32, 0.0, -0.05), "right_wrist": (0.32, 0.0, -0.05),
    "left_hip": (-0.11, -0.0393, 0), "right_hip": (0.11, -0.0839, 0),
    "left_knee": (-0.12, -0.0119, -0.45), "right_knee": (0.12, -0.0646, -0.45),
    "left_ankle": (-0.12, 0.0711, -0.9), "right_ankle": (0.12, -0.1075, -0.9),
}


def _pose(**overrides):
    layout = {**_LAYOUT, **overrides}
    pose = np.zeros((17, 3))
    for name, position in layout.items():
        pose[JOINT_INDEX[name]] = position
    return pose, np.ones(17, dtype=bool)


def _request(field, value):
    requested = np.zeros(len(SIGN_FIELD_NAMES), dtype=np.int64)
    requested[SIGN_FIELD_NAMES.index(field)] = value
    return requested


def _bend(pose, joint):
    from framepose.signs import HINGE_CHAINS_BY_JOINT

    proximal, middle, distal = HINGE_CHAINS_BY_JOINT[joint]
    return bend_direction(pose[JOINT_INDEX[middle]], pose[JOINT_INDEX[proximal]],
                          pose[JOINT_INDEX[distal]])


# ----------------------------------------------------- the dependency graph ----

def test_read_joints_are_pinned_against_the_sign_derivation_not_signfield_joints():
    """A joint outside a field's read set must never be able to change it, and
    every joint inside it must be able to."""
    rng = np.random.default_rng(7)
    for field in SIGN_FIELD_NAMES:
        index = SIGN_FIELD_NAMES.index(field)
        read = set(READ_JOINTS[field])
        outside = [name for name in JOINT_INDEX if name not in read]
        movable = set()
        for _ in range(40):
            pose, valid = _pose()
            pose = pose + rng.normal(scale=0.05, size=pose.shape)
            reference = sign_state(pose, valid)[index]
            for name in outside:
                moved = pose.copy()
                moved[JOINT_INDEX[name]] += rng.normal(scale=0.4, size=3)
                assert sign_state(moved, valid)[index] == reference, (field, name)
            for name in read:
                moved = pose.copy()
                moved[JOINT_INDEX[name]] += rng.normal(scale=0.4, size=3)
                if sign_state(moved, valid)[index] != reference:
                    movable.add(name)
        assert movable == read, f"{field}: read set overstated, {read - movable} never mattered"


def test_dependency_graph_matches_the_stated_relation():
    graph = dependency_graph(ANCHOR_ONLY)
    hip = graph["fields"]["hip_forward_depth"]
    assert hip["read_joints"] == ["left_hip", "right_hip"]
    assert hip["write_joints"] == ["right_hip", "left_hip"] or set(hip["write_joints"]) == {
        "left_hip", "right_hip"}
    assert hip["downstream_dependent_fields"] == ["left_knee_forward_bend",
                                                  "right_knee_forward_bend"]
    knee = graph["fields"]["left_knee_forward_bend"]
    assert knee["read_joints"] == ["left_hip", "left_knee", "left_ankle"]
    assert knee["write_joints"] == ["left_knee"]
    assert knee["downstream_dependent_fields"] == []

    # torso_facing reads four joints, though SignField.joints routes nine.
    from framepose.signs import SIGN_FIELDS

    routed = next(item for item in SIGN_FIELDS if item.name == "torso_facing").joints
    assert len(routed) == 9 and set(READ_JOINTS["torso_facing"]) == {
        "pelvis", "thorax", "left_shoulder", "right_shoulder"}
    # ...and a hinge field routes only its middle joint but reads all three.
    elbow = next(item for item in SIGN_FIELDS if item.name == "left_elbow_forward_bend")
    assert elbow.joints == ("left_elbow",)
    assert READ_JOINTS["left_elbow_forward_bend"] == ("left_shoulder", "left_elbow", "left_wrist")


def test_dependency_aware_policy_moves_the_hinge_dependents_into_one_write_group():
    anchor = dependency_graph(ANCHOR_ONLY)["fields"]
    aware = dependency_graph(DEPENDENCY_AWARE)["fields"]

    for field, hinges in (("hip_forward_depth", ["left_knee_forward_bend",
                                                 "right_knee_forward_bend"]),
                          ("shoulder_forward_depth", ["left_elbow_forward_bend",
                                                      "right_elbow_forward_bend"])):
        assert anchor[field]["dependents_preserved_exactly"] == []
        assert set(hinges) <= set(anchor[field]["dependents_at_risk"])
        assert set(hinges) <= set(aware[field]["dependents_preserved_exactly"])
        assert not set(hinges) & set(aware[field]["dependents_at_risk"])

    # The hinge operator is frozen and writes one joint under either policy.
    for policy in WRITE_POLICIES:
        assert write_groups("left_knee_forward_bend", policy) == (("left_knee",),)

    # torso_facing stays at risk under both: its read set spans both sides.
    assert aware["shoulder_forward_depth"]["dependents_at_risk"] == ["torso_facing"]
    assert aware["hip_forward_depth"]["dependents_at_risk"] == []


def test_torso_facing_sign_numerator_is_invariant_to_any_depth_only_write():
    """cross(up, right)'s +Y component is up_z*right_x - up_x*right_z, which
    contains no Y at all -- so a depth-only bilateral write can only move
    torso_facing through the unit normalization, never through its numerator."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        pose, _ = _pose()
        pose = pose + rng.normal(scale=0.05, size=pose.shape)
        up = pose[JOINT_INDEX["thorax"]] - pose[JOINT_INDEX["pelvis"]]
        right = pose[JOINT_INDEX["right_shoulder"]] - pose[JOINT_INDEX["left_shoulder"]]
        before = float(np.cross(up, right)[FORWARD_DEPTH_AXIS])
        moved = pose.copy()
        for name in ("left_shoulder", "right_shoulder", "pelvis", "thorax"):
            moved[JOINT_INDEX[name], FORWARD_DEPTH_AXIS] += rng.normal(scale=0.5)
        up = moved[JOINT_INDEX["thorax"]] - moved[JOINT_INDEX["pelvis"]]
        right = moved[JOINT_INDEX["right_shoulder"]] - moved[JOINT_INDEX["left_shoulder"]]
        assert float(np.cross(up, right)[FORWARD_DEPTH_AXIS]) == pytest.approx(before, abs=1e-12)


def test_unknown_write_policy_is_refused():
    pose, valid = _pose()
    with pytest.raises(ValueError, match="unknown bilateral write policy"):
        apply_branch_constraints(pose, valid, _request("hip_forward_depth", 1),
                                 fields=["hip_forward_depth"], bilateral_write_policy="nope")
    with pytest.raises(ValueError, match="unknown write policy"):
        write_groups("hip_forward_depth", "nope")


# ------------------------------------------ the dependency-aware operator ----

def _hip_flip_case():
    """A pose where the anchor-only swap destroys BOTH knee signs."""
    pose, valid = _pose()
    before = sign_state(pose, valid)
    hip = SIGN_FIELD_NAMES.index("hip_forward_depth")
    assert before[hip] != UNKNOWN
    return pose, valid, before, _request("hip_forward_depth", -int(before[hip]))


def test_anchor_only_swap_really_does_break_the_dependent_hinges():
    """Causal teeth: without this, the dependency-aware preservation test could
    pass on a fixture where nothing was ever at risk."""
    pose, valid, before, requested = _hip_flip_case()
    corrected, report = apply_branch_constraints(
        pose, valid, requested, fields=["hip_forward_depth"], bilateral_write_policy=ANCHOR_ONLY)
    assert report["fields"]["hip_forward_depth"]["outcome"] == CORRECTED
    after = sign_state(corrected, valid)

    for field in ("left_knee_forward_bend", "right_knee_forward_bend"):
        index = SIGN_FIELD_NAMES.index(field)
        assert before[index] != UNKNOWN
        assert after[index] != before[index], f"{field} was supposed to be damaged"
    # And the bend directions themselves moved, not merely the thresholded sign.
    for joint in ("left_knee", "right_knee"):
        assert not np.allclose(_bend(pose, joint), _bend(corrected, joint))


def test_dependency_aware_preserves_every_dependent_hinge_exactly():
    pose, valid, before, requested = _hip_flip_case()
    corrected, report = apply_branch_constraints(
        pose, valid, requested, fields=["hip_forward_depth"],
        bilateral_write_policy=DEPENDENCY_AWARE)
    entry = report["fields"]["hip_forward_depth"]
    assert entry["outcome"] == CORRECTED
    assert entry["write_policy"] == DEPENDENCY_AWARE
    after = sign_state(corrected, valid)

    # The requested branch is installed...
    hip = SIGN_FIELD_NAMES.index("hip_forward_depth")
    assert after[hip] == requested[hip]
    # ...and every other field is bit-identical, not merely close.
    for index, name in enumerate(SIGN_FIELD_NAMES):
        if name != "hip_forward_depth":
            assert after[index] == before[index], name
    # The dependent bend directions are numerically unchanged, not rethresholded.
    for joint in ("left_knee", "right_knee"):
        np.testing.assert_allclose(_bend(corrected, joint), _bend(pose, joint), atol=1e-12)


def test_dependency_aware_preserves_screen_space_midpoint_separation_and_bone_lengths():
    pose, valid, before, requested = _hip_flip_case()
    corrected, _ = apply_branch_constraints(
        pose, valid, requested, fields=["hip_forward_depth"],
        bilateral_write_policy=DEPENDENCY_AWARE)

    # X and Z of every joint, untouched.
    np.testing.assert_array_equal(corrected[:, 0], pose[:, 0])
    np.testing.assert_array_equal(corrected[:, 2], pose[:, 2])

    left, right = JOINT_INDEX["left_hip"], JOINT_INDEX["right_hip"]
    midpoint = lambda frame: (frame[left, FORWARD_DEPTH_AXIS] + frame[right, FORWARD_DEPTH_AXIS]) / 2
    separation = lambda frame: abs(frame[right, FORWARD_DEPTH_AXIS] - frame[left, FORWARD_DEPTH_AXIS])
    assert midpoint(corrected) == pytest.approx(midpoint(pose), abs=1e-12)
    assert separation(corrected) == pytest.approx(separation(pose), abs=1e-12)

    # Internal limb bone lengths inside each translated chain.
    for chain in BILATERAL_DEPENDENT_CHAINS["hip_forward_depth"].values():
        for near, far in zip(chain, chain[1:]):
            length = lambda frame: np.linalg.norm(frame[JOINT_INDEX[far]] - frame[JOINT_INDEX[near]])
            assert length(corrected) == pytest.approx(length(pose), abs=1e-12), (near, far)

    # Nothing outside the two translated chains moved at all.
    translated = {JOINT_INDEX[joint]
                  for chain in BILATERAL_DEPENDENT_CHAINS["hip_forward_depth"].values()
                  for joint in chain}
    for index in range(len(pose)):
        if index not in translated:
            np.testing.assert_array_equal(corrected[index], pose[index])


def test_dependency_aware_translation_is_rigid_and_shares_one_delta_per_side():
    pose, valid, before, requested = _hip_flip_case()
    corrected, report = apply_branch_constraints(
        pose, valid, requested, fields=["hip_forward_depth"],
        bilateral_write_policy=DEPENDENCY_AWARE)
    deltas = report["fields"]["hip_forward_depth"]["anchor_depth_delta_m"]
    # Equal and opposite: that is what preserves the midpoint.
    assert deltas["left_hip"] == pytest.approx(-deltas["right_hip"], abs=1e-12)
    for anchor, chain in BILATERAL_DEPENDENT_CHAINS["hip_forward_depth"].items():
        for joint in chain:
            moved = (corrected[JOINT_INDEX[joint], FORWARD_DEPTH_AXIS]
                     - pose[JOINT_INDEX[joint], FORWARD_DEPTH_AXIS])
            assert moved == pytest.approx(deltas[anchor], abs=1e-12), joint


def test_dependency_aware_no_ops_and_idempotence():
    pose, valid, before, requested = _hip_flip_case()

    # UNKNOWN is an exact no-op.
    blank = np.zeros(len(SIGN_FIELD_NAMES), dtype=np.int64)
    same, report = apply_branch_constraints(pose, valid, blank, fields=["hip_forward_depth"],
                                            bilateral_write_policy=DEPENDENCY_AWARE)
    np.testing.assert_array_equal(same, pose)

    # An already-correct branch is an exact no-op.
    hip = SIGN_FIELD_NAMES.index("hip_forward_depth")
    same, report = apply_branch_constraints(pose, valid, _request("hip_forward_depth", int(before[hip])),
                                            fields=["hip_forward_depth"],
                                            bilateral_write_policy=DEPENDENCY_AWARE)
    assert report["fields"]["hip_forward_depth"]["outcome"] == ALREADY_SATISFIED
    np.testing.assert_array_equal(same, pose)

    # Applying the same request twice changes nothing the second time.
    once, _ = apply_branch_constraints(pose, valid, requested, fields=["hip_forward_depth"],
                                       bilateral_write_policy=DEPENDENCY_AWARE)
    twice, second = apply_branch_constraints(once, valid, requested, fields=["hip_forward_depth"],
                                             bilateral_write_policy=DEPENDENCY_AWARE)
    assert second["fields"]["hip_forward_depth"]["outcome"] == ALREADY_SATISFIED
    np.testing.assert_array_equal(twice, once)

    # Deterministic.
    again, _ = apply_branch_constraints(pose, valid, requested, fields=["hip_forward_depth"],
                                        bilateral_write_policy=DEPENDENCY_AWARE)
    np.testing.assert_array_equal(again, once)


def test_shoulder_dependency_aware_preserves_elbow_bends_and_exposes_attachment_cost():
    pose, valid = _pose()
    before = sign_state(pose, valid)
    shoulder = SIGN_FIELD_NAMES.index("shoulder_forward_depth")
    requested = _request("shoulder_forward_depth", -int(before[shoulder]))
    corrected, report = apply_branch_constraints(
        pose, valid, requested, fields=["shoulder_forward_depth"],
        bilateral_write_policy=DEPENDENCY_AWARE)
    assert report["fields"]["shoulder_forward_depth"]["outcome"] == CORRECTED

    for joint in ("left_elbow", "right_elbow"):
        np.testing.assert_allclose(_bend(corrected, joint), _bend(pose, joint), atol=1e-12)

    # The cost the operator does NOT avoid, and must not claim to: the torso
    # attachment length changes when the shoulder moves relative to the thorax.
    attachment = lambda frame, side: np.linalg.norm(
        frame[JOINT_INDEX[f"{side}_shoulder"]] - frame[JOINT_INDEX["thorax"]])
    assert any(abs(attachment(corrected, side) - attachment(pose, side)) > 1e-9
               for side in ("left", "right"))


def test_replay_declares_validity_regimes_and_per_variant_write_policy():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    try:
        spec = importlib.util.spec_from_file_location(
            "replay_branch_constraints", root / "scripts" / "replay_branch_constraints.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    assert module.VALIDITY_SOURCES == {"V_ORACLE": "target_valid", "V_OBSERVED": "input_valid"}
    # Historical variants keep the anchor-only policy; the DEP pair is the new one.
    assert module.VARIANTS["C_HIP"]["policy"] == ANCHOR_ONLY
    assert module.VARIANTS["C_BILATERAL"]["policy"] == ANCHOR_ONLY
    assert module.VARIANTS["C_HIP_DEP"]["policy"] == DEPENDENCY_AWARE
    assert module.VARIANTS["C_BILATERAL_DEP"]["policy"] == DEPENDENCY_AWARE
    # A DEP variant constrains exactly the same fields as its historical twin,
    # so the only variable between them is the write policy.
    assert module.VARIANTS["C_HIP_DEP"]["fields"] == module.VARIANTS["C_HIP"]["fields"]
    assert module.VARIANTS["C_BILATERAL_DEP"]["fields"] == module.VARIANTS["C_BILATERAL"]["fields"]
    # The frozen hinge variant is unaffected by any bilateral policy.
    assert all(name.endswith("_forward_bend") for name in module.VARIANTS["C_HINGE_ALL"]["fields"])
