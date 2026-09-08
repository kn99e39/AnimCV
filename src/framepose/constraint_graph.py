"""Which joints each sign field READS, which a constraint WRITES, and what that breaks.

docs/34. `SignField.joints` is a *routing/ownership* property -- it says which
joint queries a field is allowed to condition, and for a hinge field it names
only the middle joint even though deriving that field's sign reads the
proximal, middle and distal joints. Constraint reasoning needs the other
relation and must not overload that one, so the read/write/dependent graph
lives here.

The read sets below are transcribed from `framepose.signs`' own derivation
functions, not from `SignField.joints`, and `test_constraint_dependency_graph`
pins them against those functions by perturbing one joint at a time.

Nothing here changes sign mathematics or thresholds.
"""

from __future__ import annotations

from typing import Any

from common.canonical_pose import JOINT_INDEX
from framepose.signs import HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES

CONSTRAINT_GRAPH_SCHEMA = "animcv_frame_pose_constraint_dependency_v1"

#: Write policies a bilateral constraint may use. `anchor_only` is docs/33's
#: historical pair swap; `dependency_aware` additionally translates each
#: anchor's dependent limb chain by that anchor's own exact delta.
ANCHOR_ONLY = "anchor_only"
DEPENDENCY_AWARE = "dependency_aware"
WRITE_POLICIES = (ANCHOR_ONLY, DEPENDENCY_AWARE)

#: The limb chain rigidly carried by each bilateral anchor under
#: `dependency_aware`. Each tuple is exactly the anchor plus the hinge chain it
#: is the proximal joint of, so the whole chain shares one delta.
BILATERAL_DEPENDENT_CHAINS: dict[str, dict[str, tuple[str, ...]]] = {
    "shoulder_forward_depth": {
        "left_shoulder": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_shoulder": ("right_shoulder", "right_elbow", "right_wrist"),
    },
    "hip_forward_depth": {
        "left_hip": ("left_hip", "left_knee", "left_ankle"),
        "right_hip": ("right_hip", "right_knee", "right_ankle"),
    },
}

_BILATERAL_ANCHORS: dict[str, tuple[str, str]] = {
    "shoulder_forward_depth": ("left_shoulder", "right_shoulder"),
    "hip_forward_depth": ("left_hip", "right_hip"),
}


def _read_joints() -> dict[str, tuple[str, ...]]:
    """Joints whose XYZ actually determine each field's sign.

    Transcribed from `signs._torso_facing`, `signs._bilateral_forward_depth`
    and `signs._hinge_forward_bend`. Note that `torso_facing` reads only four
    joints even though `SignField.joints` routes nine.
    """
    reads: dict[str, tuple[str, ...]] = {
        "torso_facing": ("pelvis", "thorax", "left_shoulder", "right_shoulder"),
        "shoulder_forward_depth": ("left_shoulder", "right_shoulder"),
        "hip_forward_depth": ("left_hip", "right_hip"),
    }
    for joint, chain in HINGE_CHAINS_BY_JOINT.items():
        reads[f"{joint}_forward_bend"] = tuple(chain)
    return reads


READ_JOINTS: dict[str, tuple[str, ...]] = _read_joints()


def write_groups(field: str, policy: str = ANCHOR_ONLY) -> tuple[tuple[str, ...], ...]:
    """Joints a constraint on `field` writes, grouped by common displacement.

    Each group moves by one shared delta, which is what makes a dependent
    field's preservation provable: a field whose entire read set sits inside a
    single group sees only a rigid translation, and a rigid translation leaves
    every within-set difference -- and therefore every sign the contract
    derives from those differences -- exactly unchanged.
    """
    if policy not in WRITE_POLICIES:
        raise ValueError(f"unknown write policy {policy!r}; known: {list(WRITE_POLICIES)}")
    if field in _BILATERAL_ANCHORS:
        if policy == ANCHOR_ONLY:
            return tuple((anchor,) for anchor in _BILATERAL_ANCHORS[field])
        return tuple(BILATERAL_DEPENDENT_CHAINS[field][anchor]
                     for anchor in _BILATERAL_ANCHORS[field])
    if field.endswith("_forward_bend"):
        # The hinge operator is frozen: it writes the middle joint only.
        return ((HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]][1],),)
    raise ValueError(f"no constraint write policy is declared for {field!r}")


def field_entry(field: str, policy: str = ANCHOR_ONLY) -> dict[str, Any]:
    groups = write_groups(field, policy)
    written = {joint for group in groups for joint in group}
    downstream, exact, at_risk = [], [], []
    for other in SIGN_FIELD_NAMES:
        if other == field:
            continue
        read = set(READ_JOINTS[other])
        if not (read & written):
            continue
        downstream.append(other)
        # Preserved exactly iff the whole read set rides one displacement group.
        if any(read <= set(group) for group in groups):
            exact.append(other)
        else:
            at_risk.append(other)
    return {
        "read_joints": list(READ_JOINTS[field]),
        "write_joints": sorted(written, key=lambda name: JOINT_INDEX[name]),
        "write_groups": [list(group) for group in groups],
        "downstream_dependent_fields": downstream,
        "dependents_preserved_exactly": exact,
        "dependents_at_risk": at_risk,
    }


def dependency_graph(policy: str = ANCHOR_ONLY) -> dict[str, Any]:
    """Machine-readable read/write/dependent relation for every field with a
    declared constraint. `torso_facing` has no constraint and appears only as a
    dependent."""
    from framepose.branch_constraints import SUPPORTED_FIELDS

    return {
        "schema": CONSTRAINT_GRAPH_SCHEMA,
        "write_policy": policy,
        "note": ("read_joints come from the sign derivation functions, NOT from "
                 "SignField.joints, which is a routing/ownership property"),
        "fields": {field: field_entry(field, policy)
                   for field in SIGN_FIELD_NAMES if field in SUPPORTED_FIELDS},
    }
