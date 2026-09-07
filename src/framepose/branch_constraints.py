"""Branch Constraint — SignState as an output-space constraint, not a feature.

The learned sign-conditioning lineage (docs/28-32) treats a sign field as
*information the network may learn to use*. That abstraction has a measured
weakness: an oracle sign is not guaranteed to be acted on, and a field's
apparent necessity moves when the sign embedding is injected before or after
global joint self-attention (docs/32).

This module tests the other abstraction. The Geometry Core keeps ownership of
continuous XYZ, metric depth magnitude, bone geometry and 2D correspondence; a
sign field owns nothing but the **branch**, and it is enforced on the already
reconstructed pose:

```
predicted continuous pose  +  requested SignState  ->  corrected pose
```

There are no trainable parameters, no RGB, no VLM, no ground-truth magnitude,
no bone-length targets and no neighbouring frames. The only inputs are the
predicted pose, the validity mask the Sign Contract itself reads, and the
requested -1/0/+1 branch.

Contract, per field:

```
requested UNKNOWN            exact no-op
requested already satisfied  exact no-op
requested violated           the declared branch correction below, then the
                             requested branch is verified by re-reading the
                             Sign Contract's own `sign_state`; if the read-back
                             does not confirm it, the field is reverted and
                             reported UNRESOLVED -- never silently failed, and
                             never merely "encouraged"
```

Every threshold used here is an existing Sign Contract constant
(`STABLE_FORWARD_DEPTH_M`, `UNIT_FORWARD_EPSILON`); no new magnitude or
tolerance was introduced, and no correction magnitude is tuned.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from common.canonical_pose import BILATERAL_DEPTH_NORMALIZATION, FORWARD_DEPTH_AXIS, JOINT_INDEX
from framepose.signs import (
    ALLOWED_VALUES, HINGE_CHAINS_BY_JOINT, SIGN_FIELD_COUNT, SIGN_FIELD_NAMES,
    STABLE_FORWARD_DEPTH_M, UNIT_FORWARD_EPSILON, UNKNOWN, sign_state,
)


BRANCH_CONSTRAINT_SCHEMA = "animcv_frame_pose_branch_constraint_v1"

# Per-field outcomes. `requested = already_satisfied + corrected + unresolved`
# is an identity this module guarantees (see `coverage`).
NOT_REQUESTED = "not_requested"
ALREADY_SATISFIED = "already_satisfied"
CORRECTED = "corrected"
UNRESOLVED = "unresolved"
OUTCOMES = (NOT_REQUESTED, ALREADY_SATISFIED, CORRECTED, UNRESOLVED)

_BILATERAL_PAIRS = {
    "shoulder_forward_depth": ("left_shoulder", "right_shoulder"),
    "hip_forward_depth": ("left_hip", "right_hip"),
}
_HINGE_FIELDS = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))

# Bilateral first: a hip exchange moves the proximal joint of both knee chains,
# so orientation must settle before hinge geometry is read. No hinge correction
# moves a shoulder or a hip, and no hinge chain contains another hinge's middle
# joint, so within each group the order is irrelevant.
APPLICATION_ORDER: tuple[str, ...] = ("shoulder_forward_depth", "hip_forward_depth") + _HINGE_FIELDS
SUPPORTED_FIELDS = frozenset(APPLICATION_ORDER)

# `torso_facing` is deliberately out of contract for this module: it is a
# derived cross-product branch over five joints with no single declared
# minimal correction, and the orientation channel it would compete with is
# already covered by the bilateral pair (docs/30-31).
UNSUPPORTED_FIELDS = frozenset(SIGN_FIELD_NAMES) - SUPPORTED_FIELDS


def _validate(pose: np.ndarray, valid: np.ndarray, requested: np.ndarray):
    pose = np.asarray(pose, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    requested = np.asarray(requested)
    if pose.shape != (len(JOINT_INDEX), 3):
        raise ValueError(f"pose must be ({len(JOINT_INDEX)}, 3), got {pose.shape}")
    if valid.shape != (len(JOINT_INDEX),):
        raise ValueError(f"validity must be ({len(JOINT_INDEX)},), got {valid.shape}")
    if requested.shape != (SIGN_FIELD_COUNT,):
        raise ValueError(f"requested sign state must be ({SIGN_FIELD_COUNT},), got {requested.shape}")
    illegal = np.setdiff1d(np.unique(requested), np.asarray(ALLOWED_VALUES))
    if illegal.size:
        raise ValueError(
            "a requested branch must be exactly -1, 0 or +1 -- the Sign Contract domain -- got "
            f"{sorted(int(value) for value in illegal)}")
    return pose, valid, requested.astype(np.int64)


def _bilateral_correction(pose: np.ndarray, valid: np.ndarray, field: str):
    """Exchange the pair's forward-depth offsets around their own midpoint.

    Preserved exactly: every X and Z coordinate, the pair's depth midpoint,
    |D| = |y_right - y_left| / sqrt(2), and every unrelated joint. Changed:
    only which side of the pair owns the near branch.
    """
    left, right = _BILATERAL_PAIRS[field]
    left_index, right_index = JOINT_INDEX[left], JOINT_INDEX[right]
    if not (valid[left_index] and valid[right_index]):
        return None, {"reason": "a pair joint is invalid, so the contract cannot read the branch back",
                      "joints": [left, right]}
    separation = float(
        pose[right_index, FORWARD_DEPTH_AXIS] - pose[left_index, FORWARD_DEPTH_AXIS]
    ) * BILATERAL_DEPTH_NORMALIZATION
    if abs(separation) < STABLE_FORWARD_DEPTH_M:
        # The exchange preserves |D| exactly, so a prediction already under the
        # contract's own stability floor stays under it and can never read back
        # as the requested branch. Inventing a separation here would be the
        # advisor supplying a magnitude, which its contract forbids.
        return None, {"reason": ("the predicted bilateral depth separation is under the Sign "
                                 f"Contract's stability floor ({STABLE_FORWARD_DEPTH_M} m), so an "
                                 "exchange cannot produce a readable branch"),
                      "predicted_separation_m": separation, "joints": [left, right]}
    corrected = pose.copy()
    corrected[left_index, FORWARD_DEPTH_AXIS] = pose[right_index, FORWARD_DEPTH_AXIS]
    corrected[right_index, FORWARD_DEPTH_AXIS] = pose[left_index, FORWARD_DEPTH_AXIS]
    return corrected, {"operation": "forward-depth exchange about the pair midpoint",
                       "predicted_separation_m": separation, "joints": [left, right]}


def _hinge_correction(pose: np.ndarray, valid: np.ndarray, field: str):
    """Reflect the bend's forward-depth component, moving depth only.

    With axis `a = distal - proximal` and perpendicular offset
    `o = (joint - proximal) - a ((joint - proximal).a)/|a|^2`, moving the middle
    joint's depth by `delta` changes the offset's depth component by exactly

        o'_y = o_y + delta * (a_x^2 + a_z^2) / |a|^2

    so the reflection `o'_y = -o_y` is the closed-form

        delta = -2 o_y |a|^2 / (a_x^2 + a_z^2)

    with no tuned magnitude. X and Z of every joint -- including the moved one
    -- are untouched, so the advisor's near/far bit never moves anything in
    image space. The factor `(a_x^2 + a_z^2)/|a|^2` is the squared in-image-plane
    fraction of the limb axis and is the exact degeneracy of a depth-only
    correction: it vanishes when the limb points straight along the camera
    depth axis, where no change of the middle joint's depth can change the
    branch at all.
    """
    joint = field[: -len("_forward_bend")]
    proximal, middle, distal = HINGE_CHAINS_BY_JOINT[joint]
    proximal_index = JOINT_INDEX[proximal]
    middle_index = JOINT_INDEX[middle]
    distal_index = JOINT_INDEX[distal]
    if not (valid[proximal_index] and valid[middle_index] and valid[distal_index]):
        return None, {"reason": "a chain joint is invalid, so the contract cannot read the branch back",
                      "joints": [proximal, middle, distal]}

    axis = pose[distal_index] - pose[proximal_index]
    axis_squared = float(np.dot(axis, axis))
    if axis_squared <= 1e-12:
        # Same guard the Sign Contract's own derivation uses.
        return None, {"reason": "the proximal-distal axis is degenerate",
                      "joints": [proximal, middle, distal]}

    relative = pose[middle_index] - pose[proximal_index]
    offset = relative - axis * (float(np.dot(relative, axis)) / axis_squared)
    in_plane_fraction = float(axis[0] ** 2 + axis[2] ** 2) / axis_squared
    # Reuses the contract's existing UNIT_FORWARD_EPSILON with the same meaning
    # -- a unit-vector component below it is not usable -- applied to the axis's
    # in-image-plane component. No new constant is introduced here.
    if np.sqrt(in_plane_fraction) < UNIT_FORWARD_EPSILON:
        return None, {"reason": ("the proximal-distal axis lies within "
                                 f"{UNIT_FORWARD_EPSILON} of the camera depth axis, so a depth-only "
                                 "branch correction is singular"),
                      "axis_in_plane_fraction": in_plane_fraction,
                      "joints": [proximal, middle, distal]}

    delta = -2.0 * float(offset[FORWARD_DEPTH_AXIS]) / in_plane_fraction
    if not np.isfinite(delta):
        return None, {"reason": "the closed-form depth correction is not finite",
                      "joints": [proximal, middle, distal]}
    corrected = pose.copy()
    corrected[middle_index, FORWARD_DEPTH_AXIS] = pose[middle_index, FORWARD_DEPTH_AXIS] + delta
    return corrected, {"operation": "depth-only reflection of the perpendicular bend offset",
                       "predicted_offset_forward_m": float(offset[FORWARD_DEPTH_AXIS]),
                       "axis_in_plane_fraction": in_plane_fraction,
                       "depth_delta_m": delta,
                       "joints": [proximal, middle, distal]}


def _correction(pose: np.ndarray, valid: np.ndarray, field: str):
    if field in _BILATERAL_PAIRS:
        return _bilateral_correction(pose, valid, field)
    return _hinge_correction(pose, valid, field)


def apply_branch_constraints(pose: np.ndarray, valid: np.ndarray, requested: np.ndarray, *,
                             fields: Iterable[str] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Enforce the requested branches on one predicted canonical pose.

    Returns `(corrected_pose, accounting)`. The pose is never modified in
    place, unrelated joints are never written, and every field's outcome is
    reported.
    """
    pose, valid, requested = _validate(pose, valid, requested)
    selected = tuple(APPLICATION_ORDER) if fields is None else tuple(
        name for name in APPLICATION_ORDER if name in set(fields))
    if fields is not None:
        unsupported = set(fields) - SUPPORTED_FIELDS
        if unsupported:
            raise ValueError(
                f"branch constraints are not defined for {sorted(unsupported)}; supported fields are "
                f"{sorted(SUPPORTED_FIELDS)}")
    for name in UNSUPPORTED_FIELDS:
        if requested[SIGN_FIELD_NAMES.index(name)] != UNKNOWN:
            raise ValueError(
                f"{name!r} has no declared branch correction in this module and must be UNKNOWN in "
                "the requested state; it is refused rather than silently ignored")

    corrected = pose.copy()
    accounting: dict[str, Any] = {}
    for field in selected:
        index = SIGN_FIELD_NAMES.index(field)
        want = int(requested[index])
        if want == UNKNOWN:
            accounting[field] = {"outcome": NOT_REQUESTED, "requested": want}
            continue
        before = int(sign_state(corrected, valid)[index])
        if before == want:
            accounting[field] = {"outcome": ALREADY_SATISFIED, "requested": want, "read_back_before": before}
            continue
        candidate, detail = _correction(corrected, valid, field)
        if candidate is None:
            accounting[field] = {"outcome": UNRESOLVED, "requested": want,
                                 "read_back_before": before, **detail}
            continue
        after = int(sign_state(candidate, valid)[index])
        if after != want:
            accounting[field] = {
                "outcome": UNRESOLVED, "requested": want, "read_back_before": before,
                "read_back_after_attempt": after,
                "reason": ("the correction did not read back as the requested branch under the Sign "
                           "Contract's own derivation; the field was reverted rather than reported "
                           "as satisfied"),
                **detail}
            continue
        moved = np.linalg.norm(candidate - corrected, axis=-1) * 1000.0
        accounting[field] = {
            "outcome": CORRECTED, "requested": want, "read_back_before": before,
            "read_back_after": after,
            "moved_joint_count": int((moved > 0).sum()),
            "max_joint_displacement_mm": float(moved.max()),
            "total_displacement_mm": float(moved.sum()),
            **detail}
        corrected = candidate

    final_state = sign_state(corrected, valid)
    displacement = np.linalg.norm(corrected - pose, axis=-1) * 1000.0
    return corrected, {
        "schema": BRANCH_CONSTRAINT_SCHEMA,
        "fields": accounting,
        "final_sign_state": {name: int(final_state[position])
                             for position, name in enumerate(SIGN_FIELD_NAMES)},
        "requested_branch_satisfied": {
            name: (bool(final_state[position] == requested[position])
                   if requested[position] != UNKNOWN else None)
            for position, name in enumerate(SIGN_FIELD_NAMES)},
        "moved_joint_count": int((displacement > 0).sum()),
        "max_joint_displacement_mm": float(displacement.max()),
        "total_displacement_mm": float(displacement.sum()),
    }


def apply_branch_constraints_batch(poses: np.ndarray, valid: np.ndarray, requested: np.ndarray, *,
                                   fields: Iterable[str] | None = None):
    """`apply_branch_constraints` over `(N, 17, 3)` predictions."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3:
        raise ValueError(f"poses must be (N, {len(JOINT_INDEX)}, 3), got {poses.shape}")
    corrected = np.empty_like(poses)
    reports: list[dict[str, Any]] = []
    for index in range(len(poses)):
        frame, report = apply_branch_constraints(poses[index], valid[index], requested[index],
                                                 fields=fields)
        corrected[index] = frame
        reports.append(report)
    return corrected, reports


def coverage(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-field applicability accounting with its own identity check.

    `requested = already_satisfied + corrected + unresolved` is asserted, so a
    frame the constraint could not be applied to can never be quietly dropped.
    """
    per_field: dict[str, Any] = {}
    for field in APPLICATION_ORDER:
        counts = {outcome: 0 for outcome in OUTCOMES}
        reasons: dict[str, int] = {}
        present = 0
        # Counted from the requested sign itself, NOT from the outcome buckets,
        # so the identity below is a real cross-check rather than a tautology.
        requested = 0
        for report in reports:
            entry = report["fields"].get(field)
            if entry is None:
                continue
            present += 1
            if int(entry["requested"]) != UNKNOWN:
                requested += 1
            counts[entry["outcome"]] += 1
            if entry["outcome"] == UNRESOLVED:
                reasons[entry.get("reason", "unspecified")] = reasons.get(entry.get("reason", "unspecified"), 0) + 1
        resolved_buckets = counts[ALREADY_SATISFIED] + counts[CORRECTED] + counts[UNRESOLVED]
        identity = (requested == resolved_buckets
                    and present == requested + counts[NOT_REQUESTED])
        per_field[field] = {
            "frames_considered": present,
            "requested": requested,
            "already_satisfied": counts[ALREADY_SATISFIED],
            "corrected": counts[CORRECTED],
            "unresolved": counts[UNRESOLVED],
            "unknown_or_not_requested": counts[NOT_REQUESTED],
            "outcome_bucket_total": resolved_buckets,
            "coverage_identity_holds": bool(identity),
            "unresolved_reasons": reasons,
        }
    if not all(entry["coverage_identity_holds"] for entry in per_field.values()):
        broken = [name for name, entry in per_field.items() if not entry["coverage_identity_holds"]]
        raise ValueError(
            f"coverage identity requested == already_satisfied + corrected + unresolved failed for "
            f"{broken}; a requested branch was neither satisfied, corrected nor reported unresolved")
    return {"schema": BRANCH_CONSTRAINT_SCHEMA, "frames": len(reports), "fields": per_field}
