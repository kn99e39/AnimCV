"""The Sign Contract — discrete orientation evidence, and nothing else.

AnimCV's residual frame-pose failure has two conceptually different parts:

```
POSITION MISMATCH   continuous 3D geometry error
FLIP MISMATCH       roughly correct geometry placed in the wrong branch
```

The Frame Pose Core owns the first. This module formalises the second as a
small, discrete, machine-readable contract, so a separate advisor can supply the
branch without ever touching continuous reconstruction.

Every field is derived from a quantity AnimCV **already** uses to define an
orientation or hinge failure — `common.canonical_pose`'s bilateral
forward-depth, root-yaw and bend-direction mathematics, and the evaluator's
existing `STABLE_FORWARD_DEPTH_M` stability floor. Nothing here is a new
heuristic invented because it is easy to prompt.

All fields are read in AnimCV's canonical camera frame (+X right, +Y
forward/depth, +Z up), so each one is a near/far or facing question about the
image — the kind of evidence a single RGB frame actually carries, and exactly
what a 2D-joint-only observation discards.

A field takes three values:

```
+1  positive branch
-1  negative branch
 0  degenerate or unobservable -- the quantity is at its noise floor, or a
    required joint is invalid. Never guessed.
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from common.canonical_pose import (
    BILATERAL_DEPTH_NORMALIZATION, FORWARD_DEPTH_AXIS, JOINT_INDEX, JOINT_NAMES, bend_direction,
)


SIGN_CONTRACT_SCHEMA = "animcv_frame_pose_sign_contract_v1"
SIGN_STATE_SCHEMA = "animcv_frame_pose_sign_state_v1"

# Reused verbatim from the evaluator: below this the ground-truth bilateral
# forward depth is at its noise floor and its sign carries no information. It is
# the same floor `*_forward_depth_sign_disagreement_stable` already uses, so a
# sign field and its historical metric degenerate together.
STABLE_FORWARD_DEPTH_M = 0.01

# A unit direction whose forward component is smaller than this is too close to
# the image plane for its near/far branch to be observable. 0.1 on a unit vector
# is ~5.7 degrees away from lying exactly in the frontal plane.
UNIT_FORWARD_EPSILON = 0.1

# A limb bent by less than this off its proximal-distal axis has no meaningful
# bend direction to take the sign of.
MIN_BEND_OFFSET_M = 0.02

UNKNOWN = 0
POSITIVE = 1
NEGATIVE = -1


@dataclass(frozen=True)
class SignField:
    """One discrete sign variable, tied to the geometry that defines it."""

    name: str
    quantity: str
    definition: str
    positive_means: str
    negative_means: str
    degenerate_when: str
    joints: tuple[str, ...]
    historical_metric: str

    @property
    def joint_indices(self) -> tuple[int, ...]:
        return tuple(JOINT_INDEX[name] for name in self.joints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "quantity": self.quantity, "definition": self.definition,
            "positive_means": self.positive_means, "negative_means": self.negative_means,
            "degenerate_when": self.degenerate_when, "joints": list(self.joints),
            "historical_metric": self.historical_metric,
        }


_ARM_CHAINS = {"left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
               "right_elbow": ("right_shoulder", "right_elbow", "right_wrist")}
_LEG_CHAINS = {"left_knee": ("left_hip", "left_knee", "left_ankle"),
               "right_knee": ("right_hip", "right_knee", "right_ankle")}
HINGE_CHAINS_BY_JOINT = {**_ARM_CHAINS, **_LEG_CHAINS}


def _hinge_field(joint: str) -> SignField:
    proximal, _, distal = HINGE_CHAINS_BY_JOINT[joint]
    return SignField(
        name=f"{joint}_forward_bend",
        quantity=f"sign of the +Y component of bend_direction({joint}; {proximal}, {distal})",
        definition=(f"common.canonical_pose.bend_direction gives the unit perpendicular offset of "
                    f"{joint} from the {proximal}->{distal} axis; this field is the sign of that "
                    f"unit vector's forward/depth component (axis {FORWARD_DEPTH_AXIS})"),
        positive_means=f"{joint} is farther from the camera than the {proximal}-{distal} line",
        negative_means=f"{joint} is nearer to the camera than the {proximal}-{distal} line",
        degenerate_when=(f"the perpendicular offset is under {MIN_BEND_OFFSET_M} m (limb nearly "
                         f"straight), the unit forward component is under {UNIT_FORWARD_EPSILON}, "
                         f"or any of {proximal}/{joint}/{distal} is invalid"),
        joints=(joint,),
        historical_metric=f"hinge_errors flipped / hinge_direction_mae_degrees for the {joint} chain",
    )


SIGN_FIELDS: tuple[SignField, ...] = (
    SignField(
        name="torso_facing",
        quantity="sign of the +Y component of the body-forward vector",
        definition=("body up u = thorax - pelvis, body right r = right_shoulder - left_shoulder, "
                    "body forward f = normalize(cross(u, r)); this field is sign(f_y). It is the "
                    "same construction strata._facing_angle_degrees uses, reduced to its branch"),
        positive_means="the subject faces away from the camera",
        negative_means="the subject faces toward the camera",
        degenerate_when=(f"|f_y| < {UNIT_FORWARD_EPSILON} (near-exact profile, where the facing "
                         "branch is not observable along the depth axis), the cross product is "
                         "degenerate, or pelvis/thorax/shoulders are invalid"),
        joints=("pelvis", "spine", "thorax", "neck", "head",
                "left_shoulder", "right_shoulder", "left_hip", "right_hip"),
        historical_metric="root_yaw_error_degrees 180-degree branch; the docs/12-13 yaw tail",
    ),
    SignField(
        name="shoulder_forward_depth",
        quantity="sign of D_shoulder = (y_right_shoulder - y_left_shoulder) / sqrt(2)",
        definition=("the docs/18 / docs/21 bilateral forward-depth quantity on the shoulder pair, "
                    "read on the canonical forward axis, reduced to its sign"),
        positive_means="the right shoulder is farther from the camera than the left",
        negative_means="the right shoulder is nearer to the camera than the left",
        degenerate_when=f"|D_shoulder| < {STABLE_FORWARD_DEPTH_M} m, or either shoulder is invalid",
        joints=("left_shoulder", "right_shoulder"),
        historical_metric="shoulder_forward_depth_sign_disagreement(_stable)",
    ),
    SignField(
        name="hip_forward_depth",
        quantity="sign of D_hip = (y_right_hip - y_left_hip) / sqrt(2)",
        definition="the same bilateral forward-depth quantity on the hip pair, reduced to its sign",
        positive_means="the right hip is farther from the camera than the left",
        negative_means="the right hip is nearer to the camera than the left",
        degenerate_when=f"|D_hip| < {STABLE_FORWARD_DEPTH_M} m, or either hip is invalid",
        joints=("left_hip", "right_hip"),
        historical_metric="hip_forward_depth_sign_disagreement(_stable)",
    ),
    _hinge_field("left_elbow"),
    _hinge_field("right_elbow"),
    _hinge_field("left_knee"),
    _hinge_field("right_knee"),
)

SIGN_FIELD_NAMES: tuple[str, ...] = tuple(field.name for field in SIGN_FIELDS)
SIGN_FIELD_COUNT = len(SIGN_FIELDS)


def contract() -> dict[str, Any]:
    """The machine-readable Sign Contract."""
    return {
        "schema": SIGN_CONTRACT_SCHEMA,
        "coordinate_frame": "camera_root_relative (+X right, +Y forward/depth, +Z up)",
        "values": {"positive": POSITIVE, "negative": NEGATIVE, "degenerate_or_unknown": UNKNOWN},
        "thresholds": {
            "stable_forward_depth_m": STABLE_FORWARD_DEPTH_M,
            "unit_forward_epsilon": UNIT_FORWARD_EPSILON,
            "min_bend_offset_m": MIN_BEND_OFFSET_M,
        },
        "fields": [field.to_dict() for field in SIGN_FIELDS],
        "excluded_by_contract": [
            "XYZ coordinates", "depth magnitude", "metric offsets", "bone lengths",
            "continuous pose embeddings", "image patch tokens",
        ],
    }


# ------------------------------------------------------------ derivation ----

def _sign(value: float, threshold: float) -> int:
    if not np.isfinite(value) or abs(value) < threshold:
        return UNKNOWN
    return POSITIVE if value > 0 else NEGATIVE


def _torso_facing(pose: np.ndarray, valid: np.ndarray) -> int:
    required = ("pelvis", "thorax", "left_shoulder", "right_shoulder")
    if not all(valid[JOINT_INDEX[name]] for name in required):
        return UNKNOWN
    up = pose[JOINT_INDEX["thorax"]] - pose[JOINT_INDEX["pelvis"]]
    right = pose[JOINT_INDEX["right_shoulder"]] - pose[JOINT_INDEX["left_shoulder"]]
    forward = np.cross(up, right)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        return UNKNOWN
    return _sign(float(forward[FORWARD_DEPTH_AXIS] / norm), UNIT_FORWARD_EPSILON)


def _bilateral_forward_depth(pose: np.ndarray, valid: np.ndarray, left: str, right: str) -> int:
    left_index, right_index = JOINT_INDEX[left], JOINT_INDEX[right]
    if not (valid[left_index] and valid[right_index]):
        return UNKNOWN
    value = (pose[right_index, FORWARD_DEPTH_AXIS] - pose[left_index, FORWARD_DEPTH_AXIS])
    return _sign(float(value) * BILATERAL_DEPTH_NORMALIZATION, STABLE_FORWARD_DEPTH_M)


def _hinge_forward_bend(pose: np.ndarray, valid: np.ndarray, joint: str) -> int:
    proximal, middle, distal = HINGE_CHAINS_BY_JOINT[joint]
    indices = [JOINT_INDEX[name] for name in (proximal, middle, distal)]
    if not all(valid[index] for index in indices):
        return UNKNOWN
    axis = pose[indices[2]] - pose[indices[0]]
    axis_squared = float(np.dot(axis, axis))
    if axis_squared <= 1e-12:
        return UNKNOWN
    offset = pose[indices[1]] - (pose[indices[0]] + axis * (np.dot(pose[indices[1]] - pose[indices[0]], axis) / axis_squared))
    if float(np.linalg.norm(offset)) < MIN_BEND_OFFSET_M:
        return UNKNOWN
    direction = bend_direction(pose[indices[1]], pose[indices[0]], pose[indices[2]])
    if direction is None:
        return UNKNOWN
    return _sign(float(direction[FORWARD_DEPTH_AXIS]), UNIT_FORWARD_EPSILON)


def sign_state(pose: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Derive the `(7,)` int8 sign state of one canonical pose.

    Used both to build the oracle from ground truth and to read the sign state
    back out of a prediction for scoring. It never touches continuous values
    downstream -- it only reduces them to a branch.
    """
    pose = np.asarray(pose, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if pose.shape != (len(JOINT_NAMES), 3) or valid.shape != (len(JOINT_NAMES),):
        raise ValueError("sign_state expects a canonical (17, 3) pose and (17,) validity")
    values = {
        "torso_facing": _torso_facing(pose, valid),
        "shoulder_forward_depth": _bilateral_forward_depth(pose, valid, "left_shoulder", "right_shoulder"),
        "hip_forward_depth": _bilateral_forward_depth(pose, valid, "left_hip", "right_hip"),
        "left_elbow_forward_bend": _hinge_forward_bend(pose, valid, "left_elbow"),
        "right_elbow_forward_bend": _hinge_forward_bend(pose, valid, "right_elbow"),
        "left_knee_forward_bend": _hinge_forward_bend(pose, valid, "left_knee"),
        "right_knee_forward_bend": _hinge_forward_bend(pose, valid, "right_knee"),
    }
    return np.asarray([values[name] for name in SIGN_FIELD_NAMES], dtype=np.int8)


def oracle_sign_states(target_3d: np.ndarray, target_valid: np.ndarray) -> np.ndarray:
    """`(N, 7)` oracle signs derived from ground-truth 3D only.

    This is an architecture control, not a production mechanism: it answers
    "is discrete sign evidence sufficient", with no RGB involved.
    """
    target_3d = np.asarray(target_3d)
    target_valid = np.asarray(target_valid, dtype=bool)
    return np.stack([sign_state(target_3d[index], target_valid[index])
                     for index in range(len(target_3d))]).astype(np.int8)


def neutral_sign_states(count: int) -> np.ndarray:
    """The fixed neutral control: every field `UNKNOWN`, for every frame.

    Defined once, before any result was seen, and never tuned on an outcome.
    """
    return np.zeros((count, SIGN_FIELD_COUNT), dtype=np.int8)


def to_dict(state: Sequence[int]) -> dict[str, Any]:
    if len(state) != SIGN_FIELD_COUNT:
        raise ValueError(f"a sign state has {SIGN_FIELD_COUNT} fields")
    return {"schema": SIGN_STATE_SCHEMA,
            **{name: int(value) for name, value in zip(SIGN_FIELD_NAMES, state)}}


def from_dict(payload: dict[str, Any]) -> np.ndarray:
    missing = [name for name in SIGN_FIELD_NAMES if name not in payload]
    if missing:
        raise ValueError(f"sign state is missing fields: {missing}")
    values = []
    for name in SIGN_FIELD_NAMES:
        value = payload[name]
        if value not in (POSITIVE, NEGATIVE, UNKNOWN):
            raise ValueError(f"sign field {name!r} must be -1, 0 or +1, got {value!r}")
        values.append(int(value))
    return np.asarray(values, dtype=np.int8)


def agreement(predicted: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Per-field agreement, scored only where the reference is not degenerate."""
    predicted = np.asarray(predicted, dtype=np.int8)
    reference = np.asarray(reference, dtype=np.int8)
    if predicted.shape != reference.shape:
        raise ValueError("sign arrays must align")
    report: dict[str, Any] = {}
    for index, name in enumerate(SIGN_FIELD_NAMES):
        scored = reference[:, index] != UNKNOWN
        total = int(scored.sum())
        correct = int((predicted[scored, index] == reference[scored, index]).sum())
        opposite = int((predicted[scored, index] == -reference[scored, index]).sum())
        report[name] = {
            "scored_frames": total,
            "agreement": (correct / total) if total else None,
            "disagreement_count": total - correct,
            "opposite_count": opposite,
            "predicted_degenerate_count": int((predicted[scored, index] == UNKNOWN).sum()),
            "reference_degenerate_count": int((reference[:, index] == UNKNOWN).sum()),
        }
    scored = reference != UNKNOWN
    total = int(scored.sum())
    report["overall"] = {
        "scored_fields": total,
        "agreement": (float((predicted[scored] == reference[scored]).sum()) / total) if total else None,
    }
    return report


def joint_field_matrix() -> np.ndarray:
    """`(17, 7)` mask: which joints each sign field is allowed to condition."""
    matrix = np.zeros((len(JOINT_NAMES), SIGN_FIELD_COUNT), dtype=np.float32)
    for index, field in enumerate(SIGN_FIELDS):
        for joint in field.joint_indices:
            matrix[joint, index] = 1.0
    return matrix


def summarize(states: Iterable[np.ndarray]) -> dict[str, dict[str, int]]:
    counts = {name: {"positive": 0, "negative": 0, "degenerate": 0} for name in SIGN_FIELD_NAMES}
    for state in states:
        for name, value in zip(SIGN_FIELD_NAMES, state):
            key = "positive" if value > 0 else "negative" if value < 0 else "degenerate"
            counts[name][key] += 1
    return counts
