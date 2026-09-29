"""AnimationSemantics — the pre-retargeting Animation Semantics sidecar (docs/53).

This is the one representation of *what motion semantics AnimCV can actually
provide downstream*, and of how quantities it cannot provide are recorded
without inventing them. It is a sidecar to, not an extension of, MotionGraph:

- MotionGraph (motion_graph.py, Architecture_v2 section 5) keeps owning what it
  already owns — per-landmark 2D/3D observation tracks built from a
  PoseSequence — and its serialization is unchanged. It has no schema/version
  field and is read directly by the retarget solvers, so adding fields to it
  would silently change a shared contract that historical `.motion.json`
  artifacts cannot satisfy.
- AnimationSemantics owns the FramePose-era quantities listed in
  pose/motion_ownership.py, one owner per quantity, never folded into a single
  confidence scalar. It links to a MotionGraph only by (frame_index,
  timestamp) — see `motion_graph_reference` / `check_motion_graph_alignment`.

Per frame it carries:

    articulation      Frame Pose local articulation (camera_root_relative, metres)
    root_orientation  body heading yaw, KNOWN/UNKNOWN, + historical-policy state
    foot_motion       per side ContactState CONTACT/MOVING/UNKNOWN — this means
                      contact-LIKE stance/swing evidence, not proven floor contact
    reliability       structural observation validity / in-frame state
    root_translation  always UNAVAILABLE in this schema version
    ground_height     always UNAVAILABLE in this schema version

UNAVAILABLE is not zero and not UNKNOWN. UNKNOWN means "this quantity has an
owner and an estimator, but this frame lacked the evidence"; UNAVAILABLE means
"no current evidence source can produce this quantity at all" (docs/51
not_observable). Loading refuses any numeric root translation / ground height
— including a zero vector — because nothing in this schema version may
produce one.

Blender-, FBX- and target-rig-independent: nothing here references a rig,
bone, or output format.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.serialization import read_json, write_json
from motion.motion_graph import MotionGraph
from pose.contact import ContactState
from pose.motion_ownership import OWNERSHIP_MODEL

SEMANTICS_SCHEMA = "animcv_animation_semantics_v1"
CONTACT_CALIBRATION_SCHEMA = "animcv_contact_calibration_v1"

# The exact docs/51 rule: pose.contact.classify_foot_contact with thresholds
# from pose.contact.fit_contact_thresholds at its default percentiles. A change
# to either function's semantics must bump this string.
CONTACT_RULE_VERSION = "docs51_contact_candidate_v1"
CONTACT_RULE_PERCENTILES = {"contact": 15.0, "moving": 60.0, "height": 40.0}
CONTACT_MIN_RELIABLE_RUN = 2
CONTACT_INTERPRETATION = (
    "contact_like_stance_swing_evidence: root-relative ankle speed/height "
    "consistency relative to the pelvis. NOT proven physical foot-ground "
    "contact; validated only against a KINEMATIC_PROXY reference, never an "
    "exact contact label (docs/51, docs/52)."
)

ROOT_ORIENTATION_POLICY = "historical_root_motion_estimate_segmented_v1"
ROOT_ORIENTATION_NOTES = (
    "Yaw from pose.root_motion.estimate_root_motion, unchanged, at its historical "
    "defaults (median window 5, max_yaw_step_degrees 20). The hold policy is "
    "historical. docs/52 showed it is materially active on frozen H0 (mean hold "
    "rate 27.3% of frames, ~98% of holds against small oracle rotation), but the "
    "20-degree threshold has NOT been shown optimal and was not tuned. The hold "
    "has no release: a held frame carries the last accepted yaw, and holding "
    "continues until the smoothed estimate returns within 20 degrees of that "
    "value, so a held run can be long and its yaw stale (docs/53). Treat "
    "yaw_held frames as degraded heading, not as observed heading. The "
    "estimator is applied independently to each maximal run of consecutive "
    "frames on which it can compute a yaw; frames outside such runs are UNKNOWN. "
    "Yaw is heading relative to the camera frame, not a world heading: no camera "
    "motion is estimated."
)

# Semantic field -> docs/51 ownership entry. One owner per field.
SEMANTIC_FIELDS: dict[str, str] = {
    "articulation": "local_articulation",
    "root_orientation": "body_heading_yaw",
    "foot_motion": "foot_planted_or_moving",
    "reliability": "observation_reliability",
    "root_translation": "global_horizontal_translation",
    "ground_height": "ground_height_placement",
}

FOOT_SIDES = ("left", "right")


def ownership_table() -> dict[str, dict[str, str]]:
    by_name = {entry.name: entry for entry in OWNERSHIP_MODEL}
    return {
        field_name: {"quantity": quantity, "owner": by_name[quantity].owner.value,
                     "observability": by_name[quantity].observability.value}
        for field_name, quantity in SEMANTIC_FIELDS.items()
    }


# --------------------------------------------------------------- unavailable --

UNAVAILABLE_STATUS = "unavailable"

UNAVAILABLE_REASONS: dict[str, str] = {
    "root_translation": (
        "not_observable: root-relative Frame Pose pins the pelvis to the origin and "
        "no camera-motion estimate exists; planted-foot recovery is conditional "
        "and not implemented (docs/51, docs/52). Not zero — unknown."
    ),
    "ground_height": (
        "not_observable: no wired dataset or estimator provides a floor reference "
        "in Frame Pose coordinates; ankle height is not a ground estimate "
        "(docs/51). Not zero — unknown."
    ),
}


@dataclass(frozen=True)
class Unavailable:
    """A quantity no current evidence source can produce. Carries no value."""

    quantity: str

    def __post_init__(self) -> None:
        if self.quantity not in UNAVAILABLE_REASONS:
            raise ValueError(f"{self.quantity!r} is not a declared unavailable quantity")

    def to_dict(self) -> dict[str, Any]:
        return {"status": UNAVAILABLE_STATUS}

    @classmethod
    def from_dict(cls, quantity: str, data: Any) -> "Unavailable":
        # Refuse anything that looks like a value, zero included: this schema
        # version has no producer for these quantities.
        if not isinstance(data, dict) or data != {"status": UNAVAILABLE_STATUS}:
            raise ValueError(
                f"{quantity} must serialize as {{'status': 'unavailable'}} in "
                f"{SEMANTICS_SCHEMA}; got {data!r}"
            )
        return cls(quantity)


# ---------------------------------------------------------- per-frame fields --

@dataclass(frozen=True)
class LocalArticulation:
    """Frame Pose output: joint positions aligned with AnimationSemantics.joint_names."""

    joint_positions: tuple[tuple[float, float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"joint_positions": [list(p) for p in self.joint_positions]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalArticulation":
        return cls(tuple(tuple(float(v) for v in p) for p in data["joint_positions"]))


@dataclass(frozen=True)
class RootOrientation:
    """Body heading. UNKNOWN is `known=False` with every value None — never yaw 0."""

    known: bool
    yaw_radians: float | None = None
    # estimate_root_motion's own confidence (detector-weighted, x0.25 when held).
    # Owned by Root Orientation; not observation reliability.
    historical_confidence: float | None = None
    yaw_held: bool | None = None

    def __post_init__(self) -> None:
        values = (self.yaw_radians, self.historical_confidence, self.yaw_held)
        if self.known:
            if any(v is None for v in values) or not math.isfinite(self.yaw_radians):
                raise ValueError("a KNOWN root orientation needs finite yaw, confidence and yaw_held")
        elif any(v is not None for v in values):
            raise ValueError("an UNKNOWN root orientation must carry no yaw/confidence/held value")

    def to_dict(self) -> dict[str, Any]:
        return {"status": "known" if self.known else "unknown", "yaw_radians": self.yaw_radians,
                "historical_confidence": self.historical_confidence, "yaw_held": self.yaw_held}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RootOrientation":
        status = data["status"]
        if status not in ("known", "unknown"):
            raise ValueError(f"unknown root orientation status {status!r}")
        return cls(known=status == "known", yaw_radians=data["yaw_radians"],
                   historical_confidence=data["historical_confidence"], yaw_held=data["yaw_held"])


UNKNOWN_ORIENTATION = RootOrientation(known=False)


@dataclass(frozen=True)
class FootMotion:
    """Per-side docs/51 ContactState. Contact-like evidence, not floor contact."""

    left: ContactState
    right: ContactState

    def to_dict(self) -> dict[str, str]:
        return {"left": self.left.value, "right": self.right.value}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "FootMotion":
        return cls(ContactState(data["left"]), ContactState(data["right"]))


@dataclass(frozen=True)
class ObservationReliability:
    """Structural observation state only — no learned/VLM reliability.

    joint_observation_valid: the FramePose input validity that gated every
    semantic above (input_valid AND finite Frame Pose output).
    joint_in_frame: exact structural in-frame mask of the 2D input, or None
    when the source has no normalized 2D observation.
    """

    joint_observation_valid: tuple[bool, ...]
    joint_in_frame: tuple[bool, ...] | None

    @property
    def all_joints_valid(self) -> bool:
        return all(self.joint_observation_valid)

    def to_dict(self) -> dict[str, Any]:
        return {"joint_observation_valid": list(self.joint_observation_valid),
                "joint_in_frame": list(self.joint_in_frame) if self.joint_in_frame is not None else None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationReliability":
        in_frame = data["joint_in_frame"]
        return cls(tuple(bool(v) for v in data["joint_observation_valid"]),
                   tuple(bool(v) for v in in_frame) if in_frame is not None else None)


@dataclass(frozen=True)
class SemanticFrame:
    frame_index: int
    timestamp: float
    articulation: LocalArticulation
    root_orientation: RootOrientation
    foot_motion: FootMotion
    reliability: ObservationReliability
    root_translation: Unavailable = field(default_factory=lambda: Unavailable("root_translation"))
    ground_height: Unavailable = field(default_factory=lambda: Unavailable("ground_height"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "articulation": self.articulation.to_dict(),
            "root_orientation": self.root_orientation.to_dict(),
            "foot_motion": self.foot_motion.to_dict(),
            "reliability": self.reliability.to_dict(),
            "root_translation": self.root_translation.to_dict(),
            "ground_height": self.ground_height.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticFrame":
        return cls(
            frame_index=int(data["frame_index"]),
            timestamp=float(data["timestamp"]),
            articulation=LocalArticulation.from_dict(data["articulation"]),
            root_orientation=RootOrientation.from_dict(data["root_orientation"]),
            foot_motion=FootMotion.from_dict(data["foot_motion"]),
            reliability=ObservationReliability.from_dict(data["reliability"]),
            root_translation=Unavailable.from_dict("root_translation", data["root_translation"]),
            ground_height=Unavailable.from_dict("ground_height", data["ground_height"]),
        )


# ----------------------------------------------------------------- provenance --

@dataclass(frozen=True)
class ContactCalibration:
    """Model/domain calibration of the docs/51 contact rule — never per-video.

    Fit once on the TRAIN split of the Frame Pose output domain it is applied
    to (docs/52 T1), persisted, and then consumed unchanged by every sequence.
    `sampling` records the row spacing the rule saw: classify_foot_contact
    differentiates over neighbouring rows at `source_fps` and ignores
    timestamps, so thresholds are only meaningful at the row stride they were
    fit on.
    """

    thresholds: dict[str, dict[str, float | int]]
    source: dict[str, Any]
    sampling: dict[str, Any]
    rule_version: str = CONTACT_RULE_VERSION
    percentiles: dict[str, float] = field(default_factory=lambda: dict(CONTACT_RULE_PERCENTILES))
    interpretation: str = CONTACT_INTERPRETATION

    def __post_init__(self) -> None:
        if set(self.thresholds) != set(FOOT_SIDES):
            raise ValueError("contact calibration needs exactly left/right thresholds")
        if self.source.get("split") != "train":
            raise ValueError("contact calibration must be fit on the train split only")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CONTACT_CALIBRATION_SCHEMA, "rule_version": self.rule_version,
                "percentiles": dict(self.percentiles), "thresholds": self.thresholds,
                "source": self.source, "sampling": self.sampling, "interpretation": self.interpretation}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContactCalibration":
        if data.get("schema") != CONTACT_CALIBRATION_SCHEMA:
            raise ValueError(f"not a {CONTACT_CALIBRATION_SCHEMA} payload")
        if data["rule_version"] != CONTACT_RULE_VERSION:
            raise ValueError(f"contact rule version {data['rule_version']!r} is not {CONTACT_RULE_VERSION!r}")
        return cls(thresholds=data["thresholds"], source=data["source"], sampling=data["sampling"],
                   rule_version=data["rule_version"], percentiles=data["percentiles"],
                   interpretation=data["interpretation"])


def save_contact_calibration(calibration: ContactCalibration, path: str | Path) -> None:
    write_json(path, calibration.to_dict())


def load_contact_calibration(path: str | Path) -> ContactCalibration:
    return ContactCalibration.from_dict(read_json(path))


@dataclass(frozen=True)
class SemanticsProvenance:
    """Where each owner's evidence came from. Kept per owner, never merged."""

    frame_pose: dict[str, Any]
    root_orientation: dict[str, Any]
    contact: dict[str, Any]
    reliability: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"frame_pose": self.frame_pose, "root_orientation": self.root_orientation,
                "contact": self.contact, "reliability": self.reliability}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticsProvenance":
        return cls(data["frame_pose"], data["root_orientation"], data["contact"], data["reliability"])


# ------------------------------------------------------------------- sequence --

@dataclass(frozen=True)
class AnimationSemantics:
    sequence_id: str
    source_fps: float
    coordinate_frame: str
    joint_names: tuple[str, ...]
    frames: tuple[SemanticFrame, ...]
    provenance: SemanticsProvenance

    def __post_init__(self) -> None:
        indices = [frame.frame_index for frame in self.frames]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ValueError("semantic frames must be strictly increasing in frame_index")
        for frame in self.frames:
            if len(frame.articulation.joint_positions) != len(self.joint_names):
                raise ValueError(f"frame {frame.frame_index}: articulation does not match joint_names")
            if len(frame.reliability.joint_observation_valid) != len(self.joint_names):
                raise ValueError(f"frame {frame.frame_index}: reliability does not match joint_names")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SEMANTICS_SCHEMA,
            "sequence_id": self.sequence_id,
            "source_fps": self.source_fps,
            "coordinate_frame": self.coordinate_frame,
            "joint_names": list(self.joint_names),
            "ownership": ownership_table(),
            "unavailable_quantities": dict(UNAVAILABLE_REASONS),
            "provenance": self.provenance.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnimationSemantics":
        if data.get("schema") != SEMANTICS_SCHEMA:
            raise ValueError(f"not a {SEMANTICS_SCHEMA} payload (schema={data.get('schema')!r})")
        if data.get("ownership") != ownership_table():
            raise ValueError("serialized ownership table does not match pose.motion_ownership")
        return cls(
            sequence_id=data["sequence_id"],
            source_fps=float(data["source_fps"]),
            coordinate_frame=data["coordinate_frame"],
            joint_names=tuple(data["joint_names"]),
            frames=tuple(SemanticFrame.from_dict(frame) for frame in data["frames"]),
            provenance=SemanticsProvenance.from_dict(data["provenance"]),
        )

    def content_digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_animation_semantics(semantics: AnimationSemantics, path: str | Path) -> None:
    write_json(path, semantics.to_dict())


def load_animation_semantics(path: str | Path) -> AnimationSemantics:
    return AnimationSemantics.from_dict(read_json(path))


# ------------------------------------------------------- MotionGraph linkage --

def motion_graph_reference(semantics: AnimationSemantics) -> dict[str, Any]:
    """A pointer a caller may place in MotionGraph.source_metadata.

    MotionGraph never copies any semantic quantity; it only points here.
    """
    return {"schema": SEMANTICS_SCHEMA, "sequence_id": semantics.sequence_id,
            "content_digest": semantics.content_digest(), "link_key": ["frame_index", "timestamp"]}


def check_motion_graph_alignment(
    graph: MotionGraph, semantics: AnimationSemantics, *, timestamp_tolerance: float = 1e-6,
) -> list[int]:
    """Semantic frame_indexes with no same-timestamp MotionGraph frame.

    Semantics may cover a subset of a graph's frames (e.g. a stride-sampled
    bank); an empty return means every semantic frame links exactly.
    """
    graph_timestamps = {frame.frame_index: frame.timestamp for frame in graph.frames}
    return [
        frame.frame_index for frame in semantics.frames
        if frame.frame_index not in graph_timestamps
        or abs(graph_timestamps[frame.frame_index] - frame.timestamp) > timestamp_tolerance
    ]

