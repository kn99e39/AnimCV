"""Versioned FramePose semantics; v1 remains the historical held-yaw sidecar."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.serialization import read_json, write_json
from motion.animation_semantics import (
    FootMotion, LocalArticulation, ObservationReliability, SemanticsProvenance,
    Unavailable, UNAVAILABLE_REASONS, ownership_table,
)
from pose.contact_time_aware import RULE_VERSION, TimeAwareThresholds

SEMANTICS_SCHEMA_V2 = "animcv_animation_semantics_v2"
CALIBRATION_SCHEMA_V2 = "animcv_contact_calibration_v2"
LEGACY_ROOT_POLICY_V2 = "framepose_current_bilateral_v1"
ROOT_POLICY_V2 = "framepose_current_valid_bilateral_v1"


@dataclass(frozen=True)
class CurrentRootOrientation:
    known: bool
    yaw_radians: float | None = None

    def __post_init__(self) -> None:
        if self.known != (self.yaw_radians is not None):
            raise ValueError("known orientation requires yaw; unknown carries no yaw")
        if self.yaw_radians is not None and not math.isfinite(self.yaw_radians):
            raise ValueError("yaw must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"status": "known" if self.known else "unknown", "yaw_radians": self.yaw_radians}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurrentRootOrientation":
        if set(data) != {"status", "yaw_radians"} or data["status"] not in ("known", "unknown"):
            raise ValueError("invalid v2 RootOrientation")
        return cls(data["status"] == "known", data["yaw_radians"])


@dataclass(frozen=True)
class SemanticFrameV2:
    frame_index: int
    timestamp: float
    articulation: LocalArticulation
    root_orientation: CurrentRootOrientation
    foot_motion: FootMotion
    reliability: ObservationReliability
    root_translation: Unavailable = field(default_factory=lambda: Unavailable("root_translation"))
    ground_height: Unavailable = field(default_factory=lambda: Unavailable("ground_height"))

    def to_dict(self) -> dict[str, Any]:
        return {"frame_index": self.frame_index, "timestamp": self.timestamp,
                "articulation": self.articulation.to_dict(), "root_orientation": self.root_orientation.to_dict(),
                "foot_motion": self.foot_motion.to_dict(), "reliability": self.reliability.to_dict(),
                "root_translation": self.root_translation.to_dict(), "ground_height": self.ground_height.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticFrameV2":
        return cls(int(data["frame_index"]), float(data["timestamp"]),
                   LocalArticulation.from_dict(data["articulation"]),
                   CurrentRootOrientation.from_dict(data["root_orientation"]),
                   FootMotion.from_dict(data["foot_motion"]),
                   ObservationReliability.from_dict(data["reliability"]),
                   Unavailable.from_dict("root_translation", data["root_translation"]),
                   Unavailable.from_dict("ground_height", data["ground_height"]))


@dataclass(frozen=True)
class TimeAwareCalibration:
    thresholds: dict[str, dict[str, float]]
    source: dict[str, Any]
    sampling: dict[str, Any]

    def __post_init__(self) -> None:
        if set(self.thresholds) != {"left", "right"} or self.source.get("split") != "train":
            raise ValueError("time-aware calibration requires left/right TRAIN fit")
        if self.sampling.get("time_basis") != "elapsed_seconds" or self.sampling.get("train_median_row_stride_frames", 0) <= 0:
            raise ValueError("time-aware calibration requires TRAIN time derivation")
        for side in ("left", "right"):
            TimeAwareThresholds(**self.thresholds[side])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CALIBRATION_SCHEMA_V2, "rule_version": RULE_VERSION,
                "percentiles": {"contact": 15.0, "moving": 60.0, "height": 40.0},
                "thresholds": self.thresholds, "source": self.source, "sampling": self.sampling,
                "interpretation": "contact-like root-relative foot motion; KINEMATIC_PROXY is evaluation only"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeAwareCalibration":
        if data.get("schema") != CALIBRATION_SCHEMA_V2 or data.get("rule_version") != RULE_VERSION:
            raise ValueError("not the FramePose time-aware calibration")
        if data.get("percentiles") != {"contact": 15.0, "moving": 60.0, "height": 40.0}:
            raise ValueError("contact percentiles changed")
        return cls(data["thresholds"], data["source"], data["sampling"])


def save_time_aware_calibration(calibration: TimeAwareCalibration, path: str | Path) -> None:
    write_json(path, calibration.to_dict())


def load_time_aware_calibration(path: str | Path) -> TimeAwareCalibration:
    return TimeAwareCalibration.from_dict(read_json(path))


@dataclass(frozen=True)
class AnimationSemanticsV2:
    sequence_id: str
    source_fps: float
    coordinate_frame: str
    joint_names: tuple[str, ...]
    frames: tuple[SemanticFrameV2, ...]
    provenance: SemanticsProvenance
    allow_legacy_policy: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        indices = [frame.frame_index for frame in self.frames]
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError("semantic frames must have increasing indices")
        policy = self.provenance.root_orientation.get("policy")
        if policy != ROOT_POLICY_V2 and not (self.allow_legacy_policy and policy == LEGACY_ROOT_POLICY_V2):
            raise ValueError("v2 requires the valid-only bilateral policy; old v2 requires explicit legacy loading")
        if self.provenance.contact.get("rule_version") != RULE_VERSION:
            raise ValueError("v2 requires time-aware contact policy")
        for frame in self.frames:
            if len(frame.articulation.joint_positions) != len(self.joint_names) or len(frame.reliability.joint_observation_valid) != len(self.joint_names):
                raise ValueError("joint count mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SEMANTICS_SCHEMA_V2, "sequence_id": self.sequence_id,
                "source_fps": self.source_fps, "coordinate_frame": self.coordinate_frame,
                "joint_names": list(self.joint_names), "ownership": ownership_table(),
                "unavailable_quantities": dict(UNAVAILABLE_REASONS),
                "provenance": self.provenance.to_dict(), "frames": [frame.to_dict() for frame in self.frames]}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, allow_legacy_policy: bool = False) -> "AnimationSemanticsV2":
        if data.get("schema") != SEMANTICS_SCHEMA_V2 or data.get("ownership") != ownership_table():
            raise ValueError("not an AnimationSemantics v2 payload")
        return cls(data["sequence_id"], float(data["source_fps"]), data["coordinate_frame"],
                   tuple(data["joint_names"]), tuple(SemanticFrameV2.from_dict(v) for v in data["frames"]),
                   SemanticsProvenance.from_dict(data["provenance"]), allow_legacy_policy)

    def content_digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def save_animation_semantics_v2(semantics: AnimationSemanticsV2, path: str | Path) -> None:
    write_json(path, semantics.to_dict())


def load_animation_semantics_v2(path: str | Path, *, allow_legacy_policy: bool = False) -> AnimationSemanticsV2:
    return AnimationSemanticsV2.from_dict(read_json(path), allow_legacy_policy=allow_legacy_policy)
