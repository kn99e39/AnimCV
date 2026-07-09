"""AnimationClip data model and RetargetSolver (Architecture_v2.md section 7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.serialization import read_json, write_json
from common.types import Quaternion, Vec3
from motion.motion_graph import MotionGraph
from rig.bone_mapping import BoneMappingProfile
from rig.rig_profile import RigProfile


@dataclass
class BoneTransformSample:
    frame_index: int
    bone_name: str
    location: Vec3 | None
    rotation: Quaternion
    scale: Vec3 | None
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "bone_name": self.bone_name,
            "location": list(self.location) if self.location else None,
            "rotation": list(self.rotation),
            "scale": list(self.scale) if self.scale else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoneTransformSample":
        location = data["location"]
        scale = data["scale"]
        return cls(
            frame_index=data["frame_index"],
            bone_name=data["bone_name"],
            location=tuple(location) if location is not None else None,
            rotation=tuple(data["rotation"]),
            scale=tuple(scale) if scale is not None else None,
            confidence=data["confidence"],
        )


@dataclass
class AnimationTrack:
    bone_name: str
    samples: list[BoneTransformSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bone_name": self.bone_name,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnimationTrack":
        return cls(
            bone_name=data["bone_name"],
            samples=[BoneTransformSample.from_dict(s) for s in data["samples"]],
        )


@dataclass
class AnimationClip:
    name: str
    fps: float
    tracks: dict[str, AnimationTrack] = field(default_factory=dict)
    frame_start: int = 0
    frame_end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fps": self.fps,
            "tracks": {name: track.to_dict() for name, track in self.tracks.items()},
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnimationClip":
        return cls(
            name=data["name"],
            fps=data["fps"],
            tracks={
                name: AnimationTrack.from_dict(t)
                for name, t in data["tracks"].items()
            },
            frame_start=data["frame_start"],
            frame_end=data["frame_end"],
        )


def save_animation_clip(clip: AnimationClip, path: str | Path) -> None:
    write_json(path, clip.to_dict())


def load_animation_clip(path: str | Path) -> AnimationClip:
    return AnimationClip.from_dict(read_json(path))


class RetargetSolver:
    """Converts MotionGraph movement into target-rig animation curves.

    FK-oriented, 2D-direction-driven (section 7.2, 7.4): bones mapped
    with mode "direction" get a rotation derived from the image-plane
    angle between two tracked points; bones mapped with mode
    "landmark"/"point" (a single anchor) get a translation offset
    instead, since one point alone carries no direction to rotate from.
    Bones mapped in the profile but absent from the rig, or using an
    unrecognized mapping_mode, are skipped rather than raising — partial
    mapping is required behavior (section 6.5), not an error case.
    """

    def solve(
        self,
        motion_graph: MotionGraph,
        rig_profile: RigProfile,
        mapping_profile: BoneMappingProfile,
    ) -> AnimationClip:
        from retarget.fk_solver import solve_anchor_bone, solve_direction_bone

        tracks: dict[str, AnimationTrack] = {}
        for entry in mapping_profile.entries:
            if entry.target_bone not in rig_profile.bones:
                continue

            if entry.mapping_mode == "direction":
                samples = solve_direction_bone(motion_graph, entry.target_bone, entry)
            elif entry.mapping_mode in ("landmark", "point"):
                samples = solve_anchor_bone(motion_graph, entry.target_bone, entry)
            else:
                continue

            tracks[entry.target_bone] = AnimationTrack(
                bone_name=entry.target_bone, samples=samples
            )

        frame_indices = [motion_frame.frame_index for motion_frame in motion_graph.frames]
        return AnimationClip(
            name="Generated_Motion",
            fps=motion_graph.fps,
            tracks=tracks,
            frame_start=min(frame_indices) if frame_indices else 0,
            frame_end=max(frame_indices) if frame_indices else 0,
        )
