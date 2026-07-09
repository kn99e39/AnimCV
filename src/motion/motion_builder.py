"""Build the initial dense MotionGraph directly from a PoseSequence.

This is the pre-retargeting, pre-bone-mapping motion representation: it
stores observed 2D (and, when depth was sampled — see
pose/depth_sampling.py — 3D) landmark tracks keyed by semantic name.
Bone-specific transforms are produced later by the retarget solver
(Milestone 5), and importance/lock values here stay at their defaults
until the Keyframe Optimizer (Milestone 6) fills them in.
"""

from __future__ import annotations

from typing import Any

from motion.motion_graph import MotionFrame, MotionGraph, MotionPoint, MotionTrack
from pose.pose_types import PoseSequence


class MotionGraphBuilder:
    def build(
        self, poses: PoseSequence, source_metadata: dict[str, Any] | None = None
    ) -> MotionGraph:
        track_points: dict[str, list[MotionPoint]] = {}
        motion_frames: list[MotionFrame] = []

        for pose_frame in poses.frames:
            points: dict[str, MotionPoint] = {}
            for name, landmark in pose_frame.landmarks.items():
                point = MotionPoint(
                    semantic_name=name,
                    frame_index=pose_frame.frame_index,
                    position_2d=(landmark.x, landmark.y),
                    position_3d=(
                        (landmark.x, landmark.y, landmark.z)
                        if landmark.z is not None
                        else None
                    ),
                    confidence=landmark.confidence,
                    visible=landmark.visible,
                )
                points[name] = point
                track_points.setdefault(name, []).append(point)

            motion_frames.append(
                MotionFrame(
                    frame_index=pose_frame.frame_index,
                    timestamp=pose_frame.timestamp,
                    points=points,
                    importance=0.0,
                    locked=False,
                )
            )

        tracks = {
            name: MotionTrack(semantic_name=name, points=points)
            for name, points in track_points.items()
        }

        return MotionGraph(
            frames=motion_frames,
            tracks=tracks,
            fps=poses.source_fps,
            source_metadata=dict(source_metadata or {}),
        )
