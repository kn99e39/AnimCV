"""Debug overlay renderer: draws pose landmarks onto frames for inspection.

Exports either an image sequence (directory of PNGs) or a video file,
selected by ``out_path``'s suffix.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from mediaio.frame_sequence import FrameSequence
from pose.pose_types import PoseFrame, PoseSequence

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def render_debug_overlay(frames: FrameSequence, poses: PoseSequence, out_path: str) -> None:
    pose_by_frame = {pose_frame.frame_index: pose_frame for pose_frame in poses.frames}
    out = Path(out_path)

    if out.suffix.lower() in _VIDEO_EXTENSIONS:
        _render_video(frames, pose_by_frame, out)
    else:
        _render_image_sequence(frames, pose_by_frame, out)


def _draw_overlay(image, pose_frame: PoseFrame | None):
    overlay = image.copy()
    if pose_frame is None:
        return overlay
    for landmark in pose_frame.landmarks.values():
        x, y = int(landmark.x), int(landmark.y)
        color = (0, 255, 0) if landmark.visible else (0, 0, 255)
        cv2.circle(overlay, (x, y), 4, color, -1)
        cv2.putText(
            overlay, landmark.name, (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    return overlay


def _render_image_sequence(frames: FrameSequence, pose_by_frame: dict[int, PoseFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames.frames:
        overlay = _draw_overlay(frame.image, pose_by_frame.get(frame.index))
        cv2.imwrite(str(out_dir / f"{frame.index:05d}.png"), overlay)


def _render_video(frames: FrameSequence, pose_by_frame: dict[int, PoseFrame], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, frames.fps, (frames.width, frames.height))
    try:
        for frame in frames.frames:
            overlay = _draw_overlay(frame.image, pose_by_frame.get(frame.index))
            writer.write(overlay)
    finally:
        writer.release()
