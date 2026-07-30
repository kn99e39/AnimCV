"""Small, deterministic temporal filter for retarget input landmarks."""

from __future__ import annotations

import statistics

from motion.motion_graph import MotionFrame, MotionGraph, MotionPoint


def median_filter_motion_graph(motion_graph: MotionGraph, window_size: int = 3) -> MotionGraph:
    """Return a copy with visible landmark positions median-filtered over time.

    Only complete, odd-sized neighborhoods are filtered.  Endpoints and frames
    next to an occlusion are retained exactly, so the filter cannot fabricate a
    pose across missing observations. Confidence and visibility remain raw;
    quality checks therefore still describe the original detector evidence.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("smoothing window must be an odd positive integer")
    if window_size == 1:
        return motion_graph

    radius = window_size // 2
    frames = motion_graph.frames
    filtered_frames: list[MotionFrame] = []
    for index, frame in enumerate(frames):
        points: dict[str, MotionPoint] = {}
        for name, point in frame.points.items():
            neighbors = _visible_neighbors(frames, index, name, radius)
            if len(neighbors) == window_size:
                position_2d = (
                    statistics.median(item.position_2d[0] for item in neighbors),
                    statistics.median(item.position_2d[1] for item in neighbors),
                )
                position_3d = _median_3d(neighbors)
            else:
                position_2d = point.position_2d
                position_3d = point.position_3d
            points[name] = MotionPoint(
                semantic_name=point.semantic_name,
                frame_index=point.frame_index,
                position_2d=position_2d,
                position_3d=position_3d,
                confidence=point.confidence,
                visible=point.visible,
            )
        filtered_frames.append(
            MotionFrame(
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                points=points,
                importance=frame.importance,
                locked=frame.locked,
            )
        )
    return MotionGraph(
        frames=filtered_frames,
        tracks=motion_graph.tracks,
        fps=motion_graph.fps,
        source_metadata=dict(motion_graph.source_metadata),
    )


def _visible_neighbors(
    frames: list[MotionFrame], index: int, name: str, radius: int
) -> list[MotionPoint]:
    start, end = index - radius, index + radius + 1
    if start < 0 or end > len(frames):
        return []
    points = [frame.points.get(name) for frame in frames[start:end]]
    if any(point is None or not point.visible for point in points):
        return []
    return [point for point in points if point is not None]


def _median_3d(points: list[MotionPoint]):
    if any(point.position_3d is None for point in points):
        return points[len(points) // 2].position_3d
    return tuple(statistics.median(point.position_3d[i] for point in points) for i in range(3))
