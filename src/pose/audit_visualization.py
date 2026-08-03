"""Dependency-free SVG contact sheet for 3D target quality review."""

from __future__ import annotations

from pathlib import Path

from pose.root_motion import RootMotionSequence


_BONES = (
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("pelvis", "spine"), ("spine", "thorax"), ("thorax", "neck"), ("neck", "head"),
    ("thorax", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("thorax", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
)
_VIEWS = (("정면 X/Z", 0, 2), ("측면 Y/Z", 1, 2), ("상단 X/Y", 0, 1))


def render_audit_views(root_motion: RootMotionSequence, path: str | Path, frame_indices: list[int] | None = None) -> None:
    if not root_motion.frames:
        raise ValueError("cannot render an empty root-motion sequence")
    selected = frame_indices or _default_indices(len(root_motion.frames))
    by_index = {frame.frame_index: frame for frame in root_motion.frames}
    frames = [by_index[index] for index in selected if index in by_index]
    if not frames:
        raise ValueError("none of the requested audit frame indices exist")
    panel, padding = 260, 28
    width, height = panel * len(_VIEWS), panel * len(frames)
    all_points = [point for frame in frames for point in frame.character_points.values()]
    minimum = [min(point[axis] for point in all_points) for axis in range(3)]
    maximum = [max(point[axis] for point in all_points) for axis in range(3)]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#15191f"/>']
    for row, frame in enumerate(frames):
        for column, (label, horizontal, vertical) in enumerate(_VIEWS):
            origin_x, origin_y = column * panel, row * panel
            lines.extend(_panel(frame, label, horizontal, vertical, origin_x, origin_y, panel, padding, minimum, maximum))
    lines.append("</svg>")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _panel(frame, label, horizontal, vertical, x, y, panel, padding, minimum, maximum):
    def project(point):
        span_x = max(maximum[horizontal] - minimum[horizontal], 1e-6)
        span_y = max(maximum[vertical] - minimum[vertical], 1e-6)
        usable = panel - 2 * padding
        return x + padding + (point[horizontal] - minimum[horizontal]) / span_x * usable, y + panel - padding - (point[vertical] - minimum[vertical]) / span_y * usable
    lines = [f'<rect x="{x}" y="{y}" width="{panel}" height="{panel}" fill="none" stroke="#3c4654"/>',
             f'<text x="{x + 10}" y="{y + 18}" fill="#dce5ef" font-family="sans-serif" font-size="12">f{frame.frame_index} · {label}</text>']
    for start, end in _BONES:
        if start in frame.character_points and end in frame.character_points:
            x1, y1 = project(frame.character_points[start]); x2, y2 = project(frame.character_points[end])
            lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#55c7ff" stroke-width="2"/>')
    for point in frame.character_points.values():
        px, py = project(point)
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.8" fill="#ffb86b"/>')
    return lines


def _default_indices(count):
    return sorted({0, count // 2, count - 1})
