"""Render a ground-truth + predicted skeleton window to a reviewable MP4.

Consumes the sequence JSON written by ``scripts/export_lifter_audit_sequence.py``
and overlays both skeletons in one scene -- ground truth in cyan, the
checkpoint's prediction in amber -- so a flip or a yaw wobble shows up as the
two skeletons visibly diverging, without needing a retargeted rig or mesh.

Run inside Blender::

    blender --background --python scripts/render_lifter_audit_video.py -- \\
      --sequence audit/stairs_window.json --out audit/stairs_window.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


_BONES = (
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("pelvis", "spine"), ("spine", "thorax"), ("thorax", "neck"), ("neck", "head"),
    ("thorax", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("thorax", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
)
_VIEWS = {
    "front": (0.0, -1.0, 0.18),
    "side": (1.0, 0.0, 0.18),
    "three_quarter": (1.0, -1.0, 0.42),
}
_SKELETONS = {
    "reference": {"label": "reference", "bone_color": (0.31, 0.72, 0.86, 1.0), "joint_color": (0.31, 0.72, 0.86, 1.0)},
    "estimate": {"label": "estimate", "bone_color": (0.85, 0.64, 0.25, 1.0), "joint_color": (0.85, 0.64, 0.25, 1.0)},
}


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render an overlaid GT/predicted skeleton review MP4")
    parser.add_argument("--sequence", required=True, help="JSON written by export_lifter_audit_sequence.py")
    parser.add_argument("--out", required=True, help="Output .mp4 path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--camera", choices=("front", "side", "three_quarter"), default="three_quarter")
    parser.add_argument("--fps", type=float, default=None, help="overrides the sequence file's recorded FPS")
    return parser.parse_args(argv)


def _material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _look_at(camera, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def _make_proxy(kind: str, radius: float):
    style = _SKELETONS[kind]
    bone_material = _material(f"AnimCV audit bones ({kind})", style["bone_color"])
    joint_material = _material(f"AnimCV audit joints ({kind})", style["joint_color"])
    segments = {}
    for start, end in _BONES:
        bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=radius, depth=1.0)
        segment = bpy.context.object
        segment.name = f"__animcv_{kind}_segment__{start}_{end}"
        segment.data.materials.append(bone_material)
        segments[(start, end)] = segment
    joints = {}
    for start, end in _BONES:
        for name in (start, end):
            if name in joints:
                continue
            bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=radius * 1.7)
            joint = bpy.context.object
            joint.name = f"__animcv_{kind}_joint__{name}"
            joint.data.materials.append(joint_material)
            joints[name] = joint
    return segments, joints


def _update_proxy(points: dict[str, list[float]], segments, joints) -> None:
    for (start, end), segment in segments.items():
        a, b = Vector(points[start]), Vector(points[end])
        direction = b - a
        length = max(direction.length, 1e-6)
        segment.location = (a + b) / 2
        segment.rotation_mode = "QUATERNION"
        segment.rotation_quaternion = direction.to_track_quat("Z", "Y") if direction.length > 1e-6 else (1, 0, 0, 0)
        segment.scale = (1.0, 1.0, length)
    for name, joint in joints.items():
        joint.location = Vector(points[name])


def _bounds(frames: list[dict]) -> tuple[Vector, float]:
    points = [
        Vector(value)
        for frame in frames
        for kind in ("reference", "estimate")
        for value in frame[kind].values()
    ]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return (minimum + maximum) / 2, max((maximum - minimum).length, 1.0)


def _configure_scene(args, centre: Vector, span: float, frame_count: int, fps: float) -> None:
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end, scene.frame_step = 1, frame_count, 1
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    # Blender 5.x gates image_settings.file_format's valid enum values on
    # media_type; FFMPEG is only selectable once media_type is "VIDEO".
    if hasattr(scene.render.image_settings, "media_type"):
        scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(Path(args.out).resolve())
    scene.render.fps = round(fps)
    scene.display.shading.light = "STUDIO"
    scene.display.shading.background_type = "WORLD"
    scene.display.shading.background_color = (0.025, 0.035, 0.055)
    direction = _VIEWS[args.camera]
    bpy.ops.object.camera_add(location=centre + Vector(direction).normalized() * span * 2.7)
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = span * 1.20
    _look_at(camera, centre)
    scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=centre + Vector((0, -span, span)))
    bpy.context.object.data.energy = 900
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = span * 2.0


def main() -> None:
    args = _args()
    if not args.out.lower().endswith(".mp4"):
        raise ValueError("--out must end in .mp4")
    data = json.loads(Path(args.sequence).read_text())
    frames = data["frames"]
    if not frames:
        raise ValueError("sequence file has no frames")
    fps = args.fps if args.fps is not None else data.get("fps")
    if not fps:
        raise ValueError("sequence file has no FPS recorded; pass --fps explicitly")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    centre, span = _bounds(frames)
    proxies = {kind: _make_proxy(kind, span * 0.008) for kind in _SKELETONS}
    for kind, (segments, joints) in proxies.items():
        _update_proxy(frames[0][kind], segments, joints)
    _configure_scene(args, centre, span, len(frames), fps)

    def update_for_render(scene):
        index = scene.frame_current - 1
        if 0 <= index < len(frames):
            for kind, (segments, joints) in proxies.items():
                _update_proxy(frames[index][kind], segments, joints)

    bpy.app.handlers.frame_change_pre.append(update_for_render)
    try:
        bpy.ops.render.render(animation=True)
    finally:
        bpy.app.handlers.frame_change_pre.remove(update_for_render)
    print(f"[motion-tool] rendered {len(frames)}-frame audit review video -> {args.out}")


if __name__ == "__main__":
    main()
