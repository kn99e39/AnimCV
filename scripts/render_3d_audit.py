"""Render selected prepared 3D target skeleton frames in headless Blender.

Usage:
  blender --background --python scripts/render_3d_audit.py -- \
    --root-motion constraint_ready_root_motion.json --out-dir audit_renders
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
    "front": ((0.0, -4.0, 0.6), (0.0, 0.0, 0.25)),
    "side": ((4.0, 0.0, 0.6), (0.0, 0.0, 0.25)),
    "three_quarter": ((3.2, -3.2, 2.0), (0.0, 0.0, 0.25)),
}


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-motion", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", default="0,60,119")
    return parser.parse_args(argv)


def _material(name, colour):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*colour, 1.0)
    return material


def _look_at(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def _add_segment(start, end, material):
    direction = Vector(end) - Vector(start)
    midpoint = (Vector(start) + Vector(end)) / 2
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.012, depth=direction.length, location=midpoint)
    object_ = bpy.context.object
    object_.rotation_mode = "QUATERNION"
    object_.rotation_quaternion = direction.to_track_quat("Z", "Y")
    object_.data.materials.append(material)


def _add_joint(position, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.025, location=position)
    bpy.context.object.data.materials.append(material)


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _render_frame(points, frame_index, out_dir, bone_material, joint_material):
    _clear_scene()
    for start, end in _BONES:
        if start in points and end in points:
            _add_segment(points[start], points[end], bone_material)
    for position in points.values():
        _add_joint(position, joint_material)
    bpy.ops.object.light_add(type="AREA", location=(0.0, -2.0, 3.0))
    bpy.context.object.data.energy = 600
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 5.0
    for label, (location, target) in _VIEWS.items():
        bpy.ops.object.camera_add(location=location)
        camera = bpy.context.object
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = 1.8
        _look_at(camera, target)
        bpy.context.scene.camera = camera
        bpy.context.scene.render.filepath = str(out_dir / f"frame_{frame_index:03d}_{label}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(camera, do_unlink=True)


def main():
    args = _args()
    data = json.loads(Path(args.root_motion).read_text())
    frames = {frame["frame_index"]: frame["character_points"] for frame in data["frames"]}
    requested = [int(value) for value in args.frames.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.display.shading.light = "STUDIO"
    scene.display.shading.background_type = "WORLD"
    scene.display.shading.background_color = (0.035, 0.045, 0.06)
    bone_material = _material("target_bones", (0.1, 0.65, 1.0))
    joint_material = _material("target_joints", (1.0, 0.38, 0.08))
    for index in requested:
        if index not in frames:
            raise ValueError(f"requested frame {index} is absent")
        _render_frame(frames[index], index, out_dir, bone_material, joint_material)
    print(f"[motion-tool] rendered {len(requested) * len(_VIEWS)} 3D audit PNGs -> {out_dir}")


if __name__ == "__main__":
    main()
