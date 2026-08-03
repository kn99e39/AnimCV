"""Render the posed armature's actual bone geometry for visual FBX review."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", default="0,60,119")
    return parser.parse_args(argv)


def _look_at(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def _segment(start, end, material, radius):
    direction = end - start
    if direction.length <= 1e-6:
        return
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=direction.length, location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    obj.data.materials.append(material)


def _joint(position, material, radius):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=radius, location=position)
    bpy.context.object.data.materials.append(material)


def _clear_meshes():
    for obj in list(bpy.data.objects):
        if obj.type in {"MESH", "LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def _render(armature, frame, out_dir, material, joint_material):
    bpy.context.scene.frame_set(frame)
    _clear_meshes()
    segments = []
    for bone in armature.pose.bones:
        start = armature.matrix_world @ bone.head
        end = armature.matrix_world @ bone.tail
        segments.append((start, end))
    points = [point for segment in segments for point in segment]
    minimum = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maximum = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    centre = (minimum + maximum) / 2
    span = max((maximum - minimum).length, 1.0)
    for start, end in segments:
        _segment(start, end, material, span * 0.004)
        _joint(start, joint_material, span * 0.008)
    bpy.ops.object.light_add(type="AREA", location=centre + Vector((0, -span, span)))
    bpy.context.object.data.energy = 700
    bpy.context.object.data.size = span * 2
    for label, direction in {"front": (0, -1, .2), "side": (1, 0, .2), "three_quarter": (1, -1, .5)}.items():
        bpy.ops.object.camera_add(location=centre + Vector(direction).normalized() * span * 2.5)
        camera = bpy.context.object
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = span * 1.25
        _look_at(camera, centre)
        bpy.context.scene.camera = camera
        bpy.context.scene.render.filepath = str(out_dir / f"frame_{frame:03d}_{label}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(camera, do_unlink=True)


def main():
    args = _args()
    bpy.ops.wm.open_mainfile(filepath=args.blend)
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one armature, found {len(armatures)}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = scene.render.resolution_y = 640
    scene.display.shading.light = "STUDIO"
    scene.display.shading.background_type = "WORLD"
    scene.display.shading.background_color = (.035, .045, .06)
    material = bpy.data.materials.new("rig_bones"); material.diffuse_color = (.25, .95, .55, 1)
    joint_material = bpy.data.materials.new("rig_joints"); joint_material.diffuse_color = (1, .7, .1, 1)
    for frame in [int(value) for value in args.frames.split(",")]:
        _render(armatures[0], frame, out_dir, material, joint_material)
    print(f"[motion-tool] rendered actual rig poses -> {out_dir}")


if __name__ == "__main__":
    main()
