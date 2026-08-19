"""Render an extracted Blender armature animation to a reviewable MP4.

The script deliberately renders a coloured bone/joint proxy in addition to
the original scene.  That means a rig which contains only an armature (or a
character mesh with an unhelpful material) still yields an inspectable video.

Run it inside Blender, after ``scripts/apply_motion.py`` has produced a blend
file::

    blender --background --python scripts/render_blender_animation_video.py -- \\
      --blend output/result.blend --out output/result_review.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bpy
from mathutils import Vector


OVERLAY_COLLECTION = "AnimCV_Animation_Review_Overlay"


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render an armature animation review MP4")
    parser.add_argument("--blend", required=True, help="Animated .blend exported by apply_motion.py")
    parser.add_argument("--out", required=True, help="Output .mp4 path")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--camera", choices=("front", "side", "three_quarter"), default="three_quarter")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Render every Nth source frame (the output FPS is reduced to preserve duration)")
    parser.add_argument("--hide-original-mesh", action="store_true",
                        help="Render only the animated bone/joint proxy")
    return parser.parse_args(argv)


def _look_at(camera, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def _remove_existing_overlay() -> None:
    collection = bpy.data.collections.get(OVERLAY_COLLECTION)
    if collection is None:
        return
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def _new_material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _link_to(collection, obj) -> None:
    for old_collection in list(obj.users_collection):
        old_collection.objects.unlink(obj)
    collection.objects.link(obj)


def _make_proxy(armature, radius: float):
    """Create one cylinder and one head sphere per pose bone.

    A frame-change handler updates their matrices before each render frame.
    Keeping the proxies as normal mesh objects lets Workbench render them in
    headless Blender without depending on viewport-only armature display.
    """
    collection = bpy.data.collections.new(OVERLAY_COLLECTION)
    bpy.context.scene.collection.children.link(collection)
    bone_material = _new_material("AnimCV review bones", (0.08, 0.88, 0.42, 1.0))
    joint_material = _new_material("AnimCV review joints", (1.0, 0.54, 0.04, 1.0))
    proxies = []
    for bone in armature.pose.bones:
        bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=radius, depth=1.0)
        segment = bpy.context.object
        segment.name = f"__animcv_segment__{bone.name}"
        segment.data.materials.append(bone_material)
        _link_to(collection, segment)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=radius * 1.8)
        joint = bpy.context.object
        joint.name = f"__animcv_joint__{bone.name}"
        joint.data.materials.append(joint_material)
        _link_to(collection, joint)
        proxies.append((bone.name, segment, joint))
    return proxies


def _update_proxy(armature, proxies) -> None:
    for bone_name, segment, joint in proxies:
        bone = armature.pose.bones[bone_name]
        start = armature.matrix_world @ bone.head
        end = armature.matrix_world @ bone.tail
        direction = end - start
        length = max(direction.length, 1e-6)
        segment.location = (start + end) / 2
        segment.rotation_mode = "QUATERNION"
        segment.rotation_quaternion = direction.to_track_quat("Z", "Y") if direction.length > 1e-6 else (1, 0, 0, 0)
        segment.scale = (1.0, 1.0, length)
        joint.location = start


def _bounds(armature, start: int, end: int, step: int) -> tuple[Vector, float]:
    points = []
    for frame in range(start, end + 1, step):
        bpy.context.scene.frame_set(frame)
        for bone in armature.pose.bones:
            points.extend((armature.matrix_world @ bone.head, armature.matrix_world @ bone.tail))
    if not points:
        raise RuntimeError("armature contains no pose bones")
    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return (minimum + maximum) / 2, max((maximum - minimum).length, 1.0)


def _configure_scene(args, centre: Vector, span: float, frame_start: int, frame_end: int) -> None:
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end, scene.frame_step = frame_start, frame_end, args.frame_step
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(Path(args.out).resolve())
    # Blender advances by frame_step, so lower the output FPS by the same
    # factor and preserve the clip's original wall-clock duration.
    scene.render.fps_base = scene.render.fps_base * args.frame_step
    scene.display.shading.light = "STUDIO"
    scene.display.shading.background_type = "WORLD"
    scene.display.shading.background_color = (0.025, 0.035, 0.055)
    direction = {"front": (0, -1, .18), "side": (1, 0, .18), "three_quarter": (1, -1, .42)}[args.camera]
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
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    if not args.out.lower().endswith(".mp4"):
        raise ValueError("--out must end in .mp4")
    bpy.ops.wm.open_mainfile(filepath=args.blend)
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one armature, found {len(armatures)}")
    armature = armatures[0]
    start = args.start_frame if args.start_frame is not None else bpy.context.scene.frame_start
    end = args.end_frame if args.end_frame is not None else bpy.context.scene.frame_end
    if end < start:
        raise ValueError("--end-frame must not precede --start-frame")
    centre, span = _bounds(armature, start, end, args.frame_step)
    _remove_existing_overlay()
    proxies = _make_proxy(armature, span * 0.006)
    _update_proxy(armature, proxies)
    if args.hide_original_mesh:
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH" and obj.name not in {proxy.name for _, segment, joint in proxies for proxy in (segment, joint)}:
                obj.hide_render = True
    _configure_scene(args, centre, span, start, end)

    def update_for_render(_scene):
        _update_proxy(armature, proxies)

    bpy.app.handlers.frame_change_pre.append(update_for_render)
    try:
        bpy.ops.render.render(animation=True)
    finally:
        bpy.app.handlers.frame_change_pre.remove(update_for_render)
    print(f"[motion-tool] rendered animation review video -> {args.out}")


if __name__ == "__main__":
    main()
