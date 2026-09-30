#!/usr/bin/env python3
"""Bounded Blender A/B: unchanged object-owned FK vs two-bone IK on identical frames.

Each case is an isolated pose (no keyframes, no interpolation).  The same
Armature Object rotation is used for A and B; only upperarm_l / lowerarm_l
pose-bone quaternions differ.  Frames with any UNAVAILABLE sample are refused
and recorded, never substituted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FK_COLOR = (1.0, 0.45, 0.1, 1.0)
IK_COLOR = (0.1, 0.8, 1.0, 1.0)
REQUESTED_COLOR = (1.0, 0.9, 0.1, 1.0)


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--rest", type=Path, required=True)
    parser.add_argument("--chains", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True,
                        help="label:object_fk_json:ik_json:frame_index")
    return parser.parse_args(argv)


def _chain_points(armature, proximal, distal):
    p, d = armature.pose.bones[proximal], armature.pose.bones[distal]
    return [tuple(float(v) for v in x) for x in (p.head, p.tail, d.tail)]  # armature space


def _material_object(obj, color):
    obj.color = color
    return obj


def _segment(bpy, name, start, end, color, radius):
    from mathutils import Vector

    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=1)
    obj = bpy.context.object
    obj.name = name
    direction = end - start
    obj.location = (start + end) / 2
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    obj.scale = (1, 1, direction.length)
    return _material_object(obj, color)


def _sphere(bpy, name, location, color, radius):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=16, ring_count=8)
    obj = bpy.context.object
    obj.name = name
    return _material_object(obj, color)


def _camera(bpy, name):
    data = bpy.data.cameras.new(name)
    data.type = "ORTHO"
    data.ortho_scale = 0.95
    camera = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(camera)
    return camera


def _aim(camera, location, target, up):
    from mathutils import Matrix, Vector

    location, target, up = Vector(location), Vector(target), Vector(up)
    forward = (target - location).normalized()
    right = forward.cross(up).normalized()
    true_up = right.cross(forward)
    rotation = Matrix((right, true_up, -forward)).transposed()
    camera.location = location
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = rotation.to_quaternion()


def main():
    options = _args()
    try:
        import bpy
        from mathutils import Quaternion, Vector
        from blender.executor import BlenderExecutor
        from blender.framepose_object_owner_adapter import (
            apply_object_owned_frame, bind_imported_armature,
        )
        from retarget.framepose_object_owner import load_object_owned_fk
        from retarget.framepose_target_rest import load_target_rest_pose
        from retarget.framepose_two_bone_ik import (
            load_ik_chain_set, load_two_bone_ik, target_chain_geometry,
        )
        from review_framepose_fk_blender import _make_bone_proxies, _render_setup, _update_bone_proxies

        bpy.ops.wm.read_factory_settings(use_empty=True)
        armature = BlenderExecutor().import_rig(str(options.rig))
        rest = load_target_rest_pose(options.rest)
        chain = load_ik_chain_set(options.chains).chains[0]
        geometry = target_chain_geometry(rest, chain)
        binding = bind_imported_armature(armature, rest, source_path=str(options.rig))
        _render_setup(bpy, armature)
        proxies = _make_bone_proxies(bpy, armature)
        for name in (geometry.proximal, geometry.distal):
            proxies.pop(name).hide_render = True  # replaced by explicit FK / IK proxies
        cameras = {"front": _camera(bpy, "IK review front"), "side": _camera(bpy, "IK review side")}
        scene = bpy.context.scene
        options.out_dir.mkdir(parents=True, exist_ok=True)
        report = {"schema": "animcv_framepose_two_bone_ik_blender_review_v1",
                  "rest_snapshot_digest": rest.digest(), "blender_version": bpy.app.version_string,
                  "target_chain": geometry.to_dict(), "cases": {}, "refused": {}}

        for item in options.case:
            label, fk_path, ik_path, index = item.split(":", 3)
            frame_index = int(index)
            fk = load_object_owned_fk(fk_path)
            ik = load_two_bone_ik(ik_path)
            if ik.provenance["rest_snapshot_digest"] != rest.digest():
                raise ValueError(f"{label}: IK rest digest mismatch")
            sample = next(s for s in ik.samples if s.frame_index == frame_index)
            if sample.proximal_status != "known":
                report["refused"][label] = {"sequence_id": ik.sequence_id, "frame_index": frame_index,
                                            "reason": sample.reason}
                continue
            # A: unchanged Worklog 60 object rotation + local FK.
            apply_object_owned_frame(armature, binding, fk, frame_index)
            fk_points = _chain_points(armature, geometry.proximal, geometry.distal)
            # B: same object rotation and all other pose bones; IK chain locals only.
            for name, q in ((geometry.proximal, sample.proximal_local), (geometry.distal, sample.distal_local)):
                x, y, z, w = q
                armature.pose.bones[name].rotation_quaternion = (w, x, y, z)
            bpy.context.view_layer.update()
            ik_points = _chain_points(armature, geometry.proximal, geometry.distal)
            if tuple(armature.location) != binding.location or tuple(armature.scale) != binding.scale:
                raise RuntimeError("review changed object translation or scale")
            _update_bone_proxies(bpy, armature, proxies)

            world = armature.matrix_world.copy()
            to_world = lambda p: tuple(world @ Vector(p))  # noqa: E731
            requested = sample.requested_endpoint
            comparison = next(r for r in json.loads(Path(ik_path.replace(".ik.json", ".fk_vs_ik.json")).read_text())
                              if r["frame_index"] == frame_index)
            created = []
            for tag, points, color in (("FK", fk_points, FK_COLOR), ("IK", ik_points, IK_COLOR)):
                w = [to_world(p) for p in points]
                created += [_segment(bpy, f"{tag} upperarm", w[0], w[1], color, 0.011),
                            _segment(bpy, f"{tag} lowerarm", w[1], w[2], color, 0.011),
                            _sphere(bpy, f"{tag} endpoint", w[2], color, 0.018)]
            created.append(_sphere(bpy, "requested endpoint", to_world(requested), REQUESTED_COLOR, 0.028))
            created[-1].display_type = "WIRE"

            alignment = ik.provenance["canonical_heading_zero_to_armature_xyzw"]
            aq = Quaternion((alignment[3], *alignment[:3]))
            object_rotation = world.to_quaternion()
            axis = lambda v: (object_rotation @ (aq @ Vector(v))).normalized()  # noqa: E731
            forward, up, right = axis((0, 1, 0)), axis((0, 0, 1)), axis((1, 0, 0))
            centre = Vector(to_world(geometry.root_head)) + 0.22 * ((Vector(to_world(requested)) -
                                                                     Vector(to_world(geometry.root_head))).normalized())
            views = {}
            for view, offset in (("front", forward * 3.0), ("side", -right * 3.0)):
                _aim(cameras[view], centre + offset, centre, up)
                scene.camera = cameras[view]
                path = options.out_dir / f"{label}_{view}.png"
                scene.render.filepath = str(path)
                bpy.ops.render.render(write_still=True)
                views[view] = path.name
            bpy.ops.wm.save_as_mainfile(filepath=str(options.out_dir / f"{label}.blend"))

            fk_end_err = math.dist(fk_points[2], comparison["fk_endpoint"])
            ik_end_err = math.dist(ik_points[2], sample.solved_endpoint)
            report["cases"][label] = {
                "sequence_id": ik.sequence_id, "frame_index": frame_index,
                "object_rotation_identical_for_A_and_B": True,
                "object_location_unchanged": True,
                "blender_vs_python_fk_endpoint_discrepancy": fk_end_err,
                "blender_vs_python_ik_endpoint_discrepancy": ik_end_err,
                "blender_fk_endpoint_error_to_requested": math.dist(fk_points[2], requested),
                "blender_ik_endpoint_error_to_requested": math.dist(ik_points[2], requested),
                "blender_ik_lengths": [math.dist(ik_points[0], ik_points[1]),
                                       math.dist(ik_points[1], ik_points[2])],
                "source_reach_fraction": sample.source_reach_fraction,
                "source_bend_angle_degrees": sample.source_bend_angle_degrees,
                "bend_plane_status": sample.bend_plane_status,
                "clamp_status": sample.clamp_status,
                "proximal_delta_degrees": comparison["proximal_delta_degrees"],
                "distal_delta_degrees": comparison["distal_delta_degrees"],
                "png": views, "blend": f"{label}.blend",
            }
            for obj in created:
                bpy.data.objects.remove(obj, do_unlink=True)
        (options.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"two-bone IK Blender review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
