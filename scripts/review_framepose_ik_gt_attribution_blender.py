#!/usr/bin/env python3
"""EVALUATION ONLY: Blender review of Worklog 62 GT-anchored IK attribution cases.

Run under Blender with a case-geometry JSON from
``scripts/export_framepose_ik_gt_cases.py``.  Each case is an isolated pose:
the Worklog 60 object rotation and FK pose bones place the gray body; the
left-arm variants are drawn as explicit proxies from heading-relative
armature-space points mapped through the posed ``armature.matrix_world``.

Blender's bundled Python has no PIL, so after rendering the script re-invokes
itself with a system Python (``--caption-only``) to burn English captions.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FK_COLOR = (1.0, 0.45, 0.1, 1.0)
IK_COLOR = (0.15, 0.45, 1.0, 1.0)
ORACLE_COLOR = (0.1, 0.85, 0.3, 1.0)
H0_NORM_COLOR = (0.97, 0.97, 0.97, 1.0)
GT_NORM_COLOR = (1.0, 0.9, 0.1, 1.0)
BODY_COLOR = (0.5, 0.52, 0.56, 1.0)
HIDDEN_PROXIES = ("upperarm_l", "lowerarm_l", "hand_l")

LABEL_TEXT = {
    "known_good": "known_good",
    "dynamic": "dynamic",
    "turning_review": "turning (review)",
    "tracking_loss": "tracking_loss (review)",
    "near_straight": "near_straight",
    "strongly_bent_and_largest_reach_error": "strongly_bent + largest_reach_error (same row)",
    "largest_ratio_error": "largest_ratio_error",
    "largest_direction_error": "largest_direction_error",
}


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path)
    parser.add_argument("--rest", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--caption-python", default="python3")
    parser.add_argument("--caption-only", action="store_true")
    return parser.parse_args(argv)


# ------------------------------------------------------------------ caption


def _caption_lines(label, case, view):
    n = case["numbers"]
    seq = case["sequence_id"].split(":")[1]
    side = "agree" if n["bend_side_agrees"] else "FLIP"
    return [
        f"W62 GT-anchored IK attribution | {LABEL_TEXT[label]} | {view} view",
        f"{case['sequence_id']}  frame #{case['frame_index']}   ({seq})",
        "orange = W60 direction FK   blue = W61 H0 IK   green = oracle IK (E_GT, GT bend side)",
        "thin white = H0 source chain (normalized to L1+L2)   thin yellow = GT source chain (normalized)",
        "yellow wire = E_GT   blue wire = H0 requested endpoint   gray = body (W60 FK)",
        (f"endpoint error vs E_GT (cm):  FK {n['fk_h0_endpoint_error']:.2f}   H0 IK {n['ik_h0_endpoint_error']:.2f}"
         f"   H0dir+GTreach {n['h0_direction_gt_reach_endpoint_error']:.2f}"
         f"   GTdir+H0reach {n['gt_direction_h0_reach_endpoint_error']:.2f}"),
        (f"direction err {n['endpoint_direction_error_degrees']:.1f} deg   bend-plane err "
         f"{n['bend_plane_error_degrees']:.1f} deg ({side})   ratio GT/H0 {n['gt_ratio']:.2f}/{n['h0_ratio']:.2f}"
         f"   reach GT/H0 {n['gt_reach_fraction']:.3f}/{n['h0_reach_fraction']:.3f}"),
    ]


def burn_captions(out_dir: Path, cases: dict) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=17)
    for label, case in cases.items():
        for view in ("side", "front"):
            path = out_dir / f"{label}_{view}.png"
            image = Image.open(path).convert("RGB")
            lines = _caption_lines(label, case, view)
            band = 14 + 24 * len(lines)
            canvas = Image.new("RGB", (image.width, image.height + band), (18, 18, 18))
            canvas.paste(image, (0, 0))
            draw = ImageDraw.Draw(canvas)
            for i, line in enumerate(lines):
                draw.text((12, image.height + 8 + 24 * i), line, fill=(235, 235, 235), font=font)
            canvas.save(path)


# ------------------------------------------------------------------ blender


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
    obj.scale = (1, 1, max(direction.length, 1e-6))
    obj.color = color
    return obj


def _sphere(bpy, name, location, color, radius, wire=False):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=20, ring_count=10)
    obj = bpy.context.object
    obj.name = name
    obj.color = color
    if wire:  # Workbench render ignores display_type, so build real wire geometry.
        modifier = obj.modifiers.new("wire", "WIREFRAME")
        modifier.thickness = 0.0025
        modifier.use_replace = True
        obj.show_in_front = True
    return obj


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
    camera.location = location
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = Matrix((right, true_up, -forward)).transposed().to_quaternion()


def render(options) -> int:
    import bpy
    from mathutils import Quaternion, Vector
    from blender.executor import BlenderExecutor
    from blender.framepose_object_owner_adapter import bind_imported_armature
    from retarget.framepose_target_rest import load_target_rest_pose
    from review_framepose_fk_blender import _make_bone_proxies, _render_setup, _update_bone_proxies

    data = json.loads(options.cases.read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    armature = BlenderExecutor().import_rig(str(options.rig))
    rest = load_target_rest_pose(options.rest)
    if rest.digest() != data["rest_snapshot_digest"]:
        raise ValueError("case export was built against another rest snapshot")
    binding = bind_imported_armature(armature, rest, source_path=str(options.rig))
    _render_setup(bpy, armature)
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.hide_render = True  # body shown as bone proxies only
    proxies = _make_bone_proxies(bpy, armature)
    for name in HIDDEN_PROXIES:
        if name in proxies:
            proxies.pop(name).hide_render = True
    for segment in proxies.values():
        segment.color = BODY_COLOR
    cameras = {"side": _camera(bpy, "W62 side"), "front": _camera(bpy, "W62 front")}
    scene = bpy.context.scene
    options.out_dir.mkdir(parents=True, exist_ok=True)
    alignment = data["canonical_heading_zero_to_armature_xyzw"]
    aq = Quaternion((alignment[3], *alignment[:3]))
    chain = data["target_chain"]
    report = {"schema": "animcv_framepose_ik_gt_attribution_blender_review_v1",
              "blender_version": bpy.app.version_string, "rest_snapshot_digest": rest.digest(),
              "export_sanity_checks_cm": data["sanity_checks_cm"], "refused": data["refused"],
              "cases": {}, "blender_fk_end_vs_export_max_cm": 0.0}

    for label, case in data["cases"].items():
        # Worklog 60 object rotation + all known FK pose-bone locals (body context).
        for pose_bone in armature.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = (1, 0, 0, 0)
            pose_bone.location = (0, 0, 0)
            pose_bone.scale = (1, 1, 1)
        x, y, z, w = case["rig_rotation_armature_space_xyzw"]
        armature.rotation_mode = "QUATERNION"
        armature.rotation_quaternion = Quaternion(binding.rest_rotation_wxyz) @ Quaternion((w, x, y, z))
        for name, (x, y, z, w) in case["pose_bone_local_xyzw"].items():
            armature.pose.bones[name].rotation_quaternion = (w, x, y, z)
        bpy.context.view_layer.update()
        if tuple(armature.location) != binding.location or tuple(armature.scale) != binding.scale:
            raise RuntimeError("review changed object translation or scale")
        _update_bone_proxies(bpy, armature, proxies)
        blender_fk_end = tuple(armature.pose.bones[chain["distal"]].tail)
        fk_err = math.dist(blender_fk_end, case["fk_end"])
        report["blender_fk_end_vs_export_max_cm"] = max(report["blender_fk_end_vs_export_max_cm"], fk_err)

        world = armature.matrix_world.copy()
        to_world = lambda p: world @ Vector(p)  # noqa: E731
        created = []
        root = case["root"]
        for tag, mid, end, color in (("FK", case["fk_mid"], case["fk_end"], FK_COLOR),
                                     ("H0 IK", case["ik_mid"], case["ik_end"], IK_COLOR),
                                     ("oracle", case["oracle_mid"], case["oracle_end"], ORACLE_COLOR)):
            r, m, e = to_world(root), to_world(mid), to_world(end)
            created += [_segment(bpy, f"{tag} upper", r, m, color, 0.009),
                        _segment(bpy, f"{tag} lower", m, e, color, 0.009),
                        _sphere(bpy, f"{tag} elbow", m, color, 0.013),
                        _sphere(bpy, f"{tag} end", e, color, 0.015)]
        for tag, points, color in (("H0 norm", case["h0_normalized_chain"], H0_NORM_COLOR),
                                   ("GT norm", case["gt_normalized_chain"], GT_NORM_COLOR)):
            p = [to_world(v) for v in points]
            thin = [_segment(bpy, f"{tag} upper", p[0], p[1], color, 0.0035),
                    _segment(bpy, f"{tag} lower", p[1], p[2], color, 0.0035),
                    _sphere(bpy, f"{tag} mid", p[1], color, 0.007)]
            for obj in thin:
                obj.show_in_front = True  # drawn over the thick arms they often coincide with
            created += thin
        created.append(_sphere(bpy, "E_GT", to_world(case["e_gt"]), GT_NORM_COLOR, 0.034, wire=True))
        created.append(_sphere(bpy, "H0 requested", to_world(case["h0_requested_endpoint"]),
                               IK_COLOR, 0.026, wire=True))
        created.append(_sphere(bpy, "chain root", to_world(root), (0.9, 0.9, 0.9, 1.0), 0.012))

        object_rotation = world.to_quaternion()
        axis = lambda v: (object_rotation @ (aq @ Vector(v))).normalized()  # noqa: E731
        forward, up, right = axis((0, 1, 0)), axis((0, 0, 1)), axis((1, 0, 0))
        keys = ["root", "fk_mid", "fk_end", "ik_mid", "ik_end", "oracle_mid", "oracle_end", "e_gt"]
        pts = [to_world(case[k]) for k in keys] + [to_world(v) for v in case["gt_normalized_chain"]]
        pts += [to_world(v) for v in case["h0_normalized_chain"]]
        lo = Vector([min(p[i] for p in pts) for i in range(3)])
        hi = Vector([max(p[i] for p in pts) for i in range(3)])
        centre = (lo + hi) / 2
        views = {}
        for view, offset in (("side", -right * 3.0), ("front", forward * 3.0)):
            _aim(cameras[view], centre + offset, centre, up)
            scene.camera = cameras[view]
            path = options.out_dir / f"{label}_{view}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            views[view] = path.name
        report["cases"][label] = {"sequence_id": case["sequence_id"], "frame_index": case["frame_index"],
                                  "png": views, "numbers": case["numbers"],
                                  "blender_fk_end_vs_export_cm": fk_err,
                                  "arm_extent_world": max(hi - lo)}
        for obj in created:
            bpy.data.objects.remove(obj, do_unlink=True)

    (options.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2, sort_keys=True))
    subprocess.run([options.caption_python, str(Path(__file__).resolve()), "--caption-only",
                    "--cases", str(options.cases), "--out-dir", str(options.out_dir)], check=True)
    return 0


def main():
    options = _args()
    if options.caption_only:
        burn_captions(options.out_dir, json.loads(options.cases.read_text())["cases"])
        return 0
    try:
        return render(options)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"W62 Blender review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
