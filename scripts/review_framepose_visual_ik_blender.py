#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 74): Blender review of the visual-direction IK handoff cases.

Run under Blender with a case JSON from ``scripts/export_framepose_visual_ik_cases.py``.
Each case is an isolated pose: the Worklog 60 object rotation places the gray
rest-pose body (its exported FK locals cover only the hidden left-arm bones); the left-arm candidates are drawn as explicit
proxies from heading-relative armature-space points mapped through the posed
``armature.matrix_world``: blue = P0 (historical H0 IK), orange = P1 (E0
visual direction), magenta = P2 (E1 visual direction), green = GT reference
(oracle IK reaching E_GT), yellow wire = E_GT.  Same cameras and render
settings as the Worklog 62 review (side view = depth x up).

Captions are burned with a system Python (``--caption-only``) because
Blender's bundled Python has no PIL.
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

COLORS = {"P0": (0.15, 0.45, 1.0, 1.0), "P1": (1.0, 0.45, 0.1, 1.0), "P2": (0.9, 0.15, 0.85, 1.0),
          "GT": (0.1, 0.85, 0.3, 1.0)}
E_GT_COLOR = (1.0, 0.9, 0.1, 1.0)
BODY_COLOR = (0.5, 0.52, 0.56, 1.0)
HIDDEN_PROXIES = ("upperarm_l", "lowerarm_l", "hand_l")


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


def _caption_lines(label, case, view):
    n, f = case["numbers"], case["f"]
    side = {k: ("agree" if n[k]["side_agrees_with_gt"] else "FLIP") for k in n}
    return [
        f"W74 visual chain direction -> W61 IK | {label} | {view} view",
        f"{case['sequence_id']}  frame #{case['frame_index']}",
        "blue = P0 H0 IK   orange = P1 E0 visual dir   magenta = P2 E1 visual dir   green = GT (oracle IK)",
        "yellow wire = E_GT (W62 GT-anchored target)   gray = rest pose under W60 object rotation   reach + side = H0",
        (f"wrist error vs E_GT (cm): P0 {n['P0']['solved_endpoint_error']:.2f}   P1 {n['P1']['solved_endpoint_error']:.2f}"
         f"   P2 {n['P2']['solved_endpoint_error']:.2f}   (P3 perfect-depth {n['P3']['solved_endpoint_error']:.2f})"),
        (f"chain direction error (deg): P0 {n['P0']['direction_error_degrees']:.1f}   P1 {n['P1']['direction_error_degrees']:.1f}"
         f"   P2 {n['P2']['direction_error_degrees']:.1f}   elbow (cm): P0 {n['P0']['elbow_error']:.1f}"
         f"  P1 {n['P1']['elbow_error']:.1f}  P2 {n['P2']['elbow_error']:.1f}"),
        (f"chain forward fraction f: H0 {f['h0']:+.2f}  E0 {f['e0']:+.2f}  E1 {f['e1']:+.2f}  GT {f['gt']:+.2f}"
         f"   bend side vs GT: P0 {side['P0']}  P2 {side['P2']}"),
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


def render(options) -> int:
    import bpy
    from mathutils import Quaternion, Vector
    from blender.executor import BlenderExecutor
    from blender.framepose_object_owner_adapter import bind_imported_armature
    from retarget.framepose_target_rest import load_target_rest_pose
    from review_framepose_fk_blender import _make_bone_proxies, _render_setup, _update_bone_proxies
    from review_framepose_ik_gt_attribution_blender import _aim, _camera, _segment, _sphere

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
            obj.hide_render = True
    proxies = _make_bone_proxies(bpy, armature)
    for name in HIDDEN_PROXIES:
        if name in proxies:
            proxies.pop(name).hide_render = True
    for segment in proxies.values():
        segment.color = BODY_COLOR
    cameras = {"side": _camera(bpy, "W74 side"), "front": _camera(bpy, "W74 front")}
    scene = bpy.context.scene
    options.out_dir.mkdir(parents=True, exist_ok=True)
    alignment = data["canonical_heading_zero_to_armature_xyzw"]
    aq = Quaternion((alignment[3], *alignment[:3]))
    chain = data["target_chain"]
    report = {"schema": "animcv_framepose_visual_ik_blender_review_v1", "blender_version": bpy.app.version_string,
              "rest_snapshot_digest": rest.digest(), "export_sanity_checks_cm": data["sanity_checks_cm"],
              "refused": data["refused"], "cases": {}, "blender_fk_end_vs_export_max_cm": 0.0}
    for label, case in data["cases"].items():
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
        fk_err = math.dist(tuple(armature.pose.bones[chain["distal"]].tail), case["fk_end"])
        report["blender_fk_end_vs_export_max_cm"] = max(report["blender_fk_end_vs_export_max_cm"], fk_err)
        world = armature.matrix_world.copy()
        to_world = lambda p: world @ Vector(p)  # noqa: E731
        created = []
        root = case["root"]
        drawn = {**{k: case["chains"][k] for k in ("P0", "P1", "P2")},
                 "GT": {"mid": case["oracle_mid"], "end": case["e_gt"]}}
        for tag, c in drawn.items():
            r, m, e = to_world(root), to_world(c["mid"]), to_world(c["end"])
            radius = 0.011 if tag == "GT" else 0.008
            created += [_segment(bpy, f"{tag} upper", r, m, COLORS[tag], radius),
                        _segment(bpy, f"{tag} lower", m, e, COLORS[tag], radius),
                        _sphere(bpy, f"{tag} elbow", m, COLORS[tag], 0.013),
                        _sphere(bpy, f"{tag} end", e, COLORS[tag], 0.015)]
        created.append(_sphere(bpy, "E_GT", to_world(case["e_gt"]), E_GT_COLOR, 0.034, wire=True))
        created.append(_sphere(bpy, "chain root", to_world(root), (0.9, 0.9, 0.9, 1.0), 0.012))
        object_rotation = world.to_quaternion()
        axis = lambda v: (object_rotation @ (aq @ Vector(v))).normalized()  # noqa: E731
        forward, up, right = axis((0, 1, 0)), axis((0, 0, 1)), axis((1, 0, 0))
        pts = [to_world(root), to_world(case["e_gt"]), to_world(case["oracle_mid"])]
        pts += [to_world(c[k]) for c in case["chains"].values() for k in ("mid", "end")]
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
                                  "png": views, "numbers": case["numbers"], "f": case["f"],
                                  "blender_fk_end_vs_export_cm": fk_err}
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
        print(f"W74 Blender review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
