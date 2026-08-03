"""Inspect a generated .blend and write machine-readable animation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def _fcurves(action):
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                for channelbag in strip.channelbags:
                    yield from channelbag.fcurves
    else:
        yield from action.fcurves


def main():
    args = _args()
    bpy.ops.wm.open_mainfile(filepath=args.blend)
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one armature, found {len(armatures)}")
    armature = armatures[0]
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        raise RuntimeError("armature has no animation action")
    curves = list(_fcurves(action))
    keyframes = sum(len(curve.keyframe_points) for curve in curves)
    report = {
        "passed": bool(curves and keyframes),
        "armature_name": armature.name,
        "action_name": action.name,
        "frame_range": [bpy.context.scene.frame_start, bpy.context.scene.frame_end],
        "fps": bpy.context.scene.render.fps / bpy.context.scene.render.fps_base,
        "fcurve_count": len(curves),
        "keyframe_count": keyframes,
        "animated_bones": sorted({curve.data_path.split('pose.bones["', 1)[1].split('"]', 1)[0]
                                  for curve in curves if 'pose.bones["' in curve.data_path}),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[motion-tool] Blender animation audit {'passed' if report['passed'] else 'failed'} -> {args.out}")


if __name__ == "__main__":
    main()
