#!/usr/bin/env python3
"""W105 Blender-side stages (run inside Blender 5.1; imports nothing that needs OpenCV).

  render  the historical W91 Workbench replay exactly as the W102 wrapper uses it; only the slot tuple is chosen
          (FIXED,RUNTIME = M_Main,R11 candidates; RUNTIME alone = the separate evaluation-only GT reference file)
  audit   opens every saved per-case .blend and records Action names, F-curve channels, keyframe frames,
          interpolation and a value digest per Action, so M_Main and R11 exports can be compared for identity

    blender -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- render --rig ... --rest ... --cases ... --out-dir ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]


def render(o):
    import render_runtime_heading_rig_blender as historical
    historical.MODES = tuple(o.modes.split(","))  # same wrapper pattern as W102; the renderer file is unchanged
    code = historical.render(o)
    if code != 0 or not (o.out_dir/"blender_report.json").exists():
        raise RuntimeError("STOP: historical renderer did not complete")
    return code


def audit(o):
    import bpy
    from render_runtime_heading_rig_blender import _fcurves
    report = {"schema": "w105_action_audit_v1", "blender_version": bpy.app.version_string, "files": {}}
    for blend in sorted(Path(o.render_dir).glob("*/*.blend")):
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        scene = bpy.context.scene
        acts = {}
        for act in bpy.data.actions:
            channels, digest, frames, interp = [], hashlib.sha256(), set(), set()
            for fc in sorted(_fcurves(act), key=lambda f: (f.data_path, f.array_index)):
                pts = [(kp.co[0], kp.co[1]) for kp in fc.keyframe_points]
                channels.append({"data_path": fc.data_path, "index": fc.array_index, "keys": len(pts),
                                 "frames": [round(f, 6) for f, _ in pts], "values": [float(v) for _, v in pts]})
                frames.update(round(f, 6) for f, _ in pts)
                interp.update(kp.interpolation for kp in fc.keyframe_points)
                digest.update(f"{fc.data_path}[{fc.array_index}]".encode())
                for f, v in pts:
                    digest.update(f"{f:.6f}:{v:.9f};".encode())
            acts[act.name] = {"fcurves": len(channels), "channels": channels, "keyed_frames": sorted(frames),
                              "interpolation": sorted(interp), "value_digest": digest.hexdigest()}
        armatures = [ob.name for ob in bpy.data.objects if ob.type == "ARMATURE"]
        report["files"][blend.relative_to(o.render_dir).as_posix()] = {
            "actions": acts, "armatures": armatures, "scene_fps": scene.render.fps,
            "frame_range": [scene.frame_start, scene.frame_end]}
    Path(o.out).write_text(json.dumps(report, indent=1))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("render", "audit"))
    for name in ("rig", "rest", "cases", "out_dir", "render_dir", "out"):
        ap.add_argument("--"+name.replace("_", "-"), type=Path)
    ap.add_argument("--modes", default="FIXED,RUNTIME")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--only", default=None)
    ap.add_argument("--equivalent-rig-tolerance", type=float, default=None)
    ap.add_argument("--show-mesh", action="store_true")
    ap.add_argument("--camera-centre-height", type=float, default=None)
    ap.add_argument("--ortho-scale", type=float, default=2.2)
    o = ap.parse_args(argv)
    if o.mode == "render":
        o.out_dir.mkdir(parents=True, exist_ok=False)
    try:
        return {"render": render, "audit": audit}[o.mode](o)
    except Exception as exc:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        print(f"W105 Blender stage failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
