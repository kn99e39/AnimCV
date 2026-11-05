#!/usr/bin/env python3
"""W117 Blender-side timing/interpolation probe (run inside Blender 5.1).  Evidence only.

Keys each state of a states file with the UNCHANGED W116 keying rules (`w116_blender_stages._key_state`, the historical
renderer's rules) and reports: F-curve count, keyed Blender frames, the interpolation of every keyframe point, the scene
frame rate in this evaluation context, and whether evaluating at a sub-frame (f + 0.5) equals evaluating at frame f
(CONSTANT-hold semantics) and differs from frame f + 1 where the stored values differ.

    blender -b -t 2 --python-exit-code 1 -P scripts/w117_blender_timing.py -- --rig ... [--rest ...] --states ... --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", required=True)
    ap.add_argument("--rest", type=Path)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--equivalent-rig-tolerance", type=float, default=1e-4)
    ap.add_argument("--sample-frames", default="")
    o = ap.parse_args(argv)
    try:
        import bpy
        import w116_blender_stages as B
        from render_runtime_heading_rig_blender import _fcurves
        arm, info = B._setup(o)
        bones = B._bones(arm)
        states = json.loads(Path(o.states).read_text())["states"]
        scene = bpy.context.scene
        rep = {"schema": "w117_blender_timing_v1", "blender_version": bpy.app.version_string, "rig": info,
               "scene_fps": scene.render.fps, "scene_fps_base": scene.render.fps_base, "states": {}}
        for name in sorted(states):
            st = states[name]
            act = B._key_state(arm, st, info["rest_rotation_wxyz"], f"W117T {name}")
            fcs = list(_fcurves(act))
            interp = sorted({kp.interpolation for fc in fcs for kp in fc.keyframe_points})
            keyed = sorted({int(round(kp.co[0])) for fc in fcs for kp in fc.keyframe_points})
            n = len(st["frames"])
            sample = [int(x) for x in o.sample_frames.split(",") if x] or [1, max(1, n // 2), max(1, n - 1)]
            sub = {}
            for f in sample:
                if f >= n:
                    continue
                pos = {}
                for tag, (fr, sf) in {"f": (f, 0.0), "f+0.5": (f, 0.5), "f+0.999": (f, 0.999), "f+1": (f + 1, 0.0)}.items():
                    scene.frame_set(fr, subframe=sf)
                    bpy.context.view_layer.update()
                    mw = arm.matrix_world
                    pos[tag] = {b: [list(mw @ arm.pose.bones[b].head), list(mw @ arm.pose.bones[b].tail)] for b in bones}
                sub[str(f)] = {"half_equals_frame": pos["f+0.5"] == pos["f"], "late_equals_frame": pos["f+0.999"] == pos["f"],
                               "next_differs": pos["f+1"] != pos["f"]}
            rep["states"][name] = {"fcurves": len(fcs), "interpolations": interp, "keyed_blender_frames_first_last": [keyed[0], keyed[-1]],
                                   "n_keyed_frames": len(keyed), "state_frames_first_last": [st["frames"][0], st["frames"][-1]],
                                   "blender_frame_equals_index_minus_f0_plus_1": keyed == [fi - st["frames"][0] + 1 for fi in st["frames"]],
                                   "subframe": sub}
        Path(o.out).write_text(json.dumps(rep, indent=1))
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
