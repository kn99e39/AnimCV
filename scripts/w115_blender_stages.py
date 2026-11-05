#!/usr/bin/env python3
"""W115 Blender-side stages (run inside Blender 5.1; evidence only, no renderer/exporter/rig change).

  restq   imports the W105 rig exactly as the historical renderer does (BlenderExecutor.import_rig + the renderer's own
          `_bind_equivalent`) and records the float32 armature rest rotation `binding.rest_rotation_wxyz` that the
          renderer composes with every Object key (armature.rotation_quaternion = rest_q @ objq)
  posed   opens every saved per-case .blend of a render directory and, for every Action of the case and every keyed
          frame, evaluates the armature (scene.frame_set) and records
            - the evaluated Object / pose-bone rotation_quaternion of the keyed channels (to prove the evaluated Action is
              the audited one), and
            - the world-space posed head of every non-finger bone plus the neck_02 tail
          as the exact float values Blender produced (mathutils float32 widened to Python float)

    blender -b -t 2 --python-exit-code 1 -P scripts/w115_blender_stages.py -- restq --rig ... --rest ... --out ...
    blender -b -t 2 --python-exit-code 1 -P scripts/w115_blender_stages.py -- posed --render-dir ... --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
FINGER = ("index", "middle", "pinky", "ring", "thumb")


def restq(o):
    import bpy
    from blender.executor import BlenderExecutor
    from retarget.framepose_target_rest import load_target_rest_pose
    import render_runtime_heading_rig_blender as historical
    bpy.ops.wm.read_factory_settings(use_empty=True)
    armature = BlenderExecutor().import_rig(str(o.rig))
    rest = load_target_rest_pose(o.rest)
    binding, worst = historical._bind_equivalent(armature, rest, o.equivalent_rig_tolerance, str(o.rig))
    rep = {"schema": "w115_blender_restq_v1", "blender_version": bpy.app.version_string,
           "rest_rotation_wxyz": list(binding.rest_rotation_wxyz), "location": list(binding.location),
           "scale": list(binding.scale), "equivalent_rig_max_rest_difference": worst, "rest_snapshot_digest": rest.digest()}
    Path(o.out).write_text(json.dumps(rep, indent=1))
    return 0


def posed(o):
    import bpy
    from render_runtime_heading_rig_blender import _fcurves
    report = {"schema": "w115_blender_posed_v1", "blender_version": bpy.app.version_string, "files": {}}
    for blend in sorted(Path(o.render_dir).glob("*/*.blend")):
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        scene = bpy.context.scene
        arm = next(ob for ob in bpy.data.objects if ob.type == "ARMATURE")
        bones = sorted(b.name for b in arm.pose.bones if not any(f in b.name for f in FINGER))
        acts = {}
        for act in sorted(bpy.data.actions, key=lambda a: a.name):
            arm.animation_data.action = act
            if hasattr(arm.animation_data, "action_slot") and arm.animation_data.action_slot is None and len(act.slots):
                arm.animation_data.action_slot = act.slots[0]
            curves = sorted(_fcurves(act), key=lambda f: (f.data_path, f.array_index))
            keyed = sorted({int(round(kp.co[0])) for fc in curves for kp in fc.keyframe_points})
            frames = []
            for f in keyed:
                scene.frame_set(f)
                bpy.context.view_layer.update()
                mw = arm.matrix_world
                pos = {b: list(mw @ arm.pose.bones[b].head) for b in bones}
                pos["neck_02.tail"] = list(mw @ arm.pose.bones["neck_02"].tail)
                rot = {"__object__": list(arm.rotation_quaternion)}
                for fc in curves:
                    if fc.data_path.startswith("pose.bones"):
                        b = fc.data_path.split('"')[1]
                        rot[b] = list(arm.pose.bones[b].rotation_quaternion)
                frames.append({"frame": f, "world_head": pos, "evaluated_rotation_wxyz": rot})
            acts[act.name] = {"keyed_frames": keyed, "frames": frames}
        report["files"][blend.relative_to(o.render_dir).as_posix()] = {"armature": arm.name, "bones": bones, "actions": acts,
                                                                        "armature_scale": list(arm.scale)}
    Path(o.out).write_text(json.dumps(report))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("restq", "posed"))
    for name in ("rig", "rest", "render_dir", "out"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path)
    ap.add_argument("--equivalent-rig-tolerance", type=float, default=1e-4)
    o = ap.parse_args(argv)
    try:
        return {"restq": restq, "posed": posed}[o.mode](o)
    except Exception:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
