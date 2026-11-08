#!/usr/bin/env python3
"""EXPERIMENTAL (opt-in) Blender-side stage of the video -> editable Action -> WORLD_POSITION_PIN slice (Batch 124).

Runs ONLY inside Blender's own Python (``blender -b --python-exit-code 1 -P <this file> -- <mode> ...``).  It is never
imported by the default CLI/GUI path.  It REUSES, unchanged, the frozen W116 Blender helpers (rig import through
``BlenderExecutor.import_rig`` + the historical renderer's rest-equivalence binding, CONSTANT keying rules, F-curve audit)
and the existing FBX exporter ``blender.export.export_fbx``.  It adds no exporter/importer/keying policy of its own.

Modes
  build_blend   import the known BaseRig, key ONE Animation State as ONE Action (W116 keying rules), set the scene to the
                declared fps / Action frame range, attach provenance custom properties, save a NEW .blend (refuses to
                overwrite) and report the in-process evaluation.
  readback      open a saved .blend in a fresh process and report, from that file only: the assigned Action, every F-curve
                (data path, key frames, interpolation), scene timebase, custom properties, and the evaluated world
                head/tail of every non-finger bone at every integer Action frame and at declared subframes.
  export_fbx    open a saved .blend and call the existing ``blender.export.export_fbx`` unchanged.
  fbx_readback  clean factory scene, import an FBX with the existing ``BlenderExecutor.import_rig`` (default importer
                flags) and report the same readback as ``readback``.
  render        open a saved .blend and render declared Action frames (Workbench) for visual review only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
FINGER = ("index", "middle", "pinky", "ring", "thumb")
PROP_KEY = "animcv_experimental_provenance"


def _bones(arm):
    return sorted(b.name for b in arm.pose.bones if not any(f in b.name for f in FINGER))


def _rest(arm, bones):
    out = {}
    for n in bones:
        b = arm.data.bones[n]
        out[n] = {"parent": b.parent.name if b.parent else None, "matrix_local": [list(r) for r in b.matrix_local],
                  "head_local": list(b.head_local), "tail_local": list(b.tail_local), "length": float(b.length)}
    return out


def _armature():
    import bpy
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"expected exactly one armature object, found {len(arms)}")
    return arms[0]


def _fcurves(act):
    from render_runtime_heading_rig_blender import _fcurves as frozen
    return frozen(act)


def _audit(act, frozen=True):
    """Frozen W116 audit digest (same formula) + an interpolation / key census.  ``frozen=False`` (FBX re-imports, whose
    F-curves include location/Euler/scale paths the W116 audit does not parse) uses the same digest formula over every
    F-curve and returns no per-channel quaternion store."""
    if frozen:
        import w116_blender_stages as W
        dig, stored = W._audit(act)
    else:
        h = hashlib.sha256()
        for fc in sorted(_fcurves(act), key=lambda f: (f.data_path, f.array_index)):
            h.update(f"{fc.data_path}[{fc.array_index}]".encode())
            for kp in fc.keyframe_points:
                h.update(f"{kp.co[0]:.6f}:{kp.co[1]:.9f};".encode())
        dig, stored = h.hexdigest(), None
    curves = []
    for fc in sorted(_fcurves(act), key=lambda f: (f.data_path, f.array_index)):
        kp = list(fc.keyframe_points)
        curves.append({"data_path": fc.data_path, "index": fc.array_index, "keys": len(kp),
                       "first": float(kp[0].co[0]) if kp else None, "last": float(kp[-1].co[0]) if kp else None,
                       "interpolation": sorted({k.interpolation for k in kp}),
                       "non_integer_key_times": sum(1 for k in kp if abs(k.co[0] - round(k.co[0])) > 1e-6)})
    paths = {}
    for c in curves:
        paths[c["data_path"].split(".")[-1]] = paths.get(c["data_path"].split(".")[-1], 0) + 1
    census = {}
    for c in curves:
        for i in c["interpolation"]:
            census[i] = census.get(i, 0) + 1
    return dig, stored, {"fcurves": curves, "interpolation_census_curves": census, "n_fcurves": len(curves), "property_census": paths,
                         "key_frame_min": min((c["first"] for c in curves if c["first"] is not None), default=None),
                         "key_frame_max": max((c["last"] for c in curves if c["last"] is not None), default=None)}


def _eval(arm, bones, f, sub=0.0, stored=None):
    import bpy
    scene = bpy.context.scene
    scene.frame_set(int(f), subframe=float(sub))
    bpy.context.view_layer.update()
    mw = arm.matrix_world
    rec = {"blender_frame": int(f), "subframe": float(sub),
           "world": {b: {"head": list(mw @ arm.pose.bones[b].head), "tail": list(mw @ arm.pose.bones[b].tail)} for b in bones},
           "matrix_world": [list(r) for r in mw]}
    if stored is not None and sub == 0.0:
        rec["stored_wxyz"] = {ch: [stored[ch][k][int(f)] for k in range(4)] for ch in stored if int(f) in stored[ch][0]}
    return rec


def _scene_info():
    import bpy
    s = bpy.context.scene
    return {"fps": s.render.fps, "fps_base": s.render.fps_base, "frame_start": s.frame_start, "frame_end": s.frame_end,
            "effective_fps": s.render.fps / s.render.fps_base}


def _report_readback(arm, act, subframes, kind, extra=None, frozen=True):
    import bpy
    bones = _bones(arm)
    dig, stored, census = _audit(act, frozen)
    n = int(round(census["key_frame_max"])) if census["key_frame_max"] is not None else 0
    first = int(round(census["key_frame_min"])) if census["key_frame_min"] is not None else 1
    frames = [_eval(arm, bones, f, 0.0, stored) for f in range(first, n + 1)]
    subs = [_eval(arm, bones, f, s) for f in range(first, n) for s in subframes]
    rep = {"schema": "animcv_exp_action_readback_v1", "kind": kind, "blender_version": bpy.app.version_string,
           "armature_object": arm.name, "armature_location": list(arm.location), "armature_scale": list(arm.scale),
           "rotation_mode": arm.rotation_mode, "bones": bones, "rest": _rest(arm, bones), "scene": _scene_info(),
           "actions_in_file": sorted(a.name for a in bpy.data.actions), "action_name": act.name, "value_digest": dig,
           "fcurve_census": census, "frames": frames, "subframes": subs, "subframe_offsets": list(subframes),
           "pose_bone_rotation_modes": sorted({pb.rotation_mode for pb in arm.pose.bones}),
           "ownership": {"action_assigned_to_object": arm.name, "object_type": arm.type,
                         "action_slot": getattr(getattr(arm.animation_data, "action_slot", None), "identifier", None),
                         "objects_with_animation": sorted(o.name for o in bpy.data.objects if o.animation_data and o.animation_data.action),
                         "object_level_paths": sorted({c["data_path"] for c in census["fcurves"] if not c["data_path"].startswith("pose.")}),
                         "pose_bone_paths": len({c["data_path"] for c in census["fcurves"] if c["data_path"].startswith("pose.")})}}
    if extra:
        rep.update(extra)
    return rep


def build_blend(o):
    import bpy
    import w116_blender_stages as W
    out_blend = Path(o.out_blend)
    if out_blend.exists():
        raise RuntimeError(f"refusing to overwrite {out_blend}")
    arm, info = W._setup_w105(o.rig, o.rest, 1e-4)
    states = json.loads(Path(o.states).read_text())["states"]
    st = states[o.state_name]
    act = W._key_state(arm, st, info["rest_rotation_wxyz"], o.action_name)
    n = len(st["frames"])
    s = bpy.context.scene
    s.render.fps = int(o.fps)
    s.render.fps_base = 1.0
    s.frame_start, s.frame_end = 1, n
    prov = json.loads(Path(o.provenance).read_text()) if o.provenance else {}
    arm[PROP_KEY] = json.dumps(prov, sort_keys=True)
    act[PROP_KEY] = json.dumps(prov, sort_keys=True)
    dig, stored, census = _audit(act)
    bones = _bones(arm)
    in_process = [_eval(arm, bones, f, 0.0, stored) for f in range(1, n + 1)]
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    rep = {"schema": "animcv_exp_build_blend_v1", "blender_version": bpy.app.version_string, "rig": info, "state_id": st.get("state_id"),
           "action_name": act.name, "value_digest": dig, "fcurve_census": census, "scene": _scene_info(), "bones": bones,
           "rest": _rest(arm, bones), "in_process_frames": in_process,
           "blend_sha256": hashlib.sha256(out_blend.read_bytes()).hexdigest()}
    Path(o.report).write_text(json.dumps(rep))
    return 0


def readback(o):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(o.blend))
    arm = _armature()
    act = arm.animation_data.action if arm.animation_data else None
    if act is None:
        raise RuntimeError("no Action assigned to the armature")
    prov = arm.get(PROP_KEY)
    rep = _report_readback(arm, act, [float(x) for x in o.subframes.split(",") if x], "blend",
                           {"provenance_object": json.loads(prov) if prov else None,
                            "provenance_action": json.loads(act.get(PROP_KEY)) if act.get(PROP_KEY) else None,
                            "blend_sha256": hashlib.sha256(Path(o.blend).read_bytes()).hexdigest()})
    Path(o.report).write_text(json.dumps(rep))
    return 0


def export_fbx(o):
    import bpy
    from blender.export import export_fbx as existing_export
    if Path(o.fbx).exists():
        raise RuntimeError(f"refusing to overwrite {o.fbx}")
    bpy.ops.wm.open_mainfile(filepath=str(o.blend))
    existing_export(str(o.fbx))
    Path(o.report).write_text(json.dumps({"schema": "animcv_exp_fbx_export_v1", "blender_version": bpy.app.version_string,
                                          "exporter": "src/blender/export.py export_fbx (unchanged)", "fbx": str(o.fbx),
                                          "exists": Path(o.fbx).exists(),
                                          "fbx_sha256": hashlib.sha256(Path(o.fbx).read_bytes()).hexdigest() if Path(o.fbx).exists() else None,
                                          "fbx_bytes": Path(o.fbx).stat().st_size if Path(o.fbx).exists() else None}))
    return 0


def fbx_readback(o):
    import bpy
    from blender.executor import BlenderExecutor
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm = BlenderExecutor().import_rig(str(o.fbx))
    act = arm.animation_data.action if arm.animation_data else None
    if act is None:
        raise RuntimeError("FBX import produced no assigned Action")
    rep = _report_readback(arm, act, [float(x) for x in o.subframes.split(",") if x], "fbx_reimport",
                           {"fbx_sha256": hashlib.sha256(Path(o.fbx).read_bytes()).hexdigest(),
                            "importer": "BlenderExecutor.import_rig -> bpy.ops.import_scene.fbx (default flags, unchanged)"},
                           frozen=False)
    Path(o.report).write_text(json.dumps(rep))
    return 0


def render(o):
    import bpy
    from mathutils import Vector
    from review_framepose_fk_blender import _render_setup
    from review_framepose_ik_gt_attribution_blender import _aim, _sphere
    bpy.ops.wm.open_mainfile(filepath=str(o.blend))
    arm = _armature()
    jobs = json.loads(Path(o.jobs).read_text())
    _render_setup(bpy, arm)
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = int(jobs["size"])
    cam = scene.camera
    cam.data.type = "ORTHO"
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    done, markers = [], []
    for view in jobs["views"]:
        cam.data.ortho_scale = float(view["ortho_scale"])
        centre = Vector(view["centre"])
        _aim(cam, centre + Vector(view["offset"]), centre, Vector((0, 0, 1)))
        for job in jobs["frames"]:
            scene.frame_set(int(job["blender_frame"]))
            bpy.context.view_layer.update()
            for m in markers:
                bpy.data.objects.remove(m, do_unlink=True)
            markers = [_sphere(bpy, mk["name"], tuple(mk["position"]), tuple(mk["color"]), mk.get("radius", 0.015))
                       for mk in job.get("markers", [])]
            scene.render.filepath = str(out / f"{view['name']}_f{int(job['blender_frame']):04d}.png")
            bpy.ops.render.render(write_still=True)
            done.append(scene.render.filepath)
    (out / "render_report.json").write_text(json.dumps({"schema": "animcv_exp_render_v1", "blender_version": bpy.app.version_string,
                                                        "blend": str(o.blend), "rendered": len(done)}))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("build_blend", "readback", "export_fbx", "fbx_readback", "render"))
    for name in ("rig", "rest", "states", "state_name", "action_name", "out_blend", "provenance", "report", "blend", "fbx", "jobs",
                 "out_dir"):
        ap.add_argument("--" + name.replace("_", "-"))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--subframes", default="0.25,0.5,0.75")
    o = ap.parse_args(argv)
    try:
        return {"build_blend": build_blend, "readback": readback, "export_fbx": export_fbx, "fbx_readback": fbx_readback,
                "render": render}[o.mode](o)
    except Exception:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
