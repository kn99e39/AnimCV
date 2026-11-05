#!/usr/bin/env python3
"""W116 Blender-side stages (run inside Blender 5.1). Evidence only: no renderer/exporter/rig/src change.

This file is the INDEPENDENT OUTPUT EVALUATOR side of W116. It never imports the W116 resolver; it only keys the
Animation State values it is given into a fresh Action on the rig (with the historical renderer's keying rules) and reports
what Blender actually evaluates.

  probe     W105 rig imported exactly as the historical renderer does (BlenderExecutor.import_rig + the renderer's own
            `_bind_equivalent`, tolerance 1e-4).  Records the binding, the actual imported rest geometry, the parent
            state's Action (audit value digest, same formula as scripts/w105_rig_blender.py audit), world positions
            of every non-finger bone head/tail on every keyed frame, and coordinate probes on the first frame
            (Armature Object translation, parent/local bone rotation, scale).
  evaluate  key every state of a states file as its own Action and record, per keyed frame, the evaluated world
            head/tail of every non-finger bone, the Armature Object matrix_world and the stored F-curve values.
            `--rig synthetic:<spec.json>` builds the declared synthetic armature instead of importing the W105 FBX.
  render    Workbench renders of (state, frame) pairs with target markers, for visual review only.

    blender -b -t 2 --python-exit-code 1 -P scripts/w116_blender_stages.py -- probe --rig ... --rest ... --states ... --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
FINGER = ("index", "middle", "pinky", "ring", "thumb")


def _setup_w105(rig_path, rest_path, tol):
    import bpy
    from blender.executor import BlenderExecutor
    from retarget.framepose_target_rest import load_target_rest_pose
    import render_runtime_heading_rig_blender as historical
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm = BlenderExecutor().import_rig(str(rig_path))
    rest = load_target_rest_pose(rest_path)
    binding, worst = historical._bind_equivalent(arm, rest, tol, str(rig_path))
    info = {"rig_kind": "w105_fbx", "rig_path": str(rig_path), "rest_snapshot_digest": rest.digest(),
            "rest_rotation_wxyz": list(binding.rest_rotation_wxyz), "location": list(binding.location),
            "scale": list(binding.scale), "equivalent_rig_max_rest_difference": worst}
    return arm, info


def _setup_synthetic(spec_path):
    import bpy
    spec = json.loads(Path(spec_path).read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    data = bpy.data.armatures.new(spec["rig_id"])
    arm = bpy.data.objects.new(spec["rig_id"], data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for b in spec["bones"]:
        eb = data.edit_bones.new(b["name"])
        eb.head, eb.tail, eb.roll = b["head"], b["tail"], b.get("roll", 0.0)
        if b.get("parent"):
            eb.parent = data.edit_bones[b["parent"]]
            eb.use_connect = bool(b.get("connect", False))
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.location = spec["object"]["location"]
    arm.scale = spec["object"]["scale"]
    arm.rotation_mode = "QUATERNION"
    arm.rotation_quaternion = spec["object"]["rest_rotation_wxyz"]
    bpy.context.view_layer.update()
    info = {"rig_kind": "synthetic", "rig_path": str(spec_path), "spec_sha256": hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
            "rest_rotation_wxyz": list(spec["object"]["rest_rotation_wxyz"]), "location": list(spec["object"]["location"]),
            "scale": list(spec["object"]["scale"])}
    return arm, info


def _setup(o):
    if str(o.rig).startswith("synthetic:"):
        return _setup_synthetic(str(o.rig)[len("synthetic:"):])
    return _setup_w105(o.rig, o.rest, o.equivalent_rig_tolerance)


def _bones(arm):
    return sorted(b.name for b in arm.pose.bones if not any(f in b.name for f in FINGER))


def _rest(arm, bones):
    out = {}
    for n in bones:
        b = arm.data.bones[n]
        out[n] = {"parent": b.parent.name if b.parent else None, "matrix_local": [list(r) for r in b.matrix_local],
                  "head_local": list(b.head_local), "tail_local": list(b.tail_local), "length": float(b.length)}
    return out


def _key_state(arm, state, rest_q, name):
    """Historical renderer keying rules: armature.rotation_quaternion = rest_q @ objq, pose bones (w,x,y,z), Blender
    frame = frame_index - f0 + 1, CONSTANT interpolation.  Channels are keyed in the state's channel order."""
    import bpy
    from mathutils import Quaternion
    from render_runtime_heading_rig_blender import _fcurves
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    arm.animation_data_create()
    arm.animation_data.action = act
    arm.rotation_mode = "QUATERNION"
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
    frames = state["frames"]
    f0 = frames[0]
    rq = Quaternion(rest_q)
    for i, fi in enumerate(frames):
        f = fi - f0 + 1
        for ch in state["channels"]:
            x, y, z, w = state["values"][ch][i]
            if ch == "__object__":
                arm.rotation_quaternion = rq @ Quaternion((w, x, y, z))
                arm.keyframe_insert("rotation_quaternion", frame=f)
            else:
                arm.pose.bones[ch].rotation_quaternion = (w, x, y, z)
                arm.pose.bones[ch].keyframe_insert("rotation_quaternion", frame=f)
    for fc in _fcurves(act):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"
    return act


def _audit(act):
    from render_runtime_heading_rig_blender import _fcurves
    digest, stored = hashlib.sha256(), {}
    for fc in sorted(_fcurves(act), key=lambda f: (f.data_path, f.array_index)):
        pts = [(kp.co[0], kp.co[1]) for kp in fc.keyframe_points]
        digest.update(f"{fc.data_path}[{fc.array_index}]".encode())
        for f, v in pts:
            digest.update(f"{f:.6f}:{v:.9f};".encode())
        ch = "__object__" if fc.data_path == "rotation_quaternion" else fc.data_path.split('"')[1]
        stored.setdefault(ch, {})[fc.array_index] = {int(round(f)): float(v) for f, v in pts}
    return digest.hexdigest(), stored


def _eval_frames(arm, state, bones, stored, location=None):
    import bpy
    scene = bpy.context.scene
    if location is not None:
        arm.location = location
    frames, f0 = state["frames"], state["frames"][0]
    out = []
    for fi in frames:
        f = fi - f0 + 1
        scene.frame_set(f)
        bpy.context.view_layer.update()
        mw = arm.matrix_world
        world = {b: {"head": list(mw @ arm.pose.bones[b].head), "tail": list(mw @ arm.pose.bones[b].tail)} for b in bones}
        st = {ch: [stored[ch][k][f] for k in range(4)] for ch in stored}  # stored F-curve value, wxyz index order
        out.append({"frame_index": fi, "blender_frame": f, "world": world, "matrix_world": [list(r) for r in mw],
                    "stored_wxyz": st})
    return out


def evaluate(o):
    import bpy
    arm, info = _setup(o)
    bones = _bones(arm)
    states = json.loads(Path(o.states).read_text())["states"]
    loc0 = tuple(arm.location)
    rep = {"schema": "w116_blender_evaluate_v1", "blender_version": bpy.app.version_string, "rig": info, "bones": bones,
           "rest": _rest(arm, bones), "states_file_sha256": hashlib.sha256(Path(o.states).read_bytes()).hexdigest(), "states": {}}
    for name in sorted(states):
        st = states[name]
        act = _key_state(arm, st, info["rest_rotation_wxyz"], f"W116 {name}")
        dig, stored = _audit(act)
        frames = _eval_frames(arm, st, bones, stored, st.get("armature_location"))
        arm.location = loc0
        rep["states"][name] = {"state_id": st.get("state_id"), "value_digest": dig, "fcurves": sum(len(v) for v in stored.values()),
                               "frames": frames}
    Path(o.out).write_text(json.dumps(rep))
    return 0


def probe(o):
    import bpy
    from mathutils import Quaternion, Vector
    arm, info = _setup(o)
    bones = _bones(arm)
    states = json.loads(Path(o.states).read_text())["states"]
    rep = {"schema": "w116_blender_probe_v1", "blender_version": bpy.app.version_string, "rig": info, "bones": bones,
           "rest": _rest(arm, bones), "states": {}}
    for name in sorted(states):
        st = states[name]
        act = _key_state(arm, st, info["rest_rotation_wxyz"], f"W116 {name}")
        dig, stored = _audit(act)
        rep["states"][name] = {"state_id": st.get("state_id"), "value_digest": dig, "fcurves": sum(len(v) for v in stored.values()),
                               "frames": _eval_frames(arm, st, bones, stored)}
    # coordinate probes on the first frame of the first state, Action detached, values set directly
    st = states[sorted(states)[0]]
    arm.animation_data.action = None
    rq = Quaternion(info["rest_rotation_wxyz"])

    def pose(mod=None):
        for ch in st["channels"]:
            x, y, z, w = st["values"][ch][0]
            q = Quaternion((w, x, y, z))
            if mod and ch in mod:
                q = q @ mod[ch]
            if ch == "__object__":
                arm.rotation_quaternion = rq @ q
            else:
                arm.pose.bones[ch].rotation_quaternion = q
        bpy.context.view_layer.update()
        mw = arm.matrix_world
        return {b: {"head": list(mw @ arm.pose.bones[b].head), "tail": list(mw @ arm.pose.bones[b].tail)} for b in bones}

    probes = {"base": pose()}
    loc0 = Vector(arm.location)
    arm.location = loc0 + Vector(o.probe_translation)
    probes["object_translated"] = pose()
    arm.location = loc0
    for ch in o.probe_channels.split(","):
        if ch in st["channels"]:
            probes[f"local_rot:{ch}"] = pose({ch: Quaternion((1, 0, 0), 0.1745329252)})
    probes["base_again"] = pose()
    rep["coordinate_probes"] = {"translation": list(o.probe_translation), "rotation_rad": 0.1745329252, "rotation_axis_local": [1, 0, 0],
                                "positions": probes, "armature_scale": list(arm.scale)}
    Path(o.out).write_text(json.dumps(rep))
    return 0


def render(o):
    import bpy
    from mathutils import Vector
    from review_framepose_fk_blender import _render_setup
    from review_framepose_ik_gt_attribution_blender import _aim, _sphere
    arm, info = _setup(o)
    states = json.loads(Path(o.states).read_text())["states"]
    jobs = json.loads(Path(o.jobs).read_text())["jobs"]
    _render_setup(bpy, arm)
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = o.size
    scene.render.fps = 30
    cam = scene.camera
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = o.ortho_scale
    shown = False
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.parent == arm:
            obj.hide_render = False
            shown = True
    if not shown:  # synthetic rig: bone proxies
        from review_framepose_fk_blender import _make_bone_proxies
        prox = {}
        for b in arm.pose.bones:
            bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.012, depth=1)
            prox[b.name] = bpy.context.object
            prox[b.name].color = (0.75, 0.81, 0.92, 1.0)
    else:
        prox = {}
    markers = []
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    acts = {}
    done = []
    for job in jobs:
        st = states[job["state"]]
        if job["state"] not in acts:
            acts[job["state"]] = _key_state(arm, st, info["rest_rotation_wxyz"], f"W116R {job['state']}")
        arm.animation_data.action = acts[job["state"]]
        f = job["frame_index"] - st["frames"][0] + 1
        scene.frame_set(f)
        bpy.context.view_layer.update()
        mw = arm.matrix_world
        for name, seg in prox.items():
            pb = arm.pose.bones[name]
            a, b = mw @ pb.head, mw @ pb.tail
            d = b - a
            seg.location = (a + b) / 2
            seg.rotation_mode = "QUATERNION"
            seg.rotation_quaternion = d.to_track_quat("Z", "Y")
            seg.scale = (1, 1, max(d.length, 1e-6))
        for m in markers:
            bpy.data.objects.remove(m, do_unlink=True)
        markers = []
        for mk in job.get("markers", []):
            markers.append(_sphere(bpy, mk["name"], tuple(mk["position"]), tuple(mk["color"]), mk.get("radius", 0.012)))
        centre = Vector(job["centre"])
        _aim(cam, centre + Vector(job["view_offset"]), centre, Vector((0, 0, 1)))
        scene.render.filepath = str(out / f"{job['id']}.png")
        bpy.ops.render.render(write_still=True)
        done.append(job["id"])
    (out / "render_report.json").write_text(json.dumps({"schema": "w116_render_v1", "blender_version": bpy.app.version_string,
                                                        "rig": info, "rendered": done}, indent=1))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("probe", "evaluate", "render"))
    ap.add_argument("--rig", required=True)
    for name in ("rest", "states", "out", "jobs", "out_dir"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path)
    ap.add_argument("--equivalent-rig-tolerance", type=float, default=1e-4)
    ap.add_argument("--probe-translation", type=float, nargs=3, default=(0.1, 0.0, 0.0))
    ap.add_argument("--probe-channels", default="spine_01,upperarm_l,lowerarm_l")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--ortho-scale", type=float, default=1.3)
    o = ap.parse_args(argv)
    try:
        return {"probe": probe, "evaluate": evaluate, "render": render}[o.mode](o)
    except Exception:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
