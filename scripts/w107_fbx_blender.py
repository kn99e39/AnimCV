#!/usr/bin/env python3
"""W107 Blender-side stages (run inside Blender 5.1).  Scratch copies only; no original file is ever written.

  options  record the runtime RNA properties (identifier, type, default) of ``export_scene.fbx`` and
           ``import_scene.fbx`` plus the FBX add-on version, *before* the declaration is frozen.
  probe    for every declared case (three W106 real M_Main Actions + the W106 synthetic armature): sample the Blender
           source, then for V0 (unchanged production ``export_fbx``), V0d (same arguments through a diagnostic
           operator call), V1 and V2 (one declared property changed each) export a scratch FBX, import it with default
           importer settings into a factory-empty scene and sample it at the identical integer/half-frame times.
           V3 builds the synthetic world-equivalent 2x2 (object static/animated x scale 1/0.01) and runs V0 on each.

    blender -b --factory-startup -t 2 --python-exit-code 1 -P scripts/w107_fbx_blender.py -- probe --declaration ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import w107_fbx_contract as C  # noqa: E402
# unchanged W106 measurement helpers (read-only reuse keeps V0 sampling identical to W106)
from w106_action_blender import (active_info, armature_obj, assign, channels_of, open_case, sample,  # noqa: E402
                                 scene_timing)
from w106_action_contract import channel_digest, sample_times  # noqa: E402

sha256 = C.sha256_file


# --- stage: options -----------------------------------------------------------------------------------------------
def _prop_default(p):
    if p.type == "ENUM":
        return sorted(p.default_flag) if p.is_enum_flag else p.default
    if getattr(p, "is_array", False) or getattr(p, "array_length", 0) > 0:
        return list(p.default_array)
    if p.type in ("BOOLEAN", "INT", "FLOAT", "STRING"):
        return p.default
    return None


def _op_props(op):
    rna = op.get_rna_type()
    out = {}
    for p in rna.properties:
        if p.identifier == "rna_type":
            continue
        rec = {"type": p.type, "default": _prop_default(p)}
        if p.type == "ENUM":
            rec["items"] = [i.identifier for i in p.enum_items]
        if p.type in ("FLOAT", "INT"):
            rec["range"] = [p.hard_min, p.hard_max]
        rec["description"] = p.description
        out[p.identifier] = rec
    return out


def options(o):
    import bpy
    import addon_utils
    mods = {m.__name__: m for m in addon_utils.modules()}
    fbx_mod = mods.get("io_scene_fbx")
    rep = {"schema": C.SCHEMA + ":options", "blender_version": bpy.app.version_string,
           "blender_file_version": list(bpy.app.version_file), "blender_build_hash": bpy.app.build_hash.decode()
           if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash),
           "fbx_addon": {"module": "io_scene_fbx", "found": fbx_mod is not None,
                         "version": list(getattr(fbx_mod, "bl_info", {}).get("version", ())) if fbx_mod else None,
                         "file": getattr(fbx_mod, "__file__", None) if fbx_mod else None,
                         "enabled": addon_utils.check("io_scene_fbx")[1]},
           "export_scene.fbx": _op_props(bpy.ops.export_scene.fbx),
           "import_scene.fbx": _op_props(bpy.ops.import_scene.fbx)}
    if fbx_mod and fbx_mod.__file__:
        d = Path(fbx_mod.__file__).parent
        rep["fbx_addon"]["source_sha256"] = {p.name: sha256(p) for p in sorted(d.glob("*.py"))}
    Path(o.out).write_text(json.dumps(rep, indent=1))
    return 0


# --- FBX file structure (export semantics, independent of the importer) --------------------------------------------
def fbx_structure(path):
    try:
        from io_scene_fbx import parse_fbx
        root, version = parse_fbx.parse(str(path))
    except Exception as exc:  # recorded; never retried
        return {"status": "PARSE_FAILED", "error": repr(exc)}
    counts = {"AnimationStack": 0, "AnimationLayer": 0, "AnimationCurveNode": 0, "AnimationCurve": 0}
    keys, stacks = [], []
    for e in root.elems:
        if e.id == b"Objects":
            for c in e.elems:
                name = c.id.decode()
                if name in counts:
                    counts[name] += 1
                if c.id == b"AnimationStack":
                    stacks.append(c.props[1].decode(errors="replace").split("\x00")[0] if len(c.props) > 1 else "?")
                if c.id == b"AnimationCurve":
                    for s in c.elems:
                        if s.id == b"KeyTime":
                            keys.append(len(s.props[0]))
    takes = [c.props[0].decode(errors="replace") for e in root.elems if e.id == b"Takes" for c in e.elems
             if c.id == b"Take"]
    return {"status": "PARSED", "fbx_version": version, "counts": counts, "stacks": stacks, "takes": takes,
            "curve_keys_total": int(sum(keys)), "curve_keys_min": min(keys) if keys else None,
            "curve_keys_max": max(keys) if keys else None}


# --- export / import ------------------------------------------------------------------------------------------------
def do_export(variant, fbx, v0_kwargs):
    import bpy
    if variant == "V0":
        from blender.export import export_fbx   # the real, unchanged production function
        export_fbx(str(fbx))
        return {"call": "blender.export.export_fbx(path)"}
    kw = C.variant_kwargs(v0_kwargs, variant)
    call = {k: (set(v) if k == "object_types" else v) for k, v in kw.items()}
    res = bpy.ops.export_scene.fbx(filepath=str(fbx), **call)
    return {"call": "bpy.ops.export_scene.fbx", "kwargs": kw, "result": sorted(res)}


def resolve_action(acts, name):
    match = [a for a in acts if a == name or a.endswith("|" + name)]
    if len(match) == 1:
        return match[0], "name_match"
    if not match and len(acts) == 1:
        return acts[0], "single_take"
    return None, "NO_UNIQUE_ACTION"


def rest_info(arm):
    return {b.name: [[float(v) for v in r] for r in b.matrix_local] for b in arm.data.bones}


def export_import(case_dir, variant, work, action_name, times, v0_kwargs, src_rest):
    import bpy
    fbx = case_dir/"fbx"/f"{variant}.fbx"
    fbx.parent.mkdir(exist_ok=True)
    work_sha = sha256(work)
    arm, act, info = open_case(work, action_name)
    before = channel_digest(channels_of(act))
    exp = do_export(variant, fbx, v0_kwargs)
    after = channel_digest(channels_of(act))
    rec = {"variant": variant, "export": exp, "exported_active": active_info(arm), "source_assigned": info,
           "action_digest_before_export": before, "action_digest_after_export": after,
           "action_unmodified": before == after, "work_file_unchanged": sha256(work) == work_sha,
           "fbx_sha256": sha256(fbx), "fbx_bytes": fbx.stat().st_size, "fbx_structure": fbx_structure(fbx)}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    iarm = armature_obj()
    acts = [a.name for a in bpy.data.actions]
    rec.update({"imported_actions": acts, "imported_action_count": len(acts),
                "active_after_import": active_info(iarm), "imported_timing": scene_timing(),
                "imported_armature_scale": list(iarm.scale), "imported_bones": len(iarm.data.bones)})
    chosen, how = resolve_action(acts, action_name)
    rec["selected_action"], rec["selection"] = chosen, how
    if chosen is None:
        rec["status"] = "NO_UNIQUE_ACTION"
        (case_dir/f"{variant}_meta.json").write_text(json.dumps(rec, indent=1))
        return rec
    iact = bpy.data.actions[chosen]
    rec["assigned"] = assign(iarm, iact)
    ch = channels_of(iact)
    rec["imported_fcurves"] = len(ch)
    rec["imported_keys_total"] = int(sum(c["keys"] for c in ch))
    rec["imported_interpolation"] = sorted({i for c in ch for i in c["interpolation"]})
    tgt = [c for c in ch if c["data_path"] == 'pose.bones["upperarm_l"].rotation_quaternion']
    rec["target_keys"] = [c["keys"] for c in tgt]
    obj = [c for c in ch if not c["data_path"].startswith("pose.bones")]
    rec["object_channels"] = sorted({(c["data_path"]) for c in obj})
    rec["object_keys"] = {f'{c["data_path"]}[{c["index"]}]': c["keys"] for c in obj}
    rec["imported_channel_digest"] = channel_digest(ch)
    irest = rest_info(iarm)
    rec["rest_rotation_comparable"] = {
        n: bool(max(abs(irest[n][r][c] - src_rest[n][r][c]) for r in range(3) for c in range(3)) <= 1e-4)
        for n in src_rest if n in irest}
    rec["status"] = "IMPORTED"
    rec.update(sample(iarm, times, case_dir/f"{variant}.npz"))
    (case_dir/f"{variant}_meta.json").write_text(json.dumps(rec, indent=1))
    return rec


def run_case(case_dir, case, v0_kwargs, variants):
    import bpy
    case_dir.mkdir(parents=True)
    src = Path(case["input_copy"])
    if sha256(src) != case["input_sha256"]:
        raise RuntimeError(f"STOP: scratch input {src} does not match its declared hash")
    work = case_dir/"input.blend"
    shutil.copyfile(src, work)
    name = case["action"]
    arm, act, info = open_case(work, name)
    timing = scene_timing()
    times = sample_times(timing["frame_start"], timing["frame_end"])
    ch = channels_of(act)
    src_rest = rest_info(arm)
    out = {"case": case["id"], "assigned": info, "timing": timing, "samples": len(times),
           "source_action_digest": channel_digest(ch), "source_fcurves": len(ch),
           "source_keys_total": int(sum(c["keys"] for c in ch)),
           "source_interpolation": sorted({i for c in ch for i in c["interpolation"]}),
           "stored_actions": sorted(a.name for a in bpy.data.actions), "armature_scale": list(arm.scale)}
    out.update(sample(arm, times, case_dir/"source.npz"))
    (case_dir/"source_rest.json").write_text(json.dumps(src_rest))
    out["variants"] = {}
    for v in variants:
        try:
            r = export_import(case_dir, v, work, name, times, v0_kwargs, src_rest)
            out["variants"][v] = {k: r[k] for k in r if k not in ("rest_rotation_comparable",)}
        except Exception as exc:  # recorded, never retried with other settings
            out["variants"][v] = {"status": "FAILED", "error": repr(exc)}
    out["work_sha256_end"] = sha256(work)
    return out


# --- V3 synthetic world-equivalent 2x2 -------------------------------------------------------------------------
def build_v3(path, cell):
    import bpy
    from mathutils import Quaternion, Vector
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = 30, 1.0
    scene.frame_start, scene.frame_end = int(C.V3_FRAMES[0]), int(C.V3_FRAMES[-1])
    s = C.V3_SCALE[cell]
    k = 1.0/s
    data = bpy.data.armatures.new(f"v3_{cell}")
    arm = bpy.data.objects.new("root", data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for name, head, tail, parent in C.V3_BONES:
        b = data.edit_bones.new(name)
        b.head, b.tail = Vector(head)*k, Vector(tail)*k
        if parent:
            b.parent = data.edit_bones[parent]
            b.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.scale = (s, s, s)
    arm.rotation_mode = "QUATERNION"
    act = bpy.data.actions.new(f"V3 {cell}")
    act.use_fake_user = True
    arm.animation_data_create()
    arm.animation_data.action = act
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    rest_q = data.bones["base"].matrix_local.to_quaternion()
    children = {n: j for j, n in enumerate(C.V3_CHILD_BONES)}
    for f in C.V3_FRAMES:
        g = Quaternion(C.v3_global_rotation(f))
        if C.V3_ANIMATED[cell]:
            arm.rotation_quaternion = g
            arm.keyframe_insert("rotation_quaternion", frame=f)
        else:
            pb = arm.pose.bones["base"]
            pb.rotation_quaternion = rest_q.inverted() @ g @ rest_q   # base head at origin: rotation about origin
            pb.keyframe_insert("rotation_quaternion", frame=f)
        for n, j in children.items():
            pb = arm.pose.bones[n]
            pb.rotation_quaternion = Quaternion(C.v3_child_rotation(j, f))
            pb.keyframe_insert("rotation_quaternion", frame=f)
    from w106_action_blender import fcurves
    for fc in fcurves(act):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    return act.name


def run_v3(out_dir, v0_kwargs):
    import numpy as np
    out_dir.mkdir(parents=True)
    rep = {"cells": {}}
    times = sample_times(int(C.V3_FRAMES[0]), int(C.V3_FRAMES[-1]))
    for cell in C.V3_CELLS:
        d = out_dir/cell
        d.mkdir()
        blend = d/"built.blend"
        name = build_v3(blend, cell)
        arm, act, info = open_case(blend, name)
        ch = channels_of(act)
        rec = {"action": name, "built_sha256": sha256(blend), "source_action_digest": channel_digest(ch),
               "source_fcurves": len(ch), "object_scale": list(arm.scale), "object_animated": C.V3_ANIMATED[cell],
               "timing": scene_timing()}
        rec.update(sample(arm, times, d/"source.npz"))
        src_rest = rest_info(arm)
        (d/"source_rest.json").write_text(json.dumps(src_rest))
        rec["V0"] = {k: v for k, v in export_import(d, "V0", blend, name, times, v0_kwargs, src_rest).items()
                     if k != "rest_rotation_comparable"}
        rep["cells"][cell] = rec
    ref = np.load(out_dir/C.V3_REFERENCE/"source.npz")
    for cell in C.V3_CELLS:
        z = np.load(out_dir/cell/"source.npz")
        rep["cells"][cell]["equivalence_vs_reference_m"] = float(max(np.abs(z["world_head"] - ref["world_head"]).max(),
                                                                     np.abs(z["world_tail"] - ref["world_tail"]).max()))
    return rep


# --- stage: probe -------------------------------------------------------------------------------------------------
def probe(o):
    import bpy
    decl_bytes = Path(o.declaration).read_bytes()
    if hashlib.sha256(decl_bytes).hexdigest() != o.declaration_sha:
        raise RuntimeError("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    if bpy.app.version_string != decl["environment"]["blender_version"]:
        raise RuntimeError("STOP: Blender version differs from the declaration")
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    v0_kwargs = decl["exporter"]["v0_kwargs"]
    rep = {"schema": C.SCHEMA + ":probe", "blender_version": bpy.app.version_string,
           "declaration_sha256": o.declaration_sha, "cases": {}}
    for cid, case in decl["cases"].items():
        rep["cases"][cid] = run_case(out/cid, case, v0_kwargs, decl["variant_order_testable"])
    try:
        rep["v3"] = run_v3(out/"v3", v0_kwargs)
    except Exception as exc:  # V3 failure is recorded as NOT TESTABLE by the analysis, never retried
        import traceback
        rep["v3"] = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
    (out/"probe_report.json").write_text(json.dumps(rep, indent=1))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("options", "probe"))
    for name in ("out", "declaration", "declaration_sha", "out_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args(argv)
    try:
        return {"options": options, "probe": probe}[o.mode](o)
    except Exception as exc:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        print(f"W107 Blender stage failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
