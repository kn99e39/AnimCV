#!/usr/bin/env python3
"""W108 Blender-side stages (run inside Blender 5.1).  Scratch copies only; no original file is ever written.

  options  W107's unchanged ``options`` stage (runtime export/import operator properties, add-on version/SHAs).
  probe    for each declared real case (W107 scratch inputs) and each synthetic control (built here on a copy of the
           W106 synthetic N0 file): sample the Blender source on the declared integer/quarter-frame grid; export V0
           (unchanged production ``export_fbx``) and V1 (W107 isolated no-simplification call), import each with
           default settings into a factory-empty scene and sample at identical times (W107 helpers, read-only reuse);
           decode the FBX animation curves with the add-on's own parser; then the single P2 intervention on a fresh
           V1 import: copy the selected imported Action, set only its key interpolation to CONSTANT, assign the copy
           with the same slot identifier and sample again.

    blender -b --factory-startup -t 2 --python-exit-code 1 -P scripts/w108_fbx_subframe_blender.py -- probe ...
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

import w108_fbx_subframe_contract as C  # noqa: E402
import w107_fbx_blender as B7  # noqa: E402  (unchanged W107 export/import helpers, read-only reuse)
from w106_action_blender import (active_info, armature_obj, channels_of, fcurves, open_case, sample,  # noqa: E402
                                 scene_timing)
from w106_action_contract import channel_digest  # noqa: E402

sha256 = C.sha256_file
KEY_FLOAT2 = ("co", "handle_left", "handle_right")
KEY_ENUM = ("interpolation", "handle_left_type", "handle_right_type", "easing", "type")
KEY_FLOAT = ("back", "amplitude", "period")


# --- key data (byte-level identity of everything but interpolation) -------------------------------------------------
def _enum_values(prop):
    import bpy
    return {i.identifier: i.value for i in bpy.types.Keyframe.bl_rna.properties[prop].enum_items}


def keydata(action):
    import numpy as np
    out = {}
    for fc in sorted(fcurves(action), key=lambda f: (f.data_path, f.array_index)):
        kps = fc.keyframe_points
        n = len(kps)
        rec = {}
        for p in KEY_FLOAT2:
            a = np.empty(2*n, dtype=np.float32)
            kps.foreach_get(p, a)
            rec[p] = a
        for p in KEY_ENUM:
            a = np.empty(n, dtype=np.int32)
            kps.foreach_get(p, a)
            rec[p] = a
        for p in KEY_FLOAT:
            a = np.empty(n, dtype=np.float32)
            kps.foreach_get(p, a)
            rec[p] = a
        rec["meta"] = (fc.extrapolation, len(fc.modifiers), bool(fc.mute), fc.group.name if fc.group else None, n)
        out[f"{fc.data_path}[{fc.array_index}]"] = rec
    return out


def keydata_diff(a, b, ignore=()):
    """Fields that differ between two keydata snapshots (excluding ``ignore``)."""
    if set(a) != set(b):
        return ["<curve set>"]
    bad = set()
    for k in a:
        for f in a[k]:
            if f in ignore:
                continue
            if f == "meta":
                if a[k][f] != b[k][f]:
                    bad.add("meta")
            elif a[k][f].tobytes() != b[k][f].tobytes():
                bad.add(f)
    return sorted(bad)


def interp_set(kd):
    import bpy
    names = {v: k for k, v in _enum_values("interpolation").items()}
    _ = bpy
    return sorted({names[int(x)] for r in kd.values() for x in set(r["interpolation"].tolist())})


def key_time_stats(kd):
    import numpy as np
    t = np.concatenate([r["co"][0::2] for r in kd.values()]) if kd else np.zeros(0)
    if t.size == 0:
        return {"keys": 0}
    dev = np.abs(t.astype(np.float64) - np.round(t.astype(np.float64)))
    return {"keys": int(t.size), "frame_min": float(t.min()), "frame_max": float(t.max()),
            "max_dev_from_integer": float(dev.max()), "distinct_frames": int(np.unique(t).size),
            "keys_per_curve": sorted({int(r["meta"][4]) for r in kd.values()})}


def set_interpolation(action, ident):
    import numpy as np
    val = _enum_values("interpolation")[ident]
    for fc in fcurves(action):
        n = len(fc.keyframe_points)
        fc.keyframe_points.foreach_set("interpolation", np.full(n, val, dtype=np.int32))


def assign_slot(arm, action, identifier):
    """Assign ``action`` with the slot whose identifier equals ``identifier``; returns the record or None."""
    arm.animation_data_create()
    ad = arm.animation_data
    ad.action = action
    slots = [s for s in getattr(action, "slots", []) if s.identifier == identifier]
    if len(slots) != 1:
        return None
    ad.action_slot = slots[0]
    return {"action": ad.action.name, "slot": ad.action_slot.identifier, "nla_tracks": len(ad.nla_tracks),
            "action_blend_type": ad.action_blend_type, "action_influence": float(ad.action_influence)}


# --- FBX curve decoding with the add-on parser (cross-checked on the host by the independent reader) ---------------
def fbx_addon_summary(path, fps):
    try:
        from io_scene_fbx import parse_fbx
        root, version = parse_fbx.parse(str(path))
    except Exception as exc:  # recorded; never retried
        return {"status": "PARSE_FAILED", "error": repr(exc)}
    s = C.fbx_anim_summary(root, fps)
    s["fbx_version"] = version
    return s


# --- P2 ------------------------------------------------------------------------------------------------------------
def p2(case_dir, fbx, action_name, times):
    import bpy
    rec = {"intervention": C.P2["name"]}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    arm = armature_obj()
    acts = [a.name for a in bpy.data.actions]
    chosen, how = B7.resolve_action(acts, action_name)
    rec.update({"imported_actions": len(acts), "selected_action": chosen, "selection": how,
                "active_after_import": active_info(arm), "imported_timing": scene_timing()})
    if chosen is None:
        rec["status"] = "NOT_TESTABLE"
        rec["reason"] = "no unique imported Action"
        return rec
    orig = bpy.data.actions[chosen]
    from w106_action_blender import assign
    rec["assigned_original"] = assign(arm, orig)
    slot_id = rec["assigned_original"]["slot"]
    k0 = keydata(orig)
    rec["original_digest"] = channel_digest(channels_of(orig))
    rec["original_interpolation"] = interp_set(k0)
    rec["original_key_times"] = key_time_stats(k0)
    rec["pre_sample"] = sample(arm, times, case_dir/"P2_pre.npz")   # fresh import, original Action (LINEAR)
    n_actions_before = len(bpy.data.actions)
    cp = orig.copy()
    rec["copy_name"] = cp.name
    rec["actions_added"] = len(bpy.data.actions) - n_actions_before
    k1 = keydata(cp)
    rec["copy_equals_original_before"] = keydata_diff(k0, k1)
    set_interpolation(cp, "CONSTANT")
    k2 = keydata(cp)
    rec["copy_fields_changed"] = keydata_diff(k1, k2)
    rec["copy_fields_changed_excl_interpolation"] = keydata_diff(k1, k2, ignore=("interpolation",))
    rec["copy_interpolation"] = interp_set(k2)
    rec["copy_key_counts_equal"] = all(k1[c]["meta"][4] == k2[c]["meta"][4] for c in k1)
    rec["assigned_copy"] = assign_slot(arm, cp, slot_id)
    if rec["assigned_copy"] is None:
        rec["status"] = "NOT_TESTABLE"
        rec["reason"] = f"no slot with identifier {slot_id} on the copy"
        return rec
    rec["original_unchanged_after"] = not keydata_diff(k0, keydata(orig)) and \
        channel_digest(channels_of(orig)) == rec["original_digest"]
    rec["copy_channel_digest"] = channel_digest(channels_of(cp))
    rec["copy_channel_digest_equals_original"] = rec["copy_channel_digest"] == rec["original_digest"]
    rec["armature_scale"] = list(arm.scale)
    rec["timing_after"] = scene_timing()
    rec["keydata_ok"] = (not rec["copy_equals_original_before"] and rec["copy_fields_changed_excl_interpolation"] == []
                         and rec["copy_interpolation"] == ["CONSTANT"] and rec["copy_key_counts_equal"]
                         and rec["original_unchanged_after"] and rec["copy_channel_digest_equals_original"])
    rec["status"] = "TESTED" if rec["keydata_ok"] else "NOT_TESTABLE"
    if not rec["keydata_ok"]:
        rec["reason"] = "the copy differs from the original in a field other than interpolation, or the original changed"
    rec["sample"] = sample(arm, times, case_dir/"P2.npz")
    return rec


# --- one case ------------------------------------------------------------------------------------------------------
def run_flow(case_dir, work, action_name, times, v0_kwargs, declared):
    """Source sampling, V0/V1 export-import (W107 helper), FBX decoding, P2.  ``work`` is the scratch .blend."""
    import bpy
    arm, act, info = open_case(work, action_name)
    timing = scene_timing()
    ch = channels_of(act)
    kd = keydata(act)
    out = {"assigned": info, "timing": timing, "samples": len(times),
           "source_action_digest": channel_digest(ch), "source_fcurves": len(ch),
           "source_keys_total": int(sum(c["keys"] for c in ch)),
           "source_interpolation": sorted({i for c in ch for i in c["interpolation"]}),
           "source_extrapolation": sorted({c["extrapolation"] for c in ch}),
           "source_modifiers": int(sum(c["modifiers"] for c in ch)),
           "source_key_frames": sorted({f for c in ch for f in c["frames"]}),
           "source_key_times": key_time_stats(kd),
           "stored_actions": sorted(a.name for a in bpy.data.actions), "armature_scale": list(arm.scale),
           "slots": [s.identifier for s in getattr(act, "slots", [])]}
    out["times_equal_declared"] = list(times) == list(declared["times"])
    out.update(sample(arm, times, case_dir/"source.npz"))
    src_rest = B7.rest_info(arm)
    (case_dir/"source_rest.json").write_text(json.dumps(src_rest))
    out["paths"] = {}
    for v in C.EXPORT_PATHS:
        try:
            r = B7.export_import(case_dir, v, work, action_name, times, v0_kwargs, src_rest)
            rec = {k: r[k] for k in r if k != "rest_rotation_comparable"}
            if r.get("status") == "IMPORTED":
                rec["imported_key_times"] = key_time_stats(keydata(bpy.data.actions[r["selected_action"]]))
            rec["fbx_addon_summary"] = fbx_addon_summary(case_dir/"fbx"/f"{v}.fbx", timing["fps"]/timing["fps_base"])
            out["paths"][v] = rec
        except Exception as exc:  # recorded, never retried with other settings
            out["paths"][v] = {"status": "FAILED", "error": repr(exc)}
    try:
        out["P2"] = p2(case_dir, case_dir/"fbx"/"V1.fbx", action_name, times)
    except Exception as exc:
        import traceback
        out["P2"] = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
    out["work_sha256_end"] = sha256(work)
    return out


def run_real(case_dir, case, v0_kwargs):
    case_dir.mkdir(parents=True)
    src = Path(case["input_copy"])
    if sha256(src) != case["input_sha256"]:
        raise RuntimeError(f"STOP: scratch input {src} does not match its declared hash")
    work = case_dir/"input.blend"
    shutil.copyfile(src, work)
    out = run_flow(case_dir, work, case["action"], case["times"], v0_kwargs, case)
    out["work_file_unchanged"] = out["work_sha256_end"] == case["input_sha256"]
    return out


def build_control(ctrl_dir, ctrl, spec, syn_input):
    """Synthetic control on a scratch copy of the W106 synthetic N0 file (see C.SYN_CONSTRUCTION)."""
    import bpy
    base_copy = ctrl_dir/"base_input.blend"
    shutil.copyfile(syn_input["path"], base_copy)
    if sha256(base_copy) != syn_input["sha256"]:
        raise RuntimeError("STOP: synthetic scratch input hash mismatch")
    arm, base, info = open_case(base_copy, C.SYN_BASE_ACTION)
    kb = keydata(base)
    dup = base.copy()
    dup.name = spec["action"]
    set_interpolation(dup, spec["source_interpolation"])
    kd = keydata(dup)
    rec = {"base_assigned": info, "base_digest": channel_digest(channels_of(base)),
           "dup_name": dup.name, "dup_fields_changed_excl_interpolation": keydata_diff(kb, kd, ignore=("interpolation",)),
           "dup_interpolation": interp_set(kd), "base_interpolation": interp_set(kb)}
    rec["assigned"] = assign_slot(arm, dup, info["slot"])
    rec["base_unchanged"] = not keydata_diff(kb, keydata(base))
    rec["dup_digest_equals_base"] = channel_digest(channels_of(dup)) == rec["base_digest"]
    rec["timing"] = scene_timing()
    work = ctrl_dir/"input.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(work))
    rec["work_sha256"] = sha256(work)
    rec["base_input_unchanged"] = sha256(base_copy) == syn_input["sha256"]
    rec["valid_construction"] = (rec["assigned"] is not None and rec["dup_fields_changed_excl_interpolation"] == []
                                 and rec["dup_interpolation"] == [spec["source_interpolation"]] and rec["base_unchanged"]
                                 and rec["dup_digest_equals_base"] and rec["base_input_unchanged"])
    return work, rec


def run_control(ctrl_dir, ctrl, spec, syn_input, v0_kwargs):
    ctrl_dir.mkdir(parents=True)
    work, build = build_control(ctrl_dir, ctrl, spec, syn_input)
    out = run_flow(ctrl_dir, work, spec["action"], spec["times"], v0_kwargs, spec)
    out["build"] = build
    out["work_file_unchanged"] = out["work_sha256_end"] == build["work_sha256"]
    return out


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
           "declaration_sha256": o.declaration_sha, "cases": {}, "controls": {}}
    for cid, case in decl["cases"].items():
        rep["cases"][cid] = run_real(out/cid, case, v0_kwargs)
    for ctrl, spec in decl["synthetic_controls"].items():
        try:
            rep["controls"][ctrl] = run_control(out/ctrl, ctrl, spec, decl["synthetic_input"], v0_kwargs)
        except Exception as exc:  # recorded as invalid by the analysis, never retried
            import traceback
            rep["controls"][ctrl] = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
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
        return {"options": B7.options, "probe": probe}[o.mode](o)
    except Exception as exc:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        print(f"W108 Blender stage failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
