#!/usr/bin/env python3
"""W106 host-side stages: existing BaseRig Action editability and round-trip contract (evidence only).

  verify   W105 COMPLETE marker + audit, W105 evidence-manifest identity of every used artifact, rig-asset hashes,
           then byte-verified scratch copies of the frozen .blend files into <out>/inputs and an inventory spec
  declare  predeclared selection (from Action inventory metadata only), windows, offsets, tolerances, gates and CASE
           mapping -> declaration.json + SHA-256 (``edit_outcomes_read: false``)
  analyze  every gate from the probe / repeat-probe / render outputs -> analysis_report.json and the CASE
  compose  fixed-view before/after/inverse contact sheets (rig renders only; no source pixels)
  manifest hashes of every output, plus a re-check that no original W105/legacy file changed

Scratch copies only.  No reconstruction, rig, exporter or production module is modified or re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w106_action_contract as C  # noqa: E402

W105_MANIFEST_SHA = "cb0eca93c7847a70d497bd36c57856bd8295f3d8e3a7a72913ebf876ebe96899"
WINDOWS = ("arm_natural_most", "complete_control", "high_motion", "identity_caveat", "leg_natural_most",
           "new_wrong_knee", "w102_max_source_regression", "w104_largest_harm")
W105_USED = [f"results/render/{w}/{w}.blend" for w in WINDOWS] + [
    "results/render_clip/real_clip/real_clip.blend", "results/build/cases.json", "results/build/video_cases.json",
    "results/render_audit.json", "results/render_clip_audit.json", "results/render/blender_report.json",
    "results/render_clip/blender_report.json"]
LEGACY = "examples/e2e_demo/animated_rig.blend"
RIG_ASSETS = ["examples/skinned_mannequin/BaseRig_Skinned.fbx", "examples/e2e_demo/baserig_blender_rest.json",
              "src/blender/export.py", "src/blender/executor.py", "scripts/render_runtime_heading_rig_blender.py",
              "scripts/w105_rig_blender.py"]
NATIVE_FPS = {"3dpw": 30.0, "real_clip": 50.0}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wjson(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))
    return sha256(path)


# --- verify -------------------------------------------------------------------------------------------------------
def verify(o):
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    w105 = Path(o.w105)
    rep = {"schema": C.SCHEMA + ":verify", "checks": {}, "stop": []}
    wl = (ROOT/"docs/105_WORKLOG_SOURCE_ONLY_RESTORATION_RIG_UTILITY.md").read_text(encoding="utf-8")
    rep["checks"]["w105_complete_marker"] = wl.rstrip().endswith("AUTOMATION_BATCH_STATUS: COMPLETE\nBATCH_NUMBER: 105")
    rep["checks"]["w101_w105_audit_present"] = (ROOT/"docs/AUDIT_W101_W105_2026_11_02.md").exists()
    msha = sha256(w105/"evidence_manifest.json")
    rep["checks"]["w105_manifest_sha"] = {"value": msha, "ok": msha == W105_MANIFEST_SHA}
    man = json.loads((w105/"evidence_manifest.json").read_text())["files"]
    files = {}
    for rel in W105_USED:
        p = w105/rel
        rec = {"exists": p.exists()}
        if p.exists():
            rec["sha256"] = sha256(p)
            want = man.get(rel, {}).get("sha256") if isinstance(man.get(rel), dict) else man.get(rel)
            rec["manifest_sha256"] = want
            rec["ok"] = rec["sha256"] == want
        else:
            rec["ok"] = False
        files[rel] = rec
    rep["w105_files"] = files
    rep["rig_assets"] = {r: sha256(ROOT/r) for r in RIG_ASSETS}
    rep["legacy_smoke"] = {"path": LEGACY, "sha256": sha256(ROOT/LEGACY)}
    for k, v in rep["checks"].items():
        if not (v if isinstance(v, bool) else v["ok"]):
            rep["stop"].append(k)
    rep["stop"] += [f"w105:{r}" for r, v in files.items() if not v["ok"]]
    # byte-verified scratch copies; originals are only ever read
    inputs = out.parent/"inputs"
    inputs.mkdir(exist_ok=False)
    spec = {"files": {}}
    sources = {w: w105/f"results/render/{w}/{w}.blend" for w in WINDOWS}
    sources["real_clip"] = w105/"results/render_clip/real_clip/real_clip.blend"
    sources["legacy_smoke"] = ROOT/LEGACY
    for key, src in sources.items():
        dst = inputs/f"{key}.blend"
        shutil.copyfile(src, dst)
        ok = sha256(dst) == sha256(src)
        spec["files"][key] = {"path": str(dst), "source": str(src), "sha256": sha256(dst), "copy_identical": ok}
        if not ok:
            rep["stop"].append(f"copy:{key}")
    rep["inventory_spec_sha256"] = wjson(out/"inventory_spec.json", spec)
    rep["ok"] = not rep["stop"]
    wjson(out/"verify_report.json", rep)
    print(json.dumps({"ok": rep["ok"], "stop": rep["stop"]}))
    return 0 if rep["ok"] else 2


# --- declare ------------------------------------------------------------------------------------------------------
def _target_track(action):
    comps = sorted((c for c in action["channels"] if c["data_path"] == f'pose.bones["{C.TARGET_BONE}"].rotation_quaternion'),
                   key=lambda c: c["index"])
    if len(comps) != 4:
        return None, None
    frames = comps[0]["frames"]
    quats = [tuple(c["values"][k] for c in comps) for k in range(len(frames))]
    return frames, quats


def _case_entry(cid, kind, inv_file, action_name, spec_file, native, extra):
    act = inv_file["actions"][action_name]
    frames, quats = _target_track(act)
    t = inv_file["timing"]
    win, grp = C.e1_window(len(frames)), C.e2_group(len(frames))
    delta = C.e2_offset(frames, grp)
    bones = inv_file["bones"]
    desc = bones[C.TARGET_BONE]["children_recursive"]
    entry = {
        "id": cid, "kind": kind, "input_copy": spec_file["path"], "input_sha256": spec_file["sha256"],
        "source": spec_file["source"], "action": action_name, "action_value_digest": act["value_digest"],
        "action_fcurves": act["fcurves"], "action_slots": act["slots"], "scene_timing": t,
        "saved_active_action": inv_file["active"], "armature": inv_file["armature"],
        "target_bone": C.TARGET_BONE, "target_parent": bones[C.TARGET_BONE]["parent"],
        "target_length_world_m": bones[C.TARGET_BONE]["length_world"],
        "descendants": desc, "non_target_bones": sorted(set(bones) - {C.TARGET_BONE} - set(desc)),
        "target_key_frames": frames, "n_keys": len(frames),
        "e1_window": win, "e1_window_frames": [frames[i] for i in win],
        "e1_hold_interval": list(C.hold_interval(frames, win)),
        "e1_expected_elbow_disp_m": C.expected_elbow_displacement(bones[C.TARGET_BONE]["length_world"]),
        "e2_group": grp, "e2_group_frames": [frames[i] for i in grp], "e2_offset": delta,
        "e2_new_frames_plus": [frames[i] + delta for i in grp], "e2_new_frames_minus": [frames[i] - delta for i in grp],
        "render_frames": C.render_frames(frames, win), "render": kind != "legacy_smoke",
        "render_views": ["W91 video", "W91 side"], "fbx": True}
    if native is not None:
        entry["timebase"] = native
        f2t = native["blender_frame_to_native_s"]
        entry["e1_window_native_s"] = [f2t(f) for f in entry["e1_window_frames"]]
        entry["e2_group_native_s"] = [f2t(f) for f in entry["e2_group_frames"]]
        entry["e2_new_native_s_plus"] = [f2t(f) for f in entry["e2_new_frames_plus"]]
        entry["e2_new_native_s_minus"] = [f2t(f) for f in entry["e2_new_frames_minus"]]
        entry["timebase"] = {k: v for k, v in native.items() if k != "blender_frame_to_native_s"}
    entry["e2_blender_playback_s"] = {"group": [(f - t["frame_start"])/t["effective_fps"] for f in entry["e2_group_frames"]],
                                      "offset_s": delta/t["effective_fps"]}
    entry.update(extra)
    return entry


def declare(o):
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    w105 = Path(o.w105)
    ver = json.loads(Path(o.verify).read_text())
    if not ver["ok"]:
        raise SystemExit("STOP: verify failed")
    inv = json.loads(Path(o.inventory).read_text())
    spec = json.loads((Path(o.verify).parent/"inventory_spec.json").read_text())["files"]
    audit = json.loads((w105/"results/render_audit.json").read_text())["files"]
    caudit = json.loads((w105/"results/render_clip_audit.json").read_text())["files"]
    cases = json.loads((w105/"results/build/cases.json").read_text())["cases"]
    vcases = json.loads((w105/"results/build/video_cases.json").read_text())["cases"]
    identity = {}
    windows = {}
    for w in WINDOWS:
        name = f"W91 {w} FIXED"
        f = inv["files"][w]
        a = f["actions"][name]
        identity[w] = {"inventory_digest": a["value_digest"],
                       "w105_audit_digest": audit[f"{w}/{w}.blend"]["actions"][name]["value_digest"]}
        identity[w]["ok"] = identity[w]["inventory_digest"] == identity[w]["w105_audit_digest"]
        frames, quats = _target_track(a)
        rows = cases[w]["frames"]
        windows[w] = {"identities": [r["identity"] for r in rows],
                      "target_keyed_every_sample": frames is not None and len(frames) == len(rows),
                      "target_range_deg": C.track_range_deg(quats) if quats else 0.0}
    rc_name = "W91 real_clip FIXED"
    identity["real_clip"] = {"inventory_digest": inv["files"]["real_clip"]["actions"][rc_name]["value_digest"],
                             "w105_audit_digest": caudit["real_clip/real_clip.blend"]["actions"][rc_name]["value_digest"]}
    identity["real_clip"]["ok"] = identity["real_clip"]["inventory_digest"] == identity["real_clip"]["w105_audit_digest"]
    if not all(v["ok"] for v in identity.values()):
        raise SystemExit("STOP: Action identity differs from the W105 audit")
    moving, reasons = C.select_moving_window(windows)
    if moving is None:
        raise SystemExit("STOP: no qualifying moving window (CASE D)")

    def tb_3dpw(w):
        rows = cases[w]["frames"]
        f0 = rows[0]["frame_index"]
        ok = all(abs(r["time_s"] - r["frame_index"]/30.0) < 1e-9 for r in rows)
        bl = [r["frame_index"] - f0 + 1 for r in rows]
        return {"regime": "benchmark_detector_observation (3DPW, W102 mixed-source contract)", "native_fps": 30.0,
                "source_bank_hz": 10.0, "first_frame_index": f0, "time_s_equals_frame_index_over_30": ok,
                "key_frames_equal_frame_index_minus_f0_plus_1": bl == windows_frames[w],
                "native_span_s": rows[-1]["time_s"] - rows[0]["time_s"],
                "sample_ids": [r["sample_id"] for r in rows], "native_frame_indices": [r["frame_index"] for r in rows],
                "blender_frame_to_native_s": lambda f, f0=f0: (f + f0 - 1)/30.0}

    windows_frames = {w: _target_track(inv["files"][w]["actions"][f"W91 {w} FIXED"])[0] for w in WINDOWS}
    rrows = vcases["real_clip"]["frames"]
    rc_frames = _target_track(inv["files"]["real_clip"]["actions"][rc_name])[0]
    tb_real = {"regime": "real_animcv_observation (GT-free historical clip, S0-only W105 configuration)",
               "native_fps": 50.0, "native_frames": len(rrows),
               "time_s_equals_frame_index_over_50": all(abs(r["time_s"] - r["frame_index"]/50.0) < 1e-9 for r in rrows),
               "key_frames_equal_frame_index_plus_1": [r["frame_index"] + 1 for r in rrows] == rc_frames,
               "native_span_s": rrows[-1]["time_s"] - rrows[0]["time_s"],
               "blender_frame_to_native_s": lambda f: (f - 1)/50.0}
    sel = {
        "complete_control": _case_entry("complete_control", "3dpw_complete_control", inv["files"]["complete_control"],
                                        "W91 complete_control FIXED", spec["complete_control"], tb_3dpw("complete_control"),
                                        {"why": "the W105 complete-control window (all rows complete; M_Main = R11 digests)"}),
        moving: _case_entry(moving, "3dpw_moving", inv["files"][moving], f"W91 {moving} FIXED", spec[moving], tb_3dpw(moving),
                            {"why": "first qualifying window in stable sorted case-id order (see selection_reasons)"}),
        "real_clip": _case_entry("real_clip", "real_clip", inv["files"]["real_clip"], rc_name, spec["real_clip"], tb_real,
                                 {"why": "the only historical real-clip M_Main Action (W105 S0-only configuration)"}),
    }
    leg = inv["files"]["legacy_smoke"]
    leg_actions = sorted(leg["actions"])
    leg_frames = _target_track(leg["actions"][leg_actions[0]])[0] if leg_actions else None
    legacy = {"actions": leg_actions, "target_key_frames": leg_frames, "timing": leg["timing"],
              "file_version": leg["file_version"],
              "interpolation": sorted({i for c in leg["actions"][leg_actions[0]]["channels"] for i in c["interpolation"]})
              if leg_actions else None}
    if leg_frames and len(leg_frames) >= C.E2_GROUP_KEYS + 2 and len(leg_frames) >= C.E1_WINDOW_KEYS:
        sel["legacy_smoke"] = _case_entry("legacy_smoke", "legacy_smoke", leg, leg_actions[0], spec["legacy_smoke"], None,
                                          {"why": "committed e2e_demo smoke file; separately labelled, never pooled"})
        legacy["status"] = "probed"
    else:
        legacy["status"] = (f"inventory only: {len(leg_frames or [])} target keys < the declared {C.E1_WINDOW_KEYS}-key "
                            "E1 window; no edit is declared (the rule is not widened)")
    syn_frames = C.SYN_FRAMES
    syn_grp = C.e2_group(len(syn_frames))
    synthetic = {"id": "synthetic", "kind": "synthetic", "target_key_frames": syn_frames, "n_keys": len(syn_frames),
                 "e1_window": C.e1_window(len(syn_frames)), "e2_group": syn_grp, "e2_offset": C.e2_offset(syn_frames, syn_grp),
                 "descendants": ["hand_l", "lowerarm_l"], "non_target_bones": ["base", "sibling"],
                 "target_length_world_m": 0.3, "missing_key": C.SYN_MISSING, "render": False, "fbx": True}
    synthetic["e1_hold_interval"] = list(C.hold_interval(syn_frames, synthetic["e1_window"]))
    synthetic["e1_expected_elbow_disp_m"] = C.expected_elbow_displacement(0.3)
    decl = {
        "schema": C.SCHEMA + ":declaration", "edit_outcomes_read": False, "starting_head": o.starting_head,
        "blender_version": inv["blender_version"], "verify_report_sha256": sha256(o.verify),
        "inventory_sha256": sha256(o.inventory), "w105_identity": identity,
        "selection_rule": {"moving": "first sorted case id != complete_control, all rows same_person, target keyed at every "
                                     f"sample, original target track range >= {C.MOVING_MIN_RANGE_DEG} deg",
                           "primary_slot": "FIXED (= M_Main); R11 / RUNTIME is not used",
                           "candidates": windows, "selection_reasons": reasons},
        "edit": {"E1": f"{C.TARGET_BONE} local rotation q' = q * rot(local X, +{C.E1_ANGLE_DEG} deg) on the central "
                       f"{C.E1_WINDOW_KEYS} keys; handles shifted with the value; inverse q * rot(X, -{C.E1_ANGLE_DEG})",
                 "E2": f"central {C.E2_GROUP_KEYS} keys of the {C.TARGET_BONE} rotation track (all 4 components) moved by "
                       f"+/-{C.E2_INTEGER_OFFSET} frame when both boundary gaps >= 2 frames, else +/-{C.E2_SUBFRAME_OFFSET}; "
                       "values, interpolation, other keys unchanged; reorder/merge/out-of-domain refused",
                 "E3": "exact inverse on a reopened edited copy, saved and reopened",
                 "E4": "every edited copy saved and reopened; FBX via unchanged src/blender/export.py and default importer",
                 "N0": "unedited copy, active Action pointer set to the declared Action (scratch-only), saved, reopened",
                 "intended_changed": "target bone local channels inside the CONSTANT hold interval; descendants' world "
                                     "transforms inside it (expected, not harm)",
                 "preserved": "every other F-curve bitwise; every non-target local channel; non-descendant world "
                              "transforms; target outside the hold interval; scene timing"},
        "tolerances": C.TOL, "units": "world metres (armature object scale 0.01 x armature units); quaternion components; "
                                      "degrees; 8-bit pixels",
        "sampling": "every integer frame and every half frame in [frame_start, frame_end]",
        "fbx_settings": {"exporter": "src/blender/export.py export_fbx (use_selection=False, object_types={'ARMATURE'}, "
                                     "bake_anim=True, all else Blender default)",
                         "exporter_sha256": sha256(ROOT/"src/blender/export.py"),
                         "importer": "bpy.ops.import_scene.fbx defaults into read_factory_settings(use_empty=True)",
                         "parity_frames": "integer frames (FBX bakes integer frames); half frames reported only"},
        "gates": {
            "N0": "digest + raw keys equal; local <= local_sample_untouched; world <= world_untouched; timing equal; "
                  "orig vs N0 renders 0 changed px",
            "E1": "window keys == q*offset (<= inverse_key_value), all other keys bitwise; non-target local <= "
                  "local_sample_untouched; target local outside hold unchanged and inside hold rotated by 20 deg (+/-0.01); "
                  "non-descendant world <= world_untouched; elbow displacement inside hold == 2 L sin(10 deg) "
                  "(+/- e1_disp_abs_err_m) and >= e1_min_visible_disp_m; render: every rendered hold frame >= "
                  "render_min_changed_px in >= 1 view, every rendered non-hold frame 0 changed px",
            "E2": "key times == shift_keys exactly, values bitwise, other keys bitwise, order + domain kept; target "
                  "local == CONSTANT step model at every sample (<= e2_model); non-target local/world unchanged; E2+ "
                  "changes >= 1 integer frame",
            "E3": "E1 inverse: keys within inverse_key_value after sign canonicalisation, every bone world <= "
                  "inverse_world_m vs N0; E2 inverse: keys and samples bitwise equal to N0",
            "E4": "every reopened edit: digest == in-memory digest before save and samples identical to in-memory",
            "DETERMINISM": "a repeat probe run on fresh copies gives identical digests and sample arrays",
            "FBX (separate)": "import ok, unique Action, no missing bones; baseline import vs N0 <= fbx_rt_parity_m; "
                              "edited vs baseline import non-descendant <= fbx_nontarget_m; edit effect <= fbx_effect_m"},
        "case_mapping": C.decide_case.__doc__,
        "stop_gates": ["artifact hash mismatch", "Action identity differs from W105 audit", "no qualifying moving window",
                       "an edit would require modifying an original file", "exporter settings would need changing"],
        "cases": sel, "synthetic": synthetic, "legacy_smoke": legacy, "known_negative_findings": [
            "real clip keys at native 50 fps frame indices in a 30 fps scene (0.6x playback)",
            "CONSTANT interpolation holds every sample; no in-between motion exists",
            "R11 hand-to-face flicker and other W105 harms (not tested, not repaired)",
            "no root translation / contact semantics in these Actions"]}
    path = out/"declaration.json"
    wjson(path, decl)
    s = sha256(path)
    (out/"declaration.sha256").write_text(s + "\n")
    print(s)
    return 0


# --- analyze ------------------------------------------------------------------------------------------------------
class Cond:
    def __init__(self, d, name):
        self.meta = json.loads((d/f"{name}.json").read_text())
        z = np.load(d/f"{name}.npz")
        self.a = {k: z[k] for k in z.files}
        self.bones = [str(b) for b in self.a["bones"]]
        self.idx = {b: i for i, b in enumerate(self.bones)}
        self.times = self.a["times"]
        self.ch = {(c["data_path"], c["index"]): c for c in self.meta["channels"]}


def _mx(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


def _sel(cond, names):
    return [cond.idx[n] for n in names if n in cond.idx]


def _world_diff(a, b, bones, tmask=None):
    ia, ib = _sel(a, bones), _sel(b, bones)
    tm = slice(None) if tmask is None else tmask
    d = max(_mx(a.a["world_head"][tm][:, ia] - b.a["world_head"][tm][:, ib]),
            _mx(a.a["world_tail"][tm][:, ia] - b.a["world_tail"][tm][:, ib]))
    r = _mx(a.a["world_rot"][tm][:, ia] - b.a["world_rot"][tm][:, ib])
    return d, r


def _local_diff(a, b, bones, tmask=None):
    ia, ib = _sel(a, bones), _sel(b, bones)
    tm = slice(None) if tmask is None else tmask
    return max(_mx(a.a[k][tm][:, ia] - b.a[k][tm][:, ib]) for k in ("local_quat", "local_loc", "local_scale"))


def _channels_equal_except(a, b, skip_path):
    diffs = []
    keys = set(a.ch) | set(b.ch)
    for k in sorted(keys):
        if k[0] == skip_path:
            continue
        ca, cb = a.ch.get(k), b.ch.get(k)
        if ca is None or cb is None or ca["frames"] != cb["frames"] or ca["values"] != cb["values"] or \
                ca["interpolation"] != cb["interpolation"]:
            diffs.append(f"{k[0]}[{k[1]}]")
    return diffs


def _target(cond):
    p = f'pose.bones["{C.TARGET_BONE}"].rotation_quaternion'
    comps = [cond.ch[(p, i)] for i in range(4)]
    return comps[0]["frames"], [tuple(c["values"][k] for c in comps) for k in range(len(comps[0]["frames"]))], p


def _qang(a, b):
    d = np.abs(np.sum(a*b, axis=-1)) / (np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))
    return np.degrees(2*np.arccos(np.clip(d, 0, 1)))


def _arrays_identical(a, b):
    return all(np.array_equal(a.a[k], b.a[k]) for k in a.a)


def _pix(p, q, level):
    from PIL import Image
    x = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
    y = np.asarray(Image.open(q).convert("RGB"), dtype=np.int16)
    return int(np.sum(np.max(np.abs(x - y), axis=-1) > level))


def analyze_case(d, case, render_dir=None):
    T = C.TOL
    cid = case["id"]
    res = {"id": cid, "kind": case["kind"], "gates": {}, "metrics": {}, "violations": []}
    g, m = res["gates"], res["metrics"]
    orig, n0 = Cond(d, "orig"), Cond(d, "n0_reopen")
    desc = case["descendants"]
    tgt_sub = [C.TARGET_BONE] + desc
    nondesc = case["non_target_bones"]
    all_b = orig.bones
    times = n0.times
    frames0, q0, tpath = _target(n0)
    # --- N0
    n0_keys = orig.meta["channel_digest"] == n0.meta["channel_digest"] and not _channels_equal_except(orig, n0, None)
    n0_local = _local_diff(orig, n0, all_b)
    n0_world, n0_rot = _world_diff(orig, n0, all_b)
    n0_timing = orig.meta["timing"] == n0.meta["timing"]
    m["N0"] = {"keys_identical": n0_keys, "local_max": n0_local, "world_max_m": n0_world, "world_rot_max": n0_rot,
               "timing_equal": n0_timing, "active": n0.meta["active"], "timing": n0.meta["timing"],
               "arrays_bitwise_identical": _arrays_identical(orig, n0)}
    ok = n0_keys and n0_local <= T["local_sample_untouched"] and n0_world <= T["world_untouched_m"] and \
        n0_rot <= T["world_matrix_untouched"] and n0_timing and n0.meta["active"]["action"] == case.get("action", orig.meta["action"])
    # CONSTANT semantics of the original: every channel evaluates as the step model of its own keys
    worst = 0.0
    for b in all_b:
        p = f'pose.bones["{b}"].rotation_quaternion'
        if (p, 0) not in orig.ch:
            continue
        comps = [orig.ch[(p, i)] for i in range(4)]
        pred = np.array([[C.step_value(comps[i]["frames"], comps[i]["values"], t) for i in range(4)] for t in times])
        worst = max(worst, _mx(pred - orig.a["local_quat"][:, orig.idx[b]]))
    m["N0"]["constant_step_model_max_err_all_tracks"] = worst
    m["N0"]["interpolation"] = sorted({i for c in orig.meta["channels"] for i in c["interpolation"]})
    # --- E1
    for tag in ("e1_mem", "e1_reopen"):
        e1 = Cond(d, tag)
        f1, q1, _ = _target(e1)
        off = C.e1_offset()
        win = case["e1_window"]
        exp = [C.apply_local_offset(q, off) if i in win else q for i, q in enumerate(q0)]
        exp32 = np.array(exp, dtype=np.float32).astype(np.float64)
        key_err = _mx(np.array(q1) - exp32)
        outside_bitwise = all(q1[i] == q0[i] for i in range(len(q0)) if i not in win)
        others = _channels_equal_except(e1, n0, tpath)
        lo, hi = case["e1_hold_interval"]
        hold = (times >= lo) & (times < hi)
        nt_local = _local_diff(e1, n0, [b for b in all_b if b != C.TARGET_BONE])
        it = e1.idx[C.TARGET_BONE]
        ang = _qang(e1.a["local_quat"][:, it], n0.a["local_quat"][:, it])
        tgt_out = _mx(e1.a["local_quat"][~hold, it] - n0.a["local_quat"][~hold, it])
        nd_world, nd_rot = _world_diff(e1, n0, nondesc)
        sub_out_world, _ = _world_diff(e1, n0, tgt_sub, ~hold)
        elbow = np.linalg.norm(e1.a["world_tail"][:, it] - n0.a["world_tail"][:, it], axis=-1)
        expd = case["e1_expected_elbow_disp_m"]
        hold_int = hold & (times == np.floor(times))
        mm = {"key_err": key_err, "outside_window_keys_bitwise": outside_bitwise, "other_channels_changed": others,
              "nontarget_local_max": nt_local, "target_angle_in_hold_deg": [float(ang[hold].min()), float(ang[hold].max())] if hold.any() else None,
              "target_local_outside_hold_max": tgt_out, "nondescendant_world_max_m": nd_world,
              "nondescendant_world_rot_max": nd_rot, "subtree_world_outside_hold_max_m": sub_out_world,
              "elbow_disp_hold_m": [float(elbow[hold].min()), float(elbow[hold].max())], "expected_elbow_disp_m": expd,
              "hold_samples": int(hold.sum()), "hold_integer_frames": int(hold_int.sum()),
              "descendant_world_change_max_m": _world_diff(e1, n0, desc, hold)[0] if desc else None,
              "channel_digest": e1.meta["channel_digest"]}
        e1ok = (key_err <= T["inverse_key_value"] and outside_bitwise and not others and nt_local <= T["local_sample_untouched"]
                and hold.any() and abs(float(ang[hold].min()) - C.E1_ANGLE_DEG) <= 0.01
                and abs(float(ang[hold].max()) - C.E1_ANGLE_DEG) <= 0.01 and tgt_out <= T["local_sample_untouched"]
                and nd_world <= T["world_untouched_m"] and nd_rot <= T["world_matrix_untouched"]
                and sub_out_world <= T["world_untouched_m"]
                and float(np.max(np.abs(elbow[hold] - expd))) <= T["e1_disp_abs_err_m"]
                and float(elbow[hold].min()) >= T["e1_min_visible_disp_m"])
        mm["ok"] = bool(e1ok)
        m[tag] = mm
        if tag == "e1_reopen":
            mem = Cond(d, "e1_mem")
            m["E4_e1"] = {"digest_equal_in_memory": e1.meta["channel_digest"] == e1.meta["in_memory_digest_before_save"],
                          "samples_identical_to_in_memory": _arrays_identical(e1, mem)}
    e1_numeric = m["e1_mem"]["ok"] and m["e1_reopen"]["ok"]
    # --- E1 render evidence
    render = None
    if render_dir is not None and case.get("render"):
        rd = Path(render_dir)/cid
        lv, views = T["render_diff_level"], [v.replace(" ", "_") for v in case["render_views"]]
        lo, hi = case["e1_hold_interval"]
        rows = []
        for f in case["render_frames"]:
            px = {v: _pix(rd/"n0"/f"{v}_{f:04d}.png", rd/"e1"/f"{v}_{f:04d}.png", lv) for v in views}
            null = {v: _pix(rd/"orig"/f"{v}_{f:04d}.png", rd/"n0"/f"{v}_{f:04d}.png", lv) for v in views}
            inv = {v: _pix(rd/"n0"/f"{v}_{f:04d}.png", rd/"e3_e1"/f"{v}_{f:04d}.png", lv) for v in views}
            e2p = {v: _pix(rd/"n0"/f"{v}_{f:04d}.png", rd/"e2p"/f"{v}_{f:04d}.png", lv) for v in views}
            e2m = {v: _pix(rd/"n0"/f"{v}_{f:04d}.png", rd/"e2m"/f"{v}_{f:04d}.png", lv) for v in views}
            e2inv = {v: _pix(rd/"n0"/f"{v}_{f:04d}.png", rd/"e3_e2p"/f"{v}_{f:04d}.png", lv) for v in views}
            rows.append({"frame": f, "in_hold": lo <= f < hi, "e1_vs_n0": px, "null_orig_vs_n0": null,
                         "e3_e1_vs_n0": inv, "e2p_vs_n0": e2p, "e2m_vs_n0": e2m, "e3_e2p_vs_n0": e2inv})
        in_ok = all(max(r["e1_vs_n0"].values()) >= T["render_min_changed_px"] for r in rows if r["in_hold"])
        out_ok = all(max(r["e1_vs_n0"].values()) <= T["render_max_changed_px_outside"] for r in rows if not r["in_hold"])
        null_ok = all(max(r["null_orig_vs_n0"].values()) == 0 for r in rows)
        render = {"frames": rows, "e1_visible_in_hold": in_ok, "e1_identical_outside_hold": out_ok, "null_render_identical": null_ok,
                  "e3_e1_max_changed_px": max(max(r["e3_e1_vs_n0"].values()) for r in rows),
                  "e3_e2p_max_changed_px": max(max(r["e3_e2p_vs_n0"].values()) for r in rows),
                  "e2p_changed_frames": [r["frame"] for r in rows if max(r["e2p_vs_n0"].values()) > 0],
                  "e2m_changed_frames": [r["frame"] for r in rows if max(r["e2m_vs_n0"].values()) > 0]}
        m["render"] = render
    g["N0"] = bool(ok and (render is None or render["null_render_identical"]))
    g["E1"] = bool(e1_numeric and (render is None or (render["e1_visible_in_hold"] and render["e1_identical_outside_hold"])))
    # --- E2
    e2_ok = True
    for tag in ("e2p_mem", "e2p_reopen", "e2m_mem", "e2m_reopen"):
        e2 = Cond(d, tag)
        delta = case["e2_offset"] * (1 if tag.startswith("e2p") else -1)
        f2, q2, _ = _target(e2)
        t0 = n0.meta["timing"]
        try:
            want = C.shift_keys(frames0, case["e2_group"], delta, t0["frame_start"], t0["frame_end"])
            frames_ok = [float(np.float32(x)) for x in want] == f2
        except C.ContractViolation as exc:
            want, frames_ok = str(exc), False
        values_ok = q2 == q0
        others = _channels_equal_except(e2, n0, tpath)
        it = e2.idx[C.TARGET_BONE]
        pred = np.array([[C.step_value(f2, [q[i] for q in q2], t) for i in range(4)] for t in times])
        model_err = _mx(pred - e2.a["local_quat"][:, it])
        base_pred = np.array([[C.step_value(frames0, [q[i] for q in q0], t) for i in range(4)] for t in times])
        changed = np.any(np.abs(pred - base_pred) > 0, axis=1)
        integer = times == np.floor(times)
        nt_local = _local_diff(e2, n0, [b for b in all_b if b != C.TARGET_BONE])
        nd_world, nd_rot = _world_diff(e2, n0, nondesc)
        mm = {"delta": delta, "frames_ok": frames_ok, "new_frames": f2, "expected": want, "values_bitwise": values_ok,
              "other_channels_changed": others, "step_model_err": model_err, "nontarget_local_max": nt_local,
              "nondescendant_world_max_m": nd_world, "nondescendant_world_rot_max": nd_rot,
              "changed_integer_frames": [float(t) for t in times[changed & integer]],
              "changed_half_frames": [float(t) for t in times[changed & ~integer]],
              "native_offset_s": None if "timebase" not in case else delta/case["timebase"]["native_fps"],
              "blender_offset_s": delta/t0["effective_fps"]}
        okk = (frames_ok and values_ok and not others and model_err <= T["e2_model"] and nt_local <= T["local_sample_untouched"]
               and nd_world <= T["world_untouched_m"] and nd_rot <= T["world_matrix_untouched"])
        if tag.startswith("e2p"):
            okk = okk and bool((changed & integer).any())
        mm["ok"] = bool(okk)
        e2_ok = e2_ok and okk
        m[tag] = mm
        if tag.endswith("reopen"):
            mem = Cond(d, tag.replace("reopen", "mem"))
            m["E4_" + tag[:3]] = {"digest_equal_in_memory": e2.meta["channel_digest"] == e2.meta["in_memory_digest_before_save"],
                                  "samples_identical_to_in_memory": _arrays_identical(e2, mem)}
    g["E2"] = bool(e2_ok)
    # --- E3
    e3 = Cond(d, "e3_e1_reopen")
    f3, q3, _ = _target(e3)
    canon = np.array([C.qcanon(q) for q in q3]) - np.array([C.qcanon(q) for q in q0])
    flips = sum(1 for a, b in zip(q3, q0) if np.dot(a, b) < 0)
    raw = _mx(np.array(q3) - np.array(q0))
    w3, r3 = _world_diff(e3, n0, all_b)
    others = _channels_equal_except(e3, n0, tpath)
    m["E3_e1"] = {"key_err_canonical": _mx(canon), "key_err_raw": raw, "sign_flips": flips, "frames_equal": f3 == frames0,
                  "digest_equal_n0": e3.meta["channel_digest"] == n0.meta["channel_digest"],
                  "bitwise_equal_keys": q3 == q0, "world_max_m": w3, "world_rot_max": r3, "other_channels_changed": others,
                  "samples_bitwise_identical_n0": _arrays_identical(e3, n0)}
    e3a = _mx(canon) <= T["inverse_key_value"] and f3 == frames0 and w3 <= T["inverse_world_m"] and not others
    e3b_c = Cond(d, "e3_e2p_reopen")
    m["E3_e2p"] = {"digest_equal_n0": e3b_c.meta["channel_digest"] == n0.meta["channel_digest"],
                   "keys_bitwise": not _channels_equal_except(e3b_c, n0, None),
                   "samples_bitwise_identical_n0": _arrays_identical(e3b_c, n0)}
    e3b = m["E3_e2p"]["keys_bitwise"] and m["E3_e2p"]["samples_bitwise_identical_n0"]
    g["E3"] = bool(e3a and e3b)
    e4 = [v for k, v in m.items() if k.startswith("E4_")]
    g["E4"] = bool(all(x["digest_equal_in_memory"] and x["samples_identical_to_in_memory"] for x in e4)
                   and m["e1_reopen"]["ok"] and m["e2p_reopen"]["ok"] and m["e2m_reopen"]["ok"])
    # --- FBX (separate)
    res["fbx"] = analyze_fbx(d, case, n0)
    # first failing layer
    layer = None
    if not g["E1"] or not g["E2"]:
        if (m["e1_mem"]["key_err"] > T["inverse_key_value"] or m["e1_mem"]["other_channels_changed"]
                or not m["e2p_mem"]["frames_ok"] or not m["e2p_mem"]["values_bitwise"]):
            layer = "action_representation"
        elif not m["e1_mem"]["ok"]:
            layer = "rig_hierarchy"
        elif not (m["e2p_mem"]["ok"] and m["e2m_mem"]["ok"]):
            layer = "timebase"
        else:
            layer = "serialization"
    elif not g["E3"] or not g["E4"]:
        layer = "serialization"
    res["first_failing_layer"] = layer
    return res


def analyze_fbx(d, case, n0):
    T = C.TOL
    out = {"status": {}, "label": "separate gate; does not change the Blender CASE"}
    metas = {}
    for cond in ("n0", "e1", "e2p"):
        p = d/f"fbx_{cond}_meta.json"
        if not p.exists():
            f = d/f"fbx_{cond}.json"
            out["status"][cond] = json.loads(f.read_text()).get("status") if f.exists() else "NOT_RUN"
            continue
        metas[cond] = json.loads(p.read_text())
        out["status"][cond] = metas[cond]["status"]
    if set(metas) != {"n0", "e1", "e2p"}:
        out["ok"] = False
        return out
    fb = {c: Cond(d, f"fbx_{c}") for c in metas}
    names = [b for b in n0.bones if b in fb["n0"].idx]
    integer = n0.times == np.floor(n0.times)
    half = ~integer
    rt, rtr = _world_diff(fb["n0"], n0, names, integer)
    rt_half, _ = _world_diff(fb["n0"], n0, names, half)
    nondesc = case["non_target_bones"]
    sub = [C.TARGET_BONE] + case["descendants"]
    e1n0 = {k: Cond(d, k) for k in ("e1_reopen", "e2p_reopen")}
    eff = {}
    for cond, bl in (("e1", "e1_reopen"), ("e2p", "e2p_reopen")):
        nt, _ = _world_diff(fb[cond], fb["n0"], nondesc, integer)
        ia, ib = _sel(fb[cond], sub), _sel(n0, sub)
        dfbx = fb[cond].a["world_tail"][integer][:, ia] - fb["n0"].a["world_tail"][integer][:, ia]
        dbl = e1n0[bl].a["world_tail"][integer][:, ib] - n0.a["world_tail"][integer][:, ib]
        eff[cond] = {"nontarget_max_m": nt, "effect_err_max_m": _mx(dfbx - dbl), "effect_blender_max_m": _mx(dbl)}
    m0 = metas["n0"]
    out.update({
        "rt_parity_integer_max_m": rt, "rt_parity_integer_rot_max": rtr, "rt_half_frame_max_m": rt_half,
        "edits": eff, "imported_timing": m0["imported_timing"], "blender_timing": n0.meta["timing"],
        "timing_preserved": {k: m0["imported_timing"][k] == n0.meta["timing"][k] for k in n0.meta["timing"]},
        "imported_actions": m0["imported_actions"], "matched_action": m0["matched_action"],
        "imported_interpolation": m0.get("imported_interpolation"), "target_keys": m0.get("target_keys"),
        "missing_bones": m0["missing_bones"], "extra_bones": m0["extra_bones"],
        "imported_armature_scale": m0["imported_armature_scale"],
        "local_quat_target_max_diff_vs_blender": _mx(fb["n0"].a["local_quat"][integer][:, fb["n0"].idx[C.TARGET_BONE]]
                                                     - n0.a["local_quat"][integer][:, n0.idx[C.TARGET_BONE]])})
    out["ok"] = bool(rt <= T["fbx_rt_parity_m"] and not m0["missing_bones"]
                     and all(e["nontarget_max_m"] <= T["fbx_nontarget_m"] and e["effect_err_max_m"] <= T["fbx_effect_m"]
                             for e in eff.values()))
    return out


def determinism(p1, p2):
    diffs, n = [], 0
    for j in sorted(Path(p1).rglob("*.npz")):
        rel = j.relative_to(p1)
        k = Path(p2)/rel
        n += 1
        if not k.exists():
            diffs.append(f"missing {rel}")
            continue
        a, b = np.load(j), np.load(k)
        if set(a.files) != set(b.files) or any(not np.array_equal(a[x], b[x]) for x in a.files):
            diffs.append(str(rel))
        ja, jb = j.with_suffix(".json"), k.with_suffix(".json")
        if json.loads(ja.read_text()).get("channel_digest") != json.loads(jb.read_text()).get("channel_digest"):
            diffs.append(f"digest {rel}")
    return {"compared_conditions": n, "differences": diffs, "ok": n > 0 and not diffs}


def analyze_synthetic(d, case):
    res = analyze_case(d, case)
    x = {}
    n0 = Cond(d, "n0_reopen")
    s = Cond(d, "x_sign_flip")
    i = s.idx["sibling"]
    x["sign_flip"] = {"local_channel_max_diff": _mx(s.a["local_quat"][:, i] - n0.a["local_quat"][:, i]),
                      "world_max_m": _world_diff(s, n0, s.bones)[0], "world_rot_max": _world_diff(s, n0, s.bones)[1],
                      "classification": None}
    x["sign_flip"]["classification"] = ("channel-inequal, rotation-equivalent"
                                        if x["sign_flip"]["local_channel_max_diff"] > 0.1 and x["sign_flip"]["world_max_m"] <= 1e-6
                                        else "unexpected")
    probe = json.loads((d.parent.parent/"probe_report.json").read_text())["synthetic_extras"] if (d.parent.parent/"probe_report.json").exists() else {}
    for tag in ("x_negative_forced", "x_collision_forced"):
        c = Cond(d, tag)
        f, q, _ = _target(c)
        it = c.idx[C.TARGET_BONE]
        pred = np.array([[C.step_value(f, [qq[k] for qq in q], t) for k in range(4)] for t in c.times])
        x[tag] = {"new_frames": f, "step_model_err": _mx(pred - c.a["local_quat"][:, it]),
                  "keys_outside_domain": [v for v in f if v < 1 or v > 25],
                  "duplicate_key_times": sorted({v for v in f if f.count(v) > 1}),
                  "frame1_value_changed": bool(np.any(c.a["local_quat"][0, it] != n0.a["local_quat"][0, it]))}
    for tag, win in (("x_e1_start", list(range(5))), ("x_e1_end", list(range(4, 9)))):
        c = Cond(d, tag)
        it = c.idx[C.TARGET_BONE]
        lo, hi = C.hold_interval(C.SYN_FRAMES, win)
        hold = (c.times >= lo) & (c.times < hi)
        ang = _qang(c.a["local_quat"][:, it], n0.a["local_quat"][:, it])
        x[tag] = {"hold": [lo, hi], "angle_in_hold": [float(ang[hold].min()), float(ang[hold].max())],
                  "angle_outside_max": float(ang[~hold].max()) if (~hold).any() else None,
                  "nondescendant_world_max_m": _world_diff(c, n0, case["non_target_bones"])[0]}
    # missing key on sibling: CONSTANT hold across the missing key time
    p = 'pose.bones["sibling"].rotation_quaternion'
    comps = [n0.ch[(p, k)] for k in range(4)]
    k13 = int(np.argmin(np.abs(n0.times - C.SYN_MISSING["sibling"])))
    x["missing_key"] = {"sibling_keys": comps[0]["frames"],
                        "value_at_13_equals_key_at_10": bool(np.allclose(n0.a["local_quat"][k13, n0.idx["sibling"]],
                                                                         [comps[k]["values"][comps[0]["frames"].index(10.0)] for k in range(4)], atol=0))}
    x["contract_refusals"] = {k: probe.get(k) for k in ("negative_contract", "collision_contract")}
    x["ok"] = (x["sign_flip"]["classification"] == "channel-inequal, rotation-equivalent" and x["missing_key"]["value_at_13_equals_key_at_10"]
               and all(str(x["contract_refusals"][k]).startswith("REFUSED") for k in x["contract_refusals"])
               and all(abs(x[t]["angle_in_hold"][0] - C.E1_ANGLE_DEG) <= 0.01 and abs(x[t]["angle_in_hold"][1] - C.E1_ANGLE_DEG) <= 0.01
                       and (x[t]["angle_outside_max"] is None or x[t]["angle_outside_max"] <= 1e-4)
                       and x[t]["nondescendant_world_max_m"] <= C.TOL["world_untouched_m"] for t in ("x_e1_start", "x_e1_end")))
    res["extras"] = x
    return res


def analyze(o):
    decl_bytes = Path(o.declaration).read_bytes()
    if hashlib.sha256(decl_bytes).hexdigest() != o.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    p1, p2 = Path(o.probe_dir), Path(o.repeat_dir)
    probe = json.loads((p1/"probe_report.json").read_text())
    rep = {"schema": C.SCHEMA + ":analysis", "declaration_sha256": o.declaration_sha, "cases": {}}
    rep["synthetic"] = analyze_synthetic(p1/"synthetic"/"case", decl["synthetic"])
    for cid, case in decl["cases"].items():
        rep["cases"][cid] = analyze_case(p1/cid, case, o.render_dir)
        rep["cases"][cid]["start_negative_shift"] = probe["cases"][cid].get("start_negative_shift")
    rep["determinism"] = determinism(p1, p2)
    real_ids = [c for c, v in decl["cases"].items() if v["kind"] != "legacy_smoke"]
    syn_ok = all(rep["synthetic"]["gates"].values()) and rep["synthetic"]["extras"]["ok"]
    pre = {"preflight_ok": True, "determinism_ok": rep["determinism"]["ok"], "synthetic_ok": syn_ok,
           "n0_ok": {c: rep["cases"][c]["gates"]["N0"] for c in real_ids},
           "real": {c: {k: v for k, v in rep["cases"][c]["gates"].items() if k != "N0"} for c in real_ids},
           "first_failing_layer": {c: rep["cases"][c]["first_failing_layer"] for c in real_ids}}
    case, why = C.decide_case(pre)
    rep["decision_inputs"] = pre
    rep["CASE"], rep["CASE_reason"] = case, why
    rep["fbx_summary"] = {c: rep["cases"][c]["fbx"].get("ok") for c in decl["cases"]}
    rep["fbx_summary"]["synthetic"] = rep["synthetic"]["fbx"].get("ok")
    rep["legacy_smoke"] = {"gates": rep["cases"].get("legacy_smoke", {}).get("gates"), "pooled": False}
    s = wjson(out/"analysis_report.json", rep)
    print(json.dumps({"CASE": case, "reason": why, "sha256": s, "fbx": rep["fbx_summary"],
                      "gates": {c: rep["cases"][c]["gates"] for c in rep["cases"]}, "synthetic": rep["synthetic"]["gates"],
                      "synthetic_extras_ok": rep["synthetic"]["extras"]["ok"], "determinism": rep["determinism"]["ok"]}, indent=1))
    return 0


# --- compose ------------------------------------------------------------------------------------------------------
def compose(o):
    from PIL import Image, ImageDraw
    decl = json.loads(Path(o.declaration).read_text())
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rd = Path(o.render_dir)
    made = []
    for cid, case in decl["cases"].items():
        if not case.get("render"):
            continue
        lo, hi = case["e1_hold_interval"]
        fr = case["render_frames"]
        inside = [f for f in fr if lo <= f < hi]
        pick = [inside[0], inside[len(inside)//2], [f for f in fr if not lo <= f < hi][0]]
        views = [v.replace(" ", "_") for v in case["render_views"]]
        for name, conds in (("e1", ("n0", "e1", "e3_e1")), ("e2", ("n0", "e2p", "e2m", "e3_e2p"))):
            frames = pick if name == "e1" else [f for f in fr if case["e2_group_frames"][0] - 2 <= f <= case["e2_group_frames"][-1] + 2][:8]
            s = 240
            cols = len(conds) + (1 if name == "e1" else 0)
            sheet = Image.new("RGB", (cols*s, len(frames)*len(views)*s + 20), "white")
            dr = ImageDraw.Draw(sheet)
            dr.text((4, 4), f"W106 {cid} {name}: " + " | ".join(conds) + (" | |E1-N0|" if name == "e1" else "")
                    + "  (scratch copies, evaluation only)", fill="black")
            y = 20
            for f in frames:
                for v in views:
                    ims = [Image.open(rd/cid/c/f"{v}_{f:04d}.png").convert("RGB").resize((s, s)) for c in conds]
                    for k, im in enumerate(ims):
                        sheet.paste(im, (k*s, y))
                        ImageDraw.Draw(sheet).text((k*s + 4, y + 4), f"{conds[k]} f{f} {v}", fill="black")
                    if name == "e1":
                        a = np.asarray(ims[0], dtype=np.int16)
                        b = np.asarray(ims[1], dtype=np.int16)
                        diff = np.clip(np.max(np.abs(a - b), axis=-1)*4, 0, 255).astype(np.uint8)
                        sheet.paste(Image.fromarray(diff).convert("RGB"), (len(conds)*s, y))
                    y += s
            p = out/f"{cid}_{name}.jpg"
            sheet.save(p, quality=90)
            made.append(p.name)
    print(made)
    return 0


# --- fbx-attribution (post-hoc description of the FBX baseline drift; never a gate, never re-exports) -----------
def _rot_angle_deg(ma, mb):
    ra = ma[:3, :3] / np.linalg.norm(ma[:3, :3], axis=0)
    rb = mb[:3, :3] / np.linalg.norm(mb[:3, :3], axis=0)
    return float(np.degrees(np.arccos(np.clip((np.trace(ra.T @ rb) - 1)/2, -1, 1))))


def fbx_attribution(o):
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    decl = json.loads(Path(o.declaration).read_text())
    rep = {"schema": C.SCHEMA + ":fbx_attribution", "label": "POST-HOC descriptive decomposition of the unedited FBX "
           "round-trip drift (N0 Blender vs N0 FBX import); no exporter/importer setting was varied", "cases": {}}
    for cid in list(decl["cases"]) + ["synthetic"]:
        d = Path(o.probe_dir)/(cid if cid != "synthetic" else "synthetic/case")
        a, b = Cond(d, "n0_reopen"), Cond(d, "fbx_n0")
        idx = [b.idx[n] for n in a.bones]
        it = np.where(a.times == np.floor(a.times))[0]
        drift = np.maximum(np.linalg.norm(a.a["world_head"][it] - b.a["world_head"][it][:, idx], axis=-1),
                           np.linalg.norm(a.a["world_tail"][it] - b.a["world_tail"][it][:, idx], axis=-1))
        obj = np.array([_rot_angle_deg(a.a["object_matrix"][i], b.a["object_matrix"][i]) for i in it])
        arm = []
        for k, i in enumerate(it):
            ia, ib = np.linalg.inv(a.a["object_matrix"][i]), np.linalg.inv(b.a["object_matrix"][i])
            pa = (ia[:3, :3] @ a.a["world_tail"][i].T).T + ia[:3, 3]
            pb = (ib[:3, :3] @ b.a["world_tail"][i][idx].T).T + ib[:3, 3]
            arm.append(float(np.linalg.norm(pa - pb, axis=-1).max()) * float(np.linalg.norm(a.a["object_matrix"][i][:3, 0])))
        arm = np.array(arm)
        frames = a.times[it]
        per_bone = drift.max(0)
        worst = np.argsort(-per_bone)[:5]
        rep["cases"][cid] = {
            "integer_frames": int(len(it)), "world_drift_max_m": float(drift.max()),
            "frames_world_drift_gt_1mm": int((drift.max(1) > 1e-3).sum()),
            "object_rotation_drift_deg": {"max": float(obj.max()), "median": float(np.median(obj)),
                                          "frames_gt_0.01deg": int((obj > 0.01).sum())},
            "armature_space_drift_m": {"max": float(arm.max()), "median": float(np.median(arm)),
                                       "frames_gt_1mm": int((arm > 1e-3).sum())},
            "object_location_max_diff": float(np.abs(a.a["object_matrix"][it][:, :3, 3] - b.a["object_matrix"][it][:, :3, 3]).max()),
            "worst_bones_m": {a.bones[i]: float(per_bone[i]) for i in worst},
            "worst_frames": [float(frames[i]) for i in np.argsort(-drift.max(1))[:5]]}
    s = wjson(out/"fbx_attribution.json", rep)
    print(s)
    print(json.dumps({k: {x: v[x] for x in ("world_drift_max_m", "frames_world_drift_gt_1mm", "object_rotation_drift_deg",
                                            "armature_space_drift_m")} for k, v in rep["cases"].items()}, indent=1))
    return 0


# --- manifest -----------------------------------------------------------------------------------------------------
def manifest(o):
    base = Path(o.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != "evidence_manifest.json" and "explore" not in p.parts:
            files[p.relative_to(base).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    ver = json.loads(Path(o.verify).read_text())
    w105 = Path(o.w105)
    unchanged = {rel: sha256(w105/rel) == v["sha256"] for rel, v in ver["w105_files"].items()}
    unchanged[LEGACY] = sha256(ROOT/LEGACY) == ver["legacy_smoke"]["sha256"]
    rep = {"schema": C.SCHEMA + ":manifest", "starting_head": o.starting_head, "declaration_sha256": o.declaration_sha,
           "originals_unchanged": unchanged, "all_originals_unchanged": all(unchanged.values()), "files": files}
    s = wjson(base/"evidence_manifest.json", rep)
    print(s, len(files), rep["all_originals_unchanged"])
    return 0 if rep["all_originals_unchanged"] else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("verify", "declare", "analyze", "compose", "fbx-attribution", "manifest"))
    for name in ("w105", "out_dir", "verify", "inventory", "starting_head", "declaration", "declaration_sha",
                 "probe_dir", "repeat_dir", "render_dir", "base"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args()
    return {"verify": verify, "declare": declare, "analyze": analyze, "compose": compose,
            "fbx-attribution": fbx_attribution, "manifest": manifest}[o.mode](o)


if __name__ == "__main__":
    sys.exit(main())
