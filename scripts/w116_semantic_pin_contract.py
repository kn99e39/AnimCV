#!/usr/bin/env python3
"""W116: Semantic Constraint Architecture — World-Space Joint Position Pin feasibility (evidence only, NON-PRODUCTION).

Stages (each refuses an existing output directory):
  preflight  instruction/HEAD/lock/historical identity, input hashes; builds the parent Animation States (W105 real_s0_only
             FIXED copy; declared synthetic two-bone chain) and runs the Blender probe on them (no edit exists yet)
  calibrate  instrument calibration on the UNEDITED parents only: Python FK model vs Blender, semantic rig binding
             verification, coordinate probes, synthetic ground-truth reachability of the declared synthetic targets
  declare    frozen declaration: constants, edits, requests, controls, predictions, gates, CASE rule (no outcome read)
  build      one independent resolver build: requests -> validation -> control (edit only) and candidate (edit + pin)
             Animation States for Blender; resolver statuses and internal residuals
  evaluate   Blender evaluation of a build's states (independent evaluator input)
  commit     independent evaluator + atomic commit/reject + lineage programmes + reversal (Blender re-evaluation)
  render     visual review renders of declared frames (parent / unpinned control / pinned candidate)
  decide     frozen gates -> CASE
  manifest   evidence manifest
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w116_semantic_pin_base as S  # noqa: E402

BATCH = 116
INSTRUCTION_BLOB = "3adfda964525cdd02bd315b07249087a102dd790"
STARTING_HEAD = "88ef04ea2c7db3cc6914cabe7c7123060cf47ddf"
BRANCH = "arch/single_frame_first"
LOCK = Path("C:/Users/dna10/animcv-automation/state/active_batch.json")
BLENDER = Path("C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
W102, W105, W115 = (Path(f"C:/Users/dna10/animcv-w{n}") for n in (102, 105, 115))
W115_MANIFEST_SHA = "5988f4fb8c5f848aaa23b1f0eb4500e73176d234b8b399201ffe56c539de1d2d"
W105_MANIFEST_SHA = "cb0eca93c7847a70d497bd36c57856bd8295f3d8e3a7a72913ebf876ebe96899"
W105_FIXED_DIGEST_PREFIX = "4d168d95234c"
INPUTS = {  # name: (path, expected sha256 or "w105"/"w115" manifest-bound)
    "w102_observations_predictions": (W102 / "results/historical_video/observations_predictions.npz",
                                      "6784357f7c6145e4b60e2e9201e32accd3ae00001486283a1e76f0adae61e7bd"),
    "repo_production_demo_clip": (ROOT / "examples/production_demo/source.mp4",
                                  "0dda8a5f16c67992330a1a8534cf0bfefeb5e0f2aec438577a9e1dbace773275"),
    "w105_evidence_manifest": (W105 / "evidence_manifest.json", W105_MANIFEST_SHA),
    "w105_video_cases": (W105 / "results/build/video_cases.json", "w105"),
    "w105_video_sources": (W105 / "results/build/video_sources.npz", "w105"),
    "w105_render_clip_audit": (W105 / "results/render_clip_audit.json", "w105"),
    "w105_rig_skinned_fbx": (ROOT / "examples/skinned_mannequin/BaseRig_Skinned.fbx",
                             "c2947ba7f0d088aa47dbf0e04729a991699f24c649acce252b7b902740bb61d7"),
    "w105_rig_rest": (ROOT / "examples/e2e_demo/baserig_blender_rest.json",
                      "1dcdc8268777bd433d1b73d56acef9a3f67b30f16456a80618d76fb6e70f9358"),
    "w115_evidence_manifest": (W115 / "evidence_manifest.json", W115_MANIFEST_SHA),
    "w115_decision": (W115 / "results/decision/decision.json", "w115"),
    "w115_posed": (W115 / "results/rig_replay/posed.json", "w115"),
    "w115_restq": (W115 / "results/preflight/restq.json", "w115"),
    "w115_blender_compare": (W115 / "results/blender_compare/blender_compare.json", "w115"),
}
HISTORICAL_WORKLOGS = {113: "docs/113_WORKLOG_ANIMATION_BASE_CONTRACT_FEASIBILITY.md",
                       114: "docs/114_WORKLOG_SOURCE_PRESERVING_ANIMATION_BASE_EDIT_AUTHORITY.md",
                       115: "docs/115_WORKLOG_ANIMATION_BASE_VALIDITY_REPEATABILITY_RESOLUTION_AUTHORITY.md"}
REAL_CH = ["__object__", "spine_01", "thigh_l", "calf_l", "thigh_r", "calf_r", "upperarm_l", "lowerarm_l"]
ARM = ["upperarm_l", "lowerarm_l"]
WINDOWS = {"low_evidence_109_118": list(range(109, 119)), "low_evidence_154_181": list(range(154, 182))}
REAL_BINDING_CANDIDATES = {  # declared rig points; the FIRST candidate is the canonical evaluated point
    "left_wrist": [("lowerarm_l", "tail"), ("hand_l", "head")],
    "left_elbow": [("upperarm_l", "tail"), ("lowerarm_l", "head")],
    "left_shoulder": [("clavicle_l", "tail"), ("upperarm_l", "head")],
    "right_wrist": [("lowerarm_r", "tail"), ("hand_r", "head")],
    "right_elbow": [("upperarm_r", "tail"), ("lowerarm_r", "head")],
    "right_shoulder": [("clavicle_r", "tail"), ("upperarm_r", "head")],
    "pelvis_root": [("spine_01", "head")],
    "thorax": [("spine_05", "tail"), ("neck_01", "head")],     # declared to test fail-safe: not one rig point
    "left_hand": [("hand_l", "head"), ("hand_l", "tail")],      # declared to test fail-safe: name match, two points
    "nose": [],                                                 # no rig point
}
SYN_BINDING_CANDIDATES = {
    "syn_wrist": [("lower", "tail")], "syn_elbow": [("upper", "tail"), ("lower", "head")],
    "syn_shoulder": [("base", "tail"), ("upper", "head")], "syn_hand": [("lower", "tail"), ("lower", "head")],
    "syn_finger": [("finger", "tail")], "syn_nose": [],
}
SYN_SPEC = {"rig_id": "W116_SYN_TWO_BONE", "object": {"location": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0],
                                                     "rest_rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "bones": [{"name": "base", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 0.5], "roll": 0.0},
                      {"name": "upper", "parent": "base", "head": [0.0, 0.0, 0.5], "tail": [0.3, 0.0, 0.5], "roll": 0.0,
                       "connect": True},
                      {"name": "lower", "parent": "upper", "head": [0.3, 0.0, 0.5], "tail": [0.55, 0.0, 0.5], "roll": 0.0,
                       "connect": True}]}
SYN_CH = ["__object__", "base", "upper", "lower"]
SYN_FRAMES = list(range(20))
SYN_BASE_AMPL_DEG = 15.0
SYN_LOWER_BEND_DEG = 60.0


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=1, sort_keys=True, allow_nan=False))


def load(path):
    return json.loads(Path(path).read_text())


def new_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=False)
    return p


def blender(*args, log=None):
    r = subprocess.run([str(BLENDER), "-b", "-t", "2", "--python-exit-code", "1", "-P", *map(str, args)], cwd=ROOT,
                       capture_output=True, text=True)
    if log:
        Path(log).write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"STOP: Blender stage failed ({r.returncode}); see {log}")
    return r


def runtime():
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}


def source_guard():
    return S.SourceGuard({k: v[0] for k, v in INPUTS.items()} | {"w105_rig_rest_json": INPUTS["w105_rig_rest"][0]})


# ------------------------------------------------------------------------------------------- parent states ---
def real_parent_state(vc_path):
    vc = load(vc_path)["cases"]["real_clip"]
    vals = {c: [] for c in REAL_CH}
    for fr in vc["frames"]:
        m = fr["modes"]["FIXED"]
        vals["__object__"].append(fr["object_rotation_xyzw"]["FIXED"])
        ex = dict(fr["extra_bone_locals_xyzw"]["FIXED"])
        ex.update(m.get("extra_bone_locals_xyzw", {}))  # renderer key order: mode locals overwrite extra locals
        for b in ("spine_01", "thigh_l", "calf_l", "thigh_r", "calf_r"):
            vals[b].append(ex[b])
        vals["upperarm_l"].append(m["proximal_local_xyzw"])
        vals["lowerarm_l"].append(m["distal_local_xyzw"])
    src = {"kind": "W105_video_cases_real_clip_FIXED (M_Main S0-only rig configuration)", "path": str(vc_path),
           "sha256": S.sha_file(vc_path), "sequence_id": vc["sequence_id"], "fps_source": vc["fps_sampled"],
           "time_s": [fr["time_s"] for fr in vc["frames"]], "rig_identity": "W105 BaseRig_Skinned (known output rig)",
           "video_subject_identity": "UNKNOWN (per-frame tracked subject continuity not verified)",
           "clip_sha256": INPUTS["repo_production_demo_clip"][1]}
    return S.make_state(None, "BaseRig", [fr["frame_index"] for fr in vc["frames"]], REAL_CH, vals,
                        {"kind": "ROOT_COPY_OF_SOURCE_HYPOTHESIS"}, src)


def syn_parent_state():
    vals = {c: [] for c in SYN_CH}
    for f in SYN_FRAMES:
        vals["__object__"].append([0.0, 0.0, 0.0, 1.0])
        vals["base"].append(S.q_axis_angle([1, 0, 0], math.radians(SYN_BASE_AMPL_DEG) * math.sin(math.pi * f / 20)).tolist())
        vals["upper"].append([0.0, 0.0, 0.0, 1.0])
        vals["lower"].append(S.q_axis_angle([0, 0, 1], math.radians(SYN_LOWER_BEND_DEG)).tolist())
    src = {"kind": "DECLARED_SYNTHETIC_SEQUENCE", "rig_identity": "W116_SYN_TWO_BONE (known synthetic rig)",
           "video_subject_identity": "NOT_APPLICABLE"}
    return S.make_state(None, SYN_SPEC["rig_id"], SYN_FRAMES, SYN_CH, vals, {"kind": "ROOT_SYNTHETIC"}, src)


def rig_from(rest_json, probe_rig):
    r = load(rest_json)
    bones = {n: {"parent": b["parent"], "matrix_local": b["matrix_local"], "head": b["head"], "tail": b["tail"]}
             for n, b in r["bones"].items()}
    return S.Rig({"rig_id": r["rig_id"], "bones": bones, "object": {"location": probe_rig["location"], "scale": probe_rig["scale"],
                                                                   "rest_rotation_wxyz": probe_rig["rest_rotation_wxyz"]}})


def syn_rig_from(probe):
    bones = {n: {"parent": b["parent"], "matrix_local": b["matrix_local"], "head": b["head_local"], "tail": b["tail_local"]}
             for n, b in probe["rest"].items()}
    return S.Rig({"rig_id": SYN_SPEC["rig_id"], "bones": bones, "object": SYN_SPEC["object"]})


def states_file(path, states, rig):
    dump(path, {"rig": rig, "states": {n: {k: st[k] for k in ("state_id", "frames", "channels", "values")}
                                       | ({"armature_location": st["armature_location"]} if "armature_location" in st else {})
                                       for n, st in states.items()}})


# ---------------------------------------------------------------------------------------------- preflight ---
def cmd_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"schema": "w116_preflight_v1", "stop": [], "runtime": runtime()}
    rep["head"] = git("rev-parse", "HEAD")
    rep["instruction_blob"] = git("rev-parse", "HEAD:docs/next_batch.md")
    rep["branch"] = git("rev-parse", "--abbrev-ref", "HEAD")
    rep["status_porcelain"] = git("status", "--porcelain")
    title = (ROOT / "docs/next_batch.md").read_text(encoding="utf-8").splitlines()[0]
    rep["instruction_title"] = title
    if rep["instruction_blob"] != INSTRUCTION_BLOB or not title.startswith("# BATCH 116"):
        rep["stop"].append("instruction is not the bound Batch 116 blob")
    if rep["head"] != STARTING_HEAD:
        rep["stop"].append("HEAD is not the bound starting commit")
    if rep["branch"] != BRANCH:
        rep["stop"].append("wrong branch")
    lock = load(LOCK) if LOCK.exists() else None
    rep["main_local_lock"] = lock
    if not lock or lock.get("batch") != BATCH or lock.get("instruction_blob") != INSTRUCTION_BLOB or lock.get("start_commit") != STARTING_HEAD:
        rep["stop"].append("Main-local lock is not this Batch 116 claim")
    hist = {}
    for n, wl in HISTORICAL_WORKLOGS.items():
        lines = (ROOT / wl).read_text(encoding="utf-8").rstrip().splitlines()
        hist[n] = {"marker": lines[-2:] == ["AUTOMATION_BATCH_STATUS: COMPLETE", f"BATCH_NUMBER: {n}"],
                   "blob_at_start": git("rev-parse", f"{STARTING_HEAD}:{wl}"), "blob_now": git("hash-object", str(ROOT / wl))}
        if not hist[n]["marker"] or hist[n]["blob_at_start"] != hist[n]["blob_now"]:
            rep["stop"].append(f"W{n} Worklog record not intact")
    rep["historical_worklogs"] = hist
    m105 = load(INPUTS["w105_evidence_manifest"][0])["files"]
    m115 = load(INPUTS["w115_evidence_manifest"][0])["files"]
    files = {}
    for name, (path, want) in INPUTS.items():
        got = S.sha_file(path) if Path(path).exists() else None
        if want == "w105":
            want = m105.get(Path(path).relative_to(W105).as_posix(), {}).get("sha256")
        elif want == "w115":
            want = m115.get(Path(path).relative_to(W115).as_posix(), {}).get("sha256")
        files[name] = {"path": str(path), "sha256": got, "expected": want, "ok": got is not None and got == want}
        if not files[name]["ok"]:
            rep["stop"].append(f"input {name} missing or changed")
    rep["inputs"] = files
    rep["historical_cases"] = {"w115_case": load(INPUTS["w115_decision"][0]).get("case")}
    if rep["historical_cases"]["w115_case"] != "D":
        rep["stop"].append("W115 frozen CASE is not D")
    audit = load(INPUTS["w105_render_clip_audit"][0])["files"]["real_clip/real_clip.blend"]["actions"]["W91 real_clip FIXED"]
    rep["w105_fixed_value_digest"] = audit["value_digest"]
    if not audit["value_digest"].startswith(W105_FIXED_DIGEST_PREFIX):
        rep["stop"].append("W105 FIXED digest not the published one")
    v0 = real_parent_state(INPUTS["w105_video_cases"][0])
    s0 = syn_parent_state()
    dump(out / "real_parent_state.json", v0)
    dump(out / "syn_parent_state.json", s0)
    dump(out / "syn_spec.json", SYN_SPEC)
    states_file(out / "real_parent_states.json", {"v0": v0}, "w105")
    states_file(out / "syn_parent_states.json", {"v0": s0}, "synthetic")
    rest = INPUTS["w105_rig_rest"][0]
    blender("scripts/w116_blender_stages.py", "--", "probe", "--rig", INPUTS["w105_rig_skinned_fbx"][0], "--rest", rest,
            "--states", out / "real_parent_states.json", "--out", out / "real_probe.json", log=out / "real_probe.log")
    blender("scripts/w116_blender_stages.py", "--", "probe", "--rig", f"synthetic:{out / 'syn_spec.json'}",
            "--states", out / "syn_parent_states.json", "--out", out / "syn_probe.json", "--probe-channels", "base,upper,lower",
            log=out / "syn_probe.log")
    rp = load(out / "real_probe.json")
    rep["probe"] = {"value_digest": rp["states"]["v0"]["value_digest"], "fcurves": rp["states"]["v0"]["fcurves"],
                    "rig": rp["rig"], "blender_version": rp["blender_version"]}
    rep["parent_reproduces_w105_fixed"] = rp["states"]["v0"]["value_digest"] == audit["value_digest"]
    rep["real_parent_state_id"], rep["syn_parent_state_id"] = v0["state_id"], s0["state_id"]
    rep["source_guard"] = source_guard().before
    dump(out / "preflight.json", rep)
    print(json.dumps({"stop": rep["stop"], "parent_reproduces_w105_fixed": rep["parent_reproduces_w105_fixed"]}, indent=1))
    return 1 if rep["stop"] else 0


# ---------------------------------------------------------------------------------------------- calibrate ---
def bindings_for(cands, rig_id, rest_digest, probe, state, rig):
    frames = probe["states"]["v0"]["frames"]
    ok, failed = {}, {}
    for j, c in cands.items():
        rec = S.verify_binding(j, c, rig_id, rest_digest, probe["rest"], frames, state["channels"], rig)
        (ok.__setitem__(rec["binding_id"], rec) if rec["status"] == "BOUND" else failed.__setitem__(j, rec))
    return ok, failed


def fk_error(rig, state, probe):
    worst, worst_at = 0.0, None
    for i, fr in enumerate(probe["states"]["v0"]["frames"]):
        fk = rig.fk(S.frame_vals(state, i))
        for b in fr["world"]:
            for p in ("head", "tail"):
                e = float(np.linalg.norm(rig.world(fk, b, p) - np.array(fr["world"][b][p])))
                if e > worst:
                    worst, worst_at = e, (fr["frame_index"], b, p)
    return worst, worst_at


def coord_probes(probe, rig_scale):
    cp = probe["coordinate_probes"]
    base, moved = cp["positions"]["base"], cp["positions"]["object_translated"]
    t = np.array(cp["translation"])
    shift_err = max(float(np.linalg.norm(np.array(moved[b][p]) - np.array(base[b][p]) - t)) for b in base for p in ("head", "tail"))
    out = {"object_translation_m": t.tolist(), "every_point_shifted_by_translation_max_err_m": shift_err,
           "base_again_identical": cp["positions"]["base_again"] == base, "armature_scale": cp["armature_scale"]}
    for k, pos in cp["positions"].items():
        if k.startswith("local_rot:"):
            moved_pts = sorted(f"{b}.{p}" for b in base for p in ("head", "tail")
                               if float(np.linalg.norm(np.array(pos[b][p]) - np.array(base[b][p]))) > S.NUMERIC_FLOOR_M)
            out[k] = {"moved_points": moved_pts, "n_moved": len(moved_pts)}
    lens = {b: float(np.linalg.norm(np.array(base[b]["tail"]) - np.array(base[b]["head"]))) for b in base}
    out["world_length_vs_rest_times_scale_max_err_m"] = max(abs(lens[b] - probe["rest"][b]["length"] * rig_scale) for b in base)
    return out


def cmd_calibrate(a):
    pre = Path(a.preflight)
    out = new_dir(a.out_dir)
    pf = load(pre / "preflight.json")
    if pf["stop"]:
        raise SystemExit("STOP: preflight failed")
    v0, s0 = load(pre / "real_parent_state.json"), load(pre / "syn_parent_state.json")
    rp, sp = load(pre / "real_probe.json"), load(pre / "syn_probe.json")
    rig = rig_from(INPUTS["w105_rig_rest"][0], rp["rig"])
    srig = syn_rig_from(sp)
    rep = {"schema": "w116_calibration_v1", "scope": "unedited parents only; no edit, constraint resolution or outcome exists"}
    rep["fk_model_max_error_m"] = {"real": fk_error(rig, v0, rp), "synthetic": fk_error(srig, s0, sp)}
    rest_digest = rp["rig"]["rest_snapshot_digest"]
    ok, failed = bindings_for(REAL_BINDING_CANDIDATES, "BaseRig", rest_digest, rp, v0, rig)
    sok, sfailed = bindings_for(SYN_BINDING_CANDIDATES, SYN_SPEC["rig_id"], sp["rig"]["spec_sha256"], sp, s0, srig)
    rep["bindings"] = {"real": {"bound": ok, "failed": failed}, "synthetic": {"bound": sok, "failed": sfailed}}
    rep["coordinate_probes"] = {"real": coord_probes(rp, rp["rig"]["scale"][0]), "synthetic": coord_probes(sp, 1.0)}
    # right arm channel facts
    rep["right_arm_channel_facts"] = {
        "rig_bones_exist": all(b in rp["rest"] for b in ("upperarm_r", "lowerarm_r", "hand_r")),
        "keyed_in_parent_state": [c for c in v0["channels"] if c.endswith("_r") and ("arm" in c)],
        "state_channels": v0["channels"],
        "right_wrist_state_controllable_channels": next(b["state_controllable_channels"] for b in ok.values()
                                                        if b["semantic_joint"] == "right_wrist"),
        "left_wrist_state_controllable_channels": next(b["state_controllable_channels"] for b in ok.values()
                                                       if b["semantic_joint"] == "left_wrist")}
    # reach slack of the real left arm on the parent (instrument fact on the unedited parent)
    L1, L2 = rp["rest"]["upperarm_l"]["length"] * rp["rig"]["scale"][0], rp["rest"]["lowerarm_l"]["length"] * rp["rig"]["scale"][0]
    slack = []
    for fr in rp["states"]["v0"]["frames"]:
        S_, W_ = np.array(fr["world"]["upperarm_l"]["head"]), np.array(fr["world"]["lowerarm_l"]["tail"])
        slack.append(L1 + L2 - float(np.linalg.norm(W_ - S_)))
    rep["real_left_arm"] = {"upper_m": L1, "fore_m": L2, "reach_slack_m": {"min": min(slack), "median": float(np.median(slack)),
                                                                           "max": max(slack)}}
    # synthetic geometry facts (design of synthetic ground truth)
    f0 = sp["states"]["v0"]["frames"]
    syn = {"shoulder_f0": f0[0]["world"]["upper"]["head"], "elbow_f0": f0[0]["world"]["upper"]["tail"],
           "wrist_f0": f0[0]["world"]["lower"]["tail"], "wrist_f5": f0[5]["world"]["lower"]["tail"],
           "upper_len": sp["rest"]["upper"]["length"], "lower_len": sp["rest"]["lower"]["length"],
           "shoulder_path": [fr["world"]["upper"]["head"] for fr in f0], "wrist_path": [fr["world"]["lower"]["tail"] for fr in f0]}
    rep["synthetic_geometry"] = syn
    dump(out / "calibration.json", rep)
    print(json.dumps({"fk_err": rep["fk_model_max_error_m"], "bound": sorted(b["semantic_joint"] for b in ok.values()),
                      "failed": {k: v["status"] for k, v in failed.items()}, "syn_bound": sorted(b["semantic_joint"] for b in sok.values()),
                      "syn_failed": {k: v["status"] for k, v in sfailed.items()}, "probes": rep["coordinate_probes"],
                      "right_arm": rep["right_arm_channel_facts"], "real_left_arm": rep["real_left_arm"]}, indent=1, default=str))
    return 0


# ---------------------------------------------------------------------------------------------- declaration ---
EDITS = {
    "SIDE_BEND_CHORD_POS": {"kind": "local_axis_rotation", "channels": ["spine_01"], "axis_local": [1.0, 0.0, 0.0], "sign": 1,
                            "magnitude": {"chord_m": 0.01, "at": ["lowerarm_l", "tail"]}},
    "SIDE_BEND_CHORD_NEG": {"kind": "local_axis_rotation", "channels": ["spine_01"], "axis_local": [1.0, 0.0, 0.0], "sign": -1,
                            "magnitude": {"chord_m": 0.01, "at": ["lowerarm_l", "tail"]}},
    "SIDE_BEND_5DEG_POS": {"kind": "local_axis_rotation", "channels": ["spine_01"], "axis_local": [1.0, 0.0, 0.0], "sign": 1,
                           "magnitude": {"angle_deg": 5.0}},
    "SIDE_BEND_5DEG_NEG": {"kind": "local_axis_rotation", "channels": ["spine_01"], "axis_local": [1.0, 0.0, 0.0], "sign": -1,
                           "magnitude": {"angle_deg": 5.0}},
    "ELBOW_SWING_CHORD": {"kind": "world_plane_rotation", "channels": ["upperarm_l"], "child": "lowerarm_l", "sign": 1,
                          "magnitude": {"chord_m": 0.01, "at": ["upperarm_l", "tail"]}},
    "SYN_BASE_5DEG": {"kind": "local_axis_rotation", "channels": ["base"], "axis_local": [1.0, 0.0, 0.0], "sign": 1,
                      "magnitude": {"angle_deg": 5.0}},
    "SYN_UPPER_5DEG": {"kind": "world_plane_rotation", "channels": ["upper"], "child": "lower", "sign": 1,
                       "magnitude": {"angle_deg": 5.0}},
}
# per-frame real families: (edit, [(joint, target_kind, offset)], authorized, method)
REAL_FAMILIES = {
    "RA_pos": ("SIDE_BEND_CHORD_POS", [("left_wrist", "capture", None)], ARM, "positive"),
    "RA_neg": ("SIDE_BEND_CHORD_NEG", [("left_wrist", "capture", None)], ARM, "positive"),
    "RB_pos": ("SIDE_BEND_5DEG_POS", [("left_wrist", "capture", None)], ARM, "positive"),
    "RB_neg": ("SIDE_BEND_5DEG_NEG", [("left_wrist", "capture", None)], ARM, "positive"),
    "RC_raise": (None, [("left_wrist", "offset", [0.0, 0.0, 0.01])], ARM, "positive"),
    "NR1_right_wrist_left_dof": ("SIDE_BEND_CHORD_POS", [("right_wrist", "capture", None)], ARM, "negative"),
    "NR2_right_wrist_right_dof": ("SIDE_BEND_CHORD_POS", [("right_wrist", "capture", None)], ["upperarm_r", "lowerarm_r"], "negative"),
    "NE1_elbow_swing_forearm_only": ("ELBOW_SWING_CHORD", [("left_wrist", "capture", None)], ["lowerarm_l"], "negative"),
    "NC1_conflicting_wrist_pins": (None, [("left_wrist", "capture", None), ("left_wrist", "offset", [0.0, 0.0, 0.02])], ARM, "negative"),
    "NC2_elbow_and_wrist_pins": ("SIDE_BEND_CHORD_POS", [("left_elbow", "capture", None), ("left_wrist", "capture", None)], ARM,
                                 "negative"),
}
REAL_INTERVALS = {"RH1_hold_109_118": (109, list(range(109, 119))), "RH2_hold_154_181": (154, list(range(154, 182)))}
V5_SAMPLE = {"family": "RA_pos", "frames": [0, 113, 160]}
RENDER_FRAMES = {"RA_pos": [0, 113, 160], "RB_pos": [0, 113, 160], "RB_neg": [113], "RC_raise": [113],
                 "NR1_right_wrist_left_dof": [113], "RH1_hold_109_118": [109, 113, 118], "SYN_K8": [6, 7, 10, 13, 14]}


def cmd_declare(a):
    out = new_dir(a.out_dir)
    pf = load(Path(a.preflight) / "preflight.json")
    cal = load(Path(a.calibration) / "calibration.json")
    if pf["stop"]:
        raise SystemExit("STOP: preflight failed")
    decl = {"schema": "w116_declaration_v1", "batch": BATCH, "outcomes_inspected": False,
            "preflight_sha256": S.sha_file(Path(a.preflight) / "preflight.json"),
            "calibration_sha256": S.sha_file(Path(a.calibration) / "calibration.json"),
            "preflight_dir": str(Path(a.preflight).resolve()), "calibration_dir": str(Path(a.calibration).resolve()),
            "inputs": {k: {"path": str(v[0]), "sha256": pf["inputs"][k]["sha256"]} for k, v in INPUTS.items()},
            "constants": {"TOL_M": S.TOL_M, "BOUNDARY_BAND_M": S.BOUNDARY_BAND_M, "NUMERIC_FLOOR_M": S.NUMERIC_FLOOR_M,
                          "LENGTH_TOL_M": S.LENGTH_TOL_M, "ELBOW_FLEX_LIMIT_DEG": S.ELBOW_FLEX_LIMIT_DEG, "CCD_TOL_M": S.CCD_TOL_M,
                          "COINCIDENT_REST": S.COINCIDENT_REST, "COINCIDENT_EVAL_M": S.COINCIDENT_EVAL_M,
                          "FK_MODEL_GATE_M": 1e-5, "V6_CHORD_BAND_M": [0.0099, 0.0101]},
            "world_definition": "Blender scene world: Armature Object matrix_world (location, rest_q*objq, scale 0.01 on the cm FBX "
                                "rig) applied to posed bone head/tail; metres. NOT reconstructed physical space.",
            "edits": EDITS, "real_families": {k: {"edit": v[0], "pins": v[1], "authorized": v[2], "role": v[3]}
                                              for k, v in REAL_FAMILIES.items()},
            "real_intervals": REAL_INTERVALS, "real_whole_clip_atomic": {"RW_pos": "RA_pos over all 199 frames as ONE request"},
            "windows": WINDOWS, "v5_sample": V5_SAMPLE, "render_frames": RENDER_FRAMES,
            "resolver": {"primary": "analytic (two_bone with control-elbow pole and declared flex limit; aim_distal; aim_rigid; "
                                    "two_pin_chain); every other DOF/pin structure -> RESOLVER_UNSUPPORTED",
                         "ccd": "bounded CCD, swing only, single pin; declared max_iter per request; tol CCD_TOL_M",
                         "user_edit_channels_never_authorized": True, "root_never_authorized": True},
            "synthetic": {"spec": SYN_SPEC, "base_amplitude_deg": SYN_BASE_AMPL_DEG, "lower_bend_deg": SYN_LOWER_BEND_DEG,
                          "cases": syn_cases_declared(cal)},
            "lineage_programmes": {
                "real": ["P1 commit RW_pos (atomic, 199 pins) on v0 (separate ref real/whole_clip)",
                         "P2 commit RC_raise at k = smallest frame whose per-frame RC decision is ACCEPT (rule; not population evidence)",
                         "P3 commit RA_pos at k' = smallest ACCEPTed RA_pos frame, prepared against v0, after P2 -> expect STALE_PARENT "
                         "if P2 moved the head",
                         "P4 revert the head -> values byte-equal v0, Blender evaluation bit-equal v0",
                         "P5 commit RB_neg at j = smallest REJECTed RB_neg frame, prepared against the P4 head -> REJECT, head unchanged"],
                "synthetic": ["S1 commit K1 on v0 -> ACCEPT expected", "S2 commit K2 prepared against v0 -> STALE_PARENT",
                              "S3 commit tampered C8 candidate prepared against the S1 head -> REJECT UNAUTHORIZED_CHANNEL_CHANGE",
                              "S4 revert -> values equal v0; Blender evaluation equal v0"]},
            "temporal": {"discontinuity_rule": "transition t-1->t of a point is DISCONTINUOUS if its candidate displacement exceeds "
                                               "3x the maximum parent displacement of the same point over all transitions",
                         "points": ["lower.tail", "upper.tail"], "role": "reported, not a gate"},
            "predictions_not_gates": {
                "RA/RB": "the real left arm is nearly straight (reach slack from calibration), so the side-bend sign that moves the "
                         "shoulder away from the hand is predicted largely INFEASIBLE; the other sign largely FEASIBLE",
                "NR1": "INSUFFICIENT_DOF on every frame where the edit moves the right wrist > 1 mm",
                "NR2": "CONTRACT_VIOLATION DOF_NOT_IN_STATE on every frame", "NC1": "CONFLICTING_HARD_CONSTRAINTS on every frame",
                "NE1/NC2": "mostly INFEASIBLE", "SYN_K8": "interval accepted; a discontinuity at the release transition 13->14 and "
                                                         "possibly at 6->7 (no blending is applied)"},
            "gates": GATES, "case_rule": "D if any V fails; else C if any GC fails; else B if any GB fails; else A"}
    dump(out / "declaration.json", decl)
    print(S.sha_file(out / "declaration.json"))
    return 0


GATES = {
    "V1_authentication": "preflight has no STOP (HEAD, blob, lock, historical Worklogs, every input hash)",
    "V2_parent_reproduction": "Blender Action of v0 has the W105 FIXED value digest and 32 F-curves",
    "V3_fk_model": "Python FK vs Blender on unedited parents <= 1e-5 m (real and synthetic)",
    "V4_bindings": "left_wrist, left_elbow, left_shoulder, right_wrist and syn_wrist/syn_elbow/syn_shoulder BOUND",
    "V5_per_frame_independence": "each V5 single-frame state evaluates bit-equal to the combined family Action at its frame and "
                                 "bit-equal to v0 at every other frame",
    "V6_meaningful_edits": "RA_pos/RA_neg unpinned wrist displacement within V6_CHORD_BAND_M on every frame; RC target displacement "
                           "within the band; RB control residual > TOL_M on every frame",
    "V7_gt_soundness": "no evaluated candidate frame with GT INFEASIBLE has residual <= TOL_M; synthetic GT classes equal declared",
    "V8_repeatability": "two builds byte-identical states files; two Blender evaluations of the same file identical",
    "V9_solver_vs_evaluated_distinguishable": "claimed-SOLVED but unpinned candidates (real RA_pos frame 0, synthetic K1) are "
                                              "REJECTED EVALUATED_PIN_UNSATISFIED by the evaluator",
    "GC1_express": "every positive request validates with no error; constraint JSON round-trips with identical digests",
    "GC2_bind": "declared fail-safe bindings fail as declared (thorax/left_hand AMBIGUOUS, nose MISSING; syn_hand AMBIGUOUS, "
                "syn_finger/syn_nose MISSING); requests using them are UNSUPPORTED_BINDING",
    "GC3_measure": "every request frame has a Blender-evaluated residual",
    "GC4_safe_rejection": "every negative request/frame expected to be rejected is REJECTed with its expected failure class; no "
                          "GT-INFEASIBLE frame is ACCEPTed",
    "GC5_commit_soundness": "no non-trivial ACCEPT with an evaluated residual > TOL_M or a scope/locality/length/root/limit failure; "
                            "refs unchanged after every rejection; source hashes unchanged after every stage",
    "GB1_synthetic_feasible": "K1, K2, K2b, K6b and K8 ACCEPT",
    "GB2_real_feasible": "every GT-FEASIBLE frame of RA/RB/RC per-frame requests ACCEPTs; every interval whose frames are all "
                         "GT-FEASIBLE ACCEPTs",
    "GB3_reversal": "real P4 and synthetic S4 reverts restore values byte-exactly and evaluate bit-equal to v0",
    "GB4_lineage": "STALE_PARENT on P3 (if P2 executed) and S2; P5 leaves the head unchanged; every lineage id re-verifies",
    "GB5_atomicity": "every multi-frame request is ACCEPT iff all its frames pass",
}


def syn_cases_declared(cal):
    g = cal["synthetic_geometry"]
    Sh, W0 = np.array(g["shoulder_f0"]), np.array(g["wrist_f0"])
    u = (W0 - Sh) / np.linalg.norm(W0 - Sh)
    rot20 = S.q_to_m(S.q_axis_angle([0, 0, 1], math.radians(20)))
    S7, W7 = np.array(g["shoulder_path"][7]), np.array(g["wrist_path"][7])
    k8b_offset = (0.07 * (W7 - S7) / np.linalg.norm(W7 - S7)).tolist()
    return {
        "K1": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "expect": ("ACCEPT", "NONE")},
        "K2": {"edit": "SYN_BASE_5DEG", "pins": [["syn_wrist", "capture", None]], "auth": ["upper", "lower"], "expect": ("ACCEPT", "NONE")},
        "K2b": {"pins": [["syn_wrist", "offset", (Sh + rot20 @ (W0 - Sh) - W0).tolist()]], "auth": ["upper"],
                "expect": ("ACCEPT", "NONE")},
        "K3": {"pins": [["syn_wrist", "offset", (Sh + 0.60 * u - W0).tolist()]], "auth": ["upper", "lower"],
               "expect": ("REJECT", "INFEASIBLE_REQUEST")},
        "K3b": {"pins": [["syn_wrist", "offset", (Sh + 0.10 * u - W0).tolist()]], "auth": ["upper", "lower"],
                "expect": ("REJECT", "INFEASIBLE_REQUEST")},
        "K4": {"pins": [["syn_wrist", "capture", None], ["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"],
               "expect": ("REJECT", "INFEASIBLE_REQUEST")},
        "K4b": {"pins": [["syn_elbow", "offset", [0.0, 0.0, 0.03]], ["syn_wrist", "capture", None]], "auth": ["upper", "lower"],
                "expect": ("REJECT", "INFEASIBLE_REQUEST")},
        "K5a": {"pins": [["syn_wrist", "offset", (Sh + 0.70 * u - W0).tolist()]], "auth": ["upper", "lower"],
                "expect": ("REJECT", "INFEASIBLE_REQUEST")},
        "K5b": {"pins": [["syn_wrist", "offset", (Sh + 0.70 * u - W0).tolist()]], "auth": ["base", "upper", "lower"],
                "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "K5c": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": [], "expect": ("REJECT", "INFEASIBLE_REQUEST")},
        "K5d": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["__object__"], "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "K6a": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "method": "ccd", "max_iter": 1,
                "expect": ("REJECT", "SOLVER_FAILURE")},
        "K6b": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "method": "ccd", "max_iter": 64,
                "expect": ("ACCEPT", "NONE")},
        "K7": {"pins": [["syn_wrist", "capture", None]], "auth": ["upper", "lower"], "expect": ("ACCEPT", "NONE"), "trivial": True},
        "C2": {"pins": [["syn_nose", "capture", None]], "auth": ["upper", "lower"], "expect": ("REJECT", "UNSUPPORTED_BINDING")},
        "C3": {"pins": [["syn_hand", "capture", None]], "auth": ["upper", "lower"], "expect": ("REJECT", "UNSUPPORTED_BINDING")},
        "C3b": {"pins": [["syn_finger", "capture", None]], "auth": ["upper", "lower"], "expect": ("REJECT", "UNSUPPORTED_BINDING")},
        "C4": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "mutate": "frame_ARMATURE_LOCAL",
               "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "C5a": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "mutate": "unit_cm",
                "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "C5b": {"pins": [["syn_wrist", "capture", None]], "auth": ["upper", "lower"], "mutate": "values_x100_labelled_m",
                "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "C6": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "mutate": "target_altered_no_redigest",
               "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "C6b": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "mutate": "target_altered_redigested",
                "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "C9": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "mutate": "type_ORIENTATION_PIN",
               "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "C10": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "mutate": "soft",
                "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "C12": {"edit": "SYN_UPPER_5DEG", "pins": [["syn_wrist", "capture", None]], "auth": ["upper", "lower"],
                "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "C13": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "mutate": "constraint_parent_other",
                "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "C8": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "tamper": "base_plus_1deg",
               "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "C15": {"pins": [["syn_wrist", "offset", [0.0, 0.0, 0.05]]], "auth": ["upper", "lower"], "tamper": "claim_solved_unchanged",
                "expect": ("REJECT", "SOLVER_FAILURE")},
        "C11": {"pins": [["syn_wrist", "capture", None]], "auth": ["upper", "lower"], "tamper": "armature_moved_after_capture",
                "expect": ("REJECT", "CONTRACT_VIOLATION")},
        "K8": {"interval": [7, 13], "capture_frame": 7, "pins": [["syn_wrist", "capture", None]], "auth": ["upper", "lower"],
               "expect": ("ACCEPT", "NONE")},
        "K8b": {"interval": [0, 19], "capture_frame": 7, "pins": [["syn_wrist", "offset", k8b_offset]], "auth": ["upper", "lower"],
                "expect": "GT", "design_note": "target = wrist(7) + 0.07 m along shoulder(7)->wrist(7); GT from the parent shoulder "
                                               "path is infeasible on frames 9-11 only"},
    }


# ---------------------------------------------------------------------------------------------- build ---
class Ctx:
    def __init__(self, decl):
        pre, cal = Path(decl["preflight_dir"]), load(Path(decl["calibration_dir"]) / "calibration.json")
        self.v0, self.s0 = load(pre / "real_parent_state.json"), load(pre / "syn_parent_state.json")
        self.rp, self.sp = load(pre / "real_probe.json"), load(pre / "syn_probe.json")
        self.rig = rig_from(decl["inputs"]["w105_rig_rest"]["path"], self.rp["rig"])
        self.srig = syn_rig_from(self.sp)
        self.real_b, self.real_fb = cal["bindings"]["real"]["bound"], cal["bindings"]["real"]["failed"]
        self.syn_b, self.syn_fb = cal["bindings"]["synthetic"]["bound"], cal["bindings"]["synthetic"]["failed"]
        self.capture = {}
        for (st, probe, b) in ((self.v0, self.rp, self.real_b), (self.s0, self.sp, self.syn_b)):
            for fr in probe["states"]["v0"]["frames"]:
                for bid, rec in b.items():
                    self.capture[(st["state_id"], fr["frame_index"], bid)] = fr["world"][rec["bone"]][rec["point"]]

    def side(self, syn):
        if syn:
            return self.s0, self.srig, self.syn_b, self.syn_fb, self.sp
        return self.v0, self.rig, self.real_b, self.real_fb, self.rp

    def bid(self, joint, syn):
        b = self.syn_b if syn else self.real_b
        return next((k for k, v in b.items() if v["semantic_joint"] == joint), None)

    def vctx(self, parent, syn):
        st, rig, b, fb, _ = self.side(syn)
        roots = {n for n in rig.bones if rig.is_root(n)}
        return {"bindings": b, "failed_bindings": fb, "capture": self.capture, "parent": parent, "root_bones": roots}


def make_pins(ctx, syn, parent, pin_decl, frames, auth, cap_frame=None, cid_prefix="c"):
    out = []
    for i, (joint, kind, off) in enumerate(pin_decl):
        bid = ctx.bid(joint, syn) or f"UNBOUND:{joint}"
        cf = frames[0] if cap_frame is None else cap_frame
        src = {"kind": "CAPTURED_FROM_EVALUATED_STATE" if kind == "capture" else "CAPTURED_PLUS_DECLARED_OFFSET",
               "state_id": parent["state_id"], "frame_index": cf, "binding_id": bid,
               "evaluated_by": "w116_blender_stages probe (preflight)"}
        if kind == "offset":
            src["offset_m"] = off
        t = S.captured_target(ctx.capture, src)
        t = [0.0, 0.0, 0.0] if t is None else t.tolist()
        out.append(S.make_pin(f"{cid_prefix}{i}", joint, bid, t, src, frames, auth, parent["state_id"]))
    return out


def edit_record(name, frames, params):
    if name is None:
        return None
    e = dict(EDITS[name])
    return dict(e, edit_id=name, frames=frames, params={str(k): v for k, v in params.items()})


def solve_request(ctx, syn, parent, edit_name, pins, frames, method="analytic", max_iter=64, solve=True):
    """Build control (edit only) and candidate (edit + pin) values for the given frames; returns per-frame records."""
    st, rig, b, _, _ = ctx.side(syn)
    recs, params = {}, {}
    ctrl_vals, cand_vals = {}, {}
    for k in frames:
        i = parent["frames"].index(k)
        vals = S.frame_vals(parent, i)
        e = EDITS.get(edit_name) if edit_name else None
        if e is not None:
            if "chord_m" in e["magnitude"]:
                params[k] = S.edit_param_for_chord(rig, vals, e, *e["magnitude"]["at"], e["magnitude"]["chord_m"])
            else:
                params[k] = {"angle_rad": e["sign"] * math.radians(e["magnitude"]["angle_deg"])}
        cv = S.apply_edit(rig, vals, e, params.get(k))
        rp = []
        for c in pins:
            if c["binding_id"] in b and k in c["frames"]:
                rp.append({"bone": b[c["binding_id"]]["bone"], "point": b[c["binding_id"]]["point"], "target": np.array(c["target"]),
                           "semantic_joint": c["semantic_joint"]})
        auth = pins[0]["authorized_channels"] if pins else []
        if not solve:
            status, pv, info = "NOT_RESOLVED_CONTRACT_INVALID", cv, {}
        elif rp:
            status, pv, info = S.resolve_frame(rig, cv, rp, auth, method=method, max_iter=max_iter)
        else:
            status, pv, info = "NOT_APPLICABLE", cv, {}
        ctrl_vals[k], cand_vals[k] = cv, pv
        recs[k] = {"resolver_status": status, "resolver_info": info}
    return recs, params, ctrl_vals, cand_vals


def patched_state(parent, patches, op):
    vals = {c: [list(q) for q in parent["values"][c]] for c in parent["channels"]}
    for k, v in patches.items():
        i = parent["frames"].index(k)
        for c in parent["channels"]:
            vals[c][i] = [float(x) for x in v[c]]
    return S.make_state(parent["state_id"], parent["rig_id"], parent["frames"], parent["channels"], vals, op, parent.get("source"))


def mutate_pins(pins, how, parent_other):
    if not how:
        return pins
    c = copy.deepcopy(pins[0])
    redigest = True
    if how == "frame_ARMATURE_LOCAL":
        c["coordinate_frame"] = "ARMATURE_LOCAL"
    elif how == "unit_cm":
        c["unit"] = "cm"
    elif how == "values_x100_labelled_m":
        c["target"] = [x * 100.0 for x in c["target"]]
        c["target_digest"] = S.digest({"target": c["target"], "source": c["target_source"], "binding_id": c["binding_id"],
                                       "frames": c["frames"]})
    elif how == "target_altered_no_redigest":
        c["target"][2] += 0.01
        redigest = False
    elif how == "target_altered_redigested":
        c["target"][2] += 0.01
        c["target_digest"] = S.digest({"target": c["target"], "source": c["target_source"], "binding_id": c["binding_id"],
                                       "frames": c["frames"]})
    elif how == "type_ORIENTATION_PIN":
        c["type"] = "ORIENTATION_PIN"
    elif how == "soft":
        c["hard"] = False
    elif how == "constraint_parent_other":
        c["parent_state_id"] = parent_other
    if redigest:
        c["constraint_digest"] = S.digest({k: v for k, v in c.items() if k != "constraint_digest"})
    return [c] + pins[1:]


def cmd_build(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    if S.sha_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash")
    ctx = Ctx(decl)
    guard = source_guard()
    t0 = time.time()
    v0, s0 = ctx.v0, ctx.s0
    real_states, syn_states = {"v0": v0}, {"v0": s0}
    requests = {}
    frames_all = v0["frames"]
    # ---- real per-frame families
    for fam, (edit, pin_decl, auth, role) in REAL_FAMILIES.items():
        ctrl, cand, per = {}, {}, {}
        for k in frames_all:
            pins = make_pins(ctx, False, v0, pin_decl, [k], auth, cid_prefix=f"{fam}/{k}/")
            req = {"request_id": f"{fam}/{k}", "parent_state_id": v0["state_id"], "edit": edit_record(edit, [k], {}),
                   "constraints": pins, "resolver": {"method": "analytic"}}
            errs = S.validate_request(req, ctx.vctx(v0, False))
            recs, params, cv, pv = solve_request(ctx, False, v0, edit, pins, [k], solve=not errs)
            if req["edit"]:
                req["edit"]["params"] = {str(k): params[k]}
            ctrl[k], cand[k] = cv[k], pv[k]
            per[k] = {"request": req, "validation_errors": errs, **recs[k]}
        real_states[f"{fam}__control"] = patched_state(v0, ctrl, {"kind": "EVAL_CONTROL", "family": fam})
        real_states[f"{fam}__candidate"] = patched_state(v0, cand, {"kind": "EVAL_CANDIDATE", "family": fam})
        requests[fam] = {"kind": "per_frame", "role": role, "edit": edit, "authorized": auth, "frames": per,
                         "candidate_values": {str(k): {c: list(map(float, cand[k][c])) for c in cand[k]} for k in cand}}
    # ---- real intervals and whole-clip atomic
    for name, (cap, frames) in REAL_INTERVALS.items():
        pins = make_pins(ctx, False, v0, [("left_wrist", "capture", None)], frames, ARM, cap_frame=cap, cid_prefix=f"{name}/")
        req = {"request_id": name, "parent_state_id": v0["state_id"], "edit": None, "constraints": pins, "resolver": {"method": "analytic"}}
        errs = S.validate_request(req, ctx.vctx(v0, False))
        recs, _, cv, pv = solve_request(ctx, False, v0, None, pins, frames, solve=not errs)
        real_states[f"{name}__candidate"] = patched_state(v0, pv, {"kind": "INTERVAL_CANDIDATE", "request": name})
        requests[name] = {"kind": "interval", "role": "positive_or_gt", "request": req, "validation_errors": errs, "frames": recs,
                          "authorized": ARM, "edit": None}
    pins = [p for k in frames_all for p in make_pins(ctx, False, v0, REAL_FAMILIES["RA_pos"][1], [k], ARM, cid_prefix=f"RW_pos/{k}/")]
    req = {"request_id": "RW_pos", "parent_state_id": v0["state_id"],
           "edit": edit_record("SIDE_BEND_CHORD_POS", frames_all, {k: requests["RA_pos"]["frames"][k]["request"]["edit"]["params"][str(k)]
                                                                  for k in frames_all}),
           "constraints": pins, "resolver": {"method": "analytic"}}
    requests["RW_pos"] = {"kind": "whole_clip", "role": "atomic", "request": req,
                          "validation_errors": S.validate_request(req, ctx.vctx(v0, False)),
                          "uses_states": ["RA_pos__control", "RA_pos__candidate"]}
    # ---- V5 single-frame independence states and V9 claimed-solved real control
    cvals = requests["RA_pos"]["candidate_values"]
    for k in V5_SAMPLE["frames"]:
        real_states[f"V5_RA_pos_f{k}"] = patched_state(v0, {k: cvals[str(k)]}, {"kind": "V5_SINGLE_FRAME", "frame": k})
    # ---- synthetic
    syn_req = {}
    for name, cd in decl["synthetic"]["cases"].items():
        if "interval" in cd:
            lo, hi = cd["interval"]
            frames = list(range(lo, hi + 1))
            cap = cd["capture_frame"]
        else:
            frames, cap = [0], 0
        pins = make_pins(ctx, True, s0, [tuple(p) for p in cd["pins"]], frames, cd["auth"], cap_frame=cap, cid_prefix=f"{name}/")
        pins = mutate_pins(pins, cd.get("mutate"), "0" * 64)
        edit = cd.get("edit")
        req = {"request_id": name, "parent_state_id": s0["state_id"], "edit": edit_record(edit, frames, {}), "constraints": pins,
               "resolver": {"method": cd.get("method", "analytic"), "max_iter": cd.get("max_iter", 64)}}
        errs = S.validate_request(req, ctx.vctx(s0, True))
        recs, params, cv, pv = solve_request(ctx, True, s0, edit, pins, frames, cd.get("method", "analytic"),
                                             cd.get("max_iter", 64), solve=not errs)
        if req["edit"]:
            req["edit"]["params"] = {str(k): v for k, v in params.items()}
        tamper = cd.get("tamper")
        if tamper == "base_plus_1deg":
            for k in pv:
                pv[k] = dict(pv[k])
                pv[k]["base"] = S.q_mul(np.asarray(pv[k]["base"]), S.q_axis_angle([1, 0, 0], math.radians(1.0)))
        elif tamper in ("claim_solved_unchanged",):
            pv = {k: dict(cv[k]) for k in cv}
            for k in recs:
                recs[k]["resolver_status"] = "SOLVED"
                recs[k]["resolver_info"] = {"tamper": "claims SOLVED but returns the unpinned control"}
        syn_states[f"{name}__control"] = patched_state(s0, cv, {"kind": "EVAL_CONTROL", "case": name})
        cand = patched_state(s0, pv, {"kind": "EVAL_CANDIDATE", "case": name})
        if tamper == "armature_moved_after_capture":
            cand = dict(cand, armature_location=[0.1, 0.0, 0.0])
        syn_states[f"{name}__candidate"] = cand
        syn_req[name] = {"request": req, "validation_errors": errs, "frames": recs, "declared": cd}
    # real V9 claimed-solved control (RA_pos frame 0)
    real_states["V9_RA_pos_f0_claimed"] = patched_state(
        v0, {0: {c: real_states["RA_pos__control"]["values"][c][0] for c in v0["channels"]}},
        {"kind": "V9_CLAIMED_SOLVED_UNCHANGED (resolver output replaced by the unpinned control)", "frame": 0})
    states_file(out / "real_states.json", real_states, "w105")
    states_file(out / "syn_states.json", syn_states, "synthetic")
    rep = {"schema": "w116_build_v1", "tag": a.tag, "declaration_sha256": a.declaration_sha, "wall_s": time.time() - t0,
           "real_states": {n: s["state_id"] for n, s in real_states.items()},
           "syn_states": {n: s["state_id"] for n, s in syn_states.items()},
           "requests": requests, "synthetic_requests": syn_req, "source_guard": guard.check()}
    dump(out / "build_report.json", rep)
    print(json.dumps({"tag": a.tag, "wall_s": rep["wall_s"], "n_real_states": len(real_states), "n_syn_states": len(syn_states),
                      "source_unchanged": rep["source_guard"]["unchanged"]}))
    return 0


# ---------------------------------------------------------------------------------------------- evaluate ---
def cmd_evaluate(a):
    out = new_dir(a.out_dir)
    b = Path(a.build_dir)
    blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", INPUTS["w105_rig_skinned_fbx"][0], "--rest",
            INPUTS["w105_rig_rest"][0], "--states", b / "real_states.json", "--out", out / "real_eval.json", log=out / "real_eval.log")
    blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", f"synthetic:{Path(a.preflight) / 'syn_spec.json'}",
            "--states", b / "syn_states.json", "--out", out / "syn_eval.json", log=out / "syn_eval.log")
    print(json.dumps({"real": S.sha_file(out / "real_eval.json"), "syn": S.sha_file(out / "syn_eval.json")}))
    return 0


# ---------------------------------------------------------------------------------------------- commit ---
def ev_frames(ev, name):
    return {fr["frame_index"]: fr for fr in ev["states"][name]["frames"]}


def frame_pins(ctx, syn, req, k):
    _, _, b, _, _ = ctx.side(syn)
    return [{"bone": b[c["binding_id"]]["bone"], "point": b[c["binding_id"]]["point"], "target": np.array(c["target"]),
             "semantic_joint": c["semantic_joint"]} for c in req["constraints"] if c["binding_id"] in b and k in c["frames"]]


def eval_request_frames(ctx, syn, req, frames, recs, ev, cand_name, ctrl_name, errs):
    """Independent evaluation of every frame of one request from Blender output + Blender rest only."""
    st, rig, _, _, _ = ctx.side(syn)
    rest, scale = ev["rest"], ev["rig"]["scale"][0]
    E_c, E_k, E_p = ev_frames(ev, cand_name), ev_frames(ev, ctrl_name), ev_frames(ev, "v0")
    auth = req["constraints"][0]["authorized_channels"] if req["constraints"] else []
    edit_ch = req["edit"]["channels"] if req.get("edit") else []
    allowed = {"vs_parent": sorted(set(auth) | set(edit_ch)), "vs_control": sorted(auth)}
    dom = S.domain_points(rig, sorted(set(auth) | set(edit_ch)))
    chain = ("upper", "lower") if syn else ("upperarm_l", "lowerarm_l")
    out = {}
    for k in frames:
        pins = frame_pins(ctx, syn, req, k)
        if not pins:
            out[k] = {"gt": "NOT_APPLICABLE", "evaluation": None}
            continue
        mode, _ = S._structure(rig, pins, auth)
        if errs:
            gt, margin = "NOT_ASSESSED_CONTRACT_INVALID", None
        elif mode is None:
            gt, margin = "UNDETERMINED_UNSUPPORTED_STRUCTURE", None
        else:
            gt, margin = S.gt_feasibility(mode, E_k[k], pins, rest, scale)
        evd = S.evaluate_candidate(E_c[k], E_k[k], E_p[k], pins, allowed, dom, rest, scale, chain=chain)
        out[k] = {"gt": gt, "gt_margin_m": margin, "structure": mode, "evaluation": evd}
    return out


def request_decision(errs, recs, evals):
    statuses = [recs[k]["resolver_status"] for k in recs if recs[k]["resolver_status"] != "NOT_APPLICABLE"]
    return S.decide(errs, statuses, [e["evaluation"] for e in evals.values() if e["evaluation"] is not None])


def per_frame_family(ctx, fam, R, ev):
    res = {}
    for k, fr in R["frames"].items():
        k = int(k)
        req, errs = fr["request"], [tuple(e) for e in fr["validation_errors"]]
        recs = {k: {"resolver_status": fr["resolver_status"]}}
        evals = eval_request_frames(ctx, False, req, [k], recs, ev, f"{fam}__candidate", f"{fam}__control", errs)
        dec = request_decision(errs, recs, evals)
        res[k] = {"decision": dec, "resolver_status": fr["resolver_status"], "resolver_info": fr["resolver_info"],
                  "validation_errors": errs, **evals[k]}
    return res


def summarize_family(res, frames=None):
    ks = sorted(res) if frames is None else [k for k in sorted(res) if k in frames]
    s = {"n": len(ks), "gt": {}, "resolver": {}, "decision": {}, "failure_class": {}, "reasons": {}}
    for k in ks:
        r = res[k]
        for key, val in (("gt", r.get("gt")), ("resolver", r["resolver_status"]), ("decision", r["decision"]["decision"]),
                         ("failure_class", r["decision"]["failure_class"])):
            s[key][str(val)] = s[key].get(str(val), 0) + 1
        for reason in r["decision"]["reasons"]:
            s["reasons"][reason] = s["reasons"].get(reason, 0) + 1
    evs = [res[k]["evaluation"] for k in ks if res[k].get("evaluation")]
    if evs:
        cr = [p["control_residual_m"] for e in evs for p in e["pins"]]
        pr = [p["candidate_residual_m"] for e in evs for p in e["pins"]]
        s["control_residual_m"] = {"min": min(cr), "median": float(np.median(cr)), "max": max(cr)}
        s["candidate_residual_m"] = {"min": min(pr), "median": float(np.median(pr)), "max": max(pr)}
        acc = [res[k] for k in ks if res[k]["decision"]["decision"] == "ACCEPT" and res[k].get("evaluation")]
        if acc:
            ar = [p["candidate_residual_m"] for r in acc for p in r["evaluation"]["pins"]]
            s["accepted_candidate_residual_m"] = {"min": min(ar), "median": float(np.median(ar)), "max": max(ar)}
            s["accepted_stored_changed_vs_control"] = sorted({c for r in acc for c in r["evaluation"]["stored_changed_vs_control"]})
            s["accepted_stored_changed_vs_parent"] = sorted({c for r in acc for c in r["evaluation"]["stored_changed_vs_parent"]})
            s["accepted_points_moved_vs_control"] = sorted({p for r in acc for p in r["evaluation"]["moved_vs_control_m"]})
            s["accepted_max_bone_length_error_m"] = max(r["evaluation"]["max_bone_length_error_m"] for r in acc)
            s["accepted_bend_flips"] = sum(1 for r in acc if r["evaluation"].get("bend_flip"))
            fl = [r["evaluation"]["elbow_flex_deg"] - r["evaluation"]["control_elbow_flex_deg"] for r in acc
                  if "elbow_flex_deg" in r["evaluation"]]
            if fl:
                s["accepted_elbow_flex_change_deg"] = {"min": min(fl), "median": float(np.median(fl)), "max": max(fl)}
            s["accepted_root_unchanged"] = all(r["evaluation"]["root_unchanged"] for r in acc)
            s["accepted_outside_domain_changed"] = sum(len(r["evaluation"]["outside_domain_changed_exact"]) for r in acc)
        side = {}
        for e in evs:
            for pt, d in e["moved_vs_parent_m"].items():
                side[pt] = max(side.get(pt, 0.0), d)
        s["max_moved_vs_parent_m_by_point"] = dict(sorted(side.items()))
    return s


def cmd_commit(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    if S.sha_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash")
    ctx = Ctx(decl)
    guard = source_guard()
    B = load(Path(a.build_dir) / "build_report.json")
    rev, sev = load(Path(a.eval_dir) / "real_eval.json"), load(Path(a.eval_dir) / "syn_eval.json")
    rep = {"schema": "w116_commit_v1", "declaration_sha256": a.declaration_sha, "families": {}, "intervals": {}, "synthetic": {}}
    fam_res = {}
    for fam in REAL_FAMILIES:
        fam_res[fam] = per_frame_family(ctx, fam, B["requests"][fam], rev)
        rep["families"][fam] = {"summary": summarize_family(fam_res[fam]),
                                "windows": {w: summarize_family(fam_res[fam], fr) for w, fr in WINDOWS.items()},
                                "outside_windows": summarize_family(fam_res[fam], [k for k in fam_res[fam]
                                                                                   if not any(k in f for f in WINDOWS.values())])}
    rep["per_frame"] = {fam: {str(k): v for k, v in r.items()} for fam, r in fam_res.items()}
    for name in REAL_INTERVALS:
        R = B["requests"][name]
        req, errs = R["request"], [tuple(e) for e in R["validation_errors"]]
        recs = {int(k): v for k, v in R["frames"].items()}
        evals = eval_request_frames(ctx, False, req, sorted(recs), recs, rev, f"{name}__candidate", "v0", errs)
        dec = request_decision(errs, recs, evals)
        per = {str(k): {"resolver_status": recs[k]["resolver_status"], **evals[k],
                        "frame_pass": evals[k]["evaluation"]["pins_satisfied"] if evals[k]["evaluation"] else None} for k in sorted(recs)}
        rep["intervals"][name] = {"decision": dec, "frames": per,
                                  "summary": summarize_family({k: {"decision": {"decision": "ACCEPT" if evals[k]["evaluation"]["pins_satisfied"]
                                                                                 and recs[k]["resolver_status"] == "SOLVED" else "REJECT",
                                                                                 "failure_class": "-", "reasons": []},
                                                                    "resolver_status": recs[k]["resolver_status"], **evals[k]}
                                                               for k in recs})}
    # whole clip atomic RW_pos: reuse the RA_pos per-frame evaluations under one request
    R = B["requests"]["RW_pos"]
    errs = [tuple(e) for e in R["validation_errors"]]
    ra = fam_res["RA_pos"]
    rw_dec = S.decide(errs, [ra[k]["resolver_status"] for k in ra], [ra[k]["evaluation"] for k in ra])
    rep["whole_clip"] = {"RW_pos": {"decision": rw_dec, "frames_passing": sum(1 for k in ra if ra[k]["decision"]["decision"] == "ACCEPT"),
                                    "n_frames": len(ra)}}
    # V5 per-frame independence
    v5 = {}
    comb = ev_frames(rev, "RA_pos__candidate")
    base = ev_frames(rev, "v0")
    for k in V5_SAMPLE["frames"]:
        single = ev_frames(rev, f"V5_RA_pos_f{k}")
        v5[k] = {"frame_equal_combined": single[k]["world"] == comb[k]["world"] and single[k]["stored_wxyz"] == comb[k]["stored_wxyz"],
                 "other_frames_equal_v0": all(single[j]["world"] == base[j]["world"] and single[j]["stored_wxyz"] == base[j]["stored_wxyz"]
                                              for j in single if j != k)}
    rep["v5_independence"] = v5
    # V9 real claimed-solved control
    req0 = B["requests"]["RA_pos"]["frames"]["0"]["request"]
    ev9 = eval_request_frames(ctx, False, req0, [0], {0: {"resolver_status": "SOLVED"}}, rev, "V9_RA_pos_f0_claimed", "RA_pos__control", [])
    rep["v9_real_claimed_solved"] = {"decision": S.decide([], ["SOLVED"], [ev9[0]["evaluation"]]),
                                     "residual_m": ev9[0]["evaluation"]["pins"][0]["candidate_residual_m"]}
    # synthetic cases
    syn_res = {}
    for name, R in B["synthetic_requests"].items():
        req, errs = R["request"], [tuple(e) for e in R["validation_errors"]]
        recs = {int(k): v for k, v in R["frames"].items()}
        evals = eval_request_frames(ctx, True, req, sorted(recs), recs, sev, f"{name}__candidate", f"{name}__control", errs)
        dec = request_decision(errs, recs, evals)
        cd = R["declared"]
        gts = {k: evals[k]["gt"] for k in evals}
        if cd["expect"] == "GT":
            exp = ("ACCEPT", "NONE") if all(g == "FEASIBLE" for g in gts.values()) else ("REJECT", "INFEASIBLE_REQUEST")
        else:
            exp = tuple(cd["expect"])
        syn_res[name] = {"decision": dec, "expected": exp, "matches_expected": (dec["decision"], dec["failure_class"]) == exp,
                         "gt": gts, "validation_errors": errs,
                         "resolver": {str(k): {"status": recs[k]["resolver_status"], "info": recs[k]["resolver_info"]} for k in recs},
                         "evaluation": {str(k): evals[k] for k in evals}}
    rep["synthetic"] = syn_res
    # temporal continuity (synthetic interval K8/K8b vs parent)
    rep["temporal"] = {n: temporal(sev, f"{n}__candidate") for n in ("K8", "K8b")}
    # ---------------------------------------------------------------- lineage programmes
    store = S.Store(out / "store")
    lin = {"real": [], "synthetic": []}
    v0, s0 = ctx.v0, ctx.s0
    store.put(v0)
    store.put(s0)
    store.init_ref("real/main", v0["state_id"])
    store.init_ref("real/whole_clip", v0["state_id"])
    store.init_ref("syn/main", s0["state_id"])

    def real_cand(fam, k, parent):
        vals = B["requests"][fam]["candidate_values"][str(k)]
        return patched_state(parent, {k: vals}, {"kind": "PIN_EDIT", "request": f"{fam}/{k}"})

    # P1 whole clip
    cand_rw = patched_state(v0, {k: B["requests"]["RA_pos"]["candidate_values"][str(k)] for k in v0["frames"]},
                            {"kind": "PIN_EDIT", "request": "RW_pos"})
    lin["real"].append(dict(step="P1", **store.commit("real/whole_clip", v0["state_id"], cand_rw, rw_dec)))
    acc = lambda fam: [k for k in sorted(fam_res[fam]) if fam_res[fam][k]["decision"]["decision"] == "ACCEPT"  # noqa: E731
                       and "TRIVIALLY_SATISFIED_NOT_EVIDENCE" not in fam_res[fam][k]["decision"]["reasons"]]
    rc_acc, ra_acc = acc("RC_raise"), acc("RA_pos")
    rbn_rej = [k for k in sorted(fam_res["RB_neg"]) if fam_res["RB_neg"][k]["decision"]["decision"] == "REJECT"]
    head0 = store.refs["real/main"]
    if rc_acc:
        k = rc_acc[0]
        par = store.get(head0)
        cand = real_cand("RC_raise", k, par)
        same_values_as_v0_parent = canon_eq(par["values"], v0["values"])
        dec = fam_res["RC_raise"][k]["decision"] if same_values_as_v0_parent else {"decision": "REJECT", "reasons": ["NOT_EVALUATED"],
                                                                                  "failure_class": "INVALID_EXPERIMENT"}
        lin["real"].append(dict(step="P2", frame=k, **store.commit("real/main", head0, cand, dec)))
    else:
        lin["real"].append({"step": "P2", "executed": False, "reason": "no ACCEPTed RC_raise frame"})
    if ra_acc and rc_acc:
        k = ra_acc[0]
        cand = real_cand("RA_pos", k, store.get(head0))
        lin["real"].append(dict(step="P3", frame=k, **store.commit("real/main", head0, cand, fam_res["RA_pos"][k]["decision"])))
    else:
        lin["real"].append({"step": "P3", "executed": False})
    rev_state, rrec = store.revert("real/main") if store.refs["real/main"] != v0["state_id"] else (None, {"executed": False})
    lin["real"].append(dict(step="P4", **rrec))
    if rbn_rej:
        j = rbn_rej[0]
        hp = store.refs["real/main"]
        cand = real_cand("RB_neg", j, store.get(hp))
        rec = store.commit("real/main", hp, cand, fam_res["RB_neg"][j]["decision"])
        lin["real"].append(dict(step="P5", frame=j, head_unchanged=rec["head_after"] == hp, **rec))
    # synthetic programme
    k1 = syn_res["K1"]["decision"]
    sv = load(Path(a.build_dir) / "syn_states.json")["states"]

    def syn_c(name, parent):
        return patched_state(parent, {0: {c: sv[f"{name}__candidate"]["values"][c][0] for c in s0["channels"]}},
                             {"kind": "PIN_EDIT", "request": name})

    c1 = syn_c("K1", s0)
    lin["synthetic"].append(dict(step="S1", **store.commit("syn/main", s0["state_id"], c1, k1)))
    c2 = syn_c("K2", s0)
    lin["synthetic"].append(dict(step="S2", **store.commit("syn/main", s0["state_id"], c2, syn_res["K2"]["decision"])))
    hp = store.refs["syn/main"]
    c8 = syn_c("C8", store.get(hp))
    rec8 = store.commit("syn/main", hp, c8, syn_res["C8"]["decision"])
    lin["synthetic"].append(dict(step="S3", head_unchanged=rec8["head_after"] == hp, **rec8))
    srev, srec = store.revert("syn/main") if store.refs["syn/main"] != s0["state_id"] else (None, {"executed": False})
    lin["synthetic"].append(dict(step="S4", **srec))
    # Blender re-evaluation of the reverted states
    rev_eval = {}
    lst = {}
    if rev_state is not None:
        lst["real_revert"] = rev_state
    if lst:
        states_file(out / "real_revert_states.json", lst, "w105")
        blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", INPUTS["w105_rig_skinned_fbx"][0], "--rest",
                INPUTS["w105_rig_rest"][0], "--states", out / "real_revert_states.json", "--out", out / "real_revert_eval.json",
                log=out / "real_revert_eval.log")
        e = load(out / "real_revert_eval.json")["states"]["real_revert"]
        rev_eval["real"] = {"value_digest_equal_v0": e["value_digest"] == rev["states"]["v0"]["value_digest"],
                            "frames_bit_equal_v0": all(x["world"] == y["world"] and x["stored_wxyz"] == y["stored_wxyz"]
                                                       for x, y in zip(e["frames"], rev["states"]["v0"]["frames"]))}
    if srev is not None:
        states_file(out / "syn_revert_states.json", {"syn_revert": srev}, "synthetic")
        blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", f"synthetic:{Path(decl['preflight_dir']) / 'syn_spec.json'}",
                "--states", out / "syn_revert_states.json", "--out", out / "syn_revert_eval.json", log=out / "syn_revert_eval.log")
        e = load(out / "syn_revert_eval.json")["states"]["syn_revert"]
        rev_eval["synthetic"] = {"value_digest_equal_v0": e["value_digest"] == sev["states"]["v0"]["value_digest"],
                                 "frames_bit_equal_v0": all(x["world"] == y["world"] and x["stored_wxyz"] == y["stored_wxyz"]
                                                            for x, y in zip(e["frames"], sev["states"]["v0"]["frames"]))}
    rep["lineage"] = {"programmes": lin, "store_log": store.log, "refs": store.refs,
                      "lineage_audit": {r: store.lineage(h) for r, h in store.refs.items()}, "revert_blender": rev_eval}
    # mixed W102 ownership negative control (contract level)
    rep["mixed_ownership_control"] = mixed_ownership_control()
    rep["source_guard"] = guard.check()
    rep["parent_files_unchanged"] = {"real": S.verify_state(load(Path(decl["preflight_dir"]) / "real_parent_state.json")),
                                     "syn": S.verify_state(load(Path(decl["preflight_dir"]) / "syn_parent_state.json"))}
    dump(out / "commit_report.json", rep)
    print(json.dumps({"families": {f: {k: v for k, v in r["summary"].items() if k in ("gt", "decision", "failure_class", "reasons")}
                                   for f, r in rep["families"].items()},
                      "intervals": {n: r["decision"] for n, r in rep["intervals"].items()}, "whole_clip": rep["whole_clip"],
                      "synthetic": {n: [r["decision"]["decision"], r["decision"]["failure_class"], r["decision"]["reasons"],
                                        r["matches_expected"]] for n, r in syn_res.items()},
                      "v5": v5, "v9": rep["v9_real_claimed_solved"], "revert": rev_eval,
                      "lineage": [(x.get("step"), x.get("decision"), x.get("reasons")) for x in lin["real"] + lin["synthetic"]],
                      "source": rep["source_guard"]["unchanged"]}, indent=1, default=str))
    return 0


def canon_eq(a, b):
    return S.canon(a) == S.canon(b)


def temporal(ev, cand):
    P, C = ev_frames(ev, "v0"), ev_frames(ev, cand)
    ks = sorted(P)
    out = {}
    for pt in ("lower.tail", "upper.tail"):
        b, p = pt.split(".")
        dp = [float(np.linalg.norm(np.array(P[k]["world"][b][p]) - np.array(P[k - 1]["world"][b][p]))) for k in ks[1:]]
        dc = [float(np.linalg.norm(np.array(C[k]["world"][b][p]) - np.array(C[k - 1]["world"][b][p]))) for k in ks[1:]]
        lim = 3 * max(dp)
        out[pt] = {"parent_step_m": dp, "candidate_step_m": dc, "limit_m": lim,
                   "discontinuous_transitions": [f"{k - 1}->{k}" for k, d in zip(ks[1:], dc) if d > lim]}
    return out


def mixed_ownership_control():
    import w114_rig_influence as R
    own = R.OWNERS
    res = {}
    for cfg in ("real_s0_only", "mixed_w102"):
        ch = sorted(c for c, o in own[cfg].items() if c in ("upperarm_l", "lowerarm_l") and o.startswith("S0 left arm"))
        res[cfg] = {"channels_owned_by_S0_left_arm": ch,
                    "request_authorize_by_source_S0_left_arm": "RESOLVABLE" if ch else "REJECT UNSUPPORTED_BINDING SOURCE_OWNS_NO_CHANNEL",
                    "arm_owner": own[cfg]["upperarm_l"]}
    res["historical_reference"] = "W115 MA1 (S0 left_wrist perturbation in mixed_w102 windows): no arm channel changed"
    return res


# ---------------------------------------------------------------------------------------------- render ---
def cmd_render(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    rev, sev = load(Path(a.eval_dir) / "real_eval.json"), load(Path(a.eval_dir) / "syn_eval.json")
    cr = load(Path(a.commit_dir) / "commit_report.json")
    jobs_r, jobs_s = [], []
    col = {"target": (0.1, 0.9, 0.2, 1), "unpinned": (0.95, 0.15, 0.1, 1), "pinned": (0.1, 0.4, 1.0, 1)}

    def jobs_for(name, frames, cand, ctrl, joint_pt, eval_, syn, per):
        lst = []
        for k in frames:
            pinrec = per(k)
            if not pinrec:
                continue
            p = pinrec["pins"][0]
            mk = [{"name": "target", "position": p["target"], "color": col["target"], "radius": 0.014},
                  {"name": "unpinned", "position": p["unpinned_control"], "color": col["unpinned"], "radius": 0.010},
                  {"name": "pinned", "position": p["pinned_candidate"], "color": col["pinned"], "radius": 0.008}]
            for st, tag in (("v0", "a_parent"), (ctrl, "b_unpinned"), (cand, "c_pinned")):
                for view, off in (("front", [0.0, -3.0, 0.0]), ("side", [3.0, 0.0, 0.0])):
                    lst.append({"id": f"{name}_f{k:03d}__{view}__{tag}", "state": st, "frame_index": k, "markers": mk,
                                "centre": p["target"], "view_offset": off})
        return lst

    for fam in ("RA_pos", "RB_pos", "RB_neg", "RC_raise", "NR1_right_wrist_left_dof"):
        per = cr["per_frame"][fam]
        jobs_r += jobs_for(fam, RENDER_FRAMES[fam], f"{fam}__candidate", f"{fam}__control", None, rev, False,
                           lambda k, per=per: per[str(k)]["evaluation"])
    iv = cr["intervals"]["RH1_hold_109_118"]["frames"]
    jobs_r += jobs_for("RH1_hold_109_118", RENDER_FRAMES["RH1_hold_109_118"], "RH1_hold_109_118__candidate", "v0", None, rev, False,
                       lambda k: iv[str(k)]["evaluation"])
    k8 = cr["synthetic"]["K8"]["evaluation"]
    jobs_s += jobs_for("SYN_K8", RENDER_FRAMES["SYN_K8"], "K8__candidate", "v0", None, sev, True,
                       lambda k: (k8.get(str(k)) or {}).get("evaluation") or syn_marker(sev, cr, k))
    dump(out / "real_jobs.json", {"jobs": jobs_r})
    dump(out / "syn_jobs.json", {"jobs": jobs_s})
    b = Path(a.build_dir)
    blender("scripts/w116_blender_stages.py", "--", "render", "--rig", INPUTS["w105_rig_skinned_fbx"][0], "--rest",
            INPUTS["w105_rig_rest"][0], "--states", b / "real_states.json", "--jobs", out / "real_jobs.json", "--out-dir",
            out / "real", "--ortho-scale", "1.4", log=out / "real_render.log")
    blender("scripts/w116_blender_stages.py", "--", "render", "--rig", f"synthetic:{Path(decl['preflight_dir']) / 'syn_spec.json'}",
            "--states", b / "syn_states.json", "--jobs", out / "syn_jobs.json", "--out-dir", out / "syn", "--ortho-scale", "1.6",
            log=out / "syn_render.log")
    compose(out)
    return 0


def syn_marker(sev, cr, k):
    """Outside the K8 interval there is no pin; markers show the frame-7 target and the wrist positions."""
    t = cr["synthetic"]["K8"]["evaluation"]["7"]["evaluation"]["pins"][0]["target"]
    P, C = ev_frames(sev, "v0")[k], ev_frames(sev, "K8__candidate")[k]
    return {"pins": [{"target": t, "unpinned_control": P["world"]["lower"]["tail"], "pinned_candidate": C["world"]["lower"]["tail"]}]}


def compose(out):
    from PIL import Image, ImageDraw
    for sub in ("real", "syn"):
        d = out / sub
        groups = {}
        for p in sorted(d.glob("*.png")):
            key = p.stem.split("__")[0]
            groups.setdefault(key, []).append(p)
        for key, ps in groups.items():
            ims = [Image.open(p) for p in sorted(ps)]
            w, h = ims[0].size
            grid = Image.new("RGB", (w * 3, h * 2 + 24), (20, 20, 20))
            order = {"front": 0, "side": 1}
            for p, im in zip(sorted(ps), ims):
                view, tag = p.stem.split("__")[1:]
                col = "abc".index(tag[0])
                grid.paste(im, (col * w, 24 + order[view] * h))
            dr = ImageDraw.Draw(grid)
            dr.text((6, 4), f"{key}  | a parent | b unpinned edit | c pinned edit |  green=target red=unpinned wrist blue=pinned wrist",
                    fill=(255, 255, 255))
            grid.save(out / f"review_{sub}_{key}.png")


# ---------------------------------------------------------------------------------------------- decide ---
def cmd_decide(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    if S.sha_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash")
    pf = load(Path(decl["preflight_dir"]) / "preflight.json")
    cal = load(Path(decl["calibration_dir"]) / "calibration.json")
    cr = load(Path(a.commit_dir) / "commit_report.json")
    b1, b2 = Path(a.build_dir), Path(a.build_dir2)
    e1, e2 = Path(a.eval_dir), Path(a.eval_dir2)
    g, detail = {}, {}
    g["V1_authentication"] = not pf["stop"]
    g["V2_parent_reproduction"] = pf["parent_reproduces_w105_fixed"] and pf["probe"]["fcurves"] == 32
    fk = cal["fk_model_max_error_m"]
    g["V3_fk_model"] = fk["real"][0] <= 1e-5 and fk["synthetic"][0] <= 1e-5
    bound = {b["semantic_joint"] for b in cal["bindings"]["real"]["bound"].values()}
    sbound = {b["semantic_joint"] for b in cal["bindings"]["synthetic"]["bound"].values()}
    g["V4_bindings"] = {"left_wrist", "left_elbow", "left_shoulder", "right_wrist"} <= bound and \
        {"syn_wrist", "syn_elbow", "syn_shoulder"} <= sbound
    g["V5_per_frame_independence"] = all(v["frame_equal_combined"] and v["other_frames_equal_v0"] for v in cr["v5_independence"].values())
    lo, hi = decl["constants"]["V6_CHORD_BAND_M"]
    v6 = {}
    for fam in ("RA_pos", "RA_neg"):
        d = [float(np.linalg.norm(np.array(r["evaluation"]["pins"][0]["unpinned_control"]) - np.array(r["evaluation"]["pins"][0]["before_parent"])))
             for r in cr["per_frame"][fam].values()]
        v6[fam] = {"min": min(d), "max": max(d), "ok": lo <= min(d) and max(d) <= hi}
    d = [float(np.linalg.norm(np.array(r["evaluation"]["pins"][0]["target"]) - np.array(r["evaluation"]["pins"][0]["before_parent"])))
         for r in cr["per_frame"]["RC_raise"].values()]
    v6["RC_raise_target_displacement"] = {"min": min(d), "max": max(d), "ok": lo <= min(d) and max(d) <= hi}
    for fam in ("RB_pos", "RB_neg"):
        d = [r["evaluation"]["pins"][0]["control_residual_m"] for r in cr["per_frame"][fam].values()]
        v6[fam] = {"min": min(d), "ok": min(d) > S.TOL_M}
    detail["V6"] = v6
    g["V6_meaningful_edits"] = all(v["ok"] for v in v6.values())
    bad_gt = []
    for fam, per in cr["per_frame"].items():
        for k, r in per.items():
            if r.get("gt") == "INFEASIBLE" and r.get("evaluation") and all(p["candidate_residual_m"] <= S.TOL_M for p in r["evaluation"]["pins"]):
                bad_gt.append(f"{fam}/{k}")
    for n, r in cr["intervals"].items():
        for k, f in r["frames"].items():
            if f["gt"] == "INFEASIBLE" and f["evaluation"]["pins_satisfied"]:
                bad_gt.append(f"{n}/{k}")
    syn_gt_decl = {"K1": "FEASIBLE", "K2": "FEASIBLE", "K2b": "FEASIBLE", "K3": "INFEASIBLE", "K3b": "INFEASIBLE",
                   "K5a": "INFEASIBLE", "K6b": "FEASIBLE", "K6a": "FEASIBLE", "K8": "FEASIBLE"}
    syn_gt_bad = {n: cr["synthetic"][n]["gt"] for n, want in syn_gt_decl.items()
                  if any(v != want for v in cr["synthetic"][n]["gt"].values())}
    detail["V7"] = {"gt_infeasible_but_satisfied": bad_gt, "synthetic_gt_mismatch": syn_gt_bad}
    g["V7_gt_soundness"] = not bad_gt and not syn_gt_bad
    rb = (b1 / "real_states.json").read_bytes() == (b2 / "real_states.json").read_bytes() and \
        (b1 / "syn_states.json").read_bytes() == (b2 / "syn_states.json").read_bytes()
    ev_same = all(load(e1 / f)["states"] == load(e2 / f)["states"] for f in ("real_eval.json", "syn_eval.json"))
    detail["V8"] = {"states_files_identical": rb, "evaluations_identical": ev_same}
    g["V8_repeatability"] = rb and ev_same
    v9r = cr["v9_real_claimed_solved"]["decision"]
    v9s = cr["synthetic"]["C15"]["decision"]
    g["V9_solver_vs_evaluated_distinguishable"] = all(d["decision"] == "REJECT" and "EVALUATED_PIN_UNSATISFIED" in d["reasons"]
                                                      for d in (v9r, v9s))
    # ---- GC
    pos_err = [f"{fam}/{k}" for fam, (e, p, au, role) in REAL_FAMILIES.items() if role == "positive"
               for k, r in cr["per_frame"][fam].items() if r["validation_errors"]]
    pos_err += [n for n in REAL_INTERVALS if cr["intervals"][n]["decision"]["failure_class"] == "CONTRACT_VIOLATION"]
    syn_pos = ("K1", "K2", "K2b", "K6a", "K6b", "K7", "K8", "K8b", "K3", "K3b", "K4b", "K5a")
    pos_err += [n for n in syn_pos if cr["synthetic"][n]["validation_errors"]]
    rt = roundtrip_check(b1)
    detail["GC1"] = {"positive_with_validation_errors": pos_err, "roundtrip": rt}
    g["GC1_express"] = not pos_err and rt["ok"]
    fb = cal["bindings"]["real"]["failed"]
    sfb = cal["bindings"]["synthetic"]["failed"]
    want = {"thorax": "BINDING_AMBIGUOUS", "left_hand": "BINDING_AMBIGUOUS", "nose": "BINDING_MISSING"}
    swant = {"syn_hand": "BINDING_AMBIGUOUS", "syn_finger": "BINDING_MISSING", "syn_nose": "BINDING_MISSING"}
    gc2 = all(fb.get(j, {}).get("status") == s for j, s in want.items()) and all(sfb.get(j, {}).get("status") == s for j, s in swant.items())
    gc2 = gc2 and all(cr["synthetic"][n]["decision"]["failure_class"] == "UNSUPPORTED_BINDING" for n in ("C2", "C3", "C3b"))
    g["GC2_bind"] = gc2
    missing = [f"{fam}/{k}" for fam, per in cr["per_frame"].items() for k, r in per.items()
               if not r["validation_errors"] and r.get("evaluation") is None]
    missing += [f"{n}/{k}" for n, r in cr["intervals"].items() for k, f in r["frames"].items() if f["evaluation"] is None]
    g["GC3_measure"] = not missing
    detail["GC3"] = missing
    unsafe, wrong = [], []
    for fam, (e, p, au, role) in REAL_FAMILIES.items():
        for k, r in cr["per_frame"][fam].items():
            d = r["decision"]
            if r.get("gt") == "INFEASIBLE" and d["decision"] == "ACCEPT":
                unsafe.append(f"{fam}/{k}")
            exp = expected_negative(fam, r)
            if exp and (d["decision"], d["failure_class"]) != exp:
                wrong.append({"req": f"{fam}/{k}", "got": [d["decision"], d["failure_class"], d["reasons"]], "expected": exp})
    for n, r in cr["synthetic"].items():
        if r["expected"][0] == "REJECT" and not r["matches_expected"]:
            wrong.append({"req": n, "got": [r["decision"]["decision"], r["decision"]["failure_class"], r["decision"]["reasons"]],
                          "expected": r["expected"]})
    mo = cr["mixed_ownership_control"]
    if mo["mixed_w102"]["channels_owned_by_S0_left_arm"] or not mo["real_s0_only"]["channels_owned_by_S0_left_arm"]:
        wrong.append({"req": "mixed_ownership_control"})
    detail["GC4"] = {"unsafe_accepts": unsafe, "wrong_class": wrong[:50], "n_wrong": len(wrong)}
    g["GC4_safe_rejection"] = not unsafe and not wrong
    bad_acc = []
    for fam, per in cr["per_frame"].items():
        for k, r in per.items():
            if r["decision"]["decision"] == "ACCEPT" and r.get("evaluation"):
                ev = r["evaluation"]
                if not (ev["pins_satisfied"] and ev["scope_ok"] and ev["locality_ok"] and ev["lengths_ok"] and ev["root_unchanged"]
                        and ev["joint_limit_ok"]):
                    bad_acc.append(f"{fam}/{k}")
    for n, r in cr["synthetic"].items():
        if r["decision"]["decision"] == "ACCEPT":
            for k, e in r["evaluation"].items():
                ev = e["evaluation"]
                if ev and not (ev["pins_satisfied"] and ev["scope_ok"] and ev["locality_ok"] and ev["lengths_ok"] and ev["root_unchanged"]):
                    bad_acc.append(f"{n}/{k}")
    ref_ok = all(x.get("head_after") == x.get("head_before") for x in cr["lineage"]["store_log"]
                 if x.get("op") == "commit" and x.get("decision") == "REJECT")
    src_ok = cr["source_guard"]["unchanged"] and all(cr["parent_files_unchanged"].values())
    for bd in (b1, b2):
        src_ok = src_ok and load(bd / "build_report.json")["source_guard"]["unchanged"]
    detail["GC5"] = {"bad_accepts": bad_acc, "refs_unchanged_on_reject": ref_ok, "source_unchanged": src_ok}
    g["GC5_commit_soundness"] = not bad_acc and ref_ok and src_ok
    # ---- GB
    gb1 = {n: cr["synthetic"][n]["decision"] for n in ("K1", "K2", "K2b", "K6b", "K8")}
    g["GB1_synthetic_feasible"] = all(d["decision"] == "ACCEPT" and not d["reasons"] for d in gb1.values())
    lim = []
    for fam in ("RA_pos", "RA_neg", "RB_pos", "RB_neg", "RC_raise"):
        for k, r in cr["per_frame"][fam].items():
            if r["gt"] == "FEASIBLE" and r["decision"]["decision"] != "ACCEPT":
                lim.append({"req": f"{fam}/{k}", "got": r["decision"], "resolver": r["resolver_status"]})
    for n, r in cr["intervals"].items():
        if all(f["gt"] == "FEASIBLE" for f in r["frames"].values()) and r["decision"]["decision"] != "ACCEPT":
            lim.append({"req": n, "got": r["decision"]})
    detail["GB2"] = {"feasible_not_accepted": lim[:50], "n": len(lim)}
    g["GB2_real_feasible"] = not lim
    rv = cr["lineage"]["revert_blender"]
    p4 = next(x for x in cr["lineage"]["programmes"]["real"] if x["step"] == "P4")
    s4 = next(x for x in cr["lineage"]["programmes"]["synthetic"] if x["step"] == "S4")
    g["GB3_reversal"] = bool(rv.get("real") and rv.get("synthetic")) and all(v["value_digest_equal_v0"] and v["frames_bit_equal_v0"]
                                                                            for v in rv.values()) \
        and p4.get("values_equal_restored") is True and s4.get("values_equal_restored") is True
    progs = cr["lineage"]["programmes"]
    p3 = next(x for x in progs["real"] if x["step"] == "P3")
    p2 = next(x for x in progs["real"] if x["step"] == "P2")
    s2 = next(x for x in progs["synthetic"] if x["step"] == "S2")
    p5 = next((x for x in progs["real"] if x["step"] == "P5"), None)
    stale_ok = (s2.get("reasons") == ["STALE_PARENT"]) and (p3.get("reasons") == ["STALE_PARENT"] if p2.get("decision") == "ACCEPT" else True)
    audit_ok = all(all(x["verifies"] for x in chain) for chain in cr["lineage"]["lineage_audit"].values())
    g["GB4_lineage"] = stale_ok and audit_ok and (p5 is not None and p5["head_unchanged"] and p5["decision"] == "REJECT")
    detail["GB4"] = {"P2": p2.get("decision"), "P3": p3.get("reasons"), "S2": s2.get("reasons"), "P5": p5 and p5.get("decision"),
                     "audit_ok": audit_ok}
    atom = []
    for n, r in cr["intervals"].items():
        allpass = all(f["frame_pass"] and f["resolver_status"] == "SOLVED" for f in r["frames"].values())
        if (r["decision"]["decision"] == "ACCEPT") != allpass:
            atom.append(n)
    rw = cr["whole_clip"]["RW_pos"]
    if (rw["decision"]["decision"] == "ACCEPT") != (rw["frames_passing"] == rw["n_frames"]):
        atom.append("RW_pos")
    for n in ("K8", "K8b"):
        r = cr["synthetic"][n]
        allpass = all(e["evaluation"]["pins_satisfied"] and r["resolver"][k]["status"] == "SOLVED" for k, e in r["evaluation"].items())
        if (r["decision"]["decision"] == "ACCEPT") != allpass:
            atom.append(n)
    g["GB5_atomicity"] = not atom
    detail["GB5"] = atom
    V = [k for k in g if k.startswith("V") and not g[k]]
    GC = [k for k in g if k.startswith("GC") and not g[k]]
    GB = [k for k in g if k.startswith("GB") and not g[k]]
    case = "D" if V else ("C" if GC else ("B" if GB else "A"))
    res = {"schema": "w116_decision_v1", "declaration_sha256": a.declaration_sha, "gates": g, "failed": {"V": V, "GC": GC, "GB": GB},
           "case": case, "detail": detail}
    dump(out / "decision.json", res)
    print(json.dumps({"case": case, "failed": res["failed"], "gates": g}, indent=1))
    return 0


def expected_negative(fam, r):
    gt = r.get("gt")
    if fam == "NR2_right_wrist_right_dof":
        return ("REJECT", "CONTRACT_VIOLATION")
    if fam == "NC1_conflicting_wrist_pins":
        return ("REJECT", "INFEASIBLE_REQUEST")
    if fam in ("NR1_right_wrist_left_dof", "NE1_elbow_swing_forearm_only", "NC2_elbow_and_wrist_pins") or fam.startswith(("RA", "RB", "RC")):
        if gt == "INFEASIBLE":
            return ("REJECT", "INFEASIBLE_REQUEST")
    return None


def roundtrip_check(build_dir):
    B = load(Path(build_dir) / "build_report.json")
    n = bad = 0
    for fam in B["requests"]:
        R = B["requests"][fam]
        reqs = [f["request"] for f in R["frames"].values()] if R["kind"] == "per_frame" else [R["request"]]
        for rq in reqs:
            for c in rq["constraints"]:
                n += 1
                c2 = json.loads(json.dumps(c))
                if c2["constraint_digest"] != S.digest({k: v for k, v in c2.items() if k != "constraint_digest"}):
                    bad += 1
    return {"constraints_checked": n, "digest_mismatch_after_roundtrip": bad, "ok": bad == 0 and n > 0}


# ---------------------------------------------------------------------------------------------- manifest ---
def cmd_manifest(a):
    base = Path(a.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != "evidence_manifest.json":
            files[p.relative_to(base).as_posix()] = {"sha256": S.sha_file(p), "bytes": p.stat().st_size}
    dump(base / "evidence_manifest.json", {"schema": "w116_evidence_manifest_v1", "declaration_sha256": a.declaration_sha,
                                           "starting_head": STARTING_HEAD, "files": files})
    print(S.sha_file(base / "evidence_manifest.json"), len(files))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--out-dir", required=True)  # noqa: E702
    p = sub.add_parser("calibrate"); p.add_argument("--preflight", required=True); p.add_argument("--out-dir", required=True)  # noqa: E702
    p = sub.add_parser("declare")
    for n in ("preflight", "calibration", "out_dir"):
        p.add_argument("--" + n.replace("_", "-"), required=True)
    for name, extra in (("build", ("tag",)), ("evaluate", ("build_dir", "preflight")), ("commit", ("build_dir", "eval_dir")),
                        ("render", ("build_dir", "eval_dir", "commit_dir")),
                        ("decide", ("build_dir", "build_dir2", "eval_dir", "eval_dir2", "commit_dir"))):
        p = sub.add_parser(name)
        if name != "evaluate":
            p.add_argument("--declaration", required=True)
            p.add_argument("--declaration-sha", required=True)
        for n in extra + ("out_dir",):
            p.add_argument("--" + n.replace("_", "-"), required=True)
    p = sub.add_parser("manifest"); p.add_argument("--base", required=True); p.add_argument("--declaration-sha", required=True)  # noqa: E702
    a = ap.parse_args()
    return {"preflight": cmd_preflight, "calibrate": cmd_calibrate, "declare": cmd_declare, "build": cmd_build,
            "evaluate": cmd_evaluate, "commit": cmd_commit, "render": cmd_render, "decide": cmd_decide, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
