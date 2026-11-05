#!/usr/bin/env python3
"""W117: Semantic World Pin — temporal coherence and compensating articulation (evidence only, NON-PRODUCTION).

Compares, on identical requests, rig and Blender evaluator:
  U  unpinned user edit (control)
  F  W116 frame-independent WORLD_POSITION_PIN resolver (frozen W116 code, imported unchanged)
  T  W117 temporally aware swivel resolver over the pin frames only (same authorised arm DOF, same frames as F)
  TE T plus declared entry/release transition margins (holds only; same authorised DOF; margins carry no hard pin)

Stages (each refuses an existing output directory):
  preflight  identity, W116 manifest/decision/declaration authentication, inputs, parent rebuild, fresh Blender
             evaluation of the parent (W105 FIXED digest + bit-equality to the W116 probe), timing/interpolation probe,
             W117 synthetic rig + parent trajectories probed in Blender
  calibrate  W116 framewise control parity (pure Python, all frames, against the W116 build), synthetic FK/binding
             calibration and synthetic test design from unpinned/parent geometry only (no candidate exists yet)
  declare    frozen declaration (solver, metrics, populations, gates, CASE rule) -> hash
  build      U/F/T/TE states for every declared request (Blender evaluation of U for the feasibility oracle only)
  evaluate   Blender evaluation of a build (shared independent evaluator: scripts/w116_blender_stages.py, unchanged)
  assess     independent evaluation, atomic decisions, temporal metrics, comparisons, negative controls, lineage,
             revert (Blender re-evaluation), W116 Blender parity, source/W116 guards
  render     rig-only (bone proxy) contact sheets and GIFs of U/F/T/TE from fixed cameras
  decide     frozen gates + recorded visual review -> CASE
  manifest   evidence manifest
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w116_semantic_pin_base as S  # noqa: E402
import w116_semantic_pin_contract as C  # noqa: E402
import w117_temporal_pin_base as TB  # noqa: E402

BATCH = 117
INSTRUCTION_BLOB = "b193782c6a9090672f16d848b05f01942a6cf78d"
STARTING_HEAD = "4ea6dd4a593805c915c302da6c06022a88aaa8da"
BRANCH = "arch/single_frame_first"
LOCK = Path("C:/Users/dna10/animcv-automation/state/active_batch.json")
W116 = Path("C:/Users/dna10/animcv-w116")
W116_MANIFEST_SHA = "117645586c1ab77450098ac5cde2f3ebca4ddc03bb2d14fbd425b9da168cc4ac"
W116_DECL_SHA = "ad4fe0df26a175316a060a000297796b02092c60e009325ebee685e8c6bd898f"
W116_DECISION_SHA = "bcf6ee3a34d181f06f0c15f5201c46d5644838eaa5dd2d30060786d889e75b94"
HIST = {113: "docs/113_WORKLOG_ANIMATION_BASE_CONTRACT_FEASIBILITY.md",
        114: "docs/114_WORKLOG_SOURCE_PRESERVING_ANIMATION_BASE_EDIT_AUTHORITY.md",
        115: "docs/115_WORKLOG_ANIMATION_BASE_VALIDITY_REPEATABILITY_RESOLUTION_AUTHORITY.md",
        116: "docs/116_WORKLOG_SEMANTIC_WORLD_POSITION_PIN_FEASIBILITY.md"}
UPPER, LOWER = "upperarm_l", "lowerarm_l"
ARM = [UPPER, LOWER]
REAL_FAMS = ["RA_pos", "RA_neg", "RB_pos", "RB_neg", "RC_raise"]
REAL_HOLDS = {"RH1": (109, list(range(109, 119))), "RH2": (154, list(range(154, 182)))}
HOLD_EDITS = {"none": None, "RBpos": "SIDE_BEND_5DEG_POS", "RBneg": "SIDE_BEND_5DEG_NEG"}
HOLD_EDIT_FAMILY = {"none": None, "RBpos": "RB_pos", "RBneg": "RB_neg"}
WINDOWS = C.WINDOWS

# ------------------------------------------------------------------------------------------- synthetic design ---
SYN_SPEC = {"rig_id": "W117_SYN_ARM",
            "object": {"location": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "rest_rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "bones": [{"name": "base", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 0.5], "roll": 0.0},
                      {"name": "neck", "parent": "base", "head": [0.0, 0.0, 0.5], "tail": [0.0, 0.0, 0.65], "roll": 0.0,
                       "connect": True},
                      {"name": "upper", "parent": "base", "head": [0.12, 0.0, 0.45], "tail": [0.42, 0.0, 0.45], "roll": 0.0,
                       "connect": False},
                      {"name": "lower", "parent": "upper", "head": [0.42, 0.0, 0.45], "tail": [0.67, 0.0, 0.45], "roll": 0.0,
                       "connect": True},
                      {"name": "hand", "parent": "lower", "head": [0.67, 0.0, 0.45], "tail": [0.75, 0.0, 0.45], "roll": 0.0,
                       "connect": True}]}
SYN_CH = ["__object__", "base", "upper", "lower"]
SYN_ARM = ["upper", "lower"]
N_SYN = 40
SYN_FRAMES = list(range(N_SYN))
# angle(tau) = c0 + c1 tau + c2 tau^2 + c3 tau^3 + amp sin(2 pi freq tau + phase)   [deg], tau = f / (N-1)
TRAJ = {
    "A": {"role": "evaluation", "tw": ([0, 0, 0, 0], [12, 0.8, 0.4]), "bx": ([0, 0, 5, 0], [0, 0, 0]),
          "az": ([-35, 30, 0, 0], [10, 1.3, 0.0]), "ax": ([0, 0, 0, 0], [20, 0.6, 1.0]), "flex": ([50, 0, 0, 0], [25, 0.9, 0.5])},
    "B": {"role": "evaluation", "tw": ([-8, 0, 0, 16], [0, 0, 0]), "bx": ([0, 0, 0, 0], [6, 1.1, 0.0]),
          "az": ([-20, 0, 0, 0], [-25, 0.7, 1.7]), "ax": ([-10, 0, 25, 0], [0, 0, 0]), "flex": ([75, -35, 0, 0], [8, 2.0, 0.0])},
    "C": {"role": "evaluation_near_singular", "tw": ([0, 0, 0, 0], [10, 0.7, 0.2]), "bx": ([0, 0, 0, 0], [4, 0.5, 0.3]),
          "az": ([-25, 35, 0, 0], [0, 0, 0]), "ax": ([0, 0, 0, 0], [15, 0.9, 0.0]), "flex": ([6, 0, 0, 0], [3, 1.4, 0.9])},
    "D": {"role": "development_split_smoke_only", "tw": ([0, 0, 0, 0], [9, 1.2, 2.0]), "bx": ([-4, 8, 0, 0], [0, 0, 0]),
          "az": ([-30, 0, 0, 0], [20, 0.5, 0.3]), "ax": ([0, 0, 0, 0], [12, 0.8, 1.5708]), "flex": ([40, 20, 0, 0], [0, 0, 0])},
}
SYN_EDITS = {
    "SYN_E8": {"kind": "local_axis_rotation", "channels": ["base"], "axis_local": [1.0, 0.0, 0.0], "sign": 1,
               "magnitude": {"angle_deg": 8.0}},
    "SYN_E3_POS": {"kind": "local_axis_rotation", "channels": ["base"], "axis_local": [1.0, 0.0, 0.0], "sign": 1,
                   "magnitude": {"angle_deg": 3.0}},
    "SYN_E3_NEG": {"kind": "local_axis_rotation", "channels": ["base"], "axis_local": [1.0, 0.0, 0.0], "sign": -1,
                   "magnitude": {"angle_deg": 3.0}},
}
SYN_HOLDS = {"H_A": ("A", "SYN_E8", 14, (14, 25)), "H_B": ("B", "SYN_E8", 11, (11, 24)), "H_C": ("C", "SYN_E8", 12, (12, 24)),
             "H_Cx": ("C", "SYN_E8", 12, (12, 24)), "H_D": ("D", "SYN_E8", 15, (15, 26))}
SYN_BINDING_CANDIDATES = {"syn_wrist": [("lower", "tail"), ("hand", "head")], "syn_elbow": [("upper", "tail"), ("lower", "head")],
                          "syn_shoulder": [("upper", "head")], "syn_hand": [("hand", "head"), ("hand", "tail")], "syn_nose": []}
NEAR_SINGULAR_DESIGN_SLACK_M = 0.002
INFEASIBLE_DESIGN_EXCESS_M = 0.004


def git(*a):
    return C.git(*a)


dump, load, new_dir, clean = C.dump, C.load, C.new_dir, C.clean


def blender(*args, log=None):
    return C.blender(*args, log=log)


def angle_at(p, tau):
    c, s = p
    return c[0] + c[1] * tau + c[2] * tau ** 2 + c[3] * tau ** 3 + s[0] * math.sin(2 * math.pi * s[1] * tau + s[2])


def traj_state(name):
    p = TRAJ[name]
    vals = {c: [] for c in SYN_CH}
    for f in SYN_FRAMES:
        tau = f / (N_SYN - 1)
        r = {k: math.radians(angle_at(p[k], tau)) for k in ("tw", "bx", "az", "ax", "flex")}
        vals["__object__"].append([0.0, 0.0, 0.0, 1.0])
        vals["base"].append(S.q_mul(S.q_axis_angle([0, 1, 0], r["tw"]), S.q_axis_angle([1, 0, 0], r["bx"])).tolist())
        vals["upper"].append(S.q_mul(S.q_axis_angle([0, 0, 1], r["az"]), S.q_axis_angle([1, 0, 0], r["ax"])).tolist())
        vals["lower"].append(S.q_axis_angle([0, 0, 1], r["flex"]).tolist())
    src = {"kind": "DECLARED_SYNTHETIC_TRAJECTORY", "trajectory": name, "params": p, "rig_identity": "W117_SYN_ARM (known)",
           "video_subject_identity": "NOT_APPLICABLE"}
    return S.make_state(None, SYN_SPEC["rig_id"], SYN_FRAMES, SYN_CH, vals, {"kind": "ROOT_SYNTHETIC", "trajectory": name}, src)


def syn_rig(probe):
    bones = {n: {"parent": b["parent"], "matrix_local": b["matrix_local"], "head": b["head_local"], "tail": b["tail_local"]}
             for n, b in probe["rest"].items()}
    return S.Rig({"rig_id": SYN_SPEC["rig_id"], "bones": bones, "object": SYN_SPEC["object"]})


def guard():
    files = {k: v[0] for k, v in C.INPUTS.items()} | {"w105_rig_rest_json": C.INPUTS["w105_rig_rest"][0]}
    files |= {"w116_declaration": W116 / "results/declare/declaration.json", "w116_decision": W116 / "results/decision/decision.json",
              "w116_manifest": W116 / "evidence_manifest.json",
              "w116_real_parent_state": W116 / "results/preflight/real_parent_state.json",
              "w116_build_r1_report": W116 / "results/build_r1/build_report.json",
              "w116_eval_e1_real": W116 / "results/eval_e1/real_eval.json"}
    return S.SourceGuard(files)


def w116_manifest_check():
    m = load(W116 / "evidence_manifest.json")["files"]
    bad = [f for f, r in m.items() if not (W116 / f).exists() or S.sha_file(W116 / f) != r["sha256"]]
    return {"manifest_sha256": S.sha_file(W116 / "evidence_manifest.json"), "n_files": len(m), "mismatched": bad,
            "ok": S.sha_file(W116 / "evidence_manifest.json") == W116_MANIFEST_SHA and not bad}


# --------------------------------------------------------------------------------------------- preflight ---
def cmd_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"schema": "w117_preflight_v1", "stop": [], "runtime": C.runtime()}
    rep["head"], rep["instruction_blob"] = git("rev-parse", "HEAD"), git("rev-parse", "HEAD:docs/next_batch.md")
    rep["branch"], rep["status_porcelain"] = git("rev-parse", "--abbrev-ref", "HEAD"), git("status", "--porcelain")
    title = (ROOT / "docs/next_batch.md").read_text(encoding="utf-8").splitlines()[0]
    rep["instruction_title"] = title
    if rep["instruction_blob"] != INSTRUCTION_BLOB or not title.startswith("# BATCH 117"):
        rep["stop"].append("instruction is not the bound Batch 117 blob")
    if rep["head"] != STARTING_HEAD:
        rep["stop"].append("HEAD is not the bound starting commit")
    if rep["branch"] != BRANCH:
        rep["stop"].append("wrong branch")
    lock = load(LOCK) if LOCK.exists() else None
    rep["main_local_lock"] = lock
    if not lock or lock.get("batch") != BATCH or lock.get("instruction_blob") != INSTRUCTION_BLOB or lock.get("start_commit") != STARTING_HEAD:
        rep["stop"].append("Main-local lock is not this Batch 117 claim")
    hist = {}
    for n, wl in HIST.items():
        lines = (ROOT / wl).read_text(encoding="utf-8").rstrip().splitlines()
        hist[n] = {"marker": lines[-2:] == ["AUTOMATION_BATCH_STATUS: COMPLETE", f"BATCH_NUMBER: {n}"],
                   "blob_at_start": git("rev-parse", f"{STARTING_HEAD}:{wl}"), "blob_now": git("hash-object", str(ROOT / wl))}
        if not hist[n]["marker"] or hist[n]["blob_at_start"] != hist[n]["blob_now"]:
            rep["stop"].append(f"W{n} Worklog record not intact")
    rep["historical_worklogs"] = hist
    rep["w116_case_in_worklog"] = "CASE B" in (ROOT / HIST[116]).read_text(encoding="utf-8").split("## 2.")[0]
    # W116 authentication
    rep["w116_manifest"] = w116_manifest_check()
    if not rep["w116_manifest"]["ok"]:
        rep["stop"].append("W116 evidence manifest does not re-verify")
    rep["w116_declaration_sha256"] = S.sha_file(W116 / "results/declare/declaration.json")
    rep["w116_decision_sha256"] = S.sha_file(W116 / "results/decision/decision.json")
    rep["w116_case"] = load(W116 / "results/decision/decision.json")["case"]
    if rep["w116_declaration_sha256"] != W116_DECL_SHA or rep["w116_decision_sha256"] != W116_DECISION_SHA or rep["w116_case"] != "B":
        rep["stop"].append("W116 declaration/decision/CASE not the frozen ones")
    # inputs (W102/W105/W115, manifest-bound as in W116)
    m105 = load(C.INPUTS["w105_evidence_manifest"][0])["files"]
    m115 = load(C.INPUTS["w115_evidence_manifest"][0])["files"]
    files = {}
    for name, (path, want) in C.INPUTS.items():
        got = S.sha_file(path) if Path(path).exists() else None
        if want == "w105":
            want = m105.get(Path(path).relative_to(C.W105).as_posix(), {}).get("sha256")
        elif want == "w115":
            want = m115.get(Path(path).relative_to(C.W115).as_posix(), {}).get("sha256")
        files[name] = {"path": str(path), "sha256": got, "expected": want, "ok": got is not None and got == want}
        if not files[name]["ok"]:
            rep["stop"].append(f"input {name} missing or changed")
    rep["inputs"] = files
    audit = load(C.INPUTS["w105_render_clip_audit"][0])
    fa = audit["files"]["real_clip/real_clip.blend"]
    rep["w105_audit"] = {"value_digest": fa["actions"]["W91 real_clip FIXED"]["value_digest"], "scene_fps": fa.get("scene_fps"),
                         "interpolation": fa["actions"]["W91 real_clip FIXED"].get("interpolation"),
                         "fcurves": fa["actions"]["W91 real_clip FIXED"]["fcurves"]}
    # parent rebuild with the unchanged W116 code
    v0 = C.real_parent_state(C.INPUTS["w105_video_cases"][0])
    w116_v0 = load(W116 / "results/preflight/real_parent_state.json")
    rep["parent_rebuild"] = {"state_id": v0["state_id"], "w116_state_id": w116_v0["state_id"],
                             "canonical_equal": S.canon(v0) == S.canon(w116_v0), "verifies": S.verify_state(v0)}
    if not rep["parent_rebuild"]["canonical_equal"]:
        rep["stop"].append("rebuilt parent differs from the W116 parent")
    dump(out / "real_parent_state.json", v0)
    C.states_file(out / "real_parent_states.json", {"v0": v0}, "w105")
    rest = C.INPUTS["w105_rig_rest"][0]
    blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest", rest,
            "--states", out / "real_parent_states.json", "--out", out / "real_parent_eval.json", log=out / "real_parent_eval.log")
    blender("scripts/w117_blender_timing.py", "--", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest", rest, "--states",
            out / "real_parent_states.json", "--out", out / "real_timing.json", "--sample-frames", "1,109,118,153,154,197",
            log=out / "real_timing.log")
    ev = load(out / "real_parent_eval.json")
    w116p = load(W116 / "results/preflight/real_probe.json")
    pe = ev["states"]["v0"]
    rep["parent_blender"] = {"value_digest": pe["value_digest"], "fcurves": pe["fcurves"],
                             "equals_w105_fixed_digest": pe["value_digest"] == rep["w105_audit"]["value_digest"],
                             "frames_bit_equal_w116_probe": pe["frames"] == w116p["states"]["v0"]["frames"],
                             "rig_binding_equal_w116": ev["rig"] == w116p["rig"], "blender_version": ev["blender_version"]}
    if not (rep["parent_blender"]["equals_w105_fixed_digest"] and rep["parent_blender"]["frames_bit_equal_w116_probe"]):
        rep["stop"].append("fresh Blender evaluation of the parent does not reproduce W105/W116")
    rep["timing"] = load(out / "real_timing.json")
    # synthetic rig and parents
    dump(out / "syn_spec.json", SYN_SPEC)
    parents = {f"P__{n}": traj_state(n) for n in TRAJ}
    for n, st in parents.items():
        dump(out / f"syn_parent_{n}.json", st)
    C.states_file(out / "syn_parent_states.json", parents, "synthetic")
    blender("scripts/w116_blender_stages.py", "--", "probe", "--rig", f"synthetic:{out / 'syn_spec.json'}", "--states",
            out / "syn_parent_states.json", "--out", out / "syn_probe.json", "--probe-channels", "base,upper,lower",
            log=out / "syn_probe.log")
    blender("scripts/w117_blender_timing.py", "--", "--rig", f"synthetic:{out / 'syn_spec.json'}", "--states",
            out / "syn_parent_states.json", "--out", out / "syn_timing.json", log=out / "syn_timing.log")
    rep["syn_timing"] = load(out / "syn_timing.json")
    rep["syn_parent_state_ids"] = {n: s["state_id"] for n, s in parents.items()}
    rep["real_parent_state_id"] = v0["state_id"]
    rep["source_guard"] = guard().before
    dump(out / "preflight.json", rep)
    print(json.dumps({"stop": rep["stop"], "parent_blender": rep["parent_blender"], "w116_manifest_ok": rep["w116_manifest"]["ok"]},
                     indent=1))
    return 1 if rep["stop"] else 0


# --------------------------------------------------------------------------------------------- calibrate ---
class SynCtx:
    def __init__(self, pre):
        pre = Path(pre)
        self.probe = load(pre / "syn_probe.json")
        self.rig = syn_rig(self.probe)
        self.parents = {f"P__{n}": load(pre / f"syn_parent_P__{n}.json") for n in TRAJ}
        self.rest, self.scale = self.probe["rest"], 1.0

    def bind(self):
        frames = [fr for st in self.probe["states"].values() for fr in st["frames"]]
        ok, failed = {}, {}
        for j, c in SYN_BINDING_CANDIDATES.items():
            rec = S.verify_binding(j, c, SYN_SPEC["rig_id"], self.probe["rig"]["spec_sha256"], self.rest, frames, SYN_CH, self.rig)
            (ok.__setitem__(rec["binding_id"], rec) if rec["status"] == "BOUND" else failed.__setitem__(j, rec))
        return ok, failed


def syn_ctrl(rig, parent, edit_name):
    e = SYN_EDITS.get(edit_name) if edit_name else None
    out = {}
    for i, k in enumerate(parent["frames"]):
        vals = S.frame_vals(parent, i)
        out[k] = S.apply_edit(rig, vals, e, {"angle_rad": e["sign"] * math.radians(e["magnitude"]["angle_deg"])} if e else None)
    return out


def fk_pts(rig, vals, upper, lower):
    fk = rig.fk(vals)
    return rig.world(fk, upper, "head"), rig.world(fk, upper, "tail"), rig.world(fk, lower, "tail")


def cmd_calibrate(a):
    pre = Path(a.preflight)
    out = new_dir(a.out_dir)
    pf = load(pre / "preflight.json")
    if pf["stop"]:
        raise SystemExit("STOP: preflight failed")
    rep = {"schema": "w117_calibration_v1", "scope": "W116 control parity on historical W116 outputs; synthetic FK/bindings and test "
                                                   "design from parent/unpinned geometry only. No F/T candidate of W117 exists."}
    # ---- W116 framewise control parity (pure Python, all frames)
    decl116 = load(W116 / "results/declare/declaration.json")
    ctx = C.Ctx(decl116)
    B116 = load(W116 / "results/build_r1/build_report.json")
    par = {}
    for fam in REAL_FAMS:
        edit, pin_decl, auth, _ = C.REAL_FAMILIES[fam]
        ctrl, cand, mism = {}, {}, []
        for k in ctx.v0["frames"]:
            pins = C.make_pins(ctx, False, ctx.v0, pin_decl, [k], auth, cid_prefix=f"{fam}/{k}/")
            req = {"request_id": f"{fam}/{k}", "parent_state_id": ctx.v0["state_id"], "edit": C.edit_record(edit, [k], {}),
                   "constraints": pins, "resolver": {"method": "analytic"}}
            errs = S.validate_request(req, ctx.vctx(ctx.v0, False))
            recs, params, cv, pv = C.solve_request(ctx, False, ctx.v0, edit, pins, [k], solve=not errs)
            ctrl[k], cand[k] = cv[k], pv[k]
            w = B116["requests"][fam]["frames"][str(k)]
            if recs[k]["resolver_status"] != w["resolver_status"]:
                mism.append(k)
        ids = {"control": C.patched_state(ctx.v0, ctrl, {"kind": "EVAL_CONTROL", "family": fam})["state_id"],
               "candidate": C.patched_state(ctx.v0, cand, {"kind": "EVAL_CANDIDATE", "family": fam})["state_id"]}
        par[fam] = {"status_mismatch_frames": mism, "control_id_equal_w116": ids["control"] == B116["real_states"][f"{fam}__control"],
                    "candidate_id_equal_w116": ids["candidate"] == B116["real_states"][f"{fam}__candidate"]}
    for name, (cap, frames) in C.REAL_INTERVALS.items():
        pins = C.make_pins(ctx, False, ctx.v0, [("left_wrist", "capture", None)], frames, ARM, cap_frame=cap, cid_prefix=f"{name}/")
        _, _, _, pv = C.solve_request(ctx, False, ctx.v0, None, pins, frames)
        sid = C.patched_state(ctx.v0, pv, {"kind": "INTERVAL_CANDIDATE", "request": name})["state_id"]
        par[name] = {"candidate_id_equal_w116": sid == B116["real_states"][f"{name}__candidate"]}
    rep["w116_control_parity"] = par
    rep["w116_control_parity_ok"] = all(not v.get("status_mismatch_frames") and all(x for kk, x in v.items() if kk.endswith("equal_w116"))
                                        for v in par.values())
    # ---- synthetic FK and bindings
    sc = SynCtx(pre)
    worst = 0.0
    for n, st in sc.parents.items():
        for i, fr in enumerate(sc.probe["states"][n]["frames"]):
            fk = sc.rig.fk(S.frame_vals(st, i))
            for b in fr["world"]:
                for p in ("head", "tail"):
                    worst = max(worst, float(np.linalg.norm(sc.rig.world(fk, b, p) - np.array(fr["world"][b][p]))))
    rep["syn_fk_max_error_m"] = worst
    ok, failed = sc.bind()
    rep["syn_bindings"] = {"bound": ok, "failed": failed}
    rep["syn_coordinate_probes"] = C.coord_probes(sc.probe, 1.0)
    # ---- synthetic design (parent + unpinned control geometry only; Python FK, Blender-calibrated)
    L1, L2 = sc.rig.world_len("upper"), sc.rig.world_len("lower")
    rmin, rmax = S._limits_reach(L1, L2)
    design = {"L1": L1, "L2": L2, "reach_min_m": rmin, "reach_max_m": rmax}
    for h, (tr, edit, cap, (lo, hi)) in SYN_HOLDS.items():
        P = sc.parents[f"P__{tr}"]
        U = syn_ctrl(sc.rig, P, edit)
        _, _, Wp = fk_pts(sc.rig, S.frame_vals(P, cap), "upper", "lower")
        Ss = {k: fk_pts(sc.rig, U[k], "upper", "lower") for k in SYN_FRAMES}
        rec = {"trajectory": tr, "edit": edit, "capture": cap, "interval": [lo, hi]}
        offset = [0.0, 0.0, 0.0]
        if h in ("H_C", "H_Cx"):
            d = Wp - Ss[cap][0]
            d = d / np.linalg.norm(d)
            want = rmax - NEAR_SINGULAR_DESIGN_SLACK_M if h == "H_C" else rmax + INFEASIBLE_DESIGN_EXCESS_M
            lo_s, hi_s = -0.3, 0.2
            for _ in range(200):
                mid = (lo_s + hi_s) / 2
                if max(np.linalg.norm(Wp + mid * d - Ss[k][0]) for k in range(lo, hi + 1)) > want:
                    hi_s = mid
                else:
                    lo_s = mid
            offset = (lo_s * d).tolist()
        T = Wp + np.array(offset)
        dist = {k: float(np.linalg.norm(T - Ss[k][0])) for k in range(lo, hi + 1)}
        rec.update(offset_m=offset, target=T.tolist(), slack_max_m={str(k): rmax - v for k, v in dist.items()},
                   slack_min_m={str(k): v - rmin for k, v in dist.items()},
                   min_slack_m=min(rmax - v for v in dist.values()),
                   design_feasible_all=all(rmin - S.TOL_M < v <= rmax + S.TOL_M for v in dist.values()),
                   entry_offset_m=float(np.linalg.norm(T - Ss[lo][2])), release_offset_m=float(np.linalg.norm(T - Ss[hi][2])),
                   pre_entry_to_target_m=float(np.linalg.norm(T - Ss[lo - 1][2])),
                   post_release_to_target_m=float(np.linalg.norm(T - Ss[hi + 1][2])),
                   entry_vs_release_pose_m=float(np.linalg.norm(Ss[lo - 1][2] - Ss[hi + 1][2])),
                   edit_wrist_move_min_m=min(float(np.linalg.norm(Ss[k][2] - fk_pts(sc.rig, S.frame_vals(P, k), "upper", "lower")[2]))
                                             for k in range(lo, hi + 1)),
                   control_flex_deg={str(k): math.degrees(math.acos(max(-1, min(1, float(np.dot(
                       (Ss[k][1] - Ss[k][0]) / L1, (Ss[k][2] - Ss[k][1]) / L2)))))) for k in range(lo, hi + 1)})
        design[h] = rec
    # per-frame near-singular family on trajectory C: sign rule
    pf_c = {}
    PC = sc.parents["P__C"]
    for e in ("SYN_E3_POS", "SYN_E3_NEG"):
        U = syn_ctrl(sc.rig, PC, e)
        sl = []
        for k in SYN_FRAMES:
            _, _, Wp = fk_pts(sc.rig, S.frame_vals(PC, k), "upper", "lower")
            sl.append(rmax - float(np.linalg.norm(Wp - fk_pts(sc.rig, U[k], "upper", "lower")[0])))
        pf_c[e] = {"n_feasible": sum(1 for s in sl if s > S.BOUNDARY_BAND_M - S.TOL_M), "min_slack_m": min(sl), "slack_m": sl}
    pf_c["chosen"] = max(("SYN_E3_POS", "SYN_E3_NEG"), key=lambda e: (pf_c[e]["n_feasible"], e == "SYN_E3_POS"))
    design["PF_C"] = pf_c
    # distal-only control target (N3): forearm rotated 20 deg about the elbow-local axis at frame 14 of trajectory A, no edit
    PA = sc.parents["P__A"]
    Sa, Ea, Wa = fk_pts(sc.rig, S.frame_vals(PA, 14), "upper", "lower")
    ax = np.cross(Ea - Sa, Wa - Ea)
    ax = ax / np.linalg.norm(ax)
    Wn = Ea + S.q_to_m(S.q_axis_angle(ax, math.radians(20))) @ (Wa - Ea)
    design["N3_offset_m"] = (Wn - Wa).tolist()
    rep["design"] = design
    dump(out / "calibration.json", rep)
    print(json.dumps({"parity_ok": rep["w116_control_parity_ok"], "syn_fk": worst,
                      "bound": sorted(b["semantic_joint"] for b in ok.values()), "failed": {k: v["status"] for k, v in failed.items()},
                      "design": {h: {k: v for k, v in r.items() if k in ("min_slack_m", "design_feasible_all", "entry_offset_m",
                                                                              "release_offset_m", "entry_vs_release_pose_m",
                                                                              "edit_wrist_move_min_m", "offset_m")}
                                 for h, r in design.items() if h.startswith("H_")},
                      "PF_C": {e: {k: v for k, v in r.items() if k != "slack_m"} for e, r in pf_c.items() if e != "chosen"},
                      "PF_C_chosen": pf_c["chosen"]}, indent=1, default=str))
    return 0


# --------------------------------------------------------------------------------------------- declare ---
GATES = {
    "V1_authentication": "preflight has no STOP: HEAD, blob, branch, lock, W113-W116 Worklogs intact, W116 manifest (all files) "
                         "re-verifies, W116 declaration/decision hashes and CASE B, every W102/W105/W115 input hash",
    "V2_parent_reproduction": "rebuilt parent is canonical-equal to the W116 parent; fresh Blender evaluation has the W105 FIXED value "
                              "digest and is bit-equal to the W116 probe on all 199 frames",
    "V3_w116_control_parity": "W116 resolver recomputed with unchanged W116 code reproduces W116 statuses and state ids on all frames of "
                              "RA/RB/RC and RH1/RH2 (calibration); W117 F/U whole-clip states have the W116 ids and evaluate bit-equal to "
                              "the W116 Blender evaluation on all 199 frames",
    "V4_independent_evaluator": "synthetic FK <= 1e-5 m; declared synthetic bindings BOUND and fail-safe ones fail; a claimed-SOLVED "
                                "unpinned candidate is rejected EVALUATED_PIN_UNSATISFIED (synthetic N10 and real RN10)",
    "V5_meaningful_asymmetric_design": "synthetic edit moves the unpinned wrist >= 0.01 m on every hold frame of H_A/H_B/H_C; for "
                                       "H_A/H_B/H_C (Blender-evaluated U) entry and release wrist offsets from the target are each "
                                       ">= 0.005 m and differ by >= 0.005 m, and pre-entry and post-release unpinned wrists differ by "
                                       ">= 0.005 m; real RB edits move the wrist > TOL_M on every frame",
    "V6_timing_semantics": "every keyframe CONSTANT; Blender frame = index - f0 + 1; sub-frame evaluation equals the keyed frame",
    "V7_oracle_soundness": "no candidate frame whose oracle class is INFEASIBLE meets the pin; synthetic oracle classes equal the "
                           "design classes (H_A/H_B/H_C all FEASIBLE; H_Cx has >= 1 INFEASIBLE frame)",
    "V8_repeatability": "two builds byte-identical states files and identical solver decisions; two Blender evaluations identical; the "
                        "build-time U evaluation equals the evaluate-stage U evaluation",
    "V9_population_complete": "every declared request and every pin frame of every candidate has a Blender-evaluated result",
    "C1_accept_soundness": "every ACCEPTed F/T/TE request: every pin frame residual <= 1 mm (Blender), stored changes vs U only in "
                           "authorised arm channels and only on declared frames, no point outside the arm domain changed vs U, bone "
                           "lengths, root unchanged, joint limits",
    "C2_negative_controls": "every declared negative control returns its declared decision and failure class",
    "C3_atomicity": "every multi-frame request is ACCEPT iff all its pin frames pass (NO_CHANGE_NEEDED passes) and its resolver "
                    "statuses are valid; the single-corrupted-frame intervals (N13, RN13) are REJECTed while their other frames pass",
    "C4_lineage_reversal": "refs unchanged after every REJECT; stale parents rejected; each side has an executed ACCEPT and an exact "
                           "revert (values byte-equal, Blender bit-equal to the frozen parent)",
    "C5_source_immutability": "declared inputs, W116 artefacts and parent files byte-unchanged across every stage; W116 manifest "
                              "re-verifies after the run",
    "C6_no_new_hard_failure": "T and TE ACCEPT every contract-valid evaluation request whose pin frames are all oracle-FEASIBLE "
                              "(no new hard failure relative to F); F also ACCEPTs them",
    "Q_syn(v)": "for variant v in {T, TE}: >= 1 asymmetric synthetic evaluation request (H_A, H_B, H_C, PF_C requests) with both F "
                "and v ACCEPTed shows a material improvement and no material worsening, and no synthetic evaluation request with both "
                "ACCEPTed shows a material worsening",
    "Q_real(v)": "for variant v: >= 1 real request with both F and v ACCEPTed shows a material improvement and no material worsening, "
                 "and no real request with both ACCEPTed shows a material worsening",
    "QV(v)": "the recorded visual review marks, for >= 1 synthetic and >= 1 real request counted in Q_syn(v)/Q_real(v), the improvement "
             "as visible and the v arm as not implausible",
}
CASE_RULE = ("D if any V fails. Else C if any C gate fails, or if Q_syn(v) and Q_real(v) both fail for every variant. Else A if some "
             "variant v has Q_syn(v), Q_real(v) and QV(v). Else B.")
METRICS = {
    "time_unit": "rig-Action frame (Blender frame = frame_index - f0 + 1). Real clip: one Action frame = one 50 fps source sample "
                 "(0.02 s source time) while the historical Action is played at 30 fps; no seconds conversion is claimed. Synthetic: "
                 "declared frames.",
    "sampling": "integer Action frames only (CONSTANT keys)",
    "window": "holds [a-MARGIN-1, b+MARGIN+1]; per-frame segments [lo-1, hi+1]; whole-clip all frames (clipped to the clip)",
    "D1": "introduced bend flips: transitions where the bend-plane normal (E-S)x(W-E) reverses (dot < 0) between consecutive frames, "
          "both frames with flex >= FLIP_FLEX_FLOOR_DEG, and the unpinned control has no flip at that transition",
    "D2": "peak elbow compensation step: max_k |(E^X_k - E^U_k) - (E^X_{k-1} - E^U_{k-1})| (m)",
    "D3": "peak authorised-arm rotation step excess over U: max_k max_ch (w^X_k - w^U_k), w = sign-safe geodesic angle between "
          "consecutive stored quaternions (deg/frame)",
    "D4": "holds only: peak boundary wrist compensation step over entry transitions [a-MARGIN, a] and release transitions "
          "[b+1, b+MARGIN+1] (m)",
    "improved": "D1: F >= 1 and T < F; D2-D4: T <= 0.5 F and F - T >= floor (D2 0.005 m, D3 5 deg/frame, D4 0.005 m)",
    "worse": "D1: T > F; D2-D4: T >= 2 F and T - F >= floor; side effects: S1 mean elbow deviation from U > F + 0.010 m, S2 peak "
             "upper-arm compensating rotation vs U > F + 20 deg, S3 peak |flex - flex_U| > F + 10 deg",
    "material_improvement": "improved non-empty and worse empty",
    "reported_not_gated": "pin residual max/p95/failures; wrist/elbow steps; flex; bend-plane step; second-order rotation; cumulative/"
                          "peak compensating rotation; near-singular frames (slack < 5 mm); straight frames (flex < 5 deg); entry/"
                          "release jumps; frames changed; right wrist / neck / root movement vs U and parent; solver J, iterations, "
                          "runtime, convergence",
    "thresholds_are_diagnostic": "experiment-internal, not user-facing policy",
}


def cmd_declare(a):
    out = new_dir(a.out_dir)
    pf = load(Path(a.preflight) / "preflight.json")
    cal = load(Path(a.calibration) / "calibration.json")
    if pf["stop"]:
        raise SystemExit("STOP: preflight failed")
    d = cal["design"]
    decl = {
        "schema": "w117_declaration_v1", "batch": BATCH, "outcomes_inspected": False,
        "preflight_dir": str(Path(a.preflight).resolve()), "calibration_dir": str(Path(a.calibration).resolve()),
        "preflight_sha256": S.sha_file(Path(a.preflight) / "preflight.json"),
        "calibration_sha256": S.sha_file(Path(a.calibration) / "calibration.json"),
        "w116": {"manifest_sha256": W116_MANIFEST_SHA, "declaration_sha256": W116_DECL_SHA, "decision_sha256": W116_DECISION_SHA,
                 "case": "B (frozen; not reassessed)"},
        "hard_pin": {"tolerance_m": S.TOL_M, "evaluator": "Blender scene world (w116_blender_stages evaluate, unchanged)",
                     "status": "W116 provisional diagnostic criterion, unchanged"},
        "solver": {"T": "temporal_two_bone over the pin frames only; objective J = sum|c|^2 + LAMBDA sum|dc|^2, c = body-frame elbow "
                        "compensation / L1; Viterbi DP init on GRID swivel bins + L-BFGS-B from DP and from the F swivel; lower J kept; "
                        "converged iff max|dJ/dphi| <= GTOL; frames already within TOL under U are anchored (NO_CHANGE_NEEDED, values = U); "
                        "same reach band, clamp rule and distal aim as W116; no fallback to F; non-two-bone structures -> "
                        "RESOLVER_UNSUPPORTED (SOLVER_FAILURE) or INSUFFICIENT_DOF",
                   "TE": "T plus MARGIN guide frames on each side of a hold: guide = unpinned wrist + boundary offset * "
                         "(1 - smoothstep(j/(MARGIN+1))); guide frames are NOT pins, are clamped to reach, and change only the "
                         "authorised arm channels",
                   "constants": {"LAMBDA": TB.LAMBDA, "GRID": TB.GRID, "MAXITER": TB.MAXITER, "GTOL": TB.GTOL,
                                 "RUNTIME_BUDGET_S": TB.RUNTIME_BUDGET_S, "MARGIN": TB.MARGIN},
                   "authorised_dof": {"real": ARM, "synthetic": SYN_ARM},
                   "never_authorised": "user-edit channels (spine_01 / base), __object__, root bones, unkeyed channels",
                   "tuning": "none; LAMBDA/GRID/MAXITER/GTOL/MARGIN fixed a priori. Development split = trajectory D (H_D), used "
                             "only to smoke-test the solver before this declaration; excluded from every gate."},
        "metrics": METRICS | {"constants": {"FLIP_FLEX_FLOOR_DEG": TB.FLIP_FLEX_FLOOR_DEG, "NEAR_SINGULAR_SLACK_M": TB.NEAR_SINGULAR_SLACK_M,
                                            "STRAIGHT_FLEX_DEG": TB.STRAIGHT_FLEX_DEG, "FLOORS": TB.FLOORS, "WORSE_SIDE": TB.WORSE_SIDE}},
        "synthetic": {"spec": SYN_SPEC, "trajectories": TRAJ, "edits": SYN_EDITS, "holds": SYN_HOLDS, "design": d,
                      "pf_c_edit": d["PF_C"]["chosen"]},
        "real": {"families": {f: {"edit": C.REAL_FAMILIES[f][0], "pins": C.REAL_FAMILIES[f][1]} for f in REAL_FAMS},
                 "holds": {f"H_{h}_{e}": {"interval": REAL_HOLDS[h][1], "capture": REAL_HOLDS[h][0], "edit": HOLD_EDITS[e]}
                           for h in REAL_HOLDS for e in HOLD_EDITS},
                 "low_evidence_windows": WINDOWS, "segment_rule": "maximal runs (length >= 2) of oracle-FEASIBLE frames of a per-frame "
                 "family, oracle from the Blender-evaluated U; only when a run is not the whole clip; whole-clip requests for every "
                 "family (expected REJECT where any frame is infeasible)",
                 "timebase": "source 50 fps, historical Action 30 fps; not repaired"},
        "requests_roles": {"evaluation": "H_A, H_B, H_C, PF_C whole/segments; real W_*, S_*, H_*", "noop": "NOOP_A (trivially "
                           "satisfied, not evidence)", "negative": "H_Cx, N2-N14, RN10-RN13", "development": "H_D"},
        "negative_controls": NEG_EXPECT,
        "lineage": {"synthetic": ["S1 commit the first ACCEPTed T request among [H_A, H_B, H_C] on ref syn/main (parent = its trajectory "
                                  "parent)", "S2 on ref syn/atomic commit N13 (single corrupted frame) -> REJECT, head unchanged",
                                  "S3 commit the TE candidate of the same request prepared against the old head -> STALE_PARENT",
                                  "S4 revert syn/main -> values byte-equal, Blender bit-equal to the parent"],
                    "real": ["R1 commit the first ACCEPTed T request in declared real order on ref real/main", "R2 on ref real/atomic "
                             "commit RN13 -> REJECT, head unchanged", "R3 commit another real candidate prepared against v0 -> "
                             "STALE_PARENT", "R4 on ref real/atomic commit the T candidate of H_RH2_none (decision as evaluated)",
                             "R5 revert real/main -> values byte-equal, Blender bit-equal to v0"]},
        "visual_review": "after render and before decide, the agent records visual_review.json (per rendered request: defect visible in "
                         "F, reduced in v, v arm plausible, abrupt start/stop, merely numerical, animator usefulness); decide reads it",
        "render_rule": "synthetic H_A, H_B, H_C full windows; PF_C request with the largest F D2; real H_RH1_none, H_RH1_RBneg, "
                       "H_RH2_RBneg full windows; RB_neg whole-clip +-4 frames around the F D2 peak; the RB_pos segment with the "
                       "largest F D2. Bone proxies, mesh hidden, fixed front camera centred on the first pin target.",
        "predictions_not_gates": {
            "flex": "elbow flex is fixed by |target - shoulder| for a two-bone chain; T cannot reduce flex compensation; only the "
                    "swivel (bend plane) and through it upper-arm rotation and elbow path are free",
            "entry_release": "F and T have identical entry/release wrist jumps (same pin frames); only TE can spread them",
            "real": "real whole-clip/segment requests: T mostly equal to F where the control elbow is well conditioned; differences "
                    "concentrate in near-straight frames",
        },
        "gates": GATES, "case_rule": CASE_RULE,
        "pre_declaration_disclosure": [
            "focused unit tests (tests/test_w117_temporal_pin.py) on a Python-only test rig; one test threshold corrected "
            "(smoothstep(4/5) leaves 10.4 %, the test had asserted < 10 %)",
            "preflight and calibration as recorded (parents, unpinned controls, W116 parity, synthetic design geometry)",
            "development-split smoke run of T on H_D (trajectory D, Python FK only, no Blender): SOLVED, converged, J_T <= J_F; "
            "and a 199-frame synthetic development sequence for runtime (0.5 s). No evaluation request was solved.",
            "no parameter was changed after the smoke run"],
    }
    dump(out / "declaration.json", decl)
    print(S.sha_file(out / "declaration.json"))
    return 0


NEG_EXPECT = {
    "H_Cx": {"desc": "near-singular hold with target beyond reach on some frames", "F": ["REJECT", "INFEASIBLE_REQUEST"],
             "T": ["REJECT", "INFEASIBLE_REQUEST"], "TE": ["REJECT", "INFEASIBLE_REQUEST"]},
    "N2": {"desc": "H_A with no authorised DOF", "F": ["REJECT", "INFEASIBLE_REQUEST"], "T": ["REJECT", "INFEASIBLE_REQUEST"]},
    "N3": {"desc": "single-frame forearm-only (aim_distal) feasible target; T supports two-bone only", "F": ["ACCEPT", "NONE"],
           "T": ["REJECT", "SOLVER_FAILURE"]},
    "N4": {"desc": "pin on AMBIGUOUS syn_hand binding", "T": ["REJECT", "UNSUPPORTED_BINDING"]},
    "N5": {"desc": "binding record of another rig (W116 synthetic syn_wrist) injected", "T": ["REJECT", "UNSUPPORTED_BINDING"]},
    "N6": {"desc": "two wrist pins 0.02 m apart over overlapping frames", "T": ["REJECT", "INFEASIBLE_REQUEST"]},
    "N7": {"desc": "target altered and re-digested (provenance mismatch)", "T": ["REJECT", "CONTRACT_VIOLATION"]},
    "N9": {"desc": "H_A with T maxiter = 1", "T": ["REJECT", "SOLVER_FAILURE"]},
    "N10": {"desc": "H_A T claims SOLVED but returns the unpinned control", "T": ["REJECT", "SOLVER_FAILURE"]},
    "N11": {"desc": "H_A T with base (user-edit channel) +1 deg on frame 20", "T": ["REJECT", "CONTRACT_VIOLATION"]},
    "N12": {"desc": "H_A T with __object__ +1 deg on frame 20", "T": ["REJECT", "CONTRACT_VIOLATION"]},
    "N13": {"desc": "H_A T with lower +2 deg on frame 20 (single corrupted pin frame)", "T": ["REJECT", "SOLVER_FAILURE"]},
    "N14": {"desc": "H_A TE with upper changed on frame 5 (outside the declared temporal domain)", "TE": ["REJECT", "CONTRACT_VIOLATION"]},
    "RN10": {"desc": "H_RH1_none T claims SOLVED but returns U", "T": ["REJECT", "SOLVER_FAILURE"]},
    "RN11": {"desc": "H_RH1_none T with spine_01 +1 deg on frame 113", "T": ["REJECT", "CONTRACT_VIOLATION"]},
    "RN12": {"desc": "H_RH1_none T with __object__ +1 deg on frame 113", "T": ["REJECT", "CONTRACT_VIOLATION"]},
    "RN13": {"desc": "H_RH1_none T with lowerarm_l +2 deg on frame 113", "T": ["REJECT", "SOLVER_FAILURE"]},
}


# --------------------------------------------------------------------------------------------- build ---
def rot_local(q, axis, deg):
    return S.q_mul(np.asarray(q, float), S.q_axis_angle(axis, math.radians(deg)))


def vals_to_lists(v):
    return {c: [float(x) for x in q] for c, q in v.items()}


def full_patch(parent, ctrl, cand_frames):
    """Patches: control values on every frame, candidate values on the given frames."""
    return {k: (cand_frames[k] if k in cand_frames else ctrl[k]) for k in parent["frames"]}


def run_T(rig, frames, ctrl, targets, kinds, auth, upper, lower, struct, maxiter=TB.MAXITER):
    st, vals, info = TB.temporal_two_bone(rig, frames, ctrl, targets, kinds, auth, upper, lower, maxiter=maxiter, pins_struct=struct)
    return st, vals, info


class Build:
    def __init__(self, decl, out):
        self.decl, self.out = decl, out
        self.ctx = C.Ctx(load(W116 / "results/declare/declaration.json"))
        self.sc = SynCtx(decl["preflight_dir"])
        cal = load(Path(decl["calibration_dir"]) / "calibration.json")
        self.syn_b, self.syn_fb = cal["syn_bindings"]["bound"], cal["syn_bindings"]["failed"]
        self.design = cal["design"]
        self.cap = {}
        for n, st in self.sc.parents.items():
            for fr in self.sc.probe["states"][n]["frames"]:
                for bid, rec in self.syn_b.items():
                    self.cap[(st["state_id"], fr["frame_index"], bid)] = fr["world"][rec["bone"]][rec["point"]]
        self.real, self.syn, self.req = {"v0": self.ctx.v0}, dict(self.sc.parents), {}

    # ---------------- synthetic helpers
    def syn_bid(self, joint):
        return next((k for k, v in self.syn_b.items() if v["semantic_joint"] == joint), None)

    def syn_pins(self, parent, pin_decl, frames, auth, cap_frame, prefix):
        out = []
        for i, (joint, kind, off) in enumerate(pin_decl):
            bid = self.syn_bid(joint) or f"UNBOUND:{joint}"
            src = {"kind": "CAPTURED_FROM_EVALUATED_STATE" if kind == "capture" else "CAPTURED_PLUS_DECLARED_OFFSET",
                   "state_id": parent["state_id"], "frame_index": cap_frame, "binding_id": bid,
                   "evaluated_by": "w116_blender_stages probe (W117 preflight)"}
            if kind == "offset":
                src["offset_m"] = off
            t = S.captured_target(self.cap, src)
            out.append(S.make_pin(f"{prefix}{i}", joint, bid, [0.0, 0.0, 0.0] if t is None else t.tolist(), src, frames, auth,
                                  parent["state_id"]))
        return out

    def syn_vctx(self, parent, extra_bindings=None):
        b = dict(self.syn_b) | (extra_bindings or {})
        return {"bindings": b, "failed_bindings": self.syn_fb, "capture": self.cap, "parent": parent, "root_bones": {"base"}}

    def rp(self, bindings, pins, k):
        return [{"bone": bindings[c["binding_id"]]["bone"], "point": bindings[c["binding_id"]]["point"], "target": np.array(c["target"]),
                 "semantic_joint": c["semantic_joint"]} for c in pins if c["binding_id"] in bindings and k in c["frames"]]

    def framewise(self, rig, ctrl, bindings, pins, frames, auth, method="analytic"):
        st, vals, info = {}, {}, {}
        for k in frames:
            r = self.rp(bindings, pins, k)
            if r:
                st[k], vals[k], info[k] = S.resolve_frame(rig, ctrl[k], r, auth, method=method)
            else:
                st[k], vals[k], info[k] = "NOT_APPLICABLE", ctrl[k], {}
        return st, vals, info

    def temporal(self, rig, ctrl, bindings, pins, frames, auth, upper, lower, margins=None, maxiter=TB.MAXITER, all_frames=None):
        kinds, targets = {}, {}
        for k in frames:
            r = self.rp(bindings, pins, k)
            kinds[k], targets[k] = "PIN", (r[0]["target"] if r else None)
        r0 = self.rp(bindings, pins, frames[0])
        if len(r0) != 1 or any(len(self.rp(bindings, pins, k)) != 1 for k in frames):
            struct = (None, "TEMPORAL_RESOLVER_SINGLE_PIN_PER_FRAME_ONLY")
        else:
            struct = S._structure(rig, r0, auth)
        run = list(frames)
        if margins:
            g = TB.guide_targets(rig, ctrl, lower, frames[0], frames[-1], targets[frames[0]], TB.MARGIN, all_frames)
            for k, t in g.items():
                kinds[k], targets[k] = "GUIDE", t
            run = sorted(set(frames) | set(g))
        if struct[0] != "two_bone":
            st, vals, info = TB.temporal_two_bone(rig, run, ctrl, targets, kinds, auth, upper, lower, pins_struct=struct)
        else:
            st, vals, info = run_T(rig, run, ctrl, targets, kinds, auth, upper, lower, struct, maxiter)
        return st, vals, info, run

    def add_req(self, rid, **kw):
        self.req[rid] = kw

    # ---------------- synthetic requests
    def build_syn(self):
        sc, rig = self.sc, self.sc.rig
        U_cache = {}

        def U_of(tr, edit):
            key = f"U__{tr}__{edit}"
            if key not in U_cache:
                P = sc.parents[f"P__{tr}"]
                U_cache[key] = syn_ctrl(rig, P, edit)
                self.syn[key] = C.patched_state(P, U_cache[key], {"kind": "EVAL_CONTROL", "trajectory": tr, "edit": edit})
            return key, U_cache[key]

        def hold(rid, tr, edit, cap, lo, hi, offset, role, auth=SYN_ARM, variants=("F", "T", "TE"), maxiter=TB.MAXITER,
                 mutate=None, extra_b=None, pin_decl=None):
            P = sc.parents[f"P__{tr}"]
            ukey, ctrl = U_of(tr, edit)
            frames = list(range(lo, hi + 1))
            pd = pin_decl or [("syn_wrist", "offset" if offset is not None else "capture", offset)]
            pins = self.syn_pins(P, pd, frames, auth, cap, f"{rid}/")
            pins = C.mutate_pins(pins, mutate, "0" * 64)
            req = {"request_id": rid, "parent_state_id": P["state_id"],
                   "edit": dict(SYN_EDITS[edit], edit_id=edit, frames=SYN_FRAMES) if edit else None,
                   "constraints": pins, "resolver": {"T": {"maxiter": maxiter}}}
            errs = S.validate_request(req, self.syn_vctx(P, extra_b))
            b = dict(self.syn_b) | (extra_b or {})
            cands = {}
            for v in variants:
                if errs:
                    cands[v] = {"state": ukey, "statuses": {}, "info": {"not_resolved": "CONTRACT_INVALID"}, "pin_frames": frames,
                                "margin_frames": []}
                    continue
                if v == "F":
                    st, vals, info = self.framewise(rig, ctrl, b, pins, frames, auth)
                    run, margins = frames, []
                else:
                    st, vals, info, run = self.temporal(rig, ctrl, b, pins, frames, auth, "upper", "lower", margins=(v == "TE"),
                                                        maxiter=maxiter, all_frames=SYN_FRAMES)
                    margins = [k for k in run if k not in frames]
                name = f"{rid}__{v}"
                self.syn[name] = C.patched_state(P, full_patch(P, ctrl, {k: vals[k] for k in run}), {"kind": f"{v}_CANDIDATE", "request": rid})
                cands[v] = {"state": name, "statuses": {str(k): st[k] for k in frames}, "info": info if v != "F" else
                            {str(k): info[k] for k in frames}, "pin_frames": frames, "margin_frames": margins,
                            "values": {str(k): vals_to_lists(vals[k]) for k in run}}
            self.add_req(rid, side="syn", kind="hold", role=role, parent=f"P__{tr}", U=ukey, edit=edit, auth=auth, request=req,
                         validation_errors=errs, pin_frames=frames, window=[max(0, lo - TB.MARGIN - 1), min(N_SYN - 1, hi + TB.MARGIN + 1)],
                         boundary=[lo, hi, TB.MARGIN], eval_frames=SYN_FRAMES, candidates=cands, chain=["upper", "lower"])

        for h in ("H_A", "H_B", "H_C", "H_Cx", "H_D"):
            tr, edit, cap, (lo, hi) = SYN_HOLDS[h]
            off = self.design[h]["offset_m"] if h in ("H_C", "H_Cx") else None
            role = {"H_Cx": "negative", "H_D": "development"}.get(h, "evaluation")
            hold(h, tr, edit, cap, lo, hi, off, role)
        tr, edit, cap, (lo, hi) = SYN_HOLDS["H_A"]
        hold("N2", tr, edit, cap, lo, hi, None, "negative", auth=[], variants=("F", "T"))
        hold("N3", "A", None, 14, 14, 14, self.design["N3_offset_m"], "negative", auth=["lower"], variants=("F", "T"))
        hold("N4", tr, edit, cap, lo, hi, None, "negative", variants=("T",), pin_decl=[("syn_hand", "capture", None)])
        w116_syn = load(W116 / "results/calibrate/calibration.json")["bindings"]["synthetic"]["bound"]
        wb = next((k, v) for k, v in w116_syn.items() if v["semantic_joint"] == "syn_wrist")
        hold("N6", tr, edit, cap, lo, hi, None, "negative", variants=("T",),
             pin_decl=[("syn_wrist", "capture", None), ("syn_wrist", "offset", [0.0, 0.0, 0.02])])
        hold("N7", tr, edit, cap, lo, hi, None, "negative", variants=("T",), mutate="target_altered_redigested")
        hold("N9", tr, edit, cap, lo, hi, None, "negative", variants=("T",), maxiter=1)
        # N5: pin bound to another rig's binding record
        P = sc.parents["P__A"]
        ukey, ctrl = U_of("A", edit)
        frames = list(range(lo, hi + 1))
        src = {"kind": "CAPTURED_FROM_EVALUATED_STATE", "state_id": P["state_id"], "frame_index": cap, "binding_id": wb[0],
               "evaluated_by": "w116 binding record injected"}
        pins = [S.make_pin("N5/0", "syn_wrist", wb[0], self.cap[(P["state_id"], cap, self.syn_bid("syn_wrist"))], src, frames,
                           SYN_ARM, P["state_id"])]
        req = {"request_id": "N5", "parent_state_id": P["state_id"], "edit": dict(SYN_EDITS[edit], edit_id=edit, frames=SYN_FRAMES),
               "constraints": pins, "resolver": {}}
        errs = S.validate_request(req, self.syn_vctx(P, {wb[0]: wb[1]}))
        self.add_req("N5", side="syn", kind="hold", role="negative", parent="P__A", U=ukey, edit=edit, auth=SYN_ARM, request=req,
                     validation_errors=errs, pin_frames=frames, window=[lo - 5, hi + 5], boundary=[lo, hi, TB.MARGIN], eval_frames=SYN_FRAMES,
                     candidates={"T": {"state": ukey, "statuses": {}, "info": {"not_resolved": "CONTRACT_INVALID"}, "pin_frames": frames,
                                       "margin_frames": []}}, chain=["upper", "lower"])
        # tamper controls on H_A T / TE
        base = self.req["H_A"]
        Tv = {int(k): {c: np.array(v) for c, v in x.items()} for k, x in base["candidates"]["T"]["values"].items()}
        TEv = {int(k): {c: np.array(v) for c, v in x.items()} for k, x in base["candidates"]["TE"]["values"].items()}
        _, ctrlA = U_of("A", edit)
        PA = sc.parents["P__A"]

        def tamper(rid, src_vals, statuses, variant, fn, margins):
            vv = {k: dict(v) for k, v in src_vals.items()}
            fn(vv)
            name = f"{rid}__{variant}"
            self.syn[name] = C.patched_state(PA, full_patch(PA, ctrlA, vv), {"kind": "TAMPERED", "request": rid})
            r = copy.deepcopy({k: v for k, v in base.items() if k != "candidates"})
            r["role"] = "negative"
            r["candidates"] = {variant: {"state": name, "statuses": statuses, "info": {"tamper": NEG_EXPECT[rid]["desc"]},
                                         "pin_frames": base["pin_frames"], "margin_frames": margins}}
            self.req[rid] = r

        stT = base["candidates"]["T"]["statuses"]
        tamper("N10", {k: ctrlA[k] for k in Tv}, {k: "SOLVED" for k in stT}, "T", lambda vv: None, [])
        tamper("N11", Tv, stT, "T", lambda vv: vv[20].__setitem__("base", rot_local(vv[20]["base"], [1, 0, 0], 1.0)), [])
        tamper("N12", Tv, stT, "T", lambda vv: vv[20].__setitem__("__object__", rot_local(vv[20]["__object__"], [0, 0, 1], 1.0)), [])
        tamper("N13", Tv, stT, "T", lambda vv: vv[20].__setitem__("lower", rot_local(vv[20]["lower"], [0, 0, 1], 2.0)), [])

        def n14(vv):
            vv[5] = dict(ctrlA[5])
            vv[5]["upper"] = rot_local(ctrlA[5]["upper"], [0, 0, 1], 1.0)
        tamper("N14", TEv, base["candidates"]["TE"]["statuses"], "TE", n14, base["candidates"]["TE"]["margin_frames"])
        # no-op control: trajectory A, no edit, every frame pinned to its own captured wrist
        ukey0, ctrl0 = U_of("A", None)
        pins = [p for k in SYN_FRAMES for p in self.syn_pins(PA, [("syn_wrist", "capture", None)], [k], SYN_ARM, k, f"NOOP_A/{k}/")]
        req = {"request_id": "NOOP_A", "parent_state_id": PA["state_id"], "edit": None, "constraints": pins, "resolver": {}}
        errs = S.validate_request(req, self.syn_vctx(PA))
        cands = {}
        for v in ("F", "T"):
            if v == "F":
                st, vals, info = self.framewise(rig, ctrl0, self.syn_b, pins, SYN_FRAMES, SYN_ARM)
            else:
                st, vals, info, _ = self.temporal(rig, ctrl0, self.syn_b, pins, SYN_FRAMES, SYN_ARM, "upper", "lower")
            name = f"NOOP_A__{v}"
            self.syn[name] = C.patched_state(PA, full_patch(PA, ctrl0, vals), {"kind": f"{v}_CANDIDATE", "request": "NOOP_A"})
            cands[v] = {"state": name, "statuses": {str(k): st[k] for k in SYN_FRAMES}, "info": info if v == "T" else {},
                        "pin_frames": SYN_FRAMES, "margin_frames": []}
        self.add_req("NOOP_A", side="syn", kind="whole", role="noop", parent="P__A", U=ukey0, edit=None, auth=SYN_ARM, request=req,
                     validation_errors=errs, pin_frames=SYN_FRAMES, window=[0, N_SYN - 1], boundary=None, eval_frames=SYN_FRAMES,
                     candidates=cands, chain=["upper", "lower"])

    def per_frame_syn(self, oracle):
        """PF_C: per-frame targets on trajectory C under the declared E3 edit: whole + oracle segments."""
        sc, rig = self.sc, self.sc.rig
        edit = self.decl["synthetic"]["pf_c_edit"]
        P = sc.parents["P__C"]
        ukey = f"U__C__{edit}"
        ctrl = syn_ctrl(rig, P, edit)
        self.syn[ukey] = C.patched_state(P, ctrl, {"kind": "EVAL_CONTROL", "trajectory": "C", "edit": edit})
        pins_by = {k: self.syn_pins(P, [("syn_wrist", "capture", None)], [k], SYN_ARM, k, f"PF_C/{k}/") for k in SYN_FRAMES}
        self._per_frame("syn", "PF_C", P, "P__C", ukey, ctrl, edit, pins_by, SYN_FRAMES, SYN_ARM, rig, self.syn_b, "upper", "lower",
                        oracle, self.syn_vctx(P), self.syn, dict(SYN_EDITS[edit], edit_id=edit, frames=SYN_FRAMES))

    def _per_frame(self, side, fam, P, pname, ukey, ctrl, edit, pins_by, frames, auth, rig, bindings, upper, lower, oracle, vctx,
                   store, edit_rec, F_vals=None, F_st=None):
        allpins = [p for k in frames for p in pins_by[k]]
        if F_vals is None:
            F_st, F_vals, _ = self.framewise(rig, ctrl, bindings, allpins, frames, auth)
        runs, cur = [], []
        for k in frames:
            if oracle[k] == "FEASIBLE":
                cur.append(k)
            else:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
        if len(cur) >= 2:
            runs.append(cur)
        reqs = [(f"W_{fam}", frames, "whole")] + [(f"S_{fam}_{r[0]}_{r[-1]}", r, "segment") for r in runs if r != frames]
        segF, segT = {}, {}
        for rid, fr, kind in reqs:
            pins = [p for k in fr for p in pins_by[k]]
            req = {"request_id": rid, "parent_state_id": P["state_id"], "edit": edit_rec, "constraints": pins, "resolver": {}}
            errs = S.validate_request(req, vctx)
            st, vals, info, _ = self.temporal(rig, ctrl, bindings, pins, fr, auth, upper, lower)
            cands = {}
            if kind == "whole":
                if side == "real":
                    fname = f"{fam}__candidate"
                else:
                    fname = f"{rid}__F"
                    store[fname] = C.patched_state(P, full_patch(P, ctrl, F_vals), {"kind": "F_CANDIDATE", "request": rid})
                tname = f"{rid}__T"
                store[tname] = C.patched_state(P, full_patch(P, ctrl, vals), {"kind": "T_CANDIDATE", "request": rid})
                win = [fr[0], fr[-1]]
            else:
                fname, tname = f"SEG_{fam}__F", f"SEG_{fam}__T"
                segF.update({k: F_vals[k] for k in fr})
                segT.update({k: vals[k] for k in fr})
                win = [max(frames[0], fr[0] - 1), min(frames[-1], fr[-1] + 1)]
            cands["F"] = {"state": fname, "statuses": {str(k): F_st[k] for k in fr}, "info": {}, "pin_frames": fr, "margin_frames": []}
            cands["T"] = {"state": tname, "statuses": {str(k): st[k] for k in fr}, "info": info, "pin_frames": fr, "margin_frames": []}
            self.add_req(rid, side=side, kind=kind, role="evaluation", parent=pname, U=ukey, edit=edit, auth=auth, request=req,
                         validation_errors=errs, pin_frames=fr, window=win, boundary=None,
                         eval_frames=frames if kind == "whole" else list(range(win[0], win[1] + 1)), candidates=cands,
                         chain=[upper, lower], family=fam)
        if segF:
            store[f"SEG_{fam}__F"] = C.patched_state(P, full_patch(P, ctrl, segF), {"kind": "F_CANDIDATE", "request": f"SEG_{fam}"})
            store[f"SEG_{fam}__T"] = C.patched_state(P, full_patch(P, ctrl, segT), {"kind": "T_CANDIDATE", "request": f"SEG_{fam}"})

    # ---------------- real requests
    def build_real_U(self):
        ctx = self.ctx
        self.rctrl, self.rF, self.rFst, self.rpins, self.rparams = {}, {}, {}, {}, {}
        for fam in REAL_FAMS:
            edit, pin_decl, auth, _ = C.REAL_FAMILIES[fam]
            ctrl, cand, st, pins_by, params = {}, {}, {}, {}, {}
            for k in ctx.v0["frames"]:
                pins = C.make_pins(ctx, False, ctx.v0, pin_decl, [k], auth, cid_prefix=f"{fam}/{k}/")
                req = {"request_id": f"{fam}/{k}", "parent_state_id": ctx.v0["state_id"], "edit": C.edit_record(edit, [k], {}),
                       "constraints": pins, "resolver": {"method": "analytic"}}
                errs = S.validate_request(req, ctx.vctx(ctx.v0, False))
                recs, prm, cv, pv = C.solve_request(ctx, False, ctx.v0, edit, pins, [k], solve=not errs)
                ctrl[k], cand[k], st[k], pins_by[k] = cv[k], pv[k], recs[k]["resolver_status"], pins
                if k in prm:
                    params[k] = prm[k]
            self.rctrl[fam], self.rF[fam], self.rFst[fam], self.rpins[fam], self.rparams[fam] = ctrl, cand, st, pins_by, params
            self.real[f"{fam}__control"] = C.patched_state(ctx.v0, ctrl, {"kind": "EVAL_CONTROL", "family": fam})
            self.real[f"{fam}__candidate"] = C.patched_state(ctx.v0, cand, {"kind": "EVAL_CANDIDATE", "family": fam})

    def build_real(self, oracle):
        ctx, rig, v0 = self.ctx, self.ctx.rig, self.ctx.v0
        frames = v0["frames"]
        for fam in REAL_FAMS:
            edit = C.REAL_FAMILIES[fam][0]
            er = C.edit_record(edit, frames, self.rparams[fam]) if edit else None
            self._per_frame("real", fam, v0, "v0", f"{fam}__control", self.rctrl[fam], edit, self.rpins[fam], frames, ARM, rig,
                            ctx.real_b, UPPER, LOWER, oracle[fam], ctx.vctx(v0, False), self.real, er, F_vals=self.rF[fam],
                            F_st=self.rFst[fam])
        for h, (cap, hf) in REAL_HOLDS.items():
            for e, edit in HOLD_EDITS.items():
                rid = f"H_{h}_{e}"
                fam = HOLD_EDIT_FAMILY[e]
                ctrl = self.rctrl[fam] if fam else {k: S.frame_vals(v0, i) for i, k in enumerate(frames)}
                ukey = f"{fam}__control" if fam else "v0"
                pins = C.make_pins(ctx, False, v0, [("left_wrist", "capture", None)], hf, ARM, cap_frame=cap, cid_prefix=f"{rid}/")
                er = C.edit_record(edit, frames, self.rparams[fam]) if edit else None
                req = {"request_id": rid, "parent_state_id": v0["state_id"], "edit": er, "constraints": pins, "resolver": {}}
                errs = S.validate_request(req, ctx.vctx(v0, False))
                recs, _, cv, pv = C.solve_request(ctx, False, v0, edit, pins, hf, solve=not errs)
                ctrl_same = all(S.canon(vals_to_lists(cv[k])) == S.canon(vals_to_lists(ctrl[k])) for k in hf)
                cands = {"F": None}
                fv = {k: pv[k] for k in hf}
                self.real[f"{rid}__F"] = C.patched_state(v0, full_patch(v0, ctrl, fv), {"kind": "F_CANDIDATE", "request": rid})
                cands["F"] = {"state": f"{rid}__F", "statuses": {str(k): recs[k]["resolver_status"] for k in hf}, "info": {},
                              "pin_frames": hf, "margin_frames": []}
                for v in ("T", "TE"):
                    st, vals, info, run = self.temporal(rig, ctrl, ctx.real_b, pins, hf, ARM, UPPER, LOWER, margins=(v == "TE"),
                                                        all_frames=frames)
                    self.real[f"{rid}__{v}"] = C.patched_state(v0, full_patch(v0, ctrl, {k: vals[k] for k in run}),
                                                               {"kind": f"{v}_CANDIDATE", "request": rid})
                    cands[v] = {"state": f"{rid}__{v}", "statuses": {str(k): st[k] for k in hf}, "info": info, "pin_frames": hf,
                                "margin_frames": [k for k in run if k not in hf],
                                "values": {str(k): vals_to_lists(vals[k]) for k in run}}
                self.add_req(rid, side="real", kind="hold", role="evaluation", parent="v0", U=ukey, edit=edit, auth=ARM, request=req,
                             validation_errors=errs, pin_frames=hf, window=[hf[0] - TB.MARGIN - 1, hf[-1] + TB.MARGIN + 1],
                             boundary=[hf[0], hf[-1], TB.MARGIN], eval_frames=frames, candidates=cands, chain=[UPPER, LOWER],
                             control_values_equal_family_U=ctrl_same)
        # real tamper controls on H_RH1_none T
        base = self.req["H_RH1_none"]
        Tv = {int(k): {c: np.array(v) for c, v in x.items()} for k, x in base["candidates"]["T"]["values"].items()}
        ctrl0 = {k: S.frame_vals(v0, i) for i, k in enumerate(frames)}
        stT = base["candidates"]["T"]["statuses"]
        tam = {"RN10": ({k: ctrl0[k] for k in Tv}, {k: "SOLVED" for k in stT}, lambda vv: None),
               "RN11": (Tv, stT, lambda vv: vv[113].__setitem__("spine_01", rot_local(vv[113]["spine_01"], [1, 0, 0], 1.0))),
               "RN12": (Tv, stT, lambda vv: vv[113].__setitem__("__object__", rot_local(vv[113]["__object__"], [0, 0, 1], 1.0))),
               "RN13": (Tv, stT, lambda vv: vv[113].__setitem__(LOWER, rot_local(vv[113][LOWER], [0, 0, 1], 2.0)))}
        for rid, (src, st, fn) in tam.items():
            vv = {k: dict(v) for k, v in src.items()}
            fn(vv)
            self.real[f"{rid}__T"] = C.patched_state(v0, full_patch(v0, ctrl0, vv), {"kind": "TAMPERED", "request": rid})
            r = copy.deepcopy({k: v for k, v in base.items() if k != "candidates"})
            r["role"] = "negative"
            r["candidates"] = {"T": {"state": f"{rid}__T", "statuses": st, "info": {"tamper": NEG_EXPECT[rid]["desc"]},
                                     "pin_frames": base["pin_frames"], "margin_frames": []}}
            self.req[rid] = r


def oracle_from_eval(ev_frames, pins_by, bindings, rest, scale):
    out = {}
    for k, fr in ev_frames.items():
        pins = [{"bone": bindings[c["binding_id"]]["bone"], "point": bindings[c["binding_id"]]["point"], "target": np.array(c["target"])}
                for c in pins_by[k] if c["binding_id"] in bindings]
        out[k] = S.gt_feasibility("two_bone", fr, pins, rest, scale)[0] if pins else "NOT_APPLICABLE"
    return out


def cmd_build(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    if S.sha_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash")
    g = guard()
    t0 = time.time()
    B = Build(decl, out)
    B.build_real_U()
    # unpinned controls only -> Blender -> feasibility oracle for per-frame segments
    ustates = {"v0": B.ctx.v0} | {f"{f}__control": B.real[f"{f}__control"] for f in REAL_FAMS}
    edit = decl["synthetic"]["pf_c_edit"]
    PC = B.sc.parents["P__C"]
    uC = C.patched_state(PC, syn_ctrl(B.sc.rig, PC, edit), {"kind": "EVAL_CONTROL", "trajectory": "C", "edit": edit})
    C.states_file(out / "u_real_states.json", ustates, "w105")
    C.states_file(out / "u_syn_states.json", {f"U__C__{edit}": uC}, "synthetic")
    blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest",
            C.INPUTS["w105_rig_rest"][0], "--states", out / "u_real_states.json", "--out", out / "u_real_eval.json", log=out / "u_real_eval.log")
    blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", f"synthetic:{Path(decl['preflight_dir']) / 'syn_spec.json'}",
            "--states", out / "u_syn_states.json", "--out", out / "u_syn_eval.json", log=out / "u_syn_eval.log")
    ure, use = load(out / "u_real_eval.json"), load(out / "u_syn_eval.json")
    oracle = {}
    for f in REAL_FAMS:
        fr = C.ev_frames(ure, f"{f}__control")
        oracle[f] = oracle_from_eval(fr, B.rpins[f], B.ctx.real_b, ure["rest"], ure["rig"]["scale"][0])
    fr = C.ev_frames(use, f"U__C__{edit}")
    pins_by = {k: B.syn_pins(PC, [("syn_wrist", "capture", None)], [k], SYN_ARM, k, f"PF_C/{k}/") for k in SYN_FRAMES}
    soracle = oracle_from_eval(fr, pins_by, B.syn_b, use["rest"], 1.0)
    B.build_syn()
    B.per_frame_syn(soracle)
    B.build_real(oracle)
    C.states_file(out / "real_states.json", B.real, "w105")
    C.states_file(out / "syn_states.json", B.syn, "synthetic")
    rep = {"schema": "w117_build_v1", "tag": a.tag, "declaration_sha256": a.declaration_sha, "wall_s": time.time() - t0,
           "real_states": {n: s["state_id"] for n, s in B.real.items()}, "syn_states": {n: s["state_id"] for n, s in B.syn.items()},
           "oracle_real": {f: {str(k): v for k, v in o.items()} for f, o in oracle.items()},
           "oracle_syn_PF_C": {str(k): v for k, v in soracle.items()}, "requests": B.req, "source_guard": g.check()}
    dump(out / "build_report.json", rep)
    print(json.dumps({"tag": a.tag, "wall_s": rep["wall_s"], "n_real": len(B.real), "n_syn": len(B.syn), "n_requests": len(B.req),
                      "source_unchanged": rep["source_guard"]["unchanged"]}))
    return 0


# --------------------------------------------------------------------------------------------- evaluate ---
def cmd_evaluate(a):
    out = new_dir(a.out_dir)
    b = Path(a.build_dir)
    blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest",
            C.INPUTS["w105_rig_rest"][0], "--states", b / "real_states.json", "--out", out / "real_eval.json", log=out / "real_eval.log")
    blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", f"synthetic:{Path(a.preflight) / 'syn_spec.json'}",
            "--states", b / "syn_states.json", "--out", out / "syn_eval.json", log=out / "syn_eval.log")
    print(json.dumps({"real": S.sha_file(out / "real_eval.json"), "syn": S.sha_file(out / "syn_eval.json")}))
    return 0


# --------------------------------------------------------------------------------------------- assess ---
def pins_fn(bindings, req):
    def f(k):
        return [{"bone": bindings[c["binding_id"]]["bone"], "point": bindings[c["binding_id"]]["point"], "target": np.array(c["target"]),
                 "semantic_joint": c["semantic_joint"]} for c in req["constraints"] if c["binding_id"] in bindings and k in c["frames"]]
    return f


def assess_request(rid, R, ev, rig, bindings, rest, scale):
    U = C.ev_frames(ev, R["U"])
    P = C.ev_frames(ev, R["parent"])
    auth = R["auth"]
    edit_ch = list(R["request"]["edit"]["channels"]) if R["request"].get("edit") else []
    dom = S.domain_points(rig, sorted(set(auth) | set(edit_ch))) if (auth or edit_ch) else set()
    arm_dom = S.domain_points(rig, auth) if auth else set()
    pf = pins_fn(bindings, R["request"])
    errs = [tuple(e) for e in R["validation_errors"]]
    out = {"role": R["role"], "kind": R["kind"], "side": R["side"], "validation_errors": errs, "pin_frames": [R["pin_frames"][0],
           R["pin_frames"][-1]], "n_pin_frames": len(R["pin_frames"]), "candidates": {}}
    # oracle per pin frame (Blender-evaluated U, independent of any solver)
    orc = {}
    for k in R["pin_frames"]:
        pins = pf(k)
        if errs or not pins or len(pins) != 1:
            orc[k] = ("NOT_ASSESSED", None)
            continue
        mode, _ = S._structure(rig, pins, auth)
        orc[k] = S.gt_feasibility(mode, U[k], pins, rest, scale) if mode else ("UNDETERMINED_UNSUPPORTED_STRUCTURE", None)
    out["oracle"] = {str(k): v[0] for k, v in orc.items()}
    out["oracle_margin_m"] = {str(k): v[1] for k, v in orc.items()}
    out["oracle_all_feasible"] = all(v[0] == "FEASIBLE" for v in orc.values())
    out["control_residual_m"] = {str(k): float(np.linalg.norm(np.array(U[k]["world"][p["bone"]][p["point"]]) - p["target"]))
                                 for k in R["pin_frames"] for p in pf(k)[:1]}
    for v, cd in R["candidates"].items():
        X = C.ev_frames(ev, cd["state"])
        statuses = [s for s in cd["statuses"].values() if s != "NOT_APPLICABLE"]
        frames_chk = R["eval_frames"]
        per, extra = TB.evaluate_temporal_request(X, U, P, set(R["pin_frames"]) if not errs else set(), set(cd["margin_frames"]),
                                                  frames_chk, pf, auth, edit_ch, dom, arm_dom, rest, scale, tuple(R["chain"]))
        dec = TB.temporal_decision(errs, statuses, per, extra) if errs or statuses else \
            {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": ["NOT_RESOLVED"]}
        res = [p["candidate_residual_m"] for k in R["pin_frames"] if k in per and per[k].get("evaluation")
               for p in per[k]["evaluation"]["pins"]]
        frame_pass = {str(k): bool(per[k]["evaluation"]["pins_satisfied"] and per[k]["evaluation"]["scope_ok"]
                                   and per[k]["evaluation"]["locality_ok"] and per[k]["evaluation"]["lengths_ok"]
                                   and per[k]["evaluation"]["root_unchanged"] and per[k]["evaluation"]["joint_limit_ok"])
                      for k in R["pin_frames"] if k in per and per[k].get("evaluation")}
        c = {"state": cd["state"], "decision": dec, "statuses": cd["statuses"], "frame_pass": frame_pass,
             "temporal_domain_violations": extra, "n_margin_frames": len(cd["margin_frames"]),
             "residual_m": {"max": max(res) if res else None, "p95": float(np.percentile(res, 95)) if res else None,
                            "n_over_tol": sum(1 for r in res if r > S.TOL_M), "n": len(res)},
             "frames_changed_vs_U": sum(1 for k in R["eval_frames"] if X[k]["stored_wxyz"] != U[k]["stored_wxyz"]),
             "solver": {k: cd["info"].get(k) for k in ("J_framewise", "J_temporal", "fidelity_framewise", "fidelity_temporal",
                                                         "converged", "best_start", "starts", "wall_s", "within_runtime_budget",
                                                         "anchors")} if isinstance(cd["info"], dict) and "J_temporal" in cd["info"]
             else {k: cd["info"].get(k) for k in ("mode", "reason", "tamper", "not_resolved") if k in cd["info"]}}
        ev_pins = [per[k]["evaluation"] for k in R["pin_frames"] if k in per and per[k].get("evaluation")]
        c["accepted_scope"] = {"stored_changed_vs_control": sorted({x for e in ev_pins for x in e["stored_changed_vs_control"]}),
                               "max_bone_length_error_m": max([e["max_bone_length_error_m"] for e in ev_pins] + [0.0]),
                               "root_unchanged": all(e["root_unchanged"] for e in ev_pins),
                               "bend_flips_vs_control": sum(1 for e in ev_pins if e.get("bend_flip"))}
        w = list(range(R["window"][0], R["window"][1] + 1))
        if R["role"] in ("evaluation", "noop", "development") and not errs:
            c["metrics"] = TB.temporal_metrics(X, U, w, R["chain"][0], R["chain"][1], auth,
                                               boundary=tuple(R["boundary"]) if R.get("boundary") else None)
            c["collateral"] = collateral(X, U, P, w, R["side"])
        out["candidates"][v] = c
    if R["role"] in ("evaluation", "noop", "development") and not errs:
        w = list(range(R["window"][0], R["window"][1] + 1))
        out["U_metrics_paths"] = {"wrist": {str(k): U[k]["world"][R["chain"][1]]["tail"] for k in w},
                                  "elbow": {str(k): U[k]["world"][R["chain"][0]]["tail"] for k in w},
                                  "parent_wrist": {str(k): P[k]["world"][R["chain"][1]]["tail"] for k in w}}
        L1, L2 = rest[R["chain"][0]]["length"] * scale, rest[R["chain"][1]]["length"] * scale
        rmin, rmax = S._limits_reach(L1, L2)
        sl = {}
        for k in R["pin_frames"]:
            t = pf(k)
            if t:
                sl[str(k)] = rmax - float(np.linalg.norm(t[0]["target"] - np.array(U[k]["world"][R["chain"][0]]["head"])))
        out["reach_slack_m"] = sl
        out["near_singular_frames"] = sorted(int(k) for k, s in sl.items() if s < TB.NEAR_SINGULAR_SLACK_M)
        cmp = {}
        for v in ("T", "TE"):
            if v in out["candidates"] and "F" in out["candidates"] and "metrics" in out["candidates"][v]:
                cmp[v] = TB.compare(out["candidates"]["F"]["metrics"], out["candidates"][v]["metrics"])
                cmp[v]["both_accepted"] = (out["candidates"]["F"]["decision"]["decision"] == "ACCEPT"
                                           and out["candidates"][v]["decision"]["decision"] == "ACCEPT"
                                           and "TRIVIALLY_SATISFIED_NOT_EVIDENCE" not in out["candidates"][v]["decision"]["reasons"])
        out["comparison"] = cmp
    return out


def collateral(X, U, P, w, side):
    pts = {"real": {"right_wrist": ("lowerarm_r", "tail"), "neck": ("neck_02", "tail"), "head": ("head", "tail"),
                    "pelvis_root": ("spine_01", "head")},
           "syn": {"neck": ("neck", "tail"), "base_root": ("base", "head")}}[side]
    out = {}
    for n, (b, p) in pts.items():
        if b not in X[w[0]]["world"]:
            continue
        out[n] = {"max_vs_U_m": max(float(np.linalg.norm(np.array(X[k]["world"][b][p]) - np.array(U[k]["world"][b][p]))) for k in w),
                  "max_vs_parent_m": max(float(np.linalg.norm(np.array(X[k]["world"][b][p]) - np.array(P[k]["world"][b][p]))) for k in w)}
    out["root_matrix_equal_U"] = all(X[k]["matrix_world"] == U[k]["matrix_world"] for k in w)
    return out


def cmd_assess(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    if S.sha_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash")
    g = guard()
    Bd = Path(a.build_dir)
    B = load(Bd / "build_report.json")
    rev, sev = load(Path(a.eval_dir) / "real_eval.json"), load(Path(a.eval_dir) / "syn_eval.json")
    ctx = C.Ctx(load(W116 / "results/declare/declaration.json"))
    sc = SynCtx(decl["preflight_dir"])
    cal = load(Path(decl["calibration_dir"]) / "calibration.json")
    syn_b = cal["syn_bindings"]["bound"]
    w116_syn = load(W116 / "results/calibrate/calibration.json")["bindings"]["synthetic"]["bound"]
    rep = {"schema": "w117_assess_v1", "declaration_sha256": a.declaration_sha, "requests": {}}
    for rid, R in B["requests"].items():
        if R["side"] == "real":
            rep["requests"][rid] = assess_request(rid, R, rev, ctx.rig, ctx.real_b, rev["rest"], rev["rig"]["scale"][0])
        else:
            rep["requests"][rid] = assess_request(rid, R, sev, sc.rig, dict(syn_b) | w116_syn, sev["rest"], 1.0)
    # ---- W116 Blender parity of the F/U whole-clip states
    w116e = load(W116 / "results/eval_e1/real_eval.json")
    B116 = load(W116 / "results/build_r1/build_report.json")
    par = {}
    for fam in REAL_FAMS:
        for suf in ("control", "candidate"):
            n = f"{fam}__{suf}"
            a_, b_ = rev["states"][n], w116e["states"][n]
            par[n] = {"state_id_equal": B["real_states"][n] == B116["real_states"][n],
                      "value_digest_equal": a_["value_digest"] == b_["value_digest"], "frames_bit_equal": a_["frames"] == b_["frames"]}
    par["v0"] = {"frames_bit_equal": rev["states"]["v0"]["frames"] == w116e["states"]["v0"]["frames"]}
    rep["w116_blender_parity"] = par
    # ---- build-time U evaluation equals evaluate-stage U evaluation
    ure = load(Bd / "u_real_eval.json")
    use = load(Bd / "u_syn_eval.json")
    rep["u_eval_consistency"] = all(ure["states"][n]["frames"] == rev["states"][n]["frames"] for n in ure["states"]) and \
        all(use["states"][n]["frames"] == sev["states"][n]["frames"] for n in use["states"])
    # ---- lineage programmes
    rep["lineage"] = lineage(out, decl, B, rep, ctx, sc, Bd, rev, sev)
    rep["source_guard"] = g.check()
    rep["w116_manifest_after"] = w116_manifest_check()
    pre = Path(decl["preflight_dir"])
    rep["parent_files_verify"] = {"real": S.verify_state(load(pre / "real_parent_state.json")),
                                  **{n: S.verify_state(load(pre / f"syn_parent_{n}.json")) for n in sc.parents}}
    dump(out / "assess_report.json", rep)
    summ = {rid: {v: [c["decision"]["decision"], c["decision"]["failure_class"], c["decision"]["reasons"][:3]]
                  for v, c in r["candidates"].items()} for rid, r in rep["requests"].items()}
    print(json.dumps({"decisions": summ, "w116_parity": all(all(x.values()) for x in par.values()),
                      "u_eval_consistency": rep["u_eval_consistency"], "source": rep["source_guard"]["unchanged"],
                      "w116_manifest_after": rep["w116_manifest_after"]["ok"]}, indent=0, default=str))
    return 0


SYN_LINEAGE_ORDER = ["H_A", "H_B", "H_C"]
REAL_LINEAGE_ORDER = ["H_RH1_none", "H_RH1_RBneg", "H_RH1_RBpos", "H_RH2_RBneg", "H_RH2_RBpos", "H_RH2_none", "W_RB_neg", "W_RA_neg",
                      "W_RC_raise"]


def lineage(out, decl, B, rep, ctx, sc, Bd, rev, sev):
    states = {"real": load(Bd / "real_states.json")["states"], "syn": load(Bd / "syn_states.json")["states"]}
    full = {}

    def st_full(side, name, parent):
        s = states[side][name]
        op = {"kind": "PIN_EDIT", "state": name}
        return S.make_state(parent["state_id"], parent["rig_id"], parent["frames"], parent["channels"], s["values"], op, parent.get("source"))

    store = S.Store(out / "store")
    res = {"synthetic": [], "real": []}
    dec = lambda rid, v: rep["requests"][rid]["candidates"][v]["decision"]  # noqa: E731
    # synthetic
    pick = next((r for r in SYN_LINEAGE_ORDER if dec(r, "T")["decision"] == "ACCEPT"), None)
    rev_eval = {}
    if pick:
        tr = B["requests"][pick]["parent"]
        P = sc.parents[tr]
        store.put(P)
        store.init_ref("syn/main", P["state_id"])
        store.put(sc.parents["P__A"])
        store.init_ref("syn/atomic", sc.parents["P__A"]["state_id"])
        c1 = st_full("syn", B["requests"][pick]["candidates"]["T"]["state"], P)
        res["synthetic"].append(dict(step="S1", request=pick, **store.commit("syn/main", P["state_id"], c1, dec(pick, "T"))))
        c2 = st_full("syn", B["requests"]["N13"]["candidates"]["T"]["state"], sc.parents["P__A"])
        h = store.refs["syn/atomic"]
        r2 = store.commit("syn/atomic", h, c2, dec("N13", "T"))
        res["synthetic"].append(dict(step="S2", head_unchanged=r2["head_after"] == h, **r2))
        c3 = st_full("syn", B["requests"][pick]["candidates"]["TE"]["state"], P)
        res["synthetic"].append(dict(step="S3", **store.commit("syn/main", P["state_id"], c3, dec(pick, "TE"))))
        if store.refs["syn/main"] != P["state_id"]:
            srev, srec = store.revert("syn/main")
            res["synthetic"].append(dict(step="S4", **srec))
            C.states_file(out / "syn_revert_states.json", {"syn_revert": srev}, "synthetic")
            blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", f"synthetic:{Path(decl['preflight_dir']) / 'syn_spec.json'}",
                    "--states", out / "syn_revert_states.json", "--out", out / "syn_revert_eval.json", log=out / "syn_revert_eval.log")
            e = load(out / "syn_revert_eval.json")["states"]["syn_revert"]
            rev_eval["synthetic"] = {"restores": tr, "value_digest_equal_parent": e["value_digest"] == sev["states"][tr]["value_digest"],
                                     "frames_bit_equal_parent": e["frames"] == sev["states"][tr]["frames"]}
    else:
        res["synthetic"].append({"step": "S1", "executed": False, "reason": "no ACCEPTed synthetic T request in the declared order"})
    # real
    v0 = ctx.v0
    store.put(v0)
    store.init_ref("real/main", v0["state_id"])
    store.init_ref("real/atomic", v0["state_id"])
    pick = next((r for r in REAL_LINEAGE_ORDER if r in rep["requests"] and dec(r, "T")["decision"] == "ACCEPT"), None)
    if pick:
        c1 = st_full("real", B["requests"][pick]["candidates"]["T"]["state"], v0)
        res["real"].append(dict(step="R1", request=pick, **store.commit("real/main", v0["state_id"], c1, dec(pick, "T"))))
    else:
        res["real"].append({"step": "R1", "executed": False})
    h = store.refs["real/atomic"]
    r2 = store.commit("real/atomic", h, st_full("real", "RN13__T", v0), dec("RN13", "T"))
    res["real"].append(dict(step="R2", head_unchanged=r2["head_after"] == h, **r2))
    other = next(r for r in REAL_LINEAGE_ORDER if r != pick and r in rep["requests"])
    r3 = store.commit("real/main", v0["state_id"], st_full("real", B["requests"][other]["candidates"]["F"]["state"], v0),
                      dec(other, "F"))
    res["real"].append(dict(step="R3", request=other, **r3))
    h = store.refs["real/atomic"]
    r4 = store.commit("real/atomic", h, st_full("real", B["requests"]["H_RH2_none"]["candidates"]["T"]["state"], v0),
                      dec("H_RH2_none", "T"))
    res["real"].append(dict(step="R4", head_unchanged=r4["head_after"] == h, **r4))
    if store.refs["real/main"] != v0["state_id"]:
        rv, rrec = store.revert("real/main")
        res["real"].append(dict(step="R5", **rrec))
        C.states_file(out / "real_revert_states.json", {"real_revert": rv}, "w105")
        blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest",
                C.INPUTS["w105_rig_rest"][0], "--states", out / "real_revert_states.json", "--out", out / "real_revert_eval.json",
                log=out / "real_revert_eval.log")
        e = load(out / "real_revert_eval.json")["states"]["real_revert"]
        rev_eval["real"] = {"value_digest_equal_v0": e["value_digest"] == rev["states"]["v0"]["value_digest"],
                            "frames_bit_equal_v0": e["frames"] == rev["states"]["v0"]["frames"]}
    return {"programmes": res, "store_log": store.log, "refs": store.refs,
            "lineage_audit": {r: store.lineage(hh) for r, hh in store.refs.items()}, "revert_blender": rev_eval}


# --------------------------------------------------------------------------------------------- render ---
COLORS = {"target": (0.1, 0.9, 0.2, 1), "wrist": (0.1, 0.4, 1.0, 1), "elbow": (1.0, 0.55, 0.0, 1)}


def cmd_render(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    B = load(Path(a.build_dir) / "build_report.json")
    A = load(Path(a.assess_dir) / "assess_report.json")
    ev = {"real": load(Path(a.eval_dir) / "real_eval.json"), "syn": load(Path(a.eval_dir) / "syn_eval.json")}
    sel = []  # (rid, frames)
    for rid in ("H_A", "H_B", "H_C"):
        w = A["requests"][rid]["candidates"]["F"]["metrics"]["window"]
        sel.append((rid, list(range(w[0], w[1] + 1))))
    pfc = [r for r in A["requests"] if r.startswith(("W_PF_C", "S_PF_C")) and "metrics" in A["requests"][r]["candidates"].get("F", {})]
    if pfc:
        r = max(pfc, key=lambda r: A["requests"][r]["candidates"]["F"]["metrics"]["defects"]["D2"])
        sel.append((r, peak_window(A["requests"][r], 4)))
    for rid in ("H_RH1_none", "H_RH1_RBneg", "H_RH2_RBneg"):
        w = A["requests"][rid]["candidates"]["F"]["metrics"]["window"]
        sel.append((rid, list(range(w[0], w[1] + 1))))
    sel.append(("W_RB_neg", peak_window(A["requests"]["W_RB_neg"], 4)))
    segs = [r for r in A["requests"] if r.startswith("S_RB_pos_") and "metrics" in A["requests"][r]["candidates"].get("F", {})]
    if segs:
        r = max(segs, key=lambda r: A["requests"][r]["candidates"]["F"]["metrics"]["defects"]["D2"])
        w = A["requests"][r]["candidates"]["F"]["metrics"]["window"]
        sel.append((r, list(range(w[0], w[1] + 1))))
    jobs = {"real": [], "syn": []}
    for rid, frames in sel:
        R = B["requests"][rid]
        side = R["side"]
        e = ev[side]
        tgt = [c["target"] for c in R["request"]["constraints"] if R["pin_frames"][0] in c["frames"]][0]
        variants = [("U", R["U"])] + [(v, R["candidates"][v]["state"]) for v in ("F", "T", "TE") if v in R["candidates"]]
        for v, st in variants:
            E = C.ev_frames(e, st)
            for k in frames:
                ch = R["chain"]
                mk = [{"name": "wrist", "position": E[k]["world"][ch[1]]["tail"], "color": COLORS["wrist"], "radius": 0.009},
                      {"name": "elbow", "position": E[k]["world"][ch[0]]["tail"], "color": COLORS["elbow"], "radius": 0.009}]
                tk = [c["target"] for c in R["request"]["constraints"] if k in c["frames"]]
                if tk:
                    mk.append({"name": "target", "position": tk[0], "color": COLORS["target"], "radius": 0.013})
                jobs[side].append({"id": f"{rid}__{v}__f{k:03d}", "state": st, "frame_index": k, "markers": mk, "centre": tgt,
                                   "view_offset": [0.0, -3.0, 0.0]})
    dump(out / "real_jobs.json", {"jobs": jobs["real"]})
    dump(out / "syn_jobs.json", {"jobs": jobs["syn"]})
    b = Path(a.build_dir)
    blender("scripts/w116_blender_stages.py", "--", "render", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest",
            C.INPUTS["w105_rig_rest"][0], "--states", b / "real_states.json", "--jobs", out / "real_jobs.json", "--out-dir", out / "real",
            "--ortho-scale", "1.1", "--size", "360", "--hide-mesh", log=out / "real_render.log")
    blender("scripts/w116_blender_stages.py", "--", "render", "--rig", f"synthetic:{Path(decl['preflight_dir']) / 'syn_spec.json'}",
            "--states", b / "syn_states.json", "--jobs", out / "syn_jobs.json", "--out-dir", out / "syn", "--ortho-scale", "1.2",
            "--size", "360", "--hide-mesh", log=out / "syn_render.log")
    compose(out, sel, B)
    dump(out / "render_selection.json", {"selection": sel})
    return 0


def peak_window(Ar, half):
    m = Ar["candidates"]["F"]["metrics"]
    pt = m["per_transition"]
    k = int(max(pt, key=lambda b: pt[b]["elbow_comp_step_m"]))
    lo, hi = m["window"]
    return list(range(max(lo, k - half), min(hi, k + half) + 1))


def compose(out, sel, B):
    from PIL import Image, ImageDraw
    for rid, frames in sel:
        R = B["requests"][rid]
        d = out / R["side"]
        variants = ["U", "F"] + [v for v in ("T", "TE") if v in R["candidates"]]
        ims = {(v, k): Image.open(d / f"{rid}__{v}__f{k:03d}.png").convert("RGB") for v in variants for k in frames}
        w, h = ims[(variants[0], frames[0])].size
        lo, hi = R["pin_frames"][0], R["pin_frames"][-1]
        # contact sheet: up to 12 frames, evenly spread
        cols = frames if len(frames) <= 12 else [frames[int(round(i * (len(frames) - 1) / 11))] for i in range(12)]
        sw = w // 2
        sheet = Image.new("RGB", (60 + sw * len(cols), 30 + (h // 2) * len(variants)), (15, 15, 15))
        dr = ImageDraw.Draw(sheet)
        dr.text((4, 4), f"{rid}  pin {lo}-{hi}  green=target blue=wrist orange=elbow  rows: {' / '.join(variants)}", fill=(255, 255, 255))
        for j, k in enumerate(cols):
            dr.text((60 + j * sw + 4, 18), f"f{k}{'*' if lo <= k <= hi else ''}", fill=(255, 220, 120) if lo <= k <= hi else (200, 200, 200))
            for i, v in enumerate(variants):
                sheet.paste(ims[(v, k)].resize((sw, h // 2)), (60 + j * sw, 30 + i * (h // 2)))
        for i, v in enumerate(variants):
            dr.text((6, 30 + i * (h // 2) + h // 4), v, fill=(255, 255, 255))
        sheet.save(out / f"sheet_{rid}.png")
        gif = []
        for k in frames:
            fr = Image.new("RGB", (w * len(variants), h + 22), (15, 15, 15))
            dd = ImageDraw.Draw(fr)
            for i, v in enumerate(variants):
                fr.paste(ims[(v, k)], (i * w, 22))
                dd.text((i * w + 6, 4), f"{v}  f{k}{' PIN' if lo <= k <= hi else ''}", fill=(255, 255, 255))
            gif.append(fr)
        gif[0].save(out / f"anim_{rid}.gif", save_all=True, append_images=gif[1:], duration=120, loop=0)


# --------------------------------------------------------------------------------------------- decide ---
def cmd_decide(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    if S.sha_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash")
    pf = load(Path(decl["preflight_dir"]) / "preflight.json")
    cal = load(Path(decl["calibration_dir"]) / "calibration.json")
    A = load(Path(a.assess_dir) / "assess_report.json")
    b1, b2 = Path(a.build_dir), Path(a.build_dir2)
    e1, e2 = Path(a.eval_dir), Path(a.eval_dir2)
    VR = load(a.visual_review)
    Rq = A["requests"]
    g, det = {}, {}
    g["V1_authentication"] = not pf["stop"]
    g["V2_parent_reproduction"] = pf["parent_rebuild"]["canonical_equal"] and pf["parent_blender"]["equals_w105_fixed_digest"] \
        and pf["parent_blender"]["frames_bit_equal_w116_probe"]
    par = A["w116_blender_parity"]
    g["V3_w116_control_parity"] = cal["w116_control_parity_ok"] and all(all(v.values()) for v in par.values())
    sb = {b["semantic_joint"] for b in cal["syn_bindings"]["bound"].values()}
    sf = {j: r["status"] for j, r in cal["syn_bindings"]["failed"].items()}
    v4 = {"fk": cal["syn_fk_max_error_m"] <= 1e-5, "bound": {"syn_wrist", "syn_elbow", "syn_shoulder"} <= sb,
          "failsafe": sf.get("syn_hand") == "BINDING_AMBIGUOUS" and sf.get("syn_nose") == "BINDING_MISSING"}
    for n in ("N10", "RN10"):
        d = Rq[n]["candidates"]["T"]["decision"]
        v4[n] = d["decision"] == "REJECT" and "EVALUATED_PIN_UNSATISFIED" in d["reasons"]
    det["V4"] = v4
    g["V4_independent_evaluator"] = all(v4.values())
    # V5
    syn_ev = load(e1 / "syn_eval.json")
    Bb = load(b1 / "build_report.json")
    v5 = {}
    for h in ("H_A", "H_B", "H_C"):
        R = Bb["requests"][h]
        U, P = C.ev_frames(syn_ev, R["U"]), C.ev_frames(syn_ev, R["parent"])
        lo, hi = R["pin_frames"][0], R["pin_frames"][-1]
        T = np.array(R["request"]["constraints"][0]["target"])
        W = lambda X, k: np.array(X[k]["world"]["lower"]["tail"])  # noqa: E731
        ent, rel = float(np.linalg.norm(T - W(U, lo))), float(np.linalg.norm(T - W(U, hi)))
        mv = min(float(np.linalg.norm(W(U, k) - W(P, k))) for k in R["pin_frames"])
        poses = float(np.linalg.norm(W(U, lo - 1) - W(U, hi + 1)))
        v5[h] = {"entry_offset_m": ent, "release_offset_m": rel, "edit_move_min_m": mv, "pre_post_pose_diff_m": poses,
                 "ok": ent >= 0.005 and rel >= 0.005 and abs(ent - rel) >= 0.005 and mv >= 0.01 and poses >= 0.005}
    for fam in ("RB_pos", "RB_neg"):
        cr = [v for k, v in Rq[f"W_{fam}"]["control_residual_m"].items()]
        v5[fam] = {"min_control_residual_m": min(cr), "ok": min(cr) > S.TOL_M}
    det["V5"] = v5
    g["V5_meaningful_asymmetric_design"] = all(v["ok"] for v in v5.values())
    tm = [pf["timing"]] + [pf["syn_timing"]]
    v6 = all(st["interpolations"] == ["CONSTANT"] and st["blender_frame_equals_index_minus_f0_plus_1"]
             and all(s["half_equals_frame"] and s["late_equals_frame"] for s in st["subframe"].values())
             for t in tm for st in t["states"].values())
    g["V6_timing_semantics"] = v6
    bad = []
    for rid, r in Rq.items():
        for v, c in r["candidates"].items():
            for k, cls in r["oracle"].items():
                if cls == "INFEASIBLE" and c["frame_pass"].get(k):
                    bad.append(f"{rid}/{v}/{k}")
    syn_cls = {h: set(Rq[h]["oracle"].values()) for h in ("H_A", "H_B", "H_C", "H_Cx")}
    v7_syn = syn_cls["H_A"] == {"FEASIBLE"} and syn_cls["H_B"] == {"FEASIBLE"} and syn_cls["H_C"] == {"FEASIBLE"} and \
        "INFEASIBLE" in syn_cls["H_Cx"]
    det["V7"] = {"infeasible_but_passing": bad, "synthetic_classes": {h: sorted(s) for h, s in syn_cls.items()}}
    g["V7_oracle_soundness"] = not bad and v7_syn
    same_states = all((b1 / f).read_bytes() == (b2 / f).read_bytes() for f in ("real_states.json", "syn_states.json"))
    B2 = load(b2 / "build_report.json")
    same_dec = all(Bb["requests"][r]["candidates"][v]["statuses"] == B2["requests"][r]["candidates"][v]["statuses"]
                   for r in Bb["requests"] for v in Bb["requests"][r]["candidates"])
    same_ev = all(load(e1 / f)["states"] == load(e2 / f)["states"] for f in ("real_eval.json", "syn_eval.json"))
    det["V8"] = {"states_identical": same_states, "statuses_identical": same_dec, "evaluations_identical": same_ev,
                 "u_eval_consistency": A["u_eval_consistency"]}
    g["V8_repeatability"] = same_states and same_dec and same_ev and A["u_eval_consistency"]
    missing = [f"{rid}/{v}" for rid, r in Rq.items() if not r["validation_errors"] for v, c in r["candidates"].items()
               if c["residual_m"]["n"] < r["n_pin_frames"]]
    det["V9"] = missing
    g["V9_population_complete"] = not missing
    # ---- C
    bad_acc = []
    for rid, r in Rq.items():
        for v, c in r["candidates"].items():
            if c["decision"]["decision"] == "ACCEPT":
                ok = c["residual_m"]["n_over_tol"] == 0 and not c["temporal_domain_violations"] and \
                    set(c["accepted_scope"]["stored_changed_vs_control"]) <= set(Bb["requests"][rid]["auth"]) and \
                    c["accepted_scope"]["root_unchanged"] and c["accepted_scope"]["max_bone_length_error_m"] <= S.LENGTH_TOL_M and \
                    all(c["frame_pass"].values())
                if not ok:
                    bad_acc.append(f"{rid}/{v}")
    det["C1"] = bad_acc
    g["C1_accept_soundness"] = not bad_acc
    wrong = []
    for rid, exp in NEG_EXPECT.items():
        for v, want in exp.items():
            if v == "desc":
                continue
            d = Rq[rid]["candidates"][v]["decision"]
            if [d["decision"], d["failure_class"]] != want:
                wrong.append({"req": f"{rid}/{v}", "got": [d["decision"], d["failure_class"], d["reasons"]], "expected": want})
    det["C2"] = wrong
    g["C2_negative_controls"] = not wrong
    atom = []
    for rid, r in Rq.items():
        for v, c in r["candidates"].items():
            if r["validation_errors"] or not c["frame_pass"]:
                continue
            valid = set(c["statuses"].values()) <= {"SOLVED", "NO_CHANGE_NEEDED"}
            allpass = all(c["frame_pass"].values()) and valid and not c["temporal_domain_violations"]
            if (c["decision"]["decision"] == "ACCEPT") != allpass:
                atom.append(f"{rid}/{v}")
    for n in ("N13", "RN13"):
        c = Rq[n]["candidates"]["T"]
        fp = c["frame_pass"]
        if not (c["decision"]["decision"] == "REJECT" and sum(1 for x in fp.values() if not x) == 1):
            atom.append(f"{n}: single-corrupted-frame property")
    det["C3"] = atom
    g["C3_atomicity"] = not atom
    lin = A["lineage"]
    refs_ok = all(x.get("head_after") == x.get("head_before") for x in lin["store_log"] if x.get("op") == "commit" and x.get("decision") == "REJECT")
    steps = {x["step"]: x for side in lin["programmes"].values() for x in side}
    stale = steps.get("S3", {}).get("reasons") == ["STALE_PARENT"] and steps.get("R3", {}).get("reasons") == ["STALE_PARENT"]
    acc = steps.get("S1", {}).get("decision") == "ACCEPT" and steps.get("R1", {}).get("decision") == "ACCEPT"
    rv = lin["revert_blender"]
    revert_ok = bool(rv.get("real") and rv.get("synthetic")) and all(all(v for k, v in x.items() if k != "restores") for x in rv.values()) \
        and steps.get("S4", {}).get("values_equal_restored") is True and steps.get("R5", {}).get("values_equal_restored") is True
    audit_ok = all(all(x["verifies"] for x in ch) for ch in lin["lineage_audit"].values())
    det["C4"] = {"refs_unchanged_on_reject": refs_ok, "stale": stale, "accept_executed": acc, "revert_ok": revert_ok, "audit_ok": audit_ok,
                 "atomic_heads_unchanged": [steps.get(s, {}).get("head_unchanged") for s in ("S2", "R2", "R4")]}
    g["C4_lineage_reversal"] = refs_ok and stale and acc and revert_ok and audit_ok and all(det["C4"]["atomic_heads_unchanged"])
    src_ok = A["source_guard"]["unchanged"] and A["w116_manifest_after"]["ok"] and all(A["parent_files_verify"].values())
    for bd in (b1, b2):
        src_ok = src_ok and load(bd / "build_report.json")["source_guard"]["unchanged"]
    g["C5_source_immutability"] = src_ok
    newfail = []
    for rid, r in Rq.items():
        if r["role"] != "evaluation" or r["validation_errors"] or not r["oracle_all_feasible"]:
            continue
        for v, c in r["candidates"].items():
            if c["decision"]["decision"] != "ACCEPT":
                newfail.append({"req": f"{rid}/{v}", "got": c["decision"]})
    det["C6"] = newfail
    g["C6_no_new_hard_failure"] = not newfail
    # ---- Q
    q = {}
    for v in ("T", "TE"):
        for side in ("syn", "real"):
            imp, worse = [], []
            for rid, r in Rq.items():
                if r["role"] != "evaluation" or r["side"] != side or v not in r.get("comparison", {}):
                    continue
                cm = r["comparison"][v]
                if not cm["both_accepted"]:
                    continue
                if cm["material_improvement"]:
                    imp.append(rid)
                if cm["worse"]:
                    worse.append({"req": rid, "worse": cm["worse"], "improved": cm["improved"]})
            q[f"{side}_{v}"] = {"improved": imp, "worse": worse, "pass": bool(imp) and not worse}
    det["Q"] = q
    qv = {}
    for v in ("T", "TE"):
        ok_s = [r for r in q[f"syn_{v}"]["improved"] if VR.get("requests", {}).get(r, {}).get(v, {}).get("improvement_visible") is True
                and VR["requests"][r][v].get("arm_plausible") is not False]
        ok_r = [r for r in q[f"real_{v}"]["improved"] if VR.get("requests", {}).get(r, {}).get(v, {}).get("improvement_visible") is True
                and VR["requests"][r][v].get("arm_plausible") is not False]
        qv[v] = {"syn": ok_s, "real": ok_r, "pass": bool(ok_s) and bool(ok_r)}
    det["QV"] = qv
    for v in ("T", "TE"):
        g[f"Q_syn({v})"] = q[f"syn_{v}"]["pass"]
        g[f"Q_real({v})"] = q[f"real_{v}"]["pass"]
        g[f"QV({v})"] = qv[v]["pass"]
    V = [k for k in g if k.startswith("V") and not g[k]]
    Cg = [k for k in g if k[:2] in ("C1", "C2", "C3", "C4", "C5", "C6") and not g[k]]
    no_q = all(not g[f"Q_syn({v})"] and not g[f"Q_real({v})"] for v in ("T", "TE"))
    if V:
        case = "D"
    elif Cg or no_q:
        case = "C"
    elif any(g[f"Q_syn({v})"] and g[f"Q_real({v})"] and g[f"QV({v})"] for v in ("T", "TE")):
        case = "A"
    else:
        case = "B"
    res = {"schema": "w117_decision_v1", "declaration_sha256": a.declaration_sha, "visual_review_sha256": S.sha_file(a.visual_review),
           "gates": g, "failed": {"V": V, "C": Cg}, "case": case, "detail": det}
    dump(out / "decision.json", res)
    print(json.dumps({"case": case, "gates": g, "failed": res["failed"]}, indent=1))
    return 0


# --------------------------------------------------------------------------------------------- manifest ---
def cmd_manifest(a):
    base = Path(a.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != "evidence_manifest.json" and "store" not in p.relative_to(base).parts:
            files[p.relative_to(base).as_posix()] = {"sha256": S.sha_file(p), "bytes": p.stat().st_size}
    dump(base / "evidence_manifest.json", {"schema": "w117_evidence_manifest_v1", "declaration_sha256": a.declaration_sha,
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
    for name, extra in (("build", ("tag",)), ("evaluate", ("build_dir", "preflight")), ("assess", ("build_dir", "eval_dir")),
                        ("render", ("build_dir", "eval_dir", "assess_dir")),
                        ("decide", ("build_dir", "build_dir2", "eval_dir", "eval_dir2", "assess_dir", "visual_review"))):
        p = sub.add_parser(name)
        if name != "evaluate":
            p.add_argument("--declaration", required=True)
            p.add_argument("--declaration-sha", required=True)
        for n in extra + ("out_dir",):
            p.add_argument("--" + n.replace("_", "-"), required=True)
    p = sub.add_parser("manifest"); p.add_argument("--base", required=True); p.add_argument("--declaration-sha", required=True)  # noqa: E702
    a = ap.parse_args()
    return {"preflight": cmd_preflight, "calibrate": cmd_calibrate, "declare": cmd_declare, "build": cmd_build, "evaluate": cmd_evaluate,
            "assess": cmd_assess, "render": cmd_render, "decide": cmd_decide, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
