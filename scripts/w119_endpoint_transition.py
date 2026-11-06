#!/usr/bin/env python3
"""W119: endpoint-space transition candidate with isolated per-request real-rig baselines (evidence only, NON-PRODUCTION).

Compares, per request, on the same parent, user edit U, rig, hard pin, authorised DOF and temporal budget:
  U_r  unpinned user edit (control)
  F_r  frozen W116 framewise resolver applied ONLY to this request's pin frames (isolated per-request state)
  X_r  frozen W118 offset-layer transition editor on this request only (its frozen per-side margins)
  Y_r  W119 candidate: F_r hard-pin base + wrist WORLD-SPACE guided transition on the SAME per-side margins as X_r
       (w119_endpoint_transition_base)
  TE   frozen W117 easing, historical secondary control, hold requests only (per-request states)

Stages (each refuses an existing output directory):
  preflight  identity, W113-W118 Worklogs/markers, W116/W117/W118 manifests, declarations, decisions and CASEs, inputs,
             parent rebuild + fresh Blender evaluation, synthetic parents (W117 A-D, W118 E, new W119 G) probed in Blender,
             timing probes
  calibrate  W118 build reproduction (state ids), READ-ONLY reproduction of the W118 V4 family-shared contamination,
             clean per-request F_r/X_r construction + isolation/equality proofs, H_G design from parent/unpinned geometry,
             frozen control definitions + expected population (hash).  No Y exists.
  declare    frozen declaration -> hash
  build      F_r/X_r (must equal the frozen controls) + Y_r + negative controls
  evaluate   Blender evaluation (shared independent evaluator: scripts/w116_blender_stages.py, unchanged)
  assess     footprints, isolation, display truth, decisions, quality, Y-vs-X, guide residuals, negative/display controls,
             lineage/revert (Blender), source guards
  render     timelines, step + absolute flex charts, before/after bone-proxy filmstrips (band BELOW each tile) + GIFs
  decide     frozen gates + recorded agent visual review -> CASE
  manifest   evidence manifest
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w116_semantic_pin_base as S  # noqa: E402
import w116_semantic_pin_contract as C  # noqa: E402
import w117_temporal_pin_base as TB  # noqa: E402
import w117_temporal_pin as W7  # noqa: E402
import w118_transition_edit_base as XB  # noqa: E402
import w118_transition_edit as W8  # noqa: E402
import w119_endpoint_transition_base as YB  # noqa: E402

BATCH = 119
INSTRUCTION_BLOB = "5f97c3a23f066da0081f392a4234e4b6413620bb"
STARTING_HEAD = "78809d3e641e322b62cb7f64e5b834a063b403a3"
W118_COMPLETE_COMMIT = "913bac2193cd1d2dac508a67a44c45ec9a871ea4"
W118_START_COMMIT = "0dbcbbfcc97defd98061d9416a69dac7eae6cca1"
W118_INSTRUCTION_BLOB = "2468904ff2e087fcff65c09729014d1968e6936f"
BRANCH = "arch/single_frame_first"
LOCK = W8.LOCK
W116, W117, W118 = W7.W116, W8.W117, Path("C:/Users/dna10/animcv-w118")
W118_MANIFEST_SHA = "90ac1b7fcb5a744d01ed5bd4d40ad181b894095cb4d73be9798d48bcfbf3052f"
W118_DECL_SHA = "5d61b552d2f4239801fac5e686cee5adfafdf4ac2c54f5697904af2ee4652d00"
W118_DECISION_SHA = "91200bd4423048009b8997d2fbb0a34d1d6c3f6fbcd182af51684268e075f3a1"
HIST = dict(W8.HIST) | {118: "docs/118_WORKLOG_AUTOMATIC_TRANSITION_EDIT_FOOTPRINT_VISIBILITY.md"}
UPPER, LOWER, ARM = W7.UPPER, W7.LOWER, W7.ARM
REAL_FRAMES, SYN_FRAMES = W8.REAL_FRAMES, W8.SYN_FRAMES

# ------------------------------------------------------------------------------------- new W119 held-out design ---
TRAJ_G = {"role": "held_out_evaluation_w119", "tw": ([5, -10, 0, 0], [8, 1.2, 0.6]), "bx": ([0, 0, 0, 0], [5, 0.8, 1.0]),
          "az": ([-15, -25, 30, 0], [6, 1.6, 0.3]), "ax": ([5, 0, 0, 0], [22, 0.5, 2.0]), "flex": ([35, 40, 0, 0], [12, 1.0, 1.2])}
HG_RULE = ("H_G: trajectory G, edit SYN_E8, one 10-frame hold [lo, lo+9] captured at lo, lo in [9, 22] (full 8-frame margin "
           "room both sides); eligible if every hold frame is design-feasible (W116 reach band, U shoulder) with min outer "
           "slack >= 0.010 m and min(entry, release offset) >= 0.005 m; choose the eligible lo with the largest |entry offset - "
           "release offset| (asymmetry), ties -> smaller lo.  Computed from parent/unpinned geometry only (Python FK + Blender "
           "probe capture); no F/X/Y exists when it is applied.")
SYN_PRIMARY = ["H_A", "H_B", "H_C", "H_E", "H_S0", "H_END", "DJ_B", "S_PF_C_0_29", "W_PF_C", "H_G"]
SYN_HELD_OUT = "H_G"
RH1_FAMILY = ["H_RH1_none", "H_RH1_RBneg", "H_RH1_RBpos"]
RH2_INDEPENDENT = ["H_RH2_RBpos", "DJ_R"]
CORRELATION = {"RH1_window_109_118": RH1_FAMILY + ["S_RA_pos_109_118", "S_RB_pos_109_118", "DJ_R"],
               "RH2_window_154_181": ["H_RH2_RBpos", "H_RH2_none", "H_RH2_RBneg", "S_RA_pos_154_182", "S_RB_pos_154_182", "DJ_R"]}
CONTAMINATED_W118_WINDOWS = ["S_RB_pos_109_118", "S_RB_pos_86_106", "S_RA_pos_60_61", "S_RA_pos_63_64", "S_RA_pos_78_107",
                             "S_RA_pos_109_118"]
REPRESENTATIVE = {"syn": ["H_G", "H_B", "DJ_B"], "real": ["H_RH1_none", "H_RH2_RBpos", "DJ_R", "S_RB_pos_86_106"]}
REVIEW_SCALE = ("GIF frames: one 300x300 px Blender bone-proxy tile per variant (U/F/X/Y side by side), orthographic scale 1.1 "
                "(real) / 1.2 (synthetic), front camera centred on the first pin target; filmstrip tiles are half size")

NEG_Y = {  # rid: (source request, description, expected [decision, failure_class], required reason or None)
    "NB_SHARED": ("S_RB_pos_86_106", "family-shared SEG_RB_pos__F used as F_r (isolation preflight)", ["REJECT", "CONTRACT_VIOLATION"],
                  "BASELINE_NOT_ISOLATED"),
    "NB_SIBLING": ("S_RA_pos_60_61", "F_r of S_RA_pos_60_61 plus ONE sibling-segment pin frame (63) edited", ["REJECT", "CONTRACT_VIOLATION"],
                   "BASELINE_NOT_ISOLATED"),
    "NY_U": ("H_RH1_RBpos", "Y built on the RB_neg user edit (mismatched U parent motion)", ["REJECT", "CONTRACT_VIOLATION"],
             "AB_FAIRNESS:U_MISMATCH:Y"),
    "NY_RIG": ("H_A", "Y state claims another rig identity", ["REJECT", "CONTRACT_VIOLATION"], "AB_FAIRNESS:RIG_IDENTITY_MISMATCH:Y"),
    "NY_MARGIN": ("H_A", "Y built with release margin = X margin - 1 (mismatched budget)", ["REJECT", "CONTRACT_VIOLATION"],
                  "AB_FAIRNESS:MARGIN_BUDGET_MISMATCH:Y"),
    "NY_BONE": ("H_A", "Y with base (user-edit/unauthorised) +1 deg on a transition frame", ["REJECT", "CONTRACT_VIOLATION"],
                "UNAUTHORIZED_CHANNEL_CHANGE"),
    "RY_BONE": ("H_RH1_none", "Y with spine_01 +1 deg on a transition frame (real)", ["REJECT", "CONTRACT_VIOLATION"],
                "UNAUTHORIZED_CHANNEL_CHANGE"),
    "NY_ROOT": ("H_A", "Y with __object__ +1 deg on a transition frame", ["REJECT", "CONTRACT_VIOLATION"], "ROOT_MOVED"),
    "NY_OUTSIDE": ("H_A", "Y with upper +1 deg on frame 2 (outside budget and planned domain)", ["REJECT", "CONTRACT_VIOLATION"],
                   "FOOTPRINT_EXCEEDS_DECLARED_DOMAIN"),
    "NY_PLANNED": ("H_A", "Y values unchanged; planned domain omits its outermost transition frame", ["REJECT", "CONTRACT_VIOLATION"],
                   "FOOTPRINT_EXCEEDS_DECLARED_DOMAIN"),
    "NY_GUIDE_UNREACH": ("H_B", "Y guide pushed 0.05 m beyond full reach on every transition frame (feasible hard pin)",
                         ["ACCEPT", "NONE"], None),
    "NY_INFEASIBLE": ("H_Cx", "Y on the infeasible hard pin request", ["REJECT", "INFEASIBLE_REQUEST"], None),
    "NY_CONFLICT": ("N6", "Y on contradictory hard constraints", ["REJECT", "INFEASIBLE_REQUEST"], "CONFLICTING_HARD_CONSTRAINTS"),
    "NY_DOF": ("N3", "Y on a forearm-only authorisation (missing upper-arm DOF)", ["REJECT", "INFEASIBLE_REQUEST"],
               "Y_MISSING_AUTHORIZED_DOF"),
    "NY_NONCONV": ("H_A", "Y with fault-injected skipped forearm aim (nonconvergence detection)", ["REJECT", "SOLVER_FAILURE"],
                   "NON_CONVERGED"),
    "NY_AMBIG": ("H_B", "Y pole reference forced onto the shoulder->guide axis on every transition frame", ["ACCEPT", "NONE"], None),
    "NY_PIN1": ("H_A", "Y with lower +2 deg on pin frame 20 (one corrupted hard-pin frame)", ["REJECT", "SOLVER_FAILURE"],
                "EVALUATED_PIN_UNSATISFIED"),
    "RY_PIN1": ("H_RH1_none", "Y with lowerarm_l +2 deg on pin frame 113", ["REJECT", "SOLVER_FAILURE"], "EVALUATED_PIN_UNSATISFIED"),
    "NY_FALSE": ("H_A", "Y claims SOLVED but returns U on the pin frames", ["REJECT", "SOLVER_FAILURE"], "EVALUATED_PIN_UNSATISFIED"),
}
DISPLAY_Y = {"DY1": ("H_A", "Y display: first out-of-pin changed frame relabelled UNCHANGED (false timeline label)"),
             "DY2": ("H_RH1_none", "Y display: one changed frame omitted from every channel row (real)"),
             "DY3": ("DJ_B", "Y display: disjoint changed runs collapsed to the union envelope"),
             "DY4": ("NY_OUTSIDE", "tampered Y output drawn from the declared planned domain instead of the actual output"),
             "DY5": ("NY_INFEASIBLE", "REJECTed Y displayed as ACCEPTED (rejected candidate shown as applied)")}

dump, load, new_dir, git, blender = C.dump, C.load, C.new_dir, C.git, C.blender
all_vals, vals_of, rot_local = W8.all_vals, W8.vals_of, W8.rot_local


def traj_state_G():
    vals = {c: [] for c in W7.SYN_CH}
    for f in SYN_FRAMES:
        tau = f / (W7.N_SYN - 1)
        r = {k: math.radians(W7.angle_at(TRAJ_G[k], tau)) for k in ("tw", "bx", "az", "ax", "flex")}
        vals["__object__"].append([0.0, 0.0, 0.0, 1.0])
        vals["base"].append(S.q_mul(S.q_axis_angle([0, 1, 0], r["tw"]), S.q_axis_angle([1, 0, 0], r["bx"])).tolist())
        vals["upper"].append(S.q_mul(S.q_axis_angle([0, 0, 1], r["az"]), S.q_axis_angle([1, 0, 0], r["ax"])).tolist())
        vals["lower"].append(S.q_axis_angle([0, 0, 1], r["flex"]).tolist())
    src = {"kind": "DECLARED_SYNTHETIC_TRAJECTORY", "trajectory": "G", "params": TRAJ_G, "rig_identity": "W117_SYN_ARM (known)",
           "video_subject_identity": "NOT_APPLICABLE"}
    return S.make_state(None, W7.SYN_SPEC["rig_id"], SYN_FRAMES, W7.SYN_CH, vals, {"kind": "ROOT_SYNTHETIC", "trajectory": "G"}, src)


def manifest_check(base, want):
    m = load(base / "evidence_manifest.json")["files"]
    bad = [f for f, r in m.items() if not (base / f).exists() or S.sha_file(base / f) != r["sha256"]]
    sha = S.sha_file(base / "evidence_manifest.json")
    return {"manifest_sha256": sha, "n_files": len(m), "mismatched": bad, "ok": sha == want and not bad}


def guard():
    g = W8.guard()
    files = dict(g.files) | {"w118_declaration": W118 / "results/declare/declaration.json",
                              "w118_decision": W118 / "results/decision/decision.json", "w118_manifest": W118 / "evidence_manifest.json",
                              "w118_build_r1_report": W118 / "results/build_r1/build_report.json",
                              "w118_real_states": W118 / "results/build_r1/real_states.json",
                              "w118_syn_states": W118 / "results/build_r1/syn_states.json",
                              "w118_eval_e1_real": W118 / "results/eval_e1/real_eval.json",
                              "w118_eval_e1_syn": W118 / "results/eval_e1/syn_eval.json",
                              "w118_assess": W118 / "results/assess/assess_report.json",
                              "w118_preflight": W118 / "results/preflight/preflight.json"}
    return S.SourceGuard(files)


# ------------------------------------------------------------------------------------------------- preflight ---
def cmd_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"schema": "w119_preflight_v1", "stop": [], "runtime": C.runtime(), "label": "AGENT-RUN (Main-local)"}
    rep["head"], rep["instruction_blob"] = git("rev-parse", "HEAD"), git("rev-parse", "HEAD:docs/next_batch.md")
    rep["branch"], rep["status_porcelain"] = git("rev-parse", "--abbrev-ref", "HEAD"), git("status", "--porcelain")
    title = (ROOT / "docs/next_batch.md").read_text(encoding="utf-8").splitlines()[0]
    rep["instruction_title"] = title
    if rep["instruction_blob"] != INSTRUCTION_BLOB or not title.startswith("# BATCH 119"):
        rep["stop"].append("instruction is not the bound Batch 119 blob")
    if rep["head"] != STARTING_HEAD:
        rep["stop"].append("HEAD is not the bound starting commit")
    if rep["branch"] != BRANCH:
        rep["stop"].append("wrong branch")
    lock = load(LOCK) if LOCK.exists() else None
    rep["main_local_lock"] = lock
    if not lock or lock.get("batch") != BATCH or lock.get("instruction_blob") != INSTRUCTION_BLOB or lock.get("start_commit") != STARTING_HEAD:
        rep["stop"].append("Main-local lock is not this Batch 119 claim")
    hist = {}
    for n, wl in HIST.items():
        lines = (ROOT / wl).read_text(encoding="utf-8").rstrip().splitlines()
        hist[n] = {"marker": lines[-2:] == ["AUTOMATION_BATCH_STATUS: COMPLETE", f"BATCH_NUMBER: {n}"],
                   "blob_at_start": git("rev-parse", f"{STARTING_HEAD}:{wl}"), "blob_now": git("hash-object", str(ROOT / wl))}
        if not hist[n]["marker"] or hist[n]["blob_at_start"] != hist[n]["blob_now"]:
            rep["stop"].append(f"W{n} Worklog record not intact")
    rep["historical_worklogs"] = hist
    rep["no_other_w119_worklog"] = not list((ROOT / "docs").glob("119_*"))
    if not rep["no_other_w119_worklog"]:
        rep["stop"].append("a Batch 119 Worklog already exists")
    try:
        git("merge-base", "--is-ancestor", W118_COMPLETE_COMMIT, "HEAD")
        rep["w118_complete_commit_is_ancestor"] = True
    except Exception:
        rep["w118_complete_commit_is_ancestor"] = False
        rep["stop"].append("W118 COMPLETE commit is not an ancestor of HEAD")
    rep["w118_instruction_blob_at_its_start"] = git("rev-parse", f"{W118_START_COMMIT}:docs/next_batch.md")
    if rep["w118_instruction_blob_at_its_start"] != W118_INSTRUCTION_BLOB:
        rep["stop"].append("W118 instruction provenance mismatch")
    rep["w116_manifest"] = W7.w116_manifest_check()
    rep["w117_manifest"] = W8.w117_manifest_check()
    rep["w118_manifest"] = manifest_check(W118, W118_MANIFEST_SHA)
    for k in ("w116_manifest", "w117_manifest", "w118_manifest"):
        if not rep[k]["ok"]:
            rep["stop"].append(f"{k} does not re-verify")
    fp = {}
    for n, base, dsha, csha, case in ((116, W116, W7.W116_DECL_SHA, W7.W116_DECISION_SHA, "B"),
                                      (117, W117, W8.W117_DECL_SHA, W8.W117_DECISION_SHA, "C"),
                                      (118, W118, W118_DECL_SHA, W118_DECISION_SHA, "D")):
        dec = load(base / "results/decision/decision.json")
        fp[n] = {"declaration_sha256": S.sha_file(base / "results/declare/declaration.json"),
                 "decision_sha256": S.sha_file(base / "results/decision/decision.json"), "case": dec["case"]}
        if fp[n]["declaration_sha256"] != dsha or fp[n]["decision_sha256"] != csha or fp[n]["case"] != case:
            rep["stop"].append(f"W{n} declaration/decision/CASE not the frozen ones")
    d118 = load(W118 / "results/decision/decision.json")
    fp[118]["failed"] = d118["failed"]
    fp[118]["V4_F_footprint_outside_pin"] = d118["detail"]["V4"]["F_footprint_outside_pin"]
    rep["historical_case_fingerprints"] = fp
    rep["w118_worklog_case_d"] = "CASE D" in (ROOT / HIST[118]).read_text(encoding="utf-8").split("## 2.")[0]
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
    rep["w105_audit"] = {"value_digest": fa["actions"]["W91 real_clip FIXED"]["value_digest"], "scene_fps": fa.get("scene_fps")}
    v0 = C.real_parent_state(C.INPUTS["w105_video_cases"][0])
    w118_v0 = load(W118 / "results/preflight/real_parent_state.json")
    rep["parent_rebuild"] = {"state_id": v0["state_id"], "w118_state_id": w118_v0["state_id"],
                             "canonical_equal": S.canon(v0) == S.canon(w118_v0), "verifies": S.verify_state(v0)}
    if not rep["parent_rebuild"]["canonical_equal"]:
        rep["stop"].append("rebuilt parent differs from the W118 parent")
    dump(out / "real_parent_state.json", v0)
    C.states_file(out / "real_parent_states.json", {"v0": v0}, "w105")
    rest = C.INPUTS["w105_rig_rest"][0]
    blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest", rest,
            "--states", out / "real_parent_states.json", "--out", out / "real_parent_eval.json", log=out / "real_parent_eval.log")
    blender("scripts/w117_blender_timing.py", "--", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest", rest, "--states",
            out / "real_parent_states.json", "--out", out / "real_timing.json", "--sample-frames", "1,109,118,153,154,197",
            log=out / "real_timing.log")
    ev = load(out / "real_parent_eval.json")["states"]["v0"]
    w118e = load(W118 / "results/preflight/real_parent_eval.json")["states"]["v0"]
    rep["parent_blender"] = {"value_digest": ev["value_digest"], "equals_w105_fixed_digest": ev["value_digest"] == rep["w105_audit"]["value_digest"],
                             "frames_bit_equal_w118_preflight": ev["frames"] == w118e["frames"]}
    if not (rep["parent_blender"]["equals_w105_fixed_digest"] and rep["parent_blender"]["frames_bit_equal_w118_preflight"]):
        rep["stop"].append("fresh Blender evaluation of the parent does not reproduce W105/W118")
    rep["timing"] = load(out / "real_timing.json")
    spec = load(W118 / "results/preflight/syn_spec.json")
    dump(out / "syn_spec.json", spec)
    rep["syn_spec_equal_w118"] = (out / "syn_spec.json").read_bytes() == (W118 / "results/preflight/syn_spec.json").read_bytes()
    if not rep["syn_spec_equal_w118"]:
        rep["stop"].append("synthetic spec differs from W118")
    parents = {}
    for n in ("A", "B", "C", "D", "E"):
        st = load(W118 / f"results/preflight/syn_parent_P__{n}.json")
        rb = W7.traj_state(n) if n != "E" else W8.traj_state_E()
        if S.canon(st) != S.canon(rb):
            rep["stop"].append(f"W118 synthetic parent {n} does not rebuild")
        parents[f"P__{n}"] = st
    parents["P__G"] = traj_state_G()
    for n, st in parents.items():
        dump(out / f"syn_parent_{n}.json", st)
    C.states_file(out / "syn_parent_states.json", parents, "synthetic")
    blender("scripts/w116_blender_stages.py", "--", "probe", "--rig", f"synthetic:{out / 'syn_spec.json'}", "--states",
            out / "syn_parent_states.json", "--out", out / "syn_probe.json", "--probe-channels", "base,upper,lower", log=out / "syn_probe.log")
    blender("scripts/w117_blender_timing.py", "--", "--rig", f"synthetic:{out / 'syn_spec.json'}", "--states",
            out / "syn_parent_states.json", "--out", out / "syn_timing.json", log=out / "syn_timing.log")
    sp, wp = load(out / "syn_probe.json"), load(W118 / "results/preflight/syn_probe.json")
    rep["syn_probe_parity_w118"] = {n: sp["states"][n]["frames"] == wp["states"][n]["frames"] for n in wp["states"]}
    rep["syn_rest_equal_w118"] = sp["rest"] == wp["rest"]
    if not all(rep["syn_probe_parity_w118"].values()) or not rep["syn_rest_equal_w118"] or "P__G" not in sp["states"]:
        rep["stop"].append("synthetic probe does not reproduce W118 or lacks P__G")
    rep["syn_timing"] = load(out / "syn_timing.json")
    rep["syn_parent_state_ids"] = {n: s["state_id"] for n, s in parents.items()}
    rep["source_guard"] = guard().before
    dump(out / "preflight.json", rep)
    print(json.dumps({"stop": rep["stop"], "parent_blender": rep["parent_blender"], "w118_manifest": rep["w118_manifest"]["ok"],
                      "syn_parity": rep["syn_probe_parity_w118"]}, indent=1))
    return 1 if rep["stop"] else 0


# ------------------------------------------------------------------------------------------ W119 build engine ---
def vlist(v):
    return {c: [float(x) for x in np.asarray(q, float)] for c, q in v.items()}


class W9Build:
    """W118 build reproduction + clean per-request F_r / X_r (+ Y_r and negative controls when with_y)."""

    def __init__(self, pre, hg, with_y):
        self.pre, self.hg, self.with_y = Path(pre), hg, with_y
        t0 = time.time()
        self.X = W8.XBuild(load(W118 / "results/declare/declaration.json"))
        X, B = self.X, self.X.B
        for rid, (tr, edit, holds, role) in W8.NEW_SYN.items():
            X.new_syn(rid, tr, edit, holds, role)
        for rid, (e, holds, role) in W8.NEW_REAL.items():
            X.new_real(rid, e, holds, role)
        w117_x = [r for r, R in B.req.items() if not R.get("w118_new") and R["role"] in ("evaluation", "development", "noop")]
        for rid in sorted(w117_x) + W8.X_W117_NEG + [r for r in W8.NEW_SYN if r != "Z_A"] + list(W8.NEW_REAL):
            X.make_x(rid)
        X.make_x("Z_A", m_max=0)
        self.replay_wall_s = time.time() - t0
        wb = load(W118 / "results/build_r1/build_report.json")
        mism = {"real": [n for n, s in B.real.items() if wb["real_states"].get(n) != s["state_id"]],
                "syn": [n for n, s in B.syn.items() if wb["syn_states"].get(n) != s["state_id"]]}
        missing = {"real": sorted(set(wb["real_states"]) - set(B.real) - {f"{t}__X" for t in ("RX1", "RX5", "RX6")}),
                   "syn": sorted(set(wb["syn_states"]) - set(B.syn) - {f"NX{i}__X" for i in range(1, 7)} - {"KF__X"})}
        self.parity = {"w117_replay": {k: (v if not isinstance(v, dict) else all(v.values())) for k, v in X.par.items()},
                       "w118_state_id_mismatch": mism, "w118_states_not_reproduced_excluding_tampers": missing,
                       "n_real": len(B.real), "n_syn": len(B.syn)}
        self.parity["ok"] = X.par["ok"] and not mism["real"] and not mism["syn"] and not missing["real"] and not missing["syn"]
        # ---- new held-out H_G (frozen design)
        probe = load(self.pre / "syn_probe.json")
        PG = load(self.pre / "syn_parent_P__G.json")
        B.sc.parents["P__G"] = PG
        B.syn["P__G"] = PG
        for fr in probe["states"]["P__G"]["frames"]:
            for bid, rec in B.syn_b.items():
                B.cap[(PG["state_id"], fr["frame_index"], bid)] = fr["world"][rec["bone"]][rec["point"]]
        if hg is not None:
            X.new_syn("H_G", "G", "SYN_E8", [(hg["capture"], hg["interval"][0], hg["interval"][1])], "evaluation_held_out_w119")
            X.make_x("H_G")
        self.recs = {}
        self.population = sorted(r for r, R in B.req.items() if "X" in R["candidates"])

    def side(self, R):
        return self.X.side(R)

    def bindings(self, R):
        return self.X.B.ctx.real_b if R["side"] == "real" else self.X.B.syn_b

    def fair_rec(self, P, Ust, R, pin_vals, margins=None, budget=None, rig_id=None, u_id=None):
        return {"parent_id": P["state_id"], "rig_id": rig_id or P["rig_id"], "U_state_id": u_id or Ust["state_id"],
                "constraint_digests": sorted(c["constraint_digest"] for c in R["request"]["constraints"]),
                "pin_frames": list(R["pin_frames"]), "auth": sorted(R["auth"]), "pin_values_digest": S.digest(pin_vals),
                "margins": margins, "budget_frames": budget}

    def record(self, rid):
        B = self.X.B
        R = B.req[rid]
        store, frames, rig = self.side(R)
        P, Ust = store[R["parent"]], store[R["U"]]
        ctrl = all_vals(Ust)
        pin, auth, chain = R["pin_frames"], R["auth"], R["chain"]
        rec = {"side": R["side"], "kind": R["kind"], "role": R["role"], "w118_role": R.get("w118_role"), "family": R.get("family"),
               "parent": R["parent"], "U": R["U"], "auth": auth, "chain": chain, "pin_frames": pin, "pin_runs": XB.pin_runs(pin),
               "request": R["request"], "validation_errors": R["validation_errors"], "edit": R.get("edit"),
               "correlation_groups": sorted(g for g, m in CORRELATION.items() if rid in m)}
        Fh = R["candidates"].get("F", {}).get("state")
        rec["F_hist"] = Fh
        rec["F_statuses"] = R["candidates"]["F"]["statuses"] if "F" in R["candidates"] else {}
        if "TE" in R["candidates"] and R["kind"] == "hold":
            rec["TE"] = R["candidates"]["TE"]["state"]
            rec["TE_margin_frames"] = R["candidates"]["TE"]["margin_frames"]
        if R["validation_errors"] or Fh is None:
            rec.update(contract_invalid=True, F=R["U"], X=R["U"], Y=R["U"], x_info=R["candidates"]["X"]["x_info"],
                       y_info={"request_status": "NOT_RESOLVED_CONTRACT_INVALID", "planned_frames": [],
                               "budget_frames": R["candidates"]["X"]["x_info"]["budget_frames"], "frames": {}, "sides": []})
            self.recs[rid] = rec
            return rec
        Fv = {k: vals_of(store[Fh], k) for k in pin}
        Fr = C.patched_state(P, W7.full_patch(P, ctrl, Fv), {"kind": "F_CANDIDATE", "request": rid})
        store[f"{rid}__Fr"] = Fr
        rec["F"] = f"{rid}__Fr"
        rec["F_r_state_id"] = Fr["state_id"]
        rec["F_r_equals_hist_F_state"] = Fr["state_id"] == store[Fh]["state_id"]
        rec["isolation_values_F_r"] = YB.isolation_check_values(all_vals(Fr), ctrl, pin, frames)
        rec["isolation_values_F_hist"] = YB.isolation_check_values(all_vals(store[Fh]), ctrl, pin, frames)
        rp = lambda k: B.rp(self.bindings(R), R["request"]["constraints"], k)  # noqa: E731
        eq, st_eq = [], []
        for k in pin:
            st, v, _ = S.resolve_frame(rig, ctrl[k], rp(k), auth, method="analytic")
            eq.append(all(np.array_equal(np.asarray(v[c], float), np.asarray(Fv[k][c], float)) for c in Fv[k]))
            st_eq.append(st == rec["F_statuses"].get(str(k)))
        rec["F_r_equals_frozen_single_frame_resolver"] = {"values_bit_equal": all(eq), "statuses_equal": all(st_eq), "n": len(eq)}
        m_max = 0 if rid == "Z_A" else XB.M_MAX
        xv, xi = XB.transition_edit(rig, ctrl, Fv, pin, frames, auth, chain[1], m_max=m_max)
        Xr = C.patched_state(P, W7.full_patch(P, ctrl, xv), {"kind": "X_CANDIDATE", "request": rid})
        store[f"{rid}__Xr"] = Xr
        rec["X"] = f"{rid}__Xr"
        xi = {k: v for k, v in xi.items() if k != "wall_s"}
        rec["x_info"] = xi
        rec["X_r_state_id"] = Xr["state_id"]
        rec["X_r_equals_w118_X"] = Xr["state_id"] == store[f"{rid}__X"]["state_id"]
        margins = YB.margins_of(xi)
        rec["margins"] = margins
        pinv = {str(k): vlist(Fv[k]) for k in pin}
        rec["fair"] = {"F": self.fair_rec(P, Ust, R, pinv),
                       "X": self.fair_rec(P, Ust, R, {str(k): vlist(xv[k]) for k in pin}, margins, xi["budget_frames"])}
        if self.with_y:
            yv, yi = YB.endpoint_transition(rig, ctrl, Fv, pin, frames, auth, chain[0], chain[1], margins)
            Yr = C.patched_state(P, W7.full_patch(P, ctrl, yv), {"kind": "Y_CANDIDATE", "request": rid})
            store[f"{rid}__Yr"] = Yr
            rec["Y"] = f"{rid}__Yr"
            rec["y_info"] = yi
            rec["fair"]["Y"] = self.fair_rec(P, Ust, R, {str(k): vlist(yv[k]) for k in pin}, yi["margins"], yi["budget_frames"])
            rec["_yv"] = yv
        rec["_ctrl"], rec["_Fv"] = ctrl, Fv
        self.recs[rid] = rec
        return rec

    def build_all(self):
        for rid in self.population:
            self.record(rid)
        if self.with_y:
            self.negatives()

    # ---------------- W119 negative controls
    def neg(self, rid, src, Yvals=None, F_state=None, yi=None, fair_mut=None, statuses=None, planned=None, rig_id=None):
        B = self.X.B
        R0, r0 = B.req[src], self.recs[src]
        store, frames, rig = self.side(R0)
        P = store[R0["parent"]]
        rec = {k: copy.deepcopy(v) for k, v in r0.items() if not k.startswith("_")}
        rec.update(role="negative", negative_of=src, desc=NEG_Y[rid][1], expect=NEG_Y[rid][2], expect_reason=NEG_Y[rid][3])
        if F_state is not None:
            rec["F"] = F_state
            rec["isolation_values_F_r"] = YB.isolation_check_values(all_vals(store[F_state]), r0["_ctrl"], R0["pin_frames"], frames)
        if Yvals is not None:
            st = C.patched_state(P, W7.full_patch(P, r0["_ctrl"], Yvals), {"kind": "TAMPERED_OR_CONTROL_Y", "request": rid})
            if rig_id:
                st = S.make_state(st["parent_id"], rig_id, st["frames"], st["channels"], st["values"], st["op"], st.get("source"))
            store[f"{rid}__Yr"] = st
            rec["Y"] = f"{rid}__Yr"
            rec["Y_rig_id"] = st["rig_id"]
            pinv = {str(k): vlist(Yvals[k]) for k in R0["pin_frames"] if k in Yvals}
            rec["fair"]["Y"]["pin_values_digest"] = S.digest(pinv)
        if yi is not None:
            rec["y_info"] = yi
            rec["fair"]["Y"]["margins"], rec["fair"]["Y"]["budget_frames"] = yi["margins"], yi["budget_frames"]
        if rig_id:
            rec["fair"]["Y"]["rig_id"] = rig_id
        if fair_mut:
            fair_mut(rec["fair"])
        if statuses is not None:
            rec["Y_statuses_override"] = statuses
        if planned is not None:
            rec["y_info"] = dict(rec["y_info"], planned_frames=planned(rec["y_info"]))
        self.recs[rid] = rec

    def negatives(self):
        B = self.X.B
        r = self.recs
        # baseline isolation controls
        self.neg("NB_SHARED", "S_RB_pos_86_106", F_state="SEG_RB_pos__F")
        R0 = B.req["S_RA_pos_60_61"]
        store, frames, rig = self.side(R0)
        P, ctrl = store[R0["parent"]], r["S_RA_pos_60_61"]["_ctrl"]
        sib = dict(r["S_RA_pos_60_61"]["_Fv"])
        sib[63] = vals_of(store["SEG_RA_pos__F"], 63)
        store["NB_SIBLING__Fr"] = C.patched_state(P, W7.full_patch(P, ctrl, sib), {"kind": "F_CANDIDATE_CONTAMINATED", "request": "NB_SIBLING"})
        self.neg("NB_SIBLING", "S_RA_pos_60_61", F_state="NB_SIBLING__Fr")
        # mismatched U parent motion: Y of H_RH1_RBpos resolved on RB_neg control
        R0 = B.req["H_RH1_RBpos"]
        store, frames, rig = self.side(R0)
        ctrl_neg = all_vals(store["RB_neg__control"])
        r0 = r["H_RH1_RBpos"]
        yv, yi = YB.endpoint_transition(rig, ctrl_neg, r0["_Fv"], R0["pin_frames"], frames, R0["auth"], UPPER, LOWER, r0["margins"])
        P = store[R0["parent"]]
        st = C.patched_state(P, W7.full_patch(P, ctrl_neg, yv), {"kind": "Y_ON_WRONG_U", "request": "NY_U"})
        store["NY_U__Yr"] = st
        self.neg("NY_U", "H_RH1_RBpos", yi=yi, fair_mut=lambda f: f["Y"].__setitem__("U_state_id", store["RB_neg__control"]["state_id"]))
        self.recs["NY_U"]["Y"] = "NY_U__Yr"
        # synthetic H_A family
        ra = r["H_A"]
        self.neg("NY_RIG", "H_A", Yvals=ra["_yv"], rig_id="W119_OTHER_RIG")
        R0 = B.req["H_A"]
        store, frames, rig = self.side(R0)
        mm = dict(ra["margins"])
        key = next(k for k in mm if k.endswith("release"))
        mm[key] = max(0, mm[key] - 1)
        yv, yi = YB.endpoint_transition(rig, ra["_ctrl"], ra["_Fv"], R0["pin_frames"], frames, R0["auth"], "upper", "lower", mm)
        self.neg("NY_MARGIN", "H_A", Yvals=yv, yi=yi)
        tf = sorted(k for k in ra["y_info"]["planned_frames"] if k not in set(R0["pin_frames"]))[0]

        def tam(src, fn):
            vv = {k: {c: np.array(x, float) for c, x in v.items()} for k, v in r[src]["_yv"].items()}
            fn(vv, r[src]["_ctrl"])
            return vv
        self.neg("NY_BONE", "H_A", Yvals=tam("H_A", lambda vv, c: vv[tf].__setitem__("base", rot_local(vv[tf]["base"], [1, 0, 0], 1.0))))
        self.neg("NY_ROOT", "H_A", Yvals=tam("H_A", lambda vv, c: vv[tf].__setitem__("__object__", rot_local(vv[tf]["__object__"], [0, 0, 1], 1.0))))
        self.neg("NY_OUTSIDE", "H_A", Yvals=tam("H_A", lambda vv, c: vv.__setitem__(2, dict(c[2], upper=rot_local(c[2]["upper"], [0, 0, 1], 1.0)))))
        self.neg("NY_PLANNED", "H_A", Yvals=tam("H_A", lambda vv, c: None),
                 planned=lambda yi_: [k for k in yi_["planned_frames"] if k != min(yi_["planned_frames"])])
        self.neg("NY_PIN1", "H_A", Yvals=tam("H_A", lambda vv, c: vv[20].__setitem__("lower", rot_local(vv[20]["lower"], [0, 0, 1], 2.0))))

        def false_solved(vv, c):
            for k in R0["pin_frames"]:
                vv[k] = dict(c[k])
        self.neg("NY_FALSE", "H_A", Yvals=tam("H_A", false_solved), statuses={str(k): "SOLVED" for k in R0["pin_frames"]})
        yv, yi = YB.endpoint_transition(rig, ra["_ctrl"], ra["_Fv"], R0["pin_frames"], frames, R0["auth"], "upper", "lower",
                                        ra["margins"], fault="skip_lower_aim")
        self.neg("NY_NONCONV", "H_A", Yvals=yv, yi=yi)
        # H_B guide / pole controls
        rb = r["H_B"]
        R0 = B.req["H_B"]

        def push(k, Sh, G, Lsum):
            return Sh + (G - Sh) / np.linalg.norm(G - Sh) * (Lsum + 0.05)

        def on_axis(k, Sh, G, Rr):
            return Sh + 0.5 * (G - Sh)
        yv, yi = YB.endpoint_transition(rig, rb["_ctrl"], rb["_Fv"], R0["pin_frames"], frames, R0["auth"], "upper", "lower",
                                        rb["margins"], guide_override=push)
        self.neg("NY_GUIDE_UNREACH", "H_B", Yvals=yv, yi=yi)
        yv, yi = YB.endpoint_transition(rig, rb["_ctrl"], rb["_Fv"], R0["pin_frames"], frames, R0["auth"], "upper", "lower",
                                        rb["margins"], pole_override=on_axis)
        self.neg("NY_AMBIG", "H_B", Yvals=yv, yi=yi)
        # infeasible / conflicting / missing DOF: the Y_r of the request itself
        for nid, src in (("NY_INFEASIBLE", "H_Cx"), ("NY_CONFLICT", "N6"), ("NY_DOF", "N3")):
            self.neg(nid, src)
        # real tamper controls
        rr = r["H_RH1_none"]
        rtf = sorted(k for k in rr["y_info"]["planned_frames"] if k not in set(rr["pin_frames"]))[1]
        self.neg("RY_BONE", "H_RH1_none", Yvals=tam("H_RH1_none", lambda vv, c: vv[rtf].__setitem__(
            "spine_01", rot_local(vv[rtf]["spine_01"], [1, 0, 0], 1.0))))
        self.neg("RY_PIN1", "H_RH1_none", Yvals=tam("H_RH1_none", lambda vv, c: vv[113].__setitem__(
            LOWER, rot_local(vv[113][LOWER], [0, 0, 1], 2.0))))

    def report(self):
        recs = {r: {k: v for k, v in x.items() if not k.startswith("_")} for r, x in self.recs.items()}
        B = self.X.B
        return {"records": recs, "population": self.population,
                "real_states": {n: s["state_id"] for n, s in B.real.items()}, "syn_states": {n: s["state_id"] for n, s in B.syn.items()},
                "state_meta": {side: {n: {"parent_id": s["parent_id"], "op": s["op"], "rig_id": s["rig_id"]} for n, s in st.items()}
                               for side, st in (("real", B.real), ("syn", B.syn))}}


# ------------------------------------------------------------------------------------------------ calibrate ---
def contamination_reproduction():
    """READ-ONLY: W118 V4 failure mechanism from the frozen W117/W118 artefacts (no rerun of W118 decide)."""
    wb = load(W118 / "results/build_r1/build_report.json")
    ev = load(W118 / "results/eval_e1/real_eval.json")
    sev = load(W118 / "results/eval_e1/syn_eval.json")
    req = wb["requests"]
    segs = sorted(r for r, R in req.items() if R["kind"] == "segment")
    out = {}
    for rid in segs:
        R = req[rid]
        evx = ev if R["side"] == "real" else sev
        fam = R["family"]
        sib = {s: req[s]["pin_frames"] for s in segs if req[s]["family"] == fam and s != rid}
        sib_frames = {k for v in sib.values() for k in v}
        frames = REAL_FRAMES if R["side"] == "real" else SYN_FRAMES
        Ff, Uf = C.ev_frames(evx, R["candidates"]["F"]["state"]), C.ev_frames(evx, R["U"])
        fp = XB.derive_footprint(Ff, Uf, frames, R["pin_frames"], set(), wrist=(R["chain"][1], "tail"), elbow=(R["chain"][0], "tail"))
        outside = fp["out_of_pin_changed"]
        lo, hi = R["pin_frames"][0], R["pin_frames"][-1]
        win = set(range(max(frames[0], lo - XB.M_MAX - 1), min(frames[-1], hi + XB.M_MAX + 1) + 1))
        out[rid] = {"family_shared_state": R["candidates"]["F"]["state"], "pin_runs": XB.pin_runs(R["pin_frames"]),
                    "n_out_of_pin_changed_blender": len(outside), "out_of_pin_runs": XB.runs_of(outside),
                    "all_out_of_pin_on_sibling_pins": set(outside) <= sib_frames,
                    "sibling_frames_inside_quality_window": sorted(win & sib_frames),
                    "quality_window_contaminated": bool(win & set(outside))}
    dec = load(W118 / "results/decision/decision.json")
    contaminated = sorted(r for r, x in out.items() if x["n_out_of_pin_changed_blender"])
    return {"segments": out, "w118_decision_sha256": S.sha_file(W118 / "results/decision/decision.json"), "w118_case": dec["case"],
            "w118_V4_list": dec["detail"]["V4"]["F_footprint_outside_pin"], "reproduced_contaminated_requests": contaminated,
            "reproduced_equals_w118_V4_list": contaminated == sorted(dec["detail"]["V4"]["F_footprint_outside_pin"]),
            "quality_window_contaminated": sorted(r for r, x in out.items() if x["quality_window_contaminated"]),
            "quality_window_equals_w118_worklog_six": sorted(r for r, x in out.items() if x["quality_window_contaminated"])
            == sorted(CONTAMINATED_W118_WINDOWS),
            "all_out_of_pin_on_siblings": all(x["all_out_of_pin_on_sibling_pins"] for x in out.values()),
            "note": "evaluated from the frozen W118 eval_e1 Blender output (read only); W118 V4 gate/decision not rerun or changed"}


def design_hg(pre):
    probe = load(pre / "syn_probe.json")
    rig = W7.syn_rig(probe)
    L1, L2 = rig.world_len("upper"), rig.world_len("lower")
    rmin, rmax = S._limits_reach(L1, L2)
    P = load(pre / "syn_parent_P__G.json")
    U = W7.syn_ctrl(rig, P, "SYN_E8")
    pts = {k: W7.fk_pts(rig, U[k], "upper", "lower") for k in SYN_FRAMES}
    cands = []
    for lo in range(9, 23):
        hi = lo + 9
        Wp = np.array(probe["states"]["P__G"]["frames"][lo]["world"]["lower"]["tail"])
        dist = [float(np.linalg.norm(Wp - pts[k][0])) for k in range(lo, hi + 1)]
        ent, rel = float(np.linalg.norm(Wp - pts[lo][2])), float(np.linalg.norm(Wp - pts[hi][2]))
        c = {"capture": lo, "interval": [lo, hi], "min_slack_m": min(rmax - d for d in dist), "min_inner_slack_m": min(d - rmin for d in dist),
             "entry_offset_m": ent, "release_offset_m": rel, "asymmetry_m": abs(ent - rel)}
        c["eligible"] = all(rmin < d <= rmax for d in dist) and c["min_slack_m"] >= 0.010 and min(ent, rel) >= 0.005
        cands.append(c)
    el = [c for c in cands if c["eligible"]]
    pick = max(el, key=lambda c: (c["asymmetry_m"], -c["capture"])) if el else None
    return {"rule": HG_RULE, "candidates": cands, "chosen": pick, "fk_rig": "W117 syn rig from the W119 probe"}


def cmd_calibrate(a):
    pre = Path(a.preflight)
    out = new_dir(a.out_dir)
    pf = load(pre / "preflight.json")
    if pf["stop"]:
        raise SystemExit("STOP: preflight failed")
    g = guard()
    rep = {"schema": "w119_calibration_v1", "label": "AGENT-RUN; no Y candidate exists in this stage"}
    rep["contamination_reproduction"] = contamination_reproduction()
    rep["hg_design"] = design_hg(pre)
    if rep["hg_design"]["chosen"] is None:
        raise SystemExit("STOP: no eligible H_G design")
    probe = load(pre / "syn_probe.json")
    rig = W7.syn_rig(probe)
    worst = 0.0
    for n, st in probe["states"].items():
        Pn = load(pre / f"syn_parent_{n}.json")
        for i, fr in enumerate(st["frames"]):
            fk = rig.fk(S.frame_vals(Pn, i))
            for b in fr["world"]:
                for p in ("head", "tail"):
                    worst = max(worst, float(np.linalg.norm(rig.world(fk, b, p) - np.array(fr["world"][b][p]))))
    rep["syn_fk_max_error_m"] = worst
    t0 = time.time()
    W = W9Build(pre, rep["hg_design"]["chosen"], with_y=False)
    W.build_all()
    rep["wall_s"] = time.time() - t0
    rep["w118_reproduction"] = W.parity
    recs = W.report()["records"]
    controls = {}
    for rid, r in recs.items():
        controls[rid] = {k: r.get(k) for k in ("side", "role", "kind", "parent", "U", "auth", "pin_frames", "F_r_state_id", "X_r_state_id",
                                                "margins", "contract_invalid", "correlation_groups", "F_hist")}
        controls[rid]["planned_frames_X"] = r["x_info"].get("planned_frames")
        controls[rid]["budget_frames"] = r["x_info"].get("budget_frames")
    rep["per_request_checks"] = {rid: {k: r.get(k) for k in ("F_r_equals_hist_F_state", "F_r_equals_frozen_single_frame_resolver",
                                                             "X_r_equals_w118_X", "isolation_values_F_r", "isolation_values_F_hist")}
                                 for rid, r in recs.items()}
    valid = [r for r, x in recs.items() if not x.get("contract_invalid")]
    rep["baselines_ok"] = {
        "F_r_isolated_value_level": all(recs[r]["isolation_values_F_r"]["ok"] for r in valid),
        "F_r_equals_frozen_resolver": all(recs[r]["F_r_equals_frozen_single_frame_resolver"]["values_bit_equal"] and
                                          recs[r]["F_r_equals_frozen_single_frame_resolver"]["statuses_equal"] for r in valid),
        "X_r_equals_w118_X": all(recs[r]["X_r_equals_w118_X"] for r in valid),
        "hold_F_r_equals_hist_F": all(recs[r]["F_r_equals_hist_F_state"] for r in valid if recs[r]["kind"] == "hold"),
        "segment_hist_F_not_isolated": sorted(r for r in valid if recs[r]["kind"] == "segment" and not recs[r]["isolation_values_F_hist"]["ok"])}
    population = {"syn_primary": SYN_PRIMARY, "syn_held_out": SYN_HELD_OUT, "rh1_family": RH1_FAMILY, "rh2_independent": RH2_INDEPENDENT,
                  "correlation_groups": CORRELATION, "contaminated_w118_windows_reconstructed": CONTAMINATED_W118_WINDOWS,
                  "all_requests": sorted(recs), "real_requests": sorted(r for r, x in recs.items() if x["side"] == "real"),
                  "syn_requests": sorted(r for r, x in recs.items() if x["side"] == "syn"),
                  "low_evidence_windows": C.WINDOWS, "negative_controls_expected": NEG_Y, "display_controls": DISPLAY_Y}
    frozen = {"schema": "w119_frozen_controls_v1", "controls": controls, "population": population,
              "hg_design": rep["hg_design"]["chosen"]}
    dump(out / "frozen_controls.json", frozen)
    rep["frozen_controls_sha256"] = S.sha_file(out / "frozen_controls.json")
    rep["source_guard"] = g.check()
    dump(out / "calibration.json", rep)
    cr = rep["contamination_reproduction"]
    print(json.dumps({"parity": W.parity["ok"], "baselines": rep["baselines_ok"], "fk": worst, "hg": rep["hg_design"]["chosen"],
                      "contamination": [cr["reproduced_equals_w118_V4_list"], cr["quality_window_equals_w118_worklog_six"],
                                        cr["all_out_of_pin_on_siblings"]], "controls": rep["frozen_controls_sha256"],
                      "n": len(recs)}, indent=1, default=str))
    return 0


# --------------------------------------------------------------------------------------------- declare ---
GATES = {
    "V1_authentication": "preflight has no STOP: HEAD/blob/branch/lock; W113-W118 Worklogs intact with COMPLETE markers; W118 COMPLETE "
                         "commit ancestor; W118 instruction provenance; W116/W117/W118 manifests re-verify; frozen declaration/decision "
                         "hashes and CASEs B/C/D; every W102/W105/W115 input hash; no other W119 Worklog",
    "V2_parent_reproduction": "rebuilt parent canonical-equal to W118; fresh Blender evaluation = W105 FIXED digest and bit-equal to the "
                              "W118 preflight evaluation; synthetic A-E rebuild and probe bit-equal to W118; G probed",
    "V3_baseline_reproduction": "W117 replay parity and every W118 build state id reproduced (tampers excluded) in calibrate and in both "
                                "builds; every contract-valid X_r state id equals the W118 X state id",
    "V4_contamination_reproduced_not_repaired": "read-only: every family-shared SEG F out-of-pin change lies on sibling pins; reproduced "
                                                "list equals the W118 V4 list; the six contaminated quality windows are reproduced; W118 "
                                                "decision hash and CASE D unchanged after the run",
    "V5_clean_isolated_baselines": "every contract-valid request: F_r value-level isolation ok AND Blender footprint of F_r is a subset of "
                                   "its pin frames; F_r pin values bit-equal to an independent re-solve with the frozen W116 resolver and "
                                   "statuses equal; build F_r/X_r ids and margins equal the frozen controls of calibrate",
    "V6_ab_fairness": "every non-negative request: F_r/X_r/Y_r share parent, rig, U, pin constraints, pin frames, authorisation and pin values; "
                      "X_r and Y_r have identical per-side margins and search budget",
    "V7_predeclaration": "declaration hash verified at every stage; frozen-controls hash in the declaration equals the calibrate file; "
                         "H_G design in the declaration equals the calibrate rule output; outcomes_inspected false",
    "V8_independent_blender_evaluation": "every state of both builds is Blender-evaluated; synthetic FK <= 1e-5 m; every Y guide residual and "
                                         "pin residual taken from Blender output",
    "V9_population_complete": "every declared request (incl. the six contaminated windows, whole-clip and low-evidence ones) has F_r/X_r/"
                              "Y_r footprints and display specs; every contract-valid request has every pin frame Blender-evaluated for "
                              "X_r and Y_r",
    "V10_repeatability": "two builds byte-identical states files and identical Y infos; two Blender evaluations identical",
    "V11_timing_semantics": "every keyframe CONSTANT; Blender frame = index - f0 + 1; sub-frame evaluation equals the keyed frame (real and "
                            "synthetic incl. G)",
    "C1_hard_pin_and_scope": "every ACCEPTed Y_r (and X_r): every pin-frame Blender residual <= 1 mm; changed channels vs U only authorised; "
                             "actual changed frames within the search budget and the planned domain; no point outside the arm domain; root "
                             "unchanged; lengths; joint limits",
    "C2_truthful_footprint": "every variant (F_r, X_r, Y_r, TE) of every request passes the independent display truth check; Y planned domain "
                             "within budget and equal to the X planned domain; preview id = committed id; result label = decision; REJECTed "
                             "never shown applied, ref head unchanged",
    "C3_negative_controls": "every W119 negative control returns its declared decision/class (and required reason); every display control "
                            "DY1-DY5 FAILS the truth check",
    "C4_atomicity_lineage_reversal": "Y ACCEPT iff all pin + transition frames pass and footprint/fairness/isolation hold; NY_PIN1/RY_PIN1 "
                                     "REJECT with exactly one failing frame; refs unchanged on every REJECT; stale parent rejected; executed "
                                     "ACCEPT + exact revert (values byte-equal, Blender bit-equal) on synthetic and real",
    "C5_source_immutability": "declared inputs, W116/W117/W118 artefacts and parent files byte-unchanged across every stage; three manifests "
                              "re-verify after the run",
    "C6_no_new_hard_failure": "Y_r ACCEPTs every contract-valid non-negative request whose X_r ACCEPTs (incl. H_RH2_RBpos)",
    "C7_user_edit_and_collateral": "every ACCEPTed Y_r keeps the user-edit channel and __object__ bit-equal to U on every frame and the "
                                   "collateral points exactly equal to U",
    "U_syn": "compare_y meaningful improvement of Y_r over clean X_r on the new held-out H_G",
    "U_real": "compare_y meaningful improvement on >= 1 of RH1 family {H_RH1_none, H_RH1_RBneg, H_RH1_RBpos} AND on >= 1 of the "
              "independent {H_RH2_RBpos, DJ_R}",
    "U_vis": "the recorded agent visual review marks >= 1 real request that passed compare_y as visibly improved in the actual-rig GIF/filmstrip "
             "at the declared review scale",
}
CASE_RULE = ("D if any V fails. Else C if any C gate fails, or if NO evaluation request (synthetic primary incl. H_G, or real) meets the "
             "compare_y meaningful improvement. Else A if U_syn, U_real and U_vis all pass. Else B.")
QUALITY_RULE = {
    "window": "W118 union window [first run lo - 9, last run hi + 9] clipped to the clip; identical for F_r, X_r, Y_r, TE",
    "B1": "peak boundary wrist COMPENSATION step |c_W(k) - c_W(k-1)|, c_W = W^cand - W^U, over the W118 boundary transitions",
    "improvement": f"B1(Y) <= {YB.IMPROVE_RATIO} B1(X) AND B1(X) - B1(Y) >= {YB.IMPROVE_ABS_M} m",
    "no_new_articulation_defect": "introduced-flip transitions of Y (vs U) subset of those of X; introduced near-straight frames (flex < 5 "
                                  "deg where U >= 5 deg, W118 diagnostic reference) of Y subset of those of X",
    "material_worsening_floors": {"rule": f"worse if Y - X > max({YB.WORSE_REL} X, floor)", "floors": YB.WORSE_FLOOR,
                                  "rationale": "25 % relative tolerance for redistribution of motion inside a matched budget; absolute "
                                               "floors at the W118 materiality scale (5 mm) for elbow compensation step and raw wrist "
                                               "step, and 2 deg/frame (stricter than W118's 5) for rotation-step excess; declared before "
                                               "any Y output exists, experiment-only, not a user-facing limit"},
    "contracts": "X_r and Y_r both ACCEPTed (hard pin, domain, atomicity, fairness, isolation)",
    "eligibility": "requests with a defined B1 (>= 1 out-of-pin frame in the window) and both X_r and Y_r ACCEPTed; whole-clip and "
                   "zero-budget requests are NO_TRANSITION_AVAILABLE (not a gain)",
    "secondary_reported_not_gated": "clean X_r vs F_r with the W118 rule (labelled W119-clean, does NOT re-decide W118); Y_r vs F_r; TE "
                                    "(hold requests only); costs: mean/peak elbow deviation from U, raw wrist steps, absolute flex series",
}


def cmd_declare(a):
    out = new_dir(a.out_dir)
    pf = load(Path(a.preflight) / "preflight.json")
    cal = load(Path(a.calibration) / "calibration.json")
    if pf["stop"] or not cal["w118_reproduction"]["ok"] or not all(v if isinstance(v, bool) else True for v in cal["baselines_ok"].values()):
        raise SystemExit("STOP: preflight/calibration failed")
    frozen = load(Path(a.calibration) / "frozen_controls.json")
    decl = {
        "schema": "w119_declaration_v1", "batch": BATCH, "outcomes_inspected": False,
        "preflight_dir": str(Path(a.preflight).resolve()), "calibration_dir": str(Path(a.calibration).resolve()),
        "preflight_sha256": S.sha_file(Path(a.preflight) / "preflight.json"),
        "calibration_sha256": S.sha_file(Path(a.calibration) / "calibration.json"),
        "frozen_controls_sha256": S.sha_file(Path(a.calibration) / "frozen_controls.json"),
        "hg_design": frozen["hg_design"], "traj_G": TRAJ_G, "hg_rule": HG_RULE,
        "baselines": {"w116": {"declaration": W7.W116_DECL_SHA, "decision": W7.W116_DECISION_SHA, "case": "B (immutable)"},
                      "w117": {"declaration": W8.W117_DECL_SHA, "decision": W8.W117_DECISION_SHA, "case": "C (immutable)"},
                      "w118": {"manifest": W118_MANIFEST_SHA, "declaration": W118_DECL_SHA, "decision": W118_DECISION_SHA,
                               "case": "D (immutable)", "complete_commit": W118_COMPLETE_COMMIT}},
        "candidate_Y": {
            "hard_pin_base": "F_r values on the pin frames (frozen W116 framewise resolver); never changed",
            "margins": "per side exactly the frozen X_r chosen margin of the same clean request; Y never chooses a margin; a margin "
                       "beyond the available budget is a REJECT (MARGIN_EXCEEDS_AVAILABLE_BUDGET), never widened",
            "guide": "G(k) = W_U(k) + w_j (W_F(b) - W_U(b)), w_j = 1 - smoothstep(j/(m+1)) (Python FK; soft objective, not a pin)",
            "pole": "R(k) = E_U(k) + w_j (E_F(b) - E_U(b)); elbow = circle point nearest R(k)",
            "reach_and_near_straight": "d_c = clip(|G-S|, r_min, min(r_max, r(phi_floor))), phi_floor = min(flex_U(k), flex_F(b)); "
                                       "statuses GUIDE_REACHED / GUIDE_FLEX_FLOOR_CLAMPED / GUIDE_UNREACHABLE_OUTER / _INNER",
            "bend_plane_ambiguity": f"R(k) within {YB.Y_AMBIG_DEG} deg of the shoulder->guide axis -> continuity pole (previous "
                                    "boundary-side solved elbow), flagged BEND_PLANE_AMBIGUOUS",
            "solve": "upper minimal world swing to the elbow point; lower minimal world swing aiming the wrist at the clamped guide",
            "convergence": f"FK residual to the clamped guide <= {YB.Y_INTERNAL_TOL_M} m else NON_CONVERGED (SOLVER_FAILURE)",
            "dof": "authorised upper/lower arm channels only; anything else -> Y_MISSING_AUTHORIZED_DOF (REJECT)",
            "fallback": "none; no fallback to F or X exists",
            "relation_to_W117_TE": "the wrist guide formula is that of W117 TE (boundary offset * (1 - smoothstep)); Y differs by the "
                                   "frozen X margins, the guided elbow pole, the flex floor and the closed-form per-frame solve (no W117 "
                                   "optimiser)",
            "parameters": {"Y_AMBIG_DEG": YB.Y_AMBIG_DEG, "Y_INTERNAL_TOL_M": YB.Y_INTERNAL_TOL_M,
                           "Y_FLIP_FLEX_FLOOR_DEG": YB.Y_FLIP_FLEX_FLOOR_DEG, "flex_limits_deg": S.ELBOW_FLEX_LIMIT_DEG},
            "own_margin_diagnostic": "NOT RUN in W119"},
        "hard_pin": {"tolerance_m": S.TOL_M, "evaluator": "Blender scene world (w116_blender_stages evaluate, unchanged)",
                     "status": "W116 provisional diagnostic criterion, unchanged"},
        "quality_rule": QUALITY_RULE, "gates": GATES, "case_rule": CASE_RULE,
        "population": frozen["population"],
        "negative_controls": NEG_Y, "display_controls": DISPLAY_Y,
        "lineage": {"synthetic": ["S1 commit Y_r of the first ACCEPTed among [H_G, H_A, H_B, H_E] on syn/main",
                                  "S2 NY_PIN1 on syn/atomic -> REJECT, head unchanged", "S3 next Y_r against the old head -> STALE_PARENT",
                                  "S4 revert syn/main -> byte-equal, Blender bit-equal"],
                    "real": ["R1 commit Y_r of the first ACCEPTed among RH1 family + RH2_independent on real/main",
                             "R2 RY_PIN1 on real/atomic -> REJECT", "R3 another Y_r against v0 -> STALE_PARENT",
                             "R4 revert real/main -> byte-equal, Blender bit-equal to v0"],
                    "per_request": "every X_r and Y_r previewed (PROPOSED, state id) and committed on its own ref"},
        "render_rule": {"representative": REPRESENTATIVE, "variants": "U, F_r, X_r, Y_r", "review_scale": REVIEW_SCALE,
                        "band": "footprint band drawn directly BELOW each tile with the variant label inside the row (fixes W118)"},
        "visual_review": "after render and before decide the agent records visual_review.json (agent-only, not independent)",
        "predictions_not_gates": {"synthetic": "Y holds the boundary compensation profile where the guide is reached, so B1(Y) should be "
                                               "below B1(X) when the parent motion changes inside the margin",
                                  "real": "where U is near-straight the flex floor clamps the guide (soft shortfall); the 153->154 source "
                                          "jump is unchanged in raw wrist steps for every variant"},
        "pre_declaration_disclosure": [
            "focused unit tests of w119_endpoint_transition_base on the W118 Python-only test rig (19 tests)",
            "preflight and calibration as recorded (no Y computed on any W119/W118 request)",
            "one crash-only smoke run of declare -> build -> evaluate -> assess -> render -> decide in a throwaway directory with stdout "
            "discarded unread; only exit codes and Python tracebacks were read; every stage exited 0 with no traceback (stderr empty), so no code or parameter was changed after it; directory (C:/Users/dna10/animcv-w119-smoke) deleted"],
    }
    dump(out / "declaration.json", decl)
    print(S.sha_file(out / "declaration.json"))
    return 0


def check_decl(a):
    decl = load(a.declaration)
    if S.sha_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash")
    return decl


# --------------------------------------------------------------------------------------------- build ---
def cmd_build(a):
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    cal = Path(decl["calibration_dir"])
    if S.sha_file(cal / "frozen_controls.json") != decl["frozen_controls_sha256"]:
        raise SystemExit("STOP: frozen controls changed")
    frozen = load(cal / "frozen_controls.json")
    g = guard()
    t0 = time.time()
    W = W9Build(decl["preflight_dir"], decl["hg_design"], with_y=True)
    W.build_all()
    rep = W.report()
    fc = frozen["controls"]
    mism = [r for r in fc if any(rep["records"][r].get(k) != fc[r].get(k) for k in ("F_r_state_id", "X_r_state_id", "margins"))]
    B = W.X.B
    C.states_file(out / "real_states.json", B.real, "w105")
    C.states_file(out / "syn_states.json", B.syn, "synthetic")
    rep.update({"schema": "w119_build_v1", "tag": a.tag, "declaration_sha256": a.declaration_sha, "wall_s": time.time() - t0,
                "replay_wall_s": W.replay_wall_s, "w118_reproduction": W.parity, "frozen_controls_mismatch": mism,
                "y_wall_s": {r: x["y_info"].get("wall_s") for r, x in rep["records"].items() if "y_info" in x},
                "source_guard": g.check()})
    dump(out / "build_report.json", rep)
    print(json.dumps({"tag": a.tag, "wall_s": rep["wall_s"], "n_real": len(B.real), "n_syn": len(B.syn), "records": len(rep["records"]),
                      "parity": W.parity["ok"], "frozen_mismatch": mism, "source": rep["source_guard"]["unchanged"]}))
    return 0


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
def classify_y(y, cmp_):
    d = y["decision"]
    rs = set(d["reasons"])
    if d["decision"] == "REJECT":
        if "Y_MISSING_AUTHORIZED_DOF" in rs:
            return "MISSING_AUTHORIZED_DOF"
        if "NON_CONVERGED" in rs:
            return "SOLVER_NONCONVERGENCE"
        if d["failure_class"] in ("INFEASIBLE_REQUEST", "UNSUPPORTED_BINDING"):
            return "INFEASIBLE_OR_CONFLICTING_OR_UNBOUND_REQUEST"
        if "EVALUATED_PIN_UNSATISFIED" in rs or "JOINT_LIMIT" in rs:
            return "HARD_PIN_OR_LIMIT_FAILURE"
        if d["failure_class"] == "CONTRACT_VIOLATION":
            return "OUTPUT_OR_BASELINE_CONTRACT_VIOLATION"
        return "REJECTED_OTHER"
    yi = y.get("y_info", {})
    if not yi.get("frames"):
        return "ACCEPTED_NO_TRANSITION_AVAILABLE"
    sc, fc = yi.get("status_counts", {}), yi.get("flag_counts", {})
    tag = []
    if sc.get("GUIDE_UNREACHABLE_OUTER") or sc.get("GUIDE_UNREACHABLE_INNER"):
        tag.append("GUIDE_UNREACHABLE")
    if sc.get("GUIDE_FLEX_FLOOR_CLAMPED"):
        tag.append("GUIDE_FLEX_FLOOR_CLAMPED")
    if fc.get("BEND_PLANE_AMBIGUOUS"):
        tag.append("BEND_PLANE_AMBIGUOUS")
    base = "VALID_REACHABLE_GUIDE" if not tag else "+".join(tag)
    if cmp_ is None:
        return base + "|NOT_COMPARED"
    return base + ("|IMPROVED_OVER_X" if cmp_["meaningful_improvement"] else "|VALID_BUT_NOT_BETTER_THAN_X")


def assess_record(rid, rec, ev, rig, bindings, rest, scale, frames_all, store_obj, full_state, states, Bb):
    side = rec["side"]
    U, P = C.ev_frames(ev, rec["U"]), C.ev_frames(ev, rec["parent"])
    pin, auth, chain = rec["pin_frames"], rec["auth"], rec["chain"]
    errs = [tuple(e) for e in rec["validation_errors"]]
    edit_ch = list(rec["request"]["edit"]["channels"]) if rec["request"].get("edit") else []
    dom = S.domain_points(rig, sorted(set(auth) | set(edit_ch))) if (auth or edit_ch) else set()
    arm_dom = S.domain_points(rig, auth) if auth else set()
    pf = W7.pins_fn(bindings, rec["request"])
    wb, eb = (chain[1], "tail"), (chain[0], "tail")
    variants = {"F": rec["F"], "X": rec["X"], "Y": rec["Y"]}
    if rec.get("TE"):
        variants["TE"] = rec["TE"]
    if rec.get("F_hist") and rec["kind"] == "segment" and rec["role"] != "negative":
        variants["F_shared_hist"] = rec["F_hist"]
    out = {}
    for v, name in variants.items():
        Xf = C.ev_frames(ev, name)
        fp = XB.derive_footprint(Xf, U, frames_all, pin, arm_dom, wrist=wb, elbow=eb)
        r = {"state": name, "state_id": ev["states"][name]["state_id"], "footprint": {k: x for k, x in fp.items() if k != "per_frame"}}
        if v == "X":
            planned, budget = rec["x_info"].get("planned_frames", []), rec["x_info"]["budget_frames"]
        elif v == "Y":
            planned, budget = rec["y_info"].get("planned_frames", []), rec["y_info"]["budget_frames"]
        elif v == "TE":
            planned = budget = sorted(set(pin) | set(rec.get("TE_margin_frames", [])))
        else:
            planned = budget = sorted(pin)
        spec = XB.display_spec(fp, frames_all, pin, budget, planned, r["state_id"], ev["states"][rec["parent"]]["state_id"],
                               "PROPOSED", rid, v, edit_ch)
        r["display_spec_proposed"] = spec
        r["display_check_proposed"] = XB.check_display(spec, Xf, U, frames_all, pin, r["state_id"])
        r["planned_frames"], r["budget_frames"] = planned, budget
        if v in ("F", "F_shared_hist"):
            r["isolation_blender"] = YB.isolation_check_footprint(fp, pin)
        if v in ("X", "Y"):
            evals, per = [], {}
            if not errs:
                for k in sorted(set(planned) | set(pin)):
                    e = S.evaluate_candidate(Xf[k], U[k], P[k], pf(k) if k in set(pin) else [],
                                             {"vs_parent": sorted(set(auth) | set(edit_ch)), "vs_control": sorted(auth)}, dom, rest, scale,
                                             chain=tuple(chain))
                    per[k] = e
                    evals.append(e)
            statuses = [s for s in (rec.get("Y_statuses_override") if v == "Y" and rec.get("Y_statuses_override") else rec["F_statuses"]).values()
                        if s != "NOT_APPLICABLE"]
            if errs:
                dec = S.decide(errs, statuses, evals)
            elif not statuses:
                dec = {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": ["NOT_RESOLVED"]}
            elif v == "X":
                dec = XB.decide_x([], statuses, evals, fp, budget, planned, auth, display_check=r["display_check_proposed"])
            else:
                fair = YB.ab_fairness(rec["fair"]) if rec.get("fair") else None
                if rec.get("Y_rig_id") and rec["Y_rig_id"] != rec["fair"]["F"]["rig_id"]:
                    fair = {"ok": False, "errors": sorted(set((fair or {}).get("errors", [])) | {"RIG_IDENTITY_MISMATCH:Y"})}
                iso = out["F"]["isolation_blender"]
                dec = YB.decide_y([], statuses, evals, fp, budget, planned, auth, rec["y_info"], fairness=fair,
                                  display_check=r["display_check_proposed"], isolation=iso)
                r["fairness"] = fair
                r["y_info"] = rec["y_info"]
                r["guide_residuals"] = YB.guide_residuals(Xf, rec["y_info"], chain[1]) if rec["y_info"].get("frames") else {}
            res = [p["candidate_residual_m"] for k in pin if k in per for p in per[k]["pins"]]
            r.update(decision=dec,
                     frame_pass={str(k): bool(e["pins_satisfied"] and e["scope_ok"] and e["locality_ok"] and e["lengths_ok"]
                                              and e["root_unchanged"] and e["joint_limit_ok"]) for k, e in per.items()},
                     residual_m={"max": max(res) if res else None, "n_over_tol": sum(1 for x in res if x > S.TOL_M), "n": len(res),
                                 "per_frame": {str(k): per[k]["pins"][0]["candidate_residual_m"] for k in pin if k in per and per[k]["pins"]}},
                     user_edit_and_object_equal_U=all(Xf[k]["stored_wxyz"].get(c) == U[k]["stored_wxyz"].get(c)
                                                      for k in frames_all for c in list(edit_ch) + [S.OBJECT]))
        if rec["role"] in ("evaluation", "development", "noop", "control") and not errs and v != "F_shared_hist":
            r["quality"] = XB.quality(Xf, U, frames_all, pin, chain[0], chain[1], auth)
            r["collateral"] = W7.collateral(Xf, U, P, list(range(frames_all[0], frames_all[-1] + 1)), side)
        out[v] = r
    # comparisons
    cmp_ = {}
    qF, qX, qY = (out[v].get("quality") for v in ("F", "X", "Y"))
    elig = qX is not None and qY is not None and qX["B1_boundary_wrist_comp_step_peak_m"] is not None and \
        out["X"]["decision"]["decision"] == "ACCEPT" and out["Y"]["decision"]["decision"] == "ACCEPT"
    if elig:
        cmp_["Y_vs_X"] = YB.compare_y(qX, qY)
        cmp_["X_vs_F_clean_W118_rule"] = XB.compare_x(qF, qX)
        cmp_["Y_vs_F_W118_rule"] = XB.compare_x(qF, qY)
    # commit X and Y on their own refs (preview -> commit -> result)
    if store_obj is not None:
        par = full_state(side, rec["parent"])
        for v in ("X", "Y"):
            x = out[v]
            cand = rebuild_state(Bb, side, rec[v], states[side][rec[v]], par)
            ref = f"{v.lower()}/{rid}"
            store_obj.put(par)
            store_obj.init_ref(ref, par["state_id"])
            if cand["state_id"] == par["state_id"] or cand.get("parent_id") != par["state_id"]:
                crec = {"decision": "REJECT", "reasons": ["CANDIDATE_IS_NOT_A_CHILD_OF_PARENT"], "head_before": par["state_id"],
                        "head_after": store_obj.refs[ref]}
                if x["decision"]["decision"] == "ACCEPT":
                    x["decision"] = {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": ["CANDIDATE_IS_NOT_A_CHILD_OF_PARENT"]}
            else:
                crec = store_obj.commit(ref, par["state_id"], cand, x["decision"])
            label = "ACCEPTED" if crec["decision"] == "ACCEPT" else "REJECTED"
            res_spec = dict(x["display_spec_proposed"], status=label, head_after=crec["head_after"])
            x["commit"] = {"preview_state_id": x["display_spec_proposed"]["state_id"], "candidate_state_id": cand["state_id"],
                           "parent_state_id": par["state_id"], **{k: crec.get(k) for k in ("decision", "reasons", "head_before", "head_after")},
                           "preview_id_equals_candidate": x["display_spec_proposed"]["state_id"] == cand["state_id"],
                           "head_consistent": (crec["head_after"] == cand["state_id"]) if crec["decision"] == "ACCEPT"
                           else (crec["head_after"] == par["state_id"])}
            x["display_spec_result"] = res_spec
            x["display_check_result"] = XB.check_display(res_spec, C.ev_frames(ev, rec[v]), U, frames_all, pin, cand["state_id"],
                                                         decision=crec["decision"])
    return {"variants": out, "comparisons": cmp_, "y_class": classify_y(out["Y"] | {"y_info": rec["y_info"]}, cmp_.get("Y_vs_X")),
            "eligible_primary": elig}


def rebuild_state(Bb, side, name, cs, par):
    if cs["state_id"] == par["state_id"]:
        return par
    meta = Bb["state_meta"][side][name]
    if meta["parent_id"] != par["state_id"]:
        raise SystemExit(f"STOP: state {name} is not a child of {par['state_id']}")
    st = S.make_state(par["state_id"], meta["rig_id"], par["frames"], par["channels"], cs["values"], meta["op"], par.get("source"))
    if st["state_id"] != cs["state_id"]:
        raise SystemExit(f"STOP: state {name} does not rebuild to its id")
    return st


def corrupt_display(kind, spec):
    s = copy.deepcopy(spec)
    if kind == "DY1":
        k = next((f for f, c in sorted(s["cells"].items(), key=lambda t: int(t[0])) if c == "AUTO_CHANGED"), None)
        if k is None:
            return None
        s["cells"][k] = "UNCHANGED"
    elif kind == "DY2":
        fr = sorted({f for v in s["channel_rows"].values() for f in v})
        if not fr:
            return None
        k = fr[len(fr) // 2]
        s["channel_rows"] = {c: [f for f in v if f != k] for c, v in s["channel_rows"].items()}
    elif kind == "DY3":
        if len(s["changed_runs"]) < 2:
            return None
        lo, hi = s["envelope"]
        s["changed_runs"] = [[lo, hi]]
        for f in range(lo, hi + 1):
            if s["cells"][str(f)] == "UNCHANGED":
                s["cells"][str(f)] = "AUTO_CHANGED"
    elif kind == "DY4":
        planned, pin = set(s["planned_frames"]), set(s["pin_frames"])
        s["cells"] = {f: (("PIN_CHANGED" if int(f) in pin else "AUTO_CHANGED") if int(f) in planned else "UNCHANGED") for f in s["cells"]}
        ch = sorted(planned)
        s["channel_rows"] = {c: ch for c in ("upper", "lower")}
        s["changed_runs"], s["envelope"] = XB.runs_of(ch), ([ch[0], ch[-1]] if ch else None)
    elif kind == "DY5":
        s["status"] = "ACCEPTED"
    return s


def lineage(out, Bb, A, full_state, states, rev, sev, pre):
    store = S.Store(out / "lineage_store")
    res = {"synthetic": [], "real": []}
    Rq = A["requests"]
    dec = lambda rid: Rq[rid]["variants"]["Y"]["decision"]  # noqa: E731
    recs = Bb["records"]

    def cand(side, rid):
        r = recs[rid]
        par = full_state(side, r["parent"])
        return rebuild_state(Bb, side, r["Y"], states[side][r["Y"]], par), par
    rev_eval = {}
    order = ["H_G", "H_A", "H_B", "H_E"]
    pick = next((r for r in order if dec(r)["decision"] == "ACCEPT"), None)
    if pick:
        c1, P = cand("syn", pick)
        store.put(P)
        store.init_ref("syn/main", P["state_id"])
        res["synthetic"].append(dict(step="S1", request=pick, **store.commit("syn/main", P["state_id"], c1, dec(pick))))
        c2, PA = cand("syn", "NY_PIN1")
        store.put(PA)
        store.init_ref("syn/atomic", PA["state_id"])
        h = store.refs["syn/atomic"]
        r2 = store.commit("syn/atomic", h, c2, dec("NY_PIN1"))
        res["synthetic"].append(dict(step="S2", head_unchanged=r2["head_after"] == h, **r2))
        other = next(r for r in order if r != pick)
        c3, P3 = cand("syn", other)
        res["synthetic"].append(dict(step="S3", request=other, **store.commit("syn/main", P["state_id"], c3, dec(other))))
        srev, srec = store.revert("syn/main")
        res["synthetic"].append(dict(step="S4", **srec))
        C.states_file(out / "syn_revert_states.json", {"syn_revert": srev}, "synthetic")
        blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", f"synthetic:{pre / 'syn_spec.json'}", "--states",
                out / "syn_revert_states.json", "--out", out / "syn_revert_eval.json", log=out / "syn_revert_eval.log")
        e = load(out / "syn_revert_eval.json")["states"]["syn_revert"]
        pn = recs[pick]["parent"]
        rev_eval["synthetic"] = {"restores": pn, "value_digest_equal_parent": e["value_digest"] == sev["states"][pn]["value_digest"],
                                 "frames_bit_equal_parent": e["frames"] == sev["states"][pn]["frames"]}
    v0 = full_state("real", "v0")
    store.put(v0)
    store.init_ref("real/main", v0["state_id"])
    store.init_ref("real/atomic", v0["state_id"])
    order_r = RH1_FAMILY + RH2_INDEPENDENT
    pick = next((r for r in order_r if dec(r)["decision"] == "ACCEPT"), None)
    if pick:
        c1, _ = cand("real", pick)
        res["real"].append(dict(step="R1", request=pick, **store.commit("real/main", v0["state_id"], c1, dec(pick))))
    h = store.refs["real/atomic"]
    r2 = store.commit("real/atomic", h, cand("real", "RY_PIN1")[0], dec("RY_PIN1"))
    res["real"].append(dict(step="R2", head_unchanged=r2["head_after"] == h, **r2))
    other = next(r for r in order_r if r != pick)
    res["real"].append(dict(step="R3", request=other, **store.commit("real/main", v0["state_id"], cand("real", other)[0], dec(other))))
    if store.refs["real/main"] != v0["state_id"]:
        rv, rrec = store.revert("real/main")
        res["real"].append(dict(step="R4", **rrec))
        C.states_file(out / "real_revert_states.json", {"real_revert": rv}, "w105")
        blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest",
                C.INPUTS["w105_rig_rest"][0], "--states", out / "real_revert_states.json", "--out", out / "real_revert_eval.json",
                log=out / "real_revert_eval.log")
        e = load(out / "real_revert_eval.json")["states"]["real_revert"]
        rev_eval["real"] = {"value_digest_equal_v0": e["value_digest"] == rev["states"]["v0"]["value_digest"],
                            "frames_bit_equal_v0": e["frames"] == rev["states"]["v0"]["frames"]}
    return {"programmes": res, "store_log": store.log, "refs": store.refs,
            "lineage_audit": {r: store.lineage(hh) for r, hh in store.refs.items()}, "revert_blender": rev_eval}


def cmd_assess(a):
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    g = guard()
    Bd = Path(a.build_dir)
    Bb = load(Bd / "build_report.json")
    rev, sev = load(Path(a.eval_dir) / "real_eval.json"), load(Path(a.eval_dir) / "syn_eval.json")
    ctx = C.Ctx(load(W116 / "results/declare/declaration.json"))
    pre = Path(decl["preflight_dir"])
    probe = load(pre / "syn_probe.json")
    srig = W7.syn_rig(probe)
    syn_b = load(W117 / "results/calibrate/calibration.json")["syn_bindings"]["bound"]
    w116_syn = load(W116 / "results/calibrate/calibration.json")["bindings"]["synthetic"]["bound"]
    states = {"real": load(Bd / "real_states.json")["states"], "syn": load(Bd / "syn_states.json")["states"]}
    full = {"real": {}, "syn": {}}

    def full_state(side, name):
        if name not in full[side]:
            st = load(pre / ("real_parent_state.json" if name == "v0" else f"syn_parent_{name}.json"))
            if st["state_id"] != states[side][name]["state_id"]:
                raise SystemExit(f"STOP: parent {name} differs from the build")
            full[side][name] = st
        return full[side][name]

    store = S.Store(out / "store")
    rep = {"schema": "w119_assess_v1", "declaration_sha256": a.declaration_sha, "label": "AGENT-RUN Blender evaluation", "requests": {}}
    t0 = time.time()
    for rid, rec in Bb["records"].items():
        if rec["side"] == "real":
            ev, rig, bnd, rest, scale, fr = rev, ctx.rig, ctx.real_b, rev["rest"], rev["rig"]["scale"][0], REAL_FRAMES
        else:
            ev, rig, bnd, rest, scale, fr = sev, srig, dict(syn_b) | w116_syn, sev["rest"], 1.0, SYN_FRAMES
        A = assess_record(rid, rec, ev, rig, bnd, rest, scale, fr, store, full_state, states, Bb)
        A.update(role=rec["role"], side=rec["side"], pin_frames=rec["pin_frames"], pin_runs=rec["pin_runs"],
                 validation_errors=rec["validation_errors"], correlation_groups=rec["correlation_groups"],
                 negative_of=rec.get("negative_of"), expect=rec.get("expect"), expect_reason=rec.get("expect_reason"))
        rep["requests"][rid] = A
    rep["assess_wall_s"] = time.time() - t0
    # display controls on Y
    dcs = {}
    for dc, (src, desc) in DISPLAY_Y.items():
        rec = Bb["records"][src]
        ev, fr = (rev, REAL_FRAMES) if rec["side"] == "real" else (sev, SYN_FRAMES)
        y = rep["requests"][src]["variants"]["Y"]
        base = y["display_spec_result"] if dc == "DY5" else y["display_spec_proposed"]
        s = corrupt_display(dc, base)
        if s is None:
            dcs[dc] = {"executed": False, "reason": "corruption not applicable"}
            continue
        Yf, Uf = C.ev_frames(ev, rec["Y"]), C.ev_frames(ev, rec["U"])
        chk = XB.check_display(s, Yf, Uf, fr, rec["pin_frames"], ev["states"][rec["Y"]]["state_id"],
                               decision=y["commit"]["decision"] if dc == "DY5" else None)
        dcs[dc] = {"executed": True, "source": src, "desc": desc, "check": chk}
    rep["display_controls"] = dcs
    rep["lineage"] = lineage(out, Bb, rep, full_state, states, rev, sev, pre)
    rep["source_guard"] = g.check()
    rep["manifests_after"] = {"w116": W7.w116_manifest_check(), "w117": W8.w117_manifest_check(),
                              "w118": manifest_check(W118, W118_MANIFEST_SHA)}
    rep["w118_decision_after"] = {"sha256": S.sha_file(W118 / "results/decision/decision.json"),
                                  "case": load(W118 / "results/decision/decision.json")["case"]}
    rep["parent_files_verify"] = {"real": S.verify_state(load(pre / "real_parent_state.json")),
                                  **{n: S.verify_state(load(pre / f"syn_parent_{n}.json")) for n in ("P__A", "P__B", "P__C", "P__D", "P__E", "P__G")}}
    rep["evaluated_states"] = {"real_missing": sorted(set(Bb["real_states"]) - set(rev["states"])),
                               "syn_missing": sorted(set(Bb["syn_states"]) - set(sev["states"]))}
    dump(out / "assess_report.json", rep)
    summ = {rid: [r["variants"]["Y"]["decision"]["decision"], r["variants"]["Y"]["decision"]["failure_class"], r["y_class"]]
            for rid, r in rep["requests"].items()}
    print(json.dumps({"y": summ, "source": rep["source_guard"]["unchanged"], "dc": {k: v.get("check", {}).get("ok") for k, v in dcs.items()}},
                     indent=0, default=str))
    return 0


# --------------------------------------------------------------------------------------------- render ---
COL = W8.COL
SERIES = {"U": "#8a8985", "F": "#2a78d6", "X": "#eb6834", "Y": "#1baf7a"}
VLABEL = {"U": "U   unpinned user edit", "F": "F_r W116 framewise (isolated)", "X": "X_r W118 offset layer (clean)",
          "Y": "Y_r W119 world-space guide", "TE": "TE  W117 easing (hist.)"}
GCOL = {"GUIDE_REACHED": (27, 175, 122), "GUIDE_FLEX_FLOOR_CLAMPED": (237, 161, 0), "GUIDE_UNREACHABLE_OUTER": (227, 73, 72),
        "GUIDE_UNREACHABLE_INNER": (227, 73, 72), "AMBIG": (98, 80, 214)}


def draw_timeline(path, rid, A):
    from PIL import Image, ImageDraw
    r = A["requests"][rid]
    vs = [v for v in ("F", "X", "TE", "Y") if v in r["variants"]]
    y_ = r["variants"]["Y"]
    spec0 = y_["display_spec_proposed"]
    frames = spec0["frames"]
    n = len(frames)
    cw = max(4, min(22, 1100 // n))
    L, top, rowh = 250, 100, 22
    chans = sorted(set(y_["footprint"]["channel_frames"]))
    H = top + (len(vs) + 6 + len(chans)) * (rowh + 4) + 3 * 68 + 120
    W = L + cw * n + 30
    im = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(im)
    ink, mute = (11, 11, 11), (82, 81, 78)
    d.text((10, 8), f"{rid}   [{r['side']}]   W119 edit-footprint timeline (actual Blender output vs U)", fill=ink)
    d.text((10, 24), f"hard-pin runs: {r['pin_runs']}   search budget: {len(y_['budget_frames'])} frames   planned X = planned Y: "
                     f"{r['variants']['X']['planned_frames'] == y_['planned_frames']}   Y actual runs: {y_['footprint']['runs']}", fill=mute)
    for i, v in enumerate(("X", "Y")):
        cm = r["variants"][v].get("commit")
        if cm:
            d.text((10, 40 + 16 * i), f"{v}_r state {cm['candidate_state_id'][:12]} -> {cm['decision']}ED  head after {cm['head_after'][:12]}"
                                      + ("" if cm["decision"] == "ACCEPT" else "  (candidate NOT applied)") +
                   f"  {r['variants'][v]['decision']['failure_class']} {r['variants'][v]['decision']['reasons'][:3]}",
                   fill=(180, 30, 30) if cm["decision"] != "ACCEPT" else ink)

    def x_of(i):
        return L + i * cw
    for i, f in enumerate(frames):
        if f % (10 if n > 60 else 5) == 0:
            d.line([(x_of(i), top - 6), (x_of(i), top - 1)], fill=mute)
            d.text((x_of(i) - 4, top - 20), str(f), fill=mute)
    d.text((10, top - 20), "Action frame index", fill=mute)
    y = top

    def row(label, cells, hatch=False, colmap=None):
        nonlocal y
        d.text((10, y + 5), label, fill=ink)
        for i, f in enumerate(frames):
            c = cells.get(str(f), "UNCHANGED")
            col = (colmap or COL).get(c, COL["UNCHANGED"])
            d.rectangle([x_of(i), y, x_of(i) + cw - 2, y + rowh], fill=col)
            if hatch and c != "UNCHANGED":
                for t in range(0, rowh, 5):
                    d.line([(x_of(i), y + t), (x_of(i) + cw - 2, y + min(rowh, t + cw))], fill=(90, 90, 90))
        y += rowh + 4
    pinset = set(r["pin_frames"])
    row("requested hard pin", {str(f): "PIN_CHANGED" if f in pinset else "UNCHANGED" for f in frames})
    row("search budget (hatched)", {str(f): ("PIN_UNCHANGED" if f in pinset else "AUTO_CHANGED") if f in set(y_["budget_frames"]) else "UNCHANGED"
                                    for f in frames}, hatch=True)
    row("planned domain X_r = Y_r", {str(f): ("PIN_UNCHANGED" if f in pinset else "AUTO_CHANGED") if f in set(y_["planned_frames"]) else "UNCHANGED"
                                     for f in frames})
    for v in vs:
        rej = v in ("X", "Y") and r["variants"][v].get("commit", {}).get("decision") != "ACCEPT"
        row("ACTUAL " + VLABEL[v] + (" [REJECTED]" if rej else ""), r["variants"][v]["display_spec_proposed"]["cells"], hatch=rej)
    for ch in chans:
        fs = set(y_["footprint"]["channel_frames"].get(ch, []))
        row(f"   Y_r channel {ch}", {str(f): ("PIN_CHANGED" if f in pinset else "AUTO_CHANGED") if f in fs else "UNCHANGED" for f in frames})
    yi = y_.get("y_info", {}).get("frames", {})
    gcells = {k: ("AMBIG" if "BEND_PLANE_AMBIGUOUS" in fi["flags"] else fi["status"]) for k, fi in yi.items()}
    row("   Y_r guide status", gcells, colmap=dict(GCOL, UNCHANGED=COL["UNCHANGED"]))
    d.text((10, y + 4), "wrist displacement vs U (mm)", fill=ink)
    y += 20
    bar_h = 60
    for v in [v for v in ("F", "X", "Y") if v in r["variants"]]:
        mg = r["variants"][v]["display_spec_proposed"]["magnitude"]
        mx = max([m["wrist_mm"] for m in mg.values()] + [1.0])
        d.text((10, y + bar_h // 2 - 6), f"{v}_r  (max {mx:.0f} mm)", fill=mute)
        d.line([(L, y + bar_h), (L + cw * n, y + bar_h)], fill=(200, 200, 200))
        for i, f in enumerate(frames):
            m = mg[str(f)]
            if m["wrist_mm"] > 0 or m["sub_resolution"]:
                h = max(2, int(bar_h * m["wrist_mm"] / mx))
                c = COL["PIN_CHANGED"] if f in pinset else COL["AUTO_CHANGED"]
                if m["sub_resolution"]:
                    c = (110, 110, 110)
                d.rectangle([x_of(i), y + bar_h - h, x_of(i) + cw - 2, y + bar_h], fill=c)
        y += bar_h + 8
    lx = 10
    for name, c in (("hard-pin frame (changed)", COL["PIN_CHANGED"]), ("pin frame unchanged", COL["PIN_UNCHANGED"]),
                    ("automatic out-of-pin edit", COL["AUTO_CHANGED"]), ("unchanged vs U", COL["UNCHANGED"]),
                    ("guide reached", GCOL["GUIDE_REACHED"]), ("flex-floor clamped", GCOL["GUIDE_FLEX_FLOOR_CLAMPED"]),
                    ("guide unreachable", GCOL["GUIDE_UNREACHABLE_OUTER"]), ("bend plane ambiguous", GCOL["AMBIG"])):
        d.rectangle([lx, y + 6, lx + 14, y + 18], fill=c)
        d.text((lx + 18, y + 6), name, fill=ink)
        lx += 18 + 7 * len(name) + 14
    d.text((10, y + 26), "Hatched = search budget / REJECTED (not applied). Grey bar = change below 1 mm display resolution (shown). "
                         "Time axis = Action frame index (source 50 fps / Action 30 fps not reconciled).", fill=mute)
    im.crop((0, 0, W, y + 46)).save(path)


def draw_charts(path_steps, path_flex, rid, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = A["requests"][rid]
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 5.4), dpi=110, sharex=True)
    qU = r["variants"]["F"]["quality"]
    pt0 = qU["per_transition"]
    ks = sorted(int(k) for k in pt0)
    for v in ("F", "X", "Y"):
        q = r["variants"][v].get("quality")
        if not q:
            continue
        pt = q["per_transition"]
        axs[0].plot(ks, [pt[str(k)]["wrist_comp_step_m"] * 1e3 for k in ks], color=SERIES[v], lw=2, label=VLABEL[v].strip())
        axs[1].plot(ks, [pt[str(k)]["wrist_step_m"] * 1e3 for k in ks], color=SERIES[v], lw=2, label=VLABEL[v].strip())
    axs[1].plot(ks, [pt0[str(k)]["control_wrist_step_m"] * 1e3 for k in ks], color=SERIES["U"], lw=1.5, ls="--", label=VLABEL["U"].strip())
    for ax in axs:
        for a_, b_ in r["pin_runs"]:
            ax.axvspan(a_ - 0.5, b_ + 0.5, color="#2a78d6", alpha=0.08, lw=0)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("compensation step\n(mm/frame, vs U)")
    axs[1].set_ylabel("raw wrist step\n(mm/frame)")
    axs[1].set_xlabel("Action frame index (transition k-1 -> k); shaded = hard pin")
    axs[0].set_title(f"{rid}: edit-induced (top) vs raw output (bottom) wrist motion", fontsize=9)
    axs[0].legend(fontsize=7, frameon=False)
    axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path_steps)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.5, 3.0), dpi=110)
    win = sorted(int(k) for k in qU["flex_deg"])
    ax.plot(win, [qU["control_flex_deg"][str(k)] for k in win], color=SERIES["U"], lw=1.5, ls="--", label=VLABEL["U"].strip())
    for v in ("F", "X", "Y"):
        q = r["variants"][v].get("quality")
        if q:
            ax.plot(win, [q["flex_deg"][str(k)] for k in win], color=SERIES[v], lw=2, label=VLABEL[v].strip())
    ax.axhline(TB.STRAIGHT_FLEX_DEG, color="#52514e", lw=0.8, ls=":")
    ax.text(win[0], TB.STRAIGHT_FLEX_DEG + 0.5, "5 deg W118 diagnostic reference (not a rule)", fontsize=7, color="#52514e")
    for a_, b_ in r["pin_runs"]:
        ax.axvspan(a_ - 0.5, b_ + 0.5, color="#2a78d6", alpha=0.08, lw=0)
    ax.set_ylabel("absolute elbow flex (deg)")
    ax.set_xlabel("Action frame index; shaded = hard pin")
    ax.set_title(f"{rid}: absolute elbow flex time series (Blender)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(path_flex)
    plt.close(fig)


def cmd_render(a):
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    Bb = load(Path(a.build_dir) / "build_report.json")
    A = load(Path(a.assess_dir) / "assess_report.json")
    (out / "timelines").mkdir()
    (out / "charts").mkdir()
    for rid, r in A["requests"].items():
        draw_timeline(out / "timelines" / f"timeline_{rid}.png", rid, A)
        if r["variants"]["Y"].get("quality") and r["variants"]["F"].get("quality"):
            draw_charts(out / "charts" / f"steps_{rid}.png", out / "charts" / f"flex_{rid}.png", rid, A)
    jobs = {"real": [], "syn": []}
    sel = []
    for side, rids in REPRESENTATIVE.items():
        for rid in rids:
            rec = Bb["records"][rid]
            lo, hi = A["requests"][rid]["variants"]["Y"]["quality"]["window"]
            frames = list(range(lo, hi + 1))
            sel.append((rid, frames))
            tgt = rec["request"]["constraints"][0]["target"]
            yi = rec["y_info"]["frames"]
            for v, st in (("U", rec["U"]), ("F", rec["F"]), ("X", rec["X"]), ("Y", rec["Y"])):
                for k in frames:
                    mk = []
                    tk = [c["target"] for c in rec["request"]["constraints"] if k in c["frames"]]
                    if tk:
                        mk.append({"name": "target", "position": tk[0], "color": (0.1, 0.9, 0.2, 1), "radius": 0.013})
                    if v == "Y" and str(k) in yi:
                        mk.append({"name": "guide", "position": yi[str(k)]["guide"], "color": (0.85, 0.2, 0.75, 1), "radius": 0.008})
                    jobs[side].append({"id": f"{rid}__{v}__f{k:03d}", "state": st, "frame_index": k, "markers": mk, "centre": tgt,
                                       "view_offset": [0.0, -3.0, 0.0]})
    dump(out / "real_jobs.json", {"jobs": jobs["real"]})
    dump(out / "syn_jobs.json", {"jobs": jobs["syn"]})
    b = Path(a.build_dir)
    blender("scripts/w116_blender_stages.py", "--", "render", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest",
            C.INPUTS["w105_rig_rest"][0], "--states", b / "real_states.json", "--jobs", out / "real_jobs.json", "--out-dir", out / "real",
            "--ortho-scale", "1.1", "--size", "300", "--hide-mesh", log=out / "real_render.log")
    blender("scripts/w116_blender_stages.py", "--", "render", "--rig", f"synthetic:{Path(decl['preflight_dir']) / 'syn_spec.json'}",
            "--states", b / "syn_states.json", "--jobs", out / "syn_jobs.json", "--out-dir", out / "syn", "--ortho-scale", "1.2",
            "--size", "300", "--hide-mesh", log=out / "syn_render.log")
    compose(out, sel, Bb, A)
    write_index(out, A, sel)
    dump(out / "render_selection.json", {"selection": sel, "review_scale": REVIEW_SCALE})
    return 0


def compose(out, sel, Bb, A):
    from PIL import Image, ImageDraw
    for rid, frames in sel:
        rec = Bb["records"][rid]
        d = out / rec["side"]
        variants = ["U", "F", "X", "Y"]
        ims = {(v, k): Image.open(d / f"{rid}__{v}__f{k:03d}.png").convert("RGB") for v in variants for k in frames}
        w, h = ims[("U", frames[0])].size
        cells = {v: (A["requests"][rid]["variants"][v]["display_spec_proposed"]["cells"] if v != "U" else {}) for v in variants}
        band, gap = 10, 12
        pin = set(rec["pin_frames"])

        def tile(v, k, scale=1):
            im = ims[(v, k)].copy()
            if scale != 1:
                im = im.resize((w // scale, h // scale))
            c = cells[v].get(str(k), "UNCHANGED") if v != "U" else "UNCHANGED"
            t = Image.new("RGB", (im.size[0], im.size[1] + band), COL[c])   # band BELOW the image of the same tile
            t.paste(im, (0, 0))
            return t
        cols = frames if len(frames) <= 14 else [frames[int(round(i * (len(frames) - 1) / 13))] for i in range(14)]
        sw, sh = w // 2, h // 2 + band
        sheet = Image.new("RGB", (190 + sw * len(cols), 46 + (sh + gap) * len(variants)), (252, 252, 251))
        dr = ImageDraw.Draw(sheet)
        dr.text((4, 4), f"{rid}: each row = one variant; the coloured band directly BELOW each tile is that tile's actual footprint "
                        f"class (blue pin, orange automatic edit, grey unchanged). green=pin target, magenta=Y guide", fill=(11, 11, 11))
        for j, k in enumerate(cols):
            dr.text((190 + j * sw + 4, 26), f"f{k}{' PIN' if k in pin else ''}", fill=(42, 120, 214) if k in pin else (82, 81, 78))
            for i, v in enumerate(variants):
                sheet.paste(tile(v, k, 2), (190 + j * sw, 46 + i * (sh + gap)))
        for i, v in enumerate(variants):
            dr.text((4, 46 + i * (sh + gap) + sh // 2 - 6), VLABEL[v], fill=(11, 11, 11))
        sheet.save(out / f"filmstrip_{rid}.png")
        gif = []
        for k in frames:
            fr = Image.new("RGB", (w * len(variants), h + band + 24), (252, 252, 251))
            dd = ImageDraw.Draw(fr)
            for i, v in enumerate(variants):
                fr.paste(tile(v, k), (i * w, 24))
                c = cells[v].get(str(k), "UNCHANGED") if v != "U" else ""
                dd.text((i * w + 6, 4), f"{v} f{k}{' PIN' if k in pin else ''} {c}", fill=(11, 11, 11))
            gif.append(fr)
        gif[0].save(out / f"anim_{rid}.gif", save_all=True, append_images=gif[1:], duration=150, loop=0)


def write_index(out, A, sel):
    rows = []
    for rid, r in A["requests"].items():
        y = r["variants"]["Y"]
        cm = y.get("commit", {})
        cmpy = r["comparisons"].get("Y_vs_X", {})
        rows.append(f"<tr><td>{rid}</td><td>{r['side']}</td><td>{r['role']}</td><td>{r['pin_runs']}</td><td>{y['footprint']['out_of_pin_runs']}</td>"
                    f"<td>{sorted(y['footprint']['channel_frames'])}</td><td><code>{cm.get('candidate_state_id', '')[:12]}</code></td>"
                    f"<td><b>{y['decision']['decision']}</b> {y['decision']['failure_class']}</td><td><code>{cm.get('head_after', '')[:12]}</code></td>"
                    f"<td>{r['y_class']}</td><td>{cmpy.get('B1_X_m', '')}</td><td>{cmpy.get('B1_Y_m', '')}</td>"
                    f"<td><a href='timelines/timeline_{rid}.png'>timeline</a></td></tr>")
    films = "".join(f"<h3>{rid}</h3><img src='timelines/timeline_{rid}.png'><br><img src='charts/steps_{rid}.png'><br>"
                    f"<img src='charts/flex_{rid}.png'><br><img src='filmstrip_{rid}.png'><br><img src='anim_{rid}.gif'>" for rid, _ in sel)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>W119 endpoint-space transition (scratch, not product UI)</title>
<style>body{{font-family:sans-serif;background:#fcfcfb;color:#0b0b0b;margin:20px}} td,th{{border-bottom:1px solid #ddd;padding:3px 6px;font-size:12px}}
img{{max-width:100%;margin:6px 0}}</style></head><body>
<h1>W119 Y_r vs clean X_r / F_r: actual edit footprint and rig motion (scratch diagnostic)</h1>
<p>Every timeline is drawn from the Blender-evaluated output diff vs U and independently checked against raw Blender frames.
Hatched = search budget or REJECTED (never applied). Time axis = Action frame index.</p>
<table><tr><th>request</th><th>side</th><th>role</th><th>hard-pin runs</th><th>Y actual out-of-pin runs</th><th>channels</th><th>proposed Y</th>
<th>commit</th><th>head after</th><th>Y class</th><th>B1 X (m)</th><th>B1 Y (m)</th><th></th></tr>{''.join(rows)}</table>
<h2>Representative requests (U / F_r / X_r / Y_r)</h2>{films}</body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------------------------- decide ---
def cmd_decide(a):
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    pf = load(Path(decl["preflight_dir"]) / "preflight.json")
    cal = load(Path(decl["calibration_dir"]) / "calibration.json")
    A = load(Path(a.assess_dir) / "assess_report.json")
    b1, b2, e1, e2 = Path(a.build_dir), Path(a.build_dir2), Path(a.eval_dir), Path(a.eval_dir2)
    Bb, B2 = load(b1 / "build_report.json"), load(b2 / "build_report.json")
    VR = load(a.visual_review)
    Rq = A["requests"]
    recs = Bb["records"]
    g, det = {}, {}
    g["V1_authentication"] = not pf["stop"]
    g["V2_parent_reproduction"] = pf["parent_rebuild"]["canonical_equal"] and pf["parent_blender"]["equals_w105_fixed_digest"] and \
        pf["parent_blender"]["frames_bit_equal_w118_preflight"] and all(pf["syn_probe_parity_w118"].values()) and pf["syn_rest_equal_w118"]
    valid = [r for r, x in recs.items() if not x.get("contract_invalid") and x["role"] != "negative"]
    xeq = [r for r in valid if not recs[r]["X_r_equals_w118_X"]]
    det["V3"] = {"cal": cal["w118_reproduction"]["ok"], "b1": Bb["w118_reproduction"]["ok"], "b2": B2["w118_reproduction"]["ok"],
                 "X_r_not_equal_w118_X": xeq}
    g["V3_baseline_reproduction"] = det["V3"]["cal"] and det["V3"]["b1"] and det["V3"]["b2"] and not xeq
    cr = cal["contamination_reproduction"]
    det["V4"] = {k: cr[k] for k in ("reproduced_equals_w118_V4_list", "quality_window_equals_w118_worklog_six", "all_out_of_pin_on_siblings",
                                    "reproduced_contaminated_requests", "quality_window_contaminated")} | {"w118_decision_after": A["w118_decision_after"]}
    g["V4_contamination_reproduced_not_repaired"] = cr["reproduced_equals_w118_V4_list"] and cr["quality_window_equals_w118_worklog_six"] and \
        cr["all_out_of_pin_on_siblings"] and A["w118_decision_after"]["sha256"] == W118_DECISION_SHA and A["w118_decision_after"]["case"] == "D"
    iso_bad = [r for r in valid if not (recs[r]["isolation_values_F_r"]["ok"] and Rq[r]["variants"]["F"]["isolation_blender"]["ok"])]
    eq_bad = [r for r in valid if not (recs[r]["F_r_equals_frozen_single_frame_resolver"]["values_bit_equal"]
                                       and recs[r]["F_r_equals_frozen_single_frame_resolver"]["statuses_equal"])]
    det["V5"] = {"isolation_failed": iso_bad, "resolver_equality_failed": eq_bad, "frozen_mismatch_b1": Bb["frozen_controls_mismatch"],
                 "frozen_mismatch_b2": B2["frozen_controls_mismatch"],
                 "contaminated_windows_rebuilt": {r: {"F_r_blender_outside_pin": Rq[r]["variants"]["F"]["isolation_blender"]["changed_outside_pin"],
                                                      "F_shared_hist_outside_pin_runs": Rq[r]["variants"].get("F_shared_hist", {}).get(
                                                          "isolation_blender", {}).get("runs_outside_pin")} for r in CONTAMINATED_W118_WINDOWS}}
    g["V5_clean_isolated_baselines"] = not iso_bad and not eq_bad and not Bb["frozen_controls_mismatch"] and not B2["frozen_controls_mismatch"] \
        and cal["baselines_ok"]["F_r_isolated_value_level"]
    unfair = [r for r in valid if not (Rq[r]["variants"]["Y"].get("fairness") or {}).get("ok")]
    det["V6"] = unfair
    g["V6_ab_fairness"] = not unfair
    g["V7_predeclaration"] = decl["outcomes_inspected"] is False and S.sha_file(Path(decl["calibration_dir"]) / "frozen_controls.json") == \
        decl["frozen_controls_sha256"] and decl["hg_design"] == cal["hg_design"]["chosen"] and A["declaration_sha256"] == a.declaration_sha
    det["V8"] = {"missing": A["evaluated_states"], "syn_fk": cal["syn_fk_max_error_m"]}
    g["V8_independent_blender_evaluation"] = not A["evaluated_states"]["real_missing"] and not A["evaluated_states"]["syn_missing"] and \
        cal["syn_fk_max_error_m"] <= 1e-5
    miss = []
    for rid in decl["population"]["all_requests"]:
        if rid not in Rq:
            miss.append(rid)
            continue
        for v in ("F", "X", "Y"):
            if "display_spec_proposed" not in Rq[rid]["variants"].get(v, {}):
                miss.append(f"{rid}/{v}")
        if not recs[rid].get("contract_invalid"):
            for v in ("X", "Y"):
                if Rq[rid]["variants"][v]["residual_m"]["n"] < len(recs[rid]["pin_frames"]):
                    miss.append(f"{rid}/{v} pins")
    det["V9"] = miss
    g["V9_population_complete"] = not miss and all(r in Rq for r in CONTAMINATED_W118_WINDOWS)
    same_states = all((b1 / f).read_bytes() == (b2 / f).read_bytes() for f in ("real_states.json", "syn_states.json"))
    strip = lambda yi: {k: v for k, v in yi.items() if k != "wall_s"}  # noqa: E731
    same_y = all(strip(Bb["records"][r].get("y_info", {})) == strip(B2["records"][r].get("y_info", {})) for r in recs)
    same_ev = all(load(e1 / f)["states"] == load(e2 / f)["states"] for f in ("real_eval.json", "syn_eval.json"))
    det["V10"] = {"states_identical": same_states, "y_infos_identical": same_y, "evaluations_identical": same_ev}
    g["V10_repeatability"] = same_states and same_y and same_ev
    tm = [pf["timing"], pf["syn_timing"]]
    g["V11_timing_semantics"] = all(st["interpolations"] == ["CONSTANT"] and st["blender_frame_equals_index_minus_f0_plus_1"]
                                    and all(s["half_equals_frame"] and s["late_equals_frame"] for s in st["subframe"].values())
                                    for t in tm for st in t["states"].values()) and "P__G" in pf["syn_timing"]["states"]
    # ---- C
    c1 = []
    for rid, r in Rq.items():
        for v in ("X", "Y"):
            x = r["variants"][v]
            if x["decision"]["decision"] != "ACCEPT" or (r["role"] == "negative" and v == "X"):
                continue
            fp = x["footprint"]
            ok = x["residual_m"]["n_over_tol"] == 0 and set(fp["channel_frames"]) <= set(recs[rid]["auth"]) and \
                set(fp["frames_changed"]) <= set(x["budget_frames"]) and set(fp["frames_changed"]) <= set(x["planned_frames"]) and \
                not fp["outside_arm_points_changed"] and not fp["matrix_world_changed_frames"] and all(x["frame_pass"].values())
            if not ok:
                c1.append(f"{rid}/{v}")
    det["C1"] = c1
    g["C1_hard_pin_and_scope"] = not c1
    c2 = []
    for rid, r in Rq.items():
        for v, c in r["variants"].items():
            if not c["display_check_proposed"]["ok"] and r["role"] != "negative":
                c2.append(f"{rid}/{v}/proposed")
        if r["role"] == "negative":
            continue
        y = r["variants"]["Y"]
        if not set(y["planned_frames"]) <= set(y["budget_frames"]) or (not recs[rid].get("contract_invalid")
                                                                        and y["planned_frames"] != r["variants"]["X"]["planned_frames"]):
            c2.append(f"{rid}/Y_planned")
        for v in ("X", "Y"):
            cm = r["variants"][v].get("commit")
            if not cm or not (r["variants"][v]["display_check_result"]["ok"] and cm["preview_id_equals_candidate"] and cm["head_consistent"]):
                c2.append(f"{rid}/{v}/result")
    det["C2"] = c2
    g["C2_truthful_footprint"] = not c2
    wrong = []
    for rid, (src, desc, want, reason) in NEG_Y.items():
        d = Rq[rid]["variants"]["Y"]["decision"]
        ok = [d["decision"], d["failure_class"]] == want and (reason is None or reason in d["reasons"])
        if rid == "NY_GUIDE_UNREACH":
            yi = Bb["records"][rid]["y_info"]
            ok = ok and yi["status_counts"]["GUIDE_UNREACHABLE_OUTER"] == len(yi["frames"]) > 0
        if rid == "NY_AMBIG":
            yi = Bb["records"][rid]["y_info"]
            ok = ok and yi["flag_counts"]["BEND_PLANE_AMBIGUOUS"] == len(yi["frames"]) > 0
        if rid in ("NB_SHARED", "NB_SIBLING"):
            ok = ok and not Bb["records"][rid]["isolation_values_F_r"]["ok"] and not Rq[rid]["variants"]["F"]["isolation_blender"]["ok"]
        if not ok:
            wrong.append({"req": rid, "got": d, "expected": [want, reason]})
    for dc, r in A["display_controls"].items():
        if not r.get("executed") or r["check"]["ok"]:
            wrong.append({"req": dc, "got": r})
    det["C3"] = wrong
    g["C3_negative_controls"] = not wrong
    atom = []
    for rid, r in Rq.items():
        y = r["variants"]["Y"]
        if r["validation_errors"] or not y["frame_pass"] or r["role"] == "negative":
            continue
        fp = y["footprint"]
        allpass = all(y["frame_pass"].values()) and set(recs[rid]["F_statuses"].values()) <= {"SOLVED", "NO_CHANGE_NEEDED", "NOT_APPLICABLE"} \
            and set(fp["frames_changed"]) <= set(y["planned_frames"]) and set(fp["channel_frames"]) <= set(recs[rid]["auth"]) and \
            recs[rid]["y_info"]["request_status"] == "SOLVED" and (y.get("fairness") or {}).get("ok") and r["variants"]["F"]["isolation_blender"]["ok"]
        if (y["decision"]["decision"] == "ACCEPT") != bool(allpass):
            atom.append(rid)
    for n in ("NY_PIN1", "RY_PIN1"):
        y = Rq[n]["variants"]["Y"]
        if not (y["decision"]["decision"] == "REJECT" and sum(1 for v in y["frame_pass"].values() if not v) == 1):
            atom.append(f"{n}: single-corrupted-frame")
    lin = A["lineage"]
    refs_ok = all(e.get("head_after") == e.get("head_before") for e in lin["store_log"] if e.get("op") == "commit" and e.get("decision") == "REJECT")
    steps = {e["step"]: e for side in lin["programmes"].values() for e in side}
    stale = steps.get("S3", {}).get("reasons") == ["STALE_PARENT"] and steps.get("R3", {}).get("reasons") == ["STALE_PARENT"]
    acc = steps.get("S1", {}).get("decision") == "ACCEPT" and steps.get("R1", {}).get("decision") == "ACCEPT"
    rv = lin["revert_blender"]
    revert_ok = bool(rv.get("real") and rv.get("synthetic")) and all(all(v for k, v in x.items() if k != "restores") for x in rv.values()) \
        and steps.get("S4", {}).get("values_equal_restored") is True and steps.get("R4", {}).get("values_equal_restored") is True
    audit_ok = all(all(e["verifies"] for e in ch) for ch in lin["lineage_audit"].values())
    det["C4"] = {"atomicity": atom, "refs_unchanged_on_reject": refs_ok, "stale": stale, "accept_executed": acc, "revert_ok": revert_ok,
                 "audit_ok": audit_ok, "atomic_heads_unchanged": [steps.get(s, {}).get("head_unchanged") for s in ("S2", "R2")]}
    g["C4_atomicity_lineage_reversal"] = not atom and refs_ok and stale and acc and revert_ok and audit_ok and all(det["C4"]["atomic_heads_unchanged"])
    src_ok = A["source_guard"]["unchanged"] and all(m["ok"] for m in A["manifests_after"].values()) and all(A["parent_files_verify"].values()) \
        and Bb["source_guard"]["unchanged"] and B2["source_guard"]["unchanged"]
    g["C5_source_immutability"] = src_ok
    newfail = [rid for rid in valid if Rq[rid]["role"] in ("evaluation", "development", "control", "noop")
               and Rq[rid]["variants"]["X"]["decision"]["decision"] == "ACCEPT" and Rq[rid]["variants"]["Y"]["decision"]["decision"] != "ACCEPT"]
    det["C6"] = newfail
    g["C6_no_new_hard_failure"] = not newfail
    c7 = []
    for rid, r in Rq.items():
        y = r["variants"]["Y"]
        if y["decision"]["decision"] != "ACCEPT" or r["role"] == "negative":
            continue
        col = y.get("collateral", {})
        cz = all(v["max_vs_U_m"] == 0.0 for k, v in col.items() if isinstance(v, dict))
        if not (y["user_edit_and_object_equal_U"] and cz and col.get("root_matrix_equal_U", True)):
            c7.append(rid)
    det["C7"] = c7
    g["C7_user_edit_and_collateral"] = not c7
    # ---- U
    cmpy = {rid: r["comparisons"]["Y_vs_X"] for rid, r in Rq.items() if r["role"] in ("evaluation",) and "Y_vs_X" in r["comparisons"]}
    mi = {rid for rid, c in cmpy.items() if c["meaningful_improvement"]}
    evalset = set(SYN_PRIMARY) | {rid for rid, r in Rq.items() if r["side"] == "real" and r["role"] == "evaluation"}
    det["U"] = {"comparisons": cmpy, "meaningful_improvement": sorted(mi), "any_in_evaluation_set": sorted(mi & evalset)}
    g["U_syn"] = SYN_HELD_OUT in mi
    g["U_real"] = bool(mi & set(RH1_FAMILY)) and bool(mi & set(RH2_INDEPENDENT))
    vr = VR.get("requests", {})
    okv = [rid for rid in mi if Rq[rid]["side"] == "real" and vr.get(rid, {}).get("improvement_visible") is True]
    det["U_vis"] = {"qualifying": okv, "review_sha256": S.sha_file(a.visual_review)}
    g["U_vis"] = bool(okv)
    V = [k for k in g if k.startswith("V") and not g[k]]
    Cg = [k for k in g if k[:2] in ("C1", "C2", "C3", "C4", "C5", "C6", "C7") and not g[k]]
    if V:
        case = "D"
    elif Cg or not (mi & evalset):
        case = "C"
    elif g["U_syn"] and g["U_real"] and g["U_vis"]:
        case = "A"
    else:
        case = "B"
    res = {"schema": "w119_decision_v1", "declaration_sha256": a.declaration_sha, "visual_review_sha256": S.sha_file(a.visual_review),
           "gates": g, "failed": {"V": V, "C": Cg}, "case": case, "detail": det}
    dump(out / "decision.json", res)
    print(json.dumps({"case": case, "gates": g, "failed": res["failed"], "MI": sorted(mi)}, indent=1))
    return 0


# --------------------------------------------------------------------------------------------- manifest ---
def cmd_manifest(a):
    base = Path(a.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != "evidence_manifest.json" and not ({"store", "lineage_store"} & set(p.relative_to(base).parts)):
            files[p.relative_to(base).as_posix()] = {"sha256": S.sha_file(p), "bytes": p.stat().st_size}
    dump(base / "evidence_manifest.json", {"schema": "w119_evidence_manifest_v1", "declaration_sha256": a.declaration_sha,
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
                        ("render", ("build_dir", "assess_dir")),
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
