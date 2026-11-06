#!/usr/bin/env python3
"""W120: matched two-factor attribution of the W119 transition compensation (world-space guide G vs extra flex floor F)
on isolated per-request baselines and actual Blender output (evidence only, NON-PRODUCTION).

Per request, on the same parent, user edit U, rig, hard pin, authorised DOF, per-side margins and search budget:
  U_r  unpinned user edit (control);  F_r  frozen W116 framewise resolver on this request's pins only (isolated)
  X_r  frozen W118 offset-layer editor (external unchanged baseline)
  G0F0 / G0F1 / G1F0 / G1F1  matched factor variants (w120_factor_attribution_base); G1F1 = the W119 Y formulation
  W119 Y (historical, read-only) for context.

Stages (each refuses an existing output directory):
  preflight  identity, W113-W119 Worklogs/markers, W116-W119 manifests/declarations/decisions/CASEs, inputs, parent rebuild +
             fresh Blender evaluation, synthetic parents incl. G, timing probes
  calibrate  W118/W119 baseline reproduction (F_r/X_r ids, margins, budgets = W119 frozen controls), read-only W118
             contamination reproduction, outcome-independent instrument preflight (Python-only fixtures: zero budget,
             unequal budget, precedence table exhaustive check, combined errors), negative-control expectation
             consistency, G1F1 parity digests vs W119 Y.  No factor quality is computed.
  declare    frozen declaration -> hash
  build      four factor variants per request + negative controls
  evaluate   Blender evaluation (scripts/w116_blender_stages.py, unchanged)
  assess     footprints, isolation, fairness, decisions, quality, factorial contrasts, guide residuals, G0F0 vs X_r,
             G1F1 vs W119 Y, negative/display controls, lineage/revert, source guards
  render     timelines, step/flex charts, filmstrips + GIFs (7 variants)
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
import w116_semantic_pin_base as S  # noqa: E402
import w116_semantic_pin_contract as C  # noqa: E402
import w117_temporal_pin_base as TB  # noqa: E402
import w117_temporal_pin as W7  # noqa: E402
import w118_transition_edit_base as XB  # noqa: E402
import w118_transition_edit as W8  # noqa: E402
import w119_endpoint_transition_base as YB  # noqa: E402
import w119_endpoint_transition as W9  # noqa: E402
import w120_factor_attribution_base as FB  # noqa: E402

BATCH = 120
INSTRUCTION_BLOB = "7c2630ed97875c410f0a8105c13a1660c07c514e"
STARTING_HEAD = "2d2fb07a6b66ad87540db228335a4665e9aba7bf"
W119_COMPLETE_COMMIT = "e0ccf74acb578db0f643571e4edd8f4f2b2ee36f"
W119_START_COMMIT = "78809d3e641e322b62cb7f64e5b834a063b403a3"
W119_INSTRUCTION_BLOB = "5f97c3a23f066da0081f392a4234e4b6413620bb"
BRANCH = "arch/single_frame_first"
LOCK = W8.LOCK
W116, W117, W118, W119 = W7.W116, W8.W117, W9.W118, Path("C:/Users/dna10/animcv-w119")
W119_MANIFEST_SHA = "5e8f8f70779a5fe3da08274ebee456e2ea36096d38606769217d9786d4684735"
W119_DECL_SHA = "6616d1bdf227f97cda032e56629a4366e0188af1dfc310f5185049742bba9e4b"
W119_DECISION_SHA = "ef8c1c4c8c9505d429687cb2779962a3e4fca098b140c77170b19fc707099dc4"
W119_FROZEN_CONTROLS_SHA = "3c209df62efe853383ead2cd4c22463e71cc3c31b63e3a9c7168a3a23e700669"
HIST = dict(W9.HIST) | {119: "docs/119_WORKLOG_ENDPOINT_SPACE_TRANSITION_ISOLATED_BASELINES.md"}
UPPER, LOWER = W7.UPPER, W7.LOWER
REAL_FRAMES, SYN_FRAMES = W8.REAL_FRAMES, W8.SYN_FRAMES
FACTORS = FB.FACTORS
CODE_FILES = ["scripts/w116_semantic_pin_base.py", "scripts/w116_semantic_pin_contract.py", "scripts/w116_blender_stages.py",
              "scripts/w117_temporal_pin_base.py", "scripts/w117_temporal_pin.py", "scripts/w118_transition_edit_base.py",
              "scripts/w118_transition_edit.py", "scripts/w119_endpoint_transition_base.py", "scripts/w119_endpoint_transition.py",
              "scripts/w120_factor_attribution_base.py", "scripts/w120_factor_attribution.py"]

# ------------------------------------------------------------------------------------------ population ---
SYN_PRIMARY = W9.SYN_PRIMARY            # H_A H_B H_C H_E H_S0 H_END DJ_B S_PF_C_0_29 W_PF_C H_G (W119 set, unchanged)
SYN_HELD_OUT = "H_G"
NEAR_SINGULAR = "H_C"
RH1 = ["H_RH1_none", "H_RH1_RBneg", "H_RH1_RBpos"]           # RH1 release family (correlated variants of one window)
RH2 = ["H_RH2_RBpos", "H_RH2_none", "H_RH2_RBneg"]           # RH2 release family (none/RBneg are infeasible pins)
CORRELATION = W9.CORRELATION
FAMILY_DISTINCT = {"RH1": RH1, "RH2": RH2}                  # DJ_R correlates with BOTH and never counts as distinct evidence
CONTAMINATED = W9.CONTAMINATED_W118_WINDOWS
LOW_EVIDENCE = ["S_RA_pos_109_118", "S_RB_pos_109_118", "S_RA_pos_154_182", "S_RB_pos_154_182"]
REVIEW_SET = {"H_RH1_none": "RH1 release near frame 122", "H_RH2_RBpos": "RH2 release near frame 182",
              "H_RH1_RBneg": "RH1_RBneg bend-flip counterexample", "H_G": "held-out synthetic", "H_C": "near-singular synthetic",
              "S_RB_pos_109_118": "previously contaminated window whose clean comparator changed the interpretation (W118 artefact)",
              "S_RB_pos_86_106": "previously contaminated window with a real transition", "S_RA_pos_194_198": "no transition exists"}
VARIANTS_SHOWN = ["U", "F", "X"] + list(FACTORS)
VLABEL = {"U": "U    unpinned user edit", "F": "F_r  W116 framewise (isolated)", "X": "X_r  W118 offset layer (external)",
          "G0F0": "G0F0 X-wrist target, no extra floor", "G0F1": "G0F1 X-wrist target + W119 floor",
          "G1F0": "G1F0 W119 guide, no extra floor", "G1F1": "G1F1 W119 guide + floor (= W119 Y form)"}

# rid: (source request, factor, description, injected conditions, expected [decision, class], required reasons)
CV, SF, INF = ["REJECT", "CONTRACT_VIOLATION"], ["REJECT", "SOLVER_FAILURE"], ["REJECT", "INFEASIBLE_REQUEST"]
NEG = {
    "NG_BUDGET": ("Z_A", "G1F1", "zero-budget request; factor budget built with the global M_MAX=8 (W119 Z_A defect reproduced)",
                  {"fairness_bad": True}, CV, ["AB_FAIRNESS:SEARCH_BUDGET_MISMATCH:Y"]),
    "NG_MARGIN": ("H_A", "G0F0", "release margin = X_r margin - 1 (mismatched margin)", {"fairness_bad": True}, CV,
                  ["AB_FAIRNESS:MARGIN_BUDGET_MISMATCH:Y"]),
    "NG_PARENT": ("H_A", "G1F0", "variant record claims another source parent state (P__B)", {"fairness_bad": True}, CV,
                  ["AB_FAIRNESS:PARENT_MISMATCH:Y"]),
    "NG_U": ("H_RH1_RBpos", "G0F1", "variant solved and stored on the RB_neg user edit (wrong U)",
             {"fairness_bad": True, "eval_scope": True, "fp_budget": True, "fp_channel": True, "fp_outside_arm": True}, CV,
             ["AB_FAIRNESS:U_MISMATCH:Y"]),
    "NG_RIG": ("H_A", "G1F1", "variant state claims another rig identity", {"fairness_bad": True}, CV,
               ["AB_FAIRNESS:RIG_IDENTITY_MISMATCH:Y"]),
    "NG_CDIGEST": ("H_B", "G0F0", "variant record carries another pin-constraint digest", {"fairness_bad": True}, CV,
                   ["AB_FAIRNESS:PIN_CONSTRAINT_MISMATCH:Y"]),
    "NG_AUTH": ("H_B", "G1F0", "variant record claims a narrower authorisation [upper]", {"fairness_bad": True}, CV,
                ["AB_FAIRNESS:AUTH_MISMATCH:Y"]),
    "NG_PIN_COMBINED": ("H_A", "G0F1", "lower +2 deg on pin frame 20: wrong pin value AND failed output pin",
                        {"fairness_bad": True, "eval_pin": True}, CV, ["EVALUATED_PIN_UNSATISFIED", "AB_FAIRNESS:PIN_VALUES_DIFFER_FROM_F:Y"]),
    "RG_PIN_COMBINED": ("H_RH1_none", "G1F0", "lowerarm_l +2 deg on pin frame 113 (real): wrong pin value AND failed output pin",
                        {"fairness_bad": True, "eval_pin": True}, CV, ["EVALUATED_PIN_UNSATISFIED", "AB_FAIRNESS:PIN_VALUES_DIFFER_FROM_F:Y"]),
    "NG_FALSE": ("H_A", "G1F1", "U on every pin frame while the resolver claims SOLVED (corrupted hard pin)",
                 {"fairness_bad": True, "eval_pin": True}, CV, ["EVALUATED_PIN_UNSATISFIED"]),
    "NG_NONCONV": ("H_A", "G0F0", "fault-injected skipped forearm aim; fairness otherwise valid (isolated nonconvergence)",
                   {"nonconv": True}, SF, ["NON_CONVERGED"]),
    "NG_NONCONV_PIN": ("H_A", "G1F0", "nonconvergence + lower +2 deg on pin frame 20 (combined)",
                       {"nonconv": True, "fairness_bad": True, "eval_pin": True}, SF,
                       ["NON_CONVERGED", "EVALUATED_PIN_UNSATISFIED", "AB_FAIRNESS:PIN_VALUES_DIFFER_FROM_F:Y"]),
    "NG_NONCONV_MARGIN": ("H_B", "G0F1", "nonconvergence + release margin - 1 (combined)", {"nonconv": True, "fairness_bad": True}, CV,
                          ["NON_CONVERGED", "AB_FAIRNESS:MARGIN_BUDGET_MISMATCH:Y"]),
    "NG_NONCONV_BONE": ("H_A", "G1F1", "nonconvergence + base +1 deg on a transition frame (combined; E == {NON_CONVERGED})",
                        {"nonconv": True, "eval_scope": True, "fp_channel": True, "fp_outside_arm": True}, SF,
                        ["NON_CONVERGED", "UNAUTHORIZED_CHANNEL_CHANGE"]),
    "NG_BONE": ("H_A", "G1F0", "base (unauthorised) +1 deg on a transition frame",
                {"eval_scope": True, "fp_channel": True, "fp_outside_arm": True}, CV, ["UNAUTHORIZED_CHANNEL_CHANGE"]),
    "RG_BONE": ("H_RH1_none", "G0F1", "spine_01 +1 deg on a transition frame (real)",
                {"eval_scope": True, "fp_channel": True, "fp_outside_arm": True}, CV, ["UNAUTHORIZED_CHANNEL_CHANGE"]),
    "NG_ROOT": ("H_A", "G0F0", "__object__ +1 deg on a transition frame",
                {"eval_root": True, "eval_scope": True, "fp_root": True, "fp_channel": True, "fp_outside_arm": True}, CV, ["ROOT_MOVED"]),
    "NG_OUTSIDE": ("H_A", "G1F0", "upper +1 deg on frame 2 (outside budget and planned domain)", {"fp_budget": True, "fp_domain": True},
                   CV, ["FOOTPRINT_EXCEEDS_DECLARED_DOMAIN", "CHANGE_OUTSIDE_AUTOMATIC_BUDGET"]),
    "NG_PLANNED": ("H_A", "G0F1", "values unchanged; planned domain omits its outermost transition frame", {"fp_domain": True}, CV,
                   ["FOOTPRINT_EXCEEDS_DECLARED_DOMAIN"]),
    "NG_DISPLAY": ("H_B", "G1F1", "false display footprint: first automatic change relabelled UNCHANGED before the decision",
                   {"display_bad": True}, CV, ["DISPLAY_FOOTPRINT_MISMATCH"]),
    "NG_GUIDE_UNREACH": ("H_B", "G1F1", "guide pushed 0.05 m beyond full reach on every transition frame (feasible hard pin)",
                         {}, ["ACCEPT", "NONE"], []),
    "NG_AMBIG": ("H_B", "G0F0", "pole reference on the shoulder->target axis on every transition frame", {}, ["ACCEPT", "NONE"], []),
    "NG_FLIP": ("H_B", "G1F1", "pole reference mirrored across the shoulder->target axis on alternate transition frames (forced flips)",
                {}, ["ACCEPT", "NONE"], []),
    "NG_INFEASIBLE": ("H_Cx", "G1F0", "variant on the infeasible hard-pin request", {"pin_status": "UNREACHABLE"}, INF, []),
    "NG_INFEASIBLE_U": ("H_Cx", "G0F0", "infeasible hard pin + record claims another U (combined)",
                        {"pin_status": "UNREACHABLE", "fairness_bad": True}, INF, ["AB_FAIRNESS:U_MISMATCH:Y"]),
    "NG_CONFLICT": ("N6", "G0F1", "contradictory hard constraints", {"validation": "INFEASIBLE_REQUEST"}, INF,
                    ["CONFLICTING_HARD_CONSTRAINTS"]),
    "NG_DOF": ("N3", "G1F0", "forearm-only authorisation (missing upper-arm DOF)", {"missing_dof": True}, INF, ["Y_MISSING_AUTHORIZED_DOF"]),
    "NG_SHARED": ("S_RB_pos_86_106", "G0F0", "family-shared SEG_RB_pos__F used as F_r", {"isolation_bad": True}, CV,
                  ["BASELINE_NOT_ISOLATED"]),
    "NG_SIBLING": ("S_RA_pos_60_61", "G1F1", "F_r plus ONE sibling-segment pin frame (63)", {"isolation_bad": True}, CV,
                   ["BASELINE_NOT_ISOLATED"]),
}
DISPLAY_NEG = {"DG1": ("H_A", "G0F0", "DY1", "first changed out-of-pin frame relabelled UNCHANGED"),
               "DG2": ("H_RH1_none", "G1F1", "DY2", "one changed frame omitted from every channel row (real)"),
               "DG3": ("DJ_B", "G1F0", "DY3", "disjoint changed runs collapsed to the union envelope"),
               "DG4": ("NG_OUTSIDE", "G1F0", "DY4", "tampered output drawn from the declared planned domain"),
               "DG5": ("NG_INFEASIBLE", "G1F0", "DY5", "REJECTed variant displayed as ACCEPTED")}

dump, load, new_dir, git, blender = C.dump, C.load, C.new_dir, C.git, C.blender
all_vals, vals_of, rot_local, vlist = W8.all_vals, W8.vals_of, W8.rot_local, W9.vlist


def code_blobs():
    return {f: git("hash-object", str(ROOT / f)) for f in CODE_FILES}


def guard():
    g = W9.guard()
    files = dict(g.files) | {f"w119_{n}": W119 / "results" / p for n, p in (
        ("declaration", "declare/declaration.json"), ("decision", "decision/decision.json"), ("frozen_controls", "calibrate/frozen_controls.json"),
        ("calibration", "calibrate/calibration.json"), ("preflight", "preflight/preflight.json"), ("build_r1_report", "build_r1/build_report.json"),
        ("real_states", "build_r1/real_states.json"), ("syn_states", "build_r1/syn_states.json"), ("eval_e1_real", "eval_e1/real_eval.json"),
        ("eval_e1_syn", "eval_e1/syn_eval.json"), ("assess", "assess/assess_report.json"), ("visual_review", "review/visual_review.json"))}
    files["w119_manifest"] = W119 / "evidence_manifest.json"
    return S.SourceGuard(files)


# ------------------------------------------------------------------------------------------------- preflight ---
def cmd_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"schema": "w120_preflight_v1", "stop": [], "runtime": C.runtime(), "label": "AGENT-RUN (Main-local)"}
    rep["head"], rep["instruction_blob"] = git("rev-parse", "HEAD"), git("rev-parse", "HEAD:docs/next_batch.md")
    rep["branch"], rep["status_porcelain"] = git("rev-parse", "--abbrev-ref", "HEAD"), git("status", "--porcelain")
    title = (ROOT / "docs/next_batch.md").read_text(encoding="utf-8").splitlines()[0]
    rep["instruction_title"] = title
    if rep["instruction_blob"] != INSTRUCTION_BLOB or not title.startswith("# BATCH 120"):
        rep["stop"].append("instruction is not the bound Batch 120 blob")
    if rep["head"] != STARTING_HEAD:
        rep["stop"].append("HEAD is not the bound starting commit")
    if rep["branch"] != BRANCH:
        rep["stop"].append("wrong branch")
    lock = load(LOCK) if LOCK.exists() else None
    rep["main_local_lock"] = lock
    if not lock or lock.get("batch") != BATCH or lock.get("instruction_blob") != INSTRUCTION_BLOB or lock.get("start_commit") != STARTING_HEAD:
        rep["stop"].append("Main-local lock is not this Batch 120 claim")
    hist = {}
    for n, wl in HIST.items():
        lines = (ROOT / wl).read_text(encoding="utf-8").rstrip().splitlines()
        hist[n] = {"marker": lines[-2:] == ["AUTOMATION_BATCH_STATUS: COMPLETE", f"BATCH_NUMBER: {n}"],
                   "blob_at_start": git("rev-parse", f"{STARTING_HEAD}:{wl}"), "blob_now": git("hash-object", str(ROOT / wl))}
        if not hist[n]["marker"] or hist[n]["blob_at_start"] != hist[n]["blob_now"]:
            rep["stop"].append(f"W{n} Worklog record not intact")
    rep["historical_worklogs"] = hist
    rep["no_other_w120_worklog"] = not list((ROOT / "docs").glob("120_*"))
    if not rep["no_other_w120_worklog"]:
        rep["stop"].append("a Batch 120 Worklog already exists")
    try:
        git("merge-base", "--is-ancestor", W119_COMPLETE_COMMIT, "HEAD")
        rep["w119_complete_commit_is_ancestor"] = True
    except Exception:
        rep["w119_complete_commit_is_ancestor"] = False
        rep["stop"].append("W119 COMPLETE commit is not an ancestor of HEAD")
    rep["w119_instruction_blob_at_its_start"] = git("rev-parse", f"{W119_START_COMMIT}:docs/next_batch.md")
    if rep["w119_instruction_blob_at_its_start"] != W119_INSTRUCTION_BLOB:
        rep["stop"].append("W119 instruction provenance mismatch")
    rep["w116_manifest"] = W7.w116_manifest_check()
    rep["w117_manifest"] = W8.w117_manifest_check()
    rep["w118_manifest"] = W9.manifest_check(W118, W9.W118_MANIFEST_SHA)
    rep["w119_manifest"] = W9.manifest_check(W119, W119_MANIFEST_SHA)
    for k in ("w116_manifest", "w117_manifest", "w118_manifest", "w119_manifest"):
        if not rep[k]["ok"]:
            rep["stop"].append(f"{k} does not re-verify")
    fp = {}
    for n, base, dsha, csha, case in ((116, W116, W7.W116_DECL_SHA, W7.W116_DECISION_SHA, "B"),
                                      (117, W117, W8.W117_DECL_SHA, W8.W117_DECISION_SHA, "C"),
                                      (118, W118, W9.W118_DECL_SHA, W9.W118_DECISION_SHA, "D"),
                                      (119, W119, W119_DECL_SHA, W119_DECISION_SHA, "D")):
        dec = load(base / "results/decision/decision.json")
        fp[n] = {"declaration_sha256": S.sha_file(base / "results/declare/declaration.json"),
                 "decision_sha256": S.sha_file(base / "results/decision/decision.json"), "case": dec["case"], "failed": dec.get("failed")}
        if fp[n]["declaration_sha256"] != dsha or fp[n]["decision_sha256"] != csha or fp[n]["case"] != case:
            rep["stop"].append(f"W{n} declaration/decision/CASE not the frozen ones")
    d119 = load(W119 / "results/decision/decision.json")
    fp[119]["V6_detail"] = d119["detail"]["V6"]
    fp[119]["C3_detail_requests"] = [x["req"] for x in d119["detail"]["C3"]]
    rep["w119_d_reasons_reproduced"] = (d119["failed"] == {"V": ["V6_ab_fairness"], "C": ["C3_negative_controls", "C6_no_new_hard_failure"]}
                                        and d119["detail"]["V6"] == ["Z_A"] and sorted(fp[119]["C3_detail_requests"]) == ["NY_FALSE", "NY_PIN1", "RY_PIN1"])
    if not rep["w119_d_reasons_reproduced"]:
        rep["stop"].append("W119 D reasons not as recorded")
    rep["historical_case_fingerprints"] = fp
    rep["w119_worklog_case_d"] = "**Verdict: CASE D" in (ROOT / HIST[119]).read_text(encoding="utf-8")
    if not rep["w119_worklog_case_d"]:
        rep["stop"].append("W119 Worklog CASE D statement missing")
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
    w119_v0 = load(W119 / "results/preflight/real_parent_state.json")
    rep["parent_rebuild"] = {"state_id": v0["state_id"], "w119_state_id": w119_v0["state_id"],
                             "canonical_equal": S.canon(v0) == S.canon(w119_v0), "verifies": S.verify_state(v0)}
    if not rep["parent_rebuild"]["canonical_equal"]:
        rep["stop"].append("rebuilt parent differs from the W119 parent")
    dump(out / "real_parent_state.json", v0)
    C.states_file(out / "real_parent_states.json", {"v0": v0}, "w105")
    rest = C.INPUTS["w105_rig_rest"][0]
    blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest", rest,
            "--states", out / "real_parent_states.json", "--out", out / "real_parent_eval.json", log=out / "real_parent_eval.log")
    blender("scripts/w117_blender_timing.py", "--", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest", rest, "--states",
            out / "real_parent_states.json", "--out", out / "real_timing.json", "--sample-frames", "1,109,118,122,153,154,182,197",
            log=out / "real_timing.log")
    ev = load(out / "real_parent_eval.json")["states"]["v0"]
    w119e = load(W119 / "results/preflight/real_parent_eval.json")["states"]["v0"]
    rep["parent_blender"] = {"value_digest": ev["value_digest"], "equals_w105_fixed_digest": ev["value_digest"] == rep["w105_audit"]["value_digest"],
                             "frames_bit_equal_w119_preflight": ev["frames"] == w119e["frames"]}
    if not (rep["parent_blender"]["equals_w105_fixed_digest"] and rep["parent_blender"]["frames_bit_equal_w119_preflight"]):
        rep["stop"].append("fresh Blender evaluation of the parent does not reproduce W105/W119")
    rep["timing"] = load(out / "real_timing.json")
    spec = load(W119 / "results/preflight/syn_spec.json")
    dump(out / "syn_spec.json", spec)
    rep["syn_spec_equal_w119"] = (out / "syn_spec.json").read_bytes() == (W119 / "results/preflight/syn_spec.json").read_bytes()
    if not rep["syn_spec_equal_w119"]:
        rep["stop"].append("synthetic spec differs from W119")
    parents = {}
    for n in ("A", "B", "C", "D", "E", "G"):
        st = load(W119 / f"results/preflight/syn_parent_P__{n}.json")
        rb = W7.traj_state(n) if n in "ABCD" else (W8.traj_state_E() if n == "E" else W9.traj_state_G())
        if S.canon(st) != S.canon(rb):
            rep["stop"].append(f"W119 synthetic parent {n} does not rebuild")
        parents[f"P__{n}"] = rb
    for n, st in parents.items():
        dump(out / f"syn_parent_{n}.json", st)
    C.states_file(out / "syn_parent_states.json", parents, "synthetic")
    blender("scripts/w116_blender_stages.py", "--", "probe", "--rig", f"synthetic:{out / 'syn_spec.json'}", "--states",
            out / "syn_parent_states.json", "--out", out / "syn_probe.json", "--probe-channels", "base,upper,lower", log=out / "syn_probe.log")
    blender("scripts/w117_blender_timing.py", "--", "--rig", f"synthetic:{out / 'syn_spec.json'}", "--states",
            out / "syn_parent_states.json", "--out", out / "syn_timing.json", log=out / "syn_timing.log")
    sp, wp = load(out / "syn_probe.json"), load(W119 / "results/preflight/syn_probe.json")
    rep["syn_probe_parity_w119"] = {n: sp["states"][n]["frames"] == wp["states"][n]["frames"] for n in wp["states"]}
    rep["syn_rest_equal_w119"] = sp["rest"] == wp["rest"]
    if not all(rep["syn_probe_parity_w119"].values()) or not rep["syn_rest_equal_w119"] or set(sp["states"]) != set(wp["states"]):
        rep["stop"].append("synthetic probe does not reproduce W119")
    rep["syn_timing"] = load(out / "syn_timing.json")
    rep["syn_parent_state_ids"] = {n: s["state_id"] for n, s in parents.items()}
    rep["source_guard"] = guard().before
    rep["code_blobs"] = code_blobs()
    dump(out / "preflight.json", rep)
    print(json.dumps({"stop": rep["stop"], "parent_blender": rep["parent_blender"], "w119_manifest": rep["w119_manifest"]["ok"],
                      "syn_parity": rep["syn_probe_parity_w119"]}, indent=1))
    return 1 if rep["stop"] else 0


# ------------------------------------------------------------------------------------------- build engine ---
class W20Build:
    """W119 per-request isolated F_r / X_r (reproduced via W119's own builder, no Y) + the four W120 factor variants."""

    def __init__(self, pre, hg):
        t0 = time.time()
        self.W = W9.W9Build(pre, hg, with_y=False)
        self.W.build_all()
        self.B = self.W.X.B
        self.recs = self.W.recs
        self.population = list(self.W.population)
        self.replay_wall_s = time.time() - t0

    def ctx(self, rid):
        R = self.B.req[rid]
        store, frames, rig = self.W.side(R)
        return R, store, frames, rig, store[R["parent"]], store[R["U"]]

    def x_vals(self, rid):
        rec = self.recs[rid]
        R, store, frames, rig, P, Ust = self.ctx(rid)
        X = store[rec["X"]]
        return {k: vals_of(X, k) for k in rec["x_info"].get("planned_frames", []) if k not in set(rec["pin_frames"])}

    def fair_rec(self, rid, vals, info):
        R, store, frames, rig, P, Ust = self.ctx(rid)
        pin = R["pin_frames"]
        r = self.W.fair_rec(P, Ust, R, {str(k): vlist(vals[k]) for k in pin if k in vals}, info["margins"], info["budget_frames"])
        r["solver_params_digest"] = info.get("solver_params_digest")
        return r

    def solve(self, rid, f, ctrl=None, margins=None, m_max=None, **kw):
        rec = self.recs[rid]
        R, store, frames, rig, P, Ust = self.ctx(rid)
        return FB.factor_transition(rig, rec["_ctrl"] if ctrl is None else ctrl, rec["_Fv"], R["pin_frames"], frames, R["auth"],
                                    rec["chain"][0], rec["chain"][1], rec["margins"] if margins is None else margins,
                                    rec["x_info"]["m_max"] if m_max is None else m_max, f, x_vals=self.x_vals(rid), **kw)

    def make_factors(self, rid):
        rec = self.recs[rid]
        R, store, frames, rig, P, Ust = self.ctx(rid)
        rec["factors"], rec["_fv"] = {}, {}
        if rec.get("contract_invalid"):
            for f in FACTORS:
                rec["factors"][f] = {"state": R["U"], "fair": None,
                                     "info": {"factor": f, "request_status": "NOT_RESOLVED_CONTRACT_INVALID", "planned_frames": [], "frames": {},
                                              "sides": [], "margins": {}, "budget_frames": rec["x_info"]["budget_frames"], "fallback": None}}
            return
        for f in FACTORS:
            vals, info = self.solve(rid, f)
            name = f"{rid}__{f}"
            store[name] = C.patched_state(P, W7.full_patch(P, rec["_ctrl"], vals), {"kind": "W120_FACTOR_CANDIDATE", "factor": f, "request": rid})
            rec["factors"][f] = {"state": name, "info": info, "fair": self.fair_rec(rid, vals, info)}
            rec["_fv"][f] = vals

    def build_all(self):
        for rid in self.population:
            self.make_factors(rid)
        self.negatives()

    # ---------------- negative controls
    def neg(self, rid, vals=None, info=None, ctrl=None, F_state=None, fair_mut=None, statuses=None, planned=None, rig_id=None,
            display_corrupt=None):
        src, f, desc, cond, want, reasons = NEG[rid]
        r0 = self.recs[src]
        R, store, frames, rig, P, Ust = self.ctx(src)
        rec = {k: copy.deepcopy(v) for k, v in r0.items() if not k.startswith("_")}
        rec.update(role="negative", negative_of=src, neg_factor=f, desc=desc, expect=want, expect_reasons=reasons, conditions=cond)
        fac = copy.deepcopy(r0["factors"][f])
        if F_state is not None:
            rec["F"] = F_state
            rec["isolation_values_F_r"] = YB.isolation_check_values(all_vals(store[F_state]), r0["_ctrl"], R["pin_frames"], frames)
        if vals is not None:
            st = C.patched_state(P, W7.full_patch(P, r0["_ctrl"] if ctrl is None else ctrl, vals),
                                 {"kind": "W120_NEGATIVE_CONTROL", "factor": f, "request": rid})
            if rig_id:
                st = S.make_state(st["parent_id"], rig_id, st["frames"], st["channels"], st["values"], st["op"], st.get("source"))
            store[f"{rid}__{f}"] = st
            fac["state"] = f"{rid}__{f}"
            fac["fair"]["pin_values_digest"] = S.digest({str(k): vlist(vals[k]) for k in R["pin_frames"] if k in vals})
        if info is not None:
            fac["info"] = info
            fac["fair"]["margins"], fac["fair"]["budget_frames"] = info["margins"], info["budget_frames"]
            fac["fair"]["solver_params_digest"] = info.get("solver_params_digest")
        if rig_id:
            fac["fair"]["rig_id"] = rig_id
            rec["V_rig_id"] = rig_id
        if fair_mut:
            fair_mut(fac["fair"])
        if statuses is not None:
            rec["V_statuses_override"] = statuses
        if planned is not None:
            fac["info"] = dict(fac["info"], planned_frames=planned(fac["info"]))
        if display_corrupt:
            rec["display_corrupt"] = display_corrupt
        rec["factors"] = {f: fac}
        self.recs[rid] = rec

    def tam(self, src, f, fn):
        r0 = self.recs[src]
        vv = {k: {c: np.array(x, float) for c, x in v.items()} for k, v in r0["_fv"][f].items()}
        fn(vv, r0["_ctrl"])
        return vv

    def negatives(self):
        B, r = self.B, self.recs
        # zero budget with the W119 global M_MAX (unequal budget)
        v, i = self.solve("Z_A", "G1F1", m_max=XB.M_MAX)
        self.neg("NG_BUDGET", vals=v, info=i)
        mm = dict(r["H_A"]["margins"])
        key = next(k for k in mm if k.endswith("release"))
        mm[key] = max(0, mm[key] - 1)
        v, i = self.solve("H_A", "G0F0", margins=mm)
        self.neg("NG_MARGIN", vals=v, info=i)
        pb = B.syn["P__B"]["state_id"]
        self.neg("NG_PARENT", fair_mut=lambda fr: fr.__setitem__("parent_id", pb))
        # wrong U: solved on the RB_neg control and stored on it
        R0, store, frames, rig, P, Ust = self.ctx("H_RH1_RBpos")
        ctrl_neg = all_vals(store["RB_neg__control"])
        v, i = self.solve("H_RH1_RBpos", "G0F1", ctrl=ctrl_neg)
        uneg = store["RB_neg__control"]["state_id"]
        self.neg("NG_U", vals=v, info=i, ctrl=ctrl_neg, fair_mut=lambda fr: fr.__setitem__("U_state_id", uneg))
        self.neg("NG_RIG", vals=r["H_A"]["_fv"]["G1F1"], rig_id="W120_OTHER_RIG")
        self.neg("NG_CDIGEST", fair_mut=lambda fr: fr.__setitem__("constraint_digests", ["W120_TAMPERED_DIGEST"]))
        self.neg("NG_AUTH", fair_mut=lambda fr: fr.__setitem__("auth", ["upper"]))
        lower_pin = lambda k, ch: (lambda vv, c: vv[k].__setitem__(ch, rot_local(vv[k][ch], [0, 0, 1], 2.0)))  # noqa: E731
        self.neg("NG_PIN_COMBINED", vals=self.tam("H_A", "G0F1", lower_pin(20, "lower")))
        self.neg("RG_PIN_COMBINED", vals=self.tam("H_RH1_none", "G1F0", lower_pin(113, LOWER)))
        pinA = B.req["H_A"]["pin_frames"]

        def false_solved(vv, c):
            for k in pinA:
                vv[k] = dict(c[k])
        self.neg("NG_FALSE", vals=self.tam("H_A", "G1F1", false_solved), statuses={str(k): "SOLVED" for k in pinA})
        v, i = self.solve("H_A", "G0F0", fault="skip_lower_aim")
        self.neg("NG_NONCONV", vals=v, info=i)
        v, i = self.solve("H_A", "G1F0", fault="skip_lower_aim")
        v[20] = dict(v[20], lower=rot_local(v[20]["lower"], [0, 0, 1], 2.0))
        self.neg("NG_NONCONV_PIN", vals=v, info=i)
        mb = dict(r["H_B"]["margins"])
        kb = next(k for k in mb if k.endswith("release"))
        mb[kb] = max(0, mb[kb] - 1)
        v, i = self.solve("H_B", "G0F1", margins=mb, fault="skip_lower_aim")
        self.neg("NG_NONCONV_MARGIN", vals=v, info=i)
        tfA = sorted(k for k in r["H_A"]["factors"]["G1F1"]["info"]["planned_frames"] if k not in set(pinA))[0]
        v, i = self.solve("H_A", "G1F1", fault="skip_lower_aim")
        v[tfA] = dict(v[tfA], base=rot_local(v[tfA]["base"], [1, 0, 0], 1.0))
        self.neg("NG_NONCONV_BONE", vals=v, info=i)
        self.neg("NG_BONE", vals=self.tam("H_A", "G1F0", lambda vv, c: vv[tfA].__setitem__("base", rot_local(vv[tfA]["base"], [1, 0, 0], 1.0))))
        rr = r["H_RH1_none"]
        rtf = sorted(k for k in rr["factors"]["G0F1"]["info"]["planned_frames"] if k not in set(rr["pin_frames"]))[1]
        self.neg("RG_BONE", vals=self.tam("H_RH1_none", "G0F1", lambda vv, c: vv[rtf].__setitem__(
            "spine_01", rot_local(vv[rtf]["spine_01"], [1, 0, 0], 1.0))))
        self.neg("NG_ROOT", vals=self.tam("H_A", "G0F0", lambda vv, c: vv[tfA].__setitem__(
            "__object__", rot_local(vv[tfA]["__object__"], [0, 0, 1], 1.0))))
        self.neg("NG_OUTSIDE", vals=self.tam("H_A", "G1F0", lambda vv, c: vv.__setitem__(2, dict(c[2], upper=rot_local(c[2]["upper"], [0, 0, 1], 1.0)))))
        self.neg("NG_PLANNED", vals=self.tam("H_A", "G0F1", lambda vv, c: None),
                 planned=lambda yi_: [k for k in yi_["planned_frames"] if k != min(yi_["planned_frames"])])
        self.neg("NG_DISPLAY", display_corrupt="DY1")

        def push(k, Sh, G, Lsum):
            return Sh + (G - Sh) / np.linalg.norm(G - Sh) * (Lsum + 0.05)

        def on_axis(k, Sh, G, Rr):
            return Sh + 0.5 * (G - Sh)

        def mirror(k, Sh, G, Rr):
            if k % 2 == 0:
                return Rr
            t = (G - Sh) / np.linalg.norm(G - Sh)
            rv = Rr - Sh
            return Sh + 2.0 * float(np.dot(rv, t)) * t - rv
        v, i = self.solve("H_B", "G1F1", guide_override=push)
        self.neg("NG_GUIDE_UNREACH", vals=v, info=i)
        v, i = self.solve("H_B", "G0F0", pole_override=on_axis)
        self.neg("NG_AMBIG", vals=v, info=i)
        v, i = self.solve("H_B", "G1F1", pole_override=mirror)
        self.neg("NG_FLIP", vals=v, info=i)
        self.neg("NG_INFEASIBLE")
        uo = B.syn["U__A__SYN_E8"]["state_id"] if "U__A__SYN_E8" in B.syn else "W120_OTHER_U"
        self.neg("NG_INFEASIBLE_U", fair_mut=lambda fr: fr.__setitem__("U_state_id", uo))
        self.neg("NG_CONFLICT")
        self.neg("NG_DOF")
        self.neg("NG_SHARED", F_state="SEG_RB_pos__F")
        R0, store, frames, rig, P, Ust = self.ctx("S_RA_pos_60_61")
        sib = dict(r["S_RA_pos_60_61"]["_Fv"])
        sib[63] = vals_of(store["SEG_RA_pos__F"], 63)
        store["NG_SIBLING__Fr"] = C.patched_state(P, W7.full_patch(P, r["S_RA_pos_60_61"]["_ctrl"], sib),
                                                  {"kind": "F_CANDIDATE_CONTAMINATED", "request": "NG_SIBLING"})
        self.neg("NG_SIBLING", F_state="NG_SIBLING__Fr")

    def needed_states(self):
        need = {"real": set(), "syn": set()}
        for rid, rec in self.recs.items():
            sd = "real" if rec["side"] == "real" else "syn"
            need[sd] |= {rec["parent"], rec["U"], rec["F"], rec["X"]} | {x["state"] for x in rec.get("factors", {}).values()}
            if rec.get("F_hist"):
                need[sd].add(rec["F_hist"])
        need["real"].add("RB_neg__control")
        return need

    def report(self):
        recs = {r: {k: v for k, v in x.items() if not k.startswith("_")} for r, x in self.recs.items()}
        B = self.B
        need = self.needed_states()
        return {"records": recs, "population": self.population,
                "real_states": {n: B.real[n]["state_id"] for n in sorted(need["real"])},
                "syn_states": {n: B.syn[n]["state_id"] for n in sorted(need["syn"])},
                "state_meta": {side: {n: {"parent_id": st[n]["parent_id"], "op": st[n]["op"], "rig_id": st[n]["rig_id"]} for n in sorted(need[sd])}
                               for side, st, sd in (("real", B.real, "real"), ("syn", B.syn, "syn"))},
                "all_real_state_ids": {n: s["state_id"] for n, s in B.real.items()},
                "all_syn_state_ids": {n: s["state_id"] for n, s in B.syn.items()}}

    def write_states(self, out):
        need = self.needed_states()
        C.states_file(out / "real_states.json", {n: self.B.real[n] for n in sorted(need["real"])}, "w105")
        C.states_file(out / "syn_states.json", {n: self.B.syn[n] for n in sorted(need["syn"])}, "synthetic")


# ------------------------------------------------------------------------------------------------ calibrate ---
def fixture_preflight(store_dir):
    """Outcome-independent instrument fixtures on the Python-only W118 test rig (no population request is solved)."""
    import test_w118_transition_edit as T8
    FR = list(range(T8.N))
    rep = {"rig": "W118 Python-only test rig (tests/test_w118_transition_edit.py)"}
    r, P, U, pin, base = T8.case()
    # zero budget
    x0, xi0 = XB.transition_edit(r, U, base, pin, FR, T8.AUTH, "lower", m_max=0)
    m0 = YB.margins_of(xi0)
    z = {}
    for f in FACTORS:
        v, i = FB.factor_transition(r, U, base, pin, FR, T8.AUTH, "upper", "lower", m0, xi0["m_max"], f, x_vals={})
        z[f] = {"budget_equal_X": i["budget_frames"] == xi0["budget_frames"] == sorted(pin), "planned_equal_pin": i["planned_frames"] == sorted(pin),
                "values_equal_F": all(all(np.array_equal(v[k][c], base[k][c]) for c in base[k]) for k in pin) and set(v) == set(pin),
                "request_status": i["request_status"]}
    _, yi_w119 = YB.endpoint_transition(r, U, base, pin, FR, T8.AUTH, "upper", "lower", m0)
    fX = {"margins": m0, "budget_frames": xi0["budget_frames"]}
    rep["zero_budget"] = {"factors": z, "all_ok": all(x["budget_equal_X"] and x["planned_equal_pin"] and x["values_equal_F"]
                                                     and x["request_status"] == "SOLVED" for x in z.values()),
                          "w119_y_budget_len": len(yi_w119["budget_frames"]), "x_budget_len": len(xi0["budget_frames"]),
                          "w119_defect_reproduced": yi_w119["budget_frames"] != xi0["budget_frames"]}
    base_rec = {"parent_id": "p", "rig_id": "R", "U_state_id": "u", "constraint_digests": ["c"], "pin_frames": pin, "auth": T8.AUTH,
                "pin_values_digest": "v"}
    Frec = dict(base_rec, margins=None, budget_frames=None)
    Xrec = dict(base_rec, **fX)
    ok = FB.fairness_factor(Frec, Xrec, dict(Xrec, solver_params_digest=FB.SOLVER_PARAMS_DIGEST))
    bad_budget = FB.fairness_factor(Frec, Xrec, dict(Xrec, budget_frames=yi_w119["budget_frames"]))
    rep["unequal_budget"] = {"equal_ok": ok["ok"], "unequal_errors": bad_budget["errors"],
                             "rejected": "SEARCH_BUDGET_MISMATCH:Y" in bad_budget["errors"]}
    single = {}
    for name, kw in (("wrong_U", {"U_state_id": "u2"}), ("wrong_rig", {"rig_id": "R2"}), ("wrong_constraint_digest", {"constraint_digests": ["d"]}),
                     ("wrong_source_parent", {"parent_id": "q"}), ("wrong_authorised_bone", {"auth": ["upper"]}),
                     ("wrong_margin", {"margins": {k: v + 1 for k, v in m0.items()}}), ("wrong_pin_values", {"pin_values_digest": "w"}),
                     ("wrong_solver_params", {"solver_params_digest": "other"})):
        res = FB.fairness_factor(Frec, Xrec, dict(Xrec, **kw), kw.get("solver_params_digest"))
        single[name] = {"rejected": not res["ok"], "errors": res["errors"]}
    rep["single_mismatch_controls"] = single
    rep["cross_factor_identity"] = {
        "equal": FB.cross_factor_identity({f: dict(Xrec, solver_params_digest="s") for f in FACTORS})["ok"],
        "one_margin_differs": not FB.cross_factor_identity({f: dict(Xrec, solver_params_digest="s", margins=({"a": 1} if f == "G1F0" else m0))
                                                            for f in FACTORS})["ok"]}
    # decision-level fixture controls through the actual decide_y
    disp_ok = {"ok": True, "errors": []}
    fp0 = {"frames_changed": [11], "channel_frames": {"upper": [11]}, "outside_arm_points_changed": {}, "matrix_world_changed_frames": []}
    good = {"pins_satisfied": True, "scope_ok": True, "locality_ok": True, "lengths_ok": True, "root_unchanged": True, "joint_limit_ok": True}
    pinbad = dict(good, pins_satisfied=False)
    dec = {}
    dec["pin_tamper_plus_F_mismatch"] = YB.decide_y([], ["SOLVED"], [pinbad], fp0, [10, 11, 12], [10, 11, 12], ["upper", "lower"],
                                                    {"request_status": "SOLVED"}, fairness={"ok": False, "errors": ["PIN_VALUES_DIFFER_FROM_F:Y"]},
                                                    display_check=disp_ok, isolation={"ok": True})
    dec["isolated_nonconvergence"] = YB.decide_y([], ["SOLVED"], [good], fp0, [10, 11, 12], [10, 11, 12], ["upper", "lower"],
                                                 {"request_status": "NON_CONVERGED"}, fairness={"ok": True, "errors": []},
                                                 display_check=disp_ok, isolation={"ok": True})
    dec["corrupted_pin_claims_solved"] = YB.decide_y([], ["SOLVED"], [pinbad], fp0, [10, 11, 12], [10, 11, 12], ["upper", "lower"],
                                                     {"request_status": "SOLVED"}, fairness={"ok": True, "errors": []},
                                                     display_check=disp_ok, isolation={"ok": True})
    dec["false_display"] = YB.decide_y([], ["SOLVED"], [good], fp0, [10, 11, 12], [10, 11, 12], ["upper", "lower"],
                                       {"request_status": "SOLVED"}, fairness={"ok": True, "errors": []},
                                       display_check={"ok": False, "errors": ["CELL_MISMATCH"]}, isolation={"ok": True})
    dec["out_of_domain_change"] = YB.decide_y([], ["SOLVED"], [good], dict(fp0, frames_changed=[11, 30]), [10, 11, 12], [10, 11, 12],
                                              ["upper", "lower"], {"request_status": "SOLVED"}, fairness={"ok": True, "errors": []},
                                              display_check=disp_ok, isolation={"ok": True})
    want = {"pin_tamper_plus_F_mismatch": "CONTRACT_VIOLATION", "isolated_nonconvergence": "SOLVER_FAILURE",
            "corrupted_pin_claims_solved": "SOLVER_FAILURE", "false_display": "CONTRACT_VIOLATION", "out_of_domain_change": "CONTRACT_VIOLATION"}
    rep["decision_fixtures"] = {k: {"got": d, "want_class": want[k], "ok": d["decision"] == "REJECT" and d["failure_class"] == want[k]}
                                for k, d in dec.items()}
    # stale parent on the content-addressed store (fixture states)
    if True:
        st = S.Store(Path(store_dir))
        p0 = S.make_state(None, "T118", [0], ["a"], {"a": [[0, 0, 0, 1]]}, {"kind": "FIXTURE_ROOT"})
        c1 = S.make_state(p0["state_id"], "T118", [0], ["a"], {"a": [[0, 0, 0.1, 0.995]]}, {"kind": "FIXTURE_C1"})
        c2 = S.make_state(p0["state_id"], "T118", [0], ["a"], {"a": [[0, 0.1, 0, 0.995]]}, {"kind": "FIXTURE_C2"})
        st.put(p0)
        st.init_ref("m", p0["state_id"])
        a1 = st.commit("m", p0["state_id"], c1, {"decision": "ACCEPT", "failure_class": "NONE", "reasons": []})
        a2 = st.commit("m", p0["state_id"], c2, {"decision": "ACCEPT", "failure_class": "NONE", "reasons": []})
        rep["stale_parent_fixture"] = {"first": a1["decision"], "second_reasons": a2.get("reasons"), "ok": a2.get("reasons") == ["STALE_PARENT"]}
    rep["precedence_exhaustive"] = FB.precedence_exhaustive_check()
    # better B1 but worse articulation must not count as a meaningful improvement (frozen W119 comparator)
    q0 = {"B1_boundary_wrist_comp_step_peak_m": 0.10, "introduced_flip_transitions": [], "A4_introduced_near_straight_frames": [],
          "D2_elbow_comp_step_peak_m": 0.02, "D3_arm_rot_step_excess_peak_deg": 4.0, "boundary_raw_wrist_step_peak_m": 0.05}
    c = YB.compare_y(q0, dict(q0, B1_boundary_wrist_comp_step_peak_m=0.05, introduced_flip_transitions=[12]))
    rep["better_B1_worse_articulation_fixture"] = {"B1_improved": c["B1_improved"], "meaningful": c["meaningful_improvement"],
                                                   "ok": c["B1_improved"] and not c["meaningful_improvement"]}
    rep["all_ok"] = (rep["zero_budget"]["all_ok"] and rep["zero_budget"]["w119_defect_reproduced"] and rep["unequal_budget"]["equal_ok"]
                     and rep["unequal_budget"]["rejected"] and all(x["rejected"] for x in single.values())
                     and rep["cross_factor_identity"]["equal"] and rep["cross_factor_identity"]["one_margin_differs"]
                     and all(x["ok"] for x in rep["decision_fixtures"].values()) and rep["stale_parent_fixture"]["ok"]
                     and rep["precedence_exhaustive"]["n_mismatches"] == 0 and rep["better_B1_worse_articulation_fixture"]["ok"])
    return rep


def neg_expectation_consistency():
    out = {}
    for rid, (src, f, desc, cond, want, reasons) in NEG.items():
        p = FB.predict_decision(cond)
        out[rid] = {"predicted": p, "declared": want, "ok": [p["decision"], p["failure_class"]] == want}
    return {"controls": out, "all_ok": all(x["ok"] for x in out.values())}


def cmd_calibrate(a):
    pre = Path(a.preflight)
    out = new_dir(a.out_dir)
    pf = load(pre / "preflight.json")
    if pf["stop"]:
        raise SystemExit("STOP: preflight failed")
    g = guard()
    rep = {"schema": "w120_calibration_v1", "label": "AGENT-RUN; no factor quality, Blender factor output or contrast exists in this stage"}
    rep["instrument_preflight"] = fixture_preflight(out / "fixture_store")
    rep["negative_expectations"] = neg_expectation_consistency()
    rep["contamination_reproduction"] = W9.contamination_reproduction()
    d119 = load(W119 / "results/declare/declaration.json")
    hg = d119["hg_design"]
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
    W = W20Build(pre, hg)
    rep["wall_s"] = time.time() - t0
    rep["w118_reproduction"] = W.W.parity
    fz = load(W119 / "results/calibrate/frozen_controls.json")
    if S.sha_file(W119 / "results/calibrate/frozen_controls.json") != W119_FROZEN_CONTROLS_SHA:
        raise SystemExit("STOP: W119 frozen controls changed")
    recs = W.recs
    mism, controls = [], {}
    for rid, c in fz["controls"].items():
        if rid not in recs:
            mism.append(f"{rid}:missing")
            continue
        x = recs[rid]
        for k in ("F_r_state_id", "X_r_state_id", "margins", "pin_frames", "auth", "parent", "U"):
            if x.get(k) != c.get(k):
                mism.append(f"{rid}:{k}")
        if x["x_info"].get("budget_frames") != c.get("budget_frames") or x["x_info"].get("planned_frames") != c.get("planned_frames_X"):
            mism.append(f"{rid}:budget_or_planned")
        valid = not x.get("contract_invalid")
        bp = XB.budget_plan(x["pin_frames"], REAL_FRAMES if x["side"] == "real" else SYN_FRAMES,
                            x["x_info"].get("m_max", XB.M_MAX))[1] if valid else None
        controls[rid] = {k: x.get(k) for k in ("side", "role", "kind", "parent", "U", "auth", "pin_frames", "pin_runs", "F_r_state_id",
                                                "X_r_state_id", "margins", "contract_invalid", "correlation_groups", "F_hist", "chain")}
        controls[rid].update(budget_frames=x["x_info"].get("budget_frames"), planned_frames_X=x["x_info"].get("planned_frames"),
                             m_max_X=x["x_info"].get("m_max"), factor_budget_from_X_m_max=bp,
                             factor_budget_equals_X=(bp == x["x_info"].get("budget_frames")) if valid else None,
                             F_statuses=x.get("F_statuses"), validation_errors=x["validation_errors"])
    rep["w119_frozen_controls_mismatch"] = mism
    rep["zero_budget_Z_A"] = {"m_max_X": recs["Z_A"]["x_info"]["m_max"], "budget_frames": recs["Z_A"]["x_info"]["budget_frames"],
                              "pin_frames": recs["Z_A"]["pin_frames"],
                              "budget_equals_pin": recs["Z_A"]["x_info"]["budget_frames"] == recs["Z_A"]["pin_frames"]}
    rep["budget_matching_all"] = sorted(r for r, c in controls.items() if c["factor_budget_equals_X"] is False)
    rep["baselines_ok"] = {"F_r_isolated_value_level": all(recs[r]["isolation_values_F_r"]["ok"] for r in recs if not recs[r].get("contract_invalid")),
                           "F_r_equals_frozen_resolver": all(recs[r]["F_r_equals_frozen_single_frame_resolver"]["values_bit_equal"] and
                                                             recs[r]["F_r_equals_frozen_single_frame_resolver"]["statuses_equal"]
                                                             for r in recs if not recs[r].get("contract_invalid")),
                           "X_r_equals_w118_X": all(recs[r]["X_r_equals_w118_X"] for r in recs if not recs[r].get("contract_invalid"))}
    hcx = set(recs["H_Cx"]["F_statuses"].values())
    rep["neg_condition_facts"] = {"H_Cx_F_statuses": sorted(hcx), "H_Cx_has_UNREACHABLE": "UNREACHABLE" in hcx,
                                  "N6_validation": recs["N6"]["validation_errors"], "N3_auth": recs["N3"]["auth"],
                                  "H_A_edit_channels": recs["H_A"]["request"]["edit"]["channels"] if recs["H_A"]["request"].get("edit") else None}
    # G1F1 parity digests vs W119 Y (identity only: values digests, no quality), disclosed in the declaration
    w119s = {"real": load(W119 / "results/build_r1/real_states.json")["states"], "syn": load(W119 / "results/build_r1/syn_states.json")["states"]}
    par = {}
    for rid in W.population:
        x = recs[rid]
        if x.get("contract_invalid"):
            continue
        v, i = W.solve(rid, "G1F1")
        R, store, frames, rig_, P, Ust = W.ctx(rid)
        st = C.patched_state(P, W7.full_patch(P, x["_ctrl"], v), {"kind": "W120_FACTOR_CANDIDATE", "factor": "G1F1", "request": rid})
        y = w119s["real" if x["side"] == "real" else "syn"].get(f"{rid}__Yr")
        par[rid] = {"values_equal_w119_Y": y is not None and S.digest(st["values"]) == S.digest(y["values"]),
                    "budget_equal_X": i["budget_frames"] == x["x_info"]["budget_frames"]}
    rep["g1f1_identity_parity"] = {"per_request": par, "non_ZA_all_equal": all(p["values_equal_w119_Y"] for r, p in par.items() if r != "Z_A"),
                                   "note": "values-digest identity only; computed to verify solver parity before declaration (G1F1 = W119 Y, "
                                           "whose outcomes were published in W119); no W120 factor quality was computed"}
    population = {"all_requests": sorted(W.population), "syn_primary": SYN_PRIMARY, "syn_held_out": SYN_HELD_OUT, "near_singular": NEAR_SINGULAR,
                  "rh1": RH1, "rh2": RH2, "family_distinct": FAMILY_DISTINCT, "correlation_groups": CORRELATION,
                  "contaminated_w118_windows": CONTAMINATED, "low_evidence": LOW_EVIDENCE,
                  "real_requests": sorted(r for r in W.population if recs[r]["side"] == "real"),
                  "syn_requests": sorted(r for r in W.population if recs[r]["side"] == "syn"), "negative_controls": sorted(NEG),
                  "display_controls": sorted(DISPLAY_NEG), "review_set": REVIEW_SET}
    frozen = {"schema": "w120_frozen_controls_v1", "controls": controls, "population": population, "hg_design": hg,
              "solver_params": FB.solver_params(), "solver_params_digest": FB.SOLVER_PARAMS_DIGEST}
    dump(out / "frozen_controls.json", frozen)
    rep["frozen_controls_sha256"] = S.sha_file(out / "frozen_controls.json")
    rep["source_guard"] = g.check()
    rep["stop"] = []
    if not rep["instrument_preflight"]["all_ok"]:
        rep["stop"].append("instrument preflight failed")
    if not rep["negative_expectations"]["all_ok"]:
        rep["stop"].append("declared negative expectations contradict the precedence table")
    if mism or not W.W.parity["ok"] or not all(rep["baselines_ok"].values()):
        rep["stop"].append("baseline reproduction failed")
    if rep["budget_matching_all"] or not rep["zero_budget_Z_A"]["budget_equals_pin"]:
        rep["stop"].append("budget matching failed")
    if not rep["neg_condition_facts"]["H_Cx_has_UNREACHABLE"]:
        rep["stop"].append("H_Cx condition assumption false")
    if not rep["g1f1_identity_parity"]["non_ZA_all_equal"]:
        rep["stop"].append("G1F1 does not reproduce W119 Y values")
    dump(out / "calibration.json", rep)
    print(json.dumps({"stop": rep["stop"], "fixtures": rep["instrument_preflight"]["all_ok"],
                      "precedence": [rep["instrument_preflight"]["precedence_exhaustive"]["n"], rep["instrument_preflight"]["precedence_exhaustive"]["n_mismatches"]],
                      "neg": rep["negative_expectations"]["all_ok"], "mism": mism, "parity": W.W.parity["ok"], "baselines": rep["baselines_ok"],
                      "g1f1": rep["g1f1_identity_parity"]["non_ZA_all_equal"], "facts": rep["neg_condition_facts"],
                      "controls": rep["frozen_controls_sha256"]}, indent=1, default=str))
    return 1 if rep["stop"] else 0


# --------------------------------------------------------------------------------------------- declare ---
GATES = {
    "V1_authentication": "preflight has no STOP (HEAD/blob/branch/lock; W113-W119 Worklogs intact with markers; W119 COMPLETE commit "
                         "ancestor; W119 instruction provenance; W116-W119 manifests re-verify; frozen declaration/decision hashes and CASEs "
                         "B/C/D/D; W119 D reasons as recorded; every input hash)",
    "V2_parent_reproduction": "parent canonical-equal to W119; fresh Blender evaluation = W105 FIXED digest and bit-equal to the W119 "
                              "preflight evaluation; synthetic A-E,G rebuild and probe bit-equal to W119",
    "V3_baseline_reproduction": "W117/W118 replay parity; every request's F_r/X_r state id, margins, pin frames, auth, parent, U, budget and "
                                "X planned frames equal the W119 frozen controls (calibrate and both builds)",
    "V4_contamination_reproduced_not_repaired": "W118 V4 contamination reproduced read-only as in W119; W118 and W119 decisions unchanged",
    "V5_clean_isolated_baselines": "every contract-valid non-negative request: F_r value-level and Blender-level isolation, resolver equality",
    "V6_factor_matching": "every contract-valid non-negative request and every factor: W119 ab_fairness(F_r, X_r, variant) ok (incl. "
                          "search budget, margins, pin values) and solver-parameter digest equal; the four variants' fairness records are "
                          "identical (cross_factor_identity); Z_A budget = pin frames for all four",
    "V7_predeclaration_and_instrument": "declaration hash at every stage; outcomes_inspected false; frozen-controls hash; calibrate instrument "
                                        "preflight all_ok (zero budget, unequal budget, single mismatches, decision fixtures, stale parent, "
                                        "exhaustive precedence 0 mismatches, comparator fixture); declared negative expectations equal the "
                                        "precedence-table prediction; code blobs at build equal the declared blobs",
    "V8_independent_blender_evaluation": "every needed state of both builds is Blender-evaluated; synthetic FK <= 1e-5 m; guide residuals, "
                                         "pin residuals, contrasts and G0F0-vs-X differences taken from Blender output",
    "V9_population_complete": "every declared request has F_r/X_r and four variant display specs; every contract-valid request has every pin "
                              "frame Blender-evaluated for X_r and every variant",
    "V10_repeatability": "two builds byte-identical states files and identical variant infos; two Blender evaluations identical",
    "V11_timing_semantics": "every keyframe CONSTANT; Blender frame = index - f0 + 1; sub-frame = keyed frame (real and synthetic)",
    "V12_g1f1_parity": "G1F1 stored values equal the historical W119 Y values and Blender frames bit-equal to the W119 evaluation on every "
                       "contract-valid non-negative request except Z_A (zero-budget contract control, excluded from parity)",
    "V13_negative_controls": "every W120 negative control returns its declared decision/class and required reasons; NG_GUIDE_UNREACH all "
                             "frames GUIDE_UNREACHABLE_OUTER with Blender guide shortfall >= 0.05 m - 1e-6; NG_AMBIG all frames "
                             "BEND_PLANE_AMBIGUOUS; NG_FLIP >= 1 BRANCH_FLIP_VS_PREVIOUS flag and >= 1 Blender-introduced flip; "
                             "NG_SHARED/NG_SIBLING isolation fails at value and Blender level; display controls DG1-DG5 FAIL the truth check",
    "C1_hard_pin_and_scope": "every ACCEPTed X_r and factor variant: every pin-frame Blender residual <= 1 mm; changed channels only authorised; "
                             "changed frames within budget and planned domain; no outside-arm point; root unchanged; every frame passes",
    "C2_truthful_footprint": "every variant passes the display truth check; variant planned domain within budget and equal across the four "
                             "variants and X_r; preview id = committed id; result label = decision; REJECTed never shown applied",
    "C3_atomicity_lineage_reversal": "ACCEPT iff all frames pass and fairness/isolation hold; refs unchanged on every REJECT; stale parent "
                                     "rejected; ACCEPT + exact revert (values byte-equal, Blender bit-equal) on synthetic and real",
    "C4_source_immutability": "inputs and W116-W119 artefacts byte-unchanged across every stage; four manifests re-verify after the run",
    "C5_user_edit_and_collateral": "every ACCEPTed variant keeps the user-edit channel and __object__ bit-equal to U and collateral points equal",
    "C6_no_fallback": "every variant info fallback None; no G0_REFERENCE_MISSING; REJECTed variants never counted as edits",
}
CASE_RULE = (
    "D if any V gate fails.  Else C if any C gate fails.  Else, using only factorial contrasts on eligible requests (all four variants "
    "ACCEPTed and B1(G0F0) defined): for mechanism M in {GUIDE (primary contrast guide_at_F0 = G1F0 vs G0F0), FLOOR (primary contrast "
    "floor_at_G0 = G0F1 vs G0F0)}, M_syn = >= 1 SYN_PRIMARY request with a meaningful improvement (frozen W119 comparator) in M's primary "
    "contrast; M_real = >= 1 RH1 request AND >= 1 RH2 request (DJ_R excluded as correlated with both) with that improvement; M_clean = "
    "M's primary contrast introduces no new bend flip and no new near-straight frame on any eligible SYN_PRIMARY or real evaluation "
    "request; M_vis = the recorded agent review marks >= 1 real request as visibly improved by M with motion inspected across frames.  "
    "A if some M has M_syn, M_real, M_clean and M_vis.  Else B if the primary contrasts are eligible on H_RH1_none, H_RH2_RBpos and H_G "
    "and some M has >= 1 meaningful improvement on an evaluation request in its primary OR conditional contrast (guide_at_F1 / "
    "floor_at_G1).  Else C (no supported mechanism, or contrasts not interpretable on the named release events).")
METRIC_RULE = {
    "window": "W118 union window [first run lo - 9, last run hi + 9], identical for every variant",
    "B1": "peak boundary wrist COMPENSATION step |c_W(k) - c_W(k-1)| (c_W = W^variant - W^U) over the W118 boundary transitions",
    "comparator": f"frozen W119 diagnostic comparator compare_y(ref, tgt): B1(tgt) <= {YB.IMPROVE_RATIO} B1(ref) AND B1(ref) - B1(tgt) >= "
                  f"{YB.IMPROVE_ABS_M} m AND no new bend flip / near-straight frame (vs ref) AND no material worsening of D2/D3/raw boundary "
                  f"wrist step (> max({YB.WORSE_REL} ref, floors {YB.WORSE_FLOOR})).  Material worsening of a contrast = compare_y(tgt, ref) "
                  "meaningful.  Experiment-only diagnostic, not a Production policy",
    "contrasts": {k: f"{v[1]} - {v[0]}" for k, v in FB.CONTRASTS.items()},
    "interaction": f"G1F1 - G1F0 - G0F1 + G0F0 per metric; material on B1 if |.| >= {FB.INTERACTION_ABS_M} m (experiment-only diagnostic)",
    "normalisation": "raw differences per request in metres / degrees; B1 relative change = delta / B1(reference); no pooling across "
                     "requests; family accounting counts a correlated family once",
    "reported_not_gated": ["D2 elbow comp step", "D3 rotation-step excess", "raw boundary wrist step", "mean/peak elbow deviation from U",
                           "introduced flips / near-straight frames vs U", "min flex", "endpoint jumps (entry/release/inner)",
                           "guide statuses and Blender guide residual", "G0F0 vs X_r Blender differences", "variant vs X_r and W119 Y"],
}
VISUAL_PROCEDURE = ("after render and before decide the agent records review/visual_review.json: for each REVIEW_SET request the GIF frames "
                    "are inspected as a consecutive frame sequence (method recorded: real-time playback is not available to the agent), "
                    "with per-mechanism visibility, articulation, flips/near-straight, source consistency, contact (UNKNOWN when bone "
                    "proxies cannot decide) and moved discontinuities; uninspected items are NOT REVIEWED.  Agent-only, not independent")


def cmd_declare(a):
    out = new_dir(a.out_dir)
    pf = load(Path(a.preflight) / "preflight.json")
    cal = load(Path(a.calibration) / "calibration.json")
    if pf["stop"] or cal["stop"]:
        raise SystemExit("STOP: preflight/calibration failed")
    frozen = load(Path(a.calibration) / "frozen_controls.json")
    decl = {
        "schema": "w120_declaration_v1", "batch": BATCH, "outcomes_inspected": False,
        "preflight_dir": str(Path(a.preflight).resolve()), "calibration_dir": str(Path(a.calibration).resolve()),
        "preflight_sha256": S.sha_file(Path(a.preflight) / "preflight.json"),
        "calibration_sha256": S.sha_file(Path(a.calibration) / "calibration.json"),
        "frozen_controls_sha256": S.sha_file(Path(a.calibration) / "frozen_controls.json"),
        "code_blobs": code_blobs(), "hg_design": frozen["hg_design"],
        "fingerprints": {"inputs": {k: v["sha256"] for k, v in pf["inputs"].items()}, "parent_v0": pf["parent_rebuild"]["state_id"],
                         "parent_blender_digest": pf["parent_blender"]["value_digest"], "syn_parents": pf["syn_parent_state_ids"],
                         "w119": {"manifest": W119_MANIFEST_SHA, "declaration": W119_DECL_SHA, "decision": W119_DECISION_SHA,
                                  "frozen_controls": W119_FROZEN_CONTROLS_SHA, "case": "D (immutable)", "complete_commit": W119_COMPLETE_COMMIT},
                         "w118": {"manifest": W9.W118_MANIFEST_SHA, "declaration": W9.W118_DECL_SHA, "decision": W9.W118_DECISION_SHA, "case": "D"},
                         "w117": {"declaration": W8.W117_DECL_SHA, "decision": W8.W117_DECISION_SHA, "case": "C"},
                         "w116": {"declaration": W7.W116_DECL_SHA, "decision": W7.W116_DECISION_SHA, "case": "B"}},
        "factors": FB.FACTOR_DEF, "solver_params": FB.solver_params(), "solver_params_digest": FB.SOLVER_PARAMS_DIGEST,
        "shared": "solver, pole rule, weight, ambiguity handling, tolerances, pin frames (F_r), margins (X_r), budget m_max (X_r), root "
                  "restriction and authorisation are identical for G0F0/G0F1/G1F0/G1F1; no fallback to X, F or U",
        "x_r_external": "X_r is an external unchanged baseline; G0F0 re-solves X_r's wrist with the shared solver and is not identical by "
                        "definition; its Blender difference from X_r (wrist, elbow, rotations, bend branch) is reported, not attributed to G",
        "precedence_table": FB.PRECEDENCE_TABLE, "condition_keys": FB.CONDITION_KEYS,
        "primary_mechanism_test": {"GUIDE": FB.PRIMARY_CONTRAST["GUIDE"], "FLOOR": FB.PRIMARY_CONTRAST["FLOOR"],
                                   "conditional": FB.CONDITIONAL_CONTRAST, "interaction": "G1F1 - G1F0 - G0F1 + G0F0",
                                   "context_only": "each variant vs X_r and vs historical W119 Y"},
        "metric_rule": METRIC_RULE, "gates": GATES, "case_rule": CASE_RULE,
        "hard_pin": {"tolerance_m": S.TOL_M, "evaluator": "Blender scene world (w116_blender_stages evaluate, unchanged)",
                     "status": "W116 provisional diagnostic criterion, unchanged"},
        "population": frozen["population"], "negative_controls": NEG, "display_controls": DISPLAY_NEG,
        "negative_expectation_prediction": cal["negative_expectations"],
        "lineage": {"synthetic": ["S1 commit G1F0 of the first ACCEPTed among [H_G, H_A, H_B, H_E] on syn/main",
                                  "S2 NG_PIN_COMBINED on syn/atomic -> REJECT, head unchanged",
                                  "S3 G0F1 of the next of that list against the old head -> STALE_PARENT",
                                  "S4 revert syn/main -> values byte-equal, Blender bit-equal to the parent"],
                    "real": ["R1 commit G0F1 of the first ACCEPTed among RH1 + [H_RH2_RBpos, DJ_R] on real/main",
                             "R2 RG_PIN_COMBINED on real/atomic -> REJECT", "R3 G1F0 of another of that list against v0 -> STALE_PARENT",
                             "R4 revert real/main -> byte-equal, Blender bit-equal to v0"],
                    "per_request": "X_r and every variant previewed (PROPOSED, state id) and committed on its own ref"},
        "render_rule": {"review_set": REVIEW_SET, "variants": VARIANTS_SHOWN, "labels": VLABEL,
                        "scale": "300x300 px Blender bone-proxy tile per variant, orthographic 1.1 (real) / 1.2 (synthetic), front camera "
                                 "centred on the first pin target; footprint band below each tile"},
        "visual_review_procedure": VISUAL_PROCEDURE,
        "repetition_plan": "two builds (r1, r2) byte-compared; two Blender evaluations (e1, e2) of r1 compared state by state; exact equality "
                           "required (deterministic code, no tolerance)",
        "predictions_not_gates": {"RH1": "the floor contrast carries most of the W119 RH1 gain (guide unreachable on a near-straight arm)",
                                  "RH2": "the guide contrast carries the RH2 gain (guide reachable)",
                                  "synthetic": "small or no contrasts on smooth synthetic motion"},
        "pre_declaration_disclosure": [
            "focused unit tests on the Python-only W118 test rig",
            "one crash-only smoke run of build -> evaluate -> assess -> render -> decide in a throwaway directory with stdout discarded "
            "unread; only exit codes and Python tracebacks were read",
            "calibrate computed G1F1 values digests for parity with the published W119 Y (identity only, no quality)"],
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
    blobs = code_blobs()
    if blobs != decl["code_blobs"] and not a.smoke:
        raise SystemExit("STOP: code differs from the declared blobs")
    frozen = load(cal / "frozen_controls.json")
    g = guard()
    t0 = time.time()
    W = W20Build(decl["preflight_dir"], decl["hg_design"])
    W.build_all()
    rep = W.report()
    fc = frozen["controls"]
    mism = [r for r in fc if any(rep["records"][r].get(k) != fc[r].get(k) for k in ("F_r_state_id", "X_r_state_id", "margins"))]
    W.write_states(out)
    rep.update({"schema": "w120_build_v1", "tag": a.tag, "declaration_sha256": a.declaration_sha, "wall_s": time.time() - t0,
                "replay_wall_s": W.replay_wall_s, "w118_reproduction": W.W.parity, "frozen_controls_mismatch": mism, "code_blobs": blobs,
                "variant_wall_s": {r: {f: x["info"].get("wall_s") for f, x in rec.get("factors", {}).items()} for r, rec in rep["records"].items()},
                "source_guard": g.check()})
    dump(out / "build_report.json", rep)
    print(json.dumps({"tag": a.tag, "wall_s": rep["wall_s"], "records": len(rep["records"]), "parity": W.W.parity["ok"],
                      "frozen_mismatch": mism, "source": rep["source_guard"]["unchanged"]}))
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
def classify_v(dec, info):
    rs = set(dec["reasons"])
    if dec["decision"] == "REJECT":
        if "Y_MISSING_AUTHORIZED_DOF" in rs:
            return "MISSING_AUTHORIZED_DOF"
        if "NON_CONVERGED" in rs:
            return "SOLVER_NONCONVERGENCE"
        if dec["failure_class"] in ("INFEASIBLE_REQUEST", "UNSUPPORTED_BINDING"):
            return "INFEASIBLE_OR_CONFLICTING_OR_UNBOUND_REQUEST"
        if "EVALUATED_PIN_UNSATISFIED" in rs or "JOINT_LIMIT" in rs:
            return "HARD_PIN_OR_LIMIT_FAILURE"
        return "OUTPUT_OR_BASELINE_CONTRACT_VIOLATION" if dec["failure_class"] == "CONTRACT_VIOLATION" else "REJECTED_OTHER"
    if not info.get("frames"):
        return "ACCEPTED_NO_TRANSITION_AVAILABLE"
    sc, fc = info.get("status_counts", {}), info.get("flag_counts", {})
    tag = []
    if sc.get("GUIDE_UNREACHABLE_OUTER") or sc.get("GUIDE_UNREACHABLE_INNER"):
        tag.append("TARGET_UNREACHABLE")
    if sc.get("GUIDE_FLEX_FLOOR_CLAMPED"):
        tag.append("FLEX_FLOOR_CLAMPED")
    if fc.get("BEND_PLANE_AMBIGUOUS"):
        tag.append("BEND_PLANE_AMBIGUOUS")
    if fc.get("BRANCH_FLIP_VS_PREVIOUS"):
        tag.append("BRANCH_FLIP_FLAGGED")
    return "TARGET_REACHED_ALL_FRAMES" if not tag else "+".join(tag)


def assess_record(rid, rec, ev, rig, bindings, rest, scale, frames_all, store_obj, full_state, states, Bb, w119):
    side = rec["side"]
    U, P = C.ev_frames(ev, rec["U"]), C.ev_frames(ev, rec["parent"])
    pin, auth, chain = rec["pin_frames"], rec["auth"], rec["chain"]
    errs = [tuple(e) for e in rec["validation_errors"]]
    edit_ch = list(rec["request"]["edit"]["channels"]) if rec["request"].get("edit") else []
    dom = S.domain_points(rig, sorted(set(auth) | set(edit_ch))) if (auth or edit_ch) else set()
    arm_dom = S.domain_points(rig, auth) if auth else set()
    pf = W7.pins_fn(bindings, rec["request"])
    wb, eb = (chain[1], "tail"), (chain[0], "tail")
    variants = {"F": rec["F"], "X": rec["X"]} | {f: x["state"] for f, x in rec["factors"].items()}
    if rec.get("F_hist") and rec["kind"] == "segment" and rec["role"] != "negative":
        variants["F_shared_hist"] = rec["F_hist"]
    quality_on = (rec["role"] in ("evaluation", "development", "noop", "control") or rid == "NG_FLIP") and not errs
    out = {}
    for v, name in variants.items():
        Xf = C.ev_frames(ev, name)
        fp = XB.derive_footprint(Xf, U, frames_all, pin, arm_dom, wrist=wb, elbow=eb)
        r = {"state": name, "state_id": ev["states"][name]["state_id"], "footprint": {k: x for k, x in fp.items() if k != "per_frame"}}
        if v == "X":
            planned, budget = rec["x_info"].get("planned_frames", []), rec["x_info"]["budget_frames"]
        elif v in FACTORS:
            planned, budget = rec["factors"][v]["info"].get("planned_frames", []), rec["factors"][v]["info"]["budget_frames"]
        else:
            planned = budget = sorted(pin)
        spec = XB.display_spec(fp, frames_all, pin, budget, planned, r["state_id"], ev["states"][rec["parent"]]["state_id"],
                               "PROPOSED", rid, v, edit_ch)
        r["display_spec_proposed"] = spec
        r["display_check_proposed"] = XB.check_display(spec, Xf, U, frames_all, pin, r["state_id"])
        r["planned_frames"], r["budget_frames"] = planned, budget
        if v in ("F", "F_shared_hist"):
            r["isolation_blender"] = YB.isolation_check_footprint(fp, pin)
        if v == "X" or v in FACTORS:
            evals, per = [], {}
            if not errs:
                for k in sorted(set(planned) | set(pin)):
                    e = S.evaluate_candidate(Xf[k], U[k], P[k], pf(k) if k in set(pin) else [],
                                             {"vs_parent": sorted(set(auth) | set(edit_ch)), "vs_control": sorted(auth)}, dom, rest, scale,
                                             chain=tuple(chain))
                    per[k] = e
                    evals.append(e)
            stmap = rec.get("V_statuses_override") if (v in FACTORS and rec.get("V_statuses_override")) else rec["F_statuses"]
            statuses = [s for s in stmap.values() if s != "NOT_APPLICABLE"]
            if errs:
                dec = S.decide(errs, statuses, evals)
            elif not statuses:
                dec = {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": ["NOT_RESOLVED"]}
            elif v == "X":
                dec = XB.decide_x([], statuses, evals, fp, budget, planned, auth, display_check=r["display_check_proposed"])
            else:
                fac = rec["factors"][v]
                fair = FB.fairness_factor(rec["fair"]["F"], rec["fair"]["X"], fac["fair"], fac["fair"].get("solver_params_digest"))
                disp = r["display_check_proposed"]
                if rec.get("display_corrupt"):
                    bad = W9.corrupt_display(rec["display_corrupt"], spec)
                    r["display_corrupted_for_decision"] = True
                    disp = XB.check_display(bad, Xf, U, frames_all, pin, r["state_id"])
                    r["display_check_corrupted"] = disp
                iso = out["F"]["isolation_blender"]
                dec = YB.decide_y([], statuses, evals, fp, budget, planned, auth, fac["info"], fairness=fair, display_check=disp, isolation=iso)
                r["fairness"] = fair
                r["info"] = fac["info"]
                r["guide_residuals"] = YB.guide_residuals(Xf, fac["info"], chain[1]) if fac["info"].get("frames") else {}
                r["class"] = classify_v(dec, fac["info"])
            res = [p["candidate_residual_m"] for k in pin if k in per for p in per[k]["pins"]]
            r.update(decision=dec,
                     frame_pass={str(k): bool(e["pins_satisfied"] and e["scope_ok"] and e["locality_ok"] and e["lengths_ok"]
                                              and e["root_unchanged"] and e["joint_limit_ok"]) for k, e in per.items()},
                     residual_m={"max": max(res) if res else None, "n_over_tol": sum(1 for x in res if x > S.TOL_M), "n": len(res),
                                 "per_frame": {str(k): per[k]["pins"][0]["candidate_residual_m"] for k in pin if k in per and per[k]["pins"]}},
                     user_edit_and_object_equal_U=all(Xf[k]["stored_wxyz"].get(c) == U[k]["stored_wxyz"].get(c)
                                                      for k in frames_all for c in list(edit_ch) + [S.OBJECT]))
        if quality_on and v != "F_shared_hist":
            r["quality"] = XB.quality(Xf, U, frames_all, pin, chain[0], chain[1], auth)
            r["collateral"] = W7.collateral(Xf, U, P, list(range(frames_all[0], frames_all[-1] + 1)), side)
            r["endpoint_jumps"] = FB.endpoint_jumps(r["quality"], rec["pin_runs"], rec.get("margins") or {})
        out[v] = r
    # ---- comparisons
    cmp_ = {}
    acc = lambda v: v in out and out[v].get("decision", {}).get("decision") == "ACCEPT"  # noqa: E731
    q = {v: out[v].get("quality") for v in out}
    b1 = lambda v: (q.get(v) or {}).get("B1_boundary_wrist_comp_step_peak_m")  # noqa: E731
    elig = rec["role"] != "negative" and all(acc(f) and q.get(f) is not None for f in FACTORS) and b1("G0F0") is not None
    if elig:
        cmp_["factorial"] = FB.factorial({f: q[f] for f in FACTORS})
    if rec["role"] != "negative" and acc("X") and q.get("X") is not None and b1("X") is not None:
        cmp_["vs_X"] = {f: YB.compare_y(q["X"], q[f]) for f in FACTORS if acc(f) and q.get(f) is not None}
        cmp_["X_vs_F_clean_W118_rule"] = XB.compare_x(q["F"], q["X"]) if q.get("F") else None
    # transition frames (planned minus pin; identical for all factors and X)
    tf = sorted(set(rec["x_info"].get("planned_frames", [])) - set(pin))
    if rec["role"] != "negative" and not errs and tf and "G0F0" in out:
        cmp_["G0F0_vs_X_blender"] = FB.variant_diff(C.ev_frames(ev, rec["factors"]["G0F0"]["state"]), C.ev_frames(ev, rec["X"]), tf, chain[0], chain[1])
        cmp_["G1F1_vs_X_blender"] = {k: v for k, v in FB.variant_diff(C.ev_frames(ev, rec["factors"]["G1F1"]["state"]), C.ev_frames(ev, rec["X"]),
                                                                      tf, chain[0], chain[1]).items() if k != "per_frame"}
    # historical W119 Y (read-only context)
    if w119 is not None and rec["role"] != "negative" and not rec.get("contract_invalid") and "G1F1" in out:
        yname = f"{rid}__Yr"
        if yname in w119["states"]:
            Yf = {fr["frame_index"]: fr for fr in w119["states"][yname]["frames"]}
            Uh = {fr["frame_index"]: fr for fr in w119["states"][rec["U"]]["frames"]}
            G = C.ev_frames(ev, rec["factors"]["G1F1"]["state"])
            hist = {"w119_Y_value_digest": w119["states"][yname]["value_digest"], "G1F1_value_digest": ev["states"][rec["factors"]["G1F1"]["state"]]["value_digest"],
                    "frames_bit_equal_G1F1": all(Yf[k]["world"] == G[k]["world"] and Yf[k]["stored_wxyz"] == G[k]["stored_wxyz"] for k in frames_all),
                    "U_frames_bit_equal_w119": all(Uh[k]["world"] == U[k]["world"] and Uh[k]["stored_wxyz"] == U[k]["stored_wxyz"] for k in frames_all),
                    "w119_Y_decision": w119["decisions"].get(rid)}
            if quality_on:
                qh = XB.quality(Yf, U, frames_all, pin, chain[0], chain[1], auth)
                hist["B1_w119_Y_m"] = qh["B1_boundary_wrist_comp_step_peak_m"]
            cmp_["w119_Y_hist"] = hist
    # ---- commit X and every variant on its own ref (preview -> commit -> result)
    if store_obj is not None:
        par = full_state(side, rec["parent"])
        for v in ["X"] + [f for f in FACTORS if f in out]:
            x = out[v]
            name = rec["X"] if v == "X" else rec["factors"][v]["state"]
            cand = W9.rebuild_state(Bb, side, name, states[side][name], par)
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
            res_spec = dict(x["display_spec_proposed"], status="ACCEPTED" if crec["decision"] == "ACCEPT" else "REJECTED", head_after=crec["head_after"])
            x["commit"] = {"preview_state_id": x["display_spec_proposed"]["state_id"], "candidate_state_id": cand["state_id"],
                           "parent_state_id": par["state_id"], **{k: crec.get(k) for k in ("decision", "reasons", "head_before", "head_after")},
                           "preview_id_equals_candidate": x["display_spec_proposed"]["state_id"] == cand["state_id"],
                           "head_consistent": (crec["head_after"] == cand["state_id"]) if crec["decision"] == "ACCEPT"
                           else (crec["head_after"] == par["state_id"])}
            x["display_spec_result"] = res_spec
            x["display_check_result"] = XB.check_display(res_spec, C.ev_frames(ev, name), U, frames_all, pin, cand["state_id"],
                                                         decision=crec["decision"])
    return {"variants": out, "comparisons": cmp_, "eligible_factorial": elig}


def lineage(out, Bb, A, full_state, states, rev, sev, pre):
    store = S.Store(out / "lineage_store")
    res = {"synthetic": [], "real": []}
    Rq = A["requests"]
    dec = lambda rid, f: Rq[rid]["variants"][f]["decision"]  # noqa: E731
    recs = Bb["records"]

    def cand(side, rid, f):
        r = recs[rid]
        par = full_state(side, r["parent"])
        name = r["factors"][f]["state"]
        return W9.rebuild_state(Bb, side, name, states[side][name], par), par
    rev_eval = {}
    order = ["H_G", "H_A", "H_B", "H_E"]
    pick = next((r for r in order if dec(r, "G1F0")["decision"] == "ACCEPT"), None)
    if pick:
        c1, P = cand("syn", pick, "G1F0")
        store.put(P)
        store.init_ref("syn/main", P["state_id"])
        res["synthetic"].append(dict(step="S1", request=pick, variant="G1F0", **store.commit("syn/main", P["state_id"], c1, dec(pick, "G1F0"))))
        c2, PA = cand("syn", "NG_PIN_COMBINED", "G0F1")
        store.put(PA)
        store.init_ref("syn/atomic", PA["state_id"])
        h = store.refs["syn/atomic"]
        r2 = store.commit("syn/atomic", h, c2, dec("NG_PIN_COMBINED", "G0F1"))
        res["synthetic"].append(dict(step="S2", head_unchanged=r2["head_after"] == h, **r2))
        other = next(r for r in order if r != pick)
        c3, _ = cand("syn", other, "G0F1")
        res["synthetic"].append(dict(step="S3", request=other, variant="G0F1", **store.commit("syn/main", P["state_id"], c3, dec(other, "G0F1"))))
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
    order_r = RH1 + ["H_RH2_RBpos", "DJ_R"]
    pick = next((r for r in order_r if dec(r, "G0F1")["decision"] == "ACCEPT"), None)
    if pick:
        c1, _ = cand("real", pick, "G0F1")
        res["real"].append(dict(step="R1", request=pick, variant="G0F1", **store.commit("real/main", v0["state_id"], c1, dec(pick, "G0F1"))))
    h = store.refs["real/atomic"]
    r2 = store.commit("real/atomic", h, cand("real", "RG_PIN_COMBINED", "G1F0")[0], dec("RG_PIN_COMBINED", "G1F0"))
    res["real"].append(dict(step="R2", head_unchanged=r2["head_after"] == h, **r2))
    other = next(r for r in order_r if r != pick)
    res["real"].append(dict(step="R3", request=other, variant="G1F0",
                            **store.commit("real/main", v0["state_id"], cand("real", other, "G1F0")[0], dec(other, "G1F0"))))
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


def load_w119_eval(side_file, names):
    ev = load(W119 / "results/eval_e1" / side_file)
    return {n: ev["states"][n] for n in names if n in ev["states"]}


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
    # historical W119 Y context (read-only)
    w119a = load(W119 / "results/assess/assess_report.json")["requests"]
    w119dec = {r: x["variants"]["Y"]["decision"] for r, x in w119a.items()}
    del w119a
    pop = [r for r, x in Bb["records"].items() if x["role"] != "negative"]
    names = {s: {f"{r}__Yr" for r in pop if Bb["records"][r]["side"] == s} | {Bb["records"][r]["U"] for r in pop if Bb["records"][r]["side"] == s}
             for s in ("real", "syn")}
    w119 = {"real": {"states": load_w119_eval("real_eval.json", names["real"]), "decisions": w119dec},
            "syn": {"states": load_w119_eval("syn_eval.json", names["syn"]), "decisions": w119dec}}
    store = S.Store(out / "store")
    rep = {"schema": "w120_assess_v1", "declaration_sha256": a.declaration_sha, "label": "AGENT-RUN Blender evaluation", "requests": {}}
    t0 = time.time()
    for rid, rec in Bb["records"].items():
        if rec["side"] == "real":
            ev, rig, bnd, rest, scale, fr, wh = rev, ctx.rig, ctx.real_b, rev["rest"], rev["rig"]["scale"][0], REAL_FRAMES, w119["real"]
        else:
            ev, rig, bnd, rest, scale, fr, wh = sev, srig, dict(syn_b) | w116_syn, sev["rest"], 1.0, SYN_FRAMES, w119["syn"]
        A = assess_record(rid, rec, ev, rig, bnd, rest, scale, fr, store, full_state, states, Bb, wh)
        A.update(role=rec["role"], side=rec["side"], kind=rec["kind"], pin_frames=rec["pin_frames"], pin_runs=rec["pin_runs"],
                 validation_errors=rec["validation_errors"], correlation_groups=rec["correlation_groups"], parent=rec["parent"],
                 negative_of=rec.get("negative_of"), neg_factor=rec.get("neg_factor"), expect=rec.get("expect"),
                 expect_reasons=rec.get("expect_reasons"), margins=rec.get("margins"))
        rep["requests"][rid] = A
    rep["assess_wall_s"] = time.time() - t0
    dcs = {}
    for dc, (src, f, kind, desc) in DISPLAY_NEG.items():
        rec = Bb["records"][src]
        ev, fr = (rev, REAL_FRAMES) if rec["side"] == "real" else (sev, SYN_FRAMES)
        y = rep["requests"][src]["variants"][f]
        base = y["display_spec_result"] if kind == "DY5" else y["display_spec_proposed"]
        s = W9.corrupt_display(kind, base)
        if s is None:
            dcs[dc] = {"executed": False, "reason": "corruption not applicable"}
            continue
        name = rec["factors"][f]["state"]
        chk = XB.check_display(s, C.ev_frames(ev, name), C.ev_frames(ev, rec["U"]), fr, rec["pin_frames"], ev["states"][name]["state_id"],
                               decision=y["commit"]["decision"] if kind == "DY5" else None)
        dcs[dc] = {"executed": True, "source": src, "factor": f, "desc": desc, "check": chk}
    rep["display_controls"] = dcs
    rep["lineage"] = lineage(out, Bb, rep, full_state, states, rev, sev, pre)
    rep["source_guard"] = g.check()
    rep["manifests_after"] = {"w116": W7.w116_manifest_check(), "w117": W8.w117_manifest_check(),
                              "w118": W9.manifest_check(W118, W9.W118_MANIFEST_SHA), "w119": W9.manifest_check(W119, W119_MANIFEST_SHA)}
    rep["decisions_after"] = {n: {"sha256": S.sha_file(b / "results/decision/decision.json"), "case": load(b / "results/decision/decision.json")["case"]}
                              for n, b in (("w118", W118), ("w119", W119))}
    rep["parent_files_verify"] = {"real": S.verify_state(load(pre / "real_parent_state.json")),
                                  **{n: S.verify_state(load(pre / f"syn_parent_{n}.json")) for n in ("P__A", "P__B", "P__C", "P__D", "P__E", "P__G")}}
    rep["evaluated_states"] = {"real_missing": sorted(set(Bb["real_states"]) - set(rev["states"])),
                               "syn_missing": sorted(set(Bb["syn_states"]) - set(sev["states"]))}
    dump(out / "assess_report.json", rep)
    print(json.dumps({"n": len(rep["requests"]), "source": rep["source_guard"]["unchanged"], "wall": rep["assess_wall_s"]}))
    return 0


# --------------------------------------------------------------------------------------------- render ---
COL = W8.COL
SERIES = {"U": "#8a8985", "F": "#2a78d6", "X": "#eb6834", "G0F0": "#1baf7a", "G0F1": "#eda100", "G1F0": "#e87ba4", "G1F1": "#008300"}
STYLE = {"F": "-", "X": "-", "G0F0": "--", "G0F1": "--", "G1F0": "-.", "G1F1": "-."}
MARK = {"F": "o", "X": "s", "G0F0": "^", "G0F1": "v", "G1F0": "D", "G1F1": "P"}
GCOL = W9.GCOL


def draw_timeline(path, rid, A):
    from PIL import Image, ImageDraw
    r = A["requests"][rid]
    vs = [v for v in ["F", "X"] + list(FACTORS) if v in r["variants"]]
    fv = [v for v in FACTORS if v in r["variants"]]
    x0 = r["variants"]["X"]
    frames = x0["display_spec_proposed"]["frames"]
    n = len(frames)
    cw = max(4, min(22, 1100 // n))
    L, top, rowh = 300, 40 + 16 * (len(vs) + 1), 20
    H = top + (3 + len(vs) + 3 * len(fv)) * (rowh + 4) + 120
    W = L + cw * n + 30
    im = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(im)
    ink, mute = (11, 11, 11), (82, 81, 78)
    d.text((10, 6), f"{rid}   [{r['side']}]   W120 factor timeline (actual Blender output vs U; scratch diagnostic, not product UI)", fill=ink)
    d.text((10, 20), f"hard-pin runs {r['pin_runs']}   automatic budget {len(x0['budget_frames'])} frames   margins {r.get('margins')}", fill=mute)
    for i, v in enumerate(["X"] + fv):
        x = r["variants"][v]
        cm = x.get("commit", {})
        acc = cm.get("decision") == "ACCEPT"
        d.text((10, 36 + 16 * i), f"{v:5s} -> {cm.get('decision', '?')}{'ED' if cm.get('decision') else ''}"
                                  + ("" if acc else "  (NOT applied)") + f"  {x['decision']['failure_class']} {x['decision']['reasons'][:3]}"
                                  + (f"  class {x.get('class')}" if x.get("class") else ""), fill=ink if acc else (180, 30, 30))

    def x_of(i):
        return L + i * cw
    for i, f in enumerate(frames):
        if f % (10 if n > 60 else 5) == 0:
            d.text((x_of(i) - 4, top - 14), str(f), fill=mute)
    y = top

    def row(label, cells, hatch=False, colmap=None):
        nonlocal y
        d.text((10, y + 4), label, fill=ink)
        for i, f in enumerate(frames):
            c = cells.get(str(f), "UNCHANGED")
            col = (colmap or COL).get(c, COL["UNCHANGED"])
            d.rectangle([x_of(i), y, x_of(i) + cw - 2, y + rowh], fill=col)
            if hatch and c != "UNCHANGED":
                for t in range(0, rowh, 5):
                    d.line([(x_of(i), y + t), (x_of(i) + cw - 2, y + min(rowh, t + cw))], fill=(90, 90, 90))
        y += rowh + 4
    pinset = set(r["pin_frames"])
    row("1 hard-pin domain", {str(f): "PIN_CHANGED" if f in pinset else "UNCHANGED" for f in frames})
    row("2 authorised automatic budget", {str(f): ("PIN_UNCHANGED" if f in pinset else "AUTO_CHANGED") if f in set(x0["budget_frames"]) else "UNCHANGED"
                                          for f in frames}, hatch=True)
    row("  planned transition domain (X = all)", {str(f): ("PIN_UNCHANGED" if f in pinset else "AUTO_CHANGED") if f in set(x0["planned_frames"])
                                                  else "UNCHANGED" for f in frames})
    for v in vs:
        rej = v != "F" and r["variants"][v].get("commit", {}).get("decision") != "ACCEPT"
        row("3 ACTUAL " + VLABEL.get(v, v)[:30] + (" [REJECTED]" if rej else ""), r["variants"][v]["display_spec_proposed"]["cells"], hatch=rej)
    for v in fv:
        cf = r["variants"][v]["footprint"]["channel_frames"]
        chs = sorted(cf)
        cells = {str(f): ("PIN_CHANGED" if f in pinset else "AUTO_CHANGED") for c in chs for f in cf[c]}
        row(f"   {v} bones changed: {','.join(chs)[:24]}", cells)
        yi = r["variants"][v].get("info", {}).get("frames", {})
        gc = {k: ("AMBIG" if "BEND_PLANE_AMBIGUOUS" in fi["flags"] else fi["status"]) for k, fi in yi.items()}
        row(f"4 {v} target status", gc, colmap=dict(GCOL, UNCHANGED=COL["UNCHANGED"]))
    lx = 10
    for name, c in (("hard-pin frame", COL["PIN_CHANGED"]), ("pin unchanged", COL["PIN_UNCHANGED"]), ("automatic edit", COL["AUTO_CHANGED"]),
                    ("unchanged", COL["UNCHANGED"]), ("target reached", GCOL["GUIDE_REACHED"]), ("flex-floor clamped", GCOL["GUIDE_FLEX_FLOOR_CLAMPED"]),
                    ("unreachable (clamped, disclosed)", GCOL["GUIDE_UNREACHABLE_OUTER"]), ("bend plane ambiguous", GCOL["AMBIG"])):
        d.rectangle([lx, y + 6, lx + 14, y + 18], fill=c)
        d.text((lx + 18, y + 6), name, fill=ink)
        lx += 18 + 7 * len(name) + 14
    d.text((10, y + 26), "Hatched = budget or REJECTED (never applied). Rows 3 = Blender-evaluated actual change vs U per variant. "
                         "Axis = Action frame index (source 50 fps / Action 30 fps not reconciled).", fill=mute)
    im.crop((0, 0, W, y + 46)).save(path)


def draw_charts(path_steps, path_flex, rid, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = A["requests"][rid]
    qF = r["variants"]["F"]["quality"]
    ks = sorted(int(k) for k in qF["per_transition"])
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), dpi=110, sharex=True)
    for v in ["F", "X"] + list(FACTORS):
        q = r["variants"].get(v, {}).get("quality")
        if not q:
            continue
        pt = q["per_transition"]
        kw = dict(color=SERIES[v], lw=2, ls=STYLE[v], marker=MARK[v], ms=4, label=VLABEL[v].strip())
        axs[0].plot(ks, [pt[str(k)]["wrist_comp_step_m"] * 1e3 for k in ks], **kw)
        axs[1].plot(ks, [pt[str(k)]["wrist_step_m"] * 1e3 for k in ks], **kw)
    axs[1].plot(ks, [qF["per_transition"][str(k)]["control_wrist_step_m"] * 1e3 for k in ks], color=SERIES["U"], lw=1.5, ls=":", label=VLABEL["U"].strip())
    for ax in axs:
        for a_, b_ in r["pin_runs"]:
            ax.axvspan(a_ - 0.5, b_ + 0.5, color="#2a78d6", alpha=0.08, lw=0)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False, ncol=2)
    axs[0].set_ylabel("compensation step\n(mm/frame, vs U)")
    axs[1].set_ylabel("raw wrist step\n(mm/frame)")
    axs[1].set_xlabel("Action frame index (transition k-1 -> k); shaded = hard pin")
    axs[0].set_title(f"{rid}: edit-induced (top) vs raw output (bottom) wrist motion, Blender", fontsize=9)
    fig.tight_layout()
    fig.savefig(path_steps)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
    win = sorted(int(k) for k in qF["flex_deg"])
    ax.plot(win, [qF["control_flex_deg"][str(k)] for k in win], color=SERIES["U"], lw=1.5, ls=":", label=VLABEL["U"].strip())
    for v in ["F", "X"] + list(FACTORS):
        q = r["variants"].get(v, {}).get("quality")
        if q:
            ax.plot(win, [q["flex_deg"][str(k)] for k in win], color=SERIES[v], lw=2, ls=STYLE[v], marker=MARK[v], ms=3, label=VLABEL[v].strip())
    ax.axhline(TB.STRAIGHT_FLEX_DEG, color="#52514e", lw=0.8, ls=":")
    ax.text(win[0], TB.STRAIGHT_FLEX_DEG + 0.5, "5 deg W118 diagnostic reference (not a rule)", fontsize=7, color="#52514e")
    for a_, b_ in r["pin_runs"]:
        ax.axvspan(a_ - 0.5, b_ + 0.5, color="#2a78d6", alpha=0.08, lw=0)
    ax.set_ylabel("absolute elbow flex (deg)")
    ax.set_xlabel("Action frame index; shaded = hard pin")
    ax.set_title(f"{rid}: absolute elbow flex (Blender)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, ncol=4)
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
        if r["variants"]["F"].get("quality") and r["variants"].get("G0F0", {}).get("quality"):
            draw_charts(out / "charts" / f"steps_{rid}.png", out / "charts" / f"flex_{rid}.png", rid, A)
    jobs = {"real": [], "syn": []}
    sel = []
    for rid in REVIEW_SET:
        rec = Bb["records"][rid]
        lo, hi = A["requests"][rid]["variants"]["F"]["quality"]["window"]
        frames = list(range(lo, hi + 1))
        sel.append((rid, frames))
        tgt = rec["request"]["constraints"][0]["target"]
        side = "real" if rec["side"] == "real" else "syn"
        for v in VARIANTS_SHOWN:
            st = rec["U"] if v == "U" else rec["F"] if v == "F" else rec["X"] if v == "X" else rec["factors"][v]["state"]
            yi = rec["factors"][v]["info"]["frames"] if v in FACTORS else {}
            for k in frames:
                mk = []
                tk = [c["target"] for c in rec["request"]["constraints"] if k in c["frames"]]
                if tk:
                    mk.append({"name": "target", "position": tk[0], "color": (0.1, 0.9, 0.2, 1), "radius": 0.013})
                if str(k) in yi:
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
    dump(out / "render_selection.json", {"selection": sel, "variants": VARIANTS_SHOWN, "labels": VLABEL})
    return 0


def compose(out, sel, Bb, A):
    from PIL import Image, ImageDraw
    (out / "frames").mkdir()
    for rid, frames in sel:
        rec = Bb["records"][rid]
        d = out / ("real" if rec["side"] == "real" else "syn")
        ims = {(v, k): Image.open(d / f"{rid}__{v}__f{k:03d}.png").convert("RGB") for v in VARIANTS_SHOWN for k in frames}
        w, h = ims[("U", frames[0])].size
        cells = {v: (A["requests"][rid]["variants"][v]["display_spec_proposed"]["cells"] if v != "U" else {}) for v in VARIANTS_SHOWN}
        status = {v: A["requests"][rid]["variants"][v].get("commit", {}).get("decision", "") for v in VARIANTS_SHOWN if v not in ("U", "F")}
        band = 10
        pin = set(rec["pin_frames"])

        def tile(v, k, scale=1):
            im = ims[(v, k)].copy()
            if scale != 1:
                im = im.resize((w // scale, h // scale))
            c = cells[v].get(str(k), "UNCHANGED") if v != "U" else "UNCHANGED"
            t = Image.new("RGB", (im.size[0], im.size[1] + band), COL[c])
            t.paste(im, (0, 0))
            return t
        gif = []
        rdir = out / "frames" / rid
        rdir.mkdir()
        for k in frames:
            fr = Image.new("RGB", (w * len(VARIANTS_SHOWN), h + band + 30), (252, 252, 251))
            dd = ImageDraw.Draw(fr)
            for i, v in enumerate(VARIANTS_SHOWN):
                fr.paste(tile(v, k), (i * w, 30))
                c = cells[v].get(str(k), "") if v != "U" else ""
                dd.text((i * w + 4, 2), f"{v} f{k}{' PIN' if k in pin else ''}", fill=(11, 11, 11))
                dd.text((i * w + 4, 15), (c + (" " + status[v] if status.get(v) and status[v] != "ACCEPT" else ""))[:44],
                        fill=(180, 30, 30) if status.get(v) == "REJECT" else (82, 81, 78))
            fr.save(rdir / f"f{k:03d}.png")
            gif.append(fr)
        gif[0].save(out / f"anim_{rid}.gif", save_all=True, append_images=gif[1:], duration=150, loop=0)
        # filmstrip: half-size tiles, up to 14 columns
        cols = frames if len(frames) <= 14 else [frames[int(round(i * (len(frames) - 1) / 13))] for i in range(14)]
        sw, sh, gap = w // 2, h // 2 + band, 10
        sheet = Image.new("RGB", (250 + sw * len(cols), 46 + (sh + gap) * len(VARIANTS_SHOWN)), (252, 252, 251))
        dr = ImageDraw.Draw(sheet)
        dr.text((4, 4), f"{rid}: one row per variant; band below each tile = its actual footprint class (blue pin, orange automatic edit, "
                        f"grey unchanged); green = pin target, magenta = variant wrist target", fill=(11, 11, 11))
        for j, k in enumerate(cols):
            dr.text((250 + j * sw + 4, 26), f"f{k}{' PIN' if k in pin else ''}", fill=(42, 120, 214) if k in pin else (82, 81, 78))
            for i, v in enumerate(VARIANTS_SHOWN):
                sheet.paste(tile(v, k, 2), (250 + j * sw, 46 + i * (sh + gap)))
        for i, v in enumerate(VARIANTS_SHOWN):
            dr.text((4, 46 + i * (sh + gap) + sh // 2 - 6), VLABEL[v][:40], fill=(11, 11, 11))
        sheet.save(out / f"filmstrip_{rid}.png")


def write_index(out, A, sel):
    rows = []
    for rid, r in A["requests"].items():
        cells = "".join(f"<td>{r['variants'][f]['decision']['decision'] if f in r['variants'] else ''}</td>" for f in FACTORS)
        fx = r["comparisons"].get("factorial", {}).get("values", {})
        b1 = "".join(f"<td>{(fx.get(f) or {}).get('B1_boundary_wrist_comp_step_peak_m', '')}</td>" for f in FACTORS)
        rows.append(f"<tr><td>{rid}</td><td>{r['side']}</td><td>{r['role']}</td><td>{r['pin_runs']}</td>{cells}{b1}"
                    f"<td><a href='timelines/timeline_{rid}.png'>timeline</a></td></tr>")
    films = "".join(f"<h3>{rid}</h3><img src='timelines/timeline_{rid}.png'><br><img src='charts/steps_{rid}.png'><br>"
                    f"<img src='charts/flex_{rid}.png'><br><img src='filmstrip_{rid}.png'><br><img src='anim_{rid}.gif'>" for rid, _ in sel)
    hdr = "".join(f"<th>{f}</th>" for f in FACTORS) + "".join(f"<th>B1 {f} (m)</th>" for f in FACTORS)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>W120 matched factor attribution (scratch, not product UI)</title>
<style>body{{font-family:sans-serif;background:#fcfcfb;color:#0b0b0b;margin:20px}} td,th{{border-bottom:1px solid #ddd;padding:3px 6px;font-size:12px}}
img{{max-width:100%;margin:6px 0}}</style></head><body>
<h1>W120 G (wrist target) x F (flex floor) variants vs X_r / F_r: actual Blender footprint and rig motion (scratch diagnostic)</h1>
<p>Timelines are drawn from Blender-evaluated output diff vs U and independently truth-checked. Hatched = budget or REJECTED.</p>
<table><tr><th>request</th><th>side</th><th>role</th><th>pin runs</th>{hdr}<th></th></tr>{''.join(rows)}</table>
<h2>Review set (U / F_r / X_r / G0F0 / G0F1 / G1F0 / G1F1)</h2>{films}</body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------------------------- decide ---
def states_equal(f1, f2):
    a, b = load(f1)["states"], load(f2)["states"]
    return a == b


def cmd_decide(a):
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    pf = load(Path(decl["preflight_dir"]) / "preflight.json")
    cal = load(Path(decl["calibration_dir"]) / "calibration.json")
    A = load(Path(a.assess_dir) / "assess_report.json")
    b1, b2, e1, e2 = Path(a.build_dir), Path(a.build_dir2), Path(a.eval_dir), Path(a.eval_dir2)
    Bb, B2 = load(b1 / "build_report.json"), load(b2 / "build_report.json")
    VR = load(a.visual_review)
    Rq, recs = A["requests"], Bb["records"]
    g, det = {}, {}
    pop = [r for r, x in recs.items() if x["role"] != "negative"]
    valid = [r for r in pop if not recs[r].get("contract_invalid")]
    g["V1_authentication"] = not pf["stop"]
    g["V2_parent_reproduction"] = pf["parent_rebuild"]["canonical_equal"] and pf["parent_blender"]["equals_w105_fixed_digest"] and \
        pf["parent_blender"]["frames_bit_equal_w119_preflight"] and all(pf["syn_probe_parity_w119"].values()) and pf["syn_rest_equal_w119"]
    det["V3"] = {"cal_mismatch": cal["w119_frozen_controls_mismatch"], "b1": Bb["frozen_controls_mismatch"], "b2": B2["frozen_controls_mismatch"],
                 "parity": [cal["w118_reproduction"]["ok"], Bb["w118_reproduction"]["ok"], B2["w118_reproduction"]["ok"]],
                 "X_r_not_equal_w118_X": [r for r in valid if not recs[r]["X_r_equals_w118_X"]]}
    g["V3_baseline_reproduction"] = not det["V3"]["cal_mismatch"] and not det["V3"]["b1"] and not det["V3"]["b2"] and all(det["V3"]["parity"]) \
        and not det["V3"]["X_r_not_equal_w118_X"]
    cr = cal["contamination_reproduction"]
    g["V4_contamination_reproduced_not_repaired"] = cr["reproduced_equals_w118_V4_list"] and cr["quality_window_equals_w118_worklog_six"] and \
        cr["all_out_of_pin_on_siblings"] and A["decisions_after"]["w118"]["sha256"] == W9.W118_DECISION_SHA and \
        A["decisions_after"]["w119"]["sha256"] == W119_DECISION_SHA
    iso_bad = [r for r in valid if not (recs[r]["isolation_values_F_r"]["ok"] and Rq[r]["variants"]["F"]["isolation_blender"]["ok"])]
    eq_bad = [r for r in valid if not (recs[r]["F_r_equals_frozen_single_frame_resolver"]["values_bit_equal"]
                                       and recs[r]["F_r_equals_frozen_single_frame_resolver"]["statuses_equal"])]
    det["V5"] = {"isolation_failed": iso_bad, "resolver_equality_failed": eq_bad,
                 "contaminated_windows": {r: {"F_r_outside_pin": Rq[r]["variants"]["F"]["isolation_blender"]["changed_outside_pin"],
                                              "F_shared_hist_outside_pin_runs": Rq[r]["variants"].get("F_shared_hist", {}).get("isolation_blender", {}).get("runs_outside_pin")}
                                          for r in CONTAMINATED}}
    g["V5_clean_isolated_baselines"] = not iso_bad and not eq_bad
    unfair, cross = [], []
    for r in valid:
        for f in FACTORS:
            if not Rq[r]["variants"][f]["fairness"]["ok"]:
                unfair.append(f"{r}/{f}")
        ci = FB.cross_factor_identity({f: recs[r]["factors"][f]["fair"] for f in FACTORS})
        if not ci["ok"]:
            cross.append({r: ci["errors"]})
    za = all(recs["Z_A"]["factors"][f]["info"]["budget_frames"] == recs["Z_A"]["pin_frames"] for f in FACTORS)
    det["V6"] = {"unfair": unfair, "cross_factor": cross, "Z_A_budget_is_pin_for_all": za}
    g["V6_factor_matching"] = not unfair and not cross and za
    ip = cal["instrument_preflight"]
    g["V7_predeclaration_and_instrument"] = decl["outcomes_inspected"] is False and S.sha_file(Path(decl["calibration_dir"]) / "frozen_controls.json") == \
        decl["frozen_controls_sha256"] and A["declaration_sha256"] == a.declaration_sha and ip["all_ok"] and cal["negative_expectations"]["all_ok"] \
        and Bb["code_blobs"] == decl["code_blobs"] and B2["code_blobs"] == decl["code_blobs"]
    det["V7"] = {"instrument_all_ok": ip["all_ok"], "precedence_n": ip["precedence_exhaustive"]["n"],
                 "precedence_mismatches": ip["precedence_exhaustive"]["n_mismatches"], "code_blobs_equal": Bb["code_blobs"] == decl["code_blobs"]}
    det["V8"] = {"missing": A["evaluated_states"], "syn_fk": cal["syn_fk_max_error_m"]}
    g["V8_independent_blender_evaluation"] = not A["evaluated_states"]["real_missing"] and not A["evaluated_states"]["syn_missing"] and \
        cal["syn_fk_max_error_m"] <= 1e-5
    miss = []
    for rid in decl["population"]["all_requests"]:
        if rid not in Rq:
            miss.append(rid)
            continue
        for v in ["F", "X"] + list(FACTORS):
            if "display_spec_proposed" not in Rq[rid]["variants"].get(v, {}):
                miss.append(f"{rid}/{v}")
        if not recs[rid].get("contract_invalid"):
            for v in ["X"] + list(FACTORS):
                if Rq[rid]["variants"][v]["residual_m"]["n"] < len(recs[rid]["pin_frames"]):
                    miss.append(f"{rid}/{v} pins")
    det["V9"] = miss
    g["V9_population_complete"] = not miss
    same_states = all((b1 / f).read_bytes() == (b2 / f).read_bytes() for f in ("real_states.json", "syn_states.json"))
    strip = lambda i: {k: v for k, v in i.items() if k != "wall_s"}  # noqa: E731
    same_info = all(strip(Bb["records"][r]["factors"][f]["info"]) == strip(B2["records"][r]["factors"][f]["info"])
                    for r in recs for f in Bb["records"][r].get("factors", {}))
    same_ev = all(states_equal(e1 / f, e2 / f) for f in ("real_eval.json", "syn_eval.json"))
    det["V10"] = {"states_identical": same_states, "infos_identical": same_info, "evaluations_identical": same_ev}
    g["V10_repeatability"] = same_states and same_info and same_ev
    tm = [pf["timing"], pf["syn_timing"]]
    g["V11_timing_semantics"] = all(st["interpolations"] == ["CONSTANT"] and st["blender_frame_equals_index_minus_f0_plus_1"]
                                    and all(s["half_equals_frame"] and s["late_equals_frame"] for s in st["subframe"].values())
                                    for t in tm for st in t["states"].values())
    par_bad, par_rep = [], {}
    for r in valid:
        h = Rq[r]["comparisons"].get("w119_Y_hist")
        par_rep[r] = None if h is None else {k: h[k] for k in ("frames_bit_equal_G1F1", "U_frames_bit_equal_w119", "w119_Y_value_digest", "G1F1_value_digest")}
        if r == "Z_A":
            continue
        if h is None or not (h["frames_bit_equal_G1F1"] and h["w119_Y_value_digest"] == h["G1F1_value_digest"]):
            par_bad.append(r)
    det["V12"] = {"failed": par_bad, "per_request": par_rep, "Z_A_w119_Y_decision": Rq["Z_A"]["comparisons"].get("w119_Y_hist", {}).get("w119_Y_decision")}
    g["V12_g1f1_parity"] = not par_bad
    wrong = []
    for rid, (src, f, desc, cond, want, reasons) in NEG.items():
        x = Rq[rid]["variants"][f]
        d = x["decision"]
        ok = [d["decision"], d["failure_class"]] == want and all(rs in d["reasons"] for rs in reasons)
        info = recs[rid]["factors"][f]["info"]
        if rid == "NG_GUIDE_UNREACH":
            sh = [v["to_guide_m"] for v in x["guide_residuals"].values()]
            ok = ok and info["status_counts"]["GUIDE_UNREACHABLE_OUTER"] == len(info["frames"]) > 0 and min(sh) >= 0.05 - 1e-6
        if rid == "NG_AMBIG":
            ok = ok and info["flag_counts"]["BEND_PLANE_AMBIGUOUS"] == len(info["frames"]) > 0
        if rid == "NG_FLIP":
            ok = ok and info["flag_counts"]["BRANCH_FLIP_VS_PREVIOUS"] >= 1 and len(x.get("quality", {}).get("introduced_flip_transitions", [])) >= 1
        if rid in ("NG_SHARED", "NG_SIBLING"):
            ok = ok and not recs[rid]["isolation_values_F_r"]["ok"] and not Rq[rid]["variants"]["F"]["isolation_blender"]["ok"]
        if not ok:
            wrong.append({"req": rid, "got": d, "expected": [want, reasons]})
    for dc, r in A["display_controls"].items():
        if not r.get("executed") or r["check"]["ok"]:
            wrong.append({"req": dc, "got": r})
    det["V13"] = wrong
    g["V13_negative_controls"] = not wrong
    # ---- C
    c1 = []
    for rid, r in Rq.items():
        for v in ["X"] + list(FACTORS):
            x = r["variants"].get(v)
            if x is None or x["decision"]["decision"] != "ACCEPT" or (r["role"] == "negative" and v == "X"):
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
        xp = r["variants"]["X"]["planned_frames"]
        for f in FACTORS:
            y = r["variants"][f]
            if not set(y["planned_frames"]) <= set(y["budget_frames"]) or (not recs[rid].get("contract_invalid") and y["planned_frames"] != xp):
                c2.append(f"{rid}/{f}/planned")
        for v in ["X"] + list(FACTORS):
            cm = r["variants"][v].get("commit")
            if not cm or not (r["variants"][v]["display_check_result"]["ok"] and cm["preview_id_equals_candidate"] and cm["head_consistent"]):
                c2.append(f"{rid}/{v}/result")
    det["C2"] = c2
    g["C2_truthful_footprint"] = not c2
    atom = []
    for rid in valid:
        r = Rq[rid]
        for f in FACTORS:
            y = r["variants"][f]
            if not y["frame_pass"]:
                continue
            fp = y["footprint"]
            allpass = all(y["frame_pass"].values()) and set(recs[rid]["F_statuses"].values()) <= {"SOLVED", "NO_CHANGE_NEEDED", "NOT_APPLICABLE"} \
                and set(fp["frames_changed"]) <= set(y["planned_frames"]) and set(fp["channel_frames"]) <= set(recs[rid]["auth"]) and \
                recs[rid]["factors"][f]["info"]["request_status"] == "SOLVED" and y["fairness"]["ok"] and r["variants"]["F"]["isolation_blender"]["ok"]
            if (y["decision"]["decision"] == "ACCEPT") != bool(allpass):
                atom.append(f"{rid}/{f}")
    lin = A["lineage"]
    refs_ok = all(e.get("head_after") == e.get("head_before") for e in lin["store_log"] if e.get("op") == "commit" and e.get("decision") == "REJECT")
    steps = {e["step"]: e for side in lin["programmes"].values() for e in side}
    stale = steps.get("S3", {}).get("reasons") == ["STALE_PARENT"] and steps.get("R3", {}).get("reasons") == ["STALE_PARENT"]
    acc = steps.get("S1", {}).get("decision") == "ACCEPT" and steps.get("R1", {}).get("decision") == "ACCEPT"
    rv = lin["revert_blender"]
    revert_ok = bool(rv.get("real") and rv.get("synthetic")) and all(all(v for k, v in x.items() if k != "restores") for x in rv.values()) \
        and steps.get("S4", {}).get("values_equal_restored") is True and steps.get("R4", {}).get("values_equal_restored") is True
    audit_ok = all(all(e["verifies"] for e in ch) for ch in lin["lineage_audit"].values())
    det["C3"] = {"atomicity": atom, "refs_unchanged_on_reject": refs_ok, "stale": stale, "accept_executed": acc, "revert_ok": revert_ok,
                 "audit_ok": audit_ok, "atomic_heads_unchanged": [steps.get(s, {}).get("head_unchanged") for s in ("S2", "R2")]}
    g["C3_atomicity_lineage_reversal"] = not atom and refs_ok and stale and acc and revert_ok and audit_ok and all(det["C3"]["atomic_heads_unchanged"])
    g["C4_source_immutability"] = A["source_guard"]["unchanged"] and all(m["ok"] for m in A["manifests_after"].values()) and \
        all(A["parent_files_verify"].values()) and Bb["source_guard"]["unchanged"] and B2["source_guard"]["unchanged"]
    c5 = []
    for rid, r in Rq.items():
        if r["role"] == "negative":
            continue
        for f in FACTORS:
            y = r["variants"][f]
            if y["decision"]["decision"] != "ACCEPT":
                continue
            col = y.get("collateral", {})
            cz = all(v["max_vs_U_m"] == 0.0 for k, v in col.items() if isinstance(v, dict))
            if not (y["user_edit_and_object_equal_U"] and cz and col.get("root_matrix_equal_U", True)):
                c5.append(f"{rid}/{f}")
    det["C5"] = c5
    g["C5_user_edit_and_collateral"] = not c5
    c6 = [f"{r}/{f}" for r in pop for f in FACTORS if recs[r]["factors"][f]["info"].get("fallback") is not None
          or recs[r]["factors"][f]["info"].get("request_status") == "G0_REFERENCE_MISSING"]
    det["C6"] = c6
    g["C6_no_fallback"] = not c6
    # ---- mechanism evaluation (frozen CASE rule)
    evalset_syn = set(SYN_PRIMARY)
    evalset_real = {r for r in pop if recs[r]["side"] == "real" and recs[r]["role"] == "evaluation"}
    fx = {r: Rq[r]["comparisons"]["factorial"] for r in pop if "factorial" in Rq[r]["comparisons"]}
    mech = {}
    for M, pc in FB.PRIMARY_CONTRAST.items():
        cc = FB.CONDITIONAL_CONTRAST[M]
        imp = sorted(r for r, x in fx.items() if x["contrasts"][pc]["improves"]["meaningful_improvement"])
        imp_c = sorted(r for r, x in fx.items() if x["contrasts"][cc]["improves"]["meaningful_improvement"])
        worse = sorted(r for r, x in fx.items() if x["contrasts"][pc]["reverse"]["meaningful_improvement"])
        defects = sorted(r for r, x in fx.items() if (r in evalset_syn or r in evalset_real)
                         and (x["contrasts"][pc]["new_flips"] or x["contrasts"][pc]["new_near_straight"]))
        vis = [r for r, x in VR.get("requests", {}).items() if recs.get(r, {}).get("side") == "real" and r in imp
               and x.get("mechanism_visible", {}).get(M) is True and x.get("motion_inspected") is True]
        m = {"primary_improved": imp, "conditional_improved": imp_c, "primary_material_worsening": worse, "primary_new_defect_requests": defects,
             "syn": bool(set(imp) & evalset_syn), "rh1": bool(set(imp) & set(RH1)), "rh2": bool(set(imp) & set(RH2)), "clean": not defects,
             "vis": bool(vis), "vis_requests": vis,
             "any_eval_improvement": bool((set(imp) | set(imp_c)) & (evalset_syn | evalset_real))}
        m["A"] = m["syn"] and m["rh1"] and m["rh2"] and m["clean"] and m["vis"]
        mech[M] = m
    interp = {r: r in fx for r in ("H_RH1_none", "H_RH2_RBpos", "H_G")}
    det["mechanisms"] = mech
    det["interpretable"] = interp
    det["interaction_material"] = sorted(r for r, x in fx.items() if x["interaction"]["B1_material"])
    V = [k for k in g if k.startswith("V") and not g[k]]
    Cg = [k for k in g if k.startswith("C") and not g[k]]
    if V:
        case = "D"
    elif Cg:
        case = "C"
    elif any(m["A"] for m in mech.values()):
        case = "A"
    elif all(interp.values()) and any(m["any_eval_improvement"] for m in mech.values()):
        case = "B"
    else:
        case = "C"
    res = {"schema": "w120_decision_v1", "declaration_sha256": a.declaration_sha, "visual_review_sha256": S.sha_file(a.visual_review),
           "gates": g, "failed": {"V": V, "C": Cg}, "case": case, "detail": det}
    dump(out / "decision.json", res)
    print(json.dumps({"case": case, "failed": res["failed"], "mech": {M: {k: v for k, v in m.items() if k != "vis_requests"} for M, m in mech.items()},
                      "interp": interp}, indent=1, default=str))
    return 0


# --------------------------------------------------------------------------------------------- manifest ---
def cmd_manifest(a):
    base = Path(a.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != "evidence_manifest.json" and not ({"store", "lineage_store"} & set(p.relative_to(base).parts)):
            files[p.relative_to(base).as_posix()] = {"sha256": S.sha_file(p), "bytes": p.stat().st_size}
    dump(base / "evidence_manifest.json", {"schema": "w120_evidence_manifest_v1", "declaration_sha256": a.declaration_sha,
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
        if name == "build":
            p.add_argument("--smoke", action="store_true", help="crash-only smoke run: skip the code-blob check (never used for evidence)")
    p = sub.add_parser("manifest"); p.add_argument("--base", required=True); p.add_argument("--declaration-sha", required=True)  # noqa: E702
    a = ap.parse_args()
    return {"preflight": cmd_preflight, "calibrate": cmd_calibrate, "declare": cmd_declare, "build": cmd_build, "evaluate": cmd_evaluate,
            "assess": cmd_assess, "render": cmd_render, "decide": cmd_decide, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
