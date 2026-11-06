#!/usr/bin/env python3
"""W118: automatic transition editing with explicit edit-footprint visibility (evidence only, NON-PRODUCTION).

Compares, on identical requests, rig and Blender evaluator:
  U   unpinned user edit (control)
  F   frozen W116 framewise WORLD_POSITION_PIN resolver
  T   frozen W117 temporal swivel solver (pin frames only)            [replayed with frozen W117 code]
  TE  frozen W117 fixed-margin (4) transition easing                  [replayed with frozen W117 code]
  X   W118 candidate: F hard-pin base + automatic bounded offset-layer transition editor (w118_transition_edit_base)

Stages (each refuses an existing output directory):
  preflight  identity, W113-W117 Worklogs, W116 + W117 manifest/declaration/decision authentication, inputs, parent rebuild
             + fresh Blender evaluation, synthetic parents (W117 A-D + held-out W118 E) probed in Blender, timing probe
  calibrate  frozen W117 build replay parity (pure Python, every state id), synthetic design of new W118 holds from
             parent/unpinned geometry only (no candidate exists)
  declare    frozen declaration -> hash
  build      W117 replay + new requests (U/F/T/TE) + X candidates + negative/tamper/known-footprint controls
  evaluate   Blender evaluation (shared independent evaluator: scripts/w116_blender_stages.py, unchanged)
  assess     decisions, actual footprints, display specs + independent display truth check, quality, comparisons,
             pre-commit preview -> commit -> result cards, lineage/revert (Blender), W117 parity, source guards
  render     timelines, wrist-step charts, preview/result cards, bone-proxy filmstrips + GIFs, index.html
  decide     frozen gates + recorded visual review -> CASE
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

BATCH = 118
INSTRUCTION_BLOB = "2468904ff2e087fcff65c09729014d1968e6936f"
STARTING_HEAD = "0dbcbbfcc97defd98061d9416a69dac7eae6cca1"
W117_COMPLETE_COMMIT = "01749bfec5543c251d7a91b5c629f1bef2c1181b"
BRANCH = "arch/single_frame_first"
LOCK = Path("C:/Users/dna10/animcv-automation/state/active_batch.json")
W116, W117 = W7.W116, Path("C:/Users/dna10/animcv-w117")
W117_MANIFEST_SHA = "9f90f91aa9980c366ee3185e8e0ee6939db974b0f5d9517ab7cdc6bdd9ee22d8"
W117_DECL_SHA = "fab6208100f2ed9074ea125ff9587aff83b3d62ae5ba2cbb3f7c9d9a3ed95bc2"
W117_DECISION_SHA = "f0d707513c8079406d0fec2d84eb3593d9396db1464299ddc59d545a591b4e8e"
HIST = dict(W7.HIST) | {117: "docs/117_WORKLOG_SEMANTIC_WORLD_PIN_TEMPORAL_COHERENCE.md"}
UPPER, LOWER, ARM = W7.UPPER, W7.LOWER, W7.ARM
REAL_FRAMES = list(range(199))
SYN_FRAMES = W7.SYN_FRAMES

# ---------------------------------------------------------------------------------------- new W118 synthetic design ---
TRAJ_E = {"role": "held_out_evaluation_w118", "tw": ([0, 0, 0, 0], [10, 0.9, 1.0]), "bx": ([0, 3, 0, 0], [0, 0, 0]),
          "az": ([-28, 20, 0, 0], [12, 1.1, 0.4]), "ax": ([0, 0, 0, 0], [18, 0.7, 0.3]), "flex": ([60, -20, 0, 0], [15, 1.3, 0.2])}
# rid: (trajectory, edit, [(capture, lo, hi), ...], role)
NEW_SYN = {"H_E": ("E", "SYN_E8", [(16, 16, 26)], "evaluation_held_out"),
           "H_S0": ("B", "SYN_E8", [(0, 0, 6)], "evaluation_clip_start"),
           "H_END": ("A", "SYN_E8", [(39, 32, 39)], "evaluation_clip_end"),
           "DJ_B": ("B", "SYN_E8", [(6, 6, 10), (20, 20, 24)], "evaluation_disjoint"),
           "Z_A": ("A", "SYN_E8", [(14, 14, 25)], "zero_budget_control")}
NEW_REAL = {"DJ_R": ("RBpos", [(109, 109, 118), (154, 154, 181)], "evaluation_disjoint")}
# W117 requests that get an X candidate, by role
X_W117_EVAL_SYN = ["H_A", "H_B", "H_C"]
X_W117_EVAL_REAL_HOLDS = ["H_RH1_none", "H_RH1_RBneg", "H_RH1_RBpos", "H_RH2_RBpos", "H_RH2_none", "H_RH2_RBneg"]
X_W117_NEG = ["H_Cx", "N2", "N3", "N4", "N5", "N6", "N7"]
NEG_EXPECT_X = {
    "H_Cx": ["REJECT", "INFEASIBLE_REQUEST"], "N2": ["REJECT", "INFEASIBLE_REQUEST"], "N3": ["ACCEPT", "NONE"],
    "N4": ["REJECT", "UNSUPPORTED_BINDING"], "N5": ["REJECT", "UNSUPPORTED_BINDING"], "N6": ["REJECT", "INFEASIBLE_REQUEST"],
    "N7": ["REJECT", "CONTRACT_VIOLATION"],
    "NX1": ["REJECT", "CONTRACT_VIOLATION"], "NX2": ["REJECT", "CONTRACT_VIOLATION"], "NX3": ["REJECT", "CONTRACT_VIOLATION"],
    "NX4": ["REJECT", "CONTRACT_VIOLATION"], "NX5": ["REJECT", "SOLVER_FAILURE"], "NX6": ["REJECT", "SOLVER_FAILURE"],
    "RX1": ["REJECT", "CONTRACT_VIOLATION"], "RX5": ["REJECT", "SOLVER_FAILURE"], "RX6": ["REJECT", "SOLVER_FAILURE"],
}
NEG_DESC = {
    "NX1": "H_A X with upper +1 deg on frame 2 (outside the automatic budget)",
    "NX2": "H_A X values unchanged but the candidate-declared planned domain omits its outermost transition frame",
    "NX3": "H_A X with base (user-edit channel) +1 deg on a transition frame",
    "NX4": "H_A X with __object__ +1 deg on a transition frame",
    "NX5": "H_A X claims SOLVED but returns U on the pin frames (transition frames kept)",
    "NX6": "H_A X with lower +2 deg on pin frame 20 (single corrupted pin frame)",
    "RX1": "H_RH1_none X with upperarm_l +1 deg on frame 100 (outside the automatic budget)",
    "RX5": "H_RH1_none X claims SOLVED but returns U on the pin frames",
    "RX6": "H_RH1_none X with lowerarm_l +2 deg on pin frame 113",
}
DISPLAY_CONTROLS = {
    "DC1": "H_A X display: first out-of-pin changed frame relabelled UNCHANGED and removed from channel rows",
    "DC2": "DJ_B X display: disjoint changed runs collapsed to their single union envelope (gap frames shown as changed)",
    "DC3": "H_A X display: channel row 'lower' removed (incomplete channel coverage)",
    "DC4": "NX1 actual output displayed from the solver's intended planned domain only (timeline from intended margin)",
    "DC5": "H_RH1_none X display: one out-of-pin changed frame relabelled UNCHANGED (real rig)",
    "DC6": "H_Cx X (REJECTed) display labelled ACCEPTED (rejected candidate presented as applied)",
}
KNOWN_FP = {"request": "KF", "base_U": "U__A__SYN_E8", "changes": {"3": {"upper": 1.0}, "17": {"upper": 1.0, "lower": 0.5}}}
REPRESENTATIVE = {"syn": ["H_B", "H_E", "DJ_B"], "real": ["H_RH1_none", "H_RH2_RBpos", "DJ_R"]}
Q_SYN_W117 = ["H_A", "H_B", "H_C"]
Q_REAL_HOLDS = ["H_RH1_none", "H_RH1_RBneg", "H_RH1_RBpos", "H_RH2_RBpos", "DJ_R"]

dump, load, new_dir, git = C.dump, C.load, C.new_dir, C.git
blender = C.blender


def traj_state_E():
    vals = {c: [] for c in W7.SYN_CH}
    for f in SYN_FRAMES:
        tau = f / (W7.N_SYN - 1)
        r = {k: math.radians(W7.angle_at(TRAJ_E[k], tau)) for k in ("tw", "bx", "az", "ax", "flex")}
        vals["__object__"].append([0.0, 0.0, 0.0, 1.0])
        vals["base"].append(S.q_mul(S.q_axis_angle([0, 1, 0], r["tw"]), S.q_axis_angle([1, 0, 0], r["bx"])).tolist())
        vals["upper"].append(S.q_mul(S.q_axis_angle([0, 0, 1], r["az"]), S.q_axis_angle([1, 0, 0], r["ax"])).tolist())
        vals["lower"].append(S.q_axis_angle([0, 0, 1], r["flex"]).tolist())
    src = {"kind": "DECLARED_SYNTHETIC_TRAJECTORY", "trajectory": "E", "params": TRAJ_E, "rig_identity": "W117_SYN_ARM (known)",
           "video_subject_identity": "NOT_APPLICABLE"}
    return S.make_state(None, W7.SYN_SPEC["rig_id"], SYN_FRAMES, W7.SYN_CH, vals, {"kind": "ROOT_SYNTHETIC", "trajectory": "E"}, src)


def w117_manifest_check():
    m = load(W117 / "evidence_manifest.json")["files"]
    bad = [f for f, r in m.items() if not (W117 / f).exists() or S.sha_file(W117 / f) != r["sha256"]]
    sha = S.sha_file(W117 / "evidence_manifest.json")
    return {"manifest_sha256": sha, "n_files": len(m), "mismatched": bad, "ok": sha == W117_MANIFEST_SHA and not bad}


def guard():
    g = W7.guard()
    files = dict(g.files) | {"w117_declaration": W117 / "results/declare/declaration.json",
                              "w117_decision": W117 / "results/decision/decision.json", "w117_manifest": W117 / "evidence_manifest.json",
                              "w117_build_r1_report": W117 / "results/build_r1/build_report.json",
                              "w117_real_states": W117 / "results/build_r1/real_states.json",
                              "w117_syn_states": W117 / "results/build_r1/syn_states.json",
                              "w117_eval_e1_real": W117 / "results/eval_e1/real_eval.json",
                              "w117_eval_e1_syn": W117 / "results/eval_e1/syn_eval.json",
                              "w117_assess": W117 / "results/assess/assess_report.json"}
    return S.SourceGuard(files)


# ------------------------------------------------------------------------------------------------- preflight ---
def cmd_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"schema": "w118_preflight_v1", "stop": [], "runtime": C.runtime()}
    rep["head"], rep["instruction_blob"] = git("rev-parse", "HEAD"), git("rev-parse", "HEAD:docs/next_batch.md")
    rep["branch"], rep["status_porcelain"] = git("rev-parse", "--abbrev-ref", "HEAD"), git("status", "--porcelain")
    title = (ROOT / "docs/next_batch.md").read_text(encoding="utf-8").splitlines()[0]
    rep["instruction_title"] = title
    if rep["instruction_blob"] != INSTRUCTION_BLOB or not title.startswith("# BATCH 118"):
        rep["stop"].append("instruction is not the bound Batch 118 blob")
    if rep["head"] != STARTING_HEAD:
        rep["stop"].append("HEAD is not the bound starting commit")
    if rep["branch"] != BRANCH:
        rep["stop"].append("wrong branch")
    lock = load(LOCK) if LOCK.exists() else None
    rep["main_local_lock"] = lock
    if not lock or lock.get("batch") != BATCH or lock.get("instruction_blob") != INSTRUCTION_BLOB or lock.get("start_commit") != STARTING_HEAD:
        rep["stop"].append("Main-local lock is not this Batch 118 claim")
    hist = {}
    for n, wl in HIST.items():
        lines = (ROOT / wl).read_text(encoding="utf-8").rstrip().splitlines()
        hist[n] = {"marker": lines[-2:] == ["AUTOMATION_BATCH_STATUS: COMPLETE", f"BATCH_NUMBER: {n}"],
                   "blob_at_start": git("rev-parse", f"{STARTING_HEAD}:{wl}"), "blob_now": git("hash-object", str(ROOT / wl))}
        if not hist[n]["marker"] or hist[n]["blob_at_start"] != hist[n]["blob_now"]:
            rep["stop"].append(f"W{n} Worklog record not intact")
    rep["historical_worklogs"] = hist
    try:
        git("merge-base", "--is-ancestor", W117_COMPLETE_COMMIT, "HEAD")
        rep["w117_complete_commit_is_ancestor"] = True
    except Exception:
        rep["w117_complete_commit_is_ancestor"] = False
        rep["stop"].append("W117 COMPLETE commit is not an ancestor of HEAD")
    rep["w117_worklog_case_c"] = "CASE C" in (ROOT / HIST[117]).read_text(encoding="utf-8").split("## 2.")[0]
    rep["w116_manifest"] = W7.w116_manifest_check()
    rep["w117_manifest"] = w117_manifest_check()
    for k in ("w116_manifest", "w117_manifest"):
        if not rep[k]["ok"]:
            rep["stop"].append(f"{k} does not re-verify")
    rep["w116_declaration_sha256"] = S.sha_file(W116 / "results/declare/declaration.json")
    rep["w116_decision_sha256"] = S.sha_file(W116 / "results/decision/decision.json")
    rep["w116_case"] = load(W116 / "results/decision/decision.json")["case"]
    rep["w117_declaration_sha256"] = S.sha_file(W117 / "results/declare/declaration.json")
    rep["w117_decision_sha256"] = S.sha_file(W117 / "results/decision/decision.json")
    rep["w117_case"] = load(W117 / "results/decision/decision.json")["case"]
    if rep["w116_declaration_sha256"] != W7.W116_DECL_SHA or rep["w116_decision_sha256"] != W7.W116_DECISION_SHA or rep["w116_case"] != "B":
        rep["stop"].append("W116 declaration/decision/CASE not the frozen ones")
    if rep["w117_declaration_sha256"] != W117_DECL_SHA or rep["w117_decision_sha256"] != W117_DECISION_SHA or rep["w117_case"] != "C":
        rep["stop"].append("W117 declaration/decision/CASE not the frozen ones")
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
    w117_v0 = load(W117 / "results/preflight/real_parent_state.json")
    rep["parent_rebuild"] = {"state_id": v0["state_id"], "w117_state_id": w117_v0["state_id"],
                             "canonical_equal": S.canon(v0) == S.canon(w117_v0), "verifies": S.verify_state(v0)}
    if not rep["parent_rebuild"]["canonical_equal"]:
        rep["stop"].append("rebuilt parent differs from the W117/W116 parent")
    dump(out / "real_parent_state.json", v0)
    C.states_file(out / "real_parent_states.json", {"v0": v0}, "w105")
    rest = C.INPUTS["w105_rig_rest"][0]
    blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest", rest,
            "--states", out / "real_parent_states.json", "--out", out / "real_parent_eval.json", log=out / "real_parent_eval.log")
    blender("scripts/w117_blender_timing.py", "--", "--rig", C.INPUTS["w105_rig_skinned_fbx"][0], "--rest", rest, "--states",
            out / "real_parent_states.json", "--out", out / "real_timing.json", "--sample-frames", "1,109,118,153,154,197",
            log=out / "real_timing.log")
    ev = load(out / "real_parent_eval.json")["states"]["v0"]
    w117e = load(W117 / "results/preflight/real_parent_eval.json")["states"]["v0"]
    rep["parent_blender"] = {"value_digest": ev["value_digest"], "equals_w105_fixed_digest": ev["value_digest"] == rep["w105_audit"]["value_digest"],
                             "frames_bit_equal_w117_preflight": ev["frames"] == w117e["frames"]}
    if not (rep["parent_blender"]["equals_w105_fixed_digest"] and rep["parent_blender"]["frames_bit_equal_w117_preflight"]):
        rep["stop"].append("fresh Blender evaluation of the parent does not reproduce W105/W117")
    rep["timing"] = load(out / "real_timing.json")
    # synthetic: W117 spec + parents A-D (authenticated) + held-out parent E
    spec_w117 = load(W117 / "results/preflight/syn_spec.json")
    dump(out / "syn_spec.json", spec_w117)
    rep["syn_spec_equal_w117"] = (out / "syn_spec.json").read_bytes() == (W117 / "results/preflight/syn_spec.json").read_bytes()
    if not rep["syn_spec_equal_w117"] or spec_w117 != json.loads(json.dumps(W7.SYN_SPEC)):
        rep["stop"].append("synthetic spec differs from W117")
    parents = {}
    for n in ("A", "B", "C", "D"):
        st = load(W117 / f"results/preflight/syn_parent_P__{n}.json")
        rb = W7.traj_state(n)
        if S.canon(st) != S.canon(rb):
            rep["stop"].append(f"W117 synthetic parent {n} does not rebuild")
        parents[f"P__{n}"] = st
    parents["P__E"] = traj_state_E()
    for n, st in parents.items():
        dump(out / f"syn_parent_{n}.json", st)
    C.states_file(out / "syn_parent_states.json", parents, "synthetic")
    blender("scripts/w116_blender_stages.py", "--", "probe", "--rig", f"synthetic:{out / 'syn_spec.json'}", "--states",
            out / "syn_parent_states.json", "--out", out / "syn_probe.json", "--probe-channels", "base,upper,lower", log=out / "syn_probe.log")
    blender("scripts/w117_blender_timing.py", "--", "--rig", f"synthetic:{out / 'syn_spec.json'}", "--states",
            out / "syn_parent_states.json", "--out", out / "syn_timing.json", log=out / "syn_timing.log")
    sp, wp = load(out / "syn_probe.json"), load(W117 / "results/preflight/syn_probe.json")
    rep["syn_probe_parity_w117"] = {n: sp["states"][n]["frames"] == wp["states"][n]["frames"] for n in wp["states"]}
    rep["syn_rest_equal_w117"] = sp["rest"] == wp["rest"]
    if not all(rep["syn_probe_parity_w117"].values()) or not rep["syn_rest_equal_w117"]:
        rep["stop"].append("synthetic probe does not reproduce W117")
    rep["syn_timing"] = load(out / "syn_timing.json")
    rep["syn_parent_state_ids"] = {n: s["state_id"] for n, s in parents.items()}
    rep["source_guard"] = guard().before
    dump(out / "preflight.json", rep)
    print(json.dumps({"stop": rep["stop"], "parent_blender": rep["parent_blender"], "w117_manifest": rep["w117_manifest"]["ok"],
                      "syn_parity": rep["syn_probe_parity_w117"]}, indent=1))
    return 1 if rep["stop"] else 0


# ------------------------------------------------------------------------------------------- W117 replay ---
def w117_replay():
    """Frozen W117 build replayed with frozen code and the authenticated W117 oracle (no Blender)."""
    decl = load(W117 / "results/declare/declaration.json")
    B = W7.Build(decl, None)
    B.build_real_U()
    br = load(W117 / "results/build_r1/build_report.json")
    oracle = {f: {int(k): v for k, v in o.items()} for f, o in br["oracle_real"].items()}
    soracle = {int(k): v for k, v in br["oracle_syn_PF_C"].items()}
    B.build_syn()
    B.per_frame_syn(soracle)
    B.build_real(oracle)
    par = {"real_state_ids_equal": {n: s["state_id"] == br["real_states"].get(n) for n, s in B.real.items()},
           "syn_state_ids_equal": {n: s["state_id"] == br["syn_states"].get(n) for n, s in B.syn.items()},
           "same_state_names": set(B.real) == set(br["real_states"]) and set(B.syn) == set(br["syn_states"]),
           "statuses_equal": all(B.req[r]["candidates"][v]["statuses"] == br["requests"][r]["candidates"][v]["statuses"]
                                 for r in br["requests"] for v in br["requests"][r]["candidates"])}
    par["ok"] = par["same_state_names"] and par["statuses_equal"] and all(par["real_state_ids_equal"].values()) and \
        all(par["syn_state_ids_equal"].values())
    return B, par


def cmd_calibrate(a):
    pre = Path(a.preflight)
    out = new_dir(a.out_dir)
    pf = load(pre / "preflight.json")
    if pf["stop"]:
        raise SystemExit("STOP: preflight failed")
    t0 = time.time()
    _, par = w117_replay()
    rep = {"schema": "w118_calibration_v1", "w117_replay_parity": {k: (v if not isinstance(v, dict) else
                                                                      {"n": len(v), "all": all(v.values()),
                                                                       "mismatch": [n for n, x in v.items() if not x]})
                                                                  for k, v in par.items()}, "replay_wall_s": time.time() - t0,
           "scope": "replay parity + synthetic design from parent/unpinned geometry (Python FK, Blender-calibrated); no X exists"}
    probe = load(pre / "syn_probe.json")
    rig = W7.syn_rig(probe)
    L1, L2 = rig.world_len("upper"), rig.world_len("lower")
    rmin, rmax = S._limits_reach(L1, L2)
    worst = 0.0
    for n, st in probe["states"].items():
        P = load(pre / f"syn_parent_{n}.json")
        for i, fr in enumerate(st["frames"]):
            fk = rig.fk(S.frame_vals(P, i))
            for b in fr["world"]:
                for p in ("head", "tail"):
                    worst = max(worst, float(np.linalg.norm(rig.world(fk, b, p) - np.array(fr["world"][b][p]))))
    rep["syn_fk_max_error_m"] = worst
    design = {}
    for rid, (tr, edit, holds, role) in NEW_SYN.items():
        P = load(pre / f"syn_parent_P__{tr}.json")
        U = W7.syn_ctrl(rig, P, edit)
        Ss = {k: W7.fk_pts(rig, U[k], "upper", "lower") for k in SYN_FRAMES}
        recs = []
        for cap, lo, hi in holds:
            Wp = np.array(probe["states"][f"P__{tr}"]["frames"][cap]["world"]["lower"]["tail"])
            dist = [float(np.linalg.norm(Wp - Ss[k][0])) for k in range(lo, hi + 1)]
            recs.append({"capture": cap, "interval": [lo, hi], "min_slack_m": min(rmax - d for d in dist),
                         "min_inner_slack_m": min(d - rmin for d in dist),
                         "design_feasible_all": all(rmin - S.TOL_M < d <= rmax + S.TOL_M for d in dist),
                         "entry_offset_m": float(np.linalg.norm(Wp - Ss[lo][2])), "release_offset_m": float(np.linalg.norm(Wp - Ss[hi][2])),
                         "edit_wrist_move_min_m": min(float(np.linalg.norm(Ss[k][2] - W7.fk_pts(rig, S.frame_vals(P, k), "upper", "lower")[2]))
                                                      for k in range(lo, hi + 1))})
        plan, budget = XB.budget_plan([k for _, lo, hi in holds for k in range(lo, hi + 1)], SYN_FRAMES,
                                      0 if rid == "Z_A" else XB.M_MAX)
        design[rid] = {"trajectory": tr, "edit": edit, "role": role, "holds": recs, "budget_plan": plan, "budget_frames": budget}
    rep["design"] = design
    rep["design_ok"] = all(h["design_feasible_all"] for d in design.values() for h in d["holds"]) and \
        abs(design["H_E"]["holds"][0]["entry_offset_m"] - design["H_E"]["holds"][0]["release_offset_m"]) >= 0.005
    dump(out / "calibration.json", rep)
    print(json.dumps({"replay": rep["w117_replay_parity"], "fk": worst, "design_ok": rep["design_ok"],
                      "design": {r: [(h["min_slack_m"], h["entry_offset_m"], h["release_offset_m"]) for h in d["holds"]]
                                 for r, d in design.items()}}, indent=1, default=str))
    return 0


# --------------------------------------------------------------------------------------------- declare ---
GATES = {
    "V1_authentication": "preflight has no STOP: HEAD, blob, branch, lock; W113-W117 Worklogs intact with COMPLETE markers; W117 COMPLETE "
                         "commit is an ancestor; W116 and W117 manifests re-verify (every file); W116 (B) and W117 (C) declaration/decision "
                         "hashes and CASEs; every W102/W105/W115 input hash",
    "V2_parent_reproduction": "rebuilt parent canonical-equal to the W117 parent; fresh Blender evaluation has the W105 FIXED digest and is "
                              "bit-equal to the W117 preflight evaluation; synthetic A-D parents rebuild and probe bit-equal to W117",
    "V3_baseline_replay": "frozen W117 build replayed with frozen code reproduces every W117 state id and every resolver status "
                          "(U/F/T/TE, real and synthetic); the Blender evaluation of every replayed W117 state equals W117 eval_e1 (value "
                          "digest and all frames); W117 F/T/TE decisions recomputed with the frozen W117 assess equal the W117 assess",
    "V4_independent_footprint_evaluator": "footprint derived from Blender output only: U vs U is empty; known-footprint control KF "
                                          "(declared changes on frames 3 and 17) is recovered exactly (frames, channels, runs); F footprint "
                                          "is a subset of its pin frames on every request; synthetic FK <= 1e-5 m",
    "V5_meaningful_design": "new synthetic holds design-feasible; held-out H_E entry/release offsets differ by >= 0.005 m (Blender U); W117 "
                            "V5 design facts inherited (authenticated)",
    "V6_timing_semantics": "every keyframe CONSTANT; Blender frame = index - f0 + 1; sub-frame evaluation equals the keyed frame (real + "
                           "synthetic incl. E)",
    "V7_oracle_soundness": "no frame whose oracle class (Blender U) is INFEASIBLE passes the pin for any variant; the recomputed real/PF_C "
                           "oracle equals the authenticated W117 oracle",
    "V8_repeatability": "two builds byte-identical states files and identical X plans/statuses; two Blender evaluations identical",
    "V9_population_complete": "every declared request/variant has a Blender-evaluated footprint, display spec and (for contract-valid "
                              "requests) every pin frame evaluated",
    "C1_hard_pin_and_scope": "every ACCEPTed X: every pin frame Blender residual <= 1 mm; changed channels vs U only authorised arm "
                             "channels; actual changed frames within the automatic budget and within the candidate-declared planned "
                             "domain; no evaluated point outside the arm domain changed vs U; root matrix unchanged; bone lengths; limits",
    "C2_truthful_footprint": "for EVERY variant of EVERY request the display spec passes the independent display truth check against Blender "
                             "frames (cells, channel rows, runs, envelope, magnitudes, state id); every X planned domain is within its "
                             "budget; preview state id = committed candidate id; result label = commit decision; REJECTed X never shown "
                             "as applied and its ref head unchanged",
    "C3_negative_controls": "every declared negative / tamper control returns its declared decision and class; every display corruption "
                            "control DC1-DC6 FAILS the display truth check (and is REJECTed DISPLAY_FOOTPRINT_MISMATCH)",
    "C4_atomicity_lineage_reversal": "multi-frame X requests ACCEPT iff all pin + transition frames pass and footprint checks hold; NX6/RX6 "
                                     "REJECT with exactly one failing frame; refs unchanged on every REJECT; stale parent rejected; executed "
                                     "ACCEPT + exact revert (values byte-equal, Blender bit-equal) on both sides",
    "C5_source_immutability": "declared inputs, W116/W117 artefacts and parent files byte-unchanged across every stage; W116 and W117 "
                              "manifests re-verify after the run",
    "C6_no_new_hard_failure": "X ACCEPTs every contract-valid evaluation request whose pin frames are all oracle-FEASIBLE (including the "
                              "named regression control H_RH2_RBpos)",
    "C7_user_edit_and_collateral": "every ACCEPTed X keeps the user-edit channel and __object__ stored values bit-equal to U on every frame, "
                                   "and right wrist / neck / head / pelvis (synthetic: neck / base) exactly equal to U",
    "Q_syn": "X meaningful improvement over F (compare_x) on the held-out H_E AND on >= 2 of {H_A, H_B, H_C}; and no synthetic "
             "evaluation request with both F and X ACCEPTed shows any worsening flag",
    "Q_real": "X meaningful improvement over F on >= 3 of the 5 real hold requests {H_RH1_none, H_RH1_RBneg, H_RH1_RBpos, H_RH2_RBpos, "
              "DJ_R}; and no real evaluation request with both ACCEPTed shows any worsening flag",
    "QV": "the recorded (agent) visual review marks, for >= 1 synthetic and >= 1 real request counted as improved, the boundary "
          "improvement as visible, the X arm as not implausible-looking at proxy level, and the footprint display as readable",
}
CASE_RULE = ("D if any V fails. Else C if any C gate fails, or if Q_syn and Q_real both fail. Else A if Q_syn, Q_real and QV all pass. "
             "Else B.")
QUALITY_RULE = {
    "window": "union window [first pin run lo - M_MAX - 1, last pin run hi + M_MAX + 1] clipped to the clip; same for every variant",
    "B1": "peak boundary wrist COMPENSATION step |c_W(k) - c_W(k-1)|, c_W = W^X - W^U, over entry transitions [lo-M_MAX, lo] and release "
          "transitions [hi+1, hi+M_MAX+1] of every pin run (m / Action frame)",
    "improved": "B1: X <= 0.5 F and F - X >= 0.005 m",
    "worse": "any of: B1 X >= 2F and X - F >= 0.005 m; A1 introduced bend flips X > F; A2 D3 peak arm rotation-step excess over U X >= 2F "
             "and +5 deg/frame; A3 D2 peak elbow compensation step X >= 2F and +0.005 m; A4 introduced near-straight frames (flex < 5 deg "
             "where U >= 5 deg) X > F; A5 peak elbow compensation acceleration X >= 2F and +0.005 m/frame^2",
    "meaningful_improvement": "improved non-empty and worse empty",
    "costs_reported_not_gated": "mean/peak elbow deviation from U, peak upper-arm compensating rotation, peak flex change, extra frames "
                                "outside the pin, runs, per-bone changed frame counts (W117 S1 10 mm is NOT a gate in W118)",
    "eligibility": "requests with >= 1 out-of-pin frame in the window and both F and X ACCEPTed; whole-clip requests (no out-of-pin frame) "
                   "are reported as NO_TRANSITION_AVAILABLE (X equals F; not a quality gain); Z_A (zero budget) is a control, not evidence",
}


def cmd_declare(a):
    out = new_dir(a.out_dir)
    pf = load(Path(a.preflight) / "preflight.json")
    cal = load(Path(a.calibration) / "calibration.json")
    if pf["stop"] or not cal["w117_replay_parity"]["ok"] or not cal["design_ok"]:
        raise SystemExit("STOP: preflight/calibration failed")
    decl = {
        "schema": "w118_declaration_v1", "batch": BATCH, "outcomes_inspected": False,
        "preflight_dir": str(Path(a.preflight).resolve()), "calibration_dir": str(Path(a.calibration).resolve()),
        "preflight_sha256": S.sha_file(Path(a.preflight) / "preflight.json"),
        "calibration_sha256": S.sha_file(Path(a.calibration) / "calibration.json"),
        "baselines": {"w116": {"manifest": W7.W116_MANIFEST_SHA, "declaration": W7.W116_DECL_SHA, "case": "B (frozen)"},
                      "w117": {"manifest": W117_MANIFEST_SHA, "declaration": W117_DECL_SHA, "decision": W117_DECISION_SHA,
                               "case": "C (frozen)", "complete_commit": W117_COMPLETE_COMMIT}},
        "approved_policy_vs_experiment": {
            "policy": "Architecture Owner policy B: automatic edits outside the hard-pin interval are allowed if the modified interval is "
                      "explicitly and intuitively visible; hard pins never relaxed; user edit authoritative; source immutable; reversible",
            "experimental_parameters_not_policy": {"M_MAX": XB.M_MAX, "SIGMA_STEP_M": XB.SIGMA_STEP_M, "MATERIAL_M": XB.MATERIAL_M,
                                                   "DISPLAY_RES_MM": XB.DISPLAY_RES_MM, "DISPLAY_RES_DEG": XB.DISPLAY_RES_DEG}},
        "hard_pin": {"tolerance_m": S.TOL_M, "evaluator": "Blender scene world (w116_blender_stages evaluate, unchanged)",
                     "status": "W116 provisional diagnostic criterion, unchanged"},
        "candidate_X": {
            "hard_pin_base": "frozen W116 framewise analytic resolver values on the pin frames (identical to F on pin frames)",
            "transition_editor": "per authorised channel a pre-multiplied local rotation offset layer d = q_F(b) q_U(b)^-1 faded by "
                                 "w_j = 1 - smoothstep(j/(m+1)) on the m frames beyond each pin-run boundary b; q_X = slerp(I, d, w_j) q_U",
            "automatic_selection": "per side the smallest m in [0, available] whose predicted (Python FK) peak per-frame wrist "
                                   "compensation step <= SIGMA_STEP_M; else m = available (BUDGET_SATURATED)",
            "search_budget": "available = min(M_MAX, frames to the clip boundary, floor(gap/2) to the neighbouring pin run)",
            "zero_budget_control": "Z_A uses M_MAX = 0 (must equal F exactly)",
            "fallback": "none (no fallback exists; F-equal outputs are reported as such and not counted as gains)",
            "not_used": "the W117 T optimiser and its convergence criterion"},
        "footprint": {"actual": "Blender-evaluated candidate vs U: any stored F-curve value bit difference, any evaluated point or "
                                "matrix_world difference = changed frame; material = max point displacement > MATERIAL_M; runs = maximal "
                                "consecutive changed frames; envelope = [first, last] changed frame",
                      "planned": "candidate-declared editable frames = pin frames + chosen transition frames",
                      "budget": "pin frames + available transition frames (search budget)",
                      "display": "built from the actual footprint; independently checked against raw Blender frames; sub-resolution "
                                 "changes are shown, never hidden; time axis = integer Action frame index (no seconds)"},
        "quality_rule": QUALITY_RULE,
        "synthetic": {"w117_requests_with_X": X_W117_EVAL_SYN + ["W_PF_C", "S_PF_C_*", "NOOP_A", "H_D (development)"],
                      "new": NEW_SYN, "trajectory_E": TRAJ_E, "design": cal["design"], "known_footprint_control": KNOWN_FP},
        "real": {"w117_requests_with_X": X_W117_EVAL_REAL_HOLDS + ["W_RA_neg", "W_RB_neg", "W_RC_raise", "W_RA_pos", "W_RB_pos", "S_*"],
                 "new": NEW_REAL, "named_regression_control": "H_RH2_RBpos (W117 T/TE NON_CONVERGED; label preserved)",
                 "confounder": "source discontinuity 153->154 (parent wrist step 0.485 m) = RH2 entry transition; not an editing target",
                 "timebase": "source 50 fps, historical Action 30 fps, evaluation frame-indexed; not repaired"},
        "negative_controls": {"X": NEG_EXPECT_X, "descriptions": NEG_DESC, "display": DISPLAY_CONTROLS},
        "lineage": {"synthetic": ["S1 commit X of the first ACCEPTed among [H_A, H_B, H_C, H_E] on syn/main", "S2 NX6 on syn/atomic -> "
                                  "REJECT, head unchanged", "S3 X of the next request prepared against the old head -> STALE_PARENT",
                                  "S4 revert syn/main -> byte-equal values, Blender bit-equal"],
                    "real": ["R1 commit X of the first ACCEPTed among the real holds on real/main", "R2 RX6 on real/atomic -> REJECT",
                             "R3 another X prepared against v0 -> STALE_PARENT", "R4 H_RH2_none X on real/atomic (as evaluated)",
                             "R5 revert real/main -> byte-equal, Blender bit-equal to v0"],
                    "per_request": "every X is previewed (PROPOSED card with its state id), committed on its own ref, and shown as "
                                   "ACCEPTED (head = candidate id) or REJECTED (head = parent id, NOT applied)"},
        "render_rule": {"blender_filmstrips": REPRESENTATIVE, "variants": "U, F, TE (where defined), X", "timelines": "every X request",
                        "camera": "bone proxies, mesh hidden, fixed front orthographic camera centred on the first pin target"},
        "visual_review": "after render and before decide, the agent records visual_review.json; it is agent-only (not independent)",
        "gates": GATES, "case_rule": CASE_RULE,
        "predictions_not_gates": {
            "X_on_pin_frames": "identical to F (same hard-pin base); within-pin articulation (flex, bend flips inside holds) unchanged",
            "boundary": "large boundary offsets saturate the budget; B1 falls roughly by 1.5/(m+1) of the boundary offset",
            "cost": "elbow deviation from U grows on transition frames (mechanism of the edit, not an accident)",
            "H_RH2_RBpos": "X ACCEPTs (no optimiser), entry transition coincides with the 153->154 source jump"},
        "pre_declaration_disclosure": [
            "focused unit tests of w118_transition_edit_base on a Python-only test rig",
            "design probe of candidate hold intervals from parent/unpinned geometry only (reach slack, offsets) to choose H_E/H_S0/"
            "H_END/DJ_B; no X/F/T/TE was computed on any W118 request before this declaration",
            "preflight and calibration as recorded",
            "one crash-only smoke run of declare -> build -> evaluate -> assess -> render -> decide in a throwaway directory "
            "(C:/Users/dna10/animcv-w118-smoke) with stdout discarded unread; every stage exited 0, only exit codes were read; the "
            "directory was deleted; no code or parameter was changed after it"],
    }
    dump(out / "declaration.json", decl)
    print(S.sha_file(out / "declaration.json"))
    return 0


# --------------------------------------------------------------------------------------------- build ---
def vals_of(st, k):
    return S.frame_vals(st, st["frames"].index(k))


def all_vals(st):
    return {k: S.frame_vals(st, i) for i, k in enumerate(st["frames"])}


class XBuild:
    def __init__(self, decl):
        self.decl = decl
        self.B, self.par = w117_replay()
        B = self.B
        pre = Path(decl["preflight_dir"])
        self.probe = load(pre / "syn_probe.json")
        PE = load(pre / "syn_parent_P__E.json")
        B.sc.parents["P__E"] = PE
        B.syn["P__E"] = PE
        for fr in self.probe["states"]["P__E"]["frames"]:
            for bid, rec in B.syn_b.items():
                B.cap[(PE["state_id"], fr["frame_index"], bid)] = fr["world"][rec["bone"]][rec["point"]]
        self.x = {}

    def side(self, R):
        return (self.B.real, REAL_FRAMES, self.B.ctx.rig) if R["side"] == "real" else (self.B.syn, SYN_FRAMES, self.B.sc.rig)

    # ---------------- new requests (U/F/T/TE with frozen W116/W117 code)
    def new_syn(self, rid, tr, edit, holds, role):
        B, rig = self.B, self.B.sc.rig
        P = B.sc.parents[f"P__{tr}"]
        ukey = f"U__{tr}__{edit}"
        if ukey not in B.syn:
            B.syn[ukey] = C.patched_state(P, W7.syn_ctrl(rig, P, edit), {"kind": "EVAL_CONTROL", "trajectory": tr, "edit": edit})
        ctrl = all_vals(B.syn[ukey])
        pins, frames = [], []
        for i, (cap, lo, hi) in enumerate(holds):
            fr = list(range(lo, hi + 1))
            frames += fr
            pins += B.syn_pins(P, [("syn_wrist", "capture", None)], fr, W7.SYN_ARM, cap, f"{rid}/{i}/")
        req = {"request_id": rid, "parent_state_id": P["state_id"], "edit": dict(W7.SYN_EDITS[edit], edit_id=edit, frames=SYN_FRAMES),
               "constraints": pins, "resolver": {}}
        errs = S.validate_request(req, B.syn_vctx(P))
        st, vals, _ = B.framewise(rig, ctrl, B.syn_b, pins, frames, W7.SYN_ARM)
        B.syn[f"{rid}__F"] = C.patched_state(P, W7.full_patch(P, ctrl, {k: vals[k] for k in frames}), {"kind": "F_CANDIDATE", "request": rid})
        cands = {"F": {"state": f"{rid}__F", "statuses": {str(k): st[k] for k in frames}, "info": {}, "pin_frames": frames, "margin_frames": []}}
        if len(holds) == 1:
            for v in ("T", "TE"):
                s2, v2, info, run = B.temporal(rig, ctrl, B.syn_b, pins, frames, W7.SYN_ARM, "upper", "lower", margins=(v == "TE"),
                                               all_frames=SYN_FRAMES)
                B.syn[f"{rid}__{v}"] = C.patched_state(P, W7.full_patch(P, ctrl, {k: v2[k] for k in run}), {"kind": f"{v}_CANDIDATE", "request": rid})
                cands[v] = {"state": f"{rid}__{v}", "statuses": {str(k): s2[k] for k in frames}, "info": info, "pin_frames": frames,
                            "margin_frames": [k for k in run if k not in frames]}
        lo, hi = frames[0], frames[-1]
        B.req[rid] = {"side": "syn", "kind": "hold", "role": "evaluation" if role.startswith("evaluation") else "control", "w118_role": role,
                      "parent": f"P__{tr}", "U": ukey, "edit": edit, "auth": W7.SYN_ARM, "request": req, "validation_errors": errs,
                      "pin_frames": frames, "window": [max(0, lo - TB.MARGIN - 1), min(39, hi + TB.MARGIN + 1)],
                      "boundary": [lo, hi, TB.MARGIN] if len(holds) == 1 else None, "eval_frames": SYN_FRAMES, "candidates": cands,
                      "chain": ["upper", "lower"], "w118_new": True}

    def new_real(self, rid, e, holds, role):
        B = self.B
        ctx, rig, v0 = B.ctx, B.ctx.rig, B.ctx.v0
        fam = W7.HOLD_EDIT_FAMILY[e]
        edit = W7.HOLD_EDITS[e]
        ctrl = B.rctrl[fam]
        pins, frames = [], []
        for i, (cap, lo, hi) in enumerate(holds):
            fr = list(range(lo, hi + 1))
            frames += fr
            pins += C.make_pins(ctx, False, v0, [("left_wrist", "capture", None)], fr, ARM, cap_frame=cap, cid_prefix=f"{rid}/{i}/")
        er = C.edit_record(edit, REAL_FRAMES, B.rparams[fam])
        req = {"request_id": rid, "parent_state_id": v0["state_id"], "edit": er, "constraints": pins, "resolver": {}}
        errs = S.validate_request(req, ctx.vctx(v0, False))
        st, vals, _ = B.framewise(rig, ctrl, ctx.real_b, pins, frames, ARM)
        B.real[f"{rid}__F"] = C.patched_state(v0, W7.full_patch(v0, ctrl, {k: vals[k] for k in frames}), {"kind": "F_CANDIDATE", "request": rid})
        cands = {"F": {"state": f"{rid}__F", "statuses": {str(k): st[k] for k in frames}, "info": {}, "pin_frames": frames, "margin_frames": []}}
        B.req[rid] = {"side": "real", "kind": "hold", "role": "evaluation", "w118_role": role, "parent": "v0", "U": f"{fam}__control",
                      "edit": edit, "auth": ARM, "request": req, "validation_errors": errs, "pin_frames": frames,
                      "window": [frames[0] - TB.MARGIN - 1, frames[-1] + TB.MARGIN + 1], "boundary": None, "eval_frames": REAL_FRAMES,
                      "candidates": cands, "chain": [UPPER, LOWER], "w118_new": True}

    # ---------------- X candidates
    def make_x(self, rid, m_max=XB.M_MAX):
        B = self.B
        R = B.req[rid]
        store, frames, rig = self.side(R)
        P = store[R["parent"]]
        U = store[R["U"]]
        pin = R["pin_frames"]
        if R["validation_errors"] or "F" not in R["candidates"]:
            store[f"{rid}__X"] = U
            plan, budget = XB.budget_plan(pin, frames, m_max)
            R["candidates"]["X"] = {"state": R["U"], "statuses": {}, "info": {"not_resolved": "CONTRACT_INVALID"}, "pin_frames": pin,
                                    "margin_frames": [], "x_info": {"budget_frames": budget, "planned_frames": [], "plan": plan, "sides": []}}
            return
        Fst = store[R["candidates"]["F"]["state"]]
        ctrl = all_vals(U)
        base = {k: vals_of(Fst, k) for k in pin}
        t0 = time.time()
        xv, info = XB.transition_edit(rig, ctrl, base, pin, frames, R["auth"], R["chain"][1], m_max=m_max)
        info["wall_s_total"] = time.time() - t0
        store[f"{rid}__X"] = C.patched_state(P, W7.full_patch(P, ctrl, xv), {"kind": "X_CANDIDATE", "request": rid})
        R["candidates"]["X"] = {"state": f"{rid}__X", "statuses": dict(R["candidates"]["F"]["statuses"]), "info": {}, "pin_frames": pin,
                                "margin_frames": [k for k in info["planned_frames"] if k not in set(pin)], "x_info": info}
        self.x[rid] = xv

    def tamper_x(self, rid, src, fn, statuses=None, planned_override=None):
        B = self.B
        R0 = B.req[src]
        store, frames, rig = self.side(R0)
        P, U = store[R0["parent"]], store[R0["U"]]
        ctrl = all_vals(U)
        vv = {k: {c: np.array(x, float) for c, x in v.items()} for k, v in self.x[src].items()}
        fn(vv, ctrl)
        store[f"{rid}__X"] = C.patched_state(P, W7.full_patch(P, ctrl, vv), {"kind": "TAMPERED_X", "request": rid})
        R = copy.deepcopy({k: v for k, v in R0.items() if k != "candidates"})
        R["role"] = "negative"
        xi = copy.deepcopy(R0["candidates"]["X"]["x_info"])
        if planned_override is not None:
            xi["planned_frames"] = planned_override(xi)
        R["candidates"] = {"X": {"state": f"{rid}__X", "statuses": statuses or dict(R0["candidates"]["X"]["statuses"]),
                                 "info": {"tamper": NEG_DESC[rid]}, "pin_frames": R0["pin_frames"],
                                 "margin_frames": [k for k in xi["planned_frames"] if k not in set(R0["pin_frames"])], "x_info": xi}}
        B.req[rid] = R


def rot_local(q, axis, deg):
    return S.q_mul(np.asarray(q, float), S.q_axis_angle(axis, math.radians(deg)))


def cmd_build(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    if S.sha_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash")
    g = guard()
    t0 = time.time()
    X = XBuild(decl)
    B = X.B
    for rid, (tr, edit, holds, role) in NEW_SYN.items():
        X.new_syn(rid, tr, edit, holds, role)
    for rid, (e, holds, role) in NEW_REAL.items():
        X.new_real(rid, e, holds, role)
    w117_x = [r for r, R in B.req.items() if not R.get("w118_new") and R["role"] in ("evaluation", "development", "noop")]
    for rid in sorted(w117_x) + X_W117_NEG + [r for r in NEW_SYN if r != "Z_A"] + list(NEW_REAL):
        X.make_x(rid)
    X.make_x("Z_A", m_max=0)
    # tamper controls on H_A X / H_RH1_none X
    xa = B.req["H_A"]["candidates"]["X"]["x_info"]
    tf = [k for k in xa["planned_frames"] if k not in set(B.req["H_A"]["pin_frames"])]
    t_frame = tf[0] if tf else 13
    stA = B.req["H_A"]["candidates"]["X"]["statuses"]
    X.tamper_x("NX1", "H_A", lambda vv, c: vv.__setitem__(2, dict(c[2], upper=rot_local(c[2]["upper"], [0, 0, 1], 1.0))))
    X.tamper_x("NX2", "H_A", lambda vv, c: None, planned_override=lambda xi: [k for k in xi["planned_frames"] if k != min(xi["planned_frames"])])
    X.tamper_x("NX3", "H_A", lambda vv, c: vv[t_frame].__setitem__("base", rot_local(vv[t_frame]["base"], [1, 0, 0], 1.0)))
    X.tamper_x("NX4", "H_A", lambda vv, c: vv[t_frame].__setitem__("__object__", rot_local(vv[t_frame]["__object__"], [0, 0, 1], 1.0)))

    def false_solved(vv, c, pins):
        for k in pins:
            vv[k] = dict(c[k])
    X.tamper_x("NX5", "H_A", lambda vv, c: false_solved(vv, c, B.req["H_A"]["pin_frames"]), statuses={k: "SOLVED" for k in stA})
    X.tamper_x("NX6", "H_A", lambda vv, c: vv[20].__setitem__("lower", rot_local(vv[20]["lower"], [0, 0, 1], 2.0)))
    stR = B.req["H_RH1_none"]["candidates"]["X"]["statuses"]
    X.tamper_x("RX1", "H_RH1_none", lambda vv, c: vv.__setitem__(100, dict(c[100], upperarm_l=rot_local(c[100]["upperarm_l"], [0, 0, 1], 1.0))))
    X.tamper_x("RX5", "H_RH1_none", lambda vv, c: false_solved(vv, c, B.req["H_RH1_none"]["pin_frames"]), statuses={k: "SOLVED" for k in stR})
    X.tamper_x("RX6", "H_RH1_none", lambda vv, c: vv[113].__setitem__("lowerarm_l", rot_local(vv[113]["lowerarm_l"], [0, 0, 1], 2.0)))
    # known-footprint instrument control (no solver)
    Ust = B.syn[KNOWN_FP["base_U"]]
    kv = all_vals(Ust)
    for k, chs in KNOWN_FP["changes"].items():
        for ch, deg in chs.items():
            kv[int(k)][ch] = rot_local(kv[int(k)][ch], [0, 0, 1], deg)
    B.syn["KF__X"] = C.patched_state(B.sc.parents["P__A"], kv, {"kind": "KNOWN_FOOTPRINT_CONTROL"})
    if B.ctx.v0["frames"] != REAL_FRAMES:
        raise SystemExit("STOP: real parent frames are not 0..198")
    C.states_file(out / "real_states.json", B.real, "w105")
    C.states_file(out / "syn_states.json", B.syn, "synthetic")
    rep = {"schema": "w118_build_v1", "tag": a.tag, "declaration_sha256": a.declaration_sha, "wall_s": time.time() - t0,
           "w117_replay_parity": {k: (v if not isinstance(v, dict) else all(v.values())) for k, v in X.par.items()},
           "real_states": {n: s["state_id"] for n, s in B.real.items()}, "syn_states": {n: s["state_id"] for n, s in B.syn.items()},
           "state_meta": {side: {n: {"parent_id": s["parent_id"], "op": s["op"]} for n, s in st.items()}
                          for side, st in (("real", B.real), ("syn", B.syn))},
           "requests": B.req, "x_requests": sorted(r for r, R in B.req.items() if "X" in R["candidates"]),
           "source_guard": g.check()}
    dump(out / "build_report.json", rep)
    print(json.dumps({"tag": a.tag, "wall_s": rep["wall_s"], "n_real": len(B.real), "n_syn": len(B.syn), "n_x": len(rep["x_requests"]),
                      "replay": rep["w117_replay_parity"]["ok"], "source": rep["source_guard"]["unchanged"]}))
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
def x_assess(rid, R, ev, rig, bindings, rest, scale, frames_all, store_obj, parents):
    """Footprint, display, decision, quality and commit preview/result for every variant of one request."""
    U = C.ev_frames(ev, R["U"])
    P = C.ev_frames(ev, R["parent"])
    pin = R["pin_frames"]
    auth = R["auth"]
    errs = [tuple(e) for e in R["validation_errors"]]
    edit_ch = list(R["request"]["edit"]["channels"]) if R["request"].get("edit") else []
    dom = S.domain_points(rig, sorted(set(auth) | set(edit_ch))) if (auth or edit_ch) else set()
    arm_dom = S.domain_points(rig, auth) if auth else set()
    pf = W7.pins_fn(bindings, R["request"])
    wb = (R["chain"][1], "tail")
    eb = (R["chain"][0], "tail")
    out = {}
    for v, cd in R["candidates"].items():
        Xf = C.ev_frames(ev, cd["state"])
        fp = XB.derive_footprint(Xf, U, frames_all, pin, arm_dom, wrist=wb, elbow=eb)
        rec = {"state": cd["state"], "footprint": {k: x for k, x in fp.items() if k != "per_frame"}}
        xi = cd.get("x_info")
        planned = xi["planned_frames"] if xi else sorted(set(pin) | set(cd.get("margin_frames", [])))
        budget = xi["budget_frames"] if xi else planned
        spec = XB.display_spec(fp, frames_all, pin, budget, planned, store_ids(ev, cd["state"]), ev["states"][R["parent"]]["state_id"],
                               "PROPOSED", rid, v, edit_ch)
        rec["display_check_proposed"] = XB.check_display(spec, Xf, U, frames_all, pin, ev["states"][cd["state"]]["state_id"])
        rec["display_spec_proposed"] = spec
        if v == "X":
            evals, per = [], {}
            if not errs:
                for k in sorted(set(planned) | set(pin)):
                    e = S.evaluate_candidate(Xf[k], U[k], P[k], pf(k) if k in set(pin) else [],
                                             {"vs_parent": sorted(set(auth) | set(edit_ch)), "vs_control": sorted(auth)}, dom, rest, scale,
                                             chain=tuple(R["chain"]))
                    per[k] = e
                    evals.append(e)
            statuses = [s for s in cd["statuses"].values() if s != "NOT_APPLICABLE"]
            if errs or statuses:
                dec = XB.decide_x(errs, statuses, evals, fp, budget, planned, auth, display_check=rec["display_check_proposed"])
            else:
                dec = {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": ["NOT_RESOLVED"]}
            res = [p["candidate_residual_m"] for k in pin if k in per for p in per[k]["pins"]]
            rec.update(decision=dec, planned_frames=planned, budget_frames=budget, x_info=xi,
                       frame_pass={str(k): bool(e["pins_satisfied"] and e["scope_ok"] and e["locality_ok"] and e["lengths_ok"]
                                                and e["root_unchanged"] and e["joint_limit_ok"]) for k, e in per.items()},
                       residual_m={"max": max(res) if res else None, "n_over_tol": sum(1 for r in res if r > S.TOL_M), "n": len(res),
                                   "per_frame": {str(k): per[k]["pins"][0]["candidate_residual_m"] for k in pin if k in per and per[k]["pins"]}},
                       stored_changed_vs_control=sorted({c for e in evals for c in e["stored_changed_vs_control"]}),
                       user_edit_and_object_equal_U=all(Xf[k]["stored_wxyz"].get(c) == U[k]["stored_wxyz"].get(c)
                                                        for k in frames_all for c in list(edit_ch) + [S.OBJECT]))
        if R["role"] in ("evaluation", "development", "noop", "control") and not errs:
            rec["quality"] = XB.quality(Xf, U, frames_all, pin, R["chain"][0], R["chain"][1], auth)
            rec["collateral"] = W7.collateral(Xf, U, P, list(range(frames_all[0], frames_all[-1] + 1)), R["side"])
        out[v] = rec
    # commit X: preview -> commit -> result
    if "X" in out and store_obj is not None:
        side_states, parent = parents
        cand = side_states[R["candidates"]["X"]["state"]]
        ref = f"x/{rid}"
        store_obj.put(parent)
        store_obj.init_ref(ref, parent["state_id"])
        x = out["X"]
        if cand["state_id"] == parent["state_id"] or cand.get("parent_id") != parent["state_id"]:
            crec = {"decision": "REJECT", "reasons": ["CANDIDATE_IS_NOT_A_CHILD_OF_PARENT"], "head_before": parent["state_id"],
                    "head_after": store_obj.refs[ref], "candidate": cand["state_id"]}
            if x["decision"]["decision"] == "ACCEPT":
                x["decision"] = {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": ["CANDIDATE_IS_NOT_A_CHILD_OF_PARENT"]}
        else:
            crec = store_obj.commit(ref, parent["state_id"], cand, x["decision"])
        label = "ACCEPTED" if crec["decision"] == "ACCEPT" else "REJECTED"
        res_spec = dict(x["display_spec_proposed"], status=label, head_after=crec["head_after"])
        x["commit"] = {"preview_state_id": x["display_spec_proposed"]["state_id"], "candidate_state_id": cand["state_id"],
                       "parent_state_id": parent["state_id"], **{k: crec.get(k) for k in ("decision", "reasons", "head_before", "head_after")},
                       "preview_id_equals_candidate": x["display_spec_proposed"]["state_id"] == cand["state_id"],
                       "head_consistent": (crec["head_after"] == cand["state_id"]) if crec["decision"] == "ACCEPT"
                       else (crec["head_after"] == parent["state_id"])}
        x["display_spec_result"] = res_spec
        x["display_check_result"] = XB.check_display(res_spec, C.ev_frames(ev, R["candidates"]["X"]["state"]), U, frames_all, pin,
                                                     cand["state_id"], decision=crec["decision"])
    return out


def store_ids(ev, name):
    return ev["states"][name]["state_id"]


def corrupt_display(kind, spec, fp_rec=None):
    s = copy.deepcopy(spec)
    if kind in ("DC1", "DC5"):
        k = next((f for f, c in sorted(s["cells"].items(), key=lambda t: int(t[0])) if c == "AUTO_CHANGED"), None)
        if k is None:
            return None
        s["cells"][k] = "UNCHANGED"
        s["channel_rows"] = {c: [f for f in v if f != int(k)] for c, v in s["channel_rows"].items()}
    elif kind == "DC2":
        if len(s["changed_runs"]) < 2:
            return None
        lo, hi = s["envelope"]
        s["changed_runs"] = [[lo, hi]]
        for f in range(lo, hi + 1):
            if s["cells"][str(f)] == "UNCHANGED":
                s["cells"][str(f)] = "AUTO_CHANGED"
    elif kind == "DC3":
        if "lower" not in s["channel_rows"]:
            return None
        s["channel_rows"].pop("lower")
    elif kind == "DC4":
        planned = set(s["planned_frames"])
        pin = set(s["pin_frames"])
        s["cells"] = {f: (("PIN_CHANGED" if int(f) in pin else "AUTO_CHANGED") if int(f) in planned else "UNCHANGED") for f in s["cells"]}
        s["channel_rows"] = {c: sorted(planned) for c in ("upper", "lower")}
        ch = sorted(planned)
        runs = XB.runs_of(ch)
        s["changed_runs"], s["envelope"] = runs, ([ch[0], ch[-1]] if ch else None)
    elif kind == "DC6":
        s["status"] = "ACCEPTED"
    return s


def cmd_assess(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    if S.sha_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash")
    g = guard()
    Bd = Path(a.build_dir)
    Bb = load(Bd / "build_report.json")
    rev, sev = load(Path(a.eval_dir) / "real_eval.json"), load(Path(a.eval_dir) / "syn_eval.json")
    ctx = C.Ctx(load(W116 / "results/declare/declaration.json"))
    pre = Path(decl["preflight_dir"])
    probe = load(pre / "syn_probe.json")
    srig = W7.syn_rig(probe)
    w117cal = load(W117 / "results/calibrate/calibration.json")
    syn_b = w117cal["syn_bindings"]["bound"]
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
    rep = {"schema": "w118_assess_v1", "declaration_sha256": a.declaration_sha, "requests": {}, "w117_assess": {}}
    # ---- frozen W117 assess of the F/T/TE variants (parity for W117 requests; frozen classification for new ones)
    w117A = load(W117 / "results/assess/assess_report.json")["requests"]
    for rid, R in Bb["requests"].items():
        side = R["side"]
        RR = dict(R, candidates={v: c for v, c in R["candidates"].items() if v != "X"})
        if not RR["candidates"] or rid == "KF":
            continue
        if side == "real":
            A7 = W7.assess_request(rid, RR, rev, ctx.rig, ctx.real_b, rev["rest"], rev["rig"]["scale"][0])
        else:
            A7 = W7.assess_request(rid, RR, sev, srig, dict(syn_b) | w116_syn, sev["rest"], 1.0)
        rep["w117_assess"][rid] = A7
    par7 = {}
    for rid, r in w117A.items():
        mine = rep["w117_assess"].get(rid)
        par7[rid] = mine is not None and all(mine["candidates"][v]["decision"] == c["decision"] for v, c in r["candidates"].items()) \
            and mine["oracle"] == r["oracle"]
    rep["w117_assess_parity"] = {"all": all(par7.values()), "mismatch": [r for r, ok in par7.items() if not ok]}
    # ---- X (and footprint/display of every variant)
    for rid in Bb["x_requests"]:
        R = Bb["requests"][rid]
        side = R["side"]
        if side == "real":
            ev, rig, bnd, rest, scale, fr = rev, ctx.rig, ctx.real_b, rev["rest"], rev["rig"]["scale"][0], REAL_FRAMES
        else:
            ev, rig, bnd, rest, scale, fr = sev, srig, dict(syn_b) | w116_syn, sev["rest"], 1.0, SYN_FRAMES
        par = full_state(side, R["parent"])
        cname = R["candidates"]["X"]["state"]
        sstates = {cname: rebuild_state(Bb, side, cname, states[side][cname], par)}
        rep["requests"][rid] = {"role": R["role"], "w118_role": R.get("w118_role"), "side": side, "pin_frames": R["pin_frames"],
                                "pin_runs": XB.pin_runs(R["pin_frames"]), "validation_errors": R["validation_errors"],
                                "variants": x_assess(rid, R, ev, rig, bnd, rest, scale, fr, store, (sstates, par))}
    # ---- known-footprint control + U-vs-U
    U_A = C.ev_frames(sev, KNOWN_FP["base_U"])
    kf = XB.derive_footprint(C.ev_frames(sev, "KF__X"), U_A, SYN_FRAMES, [], S.domain_points(srig, ["upper", "lower"]),
                             wrist=("lower", "tail"), elbow=("upper", "tail"))
    want = {c: sorted(int(k) for k, chs in KNOWN_FP["changes"].items() if c in chs) for c in ("upper", "lower")}
    uu = XB.derive_footprint(U_A, U_A, SYN_FRAMES, [], set(), wrist=("lower", "tail"), elbow=("upper", "tail"))
    rep["known_footprint"] = {"frames_changed": kf["frames_changed"], "channel_frames": kf["channel_frames"], "runs": kf["runs"],
                              "ok": kf["frames_changed"] == [3, 17] and kf["channel_frames"] == want and kf["runs"] == [[3, 3], [17, 17]],
                              "U_vs_U_empty": uu["frames_changed"] == []}
    # ---- display corruption controls
    dcs = {}
    srcmap = {"DC1": "H_A", "DC2": "DJ_B", "DC3": "H_A", "DC4": "NX1", "DC5": "H_RH1_none", "DC6": "H_Cx"}
    for dc, src in srcmap.items():
        R = Bb["requests"][src]
        ev, fr = (rev, REAL_FRAMES) if R["side"] == "real" else (sev, SYN_FRAMES)
        x = rep["requests"][src]["variants"]["X"]
        base_spec = x["display_spec_result"] if dc == "DC6" else x["display_spec_proposed"]
        s = corrupt_display(dc, base_spec)
        if s is None:
            dcs[dc] = {"executed": False, "reason": "corruption not applicable to this output"}
            continue
        Xf, Uf = C.ev_frames(ev, R["candidates"]["X"]["state"]), C.ev_frames(ev, R["U"])
        chk = XB.check_display(s, Xf, Uf, fr, R["pin_frames"], ev["states"][R["candidates"]["X"]["state"]]["state_id"],
                               decision=x["commit"]["decision"] if dc == "DC6" else None)
        rg = ctx.rig if R["side"] == "real" else srig
        fp = XB.derive_footprint(Xf, Uf, fr, R["pin_frames"], S.domain_points(rg, R["auth"]), wrist=(R["chain"][1], "tail"),
                                 elbow=(R["chain"][0], "tail"))
        gated = XB.decide_x([], [s_ for s_ in R["candidates"]["X"]["statuses"].values()], [], fp, x["budget_frames"], x["planned_frames"],
                            R["auth"], display_check=chk) if dc != "DC6" else None
        dcs[dc] = {"executed": True, "source": src, "desc": DISPLAY_CONTROLS[dc], "check": chk,
                   "gated_decision_with_this_display": gated}
    rep["display_controls"] = dcs
    # ---- lineage programmes
    rep["lineage"] = lineage(out, decl, Bb, rep, full_state, states, rev, sev, pre)
    # ---- W117 Blender parity of replayed states
    w7r, w7s = load(W117 / "results/eval_e1/real_eval.json"), load(W117 / "results/eval_e1/syn_eval.json")
    bp = {"real": [n for n in w7r["states"] if not (rev["states"].get(n, {}).get("value_digest") == w7r["states"][n]["value_digest"]
                                                   and rev["states"][n]["frames"] == w7r["states"][n]["frames"])],
          "syn": [n for n in w7s["states"] if not (sev["states"].get(n, {}).get("value_digest") == w7s["states"][n]["value_digest"]
                                                  and sev["states"][n]["frames"] == w7s["states"][n]["frames"])]}
    rep["w117_blender_parity"] = {"mismatch": bp, "n_real": len(w7r["states"]), "n_syn": len(w7s["states"]),
                                  "ok": not bp["real"] and not bp["syn"]}
    del w7r, w7s
    # ---- oracle parity (recomputed from this Blender U vs W117 build oracle)
    w7b = load(W117 / "results/build_r1/build_report.json")
    orc = {}
    for fam in W7.REAL_FAMS:
        Uf = C.ev_frames(rev, f"{fam}__control")
        pins_by = {k: [c for c in Bb["requests"][f"W_{fam}"]["request"]["constraints"] if k in c["frames"]] for k in REAL_FRAMES}
        o = W7.oracle_from_eval(Uf, pins_by, ctx.real_b, rev["rest"], rev["rig"]["scale"][0])
        orc[fam] = {str(k): v for k, v in o.items()} == w7b["oracle_real"][fam]
    rep["oracle_parity_w117"] = orc
    rep["source_guard"] = g.check()
    rep["w116_manifest_after"] = W7.w116_manifest_check()
    rep["w117_manifest_after"] = w117_manifest_check()
    rep["parent_files_verify"] = {"real": S.verify_state(load(pre / "real_parent_state.json")),
                                  **{n: S.verify_state(load(pre / f"syn_parent_{n}.json")) for n in ("P__A", "P__B", "P__C", "P__D", "P__E")}}
    rep["classification"] = classify(rep, Bb)
    dump(out / "assess_report.json", rep)
    summ = {rid: [r["variants"]["X"]["decision"]["decision"], r["variants"]["X"]["decision"]["failure_class"],
                  r["variants"]["X"]["decision"]["reasons"][:3]] for rid, r in rep["requests"].items()}
    print(json.dumps({"x": summ, "w117_assess_parity": rep["w117_assess_parity"]["all"], "w117_blender": rep["w117_blender_parity"]["ok"],
                      "kf": rep["known_footprint"]["ok"], "dc": {k: v.get("check", {}).get("ok") for k, v in dcs.items()},
                      "source": rep["source_guard"]["unchanged"]}, indent=0, default=str))
    return 0


def rebuild_state(Bb, side, name, cs, par):
    """Rebuild the full content-addressed state (parent link + recorded op) from the states file; verified by id."""
    if cs["state_id"] == par["state_id"]:
        return par
    meta = Bb["state_meta"][side][name]
    if meta["parent_id"] != par["state_id"]:
        raise SystemExit(f"STOP: state {name} is not a child of {par['state_id']}")
    st = S.make_state(par["state_id"], par["rig_id"], par["frames"], par["channels"], cs["values"], meta["op"], par.get("source"))
    if st["state_id"] != cs["state_id"]:
        raise SystemExit(f"STOP: state {name} does not rebuild to its id")
    return st


def classify(rep, Bb):
    out = {}
    for rid, r in rep["requests"].items():
        x = r["variants"]["X"]
        d = x["decision"]
        q = x.get("quality")
        if d["decision"] == "REJECT":
            rs = set(d["reasons"])
            if rs & {"CHANGE_OUTSIDE_AUTOMATIC_BUDGET", "FOOTPRINT_EXCEEDS_DECLARED_DOMAIN", "UNAUTHORIZED_CHANNEL_CHANGE",
                     "DISPLAY_FOOTPRINT_MISMATCH", "POINT_OUTSIDE_ARM_DOMAIN_CHANGED_VS_CONTROL", "ROOT_MOVED",
                     "OUTPUT_OUTSIDE_DECLARED_DOMAIN"}:
                c = "UNAUTHORIZED_OUTPUT_OR_FOOTPRINT_MISMATCH"
            elif d["failure_class"] in ("INFEASIBLE_REQUEST", "UNSUPPORTED_BINDING") or r["validation_errors"]:
                c = "UNREACHABLE_OR_CONFLICTING_OR_INVALID_REQUEST"
            elif "EVALUATED_PIN_UNSATISFIED" in rs or "JOINT_LIMIT" in rs:
                c = "HARD_PIN_FAILURE_AFTER_BLENDER"
            elif "NON_CONVERGED" in rs:
                c = "FEASIBLE_PIN_SOLVER_FAILED"
            else:
                c = "REJECTED_OTHER"
        else:
            F = r["variants"].get("F", {})
            if not q or q["B1_boundary_wrist_comp_step_peak_m"] is None or not F.get("quality"):
                c = "ACCEPTED_NO_TRANSITION_AVAILABLE"
            elif x["footprint"]["n_out_of_pin_changed"] == 0:
                c = "ACCEPTED_EQUALS_F"
            else:
                cmpx = XB.compare_x(F["quality"], q)
                c = "VALID_ACCEPTED_IMPROVEMENT" if cmpx["meaningful_improvement"] else "ACCEPTABLE_WITH_QUALITY_TRADE_OFF"
        out[rid] = c
    return out


def lineage(out, decl, Bb, rep, full_state, states, rev, sev, pre):
    store = S.Store(out / "lineage_store")
    res = {"synthetic": [], "real": []}
    R_ = rep["requests"]
    dec = lambda rid: R_[rid]["variants"]["X"]["decision"]  # noqa: E731

    def cand(side, rid):
        R = Bb["requests"][rid]
        par = full_state(side, R["parent"])
        return rebuild_state(Bb, side, R["candidates"]["X"]["state"], states[side][R["candidates"]["X"]["state"]], par), par

    rev_eval = {}
    order = ["H_A", "H_B", "H_C", "H_E"]
    pick = next((r for r in order if dec(r)["decision"] == "ACCEPT"), None)
    if pick:
        c1, P = cand("syn", pick)
        store.put(P)
        store.init_ref("syn/main", P["state_id"])
        res["synthetic"].append(dict(step="S1", request=pick, **store.commit("syn/main", P["state_id"], c1, dec(pick))))
        c2, PA = cand("syn", "NX6")
        store.put(PA)
        store.init_ref("syn/atomic", PA["state_id"])
        h = store.refs["syn/atomic"]
        r2 = store.commit("syn/atomic", h, c2, dec("NX6"))
        res["synthetic"].append(dict(step="S2", head_unchanged=r2["head_after"] == h, **r2))
        other = next(r for r in order if r != pick)
        c3, _ = cand("syn", other)
        res["synthetic"].append(dict(step="S3", request=other, **store.commit("syn/main", P["state_id"], c3, dec(other))))
        srev, srec = store.revert("syn/main")
        res["synthetic"].append(dict(step="S4", **srec))
        C.states_file(out / "syn_revert_states.json", {"syn_revert": srev}, "synthetic")
        blender("scripts/w116_blender_stages.py", "--", "evaluate", "--rig", f"synthetic:{pre / 'syn_spec.json'}", "--states",
                out / "syn_revert_states.json", "--out", out / "syn_revert_eval.json", log=out / "syn_revert_eval.log")
        e = load(out / "syn_revert_eval.json")["states"]["syn_revert"]
        pn = Bb["requests"][pick]["parent"]
        rev_eval["synthetic"] = {"restores": pn, "value_digest_equal_parent": e["value_digest"] == sev["states"][pn]["value_digest"],
                                 "frames_bit_equal_parent": e["frames"] == sev["states"][pn]["frames"]}
    v0 = full_state("real", "v0")
    store.put(v0)
    store.init_ref("real/main", v0["state_id"])
    store.init_ref("real/atomic", v0["state_id"])
    order_r = Q_REAL_HOLDS
    pick = next((r for r in order_r if dec(r)["decision"] == "ACCEPT"), None)
    if pick:
        c1, _ = cand("real", pick)
        res["real"].append(dict(step="R1", request=pick, **store.commit("real/main", v0["state_id"], c1, dec(pick))))
    h = store.refs["real/atomic"]
    r2 = store.commit("real/atomic", h, cand("real", "RX6")[0], dec("RX6"))
    res["real"].append(dict(step="R2", head_unchanged=r2["head_after"] == h, **r2))
    other = next(r for r in order_r if r != pick)
    res["real"].append(dict(step="R3", request=other, **store.commit("real/main", v0["state_id"], cand("real", other)[0], dec(other))))
    h = store.refs["real/atomic"]
    r4 = store.commit("real/atomic", h, cand("real", "H_RH2_none")[0], dec("H_RH2_none"))
    res["real"].append(dict(step="R4", head_unchanged=(r4["head_after"] == h) if r4["decision"] == "REJECT" else None, **r4))
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
COL = {"PIN_CHANGED": (42, 120, 214), "PIN_UNCHANGED": (160, 190, 230), "AUTO_CHANGED": (235, 104, 52), "UNCHANGED": (226, 226, 222),
       "REJECTED": (120, 120, 120)}
SERIES = {"F": "#2a78d6", "TE": "#eb6834", "X": "#1baf7a", "T": "#eda100"}
VLABEL = {"U": "U  unpinned user edit", "F": "F  W116 framewise", "T": "T  W117 temporal", "TE": "TE W117 easing (4)", "X": "X  W118 auto transition"}


def draw_timeline(path, rid, A, cell_w=None):
    from PIL import Image, ImageDraw
    r = A["requests"][rid]
    vs = [v for v in ("F", "T", "TE", "X") if v in r["variants"]]
    spec0 = r["variants"]["X"]["display_spec_proposed"]
    frames = spec0["frames"]
    n = len(frames)
    cw = cell_w or max(4, min(22, 1100 // n))
    L, top = 230, 92
    rowh = 22
    xr = r["variants"]["X"]
    xs = xr.get("display_spec_result", spec0)
    chans = sorted(set(xr["footprint"]["channel_frames"]) | set(r["variants"].get("F", {}).get("footprint", {}).get("channel_frames", {})))
    nrows = len(vs) + 2 + len(chans) + 4
    H = top + nrows * (rowh + 4) + 120
    W = L + cw * n + 30
    im = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(im)
    ink, mute = (11, 11, 11), (82, 81, 78)
    d.text((10, 8), f"{rid}   [{r['side']}]   Edit-footprint timeline (actual Blender output vs U)", fill=ink)
    d.text((10, 24), f"hard-pin runs: {r['pin_runs']}   automatic budget: {len(xr['budget_frames'])} frames   X planned: "
                     f"{len(xr['planned_frames'])}   X actual changed runs: {xr['footprint']['runs']}", fill=mute)
    cm = xr["commit"]
    d.text((10, 40), f"X state {cm['candidate_state_id'][:12]}  parent {cm['parent_state_id'][:12]}  ->  {cm['decision']}ED  "
                     f"head after commit {cm['head_after'][:12]}" + ("" if cm["decision"] == "ACCEPT" else "   (candidate NOT applied)"),
           fill=(180, 30, 30) if cm["decision"] != "ACCEPT" else ink)
    d.text((10, 56), f"decision reasons: {xr['decision']['failure_class']} {xr['decision']['reasons'][:4]}", fill=mute)

    def x_of(i):
        return L + i * cw

    # frame axis
    for i, f in enumerate(frames):
        if f % (10 if n > 60 else 5) == 0:
            d.line([(x_of(i), top - 6), (x_of(i), top - 1)], fill=mute)
            d.text((x_of(i) - 4, top - 20), str(f), fill=mute)
    d.text((10, top - 20), "Action frame index", fill=mute)
    y = top

    def row(label, cells, hatch=False):
        nonlocal y
        d.text((10, y + 5), label, fill=ink)
        for i, f in enumerate(frames):
            c = cells.get(str(f), "UNCHANGED")
            d.rectangle([x_of(i), y, x_of(i) + cw - 2, y + rowh], fill=COL[c])
            if hatch and c != "UNCHANGED":
                for t in range(0, rowh, 5):
                    d.line([(x_of(i), y + t), (x_of(i) + cw - 2, y + min(rowh, t + cw))], fill=(90, 90, 90))
        y += rowh + 4

    pinset = set(r["pin_frames"])
    row("requested hard pin", {str(f): "PIN_CHANGED" if f in pinset else "UNCHANGED" for f in frames})
    row("automatic search budget", {str(f): ("PIN_UNCHANGED" if f in pinset else "AUTO_CHANGED") if f in set(xr["budget_frames"]) else "UNCHANGED"
                                    for f in frames}, hatch=True)
    for v in vs:
        sp = r["variants"][v]["display_spec_proposed"]
        rej = v == "X" and cm["decision"] != "ACCEPT"
        row(("ACTUAL " if True else "") + VLABEL[v] + (" [REJECTED]" if rej else ""), sp["cells"], hatch=rej)
    for ch in chans:
        fs = set(xr["footprint"]["channel_frames"].get(ch, []))
        row(f"   X channel {ch}", {str(f): ("PIN_CHANGED" if f in pinset else "AUTO_CHANGED") if f in fs else "UNCHANGED" for f in frames})
    # magnitude bars: wrist displacement vs U (mm), X and TE/F
    d.text((10, y + 4), "wrist displacement vs U (mm)", fill=ink)
    y += 20
    bar_h = 60
    for v in [v for v in ("F", "TE", "X") if v in r["variants"]]:
        mg = r["variants"][v]["display_spec_proposed"]["magnitude"]
        mx = max([m["wrist_mm"] for m in mg.values()] + [1.0])
        d.text((10, y + bar_h // 2 - 6), f"{v}  (max {mx:.0f} mm)", fill=mute)
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
    # legend
    lx = 10
    for name, c in (("hard-pin frame (changed)", "PIN_CHANGED"), ("pin frame unchanged (already satisfied)", "PIN_UNCHANGED"),
                    ("automatic out-of-pin edit", "AUTO_CHANGED"), ("unchanged vs U", "UNCHANGED"), ("hatched = budget / REJECTED, not applied", "REJECTED")):
        d.rectangle([lx, y + 6, lx + 14, y + 18], fill=COL[c])
        d.text((lx + 18, y + 6), name, fill=ink)
        lx += 18 + 7 * len(name) + 16
    d.text((10, y + 26), "Grey bar = change below display resolution 1 mm (shown, not hidden). Time axis = Action frame index; no seconds "
                         "(source 50 fps / Action 30 fps not reconciled).", fill=mute)
    im = im.crop((0, 0, W, y + 46))
    im.save(path)


def draw_steps(path, rid, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = A["requests"][rid]
    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=110)
    lo_w, hi_w = r["variants"]["X"]["quality"]["window"]
    for v in ("F", "TE", "X"):
        q = r["variants"].get(v, {}).get("quality")
        if not q:
            continue
        pt = q["per_transition"]
        ks = sorted(int(k) for k in pt)
        ax.plot(ks, [pt[str(k)]["wrist_comp_step_m"] * 1e3 for k in ks], color=SERIES[v], lw=2, label=VLABEL[v].strip())
    for a_, b_ in r["pin_runs"]:
        ax.axvspan(a_ - 0.5, b_ + 0.5, color="#2a78d6", alpha=0.08, lw=0)
    ax.set_xlabel("Action frame index (transition k-1 -> k)")
    ax.set_ylabel("wrist compensation step (mm/frame)")
    ax.set_title(f"{rid}: extra wrist motion per frame relative to U (shaded = hard pin)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cmd_render(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    Bb = load(Path(a.build_dir) / "build_report.json")
    A = load(Path(a.assess_dir) / "assess_report.json")
    ev = {"real": load(Path(a.eval_dir) / "real_eval.json"), "syn": load(Path(a.eval_dir) / "syn_eval.json")}
    (out / "timelines").mkdir()
    (out / "steps").mkdir()
    for rid in A["requests"]:
        draw_timeline(out / "timelines" / f"timeline_{rid}.png", rid, A)
        if A["requests"][rid]["variants"]["X"].get("quality"):
            draw_steps(out / "steps" / f"steps_{rid}.png", rid, A)
    # Blender bone-proxy filmstrips of the representative requests
    jobs = {"real": [], "syn": []}
    sel = []
    for side, rids in REPRESENTATIVE.items():
        for rid in rids:
            R = Bb["requests"][rid]
            q = A["requests"][rid]["variants"]["X"]["quality"]
            lo, hi = q["window"]
            frames = list(range(lo, hi + 1))
            sel.append((rid, frames))
            tgt = R["request"]["constraints"][0]["target"]
            variants = [("U", R["U"])] + [(v, R["candidates"][v]["state"]) for v in ("F", "TE", "X") if v in R["candidates"]]
            for v, st in variants:
                E = C.ev_frames(ev[side], st)
                for k in frames:
                    ch = R["chain"]
                    mk = [{"name": "wrist", "position": E[k]["world"][ch[1]]["tail"], "color": (0.1, 0.4, 1.0, 1), "radius": 0.009},
                          {"name": "elbow", "position": E[k]["world"][ch[0]]["tail"], "color": (1.0, 0.55, 0.0, 1), "radius": 0.009}]
                    tk = [c["target"] for c in R["request"]["constraints"] if k in c["frames"]]
                    if tk:
                        mk.append({"name": "target", "position": tk[0], "color": (0.1, 0.9, 0.2, 1), "radius": 0.013})
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
    dump(out / "render_selection.json", {"selection": sel})
    return 0


def compose(out, sel, Bb, A):
    from PIL import Image, ImageDraw
    for rid, frames in sel:
        R = Bb["requests"][rid]
        d = out / R["side"]
        variants = ["U"] + [v for v in ("F", "TE", "X") if v in R["candidates"]]
        ims = {(v, k): Image.open(d / f"{rid}__{v}__f{k:03d}.png").convert("RGB") for v in variants for k in frames}
        w, h = ims[(variants[0], frames[0])].size
        cells = {v: (A["requests"][rid]["variants"][v]["display_spec_proposed"]["cells"] if v != "U" else {}) for v in variants}
        band = 8

        def tile(v, k, scale=1):
            im = ims[(v, k)].copy()
            if scale != 1:
                im = im.resize((w // scale, h // scale))
            c = cells[v].get(str(k), "UNCHANGED") if v != "U" else "UNCHANGED"
            t = Image.new("RGB", (im.size[0], im.size[1] + band), COL[c])
            t.paste(im, (0, band))
            return t
        cols = frames if len(frames) <= 14 else [frames[int(round(i * (len(frames) - 1) / 13))] for i in range(14)]
        sw, sh = w // 2, h // 2 + band
        sheet = Image.new("RGB", (150 + sw * len(cols), 40 + sh * len(variants)), (252, 252, 251))
        dr = ImageDraw.Draw(sheet)
        dr.text((4, 4), f"{rid}: rows {' / '.join(variants)}; band above each tile = that variant's actual footprint class "
                        f"(blue pin, orange automatic edit, grey unchanged). green=target blue=wrist orange=elbow", fill=(11, 11, 11))
        pin = set(R["pin_frames"])
        for j, k in enumerate(cols):
            dr.text((150 + j * sw + 4, 22), f"f{k}{' PIN' if k in pin else ''}", fill=(42, 120, 214) if k in pin else (82, 81, 78))
            for i, v in enumerate(variants):
                sheet.paste(tile(v, k, 2), (150 + j * sw, 40 + i * sh))
        for i, v in enumerate(variants):
            dr.text((4, 40 + i * sh + sh // 2), VLABEL[v], fill=(11, 11, 11))
        sheet.save(out / f"filmstrip_{rid}.png")
        gif = []
        for k in frames:
            fr = Image.new("RGB", (w * len(variants), h + band + 22), (252, 252, 251))
            dd = ImageDraw.Draw(fr)
            for i, v in enumerate(variants):
                fr.paste(tile(v, k), (i * w, 22))
                c = cells[v].get(str(k), "UNCHANGED") if v != "U" else "-"
                dd.text((i * w + 6, 4), f"{v} f{k}{' PIN' if k in pin else ''} {c if v != 'U' else ''}", fill=(11, 11, 11))
            gif.append(fr)
        gif[0].save(out / f"anim_{rid}.gif", save_all=True, append_images=gif[1:], duration=150, loop=0)


def write_index(out, A, sel):
    rows = []
    for rid, r in A["requests"].items():
        x = r["variants"]["X"]
        cm = x["commit"]
        fp = x["footprint"]
        rows.append(f"<tr><td>{rid}</td><td>{r['side']}</td><td>{r['pin_runs']}</td><td>{len(x['budget_frames'])}</td>"
                    f"<td>{fp['out_of_pin_runs']}</td><td>{sorted(fp['channel_frames'])}</td><td><code>{cm['candidate_state_id'][:12]}</code></td>"
                    f"<td><b>{cm['decision']}</b> {x['decision']['failure_class']}</td><td><code>{cm['head_after'][:12]}</code></td>"
                    f"<td>{A['classification'][rid]}</td><td><a href='timelines/timeline_{rid}.png'>timeline</a></td></tr>")
    films = "".join(f"<h3>{rid}</h3><img src='timelines/timeline_{rid}.png'><br><img src='steps/steps_{rid}.png'><br>"
                    f"<img src='filmstrip_{rid}.png'><br><img src='anim_{rid}.gif'>" for rid, _ in sel)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>W118 edit footprint preview (scratch, not product UI)</title>
<style>body{{font-family:sans-serif;background:#fcfcfb;color:#0b0b0b;margin:20px}} td,th{{border-bottom:1px solid #ddd;padding:3px 6px;font-size:12px}}
img{{max-width:100%;margin:6px 0}} .k{{display:inline-block;width:14px;height:12px;margin:0 4px 0 12px}}</style></head><body>
<h1>W118 automatic transition edits: actual edit footprint (scratch diagnostic)</h1>
<p>Every timeline is drawn from the <b>Blender-evaluated output diff vs the unpinned user edit U</b>, not from the solver's intended margin,
and was independently checked against the raw Blender frames. Legend:
<span class='k' style='background:#2a78d6'></span>hard-pin frame
<span class='k' style='background:#a0bee6'></span>pin frame already satisfied (unchanged)
<span class='k' style='background:#eb6834'></span>automatic out-of-pin edit
<span class='k' style='background:#e2e2de'></span>unchanged.
Hatched rows = search budget or a REJECTED candidate (never applied). Time axis = Action frame index.</p>
<p>The user edit itself (spine_01 / base) is part of U and is never attributed to the solver.</p>
<table><tr><th>request</th><th>side</th><th>hard-pin runs</th><th>budget frames</th><th>actual out-of-pin runs</th><th>channels</th>
<th>proposed state</th><th>commit result</th><th>head after</th><th>classification</th><th></th></tr>{''.join(rows)}</table>
<h2>Representative requests (U / F / TE / X)</h2>{films}</body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------------------------- decide ---
def cmd_decide(a):
    out = new_dir(a.out_dir)
    decl = load(a.declaration)
    if S.sha_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash")
    pf = load(Path(decl["preflight_dir"]) / "preflight.json")
    cal = load(Path(decl["calibration_dir"]) / "calibration.json")
    A = load(Path(a.assess_dir) / "assess_report.json")
    b1, b2, e1, e2 = Path(a.build_dir), Path(a.build_dir2), Path(a.eval_dir), Path(a.eval_dir2)
    Bb, B2 = load(b1 / "build_report.json"), load(b2 / "build_report.json")
    VR = load(a.visual_review)
    Rq = A["requests"]
    g, det = {}, {}
    g["V1_authentication"] = not pf["stop"]
    g["V2_parent_reproduction"] = pf["parent_rebuild"]["canonical_equal"] and pf["parent_blender"]["equals_w105_fixed_digest"] and \
        pf["parent_blender"]["frames_bit_equal_w117_preflight"] and all(pf["syn_probe_parity_w117"].values())
    det["V3"] = {"replay_cal": cal["w117_replay_parity"]["ok"], "replay_build": Bb["w117_replay_parity"]["ok"],
                 "blender": A["w117_blender_parity"]["ok"], "w117_assess": A["w117_assess_parity"]}
    g["V3_baseline_replay"] = det["V3"]["replay_cal"] and det["V3"]["replay_build"] and det["V3"]["blender"] and A["w117_assess_parity"]["all"]
    fsub = [rid for rid, r in Rq.items() if "F" in r["variants"]
            and not set(r["variants"]["F"]["footprint"]["frames_changed"]) <= set(r["pin_frames"])]
    det["V4"] = {"known_footprint": A["known_footprint"], "F_footprint_outside_pin": fsub, "syn_fk": cal["syn_fk_max_error_m"]}
    g["V4_independent_footprint_evaluator"] = A["known_footprint"]["ok"] and A["known_footprint"]["U_vs_U_empty"] and not fsub and \
        cal["syn_fk_max_error_m"] <= 1e-5
    # V5: Blender U offsets of H_E
    sev = load(e1 / "syn_eval.json")
    R = Bb["requests"]["H_E"]
    U = C.ev_frames(sev, R["U"])
    T = np.array(R["request"]["constraints"][0]["target"])
    lo, hi = R["pin_frames"][0], R["pin_frames"][-1]
    ent, rel = float(np.linalg.norm(T - np.array(U[lo]["world"]["lower"]["tail"]))), float(np.linalg.norm(T - np.array(U[hi]["world"]["lower"]["tail"])))
    det["V5"] = {"H_E_entry_m": ent, "H_E_release_m": rel, "design_ok": cal["design_ok"]}
    g["V5_meaningful_design"] = cal["design_ok"] and abs(ent - rel) >= 0.005 and min(ent, rel) >= 0.005
    tm = [pf["timing"], pf["syn_timing"]]
    g["V6_timing_semantics"] = all(st["interpolations"] == ["CONSTANT"] and st["blender_frame_equals_index_minus_f0_plus_1"]
                                   and all(s["half_equals_frame"] and s["late_equals_frame"] for s in st["subframe"].values())
                                   for t in tm for st in t["states"].values())
    bad = []
    for rid, r in A["w117_assess"].items():
        for v, c in r["candidates"].items():
            for k, cls in r["oracle"].items():
                if cls == "INFEASIBLE" and c["frame_pass"].get(k):
                    bad.append(f"{rid}/{v}/{k}")
    for rid, r in Rq.items():
        o = A["w117_assess"].get(rid, {}).get("oracle", {})
        fp_ = r["variants"]["X"].get("frame_pass", {})
        for k, cls in o.items():
            if cls == "INFEASIBLE" and fp_.get(k):
                bad.append(f"{rid}/X/{k}")
    det["V7"] = {"infeasible_but_passing": bad, "oracle_parity_w117": A["oracle_parity_w117"]}
    g["V7_oracle_soundness"] = not bad and all(A["oracle_parity_w117"].values())
    same_states = all((b1 / f).read_bytes() == (b2 / f).read_bytes() for f in ("real_states.json", "syn_states.json"))
    same_x = all(Bb["requests"][r]["candidates"]["X"].get("x_info", {}).get("sides") == B2["requests"][r]["candidates"]["X"].get("x_info", {}).get("sides")
                 and Bb["requests"][r]["candidates"]["X"]["statuses"] == B2["requests"][r]["candidates"]["X"]["statuses"] for r in Bb["x_requests"])
    same_ev = all(load(e1 / f)["states"] == load(e2 / f)["states"] for f in ("real_eval.json", "syn_eval.json"))
    det["V8"] = {"states_identical": same_states, "x_plans_identical": same_x, "evaluations_identical": same_ev}
    g["V8_repeatability"] = same_states and same_x and same_ev
    miss = []
    for rid, r in Rq.items():
        for v, c in r["variants"].items():
            if "display_spec_proposed" not in c:
                miss.append(f"{rid}/{v}")
        x = r["variants"]["X"]
        if not r["validation_errors"] and x["residual_m"]["n"] < len(r["pin_frames"]):
            miss.append(f"{rid}/X pins")
    det["V9"] = miss
    g["V9_population_complete"] = not miss
    # ---- C
    c1 = []
    for rid, r in Rq.items():
        x = r["variants"]["X"]
        R = Bb["requests"][rid]
        if x["decision"]["decision"] != "ACCEPT":
            continue
        fp = x["footprint"]
        ok = x["residual_m"]["n_over_tol"] == 0 and set(fp["channel_frames"]) <= set(R["auth"]) and \
            set(fp["frames_changed"]) <= set(x["budget_frames"]) and set(fp["frames_changed"]) <= set(x["planned_frames"]) and \
            not fp["outside_arm_points_changed"] and not fp["matrix_world_changed_frames"] and all(x["frame_pass"].values())
        if not ok:
            c1.append(rid)
    det["C1"] = c1
    g["C1_hard_pin_and_scope"] = not c1
    c2 = []
    for rid, r in Rq.items():
        for v, c in r["variants"].items():
            if not c["display_check_proposed"]["ok"]:
                c2.append(f"{rid}/{v}/proposed")
        x = r["variants"]["X"]
        if not set(x["planned_frames"]) <= set(x["budget_frames"]):
            c2.append(f"{rid}/planned_outside_budget")
        cm = x["commit"]
        if not (x["display_check_result"]["ok"] and cm["preview_id_equals_candidate"] and cm["head_consistent"]):
            c2.append(f"{rid}/result")
    det["C2"] = c2
    g["C2_truthful_footprint"] = not c2
    wrong = []
    for rid, want in NEG_EXPECT_X.items():
        d = Rq[rid]["variants"]["X"]["decision"]
        if [d["decision"], d["failure_class"]] != want:
            wrong.append({"req": rid, "got": [d["decision"], d["failure_class"], d["reasons"]], "expected": want})
    for dc, r in A["display_controls"].items():
        gd = r.get("gated_decision_with_this_display")
        if not r.get("executed") or r["check"]["ok"] or (gd is not None and not (gd["decision"] == "REJECT"
                                                                                  and "DISPLAY_FOOTPRINT_MISMATCH" in gd["reasons"])):
            wrong.append({"req": dc, "got": r})
    det["C3"] = wrong
    g["C3_negative_controls"] = not wrong
    atom = []
    for rid, r in Rq.items():
        x = r["variants"]["X"]
        if r["validation_errors"] or not x["frame_pass"] or r["role"] == "negative":
            continue
        valid = set(Bb["requests"][rid]["candidates"]["X"]["statuses"].values()) <= {"SOLVED", "NO_CHANGE_NEEDED"}
        fp = x["footprint"]
        allpass = all(x["frame_pass"].values()) and valid and set(fp["frames_changed"]) <= set(x["planned_frames"]) and \
            set(fp["channel_frames"]) <= set(Bb["requests"][rid]["auth"])
        if (x["decision"]["decision"] == "ACCEPT") != allpass:
            atom.append(rid)
    for n in ("NX6", "RX6"):
        x = Rq[n]["variants"]["X"]
        if not (x["decision"]["decision"] == "REJECT" and sum(1 for v in x["frame_pass"].values() if not v) == 1):
            atom.append(f"{n}: single-corrupted-frame")
    lin = A["lineage"]
    refs_ok = all(e.get("head_after") == e.get("head_before") for e in lin["store_log"] if e.get("op") == "commit" and e.get("decision") == "REJECT")
    steps = {e["step"]: e for side in lin["programmes"].values() for e in side}
    stale = steps.get("S3", {}).get("reasons") == ["STALE_PARENT"] and steps.get("R3", {}).get("reasons") == ["STALE_PARENT"]
    acc = steps.get("S1", {}).get("decision") == "ACCEPT" and steps.get("R1", {}).get("decision") == "ACCEPT"
    rv = lin["revert_blender"]
    revert_ok = bool(rv.get("real") and rv.get("synthetic")) and all(all(v for k, v in x.items() if k != "restores") for x in rv.values()) \
        and steps.get("S4", {}).get("values_equal_restored") is True and steps.get("R5", {}).get("values_equal_restored") is True
    audit_ok = all(all(e["verifies"] for e in ch) for ch in lin["lineage_audit"].values())
    commit_refs = all(r["variants"]["X"]["commit"]["head_consistent"] for r in Rq.values())
    det["C4"] = {"atomicity": atom, "refs_unchanged_on_reject": refs_ok, "stale": stale, "accept_executed": acc, "revert_ok": revert_ok,
                 "audit_ok": audit_ok, "per_request_heads_consistent": commit_refs,
                 "atomic_heads_unchanged": [steps.get(s, {}).get("head_unchanged") for s in ("S2", "R2")]}
    g["C4_atomicity_lineage_reversal"] = not atom and refs_ok and stale and acc and revert_ok and audit_ok and commit_refs and \
        all(det["C4"]["atomic_heads_unchanged"])
    src_ok = A["source_guard"]["unchanged"] and A["w116_manifest_after"]["ok"] and A["w117_manifest_after"]["ok"] and \
        all(A["parent_files_verify"].values()) and Bb["source_guard"]["unchanged"] and B2["source_guard"]["unchanged"]
    g["C5_source_immutability"] = src_ok
    newfail = []
    for rid, r in Rq.items():
        if r["role"] != "evaluation" or r["validation_errors"]:
            continue
        o = A["w117_assess"].get(rid, {}).get("oracle", {})
        if not o or not all(v == "FEASIBLE" for v in o.values()):
            continue
        if r["variants"]["X"]["decision"]["decision"] != "ACCEPT":
            newfail.append({"req": rid, "got": r["variants"]["X"]["decision"]})
    det["C6"] = newfail
    g["C6_no_new_hard_failure"] = not newfail
    c7 = []
    for rid, r in Rq.items():
        x = r["variants"]["X"]
        if x["decision"]["decision"] != "ACCEPT" or r["role"] == "negative":
            continue
        col = x.get("collateral", {})
        cz = all(v["max_vs_U_m"] == 0.0 for k, v in col.items() if isinstance(v, dict))
        if not (x["user_edit_and_object_equal_U"] and cz and col.get("root_matrix_equal_U", True)):
            c7.append(rid)
    det["C7"] = c7
    g["C7_user_edit_and_collateral"] = not c7
    # ---- Q
    cmpx, worse = {}, {"syn": [], "real": []}
    for rid, r in Rq.items():
        if r["role"] != "evaluation":
            continue
        F, X = r["variants"].get("F"), r["variants"]["X"]
        if not F or F.get("quality") is None or X.get("quality") is None:
            continue
        both = F_dec(A, rid) == "ACCEPT" and X["decision"]["decision"] == "ACCEPT"
        if not both or X["quality"]["B1_boundary_wrist_comp_step_peak_m"] is None:
            continue
        cm = XB.compare_x(F["quality"], X["quality"])
        cmpx[rid] = cm
        if cm["worse"]:
            worse[r["side"]].append({"req": rid, "worse": cm["worse"], "improved": cm["improved"]})
    mi = {rid for rid, c in cmpx.items() if c["meaningful_improvement"]}
    det["Q"] = {"comparisons": cmpx, "meaningful_improvement": sorted(mi), "worse": worse}
    g["Q_syn"] = "H_E" in mi and len(mi & set(Q_SYN_W117)) >= 2 and not worse["syn"]
    g["Q_real"] = len(mi & set(Q_REAL_HOLDS)) >= 3 and not worse["real"]
    vr = VR.get("requests", {})
    okv = [rid for rid in mi if vr.get(rid, {}).get("improvement_visible") is True and vr[rid].get("arm_plausible_at_proxy_level") is not False
           and vr[rid].get("footprint_display_readable") is True]
    det["QV"] = {"qualifying": okv}
    g["QV"] = any(Rq[r]["side"] == "syn" for r in okv) and any(Rq[r]["side"] == "real" for r in okv)
    V = [k for k in g if k.startswith("V") and not g[k]]
    Cg = [k for k in g if k[:2] in ("C1", "C2", "C3", "C4", "C5", "C6", "C7") and not g[k]]
    if V:
        case = "D"
    elif Cg or (not g["Q_syn"] and not g["Q_real"]):
        case = "C"
    elif g["Q_syn"] and g["Q_real"] and g["QV"]:
        case = "A"
    else:
        case = "B"
    res = {"schema": "w118_decision_v1", "declaration_sha256": a.declaration_sha, "visual_review_sha256": S.sha_file(a.visual_review),
           "gates": g, "failed": {"V": V, "C": Cg}, "case": case, "detail": det}
    dump(out / "decision.json", res)
    print(json.dumps({"case": case, "gates": g, "failed": res["failed"], "MI": sorted(mi), "worse": worse}, indent=1))
    return 0


def F_dec(A, rid):
    w = A["w117_assess"].get(rid, {}).get("candidates", {}).get("F")
    return w["decision"]["decision"] if w else None


# --------------------------------------------------------------------------------------------- manifest ---
def cmd_manifest(a):
    base = Path(a.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != "evidence_manifest.json" and not ({"store", "lineage_store"} & set(p.relative_to(base).parts)):
            files[p.relative_to(base).as_posix()] = {"sha256": S.sha_file(p), "bytes": p.stat().st_size}
    dump(base / "evidence_manifest.json", {"schema": "w118_evidence_manifest_v1", "declaration_sha256": a.declaration_sha,
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
