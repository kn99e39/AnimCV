#!/usr/bin/env python3
"""W122 scratch: real-time animation utility of the FROZEN world-pin guidance on prospectively selected held-out
WITHIN-SOURCE windows of the three W121 sources (evidence only; NON-PRODUCTION).

The three W121 sources are independent of the W102/W105 clip but are NOT new independent sources relative to W121.  W121's
official CASE D is immutable and is never re-decided here.  Nothing in W102/W105/W116-W121 is modified; the W121 builder
(`Src`), the W120 factor solver/assessment and the W116 Blender evaluator are imported and used unchanged.

Stages (each refuses to overwrite its output directory):
  preflight   authenticate S1-S3 originals (full SHA-256 / Commons SHA-1 / licence / every decoded-frame digest / cadence)
              against W121's manifests; frozen checkpoint identity; W122 RE-EXECUTION of the frozen M1/S0 diagnostic
              (S1-S3 + historical clip) compared byte-for-byte; re-run of the W105 FIXED-slot adapter; Blender re-probe of the
              three W121 parents; W113-W121 Worklogs / decisions / W121 evidence manifest unchanged
  windows     held-out window enumeration and deterministic selection from the UNEDITED parent + frozen W116 F feasibility only
  identity    per-window source-frame identity sheets (tracked 2D skeleton) -> the agent records identity_review.json
  calibrate   frozen U / F_r / X_r controls of the W122 population (held-out + W121 anchor + controls); no factor variant
  instrument  pre-outcome instrument preflight: W120 fixtures + 262,144-case precedence, negative expectations, REPAIRED V10
              (declared-telemetry-only exclusion + semantic corruption detected + vacuous exclusions refused) and REPAIRED
              V12/DG5 (verified-REJECT precondition, ACCEPTed variant refused) on frozen historical fixtures
  declare     frozen W122 declaration (code blobs, inputs, windows, identity, population, gates, CASE rule, blind seed)
  build       W121 Src + four W120 factors per request, negatives, W120 synthetic subset (copied unchanged)
  evaluate    Blender (W116 evaluator) of every state, real and synthetic
  assess      W120 assess_record + W122 raw-motion attribution, lineage / exact revert, display / source controls
  render      30 fps real-time Blender renders, blind A/B pair videos, annotated videos, slowed diagnostics, review sheets
  seal        hash the agent's blind ratings BEFORE the label mapping is read
  unblind     map A/B to G0F0/G1F0
  decide      frozen gates and CASE
  manifest    evidence manifest
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
import w121_independent_transfer as W1  # noqa: E402  (read-only reuse)
import w121_independent_transfer_base as AB1  # noqa: E402
import w122_realtime_utility_base as AB  # noqa: E402

BATCH = 122
INSTRUCTION_BLOB = "eebfac50f0419d4c015efbc3bf808759bf020092"
STARTING_HEAD = "75e3cf23e6cf582eb409ad2e23433aba2038f243"
BRANCH = "arch/single_frame_first"
BASE = Path("C:/Users/dna10/animcv-w122")
W121 = Path("C:/Users/dna10/animcv-w121")
W121R = W121 / "results"
W120R = Path("C:/Users/dna10/animcv-w120/results")
W121_WORKLOG = "docs/121_WORKLOG_INDEPENDENT_REAL_VIDEO_WORLD_PIN_TRANSFER.md"
W121_WORKLOG_BLOB = "f1a8c305b7c12052f839543da1a473d2c6c7effa"
W121_DECISION_PREFIX = "90adb6cb95f847e7"
W121_MANIFEST_SHA = "ff914c33c1571affeffa358067696fe2096f99de7790737a94e63ff9f3ad215f"
W121_DECLARATION_SHA = "0616efdfebb940a57dc2d5cbe410b6ca8dfea32ccec1ebb5b1ddf548d38039fc"
W120_DECISION_PREFIX = "1029ce925b0a5a80"
SOURCES = {"S1": ("File:Double Kettlebell Strict Press (side view).webm", "de8feb9d515daa2b"),
           "S2": ("File:Firejuggling 2009 ubt.ogv", "68301b0c35c98448"),
           "S3": ("File:Bent-over row - exercise demonstration video.webm", "f6b90f2f72e9020d")}
HIST_WORKLOGS = W1.HIST_WORKLOGS + [W121_WORKLOG]
HIST_CASES = {113: "D", 114: "D", 115: "D", 116: "B", 117: "C", 118: "D", 119: "D", 120: "B", 121: "D"}
REEXEC = BASE / "results/reexec_perception"
LAUNCHER = BASE / "setup/w122_frozen_launch.py"
SCENE_FPS = 30
UPPER, LOWER = W1.UPPER, W1.LOWER
ARM = W1.ARM
FACTORS = ("G0F0", "G0F1", "G1F0", "G1F1")
CODE_FILES = W1.CODE_FILES + ["scripts/w121_posthoc_checks.py", "scripts/w122_realtime_utility_base.py", "scripts/w122_realtime_utility.py"]
SYN_RIDS = ("H_G", "H_C", "DJ_B", "NG_INFEASIBLE")   # frozen W120 synthetic held-out / near-singular / disjoint-run / infeasible

clean, dump, load, sha, sha1, new_dir, now = W1.clean, W1.dump, W1.load, W1.sha, W1.sha1, W1.new_dir, W1.now


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def blender(*args, log=None):
    return W1.blender(*args, log=log)


def C_():
    return W1.C_()


def S_():
    import w116_semantic_pin_base as S
    return S


def scanon(x):
    return json.dumps(clean(x), sort_keys=True)


# ============================================================================================ preflight ===
def acq_cands():
    return load(W121R / "acquire/acquire.json")["candidates"]


def cmd_preflight(a):
    import cv2  # noqa: F401
    S = S_()
    C = C_()
    out = new_dir(a.out_dir)
    rep = {"schema": "w122_preflight_v1", "label": "AGENT-RUN on Main", "stop": []}
    # ---- repository / history
    rep["git"] = {"head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
                  "branch": git("rev-parse", "--abbrev-ref", "HEAD")}
    rep["git"]["ok"] = (rep["git"]["head"] == STARTING_HEAD and rep["git"]["instruction_blob"] == INSTRUCTION_BLOB
                        and rep["git"]["branch"] == BRANCH)
    wl = {}
    for f in HIST_WORKLOGS:
        wl[f] = {"unchanged_since_start": subprocess.run(["git", "diff", "--quiet", STARTING_HEAD, "--", f], cwd=ROOT).returncode == 0,
                 "complete_marker": "AUTOMATION_BATCH_STATUS: COMPLETE" in (ROOT / f).read_text(encoding="utf-8")}
    rep["worklogs"] = wl
    rep["w121_worklog_blob"] = git("hash-object", W121_WORKLOG)
    d121, d120 = W121R / "decision/decision.json", W120R / "decision/decision.json"
    rep["w121_decision"] = {"sha256": sha(d121), "case": load(d121)["case"]}
    rep["w120_decision"] = {"sha256": sha(d120), "case": load(d120)["case"]}
    rep["w121_declaration_sha256"] = sha(W121R / "declare/declaration.json")
    # W121 evidence manifest: every listed file re-hashed (immutable historical evidence)
    man = load(W121 / "evidence_manifest.json")
    bad = [f for f, h in man["files"].items() if not (W121 / f).exists() or sha(W121 / f) != h]
    rep["w121_manifest"] = {"sha256": sha(W121 / "evidence_manifest.json"), "n_files": man["n_files"], "mismatched": bad[:20],
                            "n_mismatched": len(bad)}
    # ---- originals: full hash, SHA-1, licence, decoded frames, cadence
    acq, ins, per = acq_cands(), load(W121R / "inspect/inspect.json"), load(W121R / "perceive/perceive.json")
    rep["sources"] = {}
    for sid, (t, pfx) in SOURCES.items():
        c = acq[t]
        p = Path(c["download"]["path"])
        meta, dig, _, _ = W1.decode(p)
        pm = np.array(meta["pos_msec"], float)
        r = {"title": t, "path": str(p), "sha256": sha(p), "sha1": sha1(p), "license": c["license"], "artist": c["artist"],
             "url": c["url"], "descriptionurl": c["descriptionurl"], "fps": meta["fps"], "frames": meta["frames"], "sizes": meta["sizes"]}
        r["sha256_prefix_ok"] = r["sha256"].startswith(pfx)
        r["sha256_equals_w121_acquire"] = r["sha256"] == c["download"]["sha256"]
        r["sha1_equals_commons"] = r["sha1"] == c["commons_sha1"]
        r["license_free"] = AB1.license_ok(c["license"])
        r["decoded_frames_equal_w121"] = dig == ins["candidates"][t]["decoded_BGR_uint8_sha256"]
        r["decoded_frame_digest_list_sha256"] = AB.sha256_bytes(json.dumps(dig).encode())
        r["fps_is_30"] = abs(meta["fps"] - 30.0) < 1e-9
        r["cadence_max_abs_ms"] = float(np.max(np.abs(pm[1:] - pm[:-1] - 1000.0 / meta["fps"])))
        r["cadence_ok"] = r["cadence_max_abs_ms"] <= AB1.CADENCE_TOL_MS
        r["not_historical"] = r["sha256"] not in AB1.EXCLUDED_SHA256 and ins["candidates"][t]["dup_vs_historical_longest_run"] < AB1.DUP_RUN_FRAMES
        pc = per["candidates"][t]
        r["segment"] = pc["segment"]["window"]
        r["observation_prediction_sha256_w121"] = pc["observation_prediction_sha256"]
        r["M1_identity_checkpoints"] = pc["M1_identity_checkpoints"]
        r["ok"] = all(r[k] for k in ("sha256_prefix_ok", "sha256_equals_w121_acquire", "sha1_equals_commons", "license_free",
                                     "decoded_frames_equal_w121", "fps_is_30", "cadence_ok", "not_historical"))
        rep["sources"][sid] = r
    # ---- frozen model identity
    fi = per["frozen_identity"]
    w = {}
    for f in sorted(W1.FROZEN["weights"].glob("*.pth")):
        w[f.name] = sha(f)
    rep["frozen_identity"] = {"checkpoint_sha256": sha(W1.FROZEN["checkpoint"]), "gate_report_sha256": sha(W1.FROZEN["gate_report"]),
                              "lock_sha256": sha(W1.FROZEN["lock"]), "diagnostic_script_sha256": sha(W1.FROZEN["diagnostic_script"]),
                              "weights": w}
    m1 = {v for s in rep["sources"].values() for v in s["M1_identity_checkpoints"].values()}
    rep["frozen_identity"]["ok"] = (rep["frozen_identity"]["checkpoint_sha256"] == fi["checkpoint_sha256"]
                                    and rep["frozen_identity"]["gate_report_sha256"] == fi["gate_report_sha256"]
                                    and rep["frozen_identity"]["lock_sha256"] == fi["lock_sha256"]
                                    and rep["frozen_identity"]["diagnostic_script_sha256"] == fi["diagnostic_script_sha256"]
                                    and m1 <= set(w.values()))
    # ---- W122 re-execution of the frozen M1/S0 diagnostic (outputs written by setup/run_reexec.sh)
    rx = {}
    pairs = {"historical": (W1.HIST_OBS, None)} | {sid: (W121R / "perceive/candidates" / W1.safe(t) / "m1s0/observations_predictions.npz", sid)
                                                     for sid, (t, _) in SOURCES.items()}
    for k, (ref, sid) in pairs.items():
        npz = REEXEC / k / "observations_predictions.npz"
        x = {"reexec_npz": str(npz), "exists": npz.exists()}
        if npz.exists():
            za, zb = np.load(npz), np.load(ref)
            x.update(file_sha256=sha(npz), reference_sha256=sha(ref), file_byte_equal=sha(npz) == sha(ref),
                     arrays_byte_equal={n: bool(np.array_equal(za[n], zb[n]) and za[n].dtype == zb[n].dtype) for n in zb.files},
                     report_ok=load(REEXEC / k / "report.json").get("checks") if (REEXEC / k / "report.json").exists() else None)
            x["ok"] = all(x["arrays_byte_equal"].values()) and set(za.files) == set(zb.files)
        else:
            x["ok"] = False
        rx[k] = x
    rx["launcher_sha256"] = sha(LAUNCHER)
    rep["perception_reexecution"] = rx
    # ---- W105 adapter re-run: historical W102 predictions -> W105 FIXED values; re-executed S1-S3 predictions -> W121 cases/v0
    rc = W1.rig_ctx()
    rep["adapter_historical"] = W1.adapter_fidelity(rc)
    prep = load(W121R / "prepare/prepare.json")
    rep["parents"] = {}
    for sid, (t, _) in SOURCES.items():
        sd = W121R / "prepare" / sid
        v0 = load(sd / "parent_state.json")
        z = np.load(REEXEC / sid / "observations_predictions.npz") if (REEXEC / sid / "observations_predictions.npz").exists() else \
            np.load(W121R / "perceive/candidates" / W1.safe(t) / "m1s0/observations_predictions.npz")
        lo, hi = prep["sources"][sid]["frames"]
        cases = []
        for i in range(lo, hi + 1):
            fr = W1.fixed_slot(z["predictions"][i], rc)
            fr.update(frame_index=i, time_s=i / 30.0, identity="UNKNOWN (GT-free; tracked-subject continuity unverified)")
            cases.append(fr)
        vals = W1.state_values_from_cases(cases)
        rec = {"w121_state_id": v0["state_id"], "state_verifies": S.verify_state(v0), "state_id_matches_prepare": v0["state_id"] == prep["sources"][sid]["parent_state_id"],
               "values_reproduced_from_reexec": S.canon({c: [[float(x) for x in q] for q in vals[c]] for c in C.REAL_CH}) == S.canon(v0["values"]),
               "w121_cases_equal": W1.json_eq(load(sd / "video_cases.json")["cases"][sid]["frames"], cases),
               "source_binding": {k: v0["source"][k] for k in ("clip_sha256", "observation_prediction_sha256", "video_subject_identity")},
               "source_binding_ok": v0["source"]["clip_sha256"] == rep["sources"][sid]["sha256"]
               and v0["source"]["observation_prediction_sha256"] == rep["sources"][sid]["observation_prediction_sha256_w121"]
               and v0["source"]["video_subject_identity"].startswith("UNKNOWN"),
               "time_s_ok": all(abs(tt - k / 30.0) < 1e-12 for tt, k in zip(v0["source"]["time_s"], v0["frames"]))}
        # Blender re-probe of the unchanged parent states file
        pd = new_dir(out / f"probe_{sid}")
        blender(ROOT / "scripts/w116_blender_stages.py", "--", "probe", "--rig", W1.RIG_FBX, "--rest", W1.RIG_REST, "--states",
                sd / "parent_states.json", "--out", pd / "probe.json", log=pd / "probe.log")
        pr, pr0 = load(pd / "probe.json"), load(sd / "probe.json")
        rig = C.rig_from(W1.RIG_REST, pr["rig"])
        ok_b, failed_b = C.bindings_for(C.REAL_BINDING_CANDIDATES, "BaseRig", pr["rig"]["rest_snapshot_digest"], pr, v0, rig)
        rec.update(probe_value_digest=pr["states"]["v0"]["value_digest"],
                   probe_digest_equal_w121=pr["states"]["v0"]["value_digest"] == prep["sources"][sid]["probe_value_digest"],
                   probe_frames_bit_equal_w121=pr["states"]["v0"]["frames"] == pr0["states"]["v0"]["frames"],
                   fk_max_error_m=C.fk_error(rig, v0, pr), bindings_equal_w121=W1.json_eq(ok_b, load(sd / "bindings.json")["bound"]),
                   left_wrist_bound=any(b["semantic_joint"] == "left_wrist" for b in ok_b.values()))
        rec["ok"] = all(rec[k] for k in ("state_verifies", "state_id_matches_prepare", "values_reproduced_from_reexec", "w121_cases_equal",
                                         "source_binding_ok", "time_s_ok", "probe_digest_equal_w121", "probe_frames_bit_equal_w121",
                                         "bindings_equal_w121", "left_wrist_bound")) and rec["fk_max_error_m"][0] <= 1e-5
        rep["parents"][sid] = rec
    rep["rig"] = {"fbx_sha256": sha(W1.RIG_FBX), "rest_sha256": sha(W1.RIG_REST)}
    rep["rig"]["ok"] = rep["rig"]["fbx_sha256"] == W1.RIG_FBX_SHA and rep["rig"]["rest_sha256"] == W1.RIG_REST_SHA
    # ---- stop conditions
    if not rep["git"]["ok"]:
        rep["stop"].append("git state")
    if not all(v["unchanged_since_start"] and v["complete_marker"] for v in wl.values()):
        rep["stop"].append("historical Worklogs")
    if rep["w121_worklog_blob"] != W121_WORKLOG_BLOB:
        rep["stop"].append("W121 Worklog blob")
    if not (rep["w121_decision"]["sha256"].startswith(W121_DECISION_PREFIX) and rep["w121_decision"]["case"] == "D"):
        rep["stop"].append("W121 decision")
    if not (rep["w120_decision"]["sha256"].startswith(W120_DECISION_PREFIX) and rep["w120_decision"]["case"] == "B"):
        rep["stop"].append("W120 decision")
    if rep["w121_declaration_sha256"] != W121_DECLARATION_SHA or rep["w121_manifest"]["sha256"] != W121_MANIFEST_SHA or bad:
        rep["stop"].append("W121 declaration / evidence manifest")
    for sid in SOURCES:
        if not rep["sources"][sid]["ok"]:
            rep["stop"].append(f"SOURCE BLOCKED {sid}")
        if not rep["parents"][sid]["ok"]:
            rep["stop"].append(f"RIG/PARENT BLOCKED {sid}")
    if not rep["frozen_identity"]["ok"]:
        rep["stop"].append("frozen model identity")
    if not all(rx[k]["ok"] for k in pairs):
        rep["stop"].append("frozen perception re-execution not byte-equal")
    if not rep["adapter_historical"]["ok"] or not rep["rig"]["ok"]:
        rep["stop"].append("adapter / rig identity")
    dump(out / "preflight.json", rep)
    print(json.dumps({"stop": rep["stop"], "sources": {s: r["ok"] for s, r in rep["sources"].items()},
                      "parents": {s: r["ok"] for s, r in rep["parents"].items()},
                      "reexec": {k: rx[k]["ok"] for k in pairs}, "manifest_mismatch": len(bad)}, indent=1))
    return 1 if rep["stop"] else 0


# ============================================================================================ windows ===
class W22Src(W1.Src):
    """W121 Src (unchanged solver path) with the W122 population: held-out NL/NH windows, W121 anchor windows (W121 request ids,
    so anchor states must reproduce W121's), per-source controls on the held-out lower window and W122 negatives."""

    def __init__(self, sid, windows=None):
        super().__init__(sid, W121R / "prepare")
        self.ho = (windows or {}).get(sid, {})

    def declare_requests(self):
        super().declare_requests()          # W121 anchors: S*_{LOW,MID,HIGH}_{none,RBpos,RBneg}, S*_ZB, S*_{REACH,DOF,CONFLICT,WRONGBIND}
        for r in self.req.values():
            r["population"] = "ANCHOR_W121_REPLICATION" if r["role"] == "evaluation" else "ANCHOR_W121_CONTROL"
        for W, h in sorted(self.ho.items()):
            if h is None:
                continue
            for e, edit in AB1.HOLD_EDITS.items():
                self.req[f"{self.sid}_{W}_{e}"] = {"window": W, "edit": edit, "edit_key": e, "cap": h["capture"], "pin_frames": list(h["frames"]),
                                                   "pin_decl": [("left_wrist", "capture", None, None)], "auth": ARM, "m_max": 8,
                                                   "role": "evaluation", "population": "HELD_OUT"}
        lo = self.ho.get("NL")
        if lo is None:
            return
        self.req[f"{self.sid}_NZB"] = {"window": "NL", "edit": "SIDE_BEND_5DEG_POS", "edit_key": "RBpos", "cap": lo["capture"],
                                       "pin_frames": list(lo["frames"]), "pin_decl": [("left_wrist", "capture", None, None)], "auth": ARM,
                                       "m_max": 0, "role": "control", "population": "HELD_OUT_CONTROL"}
        sp = {"NREACH": ([("left_wrist", "offset", [0.0, 0.0, 1.0], None)], ARM),
              "NDOF": ([("left_wrist", "capture", None, None)], [LOWER]),
              "NCONFLICT": ([("left_wrist", "capture", None, None), ("left_wrist", "offset", [0.0, 0.0, 0.02], None)], ARM),
              "NWRONGBIND": ([("left_wrist", "capture", None, "right_wrist")], ARM)}
        for k, (pd, auth) in sp.items():
            self.req[f"{self.sid}_{k}"] = {"window": "NL", "edit": "SIDE_BEND_5DEG_POS", "edit_key": "RBpos", "cap": lo["capture"],
                                           "pin_frames": list(lo["frames"]), "pin_decl": pd, "auth": auth, "m_max": 8, "role": "special",
                                           "population": "HELD_OUT_CONTROL"}

    def control_record(self, rid):
        R = super().control_record(rid)
        R["population"] = self.req[rid]["population"]
        return R

    NEG_BASE_ORDER = ["NL_RBpos", "NH_RBpos", "NL_RBneg", "NH_RBneg", "NL_none", "NH_none"]

    def neg_base(self):
        for w in self.NEG_BASE_ORDER:
            rid = f"{self.sid}_{w}"
            R = self.recs.get(rid)
            if R is None or R.get("contract_invalid"):
                continue
            if set(R["F_statuses"].values()) - {"SOLVED", "NO_CHANGE_NEEDED"}:
                continue
            m = R["margins"]
            ent = [v for k, v in m.items() if k.endswith("entry")]
            rel = [v for k, v in m.items() if k.endswith("release")]
            if ent and rel and min(ent) >= 1 and min(rel) >= 2:
                return rid
        return None

    def negatives(self, base):
        """The W121 negative-control suite (identical patterns and expectations, W121 NEG table), re-targeted to the W122
        held-out base request, the held-out zero-budget control and the held-out special requests."""
        import w118_transition_edit_base as XB
        r, s = self.recs, self.sid
        v, i = self.solve(f"{s}_NZB", "G1F1", m_max=XB.M_MAX)
        self.neg("NG_BUDGET", f"{s}_NZB", "G1F1", vals=v, info=i)
        mm = dict(r[base]["margins"])
        key = next(k for k in mm if k.endswith("release"))
        mm[key] = max(0, mm[key] - 1)
        v, i = self.solve(base, "G0F0", margins=mm)
        self.neg("NG_MARGIN", base, "G0F0", vals=v, info=i)
        self.neg("NG_PARENT", base, "G1F0", fair_mut=lambda fr: fr.__setitem__("parent_id", "0" * 64))
        self.neg("NG_RIG", base, "G1F1", vals=r[base]["_fv"]["G1F1"], rig_id="W122_OTHER_RIG")
        self.neg("NG_CDIGEST", base, "G0F0", fair_mut=lambda fr: fr.__setitem__("constraint_digests", ["W122_TAMPERED_DIGEST"]))
        self.neg("NG_AUTH", base, "G1F0", fair_mut=lambda fr: fr.__setitem__("auth", [UPPER]))
        pinb = r[base]["pin_frames"]
        mid = pinb[len(pinb) // 2]
        self.neg("NG_PIN_COMBINED", base, "G0F1",
                 vals=self.tam(base, "G0F1", lambda vv, c: vv[mid].__setitem__(LOWER, W1.rot_local(vv[mid][LOWER], [0, 0, 1], 2.0))))

        def false_solved(vv, c):
            for k in pinb:
                vv[k] = dict(c[k])
        self.neg("NG_FALSE", base, "G1F1", vals=self.tam(base, "G1F1", false_solved), statuses={str(k): "SOLVED" for k in pinb})
        v, i = self.solve(base, "G0F0", fault="skip_lower_aim")
        self.neg("NG_NONCONV", base, "G0F0", vals=v, info=i)
        tf = sorted(k for k in r[base]["factors"]["G1F0"]["info"]["planned_frames"] if k not in set(pinb))[0]
        self.neg("NG_BONE", base, "G1F0", vals=self.tam(base, "G1F0", lambda vv, c: vv[tf].__setitem__("spine_01", W1.rot_local(vv[tf]["spine_01"], [1, 0, 0], 1.0))))
        self.neg("NG_ROOT", base, "G0F0", vals=self.tam(base, "G0F0", lambda vv, c: vv[tf].__setitem__("__object__", W1.rot_local(vv[tf]["__object__"], [0, 0, 1], 1.0))))
        bud = set(r[base]["factors"]["G1F0"]["info"]["budget_frames"])
        far = next(k for k in self.frames if k not in bud)
        self.neg("NG_OUTSIDE", base, "G1F0", vals=self.tam(base, "G1F0", lambda vv, c: vv.__setitem__(far, dict(c[far], **{UPPER: W1.rot_local(c[far][UPPER], [0, 0, 1], 1.0)}))))
        self.neg("NG_PLANNED", base, "G0F1", vals=self.tam(base, "G0F1", lambda vv, c: None),
                 planned=lambda yi_: [k for k in yi_["planned_frames"] if k != min(yi_["planned_frames"])])
        self.neg("NG_DISPLAY", base, "G1F1", display_corrupt="DY1")
        sib = dict(r[base]["_Fv"])
        for w in ("NL", "NH"):
            o = f"{s}_{w}_{r[base]['edit_key']}"
            if o != base and o in r and not r[o].get("contract_invalid"):
                sib.update(r[o]["_Fv"])
        if set(sib) == set(r[base]["_Fv"]):      # no held-out sibling: use the W121 anchor windows of the same edit
            for w in ("LOW", "MID", "HIGH"):
                o = f"{s}_{w}_{r[base]['edit_key']}"
                if o in r and not r[o].get("contract_invalid"):
                    sib.update(r[o]["_Fv"])
        self.states["NG_SHARED__Fshared"] = self.C.patched_state(self.v0, {k: (sib[k] if k in sib else r[base]["_ctrl"][k]) for k in self.frames},
                                                                 {"kind": "F_CANDIDATE_FAMILY_SHARED", "request": "NG_SHARED"})
        self.neg("NG_SHARED", base, "G0F0", F_state="NG_SHARED__Fshared")
        for nid, sp in (("NG_REACH", "NREACH"), ("NG_DOF", "NDOF"), ("NG_CONFLICT", "NCONFLICT"), ("NG_WRONG_BINDING", "NWRONGBIND")):
            self.neg(nid, f"{s}_{sp}", W1.NEG[nid][1])


def f_statuses(src, edit_key, frames, cap):
    """Frozen W116 analytic framewise resolve of a capture pin on `frames` under the edit's U (F feasibility only)."""
    S = src.S
    edit = AB1.HOLD_EDITS[edit_key]
    _, ctrl = src.control(edit)
    pins = src.pins("W122_FEAS", [("left_wrist", "capture", None, None)], frames, ARM, cap)
    return [S.resolve_frame(src.rig, ctrl[k], src.rp(pins, k), ARM, method="analytic")[0] for k in frames]


def cmd_windows(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight / "preflight.json")
    if pf["stop"]:
        raise SystemExit("STOP: preflight")
    per = load(W121R / "perceive/perceive.json")
    rep = {"schema": "w122_windows_v1", "rule": AB.WINDOW_RULE, "scope": "UNEDITED parent Blender wrist path (W121 probe, re-probed "
           "bit-equal in preflight) + frozen W116 framewise F statuses only; no transition or factor variant exists", "sources": {}}
    sel = {}
    for sid, (t, _) in SOURCES.items():
        src = W1.Src(sid, W121R / "prepare")
        wrist = {fr["frame_index"]: fr["world"]["lowerarm_l"]["tail"] for fr in src.probe["states"]["v0"]["frames"]}
        w121w = [[h["frames"][0], h["frames"][-1]] for h in src.holds.values()]
        flags = per["candidates"][t]["coverage"]["possible_subject_switch_frames"]
        pre = AB.enumerate_windows(src.frames, wrist, w121w, None, flags)
        feas = {}
        for c in pre:
            if c["reasons"] == ["F_NOT_RESOLVED"]:
                feas[c["capture"]] = {e: f_statuses(src, e, c["frames"], c["capture"]) for e in AB.EDITS}
        cands = AB.enumerate_windows(src.frames, wrist, w121w, feas, flags)
        ch = AB.select_windows(cands)
        steps = AB.steps(wrist, src.frames)

        def strat(lo, hi):
            return AB.stratum([steps[k] for k in range(lo - AB.M_MAX + 1, hi + AB.M_MAX + 2) if k in steps])
        why = {}
        for c in cands:
            for r in c["reasons"] or ["FEASIBLE"]:
                why[r.split(":")[0]] = why.get(r.split(":")[0], 0) + 1
        chosen = {}
        for tag, key in (("NL", "LOWER"), ("NH", "HIGHER")):
            c = ch[key]
            chosen[tag] = None if c is None else {"capture": c["capture"], "frames": c["frames"], "D_m": c["D_m"], "domain": c["domain"],
                                                  **strat(c["frames"][0], c["frames"][-1])}
        anchors = {W: {"capture": h["capture"], "frames": h["frames"], "D_m": h["D_m"], **strat(h["frames"][0], h["frames"][-1])}
                   for W, h in src.holds.items()}
        rep["sources"][sid] = {"segment": [src.frames[0], src.frames[-1]], "w121_windows": w121w, "track_proxy_flags": flags,
                               "n_candidates": len(cands), "reason_counts": why, "median_D_m": ch["median_D_m"],
                               "n_feasible": ch["n_feasible"], "n_feasible_above_median": ch["n_feasible_above_median"],
                               "higher_reason": ch["higher_reason"], "chosen": chosen, "anchors_w121": anchors, "candidates": cands}
        sel[sid] = chosen
    dump(out / "windows.json", rep)
    dump(out / "selection.json", sel)
    print(json.dumps({s: {"n": r["n_candidates"], "feasible": r["n_feasible"], "why": r["reason_counts"], "median": r["median_D_m"],
                          "chosen": {k: v and (v["frames"][0], v["frames"][-1], round(v["D_m"], 3), v["stratum"]) for k, v in r["chosen"].items()},
                          "higher_reason": r["higher_reason"]} for s, r in rep["sources"].items()}, indent=1))


# ============================================================================================ identity ===
def window_list(sel_sid, anchors):
    out = {}
    for tag, h in sel_sid.items():
        if h:
            out[tag] = h
    for W, h in anchors.items():
        out[W] = h
    return out


def cmd_identity(a):
    """Per window: every source frame of [domain - CONTEXT, domain + CONTEXT] cropped around the tracked subject with the frozen
    tracked 2D skeleton (left arm yellow) + the full frame at the capture frame.  Evidence only for the agent's identity review."""
    import cv2
    out = new_dir(a.out_dir)
    win = load(a.windows / "windows.json")
    per = load(W121R / "perceive/perceive.json")
    acq = acq_cands()
    ev = {}
    for sid, (t, _) in SOURCES.items():
        z = np.load(W121R / "perceive/candidates" / W1.safe(t) / "m1s0/observations_predictions.npz")
        Wd, Hd = per["candidates"][t]["size"]
        ws = window_list(win["sources"][sid]["chosen"], win["sources"][sid]["anchors_w121"])
        seg = win["sources"][sid]["segment"]
        need = set()
        rng = {}
        for W, h in ws.items():
            lo = max(seg[0], h["frames"][0] - AB.M_MAX - AB.CONTEXT_FRAMES)
            hi = min(seg[1], h["frames"][-1] + AB.M_MAX + AB.CONTEXT_FRAMES)
            rng[W] = (lo, hi)
            need |= set(range(lo, hi + 1))
        frames = W1.source_frames(acq[t]["download"]["path"], sorted(need))
        x, v = z["input_2d"], z["input_valid"]
        ev[sid] = {}
        for W, (lo, hi) in rng.items():
            ks = list(range(lo, hi + 1))
            boxes = []
            for k in ks:
                p = x[k][v[k]][:, :2] * np.array([Wd, Hd])
                boxes.append(None if len(p) < 2 else [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())])
            ious = []
            for i in range(1, len(boxes)):
                A_, B_ = boxes[i - 1], boxes[i]
                if A_ is None or B_ is None:
                    ious.append(None)
                    continue
                ix = max(0.0, min(A_[2], B_[2]) - max(A_[0], B_[0]))
                iy = max(0.0, min(A_[3], B_[3]) - max(A_[1], B_[1]))
                u = (A_[2] - A_[0]) * (A_[3] - A_[1]) + (B_[2] - B_[0]) * (B_[3] - B_[1]) - ix * iy
                ious.append(ix * iy / u if u > 0 else None)
            cols = 9
            rows = math.ceil(len(ks) / cols)
            cell = 200
            sheet = np.full((rows * (cell + 18) + 30, cols * cell, 3), 245, np.uint8)
            cv2.putText(sheet, f"{sid} window {W} frames {lo}-{hi} (pin {ws[W]['frames'][0]}-{ws[W]['frames'][-1]}): tracked subject crop + frozen 2D skeleton",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
            for i, k in enumerate(ks):
                im = frames[k].copy()
                for (p, c) in ((11, 12), (12, 13), (14, 15), (15, 16), (11, 14), (1, 4), (11, 1), (14, 4), (1, 2), (2, 3), (4, 5), (5, 6), (0, 7), (7, 8)):
                    if v[k][p] and v[k][c]:
                        cv2.line(im, (int(x[k, p, 0] * Wd), int(x[k, p, 1] * Hd)), (int(x[k, c, 0] * Wd), int(x[k, c, 1] * Hd)),
                                 (0, 220, 255) if (p, c) in ((11, 12), (12, 13)) else (60, 255, 60), max(2, Wd // 300), cv2.LINE_AA)
                b = boxes[i]
                if b is None:
                    crop = im
                else:
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    s = max(b[2] - b[0], b[3] - b[1]) * 1.6 + 40
                    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
                    crop = im[y0:int(min(Hd, cy + s / 2)), x0:int(min(Wd, cx + s / 2))]
                sc = cell / max(crop.shape[:2])
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))))
                r_, c_ = divmod(i, cols)
                y, xx = 30 + r_ * (cell + 18), c_ * cell
                sheet[y:y + crop.shape[0], xx:xx + crop.shape[1]] = crop
                pin = ws[W]["frames"][0] <= k <= ws[W]["frames"][-1]
                cv2.putText(sheet, f"f{k}{' PIN' if pin else ''}", (xx + 4, y + cell + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (200, 60, 20) if pin else (40, 40, 40), 1, cv2.LINE_AA)
            cv2.imwrite(str(out / f"identity_{sid}_{W}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
            # full frame at the capture frame, tracked box drawn
            cap = ws[W]["frames"][0]
            full = frames[cap].copy()
            bb = boxes[ks.index(cap)]
            if bb:
                cv2.rectangle(full, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)
            sc = 900 / max(full.shape[:2])
            cv2.imwrite(str(out / f"identity_{sid}_{W}_full.jpg"), cv2.resize(full, (int(full.shape[1] * sc), int(full.shape[0] * sc))),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            ev[sid][W] = {"frames": [lo, hi], "bbox_iou_consecutive_min": min([q for q in ious if q is not None] or [None]) if any(q is not None for q in ious) else None,
                          "n_iou_below_0.5": sum(1 for q in ious if q is not None and q < 0.5), "boxes_missing": sum(1 for b in boxes if b is None),
                          "sheet": f"identity_{sid}_{W}.jpg", "sheet_sha256": sha(out / f"identity_{sid}_{W}.jpg"),
                          "mean_person_candidates_whole_clip": per["candidates"][t]["tracking"]["mean_person_candidates_per_frame"]}
    dump(out / "identity_evidence.json", {"schema": "w122_identity_evidence_v1", "rule": "VERIFIED only with affirmative source-backed "
                                          "evidence (agent inspection of every frame of the interval shows the frozen tracked skeleton "
                                          "on the same person, AND no other person could be confused); otherwise UNKNOWN", "windows": ev})
    print(json.dumps(ev, indent=1))


# ============================================================================================ calibrate ===
def held_out_windows(sel):
    return {sid: {"NL": v.get("NL"), "NH": v.get("NH")} for sid, v in sel.items()}


def frozen_controls_of(src):
    out = W1.frozen_controls_of(src)
    for rid in out:
        out[rid]["population"] = src.recs[rid]["population"]
    return out


def cmd_calibrate(a):
    out = new_dir(a.out_dir)
    sel = load(a.windows / "selection.json")
    rep = {"schema": "w122_calibrate_v1", "scope": "frozen controls U/F_r/X_r (Python only, no factor variant, no candidate Blender "
           "evaluation)", "sources": {}}
    controls = {}
    hw = held_out_windows(sel)
    for sid in SOURCES:
        src = W22Src(sid, hw)
        src.build_controls()
        controls[sid] = frozen_controls_of(src)
        rep["sources"][sid] = {"requests": sorted(src.recs), "neg_base": src.neg_base(),
                               "F_status_summary": {rid: sorted(set((R.get("F_statuses") or {}).values())) for rid, R in src.recs.items()},
                               "population": {rid: R["population"] for rid, R in src.recs.items()}}
    dump(out / "frozen_controls.json", controls)
    rep["frozen_controls_sha256"] = sha(out / "frozen_controls.json")
    rep["neg_source"] = next((sid for sid in SOURCES if rep["sources"][sid]["neg_base"]), None)
    # anchor replication of W121's frozen controls (same request ids, same code)
    w121fc = load(W121R / "calibrate/frozen_controls.json")
    rep["anchor_controls_equal_w121"] = {sid: sorted(r for r, x in w121fc[sid].items()
                                                     if scanon({k: v for k, v in controls[sid][r].items() if k != "population"}) != scanon(x))
                                         for sid in SOURCES}
    dump(out / "calibration.json", rep)
    print(json.dumps({"neg_source": rep["neg_source"], "anchor_mismatch": rep["anchor_controls_equal_w121"],
                      "sources": {s: {"neg_base": v["neg_base"], "n": len(v["requests"]),
                                      "status": {r: x for r, x in v["F_status_summary"].items() if v["population"][r].startswith("HELD")}}
                                  for s, v in rep["sources"].items()}}, indent=1))


# ============================================================================================ instrument ===
def cmd_instrument(a):
    """Pre-outcome instrument preflight on frozen fixtures and on frozen HISTORICAL W121 artefacts used read-only as fixtures
    (this never re-decides W121; its official CASE D is unchanged)."""
    import w118_transition_edit_base as XB
    import w119_endpoint_transition as W9
    import w120_factor_attribution as W20
    C = C_()
    out = new_dir(a.out_dir)
    rep = {"schema": "w122_instrument_v1", "label": "pre-outcome; no W122 factor variant exists"}
    rep["w120_fixture_preflight"] = W20.fixture_preflight(out / "fixture_store")
    rep["negative_expectations"] = W1.neg_expectation_consistency()
    # ---- V10 repaired: historical W121 r1 / r2 builds as the fixture (they differ ONLY in factor info wall_s)
    b1, b2 = load(W121R / "build_r1/build_report.json"), load(W121R / "build_r2/build_report.json")
    v10 = {}
    for sid in b1["sources"]:
        r1, r2 = b1["sources"][sid]["records"], b2["sources"][sid]["records"]
        raw = AB.canon(r1) == AB.canon(r2)
        rr = AB.records_repeatable(r1, r2)
        cor, path = AB.semantic_corruption(r2)
        rc = AB.records_repeatable(r1, cor)
        v10[sid] = {"raw_records_equal_including_wall_s": raw, "repaired_equal": rr["equal"], "n_compared_leaves": rr["n_compared_leaves"],
                    "n_removed_wall_s": rr["n_removed"], "non_vacuous": rr["non_vacuous"], "corrupted_path": path,
                    "corruption_detected": not rc["equal"], "corruption_differing": rc["differing_leaves"],
                    "states_sha_equal": b1["sources"][sid]["states_sha256"] == b2["sources"][sid]["states_sha256"]}
    refused = {}
    for name, pats in (("strip_all", ("*",)), ("strip_info", ("*.factors.*.info",)), ("undeclared_planned", ("*.x_info.planned_frames",)),
                       ("strip_request_status", ("*.factors.*.info.request_status",))):
        try:
            AB.validate_exclusions(pats)
            refused[name] = False
        except AB.ExclusionError as e:
            refused[name] = str(e)
    rep["V10_repaired_fixture"] = {"per_source": v10, "vacuous_or_semantic_exclusions_refused": refused,
                                   "declared_exclusions": list(AB.NONDETERMINISTIC_RECORD_PATHS),
                                   "all_ok": all(x["repaired_equal"] and x["non_vacuous"] and x["n_removed_wall_s"] > 0 and x["corruption_detected"]
                                                 and not x["raw_records_equal_including_wall_s"] for x in v10.values()) and all(refused.values())}
    # ---- V12 / DG5 repaired: verified-REJECT precondition on historical W121 variants (read-only fixtures)
    A = load(W121R / "assess/assess_report.json")
    ev = load(W121R / "eval_e1/S1_eval.json")
    frames = [fr["frame_index"] for fr in ev["states"]["v0"]["frames"]]
    dg = {}
    for rid in ("S1_REACH", "S1_HIGH_RBpos", "S1_MID_RBpos"):
        y = A["requests"][rid]["variants"]["G1F0"]
        rec = b1["sources"]["S1"]["records"][rid]
        name = rec["factors"]["G1F0"]["state"]
        try:
            s = AB.dg5_inject(y["display_spec_result"], y["commit"]["decision"], W9.corrupt_display)
            chk = XB.check_display(s, C.ev_frames(ev, name), C.ev_frames(ev, rec["U"]), frames, rec["pin_frames"], ev["states"][name]["state_id"],
                                   decision=y["commit"]["decision"])
            dg[rid] = {"decision": y["commit"]["decision"], "executed": True, "status_before": y["display_spec_result"]["status"],
                       "status_after": s["status"], "truth_check_fails": not chk["ok"], "errors": chk["errors"]}
        except AB.PreconditionError as e:
            dg[rid] = {"decision": y["commit"]["decision"], "executed": False, "refused": str(e)}
    del ev
    rep["V12_DG5_repaired_fixture"] = {"per_variant": dg,
                                       "all_ok": dg["S1_REACH"].get("truth_check_fails") is True and dg["S1_HIGH_RBpos"].get("truth_check_fails") is True
                                       and dg["S1_MID_RBpos"]["executed"] is False and "PRECONDITION" in dg["S1_MID_RBpos"]["refused"]}
    rep["all_ok"] = (rep["w120_fixture_preflight"]["all_ok"] and rep["negative_expectations"]["all_ok"]
                     and rep["V10_repaired_fixture"]["all_ok"] and rep["V12_DG5_repaired_fixture"]["all_ok"])
    dump(out / "instrument.json", rep)
    print(json.dumps({"all_ok": rep["all_ok"], "w120": rep["w120_fixture_preflight"]["all_ok"],
                      "prec": rep["w120_fixture_preflight"]["precedence_exhaustive"]["n"], "neg": rep["negative_expectations"]["all_ok"],
                      "V10": {s: {k: v for k, v in x.items() if k != "corruption_differing"} for s, x in v10.items()}, "refused": refused,
                      "DG5": dg}, indent=1, default=str))
    return 0 if rep["all_ok"] else 1


# ============================================================================================ declare ===
DISPLAY_NEG = {"DG1": ("BASE", "G0F0", "DY1", "first changed out-of-pin frame relabelled UNCHANGED"),
               "DG2": ("BASE", "G0F1", "DY2", "a changed arm channel row dropped"),
               "DG3": ("FIRST_MULTIRUN", "ANY", "DY3", "disjoint changed runs collapsed into one envelope (first real held-out variant "
                       "with >= 2 changed runs in declared order; else the frozen W120 synthetic DJ_B G1F0)"),
               "DG4": ("BASE", "G1F1", "DY4", "tampered output drawn from the declared planned domain"),
               "DG5": ("NREACH", "G1F0", "DY5", "REJECTed result shown as ACCEPTED: the neg source's controlled infeasible held-out request "
                       "S*_NREACH (+1 m); precondition decision == REJECT and displayed status REJECTED ASSERTED before injection"),
               "DG5_SYN": ("NG_INFEASIBLE", "G1F0", "DY5", "W120 synthetic genuine REJECT, same verified precondition")}
GATES = {
    "V1_source_authentication": "preflight: HEAD/blob/branch; W113-W121 Worklogs unchanged with markers; W121 Worklog blob; W121 decision "
                                "(90adb6cb, D) / declaration / evidence manifest (every file re-hashed) unchanged; W120 decision B; S1-S3 "
                                "full SHA-256 = W121 acquire, prefixes, Commons SHA-1, free licence, EVERY decoded frame digest = W121, "
                                "30 fps cadence-consistent, not the W102 clip",
    "V2_frozen_perception": "checkpoint / weights / gate report / lock / diagnostic identities; W122 RE-EXECUTION of the frozen M1/S0 "
                            "diagnostic on S1-S3 and the historical clip byte-equal to W121 / W102 stored arrays",
    "V3_rig_adapter_and_parent": "W105 FIXED adapter reproduces W105 (199/199); re-executed predictions -> W121 cases and parent values; "
                                 "parent state ids verify; W122 Blender re-probe value digest and frames bit-equal W121; FK <= 1e-5 m; "
                                 "bindings equal W121; rig identity",
    "V4_prospective_window_selection": "windows/selection computed before calibrate from U + F only; selection.json hash in the declaration; "
                                       "selection recomputed at decide from the stored candidates equals; held-out automatic domains "
                                       "disjoint from W121 windows +-8; outcomes_inspected false",
    "V5_baseline_and_anchor_reproduction": "every request's frozen U/F_r/X_r controls equal in both builds; every W121 anchor request's U/F_r/"
                                           "X_r and four factor state ids equal W121's build state ids",
    "V6_clean_isolated_baselines": "every contract-valid evaluation/control request: F_r value-level and Blender-level isolation",
    "V7_factor_matching": "every contract-valid evaluation/control request and factor: ab_fairness ok, solver digest equal, "
                          "cross_factor_identity; every zero-budget control budget = pin frames for all four",
    "V8_predeclaration_and_instrument": "declaration hash at every stage; outcomes_inspected false; frozen-controls hash; W120 fixtures "
                                        "(262,144-case precedence, 0 mismatches); negative expectations = predictor; REPAIRED V10 and V12 "
                                        "fixture preflight all_ok; code blobs at build equal declared",
    "V9_blender_evaluation_and_population": "every state of both builds and the synthetic subset Blender-evaluated; every declared request has "
                                            "F/X/4 variant records; every contract-valid request has every pin frame evaluated",
    "V10_repaired_repeatability": "both builds: states files byte-identical; records equal after removing ONLY the declared telemetry paths "
                                  "(*.factors.*.info.wall_s), non-vacuous (compared leaves > 0, equal counts); Blender evaluations "
                                  "byte-identical; the two independent assessments agree on every decision, residual, footprint and quality "
                                  "output; LIVE negative control: one semantic solver field corrupted in a copy of build 2 MUST make the "
                                  "comparison fail",
    "V11_timing_and_realtime_render": "every key at Blender frame = index - f0 + 1; source time = index / 30; every real-time video decodes "
                                      "to exactly its declared frame count at 30.000 fps with one video frame per source frame",
    "V12_repaired_negative_controls": "every negative control returns its declared decision/class/reasons; NG_SHARED isolation fails (value and "
                                      "Blender); DG1, DG2, DG4, DG5 (precondition-verified REJECT) and DG3 (executed on real or synthetic "
                                      "multi-run) FAIL the truth check; DG5_SYN likewise; source-level controls detected; inexact rollback "
                                      "detected on every source",
    "V13_blinding": "mapping generated from the declared seed and sealed at render (hash recorded); ratings complete for every declared pair, "
                    "schema-valid, hashed in seal.json BEFORE unblind; unblind refuses without a valid seal",
    "V14_identity_labelling": "identity_review.json hashed in the declaration; every window labelled VERIFIED or UNKNOWN; VERIFIED only with "
                              "recorded evidence; parent states keep video_subject_identity UNKNOWN",
    "V15_synthetic_controls": "W120 synthetic H_G / H_C / DJ_B / NG_INFEASIBLE states copied unchanged (state ids equal W120), re-evaluated "
                              "bit-equal to W120's Blender evaluation, decisions equal W120's assessment",
    "C1_hard_pin_and_scope": "every ACCEPTed X_r and factor variant: pin-frame Blender residual <= 1 mm; changed channels only authorised; "
                             "changed frames within budget and planned domain; no outside-arm point; root unchanged",
    "C2_truthful_footprint": "every variant passes the display truth check (proposed and result); planned domain within budget and equal "
                             "across X_r and four variants; preview id = committed id",
    "C3_atomicity_lineage_reversal": "per source: ACCEPT moves head; atomic REJECT keeps head; stale parent rejected; exact revert (values "
                                     "byte-equal, Blender bit-equal)",
    "C4_source_immutability": "originals, frozen W102/W105/W116-W121 inputs, rig byte-unchanged across assess",
    "C5_user_edit_and_collateral": "every ACCEPTed variant keeps spine_01 and __object__ bit-equal to U",
    "C6_no_fallback": "no fallback; no G0_REFERENCE_MISSING accepted; no infeasible request ACCEPTed",
}
METRICS = ("Per request and variant (Blender): raw wrist / elbow step (m per frame) and speed (m/s) / acceleration (m/s^2) at 30 fps "
           "over [domain - 10, domain + 10], split into entry-transition, pin, release-transition and adjacent unpinned frames; "
           "source-relative compensation step |d(W_v - W_U)|; peak arm-channel angular step; elbow flex min, near-straight (< 5 deg) "
           "frames, bend-plane flips; frozen W119 B1/D2/D3/raw-boundary comparator via the frozen W120 factorial; source-dominated "
           "spike frames (U raw step >= 50 mm).  Strata from the UNEDITED parent before outcomes.")
REVIEW_PROCEDURE = (
    "Blind pairs = every ACCEPTed factorial-eligible G0F0/G1F0 pair of (a) the complete held-out population, (b) the W121 known "
    "positives S1_MID_RBneg and S2_LOW_{none,RBpos,RBneg}, (c) every ACCEPTed S3 anchor request.  Excluded from blind review "
    "(declared now): the remaining S1/S2 anchor requests.  Labels A/B from blind_mapping(seed); review order = mapping order.  "
    "Material per pair: the 30 fps real-time MP4 (source | U | A | B, world front, world side and character-relative side views, "
    "fixed cameras), the x0.25 slowed diagnostic MP4 (labelled), and review sheets DECODED FROM THE REAL-TIME MP4 (every frame in "
    "order) plus A/B wrist and elbow world trails.  The agent records for every pair overall A_VISIBLY_BETTER / B_VISIBLY_BETTER / "
    "TIE / UNJUDGEABLE and per dimension (raw_motion_continuity, elbow_plausibility, pin_behavior, temporal_transition, "
    "source_fidelity, user_intent) A_BETTER / B_BETTER / SAME / CANNOT_JUDGE with frame/time evidence, BEFORE seal and unblind.  "
    "Hard-pin failures (every REJECTed held-out / anchor evaluation request) receive an unblinded U vs F_r coverage render.  "
    "AGENT REVIEW only: the agent inspects the decoded frames of the real-time file in order; it cannot perceive playback as a "
    "human viewer does; this is not animator or user validation.")


def cmd_declare(a):
    out = new_dir(a.out_dir)
    pf, win, cal, ins = (load(a.preflight / "preflight.json"), load(a.windows / "windows.json"), load(a.calibrate / "calibration.json"),
                         load(a.instrument / "instrument.json"))
    idr = load(a.identity_review)
    if pf["stop"] or not ins["all_ok"]:
        raise SystemExit("STOP: preflight / instrument")
    if sha(a.calibrate / "frozen_controls.json") != cal["frozen_controls_sha256"]:
        raise SystemExit("STOP: frozen controls changed")
    windows_ok = all(set(idr["windows"].get(sid, {})) >= set(window_list(win["sources"][sid]["chosen"], win["sources"][sid]["anchors_w121"]))
                     for sid in SOURCES)
    if not windows_ok:
        raise SystemExit("STOP: identity review incomplete")
    sel = load(a.windows / "selection.json")
    pairs_heldout = [f"{sid}_{W}_{e}" for sid in SOURCES for W in ("NL", "NH") if sel[sid].get(W) for e in AB.EDITS]
    pairs_known = ["S1_MID_RBneg", "S2_LOW_none", "S2_LOW_RBpos", "S2_LOW_RBneg"]
    pairs_s3 = [f"S3_{W}_{e}" for W in ("LOW", "MID", "HIGH") for e in AB.EDITS]
    decl = {"schema": "w122_declaration_v1", "batch": BATCH, "instruction_blob": INSTRUCTION_BLOB, "starting_head": STARTING_HEAD,
            "branch": BRANCH, "head_at_declare": git("rev-parse", "HEAD"), "outcomes_inspected": False,
            "dirs": {k: str(getattr(a, k)) for k in ("preflight", "windows", "identity", "calibrate", "instrument")},
            "inputs_sha256": {"preflight": sha(a.preflight / "preflight.json"), "windows": sha(a.windows / "windows.json"),
                              "selection": sha(a.windows / "selection.json"), "identity_evidence": sha(a.identity / "identity_evidence.json"),
                              "identity_review": sha(a.identity_review), "calibration": sha(a.calibrate / "calibration.json"),
                              "instrument": sha(a.instrument / "instrument.json")},
            "identity_review_path": str(a.identity_review),
            "frozen_controls_sha256": cal["frozen_controls_sha256"], "selection": sel,
            "sources": {sid: {"title": t, "clip_sha256": pf["sources"][sid]["sha256"], "license": pf["sources"][sid]["license"],
                              "artist": pf["sources"][sid]["artist"], "url": pf["sources"][sid]["url"], "fps": 30.0,
                              "segment": win["sources"][sid]["segment"], "parent_state_id": pf["parents"][sid]["w121_state_id"],
                              "independence": "independent of W102/W105; NOT a new independent source relative to W121"}
                        for sid, (t, _) in SOURCES.items()},
            "population": {sid: {"requests": cal["sources"][sid]["requests"], "population": cal["sources"][sid]["population"]} for sid in SOURCES},
            "neg_source": cal["neg_source"], "neg_base": cal["sources"][cal["neg_source"]]["neg_base"] if cal["neg_source"] else None,
            "negative_controls": {k: list(v) for k, v in W1.NEG.items()}, "display_controls": {k: list(v) for k, v in DISPLAY_NEG.items()},
            "window_rule": AB.WINDOW_RULE, "stratum_threshold_m": AB.SOURCE_DOMINATED_STEP_M, "metrics": METRICS,
            "factors": __import__("w120_factor_attribution_base").FACTOR_DEF,
            "solver_params_digest": __import__("w120_factor_attribution_base").SOLVER_PARAMS_DIGEST,
            "comparator": "frozen W119 compare_y via frozen W120 factorial (unchanged)",
            "contrasts": __import__("w120_factor_attribution_base").CONTRASTS, "gates": GATES, "case_rule": AB.CASE_RULE,
            "nondeterministic_record_paths": list(AB.NONDETERMINISTIC_RECORD_PATHS),
            "blind": {"seed": AB.BLIND_SEED, "pairs_held_out": pairs_heldout, "pairs_known_w121_positive": pairs_known,
                      "pairs_s3_anchor": pairs_s3, "eligible_only": "a pair is rendered/rated only if both G0F0 and G1F0 are ACCEPTed and "
                      "factorial-eligible; ineligible pairs are listed with their rejection", "overall": list(AB.OVERALL),
                      "dimensions": list(AB.DIMENSIONS), "dimension_values": list(AB.DIM_VALUES)},
            "review_procedure": REVIEW_PROCEDURE, "synthetic": {"rids": list(SYN_RIDS), "source": str(W120R / "build_r1"),
                                                                "jitter_probe": "NOT RUN (optional; not declared)"},
            "render": {"fps": 30, "context_frames": AB.CONTEXT_FRAMES, "views": ["world_front(-Y)", "world_side(+X)", "char_side(left, fixed at capture)"],
                       "variants": ["U", "F", "X", *FACTORS], "no_filtering_interpolation_retiming": True},
            "historical_cases": HIST_CASES,
            "code_blobs": {f: git("hash-object", f) for f in CODE_FILES if (ROOT / f).exists()}}
    dump(out / "declaration.json", decl)
    print(json.dumps({"declaration_sha256": sha(out / "declaration.json"), "neg": decl["neg_base"], "pairs_held_out": pairs_heldout}))


def check_decl(a):
    if sha(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    d = load(a.declaration)
    if d["outcomes_inspected"] is not False:
        raise SystemExit("STOP: declaration not pre-outcome")
    return d


# ============================================================================================ build ===
def syn_subset():
    """The frozen W120 synthetic subset (records + states) copied unchanged."""
    b = load(W120R / "build_r1/build_report.json")
    st = load(W120R / "build_r1/syn_states.json")
    recs = {r: b["records"][r] for r in SYN_RIDS}
    names = set()
    for r in recs.values():
        names |= {r["parent"], r["U"], r["F"], r["X"]} | {x["state"] for x in r["factors"].values()}
    states = {n: st["states"][n] for n in sorted(names)}
    meta = {n: b["state_meta"]["syn"][n] for n in names}
    return recs, states, meta, st


def cmd_build(a):
    import w120_factor_attribution_base as FB  # noqa: F401
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    blobs = {f: git("hash-object", f) for f in CODE_FILES if (ROOT / f).exists()}
    t0 = time.time()
    fc = load(Path(decl["dirs"]["calibrate"]) / "frozen_controls.json")
    w121b = load(W121R / "build_r1/build_report.json")
    hw = held_out_windows(decl["selection"])
    rep = {"schema": "w122_build_v1", "tag": a.tag, "declaration_sha256": a.declaration_sha, "code_blobs": blobs, "sources": {}}
    for sid in SOURCES:
        src = W22Src(sid, hw)
        src.build_controls()
        mism = [r for r, x in frozen_controls_of(src).items() if scanon(x) != scanon(fc[sid].get(r))]
        for rid in src.req:
            src.make_factors(rid)
        if sid == decl["neg_source"]:
            src.negatives(decl["neg_base"])
        src.write_states(out / f"{sid}_states.json")
        r = dict(src.report(True), frozen_controls_mismatch=mism, states_sha256=sha(out / f"{sid}_states.json"))
        # anchor replication: every W121 request (same id) -> same U/F/X/factor state ids as W121's build
        w = w121b["sources"][sid]
        anc = {}
        for rid, rec in w["records"].items():
            if rid.startswith("NG_") or rid not in r["records"]:
                continue
            mine = r["records"][rid]
            names = [rec["U"], rec["F"], rec["X"]] + [x["state"] for x in rec["factors"].values()]
            mnames = [mine["U"], mine["F"], mine["X"]] + [x["state"] for x in mine["factors"].values()]
            anc[rid] = all(w["state_ids"][n] == r["state_ids"][m] for n, m in zip(names, mnames))
        r["anchor_state_ids_equal_w121"] = anc
        rep["sources"][sid] = r
    recs, states, meta, st = syn_subset()
    W1.C_().states_file(out / "syn_states.json", states, st.get("rig", "synthetic") if isinstance(st.get("rig"), str) else "synthetic")
    written = load(out / "syn_states.json")["states"]
    rep["synthetic"] = {"records": recs, "state_ids": {n: s["state_id"] for n, s in written.items()},
                        "state_ids_equal_w120": set(written) == set(states) and all(
                            AB.canon(written[n]) == AB.canon({k: st["states"][n][k] for k in written[n]}) for n in written),
                        "states_sha256": sha(out / "syn_states.json")}
    rep["wall_s"] = time.time() - t0   # telemetry (not part of any record)
    dump(out / "build_report.json", rep)
    print(json.dumps({s: {"n_states": len(r["state_ids"]), "mismatch": r["frozen_controls_mismatch"],
                          "anchor_bad": [k for k, v in r["anchor_state_ids_equal_w121"].items() if not v]} for s, r in rep["sources"].items()}
                     | {"syn_ok": rep["synthetic"]["state_ids_equal_w120"]}))


def cmd_evaluate(a):
    out = new_dir(a.out_dir)
    B = load(a.build_dir / "build_report.json")
    for sid in B["sources"]:
        t0 = time.time()
        blender(ROOT / "scripts/w116_blender_stages.py", "--", "evaluate", "--rig", W1.RIG_FBX, "--rest", W1.RIG_REST, "--states",
                a.build_dir / f"{sid}_states.json", "--out", out / f"{sid}_eval.json", log=out / f"{sid}_eval.log")
        print(sid, "evaluated", round(time.time() - t0, 1), "s")
    blender(ROOT / "scripts/w116_blender_stages.py", "--", "evaluate", "--rig", f"synthetic:{W120R / 'preflight/syn_spec.json'}",
            "--states", a.build_dir / "syn_states.json", "--out", out / "syn_eval.json", log=out / "syn_eval.log")
    print("syn evaluated")


# ============================================================================================ assess ===
def pts(fr, upper=UPPER, lower=LOWER):
    return (np.array(fr["world"][upper]["head"]), np.array(fr["world"][upper]["tail"]), np.array(fr["world"][lower]["tail"]))


def flex_deg(Sh, E, W):
    u, v = E - Sh, W - E
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v))))


def attribution(Vf, Uf, rec, rng, fps=30.0, upper=UPPER, lower=LOWER):
    """W122 raw-vs-relative attribution on Blender frames over rng = [lo, hi] (inclusive)."""
    import w117_temporal_pin_base as TB
    pin = set(rec["pin_frames"])
    plo, phi = min(pin), max(pin)
    planned = set(rec.get("_planned") or [])
    ks = [k for k in range(rng[0], rng[1] + 1) if k in Vf]
    Wv = {k: pts(Vf[k], upper, lower)[2] for k in ks}
    Ev = {k: pts(Vf[k], upper, lower)[1] for k in ks}
    Wu = {k: pts(Uf[k], upper, lower)[2] for k in ks}
    rw, re_ = AB.steps(Wv, ks), AB.steps(Ev, ks)
    ru = AB.steps(Wu, ks)
    comp = {k: float(np.linalg.norm((Wv[k] - Wu[k]) - (Wv[k - 1] - Wu[k - 1]))) for k in ks if k - 1 in Wv}
    kin, kinu = AB.kinematics(Wv, ks, fps), AB.kinematics(Wu, ks, fps)
    ang = {k: max(TB.qangle(Vf[k]["stored_wxyz"][c], Vf[k - 1]["stored_wxyz"][c]) for c in (upper, lower)) for k in ks if k - 1 in Vf}
    flex = {k: flex_deg(*pts(Vf[k], upper, lower)) for k in ks}
    normals = {}
    for k in ks:
        Sh, E, W = pts(Vf[k], upper, lower)
        n = np.cross(E - Sh, W - E)
        normals[k] = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-9 else None
    flips = [k for k in ks if k - 1 in normals and normals[k] is not None and normals[k - 1] is not None and float(np.dot(normals[k], normals[k - 1])) < 0]

    def zone(k):
        if k in pin and k != plo:
            return "pin"
        if k == plo:
            return "entry_jump"
        if k == phi + 1:
            return "release_jump"
        if k in planned and k < plo:
            return "entry_transition"
        if k in planned and k > phi:
            return "release_transition"
        return "adjacent_unpinned"

    zones = {}
    for k in rw:
        z = zone(k)
        d = zones.setdefault(z, {"frames": [], "raw_wrist_peak_m": 0.0, "raw_elbow_peak_m": 0.0, "comp_peak_m": 0.0, "U_raw_wrist_peak_m": 0.0,
                                 "raw_minus_U_peak_m": -1e9})
        d["frames"].append(k)
        d["raw_wrist_peak_m"] = max(d["raw_wrist_peak_m"], rw[k])
        d["raw_elbow_peak_m"] = max(d["raw_elbow_peak_m"], re_[k])
        d["comp_peak_m"] = max(d["comp_peak_m"], comp.get(k, 0.0))
        d["U_raw_wrist_peak_m"] = max(d["U_raw_wrist_peak_m"], ru[k])
        d["raw_minus_U_peak_m"] = max(d["raw_minus_U_peak_m"], rw[k] - ru[k])
    dom = [k for k in rw if k in planned or k in pin or k == phi + 1]
    return {"range": rng, "zones": zones,
            "domain_raw_wrist_peak_m": max([rw[k] for k in dom] + [0.0]), "domain_raw_elbow_peak_m": max([re_[k] for k in dom] + [0.0]),
            "domain_comp_peak_m": max([comp.get(k, 0.0) for k in dom] + [0.0]), "domain_U_raw_wrist_peak_m": max([ru[k] for k in dom] + [0.0]),
            "domain_peak_speed_m_s": max([kin["speed_m_s"][k] for k in dom if k in kin["speed_m_s"]] + [0.0]),
            "domain_peak_accel_m_s2": max([kin["accel_m_s2"][k] for k in dom if k in kin["accel_m_s2"]] + [0.0]),
            "domain_U_peak_accel_m_s2": max([kinu["accel_m_s2"][k] for k in dom if k in kinu["accel_m_s2"]] + [0.0]),
            "domain_peak_ang_step_deg": max([ang[k] for k in dom if k in ang] + [0.0]),
            "min_flex_deg": min(flex.values()), "near_straight_frames": [k for k, f in flex.items() if f < 5.0], "bend_flips": flips,
            "source_dominated_spike_frames": [k for k in rw if ru[k] >= AB.SOURCE_DOMINATED_STEP_M],
            "series": {"raw_wrist_step_m": rw, "raw_elbow_step_m": re_, "U_raw_wrist_step_m": ru, "comp_step_m": comp,
                       "speed_m_s": kin["speed_m_s"], "accel_m_s2": kin["accel_m_s2"], "U_accel_m_s2": kinu["accel_m_s2"], "ang_step_deg": ang,
                       "flex_deg": flex}}


def render_range(rec, seg):
    pin = rec["pin_frames"]
    return [max(seg[0], min(pin) - AB.M_MAX - AB.CONTEXT_FRAMES), min(seg[1], max(pin) + AB.M_MAX + AB.CONTEXT_FRAMES)]


def lineage_w22(out, sid, B, Rq, v0, states, ev):
    """W121 lineage programme on the W122 held-out population: R1 commit the first ACCEPTed G0F1 of the held-out order, R2 an atomic
    REJECT (NG_PIN_COMBINED on the neg source, else the held-out NREACH), R3 stale parent, R4 exact revert (Blender), R5 inexact
    rollback control."""
    S = S_()
    C = C_()
    store = S.Store(out / f"lineage_{sid}")
    recs = B["records"]
    order = [f"{sid}_{w}" for w in W22Src.NEG_BASE_ORDER] + [f"{sid}_{w}" for w in W1.NEG_BASE_ORDER]
    dec = lambda rid, f: Rq[rid]["variants"][f]["decision"]  # noqa: E731

    def cand(rid, f):
        name = recs[rid]["factors"][f]["state"]
        meta = B["state_meta"][name]
        st = S.make_state(v0["state_id"], meta["rig_id"], v0["frames"], v0["channels"], states[name]["values"], meta["op"], v0.get("source"))
        if st["state_id"] != states[name]["state_id"]:
            raise SystemExit("STOP: candidate does not rebuild to its id")
        return st
    store.put(v0)
    store.init_ref("main", v0["state_id"])
    store.init_ref("atomic", v0["state_id"])
    res = []
    pick = next((r for r in order if r in Rq and dec(r, "G0F1")["decision"] == "ACCEPT"), None)
    if pick:
        res.append(dict(step="R1", request=pick, variant="G0F1", **store.commit("main", v0["state_id"], cand(pick, "G0F1"), dec(pick, "G0F1"))))
    neg = "NG_PIN_COMBINED" if "NG_PIN_COMBINED" in recs and recs["NG_PIN_COMBINED"].get("source") == sid else f"{sid}_NREACH"
    nf = "G0F1" if neg == "NG_PIN_COMBINED" else "G1F0"
    h = store.refs["atomic"]
    r2 = store.commit("atomic", h, cand(neg, nf), dec(neg, nf))
    res.append(dict(step="R2", request=neg, variant=nf, head_unchanged=r2["head_after"] == h, **r2))
    other = next((r for r in order if r in Rq and r != pick), None)
    if other:
        res.append(dict(step="R3", request=other, variant="G1F0", **store.commit("main", v0["state_id"], cand(other, "G1F0"), dec(other, "G1F0"))))
    rev = {}
    if store.refs["main"] != v0["state_id"]:
        rv, rrec = store.revert("main")
        res.append(dict(step="R4", **rrec))
        C.states_file(out / f"{sid}_revert_states.json", {"revert": rv}, "w105")
        blender(ROOT / "scripts/w116_blender_stages.py", "--", "evaluate", "--rig", W1.RIG_FBX, "--rest", W1.RIG_REST, "--states",
                out / f"{sid}_revert_states.json", "--out", out / f"{sid}_revert_eval.json", log=out / f"{sid}_revert_eval.log")
        e = load(out / f"{sid}_revert_eval.json")["states"]["revert"]
        rev = {"value_digest_equal_v0": e["value_digest"] == ev["states"]["v0"]["value_digest"],
               "frames_bit_equal_v0": e["frames"] == ev["states"]["v0"]["frames"]}
        bad = json.loads(json.dumps(rv["values"]))
        bad[UPPER][len(bad[UPPER]) // 2][0] += 1e-6
        rev["inexact_rollback_detected"] = S.canon(bad) != S.canon(v0["values"])
    return {"programme": res, "refs": store.refs, "lineage_audit": {r: store.lineage(hh) for r, hh in store.refs.items()}, "revert_blender": rev}


def first_multirun(Rq, order):
    for rid in order:
        if rid not in Rq:
            continue
        for f in FACTORS:
            y = Rq[rid]["variants"].get(f)
            if y and len(y["display_spec_proposed"].get("changed_runs", [])) >= 2:
                return rid, f
    return None, None


def semantic_outcomes(A):
    """Every decision, residual, footprint, display and quality output of an assessment (no telemetry) for V10."""
    out = {}
    for rid, r in A["requests"].items():
        for v, x in r["variants"].items():
            out[f"{rid}/{v}"] = {k: x.get(k) for k in ("decision", "residual_m", "footprint", "display_check_proposed", "display_check_result",
                                                       "quality", "class", "fairness", "commit", "frame_pass", "isolation_blender")}
        out[f"{rid}/comparisons"] = r["comparisons"]
        out[f"{rid}/attribution"] = r.get("attribution")
    for rid, x in (A.get("synthetic") or {}).get("requests", {}).items():
        out[f"SYN/{rid}"] = {"decisions": x["decisions"], "quality": {v: y.get("quality") for v, y in x["assess"]["variants"].items()}}
    return out


def cmd_assess(a):
    import w118_transition_edit_base as XB
    import w119_endpoint_transition as W9
    import w120_factor_attribution as W20
    import w120_factor_attribution_base as FB  # noqa: F401
    S = S_()
    C = C_()
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    acq = acq_cands()
    guard_files = {f"clip_{sid}": acq[t]["download"]["path"] for sid, (t, _) in SOURCES.items()}
    guard_files.update(hist_clip=W1.HIST_CLIP, hist_obs=W1.HIST_OBS, w105_cases=W1.W105_VIDEO_CASES, rig_fbx=W1.RIG_FBX, rig_rest=W1.RIG_REST,
                       w116_v0=W1.W116_V0, w120_decision=W120R / "decision/decision.json", w121_decision=W121R / "decision/decision.json",
                       w121_manifest=W121 / "evidence_manifest.json", w121_declaration=W121R / "declare/declaration.json",
                       w120_syn_states=W120R / "build_r1/syn_states.json", w120_syn_eval=W120R / "eval_e1/syn_eval.json")
    for sid in SOURCES:
        guard_files[f"w121_parent_{sid}"] = W121R / "prepare" / sid / "parent_state.json"
        guard_files[f"w121_obs_{sid}"] = W121R / "perceive/candidates" / W1.safe(SOURCES[sid][0]) / "m1s0/observations_predictions.npz"
    g = S.SourceGuard(guard_files)
    Bd = a.build_dir
    Bb = load(Bd / "build_report.json")
    store = S.Store(out / "store")
    rep = {"schema": "w122_assess_v1", "declaration_sha256": a.declaration_sha, "label": "AGENT-RUN Blender evaluation", "requests": {},
           "sources": {}}
    t0 = time.time()
    held_order = [f"{sid}_{W}_{e}" for sid in SOURCES for W in ("NL", "NH") for e in AB.EDITS]
    for sid, B in Bb["sources"].items():
        ev = load(a.eval_dir / f"{sid}_eval.json")
        states = load(Bd / f"{sid}_states.json")["states"]
        v0 = load(W121R / "prepare" / sid / "parent_state.json")
        if v0["state_id"] != states["v0"]["state_id"]:
            raise SystemExit("STOP: parent differs from the build")
        pr = load(W121R / "prepare" / sid / "probe.json")
        rig = C.rig_from(W1.RIG_REST, pr["rig"])
        bound = load(W121R / "prepare" / sid / "bindings.json")["bound"]
        frames = v0["frames"]
        seg = [frames[0], frames[-1]]
        Bshim = {"state_meta": {"real": B["state_meta"]}}
        full_state = lambda side, name, _v0=v0: _v0  # noqa: E731
        st_shim = {"real": states}
        for rid, rec in B["records"].items():
            A = W20.assess_record(rid, rec, ev, rig, bound, ev["rest"], ev["rig"]["scale"][0], frames, store, full_state, st_shim, Bshim, None)
            A.update(role=rec["role"], side="real", kind=rec["kind"], pin_frames=rec["pin_frames"], pin_runs=rec["pin_runs"],
                     validation_errors=rec["validation_errors"], parent=rec["parent"], source=sid, window=rec.get("window"),
                     edit_key=rec.get("edit_key"), negative_of=rec.get("negative_of"), neg_factor=rec.get("neg_factor"), expect=rec.get("expect"),
                     expect_reasons=rec.get("expect_reasons"), margins=rec.get("margins"),
                     population="NEGATIVE" if rec["role"] == "negative" else rec.get("population"))
            if rec["role"] in ("evaluation", "control") and not rec["validation_errors"]:
                Uf = C.ev_frames(ev, rec["U"])
                rr = render_range(rec, seg)
                att = {}
                for v in ["U", "F", "X", *FACTORS]:
                    name = rec["U"] if v == "U" else rec["F"] if v == "F" else rec["X"] if v == "X" else rec["factors"][v]["state"]
                    planned = rec["x_info"].get("planned_frames", []) if v in ("X", "U", "F") else rec["factors"][v]["info"].get("planned_frames", [])
                    att[v] = attribution(C.ev_frames(ev, name), Uf, dict(rec, _planned=planned), rr)
                A["attribution"] = att
            rep["requests"][rid] = A
        # display controls on the neg source
        if sid == decl["neg_source"]:
            dcs = {}
            base = decl["neg_base"]
            for dc, (which, f, kind, desc) in DISPLAY_NEG.items():
                if dc == "DG5_SYN":
                    continue
                if which == "BASE":
                    rid = base
                elif which == "NREACH":
                    rid = f"{sid}_NREACH"
                else:
                    rid, f = first_multirun(rep["requests"], held_order + [r for r in B["records"] if r not in held_order])
                    if rid is None:
                        dcs[dc] = {"executed": False, "reason": "no real variant with >= 2 changed runs (synthetic DJ_B used)"}
                        continue
                y = rep["requests"][rid]["variants"][f]
                rec = B["records"][rid]
                name = rec["factors"][f]["state"]
                try:
                    if kind == "DY5":
                        s = AB.dg5_inject(y["display_spec_result"], y["commit"]["decision"], W9.corrupt_display)
                    else:
                        s = W9.corrupt_display(kind, y["display_spec_proposed"])
                except AB.PreconditionError as e:
                    dcs[dc] = {"executed": False, "refused": str(e), "source": rid, "factor": f}
                    continue
                if s is None:
                    dcs[dc] = {"executed": False, "reason": "corruption not applicable", "source": rid, "factor": f}
                    continue
                chk = XB.check_display(s, C.ev_frames(ev, name), C.ev_frames(ev, rec["U"]), frames, rec["pin_frames"], ev["states"][name]["state_id"],
                                       decision=y["commit"]["decision"] if kind == "DY5" else None)
                dcs[dc] = {"executed": True, "source": rid, "factor": f, "desc": desc, "check": chk,
                           "precondition_decision": y["commit"]["decision"] if kind == "DY5" else None,
                           "status_before": y["display_spec_result"]["status"] if kind == "DY5" else None,
                           "status_after": s.get("status") if kind == "DY5" else None}
            rep["display_controls"] = dcs
        rep["sources"][sid] = {"lineage": lineage_w22(out, sid, B, rep["requests"], v0, states, ev),
                               "missing_states": sorted(set(B["state_ids"]) - set(ev["states"])),
                               "timing": W1.timing_semantics(ev, v0, 30.0)}
        del ev
    # ---- synthetic W120 subset
    sev = load(a.eval_dir / "syn_eval.json")
    w120ev = load(W120R / "eval_e1/syn_eval.json")
    sb = Bb["synthetic"]
    sst = load(Bd / "syn_states.json")["states"]
    w120b = load(W120R / "build_r1/build_report.json")
    import w117_temporal_pin as W7
    probe = load(W120R / "preflight/syn_probe.json")
    srig = W7.syn_rig(probe)
    W116 = Path("C:/Users/dna10/animcv-w116")
    W117 = Path("C:/Users/dna10/animcv-w117")
    syn_b = load(W117 / "results/calibrate/calibration.json")["syn_bindings"]["bound"]
    w116_syn = load(W116 / "results/calibrate/calibration.json")["bindings"]["synthetic"]["bound"]
    sfull = {}

    def syn_full(side, name):
        if name not in sfull:
            sfull[name] = load(W120R / f"preflight/syn_parent_{name}.json")
        return sfull[name]
    w120a = load(W120R / "assess/assess_report.json")["requests"]
    syn = {"eval_bit_equal_w120": {n: sev["states"][n]["frames"] == w120ev["states"][n]["frames"]
                                   and sev["states"][n]["value_digest"] == w120ev["states"][n]["value_digest"] for n in sev["states"]},
           "requests": {}}
    sstore = S.Store(out / "syn_store")
    import w120_factor_attribution as W20b
    for rid, rec in sb["records"].items():
        A = W20b.assess_record(rid, rec, sev, srig, dict(syn_b) | w116_syn, sev["rest"], 1.0, W20b.SYN_FRAMES, sstore, syn_full, {"syn": sst},
                               {"state_meta": w120b["state_meta"]}, None)
        dec = {v: x.get("decision") for v, x in A["variants"].items() if "decision" in x}
        w = {v: x.get("decision") for v, x in w120a[rid]["variants"].items() if "decision" in x}
        syn["requests"][rid] = {"decisions": dec, "decisions_equal_w120": dec == w, "assess": A}
    sdc = {}
    y = syn["requests"]["DJ_B"]["assess"]["variants"]["G1F0"]
    rec = sb["records"]["DJ_B"]
    s = W9.corrupt_display("DY3", y["display_spec_proposed"])
    sdc["DG3_SYN"] = {"executed": s is not None, "check": XB.check_display(s, C.ev_frames(sev, rec["factors"]["G1F0"]["state"]), C.ev_frames(sev, rec["U"]),
                                                                           W20b.SYN_FRAMES, rec["pin_frames"], sev["states"][rec["factors"]["G1F0"]["state"]]["state_id"]) if s else None}
    y = syn["requests"]["NG_INFEASIBLE"]["assess"]["variants"]["G1F0"]
    rec = sb["records"]["NG_INFEASIBLE"]
    try:
        s = AB.dg5_inject(y["display_spec_result"], y["commit"]["decision"], W9.corrupt_display)
        sdc["DG5_SYN"] = {"executed": True, "precondition_decision": y["commit"]["decision"], "status_before": y["display_spec_result"]["status"],
                          "status_after": s["status"],
                          "check": XB.check_display(s, C.ev_frames(sev, rec["factors"]["G1F0"]["state"]), C.ev_frames(sev, rec["U"]), W20b.SYN_FRAMES,
                                                    rec["pin_frames"], sev["states"][rec["factors"]["G1F0"]["state"]]["state_id"], decision=y["commit"]["decision"])}
    except AB.PreconditionError as e:
        sdc["DG5_SYN"] = {"executed": False, "refused": str(e)}
    syn["display_controls"] = sdc
    rep["synthetic"] = syn
    rep["assess_wall_s"] = time.time() - t0      # telemetry
    W1.BASE = BASE                                # W121 source-level controls, scratch files under animcv-w122/tmp
    dsrc = {"sources": {sid: {"title": SOURCES[sid][0], "clip_sha256": decl["sources"][sid]["clip_sha256"], "fps_source": 30.0}
                        for sid in SOURCES}, "dirs": {"prepare": str(W121R / "prepare")}}
    rep["source_controls"] = W1.source_controls(dsrc, acq_full(), load(W121R / "perceive/perceive.json"), None)
    rep["source_guard"] = g.check()
    dump(out / "assess_report.json", rep)
    dump(out / "semantic_outcomes.json", semantic_outcomes(rep))
    print(json.dumps({"n": len(rep["requests"]), "source": rep["source_guard"]["unchanged"], "wall": rep["assess_wall_s"]}))


def acq_full():
    return load(W121R / "acquire/acquire.json")


# ============================================================================================ render ===
VARIANTS_SHOWN = ["U", "F", "X", *FACTORS]
VIEWS = ("world_front", "world_side", "char_side")


def view_offsets(ev0, cap):
    fr = next(x for x in ev0["frames"] if x["frame_index"] == cap)
    l_, r_ = np.array(fr["world"]["upperarm_l"]["head"]), np.array(fr["world"]["upperarm_r"]["head"])
    lat = l_ - r_
    lat[2] = 0.0
    lat = lat / (np.linalg.norm(lat) or 1.0)
    return {"world_front": [0.0, -3.0, 0.0], "world_side": [3.0, 0.0, 0.0], "char_side": (lat * 3.0).tolist()}


def pair_lists(decl, A):
    b = decl["blind"]
    want = b["pairs_held_out"] + b["pairs_known_w121_positive"] + b["pairs_s3_anchor"]
    elig, inel = [], {}
    for rid in want:
        r = A["requests"].get(rid)
        if r is None:
            inel[rid] = "NOT_DECLARED"
            continue
        if r.get("eligible_factorial") and r["variants"]["G0F0"]["decision"]["decision"] == "ACCEPT" and r["variants"]["G1F0"]["decision"]["decision"] == "ACCEPT":
            elig.append(rid)
        else:
            inel[rid] = {v: r["variants"][v]["decision"] for v in ("G0F0", "G1F0")}
    return elig, inel


def project(p, centre, off, ortho, size):
    """Orthographic projection matching review_framepose_ik_gt_attribution_blender._aim(cam, centre+off, centre, Z up)."""
    d = -np.asarray(off, float)
    d /= np.linalg.norm(d)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(d, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, d)
    q = np.asarray(p, float) - np.asarray(centre, float)
    return (0.5 + float(q @ right) / ortho) * size, (0.5 - float(q @ upv) / ortho) * size


def cmd_render(a):
    import cv2
    from PIL import Image, ImageDraw
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    A = load(a.assess_dir / "assess_report.json")
    Bb = load(a.build_dir / "build_report.json")
    acq = acq_full()["candidates"]
    elig, inel = pair_lists(decl, A)
    mapping, order = AB.blind_mapping(elig, decl["blind"]["seed"])
    pid = {rid: f"P{i + 1:02d}" for i, rid in enumerate(order)}
    sealed = new_dir(out / "sealed")
    dump(sealed / "mapping.json", {"seed": decl["blind"]["seed"], "mapping": mapping, "pair_ids": pid})
    # hard-pin failures = REJECTed evaluation requests (held-out + anchor): U vs F_r coverage
    fails = sorted(r for r, x in A["requests"].items() if x["role"] == "evaluation" and x["variants"]["X"]["decision"]["decision"] == "REJECT")
    ORTHO, SIZE = 1.4, 320
    jobs_by_src = {}
    plan = {}
    for rid in elig + fails:
        r = A["requests"][rid]
        sid = r["source"]
        rec = Bb["sources"][sid]["records"][rid]
        ev0 = load(W121R / "prepare" / sid / "probe.json")["states"]["v0"]
        seg = decl["sources"][sid]["segment"]
        rr = render_range(rec, seg)
        tgt = rec["request"]["constraints"][0]["target"]
        offs = view_offsets(ev0, rec["pin_frames"][0])
        vs = VARIANTS_SHOWN if rid in elig else ["U", "F"]
        plan[rid] = {"source": sid, "range": rr, "centre": tgt, "views": offs, "variants": vs}
        for v in vs:
            st = rec["U"] if v == "U" else rec["F"] if v == "F" else rec["X"] if v == "X" else rec["factors"][v]["state"]
            for k in range(rr[0], rr[1] + 1):
                mk = [{"name": "target", "position": tgt, "color": (0.1, 0.85, 0.2, 1), "radius": 0.02}] if k in set(rec["pin_frames"]) else []
                for vn, off in offs.items():
                    jobs_by_src.setdefault(sid, []).append({"id": f"{rid}__{v}__{vn}__f{k:04d}", "state": st, "frame_index": k, "markers": mk,
                                                            "centre": tgt, "view_offset": off})
    t0 = time.time()
    for sid, jobs in jobs_by_src.items():
        dump(out / f"{sid}_jobs.json", {"jobs": jobs})
        blender(ROOT / "scripts/w116_blender_stages.py", "--", "render", "--rig", W1.RIG_FBX, "--rest", W1.RIG_REST, "--states",
                a.build_dir / f"{sid}_states.json", "--jobs", out / f"{sid}_jobs.json", "--out-dir", out / f"frames_{sid}", "--ortho-scale",
                str(ORTHO), "--size", str(SIZE), log=out / f"{sid}_render.log")
    render_s = time.time() - t0
    vids = new_dir(out / "videos")
    sheets = new_dir(out / "review_sheets")
    vinfo = {}
    ev_cache = {}

    def tile(path):
        im = cv2.imread(str(path))
        return im

    for rid in elig + fails:
        p = plan[rid]
        sid = p["source"]
        t = SOURCES[sid][0]
        ks = list(range(p["range"][0], p["range"][1] + 1))
        srcf = W1.source_frames(acq[t]["download"]["path"], ks)
        z = np.load(W121R / "perceive/candidates" / W1.safe(t) / "m1s0/observations_predictions.npz")
        bx = []
        for k in ks:
            q = z["input_2d"][k][z["input_valid"][k]][:, :2]
            bx.append(q)
        allp = np.concatenate(bx) if bx else None
        H_, W_ = srcf[ks[0]].shape[:2]
        if allp is not None and len(allp):
            cx, cy = float(np.median(allp[:, 0])) * W_, float(np.median(allp[:, 1])) * H_
            ext = max(float(np.ptp(allp[:, 0])) * W_, float(np.ptp(allp[:, 1])) * H_) * 1.15 + 40
        else:
            cx, cy, ext = W_ / 2, H_ / 2, max(W_, H_)
        x0, y0 = int(max(0, cx - ext / 2)), int(max(0, cy - ext / 2))
        x1, y1 = int(min(W_, cx + ext / 2)), int(min(H_, cy + ext / 2))
        pin = set(A["requests"][rid]["pin_frames"])
        rec = Bb["sources"][sid]["records"][rid]
        planned = set(rec["x_info"].get("planned_frames", []))

        def src_img(k):
            im = srcf[k][y0:y1, x0:x1]
            s = SIZE / max(im.shape[:2])
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
            pad = np.full((SIZE, SIZE, 3), 30, np.uint8)
            pad[(SIZE - im.shape[0]) // 2:(SIZE - im.shape[0]) // 2 + im.shape[0], (SIZE - im.shape[1]) // 2:(SIZE - im.shape[1]) // 2 + im.shape[1]] = im
            return pad

        def rimg(v, vn, k):
            return tile(out / f"frames_{sid}" / f"{rid}__{v}__{vn}__f{k:04d}.png")

        def write(path, frames_bgr, repeat=1):
            h, w = frames_bgr[0].shape[:2]
            wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
            for f in frames_bgr:
                for _ in range(repeat):
                    wr.write(f)
            wr.release()
            cap = cv2.VideoCapture(str(path))
            n = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return {"path": str(path.relative_to(out)), "frames_written": len(frames_bgr) * repeat, "frames_decoded": n, "fps": fps,
                    "sha256": sha(path), "seconds": n / fps if fps else None}

        def label(img, txt, col=(20, 20, 20), y=18):
            cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            return img

        hdr = 28
        if rid in elig:
            m = mapping[rid]
            cols = ["SRC", "U", "A", "B"]
            vmap = {"U": "U", "A": m["A"], "B": m["B"]}
            blind, ann = [], []
            for k in ks:
                canvas = np.full((hdr + 3 * SIZE, 4 * SIZE, 3), 250, np.uint8)
                tag = "PIN" if k in pin else ("AUTO-DOMAIN" if k in planned else "")
                label(canvas, f"{pid[rid]}  f{k}  t={k / 30.0:.3f}s  {tag}   [AGENT REVIEW blind pair; 30 fps real time]", (20, 20, 20) if not tag else (30, 60, 200))
                for j, vn in enumerate(VIEWS):
                    canvas[hdr + j * SIZE:hdr + (j + 1) * SIZE, 0:SIZE] = src_img(k) if j == 0 else np.full((SIZE, SIZE, 3), 250, np.uint8)
                    for i, c in enumerate(cols[1:]):
                        im = rimg(vmap[c], vn, k)
                        canvas[hdr + j * SIZE:hdr + (j + 1) * SIZE, (i + 1) * SIZE:(i + 2) * SIZE] = im
                        label(canvas[hdr + j * SIZE:hdr + (j + 1) * SIZE, (i + 1) * SIZE:(i + 2) * SIZE], f"{c} {vn}", (60, 60, 60), 14)
                    if j > 0:
                        label(canvas[hdr + j * SIZE:hdr + (j + 1) * SIZE, 0:SIZE], f"view: {vn}", (60, 60, 60), 14)
                blind.append(canvas)
                # annotated (unblinded) full variant set, world_front + char_side
                c2 = np.full((hdr + 2 * (SIZE // 2) + 30, 8 * (SIZE // 2), 3), 250, np.uint8)
                label(c2, f"{rid} f{k} t={k / 30.0:.3f}s {tag}  UNBLINDED ANNOTATED (review only after unblind)")
                for j, vn in enumerate(("world_front", "char_side")):
                    s2 = cv2.resize(src_img(k), (SIZE // 2, SIZE // 2))
                    c2[hdr + j * (SIZE // 2):hdr + (j + 1) * (SIZE // 2), 0:SIZE // 2] = s2
                    for i, v in enumerate(VARIANTS_SHOWN):
                        im = cv2.resize(rimg(v, vn, k), (SIZE // 2, SIZE // 2))
                        c2[hdr + j * (SIZE // 2):hdr + (j + 1) * (SIZE // 2), (i + 1) * (SIZE // 2):(i + 2) * (SIZE // 2)] = im
                        label(c2[hdr + j * (SIZE // 2):hdr + (j + 1) * (SIZE // 2), (i + 1) * (SIZE // 2):(i + 2) * (SIZE // 2)], v, (60, 60, 60), 12)
                # timeline bar
                yb = hdr + 2 * (SIZE // 2) + 4
                for i, kk in enumerate(ks):
                    x_a = int(i * c2.shape[1] / len(ks))
                    x_b = int((i + 1) * c2.shape[1] / len(ks)) - 1
                    col = (40, 160, 40) if kk in pin else (40, 120, 230) if kk in planned else (200, 200, 200)
                    cv2.rectangle(c2, (x_a, yb), (x_b, yb + 18), col, -1)
                    if kk == k:
                        cv2.rectangle(c2, (x_a, yb), (x_b, yb + 18), (0, 0, 0), 2)
                ann.append(c2)
            vinfo[rid] = {"pair_id": pid[rid], "frames": ks, "blind_realtime": write(vids / f"{pid[rid]}_blind_realtime_30fps.mp4", blind),
                          "blind_slowed_x0.25": write(vids / f"{pid[rid]}_blind_SLOWED_x0.25_diagnostic.mp4", blind, repeat=4),
                          "annotated_realtime": write(vids / f"{rid}_annotated_UNBLINDED_30fps.mp4", ann)}
            # review sheets DECODED from the real-time blind MP4 (every frame, in order)
            cap = cv2.VideoCapture(str(vids / f"{pid[rid]}_blind_realtime_30fps.mp4"))
            dec = []
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                dec.append(im)
            cap.release()
            # one sheet per view pair: cells = [A | B] crops around the arm for each frame, 9 per row
            cell = 150
            for vn_i, vn in enumerate(VIEWS):
                ncol = 9
                rows = math.ceil(len(dec) / ncol)
                sh = np.full((30 + rows * (cell + 16), ncol * (2 * cell + 8), 3), 252, np.uint8)
                label(sh, f"{pid[rid]} view {vn}: every decoded frame of the 30 fps blind MP4 in order; each cell = A (left) | B (right); green sphere = pin target")
                for i, im in enumerate(dec):
                    r_, c_ = divmod(i, ncol)
                    y = 30 + r_ * (cell + 16)
                    xx = c_ * (2 * cell + 8)
                    for jj, col in enumerate((2, 3)):
                        sub = im[hdr + vn_i * SIZE:hdr + (vn_i + 1) * SIZE, col * SIZE:(col + 1) * SIZE]
                        sub = sub[SIZE // 8:SIZE * 7 // 8, SIZE // 8:SIZE * 7 // 8]
                        sh[y:y + cell, xx + jj * cell:xx + (jj + 1) * cell] = cv2.resize(sub, (cell, cell))
                    k = ks[i]
                    cv2.putText(sh, f"f{k}{' PIN' if k in pin else ''}", (xx + 4, y + cell + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (30, 60, 200) if k in pin else (50, 50, 50), 1, cv2.LINE_AA)
                cv2.imwrite(str(sheets / f"{pid[rid]}_{vn}.jpg"), sh, [cv2.IMWRITE_JPEG_QUALITY, 90])
            # trails (A / B / U) from the Blender evaluation
            if sid not in ev_cache:
                ev_cache.clear()
                ev_cache[sid] = load(a.eval_dir / f"{sid}_eval.json")
            ev = ev_cache[sid]
            C = C_()
            fig = Image.new("RGB", (3 * 420, 470), (252, 252, 251))
            d = ImageDraw.Draw(fig)
            d.text((6, 4), f"{pid[rid]} wrist (dots) and elbow (crosses) world trails, one mark per 30 fps frame; U grey, A red, B blue; "
                           f"pin frames ringed", fill=(20, 20, 20))
            for j, vn in enumerate(VIEWS):
                off = p["views"][vn]
                ox = j * 420
                d.rectangle([ox + 5, 20, ox + 415, 465], outline=(200, 200, 200))
                d.text((ox + 10, 24), vn, fill=(60, 60, 60))
                for lab, col in (("U", (150, 150, 150)), ("A", (210, 40, 40)), ("B", (40, 90, 220))):
                    v = vmap[lab]
                    name = rec["U"] if v == "U" else rec["factors"][v]["state"]
                    F_ = C.ev_frames(ev, name)
                    pw, pe = [], []
                    for k in ks:
                        Sh, E, Wp = pts(F_[k])
                        pw.append(project(Wp, p["centre"], off, 0.9, 400))
                        pe.append(project(E, p["centre"], off, 0.9, 400))
                    pw = [(ox + 10 + x, 50 + y) for x, y in pw]
                    pe = [(ox + 10 + x, 50 + y) for x, y in pe]
                    d.line(pw, fill=col, width=1)
                    d.line(pe, fill=col, width=1)
                    for k, (x, y) in zip(ks, pw):
                        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=col)
                        if k in pin:
                            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=col)
                    for x, y in pe:
                        d.line([x - 3, y, x + 3, y], fill=col)
                        d.line([x, y - 3, x, y + 3], fill=col)
            fig.save(sheets / f"{pid[rid]}_trails.png")
        else:
            cov = []
            for k in ks:
                canvas = np.full((hdr + SIZE, 3 * SIZE, 3), 250, np.uint8)
                st_k = A["requests"][rid]["variants"]["X"]["decision"]
                label(canvas, f"{rid} f{k} {'PIN' if k in pin else ''} REJECTED {st_k['failure_class']} (hard-pin failure coverage; U | F_r)")
                canvas[hdr:, 0:SIZE] = src_img(k)
                canvas[hdr:, SIZE:2 * SIZE] = rimg("U", "world_front", k)
                canvas[hdr:, 2 * SIZE:] = rimg("F", "world_front", k)
                cov.append(canvas)
            vinfo[rid] = {"coverage_realtime": write(vids / f"{rid}_REJECTED_coverage_30fps.mp4", cov), "frames": ks}
            mid = cov[len(cov) // 2]
            cv2.imwrite(str(sheets / f"FAIL_{rid}.jpg"), np.concatenate([cov[i] for i in (0, len(cov) // 3, len(cov) // 2)], 0)
                        if len(cov) > 3 else mid)
    rr = {"schema": "w122_render_v1", "declaration_sha256": a.declaration_sha, "eligible_pairs": elig, "ineligible_pairs": inel,
          "pair_ids_review_order": [pid[r] for r in order], "hard_pin_failure_coverage": fails, "videos": vinfo, "plan": plan,
          "sealed_mapping_sha256": sha(sealed / "mapping.json"), "render_wall_s": render_s, "ortho_scale": ORTHO, "size": SIZE,
          "blender_render_logs": {sid: str(out / f"{sid}_render.log") for sid in jobs_by_src}}
    dump(out / "render_report.json", rr)
    print(json.dumps({"pairs": [pid[r] for r in order], "ineligible": list(inel), "fails": fails, "render_s": render_s}))


# ============================================================================================ seal / unblind ===
def cmd_seal(a):
    out = new_dir(a.out_dir)
    rr = load(a.render_dir / "render_report.json")
    R = load(a.ratings)
    pids = rr["pair_ids_review_order"]
    errs = {p: AB.validate_rating(R["ratings"].get(p, {})) for p in pids}
    seal = {"schema": "w122_seal_v1", "sealed_utc": now(), "ratings_sha256": sha(a.ratings), "ratings_path": str(a.ratings),
            "mapping_sha256_at_render": rr["sealed_mapping_sha256"], "mapping_file_read": False, "pairs": pids,
            "complete": all(p in R["ratings"] for p in pids), "schema_errors": {p: e for p, e in errs.items() if e}}
    seal["ok"] = seal["complete"] and not seal["schema_errors"]
    dump(out / "seal.json", seal)
    print(json.dumps(seal, indent=1))
    return 0 if seal["ok"] else 1


def cmd_unblind(a):
    out = new_dir(a.out_dir)
    rr = load(a.render_dir / "render_report.json")
    seal = load(a.seal_dir / "seal.json")
    if not seal["ok"] or sha(a.ratings) != seal["ratings_sha256"]:
        raise SystemExit("STOP: no valid seal (ratings changed or incomplete)")
    mp = a.render_dir / "sealed/mapping.json"
    if sha(mp) != rr["sealed_mapping_sha256"]:
        raise SystemExit("STOP: mapping changed after render")
    M = load(mp)
    R = load(a.ratings)
    inv = {v: k for k, v in M["pair_ids"].items()}
    res = {}
    for p in rr["pair_ids_review_order"]:
        rid = inv[p]
        res[rid] = {"pair_id": p, "mapping": M["mapping"][rid], "blind": R["ratings"][p], "unblinded": AB.unblind(R["ratings"][p], M["mapping"][rid])}
    dump(out / "unblinded.json", {"schema": "w122_unblind_v1", "seal_sha256": sha(a.seal_dir / "seal.json"), "mapping_sha256": sha(mp),
                                  "seed": M["seed"], "results": res})
    print(json.dumps({r: (x["pair_id"], x["unblinded"]["overall"]) for r, x in res.items()}, indent=1))


# ============================================================================================ decide ===
def cmd_decide(a):
    import w120_factor_attribution_base as FB
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    pf = load(Path(decl["dirs"]["preflight"]) / "preflight.json")
    win = load(Path(decl["dirs"]["windows"]) / "windows.json")
    cal = load(Path(decl["dirs"]["calibrate"]) / "calibration.json")
    ins = load(Path(decl["dirs"]["instrument"]) / "instrument.json")
    idr = load(decl["identity_review_path"])
    A, A2 = load(a.assess_dir / "assess_report.json"), load(a.assess_dir2 / "assess_report.json")
    B1, B2 = load(a.build_dir / "build_report.json"), load(a.build_dir2 / "build_report.json")
    rr = load(a.render_dir / "render_report.json")
    seal = load(a.seal_dir / "seal.json")
    UB = load(a.unblind_dir / "unblinded.json")
    Rq = A["requests"]
    srcs = list(SOURCES)
    g, det = {}, {}
    # V1 / V2 / V3
    det["V1"] = {"git": pf["git"]["ok"], "worklogs": all(v["unchanged_since_start"] and v["complete_marker"] for v in pf["worklogs"].values()),
                 "w121_worklog_blob": pf["w121_worklog_blob"] == W121_WORKLOG_BLOB,
                 "w121_decision": pf["w121_decision"]["sha256"].startswith(W121_DECISION_PREFIX) and pf["w121_decision"]["case"] == "D",
                 "w120_decision": pf["w120_decision"]["sha256"].startswith(W120_DECISION_PREFIX) and pf["w120_decision"]["case"] == "B",
                 "w121_manifest": pf["w121_manifest"]["n_mismatched"] == 0 and pf["w121_manifest"]["sha256"] == W121_MANIFEST_SHA,
                 **{sid: pf["sources"][sid]["ok"] for sid in srcs},
                 "w121_decision_after": A["source_guard"]["unchanged"]}
    hist_now = {n: subprocess.run(["git", "diff", "--quiet", STARTING_HEAD, "--", f], cwd=ROOT).returncode == 0 for n, f in zip(HIST_CASES, HIST_WORKLOGS)}
    det["V1"]["worklogs_unchanged_at_decide"] = all(hist_now.values())
    g["V1_source_authentication"] = all(det["V1"].values())
    rx = pf["perception_reexecution"]
    det["V2"] = {"frozen_identity": pf["frozen_identity"]["ok"], **{k: rx[k]["ok"] for k in ("historical", *srcs)}}
    g["V2_frozen_perception"] = all(det["V2"].values())
    det["V3"] = {"adapter": pf["adapter_historical"]["ok"], "rig": pf["rig"]["ok"], **{sid: pf["parents"][sid]["ok"] for sid in srcs}}
    g["V3_rig_adapter_and_parent"] = all(det["V3"].values())
    # V4 window selection recomputed from the stored candidates
    resel = {}
    for sid in srcs:
        ch = AB.select_windows(win["sources"][sid]["candidates"])
        resel[sid] = all((ch[k] is None and decl["selection"][sid].get(t) is None) or
                         (ch[k] is not None and decl["selection"][sid].get(t) and ch[k]["frames"] == decl["selection"][sid][t]["frames"])
                         for t, k in (("NL", "LOWER"), ("NH", "HIGHER")))
        for t in ("NL", "NH"):
            h = decl["selection"][sid].get(t)
            if h:
                dom = set(range(h["frames"][0] - 8, h["frames"][-1] + 9))
                for lo, hi in win["sources"][sid]["w121_windows"]:
                    if dom & set(range(lo - 8, hi + 9)):
                        resel[sid] = False
    det["V4"] = {"reselect_equal_and_disjoint": resel, "selection_hash": sha(Path(decl["dirs"]["windows"]) / "selection.json") == decl["inputs_sha256"]["selection"]}
    g["V4_prospective_window_selection"] = all(resel.values()) and det["V4"]["selection_hash"] and decl["outcomes_inspected"] is False
    # V5 baselines and anchors
    det["V5"] = {sid: {"frozen_mismatch": [B1["sources"][sid]["frozen_controls_mismatch"], B2["sources"][sid]["frozen_controls_mismatch"]],
                       "anchor_bad": [r for r, v in B1["sources"][sid]["anchor_state_ids_equal_w121"].items() if not v],
                       "n_anchor": len(B1["sources"][sid]["anchor_state_ids_equal_w121"])} for sid in srcs}
    g["V5_baseline_and_anchor_reproduction"] = all(not any(v["frozen_mismatch"]) and not v["anchor_bad"] and v["n_anchor"] > 0 for v in det["V5"].values())
    recs = {r: B1["sources"][Rq[r]["source"]]["records"][r] for r in Rq}
    pop = [r for r, x in Rq.items() if x["role"] in ("evaluation", "control")]
    valid = [r for r in pop if not Rq[r]["validation_errors"]]
    iso_bad = [r for r in valid if not (recs[r]["isolation_values_F_r"]["ok"] and Rq[r]["variants"]["F"]["isolation_blender"]["ok"])]
    det["V6"] = {"isolation_failed": iso_bad}
    g["V6_clean_isolated_baselines"] = not iso_bad
    unfair, cross = [], []
    for r in valid:
        for f in FACTORS:
            if not Rq[r]["variants"][f]["fairness"]["ok"]:
                unfair.append(f"{r}/{f}")
        ci = FB.cross_factor_identity({f: recs[r]["factors"][f]["fair"] for f in FACTORS})
        if not ci["ok"]:
            cross.append({r: ci["errors"]})
    zb = {r: all(recs[r]["factors"][f]["info"]["budget_frames"] == recs[r]["pin_frames"] for f in FACTORS)
          for r in Rq if r.endswith("_ZB") or r.endswith("_NZB")}
    det["V7"] = {"unfair": unfair, "cross_factor": cross, "zero_budget_is_pin": zb}
    g["V7_factor_matching"] = not unfair and not cross and all(zb.values()) and len(zb) >= 6
    blobs_ok = B1["code_blobs"] == decl["code_blobs"] and B2["code_blobs"] == decl["code_blobs"]
    det["V8"] = {"instrument_all_ok": ins["all_ok"], "code_blobs_equal": blobs_ok,
                 "frozen_controls_hash": sha(Path(decl["dirs"]["calibrate"]) / "frozen_controls.json") == decl["frozen_controls_sha256"],
                 "declaration_hash_assess": A["declaration_sha256"] == a.declaration_sha and A2["declaration_sha256"] == a.declaration_sha,
                 "inputs_unchanged": all(sha(Path(decl["dirs"][k]) / f) == decl["inputs_sha256"][n] for k, f, n in
                                         (("preflight", "preflight.json", "preflight"), ("windows", "windows.json", "windows"),
                                          ("calibrate", "calibration.json", "calibration"), ("instrument", "instrument.json", "instrument"))),
                 "identity_review_unchanged": sha(decl["identity_review_path"]) == decl["inputs_sha256"]["identity_review"]}
    g["V8_predeclaration_and_instrument"] = decl["outcomes_inspected"] is False and all(det["V8"].values())
    miss = []
    for sid in srcs:
        for rid in decl["population"][sid]["requests"]:
            if rid not in Rq or not {"F", "X", *FACTORS} <= set(Rq[rid]["variants"]):
                miss.append(rid)
            elif not Rq[rid]["validation_errors"]:
                for v in ["X", *FACTORS]:
                    if Rq[rid]["variants"][v]["residual_m"]["n"] != len(Rq[rid]["pin_frames"]) * len(recs[rid]["request"]["constraints"]):
                        miss.append(f"{rid}/{v}")
    e2miss = {sid: sorted(set(B1["sources"][sid]["state_ids"]) - set(load(a.eval_dir2 / f"{sid}_eval.json")["states"])) for sid in srcs}
    det["V9"] = {"missing": miss, "e1_missing": {sid: A["sources"][sid]["missing_states"] for sid in srcs}, "e2_missing": e2miss}
    g["V9_blender_evaluation_and_population"] = not miss and not any(det["V9"]["e1_missing"].values()) and not any(e2miss.values())
    # V10 repaired repeatability
    v10 = {}
    for sid in srcs:
        r1, r2 = B1["sources"][sid]["records"], B2["sources"][sid]["records"]
        rr_ = AB.records_repeatable(r1, r2)
        cor, path = AB.semantic_corruption(r2)
        rc = AB.records_repeatable(r1, cor)
        v10[sid] = {"states_sha_equal": B1["sources"][sid]["states_sha256"] == B2["sources"][sid]["states_sha256"],
                    "records_equal_declared_exclusions": rr_["equal"], "non_vacuous": rr_["non_vacuous"], "n_compared": rr_["n_compared_leaves"],
                    "n_removed": rr_["n_removed"], "differing": rr_["differing_leaves"],
                    "eval_equal": sha(a.eval_dir / f"{sid}_eval.json") == sha(a.eval_dir2 / f"{sid}_eval.json"),
                    "live_corruption_path": path, "live_corruption_detected": not rc["equal"]}
    so1, so2 = load(a.assess_dir / "semantic_outcomes.json"), load(a.assess_dir2 / "semantic_outcomes.json")
    det["V10"] = {"per_source": v10, "syn_states_equal": B1["synthetic"]["states_sha256"] == B2["synthetic"]["states_sha256"],
                  "syn_eval_equal": sha(a.eval_dir / "syn_eval.json") == sha(a.eval_dir2 / "syn_eval.json"),
                  "assessments_semantic_equal": AB.canon(so1) == AB.canon(so2), "n_semantic_items": len(so1)}
    g["V10_repaired_repeatability"] = (all(v["states_sha_equal"] and v["records_equal_declared_exclusions"] and v["non_vacuous"] and v["eval_equal"]
                                           and v["live_corruption_detected"] for v in v10.values())
                                       and det["V10"]["syn_states_equal"] and det["V10"]["syn_eval_equal"] and det["V10"]["assessments_semantic_equal"])
    # V11 timing + real-time render identity
    vid_ok = {}
    for rid, v in rr["videos"].items():
        for kind, x in v.items():
            if isinstance(x, dict) and "fps" in x:
                rep_ = 4 if "slowed" in kind else 1
                vid_ok[f"{rid}/{kind}"] = abs(x["fps"] - 30.0) < 1e-6 and x["frames_decoded"] == x["frames_written"] == rep_ * len(v["frames"])
    det["V11"] = {"timing": {sid: A["sources"][sid]["timing"]["blender_frame_equals_index_minus_f0_plus_1"] for sid in srcs},
                  "videos_bad": [k for k, x in vid_ok.items() if not x], "n_videos": len(vid_ok)}
    g["V11_timing_and_realtime_render"] = all(det["V11"]["timing"].values()) and not det["V11"]["videos_bad"] and len(vid_ok) > 0
    # V12 repaired controls
    negs = {}
    for nid, (_, f, desc, cond, want, reasons) in W1.NEG.items():
        if nid not in Rq:
            negs[nid] = {"executed": False, "ok": False}
            continue
        d = Rq[nid]["variants"][f]["decision"]
        ok = [d["decision"], d["failure_class"]] == want and set(reasons) <= set(d["reasons"])
        if nid == "NG_SHARED":
            ok = ok and not recs[nid]["isolation_values_F_r"]["ok"] and not Rq[nid]["variants"]["F"]["isolation_blender"]["ok"]
        negs[nid] = {"got": d, "want": want, "ok": ok}
    dcs = {k: {"executed": v["executed"], "failed_truth_check": v["executed"] and not v["check"]["ok"], **{kk: v.get(kk) for kk in
                                                                                                          ("source", "factor", "refused", "reason", "precondition_decision", "status_before", "status_after")}}
           for k, v in A.get("display_controls", {}).items()}
    sdc = A["synthetic"]["display_controls"]
    dg3 = (dcs.get("DG3", {}).get("failed_truth_check") or (sdc["DG3_SYN"]["executed"] and not sdc["DG3_SYN"]["check"]["ok"]))
    dg5 = (dcs.get("DG5", {}).get("failed_truth_check") and dcs["DG5"]["precondition_decision"] == "REJECT"
           and dcs["DG5"]["status_before"] == "REJECTED" and dcs["DG5"]["status_after"] == "ACCEPTED")
    dg5s = sdc["DG5_SYN"]["executed"] and not sdc["DG5_SYN"]["check"]["ok"]
    sc = {k: v["detected"] for k, v in A["source_controls"].items()}
    rb = {sid: A["sources"][sid]["lineage"]["revert_blender"].get("inexact_rollback_detected") for sid in srcs}
    det["V12"] = {"negatives": negs, "display": dcs, "display_syn": {k: {"executed": v["executed"], "failed": v["executed"] and not v["check"]["ok"]} for k, v in sdc.items()},
                  "DG3_ok": dg3, "DG5_ok": dg5, "DG5_SYN_ok": dg5s, "source_controls": sc, "inexact_rollback": rb}
    g["V12_repaired_negative_controls"] = (all(x["ok"] for x in negs.values()) and all(dcs[k]["failed_truth_check"] for k in ("DG1", "DG2", "DG4"))
                                           and bool(dg3) and bool(dg5) and bool(dg5s) and all(sc.values()) and all(rb.values()))
    # V13 blinding
    det["V13"] = {"seal_ok": seal["ok"], "ratings_hash_at_unblind": UB["seal_sha256"] == sha(a.seal_dir / "seal.json"),
                  "mapping_hash": UB["mapping_sha256"] == rr["sealed_mapping_sha256"] == seal["mapping_sha256_at_render"],
                  "seed": UB["seed"] == decl["blind"]["seed"],
                  "mapping_recomputed": AB.blind_mapping(rr["eligible_pairs"], decl["blind"]["seed"])[0] == {r: x["mapping"] for r, x in UB["results"].items()},
                  "all_pairs_rated": set(UB["results"]) == set(rr["eligible_pairs"]),
                  "seal_before_unblind": seal["sealed_utc"] <= datetime.datetime.fromtimestamp(os.path.getmtime(a.unblind_dir / "unblinded.json"),
                                                                                                datetime.timezone.utc).isoformat(timespec="seconds")}
    g["V13_blinding"] = all(det["V13"].values())
    # V14 identity
    lab = {f"{sid}/{w}": x["identity"] for sid, ws in idr["windows"].items() for w, x in ws.items()}
    det["V14"] = {"labels": lab, "valid_values": all(v in ("VERIFIED", "UNKNOWN") for v in lab.values()),
                  "verified_have_evidence": all(str(idr["windows"][k.split("/")[0]][k.split("/")[1]].get("evidence", "")).strip() for k, v in lab.items() if v == "VERIFIED"),
                  "parents_unknown": all(pf["parents"][sid]["source_binding"]["video_subject_identity"].startswith("UNKNOWN") for sid in srcs)}
    g["V14_identity_labelling"] = all(v for k, v in det["V14"].items() if k != "labels")
    sy = A["synthetic"]
    det["V15"] = {"state_ids_equal_w120": B1["synthetic"]["state_ids_equal_w120"], "eval_bit_equal_w120": all(sy["eval_bit_equal_w120"].values()),
                  "decisions_equal_w120": {r: x["decisions_equal_w120"] for r, x in sy["requests"].items()}}
    g["V15_synthetic_controls"] = det["V15"]["state_ids_equal_w120"] and det["V15"]["eval_bit_equal_w120"] and all(det["V15"]["decisions_equal_w120"].values())
    # C gates (W121 definitions)
    c1, c2, c5, c6 = [], [], [], []
    for r in pop:
        R = Rq[r]
        planned = {}
        for v in ["X", *FACTORS]:
            x = R["variants"][v]
            fp = x["footprint"]
            planned[v] = x["planned_frames"]
            if not set(x["planned_frames"]) <= set(x["budget_frames"]):
                c2.append(f"{r}/{v}:planned_not_in_budget")
            if not x["display_check_proposed"]["ok"] or not x.get("display_check_result", {}).get("ok", False):
                c2.append(f"{r}/{v}:display")
            if not x.get("commit", {}).get("preview_id_equals_candidate", False):
                c2.append(f"{r}/{v}:preview_id")
            if v != "X" and (x.get("info") or {}).get("fallback") is not None:
                c6.append(f"{r}/{v}:fallback")
            if x["decision"]["decision"] == "ACCEPT":
                if x["residual_m"]["n_over_tol"] or not all(x["frame_pass"].values()):
                    c1.append(f"{r}/{v}:pin_or_frame")
                if not (set(fp["frames_changed"]) <= set(x["budget_frames"]) and set(fp["frames_changed"]) <= set(x["planned_frames"])
                        and set(fp["channel_frames"]) <= set(recs[r]["auth"]) and not fp["outside_arm_points_changed"] and not fp["matrix_world_changed_frames"]):
                    c1.append(f"{r}/{v}:scope")
                if not x["user_edit_and_object_equal_U"]:
                    c5.append(f"{r}/{v}")
                if set((recs[r].get("F_statuses") or {}).values()) - {"SOLVED", "NO_CHANGE_NEEDED"}:
                    c6.append(f"{r}/{v}:infeasible_accepted")
                if v != "X" and (x.get("info") or {}).get("request_status") == "G0_REFERENCE_MISSING":
                    c6.append(f"{r}/{v}:g0_missing")
        if not R["validation_errors"] and len({json.dumps(p) for p in planned.values()}) != 1:
            c2.append(f"{r}:planned_differs_across_variants")
    det["C1"], det["C2"], det["C5"], det["C6"] = c1, c2, c5, c6
    g["C1_hard_pin_and_scope"], g["C2_truthful_footprint"], g["C5_user_edit_and_collateral"], g["C6_no_fallback"] = not c1, not c2, not c5, not c6
    c3 = {}
    for sid in srcs:
        L = A["sources"][sid]["lineage"]
        st = {p["step"]: p for p in L["programme"]}
        c3[sid] = {"R1_present": "R1" in st, "R1_accept_moves_head": "R1" in st and st["R1"]["decision"] == "ACCEPT" and st["R1"]["head_after"] != st["R1"]["head_before"],
                   "R2_reject_head_unchanged": st["R2"]["decision"] == "REJECT" and st["R2"]["head_unchanged"],
                   "R3_stale": "R3" in st and st["R3"]["reasons"] == ["STALE_PARENT"],
                   "R4_revert_exact": "R4" in st and st["R4"]["values_equal_restored"] and L["revert_blender"].get("value_digest_equal_v0")
                   and L["revert_blender"].get("frames_bit_equal_v0")}
    det["C3"] = c3
    g["C3_atomicity_lineage_reversal"] = all(all(v.values()) for v in c3.values())
    g["C4_source_immutability"] = A["source_guard"]["unchanged"] and A2["source_guard"]["unchanged"]
    # ---- held-out mechanism evidence
    idmap = {(sid, w): x["identity"] for sid, ws in idr["windows"].items() for w, x in ws.items()}
    held = {}
    hpop = [r for r, x in Rq.items() if x.get("population") == "HELD_OUT"]
    for r in hpop:
        R = Rq[r]
        fx = R["comparisons"].get("factorial")
        num = {"guide_primary": False, "guide_conditional": False, "floor_primary": False, "floor_conditional": False}
        harm = False
        if fx:
            num = {"guide_primary": fx["contrasts"]["guide_at_F0"]["improves"]["meaningful_improvement"],
                   "guide_conditional": fx["contrasts"]["guide_at_F1"]["improves"]["meaningful_improvement"],
                   "floor_primary": fx["contrasts"]["floor_at_G0"]["improves"]["meaningful_improvement"],
                   "floor_conditional": fx["contrasts"]["floor_at_G1"]["improves"]["meaningful_improvement"]}
            att = R.get("attribution") or {}
            harm = AB.raw_harm(att.get("G1F0", {}).get("domain_raw_wrist_peak_m"), att.get("G0F0", {}).get("domain_raw_wrist_peak_m"))
        held[r] = {"source": R["source"], "window": R["window"], "identity": idmap.get((R["source"], R["window"]), "UNKNOWN"), "numeric": num,
                   "visual": UB["results"].get(r, {}).get("unblinded"), "raw_harm": harm,
                   "decision_X": R["variants"]["X"]["decision"]["decision"], "eligible_factorial": bool(fx)}
    V = {k: v for k, v in g.items() if k.startswith("V")}
    Cg = {k: v for k, v in g.items() if k.startswith("C")}
    case = AB.case_rule(V, Cg, held, data_blocked=False)
    # anchors (replication) and summary, reported separately
    anchors = {}
    for r, R in Rq.items():
        if R.get("population") != "ANCHOR_W121_REPLICATION":
            continue
        fx = R["comparisons"].get("factorial")
        anchors[r] = {"decision_X": R["variants"]["X"]["decision"]["decision"],
                      "guide_primary": fx and fx["contrasts"]["guide_at_F0"]["improves"]["meaningful_improvement"],
                      "floor_primary": fx and fx["contrasts"]["floor_at_G0"]["improves"]["meaningful_improvement"],
                      "visual": UB["results"].get(r, {}).get("unblinded")}
    rep = {"schema": "w122_decision_v1", "declaration_sha256": a.declaration_sha, "case": case["case"], "case_detail": case, "gates": g,
           "detail": det, "held_out": held, "anchors_replication": anchors, "label": "AGENT-RUN; AGENT REVIEW; not Reviewer-verified",
           "historical_cases_unchanged": HIST_CASES, "w121_official_case": "D (immutable)"}
    dump(out / "decision.json", rep)
    print(json.dumps({"case": case["case"], "gates": g, "case_detail": {k: v for k, v in case.items() if k != "per_request"}}, indent=1, default=str))


def cmd_manifest(a):
    files = {}
    for p in sorted(BASE.rglob("*")):
        rel = p.relative_to(BASE)
        if p.is_file() and rel.parts[0] not in ("tmp", "cache") and p.name != "evidence_manifest.json":
            files[str(rel).replace("\\", "/")] = sha(p)
    dump(BASE / "evidence_manifest.json", {"schema": "w122_manifest_v1", "declaration_sha256": a.declaration_sha, "n_files": len(files), "files": files})
    print(len(files), sha(BASE / "evidence_manifest.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage")
    for n in ("out_dir", "preflight", "windows", "identity", "identity_review", "calibrate", "instrument", "declaration", "build_dir", "eval_dir",
              "build_dir2", "eval_dir2", "assess_dir", "assess_dir2", "render_dir", "ratings", "seal_dir", "unblind_dir"):
        ap.add_argument("--" + n.replace("_", "-"), type=Path)
    ap.add_argument("--declaration-sha")
    ap.add_argument("--tag")
    a = ap.parse_args()
    r = {"preflight": cmd_preflight, "windows": cmd_windows, "identity": cmd_identity, "calibrate": cmd_calibrate, "instrument": cmd_instrument,
         "declare": cmd_declare, "build": cmd_build, "evaluate": cmd_evaluate, "assess": cmd_assess, "render": cmd_render, "seal": cmd_seal,
         "unblind": cmd_unblind, "decide": cmd_decide, "manifest": cmd_manifest}[a.stage](a)
    sys.exit(r or 0)


if __name__ == "__main__":
    main()
