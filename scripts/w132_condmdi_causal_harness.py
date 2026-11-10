#!/usr/bin/env python3
"""W132 (EXPERIMENTAL, opt-in) harness: CondMDI conditioning-mechanism causal gate on the real S1 animation.

Reuses the W131 harness (``scripts/w131_condmdi_harness.py``), adapter, runner and the W129/W124 Blender stages UNCHANGED
(loaded by path / imported); W131 outputs and verdict stay frozen.  Stages (each refuses an existing output directory):

  preflight  repository / W131 COMPLETE / W131 instrument + artifact identity, W126/W129 inputs, pinned CondMDI and Motius
             clones, local checkpoint LFS hashes, model-card terms re-read at the pinned revision, reference-control
             availability decision.
  declare    prospective declaration BEFORE any new model output (fixed conditions, thresholds, rules, instrument hashes).
  causal     S1: preprocessing identity -> R0_w131code (unchanged W131 runner, exact replay) -> R0 (W132 runner, impute ON)
             -> R1 (impute OFF) -> R1_leak (OFF, W131 leak input) -> replay / flag isolation / leak / context gate /
             model-space boundary metrics / mechanism class.
  refctl     four-cell reference mask control (pos_rot_vel|pos x impute ON|OFF) on the frozen reference.
  candidate  only if the causal gate allows it: C_OFF projection -> separate .blend (+ composed on W126V) -> fresh
             readbacks -> footprint / pin / quality vs A, B, frozen W131 C_ON -> 30-fps video, sheets, blinded copies.
  assess     frozen CASE rule.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w132_condmdi_causal_harness.py <stage> --out-dir <dir> [...]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
_spec = importlib.util.spec_from_file_location("w131_harness", str(ROOT / "scripts" / "w131_condmdi_harness.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

import numpy as np                                                    # noqa: E402

import app.experimental_condmdi as C                                  # noqa: E402
import app.experimental_condmdi_causal as K                           # noqa: E402
import app.experimental_motionbricks as M                             # noqa: E402

W = Path("C:/Users/dna10/animcv-w132")
W131 = H.W
W131R = W131 / "results"
INFER132 = ROOT / "scripts" / "w132_condmdi_causal_infer.py"
BATCH = 132
STARTING_COMMIT = "2358a73adac6fa72bbbb03677e7e2769244ff9d5"
INSTRUCTION_BLOB = "5abe31a8cd1e272478ca56aa2290f102fe26cc45"
W131_DECL = ROOT / "examples" / "experimental_condmdi" / "w131_declaration.json"
W131_DECL_SHA = "dca4a891d21725f813b8cbb1867abaab8f237f59e6f36080086b2267355107e4"
FROZEN = {   # W131 evidence, hashes from the W131 evidence manifest / Worklog
    "w131_conditioning": (W131R / "run/S1/conditioning.npz", "b6e46ccfd7f51f611c47ef0f22fead413ca8b1159f0a0b7a23ec67e5fd3e8483"),
    "w131_conditioning_leak": (W131R / "run/S1/conditioning_leak.npz", "b07a8d3b2e5c37fe62d73a74cee5e1f0fe280e91f6e7ab4ed2a8fa6a23ae0913"),
    "w131_model_R1": (W131R / "run/S1/model_R1/model_io.npz", "2ec9355557ae47724447a77833c462d9508c747b4ee6ef5dcfd54404191b5b73"),
    "w131_projection": (W131R / "run/S1/projection.json", "133a2ebf12cb3e2605651f2e945792fa492b8cb1ab1b407e575ccfa1261ac598"),
    "w131_C_blend": (W131R / "run/S1/C_S1.blend", "8f49a93a7c012352696ebac48cca109d0b0176c3087dba09a8af708aece1da60"),
    "w131_run_result": (W131R / "run/S1/run_result.json", "f5fbdb8f6a1ec291a7b7047cf0d410b28abeec20167a3bfdc1dd445e669dcad1"),
    "w131_assessment": (W131R / "assess/assessment.json", "9633a3621065058f3cd0d23af1a0f6fed2baf507b578c5c00c510ed6307404ac"),
    "w131_declaration": (W131R / "declare/declaration.json", W131_DECL_SHA),
    "w131_smoke_t2m": (W131R / "smoke/smoke_t2m.npz", "c60b4b3542c2bd2461460db7afd89309cadac4f9b3262d4b6be55f6f1791ae7f"),
    "w131_smoke_result": (W131R / "smoke/result.json", "8f710d2cbbddf64105d91125376c95d244e72b00679ac4a95569bd8174751c6c"),
    "w131_evidence_manifest": (W131R / "evidence_manifest.json", "98505f804945b6a772dd2e63aa469ae5e207a2a9d17101dde6f2775ce73d7cc0"),
}
CONDMDI_REV = "0c58473066ef582b9464a66d5542e0d1b6ae66c8"
MOTIUS_REV = "d04a38c9bc9dab3f0e9e2d843f8a6b21eaa13abc"
HF_ID = "ZeyuLing/motius-condmdi-humanml3d"
REFERENCE = {"tier": "MODEL-GENERATED CLOSED-LOOP SYNTHETIC",
             "npz": FROZEN["w131_smoke_t2m"][0], "sha256": FROZEN["w131_smoke_t2m"][1],
             "origin": "W131 independent smoke: CondMDI text-to-motion (no observation), model-card caption, seed 42, CFG 2.5, ddim100, 120 samples",
             "caption": "a person walks forward and waves with the right hand",            # W131 smoke = model-card example
             "length": 120, "hidden_width": 19, "hidden_mode": "official build_observation_mask(mode='middle', joint_indices=None = all 22 joints)",
             "seed": 7,
             "seed_reason": "seed 42 is the noise that generated the reference itself (degenerate closed loop); 7 is the pre-existing W131 smoke mask-contract seed",
             "cells": {"i": {"feature_mode": "pos_rot_vel", "impute": True}, "ii": {"feature_mode": "pos", "impute": True},
                       "iii": {"feature_mode": "pos_rot_vel", "impute": False}, "iv": {"feature_mode": "pos", "impute": False}}}
AUTHENTIC_CANDIDATE = {"path": "ext/motius/tests/assets/humanml3d/004822_protocol.npz",
                       "content": "hml263 (199,263) + joints, 'HumanML3D official tiny sample' Motius test fixture",
                       "decision": "NOT USED: AMASS-derived HumanML3D motion redistributed as a test fixture with no data-licence grant; "
                                   "lawful availability not established (AMASS terms restrict redistribution). condmdi/dataset/000021.npy "
                                   "is raw 52-joint AMASS-derived positions, same status and not HML263."}
INSTRUMENTS = ["src/app/experimental_condmdi_causal.py", "scripts/w132_condmdi_causal_infer.py", "scripts/w132_condmdi_causal_harness.py",
               "tests/test_experimental_condmdi_causal.py"] + list(H.INSTRUMENTS)
BRANCHES = ("A", "B", "C_ON", "C_OFF")
NAMES = {"A": "A original (no-op)", "B": "B slerp (W129, no model)", "C_ON": "C_ON CondMDI impute ON (W131)",
         "C_OFF": "C_OFF CondMDI impute OFF (W132)"}
QA_TEXT = {
    "QA1": "C_OFF has 0 ARM-LOCAL wrist reversal flicks in steps 171->172 .. 199->200 (W131 definition)",
    "QA2": "continuity: C_OFF arm-local entry/exit C0 steps each <= 2 x B's + 10 mm AND max(entry, exit) < C_ON's AND arm-local RMS jerk < C_ON's",
    "QA3": "C_OFF has no near-straight (>175 deg) interior frame, no bend-direction flip, no hinge-opposite frame",
    "QA4": "source fidelity: C_OFF mean interior WORLD wrist deviation from A < B's AND < C_ON's",
    "QA5": "press intent (W131 rule): wrist-above-shoulder never dips > 50 mm below the lower endpoint value (171/200)",
    "QA6": "blinded agent review: C_OFF most natural (V2), not named for a flick (V1) or a wrong elbow (V3), not least faithful and ranked above BOTH B and C_ON (V4), read as an upward press (V6)",
}
VISUAL_QUESTIONS = {
    "V1": "Which panel(s) show a visible unnatural one-frame flick/teleport/reversal of the left arm (Action 171-201)?",
    "V2": "Which panel's left-arm motion through the interior looks most natural and temporally coherent?",
    "V3": "Which panel(s) show an anatomically wrong elbow (hyperextension, flip) or limb-length artefact?",
    "V4": "Rank the four panels most -> least by agreement with the source video's visible left-arm pose and timing (rack -> press).",
    "V5": "Is any boundary discontinuity visible at 171/172 or 199/200 in any panel?",
    "V6": "For each panel: does the interior read as an upward overhead press (yes) or as a swing / detour / hold / other (no)?",
}
CASE_RULE = {
    "DATA_BLOCKED": "a required authentic prerequisite (rights, exact W131 checkpoint/artifacts, S1 source/baseline/rig, W126/W129 inputs, runtime, Blender) is absent or mismatched before the S1 causal pair",
    "D": "integrity falsified: W131 R0 exact replay mismatch (either runner), ON/OFF flag-isolation check fails (OFF changes anything but the diffusion-step replacement), the R1 masked-input leak control is not bitwise equal, a negative control does not fail closed, or a declaration/instrument hash changed",
    "C": "real inference ran but R1 fails the observed-context gate, the projection is unsafe/refused, or a C_OFF Action/pin/clock/video gate fails: no candidate acceptance",
    "B": "authentic paired runs and (when built) a safe comparison, but any QA1-QA6 fails: mechanism evidence, quality negative",
    "A": "all gates pass AND QA1-QA6 pass (narrowly supported; supports only a follow-up human decision, never promotion)",
    "precedence": "DATA_BLOCKED > D > C > B > A",
}
MECHANISM_RULE = {
    "inputs": "left-wrist model-space ric (root-relative heading-frame) positions; R0 = impute ON (= W131 R1), R1 = impute OFF; "
              "effective entry = |out[114] - observed[113]|, effective exit = |observed[133] - out[132]|, internal = same with out[113]/out[133]",
    "PERSISTS_UNDER_LEARNED_CONDITIONING": "max(R1 internal entry, exit) > 0.5 x max(R0 effective entry, exit)",
    "OFF_CONTINUOUS_ONLY_BY_DEPARTING_FROM_CONTEXT": "else, if the R1 context gate fails or max(R1 effective) > 0.5 x max(R0 effective)",
    "REPLACEMENT_IMPUTATION_PRIMARY_SUPPORTED": "otherwise (R1 continuous AND consistent with the observed context)",
    "limits": "one source, one seed, one text: does not separately identify proxy torso, source jitter, text semantics or training coverage",
}
CTX_RULE = {"arm_window": f"left elbow/wrist ric error at observed samples 103..113 and 133..143 <= {K.CTX_ARM_WINDOW_MAX_M} m",
            "observed_mean": f"mean ric error over every observed joint-sample <= {K.CTX_OBS_MEAN_M} m",
            "root_height": f"max |root height error| <= {K.CTX_ROOT_HEIGHT_MAX_M} m",
            "root_yaw": f"max |root yaw error| <= {K.CTX_ROOT_YAW_MAX_RAD:.6f} rad (5 deg)",
            "consequence": "FAIL -> R1 cannot be accepted as an edit: no candidate projection (CASE C)"}

dump, load, new_dir, lf_sha, git, blender = H.dump, H.load, H.new_dir, H.lf_sha, H.git, H.blender


def now():
    return H.now()


def cm132(req, req_path):
    dump(req_path, req)
    t = time.time()
    r = subprocess.run([str(H.CM_PY), "-I", str(INFER132), str(req_path)], capture_output=True, text=True, cwd=str(W))
    Path(str(req_path) + ".log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    return r.returncode, time.time() - t


def check_declaration(a):
    if not a.declaration:
        raise SystemExit("--declaration required")
    if M.sha256_file(a.declaration) != a.declaration_sha:
        raise SystemExit("BLOCKED_DECLARATION_CHANGED")
    d = load(a.declaration)
    changed = [f for f, h in d["instrument_sha256_lf"].items() if lf_sha(f) != h]
    if changed:
        raise SystemExit(f"BLOCKED_INSTRUMENT_CHANGED {changed}")
    return d


def frozen_ok():
    return {k: {"path": str(p), "sha256": M.sha256_file(p) if p.is_file() else None, "ok": p.is_file() and M.sha256_file(p) == h}
            for k, (p, h) in FROZEN.items()}


def base_request():
    return {"hf_dir": str(H.HF_DIR), "lfs_sha256": H.LFS, "respacing": H.RESPACING, "guidance": H.GUIDANCE}


def inputs132():
    """W131 frozen inputs (hash-verified by the W131 ``inputs``) + the frozen W131 C_ON .blend copied into W132 scratch."""
    inp131 = H.inputs()
    inp = W / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    for name, src in (("S1_baseline.blend", inp131 / "S1_baseline.blend"), ("S1_W126V_accepted.blend", inp131 / "S1_W126V_accepted.blend"),
                      ("S1_W129_B.blend", inp131 / "S1_W129_B.blend"), ("S1_W131_C_ON.blend", FROZEN["w131_C_blend"][0])):
        if not (inp / name).exists():
            shutil.copy2(src, inp / name)
    want = {"S1_baseline.blend": H.S1["baseline_sha"], "S1_W126V_accepted.blend": H.S1["pin_accepted_sha"],
            "S1_W129_B.blend": H.S1["w129_B_sha"], "S1_W131_C_ON.blend": FROZEN["w131_C_blend"][1]}
    for name, h in want.items():
        if M.sha256_file(inp / name) != h:
            raise SystemExit(f"BLOCKED_SOURCE_IDENTITY {name}")
    return inp


# ------------------------------------------------------------------------------------------------ preflight ---
def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "animcv-w132-preflight/1.0"}), timeout=60) as r:
        return r.read()


def stage_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"stage": "w132_preflight", "time": now(), "schema": K.SCHEMA}
    w131_wl = ROOT / "docs/131_WORKLOG_CONDMDI_PARTIAL_JOINT_LEFT_ARM_COMPLETION.md"
    audit = ROOT / "docs/AUDIT_W126_W130_2026_11_09.md"
    rep["repo"] = {"head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
                   "origin_head": git("rev-parse", "origin/arch/single_frame_first"),
                   "starting_commit_is_ancestor": subprocess.run(["git", "merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD"], cwd=ROOT).returncode == 0,
                   "w131_complete": "AUTOMATION_BATCH_STATUS: COMPLETE" in w131_wl.read_text(encoding="utf-8"),
                   "audit_present": audit.is_file()}
    rep["repo"]["blob_ok"] = rep["repo"]["instruction_blob"] == INSTRUCTION_BLOB
    d131 = load(W131_DECL)
    rep["w131_declaration_sha_ok"] = M.sha256_file(W131_DECL) == W131_DECL_SHA
    rep["w131_instruments_unchanged"] = {f: lf_sha(f) == h for f, h in d131["instrument_sha256_lf"].items()}
    rep["frozen"] = frozen_ok()
    try:
        inp = inputs132()
        rep["inputs"] = {"dir": str(inp), "ok": True}
    except SystemExit as e:                                                       # noqa: PERF203
        rep["inputs"] = {"ok": False, "error": str(e)}
    refs = load(H.S1["run"] / "lineage" / "refs.json")
    rep["lineage"] = {"head": refs["head"], "value_digest": refs["head_blend"]["value_digest"],
                      "ok": refs["head"] == H.S1["head_state"] and refs["head_blend"]["value_digest"] == H.S1["value_digest"]}
    src = {}
    for key, rev, local in (("condmdi", CONDMDI_REV, W131 / "ext" / "condmdi"), ("motius", MOTIUS_REV, W131 / "ext" / "motius")):
        head = git("-C", str(local), "rev-parse", "HEAD")
        src[key] = {"pinned": rev, "head": head, "ok": head == rev, "clean": git("-C", str(local), "status", "--porcelain") == ""}
    cl, ml = W131 / "ext" / "condmdi", W131 / "ext" / "motius" / "motius" / "licenses" / "condmdi"
    src["condmdi"]["license_sha256"] = M.sha256_file(cl / "LICENSE")
    src["condmdi"]["license_head"] = (cl / "LICENSE").read_text(encoding="utf-8").splitlines()[:3]
    src["motius"]["condmdi_licence_files"] = {p.name: M.sha256_file(p) for p in sorted(ml.iterdir()) if p.is_file()}
    src["motius"]["licence_heads"] = {n: (ml / n).read_text(encoding="utf-8").splitlines()[:2] for n in ("LICENSE", "CLIP_LICENSE")}
    w131pf = load(W131R / "preflight" / "preflight.json")
    src["licence_files_equal_w131_preflight"] = (src["condmdi"]["license_sha256"] == w131pf["sources"]["condmdi"]["license_sha256"]
                                                 and src["motius"]["condmdi_licence_files"] == w131pf["sources"]["motius"]["condmdi_licence_files"])
    rep["sources"] = src
    t = time.time()
    rep["checkpoint_local_lfs"] = {n: M.sha256_file(H.HF_DIR / n) == h for n, h in H.LFS.items()}
    rep["checkpoint_hash_s"] = time.time() - t
    terms = {"network": True}
    try:
        api = json.loads(_get(f"https://huggingface.co/api/models/{HF_ID}/revision/{H.HF_REV}"))
        card = _get(f"https://huggingface.co/{HF_ID}/resolve/{H.HF_REV}/README.md")
        (out / "hf_README.md").write_bytes(card)
        txt = card.decode("utf-8")
        terms.update({"api_sha": api.get("sha"), "api_sha_ok": api.get("sha") == H.HF_REV, "card_license": (api.get("cardData") or {}).get("license"),
                      "gated": api.get("gated"), "readme_sha256": hashlib.sha256(card).hexdigest(),
                      "readme_equal_w131": hashlib.sha256(card).hexdigest() == M.sha256_file(W131R / "preflight" / "hf_meta" / "README.md"),
                      "frontmatter_license_mit": (re.search(r"(?m)^license:\s*(\S+)", txt) or [None, None])[1] == "mit",
                      "mentions_hymotion": any(s in txt.lower() for s in ("hy-motion", "hymotion", "hunyuan", "tencent"))})
    except Exception as e:                                                         # noqa: BLE001
        terms.update({"network": False, "error": repr(e)})
    terms["reading"] = ("checkpoint card licence MIT, code MIT, CLIP MIT; training data HumanML3D<-AMASS (non-commercial research): "
                        "declared use = local non-commercial research evaluation. The code MIT licence is NOT read as a commercial "
                        "deployment grant for the checkpoint. Agent's documentary reading, not legal advice.")
    rep["terms"] = terms
    z = np.load(REFERENCE["npz"])
    rep["reference"] = {"tier": REFERENCE["tier"], "sha_ok": M.sha256_file(REFERENCE["npz"]) == REFERENCE["sha256"],
                        "arrays": {k: list(z[k].shape) for k in z.files}, "finite": bool(all(np.isfinite(z[k]).all() for k in z.files)),
                        "authentic_candidate_found": AUTHENTIC_CANDIDATE}
    rep["reference"]["ok"] = bool(rep["reference"]["sha_ok"] and rep["reference"]["finite"] and z["relative"].shape == (120, 263)
                                  and z["absolute"].shape == (120, 263))
    rep["blender"] = Path(H.BLENDER).is_file()
    rep["runtime_python"] = H.CM_PY.is_file()
    reasons = []
    if not (rep["repo"]["blob_ok"] and rep["repo"]["starting_commit_is_ancestor"] and rep["repo"]["w131_complete"]):
        reasons.append("repo/W131 state")
    if not rep["w131_declaration_sha_ok"] or not all(rep["w131_instruments_unchanged"].values()):
        reasons.append("W131 declaration/instruments changed")
    if not all(v["ok"] for v in rep["frozen"].values()):
        reasons.append("W131 frozen artifact mismatch")
    if not rep["inputs"]["ok"] or not rep["lineage"]["ok"]:
        reasons.append("S1 baseline/W126/W129 inputs")
    if not all(v["ok"] and v["clean"] for k, v in src.items() if isinstance(v, dict)) or not src["licence_files_equal_w131_preflight"]:
        reasons.append("source clones / licence files")
    if not all(rep["checkpoint_local_lfs"].values()):
        reasons.append("checkpoint LFS hashes")
    if terms["network"] and not (terms["api_sha_ok"] and terms["card_license"] == "mit" and terms["frontmatter_license_mit"] and not terms["mentions_hymotion"]):
        reasons.append("checkpoint terms changed")
    if not (rep["blender"] and rep["runtime_python"]):
        reasons.append("runtime/Blender")
    rep["reasons"] = reasons
    rep["status"] = "OK" if not reasons else "DATA_BLOCKED"
    sha = dump(out / "preflight.json", rep)
    print(json.dumps({"status": rep["status"], "reasons": reasons, "terms": {k: terms.get(k) for k in ("card_license", "readme_equal_w131", "api_sha_ok")},
                      "sha": sha}, indent=1))
    return 0 if not reasons else 3


# ------------------------------------------------------------------------------------------------ declaration ---
def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit("preflight not OK: no declaration")
    d131 = load(W131_DECL)
    d = {"schema": "animcv_exp_w132_declaration_v1", "time": now(), "batch": BATCH, "outcomes_inspected": False,
         "new_model_output_exists": any((W / "results" / s).exists() for s in ("causal", "refctl", "candidate")),
         "starting_commit": STARTING_COMMIT, "instruction_blob": INSTRUCTION_BLOB, "repo_head_at_declaration": git("rev-parse", "HEAD"),
         "preflight_sha256": M.sha256_file(a.preflight),
         "frozen_w131": {k: {"path": str(p), "sha256": h} for k, (p, h) in FROZEN.items()},
         "w131_declaration": {"path": str(W131_DECL), "sha256": W131_DECL_SHA},
         "model": {**d131["model"], "condmdi_code_rev": CONDMDI_REV, "motius_rev": MOTIUS_REV,
                   "terms": pf["terms"]["reading"]},
         "s1": {"source": d131["source"], "time_map": d131["time_map"], "representation_mask": d131["representation"]["mask"],
                "n_observed_features": d131["representation"]["n_observed_features"],
                "conditioning_npz": "frozen W131 run/S1/conditioning.npz (reused bytes; re-derivation from the baseline must be bitwise equal)",
                "edit_domain": "upperarm_l / lowerarm_l local quaternion keys at Action 172..199 (= SOURCE 250..277), model interior samples 114..132",
                "protected": "W126 pin Action 132..140 (SOURCE 210..218), outside the edit; all non-left-arm channels, object, frames outside 172..199"},
         "config": {"text": H.TEXT, "seed": H.SEED, "guidance": H.GUIDANCE, "respacing": H.RESPACING, "length": 160,
                    "network_precision": "fp32", "deterministic": "cudnn deterministic, TF32 off, use_deterministic_algorithms(warn_only)",
                    "sampler": "official CondMDIPipeline._sample -> p_sample_loop -> p_sample -> p_mean_variance (ancestral over the ddim100 respaced schedule)"},
         "conditions": {
             "R0_w131code": "UNCHANGED W131 runner (impute=True), frozen W131 conditioning.npz: exact replay; output must be bitwise equal to frozen W131 model_R1",
             "R0": "W132 instrumented runner, impute=True: must be bitwise equal to frozen W131 model_R1 (instrumentation non-invasive)",
             "R1": "W132 instrumented runner, impute=False; every other input, flag, seed and tensor identical",
             "R1_leak": "R1 with the frozen W131 conditioning_leak.npz (UNKNOWN rig channels rotated, seed 131) AND every masked input feature replaced by seeded noise (seed 131): output must be bitwise equal to R1",
             "determinism_rerun": "not planned (R0 replays a cross-session result; R1_leak doubles as an R1 rerun); run only if needed, labelled",
             "negative_controls": "runner refuses a mismatched input / reference SHA-256 (no model load); Blender stage refuses a channel outside the edit domain and a frame outside 172..199; declaration/instrument tamper refused (unit test)"},
         "flag_isolation": {"rule": "ON/OFF pass only if: identical input/absolute/mask/effective-conditioning arrays; identical seed/text/length/guidance/respacing; equal network-call count (= 2 x steps, CFG); one identical obs_x0 and obs_mask hash on every call of both runs; identical text tokens; y option keys differ exactly by {imputate, stop_imputation_at, replacement_distribution, inpainted_motion, inpainting_mask}; requires_imputation True on every step (ON) / False on every step (OFF); reconstruction guidance never; first-step network inputs and outputs identical; first divergence at or after network call 2",
                            "consequence": "fail -> CASE D"},
         "context_gate": CTX_RULE, "mechanism_rule": MECHANISM_RULE,
         "model_space_metrics": "R0 and R1: left wrist/elbow ric and world steps 113->114, 132->133 (internal and against the observed input), context errors at 113/133, observed-feature residuals by group, left-wrist height trajectory 104..142, interior peak timing, root travel / yaw range; reported before any projection",
         "reference_control": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in REFERENCE.items()},
                               "authentic_candidate": AUTHENTIC_CANDIDATE,
                               "hidden_samples": K.official_middle_frames(120, 19)[1],
                               "root_semantics": "official grouping, identical joints/frames in every cell: pelvis pos = root xz + height (observed in all four cells), pelvis rot = root yaw (observed only in pos_rot_vel cells), pelvis vel (pos_rot_vel only); the four foot-contact features are observed in ALL cells (official joint_mask_to_feature_mask always adds them); all values are the reference's own complete model-generated features (nothing synthetic is added)",
                               "metrics": "per cell vs the reference (as conditioned: relative_to_absolute of the frozen relative tensor): hidden MPJPE (ric/world), endpoint MPJPE at the first/last hidden sample, boundary excess = max over joints of (out step - reference step) at 49->50 and 68->69, observed-feature residual by group, observed-frame MPJPE",
                               "readings": f"a cell has a boundary discontinuity if its ric boundary excess > {K.REF_DISCONTINUITY_MM} mm; replacement effect = (i vs iii), (ii vs iv); feature-group effect = (i vs ii), (iii vs iv); descriptive only",
                               "limits": "MODEL-GENERATED CLOSED-LOOP SYNTHETIC: neither held-out data nor proof of on-distribution generalisation; not real-video quality evidence; no fifth cell"},
         "candidate": {"rule": "only if replay, flag isolation, leak and the R1 context gate pass: R1 interior samples 114..132 projected with the UNCHANGED W131 projection (W129 B twist reference, 20->30 slerp, no filtering) -> C_OFF separate .blend from the frozen baseline and C_OFF composed on the W126V-accepted .blend; fresh-process readbacks; footprint/pin gates as W131 G4/G5; A / B / C_ON (frozen W131 C .blend) / C_OFF on the same rig, renders, crop and 30-fps grid",
                       "quality_criteria": QA_TEXT, "visual_questions": VISUAL_QUESTIONS,
                       "visual_protocol": "panels W/X/Y/Z = random permutation of A/B/C_ON/C_OFF sealed in blind_key_SEALED.json; the agent answers from blinded stills before reading the key; AGENT-ONLY; human native-rate review NOT RUN. Disclosed weakness: the agent knows the W131 C_ON appearance (teleport at 171/172) and the R1 model-space numbers before review"},
         "case_rule": CASE_RULE,
         "no_tuning": "one model, one checkpoint, one S1 source, one text, one seed, one mask, one sampler; no alternative seed/text/guidance/steps/window/time warp/mask/threshold, no post-hoc endpoint correction or smoothing, no fifth reference cell",
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    sha = dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha, "new_model_output_exists": d["new_model_output_exists"]}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ causal ---
def _io(p):
    z = np.load(p)
    return {k: z[k] for k in z.files}


def _bitwise(a, b, keys=("input_relative", "input_absolute", "observation_mask_abs", "effective_conditioning_normalized", "output_absolute")):
    return {k: bool(np.array_equal(a[k], b[k])) for k in keys}


def stage_causal(a):
    check_declaration(a)
    out = new_dir(a.out_dir)
    res = {"stage": "causal", "start": now(), "head": git("rev-parse", "HEAD")}
    fz = frozen_ok()
    if not all(v["ok"] for v in fz.values()):
        res["STOP"] = "DATA_BLOCKED: frozen W131 artifact mismatch"
        dump(out / "causal_result.json", res)
        return 3
    inp = inputs132()
    cond, leak = FROZEN["w131_conditioning"], FROZEN["w131_conditioning_leak"]
    # ---- preprocessing identity: re-derive the S1 conditioning from the frozen baseline ----
    H.blender(H.STAGE, "dump", out / "dump_A.log", blend=inp / "S1_baseline.blend", report=out / "dump_A.json")
    bd = load(out / "dump_A.json")
    rig = C.Rig(bd["rest"])
    tm = C.time_map()
    C.validate_time_map(tm)
    kint = C.interior_samples(tm)
    J, enc, mask = H.conditioning(rig, bd["stored"], tm, kint)
    C.validate_mask(mask, kint)
    zc = np.load(cond[0])
    res["preprocessing_identity"] = {"relative_bitwise": bool(np.array_equal(enc["relative"].astype(np.float32), zc["relative_hml263"])),
                                     "mask_bitwise": bool(np.array_equal(mask, zc["observation_mask_abs"])),
                                     "r0_bitwise": bool(np.array_equal(enc["r0"], zc["r0"])), "interior_samples": [kint[0], kint[-1]],
                                     "observed_features": int(mask.sum()),
                                     "observed_left_arm_samples_inside_interior": int(mask[4 + 3 * 19, kint].sum())}
    # ---- negative control: runner refuses a mismatched input identity (no model load) ----
    rq = {**base_request(), "mode": "causal", "text": H.TEXT, "seed": H.SEED, "length": 160, "impute": False,
          "input_npz": str(cond[0]), "input_sha256": "0" * 64, "out_dir": str(out / "neg_input_identity")}
    rc, _ = cm132(rq, out / "req_neg_input_identity.json")
    res["neg_input_identity"] = {"rc": rc, "pass": rc == 3 and (out / "neg_input_identity" / "stop.json").is_file()
                                 and not (out / "neg_input_identity" / "model_io.npz").exists()}
    runs = {}

    def go(name, req, runner):
        rc_, t_ = (H.cm if runner == "w131" else cm132)({**req, "out_dir": str(out / f"model_{name}")}, out / f"req_{name}.json")
        runs[name] = {"rc": rc_, "process_s": t_, "runner": runner}
        if rc_ != 0:
            res["runs"] = runs
            res["STOP"] = f"model run {name} failed rc={rc_}"
            dump(out / "causal_result.json", res)
            raise SystemExit(res["STOP"])
        return _io(out / f"model_{name}" / "model_io.npz")
    w131io = _io(cond[0].parent / "model_R1" / "model_io.npz")
    io_w = go("R0_w131code", {**base_request(), "mode": "infer", "text": H.TEXT, "seed": H.SEED, "length": 160, "input_npz": str(cond[0])}, "w131")
    rep_w = _bitwise(io_w, w131io)
    res["replay_R0_w131code_vs_frozen"] = rep_w
    if not all(rep_w.values()):
        res["runs"] = runs
        res["STOP"] = "W131 R0 REPLAY MISMATCH (unchanged W131 runner): causal attribution stopped (CASE D)"
        res["replay_max_abs"] = float(np.abs(io_w["output_absolute"] - w131io["output_absolute"]).max())
        dump(out / "causal_result.json", res)
        print(json.dumps({"STOP": res["STOP"], "replay": rep_w}, indent=1))
        return 3
    c132 = {**base_request(), "mode": "causal", "text": H.TEXT, "seed": H.SEED, "length": 160}
    io0 = go("R0", {**c132, "impute": True, "input_npz": str(cond[0]), "input_sha256": cond[1]}, "w132")
    rep0 = _bitwise(io0, w131io)
    res["replay_R0_w132_vs_frozen"] = rep0
    if not all(rep0.values()):
        res["runs"] = runs
        res["STOP"] = "W131 R0 REPLAY MISMATCH (instrumented W132 runner): causal attribution stopped (CASE D)"
        dump(out / "causal_result.json", res)
        print(json.dumps({"STOP": res["STOP"], "replay": rep0}, indent=1))
        return 3
    io1 = go("R1", {**c132, "impute": False, "input_npz": str(cond[0]), "input_sha256": cond[1]}, "w132")
    io1l = go("R1_leak", {**c132, "impute": False, "input_npz": str(leak[0]), "input_sha256": leak[1], "perturb_masked_seed": H.LEAK_SEED}, "w132")
    r0, r1, r1l = (load(out / f"model_{n}" / "result.json") for n in ("R0", "R1", "R1_leak"))
    n_steps = r0["n_sampling_steps"]
    iso_ok, iso = K.flag_isolation(r0["instrumentation"] | {"request": r0["request"]}, r1["instrumentation"] | {"request": r1["request"]}, io0, io1, n_steps)
    res["flag_isolation"] = {"pass": iso_ok, "checks": iso,
                             "instrumentation_summary": {n: {"net_calls": r["instrumentation"]["n_net_calls"], "uncond_calls": r["instrumentation"]["uncond_calls"],
                                                             "imputation_true_steps": sum(r["instrumentation"]["requires_imputation"]),
                                                             "y_keys": r["instrumentation"]["y_keys"]} for n, r in (("R0", r0), ("R1", r1), ("R1_leak", r1l))}}
    res["leak_R1"] = {"input_differs": bool(not np.array_equal(io1["input_relative"], io1l["input_relative"])),
                      "effective_conditioning_bitwise": bool(np.array_equal(io1["effective_conditioning_normalized"], io1l["effective_conditioning_normalized"])),
                      "output_bitwise": bool(np.array_equal(io1["output_absolute"], io1l["output_absolute"])),
                      "max_abs": float(np.abs(io1["output_absolute"] - io1l["output_absolute"]).max()),
                      "obs_x0_hash_equal": r1["instrumentation"]["obs_x0_sha"] == r1l["instrumentation"]["obs_x0_sha"]}
    res["leak_R1"]["pass"] = bool(res["leak_R1"]["input_differs"] and res["leak_R1"]["effective_conditioning_bitwise"] and res["leak_R1"]["output_bitwise"])
    # ---- model-space evidence (before any projection) ----
    ms = {}
    for n, io in (("R0_ON", io0), ("R1_OFF", io1)):
        oa, ia, mk = io["output_absolute"].astype(np.float64), io["input_absolute"].astype(np.float64), io["observation_mask_abs"]
        ms[n] = {"boundary": K.s1_boundary(oa, ia, mk), "observed_residuals": K.observed_residuals(oa, ia, mk),
                 "trajectory": K.trajectory_summary(oa)}
    ms["source_trajectory"] = K.trajectory_summary(io0["input_absolute"].astype(np.float64))
    d01 = np.abs(io0["output_absolute"].astype(np.float64) - io1["output_absolute"].astype(np.float64))
    R0r, R1r = K.ric(io0["output_absolute"]), K.ric(io1["output_absolute"])
    ms["R1_vs_R0"] = {"max_abs_feature_diff": float(d01.max()), "outputs_bitwise_equal": bool(d01.max() == 0),
                      "left_wrist_ric_diff_interior_mean_m": float(np.linalg.norm(R1r[kint, 20] - R0r[kint, 20], axis=1).mean()),
                      "left_wrist_ric_diff_interior_max_m": float(np.linalg.norm(R1r[kint, 20] - R0r[kint, 20], axis=1).max())}
    ctx_on = K.context_gate(ms["R0_ON"]["observed_residuals"])
    ctx = K.context_gate(ms["R1_OFF"]["observed_residuals"])
    label, flags = K.mechanism(ms["R0_ON"]["boundary"], ms["R1_OFF"]["boundary"], ctx["pass"])
    res.update({"runs": runs, "model_space": ms, "context_gate_R0_ON_info": ctx_on, "context_gate_R1_OFF": ctx,
                "mechanism": {"label": label, "flags": flags},
                "costs": {n: {"sample_s": r["sample_s"], "load_s": r["load_s"], "peak_vram_allocated_bytes": r["peak_vram_allocated_bytes"],
                              "peak_vram_reserved_bytes": r["peak_vram_reserved_bytes"], "rss_bytes": r["rss_bytes"]} for n, r in (("R0", r0), ("R1", r1), ("R1_leak", r1l))},
                "env": r0["env"], "motius_head_sha": r0["motius_head_sha"]})
    integrity = bool(all(rep_w.values()) and all(rep0.values()) and iso_ok and res["leak_R1"]["pass"] and res["neg_input_identity"]["pass"]
                     and all(res["preprocessing_identity"][k] for k in ("relative_bitwise", "mask_bitwise", "r0_bitwise")))
    res["integrity_pass"] = integrity
    res["candidate_allowed"] = bool(integrity and ctx["pass"])
    res["end"] = now()
    sha = dump(out / "causal_result.json", res)
    print(json.dumps({"replay": [all(rep_w.values()), all(rep0.values())], "isolation": iso_ok, "leak": res["leak_R1"]["pass"],
                      "neg": res["neg_input_identity"]["pass"], "preproc": res["preprocessing_identity"], "context_gate_R1": ctx["pass"],
                      "mechanism": label, "candidate_allowed": res["candidate_allowed"], "sha": sha}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ reference control ---
def stage_refctl(a):
    check_declaration(a)
    out = new_dir(a.out_dir)
    res = {"stage": "refctl", "start": now(), "tier": REFERENCE["tier"], "reference_sha256": M.sha256_file(REFERENCE["npz"])}
    if res["reference_sha256"] != REFERENCE["sha256"]:
        res["STOP"] = "reference identity mismatch: control NOT RUN"
        dump(out / "refctl_result.json", res)
        return 3
    L, wd = REFERENCE["length"], REFERENCE["hidden_width"]
    _, hidden = K.official_middle_frames(L, wd)
    rq = {**base_request(), "mode": "refctl", "text": REFERENCE["caption"], "seed": REFERENCE["seed"], "length": L, "hidden_width": wd,
          "reference_npz": str(REFERENCE["npz"]), "reference_sha256": "0" * 64, "feature_mode": "pos", "impute": False,
          "out_dir": str(out / "neg_reference_identity")}
    rc, _ = cm132(rq, out / "req_neg_reference_identity.json")
    res["neg_reference_identity"] = {"rc": rc, "pass": rc == 3 and not (out / "neg_reference_identity" / "model_io.npz").exists()}
    z = np.load(REFERENCE["npz"])
    cells, ios = {}, {}
    for cid, spec in REFERENCE["cells"].items():
        rq = {**base_request(), "mode": "refctl", "text": REFERENCE["caption"], "seed": REFERENCE["seed"], "length": L, "hidden_width": wd,
              "reference_npz": str(REFERENCE["npz"]), "reference_sha256": REFERENCE["sha256"], "feature_mode": spec["feature_mode"],
              "impute": spec["impute"], "out_dir": str(out / f"cell_{cid}")}
        rc, t = cm132(rq, out / f"req_cell_{cid}.json")
        if rc != 0:
            res["STOP"] = f"cell {cid} failed rc={rc}"
            dump(out / "refctl_result.json", res)
            raise SystemExit(res["STOP"])
        io = _io(out / f"cell_{cid}" / "model_io.npz")
        rr = load(out / f"cell_{cid}" / "result.json")
        ios[cid] = io
        mk = io["observation_mask_abs"]
        npm = K.official_middle_mask(L, wd, spec["feature_mode"])
        cells[cid] = {**spec, "process_s": t, "sample_s": rr["sample_s"], "peak_vram_allocated_bytes": rr["peak_vram_allocated_bytes"],
                      "official_mask_equals_numpy_replica": bool(np.array_equal(mk, npm)), "mask_group_census": K.mask_group_census(mk),
                      "n_observed_features": int(mk.sum()), "imputation_true_steps": sum(rr["instrumentation"]["requires_imputation"]),
                      "net_calls": rr["instrumentation"]["n_net_calls"],
                      "metrics": K.refctl_metrics(io["output_absolute"].astype(np.float64), io["input_absolute"].astype(np.float64), mk, hidden)}
    m_prv, m_pos = ios["i"]["observation_mask_abs"], ios["ii"]["observation_mask_abs"]
    diff_groups = sorted({C.feature_owner(int(f))[0] for f in np.nonzero((m_prv ^ m_pos).any(1))[0]})
    res["paired_mask_check"] = {"pos_subset_of_pos_rot_vel": bool(np.all(m_pos <= m_prv)), "difference_groups": diff_groups,
                                "pairs_share_masks": bool(np.array_equal(ios["i"]["observation_mask_abs"], ios["iii"]["observation_mask_abs"])
                                                          and np.array_equal(ios["ii"]["observation_mask_abs"], ios["iv"]["observation_mask_abs"])),
                                "all_cells_same_input": all(np.array_equal(ios["i"]["input_relative"], ios[c]["input_relative"]) for c in ios),
                                "hidden_samples": hidden, "hidden_observes_nothing": bool(not m_prv[:, hidden].any())}
    res["reference_roundtrip_abs_max"] = float(np.abs(ios["i"]["input_absolute"].astype(np.float64) - z["absolute"].astype(np.float64)).max())
    res["reference_own_boundary_steps_mm"] = {nm: float(K.steps(K.ric(ios["i"]["input_absolute"]), k).max() * 1e3)
                                              for nm, k in (("entry", hidden[0] - 1), ("exit", hidden[-1]))}
    res["cells"] = cells
    res["readings"] = K.refctl_readings({c: cells[c]["metrics"] for c in cells})
    res["end"] = now()
    sha = dump(out / "refctl_result.json", res)
    print(json.dumps({"readings": res["readings"], "paired": res["paired_mask_check"]["difference_groups"], "neg": res["neg_reference_identity"]["pass"],
                      "parity": {c: cells[c]["official_mask_equals_numpy_replica"] for c in cells}, "sha": sha}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ candidate ---
def stage_candidate(a):
    check_declaration(a)
    out = new_dir(Path(a.out_dir) / "S1")
    cr = load(Path(a.causal_dir) / "causal_result.json")
    res = {"stage": "candidate", "start": now(), "causal_result_sha256": M.sha256_file(Path(a.causal_dir) / "causal_result.json")}
    if not cr.get("candidate_allowed"):
        res.update({"built": False, "reason": "causal gate does not allow a candidate (integrity or R1 context gate failed): projection NOT forced"})
        dump(out / "candidate_result.json", res)
        print(json.dumps(res, indent=1))
        return 0
    log = {"steps": []}
    inp = inputs132()
    refs_before = M.sha256_file(H.S1["run"] / "lineage" / "refs.json")
    base_blend, pin_blend = inp / "S1_baseline.blend", inp / "S1_W126V_accepted.blend"
    blends = {"A": base_blend, "B": inp / "S1_W129_B.blend", "C_ON": inp / "S1_W131_C_ON.blend"}
    log["steps"].append(["dump_A", blender(H.STAGE, "dump", out / "dump_A.log", blend=base_blend, report=out / "dump_A.json")])
    bd = load(out / "dump_A.json")
    st = bd["stored"]
    rig = C.Rig(bd["rest"])
    tm = C.time_map()
    kint = C.interior_samples(tm)
    zc = np.load(FROZEN["w131_conditioning"][0])
    r0 = zc["r0"]
    io0 = _io(Path(a.causal_dir) / "model_R0" / "model_io.npz")
    io1 = _io(Path(a.causal_dir) / "model_R1" / "model_io.npz")
    frames = range(C.INTERIOR_ACTION[0], C.INTERIOR_ACTION[1] + 1)

    def values(out_abs):
        up, lo = H.project_samples(rig, st, tm, kint, out_abs.astype(np.float64), r0)
        ru, rl = C.reconstruct_30(up, frames), C.reconstruct_30(lo, frames)
        return {"upperarm_l": {str(f): M.canon_sign(ru[f], C.stored_key(st, "upperarm_l", f)).tolist() for f in frames},
                "lowerarm_l": {str(f): M.canon_sign(rl[f], C.stored_key(st, "lowerarm_l", f)).tolist() for f in frames}}
    v_on = values(io0["output_absolute"])
    w131p = load(FROZEN["w131_projection"][0])
    res["C_ON_reprojection_equals_w131"] = v_on == w131p["values"]["C"]
    try:
        v_off = values(io1["output_absolute"])
    except M.AdapterStop as e:
        res.update({"built": False, "projection_refused": f"{e.code}: {e}", "reason": "R1 projection unsafe"})
        dump(out / "candidate_result.json", res)
        return 0
    dump(out / "projection_C_OFF.json", {"interior_samples": kint, "values": v_off, "transform": "UNCHANGED W131 projection; NO filtering"})
    # ---- negative controls: the Blender stage refuses out-of-domain writes ----
    neg = {}
    for name, vals in (("channel_outside_domain", {"spine_01": {"180": [1, 0, 0, 0]}}), ("frame_outside_interior", {"upperarm_l": {"210": [1, 0, 0, 0]}})):
        spec = {"out_blend": str(out / f"neg_{name}.blend"), "interior_action_frames": [172, 199], "values_wxyz": vals, "branch": "NEG",
                "action_name": "NEG", "expected_blend_sha256": H.S1["baseline_sha"]}
        dump(out / f"neg_{name}_spec.json", spec)
        try:
            blender(H.STAGE, "build_variant", out / f"neg_{name}.log", blend=base_blend, spec=out / f"neg_{name}_spec.json", report=out / f"neg_{name}_report.json")
            neg[name] = {"pass": False}
        except RuntimeError:
            txt = (out / f"neg_{name}.log").read_text()
            neg[name] = {"pass": ("REFUSED_" in txt) and not (out / f"neg_{name}.blend").exists(), "refusal": [ln for ln in txt.splitlines() if "REFUSED_" in ln][:1]}
    res["negative_controls"] = neg
    for br, src, sha_exp, nm in (("C_OFF", base_blend, H.S1["baseline_sha"], "AnimCV_EXP W132 C_OFF S1"),
                                 ("Cpin_OFF", pin_blend, H.S1["pin_accepted_sha"], "AnimCV_EXP W132 C_OFF_on_W126V S1")):
        spec = {"out_blend": str(out / f"{br}_S1.blend"), "interior_action_frames": list(C.INTERIOR_ACTION), "values_wxyz": v_off, "branch": br,
                "action_name": nm, "expected_blend_sha256": sha_exp, "note": "W132 CondMDI impute OFF (R1), seed 42; scratch proposal, never accepted"}
        dump(out / f"build_{br}_spec.json", spec)
        log["steps"].append([f"build_{br}", blender(H.STAGE, "build_variant", out / f"build_{br}.log", blend=src, spec=out / f"build_{br}_spec.json",
                                                    report=out / f"build_{br}_report.json")])
    blends["C_OFF"] = out / "C_OFF_S1.blend"
    rb = {}
    for br, bl in list(blends.items()) + [("Cpin_OFF", out / "Cpin_OFF_S1.blend"), ("Pin", pin_blend)]:
        log["steps"].append([f"readback_{br}", H.readback(bl, out / f"readback_{br}.json", out / f"readback_{br}.log")])
        rb[br] = load(out / f"readback_{br}.json")
    i0, i1 = C.INTERIOR_ACTION
    dumps = {"A": bd}
    for br in ("B", "C_ON", "C_OFF"):
        dumps[br] = H.full_dump(br, blends[br], out)
    g4 = {}
    for br in ("C_ON", "C_OFF"):
        r = rb[br]
        pres = M.preservation(r, rb["A"], (i0, i1))
        ck = M.changed_keys(dumps[br]["stored"], st)
        census = r["fcurve_census"]
        g = {"blend_sha256": r["blend_sha256"], "action_name": r["action_name"], "actions_in_file": r["actions_in_file"],
             "n_fcurves": census["n_fcurves"], "interp": census["interpolation_census_curves"], "key_range": [census["key_frame_min"], census["key_frame_max"]],
             "scene": r["scene"], "changed_keys_channels": sorted({c[0] for c in ck}),
             "changed_keys_frames": [min((c[2] for c in ck), default=None), max((c[2] for c in ck), default=None)], "n_changed_keys": len(ck),
             "nonarm_changed": pres["nonarm_changed"][:10], "outside_changed": pres["outside_changed"][:10], "sub_outside_changed": pres["sub_outside_changed"][:10]}
        g["pass"] = bool(len(r["actions_in_file"]) == 1 and r["action_name"] != rb["A"]["action_name"] and g["n_fcurves"] == 32
                         and set(g["interp"]) == {"CONSTANT"} and g["key_range"] == [1.0, 240.0] and r["scene"]["effective_fps"] == 30.0
                         and r["scene"]["frame_start"] == 1 and r["scene"]["frame_end"] == 240 and set(g["changed_keys_channels"]) <= set(C.EDIT_CHANNELS)
                         and all(i0 <= c[2] <= i1 for c in ck) and not pres["nonarm_changed"] and not pres["outside_changed"] and not pres["sub_outside_changed"])
        g4[br] = g
    a_same = M.sha256_file(blends["A"]) == H.S1["baseline_sha"] and rb["A"]["value_digest"] == H.S1["value_digest"]
    originals = (M.sha256_file(H.S1["run"] / "lineage" / "refs.json") == refs_before and M.sha256_file(H.S1["run"] / "baseline" / "baseline.blend") == H.S1["baseline_sha"]
                 and M.sha256_file(H.S1["run"] / H.S1["pin_accepted"]) == H.S1["pin_accepted_sha"] and M.sha256_file(H.S1["w129_B"]) == H.S1["w129_B_sha"]
                 and M.sha256_file(FROZEN["w131_C_blend"][0]) == FROZEN["w131_C_blend"][1])
    g4["A_untouched_and_digest"] = bool(a_same)
    g4["lineage_and_originals_unchanged"] = bool(originals)
    g4["pass"] = bool(g4["C_OFF"]["pass"] and a_same and originals)
    prop = load(H.S1["run"] / H.S1["pin_request"] / "proposal.json")
    tgt = np.asarray(prop["constraint"]["target"])
    pin_act = [f - H.S1["segment_first"] + 1 for f in prop["pin_frames"]]
    frp, frc, frC, frA = ({f["blender_frame"]: f for f in rb[k]["frames"]} for k in ("Pin", "Cpin_OFF", "C_OFF", "A"))
    res_pin = {f: float(np.linalg.norm(np.asarray(frp[f]["world"]["hand_l"]["head"]) - tgt)) for f in pin_act}
    res_cp = {f: float(np.linalg.norm(np.asarray(frc[f]["world"]["hand_l"]["head"]) - tgt)) for f in pin_act}
    g5 = {"pin_action_frames": pin_act, "pin_residual_W126V_m": res_pin, "pin_residual_C_OFF_composed_m": res_cp, "tolerance_m": prop["constraint"]["tolerance_m"],
          "pin_frames_bit_equal_C_OFF_vs_A": all(frC[f]["world"] == frA[f]["world"] for f in pin_act),
          "pin_frames_bit_equal_Cpin_vs_W126V": all(frc[f]["world"] == frp[f]["world"] for f in pin_act),
          "composed_preservation": M.preservation(rb["Cpin_OFF"], rb["Pin"], (i0, i1))}
    g5["pass"] = bool(g5["pin_frames_bit_equal_C_OFF_vs_A"] and g5["pin_frames_bit_equal_Cpin_vs_W126V"] and all(res_cp[f] == res_pin[f] for f in pin_act)
                      and max(res_cp.values()) <= 1e-6 and not g5["composed_preservation"]["nonarm_changed"] and not g5["composed_preservation"]["outside_changed"])
    hinge = M.hinge_reference(rb["A"])
    win = (i0 - 8, i1 + 8)
    q = {}
    for br in BRANCHES:
        q[br] = M.quality(rb[br], rb["A"], (i0, i1), win, hinge)
        q[br]["arm_local"] = M.armlocal_quality(M.armlocal_wrist(dumps[br], *win), (i0, i1), win)
        q[br]["press_intent"] = H.press_intent(rb[br])
    clock = {br: {"fps": rb[br]["scene"]["effective_fps"], "frames": [rb[br]["scene"]["frame_start"], rb[br]["scene"]["frame_end"]],
                  "key_range": [rb[br]["fcurve_census"]["key_frame_min"], rb[br]["fcurve_census"]["key_frame_max"]]} for br in BRANCHES}
    g2 = bool(all(v["fps"] == 30.0 and v["frames"] == [1, 240] and v["key_range"] == [1.0, 240.0] for v in clock.values()) and kint == list(range(114, 133)))
    res.update({"built": True, "G2_clock": g2, "clock": clock, "G4": g4, "G5": g5, "quality": q, "hinge_reference": hinge.tolist(),
                "blends": {k: str(v) for k, v in blends.items()}, "readbacks": {k: M.sha256_file(out / f"readback_{k}.json") for k in rb}})
    dump(out / "candidate_result.json", res)
    vid = render_and_compose(out, blends, rb, log)
    res["video"] = vid
    res["end"] = now()
    dump(out / "run_log.json", log)
    sha = dump(out / "candidate_result.json", res)
    print(json.dumps({"G2": g2, "G4": g4["pass"], "G5": g5["pass"], "neg": {k: v["pass"] for k, v in neg.items()},
                      "C_ON_reprojection_equals_w131": res["C_ON_reprojection_equals_w131"], "video": vid.get("native_rate"), "sha": sha}, indent=1))
    return 0


def render_and_compose(out, blends, rb, log):
    """W131 compositor generalised to A / B / C_ON / C_OFF (same views, crop, size, labels, 30-fps writer)."""
    import cv2
    i0, i1 = C.INTERIOR_ACTION
    n, seg0 = 240, H.S1["segment_first"]
    fa = {f["blender_frame"]: f for f in rb["A"]["frames"]}
    allp = np.array([fa[f]["world"][b][p] for f in fa for b in fa[f]["world"] for p in ("head", "tail")])
    centre = ((allp.min(0) + allp.max(0)) / 2).tolist()
    ext = float(np.max(allp.max(0) - allp.min(0)))
    el = np.array([fa[f]["world"]["lowerarm_l"]["head"] for f in range(i0 - 8, i1 + 9)]).mean(0).tolist()
    views = [{"name": "body", "centre": centre, "offset": [0.0, -6.0, 0.0], "ortho_scale": ext * 1.15},
             {"name": "arm", "centre": el, "offset": [4.5, -4.5, 1.5], "ortho_scale": 1.3}]
    size = 320
    dump(out / "render_jobs.json", {"size": size, "views": views, "frames": [{"blender_frame": f, "markers": []} for f in range(1, n + 1)]})
    for br in BRANCHES:
        log["steps"].append([f"render_{br}", blender(H.ACTION_STAGE, "render", out / f"render_{br}.log", blend=blends[br], jobs=out / "render_jobs.json",
                                                      out_dir=out / f"frames_{br}")])
    man = load(H.S1["run"] / "manifest" / "input_manifest.json")
    want = set(range(seg0, seg0 + n))
    cap = cv2.VideoCapture(man["intake"]["clip_path"])
    src, k = {}, 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k in want:
            src[k] = im
        k += 1
    cap.release()
    z = np.load(H.S1["run"] / "perception" / "m1s0" / "observations_predictions.npz")
    Wd, Hd = man["intake"]["frame_size"]
    pts = np.concatenate([z["input_2d"][kk][z["input_valid"][kk]][:, :2] for kk in sorted(want)]) * [Wd, Hd]
    c0, c1 = pts.min(0), pts.max(0)
    half = max(c1 - c0) * 0.6 + 20
    mid = (c0 + c1) / 2
    crop = [int(max(0, mid[0] - half)), int(max(0, mid[1] - half)), int(min(Wd, mid[0] + half)), int(min(Hd, mid[1] + half))]

    def lab(img, txt, y, col=(255, 255, 255), sc=0.42):
        cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
        return img

    def tile(br, f, label, changed):
        body = cv2.imread(str(out / f"frames_{br}" / f"body_f{f:04d}.png"))
        arm = cv2.imread(str(out / f"frames_{br}" / f"arm_f{f:04d}.png"))
        lab(body, label, 16)
        lab(body, f"Action {f}", 34)
        inside = i0 <= f <= i1
        lab(arm, "left-arm close-up", 16)
        lab(arm, "CHANGED interior" if (inside and changed) else ("interior" if inside else "unchanged"), 34, (80, 80, 255) if inside else (200, 200, 200))
        return np.vstack([body, arm])

    def source_tile(f):
        kk = f + seg0 - 1
        x0, y0, x1, y1 = crop
        im = src[kk][y0:y1, x0:x1].copy()
        im = cv2.resize(im, (int(im.shape[1] * 2 * size / im.shape[0]), 2 * size))
        lab(im, "SOURCE (fixed crop)", 16)
        lab(im, f"SOURCE frame {kk}  t={kk / 30.0:.3f}s", 34)
        return im

    def bar(width, f):
        b = np.full((34, width, 3), 245, np.uint8)
        Wb = width - 20
        for kk in range(1, n + 1):
            x0 = 10 + int((kk - 1) / n * Wb)
            x1 = 10 + int(kk / n * Wb)
            cv2.rectangle(b, (x0, 4), (max(x0, x1 - 1), 18), (91, 73, 219) if i0 <= kk <= i1 else (220, 220, 220), -1)
        xc = 10 + int((f - 0.5) / n * Wb)
        cv2.line(b, (xc, 0), (xc, 33), (0, 0, 0), 2)
        cv2.putText(b, f"edited interior Action {i0}-{i1} = SOURCE {i0 + seg0 - 1}-{i1 + seg0 - 1} (red); pin Action 132-140 protected; everything else unchanged",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        return b

    def write_video(path, order, labels, changed):
        wr, written = None, 0
        for f in range(1, n + 1):
            top = np.hstack([source_tile(f)] + [tile(br, f, lb, ch) for br, lb, ch in zip(order, labels, changed)])
            fr_ = np.vstack([top, bar(top.shape[1], f)])
            if wr is None:
                wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (fr_.shape[1], fr_.shape[0]))
            wr.write(fr_)
            written += 1
        wr.release()
        cp = cv2.VideoCapture(str(path))
        nd = 0
        while cp.read()[0]:
            nd += 1
        fps = cp.get(cv2.CAP_PROP_FPS)
        cp.release()
        return {"video": str(path), "sha256": M.sha256_file(path), "frames_written": written, "frames_decoded": nd, "fps": fps,
                "native_rate": nd == n and abs(fps - 30.0) < 1e-6}
    order = list(BRANCHES)
    vid = write_video(out / "abcc_S1_30fps.mp4", order, [NAMES[b] for b in order], [b != "A" for b in order])
    sheets = {"entry": list(range(170, 176)), "midpoint": list(range(183, 189)), "exit": list(range(196, 202)), "overview": list(range(171, 202, 5))}

    def sheet(path, frames_, order_, labels, changed):
        rows = [np.hstack([np.vstack([cv2.resize(source_tile(f), (size, size)), lab(np.zeros((size, size, 3), np.uint8), f"SOURCE {f + seg0 - 1} / Action {f}", 20)])
                           for f in frames_])]
        for br, lb, ch in zip(order_, labels, changed):
            rows.append(np.hstack([tile(br, f, lb, ch) for f in frames_]))
        cv2.imwrite(str(path), np.vstack(rows))
        return str(path)
    sh = {k: sheet(out / f"sheet_{k}_S1.png", v, order, [NAMES[b] for b in order], [b != "A" for b in order]) for k, v in sheets.items()}
    perm = list(BRANCHES)
    random.SystemRandom().shuffle(perm)
    bdir = new_dir(out / "blind")
    dump(out / "blind_key_SEALED.json", {"W": perm[0], "X": perm[1], "Y": perm[2], "Z": perm[3], "note": "do not read before the blinded review is written"})
    bl = [f"panel {x}" for x in "WXYZ"]
    bsh = {k: sheet(bdir / f"blind_sheet_{k}_S1.png", v, perm, bl, [True] * 4) for k, v in sheets.items()}
    bvid = write_video(bdir / "blind_S1_30fps.mp4", perm, bl, [True] * 4)
    vid.update({"sheets": sh, "blind_sheets": bsh, "blind_video": bvid, "blind_key_sha256": M.sha256_file(out / "blind_key_SEALED.json")})
    return vid


# ------------------------------------------------------------------------------------------------ assess ---
def stage_assess(a):
    check_declaration(a)
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    cr = load(Path(a.causal_dir) / "causal_result.json")
    rf = load(Path(a.refctl_dir) / "refctl_result.json")
    cd = load(Path(a.candidate_dir) / "S1" / "candidate_result.json")
    data_blocked = pf["status"] != "OK" or "DATA_BLOCKED" in str(cr.get("STOP", ""))
    integrity = bool(cr.get("integrity_pass")) and bool(rf.get("neg_reference_identity", {"pass": True})["pass"])
    gates = {"preflight": pf["status"], "integrity_causal": cr.get("integrity_pass"), "context_gate_R1": cr.get("context_gate_R1_OFF", {}).get("pass"),
             "candidate_built": cd.get("built")}
    qa, review, ans = None, None, None
    if cd.get("built"):
        neg_ok = all(v["pass"] for v in cd["negative_controls"].values())
        integrity = integrity and neg_ok
        gates.update({"neg_controls_blender": neg_ok, "G2": cd["G2_clock"], "G4": cd["G4"]["pass"], "G5": cd["G5"]["pass"], "G7": cd["video"]["native_rate"]})
        if a.visual_review:
            vr = load(a.visual_review)
            key = load(Path(a.candidate_dir) / "S1" / "blind_key_SEALED.json")
            ub = lambda x: key.get(x, x)                                                       # noqa: E731
            ans = {k: [ub(x) for x in v] if isinstance(v, list) else (ub(v) if isinstance(v, str) else {ub(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
                   for k, v in vr["answers_blinded"].items()}
            review = {"sha256": M.sha256_file(a.visual_review), "blind_key": key}
        qa = K.candidate_qa(cd["quality"], ans)
    unsafe = not (gates["context_gate_R1"] and cd.get("built") and gates.get("G2") and gates.get("G4") and gates.get("G5") and gates.get("G7"))
    case = K.case_rule(data_blocked, integrity, unsafe, qa)
    rep = {"stage": "assess", "time": now(), "case": case, "gates": gates, "integrity": integrity, "quality_criteria": qa,
           "visual_answers_unblinded": ans, "review": review, "mechanism": cr.get("mechanism"), "refctl_readings": rf.get("readings"),
           "refctl_tier": rf.get("tier"), "rule": CASE_RULE,
           "note": "visual review = the agent's own blinded review of stills; human native-rate review NOT RUN"}
    sha = dump(out / "assessment.json", rep)
    print(json.dumps({"case": case, "gates": gates, "qa": qa, "mechanism": rep["mechanism"]["label"] if rep["mechanism"] else None, "sha": sha}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("preflight", "declare", "causal", "refctl", "candidate", "assess"))
    ap.add_argument("--out-dir", required=True)
    for k in ("preflight", "declaration", "declaration-sha", "causal-dir", "refctl-dir", "candidate-dir", "visual-review"):
        ap.add_argument("--" + k)
    a = ap.parse_args()
    return {"preflight": stage_preflight, "declare": stage_declare, "causal": stage_causal, "refctl": stage_refctl,
            "candidate": stage_candidate, "assess": stage_assess}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
