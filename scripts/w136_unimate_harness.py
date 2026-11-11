#!/usr/bin/env python3
"""W136 (EXPERIMENTAL, opt-in): UniMate v3 source-bound left-arm generation on the authentic S1 BaseRig Action.

Stages (each refuses an existing output directory; no AnimCV run directory or lineage is ever written):

  preflight  rights / access / resource / identity gate + the NO-MODEL representation roundtrip:
             repo state, W135 COMPLETE, full SHA-256 of S1 clip, A, B, C, W135 D/E, W126V pin and records; pinned official
             UniMate code commit, HF model revision, checkpoint/config/stats SHA-256 (downloaded and verified), frozen
             google/flan-t5-base revision (ungated, downloaded and verified), licence texts and required clauses, scratch
             runtime, GPU/disk; byte copies + fresh Blender dumps/readbacks of A, B, C, D, E and the W126V file; then
             B -> export adapter (GLB, Action 156..216) -> official rig_preprocess (offline rule labels, review stop,
             label/facing gate, build with --save_clips) -> official feature construction -> decode -> Blender build of the
             decoded B keys -> fresh readback == B.  No model weights are loaded in this stage.
  declare    prospective declaration (identities, rights, window, mask, prompt, seed, sampler, mapping, gates, quality
             criteria, CASE rule, controls, instrument hashes) BEFORE any model output exists.
  controls   real-data fail-closed controls (labels, missing joint, facing, handedness, joint order, time shift, scaling,
             rest mismatch, non-left-arm drift, corrupt model/source, 24/20 fps, mask domain, channel/frame edits, pin
             overlap, degenerate 6D) incl. real Blender refusals.  Never runs the model.
  run        ONE real UniMate inference (official motion-editing path + the declared joint x time mask) -> decode ->
             U (model) and Bc (PROXY_B through the identical path) Actions + Upin / Urev -> fresh readbacks -> gates,
             generated-effect, quality and source-2D metrics -> renders, native-rate video, contact sheets.
  assess     frozen CASE rule over the run evidence + the agent's still-frame review file.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w136_unimate_harness.py <stage> --out-dir <dir> [...]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np                                                    # noqa: E402

import app.experimental_unimate as U                                  # noqa: E402
import app.experimental_elbow_prior as EP                             # noqa: E402
import app.experimental_motionbricks as M                             # noqa: E402
import w129_motionbricks_harness as H                                 # noqa: E402
import w135_elbow_prior_harness as H5                                 # noqa: E402

W136 = Path("C:/Users/dna10/animcv-w136")
W135R = Path("C:/Users/dna10/animcv-w135/results")
UM_ROOT = W136 / "ext" / "UniMate"
UM_PY = W136 / "env" / "um310" / "Scripts" / "python.exe"
UM_COMMIT = "b78c7807df7cf4231d4e67f8d0ece29c5e11025b"
HF_REPO = "Linzhan/UniMate"
HF_REV = "174fc755ec248d0c98f27759b7a95648e7f05905"
MODEL_NAME = "unimate_uniml3d_f60_v3"
CKPT_REL = f"{MODEL_NAME}/checkpoints/checkpoint_step_150000.pt"
CKPT_SHA = "f991ddd87262c251f4681d18297a6163ab4f139daa28c86d5ef0a66771df7bc6"
STATS_SHA = "0ff2b9ef3ea9a15b024ab0670e55f659478251501e5596f9011d5bca4cf67773"
MODEL_DIR = W136 / "model" / MODEL_NAME
T5_REPO = "google/flan-t5-base"
T5_REV = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
T5_FILES = ["config.json", "spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "generation_config.json", "model.safetensors"]
HF_HOME = W136 / "hf"
ASSET = "S1_PROXY_B"
SEED = 10                                   # the published v3 config's sampling.seed
CFG = 3.0                                   # the published v3 config's sampling.cfg_scale
N_EULER = 50                                # the official replacement sampler's default (inbetween_sample_ode)
EXPORT_STAGE = ROOT / "src" / "blender" / "experimental_unimate_export_stage.py"
INFER = ROOT / "scripts" / "w136_unimate_infer.py"
STAGE, ACTION_STAGE = H.STAGE, H.ACTION_STAGE
S1 = H.SOURCES["S1"]
INSTRUMENTS = ["src/app/experimental_unimate.py", "src/blender/experimental_unimate_export_stage.py",
               "scripts/w136_unimate_infer.py", "scripts/w136_unimate_harness.py", "tests/test_experimental_unimate.py",
               "src/app/experimental_motionbricks.py", "src/app/experimental_elbow_prior.py",
               "src/blender/experimental_motionbricks_stage.py", "src/blender/experimental_action_stage.py",
               "scripts/w129_motionbricks_harness.py", "scripts/w135_elbow_prior_harness.py"]
FROZEN = dict(H5.FROZEN)
FROZEN.update({
    "W135_D": (str(W135R / "run/D_S1.blend"), "2a5e441f75eeead09fc06242b93b595124b3da3a4e69c054e33e42f18e54cebe"),
    "W135_E": (str(W135R / "run/E_S1.blend"), "ac53a771ef5d3dfa80bee7df460fe0b1bd49873ff0950784bbc60a44bb1dc19f"),
    "W135_declaration": (str(W135R / "declare/declaration.json"), "0c69dde0e13fb909687be8d1d0cfd07953fa51f6f645c531ab788a0a41f767fe"),
    "W135_assessment": (str(W135R / "assess/assessment.json"), "b2336b8f4e35b3e8ef4abdffbe660f051e9d69bd483add7f5c3bf6665e399177"),
    "W135_run_result": (str(W135R / "run/run_result.json"), "4934fd071ebd53ce542388eb57b342afc001313a2d6b3f1147a689658ccbd7e0"),
    "W135_evidence_manifest": (str(W135R / "evidence_manifest.json"), "8543b5ecb95567da4a8cc204720254bc09c44004f1813fb5070aa7b4ba307c31"),
})
COPY_KEYS = {"A": "A", "B": "B", "C": "C", "D": "W135_D", "E": "W135_E", "W126V_accepted": "W126V_accepted"}
NAMES = {"A": "A original (no-op)", "B": "B slerp (no model)", "C": "C MotionBricks (W129)", "D": "D W135 C-swivel",
         "E": "E W135 neutral (=B)", "U": "U UniMate v3 (model)", "Bc": "Bc PROXY_B via U path"}
RIGHTS_REQUIRED = {
    "hf_license": ["UniMate pretrained checkpoints", "Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)",
                   "They are released for non-commercial research use.", "This license covers only the rights of the UniMate authors.",
                   "reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only"],
    "hf_readme": ["license: cc-by-nc-4.0", "Checkpoints: [CC BY-NC 4.0](LICENSE). Code: MIT.",
                  "| [`unimate_uniml3d_f60_v3`](unimate_uniml3d_f60_v3) | graph attention, AdaLN text |", "74.1M | **Recommended** |",
                  "Frozen `google/flan-t5-base`", "60 frames at 30 fps"],
    "code_license": ["MIT License", "Copyright (c) 2026 Linzhan Mou"],
    "t5_readme": ["license: apache-2.0"],
}
PROVENANCE_CLAUSES = ["Third parties allege that some of its animal assets come from commercial games",
                      "treat the provenance of the Truebones-trained models as unclear", "Adobe's Mixamo terms of use",
                      "some are non-commercial", "We make no warranty that the checkpoints, or motions generated with them, are free of third-party rights."]
HTTP_UA = "Mozilla/5.0 (animcv-w136-preflight)"

load, dump, new_dir, now, git, lf_sha = H.load, H.dump, H.new_dir, H.now, H.git, H.lf_sha


def blender(script, mode, log, **kw):
    return H.blender(script, mode, log, **kw)


def readback(blend, rep, log):
    return blender(ACTION_STAGE, "readback", log, blend=blend, report=rep, subframes="0.25,0.5,0.75")


def bdump(blend, rep, log):
    return blender(STAGE, "dump", log, blend=blend, report=rep)


def _frames(rb):
    return {f["blender_frame"]: f for f in rb["frames"]}


def _clean_env():
    env = dict(os.environ)
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN"):
        env.pop(k, None)
    env["HF_HOME"] = str(HF_HOME)
    env["PYTHONUTF8"] = "1"
    return env


def um(args, log, offline=True, timeout=3600, torch_legacy_load=False):
    """Run the scratch UniMate interpreter from the pinned checkout root.  ``torch_legacy_load`` (inference only):
    torch 2.7 (needed for sm_120) defaults ``torch.load(weights_only=True)``; the official code targets torch 2.5.1,
    whose default is False.  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 restores the official semantics for the hash-verified
    official checkpoint only."""
    env = _clean_env()
    if torch_legacy_load:
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    t = time.time()
    r = subprocess.run([str(UM_PY), *[str(x) for x in args]], cwd=UM_ROOT, capture_output=True, text=True, env=env,
                       timeout=timeout, encoding="utf-8", errors="replace")
    Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return {"rc": r.returncode, "seconds": round(time.time() - t, 2), "log": str(log)}


def _norm_ws(t):
    return re.sub(r"\s+", " ", t)


def _http(url, out):
    r = subprocess.run(["curl", "-sL", "--fail", "-A", HTTP_UA, "-o", str(out), url], capture_output=True, text=True)
    return r.returncode == 0 and Path(out).is_file()


def check_declaration(a):
    if not a.declaration:
        raise SystemExit("--declaration required")
    if M.sha256_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP INTEGRITY_FAILED: declaration changed")
    d = load(a.declaration)
    changed = [f for f, h in d["instrument_sha256_lf"].items() if lf_sha(f) != h]
    if changed:
        raise SystemExit(f"STOP DECLARATION_INVALID: instrument changed {changed}")
    return d


def frozen_hashes():
    out = {}
    for k, (p, h) in FROZEN.items():
        ok = Path(p).is_file()
        got = M.sha256_file(p) if ok else None
        out[k] = {"path": p, "expected": h, "sha256": got, "ok": bool(ok and got == h)}
    return out


def rig_parents(dump_report):
    return {b: r["parent"] for b, r in dump_report["rest"].items()}


def rig_heads(dump_report):
    return {b: r["head_local"] for b, r in dump_report["rest"].items()}


# ------------------------------------------------------------------------------------------------ representation ---
def _rc(rep):
    name, r = rep["steps"][-1]
    if r["rc"] != 0:
        tail = Path(r["log"]).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{name} rc={r['rc']}: {tail}")


def representation_gate(out, inp, dB, dA, rbB, rbA, rep):
    """NO-MODEL roundtrip: B -> export adapter -> official rig_preprocess -> features -> decode -> Blender == B."""
    g = {}
    exp = out / "export"
    exp.mkdir()
    glb = exp / f"{ASSET}.glb"
    rep["steps"].append(["export", blender(EXPORT_STAGE, "export", exp / "export.log", blend=inp / "B.blend", out=glb,
                                           frames=f"{U.EXPORT_FRAMES[0]},{U.EXPORT_FRAMES[1]}", report=exp / "export_report.json",
                                           expected_sha=FROZEN["B"][1])])
    er = load(exp / "export_report.json")
    g["export"] = {k: er[k] for k in ("world_matrix_max_abs_diff_anatomical", "frames", "adapter", "glb_sha256", "n_anatomical_bones_checked")}
    if er["world_matrix_max_abs_diff_anatomical"] > 1e-5 or er["frames"] != list(U.EXPORT_FRAMES):
        raise U.UniStop("REPRESENTATION_BLOCKED", "export adapter changed world poses or frames")
    rig = out / "rig"
    rep["steps"].append(["rig_preprocess_review", um(["-m", "data_process.rig_preprocess", "run", "--input", glb, "--output_dir", rig,
                                                      "--annotate", "rule", "--save_clips", "--keep_intermediate"],
                                                     out / "rig_preprocess_review.log")])
    _rc(rep)
    ann = load(rig / "annotation.json")
    names_e = [j["raw"] for j in ann["joints"]]
    labels = [j["label"] for j in ann["joints"]]
    g["annotation"] = {"n_joints": len(names_e), "n_flagged": ann["n_flagged"], "notes": ann["notes"], "face_pair": ann["face_pair"],
                       "sources": sorted({j["source"] for j in ann["joints"]}), "sha256": M.sha256_file(rig / "annotation.json"),
                       "labels": dict(zip(names_e, labels))}
    if ann["n_flagged"] != 0 or ann["notes"]:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"annotation flagged {ann['n_flagged']} / notes {ann['notes']}")
    for b, lab in U.EXPECTED_LABELS.items():
        if dict(zip(names_e, labels)).get(b) != lab:
            raise U.UniStop("LABEL_MISMATCH", f"{b}")
    if (ann["face_pair"]["right"], ann["face_pair"]["left"]) != U.FACE_PAIR or ann["face_pair"]["body_axis"]:
        raise U.UniStop("FACING_PAIR", str(ann["face_pair"]))
    ann_copy = out / "annotation_reviewed.json"
    shutil.copyfile(rig / "annotation.json", ann_copy)
    rep["steps"].append(["rig_preprocess_build", um(["-m", "data_process.rig_preprocess", "run", "--input", glb, "--output_dir", rig,
                                                     "--profile", "general", "--save_clips", "--keep_intermediate",
                                                     "--annotation", rig / "annotation.json"], out / "rig_preprocess_build.log")])
    _rc(rep)
    g["annotation_unchanged_by_build"] = M.sha256_file(rig / "annotation.json") == M.sha256_file(ann_copy)
    summ = load(rig / "summary.json")
    g["summary"] = {k: summ.get(k) for k in ("name", "profile", "n_joints", "n_clips", "clips", "model_joint_limit", "face_pair",
                                             "dropped_translation", "notes", "annotate", "stats_dataset")}
    if summ["name"] != ASSET or summ["n_joints"] != U.N_JOINTS or summ["n_clips"] != 1 or not summ["model_joint_limit"]["fits"] or summ["notes"]:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"summary {g['summary']}")
    clip = summ["clips"][0]
    g["clip"] = clip
    g["cond_sha256"] = M.sha256_file(rig / "cond.npy")
    g["clip_sha256"] = M.sha256_file(rig / "motions" / f"{clip}.npz")
    rep["steps"].append(["features", um([INFER, "features", "--out-dir", out / "features", "--asset-dir", rig, "--name", ASSET,
                                         "--clip", clip], out / "features.log")])
    _rc(rep)
    z = np.load(out / "features" / "features.npz")
    names = [str(n) for n in z["names"]]
    par = z["parents"]
    U.check_fps(int(z["clip_fps"]))
    if int(z["n_clip_frames"]) != U.T_MODEL + 1 or z["feats"].shape != (U.T_MODEL, U.N_JOINTS, U.FEAT):
        raise U.UniStop("REPRESENTATION_BLOCKED", f"clip frames {int(z['n_clip_frames'])} / feats {z['feats'].shape}")
    fj = load(out / "features" / "features.json")["face_joint_idxs"]
    if fj.get("body_axis"):
        raise U.UniStop("FACING_PAIR", "body axis pair")
    g["cond_face_pair"] = [names[fj["r_hip"]], names[fj["l_hip"]]]
    U.validate_asset(names, par, z["clean_joint_names"], tuple(g["cond_face_pair"]), rig_parents(dB),
                     raw_annotation_labels=[dict(zip(names_e, labels))[n] for n in names])
    g["facing_cos"] = U.check_facing(z["tpos_first_frame"], names)
    g["scale"] = U.check_scale(z["offsets"], names, par, rig_heads(dB))
    dec = {b: z[f"decode_{b}"] for b in U.EDIT_CHANNELS}
    g["time_identity"] = U.time_identity(dec, dB["stored"])
    g["own_vs_official_decode"] = load(out / "features" / "features.json")["own_vs_official_decode_max"]
    if max(g["own_vs_official_decode"].values()) > 1e-9:
        raise U.UniStop("REPRESENTATION_BLOCKED", "own decoder differs from the official decode")
    ctx = U.decode_basis(z["feats"], names, par, z["tpos_local_rotations"], joints=("spine_01", "thigh_l", "thigh_r", "calf_l", "calf_r"))
    g["context_drift"] = {b: U.roundtrip_error(ctx, dB["stored"], b) for b in ctx}
    if max(g["context_drift"].values()) > U.QUAT_TOL:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"non-left-arm context drift {g['context_drift']}")
    g["mask"] = U.validate_mask(U.build_keep_mask(names, par, 71), names, par)
    # Blender roundtrip: decoded PROXY_B keys through the authorised import path == B
    vals = U.import_values(dec, dA["stored"])
    spec = {"out_blend": str(out / "R0_S1.blend"), "interior_action_frames": list(U.INTERIOR), "values_wxyz": vals,
            "branch": "W136 R0 roundtrip", "action_name": "AnimCV_EXP W136 R0 roundtrip S1", "expected_blend_sha256": FROZEN["A"][1],
            "note": "W136 no-model roundtrip: W129 B -> export -> UniMate rig_preprocess -> features -> decode -> import"}
    dump(out / "build_R0_spec.json", spec)
    rep["steps"].append(["build_R0", blender(STAGE, "build_variant", out / "build_R0.log", blend=inp / "A.blend",
                                             spec=out / "build_R0_spec.json", report=out / "build_R0_report.json")])
    rep["steps"].append(["readback_R0", readback(out / "R0_S1.blend", out / "readback_R0.json", out / "readback_R0.log")])
    rr = load(out / "readback_R0.json")
    mx = 0.0
    for p, q in zip(rr["frames"], rbB["frames"]):
        for b in q["world"]:
            for k in ("head", "tail"):
                mx = max(mx, float(np.abs(np.subtract(p["world"][b][k], q["world"][b][k])).max()))
    pres = M.preservation(rr, rbA, U.INTERIOR)
    g["blender_roundtrip"] = {"max_world_diff_vs_B_m": mx, "nonarm_changed": pres["nonarm_changed"][:5],
                              "outside_changed": pres["outside_changed"][:5], "sub_outside_changed": pres["sub_outside_changed"][:5]}
    if mx > 1e-5 or pres["nonarm_changed"] or pres["outside_changed"] or pres["sub_outside_changed"]:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"Blender roundtrip {g['blender_roundtrip']}")
    g["asset_dir"] = str(rig)
    return g


# ------------------------------------------------------------------------------------------------ preflight ---
def stage_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"stage": "preflight", "time": now(), "head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
           "origin_head": git("rev-parse", "origin/arch/single_frame_first"), "steps": [],
           "w135_complete": "AUTOMATION_BATCH_STATUS: COMPLETE" in (ROOT / "docs/135_WORKLOG_MOTIONBRICKS_ELBOW_SWIVEL_PRIOR_SOURCE_CONSTRAINED_S1.md").read_text(encoding="utf-8"),
           "w136_worklog_absent": not any((ROOT / "docs").glob("136_*"))}
    rep["frozen"] = frozen_hashes()
    stop = None
    # ---- code identity ----
    code = {"repo": "https://github.com/Friedrich-M/UniMate", "pinned_commit": UM_COMMIT,
            "head": subprocess.run(["git", "-C", str(UM_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "dirty": subprocess.run(["git", "-C", str(UM_ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip(),
            "untracked_non_cache": [ln for ln in subprocess.run(["git", "-C", str(UM_ROOT), "status", "--porcelain"], capture_output=True,
                                                                text=True).stdout.splitlines() if ln.startswith("??") and "__pycache__" not in ln],
            "files_sha256": {f: M.sha256_file(UM_ROOT / f) for f in ("LICENSE", "README.md", "unimate/inference/sample.py",
                                                                     "unimate/inference/motion_inbetweening.py", "unimate/inference/motion_editing.py",
                                                                     "unimate/inference/generate.py", "unimate/utils/motion_utils.py",
                                                                     "data_process/rig_preprocess/pipeline.py", "data_process/utils/blender_export.py",
                                                                     "data_process/rig_preprocess/README.md")}}
    code["ok"] = code["head"] == UM_COMMIT and code["dirty"] == "" and not code["untracked_non_cache"]
    rep["code"] = code
    # ---- model identity + download ----
    meta = new_dir(out / "meta")
    api_ok = _http(f"https://huggingface.co/api/models/{HF_REPO}/revision/{HF_REV}?blobs=true", meta / "unimate_api.json")
    t5_ok = _http(f"https://huggingface.co/api/models/{T5_REPO}/revision/{T5_REV}?blobs=true", meta / "t5_api.json")
    mi = load(meta / "unimate_api.json") if api_ok else {}
    ti = load(meta / "t5_api.json") if t5_ok else {}
    sib = {s["rfilename"]: s for s in mi.get("siblings", [])}
    tsib = {s["rfilename"]: s for s in ti.get("siblings", [])}
    for f, dst in (("README.md", meta / "hf_README.md"), ("LICENSE", meta / "hf_LICENSE")):
        _http(f"https://huggingface.co/{HF_REPO}/resolve/{HF_REV}/{f}", dst)
    _http(f"https://huggingface.co/{T5_REPO}/resolve/{T5_REV}/README.md", meta / "t5_README.md")
    model = {"repo": HF_REPO, "revision": HF_REV, "api_sha": mi.get("sha"), "gated": mi.get("gated"), "private": mi.get("private"),
             "card_license": (mi.get("cardData") or {}).get("license"), "checkpoint": CKPT_REL,
             "checkpoint_lfs_sha256": (sib.get(CKPT_REL, {}).get("lfs") or {}).get("sha256"),
             "checkpoint_size": sib.get(CKPT_REL, {}).get("size"),
             "stats_lfs_sha256": (sib.get(f"{MODEL_NAME}/dataset_stats.npy", {}).get("lfs") or {}).get("sha256")}
    (MODEL_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    t = time.time()
    dl, reused = {}, {}
    for f, want in ((f"{MODEL_NAME}/config.json", None), (f"{MODEL_NAME}/dataset_stats.npy", STATS_SHA), (CKPT_REL, CKPT_SHA)):
        dst = W136 / "model" / f
        fresh = meta / ("fetched_" + Path(f).name) if (want is None or not dst.exists()) else None
        if dst.exists() and want is not None and M.sha256_file(dst) == want:
            dl[f], reused[f] = True, True                # an earlier preflight's verified download (hash-checked, not trusted)
            continue
        if dst.exists() and want is None:
            dl[f] = _http(f"https://huggingface.co/{HF_REPO}/resolve/{HF_REV}/{f}", fresh) and fresh.read_bytes() == dst.read_bytes()
            reused[f] = True
            continue
        if dst.exists():
            raise SystemExit(f"STOP INTEGRITY_FAILED: {dst} exists with a different hash")
        dl[f] = _http(f"https://huggingface.co/{HF_REPO}/resolve/{HF_REV}/{f}", dst)
        reused[f] = False
    model["reused_verified_download"] = reused
    model["download_seconds"] = round(time.time() - t, 1)
    model["downloaded"] = dl
    model["local_sha256"] = {f: M.sha256_file(W136 / "model" / f) for f in dl if dl[f]}
    cfg = load(MODEL_DIR / "config.json") if dl[f"{MODEL_NAME}/config.json"] else {}
    model["config"] = {"diff_model": cfg.get("training", {}).get("diff_model"), "max_motion_length": cfg.get("dataset", {}).get("max_motion_length"),
                       "max_joints": cfg.get("dataset", {}).get("max_joints"), "feature_len": cfg.get("dataset", {}).get("feature_len"),
                       "text_encoder": cfg.get("model", {}).get("text_encoder_version"), "cfg_scale": cfg.get("sampling", {}).get("cfg_scale"),
                       "seed": cfg.get("sampling", {}).get("seed"), "attention": cfg.get("model", {}).get("attention"),
                       "text_cond": cfg.get("model", {}).get("text_cond"), "num_layers": cfg.get("model", {}).get("num_layers")}
    model["ok"] = (api_ok and mi.get("sha") == HF_REV and mi.get("gated") is False and model["checkpoint_lfs_sha256"] == CKPT_SHA
                   and model["local_sha256"].get(CKPT_REL) == CKPT_SHA and model["local_sha256"].get(f"{MODEL_NAME}/dataset_stats.npy") == STATS_SHA
                   and model["stats_lfs_sha256"] == STATS_SHA and model["config"]["text_encoder"] == T5_REPO
                   and model["config"]["cfg_scale"] == CFG and model["config"]["seed"] == SEED and model["config"]["max_motion_length"] == U.T_MODEL
                   and model["config"]["feature_len"] == U.FEAT and model["config"]["diff_model"] == "flow")
    rep["model"] = model
    # ---- text encoder (ungated, pinned revision, offline afterwards) ----
    t5 = {"repo": T5_REPO, "revision": T5_REV, "api_sha": ti.get("sha"), "gated": ti.get("gated"),
          "card_license": (ti.get("cardData") or {}).get("license"),
          "safetensors_lfs_sha256": (tsib.get("model.safetensors", {}).get("lfs") or {}).get("sha256")}
    code_snap = ("import os,json,hashlib;from huggingface_hub import snapshot_download;"
                 f"p=snapshot_download('{T5_REPO}',revision='main',allow_patterns={T5_FILES!r});"
                 "h=hashlib.sha256(open(os.path.join(p,'model.safetensors'),'rb').read()).hexdigest();print(json.dumps({'path':p,'sha':h}))")
    rep["steps"].append(["t5_snapshot", um(["-c", code_snap], out / "t5_snapshot.log", offline=False)])
    try:
        snap = json.loads([ln for ln in (out / "t5_snapshot.log").read_text(encoding="utf-8").splitlines() if ln.startswith("{")][-1])
    except IndexError:
        snap = {}
    t5["local_path"], t5["local_safetensors_sha256"] = snap.get("path"), snap.get("sha")
    t5["snapshot_commit"] = Path(snap["path"]).name if snap.get("path") else None
    t5["ok"] = bool(t5_ok and ti.get("sha") == T5_REV and ti.get("gated") is False and t5["card_license"] == "apache-2.0"
                    and t5["snapshot_commit"] == T5_REV and snap.get("sha") and snap.get("sha") == t5["safetensors_lfs_sha256"])
    rep["text_encoder"] = t5
    # ---- rights ----
    txt = {"hf_license": _norm_ws((meta / "hf_LICENSE").read_text(encoding="utf-8")) if (meta / "hf_LICENSE").is_file() else "",
           "hf_readme": _norm_ws((meta / "hf_README.md").read_text(encoding="utf-8")) if (meta / "hf_README.md").is_file() else "",
           "code_license": _norm_ws((UM_ROOT / "LICENSE").read_text(encoding="utf-8")),
           "t5_readme": _norm_ws((meta / "t5_README.md").read_text(encoding="utf-8")) if (meta / "t5_README.md").is_file() else ""}
    present = {k: {c: _norm_ws(c) in txt[k] for c in cl} for k, cl in RIGHTS_REQUIRED.items()}
    prov = {c: _norm_ws(c) in txt["hf_readme"] for c in PROVENANCE_CLAUSES}
    rights = {"required_clauses_present": present, "provenance_clauses_present": prov,
              "files_sha256": {p.name: M.sha256_file(p) for p in meta.iterdir() if p.is_file()},
              "intended_use": "private, local, non-commercial technical research evaluation on one machine; no redistribution of weights, "
                              "no public distribution of model outputs, no training/finetuning, no production or commercial use",
              "external_services": "none (offline rule labels; no LLM API; HF used only for public, ungated downloads)"}
    rights["permitted"] = all(all(v.values()) for v in present.values())
    rights["determination"] = ("PERMITTED_NONCOMMERCIAL_RESEARCH_EVALUATION" if rights["permitted"] else "RIGHTS_UNRESOLVED")
    rights["commercial"] = "NOT GRANTED (CC BY-NC 4.0 weights); training-data provenance (Truebones ZOO allegations, Mixamo terms, " \
                           "Objaverse-XL per-object licences) explicitly unresolved by the licensor"
    rep["rights"] = rights
    # ---- runtime / resources ----
    env_code = ("import sys,json,torch,numpy,transformers,bpy;print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,"
                "'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'arch':torch.cuda.get_arch_list(),"
                "'device':torch.cuda.get_device_name(0),'numpy':numpy.__version__,'transformers':transformers.__version__,'bpy':bpy.app.version_string}))")
    rep["steps"].append(["um_env", um(["-c", env_code], out / "um_env.log")])
    try:
        envr = json.loads([ln for ln in (out / "um_env.log").read_text(encoding="utf-8").splitlines() if ln.startswith("{")][-1])
    except IndexError:
        envr = {}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    du = shutil.disk_usage(str(W136))
    rep["runtime"] = {"um_env": envr, "nvidia_smi": smi, "disk_free_gb": round(du.free / 1e9, 1),
                      "freeze_sha256": M.sha256_file(W136 / "env" / "freeze.txt") if (W136 / "env" / "freeze.txt").is_file() else None,
                      "estimate": "denoiser 74.1M fp32 (~0.3 GB) + flan-t5-base encoder (~0.25 GB fp32) + activations for 1 x 71 joints x 60 "
                                  "frames: < 2 GB VRAM; disk: checkpoint 1.19 GB + T5 1 GB + env ~6 GB",
                      "ok": bool(envr.get("available") and "sm_120" in envr.get("arch", []) and du.free > 20e9)}
    rep["blender"] = subprocess.run([H.BLENDER, "-b", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    pre_ok = (rep["instruction_blob"] == "b7fd915af1d8d9d52d51454b1d27cf969df5b1c9" and rep["w135_complete"] and rep["w136_worklog_absent"]
              and all(v["ok"] for v in rep["frozen"].values()) and bool(rep["blender"]))
    if not rights["permitted"]:
        stop = "RIGHTS_UNRESOLVED"
    elif not (model["ok"] and t5["ok"] and code["ok"]):
        stop = "ACCESS_BLOCKED"
    elif not rep["runtime"]["ok"]:
        stop = "RESOURCE_BLOCKED"
    elif not pre_ok:
        stop = "DATA_BLOCKED"
    if stop is None:
        inp = new_dir(out / "inputs")
        for k, fk in COPY_KEYS.items():
            shutil.copyfile(FROZEN[fk][0], inp / f"{k}.blend")
            if M.sha256_file(inp / f"{k}.blend") != FROZEN[fk][1]:
                stop = "DATA_BLOCKED"
        for k in ("A", "B", "C", "D", "E"):
            rep["steps"].append([f"dump_{k}", bdump(inp / f"{k}.blend", out / f"dump_{k}.json", out / f"dump_{k}.log")])
            rep["steps"].append([f"readback_{k}", readback(inp / f"{k}.blend", out / f"readback_{k}.json", out / f"readback_{k}.log")])
        rep["steps"].append(["readback_Pin", readback(inp / "W126V_accepted.blend", out / "readback_Pin.json", out / "readback_Pin.log")])
        x = {}
        for k in ("A", "B", "C"):
            w = load(H5.W129R / "run/S1" / f"readback_{k}.json")
            f = load(out / f"readback_{k}.json")
            x[k] = all(p["world"] == q["world"] for p, q in zip(w["frames"], f["frames"]))
        rep["fresh_world_bit_equal_to_W129"] = x
        dA, dB = load(out / "dump_A.json"), load(out / "dump_B.json")
        try:
            EP.check_dump(dB)
            rep["representation"] = representation_gate(out, inp, dB, dA, load(out / "readback_B.json"), load(out / "readback_A.json"), rep)
        except U.UniStop as e:
            rep["representation"] = {"STOP": {"code": e.code, "detail": e.detail}}
            stop = "REPRESENTATION_BLOCKED" if e.code not in ("NOT_30_FPS",) else "DATA_BLOCKED"
        except EP.PriorStop as e:
            rep["representation"] = {"STOP": {"code": e.code, "detail": e.detail}}
            stop = "DATA_BLOCKED"
        except (RuntimeError, OSError, KeyError, ValueError, IndexError) as e:
            rep["representation"] = {"STOP": {"code": "REPRESENTATION_BLOCKED", "detail": f"{type(e).__name__}: {e}"}}
            stop = "REPRESENTATION_BLOCKED"
    rep["stop"] = stop
    rep["status"] = "OK" if stop is None else "DATA_BLOCKED"
    dump(out / "preflight.json", rep)
    print(json.dumps({"status": rep["status"], "stop": stop, "rights": rights["determination"], "model_ok": model["ok"], "t5_ok": t5["ok"],
                      "code_ok": code["ok"], "runtime_ok": rep["runtime"]["ok"], "frozen_ok": all(v["ok"] for v in rep["frozen"].values()),
                      "representation": {k: v for k, v in rep.get("representation", {}).items() if k in ("STOP", "time_identity", "context_drift",
                                                                                                           "blender_roundtrip", "facing_cos", "mask")}},
                     indent=1, default=str))
    return 0 if stop is None else 3


# ------------------------------------------------------------------------------------------------ declaration ---
MECH_GATES = {
    "M0": "rights/access/resource/identity preflight OK (pinned code, model revision, checkpoint + stats SHA-256, ungated pinned flan-t5-base, licence clauses, sm_120 runtime, every frozen SHA-256)",
    "M1": "representation: the no-model roundtrip passed in preflight (export adapter world-equal <= 1e-5; 50 joints, labels/side/hierarchy/facing pair; facing cos >= 0.95; uniform scale; clip 61 frames at 30 fps; decode == B keys <= 1e-5 at shift 0 and >= 100x worse at +-1; own == official decode; non-left-arm context drift <= 1e-5; decoded-B Blender build == B <= 1e-5 m) AND in the run the official PROXY_B tensor decodes to B's keys <= 1e-5",
    "M2": "Action: U and Bc each a separate .blend with exactly 1 separately named Action; 32 CONSTANT F-curves keyed 1..240; scene 30 fps, 1..240",
    "M3": "footprint: changed stored keys vs A only upperarm_l/lowerarm_l rotation at Action 172..199; every non-left-arm world point and the object matrix bit-equal to A on all 240 integer frames and all 717 subframes; left arm bit-equal outside 172..199; no subframe change outside [172,200)",
    "M4": "conditioning: exactly one generate_samples call; kept (mask True) entries of the raw sample equal the PROXY_B tensor <= 1e-4 (denormalised); the free set is exactly the declared support (validate_mask)",
    "M5": "rig validity: upperarm/lowerarm world length range <= 0.1 mm over 171..200 for U; all imported quaternions finite and unit; no decoder refusal",
    "M6": "hard pin: U composed onto the W126V-accepted .blend keeps every Action 132..140 hand_l residual equal to W126V's and <= 1e-6 m; pin frames bit-equal; composed footprint as M3 relative to W126V",
    "M7": "reversibility: U with A's interior keys written back (Urev) is world-bit-equal to A on every integer frame and subframe",
    "M8": "matched no-generator control: Bc (PROXY_B through the identical decode/import path) equals B in world points within 1e-5 m on every frame",
    "M9": "integrity: every frozen SHA-256 (S1 clip, A, B, C, Cpin, W126V, W126 refs/proposal, W129 and W135 records, W135 D/E) unchanged after the run",
    "M10": "negative controls (controls stage) all fail closed",
    "M11": "native-rate video: 240 frames written and decoded at 30.000 fps",
}
EFFECT_RULE = ("GENERATED_EFFECT iff max over Action 172..199 of max(|elbow_U - elbow_Bc|, |wrist_U - wrist_Bc|) (world, fresh readback) "
               ">= 10 mm; otherwise NO_GENERATED_EFFECT")
QUALITY = {
    "Q1": "boundary continuity: U arm-local (clavicle_l frame) wrist AND elbow entry (171->172) and exit (199->200) steps each <= 2 x B's corresponding step + 10 mm",
    "Q2": "no reversal flicks (consecutive steps both > 20 mm with negative dot) of the arm-local wrist or elbow in steps 171->172 .. 199->200",
    "Q3": "interior smoothness: U arm-local wrist peak step (171->172 .. 199->200) <= 2 x B's peak + 10 mm",
    "Q4": "rig plausibility: no interior frame near-straight (> 175 deg) and no interior frame bending opposite to A's dominant hinge normal (W129 quality)",
    "Q5": "source-2D fidelity: affine camera fitted ONLY on unedited torso joints (both shoulders, both hips; A readback, Action 156..215, valid detections) has RMS <= 20 px, AND U's interior mean 2D left-wrist error to the MMPose detection <= B's (detections on the self-occluded far arm are labelled hypotheses, not observed truth)",
    "Q6": "press intent: U's world wrist net rise (Z at 199 minus Z at 172) >= 0.5 x B's net rise (B's rise > 0)",
    "Q7": "agent still-frame review: U judged more natural AND more faithful to the source overhead press than B, and U not named for a flick, wrong elbow or detour",
    "Q8": "native-rate perceptual review of the 30 fps video performed (this agent cannot perceive playback: declared NOT REVIEWED in advance, so Q8 cannot pass in this run)",
}
UNUSABLE_STEP_MM = 100.0
CASE_RULE = {
    "DATA_BLOCKED": "preflight not OK: RIGHTS_UNRESOLVED / ACCESS_BLOCKED / RESOURCE_BLOCKED / DATA_BLOCKED / REPRESENTATION_BLOCKED; or CONDITIONING_BLOCKED (the official sampler cannot hold the declared context: mask refused before any sample)",
    "D": "declaration/instrument changed or model output preceded the declaration; a negative control does not fail closed (M10); integrity (M9) fails; the matched control Bc != B (M8)",
    "C": "real inference executed but: decoder/projection refusal (RIG_PROJECTION_FAILED), conditioning fidelity M4 fails, any of M1-run, M2, M3, M5, M6, M7, M11 fails, or unusable continuity (any U arm-local wrist or elbow step in 171->172 .. 199->200 > 100 mm)",
    "B": "a valid source-conditioned model Action exists (all mechanical gates pass) but NO_GENERATED_EFFECT or any of Q1-Q8 fails (improvement absent, unproven or review incomplete)",
    "A": "all mechanical gates pass AND GENERATED_EFFECT AND Q1-Q8 pass",
    "precedence": "DATA_BLOCKED > D > C > B > A; a mechanical PASS, synthetic PASS or a video file alone never yields A",
}


def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit(f"preflight not OK: DATA_BLOCKED {pf.get('stop')}")
    pdir = Path(a.preflight).parent
    model_outputs = sorted(str(p) for p in (W136 / "results").rglob("raw_model_io.npz")) + sorted(str(p) for p in (W136 / "results").rglob("U_S1.blend"))
    r = pf["representation"]
    d = {"schema": "animcv_exp_w136_declaration_v1", "time": now(), "outcomes_inspected": False, "model_output_exists": bool(model_outputs),
         "batch": 136, "starting_commit": "b641905b475ebed63c49fa6c59fe6b40b82d44f0",
         "instruction_blob": "b7fd915af1d8d9d52d51454b1d27cf969df5b1c9", "repo_head_at_declaration": git("rev-parse", "HEAD"),
         "preflight_sha256": M.sha256_file(a.preflight),
         "bindings": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in pf["frozen"].items()},
         "fresh_dump_sha256": {k: M.sha256_file(pdir / f"dump_{k}.json") for k in "ABCDE"},
         "identities": {"code": {"repo": pf["code"]["repo"], "commit": UM_COMMIT, "licence": "MIT", "files_sha256": pf["code"]["files_sha256"]},
                        "model": {"repo": HF_REPO, "revision": HF_REV, "checkpoint": CKPT_REL, "checkpoint_sha256": CKPT_SHA,
                                  "config_sha256": pf["model"]["local_sha256"][f"{MODEL_NAME}/config.json"], "stats_sha256": STATS_SHA,
                                  "card": "unimate_uniml3d_f60_v3, graph attention + AdaLN text, 74.1M denoiser parameters, 150k steps, UniML3D d0f19d9",
                                  "licence": "CC BY-NC 4.0 (weights)"},
                        "text_encoder": {"repo": T5_REPO, "revision": T5_REV, "safetensors_sha256": pf["text_encoder"]["local_safetensors_sha256"],
                                         "licence": "apache-2.0", "gated": False, "frozen": True},
                        "runtime": pf["runtime"]["um_env"], "rig_preprocess_inputs": {"glb_sha256": r["export"]["glb_sha256"],
                                                                                     "annotation_sha256": r["annotation"]["sha256"],
                                                                                     "cond_sha256": r["cond_sha256"], "clip": r["clip"],
                                                                                     "clip_sha256": r["clip_sha256"]}},
         "rights": {"determination": pf["rights"]["determination"], "intended_use": pf["rights"]["intended_use"],
                    "commercial": pf["rights"]["commercial"], "files_sha256": pf["rights"]["files_sha256"],
                    "note": "documentary reading by the agent, not legal advice and not a human licensing decision"},
         "time_rule": "native 30 fps; Action = SOURCE - 78; model window = Action 156..215 (SOURCE 234..293), feature frame i <-> Action 156+i; "
                      "the export holds poses 156..216 (velocity of frame 215 needs pose 216); footprint Action 172..199 = SOURCE 250..277 = "
                      "window frames 16..43; frames 171 and 200 frozen",
         "proxy": {"label": "PROXY_B", "what": "the frozen W129 B Action (endpoint slerp of A's 171/200 left-arm keys; A everywhere else) over the window, "
                   "exported by the declared adapter and preprocessed by the official rig_preprocess; UniMate files call it 'gt' -- it is a "
                   "GT-free animation hypothesis, never measured 3D ground truth and never a video observation",
                   "conditioning_content": "the model sees PROXY_B everywhere except the generated support; B's interior left arm is NOT visible to the "
                                           "model (masked, incl. the 171->172 velocity of the generated joints)"},
         "export_adapter": {"pelvis_root": "one static unskinned bone 'pelvis' at the thigh-head midpoint, parent of spine_01/thigh_l/thigh_r (keep offset)",
                            "object_rotation": "armature-object rotation keys moved onto pelvis (Basis = Rest^-1 R_obj Rest, rotation+location); object set to identity rotation, scale 0.01 kept",
                            "check": f"anatomical world matrices unchanged on all 240 frames (measured {r['export']['world_matrix_max_abs_diff_anatomical']:.2e}, refuse > 1e-5)",
                            "helpers": "interaction / center_of_mass (zero-weight helper roots) untouched -> pruned/merged by the official exporter",
                            "why": "without the pelvis the official exporter keeps only the heaviest root subtree (legs dropped); without the bake the sampled 'rest' "
                                   "carries the frame-1 source rotation (tilted T-pose). Both were found on input-only roundtrips before declaration."},
         "rig_preprocess": {"command": "python -m data_process.rig_preprocess run --input S1_PROXY_B.glb --annotate rule --save_clips --keep_intermediate "
                                       "(review stop) ; then --profile general --save_clips --keep_intermediate --annotation <unchanged reviewed file>",
                            "labels": "offline deterministic rule labels; 0 of 50 flagged; reviewed by the agent (annotation + preview) and gated mechanically; "
                                      "no LLM / external API", "face_pair": list(U.FACE_PAIR), "facing_cos": r["facing_cos"]},
         "mapping": {"joints": "50 = 49 anatomical BaseRig bones (same names, BaseRig hierarchy) + pelvis; joint order = the official cond order (BFS)",
                     "coordinates": "UniMate canonical Y-up, facing +Z, scale to diameter 2 (official); non-root local rotations are frame-independent",
                     "decode": "bvh(upperarm_l) = 6D slot of lowerarm_l, bvh(lowerarm_l) = 6D slot of hand_l; L_j = cum_p^-1 new_j cum_p tq_j; "
                               "Basis_j = tq_j^-1 L_j (tq = cond tpos_local_rotations = BaseRig Rest_rel, verified)",
                     "scale": r["scale"], "time_identity": r["time_identity"], "context_drift": r["context_drift"],
                     "own_vs_official_decode": r["own_vs_official_decode"]},
         "observations": {"source_2d": "MMPose COCO-17 detections of the authentic S1 video (W126 observations_predictions.npz): left elbow/wrist "
                                       "detections exist with confidence ~0.4-0.6 on a SELF-OCCLUDED far arm -> recorded as detector HYPOTHESES",
                          "unknown": "the true 3D motion of the occluded left arm (no ground truth exists); the right arm is unkeyed at rig rest in A/B",
                          "used_for": "Q5 only (2D comparison after a torso-only affine camera fit); never as a model input"},
         "inference": {"path": "official unimate.inference.sample.main --motion_edit (asset by directory, test case {'S1_PROXY_B-<clip id>': prompt}, "
                               "--gt_start_frame 0, --num_repetitions 1, --batch_size 1, --only_save_motion)",
                       "mask": "scratch hook replaces the official (B,J,1,1) joint mask by a (B,J,12,T) joint x channel x time mask: generated = strict "
                               "descendants of upperarm_l at window frames 16..43 (all 12 channels) + their velocity channels at frame 15; the official "
                               "sampler docstring: 'or any combination as long as the mask broadcasts to (B, J, D, T)'; kept joints are named via "
                               "--keep_joints (raw names); the hook refuses unless the official joint mask equals the complement of the generated joints",
                       "generated_joints": r["mask"]["generated_joints"], "n_free_entries": r["mask"]["n_free"],
                       "prompt": U.PROMPT, "seed": SEED, "cfg_scale": CFG, "sampler": f"fixed-step Euler replacement ODE, {N_EULER} steps, t in [0,1] (official default)",
                       "checkpoint_weights": "EMA weights as loaded by the official _load_checkpoint", "one_run": True,
                       "runtime_compat": "torch 2.7.1+cu128 instead of the pinned 2.5.1+cu124 (no sm_120 kernels); TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "
                                         "restores torch.load's 2.5 default for the hash-verified official checkpoint; HF/transformers offline",
                       "no": "no seed/prompt/checkpoint/cfg/step sweep, no training/finetuning, no post-hoc smoothing/IK/threshold change"},
         "import": {"channels": list(U.EDIT_CHANNELS), "frames": list(U.INTERIOR), "values": "decoded U basis quaternions, unit-norm, sign-canonical to A's key",
                    "builder": "unchanged W129 build_variant (refuses any other channel/frame), new opt-in Action from a byte copy of A",
                    "not_imported": "model root/world/torso/right-arm/legs/hand/finger output, and every frame outside 172..199"},
         "control": "Bc = the official PROXY_B tensor saved by the same run, decoded and imported through the identical path (expected == B, M8)",
         "historical": "A, B, C (W129), D and E (W135) unchanged, read-only comparisons",
         "domain": {"channels": list(U.EDIT_CHANNELS), "frames": list(U.INTERIOR), "pin_action": [U.PIN_ACTION[0], U.PIN_ACTION[-1]],
                    "pin_target_m": [-0.0927, 0.2199, 1.4506], "pin_tolerance_m": 0.001},
         "mechanical_gates": MECH_GATES, "generated_effect_rule": EFFECT_RULE, "quality_criteria": QUALITY,
         "unusable_step_mm": UNUSABLE_STEP_MM, "case_rule": CASE_RULE,
         "metrics": ["arm-local wrist (W129 armlocal_quality) and elbow (W135 elbow_metrics) steps, flicks, jerk, swivel",
                     "world wrist path (W129 quality: deviation from A, progress, hinge)", "U vs Bc / B / D differences",
                     "world shoulder motion (parent-induced, identical in all branches)", "source-2D torso-camera comparison"],
         "controls": ["mismatched joint labels (left/right forearm swap)", "missing left-arm joint", "wrong facing pair", "mirrored T-pose (handedness)",
                      "joint order permutation", "shifted frame indices (+-1)", "non-uniform scaling", "rest/armature mismatch (perturbed T-pose rotation)",
                      "non-left-arm drift (perturbed spine slot)", "corrupt checkpoint copy", "corrupt S1 clip copy", "24 fps and 20 fps",
                      "mask freeing spine_01 / right arm / frame 170 / frame 200 / nothing / keeping a support entry", "channel spine_01, frames 171/200 (module + real Blender refusals)",
                      "pin-overlap request", "degenerate and non-finite 6D", "export of a non-B source (identity refusal, real Blender)"],
         "review_protocol": "UNBLINDED agent review of fixed-time contact sheets (entry 170-175, middle 183-188, exit 196-201, overview) with source crop, "
                            "A, B, C, D, E, U; still frames only; written to a review JSON before assess. Native-rate playback is NOT perceived by this "
                            "agent (Q8 NOT REVIEWED). No human/animator review.",
         "stop_conditions": ["RIGHTS_UNRESOLVED", "ACCESS_BLOCKED", "RESOURCE_BLOCKED", "DATA_BLOCKED", "REPRESENTATION_BLOCKED", "CONDITIONING_BLOCKED",
                             "RIG_PROJECTION_FAILED", "INTEGRITY_FAILED", "DECLARATION_INVALID", "no tuning or second run after any model output"],
         "pre_declaration_disclosures": [
             "export adapter (pelvis root + object-rotation bake) chosen after input-only roundtrips of B showed the official exporter drops the legs and samples a tilted rest",
             "check_scale tolerance set to 1e-3 after the real (input-only) offsets showed a 1.7e-4 float32 spread on 3-4 cm finger bones",
             "window 156..215 chosen so the footprint has 16 frames of context on both sides; the official stage 4 keeps only the first 200 frames of a clip, so the export is restricted to the window"],
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    if d["model_output_exists"]:
        raise SystemExit(f"STOP DECLARATION_INVALID: model output exists {model_outputs}")
    sha = dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha, "generated": r["mask"]["generated_joints"], "prompt": U.PROMPT, "seed": SEED}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ controls ---
def _expect(name, fn, codes, res):
    codes = (codes,) if isinstance(codes, str) else tuple(codes)
    try:
        fn()
        res[name] = {"pass": False, "note": "NOT refused"}
    except U.UniStop as e:
        res[name] = {"pass": e.code in codes, "code": e.code, "expected": list(codes)}


def stage_controls(a):
    check_declaration(a)
    out = new_dir(a.out_dir)
    pdir = Path(a.preflight).parent
    dA, dB = load(pdir / "dump_A.json"), load(pdir / "dump_B.json")
    z = np.load(pdir / "features" / "features.npz")
    names = [str(n) for n in z["names"]]
    par = np.asarray(z["parents"])
    clean = [str(n) for n in z["clean_joint_names"]]
    feats, tq = z["feats"], z["tpos_local_rotations"]
    rp = rig_parents(dB)
    neg = {}
    # sanity: the real inputs pass
    U.validate_asset(names, par, clean, U.FACE_PAIR, rp)
    # labels / joints / facing / order
    c2 = list(clean)
    i, j = names.index("lowerarm_l"), names.index("lowerarm_r")
    c2[i], c2[j] = c2[j], c2[i]
    _expect("mismatched_labels_forearm_swap", lambda: U.validate_asset(names, par, c2, U.FACE_PAIR, rp), ("LABEL_MISMATCH", "SIDE_MISMATCH"), neg)
    k = names.index("lowerarm_l")
    n3 = [n for q, n in enumerate(names) if q != k]
    _expect("missing_left_forearm", lambda: U.validate_asset(n3, list(par[:k]) + list(par[k + 1:]), clean[:k] + clean[k + 1:], U.FACE_PAIR, rp),
            ("JOINT_COUNT", "MISSING_JOINT"), neg)
    _expect("wrong_facing_pair", lambda: U.validate_asset(names, par, clean, U.FACE_PAIR[::-1], rp), "FACING_PAIR", neg)
    mir = np.asarray(z["tpos_first_frame"], float).copy()
    mir[:, 0] *= -1
    _expect("mirrored_tpose_handedness", lambda: U.check_facing(mir, names), ("FACING", "HANDEDNESS"), neg)
    back = np.asarray(z["tpos_first_frame"], float).copy()
    back[:, [0, 2]] *= -1
    _expect("facing_backwards", lambda: U.check_facing(back, names), "FACING", neg)
    n4 = list(names)
    a_, b_ = n4.index("upperarm_l"), n4.index("upperarm_r")
    n4[a_], n4[b_] = n4[b_], n4[a_]
    _expect("joint_order_permuted", lambda: U.validate_asset(n4, par, clean, U.FACE_PAIR, rp), ("HIERARCHY", "LABEL_MISMATCH"), neg)
    # time / scale / rest / drift
    dec = U.decode_basis(feats, names, par, tq)
    for s in (+1, -1):
        sh = {b: np.roll(v, s, axis=0) for b, v in dec.items()}
        _expect(f"shifted_frames_{s:+d}", lambda sh=sh: U.time_identity(sh, dB["stored"]), ("REPRESENTATION_BLOCKED", "TIME_SHIFT"), neg)
    off = np.asarray(z["offsets"], float).copy()
    off[:, 1] *= 1.05
    _expect("non_uniform_scaling", lambda: U.check_scale(off, names, par, rig_heads(dB)), "SCALING", neg)
    tq2 = np.asarray(tq, float).copy()
    ua = names.index("upperarm_l")
    tq2[ua] = U.qmul(tq2[ua], np.array([np.cos(np.radians(5)), np.sin(np.radians(5)), 0, 0]))
    _expect("rest_mismatch_tpose_rotation", lambda: U.time_identity(U.decode_basis(feats, names, par, tq2), dB["stored"]),
            ("REPRESENTATION_BLOCKED",), neg)
    f2 = feats.copy()
    s2 = names.index("spine_02")
    f2[:, s2, 3:9] = np.stack([U.quat_to_6d(U.qmul(U.mat_to_quat(U.sixd_to_mat(f2[t, s2, 3:9])),
                                                   np.array([np.cos(0.05), 0, np.sin(0.05), 0]))) for t in range(f2.shape[0])])
    def drift():
        ctx = U.decode_basis(f2, names, par, tq, joints=("spine_01",))
        e = U.roundtrip_error(ctx, dB["stored"], "spine_01")
        if e > U.QUAT_TOL:
            raise U.UniStop("REPRESENTATION_BLOCKED", f"context drift {e:.2e}")
    _expect("non_left_arm_drift", drift, "REPRESENTATION_BLOCKED", neg)
    # corrupt model / source (file identity)
    for nm, src, sha in (("corrupt_checkpoint_copy", W136 / "model" / CKPT_REL, CKPT_SHA), ("corrupt_S1_clip_copy", Path(S1["clip"]), S1["clip_sha"])):
        raw = bytearray(Path(src).read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        bad = out / ("bad_" + Path(src).name)
        bad.write_bytes(bytes(raw))
        neg[nm] = {"pass": M.sha256_file(bad) != sha, "code": "INTEGRITY_FAILED_HASH"}
        bad.unlink()
    _expect("fps_24", lambda: U.check_fps(24), "NOT_30_FPS", neg)
    _expect("fps_20", lambda: U.check_fps(20), "NOT_30_FPS", neg)
    # mask domain
    km = U.build_keep_mask(names, par, 71)
    U.validate_mask(km, names, par)
    for nm, fn in (("mask_frees_spine_01", lambda m: m.__setitem__((names.index("spine_01"), slice(None), 20), False)),
                   ("mask_frees_right_forearm", lambda m: m.__setitem__((names.index("lowerarm_r"), slice(None), 20), False)),
                   ("mask_frees_frame_170", lambda m: m.__setitem__((names.index("lowerarm_l"), slice(None), U.action_to_index(170)), False)),
                   ("mask_frees_frame_200", lambda m: m.__setitem__((names.index("hand_l"), slice(0, 3), U.action_to_index(200)), False)),
                   ("mask_frees_upperarm_joint", lambda m: m.__setitem__((names.index("upperarm_l"), slice(None), 20), False)),
                   ("mask_keeps_support_entry", lambda m: m.__setitem__((names.index("hand_l"), 3, 25), True)),
                   ("mask_nothing_generated", lambda m: m.__setitem__((slice(0, len(names)),), True))):
        m2 = km.copy()
        fn(m2)
        _expect(nm, lambda m2=m2: U.validate_mask(m2, names, par), ("MASK_DOMAIN", "MASK_INCOMPLETE", "MASK_EMPTY"), neg)
    # domain / pin / projection
    _expect("channel_spine_01", lambda: U.check_domain(range(172, 200), channels=("spine_01",)), "DOMAIN_CHANNEL", neg)
    _expect("frames_171_200", lambda: U.check_domain(range(171, 201)), "DOMAIN_FRAMES", neg)
    _expect("pin_overlap_request", lambda: U.check_domain(range(130, 158)), "PIN_OVERLAP", neg)
    f3 = feats.copy()
    f3[20, names.index("hand_l"), 3:9] = 0.0
    _expect("degenerate_6d", lambda: U.decode_basis(f3, names, par, tq), "RIG_PROJECTION_FAILED", neg)
    f4 = feats.copy()
    f4[20, names.index("hand_l"), 5] = np.nan
    _expect("non_finite_features", lambda: U.decode_basis(f4, names, par, tq), "RIG_PROJECTION_FAILED", neg)
    # real Blender refusals (unchanged W129 build stage; new W136 export stage)
    for name, vals in (("blender_channel_outside_domain", {"spine_01": {"180": [1, 0, 0, 0]}}),
                       ("blender_frame_171", {"upperarm_l": {"171": [1, 0, 0, 0]}}),
                       ("blender_frame_200", {"lowerarm_l": {"200": [1, 0, 0, 0]}})):
        spec = {"out_blend": str(out / f"neg_{name}.blend"), "interior_action_frames": list(U.INTERIOR), "values_wxyz": vals,
                "branch": "NEG", "action_name": "NEG", "expected_blend_sha256": FROZEN["A"][1]}
        dump(out / f"neg_{name}_spec.json", spec)
        try:
            blender(STAGE, "build_variant", out / f"neg_{name}.log", blend=pdir / "inputs" / "A.blend", spec=out / f"neg_{name}_spec.json",
                    report=out / f"neg_{name}_report.json")
            neg[name] = {"pass": False}
        except RuntimeError:
            txt = (out / f"neg_{name}.log").read_text()
            neg[name] = {"pass": ("REFUSED_" in txt) and not (out / f"neg_{name}.blend").exists(),
                         "refusal": [ln for ln in txt.splitlines() if "REFUSED_" in ln][:1]}
    try:
        blender(EXPORT_STAGE, "export", out / "neg_export_identity.log", blend=pdir / "inputs" / "C.blend", out=out / "neg_export.glb",
                frames="156,216", report=out / "neg_export_report.json", expected_sha=FROZEN["B"][1])
        neg["blender_export_wrong_source"] = {"pass": False}
    except RuntimeError:
        txt = (out / "neg_export_identity.log").read_text()
        neg["blender_export_wrong_source"] = {"pass": "REFUSED_SOURCE_IDENTITY" in txt and not (out / "neg_export.glb").exists()}
    res = {"negative_controls": neg, "all_fail_closed": bool(all(v["pass"] for v in neg.values())),
           "note": "no model was loaded or run in this stage"}
    dump(out / "controls.json", res)
    print(json.dumps({"all_fail_closed": res["all_fail_closed"], "n": len(neg), "neg": {k: v["pass"] for k, v in neg.items()}}, indent=1))
    return 0 if res["all_fail_closed"] else 3


# ------------------------------------------------------------------------------------------------ source 2D ---
def source_2d(rbs, rbA, interior=U.INTERIOR, window=U.WINDOW):
    z = np.load(S1["run"] / "perception" / "m1s0" / "observations_predictions.npz")
    man = load(S1["run"] / "manifest" / "input_manifest.json")
    Wd, Hd = man["intake"]["frame_size"]
    row = {int(s): k for k, s in enumerate(z["frame_index"])}
    fa = _frames(rbA)
    W_, I_ = [], []
    for f in range(window[0], window[1] + 1):
        k = row[U.source_of(f)]
        for nm in U.FIT_POINTS:
            c = U.COCO[nm]
            if z["input_valid"][k][c]:
                W_.append(fa[f]["world"][U.RIG_OF[nm]]["head"])
                I_.append(z["input_2d"][k][c][:2] * [Wd, Hd])
    P, rms = U.fit_affine_camera(np.array(W_), np.array(I_))
    out = {"camera": "affine 2x4, least squares over Action 156..215 torso points (both shoulders = upperarm heads, both hips = thigh heads) of A",
           "fit_rms_px": rms, "n_fit_points": len(W_), "valid": rms <= U.FIT_RMS_MAX_PX, "branches": {},
           "detections": {"label": "MMPose COCO-17 detector output on the self-occluded far (left) arm: HYPOTHESIS, not observed truth",
                          "confidence_l_elb_l_wri": {}}}
    for f in range(interior[0], interior[1] + 1):
        k = row[U.source_of(f)]
        out["detections"]["confidence_l_elb_l_wri"][str(f)] = [float(z["input_2d"][k][U.COCO["l_elb"]][2]), float(z["input_2d"][k][U.COCO["l_wri"]][2])]
    for br, rb in rbs.items():
        fr = _frames(rb)
        e = {"l_wri": [], "l_elb": []}
        for f in range(interior[0], interior[1] + 1):
            k = row[U.source_of(f)]
            for nm in e:
                c = U.COCO[nm]
                if z["input_valid"][k][c]:
                    pp = U.project(P, [fr[f]["world"][U.RIG_OF[nm]]["head"]])[0]
                    e[nm].append(float(np.linalg.norm(pp - z["input_2d"][k][c][:2] * [Wd, Hd])))
        out["branches"][br] = {f"{nm}_mean_px": float(np.mean(v)) if v else None for nm, v in e.items()}
        out["branches"][br].update({f"{nm}_max_px": float(np.max(v)) if v else None for nm, v in e.items()})
        out["branches"][br]["n"] = len(e["l_wri"])
    return out


# ------------------------------------------------------------------------------------------------ run ---
def stage_run(a):
    d = check_declaration(a)
    ctl = load(a.controls)
    if not ctl["all_fail_closed"]:
        raise SystemExit("STOP INTEGRITY_FAILED: controls did not fail closed")
    pdir = Path(a.preflight).parent
    out = new_dir(Path(a.out_dir))
    log = {"stage": "run", "start": now(), "steps": []}
    inp = pdir / "inputs"
    for k, fk in COPY_KEYS.items():
        if M.sha256_file(inp / f"{k}.blend") != FROZEN[fk][1]:
            raise SystemExit(f"STOP INTEGRITY_FAILED {k}")
    dumps = {k: load(pdir / f"dump_{k}.json") for k in "ABCDE"}
    for k in "ABCDE":
        if M.sha256_file(pdir / f"dump_{k}.json") != d["fresh_dump_sha256"][k]:
            raise SystemExit(f"STOP INTEGRITY_FAILED dump {k}")
    rig = Path(load(a.preflight)["representation"]["asset_dir"])
    clip = d["identities"]["rig_preprocess_inputs"]["clip"]
    if M.sha256_file(rig / "cond.npy") != d["identities"]["rig_preprocess_inputs"]["cond_sha256"] or \
            M.sha256_file(rig / "motions" / f"{clip}.npz") != d["identities"]["rig_preprocess_inputs"]["clip_sha256"] or \
            M.sha256_file(W136 / "model" / CKPT_REL) != CKPT_SHA:
        raise SystemExit("STOP INTEGRITY_FAILED asset/checkpoint")
    res = {"declaration_sha256": a.declaration_sha}
    # ---- ONE real inference ----
    log["steps"].append(["infer", um([INFER, "infer", "--out-dir", out / "model", "--asset-dir", rig, "--name", ASSET, "--clip", clip,
                                      "--exp-dir", MODEL_DIR, "--model-path", W136 / "model" / CKPT_REL, "--seed", SEED], out / "infer.log",
                                     torch_legacy_load=True)])
    if log["steps"][-1][1]["rc"] != 0 or not (out / "model" / "infer_record.json").is_file():
        txt = (out / "infer.log").read_text(encoding="utf-8")
        code = next((c for c in ("MASK_DOMAIN", "MASK_INCOMPLETE", "MASK_EMPTY", "JOINT_ORDER", "RIG_PROJECTION_FAILED", "OUTPUT_FILES")
                     if c in txt), "INFERENCE_FAILED")
        executed = (out / "model" / "raw_model_io.npz").is_file()
        res["STOP"] = {"code": "CONDITIONING_BLOCKED" if (code.startswith("MASK") or code == "JOINT_ORDER") and not executed else
                       ("RIG_PROJECTION_FAILED" if executed else code), "detail": code, "model_executed": executed}
        dump(out / "run_result.json", res)
        dump(out / "run_log.json", log)
        print(json.dumps(res, indent=1))
        return 3
    ir = load(out / "model" / "infer_record.json")
    res["inference"] = {k: ir[k] for k in ("calls", "sampler_seconds", "total_seconds", "peak_vram_bytes_sampler", "peak_vram_bytes_total", "device",
                                           "torch", "replacement_kept_max_abs_denorm", "generated_region_max_abs_denorm", "own_vs_official_decode_max",
                                           "generated_joint_names", "mask_info")}
    res["inference"]["raw_model_io_sha256"] = M.sha256_file(out / "model" / "raw_model_io.npz")
    res["inference"]["sample_npy_sha256"] = M.sha256_file(ir["sample_npy"])
    res["inference"]["proxyB_npy_sha256"] = M.sha256_file(ir["proxyB_npy"])
    dz = np.load(out / "model" / "decoded.npz")
    dec = {k: {b: dz[f"{k}_{b}"] for b in U.EDIT_CHANNELS} for k in ("U", "Bc")}
    try:
        res["M1_run_proxyB_time_identity"] = U.time_identity(dec["Bc"], dumps["B"]["stored"])
        vU = U.import_values(dec["U"], dumps["A"]["stored"])
        vBc = U.import_values(dec["Bc"], dumps["A"]["stored"])
    except U.UniStop as e:
        res["STOP"] = {"code": e.code, "detail": e.detail, "model_executed": True}
        dump(out / "run_result.json", res)
        print(json.dumps(res, indent=1))
        return 3
    dump(out / "import_U.json", vU)
    dump(out / "import_Bc.json", vBc)
    # ---- builds ----
    blends = {k: inp / f"{k}.blend" for k in "ABCDE"}
    for br, vals, base, base_sha, nm in (("U", vU, inp / "A.blend", FROZEN["A"][1], "AnimCV_EXP W136 U UniMate S1"),
                                         ("Bc", vBc, inp / "A.blend", FROZEN["A"][1], "AnimCV_EXP W136 Bc PROXY_B control S1"),
                                         ("Upin", vU, inp / "W126V_accepted.blend", FROZEN["W126V_accepted"][1], "AnimCV_EXP W136 U_on_W126V S1")):
        spec = {"out_blend": str(out / f"{br}_S1.blend"), "interior_action_frames": list(U.INTERIOR), "values_wxyz": vals, "branch": f"W136 {br}",
                "action_name": nm, "expected_blend_sha256": base_sha,
                "note": "W136 scratch: " + ("UniMate v3 generated upperarm_l/lowerarm_l at 172..199 (model-inferred, PROXY_B-conditioned)"
                                            if br != "Bc" else "PROXY_B through the identical decode/import path (no-generator control)")}
        dump(out / f"build_{br}_spec.json", spec)
        log["steps"].append([f"build_{br}", blender(STAGE, "build_variant", out / f"build_{br}.log", blend=base, spec=out / f"build_{br}_spec.json",
                                                     report=out / f"build_{br}_report.json")])
        blends[br] = out / f"{br}_S1.blend"
    stA = dumps["A"]["stored"]
    vA = {ch: {str(f): [stA[ch][str(c)][str(f)] for c in range(4)] for f in range(U.INTERIOR[0], U.INTERIOR[1] + 1)} for ch in U.EDIT_CHANNELS}
    spec = {"out_blend": str(out / "Urev_S1.blend"), "interior_action_frames": list(U.INTERIOR), "values_wxyz": vA, "branch": "W136 Urev",
            "action_name": "AnimCV_EXP W136 U_reverted S1", "expected_blend_sha256": M.sha256_file(blends["U"]), "note": "reversibility control"}
    dump(out / "build_Urev_spec.json", spec)
    log["steps"].append(["build_Urev", blender(STAGE, "build_variant", out / "build_Urev.log", blend=blends["U"], spec=out / "build_Urev_spec.json",
                                                report=out / "build_Urev_report.json")])
    blends["Urev"] = out / "Urev_S1.blend"
    rb = {k: load(pdir / f"readback_{k}.json") for k in "ABCDE"}
    rb["Pin"] = load(pdir / "readback_Pin.json")
    for br in ("U", "Bc", "Upin", "Urev"):
        log["steps"].append([f"readback_{br}", readback(blends[br], out / f"readback_{br}.json", out / f"readback_{br}.log")])
        rb[br] = load(out / f"readback_{br}.json")
    for br in ("U", "Bc"):
        log["steps"].append([f"dump_{br}", bdump(blends[br], out / f"dump_{br}.json", out / f"dump_{br}.log")])
        dumps[br] = load(out / f"dump_{br}.json")
    # ---- mechanical gates ----
    g = {}
    i0, i1 = U.INTERIOR
    for br in ("U", "Bc"):
        r = rb[br]
        pres = M.preservation(r, rb["A"], U.INTERIOR)
        ck = M.changed_keys(dumps[br]["stored"], dumps["A"]["stored"])
        census = r["fcurve_census"]
        m2 = (len(r["actions_in_file"]) == 1 and r["action_name"] not in {rb[k]["action_name"] for k in "ABCDE"}
              and census["n_fcurves"] == 32 and set(census["interpolation_census_curves"]) == {"CONSTANT"}
              and [census["key_frame_min"], census["key_frame_max"]] == [1.0, 240.0] and r["scene"]["effective_fps"] == 30.0
              and r["scene"]["frame_start"] == 1 and r["scene"]["frame_end"] == 240)
        m3 = (set(c[0] for c in ck) <= set(U.EDIT_CHANNELS) and all(i0 <= c[2] <= i1 for c in ck)
              and not pres["nonarm_changed"] and not pres["outside_changed"] and not pres["sub_outside_changed"])
        fX = _frames(r)
        L1 = [float(np.linalg.norm(np.subtract(fX[f]["world"]["lowerarm_l"]["head"], fX[f]["world"]["upperarm_l"]["head"]))) for f in range(i0 - 1, i1 + 2)]
        L2 = [float(np.linalg.norm(np.subtract(fX[f]["world"]["hand_l"]["head"], fX[f]["world"]["lowerarm_l"]["head"]))) for f in range(i0 - 1, i1 + 2)]
        vals = vU if br == "U" else vBc
        unit = all(abs(np.linalg.norm(q) - 1.0) < 1e-9 and np.isfinite(q).all() for ch in vals.values() for q in map(np.asarray, ch.values()))
        m5 = bool(np.ptp(L1) <= 1e-4 and np.ptp(L2) <= 1e-4 and unit)
        g[br] = {"M2": bool(m2), "M3": bool(m3), "M5": m5, "changed_key_channels": sorted({c[0] for c in ck}),
                 "changed_key_frames": [min((c[2] for c in ck), default=None), max((c[2] for c in ck), default=None)], "n_changed_keys": len(ck),
                 "nonarm_changed": pres["nonarm_changed"][:5], "outside_changed": pres["outside_changed"][:5],
                 "sub_outside_changed": pres["sub_outside_changed"][:5], "upper_len_range_m": float(np.ptp(L1)), "lower_len_range_m": float(np.ptp(L2)),
                 "blend_sha256": r["blend_sha256"], "action_name": r["action_name"], "n_subframes": len(r["subframes"])}
    m4 = bool(ir["calls"] == 1 and ir["replacement_kept_max_abs_denorm"] <= 1e-4 and ir["mask_info"]["n_free"] == d["inference"]["n_free_entries"])
    pin_src = load(FROZEN["W126V_proposal"][0])
    tgt = np.asarray(pin_src["constraint"]["target"])
    fp, fc = _frames(rb["Pin"]), _frames(rb["Upin"])
    rp_ = {f: float(np.linalg.norm(np.asarray(fp[f]["world"]["hand_l"]["head"]) - tgt)) for f in U.PIN_ACTION}
    rc_ = {f: float(np.linalg.norm(np.asarray(fc[f]["world"]["hand_l"]["head"]) - tgt)) for f in U.PIN_ACTION}
    cp = M.preservation(rb["Upin"], rb["Pin"], U.INTERIOR)
    m6 = {"residual_W126V_m": rp_, "residual_composed_m": rc_, "pin_frames_bit_equal": all(fc[f]["world"] == fp[f]["world"] for f in U.PIN_ACTION),
          "composed_nonarm_changed": cp["nonarm_changed"][:5], "composed_outside_changed": cp["outside_changed"][:5],
          "composed_sub_outside_changed": cp["sub_outside_changed"][:5]}
    m6["pass"] = bool(m6["pin_frames_bit_equal"] and all(rc_[f] == rp_[f] for f in U.PIN_ACTION) and max(rc_.values()) <= 1e-6
                      and not cp["nonarm_changed"] and not cp["outside_changed"] and not cp["sub_outside_changed"])
    m7 = all(p["world"] == q["world"] and p["matrix_world"] == q["matrix_world"] for p, q in zip(rb["Urev"]["frames"], rb["A"]["frames"])) \
        and all(p["world"] == q["world"] for p, q in zip(rb["Urev"]["subframes"], rb["A"]["subframes"]))
    eb = 0.0
    for p, q in zip(rb["Bc"]["frames"], rb["B"]["frames"]):
        for b in q["world"]:
            for k in ("head", "tail"):
                eb = max(eb, float(np.abs(np.subtract(p["world"][b][k], q["world"][b][k])).max()))
    after = frozen_hashes()
    g.update({"M1_run": True, "M4": m4, "M6": m6, "M7": bool(m7), "M8": bool(eb <= 1e-5), "M8_max_world_diff_Bc_vs_B_m": eb,
              "M9": all(v["ok"] for v in after.values()), "M10": bool(ctl["all_fail_closed"]), "frozen_after": {k: v["ok"] for k, v in after.items()}})
    # ---- generated effect ----
    fU, fBc = _frames(rb["U"]), _frames(rb["Bc"])
    eff = {}
    for f in range(i0, i1 + 1):
        eff[str(f)] = {"elbow_mm": float(np.linalg.norm(np.subtract(fU[f]["world"]["lowerarm_l"]["head"], fBc[f]["world"]["lowerarm_l"]["head"]))) * 1e3,
                       "wrist_mm": float(np.linalg.norm(np.subtract(fU[f]["world"]["hand_l"]["head"], fBc[f]["world"]["hand_l"]["head"]))) * 1e3}
    mx_eff = max(max(v["elbow_mm"], v["wrist_mm"]) for v in eff.values())
    generated = {"max_mm": mx_eff, "mean_wrist_mm": float(np.mean([v["wrist_mm"] for v in eff.values()])),
                 "mean_elbow_mm": float(np.mean([v["elbow_mm"] for v in eff.values()])), "per_frame": eff,
                 "class": "GENERATED_EFFECT" if mx_eff >= 10.0 else "NO_GENERATED_EFFECT"}
    # ---- quality ----
    hinge = M.hinge_reference(rb["A"])
    WIN = (164, 207)
    q = {}
    for br in ("A", "B", "C", "D", "E", "U", "Bc"):
        qq = M.quality(rb[br], rb["A"], U.INTERIOR, WIN, hinge)
        qq.pop("per_frame", None)
        qq["arm_local"] = M.armlocal_quality(M.armlocal_wrist(dumps[br], *WIN), U.INTERIOR, WIN)
        qq["arm_local"].pop("per_frame", None)
        em = EP.elbow_metrics(dumps[br])
        em.pop("per_frame", None)
        qq["elbow"] = em
        fr_ = _frames(rb[br])
        sh = np.array([fr_[f]["world"]["upperarm_l"]["head"] for f in range(WIN[0], WIN[1] + 1)])
        qq["world_shoulder_step_max_mm"] = float(np.linalg.norm(np.diff(sh, axis=0), axis=1).max() * 1e3)
        wB = _frames(rb["B"])
        qq["wrist_dev_from_B_max_mm"] = max(float(np.linalg.norm(np.subtract(fr_[f]["world"]["hand_l"]["head"], wB[f]["world"]["hand_l"]["head"])))
                                            for f in range(i0, i1 + 1)) * 1e3
        qq["wrist_net_rise_mm"] = (fr_[i1]["world"]["hand_l"]["head"][2] - fr_[i0]["world"]["hand_l"]["head"][2]) * 1e3
        q[br] = qq
    s2d = source_2d({k: rb[k] for k in ("A", "B", "C", "D", "E", "U", "Bc")}, rb["A"])
    qU, qB = q["U"], q["B"]
    steps_U = [qU["arm_local"]["boundary_c0"]["entry_step_mm"], qU["arm_local"]["boundary_c0"]["exit_step_mm"],
               qU["elbow"]["elbow_entry_step_mm"], qU["elbow"]["elbow_exit_step_mm"]]
    qa = {"Q1": bool(qU["arm_local"]["boundary_c0"]["entry_step_mm"] <= 2 * qB["arm_local"]["boundary_c0"]["entry_step_mm"] + 10
                     and qU["arm_local"]["boundary_c0"]["exit_step_mm"] <= 2 * qB["arm_local"]["boundary_c0"]["exit_step_mm"] + 10
                     and qU["elbow"]["elbow_entry_step_mm"] <= 2 * qB["elbow"]["elbow_entry_step_mm"] + 10
                     and qU["elbow"]["elbow_exit_step_mm"] <= 2 * qB["elbow"]["elbow_exit_step_mm"] + 10),
          "Q2": bool(qU["arm_local"]["n_reversal_flicks"] == 0 and len(qU["elbow"]["elbow_reversal_flicks"]) == 0),
          "Q3": bool(qU["arm_local"]["peak_step_mm"] <= 2 * qB["arm_local"]["peak_step_mm"] + 10),
          "Q4": bool(not qU["near_straight_frames"] and not qU["hinge_opposite_frames"]),
          "Q5": bool(s2d["valid"] and s2d["branches"]["U"]["l_wri_mean_px"] is not None
                     and s2d["branches"]["U"]["l_wri_mean_px"] <= s2d["branches"]["B"]["l_wri_mean_px"]),
          "Q6": bool(qB["wrist_net_rise_mm"] > 0 and qU["wrist_net_rise_mm"] >= 0.5 * qB["wrist_net_rise_mm"])}
    unusable = max(qU["arm_local"]["peak_step_mm"], qU["elbow"]["elbow_peak_step_mm"], *steps_U) > UNUSABLE_STEP_MM
    res.update({"gates": g, "generated_effect": generated, "quality": q, "source_2d": s2d, "quality_criteria_metric": qa,
                "unusable_continuity": bool(unusable), "unusable_threshold_mm": UNUSABLE_STEP_MM,
                "blends": {k: str(v) for k, v in blends.items()}, "blend_sha256": {k: M.sha256_file(v) for k, v in blends.items()}})
    dump(out / "run_result.json", res)
    vid = render_and_compose(out, pdir, blends, log)
    res["video"] = vid
    log["end"] = now()
    dump(out / "run_log.json", log)
    dump(out / "run_result.json", res)
    print(json.dumps({"gates": {k: (v if isinstance(v, bool) else (v.get("pass") if isinstance(v, dict) and "pass" in v else
                                                                    {kk: vv for kk, vv in v.items() if kk.startswith("M")} if isinstance(v, dict) else v))
                                for k, v in g.items() if k != "frozen_after"},
                      "generated": {k: v for k, v in generated.items() if k != "per_frame"}, "qa": qa, "unusable": unusable,
                      "inference": {k: res["inference"][k] for k in ("sampler_seconds", "peak_vram_bytes_total", "replacement_kept_max_abs_denorm")},
                      "s2d": {"fit_rms_px": s2d["fit_rms_px"], "valid": s2d["valid"], "U": s2d["branches"]["U"], "B": s2d["branches"]["B"]},
                      "video": {k: vid[k] for k in ("frames_written", "frames_decoded", "fps", "native_rate")}}, indent=1, default=str))
    return 0


def render_and_compose(out, pdir, blends, log, order=("A", "B", "C", "D", "E", "U")):
    import cv2
    s = S1
    n = s["n_action"]
    i0, i1 = U.INTERIOR
    rbA = load(pdir / "readback_A.json")
    fa = _frames(rbA)
    allp = np.array([fa[f]["world"][b][p] for f in fa for b in fa[f]["world"] for p in ("head", "tail")])
    centre = ((allp.min(0) + allp.max(0)) / 2).tolist()
    ext = float(np.max(allp.max(0) - allp.min(0)))
    el = np.array([fa[f]["world"]["lowerarm_l"]["head"] for f in range(164, 208)]).mean(0).tolist()
    views = [{"name": "body", "centre": centre, "offset": [0.0, -6.0, 0.0], "ortho_scale": ext * 1.15},
             {"name": "arm", "centre": el, "offset": [4.5, -4.5, 1.5], "ortho_scale": 1.3}]
    size = 320
    jobs = {"size": size, "views": views, "frames": [{"blender_frame": f, "markers": []} for f in range(1, n + 1)]}
    dump(out / "render_jobs.json", jobs)
    for br in order:
        log["steps"].append([f"render_{br}", blender(ACTION_STAGE, "render", out / f"render_{br}.log", blend=blends[br], jobs=out / "render_jobs.json",
                                                      out_dir=out / f"frames_{br}")])
    man = load(s["run"] / "manifest" / "input_manifest.json")
    clip = man["intake"]["clip_path"]
    seg0 = s["segment"][0]
    want = set(range(seg0, seg0 + n))
    cap = cv2.VideoCapture(clip)
    src, k = {}, 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k in want:
            src[k] = im
        k += 1
    cap.release()
    z = np.load(s["run"] / "perception" / "m1s0" / "observations_predictions.npz")
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

    def tile(br, f, arm_only=False):
        arm = cv2.imread(str(out / f"frames_{br}" / f"arm_f{f:04d}.png"))
        lab(arm, NAMES[br] if arm_only else "left-arm close-up", 16)
        inside = i0 <= f <= i1
        lab(arm, f"Action {f} / SOURCE {f + seg0 - 1}" if arm_only else ("CHANGED interior" if inside and br != "A" else ("interior" if inside else "unchanged")),
            34, (80, 80, 255) if inside else (200, 200, 200))
        if arm_only:
            return arm
        body = cv2.imread(str(out / f"frames_{br}" / f"body_f{f:04d}.png"))
        lab(body, NAMES[br], 16)
        lab(body, f"Action {f}", 34)
        return np.vstack([body, arm])

    def source_tile(f, h=2 * size):
        kk = f + seg0 - 1
        x0, y0, x1, y1 = crop
        im = src[kk][y0:y1, x0:x1].copy()
        im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        lab(im, "SOURCE (fixed crop)", 16)
        lab(im, f"SOURCE {kk}  t={kk / 30.0:.3f}s", 34)
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
        cv2.putText(b, f"edited interior Action {i0}-{i1} = SOURCE {i0 + seg0 - 1}-{i1 + seg0 - 1} (red); model window Action 156-215; 30 fps native; Action = SOURCE - 78",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        return b
    path = out / "abcdeu_S1_30fps.mp4"
    wr, written = None, 0
    for f in range(1, n + 1):
        top = np.hstack([source_tile(f)] + [tile(br, f) for br in order])
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
    vid = {"video": str(path), "sha256": M.sha256_file(path), "frames_written": written, "frames_decoded": nd, "fps": fps,
           "native_rate": nd == n and abs(fps - 30.0) < 1e-6}
    sheets = {"entry": list(range(170, 176)), "middle": list(range(183, 189)), "exit": list(range(196, 202)),
              "overview": list(range(171, 201, 3))}
    sh = {}
    for name, frs in sheets.items():
        rows = [np.hstack([cv2.resize(source_tile(f, size), (size, size)) for f in frs])]
        rows += [np.hstack([tile(br, f, arm_only=True) for f in frs]) for br in order]
        p = out / f"sheet_arm_{name}_S1.png"
        cv2.imwrite(str(p), np.vstack(rows))
        sh[f"arm_{name}"] = {"path": str(p), "sha256": M.sha256_file(p)}
        rows = [np.hstack([cv2.resize(source_tile(f, size), (size, size)) for f in frs])]
        rows += [np.hstack([cv2.imread(str(out / f"frames_{br}" / f"body_f{f:04d}.png")) for f in frs]) for br in order]
        p = out / f"sheet_body_{name}_S1.png"
        cv2.imwrite(str(p), np.vstack(rows))
        sh[f"body_{name}"] = {"path": str(p), "sha256": M.sha256_file(p)}
    vid["sheets"] = sh
    return vid


# ------------------------------------------------------------------------------------------------ assess ---
def stage_assess(a):
    check_declaration(a)
    out = new_dir(a.out_dir)
    pf, ctl = load(a.preflight), load(a.controls)
    rr = load(Path(a.run_dir) / "run_result.json")
    vr = load(a.visual_review)
    if pf["status"] != "OK":
        rep = {"stage": "assess", "case": "DATA_BLOCKED", "stop": pf.get("stop")}
    elif "STOP" in rr:
        st = rr["STOP"]
        case = "DATA_BLOCKED" if st["code"] == "CONDITIONING_BLOCKED" else ("C" if st.get("model_executed") else "D")
        rep = {"stage": "assess", "case": case, "stop": st}
    else:
        g = rr["gates"]
        mech = {"M0": pf["status"] == "OK", "M1": bool(g["M1_run"]), "M2": g["U"]["M2"] and g["Bc"]["M2"], "M3": g["U"]["M3"] and g["Bc"]["M3"],
                "M4": g["M4"], "M5": g["U"]["M5"] and g["Bc"]["M5"], "M6": g["M6"]["pass"], "M7": g["M7"], "M8": g["M8"], "M9": g["M9"],
                "M10": g["M10"], "M11": rr["video"]["native_rate"]}
        qa = dict(rr["quality_criteria_metric"])
        ans = vr["answers"]
        qa["Q7"] = bool(ans["U_more_natural_than_B"] and ans["U_more_press_faithful_than_B"] and not ans["U_flick_wrong_elbow_or_detour"])
        qa["Q8"] = bool(vr.get("native_rate_reviewed", False))
        if not mech["M0"]:
            case = "DATA_BLOCKED"
        elif not (mech["M8"] and mech["M9"] and mech["M10"]):
            case = "D"
        elif not all(mech[k] for k in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M11")) or rr["unusable_continuity"]:
            case = "C"
        elif rr["generated_effect"]["class"] == "GENERATED_EFFECT" and all(qa.values()):
            case = "A"
        else:
            case = "B"
        rep = {"stage": "assess", "case": case, "mechanical": mech, "generated_effect": rr["generated_effect"]["class"], "quality_criteria": qa,
               "unusable_continuity": rr["unusable_continuity"], "visual_review": ans, "native_rate_reviewed": vr.get("native_rate_reviewed", False)}
    rep.update({"time": now(), "rule": CASE_RULE, "visual_review_sha256": M.sha256_file(a.visual_review),
                "run_result_sha256": M.sha256_file(Path(a.run_dir) / "run_result.json"),
                "note": "visual review is the agent's own unblinded still-frame review, not human/animator validation"})
    sha = dump(out / "assessment.json", rep)
    print(json.dumps({**{k: v for k, v in rep.items() if k not in ("rule",)}, "sha": sha}, indent=1, default=str))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("preflight", "declare", "controls", "run", "assess"))
    ap.add_argument("--out-dir", required=True)
    for k in ("preflight", "declaration", "declaration-sha", "controls", "run-dir", "visual-review"):
        ap.add_argument("--" + k)
    a = ap.parse_args()
    return {"preflight": stage_preflight, "declare": stage_declare, "controls": stage_controls, "run": stage_run,
            "assess": stage_assess}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
