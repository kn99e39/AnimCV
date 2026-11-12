#!/usr/bin/env python3
"""W137 (EXPERIMENTAL, opt-in): official UniMate ``--inbetween`` temporal in-betweening on the authentic S1 rig.

Re-uses the frozen W136 own-rig PROXY_B asset, checkpoint, text encoder, prompt, seed, cfg and sampler; replaces ONLY
W136's custom joint x time mask by the official frame-only mode (keep window frames 0..15 + 44..59 for every joint and
channel, generate frames 16..43 = Action 172..199 for every joint and channel).  Imports only upperarm_l/lowerarm_l
rotations at Action 172..199; analyses the raw full-body output separately (raw vs frozen-parent projection).

Stages (each refuses an existing output directory; no AnimCV run directory or lineage is ever written):
  preflight  live gate (repo/instruction blob, W136 COMPLETE, no W137 Worklog), every frozen SHA-256 (S1 clip, A..E,
             W126V pin, W136 U/Bc/Upin/Urev/R0/records/own-rig asset), pinned code files, checkpoint/stats/config, local
             pinned T5 snapshot, re-fetched licence texts and clauses, runtime/resources; byte copies + fresh Blender
             dumps/readbacks (bit-compared to W136's); NO-MODEL controls: official features of the copied asset ->
             decode == B keys (time identity), official mask contract + mode exclusivity, official RIC/FK recovery of
             PROXY_B -> similarity map to B's Blender world (world-coordinate identity, time-shift separability);
             torso-only affine camera fit (diagnostic).
  declare    prospective declaration BEFORE any W137 model output.
  controls   W136's 31 real-data controls re-executed unchanged + W137 temporal-mode controls.  Never runs the model.
  run        ONE official inference -> conditioning gates -> raw full-body recovery + raw/projected attribution ->
             T (model) / Tc (no-generator) / Tpin / Trev Actions -> fresh readbacks/dumps -> gates, metrics -> renders.
  assess     frozen CASE rule over the run evidence + the agent's still-frame review file.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w137_unimate_inbetween_harness.py <stage> --out-dir <dir> [...]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np                                                    # noqa: E402

import app.experimental_unimate as U                                  # noqa: E402
import app.experimental_unimate_inbetween as T                        # noqa: E402
import app.experimental_elbow_prior as EP                             # noqa: E402
import app.experimental_motionbricks as M                             # noqa: E402
import w129_motionbricks_harness as H                                 # noqa: E402
import w136_unimate_harness as H6                                     # noqa: E402

W137 = Path("C:/Users/dna10/animcv-w137")
W136R = H6.W136 / "results"
INFER7 = ROOT / "scripts" / "w137_unimate_inbetween_infer.py"
BATCH, BLOB, START = 137, "71205cd79aa3ddfab6c54142dd4fb41c6853cbcf", "3b8ec215a0e877955014643cfd3feca0be256eb0"
W136_WORKLOG = "docs/136_WORKLOG_UNIMATE_V3_SOURCE_BOUND_LEFT_ARM_S1_CONTINUITY_UNUSABLE.md"
W136_DECL_SHA = "47259fbfd27a2e3fd762f8ebe362e2c92f616acde5b81cad5ea6de16bf616196"
W136_MANIFEST_SHA = "b25805a4f5b07bf71e85f671d9fde289e5f7a50513ab8701922be2f7b827a91f"
W136_BOUND = ["run/U_S1.blend", "run/Bc_S1.blend", "run/Upin_S1.blend", "run/Urev_S1.blend", "preflight/R0_S1.blend",
              "declare/declaration.json", "assess/assessment.json", "run/run_result.json", "preflight/preflight.json",
              "controls/controls.json", "review/visual_review_S1.json", "run/abcdeu_S1_30fps.mp4", "run/model/raw_model_io.npz",
              "run/model/infer_record.json", "run/model/official/motion_edit/motions/S1_PROXY_B-Animation-000-rep_0-0.npy",
              "run/model/official/motion_edit/motions/S1_PROXY_B-Animation-000-gt_rep_0-0.npy", "run/readback_U.json", "run/dump_U.json",
              "preflight/rig/cond.npy", "preflight/rig/summary.json", "preflight/rig/S1_PROXY_B.glb", "preflight/rig/annotation.json",
              "preflight/rig/motions/S1_PROXY_B-Animation-000.npz", "preflight/features/features.npz",
              "preflight/readback_A.json", "preflight/readback_B.json", "preflight/readback_C.json", "preflight/readback_D.json",
              "preflight/readback_E.json", "preflight/readback_Pin.json"]
RIG_FILES = ["cond.npy", "summary.json", "S1_PROXY_B.glb", "annotation.json", "motions/S1_PROXY_B-Animation-000.npz"]
CLIP = "S1_PROXY_B-Animation-000"
CONFIG_SHA = "1ffbd379d3667de7d701ed6f3418f1c9d1a66f35d2653776255008c07c051aaa"
T5_SAFETENSORS_SHA = "1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8"
T5_LOCAL = H6.HF_HOME / "hub" / "models--google--flan-t5-base" / "snapshots" / H6.T5_REV
INSTRUMENTS = ["src/app/experimental_unimate_inbetween.py", "scripts/w137_unimate_inbetween_infer.py",
               "scripts/w137_unimate_inbetween_harness.py", "tests/test_experimental_unimate_inbetween.py"] + H6.INSTRUMENTS
COPY = {"A": "A", "B": "B", "C": "C", "D": "W135_D", "E": "W135_E", "W126V_accepted": "W126V_accepted"}
ORDER = ("A", "B", "C", "D", "E", "U", "T")
H6.NAMES.update({"T": "T UniMate --inbetween (model)", "Tc": "Tc PROXY_B via T path", "U": "U W136 UniMate joint x time"})
ARM_BONES = ("upperarm_l", "lowerarm_l", "hand_l")
MAP_BONES = ("spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "clavicle_l", "clavicle_r",
             "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r", "hand_l", "hand_r", "thigh_l", "thigh_r", "calf_l", "calf_r",
             "foot_l", "foot_r")

load, dump, new_dir, now, git, lf_sha = H.load, H.dump, H.new_dir, H.now, H.git, H.lf_sha
blender, readback, bdump, um, _frames, _http, _norm_ws = H6.blender, H6.readback, H6.bdump, H6.um, H6._frames, H6._http, H6._norm_ws


def frozen_hashes():
    out = H6.frozen_hashes()
    man = W136R / "evidence_manifest.json"
    ok_m = man.is_file() and M.sha256_file(man) == W136_MANIFEST_SHA
    out["W136_evidence_manifest"] = {"path": str(man), "expected": W136_MANIFEST_SHA, "sha256": M.sha256_file(man) if man.is_file() else None, "ok": ok_m}
    files = load(man)["files"] if ok_m else {}
    for rel in W136_BOUND:
        p = W136R / rel
        want = (files.get(rel) or {}).get("sha256")
        got = M.sha256_file(p) if p.is_file() else None
        out["W136:" + rel] = {"path": str(p), "expected": want, "sha256": got, "ok": bool(want and got == want)}
    return out


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


def model_outputs_exist():
    return sorted(str(p) for p in (W137 / "results").rglob("raw_model_io.npz")) + sorted(str(p) for p in (W137 / "results").rglob("T_S1.blend"))


def world_max_diff(rb1, rb2, bones=None, frames=None):
    mx = 0.0
    for p, q in zip(rb1["frames"], rb2["frames"]):
        if frames is not None and p["blender_frame"] not in frames:
            continue
        for b in (bones or q["world"]):
            for k in ("head", "tail"):
                mx = max(mx, float(np.abs(np.subtract(p["world"][b][k], q["world"][b][k])).max()))
    return mx


def map_to_world(pos, sim):
    return T.apply_sim((sim["s"], sim["R"], sim["t"]), np.asarray(pos).reshape(-1, 3)).reshape(np.shape(pos))


# ------------------------------------------------------------------------------------------------ preflight ---
def representation(out, inp, rbB, dB):
    """NO-MODEL: official features of the byte-copied W136 asset -> decode == B; contract; RIC/FK -> world map."""
    g, steps = {}, []
    rig = out / "rig"
    for f in RIG_FILES:
        (rig / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(W136R / "preflight" / "rig" / f, rig / f)
    man = load(W136R / "evidence_manifest.json")["files"]
    g["asset_copy_equal"] = {f: M.sha256_file(rig / f) == man["preflight/rig/" + f]["sha256"] for f in RIG_FILES}
    if not all(g["asset_copy_equal"].values()):
        raise U.UniStop("DATA_BLOCKED", "own-rig asset copy differs")
    g["asset_sha256"] = {f: M.sha256_file(rig / f) for f in RIG_FILES}
    steps.append(["features", um([H6.INFER, "features", "--out-dir", out / "features", "--asset-dir", rig, "--name", H6.ASSET, "--clip", CLIP],
                                 out / "features.log")])
    if steps[-1][1]["rc"] != 0:
        raise U.UniStop("REPRESENTATION_BLOCKED", "features")
    z = np.load(out / "features" / "features.npz")
    names, par = [str(n) for n in z["names"]], np.asarray(z["parents"])
    U.check_fps(int(z["clip_fps"]))
    if int(z["n_clip_frames"]) != U.T_MODEL + 1 or z["feats"].shape != (U.T_MODEL, U.N_JOINTS, U.FEAT):
        raise U.UniStop("REPRESENTATION_BLOCKED", "clip shape")
    w136f = np.load(W136R / "preflight" / "features" / "features.npz")
    g["features_equal_W136"] = bool(np.array_equal(z["feats"], w136f["feats"]))
    U.validate_asset(names, par, z["clean_joint_names"], U.FACE_PAIR, H6.rig_parents(dB))
    g["facing_cos"] = U.check_facing(z["tpos_first_frame"], names)
    g["scale"] = U.check_scale(z["offsets"], names, par, H6.rig_heads(dB))
    dec = {b: z[f"decode_{b}"] for b in U.EDIT_CHANNELS}
    g["time_identity"] = U.time_identity(dec, dB["stored"])
    g["own_vs_official_decode"] = load(out / "features" / "features.json")["own_vs_official_decode_max"]
    if max(g["own_vs_official_decode"].values()) > 1e-9:
        raise U.UniStop("REPRESENTATION_BLOCKED", "own decoder != official decode")
    ctx = U.decode_basis(z["feats"], names, par, z["tpos_local_rotations"], joints=("spine_01", "thigh_l", "thigh_r", "calf_l", "calf_r"))
    g["context_drift"] = {b: U.roundtrip_error(ctx, dB["stored"], b) for b in ctx}
    if max(g["context_drift"].values()) > U.QUAT_TOL:
        raise U.UniStop("REPRESENTATION_BLOCKED", "context drift")
    g["joint_order"] = names
    # official temporal-mask contract + mode exclusivity (no model)
    steps.append(["contract", um([INFER7, "contract", "--out-dir", out / "contract"], out / "contract.log")])
    c = load(out / "contract" / "contract.json") if (out / "contract" / "contract.json").is_file() else {"ok": False}
    g["contract"] = c
    if not c.get("ok"):
        raise U.UniStop("CONDITIONING_BLOCKED", "official temporal mask / exclusivity contract failed")
    # official RIC/FK recovery of PROXY_B + world-coordinate map (no model)
    np.save(out / "proxyB_features.npy", z["feats"])
    steps.append(["recover_proxyB", um([INFER7, "recover", "--out-dir", out / "recover_proxyB", "--asset-dir", rig, "--name", H6.ASSET,
                                        "--sample", out / "proxyB_features.npy", "--proxy", out / "proxyB_features.npy"], out / "recover_proxyB.log")])
    r = np.load(out / "recover_proxyB" / "recovered.npz")
    g["proxyB_ric_vs_fk_max_model_units"] = float(np.abs(r["proxyB_ric"] - r["proxyB_fk"]).max())
    wm = T.fit_world_map(r["proxyB_fk"], names, rbB, MAP_BONES)
    g["world_map"] = wm
    shifted = {}
    for s in (-1, +1):
        X = r["proxyB_fk"][max(0, s):U.T_MODEL + min(0, s)]
        frames = range(U.WINDOW[0] + max(0, -s), U.WINDOW[1] + 1 + min(0, -s))
        Y = T.world_points(rbB, MAP_BONES, frames)
        shifted[f"{s:+d}"] = T.similarity_fit(X[:, [names.index(b) for b in MAP_BONES]].reshape(-1, 3), Y.reshape(-1, 3))[3]
    g["world_map_time_shift_rms_m"] = shifted
    if min(shifted.values()) < 100 * max(wm["rms_m"], 1e-9):
        raise U.UniStop("TIME_SHIFT", f"world map not time-separable {shifted}")
    wB = map_to_world(r["proxyB_fk"], wm)
    g["proxyB_fk_world_vs_B_max_m"] = float(max(np.abs(wB[i, names.index(b)] - np.asarray(_frames(rbB)[U.index_to_action(i)]["world"][b]["head"])).max()
                                               for i in range(U.T_MODEL) for b in MAP_BONES))
    g["asset_dir"] = str(rig)
    return g, steps


def stage_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"stage": "preflight", "time": now(), "head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
           "origin_head": git("rev-parse", "origin/arch/single_frame_first"), "steps": []}
    w136 = (ROOT / W136_WORKLOG).read_text(encoding="utf-8")
    rep["w136_complete"] = "AUTOMATION_BATCH_STATUS: COMPLETE" in w136 and "BATCH_NUMBER: 136" in w136
    rep["w137_worklog_absent"] = not any((ROOT / "docs").glob("137_*")) and not any((ROOT / "docs").glob("13[89]_*"))
    rep["frozen"] = frozen_hashes()
    stop = None
    code = {"repo": "https://github.com/Friedrich-M/UniMate", "pinned_commit": H6.UM_COMMIT,
            "head": subprocess.run(["git", "-C", str(H6.UM_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "dirty": subprocess.run(["git", "-C", str(H6.UM_ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip(),
            "untracked_non_cache": [ln for ln in subprocess.run(["git", "-C", str(H6.UM_ROOT), "status", "--porcelain"], capture_output=True,
                                                                text=True).stdout.splitlines() if ln.startswith("??") and "__pycache__" not in ln]}
    w136pf = load(W136R / "preflight" / "preflight.json")
    code["files_sha256"] = {f: M.sha256_file(H6.UM_ROOT / f) for f in w136pf["code"]["files_sha256"]}
    code["files_equal_W136"] = code["files_sha256"] == w136pf["code"]["files_sha256"]
    code["ok"] = code["head"] == H6.UM_COMMIT and code["dirty"] == "" and not code["untracked_non_cache"] and code["files_equal_W136"]
    rep["code"] = code
    # model (local, hash-verified; no download) + live public metadata/licence re-check
    meta = new_dir(out / "meta")
    api_ok = _http(f"https://huggingface.co/api/models/{H6.HF_REPO}/revision/{H6.HF_REV}?blobs=true", meta / "unimate_api.json")
    t5_ok = _http(f"https://huggingface.co/api/models/{H6.T5_REPO}/revision/{H6.T5_REV}?blobs=true", meta / "t5_api.json")
    for f, dst in (("README.md", meta / "hf_README.md"), ("LICENSE", meta / "hf_LICENSE")):
        _http(f"https://huggingface.co/{H6.HF_REPO}/resolve/{H6.HF_REV}/{f}", dst)
    _http(f"https://huggingface.co/{H6.T5_REPO}/resolve/{H6.T5_REV}/README.md", meta / "t5_README.md")
    mi = load(meta / "unimate_api.json") if api_ok else {}
    ti = load(meta / "t5_api.json") if t5_ok else {}
    sib = {s["rfilename"]: s for s in mi.get("siblings", [])}
    local = {f: M.sha256_file(H6.W136 / "model" / f) for f in (H6.CKPT_REL, f"{H6.MODEL_NAME}/dataset_stats.npy", f"{H6.MODEL_NAME}/config.json")}
    cfg = load(H6.MODEL_DIR / "config.json")
    model = {"repo": H6.HF_REPO, "revision": H6.HF_REV, "api_sha": mi.get("sha"), "gated": mi.get("gated"),
             "card_license": (mi.get("cardData") or {}).get("license"), "local_sha256": local,
             "checkpoint_lfs_sha256": (sib.get(H6.CKPT_REL, {}).get("lfs") or {}).get("sha256"),
             "config": {"diff_model": cfg["training"]["diff_model"], "max_motion_length": cfg["dataset"]["max_motion_length"],
                        "max_joints": cfg["dataset"]["max_joints"], "feature_len": cfg["dataset"]["feature_len"],
                        "text_encoder": cfg["model"]["text_encoder_version"], "cfg_scale": cfg["sampling"]["cfg_scale"], "seed": cfg["sampling"]["seed"]}}
    model["ok"] = bool(api_ok and mi.get("sha") == H6.HF_REV and mi.get("gated") is False and local[H6.CKPT_REL] == H6.CKPT_SHA
                       and model["checkpoint_lfs_sha256"] == H6.CKPT_SHA and local[f"{H6.MODEL_NAME}/dataset_stats.npy"] == H6.STATS_SHA
                       and local[f"{H6.MODEL_NAME}/config.json"] == CONFIG_SHA and model["config"]["cfg_scale"] == H6.CFG
                       and model["config"]["seed"] == H6.SEED and model["config"]["max_motion_length"] == U.T_MODEL
                       and model["config"]["feature_len"] == U.FEAT and model["config"]["diff_model"] == "flow"
                       and model["config"]["text_encoder"] == H6.T5_REPO)
    rep["model"] = model
    t5_local = T5_LOCAL / "model.safetensors"
    t5 = {"repo": H6.T5_REPO, "revision": H6.T5_REV, "api_sha": ti.get("sha"), "gated": ti.get("gated"),
          "card_license": (ti.get("cardData") or {}).get("license"), "local_path": str(T5_LOCAL),
          "local_safetensors_sha256": M.sha256_file(t5_local) if t5_local.is_file() else None,
          "local_files": sorted(p.name for p in T5_LOCAL.iterdir()) if T5_LOCAL.is_dir() else []}
    t5["ok"] = bool(t5_ok and ti.get("sha") == H6.T5_REV and ti.get("gated") is False and t5["card_license"] == "apache-2.0"
                    and t5["local_safetensors_sha256"] == T5_SAFETENSORS_SHA)
    rep["text_encoder"] = t5
    txt = {"hf_license": _norm_ws((meta / "hf_LICENSE").read_text(encoding="utf-8")) if (meta / "hf_LICENSE").is_file() else "",
           "hf_readme": _norm_ws((meta / "hf_README.md").read_text(encoding="utf-8")) if (meta / "hf_README.md").is_file() else "",
           "code_license": _norm_ws((H6.UM_ROOT / "LICENSE").read_text(encoding="utf-8")),
           "t5_readme": _norm_ws((meta / "t5_README.md").read_text(encoding="utf-8")) if (meta / "t5_README.md").is_file() else ""}
    present = {k: {c: _norm_ws(c) in txt[k] for c in cl} for k, cl in H6.RIGHTS_REQUIRED.items()}
    prov = {c: _norm_ws(c) in txt["hf_readme"] for c in H6.PROVENANCE_CLAUSES}
    fsha = {p.name: M.sha256_file(p) for p in meta.iterdir() if p.is_file()}
    w136_rights = w136pf["rights"]["files_sha256"]
    rights = {"required_clauses_present": present, "provenance_clauses_present": prov, "files_sha256": fsha,
              "licence_texts_equal_W136": {k: fsha.get(k) == w136_rights.get(k) for k in ("hf_LICENSE", "hf_README.md", "t5_README.md")},
              "intended_use": "private, local, non-commercial technical research evaluation on one machine; no redistribution of weights, "
                              "no public distribution of model outputs, no training/finetuning, no production or commercial use",
              "external_services": "none (no LLM API; Hugging Face used only for public metadata/licence pages; weights already local, hash-verified)"}
    rights["permitted"] = all(all(v.values()) for v in present.values()) and all(rights["licence_texts_equal_W136"].values())
    rights["determination"] = "PERMITTED_NONCOMMERCIAL_RESEARCH_EVALUATION" if rights["permitted"] else "RIGHTS_UNRESOLVED"
    rights["commercial"] = H6.load(W136R / "preflight" / "preflight.json")["rights"]["commercial"]
    rep["rights"] = rights
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
    du = shutil.disk_usage(str(W137))
    rep["runtime"] = {"um_env": envr, "nvidia_smi": smi, "disk_free_gb": round(du.free / 1e9, 1),
                      "freeze_sha256": M.sha256_file(H6.W136 / "env" / "freeze.txt"),
                      "ok": bool(envr.get("available") and "sm_120" in envr.get("arch", []) and du.free > 20e9)}
    rep["blender"] = subprocess.run([H.BLENDER, "-b", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    pre_ok = (rep["instruction_blob"] == BLOB and rep["head"] == START and rep["w136_complete"] and rep["w137_worklog_absent"]
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
        for k, fk in COPY.items():
            shutil.copyfile(H6.FROZEN[fk][0], inp / f"{k}.blend")
        shutil.copyfile(W136R / "run" / "U_S1.blend", inp / "U.blend")
        if any(M.sha256_file(inp / f"{k}.blend") != H6.FROZEN[fk][1] for k, fk in COPY.items()) or \
                M.sha256_file(inp / "U.blend") != rep["frozen"]["W136:run/U_S1.blend"]["expected"]:
            stop = "DATA_BLOCKED"
        for k in ("A", "B", "C", "D", "E", "U"):
            rep["steps"].append([f"dump_{k}", bdump(inp / f"{k}.blend", out / f"dump_{k}.json", out / f"dump_{k}.log")])
            rep["steps"].append([f"readback_{k}", readback(inp / f"{k}.blend", out / f"readback_{k}.json", out / f"readback_{k}.log")])
        rep["steps"].append(["readback_Pin", readback(inp / "W126V_accepted.blend", out / "readback_Pin.json", out / "readback_Pin.log")])
        eq = {}
        for k, ref in (("A", "preflight/readback_A.json"), ("B", "preflight/readback_B.json"), ("C", "preflight/readback_C.json"),
                       ("D", "preflight/readback_D.json"), ("E", "preflight/readback_E.json"), ("Pin", "preflight/readback_Pin.json"),
                       ("U", "run/readback_U.json")):
            w, f = load(W136R / ref), load(out / f"readback_{k}.json")
            eq[k] = all(p["world"] == q["world"] for p, q in zip(w["frames"], f["frames"])) and \
                all(p["world"] == q["world"] for p, q in zip(w["subframes"], f["subframes"]))
        rep["fresh_world_bit_equal_to_W136"] = eq
        if not all(eq.values()):
            stop = "DATA_BLOCKED"
        dB = load(out / "dump_B.json")
        try:
            EP.check_dump(dB)
            rep["representation"], st = representation(out, inp, load(out / "readback_B.json"), dB)
            rep["steps"] += st
        except U.UniStop as e:
            rep["representation"] = {"STOP": {"code": e.code, "detail": e.detail}}
            stop = {"CONDITIONING_BLOCKED": "CONDITIONING_BLOCKED", "NOT_30_FPS": "DATA_BLOCKED", "DATA_BLOCKED": "DATA_BLOCKED"}.get(e.code, "REPRESENTATION_BLOCKED")
        except EP.PriorStop as e:
            rep["representation"] = {"STOP": {"code": e.code, "detail": e.detail}}
            stop = "DATA_BLOCKED"
        except (RuntimeError, OSError, KeyError, ValueError, IndexError) as e:
            rep["representation"] = {"STOP": {"code": "REPRESENTATION_BLOCKED", "detail": f"{type(e).__name__}: {e}"}}
            stop = "REPRESENTATION_BLOCKED"
        if stop is None:
            s2d = H6.source_2d({"A": load(out / "readback_A.json")}, load(out / "readback_A.json"))
            rep["source_2d_camera"] = {k: s2d[k] for k in ("camera", "fit_rms_px", "n_fit_points", "valid")}
    rep["stop"] = stop
    rep["status"] = "OK" if stop is None else "DATA_BLOCKED"
    dump(out / "preflight.json", rep)
    print(json.dumps({"status": rep["status"], "stop": stop, "rights": rights["determination"], "model_ok": model["ok"], "t5_ok": t5["ok"],
                      "code_ok": code["ok"], "runtime_ok": rep["runtime"]["ok"], "frozen_ok": all(v["ok"] for v in rep["frozen"].values()),
                      "frozen_bad": [k for k, v in rep["frozen"].items() if not v["ok"]], "fresh_eq": rep.get("fresh_world_bit_equal_to_W136"),
                      "representation": {k: v for k, v in rep.get("representation", {}).items() if k in
                                         ("STOP", "time_identity", "world_map_time_shift_rms_m", "proxyB_fk_world_vs_B_max_m", "features_equal_W136")},
                      "world_map_rms": rep.get("representation", {}).get("world_map", {}).get("rms_m"),
                      "camera": rep.get("source_2d_camera", {}).get("fit_rms_px")}, indent=1, default=str))
    return 0 if stop is None else 3


# ------------------------------------------------------------------------------------------------ declaration ---
MECH_GATES = {
    "M0": "rights/access/resource/identity preflight OK (pinned code files == W136's, checkpoint/stats/config SHA-256, ungated pinned flan-t5-base local snapshot SHA-256, re-fetched licence texts equal to W136's with every required clause, sm_120 runtime, every frozen SHA-256 incl. W136 artifacts bound by W136's evidence manifest, fresh A..E/Pin/U readbacks world-bit-equal to W136's)",
    "M1": "representation (no model): copied own-rig asset == W136's; official features == W136's; decode == B keys <= 1e-5 at shift 0 and >= 100x worse at +-1; own == official decode; context drift <= 1e-5; official FK(tpos_offsets) of PROXY_B maps to B's Blender world by one similarity with RMS <= 1e-4 m and >= 100x worse at +-1 frame; AND in the run the saved PROXY_B tensor decodes to B's keys (time identity)",
    "M2": "Action: T and Tc each a separate .blend with exactly 1 separately named Action; 32 CONSTANT F-curves keyed 1..240; scene 30 fps, 1..240",
    "M3": "footprint: changed stored keys vs A only upperarm_l/lowerarm_l rotation at Action 172..199; every non-left-arm world point and the object matrix bit-equal to A on all 240 integer frames and all 717 subframes; left arm bit-equal outside 172..199 (incl. Action 171 and 200); no subframe change outside [172,200)",
    "M4": "official conditioning: exactly one generate_samples call through the unmodified official --inbetween path; the keep mask received by the sampler has shape (1,1,1,60) with True exactly at window 0..15 and 44..59 (validated BEFORE sampling); the official log reports valid_length=60 and the same keep indices; every kept frame of the denormalised sample equals PROXY_B <= 1e-4 in every joint group (root, torso, head, clavicle, left arm, right arm, legs, fingers); the interior is not a PROXY_B copy (exact-equal fraction <= 0.5 overall and for the left arm)",
    "M5": "rig validity: upperarm/lowerarm world length range <= 0.1 mm over 171..200 for T; all imported quaternions finite and unit; no decoder refusal",
    "M6": "hard pin: T composed onto the W126V-accepted .blend keeps every Action 132..140 hand_l residual equal to W126V's and <= 1e-6 m; pin frames bit-equal; composed footprint as M3 relative to W126V",
    "M7": "reversibility: T with A's interior keys written back (Trev) is world-bit-equal to A on every integer frame and subframe",
    "M8": "matched no-generator control: Tc (the PROXY_B tensor saved by the same official run, through the identical decode/import path) equals B within 1e-5 m on every frame",
    "M9": "integrity: every frozen SHA-256 unchanged after the run",
    "M10": "negative controls (controls stage: W136's 31 re-executed + W137 temporal controls) all fail closed",
    "M11": "native-rate video: 240 frames written and decoded at 30.000 fps",
    "M12": "projection consistency: the frozen-parent hybrid FK (PROXY_B features with only the upperarm_l/lowerarm_l rotation slots from the sample), mapped by the no-model world map, equals T's Blender readback lowerarm_l/hand_l heads within 1e-4 m on every window frame",
}
QUALITY = {
    "Q1": "boundary continuity: T arm-local (clavicle_l frame) wrist AND elbow entry (171->172) and exit (199->200) steps each <= 2 x B's corresponding step + 10 mm",
    "Q2": "no reversal flicks (consecutive steps both > 20 mm with negative dot) of the arm-local wrist or elbow in 171->172 .. 199->200",
    "Q3": "interior smoothness: T arm-local wrist peak step <= 2 x B's + 10 mm AND T arm-local elbow peak step <= 2 x B's + 10 mm",
    "Q4": "rig plausibility (elbow flex/hinge): no interior frame near-straight (> 175 deg) and no interior frame bending opposite to A's dominant hinge normal (W129 quality)",
    "Q5": "source-2D fidelity: ONLY if the torso-only affine camera (A, Action 156..215) fits with RMS <= 20 px, T's interior mean 2D left-wrist error <= B's. The input-only preflight fit is recorded in the declaration; if it is > 20 px the camera is INVALID, source fidelity is UNKNOWN and Q5 cannot pass (it is never used as a discriminator)",
    "Q6": "intended overhead press: T world wrist net rise (Z at 199 minus Z at 172) >= 0.5 x B's net rise (B's rise > 0)",
    "Q7": "agent still-frame review: T judged more natural AND more faithful to the source overhead press than B, and T not named for a flick, wrong elbow or detour",
    "Q8": "native-rate perceptual review of the 30 fps video by a human or an agent that actually perceives playback. This agent cannot perceive playback: declared NOT REVIEWED in advance, so Q8 cannot pass and CASE A is unattainable in this run",
}
W136_ANCHOR = {
    "T_vs_U_continuity": "IMPROVED iff T's arm-local wrist AND elbow max(entry, exit, peak) steps are each lower than W136 U's AND T's total wrist+elbow flicks <= U's; otherwise NOT_IMPROVED (reported finding; does not raise the CASE)",
    "unusable_diagnostic": "W136's 100 mm arm-local step is an explicit unusable-continuity diagnostic (not an optimisation target)",
}
RAW_RULE = ("raw full-body continuity is UNUSABLE iff (a) any raw world elbow or wrist step in 171->172 .. 199->200 exceeds PROXY_B's step at the "
            "same transition by > 100 mm, or (b) any joint group's raw world entry (171->172) or exit (199->200) step exceeds PROXY_B's at the same "
            "transition by > 100 mm (raw = official FK of the raw sample with tpos_offsets and its own integrated root, mapped by the no-model world map)")
ATTRIBUTION_RULE = (
    "arm-local (clavicle_l frame) wrist/elbow motion is identical in the raw and the projected views by construction (same upperarm_l/lowerarm_l "
    "local rotations, same bone lengths), so any arm-local failure is attributed to the RAW GENERATOR's arm rotations. World-trajectory "
    "differences raw vs projected are attributed to PARENT CONTEXT (split into shoulder translation and parent orientation). If the projected "
    "world/arm-local continuity fails but the raw world continuity is usable AND raw-minus-projected > 100 mm at a failing frame, the failure is "
    "attributed to FROZEN-PARENT PROJECTION; if both fail, to the RAW GENERATOR; if neither or the evidence conflicts, UNKNOWN.")
EFFECT_RULE = ("GENERATED_EFFECT iff max over Action 172..199 of max(|elbow_T - elbow_Tc|, |wrist_T - wrist_Tc|) (world, fresh readback) "
               ">= 10 mm; otherwise NO_GENERATED_EFFECT")
CASE_RULE = {
    "DATA_BLOCKED": "preflight not OK (RIGHTS_UNRESOLVED / ACCESS_BLOCKED / RESOURCE_BLOCKED / DATA_BLOCKED / REPRESENTATION_BLOCKED / CONDITIONING_BLOCKED), or the official temporal mode refuses/cannot hold the declared 32 frames before any sample exists",
    "D": "declaration/instrument changed or model output preceded the declaration; a negative control does not fail closed (M10); integrity (M9) fails; the matched control Tc != B (M8)",
    "C": "real inference executed but: decoder/projection refusal (RIG_PROJECTION_FAILED), M4 fails, any of M1-run, M2, M3, M5, M6, M7, M11, M12 fails, projected unusable continuity (any T arm-local wrist or elbow entry/exit/peak step in 171->172 .. 199->200 > 100 mm), or raw full-body continuity UNUSABLE (RAW_RULE)",
    "B": "a valid model Action exists (all mechanical gates pass) but NO_GENERATED_EFFECT or any of Q1-Q8 fails (improvement absent, unproven or review incomplete)",
    "A": "all mechanical gates pass AND GENERATED_EFFECT AND Q1-Q8 pass (unattainable here: Q8 declared NOT REVIEWED)",
    "precedence": "DATA_BLOCKED > D > C > B > A; a mechanical PASS, synthetic PASS or a video file alone never yields A",
}


def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit(f"preflight not OK: DATA_BLOCKED {pf.get('stop')}")
    pdir = Path(a.preflight).parent
    mo = model_outputs_exist()
    r = pf["representation"]
    d = {"schema": "animcv_exp_w137_declaration_v1", "time": now(), "outcomes_inspected": False, "model_output_exists": bool(mo),
         "batch": BATCH, "starting_commit": START, "instruction_blob": BLOB, "repo_head_at_declaration": git("rev-parse", "HEAD"),
         "preflight_sha256": M.sha256_file(a.preflight),
         "bindings": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in pf["frozen"].items()},
         "fresh_dump_sha256": {k: M.sha256_file(pdir / f"dump_{k}.json") for k in ("A", "B", "C", "D", "E", "U")},
         "fresh_readback_sha256": {k: M.sha256_file(pdir / f"readback_{k}.json") for k in ("A", "B", "C", "D", "E", "U", "Pin")},
         "identities": {"code": {"repo": pf["code"]["repo"], "commit": H6.UM_COMMIT, "licence": "MIT", "files_sha256": pf["code"]["files_sha256"]},
                        "model": {"repo": H6.HF_REPO, "revision": H6.HF_REV, "checkpoint": H6.CKPT_REL, "checkpoint_sha256": H6.CKPT_SHA,
                                  "config_sha256": CONFIG_SHA, "stats_sha256": H6.STATS_SHA, "licence": "CC BY-NC 4.0 (weights)"},
                        "text_encoder": {"repo": H6.T5_REPO, "revision": H6.T5_REV, "safetensors_sha256": T5_SAFETENSORS_SHA, "licence": "apache-2.0"},
                        "runtime": pf["runtime"]["um_env"], "own_rig_asset_sha256": r["asset_sha256"], "clip": CLIP},
         "rights": {"determination": pf["rights"]["determination"], "intended_use": pf["rights"]["intended_use"], "commercial": pf["rights"]["commercial"],
                    "files_sha256": pf["rights"]["files_sha256"], "note": "documentary reading by the agent, not legal advice and not a human licensing decision"},
         "time_rule": "native 30 fps; Action = SOURCE - 78; window Action 156..215 (SOURCE 234..293), feature frame i <-> Action 156+i; interior "
                      "Action 172..199 = SOURCE 250..277 = feature frames 16..43 inclusive; frames 171 and 200 kept and frozen",
         "proxy": {"label": "PROXY_B", "what": "the frozen W136 own-rig asset (official rig_preprocess of the W129 B Action over Action 156..216 via the "
                   "W136 export adapter), re-used byte-for-byte; a GT-free animation hypothesis, never observed 3D truth. The far-left-arm 3D pose of "
                   "the source is UNKNOWN. The right arm in PROXY_B is the rig rest, not observed motion.",
                   "conditioning_content": "every joint and channel at window frames 0..15 and 44..59; NOTHING of PROXY_B at frames 16..43 (all joints "
                                           "generated, incl. root, torso, clavicles, right arm, legs)"},
         "official_mode": {"flag": "--inbetween (InferenceArgs.inbetween=True); --motion_edit False; --motion_expand False",
                           "keep_frames_arg": T.KEEP_FRAMES_ARG, "keep_frames": list(T.KEEP_FRAMES), "free_frames": list(T.FREE_FRAMES),
                           "n_keep": len(T.KEEP_FRAMES), "n_free": len(T.FREE_FRAMES),
                           "mask": "official build_keep_mask -> (1,1,1,60) bool, broadcast over all joints and channels (contract verified without the model)",
                           "contract": r["contract"], "no_hooks": "no official function replaced in behaviour; generate_samples wrapped by a read-only recorder "
                           "(identical kwargs forwarded, identical tensor returned; CPU copies, timing, VRAM); W136's joint x time hook is NOT used"},
         "inference": {"path": "official unimate.inference.sample.main (asset = copied own-rig directory, test case {'S1_PROXY_B-Animation-000': prompt}, "
                               "--gt_start_frame 0, --num_repetitions 1, --batch_size 1, --only_save_motion)",
                       "prompt": U.PROMPT, "seed": H6.SEED, "cfg_scale": H6.CFG, "sampler": f"official fixed-step Euler replacement ODE, {H6.N_EULER} steps",
                       "weights": "EMA as loaded by the official _load_checkpoint", "one_run": True,
                       "runtime_compat": "torch 2.7.1+cu128 (sm_120); TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for the hash-verified checkpoint; HF offline (as W136)",
                       "no": "no seed/prompt/checkpoint/cfg/step/window sweep, no training/finetuning, no post-hoc smoothing/IK/blending/clamp/threshold change"},
         "mapping": {"joints": r["joint_order"], "decode": "unchanged W136 decode_basis (bvh(upperarm_l)=slot(lowerarm_l), bvh(lowerarm_l)=slot(hand_l))",
                     "time_identity": r["time_identity"], "context_drift": r["context_drift"], "own_vs_official_decode": r["own_vs_official_decode"],
                     "world_map": r["world_map"], "world_map_time_shift_rms_m": r["world_map_time_shift_rms_m"],
                     "raw_positions": "official recover_unimate_joint_pos_from_rot with cond tpos_offsets (the official animate_motion FK input) and the "
                                      "official integrated root; RIC (recover_unimate_joint_pos_from_ric) recorded as a diagnostic"},
         "import": {"channels": list(U.EDIT_CHANNELS), "frames": list(U.INTERIOR), "values": "decoded T basis quaternions, unit-norm, sign-canonical to A's key",
                    "builder": "unchanged W129 build_variant (refuses any other channel/frame), new opt-in Action from a byte copy of A",
                    "not_imported": "every other generated joint (root, torso, clavicles, right arm, legs, hand, fingers) and every frame outside 172..199"},
         "control": "Tc = the PROXY_B tensor saved by the same official run (gt_rep), decoded and imported through the identical path (expected == B, M8)",
         "historical": "A, B, C (W129), D, E (W135) and W136 U are read-only frozen references; no model is re-run for them",
         "domain": {"channels": list(U.EDIT_CHANNELS), "frames": list(U.INTERIOR), "pin_action": [U.PIN_ACTION[0], U.PIN_ACTION[-1]],
                    "pin_target_m": [-0.0927, 0.2199, 1.4506], "pin_tolerance_m": 0.001},
         "mechanical_gates": MECH_GATES, "generated_effect_rule": EFFECT_RULE, "quality_criteria": QUALITY, "w136_anchor": W136_ANCHOR,
         "raw_rule": RAW_RULE, "attribution_rule": ATTRIBUTION_RULE, "unusable_step_mm": T.UNUSABLE_STEP_MM, "case_rule": CASE_RULE,
         "source_2d_camera_input_only": pf.get("source_2d_camera"),
         "metrics": ["arm-local wrist (W129 armlocal_quality) and elbow (W135 elbow_metrics) entry/exit/peak/flicks for A,B,C,D,E,U,T,Tc",
                     "world wrist path, flex range, hinge (W129 quality); wrist net rise; deviation from B", "T vs Tc (generated effect); T vs U",
                     "raw full-body: per joint group interior deviation from PROXY_B, entry/exit/peak steps and excess; root drift; RIC vs FK",
                     "raw vs projected: elbow/wrist gap per frame, shoulder-translation / parent-orientation split, raw excess steps",
                     "world shoulder step (shared, source-derived parent popping)", "source-2D torso camera (diagnostic only)"],
         "review_protocol": "UNBLINDED agent review of fixed-time contact sheets (entry 170-175, middle 183-188, exit 196-201, overview 171..198 step 3) with "
                            "source crop, A, B, C, D, E, U, T; still frames only; written to a review JSON before assess. The 30 fps video is encoded and "
                            "machine-decoded but NOT perceived at native rate by this agent (Q8 NOT REVIEWED). No human/animator review.",
         "stop_conditions": ["RIGHTS_UNRESOLVED", "ACCESS_BLOCKED", "RESOURCE_BLOCKED", "DATA_BLOCKED", "REPRESENTATION_BLOCKED", "CONDITIONING_BLOCKED",
                             "DECLARATION_INVALID", "RIG_PROJECTION_FAILED", "INTEGRITY_FAILED", "no tuning or second run after any model output"],
         "pre_declaration_disclosures": [
             "input-only exploration (no model): the official FK recovery with cond 'offsets' (bone-local rest offsets) disagrees with RIC by 1.44 model "
             "units on PROXY_B; with cond 'tpos_offsets' (the official animate_motion input) FK == RIC to 2.8e-7 and maps to B's Blender world at "
             "2.6e-6 m RMS. FK input fixed to tpos_offsets and the world-map gate set to 1e-4 m after that measurement",
             "the torso-only affine camera is known from W136 to fail its 20 px gate; it is re-fitted on A only (input-only) and recorded here",
             "W137 re-uses W136's own-rig asset by byte copy instead of re-running the export/rig_preprocess"],
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    if d["model_output_exists"]:
        raise SystemExit(f"STOP DECLARATION_INVALID: model output exists {mo}")
    sha = dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha, "keep": T.KEEP_FRAMES_ARG, "prompt": U.PROMPT, "seed": H6.SEED,
                      "camera_rms_px": (pf.get("source_2d_camera") or {}).get("fit_rms_px")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ controls ---
def stage_controls(a):
    check_declaration(a)
    out = new_dir(a.out_dir)
    pdir = Path(a.preflight).parent
    res = {}
    # (1) W136's 31 real-data controls, re-executed unchanged against W136's own declaration/preflight
    w6 = SimpleNamespace(out_dir=str(out / "w136_controls"), preflight=str(W136R / "preflight" / "preflight.json"),
                         declaration=str(W136R / "declare" / "declaration.json"), declaration_sha=W136_DECL_SHA)
    rc = H6.stage_controls(w6)
    c6 = load(out / "w136_controls" / "controls.json")
    res["w136_controls"] = {"rc": rc, "n": len(c6["negative_controls"]), "all_fail_closed": c6["all_fail_closed"],
                            "sha256": M.sha256_file(out / "w136_controls" / "controls.json")}
    # (2) W137 temporal-mode controls (real PROXY_B features, real Blender world map; no model)
    neg = {}
    z = np.load(pdir / "features" / "features.npz")
    names, par = [str(n) for n in z["names"]], np.asarray(z["parents"])
    feats = z["feats"]
    E = H6._expect
    good = T.temporal_mask()
    T.validate_temporal_mask(good)
    T.check_keep_frames(list(T.KEEP_FRAMES))
    for nm, fn in (("mask_shift_plus1", lambda m: np.roll(m, 1, axis=3)), ("mask_shift_minus1", lambda m: np.roll(m, -1, axis=3)),
                   ("mask_missing_frame_15", lambda m: _set(m, 15, False)), ("mask_missing_frame_44", lambda m: _set(m, 44, False)),
                   ("mask_extra_frame_16", lambda m: _set(m, 16, True)), ("mask_extra_frame_43", lambda m: _set(m, 43, True)),
                   ("mask_all_kept", lambda m: np.ones_like(m)), ("mask_none_kept", lambda m: np.zeros_like(m)),
                   ("mask_joint_axis", lambda m: np.broadcast_to(m, (1, 50, 1, 60)).copy()),
                   ("mask_channel_axis", lambda m: np.broadcast_to(m, (1, 1, 12, 60)).copy()), ("mask_int_dtype", lambda m: m.astype(np.int64))):
        m2 = fn(good.copy())
        E(nm, lambda m2=m2: T.validate_temporal_mask(m2), ("MASK_DOMAIN", "MASK_EMPTY", "MASK_NOT_TEMPORAL", "MASK_DTYPE"), neg)
    for nm, k in (("keep_list_default_0_m1", [0, -1]), ("keep_list_frees_171", [i for i in T.KEEP_FRAMES if i != 15]),
                  ("keep_list_frees_200", [i for i in T.KEEP_FRAMES if i != 44]), ("keep_list_keeps_172", sorted(T.KEEP_FRAMES + (16,))),
                  ("keep_list_shifted", [i + 1 for i in T.KEEP_FRAMES if i < 59]), ("keep_list_negative", list(T.KEEP_FRAMES[:-1]) + [-1]),
                  ("keep_list_unsorted", list(reversed(T.KEEP_FRAMES)))):
        E(nm, lambda k=k: T.check_keep_frames(k), ("KEEP_FRAMES_FORMAT", "KEEP_FRAMES_DOMAIN"), neg)
    E("mode_motion_edit", lambda: T.check_mode(False, True), "CONDITIONING_BLOCKED", neg)
    E("mode_both", lambda: T.check_mode(True, True), "CONDITIONING_BLOCKED", neg)
    E("mode_expand", lambda: T.check_mode(True, False, True), "CONDITIONING_BLOCKED", neg)
    # conditioning-fidelity detectors on real PROXY_B features
    fake = feats.copy()
    fake[16:44] += 0.01
    T.check_conditioning(T.conditioning_report(fake, feats, names))
    for nm, (gi, fi) in (("kept_root_changed", ("pelvis", 3)), ("kept_right_arm_changed", ("lowerarm_r", 50)), ("kept_torso_changed", ("spine_03", 15)),
                         ("kept_left_arm_changed", ("hand_l", 44))):
        f2 = fake.copy()
        f2[fi, names.index(gi), 4] += 0.05
        E(nm, lambda f2=f2: T.check_conditioning(T.conditioning_report(f2, feats, names)), "CONDITIONING_FIDELITY", neg)
    E("interior_copied_from_proxyB", lambda: T.check_conditioning(T.conditioning_report(feats.copy(), feats, names)), "INTERIOR_COPIED", neg)
    E("feature_shape_59_frames", lambda: T.conditioning_report(feats[:59], feats[:59], names), "FEATURE_SHAPE", neg)
    # world-coordinate map controls (real PROXY_B FK vs real B readback)
    rbB = load(pdir / "readback_B.json")
    rr = np.load(pdir / "recover_proxyB" / "recovered.npz")
    fk = rr["proxyB_fk"]
    T.fit_world_map(fk, names, rbB, MAP_BONES)
    E("world_map_time_shift_plus1", lambda: T.fit_world_map(np.roll(fk, 1, axis=0), names, rbB, MAP_BONES), "REPRESENTATION_BLOCKED", neg)
    mir = fk.copy()
    mir[..., 0] *= -1
    E("world_map_mirrored", lambda: T.fit_world_map(mir, names, rbB, MAP_BONES), "REPRESENTATION_BLOCKED", neg)
    sw = list(names)
    i_, j_ = sw.index("hand_l"), sw.index("hand_r")
    sw[i_], sw[j_] = sw[j_], sw[i_]
    E("world_map_joint_swap", lambda: T.fit_world_map(fk, sw, rbB, MAP_BONES), "REPRESENTATION_BLOCKED", neg)
    E("world_map_offsets_not_tpos", lambda: T.fit_world_map(_fk_with("offsets", pdir, out), names, rbB, MAP_BONES), "REPRESENTATION_BLOCKED", neg)
    res["w137_controls"] = neg
    res["all_fail_closed"] = bool(c6["all_fail_closed"] and all(v["pass"] for v in neg.values()))
    res["n_total"] = res["w136_controls"]["n"] + len(neg)
    res["note"] = "no model was loaded or run in this stage"
    dump(out / "controls.json", res)
    print(json.dumps({"all_fail_closed": res["all_fail_closed"], "n_total": res["n_total"], "w136": res["w136_controls"],
                      "w137": {k: v["pass"] for k, v in neg.items()}}, indent=1))
    return 0 if res["all_fail_closed"] else 3


def _set(m, i, v):
    m[0, 0, 0, i] = v
    return m


def _fk_with(key, pdir, out):
    """Official FK with the WRONG offsets (cond 'offsets'), computed in the UniMate env (no model) -- negative control."""
    od = out / f"ctl_fk_{key}"
    if not (od / "fk.npy").is_file():
        od.mkdir(exist_ok=True)
        code = ("import numpy as np,sys;from unimate.utils.motion_utils import recover_unimate_joint_pos_from_rot as f;"
                f"c=np.load(r'{pdir / 'rig' / 'cond.npy'}',allow_pickle=True).item()['{H6.ASSET}'];"
                f"x=np.load(r'{pdir / 'proxyB_features.npy'}');np.save(r'{od / 'fk.npy'}',f(x,np.asarray(c['parents']),np.asarray(c['{key}'],float)))")
        um(["-c", code], od / "fk.log")
    return np.load(od / "fk.npy")


# ------------------------------------------------------------------------------------------------ run ---
def stage_run(a):
    d = check_declaration(a)
    ctl = load(a.controls)
    if not ctl["all_fail_closed"]:
        raise SystemExit("STOP INTEGRITY_FAILED: controls did not fail closed")
    pf = load(a.preflight)
    pdir = Path(a.preflight).parent
    out = new_dir(Path(a.out_dir))
    log = {"stage": "run", "start": now(), "steps": []}
    inp = pdir / "inputs"
    for k, fk in COPY.items():
        if M.sha256_file(inp / f"{k}.blend") != H6.FROZEN[fk][1]:
            raise SystemExit(f"STOP INTEGRITY_FAILED {k}")
    if M.sha256_file(inp / "U.blend") != d["bindings"]["W136:run/U_S1.blend"]["sha256"]:
        raise SystemExit("STOP INTEGRITY_FAILED U")
    for k, h in d["fresh_dump_sha256"].items():
        if M.sha256_file(pdir / f"dump_{k}.json") != h:
            raise SystemExit(f"STOP INTEGRITY_FAILED dump {k}")
    rig = Path(pf["representation"]["asset_dir"])
    if {f: M.sha256_file(rig / f) for f in RIG_FILES} != d["identities"]["own_rig_asset_sha256"] or M.sha256_file(H6.W136 / "model" / H6.CKPT_REL) != H6.CKPT_SHA:
        raise SystemExit("STOP INTEGRITY_FAILED asset/checkpoint")
    dumps = {k: load(pdir / f"dump_{k}.json") for k in ("A", "B", "C", "D", "E", "U")}
    names = d["mapping"]["joints"]
    res = {"declaration_sha256": a.declaration_sha}
    # ---- ONE real inference (official --inbetween) ----
    log["steps"].append(["infer", um([INFER7, "infer", "--out-dir", out / "model", "--asset-dir", rig, "--name", H6.ASSET, "--clip", CLIP,
                                      "--exp-dir", H6.MODEL_DIR, "--model-path", H6.W136 / "model" / H6.CKPT_REL, "--seed", H6.SEED,
                                      "--keep-frames", T.KEEP_FRAMES_ARG], out / "infer.log", torch_legacy_load=True)])
    txt = (out / "infer.log").read_text(encoding="utf-8", errors="replace")
    executed = (out / "model" / "raw_model_io.npz").is_file()
    if log["steps"][-1][1]["rc"] != 0 or not (out / "model" / "infer_record.json").is_file():
        code = next((c for c in ("MASK_DOMAIN", "MASK_EMPTY", "MASK_NOT_TEMPORAL", "MASK_DTYPE", "KEEP_FRAMES_DOMAIN", "CONDITIONING_BLOCKED",
                                 "RIG_PROJECTION_FAILED", "OUTPUT_FILES", "FEATURE_SHAPE") if c in txt), "INFERENCE_FAILED")
        res["STOP"] = {"code": "CONDITIONING_BLOCKED" if not executed and code != "INFERENCE_FAILED" else ("RIG_PROJECTION_FAILED" if executed else code),
                       "detail": code, "model_executed": executed, "log_tail": txt.strip().splitlines()[-5:]}
        dump(out / "run_result.json", res)
        dump(out / "run_log.json", log)
        print(json.dumps(res, indent=1))
        return 3
    ir = load(out / "model" / "infer_record.json")
    res["inference"] = {k: ir.get(k) for k in ("calls", "sampler_seconds", "total_seconds", "peak_vram_bytes_sampler", "peak_vram_bytes_total", "device",
                                               "torch", "mask_validate", "cfg_scale", "motion_shape", "own_vs_official_decode_max", "conditioning",
                                               "wrapped_function", "sample_shape", "proxyB_shape")}
    res["inference"]["args"] = {k: ir["args"][k] for k in ("inbetween", "motion_edit", "motion_expand", "keep_frames", "gt_start_frame", "seed",
                                                            "cfg_scale", "num_repetitions", "batch_size")}
    official_line = [ln for ln in txt.splitlines() if "build_keep_mask:" in ln]
    res["inference"]["official_log_build_keep_mask"] = official_line
    mlog = re.search(r"valid_length=(\d+), keep_indices=\[([0-9, ]+)\]", official_line[-1]) if official_line else None
    log_ok = bool(mlog and int(mlog.group(1)) == U.T_MODEL and [int(x) for x in mlog.group(2).split(",")] == list(T.KEEP_FRAMES))
    for k, p in (("raw_model_io", out / "model" / "raw_model_io.npz"), ("sample_npy", ir["sample_npy"]), ("proxyB_npy", ir["proxyB_npy"])):
        res["inference"][f"{k}_sha256"] = M.sha256_file(p)
    s_feats, g_feats = np.load(ir["sample_npy"]), np.load(ir["proxyB_npy"])
    try:
        cond_ok = T.check_conditioning(ir["conditioning"])
    except U.UniStop as e:
        cond_ok = False
        res["conditioning_stop"] = {"code": e.code, "detail": e.detail}
    res["proxyB_equals_asset_features"] = bool(np.abs(g_feats - np.load(pdir / "proxyB_features.npy")).max() <= 1e-4)
    m4 = bool(ir["calls"] == 1 and ir["mask_validate"]["n_kept"] == 32 and log_ok and cond_ok and res["proxyB_equals_asset_features"]
              and ir["args"]["inbetween"] and not ir["args"]["motion_edit"] and ir["args"]["keep_frames"] == T.KEEP_FRAMES_ARG)
    dz = np.load(out / "model" / "decoded.npz")
    dec = {k: {b: dz[f"{k}_{b}"] for b in U.EDIT_CHANNELS} for k in ("T", "Tc")}
    try:
        res["M1_run_proxyB_time_identity"] = U.time_identity(dec["Tc"], dumps["B"]["stored"])
        vT = U.import_values(dec["T"], dumps["A"]["stored"])
        vTc = U.import_values(dec["Tc"], dumps["A"]["stored"])
    except U.UniStop as e:
        res["STOP"] = {"code": e.code, "detail": e.detail, "model_executed": True}
        dump(out / "run_result.json", res)
        print(json.dumps(res, indent=1))
        return 3
    dump(out / "import_T.json", vT)
    dump(out / "import_Tc.json", vTc)
    # ---- raw full-body recovery (no further model call) ----
    log["steps"].append(["recover", um([INFER7, "recover", "--out-dir", out / "recover", "--asset-dir", rig, "--name", H6.ASSET,
                                        "--sample", ir["sample_npy"], "--proxy", ir["proxyB_npy"]], out / "recover.log")])
    rr = np.load(out / "recover" / "recovered.npz")
    wm = d["mapping"]["world_map"]
    raw_w, gt_w, hyb_w = (map_to_world(rr[f"{k}_fk"], wm) for k in ("sample", "proxyB", "hybrid"))
    att = T.attribution(raw_w, gt_w, hyb_w, names)
    body = T.body_report(raw_w, gt_w, names)
    rawu = T.raw_unusable(att, body)
    root_drift = np.linalg.norm(raw_w[:, 0] - gt_w[:, 0], axis=1) * 1e3
    ric_w = map_to_world(rr["sample_ric"], wm)
    fi = [U.action_to_index(f) for f in range(U.INTERIOR[0], U.INTERIOR[1] + 1)]
    raw = {"attribution": att, "body": body, "raw_rule": rawu,
           "root_drift_mm": {"at_action_199": float(root_drift[U.action_to_index(199)]), "at_action_200": float(root_drift[U.action_to_index(200)]),
                             "max_post_gap": float(root_drift[44:].max()), "max_pre_gap": float(root_drift[:16].max())},
           "ric_vs_fk_mm": {"raw_interior_max": float(np.linalg.norm(ric_w[fi] - raw_w[fi], axis=-1).max() * 1e3),
                            "raw_kept_max": float(np.linalg.norm(np.delete(ric_w, fi, 0) - np.delete(raw_w, fi, 0), axis=-1).max() * 1e3),
                            "proxyB_max": float(np.linalg.norm(map_to_world(rr["proxyB_ric"], wm) - gt_w, axis=-1).max() * 1e3)},
           "proxyB_fk_vs_B_readback_max_m": None}
    np.savez(out / "raw_world.npz", raw=raw_w, proxyB=gt_w, hybrid=hyb_w, raw_ric=ric_w, names=np.array(names))
    # ---- builds ----
    blends = {k: inp / f"{k}.blend" for k in ("A", "B", "C", "D", "E", "U")}
    for br, vals, base, base_sha, nm in (("T", vT, inp / "A.blend", H6.FROZEN["A"][1], "AnimCV_EXP W137 T UniMate inbetween S1"),
                                         ("Tc", vTc, inp / "A.blend", H6.FROZEN["A"][1], "AnimCV_EXP W137 Tc PROXY_B control S1"),
                                         ("Tpin", vT, inp / "W126V_accepted.blend", H6.FROZEN["W126V_accepted"][1], "AnimCV_EXP W137 T_on_W126V S1")):
        spec = {"out_blend": str(out / f"{br}_S1.blend"), "interior_action_frames": list(U.INTERIOR), "values_wxyz": vals, "branch": f"W137 {br}",
                "action_name": nm, "expected_blend_sha256": base_sha,
                "note": "W137 scratch: " + ("UniMate v3 official --inbetween generated upperarm_l/lowerarm_l at 172..199 (model-inferred)"
                                            if br != "Tc" else "PROXY_B through the identical decode/import path (no-generator control)")}
        dump(out / f"build_{br}_spec.json", spec)
        log["steps"].append([f"build_{br}", blender(H6.STAGE, "build_variant", out / f"build_{br}.log", blend=base, spec=out / f"build_{br}_spec.json",
                                                     report=out / f"build_{br}_report.json")])
        blends[br] = out / f"{br}_S1.blend"
    stA = dumps["A"]["stored"]
    vA = {ch: {str(f): [stA[ch][str(c)][str(f)] for c in range(4)] for f in range(U.INTERIOR[0], U.INTERIOR[1] + 1)} for ch in U.EDIT_CHANNELS}
    spec = {"out_blend": str(out / "Trev_S1.blend"), "interior_action_frames": list(U.INTERIOR), "values_wxyz": vA, "branch": "W137 Trev",
            "action_name": "AnimCV_EXP W137 T_reverted S1", "expected_blend_sha256": M.sha256_file(blends["T"]), "note": "reversibility control"}
    dump(out / "build_Trev_spec.json", spec)
    log["steps"].append(["build_Trev", blender(H6.STAGE, "build_variant", out / "build_Trev.log", blend=blends["T"], spec=out / "build_Trev_spec.json",
                                                report=out / "build_Trev_report.json")])
    blends["Trev"] = out / "Trev_S1.blend"
    rb = {k: load(pdir / f"readback_{k}.json") for k in ("A", "B", "C", "D", "E", "U", "Pin")}
    for br in ("T", "Tc", "Tpin", "Trev"):
        log["steps"].append([f"readback_{br}", readback(blends[br], out / f"readback_{br}.json", out / f"readback_{br}.log")])
        rb[br] = load(out / f"readback_{br}.json")
    for br in ("T", "Tc"):
        log["steps"].append([f"dump_{br}", bdump(blends[br], out / f"dump_{br}.json", out / f"dump_{br}.log")])
        dumps[br] = load(out / f"dump_{br}.json")
    # ---- mechanical gates ----
    g = {}
    i0, i1 = U.INTERIOR
    for br in ("T", "Tc"):
        r = rb[br]
        pres = M.preservation(r, rb["A"], U.INTERIOR)
        ck = M.changed_keys(dumps[br]["stored"], dumps["A"]["stored"])
        census = r["fcurve_census"]
        m2 = (len(r["actions_in_file"]) == 1 and r["action_name"] not in {rb[k]["action_name"] for k in ("A", "B", "C", "D", "E", "U")}
              and census["n_fcurves"] == 32 and set(census["interpolation_census_curves"]) == {"CONSTANT"}
              and [census["key_frame_min"], census["key_frame_max"]] == [1.0, 240.0] and r["scene"]["effective_fps"] == 30.0
              and r["scene"]["frame_start"] == 1 and r["scene"]["frame_end"] == 240)
        fX, fA = _frames(r), _frames(rb["A"])
        edge = all(fX[f]["world"] == fA[f]["world"] and fX[f]["matrix_world"] == fA[f]["matrix_world"] for f in (U.CONTEXT_LAST, U.TARGET_FIRST))
        m3 = (set(c[0] for c in ck) <= set(U.EDIT_CHANNELS) and all(i0 <= c[2] <= i1 for c in ck) and edge
              and not pres["nonarm_changed"] and not pres["outside_changed"] and not pres["sub_outside_changed"])
        L1 = [float(np.linalg.norm(np.subtract(fX[f]["world"]["lowerarm_l"]["head"], fX[f]["world"]["upperarm_l"]["head"]))) for f in range(i0 - 1, i1 + 2)]
        L2 = [float(np.linalg.norm(np.subtract(fX[f]["world"]["hand_l"]["head"], fX[f]["world"]["lowerarm_l"]["head"]))) for f in range(i0 - 1, i1 + 2)]
        vals = vT if br == "T" else vTc
        unit = all(abs(np.linalg.norm(q) - 1.0) < 1e-9 and np.isfinite(q).all() for ch in vals.values() for q in map(np.asarray, ch.values()))
        g[br] = {"M2": bool(m2), "M3": bool(m3), "M5": bool(np.ptp(L1) <= 1e-4 and np.ptp(L2) <= 1e-4 and unit), "edge_171_200_bit_equal": bool(edge),
                 "changed_key_channels": sorted({c[0] for c in ck}), "changed_key_frames": [min((c[2] for c in ck), default=None), max((c[2] for c in ck), default=None)],
                 "n_changed_keys": len(ck), "nonarm_changed": pres["nonarm_changed"][:5], "outside_changed": pres["outside_changed"][:5],
                 "sub_outside_changed": pres["sub_outside_changed"][:5], "upper_len_range_m": float(np.ptp(L1)), "lower_len_range_m": float(np.ptp(L2)),
                 "blend_sha256": r["blend_sha256"], "action_name": r["action_name"], "n_subframes": len(r["subframes"])}
    pin_src = load(H6.FROZEN["W126V_proposal"][0])
    tgt = np.asarray(pin_src["constraint"]["target"])
    fp, fc = _frames(rb["Pin"]), _frames(rb["Tpin"])
    rp_ = {f: float(np.linalg.norm(np.asarray(fp[f]["world"]["hand_l"]["head"]) - tgt)) for f in U.PIN_ACTION}
    rc_ = {f: float(np.linalg.norm(np.asarray(fc[f]["world"]["hand_l"]["head"]) - tgt)) for f in U.PIN_ACTION}
    cp = M.preservation(rb["Tpin"], rb["Pin"], U.INTERIOR)
    m6 = {"residual_W126V_m": rp_, "residual_composed_m": rc_, "pin_frames_bit_equal": all(fc[f]["world"] == fp[f]["world"] for f in U.PIN_ACTION),
          "composed_nonarm_changed": cp["nonarm_changed"][:5], "composed_outside_changed": cp["outside_changed"][:5],
          "composed_sub_outside_changed": cp["sub_outside_changed"][:5], "pin_target_m": tgt.tolist()}
    m6["pass"] = bool(m6["pin_frames_bit_equal"] and all(rc_[f] == rp_[f] for f in U.PIN_ACTION) and max(rc_.values()) <= 1e-6
                      and not cp["nonarm_changed"] and not cp["outside_changed"] and not cp["sub_outside_changed"])
    m7 = all(p["world"] == q["world"] and p["matrix_world"] == q["matrix_world"] for p, q in zip(rb["Trev"]["frames"], rb["A"]["frames"])) \
        and all(p["world"] == q["world"] for p, q in zip(rb["Trev"]["subframes"], rb["A"]["subframes"]))
    eb = world_max_diff(rb["Tc"], rb["B"])
    fT = _frames(rb["T"])
    m12 = max(float(np.abs(hyb_w[i, names.index(b)] - np.asarray(fT[U.index_to_action(i)]["world"][b]["head"])).max())
              for i in range(U.T_MODEL) for b in ("upperarm_l", "lowerarm_l", "hand_l"))
    raw["proxyB_fk_vs_B_readback_max_m"] = max(float(np.abs(gt_w[i, names.index(b)] - np.asarray(_frames(rb["B"])[U.index_to_action(i)]["world"][b]["head"])).max())
                                               for i in range(U.T_MODEL) for b in MAP_BONES)
    after = frozen_hashes()
    g.update({"M1_run": True, "M4": m4, "M4_detail": {"official_log_ok": log_ok, "conditioning_ok": bool(cond_ok), "calls": ir["calls"]},
              "M6": m6, "M7": bool(m7), "M8": bool(eb <= 1e-5), "M8_max_world_diff_Tc_vs_B_m": eb,
              "M9": all(v["ok"] for v in after.values()), "M10": bool(ctl["all_fail_closed"]), "frozen_after_bad": [k for k, v in after.items() if not v["ok"]],
              "M12": bool(m12 <= 1e-4), "M12_hybrid_vs_T_readback_max_m": m12})
    # ---- generated effect ----
    fTc = _frames(rb["Tc"])
    eff = {str(f): {"elbow_mm": float(np.linalg.norm(np.subtract(fT[f]["world"]["lowerarm_l"]["head"], fTc[f]["world"]["lowerarm_l"]["head"]))) * 1e3,
                    "wrist_mm": float(np.linalg.norm(np.subtract(fT[f]["world"]["hand_l"]["head"], fTc[f]["world"]["hand_l"]["head"]))) * 1e3}
           for f in range(i0, i1 + 1)}
    mx_eff = max(max(v["elbow_mm"], v["wrist_mm"]) for v in eff.values())
    generated = {"max_mm": mx_eff, "mean_wrist_mm": float(np.mean([v["wrist_mm"] for v in eff.values()])),
                 "mean_elbow_mm": float(np.mean([v["elbow_mm"] for v in eff.values()])), "per_frame": eff,
                 "class": "GENERATED_EFFECT" if mx_eff >= 10.0 else "NO_GENERATED_EFFECT"}
    # ---- quality ----
    hinge = M.hinge_reference(rb["A"])
    WIN = (164, 207)
    q = {}
    for br in ("A", "B", "C", "D", "E", "U", "T", "Tc"):
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
        qq["wrist_dev_from_B_mean_mm"] = float(np.mean([np.linalg.norm(np.subtract(fr_[f]["world"]["hand_l"]["head"], wB[f]["world"]["hand_l"]["head"]))
                                                        for f in range(i0, i1 + 1)])) * 1e3
        qq["wrist_net_rise_mm"] = (fr_[i1]["world"]["hand_l"]["head"][2] - fr_[i0]["world"]["hand_l"]["head"][2]) * 1e3
        q[br] = qq
    s2d = H6.source_2d({k: rb[k] for k in ("A", "B", "C", "D", "E", "U", "T", "Tc")}, rb["A"])
    qT, qB, qU = q["T"], q["B"], q["U"]

    def mx_steps(qq):
        return {"wrist": max(qq["arm_local"]["boundary_c0"]["entry_step_mm"], qq["arm_local"]["boundary_c0"]["exit_step_mm"], qq["arm_local"]["peak_step_mm"]),
                "elbow": max(qq["elbow"]["elbow_entry_step_mm"], qq["elbow"]["elbow_exit_step_mm"], qq["elbow"]["elbow_peak_step_mm"])}
    qa = {"Q1": bool(qT["arm_local"]["boundary_c0"]["entry_step_mm"] <= 2 * qB["arm_local"]["boundary_c0"]["entry_step_mm"] + 10
                     and qT["arm_local"]["boundary_c0"]["exit_step_mm"] <= 2 * qB["arm_local"]["boundary_c0"]["exit_step_mm"] + 10
                     and qT["elbow"]["elbow_entry_step_mm"] <= 2 * qB["elbow"]["elbow_entry_step_mm"] + 10
                     and qT["elbow"]["elbow_exit_step_mm"] <= 2 * qB["elbow"]["elbow_exit_step_mm"] + 10),
          "Q2": bool(qT["arm_local"]["n_reversal_flicks"] == 0 and len(qT["elbow"]["elbow_reversal_flicks"]) == 0),
          "Q3": bool(qT["arm_local"]["peak_step_mm"] <= 2 * qB["arm_local"]["peak_step_mm"] + 10
                     and qT["elbow"]["elbow_peak_step_mm"] <= 2 * qB["elbow"]["elbow_peak_step_mm"] + 10),
          "Q4": bool(not qT["near_straight_frames"] and not qT["hinge_opposite_frames"]),
          "Q5": bool(s2d["valid"] and s2d["branches"]["T"]["l_wri_mean_px"] is not None
                     and s2d["branches"]["T"]["l_wri_mean_px"] <= s2d["branches"]["B"]["l_wri_mean_px"]),
          "Q6": bool(qB["wrist_net_rise_mm"] > 0 and qT["wrist_net_rise_mm"] >= 0.5 * qB["wrist_net_rise_mm"])}
    sT, sU = mx_steps(qT), mx_steps(qU)
    vs_u = {"T_max_steps_mm": sT, "U_max_steps_mm": sU,
            "T_flicks": qT["arm_local"]["n_reversal_flicks"] + len(qT["elbow"]["elbow_reversal_flicks"]),
            "U_flicks": qU["arm_local"]["n_reversal_flicks"] + len(qU["elbow"]["elbow_reversal_flicks"])}
    vs_u["class"] = "IMPROVED" if (sT["wrist"] < sU["wrist"] and sT["elbow"] < sU["elbow"] and vs_u["T_flicks"] <= vs_u["U_flicks"]) else "NOT_IMPROVED"
    proj_unusable = max(sT.values()) > T.UNUSABLE_STEP_MM
    # attribution per declared rule
    arm_fail = proj_unusable or not (qa["Q1"] and qa["Q2"] and qa["Q3"])
    if not arm_fail:
        attrib = "NONE (projected continuity criteria pass)"
    elif rawu["unusable"]:
        attrib = "RAW_GENERATOR"
    elif arm_fail and max(v["raw_minus_projected_max_mm"] for v in att["raw_vs_projected"].values()) > T.UNUSABLE_STEP_MM and not rawu["unusable"]:
        attrib = "FROZEN_PARENT_PROJECTION"
    else:
        attrib = "UNKNOWN"
    res.update({"gates": g, "generated_effect": generated, "quality": q, "source_2d": s2d, "quality_criteria_metric": qa, "T_vs_U": vs_u,
                "projected_unusable_continuity": bool(proj_unusable), "raw": raw, "raw_unusable_continuity": rawu["unusable"],
                "attribution_class": attrib, "unusable_threshold_mm": T.UNUSABLE_STEP_MM,
                "blends": {k: str(v) for k, v in blends.items()}, "blend_sha256": {k: M.sha256_file(v) for k, v in blends.items()}})
    dump(out / "run_result.json", res)
    vid = H6.render_and_compose(out, pdir, blends, log, order=ORDER)
    old = Path(vid["video"])
    new = out / "abcdeut_S1_30fps.mp4"
    old.rename(new)
    vid["video"], vid["sha256"] = str(new), M.sha256_file(new)
    res["video"] = vid
    log["end"] = now()
    dump(out / "run_log.json", log)
    dump(out / "run_result.json", res)
    print(json.dumps({"gates": {k: (v if isinstance(v, (bool, float, int)) else (v.get("pass") if isinstance(v, dict) and "pass" in v else
                                                                                   {kk: vv for kk, vv in v.items() if kk.startswith("M") or kk.startswith("edge")}
                                                                                   if isinstance(v, dict) else v))
                                for k, v in g.items() if k not in ("frozen_after_bad",)},
                      "frozen_after_bad": g["frozen_after_bad"], "generated": {k: v for k, v in generated.items() if k != "per_frame"}, "qa": qa,
                      "T_vs_U": vs_u, "proj_unusable": proj_unusable, "raw_rule": rawu, "attribution": attrib,
                      "raw_vs_projected": {k: {kk: vv for kk, vv in v.items() if kk != "per_frame_mm"} for k, v in att["raw_vs_projected"].items()},
                      "raw_excess": att["raw_excess_step_over_proxyB"], "root_drift": raw["root_drift_mm"], "ric_vs_fk": raw["ric_vs_fk_mm"],
                      "inference": {k: res["inference"][k] for k in ("sampler_seconds", "total_seconds", "peak_vram_bytes_total")},
                      "conditioning": {k: v for k, v in ir["conditioning"].items() if k != "groups"},
                      "video": {k: vid[k] for k in ("frames_written", "frames_decoded", "fps", "native_rate")}}, indent=1, default=str))
    return 0


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
        mech = {"M0": pf["status"] == "OK", "M1": bool(g["M1_run"]), "M2": g["T"]["M2"] and g["Tc"]["M2"], "M3": g["T"]["M3"] and g["Tc"]["M3"],
                "M4": g["M4"], "M5": g["T"]["M5"] and g["Tc"]["M5"], "M6": g["M6"]["pass"], "M7": g["M7"], "M8": g["M8"], "M9": g["M9"],
                "M10": g["M10"], "M11": rr["video"]["native_rate"], "M12": g["M12"]}
        qa = dict(rr["quality_criteria_metric"])
        ans = vr["answers"]
        qa["Q7"] = bool(ans["T_more_natural_than_B"] and ans["T_more_press_faithful_than_B"] and not ans["T_flick_wrong_elbow_or_detour"])
        qa["Q8"] = bool(vr.get("native_rate_reviewed", False))
        if not mech["M0"]:
            case = "DATA_BLOCKED"
        elif not (mech["M8"] and mech["M9"] and mech["M10"]):
            case = "D"
        elif (not all(mech[k] for k in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M11", "M12")) or rr["projected_unusable_continuity"]
              or rr["raw_unusable_continuity"]):
            case = "C"
        elif rr["generated_effect"]["class"] == "GENERATED_EFFECT" and all(qa.values()):
            case = "A"
        else:
            case = "B"
        rep = {"stage": "assess", "case": case, "mechanical": mech, "generated_effect": rr["generated_effect"]["class"], "quality_criteria": qa,
               "projected_unusable_continuity": rr["projected_unusable_continuity"], "raw_unusable_continuity": rr["raw_unusable_continuity"],
               "raw_rule_reasons": rr["raw"]["raw_rule"]["reasons"], "attribution_class": rr["attribution_class"], "T_vs_U": rr["T_vs_U"]["class"],
               "visual_review": ans, "native_rate_reviewed": vr.get("native_rate_reviewed", False)}
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
