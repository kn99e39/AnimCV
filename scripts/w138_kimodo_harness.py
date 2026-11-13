#!/usr/bin/env python3
"""W138 (EXPERIMENTAL, opt-in): ONE Kimodo (nvidia/Kimodo-SOMA-RP-v1.2, no-text constraints-only mode) left-arm interior
proposal on the authentic S1 rig, selected by the frozen W138 selection matrix (commit 4b64557).

Generated clip: window Action 163..208 (46 frames, 30 fps, index i <-> Action 163 + i).  Vendor ``fullbody`` keyframes are
the frozen W129 PROXY_B pose (== A outside 172..199) at Action 163..171 and 200..208 (18 < 20); NOTHING is given inside
Action 172..199.  Only ``upperarm_l`` / ``lowerarm_l`` rotations at Action 172..199 are imported (unchanged W129
``build_variant``); the raw full-body output is analysed separately in Kimodo's canonical model space.

Stages (each refuses an existing output directory; no AnimCV run directory or lineage is ever written):
  preflight  live/selection gate, every frozen SHA-256 (W137's 55 + W137 artefacts), pinned code, local checkpoint vs
             published LFS SHA-256, licence copies, runtime; byte copies + fresh Blender dumps/readbacks (bit-compared to
             W137's); NO-MODEL representation: Kimodo skeleton, numpy-vs-vendor FK, W129 Mapping on Kimodo's SOMA77,
             PROXY_B decode == B keys (time identity), vendor feature roundtrip (encode/normalise/inverse) == B keys,
             vendor constraint + mask contract; torso-only affine camera (diagnostic).
  declare    prospective declaration BEFORE any W138 model output.
  controls   negative controls (no model).
  run        ONE inference -> conditioning gates -> raw/projected analysis -> K / Kc / Kpin / Krev Actions -> readbacks ->
             gates, metrics -> renders, video, sheets.
  assess     frozen CASE rule over the run evidence + the agent's still-frame review file.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w138_kimodo_harness.py <stage> --out-dir <dir> [...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np                                                    # noqa: E402

import app.experimental_kimodo as K                                   # noqa: E402
import app.experimental_unimate as U                                  # noqa: E402
import app.experimental_elbow_prior as EP                             # noqa: E402
import app.experimental_motionbricks as M                             # noqa: E402
import w129_motionbricks_harness as H                                 # noqa: E402
import w136_unimate_harness as H6                                     # noqa: E402
import w137_unimate_inbetween_harness as H7                           # noqa: E402
import w138_generator_qualification as Q                              # noqa: E402

W138 = Path("C:/Users/dna10/animcv-w138")
W137R = H7.W137 / "results"
KM_PY = W138 / "env" / "km311" / "Scripts" / "python.exe"
KM_ROOT = W138 / "ext" / "kimodo"
CKPT_DIR = W138 / "model" / "Kimodo-SOMA-RP-v1.2"
INFER8 = ROOT / "scripts" / "w138_kimodo_infer.py"
BATCH, BLOB = 138, "8129b0b504cfdca11c313ffcd6c3a53d75e2bd32"
START_BATCH = "df9eddc1eb84b5de79f2eaab848b62b4a203e21f"            # issue commit (Batch 138 published)
START_RUN = "0d82883a063205ed0597ab7d7d9e5b1d01da31e0"              # starting commit of this (resumed) run
PRIOR_COMMITS = ("69d22159bbbbe2d39b58545ec580a0c14625be5f", "4b64557ea33589a1ee0bb6383a7f31d526138df0")
MATRIX = "examples/experimental_kimodo/w138_selection_matrix.json"
MATRIX_SHA = "4860bc9f9ea5d39c426e00d1d4cc73a24bc9c85c276f03cdf29bff4ec792c9dd"
QUAL_DIR = W138 / "results" / "qualification"
CODE_COMMIT = "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"
HF_REPO, HF_REV = "nvidia/Kimodo-SOMA-RP-v1.2", "0e3c89024d896064b6851201373f7834784a386b"
CKPT_FILES = ["model.safetensors", "config.yaml", "LICENSE", "README.md", "stats/motion/body/mean.npy", "stats/motion/body/std.npy",
              "stats/motion/global_root/mean.npy", "stats/motion/global_root/std.npy", "stats/motion/local_root/mean.npy",
              "stats/motion/local_root/std.npy"]
CODE_FILES = ["LICENSE", "kimodo/model/kimodo_model.py", "kimodo/model/load_model.py", "kimodo/model/loading.py", "kimodo/model/cfg.py",
              "kimodo/model/backbone.py", "kimodo/model/twostage_denoiser.py", "kimodo/model/diffusion.py", "kimodo/constraints.py",
              "kimodo/motion_rep/reps/kimodo_motionrep.py", "kimodo/motion_rep/reps/base.py", "kimodo/skeleton/definitions.py",
              "kimodo/skeleton/base.py", "kimodo/skeleton/kinematics.py", "kimodo/sanitize.py", "kimodo/model/llm2vec/llm2vec_wrapper.py"]
SEED, STEPS, CFG_W = 10, 100, (2.0, 2.0)
INSTRUMENTS = ["src/app/experimental_kimodo.py", "scripts/w138_kimodo_infer.py", "scripts/w138_kimodo_harness.py", "tests/test_experimental_kimodo.py",
               "scripts/w138_generator_qualification.py", "src/app/experimental_motionbricks.py", "src/app/experimental_unimate.py",
               "src/app/experimental_elbow_prior.py", "src/blender/experimental_motionbricks_stage.py", "src/blender/experimental_action_stage.py",
               "scripts/w129_motionbricks_harness.py", "scripts/w136_unimate_harness.py", "scripts/w137_unimate_inbetween_harness.py"]
COPY = {"A": "A", "B": "B", "C": "C", "D": "W135_D", "E": "W135_E", "W126V_accepted": "W126V_accepted"}
ORDER = ("A", "B", "C", "D", "E", "U", "T", "K")
NAMES = dict(H6.NAMES)
NAMES.update({"U": "U W136 UniMate joint x time", "T": "T W137 UniMate inbetween", "K": "K Kimodo no-text (model)",
              "Kc": "Kc PROXY_B via K path"})
WIN_Q = (164, 207)
MAP_BONES = H7.MAP_BONES

load, dump, new_dir, now, git, lf_sha = H.load, H.dump, H.new_dir, H.now, H.git, H.lf_sha
blender, readback, bdump, _frames, _http = H6.blender, H6.readback, H6.bdump, H6._frames, H6._http


def km(args, log, timeout=3600):
    env = dict(os.environ)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        env.pop(k, None)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(W138 / "hf_empty"), "PYTHONUTF8": "1",
                "TEXT_ENCODER_MODE": "local", "TEXT_ENCODER_URL": "http://127.0.0.1:9/", "PYTHONPATH": ""})
    t = time.time()
    r = subprocess.run([str(KM_PY), "-X", "utf8", str(INFER8), *[str(x) for x in args]], cwd=str(W138), capture_output=True, text=True,
                       env=env, timeout=timeout, encoding="utf-8", errors="replace")
    Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return {"rc": r.returncode, "seconds": round(time.time() - t, 2), "log": str(log)}


def frozen_hashes():
    return Q.frozen_evidence()


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
    res = W138 / "results"
    return sorted(str(p) for p in res.rglob("raw_model_io.npz")) + sorted(str(p) for p in res.rglob("K_S1.blend"))


def mapping_for(dump_report):
    return M.Mapping(SOMA, M.rest_frame(dump_report["rest"], H.OBJECT_REST_WXYZ, H.OBJECT_SCALE))


SOMA = None


def load_soma(skel_npz):
    global SOMA
    z = np.load(skel_npz)
    SOMA = K.kimodo_soma([str(n) for n in z["names77"]], z["parents77"], z["neutral77"])
    return SOMA, z


def proxy_clip(mp, dump_report):
    """PROXY_B over the window: 77-joint local rotations, SOMA-space roots, canonical roots, numpy FK positions (30 joints)."""
    fr = _frames(dump_report)
    loc = np.stack([mp.to_soma_local(M.BaseFrame(dump_report["rest"], fr[f])) for f in range(K.WINDOW[0], K.WINDOW[1] + 1)])
    root = loc[:, 0, :3].copy()
    shift = K.canonical_shift(root)
    root_c = root + shift
    loc_c = loc.copy()
    loc_c[:, 0, :3] = root_c
    mats = np.stack([K.mats_from_soma_local(x) for x in loc_c])
    pos30 = np.stack([K.positions30(SOMA, x) for x in loc_c])
    return {"local": loc_c, "mats": mats, "root_canonical": root_c, "shift": shift, "pos30": pos30}


# ------------------------------------------------------------------------------------------------ preflight ---
def representation(out, dB, dA, rbA, rbB):
    g, steps = {}, []
    steps.append(["skeleton", km(["skeleton", "--out-dir", out / "skeleton"], out / "skeleton.log")])
    if steps[-1][1]["rc"] != 0:
        raise K.KimodoStop("REPRESENTATION_BLOCKED", "kimodo skeleton dump failed")
    sk = load(out / "skeleton" / "skeleton.json")
    g["kimodo_skeleton"] = sk
    if not (sk["names30_equal_model_joints"] and sk["display_name"] == CKPT_DIR.name and sk["sanitize_empty"] == ["", ""]):
        raise K.KimodoStop("REPRESENTATION_BLOCKED", "skeleton names / registry folder / empty-prompt sanitisation")
    soma, z = load_soma(out / "skeleton" / "skeleton.npz")
    # numpy FK (W129 SomaSkeleton on Kimodo neutral joints) == vendor FK
    errs = []
    for b in range(len(z["fk_local"])):
        sl = K.soma_local_from_mats(soma, z["fk_local"][b], z["fk_root"][b])
        _, P = soma.fk(sl)
        errs.append(float(np.abs(P - z["fk_pos"][b]).max()))
    g["numpy_vs_vendor_fk_max_m"] = max(errs)
    if max(errs) > 1e-6:
        raise K.KimodoStop("REPRESENTATION_BLOCKED", f"numpy FK != vendor FK {max(errs):.2e}")
    g["slice30_names"] = [str(z["names77"][i]) for i in z["slice30"]]
    if g["slice30_names"] != list(K.SOMA30):
        raise K.KimodoStop("REPRESENTATION_BLOCKED", "vendor 30<-77 slice order")
    g["hip_height_m"] = float(soma.ref[0, 1])
    mp = mapping_for(dB)
    g["mapping"] = mp.report()
    g["hip_scale"] = mp.hip_scale
    fr = _frames(dB)
    nm = [M.no_mirroring_check(mp, mp.to_soma_local(M.BaseFrame(dB["rest"], fr[f]))) for f in (K.WINDOW[0], 171, 185, 200, K.WINDOW[1])]
    g["no_mirroring"] = nm
    if not all(c["ok"] for c in nm):
        raise K.KimodoStop("REPRESENTATION_BLOCKED", "mirroring / heading check failed")
    # A == B (world, bit-equal) at every keyframe: the conditioning content is source-derived A, labelled PROXY_B
    fa, fb = _frames(rbA), _frames(rbB)
    g["A_equals_B_at_keyframes"] = all(fa[f]["world"] == fb[f]["world"] and fa[f]["matrix_world"] == fb[f]["matrix_world"] for f in K.KEY_ACTION)
    if not g["A_equals_B_at_keyframes"]:
        raise K.KimodoStop("DATA_BLOCKED", "A != B at a keyframe")
    pc = proxy_clip(mp, dB)
    np.savez(out / "proxyB_clip.npz", local77=pc["mats"], root_canonical=pc["root_canonical"], key_idx=np.array(K.key_indices()),
             soma_local=pc["local"], shift=pc["shift"], pos30=pc["pos30"])
    g["proxyB_clip_sha256"] = M.sha256_file(out / "proxyB_clip.npz")
    g["canonical_shift_m"] = pc["shift"].tolist()
    # (1) direct mapping roundtrip (no Kimodo features): decode == B keys, time identity
    dec = K.decode_edit_basis(mp, soma, pc["mats"], pc["root_canonical"])
    g["mapping_time_identity"] = K.time_identity(dec, dB["stored"])
    # (2) vendor feature roundtrip + conditioning contract (no model)
    steps.append(["encode", km(["encode", "--out-dir", out / "encode", "--inp", out / "proxyB_clip.npz", "--checkpoint-dir", CKPT_DIR],
                               out / "encode.log")])
    if steps[-1][1]["rc"] != 0:
        raise K.KimodoStop("CONDITIONING_BLOCKED", "vendor encode / constraint / mask contract failed (see encode.log)")
    e = load(out / "encode" / "encode.json")
    g["encode"] = e
    ez = np.load(out / "encode" / "encoded.npz")
    g["vendor_fk_vs_numpy_fk_input_max_m"] = float(np.abs(ez["fk30_input"] - pc["pos30"]).max())
    dec2 = K.decode_edit_basis(mp, soma, ez["local77_out"], ez["root_out"])
    np.savez(out / "proxyB_feature_decode.npz", **{ch: dec2[ch] for ch in K.EDIT_CHANNELS})
    g["feature_roundtrip_quat_err"] = {ch: float(max(np.abs(M.canon_sign(dec2[ch][i], dec[ch][i]) - dec[ch][i]).max() for i in range(K.N_WIN)))
                                       for ch in K.EDIT_CHANNELS}
    g["feature_time_identity"] = K.time_identity(dec2, dB["stored"], tol=FEATURE_QUAT_TOL)
    if g["vendor_fk_vs_numpy_fk_input_max_m"] > 1e-5:
        raise K.KimodoStop("REPRESENTATION_BLOCKED", "vendor FK of the conditioning != numpy FK")
    g["mask"] = e["mask"]
    return g, steps


FEATURE_QUAT_TOL = 1e-5      # float32 vendor feature roundtrip (cont6d + global->local); input-only measurement 1.5e-7 (disclosed)


def stage_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"stage": "preflight", "time": now(), "head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
           "origin_head": git("rev-parse", "origin/arch/single_frame_first"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "steps": []}
    w137 = (ROOT / Q.W137_WORKLOG).read_text(encoding="utf-8")
    rep["w137_complete"] = "AUTOMATION_BATCH_STATUS: COMPLETE" in w137 and "BATCH_NUMBER: 137" in w137
    rep["w138_worklog_absent"] = not any((ROOT / "docs").glob("138_*")) and not any((ROOT / "docs").glob("139_*"))
    anc = {c: subprocess.run(["git", "merge-base", "--is-ancestor", c, "HEAD"], cwd=str(ROOT)).returncode == 0 for c in PRIOR_COMMITS + (START_BATCH, START_RUN)}
    msha = M.sha256_file(ROOT / MATRIX)
    mx = load(ROOT / MATRIX)
    rep["selection"] = {"prior_batch_commits_ancestors": anc, "matrix_sha256": msha, "matrix_ok": msha == MATRIX_SHA,
                        "matrix_equals_evidence_copy": msha == M.sha256_file(QUAL_DIR / "selection_matrix.json"),
                        "selected": mx["selection"]["selected"], "declared_mode": mx["matrix"]["Kimodo"]["modes"]["declared_mode"],
                        "model_output_existed_at_selection": mx["model_output_exists"], "weights_downloaded_at_selection": mx["weights_downloaded"]}
    sel_ok = (all(anc.values()) and rep["selection"]["matrix_ok"] and rep["selection"]["matrix_equals_evidence_copy"]
              and mx["selection"]["selected"] == "Kimodo" and mx["selection"]["status"] == "SELECTED")
    rep["frozen"] = frozen_hashes()
    # code (pinned clone, unmodified)
    def gk(*args):
        return subprocess.run(["git", "-C", str(KM_ROOT), *args], capture_output=True, text=True).stdout.strip()
    code = {"repo": "https://github.com/nv-tlabs/kimodo", "pinned_commit": CODE_COMMIT, "head": gk("rev-parse", "HEAD"),
            "dirty": gk("status", "--porcelain", "--untracked-files=no"),
            "untracked_non_cache": [ln for ln in gk("status", "--porcelain").splitlines() if ln.startswith("??") and "__pycache__" not in ln],
            "files_sha256": {f: M.sha256_file(KM_ROOT / f) for f in CODE_FILES}}
    code["ok"] = code["head"] == CODE_COMMIT == mx["matrix"]["Kimodo"]["pinned_commit"] and code["dirty"] == "" and not code["untracked_non_cache"]
    rep["code"] = code
    # checkpoint (local, downloaded anonymously at the pinned revision) vs published LFS SHA-256 recorded at qualification
    api = load(QUAL_DIR / "sources" / "Kimodo" / "hf_api.json")
    sib = {s["rfilename"]: s for s in api.get("siblings", [])}
    local = {f: M.sha256_file(CKPT_DIR / f) for f in CKPT_FILES}
    lfs = {f: (sib.get(f, {}).get("lfs") or {}).get("sha256") for f in CKPT_FILES}
    small = {f: M.sha256_file(QUAL_DIR / "sources" / "Kimodo" / ("hf__" + f)) for f in ("LICENSE", "README.md", "config.yaml")}
    model = {"repo": HF_REPO, "revision": HF_REV, "api_sha": api.get("sha"), "gated": api.get("gated"), "local_sha256": local, "published_lfs_sha256": lfs,
             "small_files_equal_qualification_copy": {f: local[f] == small[f] for f in small},
             "weight_sha_equals_matrix": local["model.safetensors"] == mx["matrix"]["Kimodo"]["weight_lfs_sha256"]}
    model["ok"] = bool(api.get("sha") == HF_REV == mx["matrix"]["Kimodo"]["hf_revision"] and api.get("gated") is False
                       and all(local[f] == lfs[f] for f in CKPT_FILES if lfs[f]) and all(model["small_files_equal_qualification_copy"].values())
                       and model["weight_sha_equals_matrix"])
    rep["model"] = model
    rep["rights"] = {"code_licence": "Apache-2.0 (LICENSE at the pinned commit)", "weight_licence": mx["matrix"]["Kimodo"]["card_license"],
                     "determination": "PERMITTED_NONCOMMERCIAL_RESEARCH_EVALUATION (frozen W138 matrix E1c/E1w VERIFIED_TRUE; documentary reading by the agent, not legal advice)",
                     "caveats": mx["matrix"]["Kimodo"]["notes"]["rights_caveats"], "intended_use": mx["intended_use"],
                     "gated_text_encoder": mx["gated_text_encoder"], "access_judgement": mx["matrix"]["Kimodo"]["notes"]["access_judgement"],
                     "network": "none at inference (HF_HUB_OFFLINE=1, empty HF_HOME, no token in the environment, text-encoder URL unroutable)"}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    du = shutil.disk_usage(str(W138))
    rep["runtime"] = {"km_python": str(KM_PY), "freeze_sha256": M.sha256_file(W138 / "env" / "freeze.txt"), "nvidia_smi": smi,
                      "disk_free_gb": round(du.free / 1e9, 1), "blender": subprocess.run([H.BLENDER, "-b", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]}
    stop = None
    live_ok = (rep["instruction_blob"] == BLOB and rep["branch"] == "arch/single_frame_first" and rep["w137_complete"] and rep["w138_worklog_absent"])
    if not live_ok or not sel_ok:
        stop = "STATE_CHANGED"
    elif not all(v["ok"] for v in rep["frozen"].values()):
        stop = "DATA_BLOCKED"
    elif not (code["ok"] and model["ok"]):
        stop = "ACCESS_BLOCKED"
    elif du.free < 20e9:
        stop = "RESOURCE_BLOCKED"
    if stop is None:
        inp = new_dir(out / "inputs")
        for k, fk in COPY.items():
            shutil.copyfile(H6.FROZEN[fk][0], inp / f"{k}.blend")
        shutil.copyfile(rep["frozen"]["W136:run/U_S1.blend"]["path"], inp / "U.blend")
        shutil.copyfile(rep["frozen"]["W137:run/T_S1.blend"]["path"], inp / "T.blend")
        if any(M.sha256_file(inp / f"{k}.blend") != H6.FROZEN[fk][1] for k, fk in COPY.items()) or \
                M.sha256_file(inp / "U.blend") != rep["frozen"]["W136:run/U_S1.blend"]["expected"] or \
                M.sha256_file(inp / "T.blend") != rep["frozen"]["W137:run/T_S1.blend"]["expected"]:
            stop = "DATA_BLOCKED"
        for k in ("A", "B", "C", "D", "E", "U", "T"):
            rep["steps"].append([f"dump_{k}", bdump(inp / f"{k}.blend", out / f"dump_{k}.json", out / f"dump_{k}.log")])
            rep["steps"].append([f"readback_{k}", readback(inp / f"{k}.blend", out / f"readback_{k}.json", out / f"readback_{k}.log")])
        rep["steps"].append(["readback_Pin", readback(inp / "W126V_accepted.blend", out / "readback_Pin.json", out / "readback_Pin.log")])
        eq = {}
        for k, ref in (("A", "preflight/readback_A.json"), ("B", "preflight/readback_B.json"), ("C", "preflight/readback_C.json"),
                       ("D", "preflight/readback_D.json"), ("E", "preflight/readback_E.json"), ("Pin", "preflight/readback_Pin.json"),
                       ("U", "preflight/readback_U.json"), ("T", "run/readback_T.json")):
            w, f = load(W137R / ref), load(out / f"readback_{k}.json")
            eq[k] = all(p["world"] == q["world"] for p, q in zip(w["frames"], f["frames"])) and \
                all(p["world"] == q["world"] for p, q in zip(w["subframes"], f["subframes"])) and len(w["frames"]) == len(f["frames"]) == 240
        rep["fresh_world_bit_equal_to_W137"] = eq
        if not all(eq.values()):
            stop = "DATA_BLOCKED"
        if stop is None:
            dB, dA = load(out / "dump_B.json"), load(out / "dump_A.json")
            try:
                EP.check_dump(dB)
                rep["representation"], st = representation(out, dB, dA, load(out / "readback_A.json"), load(out / "readback_B.json"))
                rep["steps"] += st
            except K.KimodoStop as e:
                rep["representation"] = {"STOP": {"code": e.code, "detail": e.detail}}
                stop = {"CONDITIONING_BLOCKED": "CONDITIONING_BLOCKED", "DATA_BLOCKED": "DATA_BLOCKED"}.get(e.code, "REPRESENTATION_BLOCKED")
            except (M.AdapterStop, EP.PriorStop) as e:
                rep["representation"] = {"STOP": {"code": e.code, "detail": e.detail}}
                stop = "REPRESENTATION_BLOCKED"
            except (RuntimeError, OSError, KeyError, ValueError, IndexError) as e:
                rep["representation"] = {"STOP": {"code": "REPRESENTATION_BLOCKED", "detail": f"{type(e).__name__}: {e}"}}
                stop = "REPRESENTATION_BLOCKED"
        if stop is None:
            s2d = H6.source_2d({"A": load(out / "readback_A.json")}, load(out / "readback_A.json"))
            rep["source_2d_camera"] = {k: s2d[k] for k in ("camera", "fit_rms_px", "n_fit_points", "valid")}
    rep["stop"] = stop
    rep["status"] = "OK" if stop is None else stop
    dump(out / "preflight.json", rep)
    r = rep.get("representation", {})
    print(json.dumps({"status": rep["status"], "selection_ok": sel_ok, "live_ok": live_ok, "code_ok": code["ok"], "model_ok": model["ok"],
                      "frozen_bad": [k for k, v in rep["frozen"].items() if not v["ok"]], "fresh_eq": rep.get("fresh_world_bit_equal_to_W137"),
                      "representation": {k: r.get(k) for k in ("STOP", "numpy_vs_vendor_fk_max_m", "hip_scale", "hip_height_m", "mapping_time_identity",
                                                                "feature_roundtrip_quat_err", "feature_time_identity", "mask", "vendor_fk_vs_numpy_fk_input_max_m",
                                                                "canonical_shift_m", "A_equals_B_at_keyframes")},
                      "encode": {k: v for k, v in r.get("encode", {}).items() if k != "mask"},
                      "camera": rep.get("source_2d_camera", {}).get("fit_rms_px")}, indent=1, default=str))
    return 0 if stop is None else 3


# ------------------------------------------------------------------------------------------------ declaration ---
MECH_GATES = {
    "M0": "selection/identity/rights/access/resources preflight OK: frozen selection matrix (sha 4860bc9f..., committed in 4b64557) selects Kimodo in the no-text mode; pinned clean Kimodo clone e1c219c; local checkpoint files == published LFS SHA-256 at revision 0e3c890 (ungated); licence/README/config byte-identical to the qualification copies; every frozen SHA-256 (W137's 55 identities + W137 manifest + W137 T/Tc/records); fresh A..E/U/T/Pin readbacks world-bit-equal to W137's",
    "M1": "representation (no model): numpy FK on Kimodo's SOMA77 == vendor FK <= 1e-6 m; vendor 30<-77 slice == the 30 model joints; no mirroring; A == B at all 18 keyframes; PROXY_B direct mapping decode == B keys <= 1e-5 at shift 0 and >= 100x worse at +-1; vendor feature roundtrip (encode/normalise/unnormalise/inverse/30->77) decode == B keys <= 1e-5 at shift 0 and >= 100x worse at +-1; vendor FK of the conditioning == numpy FK <= 1e-5 m",
    "M2": "Action: K and Kc each a separate .blend with exactly 1 separately named Action; 32 CONSTANT F-curves keyed 1..240; scene 30 fps, 1..240",
    "M3": "footprint: changed stored keys vs A only upperarm_l/lowerarm_l rotation at Action 172..199; every non-left-arm world point and the object matrix bit-equal to A on all 240 integer frames and all subframes; left arm bit-equal outside 172..199 (incl. Action 171 and 200); no subframe change outside [172,200)",
    "M4": "conditioning: exactly one Kimodo._generate call via the official Kimodo.__call__ with an empty prompt (null encoder called once with [''], vendor zeroes the empty prompt); the motion_mask received by the sampler, validated BEFORE sampling, constrains exactly window rows 0..8 and 37..45 (Action 163..171, 200..208) and NO interior row; the received observed_motion equals the no-model vendor encode-stage observed_motion (<= 1e-5); raw keyframe adherence: mean joint-position error of the raw output over the 18 keyframes x 30 joints <= 0.05 m; interior not a PROXY_B copy (exact-equal joint fraction <= 0.5); the numpy 77-joint FK of the returned rotations equals the vendor 30-joint posed joints on the 26 non-hand-end joints <= 1e-4 m",
    "M5": "rig validity: upperarm/lowerarm world length range <= 0.1 mm over 171..200 for K; all imported quaternions finite and unit; no decoder refusal (proper rotations)",
    "M6": "hard pin: K composed onto the W126V-accepted .blend keeps every Action 132..140 hand_l residual equal to W126V's and <= 1e-6 m; pin frames bit-equal; composed footprint as M3 relative to W126V",
    "M7": "reversibility: K with A's interior keys written back (Krev) is world-bit-equal to A on every integer frame and subframe",
    "M8": "matched no-generator control: Kc (PROXY_B through the vendor feature roundtrip and the identical decode/import path) equals B within 1e-5 m on every frame",
    "M9": "integrity: every frozen SHA-256 unchanged after the run; local checkpoint SHA-256 unchanged",
    "M10": "negative controls (controls stage) all fail closed",
    "M11": "native-rate video: 240 frames written and decoded at 30.000 fps",
    "M12": "import fidelity: K's Blender-stored upperarm_l/lowerarm_l keys at 172..199 equal the imported quaternions <= 1e-6 AND the W129 basis recovered from K's evaluated pose matrices equals them <= 1e-5",
}
QUALITY = H7.QUALITY | {
    "Q1": "boundary continuity: K arm-local (clavicle_l frame) wrist AND elbow entry (171->172) and exit (199->200) steps each <= 2 x B's corresponding step + 10 mm",
    "Q2": "no reversal flicks (consecutive steps both > 20 mm with negative dot) of the arm-local wrist or elbow in 171->172 .. 199->200",
    "Q3": "interior smoothness: K arm-local wrist peak step <= 2 x B's + 10 mm AND K arm-local elbow peak step <= 2 x B's + 10 mm",
    "Q4": "rig plausibility (elbow flex/hinge): no interior frame near-straight (> 175 deg) and no interior frame bending opposite to A's dominant hinge normal (W129 quality)",
    "Q5": "source-2D fidelity: ONLY if the torso-only affine camera (A, Action 156..215) fits with RMS <= 20 px, K's interior mean 2D left-wrist error <= B's; the input-only fit is recorded here; > 20 px -> camera INVALID, source fidelity UNKNOWN, Q5 cannot pass",
    "Q6": "intended overhead press: K world wrist net rise (Z at 199 minus Z at 172) >= 0.5 x B's net rise (B's rise > 0)",
    "Q7": "agent still-frame review: K judged more natural AND more faithful to the source overhead press than B, and K not named for a flick, wrong elbow or detour",
    "Q8": "native-rate perceptual review of the 30 fps video by a human or an agent that actually perceives playback. This agent cannot perceive playback: declared NOT REVIEWED in advance, so Q8 cannot pass and CASE A is unattainable in this run",
}
ANCHORS = {
    "K_vs_U_and_K_vs_T": "IMPROVED_OVER_X iff K's arm-local wrist AND elbow max(entry, exit, peak) steps are each lower than X's AND K's total wrist+elbow flicks <= X's (X = W136 U, W137 T); reported findings, they do not raise the CASE",
    "unusable_diagnostic": "100 mm arm-local step = explicit unusable-continuity diagnostic (W136/W137), not an optimisation target",
}
RAW_RULE = ("raw full-body continuity is UNUSABLE iff (a) any raw elbow (LeftForeArm) or wrist (LeftHand) step in 171->172 .. 199->200 exceeds "
            "PROXY_B's step at the same transition by > 100 mm, or (b) any joint group's raw entry (171->172) or exit (199->200) step exceeds "
            "PROXY_B's at the same transition by > 100 mm. Raw = the vendor 30-joint posed joints of the returned features (rotation path; "
            "numpy 77-joint FK cross-check on the 26 joints unaffected by the vendor relaxed-hand insertion <= 1e-4 m) in the canonical model "
            "space (metres, Kimodo SOMA proportions); PROXY_B = the same 30-joint FK of the conditioning clip. NOTE: unlike W137, "
            "raw frame 171/200 are the model's own (soft-constrained) poses, not exact copies.")
ATTRIBUTION_RULE = H7.ATTRIBUTION_RULE + " (W138: projected = PROXY_B 77-joint locals with only LeftArm/LeftForeArm local rotations from the raw output, Kimodo FK, model space.)"
EFFECT_RULE = ("GENERATED_EFFECT iff max over Action 172..199 of max(|elbow_K - elbow_Kc|, |wrist_K - wrist_Kc|) (world, fresh readback) "
               ">= 10 mm; otherwise NO_GENERATED_EFFECT")
CASE_RULE = {
    "STOP": "preflight not OK (STATE_CHANGED / DATA_BLOCKED / ACCESS_BLOCKED / RESOURCE_BLOCKED / REPRESENTATION_BLOCKED / CONDITIONING_BLOCKED) or the vendor conditioning refuses the declared keyframes before any sample exists -> the named STOP",
    "D": "declaration/instrument changed or model output preceded the declaration (DECLARATION_INVALID); a negative control does not fail closed (M10); integrity (M9) fails; the matched control Kc != B (M8) (INTEGRITY_FAILED)",
    "C": "real inference executed but: decoder/projection refusal (RIG_PROJECTION_FAILED), M4 fails, any of M1-run, M2, M3, M5, M6, M7, M11, M12 fails, projected unusable continuity (any K arm-local wrist or elbow entry/exit/peak step in 171->172 .. 199->200 > 100 mm), or raw full-body continuity UNUSABLE (RAW_RULE)",
    "B": "a valid model Action exists (all mechanical gates pass) but NO_GENERATED_EFFECT or any of Q1-Q8 fails (improvement absent, unproven or review incomplete)",
    "A": "all mechanical gates pass AND GENERATED_EFFECT AND Q1-Q8 pass (unattainable here: Q8 declared NOT REVIEWED)",
    "precedence": "STOP > D > C > B > A; a mechanical PASS, synthetic PASS or a video file alone never yields A",
}


def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit(f"preflight not OK: {pf.get('stop')}")
    pdir = Path(a.preflight).parent
    mo = model_outputs_exist()
    r = pf["representation"]
    d = {"schema": "animcv_exp_w138_declaration_v1", "time": now(), "outcomes_inspected": False, "model_output_exists": bool(mo),
         "batch": BATCH, "instruction_blob": BLOB, "batch_issue_commit": START_BATCH, "run_starting_commit": START_RUN,
         "resumed_from_commits": list(PRIOR_COMMITS), "repo_head_at_declaration": git("rev-parse", "HEAD"),
         "preflight_sha256": M.sha256_file(a.preflight), "selection_matrix": {"path": MATRIX, "sha256": MATRIX_SHA, "selected": "Kimodo",
                                                                               "mode": "no_text_constraints_only"},
         "bindings": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in pf["frozen"].items()},
         "fresh_dump_sha256": {k: M.sha256_file(pdir / f"dump_{k}.json") for k in ("A", "B", "C", "D", "E", "U", "T")},
         "fresh_readback_sha256": {k: M.sha256_file(pdir / f"readback_{k}.json") for k in ("A", "B", "C", "D", "E", "U", "T", "Pin")},
         "identities": {"code": {"repo": pf["code"]["repo"], "commit": CODE_COMMIT, "licence": "Apache-2.0", "files_sha256": pf["code"]["files_sha256"]},
                        "model": {"repo": HF_REPO, "revision": HF_REV, "local_dir": str(CKPT_DIR), "files_sha256": pf["model"]["local_sha256"],
                                  "licence": pf["rights"]["weight_licence"], "gated": False},
                        "text_encoder": "NONE: null encoder (zeros of the vendor shape [B,1,4096], lengths 1) through the documented load_model(text_encoder=...) parameter; prompt ''; the gated meta-llama/Meta-Llama-3-8B-Instruct base and the LLM2Vec adapters are never requested",
                        "runtime": {"python": str(KM_PY), "freeze_sha256": pf["runtime"]["freeze_sha256"], "torch": r["kimodo_skeleton"]["torch"],
                                    "cuda": r["kimodo_skeleton"]["cuda"], "device": r["kimodo_skeleton"]["device"], "nvidia_smi": pf["runtime"]["nvidia_smi"]},
                        "skeleton_npz_sha256": M.sha256_file(pdir / "skeleton" / "skeleton.npz"), "proxyB_clip_sha256": r["proxyB_clip_sha256"],
                        "encode_sha256": {"encode.json": M.sha256_file(pdir / "encode" / "encode.json"), "encoded.npz": M.sha256_file(pdir / "encode" / "encoded.npz"),
                                          "constraints.json": M.sha256_file(pdir / "encode" / "constraints.json")}},
         "rights": pf["rights"],
         "time_rule": "native 30 fps; Action = SOURCE - 78; generated window Action 163..208 (SOURCE 241..286, 46 frames), index i <-> Action 163+i; "
                      "keyframes = indices 0..8 and 37..45 (Action 163..171, 200..208; 18 < 20); interior Action 172..199 = SOURCE 250..277 = indices 9..36 "
                      "(28 frames) carries NO constraint; Action 171 and 200 stay frozen in the Blender Action",
         "proxy": {"label": "PROXY_B", "what": "the frozen W129 B Action (endpoint slerp inside 172..199; == A elsewhere, bit-equal at every keyframe) "
                   "mapped by the unchanged W129 closed-form Mapping onto Kimodo's own SOMA77 (identity rest, neutral-joint offsets, hip height "
                   f"{r['hip_height_m']:.4f} m, hip scale {r['hip_scale']:.6f}); a GT-free source-derived hypothesis, never observed 3D truth. The "
                   "far-left-arm 3D of the source is UNKNOWN. BaseRig channels undriven in the frozen Action (right arm, neck, etc.) are rig rest; SOMA "
                   "joints with no BaseRig counterpart (Jaw, eyes, finger/toe ends) are SOMA rest (SYNTHETIC).",
                   "conditioning_content": "vendor FullBodyConstraintSet: all 30 SOMA30 joint positions (vendor FK of the vendor 77->30 slice), the "
                                           "default smooth root 2D (= root XZ), root height and the derived global heading at the 18 keyframes ONLY. "
                                           "Interior: nothing (no position, rotation, root, heading or end-effector).",
                   "canonical": f"root XZ translated to (0,0) at window frame 0 (shift {r['canonical_shift_m']}); first_heading_angle = vendor "
                                f"compute_heading_angle at frame 0 ({r['encode']['first_heading_angle_rad']:.6f} rad); no rotation of the data"},
         "inference": {"path": "kimodo.load_model('Kimodo-SOMA-RP-v1.2', device='cuda:0', text_encoder=<null>) with CHECKPOINT_DIR = the local verified "
                               "checkpoint parent, HF offline; ONE Kimodo.__call__('', 46, num_denoising_steps=100, constraint_lst=[fullbody], "
                               "cfg_weight=[2.0, 2.0], cfg_type='separated', num_samples=None, first_heading_angle=<frame-0 heading>, post_processing=False)",
                       "seed": SEED, "steps": STEPS, "cfg": {"type": "separated", "weight": list(CFG_W), "source": "vendor defaults (Kimodo.__call__ default, configuration guide)"},
                       "seed_note": "seed 10 = the W136/W137 seed, chosen for consistency, not tuned; kimodo.tools.seed_everything(10) immediately before the call",
                       "post_processing": "OFF (declared): the vendor optional post-processor needs a C++ extension and would alter the raw generator output; "
                                          "AnimCV-side preservation never depends on the model",
                       "one_run": True, "recorder": "Kimodo._generate wrapped read-only: validates the received motion_mask BEFORE sampling, counts calls "
                       "(a second call raises), records observed_motion/mask/heading and the returned features; identical arguments forwarded",
                       "no": "no seed/prompt/cfg/step/window/keyframe sweep, no second inference, no training/finetuning, no post-hoc smoothing/IK/blending/clamp/threshold change"},
         "mapping": {"skeleton": "Kimodo somaskel77 (names, parents, neutral joints from the pinned package assets)", "mapping": r["mapping"],
                     "numpy_vs_vendor_fk_max_m": r["numpy_vs_vendor_fk_max_m"], "mapping_time_identity": r["mapping_time_identity"],
                     "feature_time_identity": r["feature_time_identity"], "feature_roundtrip_quat_err": r["feature_roundtrip_quat_err"],
                     "decode": "output local_rot_mats (vendor somaskel77 via to_SOMASkeleton77) -> W129 base_basis_from_soma for upperarm_l (LeftArm) and lowerarm_l (LeftForeArm)",
                     "mask_contract": r["mask"]},
         "import": {"channels": list(K.EDIT_CHANNELS), "frames": list(K.INTERIOR), "values": "decoded K basis quaternions, unit-norm, sign-canonical to A's key",
                    "builder": "unchanged W129 build_variant (refuses any other channel/frame), new opt-in Action from a byte copy of A",
                    "not_imported": "every other generated joint and every frame outside 172..199 (incl. the generated keyframe frames 163..171, 200..208)"},
         "control": "Kc = PROXY_B through the vendor feature roundtrip (preflight encode stage) and the identical decode/import path (expected == B within 1e-5 m, M8)",
         "historical": "A, B, C (W129), D, E (W135), W136 U and W137 T are read-only frozen references; no model is re-run for them",
         "domain": {"channels": list(K.EDIT_CHANNELS), "frames": list(K.INTERIOR), "pin_action": [K.PIN_ACTION[0], K.PIN_ACTION[-1]],
                    "pin_target_m": [-0.0927, 0.2199, 1.4506], "pin_tolerance_m": 0.001},
         "mechanical_gates": MECH_GATES, "generated_effect_rule": EFFECT_RULE, "quality_criteria": QUALITY, "anchors": ANCHORS,
         "raw_rule": RAW_RULE, "attribution_rule": ATTRIBUTION_RULE, "unusable_step_mm": K.UNUSABLE_STEP_MM, "adherence_mean_max_m": K.ADHERENCE_MEAN_MAX_M,
         "case_rule": CASE_RULE, "source_2d_camera_input_only": pf.get("source_2d_camera"),
         "metrics": ["arm-local wrist (W129 armlocal_quality) and elbow (W135 elbow_metrics) entry/exit/peak/flicks for A,B,C,D,E,U,T,K,Kc",
                     "world wrist path, flex range, hinge (W129 quality); wrist net rise; deviation from B", "K vs Kc (generated effect); K vs U; K vs T",
                     "raw full body (model space): per joint group interior deviation from PROXY_B, entry/exit/peak steps and excess; keyframe adherence; "
                     "root drift; positions block vs rotations; neck height (collapse)", "raw vs projected (model space)",
                     "world shoulder step (shared, source-derived parent popping)", "source-2D torso camera (diagnostic only)"],
         "review_protocol": "UNBLINDED agent review of fixed-time contact sheets (entry 170-175, middle 183-188, exit 196-201, overview 171..198 step 3) with "
                            "source crop, A, B, C, D, E, U, T, K; still frames only; written to a review JSON before assess. The 30 fps video is encoded and "
                            "machine-decoded but NOT perceived at native rate by this agent (Q8 NOT REVIEWED). No human/animator review.",
         "stop_conditions": ["STATE_CHANGED", "RIGHTS_UNRESOLVED", "ACCESS_BLOCKED", "RESOURCE_BLOCKED", "DATA_BLOCKED", "REPRESENTATION_BLOCKED",
                             "CONDITIONING_BLOCKED", "DECLARATION_INVALID", "RIG_PROJECTION_FAILED", "INTEGRITY_FAILED", "no tuning or second run after any model output"],
         "pre_declaration_disclosures": [
             "resumed Batch: the qualification instrument (69d2215) and the frozen selection matrix (4b64557) were committed by an earlier, interrupted run "
             "of this Batch and are treated as binding; the checkpoint was downloaded in this run (anonymous, pinned revision) after the selection and "
             "verified against the published LFS SHA-256 before any use",
             "input-only measurements before this declaration (no model): numpy-vs-vendor FK, the direct mapping roundtrip and the vendor float32 "
             f"feature roundtrip (max quaternion error {max(r['feature_roundtrip_quat_err'].values()):.2e}); the feature-roundtrip tolerance 1e-5 (= the W136/W137 "
             "quaternion tolerance) and the M8 tolerance 1e-5 m (= W137) were confirmed after that measurement",
             "the window (Action 163..208), the 9+9 keyframe split, no post-processing and the 0.05 m adherence bound were chosen before any output from "
             "the vendor documentation (sparse keyframes < 20; fullbody in-betweening benchmarked in the no-text mode)",
             "the torso-only affine camera is known from W136/W137 to fail its 20 px gate; re-fitted on A only (input-only) and recorded here"],
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    if d["model_output_exists"]:
        raise SystemExit(f"STOP DECLARATION_INVALID: model output exists {mo}")
    sha = dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha, "keyframes": list(K.KEY_ACTION), "seed": SEED, "steps": STEPS,
                      "camera_rms_px": (pf.get("source_2d_camera") or {}).get("fit_rms_px")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ controls ---
def _expect(name, fn, codes, res):
    codes = (codes,) if isinstance(codes, str) else tuple(codes)
    try:
        fn()
        res[name] = {"pass": False, "note": "NOT refused"}
    except (K.KimodoStop, U.UniStop, M.AdapterStop) as e:
        res[name] = {"pass": e.code in codes, "code": e.code, "expected": list(codes)}


def stage_controls(a):
    d = check_declaration(a)
    out = new_dir(a.out_dir)
    pdir = Path(a.preflight).parent
    load_soma(pdir / "skeleton" / "skeleton.npz")
    neg = {}
    E = _expect
    keys = K.key_indices()
    K.check_keyframes(keys)
    for nm, k in (("keys_shifted_plus1", [i + 1 for i in keys]), ("keys_include_172", sorted(keys + [9])), ("keys_include_199", sorted(keys + [36])),
                  ("keys_missing_171", [i for i in keys if i != 8]), ("keys_missing_200", [i for i in keys if i != 37]),
                  ("keys_20_frames", sorted(keys + [9, 36])),
                  ("keys_unsorted", list(reversed(keys))), ("keys_duplicate", keys[:1] + keys), ("keys_out_of_window", keys[:-1] + [46]),
                  ("keys_dense_whole_window", list(range(46)))):
        E(nm, lambda k=k: K.check_keyframes(k), ("KEYFRAMES_FORMAT", "KEYFRAMES_DOMAIN"), neg)
    good = np.load(pdir / "encode" / "encoded.npz")["motion_mask"].astype(bool)
    K.validate_motion_mask(good)
    for nm, fn in (("mask_interior_row_172", lambda m: _row(m, 9, True)), ("mask_missing_key_171", lambda m: _row(m, 8, False)),
                   ("mask_missing_key_200", lambda m: _row(m, 37, False)), ("mask_shift_plus1", lambda m: np.roll(m, 1, axis=0)),
                   ("mask_all_rows", lambda m: np.ones_like(m)), ("mask_none", lambda m: np.zeros_like(m)), ("mask_45_rows", lambda m: m[:45]),
                   ("mask_float_values", lambda m: m.astype(float) * 0.5)):
        m2 = fn(good.copy())
        E(nm, lambda m2=m2: K.validate_motion_mask(m2), ("MASK_DOMAIN", "MASK_EMPTY", "MASK_SHAPE", "MASK_DTYPE"), neg)
    E("text_prompt_requested", lambda: K.check_null_text(["A person presses overhead."]), "TEXT_MODE_BLOCKED", neg)
    E("text_prompt_whitespace_ok_but_mixed", lambda: K.check_null_text(["", "x"]), "TEXT_MODE_BLOCKED", neg)
    E("fps_25", lambda: K.check_fps_frames(25, K.N_WIN), "NOT_30_FPS", neg)
    E("window_45_frames", lambda: K.check_fps_frames(30, 45), "WINDOW_LENGTH", neg)
    pc = np.load(pdir / "proxyB_clip.npz")
    E("root_not_canonical", lambda: K.check_canonical(pc["root_canonical"] + np.array([0.1, 0.0, 0.0])), "CANONICAL_ORIGIN", neg)
    E("window_action_outside", lambda: K.action_to_index(162), "WINDOW_DOMAIN", neg)
    # representation controls on the real PROXY_B clip (no model)
    dB = load(pdir / "dump_B.json")
    mp = mapping_for(dB)
    dec = K.decode_edit_basis(mp, SOMA, pc["local77"], pc["root_canonical"])
    K.time_identity(dec, dB["stored"])
    E("decode_time_shift_plus1", lambda: K.time_identity({c: np.roll(v, 1, axis=0) for c, v in dec.items()}, dB["stored"]),
      ("REPRESENTATION_BLOCKED", "TIME_SHIFT"), neg)
    sw = pc["local77"].copy()
    il, ir = SOMA.idx["LeftArm"], SOMA.idx["RightArm"]
    sw[:, [il, ir]] = sw[:, [ir, il]]
    E("decode_left_right_swap", lambda: K.time_identity(K.decode_edit_basis(mp, SOMA, sw, pc["root_canonical"]), dB["stored"]),
      ("REPRESENTATION_BLOCKED", "TIME_SHIFT"), neg)
    E("decode_parent_child_swap", lambda: K.time_identity(K.decode_edit_basis(mp, SOMA, _swap(pc["local77"], SOMA.idx["LeftArm"], SOMA.idx["LeftForeArm"]),
                                                                              pc["root_canonical"]), dB["stored"]), ("REPRESENTATION_BLOCKED", "TIME_SHIFT"), neg)
    E("mapping_mirrored_axis", lambda: M.Mapping(SOMA, M.rest_frame(dB["rest"], H.OBJECT_REST_WXYZ, H.OBJECT_SCALE),
                                                 A=np.diag([-1.0, 1.0, 1.0]) @ M.A_BLENDER_TO_SOMA), "AXIS_MAPPING_IMPROPER", neg)
    bad = pc["local77"][0].copy()
    bad[SOMA.idx["LeftArm"]] *= -1.0
    E("improper_rotation_refused", lambda: K.soma_local_from_mats(SOMA, bad, pc["root_canonical"][0]), "RIG_PROJECTION_FAILED", neg)
    nan = pc["local77"][0].copy()
    nan[SOMA.idx["LeftForeArm"], 0, 0] = np.nan
    E("nonfinite_rotation_refused", lambda: K.soma_local_from_mats(SOMA, nan, pc["root_canonical"][0]), "RIG_PROJECTION_FAILED", neg)
    E("soma_missing_joint", lambda: M.SomaSkeleton([n if n != "LeftForeArm" else "X" for n in SOMA.names], SOMA.parents, SOMA.ref),
      "SOMA_SKELETON_MISSING_JOINTS", neg)
    # import-domain controls
    dA = load(pdir / "dump_A.json")
    vals = K.import_values(dec, dA["stored"])
    E("import_frame_171", lambda: U.check_domain([171] + list(range(172, 200))), "DOMAIN_FRAMES", neg)
    E("import_frame_200", lambda: U.check_domain(list(range(172, 201))), "DOMAIN_FRAMES", neg)
    E("import_channel_clavicle", lambda: U.check_domain(range(172, 200), ["upperarm_l", "clavicle_l"]), "DOMAIN_CHANNEL", neg)
    E("import_pin_frames", lambda: U.check_domain(range(132, 141)), "PIN_OVERLAP", neg)
    z0 = {c: v.copy() for c, v in dec.items()}
    z0["lowerarm_l"][K.action_to_index(180)] = 0.0
    E("import_zero_quaternion", lambda: K.import_values(z0, dA["stored"]), "RIG_PROJECTION_FAILED", neg)
    # conditioning detectors on the real PROXY_B positions (synthetic 'raw' variants, no model)
    p30 = pc["pos30"]
    fake = p30.copy()
    fake[K.interior_indices()] += 0.01
    K.check_conditioning(K.conditioning_report(fake, p30))
    far = fake.copy()
    far[K.key_indices()] += 0.06
    E("keyframes_ignored", lambda: K.check_conditioning(K.conditioning_report(far, p30)), "CONDITIONING_FIDELITY", neg)
    E("interior_copied_from_proxyB", lambda: K.check_conditioning(K.conditioning_report(p30.copy(), p30)), "INTERIOR_COPIED", neg)
    E("positions_45_frames", lambda: K.conditioning_report(p30[:45], p30[:45]), "POSITION_SHAPE", neg)
    # unauthorised-channel build refused by the unchanged W129 Blender build_variant (real Blender, no save)
    spec = {"out_blend": str(out / "NEG_clavicle.blend"), "interior_action_frames": list(K.INTERIOR),
            "values_wxyz": {**vals, "clavicle_l": vals["upperarm_l"]}, "branch": "NEG", "action_name": "NEG", "expected_blend_sha256": H6.FROZEN["A"][1],
            "note": "negative control"}
    dump(out / "neg_clavicle_spec.json", spec)
    try:
        blender(H6.STAGE, "build_variant", out / "neg_clavicle.log", blend=pdir / "inputs" / "A.blend", spec=out / "neg_clavicle_spec.json",
                report=out / "neg_clavicle_report.json")
        neg["blender_build_unauthorised_channel"] = {"pass": False, "note": "NOT refused"}
    except RuntimeError:
        neg["blender_build_unauthorised_channel"] = {"pass": not (out / "NEG_clavicle.blend").exists(), "code": "BLENDER_REFUSED"}
    spec2 = dict(spec, values_wxyz={ch: {**v, "171": v["172"]} for ch, v in vals.items()}, out_blend=str(out / "NEG_171.blend"))
    dump(out / "neg_171_spec.json", spec2)
    try:
        blender(H6.STAGE, "build_variant", out / "neg_171.log", blend=pdir / "inputs" / "A.blend", spec=out / "neg_171_spec.json",
                report=out / "neg_171_report.json")
        neg["blender_build_frame_171"] = {"pass": False, "note": "NOT refused"}
    except RuntimeError:
        neg["blender_build_frame_171"] = {"pass": not (out / "NEG_171.blend").exists(), "code": "BLENDER_REFUSED"}
    res = {"negative_controls": neg, "n": len(neg), "all_fail_closed": all(v["pass"] for v in neg.values()),
           "note": "no model was loaded or run in this stage", "declaration_sha256": a.declaration_sha}
    dump(out / "controls.json", res)
    print(json.dumps({"all_fail_closed": res["all_fail_closed"], "n": res["n"], "failed": [k for k, v in neg.items() if not v["pass"]]}, indent=1))
    return 0 if res["all_fail_closed"] else 3


def _row(m, i, v):
    m[i, :] = v
    return m


def _swap(x, i, j):
    y = x.copy()
    y[:, [i, j]] = y[:, [j, i]]
    return y


# ------------------------------------------------------------------------------------------------ run ---
def stage_run(a):
    d = check_declaration(a)
    ctl = load(a.controls)
    if not ctl["all_fail_closed"]:
        raise SystemExit("STOP INTEGRITY_FAILED: controls did not fail closed")
    pf = load(a.preflight)
    pdir = Path(a.preflight).parent
    if model_outputs_exist():
        raise SystemExit(f"STOP DECLARATION_INVALID: a model output already exists {model_outputs_exist()}")
    out = new_dir(Path(a.out_dir))
    log = {"stage": "run", "start": now(), "steps": []}
    inp = pdir / "inputs"
    for k, fk in COPY.items():
        if M.sha256_file(inp / f"{k}.blend") != H6.FROZEN[fk][1]:
            raise SystemExit(f"STOP INTEGRITY_FAILED {k}")
    for k in ("U", "T"):
        if M.sha256_file(inp / f"{k}.blend") != d["bindings"][{"U": "W136:run/U_S1.blend", "T": "W137:run/T_S1.blend"}[k]]["sha256"]:
            raise SystemExit(f"STOP INTEGRITY_FAILED {k}")
    for k, h in d["fresh_dump_sha256"].items():
        if M.sha256_file(pdir / f"dump_{k}.json") != h:
            raise SystemExit(f"STOP INTEGRITY_FAILED dump {k}")
    if {f: M.sha256_file(CKPT_DIR / f) for f in CKPT_FILES} != d["identities"]["model"]["files_sha256"] or \
            M.sha256_file(pdir / "proxyB_clip.npz") != d["identities"]["proxyB_clip_sha256"]:
        raise SystemExit("STOP INTEGRITY_FAILED checkpoint/clip")
    soma, _ = load_soma(pdir / "skeleton" / "skeleton.npz")
    dumps = {k: load(pdir / f"dump_{k}.json") for k in ("A", "B", "C", "D", "E", "U", "T")}
    mp = mapping_for(dumps["B"])
    pc = np.load(pdir / "proxyB_clip.npz")
    res = {"declaration_sha256": a.declaration_sha}
    # ---- ONE real inference ----
    log["steps"].append(["infer", km(["infer", "--out-dir", out / "model", "--inp", pdir / "proxyB_clip.npz", "--checkpoint-dir", CKPT_DIR,
                                      "--seed", SEED, "--steps", STEPS], out / "infer.log")])
    txt = (out / "infer.log").read_text(encoding="utf-8", errors="replace")
    executed = (out / "model" / "raw_model_io.npz").is_file()
    if log["steps"][-1][1]["rc"] != 0 or not (out / "model" / "infer_record.json").is_file():
        code = next((c for c in ("MASK_DOMAIN", "MASK_EMPTY", "MASK_SHAPE", "MASK_DTYPE", "KEYFRAMES_DOMAIN", "TEXT_MODE_BLOCKED", "CANONICAL_ORIGIN",
                                 "NOT_30_FPS", "WINDOW_LENGTH") if c in txt), "INFERENCE_FAILED")
        res["STOP"] = {"code": "CONDITIONING_BLOCKED" if (not executed and code != "INFERENCE_FAILED") else ("RIG_PROJECTION_FAILED" if executed else code),
                       "detail": code, "model_executed": executed, "log_tail": txt.strip().splitlines()[-8:]}
        dump(out / "run_result.json", res)
        dump(out / "run_log.json", log)
        print(json.dumps(res, indent=1))
        return 3
    ir = load(out / "model" / "infer_record.json")
    raw = np.load(out / "model" / "raw_model_io.npz")
    res["inference"] = {k: ir.get(k) for k in ("calls", "mask_validate", "sampler_seconds", "call_seconds", "load_seconds", "total_seconds", "peak_vram_bytes",
                                               "text_encoder_calls", "device", "torch", "tf32_matmul", "first_heading_angle_rad", "args", "model_fps",
                                               "model_skeleton", "text_encoder_type", "output_shapes", "raw_model_io_sha256")}
    enc = np.load(pdir / "encode" / "encoded.npz")
    obs_eq = float(np.abs(raw["in_observed_motion"][0] - enc["observed_motion"]).max())
    # ---- raw output in model space ----
    # raw30 = the vendor's own 30-joint view (inverse of the returned features, rotation path) -- the representation the constraint
    # positions were built in.  The vendor 77-joint output inserts relaxed-hand finger rotations, which move only the four
    # hand-end joints; the numpy 77-joint FK cross-check therefore covers the other 26 joints.
    raw_loc = np.stack([K.soma_local_from_mats(soma, raw["out_local77"][i], raw["out_root"][i]) for i in range(K.N_WIN)])
    raw30_np = np.stack([K.positions30(soma, x) for x in raw_loc])
    raw30 = np.asarray(raw["out_posed30_rot"], float)
    body26 = [i for i, j in enumerate(K.SOMA30) if not j.endswith(("ThumbEnd", "MiddleEnd"))]
    vendor_vs_numpy = float(np.abs(raw30_np[:, body26] - raw30[:, body26]).max())
    proxy30 = pc["pos30"]
    cond = K.conditioning_report(raw30, proxy30)
    try:
        cond_ok = K.check_conditioning(cond)
    except K.KimodoStop as e:
        cond_ok = False
        res["conditioning_stop"] = {"code": e.code, "detail": e.detail}
    m4 = bool(ir["calls"] == 1 and ir["mask_validate"]["constrained_rows"] == K.key_indices() and ir["text_encoder_calls"] == [[""]]
              and obs_eq <= 1e-5 and vendor_vs_numpy <= 1e-4 and cond_ok and ir["args"]["prompt"] == "" and not ir["args"]["post_processing"])
    res["conditioning"] = {"report": cond, "observed_motion_vs_encode_max": obs_eq, "numpy_fk_vs_vendor_posed30_max_m": vendor_vs_numpy, "M4": m4}
    # ---- decode + import ----
    try:
        decK = K.decode_edit_basis(mp, soma, raw["out_local77"], raw["out_root"])
        decKc = {ch: np.load(pdir / "proxyB_feature_decode.npz")[ch] for ch in K.EDIT_CHANNELS}
        res["M1_run_feature_time_identity_Kc"] = K.time_identity(decKc, dumps["B"]["stored"], tol=FEATURE_QUAT_TOL)
        vK = K.import_values(decK, dumps["A"]["stored"])
        vKc = K.import_values(decKc, dumps["A"]["stored"])
    except (K.KimodoStop, U.UniStop, M.AdapterStop) as e:
        res["STOP"] = {"code": "RIG_PROJECTION_FAILED" if e.code != "TIME_SHIFT" else e.code, "detail": f"{e.code}: {e.detail}", "model_executed": True}
        dump(out / "run_result.json", res)
        print(json.dumps(res, indent=1))
        return 3
    dump(out / "import_K.json", vK)
    dump(out / "import_Kc.json", vKc)
    # ---- raw full-body analysis (model space) ----
    hyb_loc = K.hybrid_local(pc["soma_local"], raw_loc, soma)
    hyb30 = np.stack([K.positions30(soma, x) for x in hyb_loc])
    att = K.attribution(raw30, proxy30, hyb30)
    body = K.body_report(raw30, proxy30)
    rawu = K.raw_unusable(att, body)
    root_drift = np.linalg.norm(raw30[:, 0] - proxy30[:, 0], axis=1) * 1e3
    fi = K.interior_indices()
    rawan = {"attribution": att, "body": body, "raw_rule": rawu,
             "root_drift_mm": {"per_frame": root_drift.tolist(), "max_interior": float(root_drift[fi].max()), "at_action_171": float(root_drift[K.action_to_index(171)]),
                               "at_action_200": float(root_drift[K.action_to_index(200)]), "max_keyframes": float(root_drift[K.key_indices()].max())},
             "positions_block_vs_rotations_mm": {"interior_max": float(np.linalg.norm(raw["out_posed30_pos"][fi] - raw["out_posed30_rot"][fi], axis=-1).max() * 1e3),
                                                 "keyframes_max": float(np.linalg.norm(raw["out_posed30_pos"][K.key_indices()] - raw["out_posed30_rot"][K.key_indices()], axis=-1).max() * 1e3)},
             "neck_height_mm": {"raw": K.neck_height_mm(raw30), "proxyB": K.neck_height_mm(proxy30)},
             "space": "Kimodo canonical model space (metres, Kimodo SOMA proportions); hip scale vs BaseRig " + f"{mp.hip_scale:.6f}"}
    np.savez(out / "raw_model_space.npz", raw30=raw30, proxy30=proxy30, hybrid30=hyb30, raw_local=raw_loc, names30=np.array(K.SOMA30))
    # ---- builds ----
    blends = {k: inp / f"{k}.blend" for k in ("A", "B", "C", "D", "E", "U", "T")}
    for br, vals, base, base_sha, nm in (("K", vK, inp / "A.blend", H6.FROZEN["A"][1], "AnimCV_EXP W138 K Kimodo notext S1"),
                                         ("Kc", vKc, inp / "A.blend", H6.FROZEN["A"][1], "AnimCV_EXP W138 Kc PROXY_B control S1"),
                                         ("Kpin", vK, inp / "W126V_accepted.blend", H6.FROZEN["W126V_accepted"][1], "AnimCV_EXP W138 K_on_W126V S1")):
        spec = {"out_blend": str(out / f"{br}_S1.blend"), "interior_action_frames": list(K.INTERIOR), "values_wxyz": vals, "branch": f"W138 {br}",
                "action_name": nm, "expected_blend_sha256": base_sha,
                "note": "W138 scratch: " + ("Kimodo-SOMA-RP-v1.2 no-text constraints-only generated upperarm_l/lowerarm_l at 172..199 (model-inferred)"
                                            if br != "Kc" else "PROXY_B through the vendor feature roundtrip and the identical decode/import path (no-generator control)")}
        dump(out / f"build_{br}_spec.json", spec)
        log["steps"].append([f"build_{br}", blender(H6.STAGE, "build_variant", out / f"build_{br}.log", blend=base, spec=out / f"build_{br}_spec.json",
                                                     report=out / f"build_{br}_report.json")])
        blends[br] = out / f"{br}_S1.blend"
    stA = dumps["A"]["stored"]
    vA = {ch: {str(f): [stA[ch][str(c)][str(f)] for c in range(4)] for f in range(K.INTERIOR[0], K.INTERIOR[1] + 1)} for ch in K.EDIT_CHANNELS}
    spec = {"out_blend": str(out / "Krev_S1.blend"), "interior_action_frames": list(K.INTERIOR), "values_wxyz": vA, "branch": "W138 Krev",
            "action_name": "AnimCV_EXP W138 K_reverted S1", "expected_blend_sha256": M.sha256_file(blends["K"]), "note": "reversibility control"}
    dump(out / "build_Krev_spec.json", spec)
    log["steps"].append(["build_Krev", blender(H6.STAGE, "build_variant", out / "build_Krev.log", blend=blends["K"], spec=out / "build_Krev_spec.json",
                                                report=out / "build_Krev_report.json")])
    blends["Krev"] = out / "Krev_S1.blend"
    rb = {k: load(pdir / f"readback_{k}.json") for k in ("A", "B", "C", "D", "E", "U", "T", "Pin")}
    for br in ("K", "Kc", "Kpin", "Krev"):
        log["steps"].append([f"readback_{br}", readback(blends[br], out / f"readback_{br}.json", out / f"readback_{br}.log")])
        rb[br] = load(out / f"readback_{br}.json")
    for br in ("K", "Kc"):
        log["steps"].append([f"dump_{br}", bdump(blends[br], out / f"dump_{br}.json", out / f"dump_{br}.log")])
        dumps[br] = load(out / f"dump_{br}.json")
    # ---- mechanical gates ----
    g = {}
    i0, i1 = K.INTERIOR
    for br in ("K", "Kc"):
        r = rb[br]
        pres = M.preservation(r, rb["A"], K.INTERIOR)
        ck = M.changed_keys(dumps[br]["stored"], dumps["A"]["stored"])
        census = r["fcurve_census"]
        m2 = (len(r["actions_in_file"]) == 1 and r["action_name"] not in {rb[k]["action_name"] for k in ("A", "B", "C", "D", "E", "U", "T")}
              and census["n_fcurves"] == 32 and set(census["interpolation_census_curves"]) == {"CONSTANT"}
              and [census["key_frame_min"], census["key_frame_max"]] == [1.0, 240.0] and r["scene"]["effective_fps"] == 30.0
              and r["scene"]["frame_start"] == 1 and r["scene"]["frame_end"] == 240)
        fX, fA = _frames(r), _frames(rb["A"])
        edge = all(fX[f]["world"] == fA[f]["world"] and fX[f]["matrix_world"] == fA[f]["matrix_world"] for f in (i0 - 1, i1 + 1))
        m3 = (set(c[0] for c in ck) <= set(K.EDIT_CHANNELS) and all(i0 <= c[2] <= i1 for c in ck) and edge
              and not pres["nonarm_changed"] and not pres["outside_changed"] and not pres["sub_outside_changed"])
        L1 = [float(np.linalg.norm(np.subtract(fX[f]["world"]["lowerarm_l"]["head"], fX[f]["world"]["upperarm_l"]["head"]))) for f in range(i0 - 1, i1 + 2)]
        L2 = [float(np.linalg.norm(np.subtract(fX[f]["world"]["hand_l"]["head"], fX[f]["world"]["lowerarm_l"]["head"]))) for f in range(i0 - 1, i1 + 2)]
        vals = vK if br == "K" else vKc
        unit = all(abs(np.linalg.norm(q) - 1.0) < 1e-9 and np.isfinite(q).all() for ch in vals.values() for q in map(np.asarray, ch.values()))
        st = dumps[br]["stored"]
        key_err = max(abs(st[ch][str(c)][str(f)] - vals[ch][str(f)][c]) for ch in K.EDIT_CHANNELS for f in range(i0, i1 + 1) for c in range(4))
        frd = _frames(dumps[br])
        basis_err = 0.0
        for f in range(i0, i1 + 1):
            bb = mp.basis_from_base_frame(M.BaseFrame(dumps[br]["rest"], frd[f]))
            for ch in K.EDIT_CHANNELS:
                basis_err = max(basis_err, float(np.abs(M.canon_sign(bb[ch], np.asarray(vals[ch][str(f)])) - np.asarray(vals[ch][str(f)])).max()))
        g[br] = {"M2": bool(m2), "M3": bool(m3), "M5": bool(np.ptp(L1) <= 1e-4 and np.ptp(L2) <= 1e-4 and unit), "edge_171_200_bit_equal": bool(edge),
                 "M12": bool(key_err <= 1e-6 and basis_err <= 1e-5), "stored_key_vs_import_max": key_err, "evaluated_basis_vs_import_max": basis_err,
                 "changed_key_channels": sorted({c[0] for c in ck}), "changed_key_frames": [min((c[2] for c in ck), default=None), max((c[2] for c in ck), default=None)],
                 "n_changed_keys": len(ck), "nonarm_changed": pres["nonarm_changed"][:5], "outside_changed": pres["outside_changed"][:5],
                 "sub_outside_changed": pres["sub_outside_changed"][:5], "upper_len_range_m": float(np.ptp(L1)), "lower_len_range_m": float(np.ptp(L2)),
                 "blend_sha256": r["blend_sha256"], "action_name": r["action_name"], "n_subframes": len(r["subframes"])}
    pin_src = load(H6.FROZEN["W126V_proposal"][0])
    tgt = np.asarray(pin_src["constraint"]["target"])
    fp, fc = _frames(rb["Pin"]), _frames(rb["Kpin"])
    rp_ = {f: float(np.linalg.norm(np.asarray(fp[f]["world"]["hand_l"]["head"]) - tgt)) for f in K.PIN_ACTION}
    rc_ = {f: float(np.linalg.norm(np.asarray(fc[f]["world"]["hand_l"]["head"]) - tgt)) for f in K.PIN_ACTION}
    cp = M.preservation(rb["Kpin"], rb["Pin"], K.INTERIOR)
    m6 = {"residual_W126V_m": rp_, "residual_composed_m": rc_, "pin_frames_bit_equal": all(fc[f]["world"] == fp[f]["world"] for f in K.PIN_ACTION),
          "composed_nonarm_changed": cp["nonarm_changed"][:5], "composed_outside_changed": cp["outside_changed"][:5],
          "composed_sub_outside_changed": cp["sub_outside_changed"][:5], "pin_target_m": tgt.tolist()}
    m6["pass"] = bool(m6["pin_frames_bit_equal"] and all(rc_[f] == rp_[f] for f in K.PIN_ACTION) and max(rc_.values()) <= 1e-6
                      and not cp["nonarm_changed"] and not cp["outside_changed"] and not cp["sub_outside_changed"])
    m7 = all(p["world"] == q["world"] and p["matrix_world"] == q["matrix_world"] for p, q in zip(rb["Krev"]["frames"], rb["A"]["frames"])) \
        and all(p["world"] == q["world"] for p, q in zip(rb["Krev"]["subframes"], rb["A"]["subframes"]))
    eb = H7.world_max_diff(rb["Kc"], rb["B"])
    after = frozen_hashes()
    ck_after = {f: M.sha256_file(CKPT_DIR / f) for f in CKPT_FILES} == d["identities"]["model"]["files_sha256"]
    g.update({"M1_run": True, "M4": m4, "M6": m6, "M7": bool(m7), "M8": bool(eb <= 1e-5), "M8_max_world_diff_Kc_vs_B_m": eb,
              "M9": bool(all(v["ok"] for v in after.values()) and ck_after), "M10": bool(ctl["all_fail_closed"]),
              "frozen_after_bad": [k for k, v in after.items() if not v["ok"]], "checkpoint_unchanged": ck_after})
    # ---- generated effect ----
    fK, fKc = _frames(rb["K"]), _frames(rb["Kc"])
    eff = {str(f): {"elbow_mm": float(np.linalg.norm(np.subtract(fK[f]["world"]["lowerarm_l"]["head"], fKc[f]["world"]["lowerarm_l"]["head"]))) * 1e3,
                    "wrist_mm": float(np.linalg.norm(np.subtract(fK[f]["world"]["hand_l"]["head"], fKc[f]["world"]["hand_l"]["head"]))) * 1e3}
           for f in range(i0, i1 + 1)}
    mx_eff = max(max(v["elbow_mm"], v["wrist_mm"]) for v in eff.values())
    generated = {"max_mm": mx_eff, "mean_wrist_mm": float(np.mean([v["wrist_mm"] for v in eff.values()])),
                 "mean_elbow_mm": float(np.mean([v["elbow_mm"] for v in eff.values()])), "per_frame": eff,
                 "class": "GENERATED_EFFECT" if mx_eff >= 10.0 else "NO_GENERATED_EFFECT"}
    # ---- quality ----
    hinge = M.hinge_reference(rb["A"])
    q = {}
    for br in ("A", "B", "C", "D", "E", "U", "T", "K", "Kc"):
        qq = M.quality(rb[br], rb["A"], K.INTERIOR, WIN_Q, hinge)
        pf_ = qq.pop("per_frame", None)
        qq["flex_per_frame"] = dict(zip(map(str, pf_["frames"]), pf_["flex_deg"]))
        qq["arm_local"] = M.armlocal_quality(M.armlocal_wrist(dumps[br], *WIN_Q), K.INTERIOR, WIN_Q)
        qq["arm_local"].pop("per_frame", None)
        em = EP.elbow_metrics(dumps[br])
        em.pop("per_frame", None)
        qq["elbow"] = em
        fr_ = _frames(rb[br])
        sh = np.array([fr_[f]["world"]["upperarm_l"]["head"] for f in range(WIN_Q[0], WIN_Q[1] + 1)])
        qq["world_shoulder_step_max_mm"] = float(np.linalg.norm(np.diff(sh, axis=0), axis=1).max() * 1e3)
        wB = _frames(rb["B"])
        dv = [float(np.linalg.norm(np.subtract(fr_[f]["world"]["hand_l"]["head"], wB[f]["world"]["hand_l"]["head"]))) for f in range(i0, i1 + 1)]
        qq["wrist_dev_from_B_max_mm"], qq["wrist_dev_from_B_mean_mm"] = max(dv) * 1e3, float(np.mean(dv)) * 1e3
        qq["wrist_net_rise_mm"] = (fr_[i1]["world"]["hand_l"]["head"][2] - fr_[i0]["world"]["hand_l"]["head"][2]) * 1e3
        q[br] = qq
    s2d = H6.source_2d({k: rb[k] for k in ("A", "B", "C", "D", "E", "U", "T", "K", "Kc")}, rb["A"])
    qK, qB = q["K"], q["B"]

    def mx_steps(qq):
        return {"wrist": max(qq["arm_local"]["boundary_c0"]["entry_step_mm"], qq["arm_local"]["boundary_c0"]["exit_step_mm"], qq["arm_local"]["peak_step_mm"]),
                "elbow": max(qq["elbow"]["elbow_entry_step_mm"], qq["elbow"]["elbow_exit_step_mm"], qq["elbow"]["elbow_peak_step_mm"])}
    qa = {"Q1": bool(qK["arm_local"]["boundary_c0"]["entry_step_mm"] <= 2 * qB["arm_local"]["boundary_c0"]["entry_step_mm"] + 10
                     and qK["arm_local"]["boundary_c0"]["exit_step_mm"] <= 2 * qB["arm_local"]["boundary_c0"]["exit_step_mm"] + 10
                     and qK["elbow"]["elbow_entry_step_mm"] <= 2 * qB["elbow"]["elbow_entry_step_mm"] + 10
                     and qK["elbow"]["elbow_exit_step_mm"] <= 2 * qB["elbow"]["elbow_exit_step_mm"] + 10),
          "Q2": bool(qK["arm_local"]["n_reversal_flicks"] == 0 and len(qK["elbow"]["elbow_reversal_flicks"]) == 0),
          "Q3": bool(qK["arm_local"]["peak_step_mm"] <= 2 * qB["arm_local"]["peak_step_mm"] + 10
                     and qK["elbow"]["elbow_peak_step_mm"] <= 2 * qB["elbow"]["elbow_peak_step_mm"] + 10),
          "Q4": bool(not qK["near_straight_frames"] and not qK["hinge_opposite_frames"]),
          "Q5": bool(s2d["valid"] and s2d["branches"]["K"]["l_wri_mean_px"] is not None
                     and s2d["branches"]["K"]["l_wri_mean_px"] <= s2d["branches"]["B"]["l_wri_mean_px"]),
          "Q6": bool(qB["wrist_net_rise_mm"] > 0 and qK["wrist_net_rise_mm"] >= 0.5 * qB["wrist_net_rise_mm"])}
    sK = mx_steps(qK)
    anchors = {}
    for X in ("U", "T"):
        sX = mx_steps(q[X])
        fl = {"K": qK["arm_local"]["n_reversal_flicks"] + len(qK["elbow"]["elbow_reversal_flicks"]),
              X: q[X]["arm_local"]["n_reversal_flicks"] + len(q[X]["elbow"]["elbow_reversal_flicks"])}
        anchors[f"K_vs_{X}"] = {"K_max_steps_mm": sK, f"{X}_max_steps_mm": sX, "flicks": fl,
                                "class": "IMPROVED" if (sK["wrist"] < sX["wrist"] and sK["elbow"] < sX["elbow"] and fl["K"] <= fl[X]) else "NOT_IMPROVED"}
    proj_unusable = max(sK.values()) > K.UNUSABLE_STEP_MM
    arm_fail = proj_unusable or not (qa["Q1"] and qa["Q2"] and qa["Q3"])
    if not arm_fail:
        attrib = "NONE (projected continuity criteria pass)"
    elif rawu["unusable"]:
        attrib = "RAW_GENERATOR"
    elif max(v["raw_minus_projected_max_mm"] for v in att["raw_vs_projected"].values()) > K.UNUSABLE_STEP_MM:
        attrib = "FROZEN_PARENT_PROJECTION"
    else:
        attrib = "UNKNOWN"
    res.update({"gates": g, "generated_effect": generated, "quality": q, "source_2d": s2d, "quality_criteria_metric": qa, "anchors": anchors,
                "projected_unusable_continuity": bool(proj_unusable), "raw": rawan, "raw_unusable_continuity": rawu["unusable"],
                "attribution_class": attrib, "unusable_threshold_mm": K.UNUSABLE_STEP_MM,
                "blends": {k: str(v) for k, v in blends.items()}, "blend_sha256": {k: M.sha256_file(v) for k, v in blends.items()}})
    dump(out / "run_result.json", res)
    vid = render_and_compose(out, pdir, blends, log, order=ORDER)
    res["video"] = vid
    log["end"] = now()
    dump(out / "run_log.json", log)
    dump(out / "run_result.json", res)
    print(json.dumps({"gates": {k: (v if isinstance(v, (bool, float, int)) else (v.get("pass") if isinstance(v, dict) and "pass" in v else
                                                                                   {kk: vv for kk, vv in v.items() if kk.startswith("M") or kk.startswith("edge")}
                                                                                   if isinstance(v, dict) else v))
                                for k, v in g.items() if k not in ("frozen_after_bad",)},
                      "frozen_after_bad": g["frozen_after_bad"], "conditioning": {k: v for k, v in cond.items() if k != "groups"},
                      "obs_eq": obs_eq, "generated": {k: v for k, v in generated.items() if k != "per_frame"}, "qa": qa,
                      "anchors": {k: v["class"] for k, v in anchors.items()}, "K_steps": sK, "proj_unusable": proj_unusable, "raw_rule": rawu,
                      "attribution": attrib, "raw_excess": att["raw_excess_step_over_proxyB"],
                      "raw_vs_projected": att["raw_vs_projected"], "root_drift_max_interior": rawan["root_drift_mm"]["max_interior"],
                      "inference": {k: res["inference"][k] for k in ("sampler_seconds", "call_seconds", "total_seconds", "peak_vram_bytes")},
                      "video": {k: vid[k] for k in ("frames_written", "frames_decoded", "fps", "native_rate")}}, indent=1, default=str))
    return 0


def render_and_compose(out, pdir, blends, log, order=ORDER):
    """W136 renderer/compositor (copied; only the interior bar label states the W138 window) for the 9-column video and sheets."""
    import cv2
    s = H6.S1
    n = s["n_action"]
    i0, i1 = K.INTERIOR
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
        log["steps"].append([f"render_{br}", blender(H6.ACTION_STAGE, "render", out / f"render_{br}.log", blend=blends[br], jobs=out / "render_jobs.json",
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
        cv2.putText(b, f"edited interior Action {i0}-{i1} = SOURCE {i0 + seg0 - 1}-{i1 + seg0 - 1} (red); K window Action 163-208 (U/T: 156-215); 30 fps native; Action = SOURCE - 78",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        return b
    path = out / "abcdeutk_S1_30fps.mp4"
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
    sheets = {"entry": list(range(170, 176)), "middle": list(range(183, 189)), "exit": list(range(196, 202)), "overview": list(range(171, 201, 3))}
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
        rep = {"stage": "assess", "case": pf.get("stop"), "stop": pf.get("stop")}
    elif "STOP" in rr:
        st = rr["STOP"]
        case = "CONDITIONING_BLOCKED" if st["code"] == "CONDITIONING_BLOCKED" else ("C" if st.get("model_executed") else "D")
        rep = {"stage": "assess", "case": case, "stop": st}
    else:
        g = rr["gates"]
        mech = {"M0": pf["status"] == "OK", "M1": bool(g["M1_run"]), "M2": g["K"]["M2"] and g["Kc"]["M2"], "M3": g["K"]["M3"] and g["Kc"]["M3"],
                "M4": g["M4"], "M5": g["K"]["M5"] and g["Kc"]["M5"], "M6": g["M6"]["pass"], "M7": g["M7"], "M8": g["M8"], "M9": g["M9"],
                "M10": g["M10"], "M11": rr["video"]["native_rate"], "M12": g["K"]["M12"] and g["Kc"]["M12"]}
        qa = dict(rr["quality_criteria_metric"])
        ans = vr["answers"]
        qa["Q7"] = bool(ans["K_more_natural_than_B"] and ans["K_more_press_faithful_than_B"] and not ans["K_flick_wrong_elbow_or_detour"])
        qa["Q8"] = bool(vr.get("native_rate_reviewed", False))
        if not (mech["M8"] and mech["M9"] and mech["M10"]):
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
               "raw_rule_reasons": rr["raw"]["raw_rule"]["reasons"], "attribution_class": rr["attribution_class"],
               "anchors": {k: v["class"] for k, v in rr["anchors"].items()}, "visual_review": ans,
               "native_rate_reviewed": vr.get("native_rate_reviewed", False)}
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
