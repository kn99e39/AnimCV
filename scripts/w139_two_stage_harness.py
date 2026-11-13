#!/usr/bin/env python3
"""W139 (EXPERIMENTAL, opt-in, scratch only): Two-Stage Generative Animation feasibility on the authentic S1 rig.

Stage 1 = frozen W129 MotionBricks left-arm completion (W129 C_S1.blend; no re-run).  Bridge = W136 world-pose-preserving
export -> official UniMate rig_preprocess -> official features (identical for A/B/C).  Stage 2 = frozen UniMate v3
(W136/W137 checkpoint, prompt, seed 10, cfg 3.0, 50-step Euler) through the NON-OFFICIAL scheduled-replacement sampler
(flow-SDEdit on the W136 left-arm free set G, pinned through produced state K_S = 25, t_s = 0.5), built on the official
replacement primitive.  Only upperarm_l/lowerarm_l rotations at Action 172..199 are imported, into separate scratch
Actions; nothing enters a lineage.

Conditions: S2A (original AnimCV reference -> Stage 2), S2B (deterministic slerp -> Stage 2), S2C (MotionBricks ->
Stage 2), D (MotionBricks direct = W129 C).  Mechanism controls: K0B/K0C (k_s = 0 on PROXY_B / PROXY_C), S2B_rep
(replicate -> noise floor).  Model-native capability control: NREF (seed 11, unconstrained) -> FK re-encode -> N0 (k_s=0),
NL (k_s=25 from a B-analogue slerp init), NIB (official --inbetween frame mask).

Stages (each refuses an existing output directory):
  preflight  repo/branch state, frozen identities (W126/W129/W135/W136/W137, manifests), code/model/T5/rights, MotionBricks
             rights record, runtime; byte copies + fresh dumps/readbacks (bit-compared to W137's); the bridge for A, B, C
             (export, rig_preprocess, features, time identity, context drift, Blender roundtrip); cross-asset cond
             equality; G-only differences; per-condition world maps; pre-output injected-signal power; MotionBricks
             endpoint-correction decomposition; camera fit (diagnostic).  NO model.
  declare    prospective declaration BEFORE any W139 model output (refuses if any exists).
  controls   real-data negative controls (W136's 31 re-executed + W139 controls).  NO model.
  run        the declared model runs -> gates -> decomposition -> Actions -> readbacks -> metrics -> renders/videos.
  assess     prospective judgment over the run evidence + the agent's blinded still-frame review file.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w139_two_stage_harness.py <stage> --out-dir <dir> [...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np                                                    # noqa: E402

import app.experimental_unimate as U                                  # noqa: E402
import app.experimental_unimate_inbetween as T                        # noqa: E402
import app.experimental_two_stage as TS                               # noqa: E402
import app.experimental_elbow_prior as EP                             # noqa: E402
import app.experimental_motionbricks as M                             # noqa: E402
import w129_motionbricks_harness as H                                 # noqa: E402
import w136_unimate_harness as H6                                     # noqa: E402
import w137_unimate_inbetween_harness as H7                           # noqa: E402

W139 = Path("C:/Users/dna10/animcv-w139")
W129R = Path("C:/Users/dna10/animcv-w129/results")
W137R = Path("C:/Users/dna10/animcv-w137/results")
W136R = H6.W136 / "results"
INFER9 = ROOT / "scripts" / "w139_two_stage_infer.py"
BATCH, BRANCH = 139, "Dual_Stage_Generator"
W137_MANIFEST_SHA = "8561d239f6a95aa849da92555d35c9d1e38ae79768ac0a997f6bd53c9a5b7a19"
W137_BOUND = ["run/T_S1.blend", "run/readback_T.json", "run/dump_T.json", "declare/declaration.json", "assess/assessment.json",
              "run/run_result.json", "run/model/raw_model_io.npz", "preflight/readback_A.json", "preflight/readback_B.json",
              "preflight/readback_C.json", "preflight/readback_Pin.json", "preflight/readback_U.json"]
W129_SOMA = W129R / "preflight" / "soma_skeleton.json"
W129_ENGINE = W129R / "run" / "S1" / "model_R1" / "engine_output.npz"
W129_ENGINE_SHA = "8a528ef1bf7976f22bd6212eddf7c5b525ee5f30e27ee01f501d565baac6a100"
W136_U_IO = W136R / "run" / "model" / "raw_model_io.npz"
W136_U_IO_SHA = "14e80250f79af5d3bdee6f3c8efa5792c61bddfdd6d5f45ee4b60affaaac4d0a"
MB_AGREEMENT = Path("C:/Users/dna10/animcv-w135/rights/nvidia_open_model_agreement.html")
MB_AGREEMENT_SHA = "95ff88bc54c7507b9ad68303bbd0d27231a4063a5b16347fd6de362267095873"
MB_CLAUSES = ["Works are commercially usable", "NVIDIA does not claim ownership to any outputs generated using the Works or Derivative Works"]
STATS = H6.MODEL_DIR / "dataset_stats.npy"
CONDS = {"A": "S1_PROXY_A", "B": "S1_PROXY_B", "C": "S1_PROXY_C"}          # bridge input -> asset name (GLB stem)
COPY = {"A": "A", "B": "B", "C": "C", "W126V_accepted": "W126V_accepted"}
RUNS = {  # name: (proxy, k_s, mask, official_mode, seed, x1)
    "K0B": ("B", 0, "w136", "motion_edit", TS.SEED_COMPLETION, None),
    "K0C": ("C", 0, "w136", "motion_edit", TS.SEED_COMPLETION, None),
    "S2B": ("B", TS.K_S, "w136", "motion_edit", TS.SEED_COMPLETION, None),
    "S2B_rep": ("B", TS.K_S, "w136", "motion_edit", TS.SEED_COMPLETION, None),
    "S2A": ("A", TS.K_S, "w136", "motion_edit", TS.SEED_COMPLETION, None),
    "S2C": ("C", TS.K_S, "w136", "motion_edit", TS.SEED_COMPLETION, None),
    "NREF": ("B", 0, "empty", "motion_edit", TS.SEED_NATIVE_REF, None),
    "N0": ("B", 0, "w136", "motion_edit", TS.SEED_COMPLETION, "nref_star"),
    "NL": ("B", TS.K_S, "w136", "motion_edit", TS.SEED_COMPLETION, "nB_init"),
    "NIB": ("B", 0, "official", "inbetween", TS.SEED_COMPLETION, "nref_star"),
}
CANDIDATES = ("S2A", "S2B", "S2C")
INSTRUMENTS = ["src/app/experimental_two_stage.py", "src/app/experimental_two_stage_sampler.py", "scripts/w139_two_stage_infer.py",
               "scripts/w139_two_stage_harness.py", "tests/test_experimental_two_stage.py"] + H7.INSTRUMENTS
NAMES = {"A": "A original reference (no-op)", "B": "B W129 slerp (no model)", "D": "D Stage-1 MotionBricks direct (W129 C)",
         "S2A": "S2A reference -> UniMate", "S2B": "S2B slerp -> UniMate", "S2C": "S2C MotionBricks -> UniMate (two-stage)",
         "U": "U W136 (NON-EQUIVALENT ref)", "T": "T W137 (NON-EQUIVALENT ref)"}
ORDER = ("A", "B", "D", "S2A", "S2B", "S2C", "U", "T")
MAP_BONES = H7.MAP_BONES
WIN = (164, 207)

load, dump, new_dir, now, git, lf_sha = H.load, H.dump, H.new_dir, H.now, H.git, H.lf_sha
blender, readback, bdump, um, _frames = H6.blender, H6.readback, H6.bdump, H6.um, H6._frames


def frozen_hashes():
    out = H7.frozen_hashes()
    man = W137R / "evidence_manifest.json"
    ok_m = man.is_file() and M.sha256_file(man) == W137_MANIFEST_SHA
    out["W137_evidence_manifest"] = {"path": str(man), "expected": W137_MANIFEST_SHA, "sha256": M.sha256_file(man) if man.is_file() else None, "ok": ok_m}
    files = load(man)["files"] if ok_m else {}
    for rel in W137_BOUND:
        p = W137R / rel
        want = (files.get(rel) or {}).get("sha256")
        got = M.sha256_file(p) if p.is_file() else None
        out["W137:" + rel] = {"path": str(p), "expected": want, "sha256": got, "ok": bool(want and got == want)}
    for k, p, h in (("W129_engine_output", W129_ENGINE, W129_ENGINE_SHA), ("W136_U_raw_model_io", W136_U_IO, W136_U_IO_SHA),
                    ("W135_MB_agreement", MB_AGREEMENT, MB_AGREEMENT_SHA)):
        got = M.sha256_file(p) if p.is_file() else None
        out[k] = {"path": str(p), "expected": h, "sha256": got, "ok": got == h}
    return out


def check_declaration(a):
    d = H6.check_declaration(a)
    return d


def _is_stub(p):
    rec = Path(p).parent / "infer_record.json"
    return rec.is_file() and "STUB_MODEL" in load(rec)


def model_outputs_exist():
    """Every real (non-stub) W139 model output anywhere under the W139 scratch root."""
    if not W139.is_dir():
        return []
    hits = []
    for pat in ("raw_model_io.npz", "trajectory.npz", "first_free_tokens.npz", "*-rep_0-*.npy"):
        for p in W139.rglob(pat):
            q = p
            while q.parent != q and not (q / "infer_record.json").is_file() and q != W139:
                q = q.parent
            rec = q / "infer_record.json"
            if rec.is_file() and "STUB_MODEL" in load(rec):
                continue
            hits.append(str(p))
    return sorted(hits)


def stats_norm():
    s = np.load(STATS, allow_pickle=True).item()
    k = sorted(s)[0] if "objaverse" not in s else "objaverse"
    st = s[k]
    mean = np.concatenate([np.asarray(st["mean_root"])[None], np.repeat(np.asarray(st["mean_local"])[None], U.N_JOINTS - 1, 0)])
    std = np.concatenate([np.asarray(st["std_root"])[None], np.repeat(np.asarray(st["std_local"])[None], U.N_JOINTS - 1, 0)])
    return mean, std, k


def normalise(feats, mean, std):
    """(T,J,12) denormalised -> (1,J,12,T) normalised, as conditioning.py (x - mean) / std."""
    return ((np.transpose(np.asarray(feats, np.float64), (1, 2, 0)) - mean[:, :, None]) / std[:, :, None])[None].astype(np.float32)


# ------------------------------------------------------------------------------------------------ bridge ---
def bridge(out, key, inp, dX, rbX, dA, rbA, ann_ref=None):
    """NO-MODEL bridge for one condition: X.blend -> export -> rig_preprocess -> features -> decode == X keys -> Blender roundtrip."""
    asset = CONDS[key]
    g, steps = {}, []
    od = out / f"bridge_{key}"
    od.mkdir()
    glb = od / f"{asset}.glb"
    steps.append(["export", blender(H6.EXPORT_STAGE, "export", od / "export.log", blend=inp / f"{key}.blend", out=glb,
                                    frames=f"{U.EXPORT_FRAMES[0]},{U.EXPORT_FRAMES[1]}", report=od / "export_report.json",
                                    expected_sha=H6.FROZEN[COPY[key]][1])])
    er = load(od / "export_report.json")
    g["export"] = {k: er[k] for k in ("world_matrix_max_abs_diff_anatomical", "frames", "adapter", "glb_sha256", "n_anatomical_bones_checked")}
    if er["world_matrix_max_abs_diff_anatomical"] > 1e-5 or er["frames"] != list(U.EXPORT_FRAMES):
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{key}: export adapter changed world poses or frames")
    rig = od / "rig"
    steps.append(["rig_preprocess_review", um(["-m", "data_process.rig_preprocess", "run", "--input", glb, "--output_dir", rig,
                                               "--annotate", "rule", "--save_clips", "--keep_intermediate"], od / "rig_review.log")])
    if steps[-1][1]["rc"] != 0:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{key}: rig_preprocess review rc")
    ann = load(rig / "annotation.json")
    labels = {j["raw"]: j["label"] for j in ann["joints"]}
    g["annotation"] = {"n_flagged": ann["n_flagged"], "notes": ann["notes"], "face_pair": ann["face_pair"], "sha256": M.sha256_file(rig / "annotation.json"),
                       "labels_equal_reference": (ann_ref is None or labels == ann_ref)}
    if ann["n_flagged"] != 0 or ann["notes"] or not g["annotation"]["labels_equal_reference"]:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{key}: annotation {g['annotation']}")
    for b, lab in U.EXPECTED_LABELS.items():
        if labels.get(b) != lab:
            raise U.UniStop("LABEL_MISMATCH", f"{key}: {b}")
    if (ann["face_pair"]["right"], ann["face_pair"]["left"]) != U.FACE_PAIR or ann["face_pair"]["body_axis"]:
        raise U.UniStop("FACING_PAIR", f"{key}: {ann['face_pair']}")
    shutil.copyfile(rig / "annotation.json", od / "annotation_reviewed.json")
    steps.append(["rig_preprocess_build", um(["-m", "data_process.rig_preprocess", "run", "--input", glb, "--output_dir", rig, "--profile", "general",
                                              "--save_clips", "--keep_intermediate", "--annotation", rig / "annotation.json"], od / "rig_build.log")])
    if steps[-1][1]["rc"] != 0:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{key}: rig_preprocess build rc")
    summ = load(rig / "summary.json")
    g["summary"] = {k: summ.get(k) for k in ("name", "n_joints", "n_clips", "clips", "notes", "stats_dataset")}
    if summ["name"] != asset or summ["n_joints"] != U.N_JOINTS or summ["n_clips"] != 1 or summ["notes"] or summ["clips"] != [f"{asset}-Animation-000"]:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{key}: summary {g['summary']}")
    clip = summ["clips"][0]
    steps.append(["features", um([H6.INFER, "features", "--out-dir", od / "features", "--asset-dir", rig, "--name", asset, "--clip", clip], od / "features.log")])
    if steps[-1][1]["rc"] != 0:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{key}: features rc")
    z = np.load(od / "features" / "features.npz")
    names, par = [str(n) for n in z["names"]], np.asarray(z["parents"])
    U.check_fps(int(z["clip_fps"]))
    if int(z["n_clip_frames"]) != U.T_MODEL + 1 or z["feats"].shape != (U.T_MODEL, U.N_JOINTS, U.FEAT):
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{key}: clip shape")
    U.validate_asset(names, par, z["clean_joint_names"], U.FACE_PAIR, H6.rig_parents(dX))
    dec = {b: z[f"decode_{b}"] for b in U.EDIT_CHANNELS}
    g["time_identity"] = U.time_identity(dec, dX["stored"])
    g["own_vs_official_decode"] = load(od / "features" / "features.json")["own_vs_official_decode_max"]
    if max(g["own_vs_official_decode"].values()) > 1e-9:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{key}: own decoder != official")
    ctx = U.decode_basis(z["feats"], names, par, z["tpos_local_rotations"], joints=("spine_01", "thigh_l", "thigh_r", "calf_l", "calf_r"))
    g["context_drift"] = {b: U.roundtrip_error(ctx, dX["stored"], b) for b in ctx}
    if max(g["context_drift"].values()) > U.QUAT_TOL:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{key}: context drift")
    gp = np.load(rig / "motions" / f"{clip}.npz")["global_positions"]
    low = gp[..., 1].min(0)
    g["grounding"] = {"min_y_joint": names[int(np.argmin(low))], "left_arm_chain_min_y": float(min(low[names.index(b)] for b in ("upperarm_l", "lowerarm_l", "hand_l")))}
    # Blender roundtrip: decoded keys through the authorised import path == X (bridge is lossless)
    vals = U.import_values(dec, dA["stored"])
    spec = {"out_blend": str(od / f"R0_{key}_S1.blend"), "interior_action_frames": list(U.INTERIOR), "values_wxyz": vals,
            "branch": f"W139 R0 {key}", "action_name": f"AnimCV_EXP W139 R0 {key} roundtrip S1", "expected_blend_sha256": H6.FROZEN["A"][1],
            "note": f"W139 no-model bridge roundtrip of {key}"}
    dump(od / "build_R0_spec.json", spec)
    steps.append(["build_R0", blender(H6.STAGE, "build_variant", od / "build_R0.log", blend=inp / "A.blend", spec=od / "build_R0_spec.json",
                                      report=od / "build_R0_report.json")])
    steps.append(["readback_R0", readback(od / f"R0_{key}_S1.blend", od / "readback_R0.json", od / "readback_R0.log")])
    rr = load(od / "readback_R0.json")
    pres = M.preservation(rr, rbA, U.INTERIOR)
    g["blender_roundtrip_max_world_diff_m"] = H7.world_max_diff(rr, rbX)
    if g["blender_roundtrip_max_world_diff_m"] > 1e-5 or pres["nonarm_changed"] or pres["outside_changed"] or pres["sub_outside_changed"]:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"{key}: Blender roundtrip {g['blender_roundtrip_max_world_diff_m']}")
    g["asset_dir"], g["asset"], g["clip"] = str(rig), asset, clip
    g["features_npy"] = str(od / "features_feats.npy")
    np.save(od / "features_feats.npy", z["feats"].astype(np.float32))
    g["rig_sha256"] = {f: M.sha256_file(rig / f) for f in ("cond.npy", "summary.json", "annotation.json", f"motions/{clip}.npz")}
    g["features_sha256"] = M.sha256_file(od / "features" / "features.npz")
    return g, steps, labels, names, par


def cond_equality(bridges):
    ref = np.load(Path(bridges["B"]["asset_dir"]) / "cond.npy", allow_pickle=True).item()[CONDS["B"]]
    out = {}
    for k in ("A", "C"):
        c = np.load(Path(bridges[k]["asset_dir"]) / "cond.npy", allow_pickle=True).item()[CONDS[k]]
        diff = []
        for f, v in ref.items():
            if f in ("object_type",):
                continue
            w = c.get(f)
            if isinstance(v, np.ndarray) or isinstance(w, np.ndarray):
                if not (isinstance(w, np.ndarray) and isinstance(v, np.ndarray) and v.shape == w.shape and np.array_equal(v, w)):
                    diff.append(f)
            elif v != w:
                diff.append(f)
        out[k] = {"differing_fields": diff, "ok": not diff, "object_type": [str(ref.get("object_type")), str(c.get("object_type"))]}
    return out


def g_only_differences(x1, g_valid, keep_valid):
    rep = {}
    for p, q in (("A", "B"), ("C", "B"), ("A", "C")):
        d = np.abs(x1[p][0] - x1[q][0])
        rep[f"{p}-{q}"] = {"max_abs_on_keep": float(d[keep_valid].max()), "max_abs_on_G": float(d[g_valid].max()),
                           "n_differing_on_G": int((d[g_valid] > 0).sum())}
    rep["ok"] = all(v["max_abs_on_keep"] == 0.0 for v in rep.values() if isinstance(v, dict))
    return rep


def correction_decomposition(dA, dB, dC):
    """NO MODEL: split the W129 Stage-1 interior basis into MotionBricks' deterministic endpoint-correction ramp and the
    uncorrected (learned) remainder, per frame, in BaseRig basis space (same closed-form mapping as W129)."""
    soma = M.SomaSkeleton.load(W129_SOMA)
    mp = M.Mapping(soma, M.rest_frame(dA["rest"], H.OBJECT_REST_WXYZ, H.OBJECT_SCALE))
    eo = np.load(W129_ENGINE)
    lp, sd, ed = np.asarray(eo["local_poses"], float), np.asarray(eo["start_deltas"], float), np.asarray(eo["end_deltas"], float)
    ctxl, tgtl = np.asarray(eo["context_local"], float), np.asarray(eo["target_local"], float)
    raw_lp = lp.copy()
    endpoint = {}
    for jn in ("LeftArm", "LeftForeArm"):
        j = soma.idx[jn]
        raw_q, _ = TS.undo_endpoint_correction(lp[:, j, 3:], sd[j, 3:], ed[j, 3:])
        raw_lp[:, j, 3:] = np.concatenate([raw_q[:, 1:], raw_q[:, :1]], -1)              # back to xyzw
        endpoint[jn] = {"corrected_first_vs_context_last_deg": float(TS.qangle_deg(TS.xyzw_to_wxyz(lp[0, j, 3:]), TS.xyzw_to_wxyz(ctxl[-1, j, 3:]))),
                        "corrected_last_vs_target_first_deg": float(TS.qangle_deg(TS.xyzw_to_wxyz(lp[-1, j, 3:]), TS.xyzw_to_wxyz(tgtl[0, j, 3:]))),
                        "start_delta_deg": float(TS.qangle_deg(TS.xyzw_to_wxyz(sd[j, 3:]), [1, 0, 0, 0])),
                        "end_delta_deg": float(TS.qangle_deg(TS.xyzw_to_wxyz(ed[j, 3:]), [1, 0, 0, 0]))}
    per = {}
    for k, f in enumerate(range(U.CONTEXT_LAST, U.TARGET_FIRST + 1)):
        if not U.INTERIOR[0] <= f <= U.INTERIOR[1]:
            continue
        bc = mp.base_basis_from_soma(lp[k])
        br = mp.base_basis_from_soma(raw_lp[k])
        row = {}
        for ch in U.EDIT_CHANNELS:
            qC = np.array([dC["stored"][ch][str(c)][str(f)] for c in range(4)])
            qB = np.array([dB["stored"][ch][str(c)][str(f)] for c in range(4)])
            row[ch] = {"C_vs_C_mapped_deg": float(TS.qangle_deg(bc[ch], qC)), "C_vs_Craw_deg": float(TS.qangle_deg(bc[ch], br[ch])),
                       "Craw_vs_B_deg": float(TS.qangle_deg(br[ch], qB)), "C_vs_B_deg": float(TS.qangle_deg(qC, qB))}
            row[ch]["correction_dominated"] = bool(row[ch]["C_vs_Craw_deg"] >= 0.5 * max(row[ch]["C_vs_B_deg"], 1e-9))
        per[str(f)] = row
    dom = {ch: [int(f) for f, r in per.items() if r[ch]["correction_dominated"]] for ch in U.EDIT_CHANNELS}
    return {"endpoint_identity": endpoint, "per_frame": per, "correction_dominated_frames": dom,
            "mapping_check_max_deg": max(r[ch]["C_vs_C_mapped_deg"] for r in per.values() for ch in U.EDIT_CHANNELS),
            "rule": "frame is correction-dominated for a channel iff angle(C, C_uncorrected) >= 0.5 x angle(C, B)",
            "formula": "q_corr[f] = q_raw[f] * slerp(1, dS, 1 - f/29) * slerp(1, dE, f/29) (motionbricks/model/soma/kernels.py apply_endpoint_deltas_kernel)"}


# ------------------------------------------------------------------------------------------------ preflight ---
def stage_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"stage": "preflight", "time": now(), "head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
           "origin_single_frame_first": git("rev-parse", "origin/arch/single_frame_first"), "steps": []}
    rep["status_porcelain"] = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.splitlines()
    w138 = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "--name-only", "origin/arch/single_frame_first", "docs/"], capture_output=True, text=True).stdout
    rep["w138"] = {"worklog_on_origin_single_frame_first": [l for l in w138.splitlines() if l.startswith("docs/138_")],
                   "human_decision": "2026-11-13: user directed W139 to run in parallel on Dual_Stage_Generator while W138 continues on "
                                     "arch/single_frame_first; the spec's W138-COMPLETE gate and branch line are superseded by that explicit decision"}
    rep["w139_worklog_absent"] = not any((ROOT / "docs").glob("139_*"))
    rep["frozen"] = frozen_hashes()
    stop = None
    code = {"head": subprocess.run(["git", "-C", str(H6.UM_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "dirty": subprocess.run(["git", "-C", str(H6.UM_ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()}
    w137pf = load(W137R / "preflight" / "preflight.json")
    code["files_sha256"] = {f: M.sha256_file(H6.UM_ROOT / f) for f in w137pf["code"]["files_sha256"]}
    code["files_equal_W137"] = code["files_sha256"] == w137pf["code"]["files_sha256"]
    code["extra_sha256"] = {f: M.sha256_file(H6.UM_ROOT / f) for f in ("unimate/models/flow/transport.py", "unimate/inference/motion_expansion.py",
                                                                        "unimate/models/denoiser/graph_adaln.py", "unimate/models/denoiser/blocks/common.py")}
    code["ok"] = code["head"] == H6.UM_COMMIT and code["dirty"] == "" and code["files_equal_W137"]
    rep["code"] = code
    meta = new_dir(out / "meta")
    api_ok = H6._http(f"https://huggingface.co/api/models/{H6.HF_REPO}/revision/{H6.HF_REV}?blobs=true", meta / "unimate_api.json")
    t5_ok = H6._http(f"https://huggingface.co/api/models/{H6.T5_REPO}/revision/{H6.T5_REV}?blobs=true", meta / "t5_api.json")
    for f, dst in (("README.md", meta / "hf_README.md"), ("LICENSE", meta / "hf_LICENSE")):
        H6._http(f"https://huggingface.co/{H6.HF_REPO}/resolve/{H6.HF_REV}/{f}", dst)
    H6._http(f"https://huggingface.co/{H6.T5_REPO}/resolve/{H6.T5_REV}/README.md", meta / "t5_README.md")
    mi = load(meta / "unimate_api.json") if api_ok else {}
    ti = load(meta / "t5_api.json") if t5_ok else {}
    sib = {s["rfilename"]: s for s in mi.get("siblings", [])}
    local = {f: M.sha256_file(H6.W136 / "model" / f) for f in (H6.CKPT_REL, f"{H6.MODEL_NAME}/dataset_stats.npy", f"{H6.MODEL_NAME}/config.json")}
    cfg = load(H6.MODEL_DIR / "config.json")
    model = {"repo": H6.HF_REPO, "revision": H6.HF_REV, "api_sha": mi.get("sha"), "gated": mi.get("gated"), "local_sha256": local,
             "checkpoint_lfs_sha256": (sib.get(H6.CKPT_REL, {}).get("lfs") or {}).get("sha256"),
             "config": {"diff_model": cfg["training"]["diff_model"], "cfg_scale": cfg["sampling"]["cfg_scale"], "seed": cfg["sampling"]["seed"],
                        "max_motion_length": cfg["dataset"]["max_motion_length"], "t_sampling": cfg["training"].get("t_sampling")}}
    model["ok"] = bool(api_ok and mi.get("sha") == H6.HF_REV and mi.get("gated") is False and local[H6.CKPT_REL] == H6.CKPT_SHA
                       and model["checkpoint_lfs_sha256"] == H6.CKPT_SHA and local[f"{H6.MODEL_NAME}/dataset_stats.npy"] == H6.STATS_SHA
                       and local[f"{H6.MODEL_NAME}/config.json"] == H7.CONFIG_SHA and model["config"]["diff_model"] == "flow")
    rep["model"] = model
    t5_local = H7.T5_LOCAL / "model.safetensors"
    t5 = {"api_sha": ti.get("sha"), "gated": ti.get("gated"), "card_license": (ti.get("cardData") or {}).get("license"),
          "local_safetensors_sha256": M.sha256_file(t5_local) if t5_local.is_file() else None}
    t5["ok"] = bool(t5_ok and ti.get("sha") == H6.T5_REV and ti.get("gated") is False and t5["local_safetensors_sha256"] == H7.T5_SAFETENSORS_SHA)
    rep["text_encoder"] = t5
    txt = {"hf_license": H6._norm_ws((meta / "hf_LICENSE").read_text(encoding="utf-8")) if (meta / "hf_LICENSE").is_file() else "",
           "hf_readme": H6._norm_ws((meta / "hf_README.md").read_text(encoding="utf-8")) if (meta / "hf_README.md").is_file() else "",
           "code_license": H6._norm_ws((H6.UM_ROOT / "LICENSE").read_text(encoding="utf-8")),
           "t5_readme": H6._norm_ws((meta / "t5_README.md").read_text(encoding="utf-8")) if (meta / "t5_README.md").is_file() else ""}
    present = {k: {c: H6._norm_ws(c) in txt[k] for c in cl} for k, cl in H6.RIGHTS_REQUIRED.items()}
    fsha = {p.name: M.sha256_file(p) for p in meta.iterdir() if p.is_file()}
    w137_rights = w137pf["rights"]["files_sha256"]
    mb_txt = MB_AGREEMENT.read_text(encoding="utf-8", errors="replace") if MB_AGREEMENT.is_file() else ""
    rights = {"unimate_required_clauses_present": present,
              "unimate_licence_texts_equal_W137": {k: fsha.get(k) == w137_rights.get(k) for k in ("hf_LICENSE", "hf_README.md", "t5_README.md")},
              "motionbricks": {"weights_agreement_file": str(MB_AGREEMENT), "sha256_ok": rep["frozen"]["W135_MB_agreement"]["ok"],
                               "clauses_present": {c: H6._norm_ws(c) in H6._norm_ws(mb_txt) for c in MB_CLAUSES},
                               "code": "Apache-2.0 (NVlabs/motionbricks 54fc563)", "determination_W135": "PERMITTED (local use, no redistribution)",
                               "caveat": "agreement text fetched 2026-11-10 (W135); the version in force at the W129 download was not archived; W139 does not re-run MotionBricks"},
              "intended_use": "private, local, non-commercial technical research evaluation on one machine; no redistribution of weights or "
                              "outputs, no training/finetuning, no production or commercial use; the chained result inherits UniMate's CC BY-NC 4.0 terms",
              "external_services": "none (no LLM API; Hugging Face public metadata/licence pages only; weights already local, hash-verified)"}
    rights["permitted"] = bool(all(all(v.values()) for v in present.values()) and all(rights["unimate_licence_texts_equal_W137"].values())
                               and rights["motionbricks"]["sha256_ok"] and all(rights["motionbricks"]["clauses_present"].values()))
    rights["determination"] = "PERMITTED_NONCOMMERCIAL_RESEARCH_EVALUATION" if rights["permitted"] else "RIGHTS_UNRESOLVED"
    rights["commercial"] = "NOT GRANTED (UniMate weights CC BY-NC 4.0; a human-owned decision)"
    rep["rights"] = rights
    env_code = ("import sys,json,torch,numpy;print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,"
                "'cudnn':torch.backends.cudnn.version(),'available':torch.cuda.is_available(),'arch':torch.cuda.get_arch_list(),"
                "'device':torch.cuda.get_device_name(0),'numpy':numpy.__version__}))")
    rep["steps"].append(["um_env", um(["-c", env_code], out / "um_env.log")])
    try:
        envr = json.loads([ln for ln in (out / "um_env.log").read_text(encoding="utf-8").splitlines() if ln.startswith("{")][-1])
    except IndexError:
        envr = {}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    W139.mkdir(exist_ok=True)
    du = shutil.disk_usage(str(W139))
    rep["runtime"] = {"um_env": envr, "nvidia_smi": smi, "disk_free_gb": round(du.free / 1e9, 1), "freeze_sha256": M.sha256_file(H6.W136 / "env" / "freeze.txt"),
                      "w136_infer_record_env": {k: load(W136R / "run" / "model" / "infer_record.json").get(k) for k in ("device", "torch")},
                      "ok": bool(envr.get("available") and "sm_120" in envr.get("arch", []) and du.free > 20e9)}
    rep["blender"] = subprocess.run([H.BLENDER, "-b", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    pre_ok = (rep["branch"] == BRANCH and rep["w139_worklog_absent"] and all(v["ok"] for v in rep["frozen"].values()) and bool(rep["blender"]))
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
        shutil.copyfile(W137R / "run" / "T_S1.blend", inp / "T.blend")
        if any(M.sha256_file(inp / f"{k}.blend") != H6.FROZEN[fk][1] for k, fk in COPY.items()) or \
                M.sha256_file(inp / "U.blend") != rep["frozen"]["W136:run/U_S1.blend"]["expected"] or \
                M.sha256_file(inp / "T.blend") != rep["frozen"]["W137:run/T_S1.blend"]["expected"]:
            stop = "DATA_BLOCKED"
        for k in ("A", "B", "C", "U", "T"):
            rep["steps"].append([f"dump_{k}", bdump(inp / f"{k}.blend", out / f"dump_{k}.json", out / f"dump_{k}.log")])
            rep["steps"].append([f"readback_{k}", readback(inp / f"{k}.blend", out / f"readback_{k}.json", out / f"readback_{k}.log")])
        rep["steps"].append(["readback_Pin", readback(inp / "W126V_accepted.blend", out / "readback_Pin.json", out / "readback_Pin.log")])
        eq = {}
        for k, ref in (("A", "preflight/readback_A.json"), ("B", "preflight/readback_B.json"), ("C", "preflight/readback_C.json"),
                       ("Pin", "preflight/readback_Pin.json"), ("U", "preflight/readback_U.json"), ("T", "run/readback_T.json")):
            w, f = load(W137R / ref), load(out / f"readback_{k}.json")
            eq[k] = all(p["world"] == q["world"] for p, q in zip(w["frames"], f["frames"])) and \
                all(p["world"] == q["world"] for p, q in zip(w["subframes"], f["subframes"]))
        rep["fresh_world_bit_equal_to_W137"] = eq
        if not all(eq.values()):
            stop = "DATA_BLOCKED"
        try:
            dumps = {k: load(out / f"dump_{k}.json") for k in ("A", "B", "C")}
            rbs = {k: load(out / f"readback_{k}.json") for k in ("A", "B", "C")}
            for k in dumps:
                EP.check_dump(dumps[k])
            br, labels_ref, names, par = {}, None, None, None
            for k in ("B", "A", "C"):
                br[k], st, lab, names, par = bridge(out, k, inp, dumps[k], rbs[k], dumps["A"], rbs["A"], labels_ref)
                labels_ref = labels_ref or lab
                rep["steps"] += st
            rep["bridge"] = br
            w136f = np.load(W136R / "preflight" / "features" / "features.npz")["feats"]
            fB = np.load(br["B"]["features_npy"])
            rep["proxyB_features_vs_W136"] = {"max_abs": float(np.abs(fB - w136f.astype(np.float32)).max()),
                                              "bit_equal_float64": bool(np.array_equal(np.load(Path(br["B"]["asset_dir"]).parent / "features" / "features.npz")["feats"], w136f))}
            rep["cond_equality"] = cond_equality(br)
            mean, std, sk = stats_norm()
            x1 = {k: normalise(np.load(Path(br[k]["asset_dir"]).parent / "features" / "features.npz")["feats"], mean, std) for k in ("A", "B", "C")}
            km = U.build_keep_mask(names, par, U.N_JOINTS)
            g = TS.sched_set(km, U.N_JOINTS)
            rep["g_only"] = g_only_differences(x1, g, km)
            rep["stats_key"] = sk
            rep["injected_signal_ratio"] = {"C_vs_B": TS.injected_signal_ratio(x1["B"], x1["C"], g), "A_vs_B": TS.injected_signal_ratio(x1["B"], x1["A"], g)}
            np.savez(out / "x1_normalised_offline.npz", **x1)
            rep["x1_offline_sha256"] = M.sha256_file(out / "x1_normalised_offline.npz")
            rep["features_npy_sha256"] = {k: M.sha256_file(br[k]["features_npy"]) for k in br}
            # per-condition world maps (each PROXY_X FK vs X readback)
            wms = {}
            for k in ("A", "B", "C"):
                od = out / f"recover_proxy_{k}"
                rep["steps"].append([f"recover_{k}", um([INFER9, "recover", "--out-dir", od, "--asset-dir", br[k]["asset_dir"], "--name", CONDS[k],
                                                         "--proxy", br[k]["features_npy"], "--samples", f"self={br[k]['features_npy']}"], out / f"recover_{k}.log")])
                rr = np.load(od / "recovered.npz")
                wms[k] = T.fit_world_map(rr["proxy_fk"], names, rbs[k], MAP_BONES)
            rep["world_maps"] = wms
            rep["world_map_spread"] = {"s": float(np.ptp([wms[k]["s"] for k in wms])), "t_m": float(np.max(np.ptp([wms[k]["t"] for k in wms], axis=0)))}
            rep["correction_decomposition"] = correction_decomposition(dumps["A"], dumps["B"], dumps["C"])
            if not (rep["g_only"]["ok"] and all(v["ok"] for v in rep["cond_equality"].values()) and rep["proxyB_features_vs_W136"]["max_abs"] <= 1e-6):
                stop = "REPRESENTATION_BLOCKED"
            rep["names"], rep["parents"] = names, [int(p) for p in par]
        except U.UniStop as e:
            rep["bridge_stop"] = {"code": e.code, "detail": e.detail}
            stop = "REPRESENTATION_BLOCKED"
        except (RuntimeError, OSError, KeyError, ValueError, IndexError, EP.PriorStop, TS.TwoStageStop) as e:
            rep["bridge_stop"] = {"code": "REPRESENTATION_BLOCKED", "detail": f"{type(e).__name__}: {e}"}
            stop = "REPRESENTATION_BLOCKED"
        if stop is None:
            s2d = H6.source_2d({"A": load(out / "readback_A.json")}, load(out / "readback_A.json"))
            rep["source_2d_camera"] = {k: s2d[k] for k in ("camera", "fit_rms_px", "n_fit_points", "valid")}
    rep["stop"] = stop
    rep["status"] = "OK" if stop is None else "DATA_BLOCKED"
    dump(out / "preflight.json", rep)
    print(json.dumps({"status": rep["status"], "stop": stop, "branch": rep["branch"], "rights": rights["determination"], "model_ok": model["ok"],
                      "t5_ok": t5["ok"], "code_ok": code["ok"], "runtime_ok": rep["runtime"]["ok"],
                      "frozen_bad": [k for k, v in rep["frozen"].items() if not v["ok"]], "fresh_eq": rep.get("fresh_world_bit_equal_to_W137"),
                      "bridge_stop": rep.get("bridge_stop"), "g_only": rep.get("g_only"), "cond_equality": rep.get("cond_equality"),
                      "proxyB_vs_W136": rep.get("proxyB_features_vs_W136"), "power": rep.get("injected_signal_ratio"),
                      "world_map_rms": {k: v["rms_m"] for k, v in rep.get("world_maps", {}).items()},
                      "correction_dominated": (rep.get("correction_decomposition") or {}).get("correction_dominated_frames"),
                      "camera": rep.get("source_2d_camera", {}).get("fit_rms_px")}, indent=1, default=str))
    return 0 if stop is None else 3


# ------------------------------------------------------------------------------------------------ declaration ---
INFO_RULE = {
    "I1_visibility": "implementation check on NON-DISCARDED evaluations only (the per-run recorded-state checks -- every state j<=K_S equals "
                     "the scheduled interpolant on keep|G bit-exactly, j>K_S on keep -- are PREREQUISITES, failure = BLOCKED); the network input at "
                     "Euler step K_S (first evaluation "
                     "whose output is not overwritten on G) equals state K_S; cross-run state K_S difference equals t_s (x1_C - x1_B) on G "
                     f"within {TS.CROSS_TOL} and 0 on kept valid entries; the input-layer token embeddings of step K_S differ on G cells and agree "
                     "(<= 1e-5) elsewhere. Steps 0..K_S-1 are reported as computed-and-discarded and are never evidence.",
    "I2_learned_reading": f"gamma = <dx1hat_K_S, dx1>/|dx1|^2 on G (x1hat = x + (1-t) v, the network's own clean-data estimate at the first free "
                          f"evaluation) >= {TS.GAMMA_MIN}; passive state-blind null = t_s = 0.5, context-only regeneration ~0, perfect denoiser 1",
    "I3_network_response": f"network term N = dout - t_s dx1 on G with ||N|| >= {TS.NET_RATIO_MIN} ||t_s dx1|| AND decoded (hybrid FK, PROXY_C context, "
                           f"world map C) max wrist/elbow |out_C - null_C| over Action 172..199 >= {TS.NET_MM_MIN} mm AND >= {TS.NOISE_FACTOR} x the "
                           "S2B/S2B_rep decoded noise floor AND ||N|| >= 10 x feature noise floor x sqrt(|G|); null_C = out_B + t_s (x1_C - x1_B) on G",
    "learned_path": "I1 AND I2 AND I3; direction reported as AMPLIFIES / WASHES_OUT / ORTHOGONAL from cos(N, dx1) (+-0.1)",
    "reported_not_ruled": "beta = <dout, dx1>/|dx1|^2 (passive null 0.5), gamma at steps K_S+5, 40, 49; the same decomposition for A vs B; "
                          "Stage-1 content split into MotionBricks endpoint-correction ramp vs uncorrected remainder (preflight, no model); "
                          "S2C-S2B effects at correction-dominated frames are attributed to 'Stage 1 incl. endpoint correction'",
}
CONTROL_RULE = {
    "K0_primary": f"K0B (PROXY_B, k_s=0) and K0C (PROXY_C, k_s=0): identical eps sha and raw outputs equal on valid joints within {TS.K0_PAIR_TOL} "
                  "(normalised) -> Stage-1 content cannot reach the output by any other path; failure -> BLOCKED",
    "K0_secondary": f"K0B vs frozen W136 U raw output <= {TS.K0_W136_TOL} normalised and decoded wrist/elbow <= {TS.K0_W136_MM} mm: reported "
                    "(WARNING, not BLOCKED, unless the primary pair also fails); x1 equality with W136's x1_known reported separately",
    "noise_floor": "S2B_rep: a separate-process replicate of S2B; max |S2B - S2B_rep| (normalised G entries; decoded wrist/elbow mm) is the floor",
    "eps": "every seed-10 run has the same eps sha; NREF (seed 11) eps sha differs and |corr(eps_NREF, eps_seed10)| < EPS_CORR_MAX = 0.03 on the 36,000 valid entries, else the "
           "native control is INVALID (reported; no reseed)",
    "sampler": "per run: scheduled_sampler_calls == 1, generate_samples calls == 1, network forward calls == 100 (50 steps x CFG 2), "
               "per-state mask sha recorded; K1 (k_s=50 copy) is a synthetic test only",
    "x1": "for every primary run x1_known_used is bit-equal to the official x1_known (no injection); S2A/S2B/S2C tensors are bit-equal on kept "
          f"valid entries pairwise (BRIDGE_LEAK otherwise) and equal the offline float64-normalised bridge tensors within X1_OFFLINE_TOL = {TS.X1_OFFLINE_TOL} "
          "(float32 rounding); K0 primary additionally requires K0B/S2B/S2B_rep and K0C/S2C to receive identical x1 and K0C's x1 to differ from K0B's on G",
}
QUALITY_RULE = {
    "usability_floor": f"arm-local wrist (W129 armlocal_quality) and elbow (W135 elbow_metrics) peak steps (which include entry 171->172 and exit "
                       f"199->200) <= {TS.UNUSABLE_STEP_MM} mm, zero wrist and elbow reversal flicks, no near-straight (>175 deg) or hinge-opposite interior frame",
    "intent": f"camera-free press intent on the fresh world readback: P1 hand_l head above upperarm_l head on every interior frame (frozen "
              f"historical values: A, B pass; W129 C, U, T fail); P2 min interior wrist z >= z(171) - {TS.P2_DROP_MM} mm. Thresholds set after "
              "seeing only frozen historical branches, never any W139 output",
    "non_inferiority": "every arm-local wrist/elbow entry, exit, peak step <= 2 x W129 B's + 10 mm (Q1/Q3 form)",
    "acceptable": "usability_floor AND P1 AND P2 AND non_inferiority",
    "fidelity_proxy": "mean interior world wrist distance to A (the GT-free source-derived reconstruction; agreement with a hypothesis, NOT fidelity "
                      "to ground truth; Q5 2D camera is INVALID and source 2D fidelity is UNKNOWN)",
    "superiority": f"S2C metric-superior to X in {{S2A, S2B, D, W129 B}} iff S2C acceptable AND (X not acceptable OR S2C fidelity_proxy <= X's - "
                   f"{TS.FIDELITY_MARGIN_MM} mm); METRIC_SUPERIORITY iff superior to all four",
    "agent_review": "NOMINALLY blinded (letters from an os.urandom secret sealed in SEALED_blind_key/; the reviewing agent has filesystem access, "
                    "so this is not a strong blind); review JSON written before the key is opened; REPORTED ONLY, never decisive",
    "human_review": "none planned: perceptual acceptance PENDING; Q8 NOT REVIEWED",
}
JUDGMENT_RULE = {
    "BLOCKED": "a prerequisite failed: preflight not OK, declaration/instrument/committed-blob/preflight/controls binding, a negative control "
               "not failing closed, K0 primary, eps identity / sampler-call / recorded-state gates of the PRIMARY runs (K0B, K0C, S2B, S2B_rep, S2A, "
               "S2C), x1 official/offline/BRIDGE_LEAK gates, or any mechanical contract gate of S2A/S2B/S2C (footprint, Action census, rig validity, "
               "hard pin, reversibility, bridge roundtrip, projection consistency), integrity after the run, native-rate videos",
    "NOT_YET_SUPPORTED": "prerequisites OK but learned_path is false (no learned Stage-1 information path beyond passive carry-through)",
    "PARTIALLY_SUPPORTED": "learned_path true but METRIC_SUPERIORITY false, OR METRIC_SUPERIORITY true without a human native-rate review "
                           "(sub-label METRIC_SUPERIORITY_INDICATED; perceptual acceptance PENDING)",
    "SUPPORTED": "learned_path AND METRIC_SUPERIORITY AND a human native-rate review (not available in this batch)",
    "scope": "any judgment applies ONLY to MotionBricks(W129, frozen) -> this bridge -> UniMate v3 through the NON-OFFICIAL scheduled-replacement "
             "sampler at t_s = 0.5 (no t_s sweep), one clip (S1), one seed; not to UniMate's official modes and not to the general Two-Stage "
             "architecture",
    "native_control": "run AFTER the primary result is written, isolated (its failure is recorded as FAILED); status INFORMATIVE / "
                      "CONTROL_UNINFORMATIVE (admissibility) / INVALID (noise independence, sampler or state gates); reported separately; it scopes "
                      "the interpretation (Stage-2 capability) and never by itself changes the judgment",
    "human_review": "hard-coded false in assess for this batch (no human native-rate review is planned): SUPPORTED is unattainable",
}
REJECTED_ALTERNATIVES = {
    "official_inbetween_sparse_stage1_keyframes": "fully official and hook-free, but (a) at kept interior keyframes every joint incl. the imported "
        "upperarm_l/lowerarm_l slots is an exact Stage-1 copy (copy confound: Stage 2 has zero effect there) and (b) at free frames the WHOLE body is "
        "regenerated, a larger scratch domain matching W137's failure mode; rejected as the primary path before any output; not run",
    "official_motion_edit_keep_stage1_arm": "exact passthrough of the Stage-1 arm: no Stage-2 effect on the imported channels by construction",
    "official_inbetween_frames_16_43": "discards all Stage-1 interior content (W137): no information transfer",
    "w136_mask_k0": "discards all Stage-1 interior content; kept only as the negative control K0",
}


def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit(f"preflight not OK: {pf.get('stop')}")
    pdir = Path(a.preflight).parent
    mo = model_outputs_exist()
    if mo:
        raise SystemExit(f"STOP DECLARATION_INVALID: model output exists {mo}")
    d = {"schema": "animcv_exp_w139_declaration_v1", "time": now(), "outcomes_inspected": False, "model_output_exists": False, "batch": BATCH,
         "branch": pf["branch"], "repo_head_at_declaration": git("rev-parse", "HEAD"), "preflight_sha256": M.sha256_file(a.preflight),
         "human_decisions": [pf["w138"]["human_decision"], "commits on Dual_Stage_Generator are local only (no push; a push would create a remote branch)"],
         "bindings": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in pf["frozen"].items()},
         "fresh_dump_sha256": {k: M.sha256_file(pdir / f"dump_{k}.json") for k in ("A", "B", "C", "U", "T")},
         "fresh_readback_sha256": {k: M.sha256_file(pdir / f"readback_{k}.json") for k in ("A", "B", "C", "U", "T", "Pin")},
         "x1_offline_sha256": pf["x1_offline_sha256"], "features_npy_sha256": pf["features_npy_sha256"],
         "identities": {"stage1": {"model": "NVlabs MotionBricks-SOMA-RP-v1 (mmm_v6_soma_fp32.onnx 856419dc...), code 54fc563, W129 R1 output",
                                   "engine_output_sha256": W129_ENGINE_SHA, "projection": "W129 closed-form SOMA->BaseRig left-arm mapping",
                                   "action": "W129 C_S1.blend e04efb35...", "re_run": False},
                        "stage2": {"code": {"repo": "Friedrich-M/UniMate", "commit": H6.UM_COMMIT, "files_sha256": pf["code"]["files_sha256"],
                                            "extra_sha256": pf["code"]["extra_sha256"]},
                                   "model": {"repo": H6.HF_REPO, "revision": H6.HF_REV, "checkpoint_sha256": H6.CKPT_SHA, "config_sha256": H7.CONFIG_SHA,
                                             "stats_sha256": H6.STATS_SHA, "licence": "CC BY-NC 4.0"},
                                   "text_encoder": {"repo": H6.T5_REPO, "revision": H6.T5_REV, "safetensors_sha256": H7.T5_SAFETENSORS_SHA}},
                        "runtime": pf["runtime"]["um_env"], "bridge_assets": {k: {"rig_sha256": v["rig_sha256"], "features_sha256": v["features_sha256"],
                                                                                  "asset": v["asset"], "clip": v["clip"]} for k, v in pf["bridge"].items()}},
         "rights": {k: pf["rights"][k] for k in ("determination", "intended_use", "commercial", "motionbricks")},
         "time_rule": "native 30 fps; Action = SOURCE - 78; window Action 156..215, feature frame i <-> Action 156+i; interior 172..199 = frames 16..43; "
                      "171 and 200 frozen; the 61st exported pose (216) only feeds frame 59's velocity",
         "edit_authority": {"channels": list(U.EDIT_CHANNELS), "frames": list(U.INTERIOR), "pin_action": [U.PIN_ACTION[0], U.PIN_ACTION[-1]],
                            "pin_target_m": [-0.0927, 0.2199, 1.4506], "pin_tolerance_m": 0.001, "same_for_all_conditions": True},
         "provenance": {"A_arm": "GT-free video-derived hypothesis (W102/W105 via W126 replay); the far left arm's 3D is UNKNOWN",
                        "B_arm": "deterministic endpoint slerp (no model)", "C_arm": "MotionBricks model output INCLUDING its deterministic endpoint "
                        "correction (31-54 deg ramps); a hypothesis, never an observation",
                        "inherited_hard_context": "the right arm (rig rest = UNKNOWN), the synthetic static pelvis, the source-derived torso/shoulder "
                        "popping (up to 244 mm/frame) are fed to Stage 2 as kept context in every condition; recorded as an inherited "
                        "distribution-mismatch factor, never as observed truth"},
         "bridge": {"steps": "Blender Action -> W136 export adapter (world-pose preserving, static pelvis) -> official rig_preprocess (rule annotation, "
                             "reviewed annotation reused, general profile) -> official compute_unimate_motion_feats -> official normalisation",
                    "gates": "time identity vs the condition's own stored keys; own==official decode; context drift; Blender roundtrip == condition "
                             "readback <= 1e-5 m; cond numeric fields equal across A/B/C; x1 differences confined to G; PROXY_B == W136 features",
                    "preflight_results": {"g_only": pf["g_only"], "cond_equality": pf["cond_equality"], "proxyB_vs_W136": pf["proxyB_features_vs_W136"],
                                          "world_maps": pf["world_maps"], "grounding": {k: v["grounding"] for k, v in pf["bridge"].items()}}},
         "stage2_conditioning": {"label": "NON-OFFICIAL SAMPLER PROCEDURE built on the official replacement primitive (flow SDEdit)",
                                 "module": "app.experimental_two_stage_sampler.scheduled_inbetween_sample_ode (verbatim copy of official "
                                           "inbetween_sample_ode + per-state mask); patched as the module attribute "
                                           "unimate.inference.motion_inbetweening.inbetween_sample_ode only inside the scratch runner",
                                 "keep_mask": "W136 joint x time mask (validated; 5424 free valid entries); G = (~keep)[valid joints]",
                                 "schedule": "state j (0 = initial replacement, i+1 = after Euler step i) uses keep | (G if j <= K_S); integer index",
                                 "K_S": TS.K_S, "t_s": 0.5, "n_steps": TS.N_STEPS,
                                 "derivation": "state K_S on G = (1-t_s) eps + t_s x_stage1 = the linear ICPlan training marginal; first non-discarded "
                                               "evaluation at t_s; k_s=0 is behaviourally identical to the official function (synthetic test)",
                                 "t_s_rationale": "config t_sampling = logit_normal(0,1) has its mode at t = 0.5 (the network is best trained there); "
                                                  "single prospectively fixed value, no sweep",
                                 "power_pre_output": pf["injected_signal_ratio"],
                                 "prompt": U.PROMPT, "seed": TS.SEED_COMPLETION, "cfg": H6.CFG, "weights": "EMA", "batch_size": 1,
                                 "official_path": "unimate.inference.sample.main --motion_edit (asset + pinned clip, gt_start_frame 0) with the W136 mask hook"},
         "runs": {k: {"proxy": v[0], "k_s": v[1], "mask": v[2], "official_mode": v[3], "seed": v[4], "x1": v[5] or f"official clip of PROXY_{v[0]}"}
                  for k, v in RUNS.items()},
         "rejected_alternatives": REJECTED_ALTERNATIVES,
         "conditions": {"S2A": "PROXY_A (original AnimCV reference) -> Stage 2 -> Action", "S2B": "PROXY_B (W129 slerp) -> Stage 2 -> Action",
                        "S2C": "PROXY_C (W129 MotionBricks) -> Stage 2 -> Action", "D": "W129 C_S1.blend (Stage 1 direct, frozen)",
                        "references": "A, W129 B (plain slerp), W136 U and W137 T (NON-EQUIVALENT: different masks/domains)"},
         "import": "decoded upperarm_l/lowerarm_l basis (unchanged W136 decode_basis) at 172..199 via the unchanged W129 build_variant on a byte copy "
                   "of A; Xpin on W126V-accepted; Xrev restores A's keys",
         "native_control": {"NREF": "seed 11, empty keep mask, k_s=0 (unconstrained 50-step Euler generation; independent noise)",
                            "re_encode": "official decode -> FK(tpos_offsets, integrated root) -> official features; 61st pose = repeat of the 60th, "
                                         "then frame 59's velocity channels restored from the raw sample (no pose 60 exists)",
                            "admissibility": "re-encode RIC vs FK <= 1e-4; neck_02-above-pelvis height >= 0.5 x its frame-0 value on every frame; "
                                             "hidden-interval (window 15..44) wrist path >= 100 mm in world scale (C map s); poison check identical; "
                                             "else CONTROL_UNINFORMATIVE",
                            "hidden": "the W136 free set G of N_ref* (left-arm descendants, frames 16..43 + velocity at 15)",
                            "N0": "k_s=0, x1 = N_ref* (W136 mask)", "NL": "k_s=25, x1 = N_B: hidden local rotations slerped between kept frames 15 and "
                            "44 -> FK -> official features (built twice incl. a NaN-poisoned copy; must be identical)",
                            "NIB": "official --inbetween frame mask (window 0..15, 44..59 kept), x1 = N_ref* (injected); DISCLOSED LEAK: kept frame 15 "
                                   "includes its velocity channels, which encode N_ref*'s hidden frame-16 positions (as W137) -> an official-mode reference "
                                   "with a one-frame velocity leak, not a leak-free completion",
                            "metrics": "FK wrist/elbow error vs N_ref* over window 16..43 (world scale), decoded upperarm/lowerarm angle error, "
                                       "boundary steps vs N_ref*'s own, flicks, RIC-FK gap; NL refinement gain = error(N_B init) - error(NL)",
                            "x1_injection": "disclosed hook: x1_known replaced by the normalised features (normalisation verified on the official tensor)"},
         "information_rule": INFO_RULE, "control_rule": CONTROL_RULE, "quality_rule": QUALITY_RULE, "judgment_rule": JUDGMENT_RULE,
         "thresholds": {k: getattr(TS, k) for k in ("GAMMA_MIN", "NET_RATIO_MIN", "NET_MM_MIN", "NOISE_FACTOR", "CROSS_TOL", "K0_PAIR_TOL",
                                                    "K0_W136_TOL", "K0_W136_MM", "UNUSABLE_STEP_MM", "P2_DROP_MM", "FIDELITY_MARGIN_MM", "FLICK_MIN_MM", "EPS_CORR_MAX")},
         "metrics": ["arm-local wrist/elbow entry/exit/peak/flicks, flex, hinge (W129/W135) for A, B, D, S2A, S2B, S2C, U, T",
                     "P1/P2 intent, fidelity proxy (deviation from A), deviation from B, world shoulder step (inherited)",
                     "raw vs projected: per condition official FK of the raw sample (own integrated root) vs hybrid (own PROXY_X + generated arm slots), "
                     "world map of that condition, W137 attribution/body/raw rule",
                     "inherited defects: non-arm world points bit-equal across all candidates (by M3); root/torso identical by construction",
                     "contact: NOT SUPPORTABLE (no hand-object contact model)"],
         "mechanical_gates": {"M2": "one named Action, 32 CONSTANT F-curves keyed 1..240, 30 fps", "M3": "changed keys only upperarm_l/lowerarm_l at "
                              "172..199; non-arm world points/object matrix bit-equal to A on 240 frames + subframes; 171/200 bit-equal",
                              "M5": "upper/lower world length range <= 0.1 mm; unit finite quaternions", "M6": "Xpin residuals at 132..140 equal W126V's, "
                              "<= 1e-6 m; pin frames bit-equal", "M7": "Xrev world-bit-equal to A", "M8": "bridge roundtrip (preflight R0) for X",
                              "M9": "frozen identities unchanged after the run", "M11": "videos decode at 30.000 fps with every frame",
                              "M12": "hybrid FK (own proxy + generated arm slots) mapped by the condition's world map == X readback <= 1e-4 m"},
         "video": {"order": list(ORDER), "names": NAMES, "extra": ["blinded S2A/S2B/S2C/D video + sheets (letters keyed by run_result sha)",
                                                                  "interior loop Action 164..207 at 30 fps"], "codec": "mp4v (no ffmpeg/H.264 available)"},
         "budget": {"model_processes_max": 12, "vram_peak_gb_max_per_process": 2.0, "gpu_wall_min_max": 60, "disk_gb_max": 5,
                    "retry": "at most one logged infrastructure retry per run with no parameter change"},
         "pre_declaration_disclosures": [
             "input-only exploration (no model): native_prep on the frozen W136 PROXY_B features: re-encode RIC==FK to 1e-15, idempotent to 9e-16; "
             "the B-analogue builder reproduces PROXY_B on G to 9e-7 (i.e. it is built exactly as B); NaN-poisoned build identical",
             "the 6 W139 design choices were revised after an adversarial design review (passive-null information criteria, seed 11 for the native "
             "reference, camera-free intent terms, W129 B in the superiority set, agent review reported-only, K0 inside W139) before any output",
             "PROXY_A/PROXY_B/PROXY_C are re-built by W139 (PROXY_B must equal W136's features); W136/W137 files are only read",
             "DEV DRY RUNS before this declaration (C:/Users/dna10/animcv-w139/dev): preflight/declare/controls try-runs and --dev-stub runs "
             "of the whole run stage with a WEIGHT-FREE stub network (no pretrained network evaluated). They exposed instrument defects that were "
             "fixed (float32/float64 normalisation, flick units, integrity bindings, native isolation, blinding, stub labelling). The stub runs also "
             "computed quality/intent/acceptability for the FROZEN historical branches A, B, D(=W129 C), U, T with these thresholds; no Stage-2 "
             "output of the pretrained model existed or was seen",
             "an independent adversarial code review (3 reviewers) was run on the instruments before this declaration",
             "SUPERSEDED declaration v1 (sha256 39f21e28b9e37141b800068db7553243b21bda4ac379a26dd3e94fa54ff4b8e0, commit 65ae489): the official run "
             "stage refused it BEFORE any inference because its committed-blob check compared the CRLF working-tree bytes with the LF git blob "
             "(core.autocrlf); fixed by comparing the LF-normalised bytes; no model output, run directory or outcome existed; no rule, threshold, "
             "run or mask changed"],
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    sha = dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha, "K_S": TS.K_S, "runs": list(RUNS), "power": pf["injected_signal_ratio"]}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ controls ---
def stage_controls(a):
    check_declaration(a)
    out = new_dir(a.out_dir)
    pdir = Path(a.preflight).parent
    pf = load(a.preflight)
    res = {}
    w6 = SimpleNamespace(out_dir=str(out / "w136_controls"), preflight=str(W136R / "preflight" / "preflight.json"),
                         declaration=str(W136R / "declare" / "declaration.json"), declaration_sha=H7.W136_DECL_SHA)
    rc = H6.stage_controls(w6)
    c6 = load(out / "w136_controls" / "controls.json")
    res["w136_controls"] = {"rc": rc, "n": len(c6["negative_controls"]), "all_fail_closed": c6["all_fail_closed"]}
    neg = {}
    E = H6._expect
    names, par = pf["names"], np.asarray(pf["parents"])
    km = U.build_keep_mask(names, par, 71)
    TS.check_w136_free_set(km, names, par)
    for nm, (j, c, t) in (("mask_spine_freed", ("spine_03", 0, 20)), ("mask_upperarm_slot_freed", ("upperarm_l", 3, 20)),
                          ("mask_lowerarm_kept_interior", ("lowerarm_l", 4, 20))):
        m2 = km.copy()
        m2[names.index(j), c, t] = not m2[names.index(j), c, t]
        E(nm, lambda m2=m2: TS.check_w136_free_set(m2, names, par), ("MASK_DOMAIN", "MASK_INCOMPLETE"), neg)
    for nm, v in (("k_s_negative", -1), ("k_s_too_large", 51), ("k_s_bool", True), ("k_s_float", 25.0)):
        E(nm, lambda v=v: TS.validate_k_s(v), "SCHEDULE", neg)
    x1 = dict(np.load(pdir / "x1_normalised_offline.npz"))
    g = TS.sched_set(km[:U.N_JOINTS], U.N_JOINTS)

    def leak_check(x):
        r = g_only_differences(x, g, km[:U.N_JOINTS])
        if not r["ok"]:
            raise TS.TwoStageStop("BRIDGE_LEAK", json.dumps({k: v for k, v in r.items() if k != "ok"}))
        return r
    leak_check(x1)
    for nm, (j, c, t) in (("leak_kept_spine", ("spine_02", 1, 30)), ("leak_kept_upperarm_slot", ("upperarm_l", 5, 25)),
                          ("leak_kept_lowerarm_frame_171_rot", ("lowerarm_l", 4, 15)), ("leak_kept_after_window", ("hand_l", 0, 44))):
        x2 = {k: v.copy() for k, v in x1.items()}
        x2["C"][0, names.index(j), c, t] += 1e-3
        E(nm, lambda x2=x2: leak_check(x2), "BRIDGE_LEAK", neg)
    dA = load(pdir / "dump_A.json")
    dec = {b: np.tile(np.array([1.0, 0, 0, 0]), (U.T_MODEL, 1)) for b in U.EDIT_CHANNELS}
    E("import_frame_171", lambda: U.import_values(dec, dA["stored"], frames=range(171, 200)), "DOMAIN_FRAMES", neg)
    E("import_frame_200", lambda: U.import_values(dec, dA["stored"], frames=range(172, 201)), "DOMAIN_FRAMES", neg)
    E("import_pin_frame", lambda: U.import_values(dec, dA["stored"], frames=range(132, 141)), "PIN_OVERLAP", neg)
    E("import_hand_channel", lambda: U.import_values({**dec, "hand_l": dec["upperarm_l"]}, dA["stored"]), "DOMAIN_CHANNEL", neg)
    bad = {b: v.copy() for b, v in dec.items()}
    bad["lowerarm_l"][20] = np.nan
    E("import_nan", lambda: U.import_values(bad, dA["stored"]), "RIG_PROJECTION_FAILED", neg)

    def eps_gate(e10, e11):
        c = float(np.corrcoef(e10.ravel(), e11.ravel())[0, 1])
        if np.array_equal(e10, e11) or abs(c) >= TS.EPS_CORR_MAX:
            raise TS.TwoStageStop("NATIVE_NOISE_SHARED", f"corr {c}")
        return c
    rng = np.random.default_rng(0)
    e = rng.standard_normal(36000)
    eps_gate(e, rng.standard_normal(36000))
    E("native_eps_identical", lambda: eps_gate(e, e.copy()), "NATIVE_NOISE_SHARED", neg)
    E("native_eps_correlated", lambda: eps_gate(e, e + 0.5 * rng.standard_normal(36000)), "NATIVE_NOISE_SHARED", neg)
    cq = cond_equality_ctl(pf, out)
    E("cond_tpos_offsets_mismatch", cq, "COND_MISMATCH", neg)
    # real Blender refusal: build_variant onto a base with the wrong expected SHA
    spec = {"out_blend": str(out / "refused.blend"), "interior_action_frames": list(U.INTERIOR),
            "values_wxyz": U.import_values({b: np.tile(np.array([1.0, 0, 0, 0]), (U.T_MODEL, 1)) for b in U.EDIT_CHANNELS}, dA["stored"]),
            "branch": "W139 ctl", "action_name": "AnimCV_EXP W139 refused", "expected_blend_sha256": "0" * 64, "note": "negative control"}
    dump(out / "ctl_wrong_base_spec.json", spec)
    try:
        blender(H6.STAGE, "build_variant", out / "ctl_wrong_base.log", blend=pdir / "inputs" / "A.blend", spec=out / "ctl_wrong_base_spec.json",
                report=out / "ctl_wrong_base_report.json")
        neg["blender_wrong_base_sha"] = {"pass": False, "detail": "build succeeded"}
    except RuntimeError:
        txt = (out / "ctl_wrong_base.log").read_text(encoding="utf-8", errors="replace")
        neg["blender_wrong_base_sha"] = {"pass": "REFUSED_BASELINE_IDENTITY" in txt and not (out / "refused.blend").exists(), "detail": "refused"}
    res["w139_controls"] = neg
    res["all_fail_closed"] = bool(c6["all_fail_closed"] and all(v["pass"] for v in neg.values()))
    res["n_total"] = res["w136_controls"]["n"] + len(neg)
    res["note"] = "no model was loaded or run in this stage"
    res["declaration_sha256"] = a.declaration_sha
    dump(out / "controls.json", res)
    print(json.dumps({"all_fail_closed": res["all_fail_closed"], "n_total": res["n_total"], "w136": res["w136_controls"],
                      "w139": {k: v["pass"] for k, v in neg.items()}}, indent=1))
    return 0 if res["all_fail_closed"] else 3


def cond_equality_ctl(pf, out):
    def f():
        br = {k: dict(v) for k, v in pf["bridge"].items()}
        tmp = out / "ctl_tmp_cond"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            c = np.load(Path(br["C"]["asset_dir"]) / "cond.npy", allow_pickle=True).item()
            c[CONDS["C"]] = dict(c[CONDS["C"]])
            c[CONDS["C"]]["tpos_offsets"] = np.asarray(c[CONDS["C"]]["tpos_offsets"]) + 1e-6
            np.save(tmp / "cond.npy", c, allow_pickle=True)
            br["C"]["asset_dir"] = str(tmp)
            r = cond_equality(br)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if not r["C"]["ok"]:
            raise TS.TwoStageStop("COND_MISMATCH", str(r["C"]["differing_fields"]))
        return r
    return f


# ------------------------------------------------------------------------------------------------ run ---
def _um_ok(r, what):
    if r["rc"] != 0:
        tail = Path(r["log"]).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        raise RuntimeError(f"{what} rc={r['rc']}: {tail}")
    return r


def _infer(out, name, pf, inputs, stub=False):
    proxy, k_s, mask, mode, seed, x1 = RUNS[name]
    b = pf["bridge"][proxy]
    args = [INFER9, "infer", "--out-dir", out / name, "--asset-dir", b["asset_dir"], "--name", b["asset"], "--clip", b["clip"],
            "--exp-dir", H6.MODEL_DIR, "--model-path", H6.W136 / "model" / H6.CKPT_REL, "--seed", seed, "--k-s", k_s, "--mask", mask,
            "--official-mode", mode]
    if x1:
        args += ["--x1-npy", inputs[x1]]
    if stub:
        args += ["--stub-model"]
    r = um(args, out / f"{name}.log", torch_legacy_load=True)
    r["processes"] = 1
    if r["rc"] != 0:
        txt = (out / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        if (out / name / "raw_model_io.npz").exists() or "UniStop" in txt or "TwoStageStop" in txt:
            raise RuntimeError(f"INFERENCE_FAILED {name}: {txt.strip().splitlines()[-3:]}")
        shutil.move(str(out / name), str(out / f"{name}_failed_attempt"))
        r2 = um(args, out / f"{name}_retry.log", torch_legacy_load=True)          # one logged infrastructure retry, same args
        if r2["rc"] != 0:
            raise RuntimeError(f"INFERENCE_FAILED {name} after one retry")
        r = {**r2, "retried": True, "processes": 2}
    rec = load(out / name / "infer_record.json")
    if bool(stub) != ("STUB_MODEL" in rec):
        raise RuntimeError(f"STUB_MISMATCH {name}")
    return r


def _traj(out, name):
    z = np.load(out / name / "trajectory.npz")
    return {k: z[k] for k in z.files}


def _decoded_world(out, tag, pf, proxy_key, samples, wm):
    """Official FK of each sample and of its hybrid (proxy + generated arm slots), mapped to world by the condition's map."""
    b = pf["bridge"][proxy_key]
    od = out / f"recover_{tag}"
    _um_ok(um([INFER9, "recover", "--out-dir", od, "--asset-dir", b["asset_dir"], "--name", b["asset"], "--proxy", b["features_npy"],
               "--samples", ",".join(f"{k}={v}" for k, v in samples.items())], out / f"recover_{tag}.log"), f"recover_{tag}")
    rr = np.load(od / "recovered.npz")
    return {k: H7.map_to_world(rr[k], wm) for k in rr.files if k.endswith("_fk")}, rr


def _arm_mm(w1, w2, names, frames):
    return max(float(np.linalg.norm(w1[i, names.index(b)] - w2[i, names.index(b)])) * 1e3 for i in frames for b in ("lowerarm_l", "hand_l"))


def _arm_mm_per_frame(w1, w2, names, frames):
    return {U.index_to_action(i): max(float(np.linalg.norm(w1[i, names.index(b)] - w2[i, names.index(b)])) * 1e3 for b in ("lowerarm_l", "hand_l"))
            for i in frames}


def quality_of(rb, dmp, rbA, rbB, hinge):
    qq = M.quality(rb, rbA, U.INTERIOR, WIN, hinge)
    qq.pop("per_frame", None)
    qq["arm_local"] = M.armlocal_quality(M.armlocal_wrist(dmp, *WIN), U.INTERIOR, WIN)
    qq["arm_local"].pop("per_frame", None)
    em = EP.elbow_metrics(dmp)
    em.pop("per_frame", None)
    qq["elbow"] = em
    fr = _frames(rb)
    fA, fB = _frames(rbA), _frames(rbB)
    i0, i1 = U.INTERIOR
    sh = np.array([fr[f]["world"]["upperarm_l"]["head"] for f in range(WIN[0], WIN[1] + 1)])
    qq["world_shoulder_step_max_mm"] = float(np.linalg.norm(np.diff(sh, axis=0), axis=1).max() * 1e3)
    dev = lambda ref: [float(np.linalg.norm(np.subtract(fr[f]["world"]["hand_l"]["head"], ref[f]["world"]["hand_l"]["head"]))) * 1e3 for f in range(i0, i1 + 1)]  # noqa: E731
    qq["fidelity_dev_A_mean_mm"], qq["fidelity_dev_A_max_mm"] = float(np.mean(dev(fA))), float(np.max(dev(fA)))
    qq["dev_B_mean_mm"], qq["dev_B_max_mm"] = float(np.mean(dev(fB))), float(np.max(dev(fB)))
    qq["wrist_net_rise_mm"] = (fr[i1]["world"]["hand_l"]["head"][2] - fr[i0]["world"]["hand_l"]["head"][2]) * 1e3
    qq["intent"] = TS.intent_metrics(fr)
    qq["max_steps"] = TS.max_steps(qq)
    return qq


def _git_bytes_sha(spec):
    r = subprocess.run(["git", "-C", str(ROOT), "show", spec], capture_output=True)
    import hashlib
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def _verify_bound(a, d, pf, pdir, stub):
    """Run/assess-time integrity: declaration committed, preflight/controls/readbacks/x1/features/UniMate/checkpoint as declared."""
    rel = "examples/experimental_two_stage/w139_declaration.json"
    info = {"committed_blob_sha256": _git_bytes_sha(f"HEAD:{rel}"),
            "path_clean": subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", rel], capture_output=True, text=True).stdout.strip() == ""}
    import hashlib
    wt = (ROOT / rel).read_bytes() if (ROOT / rel).is_file() else b""
    info["working_file_sha256"] = hashlib.sha256(wt).hexdigest()
    info["working_file_lf_sha256"] = hashlib.sha256(wt.replace(b"
", b"
")).hexdigest()   # core.autocrlf stores LF blobs
    if not stub and (info["working_file_sha256"] != a.declaration_sha or info["committed_blob_sha256"] != info["working_file_lf_sha256"]
                     or not info["path_clean"]):
        raise SystemExit(f"STOP DECLARATION_INVALID: committed declaration != --declaration-sha ({info})")
    if M.sha256_file(a.preflight) != d["preflight_sha256"]:
        raise SystemExit("STOP INTEGRITY_FAILED: preflight differs from the declared one")
    for k, h in d["fresh_readback_sha256"].items():
        if M.sha256_file(pdir / f"readback_{k}.json") != h:
            raise SystemExit(f"STOP INTEGRITY_FAILED readback {k}")
    for k, h in d["fresh_dump_sha256"].items():
        if M.sha256_file(pdir / f"dump_{k}.json") != h:
            raise SystemExit(f"STOP INTEGRITY_FAILED dump {k}")
    if M.sha256_file(pdir / "x1_normalised_offline.npz") != d["x1_offline_sha256"]:
        raise SystemExit("STOP INTEGRITY_FAILED x1 offline")
    for k, h in d["features_npy_sha256"].items():
        if M.sha256_file(pf["bridge"][k]["features_npy"]) != h:
            raise SystemExit(f"STOP INTEGRITY_FAILED features {k}")
    for k, v in d["identities"]["bridge_assets"].items():
        if {f: M.sha256_file(Path(pf["bridge"][k]["asset_dir"]) / f) for f in v["rig_sha256"]} != v["rig_sha256"]:
            raise SystemExit(f"STOP INTEGRITY_FAILED bridge {k}")
    for k, fk in COPY.items():
        if M.sha256_file(pdir / "inputs" / f"{k}.blend") != H6.FROZEN[fk][1]:
            raise SystemExit(f"STOP INTEGRITY_FAILED {k}")
    um_head = subprocess.run(["git", "-C", str(H6.UM_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    um_dirty = subprocess.run(["git", "-C", str(H6.UM_ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    files = {f: M.sha256_file(H6.UM_ROOT / f) for f in d["identities"]["stage2"]["code"]["files_sha256"]}
    if um_head != H6.UM_COMMIT or um_dirty or files != d["identities"]["stage2"]["code"]["files_sha256"]:
        raise SystemExit("STOP INTEGRITY_FAILED UniMate code")
    if M.sha256_file(H6.W136 / "model" / H6.CKPT_REL) != H6.CKPT_SHA or M.sha256_file(H7.T5_LOCAL / "model.safetensors") != H7.T5_SAFETENSORS_SHA:
        raise SystemExit("STOP INTEGRITY_FAILED checkpoint / T5")
    return info


PRIMARY = ("K0B", "K0C", "S2B", "S2B_rep", "S2A", "S2C")
NATIVE = ("NREF", "N0", "NL", "NIB")


def stage_run(a):
    d = check_declaration(a)
    stub = bool(a.dev_stub)
    out_res = Path(a.out_dir).resolve()
    if stub and (W139 / "results").resolve() in out_res.parents:
        raise SystemExit("--dev-stub output may never be written under the W139 results root")
    if not stub and (W139 / "results").resolve() not in out_res.parents:
        raise SystemExit("a real run must write under the W139 results root")
    ctl = load(a.controls)
    if not ctl["all_fail_closed"] or (not stub and ctl.get("declaration_sha256") != a.declaration_sha):
        raise SystemExit("STOP INTEGRITY_FAILED: controls not fail-closed or not bound to this declaration")
    pf = load(a.preflight)
    pdir = Path(a.preflight).parent
    bound = _verify_bound(a, d, pf, pdir, stub)
    out = new_dir(Path(a.out_dir))
    log = {"stage": "run", "start": now(), "head": git("rev-parse", "HEAD"), "bound": bound, "steps": []}
    LBL = "STUB_DRY_RUN " if stub else ""
    inp = pdir / "inputs"
    names, par = pf["names"], np.asarray(pf["parents"])
    km = U.build_keep_mask(names, par, U.N_JOINTS)
    g = TS.sched_set(km, U.N_JOINTS)
    keep_v = km
    res = {"declaration_sha256": a.declaration_sha, "preflight_sha256": M.sha256_file(a.preflight), "runs": {}, "STUB_DRY_RUN": stub}
    # ================= primary model runs =================
    inputs = {}
    for nm in PRIMARY:
        log["steps"].append([nm, _infer(out, nm, pf, inputs, stub)])
    rec = {nm: load(out / nm / "infer_record.json") for nm in PRIMARY}
    io = {nm: np.load(out / nm / "raw_model_io.npz") for nm in PRIMARY}
    tr = {nm: _traj(out, nm) for nm in PRIMARY}

    def run_summary(nm, r):
        x = {k: r.get(k) for k in ("k_s", "mask", "official_mode", "x1_injected", "calls_generate_samples", "scheduled_sampler_calls", "network_forward_calls",
                                   "eps_sha256", "sampler_seconds", "total_seconds", "peak_vram_bytes_total", "normalisation_identity_max_abs",
                                   "saved_vs_recorded_output_max_abs", "decode", "device", "torch", "cuda", "cudnn", "keep_valid_true", "sched_valid_true")}
        x["raw_model_io_sha256"] = M.sha256_file(out / nm / "raw_model_io.npz")
        return x
    for nm in PRIMARY:
        res["runs"][nm] = run_summary(nm, rec[nm])
    gates = {"sampler_calls": all(rec[n]["calls_generate_samples"] == 1 and rec[n]["scheduled_sampler_calls"] == 1
                                  and rec[n]["network_forward_calls"] == 2 * TS.N_STEPS for n in PRIMARY),
             "eps_seed10_identical": len({rec[n]["eps_sha256"] for n in PRIMARY}) == 1}
    st_chk = {}
    for nm in PRIMARY:
        x1u = io[nm]["x1_known_used_normalised"][:, :U.N_JOINTS]
        kmn = np.broadcast_to(io[nm]["keep_mask"], tuple(io[nm]["motion_shape"]))[0, :U.N_JOINTS]
        gn = io[nm]["sched_mask"][:U.N_JOINTS]
        st_chk[nm] = TS.check_recorded_states(tr[nm]["states"], tr[nm]["eps"], x1u, kmn, gn, int(rec[nm]["k_s"]), tr[nm]["t"])
        st_chk[nm]["final_equals_raw_output"] = bool(np.array_equal(tr[nm]["states"][-1], io[nm]["raw_output_normalised"][:, :U.N_JOINTS]))
        st_chk[nm]["sched_equals_declared_G"] = bool(np.array_equal(gn, g))
    gates["recorded_states"] = {k: bool(v["ok"] and v["final_equals_raw_output"] and v["sched_equals_declared_G"]) for k, v in st_chk.items()}
    x1r = {k: io[f"S2{k}"]["x1_known_used_normalised"][:, :U.N_JOINTS] for k in ("A", "B", "C")}
    off = np.load(pdir / "x1_normalised_offline.npz")
    gates["x1_used_equals_official"] = all(np.array_equal(io[n]["x1_known_used_normalised"], np.load(out / n / "x1_known_official.npy")) for n in PRIMARY)
    gates["x1_g_only"] = g_only_differences(x1r, g, keep_v)
    gates["x1_equals_offline_max_abs"] = float(max(np.abs(x1r[k] - off[k]).max() for k in x1r))
    gates["x1_equals_offline"] = bool(gates["x1_equals_offline_max_abs"] <= TS.X1_OFFLINE_TOL)
    gates["x1_runs_consistent"] = bool(np.array_equal(io["K0B"]["x1_known_used_normalised"], io["S2B"]["x1_known_used_normalised"])
                                       and np.array_equal(io["K0C"]["x1_known_used_normalised"], io["S2C"]["x1_known_used_normalised"])
                                       and np.array_equal(io["S2B_rep"]["x1_known_used_normalised"], io["S2B"]["x1_known_used_normalised"]))
    k0 = {"pair_max_abs": float(np.abs(io["K0B"]["raw_output_normalised"][:, :U.N_JOINTS] - io["K0C"]["raw_output_normalised"][:, :U.N_JOINTS]).max()),
          "K0C_x1_differs_on_G": bool(np.abs(io["K0C"]["x1_known_used_normalised"][0, :U.N_JOINTS][g] - io["K0B"]["x1_known_used_normalised"][0, :U.N_JOINTS][g]).max() > 0)}
    k0["primary_ok"] = bool(k0["pair_max_abs"] <= TS.K0_PAIR_TOL and rec["K0B"]["eps_sha256"] == rec["K0C"]["eps_sha256"] and gates["x1_runs_consistent"]
                            and k0["K0C_x1_differs_on_G"])
    w136 = np.load(W136_U_IO)
    k0["vs_W136_U_max_abs"] = float(np.abs(io["K0B"]["raw_output_normalised"][:, :U.N_JOINTS] - w136["raw_output_normalised"][:, :U.N_JOINTS]).max())
    k0["x1_vs_W136_max_abs"] = float(np.abs(io["K0B"]["x1_known_used_normalised"][:, :U.N_JOINTS] - w136["x1_known_proxyB_normalised"][:, :U.N_JOINTS]).max())
    # ================= decoded worlds =================
    wms = pf["world_maps"]
    samp = {nm: rec[nm]["sample_npy"] for nm in PRIMARY}
    std = io["S2B"]["std"][0, :U.N_JOINTS]
    mean = io["S2B"]["mean"][0, :U.N_JOINTS]

    def denorm(xn):
        return np.transpose(np.asarray(xn, np.float64)[0] * std[:, :, None] + mean[:, :, None], (2, 0, 1)).astype(np.float32)
    t_s = float(tr["S2B"]["t"][TS.K_S])
    np.save(out / "nullC_denorm.npy", denorm(TS.null_output(io["S2B"]["raw_output_normalised"][:, :U.N_JOINTS], x1r["B"], x1r["C"], g, t_s)))
    np.save(out / "nullA_denorm.npy", denorm(TS.null_output(io["S2B"]["raw_output_normalised"][:, :U.N_JOINTS], x1r["B"], x1r["A"], g, t_s)))
    w136_sample = W136R / "run" / "model" / "official" / "motion_edit" / "motions" / "S1_PROXY_B-Animation-000-rep_0-0.npy"
    worlds = {}
    worlds["C"], _ = _decoded_world(out, "C", pf, "C", {"S2C": samp["S2C"], "nullC": out / "nullC_denorm.npy", "K0C": samp["K0C"]}, wms["C"])
    worlds["B"], _ = _decoded_world(out, "B", pf, "B", {"S2B": samp["S2B"], "S2B_rep": samp["S2B_rep"], "K0B": samp["K0B"], "W136U": w136_sample}, wms["B"])
    worlds["A"], _ = _decoded_world(out, "A", pf, "A", {"S2A": samp["S2A"], "nullA": out / "nullA_denorm.npy"}, wms["A"])
    fi = list(range(U.action_to_index(U.INTERIOR[0]), U.action_to_index(U.INTERIOR[1]) + 1))
    noise_mm = _arm_mm(worlds["B"]["S2B_hybrid_fk"], worlds["B"]["S2B_rep_hybrid_fk"], names, fi)
    noise_feat = float(np.abs(io["S2B"]["raw_output_normalised"][:, :U.N_JOINTS] - io["S2B_rep"]["raw_output_normalised"][:, :U.N_JOINTS])[0][g].max())
    k0["vs_W136_U_decoded_mm"] = _arm_mm(worlds["B"]["K0B_hybrid_fk"], worlds["B"]["W136U_hybrid_fk"], names, fi)
    k0["secondary_warning"] = bool(not (k0["vs_W136_U_max_abs"] <= TS.K0_W136_TOL and k0["vs_W136_U_decoded_mm"] <= TS.K0_W136_MM))
    # ================= information decomposition =================
    dec = {}
    cd = pf["correction_decomposition"]["correction_dominated_frames"]
    both = sorted(set(cd["upperarm_l"]) & set(cd["lowerarm_l"]))
    neither = [f for f in range(U.INTERIOR[0], U.INTERIOR[1] + 1) if f not in set(cd["upperarm_l"]) | set(cd["lowerarm_l"])]
    for p, x in (("C", "S2C"), ("A", "S2A")):
        dd = TS.transfer_decomposition(tr["S2B"], tr[x], x1r["B"], x1r[p], g, TS.K_S)
        dd["decoded_network_term_mm"] = _arm_mm(worlds[p][f"{x}_hybrid_fk"], worlds[p][f"null{p}_hybrid_fk"], names, fi)
        dd["decoded_output_diff_vs_S2B_mm"] = _arm_mm(worlds[p][f"{x}_hybrid_fk"], worlds["B"]["S2B_hybrid_fk"], names, fi)
        if p == "C":
            pf_out = _arm_mm_per_frame(worlds[p][f"{x}_hybrid_fk"], worlds["B"]["S2B_hybrid_fk"], names, fi)
            pf_net = _arm_mm_per_frame(worlds[p][f"{x}_hybrid_fk"], worlds[p][f"null{p}_hybrid_fk"], names, fi)
            grp = lambda fr_, m: {"frames": fr_, "mean_mm": float(np.mean([m[f] for f in fr_])) if fr_ else None,  # noqa: E731
                                  "max_mm": float(np.max([m[f] for f in fr_])) if fr_ else None}
            dd["correction_attribution"] = {"rule": "frames where BOTH channels are correction-dominated vs NEITHER (preflight decomposition)",
                                            "S2C_minus_S2B": {"correction_dominated": grp(both, pf_out), "learned_dominated": grp(neither, pf_out)},
                                            "network_term": {"correction_dominated": grp(both, pf_net), "learned_dominated": grp(neither, pf_net)},
                                            "per_frame_S2C_minus_S2B_mm": pf_out, "per_frame_network_term_mm": pf_net}
        dec[p] = dd
    sB, sC = tr["S2B"]["states"][TS.K_S], tr["S2C"]["states"][TS.K_S]
    cross = np.abs((sC.astype(np.float64) - sB) - t_s * (x1r["C"].astype(np.float64) - x1r["B"]))[0]
    i1 = {"first_free_input_equals_state_ks": bool(np.array_equal(tr["S2B"]["first_free_step_input"], sB) and np.array_equal(tr["S2C"]["first_free_step_input"], sC)),
          "cross_on_G_max": float(cross[g].max()), "cross_on_keep_max": float(np.abs((sC - sB)[0][keep_v]).max())}
    tokB, tokC = np.load(out / "S2B" / "first_free_tokens.npz"), np.load(out / "S2C" / "first_free_tokens.npz")
    cells = g.any(axis=1)
    tk = {}
    for br in ("cond", "uncond"):
        dtk = np.abs(tokC[br] - tokB[br]).max(axis=-1)[0]
        tk[br] = {"shape": list(tokB[br].shape)}
        if dtk.ndim == 2 and dtk.shape[0] == U.T_MODEL + 1 and dtk.shape[1] >= U.N_JOINTS:
            dd_ = dtk[1:, :U.N_JOINTS].T
            tk[br].update({"max_on_G_cells": float(dd_[cells].max()), "max_off_G_cells": float(dd_[~cells].max()), "max_prefix": float(dtk[0, :U.N_JOINTS].max())})
        else:
            tk[br]["layout"] = "unrecognised"
    i1["tokens"] = tk
    tok_ok = all(v.get("max_on_G_cells", 0) > 0 and v.get("max_off_G_cells", 1) <= 1e-5 for v in tk.values())
    i1["ok"] = bool(i1["first_free_input_equals_state_ks"] and i1["cross_on_G_max"] <= TS.CROSS_TOL and i1["cross_on_keep_max"] == 0.0 and tok_ok)
    info = TS.information_path(i1["ok"], dec["C"], dec["C"]["decoded_network_term_mm"], noise_mm, noise_feat)
    # ================= decode / import / build =================
    dumps = {k: load(pdir / f"dump_{k}.json") for k in ("A", "B", "C", "U", "T")}
    rb = {k: load(pdir / f"readback_{k}.json") for k in ("A", "B", "C", "U", "T", "Pin")}
    vals, blends = {}, {"A": inp / "A.blend", "B": inp / "B.blend", "D": inp / "C.blend", "U": inp / "U.blend", "T": inp / "T.blend"}
    stA = dumps["A"]["stored"]
    for x in CANDIDATES:
        dz = np.load(out / x / "decoded.npz")
        if "S_upperarm_l" not in dz.files:
            res.update({"STOP": {"code": "RIG_PROJECTION_FAILED", "run": x, "model_executed": True}, "gates": gates})
            dump(out / "run_result.json", res)
            dump(out / "run_result_final.json", res)
            return 3
        vals[x] = U.import_values({b: dz[f"S_{b}"] for b in U.EDIT_CHANNELS}, stA)
        dump(out / f"import_{x}.json", vals[x])
    vA = {ch: {str(f): [stA[ch][str(c)][str(f)] for c in range(4)] for f in range(U.INTERIOR[0], U.INTERIOR[1] + 1)} for ch in U.EDIT_CHANNELS}
    for x in CANDIDATES:
        for br, base, base_sha, v, nm_ in ((x, inp / "A.blend", H6.FROZEN["A"][1], vals[x], f"AnimCV_EXP W139 {LBL}{x} S1"),
                                          (f"{x}pin", inp / "W126V_accepted.blend", H6.FROZEN["W126V_accepted"][1], vals[x], f"AnimCV_EXP W139 {LBL}{x}_on_W126V S1")):
            spec = {"out_blend": str(out / f"{br}_S1.blend"), "interior_action_frames": list(U.INTERIOR), "values_wxyz": v, "branch": f"W139 {LBL}{br}",
                    "action_name": nm_, "expected_blend_sha256": base_sha,
                    "note": (f"{LBL}W139 scratch {x}: " + ("weight-free STUB network (dev dry run, NOT model output)" if stub else
                             "UniMate v3 scheduled-replacement (t_s=0.5) generated upperarm_l/lowerarm_l at 172..199 (model-inferred)"))}
            dump(out / f"build_{br}_spec.json", spec)
            log["steps"].append([f"build_{br}", blender(H6.STAGE, "build_variant", out / f"build_{br}.log", blend=base, spec=out / f"build_{br}_spec.json",
                                                         report=out / f"build_{br}_report.json")])
            blends[br] = out / f"{br}_S1.blend"
        spec = {"out_blend": str(out / f"{x}rev_S1.blend"), "interior_action_frames": list(U.INTERIOR), "values_wxyz": vA, "branch": f"W139 {LBL}{x}rev",
                "action_name": f"AnimCV_EXP W139 {LBL}{x}_reverted S1", "expected_blend_sha256": M.sha256_file(blends[x]), "note": "reversibility control"}
        dump(out / f"build_{x}rev_spec.json", spec)
        log["steps"].append([f"build_{x}rev", blender(H6.STAGE, "build_variant", out / f"build_{x}rev.log", blend=blends[x], spec=out / f"build_{x}rev_spec.json",
                                                       report=out / f"build_{x}rev_report.json")])
        blends[f"{x}rev"] = out / f"{x}rev_S1.blend"
        for br in (x, f"{x}pin", f"{x}rev"):
            log["steps"].append([f"readback_{br}", readback(blends[br], out / f"readback_{br}.json", out / f"readback_{br}.log")])
            rb[br] = load(out / f"readback_{br}.json")
        log["steps"].append([f"dump_{x}", bdump(blends[x], out / f"dump_{x}.json", out / f"dump_{x}.log")])
        dumps[x] = load(out / f"dump_{x}.json")
    # ================= mechanical gates =================
    pin_src = load(H6.FROZEN["W126V_proposal"][0])
    tgt = np.asarray(pin_src["constraint"]["target"])
    fp = _frames(rb["Pin"])
    rp_ = {f: float(np.linalg.norm(np.asarray(fp[f]["world"]["hand_l"]["head"]) - tgt)) for f in U.PIN_ACTION}
    mech = {}
    i0, i1_ = U.INTERIOR
    for x in CANDIDATES:
        r = rb[x]
        pres = M.preservation(r, rb["A"], U.INTERIOR)
        ck = M.changed_keys(dumps[x]["stored"], dumps["A"]["stored"])
        census = r["fcurve_census"]
        fX, fA = _frames(r), _frames(rb["A"])
        edge = all(fX[f]["world"] == fA[f]["world"] and fX[f]["matrix_world"] == fA[f]["matrix_world"] for f in (U.CONTEXT_LAST, U.TARGET_FIRST))
        L1 = [float(np.linalg.norm(np.subtract(fX[f]["world"]["lowerarm_l"]["head"], fX[f]["world"]["upperarm_l"]["head"]))) for f in range(i0 - 1, i1_ + 2)]
        L2 = [float(np.linalg.norm(np.subtract(fX[f]["world"]["hand_l"]["head"], fX[f]["world"]["lowerarm_l"]["head"]))) for f in range(i0 - 1, i1_ + 2)]
        unit = all(abs(np.linalg.norm(q) - 1.0) < 1e-9 and np.isfinite(q).all() for ch in vals[x].values() for q in map(np.asarray, ch.values()))
        fc = _frames(rb[f"{x}pin"])
        rc_ = {f: float(np.linalg.norm(np.asarray(fc[f]["world"]["hand_l"]["head"]) - tgt)) for f in U.PIN_ACTION}
        cp = M.preservation(rb[f"{x}pin"], rb["Pin"], U.INTERIOR)
        m6 = bool(all(fc[f]["world"] == fp[f]["world"] for f in U.PIN_ACTION) and all(rc_[f] == rp_[f] for f in U.PIN_ACTION) and max(rc_.values()) <= 1e-6
                  and not cp["nonarm_changed"] and not cp["outside_changed"] and not cp["sub_outside_changed"])
        m7 = all(p_["world"] == q_["world"] and p_["matrix_world"] == q_["matrix_world"] for p_, q_ in zip(rb[f"{x}rev"]["frames"], rb["A"]["frames"])) and \
            all(p_["world"] == q_["world"] for p_, q_ in zip(rb[f"{x}rev"]["subframes"], rb["A"]["subframes"]))
        pk = x[-1]
        hyb = worlds[pk][f"{x}_hybrid_fk"]
        m12 = max(float(np.abs(hyb[i, names.index(b)] - np.asarray(fX[U.index_to_action(i)]["world"][b]["head"])).max())
                  for i in range(U.T_MODEL) for b in ("upperarm_l", "lowerarm_l", "hand_l"))
        mech[x] = {"M2": bool(len(r["actions_in_file"]) == 1 and census["n_fcurves"] == 32 and set(census["interpolation_census_curves"]) == {"CONSTANT"}
                              and [census["key_frame_min"], census["key_frame_max"]] == [1.0, 240.0] and r["scene"]["effective_fps"] == 30.0),
                   "M3": bool(set(c[0] for c in ck) <= set(U.EDIT_CHANNELS) and all(i0 <= c[2] <= i1_ for c in ck) and edge and not pres["nonarm_changed"]
                              and not pres["outside_changed"] and not pres["sub_outside_changed"]),
                   "n_changed_keys": len(ck), "M5": bool(np.ptp(L1) <= 1e-4 and np.ptp(L2) <= 1e-4 and unit), "upper_len_range_m": float(np.ptp(L1)),
                   "lower_len_range_m": float(np.ptp(L2)), "M6": m6, "M6_residual_max_m": max(rc_.values()), "M7": bool(m7),
                   "M8_bridge_roundtrip_m": pf["bridge"][pk]["blender_roundtrip_max_world_diff_m"], "M8": bool(pf["bridge"][pk]["blender_roundtrip_max_world_diff_m"] <= 1e-5),
                   "M12_max_m": m12, "M12": bool(m12 <= 1e-4), "blend_sha256": r["blend_sha256"], "action_name": r["action_name"]}
    raw = {}
    for x in CANDIDATES:
        W_ = worlds[x[-1]]
        b_ = pf["bridge"][x[-1]]
        att = T.attribution(W_[f"{x}_fk"], W_["proxy_fk"], W_[f"{x}_hybrid_fk"], names)
        body = T.body_report(W_[f"{x}_fk"], W_["proxy_fk"], names)
        ru = T.raw_unusable(att, body)
        drift = np.linalg.norm(W_[f"{x}_fk"][:, 0] - W_["proxy_fk"][:, 0], axis=1) * 1e3
        rr_ = np.load(out / f"recover_{x[-1]}" / "recovered.npz")
        raw[x] = {"attribution": {k: v for k, v in att.items() if k != "raw_vs_projected"},
                  "raw_vs_projected": {k: {kk: vv for kk, vv in v.items() if kk != "per_frame_mm"} for k, v in att["raw_vs_projected"].items()},
                  "body": body, "raw_rule": ru, "root_drift_max_mm": float(drift.max()),
                  "ric_vs_fk_interior_max_mm": float(np.linalg.norm(H7.map_to_world(rr_[f"{x}_ric"], wms[x[-1]])[fi] - W_[f"{x}_fk"][fi], axis=-1).max() * 1e3),
                  "proxy": b_["asset"]}
    hinge = M.hinge_reference(rb["A"])
    rbq = {"A": rb["A"], "B": rb["B"], "D": rb["C"], "U": rb["U"], "T": rb["T"], **{x: rb[x] for x in CANDIDATES}}
    dq = {"A": dumps["A"], "B": dumps["B"], "D": dumps["C"], "U": dumps["U"], "T": dumps["T"], **{x: dumps[x] for x in CANDIDATES}}
    q = {k: quality_of(rbq[k], dq[k], rb["A"], rb["B"], hinge) for k in ORDER}
    for k in q:
        q[k]["usability_floor"] = TS.usability_floor(q[k])
        q[k]["non_inferior_to_B"] = TS.non_inferior(q[k], q["B"])
        q[k]["acceptable"] = TS.acceptable(q[k], q[k]["intent"], q["B"])
    sup = TS.superiority({"acceptable": q["S2C"]["acceptable"], "fidelity_dev_A_mean_mm": q["S2C"]["fidelity_dev_A_mean_mm"]},
                         {k: {"acceptable": q[k]["acceptable"], "fidelity_dev_A_mean_mm": q[k]["fidelity_dev_A_mean_mm"]} for k in ("S2A", "S2B", "D", "B")})
    s2d = H6.source_2d({k: rbq[k] for k in ORDER}, rb["A"])
    res.update({"gates": gates, "recorded_state_checks": st_chk, "K0": k0, "noise_floor": {"decoded_mm": noise_mm, "feature_max_abs_G": noise_feat},
                "I1": i1, "decomposition": dec, "information_path": info, "mechanical": mech, "raw": raw, "quality": q, "superiority": sup,
                "source_2d": {k: s2d[k] for k in ("valid", "fit_rms_px")},
                "blends": {k: str(v) for k, v in blends.items()}, "blend_sha256": {k: M.sha256_file(v) for k, v in blends.items()}})
    dump(out / "run_result.json", res)                                                # primary result BEFORE the native control
    # ================= model-native capability control (isolated) =================
    nat_steps = []
    try:
        nat_steps.append(["NREF", _infer(out, "NREF", pf, inputs, stub)])
        nat_steps.append(["native_prep", _um_ok(um([INFER9, "native_prep", "--out-dir", out / "native_prep", "--asset-dir", pf["bridge"]["B"]["asset_dir"],
                                                    "--name", pf["bridge"]["B"]["asset"], "--ref", load(out / "NREF" / "infer_record.json")["sample_npy"]],
                                                   out / "native_prep.log"), "native_prep")])
        inputs = {"nref_star": out / "native_prep" / "nref_star.npy", "nB_init": out / "native_prep" / "nB_init.npy"}
        for nm in ("N0", "NL", "NIB"):
            nat_steps.append([nm, _infer(out, nm, pf, inputs, stub)])
        nrec = {nm: load(out / nm / "infer_record.json") for nm in NATIVE}
        for nm in NATIVE:
            res["runs"][nm] = run_summary(nm, nrec[nm])
        nio = {nm: np.load(out / nm / "raw_model_io.npz") for nm in NATIVE}
        ntr = {nm: _traj(out, nm) for nm in NATIVE}
        ng = {"sampler_calls": all(nrec[n]["calls_generate_samples"] == 1 and nrec[n]["scheduled_sampler_calls"] == 1
                                   and nrec[n]["network_forward_calls"] == 2 * TS.N_STEPS for n in NATIVE),
              "eps_completions_equal_primary": all(nrec[n]["eps_sha256"] == rec["S2B"]["eps_sha256"] for n in ("N0", "NL", "NIB")),
              "native_eps_corr": float(np.corrcoef(ntr["NREF"]["eps"].ravel(), tr["S2B"]["eps"].ravel())[0, 1])}
        ng["native_eps_independent"] = bool(nrec["NREF"]["eps_sha256"] != rec["S2B"]["eps_sha256"] and abs(ng["native_eps_corr"]) < TS.EPS_CORR_MAX)
        nst = {}
        for nm in NATIVE:
            kmn = np.broadcast_to(nio[nm]["keep_mask"], tuple(nio[nm]["motion_shape"]))[0, :U.N_JOINTS]
            c_ = TS.check_recorded_states(ntr[nm]["states"], ntr[nm]["eps"], nio[nm]["x1_known_used_normalised"][:, :U.N_JOINTS], kmn,
                                          nio[nm]["sched_mask"][:U.N_JOINTS], int(nrec[nm]["k_s"]), ntr[nm]["t"])
            nst[nm] = bool(c_["ok"] and np.array_equal(ntr[nm]["states"][-1], nio[nm]["raw_output_normalised"][:, :U.N_JOINTS]))
        ng["recorded_states"] = nst
        nat = native_report(out, pf, names, par, nrec, wms)
        nat["gates"] = ng
        valid = bool(ng["native_eps_independent"] and ng["sampler_calls"] and all(nst.values()) and ng["eps_completions_equal_primary"])
        nat["status"] = ("INVALID" if not valid else ("CONTROL_UNINFORMATIVE" if not nat["admissible"]["ok"] else "INFORMATIVE"))
    except (RuntimeError, OSError, KeyError, ValueError, U.UniStop) as e:
        nat = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}
    nat["steps"] = nat_steps
    log["steps"] += nat_steps
    res["native"] = nat
    # ================= budget / integrity after =================
    procs = sum(st[1].get("processes", 1) for st in log["steps"] if st[0] in RUNS)
    vram = max(int(res["runs"][n].get("peak_vram_bytes_total") or 0) for n in res["runs"])
    gpu_s = sum(float(res["runs"][n].get("total_seconds") or 0) for n in res["runs"])
    res["budget"] = {"model_processes": procs, "peak_vram_gb": vram / 1e9, "model_wall_min": gpu_s / 60.0,
                     "ok": bool(procs <= 12 and vram / 1e9 <= 2.0 and gpu_s / 60.0 <= 60.0)}
    after = frozen_hashes()
    res["frozen_after_ok"] = all(v["ok"] for v in after.values())
    res["frozen_after_bad"] = [k for k, v in after.items() if not v["ok"]]
    dump(out / "run_result.json", res)
    # ================= videos =================
    old_names = dict(H6.NAMES)
    try:
        H6.NAMES.update({k: LBL + v for k, v in NAMES.items()})
        vid = H6.render_and_compose(out, pdir, {k: blends[k] for k in ORDER}, log, order=ORDER)
        newp = out / f"w139_{'STUB_' if stub else ''}named_S1_30fps.mp4"
        Path(vid["video"]).rename(newp)
        vid["video"], vid["sha256"] = str(newp), M.sha256_file(newp)
        import os
        secret = os.urandom(32).hex()
        key = TS.blind_key(secret, ["S2A", "S2B", "S2C", "D"])
        bdir = out / "blind"
        (bdir / "inputs").mkdir(parents=True)
        for L, k in key.items():
            shutil.copyfile(blends[k], bdir / "inputs" / f"{L}.blend")
        H6.NAMES.update({L: LBL + L for L in key})
        blog = {"steps": []}
        bvid = H6.render_and_compose(bdir, pdir, {L: bdir / "inputs" / f"{L}.blend" for L in key}, blog, order=tuple(sorted(key)))
        for L in key:
            (bdir / "inputs" / f"{L}.blend").unlink()
        sealed = out / "SEALED_blind_key"
        sealed.mkdir()
        dump(sealed / "blind_key.json", {"key": key, "secret": secret, "note": "do not open before visual_review JSON exists"})
        dump(out / "render_log_blind_SEALED.json", blog)
        for pth in bdir.glob("render_*.log"):
            shutil.move(str(pth), str(sealed / pth.name))
        loop = loop_video(out, vid["video"], stub)
        res["video"] = {"named": vid, "blind": {k: v for k, v in bvid.items() if k in ("video", "frames_written", "frames_decoded", "fps", "native_rate")},
                        "loop_interior": loop, "blind_key_sha256": TS.sha_json(key),
                        "blinding": "nominal: letters from an os.urandom secret sealed in SEALED_blind_key/; the reviewer is the agent with filesystem access"}
    finally:
        H6.NAMES.clear()
        H6.NAMES.update(old_names)
    log["end"] = now()
    dump(out / "run_log.json", log)
    dump(out / "run_result_final.json", res)
    print(json.dumps({"STUB": stub, "gates": {k: v for k, v in gates.items() if k != "x1_g_only"}, "x1_g_only_ok": gates["x1_g_only"]["ok"], "K0": k0,
                      "noise_floor": res["noise_floor"], "I1_ok": i1["ok"], "tokens": i1["tokens"],
                      "decomposition_C": {k: v for k, v in dec["C"].items() if k not in ("discarded_steps", "correction_attribution")},
                      "correction_attribution": {k: v for k, v in dec["C"]["correction_attribution"].items() if not k.startswith("per_frame")},
                      "decomposition_A": {k: dec["A"][k] for k in ("beta", "gamma_first_free", "network_over_carry", "cos_network_dx1", "decoded_network_term_mm")},
                      "info": info, "mechanical": {k: {kk: v[kk] for kk in ("M2", "M3", "M5", "M6", "M7", "M8", "M12")} for k, v in mech.items()},
                      "max_steps": {k: q[k]["max_steps"] for k in q}, "intent": {k: q[k]["intent"] for k in q},
                      "fidelity": {k: q[k]["fidelity_dev_A_mean_mm"] for k in q}, "acceptable": {k: q[k]["acceptable"] for k in q}, "superiority": sup,
                      "raw_rule": {k: v["raw_rule"] for k, v in raw.items()}, "native_status": nat.get("status"), "native": nat.get("summary"),
                      "budget": res["budget"], "frozen_after_bad": res["frozen_after_bad"],
                      "video": {"named_native_rate": res["video"]["named"].get("native_rate"), "loop": res["video"]["loop_interior"]}}, indent=1, default=str))
    return 0


def loop_video(out, src, stub=False):
    import cv2
    cap = cv2.VideoCapture(str(src))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
    sel = frames[WIN[0] - 1:WIN[1]]
    p = out / f"w139_{'STUB_' if stub else ''}interior_loop_164_207_x3_30fps.mp4"
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (sel[0].shape[1], sel[0].shape[0]))
    for _ in range(3):
        for im in sel:
            wr.write(im)
    wr.release()
    cp = cv2.VideoCapture(str(p))
    n = 0
    while cp.read()[0]:
        n += 1
    fps = cp.get(cv2.CAP_PROP_FPS)
    cp.release()
    return {"path": str(p), "frames": n, "fps": fps, "native_rate": bool(n == 3 * len(sel) and abs(fps - 30.0) < 1e-6), "sha256": M.sha256_file(p)}


def native_report(out, pf, names, par, rec, wms):
    """Model-native capability control metrics (model space -> world scale with the C map scale, as declared)."""
    np_rep = load(out / "native_prep" / "native_prep.json")
    z = np.load(out / "native_prep" / "native_prep.npz")
    s = float(wms["C"]["s"])
    nstar = z["nref_star"]
    fk_star = z["fk_star"]
    fi = list(range(16, 44))
    neck, pel = names.index("neck_02"), names.index("pelvis")
    hgt = fk_star[:, neck, 1] - fk_star[:, pel, 1]
    wr = names.index("hand_l")
    path_mm = float(np.sum(np.linalg.norm(np.diff(fk_star[15:45, wr], axis=0), axis=1)) * s * 1e3)
    adm = {"ric_vs_fk": np_rep["nref_star_ric_vs_fk_max"], "no_fold_min_ratio": float(hgt.min() / hgt[0]) if hgt[0] > 0 else None,
           "hidden_wrist_path_mm": path_mm, "poison_equal": np_rep["poison_equal"]}
    adm["ok"] = bool(adm["ric_vs_fk"] <= 1e-4 and adm["no_fold_min_ratio"] is not None and adm["no_fold_min_ratio"] >= 0.5 and path_mm >= 100.0
                     and np_rep["poison_equal"] and np_rep["poisoned_finite_on_G"])
    samples = {k: rec[k]["sample_npy"] for k in ("N0", "NL", "NIB")}
    samples["NBinit"] = str(out / "native_prep" / "nB_init.npy")
    samples["NREF_raw"] = rec["NREF"]["sample_npy"]
    np.save(out / "native_prep_nref_star_f32.npy", nstar.astype(np.float32))
    _um_ok(um([INFER9, "recover", "--out-dir", out / "recover_native", "--asset-dir", pf["bridge"]["B"]["asset_dir"], "--name", pf["bridge"]["B"]["asset"],
               "--proxy", out / "native_prep_nref_star_f32.npy", "--samples", ",".join(f"{k}={v}" for k, v in samples.items())], out / "recover_native.log"),
           "recover_native")
    rr = np.load(out / "recover_native" / "recovered.npz")
    ref = rr["proxy_fk"]
    tq = np.load(Path(pf["bridge"]["B"]["asset_dir"]) / "cond.npy", allow_pickle=True).item()[pf["bridge"]["B"]["asset"]]["tpos_local_rotations"]
    dref = U.decode_basis(nstar, names, par, tq)
    rep = {}
    for k in ("N0", "NL", "NIB", "NBinit"):
        fk = rr[f"{k}_fk"]
        e = TS.arm_errors(fk * s, ref * s, names, fi)
        f = np.load(samples[k])
        try:
            dk = U.decode_basis(f, names, par, tq)
            ang = {b: float(np.max(TS.qangle_deg(dk[b][fi], dref[b][fi]))) for b in dk}
        except U.UniStop as ex:
            ang = {"STOP": ex.code}
        steps = {}
        for nm_, b in (("wrist", "hand_l"), ("elbow", "lowerarm_l")):
            j = names.index(b)
            seq_m = fk[15:45, j] * s
            st = np.linalg.norm(np.diff(seq_m, axis=0), axis=1) * 1e3
            rs = np.linalg.norm(np.diff(ref[15:45, j] * s, axis=0), axis=1) * 1e3
            steps[nm_] = {"entry_mm": float(st[0]), "exit_mm": float(st[-1]), "peak_mm": float(st.max()), "ref_entry_mm": float(rs[0]),
                          "ref_exit_mm": float(rs[-1]), "ref_peak_mm": float(rs.max()), "flicks": T.flicks(seq_m), "ref_flicks": T.flicks(ref[15:45, j] * s)}
        rep[k] = {"fk_error_mm": e, "max_angle_error_deg": ang, "steps": steps, "ric_vs_fk_max_model_units": float(np.abs(rr[f"{k}_ric"] - fk).max())}
    rep["NL_refinement_gain_wrist_mean_mm"] = rep["NBinit"]["fk_error_mm"]["wrist"]["mean"] - rep["NL"]["fk_error_mm"]["wrist"]["mean"]
    rep["disclosure"] = ("NIB keeps the whole window frame 15 incl. its velocity channels, which encode N_ref*'s hidden frame-16 positions "
                         "(the W137 leak): NIB is an official-mode reference with a one-frame velocity leak, not a leak-free completion")
    rep["admissible"] = adm
    rep["summary"] = {"admissible": adm, "wrist_mean_err_mm": {k: rep[k]["fk_error_mm"]["wrist"]["mean"] for k in ("N0", "NL", "NIB", "NBinit")},
                      "elbow_mean_err_mm": {k: rep[k]["fk_error_mm"]["elbow"]["mean"] for k in ("N0", "NL", "NIB", "NBinit")},
                      "NL_refinement_gain_wrist_mean_mm": rep["NL_refinement_gain_wrist_mean_mm"], "prep": np_rep}
    return rep


# ------------------------------------------------------------------------------------------------ assess ---
def stage_assess(a):
    d = check_declaration(a)
    out = new_dir(a.out_dir)
    pf, ctl = load(a.preflight), load(a.controls)
    run_dir = Path(a.run_dir)
    rr = load(run_dir / "run_result_final.json") if (run_dir / "run_result_final.json").is_file() else load(run_dir / "run_result.json")
    if rr.get("STUB_DRY_RUN") or any("STUB_MODEL" in load(p) for p in run_dir.glob("*/infer_record.json")):
        raise SystemExit("refusing to assess a --dev-stub dry run")
    if rr.get("declaration_sha256") != a.declaration_sha or M.sha256_file(a.preflight) != d["preflight_sha256"] or ctl.get("declaration_sha256") != a.declaration_sha:
        raise SystemExit("STOP INTEGRITY_FAILED: run/preflight/controls not bound to this declaration")
    vr = load(a.visual_review)
    human = False                                                                      # declared: no human native-rate review in this batch
    if pf["status"] != "OK":
        rep = {"judgment": "BLOCKED", "why": f"preflight {pf.get('stop')}"}
    elif "STOP" in rr:
        rep = {"judgment": "BLOCKED", "why": rr["STOP"]}
    else:
        g = rr["gates"]
        mech_ok = all(all(v[k] for k in ("M2", "M3", "M5", "M6", "M7", "M8", "M12")) for v in rr["mechanical"].values())
        prereq = {"preflight": pf["status"] == "OK", "controls": bool(ctl["all_fail_closed"]), "sampler_calls": bool(g["sampler_calls"]),
                  "eps_seed10_identical": bool(g["eps_seed10_identical"]), "recorded_states": all(g["recorded_states"].values()),
                  "x1_used_equals_official": bool(g["x1_used_equals_official"]), "bridge_leak_free": bool(g["x1_g_only"]["ok"]),
                  "x1_equals_offline": bool(g["x1_equals_offline"]), "mechanical": mech_ok, "integrity_after": bool(rr["frozen_after_ok"]),
                  "video_native_rate": bool((rr.get("video") or {}).get("named", {}).get("native_rate") and (rr.get("video") or {}).get("loop_interior", {}).get("native_rate"))}
        controls_ok = bool(rr["K0"]["primary_ok"])
        jd = TS.judgment(all(prereq.values()), controls_ok, rr["information_path"], rr["superiority"]["all"], human)
        rep = {**jd, "prerequisites": prereq, "K0_primary_ok": controls_ok,
               "K0_secondary": {k: rr["K0"].get(k) for k in ("vs_W136_U_max_abs", "vs_W136_U_decoded_mm", "x1_vs_W136_max_abs", "secondary_warning")},
               "information_path": rr["information_path"], "superiority": rr["superiority"],
               "acceptable": {k: v["acceptable"] for k, v in rr["quality"].items()}, "native_status": rr["native"].get("status"),
               "agent_review": vr.get("answers"), "agent_review_reported_only": True, "human_native_rate_review": human, "budget": rr.get("budget")}
    rep.update({"stage": "assess", "time": now(), "rule": JUDGMENT_RULE, "visual_review_sha256": M.sha256_file(a.visual_review),
                "run_result_sha256": M.sha256_file(run_dir / "run_result_final.json") if (run_dir / "run_result_final.json").is_file() else None,
                "note": "agent review = the agent's own nominally blinded still-frame review, reported only; not human/animator validation; perceptual acceptance PENDING"})
    sha = dump(out / "assessment.json", rep)
    print(json.dumps({**{k: v for k, v in rep.items() if k != "rule"}, "sha": sha}, indent=1, default=str))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("preflight", "declare", "controls", "run", "assess"))
    ap.add_argument("--out-dir", required=True)
    for k in ("preflight", "declaration", "declaration-sha", "controls", "run-dir", "visual-review"):
        ap.add_argument("--" + k)
    ap.add_argument("--dev-stub", action="store_true", help="DEV DRY RUN ONLY: weight-free stub network; never under the results root")
    a = ap.parse_args()
    return {"preflight": stage_preflight, "declare": stage_declare, "controls": stage_controls, "run": stage_run, "assess": stage_assess}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
