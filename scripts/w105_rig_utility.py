#!/usr/bin/env python3
"""W105: does W104's source-only raw-joint restoration (R11) improve actual BaseRig animation? (evidence-only diagnostic)

Stages (each refuses to overwrite its output directory):
  verify   frozen W102/W103/W104 identities, rig asset / rig-script identity with the W102 snapshot, real-clip provenance,
           M input and R11 input reconstruction from retained raw M1 landmarks (no 3D outcome, no GT, no rig output)
  declare  hashed contract: populations, exclusions, rig/solver/render/timing configuration, metric definitions, gates,
           CASE mapping and the deterministic render-window selection. Reads no new rig output.
  build    GT-FREE: frozen M_Main and R11 S0 sources (W104 R11 reproduced byte-exactly; the same fixed rule applied to the
           remaining tracked rows only to keep identity caveats visible), W102 rig exports for both conditions, the
           Blender case file for the declared windows, and the GT-free historical real-clip configuration
  analyze  evaluation-only GT: paired rig metrics, temporal events, harm attribution, gates, CASE; GT-through-rig
           reference cases written to a SEPARATE file
  compose  review videos / contact sheets from the Blender renders (unavailable samples blanked, never held)
  manifest evidence manifest over every W105 output

Only the 3D source pose differs between the two rig conditions. The W102 contract (frozen HMR pelvis/thorax parent,
frozen W87 J1X left arm, right arm at rest, S0 legs, W95 two-bone IK, zero-twist locals) is reused unchanged; no solver,
renderer, camera, threshold, mask, crop or S0 setting is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

from w103_observability import (DIRECT, JOINTS, SEGMENTS, SMPL_OF, W102_MANIFEST_SHA, S0_CHECKPOINT_SHA, BANK_DIGEST,  # noqa: E402
                                HMR_SHA, angle_deg, distribution, dump, sha, video_bootstrap)

# ------------------------------------------------------------------ frozen identities
W104_MANIFEST_SHA = "a6fa3874b7959b74c8859a5ffbf983728daaed67519e1a9598be899a10c91aeb"
W104_DECLARATION_SHA = "6a64c8d1ffc57f928fdf2fbaaf40dffd5b848a927b662e90ff584257bfbae487"
W103_MANIFEST_SHA = "3aad853cc52bc38efb85cd59833c031410df2788236fb6fd672ad1236de14f46"
W104_WORKLOG = "docs/104_WORKLOG_MISSING_JOINT_CAUSALITY_AND_RECOVERABILITY.md"
CLIP = "examples/production_demo/source.mp4"
CLIP_SHA = "0dda8a5f16c67992330a1a8534cf0bfefeb5e0f2aec438577a9e1dbace773275"
W102_USED = ["results/evaluation/rows.json", "data/hmr_pred_full.npz", "data/w102_j1x_predictions_3dpw.npz",
             "data/w102_j1x_report.json", "results/handoff/observations.npz", "results/handoff/observations_side.npz",
             "results/historical_video/observations_predictions.npz", "results/historical_video/report.json",
             "results/video_inspection/inspection.json", "results/rig/cases.json", "results/rig/rig_report.json",
             "results/rig_renders_complete/blender_report.json", "data/population.json"]
RIG_ASSETS = {"rest": "examples/e2e_demo/baserig_blender_rest.json", "calibration": "examples/e2e_demo/framepose_fk_calibration.json",
              "chains": "examples/e2e_demo/framepose_ik_chains.json", "rig": "examples/skinned_mannequin/BaseRig_Skinned.fbx"}
RIG_SCRIPTS = ["scripts/w102_rig_review.py", "scripts/render_runtime_heading_rig_blender.py",
               "scripts/run_pelvis_frame_leg_articulation.py", "scripts/run_torso_frame_rig_replay.py",
               "scripts/run_external_orientation_handoff.py", "src/retarget/framepose_two_bone_ik.py",
               "src/retarget/framepose_fk.py", "src/retarget/framepose_target_rest.py", "src/motion/torso_frame.py"]
BLENDER_ARGS = ["-b", "-t", "2", "--python-exit-code", "1"]
RENDER_SETTINGS = {"show_mesh": True, "size": 480, "ortho_scale": 3.2, "equivalent_rig_tolerance": 1e-4,
                   "renderer": "historical W91 Workbench replay (scripts/render_runtime_heading_rig_blender.py) via the W102 wrapper",
                   "interpolation": "CONSTANT", "scene_fps": 30}

WINDOW = 24                 # samples per declared 3DPW render window (10 Hz bank cadence: 2.4 s)
SEED = 1337
LEG_TOKENS = (1, 2, 3, 4, 5, 6)
ARM_TOKENS = (11, 12, 13, 14, 15, 16)
LEFT_ARM_TOKENS = (11, 12, 13)
LEG_BONES = (("l", "thigh"), ("l", "shin"), ("r", "thigh"), ("r", "shin"))
LEG_IDX = {"l": (1, 2, 3), "r": (4, 5, 6)}
# thresholds: every value is inherited from an existing W95/W97/W102/W104 contract (see DECLARATION_TEXT)
DIR_FAIL_DEG = 45.0         # W102/W104 limb failure: segment direction error > 45 deg
FOLD_DEG = 45.0             # W102 45 deg event scale applied to rig knee flexion excess over the GT-through-rig knee
STEP_DEG, GT_QUIET_DEG = 45.0, 15.0      # W102 direction-step event
THIGH_STEP_DEG, GT_THIGH_QUIET = 90.0, 45.0  # W95/W102 severe local-thigh step
HARM_RATIO = 0.25           # W104 HARM ratio
MATERIAL_NET_SHARE = 0.10
TRANSLATION_BAND_DEG = 5.0  # descriptive only
PCK_RESTORED = 0.10         # W101/W104 restored 2D accuracy scale
REGIME = {"M": "real_animcv_observation (M_Main, frozen W102)",
          "R11": "real_animcv_observation source-only restoration (evaluation-only W104 rule; not a production policy)",
          "GT_RIG": "evaluation-only GT 3D mapped through the same fixed rig (reference, never a candidate)"}


# ------------------------------------------------------------------ small helpers
def lf_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v/np.where(n > 0, n, np.nan)


def ang(a, b):
    return float(angle_deg(np.asarray(a, float), np.asarray(b, float)))


def video_of_sid(sid):
    return str(sid).split(":")[1]


def leg_restored(new):
    return np.asarray(new)[..., list(LEG_TOKENS)].any(-1)


def arm_only_restored(new):
    new = np.asarray(new)
    return new[..., list(ARM_TOKENS)].any(-1) & ~leg_restored(new)


def require_declaration(args):
    d = Path(args.declaration)
    if sha(d) != args.declaration_sha or args.declaration_sha is None:
        raise RuntimeError("STOP: declaration hash mismatch")
    return json.loads(d.read_text())


def windows_by_sid(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["sid"]].append(i)
    for sid in groups:
        groups[sid].sort(key=lambda i: rows[i]["frame"])
    return dict(sorted(groups.items()))


def best_window(groups, score, used, length=WINDOW, admissible=None):
    """Deterministic: maximal score, ties by (sequence id, start frame). Windows are disjoint from ``used``."""
    best = None
    for sid, idx in groups.items():
        for s in range(0, len(idx)-length+1):
            w = idx[s:s+length]
            if used & set(w) or (admissible is not None and not admissible(w)):
                continue
            key = (-score(w), sid, s)
            if best is None or key < best[0]:
                best = (key, w)
    return None if best is None else best[1]


def centred_window(groups, rows, row, used, length=WINDOW):
    sid = rows[row]["sid"]; idx = groups[sid]; k = idx.index(row)
    s = max(0, min(k-length//2, len(idx)-length))
    w = idx[s:s+length]
    return None if used & set(w) else w


# ------------------------------------------------------------------ context loaders
def load_all(args, need_raw=True):
    from w103_observability import load_context, load_predictions
    from w104_missing_joint import raw_landmarks
    bank, obs, side, positions = load_context(args.w102, args.w101_data)
    rows = json.loads((Path(args.w102)/"results/evaluation/rows.json").read_text())
    pred, _ = load_predictions(args.w102, bank, positions)
    ctx = {"bank": bank, "obs": obs, "side": side, "positions": positions, "rows": rows, "pred_M": pred["M"].astype(np.float32)}
    if need_raw:
        ctx["lm"], ctx["lm_audit"] = raw_landmarks(args.w102, bank, positions, obs)
    return ctx


def rig_context():
    from w102_rig_review import context
    return context(ROOT/RIG_ASSETS["rest"], ROOT/RIG_ASSETS["calibration"], ROOT/RIG_ASSETS["chains"])


def hmr_frames(w102, positions):
    """Frozen GT-free W97 HMR2 pelvis/thorax per population row (None where W97 marked the frame invalid)."""
    from run_external_orientation_handoff import handoff_frames
    path = Path(w102)/"data/hmr_pred_full.npz"
    if sha(path) != HMR_SHA:
        raise RuntimeError("STOP: frozen HMR2 identity failed")
    hp = np.load(path, allow_pickle=False)
    h = {int(p): i for i, (pop, p) in enumerate(zip(hp["population"], hp["index"][0])) if str(pop).startswith("3dpw") and hp["valid"][i]}
    out = []
    for p in positions:
        if int(p) not in h:
            out.append((None, None)); continue
        Ro, Rt = handoff_frames(hp["global_orient"][h[int(p)]], hp["joints24"][h[int(p)]])[:2]
        out.append((Ro, Rt))
    return out


def load_j1x(w102, n_bank):
    from run_visual_depth_motion_attribution import W87_PREDICTIONS
    path = Path(w102)/"data/w102_j1x_predictions_3dpw.npz"
    if sha(path) != W87_PREDICTIONS["3dpw"]:
        raise RuntimeError("STOP: frozen J1X identity changed")
    j1 = np.load(path, allow_pickle=False)
    if any(j1[k].shape != (n_bank, 3) for k in ("left_arm_g1_elbow", "left_arm_g1_wrist")):
        raise RuntimeError("STOP: J1X bank row shape mismatch")
    return j1


# ------------------------------------------------------------------ GT-free rig measurement (W95 evaluate, no references)
def rig_measure(j, Ro, Rt, ctx):
    """GT-free per-row rig quantities for the W102 leg contract. None when the parent or a chain is unavailable.

    Uses exactly the W102 ``pose_export`` (for the exported locals) and W95 ``evaluate`` without references.
    """
    import run_pelvis_frame_leg_articulation as w95
    from w102_rig_review import pose_export
    if j is None or Ro is None or Rt is None:
        return None
    j = np.asarray(j, float); j = j-j[0]
    exp = pose_export(j, Ro, Rt, ctx)
    ev = w95.evaluate(j, Ro, Rt, ctx)
    if exp is None or ev is None:
        return None
    Am = ctx["Am"]
    out = {"export": exp, "viol": {k: bool(v) for k, v in ev["viol"].items()}, "legs": {}}
    for s, (h, k, a) in LEG_IDX.items():
        leg = ev["legs"][s]; p = leg["pose"]
        u, v = j[k]-j[h], j[a]-j[k]
        tgt = ctx["legs"][s]["target"]
        out["legs"][s] = {
            "thigh_w": leg["thigh_w"].tolist(), "shin_w": leg["shin_w"].tolist(),
            "ankle_rel_w": (Ro @ (Am.T @ (np.asarray(p.end)-np.asarray(p.root)))).tolist(),
            "knee": float(leg["knee"]), "hinge": None if leg["hinge"] is None else float(leg["hinge"]),
            "hip_flex": float(leg["hip_flex"]), "hip_abd": float(leg["hip_abd"]),
            "source_knee": ang(u, v),
            "s2r_thigh": ang(leg["thigh_w"], u), "s2r_shin": ang(leg["shin_w"], v),
            "source_ratio": float(np.linalg.norm(u)/max(np.linalg.norm(v), 1e-12)),
            "target_ratio": float(tgt.proximal_length/tgt.distal_length),
            "reach_fraction": float(p.reach_fraction), "clamp": bool(p.clamp), "side_status": str(p.side_status),
            "thigh_local": list(exp["bones"][f"thigh_{s}"]), "calf_local": list(exp["bones"][f"calf_{s}"])}
    return out


def s0_left_arm(j, Ro, Rt, ctx):
    """SECONDARY (not the W102 arm policy): S0 left arm through the existing left-arm two-bone chain, shared parent."""
    from w102_rig_review import arm_export
    if j is None or Ro is None or Rt is None:
        return None
    j = np.asarray(j, float)
    a = arm_export(j[12]-j[11], j[13]-j[11], Ro, Rt, ctx)
    if a is None:
        return None
    from run_torso_frame_rig_replay import arm_pose
    pose = arm_pose(j[12]-j[11], j[13]-j[11], Rt, ctx["alignment"], ctx["target"], ctx["rest"])
    Am = ctx["Am"]
    return {"export": a, "upper_w": unit(Rt @ (Am.T @ np.asarray(pose.upper))).tolist(),
            "fore_w": unit(Rt @ (Am.T @ np.asarray(pose.forearm))).tolist(), "elbow": float(pose.bend_angle_degrees())}


# ------------------------------------------------------------------ stage: verify
def cmd_verify(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    w102, w104 = Path(args.w102), Path(args.w104)
    rep = {"schema": "w105_verify_v1", "checks": {}}
    # W104 artifacts + published COMPLETE marker
    if sha(w104/"evidence_manifest.json") != W104_MANIFEST_SHA:
        raise RuntimeError("STOP: W104 manifest identity failed")
    m104 = json.loads((w104/"evidence_manifest.json").read_text())
    f104 = {rel: sha(w104/rel) == e["sha256"] for rel, e in m104["files"].items()}
    if not all(f104.values()) or m104["declaration_sha256"] != W104_DECLARATION_SHA:
        raise RuntimeError("STOP: a W104 artifact differs from its manifest")
    wl = (ROOT/W104_WORKLOG).read_text(encoding="utf-8").rstrip().splitlines()
    rep["checks"]["w104"] = {"manifest_sha256": W104_MANIFEST_SHA, "files": len(f104), "all_match": True,
                             "declaration_sha256": W104_DECLARATION_SHA,
                             "worklog_complete_marker": wl[-2:] == ["AUTOMATION_BATCH_STATUS: COMPLETE", "BATCH_NUMBER: 104"]}
    if not rep["checks"]["w104"]["worklog_complete_marker"]:
        raise RuntimeError("STOP: W104 COMPLETE marker missing")
    # W102 / W103 identities
    if sha(w102/"evidence_manifest.json") != W102_MANIFEST_SHA:
        raise RuntimeError("STOP: W102 manifest identity failed")
    m102 = json.loads((w102/"evidence_manifest.json").read_text())["files"]
    f102 = {rel: sha(w102/rel) == m102[rel]["sha256"] for rel in W102_USED}
    rep["checks"]["w102_used_files_match_manifest"] = {"files": f102, "all": all(f102.values())}
    if not all(f102.values()):
        raise RuntimeError("STOP: a W102 artifact differs from its manifest")
    rep["checks"]["w103_manifest"] = sha(Path(args.w103)/"evidence_manifest.json") == W103_MANIFEST_SHA
    if not rep["checks"]["w103_manifest"]:
        raise RuntimeError("STOP: W103 manifest identity failed")
    if sha(Path(args.w101_data)/"s0_checkpoint.pt") != S0_CHECKPOINT_SHA:
        raise RuntimeError("STOP: S0 checkpoint identity failed")
    rep["checks"]["s0_checkpoint_sha256"] = S0_CHECKPOINT_SHA
    # rig assets / rig code equal to the W102 Main source snapshot (LF-normalized; Windows checkout is CRLF)
    snap = w102/"code"
    assets = {k: {"path": p, "sha256": sha(ROOT/p), "lf_sha256": lf_sha(ROOT/p), "w102_snapshot_lf_sha256": lf_sha(snap/p)}
              for k, p in RIG_ASSETS.items()}
    code = {p: lf_sha(ROOT/p) == lf_sha(snap/p) for p in RIG_SCRIPTS}
    rep["checks"]["rig_assets"] = assets
    rep["checks"]["rig_code_equal_w102_snapshot"] = code
    if not all(a["lf_sha256"] == a["w102_snapshot_lf_sha256"] for a in assets.values()) or not all(code.values()):
        raise RuntimeError("STOP: rig asset or rig code differs from the W102 snapshot")
    w102_rig = json.loads((w102/"results/rig_renders_complete/blender_report.json").read_text())
    rep["checks"]["w102_rig_render_report"] = {"blender_version": w102_rig["blender_version"], "interpolation": w102_rig["interpolation"],
                                              "scene_fps": w102_rig["scene_fps"], "equivalent_rig_max_rest_difference": w102_rig["equivalent_rig_max_rest_difference"]}
    # population / rows / source identity
    ctx = load_all(args)
    rows, pos, obs, bank = ctx["rows"], ctx["positions"], ctx["obs"], ctx["bank"]
    rep["checks"]["bank_content_digest"] = bank.content_digest()
    if bank.content_digest() != BANK_DIGEST:
        raise RuntimeError("STOP: bank identity")
    align = all(r["p"] == int(p) and r["sample_id"] == str(s) for r, p, s in zip(rows, pos, obs["sample_id"]))
    times = all(bank.samples[int(p)].frame_index == r["frame"] and abs(bank.samples[int(p)].timestamp-r["timestamp"]) < 1e-12
                for r, p in zip(rows, pos))
    jm = np.array([r["J"]["M"] for r in rows]); pm = ctx["pred_M"].astype(float)
    dM = float(np.abs(jm-(pm-pm[:, :1])).max())  # rows.json stores root-relative joints
    rep["checks"]["rows"] = {"rows": len(rows), "row_order_equals_population": align, "frame_timestamp_equal_bank": times,
                             "rows_json_M_vs_stored_prediction_root_relative_max_abs_m": dM,
                             "identity_counts": dict(Counter(r["identity"] for r in rows)),
                             "split_counts": dict(Counter(r["split"] for r in rows))}
    if not (align and times and dM < 1e-6):
        raise RuntimeError("STOP: actor/frame/time alignment")
    rep["checks"]["raw_landmark_provenance"] = ctx["lm_audit"]
    # R11 input reconstruction (GT-free) against the stored W104 inputs
    rz = np.load(w104/"results/infer/restoration_inputs.npz")
    rec = build_r11_inputs(ctx)
    same = np.asarray(ctx["side"]["identity"]) == "same_person"
    w104rows = rz["rows"]
    rep["checks"]["r11_reconstruction"] = {
        "w104_rows": int(len(w104rows)),
        "rows_equal": bool(np.array_equal(w104rows, np.flatnonzero(same & rec["changed"]))),
        "new_tokens_equal": bool(np.array_equal(rz["new_tokens"], rec["new"][w104rows])),
        "R11_input_byte_equal": bool(rz["R11"].tobytes() == rec["geometry"][w104rows].tobytes()),
        "same_person_noop_rows": int((same & ~rec["changed"] & rec["present"]).sum()),
        "same_person_noop_input_equals_M_geometry": bool(np.array_equal(rec["geometry"][same & ~rec["changed"]],
                                                                         rec["geometry_M"][same & ~rec["changed"]])),
        "non_same_person_rows_rule_would_change": dict(Counter(np.asarray(ctx["side"]["identity"])[~same & rec["changed"]])),
        "rows_without_subject": int((~rec["present"]).sum())}
    r = rep["checks"]["r11_reconstruction"]
    if not (r["rows_equal"] and r["new_tokens_equal"] and r["R11_input_byte_equal"] and r["same_person_noop_input_equals_M_geometry"]):
        raise RuntimeError("STOP: frozen R11 input cannot be reconstructed")
    # real clip provenance
    rep["checks"]["real_clip"] = verify_clip(w102)
    dump(out/"verify_report.json", rep)
    print(json.dumps(rep, default=str)[:3000])


def build_r11_inputs(ctx):
    """GT-free: the fixed W104 R11 rule on every row with retained raw landmarks (identity labels are NOT consulted)."""
    from framepose.crops import crop_box, geometry_in_crop
    from w104_missing_joint import restore_row, restoration_inputs
    obs, bank, pos, lm = ctx["obs"], ctx["bank"], ctx["positions"], ctx["lm"]
    xm, vm = obs["input_2d_M"], obs["input_valid_M"]
    n = len(pos)
    geo = np.zeros((n, 17, 4), np.float32); geoM = np.zeros((n, 17, 4), np.float32)
    new = np.zeros((n, 17), bool); present = np.isfinite(lm[..., 0]).any(1)
    for i in range(n):
        size = bank.samples[int(pos[i])].image_size
        geoM[i] = geometry_in_crop(xm[i], vm[i], size, crop_box(xm[i], vm[i], size))
        geo[i] = geoM[i]
        if not present[i]:
            continue
        raw, have, nw = restore_row(lm[i], size, xm[i], vm[i])
        if nw.any():
            geo[i] = restoration_inputs(xm[i], vm[i], raw, have, size)[0]["R11"]; new[i] = nw
    return {"geometry": geo, "geometry_M": geoM, "new": new, "changed": new.any(1), "present": present}


def verify_clip(w102):
    from w102_video_diagnostic import decode
    insp = json.loads((Path(w102)/"results/video_inspection/inspection.json").read_text())
    seq, digests = decode(ROOT/CLIP)
    out = {"repo_clip_sha256": sha(ROOT/CLIP), "w102_historical_source_sha256": sha(Path(w102)/"data/historical_source.mp4"),
           "inspection_clip_sha256": insp["clip_sha256"], "frames": len(seq.frames), "fps": seq.fps, "size": [seq.width, seq.height],
           "decoded_digests_equal_inspection": digests == insp["decoded_BGR_uint8_sha256"],
           "timestamps_equal_inspection": [f.timestamp for f in seq.frames] == insp["timestamp"],
           "rights": "undocumented; private diagnostic only; not redistributed; not unseen",
           "readme_note": "examples/production_demo/README.md describes a 120-frame audit; the clip itself decodes to 199 frames "
                          "(W102 used all 199 decoded frames)"}
    if not (out["repo_clip_sha256"] == out["w102_historical_source_sha256"] == insp["clip_sha256"] == CLIP_SHA
            and out["decoded_digests_equal_inspection"] and out["timestamps_equal_inspection"] and out["frames"] == 199 and out["fps"] == 50.0):
        raise RuntimeError("STOP: real-clip provenance")
    return out


# ------------------------------------------------------------------ stage: declare
DECLARATION_TEXT = """
W105 declaration (written before any W105 rig outcome is computed or viewed).

Candidates (only the 3D source differs): M_Main = frozen W102 S0 predictions; R11 = W104 fixed source-only restoration
(every M-invalid token with a finite raw RTMPose coordinate restored with its raw coordinate+confidence, production
recrop). W104 rows use the stored W104 prediction, required byte-equal to a fresh forward. Rows outside the W104
same-person population receive the identical GT-free rule only to keep identity caveats visible; they never enter a gate.

Rig contract (W102, unchanged): W97 frozen HMR2 pelvis (Object) and thorax (spine_01) frames shared by both conditions;
W87 frozen J1X left arm shared; right arm at rest; S0 legs through W95 two-bone IK with zero-twist locals, the BaseRig
rest snapshot, FK calibration and chain file; skinned BaseRig equivalent bound at 1e-4; Blender 5.1 Workbench, 480 px,
orthographic scale 3.2, the historical video/side cameras, CONSTANT interpolation, keys at original frame indices.
Bones consuming S0 geometry: thigh_l, calf_l, thigh_r, calf_r only. Arm-only restorations reach the rig only through
S0's whole-body response on the legs. Right arm: no rig chain (exclusion "no rig chain").
SECONDARY (not the W102 arm policy, reported separately, never in a gate): S0 left arm through the existing left-arm
two-bone chain with the same HMR thorax.

Populations: full W102 denominator 10,483 actor-frames by identity and split; W104 same-person restoration rows (1,225;
1,223 PARTIAL with a restored limb token); no-op complete same-person rows (5,763). Rig-eligible = HMR parent and
thorax available and M, R11 and GT-through-rig chains non-degenerate. Exclusion reasons: no_HMR_parent,
HMR_thorax_degenerate, degenerate chain (per condition), GT camera invalid (sensitivity run only).
P_leg (primary) = rig-eligible PARTIAL rows with >= 1 restored hip/knee/ankle token. P_arm = arm-only PARTIAL.

Metrics (GT evaluation-only, computed after build): per leg bone the world direction of the rig bone vs the same bone of
GT mapped through the same rig with the same parent (rig_dir_ref, primary) and vs the raw GT segment (rig_dir_gt);
rig knee flexion abs error vs GT-through-rig; rig knee branch (both >= 20 deg, normals > 90 deg; W97); rig limb failure
= any bone of that leg rig_dir_ref > 45 deg or a wrong rig knee branch (W102/W104 scale); false fold = rig knee exceeds
GT-through-rig knee by > 45 deg; W95 leg ROM violations (hip flexion/extension/abduction/adduction, knee flexion max,
knee hinge deviation); source-to-rig discrepancy = angle between rig bone and its own source segment (W102), source
thigh/shin length ratio vs target ratio, reach fraction, clamp; ankle endpoint relative to hip vs GT-through-rig
(armature units). Source metrics: W103/W104 s0_metrics on the same rows.
Temporal: original same-person bank neighbours (rows.json next), both rig-eligible, at least one in the restoration
set; events = rig bone direction step > 45 deg while GT step < 15 deg (W102) and local thigh step > 90 deg while GT
thigh step < 45 deg (W95/W102). 10 Hz sparse cadence; no interpolation is evaluated. Mask-transition pairs = M leg
validity differs between the two rows.
Uncertainty: video-clustered bootstrap, 2,000 replicates, seed 1337 (W103 video_bootstrap).

Gates on P_leg (primary run; the crosscountry root-behind-camera rows are removed in a separately reported sensitivity):
  RIG_BENEFIT: mean delta rig_dir_ref (R11-M) CI upper < 0 AND rig limb-failure row-rate change CI upper < 0.
  MATERIAL: rig-failure rows recovered minus newly failed >= 10% of M rig-failure rows.
  RIG_HARM: newly rig-failed > 0.25 x recovered; or new false-fold rows > 0.25 x removed; or new leg-ROM rows > 0.25 x
            removed; or p99 rig_dir_ref worse; or temporal event delta per pair CI lower > 0.
  SPLIT: delta rig_dir_ref point estimate < 0 in both VALIDATION and TEST.
  SOURCE_BENEFIT: source MPJPE delta CI upper < 0 on the same rows.
CASE: D if identity/reproduction/rig-equivalence fails; A if RIG_BENEFIT and MATERIAL and SPLIT and not RIG_HARM;
B if SOURCE_BENEFIT and (not RIG_BENEFIT or RIG_HARM); otherwise C. A is scoped to restored leg joints in this
reused population; arm-restoration rig utility is not testable under the W102 contract. No CASE authorizes adoption.

Attribution of every R11 rig failure / harm row (ordered rules): F GT invalid (root behind camera); E identity (not
same-person); D source-to-rig/IK (source leg segments both <= 45 deg from GT but rig_dir_ref > 45 deg); A raw 2D
localization (a restored token on that leg > 0.10 of the GT limb-box diagonal); B frozen S0 response (restored tokens
within 0.10 yet a source leg segment > 45 deg); G unresolved. C (crop) is not repeated; W104 R01 bounds it.
R11 harm cases for inspection: source newly severe, source newly wrong knee, rig newly failed, rig_dir_ref delta > +15
deg on any leg, and W104's largest harm row (outdoors_parcours_00 #1018).

Render windows (24 consecutive original bank samples within one actor sequence, disjoint, ties by sequence id then start):
leg_natural_most (most rows with a restored leg token), arm_natural_most (most arm-only restored rows),
w104_largest_harm (centred on parcours_00 #1018), new_wrong_knee (centred on the W104 R11 newly-wrong-knee row with the
largest source MPJPE increase), w102_max_source_regression (W102's 36-sample cafe actor1 window, existing BaseRig
context), high_motion (largest GT-free mean observed 2D limb-direction step among windows with >= 3 restoration rows),
identity_caveat (most non-same-person rows the GT-free rule would change), complete_control (first all-same-person,
all-no-op window). Unavailable samples are blanked, never held.
Rig/Action equivalence (CASE D if violated): W104 R11 reproduced byte-exactly (or <= 0.1 mm); every unchanged
same-person row has an identical rig export; imported skinned rig within 1e-4 of the rest snapshot; Blender FK bone
tails within 1e-3 armature units of the exported points; M and R11 Actions with identical channels and keyed frames,
and identical key values on every rendered sample whose source is unchanged.
Real clip (GT-free, private, not unseen): HMR parent and J1X do not exist for this clip, so the W102 contract cannot be
executed on it. A labelled S0-only configuration is used for both conditions: W93 torso frames from each condition's own
S0 output as Object/thorax (W95 A_S0 policy), S0 legs, S0 left arm through the existing chain, right arm at rest.
Keys at original 50 fps frame indices; the historical renderer's scene rate (30) is not changed and is reported.
"""


def cmd_declare(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    w104 = Path(args.w104)
    ctx = load_all(args)
    rows, obs, side = ctx["rows"], ctx["obs"], ctx["side"]
    same = np.asarray(side["identity"]) == "same_person"
    rz = np.load(w104/"results/infer/restoration.npz")
    wrows, new = rz["rows"], rz["new_tokens"]
    partial = new[:, DIRECT].any(1)
    rec = build_r11_inputs(ctx)
    hm = hmr_frames(args.w102, ctx["positions"])
    has_parent = np.array([Ro is not None for Ro, _ in hm]); has_thorax = np.array([Rt is not None for _, Rt in hm])
    groups = windows_by_sid(rows)
    in_w104 = np.zeros(len(rows), bool); in_w104[wrows] = True
    newfull = np.zeros((len(rows), 17), bool); newfull[wrows] = new
    legrow = leg_restored(newfull) & in_w104
    armrow = arm_only_restored(newfull) & in_w104
    # W104-published source-level classifications used for the two harm-oriented windows (no rig outcome)
    from w104_analysis import outcome
    G = ctx["bank"].arrays["target_3d"][ctx["positions"]].astype(float); G = G-G[:, :1]
    mM = outcome(ctx["pred_M"][wrows].astype(float), G[wrows]); mR = outcome(rz["pred_R11"].astype(float), G[wrows])
    dmp = mR["mpjpe"]-mM["mpjpe"]
    newknee = ((np.nan_to_num(mM["knee_wrong"], nan=-1) == 0) & (mR["knee_wrong"] == 1)).any(1) & partial
    used, sel = set(), {}
    sel["leg_natural_most"] = best_window(groups, lambda w: int(legrow[w].sum()), used)
    used |= set(sel["leg_natural_most"])
    sel["arm_natural_most"] = best_window(groups, lambda w: int(armrow[w].sum()), used)
    used |= set(sel["arm_natural_most"])
    harm_row = int(next(i for i, r in enumerate(rows) if r["sid"] == "3dpw:outdoors_parcours_00:actor0" and r["frame"] == 1018))
    sel["w104_largest_harm"] = centred_window(groups, rows, harm_row, used)
    used |= set(sel["w104_largest_harm"] or [])
    sel_new_knee_row = None
    order = np.argsort(-np.where(newknee, dmp, -np.inf), kind="stable")
    for k in order[:int(newknee.sum())]:
        w = centred_window(groups, rows, int(wrows[k]), used)
        if w is not None:
            sel["new_wrong_knee"] = w; sel_new_knee_row = int(wrows[k]); break
    used |= set(sel.get("new_wrong_knee", []))
    w102c = json.loads((Path(args.w102)/"results/rig/cases.json").read_text())["cases"]["maximum_source_regression"]
    row_of_pos = {r["p"]: i for i, r in enumerate(rows)}
    w = [row_of_pos[f["bank_position"]] for f in w102c["frames"]]
    sel["w102_max_source_regression"] = w if not used & set(w) else None
    used |= set(w)
    xm, vm = obs["input_2d_M"], obs["input_valid_M"]
    sizes = np.array([ctx["bank"].samples[int(p)].image_size for p in ctx["positions"]], float)
    LIMB_SEGS = [(11, 12), (12, 13), (14, 15), (15, 16), (1, 2), (2, 3), (4, 5), (5, 6)]
    step = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        nxt = r.get("next")
        if not nxt:
            continue
        j = i+1 if i+1 < len(rows) and rows[i+1]["sample_id"] == nxt else None
        if j is None:
            continue
        s = [ang((xm[i, b, :2]-xm[i, a, :2])*sizes[i], (xm[j, b, :2]-xm[j, a, :2])*sizes[j])
             for a, b in LIMB_SEGS if vm[i, a] and vm[i, b] and vm[j, a] and vm[j, b]]
        step[i] = np.mean(s) if s else np.nan
    sel["high_motion"] = best_window(groups, lambda w: float(np.nanmean(step[w])) if np.isfinite(step[w]).any() else -1.0, used,
                                     admissible=lambda w: int(in_w104[w].sum()) >= 3)
    used |= set(sel["high_motion"] or [])
    idc = ~same & rec["changed"]
    sel["identity_caveat"] = best_window(groups, lambda w: int(idc[w].sum()), used)
    used |= set(sel["identity_caveat"] or [])
    noop = same & ~rec["changed"]
    sel["complete_control"] = best_window(groups, lambda w: 0, used, admissible=lambda w: bool(noop[w].all()))
    windows = {k: {"rows": [int(i) for i in v], "sequence_id": rows[v[0]]["sid"], "start_frame": rows[v[0]]["frame"],
                   "samples": len(v), "split": rows[v[0]]["split"],
                   "identity": dict(Counter(rows[i]["identity"] for i in v)),
                   "w104_restoration_rows": int(in_w104[v].sum()), "leg_restored_rows": int(legrow[v].sum()),
                   "arm_only_restored_rows": int(armrow[v].sum()), "non_same_person_rule_changed": int(idc[v].sum())}
               for k, v in sel.items() if v is not None}
    decl = {"schema": "w105_declaration_v1", "text": DECLARATION_TEXT.strip(), "seed": SEED, "window_samples": WINDOW,
            "GT_used_for": "only W104-published source-level per-row classifications for the new_wrong_knee window; no rig outcome read",
            "rig_outputs_read": False, "w104_declaration_sha256": W104_DECLARATION_SHA,
            "thresholds": {"DIR_FAIL_DEG": DIR_FAIL_DEG, "FOLD_DEG": FOLD_DEG, "STEP_DEG": STEP_DEG, "GT_QUIET_DEG": GT_QUIET_DEG,
                           "THIGH_STEP_DEG": THIGH_STEP_DEG, "GT_THIGH_QUIET": GT_THIGH_QUIET, "HARM_RATIO": HARM_RATIO,
                           "MATERIAL_NET_SHARE": MATERIAL_NET_SHARE, "PCK_RESTORED": PCK_RESTORED, "TRANSLATION_BAND_DEG": TRANSLATION_BAND_DEG},
            "rig": {"assets": {k: {"path": p, "sha256": sha(ROOT/p)} for k, p in RIG_ASSETS.items()},
                    "render": RENDER_SETTINGS, "blender": "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe " + " ".join(BLENDER_ARGS)},
            "populations": {"rows": len(rows), "identity": dict(Counter(r["identity"] for r in rows)),
                            "w104_restoration_rows": int(len(wrows)), "w104_partial_rows": int(partial.sum()),
                            "partial_leg_restored": int(legrow.sum()), "partial_arm_only_restored": int(armrow.sum()),
                            "same_person_noop_rows": int((noop & rec["present"]).sum()),
                            "w104_rows_without_HMR_parent": int((~has_parent[wrows]).sum()),
                            "w104_rows_without_HMR_thorax": int((has_parent & ~has_thorax)[wrows].sum()),
                            "non_same_person_rows_rule_would_change": dict(Counter(np.asarray(side["identity"])[idc]))},
            "selection": {"windows": windows, "new_wrong_knee_centre_row": sel_new_knee_row,
                          "w104_largest_harm_row": harm_row}}
    dump(out/"declaration.json", decl)
    print("declaration sha256", sha(out/"declaration.json"))
    print(json.dumps(decl["populations"]), json.dumps({k: {kk: v[kk] for kk in v if kk != "rows"} for k, v in windows.items()}, default=str))


# ------------------------------------------------------------------ stage: build (GT-free)
def cmd_build(args):
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from framepose.train import load_checkpoint
    from w104_missing_joint import s0_forward
    decl = require_declaration(args)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    w104 = Path(args.w104)
    ctx = load_all(args)
    rows, pos = ctx["rows"], ctx["positions"]
    rec = build_r11_inputs(ctx)
    rz = np.load(w104/"results/infer/restoration.npz")
    model, _ = load_checkpoint(Path(args.w101_data)/"s0_checkpoint.pt", device="cuda")
    wrows = rz["rows"]
    fresh_w104 = s0_forward(model, torch, rec["geometry"][wrows])
    other = np.flatnonzero(rec["changed"] & ~np.isin(np.arange(len(rows)), wrows))
    fresh_other = s0_forward(model, torch, rec["geometry"][other]) if len(other) else np.zeros((0, 17, 3), np.float32)
    pR = ctx["pred_M"].copy(); pR[wrows] = rz["pred_R11"]; pR[other] = fresh_other
    report = {"schema": "w105_build_v1", "declaration_sha256": args.declaration_sha, "GT_read": False,
              "r11_w104_rows_fresh_forward_byte_equal": bool(fresh_w104.tobytes() == rz["pred_R11"].tobytes()),
              "r11_w104_rows_fresh_forward_max_abs_mm": float(1000*np.abs(fresh_w104-rz["pred_R11"]).max()),
              "rows_changed_by_rule": int(rec["changed"].sum()), "w104_rows": int(len(wrows)),
              "additional_rows_outside_w104_population": int(len(other)),
              "additional_rows_identity": dict(Counter(rows[i]["identity"] for i in other)),
              "additional_run_note": "same fixed GT-free W104 rule, new S0 forward (batch layout differs); identity/coverage caveat only"}
    if not report["r11_w104_rows_fresh_forward_byte_equal"] and report["r11_w104_rows_fresh_forward_max_abs_mm"] > 0.1:
        raise RuntimeError("STOP: W104 R11 cannot be reproduced (CASE D)")
    np.savez_compressed(out/"sources.npz", pred_M=ctx["pred_M"], pred_R11=pR, new_tokens=rec["new"], changed=rec["changed"],
                        w104_rows=wrows, other_rows=other)
    # rig exports for every row with a parent, both conditions (+ secondary S0 left arm)
    rc = rig_context()
    hm = hmr_frames(args.w102, pos)
    j1 = load_j1x(args.w102, len(ctx["bank"]))
    rig = {"M": [None]*len(rows), "R11": [None]*len(rows)}
    arm = {"M": [None]*len(rows), "R11": [None]*len(rows)}
    excl = Counter()
    for i in range(len(rows)):
        Ro, Rt = hm[i]
        if Ro is None:
            excl["no_HMR_parent"] += 1; continue
        if Rt is None:
            excl["HMR_thorax_degenerate"] += 1; continue
        rig["M"][i] = rig_measure(ctx["pred_M"][i], Ro, Rt, rc)
        rig["R11"][i] = rig["M"][i] if not rec["changed"][i] else rig_measure(pR[i], Ro, Rt, rc)
        arm["M"][i] = s0_left_arm(ctx["pred_M"][i], Ro, Rt, rc)
        arm["R11"][i] = arm["M"][i] if not rec["changed"][i] else s0_left_arm(pR[i], Ro, Rt, rc)
        for c in ("M", "R11"):
            if rig[c][i] is None:
                excl[f"degenerate_chain_{c}"] += 1
    report["rig_exclusions_all_rows"] = dict(excl)
    # no-op control: recompute R11 rig on unchanged same-person rows from its own (identical) tensor
    same = np.asarray(ctx["side"]["identity"]) == "same_person"
    noop = np.flatnonzero(same & ~rec["changed"] & np.array([rig["M"][i] is not None for i in range(len(rows))]))
    eq = sum(json.dumps(rig_measure(pR[i], *hm[i], rc)["export"]) == json.dumps(rig["M"][i]["export"]) for i in noop)
    report["noop_rows_rig_export_identical"] = {"rows": int(len(noop)), "identical": int(eq),
                                                "R11_source_bytes_equal_M": bool(pR[noop].tobytes() == ctx["pred_M"][noop].tobytes())}
    with (out/"rig_rows.json").open("w") as fh:
        json.dump({"rig": rig, "arm": arm}, fh)
    # Blender case file for the declared windows (candidates only; no GT)
    cases = {}
    for name, w in decl["selection"]["windows"].items():
        cases[name] = window_case(name, w["rows"], rows, ctx, hm, j1, rig, pR, rec, rc)
    dump(out/"cases.json", {"rest_snapshot_digest": rc["rest"].digest(), "canonical_to_armature_xyzw": list(rc["alignment"]),
                            "target_chain": {"root_head": list(rc["target"].root_head)}, "cases": cases})
    # real clip (GT-free, S0-only configuration)
    report["real_clip"] = build_clip(args, out, model, torch, rc)
    report["outputs_sha256"] = {f: sha(out/f) for f in ("sources.npz", "rig_rows.json", "cases.json", "video_cases.json", "video_sources.npz")}
    report["gpu_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated()/2**20
    dump(out/"build_report.json", report)
    print(json.dumps(report, default=str)[:3000])


def frame_record(bank, p, slot_poses, slot_arms, identity, labels):
    s = bank.samples[int(p)]
    fr = {"bank_position": int(p), "frame_index": s.frame_index, "time_s": s.timestamp, "image": s.image_reference.relative_path,
          "identity": identity, "runtime_status": "DIAGNOSTIC", "runtime_reasons": [], "yaw_deg": {}, "modes": {}, "forward_cam": {},
          "mode_lines": {}, "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {}, "fk_points_armature": {}, "gt_valid": False}
    for slot, pose in slot_poses.items():
        if pose is None:
            fr["yaw_deg"][slot] = None; fr["mode_lines"][slot] = ["UNAVAILABLE; no source/GT fallback"]; continue
        e = pose["export"]
        fr["object_rotation_xyzw"][slot] = e["object"]; fr["extra_bone_locals_xyzw"][slot] = dict(e["bones"])
        fr["fk_points_armature"][slot] = e["points"]; fr["forward_cam"][slot] = e["forward"]
        Rt_fwd = e["forward"]; fr["yaw_deg"][slot] = math.degrees(math.atan2(-Rt_fwd[0], Rt_fwd[1]))
        a = slot_arms.get(slot)
        if a is not None:
            fr["modes"][slot] = a
        else:
            for bone in ("upperarm_l", "lowerarm_l"):
                fr["extra_bone_locals_xyzw"][slot][bone] = [0., 0., 0., 1.]
        fr["mode_lines"][slot] = [labels[slot]]
    return fr


def window_case(name, w, rows, ctx, hm, j1, rig, pR, rec, rc):
    from w102_rig_review import arm_export
    frames = []
    for i in w:
        r = rows[i]; p = r["p"]; Ro, Rt = hm[i]
        shared = None if Rt is None else arm_export(j1["left_arm_g1_elbow"][p], j1["left_arm_g1_wrist"][p], Ro, Rt, rc)
        # W102 build shows the M slot only for tracked rows; R11 is defined wherever M is
        poses = {"FIXED": rig["M"][i] if r["tracked"] else None, "RUNTIME": rig["R11"][i] if r["tracked"] else None}
        fr = frame_record(ctx["bank"], p, poses, {"FIXED": shared, "RUNTIME": shared}, r["identity"],
                          {"FIXED": "M_Main legs", "RUNTIME": "R11 legs (source-only restoration)"})
        if len(fr["object_rotation_xyzw"]) == 2 and (fr["object_rotation_xyzw"]["FIXED"] != fr["object_rotation_xyzw"]["RUNTIME"]
                                                     or fr["extra_bone_locals_xyzw"]["FIXED"]["spine_01"] != fr["extra_bone_locals_xyzw"]["RUNTIME"]["spine_01"]
                                                     or fr["modes"].get("FIXED") != fr["modes"].get("RUNTIME")):
            raise RuntimeError("STOP: common parent / arm invariant failed")
        fr["row"] = int(i); fr["sample_id"] = r["sample_id"]; fr["split"] = r["split"]
        fr["restored_tokens"] = [JOINTS[k] for k in np.flatnonzero(rec["new"][i])]
        fr["source_changed"] = bool(rec["changed"][i])
        frames.append(fr)
    delta = np.diff([f["time_s"] for f in frames])
    return {"case": name, "sequence_id": rows[w[0]]["sid"], "start_frame": rows[w[0]]["frame"],
            "fps_sampled": 1/float(np.median(delta)) if len(delta) else 10., "frames": frames,
            "labels": {"FIXED": "M_Main (S0 legs)", "RUNTIME": "R11 (S0 legs, restored raw joints)"}, "short_labels": {"FIXED": "M", "RUNTIME": "R11"},
            "title": "W105 M_Main vs R11 | W102 BaseRig contract: shared HMR parent, shared frozen J1X left arm",
            "header_note": "Only S0 leg geometry differs; association labels evaluation-only; unavailable panels blanked"}


def build_clip(args, out, model, torch, rc):
    """Real clip: M reproduction (byte-equal to W102), R11 by the W104 rule, GT-free S0-only rig configuration."""
    from framepose.crops import crop_box, geometry_in_crop
    from motion.torso_frame import torso_frames
    from w101_reference_estimator import to_input
    from w102_rig_review import arm_export
    from w104_missing_joint import VIDEO_PREDICTIONS, restore_row, restoration_inputs, s0_forward
    z = np.load(Path(args.w102)/VIDEO_PREDICTIONS)
    lm, x0, v0, geo0, p0, ts = z["landmarks"], z["input_2d"], z["input_valid"], z["geometry"], z["predictions"], z["timestamp"]
    insp = json.loads((Path(args.w102)/"results/video_inspection/inspection.json").read_text())
    size = tuple(insp["size"])
    rr = [to_input(l if np.isfinite(l[:, 0]).any() else None, size) for l in lm]
    x = np.stack([r[0] for r in rr]); v = np.stack([r[1] for r in rr])
    geo = np.stack([geometry_in_crop(a, b, size, crop_box(a, b, size)) for a, b in rr])
    pM = s0_forward(model, torch, geo)
    R = geo.copy(); new = np.zeros((len(lm), 17), bool)
    for i in range(len(lm)):
        raw, have, nw = restore_row(lm[i], size, x[i], v[i])
        if nw.any():
            R[i] = restoration_inputs(x[i], v[i], raw, have, size)[0]["R11"]; new[i] = nw
    pR = s0_forward(model, torch, R)
    changed = new.any(1)
    dev = 1000*np.linalg.norm((pR-pR[:, :1])-(pM-pM[:, :1]), axis=-1).mean(-1)
    np.savez_compressed(out/"video_sources.npz", pred_M=pM, pred_R11=pR, new_tokens=new, input_valid=v, timestamp=ts)
    frames, avail = [], {"M": 0, "R11": 0}
    for i in range(len(lm)):
        poses, arms, parents = {}, {}, {}
        for slot, P in (("FIXED", pM[i]), ("RUNTIME", pR[i])):
            j = np.asarray(P, float); j = j-j[0]
            Rt, Rp = torso_frames(j)
            parents[slot] = (Rp, Rt)
            poses[slot] = rig_measure(j, Rp, Rt, rc) if Rp is not None and Rt is not None else None
            arms[slot] = None if poses[slot] is None else arm_export(j[12]-j[11], j[13]-j[11], Rp, Rt, rc)
        fr = {"bank_position": -1, "frame_index": i, "time_s": float(ts[i]), "image": f"frame_{i:04d}", "identity": "single_subject_GT_free",
              "runtime_status": "DIAGNOSTIC", "runtime_reasons": [], "yaw_deg": {}, "modes": {}, "forward_cam": {}, "mode_lines": {},
              "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {}, "fk_points_armature": {}, "gt_valid": False}
        for slot in ("FIXED", "RUNTIME"):
            p = poses[slot]
            if p is None:
                fr["yaw_deg"][slot] = None; continue
            avail["M" if slot == "FIXED" else "R11"] += 1
            fr["object_rotation_xyzw"][slot] = p["export"]["object"]; fr["extra_bone_locals_xyzw"][slot] = dict(p["export"]["bones"])
            fr["fk_points_armature"][slot] = p["export"]["points"]; fr["forward_cam"][slot] = p["export"]["forward"]
            fr["yaw_deg"][slot] = math.degrees(math.atan2(-p["export"]["forward"][0], p["export"]["forward"][1]))
            if arms[slot] is not None:
                fr["modes"][slot] = arms[slot]
            else:
                for bone in ("upperarm_l", "lowerarm_l"):
                    fr["extra_bone_locals_xyzw"][slot][bone] = [0., 0., 0., 1.]
        fr["restored_tokens"] = [JOINTS[k] for k in np.flatnonzero(new[i])]
        fr["source_changed"] = bool(changed[i])
        fr["rig"] = {s: poses[s] and {"legs": poses[s]["legs"], "viol": poses[s]["viol"]} for s in poses}
        fr["arm_rig"] = {}
        for s in ("FIXED", "RUNTIME"):
            a = arms[s]
            if a is not None:
                from run_torso_frame_rig_replay import arm_pose
                Rp, Rt = parents[s]
                j = np.asarray(pM[i] if s == "FIXED" else pR[i], float); j = j-j[0]
                ap = arm_pose(j[12]-j[11], j[13]-j[11], Rt, rc["alignment"], rc["target"], rc["rest"])
                fr["arm_rig"][s] = {"upper_w": unit(Rt @ (rc["Am"].T @ np.asarray(ap.upper))).tolist(),
                                    "fore_w": unit(Rt @ (rc["Am"].T @ np.asarray(ap.forearm))).tolist(),
                                    "elbow": float(ap.bend_angle_degrees())}
        fr["parent"] = {s: None if parents[s][0] is None else {"Rp": parents[s][0].tolist(), "Rt": parents[s][1].tolist()} for s in parents}
        frames.append(fr)
    dump(out/"video_cases.json", {"rest_snapshot_digest": rc["rest"].digest(), "canonical_to_armature_xyzw": list(rc["alignment"]),
                                  "target_chain": {"root_head": list(rc["target"].root_head)},
                                  "cases": {"real_clip": {"case": "real_clip", "sequence_id": "production_demo (historical, private)",
                                                          "start_frame": 0, "fps_sampled": 50.0, "frames": frames,
                                                          "labels": {"FIXED": "M_Main (S0-only config)", "RUNTIME": "R11 (S0-only config)"},
                                                          "short_labels": {"FIXED": "M", "RUNTIME": "R11"},
                                                          "title": "W105 real clip | GT-free S0-only configuration (NOT the W102 HMR/J1X contract)",
                                                          "header_note": "W93 torso frames from each condition's S0; S0 legs + S0 left arm; right arm rest"}}})
    return {"frames": len(lm), "input_reproduced": bool(np.array_equal(x, x0) and np.array_equal(v, v0)),
            "geometry_byte_equal": bool(geo.tobytes() == geo0.tobytes()), "M_prediction_byte_equal_w102": bool(pM.tobytes() == p0.tobytes()),
            "frames_changed_by_R11": int(changed.sum()), "restored_tokens": dict(Counter(JOINTS[k] for k in np.flatnonzero(new.any(0)) for _ in range(int(new[:, k].sum())))),
            "mean_joint_deviation_from_M_mm": float(dev[changed].mean()) if changed.any() else 0.0,
            "rig_available_frames": avail, "fps": 50.0, "configuration": "GT-free S0-only (W93/W95 A_S0 parent); not the W102 contract"}


# ------------------------------------------------------------------ stage: analyze (evaluation-only GT)
def gt_rig(G, hm, rc, rows_idx):
    out = {}
    for i in rows_idx:
        Ro, Rt = hm[i]
        out[int(i)] = rig_measure(G[i], Ro, Rt, rc) if Ro is not None and Rt is not None else None
    return out


def gt_left_arm(G, hm, rc, rows_idx):
    return {int(i): (s0_left_arm(G[i], *hm[i], rc) if hm[i][0] is not None and hm[i][1] is not None else None) for i in rows_idx}


def leg_metrics(c, ref, G):
    """Per leg: rig bone errors vs GT-through-rig (ref) and vs raw GT segments, knee, branch, fold."""
    out = {}
    for s, (h, k, a) in LEG_IDX.items():
        L, R = c["legs"][s], ref["legs"][s]
        dref = [ang(L["thigh_w"], R["thigh_w"]), ang(L["shin_w"], R["shin_w"])]
        dgt = [ang(L["thigh_w"], G[k]-G[h]), ang(L["shin_w"], G[a]-G[k])]
        nc, nr = np.cross(L["thigh_w"], L["shin_w"]), np.cross(R["thigh_w"], R["shin_w"])
        elig = L["knee"] >= 20.0 and R["knee"] >= 20.0
        wrong = (ang(nc, nr) > 90.0) if elig else None
        out[s] = {"dir_ref": dref, "dir_gt": dgt, "knee_err": abs(L["knee"]-R["knee"]), "knee_wrong": wrong,
                  "fail": bool(max(dref) > DIR_FAIL_DEG or wrong is True), "fold": bool(L["knee"]-R["knee"] > FOLD_DEG),
                  "ankle_err": float(np.linalg.norm(np.asarray(L["ankle_rel_w"])-np.asarray(R["ankle_rel_w"]))),
                  "s2r": max(L["s2r_thigh"], L["s2r_shin"])}
    return out


def pair_events(ra, rb, Ga, Gb):
    """(direction-step events, local-thigh-step events) for one original neighbour pair of one condition."""
    from retarget.framepose_two_bone_ik import quaternion_angle_degrees
    n1 = n2 = 0
    for s, (h, k, an) in LEG_IDX.items():
        for bone, (p0, p1) in (("thigh", (h, k)), ("shin", (k, an))):
            g = ang(Ga[p1]-Ga[p0], Gb[p1]-Gb[p0])
            st = ang(ra["legs"][s][bone+"_w"], rb["legs"][s][bone+"_w"])
            n1 += int(st > STEP_DEG and g < GT_QUIET_DEG)
        gth = ang(Ga[k]-Ga[h], Gb[k]-Gb[h])
        q = quaternion_angle_degrees(ra["legs"][s]["thigh_local"], rb["legs"][s]["thigh_local"])
        n2 += int(q > THIGH_STEP_DEG and gth < GT_THIGH_QUIET)
    return n1, n2


def decide_case(b, tmp):
    """Declared gates on P_leg (see DECLARATION_TEXT)."""
    rb = b["rig_dir_ref"]["delta"]["ci95"] is not None and b["rig_dir_ref"]["delta"]["ci95"][1] < 0 and \
        b["rig_fail_rows"]["rate_change"]["ci95"] is not None and b["rig_fail_rows"]["rate_change"]["ci95"][1] < 0
    f = b["rig_fail_rows"]
    material = (f["recovered"]-f["new"]) >= MATERIAL_NET_SHARE*max(f["M"], 1)
    tci = tmp.get("dir_step_events", {}).get("delta_per_pair", {}).get("ci95")
    tci2 = tmp.get("thigh_step_events", {}).get("delta_per_pair", {}).get("ci95")
    harm = {"rig_fail": f["new"] > HARM_RATIO*f["recovered"],
            "false_fold": b["false_fold_rows"]["new"] > HARM_RATIO*b["false_fold_rows"]["removed"],
            "leg_rom": b["leg_rom_rows"]["new"] > HARM_RATIO*b["leg_rom_rows"]["removed"],
            "p99_worse": b["rig_dir_ref"]["p99"][1] > b["rig_dir_ref"]["p99"][0],
            "temporal": bool((tci is not None and tci[0] > 0) or (tci2 is not None and tci2[0] > 0))}
    sp = all(b["rig_dir_ref_delta_by_split"][s]["mean"] is not None and b["rig_dir_ref_delta_by_split"][s]["mean"] < 0 for s in ("validation", "test"))
    sb = b["source_dMPJPE"]["ci95"] is not None and b["source_dMPJPE"]["ci95"][1] < 0
    H = any(harm.values())
    case = "A" if (rb and material and sp and not H) else ("B" if (sb and (not rb or H)) else "C")
    return {"RIG_BENEFIT": bool(rb), "MATERIAL": bool(material), "RIG_HARM": bool(H), "harm_components": harm, "SPLIT": bool(sp),
            "SOURCE_BENEFIT": bool(sb), "CASE": case}


def contract_checks(brep, args):
    """CASE D conditions: reproduction, no-op identity, rig equivalence, FK residual and Action identity."""
    out = {"r11_reproduced": bool(brep["r11_w104_rows_fresh_forward_byte_equal"] or brep["r11_w104_rows_fresh_forward_max_abs_mm"] <= 0.1),
           "noop_rig_identical": brep["noop_rows_rig_export_identical"]["identical"] == brep["noop_rows_rig_export_identical"]["rows"]
                                 and brep["noop_rows_rig_export_identical"]["R11_source_bytes_equal_M"]}
    br = json.loads(Path(args.render_report).read_text())
    fk = [v for c in br["cases"].values() for v in list(c.get("fk_points_vs_export_max", {}).values())]
    out["rest_difference"] = br["equivalent_rig_max_rest_difference"]
    out["max_fk_points_vs_export"] = max(fk) if fk else None
    out["rig_equivalent"] = br["equivalent_rig_max_rest_difference"] is not None and br["equivalent_rig_max_rest_difference"] <= 1e-4
    out["fk_within_1e-3"] = bool(fk) and max(fk) <= 1e-3
    out["action_identity"] = action_identity(json.loads(Path(args.audit_report).read_text()), json.loads((Path(args.build_dir)/"cases.json").read_text()))
    out["pass"] = bool(out["r11_reproduced"] and out["noop_rig_identical"] and out["rig_equivalent"] and out["fk_within_1e-3"]
                       and out["action_identity"]["pass"])
    return out


def action_identity(audit, cases):
    """M (FIXED) vs R11 (RUNTIME) Actions: same channels and keyed frames; same values on unchanged-source samples."""
    res, ok = {}, True
    for name, case in cases["cases"].items():
        f = audit["files"].get(f"{name}/{name}.blend")
        if f is None:
            res[name] = "missing .blend"; ok = False; continue
        A = {k.rsplit(" ", 1)[1]: v for k, v in f["actions"].items() if k.startswith(f"W91 {name} ")}
        a, b = A.get("FIXED"), A.get("RUNTIME")
        if a is None or b is None:
            res[name] = "missing Action"; ok = False; continue
        chan = lambda x: [(c["data_path"], c["index"], c["frames"]) for c in x["channels"]]  # noqa: E731
        f0 = case["frames"][0]["frame_index"]
        unchanged = {fr["frame_index"]-f0+1 for fr in case["frames"] if not fr["source_changed"]}
        mism = 0
        for ca, cb in zip(a["channels"], b["channels"]):
            for fa, va, vb in zip(ca["frames"], ca["values"], cb["values"]):
                if round(fa) in unchanged and abs(va-vb) > 1e-9:
                    mism += 1
        same_layout = chan(a) == chan(b)
        res[name] = {"same_channels_and_keyed_frames": same_layout, "fcurves": a["fcurves"], "keyed_frames": len(a["keyed_frames"]),
                     "interpolation": sorted(set(a["interpolation"]) | set(b["interpolation"])),
                     "unchanged_source_samples": len(unchanged), "unchanged_sample_value_mismatches": mism,
                     "value_digest_equal": a["value_digest"] == b["value_digest"], "scene_fps": f["scene_fps"]}
        ok &= same_layout and mism == 0
    return {"pass": bool(ok), "cases": res}


LEG_ROM = ("hip_flexion", "hip_extension", "hip_abduction", "hip_adduction", "knee_flexion", "knee_hinge")


def cmd_analyze(args):
    from w103_observability import raw_references
    from w104_analysis import outcome
    from eval_real_observation_estimator import CANON, SOURCE_OF
    decl = require_declaration(args)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    bd = Path(args.build_dir)
    brep = json.loads((bd/"build_report.json").read_text())
    if brep["declaration_sha256"] != args.declaration_sha or any(sha(bd/f) != h for f, h in brep["outputs_sha256"].items()):
        raise RuntimeError("STOP: build outputs changed")
    contract = contract_checks(brep, args)
    ctx = load_all(args)
    rows, pos, bank = ctx["rows"], ctx["positions"], ctx["bank"]
    src = np.load(bd/"sources.npz")
    pM, pR, new, wrows = src["pred_M"].astype(float), src["pred_R11"].astype(float), src["new_tokens"], src["w104_rows"]
    rr = json.loads((bd/"rig_rows.json").read_text()); rig, arm = rr["rig"], rr["arm"]
    raw = raw_references(Path(args.w102)/"data/raw", bank, pos)
    G = bank.arrays["target_3d"][pos].astype(float); G = G-G[:, :1]
    if not raw["valid"].all() or np.abs(raw["G"]-G).max() > 1e-6:
        raise RuntimeError("STOP: GT alignment")
    behind = raw["root"][:, 1] <= 0
    rc = rig_context(); hm = hmr_frames(args.w102, pos)
    same = np.asarray(ctx["side"]["identity"]) == "same_person"
    vid = np.array([video_of_sid(r["sid"]) for r in rows]); split = np.array([r["split"] for r in rows])
    partial = np.zeros(len(rows), bool); partial[wrows] = new[wrows][:, DIRECT].any(1)
    legrow, armrow = partial & leg_restored(new), partial & arm_only_restored(new)
    # neighbours for temporal pairs
    idx_of = {r["sample_id"]: i for i, r in enumerate(rows)}
    pairs = [(i, idx_of[r["next"]]) for i, r in enumerate(rows) if r.get("next") in idx_of]
    need = set(np.flatnonzero(partial)) | {b for a, b in pairs if partial[a]} | {a for a, b in pairs if partial[b]}
    ref = gt_rig(G, hm, rc, sorted(need))
    # ---- per-row evaluation table
    T = {}
    for i in sorted(need):
        if rig["M"][i] is None or rig["R11"][i] is None or ref[i] is None:
            continue
        T[i] = {c: leg_metrics(rig[c][i], ref[i], G[i]) for c in ("M", "R11")}
        for c in ("M", "R11"):
            v = rig[c][i]["viol"]
            T[i][c]["rom"] = bool(any(v.get(k, False) for k in LEG_ROM))
    elig = np.array([i in T for i in range(len(rows))])
    excl = Counter()
    for i in np.flatnonzero(partial):
        if hm[i][0] is None: excl["no_HMR_parent"] += 1
        elif hm[i][1] is None: excl["HMR_thorax_degenerate"] += 1
        elif rig["M"][i] is None: excl["degenerate_chain_M"] += 1
        elif rig["R11"][i] is None: excl["degenerate_chain_R11"] += 1
        elif ref[i] is None: excl["degenerate_chain_GT_RIG"] += 1
    sm = {"M": outcome(pM, G), "R11": outcome(pR, G)}

    def rowvals(sel, fn):
        return np.array([fn(T[i]) for i in np.flatnonzero(sel)])

    def mean_dir(t, c):
        return float(np.mean([x for s in "lr" for x in t[c][s]["dir_ref"]]))

    def block(sel):
        sel = sel & elig
        V = vid[sel]; I = np.flatnonzero(sel)
        dM, dR = rowvals(sel, lambda t: mean_dir(t, "M")), rowvals(sel, lambda t: mean_dir(t, "R11"))
        fM = rowvals(sel, lambda t: t["M"]["l"]["fail"] or t["M"]["r"]["fail"]); fR = rowvals(sel, lambda t: t["R11"]["l"]["fail"] or t["R11"]["r"]["fail"])
        oM = rowvals(sel, lambda t: t["M"]["l"]["fold"] or t["M"]["r"]["fold"]); oR = rowvals(sel, lambda t: t["R11"]["l"]["fold"] or t["R11"]["r"]["fold"])
        rM, rR = rowvals(sel, lambda t: t["M"]["rom"]), rowvals(sel, lambda t: t["R11"]["rom"])
        gM = rowvals(sel, lambda t: np.mean([x for s in "lr" for x in t["M"][s]["dir_gt"]])); gR = rowvals(sel, lambda t: np.mean([x for s in "lr" for x in t["R11"][s]["dir_gt"]]))
        kM = rowvals(sel, lambda t: np.mean([t["M"][s]["knee_err"] for s in "lr"])); kR = rowvals(sel, lambda t: np.mean([t["R11"][s]["knee_err"] for s in "lr"]))
        aM = rowvals(sel, lambda t: np.mean([t["M"][s]["ankle_err"] for s in "lr"])); aR = rowvals(sel, lambda t: np.mean([t["R11"][s]["ankle_err"] for s in "lr"]))
        sM = rowvals(sel, lambda t: max(t["M"][s]["s2r"] for s in "lr")); sR = rowvals(sel, lambda t: max(t["R11"][s]["s2r"] for s in "lr"))
        kw = {c: [t[c][s]["knee_wrong"] for i in I for t in [T[i]] for s in "lr"] for c in ("M", "R11")}
        kn = [(a, b) for a, b in zip(kw["M"], kw["R11"]) if a is not None and b is not None]
        srcM, srcR = sm["M"]["mpjpe"][sel], sm["R11"]["mpjpe"][sel]
        if not len(I):
            return {"rows": 0}
        return {"rows": int(len(I)), "videos": int(len(set(V))), "split": dict(Counter(split[sel])),
                "source_dMPJPE": video_bootstrap(V, srcR-srcM), "source_mean_mpjpe": [float(srcM.mean()), float(srcR.mean())],
                "source_severe": [int(sm["M"]["severe"][sel].sum()), int(sm["R11"]["severe"][sel].sum())],
                "rig_dir_ref": {"M": distribution(dM), "R11": distribution(dR), "delta": video_bootstrap(V, dR-dM),
                                "p99": [float(np.percentile(dM, 99)), float(np.percentile(dR, 99))]},
                "rig_dir_gt": {"M_mean": float(gM.mean()), "R11_mean": float(gR.mean()), "delta": video_bootstrap(V, gR-gM)},
                "rig_knee_abs_err": {"M_mean": float(kM.mean()), "R11_mean": float(kR.mean()), "delta": video_bootstrap(V, kR-kM)},
                "rig_ankle_endpoint_err_armature_units": {"M_mean": float(aM.mean()), "R11_mean": float(aR.mean()), "delta": video_bootstrap(V, aR-aM)},
                "rig_fail_rows": {"M": int(fM.sum()), "R11": int(fR.sum()), "new": int((fR & ~fM).sum()), "recovered": int((fM & ~fR).sum()),
                                  "rate_change": video_bootstrap(V, fR.astype(float)-fM)},
                "false_fold_rows": {"M": int(oM.sum()), "R11": int(oR.sum()), "new": int((oR & ~oM).sum()), "removed": int((oM & ~oR).sum())},
                "leg_rom_rows": {"M": int(rM.sum()), "R11": int(rR.sum()), "new": int((rR & ~rM).sum()), "removed": int((rM & ~rR).sum())},
                "rig_knee_branch_legs": {"both_eligible": len(kn), "M_wrong": sum(a for a, b in kn), "R11_wrong": sum(b for a, b in kn),
                                         "newly_wrong": sum((not a) and b for a, b in kn), "fixed": sum(a and not b for a, b in kn)},
                "source_to_rig_max_deg": {"M": distribution(sM), "R11": distribution(sR), "M_over_45": int((sM > 45).sum()), "R11_over_45": int((sR > 45).sum())},
                "rig_dir_ref_delta_by_split": {s: video_bootstrap(V[split[sel] == s], (dR-dM)[split[sel] == s]) for s in ("validation", "test")},
                "_rows": I, "_dM": dM, "_dR": dR}

    def temporal(sel_rows, keep):
        res = {"pairs": 0}
        ev = {c: [] for c in ("M", "R11")}; th = {c: [] for c in ("M", "R11")}; V = []; trans = []
        vm = ctx["obs"]["input_valid_M"]
        for a, b in pairs:
            if not (sel_rows[a] or sel_rows[b]) or a not in T or b not in T or not (same[a] and same[b]) or not (keep[a] and keep[b]):
                continue
            V.append(vid[a]); trans.append(bool((vm[a, list(LEG_TOKENS)] != vm[b, list(LEG_TOKENS)]).any()))
            for c in ("M", "R11"):
                n1, n2 = pair_events(rig[c][a], rig[c][b], G[a], G[b])
                ev[c].append(n1); th[c].append(n2)
        if not V:
            return res
        e = {c: np.array(v, float) for c, v in ev.items()}; t2 = {c: np.array(v, float) for c, v in th.items()}; tr = np.array(trans)
        res = {"pairs": len(V), "mask_transition_pairs": int(tr.sum()),
               "dir_step_events": {"M": int(e["M"].sum()), "R11": int(e["R11"].sum()), "delta_per_pair": video_bootstrap(V, e["R11"]-e["M"])},
               "thigh_step_events": {"M": int(t2["M"].sum()), "R11": int(t2["R11"].sum()), "delta_per_pair": video_bootstrap(V, t2["R11"]-t2["M"])},
               "at_mask_transitions": {"pairs": int(tr.sum()), "dir_step_M": int(e["M"][tr].sum()), "dir_step_R11": int(e["R11"][tr].sum()),
                                       "thigh_step_M": int(t2["M"][tr].sum()), "thigh_step_R11": int(t2["R11"][tr].sum())},
               "not_at_transitions": {"pairs": int((~tr).sum()), "dir_step_M": int(e["M"][~tr].sum()), "dir_step_R11": int(e["R11"][~tr].sum())}}
        return res

    report = {"schema": "w105_analysis_v1", "declaration_sha256": args.declaration_sha, "regimes": REGIME,
              "reference_validity": {"rows_root_behind_camera": int(behind.sum()), "partial_rows": int((behind & partial).sum())},
              "denominators": {"all_rows": len(rows), "identity": dict(Counter(r["identity"] for r in rows)),
                               "split_identity": {s: dict(Counter(r["identity"] for r in rows if r["split"] == s)) for s in ("validation", "test")},
                               "w104_restoration_rows": int(len(wrows)), "partial": int(partial.sum()), "partial_leg": int(legrow.sum()),
                               "partial_arm_only": int(armrow.sum()), "partial_rig_eligible": int((partial & elig).sum()),
                               "partial_exclusions": dict(excl), "noop_same_person": brep["noop_rows_rig_export_identical"]}}
    runs = {}
    for run, keep in (("primary", np.ones(len(rows), bool)), ("reference_valid_sensitivity", ~behind)):
        res = {}
        for name, sel in (("P_leg", legrow), ("P_arm", armrow), ("P_all", partial)):
            res[name] = block(sel & keep)
        tmp = temporal(legrow, keep)
        res["temporal_P_leg"] = tmp
        res["temporal_P_all"] = temporal(partial, keep)
        res["gates"] = decide_case(res["P_leg"], tmp)
        runs[run] = res
        print(run, json.dumps(res["gates"]), flush=True)
    report["contract"] = contract
    report["CASE"] = runs["primary"]["gates"]["CASE"] if contract["pass"] else "D"
    P = runs["primary"]["P_leg"]
    # strata (primary, P_leg)
    I, dd = P["_rows"], P["_dR"]-P["_dM"]
    lmr = ctx["lm"]; srcidx = [CANON.index(SOURCE_OF[n]) for n in JOINTS]; conf = lmr[:, srcidx, 2]
    minconf = np.where(new[:, DIRECT], conf[:, DIRECT], np.inf).min(1)
    uv = raw["uv"]; gt12 = uv[:, [SMPL_OF[j] for j in DIRECT]]
    diag = np.linalg.norm(gt12.max(1)-gt12.min(1), axis=-1)
    err2d = np.linalg.norm(lmr[:, srcidx, :2][:, DIRECT]-gt12, axis=-1)/diag[:, None]
    nleg = new[:, list(LEG_TOKENS)].sum(1)
    strat = {"split": split[I], "restored_leg_tokens": np.minimum(nleg[I], 3).astype(str),
             "min_restored_conf": np.where(minconf[I] < .1, "[0,.1)", np.where(minconf[I] < .2, "[.1,.2)", "[.2,.3)")),
             "restored_leg_joint_kind": np.array(["ankle_only" if not new[i, [1, 2, 4, 5]].any() else ("knee_or_hip_present") for i in I]),
             "restored_2d_all_within_0.10_eval_only": np.array(["yes" if np.all(err2d[i][np.asarray(new[i][DIRECT])] <= PCK_RESTORED) else "no" for i in I]),
             "M_rig_fail": np.array(["fail" if (T[i]["M"]["l"]["fail"] or T[i]["M"]["r"]["fail"]) else "ok" for i in I])}
    report["P_leg_strata"] = {k: {str(v): {"rows": int((lab == v).sum()), "delta_rig_dir_ref": video_bootstrap(vid[I][lab == v], dd[lab == v])}
                                  for v in sorted(set(lab))} for k, lab in strat.items()}
    # translation: source leg segment direction change vs rig change
    seg_names = list(SEGMENTS)
    legseg = [seg_names.index(n) for n in ("thigh_l", "shin_l", "thigh_r", "shin_r")]
    ds = sm["R11"]["dir"][I][:, legseg].mean(1)-sm["M"]["dir"][I][:, legseg].mean(1)
    band = TRANSLATION_BAND_DEG
    cat = lambda x: np.where(x < -band, "improve", np.where(x > band, "worse", "same"))  # noqa: E731
    cs, cr = cat(ds), cat(dd)
    report["translation_P_leg"] = {"source_leg_dir_delta": distribution(ds), "rig_dir_ref_delta": distribution(dd),
                                   "pearson": float(np.corrcoef(ds, dd)[0, 1]),
                                   "table_source_x_rig": {f"{a}|{b}": int(((cs == a) & (cr == b)).sum()) for a in ("improve", "same", "worse") for b in ("improve", "same", "worse")},
                                   "band_deg": band, "note": "descriptive only"}
    # IK-only artifacts: rows where the GT-through-rig reference itself departs from GT segments
    refdev = [max(ang(ref[i]["legs"][s]["thigh_w"], G[i][LEG_IDX[s][1]]-G[i][LEG_IDX[s][0]]),
                  ang(ref[i]["legs"][s]["shin_w"], G[i][LEG_IDX[s][2]]-G[i][LEG_IDX[s][1]])) for i in I for s in "lr"]
    report["fixed_ik_reference_departure_P_leg"] = {"GT_through_rig_vs_GT_segment_max_deg": distribution(refdev),
                                                    "legs_over_45": int((np.asarray(refdev) > 45).sum()),
                                                    "note": "the unchanged IK transfers reach fraction/target ratio, so even perfect source does not reproduce GT segment directions"}
    # harm inspection and attribution (all R11 harm cases in P_all)
    report["harm_cases"] = harm_cases(rows, T, sm, new, partial & elig, behind, same, err2d, conf, G, wrows, args, ctx)
    report["attribution_R11_rig_failures_P_leg"] = Counter(c["category"] for c in attribute_rows(
        [i for i in I if T[i]["R11"]["l"]["fail"] or T[i]["R11"]["r"]["fail"]], T, sm, new, behind, same, err2d, "R11"))
    report["attribution_M_rig_failures_P_leg"] = Counter(c["category"] for c in attribute_rows(
        [i for i in I if T[i]["M"]["l"]["fail"] or T[i]["M"]["r"]["fail"]], T, sm, new, behind, same, err2d, "M"))
    # stable-but-wrong vs jitter on P_leg rows with a rig failure
    report["stable_wrong_vs_jitter"] = stable_vs_jitter(I, T, pairs, rig, G)
    # secondary: S0 left arm through the existing chain (NOT the W102 arm policy)
    report["secondary_S0_left_arm"] = left_arm_secondary(arm, G, hm, rc, partial, new, vid)
    # GT-referenced per-window summary + GT-through-rig reference cases (separate file)
    cases = json.loads((bd/"cases.json").read_text())
    report["windows"] = window_summary(cases, T, sm, rows)
    gt_cases(out, cases, G, hm, rc, bank)
    report["runs"] = {r: {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v) for k, v in res.items()}
                      for r, res in runs.items()}
    report["real_clip"] = clip_metrics(bd)
    np.savez_compressed(out/"per_row.npz", rows=I, rig_dir_ref_M=P["_dM"], rig_dir_ref_R11=P["_dR"], source_mpjpe_M=sm["M"]["mpjpe"][I],
                        source_mpjpe_R11=sm["R11"]["mpjpe"][I], video=vid[I], split=split[I])
    dump(out/"analysis_report.json", report)
    print("CASE", report["CASE"], "analysis sha256", sha(out/"analysis_report.json"))


def attribute_rows(I, T, sm, new, behind, same, err2d, cond):
    out = []
    seg_names = list(SEGMENTS)
    for i in I:
        for s in "lr":
            t = T[i][cond][s]
            if not t["fail"]:
                continue
            segs = [seg_names.index(f"thigh_{s}"), seg_names.index(f"shin_{s}")]
            src_ok = bool((sm[cond]["dir"][i][segs] <= DIR_FAIL_DEG).all())
            toks = [k for k in (LEG_IDX[s]) if new[i, k]]
            e = [err2d[i][DIRECT.index(k)] for k in toks]
            if behind[i]: cat = "F_GT_invalid"
            elif not same[i]: cat = "E_identity"
            elif src_ok: cat = "D_source_to_rig_IK"
            elif cond == "R11" and e and max(e) > PCK_RESTORED: cat = "A_raw_2D_localization"
            elif cond == "R11" and e: cat = "B_frozen_S0_response"
            elif cond == "M" and not toks and not src_ok: cat = "B_frozen_S0_response(no restored token on this leg)"
            elif cond == "M": cat = "missing_token_fold(W104 mechanism)"
            else: cat = "G_unresolved"
            out.append({"row": int(i), "leg": s, "category": cat})
    return out


def harm_cases(rows, T, sm, new, sel, behind, same, err2d, conf, G, wrows, args, ctx):
    I = np.flatnonzero(sel)
    seg_names = list(SEGMENTS)
    cases = []
    for i in I:
        t = T[i]
        reasons = []
        if sm["R11"]["severe"][i] and not sm["M"]["severe"][i]: reasons.append("source_newly_severe")
        kw = [(sm["M"]["knee_wrong"][i][k] == 0) and (sm["R11"]["knee_wrong"][i][k] == 1) for k in range(2)]
        if any(kw): reasons.append("source_newly_wrong_knee")
        if any(t["R11"][s]["fail"] and not t["M"][s]["fail"] for s in "lr"): reasons.append("rig_newly_failed")
        if any(np.mean(t["R11"][s]["dir_ref"])-np.mean(t["M"][s]["dir_ref"]) > 15.0 for s in "lr"): reasons.append("rig_dir_ref_delta_over_15")
        if rows[i]["sid"] == "3dpw:outdoors_parcours_00:actor0" and rows[i]["frame"] == 1018: reasons.append("w104_largest_harm")
        if not reasons:
            continue
        toks = np.flatnonzero(new[i][DIRECT])
        x = ctx["obs"]["input_2d_M"][i]
        att = []
        for s in "lr":
            segs = [seg_names.index(f"thigh_{s}"), seg_names.index(f"shin_{s}")]
            src_ok = bool((sm["R11"]["dir"][i][segs] <= DIR_FAIL_DEG).all())
            lt = [k for k in LEG_IDX[s] if new[i, k]]
            e = [err2d[i][DIRECT.index(k)] for k in lt]
            if not (t["R11"][s]["fail"] or np.mean(t["R11"][s]["dir_ref"]) > np.mean(t["M"][s]["dir_ref"])+15 or reasons[0].startswith("source")):
                continue
            if behind[i]: c = "F_GT_invalid"
            elif not same[i]: c = "E_identity"
            elif src_ok and t["R11"][s]["fail"]: c = "D_source_to_rig_IK"
            elif e and max(e) > PCK_RESTORED: c = "A_raw_2D_localization"
            elif e: c = "B_frozen_S0_response"
            else: c = "B_frozen_S0_response(arm-side restoration, indirect leg response)" if not lt and new[i][list(ARM_TOKENS)].any() else "G_unresolved"
            att.append({"leg": s, "category": c, "restored_2d_err_diag": [float(v) for v in e]})
        cases.append({"row": int(i), "sample_id": rows[i]["sid"]+f"#{rows[i]['frame']}", "split": rows[i]["split"], "reasons": reasons,
                      "restored_tokens": [JOINTS[DIRECT[k]] for k in toks],
                      "restored_conf": [float(conf[i][DIRECT[k]]) for k in toks],
                      "restored_2d_err_diag_eval_only": [float(err2d[i][k]) for k in toks],
                      "source_mpjpe": [float(sm["M"]["mpjpe"][i]), float(sm["R11"]["mpjpe"][i])],
                      "rig_dir_ref_mean": [float(np.mean([v for s in "lr" for v in t["M"][s]["dir_ref"]])), float(np.mean([v for s in "lr" for v in t["R11"][s]["dir_ref"]]))],
                      "rig_fail": {c: [t[c][s]["fail"] for s in "lr"] for c in ("M", "R11")},
                      "attribution": att})
    summary = Counter(r for c in cases for r in c["reasons"])
    cat = Counter(a["category"] for c in cases for a in c["attribution"])
    return {"rows": len(cases), "reasons": dict(summary), "attribution": dict(cat), "cases": cases}


def stable_vs_jitter(I, T, pairs, rig, G):
    nb = defaultdict(list)
    for a, b in pairs:
        nb[a].append(b); nb[b].append(a)
    out = {}
    for c in ("M", "R11"):
        failing = [i for i in I if T[i][c]["l"]["fail"] or T[i][c]["r"]["fail"]]
        jitter = stable = 0
        for i in failing:
            steps = []
            for j in nb[i]:
                if j in T and rig[c][j] is not None:
                    for s, (h, k, a) in LEG_IDX.items():
                        for bone in ("thigh", "shin"):
                            steps.append(ang(rig[c][i]["legs"][s][bone+"_w"], rig[c][j]["legs"][s][bone+"_w"]))
            if not steps:
                continue
            if max(steps) > STEP_DEG: jitter += 1
            else: stable += 1
        out[c] = {"failing_rows": len(failing), "with_evaluable_neighbour": jitter+stable,
                  "neighbour_step_over_45 (jitter-like)": jitter, "all_neighbour_steps_le_45 (stable-but-wrong)": stable}
    return out


def left_arm_secondary(arm, G, hm, rc, partial, new, vid):
    sel = partial & new[:, list(LEFT_ARM_TOKENS)].any(1)
    I = [i for i in np.flatnonzero(sel) if arm["M"][i] is not None and arm["R11"][i] is not None]
    ref = gt_left_arm(G, hm, rc, I)
    I = [i for i in I if ref[i] is not None]
    dM = np.array([np.mean([ang(arm["M"][i]["upper_w"], ref[i]["upper_w"]), ang(arm["M"][i]["fore_w"], ref[i]["fore_w"])]) for i in I])
    dR = np.array([np.mean([ang(arm["R11"][i]["upper_w"], ref[i]["upper_w"]), ang(arm["R11"][i]["fore_w"], ref[i]["fore_w"])]) for i in I])
    fM = np.array([max(ang(arm["M"][i]["upper_w"], ref[i]["upper_w"]), ang(arm["M"][i]["fore_w"], ref[i]["fore_w"])) > DIR_FAIL_DEG for i in I])
    fR = np.array([max(ang(arm["R11"][i]["upper_w"], ref[i]["upper_w"]), ang(arm["R11"][i]["fore_w"], ref[i]["fore_w"])) > DIR_FAIL_DEG for i in I])
    V = vid[I] if len(I) else []
    right = int((partial & new[:, [14, 15, 16]].any(1)).sum())
    return {"label": "SECONDARY: S0 left arm through the existing left-arm two-bone chain with the shared HMR thorax; NOT the W102 arm "
                     "policy (frozen J1X) and never in a gate",
            "rows": len(I), "rig_arm_dir_vs_GT_through_chain": {"M_mean": float(dM.mean()) if len(I) else None, "R11_mean": float(dR.mean()) if len(I) else None,
                                                                 "delta": video_bootstrap(V, dR-dM) if len(I) else None},
            "rig_arm_fail_rows": {"M": int(fM.sum()), "R11": int(fR.sum()), "new": int((fR & ~fM).sum()), "recovered": int((fM & ~fR).sum())},
            "right_arm_restored_rows_without_rig_chain": right}


def window_summary(cases, T, sm, rows):
    out = {}
    for name, case in cases["cases"].items():
        R = [f["row"] for f in case["frames"]]
        ev = [i for i in R if i in T]
        out[name] = {"samples": len(R), "rows_with_rig_eval": len(ev), "source_changed_rows": sum(f["source_changed"] for f in case["frames"]),
                     "identity": dict(Counter(rows[i]["identity"] for i in R)),
                     "source_mpjpe_mean": [float(sm["M"]["mpjpe"][R].mean()), float(sm["R11"]["mpjpe"][R].mean())],
                     "rig_fail_rows_among_evaluated": [sum(T[i]["M"]["l"]["fail"] or T[i]["M"]["r"]["fail"] for i in ev),
                                                       sum(T[i]["R11"]["l"]["fail"] or T[i]["R11"]["r"]["fail"] for i in ev)],
                     "rig_dir_ref_mean_evaluated": [float(np.mean([np.mean([v for s in "lr" for v in T[i]["M"][s]["dir_ref"]]) for i in ev])) if ev else None,
                                                    float(np.mean([np.mean([v for s in "lr" for v in T[i]["R11"][s]["dir_ref"]]) for i in ev])) if ev else None]}
    return out


def gt_cases(out, cases, G, hm, rc, bank):
    """Evaluation-only GT-through-rig reference, in a SEPARATE case file (slot RUNTIME), never mixed into candidate exports."""
    from w102_rig_review import arm_export
    gc = {}
    for name, case in cases["cases"].items():
        frames = []
        for f in case["frames"]:
            i = f["row"]; Ro, Rt = hm[i]
            pose = rig_measure(G[i], Ro, Rt, rc) if Ro is not None and Rt is not None else None
            shared = f["modes"].get("FIXED")
            fr = frame_record(bank, f["bank_position"], {"RUNTIME": pose},
                              {"RUNTIME": shared}, f["identity"], {"RUNTIME": "GT legs through the same rig (EVALUATION ONLY)"})
            fr["row"] = i
            frames.append(fr)
        gc[name] = dict(case, frames=frames, labels={"RUNTIME": "GT-through-rig reference (EVALUATION ONLY)"}, short_labels={"RUNTIME": "GT"},
                        title="W105 evaluation-only GT reference (not a candidate)")
    dump(out/"gt_reference_cases.json", {k: cases[k] for k in ("rest_snapshot_digest", "canonical_to_armature_xyzw", "target_chain")} | {"cases": gc})


def clip_metrics(bd):
    vc = json.loads((bd/"video_cases.json").read_text())["cases"]["real_clip"]["frames"]
    vs = np.load(bd/"video_sources.npz")
    valid = vs["input_valid"]
    ch = np.array([f["source_changed"] for f in vc])
    out = {"frames": len(vc), "frames_changed": int(ch.sum()), "GT": "none", "configuration": "GT-free S0-only (W93/W95 A_S0 parent)"}
    from motion.torso_frame import geodesic_degrees
    par = [geodesic_degrees(np.array(f["parent"]["FIXED"]["Rp"]), np.array(f["parent"]["RUNTIME"]["Rp"])) for f in vc if f["source_changed"]]
    out["parent_pelvis_change_deg_changed_frames"] = distribution(par)
    arm_change = [max(ang(f["arm_rig"]["FIXED"]["upper_w"], f["arm_rig"]["RUNTIME"]["upper_w"]), ang(f["arm_rig"]["FIXED"]["fore_w"], f["arm_rig"]["RUNTIME"]["fore_w"]))
                  for f in vc if f["source_changed"] and len(f["arm_rig"]) == 2]
    out["left_arm_rig_bone_change_deg_changed_frames"] = distribution(arm_change)
    leg_change = [max(ang(f["rig"]["FIXED"]["legs"][s][b+"_w"], f["rig"]["RUNTIME"]["legs"][s][b+"_w"]) for s in "lr" for b in ("thigh", "shin"))
                  for f in vc if f["source_changed"] and f["rig"]["FIXED"] and f["rig"]["RUNTIME"]]
    out["leg_rig_bone_change_deg_changed_frames"] = distribution(leg_change)
    el = {s: np.array([f["arm_rig"][s]["elbow"] if s in f["arm_rig"] else np.nan for f in vc]) for s in ("FIXED", "RUNTIME")}
    out["left_elbow_flexion_rig"] = {"complete_frames_M": distribution(el["FIXED"][~ch]), "changed_frames_M": distribution(el["FIXED"][ch]),
                                     "changed_frames_R11": distribution(el["RUNTIME"][ch])}
    steps = {}
    trans = np.array([bool(valid[i, 13] != valid[i-1, 13]) for i in range(1, len(vc))])
    for s in ("FIXED", "RUNTIME"):
        st = []
        for i in range(1, len(vc)):
            a, b = vc[i-1]["arm_rig"].get(s), vc[i]["arm_rig"].get(s)
            st.append(max(ang(a["upper_w"], b["upper_w"]), ang(a["fore_w"], b["fore_w"])) if a and b else np.nan)
        st = np.array(st)
        ls = []
        for i in range(1, len(vc)):
            a, b = vc[i-1]["rig"].get(s), vc[i]["rig"].get(s)
            ls.append(max(ang(a["legs"][x][y+"_w"], b["legs"][x][y+"_w"]) for x in "lr" for y in ("thigh", "shin")) if a and b else np.nan)
        ls = np.array(ls)
        steps[s] = {"left_arm_step_deg": distribution(st), "left_arm_steps_over_45": int(np.nansum(st > STEP_DEG)),
                    "left_arm_steps_over_45_at_wrist_mask_transitions": int(np.nansum(st[trans] > STEP_DEG)),
                    "left_arm_step_at_transitions": distribution(st[trans]), "left_arm_step_elsewhere": distribution(st[~trans]),
                    "leg_steps_over_45": int(np.nansum(ls > STEP_DEG))}
    out["temporal_50fps"] = {"wrist_mask_transitions": int(trans.sum()), "M": steps["FIXED"], "R11": steps["RUNTIME"],
                             "note": "GT-free; consecutive original 50 fps frames; no GT-stability condition is available"}
    viol = {s: sum(bool(f["rig"][s] and any(f["rig"][s]["viol"].get(k, False) for k in LEG_ROM)) for f in vc) for s in ("FIXED", "RUNTIME")}
    out["leg_rom_violation_frames"] = viol
    return out


# ------------------------------------------------------------------ stage: compose
def cmd_compose(args):
    import cv2
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    data = json.loads(Path(args.cases).read_text())
    gt = json.loads(Path(args.gt_cases).read_text())["cases"] if args.gt_cases else {}
    clip = None
    if args.clip:
        from w102_video_diagnostic import decode
        clip = decode(ROOT/CLIP)[0]
    report = {}
    for name, case in data["cases"].items():
        cols = 3 + (1 if name in gt else 0)
        W, H = 400*cols, 900
        fps_out = 50. if clip is not None else 30.
        writer = cv2.VideoWriter(str(out/(name+".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (W, H))
        if not writer.isOpened():
            raise RuntimeError("STOP: review encoder unavailable")
        written = 0
        for i, fr in enumerate(case["frames"]):
            canvas = np.full((H, W, 3), 22, np.uint8)
            im = clip.frames[fr["frame_index"]].image if clip is not None else cv2.imread(str(Path(args.image_root)/fr["image"]))
            canvas[40:440, :400] = cv2.resize(im, (400, 400))
            lines = [name, case["sequence_id"], f"frame {fr['frame_index']} | {fr['time_s']:.3f}s", f"association: {fr['identity']}",
                     "restored: " + (",".join(t.replace("left_", "L ").replace("right_", "R ") for t in fr.get("restored_tokens", [])) or "none"),
                     "only S0 source differs (M vs R11)", case["header_note"][:60], "sparse 10 Hz cadence" if clip is None else "original 50 fps"]
            for n, line in enumerate(lines):
                cv2.putText(canvas, line, (8, 470+28*n), cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 230, 230), 1, cv2.LINE_AA)
            panels = [("FIXED", args.render_dir, case["labels"]["FIXED"], (240, 230, 200)), ("RUNTIME", args.render_dir, case["labels"]["RUNTIME"], (240, 200, 240))]
            if name in gt:
                panels.append(("RUNTIME", args.gt_render_dir, "GT through rig: EVALUATION ONLY", (120, 240, 120)))
            for col, (slot, rdir, label, color) in enumerate(panels, 1):
                cv2.putText(canvas, label[:44], (col*400+6, 26), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
                frames_src = gt[name]["frames"][i] if rdir == args.gt_render_dir else fr
                if slot in frames_src["object_rotation_xyzw"]:
                    for y, view in ((40, "video"), (460, "side")):
                        panel = cv2.imread(str(Path(rdir)/name/f"{slot}_{view}_{i:03d}.png"))
                        if panel is None:
                            raise RuntimeError(f"STOP: missing render {name} {slot} {view} {i}")
                        canvas[y:y+400, col*400:(col+1)*400] = cv2.resize(panel, (400, 400))
                else:
                    cv2.putText(canvas, "UNAVAILABLE - no fallback", (col*400+8, 240), cv2.FONT_HERSHEY_SIMPLEX, .55, (50, 80, 250), 1)
            nxt = case["frames"][i+1]["time_s"] if i+1 < len(case["frames"]) else fr["time_s"]+1/case["fps_sampled"]
            for _ in range(max(1, int(round((nxt-fr["time_s"])*fps_out)))):
                writer.write(canvas); written += 1
            if fr.get("source_changed") or i in (0, len(case["frames"])//2, len(case["frames"])-1):
                if clip is None or i % 5 == 0 or i in (0, len(case["frames"])-1):
                    cv2.imwrite(str(out/f"{name}_{i:03d}.jpg"), canvas)
        writer.release()
        report[name] = {"video_frames_written": written, "fps": fps_out, "samples": len(case["frames"])}
    dump(out/"compose_report.json", report)


# ------------------------------------------------------------------ stage: manifest
def cmd_manifest(args):
    base = Path(args.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != "evidence_manifest.json" and p.suffix != ".png":
            files[p.relative_to(base).as_posix()] = {"sha256": sha(p), "bytes": p.stat().st_size}
    pngs = sorted(p for p in base.rglob("*.png"))
    h = hashlib.sha256()
    for p in pngs:
        h.update(p.relative_to(base).as_posix().encode()); h.update(bytes.fromhex(sha(p)))
    dump(base/"evidence_manifest.json", {"schema": "w105_evidence_manifest_v1", "starting_head": args.starting_head,
                                         "declaration_sha256": args.declaration_sha, "files": files,
                                         "render_png": {"count": len(pngs), "combined_sha256": h.hexdigest()}})
    print("manifest sha256", sha(base/"evidence_manifest.json"), len(files), "files", len(pngs), "png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("verify", "declare", "build", "analyze", "compose", "manifest"))
    for name in ("w102", "w101_data", "w103", "w104", "out_dir", "declaration", "declaration_sha", "build_dir", "cases", "gt_cases",
                 "render_dir", "gt_render_dir", "image_root", "base", "starting_head", "render_report", "audit_report"):
        ap.add_argument("--"+name.replace("_", "-"), default=None)
    ap.add_argument("--clip", action="store_true")
    a = ap.parse_args()
    {"verify": cmd_verify, "declare": cmd_declare, "build": cmd_build, "analyze": cmd_analyze, "compose": cmd_compose,
     "manifest": cmd_manifest}[a.stage](a)


if __name__ == "__main__":
    main()
