#!/usr/bin/env python3
"""W113: Animation Base information-preservation & selective-edit contract feasibility (evidence only).

Stages (each refuses an existing output directory):
  preflight  instruction/HEAD/W112 marker/lock/input authentication, runtime versions
  declare    frozen declaration: inputs + hashes, ownership matrix, minimum fields, directives, controls, CASE logic
  build      scratch view join (synthetic / real / benchmark), contract controls, real coverage, existing-path
             replays (MotionGraph -> AnimationClip, FrameBank -> Lifted -> semantics pieces), rig association
  compare    repeat comparison of two independent builds
  figures    timeline / trajectory review (synthetic perturbation labelled as such)
  decide     frozen CASE logic over the recorded decision inputs
  manifest   evidence manifest

NON-PRODUCTION. No src/ module, schema, model, threshold, rig, FBX/Blender exporter or default changes. No GT enters
any view; the real clip is GT-free. No inference is run: every 2D/3D array is a stored, hash-verified artefact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w113_animation_base_view as V  # noqa: E402

BATCH = 113
INSTRUCTION_BLOB = "ce299256622abb2ece5cbbd7196b2d6b8c59645f"
STARTING_HEAD = "011d79112ade873cc2097d694c04cdfe6c0031ce"
W112_WORKLOG = "docs/112_WORKLOG_FBX_WITHIN_PROPERTY_INTERPOLATION_BOUNDARY.md"
W102 = Path("C:/Users/dna10/animcv-w102")
W105 = Path("C:/Users/dna10/animcv-w105")
W101 = Path("C:/Users/dna10/animcv-w101")
LOCK = Path("C:/Users/dna10/animcv-automation/state/active_batch.json")
# Frozen identities recorded by W102/W105 (their Worklogs and manifests).
INPUTS = {
    "w102_observations_predictions": (W102 / "results/historical_video/observations_predictions.npz",
                                      "6784357f7c6145e4b60e2e9201e32accd3ae00001486283a1e76f0adae61e7bd"),
    "w102_video_report": (W102 / "results/historical_video/report.json",
                          "e6e643071aadc1def75d86697acaeed56a0db6014c6620b42908a6359408b4fa"),
    "w102_video_inspection": (W102 / "results/video_inspection/inspection.json",
                              "dd25b7af0183754b7e78d07df9df71c7f718e43fb3f7489545d9c8b5c10d3149"),
    "w102_historical_source_clip": (W102 / "data/historical_source.mp4",
                                    "0dda8a5f16c67992330a1a8534cf0bfefeb5e0f2aec438577a9e1dbace773275"),
    "repo_production_demo_clip": (ROOT / "examples/production_demo/source.mp4",
                                  "0dda8a5f16c67992330a1a8534cf0bfefeb5e0f2aec438577a9e1dbace773275"),
    "w105_evidence_manifest": (W105 / "evidence_manifest.json",
                               "cb0eca93c7847a70d497bd36c57856bd8295f3d8e3a7a72913ebf876ebe96899"),
    "w105_video_sources": (W105 / "results/build/video_sources.npz", None),
    "w105_video_cases": (W105 / "results/build/video_cases.json", None),
    "w105_render_clip_audit": (W105 / "results/render_clip_audit.json", None),
    "s0_checkpoint": (W101 / "data/s0_checkpoint.pt", None),
    "bench_bank_index": (W101 / "data/bank_3dpw_paired_v2.json",
                         "6fe13cdc91e1fdec8defd647b230cf688fa514a2799956ed40e7f805b13e0fe5"),
    "bench_bank_arrays": (W101 / "data/bank_3dpw_paired_v2.npz",
                          "909090f7aa78f3c25f56fbd335eb5b2521f87c06e92d59a28010b0b28ba6b79e"),
    "bench_frozen_prediction_validation": (W101 / "data/frozen_prediction_validation.npy",
                                           "402d70a2bed79ae31919ff86905e19c8972108ea988d6cfa5deac514562132b9"),
    "legacy_mapping": (ROOT / "examples/production_demo/mapping.json", None),
    "legacy_rig_fbx": (ROOT / "examples/BaseRig.fbx", None),
    "w105_rig_skinned_fbx": (ROOT / "examples/skinned_mannequin/BaseRig_Skinned.fbx", None),
    "w105_rig_rest": (ROOT / "examples/e2e_demo/baserig_blender_rest.json", None),
}
BENCH_BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
REAL_SEQUENCE_ID = "w102_historical_video:0dda8a5f16c67992"
REAL_ACTOR_ID = "single_tracked_subject"
REAL_ACTOR_STATUS = ("single_subject_contract (Architecture_v2 1.3); per-frame SubjectTracker selection produced "
                     "inside mmpose_adapter._process_tracked_sequence but never serialized")
RIGHT_ARM = ["right_shoulder", "right_elbow", "right_wrist"]
TORSO = ["pelvis", "left_hip", "right_hip", "thorax", "left_shoulder", "right_shoulder"]
# Which S0 joints each W105 S0-only real-clip Action channel consumes (scripts/w105_rig_utility.py build_clip,
# motion/torso_frame.torso_frames, w102_rig_review.arm_export): own joints + the torso parent frame.
W105_BONE_DEPENDENCIES = {
    "thigh_l": ["left_hip", "left_knee"] + TORSO, "calf_l": ["left_hip", "left_knee", "left_ankle"] + TORSO,
    "thigh_r": ["right_hip", "right_knee"] + TORSO, "calf_r": ["right_hip", "right_knee", "right_ankle"] + TORSO,
    "upperarm_l": ["left_shoulder", "left_elbow"] + TORSO,
    "lowerarm_l": ["left_shoulder", "left_elbow", "left_wrist"] + TORSO,
    "spine_01": TORSO, "__object__": TORSO,
}


def sha(p):
    return V.sha_file(p)


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=1, sort_keys=True, allow_nan=False))


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    return v


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def new_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=False)
    return p


# ---------------------------------------------------------------------------------------------- preflight ---

def cmd_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"schema": "w113_preflight_v1", "stop": []}
    rep["head"] = git("rev-parse", "HEAD")
    rep["instruction_blob"] = git("rev-parse", "HEAD:docs/next_batch.md")
    rep["branch"] = git("rev-parse", "--abbrev-ref", "HEAD")
    rep["status_porcelain"] = git("status", "--porcelain")
    first = (ROOT / "docs/next_batch.md").read_text(encoding="utf-8").splitlines()[0]
    rep["instruction_title"] = first
    wl = (ROOT / W112_WORKLOG).read_text(encoding="utf-8").rstrip().splitlines()
    rep["w112_marker"] = wl[-2:] == ["AUTOMATION_BATCH_STATUS: COMPLETE", "BATCH_NUMBER: 112"]
    if rep["instruction_blob"] != INSTRUCTION_BLOB or not first.startswith("# BATCH 113"):
        rep["stop"].append("instruction is not the bound Batch 113 blob")
    if rep["branch"] != "arch/single_frame_first":
        rep["stop"].append("wrong branch")
    if not rep["w112_marker"]:
        rep["stop"].append("W112 Worklog lacks its COMPLETE marker")
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else None
    rep["main_local_lock"] = lock
    if not lock or lock.get("batch") != BATCH or lock.get("instruction_blob") != INSTRUCTION_BLOB \
            or lock.get("start_commit") != STARTING_HEAD:
        rep["stop"].append("Main-local lock is not this Batch 113 claim")
    rep["lock_scope_note"] = "Main-local O_EXCL lock only; it does not prevent executors on other hosts"
    files = {}
    for name, (path, want) in INPUTS.items():
        got = sha(path) if Path(path).exists() else None
        files[name] = {"path": str(path), "sha256": got, "expected": want, "bytes": Path(path).stat().st_size if got else None,
                       "ok": got is not None and (want is None or got == want)}
        if not files[name]["ok"]:
            rep["stop"].append(f"input {name} missing or changed")
    m102 = json.loads((W102 / "evidence_manifest.json").read_text())["files"]
    m105 = json.loads((W105 / "evidence_manifest.json").read_text())["files"]
    cross = {
        "w102_manifest_npz": m102["results/historical_video/observations_predictions.npz"]["sha256"]
        == files["w102_observations_predictions"]["sha256"],
        "w102_manifest_report": m102["results/historical_video/report.json"]["sha256"] == files["w102_video_report"]["sha256"],
        "w102_report_binds_npz": json.loads(INPUTS["w102_video_report"][0].read_text())["observation_prediction_sha256"]
        == files["w102_observations_predictions"]["sha256"],
        "w105_manifest_video_sources": m105.get("results/build/video_sources.npz", {}).get("sha256")
        == files["w105_video_sources"]["sha256"],
        "w105_manifest_video_cases": m105.get("results/build/video_cases.json", {}).get("sha256")
        == files["w105_video_cases"]["sha256"],
        "w105_manifest_render_clip_audit": m105.get("results/render_clip_audit.json", {}).get("sha256")
        == files["w105_render_clip_audit"]["sha256"],
    }
    from run_real_observation_s0 import S0_CHECKPOINT_SHA
    cross["s0_checkpoint_identity"] = files["s0_checkpoint"]["sha256"] == S0_CHECKPOINT_SHA
    rep["cross_checks"] = cross
    for k, ok in cross.items():
        if not ok:
            rep["stop"].append(f"cross-check failed: {k}")
    rep["inputs"] = files
    rep["runtime"] = runtime()
    try:
        raw = subprocess.run(["tasklist", "/FI", "IMAGENAME eq claude.exe"], capture_output=True).stdout or b""
        rep["processes"] = raw.decode("utf-8", errors="replace").strip().splitlines()[3:]
    except OSError:
        rep["processes"] = None
    rep["pass"] = not rep["stop"]
    dump(out / "preflight.json", rep)
    print(json.dumps({"pass": rep["pass"], "stop": rep["stop"]}))
    return 0 if rep["pass"] else 2


def runtime():
    import numpy
    r = {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__}
    for mod in ("matplotlib", "torch"):
        try:
            r[mod] = __import__(mod).__version__
        except Exception as e:  # pragma: no cover
            r[mod] = f"unavailable: {e}"
    return r


# ----------------------------------------------------------------------------------------------- declaration ---

OWNERSHIP = [
    # (quantity, boundary, status, code location, note)
    ("obs2d_xy_conf_visible", "mmpose_adapter -> PoseSequence", "PRODUCED", "src/pose/mmpose_adapter.py:280-339",
     "x,y px, score, visible=score>=0.3; neck/pelvis/spine are midpoints, head=nose; no derived flag"),
    ("obs2d_xy_conf_visible", "PoseSequence -> MotionGraph", "PASSED", "src/motion/motion_builder.py:27-40",
     "position_2d, confidence, visible copied; derived-vs-observed still implicit in name only"),
    ("obs2d_xy_conf_visible", "MotionGraph -> AnimationClip", "DROPPED", "src/retarget/fk_solver.py:108-158",
     "only min(conf_a,conf_b) per bone-frame survives; 0.0 = held rotation for missing/invisible input"),
    ("obs2d_xy_conf_valid", "real observation -> FramePose input (scripts)", "PASSED/AMBIGUOUS",
     "scripts/w101_reference_estimator.py:60-75", "invalid joints zero-filled (xy=0, conf=0) in input_2d; raw "
     "landmarks retained only in the W102 npz"),
    ("obs2d_conf", "FrameBank -> LiftedPoseSequence", "AMBIGUOUS", "src/pose/framepose_bridge.py:101-110",
     "confidence=input_2d[...,2]; depth_uncertainty=1-confidence is a 2D proxy, not 3D reliability"),
    ("obs2d_xy_conf", "Lifted -> AnimationSemantics v1/v2", "DROPPED", "src/motion/animation_semantics.py:218-243",
     "only joint_observation_valid + joint_in_frame survive"),
    ("in_frame", "FrameBank -> AnimationSemantics", "PRODUCED", "src/framepose/nonlinear_reconstruction.py:77-82",
     "structural in-frame mask, requires input_valid"),
    ("est3d_xyz", "frozen S0 -> Lifted -> AnimationSemantics", "PASSED", "src/motion/animation_semantics_v2_bridge.py:101-121",
     "camera_root_relative metres articulation"),
    ("est3d_xyz", "-> MotionGraph", "UNAVAILABLE", "src/motion/motion_builder.py:33-37",
     "MotionGraph.position_3d is (image_x,image_y,relative_depth) mixed units or None; FramePose 3D never enters it"),
    ("est3d_xyz", "-> AnimationClip (CLI path)", "UNAVAILABLE", "src/app/cli.py:951-996",
     "CLI retarget consumes MotionGraph only; FramePose->rig exists only as library code used by scripts"),
    ("est3d_xyz", "-> W105 S0-only rig Action (scripts)", "PASSED (transformed)", "scripts/w105_rig_utility.py:657-720",
     "rotations only; no per-joint 3D, no validity"),
    ("est3d_confidence", "all boundaries", "UNAVAILABLE", "src/pose/framepose_bridge.py:107",
     "no calibrated 3D reliability producer"),
    ("sequence/sample/frame identity", "FrameSample (bank)", "PRODUCED", "src/framepose/contract.py:105-160",
     "sample_id, sequence_id(actor), frame_index, timestamp, fps, image_reference, observation provenance"),
    ("sequence/actor identity", "PoseSequence/MotionGraph", "DROPPED/AMBIGUOUS", "src/motion/motion_graph.py:96-121",
     "only source_metadata={'pose_source': path}; no sequence/actor id"),
    ("actor/track per frame", "mmpose_adapter", "PRODUCED-internal, not serialized", "src/pose/mmpose_adapter.py:189-232",
     "SubjectTracker selection per frame aggregated into last_tracking_report only"),
    ("identity + source", "AnimationClip", "DROPPED", "src/retarget/solver.py:66-96",
     "name='Generated_Motion', fps, frame_start/end; no source digest"),
    ("frame_index", "MotionGraph -> AnimationClip", "PASSED", "src/retarget/solver.py:16-46", "per sample"),
    ("timestamp", "MotionGraph -> AnimationClip", "DROPPED (fps-derivable)", "src/retarget/solver.py:66-96",
     "AnimationClip has fps + frame indices, no timestamps"),
    ("timestamp/fps", "W105 Action (Blender)", "AMBIGUOUS", "docs/105 section 10",
     "keys at frame_index+1 but scene fps 30 != source 50; source time only in sidecar"),
    ("coordinate frame/units", "FrameBank/Lifted/Semantics", "PRODUCED/PASSED", "src/framepose/contract.py:47-50",
     "camera_root_relative metres; COORDINATE_CONVENTIONS frames"),
    ("coordinate frame/units", "MotionGraph", "AMBIGUOUS", "COORDINATE_CONVENTIONS.md 'Existing data compatibility'",
     "position_3d mixed units"),
    ("root_yaw", "AnimationSemantics v1/v2", "PRODUCED (derived)", "src/motion/animation_semantics_v2_bridge.py:89-98",
     "camera-relative yaw, KNOWN/UNKNOWN"),
    ("contact-like", "AnimationSemantics v1/v2", "PRODUCED (calibration-bound)", "src/motion/animation_semantics_v2_bridge.py:24-60",
     "requires an H0 TRAIN calibration; no real_animcv 50 fps calibration exists"),
    ("root_translation/ground_height", "AnimationSemantics", "UNAVAILABLE", "src/motion/animation_semantics.py:110-150",
     "explicit, load refuses numeric values"),
    ("MotionFrame.locked", "MotionGraph", "PRODUCED (always False)", "src/motion/motion_builder.py:42-50",
     "frame-wide; no producer ever sets True; temporal_filter copies it"),
    ("MotionFrame.locked", "-> AnimationClip/optimize", "DROPPED", "src/app/cli.py:999-1008",
     "collapse_animation_clip(locked_frames=...) exists but the CLI never passes it; semantics = key retention"),
    ("BoneMappingEntry.locked", "mapping", "PRODUCED, never consumed", "src/rig/bone_mapping.py:25-47", "no consumer"),
    ("per-joint user lock / edit permission / edit provenance", "all", "UNAVAILABLE", "-", "no producer or consumer"),
    ("AnimationClip confidence", "AnimationClip -> Blender", "DROPPED", "src/blender/", "not written to Actions"),
]

MINIMUM_FIELDS = {
    "M1_obs2d": "per joint-frame 2D xy (px) + detector confidence + present/valid status",
    "M2_obs2d_kind": "observed vs derived/renamed 2D joint distinction",
    "M3_est3d": "3D xyz with coordinate frame, unit and finite status",
    "M4_observed_vs_inferred": "per cell observed-2D vs inferred-3D (input-supported or not) vs unknown",
    "M5a_sequence_frame_time": "sequence identity, frame_index, timestamp, fps",
    "M5b_actor_track": "actor identity and per-frame track continuity evidence",
    "M6_source_provenance": "source hashes of clip/frames, observation artefact, estimator identity",
    "M7_user_directive": "per joint-frame preserve/permit/prohibit state (synthetic directive; no production producer)",
    "M8_unavailable": "root translation, ground height, physical contact, calibrated 3D confidence, GT error: UNAVAILABLE",
}

CASE_LOGIC = {
    "precedence": "D > B > A > C",
    "D": ["preflight stop", "real input authentication failure", "GT leakage into any candidate view",
          "mixed-unit metric reinterpretation accepted", "any declared positive or negative control not behaving as "
          "declared", "repeat mismatch of digests/status/masks/control outcomes", "src/ or input mutation",
          "competing executor in the Main-local lock", "view not auditable (cannot serialize/round-trip)"],
    "B": "a minimum field M1-M6 present in an upstream SERIALIZED artefact of the traced path is lost/ambiguous in a "
         "downstream serialized artefact AND the scratch association cannot restore it identity-safely from frozen "
         "inputs (internal unserialized variables are a missing producer -> C, not B)",
    "A": "valid real+synthetic runs; every minimum field with a producer is available or identity-safely joinable; "
         "all directive/edit/unknown tests pass non-vacuously (real permitted set non-empty); negative controls "
         "discriminate; rig association demonstrated on the existing W105 Action replay and the legacy AnimationClip; "
         "unproduced quantities classified UNAVAILABLE; no minimum field left without a producer except M7 "
         "(synthetic by design) and M8 (UNAVAILABLE by design); rig-boundary permission semantics unambiguous",
    "C": "validity gates pass but A not met and no demonstrated irrecoverable B (e.g. M5b without a serialized "
         "producer, rig permission semantics ambiguous, a rig boundary untested, empty permitted set)",
}


def real_directive_rule():
    return {
        "RS1": {"preserve": [{"id": "P_ref_span", "joints": "ALL", "spans": [[0, 99]]},
                             {"id": "P_right_arm", "joints": RIGHT_ARM, "spans": "ALL"}],
                "permit": [{"id": "E_low_evidence_left_wrist", "joints": ["left_wrist"],
                            "rule": "maximal frame runs where input_valid(left_wrist)==False, frame_index >= 100"}],
                "default": "prohibit", "expected": "ACCEPT"},
        "RS2_conflict": {"base": "RS1", "add_permit": {"id": "E_conflict", "joints": ["right_knee"], "spans": [[90, 110]]},
                         "expected": "LOCK_CONFLICT"},
        "edits_on_RS1": {"E_ok": "all RS1 permitted cells, delta (0,0,+0.01) m -> ACCEPT + exact locality",
                         "E_lock": "(50, left_wrist) -> LOCK_CONFLICT_EDIT",
                         "E_unauth": "(150, left_knee) -> EDIT_NOT_AUTHORIZED",
                         "E_obs": "signal obs2d_xy_px -> OBSERVATION_IMMUTABLE",
                         "E_stale": "bound to a different digest -> DIRECTIVE_STALE_VIEW"},
        "vacuity": "if RS1 permit resolves to zero cells, positive real edit is vacuous -> not A",
    }


DIRECTIVE_CONTROLS = {
    "directive_actor_swapped": "IDENTITY_MISMATCH", "directive_sequence_swapped": "IDENTITY_MISMATCH",
    "directive_stale_view_digest": "DIRECTIVE_STALE_VIEW", "directive_on_observation": "OBSERVATION_IMMUTABLE",
    "directive_unknown_joint": "JOINT_UNKNOWN", "directive_span_out_of_range": "FRAME_OUT_OF_RANGE",
    "lock_conflict_set": "LOCK_CONFLICT", "edit_on_locked_cell": "LOCK_CONFLICT_EDIT",
    "edit_on_prohibited_cell": "EDIT_NOT_AUTHORIZED", "edit_on_observation": "OBSERVATION_IMMUTABLE",
    "edit_stale_view_digest": "DIRECTIVE_STALE_VIEW",
    "edit_on_unknown_cell (synthetic only)": "EDIT_UNKNOWN_TARGET",
    "mixed_unit_motion_graph_position_3d": "MIXED_UNIT_3D",
}


def cmd_declare(a):
    out = new_dir(a.out_dir)
    pre = json.loads(Path(a.preflight).read_text())
    if not pre["pass"]:
        raise RuntimeError("STOP: preflight did not pass")
    decl = {
        "schema": "w113_declaration_v1", "outcomes_inspected": False,
        "starting_head": STARTING_HEAD, "instruction_blob": INSTRUCTION_BLOB,
        "inputs": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in pre["inputs"].items()},
        "regimes": {"real": "real_animcv_observation (W102 M1 MMPose+RTMDet on examples/production_demo/source.mp4; "
                            "frozen S0 predictions; GT-free; private; not unseen)",
                    "benchmark": "benchmark_detector_observation (3DPW shipped detector, bank 75519e63, frozen S0 "
                                 "VALIDATION prediction); optional, never pooled",
                    "synthetic": "oracle_geometry/synthetic_projection tiny known 2D+3D sequence; contract test only"},
        "frame_selection": {"real": "all 199 decoded frames, original order, frame_index 0..198, timestamps i/50",
                            "benchmark": "first VALIDATION sequence in bank order, all its rows",
                            "synthetic": "8 frames at 25 fps"},
        "joint_order": list(V.JOINTS), "obs2d_kind": V.OBS2D_KIND,
        "coordinate_transforms": {"obs2d": "raw landmark px (image I), no transform; normalized only for in_frame",
                                  "est3d": "stored S0 output, camera_root_relative metres, no transform",
                                  "root_yaw": "current_root_orientation on build_h0_lifted_sequence, camera frame"},
        "gt_policy": "no target_3d/target_valid/strata/SMPL/evaluation label in any view; scratch FrameBank target "
                     "arrays are NaN/False placeholders required by the existing class; no evaluation-only reference "
                     "is examined in W113",
        "ownership_matrix": [dict(zip(("quantity", "boundary", "status", "code", "note"), r)) for r in OWNERSHIP],
        "minimum_fields": MINIMUM_FIELDS,
        "view_negative_controls": {k: c for k, (_, c) in V.VIEW_CONTROLS.items()},
        "directive_controls": DIRECTIVE_CONTROLS,
        "unknown_not_reject": {"absent landmark": "obs2d UNKNOWN (NaN, present=False)",
                               "finite S0 on invalid input": "inferred_3d_input_unsupported (kept, labelled)",
                               "root yaw not computable": "UNKNOWN (NaN, known=False)",
                               "contact-like without regime-matched calibration": "UNKNOWN (not computed)"},
        "unavailable": V.UNAVAILABLE,
        "real_directives": real_directive_rule(),
        "synthetic_directives": "preserve right arm all frames + all joints frames 0-1; permit left_wrist 3-5 and "
                                "left_elbow 4 (NaN est3d); edits: ok=left_wrist 3-5; lock=right_wrist 4; "
                                "unauth=left_knee 4; unknown=left_elbow 4; conflict set adds permit left_wrist 0-5",
        "rig": {"w105_replay": "unchanged scripts/w105_rig_blender.py render+audit of W105 video_cases.json into "
                               "scratch; Action value_digest must equal W105's; keys must be frame_index+1",
                "bone_dependencies": W105_BONE_DEPENDENCIES,
                "bone_permission_rule": "locked if any dependency cell preserved; editable if all dependency cells "
                                        "permitted; partial if some permitted; else prohibited",
                "legacy": "unchanged MotionGraphBuilder + RetargetSolver(validate_quality=False) on a PoseSequence "
                          "reconstructed losslessly from the W102 landmarks; quality report recorded"},
        "counterexample_definitions": {
            "structurally_valid_possibly_wrong_3d": "S0 limb direction step > 45 deg between consecutive frames "
                                                    "whose limb joints are all input_valid (GT-free indicator only)",
            "low_confidence_useful_2d": "not establishable GT-free; report raw windows descriptively"},
        "failure_windows": {"side_on_wrist_jump": [109, 118], "walking_away_wrist_occluded": [154, 181]},
        "repeat": "two independent builds from unchanged inputs; compare view digests, status arrays, masks, "
                  "control outcomes, coverage, legacy/semantic serialization digests",
        "case_logic": CASE_LOGIC,
    }
    dump(out / "declaration.json", decl)
    print(sha(out / "declaration.json"))
    return 0


# ----------------------------------------------------------------------------------------------------- build ---

def load_declaration(a):
    d = Path(a.declaration)
    if sha(d) != a.declaration_sha:
        raise RuntimeError("STOP: declaration hash mismatch")
    decl = json.loads(d.read_text())
    for name, rec in decl["inputs"].items():
        if sha(rec["path"]) != rec["sha256"]:
            raise RuntimeError(f"STOP: input {name} changed since declaration")
    return decl


def source(decl, *names):
    return {n: {"path": decl["inputs"][n]["path"], "sha256": decl["inputs"][n]["sha256"]} for n in names}


def synthetic_view(tmp: Path):
    """Tiny known 2D+3D sequence (oracle-style synthetic projection). Contract test only, not animation."""
    T, fps = 8, 25.0
    rest = {"pelvis": (0, 0, 0), "left_hip": (-.1, 0, 0), "left_knee": (-.1, 0, -.45), "left_ankle": (-.1, 0, -.9),
            "right_hip": (.1, 0, 0), "right_knee": (.1, 0, -.45), "right_ankle": (.1, 0, -.9), "spine": (0, 0, .25),
            "thorax": (0, 0, .5), "neck": (0, 0, .55), "head": (0, 0, .7), "left_shoulder": (-.18, 0, .5),
            "left_elbow": (-.2, 0, .25), "left_wrist": (-.2, 0, 0), "right_shoulder": (.18, 0, .5),
            "right_elbow": (.2, 0, .25), "right_wrist": (.2, 0, 0)}
    base = np.array([rest[n] for n in V.JOINTS], np.float64)
    est = np.repeat(base[None], T, 0)
    for t in range(T):
        est[t, V.JI["left_wrist"], 1] = 0.2 * math.sin(t / 2)
        est[t, V.JI["right_wrist"], 1] = -0.2 * math.sin(t / 2)
    W, H, f, depth = 1280, 720, 1000.0, 3.0
    xy = np.stack([f * est[..., 0] / (est[..., 1] + depth) + W / 2, H / 2 - f * est[..., 2] / (est[..., 1] + depth)], -1)
    conf = np.full((T, V.J), 0.9)
    present = np.ones((T, V.J), bool)
    conf[3:6, V.JI["left_wrist"]] = 0.2
    present[6, V.JI["left_ankle"]] = False
    xy[~present] = np.nan
    conf[~present] = np.nan
    valid = present & (conf >= 0.3)
    est = est.astype(np.float32)
    est[4, V.JI["left_elbow"]] = np.nan
    np.savez(tmp / "synthetic_source.npz", xy=xy, conf=conf, est=est)
    yaw = [0.0 if valid[t, [V.JI["left_shoulder"], V.JI["right_shoulder"]]].all() else None for t in range(T)]
    yaw[5] = None
    return V.build_view(
        sequence_id="synthetic:w113_contract_sequence", actor_id="synthetic_actor_0",
        actor_identity_status="synthetic_known", regime="oracle_geometry", backend="synthetic_projection",
        image_size=(W, H), fps=fps, fps_source="synthetic declaration", frame_index=np.arange(T),
        timestamp=np.arange(T) / fps, obs2d_xy_px=xy, obs2d_conf=conf, obs2d_present=present, obs2d_visible=valid,
        input_valid=valid, est3d_xyz=est, root_yaw=yaw,
        sources={"synthetic_source": {"path": str(tmp / "synthetic_source.npz"), "sha256": sha(tmp / "synthetic_source.npz")}},
        producers={"est3d": "synthetic known 3D", "root_yaw": "synthetic", "contact_status": "not computed"},
        notes={"label": "SYNTHETIC CONTRACT SEQUENCE, not animation evidence"})


def scratch_real_bank(z, size, fps):
    """In-memory FrameBank for the existing FramePose bridge. Placeholder NaN/False target arrays: no GT exists."""
    from framepose.contract import FrameBank, FrameSample, Modality, make_sample_id
    from framepose.observations import ObservationProvenance
    prov = ObservationProvenance("mmpose", "w102_m1_rtmpose_rtmdet", "real_animcv_observation",
                                 {"w102_npz_sha256": INPUTS["w102_observations_predictions"][1]})
    samples = [FrameSample(make_sample_id(REAL_SEQUENCE_ID, int(i)), "w102_historical_video", REAL_SEQUENCE_ID, int(i),
                           "test", tuple(size), Modality(True, False, True, False), prov, float(t), fps)
               for i, t in zip(z["frame_index"], z["timestamp"])]
    T = len(samples)
    return FrameBank(samples, {"input_2d": z["input_2d"], "input_valid": z["input_valid"],
                               "target_3d": np.full((T, 17, 3), np.nan, np.float32), "target_valid": np.zeros((T, 17), bool)})


def real_inputs(decl):
    z = dict(np.load(decl["inputs"]["w102_observations_predictions"]["path"], allow_pickle=False))
    rep = json.loads(Path(decl["inputs"]["w102_video_report"]["path"]).read_text())
    insp = json.loads(Path(decl["inputs"]["w102_video_inspection"]["path"]).read_text())
    if rep["decoded_BGR_uint8_sha256"] != insp["decoded_BGR_uint8_sha256"] or rep["clip_sha256"] != insp["clip_sha256"]:
        raise RuntimeError("STOP: W102 report/inspection identity disagree")
    if rep["clip_sha256"] != decl["inputs"]["repo_production_demo_clip"]["sha256"]:
        raise RuntimeError("STOP: clip identity")
    if not np.array_equal(z["timestamp"], np.asarray(insp["timestamp"])) or len(z["timestamp"]) != rep["frames"]:
        raise RuntimeError("STOP: timestamp identity")
    return z, rep, insp


def real_view(decl, z, rep, insp):
    from motion.animation_semantics_v2_bridge import current_root_orientation
    from pose.framepose_bridge import build_h0_lifted_sequence
    lm = z["landmarks"]  # (T,16,4) px x, y, score, visible ; NaN rows = absent
    T = len(lm)
    rows = [V.CANON.index(V.CANON_OF[n]) for n in V.JOINTS]
    raw = lm[:, rows]
    present = np.isfinite(raw[..., 0])
    size, fps = tuple(insp["size"]), float(rep["fps"])
    bank = scratch_real_bank(z, size, fps)
    lifted = build_h0_lifted_sequence(bank, z["predictions"], REAL_SEQUENCE_ID, fps=fps)
    yaw = [o.yaw_radians for o in current_root_orientation(lifted)]
    view = V.build_view(
        sequence_id=REAL_SEQUENCE_ID, actor_id=REAL_ACTOR_ID, actor_identity_status=REAL_ACTOR_STATUS,
        regime="real_animcv_observation", backend="mmpose", image_size=size, fps=fps,
        fps_source="W102 report fps (cv2 CAP_PROP_FPS) = inspection fps", frame_index=z["frame_index"],
        timestamp=z["timestamp"], obs2d_xy_px=raw[..., :2], obs2d_conf=np.where(present, raw[..., 2], np.nan),
        obs2d_present=present, obs2d_visible=present & (np.nan_to_num(raw[..., 3]) > .5), input_valid=z["input_valid"],
        est3d_xyz=z["predictions"], root_yaw=yaw,
        sources=source(decl, "w102_observations_predictions", "w102_video_report", "w102_video_inspection",
                       "repo_production_demo_clip", "s0_checkpoint"),
        producers={"obs2d": {"M1_identity": rep["M1_identity"], "environment_fingerprint": rep["environment_fingerprint"]},
                   "est3d": {"model": "frozen S0_neutral_sign (W102 run)", "checkpoint_sha256": decl["inputs"]["s0_checkpoint"]["sha256"]},
                   "root_yaw": "motion.animation_semantics_v2_bridge.current_root_orientation via "
                               "pose.framepose_bridge.build_h0_lifted_sequence (unchanged) on a scratch FrameBank",
                   "contact_status": "UNKNOWN: no real_animcv_observation/50 fps TRAIN calibration exists; 3DPW "
                                     "calibration not transferred"},
        per_frame_image_digest=rep["decoded_BGR_uint8_sha256"],
        notes={"rights": rep["rights"], "unseen": "false", "gt": "none exists for this clip"})
    return view, lifted, bank


def bench_view(decl):
    from framepose.contract import FrameBank
    from motion.animation_semantics_v2_bridge import current_root_orientation
    from pose.framepose_bridge import build_h0_lifted_sequence, sequence_frame_positions
    bank = FrameBank.load(decl["inputs"]["bench_bank_index"]["path"])
    if bank.content_digest() != BENCH_BANK_DIGEST:
        raise RuntimeError("STOP: benchmark bank digest")
    val = bank.indices("validation")
    pred = np.load(decl["inputs"]["bench_frozen_prediction_validation"]["path"])
    if pred.shape != (len(val), 17, 3):
        raise RuntimeError("STOP: benchmark prediction shape")
    h0 = np.full((len(bank), 17, 3), np.nan, np.float32)
    h0[val] = pred
    seq = bank.samples[int(val[0])].sequence_id
    pos = sequence_frame_positions(bank, seq)
    lifted = build_h0_lifted_sequence(bank, h0, seq)
    yaw = [o.yaw_radians for o in current_root_orientation(lifted)]
    s0 = bank.samples[pos[0]]
    w, h = s0.image_size
    x = bank.arrays["input_2d"][pos].astype(np.float64)
    v = bank.arrays["input_valid"][pos]
    xy = np.where(v[..., None], x[..., :2] * [w, h], np.nan)
    view = V.build_view(
        sequence_id=seq, actor_id=seq.rsplit(":", 1)[-1], actor_identity_status="dataset_actor_label_in_sequence_id",
        regime=s0.observation.regime, backend=s0.observation.backend, image_size=(w, h), fps=float(s0.fps),
        fps_source="FrameSample.fps", frame_index=[bank.samples[p].frame_index for p in pos],
        timestamp=[bank.samples[p].timestamp for p in pos], obs2d_xy_px=xy, obs2d_conf=np.where(v, x[..., 2], np.nan),
        obs2d_present=v, obs2d_visible=v, input_valid=v, est3d_xyz=h0[pos], root_yaw=yaw,
        sources=source(decl, "bench_bank_index", "bench_bank_arrays", "bench_frozen_prediction_validation"),
        producers={"est3d": "frozen S0 VALIDATION prediction (W101/W102)", "root_yaw": "current_root_orientation",
                   "contact_status": "not computed in W113"},
        obs2d_kind={n: "benchmark_shipped_detector_2d" for n in V.JOINTS},
        notes={"raw_below_threshold": "bank keeps no raw below-threshold coordinate: invalid cells are UNKNOWN",
               "bank_stratification_fields": "excluded: FrameSample stratification is computed from target_3d"})
    return view, {"sequence_id": seq, "rows": len(pos), "sample_ids_unique": len({bank.samples[p].sample_id for p in pos}) == len(pos),
                  "split": s0.split, "observation": s0.observation.to_dict()}


def expected_of(view, decl_sources, frame_count=None):
    m = view["meta"]
    e = {"sources": {k: v["sha256"] for k, v in decl_sources.items()}, "fps": m["fps"], "sequence_id": m["sequence_id"],
         "actor_id": m["actor_id"], "regime": m["regime"], "backend": m["backend"]}
    if frame_count:
        e["frame_count"] = frame_count
    return e


def directive_suite(view, preserve, permit, edits, *, synthetic=False):
    """Run declared directive/edit tests. `preserve`/`permit` are directive lists; `edits` name -> (cells, expected)."""
    res = {}
    dset = V.make_directive_set(view, "base", preserve + permit)
    masks = V.resolve_directives(view, dset)
    res["base"] = {"expected": "ACCEPT", "observed": "ACCEPT", "ok": True,
                   "preserve_cells": int(masks["preserve"].sum()), "permit_cells": int(masks["permit"].sum())}
    ctl = {
        "directive_actor_swapped": V.make_directive_set(view, "c1", preserve + permit, actor_id="other_actor"),
        "directive_sequence_swapped": V.make_directive_set(view, "c2", preserve + permit, sequence_id="other_seq"),
        "directive_stale_view_digest": V.make_directive_set(view, "c3", preserve + permit, view_content_digest="f" * 64),
        "directive_on_observation": V.make_directive_set(view, "c4", preserve + [dict(permit[0], signal="obs2d_xy_px")]),
        "directive_unknown_joint": V.make_directive_set(view, "c5", preserve + [dict(permit[0], joints=["tail"])]),
        "directive_span_out_of_range": V.make_directive_set(
            view, "c6", preserve + [dict(permit[0], spans=[[0, int(view["arrays"]["frame_index"][-1]) + 5]])]),
    }
    for name, d in ctl.items():
        try:
            V.resolve_directives(view, d)
            obs = "ACCEPT"
        except V.ContractError as e:
            obs = e.code
        res[name] = {"expected": DIRECTIVE_CONTROLS[name], "observed": obs, "ok": obs == DIRECTIVE_CONTROLS[name]}
    edited = None
    for name, (cells, expected, signal, digest) in edits.items():
        req = V.make_edit(view, name, cells, signal=signal)
        if digest:
            req["view_content_digest"] = digest
        try:
            after, log = V.apply_mock_edit(view, dset, req)
            obs = "ACCEPT"
        except V.ContractError as e:
            obs, after = e.code, None
        res[name] = {"expected": expected, "observed": obs, "ok": obs == expected, "cells": len(cells)}
        if obs == "ACCEPT" and expected == "ACCEPT":
            requested = np.zeros_like(masks["permit"])
            row = {int(f): i for i, f in enumerate(view["arrays"]["frame_index"])}
            for f, n in cells:
                requested[row[int(f)], V.JI[n]] = True
            loc = V.locality(view, after, masks["permit"], requested)
            res[name]["locality"] = loc
            res[name]["ok"] = res[name]["ok"] and loc["pass"]
            try:
                V.validate_view(after, verify_files=True)
                res[name]["edited_view_valid"] = True
            except V.ContractError as e:
                res[name]["edited_view_valid"] = e.code
                res[name]["ok"] = False
            edited = after
    return res, masks, edited, dset


def conflict_check(view, preserve, permit, extra):
    try:
        V.resolve_directives(view, V.make_directive_set(view, "conflict", preserve + permit + [extra]))
        obs = "ACCEPT"
    except V.ContractError as e:
        obs = e.code
    return {"expected": "LOCK_CONFLICT", "observed": obs, "ok": obs == "LOCK_CONFLICT"}


def roundtrip(view, stem):
    V.save_view(view, stem)
    back = V.load_view(stem)
    same = V.content_digest(back) == view["meta"]["content_digest"] and all(
        np.array_equal(back["arrays"][k], view["arrays"][k], equal_nan=view["arrays"][k].dtype.kind == "f")
        and back["arrays"][k].dtype == view["arrays"][k].dtype for k in view["arrays"])
    try:
        V.validate_view(back)
        valid = True
    except V.ContractError as e:
        valid = e.code
    return {"digest_and_arrays_identical": bool(same), "reloaded_valid": valid,
            "json_sha256": sha(Path(stem).with_suffix(".json")), "npz_sha256": sha(Path(stem).with_suffix(".npz"))}


def coverage(view, masks=None):
    a, st = view["arrays"], V.cell_status(view)
    T = len(a["frame_index"])
    derived = np.array([not k.startswith("observed_2d") for k in
                        (view["meta"]["signals"]["obs2d_xy_px"]["per_joint_kind"][n] for n in V.JOINTS)])
    def per_joint(m):
        return {n: int(m[:, i].sum()) for i, n in enumerate(V.JOINTS)}
    c = {"frames": T, "joints": V.J, "cells": T * V.J,
         "present": int(a["obs2d_present"].sum()), "absent_unknown": int((~a["obs2d_present"]).sum()),
         "observed_valid": int((st["obs2d_status"] == 1).sum()), "below_threshold": int((st["obs2d_status"] == 2).sum()),
         "visible_flag": int(a["obs2d_visible"].sum()), "in_frame": int(a["in_frame"].sum()),
         "derived_or_renamed_2d_cells": int(derived.sum() * T),
         "est3d_finite": int(a["est3d_finite"].sum()), "inferred_input_supported": int((st["est3d_status"] == 1).sum()),
         "inferred_input_unsupported": int((st["est3d_status"] == 2).sum()), "est3d_unknown": int((st["est3d_status"] == 0).sum()),
         "joinable_cells": T * V.J, "provenance_covered_cells": T * V.J if view["meta"]["sources"] else 0,
         "lockable_cells": T * V.J, "root_yaw_known_frames": int(a["root_yaw_known"].sum()),
         "visible_vs_valid_disagree": int((a["obs2d_visible"] != a["input_valid"]).sum()),
         "per_joint": {"present": per_joint(a["obs2d_present"]), "observed_valid": per_joint(st["obs2d_status"] == 1),
                       "below_threshold": per_joint(st["obs2d_status"] == 2),
                       "inferred_input_unsupported": per_joint(st["est3d_status"] == 2)}}
    if masks is not None:
        c["preserve_cells"] = int(masks["preserve"].sum())
        c["eligible_to_edit_cells"] = int(masks["permit"].sum())
        c["prohibited_cells"] = int((~masks["preserve"] & ~masks["permit"]).sum())
        c["eligible_by_status"] = {V.EST3D_STATUS[k]: int((masks["permit"] & (st["est3d_status"] == k)).sum()) for k in V.EST3D_STATUS}
    return c


def real_counterexamples(view, decl):
    from eval_external_hmr2_baseline import LIMBS, ang
    a = view["arrays"]
    p, v = a["est3d_xyz"].astype(np.float64), a["input_valid"]
    events = []
    for i in range(1, len(p)):
        for name, (x, y) in LIMBS.items():
            if v[i, [x, y]].all() and v[i - 1, [x, y]].all():
                d = ang(p[i, x] - p[i, y], p[i - 1, x] - p[i - 1, y])
                if d > 45:
                    events.append({"frame_index": int(a["frame_index"][i]), "limb": name, "step_deg": float(d),
                                   "min_conf": float(np.nanmin(a["obs2d_conf"][[i - 1, i]][:, [x, y]]))})
    lw, nose, lh = V.JI["left_wrist"], V.JI["head"], V.JI["left_hip"]
    windows = {}
    for wname, (s, e) in decl["failure_windows"].items():
        rows = np.flatnonzero((a["frame_index"] >= s) & (a["frame_index"] <= e))
        windows[wname] = {"frames": [int(s), int(e)], "rows": int(len(rows)),
                          "left_wrist_status": Counter(V.OBS2D_STATUS[int(k)] for k in V.cell_status(view)["obs2d_status"][rows, lw]),
                          "left_wrist_conf_range": [float(np.nanmin(a["obs2d_conf"][rows, lw])), float(np.nanmax(a["obs2d_conf"][rows, lw]))],
                          "left_wrist_y_px": [round(float(y), 1) for y in a["obs2d_xy_px"][rows, lw, 1]],
                          "head_nose_y_px": [round(float(y), 1) for y in a["obs2d_xy_px"][rows, nose, 1]],
                          "left_hip_y_px": [round(float(y), 1) for y in a["obs2d_xy_px"][rows, lh, 1]],
                          "est3d_status": Counter(V.EST3D_STATUS[int(k)] for k in V.cell_status(view)["est3d_status"][rows, lw]),
                          "root_yaw_known": int(a["root_yaw_known"][rows].sum())}
    return {"structurally_valid_possibly_wrong_3d": {"definition": decl["counterexample_definitions"]["structurally_valid_possibly_wrong_3d"],
                                                     "events": events, "count": len(events),
                                                     "verification": "GT-free: possibly wrong, NOT verified error"},
            "low_confidence_useful_2d": {"status": "NOT ESTABLISHED (no GT); see raw failure windows"},
            "failure_windows": windows}


def legacy_path(z, rep, decl, out):
    """Existing CLI-path code, unchanged: PoseSequence -> MotionGraph -> RetargetSolver -> AnimationClip."""
    from common.serialization import write_json
    from motion.motion_builder import MotionGraphBuilder
    from motion.motion_io import load_motion_graph
    from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
    from retarget.quality import RetargetQualityConfig, assess_retarget_quality
    from retarget.solver import RetargetSolver, load_animation_clip, save_animation_clip
    from rig.bone_mapping import load_bone_mapping_profile
    from rig.rig_parser import RigParser
    lm = z["landmarks"]
    frames = []
    for i, t in zip(z["frame_index"], z["timestamp"]):
        marks = {n: PoseLandmark(n, float(r[0]), float(r[1]), float(r[2]), bool(r[3] > .5))
                 for n, r in zip(V.CANON, lm[int(i)]) if np.isfinite(r[0])}
        frames.append(PoseFrame(int(i), float(t), marks))
    poses = PoseSequence(frames, float(rep["fps"]), "canonical_v1", float(rep["M1_identity"]["config"]["visibility_threshold"]))
    pose_path = out / "legacy_pose_reconstructed.json"
    write_json(pose_path, poses.to_dict())
    graph = MotionGraphBuilder().build(PoseSequence.from_dict(json.loads(pose_path.read_text())),
                                       source_metadata={"pose_source": str(pose_path)})
    write_json(out / "legacy_motion.json", graph.to_dict())
    graph = load_motion_graph(out / "legacy_motion.json")
    rig = RigParser().load(decl["inputs"]["legacy_rig_fbx"]["path"])
    mapping = load_bone_mapping_profile(decl["inputs"]["legacy_mapping"]["path"])
    cfg = RetargetQualityConfig(min_visibility_rate=0.60, min_mean_confidence=0.30, max_direction_step_degrees=120.0)
    quality = assess_retarget_quality(graph, rig, mapping, cfg).to_dict()
    clip = RetargetSolver().solve(graph, rig, mapping, quality_config=cfg, validate_quality=False, smoothing_window=1)
    save_animation_clip(clip, out / "legacy_animation.json")
    return pose_path, out / "legacy_motion.json", out / "legacy_animation.json", quality, mapping


def legacy_accounting(view, pose_path, motion_path, clip_path, quality, mapping, preserve_mask):
    from optimize.collapse import collapse_animation_clip
    from retarget.solver import load_animation_clip
    a = view["arrays"]
    mg = json.loads(Path(motion_path).read_text())
    clip_json = json.loads(Path(clip_path).read_text())
    T = len(a["frame_index"])
    # 2D survival through the serialized MotionGraph
    exact_xy = exact_conf = exact_vis = absent_ok = present_cells = 0
    for r, f in enumerate(mg["frames"]):
        for jn in V.CANON:
            jj = V.JI[jn]
            p = f["points"].get(jn)
            if a["obs2d_present"][r, jj]:
                present_cells += 1
                if p is not None:
                    exact_xy += int(tuple(p["position_2d"]) == tuple(a["obs2d_xy_px"][r, jj]))
                    exact_conf += int(p["confidence"] == a["obs2d_conf"][r, jj])
                    exact_vis += int(p["visible"] == bool(a["obs2d_visible"][r, jj]))
            else:
                absent_ok += int(p is None)
    try:
        V.reject_mixed_unit_motion_graph(mg)
        pos3d = "no position_3d (UNAVAILABLE in MotionGraph)"
    except V.ContractError as e:
        pos3d = e.code
    mg_mixed_probe = copy.deepcopy(mg)
    first = next(iter(mg_mixed_probe["frames"][0]["points"].values()))
    first["position_3d"] = [first["position_2d"][0], first["position_2d"][1], 0.37]
    try:
        V.reject_mixed_unit_motion_graph(mg_mixed_probe)
        mixed_probe = "ACCEPTED"
    except V.ContractError as e:
        mixed_probe = e.code
    mg_keys = sorted(mg)
    frame_keys = sorted(mg["frames"][0])
    point_keys = sorted(next(iter(mg["frames"][0]["points"].values())))
    # AnimationClip
    row = {int(f): i for i, f in enumerate(a["frame_index"])}
    bones = {e.target_bone: list(e.source_names) for e in mapping.entries}
    if set(clip_json["tracks"]) - set(bones):
        raise RuntimeError("STOP: clip track without a mapping entry (IK chain) is not covered by this accounting")
    zero_conf = zero_conf_missing = zero_conf_observed = held_equal = samples = unmatched = 0
    conf_exact = 0
    bone_rows = {}
    for bone, tr in clip_json["tracks"].items():
        prev = None
        srcs = bones[bone]
        for s in tr["samples"]:
            samples += 1
            r = row.get(int(s["frame_index"]))
            if r is None:
                unmatched += 1
                continue
            cells = [V.JI[n] for n in srcs]
            vis = all(a["obs2d_present"][r, c] and a["obs2d_visible"][r, c] for c in cells)
            if s["confidence"] == 0.0:
                zero_conf += 1
                if vis:
                    zero_conf_observed += 1
                else:
                    zero_conf_missing += 1
                    held_equal += int(prev is not None and s["rotation"] == prev)
            elif vis:
                conf_exact += int(s["confidence"] == min(a["obs2d_conf"][r, c] for c in cells))
            prev = s["rotation"]
            bone_rows.setdefault(bone, set()).add(r)
    collapse_probe = {}
    clip = load_animation_clip(clip_path)
    locked = {int(f) for f in a["frame_index"][preserve_mask.any(1)]}
    opt, _ = collapse_animation_clip(clip, preset="medium", locked_frames=locked)
    kept_locked = all(locked <= {s.frame_index for s in t.samples} for t in opt.tracks.values())
    opt_default, _ = collapse_animation_clip(clip, preset="medium")
    collapse_probe = {"locked_frames_passed": len(locked), "all_locked_frames_kept_as_keys": bool(kept_locked),
                      "keys_with_locked_frames": sum(len(t.samples) for t in opt.tracks.values()),
                      "keys_cli_default_no_locks": sum(len(t.samples) for t in opt_default.tracks.values()),
                      "semantics": "key retention only; values still re-derived from the solver; CLI never passes locks"}
    return {
        "pose_sequence_sha256": sha(pose_path), "motion_graph_sha256": sha(motion_path), "animation_clip_sha256": sha(clip_path),
        "motion_graph": {"top_keys": mg_keys, "frame_keys": frame_keys, "point_keys": point_keys,
                         "source_metadata": mg["source_metadata"], "fps": mg["fps"], "frames": len(mg["frames"]),
                         "present_cells_canon16": present_cells, "xy_exact": exact_xy, "conf_exact": exact_conf,
                         "visible_exact": exact_vis, "absent_cells_without_point": absent_ok,
                         "absent_cells_canon16": int(sum(1 for r in range(T) for jn in V.CANON if not a["obs2d_present"][r, V.JI[jn]])),
                         "locked_true_frames": sum(1 for f in mg["frames"] if f["locked"]),
                         "timestamps_exact": all(f["timestamp"] == float(a["timestamp"][r]) for r, f in enumerate(mg["frames"])),
                         "position_3d": pos3d, "mixed_unit_probe": mixed_probe,
                         "derived_2d_flag_present": any("derived" in k for k in point_keys),
                         "sequence_or_actor_id_present": any(k in mg["source_metadata"] for k in ("sequence_id", "actor_id")),
                         "source_hash_present": any("sha" in k for k in mg["source_metadata"])},
        "animation_clip": {"top_keys": sorted(clip_json), "sample_keys": sorted(next(iter(clip_json["tracks"].values()))["samples"][0]),
                           "name": clip_json["name"], "fps": clip_json["fps"], "frame_range": [clip_json["frame_start"], clip_json["frame_end"]],
                           "tracks": sorted(clip_json["tracks"]), "samples": samples, "unmatched_frames": unmatched,
                           "bones_covering_all_frames": sum(1 for r in bone_rows.values() if len(r) == T),
                           "confidence_zero_samples": zero_conf, "confidence_zero_with_missing_input": zero_conf_missing,
                           "confidence_zero_with_observed_input": zero_conf_observed,
                           "confidence_zero_held_previous_rotation": held_equal,
                           "confidence_equals_min_source_conf_when_observed": conf_exact,
                           "timestamps_present": any("timestamp" in k for k in sorted(next(iter(clip_json["tracks"].values()))["samples"][0])),
                           "fps_derivable_time_exact": bool(np.abs(a["frame_index"] / clip_json["fps"] - a["timestamp"]).max() < 1e-12),
                           "source_identity_present": any(k not in ("name", "fps", "tracks", "frame_start", "frame_end") for k in clip_json),
                           "lock_or_permission_present": False},
        "quality_report": {"passed": quality.get("passed"), "summary": {k: v for k, v in quality.items() if k != "chains"}},
        "collapse_locked_frames_probe": collapse_probe,
        "sidecar_association": {"rule": "bone-frame -> view row by unique frame_index; source joints from mapping",
                                "unique_frame_index_in_view": len(row) == T, "unmatched": unmatched,
                                "identity_binding": "clip carries no source digest: a sidecar must record (clip sha256, "
                                                    "view digest) at creation; frame_index alone is not identity-safe"},
    }


def semantics_accounting(view, lifted, bank):
    from framepose.nonlinear_reconstruction import in_frame_mask
    from motion.animation_semantics import LocalArticulation, ObservationReliability
    from motion.animation_semantics_v2_bridge import current_root_orientation
    a = view["arrays"]
    T = len(a["frame_index"])
    conf = np.array([[f.points[n].confidence for n in V.JOINTS] for f in lifted.frames])
    du = np.array([[f.points[n].depth_uncertainty for n in V.JOINTS] for f in lifted.frames])
    ov = np.array([[f.points[n].observation_valid for n in V.JOINTS] for f in lifted.frames])
    pos = np.array([[f.points[n].position for n in V.JOINTS] for f in lifted.frames])
    inv = ~a["input_valid"]
    below = a["obs2d_present"] & inv
    inf = in_frame_mask(bank.arrays["input_2d"], bank.arrays["input_valid"])
    frames = [{"frame_index": f.frame_index, "timestamp": f.timestamp,
               "articulation": LocalArticulation(tuple(f.points[n].position for n in V.JOINTS)).to_dict(),
               "reliability": ObservationReliability(tuple(bool(x) for x in ov[r]), tuple(bool(x) for x in inf[r])).to_dict(),
               "root_orientation": o.to_dict()}
              for r, (f, o) in enumerate(zip(lifted.frames, current_root_orientation(lifted)))]
    blob = json.dumps(frames, sort_keys=True).encode()
    return {
        "lifted_frames": len(lifted.frames), "lifted_backend": lifted.backend, "lifted_units": lifted.units,
        "lifted_frame_index_exact": [f.frame_index for f in lifted.frames] == a["frame_index"].tolist(),
        "lifted_timestamp_exact": [f.timestamp for f in lifted.frames] == a["timestamp"].tolist(),
        "articulation_exact": bool(np.array_equal(pos.astype(np.float32), a["est3d_xyz"])),
        "observation_valid_equals_input_valid_and_finite": bool(np.array_equal(ov, a["input_valid"] & a["est3d_finite"])),
        "confidence_of_invalid_cells": {"cells": int(inv.sum()), "zero": int((conf[inv] == 0).sum()),
                                        "raw_score_retained_in_view_for_below_threshold": int(below.sum()),
                                        "raw_score_lost_in_bridge_for_below_threshold": int((conf[below] != a["obs2d_conf"][below]).sum())},
                                        # raw score recovered by joining the W102 landmarks of the same row
        "depth_uncertainty_proxy": {"equals_1_minus_conf": bool(np.allclose(du, 1 - conf)),
                                    "reported_as": "2D proxy; never placed in the view as 3D confidence"},
        "in_frame_mask_equals_view": bool(np.array_equal(inf, a["in_frame"])),
        "root_yaw_known_equals_view": bool(np.array_equal(np.array([f["root_orientation"]["status"] == "known" for f in frames]),
                                                          a["root_yaw_known"])),
        "semantic_frame_pieces_sha256": hashlib.sha256(blob).hexdigest(),
        "fields_absent_from_semantics": ["obs2d_xy_px", "obs2d_conf", "obs2d_visible", "derived_2d_kind", "actor_id",
                                         "per_frame_image_digest", "user_lock", "edit_provenance"],
        "full_v2_serialization": "NOT EXECUTED: AnimationSemanticsV2 requires a contact calibration; none exists for "
                                 "real_animcv_observation at 50 fps and transferring the 3DPW calibration is not authorized",
    }


def rig_association(decl, view, masks, z, audit_path):
    vs = np.load(decl["inputs"]["w105_video_sources"]["path"])
    cases = json.loads(Path(decl["inputs"]["w105_video_cases"]["path"]).read_text())["cases"]["real_clip"]
    w105_audit = json.loads(Path(decl["inputs"]["w105_render_clip_audit"]["path"]).read_text())
    a = view["arrays"]
    res = {"w105_sources": {"pred_M_byte_equal_w102": bool(np.array_equal(vs["pred_M"], z["predictions"])),
                            "input_valid_equal": bool(np.array_equal(vs["input_valid"], z["input_valid"])),
                            "timestamp_equal": bool(np.array_equal(vs["timestamp"], z["timestamp"]))},
           "w105_cases": {"frames": len(cases["frames"]), "frame_index_equal": [f["frame_index"] for f in cases["frames"]] == a["frame_index"].tolist(),
                          "time_equal": [f["time_s"] for f in cases["frames"]] == a["timestamp"].tolist(),
                          "identity_label": sorted({f["identity"] for f in cases["frames"]}),
                          "carries_validity_or_confidence": any(k in cases["frames"][0] for k in ("input_valid", "confidence")),
                          "labels": cases["labels"]}}
    if audit_path is None or not Path(audit_path).exists():
        res["replay"] = "NOT RUN"
        return res
    new = json.loads(Path(audit_path).read_text())
    ref_file = w105_audit["files"]["real_clip/real_clip.blend"]
    new_file = new["files"]["real_clip/real_clip.blend"]
    res["replay"] = {"audit_sha256": sha(audit_path), "scene_fps": new_file["scene_fps"], "frame_range": new_file["frame_range"],
                     "source_fps": view["meta"]["fps"], "armatures": new_file["armatures"],
                     "actions": {}}
    for name, act in new_file["actions"].items():
        ref = ref_file["actions"].get(name)
        keyed = [int(round(f)) for f in act["keyed_frames"]]
        bones = sorted({ch["data_path"].split('"')[1] if '"' in ch["data_path"] else "__object__" for ch in act["channels"]})
        res["replay"]["actions"][name] = {
            "value_digest": act["value_digest"], "w105_value_digest": ref and ref["value_digest"],
            "value_digest_equal": bool(ref and ref["value_digest"] == act["value_digest"]), "fcurves": act["fcurves"],
            "keyed_frames_equal_frame_index_plus_1": keyed == (a["frame_index"] + 1).tolist(),
            "interpolation": act["interpolation"], "bones": bones,
            "bones_without_declared_dependency": [b for b in bones if b not in W105_BONE_DEPENDENCIES]}
    # bone-frame permission association under RS1 (M_Main = FIXED action)
    perm = {}
    for bone, deps in W105_BONE_DEPENDENCIES.items():
        cols = [V.JI[n] for n in deps]
        pres, pe = masks["preserve"][:, cols], masks["permit"][:, cols]
        locked = pres.any(1)
        editable = pe.all(1) & ~locked
        partial = pe.any(1) & ~pe.all(1) & ~locked
        perm[bone] = {"locked": int(locked.sum()), "editable": int(editable.sum()), "partial_permit": int(partial.sum()),
                      "prohibited": int((~locked & ~pe.any(1)).sum())}
    res["bone_frame_permission_RS1"] = perm
    res["bone_permission_semantics"] = ("AMBIGUOUS at the rig boundary: a permitted joint cell (left_wrist) feeds a "
                                        "bone (lowerarm_l) whose other dependency cells are prohibited; no existing "
                                        "owner defines bone-level edit authority")
    res["time_at_rig"] = {"action_frame_rate": new_file["scene_fps"], "source_fps": view["meta"]["fps"],
                          "action_time_equals_source_time": new_file["scene_fps"] == view["meta"]["fps"],
                          "recoverable_via_sidecar_frame_index": True}
    return res


def cmd_build(a):
    decl = load_declaration(a)
    out = new_dir(a.out_dir)
    t0 = time.perf_counter()
    rep = {"schema": "w113_build_v1", "declaration_sha256": a.declaration_sha, "tag": a.tag}
    # ---- synthetic
    syn = synthetic_view(out)
    syn_src = {"synthetic_source": syn["meta"]["sources"]["synthetic_source"]}
    rep["synthetic"] = {"validate": V.validate_view(syn), "controls": V.run_view_controls(syn, expected_of(syn, syn_src)),
                        "coverage": coverage(syn), "roundtrip": roundtrip(syn, out / "view_synthetic")}
    pres = [{"id": "P_right_arm", "kind": "preserve", "signal": "est3d_xyz", "joints": RIGHT_ARM, "spans": [[0, 7]]},
            {"id": "P_first", "kind": "preserve", "signal": "est3d_xyz", "joints": list(V.JOINTS), "spans": [[0, 1]]}]
    perm = [{"id": "E_lw", "kind": "permit", "signal": "est3d_xyz", "joints": ["left_wrist"], "spans": [[3, 5]]},
            {"id": "E_le", "kind": "permit", "signal": "est3d_xyz", "joints": ["left_elbow"], "spans": [[4, 4]]}]
    edits = {"E_ok": ([(3, "left_wrist"), (4, "left_wrist"), (5, "left_wrist")], "ACCEPT", "est3d_xyz", None),
             "edit_on_locked_cell": ([(4, "right_wrist")], "LOCK_CONFLICT_EDIT", "est3d_xyz", None),
             "edit_on_prohibited_cell": ([(4, "left_knee")], "EDIT_NOT_AUTHORIZED", "est3d_xyz", None),
             "edit_on_unknown_cell (synthetic only)": ([(4, "left_elbow")], "EDIT_UNKNOWN_TARGET", "est3d_xyz", None),
             "edit_on_observation": ([(3, "left_wrist")], "OBSERVATION_IMMUTABLE", "obs2d_xy_px", None),
             "edit_stale_view_digest": ([(3, "left_wrist")], "DIRECTIVE_STALE_VIEW", "est3d_xyz", "e" * 64)}
    sres, smasks, sedited, _ = directive_suite(syn, pres, perm, edits, synthetic=True)
    sres["lock_conflict_set"] = conflict_check(syn, pres, perm, {"id": "E_c", "kind": "permit", "signal": "est3d_xyz",
                                                                "joints": ["left_wrist"], "spans": [[0, 5]]})
    rep["synthetic"]["directives"] = sres
    rep["synthetic"]["edited_roundtrip"] = roundtrip(sedited, out / "view_synthetic_edited") if sedited else None
    # ---- real
    z, wrep, insp = real_inputs(decl)
    real, lifted, sbank = real_view(decl, z, wrep, insp)
    real_src = source(decl, "w102_observations_predictions", "w102_video_report", "w102_video_inspection",
                      "repo_production_demo_clip", "s0_checkpoint")
    rexp = expected_of(real, real_src, frame_count=int(wrep["frames"]))
    rexp["fps"] = float(insp["fps"])
    rep["real"] = {"validate": V.validate_view(real, expected=rexp), "controls": V.run_view_controls(real, rexp),
                   "roundtrip": roundtrip(real, out / "view_real"),
                   "est3d_root_centred_max_abs_m": float(np.abs(z["predictions"][:, 0]).max()),
                   "est3d_median_thigh_length_m": float(np.median(np.linalg.norm(z["predictions"][:, 1] - z["predictions"][:, 2], axis=-1))),
                   "input_2d_zero_fill_of_invalid": int(((z["input_2d"][~z["input_valid"]]) == 0).all(-1).sum()),
                   "invalid_cells": int((~z["input_valid"]).sum()),
                   "thorax_equals_neck_2d": bool(np.array_equal(real["arrays"]["obs2d_xy_px"][:, V.JI["thorax"]],
                                                                real["arrays"]["obs2d_xy_px"][:, V.JI["neck"]], equal_nan=True))}
    fi = real["arrays"]["frame_index"]
    lw_invalid = ~real["arrays"]["input_valid"][:, V.JI["left_wrist"]] & (fi >= 100)
    spans = [[int(fi[s]), int(fi[e])] for s, e in V.runs(lw_invalid)]
    rpres = [{"id": "P_ref_span", "kind": "preserve", "signal": "est3d_xyz", "joints": list(V.JOINTS), "spans": [[0, 99]]},
             {"id": "P_right_arm", "kind": "preserve", "signal": "est3d_xyz", "joints": RIGHT_ARM, "spans": [[int(fi[0]), int(fi[-1])]]}]
    rperm = [{"id": "E_low_evidence_left_wrist", "kind": "permit", "signal": "est3d_xyz", "joints": ["left_wrist"], "spans": spans}]
    ok_cells = [(int(f), "left_wrist") for f in fi[lw_invalid]]
    redits = {"E_ok": (ok_cells, "ACCEPT", "est3d_xyz", None),
              "edit_on_locked_cell": ([(50, "left_wrist")], "LOCK_CONFLICT_EDIT", "est3d_xyz", None),
              "edit_on_prohibited_cell": ([(150, "left_knee")], "EDIT_NOT_AUTHORIZED", "est3d_xyz", None),
              "edit_on_observation": ([(150, "left_wrist")], "OBSERVATION_IMMUTABLE", "obs2d_xy_px", None),
              "edit_stale_view_digest": (ok_cells[:1] or [(150, "left_wrist")], "DIRECTIVE_STALE_VIEW", "est3d_xyz", "e" * 64)}
    if not spans:
        rep["real"]["vacuous_permit"] = True
        rperm = [{"id": "E_low_evidence_left_wrist", "kind": "permit", "signal": "est3d_xyz", "joints": ["left_wrist"], "spans": []}]
    rres, rmasks, redited, _ = directive_suite(real, rpres, rperm, redits)
    rres["lock_conflict_set"] = conflict_check(real, rpres, rperm, {"id": "E_conflict", "kind": "permit", "signal": "est3d_xyz",
                                                                    "joints": ["right_knee"], "spans": [[90, 110]]})
    rep["real"]["directives"] = rres
    rep["real"]["permit_spans"] = spans
    rep["real"]["vacuous_permit"] = not spans
    rep["real"]["edited_roundtrip"] = roundtrip(redited, out / "view_real_edited") if redited else None
    rep["real"]["coverage"] = coverage(real, rmasks)
    rep["real"]["counterexamples"] = real_counterexamples(real, decl)
    np.savez(out / "real_masks.npz", preserve=rmasks["preserve"], permit=rmasks["permit"],
             **{k: v for k, v in V.cell_status(real).items()})
    if redited is not None:
        np.save(out / "real_edited_est3d.npy", redited["arrays"]["est3d_xyz"])
    # existing paths
    rep["real"]["legacy"] = legacy_accounting(real, *legacy_path(z, wrep, decl, out), rmasks["preserve"])
    rep["real"]["semantics"] = semantics_accounting(real, lifted, sbank)
    rep["real"]["rig"] = rig_association(decl, real, rmasks, z, a.rig_audit)
    # ---- benchmark (optional, separate)
    bview, binfo = bench_view(decl)
    bsrc = source(decl, "bench_bank_index", "bench_bank_arrays", "bench_frozen_prediction_validation")
    bexp = expected_of(bview, bsrc)
    bfi = bview["arrays"]["frame_index"]
    binv = ~bview["arrays"]["input_valid"].any(0)
    rep["benchmark"] = {"info": binfo, "validate": V.validate_view(bview, expected=bexp),
                        "controls": V.run_view_controls(bview, bexp), "coverage": coverage(bview),
                        "roundtrip": roundtrip(bview, out / "view_benchmark"),
                        "frame_stride": sorted(set(np.diff(bfi).tolist())),
                        "bank_strata_gt_derived_keys": sorted(k for k in json.loads(Path(decl["inputs"]["bench_bank_index"]["path"]).read_text())["samples"][0]["strata"]
                                                              if "target" in k or "depth" in k or "facing" in k or "bend" in k),
                        "strata_in_view": any(t in k.lower() for k in list(V._walk_keys(bview["meta"])) + list(bview["arrays"])
                                              for t in V.FORBIDDEN_TOKENS)}
    rep["digests"] = {"synthetic": syn["meta"]["content_digest"], "real": real["meta"]["content_digest"],
                      "real_edited": redited and redited["meta"]["content_digest"], "benchmark": bview["meta"]["content_digest"],
                      "synthetic_edited": sedited and sedited["meta"]["content_digest"]}
    rep["cost"] = {"build_seconds": time.perf_counter() - t0, "real_frames": int(len(fi)),
                   "bytes": {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}}
    dump(out / "build_report.json", rep)
    print(json.dumps({"tag": a.tag, "digests": rep["digests"], "seconds": round(rep["cost"]["build_seconds"], 2)}))
    return 0


# --------------------------------------------------------------------------------------------- repeat compare ---

# Differences that are expected ONLY because an artefact records its own build-directory path (synthetic source path
# inside the synthetic view; the CLI's MotionGraph source_metadata.pose_source). Each is re-checked path-normalized.
PATH_INDUCED = (".synthetic.validate.content_digest", ".synthetic.roundtrip.json_sha256", ".synthetic.edited_roundtrip.json_sha256",
                ".digests.synthetic", ".digests.synthetic_edited", ".real.legacy.motion_graph.source_metadata.pose_source",
                ".real.legacy.motion_graph_sha256")


def _normalized_json(path, drop):
    d = json.loads(Path(path).read_text())
    for keys in drop:
        node = d
        for k in keys[:-1]:
            node = node[k]
        node.pop(keys[-1], None)
    return d


def cmd_compare(a):
    out = new_dir(a.out_dir)
    b1, b2 = Path(a.first), Path(a.second)
    r1 = json.loads((b1 / "build_report.json").read_text())
    r2 = json.loads((b2 / "build_report.json").read_text())
    diffs = []

    def walk(x, y, path=""):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{path}.{k}")
        elif x != y:
            diffs.append(path)
    walk({k: v for k, v in r1.items() if k not in ("cost", "tag")}, {k: v for k, v in r2.items() if k not in ("cost", "tag")})
    files = {}
    for name in ("view_real.npz", "view_real.json", "view_real_edited.npz", "view_real_edited.json", "view_benchmark.npz",
                 "view_benchmark.json", "view_synthetic.npz", "view_synthetic_edited.npz", "synthetic_source.npz",
                 "legacy_pose_reconstructed.json", "legacy_animation.json", "real_edited_est3d.npy", "real_masks.npz"):
        files[name] = sha(b1 / name) == sha(b2 / name)
    norm = {
        "view_synthetic.json": _normalized_json(b1 / "view_synthetic.json", [("sources", "synthetic_source", "path"), ("content_digest",)])
        == _normalized_json(b2 / "view_synthetic.json", [("sources", "synthetic_source", "path"), ("content_digest",)]),
        "view_synthetic_edited.json": _normalized_json(b1 / "view_synthetic_edited.json", [("sources", "synthetic_source", "path"), ("content_digest",), ("parent_content_digest",), ("edits",)])
        == _normalized_json(b2 / "view_synthetic_edited.json", [("sources", "synthetic_source", "path"), ("content_digest",), ("parent_content_digest",), ("edits",)]),
        "legacy_motion.json": _normalized_json(b1 / "legacy_motion.json", [("source_metadata", "pose_source")])
        == _normalized_json(b2 / "legacy_motion.json", [("source_metadata", "pose_source")]),
    }
    unexpected = [d for d in diffs if d not in PATH_INDUCED]
    rep = {"schema": "w113_repeat_v1", "report_differences": diffs, "unexpected_differences": unexpected,
           "file_identity": files, "path_normalized_identity": norm,
           "real_and_benchmark_digests_equal": all(r1["digests"][k] == r2["digests"][k] for k in ("real", "real_edited", "benchmark")),
           "pass": not unexpected and all(files.values()) and all(norm.values())
           and all(r1["digests"][k] == r2["digests"][k] for k in ("real", "real_edited", "benchmark"))}
    dump(out / "repeat_report.json", rep)
    print(json.dumps({"pass": rep["pass"], "differences": diffs[:20]}))
    return 0


# --------------------------------------------------------------------------------------------------- figures ---

def cmd_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    out = new_dir(a.out_dir)
    b = Path(a.build_dir)
    view = V.load_view(b / "view_real")
    m = np.load(b / "real_masks.npz")
    edited = np.load(b / "real_edited_est3d.npy") if (b / "real_edited_est3d.npy").exists() else None
    fi = view["arrays"]["frame_index"]
    st = m["obs2d_status"].T
    fig, ax = plt.subplots(3, 1, figsize=(13, 9.5), gridspec_kw={"height_ratios": [3, 3, 1.6]}, constrained_layout=True)
    cmap = ListedColormap(["#c8c8c8", "#2a6fb0", "#e3a33b"])
    ax[0].imshow(st, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2,
                 extent=(fi[0] - .5, fi[-1] + .5, V.J - .5, -.5))
    ax[0].set_yticks(range(V.J), [f"{n} [{V.OBS2D_KIND[n].replace('observed_2d', 'obs').replace('derived_2d_', 'der:')[:22]}]"
                                  for n in V.JOINTS], fontsize=7)
    ax[0].set_title("Real clip (real_animcv_observation, GT-free): 2D observation status per joint-frame\n"
                    "blue = observed valid, amber = present below validity threshold, grey = absent (UNKNOWN)", fontsize=9)
    perm = np.zeros(st.shape)
    perm[m["preserve"].T] = 1
    perm[m["permit"].T] = 2
    ax[1].imshow(perm, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#f2f2f2", "#5b5b5b", "#d1495b"]),
                 vmin=0, vmax=2, extent=(fi[0] - .5, fi[-1] + .5, V.J - .5, -.5))
    ax[1].set_yticks(range(V.J), V.JOINTS, fontsize=7)
    ax[1].set_title("Synthetic user directive RS1 on est3d cells: dark = preserved (lock), red = permitted edit, "
                    "light = prohibited by default", fontsize=9)
    lw = V.JI["left_wrist"]
    ax[2].plot(fi, view["arrays"]["est3d_xyz"][:, lw, 2], color="#2a6fb0", lw=1.2, label="frozen S0 left_wrist z (m)")
    if edited is not None:
        changed = (edited[:, lw] != view["arrays"]["est3d_xyz"][:, lw]).any(-1)
        ax[2].scatter(fi[changed], edited[changed, lw, 2], s=9, color="#d1495b",
                      label="SYNTHETIC mock +0.01 m (not animation)")
    ax[2].set_xlabel("frame_index (50 fps)")
    ax[2].legend(fontsize=7, loc="upper left", frameon=False)
    for x in (109, 118, 154, 181):
        for k in range(3):
            ax[k].axvline(x, color="k", lw=.5, ls=":")
    fig.savefig(out / "real_timeline.png", dpi=110)
    plt.close(fig)
    print(out / "real_timeline.png")
    return 0


# ---------------------------------------------------------------------------------------------------- decide ---

def case_of(invalid, b_hits, a_conditions) -> str:
    """Frozen precedence D > B > A > C."""
    if invalid:
        return "D"
    if b_hits:
        return "B"
    if a_conditions and all(a_conditions.values()):
        return "A"
    return "C"


def all_ok(d):
    return all(v.get("ok", True) for k, v in d.items() if isinstance(v, dict) and "ok" in v)


def cmd_decide(a):
    decl = load_declaration(a)
    out = new_dir(a.out_dir)
    pre = json.loads(Path(a.preflight).read_text())
    r = json.loads((Path(a.build_dir) / "build_report.json").read_text())
    rp = json.loads(Path(a.repeat).read_text())
    post_src = git("status", "--porcelain", "--", "src", "docs/next_batch.md")
    inputs_now = {k: sha(v["path"]) == v["sha256"] for k, v in decl["inputs"].items()}
    real, syn, ben = r["real"], r["synthetic"], r["benchmark"]
    rig = real["rig"]
    replay = rig.get("replay")
    fixed = replay["actions"].get("W91 real_clip FIXED") if isinstance(replay, dict) else None
    D = {
        "preflight": pre["pass"],
        "inputs_unchanged": all(inputs_now.values()),
        "src_and_instruction_unmodified": post_src == "",
        "synthetic_controls": all_ok(syn["controls"]) and all_ok(syn["directives"]),
        "real_controls": all_ok(real["controls"]) and all_ok(real["directives"]),
        "benchmark_controls": all_ok(ben["controls"]),
        "roundtrips": all(x["digest_and_arrays_identical"] and x["reloaded_valid"] is True
                          for x in (syn["roundtrip"], real["roundtrip"], ben["roundtrip"])
                          + tuple(y for y in (syn["edited_roundtrip"], real["edited_roundtrip"]) if y)),
        "no_gt_in_views": not ben["strata_in_view"],
        "mixed_unit_rejected": real["legacy"]["motion_graph"]["mixed_unit_probe"] == "MIXED_UNIT_3D",
        "repeat": rp["pass"],
        "lock_is_this_batch": bool(pre["main_local_lock"] and pre["main_local_lock"].get("batch") == BATCH),
    }
    invalid = [k for k, v in D.items() if not v]
    lg, se = real["legacy"], real["semantics"]
    fields = {
        "M1_obs2d": {"available_in_view": True, "upstream_serialized": "W102 npz landmarks",
                     "lost_at": ["AnimationClip (only min-conf per bone)", "FramePose input_2d zero-fill (raw score)",
                                 "AnimationSemantics (xy/conf)"],
                     "scratch_recoverable": se["lifted_frame_index_exact"] and lg["motion_graph"]["xy_exact"] == lg["motion_graph"]["present_cells_canon16"]},
        "M2_obs2d_kind": {"available_in_view": True, "upstream_serialized": None,
                          "note": "never serialized by any contract; recoverable only from the adapter's static rule",
                          "scratch_recoverable": True},
        "M3_est3d": {"available_in_view": True, "lost_at": ["MotionGraph/AnimationClip CLI path (never enters)"],
                     "scratch_recoverable": se["articulation_exact"]},
        "M4_observed_vs_inferred": {"available_in_view": True, "scratch_recoverable": True},
        "M5a_sequence_frame_time": {"available_in_view": True,
                                    "lost_at": ["AnimationClip timestamps (fps-derivable)", "MotionGraph/AnimationClip sequence id",
                                                "Blender Action time (30 vs 50 fps)"],
                                    "scratch_recoverable": lg["animation_clip"]["fps_derivable_time_exact"]
                                    and lg["sidecar_association"]["unique_frame_index_in_view"] and bool(fixed and fixed["keyed_frames_equal_frame_index_plus_1"])},
        "M5b_actor_track": {"available_in_view": "sequence-level label only", "producer_serialized": False,
                            "note": "per-frame SubjectTracker selection is internal to the adapter and never serialized"},
        "M6_source_provenance": {"available_in_view": True, "lost_at": ["MotionGraph (path only)", "AnimationClip", "Action"],
                                 "scratch_recoverable": True,
                                 "note": "sidecar must bind artefact sha256 at creation; frame_index alone is not identity-safe"},
        "M7_user_directive": {"production_producer": False, "scratch_enforced": real["directives"]["E_ok"]["ok"] and syn["directives"]["E_ok"]["ok"]},
        "M8_unavailable": {"classified": True},
    }
    B_hits = [k for k, f in fields.items() if k in ("M1_obs2d", "M3_est3d", "M5a_sequence_frame_time", "M6_source_provenance")
              and f.get("lost_at") and f.get("scratch_recoverable") is False]
    A_conditions = {
        "valid": not invalid,
        "producer_fields_joinable": not B_hits and all(fields[k].get("scratch_recoverable", True) for k in fields if "scratch_recoverable" in fields[k]),
        "directives_enforced_non_vacuous": fields["M7_user_directive"]["scratch_enforced"] and not real["vacuous_permit"],
        "rig_replay_identical": bool(fixed and fixed["value_digest_equal"]),
        "rig_association": bool(fixed and fixed["keyed_frames_equal_frame_index_plus_1"]) and lg["sidecar_association"]["unmatched"] == 0,
        "M5b_has_serialized_producer": fields["M5b_actor_track"]["producer_serialized"],
        "rig_permission_semantics_unambiguous": not str(rig.get("bone_permission_semantics", "")).startswith("AMBIGUOUS"),
    }
    case = case_of(invalid, B_hits, A_conditions)
    rep = {"schema": "w113_decision_v1", "declaration_sha256": a.declaration_sha, "case": case, "validity_gates": D,
           "invalid": invalid, "minimum_fields": fields, "B_hits": B_hits, "A_conditions": A_conditions,
           "A_unmet": [k for k, v in A_conditions.items() if not v], "precedence": CASE_LOGIC["precedence"]}
    dump(out / "decision.json", rep)
    print(json.dumps({"case": case, "invalid": invalid, "B_hits": B_hits, "A_unmet": rep["A_unmet"]}))
    return 0


def cmd_manifest(a):
    base = Path(a.base)
    every = [p for p in sorted(base.rglob("*")) if p.is_file() and p.name != "evidence_manifest.json"]
    pngs = [p for p in every if p.suffix == ".png" and "rig_replay" in p.parts]
    files = {p.relative_to(base).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in every if p not in pngs}
    h = hashlib.sha256()
    for p in pngs:
        h.update(p.relative_to(base).as_posix().encode())
        h.update(sha(p).encode())
    rep = {"schema": "w113_evidence_manifest_v1", "starting_head": STARTING_HEAD, "declaration_sha256": a.declaration_sha,
           "files": files, "render_pngs": {"count": len(pngs), "combined_digest": h.hexdigest()},
           "total_bytes": sum(p.stat().st_size for p in base.rglob("*") if p.is_file())}
    dump(base / "evidence_manifest.json", rep)
    print(sha(base / "evidence_manifest.json"), len(files), len(pngs))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--out-dir", required=True)
    p = sub.add_parser("declare"); p.add_argument("--out-dir", required=True); p.add_argument("--preflight", required=True)
    for name in ("build", "decide"):
        p = sub.add_parser(name)
        p.add_argument("--declaration", required=True); p.add_argument("--declaration-sha", required=True)
        p.add_argument("--out-dir", required=True)
    sub.choices["build"].add_argument("--tag", required=True)
    sub.choices["build"].add_argument("--rig-audit", default=None)
    sub.choices["decide"].add_argument("--preflight", required=True)
    sub.choices["decide"].add_argument("--build-dir", required=True)
    sub.choices["decide"].add_argument("--repeat", required=True)
    p = sub.add_parser("compare"); p.add_argument("--first", required=True); p.add_argument("--second", required=True)
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("figures"); p.add_argument("--build-dir", required=True); p.add_argument("--out-dir", required=True)
    p = sub.add_parser("manifest"); p.add_argument("--base", required=True); p.add_argument("--declaration-sha", required=True)
    a = ap.parse_args(argv)
    return {"preflight": cmd_preflight, "declare": cmd_declare, "build": cmd_build, "compare": cmd_compare,
            "figures": cmd_figures, "decide": cmd_decide, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
