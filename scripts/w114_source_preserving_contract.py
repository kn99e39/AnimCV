#!/usr/bin/env python3
"""W114: source-preserving Animation Base & edit authority feasibility (evidence only).

Stages (each refuses an existing output directory):
  preflight      instruction/HEAD/W113 marker/lock, input + historical-artefact authentication, runtime
  declare        frozen declaration: inputs, hypothesis, controls with exact expected codes, directive sets, edit
                 programmes, rig perturbations with expected channel sets, gates, CASE logic (no outcome read)
  build          synthetic / real / benchmark candidate programmes, negative controls, score accounting, the
                 unchanged existing-path control (W113 helpers), frozen-rig influence (real S0-only clip and two
                 W102 mixed-source windows), Blender case file for the Action-level replay
  blender-compare  Action-level comparison of the Blender replay (FIXED = unperturbed, RUNTIME = perturbed)
  compare        repeat comparison of two independent builds
  decide         frozen gate / CASE logic over the recorded inputs
  manifest       evidence manifest

NON-PRODUCTION. No src/ module, schema, model, threshold, rig, solver, FBX/Blender exporter or default changes. No
inference is run; every 2D/3D array is a stored, hash-verified artefact. No evaluation GT is read (the J1X npz's target
arrays and the bank's target_3d/strata are never accessed).
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w114_source_preserving_base as B  # noqa: E402
import w114_rig_influence as R  # noqa: E402

BATCH = 114
INSTRUCTION_BLOB = "6d8ce038a4a2a878aef3136f2f4723d6efebd68c"
STARTING_HEAD = "fd24ef0817eb185e4d9ed2ad80bfb282f7bb78b6"
W113_WORKLOG = "docs/113_WORKLOG_ANIMATION_BASE_CONTRACT_FEASIBILITY.md"
W101, W102, W105, W113 = (Path(f"C:/Users/dna10/animcv-w{n}") for n in (101, 102, 105, 113))
LOCK = Path("C:/Users/dna10/animcv-automation/state/active_batch.json")
BLENDER = Path("C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
INPUTS = {
    "w102_observations_predictions": (W102 / "results/historical_video/observations_predictions.npz",
                                      "6784357f7c6145e4b60e2e9201e32accd3ae00001486283a1e76f0adae61e7bd"),
    "w102_video_report": (W102 / "results/historical_video/report.json",
                          "e6e643071aadc1def75d86697acaeed56a0db6014c6620b42908a6359408b4fa"),
    "w102_video_inspection": (W102 / "results/video_inspection/inspection.json",
                              "dd25b7af0183754b7e78d07df9df71c7f718e43fb3f7489545d9c8b5c10d3149"),
    "repo_production_demo_clip": (ROOT / "examples/production_demo/source.mp4",
                                  "0dda8a5f16c67992330a1a8534cf0bfefeb5e0f2aec438577a9e1dbace773275"),
    "w102_hmr_pred_full": (W102 / "data/hmr_pred_full.npz",
                           "fc5d330565205f4792cd3641671633decbab4e24d70522d4dd85f045d87a7003"),
    "w102_j1x_predictions_3dpw": (W102 / "data/w102_j1x_predictions_3dpw.npz",
                                  "cb9dc51e3f84ddc9b45c5ada5e941ceaaad5ee73440a6e387ee0c3f2b689c46f"),
    "w105_evidence_manifest": (W105 / "evidence_manifest.json",
                               "cb0eca93c7847a70d497bd36c57856bd8295f3d8e3a7a72913ebf876ebe96899"),
    "w105_video_sources": (W105 / "results/build/video_sources.npz", "w105"),
    "w105_video_cases": (W105 / "results/build/video_cases.json", "w105"),
    "w105_cases": (W105 / "results/build/cases.json", "w105"),
    "w105_sources": (W105 / "results/build/sources.npz", "w105"),
    "w105_render_clip_audit": (W105 / "results/render_clip_audit.json", "w105"),
    "s0_checkpoint": (W101 / "data/s0_checkpoint.pt", "6c12fea9f84ea0493fada8b9acc96cd5c70fdd5d9114603e954f9412f4247b44"),
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
    "w113_declaration": (W113 / "results/declare/declaration.json",
                         "c3f7485c73fe9b9d6a6933fd9c1f45357b349b423177cdb248fe48c66258ef6b"),
    "w113_decision_frozen": (W113 / "results/decision_frozen/decision_frozen.json",
                             "2369289f604a574fba141ed43b5612a37fac6d78938e75e6bac82c6818555d09"),
    "w113_evidence_manifest": (W113 / "evidence_manifest.json",
                               "c4c1a6e70a848ea94533faa12c393654bd8c2d2956cdb3e52e80b28ed9491d72"),
}
BENCH_BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
REAL_SEQUENCE_ID = "w102_historical_video:0dda8a5f16c67992"
DELTA = (0.0, 0.0, 0.01)       # W113's declared mock-edit delta (metres, camera Z), reused unchanged
RIG_TOL = 1e-9                 # quaternion-component tolerance for "channel changed" (declared before any rig outcome)
ACTION_FPS = 30.0              # W105 RENDER_SETTINGS scene_fps (unchanged historical renderer)
RIGHT_ARM = ["right_shoulder", "right_elbow", "right_wrist"]
MIXED_WINDOWS = ("complete_control", "arm_natural_most")


def sha(p):
    return B.sha_file(p)


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    return v


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=1, sort_keys=True, allow_nan=False))


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def new_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=False)
    return p


def runtime():
    import numpy
    r = {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__}
    for mod in ("torch",):
        try:
            r[mod] = __import__(mod).__version__
        except Exception as e:  # pragma: no cover
            r[mod] = f"unavailable: {e}"
    return r


# ---------------------------------------------------------------------------------------------- preflight ---

def cmd_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"schema": "w114_preflight_v1", "stop": []}
    rep["head"] = git("rev-parse", "HEAD")
    rep["instruction_blob"] = git("rev-parse", "HEAD:docs/next_batch.md")
    rep["branch"] = git("rev-parse", "--abbrev-ref", "HEAD")
    rep["status_porcelain"] = git("status", "--porcelain")
    first = (ROOT / "docs/next_batch.md").read_text(encoding="utf-8").splitlines()[0]
    rep["instruction_title"] = first
    if rep["instruction_blob"] != INSTRUCTION_BLOB or not first.startswith("# BATCH 114"):
        rep["stop"].append("instruction is not the bound Batch 114 blob")
    if rep["branch"] != "arch/single_frame_first":
        rep["stop"].append("wrong branch")
    wl = (ROOT / W113_WORKLOG).read_text(encoding="utf-8").rstrip().splitlines()
    rep["w113_marker"] = wl[-2:] == ["AUTOMATION_BATCH_STATUS: COMPLETE", "BATCH_NUMBER: 113"]
    rep["w113_worklog_blob_at_start"] = git("rev-parse", f"{STARTING_HEAD}:{W113_WORKLOG}")
    rep["w113_worklog_blob_now"] = git("hash-object", str(ROOT / W113_WORKLOG))
    rep["w113_frozen_case"] = json.loads(INPUTS["w113_decision_frozen"][0].read_text()).get("case")
    if not rep["w113_marker"] or rep["w113_worklog_blob_at_start"] != rep["w113_worklog_blob_now"] or rep["w113_frozen_case"] != "D":
        rep["stop"].append("W113 historical record not intact")
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else None
    rep["main_local_lock"] = lock
    if not lock or lock.get("batch") != BATCH or lock.get("instruction_blob") != INSTRUCTION_BLOB \
            or lock.get("start_commit") != STARTING_HEAD:
        rep["stop"].append("Main-local lock is not this Batch 114 claim")
    m105 = json.loads(INPUTS["w105_evidence_manifest"][0].read_text())["files"]
    m113 = json.loads(INPUTS["w113_evidence_manifest"][0].read_text())["files"]
    files = {}
    for name, (path, want) in INPUTS.items():
        got = sha(path) if Path(path).exists() else None
        if want == "w105":
            rel = Path(path).relative_to(W105).as_posix()
            want = m105.get(rel, {}).get("sha256")
        files[name] = {"path": str(path), "sha256": got, "expected": want, "ok": got is not None and (want is None or got == want)}
        if not files[name]["ok"]:
            rep["stop"].append(f"input {name} missing or changed")
    rep["cross_checks"] = {
        "w113_manifest_declaration": m113["results/declare/declaration.json"]["sha256"] == files["w113_declaration"]["sha256"],
        "w113_manifest_decision_frozen": m113["results/decision_frozen/decision_frozen.json"]["sha256"] == files["w113_decision_frozen"]["sha256"],
        "w102_report_binds_npz": json.loads(INPUTS["w102_video_report"][0].read_text())["observation_prediction_sha256"]
        == files["w102_observations_predictions"]["sha256"],
        "blender_5_1_present": BLENDER.exists(),
    }
    from run_real_observation_s0 import S0_CHECKPOINT_SHA
    rep["cross_checks"]["s0_checkpoint_identity"] = files["s0_checkpoint"]["sha256"] == S0_CHECKPOINT_SHA
    for k, ok in rep["cross_checks"].items():
        if not ok:
            rep["stop"].append(f"cross-check failed: {k}")
    rep["inputs"] = files
    rep["runtime"] = runtime()
    rep["pass"] = not rep["stop"]
    dump(out / "preflight.json", rep)
    print(json.dumps({"pass": rep["pass"], "stop": rep["stop"]}))
    return 0 if rep["pass"] else 2


# ----------------------------------------------------------------------------------------------- declaration ---

# Static producer -> consumer ownership (code facts; file:line at the starting commit). Status vocabulary:
# PRESENT / DERIVED / LOST / UNKNOWN / UNAVAILABLE.
OWNERSHIP = [
    ("raw 2D xy", "MMPose adapter -> PoseSequence -> MotionGraph", "PRESENT", "src/pose/mmpose_adapter.py:280-339; src/motion/motion_builder.py:27-40"),
    ("raw 2D xy", "MotionGraph -> AnimationClip", "LOST", "src/retarget/fk_solver.py:108-158"),
    ("raw 2D xy of invalid joints", "real observation -> S0 input", "LOST (zero-filled)", "scripts/w101_reference_estimator.py:60-75"),
    ("detector score", "adapter -> PoseSequence -> MotionGraph", "PRESENT (raw, values > 1 kept)", "src/motion/motion_builder.py:27-40"),
    ("detector score", "FrameBank -> LiftedPoseSequence", "DERIVED (depth_uncertainty = 1 - score; invalid -> 0)", "src/pose/framepose_bridge.py:101-110"),
    ("detector score", "MotionGraph -> AnimationClip", "DERIVED (min over bone joints; 0.0 = held rotation)", "src/retarget/fk_solver.py:108-158"),
    ("detector score", "W105 rig Action", "LOST", "scripts/w105_rig_utility.py:657-735"),
    ("observed vs derived 2D joint", "adapter", "LOST (midpoints/renamed joints carry no flag)", "src/pose/mmpose_adapter.py:280-339"),
    ("observation validity", "W102 NPZ input_valid -> S0 input / bridge", "PRESENT", "scripts/w101_reference_estimator.py:60-75"),
    ("observation validity", "W105 rig Action", "LOST", "scripts/w105_rig_utility.py:657-735"),
    ("S0 3D output", "S0 -> Lifted -> AnimationSemantics", "PRESENT (camera_root_relative, metres)", "src/motion/animation_semantics_v2_bridge.py:101-121"),
    ("S0 3D output", "MotionGraph / CLI AnimationClip", "UNAVAILABLE (position_3d mixed-unit or None)", "src/motion/motion_builder.py:33-37"),
    ("S0 3D output", "W105 real-clip Action", "DERIVED (rotations; torso frame, leg IK, left-arm IK)", "scripts/w105_rig_utility.py:657-720"),
    ("3D reliability", "everywhere", "UNAVAILABLE", "src/pose/framepose_bridge.py:107"),
    ("source/model provenance", "PoseSequence -> MotionGraph", "LOST (source_metadata = pose path only)", "src/motion/motion_graph.py:96-121"),
    ("source/model provenance", "AnimationClip / W105 Action", "LOST (name Generated_Motion / W91 action name)", "src/retarget/solver.py:66-96"),
    ("frame index", "-> AnimationClip / W105 Action", "PRESENT as key index (Action: frame_index + 1)", "scripts/render_runtime_heading_rig_blender.py:148-150"),
    ("timestamp", "-> AnimationClip", "LOST (derivable from fps x index)", "src/retarget/solver.py:66-96"),
    ("timestamp", "-> W105 Action", "LOST (scene 30 fps vs 50 fps source)", "scripts/render_runtime_heading_rig_blender.py:140"),
    ("per-frame tracked subject", "MMPose adapter", "LOST (produced internally, never serialized)", "src/pose/mmpose_adapter.py:189-232"),
    ("sequence identity", "W102 report", "PRESENT (clip sha256 + decoded frame digests)", "W102 report.json"),
    ("user preserve/permit", "everywhere", "UNAVAILABLE (MotionFrame.locked has no producer)", "src/motion/motion_builder.py:42-50"),
    ("bone/channel source ownership", "W105 rig", "UNKNOWN until measured (W114 Part 4)", "scripts/w105_rig_utility.py; scripts/w102_rig_review.py"),
]

# name -> (expected code, what it exercises). Applies to every population unless marked NOT APPLICABLE at run time
# with a stated reason (never counted as PASS).
CONTROLS = {
    "positive_ingest_unchanged": ("PASS", "unchanged evidence verifies; finite scores > 1 must not invalidate"),
    "positive_valid_edit": ("ACCEPT", "a permitted, fresh, well-formed mock edit is applied"),
    "unauthorized_edit": ("EDIT_NOT_AUTHORIZED", "target a default-prohibited cell"),
    "preserve_permit_conflict": ("LOCK_CONFLICT", "constraints permitting a preserved cell"),
    "edit_on_preserved_cell": ("LOCK_CONFLICT_EDIT", "target a preserved cell"),
    "stale_source_digest": ("STALE_EVIDENCE", "edit bound to another evidence digest"),
    "stale_edit_parent": ("STALE_PARENT", "edit prepared against a superseded state"),
    "missing_source_evidence": ("EVIDENCE_MISSING", "evidence without its raw score array"),
    "missing_source_provenance": ("PROVENANCE_MISSING", "evidence source without sha256"),
    "mismatched_frame_index": ("FRAME_MISMATCH", "target frame not in the evidence"),
    "mismatched_timestamp": ("TIME_MISMATCH", "target timestamp off by one source frame period"),
    "subject_identity_ambiguity": ("IDENTITY_UNVERIFIED", "cross-frame copy over frames without verified continuity"),
    "duplicate_joint_identity": ("JOINT_DUPLICATE", "evidence with a duplicated joint name"),
    "coordinate_unit_mismatch": ("COORDINATE_MISMATCH", "edit declared in pixels"),
    "edit_on_immutable_observation": ("OBSERVATION_IMMUTABLE", "edit targeting the raw score signal"),
    "modified_immutable_observation": ("EVIDENCE_MODIFIED", "evidence score changed after ingest (digest not updated)"),
    "unknown_to_zero_coercion": ("UNKNOWN_COERCED_TO_ZERO", "absent observation set to 0 (needs an UNKNOWN cell)"),
    "manufactured_3d_confidence": ("CONFIDENCE_MANUFACTURED_3D", "evidence carrying est3d_reliability = 1 - score"),
    "nonfinite_score_on_present": ("SCORE_MISSING", "present observation with NaN score"),
    "score_clipped_to_unit_interval": ("SOURCE_MISMATCH", "scores clipped to <= 1 vs source (needs a score > 1)"),
    "invalid_source_to_rig_association": ("ASSOCIATION_INVALID", "association claiming a channel owner the rig does not use"),
    "source_change_no_rig_influence": ("NO_RIG_INFLUENCE", "rig-claimed edit of a source the rig does not consume"),
    "output_change_outside_authorized_domain": ("OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN", "permitted source edit whose observed rig change depicts non-permitted joints"),
    "reversal_not_restoring_parent": ("REVERT_MISMATCH", "revert record with tampered restore values"),
    "edit_on_unknown_target": ("EDIT_UNKNOWN_TARGET", "permitted but non-finite 3D cell (needs an UNKNOWN 3D cell)"),
}
RIG_CONTROLS = ("invalid_source_to_rig_association", "source_change_no_rig_influence", "output_change_outside_authorized_domain")

# Frozen-rig perturbations. Rows perturbed: even frame_index (real) / even window position (mixed); others are
# temporal-locality controls. delta = DELTA unless stated. `expect` = channels predicted to change (code analysis of
# torso_frames / two-bone IK / W102 contract BEFORE any perturbed output exists). `authority` = expected rig-enforced
# outcome when the edit is permitted on exactly the perturbed cells and claims rig influence (real clip only).
RIG_CONDITIONS = {
    "real": {
        "A1_right_wrist": {"joints": ["right_wrist"], "kind": "A", "expect": [], "authority": "NO_RIG_INFLUENCE",
                           "why": "right arm is held at rest in the W105 real-clip configuration"},
        "A2_head": {"joints": ["head"], "kind": "A", "expect": [], "authority": "NO_RIG_INFLUENCE",
                    "why": "no channel consumes head/neck/spine"},
        "B1_left_wrist": {"joints": ["left_wrist"], "kind": "B", "expect": ["upperarm_l", "lowerarm_l"],
                          "authority": "OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN",
                          "why": "arm_export two-bone IK targets the wrist: the upper-arm (elbow placement) also moves"},
        "B2_left_elbow_and_wrist": {"joints": ["left_elbow", "left_wrist"], "kind": "B", "expect": ["upperarm_l", "lowerarm_l"],
                                    "authority": "ACCEPT", "why": "both depicted arm joints permitted"},
        "C1_right_shoulder_along_trunk_axis": {"joints": ["right_shoulder"], "kind": "C", "along_trunk_axis": True, "expect": [],
                                               "authority": "NO_RIG_INFLUENCE",
                                               "why": "torso_frames projects the bilateral axis perpendicular to u = thorax - pelvis; "
                                                      "W113's conservative map lists right_shoulder for all 8 channels"},
        "D1_thorax": {"joints": ["thorax"], "kind": "D", "expect": list(R.CHANNELS), "authority": "OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN",
                      "why": "u = thorax - pelvis defines both torso frames: Object, spine, legs (pelvis frame) and arm (thorax frame)"},
        "D2_left_shoulder": {"joints": ["left_shoulder"], "kind": "D", "expect": ["spine_01", "upperarm_l", "lowerarm_l"],
                             "authority": "OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN",
                             "why": "shoulder enters the thorax frame and the arm vectors, not the pelvis frame or legs"},
    },
    "mixed": {
        "MA1_s0_left_wrist": {"source": "s0", "joint": "left_wrist", "kind": "A", "expect": [], "why": "J1X owns the arm in the W102 contract"},
        "MA2_s0_left_shoulder": {"source": "s0", "joint": "left_shoulder", "kind": "A", "expect": [], "why": "HMR2 owns the torso frames"},
        "MB1_s0_left_knee": {"source": "s0", "joint": "left_knee", "kind": "B", "expect": ["thigh_l", "calf_l"], "why": "S0 legs through two-bone IK"},
        "MB2_j1x_left_wrist": {"source": "j1x_wrist", "kind": "B", "expect": ["upperarm_l", "lowerarm_l"], "why": "J1X arm through the arm chain"},
        "MD1_hmr_left_shoulder": {"source": "hmr_j24", "smpl_index": 16, "kind": "D", "expect": ["spine_01", "upperarm_l", "lowerarm_l"],
                                  "why": "HMR2 joints feed the thorax frame (spine_01 relative to global_orient pelvis, arm frame)"},
    },
}

DECISION = {
    "precedence": "D > C > B > A",
    "validity_D": {
        "V1_inputs_authenticated": "preflight pass and every build re-verifies input hashes",
        "V2_w113_intact": "W113 Worklog blob unchanged, marker present, frozen decision CASE D, W113 declaration/decision hashes",
        "V3_evidence_source_exact": "real and benchmark evidence equal their stored source arrays byte-for-byte (scores incl. > 1)",
        "V4_positive_ingest": "unchanged real, benchmark and synthetic evidence verify; a finite score > 1 never invalidates",
        "V5_rig_reproduction": "replicated frozen rig reproduces W105 video_cases FIXED slot (all 199 frames) and both mixed windows; "
                               "Blender FIXED Actions reproduce the W105 FIXED value digest",
        "V6_repeatability": "two independent builds: identical evidence/state/association digests and rig outcomes",
        "V7_synthetic_controls": "every applicable synthetic control returns exactly its code, atomically",
        "V8_rig_positive_control": "in each configuration at least one declared B or D perturbation changes a channel",
    },
    "core_C": {
        "G1_evidence_immutable": "evidence digest, arrays and source file hashes unchanged after every real/benchmark operation; "
                                 "in-place write to evidence refused",
        "G2_edit_locality": "accepted edits change exactly their targets; 0 unintended cells; originals equal parent values",
        "G3_reversal": "reverts restore the parent state byte-exactly; lineage ids recompute; tampered revert refused",
        "G4_stale_rejection": "stale parent and stale evidence rejected with head/state count unchanged",
        "G5_permission": "permitted set non-vacuous; unauthorized / preserved / conflicting operations rejected atomically; "
                         "every contract (non-rig) programme step returns its declared code",
        "G6_score_semantics": "raw scores preserved exactly incl. > 1; no 3D reliability manufactured; UNKNOWN never zero",
        "G7_controls_real_bench": "every applicable real/benchmark control returns exactly its code, atomically",
        "G8_rig_influence_distinguished": "observed rig influence is measured for every real/mixed condition; unperturbed rows "
                                          "bit-identical; every candidate rig-authority outcome (conditions, programme rig steps, "
                                          "applicable real rig controls) equals the independent re-derivation from observed influence "
                                          "and is atomic; Blender Action changes equal case-level changes. Mismatches between "
                                          "predicted and observed channel sets are reported, not gated",
    },
    "boundary_B_triggers": {
        "T1_identity": "real per-frame subject continuity unverified on >= 1 frame, so cross-frame operations fail closed on authentic data",
        "T2_time": "W105 Action time differs from source time on >= 1 key and the unchanged Action carries no source-time field",
        "T3_authority": "on the real clip the single-joint perturbation B1 (left_wrist) changes at least one channel whose "
                        "depicted joints are not {left_wrist}: joint permission alone does not confer bone authority",
    },
    "case_rule": "D if any V fails; else C if any G fails; else B if any T holds; else A",
    "non_claims": "no CASE implies Generator quality, reconstruction correctness, calibrated reliability or production readiness",
}


def cmd_declare(a):
    pre = json.loads(Path(a.preflight).read_text())
    if not pre.get("pass"):
        raise RuntimeError("STOP: preflight did not pass")
    out = new_dir(a.out_dir)
    decl = {
        "schema": "w114_declaration_v1", "batch": BATCH, "instruction_blob": INSTRUCTION_BLOB, "starting_head": STARTING_HEAD,
        "outcomes_inspected": False,
        "development_disclosure": "Before this declaration the instrument was developed against synthetic data and the "
                                  "UNPERTURBED frozen rig was replayed to check byte reproduction of W105 (real clip 199/199, "
                                  "complete_control 24/24, arm_natural_most 24/24). No perturbed rig output, real/benchmark "
                                  "edit, control or candidate outcome was computed before this file was hashed.",
        "inputs": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in pre["inputs"].items()},
        "regimes": {"real": "real_animcv_observation (W102 M1 MMPose+RTMDet, GT-free, private, not unseen)",
                    "benchmark": "benchmark_detector_observation (3DPW shipped detector; first VALIDATION sequence; never pooled)",
                    "mixed_rig_windows": "real_animcv_observation on 3DPW images (W102 M_Main S0) + frozen HMR2 + frozen J1X; "
                                         "rig ownership probe only; identity labels never read",
                    "synthetic": "oracle_geometry synthetic contract sequence (8 frames, 25 fps), contract test only"},
        "hypothesis_H1": "Immutable source evidence + versioned editable state can establish a minimum contract for controlled "
                         "editing without changing the production MotionGraph schema.",
        "control": "unchanged existing path: W113 legacy CLI replay (PoseSequence -> MotionGraph -> AnimationClip), W113 FramePose "
                   "bridge replay, W105 frozen rig and Action",
        "candidate": "scripts/w114_source_preserving_base.py (scratch only)",
        "ownership_static": [dict(zip(("field", "boundary", "status", "location"), r)) for r in OWNERSHIP],
        "score_contract": dict(B.SCORE_SEMANTICS, fail_safe={
            "absent": "UNKNOWN: xy and score NaN, present=False; any value -> UNKNOWN_COERCED(_TO_ZERO)",
            "present_nonfinite_score": "evidence rejected (SCORE_MISSING); no operation proceeds",
            "3d_reliability": "UNAVAILABLE; any array -> CONFIDENCE_MANUFACTURED_3D"}),
        "subject_identity": {"real": "per-frame association UNAVAILABLE (never serialized): continuity verified on 0 frames",
                             "benchmark": "dataset actor label in sequence_id, not runtime-verified: verified on 0 frames",
                             "synthetic": "known; frame 5 declared unverified to exercise the control",
                             "cross_frame_rule": "operations whose replacement at frame f is computed from another frame "
                                                 "(copy_from_frame) require verified continuity on every involved frame"},
        "directives": {
            "RS1": "W113 RS1 unchanged: preserve all joints frames 0-99; preserve right arm all frames; permit left_wrist on "
                   "maximal runs where input_valid is false and frame >= 100; default prohibit",
            "RS1_arm": "RS1 with the permit widened to {left_elbow, left_wrist} on the same runs",
            "RS_head": "RS1 plus permit head on the same runs (an unconsumed source for the no-influence control)",
            "RS2": "RS1 plus permit right_knee frames 90-110 -> LOCK_CONFLICT",
            "RSB": "benchmark: preserve every input-supported 3D cell, permit every input-unsupported cell (per-joint runs)",
            "SS1": "synthetic: preserve right arm all frames and all joints frames 0-1; permit left_wrist 3-5, left_elbow 4, "
                   "head 3-5",
            "SS1_arm": "synthetic: permit left_wrist frames 3-5 and left_elbow frames 3 and 5 (frame 4 elbow is UNKNOWN 3D)"},
        "edit_programme": {
            "real": [
                ["R1", "E1 offset DELTA on all RS1-permitted cells (source-level)", "ACCEPT"],
                ["R2", "E2 offset (0.005, 0, 0) on RS1-permitted cells in frames 109-118", "ACCEPT"],
                ["R3", "E3 = E2 cells prepared against the state after E1 (stale parent)", "STALE_PARENT"],
                ["R4", "E4 bound to evidence digest f*64", "STALE_EVIDENCE"],
                ["R5", "E5 copy left_wrist 108 -> 109 (cross-frame)", "IDENTITY_UNVERIFIED"],
                ["R6", "revert E2 (state equals post-E1 bytes)", "ACCEPT"],
                ["R7", "tampered revert of E1", "REVERT_MISMATCH"],
                ["R8", "revert E1 (state equals hypothesis bytes)", "ACCEPT"],
                ["R9", "rig-enforced E1 under RS1 (PREDICTION; graded by consistency with observed influence)", "OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN"],
                ["R10", "rig-enforced E1 under RS1_arm (PREDICTION; consistency-graded)", "ACCEPT"],
                ["R11", "revert R10 if accepted (rig channels bit-equal to the hypothesis channels)", "ACCEPT"],
                ["R12", "rig-claimed head edit on RS_head runs (PREDICTION; consistency-graded)", "NO_RIG_INFLUENCE"],
                ["R13", "unclaimed rig-enforced head edit on RS_head runs (PREDICTION; consistency-graded)", "ACCEPT"],
                ["R14", "revert R13 if accepted", "ACCEPT"]],
            "rig_step_grading": "rig-dependent outcomes are predictions from code analysis; the candidate is graded on whether "
                                "its outcome equals an independent re-derivation of the declared authority rule from the observed "
                                "influence (separate code path); prediction mismatches are reported and explained, not hidden",
            "benchmark": [
                ["B1", "EB1 offset DELTA on all RSB-permitted cells", "ACCEPT (vacuous if none)"],
                ["B2", "edit a preserved cell", "LOCK_CONFLICT_EDIT"],
                ["B3", "EB1 cells prepared against the root after EB1 (stale parent)", "STALE_PARENT"],
                ["B4", "copy from previous row (cross-frame)", "IDENTITY_UNVERIFIED"],
                ["B5", "revert EB1 (state equals hypothesis bytes)", "ACCEPT"]],
            "synthetic": "same structure as real on SS1/SS1_arm with a declared toy rig (channels torso, upper_l (uses elbow AND "
                         "wrist), lower_l, head unconsumed)"},
        "controls": {k: {"expected": v[0], "exercise": v[1]} for k, v in CONTROLS.items()},
        "rig_controls_population": "real (frozen W105 S0-only rig) and synthetic (toy rig); benchmark NOT APPLICABLE (no frozen "
                                   "rig configuration consumes the benchmark-detector S0 predictions)",
        "rig": {"tolerance_quaternion_component": RIG_TOL, "delta_m": DELTA, "perturbed_rows": "even frame_index / even window position",
                "channels": list(R.CHANNELS), "depicts": R.DEPICTS, "owners": R.OWNERS, "w113_dependency_map": R.W113_DEPENDENCIES,
                "conditions": RIG_CONDITIONS, "mixed_windows": list(MIXED_WINDOWS),
                "authority_rule": "a changed channel-frame is authorized iff every joint it depicts (tail-joint projection) is "
                                  "permitted and none is preserved at that frame; fixed/rest channels may not change; a rig-claimed "
                                  "edit with no changed channel is NO_RIG_INFLUENCE",
                "blender": {"settings": "W105 render (size 480, ortho 3.2, show mesh, equivalence 1e-4, FIXED,RUNTIME)",
                            "comparison": "Action channel-frame changed iff any of its 4 F-curve values differs (exact)",
                            "expected": "Action-level changed channel-frames equal case-level changed channel-frames"}},
        "time": {"source_fps": 50.0, "action_fps": ACTION_FPS, "action_key": "frame_index + 1",
                 "mismatch_rule": "|frame_index/30 - timestamp| > 1e-9 s counts as a mismatched key"},
        "decision": DECISION,
    }
    dump(out / "declaration.json", decl)
    print(sha(out / "declaration.json"))
    return 0


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


# ------------------------------------------------------------------------------------------------- populations ---

def synthetic_evidence(tmp: Path):
    T, fps = 8, 25.0
    rest = {"pelvis": (0, 0, 0), "left_hip": (-.1, 0, 0), "left_knee": (-.1, 0, -.45), "left_ankle": (-.1, 0, -.9),
            "right_hip": (.1, 0, 0), "right_knee": (.1, 0, -.45), "right_ankle": (.1, 0, -.9), "spine": (0, 0, .25),
            "thorax": (0, 0, .5), "neck": (0, 0, .55), "head": (0, 0, .7), "left_shoulder": (-.18, 0, .5),
            "left_elbow": (-.2, 0, .25), "left_wrist": (-.2, 0, 0), "right_shoulder": (.18, 0, .5),
            "right_elbow": (.2, 0, .25), "right_wrist": (.2, 0, 0)}
    est = np.repeat(np.array([rest[n] for n in B.JOINTS], np.float64)[None], T, 0)
    for t in range(T):
        est[t, B.JI["left_wrist"], 1] = 0.2 * math.sin(t / 2)
        est[t, B.JI["right_wrist"], 1] = -0.2 * math.sin(t / 2)
    W, H, f, depth = 1280, 720, 1000.0, 3.0
    xy = np.stack([f * est[..., 0] / (est[..., 1] + depth) + W / 2, H / 2 - f * est[..., 2] / (est[..., 1] + depth)], -1)
    score = np.full((T, B.J), 0.9)
    present = np.ones((T, B.J), bool)
    score[3:6, B.JI["left_wrist"]] = 0.2
    score[2, B.JI["left_knee"]] = 1.07          # finite score above 1: must be preserved, must not invalidate
    present[6, B.JI["left_ankle"]] = False      # UNKNOWN observation
    xy[~present] = np.nan
    score[~present] = np.nan
    valid = present & (score >= 0.3)
    est = est.astype(np.float32)
    est[4, B.JI["left_elbow"]] = np.nan         # UNKNOWN 3D cell
    np.savez(tmp / "synthetic_source.npz", xy=xy, score=score, est=est, valid=valid)
    cont = np.ones(T, bool)
    cont[5] = False
    ev = B.build_evidence(
        sequence_id="synthetic:w114_contract_sequence", regime="oracle_geometry", backend="synthetic_projection",
        image_size=(W, H), fps=fps, fps_source="synthetic declaration", frame_index=np.arange(T), timestamp=np.arange(T) / fps,
        obs2d_xy_px=xy, obs2d_score_raw=score, obs2d_present=present, obs2d_visible=valid, input_valid=valid,
        obs2d_kind={n: "observed_2d" for n in B.JOINTS}, est3d_xyz=est, est3d_model={"model": "synthetic known 3D"},
        sources={"synthetic_source": {"path": str(tmp / "synthetic_source.npz"), "sha256": sha(tmp / "synthetic_source.npz")}},
        subject={"sequence_subject": "synthetic_actor_0", "per_frame_association": "synthetic known; frame 5 declared unverified"},
        subject_continuity_verified=cont, validity_rule="score >= 0.3 (synthetic)",
        notes={"label": "SYNTHETIC CONTRACT SEQUENCE, not animation evidence"})
    return ev, {"obs2d_score_raw": score, "obs2d_xy_px": xy, "est3d_xyz_hypothesis": est, "input_valid": valid}


def toy_rig():
    """Declared synthetic rig: torso <- (thorax - pelvis); upper_l <- (elbow - shoulder, |wrist - shoulder|) [IK-like
    coupling]; lower_l <- (wrist - elbow); head is consumed by nothing."""
    def fn(v):
        v = np.asarray(v, np.float64)
        s, e, w = v[:, B.JI["left_shoulder"]], v[:, B.JI["left_elbow"]], v[:, B.JI["left_wrist"]]
        pad = lambda x, k: np.concatenate([x, k[:, None]], -1)
        return {"torso": pad(v[:, B.JI["thorax"]] - v[:, B.JI["pelvis"]], np.ones(len(v))),
                "upper_l": pad(e - s, np.linalg.norm(w - s, axis=-1)), "lower_l": pad(w - e, np.ones(len(v)))}
    return B.RigModel("synthetic_toy_rig", fn, {"torso": ["pelvis", "thorax"], "upper_l": ["left_elbow"], "lower_l": ["left_wrist"]},
                      {"torso": "synthetic thorax/pelvis", "upper_l": "synthetic left arm", "lower_l": "synthetic left arm"}, tol=RIG_TOL)


def real_rig(rc):
    return B.RigModel("w105_real_s0_only", lambda v: R.real_channels(v, rc), R.DEPICTS, R.OWNERS["real_s0_only"], tol=RIG_TOL)


def real_inputs(decl):
    z = dict(np.load(decl["inputs"]["w102_observations_predictions"]["path"], allow_pickle=False))
    rep = json.loads(Path(decl["inputs"]["w102_video_report"]["path"]).read_text())
    insp = json.loads(Path(decl["inputs"]["w102_video_inspection"]["path"]).read_text())
    if rep["decoded_BGR_uint8_sha256"] != insp["decoded_BGR_uint8_sha256"] or rep["clip_sha256"] != insp["clip_sha256"] \
            or rep["clip_sha256"] != decl["inputs"]["repo_production_demo_clip"]["sha256"]:
        raise RuntimeError("STOP: W102 clip identity")
    if not np.array_equal(z["timestamp"], np.asarray(insp["timestamp"])) or len(z["timestamp"]) != rep["frames"]:
        raise RuntimeError("STOP: timestamp identity")
    return z, rep, insp


def real_evidence(decl, z, rep, insp):
    import w113_animation_base_view as V
    lm = z["landmarks"]
    rows = [V.CANON.index(V.CANON_OF[n]) for n in B.JOINTS]
    raw = lm[:, rows]
    present = np.isfinite(raw[..., 0])
    ev = B.build_evidence(
        sequence_id=REAL_SEQUENCE_ID, regime="real_animcv_observation", backend="mmpose", image_size=tuple(insp["size"]),
        fps=float(rep["fps"]), fps_source="W102 report fps = inspection fps", frame_index=z["frame_index"], timestamp=z["timestamp"],
        obs2d_xy_px=raw[..., :2], obs2d_score_raw=raw[..., 2], obs2d_present=present,
        obs2d_visible=present & (np.nan_to_num(raw[..., 3]) > .5), input_valid=z["input_valid"], obs2d_kind=V.OBS2D_KIND,
        est3d_xyz=z["predictions"],
        est3d_model={"model": "frozen S0_neutral_sign (W102 run)", "checkpoint_sha256": decl["inputs"]["s0_checkpoint"]["sha256"],
                     "frame": "camera_root_relative", "unit": "metres"},
        sources=source(decl, "w102_observations_predictions", "w102_video_report", "w102_video_inspection",
                       "repo_production_demo_clip", "s0_checkpoint"),
        subject={"sequence_subject": "single_tracked_subject (Architecture_v2 1.3 single-character contract)",
                 "per_frame_association": "UNAVAILABLE: SubjectTracker selection is produced inside "
                                          "mmpose_adapter._process_tracked_sequence and never serialized; W102 keeps only "
                                          "aggregates (tracked 199/199, mean 1.76 person candidates per frame)"},
        subject_continuity_verified=np.zeros(len(z["frame_index"]), bool),
        validity_rule=f"W102 input_valid (adapter visibility at score >= {rep['M1_identity']['config']['visibility_threshold']}, "
                      "derived joints from their sources); recorded, not recomputed",
        frame_digests=rep["decoded_BGR_uint8_sha256"],
        notes={"rights": rep["rights"], "unseen": "false", "gt": "none exists for this clip"})
    ref = {"obs2d_score_raw": raw[..., 2], "obs2d_xy_px": raw[..., :2], "est3d_xyz_hypothesis": z["predictions"],
           "input_valid": z["input_valid"], "frame_index": z["frame_index"], "timestamp": z["timestamp"]}
    return ev, ref


def bench_evidence(decl):
    from framepose.contract import FrameBank
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
    pos = np.array([i for i in val if bank.samples[int(i)].sequence_id == seq])
    pos = pos[np.argsort([bank.samples[int(p)].frame_index for p in pos])]
    s0 = bank.samples[int(pos[0])]
    w, h = s0.image_size
    x = bank.arrays["input_2d"][pos]
    v = bank.arrays["input_valid"][pos]
    xy = np.where(v[..., None], x[..., :2].astype(np.float64) * [w, h], np.nan)
    score = np.where(v, x[..., 2].astype(np.float64), np.nan)
    ev = B.build_evidence(
        sequence_id=seq, regime=s0.observation.regime, backend=s0.observation.backend, image_size=(w, h), fps=float(s0.fps),
        fps_source="FrameSample.fps", frame_index=[bank.samples[int(p)].frame_index for p in pos],
        timestamp=[bank.samples[int(p)].timestamp for p in pos], obs2d_xy_px=xy, obs2d_score_raw=score, obs2d_present=v,
        obs2d_visible=v, input_valid=v, obs2d_kind={n: "benchmark_shipped_detector_2d" for n in B.JOINTS}, est3d_xyz=h0[pos],
        est3d_model={"model": "frozen S0 VALIDATION prediction (W101/W102)", "frame": "camera_root_relative", "unit": "metres"},
        sources=source(decl, "bench_bank_index", "bench_bank_arrays", "bench_frozen_prediction_validation"),
        subject={"sequence_subject": seq.rsplit(":", 1)[-1],
                 "per_frame_association": "dataset actor label in sequence_id; not verified by an AnimCV runtime tracker"},
        subject_continuity_verified=np.zeros(len(pos), bool),
        validity_rule="bank input_valid (benchmark adapter); bank keeps no raw below-threshold coordinate",
        notes={"xy": "obs2d_xy_px = bank normalized xy x image size (derived unit conversion); score copied exactly",
               "excluded": "FrameSample stratification (computed from target_3d) and target arrays are never read"})
    inv = ~v
    placeholder = {"invalid_cells": int(inv.sum()), "bank_xy_zero_at_invalid": int((x[inv][:, :2] == 0).all(-1).sum()),
                   "bank_score_zero_at_invalid": int((x[inv][:, 2] == 0).sum()),
                   "bank_values_finite_at_invalid": int(np.isfinite(x[inv]).all(-1).sum())}
    ref = {"obs2d_score_raw": score, "est3d_xyz_hypothesis": h0[pos], "input_valid": v}
    return ev, ref, {"sequence_id": seq, "rows": int(len(pos)), "split": s0.split, "placeholder_values_in_bank": placeholder,
                     "frame_stride": sorted(set(np.diff([bank.samples[int(p)].frame_index for p in pos]).tolist()))}


# --------------------------------------------------------------------------------------------- directive sets ---

def runs_of(mask, fi):
    out, i, m = [], 0, np.asarray(mask, bool)
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            out.append([int(fi[i]), int(fi[j])])
            i = j + 1
        else:
            i += 1
    return out


def real_directives(ev):
    a = ev["arrays"]
    fi = a["frame_index"]
    lw = ~a["input_valid"][:, B.JI["left_wrist"]] & (fi >= 100)
    spans = runs_of(lw, fi)
    base = [{"id": "P_ref_span", "kind": "preserve", "joints": list(B.JOINTS), "spans": [[0, 99]]},
            {"id": "P_right_arm", "kind": "preserve", "joints": RIGHT_ARM, "spans": [[int(fi[0]), int(fi[-1])]]}]
    sets = {
        "RS1": base + [{"id": "E_lw", "kind": "permit", "joints": ["left_wrist"], "spans": spans}],
        "RS1_arm": base + [{"id": "E_arm", "kind": "permit", "joints": ["left_elbow", "left_wrist"], "spans": spans}],
        "RS_head": base + [{"id": "E_lw", "kind": "permit", "joints": ["left_wrist"], "spans": spans},
                           {"id": "E_head", "kind": "permit", "joints": ["head"], "spans": spans}],
        "RS2": base + [{"id": "E_lw", "kind": "permit", "joints": ["left_wrist"], "spans": spans},
                       {"id": "E_conflict", "kind": "permit", "joints": ["right_knee"], "spans": [[90, 110]]}],
    }
    cells = [(int(f), "left_wrist") for f in fi[lw]]
    return {k: B.make_constraints(ev, k, v) for k, v in sets.items()}, cells, spans


def bench_directives(ev):
    a = ev["arrays"]
    fi = a["frame_index"]
    sup = a["est3d_support"]
    dirs = []
    for j, n in enumerate(B.JOINTS):
        for kind, code in (("preserve", 1), ("permit", 2)):
            sp = runs_of(sup[:, j] == code, fi)
            if sp:
                dirs.append({"id": f"{kind}_{n}", "kind": kind, "joints": [n], "spans": sp})
    cells = [(int(fi[r]), B.JOINTS[c]) for r, c in np.argwhere(sup == 2)]
    return B.make_constraints(ev, "RSB", dirs), cells


def synth_directives(ev):
    sets = {
        "SS1": [{"id": "P_right_arm", "kind": "preserve", "joints": RIGHT_ARM, "spans": [[0, 7]]},
                {"id": "P_first", "kind": "preserve", "joints": list(B.JOINTS), "spans": [[0, 1]]},
                {"id": "E_lw", "kind": "permit", "joints": ["left_wrist"], "spans": [[3, 5]]},
                {"id": "E_le", "kind": "permit", "joints": ["left_elbow"], "spans": [[4, 4]]},
                {"id": "E_head", "kind": "permit", "joints": ["head"], "spans": [[3, 5]]}],
        "SS1_arm": [{"id": "E_lw", "kind": "permit", "joints": ["left_wrist"], "spans": [[3, 5]]},
                    {"id": "E_le", "kind": "permit", "joints": ["left_elbow"], "spans": [[3, 3], [5, 5]]}],
    }
    return {k: B.make_constraints(ev, k, v) for k, v in sets.items()}


# ---------------------------------------------------------------------------------------------------- runner ---

def pristine(ev):
    return {"digest": ev["meta"]["evidence_digest"], "bytes": {k: v.tobytes() for k, v in ev["arrays"].items()},
            "meta": B.canonical({k: v for k, v in ev["meta"].items()})}


def evidence_unchanged(ev, snap) -> bool:
    return (ev["meta"]["evidence_digest"] == snap["digest"] == B.evidence_digest(ev)
            and all(ev["arrays"][k].tobytes() == b for k, b in snap["bytes"].items())
            and B.canonical(ev["meta"]) == snap["meta"])


def attempt(fn, lin=None, ev=None, snap=None):
    """Run one operation; capture ACCEPT/PASS or the ContractError code; verify atomicity on rejection."""
    before = lin.snapshot() if lin is not None else None
    try:
        res = fn()
        code = "ACCEPT" if lin is not None else "PASS"
    except B.ContractError as e:
        res, code = str(e), e.code
    rec = {"observed": code}
    if lin is not None and code not in ("ACCEPT", "PASS"):
        rec["atomic_head_unchanged"] = lin.snapshot() == before
    if ev is not None and snap is not None:
        rec["evidence_unchanged"] = evidence_unchanged(ev, snap)
    return code, res, rec


def mutated(ev, fn, redigest=True):
    v = {"meta": copy.deepcopy(ev["meta"]), "arrays": {k: np.array(x, copy=True) for k, x in ev["arrays"].items()}}
    fn(v)
    if redigest:
        v["meta"]["evidence_digest"] = B.evidence_digest(v)
    return v


def source_level_after(ev, cs, rec):
    """Values the edit would produce, applied source-level on a throwaway lineage at the same head (no rig check)."""
    lin = B.Lineage(ev)
    r = dict(rec, base_state_id=lin.head)
    try:
        return B.apply_edit(lin, ev, cs, r)["values"]
    except B.ContractError:
        return None


def independent_authority(rig, ev, cs, before_vals, after_vals, rig_claim) -> str:
    """Re-derive the declared authority rule from observed influence with separate code (min over q / -q distance)."""
    if after_vals is None:
        return "SOURCE_LEVEL_REJECT"
    m = B.resolve_constraints(ev, cs)
    b, a = rig.channels(before_vals), rig.channels(after_vals)
    any_change, bad = False, False
    for c in b:
        x, y = np.asarray(b[c], float), np.asarray(a[c], float)
        nx, ny = np.isnan(x).any(-1), np.isnan(y).any(-1)
        d = np.minimum(np.abs(x - y).max(-1), np.abs(x + y).max(-1))
        d = np.where(nx & ny, 0.0, np.where(nx ^ ny, np.inf, d))
        rows = np.flatnonzero(d > rig.tol)
        if not len(rows):
            continue
        any_change = True
        dep = rig.depicts.get(c)
        if not dep:
            bad = True
            continue
        cols = [B.JI[n] for n in dep]
        bad |= bool(m["preserve"][np.ix_(rows, cols)].any() or not m["permit"][np.ix_(rows, cols)].all())
    if rig_claim and not any_change:
        return "NO_RIG_INFLUENCE"
    return "OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN" if bad else "ACCEPT"


def run_controls(pop, ev, ref, cs, ok_cells, preserved_cell, prohibited_cell, rig=None, cs_unconsumed=None,
                 unconsumed_cells=None, out_of_domain=None, cross=None):
    """Every declared control on one population. NOT APPLICABLE is recorded with its reason, never as PASS."""
    snap = pristine(ev)
    real_population = pop != "synthetic"
    res = {}
    fresh = lambda: B.Lineage(ev)

    def record(name, code, extra=None, na=None):
        exp = CONTROLS[name][0]
        if na:
            res[name] = {"expected": exp, "observed": "NOT_APPLICABLE", "reason": na, "ok": None}
            return
        r = {"expected": exp, "observed": code}
        r.update(extra or {})
        r["ok"] = bool(code == exp and r.get("atomic_head_unchanged", True) and r.get("evidence_unchanged", True))
        res[name] = r

    def edit_case(name, cells, cs_=None, lin=None, **kw):
        lin = lin or fresh()
        mod = kw.pop("mod", None)
        evx = kw.pop("ev", ev)
        rig_obj = kw.pop("rig_obj", None)
        rec = B.make_mock_edit(lin, ev, cs_ or cs, name, cells, DELTA, **kw)
        if mod:
            mod(rec)
        situation = None
        if rig_obj is not None and real_population:
            # the expected code of a rig control is only exercisable if the frozen rig actually produces the situation
            situation = independent_authority(rig_obj, ev, cs_ or cs, lin.state()["values"],
                                              source_level_after(ev, cs_ or cs, rec), rec.get("rig_claim"))
            if situation != CONTROLS[name][0]:
                code, _, r = attempt(lambda: B.apply_edit(lin, evx, cs_ or cs, rec, rig=rig_obj), lin, ev, snap)
                record(name, None, na=f"real population did not produce the situation (independent rule -> {situation}); "
                                      f"candidate returned {code}, consistent={code == situation}")
                return
        code, _, r = attempt(lambda: B.apply_edit(lin, evx, cs_ or cs, rec, rig=rig_obj), lin, ev, snap)
        if situation is not None:
            r["independent_rule"] = situation
        record(name, code, r)

    code, _, r = attempt(lambda: B.verify_evidence(ev), None, ev, snap)
    record("positive_ingest_unchanged", code, r)
    edit_case("positive_valid_edit", ok_cells[:1])
    edit_case("unauthorized_edit", [prohibited_cell])
    extra = copy.deepcopy(cs)
    extra["directives"].append({"id": "E_conflict", "kind": "permit", "joints": [preserved_cell[1]],
                                "spans": [[preserved_cell[0], preserved_cell[0]]]})
    code, _, r = attempt(lambda: B.resolve_constraints(ev, extra), None, ev, snap)
    record("preserve_permit_conflict", code, r)
    edit_case("edit_on_preserved_cell", [preserved_cell])
    edit_case("stale_source_digest", ok_cells[:1], mod=lambda rec: rec.update(evidence_digest="f" * 64))
    lin = fresh()
    B.apply_edit(lin, ev, cs, B.make_mock_edit(lin, ev, cs, "setup", ok_cells[:1], DELTA))
    root = lin.order[0]
    edit_case("stale_edit_parent", ok_cells[:1], lin=lin, base=root)
    for name, fn in (("missing_source_evidence", lambda v: v["arrays"].pop("obs2d_score_raw")),
                     ("missing_source_provenance", lambda v: v["meta"]["sources"][sorted(v["meta"]["sources"])[0]].pop("sha256")),
                     ("duplicate_joint_identity", lambda v: v["meta"]["joint_names"].__setitem__(1, v["meta"]["joint_names"][0])),
                     ("manufactured_3d_confidence", lambda v: v["arrays"].__setitem__("est3d_reliability", 1.0 - np.nan_to_num(v["arrays"]["obs2d_score_raw"]))),
                     ("nonfinite_score_on_present", lambda v: v["arrays"]["obs2d_score_raw"].__setitem__(
                         tuple(np.argwhere(v["arrays"]["obs2d_present"])[0]), np.nan))):
        edit_case(name, ok_cells[:1], ev=mutated(ev, fn))
    edit_case("modified_immutable_observation", ok_cells[:1], ev=mutated(
        ev, lambda v: v["arrays"]["obs2d_score_raw"].__setitem__(tuple(np.argwhere(v["arrays"]["obs2d_present"])[0]),
                                                                 v["arrays"]["obs2d_score_raw"][tuple(np.argwhere(v["arrays"]["obs2d_present"])[0])] + 0.01),
        redigest=False))
    absent = np.argwhere(~ev["arrays"]["obs2d_present"])
    if len(absent):
        def coerce(v, c=tuple(absent[0])):
            v["arrays"]["obs2d_xy_px"][c] = 0.0
            v["arrays"]["obs2d_score_raw"][c] = 0.0
        edit_case("unknown_to_zero_coercion", ok_cells[:1], ev=mutated(ev, coerce))
    else:
        record("unknown_to_zero_coercion", None, na="population has no absent (UNKNOWN) observation cell")
    if (np.nan_to_num(ev["arrays"]["obs2d_score_raw"]) > 1).any():
        clipped = mutated(ev, lambda v: v["arrays"]["obs2d_score_raw"].__setitem__(
            slice(None), np.minimum(v["arrays"]["obs2d_score_raw"], 1.0)))
        code, _, r = attempt(lambda: (B.verify_evidence(clipped), B.verify_against_source(clipped, ref)), None, ev, snap)
        record("score_clipped_to_unit_interval", code, r)
    else:
        record("score_clipped_to_unit_interval", None, na="population has no score above 1; clipping is a no-op")
    fi, ts = ev["arrays"]["frame_index"], ev["arrays"]["timestamp"]
    edit_case("mismatched_frame_index", [(int(fi[-1]) + 1, ok_cells[0][1])])
    edit_case("mismatched_timestamp", ok_cells[:1], mod=lambda rec: rec.update(
        target_timestamps=[t + 1.0 / ev["meta"]["fps"] for t in rec["target_timestamps"]]))
    if cross:
        edit_case("subject_identity_ambiguity", [cross[1]], op="copy_from_frame", source_frame=cross[0])
    else:
        record("subject_identity_ambiguity", None, na="no permitted cross-frame target")
    edit_case("coordinate_unit_mismatch", ok_cells[:1], unit="pixels")
    edit_case("edit_on_immutable_observation", ok_cells[:1], signal="obs2d_score_raw")
    unknown_permitted = [c for c in np.argwhere(~np.isfinite(ev["arrays"]["est3d_xyz_hypothesis"]).all(-1))
                         if B.resolve_constraints(ev, cs)["permit"][tuple(c)]]
    if unknown_permitted:
        r0, c0 = unknown_permitted[0]
        edit_case("edit_on_unknown_target", [(int(fi[r0]), B.JOINTS[c0])])
    else:
        record("edit_on_unknown_target", None, na="population has no permitted UNKNOWN 3D cell")
    lin = fresh()
    B.apply_edit(lin, ev, cs, B.make_mock_edit(lin, ev, cs, "to_revert", ok_cells[:2], DELTA))
    rv = B.make_revert(lin, "to_revert")
    bad = bytearray(bytes.fromhex(rv["restore_values_hex"]))
    bad[0] ^= 1
    rv["restore_values_hex"] = bytes(bad).hex()
    code, _, r = attempt(lambda: B.apply_revert(lin, ev, rv), lin, ev, snap)
    record("reversal_not_restoring_parent", code, r)
    if rig is None:
        for name in RIG_CONTROLS:
            record(name, None, na=f"no frozen rig configuration consumes the {pop} population")
    else:
        lin = fresh()
        assoc = B.make_association(rig, ev, lin, action_fps=ACTION_FPS)
        B.validate_association(assoc, rig, ev, lin)
        bad_assoc = copy.deepcopy(assoc)
        bad_assoc["channel_owner"][sorted(rig.depicts)[-1]] = "j1x_left_arm (not consumed by this configuration)"
        code, _, r = attempt(lambda: B.validate_association(bad_assoc, rig, ev, lin), None, ev, snap)
        record("invalid_source_to_rig_association", code, r)
        edit_case("source_change_no_rig_influence", unconsumed_cells, cs_=cs_unconsumed, rig_claim=True, rig_obj=rig)
        edit_case("output_change_outside_authorized_domain", out_of_domain, rig_obj=rig)
    return res


def control_summary(res):
    vals = list(res.values())
    return {"applicable": sum(v["ok"] is not None for v in vals), "passed": sum(v["ok"] is True for v in vals),
            "failed": sorted(k for k, v in res.items() if v["ok"] is False),
            "not_applicable": sorted(k for k, v in res.items() if v["ok"] is None)}


def programme_step(log, sid, desc, expected, fn, lin, ev, snap):
    code, res, r = attempt(fn, lin, ev, snap)
    r.update({"step": sid, "what": desc, "expected": expected})
    r["ok"] = bool(code == expected and r.get("atomic_head_unchanged", True) and r.get("evidence_unchanged", True))
    log.append(r)
    return code, res


def locality(before_state, after_state, cells, ev):
    row = {int(f): i for i, f in enumerate(ev["arrays"]["frame_index"])}
    req = np.zeros(before_state["values"].shape[:2], bool)
    for f, n in cells:
        req[row[int(f)], B.JI[n]] = True
    a, b = before_state["values"], after_state["values"]
    changed = ~((a == b) | (np.isnan(a) & np.isnan(b))).all(-1)
    return {"requested": int(req.sum()), "changed": int(changed.sum()), "changed_equals_requested": bool(np.array_equal(changed, req)),
            "unintended": int((changed & ~req).sum()), "unedited_bytes_identical": bool(a[~req].tobytes() == b[~req].tobytes()),
            "origin_changed_equals_requested": bool(np.array_equal(after_state["origin"] != before_state["origin"], req))}


def edit_programme(pop, ev, cs_main, ok_cells, cs_arm=None, rig=None, cs_head=None, head_cells=None, cross=None, E2_filter=None,
                   cross_expected="IDENTITY_UNVERIFIED"):
    snap = pristine(ev)
    lin = B.Lineage(ev)
    log, extra = [], {}
    root = lin.head
    e1 = B.make_mock_edit(lin, ev, cs_main, "E1", ok_cells, DELTA)
    code, s1 = programme_step(log, "E1", "offset DELTA on all permitted cells", "ACCEPT",
                              lambda: B.apply_edit(lin, ev, cs_main, e1), lin, ev, snap)
    if code != "ACCEPT":
        return {"log": log, "aborted": "E1 not accepted"}
    extra["E1_locality"] = locality(lin.states[root], s1, ok_cells, ev)
    extra["E1_originals_equal_parent"] = lin.records["E1"]["record"]["original_values_hex"] == \
        lin.states[root]["values"][tuple(np.array([[{int(f): i for i, f in enumerate(ev["arrays"]["frame_index"])}[f], B.JI[n]]
                                                    for f, n in ok_cells]).T)].tobytes().hex()
    s1_id = lin.head
    e2_cells = [c for c in ok_cells if E2_filter(c)] if E2_filter else ok_cells[:max(1, len(ok_cells) // 4)]
    e2 = B.make_mock_edit(lin, ev, cs_main, "E2", e2_cells, (0.005, 0.0, 0.0))
    code, s2 = programme_step(log, "E2", "second offset on a subset", "ACCEPT", lambda: B.apply_edit(lin, ev, cs_main, e2), lin, ev, snap)
    extra["E2_locality"] = locality(lin.states[s1_id], s2, e2_cells, ev)
    e3 = B.make_mock_edit(lin, ev, cs_main, "E3", e2_cells, (0.005, 0.0, 0.0), base=s1_id)
    programme_step(log, "E3", "stale parent", "STALE_PARENT", lambda: B.apply_edit(lin, ev, cs_main, e3), lin, ev, snap)
    e4 = B.make_mock_edit(lin, ev, cs_main, "E4", e2_cells, DELTA)
    e4["evidence_digest"] = "f" * 64
    programme_step(log, "E4", "stale evidence", "STALE_EVIDENCE", lambda: B.apply_edit(lin, ev, cs_main, e4), lin, ev, snap)
    if cross:
        e5 = B.make_mock_edit(lin, ev, cs_main, "E5", [cross[1]], op="copy_from_frame", source_frame=cross[0])
        programme_step(log, "E5", "cross-frame copy", cross_expected, lambda: B.apply_edit(lin, ev, cs_main, e5), lin, ev, snap)
    code, s3 = programme_step(log, "RV2", "revert E2", "ACCEPT", lambda: B.apply_revert(lin, ev, B.make_revert(lin, "E2")), lin, ev, snap)
    extra["RV2_equals_post_E1"] = bool(s3["values"].tobytes() == lin.states[s1_id]["values"].tobytes()
                                       and s3["origin"].tobytes() == lin.states[s1_id]["origin"].tobytes())
    bad = B.make_revert(lin, "E1", "RV1_tampered")
    h = bytearray(bytes.fromhex(bad["restore_values_hex"]))
    h[-1] ^= 1
    bad["restore_values_hex"] = bytes(h).hex()
    programme_step(log, "RV1_tampered", "tampered revert of E1", "REVERT_MISMATCH", lambda: B.apply_revert(lin, ev, bad), lin, ev, snap)
    code, s4 = programme_step(log, "RV1", "revert E1", "ACCEPT", lambda: B.apply_revert(lin, ev, B.make_revert(lin, "E1")), lin, ev, snap)
    extra["RV1_equals_hypothesis"] = bool(s4["values"].tobytes() == lin.states[root]["values"].tobytes()
                                          and s4["origin"].tobytes() == lin.states[root]["origin"].tobytes()
                                          and s4["values"].tobytes() == ev["arrays"]["est3d_xyz_hypothesis"].tobytes())
    rig_log = []
    if rig is not None:
        base_ch = rig.channels(lin.state(root)["values"])

        def rig_step(sid, desc, predicted, cs_, cells, **kw):
            """Rig-dependent step: the prediction is reported; correctness = consistency with observed influence."""
            rec = B.make_mock_edit(lin, ev, cs_, sid, cells, DELTA, **kw)
            indep = independent_authority(rig, ev, cs_, lin.state()["values"], source_level_after(ev, cs_, rec), kw.get("rig_claim"))
            code, res, r = attempt(lambda: B.apply_edit(lin, ev, cs_, rec, rig=rig), lin, ev, snap)
            r.update({"step": sid, "what": desc, "predicted": predicted, "prediction_match": code == predicted,
                      "independent_rule": indep, "consistent": code == indep and r.get("atomic_head_unchanged", True)
                      and r.get("evidence_unchanged", True)})
            rig_log.append(r)
            return code, res

        code, msg = rig_step("R9", "rig-enforced E1 under main constraints", "OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN", cs_main, ok_cells)
        extra["R9_message"] = msg if isinstance(msg, str) else None
        code, s10 = rig_step("R10", "rig-enforced E1 under arm-widened constraints", "ACCEPT", cs_arm, ok_cells)
        if code == "ACCEPT":
            extra["R10_association"] = lin.records["R10"]["record"]["output_association"]
            code, s11 = programme_step(log, "R11", "revert R10", "ACCEPT",
                                       lambda: B.apply_revert(lin, ev, B.make_revert(lin, "R10")), lin, ev, snap)
            ch11 = rig.channels(s11["values"])
            extra["R11_channels_bit_equal_hypothesis"] = all(np.array_equal(base_ch[c], ch11[c], equal_nan=True) for c in base_ch)
        else:
            extra["R11_channels_bit_equal_hypothesis"] = "NOT RUN (R10 not accepted)"
        rig_step("R12", "rig-claimed unconsumed-source edit", "NO_RIG_INFLUENCE", cs_head, head_cells, rig_claim=True)
        code, s13 = rig_step("R13", "unclaimed source-only edit under rig check", "ACCEPT", cs_head, head_cells)
        if code == "ACCEPT":
            extra["R13_association_changed"] = lin.records["R13"]["record"]["output_association"]["changed_channel_frames"]
            programme_step(log, "R14", "revert R13", "ACCEPT", lambda: B.apply_revert(lin, ev, B.make_revert(lin, "R13")), lin, ev, snap)
        extra["final_equals_hypothesis"] = bool(lin.state()["values"].tobytes() == ev["arrays"]["est3d_xyz_hypothesis"].tobytes())
    extra["rig_log"] = rig_log
    extra["rig_steps_consistent"] = all(r["consistent"] for r in rig_log)
    extra["rig_steps_prediction_match"] = all(r["prediction_match"] for r in rig_log)
    extra["lineage_audit"] = lin.audit()
    extra["evidence_unchanged_end"] = evidence_unchanged(ev, snap)
    extra["source_files_unchanged"] = all(sha(s["path"]) == s["sha256"] for s in ev["meta"]["sources"].values() if s.get("path"))
    try:
        ev["arrays"]["obs2d_score_raw"][0, 0] = 0.0
        extra["inplace_write_refused"] = False
    except ValueError:
        extra["inplace_write_refused"] = True
    extra["head_state_id"] = lin.head
    extra["state_ids"] = list(lin.order)
    return {"log": log, "all_ok": all(s["ok"] for s in log), **extra}


# -------------------------------------------------------------------------------------------- rig influence ---

def real_rig_conditions(ev, rig, rc):
    fi = ev["arrays"]["frame_index"]
    base_vals = np.array(ev["arrays"]["est3d_xyz_hypothesis"])
    base = rig.channels(base_vals)
    pert = (fi % 2 == 0)
    rows_p, rows_u = np.flatnonzero(pert), np.flatnonzero(~pert)
    out = {}
    states = {}
    for name, cond in RIG_CONDITIONS["real"].items():
        cells = [(int(fi[r]), n) for r in rows_p for n in cond["joints"]]
        if cond.get("along_trunk_axis"):
            u = base_vals[:, B.JI["thorax"]].astype(np.float64) - base_vals[:, B.JI["pelvis"]].astype(np.float64)
            u = u / np.linalg.norm(u, axis=-1, keepdims=True)
            delta = (0.01 * u[rows_p]).tolist()
        else:
            delta = DELTA
        permit = B.make_constraints(ev, f"permit_{name}", [{"id": "P", "kind": "permit", "joints": cond["joints"],
                                                             "spans": [[int(fi[r]), int(fi[r])] for r in rows_p]}])
        lin = B.Lineage(ev)
        st = B.apply_edit(lin, ev, permit, B.make_mock_edit(lin, ev, permit, name, cells, delta))
        after = rig.channels(st["values"])
        cp = R.changed_sets(base, after, RIG_TOL, rows_p)
        cu = R.changed_sets(base, after, RIG_TOL, rows_u)
        observed = sorted(c for c in R.CHANNELS if cp[c]["changed_frames"])
        w113_pred = sorted(c for c, deps in R.W113_DEPENDENCIES.items() if set(deps) & set(cond["joints"]))
        lin2 = B.Lineage(ev)
        code, _, r = attempt(lambda: B.apply_edit(lin2, ev, permit, B.make_mock_edit(lin2, ev, permit, name, cells, delta,
                                                                                     rig_claim=True), rig=rig), lin2)
        indep = independent_authority(rig, ev, permit, base_vals, st["values"], True)
        out[name] = {"kind": cond["kind"], "joints": cond["joints"], "perturbed_rows": int(len(rows_p)),
                     "delta": "0.01 m along per-frame unit(thorax - pelvis)" if cond.get("along_trunk_axis") else DELTA,
                     "expected_channels": sorted(cond["expect"]), "observed_channels": observed,
                     "match": observed == sorted(cond["expect"]), "per_channel_perturbed": cp,
                     "unperturbed_rows_bit_changes": int(sum(v["bit_changed_frames"] for v in cu.values())),
                     "w113_conservative_prediction": w113_pred,
                     "w113_predicted_but_unchanged": sorted(set(w113_pred) - set(observed)),
                     "changed_but_not_w113_predicted": sorted(set(observed) - set(w113_pred)),
                     "authority": {"predicted": cond["authority"], "observed": code, "prediction_match": code == cond["authority"],
                                   "independent_rule": indep, "consistent": code == indep,
                                   "atomic": r.get("atomic_head_unchanged", True)}}
        states[name] = st["values"]
    return out, states, base_vals


def mixed_rig_conditions(decl, rc):
    cases = json.loads(Path(decl["inputs"]["w105_cases"]["path"]).read_text())["cases"]
    pm = np.load(decl["inputs"]["w105_sources"]["path"])["pred_M"]
    res = {}
    for wname in MIXED_WINDOWS:
        case = cases[wname]
        src = R.MixedSources(case, pm, decl["inputs"]["w102_hmr_pred_full"]["path"], decl["inputs"]["w102_j1x_predictions_3dpw"]["path"])
        slots = R.mixed_slots(src, rc)
        base = R.mixed_channels(slots)
        n = len(src.rows)
        rows_p, rows_u = np.arange(0, n, 2), np.arange(1, n, 2)
        w = {"sequence_id": case["sequence_id"], "frames": n, "split": sorted({f["split"] for f in case["frames"]}),
             "reproduction": R.mixed_reproduction(src, slots), "conditions": {}}
        for name, cond in RIG_CONDITIONS["mixed"].items():
            kw = {}
            if cond["source"] == "s0":
                s0 = src.s0.copy()
                s0[rows_p, B.JI[cond["joint"]]] += np.asarray(DELTA, np.float32)
                kw["s0"] = s0
            elif cond["source"] == "j1x_wrist":
                jw = src.j1x_wrist.copy()
                jw[rows_p] += np.asarray(DELTA)
                kw["j1x_wrist"] = jw
            elif cond["source"] == "hmr_j24":
                j24 = src.hmr_j24.copy()
                j24[rows_p, cond["smpl_index"]] += np.asarray(DELTA, np.float32)
                kw["hmr_j24"] = j24
            after = R.mixed_channels(R.mixed_slots(src, rc, **kw))
            cp = R.changed_sets(base, after, RIG_TOL, rows_p)
            cu = R.changed_sets(base, after, RIG_TOL, rows_u)
            observed = sorted(c for c in R.CHANNELS if cp[c]["changed_frames"])
            w["conditions"][name] = {"kind": cond["kind"], "source": cond["source"], "expected_channels": sorted(cond["expect"]),
                                     "observed_channels": observed, "match": observed == sorted(cond["expect"]),
                                     "per_channel_perturbed": cp,
                                     "unperturbed_rows_bit_changes": int(sum(v["bit_changed_frames"] for v in cu.values()))}
        res[wname] = w
    return res


def blender_case_file(ev, rc, base_vals, states, extra_states, out):
    ts = ev["arrays"]["timestamp"]
    base_frames = R.real_case_frames(base_vals, ts, rc, "hypothesis")
    cases = {}
    for name, vals in list(states.items()) + list(extra_states.items()):
        fr = R.merge_slots(base_frames, R.real_case_frames(vals, ts, rc, name),
                           {"FIXED": "W114 unperturbed hypothesis", "RUNTIME": f"W114 {name} (MOCK)"})
        cases[name] = {"case": name, "sequence_id": REAL_SEQUENCE_ID, "start_frame": 0, "fps_sampled": 50.0, "frames": fr,
                       "labels": {"FIXED": "hypothesis (W105 M)", "RUNTIME": f"{name} MOCK perturbation"},
                       "short_labels": {"FIXED": "H", "RUNTIME": "P"},
                       "title": f"W114 {name} | frozen W105 S0-only rig | MOCK source perturbation, not animation",
                       "header_note": "diagnostic replay of source-to-rig influence; not quality evidence"}
    dump(out / "w114_rig_cases.json", {"rest_snapshot_digest": rc["rest"].digest(), "canonical_to_armature_xyzw": list(rc["alignment"]),
                                       "target_chain": {"root_head": list(rc["target"].root_head)}, "cases": cases})
    return {"cases": sorted(cases), "sha256": sha(out / "w114_rig_cases.json")}


def time_identity(ev, rig_audit=None):
    fi, ts = ev["arrays"]["frame_index"], ev["arrays"]["timestamp"]
    t_action = fi / ACTION_FPS
    mis = np.abs(t_action - ts) > 1e-9
    return {"source_fps": ev["meta"]["fps"], "action_fps": ACTION_FPS, "keys": int(len(fi)), "mismatched_keys": int(mis.sum()),
            "max_mismatch_s": float(np.abs(t_action - ts).max()), "mismatch_at_last_key_s": float(t_action[-1] - ts[-1]),
            "source_duration_s": float(ts[-1] - ts[0]), "action_duration_s": float(t_action[-1] - t_action[0]),
            "sidecar_recovery": "association maps action_key_frame -> frame_index -> source_time for every key",
            "identity": {
                "source_sequence": {"status": "PRESENT", "value": ev["meta"]["sequence_id"], "binding": "clip sha256 + decoded frame digests"},
                "frame": {"status": "PRESENT", "frames_with_digest": len(ev["meta"]["frame_digests"] or [])},
                "tracked_subject": {"status": "UNKNOWN per frame", "frames_unverified": int((~ev["arrays"]["subject_continuity_verified"]).sum()),
                                    "frames": int(len(fi))},
                "candidate_animation": {"status": "PRESENT (state_id lineage)", "binding": "evidence_digest + parent + record digest + values"},
                "rig_channel": {"status": "PRESENT", "value": "Action name / F-curve data_path[index] (W91 renderer)"}}}


# ---------------------------------------------------------------------------------------------------- build ---

def cmd_build(a):
    decl = load_declaration(a)
    out = new_dir(a.out_dir)
    t0 = time.perf_counter()
    rep = {"schema": "w114_build_v1", "declaration_sha256": a.declaration_sha, "tag": a.tag, "GT_read": False}
    # ---- synthetic
    sev, sref = synthetic_evidence(out)
    scs = synth_directives(sev)
    toy = toy_rig()
    s_ok = [(3, "left_wrist"), (4, "left_wrist"), (5, "left_wrist")]
    rep["synthetic"] = {
        "evidence_digest": sev["meta"]["evidence_digest"], "scores": B.score_accounting(sev, 0.3),
        "source_exact": _try(lambda: B.verify_against_source(sev, sref)),
        "controls": run_controls("synthetic", sev, sref, scs["SS1"], s_ok, (1, "left_wrist"), (4, "left_knee"), rig=toy,
                                 cs_unconsumed=scs["SS1"], unconsumed_cells=[(3, "head")], out_of_domain=s_ok, cross=(5, (4, "left_wrist"))),
        "programme": edit_programme("synthetic", sev, scs["SS1"], s_ok, cs_arm=scs["SS1_arm"], rig=toy, cs_head=scs["SS1"],
                                    head_cells=[(3, "head"), (5, "head")], cross=(5, (4, "left_wrist")),
                                    E2_filter=lambda c: c[0] == 3)}
    lin = B.Lineage(sev)
    pos = B.make_mock_edit(lin, sev, scs["SS1"], "X_verified", [(3, "left_wrist")], op="copy_from_frame", source_frame=2)
    code, _, r = attempt(lambda: B.apply_edit(lin, sev, scs["SS1"], pos), lin)
    rep["synthetic"]["cross_frame_verified_positive"] = dict(r, expected="ACCEPT", ok=code == "ACCEPT")
    rep["synthetic"]["controls_summary"] = control_summary(rep["synthetic"]["controls"])
    # ---- real
    z, wrep, insp = real_inputs(decl)
    ev, ref = real_evidence(decl, z, wrep, insp)
    rc = R.rig_context()
    rig = real_rig(rc)
    sets, ok_cells, spans = real_directives(ev)
    masks = B.resolve_constraints(ev, sets["RS1"])
    rep["real"] = {"evidence_digest": ev["meta"]["evidence_digest"], "verify": _try(lambda: B.verify_evidence(ev)),
                   "source_exact": _try(lambda: B.verify_against_source(ev, ref)), "scores": B.score_accounting(ev, 0.3),
                   "permit_spans": spans, "permitted_cells": len(ok_cells),
                   "RS1_masks": {"preserve": int(masks["preserve"].sum()), "permit": int(masks["permit"].sum()),
                                 "prohibited_default": int((~masks["preserve"] & ~masks["permit"]).sum())},
                   "RS2": _try(lambda: B.resolve_constraints(ev, sets["RS2"])),
                   "support": {B.EST3D_SUPPORT[k]: int((ev["arrays"]["est3d_support"] == k).sum()) for k in B.EST3D_SUPPORT},
                   "derived_2d_cells": int(sum(not k.startswith("observed_2d") for k in ev["meta"]["obs2d_kind"].values()) * len(z["frame_index"])),
                   "input_2d_zero_fill_of_invalid": int((z["input_2d"][~z["input_valid"]] == 0).all(-1).sum()),
                   "invalid_input_cells": int((~z["input_valid"]).sum())}
    head_cells = [(f, "head") for f, _ in ok_cells]
    rep["real"]["controls"] = run_controls("real", ev, ref, sets["RS1"], ok_cells, (50, "left_wrist"), (150, "left_knee"), rig=rig,
                                           cs_unconsumed=sets["RS_head"], unconsumed_cells=head_cells, out_of_domain=ok_cells,
                                           cross=(108, (109, "left_wrist")))
    rep["real"]["controls_summary"] = control_summary(rep["real"]["controls"])
    rep["real"]["programme"] = edit_programme("real", ev, sets["RS1"], ok_cells, cs_arm=sets["RS1_arm"], rig=rig, cs_head=sets["RS_head"],
                                              head_cells=head_cells, cross=(108, (109, "left_wrist")),
                                              E2_filter=lambda c: 109 <= c[0] <= 118)
    # ---- frozen rig influence
    base_frames = R.real_case_frames(np.array(ev["arrays"]["est3d_xyz_hypothesis"]), ev["arrays"]["timestamp"], rc, "h")
    vc = json.loads(Path(decl["inputs"]["w105_video_cases"]["path"]).read_text())["cases"]["real_clip"]["frames"]
    mine = json.loads(json.dumps(clean(R.slot_view(base_frames, "SLOT"))))
    rep["rig_reproduction_real"] = {"frames": len(vc), "byte_equal_frames": sum(x == y for x, y in zip(mine, R.slot_view(vc, "FIXED")))}
    conds, states, base_vals = real_rig_conditions(ev, rig, rc)
    rep["rig_real"] = conds
    rep["rig_mixed"] = mixed_rig_conditions(decl, rc)
    lin = B.Lineage(ev)
    extra_states = {}
    try:
        acc = B.apply_edit(lin, ev, sets["RS1_arm"], B.make_mock_edit(lin, ev, sets["RS1_arm"], "E_RS1_arm", ok_cells, DELTA), rig=rig)
        extra_states["E_RS1_arm_accepted"] = acc["values"]
    except B.ContractError as e:
        rep["blender_E_RS1_arm"] = f"not accepted: {e.code}"
    rep["blender_case_file"] = blender_case_file(ev, rc, base_vals, states, extra_states, out)
    rep["time_identity"] = time_identity(ev)
    rep["real"]["evidence_unchanged_after_rig"] = ev["meta"]["evidence_digest"] == B.evidence_digest(ev)
    # ---- existing-path control (W113 helpers, unchanged code)
    rep["existing_path"] = existing_path(decl, z, wrep, insp, masks["preserve"], out)
    # ---- benchmark (separate)
    bev, bref, binfo = bench_evidence(decl)
    bcs, bcells = bench_directives(bev)
    bm = B.resolve_constraints(bev, bcs)
    pres = np.argwhere(bm["preserve"])
    bfi = bev["arrays"]["frame_index"]
    bcross = None
    if bcells:
        r_ = int(np.flatnonzero(bfi == bcells[0][0])[0])
        bcross = (int(bfi[r_ - 1]), bcells[0]) if r_ > 0 else None
    rep["benchmark"] = {"info": binfo, "evidence_digest": bev["meta"]["evidence_digest"], "verify": _try(lambda: B.verify_evidence(bev)),
                        "source_exact": _try(lambda: B.verify_against_source(bev, bref)), "scores": B.score_accounting(bev),
                        "permitted_cells": len(bcells), "preserved_cells": int(bm["preserve"].sum()),
                        "support": {B.EST3D_SUPPORT[k]: int((bev["arrays"]["est3d_support"] == k).sum()) for k in B.EST3D_SUPPORT}}
    if bcells:
        prohibited = None  # RSB leaves no default-prohibited cell (preserve and permit are complementary on finite cells)
        unsup = ~np.isfinite(bev["arrays"]["est3d_xyz_hypothesis"]).all(-1)
        prohibited = (int(bfi[np.argwhere(unsup)[0][0]]), B.JOINTS[np.argwhere(unsup)[0][1]]) if unsup.any() else None
        rep["benchmark"]["controls"] = run_controls(
            "benchmark", bev, bref, bcs, bcells, (int(bfi[pres[0][0]]), B.JOINTS[pres[0][1]]),
            prohibited or (int(bfi[pres[0][0]]), B.JOINTS[pres[0][1]]), cross=bcross)
        if prohibited is None:
            rep["benchmark"]["controls"]["unauthorized_edit"] = {
                "expected": "EDIT_NOT_AUTHORIZED", "observed": "NOT_APPLICABLE", "ok": None,
                "reason": "RSB covers every finite cell with preserve or permit; no default-prohibited cell exists"}
        rep["benchmark"]["controls_summary"] = control_summary(rep["benchmark"]["controls"])
        rep["benchmark"]["programme"] = bench_programme(bev, bcs, bcells, bcross)
    else:
        rep["benchmark"]["vacuous"] = True
    rep["digests"] = {"synthetic": sev["meta"]["evidence_digest"], "real": ev["meta"]["evidence_digest"],
                      "benchmark": bev["meta"]["evidence_digest"], "real_programme_head": rep["real"]["programme"].get("head_state_id"),
                      "real_programme_states": rep["real"]["programme"].get("state_ids"),
                      "benchmark_programme_states": rep["benchmark"].get("programme", {}).get("state_ids"),
                      "synthetic_programme_states": rep["synthetic"]["programme"].get("state_ids"),
                      "rig_case_file": rep["blender_case_file"]["sha256"]}
    rep["cost"] = {"build_seconds": time.perf_counter() - t0, "real_frames": int(len(z["frame_index"]))}
    dump(out / "build_report.json", rep)
    print(json.dumps({"tag": a.tag, "seconds": round(rep["cost"]["build_seconds"], 2), "real_digest": rep["digests"]["real"]}))
    return 0


def _try(fn):
    try:
        r = fn()
        return {"observed": "PASS", "result": clean(r) if not isinstance(r, tuple) else None}
    except B.ContractError as e:
        return {"observed": e.code, "message": str(e)}


def bench_programme(ev, cs, cells, cross):
    snap = pristine(ev)
    lin = B.Lineage(ev)
    log = []
    root = lin.head
    e1 = B.make_mock_edit(lin, ev, cs, "EB1", cells, DELTA)
    code, s1 = programme_step(log, "B1", "offset DELTA on all RSB-permitted cells", "ACCEPT", lambda: B.apply_edit(lin, ev, cs, e1), lin, ev, snap)
    out = {"log": log}
    if code == "ACCEPT":
        out["EB1_locality"] = locality(lin.states[root], s1, cells, ev)
    sup = ev["arrays"]["est3d_support"]
    r0, c0 = np.argwhere(sup == 1)[0]
    e2 = B.make_mock_edit(lin, ev, cs, "EB2", [(int(ev["arrays"]["frame_index"][r0]), B.JOINTS[c0])], DELTA)
    programme_step(log, "B2", "edit a preserved cell", "LOCK_CONFLICT_EDIT", lambda: B.apply_edit(lin, ev, cs, e2), lin, ev, snap)
    e3 = B.make_mock_edit(lin, ev, cs, "EB3", cells, DELTA, base=root)
    programme_step(log, "B3", "stale parent", "STALE_PARENT", lambda: B.apply_edit(lin, ev, cs, e3), lin, ev, snap)
    if cross:
        e4 = B.make_mock_edit(lin, ev, cs, "EB4", [cross[1]], op="copy_from_frame", source_frame=cross[0])
        programme_step(log, "B4", "cross-frame copy", "IDENTITY_UNVERIFIED", lambda: B.apply_edit(lin, ev, cs, e4), lin, ev, snap)
    code, s5 = programme_step(log, "B5", "revert EB1", "ACCEPT", lambda: B.apply_revert(lin, ev, B.make_revert(lin, "EB1")), lin, ev, snap)
    out["B5_equals_hypothesis"] = bool(s5["values"].tobytes() == ev["arrays"]["est3d_xyz_hypothesis"].tobytes())
    out["lineage_audit"] = lin.audit()
    out["evidence_unchanged_end"] = evidence_unchanged(ev, snap)
    out["all_ok"] = all(s["ok"] for s in log)
    out["state_ids"] = list(lin.order)
    return out


def existing_path(decl, z, wrep, insp, preserve_mask, out):
    """The control: unchanged existing code paths, via the W113 helpers (no W113 artefact is touched)."""
    import w113_animation_base_contract as C
    sub = new_dir(out / "existing_path")
    view, lifted, sbank = C.real_view(decl, z, wrep, insp)
    legacy = C.legacy_accounting(view, *C.legacy_path(z, wrep, decl, sub), preserve_mask)
    sem = C.semantics_accounting(view, lifted, sbank)
    du = np.array([[f.points[n].depth_uncertainty for n in B.JOINTS] for f in lifted.frames])
    mg = json.loads((sub / "legacy_motion.json").read_text())
    above = sum(1 for f in mg["frames"] for p in f["points"].values() if p["confidence"] > 1)
    return {"legacy": legacy, "semantics": sem, "bridge_depth_uncertainty_negative": int((du < 0).sum()),
            "bridge_depth_uncertainty_min": float(du.min()), "motion_graph_scores_above_1": int(above)}


# --------------------------------------------------------------------------------------- blender comparison ---

def cmd_blender_compare(a):
    decl = load_declaration(a)
    out = new_dir(a.out_dir)
    build = json.loads((Path(a.build_dir) / "build_report.json").read_text())
    audit = json.loads(Path(a.audit).read_text())
    w105 = json.loads(Path(decl["inputs"]["w105_render_clip_audit"]["path"]).read_text())
    ref_fixed = w105["files"]["real_clip/real_clip.blend"]["actions"]["W91 real_clip FIXED"]["value_digest"]
    res = {"w105_fixed_value_digest": ref_fixed, "cases": {}}
    for f, v in audit["files"].items():
        case = f.split("/")[0]
        acts = v["actions"]
        fx, rt = acts[f"W91 {case} FIXED"], acts[f"W91 {case} RUNTIME"]
        bones = {}
        chan = {}
        for cf, cr in zip(fx["channels"], rt["channels"]):
            if (cf["data_path"], cf["index"]) != (cr["data_path"], cr["index"]) or cf["frames"] != cr["frames"]:
                raise RuntimeError("STOP: FIXED/RUNTIME channel layout differs")
            b = cf["data_path"].split('"')[1] if '"' in cf["data_path"] else "__object__"
            diff = np.array(cf["values"]) != np.array(cr["values"])
            chan.setdefault(b, np.zeros(len(diff), bool))
            chan[b] |= diff
            bones[b] = cf["frames"]
        cond = build["rig_real"].get(case)
        case_level = None
        if cond is not None:
            case_level = {c: cond["per_channel_perturbed"][c]["changed_frames"] for c in R.CHANNELS}
        res["cases"][case] = {
            "scene_fps": v["scene_fps"], "frame_range": v["frame_range"], "fixed_value_digest": fx["value_digest"],
            "fixed_equals_w105_fixed": fx["value_digest"] == ref_fixed, "fcurves": fx["fcurves"], "interpolation": fx["interpolation"],
            "keyed_frames_equal_frame_index_plus_1": [int(round(k)) for k in fx["keyed_frames"]] == list(range(1, 200)),
            "action_changed_frames": {b: int(m.sum()) for b, m in chan.items()},
            "case_level_changed_frames_perturbed_rows": case_level,
            "action_equals_case_level": None if case_level is None else all(int(chan[c].sum()) == case_level[c] for c in R.CHANNELS)}
    acc = res["cases"].get("E_RS1_arm_accepted")
    if acc:
        assoc = build["real"]["programme"].get("R10_association", {}).get("changed_channel_frames", {})
        acc["association_changed"] = assoc
        acc["action_equals_association"] = all(acc["action_changed_frames"].get(c, 0) == assoc.get(c, 0) for c in R.CHANNELS)
    res["all_fixed_reproduce_w105"] = all(c["fixed_equals_w105_fixed"] for c in res["cases"].values())
    res["all_action_equal_case_level"] = all(c.get("action_equals_case_level") is not False and c.get("action_equals_association", True)
                                             for c in res["cases"].values())
    dump(out / "blender_compare.json", res)
    print(json.dumps({"fixed_reproduce": res["all_fixed_reproduce_w105"], "action_equal": res["all_action_equal_case_level"]}))
    return 0


# ----------------------------------------------------------------------------------------------- repeat ---

def cmd_compare(a):
    out = new_dir(a.out_dir)
    x = json.loads((Path(a.first) / "build_report.json").read_text())
    y = json.loads((Path(a.second) / "build_report.json").read_text())
    diffs = []

    def walk(p, q, path=""):
        if isinstance(p, dict) and isinstance(q, dict):
            for k in sorted(set(p) | set(q)):
                walk(p.get(k), q.get(k), f"{path}.{k}")
        elif isinstance(p, list) and isinstance(q, list) and len(p) == len(q):
            for i, (u, v) in enumerate(zip(p, q)):
                walk(u, v, f"{path}[{i}]")
        elif p != q:
            diffs.append(path)
    walk(x, y)
    # declared path-induced / timing differences: the build tag, wall time, and the synthetic source path digest chain
    allowed = (".tag", ".cost.build_seconds")
    synthetic_path = [d for d in diffs if d.startswith(".synthetic") or d.startswith(".digests.synthetic")]
    other = [d for d in diffs if d not in allowed and d not in synthetic_path]
    rep = {"differences": diffs, "allowed_timing_or_tag": [d for d in diffs if d in allowed],
           "synthetic_source_path_induced": synthetic_path, "unexplained": other,
           "real_digest_equal": x["digests"]["real"] == y["digests"]["real"],
           "benchmark_digest_equal": x["digests"]["benchmark"] == y["digests"]["benchmark"],
           "real_programme_states_equal": x["digests"]["real_programme_states"] == y["digests"]["real_programme_states"],
           "rig_case_file_equal": x["digests"]["rig_case_file"] == y["digests"]["rig_case_file"],
           "rig_outcomes_equal": x["rig_real"] == y["rig_real"] and x["rig_mixed"] == y["rig_mixed"]}
    rep["repeatable"] = not other and all(rep[k] for k in ("real_digest_equal", "benchmark_digest_equal", "real_programme_states_equal",
                                                            "rig_case_file_equal", "rig_outcomes_equal"))
    dump(out / "repeat_report.json", rep)
    print(json.dumps({"repeatable": rep["repeatable"], "unexplained": other[:10], "synthetic_path": len(synthetic_path)}))
    return 0


# ----------------------------------------------------------------------------------------------- decide ---

def cmd_decide(a):
    decl = load_declaration(a)
    out = new_dir(a.out_dir)
    pre = json.loads(Path(a.preflight).read_text())
    b = json.loads((Path(a.build_dir) / "build_report.json").read_text())
    rp = json.loads(Path(a.repeat).read_text())
    bl = json.loads(Path(a.blender).read_text())
    r, bm, s = b["real"], b["benchmark"], b["synthetic"]
    rprog, bprog, sprog = r["programme"], bm.get("programme", {}), s["programme"]
    ok_ctl = lambda summ: not summ["failed"]
    V = {
        "V1_inputs_authenticated": bool(pre["pass"]),
        "V2_w113_intact": bool(pre["w113_marker"] and pre["w113_worklog_blob_at_start"] == pre["w113_worklog_blob_now"]
                               and pre["w113_frozen_case"] == "D" and pre["inputs"]["w113_declaration"]["ok"]
                               and pre["inputs"]["w113_decision_frozen"]["ok"]),
        "V3_evidence_source_exact": r["source_exact"]["observed"] == "PASS" and bm["source_exact"]["observed"] == "PASS",
        "V4_positive_ingest": r["verify"]["observed"] == "PASS" and bm["verify"]["observed"] == "PASS"
                              and s["controls"]["positive_ingest_unchanged"]["ok"] is True and r["scores"]["above_1"] > 0,
        "V5_rig_reproduction": b["rig_reproduction_real"]["byte_equal_frames"] == b["rig_reproduction_real"]["frames"] == 199
                               and all(w["reproduction"]["byte_equal_frames"] == w["reproduction"]["frames_with_fixed_slot"] > 0
                                       for w in b["rig_mixed"].values()) and bl["all_fixed_reproduce_w105"],
        "V6_repeatability": bool(rp["repeatable"]),
        "V7_synthetic_controls": ok_ctl(s["controls_summary"]) and sprog.get("all_ok", False) and sprog.get("rig_steps_consistent", False)
                                 and sprog.get("rig_steps_prediction_match", False) and s["cross_frame_verified_positive"]["ok"],
        "V8_rig_positive_control": any(c["observed_channels"] for c in b["rig_real"].values() if c["kind"] in "BD")
                                   and all(any(c["observed_channels"] for c in w["conditions"].values() if c["kind"] in "BD")
                                           for w in b["rig_mixed"].values()),
    }
    G = {
        "G1_evidence_immutable": bool(rprog.get("evidence_unchanged_end") and rprog.get("source_files_unchanged") and rprog.get("inplace_write_refused")
                                      and bprog.get("evidence_unchanged_end", False) and r.get("evidence_unchanged_after_rig")
                                      and all(st.get("evidence_unchanged", True) for st in rprog["log"] + bprog.get("log", []))),
        "G2_edit_locality": bool(rprog["E1_locality"]["changed_equals_requested"] and rprog["E1_locality"]["unintended"] == 0
                                 and rprog["E2_locality"]["changed_equals_requested"] and rprog["E1_originals_equal_parent"]
                                 and bprog.get("EB1_locality", {}).get("changed_equals_requested", False)),
        "G3_reversal": bool(rprog["RV2_equals_post_E1"] and rprog["RV1_equals_hypothesis"] and rprog.get("R11_channels_bit_equal_hypothesis")
                            and rprog.get("final_equals_hypothesis") and rprog["lineage_audit"]["ids_recomputed_equal"]
                            and bprog.get("B5_equals_hypothesis", False) and bprog.get("lineage_audit", {}).get("ids_recomputed_equal", False)),
        "G4_stale_rejection": all(st["ok"] for st in rprog["log"] + bprog.get("log", []) if st["step"] in ("E3", "E4", "B3")),
        "G5_permission": bool(r["permitted_cells"] > 0 and r["RS2"]["observed"] == "LOCK_CONFLICT"
                              and all(r["controls"][k]["ok"] for k in ("unauthorized_edit", "edit_on_preserved_cell", "preserve_permit_conflict"))
                              and rprog.get("all_ok", False) and bprog.get("all_ok", False)),
        "G6_score_semantics": bool(r["source_exact"]["observed"] == "PASS" and r["scores"]["above_1"] > 0
                                   and r["controls"]["manufactured_3d_confidence"]["ok"] and r["controls"]["score_clipped_to_unit_interval"]["ok"]
                                   and bm["controls"]["unknown_to_zero_coercion"]["ok"] is not False),
        "G7_controls_real_bench": ok_ctl(r["controls_summary"]) and ok_ctl(bm.get("controls_summary", {"failed": ["missing"]})),
        "G8_rig_influence_distinguished": bool(all(c["unperturbed_rows_bit_changes"] == 0 for c in b["rig_real"].values())
                                               and all(c["unperturbed_rows_bit_changes"] == 0 for w in b["rig_mixed"].values() for c in w["conditions"].values())
                                               and all(c["authority"]["consistent"] and c["authority"]["atomic"] for c in b["rig_real"].values())
                                               and rprog.get("rig_steps_consistent", False)
                                               and all(r["controls"][k].get("ok") is not False for k in RIG_CONTROLS)
                                               and bl["all_action_equal_case_level"]),
    }
    b1 = b["rig_real"]["B1_left_wrist"]["observed_channels"]
    T = {
        "T1_identity": b["time_identity"]["identity"]["tracked_subject"]["frames_unverified"] > 0,
        "T2_time": b["time_identity"]["mismatched_keys"] > 0,
        "T3_authority": any(set(R.DEPICTS[c]) != {"left_wrist"} for c in b1),
    }
    case = "D" if not all(V.values()) else "C" if not all(G.values()) else "B" if any(T.values()) else "A"
    rep = {"schema": "w114_decision_v1", "declaration_sha256": a.declaration_sha, "validity": V, "core": G, "boundary_triggers": T,
           "case": case, "rule": DECISION["case_rule"],
           "expectation_mismatches": {"real": sorted(k for k, c in b["rig_real"].items() if not c["match"]),
                                      "mixed": sorted(f"{w}:{k}" for w, x in b["rig_mixed"].items() for k, c in x["conditions"].items() if not c["match"])}}
    dump(out / "decision.json", rep)
    print(json.dumps({"case": case, "V": V, "G": G, "T": T}))
    return 0


def cmd_manifest(a):
    base = Path(a.base)
    every = [p for p in sorted(base.rglob("*")) if p.is_file() and p.name != "evidence_manifest.json"]
    pngs = [p for p in every if p.suffix == ".png"]
    files = {p.relative_to(base).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in every if p not in pngs}
    h = hashlib.sha256()
    for p in pngs:
        h.update(p.relative_to(base).as_posix().encode())
        h.update(sha(p).encode())
    rep = {"schema": "w114_evidence_manifest_v1", "starting_head": STARTING_HEAD, "declaration_sha256": a.declaration_sha,
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
    for name in ("build", "decide", "blender-compare"):
        p = sub.add_parser(name)
        p.add_argument("--declaration", required=True); p.add_argument("--declaration-sha", required=True)
        p.add_argument("--out-dir", required=True)
    sub.choices["build"].add_argument("--tag", required=True)
    sub.choices["blender-compare"].add_argument("--build-dir", required=True)
    sub.choices["blender-compare"].add_argument("--audit", required=True)
    for k in ("preflight", "build-dir", "repeat", "blender"):
        sub.choices["decide"].add_argument(f"--{k}", required=True)
    p = sub.add_parser("compare"); p.add_argument("--first", required=True); p.add_argument("--second", required=True)
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("manifest"); p.add_argument("--base", required=True); p.add_argument("--declaration-sha", required=True)
    a = ap.parse_args(argv)
    return {"preflight": cmd_preflight, "declare": cmd_declare, "build": cmd_build, "blender-compare": cmd_blender_compare,
            "compare": cmd_compare, "decide": cmd_decide, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
