#!/usr/bin/env python3
"""W115: Animation Base validity — repeatability, numerical resolution and rig authority (evidence only).

Re-tests the UNCHANGED W114 scratch candidate (`w114_source_preserving_base`), its controls, programmes, frozen-rig
harness and perturbations, with two prospectively corrected measurement contracts (`w115_validity_base`):
  - field-scoped semantic repeat comparison with a declared path allowlist and adversarial source mutations
  - storage-aware rig influence: case-level float64 -> predicted stored float32 F-curve values -> audited Action values,
    compared exactly at every layer (no tolerance), plus local vs world-space lock semantics evaluated separately

Stages (each refuses an existing output directory):
  preflight       instruction / HEAD / lock / W113+W114 historical integrity / input hashes / Blender rest rotation /
                  storage-model calibration on the UNPERTURBED historical FIXED Action only
  declare         frozen declaration (inputs, allowlist, adversarial controls, numeric contract, lock semantics,
                  representative edits, expected outcomes, gates, CASE rule); no outcome read
  build           one independent scratch build (W114 programmes/controls replayed, storage-aware rig, existing path,
                  bundles for semantic comparison, Blender case file incl. mixed windows and the C1 float64 diagnostic)
  repeat          semantic comparison of two builds + positive and adversarial controls + W114 V6 instrument diagnosis
  blender-compare storage exactness, Action change sets, posed world positions, local/world lock evaluation
  decide          frozen gates -> CASE
  review          reporting only: concrete before/after frames
  manifest        evidence manifest

NON-PRODUCTION. No src/ module, schema, model, threshold, rig, solver, FBX/Blender exporter, renderer, default, W113/W114
script, test or artefact is changed. No inference. No evaluation GT is read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w114_rig_influence as R  # noqa: E402
import w114_source_preserving_base as B  # noqa: E402
import w114_source_preserving_contract as C  # noqa: E402
import w115_validity_base as W  # noqa: E402

BATCH = 115
INSTRUCTION_BLOB = "54698373c7b91d4d367c9e091a394f83e80cc889"
STARTING_HEAD = "d017d1b425fa3f5c1c3423e48b1401ab68380ba0"
BRANCH = "arch/single_frame_first"
W113_WORKLOG = "docs/113_WORKLOG_ANIMATION_BASE_CONTRACT_FEASIBILITY.md"
W114_WORKLOG = "docs/114_WORKLOG_SOURCE_PRESERVING_ANIMATION_BASE_EDIT_AUTHORITY.md"
W114_FINAL_COMMIT = "f7239ca4b4934318b44719dfc318216661cc723a"
W114_FROZEN_FILES = ("scripts/w114_source_preserving_base.py", "scripts/w114_rig_influence.py",
                     "scripts/w114_source_preserving_contract.py", "scripts/w114_source_preserving_posthoc.py",
                     "tests/test_w114_source_preserving_base.py")
W114 = Path("C:/Users/dna10/animcv-w114")
LOCK = Path("C:/Users/dna10/animcv-automation/state/active_batch.json")
BLENDER = Path("C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
W114_MANIFEST_SHA = "9f2af3d64aaae2f021d66f5b94beae1982e74097ca49ab90e20a69fb3208a2a0"
W114_ARTEFACTS = {  # relative to W114, expected sha = W114 evidence manifest entry
    "w114_declaration": "results/declare/declaration.json", "w114_decision": "results/decision/decision.json",
    "w114_repeat_report": "results/repeat/repeat_report.json", "w114_blender_compare": "results/blender_compare/blender_compare.json",
    "w114_render_audit": "results/rig_replay/render_audit.json", "w114_rig_cases": "results/build_r1/w114_rig_cases.json",
    "w114_build_r1_report": "results/build_r1/build_report.json", "w114_build_r2_report": "results/build_r2/build_report.json",
    "w114_posthoc_c1": "results/posthoc_c1_float32/c1_float32_diagnostic.json"}
W114_DECLARATION_SHA = "f5fb2451b06126f9133aeccaabdd68f0eeb3fca1243a6e45fda36ab6851f1aa1"
W114_DECISION_SHA = "57ea33fc63ad876e03660040d513851547884fc2ca6714c89d9ed2088f837e05"
DELTA = C.DELTA
RIGHT_ARM = C.RIGHT_ARM

sha = B.sha_file
dump = C.dump
new_dir = C.new_dir
clean = C.clean


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def blender(*args, log=None):
    r = subprocess.run([str(BLENDER), "-b", "-t", "2", "--python-exit-code", "1", "-P", *args], cwd=ROOT,
                       capture_output=True, text=True)
    if log:
        Path(log).write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"STOP: Blender stage failed ({r.returncode})")
    return r


# --------------------------------------------------------------------------------------------- preflight ---

def cmd_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"schema": "w115_preflight_v1", "stop": []}
    rep["head"] = git("rev-parse", "HEAD")
    rep["instruction_blob"] = git("rev-parse", "HEAD:docs/next_batch.md")
    rep["branch"] = git("rev-parse", "--abbrev-ref", "HEAD")
    rep["status_porcelain"] = git("status", "--porcelain")
    title = (ROOT / "docs/next_batch.md").read_text(encoding="utf-8").splitlines()[0]
    rep["instruction_title"] = title
    if rep["instruction_blob"] != INSTRUCTION_BLOB or not title.startswith("# BATCH 115"):
        rep["stop"].append("instruction is not the bound Batch 115 blob")
    if rep["head"] != STARTING_HEAD:
        rep["stop"].append("HEAD is not the bound starting commit")
    if rep["branch"] != BRANCH:
        rep["stop"].append("wrong branch")
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else None
    rep["main_local_lock"] = lock
    if not lock or lock.get("batch") != BATCH or lock.get("instruction_blob") != INSTRUCTION_BLOB or lock.get("start_commit") != STARTING_HEAD:
        rep["stop"].append("Main-local lock is not this Batch 115 claim")
    hist = {}
    for wl, n in ((W113_WORKLOG, 113), (W114_WORKLOG, 114)):
        lines = (ROOT / wl).read_text(encoding="utf-8").rstrip().splitlines()
        hist[n] = {"marker": lines[-2:] == ["AUTOMATION_BATCH_STATUS: COMPLETE", f"BATCH_NUMBER: {n}"],
                   "blob_at_start": git("rev-parse", f"{STARTING_HEAD}:{wl}"), "blob_now": git("hash-object", str(ROOT / wl))}
        if not hist[n]["marker"] or hist[n]["blob_at_start"] != hist[n]["blob_now"]:
            rep["stop"].append(f"W{n} Worklog record not intact")
    rep["historical_worklogs"] = hist
    frozen = {f: {"w114_final": git("rev-parse", f"{W114_FINAL_COMMIT}:{f}"), "head": git("rev-parse", f"HEAD:{f}"),
                  "worktree": git("hash-object", str(ROOT / f))} for f in W114_FROZEN_FILES}
    rep["w114_frozen_scripts"] = frozen
    if any(len({v["w114_final"], v["head"], v["worktree"]}) != 1 for v in frozen.values()):
        rep["stop"].append("a frozen W114 script/test differs from its W114 version")
    man_path = W114 / "evidence_manifest.json"
    rep["w114_manifest_sha256"] = sha(man_path) if man_path.exists() else None
    if rep["w114_manifest_sha256"] != W114_MANIFEST_SHA:
        rep["stop"].append("W114 evidence manifest changed or missing")
    man = json.loads(man_path.read_text())["files"]
    files = {}
    for name, rel in W114_ARTEFACTS.items():
        p = W114 / rel
        got = sha(p) if p.exists() else None
        files[name] = {"path": str(p), "sha256": got, "expected": man[rel]["sha256"], "ok": got == man[rel]["sha256"]}
        if not files[name]["ok"]:
            rep["stop"].append(f"W114 artefact {name} missing or changed")
    if files["w114_declaration"]["sha256"] != W114_DECLARATION_SHA or files["w114_decision"]["sha256"] != W114_DECISION_SHA:
        rep["stop"].append("W114 declaration/decision identity")
    rep["w114_frozen_case"] = json.loads(Path(files["w114_decision"]["path"]).read_text()).get("case")
    if rep["w114_frozen_case"] != "D":
        rep["stop"].append("W114 frozen CASE is not D")
    m105 = json.loads(C.INPUTS["w105_evidence_manifest"][0].read_text())["files"]
    for name, (path, want) in C.INPUTS.items():
        got = sha(path) if Path(path).exists() else None
        if want == "w105":
            want = m105.get(Path(path).relative_to(C.W105).as_posix(), {}).get("sha256")
        files[name] = {"path": str(path), "sha256": got, "expected": want, "ok": got is not None and (want is None or got == want)}
        if not files[name]["ok"]:
            rep["stop"].append(f"input {name} missing or changed")
    from run_real_observation_s0 import S0_CHECKPOINT_SHA
    rep["cross_checks"] = {
        "s0_checkpoint_identity": files["s0_checkpoint"]["sha256"] == S0_CHECKPOINT_SHA,
        "w102_report_binds_npz": json.loads(C.INPUTS["w102_video_report"][0].read_text())["observation_prediction_sha256"]
        == files["w102_observations_predictions"]["sha256"],
        "blender_5_1_present": BLENDER.exists()}
    for k, ok in rep["cross_checks"].items():
        if not ok:
            rep["stop"].append(f"cross-check failed: {k}")
    rep["inputs"] = files
    # Blender rest rotation (storage-model constant) and calibration on the UNPERTURBED historical FIXED Action only
    blender("scripts/w115_blender_stages.py", "--", "restq", "--rig", files["w105_rig_skinned_fbx"]["path"], "--rest",
            files["w105_rig_rest"]["path"], "--out", str(out / "restq.json"), log=out / "restq.log")
    rq = json.loads((out / "restq.json").read_text())
    rep["blender_rest"] = rq
    cases = json.loads(Path(files["w114_rig_cases"]["path"]).read_text())["cases"]
    audit = json.loads(Path(files["w114_render_audit"]["path"]).read_text())["files"]
    calib = {}
    for name, case in cases.items():
        act = audit[f"{name}/{name}.blend"]["actions"][f"W91 {name} FIXED"]
        keys = [fr["frame_index"] + 1 for fr in case["frames"]]
        pa = W.stored_arrays(W.stored_case_values(case["frames"], "FIXED", rq["rest_rotation_wxyz"]), keys)
        aa = W.audit_arrays(act, keys)
        calib[name] = {c: int(W._neq(pa[c], aa[c]).sum()) for c in W.CH}
    rep["storage_model_calibration_FIXED_only"] = {
        "scope": "W114 case file FIXED slot (unperturbed hypothesis, equal to W105 FIXED) vs the W114 audited FIXED Actions; "
                 "no RUNTIME (perturbed) value was read",
        "mismatched_channel_frames": calib, "exact": all(v == 0 for d in calib.values() for v in d.values())}
    if not rep["storage_model_calibration_FIXED_only"]["exact"]:
        rep["stop"].append("storage model does not reproduce the unperturbed FIXED Action exactly")
    rep["runtime"] = C.runtime()
    rep["pass"] = not rep["stop"]
    dump(out / "preflight.json", rep)
    print(json.dumps({"pass": rep["pass"], "stop": rep["stop"]}))
    return 0 if rep["pass"] else 2


# ------------------------------------------------------------------------------------------- declaration ---

ALLOWLIST = [
    {"id": "PV0_build_root", "where": "build_report.json", "field": ["build_root"], "kind": "build_label",
     "why": "absolute output directory of this build (recorded so allowlisted paths can be checked against it)"},
    {"id": "PV1_build_tag", "where": "build_report.json", "field": ["tag"], "kind": "build_label", "why": "CLI build label"},
    {"id": "PV2_wall_time", "where": "build_report.json", "field": ["cost", "build_seconds"], "kind": "timing", "why": "wall clock"},
    {"id": "PV3_legacy_pose_source", "where": "existing_path/legacy_motion.json", "field": ["source_metadata", "pose_source"],
     "kind": "own_build_path", "referent": "existing_path/legacy_pose_reconstructed.json",
     "why": "W113 legacy_path writes the build's own reconstructed PoseSequence and passes its absolute path as MotionGraph "
            "source_metadata.pose_source (W113 helper, unchanged; W114 V6 omitted it)",
     "causal_byte_dependencies": ["existing_path/legacy_motion.json file bytes and sha256",
                                  "build_report.json existing_path.legacy.motion_graph.source_metadata.pose_source",
                                  "build_report.json existing_path.legacy.motion_graph_sha256"]},
    {"id": "PV3r_report_pose_source", "where": "build_report.json",
     "field": ["existing_path", "legacy", "motion_graph", "source_metadata", "pose_source"], "kind": "own_build_path",
     "referent": "existing_path/legacy_pose_reconstructed.json", "why": "report copy of PV3"},
    {"id": "PV3s_report_motion_graph_sha", "where": "build_report.json", "field": ["existing_path", "legacy", "motion_graph_sha256"],
     "kind": "derived_file_sha", "of": "existing_path/legacy_motion.json",
     "why": "sha256 of the MotionGraph file that embeds PV3; accepted only if it equals the actual sha of the build's own file "
            "and the two MotionGraphs are equal after normalizing PV3 alone"},
    {"id": "PV4_synthetic_source_path", "where": "bundles/synthetic/evidence_meta.json", "field": ["sources", "synthetic_source", "path"],
     "kind": "own_build_path", "referent": "synthetic_source.npz",
     "why": "the W114 synthetic population writes its source npz into the build directory and records its path as evidence "
            "provenance; the recorded sha256 of that source is NOT normalized",
     "causal_byte_dependencies": ["synthetic evidence digest", "synthetic constraints digests (bound to the evidence digest)",
                                  "synthetic edit-record digests and state ids",
                                  "build_report.json synthetic.evidence_digest / synthetic.programme.head_state_id / "
                                  "synthetic.programme.state_ids[*] / digests.synthetic / digests.synthetic_programme_states[*]"]},
]
PV4_DERIVED_REPORT_FIELDS = [["synthetic", "evidence_digest"], ["synthetic", "programme", "head_state_id"],
                             ["synthetic", "programme", "state_ids", "*"], ["digests", "synthetic"],
                             ["digests", "synthetic_programme_states", "*"]]
NEVER_NORMALIZED = ["any source sha256 or digest", "frame digests", "image identity", "model identity / checkpoint sha",
                    "an input path in any non-allowlisted field (identity-bearing)", "frame ordering", "timestamps",
                    "joint names", "constraints / permission digests", "edit parent links", "values / scores / validity"]

ADVERSARIAL = {
    "N1_same_filename_different_pose_bytes": "existing_path/legacy_pose_reconstructed.json rewritten under the same name with one landmark x + 1 px",
    "N1b_same_filename_different_synthetic_source": "synthetic_source.npz rewritten under the same name with one score changed (evidence untouched)",
    "N2a_changed_source_digest": "real evidence sources.w102_observations_predictions.sha256 changed (bundle re-chained consistently)",
    "N2b_changed_frame_digest": "real evidence frame_digests[17] changed (re-chained)",
    "N3_changed_identity_bearing_source_path": "real evidence sources.repo_production_demo_clip.path changed to another file location (re-chained)",
    "N4a_stale_content_hash": "real evidence score array cell changed WITHOUT recomputing the evidence digest",
    "N4b_mismatched_recorded_digest": "real evidence_digest field replaced, content unchanged",
    "N5a_modified_timestamp": "real timestamp[50] + one source frame period (re-chained)",
    "N5b_modified_frame_order": "real rows 10 and 11 swapped in every evidence array (re-chained)",
    "N5c_modified_joint_name": "real joint_names[13] renamed (re-chained)",
    "N6a_modified_edit_parent": "real lineage state 2 re-parented to the root (ids re-chained)",
    "N6b_modified_permission_digest": "real lineage state 1 record bound to the RS1_arm constraints digest (ids re-chained)",
    "N7a_motion_graph_content": "existing_path/legacy_motion.json: one point confidence changed",
    "N7b_animation_clip_content": "existing_path/legacy_animation.json: one rotation component changed",
    "N8a_generated_looking_source_path": "real evidence sources.w102_observations_predictions.path set to <build_root>/existing_path/"
                                         "observations_predictions.npz, the real file copied there (same sha; re-chained)",
    "N8b_pose_source_outside_own_build": "MotionGraph pose_source set to the OTHER build's pose file (same relative name)",
    "N9_synthetic_substantive_change": "synthetic evidence score[2, left_knee] + 0.01 (re-chained; would hide under a prefix exemption)",
    "N10_report_field_under_synthetic_prefix": "build_report synthetic.controls_summary.passed changed",
}
POSITIVE = {"P0_unmutated_copy": "byte copy of build r2 in another directory vs r1", "P1_r1_vs_r2": "two independent builds",
            "P2_self": "r1 vs r1", "P3_r2_vs_r1": "symmetry"}

REAL_EXPECT = {"A1_right_wrist": [], "A2_head": [], "B1_left_wrist": ["lowerarm_l", "upperarm_l"],
               "B2_left_elbow_and_wrist": ["lowerarm_l", "upperarm_l"], "D1_thorax": sorted(R.CHANNELS),
               "D2_left_shoulder": ["lowerarm_l", "spine_01", "upperarm_l"]}
MIXED_EXPECT = {k: sorted(v["expect"]) for k, v in C.RIG_CONDITIONS["mixed"].items()}
C1_STRUCTURALLY_INDEPENDENT = ["__object__", "thigh_l", "calf_l", "thigh_r", "calf_r"]

REPRESENTATIVE = {  # mock edit -> Blender case, constraint set, predicted verdicts (local / world on non-permitted & preserved)
    "M1_E1_under_RS1": {"case": "E_RS1_arm_accepted", "constraints": "RS1", "pred": {"local_nonpermitted": "VIOLATED", "world_nonpermitted": "VIOLATED",
                        "local_preserved": "SATISFIED", "world_preserved": "SATISFIED"}},
    "M1b_E1_under_RS1_arm": {"case": "E_RS1_arm_accepted", "constraints": "RS1_arm", "pred": {"local_nonpermitted": "SATISFIED",
                             "world_nonpermitted": "SATISFIED", "local_preserved": "SATISFIED", "world_preserved": "SATISFIED"}},
    "M2_B1_left_wrist": {"case": "B1_left_wrist", "constraints": "permit_B1_left_wrist", "pred": {"local_nonpermitted": "VIOLATED",
                         "world_nonpermitted": "VIOLATED", "local_preserved": "VACUOUS", "world_preserved": "VACUOUS"}},
    "M3_B2_left_elbow_and_wrist": {"case": "B2_left_elbow_and_wrist", "constraints": "permit_B2_left_elbow_and_wrist",
                                   "pred": {"local_nonpermitted": "SATISFIED", "world_nonpermitted": "SATISFIED",
                                            "local_preserved": "VACUOUS", "world_preserved": "VACUOUS"}},
    "M4_D2_left_shoulder_LW": {"case": "D2_left_shoulder", "constraints": "LW_D2", "pred": {"local_nonpermitted": "SATISFIED",
                               "world_nonpermitted": "VIOLATED", "local_preserved": "SATISFIED", "world_preserved": "VIOLATED"}},
    "M5_D1_thorax": {"case": "D1_thorax", "constraints": "permit_D1_thorax", "pred": {"local_nonpermitted": "VIOLATED",
                     "world_nonpermitted": "VIOLATED", "local_preserved": "VACUOUS", "world_preserved": "VACUOUS"}},
    "M6_A1_right_wrist": {"case": "A1_right_wrist", "constraints": "permit_A1_right_wrist", "pred": {"local_nonpermitted": "SATISFIED",
                          "world_nonpermitted": "SATISFIED", "local_preserved": "VACUOUS", "world_preserved": "VACUOUS"}},
    "M7_C1_float32": {"case": "C1_right_shoulder_along_trunk_axis", "constraints": "permit_C1_right_shoulder_along_trunk_axis",
                      "pred": "REPORTED (sub-resolution numerics; no verdict predicted)"},
    "M7b_C1_float64_diagnostic": {"case": "C1_float64_exact_diagnostic", "constraints": "permit_C1_right_shoulder_along_trunk_axis",
                                  "pred": {"local_nonpermitted": "SATISFIED", "world_nonpermitted": "SATISFIED",
                                           "local_preserved": "VACUOUS", "world_preserved": "VACUOUS"}},
    "M8_mixed_MA1_s0_left_wrist": {"case": "MX_{w}_MA1_s0_left_wrist", "constraints": "mixed permit left_wrist on perturbed rows",
                                   "pred": {"local_nonpermitted": "SATISFIED", "world_nonpermitted": "SATISFIED"}},
    "M9_mixed_MB2_j1x_left_wrist": {"case": "MX_{w}_MB2_j1x_left_wrist", "constraints": "mixed permit left_wrist on perturbed rows",
                                    "pred": {"local_nonpermitted": "VIOLATED", "world_nonpermitted": "VIOLATED"}},
    "M10_E5_cross_frame_copy": {"case": None, "constraints": "RS1", "pred": "UNKNOWABLE (refused: IDENTITY_UNVERIFIED)"},
}

DECISION = {
    "precedence": "D > C > B > A",
    "validity_D": {
        "V1_inputs_authenticated": "preflight pass (instruction, HEAD, lock, inputs, W114 artefacts + frozen scripts) and every stage "
                                   "re-verifies the declaration and input hashes",
        "V2_history_intact": "W113/W114 Worklog blobs unchanged + markers; W114 declaration/decision identities; W114 CASE D",
        "V3_rig_reproduction": "replicated frozen rig reproduces W105 video_cases FIXED (199/199) and both mixed windows (24/24); "
                               "Blender FIXED Actions of every real case equal the W105 FIXED value digest",
        "V4_storage_exact": "for every Blender case and both slots the predicted stored float32 value equals the audited F-curve "
                            "value for every channel, key and component (exact), with identical key sets",
        "V5_repeatability": "semantic comparator: P0-P3 equivalent; every raw byte difference r1/r2 is a declared allowlist entry "
                            "whose guard holds; in-process repeated same-input controls bit-identical",
        "V6_adversarial": "every N-control is rejected (0 false accepts); for re-chained bundle mutations the rejection comes from "
                          "the semantic chain, not only from integrity",
        "V7_synthetic_controls": "W114 synthetic controls/programme replayed: all applicable pass, cross-frame verified positive ACCEPT",
        "V8_numeric_attribution": "C1 float64 exact-direction diagnostic: 0 stored (L2) channel-frame changes; C1 float32: every row with "
                                  "an L1 change has a nonzero stored perpendicular displacement",
        "V9_world_instrument": "evaluated rotations equal audited F-curve values on every keyed channel-frame; FIXED world positions "
                               "bit-identical across cases sharing a FIXED Action; every world point whose dependency channels are "
                               "bit-identical in storage is bit-identical",
        "V10_locks_assessable": "local and world verdicts computed for every representative edit except the declared UNKNOWABLE one",
    },
    "core_C": {
        "G1_evidence_immutable": "as W114 G1 (storage-aware replay)", "G2_edit_locality": "as W114 G2",
        "G3_reversal": "as W114 G3 (R11 channels bit-equal at storage resolution)", "G4_stale_rejection": "as W114 G4",
        "G5_permission": "as W114 G5", "G6_score_semantics": "as W114 G6", "G7_controls_real_bench": "as W114 G7 (storage-aware rig controls)",
        "G8_storage_aware_influence": "(a) unperturbed rows L1 bit-identical in every condition; (b) A1/A2/MA1/MA2 L1 bit-identical on "
                                      "every row and channel; (c) B/D conditions: set of channels with a stored rotation change on "
                                      "perturbed rows equals the declared expectation; (d) C1 float32: Object and the four leg "
                                      "channels L1 bit-identical; (e) every storage-aware candidate authority outcome equals the "
                                      "independent re-derivation and is atomic; (f) audited Action change sets equal predicted "
                                      "stored change sets",
        "G9_authority_domain": "storage-aware programme: R9 OUTSIDE, R10 ACCEPT with changed stored channel-frames exactly "
                               "{upperarm_l: 38, lowerarm_l: 38}, R11 restores channels bit-exactly, R12 NO_RIG_INFLUENCE, R13 ACCEPT "
                               "with 0 changed channel-frames, final state = hypothesis bytes",
    },
    "boundary_B_triggers": {
        "T1_identity": "per-frame subject continuity unverified on >= 1 authentic frame; cross-frame operations fail closed",
        "T2_time": "Action key time differs from source time on >= 1 key; the Action carries no source time",
        "T3_local_world_divergence": ">= 1 representative edit is SATISFIED under local non-permitted locks but VIOLATED under world "
                                     "locks (non-permitted or preserved): local authority is not world-space preservation",
        "T4_joint_vs_bone_authority": "B1 (left_wrist only) changes a stored channel that depicts a joint other than left_wrist",
    },
    "case_rule": "D if any V fails; else C if any G fails; else B if any T holds; else A",
    "A_reachability": "A requires no T. T1 and T2 are structural facts of the authentic data already reported by W114, so A is "
                      "predicted unreachable on these populations; it remains a defined predicate, not a vacuous one",
    "non_claims": "no CASE implies world-space preserve, real occlusion recovery, Generator quality, animation quality or Production "
                  "readiness",
}


def cmd_declare(a):
    pre = json.loads(Path(a.preflight).read_text())
    if not pre.get("pass"):
        raise RuntimeError("STOP: preflight did not pass")
    out = new_dir(a.out_dir)
    decl = {
        "schema": "w115_declaration_v1", "batch": BATCH, "instruction_blob": INSTRUCTION_BLOB, "starting_head": STARTING_HEAD,
        "outcomes_inspected": False,
        "prior_knowledge_disclosure": "W114 outcomes for the SAME interventions are published (W114 Worklog/artefacts) and were read "
                                      "while designing this declaration; expected channel sets for A/B/D conditions equal the W114 "
                                      "code-analysis predictions that W114 observed. Before this file was hashed: the Blender rest "
                                      "rotation was probed, the storage model was calibrated on the UNPERTURBED historical FIXED "
                                      "Action only (exact, all 8 W114 cases), and the posed-joint extraction was smoke-tested on one "
                                      "historical W114 .blend (evaluated rotations = audited F-curves). No W115 build, perturbed W115 "
                                      "output, comparator outcome or lock outcome existed.",
        "inputs": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in pre["inputs"].items()},
        "blender_rest_rotation_wxyz": pre["blender_rest"]["rest_rotation_wxyz"],
        "regimes": {"real": "real_animcv_observation (W102 M1 MMPose+RTMDet, GT-free, private, not unseen; 199 frames, 50 fps)",
                    "benchmark": "benchmark_detector_observation (3DPW shipped detector, courtyard_basketball_01 actor0, VALIDATION, "
                                 "319 rows; not an untouched holdout)",
                    "mixed_rig_windows": "real_animcv_observation on 3DPW images (W102 M_Main S0) + frozen HMR2 + frozen J1X; two W105 "
                                         "windows x 24 rows; rig ownership probe only; identity labels never read",
                    "synthetic": "oracle_geometry (W114 synthetic contract sequence, 8 frames, 25 fps)"},
        "candidate": "UNCHANGED W114 scripts/w114_source_preserving_base.py (blob 4ab86f57); programmes/controls replayed with the "
                     "unchanged W114 contract functions",
        "identical_to_w114": {"datasets": "W102 real clip, W113 3DPW sequence, W105 mixed windows, W114 synthetic", "frames": "all",
                              "perturbations": "C.RIG_CONDITIONS (DELTA 0.01 m camera Z; C1 0.01 m along unit(thorax - pelvis)); even "
                                               "rows perturbed, odd rows locality controls",
                              "directives": "RS1, RS1_arm, RS_head, RS2, RSB, SS1, SS1_arm", "edit_programmes": "W114 real/benchmark/synthetic"},
        "new_w115_parameters": {
            "LW_D2": "permit {thorax, left_shoulder, right_shoulder, left_elbow, left_wrist} on the D2 perturbed rows; preserve "
                     "{right_elbow, right_wrist, neck, head} on all frames; default prohibit (constructed to separate local from "
                     "world locks under the unchanged D2 intervention)",
            "C1_float64_exact_diagnostic": "frozen rig on float64 values (float32 hypothesis widened exactly) + exact float64 0.01 m "
                                           "trunk-axis displacement; DIAGNOSTIC, not a candidate state",
            "storage_aware_rig": "W114 RigModel whose channels are the predicted stored float32 values, tol 0 (exact)",
            "mixed_blender_cases": "the two mixed windows' 5 conditions replayed in Blender (FIXED = recomputed unperturbed, RUNTIME = perturbed)"},
        "numeric_contract": {
            "L0_source": "candidate state float32 (W114 by construction); changed cell iff float32 bytes differ; the C1 stored "
                         "displacement is measured against the float64 trunk axis of the float32 base",
            "L1_case_level": "frozen rig (np.asarray(P, float) -> float64 arithmetic) -> case-level float64 quaternion components; "
                             "bit change reported; continuous pre-cast deltas (hemisphere-aligned max component, geodesic deg) reported",
            "L2_predicted_storage": "bone keys: float32(component) of the case value; Object keys: float32 mathutils product "
                                    "rest_q @ to_q(objq) (BLI mul_qt_qtqt term order); renderer key order (mode locals overwrite "
                                    "extra locals at the same frame); changed iff float32 values differ exactly; stored rotation "
                                    "change iff hemisphere-aligned float32 difference is nonzero (sign flip reported separately)",
            "L3_action": "audited F-curve keyframe values (Blender float32 widened exactly to Python float); changed iff any of the "
                         "four values differs",
            "equality_rules": "exact at every layer; no tolerance is declared or used",
            "classes": {"invariant": "L1 bit-identical", "below_output_resolution": "L1 changed, L2 identical",
                        "material_stored": "L2 changed", "missing_or_contradictory": "NaN pattern mismatch or L2 change without L1 change"},
            "output_resolution_account": "float32 spacing near |q| in [0.5, 1) is 2^-24..2^-23 (5.96e-8..1.19e-7); smaller components "
                                         "have finer spacing, so whether a continuous delta reaches storage depends on the value, not "
                                         "on a global threshold; W114's 1e-9 float64 tolerance is therefore not comparable with exact "
                                         "float32 Action comparison",
            "c1_analytic": "torso_frames: R_t = frame(u, R_sh - L_sh) with perp_u(s) = s - (s.u)u; R_sh + a u leaves perp_u unchanged "
                           "exactly; R_p, u, legs and the left-arm vectors do not read right_shoulder; so C1 is analytically invariant "
                           "for every channel and structurally independent (no code path) for Object and legs"},
        "path_semantics": {"allowlist": ALLOWLIST, "pv4_derived_report_fields": PV4_DERIVED_REPORT_FIELDS,
                           "never_normalized": NEVER_NORMALIZED,
                           "guard": "a declared field is normalized only if its value lies under the build's own recorded "
                                    "build_root; the remainder must be identical across builds; the referent file bytes must be "
                                    "equal across builds and equal the recorded sha256 where one is recorded",
                           "method": "normalize-then-rehash: placeholder substituted ONLY in the declared fields, then the evidence "
                                     "digest, constraints digests, edit-record digests and state ids are recomputed in lineage order "
                                     "and compared; every stored digest is first re-verified against its own content (integrity)",
                           "w114_v6_diagnosis": "W114 compare exempted '.synthetic*' and '.digests.synthetic*' by PREFIX and omitted "
                                                "the pose_source / MotionGraph sha fields"},
        "adversarial_controls": ADVERSARIAL, "positive_controls": POSITIVE,
        "rig": {"real_expected_stored_channel_sets": REAL_EXPECT, "mixed_expected_stored_channel_sets": MIXED_EXPECT,
                "c1": {"float32": "classification reported, not gated except structural independence of " + ", ".join(C1_STRUCTURALLY_INDEPENDENT),
                       "float64_diagnostic": "0 stored channel-frame changes (V8)"},
                "unconsumed_no_effect": ["A1_right_wrist", "A2_head", "MA1_s0_left_wrist", "MA2_s0_left_shoulder"],
                "authority_rule": "W114 rule unchanged (tail-joint projection; preserved / not-permitted depicted joint -> OUTSIDE; "
                                  "rig-claimed no-change -> NO_RIG_INFLUENCE) applied to STORED channel changes"},
        "lock_semantics": {
            "local": "a locked (frame, joint) cell holds iff every keyed channel whose tail-joint projection depicts the joint has "
                     "bit-identical stored (audited Action) values at that frame; joints depicted by no keyed channel hold trivially "
                     "(counted separately)",
            "world": "a locked cell holds iff the Blender-evaluated world position of the joint's declared rig point is bit-identical "
                     "at that frame; displacement magnitudes reported, never thresholded",
            "world_points": W.WORLD_POINT,
            "lock_sets": {"nonpermitted": "every (frame, joint) cell not permitted by the edit's constraint set (preserve + default "
                                          "prohibit)", "preserved": "cells explicitly preserved by the constraint set (VACUOUS if none)"},
            "policy": "UNRESOLVED HUMAN PRODUCT CHOICE. Both are reported per edit; neither is adopted; a locally allowed edit that "
                      "moves a world-locked joint is never called user-preserving"},
        "representative_edits": REPRESENTATIVE,
        "identity_time": {"cross_frame_rule": "unchanged W114: copy_from_frame needs verified continuity on every involved frame",
                          "time": "source 50 fps vs Action 30 fps at key frame_index + 1; sidecar recovery is not Action/FBX time"},
        "decision": DECISION,
        "stop_conditions": ["required artefact fails authentication", "a canonicalization hides a real source mutation (V6)",
                            "numerical attribution would require an outcome-fitted tolerance (V4/V8 fail -> D, no tolerance is "
                            "introduced)", "an ambiguous lock policy must be decided to continue (never: both reported)",
                            "source-to-rig effects cannot be separated (V8/V9)", "any Production/schema/model/rig/exporter change"],
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


# ------------------------------------------------------------------------------------------------- build ---

class capture_lineages:
    """Record every W114 Lineage constructed inside the block (the W114 functions are called unchanged)."""

    def __enter__(self):
        self.orig, self.items = B.Lineage, []
        items, orig = self.items, self.orig

        class Recorded(orig):
            def __init__(s, ev):
                super().__init__(ev)
                items.append(s)
        B.Lineage = Recorded
        return self

    def __exit__(self, *exc):
        B.Lineage = self.orig

    def find(self, order):
        hits = [l for l in self.items if list(l.order) == list(order)]
        if len(hits) < 1:
            raise RuntimeError("programme lineage not captured")
        return hits[0]


def real_frames_fn(ts, rc):
    return lambda values: R.real_case_frames(values, ts, rc, "x")


def layers(frames, slot, rq):
    keys = list(range(1, len(frames) + 1))
    return W.case_channels(frames, slot), W.stored_arrays(W.stored_case_values(frames, slot, rq), keys)


def summarize(cls):
    return {c: {k: v for k, v in r.items() if not k.startswith("changed_rows")} for c, r in cls.items()}


def stored_set(cls):
    return sorted(c for c, r in cls.items() if r["material_stored_rotation"])


def real_conditions(ev, rc, rq, srig):
    fi, ts = ev["arrays"]["frame_index"], ev["arrays"]["timestamp"]
    base = np.array(ev["arrays"]["est3d_xyz_hypothesis"])
    base_frames = R.real_case_frames(base, ts, rc, "h")
    L1b, L2b = layers(base_frames, "SLOT", rq)
    rows_p, rows_u = np.flatnonzero(fi % 2 == 0), np.flatnonzero(fi % 2 == 1)
    out, states, constraints = {}, {}, {}
    u = base[:, B.JI["thorax"]].astype(np.float64) - base[:, B.JI["pelvis"]].astype(np.float64)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    for name, cond in C.RIG_CONDITIONS["real"].items():
        cells = [(int(fi[r]), n) for r in rows_p for n in cond["joints"]]
        delta = (0.01 * u[rows_p]).tolist() if cond.get("along_trunk_axis") else DELTA
        permit = B.make_constraints(ev, f"permit_{name}", [{"id": "P", "kind": "permit", "joints": cond["joints"],
                                                             "spans": [[int(fi[r]), int(fi[r])] for r in rows_p]}])
        constraints[f"permit_{name}"] = permit
        lin = B.Lineage(ev)
        st = B.apply_edit(lin, ev, permit, B.make_mock_edit(lin, ev, permit, name, cells, delta))
        fr = R.real_case_frames(st["values"], ts, rc, name)
        L1a, L2a = layers(fr, "SLOT", rq)
        cp, cu = W.classify(L1b, L1a, L2b, L2a, rows_p), W.classify(L1b, L1a, L2b, L2a, rows_u)
        src_changed = ~((st["values"] == base) | (np.isnan(st["values"]) & np.isnan(base))).all(-1)
        lin2 = B.Lineage(ev)
        code, _, r = C.attempt(lambda: B.apply_edit(lin2, ev, permit, B.make_mock_edit(lin2, ev, permit, name, cells, delta, rig_claim=True),
                                                    rig=srig), lin2)
        indep = C.independent_authority(srig, ev, permit, base, st["values"], True)
        rec = {"kind": cond["kind"], "joints": cond["joints"], "perturbed_rows": int(len(rows_p)),
               "source_layer": {"changed_cells": int(src_changed.sum()),
                                "changed_cells_per_joint": {n: int(src_changed[:, B.JI[n]].sum()) for n in B.JOINTS if src_changed[:, B.JI[n]].any()},
                                "changed_cells_on_unperturbed_rows": int(src_changed[rows_u].sum())},
               "perturbed": summarize(cp), "unperturbed": summarize(cu),
               "unperturbed_rows_l1_bit_changes": int(sum(v["l1_bit_changed"] for v in cu.values())),
               "stored_rotation_channels": stored_set(cp),
               "expected_stored_channels": REAL_EXPECT.get(name),
               "authority_storage_aware": {"observed": code, "independent_rule": indep, "consistent": code == indep,
                                           "atomic": r.get("atomic_head_unchanged", True), "w114_prediction": cond["authority"]}}
        if cond.get("along_trunk_axis"):
            stored = st["values"][rows_p, B.JI["right_shoulder"]].astype(np.float64) - base[rows_p, B.JI["right_shoulder"]].astype(np.float64)
            par = (stored * u[rows_p]).sum(-1, keepdims=True) * u[rows_p]
            perp = np.linalg.norm(stored - par, axis=-1)
            any_l1 = np.zeros(len(fi), bool)
            for c in W.CH:
                any_l1[cp[c]["changed_rows_l1"]] = True
            rec["c1_stored_displacement"] = {
                "perpendicular_max_m": float(perp.max()), "perpendicular_median_m": float(np.median(perp)),
                "rows_with_zero_perpendicular": int((perp == 0).sum()),
                "parallel_minus_0p01_max_abs_m": float(np.abs(np.linalg.norm(par, axis=-1) - 0.01).max()),
                "rows_l1_changed": int(any_l1[rows_p].sum()),
                "rows_l1_changed_with_zero_perpendicular": int((any_l1[rows_p] & (perp == 0)).sum())}
            rec["changed_rows_l1"] = {c: cp[c]["changed_rows_l1"] for c in W.CH}
            rec["changed_rows_l2"] = {c: cp[c]["changed_rows_l2"] for c in W.CH}
        out[name] = rec
        states[name] = st["values"]
    # C1 float64 exact-direction DIAGNOSTIC (outside the candidate)
    base64 = base.astype(np.float64)
    p64 = base64.copy()
    p64[rows_p, B.JI["right_shoulder"]] += 0.01 * u[rows_p]
    f64b = R.real_case_frames(base64, ts, rc, "h64")
    f64a = R.real_case_frames(p64, ts, rc, "c1_64")
    L1b64, L2b64 = layers(f64b, "SLOT", rq)
    L1a64, L2a64 = layers(f64a, "SLOT", rq)
    c64 = W.classify(L1b64, L1a64, L2b64, L2a64, rows_p)
    out["C1_float64_exact_diagnostic"] = {
        "label": "DIAGNOSTIC: float64 values + exact float64 displacement; not a candidate state, not a Production computation",
        "base64_frames_equal_float32_frames": json.dumps(clean(f64b)) == json.dumps(clean(base_frames)),
        "perturbed": summarize(c64), "stored_change_channel_frames": int(sum(v["material_stored"] for v in c64.values())),
        "l1_changed_channel_frames": int(sum(v["l1_bit_changed"] for v in c64.values()))}
    states["C1_float64_exact_diagnostic"] = p64
    return out, states, constraints, base, base_frames


def mixed_conditions(decl, rc, rq):
    cases = json.loads(Path(decl["inputs"]["w105_cases"]["path"]).read_text())["cases"]
    pm = np.load(decl["inputs"]["w105_sources"]["path"])["pred_M"]
    res, blender_cases = {}, {}
    for wname in C.MIXED_WINDOWS:
        case = cases[wname]
        src = R.MixedSources(case, pm, decl["inputs"]["w102_hmr_pred_full"]["path"], decl["inputs"]["w102_j1x_predictions_3dpw"]["path"])
        slots = R.mixed_slots(src, rc)
        times = [float(f.get("time_s", i)) for i, f in enumerate(case["frames"])]
        base_frames = W.slot_frames(slots, times, "base")
        L1b, L2b = layers(base_frames, "SLOT", rq)
        n = len(src.rows)
        rows_p, rows_u = np.arange(0, n, 2), np.arange(1, n, 2)
        w = {"sequence_id": case["sequence_id"], "frames": n, "split": sorted({f["split"] for f in case["frames"]}),
             "reproduction": R.mixed_reproduction(src, slots), "conditions": {}}
        for name, cond in C.RIG_CONDITIONS["mixed"].items():
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
            after_frames = W.slot_frames(R.mixed_slots(src, rc, **kw), times, name)
            L1a, L2a = layers(after_frames, "SLOT", rq)
            cp, cu = W.classify(L1b, L1a, L2b, L2a, rows_p), W.classify(L1b, L1a, L2b, L2a, rows_u)
            ca = W.classify(L1b, L1a, L2b, L2a)
            w["conditions"][name] = {"kind": cond["kind"], "source": cond["source"], "perturbed": summarize(cp),
                                     "unperturbed_rows_l1_bit_changes": int(sum(v["l1_bit_changed"] for v in cu.values())),
                                     "all_rows_l1_bit_changes": int(sum(v["l1_bit_changed"] for v in ca.values())),
                                     "stored_rotation_channels": stored_set(cp), "expected_stored_channels": MIXED_EXPECT[name]}
            key = f"MX_{wname}_{name}"
            blender_cases[key] = {"case": key, "sequence_id": case["sequence_id"], "start_frame": 0, "fps_sampled": case.get("fps_sampled"),
                                  "frames": R.merge_slots(base_frames, after_frames, {"FIXED": "W115 recomputed W102 contract (unperturbed)",
                                                                                      "RUNTIME": f"W115 {name} (MOCK)"}),
                                  "labels": {"FIXED": "unperturbed", "RUNTIME": f"{name} MOCK perturbation"},
                                  "short_labels": {"FIXED": "H", "RUNTIME": "P"},
                                  "title": f"W115 {key} | frozen W102 mixed contract | MOCK source perturbation, not animation",
                                  "header_note": "diagnostic replay of source-to-rig influence; not quality evidence"}
        res[wname] = w
    return res, blender_cases


def real_blender_cases(ev, rc, base_frames, states):
    ts = ev["arrays"]["timestamp"]
    cases = {}
    for name, vals in states.items():
        fr = R.merge_slots(base_frames, R.real_case_frames(vals, ts, rc, name),
                           {"FIXED": "W115 unperturbed hypothesis", "RUNTIME": f"W115 {name} (MOCK)"})
        cases[name] = {"case": name, "sequence_id": C.REAL_SEQUENCE_ID, "start_frame": 0, "fps_sampled": 50.0, "frames": fr,
                       "labels": {"FIXED": "hypothesis (W105 M)", "RUNTIME": f"{name} MOCK perturbation"},
                       "short_labels": {"FIXED": "H", "RUNTIME": "P"},
                       "title": f"W115 {name} | frozen W105 S0-only rig | MOCK source state, not animation",
                       "header_note": "diagnostic replay of source-to-rig influence; not quality evidence"}
    return cases


def lw_d2_constraints(ev):
    fi = ev["arrays"]["frame_index"]
    rows_p = np.flatnonzero(fi % 2 == 0)
    return B.make_constraints(ev, "LW_D2", [
        {"id": "E_D2", "kind": "permit", "joints": ["thorax", "left_shoulder", "right_shoulder", "left_elbow", "left_wrist"],
         "spans": [[int(fi[r]), int(fi[r])] for r in rows_p]},
        {"id": "P_world", "kind": "preserve", "joints": ["right_elbow", "right_wrist", "neck", "head"], "spans": [[int(fi[0]), int(fi[-1])]]}])


def cmd_build(a):
    decl = load_declaration(a)
    out = new_dir(a.out_dir)
    root = str(out.resolve())
    t0 = time.perf_counter()
    rq = decl["blender_rest_rotation_wxyz"]
    rep = {"schema": "w115_build_v1", "declaration_sha256": a.declaration_sha, "tag": a.tag, "build_root": root, "GT_read": False}
    bundles = new_dir(out / "bundles")
    # ---- synthetic (unchanged W114 replay; toy rig as declared by W114)
    sev, sref = C.synthetic_evidence(Path(root))
    scs = C.synth_directives(sev)
    toy = C.toy_rig()
    s_ok = [(3, "left_wrist"), (4, "left_wrist"), (5, "left_wrist")]
    with capture_lineages() as cap:
        sprog = C.edit_programme("synthetic", sev, scs["SS1"], s_ok, cs_arm=scs["SS1_arm"], rig=toy, cs_head=scs["SS1"],
                                 head_cells=[(3, "head"), (5, "head")], cross=(5, (4, "left_wrist")), E2_filter=lambda c: c[0] == 3)
    W.save_bundle(bundles / "synthetic", sev, scs, cap.find(sprog["state_ids"]))
    rep["synthetic"] = {"evidence_digest": sev["meta"]["evidence_digest"], "scores": B.score_accounting(sev, 0.3),
                        "source_exact": C._try(lambda: B.verify_against_source(sev, sref)),
                        "controls": C.run_controls("synthetic", sev, sref, scs["SS1"], s_ok, (1, "left_wrist"), (4, "left_knee"), rig=toy,
                                                   cs_unconsumed=scs["SS1"], unconsumed_cells=[(3, "head")], out_of_domain=s_ok,
                                                   cross=(5, (4, "left_wrist"))),
                        "programme": sprog}
    lin = B.Lineage(sev)
    pos = B.make_mock_edit(lin, sev, scs["SS1"], "X_verified", [(3, "left_wrist")], op="copy_from_frame", source_frame=2)
    code, _, r = C.attempt(lambda: B.apply_edit(lin, sev, scs["SS1"], pos), lin)
    rep["synthetic"]["cross_frame_verified_positive"] = dict(r, expected="ACCEPT", ok=code == "ACCEPT")
    rep["synthetic"]["controls_summary"] = C.control_summary(rep["synthetic"]["controls"])
    # ---- real
    z, wrep, insp = C.real_inputs(decl)
    ev, ref = C.real_evidence(decl, z, wrep, insp)
    ev_again, _ = C.real_evidence(decl, z, wrep, insp)
    rc = R.rig_context()
    fi, ts = ev["arrays"]["frame_index"], ev["arrays"]["timestamp"]
    srig = W.storage_rig("w105_real_s0_only_storage_float32", real_frames_fn(ts, rc), rq, R.DEPICTS, R.OWNERS["real_s0_only"])
    sets, ok_cells, spans = C.real_directives(ev)
    sets["LW_D2"] = lw_d2_constraints(ev)
    masks = B.resolve_constraints(ev, sets["RS1"])
    head_cells = [(f, "head") for f, _ in ok_cells]
    rep["real"] = {"evidence_digest": ev["meta"]["evidence_digest"], "verify": C._try(lambda: B.verify_evidence(ev)),
                   "source_exact": C._try(lambda: B.verify_against_source(ev, ref)), "scores": B.score_accounting(ev, 0.3),
                   "permit_spans": spans, "permitted_cells": len(ok_cells),
                   "RS1_masks": {"preserve": int(masks["preserve"].sum()), "permit": int(masks["permit"].sum()),
                                 "prohibited_default": int((~masks["preserve"] & ~masks["permit"]).sum())},
                   "RS2": C._try(lambda: B.resolve_constraints(ev, sets["RS2"])),
                   "support": {B.EST3D_SUPPORT[k]: int((ev["arrays"]["est3d_support"] == k).sum()) for k in B.EST3D_SUPPORT},
                   "derived_2d_cells": int(sum(not k.startswith("observed_2d") for k in ev["meta"]["obs2d_kind"].values()) * len(fi)),
                   "invalid_input_cells": int((~z["input_valid"]).sum()),
                   "present_cells": int(ev["arrays"]["obs2d_present"].sum()), "valid_cells": int(ev["arrays"]["input_valid"].sum())}
    rep["real"]["controls"] = C.run_controls("real", ev, ref, sets["RS1"], ok_cells, (50, "left_wrist"), (150, "left_knee"), rig=srig,
                                             cs_unconsumed=sets["RS_head"], unconsumed_cells=head_cells, out_of_domain=ok_cells,
                                             cross=(108, (109, "left_wrist")))
    rep["real"]["controls_summary"] = C.control_summary(rep["real"]["controls"])
    with capture_lineages() as cap:
        rprog = C.edit_programme("real", ev, sets["RS1"], ok_cells, cs_arm=sets["RS1_arm"], rig=srig, cs_head=sets["RS_head"],
                                 head_cells=head_cells, cross=(108, (109, "left_wrist")), E2_filter=lambda c: 109 <= c[0] <= 118)
    rep["real"]["programme"] = rprog
    # ---- rig influence (storage-aware), real + C1 float64 diagnostic + LW_D2 authority
    conds, states, cons, base_vals, base_frames = real_conditions(ev, rc, rq, srig)
    rep["rig_real"] = conds
    sets.update(cons)
    lin = B.Lineage(ev)
    d2_cells = [(int(fi[rr]), "left_shoulder") for rr in np.flatnonzero(fi % 2 == 0)]
    code, _, r = C.attempt(lambda: B.apply_edit(lin, ev, sets["LW_D2"], B.make_mock_edit(lin, ev, sets["LW_D2"], "LW_D2", d2_cells, DELTA),
                                                rig=srig), lin)
    rep["rig_real"]["D2_left_shoulder"]["LW_D2_candidate_authority"] = {
        "observed": code, "independent_rule": C.independent_authority(srig, ev, sets["LW_D2"], base_vals, states["D2_left_shoulder"], False),
        "atomic": r.get("atomic_head_unchanged", True), "lineage_head_values_equal_D2_state": code == "ACCEPT"
        and lin.state()["values"].tobytes() == states["D2_left_shoulder"].tobytes()}
    lin = B.Lineage(ev)
    acc = B.apply_edit(lin, ev, sets["RS1_arm"], B.make_mock_edit(lin, ev, sets["RS1_arm"], "E_RS1_arm", ok_cells, DELTA), rig=srig)
    states["E_RS1_arm_accepted"] = acc["values"]
    rep["E_RS1_arm_association"] = acc["record"]["output_association"]["changed_channel_frames"]
    rep["w114_tolerance_reproduction_C1"] = {c: v["changed_frames"] for c, v in R.changed_sets(
        R.real_channels(base_vals, rc), R.real_channels(states["C1_right_shoulder_along_trunk_axis"], rc), C.RIG_TOL,
        np.flatnonzero(fi % 2 == 0)).items()}
    # ---- real rig reproduction + in-process repeated same-input controls
    vc = json.loads(Path(decl["inputs"]["w105_video_cases"]["path"]).read_text())["cases"]["real_clip"]["frames"]
    mine = json.loads(json.dumps(clean(R.slot_view(base_frames, "SLOT"))))
    rep["rig_reproduction_real"] = {"frames": len(vc), "byte_equal_frames": sum(x == y for x, y in zip(mine, R.slot_view(vc, "FIXED")))}
    again = R.real_case_frames(np.array(ev_again["arrays"]["est3d_xyz_hypothesis"]), ev_again["arrays"]["timestamp"], rc, "h")
    rep["in_process_repeat"] = {"real_evidence_digest_equal": ev_again["meta"]["evidence_digest"] == ev["meta"]["evidence_digest"],
                                "real_base_case_frames_equal": json.dumps(clean(again)) == json.dumps(clean(base_frames)),
                                "storage_rig_channels_equal": all(np.array_equal(x, y, equal_nan=True) for x, y in zip(
                                    srig.channels(base_vals).values(), srig.channels(base_vals).values()))}
    # ---- mixed windows
    rep["rig_mixed"], mixed_cases = mixed_conditions(decl, rc, rq)
    # ---- Blender case file (real + C1 diagnostic + accepted RS1_arm edit + mixed windows)
    cases = real_blender_cases(ev, rc, base_frames, states)
    cases.update(mixed_cases)
    dump(out / "w115_rig_cases.json", {"rest_snapshot_digest": rc["rest"].digest(), "canonical_to_armature_xyzw": list(rc["alignment"]),
                                       "target_chain": {"root_head": list(rc["target"].root_head)}, "cases": cases})
    # ---- lock specification (masks of each representative edit's constraint set)
    lock_spec = {}
    for mname, m in REPRESENTATIVE.items():
        cs_id = m["constraints"]
        if cs_id in sets:
            mk = B.resolve_constraints(ev, sets[cs_id])
            lock_spec[mname] = {"case": m["case"], "constraints": cs_id, "nonpermitted": (~mk["permit"]).astype(int).tolist(),
                                "preserved": mk["preserve"].astype(int).tolist()}
    rep["lock_spec_digest"] = hashlib.sha256(B.canonical(lock_spec).encode()).hexdigest()
    dump(out / "lock_spec.json", lock_spec)
    # ---- time, identity
    rep["time_identity"] = C.time_identity(ev)
    rep["real"]["evidence_unchanged_after_rig"] = ev["meta"]["evidence_digest"] == B.evidence_digest(ev)
    W.save_bundle(bundles / "real", ev, {k: v for k, v in sets.items()}, cap.find(rprog["state_ids"]))
    # ---- existing path (unchanged W113 helpers via W114)
    rep["existing_path"] = C.existing_path(decl, z, wrep, insp, masks["preserve"], Path(root))
    # ---- benchmark
    bev, bref, binfo = C.bench_evidence(decl)
    bcs, bcells = C.bench_directives(bev)
    bm = B.resolve_constraints(bev, bcs)
    pres = np.argwhere(bm["preserve"])
    bfi = bev["arrays"]["frame_index"]
    r_ = int(np.flatnonzero(bfi == bcells[0][0])[0])
    bcross = (int(bfi[r_ - 1]), bcells[0]) if r_ > 0 else None
    rep["benchmark"] = {"info": binfo, "evidence_digest": bev["meta"]["evidence_digest"], "verify": C._try(lambda: B.verify_evidence(bev)),
                        "source_exact": C._try(lambda: B.verify_against_source(bev, bref)), "scores": B.score_accounting(bev),
                        "permitted_cells": len(bcells), "preserved_cells": int(bm["preserve"].sum()),
                        "support": {B.EST3D_SUPPORT[k]: int((bev["arrays"]["est3d_support"] == k).sum()) for k in B.EST3D_SUPPORT},
                        "present_cells": int(bev["arrays"]["obs2d_present"].sum())}
    unsup = ~np.isfinite(bev["arrays"]["est3d_xyz_hypothesis"]).all(-1)
    prohibited = (int(bfi[np.argwhere(unsup)[0][0]]), B.JOINTS[np.argwhere(unsup)[0][1]]) if unsup.any() else None
    rep["benchmark"]["controls"] = C.run_controls("benchmark", bev, bref, bcs, bcells, (int(bfi[pres[0][0]]), B.JOINTS[pres[0][1]]),
                                                  prohibited or (int(bfi[pres[0][0]]), B.JOINTS[pres[0][1]]), cross=bcross)
    if prohibited is None:
        rep["benchmark"]["controls"]["unauthorized_edit"] = {
            "expected": "EDIT_NOT_AUTHORIZED", "observed": "NOT_APPLICABLE", "ok": None,
            "reason": "RSB covers every finite cell with preserve or permit; no default-prohibited cell exists"}
    rep["benchmark"]["controls_summary"] = C.control_summary(rep["benchmark"]["controls"])
    with capture_lineages() as cap:
        bprog = C.bench_programme(bev, bcs, bcells, bcross)
    rep["benchmark"]["programme"] = bprog
    W.save_bundle(bundles / "benchmark", bev, {"RSB": bcs}, cap.find(bprog["state_ids"]))
    rep["digests"] = {"synthetic": sev["meta"]["evidence_digest"], "real": ev["meta"]["evidence_digest"],
                      "benchmark": bev["meta"]["evidence_digest"], "real_programme_states": rprog.get("state_ids"),
                      "benchmark_programme_states": bprog.get("state_ids"), "synthetic_programme_states": sprog.get("state_ids"),
                      "rig_case_file": sha(out / "w115_rig_cases.json")}
    rep["cost"] = {"build_seconds": time.perf_counter() - t0}
    dump(out / "build_report.json", rep)
    print(json.dumps({"tag": a.tag, "seconds": round(rep["cost"]["build_seconds"], 2), "real_digest": rep["digests"]["real"]}))
    return 0


# ------------------------------------------------------------------------------------- semantic comparison ---

def _files(d: Path):
    return sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())


def compare_builds(d1: Path, d2: Path) -> dict:
    """Field-scoped semantic comparison of two build directories (declared allowlist only)."""
    d1, d2 = Path(d1), Path(d2)
    r1, r2 = json.loads((d1 / "build_report.json").read_text()), json.loads((d2 / "build_report.json").read_text())
    roots = (r1.get("build_root", ""), r2.get("build_root", ""))
    f1, f2 = _files(d1), _files(d2)
    res = {"raw_identical_files": [], "raw_differing_files": [], "declared_differences": [], "unexplained": [], "checks": {}}
    for f in sorted(set(f1) ^ set(f2)):
        res["unexplained"].append(f"file present in one build only: {f}")
    common = sorted(set(f1) & set(f2))
    for f in common:
        (res["raw_identical_files"] if sha(d1 / f) == sha(d2 / f) else res["raw_differing_files"]).append(f)
    differing = set(res["raw_differing_files"])

    def guard(value_a, value_b, referent):
        ra, rb = W.own_root_relative(value_a, roots[0]), W.own_root_relative(value_b, roots[1])
        if ra is None or rb is None:
            return False, "value does not lie under the build's own recorded root"
        if ra != rb or ra != referent:
            return False, f"relative remainder differs or is not the declared referent ({ra} / {rb})"
        fa, fb = d1 / ra, d2 / rb
        if not fa.exists() or not fb.exists() or sha(fa) != sha(fb):
            return False, "referent bytes differ across builds"
        return True, "ok"

    # ---- existing-path MotionGraph (PV3)
    mg = "existing_path/legacy_motion.json"
    mg_equal_norm = False
    if mg in common:
        g1, g2 = json.loads((d1 / mg).read_text()), json.loads((d2 / mg).read_text())
        ok, why = guard(g1["source_metadata"]["pose_source"], g2["source_metadata"]["pose_source"], ALLOWLIST[3]["referent"])
        n1, n2 = copy.deepcopy(g1), copy.deepcopy(g2)
        n1["source_metadata"]["pose_source"] = n2["source_metadata"]["pose_source"] = W.PLACEHOLDER
        mg_equal_norm = ok and B.canonical(n1) == B.canonical(n2)
        res["checks"]["PV3_motion_graph"] = {"guard": why, "normalized_equal": mg_equal_norm,
                                             "residual_fields": [".".join(map(str, p)) for p in W.json_diff(n1, n2)][:20]}
        if mg in differing:
            if mg_equal_norm:
                res["declared_differences"].append({"file": mg, "id": "PV3_legacy_pose_source"})
            else:
                res["unexplained"].append(f"{mg}: not equal after normalizing PV3 only ({why})")
            differing.discard(mg)
    # ---- bundles
    for pop in ("synthetic", "real", "benchmark"):
        b1p, b2p = d1 / "bundles" / pop, d2 / "bundles" / pop
        if not b1p.exists() or not b2p.exists():
            res["unexplained"].append(f"bundle {pop} missing")
            continue
        try:
            b1, b2 = W.load_bundle(b1p), W.load_bundle(b2p)
        except Exception as e:  # unreadable bundle
            res["unexplained"].append(f"bundle {pop} unreadable: {e}")
            continue
        i1, i2 = W.bundle_integrity(b1), W.bundle_integrity(b2)
        allow = [e["field"] for e in ALLOWLIST if e["where"] == f"bundles/{pop}/evidence_meta.json"]
        c1, ref1, p1 = W.normalized_chain(b1, allow, roots[0], d1)
        c2, ref2, p2 = W.normalized_chain(b2, allow, roots[1], d2)
        ref_ok = True
        for k in set(ref1) | set(ref2):
            a_, b_ = ref1.get(k), ref2.get(k)
            recorded = (W.get_field(b1["meta"], k.split(".")[:-1] + ["sha256"]), W.get_field(b2["meta"], k.split(".")[:-1] + ["sha256"]))
            entry = next(e for e in ALLOWLIST if ".".join(e["field"]) == k)
            ref_ok &= bool(a_ and b_ and a_["relative"] == b_["relative"] == entry["referent"] and a_["sha256"] == b_["sha256"]
                           and a_["sha256"] == recorded[0] and b_["sha256"] == recorded[1])
        chain_eq = c1 == c2
        res["checks"][f"bundle_{pop}"] = {"integrity": [i1, i2], "guard_problems": p1 + p2, "referents_ok": ref_ok,
                                          "normalized_chain_equal": chain_eq, "raw_evidence_digest_equal":
                                          b1["meta"].get("evidence_digest") == b2["meta"].get("evidence_digest"),
                                          "normalized_evidence_digest": [c1["evidence_digest"], c2["evidence_digest"]]}
        bundle_files = [f for f in list(differing) if f.startswith(f"bundles/{pop}/")]
        sem_ok = i1["ok"] and i2["ok"] and not p1 and not p2 and ref_ok and chain_eq
        if not sem_ok:
            res["unexplained"].append(f"bundle {pop}: semantic chain/integrity/guard failure")
        elif bundle_files:
            if not allow:
                res["unexplained"].append(f"bundle {pop}: raw bytes differ but the population has no allowlisted field")
            else:
                for f in bundle_files:
                    res["declared_differences"].append({"file": f, "id": "PV4_synthetic_source_path"})
        for f in bundle_files:
            differing.discard(f)
        res["checks"][f"bundle_{pop}"]["semantic_ok"] = sem_ok
    # ---- build report (field-scoped)
    if "build_report.json" in differing or True:
        diffs = W.json_diff(r1, r2)
        pv4_ok = res["checks"].get("bundle_synthetic", {}).get("semantic_ok", False)
        for p in diffs:
            ent = next((e for e in ALLOWLIST if e["where"] == "build_report.json" and list(p) == e["field"]), None)
            if ent is not None:
                if ent["kind"] in ("build_label", "timing"):
                    res["declared_differences"].append({"file": "build_report.json", "field": list(p), "id": ent["id"]})
                    continue
                if ent["kind"] == "own_build_path":
                    ok, why = guard(W.get_field(r1, p), W.get_field(r2, p), ent["referent"])
                    if ok:
                        res["declared_differences"].append({"file": "build_report.json", "field": list(p), "id": ent["id"]})
                        continue
                    res["unexplained"].append(f"build_report {'.'.join(map(str, p))}: guard failed ({why})")
                    continue
                if ent["kind"] == "derived_file_sha":
                    ok = (W.get_field(r1, p) == sha(d1 / ent["of"]) and W.get_field(r2, p) == sha(d2 / ent["of"]) and mg_equal_norm)
                    if ok:
                        res["declared_differences"].append({"file": "build_report.json", "field": list(p), "id": ent["id"]})
                        continue
                    res["unexplained"].append(f"build_report {'.'.join(map(str, p))}: derived sha not explained")
                    continue
            if any(W.matches(list(p), pat) for pat in PV4_DERIVED_REPORT_FIELDS):
                own = _pv4_derived_consistent(r1, d1, p) and _pv4_derived_consistent(r2, d2, p)
                if pv4_ok and own:
                    res["declared_differences"].append({"file": "build_report.json", "field": list(p), "id": "PV4_synthetic_source_path (derived)"})
                    continue
                res["unexplained"].append(f"build_report {'.'.join(map(str, p))}: PV4-derived field not explained")
                continue
            res["unexplained"].append(f"build_report {'.'.join(map(str, p))}")
        differing.discard("build_report.json")
    for f in sorted(differing):
        res["unexplained"].append(f"raw bytes differ, no declared semantics: {f}")
    res["equivalent"] = not res["unexplained"]
    return res


def _pv4_derived_consistent(rep, d, path):
    """A PV4-derived report value must equal the corresponding raw digest in the build's own synthetic bundle."""
    b = W.load_bundle(Path(d) / "bundles" / "synthetic")
    v = W.get_field(rep, path)
    states = [s["state_id"] for s in b["states"]]
    if path[-1] == "evidence_digest" or list(path) == ["digests", "synthetic"]:
        return v == b["meta"]["evidence_digest"]
    if path[-1] == "head_state_id":
        return v == states[-1]
    return isinstance(path[-1], int) and path[-1] < len(states) and v == states[path[-1]]


def _rechain(bdir: Path, mutate):
    """Apply `mutate(bundle)` to a bundle on disk and recompute every digest consistently (a self-consistent forgery)."""
    bd = W.load_bundle(bdir)
    mutate(bd)
    ev = {"meta": bd["meta"], "arrays": bd["arrays"]}
    ed = B.evidence_digest(ev)
    bd["meta"]["evidence_digest"] = ed
    cmap = {}
    for cid, c in bd["constraints"].items():
        old = B.constraints_digest(c)
        c["evidence_digest"] = ed
        cmap[old] = B.constraints_digest(c)
    smap = {None: None}
    for i, s in enumerate(bd["states"]):
        rec = s["record"]
        if rec is not None:
            rec["evidence_digest"] = ed
            if "constraints_digest" in rec:
                rec["constraints_digest"] = cmap.get(rec["constraints_digest"], rec["constraints_digest"])
            rec["base_state_id"] = smap[rec["base_state_id"]] if rec["base_state_id"] in smap else rec["base_state_id"]
        s["parent"] = smap.get(s["parent"], s["parent"])
        v, o = bd["lineage_arrays"][f"values_{i:03d}"], bd["lineage_arrays"][f"origin_{i:03d}"]
        new = B.state_id(ed, s["parent"], B.record_digest(rec), v, o)
        smap[s["state_id"]] = new
        s["state_id"] = new
    shutil.rmtree(bdir)
    _write_bundle(bdir, bd)


def _write_bundle(bdir, bd):
    bdir.mkdir(parents=True)
    (bdir / "evidence_meta.json").write_text(B.canonical(bd["meta"]))
    np.savez(bdir / "evidence_arrays.npz", **bd["arrays"])
    (bdir / "constraints.json").write_text(B.canonical(bd["constraints"]))
    (bdir / "lineage.json").write_text(json.dumps(bd["states"], sort_keys=True, separators=(",", ":"), allow_nan=False))
    np.savez(bdir / "lineage_arrays.npz", **bd["lineage_arrays"])


def adversarial_mutations(r1: Path):
    """name -> function(copy_dir) applying the declared mutation to a byte copy of build r2."""
    def rewrite_json(p, fn):
        d = json.loads(p.read_text())
        fn(d)
        p.write_text(json.dumps(d, indent=1, sort_keys=True))

    def bundle_raw(cdir, pop, fn):
        bdir = cdir / "bundles" / pop
        bd = W.load_bundle(bdir)
        fn(bd)
        shutil.rmtree(bdir)
        _write_bundle(bdir, bd)

    def swap_rows(bd):
        for k, v in bd["arrays"].items():
            v = np.array(v)
            v[[10, 11]] = v[[11, 10]]
            bd["arrays"][k] = v

    def set_arr(bd, key, idx, add):
        v = np.array(bd["arrays"][key])
        v[idx] = v[idx] + add
        bd["arrays"][key] = v

    def reparent(bd):
        s = bd["states"]
        root = s[0]["state_id"]
        s[2]["parent"] = root
        s[2]["record"]["base_state_id"] = root

    def rebind(bd):
        rec = bd["states"][1]["record"]
        rec["constraints_digest"] = B.constraints_digest(bd["constraints"]["RS1_arm"])
        rec["constraints_id"] = "RS1_arm"

    def n8a(cdir):
        root = json.loads((cdir / "build_report.json").read_text())["build_root"]
        bd = W.load_bundle(cdir / "bundles" / "real")
        src = bd["meta"]["sources"]["w102_observations_predictions"]["path"]
        shutil.copyfile(src, cdir / "existing_path" / "observations_predictions.npz")
        _rechain(cdir / "bundles" / "real", lambda b: b["meta"]["sources"]["w102_observations_predictions"].__setitem__(
            "path", f"{root}/existing_path/observations_predictions.npz"))

    def n1b(cdir):
        with np.load(cdir / "synthetic_source.npz") as z:
            d = {k: np.array(z[k]) for k in z.files}
        d["score"][2, 2] += 0.01
        np.savez(cdir / "synthetic_source.npz", **d)

    def n7a(d):
        d["frames"][0]["points"][sorted(d["frames"][0]["points"])[0]]["confidence"] += 0.01

    def n7b(d):
        t = d["tracks"][sorted(d["tracks"])[0]] if isinstance(d["tracks"], dict) else d["tracks"][0]
        samples = t["samples"] if isinstance(t, dict) and "samples" in t else t
        s0 = samples[0] if isinstance(samples, list) else samples[sorted(samples)[0]]
        s0["rotation"][0] += 0.01

    other = str(Path(r1).resolve())
    return {
        "N1_same_filename_different_pose_bytes": lambda c: rewrite_json(
            c / "existing_path/legacy_pose_reconstructed.json",
            lambda d: d["frames"][0]["landmarks"][sorted(d["frames"][0]["landmarks"])[0]].__setitem__(
                "x", d["frames"][0]["landmarks"][sorted(d["frames"][0]["landmarks"])[0]]["x"] + 1.0)),
        "N1b_same_filename_different_synthetic_source": n1b,
        "N2a_changed_source_digest": lambda c: _rechain(c / "bundles/real", lambda b: b["meta"]["sources"]["w102_observations_predictions"].__setitem__(
            "sha256", "0" * 64)),
        "N2b_changed_frame_digest": lambda c: _rechain(c / "bundles/real", lambda b: b["meta"]["frame_digests"].__setitem__(17, "0" * 64)),
        "N3_changed_identity_bearing_source_path": lambda c: _rechain(c / "bundles/real", lambda b: b["meta"]["sources"]["repo_production_demo_clip"].__setitem__(
            "path", "C:/Users/dna10/animcv-w102/examples/production_demo/source.mp4")),
        "N4a_stale_content_hash": lambda c: bundle_raw(c, "real", lambda b: set_arr(b, "obs2d_score_raw", (5, 3), 0.01)),
        "N4b_mismatched_recorded_digest": lambda c: bundle_raw(c, "real", lambda b: b["meta"].__setitem__("evidence_digest", "f" * 64)),
        "N5a_modified_timestamp": lambda c: _rechain(c / "bundles/real", lambda b: set_arr(b, "timestamp", 50, 0.02)),
        "N5b_modified_frame_order": lambda c: _rechain(c / "bundles/real", swap_rows),
        "N5c_modified_joint_name": lambda c: _rechain(c / "bundles/real", lambda b: b["meta"]["joint_names"].__setitem__(13, "left_hand")),
        "N6a_modified_edit_parent": lambda c: _rechain(c / "bundles/real", reparent),
        "N6b_modified_permission_digest": lambda c: _rechain(c / "bundles/real", rebind),
        "N7a_motion_graph_content": lambda c: rewrite_json(c / "existing_path/legacy_motion.json", n7a),
        "N7b_animation_clip_content": lambda c: rewrite_json(c / "existing_path/legacy_animation.json", n7b),
        "N8a_generated_looking_source_path": n8a,
        "N8b_pose_source_outside_own_build": lambda c: rewrite_json(c / "existing_path/legacy_motion.json", lambda d: d["source_metadata"].__setitem__(
            "pose_source", str(Path(other) / "existing_path" / "legacy_pose_reconstructed.json"))),
        "N9_synthetic_substantive_change": lambda c: _rechain(c / "bundles/synthetic", lambda b: set_arr(b, "obs2d_score_raw", (2, B.JI["left_knee"]), 0.01)),
        "N10_report_field_under_synthetic_prefix": lambda c: rewrite_json(c / "build_report.json", lambda d: d["synthetic"]["controls_summary"].__setitem__(
            "passed", d["synthetic"]["controls_summary"]["passed"] - 1)),
    }


def cmd_repeat(a):
    out = new_dir(a.out_dir)
    r1, r2 = Path(a.first), Path(a.second)
    rep = {"schema": "w115_repeat_v1", "positive": {}, "adversarial": {}}
    rep["positive"]["P1_r1_vs_r2"] = compare_builds(r1, r2)
    rep["positive"]["P2_self"] = compare_builds(r1, r1)
    rep["positive"]["P3_r2_vs_r1"] = compare_builds(r2, r1)
    adv = new_dir(out / "adversarial")
    p0 = adv / "P0_unmutated_copy"
    shutil.copytree(r2, p0)
    rep["positive"]["P0_unmutated_copy"] = compare_builds(r1, p0)
    for name, fn in adversarial_mutations(r1).items():
        cdir = adv / name
        shutil.copytree(r2, cdir)
        try:
            fn(cdir)
            res = compare_builds(r1, cdir)
            mutated = [f for f in _files(cdir) if not (r2 / f).exists() or sha(cdir / f) != sha(r2 / f)]
            rep["adversarial"][name] = {"what": ADVERSARIAL[name], "rejected": not res["equivalent"], "unexplained": res["unexplained"],
                                        "mutated_files": mutated, "checks": res["checks"]}
        except Exception as e:  # a mutation that cannot be applied is a failed control, never a pass
            rep["adversarial"][name] = {"what": ADVERSARIAL[name], "rejected": None, "error": repr(e)}
    bundle_mut = [k for k in ADVERSARIAL if k.startswith(("N2", "N3", "N5", "N6", "N8a", "N9"))]
    for k in bundle_mut:
        x = rep["adversarial"].get(k, {})
        pop = "synthetic" if k.startswith("N9") else "real"
        chk = x.get("checks", {}).get(f"bundle_{pop}", {})
        x["rejected_by_semantic_chain"] = chk.get("normalized_chain_equal") is False
        x["integrity_after_forgery"] = [i["ok"] for i in chk.get("integrity", [])]
    rep["positive_all_equivalent"] = all(v["equivalent"] for v in rep["positive"].values())
    rep["adversarial_all_rejected"] = all(v.get("rejected") is True for v in rep["adversarial"].values())
    rep["adversarial_false_accepts"] = sorted(k for k, v in rep["adversarial"].items() if v.get("rejected") is not True)
    rep["rechained_rejected_semantically"] = all(rep["adversarial"][k].get("rejected_by_semantic_chain") for k in bundle_mut)
    p1 = rep["positive"]["P1_r1_vs_r2"]
    rep["r1_r2_accounting"] = {"raw_identical_files": len(p1["raw_identical_files"]), "raw_differing_files": p1["raw_differing_files"],
                               "declared_differences": p1["declared_differences"], "unexplained": p1["unexplained"]}
    b1, b2 = json.loads((r1 / "build_report.json").read_text()), json.loads((r2 / "build_report.json").read_text())
    rep["in_process_repeat"] = [b1["in_process_repeat"], b2["in_process_repeat"]]
    rep["semantic_equal_categories"] = {
        "source_evidence_real_benchmark": b1["digests"]["real"] == b2["digests"]["real"] and b1["digests"]["benchmark"] == b2["digests"]["benchmark"],
        "synthetic_normalized": p1["checks"].get("bundle_synthetic", {}).get("normalized_chain_equal"),
        "observations_hypotheses": all(p1["checks"].get(f"bundle_{p}", {}).get("normalized_chain_equal") for p in ("real", "benchmark")),
        "edit_lineage": b1["digests"]["real_programme_states"] == b2["digests"]["real_programme_states"]
        and b1["digests"]["benchmark_programme_states"] == b2["digests"]["benchmark_programme_states"],
        "rig_outputs": b1["rig_real"] == b2["rig_real"] and b1["rig_mixed"] == b2["rig_mixed"]
        and b1["digests"]["rig_case_file"] == b2["digests"]["rig_case_file"]}
    # ---- W114 V6 instrument diagnosis: the frozen W114 compare on copies of W114's own historical build reports
    diag = new_dir(out / "w114_compare_diagnosis")
    w1, w2 = W114 / "results/build_r1", W114 / "results/build_r2"
    runs = {}
    for name, fn in {"unmodified": None,
                     "synthetic_field_mutated": lambda d: d["synthetic"]["controls_summary"].__setitem__("passed", d["synthetic"]["controls_summary"]["passed"] - 1),
                     "synthetic_score_accounting_mutated": lambda d: d["synthetic"]["scores"].__setitem__("above_1", d["synthetic"]["scores"]["above_1"] + 1)}.items():
        dd = new_dir(diag / name)
        a1, a2 = new_dir(dd / "first"), new_dir(dd / "second")
        shutil.copyfile(w1 / "build_report.json", a1 / "build_report.json")
        d2 = json.loads((w2 / "build_report.json").read_text())
        if fn:
            fn(d2)
        (a2 / "build_report.json").write_text(json.dumps(d2))
        C.cmd_compare(Namespace(first=str(a1), second=str(a2), out_dir=str(dd / "out")))
        w = json.loads((dd / "out" / "repeat_report.json").read_text())
        runs[name] = {"w114_repeatable": w["repeatable"], "w114_unexplained": w["unexplained"],
                      "w114_counted_as_synthetic_path_induced": len(w["synthetic_source_path_induced"])}
    rep["w114_compare_diagnosis"] = runs
    dump(out / "repeat_report.json", rep)
    print(json.dumps({"positive_all_equivalent": rep["positive_all_equivalent"], "adversarial_all_rejected": rep["adversarial_all_rejected"],
                      "false_accepts": rep["adversarial_false_accepts"], "unexplained_r1_r2": p1["unexplained"][:5]}))
    return 0


# ------------------------------------------------------------------------------------------ blender compare ---

def cmd_blender_compare(a):
    decl = load_declaration(a)
    out = new_dir(a.out_dir)
    rq = decl["blender_rest_rotation_wxyz"]
    bdir = Path(a.build_dir)
    build = json.loads((bdir / "build_report.json").read_text())
    cases = json.loads((bdir / "w115_rig_cases.json").read_text())["cases"]
    lock_spec = json.loads((bdir / "lock_spec.json").read_text())
    audit = json.loads(Path(a.audit).read_text())["files"]
    posed = json.loads(Path(a.posed).read_text())["files"]
    w105 = json.loads(Path(decl["inputs"]["w105_render_clip_audit"]["path"]).read_text())
    ref_fixed = w105["files"]["real_clip/real_clip.blend"]["actions"]["W91 real_clip FIXED"]["value_digest"]
    deps = W.world_dependencies(decl["inputs"]["w105_rig_rest"]["path"])
    res = {"schema": "w115_blender_compare_v1", "w105_fixed_value_digest": ref_fixed, "cases": {}, "world_dependencies": deps}
    world = {}
    fixed_world = {}
    for name, case in cases.items():
        frames = case["frames"]
        keys = [fr["frame_index"] - frames[0]["frame_index"] + 1 for fr in frames]
        acts = audit[f"{name}/{name}.blend"]["actions"]
        pz = posed[f"{name}/{name}.blend"]["actions"]
        rec = {"scene_fps": audit[f"{name}/{name}.blend"]["scene_fps"]}
        stored, aud, mism, keyset_ok = {}, {}, {}, True
        for slot in ("FIXED", "RUNTIME"):
            act = acts[f"W91 {name} {slot}"]
            pred = W.stored_case_values(frames, slot, rq)
            pa, aa = W.stored_arrays(pred, keys), W.audit_arrays(act, keys)
            mism[slot] = {c: int(W._neq(pa[c], aa[c]).sum()) for c in W.CH}
            pred_keys = sorted({k for c in W.CH for k in pred[c]})
            keyset_ok &= [int(round(k)) for k in act["keyed_frames"]] == pred_keys
            stored[slot], aud[slot] = pa, aa
            if slot == "FIXED":
                rec["fixed_value_digest"] = act["value_digest"]
        rec["storage_mismatched_channel_frames"] = mism
        rec["storage_exact"] = all(v == 0 for d in mism.values() for v in d.values()) and keyset_ok
        rec["keysets_equal_prediction"] = keyset_ok
        if not name.startswith("MX_"):
            rec["fixed_equals_w105_fixed"] = rec["fixed_value_digest"] == ref_fixed
        l3 = {c: W._neq(aud["FIXED"][c], aud["RUNTIME"][c]) for c in W.CH}
        l2 = {c: W._neq(stored["FIXED"][c], stored["RUNTIME"][c]) for c in W.CH}
        rec["action_changed_channel_frames"] = {c: int(l3[c].sum()) for c in W.CH}
        rec["predicted_stored_changed_channel_frames"] = {c: int(l2[c].sum()) for c in W.CH}
        rec["action_change_sets_equal_predicted"] = all(np.array_equal(l3[c], l2[c]) for c in W.CH)
        # world positions + evaluated rotations
        pos, rot_ok = {}, True
        for slot in ("FIXED", "RUNTIME"):
            fr_by = {f["frame"]: f for f in pz[f"W91 {name} {slot}"]["frames"]}
            pos[slot] = [fr_by.get(k, {}).get("world_head") for k in keys]
            for i, k in enumerate(keys):
                f = fr_by.get(k)
                if f is None:
                    continue
                for ch, q in f["evaluated_rotation_wxyz"].items():
                    if ch in aud[slot] and not np.isnan(aud[slot][ch][i]).any() and not np.array_equal(np.asarray(q), aud[slot][ch][i]):
                        rot_ok = False
        rec["evaluated_rotations_equal_audit"] = rot_ok
        struct_bad = struct_n = 0
        for i in range(len(keys)):
            if pos["FIXED"][i] is None or pos["RUNTIME"][i] is None:
                continue
            for j, p in W.WORLD_POINT.items():
                if not any(l3[c][i] for c in deps[j]):
                    struct_n += 1
                    struct_bad += int(pos["FIXED"][i][p] != pos["RUNTIME"][i][p])
        rec["world_structural"] = {"points_with_unchanged_dependencies": struct_n, "of_which_moved": struct_bad}
        group = "real" if not name.startswith("MX_") else name.split("_M")[0]
        fixed_world.setdefault(group, set()).add(json.dumps(pos["FIXED"]))
        world[name] = (pos, stored, aud, keys)
        res["cases"][name] = rec
    res["fixed_world_identical_within_groups"] = {g: len(v) == 1 for g, v in fixed_world.items()}
    # ---- local / world lock evaluation of representative edits
    locks = {}
    for mname, m in REPRESENTATIVE.items():
        if m["case"] is None:
            locks[mname] = {"verdicts": "UNKNOWABLE", "reason": "operation refused before any state exists (IDENTITY_UNVERIFIED); per-frame "
                                                                "subject identity is UNKNOWN, so neither lock can be evaluated"}
            continue
        if m["case"].startswith("MX_"):
            for wname in C.MIXED_WINDOWS:
                key = m["case"].format(w=wname)
                pos, stored, aud, keys = world[key]
                n = len(keys)
                nonperm = np.ones((n, B.J), bool)
                nonperm[np.arange(0, n, 2), B.JI["left_wrist"]] = False
                locks[f"{mname}:{wname}"] = _lock_eval(pos, aud, nonperm, None, m["pred"])
            continue
        spec = lock_spec[mname]
        pos, stored, aud, keys = world[spec["case"]]
        locks[mname] = _lock_eval(pos, aud, np.asarray(spec["nonpermitted"], bool), np.asarray(spec["preserved"], bool), m["pred"])
        locks[mname]["constraints"] = spec["constraints"]
        locks[mname]["case"] = spec["case"]
    res["locks"] = locks
    res["all_storage_exact"] = all(c["storage_exact"] for c in res["cases"].values())
    res["all_action_change_sets_equal_predicted"] = all(c["action_change_sets_equal_predicted"] for c in res["cases"].values())
    res["all_fixed_reproduce_w105"] = all(c.get("fixed_equals_w105_fixed", True) for c in res["cases"].values())
    res["all_rotations_evaluated_equal_audit"] = all(c["evaluated_rotations_equal_audit"] for c in res["cases"].values())
    res["world_structural_ok"] = all(c["world_structural"]["of_which_moved"] == 0 for c in res["cases"].values())
    res["examples"] = _examples(world, cases, build)
    dump(out / "blender_compare.json", res)
    print(json.dumps({k: res[k] for k in ("all_storage_exact", "all_action_change_sets_equal_predicted", "all_fixed_reproduce_w105",
                                          "all_rotations_evaluated_equal_audit", "world_structural_ok")}),
          json.dumps({k: {kk: vv.get("verdict") for kk, vv in v.items() if isinstance(vv, dict) and "verdict" in vv}
                      for k, v in locks.items() if isinstance(v, dict)}))
    return 0


def _lock_eval(pos, aud, nonperm, preserved, pred):
    rows = [i for i in range(len(pos["FIXED"])) if pos["FIXED"][i] is not None and pos["RUNTIME"][i] is not None]
    unknown_rows = len(pos["FIXED"]) - len(rows)
    rec = {"rows_evaluated": len(rows), "rows_without_keys_unknown": unknown_rows,
           "local_nonpermitted": W.local_lock(aud["FIXED"], aud["RUNTIME"], nonperm, rows),
           "world_nonpermitted": W.world_lock(pos["FIXED"], pos["RUNTIME"], nonperm, rows)}
    if preserved is not None:
        rec["local_preserved"] = W.local_lock(aud["FIXED"], aud["RUNTIME"], preserved, rows)
        rec["world_preserved"] = W.world_lock(pos["FIXED"], pos["RUNTIME"], preserved, rows)
    rec["predicted"] = pred
    if isinstance(pred, dict):
        rec["prediction_match"] = {k: rec[k]["verdict"] == v for k, v in pred.items() if k in rec}
    loc = rec["local_nonpermitted"]["verdict"]
    wv = [rec[k]["verdict"] for k in ("world_nonpermitted", "world_preserved") if k in rec]
    rec["local_satisfied_world_violated"] = loc == "SATISFIED" and "VIOLATED" in wv
    rec["fully_user_preserving_under_both"] = loc in ("SATISFIED",) and all(v in ("SATISFIED", "VACUOUS") for v in wv) \
        and rec.get("local_preserved", {"verdict": "VACUOUS"})["verdict"] in ("SATISFIED", "VACUOUS")
    return rec


def _examples(world, cases, build):
    """Reporting only: concrete frames with identities, stored values and world displacements."""
    out = {}

    def frame_rec(case, i, chans, joints):
        pos, stored, aud, keys = world[case]
        fr = cases[case]["frames"][i]
        if pos["FIXED"][i] is None or pos["RUNTIME"][i] is None:
            return {"case": case, "frame_index": fr["frame_index"], "status": "no keys at this frame (rig emitted no slot)"}
        return {"case": case, "frame_index": fr["frame_index"], "action_key": keys[i], "source_time_s": fr.get("time_s"),
                "action_time_s_at_30fps": (keys[i] - 1) / 30.0,
                "channels": {c: {"FIXED_wxyz": aud["FIXED"][c][i].tolist(), "RUNTIME_wxyz": aud["RUNTIME"][c][i].tolist(),
                                 "stored_changed": bool(W._neq(aud["FIXED"][c][i:i + 1], aud["RUNTIME"][c][i:i + 1])[0])} for c in chans},
                "world_m": {j: {"FIXED": pos["FIXED"][i][W.WORLD_POINT[j]], "RUNTIME": pos["RUNTIME"][i][W.WORLD_POINT[j]],
                                "displacement_m": float(np.linalg.norm(np.subtract(pos["FIXED"][i][W.WORLD_POINT[j]], pos["RUNTIME"][i][W.WORLD_POINT[j]])))}
                            for j in joints}}
    c1 = build["rig_real"]["C1_right_shoulder_along_trunk_axis"]
    rows = c1.get("changed_rows_l2", {}).get("lowerarm_l") or c1.get("changed_rows_l1", {}).get("lowerarm_l") or [0]
    out["C1_float32"] = frame_rec("C1_right_shoulder_along_trunk_axis", rows[0], ["spine_01", "upperarm_l", "lowerarm_l", "__object__"],
                                  ["left_wrist", "left_elbow", "right_shoulder", "thorax"])
    out["C1_float64_diagnostic"] = frame_rec("C1_float64_exact_diagnostic", rows[0], ["spine_01", "upperarm_l", "lowerarm_l"],
                                             ["left_wrist", "left_elbow"])
    out["RS1_RS1_arm_frame_113"] = frame_rec("E_RS1_arm_accepted", 113, ["upperarm_l", "lowerarm_l", "spine_01"],
                                             ["left_wrist", "left_elbow", "left_shoulder", "right_wrist"])
    out["D2_LW_frame_110"] = frame_rec("D2_left_shoulder", 110, ["spine_01", "upperarm_l", "lowerarm_l"],
                                       ["right_wrist", "right_elbow", "head", "neck", "left_wrist"])
    for wname in C.MIXED_WINDOWS:
        out[f"mixed_MA1_{wname}_pos0"] = frame_rec(f"MX_{wname}_MA1_s0_left_wrist", 0, ["upperarm_l", "lowerarm_l"], ["left_wrist", "left_elbow"])
        out[f"mixed_MB2_{wname}_pos0"] = frame_rec(f"MX_{wname}_MB2_j1x_left_wrist", 0, ["upperarm_l", "lowerarm_l"], ["left_wrist", "left_elbow"])
    out["label"] = "REPORTING ONLY: MOCK source edits on frozen inputs; renders and values are diagnostic, not animation quality"
    return out


# ------------------------------------------------------------------------------------------------ decide ---

def cmd_decide(a):
    load_declaration(a)
    out = new_dir(a.out_dir)
    pre = json.loads(Path(a.preflight).read_text())
    b = json.loads((Path(a.build_dir) / "build_report.json").read_text())
    rp = json.loads(Path(a.repeat).read_text())
    bl = json.loads(Path(a.blender).read_text())
    r, bm, s = b["real"], b["benchmark"], b["synthetic"]
    rprog, bprog, sprog = r["programme"], bm["programme"], s["programme"]
    ok_ctl = lambda summ: not summ["failed"]  # noqa: E731
    c1 = b["rig_real"]["C1_right_shoulder_along_trunk_axis"]
    c164 = b["rig_real"]["C1_float64_exact_diagnostic"]
    real_conds = {k: v for k, v in b["rig_real"].items() if k != "C1_float64_exact_diagnostic"}
    lock_recs = [v for v in bl["locks"].values() if isinstance(v, dict) and "local_nonpermitted" in v]
    V = {
        "V1_inputs_authenticated": bool(pre["pass"]),
        "V2_history_intact": bool(all(h["marker"] and h["blob_at_start"] == h["blob_now"] for h in pre["historical_worklogs"].values())
                                  and pre["w114_frozen_case"] == "D" and pre["inputs"]["w114_declaration"]["ok"] and pre["inputs"]["w114_decision"]["ok"]),
        "V3_rig_reproduction": b["rig_reproduction_real"]["byte_equal_frames"] == b["rig_reproduction_real"]["frames"] == 199
                               and all(w["reproduction"]["byte_equal_frames"] == w["reproduction"]["frames_with_fixed_slot"] > 0 for w in b["rig_mixed"].values())
                               and bl["all_fixed_reproduce_w105"],
        "V4_storage_exact": bool(bl["all_storage_exact"]),
        "V5_repeatability": bool(rp["positive_all_equivalent"] and all(all(v.values()) for v in rp["in_process_repeat"])),
        "V6_adversarial": bool(rp["adversarial_all_rejected"] and rp["rechained_rejected_semantically"]),
        "V7_synthetic_controls": ok_ctl(s["controls_summary"]) and sprog.get("all_ok", False) and sprog.get("rig_steps_consistent", False)
                                 and s["cross_frame_verified_positive"]["ok"],
        "V8_numeric_attribution": c164["stored_change_channel_frames"] == 0 and c164["base64_frames_equal_float32_frames"]
                                  and c1["c1_stored_displacement"]["rows_l1_changed_with_zero_perpendicular"] == 0,
        "V9_world_instrument": bool(bl["all_rotations_evaluated_equal_audit"] and all(bl["fixed_world_identical_within_groups"].values())
                                    and bl["world_structural_ok"]),
        "V10_locks_assessable": all(rec["local_nonpermitted"]["verdict"] != "VACUOUS" and rec["world_nonpermitted"]["verdict"] != "VACUOUS"
                                    for rec in lock_recs) and bl["locks"]["M10_E5_cross_frame_copy"]["verdicts"] == "UNKNOWABLE",
    }
    bd = [k for k, v in real_conds.items() if v["kind"] in "BD"]
    G = {
        "G1_evidence_immutable": bool(rprog.get("evidence_unchanged_end") and rprog.get("source_files_unchanged") and rprog.get("inplace_write_refused")
                                      and bprog.get("evidence_unchanged_end", False) and r.get("evidence_unchanged_after_rig")
                                      and all(st.get("evidence_unchanged", True) for st in rprog["log"] + bprog.get("log", []))),
        "G2_edit_locality": bool(rprog["E1_locality"]["changed_equals_requested"] and rprog["E1_locality"]["unintended"] == 0
                                 and rprog["E2_locality"]["changed_equals_requested"] and rprog["E1_originals_equal_parent"]
                                 and bprog.get("EB1_locality", {}).get("changed_equals_requested", False)),
        "G3_reversal": bool(rprog["RV2_equals_post_E1"] and rprog["RV1_equals_hypothesis"] and rprog.get("R11_channels_bit_equal_hypothesis") is True
                            and rprog.get("final_equals_hypothesis") and rprog["lineage_audit"]["ids_recomputed_equal"]
                            and bprog.get("B5_equals_hypothesis", False) and bprog.get("lineage_audit", {}).get("ids_recomputed_equal", False)),
        "G4_stale_rejection": all(st["ok"] for st in rprog["log"] + bprog.get("log", []) if st["step"] in ("E3", "E4", "B3")),
        "G5_permission": bool(r["permitted_cells"] > 0 and r["RS2"]["observed"] == "LOCK_CONFLICT"
                              and all(r["controls"][k]["ok"] for k in ("unauthorized_edit", "edit_on_preserved_cell", "preserve_permit_conflict"))
                              and rprog.get("all_ok", False) and bprog.get("all_ok", False)),
        "G6_score_semantics": bool(r["source_exact"]["observed"] == "PASS" and r["scores"]["above_1"] > 0
                                   and r["controls"]["manufactured_3d_confidence"]["ok"] and r["controls"]["score_clipped_to_unit_interval"]["ok"]
                                   and bm["controls"]["unknown_to_zero_coercion"]["ok"] is not False),
        "G7_controls_real_bench": ok_ctl(r["controls_summary"]) and ok_ctl(bm["controls_summary"]),
        "G8_storage_aware_influence": bool(
            all(v["unperturbed_rows_l1_bit_changes"] == 0 for v in real_conds.values())
            and all(c["unperturbed_rows_l1_bit_changes"] == 0 for w in b["rig_mixed"].values() for c in w["conditions"].values())
            and all(sum(x["l1_bit_changed"] for x in real_conds[k]["perturbed"].values()) == 0 for k in ("A1_right_wrist", "A2_head"))
            and all(w["conditions"][k]["all_rows_l1_bit_changes"] == 0 for w in b["rig_mixed"].values() for k in ("MA1_s0_left_wrist", "MA2_s0_left_shoulder"))
            and all(real_conds[k]["stored_rotation_channels"] == REAL_EXPECT[k] for k in bd)
            and all(c["stored_rotation_channels"] == c["expected_stored_channels"] for w in b["rig_mixed"].values()
                    for c in w["conditions"].values() if c["kind"] in "BD")
            and all(c1["perturbed"][ch]["l1_bit_changed"] == 0 for ch in C1_STRUCTURALLY_INDEPENDENT)
            and all(v["authority_storage_aware"]["consistent"] and v["authority_storage_aware"]["atomic"] for v in real_conds.values())
            and rprog.get("rig_steps_consistent", False)
            and all(r["controls"][k].get("ok") is not False for k in C.RIG_CONTROLS)
            and bl["all_action_change_sets_equal_predicted"]),
        "G9_authority_domain": bool(
            [x["observed"] for x in rprog["rig_log"]] == ["OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN", "ACCEPT", "NO_RIG_INFLUENCE", "ACCEPT"]
            and {k: v for k, v in rprog.get("R10_association", {}).get("changed_channel_frames", {}).items()} == {"upperarm_l": 38, "lowerarm_l": 38}
            and rprog.get("R11_channels_bit_equal_hypothesis") is True and rprog.get("R13_association_changed") == {}
            and rprog.get("final_equals_hypothesis") is True),
    }
    T = {
        "T1_identity": b["time_identity"]["identity"]["tracked_subject"]["frames_unverified"] > 0
                       and r["controls"]["subject_identity_ambiguity"]["observed"] == "IDENTITY_UNVERIFIED",
        "T2_time": b["time_identity"]["mismatched_keys"] > 0,
        "T3_local_world_divergence": any(rec["local_satisfied_world_violated"] for rec in lock_recs),
        "T4_joint_vs_bone_authority": any(set(R.DEPICTS[c]) != {"left_wrist"} for c in b["rig_real"]["B1_left_wrist"]["stored_rotation_channels"]),
    }
    case = "D" if not all(V.values()) else "C" if not all(G.values()) else "B" if any(T.values()) else "A"
    rep = {"schema": "w115_decision_v1", "declaration_sha256": a.declaration_sha, "validity": V, "core": G, "boundary_triggers": T,
           "case": case, "rule": DECISION["case_rule"],
           "lock_prediction_mismatches": {k: [kk for kk, ok in v.get("prediction_match", {}).items() if not ok] for k, v in bl["locks"].items()
                                          if isinstance(v, dict) and v.get("prediction_match") and not all(v["prediction_match"].values())}}
    dump(out / "decision.json", rep)
    print(json.dumps({"case": case, "V": V, "G": G, "T": T}))
    return 0


def cmd_manifest(a):
    base = Path(a.base)
    every = [p for p in sorted(base.rglob("*")) if p.is_file() and p.name != "evidence_manifest.json"]
    pngs = [p for p in every if p.suffix == ".png"]
    files = {p.relative_to(base).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in every if p.suffix != ".png"}
    h = hashlib.sha256()
    for p in pngs:
        h.update(p.relative_to(base).as_posix().encode())
        h.update(sha(p).encode())
    rep = {"schema": "w115_evidence_manifest_v1", "starting_head": STARTING_HEAD, "declaration_sha256": a.declaration_sha,
           "files": files, "render_pngs": {"count": len(pngs), "combined_digest": h.hexdigest()},
           "total_bytes": sum(p.stat().st_size for p in every)}
    dump(base / "evidence_manifest.json", rep)
    print(sha(base / "evidence_manifest.json"), len(files), len(pngs))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--out-dir", required=True)
    p = sub.add_parser("declare"); p.add_argument("--out-dir", required=True); p.add_argument("--preflight", required=True)
    for name in ("build", "decide", "blender-compare", "manifest"):
        p = sub.add_parser(name)
        p.add_argument("--declaration", required=name != "manifest"); p.add_argument("--declaration-sha", required=True)
        p.add_argument("--out-dir", required=name != "manifest")
    sub.choices["build"].add_argument("--tag", required=True)
    for k in ("build-dir", "audit", "posed"):
        sub.choices["blender-compare"].add_argument(f"--{k}", required=True)
    for k in ("preflight", "build-dir", "repeat", "blender"):
        sub.choices["decide"].add_argument(f"--{k}", required=True)
    sub.choices["manifest"].add_argument("--base", required=True)
    p = sub.add_parser("repeat")
    for k in ("first", "second", "out-dir"):
        p.add_argument(f"--{k}", required=True)
    a = ap.parse_args(argv)
    return {"preflight": cmd_preflight, "declare": cmd_declare, "build": cmd_build, "repeat": cmd_repeat,
            "blender-compare": cmd_blender_compare, "decide": cmd_decide, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
