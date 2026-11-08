# BATCH 124 — Opt-in Real-Video to Editable Blender Action with Reversible World Pin

DIRECTION

PRODUCTION-ORIENTED ITERATIVE DEVELOPMENT. Build one demonstrable, bounded, opt-in animation-authoring vertical slice from authentic video to an editable Blender Action and a user-authored WORLD_POSITION_PIN edit with truthful footprint and exact undo. This is an ENGINEERING INTEGRATION batch, not another pose-score/solver-threshold diagnostic. Preserve the default legacy CLI/GUI and historical experiments. A successful vertical slice is not a general-purpose Generator, universal retargeter, production-ready FBX interchange or a quality guarantee.

PURPOSE

The current normal CLI/GUI still follows legacy MotionGraph -> RetargetSolver -> AnimationClip. Existing FramePose, Semantics, BaseRig retarget and Blender code are separate; W116–W122 demonstrated isolated World Pin, edit-state and real-rig experiments but did not deliver a coherent opt-in user workflow. W123's learned assessor showed in-domain triage value (official CASE B) but failed missing-input coverage (0.737) and independent-domain coverage (0.421); it is NOT safe to use as automatic edit authority. The product needs a usable, inspectable Action rather than another reliability or guide sweep.

CENTRAL INTENT

Can a user, through ONE documented opt-in workflow on the existing known BaseRig, supply a real dynamic video and an optional explicitly authored Blender-scene-world left-wrist position pin, obtain a provenance-bound editable .blend Action, see precisely which frames and bones changed, and accept/reject/revert the edit without losing the original Action or silently violating hard constraints? What is actually usable at real-time playback, and what remains blocked at FBX continuous-time roundtrip?

CURRENT ASSESSMENT / HUMAN ASSESSMENT

- Official history is immutable: W113–W115 D, W116 B, W117 C, W118 D, W119 D, W120 B, W121 D, W122 B, W123 B. W123 has BATCH_NUMBER: 123 and AUTOMATION_BATCH_STATUS: COMPLETE at observed HEAD 1909613ff754a179bfe2393b612fdabdccc2d80a. Recheck remotely.
- W121/W122 S1 kettlebell, S2 juggling and S3 bent-over row are authentic licensed, GT-FREE clips, NOT new independent sources relative to those Worklogs. S1/S3 are preferable single-performer engineering replays; S2 has subject-identity ambiguity and must not be quietly treated as clean.
- The frozen W121 S0-only W105 FIXED-slot BaseRig adapter produced 30fps Actions and W116–W122 tested a left-wrist pin. It is an explicit PARTIAL rig: right arm may remain at rest, root translation/foot contact and general body quality are not established. Source steps may be artefacts (notably S1 f251).
- W107–W112 found FBX interpolation/subframe/importer failures. A .blend file, a successfully exported .fbx, and a faithful reimport are DIFFERENT achievements.
- The owner approved Semantic Constraint System, WORLD_POSITION_PIN in Blender scene-world metres, separate immutable source and versioned editable state, atomic accept/reject/revert, and bounded automatic out-of-pin editing ONLY with truthful frame/Bone disclosure. The owner did NOT approve new constraint types, universal elbow limits, production numerical tolerances, a trained Generator, a schema replacement, or unbounded auto-edit.
- The owner subsequently approved production-oriented development: a narrow, safely reversible Production integration with explicit experimental gating is permitted; W123's batch-specific src/ prohibition is not permanent. Do not interpret this as permission to change the default pipeline or deploy W123 reliability as a trust policy.

HUMAN-READABLE SUMMARY

Direction: First opt-in real-video -> known-rig editable Action + one semantic edit, using existing frozen components.

Purpose: Close a real user capability and expose integration/quality defects at the final Blender/FBX boundary.

Central Intent: Does the integrated workflow preserve source/time/rig authority, apply a real user pin, disclose every actual edit and revert exactly, while leaving default behavior untouched?

DO: bounded opt-in command/entry point, actual source-to-rig execution, editable Action, user pin request, independent Blender evaluation, footprint/timeline and undo, real-time visual A/B, FBX export/readback characterization, focused regression, milestone-status documentation.

DO NOT: train/fine-tune FramePose, Reliability or Generator; implement Style/Completion, new constraint types, full UI, general arbitrary-rig mapping, default policy changes, broad MotionGraph/AnimationClip migration, solver threshold tuning, FBX patchwork repair, source resampling or fabricated missing pose. Agent has no next-Batch/Architecture decision authority.

Decision enabled: Is this bounded workflow actually useful as an EXPERIMENTAL authoring capability, which exact boundary fails, and can it remain opt-in without regressing existing product paths?

HISTORICAL BASELINE / PRE-FLIGHT

Repository kn99e39/AnimCV; sole authorized existing branch arch/single_frame_first. Never create, switch, delete, merge, rebase, reset or force-push branches. Fetch/compare current remote HEAD before any edit; stop on concurrent changes or an unresolved worker state. Read AGENTS.md, PRODUCT_VISION.md, Architecture_v3_FramePose.md, Architecture_v2.md, COORDINATE_CONVENTIONS.md, docs/00_마스터_플랜.md, docs/01_프레임워크_현황.md, docs/next_batch.md, docs/AUDIT_W116_W120_2026_11_06.md, W105, W107–W112, W116–W123 Worklogs as relevant. Preserve the W125 cumulative-audit checkpoint (W121–W125, before W126).

Inspect ACTUAL implementation, not only Worklogs: src/app/cli.py, src/framepose/model.py, contract.py, observations.py, src/motion/animation_semantics_v2_bridge.py, src/retarget/framepose_fk.py, framepose_two_bone_ik.py, framepose_object_owner.py, src/blender/executor.py, export.py, framepose_object_owner_adapter.py; scripts/w102_video_diagnostic.py, w105_rig_utility.py, w116_semantic_pin_base.py, w116_blender_stages.py, w118_transition_edit_base.py, w121_independent_transfer.py, w122_realtime_utility.py; relevant tests. Identify which boundaries are truly reusable versus research-specific and write an explicit dependency/authority map BEFORE implementing.

Freeze before outcomes: exact original clip SHA-256/license/source URI, model/checkpoint/config digests, W105 rig FBX/rest/calibration/mapping identity, source frame indices/timestamps/cadence, baseline Action values, supported bone/joint subset, request semantics, permitted edit DOF, experimental acceptance tolerances (reuse W116's declared 1 mm ONLY as an experimental diagnostic, not a new Production default), and real-playback/FBX readback checks. No post-outcome gate changes.

PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE:
- Existing src default CLI/GUI behavior, MotionGraph and AnimationClip schemas, all historical Worklogs/CASEs, original assets, model weights/checkpoints, source observations, frozen numerical and coordinate contracts.
- GT-free inference; source, observation, pose hypothesis, user hard pin, and edited Action remain distinguishable. Never treat 2D score or W123 q90 as authorization.
- Rig identity and Blender scene-world metres. Keep actual source timestamps, fps and explicit source-frame -> Action-frame mapping. Unknown/invalid is NOT zero, rest pose, interpolated truth or user intent.
- Original Action/source files, editable keyframe curves, and failure provenance.

CHANGE ONLY (bounded, reviewable):
- A small opt-in entry point in src/app/cli.py or an equivalent documented user-facing command, with minimal new src/app or src/blender integration modules as required; focused tests and a small launcher are allowed. Existing retarget/model/FBX implementations may be CALLED but not silently rewritten.
- New per-run, outside-repository artifacts: immutable input manifest, baseline .blend, candidate .blend, accepted .blend (if accepted), reversible edit-state/lineage, actual footprint/timeline and A/B video/evaluation reports. Do not overwrite existing user outputs.
- Docs for this opt-in EXPERIMENTAL capability, usage and limitations; W124 Worklog. If and only if a meaningful milestone is actually demonstrated, refresh living status in docs/00_마스터_플랜.md, docs/01_프레임워크_현황.md and docs/README.md with evidence-level labels; do not rewrite normative architecture or claim a Production default migration.
- The Agent may choose minimal implementation factoring, but must not make stable Production code silently depend on an undocumented W12x scratch workspace or hardcoded C:\\Users paths. Reuse frozen algorithms through an explicit experimental adapter or small testable extraction with parity checks. No duplicate new framework.

DO NOT TUNE:
- Frozen M1/RTMPose/S0 model, raw 2D validity/scores, crop, tracking, historical W105 adapter math, W116 pin semantics, W118–W122 guides/flex-floor/quality gates, root/timebase/coordinate conversions, default keyframe collapse, FBX exporter/importer policies.
- No new elbow-flex floor, smoothing, source correction, trust threshold, guessed subject ID, heuristic retry, hidden pin relaxation or fallback to GT/oracle inputs.

REQUIRED EVIDENCE — A. AUTHENTIC END-TO-END INPUT

Use an already authenticated W121/W122 original S1 and/or S3 (prefer both when available) as engineering regression, not new-source generalization. If not locally available, legitimately reacquire the identical licensed original and verify hash; otherwise DATA BLOCKED. No download bypass or redistribution. Re-run the frozen M1 + S0 perception on source frames when the environment permits; no GT. Reuse stored arrays only as an EXPLICIT offline-replay fallback, labelled NOT live-video E2E. Compare the same-source prediction and rig Action with the frozen W121 baseline where available. No dependence on W123 assessor for a successful run.

Require actual image/video identity, known source fps and presentation timestamp consistency. For this bounded slice accept the already validated 30fps cadence; other cadence, VFR, missing model, ambiguous mapping, missing target bone, untracked source identity or unsupported frame must cause a clear UNSUPPORTED/BLOCKED outcome rather than silently resampling, retiming or fabricating. Subject association may remain UNKNOWN, but disclose it and avoid verified-identity claims. Do not use the historical 50fps video as a 30fps real-time success.

REQUIRED EVIDENCE — B. USER-FACING ACTION

One documented command/workflow must create a new, separately named Blender .blend containing the imported authoritative BaseRig and an editable Action from the frozen source hypothesis. Do not present an arbitrary rig as supported. The output must preserve frame mapping and provenance in a machine-readable sidecar, state which body channels are actually driven (including the right-arm-rest limitation), and report unavailable frames/bones. Inspect Blender F-curves, interpolation, armature/object ownership, action duration, source-time mapping and pose readback at integer frames AND selected subframes. Do not silently use legacy mixed-unit MotionPoint.position_3d. The no-edit baseline must be reproducible and independently inspectable.

REQUIRED EVIDENCE — C. EXPLICIT SEMANTIC EDIT + UNDO

Support one optional user-authored WORLD_POSITION_PIN request for the VERIFIED left_wrist binding on the known rig, with declared frame interval, Blender-scene-world target in metres and explicit authorized DOF. Use the existing W116 contract/solver as EXPERIMENTAL; the existing 1mm gate is a frozen diagnostic check, NOT a universal accuracy promise. Do not automatically switch to W120 world guide or flex-floor based on context-specific positive evidence. Out-of-pin transition edits are ALLOWED by the owner only if bounded, disclosed and reversible; they are NOT mandatory. If attempted, the actual edit domain must be derived from Blender-evaluated values, not the planned domain. Hard Pin never relaxes to make a candidate pass.

Create separate baseline/candidate/accepted/reverted states, content-addressed lineage and explicit ACCEPT/REJECT. A rejected, infeasible, unsupported, stale or failed candidate must not mutate the accepted Action or original source. Exact revert must reproduce baseline values and Blender-evaluated pose. Show an understandable timeline/HTML/PNG or equivalent view of requested hard-pin frames, permitted temporal domain, ACTUAL changed frames/Bones, unchanged frames and rejected edits. Do not implement a full UI. Include a genuinely nonzero feasible pin edit and an infeasible or wrong-binding negative control; no-op pin alone is insufficient.

REQUIRED EVIDENCE — D. FINAL MEDIUM, REAL-TIME A/B, INTERCHANGE

For the real clip, generate a 30fps side-by-side video of ORIGINAL SOURCE, unedited Action, edited Action (if accepted) with frame/Bone footprint visible and truthful. Inspect at native playback speed, not only decoded stills. Evaluate source fidelity separately from user-pin compliance, transition continuity, elbow/straight-arm/flip costs, contact/root limitations and actual editability; report negatives. Do not claim improved quality solely because a wrist residual or source-relative jump decreased.

Attempt FBX export from the opt-in .blend through the existing exporter and a clean Blender reimport, preserving original Action for comparison. Record integer-frame and selected subframe world positions/rotations, F-curve interpolation, duration/timebase and mismatches. Known W107–W112 issues are expected potential FAILS: do not fix by changing importer flags, key insertion, global defaults or exporter policy within W124. If .blend works but FBX fails, label Blender Action usable/FBX NOT VERIFIED or FAIL separately. An exported file existing on disk is not a roundtrip PASS.

REQUIRED EVIDENCE — E. SAFETY / REGRESSION / OBSERVABILITY

Test synthetic fixture and actual BaseRig with: correct source/rig/pose digest; stale manifest; changed source byte; wrong fps/timestamp; unknown joint; ambiguous binding; invalid numeric target; wrong coordinate frame; out-of-reach pin; unauthorized bone; stale parent; rejected candidate leaves accepted unchanged; exact undo; truthful footprint; absent Blender; missing checkpoint; missing GT-free observation; no-op vs nonzero edit; correct quaternion/time mapping. Test default CLI/GUI entry points are unchanged, no hidden new model dependencies and no historical Worklog changes. Focused tests first, then relevant existing regression tests (CLI, Blender, FK/IK, W105/W116/W118, canonical pose, MotionGraph/export) and a bounded real Blender replay. Do not report a test PASS unless actually run; distinguish mock, unit, Blender, real-video and FBX tests.

SYNTHETIC + REAL REPLAY

- Synthetic: identity/no-op, feasible nonzero left-wrist pin, unreachable pin, stale source, invalid binding, hard pin conflict, timebase mismatch, reject/undo and footprint tamper. No GT leakage or fallback.
- Real: at least one authenticated S1 or S3 30fps clip, same frozen model/rig baseline, baseline Action, user nonzero pin edit, actual Blender evaluation and A/B native-rate playback. Prefer both S1/S3 for a second engineering replay; S2 is a deliberately disclosed identity-risk negative, not a clean success.
- Do not relabel W121/W122 clips as new independent evidence, and do not claim 3D GT accuracy.

STOP CONDITIONS

STOP and report precise BLOCKED/INVALID reason if essential licensed video, frozen checkpoint, authoritative rig, Blender or runtime dependencies are unavailable; if source/rig identity or cadence fails; if existing parity is lost; if a major Production schema/solver/quality policy decision is needed; if only unsafe source-fabrication or unauthorized hard-pin relaxation would complete the task; if scope expands to new constraint types/training/full UI/FBX patchwork; or if concurrent remote changes cannot be reconciled without prohibited branch operations. No fake PASS, fallback asset, silent default switch or post-hoc threshold sweep. An E2E output without a real clip is NOT a completed vertical slice.

FINAL REPORT

Create docs/124_WORKLOG_OPT_IN_VIDEO_TO_EDITABLE_ACTION_WORLD_PIN.md with:
1. exact HEAD, code changes, branch, frozen source/model/rig hashes, machine/environment and real versus mocked stages;
2. one user-reproducible command/example, output directory structure, dependency boundaries, default-path isolation;
3. provenance/timebase/binding/UNKNOWN and support accounting;
4. real clip -> Blender Action evidence, editable F-curves and independent pose readback;
5. user pin request, resolver result, hard-pin residual, authorized changed frames/Bones, visual timeline, exact undo;
6. native-rate source vs baseline vs edited visual and numerical comparison, failure/side-effect examples, actual user/agent review attribution;
7. separate .blend success, FBX export and reimport/subframe verdicts; explicit W107–W112 known failures;
8. synthetic/focused/real Blender/FBX/regression tests: RUN/PASS/FAIL/NOT RUN; missing artifacts and reasons;
9. safety, performance, maintenance and rollback risks; no generalization claims;
10. official engineering CASE and completion marker. Do NOT propose W125 or ask for next work.

Decision labels (do not conflate):
- CASE A: authentic live-video E2E, editable Action, nonzero hard pin, truthful footprint/undo, independent Blender validation AND faithful tested FBX time/interpolation roundtrip, with no blocking validity defect.
- CASE B: authentic live-video E2E plus usable editable .blend and safe pin/undo/footprint demonstrated, but FBX/animation fidelity/coverage is limited and explicitly reported.
- CASE C: valid controls but core real-video-to-usable-Action or safe pin authoring capability not demonstrated.
- CASE D: evidence invalidated by integrity/contract/gate contamination. DATA BLOCKED / NO CASE if essential authenticated assets/runtime are absent before evaluation.
Historical CASEs are never modified. A synthetic PASS or an agent-rated render alone cannot establish A.

COMPLETION CONDITION

The batch is COMPLETE only after a single documented opt-in user workflow is implemented or a precise STOP is recorded; declared controls, actual run status, user-visible edit footprint, Action/FBX readback and reversibility are honestly accounted for; all changed files/tests and output hashes are reported; and the Worklog ends with exactly:
AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 124

Claude is Evidence Producer only. It must NOT decide or issue the next Batch, recommend Architecture policy, or modify docs/next_batch.md.
