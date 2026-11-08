# BATCH 125 — Crash-Consistent Editable Blender Action Transactions and Recovery

DIRECTION

PRODUCTION-ORIENTED ITERATIVE DEVELOPMENT. Complete the safety and durability of the existing opt-in real-video -> editable Blender Action -> user WORLD_POSITION_PIN workflow. This is a BOUNDED ENGINEERING INTEGRATION batch, not a new model, a new constraint type, a broad editor rewrite, or another pose/solver diagnostic. Work only on the existing branch arch/single_frame_first; never create, switch, delete, merge, rebase, reset, or force-push branches. Do not issue a future Batch.

PURPOSE

W124 produced the first opt-in user workflow and officially completed CASE B. Agent-run S1/S3 live-video -> frozen FramePose -> W105 BaseRig .blend -> nonzero left-wrist Hard Pin -> truthful footprint -> normal-case accept/reject/revert was demonstrated. FBX round-trip FAILED and real animation remains partial/GT-free. The next urgent product risk is that an accepted or reverted Action must remain coherent when the CLI or Blender fails midway, not merely when a happy-path test passes.

Direct GitHub source review of W124 found a concrete transactional exposure: cmd_commit changes and saves the persistent head BEFORE copying/checking the accepted .blend, updating the blend registry, writing commit.json, and rendering the final timeline; cmd_revert changes and saves the head BEFORE building and independently reading back the reverted .blend. PersistentStore.save writes refs.json directly and appends log.jsonl separately. W116 Store.commit is atomic only as an in-memory head comparison. A process interruption or I/O/Blender failure can therefore leave the authoritative head, Action artifact, lineage log and user-visible status inconsistent. This is CODE-INSPECTED RISK, not an independently reproduced failure; W124's normal-case tests and official CASE B are not retroactively changed.

CENTRAL INTENT

Can the already approved opt-in WORLD_POSITION_PIN authoring workflow guarantee that ACCEPT, REJECT and REVERT publish one coherent, recoverable user-visible Action/state transition under injected failures and competing CLI processes, with no accepted head pointing at a missing/unverified .blend and no ghost accepted timeline, while preserving exact Hard Pin, immutable source, actual changed-frame/Bone footprint, native 30fps Action, and the existing default CLI/GUI behavior?

CURRENT ASSESSMENT / HUMAN ASSESSMENT

- Official W124 COMPLETE: docs/124_WORKLOG_OPT_IN_VIDEO_TO_EDITABLE_ACTION_WORLD_PIN.md, CASE B, observed reviewer HEAD 6d938ec57b5a7e392cb75c5afbc0c0aaecc8bc77. The final HEAD includes a concurrent automation-script merge; distinguish that history from W124's opt-in code. Never modify scripts/batch_automation.py or tests/test_batch_automation.py in this Batch and do not perform any merge.
- W124 Agent-run: S1/S3 source-backed editable .blend and left-wrist pin/undo; actual changed frames/bones disclosed; 35 focused + 226 selected regression tests. No Reviewer-local Blender rerun. S1 left-arm source mismatch, S3 uninformative front close-up, right arm at rest, no root translation/contact; W124 FBX round-trip FAIL (integer drift 3.90/6.77 mm, subframe drift 0.69/0.21 m, CONSTANT -> LINEAR, scene range 250). Do not reinterpret a successful transaction as better motion quality or FBX portability.
- User-approved policy: Semantic Constraint System, Blender scene-world WORLD_POSITION_PIN, immutable Source Evidence, versioned editable state, exact atomic accept/reject/revert, bounded disclosed edits. W124 uses zero out-of-pin budget; this Batch does not silently enable a transition solver. The 1 mm diagnostic is historical, not a permanent Production threshold.
- W123 reliability CASE B remains scratch, not runtime authority. All W113–W124 Worklogs/CASEs are immutable. The W116–W120 audit is complete; W125 COMPLETE will trigger the mandatory W121–W125 cumulative audit BEFORE W126.

HUMAN-READABLE SUMMARY

Direction: harden the existing experimental animation-authoring vertical slice for interrupted/competing operations.

Purpose: prevent loss, mispublication or misrepresentation of a user's accepted/reverted Blender Action.

Central Intent: after any injected failure or restart, the visible head must resolve to exactly one fully validated .blend and lineage state, or the previous head must remain authoritative with an explicit recoverable STOP; a rejected edit never moves the head.

DO: bounded transaction staging, durable publication/recovery, focused fault-injection and concurrency tests, actual BaseRig/real-video edit replay, timeline/undo verification, native-rate A/B, unchanged FBX limitation and default-path regressions.

DO NOT: tune FramePose, W105 retarget, W116 pin solver, elbow thresholds, W118–W122 temporal guides, animation interpolation, model weights, perception, FBX exporter/importer, schema of MotionGraph/AnimationClip, new constraint types, trained Generator, broad GUI or default command behavior.

Decision enabled: is the opt-in editing workflow safe enough to preserve user-authored animation across process failures, and what failure modes or platform limits still prevent relying on it?

HISTORICAL BASELINE / PREFLIGHT

Before changes, inspect remote HEAD, docs/next_batch.md, AGENTS.md, PRODUCT_VISION.md, Architecture_v3_FramePose.md, Architecture_v2.md, COORDINATE_CONVENTIONS.md, docs/AUDIT_W116_W120_2026_11_06.md, W121–W124 Worklogs and relevant W116/W118 scripts/tests. Inspect actual src/app/experimental_video_action.py (PersistentStore, cmd_commit, cmd_revert, load_run, blend_registry, timeline), src/blender/experimental_action_stage.py, src/app/cli.py, scripts/w116_semantic_pin_base.py Store, src/blender/executor.py and export.py, tests/test_experimental_video_action.py, W124 harness and the W124 usage doc. Record the current code paths, write order, failure windows, filesystem/OS assumptions and exact W124 normal-case control hashes BEFORE implementing.

Freeze one transaction/recovery design, named failure points, comparison population, platform scope and acceptance gates BEFORE observing real S1/S3 outcomes. Preserve W124's source/rig/model/checkpoint identity, action frame mapping, known BaseRig, frozen algorithms, original user request and immutable baseline. The W121/W122 S1/S3 clips are engineering regression, NOT new independent sources. Do not use GT or W123 q90 to make edits.

PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE:
- All original clips, frozen observation/model assets, BaseRig, baseline .blend, source provenance and frame/time/coordinate mapping; original and accepted Action content, W116 user Hard Pin and its independent Blender evaluation.
- W124 existing command names and documented normal-case behavior, original no-overwrite output policy, source/rig digest guards, rejected/stale/unauthorized failure classes, and actual evaluator-derived footprint.
- Default CLI/GUI and MotionGraph/AnimationClip paths, historical Worklogs and CASEs, W124 FBX baseline failures and known partial-rig/animation-quality limitations.
- Content-addressed immutable state records and version lineage. Do not treat the existence of a candidate .blend as a committed edit.

CHANGE ONLY:
- Minimal opt-in transaction and recovery implementation in src/app/experimental_video_action.py and, if strictly needed, a small opt-in helper under src/app or src/blender; focused tests, a bounded W125 harness, usage documentation and W125 Worklog. A dated status update to docs/00_마스터_플랜.md, docs/01_프레임워크_현황.md and docs/README.md is allowed ONLY for an actually demonstrated milestone.
- An experimental per-run transaction journal/manifest or versioned metadata may be added without changing Production MotionGraph/AnimationClip schemas. Existing W124 runs must remain readable or fail with an explicit non-destructive compatibility STOP; never silently rewrite them.
- Agent may choose a small transaction mechanism, but must document authority, staging, expected-parent compare-and-swap, cross-process exclusion, durable publication point, crash recovery and rollback. Distinguish process-kill durability from untested OS/power-loss guarantees.

DO NOT TUNE:
- No threshold sweeps, silent heuristic retries, new smoothing/temporal margins, hidden hard-pin relaxation, automatic edit beyond authorized frames, fake successful FBX roundtrip, source retiming, new subject identity, rig remapping or post-outcome gate changes.
- No modifications to scripts/batch_automation.py, the default FBX exporter/importer, frozen W1xx algorithm files or any previous Worklog. No branch operations of any kind.

REQUIRED EVIDENCE — A. TRANSACTION AUTHORITY AND FAILURE MODEL

Document the current normal-case head/lineage/accepted .blend/timeline invariants. Implement a single well-defined publication point: before it, the old head remains authoritative; after it, the new head resolves to a fully staged, hash-verified and independently Blender-readback-verified Action, with coherent metadata and user-visible status. Stage the new .blend, lineage record, footprint/status and needed reports BEFORE publishing the head. Use atomic filesystem operations where supported, explicit flush/durability assumptions, and fail-closed concurrent operation protection or a genuine CAS. A partially written journal/refs/log must be detectable and recoverable without inventing accepted state.

Cover ACCEPT, explicit REJECT, gate-rejected ACCEPT attempt, stale-parent ACCEPT, REVERT, and repeated/restarted invocations. A failed REVERT must leave the prior accepted head intact; a successful REVERT must have exact values and actual Blender integer/subframe equality to the restored Action. Rejected or infeasible requests must never publish a new head. Candidate staging must never mutate the original baseline or source.

Define recovery of interrupted staging, incomplete publication and metadata/log divergence; expose a clear user-facing status/repair action if needed. Do not silently delete orphan user artifacts or silently mark an incomplete operation ACCEPTED. State what happens if a process is terminated after the durable publication point but before the success message.

REQUIRED EVIDENCE — B. PROSPECTIVE FAULT INJECTION / CONCURRENCY

Create deterministic fault injection at meaningful boundaries, at least: before staging; after state file write; after candidate .blend copy; before head publication; immediately after head publication but before ancillary log/status; during Blender generation/readback for REVERT; during final timeline/metadata output. Exercise failures as exceptions AND at least one separate-process forced termination where available. Reopen the run from a new process after each and independently verify head -> state -> .blend -> readback -> displayed footprint consistency, exact old/new head identity and absence of an incorrectly accepted candidate.

Test two competing CLI operations against the same parent; at most one may publish, the other must fail stale/locked with no lost update. No lock stealing, hidden expiry, force overwrite or broad cleanup. Test a tampered staged .blend and a tampered manifest/ref/journal, and report which failures can be repaired safely versus must STOP. Freeze expected outcomes and do not relax them after a failed test.

REQUIRED EVIDENCE — C. REAL VIDEO / REAL BLENDER USER PATH

Use W124's authenticated S1 and preferably S3 original clips and known BaseRig, with source/Action digests checked against W124. A verified W124 saved .blend + source provenance may be reused as the real-video starting point; label this as a saved-artifact replay, NOT a fresh live-perception E2E. If live perception is rerun, report it separately. If the authoritative files are unavailable or hashes differ, STOP/DATA BLOCKED rather than substitute synthetic motion.

On the real Action, perform a genuinely nonzero left-wrist Hard Pin proposal, ACCEPT, deliberate injected interrupted ACCEPT or REVERT, restart/recover, explicit REJECT, successful exact REVERT, and inspect the recovered Blender Action and actual frame/Bone footprint. Compare before/after world positions at integer frames and subframes, authorized channels, pin residual, source and action timestamps, and immutable baseline. Confirm no accepted state points to a missing or unverified .blend.

Generate or reuse a native 30fps original-source | baseline | edited A/B view with truthful timeline; separately disclose source-fidelity defects, transition jumps, elbow/flip costs, partial rig and visibility/occlusion. A correct transaction is NOT proof of animation quality. Recheck existing FBX export/readback as an unchanged external baseline; do not repair FBX in this Batch. If the external renderer or Blender cannot run, label NOT RUN, not PASS.

REQUIRED EVIDENCE — D. OBSERVABILITY / USER UTILITY

A user must be able to determine the authoritative accepted/reverted .blend, current head and state lineage, whether a proposal is merely staged or actually accepted, what was changed by frame/Bone, and whether an interrupted operation needs recovery. The timeline and status must reflect the committed head, not an orphan candidate. Preserve exact undo semantics and an unmodified source. Document one reproducible CLI sequence including a simulated interruption and subsequent recovery/status inspection, with output hashes and expected visible result.

SYNTHETIC + REAL REPLAY

Synthetic: identity/no-op, nonzero feasible pin, unreachable/wrong-binding, stale parent, rejected candidate, stage-file tamper, interrupted ACCEPT/REVERT, cross-process collision, interrupted log/refs publication, restart recovery, exact undo and footprint truth. Use mock FK only when explicitly labelled MOCK; include a real Blender BaseRig test.

Real: S1 and preferably S3 30fps engineering replay on the same frozen rig/model/Action; unchanged original .blend and request; a recovered accepted/reverted .blend; Blender-evaluated pin, frame/Bone footprint, Action and native-rate review. No independent-source or general animation-quality claim from the same clips.

TESTING

Focused tests FIRST, including deterministic fault-injection and a real Blender BaseRig case; then relevant CLI, W116/W118, W124, FK/retarget, Blender/export and lineage regressions. Distinguish unit/mock, process-kill, actual Blender, authentic-video replay, and FBX. Run only bounded relevant regressions; do not misreport unrun full-suite tests. Freeze gates and report RUN/PASS/FAIL/NOT RUN separately. Compare default CLI parser/GUI isolation and no new heavy imports. Verify original assets, historical Worklogs, baseline .blend, existing accepted files and source are not overwritten.

STOP CONDITIONS

STOP with a precise BLOCKED/INVALID report if safe publication/recovery cannot be achieved without Production schema migration, default pipeline changes, unauthorized branch operations, changing frozen W116 hard-pin semantics, new numerical policy, overwriting historical outputs, or hiding a divergent state. STOP if remote changes or concurrent workers make publication unsafe. If an OS-specific durability guarantee cannot be established, explicitly scope the supported process-failure guarantee rather than claim power-loss safety. No post-hoc threshold or gate changes, no manufactured artifacts, no silent success after failure.

FINAL REPORT

Create docs/125_WORKLOG_CRASH_CONSISTENT_EDITABLE_ACTION_TRANSACTIONS.md with: starting/final commit and code diff; pre-outcome declaration, authority map and old failure windows; transaction design and durability limits; exact failure-point matrix, crash/restart and concurrency outcomes; original/accepted/reverted state and .blend hashes; independent Blender world-pin/footprint/undo evidence; S1/S3 real source and 30fps A/B with source-quality negatives; unchanged FBX baseline; focused/regression/real execution status; performance/maintenance/rollback and any unrecovered orphan; honest official engineering CASE and remaining blockers. Every numeric/visual result must be labelled Agent-run unless independently reproduced. Do not recommend or draft W126.

CASE A: all declared transaction integrity and restart/concurrency gates pass, with actual real Blender source-bound accept/revert/footprint and default-path isolation; this is TRANSACTION CAPABILITY support only, NOT general Animation/FBX Production readiness.
CASE B: meaningful real Blender transaction safety improvement, but a declared platform/failure class or user-visible recovery path remains limited and explicitly bounded.
CASE C: valid experiment but crash-consistent real user workflow not demonstrated.
CASE D: validity/contract contamination prevents trustworthy interpretation.
DATA BLOCKED / NO CASE: essential real Action/Blender/runtime assets unavailable before real evaluation. Historical W124 CASE B is immutable.

COMPLETION CONDITION

Batch 125 is COMPLETE only after an opt-in transactional accept/reject/revert implementation OR a precise STOP is documented, prospective fault injection and cross-process recovery are honestly evaluated, the user-visible head/Action/footprint and real Blender evidence are reported with hashes, unchanged baselines and regressions are accounted for, and the Worklog ends with exactly:
AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 125

W125 COMPLETE triggers a mandatory independent cumulative audit of W121–W125 BEFORE W126 publication. Claude is Evidence Producer only; no next-Batch or Architecture recommendation, no modification to docs/next_batch.md.