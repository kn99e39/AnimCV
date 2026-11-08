# BATCH 126 — Commit-Time Semantic Gate Integrity for Reversible World Pin

DIRECTION

PRODUCTION-ORIENTED ITERATIVE DEVELOPMENT. Close one demonstrated user-authoring safety hole in the existing EXPERIMENTAL, opt-in video -> editable Blender Action -> WORLD_POSITION_PIN path. This is a bounded Production Engineering batch, not a new pose/animation research mechanism. Use ONLY the existing `arch/single_frame_first` branch; do not create, switch, delete, merge, rebase, reset, or force-push branches. Do not propose, issue or edit any future Batch.

PURPOSE

W124 CASE B delivered the opt-in real-video/known-BaseRig editable Action and normal-case pin/undo. W125 officially completed CASE A **for process-failure transaction capability only**: staged, hash-checked Blender readback before head publication, recovery and cross-process exclusion. W125 did NOT establish semantic authorization of the candidate being published. Its Worklog explicitly discloses a post-outcome exploratory failure: editing an UNREACHABLE candidate's unsealed `proposal.json` decision to ACCEPT allowed `commit --decision accept` to publish a 136 mm hard-pin violation with exit 0. That is AGENT-RUN evidence, not independently rerun by the Reviewer. The Reviewer independently confirmed the corresponding code path: `_commit` trusts the proposal's decision, while `_publication_checks` verifies copy/readback/footprint parity against the same proposal and merely reports pin residual without gating it. `request.json` and `bindings.json` also lack an explicit decision-time immutable identity guard.

The mandatory W121–W125 cumulative GitHub code/Architecture audit has been completed at `docs/AUDIT_W121_W125_2026_11_09.md`, commit `1f390d19e0030dbd27584d3771813fe4db0a9eec`. Its verdict is PASS WITH CRITICAL DOCUMENTED PRODUCTION BLOCKER, not Production readiness. Historical W121 D, W122 B, W123 B, W124 B and W125 A remain frozen.

CENTRAL INTENT

Can the existing opt-in editor make **the commit-time semantic decision independently authoritative**—so no edited/stale/contradictory proposal, request, binding or candidate metadata can cause a gate-rejected or infeasible WORLD_POSITION_PIN to publish—while preserving W125 crash-consistent publication/recovery, exact Hard Pin and undo, truthful frame/Bone footprint, existing real BaseRig Action, and unchanged default CLI/GUI?

CURRENT ASSESSMENT / HUMAN ASSESSMENT

- CODE-VERIFIED: `src/app/experimental_video_action.py` writes `edits/<id>/proposal.json` with mutable `decision`, `constraint`, `footprint`, `display`, `candidate_blend` and `pin_frames`. `_commit` branches on `prop["decision"]["decision"]`; `_accept` passes that decision to W116 Store. `_publication_checks` compares accepted copy to candidate and recomputes footprint, but does not require a fresh independent frozen W116/W118 semantic gate PASS or bound original request/binding identity. `run.json` guards the original baseline .blend, sidecar, manifest and baseline readback; it does not guard `baseline/bindings.json` or `edits/<id>/request.json`.
- W125's fault matrix, concurrency, legacy adoption and recovery have value and must NOT be discarded. Its CASE A is only a transaction-capability result, not permission to accept a bad pin.
- The owner already approved immutable Source Evidence, explicit hard WORLD_POSITION_PIN, bounded truthful edit footprint, atomic accept/reject/revert and Production-oriented integration. **Enforcing that existing contract does not require choosing a new joint-quality policy.** Do not silently introduce new tolerance, default editing policy or semantic constraint type.
- Real S1/S3 saved-artifact replays are appropriate for engineering regression, not new independent source generalization. W124 FBX round-trip FAIL and limited source fidelity remain negative baselines.

HUMAN-READABLE SUMMARY

**Direction:** make user-directed semantic edits safe at the moment they become an accepted Blender Action. **Purpose:** remove the documented proposal/gate bypass without rebuilding the architecture or expanding feature scope. **Central Intent:** only an independently revalidated, source-/request-/binding-/candidate-bound Hard Pin may be published; tampering, stale context and disagreement must STOP without moving the head. **DO:** implement minimal opt-in decision provenance and commit-time gate revalidation, test the real negative reproduction, preserve crash/recovery and real Blender editing. **DO NOT:** train a Generator, change FramePose or solver thresholds, relax Hard Pin, add constraints, repair FBX, modify default CLI/GUI or migrate Production MotionGraph/AnimationClip schemas. **Decision enabled:** whether the experimental editor now has a trustworthy semantic acceptance boundary, distinct from W125's transactional durability and from still-limited animation/FBX quality.

HISTORICAL BASELINE / PREFLIGHT

Before editing, read remote HEAD, `docs/next_batch.md`, `AGENTS.md`, `PRODUCT_VISION.md`, `Architecture_v3_FramePose.md`, relevant `Architecture_v2.md`, `COORDINATE_CONVENTIONS.md`, `docs/AUDIT_W121_W125_2026_11_09.md`, W121–W125 official Worklogs, `docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md`, the W124/W125 focused tests and harnesses. Inspect actual `src/app/experimental_video_action.py` (`_propose`, `propose_core`, `_commit`, `_accept`, `_publication_checks`, `_finish`, `_recover`, `load_run`, `verify_head`), `src/app/experimental_action_txn.py`, `src/blender/experimental_action_stage.py`, frozen W116/W118 semantic gate/evaluator and state/footprint code. Check the currently published W125 instruction blob `94355c1c1cdaa48d5ab5db7a21926555a3990ebb` only as historical provenance; this W126 prompt is now the instruction.

Record W125's accepted/rejected candidate and crash/recovery baselines, exact code and artifact hashes, source/rig/timebase identities, prior negative reproduction and the trust boundary. Freeze a threat model, binding strategy, negative-control matrix, expected failure-class precedence and CASE gates **BEFORE** examining W126 outcomes. A local SHA/seal is a consistency check, not proof of authenticity against a hostile actor able to rewrite all files. Do not claim such adversarial security.

PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE: all frozen FramePose/perception/retarget and W116/W118 algorithms, known W105 BaseRig, original source/video provenance, 30 fps source-to-Action mapping, original and accepted Blender files, content-addressed states, exact undo, explicit authorized DOF, hard scene-world wrist pin, actual Blender-evaluated footprint, W125 v2 head/journal publication semantics, OS run lock, recovery and legacy read-only compatibility. Keep all historical Worklogs/CASEs and the unchanged failing FBX baseline. Preserve the default CLI/GUI and existing public command behavior for valid requests.

CHANGE ONLY: bounded opt-in `src/app/experimental_video_action.py` and, if necessary, `src/app/experimental_action_txn.py` or one small opt-in helper; focused tests and a W126 prospective harness; experimental usage doc; W126 Worklog. Versioned slice-private proposal/decision evidence is permitted, **not** a migration of MotionGraph/AnimationClip or normative Production schemas. A dated update to living milestone/status docs is allowed only for a demonstrated capability and must preserve older history. No branch operations or edits to `docs/next_batch.md`, automation scripts, W1xx frozen code, existing FBX importer/exporter, previous Worklogs, original assets, default GUI or model checkpoints.

DO NOT TUNE: no numerical gate changes, threshold sweeps, automatic post-hoc gate relaxation, new solver heuristic or retiming, inferred permanent elbow/flex policy, unbounded transition editing, unapproved style/Generator training, new constraint types, silent repair of FBX or invented human visual ratings. W124's 1 mm pin check is an **existing experimental diagnostic**, not a new permanent Production tolerance.

REQUIRED EVIDENCE — A. DECISION AUTHORITY AND TRUST BOUNDARY

Establish and document an independently verifiable chain from (1) immutable source/manifest and original Blender parent state/readback, (2) original user request bytes and evaluated capture target, (3) verified rig binding and explicit authorized channels, (4) candidate state/.blend and fresh Blender readback, (5) frozen W116/W118 solver statuses, pin/side-effect/footprint/display gates, to (6) the publication decision. Neither `proposal.json`'s `decision` field nor a value derived solely from that file may serve as independent gate authority. A fresh revalidation must use authoritative input identities and independently evaluated Blender state, and detect if the candidate differs from what the original frozen request/solver would produce. If recomputation is used, it must use the unchanged frozen algorithms and identical source, frame, rig and DOF contracts.

At COMMIT, before the W125 publication point, fail closed on any unbound/mutated request, binding, target, candidate, parent, gate decision, allowed-domain or displayed footprint. Check the actual pin residual and all relevant existing frozen semantic gates, including solver failure/reachability and unauthorized motion; parity to a tampered proposal is insufficient. Preserve correct failure-class precedence and explain what is recomputed vs what is hash-bound. Repeat any essential final checks after staging/fresh Blender readback and before publishing; do not let a stale head or TOCTOU race publish. A rejected proposal must not become ACCEPTED by editing metadata or changing the CLI decision flag.

Protect the user-visible proposal/status/timeline from falsely showing a gate PASS when authoritative revalidation fails. A failed semantic authorization must leave the head and existing accepted Action unchanged; classify the proposal clearly and preserve evidence without overwriting historical artifacts. Revert and recover must remain exact, and recovery must never invent a valid semantic decision. Existing W124/W125 runs must remain inspectable; if old proposals cannot establish trusted provenance, require an explicit non-destructive re-propose/adopt/STOP rather than silently grandfathering an unverified ACCEPT. Do not retroactively reclassify historical accepted runs or rewrite historical Worklogs.

REQUIRED EVIDENCE — B. PROSPECTIVE NEGATIVE CONTROLS

Reproduce the W125 exploratory **UNREACHABLE candidate -> hand-edit proposal decision to ACCEPT -> CLI accept** bypass on a COPY of a real Blender BaseRig fixture. The W126 implementation MUST block publication, and the head/.blend/lineage/timeline must remain unchanged. Freeze exact expected outcomes for each negative case, including: changed proposal decision/status; forged low residual; modified pin target, pin frames, constraint digest, authorized channels, binding id or changed-frame/Bone footprint; changed request bytes or substituted binding record; candidate state or .blend tamper; stale parent; conflicting decision/status; unreachable/nonconverged candidate with a real candidate file; altered or missing independent readback. A benign reserialization/formatting control should be specified if the chosen binding strategy distinguishes semantic from byte identity. Include negative controls for stale/rejected decisions and for errors with simultaneous causes so classification precedence is tested prospectively, not adjusted after seeing results.

Every rejected control must prove: no head generation change, no accepted .blend publication, no false ACCEPTED display, no unauthorized user edit, and no silent repair of source/rig/binding. A passing SHA check alone is not a semantic gate PASS.

REQUIRED EVIDENCE — C. VALID ACCEPT, TRANSACTIONS AND REAL BLENDER

Demonstrate a legitimate nonzero feasible user pin still ACCEPTs on the known real BaseRig, has an independently verified scene-world residual under the unchanged experimental gate, and reports the **actual** changed frames/Bones from the Blender evaluator. Include a no-op case (not evidence of improvement), explicit REJECT, stale-parent refusal, exact REVERT and immutable original. Exercise W125 interruption/restart recovery both before and after publication, competing CLI operations and tampered journals/refs to show the new gate checks do not break atomicity or create a new acceptance bypass through recovery.

On the authenticated W124/W125 S1 and S3 sources, use original saved artifacts with verified hashes and label them **SAVED-ARTIFACT REAL-VIDEO REPLAY**; do not claim fresh perception. Execute one valid accept/reject/revert and at least one tampered-proposal attempted accept per source if assets are available. Reopen saved .blend in a fresh Blender process, verify integer/subframe values, hard-pin residual, footprint/lineage/undo, and user-visible timeline/status. Include source | baseline | edited native-30fps A/B (existing verified render may be reused with its hash if byte-identical; no fabricated fresh visual review). Separately disclose source-fidelity errors, partial rig, transition jumps, near-straight elbow and unverified subject/GT. Characterize the existing FBX export/reimport as unchanged FAIL; do not repair it.

REQUIRED EVIDENCE — D. REPRODUCIBILITY AND MAINTENANCE

Freeze and hash the pre-outcome implementation, harness, data identity, controls, expected error classes and decision gates. Maintain a clear, minimal compatibility policy for unsealed W124/W125 proposals and interrupted transactions. Verify source/rig/frozen-code guards, no default CLI/GUI behavior change, no heavy imports on the default path and no new Production schema. Report extra cost of commit-time verification, storage and user-facing error/re-propose procedure. Include a minimal end-user CLI reproduction of a tampered proposal being refused and a valid proposal being accepted/undone.

SYNTHETIC + REAL REPLAY

Synthetic/controlled: valid nonzero pin, trivial/no-op, unreachable, solver nonconvergence, wrong binding, wrong authorized DOF, altered request, edited proposal fields, forged decision and residual, mismatched candidate/footprint/readback, stale head, rejected request, concurrent commits, injected pre/post-publication kill, recovery, exact undo and legacy compatibility. Mock FK tests are explicitly MOCK; include actual Blender BaseRig tests. Preserve negative results and every original W125 fault baseline.

Real: S1/S3 original video provenance + frozen Action identity; valid user pin and tampered gate-rejected pin; actual Blender world points, frame/Bone footprint, subframes, accepted/reverted .blend hashes, timeline truth and 30fps review. S1/S3 are correlated engineering regression sources already used in W121–W125, not new generalization evidence. If local files are unavailable or hashes disagree, STOP/DATA BLOCKED rather than substitute synthetic motion for the required real tests.

TESTING

Run focused W126 tests FIRST, then relevant W124/W125 transaction, CLI, W116/W118, Blender/FK/retarget and lineage regressions. Reuse a bounded but representative subset of the W125 fault matrix plus the critical pre/post-publication/recovery/concurrency cases; if any previously passing W125 failure class regresses, mark FAIL. Distinguish UNIT/MOCK, actual Blender, separate-process fault injection, real saved-video replay, native-rate video review and FBX. Report RUN/PASS/FAIL/NOT RUN and exact command/environment. Do not claim unrun full-suite tests or reviewer-independent reruns. Verify no original source, baseline, Worklog or accepted file was overwritten.

STOP CONDITIONS

STOP with a precise BLOCKED/INVALID Worklog if a trustworthy commit-time semantic decision cannot be obtained without changing frozen W116/W118 gate mathematics, choosing a new human joint/quality policy, changing the default CLI/GUI, migrating Production schemas, weakening hard pin, overwriting historical runs or conducting forbidden branch operations. STOP on concurrent remote changes that make safe publication impossible. STOP if candidate/parent/readback provenance is irrecoverably ambiguous; never publish on the strength of `proposal.json` alone. Do not fix failures by tuning thresholds, changing test populations or post-hoc relaxing gates.

FINAL REPORT

Write `docs/126_WORKLOG_COMMIT_TIME_WORLD_PIN_GATE_INTEGRITY.md` with starting/final HEAD, pre-outcome declaration and file hashes; the exact W125 bypass and independent CODE explanation; chosen trust boundary and compatibility behavior; source/diff/test inventory; prospective negative-control matrix including the original unreachable-bypass reproduction; semantic decision and Blender-evaluated pin/footprint results; W125 fault/concurrency/recovery parity; real S1/S3 saved-artifact evidence and 30fps A/B with quality negatives; unchanged FBX FAIL; performance, maintenance, rollback, unsupported threat classes and unrun tests. Separate Agent-run numbers from Reviewer verification. Update only relevant living milestone docs if a real capability is demonstrated. Do not propose or draft W127.

CASE A: every declared semantic authorization/negative-control and transaction regression gate passes, with actual Blender and S1/S3 source-bound accept/reject/revert/footprint, without weakening existing policy. **This supports only the bounded opt-in semantic gate integrity**, not full animation quality, FBX portability or a Production default.
CASE B: meaningful real Blender gate-integrity improvement but a declared class or compatibility path remains limited and transparently bounded.
CASE C: valid work but trustworthy commit-time semantic authorization is not demonstrated.
CASE D: invalid comparison, test/contract contamination or untrustworthy gate instrumentation.
DATA BLOCKED / NO CASE: essential authenticated real Blender/video artifacts unavailable before real evaluation. Preserve W125 CASE A and every earlier verdict exactly.

COMPLETION CONDITION

W126 is COMPLETE only after the commit-time semantic acceptance boundary is implemented or a precise STOP is documented; the original W125 bypass and prospectively frozen negative controls are honestly tested; actual Blender hard-pin/footprint, W125 transaction/recovery compatibility, real-video saved-artifact replay, unchanged FBX and source-quality limitations are reported with evidence; and the Worklog ends with exactly:
AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 126

The next five-Worklog cumulative audit is W126–W130 after W130 COMPLETE and before W131 issuance. Claude is Evidence Producer only, with no Architecture decision or next-Batch issuance authority.
