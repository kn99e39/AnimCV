# BATCH 106 — Existing Rig Action Editability and Round-Trip Contract (Evidence Only)

## CURRENT ASSESSMENT — SUCCESS / FAILURE / UNCERTAINTY

W105 is COMPLETE and conditionally CASE A on 417 uniquely associated 3DPW leg-restoration rows. Fixed R11 improves W102 rig geometry there, but the 199-frame historical real clip contains only wrist omissions and shows a new 80 ms hand-to-face flicker. W105's source-only R11 remains diagnostic, not a production observation policy. W101–W105 were cumulatively audited at `docs/AUDIT_W101_W105_2026_11_02.md` (PASS WITH DOCUMENTED LIMITS; Main-local binary evidence not independently replayed). Do not reopen the W105 audit as a prerequisite.

The product is **editable animation creation**, not an endless pose-estimation benchmark. W105 proved that the evaluation-only Blender output has inspectable Actions (32 F-curves, keyed sample frames, CONSTANT interpolation), **not** that a user can reliably edit, preserve, save, and reuse the resulting motion. In the real-clip diagnostic, 199 keys are at native 50 fps frame indices but the saved Blender scene plays at 30 fps (0.6× actual speed). In 3DPW, the source bank is 10 Hz. Existing BaseRig Actions are a useful but unvalidated boundary between reconstruction and authoring.

**Human Assessment:** SUCCESS — a concrete rig Action is generated and inspectable, and the fixed W102 contract supports controlled rig comparison. FAILURE — real-clip playback timing is mismatched in the saved scene, and R11 arm flicker is visibly harmful. UNCERTAIN — whether existing Actions can support a reversible, local expressive edit and survive Blender/FBX export without unintentional track changes; whether the apparent editability is merely keyframe existence. ALTERNATIVE EXPLANATIONS — action-level edits may work while rig/world transforms, sampling semantics, FBX import or playback remain defective; differences may arise from exporter or rest-pose conventions rather than edit representation.

This is an **artifact-contract diagnostic**, not authorization to adopt R11, repair retargeting, or build a style/authoring subsystem.

## HUMAN-READABLE SUMMARY

**DIRECTION:** Measure the actual controllability, isolation, timing and round-trip reliability of *already exported* BaseRig Actions using bounded, reversible scratch edits.

**PURPOSE:** Determine whether the current reconstructed rig Action is a credible editable handoff artifact for a future user-directed animation workflow, or merely an evaluation render. Identify the first blocking ownership/serialization/timing boundary without redesigning it.

**CENTRAL INTENT:** Can one meaningful local bone-motion edit and one local timing edit be applied to an existing generated BaseRig Action, produce the intended observable rig change, preserve declared non-target animation, and survive save/reopen and optional FBX round-trip—without modifying reconstruction, rig, or production code?

**DO:** Reuse W102/W105 frozen exports and historical clip where available; measure original Action, null controls, edit/inverse, local FK and visual consequences, Blender save/reopen, timing and export/import parity; retain negative cases and sample provenance.

**DO NOT:** Implement an animation editor, motion-style model, automatic retiming/filtering, new Motion IR, new observation policy, IK/root/contact repair, runtime FPS fix, exporter rewrite or production integration. Do not train or infer a new pose, tune thresholds, alter a frozen rig, or acquire/publish media.

**DECISION ENABLED:** Whether the current Action can serve as an inspectable/editable output boundary, and which *existing* contract blocks it if not. No CASE approves a new authoring architecture or deployment.

## DIRECTION AND OWNERSHIP

Work only in `kn99e39/AnimCV`, existing `arch/single_frame_first` branch. Read `AGENTS.md`, `PRODUCT_VISION.md`, `Architecture_v3_FramePose.md`, `Architecture_v2.md` (Motion Graph/retarget/Blender export sections), `COORDINATE_CONVENTIONS.md`, cumulative audits, W99A/W100, W102 and W104–W105, and the automation worker contract. Preflight remote HEAD, instruction blob SHA, working tree, lock and active jobs. Follow the existing automation worker's commit/publication protocol; do not create, switch, delete, merge, rebase or reset branches, force-push, resolve conflicts, or alter `docs/next_batch.md`. No independent next-Batch recommendation.

This is not a decision to prioritize style authoring over unresolved production safety. It is a one-time diagnostic of the current artifact boundary. Stop rather than implementing an editor or changing architecture.

## PURPOSE AND CENTRAL INTENT

**Architecture hypothesis H106:** The current BaseRig Action is sufficiently self-contained to support *at least one bounded manual-style local motion edit* with explicit preservation of other channels and deterministic persistence. If this fails, the failing boundary (Action representation, rig hierarchy, timebase, serialization, exporter or missing information) can be identified without changing the production system.

The falsifier is **not** lower MPJPE. A valid Action must be editable and replayable at the right time, and a deliberate edit must remain visible while unrelated channels are preserved. A scratch-only edit proves an artifact capability, not a user-facing authoring system.

## HISTORICAL BASELINE

Preserve byte-identical all W102–W105 outputs, prior Worklogs/verdicts, W99A production modules, original S0/HMR/J1X predictions, W91 renderer, W95 two-bone IK, BaseRig asset/rest/calibration, frame/timestamp alignment, original interpolation and Action values. Preserve the complete-frame no-op evidence and all W105 harms (18 newly failed individual legs, seated-cafe source/rig regression, real-clip R11 flicker). Never overwrite existing `.blend`, `.fbx`, media, reports or checkpoints.

Primary artifacts:
- W105 Main-local `C:/Users/dna10/animcv-w105/results/` with manifest and saved candidate `.blend` files; W102 Main-local rig exports if needed. Verify hashes and case/slot identity against Worklogs; do not treat inaccessible outputs as verified.
- 3DPW `complete_control` and one predeclared, usable moving-window Action under the frozen W102 mixed-source contract (10 Hz); keep M_Main as primary. W105 R11 is optional **negative comparison only**, not a new baseline.
- Historical real-clip **M_Main** S0-only Action, 199 native frame indices at 50 fps, saved scene 30 fps; explicitly separate this from the W102 mixed-source contract. Do not redistribute the source clip or stills externally.
- The committed `examples/e2e_demo/animated_rig.blend`/`.fbx` may serve as a **legacy smoke** only, separately labelled and never pooled with W102/W105 metrics.

## PRESERVE / CHANGE ONLY / DO NOT TUNE

**PRESERVE:** Original Actions, original keys, rest and bone naming, rigs, source timestamp/cadence, all observation/model contracts and negative results. Keep original files read-only.

**CHANGE ONLY:** New isolated W106 diagnostic scripts, focused tests, W106 Worklog and scratch-only copies of Blender Actions/exports and reports. Test edits are explicit evaluation interventions on copied Actions, not new production functionality. Any Blender scene FPS change is scratch-only, labelled sensitivity; never overwrite the original.

**DO NOT TUNE:** S0, raw RTMPose confidence/visibility, subject identity, pose/rig targets, keyframe collapse, interpolation, frame-rate metadata in baseline, IK/retarget, FBX exporter flags, root/torso ownership, contact, temporal stabilization, thresholds or error tolerances after viewing results.

## 1. PREFLIGHT AND DECLARATION (BEFORE EDIT RESULTS)

1. Verify W105 COMPLETE and audit, immutable source Action identities, Blender version, scene FPS, frame indices, armature, active Action, layered F-curves, channel paths, rest/calibration, file hashes, original native timestamp contracts and private-media provenance. Record exactly which outputs are available; no hidden substitution if absent.
2. Select **before viewing edit outcomes** one complete-control Action and one real moving Action with an existing animated left-arm or leg track. Select a deterministic window from existing metadata (for example, first qualifying moving window by stable sorted case ID); state why it is eligible. Do not cherry-pick a visually successful edit.
3. Freeze an evaluation declaration JSON + SHA: exact file/case/window, armature and target bone/channel, selected keys, edit operation, intended changed descendants, non-target local channels, original and intended timestamps, tolerances, measurement units, evaluation views, optional FBX settings, expected stop gates and CASE mapping. No post-hoc widening of gates.
4. If no existing real Action can be opened and associated with a reliable timestamp/rig contract, record **CASE D**. Do not reconstruct a replacement with new inference or invent an editability PASS.

## 2. PAIRED EDITABILITY PROBES (COPIES ONLY)

Use **M_Main** and exact copied baseline as the paired control. Do not apply R11 by default.

- **N0 null/reopen:** Copy an original Action to scratch, save/reopen without edits, sample all keyed frames and a declared set of between-key frames, and compare local channels, world bone transforms, keyframe counts/locations, scene timing and rendered poses to the original. Measure drift before interpreting any edit.
- **E1 local motion-amplitude edit:** Apply a deterministic, bounded local rotation offset to **one already animated bone** over a predeclared short window (e.g. a modest forearm/upper-arm swing or leg-lift adjustment). Use only existing Blender Action editing, no new solver. Define the intended target and hierarchical descendants before execution. Confirm the edit changes the intended rig motion visibly, not merely numeric curves; verify all non-target **local** channels and non-descendant world transforms remain unchanged within the declared numerical tolerance. Descendant world changes are expected and must not be mislabelled harm.
- **E2 local timing edit:** In an independent scratch copy, move a small selected group of keys on one declared track by a fixed positive/negative offset while preserving order and frame-domain bounds. Verify the intended phase/timing change and that other tracks retain their original keys and sampled local values. Do not infer high-FPS motion from 10 Hz data or silently interpolate.
- **E3 reversibility:** In separate copies, apply exact inverse edits and compare Action value/key digests and sampled FK transforms to N0. If an inverse cannot exactly recover because of quaternion sign, key merging or Blender normalization, record the discrepancy and its animation impact rather than relaxing the gate.
- **E4 save/reopen and FBX:** Save/reopen E1 and E2; compare edit effect and non-target preservation. If the existing export/import environment supports FBX, export a copied Action through **unchanged existing settings**, import into a clean scratch scene, and compare baseline and edited versions *relative to the unedited FBX round-trip drift*. Record rest-pose, bone naming, channel and timebase differences. If FBX is not available or reliable, label the Blender-only conclusion; do not silently claim FBX success.

Keep each intervention independent; never combine an E1/E2 edit into a new optimized or production policy.

## 3. TIMEBASE AND ANIMATION SEMANTICS

For the W105 real clip, record original 50 fps timestamps, 199 keyed source-frame indices and saved Blender 30 fps scene rate. Quantify actual duration/playback factor and whether save/reopen or FBX preserves that mismatch. An **optional scratch-only 50 fps scene** may demonstrate the timebase effect, but it is not a production fix. Keep original 10 Hz 3DPW bank semantics separate and do not interpret 30 fps/50 fps Blender playback as measured temporal reconstruction.

Report original CONSTANT interpolation and its consequences for in-between-frame appearance. Do not change interpolation to make the edit appear smoother. Inspect the existing flicker and missing root/contact semantics as known limitations; do not repair them or claim that the test covers them.

## 4. EVIDENCE AND CONTROLS

Produce machine-readable per-case baseline/edit/reopen/round-trip reports with original source IDs, Action and rig hashes, key/frame/timestamp alignment, per-channel value/keyframe deltas, quaternion/FK/world-transform deltas, root and unaffected-bone preservation, scene FPS, render evidence and manifest. Include at least:
- Synthetic minimal armature/Action probe to test expected local-vs-descendant FK behavior, edit inversion, quaternion equivalence and serialization independently of the W105 rig.
- Actual existing W105 3DPW BaseRig Action(s) and the GT-free historical real-clip Action; if unavailable, report exact missing prerequisites and a narrower verdict, never a synthetic-only product PASS.
- No-op and unrelated-channel controls; unedited FBX round-trip reference; deterministic repeated run on copies; missing-key/CONSTANT interpolation boundary; start/end window and negative-key cases.
- One fixed-view before/after paired Blender render or contact sheet for each real edit, plus a neutral numeric report. Private source frames stay local. Do not infer user preference, expressive style transfer, contact quality or unseen-scene performance.
- Focused tests plus relevant existing Action/export/rig tests, not a blanket full regression absent a shared production change. Report executed, skipped, failing and nonexecuted checks separately.

A significant result is a **visible, controlled rig edit with preservation and replay**, not simply a modified F-curve or a passing unit test. Distinguish synthetic contract evidence, actual saved-Action behavior, real-clip playback and future product usability.

## 5. PREDECLARED DECISION RULES

Before viewing results, specify and freeze numerical tolerances justified by Blender quaternion/FK precision and the original no-op/FBX drift. Do not set tolerance so wide that a visibly different pose passes. Distinguish channel-value equivalence from quaternion-sign equivalence and world-transform equivalence.

- **CASE A — Bounded Action editability supported:** N0 identity holds; E1 produces a visible intended local motion change without undeclared non-target change; E2 changes intended timing while other tracks remain unchanged; inverse and Blender save/reopen preserve the declared contract; real W105 Action(s) are tested. FBX may be labelled a separate unresolved gate if not available; do not claim end-to-end export unless it passes.
- **CASE B — Action boundary fails:** A real Action is available and measured, but local editing, preservation, inversion or save/reopen violates the predeclared contract, or the intended visible edit cannot be achieved. Attribute the first failing layer.
- **CASE C — Mixed / inconclusive:** The probes work only in synthetic/legacy data, only for one track or scene, or timing/export effects cannot be isolated; report conditional sub-results and failures.
- **CASE D — Contract blocked:** Required frozen artifacts, reliable Action identity, Blender/rig environment, source timing, reproducible test setup or baseline equivalence cannot be established. Stop without fabricating results.

The original real-clip 30/50 fps mismatch is a **known negative finding**, not automatically a failed local edit; it prevents any claim of native-time playback or complete end-to-end authoring readiness. A Blender-only CASE A must remain explicitly Blender-only. No CASE authorizes production code or architecture changes.

## STOP CONDITIONS

STOP and report CASE D or a narrower negative outcome if frozen artifacts/hashes cannot be verified; a required file is missing; GT enters edit construction; edit results require modifying original Action/rig; a meaningful edit cannot be predeclared without selecting on success; timebase cannot be associated with source; exporter requires changing production settings; identity/camera/reference contracts are conflated; existing worker lock/branch/push contract is violated; or unexpected resource, license, privacy or safety problems arise.

Do not repair the underlying product system, implement a new editor, retargeting policy, root-motion method, frame-rate fix, automatic filtering, observation threshold or style subsystem in response to a failure. Negative evidence is valid completion.

## FINAL EVIDENCE-ONLY REPORT

Write `docs/106_WORKLOG_EXISTING_RIG_ACTION_EDITABILITY_AND_ROUNDTRIP.md` with: original commit and exact data/rig/Action fingerprints; predeclared gates and SHA; selected cases/keys and why; paired synthetic/real results; native-time versus Blender-time measurements; before/after/inverse/null controls; save/reopen and optional FBX drift; concrete edit visual evidence; preservation violations and negative examples; test counts, provenance, reproduction commands, scratch outputs and resource use; precise CASE A/B/C/D; and clear separation of current implementation from future animation authoring.

Explicitly state whether the result proves only **editability of an existing Blender Action** or also an FBX round-trip. Do not claim user intent understanding, style transfer, production animation quality, GT-free root/contact reconstruction or user-facing editing UX.

Do **NOT** include Next Work Recommendation, Next Architecture Decision, Proposed Next Batch or Next Implementation Plan. No Agent-led next-work suggestion.

## COMPLETION CONDITION

Complete only when the declared real Action contract has been tested or a precise blocker has been preserved, evidence and limitations are written, focused tests and artifact manifests are recorded, and the final Worklog plus permitted diagnostic files are committed under the existing worker protocol. Add `AUTOMATION_BATCH_STATUS: COMPLETE` and `BATCH_NUMBER: 106` **only after** all outputs and Worklog are finalized and safely committed. Do not modify `docs/next_batch.md`, the existing baseline or any production path. A supported failure, mixed outcome or blocked result is valid completion.
