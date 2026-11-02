# BATCH 105 — Source-Only Joint Restoration: Rig-Level Utility and Failure Boundaries

## CURRENT ASSESSMENT — SUCCESS / FAILURE / UNCERTAINTY

W103 (CASE C) separated the frozen S0 failure regimes: most severe frames and wrong-knee branches coincide with missing limb observations, while residual depth-sign errors on complete inputs have mixed and largely unresolved causes. W104 (declared CASE A) established a controlled causal effect of removing valid joints. On 4,803 well-observed rows, removing tokens at a fixed crop increased MPJPE by +84.5 mm; recropping alone contributed +10.0 mm on average, with large out-of-range exceptions for bilateral leg loss. In 1,223 naturally partial same-person rows, restoring the *existing* below-0.3 raw RTMPose coordinates (R11) improved MPJPE by 33.9 mm, reduced severe frames 133 to 24, and fixed 174 knee branches while newly breaking 13. These are strong frozen-S0 diagnostic results, not approval to change production masking.

The decisive missing evidence is downstream **animation usefulness**. W104's real 199-frame video contains only 38 left-wrist omissions and no natural knee/ankle omissions; its GT-free R11 output changed, but the quality of that change is unknown. W104 did not generate a new BaseRig render. W102 showed that source-to-rig discrepancies can exceed 90 degrees under the existing fixed-length zero-twist IK, so source MPJPE gains need not translate to better editable rig motion. W104's evaluation also excludes other-person/ambiguous actor associations, contains reused 3DPW TEST data, and has a limited low-confidence failure distribution. No unseen generalization or production benefit has been demonstrated.

This Batch must close the *source geometry -> fixed rig animation* evidence gap, inspect harm and distribution limitations, and determine whether the diagnostic R11 effect remains useful in the actual animation medium. It must not adopt R11 as a production policy or expand into another estimator research loop.

## HUMAN-READABLE SUMMARY

**DIRECTION:** Evaluate the downstream animation consequences and failure boundaries of W104's source-only raw-joint restoration, with frozen geometry, rig, and observation contracts.

**PURPOSE:** Determine whether W104's measured reconstruction gain produces better rig-compatible motion rather than merely improving a 3D metric, and identify where the gain is negated by low-confidence hallucinations, temporal instability, identity coverage, or existing IK/retargeting limitations.

**CENTRAL INTENT:**

> With the existing observation source, frozen S0, and exactly the same BaseRig/FK/IK path, does the W104 R11 diagnostic produce a demonstrably more faithful and usable animation than M_Main on naturally partial observations, without material new animation failures? Where is any remaining failure owned: 2D observation, frozen S0, or the fixed retargeting contract?

**DO:** Verify W102–W104 identities; compare M_Main and fixed R11 on the same real actor/time samples; generate and inspect matched BaseRig animations where feasible; quantify source-to-rig attribution, temporal and anatomical consequences, harms, and unchanged complete-frame controls; include the existing real-video clip as a limited GT-free diagnostic.

**DO NOT:** Modify production observation thresholds, masks, S0, checkpoints, detector, tracking, crop, IK, retargeting, root orientation, animation semantics, rig assets or runtime. Do not introduce a new imputation/temporal policy, fuse HMR2, train, optimize the render, acquire new footage, or implement an animation authoring subsystem.

**DECISION ENABLED:** Whether source-only low-confidence restoration has evidence of *downstream animation utility* sufficient to justify a later human-owned observation-policy decision, or whether rig/identity/temporal risks make that claim premature. No CASE approves implementation.

## DIRECTION

Continue only on the existing branch `arch/single_frame_first` in `kn99e39/AnimCV`. Read `AGENTS.md`, `PRODUCT_VISION.md`, `Architecture_v3_FramePose.md`, `Architecture_v2.md` for downstream ownership, `COORDINATE_CONVENTIONS.md`, the cumulative audit, W95–W104 (especially W102 rig review, W103 CASE C and W104 CASE A), and the current automation worker's safety contract. Confirm remote HEAD, instruction blob SHA, dedicated checkout and running-job/lock state before doing any work.

No branch creation, switching, deletion, merge, rebase, reset, force-push or conflict resolution. Do not modify `docs/next_batch.md`: it is ChatGPT-owned and READ-ONLY. Preserve W104's COMPLETE marker and every historical verdict. Use the worker's existing lock, completion and publication protocol. Do not create or archive a copy of this prompt.

## PURPOSE

W104 establishes a promising *diagnostic intervention*, not a safe policy. A lower MPJPE can coexist with wrong limb direction, incorrect contact, motion pops, unnatural fixed-length IK, uneditable curves, or subject identity failures. The Product Vision requires validation in the final animation medium and avoidance of an endless isolated pose-estimator loop. Test that exact gap without changing the method.

## CENTRAL INTENT

Can the source-only R11 observation condition, already evaluated in W104, improve **actual rig animation fidelity and usability** compared with the unchanged M_Main condition, on the same fixed frames and rig, while preserving all negative examples and distinguishing source-pose error from retargeting artifacts?

## HISTORICAL BASELINE

Preserve:
- W102 B_Main/M_Main observations, predictions, same-person association accounting and historical real clip; W102 BaseRig rest, calibration, FK/IK, cameras, zero-twist local policy and rendered cases.
- W103 source/GT validity sensitivity, well/partial/mislocalized groups, severe/knee/depth failures, qualitative identities and frozen HMR contextual outputs.
- W104 F00/F10/F01/F11, R11/R10/R01/Rc, privileged P_oracle/P_bench, predeclared declaration hash `6a64c8d1ffc57f928fdf2fbaaf40dffd5b848a927b662e90ff584257bfbae487`, and the final CASE A report.
- Original RGB bytes, actor IDs, timestamps, camera and coordinate contracts, S0 weights, model-visible tensor construction, M1 batch_size=1 and production 0.3 validity threshold.
- Original GT-free runtime boundary: GT, benchmark identities and oracle projections are evaluation-only.
- W99A production batch_size=8 default and W100 M8 parity failure.
- No historical test, Worklog, source artifact, model or rig may be rewritten.

W104's raw outputs and manifest are stored locally on Main, not committed. Verify their availability and full recorded hashes before relying on them. If unavailable, reproduce only the fixed W104 conditions needed for this Batch using its published scripts and declared identity, in a fresh non-overwriting output directory, and report the additional run explicitly. Do not fabricate evidence or silently replace an artifact.

## PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE: Frozen M_Main and R11 source geometries, actor/frame selection semantics, exact timestamps, fixed BaseRig/skinned equivalent, W102 rest and FK/IK/zero-twist policy, camera, renderer, animation interpolation, and identical per-case rig configuration.

CHANGE ONLY: New isolated W105 diagnostic/evaluation scripts, focused tests, fresh evidence manifests, matched rig exports/renders and `docs/105_WORKLOG_SOURCE_ONLY_RESTORATION_RIG_UTILITY.md`. All edits must be diagnostic-only. No `src/` production module edits. Do not modify `docs/next_batch.md` or existing W102–W104 files.

DO NOT TUNE: Joint visibility threshold, raw coordinate choice, crop, confidence, association threshold, S0 settings, temporal smoothing, motion interpolation, FK/IK reach, target bone lengths, pelvis/thorax ownership, root motion, contact policy, cameras, material/render settings, or selection/acceptance thresholds after viewing results.

## 1. PREFLIGHT, DATA AND DECLARATION

Verify W104's remote COMPLETE marker and the instruction/source/checkpoint/bank identities. Reconstruct or verify byte identity for the M_Main and R11 geometries on all 1,223 naturally partial same-person rows and the unchanged complete-row controls. Verify no GT is consulted by R11 input construction. Record source image hashes, source videos, actor identities, original frame numbers and timestamps. Keep VAL and TEST separate, and report the entire W102 actor-frame/identity denominator; never turn same-person conditional quality into whole-video success.

Before viewing new rig-level results, create a hashed declaration recording:
- Eligible populations and exclusion reasons (missing source, ambiguous identity, invalid GT camera, no rig chain, no original neighbor).
- Exact rig, rest, solver, renderer and timing configuration.
- Deterministic case selection, random seed and sampling frame/sequence caps.
- Metric definitions, evaluation-only GT access and uncertainty clustering.
- A fixed interpretation contract for improvement, harm and unresolved cases.
- Known W102 source-to-rig length-ratio limitations and their handling.
Do not select only large W104 improvements. Include both improvement and harm strata, natural missing knees/ankles and wrists/elbows, complete-row no-op controls, high-motion segments, multi-actor/identity caveats, and at least one severe failure with existing BaseRig context. W104's already reported example frames are illustrative, not independent holdout evidence.

## 2. PAIRED SOURCE-TO-RIG CONTRACT

Use the existing W102 BaseRig pipeline and, where possible, `scripts/w102_rig_review.py` with the existing skinned rig, rest snapshot, calibration, chain mapping and Blender executable. Only the 3D source pose is permitted to change between M_Main and R11. Preserve any pre-existing mixed-source/J1X arm policy exactly; do not accidentally present a frozen external arm track as newly recovered by S0. Report which bones actually consume S0 geometry.

Verify:
- Same actor, original frame index and timestamp on both conditions.
- Identical target rig, rest and per-bone length.
- Identical root/pelvis/thorax and zero-twist/IK policies.
- Identical animation sampling and interpolation.
- Identical camera, mesh, lights and viewport.
- Reproducible Action/bone names, keyframe times and animation export identity.
- Unchanged complete frames are byte/numerically identical where the path should be a no-op.
- Missing/invalid source samples are not silently bridged or dropped.
- No privileged GT, oracle or benchmark observation enters the exported candidate.

Do not claim an end-to-end production pipeline exists because evaluation-only Blender outputs render.

## 3. PRIMARY RIG-LEVEL UTILITY COMPARISON

On predeclared natural PARTIAL cases, compare M_Main vs R11 through the same rig. Report, per split and anatomical stratum:
- Source-level root-relative 3D MPJPE and per-joint/limb errors against the same evaluation-only GT.
- Source-level knee branch, limb direction, torso and depth-sign error changes.
- Rig bone direction and relevant joint-angle changes relative to the corresponding source and, where defensible, GT geometry mapped through the same fixed rig reference.
- Source-to-rig discrepancy (angular, reach and length-ratio effects), separating improved source geometry from unchanged IK artifacts.
- Invalid or degenerate rig poses, extreme knee/elbow folds, bone flips, foot/hand endpoint behavior, and original-timestamp temporal events.
- Relevant contact/foot-slide *proxies* only if computable from existing inputs; no unmeasured contact GT or world-trajectory accuracy claim.
- Action/keyframe editability of the diagnostic export (inspectable animation curves and timing, not a claim of style-authoring capability).
- Material regressions and severe-tail distributions, not just average improvement.

Use existing W102 event definitions (for example source/rig direction steps >45 degrees and knee-branch rules) where compatible. Declare any additional rig-specific threshold from existing W95/W102 contracts or physical rig geometry before evaluation; do not threshold-sweep to force a favorable verdict.

Treat GT-referenced 3D and perceptual animation quality as separate evidence. A good MPJPE without better rig motion is not a pass.

## 4. TEMPORAL AND IDENTITY BOUNDARIES

Compare consecutive original same-person observations without crossing actor, sequence, missing or ambiguous boundaries. Preserve the 3DPW bank's sparse 10 Hz sampling: do not present held/interpolated 30/50 fps playback as measured high-frequency reconstruction.

Quantify whether restoration causes or removes abrupt limb/torso direction steps, knee-branch changes, and frame-to-frame discontinuities at natural mask transitions (valid -> invalid -> valid). Distinguish a stable but wrong pose from temporal jitter.

Keep identity/coverage failures visible in the full denominator. Do not fill another actor's missing joints or infer association from GT at runtime. The original W102 selected-subject limitation is unchanged.

If temporal context suggests a possible solution, document evidence only. Do not implement propagation, filtering, prediction or hold policies.

## 5. REAL-VIDEO QUALITATIVE EVIDENCE

Use `examples/production_demo/source.mp4` only as the previously authorized historical real-video diagnostic. Source URL and redistribution rights are undocumented; do not redistribute or replace the source video, and do not call it unseen. Verify its actual content/visibility and W102's 199-frame/50-fps provenance before use.

Produce matched GT-free M_Main and R11 rig render(s) or viewable local side-by-side animation where technically feasible, preserving the exact original timestamps and fixed rig. The clip's naturally missing observations are left wrist only; explicitly state that it does **not** validate low-confidence leg recovery. Evaluate whether the arm-side change is visually useful or harmful, temporal behavior and whether the output remains editable. No GT 3D accuracy may be claimed.

For naturally missing legs, use selected 3DPW source images and corresponding fixed-rig renders with GT in a separate labelled evaluation-only panel. Keep the image/rig camera distinction explicit. If Blender or render dependencies fail, preserve the blocker; do not patch the solver or change cameras to make a candidate look better.

Do not publish external media or create new data-acquisition tasks.

## 6. HARM, ALTERNATIVE EXPLANATIONS AND TRANSFER LIMITS

Inspect *all* natural R11 harm cases (including the reported `outdoors_parcours_00` frame 1018) and severe/new-wrong-knee cases, not only the largest gains. Stratify by restored joint type, raw confidence, number of restored tokens, pose/occlusion, image location, and observed 2D error where evaluation GT is valid.

Distinguish:
A. Raw 2D localization failure / low-confidence hallucination.
B. Frozen S0 response to validity or input geometry.
C. Crop change (already controlled in W104; do not repeat the full factorial).
D. Source-to-rig/IK limitation.
E. Identity/coverage uncertainty.
F. GT, annotation or camera validity.
G. Unresolved visual ambiguity.

Do not claim low-confidence coordinates are universally trustworthy. W104 found only 42.5% PCK@0.10 among restored limb tokens and 16–21% for ankles, yet an average 3D gain relative to folded missing-limb predictions. Report that distinction and any observed rig regressions.

Use VAL/TEST as reused diagnostic data; neither is an untouched holdout. Do not infer population-wide or cross-detector generalization from the 199-frame clip or 33-video paired subset. No new detector run, training or alternative method is authorized.

## 7. INTERPRETATION AND DECISION CASES

Predeclare the operational metrics and any bounded harm thresholds before viewing new rig results. Use video-clustered uncertainty where meaningful; do not treat many frames of one video as independent.

**CASE A — Downstream animation utility supported (conditional):** Paired rig evidence on naturally partial same-person inputs shows material improvement in anatomical fidelity or animation usability from fixed R11, with tolerable measured harms and no hidden changes to rig or temporal policy. Explicitly scope by observed joints/videos and note that production adoption is still unauthorized.

**CASE B — Source gain does not translate:** R11 improves source-level reconstruction but fixed-rig output is not materially better, or meaningful rig/temporal harms offset the benefit. Attribute whether the limiting mechanism is source, IK, or representation.

**CASE C — Mixed or unresolved:** Improvement is confined to some joints/scenes, real-video quality is indeterminate, or source/rig/identity mechanisms cannot be separated. Preserve negative evidence and report the limits.

**CASE D — Contract blocked:** Required source identity, GT separation, rig equivalence, original timestamps, provenance or reproducible evaluation cannot be established. Do not manufacture a qualitative PASS.

The CASE is an evidence verdict, **not** a decision to change the production observation policy. No CASE authorizes a threshold change, default change, model replacement, retargeting fix, training, production integration or a new downstream subsystem.

## 8. TESTING

Run focused synthetic and stored-real tests for:
- W104 source geometry and input identity; M/R11 no-op on complete frames.
- GT-free construction and correct evaluation-only GT isolation.
- Same-actor/frame/timestamp and split alignment.
- Rig/rest/chain/camera equality between conditions.
- Source-to-rig angular metrics and length-ratio handling.
- Knee/limb/temporal events including degenerate and missing cases.
- Animation Action/keyframe identity and frame spacing.
- No hidden interpolation across invalid samples.
- Case selection, uncertainty and denominator accounting.
- Renderer exit codes and output manifests when Blender runs.

Execute relevant W104/W102 focused regression tests if the new diagnostic imports shared helpers. Broad production regression is not required when no shared production contract changes. Report exactly which tests ran, failed, were fixed, or were skipped. No test or render may be reported PASS unless executed.

## STOP CONDITIONS

STOP and report evidence-bounded CASE D or a narrower verdict if:
- The frozen M/R11 artifact or original source identity cannot be established.
- Required GT-free/evaluation-only separation fails.
- Rig equivalence cannot be maintained without modifying historical solvers/assets.
- Actor/frame/time alignment is ambiguous.
- New results require adjusting masks, threshold, S0, IK or render settings.
- The real-video clip cannot be used without unauthorized redistribution or data changes.
- Another project or active Agent job would be disrupted.
- Git publication or automation lock ownership is unsafe.
- A Major Architecture Decision or user-only authorization is needed.
- The central animation-utility question has received sufficient supported positive, negative or mixed evidence.

Do not automatically expand to training, a new dataset, production integration, temporal modeling, root/IK repair or style authoring. A valid negative or blocked outcome is completion.

## FINAL REPORT

Create `docs/105_WORKLOG_SOURCE_ONLY_RESTORATION_RIG_UTILITY.md` with:
- Starting HEAD, branch, worktree and instruction identity.
- Verified W102–W104 artifact hashes and any necessary deterministic reproduction.
- Predeclared case/metric contract and selection manifest.
- Full population and selected-rig denominators.
- Same-source/same-rig invariants and Blender/Action provenance.
- M_Main vs R11 source and rig metrics, uncertainty, temporal and harm distributions.
- Representative improvement, failure, and regression renders with exact provenance.
- Real-video GT-free quality judgment and its explicit limits.
- Attribution among observation, S0, retargeting, identity and reference errors.
- Executed and omitted tests; resources, changed files and exact reproduction commands.
- CASE A/B/C/D with negative and unresolved findings.

Do NOT include Next Work Recommendation, Next Architecture Decision, Recommended Next Experiment, Proposed Next Batch or Next Implementation Plan. Do not copy this instruction into the Worklog.

Follow the existing automation worker's final publication protocol. Add `AUTOMATION_BATCH_STATUS: COMPLETE` and `BATCH_NUMBER: 105` only after all authorized artifacts and Worklog are finalized and safely committed. Preserve any failed/interrupted state without a false COMPLETE marker.

## COMPLETION CONDITION

The Batch is complete when paired, reproducible evidence answers:

**"Does the fixed W104 source-only raw-joint restoration improve actual rig animation under the existing AnimCV BaseRig and temporal contracts, or are its 3D gains undermined by retargeting, low-confidence localization, temporal artifacts or identity/coverage limitations?"**

A supported negative, mixed, or blocked outcome is valid completion. Do not implement the method, alter production defaults, choose the next Architecture direction, or initiate downstream work.