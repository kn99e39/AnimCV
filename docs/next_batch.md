# BATCH 104 — Missing-Joint Causality and Recoverability in Frozen S0

## DIRECTION

Continue AnimCV on the existing `arch/single_frame_first` branch only. Verify the remote HEAD, clean/safe execution workspace, prior Worklog completion and current instruction blob before any work. Never create, switch, merge, rebase, rename, delete or reset branches; never force-push or resolve unrelated conflicts.

Read `AGENTS.md`, `PRODUCT_VISION.md`, `Architecture_v3_FramePose.md`, `COORDINATE_CONVENTIONS.md`, the cumulative architecture/codebase audit and Worklogs 97–103 (including 99A). Treat Worklogs as evidence, not as automatic proof. Respect the automation worker's exclusive execution claim and the immutable, ChatGPT-owned `docs/next_batch.md`. Do not edit, delete, archive, reproduce or rewrite this instruction file.

This is an **evidence-only Estimator/Observation architecture diagnostic**, not a new model or runtime implementation. Execute on the authorized Main environment if the W102/W103 frozen artifacts and execution stack are available and do not interfere with other work. Do not silently move to LabServer63/69, install dependencies, change precision or alter the production environment.

## PURPOSE

W102 established that actual M1 RTMDet/RTMPose observations can improve frozen S0 over benchmark 2D on correctly associated people. W103 finished with CASE C and established a much sharper distinction:

- Of 150 M1 severe (>200 mm) same-person frames, 133 have missing limb joints; 185/199 wrong knee-branch cases have a missing joint in the affected leg.
- Among 4,803 well-observed M1 rows, only 3 severe cases remain, and 1/4,051 eligible knees has a wrong branch.
- M1's gain over benchmark observations is concentrated where M1 supplies more valid joints: M−B MPJPE ≈ −46.5 mm on 2,457 such rows, versus +0.28 mm on 4,083 identical-mask rows.
- W102's Mmask probe changed masks and sometimes crop geometry together. W103's crop-nearly-unchanged subset still lost 35.9 mm when M-only joints were removed, but this is not a randomized or full factorial causal intervention.
- On well-observed rows, limb depth-sign errors remain; W103's 2D-neighbor attribution is mixed and mostly unresolved. HMR2's better results are contextual, not proof of an RGB causal effect.
- W103 identified a GT root-behind-camera defect in one 3DPW sequence. Its primary result and reference-validity sensitivity are both preserved; do not silently erase either.

The next architecture uncertainty is **not** whether missingness correlates with failure. It is whether the absence of specific model-visible joints *causally produces* the observed failures, whether changes in the valid-joint crop amplify the effect, and what usable evidence exists to recover those joints without privileged inference inputs.

The product objective remains editable, controllable rig animation. A better MPJPE alone does not establish usable animation, and this Batch must not become an endless reconstruction-only optimization loop.

## CENTRAL INTENT

**When frozen S0 receives the same real M1 pose with predeclared joint evidence selectively removed, does the removal itself cause severe 3D/knee failures, independently of crop normalization? For naturally missing joints, how often does existing, GT-free observation evidence actually contain recoverable information, and where is new information or a different architecture likely necessary?**

The answer must separate: (1) causal sensitivity to missing tokens, (2) normalization/crop interaction, (3) reliability of currently suppressed raw 2D detections, (4) temporal availability of independently observed joints, and (5) failures not explained by these mechanisms.

No conclusion in this Batch authorizes adoption of a new mask rule, temporal reconstruction, visual model, or runtime fallback.

## CURRENT ASSESSMENT — SUCCESS / FAILURE / UNCERTAINTY

Succeeded: W102 same-stack B/M comparison and W103 input-tensor reproduction, missingness stratification, video-aware statistics and reference-validity sensitivity.

Failed or limited: missing-joint frames still exhibit catastrophic errors; the W102 mask intervention is partly confounded by crop; same-person evaluation excludes identity failures; real-video/rig usability remains unproven. Most complete-input depth-sign errors have no comparable independent 2D neighbor.

Uncertain: whether single-joint removal is sufficient to induce a knee/severe error, whether anatomical pattern or number of removed joints matters more, how much crop normalization contributes, whether low-confidence raw M1 joints are useful or dangerous, and whether neighboring real frames contain the missing evidence.

Why this Batch: it can reduce these uncertainties **without a Major Architecture Decision** by testing fixed S0 with isolated, evaluation-only counterfactual inputs. Architecture selection remains with the human Reviewer.

## HISTORICAL BASELINE

Preserve all Worklogs 95–103 and their original verdicts, artifacts, hashes, data splits, image bytes, actor IDs, camera/coordinate conventions, checkpoints, bank, W102 B_Main/M_Main/Mmask/Mconf/Moa, W103 input and similarity declarations, W97/W98 external HMR2 evidence, W99A batch defaults and existing BaseRig/IK/Blender outputs.

The original W103 CASE C is not rewritten. Its invalid-root reference sensitivity must remain separately identifiable. The same-person 3DPW population is repeatedly used for architecture diagnosis and is **not an untouched holdout**.

## PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE: frozen S0 weights, exact architecture, confidence and mask policy (including current 0.3 rule), production observation adapter, checkpoint/bank, M1 source data, crop and derived-joint contracts, execution precision/batching, historical metrics and artifacts, rig contracts and default behavior.

CHANGE ONLY: isolated W104 diagnostic scripts, controlled counterfactual input tensors, new evidence/figures/manifests, focused tests, evaluation-only fixed-rig comparisons when justified, and the single new `docs/104_WORKLOG_MISSING_JOINT_CAUSALITY_AND_RECOVERABILITY.md`.

DO NOT TUNE: S0/MMPose/HMR2/J1X weights; detector thresholds; validity/confidence policy; camera mapping; crop rule; skeleton; IK/retargeting; smoothing; interpolation; missing-frame policy; model fusion; runtime filtering. No production `src/` changes. Do not create or modify `docs/next_batch.md`, historical Worklogs, previous outputs or automation infrastructure.

## 1. PREFLIGHT AND DECLARATION

Verify artifact identities from W102/W103, current source and frozen model hashes, raw M1 observation provenance, same-person association, image/timestamp alignment, and availability of the authorized Main inference environment. If the source observations needed for a probe are not retained, explicitly mark that probe unavailable rather than manufacturing them.

Before looking at new S0 outcomes, persist a declaration containing: population, sampling seed, target joint sets, mask patterns, confidence policy, crop-factorial construction, control groups, statistical units, outcome metrics, qualitative selection rules and interpretation gates. Hash this declaration. Its construction may inspect source 2D inputs and W103 missingness frequencies, but must not use S0 error or GT to select favorable intervention patterns.

Use the W102/W103 same-person populations for paired accuracy, preserving the complete actor-frame and association denominators separately. Stratify by split, source video, actor, posture, visibility, limb and baseline error; never filter failures out after results are seen. Use the physically invalid GT-root sequence as a separately reported sensitivity rather than silently deleting it.

## 2. CONTROLLED JOINT-REMOVAL EXPERIMENT

Use **real M1 well-observed input frames** as the intervention substrate. Construct deterministic counterfactual missingness from the original valid 2D observations without using GT to form the candidate.

Predeclare representative patterns from the observed W103 missingness distribution, including single-joint hip/knee/ankle and shoulder/elbow/wrist removal, same-limb multi-joint loss, bilateral loss, and count-matched unrelated-joint negative controls. Keep each sample's original 2D coordinates and confidence unchanged except where the declared intervention explicitly removes evidence.

Implement and verify a **2 × 2 factorial** for each eligible sample/pattern:
- F00: original valid joints and original crop (frozen M_Main control).
- F10: intervention mask with the **original fixed crop** (mask-only diagnostic).
- F01: original validity but crop recomputed as though the selected joints were missing (crop-only diagnostic).
- F11: intervention mask and production-consistent recrop (combined missingness diagnostic).

F01/F10 are **counterfactual diagnostics, not valid production observations**. All conditions must honor canonical joint identity, derived midpoint rules, neutral sign state and exact S0 input shape; record any pattern for which a well-defined factorial cannot be constructed. No GT, target 3D or HMR2 output may enter S0 inputs. Verify F00 reproduces the stored M_Main prediction to the declared numerical budget before interpreting changes.

Analyze main effects and interactions rather than attributing F11−F00 entirely to the mask. Distinguish the number of removed joints, their anatomical location and crop changes. Count newly induced failures and newly improved cases, not only average errors. If large crop shifts or degenerate masks make a condition uninterpretable, retain the denominator and label it as such.

## 3. NATURALLY MISSING JOINTS: RECOVERABILITY BOUNDARY

On the existing naturally partial M1 rows, distinguish **missing observation** from **discarded but finite raw keypoint prediction**. Audit what was actually retained upstream of the validity policy. Record pixel bounds, confidence, finite values, missing-joint identity and source hashes.

Where raw M1 below-threshold coordinates are available, construct an evaluation-only diagnostic input that restores them under a **fixed, predeclared rule based solely on source availability**. Do not search confidence thresholds or select restored joints based on GT accuracy or predicted 3D improvement. Keep the original production M_Main condition unchanged. Separate the effect of newly added joint coordinates, confidence values and crop changes when the available evidence supports it. Compare the restored 2D coordinates to GT **after** inference, with the COCO/SMPL semantic mismatch acknowledged.

Use existing benchmark observations and, if coordinate semantics and camera validity permit, GT projections as **explicitly privileged reference/upper-bound counterfactuals** only. They are not GT-free candidates and cannot be used to claim deployment feasibility. Report the exact missingness and support conditions under which each comparison is meaningful. Do not mix oracle/benchmark results with real M1 outcomes in one success rate.

For neighboring real M1 frames, audit how often a missing joint is independently observed before/after the target frame in the **same verified person track**, at original timestamps. Characterize gaps, motion, occlusion and association uncertainty. This is a measurement of **potential temporal evidence**, not an interpolation/propagation implementation. Do not fill missing joints using GT or future frames and then call the result a valid single-frame estimator.

If the retained M1 artifacts do not expose pre-threshold coordinates, report that limitation and complete the factorial and temporal availability portions that remain valid.

## 4. OUTCOMES AND CAUSAL ACCOUNTING

Use fixed W102/W103 metric definitions:
- Root-relative MPJPE and per-joint/limb errors.
- Severe MPJPE > 200 mm and severe-tail quantiles.
- Knee branch, bend eligibility, limb directions and depth-sign errors.
- Torso orientation and original-neighbor temporal events.
- Validity/crop shifts and model-visible tensor differences.
- Changed and newly introduced failures, recovery and collateral harm.
- Actual coverage and association denominators (not accuracy only on surviving rows).

Report effect sizes as paired changes on the **same sample**, video-clustered uncertainty and sequence-level failure clusters. Avoid treating counterfactual copies of one frame as independent observations. Keep valid, partial, mislocalized, unresolved-association and reference-invalid populations distinct.

The factorial is an intervention on model inputs; it can establish S0's **input sensitivity**, but cannot alone prove that naturally missing observations have the same cause or that a proposed production fix is safe. Restoring noisy coordinates is not necessarily beneficial, and a lower mean MPJPE may conceal worse tail failures.

Include no-change and count-matched negative controls. Check if effects reverse or amplify in difficult poses. Do not claim a universal depth-observability ceiling from the finite 3DPW population.

## 5. REAL-VIDEO AND RIG CONSEQUENCES

Where the stored W102 real-video diagnostic or W103/W102 qualitative assets are accessible, inspect representative GT-free motion and missingness events. Use existing rig and camera conventions for a bounded side-by-side **diagnostic** of a few predeclared windows when informative. Change only the source 3D pose between controlled conditions; preserve BaseRig, bone lengths, rest pose, FK/IK, timestamps, interpolation, renderer, root/pelvis/thorax and any shared arm source. Do not correct the estimator by changing retargeting.

Report whether the source differences are visible as folded legs, knee flips, invalid frames, discontinuities or unusable rig motion. Do not claim full-body animation success from the historical mixed-source rig, and do not invent GT accuracy for a real video. If the relevant real-video assets or Blender environment are absent, report the limitation rather than moving hosts or acquiring new data.

## 6. INTERPRETATION AND DECISION CASES

CASE A — Missing evidence is causally dominant and some **existing GT-free** evidence is demonstrably recoverable. Controlled removal causes substantial failures beyond crop-only controls, and a predeclared source-only restoration improves paired outcomes without hiding meaningful regressions. This supports further consideration of a bounded observation-handling architecture, not adoption.

CASE B — Missing evidence is causally dominant but **not reliably recoverable** from retained source-only observations. Controlled removal causes the failure, while raw low-confidence or temporal availability evidence is absent, unreliable or harmful. This supports the need to evaluate a genuinely additional evidence source, without selecting one.

CASE C — Mixed or non-missingness mechanisms. Mask and crop effects interact substantially, the naturally missing population is not explained by controlled removal, recovery introduces significant harm, or other failures dominate important use cases. Report the mechanisms separately.

CASE D — Diagnostic contract blocked. Artifact/provenance identity, same-stack reproduction, input validity, comparable populations or safe execution cannot be established.

Mixed outcomes may be reported by joint group, posture and failure type. A negative result or CASE D is a valid completed Batch. None authorizes production changes, new model training, a detector threshold change or an automatic next Batch.

## 7. TESTING

Run focused synthetic and stored-real fixtures for:
- Frozen checkpoint/bank/input artifact identities and prediction reproduction.
- Exact tensor semantics and derived-joint handling.
- Four factorial conditions, crop isolation and no-op equality.
- GT isolation of intervention construction.
- Deterministic source-only sampling and no outcome-driven selection.
- Missing and low-confidence raw M1 provenance.
- Same-person/sequence/time isolation.
- Counterfactual reference-regime labels.
- Joint, knee, depth and severe-tail metrics.
- Video-aware uncertainty and paired denominators.
- Degenerate/empty-mask refusal and invalid camera-reference sensitivity.
- Fixed-rig invariants if rendering is performed.

Do not run broad production regression unless a shared contract actually changed (none is authorized). Record failed attempts, skipped tests and resource usage. Do not call unexecuted tests PASS.

## STOP CONDITIONS

Stop if historical identity or same-stack reproducibility fails materially; the 2×2 factorial cannot isolate the intended variables; an intervention requires GT during real-source input construction; actor/temporal alignment is unsafe; meaningful comparison requires changing production code or thresholds; an unrelated process owns the execution resources; a Git conflict/branch operation would be required; the work would overwrite another session's files; or the central causal question has received a supported positive, negative or mixed answer.

Do not broaden into training, detector replacement, visual/RGB fusion, temporal inference implementation, rig repair, style subsystem development or a new Architecture proposal. Report the exact blocker.

## FINAL REPORT

Create only the new numbered Worklog:
`docs/104_WORKLOG_MISSING_JOINT_CAUSALITY_AND_RECOVERABILITY.md`.

Include starting branch/HEAD and working-tree state; W102/W103 identities; predeclared intervention manifest/hash; populations; F00/F10/F01/F11 construction and exact invariants; per-joint missingness effects; mask/crop interaction; natural missingness and low-confidence source-only recovery; privileged upper-bound controls separately; temporal evidence availability; GT-free video and rig consequences if available; video-level uncertainty; representative success/failure/harm; tests executed/skipped; reproduction commands; changed files; CASE A/B/C/D and limitations.

The Worklog is evidence-only. Do not include Next Work Recommendation, Next Architecture Decision, Recommended Next Experiment, Proposed Next Batch or Next Implementation Plan. Do not copy this Batch prompt into the Worklog.

Follow the automation worker's final publication protocol: publish the explicit COMPLETE marker for this Batch only after all authorized artifacts, tests and Worklog are finalized and safely committed. Do not mark COMPLETE when publication or execution is still uncertain.

## COMPLETION CONDITION

The Batch is complete when independently interpretable, paired evidence answers:

**"Does removing specific real M1 joints causally create frozen S0's severe and knee failures, how much of that effect is crop normalization, and can the naturally missing evidence be recovered from information actually available to a GT-free runtime without introducing unacceptable regressions?"**

A negative, mixed or blocked result is valid completion. Stop after publishing the evidence; do not implement a follow-on solution or select the next Architecture direction.
