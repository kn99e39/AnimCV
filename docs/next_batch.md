# BATCH 115 — Animation Base Validity: Repeatability, Numerical Resolution, and Rig Authority

DIRECTION
Continue the user-approved reference-guided Video-to-Animation Generator architecture program through ONE bounded evidence-only Batch. Re-evaluate W114's immutable-evidence and versioned-edit-state hypothesis using prospectively corrected measurement contracts. Do NOT continue FBX repairs, optimize Pose, train a Generator, or modify Production.

PURPOSE
W114 returned CASE D INVALID for two distinct experimental-instrument errors. V6 treated a build-specific legacy MotionGraph pose_source path and its byte hash as semantic differences. G8 compared source-to-rig outputs with 1e-9 precision even though the candidate and Blender outputs used float32. The W114 C1 trunk-axis case had differences below representable Action precision. The earlier declared gates cannot be retrospectively changed. New, predeclared tests are necessary to determine whether the abstraction works under valid representation-aware controls.

W114 also established a distinct uncertainty: "preserve" may mean local channel values or world-space positions of posed joints. This is a potential human product decision. W115 MUST compare both meanings without silently adopting either as the Production rule.

CENTRAL INTENT
Can a source-preserving Animation Base with immutable evidence and reversible editable states satisfy its bounded integrity, provenance, repeatability, user-edit and actual rig-influence contracts when semantic build-path volatility and numerical storage resolution are handled rigorously, while rejecting genuine identity changes and explicitly preserving unresolved user-intent semantics?

Do not merely make the prior test pass. Test whether the abstraction, not the measurement instrument, is responsible for failures.

CURRENT ASSESSMENT
- W113 CASE D INVALID remains historical: detector score is not a bounded [0,1] probability and cannot be directly treated as 3D reliability.
- W114 CASE D INVALID remains historical. Frozen V6 and G8 failed; W114 post-hoc checks are diagnostic only and do not supply a retroactive CASE.
- W114 agent-reported results: real and benchmark observation views, reversible local mock edits, constraints, source digests, and representative source-to-rig influence behaved usefully under limited controls. The real clip contains 199 frames at 50 fps and only 38 low-evidence left-wrist cells. Blender Action is 30 fps. Per-frame same-person identity was not serialized. The tested W105 rig has real_s0_only and mixed_w102 variants with different arm ownership.
- W114 reported C1 float32 transverse residual up to 1.62e-8m, whereas a float64 exact-direction diagnostic was invariant to approximately 3.4e-15. This does not authorize changing Production computation or choosing an arbitrary tolerance.
- W114's world-space versus local-channel preservation semantics remain undecided.

HUMAN-READABLE SUMMARY
Direction: predeclared instrument-validity and edit-authority architecture diagnostic.
Purpose: independently test the same source-preserving representation without two known measurement defects.
Central Intent: is observed success attributable to real source/edit/rig contracts rather than path normalization, precision artifacts or overly weak user constraints?
DO: verify historical baselines; classify semantic and volatile path fields; run adversarial provenance controls; compare rig outputs in their actual storage representation; test local and world-space edit effects separately; replay existing real/benchmark/synthetic cases; report a bounded verdict.
DO NOT: patch frozen W114, sweep thresholds, choose a Production lock policy, change source models/rig/schema/exporters, or advance to W116.
Decision enabled: whether a minimum trustworthy Animation Base representation is supported, partially supported, contradicted or still invalid, and which evidence boundaries remain unknown.

HISTORICAL BASELINE
Repo: kn99e39/AnimCV. Existing branch ONLY: arch/single_frame_first.
Previously observed HEAD: 380a0e446c08912e2104cc2d1c2cb1fa2d1f0fae. Verify current remote HEAD before working. W114 is COMPLETE, and its original next_batch instruction and Worklog must remain unchanged.
Read AGENTS.md, PRODUCT_VISION.md, Architecture_v3_FramePose.md and Architecture_v2.md with declared precedence; W113/W114 Worklogs, frozen W113/W114 instruments, tests, manifests and W102-W105 real/rig baselines. Verify SHA-based provenance locally. Do not claim local binaries were checked merely because Worklog hashes are available.
No branch creation/switch/deletion, merge/rebase/reset/force operation. Preserve all historical Worklogs, CASEs, checkpoints, datasets, seed, source images, predictions, numerical and source contracts.
PRESERVE: all src/ Production modules, MotionGraph/AnimationClip schema, detector scores and validity policy, source video FPS and timecodes, frozen S0/HMR2/J1X, W105 rig and FK/IK, calibrated/uncalibrated labels, exporter/importer, W113/W114 scripts and tests.
CHANGE ONLY: new W115 scratch scripts, focused tests, evidence artifacts and W115 Worklog.
DO NOT TUNE: masks, thresholds, pose model, rig configuration, 0.01m controlled perturbations, animation timing, comparison gates after results, smoothing, rendering, inference, retargeting or confidence interpretation.
The Agent controls technical implementation details within these fixed evidence requirements, with new test parameters declared before outcomes.

REQUIRED EVIDENCE — 1. PROSPECTIVE CONTRACT
Inspect W114's V6 and G8 implementation and determine exactly why their declared premises fail. Before new real/rig outcomes, serialize and hash a W115 declaration specifying:
- identical input datasets, frame/joint/actor indices, model hashes, case IDs, perturbation magnitudes and output formats;
- positive controls, negative controls and exact expected outcomes;
- explicit allowlist of build-generated volatile fields and their causal byte dependencies;
- semantic comparison method for stable source identities and content;
- numeric dtypes, casting points, equality rules and output-resolution account;
- preservation interpretations and what counts as an unresolved human policy choice;
- exact gates for CASE A/B/C/D and STOP conditions.
Do not use GT-derived targets, GT stratification or evaluation identity labels to construct runtime inputs, edit masks or authorization.

REQUIRED EVIDENCE — 2. PATH SEMANTICS AND REPEATABILITY
Run two independent scratch builds under distinct output paths and repeated same-input controls. Report separately:
(A) raw file-byte identity;
(B) equality of semantic source, observations, hypotheses, edit lineage and rig outputs;
(C) declared build-path-dependent differences, including legacy MotionGraph pose_source and its resulting file SHA.
Only normalize fields shown before results to reference the build directory rather than the actual input identity. Never normalize away a genuine source digest, image identity, model identity, input path when identity-bearing, frame ordering, timestamps, joint names or user constraint.
Adversarial negatives must include:
- same apparent filename with different real bytes;
- changed source or frame digest;
- changed authentic identity-bearing source path;
- stale or mismatched content hash;
- modified frame timestamp/order/joint;
- modified edit parent or permission digest;
- substantive MotionGraph/AnimationClip content change;
- a path resembling a generated path that is actually source evidence.
A canonicalized comparator that falsely accepts any true source mutation cannot pass. Report how path normalization was limited; do not sweep alternate allowlists after seeing failures.

REQUIRED EVIDENCE — 3. REPRESENTATION-RESOLUTION RIG TEST
Preserve the original W114 rig conditions A1/A2, B1/B2, C1, D1/D2, RS1/RS1_arm and mixed MA/MB/MD controls where applicable. Keep W114 interventions unchanged.
Inspect the actual source float32 -> rig computation -> quaternion -> Action F-curve storage stages. Prospectively define comparisons based on actual storage semantics rather than choosing a passing numerical tolerance. Prefer exact comparisons AFTER reproducing the documented representation casting, and report pre-cast and post-cast continuous deltas separately.
For C1 include original float32 candidate behavior plus an explicitly diagnostic float64 exact-direction control, without changing the Production rig. Separate:
- analytically zero/invariant behavior when supported;
- nonzero but below output resolution;
- actual material change in stored Action values;
- missing or contradictory attribution evidence.
Measure changed channel-frame counts at source/case/Action layers, float32 representation deltas, physical/angular magnitudes and independent ownership derivation. Include unconsumed-source no-effect controls and meaningful positive source-backed influence. Do NOT infer global dependency or derivative from one perturbation, and do NOT erase small-but-nonzero observations.
If numerical equivalence cannot be defined without observing W115 results, STOP as invalid.

REQUIRED EVIDENCE — 4. EDIT AUTHORITY VS USER INTENT
Use W114's unchanged scratch candidate, source evidence, version lineage and deterministic mock edits to verify:
- immutable source and original score >1 preservation;
- provenance, parent digest, stale-state refusal, atomicity and exact revert;
- permitted vs preserved vs prohibited joint/frame cells;
- actual source owners and output channels under real_s0_only and mixed_w102;
- correct negative control where S0 source changes do not drive the mixed W102 arm;
- actual wrist-to-upperarm plus lowerarm effect and torso/shared influence;
- output changes outside the local channel domain and outside world-space posed-joint preservation.

Evaluate TWO explicit, separate interpretations: local Bone/F-curve value preservation, and posed world-space joint-position preservation. Report which is satisfied, violated or unknowable for each representative mock edit. Do NOT choose which semantics the user intends as a permanent Production rule. Do not label a locally allowed edit a fully user-preserving edit if it moves a world-space locked joint.

Unverified per-frame subject identity must stay UNKNOWN and cross-frame edits requiring verified identity must fail closed. The 50fps source vs 30fps Action mismatch must remain explicit; sidecar-only recovery is not proof that Action/FBX preserves time.

REQUIRED EVIDENCE — 5. POPULATIONS AND CONTROLS
Keep results separate:
- real_animcv_observation: authentic GT-free 199-frame W102 clip, 38 low-confidence left-wrist cells;
- benchmark_detector_observation: W113 3DPW 319-row sequence, not untouched holdout;
- real observation mixed with frozen HMR2/J1X/S0 windows for rig attribution;
- synthetic oracle_geometry: integrity, edit, precision, path, identity and negative controls.
Keep W114's fixed source input, predictions, constraints and perturbations. Synthetic edits are NOT recovered animations.
Report exact numerator/denominator accounting for observed, derived, valid, missing, unknown; scores >1; authored and reverted edit cells; output channel-frames; false accepted mutation controls; source identity verified/unknown; eligible and rejected cross-frame operations; local/world lock outcomes; and real source vs Action time mapping.
Provide qualitative before/after on real concrete frames, including C1, RS1/RS1_arm and a mixed arm negative control, with authentic source/target identity and provenance.
Distinguish synthetic contract correctness, authentic replay, causal rig influence, practical animation quality and Architecture viability. Do NOT claim the latter two from successful mocks.

TESTING
Focused tests for semantic canonicalization and adversarial source changes, correct independent repeat, numeric comparator at casting boundary, error controls, edit lineage and rollback, local/world authority divergence, real/benchmark replay and frozen-rig source influence. Run appropriate focused legacy regressions; full suite optional for diagnostic-only work without Production changes. Report precise commands, PASSED/FAILED/NOT RUN counts. Preserve failed baselines. The W115 reviewer audit of cumulative W111-W115 code changes will be performed by the Architecture Reviewer AFTER W115 Worklog submission, not as an Agent recommendation.

STOP CONDITIONS
Stop and preserve evidence if required artifacts fail authentication, canonicalization hides a real source mutation, numerical attribution requires an outcome-fitted tolerance, an ambiguous user lock policy must be decided to continue, source-to-rig effects cannot be separated under the frozen pipeline, or any Production/schema/model/rig/exporter modification is necessary. Do not work around invalidity by changing gates, tuning parameters or implementing heuristics.

VERDICT CONTRACT
Prospectively specify non-vacuous A/B/C/D predicates. Use these interpretations:
A — BOUNDED SUPPORT only for demonstrated immutable evidence, authoritative lineage, repeatability under declared path semantics, adversarial rejection, correct edit/reversal and storage-aware local rig influence within an explicit LIMITED authority domain. Does not validate world-space preserve, real occlusion recovery, Generator quality or Production readiness.
B — PARTIAL SUPPORT if valid controls support some guarantees while a material local/world authority or downstream identity/time guarantee remains unresolved or violated. Do not downgrade a necessary declared gate after outcomes.
C — NOT SUPPORTED if valid controlled results contradict the candidate's minimum claimed guarantees.
D — INVALID if source, numerical or repeatability control fails to establish a trustworthy verdict.
D takes precedence. No retroactive W114 reclassification; W114 post-hoc remains post-hoc.

FINAL REPORT
Write W115 Worklog covering official verdict, claims separated by evidence level, exact baseline hashes, declared hypotheses and controls, path canonicalization with negative tests, numeric precision/rig attribution, both lock semantics, real-vs-synthetic-vs-benchmark accounting, reproduction, exact changed files, tests, uncertainties, failure examples and alternate explanations.
Do NOT claim architecture success automatically. Report negative results honestly.
Do NOT ask for or include Next Work Recommendation / Next Architecture Decision / Recommended Next Experiment / Proposed Next Batch / Next Implementation Plan. Do not start W116.

COMPLETION CONDITION
Conclude only when the evidence can determine whether:
1. stable source provenance survives controlled independent builds without false equivalence under adversarial changes;
2. float32/Blender output semantics explain W114 C1 without outcome-driven tolerance tuning;
3. reversible edits and actual source-to-rig effects hold within the declared authority domain;
4. local-channel versus world-space user-preservation requirements can be kept separately assessable without silently choosing the Production policy;
5. the minimum Animation Base representation is boundedly viable, partially viable, contradicted or still untestable.

Negative/invalid outcomes count as completed evidence if honestly justified. Stop before any Production integration, trained Generator or Batch 116.
