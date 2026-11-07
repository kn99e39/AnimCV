# BATCH 122 — Real-Time Animation Utility of Frozen World-Pin Guidance on Independent-Video Motion

DIRECTION

Continue the approved AnimCV Video-to-Animation Generator/Editor architecture:
immutable source evidence, editable rig state, semantic WORLD_POSITION_PIN,
bounded visible automatic transition edits, independent Blender evaluation,
and exact accept/reject/revert.

This is a bounded, evidence-only test of ANIMATION UTILITY, not a solver
improvement or another attempt to retroactively repair a historical CASE.

Use the three authenticated dynamic source videos and frozen source-to-rig
pipeline from W121, with newly predeclared held-out WITHIN-SOURCE windows.
The sources are independent of W102/W105, but NOT new independent sources
relative to W121. State this distinction everywhere.

PURPOSE

W121 completed three-source acquisition and matched factor evaluation but
its official CASE D — INVALID is immutable: V10 compared nondeterministic
wall_s in repeat builds, and V12/DG5 applied a REJECT->ACCEPT display
corruption to an already ACCEPTed candidate. W121's post-hoc mechanical B
is NOT an official verdict and must never be adopted.

The W121 exploratory pattern is nevertheless decision-relevant:
- Four numeric guide improvements were clustered into two windows:
  S1_MID_RBneg and S2_LOW_{none,RBpos,RBneg}.
- There was no meaningful guide improvement in S3.
- In some cases a smaller SOURCE-RELATIVE compensation peak corresponded
  to a LARGER raw wrist step, because the guide returned toward fast,
  discontinuous unedited source motion.
- The agent's frame-crop review saw differences for 1–3 frames, but did
  NOT establish improved continuity; real-time playback was NOT reviewed.
- S1/S2/S3 source parents have reported raw wrist jumps of 0.877/0.628/
  0.228 m per frame, respectively. S2 is a crowded two-juggler source,
  and subject association remains UNKNOWN.
- Ten of 27 original evaluation requests were infeasible, especially
  every HIGH travel window. A source-relative metric alone does not
  establish useful animation or practical user intent satisfaction.

The next meaningful uncertainty is not whether the W121 timing and
display-control mistakes can be cosmetically corrected. It is whether
the frozen guide actually improves VIEWED REAL-TIME rig animation on
new, prospectively selected edits, rather than merely moving error
from edit-induced compensation back into source motion.

CENTRAL INTENT

On source-authenticated real motion, with frozen source-to-rig inference
and W120 factor solvers, does world-space wrist guidance yield a
reproducible, visually legible improvement in the real-time playback of
user-pinned Blender animation, WITHOUT increasing raw discontinuity,
articulation damage, hidden edits or hard-pin failure?

Disentangle:
(A) unedited source/Animation Base discontinuity,
(B) user-edit and pin-induced compensation,
(C) transition mechanism contribution, and
(D) subjective agent-visible animation quality.

The central question is about REAL-TIME ANIMATION UTILITY and
architecture ownership of source-vs-edit error, not the ability to
tune a wrist guide or relabel W121's CASE.

HUMAN ASSESSMENT / CURRENT ASSESSMENT

Established:
- User approved WORLD_POSITION_PIN, strict hard pins, separated source
  and editable states, bounded visible out-of-pin automatic editing,
  and atomic accept/reject/revert.
- W116 B established bounded pin feasibility; W117 C failed temporal
  utility; W118 D and W119 D had invalid instruments; W120 B isolated
  context-specific guide benefit in one historical clip.
- W121 D is complete. Three newly sourced, licensed dynamic videos
  and a frozen W102 M1/S0 -> W105 rig adapter were reported usable by
  the Agent, with byte-exact historical reproduction.
- W121's invalidity is explained by two identifiable predeclared
  instrument errors; its positive numeric patterns remain exploratory.
- W116–W120 cumulative audit was completed at 60f32c7.

Not established:
- Independent reviewer re-execution of W121's local Blender or video
  artifacts; source-to-performer 3D fidelity; verified S2 subject track;
  visually improved real-time animation; transfer beyond these three
  videos; a Production-safe quality threshold; Generator training.

Why another Batch matters:
- W121's numeric guide gains are not sufficient evidence of improved
  animation. A prospective held-out-episode real-time evaluation can
  discriminate editor usefulness from source-noise-limited results.
- The two W121 validity failures are preflight defects to avoid in the
  NEW experiment, not reasons to change W121 history.

HUMAN-READABLE SUMMARY

Direction:
Evaluate real-time, user-visible animation utility of frozen world-pin
transition guidance on existing independent-video sources.

Purpose:
Determine whether improvements in source-relative transition metrics
translate to visually better pinned animation when raw source jumps,
subject identity and articulation are accounted for.

Central Intent:
Do frozen G1-vs-G0 world-space guide contrasts produce real-time motion
quality improvements on newly selected within-source hold episodes,
with strict pin, authorization, visible footprint and undo guarantees?

Do:
- Verify W121 originals, source identities and frozen pipeline hashes.
- Correct the two known W121 measurement-spec defects prospectively
  in a NEW W122-only instrument before any candidate outcomes.
- Predeclare new disjoint held-out episodes using U and frozen
  feasibility information only; account for all eligible/failed cases.
- Run matched frozen controls and actual Blender world evaluation.
- Render 30 fps real-time side-by-side motion with anonymized variants.
- Measure raw motion, source-relative compensation and articulation
  separately, with full negative results and uncertainty.
- Include existing synthetic cases as causal sanity controls.

Do NOT:
- Reassign W121 D to B or patch W121/W120 source.
- Modify frozen model, rig, timebase, solver, margins, flex floor,
  objective, acceptance thresholds, authorization or Production code.
- Choose episodes after seeing factor or playback outcomes.
- Claim source 3D GT, tracked-subject identity or animator validation
  without actual evidence.
- Train a Generator, build Production UI, add constraint types or
  select permanent joint-quality limits.

Decision enabled:
Whether the world-space guide has bounded, reproducible, visually
meaningful real-time editing utility on the existing source set, is
limited to a source-relative numerical effect, or remains untestable
because of source/perception/measurement validity.

HISTORICAL BASELINE

Repository: kn99e39/AnimCV
Existing branch ONLY: arch/single_frame_first
Observed W121 COMPLETE HEAD: 3fc090817575a4cd134fb6af323d8512f955533b.
W121 Worklog: docs/121_WORKLOG_INDEPENDENT_REAL_VIDEO_WORLD_PIN_TRANSFER.md
W121 Worklog blob: f1a8c305b7c12052f839543da1a473d2c6c7effa
Current instruction before issue: W121 blob
1d9f48f88b12b348b80ebb1b4a5ff15dfa403dfa.

Before doing anything, fetch the remote and check HEAD, this new W122
instruction blob, current Worklog sequence, AGENTS.md and normative
architecture. Never overwrite newer concurrent work.

Read:
- AGENTS.md, PRODUCT_VISION.md,
  Architecture_v3_FramePose.md, Architecture_v2.md (precedence)
- docs/AUDIT_W116_W120_2026_11_06.md
- docs/116–121 Worklogs, especially W120 and W121 official verdicts
- scripts/w116_semantic_pin*, scripts/w116_blender_stages.py
- scripts/w118_transition_edit*, scripts/w119_endpoint_transition*
- scripts/w120_factor_attribution*, scripts/w121_independent_transfer*
- scripts/w121_posthoc_checks.py (READ-ONLY HISTORICAL DIAGNOSTIC)
- tests/test_w120_factor_attribution.py,
  tests/test_w121_independent_transfer.py
- W121 acquisition/inspection/perception/prepare/declaration/decision
  manifests and locally stored originals, if actually present.

Preserve historical official CASEs exactly:
W113 D, W114 D, W115 D, W116 B, W117 C, W118 D, W119 D,
W120 B, W121 D.

W121's post-hoc "would be B" is explanatory only and is NOT
an authorized result, a valid new baseline or a source of new gates.

PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE:
- All original W102/W105 and W116–W121 assets, Worklogs and verdicts.
- W121 three original video bytes and verified licence/provenance,
  hashes, decoded frame/time mapping, model/checkpoint identity,
  frozen W102 M1/S0 inference and W105 FIXED-slot adapter.
- W116 scene-world metre pin semantics and scratch 1 mm diagnostic
  (NOT a Production tolerance).
- W118 automatic bounded transition scope and truthful actual footprint.
- W119 clean per-request U_r/F_r/X_r isolation and frozen comparator.
- W120 four variants G0F0/G0F1/G1F0/G1F1 and identical solver,
  pole, guide, floor, budgets, margins, source/rig/DOF and gates.
- Original 30 fps source/Action timing for S1/S2/S3; no frame
  re-timing, resampling or interpolation to hide discontinuities.
- Frozen real-video and synthetic negatives, source guard, exact undo.
- W121 original declaration, results, evidence manifest and failed V10
  and V12 controls. They are immutable historical evidence.

CHANGE ONLY:
- New W122-only scratch code, tests, outputs and numbered Worklog.
- New predeclared held-out windows/requests derived from existing
  frozen source states, before new solver outcomes.
- NEW W122 instrument validity rules and controls that explicitly
  correct W121's known mistakes BEFORE any new candidate execution.
- Blender scratch Action evaluation, real-time comparison renders,
  non-Production annotated timelines and visual review artifacts.
- Small attributable visual previews in the repository only when
  licence and repository size permit; original videos remain outside.
- Deterministic synthetic copies for causal probes; never overwrite
  historical synthetic states.

DO NOT TUNE:
- Any model weight, 2D threshold, source association algorithm,
  frame-to-rig mapping, rig rest geometry, or Animation Base values.
- W116–W120 solver, hard-pin or source-preserving contracts.
- W119/W120 numerical quality comparator or thresholds.
- W121 source selection after seeing solver results.
- New visual-rating rubric or CASE gates after seeing W122 outcomes.
- User-edit amplitudes, clip FPS, temporal margins, budget, DOF,
  flex floor, guide weight or Blender playback rate.
- No Production src/, UI, schema, retargeter, FBX, trained Generator.

REQUIRED EVIDENCE — PART 1: DATA AND REPRODUCTION PREFLIGHT

Authenticate S1/S2/S3 from W121's declared Commons originals:
- S1: Double Kettlebell Strict Press (side view).webm,
  original SHA-256 prefix de8feb9d515daa2b.
- S2: Firejuggling 2009 ubt.ogv, prefix 68301b0c35c98448.
- S3: Bent-over row - exercise demonstration video.webm,
  prefix f6b90f2f72e9020d.

Verify full hashes against W121's original acquisition manifest, exact
decoded frame fingerprints, licences, source time, checkpoint identity,
and the W121 parent state hashes and rig bindings.

These are three sources independent from the historical W102 clip.
They are NOT a new source-acquisition population in W122. Do not
download substitutes, circumvent HTTP 429, or use re-encoded W102.

For S2, inspect source frames and frozen detection boxes/landmarks
BEFORE solver outcome inspection. Record identity continuity as
VERIFIED only if there is affirmative source-backed evidence for
the same person across the relevant interval. Otherwise UNKNOWN;
do not assume the frozen tracker proves identity. Track-switch
ambiguity may exclude a request from strong source-fidelity claims
but MUST remain in coverage and negative-result accounting.

Verify historical W102 reproduction and W105 rig adapter fidelity
from the frozen artifacts, or rerun the frozen path if needed.
Distinguish agent-stored hashes from independently re-executed checks.
If original video or rig identity is missing or invalid, report
SOURCE/PERCEPTION/RIG BLOCKED, not a successful transfer.

REQUIRED EVIDENCE — PART 2: PROSPECTIVE INSTRUMENT REPAIR

Build a NEW W122 validity instrument. Never patch W121 or reinterpret
its official verdict. Before building or viewing ANY W122 factor outcome:

V10 repair:
- Repeatability compares deterministic states, exact Blender
  evaluations, and semantic solver records AFTER excluding ONLY
  explicitly enumerated nondeterministic timing/telemetry fields.
- Start with the known W121 wall_s issue; enumerate every excluded
  field in the pre-outcome declaration. Do not drop solver statuses,
  planned/changed frames, residuals, source IDs, state IDs, margins,
  budgets, branch flags or quality outputs.
- Negative control: alter a semantic solver field in the second run;
  repeatability MUST fail. Merely stripping all records is invalid.

V12 / display repair:
- The REJECT->ACCEPT label-corruption test MUST use a prospectively
  selected genuinely REJECTed variant; assert the precondition
  decision == REJECT before injecting the display corruption.
- Require that the corruption changes the displayed status and that
  the independent truth check detects it.
- Keep controls for missing changed frame/bone, wrong pin domain,
  collapsed disjoint runs, hidden change, false ACCEPT, stale parent,
  inexact revert, wrong source/model/FPS, baseline contamination,
  hard-pin violation and error-class precedence.
- If no eligible REJECT exists, predeclare a controlled infeasible
  request and verify it, not an after-the-fact substitute.

Run independent fixture and negative-control preflight before outcome
inspection. Freeze a W122 declaration with code blobs, input hashes,
source windows, request IDs, negative expectations, numerical and
visual metrics, review order, validity gates and CASE rule.
If either corrected gate cannot be made non-vacuous prospectively,
STOP as INSTRUMENT BLOCKED. Do not run candidates then repair gates.

REQUIRED EVIDENCE — PART 3: NEW WITHIN-SOURCE USER EDITS

Use the existing W121 S1/S2/S3 30fps parent Actions, no new source.
Construct a NEW prospective hold population distinct from W121's
LOW/MID/HIGH windows.

- 9-frame WORLD_POSITION_PIN at the captured left wrist scene-world
  position; existing W121 user edits none, spine_01 +5° and −5°.
- Same authorized {upperarm_l, lowerarm_l}, frozen W118 m_max = 8,
  per-request isolated U/F/X and matched W120 factor mechanisms.
- Enumerate candidate windows entirely from unedited Blender U and
  frozen W116 F feasibility, not from factor outcomes.
- Exclude overlap with W121 pin windows AND their ±8-frame transition
  budgets; remain within one authenticated shot/track.
- Select at most two feasible NEW windows per source, one lower-travel
  and one higher-travel, by a deterministic rank rule declared before
  any factor run. Preserve the full candidate list and tie rules.
- Do not force selection of a high-travel feasible episode if none
  exists. Report how many windows were feasible/infeasible and why.
- Evaluate all three edit variants on selected windows; keep every
  rejection in the declared denominator.
- Also evaluate W121's predeclared original windows as a SEPARATE
  labelled REPLICATION/ANCHOR population, never as new held-out
  episodes. Do not combine them to inflate independent evidence.
- Include at least one frozen infeasible and one zero-budget control
  per processable source. Do not exclude them post-hoc.

The previous W121 guide-positive S1_MID and S2_LOW are KNOWN from
history, not held-out discoveries. They may inform a hypothesis but
must never count as independent W122 holdout wins.

REQUIRED EVIDENCE — PART 4: MATCHED FROZEN BLENDER EVALUATION

For every predeclared request use:
U_r — unpinned user edit,
F_r — isolated W116 framewise pin,
X_r — W118 transition baseline,
G0F0/G0F1/G1F0/G1F1 — frozen W120 matched factors.

Primary GUIDE contrast G1F0 vs G0F0; conditional GUIDE G1F1 vs G0F1.
Primary FLOOR contrast G0F1 vs G0F0; conditional FLOOR G1F1 vs G1F0.
Keep all contrasts, interactions, and X_r vs G0F0 differences labelled.

Enforce:
- same source, parent, rig, pin, user edit, auth, budget and margins;
- source video and checkpoint immutability;
- independent Blender world-pin evaluation, not just Python FK;
- no new infeasible acceptance or feasible nonconvergence masked;
- changed frame/bone footprint subset of authorised temporal domain;
- user-edit spine and object/root preserved as specified;
- preview/result status matches true commit and actual Blender state;
- exact revert and inexact rollback detection;
- valid zero-budget and negative error precedence.

Do not treat the G0F0 wrist matching X_r as full joint-pose parity.
Do not tune any method to improve W122 quality.

REQUIRED EVIDENCE — PART 5: SOURCE VS EDIT ERROR ATTRIBUTION

On actual Blender-evaluated animation, in metres and real seconds,
report per request and contrast:

1. U raw wrist/elbow step, peak angular step, discontinuity timing.
2. F_r/X_r/G0/G1 raw wrist and elbow steps in pin/entry/release
   and adjacent unpinned frames, not only compensation residual.
3. Source-relative edit-induced compensation and release errors,
   explicitly distinguished from raw motion.
4. Changes in speed/acceleration at the 30 fps real-time rate,
   source-relative and absolute; identify source-dominated spikes.
5. Elbow flex, near-straight frames, bend-plane flips, arm rotation,
   user-edit preservation, pin reachability, solver failure and
   collateral change.
6. Exact actual changed frame/channel footprint and the timeline
   disclosure, including rejected candidates and reversibility.
7. Source and subject-association confidence; no video 3D GT claim.

Predeclare a source-dominated vs edit-induced diagnostic partition
from U's raw steps BEFORE looking at factor outcomes. Treat it as
experimental stratification, not a permanent source-quality threshold.
Report both aggregate and per-source/per-window results; do not
treat edit variants of the same window as independent episodes.

An improvement in compensation peak that restores a large raw U
jump is NOT automatically an animation-quality improvement.

REQUIRED EVIDENCE — PART 6: REAL-TIME PLAYBACK UTILITY

This is the essential NEW evidence missing from W121.

Render consecutive full-motion segments at EXACTLY 30 fps in the
unchanged Blender rig, with no extra temporal filtering, frame
interpolation, motion smoothing, motion-blur masking or retiming.
Include full entry/hold/release and sufficient adjacent context.

- Source frames, U, F_r, X_r and G0/G1 variants must be synchronised
  on an explicit source-time axis; no silent camera or scale changes.
- Show world wrist pin target, relevant elbow/wrist paths and the
  actual planned vs changed timeline. Keep a clean playback view too.
- Use front and side views; distinguish world-axis views from
  character-relative views. Show both real-time and optionally
  slowed diagnostic playback, clearly labelled.
- At least the complete new held-out population, the W121 known
  positives, S3 negative/neutral, and all hard-pin failures must
  receive predeclared visual coverage. If full coverage is infeasible,
  declare the fixed sample and excluded population BEFORE review.
- Randomize/anonymize paired G0F0/G1F0 labels by a fixed predeclared
  seed, retain sealed mapping until ratings are recorded.
- Before unblinding, record for each pair:
  visibly better G0 / visibly better G1 / tie / unjudgeable;
  whether raw motion continuity, elbow plausibility, pin behavior,
  temporal transition, source fidelity and user intent improve,
  worsen or cannot be judged, with specific frame/time evidence.
- Inspect full-motion playback rather than stills alone. If playback
  cannot be generated/inspected, do NOT claim a positive visual result.
- The Agent is not an independent animator. Label all such judgments
  AGENT REVIEW, not professional user validation.
- Publish compact representative attributable evidence if rights and
  size permit; otherwise report exact local paths and hashes.
  No unauthorized original-video redistribution.

Do not change ratings or the visual rubric after label reveal.
A small visible difference is not synonymous with useful improvement.

SYNTHETIC + REAL REPLAY

Real:
- Three W121 sources, with new held-out within-source windows plus
  separately labelled W121 historical replication windows.
- Explicit source-quality/identity limitations; retain negatives.

Synthetic:
- Reuse frozen W120 H_G held-out and H_C near-singular controls,
  with original states unchanged.
- Optionally create ONE prospectively fixed deterministic scratch
  jitter perturbation of a copy of a synthetic parent, e.g. ±2°
  alternating upperarm_l rotation; declare exact frames, amplitude,
  seed and expected raw/source-relative effect before running.
  This is a controlled rig-input perturbation, NOT simulated
  real-video perception evidence.
- Test hard pin, no hidden edit, negative control, rollback, and
  whether source discontinuity can dominate a compensation metric.
- Keep synthetic conclusions out of real-video utility denominators.

DECISION GATES — PREDECLARE BEFORE CANDIDATE OUTCOMES

DATA BLOCKED / NO CASE:
No authenticated W121 original video plus frozen usable rig input
can be recovered; report exactly what is missing. Do not substitute
W102 or invent a new source.

CASE D — INVALID:
Any material predeclared source identity, timebase, subject-binding,
reproducibility, factor fairness, independent evaluator, population
selection, negative control or review-blinding validity gate fails.
W121's V10/V12 errors must NOT recur. A source with unverified
person continuity is labelled UNKNOWN, not silently VERIFIED.

CASE C — NOT SUPPORTED:
Valid tests show no meaningful real-time guide utility on the
prospective held-out episodes, or a material pin/authorization/
reversibility/quality-safety failure. Separate mechanism failure
from source-input limitation; never claim a better Animation Base.

CASE B — PARTIAL / CONTEXT-SPECIFIC:
Valid tests show limited or mixed real-time utility, numerical-only
guide benefits without convincing playback improvement, or benefits
restricted to individual windows/sources, with clear limitations.

CASE A — BOUNDED WITHIN-SOURCE UTILITY SUPPORT:
All validity and hard-contract gates pass; the prospective held-out
episodes show reproducible meaningful guide benefits by the frozen
numeric comparator AND blinded real-time AGENT REVIEW, on at least
two distinct held-out windows from at least two source-verified
clips, without new material raw-motion/articulation/collateral harm.
This is NOT new-source generalization, independent animator
validation or Production approval. If fewer than two verified
sources have valid eligible episodes, A is unavailable.

Require explicit numerical and visual agreement; a source-relative
improvement alone can yield at most B. Predeclare how to handle
ties, unjudgeable video, conflicting raw/relative evidence and
infeasible episodes. Never change gates after outcomes.

TESTING

Run focused tests for:
- source identity, video/frame/cadence and checkpoint provenance;
- W121 frozen artifact reproduction and W122 new window exclusion;
- non-vacuous V10 repeatability excluding ONLY declared telemetry,
  with semantic-field corruption detected;
- V12 REJECT->ACCEPT display corruption with verified precondition;
- independent error-class precedence, budget zero and isolation;
- matched factor fairness and no hidden tuning;
- hard-pin world residual, actual changed frame/channel footprint,
  rejected candidate status and exact/inexact rollback;
- time-axis and 30 fps real-time render identity;
- randomized blind label mapping and fixed visual rating records;
- separate historical, new-held-out and synthetic denominators.

Report exact commands and PASSED/FAILED/NOT RUN. Run relevant
focused regressions; do not waste time on unrelated full suite
for diagnostic-only changes. Do not report local Blender runs
as Reviewer independently executed.

STOP CONDITIONS

Stop and report the specific blocker without changing Production if:
- source/video/rig/checkpoint/parent identity cannot be verified;
- historical W121 files/results must be rewritten;
- prospectively valid source/track/time mapping is unavailable;
- corrected V10/V12 negative controls are vacuous or failing;
- new held-out window selection requires seeing factor outcomes;
- frozen W120 solver requires unapproved code/parameter changes;
- the matched budget, source, user edit, rig or authorized DOF differs;
- any source guard, pin evaluator, hidden-edit or rollback gate fails;
- 30 fps real-time playback cannot be made or inspected (then do
  not claim visual support; report the bounded evidence limitation);
- user policy on new constraint types, training, Production schema/UI
  or permanent joint-quality thresholds becomes necessary.

Do not fix a failed result by swapping clips, widening budgets,
changing tolerances, filtering raw animation, hiding negative cases
or rewriting historical CASEs.

FINAL REPORT

Create docs/122_WORKLOG_<descriptive_name>.md containing:
1. Official W122 CASE or DATA BLOCKED and its exact scope;
2. Frozen W121 D, W120 B and all earlier historical verdicts;
3. Source, license, exact video/frame/model/rig hashes and continuity;
4. Prospective instrument declaration and non-vacuous V10/V12 repairs;
5. New held-out window candidate/selection/coverage accounting;
6. Original W121 replication separately from held-out episodes;
7. Synthetic controls separately from real-source results;
8. Matched factor/zero-budget/negative and pin safety results;
9. Raw vs source-relative motion and per-source quality attribution;
10. Actual Blender changed frame/bone scope, timeline and reversal;
11. Full real-time visual method, blind mapping, ratings, evidence;
12. What was visible vs what was only a numerical improvement;
13. Every failure, rejected/unreachable edit and alternative explanation;
14. Files changed, exact commands, tests, environment, artifacts/hashes;
15. Explicit non-claims about user/animator validation, GT fidelity,
    new-source generalization, Production quality or Generator training.

All local visual judgments and Blender runs must be marked AGENT-RUN
or AGENT REVIEW. Do not claim independent Reviewer reproduction.

Do NOT provide:
Next Work Recommendation, Next Architecture Decision,
Recommended Next Experiment, Proposed Next Batch,
Next Implementation Plan.

Do not begin W123, train models, or modify Production code.

COMPLETION CONDITION

W122 is complete only if the Reviewer can distinguish:
1. Whether W121's known instrument defects were prospectively avoided
   in a new, non-vacuous, validity-controlled experiment;
2. Whether new within-source held-out pin episodes were selected
   without seeing candidate outcomes, with all failures accounted;
3. Whether frozen guide/floor effects improve ACTUAL real-time
   Blender animation rather than only source-relative metrics;
4. Whether source/perception discontinuity and track uncertainty
   explain or limit the apparent guide benefit;
5. Whether strict hard pins, bounded visible edits, source identity,
   user intent, exact rollback and negative controls remain valid;
6. Whether the evidence supports bounded within-source utility,
   context-specific benefit, no support, invalidity, or data blockage.

A negative result is valid evidence. W121's official CASE D remains
unchanged under every outcome. No automatic W123 execution.
