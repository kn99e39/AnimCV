# BATCH 114 — Source-Preserving Animation Base & Edit Authority Feasibility

DIRECTION

Advance AnimCV's user-approved transition from a
Video-to-Animation Converter toward a Reference-Guided,
Reliability-Aware Animation Generator.

This Batch tests the minimum architectural contract needed
to preserve source evidence while permitting controlled,
reversible modifications of inferred animation.

Investigate whether logically separating immutable source
evidence from editable animation state resolves the
information-ownership and edit-authority failures observed
in W113.

This is a bounded architecture-evidence Batch.
It is NOT a Generator implementation Batch.


PURPOSE

AnimCV must eventually construct production-usable animation
from imperfect video-derived motion, learned motion priors,
explicit user constraints, and iterative editing.

The current pipeline cannot safely assume that:
- Detector confidence represents calibrated 3D reliability.
- A reconstructed 3D pose is necessarily correct.
- A joint-level edit permission implies bone-level authority.
- An existing AnimationClip retains the information that
  produced it.
- An edit of a source value necessarily changes the final rig.
- A visually natural result preserves the reference motion.

W113 demonstrated failures or missing contracts at these
boundaries, but its frozen evaluation was CASE D INVALID.

W114 must determine whether a minimum source-preserving
animation representation can establish the required
information and edit-authority contracts.

A negative outcome is acceptable.


CENTRAL INTENT

Can AnimCV represent immutable observation/reconstruction
evidence and editable animation state separately, while
preserving identity, time, coordinates, provenance, user
constraints, and the actual source-to-rig influence needed
for selective editing?

More specifically:

Can a versioned, source-preserving edit representation allow
an authorized change to affect only its legitimate motion
targets, without modifying original observations, silently
manufacturing reliability, losing source identity, or
mistaking rig dependencies for actual edit authority?

Determine whether this architectural separation is viable.

Do NOT optimize pose accuracy.
Do NOT train a Generator.
Do NOT treat additional fields alone as proof of viability.


CURRENT ASSESSMENT

W113 official verdict:
CASE D — INVALID.

The frozen W113 declaration assumed detector confidence
belonged to [0, 1].

Authentic data contradicted that assumption:
- Real clip: 69 / 3,383 present joint-frame cells > 1.
- Benchmark: 1,350 / 353,457 valid cells > 1.
- Existing depth_uncertainty = 1 - confidence produces
  negative values.

The W113 post-hoc diagnostics are not a replacement verdict.

They provide preliminary evidence that:
- Source observations and inferred 3D states can be joined.
- A synthetic joint-level edit can be isolated to 38 cells.
- User locks and conflicts can be represented and rejected.
- Source identity is not consistently serialized downstream.
- Existing frame-level locking is insufficient for joint-level
  editing.
- W105's conservative joint-to-bone dependency policy can
  prohibit every bone edit.
- Source time and Blender Action playback time can disagree.

None of this proves that real animation generation,
reconstruction correction, or production-ready editing works.

Preserve W113 CASE D unchanged.


HUMAN-READABLE SUMMARY

Direction:
Test a source-preserving, reversible Animation Base contract.

Purpose:
Establish whether an eventual Generator/Editor can modify
selected inferred motion while preserving authentic source
information and user-imposed constraints.

Central Intent:
Determine whether immutable evidence plus a versioned edit
layer creates a viable boundary between reconstruction,
generation, and rig animation.

Do:
- Diagnose the existing source-to-output information flow.
- Build a minimal isolated candidate representation.
- Compare it against unchanged existing paths.
- Test reversible edits and ownership.
- Verify actual rig influence instead of assuming it.
- Use authentic real observations and synthetic controls.
- Report remaining missing information and ambiguities.

Do NOT:
- Modify production architecture or data schemas.
- Train or implement a learned Generator.
- Repair the estimator, detector, FK/IK, or FBX exporter.
- Introduce confidence calibration or heuristic repair.
- Convert a synthetic contract PASS into a production claim.
- Choose subsequent Architecture work.

Decision enabled:
Whether the separation of source evidence and editable motion
is sufficiently viable to justify a future production-facing
Animation Base contract.


HISTORICAL BASELINE

Repository:
kn99e39/AnimCV

Existing branch ONLY:
arch/single_frame_first

Expected starting state:
W113 COMPLETE.
Last observed HEAD: 8fe00318f21a4735ed374eeecdeaf1dac5634fa3.

Verify the current remote state before any work.
Do not assume it is unchanged.

Relevant records:
- PRODUCT_VISION.md
- Architecture_v3_FramePose.md
- Architecture_v2.md
- docs/113_WORKLOG_ANIMATION_BASE_CONTRACT_FEASIBILITY.md
- W102-W105 observation, estimator, and rig evidence
- W113 scripts, tests, declaration, and evidence identities

W113 frozen CASE D must remain historically intact.
Its post-hoc runs must remain explicitly labelled post-hoc.

Preserve:
- W102 real observation arrays and source video.
- Frozen S0 checkpoints and predictions.
- W105 rig sources, calibration, Action and replay baselines.
- Existing MotionGraph and AnimationClip serialization.
- Existing user-visible workflow behavior.
- Existing production score thresholds.
- Existing test expectations and Worklog verdicts.

No branches may be created, switched, merged, rebased,
reset, or deleted.

Do not alter historical artifacts or reinterpret past verdicts.


ARCHITECTURE HYPOTHESIS

H1:

A logically separated representation consisting of
immutable source evidence and versioned editable state
can establish a useful minimum contract for controlled
animation editing without requiring immediate modification
of the production MotionGraph schema.

The candidate must distinguish:

1. Source evidence
   Actual 2D observations, detector scores, source validity,
   frame/time identity, derived observation provenance.

2. Reconstruction hypothesis
   Inferred 3D pose, coordinate system, unit, model/source
   identity, and whether the estimate has direct input support.

3. Editable animation state
   Candidate motion values, modification identity,
   parent state, and explicit authorized targets.

4. User constraints
   Preserve, permit, prohibit, and conflict semantics.

5. Output association
   The source values and operations that actually influence
   generated Rig/AnimationClip channels.

These are logical responsibilities.
They do not require separate physical files or classes.

The Agent may select an efficient scratch implementation,
provided the information boundaries and invariants are
preserved.

Compare against the unchanged existing path.

Do not assume H1 is correct.


PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE:

- Production source modules under src/.
- Existing MotionGraph and AnimationClip schemas.
- Existing FramePose model and source pipeline.
- Existing estimator weights and numerical policies.
- Existing thresholds, masks, crop logic and retargeting.
- W105 mixed-source rig composition.
- Existing source frame order and timestamps.
- Historical data, hashes, decisions, and baselines.
- Existing Blender/FBX behavior.

CHANGE ONLY:

- New isolated diagnostic scripts and focused tests.
- Scratch-only candidate representation.
- Scratch-only reversible edit records or overlays.
- Controlled synthetic input perturbations.
- Evidence and reporting artifacts for this Batch.
- W114 Worklog.

The scratch representation must not become an implicit
production format.

DO NOT TUNE:

- Detector confidence threshold.
- S0 or HMR inference settings.
- IK or FK parameters.
- Bone lengths or reference calibration.
- Output frame rates.
- Smoothing or interpolation.
- Generation parameters.
- Acceptance tolerances after observing results.

All new numerical perturbations, comparisons, validity
gates and decision thresholds must be declared before
their outcomes are examined.

Prefer existing declared numerical contracts and quantities
derived from the frozen input.


REQUIRED EVIDENCE

PART 1 — SOURCE SEMANTICS AND OWNERSHIP

Trace the actual producer-to-consumer path for:

- Raw 2D joint coordinates.
- Detector scores.
- Observed and derived joint status.
- Observation validity.
- S0 3D output.
- Source and model provenance.
- Frame and timestamp.
- Subject/track identity.
- User-preserved and editable regions.
- Bone/channel source ownership.

Identify information as:
PRESENT / DERIVED / LOST / UNKNOWN / UNAVAILABLE.

A field being present upstream does not establish that
downstream code receives or uses it.

Explicitly distinguish a source's numeric score from:
- observation validity;
- calibrated probability;
- 3D pose reliability;
- permission to edit.

Preserve raw detector scores exactly.

Do not clip, normalize, sigmoid, or otherwise reinterpret
values above 1.

A finite detector score above 1 must not alone cause the
W113-style invalidity failure.

Missing or non-finite data require an explicit status and
the predeclared fail-safe behavior.

Do not derive calibrated 3D uncertainty from 1 - score.

For unavailable 3D reliability, record UNKNOWN or
UNAVAILABLE rather than manufacturing a number.


PART 2 — MINIMUM CANDIDATE ANIMATION BASE

Construct a scratch-only versioned representation capable of:

- Referencing immutable source evidence.
- Preserving source-relative frame/time mapping.
- Preserving 2D and 3D coordinate semantics.
- Preserving the observed/derived/inferred distinction.
- Carrying unknown or unavailable information explicitly.
- Binding a candidate edit to an exact source state.
- Recording original values separately from replacements.
- Expressing joint/time-scoped user constraints.
- Reconstructing the original state without changing
  the source evidence.

Provide a reproducible baseline-to-candidate comparison.

Do not mutate the historical source arrays in place.

An edit must contain enough provenance to determine:
- which input state it was based on;
- what target it intended to change;
- which values it actually changed;
- which constraints governed the change;
- which source values remained unchanged.

A stale edit must not silently apply to a newer state.

Multiple edits and reversals must preserve an auditable
state lineage.

Implement only the minimum operations required to test
these invariants.

This is not a generalized editor or version-control system.


PART 3 — REAL-OBSERVATION REPLAY

Use the W102 authentic GT-free real clip and the W113
retained real-observation evidence.

Preserve all 199 frames and their original source timebase.

Retain the W113 38-cell left-wrist region as one controlled
edit-permission case.

Use deterministic synthetic modifications of authentic
stored values.

Label these as MOCK EDITS, not reconstruction corrections.

Evaluate separately:
- immutable evidence preservation;
- changed 3D state;
- permitted vs prohibited regions;
- stale-state rejection;
- reversal accuracy;
- provenance preservation;
- temporal identity;
- source score semantics.

Include a separate benchmark-observation replay using
the existing W113 3DPW sequence.

Do not pool benchmark and real-clip regimes.

The benchmark is not a new independent holdout.

Do not access evaluation GT to construct runtime candidate
values, edit masks, confidence or source associations.

Do not fabricate absent raw observations.

Do not silently convert UNKNOWN into zero.


PART 4 — ACTUAL SOURCE-TO-RIG AUTHORITY

This is a critical part of W114.

Do not equate a geometric dependency with permission to
edit a bone.

First establish the actual source ownership of the frozen
W105 rig channels.

Distinguish sources such as:
- S0-derived leg motion;
- HMR-derived torso/root context;
- J1X-derived arm motion;
- fixed/rest-pose components.

Confirm ownership from the existing implementation and
controlled outputs rather than from naming conventions.

Construct a controlled source-perturbation test that
distinguishes:

A. A modified value that the rig does not consume.

B. A modified value that the rig actually consumes.

C. A dependency that appears in a conservative map but
   does not change the output under the tested intervention.

D. An intervention that legitimately affects multiple
   channels because of shared computation.

Use the existing frozen rig implementation.
Do not change solver behavior.

For each condition:
- hold unrelated source values fixed;
- declare the perturbation before observing output;
- identify expected and observed affected channels;
- measure changes and unchanged-channel preservation;
- explain mismatches;
- report whether edit authority can be safely assigned.

Use the real clip for controlled replay where supported.

A positive rig-influence test must use a source that
actually feeds the tested rig channel.

An edit to an ignored source is a valid negative control,
not a successful animation-edit demonstration.

Do not assume that the S0 left-wrist value controls W105's
left-arm Action simply because it has the same joint name.

If an authentic low-confidence joint has no direct influence
on the tested rig, record that boundary explicitly.

A separate controlled source-backed perturbation may be
used to test positive rig influence, but it must not be
described as natural occlusion recovery.

This phase determines local influence under frozen
conditions, not a universal global bone dependency model.

Do not infer global causality from one finite perturbation.


PART 5 — IDENTITY AND TIME

Distinguish:

- Source sequence identity.
- Frame identity.
- Tracked subject identity.
- Candidate animation identity.
- Rig channel identity.

Do not substitute a generic "single subject" label for
proof of uninterrupted same-person tracking.

If per-frame subject association is unavailable, preserve
that uncertainty and fail closed for operations that require
verified cross-frame identity.

Evaluate time correspondence in seconds, not only frame
indices.

Report the unchanged 50 fps source versus 30 fps Action
mismatch explicitly.

A source-time sidecar may be used for diagnostic association.

Do not claim that an unchanged Action or FBX file preserves
that time mapping merely because a sidecar exists.

No FBX export/import modifications are authorized.


PART 6 — SYNTHETIC CONTRACT AND NEGATIVE CONTROLS

Include focused synthetic tests for:

- Unauthorized edits.
- Preserve/permit conflicts.
- Stale source digests.
- Stale edit parents.
- Missing source evidence.
- Missing source provenance.
- Mismatched frame indices.
- Mismatched timestamps.
- Subject-identity ambiguity.
- Duplicate joint identities.
- Coordinate or unit mismatch.
- Modified immutable observations.
- Unknown-to-zero coercion.
- Manufactured 3D confidence.
- Invalid source-to-rig association.
- A source change with no actual rig influence.
- An actual output change outside the authorized domain.
- Reversal that does not restore its parent state.

Controls must have declared expected outcomes before
execution.

Reject invalid operations without partial application.

Do not modify controls after observing their results.

If a real population cannot exercise a specific control,
label it NOT APPLICABLE and provide the synthetic evidence
separately.

Never count NOT APPLICABLE as PASS.


QUANTITATIVE ACCOUNTING

Provide exact denominators for:

- Source joint-frame cells.
- Available and missing observations.
- Derived and directly observed values.
- Supported and unsupported 3D estimates.
- Eligible and prohibited edit cells.
- Actually changed cells.
- Unintentionally changed cells.
- Restored cells after reversal.
- Source fields lost at each boundary.
- Frames with unknown subject continuity.
- Source-to-rig channels exercised.
- Expected versus actual affected bone channels.
- Source and Action time mismatches.
- Negative controls passed, failed and not applicable.

Separate:
- Synthetic correctness.
- Real-observation contract behavior.
- Rig influence evidence.
- Architecture viability.

Do not use aggregate PASS counts to obscure a critical
contract failure.


QUALITATIVE REVIEW

Inspect representative cases for:

- A reliable observed joint that must remain unchanged.
- An input-unsupported joint allowed for mock editing.
- A low-score observation that must not be assumed wrong.
- A conflicting user directive.
- A source edit that does not affect the final rig.
- A source edit that does affect the final rig.
- A timebase mismatch.
- An unresolved subject-identity condition.

Provide frame indices, joint/channel identities,
source references, and before/after evidence.

If images or animation renders are produced, distinguish
their diagnostic purpose from quality validation.

A visually plausible pose is not proof of correct source
ownership or of good generated motion.


CONTROLLED COMPARISON

Control:
Existing unmodified MotionGraph/AnimationClip and W105
source-to-rig behavior.

Candidate:
Scratch-only source-evidence plus versioned editable-state
representation.

Keep model predictions, observations, rig configuration,
source timing, thresholds and original source data fixed.

Only the representation and declared mock edits may differ.

Do not attribute improvements to representation design
if other relevant factors were changed.

If a required condition cannot be controlled, explain
the resulting attribution limitation.


TESTING

Run focused tests after meaningful implementation units.

At minimum:
- candidate serialization and reconstruction;
- immutable source preservation;
- versioned edits and reversal;
- score-domain semantics;
- permission/conflict handling;
- negative controls;
- controlled source-to-rig influence;
- authentic source replay;
- repeatability.

Use existing relevant regressions when a shared contract
could be affected.

A full test suite is not mandatory for a diagnostic-only
Batch with no production changes.

Report exact test commands and execution outcomes.

Tests not executed must be listed as NOT RUN.

Do not claim external runtime or engine compatibility.


STOP CONDITIONS

Stop implementation and report if:

- A required source cannot be authenticated.
- Real data contradicts a predeclared fundamental contract.
- Source identity or numerical provenance is unreliable.
- Historical baseline identities cannot be preserved.
- A necessary modification requires changing production
  schema, models, rig behavior, or exporter semantics.
- The candidate cannot distinguish source evidence
  from generated or edited values.
- A purported authorized edit modifies immutable source data.
- Required rig behavior cannot be causally attributed under
  the frozen controls.
- A new major Architecture Decision is necessary to
  continue within the approved scope.

Preserve partial evidence on a STOP.

Do not bypass a stop condition through tolerance changes,
new heuristics, new model training, or revised acceptance
criteria.

Do not implement an unapproved alternative architecture.


DECISION CONTRACT

Declare explicit gates before examining candidate outcomes.

Use:

CASE A — BOUNDED SUPPORT
The minimum source-preserving representation satisfies
its declared integrity, provenance, edit/reversal, permission,
time/identity safety and controlled rig-influence contracts
on the tested populations.

Support is limited to those tested contracts.
It does not prove Generator quality or production readiness.

CASE B — PARTIAL SUPPORT
Source-preserving representation and bounded editing work,
but a material downstream authority, identity, or output
boundary remains unresolved under valid experiments.

Describe the successful boundary and failed boundary
separately.

CASE C — NOT SUPPORTED
Valid evidence shows that the candidate separation cannot
provide the required minimum guarantees within the declared
scope, or that essential information is genuinely
unavailable rather than merely lost in a downstream format.

Identify the observed mechanism and alternative explanations.

CASE D — INVALID
Fundamental input, provenance, reproducibility, positive
control, or declared validity assumptions fail, preventing
a trustworthy architecture verdict.

D has precedence over the other outcomes.

Do not promote a post-hoc diagnostic to a declared CASE.
Do not reinterpret W113's historical CASE D.

Negative evidence is a valid completion outcome.


FINAL REPORT

Write a W114 Worklog with:

1. Official verdict and scope.
2. Verified historical baseline identities.
3. Declared architecture hypothesis and controls.
4. Producer/consumer ownership matrix.
5. Source score semantics and validity accounting.
6. Candidate information-preservation results.
7. Edit locality and reversal accounting.
8. Real and synthetic results, reported separately.
9. Actual source-to-rig influence accounting.
10. Identity, provenance, and timebase limitations.
11. Representative positive and negative cases.
12. Alternative explanations and unresolved uncertainty.
13. Changed files and tests.
14. Reproduction instructions.
15. Explicit non-claims and scope limitations.

Report measured observations rather than conclusions based
only on successful implementation.

Do NOT claim architecture success automatically.
Report negative results honestly.

Do not request or provide:
- Next Work Recommendation.
- Next Architecture Decision.
- Recommended Next Experiment.
- Proposed Next Batch.
- Next Implementation Plan.

The Agent produces evidence.
The Architecture Reviewer owns subsequent decisions.

Do not begin Batch 115.


COMPLETION CONDITION

W114 is complete only when the report provides sufficient
controlled evidence to determine:

1. Whether immutable source evidence can be preserved while
   editable animation values evolve independently.

2. Whether user constraints can safely authorize or
   prohibit changes at the intended granularity.

3. Whether actual source-to-rig influence can be
   distinguished from mere dependency or naming similarity.

4. Whether identity, provenance, timing and score semantics
   remain valid or explicitly UNKNOWN across the tested
   boundaries.

5. Whether the candidate representation is a defensible
   foundation for a future conditional Animation Generator,
   within its declared and limited scope.

A negative or invalid result is acceptable.

If a judgment cannot be made, identify exactly which
evidence is missing and why.

Do not proceed to Generator training, Production schema
changes, general Animation Authoring, or any subsequent
Batch.
