# BATCH 116 — Semantic Constraint Architecture: World-Space Joint Position Pin Feasibility

DIRECTION

Advance AnimCV toward a Reference-Guided,
Reliability-Aware Video-to-Animation Generator.

The human Architecture Owner has approved a Semantic
Constraint Architecture as the long-term basis of
user-directed Animation Editing.

The first bounded capability is:
WORLD-SPACE JOINT POSITION PINNING.

Investigate whether a semantic constraint can be expressed
independently of the solver and enforced in an actual
rigged animation, while preserving immutable source
evidence and reversible edit history.

This Batch must produce functional animation-editing
evidence, not another isolated numeric-contract exercise.

Do not implement the full constraint system.
Do not train a Generator.


PURPOSE

AnimCV's objective is editable, production-usable
animation, not merely 3D pose reconstruction.

Users must eventually be able to issue intentions such as:
- Keep this hand at this position.
- Preserve this segment of the motion.
- Prevent foot sliding.
- Change the lower-body style without changing the upper
  body's intended action.

These are different semantic constraints.

The approved architecture must represent them distinctly
rather than using one generic Bone Lock.

W116 focuses exclusively on World-Space Joint Position
Pinning, while defining boundaries that can later support
other constraint types.

The result must establish whether this abstraction
enables a meaningful animation edit in the existing
AnimCV rig environment.


CENTRAL INTENT

Can a solver-independent Semantic Constraint describing
a joint's desired world-space position be enforced on
a real rigged Animation State, by adjusting only
authorized degrees of freedom, without modifying source
evidence or silently violating user constraints?

Can the system also detect when a requested edit is
unreachable, contradictory, unsupported by the rig,
or outside the authorized modification domain?

Determine whether Semantic Constraints provide an
architecturally viable separation between user intent,
animation generation, rig adaptation, and output
validation.

Do not equate mathematical constraint representation
with successful animation editing.

Do not equate a solver producing output with the
requested user constraint actually being satisfied.


CURRENT ASSESSMENT

Historical findings:

W113: CASE D INVALID.
Detector score [0,1] was an invalid assumption.
2D detector score is not calibrated 3D reliability.

W114: CASE D INVALID.
Repeatability and quaternion precision instruments failed.
Scratch source-preserving, reversible editing showed
bounded positive diagnostic behavior.

W115: CASE D INVALID.
Its V8 assumption incorrectly required mathematically
invariant input to cause zero changed float32 stored
values after floating-point evaluation.

W115 also reported a material user-intent failure:
a locally accepted edit could move world-space locked
joint positions by up to approximately 3.4 mm.

Conversely, tiny numerical changes to stored local
quaternions could occur without measured world-space
joint displacement.

Therefore:
- Local channel-value preservation is not equivalent
  to world-space joint-position preservation.
- Output constraints must be evaluated in their own
  physical/semantic domain.
- Source identity, numerical representation and actual
  rig behavior must remain distinguishable.

The cumulative W111-W115 audit has been completed.
No earlier CASE is to be retrospectively changed.

The immutable-source + versioned-edit representation
remains a scratch architectural hypothesis, not a
production implementation.


HUMAN-READABLE SUMMARY

Direction:
Build and test a minimal World-Space Joint Position
Constraint using a scratch animation-editing prototype.

Purpose:
Determine whether meaningful user-directed edits can
be implemented without losing source information or
violating the user's intended positional lock.

Central Intent:
Is semantic, output-space constraint enforcement viable
as an independent contract between editable motion
and the rigged output?

Do:
- Define a minimal semantic constraint contract.
- Validate semantic joint-to-rig bindings.
- Apply controlled source-backed animation edits.
- Use a bounded candidate constraint resolver.
- Measure actual Blender world-space joint positions.
- Compare constrained and unconstrained outcomes.
- Include feasible and infeasible requests.
- Validate preservation, rollback and output locality.
- Report real-rig and synthetic evidence separately.

Do NOT:
- Implement the whole Semantic Constraint framework.
- Introduce a trained Generator or style model.
- Change production Rig, FK/IK, export, source data,
  FramePose or MotionGraph schemas.
- Assume all joints are controllable.
- Manufacture source identity or confidence.
- Resolve constraint failure using hidden fallback.
- Select future research tasks.

Decision enabled:
Whether semantic world-position pinning offers a
defensible functional foundation for an Animation
Generator/Editor.


HISTORICAL BASELINE

Repository:
kn99e39/AnimCV

Existing branch ONLY:
arch/single_frame_first

Previously observed HEAD:
c6f45e436500af73279a681fa0a5bec4d2290406

Verify the current remote and local state before work.
Do not assume the observed HEAD is still current.

Read:
- AGENTS.md
- PRODUCT_VISION.md
- Architecture_v3_FramePose.md
- Architecture_v2.md
- W113, W114, W115 Worklogs
- docs/AUDIT_W111_W115_2026_11_05.md
- W102-W105 source and rig evidence
- W114/W115 scratch Animation Base and rig instruments

Architecture precedence remains unchanged.

Preserve all historical CASE verdicts, checkpoints,
inputs, datasets, models, source images, original Actions,
rigs, numerical contracts, and evaluation artifacts.

Use existing source data and the frozen W105 rig
configurations wherever possible.

Do not create, switch, merge, rebase, reset or delete
branches. Do not force-push.

No existing production module under src/ may be modified.


APPROVED ARCHITECTURE POLICY

The user-approved policy is:

SEMANTIC CONSTRAINT SYSTEM.

Constraints describe the desired animation behavior,
not merely internal Bone values.

Different semantic requirements must remain distinct:
- World position preservation.
- Orientation preservation.
- Original motion preservation.
- Contact preservation.
- Editing-region restrictions.
- Soft style or motion preferences.

Only WORLD_POSITION_PIN is within the implementation
and validation scope of this Batch.

A semantic constraint is independent of the algorithm
used to satisfy it.

A solver or future Generator may propose a new Animation
State, but an independent evaluator must check whether
the requested constraints are satisfied.

Source evidence and historical animation states are
never overwritten by constraint solving.

A hard user constraint takes precedence over a generated
motion preference.

An impossible request must produce an explicit rejection
or unsatisfied result.

No hidden constraint relaxation is permitted.

Unspecified joints are not automatically hard-locked.
However, the solver must have an explicit authorized
degree-of-freedom domain, and effects outside the
declared output domain must be measured and reported.


ARCHITECTURE HYPOTHESIS

H1:

A minimal semantic constraint contract, combined with
a bounded rig-aware resolver and independent world-space
output evaluator, can enforce selected joint-position
locks while permitting meaningful animation modification.

The architecture must distinguish:

1. Immutable Source Evidence
   Original video-derived observations and reconstruction
   hypotheses, with source identity.

2. Editable Animation State
   Versioned candidate animation values and parent lineage.

3. Semantic Constraint
   A declarative description of a user-desired property.

4. Rig Binding
   Verified correspondence between semantic joints
   and evaluated points on a specific rig.

5. Constraint Resolver
   A bounded mechanism that may change authorized rig
   degrees of freedom.

6. Constraint Evaluator
   An independent measurement of actual output behavior.

7. Edit Commit / Rejection
   Accept only a valid candidate state, or preserve
   the previous state with a complete failure report.

These responsibilities must be distinguishable even
if the scratch implementation uses fewer physical modules.

Do not assume H1.


MINIMUM SEMANTIC CONSTRAINT CONTRACT

Implement only the information necessary for a
WorldPositionPin constraint.

Required semantics:

- constraint identity;
- constraint type = WORLD_POSITION_PIN;
- semantic joint identity;
- verified rig binding identity;
- target coordinate frame;
- target position, with explicit units;
- source of the target position;
- evaluation frame or declared interval;
- hard constraint status;
- authorized solver degrees of freedom;
- relevant source state or parent state identity;
- exact evaluation result and failure reason.

For this Batch, WORLD means the actual Blender scene
world coordinate system, including the Armature Object
transform and posed skeleton.

It does NOT mean the physical world coordinate system
reconstructed from monocular video.

A position captured from an existing rig pose must
record the source frame and evaluated rig state.

The target must not silently move when the Armature
Object transform changes.

The mapping of a semantic wrist or ankle to a Bone
head, tail or other rig evaluation point must be
verified, not inferred from naming similarity alone.

Do not define unimplemented future constraint types
as operationally supported.

A small extension point is acceptable.
A general-purpose rule engine is not authorized.


WORLD-POSITION PIN SEMANTICS

A pin requests that an evaluated joint position match
a declared target in scene-world metres.

Measure the residual:

    r = || P_world_after - P_world_target ||

P_world_after must come from the final evaluated rig,
not solely from a pre-rig 3D pose array or local F-curve.

For the first experiment:
- use reference positions captured before editing;
- use hard pins;
- restrict the solver's changeable Bone channels;
- preserve the initial source and unchanged rig;
- treat unspecified joints as unconstrained but not
  automatically authorized for arbitrary modification.

Do not equate "hard" with "mathematically guaranteed
solvable".

The solver must reject or report unreachable targets,
conflicting constraints, missing degrees of freedom,
invalid mappings and output-contract violations.

Record both position residuals and every resulting
side effect on other evaluated joints.

Declare the experimental positional acceptance target
before outcome inspection. Use a concrete diagnostic
target of 0.001 m (1 mm) for feasible 0.01 m
controlled-edit cases, with separate accounting of
numerical evaluation error. This is a provisional
experiment criterion, NOT an approved Production
tolerance.

Do not adjust it after observing results.

Report exact residuals even when the criterion passes.


PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE:
- FramePose observations and reconstruction predictions.
- W114/W115 source-preserving representation baselines.
- Existing W105 source-to-rig behavior.
- Original Blender Action and rig.
- Original model parameters and weights.
- Original source time and source identity.
- Existing historical Worklogs and decisions.
- All production modules and existing schemas.

CHANGE ONLY:
- New isolated W116 diagnostic scripts and tests.
- Scratch Semantic Constraint representation.
- Scratch rig-binding validation.
- Scratch-only candidate constraint resolver.
- Scratch-only editable Animation States.
- Scratch-generated Action copies or rig evaluations.
- W116 evidence and Worklog.

DO NOT TUNE:
- Detector thresholds or confidence.
- S0/HMR/J1X predictions.
- Original rig bone lengths.
- Rig rest pose and source calibration.
- Source frame rate.
- FBX exporter/importer.
- Existing pose smoothing or estimation behavior.
- W115 numerical contracts.
- Evaluation tolerance after results.

The Agent may select an appropriate bounded solver
algorithm and data structures.

Solver parameters, allowed bones, joint-angle limits,
failure thresholds and test interventions must be
declared before examining candidate outcomes.

Do not select successful frames post-hoc and present
them as the entire population.


REQUIRED EVIDENCE — PART 1
SEMANTIC JOINT-TO-RIG BINDING

Inspect actual W105 rig ownership and Blender-evaluated
skeleton point locations.

At minimum identify:
- left wrist;
- left elbow;
- left shoulder;
- right wrist where available;
- relevant root and torso transforms.

Verify how a semantic joint maps to the evaluated rig.

Measure the frozen baseline's world positions.

Check the effect of:
- Armature Object transform;
- parent Bone transforms;
- local Bone rotations;
- rig scale and coordinate conversion.

Explicitly distinguish an actual missing controllable
right-arm channel from an available left-arm chain.

A semantic name match does not prove ownership.

A missing or ambiguous joint binding must fail safely.


REQUIRED EVIDENCE — PART 2
MINIMUM CONSTRAINT RESOLVER

Create a bounded scratch resolver for feasible
joint-position requests.

The implementation technique is not prescribed.
A suitable IK or constrained kinematic optimizer may
be used if justified.

The resolver must operate on a copied editable state
and authorized rig degrees of freedom.

It must not directly modify immutable observations.

For each request:
- load exact parent state;
- bind constraint to the rig;
- identify authorized degrees of freedom;
- evaluate the unconstrained proposed edit;
- attempt to enforce the world pin;
- evaluate final pose in Blender;
- verify hard constraints and change scope;
- accept atomically or reject without changing parent.

Measure solver changes independently of semantic
constraint declarations.

Do not bypass the evaluator by assigning the target
position directly as the reported result.

Do not move the whole character root to satisfy a
hand pin unless the experimental authorization
explicitly permits root movement.

Do not lengthen bones, modify the rest rig, teleport
unrelated joints or silently change original Action
timing to satisfy a pin.

Declare a bounded execution and convergence policy.
Report failure rather than running unrestricted retries.


REQUIRED EVIDENCE — PART 3
CONTROLLED A/B RIG EDITING

Use the same frozen source pose, rig, user edit and
time state for two comparisons:

CONTROL:
The requested animation modification without a
world-position pin.

CANDIDATE:
The same modification with the semantic
WorldPositionPin enforced.

The solver and rig evaluator must otherwise be
identical between both conditions.

Use meaningful nonzero pose edits. A zero-change
edit or an already-satisfied no-op cannot establish
that pin enforcement works.

Test feasible cases where:
- the left hand can be kept in place by changing
  authorized arm joints;
- a shoulder/torso-related edit affects the hand;
- a world-position pin requires compensating Bone
  rotations.

Use the existing W115 observations as historical
reference, including the demonstrated divergence
between local and world locks.

Do not claim the W115 right-wrist case is solvable
with the frozen left-arm-only degrees of freedom.

Also include explicit negative cases where the
available rig cannot satisfy the requested constraint.

Report:
- before-edit world position;
- desired target;
- unpinned edit position;
- pinned edit position;
- unpinned and pinned position residuals;
- affected Bone channels;
- changed other-joint world positions;
- root movement;
- solver convergence and failure status;
- reversible state lineage.

The user intent is satisfied only if the evaluated
world position meets the declared criterion and
other applicable constraints remain satisfied.


REQUIRED EVIDENCE — PART 4
SYNTHETIC FEASIBILITY AND NEGATIVE CONTROLS

Use a simple known two-bone kinematic chain with
ground-truth reachability.

Include:
- reachable pin with nonzero initial displacement;
- reachable pin under an authorized shoulder change;
- target beyond chain maximum reach;
- conflicting hard constraints;
- missing or invalid joint binding;
- incorrect coordinate frame;
- incorrect unit conversion;
- locked root or insufficient degrees of freedom;
- target altered after state binding;
- stale candidate parent;
- unauthorized Bone modification;
- solver non-convergence;
- unchanged original source evidence;
- exact revert to parent state.

Negative controls must distinguish:
- a valid but infeasible user request;
- an unsupported rig binding;
- a solver failure;
- a semantic-contract violation;
- an invalid experiment.

These are not interchangeable failure classes.

An expected infeasible request should PASS its rejection
control without being counted as a successful pin.

Do not let solver failure change the immutable input.


REQUIRED EVIDENCE — PART 5
REAL-RIG AND REAL-OBSERVATION REPLAY

Use the frozen W105 real_s0_only rig and authentic
W102 real-observation clip where supported.

Keep the original 199 frames and 50 fps source evidence.

Include authentic low-evidence wrist windows
109–118 and 154–181 as diagnostic populations.

Perform edits on copied rig states only.

The original source is GT-free.
Do not claim a new animation is closer to the video's
true 3D motion without ground-truth evidence.

Per-frame tracked subject continuity remains UNKNOWN.
Do not claim verified source-person continuity across
the entire clip.

For cross-frame and interval-oriented tests, distinguish:
- known synthetic rig identity;
- known output rig identity;
- unverified video subject identity.

Use a short controlled synthetic sequence with a
known rig identity to test maintaining a fixed
world-position pin over multiple frames.

Report temporal continuity and whether the constraint
creates discontinuities at interval boundaries.

Authentic real-clip replay can validate output
joint positions and rig behavior, but not verified
cross-frame actor correspondence.

Use the frozen mixed W102 configuration as a negative
ownership control when appropriate:
S0 arm values do not drive its J1X-owned arm.

Do not repair source tracking, timebase or retargeting
within W116.


REQUIRED EVIDENCE — PART 6
ACTUAL BLENDER OUTPUT VALIDATION

Independently evaluate the generated scratch Action
on the unchanged frozen rig.

Use the evaluated Armature Object/world transforms
and correctly bound joint positions.

Compare these against the declared constraint targets.

Do not infer success solely from:
- a scratch solver's internal state;
- a 3D pose array;
- identical local F-curves;
- generated keyframe count;
- a successful Blender export;
- a visually similar screenshot.

For each representative real-rig edit, report
target error and changed joint positions in metres.

Include useful visual review of the before,
unconstrained and constrained motions.

Inspect:
- wrist/hand drift;
- unexpected shoulder or elbow motion;
- bone lengths;
- abrupt articulation;
- joint bend or flip;
- root displacement;
- frame-to-frame discontinuity in synthetic sequences.

Do not declare production animation quality based
on satisfying one point constraint.

Do not automatically add secondary heuristics to
hide unnatural solutions.

Report any unnatural result as a trade-off or failure.


QUANTITATIVE ACCOUNTING

Report, with explicit denominators:

- requested constraints;
- valid and invalid constraint specifications;
- bound and unbound joint identities;
- feasible and infeasible cases;
- resolved and unresolved hard constraints;
- baseline world-position residuals;
- candidate world-position residuals;
- criterion-pass and criterion-fail cases;
- accepted and rejected edits;
- intentionally changed Bone channels;
- unintended channel changes;
- unchanged and changed world-space joints;
- root movement;
- source evidence preserved;
- successful exact reversals;
- solver failures and reasons;
- temporally discontinuous frames where evaluated;
- source identity verified vs UNKNOWN cases.

Maintain separate result groups:
1. Synthetic contract correctness.
2. Synthetic kinematic feasibility.
3. Real-rig Blender output behavior.
4. Authentic video-derived replay.
5. Animation usability observations.
6. Architecture viability.

No aggregate PASS count may conceal failure of a
required hard constraint.


QUALITATIVE REVIEW

Include at least:
- one successful left-wrist world pin;
- the identical edit without the pin;
- one case where compensating arm movement is needed;
- one unreachable or contradictory target;
- one case where the rig has no authorized channel;
- one case where local Bone values and world positions
  lead to different preservation judgments;
- one successful revert;
- one temporal or articulation failure, if observed.

Identify source frame, joint, rig channel, target,
before/after result and evidence provenance.

Negative results must remain visible.

Do not describe mock or constrained edits as
learned motion generation.


TESTING

Run focused tests after meaningful implementation units.

Tests must cover:
- semantic constraint serialization and validation;
- coordinate-frame and unit handling;
- verified rig binding;
- solver authorization scope;
- feasible and infeasible targets;
- constraint evaluation independently of solver;
- source immutability;
- edit lineage, stale rejection and rollback;
- representative Blender A/B replay;
- synthetic sequence continuity;
- numerical measurement validity.

Run relevant existing focused regressions where
shared diagnostic contracts are used.

No unnecessary full regression for diagnostic-only code.

Tests not executed must be reported as NOT RUN.

All original model and rig functionality must
remain unchanged.


STOP CONDITIONS

Stop and preserve the evidence if:

- the source or rig baseline cannot be authenticated;
- the intended joint cannot be unambiguously bound;
- world coordinates cannot be evaluated consistently;
- a required candidate change needs Production code edits;
- solver behavior requires changing the original rig;
- hard constraints must be silently relaxed to proceed;
- a required positive control has no real controllable
  degrees of freedom;
- the test cannot distinguish a solver result from an
  actual evaluated rig result;
- a major unapproved Architecture Decision is necessary.

Do not replace a failed constraint by a heuristic pose
correction and call it success.

Do not tune tolerances after observing outcomes.

A feasible request that the selected resolver cannot
satisfy is evidence of a resolver limitation.
It must not be hidden as an invalid input.

Continue unrelated predeclared valid controls only when
doing so does not violate a STOP condition.


DECISION CONTRACT

Declare exact objective gates before inspecting results.

Use the following meanings:

CASE A — BOUNDED SUPPORT

A valid semantic world-position contract is successfully
enforced on predeclared feasible synthetic and real-rig
cases with meaningful nonzero edits.

The evaluated output satisfies the declared positional
acceptance target, the authorized change scope, source
immutability, failure handling and reversal contracts.

No claim of general production viability, style generation
or arbitrary rig support is permitted.

CASE B — PARTIAL SUPPORT

Semantic constraints, rig bindings, and independent
evaluation are valid, but one or more material execution
guarantees remain unresolved or fail in valid cases.

Distinguish solver limitation from a failure of the
semantic representation.

CASE C — NOT SUPPORTED

Valid controlled evidence contradicts a required minimum
architectural guarantee, such as inability to express,
bind, measure, or safely reject user constraints under
the declared scope.

CASE D — INVALID

Source authentication, test controls, rig identity,
coordinate validation or experiment reproducibility
fails such that an architecture verdict cannot be trusted.

D supersedes other CASE outcomes.

Do not reclassify W113, W114 or W115.

A negative result is legitimate.


FINAL REPORT

Write a W116 Worklog that includes:

1. Official CASE and experimental scope.
2. Historical baseline fingerprints.
3. Semantic constraint schema and semantics.
4. Rig binding and world-coordinate definitions.
5. Candidate solver and declared degrees of freedom.
6. Synthetic feasibility results.
7. Real-rig A/B results.
8. Authentic observation replay limitations.
9. World-space error and output-change accounting.
10. Hard-constraint failures and infeasible cases.
11. Animation quality and temporal observations.
12. Edit lineage and reversal evidence.
13. Alternative explanations.
14. Unresolved uncertainties and non-claims.
15. Exact changed files, tests, and reproducibility.

Do NOT claim architecture success automatically.
Report negative results honestly.

Implementation completion, synthetic correctness,
rig output correctness, animation usability, and
Architecture viability are different levels of evidence.

Do NOT provide:
- Next Work Recommendation.
- Next Architecture Decision.
- Recommended Next Experiment.
- Proposed Next Batch.
- Next Implementation Plan.

The Agent is an Evidence Producer.
The human Architecture Owner and Architecture Reviewer
decide subsequent work.

Do not start W117.


COMPLETION CONDITION

W116 is complete when controlled evidence can determine:

1. Whether a user-authored world-position constraint can
   be represented independently from its solving method.

2. Whether the constraint can be enforced on an actual
   animated rig, rather than only within a pose array.

3. Whether authorized compensating Bone movements can
   preserve a pinned joint's world-space position during
   meaningful edits.

4. Whether impossible, unsupported, conflicting and
   solver-failed requests are correctly distinguished
   and handled without corrupting the parent state.

5. Whether the Source-Preserving Animation Base and
   Semantic Constraint separation provides a defensible
   foundation for future Generator/Editor development.

A negative result is valid.

If any architectural judgment remains impossible,
report exactly which evidence is missing and why.

Do not proceed to additional constraint types,
Generator training, Production integration, FBX repair,
or Batch 117.
