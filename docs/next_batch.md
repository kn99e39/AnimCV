# BATCH 117 — Semantic World Pin: Temporal Coherence and Compensating Articulation

DIRECTION

Advance AnimCV toward a Reference-Guided, Reliability-Aware
Video-to-Animation Generator/Editor.

Human-approved Architecture policy: SEMANTIC CONSTRAINT SYSTEM.
The initial supported semantic type remains WORLD_POSITION_PIN,
defined in Blender scene-world metres. This Batch does NOT
authorize another semantic constraint type, a Production policy,
a trained Generator, or a Production schema change.

W116 established bounded evidence that an independent semantic
pin contract, rig binding, solver, Blender evaluator, and
reversible edit state can cooperate on a real rig. It also
exposed a major practical gap: satisfying the endpoint pin
alone can yield abrupt, unnatural, or excessively compensating
arm motion, particularly near straight-arm singularities and
pin-interval boundaries.

W117 investigates that gap as a SINGLE bounded engineering
question, without repairing or reclassifying W116.


PURPOSE

A usable animation editor must preserve the user's explicit
world-position pin while producing motion that can actually be
inspected, edited, and used on a rig.

The first functional World Pin is not sufficient if the
resulting elbow snaps, bends implausibly, or abruptly changes
motion at the start/end of a pin interval.

Evaluate whether a temporally aware choice of authorized
compensating arm rotations can improve animation behavior
relative to the W116 frame-independent solution, WITHOUT
violating the hard pin or overriding the user's original edit.

Do not silently convert temporal smoothness into a new
user-authored semantic constraint. It is a SOLVER QUALITY
OBJECTIVE within the already-approved WORLD_POSITION_PIN
experiment, not an independently supported constraint type.


CENTRAL INTENT

On an unchanged frozen rig and source-backed Animation State,
can a bounded, temporally aware World Position Pin resolver
reduce discontinuity, elbow flips, and excessive compensating
articulation relative to the W116 framewise pin resolver,
while preserving:

(1) evaluated Blender world-position hard-pin compliance;
(2) exactly the user-authorized edit channels and solver DOF;
(3) immutable source observations and versioned state lineage;
(4) explicit infeasible/unsupported/rejected outcomes; and
(5) the ability to revert the edit exactly?

Determine whether temporally coherent pinning is feasible
under this limited ownership model, and identify cases where
joint reach, kinematic singularity, or conflicting objectives
make it impossible.

Do not infer improved animation quality from a smaller
position residual or from smooth numerical curves alone.


CURRENT ASSESSMENT / HUMAN ASSESSMENT

Historical W113, W114 and W115 remain CASE D INVALID.
W111–W115 cumulative code audit is recorded at
docs/AUDIT_W111_W115_2026_11_05.md.

W116 official Worklog is COMPLETE, CASE B PARTIAL SUPPORT.
Its frozen outcome MUST remain B.

Independent Architecture Reviewer examined:
- W116 Worklog and W116 code changes;
- W116 base/contract/Blender evaluation and tests;
- AGENTS.md, PRODUCT_VISION.md, Architecture v3/v2;
- existing W111–W115 audit and current next_batch.

W116 changed only new scratch scripts/tests/Worklog, not src/.
The code's GB4 and GB5 gates substantiate the Worklog's
explanation of two instrument defects:
- GB4 hard-coded a rejected RB_neg lineage input, although
  RB_neg was accepted on all 199 frames.
- GB5 incorrectly required every passing frame to have
  resolver status SOLVED, excluding NO_CHANGE_NEEDED
  capture frames. The resolver's actual acceptance path
  already treats NO_CHANGE_NEEDED as a valid status.
Neither gate may be retrospectively repaired to upgrade W116.

W116 agent-reported evidence (NOT independently re-executed
on Main by this Reviewer):
- 736/736 independently classified feasible real-rig
  per-frame positive cases were accepted; 258/258
  infeasible were rejected; one boundary case excluded.
- Accepted Blender-evaluated pin residual <= 0.98 mm;
  734/736 <= 6.8e-7 m.
- Only authorized upperarm_l/lowerarm_l channels changed
  against the unpinned control, with no reported root or
  outside-domain changes.
- Real-rig elbow flex compensation reached +34 degrees
  for a fixed 5-degree torso edit.
- Some accepted solutions approached a fully straight arm
  and exhibited a bend-plane flip.
- W116's real parent Action already has discontinuities:
  wrist jump up to 0.485 m/frame; its 3x-parent-max
  discontinuity rule was non-discriminative.
- The W116 synthetic hold was symmetric around the
  release boundary, accidentally masking the predicted
  release discontinuity.
- The real source has 199 frames at 50 fps, but the
  frozen W105 Action is keyed at 30 fps. Do not fix
  or hide this historical mismatch in W117.
- W116 output-rig identity is known, while video
  subject continuity is UNKNOWN across real frames.
- W116's Blender renders and numerical artifacts are
  Main-local; this Reviewer did not independently
  inspect or rerun those binaries.

These observations support a temporal-quality feasibility
test. They do NOT approve straight-arm acceptability,
large elbow compensation, unpinned-body side-effect limits,
anatomical production policies, or a new constraint type.

W117 is not a W116 gate-repair Batch. It must answer
the animation usability question above, not merely
recompute a CASE.


HUMAN-READABLE SUMMARY

Direction:
Evaluate temporal coherence of WORLD_POSITION_PIN output
on an actual rig, preserving W116's semantic contract.

Purpose:
Test whether the user can keep a hand pinned during a
meaningful motion edit without an abrupt or visually
implausible compensating arm trajectory.

Central Intent:
Does a temporally aware solver improve actual rigged
animation over a frame-independent pin solver while
preserving hard user intent and edit authority?

Do:
- Reuse frozen W116 real rig, parent, bindings and
  world-position evaluation semantics.
- Build a bounded temporally aware solver candidate
  operating only on authorized arm DOF.
- Compare unpinned, W116-framewise, and temporal
  candidates on identical predeclared edits/targets.
- Use asymmetric synthetic sequences with known rig
  identity, known feasibility and nontrivial entry/
  release motion; include near-singular controls.
- Independently evaluate complete Actions in Blender.
- Measure pin residual, temporal behavior, elbow
  articulation, collateral motion, reversal, and
  solver failure classes.
- Report quantitative and visual evidence separately.

Do NOT:
- Implement orientation/contact/motion-preservation
  as new semantic constraint types.
- Introduce a trained Generator or learned style.
- Modify any Production src/, existing rig, original
  Action, existing Worklog/CASE, FBX pipeline, source
  estimator, model weights, thresholds, or timebase.
- Relax hard pins, alter user edits, move the root,
  change bone lengths, or hide unreachable cases.
- Choose user-facing acceptance policy for straight
  arms, elbow compensation or other unpinned joints.
- Select the next Batch or recommend downstream work.

Decision enabled:
Whether a temporal solution-selection mechanism within
WORLD_POSITION_PIN is a viable next layer of editable
rig animation, and what physical or policy limitations
must remain explicit.


HISTORICAL BASELINE

Repository: kn99e39/AnimCV
Existing branch ONLY: arch/single_frame_first
W116 completion HEAD observed by Reviewer:
a20d7cc94f7798a8a4bc6dbe1ff9f8dc0d33116b

Verify fresh remote/local HEAD, branch and working tree
before starting. Do not assume the observed HEAD remains
current.

Read before implementation:
- AGENTS.md
- PRODUCT_VISION.md
- Architecture_v3_FramePose.md
- relevant Architecture_v2.md sections
- docs/116_WORKLOG_SEMANTIC_WORLD_POSITION_PIN_FEASIBILITY.md
- docs/115_WORKLOG_ANIMATION_BASE_VALIDITY_REPEATABILITY_RESOLUTION_AUTHORITY.md
- docs/AUDIT_W111_W115_2026_11_05.md
- scripts/w116_semantic_pin_base.py
- scripts/w116_semantic_pin_contract.py
- scripts/w116_blender_stages.py
- tests/test_w116_semantic_pin.py
- W105 rig/source evidence and W102 authentic observation
- current docs/next_batch.md

Authenticate W116 and W105 parent fingerprints before
using their outputs as a baseline. Existing W116 local
artifact paths are environment-specific. If the frozen
baseline cannot be reproduced or authenticated, STOP
rather than inventing replacements.

Historical W116 numerical outcomes and CASE are fixed.
Use its framewise algorithm as the CONTROL and preserve
the original test logic; do not patch its GB4/GB5
decision implementation in place.

No branch create/switch/merge/rebase/reset/delete or
force push. No modifications to existing Production
modules, original data, rigs, checkpoints or Worklogs.


ARCHITECTURE HYPOTHESIS

H1: Within a declared reachable and authorized motion
interval, a temporal solver can choose a sequence of
compensating arm rotations that satisfies the same
world-position pins as W116's framewise solver but
improves temporal and articulation quality.

The solver's optimization objective is NOT the semantic
constraint contract. Hard world-position pin compliance
is independently evaluated and cannot be traded away
for smoothness.

The semantic request and its provenance are immutable.
Candidate Animation States are separate from the
source and from each other. Rejected candidates never
advance a state ref.

The same hard pin must be evaluated in actual Blender
scene-world metres, with the W116 provisional 1 mm
diagnostic tolerance unchanged.

The temporal solver may use information from neighboring
frames within a declared interval, but may change only
authorized upperarm_l/lowerarm_l channels. The user-edited
spine channel, object/root transforms and other channels
are immutable relative to the corresponding control.

For multi-frame requests, accept/reject atomically:
a single hard-pin violation or unauthorized change
rejects the entire requested interval.

If temporal optimization cannot improve the result
under these constraints, report that as evidence.
Do not add hidden compensating torso/root changes.


PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE:
- W116 official CASE B and all frozen gate outcomes.
- W116 framewise control implementation and outputs.
- W105 original Action, Rig, rest pose and FBX.
- W102 observations/predictions, clip, hashes and
  subject-identity UNKNOWN status.
- W114/W115 source-preserving evidence and Worklogs.
- Immutable parent state, source time and edit lineage.
- Existing model weights, rig scale, Blender setup.
- W116 hard pin 1 mm diagnostic criterion.
- All previous synthetic and real negative findings.

CHANGE ONLY:
- New isolated W117 scratch scripts and focused tests.
- Scratch temporally aware solver candidates.
- New copied/derived Action and Animation State artifacts.
- W117 declaration, evidence, renders and Worklog.
- New W117-specific test fixtures if needed.

DO NOT TUNE:
- Historical W116 gates, thresholds, CASE, test outputs.
- Rig bone lengths/rest pose or armature object transform.
- S0/J1X ownership or source observations.
- Retargeter, FBX exporter/importer, or frame rate.
- 1 mm pin criterion after outcomes.
- Solver weights/limits on evaluation frames after
  seeing evaluation results.

The Agent may choose an appropriate bounded temporal
optimizer, pole-continuity method or candidate search.
Its objective, regularization, joint-angle limits,
iterations, fallback behavior and runtime budget must
be frozen in a hashed pre-outcome declaration.

If tuning is necessary, use a separate declared
synthetic development split, freeze parameters, and
report the split boundary. No outcome-driven search
over the held-out evaluation cases.


REQUIRED EVIDENCE — PART 1
BASELINE AUTHENTICATION AND EVALUATION CONTRACT

Reproduce or authenticate the W116 parent and
framewise control without modifying W116.

Verify:
- original input/rig/rest hashes;
- source-backed Animation State identity and channels;
- Blender scene-world binding and units;
- actual evaluated joint positions;
- W116 control output for representative frames;
- allowed solver channels;
- Action timing and constant interpolation semantics.

Report which evidence was directly rerun and which
was only inherited from an authenticated artifact.

If the frozen control cannot be reproduced, STOP.

Define a single independent Blender evaluator shared
by both framewise and temporal candidates.
The evaluator must not trust solver-reported residuals.

A status of NO_CHANGE_NEEDED may be a valid passing
frame when its evaluated pin is satisfied. Distinguish
it from an unsupported or nonconverged solver.

Declare all evaluation families, population sizes,
temporal metrics and decision gates before outcome
inspection. Avoid an RB_neg-like sign assumption:
classify feasibility independently, and evaluate
both signs rather than selecting only rejected cases.


REQUIRED EVIDENCE — PART 2
ASYMMETRIC SYNTHETIC SEQUENCE

Construct a small synthetic rig with known identity
and reachable World Position Pins over a nontrivial
interval. Use a deliberately ASYMMETRIC motion
trajectory: entry and release poses must differ
from the hold target and from each other.

Include:
- smooth reference arm motion without pins;
- a meaningful nonzero torso/shoulder edit;
- a hard wrist pin held for a specified interval;
- a release boundary with nonzero predicted drift
  under naive framewise pinning;
- elbow-pole continuity where the chain has slack;
- a near-straight-arm or nearly singular interval;
- an unreachable target and a conflicting/invalid
  request that must be rejected without state change;
- a zero-motion/no-op control not counted as success.

Do not make the target artificially symmetric at the
release, as happened in W116 K8.

Use an independently constructed feasibility oracle
or analytic geometry for known synthetic cases.
Record reach margins, joint limits and DOF.

Evaluate at least two synthetic trajectories with
different motion phase/curvature and one near-singular
case. Predetermine frames and seed.

For every trajectory compare:
U: unpinned user edit;
F: W116 frame-independent pin resolver;
T: temporally aware pin resolver.

All three share the exact parent, edit, target,
interval and Blender rig.

If T cannot satisfy the same hard pin, it fails that
case even if it looks smoother.


REQUIRED EVIDENCE — PART 3
AUTHENTIC REAL-RIG REPLAY

Use the frozen W105 real_s0_only rig and authentic
W102 observation-backed parent, 199 frames.

Preserve original source timebase (50 fps) and the
historical Action frame-rate behavior (30 fps);
compare motion in explicitly declared rig-Action
time units. Do not claim timebase correctness.

Use the same W116 user-edit families as baseline
where meaningful, including both RB_pos/RB_neg,
and predeclared left-wrist hold windows 109–118
and 154–181.

Include the low-evidence wrist windows as labelled
subpopulations, not as GT.

Do not select only frames where T succeeds. Report
all declared frames, independently classified
feasible, infeasible and boundary.

Keep the video subject identity UNKNOWN; only the
output rig identity is known across frames.

For each case report:
- parent, unpinned, framewise and temporal world
  wrist/elbow trajectories;
- hard-pin residual and acceptance per frame;
- authorized arm channel changes;
- preserved spine edit;
- other evaluated joint movement, especially right
  wrist and neck;
- root movement;
- reach slack, joint flex and bend-plane behavior;
- temporal steps at entry and release;
- source state lineage and exact revert.

The existing real Action is discontinuous. Therefore
do not use W116's 3x-parent-max jump threshold as
the sole temporal-quality metric. Compare against
the same framewise and unpinned controls and report
raw absolute values.

Do not claim improved performer fidelity without
true 3D ground truth.


REQUIRED EVIDENCE — PART 4
TEMPORAL QUALITY AND ARTICULATION ACCOUNTING

Define all temporal metrics in the pre-outcome
declaration, including the time unit and sampling.

At minimum:
- per-frame wrist pin residual (max, p95, failures);
- world wrist and elbow displacement per frame;
- elbow-flex angle, signed bend-plane/pole continuity;
- first- and second-order rotation changes of the
  authorized arm channels, with quaternion sign
  ambiguity handled correctly;
- entry/release jump relative to the unpinned and
  framewise candidates;
- near-singularity reach margin and bend flips;
- cumulative/peak compensating arm rotation;
- any non-authorized channel or point change;
- exact number of accepted/rejected interval edits;
- solver iteration, runtime and convergence status.

A lower total variation or acceleration metric is
not automatically better animation: excessive
static holds, unnatural elbow configurations and
unwanted changes can improve a smoothness number.

Include representative Blender rig-only renders or
short animations showing U/F/T from consistent
camera positions, with elbow/wrist visible.
Avoid exposing the private source video.

Qualitative review must explicitly describe:
- what improved;
- what became worse;
- whether the arm appears anatomically plausible;
- whether a pin starts/stops abruptly;
- whether the improvement is merely numerical;
- whether it would plausibly help an animator.

Do not promote a user-facing threshold for acceptable
elbow angle, collateral motion or release behavior.
Report measured trade-offs to the Architecture Owner.


REQUIRED EVIDENCE — PART 5
FAILURE, OWNERSHIP AND REVERSIBILITY

Negative controls must include:
- unreachable target;
- insufficient authorized DOF;
- unsupported binding or wrong rig identity;
- conflicting pins or altered target provenance;
- stale parent;
- solver non-convergence;
- a candidate with a false SOLVED claim;
- unauthorized armature/root/spine modification;
- one valid feasible interval containing a single
  deliberately corrupted pin result;
- rejected interval leaving its ref unchanged;
- successful exact revert to the frozen parent.

Preserve distinct failure classes:
INFEASIBLE_REQUEST, UNSUPPORTED_BINDING,
SOLVER_FAILURE, CONTRACT_VIOLATION,
INVALID_EXPERIMENT.

Do not silently turn a feasible solver failure into
an infeasible user request.

Ensure one bad frame rejects the entire interval
rather than partially committing the successful frames.

Inspect for unintended changes to all stored channels
and evaluated points outside the authorized influence
domain. User-edit side effects present in both U/F/T
are not to be falsely attributed to pin compensation.

The original source and parent files must be
byte-unchanged after the full run.


REQUIRED EVIDENCE — PART 6
CAUSAL COMPARISON AND DECISION GATES

Before looking at held-out outcomes, define a frozen
A/B decision contract that distinguishes:

V — Validity:
authenticated baseline, correct target/binding,
known rig identity, independent Blender evaluation,
predeclared feasible populations, reproducible runs,
and meaningful nonzero edits.

C — Hard contract:
every accepted pin meets the unchanged 1 mm criterion
in Blender; edits are authorized; source and parent
remain unchanged; invalid requests are safely rejected;
revert is exact; no hidden fallback/relaxation.

Q — Temporal usefulness:
compare T against F and U on identical requests,
including asymmetric synthetic entry/release cases,
near-singular failures and real-rig cases.
Require a predeclared, material improvement on at
least one actual temporal or articulation defect
without introducing a worse hard-constraint failure.
Define the quantitative meaning of "material" before
evaluation and include a visual review.

Do not let an aggregate improvement conceal a single
hard-pin violation or unauthorized channel change.

If the quality objective is not met, record a negative
result. Do not adjust thresholds, frame populations,
weights or target positions after outcome inspection.

Suggested interpretation of CASE labels, frozen
in the declaration before results:

CASE A — BOUNDED SUPPORT:
V and C pass, and T shows the predeclared meaningful
temporal/articulation improvement on valid synthetic
AND at least one authenticated real-rig case, with
no newly introduced hard failure. No production
quality claim.

CASE B — PARTIAL SUPPORT:
V and C pass but real-rig improvement is unproven,
mixed, or materially offset by side effects, or the
synthetic benefit does not transfer.

CASE C — NOT SUPPORTED:
V passes but a valid controlled result contradicts
the minimum contract or the hypothesized temporal
improvement under the declared feasible scope.

CASE D — INVALID:
Validity, authentication, evaluation or pre-outcome
experimental contract fails; D supersedes A/B/C.

Preserve W116 B regardless of W117 outcome.

The Agent must not invent a more favorable CASE
after inspecting results.


TESTING

Run focused tests after implementation units.

Test:
- immutable source/parent identity;
- independent Blender world-coordinate evaluation;
- W116 baseline replay parity;
- temporal solver authorization and convergence;
- quaternion temporal metrics (including sign);
- asymmetric synthetic entry/release controls;
- singular and infeasible cases;
- false-SOLVED rejection;
- interval atomicity with one bad frame;
- stale parent and exact revert;
- repeated runs and declared gates.

Run only relevant existing regressions. Full suite
is not required for scratch-only code.

Explicitly distinguish AGENT-RUN, NOT RUN and
Reviewer-independent evidence.


STOP CONDITIONS

STOP and preserve evidence if:
- W116 or W105 baseline cannot be authenticated;
- actual Blender output cannot be evaluated;
- a meaningful temporal test requires changing
  original rig/rest/source or Production code;
- the solver requires root/spine or non-authorized
  changes to satisfy the pin;
- a hard pin must be silently relaxed;
- synthetic controls cannot distinguish a real
  temporal improvement from a symmetric no-op;
- an experimental threshold or quality policy
  must be chosen post-outcome;
- a new Major Architecture Decision is required;
- local/remote branch diverges or an unapproved
  change appears in shared files.

Report the blocking reason and do not expand scope.

A negative quality result is acceptable and does
not authorize further heuristic patches.


FINAL REPORT

Write one W117 Worklog containing:
1. Official CASE and frozen pre-outcome gates.
2. Exact baseline and environment fingerprints.
3. W116 control parity and declared test population.
4. Solver objective, authorized DOF, parameter freeze.
5. Synthetic asymmetric/near-singular results.
6. Authentic real-rig U/F/T comparison.
7. Blender-evaluated pin residuals and failure counts.
8. Temporal, articulation and collateral metrics.
9. Visual review with reproducible rig-only evidence.
10. Atomic rejection, state lineage and exact reversal.
11. Independent evaluator and negative controls.
12. Alternative explanations and quality trade-offs.
13. Source-identity, GT-free and timebase limitations.
14. What is not demonstrated.
15. Changed files, focused tests, hashes and replay.

Separate synthetic correctness, rig output correctness,
temporal usefulness and Architecture viability.

Do not provide Next Work Recommendation, Next
Architecture Decision, Proposed Next Batch, or
Next Implementation Plan. Claude is an Evidence
Producer, not the Architecture Reviewer.

Do not start W118.


COMPLETION CONDITION

W117 is complete only when the evidence permits an
independent Architecture Reviewer to decide:

1. Whether temporal coupling can improve actual
   Blender-rig World Pin animation compared with
   W116's frame-independent solver.
2. Whether hard pins, user edit ownership, source
   immutability and atomic reversal survive the
   temporal resolver.
3. Whether improvement holds on deliberately
   asymmetric synthetic controls and at least one
   authentic real-rig case, or why it fails.
4. Whether near-singular geometry, entry/release
   discontinuity or collateral articulation prevents
   meaningful animation usability.
5. Which unresolved issues are experimental limits
   versus new user-facing policy decisions.

A negative or partial result is valid. Do not
implement additional semantic constraint types,
a trained Generator, Production integration,
FBX repair or Batch 118.
