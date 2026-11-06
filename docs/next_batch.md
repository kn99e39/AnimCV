# BATCH 118 — Automatic Transition Editing with Explicit Edit-Footprint Visibility

DIRECTION

Advance AnimCV toward a reference-guided, reliability-aware
Video-to-Animation Generator/Editor that produces usable, editable
rig animation under user-authored semantic constraints.

The human Architecture Owner has explicitly approved policy B:
the system MAY automatically modify frames outside a requested
hard-pin interval to make the transition more natural, PROVIDED
that the modified interval is explicitly and intuitively visible
to the user.

This is an Architecture POLICY approval, not evidence that
automatic smoothing works, nor permission to alter any arbitrary
Bone, time span, source observation, or Production subsystem.

Only WORLD_POSITION_PIN is in scope. This Batch is a bounded
scratch experiment and must not implement a Production UI,
trained Generator, new semantic constraint type, or schema change.


PURPOSE

W117 showed a genuine editing trade-off. A temporal swivel
optimizer restricted to pinned frames did not establish reliable
animation-quality improvement. Its transition-easing variant
(TE) softened boundary jumps but necessarily changed frames
outside the pin interval. Under W117's frozen side-effect
criterion those changes counted as worse. The human owner has
now authorized automatic transition edits outside the pin
interval, with an explicit, intuitive display of the edit range.

The new decision does NOT make W117 CASE C invalid or
retroactively turn TE into a success. It removes one policy
blocker and permits a new, controlled test of whether a visible,
bounded automatic edit can improve an animator's actual result.

The next uncertainty is whether the solver can deliver a
meaningful improvement, preserve hard pins and user edits,
and truthfully show where and how the animation was changed.


CENTRAL INTENT

Can AnimCV automatically choose a bounded transition-edit
region outside a user's WORLD_POSITION_PIN interval, improve
the actual Blender-rig animation, and expose an accurate,
intuitive, independently verifiable edit footprint that
distinguishes pinned frames, automatically changed frames,
unchanged frames, and failed/rejected edits?

Can it do so without silently modifying unauthorized channels,
relaxing hard pins, corrupting source evidence, or concealing
a solver failure or animation-quality trade-off?


HUMAN ASSESSMENT

Historical official results are frozen:

W113 CASE D INVALID; W114 CASE D INVALID;
W115 CASE D INVALID; W116 CASE B PARTIAL SUPPORT;
W117 CASE C NOT SUPPORTED.

W116 showed bounded functional feasibility of a semantic
WORLD_POSITION_PIN, separate Rig Binding, resolver, independent
Blender evaluation and reversible Animation State on a frozen
W105 rig. Its real-rig feasibility numbers are Agent-run
evidence, not independent Reviewer reruns.

W117 is officially COMPLETE at
01749bfec5543c251d7a91b5c629f1bef2c1181b.
Its declaration and CASE are immutable. W117's T solver
did not materially improve held-out synthetic trajectories,
had mixed real-rig effects, and introduced a new hard failure:
H_RH2_RBpos was feasible and accepted by F but rejected by
T/TE as NON_CONVERGED under the frozen absolute gradient
criterion. A post-hoc Blender check found the rejected
candidate's pin residuals acceptable; this does NOT erase
the declared failure.

W117's TE used four guide frames on either side of a hold.
It visibly softened selected entry/release boundaries but
deviated from unpinned elbow motion on those frames.
The former W117 S1 side-effect cap is a HISTORICAL
EXPERIMENTAL GATE, not an approved user-facing limit.
W118 must measure and disclose such deviations, not silently
reapply that historical cap as a universal policy.

W117's own visual review was agent-authored, covered only
three sheets, and was not independent. A successful rig pin
is not equivalent to anatomically plausible animation.

The authentic W102/W105 clip is GT-free; subject identity is
UNKNOWN on 199/199 frames. Output rig identity is known.
The source is 50 fps, historical Action plays at 30 fps,
and evaluation used frame-indexed 25 fps factory context.
No source-to-Action timing correctness is established.


HUMAN-READABLE SUMMARY

Direction:
Automatic, bounded transition editing with an accurate,
visually inspectable edit footprint.

Purpose:
Test whether the now-approved out-of-pin editing freedom
delivers a practically useful Blender animation, rather
than only a lower numeric smoothness score.

Central Intent:
Can an automatic resolver improve boundary motion while
making its exact edits visible and reversible, without
violating WORLD_POSITION_PIN or user-edit ownership?

Do:
- Keep the pinned interval and actual edit interval separate.
- Select bounded transition-edit frames automatically using
  a predeclared policy; do not require a user-supplied range.
- Independently measure the actual changed frames and Bones.
- Show both intended edit coverage and actual edit footprint
  in a clear timeline plus before/after rig motion.
- Compare against identical unpinned, W116 framewise and
  frozen W117 temporal/TE baselines.
- Evaluate meaningful boundary quality and its costs.
- Preserve hard pins, authorized DOF, source and reversal.
- Include synthetic asymmetric and authentic real-rig cases.

Do NOT:
- Ask the user to approve each automatic margin.
- Implement a Production timeline UI or new editor module.
- Train a Generator or add another semantic constraint type.
- Change source reconstruction, rig, FBX, MotionGraph,
  FramePose, timebase, historical Worklogs or CASE.
- Declare a universal allowed elbow-deviation threshold.
- Treat visually hidden or mislabeled edits as acceptable.
- Hide solver nonconvergence with an unreported fallback.
- Request or propose the next Batch.

Decision enabled:
Whether bounded, transparent automatic transition editing
is a viable practical behavior for the Semantic Constraint
Generator/Editor architecture.


HISTORICAL BASELINE

Repository: kn99e39/AnimCV
Existing branch ONLY: arch/single_frame_first

Expected pre-issuance HEAD:
01749bfec5543c251d7a91b5c629f1bef2c1181b

Read and honor:
- AGENTS.md
- PRODUCT_VISION.md
- Architecture_v3_FramePose.md
- Architecture_v2.md (standing MotionGraph/Rig/Blender scope)
- docs/AUDIT_W111_W115_2026_11_05.md
- docs/116_WORKLOG_SEMANTIC_WORLD_POSITION_PIN_FEASIBILITY.md
- docs/117_WORKLOG_SEMANTIC_WORLD_PIN_TEMPORAL_COHERENCE.md
- scripts/w116_semantic_pin_base.py
- scripts/w116_semantic_pin_contract.py
- scripts/w116_blender_stages.py
- scripts/w117_temporal_pin_base.py
- scripts/w117_temporal_pin.py
- scripts/w117_blender_timing.py
- tests/test_w116_semantic_pin.py
- tests/test_w117_temporal_pin.py
- docs/next_batch.md

Verify the live HEAD, issued instruction blob, Worklog
COMPLETE markers and experiment source identity yourself.
Do not assume an external Claude worker has already started.

The W117 declared experimental artifacts live on Main-local
storage and are not independently authenticated by the
Architecture Reviewer. Verify them on the execution host.
If they cannot be authenticated, STOP; do not fabricate data.

Do not create, switch, merge, rebase, reset, delete branches
or force-push. Do not alter historical source assets.


APPROVED HUMAN POLICY AND SCOPE BOUNDARIES

1. A WORLD_POSITION_PIN defines a HARD CONSTRAINT DOMAIN:
   the requested scene-world position must be preserved
   on exactly its specified frames, under the existing
   provisional W116 1 mm diagnostic acceptance criterion.

2. Automatic temporal transition edits MAY occur outside
   that domain to improve continuity. No separate user
   approval of every extension is required.

3. The automatic selection mechanism MUST be bounded,
   deterministic/reproducible for fixed inputs, and
   declared before examining evaluation outcomes.
   Its search budget, maximum extent and behavior at clip
   boundaries are EXPERIMENTAL parameters, not Production
   defaults or new user-facing policies.

4. The actual modification footprint MUST be clearly
   visible to the user: time intervals and changed channels
   must be inspectable, including any out-of-pin changes.
   Displayed coverage must be based on actual candidate
   output, not only the solver's intended margin.

5. Hard pins are never silently relaxed. The user edit
   itself remains authoritative. Source evidence remains
   immutable. All candidate states are reversible.

6. Unspecified joints are not automatically hard-locked.
   Only explicitly authorized Bone channels may be changed
   by the constraint resolver. Changes to other evaluated
   joints caused by the original user edit must not be
   misattributed to automatic pin compensation.

7. The user has NOT fixed a Production limit on elbow
   deviation, straight-arm use, compensation angles or
   the maximum number of transition frames. Measure those
   trade-offs honestly; do not invent a permanent policy.

8. This approval concerns automatic edit behavior and
   edit-range VISIBILITY. It is not approval to build a
   Production UI, add new constraint types, alter source
   tracking, train a model or change downstream contracts.


ARCHITECTURE HYPOTHESIS

H1: A WorldPositionPin can retain its narrow hard-pin
domain while a separate automatic transition editor
operates on a bounded, explicitly recorded edit domain.

H2: An independent Blender evaluator can derive a
truthful per-frame/per-channel ACTUAL EDIT FOOTPRINT
from the candidate vs the unpinned user-edit control,
and that footprint can be displayed clearly enough
for an animator to understand what the system changed.

H3: Within this ownership model, at least one controlled
automatic transition strategy can reduce meaningful
boundary discontinuity without introducing a new hard
pin, ownership, atomicity or reversal failure.

H1/H2/H3 are separate. An accurate display alone is
not evidence of better animation. A smoother animation
with a false footprint is not a viable editor result.


PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE:
- All historical Worklogs and frozen CASE verdicts.
- W116 and W117 code, declarations, decisions and manifests.
- W105 original Rig, rest pose, Action and source evidence.
- FramePose/MotionGraph/Blender/FBX production behavior.
- Source time, identity and coordinate contracts.
- W116 1 mm provisional hard-pin diagnostic criterion.
- User-edit Bone channels and immutable parent states.

CHANGE ONLY:
- New isolated W118 scripts and focused tests.
- Scratch edit-domain/footprint metadata and renderer.
- Scratch copied candidate Animation States and Actions.
- W118 declaration, outputs, renders and Worklog.
- New scratch diagnostic visualization (not Product UI).

DO NOT TUNE:
- W116/W117 frozen thresholds, solvers, declared gates
  or historical result files.
- Input pose estimates, confidence or source identity.
- Bone lengths, rest pose, original Action or retargeter.
- Timebase, Blender exporter/importer or rig mapping.
- W118 quality gates, selected cases or edit budget
  after the pre-outcome declaration.
- A candidate until it appears successful on selected
  evaluation cases.

Do not modify existing production src/ modules.
Do not rewrite previous scripts/tests/Worklogs.


REQUIRED EVIDENCE — PART 1
SEPARATE PIN DOMAIN, AUTOMATIC BUDGET AND ACTUAL FOOTPRINT

Represent separately:
- requested pin frames;
- the predeclared automatic edit SEARCH BUDGET;
- candidate-declared editable frame domain;
- authorized Bone channels;
- actual changed channel/frame pairs vs the unpinned
  user-edit control U;
- independently evaluated joint displacement vs U;
- candidate status (proposed/accepted/rejected);
- original source/parent identity and reversal lineage.

The algorithm chooses its transition domain automatically.
The user does not need to specify it.

The exact changed range may be non-contiguous. Do not
reduce it to a misleading single continuous interval.
Store both the union envelope and each contiguous run.

Differentiate numerical F-curve differences from material
evaluated pose differences. Report exact changed values
and a declared display resolution; never hide changes
solely because they are smaller than the display threshold.

Report the time axis explicitly as Action frame indices.
Do not convert to seconds using the source's 50 fps while
the historical Action is 30 fps. If showing seconds,
provide the explicit, verified frame-rate interpretation.

If a candidate changes a frame outside its declared
automatic budget, reject it even if the pin is satisfied.


REQUIRED EVIDENCE — PART 2
INTUITIVE EDIT-RANGE VISIBILITY

Produce a standalone scratch visual preview that an
animator can read without parsing raw JSON.

At minimum:
- a full-sequence frame timeline with clear legend;
- hard-pin frames distinctly highlighted;
- automatic out-of-pin changed frames highlighted
  differently from pinned frames;
- unchanged frames and rejected candidates distinct;
- exact start/end frame labels and disjoint runs;
- authorized Bone/channel change coverage;
- before/after motion or rig-proxy animation;
- an indication of how much movement changed, not
  only whether a frame was touched.

Include a side-by-side comparison of U, frozen F,
frozen W117 TE and the W118 candidate for at least
the declared representative cases.

Show the candidate footprint BEFORE committing a
scratch state, then show the ACCEPTed or REJECTed
result with matching state ID. A rejected candidate
must not be presented as applied.

Validate the display against an independently derived
Blender output diff. A timeline generated only from
the solver's intended margin is insufficient.

At least one negative control must deliberately make
the displayed interval or channel coverage incomplete
and verify that the display/contract check fails.

Visual inspection is not a substitute for the numeric
truth check. Clearly state whether the visuals were
reviewed only by the Agent or independently.


REQUIRED EVIDENCE — PART 3
CONTROLLED TRANSITION EDITING

Use the same parent rig, pin targets, user edit,
source frames and independent Blender evaluator for
all compared variants.

Controls:
U — unpinned user edit.
F — frozen W116 frame-independent pin resolver.
T — frozen W117 temporal-only solver, where applicable.
TE — frozen W117 fixed-margin transition easing.

Candidate:
One bounded W118 automatic transition strategy, with
predeclared search budget and objective/selection rule.

The Agent may choose the internal algorithm. A framewise
hard-pin base plus separate transition editor is an
acceptable candidate if justified. Do not require the
known W117 T optimizer as a dependency when it introduces
a nonconvergence failure unrelated to hard-pin feasibility.

Any fallback must be explicitly declared, logged per
request and compared against the baseline. A fallback
that merely returns F is not counted as a quality gain.

No hidden frame expansion, root movement, unauthorized
channel edit, bone stretching, hard-pin relaxation or
source alteration is allowed.

For every candidate, distinguish:
- feasible pin but temporal optimizer failed;
- hard-pin failure after Blender evaluation;
- unreachable or conflicting request;
- unauthorized output or footprint mismatch;
- acceptable output with a quality trade-off;
- valid accepted improvement.

Do not turn solver-convergence tuning into the central
purpose of this Batch. W117 C6 remains historical.


REQUIRED EVIDENCE — PART 4
MEASURE ACTUAL ANIMATION QUALITY AND EDIT COST

On Blender-evaluated output, measure:
- hard-pin residual per frame;
- boundary wrist displacement and step excess over U;
- elbow world-space trajectory and step/acceleration;
- upper/lower arm quaternion angular steps;
- bend-plane flips and near-straight cases;
- peak and mean elbow deviation from U;
- changed frame count and number of disjoint edit runs;
- per-Bone changed channel/frame counts;
- extra frames outside pin interval;
- root and non-arm collateral vs U;
- solver runtime, failures and fallback usage;
- reversal and source-integrity results.

Report both absolute output motion and the extra
compensation relative to U. A source jump already
present in U is not automatically a solver defect.

Do not reuse W117's frozen S1 10 mm limit as a new
hard user-policy gate. It is a cost metric in W118.

Predeclare meaningful improvement criteria before
looking at candidate outcomes. Quality criteria must
cover at least boundary discontinuity and introduced
articulation defects; a single scalar smoothness score
is not sufficient. Do not hide worsening in one metric
behind an aggregate improvement in another.

Use visual review to assess whether the observed change
would actually help an animator. Avoid claims of
anatomical plausibility without a reference.


SYNTHETIC + REAL REPLAY

Synthetic:
- Use W117's asymmetric holds H_A/H_B/H_C, including
  the near-singular case and nonzero user edit.
- Include at least one held-out asymmetric transition
  not used to choose candidate parameters.
- Include unreachable and contradictory requests,
  clip-start/end clipping, zero margin, disjoint
  intervals and a display-footprint corruption control.
- Compare no-pin, F, TE and W118 on identical states.
- Include a short multi-frame animation with real
  entry/release behavior, not only static poses.

Real:
- Reuse the authentic W102-derived W105 S0-only rig
  and 199-frame Action, with original evidence.
- Explicitly evaluate holds 109–118 and 154–181,
  plus the feasible W117 real requests that exposed
  improvements and regressions.
- Preserve the W117 H_RH2_RBpos hard-failure case as
  a named regression control: do not relabel it.
- Include source discontinuity at 153→154 as a
  confounder, not a new editing target.
- Evaluate the same side-bend signs, feasible
  populations and independent feasibility oracle
  as the frozen baselines.
- Display a real-rig timeline and representative
  before/after playback with actual edit footprint.

The authentic source has NO ground-truth 3D motion.
Subject continuity is UNKNOWN; only output rig identity
is known. Do not claim performer fidelity or video
identity continuity. Do not silently repair the
50fps-source / 30fps-Action mismatch.


TESTING

Preflight authentication:
- HEAD, branch, instruction blob, historical Worklogs;
- W116/W117 declarations, manifests, baseline identity;
- source and Rig/Action fingerprints;
- exact parent and frozen F/T/TE replay where possible.

Focused tests:
- pin domain vs automatic edit budget separation;
- deterministic selection and boundary clipping;
- actual changed-frame and changed-Bone detection;
- disjoint interval representation;
- coordinate/time/frame labeling;
- displayed timeline vs independent Blender diff;
- false/missing edit-range disclosure rejection;
- unauthorized frame/channel modification rejection;
- hard-pin evaluation independent of solver;
- feasible/infeasible and nonconvergence distinctions;
- atomic commit, stale parent and exact revert;
- source immutability and repeatability.

Run focused relevant existing regressions only.
Do not run unnecessary full suites for scratch code.
Report unrun tests as NOT RUN, never PASS.

Use independent Blender output evaluation rather than
only Python FK or candidate internal residuals.


STOP CONDITIONS

STOP with evidence preserved if:
- a required historical baseline cannot be authenticated;
- W117 is not officially COMPLETE or next_batch no longer
  points to W118 at the declared instruction blob;
- the selected rig cannot bind the pin unambiguously;
- actual frame/channel changes cannot be measured;
- the proposed display cannot be checked against output;
- hard pins require silent relaxation;
- the candidate requires changing Production code,
  original Action, source, rig or timebase;
- no predeclared positive control has authorized DOF;
- a new major user-facing Architecture policy is required;
- result selection requires post-hoc threshold tuning.

Do not expand into a full UI, animation-style generator,
additional semantic constraints or Production migration.

Do not issue W119.


DECISION CONTRACT

Freeze populations, solver parameters, footprint display
criteria, improvement criteria and CASE rule before any
evaluation of the W118 candidate.

CASE A — BOUNDED SUPPORT:
The automatic transition editor improves declared
meaningful boundary motion on predeclared synthetic
AND real-rig cases, with independently verified hard
pins, authorized channels, complete truthful footprint,
atomicity and reversal, and no new required hard failure.
Visual evidence supports the claimed editing utility.
Do not infer general Production readiness.

CASE B — PARTIAL SUPPORT:
The separation and disclosure contracts hold but quality
gain is mixed, the visual evidence is inconclusive,
or important execution/quality limits remain.

CASE C — NOT SUPPORTED:
Valid controlled evidence contradicts a required minimum
guarantee (hard-pin safety, truthful footprint, authorized
scope, atomicity) or no declared candidate demonstrates
useful improvement under the frozen rule.

CASE D — INVALID:
Source authentication, baseline reproduction, declared
controls, coordinate/time contracts, or repeatability
fails so an architecture verdict is untrustworthy.
D supersedes other CASE outcomes.

Preserve W116 B and W117 C, regardless of W118 result.


FINAL REPORT

Write a W118 Worklog with:
1. Official CASE, frozen gates and declaration hash.
2. Source, rig, branch, baseline and blob fingerprints.
3. Approved policy vs experimental implementation.
4. Separate pin, search-budget, planned and actual domains.
5. Candidate algorithm, predeclared parameters and runtime.
6. Full synthetic and authentic real-rig A/B comparisons.
7. Independent Blender pin and change-footprint accounting.
8. Timeline visuals, labels, reviewed frames and limitations.
9. Animation-quality benefits and measured edit costs.
10. Hard failures, negative controls and fallback accounting.
11. Source preservation, lineage and exact reversal.
12. Timing, identity, GT-free and quality non-claims.
13. Alternative explanations and unresolved uncertainties.
14. Exact files, tests, environment and replay commands.

Distinguish CODE evidence, Agent-run outputs, visual
judgments and unverified Main-local artifacts.

Do not claim that a pretty timeline proves animation
quality. Do not claim that a smooth arm motion is faithful
to the source performer.

Do not include:
- Next Work Recommendation;
- Next Architecture Decision;
- Recommended Next Experiment;
- Proposed Next Batch;
- Next Implementation Plan.

Claude is an Evidence Producer, not the Architecture Owner.


COMPLETION CONDITION

W118 is complete only when the evidence allows the
Architecture Reviewer to determine:

1. Whether a WORLD_POSITION_PIN hard interval and an
   automatically chosen out-of-pin edit interval can
   coexist without confusing their semantics.

2. Whether every actual automatically modified frame
   and Bone can be truthfully exposed in an intuitive
   user-inspectable timeline, before acceptance and
   after accept/reject.

3. Whether the automatic transition edits improve
   meaningful Blender-rig motion on controlled synthetic
   and authentic replay, not just internal solver metrics.

4. Whether hard pins, user edits, authorized DOF,
   immutable evidence, atomicity and exact reversal
   survive the automatic extension.

5. Whether the added animation-quality benefit justifies
   the measured extra edits under the APPROVED policy,
   without inventing unapproved Production thresholds.

Negative results are legitimate. If any decision is
blocked, report exactly what evidence is missing and why.

Do not start W119, implement a Production UI, train a
Generator, or expand the semantic constraint taxonomy.
