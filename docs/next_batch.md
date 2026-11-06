# BATCH 119 — Endpoint-Space Transition Compensation with Isolated Real-Rig Baselines

DIRECTION

Advance AnimCV's human-approved Semantic Constraint Architecture toward
usable, inspectable, reversible Blender-rig animation editing.

The approved policy remains:
- WORLD_POSITION_PIN is a hard Blender scene-world positional constraint.
- Adjacent frames MAY be edited automatically to smooth a pin transition.
- The planned edit domain and actual changed frames/Bones MUST be visible.
- Source evidence, user-authored edits, and other hard constraints retain
  their ownership; all edits must be bounded and reversible.

W119 is an isolated architecture-evidence experiment. It is not a
Production migration, general-purpose Editor, or learned Generator.

Do not modify any existing production source module or historical Batch.


PURPOSE

W118 has officially completed as CASE D INVALID. Its prospective V4 gate
incorrectly assumed that each frozen W117 segment request had a separate
frame-independent F state. In fact, F/T segment candidates were stored
as family-shared states: a segment's Blender footprint included the
pin frames of sibling segments. Six segment quality windows therefore
had contaminated F comparisons. The independent footprint evaluator
correctly detected those changes; the declaration's population
assumption was wrong.

This is not a reason to erase W118's negative evidence. Even if V4 were
mechanically ignored, its own frozen rule would yield C: synthetic
near-straight regression and no qualifying real hold improvement.
W118's X rotation-offset transition editor produced large synthetic
boundary gains but mixed real-rig results, introduced near-straight
frames and a bend flip, and often saturated its eight-frame budget.
The real source also has pre-existing discontinuities and a nearly
straight arm. Smoothing a local quaternion offset does not guarantee
smooth wrist displacement when the parent motion changes.

At the same time, W118 reported bounded evidence for distinct pin,
budget, planned and actual edit domains, truthful Blender-derived
footprint disclosure, source preservation, atomic rejection and
reversal. Those are agent-run observations, not independently rerun
Reviewer results.

The next useful question is MECHANISTIC and PRODUCT-RELEVANT:
can a transition method guided by the desired end-effector motion,
rather than merely fading a fixed local rotation offset, preserve
the hard pin and visible edit footprint while producing a genuinely
better real-rig transition without new articulation failures?

Clean per-request baselines are a mandatory precondition for that
question, not a stand-alone gate-patching Batch.


CENTRAL INTENT

Under independently isolated, source-identical per-request baselines,
can a bounded WORLD-SPACE END-EFFECTOR-GUIDED transition candidate
improve the actual Blender-rig animation beyond W118's local
quaternion-offset candidate, while preserving hard WORLD_POSITION_PIN,
authorized channels, truthful visible footprint and exact rollback?

In particular, does representing the transition in end-effector
world space prevent the near-straight / bend-flip regressions and
source-motion confounding seen in W118, or are those failures
intrinsic to the current two-bone rig and source trajectory?

A valid negative answer is a useful architecture result.


HUMAN ASSESSMENT

The Architecture Owner has approved:
1. Semantic Constraint System as the user-intent contract.
2. WORLD_POSITION_PIN as the initial implemented constraint.
3. Automatic bounded edits outside the pin interval.
4. An intuitive, truthful display of planned and actual edits.

The Owner has NOT approved:
- a permanent transition budget, numerical smoothness threshold,
  allowed elbow deviation, bend-angle policy or straight-arm limit;
- a Production timeline UI, Production schema migration,
  trained Generator, new constraint type or arbitrary root motion.

Historical official verdicts are immutable:
W113 D, W114 D, W115 D, W116 B, W117 C, W118 D.

W116:
bounded Blender world-pin feasibility, source-preserving scratch
states and atomic edits; temporal usability not established.

W117:
temporal swivel smoothing did not improve held-out synthetic cases
materially, had mixed real effects, and introduced a frozen
NON_CONVERGED rejection of a feasible request.

W118:
completed at HEAD 913bac2193cd1d2dac508a67a44c45ec9a871ea4.
Worklog:
docs/118_WORKLOG_AUTOMATIC_TRANSITION_EDIT_FOOTPRINT_VISIBILITY.md.
Its frozen V4 baseline-scope error yields CASE D. The shared-state
segment comparisons are NOT admissible as isolated per-request A/B.
Its X editor improved synthetic boundary compensation but failed
declared real quality gates; near-straight frames and bend flips
were introduced in selected cases. No retroactive reclassification.

All Blender numbers, tests, screenshots, visual assessments and
local manifests in W116–W118 were produced by the external Agent.
The Reviewer inspected repository code and Worklogs but has not
independently executed the Main-local Blender experiment.

The W111–W115 cumulative audit is complete at
docs/AUDIT_W111_W115_2026_11_05.md.
The next cumulative audit is W120 (W116–W120), before W121.


HUMAN-READABLE SUMMARY

Direction:
Test a genuinely different, end-effector-space transition mechanism
with clean per-request real-rig controls.

Purpose:
Determine whether automatic, visible transition edits can improve
usable Blender animation rather than merely reduce a synthetic
compensation metric.

Central Intent:
Can a world-space wrist trajectory guide produce better real-rig
pin entry/release behavior than W118's rotation-offset fading,
without introducing articulation or ownership regressions?

Do:
- Authenticate and preserve W116–W118 baselines.
- Construct independent, isolated per-request F and X controls.
- Use the same parent, user edit, rig, hard pin and authorized DOF.
- Evaluate one bounded end-effector-guided candidate against F/X.
- Match temporal budgets and disclose actual Blender footprints.
- Distinguish raw motion discontinuities from edit-induced changes.
- Measure elbow/flex/flip costs and view actual rig animation.
- Test synthetic held-out and authentic real-rig populations.
- Preserve source, historical decisions and exact reversal.

Do NOT:
- Change W118's V4 gate, decision, CASE or shared historical states.
- Count contaminated W118 segment A/B comparisons as clean.
- Re-label W117 NON_CONVERGED as solved.
- Fix the real clip's source discontinuities or timebase.
- Tune thresholds or choose examples after seeing outcomes.
- Train a Generator or implement a Production UI/schema.
- Expand beyond WORLD_POSITION_PIN or modify the rig.
- Ask the Agent to recommend the next Batch.

Decision enabled:
Whether endpoint-space transition ownership is a viable next
architecture mechanism for useful Semantic Constraint editing,
or whether current rig/observation limitations dominate.


HISTORICAL BASELINE

Repository: kn99e39/AnimCV
Existing branch ONLY: arch/single_frame_first

Last observed HEAD:
913bac2193cd1d2dac508a67a44c45ec9a871ea4

The current instruction was W118 at blob
2468904ff2e087fcff65c09729014d1968e6936f.

Verify actual HEAD, branch, instruction blob, last Worklog and
input provenance before any experiment. Do not assume these
historical values remain current after W119 issuance.

Read:
- AGENTS.md
- PRODUCT_VISION.md
- Architecture_v3_FramePose.md
- relevant Architecture_v2.md
- docs/AUDIT_W111_W115_2026_11_05.md
- W116, W117 and W118 Worklogs and frozen declarations
- W116 world-pin rig adapter, Blender evaluator and state store
- W117 temporal metrics and frozen F/T/TE replay
- W118 transition_edit, derive_footprint, display_spec,
  check_display, assess and decision code
- corresponding focused tests and W105 source-to-rig evidence

The official W118 COMPLETE marker must be present:
AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 118

No other Batch 119 Worklog may already exist.

Do not create, switch, merge, rebase, reset or delete branches.
Do not force-push. Do not overwrite a concurrent Agent's work.

No existing src/ Production module, normative Architecture document,
W116–W118 script/test/Worklog, frozen declaration, original
Action, source data, model, checkpoint, rig or FBX pipeline may
be modified.

All new implementation belongs in isolated W119 scratch
scripts/tests and a new W119 Worklog.


ARCHITECTURE HYPOTHESIS

H1 — End-effector-space ownership:

For transition frames outside a hard-pin interval, explicitly
planning a world-space wrist trajectory relative to the
unmodified user-edit motion, then resolving it on the existing
two-bone chain with authorized DOF, can avoid at least some
failure modes of fading a fixed local quaternion offset.

The wrist trajectory is a GUIDE, not an additional hard pin
outside the requested interval. The requested pin remains hard.

This hypothesis must be falsifiable against:
- source-motion discontinuity;
- reachability and near-straight singularity;
- elbow swivel / bend-plane ambiguity;
- solver convergence or branch flips;
- insufficient authorized degrees of freedom;
- unacceptable collateral articulation;
- excessive or hidden edit footprint.

The new candidate is labelled Y. It is a scratch test of a
mechanism, not a permanent Architecture implementation.

The Agent may choose a bounded IK/kinematic method and
regularization, but must predeclare its formulation, exact
parameters, numerical criteria, DOF and fallback policy
BEFORE evaluating the held-out population.

Do not optimize the candidate by viewing held-out outcomes.


PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE:
- Original video-derived observations and source identity.
- W102 authentic clip and 199-frame observations.
- W105 frozen rig, rest lengths and original Actions.
- W116 per-frame F world-pin resolver and 1 mm
  provisional diagnostic pin criterion.
- W117 frozen T/TE results and convergence failures.
- W118 X algorithm, margins, historical states and CASE D.
- Parent/user-edit motion U, hard-pin targets and authorized DOF.
- Original source fps, Action frame axis and timing mismatch.
- Source-preserving Animation State lineage and reversal.
- Existing negative controls, seeds and evidence manifests.

CHANGE ONLY:
- New W119 scratch baseline-isolation harness.
- New bounded endpoint-guided Y candidate.
- New W119 independent Blender output accounting.
- New scratch display and comparison artifacts as needed.
- New focused W119 tests and W119 Worklog.

DO NOT TUNE:
- W116/W117/W118 historical algorithms or gate results.
- W118 X M_MAX=8, SIGMA_STEP_M=0.010 or recorded margins.
- Historical source, root, rest rig, user edit or retargeting.
- Source 50 fps / historical Action 30 fps.
- Previous decision classifications.
- Evaluation thresholds after held-out results are seen.

For the primary Y-vs-X comparison, Y must use the SAME
per-side temporal edit margins as the frozen X candidate
on the corresponding clean per-request source state.
This isolates the proposed representation mechanism
from the choice of margin width.

If Y cannot use that margin, report UNSUPPORTED or
FAILURE; do not silently widen it.

A separate explicitly declared diagnostic may study Y's
own automatic margin selection, but it cannot replace
the primary matched-budget comparison or establish a
Production default.


REQUIRED EVIDENCE — PART 1
SOURCE AND BASELINE AUTHENTICATION

Verify:
- current branch and HEAD;
- W118 instruction provenance and official COMPLETE marker;
- W116–W118 Worklog/declaration/decision/CASE integrity;
- source/rig/Action hashes and frozen input ownership;
- frozen W117 and W118 local manifests if available;
- W105 Blender-evaluated parent parity where possible.

If Main-local artifacts are unavailable or unauthenticated,
stop rather than manufacturing baselines.

No claim of independent Reviewer execution may be made.
The Agent must label its own Blender and local-manifest runs.

Reproduce the W118 V4 failure mechanism in a read-only
diagnostic: show that a family-shared SEG_<family>__F
contains sibling segment changes. Preserve the failure.

Do not change the frozen V4 gate or rerun W118 decide
with a new criterion.


REQUIRED EVIDENCE — PART 2
CLEAN PER-REQUEST BASELINES

Create isolated scratch per-request states against the
same authenticated U parent:

U_r: unpinned user edit.
F_r: frozen W116 framewise pin resolver applied ONLY
     to the requested pin frames.
X_r: frozen W118 transition-offset mechanism applied
     ONLY to that request, with its original declared
     per-side margins and authorized channels.
Y_r: the new candidate on the same request and budget.

For F_r, prove that its actual changed-frame footprint
is a subset of the request's pin frames, independently
from Blender output.

For X_r, prove the same source, hard pin, rig, edit
authorization and temporal budget as F_r/Y_r.

A family-shared historical SEG state is NOT a valid
isolated F_r. No sibling segment edits are permitted
in the per-request candidate.

Reconstruct all previously contaminated segment
families rather than silently dropping their cases.
If the reconstruction cannot be authenticated or
its equality to the frozen single-frame resolver
cannot be established, mark those cases INVALID
and do not use them in a quality gate.

Add a negative control that deliberately uses the
family-shared F state and confirms that the baseline
isolation preflight rejects it.

Keep W118's historical numbers and decision untouched.

Before any Y held-out evaluation, freeze and hash the
per-request control definitions and the expected
case population.


REQUIRED EVIDENCE — PART 3
BOUNDED WORLD-SPACE TRANSITION CANDIDATE

Y must operate only on the existing left-arm two-bone
chain and the authorized upper/lower arm channels.

Inside pin frames:
- use the same frozen F hard-pin result;
- do not move the pin target or relax the hard pin;
- do not alter the user-edit channel, root or object.

Outside pin frames, within the matched X margin:
- construct a target wrist WORLD-SPACE guide between
  the boundary pinned pose and the original U motion;
- resolve the guide on the unchanged rig with a
  bounded kinematic method;
- explicitly handle reachability, near-straight
  degeneracy and bend-plane branch ambiguity;
- preserve the declared edit-domain contract;
- record residuals and all compensating articulation.

The guide is a soft editing objective, not a hard
constraint outside the pin domain. Failure to reach
the guide may be reported without claiming a hard-pin
failure, but any actual hard-pin failure is critical.

The candidate must distinguish:
- valid and reachable guide;
- unreachable guide;
- missing authorized DOF;
- solver nonconvergence;
- bend-plane ambiguity;
- output contract violation;
- valid but poor animation.

Do not fix reachability by changing bone lengths,
moving the root, editing the source motion, changing
the timebase or adding unauthorized channels.

Do not use unreported fallback to F or X.
A fallback, if tested, must be a separately declared
candidate and cannot be counted as Y success.

Use one prospectively declared Y formulation.
No post-hoc objective/threshold sweep.


REQUIRED EVIDENCE — PART 4
FAIR SYNTHETIC AND REAL-RIG A/B

Use identical parent, U, rig, pin, frame range and
authorized DOF for each F_r, X_r and Y_r comparison.

Synthetic:
- asymmetric holds H_A, H_B, H_C;
- disjoint DJ_B;
- clip-start and clip-end cases;
- near-straight / singular trajectory;
- infeasible hard pin and conflicting constraints;
- at least one genuinely NEW held-out asymmetric
  trajectory with frozen design and no evaluation
  used to choose Y parameters.

Real:
- H_RH1_none and RH1 bend-sign variants;
- H_RH2_RBpos, including the W117 convergence
  regression as an unchanged historical control;
- DJ_R;
- isolated S_RA_pos_* and S_RB_pos_* requests,
  including the six contaminated W118 windows;
- whole-clip requests with no transition availability;
- W102/W105 low-evidence windows 109–118 and 154–181.

Treat multiple bend-sign variants of one RH1 hold
as a correlated family, not independent successes.

Do not exclude an awkward source jump or near-straight
frame because it hurts Y.

Real clip is GT-free and video subject identity UNKNOWN
for all 199 frames. Evaluate known output-rig motion only.
Do not claim fidelity to the performer or true 3D motion.

W118's X numbers may be shown as historical reference,
but the PRIMARY comparison uses reconstructed clean
F_r and X_r with the same per-request ownership.

T/TE may be included as historical secondary controls,
but only if their segment states are equally isolated.
Do not silently use family-shared T/TE in primary A/B.


REQUIRED EVIDENCE — PART 5
INDEPENDENT BLENDER OUTPUT AND FOOTPRINT

Evaluate every accepted candidate on the unchanged
actual Blender rig, not only with Python FK.

Use independently evaluated Blender world positions
for hard pins, wrist guides, elbow and collateral points.

Re-use W118's scratch footprint/disclosure contract,
but do not trust the solver's planned frame list as
the actual changed-frame truth.

Record for every request:
- exact pin frames, budget, planned and actual domains;
- changed frames and channels, contiguous runs;
- any numerical-only changes;
- world-pin residuals and pass/fail;
- wrist guide residuals outside the pin;
- elbow displacement and flex;
- bend-plane flips and near-straight frames;
- raw wrist/elbow steps and source-relative
  compensation steps;
- root and unrelated joint movement;
- accepted/rejected state and reason;
- state lineage and exact reversal.

Actual changed frames must be within declared budget
and planned domain. Unauthorized channels and any
violated hard pin require atomic rejection.

A rejected candidate must not appear as an applied
edit in the timeline.

Provide clear timelines distinguishing requested
pin, planned transition and actual changed frames/Bones,
and a before/after rig animation view.

Correct the W118 filmstrip-band readability defect
in the NEW W119 scratch visualization only.
Do not edit the historical W118 renders.


REQUIRED EVIDENCE — PART 6
MEANINGFUL ANIMATION QUALITY

Evaluate BOTH:
(A) edit-induced compensation continuity relative to U;
(B) raw output motion continuity in the actual rig.

A smaller compensation step can coexist with an
unchanged 0.485 m source discontinuity. Report both.
Do not call unchanged source motion a new edit defect.

Report, on the same matched-budget windows:
- peak and distribution of wrist compensation steps;
- raw wrist steps and their changes vs U;
- elbow compensation steps and acceleration;
- arm rotation step excess over U;
- introduced bend flips and near-straight frames;
- peak/mean elbow deviation from U;
- number of additional modified frames and channels;
- solver failure/rejection rate and runtime.

For near-straight and flip metrics, show the absolute
angle/time series, not only a binary threshold.

The historical W118 5-degree near-straight flag is
a DIAGNOSTIC reference, not a user-facing rule.
Do not declare a Production joint-quality limit.

Predeclared primary improvement test:
- compare Y_r to clean X_r on the same frames;
- B1 wrist compensation peak must decrease by >= 20%
  AND by >= 0.010 m;
- no newly introduced bend flip or near-straight
  crossing relative to clean X_r;
- no material worsening of peak elbow compensation
  step, rotation-step excess or raw wrist discontinuity;
- all hard-pin, domain and atomicity contracts pass.

For the last "material worsening" condition, declare
a numeric experiment-only floor and rationale BEFORE
held-out evaluation; do not tune it after outcomes.

A useful real-rig result must appear in at least
one RH1-family request and one independent RH2/DJ_R
request, not just multiple correlated RH1 variants.
At least one qualifying real result must be visibly
discernible in actual rig playback/filmstrip at a
declared review scale.

A new held-out synthetic case must also meet the
same improvement criterion, and no valid synthetic
control may introduce a new critical hard failure.

If these targets are not met, report the negative result.
Do not reclassify a near-pass or change the thresholds.

The Agent's own visual review is not independent.
State exactly which GIFs/frames were actually inspected.
Do not substitute numerical plots for actual rig review.


REQUIRED EVIDENCE — PART 7
NEGATIVE CONTROLS AND REVERSIBILITY

Predeclare expected results for:
- family-shared segment F erroneously used as F_r;
- a sibling segment change outside the request pin;
- mismatched U parent or rig identity;
- mismatched margin/budget between X_r and Y_r;
- changed unauthorized Bone;
- changed root/object;
- actual changed frame outside planned domain;
- false timeline label or omitted changed frame;
- unreachable guide but feasible hard pin;
- infeasible hard pin;
- contradictory hard constraints;
- solver nonconvergence;
- bend-plane branch ambiguity;
- one corrupted hard-pin frame;
- stale parent;
- rejected candidate displayed as applied;
- exact revert after accepted multi-frame edit.

Expected rejection is not successful animation quality.
A valid but unsolved Y guide must not corrupt the parent.

No silent fallback, mutation of source evidence,
hidden root motion or extra transition expansion.


TESTING

Run focused W119 tests for:
- isolated per-request baseline construction;
- detection of family-shared contamination;
- equal-parent/equal-budget A/B;
- world-space guide and reachability;
- singularity and branch ambiguity;
- hard-pin preservation;
- actual Blender frame/channel accounting;
- display truth and rejected-status checks;
- source immutability, stale rejection and revert;
- temporal metric attribution;
- deterministic repeatability and provenance.

Run relevant existing W116–W118 and rig tests only
where contracts are reused. Do not run the entire
suite merely for isolated scratch changes.

Report actual commands, pass/fail, NOT RUN items,
environment, code and artifact hashes.

Tests and Blender results must be labelled AGENT-RUN.


STOP CONDITIONS

Stop and preserve evidence if:
- W118 COMPLETE or historical hashes cannot be verified;
- clean per-request F/X baselines cannot be constructed
  without altering frozen historical data;
- a shared-state segment contaminates the primary A/B;
- held-out outcomes were viewed before Y declaration;
- the target rig/joint binding is ambiguous;
- independent Blender evaluation is unavailable;
- the candidate requires Production or rig changes;
- user edit/root/hard pins must be changed to proceed;
- source identity/timebase must be fabricated or repaired;
- evaluation cannot distinguish source discontinuity
  from edit-induced discontinuity;
- a new Major Architecture/user policy decision is
  necessary to interpret the experiment;
- a concurrent branch or instruction conflict appears.

Do not silently patch the invalid historical V4 gate
or change W118's official CASE.

If a new experimental solver fails, record the failure.
Do not respond with uncontrolled parameter sweeps.


DECISION CONTRACT

Freeze validity, correctness and utility gates before
building the new held-out Y evaluations.

CASE A — BOUNDED SUPPORT:
All validity/contract gates pass, clean isolated
baselines and independent Blender footprints are sound,
hard pins and rollback hold, and the declared new
synthetic and independent real-family utility gates
pass with a visible actual-rig improvement.
This is not Production readiness or anatomical proof.

CASE B — PARTIAL SUPPORT:
All validity/contract gates pass and the new
representation demonstrates some bounded improvement,
but the required cross-family real-rig utility or
articulation trade-off remains mixed/unproven.

CASE C — NOT SUPPORTED:
Valid tests show the endpoint-space approach does not
provide a material improvement in the declared scope,
or it introduces critical hard-pin, ownership,
rollback or articulation regressions.

CASE D — INVALID:
Baseline isolation, provenance, independent evaluation,
predeclaration, population completeness, comparison
fairness or reproducibility fails. D supersedes
all other CASE outcomes.

Do not upgrade W118 D or any other historical CASE.

Do not report a passing numerical metric as proof of
user-facing animation quality.


FINAL REPORT

Write one W119 Worklog with:
1. Official CASE and frozen declaration identity.
2. Historical baseline and immutable CASE fingerprints.
3. W118 V4 contamination reproduction, not repair.
4. Isolated per-request U_r/F_r/X_r/Y_r provenance.
5. Y endpoint-guide formulation, DOF and parameters.
6. Synthetic held-out and negative controls.
7. Authentic real-rig A/B with correlated families noted.
8. Blender pin residuals and exact changed-frame/Bone sets.
9. Raw vs source-relative temporal motion accounting.
10. Bend-plane, near-straight and elbow costs.
11. Timeline and actual rig visual review, inspected items.
12. Hard failures, rejection classes and no-fallback proof.
13. Source immutability, atomic lineage and exact reversal.
14. Alternative explanations and non-claims.
15. Exact changed files, tests, hashes and replay commands.

Do NOT provide:
- Next Work Recommendation;
- Next Architecture Decision;
- Next Experiment;
- Next Batch proposal;
- Production migration recommendation.

The Agent is an Evidence Producer.
The human Architecture Owner and independent Reviewer
own architecture and subsequent Batch issuance.

Do not start W120.


COMPLETION CONDITION

W119 is complete only when controlled evidence answers:

1. Can W118's shared-segment baseline contamination be
   excluded from a new fair, isolated per-request A/B,
   without rewriting or upgrading W118?

2. Can a bounded end-effector-world-space guide be
   resolved on the actual Blender rig while maintaining
   the same hard pin, user edit, DOF and temporal budget
   as clean W118 X?

3. Does that mechanism reduce real-rig transition
   discontinuities without introducing near-straight,
   bend-plane, collateral or solver failures?

4. Are the actual modified frames/Bones independently
   disclosed, contract-valid and exactly reversible?

5. Does the evidence justify endpoint-space transition
   ownership as a practical Semantic Constraint Editor
   mechanism, or does it expose a more fundamental
   limitation of the source trajectory/two-bone rig?

A negative or INVALID outcome is legitimate.
Report missing evidence precisely.
Do not start W120 or modify Production systems.
