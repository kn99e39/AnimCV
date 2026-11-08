# BATCH 123 — Reliability-Aware Animation Base: Learned 3D Pose Error Estimation and Decision Utility

DIRECTION

Continue AnimCV toward a Reference-Guided, Reliability-Aware
Video-to-Animation Generator and user-controllable Semantic Constraint Editor.

USER-APPROVED ARCHITECTURE DIRECTION:
Introduce an explicit, learned per-frame/per-joint 3D reconstruction
uncertainty or expected-error signal to inform how strongly a future
Generator should trust a reconstructed reference motion.

The user approved removing the W123 HOLD and proceeding with this
bounded uncertainty hypothesis. This approval DOES NOT authorize
a trained Motion Generator, Production schema changes, unbounded
training, modifying FramePose weights or relaxing Hard Constraints.

For W123, build an isolated, calibrated Reliability Assessor/Head
candidate around a frozen existing 3D FramePose model and measure
whether that signal genuinely predicts error and improves a bounded,
GT-free-at-decision-time review/selection task.

Prioritize evidence about useful estimation-error uncertainty,
not a decorative confidence output.

PURPOSE

W122 official CASE B (limited/context-specific) had only one
held-out release window with agent-rated visual benefit, and its
S1_NH event involved a ~400 mm/frame one-frame change in the
self-occluded far-arm reconstruction. Following an erroneous
source hypothesis can improve a source-relative comparison while
making the animation less faithful to the performer.

The fundamental issue is distinct from the World Pin contract:
how should the eventual Generator differentiate well-supported
reference motion from possibly incorrect reconstruction?

Existing RTMPose scores describe 2D detector outputs. W113
established that they are not bounded calibrated probabilities:
actual scores can exceed 1 and '1 - confidence' can go negative.
These are not valid calibrated 3D uncertainty estimates.

Architecture_v2's Keyframe Importance and Confidence history is
not equivalent to a model that predicts 3D reconstruction error.
Architecture_v3 currently specifies FramePoseEstimator output as
(B,17,3), without a learned calibrated 3D uncertainty head.
No existing result establishes that a trained uncertainty head
will be useful, calibrated or generalizable.

W123 tests this missing reliability-information layer before
the additional cost and entanglement of a Motion Generator.

CENTRAL INTENT

Can a separate learned assessor, using only authentic runtime-
available observations and the frozen model's own predictions
or representations, predict frame/joint-specific 3D pose error
with calibration and transfer value beyond raw detector scores
and simple unlearned baselines?

Does that information improve a PREDECLARED, label-free-at-decision
review/retention/alternative-selection task on unseen examples,
rather than merely correlating with GT in a diagnostic probe?

What failures occur on occluded, mislocalized, high-velocity,
out-of-distribution and high-confidence-wrong predictions?

Do not treat probe recoverability, correlation or low training
loss as proof that a future Generator will use uncertainty
correctly or produce better animation.

CURRENT ASSESSMENT

Historical official cases remain unchanged:
W113 D, W114 D, W115 D, W116 B, W117 C,
W118 D, W119 D, W120 B, W121 D, W122 B.

W113: unbounded RTMPose scores and unjustified 3D uncertainty
proxy; do not equate detector raw score with calibrated 3D trust.

W103: failure depends on 2D observation, missingness and frozen
model behavior, but its GT-only analysis cannot be promoted to
an inference-time error predictor.

W116–W122: scratch world-pin and authorized visible edit
contracts have useful limited evidence; temporal guide quality
is context-specific and may follow erroneous source jumps.

W121 acquired three genuinely independent dynamic GT-free clips:
S1 kettlebell, S2 juggling, S3 bent-over row. W122 reused them
for held-out WITHIN-SOURCE windows, did not establish independent
real-time viewer utility, and did not establish 3D Ground Truth.
Do NOT label these real clips as GT-scored.

There has been an official W116–W120 cumulative repository audit.
The W125 audit boundary is unchanged.

HUMAN-READABLE SUMMARY

Direction:
Test a frozen-model, sidecar learned joint-level 3D error assessor.

Purpose:
Establish whether a future Generator can distinguish trusted
motion references from uncertain reconstruction hypotheses.

Central Intent:
Is the model's predicted uncertainty actually calibrated and
decision-useful, rather than a proxy that follows 2D visibility
or merely identifies obvious high-error examples?

Do:
- Audit frozen FramePose model, inference inputs and available GT.
- Design a causal/leakage-safe separation of training,
  calibration and held-out evaluation at video/subject level.
- Compare a small learned assessor against fixed/unlearned
  confidence, visibility and geometry baselines.
- Evaluate error prediction, calibration, coverage, high-
  confidence failures and domain transfer.
- Test prospective utility of a GT-free-at-decision-time
  triage/selection/retention decision, if controlled candidates
  are available.
- Preserve GT-free real-video outputs and all historical records.

Do NOT:
- Modify or fine-tune existing FramePose/RTMPose models.
- Train Motion Generator/Diffusion or new pose backbone.
- Write uncertainty into existing Production MotionGraph,
  AnimationClip, Blender/FBX or keyframe output schemas.
- Choose a permanent Generator trust-weighting policy.
- Suppress source motion, overwrite estimates or treat
  high uncertainty as permission to violate hard constraints.
- Request or decide the next Batch.

Decision enabled:
Whether the proposed Reliability layer is sufficiently
valid and useful to remain a candidate input to a future
Generator/Editor, or remains unsubstantiated.

HISTORICAL BASELINE

Repository: kn99e39/AnimCV
Existing branch ONLY: arch/single_frame_first.
Last observed pre-issue HEAD: 08d6851e8bcc220ed07e130cf279f01d3b720587.
Verify current remote/local HEAD and instruction blob before work.

Read:
- AGENTS.md, PRODUCT_VISION.md
- Architecture_v3_FramePose.md and Architecture_v2.md
- docs/103, 102, 105 and 113 Worklogs
- docs/121 and 122 Worklogs and their separate real-video limits
- docs/AUDIT_W116_W120_2026_11_06.md
- existing FramePose/Core, frozen W102/S0 evaluation contracts,
  3DPW/raw-reference validity and sample identity code
- current docs/next_batch.md

Preserve all previous checkpoints, model weights, datasets,
splits, manifests, source video, rig, Blender Actions, scripts,
Worklogs and CASE decisions.

No branch create/switch/delete/merge/rebase/reset or force-push.
Do not change any existing Production src/ module or model.
No Production inference contract or keyframe-collapse behavior
may be altered in W123.

INFORMATION OWNERSHIP AND SEMANTICS

Distinguish:
1. Immutable original RGB and timestamp/source identity.
2. Raw 2D detector observation: xy, score, validity and provenance.
3. Frozen reconstructed 3D pose hypotheses and optional frozen
   intermediate representations.
4. Learned assessor estimate of 3D error or uncertainty.
5. Calibration and probe recoverability.
6. A proposed runtime decision that uses the estimate.
7. A future Generator's actual behavior (NOT implemented here).
8. Final edited Blender Action and animator utility (NOT proven here).

The 3D uncertainty signal concerns expected error of a SPECIFIC
frozen model prediction, joint, frame, coordinate frame,
normalization, regime and model/checkpoint. It is not:
- detector probability;
- absolute ground-truth confidence;
- epistemic-versus-aleatoric decomposition without evidence;
- a hard user constraint;
- a statement about whether visible movement is artistically good.

Use uncertainty at source-frame/joint resolution initially.
A Blender keyframe is a later sampling/optimization artifact.
Do not lose uncertainty at frames removed by keyframe collapse.

Hard user-authored WORLD_POSITION_PIN constraints remain binding
regardless of predicted uncertainty. Only reference-trust
SOFT weighting or candidate review priority may respond to it.

PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE:
- Frozen FramePose checkpoint and inference implementation.
- RTMPose raw scores, source validity and original observations.
- W102/W103/W113 sample identity, raw-reference/eval-only GT rules.
- Historical 3DPW VAL/TEST and derived candidate comparisons.
- W121/W122 source images, clips, rig outputs and W122 CASE.
- Existing Real AnimCV (GT-free) runtime semantics.
- All existing source/manifests, branch and Worklogs.

CHANGE ONLY:
- Isolated W123 assessor/uncertainty research code and tests.
- Scratch training/calibration artifacts and metadata outside
  existing model/checkpoint locations.
- Scratch identity-bound reliability sidecar outputs.
- W123 prospective declaration, evidence and Worklog.
- Small diagnostic-only error-ranking/decision replay, if valid.

DO NOT TUNE:
- Frozen FramePose/RTMPose checkpoint, crop, threshold, masks,
  2D observations, source timing, source-to-rig conversion.
- W116–W122 world-pin thresholds, solver, candidate interpolation,
  guide, flex-floor, or animation-quality criteria.
- Existing historical evaluation-gate decisions.
- Test thresholds, sample selection, baselines or source splits
  after seeing held-out outputs.

Small assessor training is permitted ONLY within this diagnostic
scope. Architecture of assessor, feature representation, loss
and bounded optimization choices are Agent-selectable, provided
they are declared before final held-out assessment.

REQUIRED EVIDENCE — 1
DATA AND SPLIT VALIDITY

Identify exact frozen predictor, dataset(s), GT coordinate frames,
raw 3DPW reference alignment, validity masks, actor associations,
frame rate, observation regime and model training-data exposure.

IMPORTANT: W102/W103/W105 3DPW VAL/TEST has repeatedly informed
design and is NOT a fresh untouched model-system holdout.
Do not call this clean independent generalization simply because
an assessor is trained on other samples.

Build disjoint assessor TRAIN / CALIBRATION / FINAL EVALUATION
populations by original source video/actor identity, and keep
correlated frames, multiple people/crops of the same image,
near-duplicate scenes and derivative observations from crossing
partitions. Where possible, obtain an independent GT-supported
evaluation dataset compatible with the frozen predictor; retain
distinct source regimes and camera calibration.

Record whether the original frozen FramePose checkpoint may
have been trained on each dataset. If historical exposure
cannot be proved absent, disclose it as a limitation.

Confirm target 3D joint errors are physically meaningful:
- same identity, joints, units, root-relative vs global alignment;
- correct coordinate and camera transforms;
- raw GT validity independent of detector visibility;
- no invalid targets silently treated as zero;
- no GT in runtime features, candidate selection or the assessor
  evaluation decision.

Describe missing/out-of-support joints explicitly.
Group statistics by source video, actor and observation regime.
Do not pool oracle geometry and actual detector observations
without retaining separate metrics and source identities.

If no sufficiently independent, authenticated and well-aligned
GT supervision/evaluation population can be established,
STOP with DATA BLOCKED / NO ARCHITECTURE CASE.
Do not train on W122 GT-free footage as if labels existed.

REQUIRED EVIDENCE — 2
PROSPECTIVE RELIABILITY TARGET AND BASELINES

Declare one PRIMARY uncertainty target and output semantics
BEFORE looking at final evaluation errors, for example:
a positive predicted per-joint 3D absolute error scale in
metres, or an explicitly calibrated probability of exceeding
a predefined per-joint error level.

A score ordering without stable scale is a RISK SCORE,
not a calibrated probability.

Baseline comparators must include:
A. constant/per-joint and/or dataset-prior error estimates;
B. raw 2D detector score plus missing/validity status,
   with its ACTUAL measured non-[0,1] domain preserved;
C. an unlearned runtime-available geometry/temporal-consistency
   heuristic where supported (may report UNAVAILABLE).

The learned assessor candidate may use legitimate runtime-
available features, such as frozen 2D xy/score/validity,
frozen 3D pose, joint geometry and optional frozen feature
tokens, but MUST NOT use hidden GT, future edited output,
benchmark stratification labels, actor GT ID or a privileged
oracle available only in evaluations.

If temporal context is used, declare online/offline scope and
actual source time, identity continuity and future-frame access.
Unverified identity must not be silently treated as known.
Do not require production changes to fetch hidden-state tokens.

Predeclare feature allowlist, normalization, training strategy,
loss, training/calibration splits, seeds, model-selection rules,
and the final-evaluation frozen metrics. Use only development
data to choose hyperparameters and calibrate reliability.
No post-hoc hyperparameter or threshold sweep.

REQUIRED EVIDENCE — 3
LEARNED ERROR ESTIMATION AND CALIBRATION

Train a small candidate Reliability Assessor/Head around the
FROZEN pose model; the original model predictions must remain
byte-identical wherever deterministic baseline comparison is
defined.

Before testing the candidate:
- confirm feature/GT access separation;
- verify output finite, nonnegative and unit-consistent;
- verify missingness has explicit supported/unsupported status;
- verify checkpoint provenance, seed and exact code identity;
- declare numerical and calibration reporting criteria.

On final evaluation, report at minimum:
- per-joint and per-frame target support/coverage;
- actual 3D error distributions and prediction distributions;
- ranking discrimination (e.g. Spearman/PR-AUC for materially
  wrong predictions) with valid denominators;
- calibration error by predicted-error bin, quantile coverage,
  error-conditioned coverage/rejection or equivalent appropriate
  metric for chosen prediction semantics;
- mean/median/p90 error within predicted reliability strata;
- false-reassuring high-confidence / high-error counts;
- missingness, occlusion, detector score >1, near-straight,
  high-motion and camera/viewpoint strata where supported;
- video-cluster uncertainty/intervals and small-sample limits;
- calibration on training-domain vs independently sourced
  evaluation domain, explicitly separated.

A low average regression loss is not enough.
Report every baseline and learned candidate under paired,
identical observations and target support.
Explain how a model could look calibrated marginally while
failing badly on the occluded arm or on a new motion domain.

Do not assume numeric scores can be interpreted as per-keyframe
likelihood of correctness until calibrated to a declared event.

REQUIRED EVIDENCE — 4
REAL VIDEO GT-FREE APPLICATION

Replay assessor inference on the W121/W122 independently
sourced clips S1 kettlebell, S2 juggling and S3 bent-over row,
using frozen source inputs/checkpoints, with separate identity
and time evidence. Do not treat W122 as a fresh new source
relative to W121.

Inspect representative W122 S1_NH event around f251 and any
other predeclared unstable observations.

Report:
- detector score and validity;
- predicted uncertainty for relevant arm joints;
- source/FramePose trajectory changes;
- supported/unsupported/unknown source identity and binding;
- whether obvious instability is flagged and whether normal
  motion is also flagged as uncertain;
- examples of false reassurance or indiscriminate rejection.

These are qualitative/sanity and domain-shift stress tests.
The ground truth 3D error on S1/S2/S3 is UNKNOWN.
A high estimated uncertainty for the 400 mm event does NOT
prove the assessor correctly identified the actual 3D error.
A low uncertainty is a failure warning, not proof that the
motion is correct.

No production rendering/export or Motion Generator change
is allowed in this Batch.

REQUIRED EVIDENCE — 5
GT-FREE-AT-DECISION UTILITY

Assess whether the reliability signal has actual decision
utility beyond ranking correlation.

Choose and DECLARE one minimal decision task using only
runtime-available features, for example:
- prioritizing a bounded number of suspect frame/joints
  for human review;
- selective acceptance/abstention of the frozen 3D reference;
- conditional selection between TWO pre-existing frozen
  candidates with identical source support and no GT inputs.

Evaluate policy usefulness on held-out GT-supported data
AFTER decisions are frozen:
- error coverage and false-negative rate for a fixed review
  budget or abstention rate;
- relative value compared with constant, raw-detector and
  geometry baselines;
- failure concentration on high-confidence erroneous joints;
- per-video/actor generalization;
- impact on valid high-motion observations vs obvious artifacts.

If comparing an alternative candidate, both alternatives must
be independently available in the same output coordinate
system and identical input regime; do not introduce a GT-derived
candidate, retrain a Generator or select the better candidate
with GT at inference time.

A synthetic oracle-policy UPPER BOUND may be reported only as
evaluation-only and never counted as deployable performance.

If only error detection but no actual paired editing alternative
can be tested, label evidence as TRIAGE UTILITY, not Generator
decision utility or final Animation improvement.

Do not claim the future Generator causally uses reliability;
Generator behavior is out of scope.

REQUIRED EVIDENCE — 6
RELIABILITY-AWARE BASE CONTRACT (SCRATCH ONLY)

Create a minimal source-bound SIDECAR representation if needed:
- source frame and timestamp;
- semantic joint and coordinate frame;
- model/checkpoint and observation provenance;
- frozen pose hypothesis identity;
- reliability output type, units, calibration identity;
- supported/missing/unknown flags;
- original raw 2D score retained distinctly;
- diagnostic error and evaluation GT kept in a separate,
  evaluation-only artifact never read at runtime.

Do not overwrite confidence fields in Production MotionGraph
or AnimationClip. Do not rename detector score into calibrated
3D confidence. Do not collapse frame-level uncertainty to a
sparse list of stored Blender keyframe indices.

Show how a FUTURE Generator could consume a soft reference
weight while a user-authored Hard Pin remains independent,
without implementing this Generator or choosing permanent
weight mapping.

Synthetic contracts should prove provenance preservation,
no GT leakage, invalid/missing handling, score>1 handling,
time/joint indexing, reordering rejection, stale checkpoint
rejection and separation of soft trust vs hard locks.

QUANTITATIVE AND QUALITATIVE ACCOUNTING

Provide actual numerators, denominators, uncertainty intervals
and source IDs for all decision metrics. Separate:
1. Assessor implementation and provenance correctness;
2. Synthetic sidecar/GT-isolation contract correctness;
3. Held-out GT error prediction and calibration;
4. GT-free decision/triage utility;
5. Real-video stress-test observations without GT;
6. Future Generator Architecture viability (UNPROVEN).

Review representative:
- well-observed correct pose;
- high-confidence wrong pose;
- missing or occluded joint;
- fast but valid motion;
- visible detector score >1;
- W122 S1_NH suspicious wrist event;
- genuine false positive/false negative;
- domain-transfer failure.

Label reference-supported facts vs agent inference. Preserve
negative examples, data gaps, prior Worklogs and failed baselines.

DECISION CONTRACT

Freeze precise gates before final held-out results.

CASE A — BOUNDED SUPPORT:
The assessor is demonstrably calibrated or has valid stated
uncertainty semantics on authenticated held-out GT data,
beats meaningful baselines on required ranking/calibration
and decision-utility endpoints, preserves all source/GT
separation contracts, and has no material hidden high-
confidence error failure on the declared critical cases.
Only supports a bounded future soft-reference input.

CASE B — PARTIAL SUPPORT:
Valid measurement supports some error prediction, calibration
or triage utility, but generalization, decision utility or
critical false-reassurance behavior is mixed/unresolved.

CASE C — NOT SUPPORTED:
Valid controlled comparison shows the learned reliability
mechanism is not meaningfully calibrated/decision-useful
relative to simpler baselines under the tested conditions,
or reliably fails a required minimum guarantee.

CASE D — INVALID:
GT alignment, source/model identity, train/eval leakage,
target support, score semantics, assessor/evaluator separation
or prospective controls are invalid.

DATA BLOCKED — NO ARCHITECTURE CASE:
No valid GT-supported independent training/evaluation basis
or necessary frozen features can be obtained without an
unapproved model/Production change.

D supersedes A/B/C when a test occurred with invalid controls.
Never reinterpret old historical CASEs based on this Batch.

STOP CONDITIONS

STOP and preserve evidence if:
- authentic aligned GT cannot be safely used without leakage;
- model exposure/splits cannot be made distinguishable and
  inference claims would require treating reused TEST as unseen;
- any GT or privileged annotation enters learned inference;
- frozen FramePose output must change or require fine-tuning;
- an already-used historical source is falsely called unseen;
- raw detector score is silently clipped/interpreted as prob;
- soft reliability overrides explicit Hard Pin;
- training requires unapproved full-model/Generator changes;
- a valid decision test requires changing W116–W122 solvers;
- tests need post-outcome labels, thresholds or clip selection;
- independent predictive calibration cannot be evaluated;
- repository histories, source artifacts or branches would move.

Preserve valid partial diagnostics. Do not keep training or
sweep a failed hypothesis until an arbitrary positive appears.

TESTING

Run focused tests for:
- data/source/split and actor/video identity;
- 3D target coordinate, metric and GT alignment;
- inference feature allowlist and no-GT dependency;
- source/model/assessor/calibration identity and replay;
- detector raw scores above 1, missingness and provenance;
- bounded assessor repeatability and model-weights immutability;
- quantile calibration/ranking metric correctness;
- decision/task controls including negative and no-op cases;
- explicit hard constraint independence and sidecar integrity.

Use relevant focused frozen-model regressions, not reflexive
full Pytest. Report exact commands and NOT RUN tests.
Agent-run GPU/Blender numbers are NOT reviewer-reexecuted.

FINAL REPORT

Produce W123 Worklog with:
1. Official CASE or DATA BLOCKED and exact Central Intent;
2. Source/GT alignment, provenance and actual model exposure;
3. Original and assessor split/sample IDs and identities;
4. Declared target, baseline, features, training/calibration
   strategy and gates, with pre-outcome fingerprints;
5. Training and calibration numerical results;
6. Paired held-out baseline/learned comparisons by regime;
7. High-confidence wrong, occlusion and domain-shift failures;
8. Decision/triage utility separate from score correlation;
9. GT-free W121/W122 real clip stress test and non-claims;
10. Scratch sidecar/GT-isolation and hard-lock invariants;
11. Negative controls, alternatives and failure mechanisms;
12. Source preservation, exact changed files, tests and
    reproducibility;
13. Unresolved uncertainty and Architecture hypothesis status.

Do NOT claim Architecture success automatically.
Report negative results honestly.

No Next Work Recommendation, Next Architecture Decision,
Recommended Next Experiment, Proposed Next Batch or
Next Implementation Plan. Do not start W124.

COMPLETION CONDITION

W123 is COMPLETE when controlled evidence enables the
independent Reviewer to judge:
- Whether an actual model-specific joint/frame 3D error
  signal can be learned without GT/inference contamination;
- Whether it is calibrated and useful beyond a raw 2D score
  on clearly defined GT-supported held-out populations;
- Whether it can improve a real GT-free-at-decision task,
  or only an offline probe/error ranking;
- Whether the W122 problematic observations illustrate a
  meaningful stress test without fictional 3D ground truth;
- Whether the reliability sidecar is an architecturally
  defensible candidate for future soft reference weighting,
  while Hard Pins and original evidence remain independent.

A negative or blocked result is valid. Report exactly what
cannot be judged. Never automatically implement a Generator,
Production schema changes or W124.
