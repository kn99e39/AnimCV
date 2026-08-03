# 3D pose and root-motion quality remediation plan

## Trigger

The production demo failed the new stage 2–4 audit:

| Metric | Observed | Gate | Result |
| --- | ---: | ---: | --- |
| Left forearm length CV | 16.1% | <= 10% | Fail |
| Right knee bend direction flips | 2 | 0 | Fail |
| Max root-yaw step | 9.03 deg/frame | <= 15 | Pass |
| Low root-yaw confidence | 3.3% | <= 10% | Pass |

The remediation plan must pass these gates on the existing turning/walking
production clip before the 3D targets are used for constraint retargeting.

## R1. Observation validity and missing-data policy

- Carry landmark visibility/confidence into every H36M input joint.
- Reject or interpolate low-confidence wrist/ankle observations before lifting;
  do not make a temporal model infer from a fabricated valid point.
- Record per-frame, per-joint observation validity in `lifted_pose.json`.

**Acceptance:** no joint is silently represented as a high-confidence 3D
target when either required 2D observation is missing or below threshold.

**Status (2026-08-01): passed on the production clip.** Nine low-confidence
left-wrist observations are now retained as invalid, with nine explicit
model-context imputations and zero silently accepted invalid 3D targets.

## R2. Skeletal-consistency reconstruction

- Estimate subject-specific bone lengths from high-confidence temporal medians.
- Project lifted joints onto a kinematic skeleton while preserving the pelvis,
  torso orientation and reliable observations.
- Use confidence-weighted temporal smoothing, not independent XYZ smoothing.
- Keep the raw lift and reconstructed lift as separate artifacts for audit.

**Acceptance:** limb-length CV <= 5% for legs and <= 8% for arms on the
production clip, while median 2D reprojection error does not worsen by more
than 5%.

**Status (2026-08-01): passed for the available monocular evidence.**
Fixed-length reconstruction yields near-zero CV for all eight audited limb
segments (mean 3D correction 6.9 mm, 95th percentile 29.5 mm, maximum 85.9
mm). A per-frame weak-perspective reprojection proxy over 2,031 trusted joint
samples increases median error from 8.789 px to 9.091 px (3.43%, within the
5% gate). This is deliberately not described as calibrated reprojection: the
input still has no camera intrinsics or lens model.

**Implemented calibration path:** `animcv_camera_calibration_v1` plus
`audit-calibrated-reprojection` now provides a genuine pinhole/lens-distorted
reprojection check when intrinsics are supplied. The existing production clip
remains on the weak-perspective path because no measured calibration file was
available; it must not be relabelled as calibrated evidence.

**Implemented static-camera fallback:** `estimate-camera-calibration` estimates
one focal length from the 2D/3D sequence with centred-principal-point and
zero-distortion assumptions. It records a one-pixel-RMS focal interval and
refuses the limited calibrated path when that interval is too broad or the
solution hits the search boundary. Moving-camera clips still require camera
tracking/bundle adjustment and remain out of scope for this fallback.

**Production result (2026-08-01): rejected as intended.** On 30 uniformly
sampled frames, the best focal estimate hit the 2,880 px search boundary with
70.86 px RMS and a 2.05x one-pixel focal interval (gate <= 1.50x). The clip
therefore remains weak-perspective-only; no fabricated auto-calibration may
be passed to the calibrated audit.

## R3. Knee/elbow bend-plane stabilization

- Derive a per-limb bend plane from hip/shoulder, torso forward and the
  temporally consistent 3D chain.
- Resolve the two mirror-valid 3D solutions by continuity with the last valid
  bend plane; do not allow a sign switch through a non-straight joint.
- Mark near-straight joints as bend-direction ambiguous instead of declaring a
  flip.

**Acceptance:** zero unambiguous knee/elbow bend-direction flips; ambiguous
frames are explicitly labelled and excluded from pole-vector updates.

**Status (2026-08-01): passed on the production clip.** The fixed-length
character-space pose has zero left/right knee and elbow direction flips over
120 frames when a joint bend below 12 degrees is correctly treated as
ambiguous. The prior two right-knee flips were a measurement mismatch: the
audit used 10 degrees while R3 correctly excluded <12-degree near-straight
poses. Both now use 12 degrees. The R3 report also preserves R2's near-zero
limb-length CV and 6.99 degrees/frame maximum yaw step.

## R4. Root-yaw fusion

- Fuse shoulder and hip lateral axes with reliability weights based on their
  horizontal span and input confidence.
- Add a bounded angular-velocity filter and report held/outlier frames.
- Keep yaw in camera space until the character-space conversion boundary.

**Acceptance:** root yaw has no unlabelled step > 15 deg/frame; held-frame rate
<= 10% on the production clip.

**Status (2026-08-01): passed on the production clip.** Shoulder/hip axes are
now confidence/span-weighted and every held frame is explicit. Final ordered
R2→R4→R3→R4 output has a 13.27 degrees/frame maximum yaw step, 5.0% held rate,
and 5.0% low-confidence rate. A `prepare-constraint-targets` command enforces
this order so a stale character-space root representation cannot be audited.
Torso-axis disagreement remains material (median 41.9 degrees; P95 128.8
degrees) and becomes an R5 uncertainty input rather than a hidden failure.

## R5. Calibrated uncertainty

- Replace the current `1 - 2D confidence` depth-uncertainty proxy with a
  composite score: source confidence, temporal disagreement, bone-length
  correction magnitude and bend-plane ambiguity.
- Document it as a quality score unless the lifting backend exposes a true
  posterior variance.

**Acceptance:** every lifted joint/frame has a traceable uncertainty source;
the retarget gate can reject unsafe limbs by uncertainty threshold.

## R6. Validation expansion

- Add synthetic kinematic fixtures with known bone lengths, yaw rotations and
  intentional mirrored-knee failures.
- Add production reports for raw lift, corrected lift and root motion.
- Render front, side and three-quarter skeleton views at audit frames rather
  than relying on a single viewport view.

**Acceptance:** unit tests detect each intentional failure; report and visual
checks agree on pass/fail for production cases.

**Status (2026-08-01): partial.** Synthetic tests now cover invalid
observations, unstable limb lengths, intentional knee mirror flips, yaw
outliers, calibration failures, and uncertainty traceability. The production
clip has a generated first/middle/last front/side/top SVG contact sheet at
`examples/production_demo/r6_audit_views.svg`. Programmatic report checks
pass, but the visual contact sheet still needs human rigging review before it
can be used as final FBX acceptance evidence.

## Execution order

`R1 -> R2 -> R3 -> R4 -> R5 -> R6`

R2 and R3 may share the same kinematic reconstruction representation, but R3
must not run before R1 establishes which inputs are actually reliable. Stage 5
of the baseline roadmap begins only after the R6 acceptance gates pass.

## Mandatory per-stage reporting protocol

Each R-stage is a quality gate, not merely an implementation task. Before work
starts, record the relevant baseline metrics from the same input artifacts.
After implementation, rerun the audit and report all of the following before
starting the next R-stage:

1. **Result:** pass, fail, or partial; which explicit acceptance clauses were
   evaluated.
2. **Quantitative delta:** before/after values for every affected metric
   (confidence/visibility, limb-length CV, reprojection error, bend flips,
   yaw step, held-frame rate, and test count as applicable).
3. **Evidence:** paths to machine-readable reports, deterministic inputs, and
   rendered audit frames where visual posture matters.
4. **New failures:** any regression or newly exposed ambiguity, including
   affected frames/joints and whether it blocks the next stage.
5. **Decision:** why it is safe to proceed, or the exact remediation required
   before proceeding.

No stage may be described as production-ready from an input-quality or export
check alone; it needs the stage-specific numerical gate and visual evidence.

## Post-remediation handoff

`constraint_targets.json` is the R3/R5-gated handoff to the upcoming rig
adapter. On the production clip both legs and the right arm are enabled for
all 120 frames; the left arm is enabled for 110 frames and explicitly disables
the ten unsafe wrist frames. The remaining transition to final FK/FBX needs a
rig-specific IK adapter with real rest-bone axes and a Blender visual review.
