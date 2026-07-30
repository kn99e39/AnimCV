# Motion-quality audit

Status: **not approved for direct adoption in a game-production pipeline**.

AnimCV currently proves that it can produce Blender-editable draft animation
from suitable input. It does not yet prove stable, production-ready motion
quality across performers, cameras, actions, or target rigs.

## Executed checks

| Check | Input | Result | Interpretation |
| --- | --- | --- | --- |
| Static single-person E2E | `examples/e2e_demo/person.mp4` | Pose, MotionGraph, FBX parsing, retargeting, key reduction, and Blender export completed. | Pipeline connectivity is verified, but repeated still frames do not measure motion fidelity. |
| Low-confidence multi-person action | `examples/quality_demo/source.mp4` | Default quality gate rejected both mapped left-arm tracks: 0% visibility and mean confidence 0.14 / 0.13. | Correctly avoids exporting an empty held-pose animation. |
| Crowded official MMPose demo | `examples/quality_demo/official_demo.mp4` | Default gate rejected both mapped tracks: 0% visibility and mean confidence 0.04–0.10. | Correctly rejects a distant multi-person scene, which is outside the current single-subject design. |
| Synthetic smooth arm swing | `tests/test_retarget_solver.py` | Passes the gate and produces a 90-degree arm rotation. | Solver mathematics and the positive gate path are covered, but this is not visual validation. |
| Invisible landmarks | `tests/test_retarget_solver.py` | Gate rejects the solve; no animation is returned. | Prevents the previously observed false-success condition. |
| One-frame 180-degree direction change | `tests/test_retarget_solver.py` | Gate rejects the solve. | Detects gross landmark swaps/outliers. |
| Single-frame coordinate spike | `tests/test_temporal_filter.py` | Three-frame median filter removes the isolated spike without altering endpoints. | Valid landmarks are smoothed only after quality validation. |
| Failed quality-report persistence | `examples/quality_demo/source_quality_report.json` | Gate rejected output and wrote per-mapping metrics. | Failures are inspectable and machine-readable. |

## Open quality issues

### Q-001 — No identity tracking in multi-person footage

`PoseEstimator` has no subject identity model. It currently estimates one pose
over the full image, so it cannot reliably isolate the intended performer in a
crowd or keep an identity across time.

**Implementation finding (2026-07-27):** the current adapter does not supply
person bounding boxes to MMPose at all. MMPose therefore treats the complete
image as one bbox and returns one full-frame pose. Identity tracking cannot be
implemented correctly until AnimCV adds a person-detector stage and a policy
for selecting the intended subject.

### Q-002 — No temporal pose filtering or recovery

Visible landmarks now receive a three-frame median filter after the raw input
passes validation, which suppresses isolated coordinate spikes. There is still
no adaptive filtering, outlier repair, or short-occlusion interpolation;
smaller repeated jitter can therefore still reach Blender.

### Q-003 — No production motion-fidelity benchmark

There is no labelled video-to-motion benchmark, ground-truth skeleton/animation
comparison, or artist review protocol. Keyframe count and successful `.blend`
export are not measures of animation quality.

### Q-004 — 2D camera ambiguity remains

Without optional depth sampling, the solver infers directions in image space.
Camera motion, foreshortening, and out-of-plane limb movement cannot be
reconstructed reliably enough for unreviewed production output.

## Minimum evidence before production adoption

1. Curate consented, single-performer clips covering the intended game actions,
   camera angles, clothing, and occlusion cases.
2. Add tracked-subject selection and identity continuity checks.
3. Add temporal filtering plus a per-frame/per-bone quality report.
4. Retarget the clips onto representative production rigs and have animators
   score edit effort, foot sliding, limb jitter, and anatomical plausibility.
5. Define acceptance thresholds from those results and run them as regression
   tests before releases.
