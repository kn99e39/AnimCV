# Production quality demo

This is a reproducible 120-frame, 50 fps end-to-end audit using a public,
single-performer motion clip and `examples/BaseRig.fbx` (53-bone rig).

## Retarget quality-gate result

The tracked-subject path passes the *input* retarget quality gate for all eight
mapped major limb chains (both upper/lower arms and thighs/calves). The
exported FBX has one armature, a 1--120 action range, 519 animation curves,
and retains the source's 50 fps rate when re-imported in Blender 5.1.

It does **not** pass physical-animation quality review. The source performer
turns side-on while walking, but the current pipeline has only 2D landmarks.
It maps thigh and calf directions independently around a fixed world axis and
does not solve a knee hinge, pole vector, joint limits, or root/body yaw. The
result preserves the source's 2D knee angles while allowing visually invalid
3D knee bend directions. This demo is therefore a reproducible failure case,
not a production-ready animation claim.

| Metric | Result |
| --- | ---: |
| Input frames | 120 |
| Source / exported FPS | 50 |
| Mapped chains | 8 / 8 passed |
| Lowest visibility rate | 92.5% (left forearm) |
| Lowest mean landmark confidence | 0.720 (left forearm) |
| Largest direction step | 46.3 degrees (left forearm) |
| Retarget samples | 960 (8 x 120) |
| Optimized samples | 92 |

## Artifacts

- `source.mp4` and `frames/`: original clip and deterministic extracted frames.
- `pose.json`: deliberately retained negative control from whole-frame pose
  inference; it fails the quality gate.
- `pose_tracked.json`: accepted pose sequence produced with the initial subject
  box plus detector-based subject tracking.
- `lifted_pose.json`: 81-frame temporal VideoPose3D output in the documented
  pelvis-relative camera frame. It is an intermediate 3D target, not yet an
  FBX input: root yaw, global translation, and joint constraints are added in
  later pipeline stages.
- `root_motion.json`: smoothed root yaw, forward/right vectors, confidence,
  and the lifted joints transformed into character-root space. Global root
  translation remains deliberately absent until foot-contact analysis.
- `kinematic_pose.json` and `kinematic_root_motion.json`: R2's separate,
  fixed-length reconstruction and its character-space representation.
- `kinematic_reprojection_report.json`: R2's 2D-preservation gate. It is a
  weak-perspective proxy because this source contains no calibrated camera;
  it passes with a 3.43% median-error increase (8.789 px to 9.091 px) over
  2,031 trusted joint samples.
- `../camera_calibration.example.json`: input schema for the stronger
  `audit-calibrated-reprojection` path. It is an example only and was not
  applied to this source clip, so this demo remains uncalibrated.

  The optional `estimate-camera-calibration` fallback is restricted to static
  cameras and writes an uncertainty report before it can be used. It is not a
  replacement for the example's checkerboard-derived calibration input.
- `auto_camera_calibration_report.json`: the static-camera fallback's result
  on this clip. It correctly **rejects** the estimate: focal reaches the
  search boundary, RMS is 70.86 px, and focal uncertainty is 2.05x versus the
  1.50x gate. `auto_camera_calibration.json` is retained for diagnosis only,
  and must not be used for calibrated auditing.
- `bend_stabilized_pose.json`, `bend_stabilized_root_motion.json`, and
  `bend_stabilized_quality_report.json`: R3 output and numerical audit. Both
  knees and elbows have zero unambiguous bend-direction flips at the
  12-degree ambiguity threshold. This validates target reconstruction, not
  final rig retargeting.
- `constraint_ready_pose.json`, `constraint_ready_root_motion.json`, and
  `constraint_ready_quality_report.json`: the only valid ordered R2→R4→R3→R4
  target preparation output. It passes all numerical 3D gates.
- `r5_uncertainty_report.json` and `constraint_targets.json`: traceable
  quality scores and rig-independent 3D end-effector/pole targets. The left
  arm disables ten unsafe wrist frames; no target is fabricated for them.
- `r6_audit_views.svg`: 0/60/119-frame front, side and top target views for
  manual rigging review; this is not a final rendered-FBX approval.
- `mapping.json`, `motion.json`, `quality_report.json`: mapping, motion graph,
  and machine-readable quality-gate evidence.
- `animation.json` and `animation_optimized.json`: dense retarget output and
  its medium-collapse version.
- `animated_BaseRig.blend` and `animated_BaseRig.fbx`: final Blender scene and
  game-pipeline interchange artifact.

## Scope and limitation

This validates tracking and the input quality gate on a clean, fully visible,
single-person clip; it does not establish game-ready animation reliability.
For turning, occlusion, multiple people, fast camera motion, root motion, and
depth-sensitive joints, the pipeline needs temporally coherent 3D pose lifting
plus rig-aware IK constraints before its FBX output can be accepted.
