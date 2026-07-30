# Dynamic-input quality check

This test uses the five-frame MMPose PoseTrack test clip in `source.mp4`.
It is intentionally a difficult input: a broadcast soccer scene with several
people, partial occlusion, motion blur, and a player lying on the ground.

## Results

With the default visibility threshold of `0.3`, every left-arm landmark used
by `mapping.json` was below the threshold in every frame:

| Landmark | Mean confidence | Visible frames |
| --- | ---: | ---: |
| left_shoulder | 0.205 | 0 / 5 |
| left_elbow | 0.136 | 0 / 5 |
| left_wrist | 0.161 | 0 / 5 |

Despite that, the current retargeter wrote two static tracks and the optimizer
reduced each from five keys to two. This exposes a quality/control-flow issue:
low-confidence or invisible landmarks are not surfaced as an unusable mapping
before retargeting.

At a deliberately low threshold of `0.1`, the same landmarks become visible,
but the inferred rotations are unstable: the maximum change from the first
frame is 164.993 degrees for `upper_arm.L` and 67.140 degrees for
`forearm.L`. The optimizer correctly keeps all five keys because it cannot
safely simplify that noisy signal.

Artifacts without the suffix use the default threshold. Artifacts with the
`threshold_010` suffix use the low threshold. The raw input, extracted frames,
pose results, MotionGraphs, and raw/optimized animations are all retained for
inspection.

This is a robustness test, not a claim that RTMPose should produce a usable
single-person motion capture from crowded sports footage. It demonstrates
that AnimCV needs an explicit quality gate before producing a seemingly valid
animation from low-confidence landmarks.

## Additional crowded-scene check

`official_demo.mp4` and `official_frames/` are the five-frame official MMPose
demo clip. It is also a distant multi-person scene. Its generated
`official_pose.json` / `official_motion.json` have mean confidences of 0.04,
0.10, and 0.14 for left shoulder, elbow, and wrist respectively; all are
invisible at the default threshold. `official_retarget.log` records that the
quality gate rejected the retarget operation and no `official_animation.json`
was written. The root-level `QUALITY_AUDIT.md` tracks this and the remaining
production-readiness gaps.

`source_quality_report.json` is retained for the sports-input rejection and
shows the exact failed visibility and confidence metrics for each mapped bone.
