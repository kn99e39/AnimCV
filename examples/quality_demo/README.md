# Dynamic-input quality check

This test uses the five-frame MMPose PoseTrack test clip in `source.mp4`.
It is intentionally a difficult input: a broadcast soccer scene with several
people, partial occlusion, motion blur, and a player lying on the ground.
Retargeted onto `examples/BaseRig.fbx` (a real 53-bone UE-mannequin-style
skeleton) via `mapping.json` (`upperarm_l`/`lowerarm_l`, left arm only) —
this demo previously used a hand-built 3-bone placeholder armature.

## Results

With the default visibility threshold of `0.3`, every left-arm landmark used
by `mapping.json` was below the threshold in every frame. `retarget`'s
quality gate now correctly rejects this input by default:

```
retarget input quality check failed (upperarm_l: visibility 0% is below 60%,
mean confidence 0.14 is below 0.30; lowerarm_l: visibility 0% is below 60%,
mean confidence 0.13 is below 0.30)
```

(`source_quality_report.json` has the exact per-mapping numbers.) The
`animation.json` / `animation_optimized.json` artifacts here were produced
with `--skip-quality-check` specifically to reproduce what the *old*,
pre-quality-gate behavior looked like: the retargeter wrote two static
tracks (five identical keyframes each) and the optimizer collapsed each to
2 keys. That gap between "obviously bad input" and "silently accepted
output" is exactly why the quality gate exists now — see the
"Additional crowded-scene check" section below for the current, correct
behavior on an equally bad input.

At a deliberately low `estimate-pose --visibility-threshold 0.1`, the same
landmarks are marked visible (this only affects pose estimation's own
visible/invisible flag, not confidence values or retarget's own gate, so
`--skip-quality-check` is still used here too), but the inferred rotations
are unstable: `optimize` keeps all 5 keys for `upperarm_l` (0 removed) and
4 of 5 for `lowerarm_l` (1 removed, max_error 0.581) because the signal is
too noisy to safely simplify — see `animation_threshold_010_optimized.json`.

Artifacts without a suffix use the default threshold; artifacts with the
`threshold_010` suffix use the low one. The raw input, extracted frames,
pose results, MotionGraphs, and raw/optimized animations are all retained
for inspection. `animated_rig_low_confidence.blend`/`.fbx` are the
Blender-exported result of the default-threshold (`--skip-quality-check`)
animation, for visual inspection of just how meaningless it is.

This is a robustness test, not a claim that RTMPose should produce a usable
single-person motion capture from crowded sports footage.

## Additional crowded-scene check

`official_demo.mp4` and `official_frames/` are the five-frame official MMPose
demo clip. It is also a distant multi-person scene. Its generated
`official_pose.json` / `official_motion.json` have mean confidences of 0.04,
0.10, and 0.14 for left shoulder, elbow, and wrist respectively; all are
invisible at the default threshold. `official_retarget.log` records that the
quality gate rejected the retarget operation (no `--skip-quality-check` was
passed here) and no `official_animation.json` was written — this is the
gate working as intended, unmodified. The root-level `QUALITY_AUDIT.md`
tracks this and the remaining production-readiness gaps.
