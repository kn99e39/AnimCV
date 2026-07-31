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
