# E2E pose-pipeline sample

This directory contains a small, reproducible end-to-end sample built on
Windows (Python 3.11) using the cached default RTMPose-tiny model,
`person.mp4`, and `examples/BaseRig.fbx` (a real UE-mannequin-style
skeleton, 53 bones — used in place of the earlier throwaway 3-bone
armature this demo originally shipped with).

`person.jpg` is MMPose's COCO test image. `person.mp4` repeats that image for
eight frames at 8 fps, so it is a smoke test of the video and pose pipeline,
not a motion-quality demonstration.

Generated artifacts:

- `frames/`: eight decoded PNG frames and `metadata.json`
- `pose.json`: 16 detected COCO landmarks per frame
- `motion.json`: the MotionGraph generated from those poses
- `rig_profile.json`: `examples/BaseRig.fbx` parsed via `parse-rig`
- `mapping.json`: maps `upperarm_l`/`lowerarm_l` (BaseRig's left-arm bones)
  from the `left_shoulder`/`left_elbow`/`left_wrist` landmarks
- `mapping_upper_arm_only.json`: a quality-gate-passing subset that excludes
  the low-confidence wrist in this particular still image
- `animation.json` / `animation_optimized.json`: raw and collapsed
  retargeted animation (built with `--skip-quality-check` since the still
  image's wrist landmark alone would otherwise fail the quality gate);
  each track changes from 8 to 2 keys
- `animated_rig.blend` / `animated_rig.fbx`: the final Blender-exported
  artifacts, built directly from `examples/BaseRig.fbx`

Reproduce the generated artifacts from the repository root:

```bash
python -m app.cli extract-frames \
  --video examples/e2e_demo/person.mp4 --out examples/e2e_demo/frames
python -m app.cli estimate-pose \
  --frames examples/e2e_demo/frames --out examples/e2e_demo/pose.json --device cpu
python -m app.cli build-motion \
  --pose examples/e2e_demo/pose.json --out examples/e2e_demo/motion.json
python -m app.cli parse-rig \
  --rig examples/BaseRig.fbx --out examples/e2e_demo/rig_profile.json
python -m app.cli retarget \
  --motion examples/e2e_demo/motion.json --rig examples/BaseRig.fbx \
  --mapping examples/e2e_demo/mapping.json --out examples/e2e_demo/animation.json \
  --skip-quality-check
python -m app.cli optimize \
  --animation examples/e2e_demo/animation.json --collapse medium \
  --out examples/e2e_demo/animation_optimized.json
python -m app.cli export-blender \
  --animation examples/e2e_demo/animation_optimized.json \
  --rig examples/BaseRig.fbx --out examples/e2e_demo/animated_rig.blend \
  --fbx-out examples/e2e_demo/animated_rig.fbx
```

For the final generated `.blend`, Blender 5.1 reports an action named
`Generated_Motion` containing eight quaternion F-curves: four for
`upperarm_l` and four for `lowerarm_l`, with two keyframes in each curve
(verified by reopening the `.blend` headlessly and inspecting the action).

`quality_report.json` records a quality-gate pass for the upper-arm-only
mapping: 100% visibility, mean confidence 0.400, and a maximum consecutive
direction change of 3.37 degrees. `animation_quality_filtered.json` is the
corresponding retarget output (no `--skip-quality-check` needed since this
one bone alone passes the gate):

```bash
python -m app.cli retarget \
  --motion examples/e2e_demo/motion.json --rig examples/BaseRig.fbx \
  --mapping examples/e2e_demo/mapping_upper_arm_only.json \
  --out examples/e2e_demo/animation_quality_filtered.json \
  --quality-report examples/e2e_demo/quality_report.json
```

`parse-rig`/`retarget` need `pyassimp` plus the native `assimp` shared
library on your system (not just `pip install pyassimp`) — see the
project's main `README.md`.
