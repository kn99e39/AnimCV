# E2E pose-pipeline sample

This directory contains a small, reproducible end-to-end sample created on
2026-07-27 using the project's `.venv` (Python 3.11), the cached default
RTMPose-tiny model, and `person.mp4`.

`person.jpg` is MMPose's COCO test image. `person.mp4` repeats that image for
eight frames at 8 fps, so it is a smoke test of the video and pose pipeline,
not a motion-quality demonstration.

Generated artifacts:

- `frames/`: eight decoded PNG frames and `metadata.json`
- `pose.json`: 16 detected COCO landmarks per frame
- `motion.json`: the MotionGraph generated from those poses
- `demo_rig.blend` / `demo_rig.fbx`: a minimal three-bone Armature created
  specifically for this test
- `rig_profile.json` and `mapping.json`: the parsed rig profile and its
  left-arm landmark mapping
- `mapping_upper_arm_only.json`: a quality-gate-passing subset that excludes
  the low-confidence wrist in this particular still image
- `animation.json` / `animation_optimized.json`: raw and collapsed
  retargeted animation; each track changes from 8 to 2 keys
- `animated_rig.blend` / `animated_rig.fbx`: the final Blender-exported
  artifacts

Reproduce the generated artifacts from the repository root:

```bash
.venv/bin/python -m app.cli extract-frames \
  --video examples/e2e_demo/person.mp4 --out examples/e2e_demo/frames
.venv/bin/python -m app.cli estimate-pose \
  --frames examples/e2e_demo/frames --out examples/e2e_demo/pose.json --device cpu
.venv/bin/python -m app.cli build-motion \
  --pose examples/e2e_demo/pose.json --out examples/e2e_demo/motion.json
LD_LIBRARY_PATH=/opt/homebrew/opt/assimp/lib .venv/bin/python -m app.cli retarget \
  --motion examples/e2e_demo/motion.json --rig examples/e2e_demo/demo_rig.fbx \
  --mapping examples/e2e_demo/mapping.json --out examples/e2e_demo/animation.json
.venv/bin/python -m app.cli optimize \
  --animation examples/e2e_demo/animation.json --collapse medium \
  --out examples/e2e_demo/animation_optimized.json
.venv/bin/python -m app.cli export-blender \
  --animation examples/e2e_demo/animation_optimized.json \
  --rig examples/e2e_demo/demo_rig.blend --out examples/e2e_demo/animated_rig.blend \
  --fbx-out examples/e2e_demo/animated_rig.fbx
```

For the final generated `.blend`, Blender 5.1.2 reports an action named
`Generated_Motion` containing eight quaternion F-curves: four for
`upper_arm.L` and four for `forearm.L`, with two keyframes in each curve.

`quality_report.json` records a later quality-gate pass for the upper-arm-only
mapping: 100% visibility, mean confidence 0.400, and a maximum consecutive
direction change of 3.366 degrees. `animation_quality_filtered.json` is the
corresponding retarget output with the default three-frame smoothing enabled.

On Apple Silicon Homebrew, the currently installed pyassimp version needs
`LD_LIBRARY_PATH=/opt/homebrew/opt/assimp/lib` when parsing the FBX. This is
an environment/library-discovery issue, not part of the sample data.
