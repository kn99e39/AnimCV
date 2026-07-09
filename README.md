# AnimCV

Assisted video-to-armature motion capture and retargeting tool. See
[Architecture_v2.md](Architecture_v2.md) for the full design.

Status: **Milestone 7 — Blender export — all 7 milestones from
Architecture_v2.md section 11 are now implemented.** Core data
model/JSON serialization, the video/pose pipeline, an Assimp-based
`RigParser`, a CLI-driven interactive bone mapper, a 2D-direction-driven
FK retarget solver, an RDP-style keyframe importance/collapse
optimizer, and a real headless-Blender executor are all in place. See
`result/result_mil*.txt` for a detailed log of what was actually
implemented/verified at each milestone (including two real bugs only
found by testing against an actual Blender install — see
`result/result_mil7.txt`).

## Deviation from Architecture_v2.md section 9

The `io/` package was renamed to `mediaio/`. Since each `src/*` folder is
imported as a top-level package (e.g. `from mediaio.frame_sequence import
Frame`), a package literally named `io` shadows the Python standard
library's own `io` module for any code importing it after `src` is added
to `sys.path` — `pathlib`, `json`, etc. rely on stdlib `io` internally.
Everything else matches the documented structure.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -e ".[dev]"
```

Blender is optional at this stage — the project must run without it
(Milestone 1 acceptance criteria). MMPose is only required for
`estimate-pose`; install it with `pip install -e ".[pose]"` (heavy:
pulls in mmcv/mmengine/mmdet/torch). `estimate-pose` also needs a model
config and checkpoint (`--pose-config` / `--pose-checkpoint`) from the
MMPose model zoo — see `third_party/mmpose/configs`.

`parse-rig` needs `pyassimp` plus the native `assimp` shared library
installed on your system (not just `pip install pyassimp` — that only
gets you the Python binding). Without the native library, `pyassimp`
raises at import time, which the CLI reports as a normal error rather
than crashing.

## Running tests

```bash
pytest
```

## CLI

```bash
python -m app.cli extract-frames --video input.mp4 --out cache/frames
python -m app.cli estimate-pose --frames cache/frames --out cache/pose.json \
  --pose-config third_party/mmpose/configs/.../some_config.py \
  --pose-checkpoint /path/to/checkpoint.pth
python -m app.cli parse-rig --rig character.fbx --out cache/rig_profile.json
python -m app.cli create-mapping --rig character.fbx --frame cache/frames/00000.png --out profiles/mapping.json
python -m app.cli build-motion --pose cache/pose.json --out cache/motion_graph.json
python -m app.cli retarget --motion cache/motion_graph.json --rig character.fbx --mapping profiles/mapping.json --out cache/animation.json
python -m app.cli optimize --animation cache/animation.json --collapse medium --out cache/animation_optimized.json
python -m app.cli export-blender --animation cache/animation_optimized.json --rig character.fbx --out result.blend
```

Every command runs for real. `parse-rig` is not in Architecture_v2.md
section 10's CLI list — it was added because Milestone 3's acceptance
criteria requires producing an inspectable `rig_profile.json` as a
standalone artifact (consistent with "make every pipeline stage
cacheable to JSON", section 14.9), and no listed command covers that
step on its own.

`export-blender` shells out to the real `blender` executable
(`--background --python scripts/apply_motion.py -- ...`, section
4.10's "Headless Batch Mode") since this project's own venv Python
cannot `import bpy`. It autodetects Blender via `BLENDER_EXECUTABLE`,
`PATH`, or the default Windows install location under
`Program Files\Blender Foundation`, or takes an explicit
`--blender-executable`. `--fbx-out` is optional; without it, only the
`.blend` is written.

`optimize` runs an RDP-style simplification (section 8.3): the
first/last sample of every bone track, any frame passed as
`locked_frames` (Python API only — `optimize/collapse.py`'s
`collapse_animation_clip`/`collapse_track` accept it, but the CLI
doesn't expose a per-frame-lock flag yet, see `result/result_mil6.txt`),
and any frame whose importance score (section 8.2: angular velocity +
acceleration + a confidence-delta proxy + local extrema, weighted,
since this MVP has no reconstruction-error or visibility-change signal
to score against) is >= 0.75 are kept unconditionally; everything else
is dropped if the slerp/linear interpolation error between its
neighbors stays under the `--collapse` preset's threshold
(`light`/`medium`/`aggressive`, or `custom` with `--threshold`). `none`
disables collapse entirely.

`retarget` is a 2D-direction-driven FK solver (section 7.2/7.4, no
depth): `direction`-mode mappings become a bone rotation from the
image-plane angle between two tracked points (relative to their angle
in the first visible frame); `landmark`/`point`-mode mappings become a
translation offset from that single point's reference-frame position.
Bones the mapping profile names but the rig doesn't have, or an
unrecognized `mapping_mode`, are silently skipped rather than raising —
partial mapping is expected, not an error (section 6.5).

`create-mapping` prompts once per rig bone (in name order) on stdin:

```text
upper_arm.L> (landmark <name> | direction <a> <b> | custom_point <id> | skip) direction left_shoulder left_elbow
forearm.L> (landmark <name> | direction <a> <b> | custom_point <id> | skip) direction left_elbow left_wrist
hips> (landmark <name> | direction <a> <b> | custom_point <id> | skip) landmark pelvis
```

Type `skip` (or just press enter) to leave a bone unmapped, or `done`
to stop early and keep whatever was answered so far. See
`src/ui/mapping_ui.py` for the full command grammar; a bone *chain*
(section 6.3) is just several consecutive `direction` answers.

## Blender integration

Only `src/blender/*.py` and `scripts/apply_motion.py` import `bpy`, and
only inside function bodies (Architecture_v2.md section 3.4: core logic
must stay importable without Blender). `write_keyframes` keyframes
`rotation_quaternion` for direction-mapped bones and `location` for
landmark/point-mapped bones, then sets every fcurve's keyframe
interpolation to Bezier so the result is graph-editor-editable, not a
wall of linear steps.

Verified against real local installs of both Blender 4.5 LTS and 5.1:
build a small in-Blender armature fixture, retarget+optimize a
synthetic swinging-arm animation onto it via the full CLI pipeline, and
reopen the output `.blend` headlessly to confirm the exact expected
fcurve/keyframe/interpolation counts are present. This surfaced two real
bugs no amount of code review would have caught without an actual
Blender to run against — see `result/result_mil7.txt` for both:

1. Blender 4.4+'s layered animation system removed the flat
   `Action.fcurves` accessor (nested under `layers -> strips ->
   channelbags -> fcurves` instead); 4.5 still has both, 5.1 only has
   the new one.
2. `blender --background --python script.py` exits with code 0 even
   when the script raises an unhandled (non-`SystemExit`) exception, so
   `apply_motion.py` now catches broadly and returns an explicit exit
   code, and the CLI additionally checks that the expected output file
   was actually created.

## Third-party references

`third_party/` holds reference checkouts of upstream repos used as
implementation references (not installed as dependencies):

- `third_party/mmpose` — pose estimation (S-Tier dependency, section 3.2)
- `third_party/Depth-Anything-V2` — kept for future extension reference
  only; explicitly excluded from v2 scope (section 1.3, 14.1)
