# AnimCV

Assisted video-to-armature motion capture and retargeting tool. See
[Architecture_v2.md](Architecture_v2.md) for the full design.

Status: **All 7 milestones from Architecture_v2.md section 11 are
implemented, plus a post-v2 quality pass (rest-pose axis correction,
depth-assisted 3D retargeting, 2-bone IK — see "Beyond v2" below).**
Core data model/JSON serialization, the video/pose pipeline, an
Assimp-based `RigParser`, a CLI-driven interactive bone mapper, a
retarget solver, an RDP-style keyframe importance/collapse optimizer,
and a real headless-Blender executor are all in place. See
`result/result_mil*.txt` for a detailed log of what was actually
implemented/verified at each milestone (including several real bugs
only found by testing against an actual Blender/torch install — see
`result/result_mil7.txt` and `result/result_mil8.txt`).

## Beyond v2

Architecture_v2.md section 1.3/14.1 explicitly excludes Depth Anything
V2 from v2's scope. It's used anyway, added on explicit request after
all 7 milestones were done, to improve retargeting quality — see
`result/result_mil8.txt` for the full writeup and why this is flagged
as a deliberate deviation, not an oversight. Also added, neither in
v2's schema: 2-bone IK chains (`rig.bone_mapping.IKChainEntry`,
`retarget/ik_solver.py`) and rest-pose axis correction
(`retarget/axis_utils.apply_rest_pose_correction*`).

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
.venv/Scripts/activate    # Windows
source .venv/bin/activate # macOS / Linux
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

`estimate-pose --depth-checkpoint` needs `torch` + Depth Anything V2's
own dependencies (see `third_party/Depth-Anything-V2/requirements.txt`)
and a downloaded checkpoint — `vits` is the only Apache-2.0 size, the
rest are CC-BY-NC-4.0. Leave `--depth-device` at its default `auto`:
Depth Anything V2's own `infer_image` always places its input tensor on
whichever of cuda/mps/cpu is available with no way to override that, so
an explicit device that doesn't match will raise a clear error rather
than crash deep in the model's forward pass (see `result/result_mil8.txt`).

## Running tests

```bash
pytest
```

## CLI

```bash
python -m app.cli extract-frames --video input.mp4 --out cache/frames
python -m app.cli estimate-pose --frames cache/frames --out cache/pose.json \
  --pose-config third_party/mmpose/configs/.../some_config.py \
  --pose-checkpoint /path/to/checkpoint.pth \
  --depth-checkpoint /path/to/depth_anything_v2_vits.pth  # optional, for 3D-aware retargeting
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
`PATH`, then a per-OS default install location (Windows:
`Program Files\Blender Foundation\Blender <version>\blender.exe`;
macOS: `/Applications/Blender*.app` or `~/Applications/Blender*.app`;
Linux: `/usr/bin/blender`, `/opt/blender/blender`, Snap/Flatpak paths —
see `_default_blender_search_paths` in `src/app/cli.py`), or takes an
explicit `--blender-executable`. `--fbx-out` is optional; without it,
only the `.blend` is written.

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

`retarget` is FK-based (section 7.2/7.4): `direction`-mode mappings
become a bone rotation between two tracked points (relative to the
first visible frame); `landmark`/`point`-mode mappings become a
translation offset from that single point's reference-frame position.
When depth was sampled (`estimate-pose --depth-checkpoint`, see
`result/result_mil8.txt`), both automatically switch from the
2D-image-plane approximation to a real 3D rotation/offset wherever 3D
data is available for the relevant points, falling back to 2D
otherwise. Every rotation/translation is also re-expressed in each
target bone's own local space using `RigProfile`'s `rest_local_matrix`
when present (`retarget/axis_utils.apply_rest_pose_correction*`).
`mapping_profile.ik_chains` (not in Architecture_v2.md's schema — see
`rig.bone_mapping.IKChainEntry`) additionally run a 2-bone analytic IK
solve (`retarget/ik_solver.py`) for root->mid->end chains like
shoulder-elbow-wrist. Bones/chains the mapping profile names but the
rig doesn't have, or an unrecognized `mapping_mode`, are silently
skipped rather than raising — partial mapping is expected, not an
error (section 6.5).

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

After the per-bone loop, a second prompt collects optional IK chains
(not in Architecture_v2.md's schema), one per line:

```text
ik-chain> (<root_bone> <mid_bone> <end_bone> <root_source> <mid_source> <end_source> [root_axis] [mid_axis] | done) upper_arm.L forearm.L hand.L left_shoulder left_elbow left_wrist
```

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

## Building a standalone executable

OS-specific scripts live under `windows/` and `mac/` (not mixed at the
repo root) so it's never ambiguous which one to run on which machine.
Both sets assume the project root as their working directory even
though they live one level down — each hops back to it itself
(`cd /d "%~dp0.."` / `cd "$(dirname "${BASH_SOURCE[0]}")/.."`), so
`windows\build_windows.ps1` / `bash mac/build_full_mac.command` work the
same whether invoked from the repo root or by double-clicking inside
the subfolder.

`windows/build_windows.ps1` builds a PyInstaller onedir bundle into
`build_output/windows/motion-tool/motion-tool.exe`. It only bundles
the base `dependencies` (numpy, opencv-python, pyyaml) — the heavy
optional extras (mmpose, depth, pyassimp's native lib) are not
bundled, same tradeoff as the Setup section above. `export-blender`
still needs a real Blender install on the machine running the exe;
`scripts/` and `src/` are copied next to the exe as plain files (not
just frozen into it) because `scripts/apply_motion.py` runs under
Blender's own bundled Python as a subprocess, not inside this exe's
Python, and inserts `src/` into `sys.path` itself. See
`result/result_windows_build.txt` for what was verified.

For a build that also bundles mmpose/depth-anything-v2 (so
`estimate-pose` works without a separate `pip install` on the target
machine), use `windows/build_full_windows.bat` (Windows, CUDA torch) or
`mac/build_full_mac.command` (macOS, CPU torch — no CUDA on Mac). These pull in
several GB of dependencies (torch + mmcv/mmengine/mmdet) and the
PyInstaller-frozen bundle itself was not exercised end-to-end against a
real mmpose checkpoint before shipping either script — mmcv/mmdet's
registry-based dynamic imports are a known PyInstaller pain point, see
the NOTE each script prints after building. (Running from a plain venv,
not a frozen exe, *was* verified against a real checkpoint on macOS —
see `mac/setup_mac.command` and README_EXEC.md's Mac support section.)
Neither script installs pyassimp's native `assimp` library (for
`parse-rig`) — that's a manual step (see Setup above /
`brew install assimp` on Mac).

For just running the CLI from source on Mac (no bundled exe, much
faster to iterate on than a PyInstaller rebuild), use
`bash mac/setup_mac.command` instead — a one-time setup that installs
everything needed (including the mmpose/mmcv/mmdet stack's several
macOS-specific build workarounds) into a normal `.venv`.

## Third-party references

`third_party/` holds reference checkouts of upstream repos used as
implementation references (not installed as dependencies):

- `third_party/mmpose` — pose estimation (S-Tier dependency, section 3.2)
- `third_party/Depth-Anything-V2` — used by `pose/depth_estimator.py`
  for optional 3D-aware retargeting (see "Beyond v2" above); v2 itself
  excludes it (section 1.3, 14.1), used here on explicit instruction
