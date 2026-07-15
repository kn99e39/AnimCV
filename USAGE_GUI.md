# AnimCV Motion Tool — GUI Usage

Video-to-armature motion capture and retargeting. Run
`motion-tool-gui.exe` (Windows) or `motion-tool-gui.app` (macOS) — no
install needed, everything required is bundled in this folder.

The app is one window with 8 tabs, run roughly left to right — each
tab's output feeds the next one:

## 1. Frames
Pick an input video and an output folder, optionally a target FPS, then
**Extract Frames**. Produces a folder of numbered PNGs used by every
later step.

If you only want a specific clip inside a longer video as your
reference, you can either type start/end frame numbers directly, or —
more like a video editor — press **Load Preview** and drag the **Start
frame** / **End frame** sliders: the frame under whichever slider you're
moving shows live in the preview above, and the sliders stay in sync
with the numbers (edit either). The range is inclusive and uses
original-video frame indices; leave it blank/full for the whole video.

## 2. Pose
Point at the frames folder from step 1, plus an MMPose model config
and checkpoint file (`.py` + `.pth`). The app bundles MMPose itself,
but lets you use its tested default or choose another compatible model.

### Fastest option: the tested default

1. Click **Use Default Model (RTMPose-tiny)**. It fills in the matching
   config shipped with MMPose and the matching checkpoint name.
2. Leave Device as `cpu` unless you have a compatible CUDA setup.
3. Click **Run Pose Estimation**. On the first run the ~13 MB checkpoint
   downloads from OpenMMLab into `~/.cache/animcv/models`; later runs
   reuse that copy automatically.

### Use another MMPose model

1. Open the official [MMPose model zoo](https://github.com/open-mmlab/mmpose)
   and choose a **top-down, 2D body keypoint** model trained on the
   **COCO 17-keypoint** schema. AnimCV converts that schema into its
   own body landmarks and processes only the highest-confidence person
   in each frame.
2. Obtain the config and checkpoint explicitly listed as a pair for
   one model. Do not mix files from different rows, model sizes, input
   resolutions, or datasets.
3. Save the checkpoint (`.pth`) in a stable folder you control, such as
   `AnimCV-models/`. For the config, preserve its official relative
   directory structure and any `_base_` files it imports; copying only
   a single config file can make it fail to load. A config already
   included with your MMPose installation is also fine.
4. In the Pose tab, browse to that config `.py` and checkpoint `.pth`.
   Use `cpu`, or `cuda` only when your installed PyTorch/CUDA setup
   supports it.
5. Run a short frame range first and confirm that the detected landmarks
   look sensible before processing a full video.

If loading fails, first verify that the config still finds its `_base_`
files and that the checkpoint came from the exact same model-zoo entry.
An arbitrary pose, hand, face, bottom-up, or non-COCO checkpoint is not
interchangeable with this pipeline.

Optionally set a Depth Anything V2 checkpoint (`.pth`) to get real
3D-aware retargeting later instead of a 2D approximation — leave its
device on `auto`. **Run Pose Estimation** writes `pose.json`.

## 3. Rig
Point at your character rig file (`.fbx`, or anything Assimp can
read) and **Parse Rig**. Lists the bones found — needed for the
Mapping tab next. Requires the native `assimp` library installed on
this machine (not bundled — see Setup below); without it this tab
reports an error rather than crashing.

## 4. Mapping
Click-based bone mapping: load a frame + its detected landmarks, then
click a rig bone on the left and a landmark (or a second landmark, for
a rotation "direction" mapping) to assign it. Optional IK chains
(root→mid→end, e.g. shoulder→elbow→wrist) can be added on the right
for two-bone limbs. **Save Mapping** writes the mapping profile used
by Retarget.

## 5. Motion
Point at `pose.json` from step 2 and **Build Motion Graph**. Turns raw
per-frame landmarks into a track-based motion graph.

## 6. Retarget
Combines the motion graph (step 5), the rig file (step 3), and the
mapping profile (step 4) into an animation clip.

## 7. Optimize
Collapses the dense per-frame animation into sparse keyframes (RDP-style
simplification). Pick a preset (`light`/`medium`/`aggressive`) or
`custom` with your own error threshold; `none` skips this step.

## 8. Export
Writes the final animation into your rig file and saves a `.blend`
(optionally also an `.fbx`). Needs a real Blender install on this
machine — it's auto-detected from `PATH` or the default per-OS install
location, or point at one explicitly. Not bundled with this app.

## Setup this app does NOT do for you
- **Blender** (step 8 / Export): install from blender.org.
- **assimp** (step 3 / Rig, and therefore step 6 / Retarget): the
  native library, not just a Python binding — on Windows put
  `assimp.dll` on `PATH`; on macOS `brew install assimp`.

## Where things came from
This is the same pipeline as AnimCV's command-line tool
(`motion-tool`), just wrapped in a GUI — each tab calls the same
underlying code the CLI commands do. See the project's `README.md` /
`README_EXEC.md` (Korean) in the source repository for the full
architecture and CLI reference if you need more detail than this file.
