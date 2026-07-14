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
and checkpoint file (`.py` + `.pth`). Don't have one? Click **Use
Default Model (RTMPose-tiny)** to fill both in automatically — the
checkpoint (~13MB) downloads once to a local cache and is reused after
that. To use a different model instead, download a matching config +
checkpoint pair from the
[MMPose model zoo](https://github.com/open-mmlab/mmpose) and point at
those files directly. Device is `cpu` unless you have a matching CUDA
GPU. Optionally set a Depth Anything V2 checkpoint (`.pth`) to get
real 3D-aware retargeting later instead of a 2D approximation — leave
its device on `auto`. **Run Pose Estimation** writes `pose.json`.

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
