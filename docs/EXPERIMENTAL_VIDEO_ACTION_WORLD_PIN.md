# EXPERIMENTAL — Video → Editable Blender Action → Left-Wrist World Pin (opt-in)

Status: **EXPERIMENTAL, opt-in, introduced in Batch 124** (evidence: `docs/124_WORKLOG_OPT_IN_VIDEO_TO_EDITABLE_ACTION_WORLD_PIN.md`).
It is **not** the default pipeline. The default `motion-tool` commands and the GUI (MotionGraph → RetargetSolver →
AnimationClip → `export-blender`) are unchanged and never call this code. It is not a general retargeter, not a trained
generator, not a style system, and not a quality guarantee.

## What it does

One bounded workflow on the **known W105 BaseRig only** (`examples/skinned_mannequin/BaseRig_Skinned.fbx`):

1. **build** — authenticate one source clip (SHA-256 you declare, a provenance record, declared fps), check the container's
   presentation timestamps, run the **frozen** W102 M1 + S0 perception live, map the S0 hypothesis to the BaseRig with the
   **frozen** W105 GT-free FIXED-slot adapter, and write a **new** `baseline.blend` with one editable Action. A second Blender
   process re-opens the file and reads it back (F-curves, interpolation, timebase, world pose at integer frames and at
   subframes 0.25/0.5/0.75).
2. **propose** — one user-authored `WORLD_POSITION_PIN` request for `left_wrist`, in Blender scene-world metres, on an
   inclusive source-frame interval. The frozen W116 contract validates it, the frozen W116 analytic two-bone resolver
   changes **only** `upperarm_l`/`lowerarm_l` on the pin frames (no automatic out-of-pin edit), the candidate is keyed into
   its own `.blend`, read back independently, and gated (hard pin ≤ 1 mm *experimental diagnostic*, scope, locality, root,
   bone lengths, joint limit, truthful footprint). Nothing is committed. A timeline PNG shows requested pin frames, the
   permitted domain, the **actual** changed frames/bones from Blender, and the gate status.
3. **commit** — explicit `--decision accept|reject`. Atomic; a stale parent is refused; a gate-rejected candidate can never
   be accepted. An accepted edit is copied to `accepted/accepted_NN_<id>.blend`; the baseline file is never modified.
4. **revert** — creates a new state restoring the head's parent values and a new `.blend`; its readback must be bit-equal
   to the restored state's.
5. **review** — 30 fps side-by-side video: original source | baseline Action | accepted edited Action, with the per-frame
   footprint overlaid.
6. **fbx** — exports a state's `.blend` through the existing `blender.export.export_fbx` and re-imports it with the existing
   default importer, and **reports** integer-frame/subframe/interpolation/timebase differences. Nothing is repaired: an FBX
   file on disk is not a round-trip PASS (see W107–W112 for the known failures).

## Requirements

- Blender 5.1 (`--blender` or `ANIMCV_BLENDER`); the project Python environment with OpenCV, numpy, matplotlib, torch and
  mmpose (for live perception).
- The frozen perception assets, described in a JSON like `examples/experimental_video_action/perception_assets.template.json`.
  Their SHA-256 identities are fixed in `src/app/experimental_video_action.py` (`PERCEPTION_EXPECTED`); any mismatch stops.
- A source clip at **exactly 30 fps** with container presentation steps within 1 ms of 1000/30. Other cadences, variable
  frame rate, or a declared fps that disagrees with the container are refused (`UNSUPPORTED_*`), never resampled.
- Run from the repository root with `PYTHONPATH=src` (an installed `app` package from another checkout would shadow it).

## Example

```bash
export PYTHONPATH=src
PY=python   # the project environment
$PY -m app.cli experimental-video-action build \
    --clip /data/clip.webm --clip-sha256 <sha256> --source-record source.json --declared-fps 30 \
    --segment 79 318 --perception-assets perception_assets.json --blender "<blender.exe>" --out runs/my_run
# copy the printed baseline_state_id into a request (see examples/experimental_video_action/left_wrist_pin_request.example.json)
$PY -m app.cli experimental-video-action propose --run runs/my_run --request my_pin.json
$PY -m app.cli experimental-video-action commit  --run runs/my_run --request-id hold_left_wrist_example --decision accept
$PY -m app.cli experimental-video-action review  --run runs/my_run --request-id hold_left_wrist_example
$PY -m app.cli experimental-video-action fbx     --run runs/my_run --state head
$PY -m app.cli experimental-video-action revert  --run runs/my_run
$PY -m app.cli experimental-video-action status  --run runs/my_run
```

`source.json` must contain `title`, `license`, `author`, `source_uri`. The slice records these values; it does not judge them.

## Request format

```json
{"schema": "animcv_exp_world_pin_request_v1", "request_id": "hold_1", "parent_state_id": "<current head>",
 "type": "WORLD_POSITION_PIN", "semantic_joint": "left_wrist", "coordinate_frame": "SCENE_WORLD", "unit": "m",
 "frames": [210, 218],
 "target": {"kind": "CAPTURED_FROM_EVALUATED_STATE", "capture_frame": 210},
 "authorized_channels": ["lowerarm_l", "upperarm_l"]}
```

- `target.kind` is `CAPTURED_FROM_EVALUATED_STATE` (hold the wrist where Blender evaluates it at `capture_frame`) or
  `CAPTURED_PLUS_DECLARED_OFFSET` with `offset_m` = [x, y, z] in Blender scene-world metres. Absolute targets are not
  supported in this slice (frozen W116 target-provenance contract).
- Only `left_wrist` and exactly `{upperarm_l, lowerarm_l}` are supported. Other joints, ambiguous bindings, unknown joints,
  other coordinate frames/units, non-finite numbers, frames outside the Action, and a stale `parent_state_id` are rejected
  with a reason and no change.

## Output directory (one run; nothing is overwritten)

```
run/
  run.json                       run record (immutable-file hashes)
  manifest/input_manifest.json   read-only: clip identity, timebase, perception identity, rig identity, frozen-code hashes
  manifest/inspection.json       clip + decoded-frame identity for the frozen diagnostic
  perception/m1s0/               frozen M1+S0 outputs (live) and log
  rig/video_cases.json           W105 FIXED-slot per-frame rig values
  baseline/baseline.blend        unedited Action (immutable)  + action_sidecar.json, bindings.json, build/readback reports
  edits/<request_id>/            request.json, proposal.json, candidate.blend (if any), timeline.png, commit.json, timeline_final.png
  accepted/accepted_NN_<id>.blend
  reverted/reverted_<id>.blend   + exact-undo report
  lineage/                       content-addressed read-only states, refs.json, log.jsonl, blends.json
  review/<request_id>/           30 fps A/B video, stills, render logs
  fbx/<state>/                   exported .fbx, re-import readback, round-trip report
```

## Known limitations (do not read the output as more than it is)

- **Partial rig.** Driven: Armature Object rotation, `spine_01`, `thigh_l/r`, `calf_l/r`, `upperarm_l`, `lowerarm_l`.
  The right arm, upper spine, neck/head, clavicles, hands, fingers and feet stay at rest. No root translation, no foot
  contact. Left/right arm assignment and 3D accuracy are unverified GT-free hypotheses; subject identity is `UNKNOWN`.
- **Keys are CONSTANT** (frozen keying rule): playback holds each source frame; subframes equal the previous integer frame.
- **Hard pin only on the pin frames.** Release/entry discontinuities at the pin boundaries are not smoothed (no automatic
  transition edit in this slice); they are disclosed in `proposal.json` (`quality_disclosure`).
- **The 1 mm pin check is an experimental diagnostic**, not a Production tolerance.
- **FBX:** the existing exporter/importer are used unchanged; integer-frame drift, CONSTANT→LINEAR interpolation and scene
  range loss are expected (W107–W112) and reported, not fixed.
- Only the BaseRig is supported. Any other rig, cadence, joint or edit type is out of scope.
