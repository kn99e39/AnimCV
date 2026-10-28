# Worklog 99A — Runtime Parallelization and Vectorization (2026-10-28)

## Scope and starting state

**Branch and HEAD.** `arch/single_frame_first`, starting HEAD `01c0dc3` (Worklog 99).

**Origin.** A user-directed side task: use parallelization and vectorization, including in the production path.

**What the user waived.** Byte-level reproduction of historical worklog numbers. The existing test suite was still run,
because it takes seconds.

**What was left alone:**
- historical experiment scripts (W93–W99), per the repo rule;
- model weights;
- schemas and production defaults other than the new batch-size fields below.

**Two production paths are in scope:**
1. **The public CLI/GUI chain:** frames → MMPose 2D → motion graph → retarget → optimize → Blender export.
2. **The FramePose library modules in `src/`:** FramePose FK, two-bone IK, direction candidate, contact and semantics.
   Today only scripts call them; the CLI and GUI do not run FramePose.

## Changes

### Shared helper

`src/common/parallel.py` provides:
- `thread_map` for I/O and GIL-releasing native code;
- `process_map` for independent pure-Python work, such as per-sequence replays in future experiment scripts;
- `worker_count`.

Both map functions preserve input order. `ANIMCV_WORKERS=1` forces serial execution everywhere.

### Public CLI/GUI path

| Stage | File | Change |
|---|---|---|
| Frame extraction | `app/cli.py`, `ui/gui_app.py` | PNG writes on a thread pool (cv2 encoding releases the GIL). The GUI still reports progress as writes complete |
| Frame loading | `mediaio/video_loader.py` | Image decoding on a thread pool; order preserved |
| 2D pose (MMPose + RTMDet) | `pose/mmpose_adapter.py` | Detector-first tracking in three phases, see below |
| Depth (Depth Anything V2) | `pose/depth_estimator.py`, `app/cli.py`, `ui/gui_app.py` | `infer_frames` batches same-sized frames (`batch_size`, default 4) with upstream's own `image2tensor` and the same bilinear resize as `infer_image`. It falls back to per-frame inference when batching is off or the model has no `image2tensor` |
| Retarget median filter | `retarget/temporal_filter.py` | Per-landmark window medians with `sliding_window_view`. Same filtering rule (whole window present and visible; 3D median only when every neighbour has 3D) |
| Keyframe importance | `optimize/importance.py` | Velocity, acceleration, confidence deltas and extrema as array operations |
| Keyframe collapse (RDP) | `optimize/collapse.py` | In-segment interpolation errors (slerp angle or lerp distance) evaluated for all samples at once. The recursion became an explicit stack |
| Blender keyframing | `blender/keyframe_writer.py` | Parent-track lookup by bisection over a once-built index, replacing a full scan per sample (O(N²) → O(N log N)). The per-sample `scene.frame_set` was removed: `keyframe_insert` already takes the frame, and the rest matrices it uses are frame-independent |

**MMPose three-phase flow:**
1. Detection over all frames, `batch_size` frames per forward pass. The default is 8, set in `MMPoseConfig`.
2. Subject selection in frame order. It stays serial because the tracker uses the previous box, but it is cheap.
3. Top-down pose on the selected boxes, `batch_size` crops per forward pass.

The batching uses the same test pipelines and collation as `inference_detector` / `inference_topdown`. Those APIs run
one image per pass, or only the boxes of one image.

### FramePose library

| Module | Change | Output |
|---|---|---|
| `retarget/framepose_fk.py` | `_rest_direction` / `_rest_rotation` cached per `RestBone`, a frozen, hashable value object; `solve_framepose_fk` hoists per-bone rest values and mapped parents out of the frame loop | bit-identical (same functions, same inputs, computed once) |
| `retarget/framepose_two_bone_ik.py` | uses the cached rest values; joint-name lookup cached | bit-identical |
| `retarget/framepose_direction_candidate.py` | rest axis, rest quaternion (recursive matrix products), mapped ancestor and rest world axis computed once per bone; the contract-gap reasons and their order are preserved; joint lookup cached | bit-identical |
| `pose/contact_time_aware.py` | height-window std over a bisected slice when timestamps are monotonic (was a full scan per frame, O(N²)); old path kept for non-monotonic input | identical values (same elements, same order) |
| `motion/animation_semantics_bridge.py` | per-frame yaw computability tested with the fused yaw observation directly, instead of running the whole root-motion estimator per frame | same decision: the estimator raises exactly when that observation raises |

The rest-value caches also speed up every script that imports these helpers.

## Measurements (local M1 Pro)

**MMPose on CPU** (16 real 3DPW frames, models warm):

| Path | Time |
|---|---|
| Previous per-frame path | 4.13 s |
| Batch 1 | 4.63 s |
| Batch 4 | 3.55 s |
| **Batch 8 (default)** | **3.15 s (−24%)** |
| Batch 16 | 9.90 s (memory-bound on this machine) |

- Landmarks match the per-frame path within 0.0015 px.
- `device=mps` fails in mmcv's NMS (`nms_impl: implementation for device mps:0 not found`). This happens in the old
  per-frame path too, so it is not caused by this change.

**Keyframe collapse** (20 tracks × 3000 frames): 1.47 s → 0.51 s, with the same kept-key count (10,957).

**Retarget median filter** (17 landmarks × 3000 frames, window 5): 0.24 s → 0.16 s.

**Not measured:**
- Blender keyframing (needs a Blender run). The change removes an O(N²) scan and one depsgraph evaluation per sample.
- Depth batching (the Depth Anything V2 source is not checked out locally).

## Deliberately not changed

- **Public-path FK/IK solvers (`retarget/fk_solver.py`, `ik_solver.py`):** small scalar maths per bone and frame that
  produces per-frame dataclasses. Hold-last-value semantics make it sequential per bone, so vectorizing would gain
  little.
- **Process-parallel keyframe collapse:** about 20 tracks per clip. Pickling a track costs more than collapsing it once
  vectorized.
- **Training-time device selection (`framepose/train.py`, `context_refiner.py`):** training runs on CUDA on
  LabServer63, and changing its fallback would alter training reproducibility.
- **Historical scripts:** not edited. New experiment scripts can use `process_map` across sequences.

## Tests

**New file:** `tests/test_parallel_vectorized_paths.py` (7 tests):
- order preservation and serial fallback of the helpers;
- vectorized RDP errors equal the scalar `_interpolation_error`, including the near-identity branch and sign flip;
- vectorized importance equals the scalar formula;
- the median filter equals its scalar definition, with occlusion, a missing landmark and missing 3D;
- bisected parent lookup;
- batched depth equals per-frame depth on a fake upstream-contract model;
- value-keyed rest caches.

**Changed test:** `test_cli.py` now patches `infer_frames` instead of `infer_frame`.

**Runs:**
- Full suite: 1003 passed, 57 skipped locally (system Python).
- With the torch-equipped `.venv`, the new, depth, MMPose-adapter and CLI tests pass (44).
- A real-model MMPose smoke run matched the old path (above).
- LabServer63 docker run: see the commit notes.

**Not run, per the user's instruction:** byte-level reproduction of historical worklog outputs. Most FramePose changes
are bit-identical by construction.

## Changed files

**New:**
- `src/common/parallel.py`
- `tests/test_parallel_vectorized_paths.py`
- this worklog

**Modified:**
- `src/app/cli.py`, `src/ui/gui_app.py`, `src/mediaio/video_loader.py`
- `src/pose/mmpose_adapter.py`, `src/pose/depth_estimator.py`, `src/pose/contact_time_aware.py`
- `src/retarget/temporal_filter.py`, `src/optimize/importance.py`, `src/optimize/collapse.py`
- `src/blender/keyframe_writer.py`
- `src/retarget/framepose_fk.py`, `src/retarget/framepose_two_bone_ik.py`, `src/retarget/framepose_direction_candidate.py`
- `src/motion/animation_semantics_bridge.py`
- `tests/test_cli.py`
