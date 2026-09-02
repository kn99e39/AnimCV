# Architecture v3 — Frame-First Pose Core

## Status

This document supersedes `Architecture_v2.md` **only for the learning core**.
Everything `Architecture_v2.md` says about video intake, rig parsing, bone
mapping, keyframe collapse and Blender export is unchanged and still binding.

What changes here is the primary supervised abstraction of AnimCV's perception
stage.

## 1. What was demoted, and why

### 1.1 The previous primary abstraction

```
Temporal 2D Joint Observations  ->  Temporal Lifter  ->  center-frame Canonical 3D Pose
```

`src/training/temporal_lifter.py` (`dilated_tcn_v1`, receptive field 127) is a
sequence-in / one-pose-out model. It consumes an 81-frame window of normalized
2D joints and emits the canonical root-relative 17-joint 3D pose of the window's
center frame.

### 1.2 Why it is no longer the primary architecture

The A5–A16 program (docs/09–docs/22) established, with fingerprinted and
controlled runs, that:

- Root/torso orientation error (`root_yaw_p95_degrees`) never reached the
  promotion gate under any loss variant tried (A6, A7, A10, A11, A12, A14, A16).
- The dominant residual is **monocular forward-depth ambiguity**: the model
  cannot decide the sign of the bilateral forward-depth difference
  `D = (y_right - y_left)/sqrt(2)` for shoulders and hips (docs/18, docs/21).
- Auxiliary orientation losses either did not move the tail (docs/12–13:
  gradient ratio 3,170x, tail already small at convergence) or damaged pose
  geometry (docs/10 A6/A7, docs/14 A12).
- Generalization was source-dependent (docs/17, docs/22 A16), and loss,
  readout and temporal effects were entangled to the point that single-hypothesis
  attribution required a dedicated diagnostic script per candidate.

None of these are tuning failures. They are **information** failures. A stream
of 2D joint coordinates does not contain the evidence needed to disambiguate
which shoulder is nearer the camera. That evidence exists in the RGB frame and
was discarded before the model ever saw it:

```
foreshortening, self-occlusion, limb overlap, body-facing cues, silhouette,
clothing/body contour, near/far limb ordering, shading and other monocular
depth cues
```

Adding more temporal context does not create this evidence; it averages over
frames that are each individually ambiguous.

### 1.3 The temporal lifter's new role

The temporal lifter is **preserved unchanged** and reclassified as the
**Legacy Temporal Pose Baseline**:

1. a reproducible historical baseline (A9 remains the fingerprinted reference);
2. a future temporal-context provider for Layer B;
3. a future temporal-refinement component for Layer C;
4. a diagnostic comparison for any frame-first result.

No historical checkpoint, fingerprint, report, loss implementation, evaluator
contract or diagnostic script is renamed, rewritten or re-measured by this
document. `src/training/temporal_lifter.py` and every `scripts/*` diagnostic
built on it stay exactly as executed.

## 2. The new primary abstraction

```
Frame-centric RGB + Explicit 2D Geometry  ->  Canonical Root-relative 3D Pose
```

The fundamental supervised prediction unit is:

```
one frame -> one 3D pose
```

Temporal continuity is explicitly deferred. A frame that is geometrically wrong
cannot be repaired by smoothing it against its neighbours; it can only be made
consistently wrong.

## 3. Layered contract

Only **Layer A** is implemented as the new core. Layer B interfaces are reserved
where free. Layers B–E are not implemented in this batch.

### Layer A — Frame Pose Core (implemented)

```
one frame  ->  one root-relative canonical 17-joint 3D pose
```

Inputs: explicit 2D joint geometry (always) and, optionally, an RGB-derived
visual representation of the same frame. Output: `(17, 3)` root-relative metric
joint positions in the canonical camera frame.

Module: `src/framepose/`.

### Layer B — Optional Temporal Context (reserved, not implemented)

```
additional neighbouring-frame evidence  ->  improve Pose_t
```

May consume several frames, still predicts exactly one target frame. The frame
sample contract therefore carries `sequence_id`, `frame_index`, `fps` and
explicit `neighbors` references so a future Layer B needs no re-ingest.

### Layer C — Temporal Stabilization (not implemented)

```
individually good frame poses  ->  temporally coherent sequence
```

Layer C must never become responsible for fixing frame geometry. Temporal
smoothness is explicitly **not** a promotion criterion for Layer A.

### Layer D — Animation Semantics (not implemented)

Root motion, contact, foot locking, constraints.

### Layer E — Target-Rig Application (not implemented)

Motion IR, RigProfile, IK / retarget, editable keyframes. This is where
`Architecture_v2.md` sections 5–8 resume.

## 4. Frame sample contract

Schema `animcv_frame_pose_bank_v1`. One sample is independently addressable by a
stable identity:

```
sample_id = "<source>:<sequence>:<actor>#<frame_index:06d>"
```

Every sample preserves:

| Field | Meaning |
| --- | --- |
| `sample_id` | stable, addressable identity |
| `source` | dataset of origin (`3DPW`, `MPI-INF-3DHP`, `AMASS`, ...) |
| `sequence_id` | originating clip/actor, never discarded |
| `frame_index` | index within the source sequence |
| `timestamp`, `fps` | where the source provides them |
| `split` | `train` / `validation` / `test` |
| `image_size` | source pixel dimensions |
| `modality` | `has_rgb`, `has_2d`, `has_3d`, `has_camera` |
| `image_reference` | `{root_key, relative_path}` — never inlined pixels |
| `neighbors` | `{previous, next}` sample ids (Layer B reservation) |
| `strata` | analysis strata, GT-derived, evaluation/diagnostic use only |

Numeric arrays live beside the JSON index in one `.npz` companion, aligned by
sample order: `input_2d (N,17,3)`, `input_valid (N,17)`, `target_3d (N,17,3)`,
`target_valid (N,17)`. Both files are SHA-256 fingerprinted together.

Frame-first does **not** discard sequence identity. `sequence_id` and
`frame_index` are mandatory, and splits are isolated at sequence granularity.

## 5. Modality availability is explicit, never fabricated

| Source | has_2d | has_3d | has_rgb | has_camera |
| --- | --- | --- | --- | --- |
| 3DPW | yes | yes | **yes** (`imageFiles/`) | intrinsics available |
| MPI-INF-3DHP | yes | yes | no (only `annot.mat` is intaken) | calibration available |
| AMASS | yes (projected) | yes | no (mocap, no imagery) | synthetic |

Geometry-only sources stay usable for geometry-only work. RGB is never
synthesised for a source that has none. The controlled F0/F1/F2 comparison runs
on a **paired-modality subset** where every candidate sees exactly the same
frames — in this repository that is 3DPW, because it is the only intaken source
with real imagery.

## 6. Coordinate and skeleton contract (unchanged)

- Canonical 17 joints, `pose.pose_lifter.H36M_NAMES` order. Unchanged.
- Canonical camera frame: `+X = right`, `+Y = forward/depth`, `+Z = up`.
  Unchanged.
- Targets are pelvis-relative metres. Unchanged.
- Bilateral forward-depth quantity is the same `D = (y_R - y_L)/sqrt(2)` used by
  docs/18 and docs/21, read on `FORWARD_DEPTH_AXIS = 1`.

No new skeleton and no new axis semantics are introduced by Layer A.

## 7. Model interface

```
FramePoseEstimator(geometry, image_tokens?) -> (B, 17, 3)
```

- **Geometry path (always present).** Per-joint token from crop-normalized 2D
  position, confidence, validity and a learned joint-identity embedding. The
  visual path never replaces it.
- **Visual path (optional).** Frozen pretrained backbone -> spatial patch tokens
  -> linear projection + learned patch positional embedding.
- **Fusion.** Joint queries consume image tokens through a fixed, minimal
  geometry-aware block: cross-attention (joint queries <- image tokens),
  self-attention among joint queries, FFN. Depth and width are fixed once
  (`depth=2`, `dim=256`, `heads=8`, derived from the temporal lifter's
  `channels=256`); no depth/width sweep.
- **Head.** Shared per-joint MLP -> `(17, 3)`.

Explicitly excluded from the inference contract: natural-language generation,
text decoding, autoregressive sampling, any language decoder in the hot path.

## 8. Observation backends

Three candidates over the same paired frames, the same fusion contract, the same
pose head, the same loss, the same optimizer, the same evaluator:

| Candidate | Visual evidence | Purpose |
| --- | --- | --- |
| **F0** | none | what one frame's explicit 2D geometry alone can recover |
| **F1** | conventional lightweight pretrained vision encoder | does restoring RGB monocular evidence help |
| **F2** | one lightweight vision-language pretrained representation | does language-aligned pretraining add orientation/depth priors beyond ordinary vision pretraining |

Backbones are frozen first. Parameter-efficient adaptation is conditional on
frozen F2 showing real frame-level benefit; full fine-tuning is not performed.

## 9. Portability

Dataset-specific code terminates at `src/framepose/sources.py` adapters. The
core consumes only the Frame Pose Contract plus modality metadata. Onboarding a
future commercial paired dataset should require:

```
dataset adapter  +  modality mapping  +  canonical joint mapping  +  image preprocessing metadata
```

and no change to `bank.py`, `model.py`, `losses.py`, `train.py` or
`evaluate.py`. Image roots are referenced by key, not absolute path, so a bank
built on one machine is usable on another by remapping the key.

## 10. Measured status of the observation backends

The controlled F0/F1/F2 comparison was executed on the 21,817-frame 3DPW paired
bank (docs/23). Result, on validation and on test, on every metric and in 37 of
37 test sequences: **F0 (geometry only) wins.** Both RGB candidates fit the
training frames far better and generalize far worse, and a token-substitution
diagnosis shows the visual path is used heavily but carries ~4.6x less
transferable pose information on unseen scenes than on seen ones.

Consequences now binding on Layer A:

- The Frame Pose Core's current primary configuration is **F0 — explicit 2D
  geometry only.** The visual path stays implemented and tested but is not the
  default.
- The frozen-F2 precondition for parameter-efficient VLM adaptation
  (section 8) is **unmet**; that branch is stopped, not deferred.
- Restoring RGB evidence remains the right hypothesis about the *information*;
  it is the data regime and the visual-path regularisation, not the fusion
  interface, that this batch found wanting. Any retry needs a substantially
  larger or more scene-diverse paired corpus, and should re-run exactly this
  comparison.

## 11. What Layer A deliberately does not do

Animation stabilization, contact, root motion, IK, retargeting, Motion Graph
work, temporal smoothing and temporal losses are out of scope for the frame
core and are not implemented here.
