# Worklog — Frame-First Architecture Migration (2026-09-02)

> Architecture migration batch, executed on branch `arch/single_frame_first`
> (branched from `On_Work` at `4b676fc`). The Legacy Temporal Pose Baseline
> (`src/training/temporal_lifter.py`, every A9–A16 checkpoint, fingerprint,
> report, loss implementation, evaluator contract and diagnostic script) is
> **untouched** — `git diff On_Work..HEAD` adds files and modifies only
> `CLAUDE.md` and `pyproject.toml`. Nothing historical was renamed, rewritten
> or re-measured.

## 1. Why the temporal lifter was demoted from primary status

The previous primary abstraction was

```
Temporal 2D Joint Observations -> Temporal Lifter -> center-frame Canonical 3D Pose
```

The A5–A16 program established, under fingerprinted and controlled conditions,
that the residual is not a tuning problem:

- `root_yaw_p95_degrees` never reached its gate under any loss variant (A6, A7,
  A10, A11, A12, A14, A16).
- The dominant failure is monocular forward-depth ambiguity — the sign of
  `D = (y_R - y_L)/sqrt(2)` on shoulders and hips (docs/18, docs/21).
- Orientation auxiliaries either did not move the tail (docs/12–13: gradient
  ratio 3,170x, tail already small at convergence) or damaged pose geometry
  (docs/10 A6/A7, docs/14 A12).
- Generalization was source-dependent (docs/17, docs/22 A16), and loss, readout
  and temporal effects were entangled enough to need a bespoke diagnostic per
  candidate.

A stream of 2D joint coordinates does not contain the evidence that decides
which shoulder is nearer the camera. Averaging more individually ambiguous
frames does not create it. The evidence exists in the RGB frame and was being
discarded before the model saw it.

The temporal lifter is preserved as the **Legacy Temporal Pose Baseline**: a
reproducible historical baseline, a future Layer B context provider, a future
Layer C refiner, and a diagnostic comparison. Its new role is written into
`Architecture_v3_FramePose.md` section 1.3.

## 2. New architecture and layer boundaries

`Architecture_v3_FramePose.md` (new, root, alongside the still-binding
`Architecture_v2.md`) defines five layers. Only Layer A is implemented here.

| Layer | Contract | Status |
| --- | --- | --- |
| A — Frame Pose Core | one frame -> one root-relative canonical 17-joint 3D pose | **implemented** |
| B — Optional Temporal Context | neighbouring-frame evidence improves `Pose_t` | interface reserved (`neighbors`, `sequence_id`, `fps` carried in the contract) |
| C — Temporal Stabilization | good frame poses -> coherent sequence | not implemented |
| D — Animation Semantics | root motion, contact, foot locking | not implemented |
| E — Target-Rig Application | Motion IR, RigProfile, IK/retarget | not implemented |

## 3. Module and data-flow boundaries

```
prepared lifter datasets (read-only)
        |
   framepose.sources        <- the ONLY dataset-specific code
        |
   framepose.bank  +  framepose.strata      -> fingerprinted FrameBank (.json + .npz)
        |
   framepose.crops           -> one deterministic person-centric crop per frame
        |                        (shared by every RGB candidate)
        +--> framepose.backbones (frozen) -> framepose.features (bank-keyed cache)
        |
   framepose.train  ->  framepose.model  ->  framepose.losses
        |
   framepose.evaluate  (per-frame)   /   framepose.screening (pre-training, no training)
```

Nothing below `sources.py` knows what a dataset is; the core consumes the Frame
Pose Contract plus modality metadata only.

## 4. Frame sample contract

Schema `animcv_frame_pose_bank_v1`. Stable identity is
`"<source>:<sequence>:<actor>#<frame_index:06d>"`, e.g.
`3dpw:courtyard_arguing_00:actor0#000000`.

Each sample carries `source`, `sequence_id`, `frame_index`, `timestamp`, `fps`,
`split`, `image_size`, the four modality flags, an `image_reference`
(`{root_key, relative_path}` — never inlined pixels, and remappable on another
machine), `neighbors` (`previous`/`next` sample ids, the Layer B reservation),
and its `strata`.

Numeric arrays live in an aligned `.npz` companion (`input_2d (N,17,3)`,
`input_valid (N,17)`, `target_3d (N,17,3)`, `target_valid (N,17)`); index and
arrays are fingerprinted together and additionally by a path-independent
`content_digest`.

Frame-first does not discard sequence identity: decimation is applied per
sequence, `neighbors` never crosses a sequence boundary, and splits are isolated
at sequence granularity (`assert_split_isolation`, checked before assembly so a
leaking intake is reported as leakage rather than as colliding ids).

## 5. Modality availability by current data source

Declared in `framepose.sources.SOURCE_SPECS`, never inferred and never
fabricated:

| Source | has_2d | has_3d | has_rgb | has_camera | Image path |
| --- | :-: | :-: | :-: | :-: | --- |
| 3DPW | yes | yes | **yes** | yes | `imageFiles/<sequence>/image_<index:05d>.jpg` |
| MPI-INF-3DHP | yes | yes | no | yes | none — only `annot.mat` + `camera.calibration` are intaken |
| AMASS | yes | yes | no | no | none — marker mocap, there is no photograph to restore |

Consequence for this batch: **the paired-modality subset is 3DPW only.** MPI and
AMASS remain fully usable for geometry-only work and are excluded from the
RGB/VLM comparison by construction (`--require-rgb`), rather than by being given
a fabricated image. This is the honest cost of the comparison and it bounds what
the verdict can claim (Section 17).

## 6. Frame research bank

`scripts/build_frame_bank.py`, one invocation, deterministic (no RNG — selection
is per-sequence stride only):

```
--source 3DPW:train=/data/3dpw/prepared/train.json        --train-stride 2
--source 3DPW:validation=/data/3dpw/prepared/validation.json --validation-stride 3
--source 3DPW:test=/data/3dpw/prepared/holdout.json       --test-stride 5
--image-root 3dpw_images=/data/datasets/3dpw/imageFiles --require-rgb
```

| Split | 3DPW official split | Frames | Actor-sequences |
| --- | --- | ---: | ---: |
| train | train | 11,334 | 34 |
| validation | validation | 3,407 | 16 |
| test | test | 7,076 | 37 |
| **total** | | **21,817** | **87** |

`content_digest = 75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536`.
Every retained sample has real imagery that was verified to exist on disk during
construction; `retained_frames == available_frames` for all three splits, so
decimation is the only filter that fired.

`scripts/verify_frame_bank_images.py` (new) re-checked 2,000 samples against the
actual JPEGs: 0 missing images, 0 declared-vs-actual `image_size` mismatches
(3DPW mixes 1080x1920 and 1920x1080 sequences, and a silent swap would have put
every crop and every geometry token in the wrong frame), 0 crops failing to
contain their observed joints.

### Strata

Angular boundaries are fixed and physical; every quantile boundary is fitted on
the **train split only** and applied unchanged to validation and test. Test
ground truth participates in no threshold, no parameter and no selection.

| Stratum | Buckets |
| --- | --- |
| `facing` | frontal (<=30 deg) / near_frontal (<=60) / profile (<=120) / back_facing |
| `yaw` | low_yaw (<=45 deg) / high_yaw |
| `visibility` | fully_visible / partially_visible |
| `confidence` | low_confidence (< train q25 = 0.763) / normal_confidence |
| `torso_scale` | small / medium / large projected torso (train q25 0.131, q75 0.177 of image height) |
| `forward_depth` | near_zero (< train q25 = 0.079 m) / medium / large (> q75 = 0.218 m) |
| `articulation` | rare (min bend < train q10 = 72.8 deg) / typical |

Test-split coverage is genuinely diverse: 2,215 frontal / 1,406 near-frontal /
1,750 profile / 1,534 back-facing; 2,836 partially visible; 3,142 low-confidence;
1,779 near-zero forward depth; 934 rare articulation.

## 7. Shared frame-level training objective

`framepose.losses.BASELINE_GEOMETRY_V1` is the A5/A9 established stable geometry
objective evaluated per frame: masked smooth-L1 coordinate loss plus canonical
bone (0.25), torso-axis (0.15) and hinge-bend (0.15) vector terms. Its
structural terms are *imported from* `training.temporal_lifter`, and
`tests/test_frame_pose_training.py` asserts numeric equality with
`_supervision_loss` under the matching `TrainingConfig` — so "the same objective
as A9" is checked, not claimed.

Deliberately absent, per the migration direction: yaw tail, SRD, sign
classification, temporal derivative, local-frame auxiliary, source weighting and
every historical failed orientation auxiliary. The first question is the
observation architecture, not the loss.

### Pre-training screening (no training)

`scripts/screen_frame_losses.py`, 10 fixed real 256-frame batches at
initialization. No acceptance threshold is encoded anywhere.

| Contract | raw loss | grad norm | grad/base | cosine vs base | stable |
| --- | ---: | ---: | ---: | ---: | :-: |
| `baseline_geometry_v1` | 0.2277 | 5.080 | 1.000 | 1.000 | yes |
| `coordinate_only_v1` | 0.2228 | 5.079 | 0.9997 | 0.99999 | yes |

Reading: at initialization the three structural terms are ~2% of the raw loss
and shift the gradient direction by ~1e-5 in cosine. They are a shape prior that
costs nothing and moves nothing early — which is exactly why keeping them fixed
across F0/F1/F2 is safe, and why a loss search would have been the wrong first
question.

Per-joint gradient ownership at initialization (top entries) shows each term
owning the geometry it is supposed to own: `coordinate` -> ankles/knees/wrists
(limb extremities, 0.10/0.094/0.075); `torso` -> shoulders 0.37 each and hips
0.13 each; `hinge` -> knees and elbows; `bone` spread across ankles, shoulders
and thorax. Per-frame loss and per-frame error rank-correlate at 0.970, and the
worst-error decile carries 12.5% of the batch loss — the objective is not
dominated by a tail.

## 8. The three controlled observation backends

All three run on the identical 21,817-frame paired bank, the identical crop, the
identical geometry tokens, the identical loss contract, optimizer, seed (1337),
schedule and evaluator. `tests/test_frame_pose_scripts.py` asserts that the
matrix's per-candidate configs differ in `backbone` and nothing else.

**Shared model (`framepose.model.FramePoseEstimator`).** 17 canonical joint
queries, each built from `(x, y)` in the crop-normalized frame, confidence,
validity and a learned joint-identity embedding. Fusion is 2 pre-norm blocks of
`self-attention over joint queries -> (cross-attention into image tokens) -> FFN`
at width 256 / 8 heads (width inherited from the Legacy Temporal Pose Baseline's
`channels=256`). Head is a shared per-joint MLP to `(17, 3)`. No depth or width
sweep was run.

**F0 — geometry only.** No image tokens; the cross-attention sublayer is absent.
This is the honest cost of the ablation and is reported rather than hidden: F0
has 1,652,227 trainable parameters against F1/F2's 2,427,139. F1-vs-F2 is
therefore the parameter-identical comparison, and F0 is the information
reference.

**F1 — conventional vision + geometry.** `vit_base_patch16_224.augreg_in21k_ft_in1k`
(`timm/…`, Apache-2.0, 85,798,656 parameters, 0 trainable), ImageNet-21k
supervised pretraining fine-tuned on ImageNet-1k. Vision-only supervision.

**F2 — vision-language representation + geometry.**
`vit_base_patch16_siglip_224.webli` (`timm/ViT-B-16-SigLIP`, Apache-2.0,
92,884,224 parameters, 0 trainable), SigLIP image-text sigmoid contrastive
pretraining on WebLI. This is the visual tower that lightweight open VLM stacks
(PaliGemma, SmolVLM, Idefics) are built on, taken without the language decoder —
exactly the "use its learned representation, keep no language decoder in the hot
path" contract.

The pair is deliberately architecture-matched: both are ViT-B/16 at 224x224,
both emit a 14x14 = 196 patch-token grid at width 768, both normalize with
mean/std 0.5. The manipulated variable between F1 and F2 is the pretraining
objective and nothing else. Only one VLM family was used; no model shopping.

**Weight provenance** (SHA-256 over the loaded `state_dict`, recorded in each
feature cache):

| Candidate | timm model | weights_sha256 | params | trainable |
| --- | --- | --- | ---: | ---: |
| F1 | `vit_base_patch16_224.augreg_in21k_ft_in1k` | `a848493a8022…af9f08` | 85,798,656 | 0 |
| F2 | `vit_base_patch16_siglip_224.webli` | `78949bce2fb6…4e6cc8` | 92,884,224 | 0 |

Both caches record `text_encoder_loaded=false`, `language_decoder_loaded=false`,
`autoregressive_generation_used=false`, and
`tests/test_frame_pose_backbone_weights.py` asserts on the training host that
neither loaded module exposes a `text_model`/`text_encoder`/`lm_head` attribute.

## 9. Image preprocessing control

One crop contract (`framepose.crops.CROP_CONTRACT`, fixed before any candidate
was trained, never tuned on a holdout): square box on the valid-2D-joint bounding
box, side `max(w, h) * 1.5` (margin 0.25 per side) with a 32 px floor, never
clamped inward, constant-black padding outside the source, bilinear resample to
224x224, geometry mapped `pixel -> (pixel - origin)/side -> 2u - 1`. F1 and F2
receive byte-identical crops; the only per-backbone difference is each tower's
own declared normalization, which here happens to be identical (0.5/0.5).

Because the backbones are frozen, features are a pure function of
`(frame, crop, weights)`. `scripts/cache_frame_features.py` materialises them
once per backbone as `(21817, 196, 768)` float16 (6.57 GB each), keyed to the
bank's `content_digest` and sample-order digest; `load_feature_cache` refuses a
cache built for a different bank. This makes F1/F2 training cost the same as F0
and makes replay exact.
