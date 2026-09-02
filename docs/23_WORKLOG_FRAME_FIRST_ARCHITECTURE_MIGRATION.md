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

## 10. Controlled run and training throughput

`scripts/run_frame_pose_experiments.py`, one invocation, candidates `F0,F1,F2`,
200 epochs, batch 256, AdamW lr 3e-4 -> 1e-5 cosine, weight decay 1e-4, seed
1337, AMP, eager backend, `--evaluate-every 10`, selection on validation MPJPE.
`experiment_matrix.json` records that only `backbone` differs between the three
candidate configs.

| Candidate | trainable params | frames/s | train wall time | peak VRAM | selected epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| F0 | 1,652,227 | 6,903 | 328 s | 172 MB | 129 |
| F1 | 2,427,139 | 1,039 | 2,181 s | 793 MB | 139 |
| F2 | 2,427,139 | 1,049 | 2,162 s | 793 MB | 169 |

Batch size was fixed once from memory feasibility and held across candidates; no
batch-size sweep, and GPU utilisation was not a target.

## 11. Frame-level quantitative comparison

Mean over frames, 3DPW official splits, bank
`75519e6394a764e3…0ed536`.

| | val MPJPE | val PA | **test MPJPE** | test PA | test MPJPE p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **F0** geometry only | **72.27** | **52.22** | **80.34** | **57.85** | **139.54** |
| F1 vision + geometry | 73.66 | 54.26 | 108.25 | 72.23 | 172.98 |
| F2 vision-language + geometry | 74.54 | 54.58 | 102.85 | 67.95 | 162.58 |

### Orientation and forward-depth (test)

| | root yaw MAE | yaw P95 | hinge dir MAE | shoulder \|D\| residual | shoulder sign disagreement (stable) | hip sign disagreement (stable) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **F0** | **10.81** | **27.66** | **25.34** | **49.39 mm** | **0.112 (0.104)** | **0.148 (0.111)** |
| F1 | 21.50 | 66.64 | 32.00 | 94.92 mm | 0.254 (0.248) | 0.278 (0.252) |
| F2 | 19.29 | 48.23 | 28.82 | 92.80 mm | 0.232 (0.226) | 0.261 (0.225) |

The geometry-only frame core is better on every orientation and forward-depth
metric — the exact quantities the RGB path was introduced to fix.

### Per-source and per-sequence

Only one source can carry this comparison (Section 5), so per-source is 3DPW
throughout and no cross-source claim is made. Per **sequence**, the result is not
an averaging artifact: **F1 and F2 are worse than F0 on 37 of 37 test
sequences.** Per-sequence test MPJPE: F0 min 58.8 / median 79.0 / max 119.3;
F1 77.4 / 103.9 / 143.9; F2 80.2 / 106.5 / 145.9.

### Per-stratum (test, mean delta vs F0; positive = worse)

| Stratum bucket | F1 - F0 | F2 - F0 | F2 - F1 |
| --- | ---: | ---: | ---: |
| frontal | +18.9 | +18.3 | -0.6 |
| near_frontal | +33.2 | +28.6 | -4.6 |
| profile | +34.6 | +26.1 | -8.4 |
| back_facing | +28.7 | +18.4 | -10.3 |
| near_zero_forward_depth | +14.8 | +14.6 | -0.2 |
| large_forward_depth | +36.6 | +27.3 | -9.3 |
| fully_visible | +29.2 | +25.6 | -3.6 |
| partially_visible | +26.0 | +17.9 | -8.1 |
| low_confidence | +28.9 | +18.7 | -10.2 |
| rare_articulation | +31.0 | +21.5 | -9.5 |

There is **no stratum in which either RGB candidate beats geometry-only**.
F1 regresses on 5,998 of 7,076 test frames, F2 on 5,682.

## 12. Frame-by-frame review

`scripts/export_frame_pose_review.py`, fixed selection (alphabetically first four
test sequences, chosen before any candidate was compared), 850 frames with GT and
all three predictions individually addressable
(`/output/framepose/review_test.json`).

| Review sequence | frames | MPJPE F0 / F1 / F2 | yaw F0 / F1 / F2 | shoulder sign-disagreement F0 / F1 / F2 |
| --- | ---: | --- | --- | --- |
| `downtown_arguing_00:actor0` | 180 | 68.2 / 137.9 / 112.7 | 8.3 / 44.1 / 13.7 | .050 / .383 / .039 |
| `downtown_arguing_00:actor1` | 179 | 61.3 / 82.5 / 88.8 | 8.2 / 12.1 / 23.2 | .084 / .179 / .106 |
| `downtown_bar_00:actor0` | 250 | 94.4 / 135.5 / 121.3 | 11.6 / 17.7 / 18.4 | .100 / .352 / .208 |
| `downtown_bar_00:actor1` | 241 | 73.5 / 103.9 / 84.3 | 13.7 / 18.7 / 16.9 | .025 / .129 / .087 |

The per-frame view is not uniformly negative and that is worth recording
honestly: over `downtown_arguing_00:actor0` frames 0–35, F1 and F2 both hold a
*smaller* shoulder forward-depth residual than F0 (e.g. frame 10: F0 -149.8 mm,
F1 -83.4, F2 -89.8) and a smaller yaw error, while F0 wins on total MPJPE. The
visual candidates are not failing to see orientation on every frame; they are
failing to keep whatever they see when the scene changes (Section 13). No
sequence playback was used as evidence.

## 13. Why the RGB candidates lose — visual-feature usage diagnosis

Because both RGB candidates failed to beat the control, the migration direction
requires diagnosing feature usage before adding complexity (Case C). No training
was involved: `scripts/diagnose_visual_feature_usage.py` replays each trained
checkpoint with its image tokens substituted.

MPJPE mm (delta vs that split's `real`):

| Candidate | split | real | zero | shuffled | neighbour |
| --- | --- | ---: | ---: | ---: | ---: |
| F1 | train | 15.28 | 242.13 (+226.9) | 108.82 (**+93.5**) | 80.58 (+65.3) |
| F1 | validation | 72.91 | 228.94 (+156.0) | 111.58 (**+38.7**) | 96.24 (+23.3) |
| F1 | test | 108.10 | 255.50 (+147.4) | 128.14 (**+20.0**) | 119.71 (+11.6) |
| F2 | train | 13.95 | 227.77 (+213.8) | 114.34 (**+100.4**) | 85.24 (+71.3) |
| F2 | validation | 73.78 | 226.63 (+152.9) | 113.46 (**+39.7**) | 97.89 (+24.1) |
| F2 | test | 102.84 | 247.41 (+144.6) | 125.08 (**+22.2**) | 115.49 (+12.6) |

Three facts follow:

1. **The visual path is used, heavily.** Silencing it (`zero`) costs 145–227 mm
   on every split. This is not a case of the fusion ignoring the image.
2. **What it extracts does not transfer.** Replacing the tokens with an
   unrelated frame's costs +93.5 mm on train but only +20.0 mm on test for F1
   (+100.4 / +22.2 for F2). The pose information the model recovers from an
   image shrinks ~4.6x from seen to unseen scenes.
3. **It is not purely scene identity.** `neighbour` (same scene and subject,
   different pose) still costs +65 mm on train, so the model does read
   frame-specific content — but the transferable remainder on test is +11.6 mm.

The generalization gap makes the same point directly:

| Candidate | train MPJPE | test MPJPE | gap | ratio |
| --- | ---: | ---: | ---: | ---: |
| F0 | 25.60 | 80.08 | +54.5 | 3.13x |
| F1 | 15.28 | 108.10 | +92.8 | 7.08x |
| F2 | 13.95 | 102.84 | +88.9 | 7.37x |

Adding the frozen visual path cut training error by 40–46% and roughly doubled
the generalization gap. With 11,334 training frames drawn from 34 actor-sequences
of one dataset, 196 frozen ViT patch tokens are enough capacity to fit those
scenes' appearance-to-pose mapping, and that mapping does not survive a change of
subject, clothing and background.

This is a **data-regime and fusion-regularisation** finding, not proof that RGB
lacks the evidence. The distinction matters for what should be tried next and is
stated as such rather than collapsed into "RGB does not help".

### Trained-state loss screening

`screening_trained.json` re-ran the fixed 10-batch screen at the F0-trained
state. The structural terms' gradient ratio moves from 0.9997 at initialization
to 0.954 with cosine 0.9974 — the shape prior becomes measurably more
distinguishable from the coordinate term once the model has converged, while
staying aligned in direction. The worst-error decile's loss share rises from
0.125 to 0.284: at convergence the objective is increasingly carried by hard
frames. Both are recorded as measurements; no acceptance rule is encoded, and
nothing here was used to pick a candidate.

## 14. Observation-architecture verdict

**Case C**, with the mechanism of Case E.

- F1 ~ F0 and F2 ~ F0 was the Case C hypothesis; the measured outcome is
  stronger and must be stated as measured: **both RGB candidates are worse than
  geometry-only**, on validation (the split selection used) and on test, on every
  metric, in every stratum, and in 37 of 37 test sequences.
- The Case E signature is present: the visual candidates fit training frames far
  better (15.3 / 14.0 mm vs F0's 25.6) and generalize far worse.
- Case D (helps some sources, damages others) cannot be evaluated — only one
  source in this repository carries paired RGB, and that limitation is reported
  rather than papered over.

Per Case C, **no VLM fine-tuning was started.**

**Does lightweight VLM pretraining help beyond ordinary visual pretraining?**
No consistent advantage. F2 beats F1 by 5.40 mm mean on test, and that advantage
does concentrate in exactly the predicted places — back-facing -10.3, profile
-8.4, large forward-depth -9.3, low-confidence -10.2, rare articulation -9.5,
partially visible -8.1 — with test yaw P95 48.23 vs 66.64. But F2 is **worse**
than F1 on validation (+0.87 mm) and on per-sequence median test MPJPE (106.5 vs
103.9). A directional hint that survives in one summary and reverses in two is
not evidence; the honest reading is **F2 ~ F1**, both dominated by F0.

**Answer to the batch's central question:** on identical paired-frame evidence,
in this data regime and with this fusion, a lightweight vision-language
representation provides **no measurable 3D orientation or depth benefit** beyond
geometry-only, and no reliable benefit beyond conventional visual pretraining.

### Section 13 gate: not reached

Frozen F2 produced no real frame-level evidence that the vision-language
representation is useful, so the precondition for parameter-efficient adaptation
is unmet. **No LoRA/adapter candidate was trained, no rank or target-module set
was swept, and the VLM fine-tuning branch is stopped.** Negative evidence is the
result.

### What the frame-first core itself established

Separately from the RGB question, the answer to *"can AnimCV treat frame-level
pose correctness as its primary perception contract?"* is **yes, and the
frame-only baseline is already competitive with the Legacy Temporal Pose
Baseline.** F0 reaches 57.85 mm PA-MPJPE and 27.66 deg root-yaw P95 on 3DPW test
from a single frame, against A9's 75.31 mm and 34.77 deg from an 81-frame window
— and A9's yaw P95 gate failure is what motivated this migration. These are not
strictly comparable runs (different frame subsets, different training mixture:
A9 trained on MPI + 3DPW + AMASS, F0 on 3DPW only), so this is a strong
indication rather than a promotion, and it is reported as such. It does say the
frame-first abstraction is viable on its own terms.

## 15. Portability assessment

Dataset knowledge is confined to `framepose/sources.py`: a `SourceSpec` plus an
image-path callable. Nothing in `bank.py`, `crops.py`, `model.py`, `losses.py`,
`train.py`, `evaluate.py` or `screening.py` mentions 3DPW, MPI or AMASS, and the
tests assert that a geometry-only source is excluded from a paired bank by
modality rather than by name. Imagery is addressed by `(root_key, relative_path)`,
so a bank built here is consumable elsewhere by remapping one key.

Onboarding a future paired commercial dataset requires: one `SourceSpec`
(modality flags + image reference), the existing canonical joint mapping in its
adapter, and its preprocessing metadata. No core change.

Not yet portable: the frame contract assumes the prepared
`animcv_supervised_3d_lifter_dataset_v2` intake shape, so a new dataset still
needs an adapter down to that artifact first. That is the same boundary the
Legacy Temporal Pose Baseline already has.

## 16. Updated role of the historical temporal lifter

`src/training/temporal_lifter.py` is unmodified and stays the **Legacy Temporal
Pose Baseline**: reproducible historical baseline, future Layer B context
provider, future Layer C refiner, and diagnostic comparison. Its structural loss
terms are now *also* the frame core's shared objective, imported rather than
copied, with an equality test.

Layer B's comparison (`F0` vs `F0 + short temporal context` vs `F0 + long
temporal context`) is **not** implemented here. The contract reserves what it
needs — `sequence_id`, `frame_index`, `fps`, `timestamp`, and `neighbors` that
never cross a sequence boundary — so that comparison needs no re-ingest.

## 17. Limits of this batch

- **One source.** Only 3DPW carries paired RGB in this repository, so every
  conclusion about visual evidence is a 3DPW conclusion. Case D is untestable.
- **One data regime.** 11,334 training frames from 34 actor-sequences. The
  diagnosis in Section 13 says the failure is regime-shaped; a substantially
  larger or more scene-diverse paired corpus could change the answer, and this
  batch cannot say by how much.
- **One fusion design.** Fixed width 256 / depth 2 / 196 tokens, no sweep, by
  instruction. A different visual-regularisation regime (token dropout, stronger
  augmentation, spatially constrained joint-to-patch attention) was not tried.
- **One VLM family.** SigLIP's ViT-B/16 tower only, by instruction.
- **No temporal anything.** No smoothing, no temporal loss, no root motion, no
  contact, no IK, no retarget, no Motion Graph work.

## 18. Artifacts, tests and synchronization

Server outputs, all under `LabServer63:~/animcv-output/framepose/`:

```
bank_3dpw_paired_v1.json / .npz / _report.json     the frame bank + fingerprint
bank_image_verification.json                       2,000-sample image contract check
features_v1/{vit_in21k,siglip}/                    frozen tokens + weight provenance
experiments_v1/{F0,F1,F2}/                         checkpoint, training + evaluation reports, predictions
experiments_v1/experiment_matrix.json              the controlled matrix and comparisons
experiments_v1/compare_{split}_{X}_vs_{Y}.json     per-frame deltas
screening_initial.json / screening_trained.json    pre-training loss screening
visual_usage_F1.json / visual_usage_F2.json        token-substitution diagnosis
review_test.json                                   850-frame frame-by-frame review
f0_split_mpjpe.json                                F0 train/validation/test MPJPE
```

Tests: 516 passed / 1 skipped in the full macOS suite (the skip is the
timm-gated backbone test); 54 passed in the `animcv-framepose:cuda118` container
on the training host, including the real frozen-backbone provenance and
no-text-generation assertions.

Branch `arch/single_frame_first`, branched from `On_Work` at `4b676fc`, in sync
across the macOS checkout, `origin`, and `LabServer63:/home/nd/AnimCV`.
