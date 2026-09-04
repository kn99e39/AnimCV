# Architecture v3 — Frame-First Pose Core

## Status and precedence

This document is **normative** for AnimCV's perception stage. Where it and
`Architecture_v2.md` disagree, **this document wins**, and `Architecture_v2.md`
is to be read as historical for those subjects.

`Architecture_v3_FramePose.md` supersedes `Architecture_v2.md` for:

| Subject | v3 ruling |
| --- | --- |
| Perception ownership | The Frame Pose Core (`src/framepose/`) is the primary perception contract. |
| 2D pose observation | MMPose is the **Geometry Observation Layer** and nothing else (section 5). |
| Frame-pose learning | `one frame -> one root-relative canonical 17-joint 3D pose` (Layer A). |
| Role of temporal lifting | The temporal lifter is the Legacy Temporal Pose Baseline, not the primary core (section 1.3). |
| Visual / VLM evidence fusion | Complementary visual evidence fused into joint queries; never a replacement for explicit geometry (sections 8-9). |

`Architecture_v2.md` **remains authoritative**, unchanged, for everything this
document does not replace:

```
video/image intake and frame extraction
rig parsing and RigProfile
bone mapping and mapping profiles
Motion Graph and MotionPoint contracts
keyframe importance and collapse
Blender isolation boundary and export
retargeting boundaries and downstream animation contracts
```

There is exactly one normative primary perception pipeline in this repository,
and it is the one described here.

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

What the experiments support — and what they do not — matters here, because the
whole migration rests on it.

**Supported by the evidence:** the current temporal 2D lifter, under the current
data regime, did not resolve the orientation and forward-depth generalization
problem, and repeated loss-side attempts to make it do so either failed to move
the tail or damaged pose geometry.

**Also true, independent of those experiments:** the 2D-joint-only observation
contract *discards* appearance evidence that is present in the RGB frame and
bears directly on the ambiguous quantity:

```
foreshortening, self-occlusion, limb overlap, body-facing cues, silhouette,
clothing/body contour, near/far limb ordering, shading and other monocular
depth cues
```

**Not claimed:** that a temporal 2D sequence can never carry additional
orientation or depth evidence. It plainly can — motion parallax, occlusion
ordering over time and limb-swing phase are real cues in a 2D sequence. The
measured claim is narrower: *this* temporal architecture on *this* data did not
extract enough of it. Temporal context therefore remains a valid future source
of additional evidence, as Layer B.

What changes is the **primary research contract**: frame-level pose correctness
comes first, and temporal context is evaluated later as an addition to a frame
that is already geometrically right.

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

Schema `animcv_frame_pose_bank_v2`. One sample is independently addressable by a
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
| `observation` | 2D provenance: `backend`, `observation_type`, `regime`, sensor `detail` (section 5.1) |
| `image_reference` | `{root_key, relative_path}` — never inlined pixels |
| `neighbors` | `{previous, next}` sample ids (Layer B reservation) |
| `strata` | analysis strata, GT-derived, evaluation/diagnostic use only |

Numeric arrays live beside the JSON index in one `.npz` companion, aligned by
sample order: `input_2d (N,17,3)`, `input_valid (N,17)`, `target_3d (N,17,3)`,
`target_valid (N,17)`. Both files are SHA-256 fingerprinted together.

`content_digest` covers sample identity, split, source and those arrays under a
fixed domain separator (`CONTENT_DIGEST_DOMAIN`). It deliberately excludes
metadata such as observation provenance and the index schema label: adding
metadata must not invalidate feature caches and experiment reports keyed to the
frames, while a genuinely different sensor — which moves `input_2d` — still
does.

Frame-first does **not** discard sequence identity. `sequence_id` and
`frame_index` are mandatory, and splits are isolated at sequence granularity.

## 5. MMPose is the Geometry Observation Layer

AnimCV's production perception flow is two sensors feeding one core:

```
RGB frame
   |
   +--> MMPose (pose/mmpose_adapter.py)  ->  2D joints + confidence + validity
   |         RTMDet person detector -> RTMPose top-down keypoints
   |         canonical_v1 landmark schema
   |
   +--> optional frozen visual encoder  ->  spatial visual evidence
   |
   v
Frame Pose Core (src/framepose/)
   |
   v
canonical root-relative 3D pose (17 x 3, +X right / +Y forward / +Z up)
```

**MMPose owns exactly one box: the 2D Geometry Observation Layer.** It is not
the 3D Pose Core, not the temporal solver, and not the RGB reasoning layer.

Two consequences are binding:

1. `pose/pose_lifter.py`'s `VideoPose3DLifter` — MMPose's own temporal
   2D->3D lifter, reached by the `lift-pose3d` CLI command — is **legacy and
   reference only**. It is preserved and still runs, but MMPose is not AnimCV's
   3D solver. The two 3D paths that remain architecturally live are the Frame
   Pose Core (primary) and the Legacy Temporal Pose Baseline (`lift-supervised-3d`).
2. MMPose stays behind its adapter. No `framepose` module imports `mmpose`,
   `mmdet` or `mmengine`, and a test enforces it, so the geometry-only runtime
   never pulls the OpenMMLab stack. The real-observation regime is therefore a
   two-stage flow across two environments: the `pose` extra
   (`Dockerfile.pose`) produces observations, the `frame-pose` extra
   (`Dockerfile.framepose`) consumes them.

### 5.1 Three evaluation regimes, always labelled

Every frame sample records where its 2D geometry came from
(`framepose/observations.py`), and every bank, training report and evaluation
report is labelled. The distinction that matters is **whether a learned 2D
detector contributes to the observation error** — not whether the keypoints
happened to ship with a dataset.

| Regime | 2D source | Detector error present? | Purpose |
| --- | --- | :-: | --- |
| `oracle_geometry` | annotated/projected ground truth, or deterministic synthetic projection from known 3D | no | isolate the 3D reconstruction architecture from 2D sensor error |
| `benchmark_detector_observation` | a fixed external detector's output distributed with a benchmark (3DPW's shipped OpenPose-format keypoints) | **yes** | compare 3D reconstruction architectures under one fixed, externally defined observation |
| `real_animcv_observation` | AnimCV's own MMPose + RTMDet sensor with its checkpoints and preprocessing | **yes** | measure actual AnimCV perception behaviour end to end |

A fourth value, `historical_unknown`, exists only for artifacts whose provenance
cannot be resolved; any interpretation that depends on observation quality must
refuse it (`assert_quality_interpretable`).

Backend, observation type and regime are separate fields, but they are not
independent: what produced an observation determines whether a detector is in
its error, so the mapping is enforced rather than inferred from "did a dataset
provide it".

```
backend                 observation_type              regime
dataset_ground_truth    projected_ground_truth_2d     oracle_geometry                 (MPI-INF-3DHP)
synthetic_projection    synthetic_virtual_camera_2d   oracle_geometry                 (AMASS)
dataset_detector        dataset_shipped_detector_2d   benchmark_detector_observation  (3DPW)
mmpose                  estimated_2d                  real_animcv_observation         (AnimCV)
```

**Detector output is never oracle geometry.** Constructing that combination
raises.

Results from different regimes are **not comparable without the label**, and
dataset-provided geometry is never described as "the MMPose pipeline". A frame
set that mixes regimes is refused unless a bank is built with the explicit
`allow_mixed_regime` opt-in, which labels it `mixed`; the experiment runner
still refuses to interpret one.

For an estimated observation the record carries model, checkpoint, detector,
preprocessing and version, and its `cache_key` changes when any of those change;
`observation_cache_key` additionally binds the exact input frame. A cached
MMPose observation is therefore invalidated by a change of model, weights,
config, preprocessing or input image.

**Migration.** A historical artifact is resolved deterministically from its
recorded backend, which is unambiguous — 3DPW samples once labelled
`oracle_geometry` become `benchmark_detector_observation`, while ground-truth
and synthetic sources stay `oracle_geometry`. Not every old `oracle_geometry`
label maps to the same new regime, and an artifact that cannot be resolved
becomes `historical_unknown` rather than being guessed at.

**Current measured status:** every result in this repository is
`benchmark_detector_observation` (3DPW's shipped keypoints). No MMPose
observations exist for the research bank, so the Real AnimCV Observation regime
is implemented as a contract and reported as *not yet measured*. No such data is
fabricated.

### 5.2 Two digests, two jobs

| Digest | Covers | Job |
| --- | --- | --- |
| `content_digest` | sample identity, split, source, and the numeric arrays, under a frozen domain separator | governs feature-cache validity — a cache is reusable exactly while these are unchanged |
| `provenance_fingerprint` | observation provenance, modality, image references | detects a provenance change, including a corrected regime label, **without** invalidating a cache whose frames, crops and images did not move |

A real sensor change moves both, because different keypoints change `input_2d`.
Correcting a human-readable regime label moves only the second. Neither digest
is overloaded with the other's responsibility.

## 6. Modality availability is explicit, never fabricated

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

## 7. Coordinate and skeleton contract (unchanged)

- Canonical 17 joints, `pose.pose_lifter.H36M_NAMES` order. Unchanged.
- Canonical camera frame: `+X = right`, `+Y = forward/depth`, `+Z = up`.
  Unchanged.
- Targets are pelvis-relative metres. Unchanged.
- Bilateral forward-depth quantity is the same `D = (y_R - y_L)/sqrt(2)` used by
  docs/18 and docs/21, read on `FORWARD_DEPTH_AXIS = 1`.

No new skeleton and no new axis semantics are introduced by Layer A.

## 8. Model interface

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

## 9. Observation backends

Three candidates over the same paired frames, the same fusion contract, the same
pose head, the same loss, the same optimizer, the same evaluator:

| Candidate | Visual evidence | Purpose |
| --- | --- | --- |
| **F0** | none | what one frame's explicit 2D geometry alone can recover |
| **F1** | frozen ImageNet-pretrained ViT-B/16 patch tokens | does a conventional visual representation help |
| **F2** | frozen **vision-language-pretrained visual tower** (SigLIP ViT-B/16 image tower) patch tokens | does vision-language pretraining add orientation/depth priors beyond ordinary vision pretraining |

Backbones are frozen first. Parameter-efficient adaptation is conditional on
frozen F2 showing real frame-level benefit; full fine-tuning is not performed.

### 9.1 What each comparison can and cannot establish

**F1 vs F2 is the clean, architecture-matched comparison.** Both are ViT-B/16 at
224x224 emitting a 14x14x768 patch grid with the same normalization, both frozen,
and the trainable model is parameter-identical (2,427,139 each). Only the
pretraining objective differs, so a difference between them is attributable to
pretraining.

**F0 vs F1/F2 is not a pure information-only comparison.** F0 has no
cross-attention sublayer and no image projection, so it is a different model
(1,652,227 trainable parameters) as well as a different observation. A result
comparing them is a statement about *the tested architectures*:

```
valid:    "the tested visual-fusion architectures generalize worse than the
           tested geometry-only architecture"

invalid:  "RGB evidence itself damages pose estimation"
```

A parameter-matched geometry-only control (same block count, cross-attention
into a learned constant) would be needed to make F0-vs-F1/F2 causal about
information alone. It has not been trained.

### 9.2 F2 scope

F2 loads the SigLIP **image tower only**. No text encoder, no multimodal
projector, no language decoder and no autoregressive generation are present in
the pose path, and the feature cache records
`text_encoder_loaded=false`, `language_decoder_loaded=false`,
`autoregressive_generation_used=false`.

F2 is therefore a test of a **vision-language-pretrained visual tower's
representation**. It is *not* a test of full VLM reasoning, of multimodal
decoder reasoning, or of a fine-tuned VLM. Its result is scoped to the
representation actually used.

## 10. Portability

Dataset-specific code terminates at `src/framepose/sources.py` adapters. The
core consumes only the Frame Pose Contract plus modality metadata. Onboarding a
future commercial paired dataset should require:

```
dataset adapter  +  modality mapping  +  canonical joint mapping  +  image preprocessing metadata
```

and no change to `bank.py`, `model.py`, `losses.py`, `train.py` or
`evaluate.py`. Image roots are referenced by key, not absolute path, so a bank
built on one machine is usable on another by remapping the key.

## 11. Measured status of the observation backends

The controlled F0/F1/F2 comparison was executed on the 21,817-frame 3DPW paired
bank (docs/23), in the **`benchmark_detector_observation`** regime — 3DPW's own
shipped detector keypoints. The valid reading is therefore:

```
valid:    "F0/F1/F2 compare 3D reconstruction architectures under the SAME fixed
           3DPW dataset-shipped detector observations"

invalid:  "F0/F1/F2 isolate the 3D core from 2D observation error"
```

Result, on validation and on test, on every metric and in 37 of
37 test sequences: **the tested geometry-only architecture (F0) wins.** Both
visual-fusion candidates fit the training frames far better and generalize far
worse. A token-substitution diagnosis shows the visual path is heavily depended
on, but that substituting an unrelated frame's tokens costs far less error on
unseen splits than on train — consistent with the fusion having fitted
scene-specific appearance rather than a transferable appearance-to-depth cue.

Consequences now binding on Layer A:

- The Frame Pose Core's current primary configuration is **F0 — explicit 2D
  geometry only.** The visual path stays implemented and tested but is not the
  default.
- The frozen-F2 precondition for parameter-efficient adaptation of the
  vision-language tower (section 9) is **unmet**; that branch is stopped, not
  deferred.
- The F0-vs-F1/F2 gap is a statement about the tested architectures, not about
  RGB evidence as such (section 9.1).
- Restoring RGB evidence remains the right hypothesis about the *information*;
  it is the data regime and the visual-path regularisation, not the fusion
  interface, that this batch found wanting. Any retry needs a substantially
  larger or more scene-diverse paired corpus, and should re-run exactly this
  comparison.

## 12. What Layer A deliberately does not do

Animation stabilization, contact, root motion, IK, retargeting, Motion Graph
work, temporal smoothing and temporal losses are out of scope for the frame
core and are not implemented here.


## 13. Canonical pose mathematics ownership

The canonical 17-joint geometry both learning cores need — bone/torso/hinge
indices, the vector and hinge objectives, similarity alignment, root-yaw and
bend-direction metrics — is owned by **`src/common/canonical_pose.py`**, a
neutral module that knows nothing about temporal windows, training configs,
FramePose, experiment identifiers or dataset sources.

```
        common.canonical_pose      (single owner of the mathematics)
             /              \
   Frame Pose Core     Legacy Temporal Pose Baseline
```

Not:

```
   Frame Pose Core  ->  Legacy Temporal Pose Baseline
```

`training/temporal_lifter.py` re-exports the moved names as **direct aliases**,
so every historical caller, script and test keeps working and no second formula
can drift from the one A9–A16 and F0–F2 were measured with.

This was an ownership move, never a redesign. Reductions, masks, epsilons and
dtype behaviour are unchanged, and `tests/test_canonical_pose_parity.py` pins
them **bitwise** against values captured before the move
(`tests/fixtures/canonical_pose_reference_v1.json`). Both sides are checked: the
Frame Pose Core's objective is bitwise identical to the historical
`_supervision_loss`, and its evaluator uses the historical metric mathematics.

## 14. FramePose execution policy

`torch.compile` over forward + loss (backward and the optimizer step stay eager)
is the accepted execution path, and it is verified compatible with the FramePose
graph on the training host.

- **Historical F0/F1/F2 remain eager runs.** Their reports and checkpoints record
  `execution_backend: "eager"`, and that provenance is not rewritten.
- **Future FramePose training uses the compiled path** unless an experiment
  explicitly requires eager for a controlled comparison against a historical
  eager result. Whichever is used is recorded in the training report, the
  checkpoint and `experiment_matrix.json`.

No compile mode is changed and no further benchmarking is performed here.

## 15. The perception flow, in one picture

```
RGB frame
    |
    +--> Geometry Observation Layer            (explicit 2D joints + confidence + validity)
    |        provider, always labelled by regime:
    |            oracle / projected GT      -> oracle_geometry
    |            benchmark detector         -> benchmark_detector_observation
    |            AnimCV MMPose + RTMDet     -> real_animcv_observation
    |
    +--> optional visual evidence encoder      (frozen ViT patch tokens; complementary)
             |
             v
       Frame Pose Core        (Layer A — src/framepose/)
             |
             v
   canonical frame 3D pose    (17 x 3, root-relative, +X right / +Y forward / +Z up)
             |
             v
   Layer B  optional temporal context   (not implemented)
   Layer C  temporal stabilization      (not implemented)
   Layer D  animation semantics         (not implemented)
   Layer E  target-rig application      (Architecture_v2.md resumes here)
```

Four statements this document exists to make unambiguous:

- **Observation provenance is not 3D reasoning.** Where the 2D came from is
  recorded per sample and never inferred from a result.
- **A benchmark detector is not an oracle.** Dataset-shipped detector keypoints
  carry detector error and are labelled as such.
- **MMPose is not the 3D solver in the primary path.** It is the Geometry
  Observation Layer; `VideoPose3DLifter` / `lift-pose3d` is legacy and reference
  only.
- **The Temporal Lifter is not the owner of canonical pose mathematics.** It is
  a consumer of `common.canonical_pose`, exactly as the Frame Pose Core is.
