# Worklog — Architecture Reconciliation + Third-Party Dependency Audit (2026-09-03)

> Audit batch on `arch/single_frame_first`. No pose candidate was trained; no
> A-series or F-series model was retrained. The only GPU work was a rebuild of
> the frame bank and one 2-epoch execution-backend compatibility smoke test.
>
> Everything measured in docs/23 remains valid and is unchanged; several of its
> *interpretations* are corrected here and in the source documents.

## 1. Branch verification

| | |
| --- | --- |
| Branch | `arch/single_frame_first` |
| HEAD at audit start | `7f1c23549abf51fa489f2ca3b90b9b3673fd134f` |
| Merge-base with `On_Work` | `4b676fcf72abf8966846e270048caff3d68d5706` |
| `On_Work` vs `origin/On_Work` | identical; `On_Work` has not moved since the branch |
| Working tree | clean except untracked `.vscode/` (not touched) |

`git diff --name-status On_Work...HEAD` at audit start: **32 added files, 2
modified**. Filtering for anything but an addition
(`--diff-filter=MDRT`) returns exactly two paths:

```
M CLAUDE.md        additive: the LabServer63 section
M pyproject.toml   additive: the `frame-pose` optional-dependency group
```

### Historical preservation — verified, not assumed

- No A9–A16 implementation, checkpoint metadata, report, fingerprint, evaluator
  or diagnostic is modified. `src/training/temporal_lifter.py`, every
  `scripts/*diagnose*`/`*attribute*`/`*probe*`/`*repair*`, every `docs/09`–
  `docs/22`, and `runs/mpi_s1s6_cam0/checkpoint.pth` are byte-identical to
  `On_Work`.
- Cross-check: of every tracked file mentioning `A9`…`A16`, the only ones that
  differ from `On_Work` are the three files this migration *added*
  (`Architecture_v3_FramePose.md`, `docs/23`, `src/framepose/losses.py`).
- Server-side historical outputs were not written to: the migration wrote only
  under `~/animcv-output/framepose/`.

**Verdict: the Worklog's preservation claim is confirmed by the repository.**

## 2. Worklog claims verified vs contradicted

| docs/23 claim | Status |
| --- | --- |
| Historical files untouched | **Verified as of this audit** (Section 1). Superseded by docs/25: `temporal_lifter.py` was later mechanically changed to consume `common.canonical_pose`, with historical behaviour and mathematics preserved. |
| Bank 21,817 frames / 34+16+37 sequences / digest `75519e63…` | **Verified**; reproduced exactly on rebuild |
| Frame-core loss equals the A5/A9 objective | **Verified**; enforced by an equality test against `_supervision_loss` |
| F0 1,652,227 vs F1/F2 2,427,139 trainable parameters | **Verified** in code and in the run reports |
| F1/F2 parameter-identical | **Verified**; enforced by a test |
| Frozen towers, 0 trainable, no text encoder / language decoder | **Verified** in the cache provenance and by a live test on the training host |
| "Caching makes F1/F2 training cost the same as F0" | **Contradicted** — measured 6,903 vs ~1,040 frames/s. Corrected (Section 11) |
| "Pose information shrinks ~4.6x" | **Overstated** — the measurement is content-dependence, not information. Reworded (Section 10) |
| "A stream of 2D joint coordinates does not contain the evidence" | **Stronger than the evidence.** Corrected (Section 8) |
| "Both RGB candidates are worse than geometry-only" | **True as measured but wrongly framed** as an information result. Corrected (Section 9) |
| "v2 and v3 both binding" | **Ambiguous.** Resolved (Section 7) |
| 2D observation provenance | **Absent.** Added (Section 5) |

## 3. Actual FramePose module graph

Traced from imports, not from filenames:

```
prepared lifter datasets (read-only JSON, animcv_supervised_3d_lifter_dataset_v2)
        |
        |  sources.load_prepared_dataset / frames_from_prepared_dataset
        |  observations.resolve_dataset_observation   <- 2D provenance
        v
   bank.build_bank ── strata.frame_quantities/fit_thresholds/assign_strata
        |            └─ contract.assert_split_isolation
        v
   contract.FrameBank  (bank.json + bank.npz, content_digest, fingerprint)
        |
        +--> crops.crop_box / geometry_in_crop ──> train.geometry_tensor  (17x4)
        |
        +--> crops.render_crop ──> backbones.FrozenVisualBackbone (timm, lazy)
        |                              └─> features.build_feature_cache  (N,196,768 fp16)
        v
   train.train_candidate
        ├─ model.build_model      joint queries -> [self-attn -> cross-attn -> FFN] x2 -> head
        ├─ losses.compute_loss    coordinate + bone/torso/hinge (imported from temporal_lifter)
        └─ selection on validation only
        v
   evaluate.evaluate_predictions   per-frame metrics, per-source/sequence/stratum
   screening.screen_contracts      pre-training gradient measurement (no training)
```

Two couplings are worth naming explicitly because they are load-bearing:

1. `framepose/losses.py` and `framepose/evaluate.py` **import private helpers
   from `training.temporal_lifter`** (`_vector_loss`, `_hinge_loss`,
   `BONE_INDICES`, `TORSO_INDICES`, `_similarity_align`,
   `_root_yaw_error_degrees`, `_bend_direction`). This is deliberate — it is
   what makes "the same objective and the same metric definitions as A9" a
   checkable statement — but it means the Legacy Temporal Pose Baseline is now
   a **shared production dependency of the new core**, not a dead module.
   Changing those helpers changes the frame core. The loss side is protected by
   an equality test; the evaluator side is protected only by the metric tests.
2. `framepose/contract.py` imports `H36M_NAMES` from `pose.pose_lifter`. That
   module lazily imports MMPose inside `VideoPose3DLifter` only, so the frame
   core does not pull the OpenMMLab stack — now enforced by a test
   (`tests/test_frame_pose_dependency_isolation.py`).

## 4. MMPose architectural ownership

Before this audit the branch encoded MMPose's role only implicitly. It now
states it (`Architecture_v3_FramePose.md` section 5):

**MMPose is the 2D Geometry Observation Layer, and nothing else.** It is not the
3D Pose Core, not the temporal solver, and not the RGB reasoning layer.

Evidence from the repository:

- Production 2D path: `pose/mmpose_adapter.py` (`PoseEstimator`, RTMPose-tiny via
  `pose/default_model.py`) with an RTMDet-tiny person detector
  (`pose/default_detector.py`), emitting `canonical_v1` landmarks with
  confidence and a visibility threshold. Reached by `motion-tool estimate-pose`.
- **Ambiguity found and resolved:** `pose/pose_lifter.py`'s `VideoPose3DLifter`
  uses MMPose's *own* temporal 2D->3D lifter (`inference_pose_lifter_model`),
  exposed as `motion-tool lift-pose3d`. That is MMPose acting as a 3D solver,
  which contradicts the agreed ownership. It is now documented as **legacy and
  reference only**; it is preserved and still runs, but the architecturally live
  3D paths are the Frame Pose Core (primary) and the Legacy Temporal Pose
  Baseline (`lift-supervised-3d`).
- Import boundary: no `framepose` module imports `mmpose`/`mmdet`/`mmengine`,
  and a test enforces it. The real-observation regime is therefore a two-stage
  flow across two environments — `Dockerfile.pose` (the `pose` extra) produces
  observations, `Dockerfile.framepose` (the `frame-pose` extra) consumes them.

## 5. 2D observation provenance contract

Added: `src/framepose/observations.py`, and `FrameSample.observation` in bank
schema `animcv_frame_pose_bank_v2`.

```
ObservationProvenance(backend, observation_type, regime, detail)
    .cache_key()                       digest over sensor identity
observation_cache_key(prov, image)     sensor identity + the exact input frame
```

Registered backends, resolved from the `input_kind` the prepared artifacts
**already declared** rather than guessed per source:

| `input_kind` | backend | observation_type | regime |
| --- | --- | --- | --- |
| `dataset_ground_truth_2d` (MPI-INF-3DHP) | `dataset_ground_truth` | `projected_ground_truth_2d` | oracle |
| `official_3dpw_2d_detection` (3DPW) | `dataset_detector` | `dataset_shipped_detector_2d` | oracle |
| `synthetic_virtual_camera_gt_2d` (AMASS) | `synthetic_projection` | `synthetic_virtual_camera_2d` | oracle |
| — (AnimCV's own sensor) | `mmpose` | `estimated_2d` | real |

Enforced invariants, each with a test:

- A dataset backend may not claim `real_observation`, and `mmpose` may not claim
  `oracle_geometry`.
- An unregistered `input_kind` raises rather than silently defaulting.
- `mmpose_observation(...).cache_key()` changes when **any** of pose config,
  pose checkpoint, pose weights digest, detector config, detector checkpoint,
  visibility threshold or input size changes; `observation_cache_key` also
  changes with the input image. That is the cache-invalidation rule the
  architecture requires.
- A mixed-regime frame set is **refused**, not pooled.

Note on 3DPW: its shipped 2D keypoints are *detector output* (OpenPose-format,
18 joints) released with the dataset — not projected ground truth, and not
MMPose. The contract records that distinction rather than flattening it into
"GT 2D".

## 6. Oracle Geometry vs Real Observation

| Regime | 2D source | Purpose | Status |
| --- | --- | --- | --- |
| `oracle_geometry` | dataset-provided 2D | isolate the 3D reconstruction architecture from 2D sensor error | **all existing results** |
| `real_observation` | AnimCV's MMPose sensor | measure actual AnimCV perception end to end | **contract implemented, not yet measured** |

Every bank, training report and evaluation report now carries the label, and
docs/23 opens with it. No MMPose observations exist for the research bank and
none were fabricated.

The frame bank was rebuilt with provenance as
`bank_3dpw_paired_v2.json`, leaving the v1 artifact untouched. It reproduces
`content_digest = 75519e6394a764e3…0ed536` **exactly**, and both existing
feature caches load against it — so the recorded F0/F1/F2 results remain
attached to the same frames.

### A digest bug found and fixed while doing this

Bumping the bank schema to v2 moved `content_digest`, because the digest hashed
`BANK_SCHEMA` as its domain separator. That silently invalidated every feature
cache and experiment report keyed to it — the opposite of what the digest is
documented to mean. Fixed by freezing `CONTENT_DIGEST_DOMAIN` at the historical
string and pinning the fixture digest in a test, so no future metadata change
can drift it again.

## 7. Architecture v2 / v3 precedence

Previously `Architecture_v3_FramePose.md` said it superseded v2 "only for the
learning core", while docs/23 treated both as binding — two normative primary
perception pipelines. Resolved:

**v3 wins, and v2 is historical, for:** perception ownership; 2D pose
observation (MMPose's role); frame-pose learning; the role of temporal lifting;
visual/VLM evidence fusion.

**v2 remains authoritative, unchanged, for:** video/image intake; rig parsing
and RigProfile; bone mapping and profiles; Motion Graph and MotionPoint;
keyframe importance and collapse; the Blender isolation boundary and export;
retargeting boundaries and downstream animation contracts.

`CLAUDE.md` now points at that precedence so the third document cannot
reintroduce the ambiguity.

## 8. Temporal-lifter rationale — corrected

Removed from both `Architecture_v3_FramePose.md` and docs/23:

> "A stream of 2D joint coordinates does not contain the evidence needed to
> disambiguate which shoulder is nearer the camera. Adding more temporal context
> does not create this evidence."

That is stronger than the experiments establish. Motion parallax, occlusion
ordering over time and limb-swing phase are real orientation/depth cues in a 2D
sequence. What was actually shown is narrower and is now what both documents
say:

- the current temporal 2D lifter, in the current data regime, did not resolve
  the orientation/depth generalization problem;
- the 2D-joint-only observation contract *discards* appearance cues that are
  present in the RGB frame and bear on that ambiguity;
- frame pose correctness is now the primary research contract;
- temporal context remains a valid future evidence source as **Layer B**.

The Legacy Temporal Pose Baseline is preserved unchanged.

## 9. F0/F1/F2 causal-control status

Verified against the actual model graph and run reports:

| | trainable params | cross-attention | image projection |
| --- | ---: | :-: | :-: |
| F0 | 1,652,227 | absent | absent |
| F1 | 2,427,139 | present | present |
| F2 | 2,427,139 | present | present |

**F0 vs F1/F2 varies architecture *and* observation.** It is not an
information-only control. The measured result is unchanged; its interpretation
is now scoped:

```
valid:    "the tested visual-fusion architectures generalize worse than the
           tested geometry-only architecture, on this bank and this regime"
invalid:  "RGB evidence itself damages pose estimation"
```

A parameter-matched geometry-only control (same block count, cross-attention
reading a learned constant) would be required for the stronger claim. It was
**not** trained, and this batch does not train it.

**F1 vs F2 is verified clean**: same ViT-B/16 at 224x224, same 14x14x768 token
grid, same 0.5/0.5 normalization, both frozen with 0 trainable parameters, and a
parameter-identical trainable model. A difference between them is attributable
to pretraining objective.

## 10. F2 terminology and visual-usage wording

**F2 is a vision-language-pretrained visual tower.** The loaded module is
`vit_base_patch16_siglip_224.webli` — the SigLIP *image tower* only. No text
encoder, no multimodal projector, no language decoder, no autoregressive
generation; the cache provenance records all three flags as false and a live
test on the training host asserts the module exposes no
`text_model`/`text_encoder`/`lm_head`. Both documents now say the result is
scoped to that representation, and is **not** a test of full VLM reasoning,
multimodal decoder reasoning, or a fine-tuned VLM.

**Visual-usage diagnostic.** All measured numbers are preserved. The wording now
separates two different things:

- `zero` substitution (+145–227 mm on every split) is an **out-of-distribution
  intervention** — an all-zero token block is far outside anything the fusion
  saw, so a large error is partly expected from distribution shift alone. It
  rules out the fusion ignoring the image; it does not by itself demonstrate
  useful use of it.
- `shuffled` and `neighbour` keep the feature distribution intact and change only
  *which frame* the content describes. These are the load-bearing evidence, and
  they are described as **content-dependence**, not as an information measure.
  The "~4.6x" is now stated as a ratio of error penalties, with an explicit note
  that it does not license an information-theoretic claim.

## 11. FramePose execution backend

The abstraction was already present — `CandidateConfig.compile_training_graph`,
`torch.compile` over forward + loss with backward and the optimizer step eager
(the same shape docs/20 accepted), and `execution_backend` recorded in every
training report and checkpoint. The F0/F1/F2 runs were executed **eager**, which
is the opt-in default, and that provenance is recorded on those results.

Focused compatibility smoke test on the training host (torch 2.1.2+cu118, F0, 2
epochs, real bank — `execution_backend_smoke.json`):

| | final train loss | validation MPJPE | backend recorded |
| --- | ---: | ---: | --- |
| eager | 0.022212898 | 168.2728 | `eager` |
| compiled | 0.022212935 | 168.2744 | `compiled` |

Relative loss difference `1.68e-6` — AMP-noise scale, matching docs/20's
eager-vs-compiled finding. **`torch.compile` is compatible with the FramePose
graph on the training host.**

The runner now records the chosen backend in `experiment_matrix.json["shared"]`.
No throughput claim is made from this test: at 2 epochs (90 steps) the compiled
run is dominated by compilation warmup, so its frames/s figure is not a
steady-state measurement and is not reported as one. No throughput tuning was
performed.

## 12. Reporting contradiction fixed

docs/23 said feature caching "makes F1/F2 training cost the same as F0" while
the same document reported 6,903 vs ~1,040 frames/s. Both the docstring in
`framepose/features.py` and docs/23 now say the correct thing:

- caching removes **backbone inference** from the training loop entirely;
- it does **not** make a visual candidate as cheap as F0, because the fusion
  model still runs cross-attention over 196 image tokens per frame — measured
  ~6.6x throughput difference.

All raw numbers are preserved.

## 13. Third-party dependency audit

Scope: `pyproject.toml` extras, all three Dockerfiles, `third_party/`, every
import in `src/`, `scripts/`, `tests/`, and the installed environments (macOS
authoring venv; `animcv-train:cuda118`; `animcv-framepose:cuda118`).

| Dependency | Actual imports / owners | Historical owner | Future owner | Classification | Action |
| --- | --- | --- | --- | --- | --- |
| **MMPose** 1.3.2 | `pose/mmpose_adapter.py` (lazy), `pose/default_model.py`, `pose/pose_lifter.py` | v2 §3.2 pose backend **and** VideoPose3D 3D lifter | **2D Geometry Observation Layer only** | **KEEP** | Ownership written into v3 §5; `VideoPose3DLifter`/`lift-pose3d` demoted to legacy/reference |
| **MMDetection** 3.3.0 | `pose/mmpose_adapter.py`, `pose/default_detector.py` (RTMDet-tiny) | person detector for top-down MMPose | same — part of the Observation Layer | **KEEP** | none |
| **MMEngine / MMCV** | `mmengine.structures.InstanceData`, `init_default_scope`; otherwise transitive | OpenMMLab runtime | same | **KEEP** | none |
| **OpenCV** (`opencv-python`) | `mediaio/video_loader.py`, `app/cli.py`, `ui/gui_app.py`, `ui/debug_viewer.py` | video/image IO, debug overlays | same, plus any future custom tracking | **KEEP** | none |
| **Pillow** | `framepose/features.py`, `scripts/verify_frame_bank_images.py`, tests | — (new) | Frame Pose Core image decode | **KEEP** | already scoped to the `frame-pose` extra |
| **timm** 1.0.29 | `framepose/backbones.py` (lazy, inside the constructor) | — (new) | F1/F2 frozen visual towers | **KEEP AS OPTIONAL** | kept out of the geometry-only runtime; lazy import + isolation test |
| **torch** | pervasive | training/inference | same | **KEEP** | none |
| **torchvision** 0.16.2 | **no repo import**; timm dependency | — | — | **KEEP AS TRANSITIVE** | none |
| **huggingface_hub, safetensors** | **no repo import**; timm weight loading | — | — | **KEEP AS TRANSITIVE** | none |
| **transformers / any VLM runtime** | **not installed, not imported, not referenced** | — | — | **NOT PRESENT** | none — no language stack was ever added; F2 ran through timm |
| **SciPy** | `pose/mpi3dhp_adapter.py` (`loadmat`) | MPI-INF-3DHP intake | same | **KEEP** (`training` extra) | none |
| **smplx** | `pose/amass_adapter.py` | AMASS intake | same | **KEEP** (`training` extra) | none |
| **pyassimp** | `rig/rig_parser.py` (lazy) | rig parsing (v2 §3.3) | v2 Layer E, unchanged | **KEEP** | none |
| **bpy / mathutils** | `blender/*`, render scripts | Blender export (v2 §3.4) | v2 Layer E, unchanged | **KEEP** | none |
| **PyYAML** | `app/config.py` | config files | same | **KEEP** | none |
| **Depth Anything V2** | `pose/depth_estimator.py` (lazy), `pose/depth_sampling.py`, `app/cli.py --depth-*`, `ui/gui_app.py`, `tests/test_depth.py` | out-of-v2-scope relative-depth hint feeding `MotionPoint.position_3d` | **overlaps the Frame Pose Core**; no distinct Layer A owner | **KEEP AS LEGACY/OPTIONAL** | classified only; live CLI/GUI/test owners, not removed |
| **MediaPipe** | **absent** — no import, no extra, no mention anywhere | never integrated | — | **NOT A DEPENDENCY** | none |
| **DWPose** | **absent** — named only in v2 §1.3's exclusion list | never integrated | a future whole-body 2D estimator would be *another backend of the Geometry Observation Layer* | **NOT A DEPENDENCY** | none |
| **SAM2** | **absent** — named only in v2 §1.3's exclusion list | never integrated | plausible future owner: segmentation, occlusion, user-guided region tracking for arbitrary rigs | **NOT A DEPENDENCY** | none |
| **openmim** | `Dockerfile.pose` only (installs the prebuilt `mmcv` wheel) | build tooling | same | **KEEP** (build tooling) | none |
| `third_party/mmpose`, `third_party/Depth-Anything-V2` | **broken gitlinks**: recorded as submodule entries with **no `.gitmodules`**, directories empty; referenced in `pose/depth_estimator.py` docstrings as the API reference | reference checkouts | unclear | **UNRESOLVED** | reported, deliberately **not** removed |

### Environment separation (verified by inspection and by `pip list`)

| Image / extra | Contents | Owner |
| --- | --- | --- |
| `[pose]` / `Dockerfile.pose` | mmcv 2.1.0, mmdet 3.3.0, mmpose 1.3.2, numpy<2, opencv<4.10 | Geometry Observation Layer |
| `[training]` / `Dockerfile.train` | torch, scipy, smplx | Legacy Temporal Pose Baseline |
| `[frame-pose]` / `Dockerfile.framepose` | torch, timm, pillow (+torchvision, huggingface_hub, safetensors transitively) | Frame Pose Core |

The frame-core image contains **no** OpenMMLab package and **no** language
stack. That separation is now enforced by a test, not only by convention.

## 14. Dependencies actually removed

**None.** Every framework in the table has a current import, a live CLI/GUI
path, a test, a historical-reproduction role, or an intentionally supported
future path. MediaPipe, DWPose, SAM2 and `transformers` required no removal
because they were never dependencies of this repository in the first place. The
two `third_party/` gitlinks are ambiguous and were classified rather than
deleted, per the batch's own removal-safety rule.

## 15. Files changed in this batch

```
A  docs/24_WORKLOG_ARCHITECTURE_RECONCILIATION_AND_DEPENDENCY_AUDIT.md
A  src/framepose/observations.py
A  tests/test_frame_pose_observations.py
A  tests/test_frame_pose_dependency_isolation.py
M  Architecture_v3_FramePose.md      precedence, MMPose ownership, regimes, F0/F1/F2 scope, F2 terminology
M  CLAUDE.md                         architecture precedence pointer
M  docs/23_WORKLOG_...MIGRATION.md   regime label, rationale, causal control, F2 scope, throughput, diagnostic wording
M  src/framepose/contract.py         observation field, bank schema v2, frozen content-digest domain
M  src/framepose/sources.py          provenance resolution from input_kind; MMPose source-spec helper
M  src/framepose/bank.py             regime labelling, observation summary, unlabelled-bank refusal
M  src/framepose/evaluate.py         regime label on every report and per-frame record
M  src/framepose/train.py            regime + observation in the training report
M  src/framepose/features.py         corrected caching-cost docstring
M  scripts/build_frame_bank.py       prints regime + observation
M  scripts/run_frame_pose_experiments.py  refuses mixed regimes; records execution backend
```

## 16. Tests

New focused coverage:

- `tests/test_frame_pose_observations.py` (18): provenance read from the
  prepared artifact, every source registered, unknown `input_kind` refused,
  MMPose-vs-dataset regime invariants, cache invalidation across seven sensor
  fields and the input image, single-regime bank labelling, mixed-regime
  refusal, provenance not disturbing `content_digest`, v1 bank loading as
  unlabelled, evaluation reports carrying the regime, and the pinned
  content-digest domain.
- `tests/test_frame_pose_dependency_isolation.py` (5): the frame bank path
  imports with **no** heavy backend present; model/loss/evaluate import without
  timm/cv2/mmpose; the backbone registry imports without timm; MMPose stays
  behind its adapter; no `framepose` module imports `mmpose`/`mmdet`/`mmengine`.

Full regression at closure: **541 passed, 1 skipped** on macOS (the skip is the
timm-gated backbone test), and the framepose suite re-run in
`animcv-framepose:cuda118` on the training host.

GPU work: frame-bank rebuild (no training) and one 2-epoch execution-backend
compatibility smoke test. **No pose candidate was trained.**

## 17. Can this branch be the architectural baseline?

Yes. The six ambiguities named as the completion condition are closed:

| Question | Resolution |
| --- | --- |
| MMPose's role | 2D Geometry Observation Layer only; `VideoPose3DLifter` demoted to legacy/reference (v3 §5) |
| Provenance of 2D observations | `ObservationProvenance` on every sample, resolved from the artifact's own `input_kind`, with model/weights/config/preprocessing/image invalidation |
| FramePose vs Temporal-Lifter ownership | FramePose is the primary core; the temporal lifter is the Legacy Temporal Pose Baseline **and** a shared source of the loss/metric definitions (§3, named explicitly) |
| Architecture v2 vs v3 precedence | v3 normative for perception; v2 authoritative for Motion Graph / RigProfile / Blender / retarget (§7) |
| Meaning of F0/F1/F2 evidence | F1-vs-F2 causal about pretraining; F0-vs-F1/F2 a statement about tested architectures, not about RGB evidence (§9) |
| Ownership of third-party frameworks | Table in §13; nothing unowned, nothing removed, two gitlinks flagged UNRESOLVED |

Remaining known gaps, recorded rather than hidden:

- The Real Observation regime is contract-only; no MMPose observation bank
  exists yet.
- No parameter-matched geometry-only control exists, so the F0-vs-F1/F2 gap
  cannot be made causal about information alone.
- `third_party/` carries two gitlinks with no `.gitmodules`.
- `framepose/evaluate.py`'s reuse of the legacy metric helpers is protected only
  by metric tests, not by an equality test like the loss side has.
