# Worklog — Final Architecture-Contract Closure (2026-09-04)

> Closure batch on `arch/single_frame_first`. No model trained, no measured
> number altered. The three issues docs/24 left open are closed: the observation
> taxonomy mislabelled detector output as oracle geometry, the Frame Pose Core
> depended on the Legacy Temporal Lifter for its mathematics, and two
> `third_party/` gitlinks were unresolved.

## 1. Final observation taxonomy

The distinction that matters is **whether a learned 2D detector contributes to
the observation error** — not whether the keypoints shipped with a dataset. The
previous two-way split failed exactly there.

| Regime | 2D source | Detector error? | Purpose |
| --- | --- | :-: | --- |
| `oracle_geometry` | annotated/projected ground truth; deterministic synthetic projection from known 3D | no | isolate the 3D reconstruction architecture from 2D sensor error |
| `benchmark_detector_observation` | a benchmark's own distributed detector output (3DPW's shipped OpenPose-format keypoints) | **yes** | compare 3D reconstruction architectures under one fixed, externally defined observation |
| `real_animcv_observation` | AnimCV's MMPose + RTMDet sensor with its checkpoints and preprocessing | **yes** | measure actual AnimCV perception behaviour end to end |
| `historical_unknown` | provenance unresolvable | unknown | **refuses** any interpretation depending on observation quality |

Backend, observation type and regime remain separate fields, but they are not
independent — what produced an observation determines whether a detector is in
its error. The mapping is therefore **enforced**, not inferred from "a dataset
provided it":

```
backend                 observation_type              regime
dataset_ground_truth    projected_ground_truth_2d     oracle_geometry                 (MPI-INF-3DHP)
synthetic_projection    synthetic_virtual_camera_2d   oracle_geometry                 (AMASS)
dataset_detector        dataset_shipped_detector_2d   benchmark_detector_observation  (3DPW)
mmpose                  estimated_2d                  real_animcv_observation         (AnimCV)
```

`ObservationProvenance(BACKEND_DATASET_DETECTOR, ..., REGIME_ORACLE)` raises:
*"detector output is never oracle geometry merely because it shipped with a
dataset"*.

A bank pooling sources from different regimes is refused unless built with the
explicit `allow_mixed_regime` opt-in, which labels it `mixed`; the experiment
runner still refuses to interpret one. `assert_quality_interpretable` blocks any
observation-quality claim made from a `historical_unknown` or `mixed` artifact.

## 2. Existing F0/F1/F2 regime correction

The lineage's input is `official_3dpw_2d_detection` — 3DPW's own released
detector keypoints. It is now classified **`benchmark_detector_observation`**.

```
valid:    "F0/F1/F2 compare 3D reconstruction architectures under the SAME fixed
           3DPW dataset-shipped detector observations"

invalid:  "F0/F1/F2 isolate the 3D core from 2D observation error"
```

No metric, prediction, checkpoint, frame selection or feature cache changed.
docs/23 now opens with the corrected label and states plainly that an earlier
revision of it said `oracle_geometry` and was wrong.

## 3. Historical artifact migration behaviour

`ObservationProvenance.from_dict` resolves a recorded regime through
`migrate_regime`, which trusts the **backend** because the backend determines
the regime by definition:

| Recorded | Backend | Resolves to |
| --- | --- | --- |
| `oracle_geometry` | `dataset_detector` | `benchmark_detector_observation` (+ `detail.migrated_from_regime`) |
| `oracle_geometry` | `dataset_ground_truth` | `oracle_geometry` (unchanged) |
| `oracle_geometry` | `synthetic_projection` | `oracle_geometry` (unchanged) |
| `real_observation` | `mmpose` | `real_animcv_observation` |
| anything | unrecognised backend, unrecognised label | `historical_unknown` |
| absent (bank schema v1) | — | `UNRECORDED` / `historical_unknown` |

Not every old `oracle_geometry` label maps to the same new regime, and nothing
is guessed. A v1 bank still loads and is marked `historical_unknown`; any
quality claim from it is refused rather than silently made.

## 4. Digest / fingerprint semantics

Two digests, two jobs, neither overloaded:

| Digest | Covers | Job |
| --- | --- | --- |
| `content_digest` | sample identity, split, source, numeric arrays, under a frozen domain separator | governs **feature-cache validity** |
| `provenance_fingerprint` | observation provenance, modality, image references | detects a **provenance change**, including a corrected regime label |

Correcting the regime label of the existing lineage therefore moved
`provenance_fingerprint` and left `content_digest` — and every 6.57 GB feature
cache — untouched, which is exactly the required behaviour. A genuine sensor
change moves both, because different keypoints change `input_2d`.
`FrameBank.fingerprint()` reports both.

## 5. Old vs new dependency graph

Before:

```
framepose.losses    ──> training.temporal_lifter._vector_loss / _hinge_loss / BONE_INDICES ...
framepose.evaluate  ──> training.temporal_lifter._similarity_align / _root_yaw_error_degrees / _bend_direction
```

The legacy module was the implementation owner of the new core's mathematics.

After:

```
              common.canonical_pose          (single owner)
                 /              \
      Frame Pose Core     Legacy Temporal Pose Baseline
```

`grep -rn "temporal_lifter" src/framepose/` now returns only two docstring
mentions and **no import**, enforced by
`tests/test_frame_pose_dependency_isolation.py`.

## 6. Neutral canonical-pose mathematics ownership

`src/common/canonical_pose.py` owns: the canonical joint index map; `BONES`,
`HINGE_CHAINS`, `TORSO_PAIRS`, `YAW_PAIRS`, `END_EFFECTOR_NAMES` and their
resolved index tuples; `FORWARD_DEPTH_AXIS`, `BILATERAL_DEPTH_NORMALIZATION`,
`VECTOR_NORMALIZATION_EPS`; the torch-side `masked_mean`, `masked_chain_mean`,
`bend_vectors`, `vector_loss`, `hinge_loss`; and the numpy-side `angle_delta`,
`similarity_align`, `root_yaw_error_degrees`, `bend_direction`, `hinge_errors`.

It knows nothing about temporal windows, training configs, FramePose, experiment
identifiers or dataset sources, and imports no backend (`torch` is passed in).

`training/temporal_lifter.py` re-exports every moved name as a **direct alias**
(`_vector_loss = canonical_pose.vector_loss`, …) rather than a wrapper, so there
is exactly one implementation and no second formula can drift. Its training
loop, public API, config defaults and historical behaviour are untouched;
6,612 characters of duplicated implementation were removed, nothing was
rewritten.

## 7. Loss parity results

`tests/test_canonical_pose_parity.py` against
`tests/fixtures/canonical_pose_reference_v1.json`, captured from the pre-move
implementation at commit `68a5897`:

| Quantity | Result |
| --- | --- |
| `vector_loss` (bone indices) | **bitwise identical** |
| `vector_loss` (torso indices) | **bitwise identical** |
| `hinge_loss` | **bitwise identical** |
| `bend_vectors` (all elements) | **bitwise identical** |
| `masked_chain_mean`, `masked_mean` | **bitwise identical** |
| FramePose `compute_loss(BASELINE_GEOMETRY_V1)` vs legacy `_supervision_loss` | **bitwise identical** |
| FramePose `loss_components` bone/torso/hinge | **bitwise identical** to the fixture |

End-to-end spot check that the historical objectives still evaluate as before,
with the A14 and A16 branches on the same input: A9 `2.2295446396`,
A14 `2.1395099163`, A16 `2.2654564381`.

## 8. Evaluator parity results

This closes the asymmetry docs/24 named — the evaluator side had no direct
equality test.

| Quantity | Result |
| --- | --- |
| `root_yaw_error_degrees` | **bitwise identical** (both platforms) |
| `bend_direction` | **bitwise identical** (both platforms) |
| `angle_delta` | **bitwise identical** |
| `hinge_errors` joint order and `flipped` flags | **exact** |
| `similarity_align` | strict tolerance — see below |
| `hinge_errors` `error_degrees` | strict tolerance — see below |

Two quantities route through BLAS/LAPACK, whose last mantissa bits are
implementation-dependent. Measured macOS-Accelerate vs Linux-OpenBLAS difference
on the identical fixture input:

```
similarity_align      max abs 8.88e-16   max rel 4.56e-15   (numpy.linalg.svd)
hinge error_degrees   max abs 2.84e-14 degrees              (numpy.dot then arccos)
```

Bitwise equality is not technically available there, so those two use a strict
float64 tolerance (`1e-12`, and `1e-9` degrees) — orders of magnitude tighter
than any real formula change. Everything else is required back exactly.

`framepose.evaluate` is additionally asserted to hold the *same function
objects* as `common.canonical_pose`, and its `_hinge_direction_error` is checked
to agree with the historical per-chain errors on the same fixture.

## 9. Legacy Temporal Pose Baseline preservation

`src/training/temporal_lifter.py` **was** mechanically modified in this batch:
the canonical pose mathematics moved out to `common.canonical_pose`, which it now
consumes and re-exports. The accurate statement is therefore *historical
behaviour and mathematical semantics are preserved; implementation ownership of
the shared mathematics changed*.

- No checkpoint, prediction, metric, fingerprint or report altered.
- `TrainingConfig` defaults, the training loop, the augmentation path, the
  telemetry snapshot, the A14/A16 branches and the evaluator's gate criteria are
  unchanged.
- The private helper names every historical script and test imports
  (`_vector_loss`, `_hinge_loss`, `_similarity_align`, `_root_yaw_error_degrees`,
  `_hinge_errors`, `_bend_direction`, `_angle_delta`, `_bend_vectors`,
  `_masked_chain_mean`, `_masked_mean`) and every constant (`BONES`,
  `HINGE_CHAINS`, `END_EFFECTOR_NAMES`, `BONE_INDICES`, `TORSO_INDICES`,
  `HINGE_INDICES`, `END_EFFECTOR_INDICES`, `YAW_INDICES`,
  `VECTOR_NORMALIZATION_EPS`, `FORWARD_DEPTH_AXIS`,
  `BILATERAL_DEPTH_NORMALIZATION`) still resolve, now to the neutral owner.
- 508 tests pass on the training host and 556 on macOS, including the full
  historical temporal-lifter and evaluator suites.

## 10. MMPose final ownership

Refined: the **Geometry Observation Layer** is the abstraction, and MMPose +
RTMDet is its *current Real AnimCV backend* — one provider alongside the oracle
and benchmark-detector providers. MMPose is strictly a 2D observation backend:
not the abstraction itself, not the 3D Pose Core, not the temporal solver, not
the RGB reasoning layer. A future detector or whole-body estimator would be
another backend of the layer. `pose/pose_lifter.py`'s `VideoPose3DLifter`
(`lift-pose3d`), which uses MMPose's own 2D→3D lifter, remains **legacy and
reference only**. No `framepose` module imports `mmpose`, `mmdet` or `mmengine`.

## 11. Broken gitlink resolution

Both pins were verified to exist upstream (each is the current `main` tip), then
each was decided on **installation evidence**, not tidiness.

**`third_party/mmpose` — REMOVED.** MMPose is installed through pip/mim (the
`pose` extra; `Dockerfile.pose` runs `mim install mmcv==2.1.0` then
`pip install mmpose==1.3.2`). No code path referenced the checkout, and
`pose/default_model.py` resolves the default RTMPose-tiny config from the
*installed* package's `.mim/configs`. The gitlink had no functional owner. Stale
references in `README.md` and `README_EXEC.md` that told users to look in
`third_party/mmpose/configs` were corrected to point at the installed package.

**`third_party/Depth-Anything-V2` — KEPT PROPERLY.** Depth Anything V2 has no
PyPI distribution: `pose/depth_estimator.py` imports `depth_anything_v2.dpt`
from a source checkout, and its `ImportError` directs users to
`third_party/Depth-Anything-V2/requirements.txt`. The checkout **is** the
installation mechanism. A `.gitmodules` now records the path, the upstream URL
(`https://github.com/DepthAnything/Depth-Anything-V2.git`), the existing pin
`a561b849…` and the documented owner. `git submodule status` recognises it;
it stays uninitialised by default because the feature is optional.

A test asserts that no tracked gitlink lacks a `.gitmodules` entry.

## 12. Depth Anything ownership

Resolving the gitlink did **not** touch the feature. The runtime path
(`motion-tool estimate-pose --depth-checkpoint …`, the GUI equivalent,
`pose/depth_sampling.py`, `tests/test_depth.py`) is intentionally supported and
unchanged. It is classified **LEGACY / OPTIONAL** and is explicitly **not part
of Frame Pose Layer A**: it produces relative depth for
`MotionPoint.position_3d`, which `COORDINATE_CONVENTIONS.md` already forbids new
3D retarget code from consuming. Its installation mechanism is now valid and
unambiguous.

## 13. Final third-party dependency table

No entry remains UNRESOLVED.

| Dependency | Kind | Actual owner | Classification |
| --- | --- | --- | --- |
| MMPose 1.3.2 | runtime | `pose/mmpose_adapter.py`, `pose/default_model.py`; `pose/pose_lifter.py` (legacy 3D path) | **KEEP** — 2D Geometry Observation Layer |
| MMDetection 3.3.0 | runtime | `pose/mmpose_adapter.py`, `pose/default_detector.py` (RTMDet-tiny) | **KEEP** — part of the Observation Layer |
| MMEngine / MMCV | runtime (2 direct uses, else transitive) | `InstanceData`, `init_default_scope` | **KEEP** |
| openmim | build | `Dockerfile.pose` (prebuilt `mmcv` wheel) | **KEEP** — build dependency |
| OpenCV | runtime | `mediaio/video_loader.py`, `app/cli.py`, `ui/*` | **KEEP** |
| Pillow | runtime | `framepose/features.py`, `scripts/verify_frame_bank_images.py` | **KEEP** — Frame Pose Core image decode |
| torch | runtime | pervasive | **KEEP** |
| timm 1.0.29 | optional research | `framepose/backbones.py` (lazy) | **KEEP AS OPTIONAL** — F1/F2 towers, never required by the geometry-only runtime |
| torchvision | transitive | timm | **KEEP AS TRANSITIVE** |
| huggingface_hub, safetensors | transitive | timm weight loading | **KEEP AS TRANSITIVE** |
| SciPy | runtime | `pose/mpi3dhp_adapter.py` (`loadmat`) | **KEEP** (`training` extra) |
| smplx | runtime | `pose/amass_adapter.py` | **KEEP** (`training` extra) |
| pyassimp | runtime | `rig/rig_parser.py` | **KEEP** |
| bpy / mathutils | runtime | `blender/*`, render scripts | **KEEP** |
| PyYAML | runtime | `app/config.py` | **KEEP** |
| Depth Anything V2 | legacy feature (source checkout, submodule) | `pose/depth_estimator.py`, `app/cli.py --depth-*`, GUI, `tests/test_depth.py` | **KEEP AS LEGACY/OPTIONAL** — not Layer A; gitlink now declared |
| `third_party/Depth-Anything-V2` | historical source reference **and** installation mechanism | as above | **RESOLVED** — `.gitmodules` with URL + pin + owner |
| `third_party/mmpose` | — | none | **RESOLVED — REMOVED** (pip/mim is the mechanism) |
| transformers / any VLM runtime | — | not installed, not imported, not referenced | **NOT PRESENT** |
| MediaPipe | — | absent everywhere | **NOT A DEPENDENCY** |
| DWPose | — | named only in Architecture_v2 §1.3's exclusion list | **NOT A DEPENDENCY** — a future whole-body 2D estimator would be another backend of the Geometry Observation Layer, not something owned by MMPose |
| SAM2 | — | named only in Architecture_v2 §1.3's exclusion list | **NOT A DEPENDENCY** — plausible future owner for segmentation / occlusion / user-guided tracking |

## 14. Future FramePose execution policy

`torch.compile` over forward + loss (backward and the optimizer step eager) is
the accepted path and is verified compatible with the FramePose graph on the
training host (docs/24 §11: relative loss delta `1.68e-6`).

- **Historical F0/F1/F2 stay eager.** Their reports and checkpoints record
  `execution_backend: "eager"`, and that provenance is not rewritten.
- **Future FramePose training uses the compiled path**, unless an experiment
  explicitly needs eager to compare against a historical eager result.
- The runner's `--compile-training-graph` flag stays **opt-in by default on
  purpose**: flipping the default would change what the recorded historical
  command reproduces. The policy is documented rather than encoded, and the
  chosen backend is recorded in the training report, the checkpoint and
  `experiment_matrix.json["shared"]["execution_backend"]`.

No compile mode changed, no benchmark re-run.

## 15. Files changed

```
A  .gitmodules
A  docs/25_WORKLOG_FINAL_ARCHITECTURE_CONTRACT_CLOSURE.md
A  src/common/canonical_pose.py
A  tests/fixtures/canonical_pose_reference_v1.json
A  tests/test_canonical_pose_parity.py
D  third_party/mmpose                    (gitlink)
M  Architecture_v3_FramePose.md          three-way taxonomy, two-digest contract, lineage relabel,
                                         canonical-math ownership, execution policy, closing flow
M  CLAUDE.md                             regime vocabulary, canonical-math ownership
M  README.md, README_EXEC.md             stale third_party/mmpose config paths
M  docs/23_WORKLOG_...MIGRATION.md       lineage relabelled to benchmark_detector_observation
M  src/common/… , src/framepose/{observations,contract,bank,evaluate,losses}.py
M  src/training/temporal_lifter.py       delegates to the neutral owner (aliases only)
M  scripts/{build_frame_bank,run_frame_pose_experiments}.py
M  tests/{test_frame_pose_observations,test_frame_pose_contract,test_frame_pose_dependency_isolation}.py
```

## 16. Tests

New: `tests/test_canonical_pose_parity.py` (7) — bitwise/strict-tolerance parity
for every moved formula, both loss and evaluator sides, plus a delegation check
that makes drift impossible by construction.

Extended: `test_frame_pose_observations.py` (23) — detector-is-not-oracle
invariant, GT/synthetic-is-oracle invariant, MMPose-is-real invariant, backend↔
regime enforcement, migration semantics including the deliberate non-uniform
mapping of old `oracle_geometry`, quality-interpretation refusal, mixed-regime
refusal and opt-in, and the content-digest vs provenance-fingerprint contract.
`test_frame_pose_dependency_isolation.py` (8) — adds the no-`temporal_lifter`
edge, canonical-math backend isolation, and resolved-gitlink state.

Full regression: **556 passed / 1 skipped** on macOS; **508 passed** on the
training host (non-GUI subset — the frame-pose image has no tkinter). No
training run.

## 17. Closure

Both completion questions answer **yes**:

- `arch/single_frame_first` no longer confuses detector observations with oracle
  geometry — the taxonomy separates them, the mapping is enforced, historical
  artifacts migrate deterministically, and the F0/F1/F2 lineage is relabelled
  without touching a measured number.
- The Legacy Temporal Lifter is no longer the implementation owner of the Frame
  Pose Core's mathematics — both consume `common.canonical_pose`, verified
  bitwise.
- Every tracked major third-party framework has a resolved, functional ownership
  state; the one remaining gitlink is a declared submodule with a real
  installation role, and the other is gone.

Remaining known gap, unchanged and recorded: the Real AnimCV Observation regime
is contract-only, and no parameter-matched geometry-only control exists.
