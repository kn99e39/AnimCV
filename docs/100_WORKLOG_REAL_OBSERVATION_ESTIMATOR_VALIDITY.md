# Worklog — Real Observation Estimator Validity (2026-10-28)

## Starting state and environment

**Branch and HEAD.** `arch/single_frame_first`, starting HEAD `3e223d1` ("docs: Worklog 99A server test results"). Local,
origin and LabServer63 were at the same commit. The worktree was clean apart from the usual untracked `.DS_Store`,
`.animcv_sync_stage/` and `docker/` on the server.

**Committed W99 / W99A state:**
- W99 is `01c0dc3` (CASE B; candidate R not promoted).
- W99A is `224caef` plus documentation commit `3e223d1`.
- W99A changed 15 production/test files. `src/pose/mmpose_adapter.py` gained the three-phase batched flow
  (`_detect_batch`, serial `SubjectTracker`, `_pose_batch`; `MMPoseConfig.batch_size = 8`).
- The cumulative audit is `4cc453e`.
- None of these was modified.

**Reading.** PRODUCT_VISION.md, Architecture_v3_FramePose.md, COORDINATE_CONVENTIONS.md, Worklogs 91–99, 99A and the
cumulative audit. W99 CASE B is accepted within its scope.

**Machines:**
- **Local M1 Pro:** RTMDet/RTMPose on CPU (the production default device) in the pose environment `.venv`
  (mmpose 1.3.2, mmdet 3.3.0, mmengine 0.10.7, torch 2.13.0).
- **LabServer63:** frozen S0 inference in the original FramePose image `animcv-framepose:cuda118`, on CUDA. No server
  image contains MMPose.

## 0. Answer

**CASE D — evaluation contract blocked at the first gate.**

The W99A batched MMPose implementation at its default `batch_size = 8` does not meet the numerical observation contract
declared before measurement: keypoint |dx|, |dy| ≤ 0.5 px against `batch_size = 1` and against the historical per-frame
implementation.

- **Where it failed.** On `downtown_bar_00`, two of 432 frames each have one keypoint moved by exactly one RTMPose SimCC
  output bin: left shoulder in frame 1158 and right hip in frame 1168, both by 2.18–2.20 px in the image. The derived
  neck, spine and pelvis landmarks move by half or a quarter of a bin.
- **Mechanism.** Batch-composition floating-point noise (confidences differ by about 1e-7) flips the SimCC argmax between
  neighbouring bins.
- **What did not change.** Everything else matched exactly: detections, the tracked subject, coverage, visibility,
  ordering and determinism.
- **What batch 1 does.** `batch_size = 1` reproduces the historical per-frame implementation bit for bit in every gate
  case.

As the directive requires, the frozen-estimator comparison (S0 on benchmark vs AnimCV observations) was **not run**. The
production implementation was not patched, and the parity contract was not relaxed after the fact.

The completion question cannot be answered from this batch. Its decisive blocker is an observation-backend contract
violation in the W99A batching, not the estimator.

**What was established before the stop:**
- Trusted checkpoints: both are byte-identical to the official OpenMMLab URLs.
- Sequential, single-thread model initialisation, so no concurrent use of the global `torch.load` patch.
- Exact reproduction of the pinned historical S0 predictions in the S0 harness (condition B).
- The full S0 observation contract (§3).

## 1. Actual MMPose implementation and checkpoints

| Item | Value |
|---|---|
| Detector | RTMDet-tiny, `rtmdet_tiny_8xb32-300e_coco.py` (mmdet `.mim`), `rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth`, sha256 `78e30dcc…54a7fb` |
| Pose model | RTMPose-tiny, `rtmpose-t_8xb256-420e_coco-256x192.py` (input 192×256, SimCC split ratio 2.0), `rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth`, sha256 `77ffc7e8…a6fb7` |
| Provenance | the official URLs in `src/pose/default_detector.py` / `default_model.py` |
| Production settings | person label 0, detector score ≥ 0.3, `visibility_threshold` 0.3, `subject_box` None (the tracker starts on confidence × area) |

**Integrity check.** Both URLs were downloaded again into an isolated scratch folder and only byte-hashed (not loaded).
They are identical to the cached files. The RTMPose file's name carries `e613ba3f` while its sha256 begins `77ffc7e8`;
that is a property of the official file, not a local alteration.

**Loading.** Models are initialised once, in the main thread, sequentially (`threading.active_count() == 1` at
initialisation). `torch.load` goes through the production `_mmengine_checkpoint_compat` exactly once per model, with no
concurrent entry. Only these two local, hash-pinned checkpoints were loaded.

## 2. First gate — W99A MMPose batch correctness

### Declared before measurement (`PREDECLARED` in `scripts/run_real_observation_mmpose.py`)

**Variants:**
- **M1:** the production `PoseEstimator`, `batch_size = 1`.
- **M8:** the production `PoseEstimator`, `batch_size = 8` (the default).
- **M8 repeat:** a second M8 run, for determinism.
- **H:** the historical per-frame `PoseEstimator` loaded verbatim from git (`01c0dc3:src/pose/mmpose_adapter.py`, blob
  sha256 `0582a2a4…`). No production file was touched.

**Detection parity.** The production `_detect_batch` helper with 1 vs 8 frames per pass. Batching is controlled
explicitly through `MMPoseConfig.batch_size`, not through `ANIMCV_WORKERS`.

**Parity contract:**
- identical person-candidate counts;
- boxes within 0.5 px, scores within 1e-3;
- selected-subject IoU ≥ 0.99;
- keypoints within 0.5 px, confidences within 1e-3;
- visibility identical (except near-threshold flips, reported);
- identical frame order and coverage;
- bitwise-deterministic M8.

The stated justification for 0.5 px was W99A's 16-frame smoke result (≤ 0.0015 px). In hindsight, the claim that 0.5 px
was "below the SimCC resolution in image space" was wrong: one SimCC bin is 0.5 input px, which is about 2.2 px in image
space for a 700 px box. The contract was **not** redefined after this was observed.

**Population** (fixed before running). Six 3DPW videos, all evaluation frames in frame order, 1,893 frames:

| Video | Frames | Why it was chosen |
|---|---|---|
| downtown_walkBridge_01 | 237 | one clearly visible person (landscape) |
| downtown_bar_00 | 432 | multiple people and occlusion (portrait) |
| downtown_arguing_00 | 337 | two interacting people |
| downtown_runForBus_01 | 193 | distant, moving person among many |
| flat_guitar_01 | 150 | seated, deep flexion |
| outdoors_parcours_00 | 544 | long sequence |

Two controlled cases (56 frames):
- 2 leading black frames plus 6 black frames inserted mid-sequence: no detection, and the person leaving and returning;
- 12 landscape (1920×1080) and 12 portrait (1080×1920) frames in one sequence: varying image sizes.

### Results (1,949 frames)

| Case | Frames | Person candidates / frame | M1 vs M8 max keypoint Δ | H vs M1 | H vs M8 | M8 repeat | Detections 1 vs 8 | Selected subject | Tracked frames |
|---|---|---|---|---|---|---|---|---|---|
| walkBridge_01 | 237 | 5.76 | 0.0 px | 0.0 | 0.0 | identical | identical | IoU 1.0 | 237/237 |
| **bar_00** | 432 | 2.39 | **2.203 px** | 0.0 | **2.203 px** | identical | identical | IoU 1.0 | 432/432 |
| arguing_00 | 337 | 2.88 | 0.0 | 0.0 | 0.0 | identical | identical | IoU 1.0 | 337/337 |
| runForBus_01 | 193 | 7.67 | 0.0 | 0.0 | 0.0 | identical | identical | IoU 1.0 | 193/193 |
| flat_guitar_01 | 150 | 1.88 | 0.0 | 0.0 | 0.0 | identical | identical | IoU 1.0 | 150/150 |
| parcours_00 | 544 | 1.11 | 0.0 | 0.0 | 0.0 | identical | identical | IoU 1.0 | 544/544 |
| leave / return / empty | 32 | 5.03 | 0.0 | 0.0 | 0.0 | identical | identical | IoU 1.0 | 24/32 (8 empty frames reported) |
| mixed image sizes | 24 | 3.58 | 0.0 | 0.0 | 0.0 | identical | identical | IoU 1.0 | 24/24 |

**Across every case:**
- confidence differences ≤ 1.7e-6;
- zero visibility flips, zero presence mismatches;
- identical tracking reports between M1, M8 and H.

**Forward passes.** M8 ran detector and pose passes of 8 frames (the remainder batch at the end of a video). M1 ran
passes of 1, so `batch_size` controls the model batch.

**CPU time** (warm models, same process): M8 is 26–33% faster than M1 and H on every video. For example, bar_00 took
70 s vs 102 s (M1) and 99 s (H).

### Smallest reproduction (`gate-detail`, `gate/gate_detail_downtown_bar_00.json`)

| Frame (position in batch) | Landmark | M1 vs M8 | SimCC bin in image px | In bins | Confidence M1 / M8 |
|---|---|---|---|---|---|
| 1158 (slot 3) | left_shoulder | 2.2032 px | 2.2032 | 1.000 | 0.50898 / 0.50898 |
| 1158 | neck (shoulder midpoint) | 1.1016 | 2.2032 | 0.500 | 0.50898 / 0.50898 |
| 1158 | spine | 0.5508 | 2.2032 | 0.250 | 0.45063 / 0.45063 |
| 1168 (slot 7) | right_hip | 2.1831 | 2.1831 | 1.000 | 0.56558 / 0.56558 |
| 1168 | pelvis (hip midpoint) | 1.0915 | 2.1831 | 0.500 | 0.46442 / 0.46442 |
| 1168 | spine | 0.5458 | 2.1831 | 0.250 | 0.46442 / 0.46442 |

Six of 6,912 landmark values on bar_00 differ, and 0 of the remaining 24,000+ gate landmark values do. The mechanism is a
discrete argmax flip of the SimCC classifier between adjacent bins. It is triggered by floating-point differences of the
batched forward pass (confidence changes of about 1e-7), and it propagates through the canonical midpoint rules.

**Gate verdict: FAIL** under the declared contract (the keypoint bound). Detection, identity, coverage, ordering and
determinism parity all hold. **STOP** per §1 and §14 of the directive.

## 3. Observation contract audit

This is a code-level audit. It needs no observations and was completed before the stop.

**S0 input** (`framepose/contract.py`, `crops.py`, `train.py:93`):
- `input_2d` (17, 3) = `[x/W, y/H, conf]`, with W and H the bank's recorded image size (from the 3DPW intrinsics:
  `(round(2cx), round(2cy))`);
- `input_valid` (17,);
- geometry is re-expressed in a square crop around the valid joints (side = 1.5 × extent, ≥ 32 px) as
  `(x, y, conf, valid)`, zeroed where invalid;
- signs are neutral (zeros).

**Joint order:** pelvis, l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle, spine, thorax, neck, head, l_shoulder,
l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist.

**Benchmark mapping** (3DPW shipped OpenPose-18; `three_dpw_adapter`, `temporal_lifter.build_dataset`):
- the 12 limb joints are direct;
- head = nose;
- neck = **thorax** = shoulder midpoint (OpenPose's own neck keypoint is unused);
- pelvis = hip midpoint;
- spine = midpoint(neck, pelvis);
- midpoint confidence = min;
- valid = confidence > 0, invalid written as zeros;
- no in-frame masking.

Bank provenance: `backend = dataset_detector`, regime migrated to `benchmark_detector_observation`.

**AnimCV observation** (`pose/mmpose_adapter._extract_canonical_landmarks`):
- COCO-17 → the same derived landmarks (head = nose, neck = shoulder midpoint, pelvis = hip midpoint,
  spine = midpoint, min confidence);
- `visible = score ≥ 0.3`;
- coordinates and confidences are never zeroed;
- thorax is not produced, so the benchmark rule `thorax := neck` applies.

**Differences an adapter must carry explicitly:**
1. **Validity.** Benchmark `conf > 0` vs production `score ≥ 0.3`. The declared adapter uses the production flag and
   writes invalid joints as zeros, so a different validity distribution is part of the observation, not corrected.
2. **Confidence scale.** OpenPose part-affinity confidence vs RTMPose SimCC score. They are not calibrated to each
   other, and S0 consumes confidence as an input feature.
3. **No-subject frames.** The real backend can return no landmarks, so the row must stay with all joints invalid,
   whereas the benchmark build drops rows without a valid pelvis.
4. **Identity.** The benchmark observation is pre-associated to a dataset actor. The real backend tracks one subject
   per video (conf × area start, then IoU), so on 2-actor 3DPW videos one actor's rows necessarily refer to another
   person unless the tracker happens to follow that actor.
5. **Provenance.** A real observation requires the `real_animcv_observation` regime with `mmpose_observation(...)`
   fields and per-image byte digests (`observation_cache_key` refuses paths). No canonical_v1 → bank adapter existed
   before this batch.

**Implemented but not executed because of the stop:** `scripts/eval_real_observation_estimator.py` implements this
adapter (`build-observations`), the identity classes, the oracle-associated control, the probes, the S0 evaluation and
the rig replay.

**Mapping blocker:** none was found. The mapping is establishable without GT.

## 4. S0 harness identity (condition B)

`scripts/run_real_observation_s0.py` reproduces the historical inference contract (`run_sign_experiments._evaluate`):
- checkpoint `LabServer63:~/animcv-output/framepose/sign_v1/S0/checkpoint.pt`, sha256 `6c12fea9…47b44`,
  candidate `S0_neutral_sign`;
- `geometry_tensor`, neutral signs, `predict` batch 512, AMP off;
- frozen bank `bank_3dpw_paired_v2` (content digest `75519e63…0ed536`).

On CUDA in `animcv-framepose:cuda118`, condition B reproduced the pinned predictions bit for bit:
- VALIDATION: 3,407 rows, `402d70a2…`;
- TEST: 7,076 rows, `4357c7bd…`.

The harness would hold every S0 variable fixed except the observation arrays. It was run only for B.

## 5. Not executed (stopped)

Each item below depends on actual AnimCV observations from the gated backend:
- the full-population observation run;
- S0 on M / Moa / probes;
- coverage and identity accounting;
- paired 3D metrics;
- failure attribution;
- the HMR 2.0b comparison;
- J1X;
- the unseen-video run;
- the rig replay and Blender.

No reconstruction result exists for them, and none is implied.

- **J1X (audited):** its runtime inputs are 2D-derived crop geometry, plus Qwen3-VL-8B readouts sampled at the 448 px
  FramePose crop and the detector joint positions. Both depend on the 2D observation. The frozen readout caches were
  pruned from LabServer63 on 2026-10-28, so a J1X real-observation run would need new RGB features: two inputs would
  change, not the detector alone.
- **Unseen video (identified, not run):**
  - `examples/production_demo/source.mp4`: 199 frames, 50 fps, a public single-performer clip. It was used only in the
    legacy public-path audits of Worklogs 02–04, never for FramePose/S0 selection.
  - `examples/quality_demo/source.mp4` and `official_demo.mp4`: 5-frame MMPose PoseTrack test clips.
  - `examples/e2e_demo/person.mp4` repeats one still image. No independent 3D reference exists for any of them.
- **3DHP:** not prepared.

## 6. TEST reuse and generalization caveats

The 3DPW VALIDATION/TEST identities (10,483 rows) were declared as the population. They have informed many earlier
architecture decisions and are diagnostic, not untouched holdouts. No population statistic was computed.

## 7. W99A production-compatibility boundaries

**Directly measured here:**
- MMPose batch 1 = historical, bitwise;
- batch 8 ≠ historical by one SimCC bin in 2 of 1,949 frames.

The latter is the reason for this CASE.

**Unverified by this batch** (unchanged, and not validated through the estimator experiment):
- depth batching (no real-model parity measurement);
- Blender keyframing without per-sample `frame_set` (no Blender run);
- keyframe collapse (only the key count was compared, not key placement or motion);
- the FramePose rest-value caches (rely on `RestBone` immutability);
- contact / semantics changes;
- frame I/O threading;
- the median filter and importance vectorization (unit-test equivalence only).

## 8. Tests

**New file:** `tests/test_real_observation_estimator.py` (6 tests):
- benchmark representation rules (thorax = neck = shoulder midpoint, x/W y/H, invisible → zeros, no subject → all
  invalid);
- W73 identity distance and classes;
- raw coordinates for the mask probe;
- the SMPL projection joint map;
- GT-free construction of the runtime condition;
- gate parity helpers.

**Runs:**
- 6 passed with the system Python.
- 20 passed in `.venv`, together with `test_mmpose_adapter.py`, `test_subject_tracker.py` and
  `test_parallel_vectorized_paths.py`.
- 16 passed on LabServer63 (docker `animcv-external:cu124`), together with `test_parallel_vectorized_paths.py` and
  `test_subject_tracker.py`.
- The gate itself, the gate-detail reproduction and the S0 condition-B reproduction are experiment runs (§2, §4).

**Not run:** broad regression (no shared contract changed) and any test of the unexecuted evaluation paths.

## 9. Reproduction

```
# gate (local, pose environment, CPU)
.venv/bin/python scripts/run_real_observation_mmpose.py gate --population ~/animcv-local-data/w97/population.json \
    --image-root ~/animcv-local-data/w91/img --out-dir ~/animcv-local-data/w100/gate
.venv/bin/python scripts/run_real_observation_mmpose.py gate-detail --video downtown_bar_00 \
    --population ~/animcv-local-data/w97/population.json --image-root ~/animcv-local-data/w91/img --out-dir ~/animcv-local-data/w100/gate
# S0 harness identity (LabServer63)
docker run --rm --gpus all -e PYTHONPATH=/w/src -v ~/AnimCV:/w -v ~/animcv-output:/o -v ~/animcv-data/external/w100:/x -w /w \
    --entrypoint python3 animcv-framepose:cuda118 scripts/run_real_observation_s0.py \
    --bank /o/framepose/bank_3dpw_paired_v2.json --checkpoint /o/framepose/sign_v1/S0/checkpoint.pt \
    --observations /x/obs_empty.npz --out-dir /x/s0_check
```

**Evidence:**
- Local: `~/animcv-local-data/w100/gate/{gate_report.json, gate_detail_downtown_bar_00.json, historical_mmpose_adapter_01c0dc3.py}`.
- Server: `~/animcv-data/external/w100/s0_check/s0_report.json`.
- Archived on LabServer63 at `~/animcv-output/framepose_real_observation_estimator_archive/`.

The 3DPW frames used are the bank's frames from `~/animcv-data/datasets/3dpw/imageFiles`.

## 10. Changed files

**New files:**
- `scripts/run_real_observation_mmpose.py`: gate, gate-detail, and the observe and unseen modes; the last two were not
  executed.
- `scripts/run_real_observation_s0.py`: S0 harness, executed for condition B.
- `scripts/eval_real_observation_estimator.py`: observation adapter, evaluation and rig replay; not executed.
- `tests/test_real_observation_estimator.py`.
- This worklog.

No production module, CLI, GUI, FramePose or retarget file changed. No historical worklog, result or the audit
document changed.

## 11. CASE and limits

**CASE D.** The interpretable comparison is blocked at Gate 1, because the W99A default batching violates the declared
observation parity contract.

**Limits of what the failure means:**
- The violation is small in magnitude and rare: one SimCC bin, 2 frames of 1,949.
- It is discrete and fully explained.
- Batch 1 is bitwise historical.

It still means the default production backend is not numerically identical to the observation contract that preceded
W99A, beyond the tolerance declared for this study.

**What the batch therefore does not establish:**
- whether frozen S0 remains reliable under real AnimCV observations;
- which layer would limit it.

**What it does establish:**
- trusted model provenance;
- safe sequential loading;
- an exact S0 harness;
- a complete, GT-free observation mapping;
- the identity problem the real backend introduces on multi-actor 3DPW videos.
