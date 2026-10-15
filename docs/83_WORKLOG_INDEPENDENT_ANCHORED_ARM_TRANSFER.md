# Worklog — Independent Cross-Domain Validation of Geometry-Anchored Visual Correction (2026-10-15)

**Question.** Does Worklog 82's frozen geometry-anchored RGB correction (R1) keep its connected-arm accuracy and its temporal advantage over the direct RGB-conditioned arm (J1), and its out-of-plane gains over J0 and R0, on an independently selected paired RGB↔3D video source?

**Answer.** No. **CASE C (no reliable external advantage)** by the predeclared rule. The source is the MPI-INF-3DHP official test set: 6 subjects, 2,570 evaluated rows, 2,307 pairs.

**What fails and what holds:**

| comparison (3DHP, sequence-cluster 95% interval) | wrist %L | elbow %L | out-of-plane wrist %L | upper-arm motion residual (0.4 s) |
|---|---|---|---|---|
| R1 − J1 | **+2.42 [+0.86, +4.40]**, better in 1/6 | **+1.35 [+0.45, +2.57]**, better in 0/6 | **+6.12 [+2.69, +9.87]**, better in 0/5 | +0.26 [−0.66, +1.25], better in 2/6 |
| R1 − J0 | **−2.40 [−4.08, −0.89]**, better in 6/6 | −1.33 [−3.26, +0.50] | **−7.62 [−10.16, −4.41]**, better in 5/5 | −0.76 [−2.80, +0.77] |
| R1 − R0 | −1.40 [−2.89, +0.02] | −1.00 [−2.68, +0.79] | **−4.88 [−7.10, −2.36]**, better in 4/5 | −0.51 [−2.25, +0.78] |
| R0 − J0 | **−1.00 [−1.65, −0.39]**, better in 5/6 | −0.34 [−0.85, +0.19] | **−2.73 [−3.48, −1.81]** | −0.25 [−0.68, +0.17] |

%L = percent of the GT total arm length (upper arm + forearm).

- **Static order reverses relative to 3DPW.** On 3DHP the ranking is J1 < R1 < R0 ≲ J0. R1 is significantly worse than J1 on wrist, elbow, upper-arm, chain and out-of-plane wrist, and improves on J1 in at most 1 of 6 sequences. On 3DPW the identical canonical metric code gives the Worklog 82 order: R1 beats J1, wrist −0.96 [−1.62, −0.23].
- **R1's RGB-specific out-of-plane gain transfers.** Against R0 and J0, out-of-plane wrist improves by −4.9 and −7.6 %L. The absolute size of the correction is about the same as on 3DPW: in J0's worst-quartile rows R1 reduces wrist error by 12.6 %L (3DPW: 12.8).
- **What changes is the anchor.** J0's 2D-only prior degrades much more than J1 under the domain shift:
  - J1 − J0 wrist is −4.83 %L on 3DHP vs −1.48 on 3DPW;
  - J1 − J0 out-of-plane wrist is −13.7 vs −7.3.
  - R1 closes only about half of that larger anchor deficit.
- **Temporal advantage over J1 does not transfer.**
  - At the primary 0.4 s interval, R1 − J1 upper residual is +0.26° (not significant) and forearm residual is +1.66° [+0.13, +3.31] (worse).
  - In the 0.16 s outdoor secondary analysis (TS5–TS6, two sequences, descriptive only), R1 is less stable than J1 (+2.68° upper) and J0 (+1.07°). Its correction changes 10.6–18.7° per 0.16 s, against R0's 3.3–6.7°.
- **Limits of this evidence.** It covers canonical connected-arm geometry only; rig-space IK is not reported. There are only 6 subject/sequence clusters, so every generalization claim is weak by construction.

## Starting state and files

- **Start:** `arch/single_frame_first` at `394967f` (Worklog 82 docs). Local, origin and LabServer63 were equal, and the worktree was clean.
- **Commits, in order:**
  1. `f143cd8`: protocol declaration, committed before any candidate prediction.
  2. `25c21bf`: feasibility-gate amendment (identity and camera).
  3. `7244668`: reproduction gate fed the stored Worklog 82 anchor.
  4. `1fcf093`: validity rule limited to candidates.
  5. `6aeb574`: GT pair-amplitude bug fix.
  6. The commit adding this worklog.

  Commits 2–5 are explained under *Protocol changes*.
- **Files added (diagnostic and evaluation only; nothing in production imports them):**
  - `src/framepose/mpi3dhp_test_transfer.py`: 3DHP test joint table, camera conversion, population and pair rules, calibration check, canonical connected-arm metrics.
  - `scripts/build_mpi3dhp_test_external_bank.py`: feasibility gates, re-run detector, Worklog 73 association, external bank.
  - `scripts/predict_framepose_anchored_arm_transfer.py`: frozen-artifact identities, historical reproduction gate, frozen J0/J1/R0/R1 on 3DHP.
  - `scripts/run_framepose_anchored_arm_transfer_eval.py`: predeclared `PROTOCOL` / `DECISION`, metrics, contrasts, regimes, correction behaviour.
  - `scripts/review_framepose_anchored_arm_transfer.py`: canonical 3D review panels.
  - `tests/test_framepose_anchored_arm_transfer.py`
  - this worklog
- **Reused unchanged:**
  - the Worklog 73 `ExternalBank`, `associate`, OpenPose COCO-18 detector and `to_canonical_input`;
  - the Worklog 68/73 Qwen extractor (`extract_external_qwen3vl_features.py`);
  - Worklog 80/82 model code;
  - Worklog 75–77 statistics (`_contrast`, `_bootstrap`, `_stats`).
- **Unchanged:** every historical module, checkpoint, cache and report; production FramePose/FK/IK/Blender; H0, E1, Root Orientation and BaseRig.

## Frozen artifacts (sha256, verified before use)

| artifact | sha256 |
|---|---|
| J0 `J0_GEOMETRY_ONLY.pt` (Worklog 80) | `5332b691a58b4135ce9cb4188dba0b0f4050bf11d56b478a88de03386d5f3ba5` |
| J1 `J1_GEOMETRY_PLUS_VISION.pt` (Worklog 80) | `25f0c60f48c51bd474369fab924b3ea70e2d040364091479cd1697a8f8e8ee4b` |
| R0 `R0_GEOMETRY_CORRECTION.pt` (Worklog 82) | `2f015d67d17775c917db17dd23da735b51c278dcfeb579f8f4a31bb4f8c9b79b` |
| R1 `R1_VISUAL_CORRECTION.pt` (Worklog 82) | `873777734d9f44d18951830b111ff2ce6f9cf3d8912a52741e0125053cdd3fab` |
| Worklog 80 predictions | `eaf3f57c…` |
| Worklog 82 predictions | `a8bfbaea…` |
| Worklog 68 3DPW Qwen readout | `f5d0eb9c…` (model fingerprint `a0d72ded…`) |

The frozen state digests were identical before and after evaluation. No weight was trained, reselected or adapted, and every model ran in eval mode.

## Dataset selection and identity

- **Source:** the MPI-INF-3DHP official test set, `mpi_inf_3dhp_test_set.zip` from `vcai.mpi-inf.mpg.de`.
  - Last-Modified 2021-07-11, 6,936,910,586 bytes, sha256 `eab3d09f…`.
  - Six sequences TS1–TS6, one subject and one camera each:
    - TS1–TS2: studio with green screen;
    - TS3–TS4: studio without green screen;
    - TS5–TS6: outdoor.
- **Why this source:**
  - RGB frames with dense camera-space 3D (mm) and 2D annotations;
  - official per-sequence calibration;
  - subjects, scenes and an outdoor setting none of the model training sources contain;
  - frame-ordered video;
  - activities with substantial out-of-plane arm motion;
  - compatible with the re-run detector.
- **Why not Human3.6M:** it is a single-studio source (doc 08 rejected it on scene diversity) and needs a separate licence and video decoding. It was not evaluated, and no second source will be.
- **Annotation files:** `annot_data.mat` (MATLAB v7.3) with `annot2` (N×17×2 px), `annot3` (N×17×3 camera mm), `univ_annot3`, `valid_frame` and `activity_annotation`.
  - Read with h5py 3.11.0, installed into a side directory for annotation reading only.
  - File hashes are recorded in the build report.
- **Joint order:** the official `mpii_get_joints('relevant')` order. Left shoulder, elbow and wrist are indices 5, 6 and 7, with the subject's anatomical left; the test checks the mapping by name.

**Frame rate.** The official README does not state it. The community convention is 25 fps for TS1–TS4 (studio) and 50 fps for TS5–TS6 (GoPro outdoor).

| evidence | fits 25 / 50 fps? |
|---|---|
| TS3 GT walking pelvis speed: 41.5 mm/frame median | ≈1.0 m/s at 25 fps; 2.1 m/s at 50 fps would be implausible for walking |
| TS5 detector-agreement curve is flat over ±1 frame | expected at 50 fps |

Pairs are defined in frames, so a wrong outdoor frame rate would only change the stated seconds.

**Official evaluation frames (`valid_frame`):**

| sequence | rows | valid | valid spacing |
|---|---:|---:|---|
| TS1 | 6,151 | 603 | every 10th frame |
| TS2 | 6,080 | 540 | every 10th frame |
| TS3 | 5,838 | 506 | every 10th frame |
| TS4 | 6,007 | 558 | every 10th frame |
| TS5 | 320 | 276 | consecutive |
| TS6 | 492 | 392 | consecutive |

Non-valid frames are never used, neither as rows nor as temporal partners.

## Independence audit

- **Not used in Worklog 73/80/82 training or selection.** J0/J1 and R0/R1 were trained and selected on the 3DPW FrameBank (21,817 rows, all `source = 3DPW`) and the BEDLAM external banks (21,232 train and 1,330 validation rows, all `source = BEDLAM`), as verified from the bank indices. E1 uses the same sources.
- **The test set was never fetched or inspected before.** No earlier model prediction was scored on it. `scripts/training/build_mpi_inf_3dhp_manifest.py` states it was "not fetched".
- **The 3DHP training set was used in the legacy phase.** That was 2D/3D annotations of S1–S8, no RGB, for earlier temporal lifters (docs 04, 06, 09). Those lifters are not ancestors of J0/J1/R0/R1, whose inputs are the detector 2D and the Qwen readout. Whether TS1–TS6 subjects also appear in S1–S8 cannot be established from the shipped files. The studio is the same capture facility.
- **Pretraining exposure is unknowable.** The detector was trained on COCO, and Qwen3-VL-8B was pretrained on web-scale data; whether either saw 3DHP images cannot be established.
- **Claim:** this is the first evaluation of these architectures on these subjects, scenes and motions. It is not a claim that no model component has ever seen the images.

## Feasibility gates (run before any Qwen extraction or candidate prediction)

| sequence | image offset (row k → image) | agreement at chosen offset / ±3 frames | half-sequence argmins | pinhole fit residual (median px) | official-intrinsics reprojection (median px) | detector left wrist nearer GT left |
|---|---|---|---|---|---|---|
| TS1 | k+1 (nominal) | 0.031 / 0.043, 0.042 | 0, 0 | 0.07 | 0.00 | 521/528 |
| TS2 | k+1 | 0.038 / 0.066, 0.065 | 0, 0 | 0.12 | **4.74** | 387/404 |
| TS3 | **k (offset −1)** | 0.036 / 0.072, 0.071 | −1, −1 | 0.07 | 0.00 | 344/370 |
| TS4 | k+1 | 0.033 / 0.073, 0.070 | 0, 0 | 0.08 | 0.00 | 397/414 |
| TS5 | k+1 | 0.057 / 0.076, 0.098 | 0, −1 | 0.10 | 0.00 | 195/195 |
| TS6 | k+1 | 0.032 / 0.041, 0.041 | 0, 0 | 0.04 | 0.00 | 316/331 |

Agreement = the best detection's median limb-joint distance to `annot2`, as a fraction of the GT limb-box diagonal.

- **RGB↔annotation identity.** The nominal mapping is row k → `img_{k+1:06d}.jpg`.
  - It is confirmed for five sequences by a V-shaped agreement curve over image offsets −10…+10, probed at every 10th valid row.
  - **TS3 is shifted by one frame:** its annotation row k matches `img_{k:06d}`, with agreement 0.029 against 0.042 at the nominal image in the first half, consistently in both halves. TS3 ships one more image than annotation rows.
  - TS5 resolves only to ±1 frame (±20 ms at 50 fps). The nominal offset was within 0.4% of the minimum, so it was kept.
- **Camera convention.**
  - `annot3` is camera-frame millimetres with OpenCV axes (+x right, +y down, +z forward). `annot2` is its exact undistorted pinhole projection; per-sequence fits leave 0.04–0.12 px median residual.
  - TS2 was annotated with slightly different studio intrinsics than the single listed camera (fx +3.7 px, principal point −2.1 / −4.4 px). That is why its official-intrinsics reprojection is 4.7 px.
  - The listed TS5/TS6 lens distortion is not applied to `annot2`. The shipped TS5/TS6 images are already undistorted: detector agreement with the distorted projection is worse (TS5 0.067 vs 0.057; TS6 0.035 vs 0.032).
  - `pixelAspect` follows fy = fx · aspect: the fitted fy/fx is 0.9932, against the listed 0.9932.
- **Canonical target.** AnimCV (X right, Y depth, Z up) = OpenCV (x, z, −y), in metres, pelvis subtracted. This is the same conversion as the 3DPW and BEDLAM adapters. Intrinsics never enter an input or a metric.
- **Left/right.** On associated rows where both wrists are detected, the detector's left wrist is nearer GT left in 93–100% of rows.
- **Detector and crop parity.** The same contract as Worklog 73 BEDLAM:
  - the re-run OpenPose COCO-18 detector (Hzzone port, CMU weights `25a948c1…`, single scale, net height 368) on the full image;
  - the exact 3DPW conversion rules;
  - the Worklog 68 448 px crop built from detector joints;
  - the readout at the detector left shoulder, elbow and wrist.
- **How GT is used.** GT 2D is used only by the unchanged Worklog 73 `associate`: at least 6 of 12 limb joints within 0.1 of the GT limb-box diagonal, to pick the subject among detections. GT never enters `input_2d`, the crop, the readout or the model; the test checks that the builder's input lines never read `annot2`/`annot3`.

## Evaluation population

| sequence | valid frames | association failures | pelvis invalid | bank rows | evaluated rows (left arm GT-valid and detector-visible) | 0.4 s pairs |
|---|---:|---:|---:|---:|---:|---:|
| TS1 | 603 | 9 | 3 | 591 | 571 | 554 |
| TS2 | 540 | 48 | 1 | 491 | 457 | 408 |
| TS3 | 506 | 13 | 1 | 492 | 462 | 425 |
| TS4 | 558 | 13 | 1 | 544 | 507 | 459 |
| TS5 | 276 | 51 | 7 | 218 | 196 | 148 |
| TS6 | 392 | 0 | 0 | 392 | 377 | 313 |
| **total** | **2,875** | 134 | 13 | **2,728** | **2,570** | **2,307** |

- **Rows and pairs.**
  - 158 bank rows are not evaluated: the left-arm target is invalid, in the same sense as the Worklog 80 mask.
  - 263 evaluated rows have no evaluated partner 0.4 s later.
  - Secondary outdoor 0.16 s pairs: 500.
- **Coverage is identical:** J0, J1, R0 and R1 all have 2,570 rows, every output is finite, and the connected-arm contract holds to ≤ 4.4e-16 (|E| + |W−E| = 1, |W| ≤ 1).
- **Clusters:** one subject per sequence, so there are 6 independent clusters.
- **Regime distribution, 3DHP vs 3DPW test:**
  - out-of-plane (|f_GT| ≥ 0.5): 29% vs 24%;
  - near-straight (bend < 15°): 20% vs 16%;
  - near-folded (bend ≥ 90°): 15% vs 12%;
  - near-plane: 35% in both.

## Historical reproduction gate (passed before any 3DHP prediction)

Re-run on all 21,817 3DPW FrameBank rows, with the Worklog 68 cache verified by digest. Tolerance: 1e-6.

| check | max abs difference |
|---|---:|
| J0, J1 raw / elbow / wrist / rho vs Worklog 80 | **0.0** (bit-exact) |
| fresh J0 raw vs Worklog 82 stored anchor raw | 2.98e-8 |
| R0, R1 correction / elbow / wrist / rho / u / v vs Worklog 82 (fed the stored anchor) | ≤ 6.8e-7 |
| normalized targets | 0.0; mask identical |
| stored corrected arms: connected-arm contract | ≤ 4.4e-16 |

The end-to-end path (fresh anchor → R) differs by up to 1.4e-6. This is reported, not gated. Worklog 82 batched its anchor forward differently (2.98e-8 from Worklog 80), and unit-normalizing small raw vectors amplifies that round-off.

## Protocol changes (all before any candidate prediction or metric)

1. **Identity gate.** The first build assumed offset 0 and stopped on TS3. The second build required identical half-sequence argmins and stopped on the 50 fps TS5, whose curve is flat over ±1 frame. The final rule, committed in `25c21bf` before Qwen extraction:
   - keep the nominal offset unless it is more than 5% worse than the minimum;
   - half-sequence argmins within ±1 frame;
   - agreement at least 25% worse 3 frames away.
2. **Camera gate.** The official-intrinsics reprojection was replaced by a per-sequence pinhole fit, because of TS2's intrinsics. It must leave ≤ 2 px residual, focal lengths within 1% and principal point within 10 px of the official calibration.
3. **Reproduction gate.** The first prediction run stopped: J0/J1 were bit-exact, but R0/R1 were compared through a freshly computed anchor (1.4e-6 > 1e-6). The gate now feeds R0/R1 their exact historical input, the stored Worklog 82 anchor. Fixed in `7244668` before any 3DHP prediction existed.
4. **Two implementation bugs**, caught and fixed before any metric was read:
   - the validity rule had included the internal float32 anchor (5.96e-8), which is not a candidate (`1fcf093`);
   - the GT pair amplitude used the first frame twice. It crashed with a zero division before producing any number, and is now covered by a test (`6aeb574`).

No population, metric, contrast, threshold or decision rule was changed after a candidate number was seen.

## Canonical geometry and its relation to the historical metrics

- **Definitions.** S = 0, E = (elbow − shoulder)/L, W = (wrist − shoulder)/L, with L = GT upper + forearm length.
  - Positions are in %L.
  - Directions are angles of unit vectors (atan2).
  - Bend side is the Worklog 61 observed side (the elbow's offset from the shoulder→wrist line). The GT side is re-perpendicularized to the candidate chain and is undefined below a 1° bend.
- **Not interchangeable with Worklogs 76–82.** Those are BaseRig rig-space IK endpoint errors in cm, measured after H0-heading alignment. These are camera-frame, scale-free connected-arm errors.
- **No rig-space metrics:** 3DHP has no faithful H0 heading owner input, and no new heading owner was invented.
- **3DPW reference with the same code.** The same canonical code on the 3DPW test split (6,003 rows, 5,504 pairs at 1/6 s, 37 sequences) gives the Worklog 82 pattern:
  - R1 better than J1 static (wrist −0.96 %L [−1.62, −0.23]) and temporal (−0.93°/pair [−1.29, −0.59], 29/37);
  - R1 worse than J0 temporally (+2.51°/pair [2.21, 2.81]; the rig-space Worklog 82 value is also +2.51).

  So the 3DHP reversal is not an artifact of the metric change.

## Static geometry (3DHP, 2,570 rows; mean / p50 / p95 / max; sequence-weighted mean)

| metric | J0 | J1 | R0 | R1 |
|---|---|---|---|---|
| elbow %L | 22.4 / 17.8 / 53.1 / 107 (21.9) | **19.7** / 17.1 / 42.1 / 96 (19.5) | 22.1 / 17.5 / 52.7 / 106 (21.7) | 21.1 / 17.6 / 45.9 / 106 (20.8) |
| wrist %L | 29.6 / 22.8 / 71.5 / 164 (29.4) | **24.8** / 20.3 / 58.5 / 170 (24.7) | 28.6 / 22.9 / 70.9 / 164 (28.5) | 27.2 / 21.3 / 67.1 / 181 (27.0) |
| upper-arm direction ° | 23.1 | **19.9** | 22.7 | 21.6 |
| forearm direction ° | 25.1 | **23.4** | 25.4 | 25.3 |
| chain direction ° | 19.7 | **16.1** | 19.0 | 18.2 |
| reach %L | 7.9 | **6.5** | 7.8 | 6.8 |
| bend-side ° (where defined) | **33.6** | 38.0 | 37.0 | 40.5 |
| bend-angle error ° | 20.9 | **17.7** | 20.5 | 18.0 |

- **Every candidate is much worse here than on 3DPW.** With the same code, 3DPW elbow is 12.1–13.1 %L and wrist 19.6–22.0 %L.
- **J0 degrades most:** elbow +71%, against J1's +55%.
- **Bend side is the one metric where J0 is best.** Every RGB-conditioned candidate rotates the bend plane further from GT.
- **Paired contrasts:** see the table at the top. For R1 − J1, the moving-block intervals (10 or 25 rows) exclude 0 on every static metric; the sequence-cluster intervals exclude 0 on every one except forearm direction (+1.93° [−0.06, +3.58]). Moving-block intervals are narrower, as expected with only 6 clusters.

## Temporal motion (primary: 2,307 pairs at 0.4 s)

| residual vs GT motion | J0 | J1 | R0 | R1 |
|---|---|---|---|---|
| upper-arm direction °/pair | 20.0 | **19.0** | 19.8 | 19.3 |
| forearm direction ° | 24.9 | **22.9** | 24.8 | 24.6 |
| chain direction ° | 19.4 | **17.7** | 19.1 | 18.5 |
| elbow motion %L | 19.2 | **18.0** | 18.9 | 18.4 |
| wrist motion %L | 29.3 | **27.6** | 28.8 | 28.4 |
| reach variation %L | 8.5 | 8.2 | 8.6 | 8.4 |
| bend-side variation ° | 36.3 | 36.8 | 36.4 | 37.0 |
| amplitude ratio to GT (upper change; GT 28.3°) | 0.91 | 1.03 | 0.92 | 0.99 |
| amplitude ratio (wrist displacement; GT 46.4 %L) | 0.85 | 0.98 | 0.88 | 0.95 |

- **R1 vs J1:** upper +0.26° [−0.66, +1.25], 2/6 sequences; forearm **+1.66° [+0.13, +3.31]**; chain **+0.78° [+0.19, +1.66]**. R1 is not steadier than J1.
- **R1 vs J0:** upper −0.76° [−2.80, +0.77]; the moving-block-10 interval excludes 0, but the sequence-cluster interval does not.
  - At 0.4 s, frame errors decorrelate, so the residual mostly measures the change of static error, and the more accurate candidate scores lower.
  - Worklog 82's "R1 materially less stable than J0" is **not reproduced at this interval**. That is not evidence that R1's jitter is gone.
- **Discontinuities.** Thresholds are the GT p99 adjacent change on the same pairs (upper 111°, wrist 160 %L at 0.4 s). Candidate-specific rates are ≤ 1.3% for all candidates, and R1 is not worse than J0 or J1.
- **GT motion is smooth.** The annotation is not noisy: the GT direction change grows almost linearly with lag (TS5: 1.95°, 3.68°, 6.47°, 11.1° at 1, 2, 4, 8 frames) and its second difference is ≈ 1°. The large residuals reflect fast true motion (TS6: 22° per 0.16 s, against 3DPW's 6.8° per 1/6 s) that no candidate tracks well.

**Secondary, descriptive (TS5–TS6, 500 pairs at 0.16 s; 2 clusters, so intervals are not meaningful):**

| candidate | upper residual ° | forearm residual ° | wrist motion %L |
|---|---:|---:|---:|
| J0 | 19.3 | 24.4 | 28.9 |
| J1 | 17.7 | 23.9 | 29.2 |
| R0 | 19.1 | 23.5 | 28.3 |
| R1 | 20.3 | 25.5 | 31.1 |

- **R1 is the least stable candidate:** R1 − J1 upper +2.68°, R1 − J0 +1.07°, R1 − R0 +1.21°.
- **The R1 correction itself moves a lot:** it changes 10.6° (TS5) and 18.7° (TS6) per 0.16 s on the upper arm, against R0's 3.3° and 6.7°. Worklog 82's correction-induced jitter (9.7° per 1/6 s on 3DPW) is consistent with this.
- **R1 over-moves:** its amplitude ratio to GT is 1.31 for upper change, against J0's 1.11.

## Uncertainty and concentration

| sequence | R1 − J1 wrist %L | R1 − J1 elbow | R1 − J0 wrist | R1 − R0 wrist | R1 − J1 upper residual ° | R1 − J0 upper residual ° |
|---|---:|---:|---:|---:|---:|---:|
| TS1 (studio, green screen) | +1.81 | +1.98 | −4.30 | −3.76 | +0.77 | +0.86 |
| TS2 (studio, green screen) | +0.41 | +0.52 | −0.72 | +0.77 | −1.23 | +0.85 |
| TS3 (studio) | +3.59 | +0.51 | −0.01 | +0.04 | +0.50 | −0.46 |
| TS4 (studio) | +1.50 | +0.40 | −2.03 | −0.61 | −0.87 | −2.04 |
| TS5 (outdoor) | −0.63 | +0.32 | −1.67 | −1.90 | +0.05 | +2.08 |
| TS6 (outdoor) | +7.18 | +4.22 | −5.41 | −3.03 | +2.73 | −5.61 |

- **R1's static deficit to J1 is broad.** It appears in 5/6 sequences (elbow 6/6) and is not driven by one scene. TS6 contributes the largest share of the net wrist deficit (43%, below the 50% dominance line); without TS6, the deficit remains in 4/5 sequences.
- **R1's gain over J0 is broad too:** wrist improves in 6/6 sequences, and TS1 contributes the most (40%).
- **Worst regressions** are R1 − J1 wrist +93, +91, +83 and +72 %L, at TS6 rows 300, 60, 129 and 185, and +70 at TS1 row 800. In every one, J0 is already wrong (80–143 %L) and J1 is right (17–53 %L). R1 inherits the anchor's gross error and its correction does not recover it, even when the correction is large (101° at TS6 row 300).
- **Per-row shares:** R1's wrist is worse than J1 on 56% of rows, and worse than J0 on 45%.

## Pose and depth regimes (GT-derived bins, evaluation only)

| regime | rows | wrist %L J0 / J1 / R0 / R1 | elbow %L J0 / J1 / R0 / R1 | upper residual ° J0 / J1 / R0 / R1 | R1 − J1 wrist (sequences better) |
|---|---:|---|---|---|---|
| near-straight [0, 15) | 525 | 38.8 / **30.1** / 36.8 / 35.5 | 27.7 / **19.8** / 26.0 / 23.0 | 29.7 / **23.1** / 28.6 / 25.0 | +5.46 (1/6) |
| moderate [15, 45) | 942 | 26.6 / **22.6** / 25.8 / 24.5 | 20.1 / **18.9** / 19.8 / 19.4 | 17.0 / 16.9 / 16.8 / **16.6** | +1.90 (2/6) |
| strong [45, 90) | 718 | 27.2 / **23.4** / 26.6 / 25.0 | 21.2 / **19.5** / 21.3 / 20.7 | 17.6 / **17.5** / 17.6 / 18.1 | +1.57 (1/6) |
| near-folded [90, 180] | 385 | 28.8 / **25.4** / 27.8 / 26.5 | 23.1 / **22.1** / 23.8 / 23.2 | **18.9** / 21.5 / 19.3 / 20.1 | +1.12 (3/6) |
| near-plane, abs(f) < sin 10° | 890 | **15.8** / 17.4 / 16.0 / 18.1 | **16.3** / 16.3 / 16.4 / 17.1 | **15.1** / 15.5 / 15.2 / 15.6 | +0.66 (2/6) |
| intermediate | 924 | 26.7 / **23.0** / 26.0 / 24.1 | 19.0 / **18.3** / 18.9 / 19.0 | 18.5 / **18.2** / 18.3 / 18.4 | +1.09 (1/6) |
| out-of-plane, abs(f) ≥ 0.5 | 756 | 49.3 / **35.6** / 46.6 / 41.7 | 33.8 / **25.5** / 32.6 / 28.3 | 27.5 / **24.1** / 26.8 / 24.6 | +6.12 (0/5) |

f = depth component of the GT shoulder→wrist direction (camera frame).

Which historical patterns persist:

- **J0's near-plane strength persists.** It is the best candidate there.
- **The near-plane false-depth failure persists:** R1 − J0 wrist is +2.3 %L on near-plane rows.
- **J1's out-of-plane RGB evidence persists and is stronger:** J1 − J0 out-of-plane wrist is −13.7 %L.
- **R1's improvement over J1 does not persist.** It reverses in every regime, most of all for out-of-plane and near-straight arms.
- **Moderate-bend failure** (R1 − J1 elbow +0.5) is part of the general reversal.
- **J0's temporal superiority** is not observable at 0.4 s. Near-folded is the one regime where J0 has the lowest residual.

## Correction behaviour (R0 / R1, relative to the frozen J0 anchor)

| | 3DHP R0 | 3DHP R1 | 3DPW R0 | 3DPW R1 |
|---|---:|---:|---:|---:|
| mean upper / forearm correction rotation ° | 7.3 / 15.7 | 17.2 / 19.1 | 5.7 / 12.4 | 13.4 / 14.3 |
| mean wrist displacement %L | 7.2 | 18.8 | 4.8 | 14.3 |
| alignment of wrist displacement with the needed correction (mean cos) | 0.23 | 0.33 | 0.20 | 0.34 |
| wrist error change vs anchor %L | −1.00 | −2.40 | −0.75 | −2.44 |
| share of rows where the correction reduces wrist error | 59% | 55% | 58% | 56% |
| J0 worst quartile: wrist change %L (share reduced) | −3.4 (73%) | **−12.6 (72%)** | −2.4 (67%) | **−12.8 (78%)** |
| other rows: wrist change %L | −0.19 | **+1.00** | −0.21 | **+1.03** |
| adjacent correction change, upper / forearm ° | 6.5 / 10.8 (0.4 s) | 16.6 / 18.2 (0.4 s) | 2.5 / 3.9 | 9.7 / 9.9 |
| excess upper residual vs J0 °/pair | −0.25 | −0.76 | +0.23 | +2.51 |

- **Useful visual correction:** a large, domain-stable repair where J0 is badly wrong. In the J0 worst quartile R1 reduces wrist error by −12.6 %L, the same as the −12.8 %L on 3DPW.
- **Errors introduced by the correction:** +1.0 %L of wrist error on the rows J0 already handles. This is also the same as on 3DPW, and includes the near-plane false-depth failures.
- **Errors already present in J0:** the anchor is far worse on 3DHP (wrist 29.6 vs 22.0 %L on 3DPW), and the correction cannot make up the difference to J1.
- **Correction motion:**
  - its frame-to-frame change correlates with GT motion (Spearman 0.44);
  - at 0.4 s, R1's excess residual over J0 is negative, because the interval is long enough that the static gain dominates;
  - at 0.16 s outdoor, the excess is positive (+1.07°).

## Real-scene review (canonical 3D; no Blender or rig claim)

Rendered by `scripts/review_framepose_anchored_arm_transfer.py`. The case rules are fixed in its docstring and applied to the evaluation rows and pairs.

**Panel layout:**
- the exact source image, with the detector left shoulder/elbow/wrist (the runtime input) drawn in white;
- a front view (X right × Z up) and a side view (depth × up) of the unit arm for GT (green), J0 (orange), J1 (magenta), R0 (blue) and R1 (cyan);
- per-candidate errors.

**Location:**
- server: `~/animcv-output/framepose_anchored_arm_transfer_review/`;
- local QE copy (images only): `~/animcv-output/83_framepose_anchored_arm_transfer/`.

| file | case |
|---|---|
| `typical_static_improvement_vs_j1.png` | median of the rows where R1 beats J1 |
| `typical_static_regression_vs_j1.png` | median of the rows where R1 loses to J1 |
| `large_out_of_plane_correction.png` | TS3 row 130: a 103° R1 correction flips the arm upward; R1 wrist 81.7 %L against J0 55.5 and J1 13.0 (a correction-induced failure) |
| `near_plane_failure.png` | near-plane row where R1 adds false depth relative to J0 |
| `moderate_bend_example.png` | median R1 elbow error in [15°, 45°) |
| `largest_r1_temporal_improvement_vs_j1.png` | 4-frame strip |
| `largest_r1_temporal_regression_vs_j1.png` | 4-frame strip |
| `stable_sequence_TS1.png` | TS1 rows 3100–3130 |
| `unstable_sequence_TS6.png` | TS6 rows 301–361 |

## Architecture classification — CASE C (no reliable external advantage)

| predeclared condition | result |
|---|---|
| static_vs_j1: R1 − J1 wrist < 0 and forearm < 0, both significant; elbow not significantly worse | **false**: wrist +2.42 and elbow +1.35 are significantly worse |
| temporal_vs_j1: R1 − J1 upper residual < 0 significant; forearm residual not significantly worse | **false**: +0.26 n.s.; forearm +1.66 significantly worse |
| visual_gain: (R1 − R0 out-of-plane wrist or forearm) and (R1 − J0 out-of-plane wrist or forearm) | **true**: −4.88 and −7.62 |
| distributed: ≥ 4/6 sequences and no single sequence ≥ 50% | **false** (R1 does not improve on J1) |
| validity | **true** |
| R1 materially less stable than J0 at 0.4 s | false (not significant) |
| R0 useful | true (wrist −1.00, out-of-plane −2.73) |

**What failed.** R1's static and temporal advantages over J1 both fail on the independent source. The static order reverses broadly across subjects, scenes and regimes; it is not concentrated in one sequence.

**Domain evidence.** All four candidates degrade by roughly 1.5×. The 2D-only J0 anchor degrades most, and J1's direct RGB conditioning degrades least. The shift sources visible in the data:
- an elevated studio camera at 2048² resolution;
- green-screen and outdoor scenes;
- floor, exercise and sports activities;
- faster arm motion;
- more out-of-plane and near-straight arms.

**Errors the correction introduces.** Yes, at the same rate as on 3DPW: +1.0 %L where J0 is fine, near-plane false depth, and large-angle flips (TS3 row 130).

**What does transfer.** The useful repair of gross J0 errors (−12.6 %L in J0's worst quartile, matching 3DPW) and the RGB-specific out-of-plane gain over R0 and J0.

**Interpretation.** Anchoring to a 2D-only arm stabilizes R1 in-domain, but it also ties R1 to an anchor whose depth prior does not generalize. Out of domain, direct RGB conditioning (J1) is the more robust of the evaluated arms. This is a statement about the four frozen candidates on one independent source with 6 clusters. It is not a rejection of anchored-correction architectures in general.

**What this worklog does not do:**
- no retraining on 3DHP and no second dataset;
- no smoothing, gating or J0/R1 selection;
- no production change;
- no automatic next step.

Production readiness is not claimed for any candidate.

## Focused tests

`tests/test_framepose_anchored_arm_transfer.py` (8 tests) covers:
- the left-arm mapping, checked by joint name;
- the SMPL-X association index table;
- camera conversion (axes, units, root);
- population and pair intervals, and image naming;
- pairs that never cross a missing partner, and GT amplitude;
- canonical static metrics (rotations, bend-side swing, degeneracy);
- Worklog 77 motion-residual invariants (constant offset, frozen candidate);
- deterministic clustered contrasts and decision helpers;
- detector-owned inputs (no `annot2`/`annot3` on input lines);
- gate order (reproduction before any 3DHP load);
- no training, no smoothing, no production imports.

Results: `pytest tests/test_framepose_*.py` gives **189 passed, 14 skipped** locally (the skips need torch), and the full set in `animcv-framepose:cuda118` on LabServer63 gives **203 passed** at the final commit, including the new file's 8/8.

## Artifact identities (LabServer63 `~/animcv-output/framepose_anchored_arm_transfer/`)

| artifact | sha256 |
|---|---|
| `bank/mpi3dhp_test.{json,npz}` (content digest) | `7260782e49a037fe9a583de8ef2917ce324686f3d19543d9a7bda95729d21aa9` |
| `bank/mpi3dhp_test_build_report.json` | `57cb9ef1…` |
| `qwen/readout.npz` (2,728 samples; determinism max diff 0.0) | `acb5de8f…` |
| `qwen/manifest.json` (cache digest `ace3899e…`, sample-id hash `c6a99702…`) | `6da40e49…` |
| `predictions/predictions_3dhp.npz` | `48a0c845…` |
| `predictions/prediction_report.json` | `c4fdabe7…` |
| `eval/report.json` | `466ba37e…` |
| `eval/rows_3dhp.json` / `eval/pairs_3dhp.json` | `2ed3d930…` / `122da877…` |

Runtimes:
- bank and Qwen: `animcv-external:cu124` (torch 2.4.1+cu124);
- predictions and evaluation: `animcv-framepose:cuda118` (torch 2.1.2+cu118, the Worklog 80/82 image);
- GPU: RTX 3080 Ti.

## Reproduction

    D=$HOME/animcv-data/mpi_inf_3dhp_test/mpi_inf_3dhp_test_set
    docker run --rm --gpus all -w /tmp -e PYTHONPATH=/pylib:/workspace/src:/workspace/scripts \
      -v $HOME/AnimCV:/workspace:ro -v $HOME/animcv-data/pylib_h5py:/pylib:ro -v $D:/data:ro \
      -v $HOME/animcv-output:/output animcv-external:cu124 \
      /workspace/scripts/build_mpi3dhp_test_external_bank.py --root /data --out /output/framepose_anchored_arm_transfer/bank
    docker run --rm --gpus all -w /tmp -e PYTHONPATH=/workspace/src:/workspace/scripts -v $HOME/AnimCV:/workspace:ro \
      -v $HOME/animcv-hf-cache:/hf:ro -v $D:/data:ro -v $HOME/animcv-output:/output animcv-external:cu124 \
      /workspace/scripts/extract_external_qwen3vl_features.py --external-bank /output/framepose_anchored_arm_transfer/bank/mpi3dhp_test \
      --image-root /data --snapshot /hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
      --w67-manifest /output/framepose_vlm_depth_advisor/model_manifest.json --out-dir /output/framepose_anchored_arm_transfer/qwen
    docker run --rm --gpus all -v $HOME/AnimCV:/workspace:ro -v $HOME/animcv-output:/output \
      -e PYTHONPATH=/workspace/src:/workspace/scripts -w /workspace animcv-framepose:cuda118 \
      scripts/predict_framepose_anchored_arm_transfer.py --bank /output/framepose/bank_3dpw_paired_v2.json \
      --cache-dir /output/framepose_qwen3vl_visual_cache --w80-joint-dir /output/framepose_joint_arm_geometry \
      --w82-dir /output/framepose_anchored_arm_correction --external-bank /output/framepose_anchored_arm_transfer/bank/mpi3dhp_test \
      --external-cache /output/framepose_anchored_arm_transfer/qwen --out-dir /output/framepose_anchored_arm_transfer/predictions
    docker run ... animcv-framepose:cuda118 scripts/run_framepose_anchored_arm_transfer_eval.py \
      --external-bank /output/framepose_anchored_arm_transfer/bank/mpi3dhp_test --bank /output/framepose/bank_3dpw_paired_v2.json \
      --predictions-dir /output/framepose_anchored_arm_transfer/predictions --out-dir /output/framepose_anchored_arm_transfer/eval
    docker run ... animcv-external:cu124 /workspace/scripts/review_framepose_anchored_arm_transfer.py \
      --eval-dir /output/framepose_anchored_arm_transfer/eval --bank-json /output/framepose_anchored_arm_transfer/bank/mpi3dhp_test \
      --image-root /data --out-dir /output/framepose_anchored_arm_transfer_review

## Limitations of the cross-domain comparison

- **Six clusters.** One subject per sequence. Every sequence-cluster interval rests on 6 groups, and the outdoor secondary analysis on 2. The direction of the R1 − J1 static reversal is consistent across 5–6 of 6 sequences, but its magnitude is uncertain.
- **Canonical geometry only.** These numbers are not BaseRig IK cm and must not be compared with Worklogs 76–82 tables except through the 3DPW canonical reference computed here.
- **Temporal interval.** The primary pairs span 0.4 s (the official TS1–TS4 valid spacing), against 1/6 s on 3DPW. At 0.4 s, residuals mix static error with jitter, so short-interval stability is only probed on the two outdoor sequences. The TS5–TS6 frame rate (50 fps) is a convention the README does not confirm.
- **Population.** Official valid frames only. The detector association drops 4.7% of valid frames (TS5: 18%).
- **Dataset quirks.** TS3's one-frame image shift and TS2's intrinsics were found and handled by predeclared-before-prediction gates. They are dataset properties and would have to be handled the same way in any reuse of this source.
