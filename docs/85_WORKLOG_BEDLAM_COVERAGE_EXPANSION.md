# Worklog — BEDLAM Coverage Expansion and Controlled J1/R1 Generalization (2026-10-17)

**Question.** If the BEDLAM training coverage of both J1 (direct RGB joint arm) and R1 (geometry-anchored RGB correction) is expanded from the existing on-disk assets, with every other part of the training contract fixed, does R1 recover more of its Worklog 83 cross-domain disadvantage than J1 does?

**Answer.** No. **CASE B: both architectures benefit, and direct RGB remains stronger.**

On the MPI-INF-3DHP official test (developmental, 6 subject clusters):

| effect (3DHP, sequence-cluster 95% interval) | wrist %L | elbow %L | out-of-plane wrist %L | upper-arm residual (0.4 s) |
|---|---|---|---|---|
| ΔJ1 = J1X − J1 | **−1.53 [−2.04, −0.82]**, 5/6 | **−0.99 [−1.36, −0.61]**, 6/6 | **−2.11 [−4.16, −0.16]** | **−1.88 [−2.39, −1.44]**, 6/6 |
| ΔR1 = R1X − R1 | **−1.40 [−3.10, −0.45]**, 6/6 | **−1.32 [−2.36, −0.42]**, 6/6 | **−3.88 [−6.55, −1.84]**, 5/5 | **−1.21 [−2.13, −0.46]**, 6/6 |
| interaction ΔR1 − ΔJ1 | +0.13 [−2.12, +1.49] | −0.32 [−1.67, +0.81] | −1.77 [−6.08, +1.96] | +0.67 [−0.51, +1.79] |
| remaining deficit R1 − J1 → R1X − J1X | +2.42 → **+2.55 [+1.43, +3.32]** | +1.35 → **+1.02 [+0.52, +1.39]** | +6.12 → **+4.35 [+3.25, +5.75]** | +0.26 → **+0.94 [+0.71, +1.17]** |

%L = percent of the GT total arm length. Bold = the interval excludes 0.

- **Coverage helps both architectures.** Each improves significantly and broadly on 3DHP: wrist, elbow, upper-arm and chain direction, and the temporal residuals.
- **R1 does not gain more than J1.** The interaction is not significant on any metric. R1's wrist deficit to J1 does not narrow (+2.42 → +2.55 %L) and remains significant in 5/6 sequences. Its elbow and out-of-plane deficits narrow but stay significant. Its 0.4 s temporal deficit grows and becomes significant (0/6 sequences favour R1X).
- **R1's gain is concentrated.** 59% of R1's 3DHP wrist gain comes from one outdoor sequence, TS6, where its deficit to J1 shrinks from +7.18 to +1.79 %L. In all four studio sequences J1 gains more, and the deficit widens.
- **The expansion does make R1 less anchor-bound,** without closing the gap. R1X repairs more of J0's large errors: −16.8 vs −12.6 %L in J0's worst-wrist quartile (interval [−7.29, −1.15]). It damages accurate anchor rows less (+0.54 vs +1.00 %L) and fixes more out-of-plane depth signs (93.9% vs 89.4%; anchor 87.4%). The near-plane false-depth penalty against J0 is unchanged (+2.46 vs +2.30 %L).
- **On 3DPW test (historical comparison only), the expansion changes neither model's static accuracy significantly.** R1X still beats J1X temporally (−0.53°/pair [−0.84, −0.22]). R1X remains materially less stable than J0 (+2.39°/pair, unchanged).

Under this fixed-budget BEDLAM coverage intervention, the frozen-anchor limitation persists. Greater pose and depth coverage inside BEDLAM narrows R1's out-of-plane and elbow deficits. It does not remove its wrist deficit on the 3DHP studio scenes. These are developmental populations, not an independent final test.

## Directive and decision rules

- **The directive was truncated.** The pasted Worklog 85 text ends inside §15 ("For R1_ORIGINAL and"), so §15's remainder, the decision logic, the DO-NOT list, the tests and the report filename were missing.
- **How the decision rules were set.** The anchor-dependence diagnostics follow the Worklog 84 §13 list. The CASE A–D rules follow the Worklog 84 §17 structure, adapted to the architecture × coverage interaction.
- **They were committed before any expanded model existed:** `DECISION` in `scripts/run_framepose_bedlam_expanded_arms_eval.py`, commit `14aa574`.
- **Report filename:** chosen by this worklog.

## Starting state and files

- **Start:** `arch/single_frame_first` at `08f49c2` (Worklog 84). Local, origin and LabServer63 were equal, and the local worktree was clean. The server's pre-existing untracked `.DS_Store`, `.animcv_sync_stage/` and `docker/` predate this batch.
- **Commits:**
  - `14aa574`: audit, selection, bank builder, training wrapper, prediction and evaluation runners, and the predeclared rules, all before expanded training;
  - `6afd666`: the builder fix (label arrays materialized once, outputs unchanged) and the review renderer;
  - the commit adding this worklog.
- **Added (diagnostic and evaluation only; no `src/` change):**
  - `scripts/audit_bedlam_training_coverage.py`
  - `scripts/select_bedlam_coverage_expansion.py`
  - `scripts/build_bedlam_expanded_bank.py`
  - `scripts/train_framepose_bedlam_expanded_arms.py`
  - `scripts/predict_framepose_bedlam_expanded_arms.py`
  - `scripts/run_framepose_bedlam_expanded_arms_eval.py`
  - `scripts/review_framepose_bedlam_expanded_arms.py`
  - `tests/test_framepose_bedlam_coverage_expansion.py`
  - this worklog
- **Reused unchanged:**
  - Worklog 73: `process_member`, `finish_scene`, `associate`, `camera_joints`, `SceneBodyIndex`, the OpenPose COCO-18 detector and the subject split;
  - the Worklog 68/73 Qwen extractor;
  - Worklog 80 `train_joint` and Worklog 82 `train_correction`;
  - Worklog 83 canonical metrics, the 3DHP bank and its cache;
  - Worklog 75–77 statistics.
- **Unchanged:** every historical checkpoint, bank, cache and report; J0 (frozen); production FramePose/FK/IK/Blender; H0, E1, Root Orientation and BaseRig.

## 1. BEDLAM training coverage audit (every row of the 26 training scenes)

`~/animcv-output/framepose_bedlam_coverage/audit/`: `rows.npz` sha256 `4ef83bad…`, `audit_report.json` `36bd9c24…`.

| quantity | value |
|---|---:|
| label rows (26 Worklog 73 training scenes) | 951,066 |
| rows whose image is in the on-disk tar | 951,066 (no tar truncated; the `closeup_suburb_d` tar that failed in Worklog 73 is now complete) |
| unique images / sequences | 286,228 / 9,450 |
| person recovered (`SceneBodyIndex`) | 949,571 (1,495 ambiguous or too far) |
| rows of BEDLAM **validation** subjects | 158,982: never eligible |
| valid GT left arm / left arm inside the image | 951,051 / 934,564 |
| SMPL-X reprojection onto `gtkps` | max 0.007 px |
| historically selected by Worklog 73 / in the final TRAIN bank | 28,348 / 21,232 |
| persons / subjects (training subjects) | 271 / 198 (159) |

- **Viewpoint is reported only by scene category** (closeup, zoom, orbit, static, pitchUp52/pitchDown52, plain).
- **The camera-pitch descriptor failed validation and is not used.** A camera pitch computed from `cam_ext` (world +Y up) gave +14.8° for `pitchDown52` and −9.6° for `pitchUp52`. Neither the sign nor the magnitude matches the scene names, so the `cam_ext` convention is not verified. The descriptor drove no selection.

## 2. Train/validation identity isolation

- **Expanded training rows:** only training subjects (sha256(subject) mod 5 ≠ 0, the Worklog 73 rule), from training scenes, never selected by Worklog 73. Rows Worklog 73 selected and then dropped stay excluded.
- **Checks before training,** all equal to 0:
  - subject overlap with BEDLAM validation;
  - sequence overlap;
  - image overlap;
  - validation-subject rows in TRAIN;
  - new rows from validation scenes.
- **Other splits:** 3DPW splits unchanged; MPI-INF-3DHP TEST evaluation-only.
- **Residual uncertainty.** 1,495 rows with unrecoverable identity are excluded, not assumed independent. The subject split shares the Worklog 73 limitation: BEDLAM bodies repeat across scenes, so independence holds at the subject level, not the scene level.

## 3. Coverage: historical bank vs eligible pool vs expansion

The coverage cell is GT bend (4 bins) × |f| (3 bins) × sign(f) × upper-arm elevation (3 bins), 72 cells in all, where f is the depth component of the shoulder→wrist direction.

| group | rows | cells with ≥ 100 rows | entropy (bits) | near-straight (bend < 15°) | abs(f) ≥ 0.5 | f < 0 | tracks (scene, seq, person) |
|---|---:|---:|---:|---:|---:|---:|---:|
| historical TRAIN | 21,232 | 29 | 4.60 | 113 (0.5%) | 26.4% | 34.8% | 15,474 |
| eligible unused pool | 753,623 | 65 | 4.68 | 3,569 | 26.4% | — | 28,826 |
| after near-duplicate removal | 313,123 | 58 | 4.95 | 1,839 | 32.3% | — | 28,283 |
| selected new | 80,000 | 58 | 5.54 | 1,839 (all available) | 34.5% | 46.5% | 22,210 |
| expanded population (pre-association) | 101,232 | 58 | 5.40 | 1,952 | 32.8% | 44.0% | 25,841 |

- **Sample density:** large increase. 4.3× the eligible BEDLAM rows after association (17,451 → 75,311 GT-valid rows), and tracks go from 1–3 to up to 8+ rows each.
- **Pose and depth coverage:** the 28 cells with fewer than 20 historical rows all reach ≥ 20 rows. Near-straight arms go from 0.5% to 1.9%, and negative-depth arms from 35% to 44%.
- **3DHP-relevant sparse cells are filled.** 22.3% of the 3DHP evaluation rows (574) fell in cells where the historical bank had fewer than 20 rows; after expansion, 0%. The median BEDLAM support per 3DHP-weighted cell rises from 616 to 2,968 rows.
- **The overall distribution does not move toward 3DHP:** distribution overlap Σmin(p, q) is 0.692 → 0.691. Near-straight arms are 20% of 3DHP rows but under 2% of BEDLAM even after expansion; BEDLAM barely contains them.
- **Appearance coverage:** the same 220 persons, 159 subjects and 26 scenes; more frames, viewpoints and camera motions of the same bodies.
- **Within-BEDLAM composition shifts.** Closeup rows drop from 22% to 9.4% of the BEDLAM pool, because sqrt-cell allocation is not scene-stratified. Static, zoom and orbit rows rise.
- **No new dataset family is added.**

## 4. The predeclared expansion policy (`POLICY`, commit 14aa574)

1. Eligible = image in the tar, person recovered, training subject, valid GT left arm, left arm inside the image, and never selected by Worklog 73.
2. Near-duplicates are removed inside each (scene, sequence, person) track, in frame order, seeded by the track's historical rows. A row is kept only if ‖ΔE‖ + ‖ΔW‖ ≥ 0.15 from every kept row, in the camera frame, so camera motion counts as variation. This removed 440,500 near-duplicate rows (58% of eligible).
3. **80,000** new rows are allocated over the 72 cells in proportion to √(deduplicated count), water-filled, and drawn uniformly within each cell with seed 1337.
4. **Budget:** about 3 h of detector and about 3 h of Qwen on one RTX 3080 Ti; about 125 GB of token grids.

- **Determinism:** the manifest (`expansion_manifest.json`, sha256 `bcba5367…`, internal `5e4d7abd…`) is identical when produced locally and on the server.
- **No model output was read by the selection.** No population size was swept.

## 5. Detector, Qwen and bank identity

- **Detector stage.** The unchanged Worklog 73 TRAIN pipeline was applied to the 80,000 manifest rows: SMPL-X camera joints (max reprojection 0.0002–0.007 px), images streamed from the tars (closeups rotated clockwise), the re-run OpenPose COCO-18 detector, GT-2D identity association and the 3DPW conversion.
  - Kept: **69,761** (87%).
  - Dropped: too few correct limb joints 9,681; pelvis invalid 540; no detection 18.
  - Bank content digest `ea54a0d9…`; build 11,674 s for 68,461 unique images.
- **Builder slowdown, fixed with outputs unchanged.** The first build attempt spent most of its time in Worklog 73 `finish_scene`, which reads `z["gtkps"]` per row; `NpzFile` re-reads the whole array on every access. The label arrays are now materialized once.
- **Qwen.** The unchanged Worklog 68 extractor on the new bank: 69,761 samples in 8,404 s (8.3 rows/s), determinism max diff 0.0, readout `9e9d667f…`, manifest `e15b2fd4…`.
- **Reuse.** Historical BEDLAM TRAIN and validation banks and caches are reused by digest (`56f41663…`, `9556799f…`).

## 6. Fixed training control

| | historical | expanded |
|---|---:|---:|
| 3DPW TRAIN eligible rows | 11,334 | 11,334 (identical draws) |
| BEDLAM TRAIN eligible rows (GT-valid) | 17,451 | 75,311 (57,860 new) |
| BEDLAM draws over training (9,000 steps × 256, 50 : 50) | 1,133,400 | 1,133,400 |
| BEDLAM draws per eligible row | 64.9 | 15.0 |

- **Unchanged:** optimizer, schedule, initialization, 200 epochs and selection by lowest 3DPW validation objective.
- **What the fixed budget changes:** each BEDLAM example is visited about 4.3× less often. The 3DPW draw sequence is identical because the sampler draws 3DPW first from the same generator.

## 7. Reproduction gates (all passed)

| gate | result |
|---|---|
| frozen checkpoints, predictions, banks, caches | hashes equal the pinned Worklog 80/82/83 values |
| **training pipeline:** the wrapper retrains J1 and R1 on the HISTORICAL population in the historical row order | **bit-exact** checkpoints (J1 epoch 109, R1 epoch 59; state max diff 0.0; 3DPW outputs 0.0); anchor raw vs Worklog 82 0.0. Run twice, before and inside the expanded training job |
| frozen-J0 anchor audit (Worklog 82 criterion: TRAIN error < 0.5 × validation error is severe) | 3DPW 0.87, BEDLAM-expanded 1.16: not severe. J0 objective: historical BEDLAM rows 0.431, new rows 0.561, BEDLAM validation 0.458 |
| prediction stage: historical outputs vs Worklog 80/82 (3DPW) and Worklog 83 (3DHP) | ≤ 1e-6 |
| connected-arm contract, every candidate, both populations | ≤ 4.4e-16, all finite, identical coverage |

Because the training pipeline reproduces the historical checkpoints bit for bit, every J1X − J1 and R1X − R1 difference is caused by the BEDLAM TRAIN population alone.

## 8. Training and validation

| candidate | selected epoch | 3DPW validation objective | BEDLAM validation objective | train objective (at selection) |
|---|---:|---:|---:|---:|
| J1 (historical) | 109 | 0.2747 | 0.3992 | 0.1848 |
| **J1_BEDLAM_EXPANDED** | 129 | **0.2585** | **0.3790** | 0.3403 |
| R1 (historical) | 59 | 0.2729 | 0.4017 | 0.2279 |
| **R1_BEDLAM_EXPANDED** | 89 | **0.2713** | **0.3880** | 0.3740 |

Objective = normalized elbow + wrist error.

- **Validation improves for both models on both sources;** J1 improves more on both.
- **The train objective roughly doubles.** That is expected: the BEDLAM TRAIN pool is now dominated by harder, less-repeated rows, and the frozen J0 is itself 30% worse on the new rows (0.561 vs 0.431).
- **Over-fitting gap:** the train/validation gap that marked historical over-fitting (train 0.18 vs validation 0.27) is gone.
- **Curves:** evaluated every 10 epochs; stored in `training/training_report.json` (`48b0f56c…`).
- **Checkpoints:** J1_BEDLAM_EXPANDED `82fd6bcb…`, R1_BEDLAM_EXPANDED `98b8af6c…`.

## 9. Static geometry

**MPI-INF-3DHP (2,570 rows; mean / p50 / p95 / max):**

| metric | J0 | J1 | J1X | R1 | R1X |
|---|---|---|---|---|---|
| wrist %L | 29.6 / 22.8 / 71.5 / 164 | 24.8 / 20.3 / 58.5 / 170 | **23.2** / 18.6 / 56.5 / 183 | 27.2 / 21.3 / 67.1 / 181 | 25.8 / 20.5 / 62.9 / 167 |
| elbow %L | 22.4 / 17.8 / 53.1 / 107 | 19.7 / 17.1 / 42.1 / 96 | **18.7** / 16.2 / 40.2 / 100 | 21.1 / 17.6 / 45.9 / 106 | 19.8 / 16.4 / 42.6 / 103 |
| upper-arm ° | 23.1 | 19.9 | **18.8** | 21.6 | 20.0 |
| forearm ° | 25.1 | 23.4 | **21.5** | 25.3 | 24.4 |
| chain ° | 19.7 | 16.1 | **15.4** | 18.2 | 17.0 |
| reach %L | 7.9 | 6.5 | **6.0** | 6.8 | 6.6 |
| bend side ° | 33.6 | 38.0 | **33.2** | 40.5 | 36.2 |
| bend angle ° | 20.9 | 17.7 | **16.5** | 18.0 | 17.8 |

- **3DHP contrasts.** Forearm direction is the one place R1 gains less than J1: ΔJ1 −1.95° [−2.54, −1.20] vs ΔR1 −0.95° [−2.48, +0.47], so R1X − J1X forearm widens to **+2.93° [+1.77, +3.71]**.
- **Bend side improves for every RGB candidate.** J1X matches J0 (33.2 vs 33.6°), the first RGB arm to do so.
- **3DPW test (6,003 rows, 37 sequences).**
  - No significant static change for either model: ΔJ1 wrist −0.32 [−1.09, +0.48], ΔR1 wrist +0.13 [−0.58, +0.87].
  - The exception is J1's out-of-plane wrist, −2.26 [−3.82, −0.90], and forearm, −0.76° [−1.39, −0.20].
  - R1X − J1X wrist is −0.51 [−1.33, +0.31]; the historical R1 − J1 was −0.96, significant. On 3DPW, R1's static edge over J1 shrinks to non-significance.
- **Rig space:** not reported. Canonical metrics only; no faithful H0 heading exists for 3DHP. The 3DPW canonical numbers are not interchangeable with Worklog 76–82 cm.

## 10. Temporal results

| residual vs GT motion | J0 | J1 | J1X | R1 | R1X |
|---|---:|---:|---:|---:|---:|
| 3DHP 0.4 s, upper-arm °/pair | 20.0 | 19.0 | **17.1** | 19.3 | 18.1 |
| 3DHP 0.4 s, wrist motion %L | 29.3 | 27.6 | **25.8** | 28.4 | 26.9 |
| 3DHP TS5–TS6 0.16 s, upper-arm ° (descriptive, 2 clusters) | 19.3 | 17.7 | **16.4** | 20.3 | 18.5 |
| 3DPW 1/6 s, upper-arm °/pair | **6.46** | 9.91 | 9.38 | 8.98 | 8.85 |
| 3DPW 1/6 s, wrist motion %L | **10.2** | 14.7 | 13.9 | 13.3 | 13.4 |

- **3DHP, 0.4 s.**
  - Both models improve: ΔJ1 −1.88° [−2.39, −1.44], ΔR1 −1.21° [−2.13, −0.46].
  - The R1X − J1X upper residual is now **+0.94° [+0.71, +1.17]**, with 0/6 sequences favouring R1X.
  - At 0.4 s the residual largely tracks static accuracy, so this is not a short-interval jitter measurement.
- **3DHP, 0.16 s outdoor (descriptive only).**
  - R1X − R1 −1.86°; J1X − J1 −1.24°; R1X − J1X +2.05° (0/2 sequences).
  - R1's correction still changes 16.2°/17.8° per 0.16 s on the upper arm and forearm (historical 16.0°/18.5°). The coverage expansion did not reduce correction jitter.
- **3DPW.**
  - J1X is steadier than J1 (−0.53°/pair [−0.78, −0.26], 28/37).
  - R1X is unchanged against R1 (−0.13 [−0.40, +0.14]).
  - R1X stays steadier than J1X (−0.53 [−0.84, −0.22]) and much less stable than J0 (+2.39 [+2.09, +2.67], 1/37). The R1-vs-J0 temporal penalty is unchanged.
- **Amplitude ratio to GT** (upper-arm change):
  - 3DHP 0.4 s: J0 0.91, J1 1.03, J1X 1.00, R1 0.99, R1X 0.99;
  - 3DPW: J0 0.99, J1 1.58, J1X 1.51, R1 1.45, R1X 1.45.

  The candidates' motion is not damped. On 3DPW the RGB models over-move against GT.

## 11. Anchor-dependence analysis (GT categories: analysis only)

| R1 candidate (3DHP) | R1 | R1X | R1X − R1 (cluster CI) |
|---|---:|---:|---|
| mean correction rotation, upper / forearm ° | 17.2 / 19.1 | 16.8 / 18.6 | — |
| alignment of the wrist displacement with the needed GT displacement (mean cos) | 0.33 | 0.40 | — |
| wrist error change vs anchor %L | −2.40 | −3.80 | — |
| **recovery in J0's worst-wrist quartile (J0 ≥ 40.9 %L)** | −12.6 (72% of rows reduced) | **−16.8** (82%) | **−4.23 [−7.29, −1.15]**, 5/6 |
| damage on the other 3/4 of rows | +1.00 (51% worsened) | +0.54 (49%) | −0.45 [−1.08, +0.20] |
| near-plane false depth (wrist vs J0) | +2.30 | +2.46 | +0.16 [−1.01, +1.11] |
| out-of-plane depth-sign correct (anchor 87.4%) | 89.4% | 93.9% | out-of-plane wrist −3.88 [−6.55, −1.84] |
| Spearman(J0 wrist error, final wrist error) | 0.61 | 0.56 | — |
| adjacent correction change, upper / forearm ° (0.4 s) | 16.6 / 18.2 | 17.2 / 18.0 | — |

- **Expanded supervision makes R1 measurably less sensitive to J0's domain-shifted errors.** It recovers more, fixes more depth signs, damages less and depends less on anchor error rank.
- **It still ends behind J1X, which improves as much on the same data.** The anchor limitation is reduced but persists.
- **The correction's temporal variation and its near-plane false-depth behaviour are unchanged.**
- **3DPW (historical domain):**
  - recovery −12.8 → −13.9 %L;
  - damage on accurate anchor rows +1.03 → +1.55 %L;
  - near-plane false depth +1.52 → +2.13 %L.

  On the source domain, the expanded correction intervenes slightly more aggressively.

## 12. Regimes and Worklog 83 failure modes (3DHP, wrist %L)

| regime | rows | J1 → J1X | R1 → R1X | deficit R1 − J1 → R1X − J1X | interaction |
|---|---:|---|---|---|---:|
| near-straight [0, 15) | 525 | 30.1 → 28.2 | 35.5 → 32.2 | +5.46 → +4.02 | −1.44 |
| moderate [15, 45) | 942 | 22.6 → 21.8 | 24.5 → 23.6 | +1.90 → +1.79 | −0.12 |
| strong [45, 90) | 718 | 23.4 → 21.4 | 25.0 → 24.1 | +1.57 → +2.75 | +1.17 |
| near-folded [90, 180] | 385 | 25.4 → 23.4 | 26.5 → 25.4 | +1.12 → +2.03 | +0.90 |
| near-plane, abs(f) < sin 10° | 890 | 17.4 → 16.7 | 18.1 → 18.2 | +0.66 → +1.52 | +0.86 |
| intermediate | 924 | 23.0 → 21.1 | 24.1 → 23.2 | +1.09 → +2.07 | +0.97 |
| out-of-plane, abs(f) ≥ 0.5 | 756 | 35.6 → 33.5 | 41.7 → 37.8 | +6.12 → +4.35 | −1.77 |

How the Worklog 83 failure modes fared:

- **Large anchor errors and out-of-plane cases where J1 beat R1:** R1 gains more there, and the deficit narrows but stays large.
- **Near-plane false depth:** unchanged. R1X is no better near-plane (+0.16), while J1X improves.
- **Strong and folded bends:** J1 gains more, and the deficit widens.
- **Fast outdoor motion (TS6):** the largest R1 gain (−5.64 %L, deficit +7.18 → +1.79) and the largest temporal gain (−3.57°).
- **Static accuracy vs short-interval jitter:** static accuracy improved, and correction jitter did not change.

**Per sequence (3DHP wrist, deficit R1 − J1 → R1X − J1X):**

| sequence | deficit change |
|---|---|
| TS1 | +1.81 → +2.71 |
| TS2 | +0.41 → +2.12 |
| TS3 | +3.59 → +4.15 |
| TS4 | +1.50 → +3.45 |
| TS5 | −0.63 → −1.59 |
| TS6 | +7.18 → +1.79 |

- **Concentration:** ΔR1 improves in 6/6 sequences, but TS6 supplies 59% of the net gain. ΔJ1's largest contributor is TS4 (33%).
- **Severe tails:** the R1X wrist maximum is 167 %L (R1 181); the J1X maximum is 183 (J1 170).

## 13. Real-scene review (canonical 3D; no Blender / IK claim)

**Panel layout:**
- the exact source image, with the detector left arm (runtime input) in white;
- a front view (X × Z) and a side view (depth × up);
- arms drawn: GT green, J0 orange, J1 magenta, J1X purple dashed, R1 cyan, R1X blue dashed;
- per-candidate errors and correction magnitudes in each caption.

**Location:**
- server: `~/animcv-output/framepose_bedlam_coverage_review/`;
- local QE copy (images only): `~/animcv-output/85_framepose_bedlam_coverage/`.

| file | case (fixed rule) |
|---|---|
| `successful_transfer_r1x_vs_r1.png` | TS6 row 415: R1 wrist 45.9 → R1X 37.2 %L |
| `failed_transfer_r1x_vs_r1.png` | TS5 row 180 (median of rows where R1X is worse) |
| `large_frozen_anchor_error.png` | TS3 row 4300 (median R1X inside J0's worst quartile) |
| `successful_rgb_correction.png` | TS4 row 570 (largest R1X reduction vs the anchor) |
| `correction_induced_false_depth.png` | TS1 row 590 (near-plane, R1X worse than J0) |
| `out_of_plane_geometry.png` | TS4 row 5490 |
| `moderate_bend_error.png` | TS2 row 4880 |
| `largest_temporal_regression_r1x_vs_r1.png` | TS4 rows 5490–5520 strip |
| `largest_temporal_improvement_r1x_vs_r1.png` | TS4 rows 2660–2690 strip |

## 14. Architecture classification — CASE B

| predeclared condition (3DHP primary) | result |
|---|---|
| r1_gain: ΔR1 wrist < 0 holds | **true**: −1.40 [−3.10, −0.45] |
| j1_gain: ΔJ1 wrist < 0 holds | **true**: −1.53 [−2.04, −0.82] |
| r1_gains_more: interaction wrist < 0 holds | **false**: +0.13 [−2.12, +1.49] |
| deficit_narrows: R1X − J1X wrist not significant, or ≤ 50% of the historical deficit | **false**: +2.55 [+1.43, +3.32] vs +2.42 |
| temporal_ok: ΔR1 upper residual not significantly > 0 (3DHP and 3DPW) | true |
| distributed: ΔR1 wrist ≥ 4/6 sequences and no single sequence ≥ 50% | **false**: 6/6, but TS6 = 59% |
| validity | true |

**Conclusion.** Additional BEDLAM coverage helps both architectures, and direct RGB remains stronger. The frozen-anchor dependency remains a plausible architectural limitation. It is not proven to be the sole cause:
- the intervention added pose, depth and viewpoint density, not new subjects, scenes or dataset families;
- 3DHP has only 6 clusters;
- the gain pattern is mixed: R1 gains more where J0 fails badly (outdoor, out-of-plane, near-straight), J1 gains more elsewhere.

Not done, per the stop rules:
- no retraining of J0, no correction-magnitude tuning, no fusion or gating;
- no second dataset, no production change;
- no automatic next step.

## 15. Focused tests

`tests/test_framepose_bedlam_coverage_expansion.py` (5 tests) covers:
- coverage cells and their bin edges;
- track-local near-duplicate removal seeded by history;
- sqrt allocation with water-filling and no oversampling;
- eligibility excluding validation subjects, history, non-RGB and out-of-image arms;
- builder reuse of the Worklog 73 pipeline;
- the training wrapper calling the historical `train_joint` and `train_correction`, with gate order and leakage STOPs;
- no 3DHP or test-split use in training;
- no `src/` coupling.

Results:
- `pytest tests/test_framepose_*.py`: **198 passed, 14 skipped** locally (the skips need torch);
- LabServer63, `animcv-framepose:cuda118`: **212 passed**.

## 16. Artifact identities (LabServer63 `~/animcv-output/framepose_bedlam_coverage/`)

| artifact | sha256 |
|---|---|
| `audit/rows.npz`, `audit/audit_report.json` | `4ef83bad…`, `36bd9c24…` |
| `selection/expansion_manifest.json` | `bcba5367…` |
| `expansion_bank/bedlam_expansion_new` (content digest) / build report | `ea54a0d9…` / `c1898038…` |
| `qwen_expansion/manifest.json` / readout | `e15b2fd4…` / `9e9d667f…` |
| `training/training_report.json` | `48b0f56c…` |
| `J1_BEDLAM_EXPANDED.pt` / `R1_BEDLAM_EXPANDED.pt` | `82fd6bcb…` / `98b8af6c…` |
| `predictions/prediction_report.json`; `predictions_3dhp.npz`; `predictions_3dpw.npz` | `e4ddc99a…`; `c70f3eb2…`; `81eae1cb…` |
| `eval/report.json` | `3132b9d3…` |

Runtimes:
- audit, bank, Qwen and review: `animcv-external:cu124`;
- training, prediction and evaluation: `animcv-framepose:cuda118` (the Worklog 80/82 image);
- GPU: RTX 3080 Ti.

## Generalization limitations

- **Developmental evaluation only.** 3DPW test and MPI-INF-3DHP test have both informed architecture decisions, so this is controlled developmental transfer, not independent final validation. No unseen evaluation partition exists on the server (Worklog 84).
- **Six clusters.** 3DHP's subject-cluster intervals rest on 6 groups, and the outdoor 0.16 s analysis on 2.
- **One fixed allocation.** One expansion size and one policy were tested. The outcome is specific to sqrt-cell allocation with fixed-budget training, where each BEDLAM row is visited 15× instead of 65×.
- **Coverage is within BEDLAM.** BEDLAM itself barely contains near-straight arms, which are 20% of 3DHP.
- **The pitch descriptor is unverified,** so camera elevation could not be audited.
