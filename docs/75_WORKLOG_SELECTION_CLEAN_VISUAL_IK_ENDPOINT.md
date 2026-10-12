# Worklog — Selection-Clean Test-Set Visual IK Endpoint Attribution (2026-10-12)

**Question.** On the untouched 3DPW test population, does BEDLAM-supervised E1 improve actual chain-endpoint geometry over frozen H0, or only over the weaker E0 visual-depth baseline?

**Answer.** Yes, it improves over both. **CASE A.**

The Worklog 61/62 endpoint contract is evaluated analytically. It is first proven identical to Worklog 74's solved IK on all 395 Worklog 74 rows, to 5e-14 cm.

**Test results (6,003 rows, 37 sequences):**

| comparison | endpoint change | rows improved | sequences improved | 95% interval |
|---|---|---|---|---|
| T2 − T0 (E1 vs H0) | −2.99 cm (14.88 → 11.89) | 60% | 32/37 | excludes 0 under every predeclared procedure |
| T2 − T1 (E1 vs E0) | −2.21 cm | 62% | 33/37 | excludes 0 under every predeclared procedure |

- **Endpoint p95** improves from 37.08 to 29.96 cm.
- **Worklog 74's validation result was a population effect.** Its two clean sequences are among H0's best validation sequences (dancing actor0 3rd and hug actor1 6th of 16 by H0 chain depth error). On the other 13 validation sequences E1 also beats H0, by −2.11 cm with 12/13 sequences improved.
- **This is endpoint-only evidence.** No elbow or full-pose quality is claimed.

## Starting state and files

- **Start:** `arch/single_frame_first` at `7749bef49c06c3df39c05137f7d9eb3a9f3eebcb` (local = remote), with a clean worktree.
- **Commits:**
  - `11b4966`: evaluation module, runner, tests
  - `07dc71d`: review panels
  - the commit adding this worklog
- **Files added (evaluation only):**
  - `src/retarget/framepose_visual_ik_endpoint.py`
  - `scripts/run_framepose_visual_ik_endpoint_test.py`
  - `scripts/render_framepose_visual_ik_endpoint_review.py`
  - `tests/test_framepose_visual_ik_endpoint.py`
  - this worklog
- **Unchanged:** every historical file, checkpoint and artifact, and production FramePose, FK, IK, Root Orientation and the Blender writer. Worklogs 61/62/73/74 results stay as recorded.

## Artifact provenance and the H0 test-geometry gate (Gate 1)

**Frozen H0 test predictions exist.** `framepose/context_h0_v2/prediction_test.npy` has shape (7076, 17, 3), produced by the documented materialization:
- checkpoint `sign_attr_v2/O_BILATERAL/checkpoint.pt`;
- `sign_source: oracle`, with sign fields `shoulder_forward_depth` and `hip_forward_depth`;
- `bank_content_digest` `75519e63…` (checked equal to the FrameBank used here);
- validation predictions `88a007fc…` (pinned, as in Worklogs 52–74).

So no H0 inference, retraining or sign-contract change was needed, and Gate 1 passes (not CASE D).

**How H0 is assembled.** H0 test geometry goes through the production `assemble_h0`, then `build_h0_lifted_sequence`. That gives Worklog 61's `observation_valid` (detector `input_valid` AND finite H0) and the current-frame yaw from `current_root_orientation`, exactly as AnimationSemantics v2 builds them.

**Identity checks:**
- **Pinned, checked at runtime:** Worklog 73 `report.json` (`1c05d50a…`) and `rows_heldout.json` (`4e17bd77…`), and Worklog 74 `report.json` (`8158e8b2…`).
- **Exact Worklog 73 row identity:** every Worklog 73 row must match the FrameBank `sample_id` and position. Its `chain_f_h0` must equal the H0 geometry (≤ 1e-6) and its `chain_f_gt` the 3DPW GT (≤ 1e-5). Duplicates are refused.
- **GT:** 3DPW `jointPositions` via `_matched_oracle_sequence` (camera frame, `campose_valid`), as in Worklogs 62/74.
- **Target:** the Worklog 61 BaseRig chain, L1 = 27.771168, L2 = 27.251084, L = 55.022252 armature units (cm).

## Analytic endpoint contract

With every candidate keeping the frozen H0 reach fraction r_H0:

    error = | clamp(r_H0 L) · d_c − r_GT L · d_GT |
          = L · sqrt((r_H0 − r_GT)² + 2 r_H0 r_GT (1 − d_c · d_GT))   (unclamped rows)

- **Rotation invariance:** candidate and reference share the rigid runtime transform `A · Rz(−yaw_H0)`, so the distance is evaluated in the camera frame.
- **Shared Worklog 61 definitions:** reach fraction, chain direction and bend angle come from `source_chain_geometry`. The clamp interval and labels are those of `solve_two_bone`.
- **Clamps:** clamped rows would use the clamped distance and be counted separately. No row clamped in any population.
- **Visual directions:** rebuilt with the Worklog 64–73 `reconstruct_direction(f, observed_plane(detector 2D))`.
- **No elbow:** no elbow is reconstructed and no elbow metric is reported.

## Worklog 74 parity (Gate 2, before any test row was evaluated)

The same pipeline (H0 npy, FrameBank plane, 3DPW GT, Worklog 73 f) was run on the exact 395 Worklog 74 rows. It was compared with Worklog 74's persisted solved-IK endpoint and direction errors. The tolerance is the Worklog 61/74 contract limit: 1e-9 cm, and 1e-6° for directions.

| candidate | max \|Δ endpoint\| (cm) | max \|Δ direction\| (°) |
|---|---:|---:|
| T0 H0 vs P0 | 4.6e-14 | 3.2e-13 |
| T1 E0 vs P1 | 2.8e-14 | 1.3e-12 |
| T2 E1 vs P2 | 2.9e-14 | 8.9e-13 |
| T3 GT f vs P3 | 3.3e-14 | 1.2e-11 |
| T4 GT direction vs Worklog 62 D | 5.1e-14 | 0 |

395/395 rows matched, so the analytic metric reproduces the historical IK endpoint metric to machine precision.

## Candidates

| candidate | chain direction | reach | GT use |
|---|---|---|---|
| **T0_H0** | frozen H0 shoulder→wrist | H0 | none directly (H0 inputs include the two oracle torso sign fields) |
| **T1_E0_VISUAL** | E0 chain f + detector plane | H0 | none in the direction |
| **T2_E1_VISUAL** | E1 chain f + the same detector plane | H0 | none in the direction |
| **T3_GT_F_ORACLE** | GT chain f + the same detector plane | H0 | GT f (evaluation oracle) |
| **T4_GT_DIRECTION_REFERENCE** | GT 3D chain direction | H0 | GT direction (reference) |

**Only the direction changes.** Target root, L1/L2, H0 reach, the GT endpoint reference, the population and the validity rules are shared. Upper and forearm predictions are not used, and no skeleton is synthesized.

## Test population and exclusions (Gate 3)

| quantity | rows |
|---|---:|
| original FrameBank test rows | 7,076 (37 actor sequences) |
| rows with a Worklog 73 E0/E1 prediction | 6,003 |
| yaw unknown | 0 |
| GT invalid | 0 |
| degenerate chain or observed plane | 0 |
| **common eligible** | **6,003** (every candidate valid on every row) |

**Exclusion flags (non-exclusive):**
- no Worklog 73 prediction: 1,073
- invalid H0 source joint: shoulder 87, elbow 633, wrist 1,059

All rows with an invalid source joint are among the 1,073 rows that Worklog 73 had already found ineligible. So the sequential exclusion is entirely "no Worklog 73 prediction". Per-sequence coverage ranges from 48% (`downtown_bar_00` actor1) to 100%, and the full table is in `report.json`.

**Selection-clean:** no test row influenced any checkpoint, epoch, architecture, ratio, threshold or sampling decision. E0/E1 and H0 epochs were selected on validation, and no model was trained here.

## Test results (6,003 rows; mean / p50 / p95 / max)

| candidate | chain \|f\| error | chain direction error (°) | endpoint error (cm) | clamps |
|---|---|---|---|---:|
| T0 H0 | .271 / .192 / .810 / 1.728 | 18.21 / 12.81 / 52.78 / 131.29 | 14.88 / 11.46 / 37.08 / 90.60 | 0 |
| T1 E0 | .251 / .192 / .707 / 1.826 | 16.78 / 12.75 / 46.57 / 136.88 | 14.10 / 11.44 / 36.20 / 69.71 | 0 |
| **T2 E1** | **.200 / .152 / .558 / 1.427** | **13.78 / 10.57 / 37.48 / 98.66** | **11.89 / 9.63 / 29.96 / 76.28** | 0 |
| T3 GT f, observed plane | 0 | 3.45 / 2.19 / 10.87 / 63.13 | 4.53 / 3.34 / 12.29 / 44.47 | 0 |
| T4 GT direction, H0 reach | 0 | 0 | 2.98 / 1.60 / 10.31 / 41.23 | 0 |

Coverage is 6,003/6,003 for every candidate.

**Paired endpoint differences (cm):**

| comparison | mean (frame-weighted) | p50 | p95 | rows improved | sequence-weighted mean | sequences improved / regressed |
|---|---:|---:|---:|---:|---:|---|
| **T2 − T1** | **−2.21** | −1.40 | +7.90 | 62% | −2.19 | **33 / 4** |
| **T2 − T0** | **−2.99** | −1.78 | +12.67 | 60% | −2.79 | **32 / 5** |
| T1 − T0 | −0.78 | −0.35 | +15.28 | 52% | −0.60 | 25 / 12 |
| T3 − T2 | −7.36 | −5.01 | −0.07 | 99.5% | −7.17 | 37 / 0 |
| T4 − T3 | −1.54 | −0.78 | −0.01 | 100% | −1.58 | 37 / 0 |

**Direction differences:** T2 − T1 is −2.99° mean and T2 − T0 is −4.43° mean.

## Sequence-aware uncertainty (predeclared)

The intervals are 95% bootstrap intervals with 2,000 replicates and seed 1337. Two procedures were declared before the run:
- **Sequence cluster:** resample sequences, then report frame- and sequence-weighted means.
- **Moving block within sequences:** block lengths 10 and 25, retained from Worklog 74's sensitivity accounting.

| comparison | sequence cluster, frame-weighted | sequence cluster, sequence-weighted | moving block 10 | moving block 25 |
|---|---|---|---|---|
| **T2 − T1** | [−2.90, −1.59] | [−2.80, −1.62] | [−2.58, −1.99] | [−2.71, −1.94] |
| **T2 − T0** | [−4.76, −1.50] | [−4.18, −1.56] | [−3.71, −2.63] | [−4.10, −2.63] |
| T1 − T0 | [−2.19, +0.55] | [−1.82, +0.56] | [−1.36, −0.39] | [−1.64, −0.45] |

T2's advantage over both T1 and T0 excludes zero under every procedure, including the most conservative sequence-cluster bootstrap. E0's advantage over H0 does not survive the sequence-cluster bootstrap.

**Ownership and regressions:**

| comparison | largest single-sequence contribution | worst regressions (summed cm; mean cm) |
|---|---|---|
| T2 − T0 | `downtown_bar_00` actor0, 26% of the net (H0 chain \|f\| error .766 there vs E1 .311) | `downtown_stairs_00` a0 (+1,081; +5.25), `flat_packBags_00` a0 (+586; +2.33), `downtown_upstairs_00` a0 (+184; +1.34) |
| T2 − T1 | `downtown_bar_00` actor0, 15% of the net | `downtown_cafe_00` a1 (+0.56), `downtown_warmWelcome_00` a0 (+0.90), `downtown_weeklyMarket_00` a0 (+0.16) |

**Without `downtown_bar_00` actor0:**
- T2 − T0 = −2.30 cm (60% of rows improved), and endpoint p95 goes 33.41 → 27.79.
- T2 − T1 = −1.97 cm.

H0's chain depth is better than E1's in only 6 of the 37 test sequences.

**Scene families** (T2 − T0 / T2 − T1, cm):
- downtown (32 sequences, 5,156 rows): −2.95 / −2.31, with 28/32 and 28/32 sequences improved
- flat (2): −2.79 / −1.68
- office (2): −3.35 / −1.14
- outdoors (1): −3.96 / −2.18

## Comparison with the validation results (reported separately, selection-contaminated)

Validation is the split that selected the E0/E1 and H0 epochs.

| population | rows / sequences | T0 | T1 | T2 | T2 − T1 (sequence-cluster CI) | T2 − T0 (sequence-cluster CI) | sequences improved (T2 vs T0) |
|---|---|---:|---:|---:|---|---|---|
| Worklog 74 population (unchanged) | 395 / 3 | 12.30 | 12.28 | 11.72 | −0.56 [−1.66, −0.12] | −0.58 [−3.32, +1.44] | 1 / 3 |
| other validation sequences | 2,661 / 13 | 11.80 | 10.34 | 9.70 | −0.65 [−1.06, −0.31] | −2.11 [−3.27, −1.21] | 12 / 13 |
| all eligible validation | 3,056 / 16 | 11.87 | 10.59 | 9.96 | −0.64 [−1.02, −0.33] | −1.91 [−2.99, −1.04] | 13 / 16 |
| **test (selection-clean)** | **6,003 / 37** | **14.88** | **14.10** | **11.89** | **−2.21 [−2.90, −1.59]** | **−2.99 [−4.76, −1.50]** | **32 / 37** |

- **Does T2 vs T1 reproduce Worklog 73's test-direction gain at the endpoint?** Yes. Worklog 73's test chain gain (\|f\| .251 → .200, direction 16.8° → 13.8°) becomes −2.21 cm at the endpoint, with 33/37 sequences improved. On validation the same comparison is smaller (−0.64 cm), consistent with Worklog 73's smaller validation gain.
- **Does T2 vs T0 change sign relative to Worklog 74?** On the Worklog 74 population it was +0.80 cm without crosscountry and −0.58 pooled. On test it is −2.99 cm, and on the 13 other validation sequences −2.11 cm.
- **Is the difference due to population composition?** Yes. Worklog 74's matched population is the only 3 sequences with historical Worklog 57/61/62 replays, and two of them are unusually favourable to H0:

| validation sequence | H0 chain \|f\| error (rank of 16) | E1 \|f\| error | T2 − T0 (cm) |
|---|---|---:|---:|
| courtyard_dancing_00 actor0 | .134 (3rd best) | .159 | +1.44 |
| courtyard_hug_00 actor1 | .181 (6th) | .189 | +0.53 |
| outdoors_crosscountry_00 actor0 | .284 (worst) | .212 | −3.32 |

  The other 13 validation sequences, and the test split, are not like dancing actor0 and hug actor1. The only other validation sequences where E1 does not beat H0 are hug actor0 (−0.00), drinking actor1 (+0.51) and parcours_00 (−0.24).
- **No headline merges validation and test.** The test numbers above are the selection-clean evidence.

## Depth and pose regimes (test, evaluation-only)

| stratum | n | T0 | T1 | T2 | T3 | T4 | T2 − T1 (rows improved; sequences improved) | T2 − T0 (rows improved; sequences improved) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| \|GT chain f\| near plane (< sin 10°) | 2,117 | 9.66 | 10.03 | 8.01 | 2.76 | 1.63 | −2.02 (64%; 32/37) | −1.65 (62%; 27/37) |
| intermediate | 2,453 | 13.46 | 13.27 | 11.70 | 4.13 | 2.62 | −1.57 (58%; 26/37) | −1.76 (55%; 23/37) |
| strongly out of plane (≥ .5) | 1,433 | 25.01 | 21.52 | 17.94 | 7.81 | 5.61 | −3.59 (67%; 29/36) | **−7.07** (67%; 23/36) |
| H0 \|reach error\| Q1 (< .010) | 1,500 | 10.60 | 10.96 | 9.04 | 2.11 | 0.26 | −1.92 | −1.56 |
| Q2 | 1,501 | 12.68 | 12.33 | 9.85 | 2.68 | 1.02 | −2.48 | −2.83 |
| Q3 | 1,501 | 14.72 | 13.61 | 11.72 | 4.05 | 2.57 | −1.88 | −3.00 |
| Q4 (≥ .072) | 1,501 | 21.50 | 19.49 | 16.94 | 9.27 | 8.07 | −2.56 | −4.56 |
| GT bend 0–15° | 931 | 9.04 | 10.28 | 8.68 | 2.61 | 1.24 | −1.60 | **−0.37 (51%)** |
| 15–45° | 2,716 | 14.97 | 13.67 | 11.57 | 3.77 | 2.15 | −2.10 | −3.40 |
| 45–90° | 1,606 | 16.71 | 16.54 | 13.81 | 5.14 | 3.68 | −2.74 | −2.90 |
| ≥ 90° | 750 | 17.85 | 15.15 | 12.90 | 8.31 | 6.67 | −2.25 | −4.96 |
| min detector arm confidence Q1 (< .716) | 1,500 | 18.38 | 16.77 | 14.96 | 6.57 | 4.61 | −1.81 | −3.42 |
| Q2 | 1,501 | 15.85 | 14.86 | 12.26 | 4.36 | 2.80 | −2.59 | −3.58 |
| Q3 | 1,501 | 12.79 | 12.79 | 10.69 | 3.76 | 2.52 | −2.10 | −2.10 |
| Q4 (≥ .847) | 1,501 | 12.50 | 11.98 | 9.64 | 3.41 | 2.01 | −2.34 | −2.86 |

- **T2 is ahead of both T1 and T0 in every stratum.** The largest gain over H0 is on strongly out-of-plane chains (−7.07 cm); the smallest is on nearly straight GT arms (−0.37 cm, 51% of rows).
- **Near-plane false depth does not increase:** T2 beats T0 on near-plane rows (−1.65 cm, 62%).
- **These strata are descriptive.** No threshold, gate or selector was derived from them.

## Error ladder and attribution (test means, cm; order-dependent)

| | T0 H0 | T1 E0 | T2 E1 | T3 perfect f, observed plane | T4 perfect direction, H0 reach |
|---|---:|---:|---:|---:|---:|
| endpoint mean | 14.88 | 14.10 | 11.89 | 4.53 | 2.98 |

- **Remaining continuous depth error** (T2 → T3): 7.36 cm. This is still the largest share.
- **Observed-plane error** (T3 → T4): 1.54 cm.
- **H0 reach error** (T4 → 0): 2.98 cm when reach is fixed last.
- **Order-dependence:** reach and direction errors combine roughly in quadrature, not additively, so these shares depend on the order in which owners are fixed. They are not independent linear components.

## Real-scene review exports

Seven test rows were chosen by fixed rules from the common-valid population. Each panel shows:
- the FramePose person crop with the detector left shoulder/elbow/wrist;
- unit shoulder→wrist directions (camera frame) for GT, H0, E0 and E1, in a side view (depth × up) and a front view;
- the exact identity, the four chain f values, all five analytic endpoint errors and the direction errors.

These are direction plots. No Blender/IK pose was rendered for this batch.

| rule | row | f GT / H0 / E0 / E1 | endpoint T0 / T1 / T2 (cm) |
|---|---|---|---|
| largest T2 gain vs T0 | downtown_bar_00 a0 #775 | −.86 / +.59 / −.47 / −.84 | 81.02 / 30.88 / 8.32 |
| largest T2 regression vs T0 | downtown_sitOnStairs_00 a0 #325 | +.49 / +.58 / −.88 / −.93 | 4.87 / 63.79 / 67.71 |
| largest T2 gain vs T1 | downtown_runForBus_01 a0 #185 | −.72 / −.16 / +.16 / −.59 | 28.75 / 42.02 / 9.06 |
| typical (median T2 error) | downtown_enterShop_00 a0 #1113 | +.22 / −.00 / +.06 / +.05 | 12.23 / 8.92 / 9.63 |
| strongly out of plane (median T0) | downtown_sitOnStairs_00 a1 #812 | −.52 / −.12 / −.14 / −.44 | 20.67 / 19.47 / 7.36 |
| near-plane false depth (largest \|E1 f\|) | flat_packBags_00 a0 #446 | −.16 / −.14 / −.58 / −.88 | 8.08 / 16.67 / **30.23** |
| lowest detector arm confidence (.05) | downtown_arguing_00 a0 #695 | +.99 / +.30 / +.37 / +.51 | 53.09 / 50.38 / 44.28 |

- **Largest gain (#775):** H0 has the depth sign wrong and E1 gets it right.
- **Largest regression (#325):** E1 (and E0) puts the chain behind the shoulder against GT's forward chain. This is a visual depth-sign failure on a frontal, clearly visible arm.
- **Near-plane row #446:** E1 invents strong depth (f −.88 vs GT −.16) and is 22 cm worse than H0.
- **Lowest-confidence row:** all candidates underestimate a nearly depth-aligned chain.
- **The negative cases are kept.**

Images are in `~/animcv-output/75_framepose_visual_ik_endpoint_test/` (local). Their index is on LabServer63 at `framepose_visual_ik_endpoint_test/review/`.

## Architecture classification

**CASE A: test-set downstream visual advantage.**

| requirement | evidence | met |
|---|---|---|
| T2 materially improves T1 | −2.21 cm (−15.7%), 62% of rows, 33/37 sequences; every CI excludes 0 | yes |
| T2 materially improves T0 | −2.99 cm (−20.1%), 60% of rows, 32/37 sequences; every CI excludes 0 | yes |
| holds on the selection-clean test population | test split only; no test-based selection | yes |
| distributed across sequences | largest share 26%; −2.30 cm without it; 4 scene families all improve | yes |
| endpoint p95 does not regress | 37.08 → 29.96 (33.41 → 27.79 without the largest sequence) | yes |
| coverage acceptable | 6,003/7,076 (all Worklog 73-eligible rows); every candidate valid on all | yes |
| not one scene or oracle-only | real candidates T1/T2 vs T0; oracles only in T3/T4 | yes |

**Answer to the completion question.** Yes. On the selection-clean 3DPW test population, E1's BEDLAM-supervised chain-depth advantage becomes a genuine endpoint advantage over frozen H0 (−2.99 cm, 32/37 sequences), not only over E0.
- **Comparison is tilted toward H0, not E1:** H0 benefits from two oracle torso sign fields and E1 does not.
- **Worklog 74's negative result:** it reflects an H0-favourable three-sequence validation population.

**Limits:**
- **Endpoint only:** the elbow, bend side and full IK pose were not reconstructed on test sequences, so no elbow or pose-quality claim is made. Worklog 74 showed only that the IK carries the supplied chain direction to the endpoint.
- **Regressions remain:** E1 still has depth-sign failures (largest-regression row; `downtown_stairs_00` +5.25 cm mean). Its remaining depth error (7.4 cm to the GT-f oracle) is the largest owner.
- **Single run:** only the seed-1337 Worklog 73 checkpoints were saved.

**Per the decision logic:**
- E1 is justified as a candidate chain-direction evidence provider for a later, bounded FramePose integration test.
- It is not deployed, and no fusion, gate, selector or production change was made.

## Focused tests

`tests/test_framepose_visual_ik_endpoint.py` has 5 tests. They cover:
- **Endpoint parity:** the analytic endpoint equals the Worklog 74 solved IK endpoint, for several yaws and visual directions, including the exact H0 direction rebuilt from its own f and plane. It also equals the DIRECTION's closed form.
- **Worklog 62 reproduction:** T0 and T4 reproduce Worklog 62 `ik_h0` and `gt_direction_h0_reach`.
- **Geometry:** H0 reach and direction derivation (Worklog 61 `source_chain_geometry`), fixed target lengths, and degenerate/NaN chains refused.
- **Clamps:** clamp semantics identical to `solve_two_bone`, with clamped rows using the clamped distance.
- **Reconstruction:** visual direction identical to the Worklog 74 reconstruction.
- **Static contracts:**
  - the parity STOP runs before the test split;
  - identity, FrameBank-association, duplicate, H0-materialization and chain-f identity checks are present;
  - production lifting and yaw functions are used;
  - no training code;
  - GT-free module;
  - no production import.

**Results:**
- 45 passed in the training image, with the unchanged Worklog 74 (7), Worklog 62 (7), Worklog 61 (12) and Worklog 73 (14) tests.
- Locally: 5 passed.

No production file changed, so the full regression was not run.

## Outputs

On LabServer63, `~/animcv-output/framepose_visual_ik_endpoint_test/` contains:
- `report.json` (SHA-256 `89218babea89033537895a851a013e2b9615ed97314de8945a84b1168424dea7`)
- `rows_test.json`, `rows_validation.json`
- `review/`
