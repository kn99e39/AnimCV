# Worklog — Out-of-Fold Arm Geometry Conditioning (2026-10-13)

## Question and scope

Does replacing the geometry owner's in-sample E1 TRAIN directions with
out-of-fold directions resolve Worklog 78's reach/bend-owner generalization
failure? Does frozen Qwen content help after that replacement?

Worklog 78 remains accepted as CASE C under its original contract. This is a
separate stacking attribution experiment, not architecture repair. Every result
here is **benchmark_detector_observation**. BEDLAM and 3DPW are reported
separately; detector observations are not oracle geometry.

## Answer

**No.** Training the owners on out-of-fold E1 directions does not resolve the Worklog 78 failure. **CASE C.**

- **The suspected mismatch was real, and OOF corrected it on the input side.** 3DPW TRAIN chain-f MAE moves from 0.081 (in-sample) to 0.171 (OOF), against 0.167 for untouched validation and 0.200 for test. Stable-sign accuracy moves from 98.1% to 89.1%, against 88.1% / 86.1% held out.
- **The owners did not improve.**
  - G0_OOF vs G0_IN, elbow: +0.03 cm [−0.02, +0.07].
  - G1_OOF vs G1_IN, elbow: +0.18 cm [+0.06, +0.30], worse in 28/37 sequences.
  - Neither OOF owner beats B1 at the elbow: G0_OOF +0.42 cm, G1_OOF +0.39 cm, both intervals including 0, with 28/37 and 23/37 sequences worse.
- **The learned-reach failure persists.** Swapping in OOF reach alone worsens the elbow by +0.65 cm (G0) and +0.57 cm (G1); the in-sample owners gave +0.69 / +0.42 cm.
- **The partial bend-side signal persists.** Side error vs B1 is −4.9° (G0_OOF) and −3.6° (G1_OOF).
- **Temporal behavior is unchanged.**
  - G0_OOF remains steadier than B1 (upper-arm residual −1.21°/pair, 34/37 sequences).
  - G1_OOF remains no steadier than B1 and jumpier than G0_OOF (+1.01°/pair, 32/37).
- **Qwen content still adds no reliable held-out benefit after the correction.** G1_OOF − G0_OOF elbow is −0.03 cm [−0.41, +0.35], and side error is +1.3° worse.
- **Conclusion.** The in-sample E1 conditioning mismatch is not the dominant cause of the Worklog 78 failure. The sequential "E1 direction → reach/bend regression" abstraction is closed without further head tuning.

## Starting state and preservation

- Branch: `arch/single_frame_first`.
- Starting HEAD: `7a1c18c90e760915b7f6ffea4ec87021c93d5c29`.
- Both local and LabServer63 fetched origin before work and matched that HEAD.
- Local worktree was clean. Server's existing `.DS_Store`, `.animcv_sync_stage/`
  and `docker/` were preserved.
- All training, inference measurements and tests run on LabServer63, using
  `animcv-framepose:cuda118`. Review rendering uses the macOS Blender 5.1.2 that
  rendered the Worklog 76–78 reviews (see "Real-scene review exports").
- **Handover.** The batch was started by another agent, which produced the audit,
  cross-fit teachers, OOF owners, contract verification, the held-out evaluation and
  the comparison report. It was then taken over and completed: the executed sources
  were checked against `execution_source.json` (audit / contract / training hashes
  equal), every gate and artifact was re-checked, the review export was corrected
  (below), and the evaluation write-up and classification were added.
- Historical modules, checkpoints, caches and outputs are read-only inputs.
  Production FramePose/FK/IK, canonical mathematics, BaseRig, root orientation,
  Qwen, H0, existing splits and the Worklog 61 solver are unchanged.

New diagnostic files:

- `scripts/audit_framepose_oof_conditioning.py`
- `scripts/framepose_oof_contract.py`
- `scripts/train_framepose_oof_conditioning.py`
- `scripts/verify_framepose_oof_artifacts.py`
- `scripts/compare_framepose_oof_owners.py`
- `scripts/export_framepose_oof_cases.py`
- `tests/test_framepose_oof_conditioning.py`
- This worklog.

The held-out IK/temporal evaluation reuses the unchanged Worklog 78
`run_framepose_arm_geometry_owner_eval.py` (pointed at the OOF owner directory).
The review uses the unchanged Worklog 77 renderer.

Server output root: `~/animcv-output/framepose_arm_geometry_oof/`. Completed
artifacts are written exclusively; rerunning never overwrites them. The exact
executed audit/contract/training source hashes and starting HEAD are recorded in
`execution_source.json`.

## Historical parity gate — PASS

The unchanged Worklog 78 evaluator was rerun into `historical_eval/`, followed
by independent saved-checkpoint forward reproduction in `audit/`.

- Frozen full-data E1 vs Worklog 73 held-out f: maximum difference **0**.
- E1 vs Worklog 78 stored f: maximum difference **0**.
- Saved G0/G1 checkpoint reach and raw side predictions: maximum differences
  **0**, including all historical test inputs.
- Historical test row records: exact equality, **6,003 rows / 37 sequences**.
- Historical pair records: exact equality, **5,504 adjacent pairs**.
- B0/B1 joints, rotations, reach, status, clamps, reference and H0 yaw: identical.
- Worklog 76 temporal thresholds and B0/B1 discontinuity rates: identical.

Thus strict same-environment retraining of the historical owners was unnecessary:
the original checkpoints and complete geometry outputs reproduce exactly in the
same CUDA/torch image. Their original training results remain historical controls.

## Original E1 distribution audit

Chain f is continuous forward-depth fraction. Stable sign uses Worklog 73's
`|f_GT| > sin(10°)`. Correct-sign magnitude error uses only stable rows where
the prediction sign is correct. Invalid targets are counted, never silently
treated as valid.

| source / split | population / valid chain targets | mean absolute f error | stable sign accuracy | correct-sign magnitude error | sequence-weighted MAE |
|---|---:|---:|---:|---:|---:|
| 3DPW TRAIN, in-sample | 11,334 / 10,411 | 0.08062 | 98.14% | 0.08563 | 0.08381 |
| 3DPW validation, selected E1 epoch | 3,407 / 3,058 | 0.16672 | 88.14% | 0.15867 | 0.15485 |
| 3DPW test, selection-clean | 7,076 / 6,015 | 0.19960 | 86.09% | 0.19141 | 0.19600 |
| BEDLAM TRAIN, in-sample | 21,232 / 17,451 | 0.17183 | 89.31% | 0.16388 | 0.17339 |
| BEDLAM validation | 1,330 / 1,121 | 0.21696 | 81.64% | 0.21102 | 0.20961 |

The E1-only valid-target population is broader than the IK population: the latter
also requires the unchanged runtime/reference geometry. It is not a changed
test split. Owner evaluation still uses exactly the historical 6,003 rows.

| source / split | near-plane MAE | intermediate MAE | out-of-plane MAE |
|---|---:|---:|---:|
| 3DPW TRAIN | 0.06447 | 0.08713 | 0.10140 |
| 3DPW validation | 0.11999 | 0.18864 | 0.22343 |
| 3DPW test | 0.13891 | 0.20283 | 0.28362 |
| BEDLAM TRAIN | 0.11636 | 0.16958 | 0.25888 |
| BEDLAM validation | 0.14061 | 0.22674 | 0.34399 |

Near-plane is `|f_GT| < sin(10°)`; out-of-plane is `|f_GT| >= 0.5`.
`audit/audit.json` retains signed/magnitude prediction percentiles, all regimes,
per-sequence metrics and error ownership shares. These are descriptive
statistics. Source-domain differences, different GT distributions and epoch
selection prevent treating TRAIN/held-out differences alone as causal proof.

## Cross-fitting contract, declared before training

One fixed **5-fold** partition, with no fold-count or assignment sweep:

- 3DPW groups are entire capture sequences, including both actors: 24 TRAIN
  captures, 11,334 rows. Subject identity is not available in the FrameBank.
- BEDLAM groups are entire existing sequences, including all people and frames:
  8,821 TRAIN sequences, 21,232 rows.
- Assignment is label-blind and deterministic, separately by source: descending
  group row count, SHA-256 of `1337|group` for ties, least-loaded fold for placement.
- Each TRAIN row receives exactly one prediction from the corresponding outer
  teacher; its entire declared capture/sequence is absent from teacher fitting
  **and teacher selection**.
- Original validation/test labels are used by neither teacher optimization nor
  teacher selection. Each outer complement has an inner sequence-grouped
  five-way partition; inner fold 0 supplies the 3DPW selection objective.
- Initial teacher fitting retains the W73 graph, targets, frozen Qwen readout,
  50/50 source sampler, optimizer, 200 epochs, 11,334 draws/epoch, 9,000 steps,
  seed and every-ten-epoch selection cadence.
- Selection minimizes inner 3DPW masked MAE over the original three f outputs.
- A freshly initialized teacher then fits **all** outer TRAIN complement rows
  eligible under W73 for `selected_epoch + 1` epochs, using the same full
  200-epoch cosine schedule. This avoids sacrificing inner-selection coverage
  in the final teacher; the outer predicted rows remain completely excluded.

| fold | excluded 3DPW rows | excluded BEDLAM rows |
|---:|---:|---:|
| 0 | 2,288 | 4,247 |
| 1 | 2,314 | 4,247 |
| 2 | 2,280 | 4,246 |
| 3 | 2,335 | 4,246 |
| 4 | 2,117 | 4,246 |

The immutable `crossfit/fold_manifest.json` binds every sample ID, feature-row
position, split, source, group and fold. Its SHA-256 is
`e0550d2891bafd112fd00e85bdf633ace76a59f2423317dce91738e18e9a35fd`.
Each `fold_N_provenance.json` records initial fit, selection, refit and actual
optimizer populations, curves, checkpoint hash and repeated-inference equality.

**Limits fixed before interpreting owners.** BEDLAM subjects can repeat across
different sequence folds. Joining both sequence and subject identities would
leave only two connected components (17,118 / 4,114 rows), a severe coverage
imbalance. The permitted sequence-level contract was chosen to preserve useful
coverage. This experiment does not establish subject-disjoint generalization or
eliminate all feature memorization. OOF also reduces each final teacher's data
coverage to approximately 80% and uses internal epoch selection plus refitting;
these effects cannot be separated from removal of in-sample optimism. Worse
OOF predictions are not, by themselves, proof of leakage.

## OOF direction distributions

| source TRAIN | in-sample MAE → OOF MAE | in-sample sign → OOF sign | OOF correct-sign magnitude error |
|---|---:|---:|---:|
| 3DPW | 0.08062 → 0.17058 | 98.14% → 89.14% | 0.16903 |
| BEDLAM | 0.17183 → 0.20724 | 89.31% → 85.39% | 0.18713 |

| OOF source TRAIN | near-plane MAE | intermediate MAE | out-of-plane MAE |
|---|---:|---:|---:|
| 3DPW | 0.12262 | 0.17451 | 0.26538 |
| BEDLAM | 0.14509 | 0.20035 | 0.31212 |

3DPW OOF predictions are reasonably comparable to untouched full-data E1
inference: MAE 0.17058 vs validation 0.16672 / test 0.19960, and prediction
magnitude mean / median / p95 of 0.24293 / 0.17626 / 0.72072 vs test
0.24749 / 0.17442 / 0.71873. Original TRAIN magnitude was
0.28039 / 0.19137 / 0.81522. Stable sign and every depth-regime MAE move closer
to held-out conditions. BEDLAM error/sign and conditional errors also move
closer, while its magnitude mean is slightly farther from validation
(0.26658 vs 0.25180, originally 0.26470).

This intervention makes TRAIN conditioning more representative on the principal
error measures; it does not make source/split distributions identical. Validation
and test remain separate, and teacher coverage/selection remain limitations.
Full sequence accounting is in `crossfit/oof_distribution.json`.

## Owner controls

G0_OOF and G1_OOF call the unchanged Worklog 78 `train_owner`. Only TRAIN E1
conditioning changes. The same Worklog 78 definition reprojects bend-side
targets relative to the corresponding OOF direction. Reach targets, input
geometry, graph/width/parameter count, loss weights, eligibility, 50/50 source
ratio, optimization budget and 3DPW validation selection rule are preserved.
G0 has zero visual tensors; G1 has the unchanged frozen Qwen tensors. Validation
and test use the original frozen full-data E1 for **all four** owners.

No new backbone, temporal observation, smoothing, loss, gate, fallback, selector
or production integration is introduced.

Both owners have 317,636 parameters, 9,000 steps, 2,266,800 draws and exactly
1,133,400 draws per source. Their initialization digest, also identical to the
historical controls, is
`89cc1e8fe02d6e9eee3c93248c7dce28f80d1bc3bcb9f8f7c4661576dfe2ea66`.

Objective = reach MAE + masked side `(1 - cosine)`:

| epoch | G0_OOF TRAIN / 3DPW validation / BEDLAM validation | G1_OOF TRAIN / 3DPW validation / BEDLAM validation |
|---:|---|---|
| 9 | 0.2034 / 0.1988 / 0.2569 | 0.1089 / 0.1657 / 0.2188 |
| 49 | 0.1450 / 0.1518 / 0.2214 | 0.0499 / 0.1339 / 0.1728 |
| 89 | 0.1274 / 0.1503 / 0.2162 | **0.0356 / 0.1256 / 0.1766 (selected)** |
| 109 | 0.1228 / 0.1457 / 0.2145 | 0.0325 / 0.1261 / 0.1778 |
| 189 | **0.1133 / 0.1376 / 0.2172 (selected)** | 0.0222 / 0.1292 / 0.1878 |
| 199 | 0.1148 / 0.1388 / 0.2144 | 0.0220 / 0.1298 / 0.1893 |

At selection, G0_OOF TRAIN reach MAE / side angle are 0.05098 / 15.33°,
and validation 0.05004 / 18.64°. G1_OOF TRAIN is 0.02461 / 6.26°,
validation 0.04007 / 17.87°. The G1 train-to-validation gap persists.

## Real-data contract verification — PASS

`contract_verification.json` validates the actual artifacts, not just synthetic
tests:

- Exactly one OOF prediction for **all 32,566 TRAIN rows**, including rows not
  eligible for teacher supervision; no original validation/test row is fitted
  or used for teacher selection.
- Deterministic folds; no capture/sequence leakage in outer prediction or inner
  selection; exact sample-ID/feature-row matches.
- Actual eligible refit optimizer populations equal the allowed outer TRAIN
  complement; manifest hashes are bound into every teacher checkpoint.
- Reloaded five teacher checkpoints reproduce all saved OOF f predictions with
  maximum difference **0**. Both reloaded owners reproduce stored reach/side
  outputs with maximum differences **0**.
- Frozen full-data E1 identity, validation/test geometry inputs and unchanged
  target construction reproduce exactly.
- All 159 BEDLAM TRAIN subjects occur in multiple sequence folds. This is an
  explicit limitation of the declared sequence-level contract, not a claim of
  subject isolation.



## Held-out evaluation — gates

The unchanged Worklog 78 evaluator was run with the OOF owner directory (`oof_eval/`):

**Gate 1 passes.** Everything below matches the in-sample evaluation exactly (max diff 0):
- 6,003 rows, 37 sequences, 5,504 pairs, identical to Worklog 77;
- B0/B1 joints and rotations;
- oracle elbows, reference, yaw;
- frozen E1 input;
- B0/B1 discontinuity rates, using the Worklog 76 thresholds.

**Gate 2 passes.**
- All learned and attribution arms are known on every row: 0 unavailable, 0 clamped, rig residuals ≤ 7.8e-14 cm.
- OOF reach ranges: G0 [0.164, 0.998], G1 [0.277, 0.996].

**Comparison report (`comparison/report.json`).** It merges the in-sample and OOF evaluations row by row and pair by pair. The merge refuses any change to the reference, E1 input, GT, yaw, B0/B1/C111 or the population.

## Test static geometry (6,003 rows)

| candidate | wrist cm mean / p50 / p95 / max | elbow cm mean / p50 / p95 / max | upper ° | forearm ° | reach MAE | side err ° | \|bend err\| ° | proximal / distal rot vs ref ° |
|---|---|---|---:|---:|---:|---:|---:|---|
| B0 | 14.88 / 11.46 / 37.08 / 90.6 | 8.25 / 6.09 / 21.54 / 49.9 | 17.42 | 22.57 | 0.054 | 23.4 | 14.3 | 17.8 / 24.6 |
| B1 | 11.89 / 9.63 / 29.96 / 76.3 | 7.64 / 6.11 / 18.98 / 44.2 | 15.95 | 18.33 | 0.054 | 23.2 | 14.3 | 16.3 / 24.3 |
| G0_IN (Worklog 78) | 11.93 / 9.54 / 29.76 / 73.6 | 8.03 / 6.58 / 19.61 / 42.1 | 16.75 | 17.97 | 0.058 | 18.4 | 16.5 | 17.2 / 24.7 |
| G1_IN (Worklog 78) | 11.79 / 9.48 / 29.60 / 73.2 | 7.85 / 6.41 / 19.32 / 50.1 | 16.40 | 17.17 | 0.054 | 19.3 | 14.2 | 16.8 / 22.9 |
| **G0_OOF** | 11.95 / 9.57 / 29.89 / 73.5 | 8.06 / 6.64 / 19.50 / 42.4 | 16.81 | 17.95 | 0.057 | 18.3 | 16.4 | 17.2 / 24.7 |
| **G1_OOF** | 11.80 / 9.48 / 29.63 / 73.2 | 8.03 / 6.65 / 19.46 / 49.4 | 16.78 | 17.51 | 0.057 | 19.6 | 15.3 | 17.2 / 24.2 |

**Paired comparisons:**
- Intervals are frame-weighted 95%: SC = sequence-cluster bootstrap; MB10 / MB25 = moving-block bootstrap with block 10 / 25. All use 2,000 replicates, seed 1337, as in Worklogs 75–78.
- The sequence-weighted mean follows the frame mean in parentheses.
- "Sequences" counts sequences that improve / get worse.

| comparison · metric | mean (seq-weighted) | SC | MB10 | MB25 | sequences |
|---|---|---|---|---|---|
| **G0_OOF − G0_IN elbow (cm)** | +0.028 (+0.024) | [−0.018, +0.070] | [+0.003, +0.047] | [−0.005, +0.050] | 15 / 22 |
| G0_OOF − G0_IN reach MAE | −0.0003 | [−0.002, +0.001] | | | 19 / 18 |
| G0_OOF − G0_IN side error (°) | −0.09 | [−0.35, +0.14] | | | 20 / 17 |
| **G1_OOF − G1_IN elbow** | **+0.179** (+0.131) | [+0.056, +0.295] | [+0.100, +0.242] | [+0.072, +0.245] | 9 / 28 |
| G1_OOF − G1_IN reach MAE | +0.003 | [+0.001, +0.006] | | | 11 / 26 |
| G1_OOF − G1_IN side error | +0.25 | [−0.32, +0.91] | | | 17 / 20 |
| G1_OOF − G1_IN forearm (°) | +0.34 | [+0.10, +0.64] | | | 11 / 26 |
| **G1_OOF − G0_OOF elbow** | −0.026 (−0.134) | [−0.412, +0.350] | [−0.119, +0.224] | [−0.099, +0.312] | 22 / 15 |
| G1_OOF − G0_OOF side error | +1.31 | [+0.20, +2.43] | | | 15 / 22 |
| **G0_OOF − B1 elbow** | +0.416 (+0.404) | [−0.048, +0.850] | [+0.142, +0.566] | [+0.025, +0.545] | 9 / 28 |
| G0_OOF − B1 side error | −4.94 | [−7.57, −2.64] | | | 29 / 8 |
| **G1_OOF − B1 elbow** | +0.390 (+0.270) | [−0.199, +0.992] | [+0.159, +0.655] | [+0.093, +0.685] | 14 / 23 |
| G1_OOF − B1 wrist | −0.08 | [−0.29, +0.17] | | | 21 / 16 |
| G1_OOF − B1 forearm | −0.82 | [−2.27, +0.58] | | | 21 / 16 |
| G0_OOF / G1_OOF − B0 wrist | −2.93 / −3.07 | both exclude 0 | | | 31 / 6 and 32 / 5 |

**Wrist and forearm.** The E1 advantages over B0 are preserved, and neither OOF owner moves the wrist materially vs B1.

**Coverage and rig.** Coverage, clamps (none) and rig invariants are identical across all six candidates.

## Test temporal stability (5,504 pairs; Worklog 76/77/78 contract and thresholds)

| per pair | B0 | B1 | G0_IN | G1_IN | **G0_OOF** | **G1_OOF** |
|---|---:|---:|---:|---:|---:|---:|
| upper-arm direction residual (°) | 8.81 | 10.97 | 9.66 | 10.81 | 9.76 | 10.77 |
| proximal rotation residual (°) | 9.01 | 11.23 | 9.86 | 11.07 | 9.97 | 11.04 |
| forearm direction residual (°) | 12.95 | 12.16 | 11.48 | 12.72 | 11.33 | 12.49 |
| elbow motion residual (cm) | 4.27 | 5.32 | 4.68 | 5.24 | 4.73 | 5.22 |
| wrist motion residual (cm) | 7.76 | 8.08 | 7.97 | 8.08 | 7.99 | 8.04 |
| proximal discontinuity rate | 1.29% | 3.12% | 2.11% | 2.96% | 2.09% | 2.78% |
| elbow discontinuity rate | 1.34% | 3.22% | 2.22% | 2.93% | 2.25% | 2.78% |
| learned reach change | 0.039 (H0) | 0.039 (H0) | 0.030 | 0.039 | 0.029 | 0.039 |
| learned side change (°) | 16.9 (H0) | 17.6 (H0) | 14.7 | 16.7 | 14.6 | 16.1 |

| comparison | upper-arm residual (°/pair) | proximal discontinuity | elbow residual (cm) | reach / side change |
|---|---|---|---|---|
| G0_OOF − G0_IN | +0.10 [+0.07, +0.14], worse in 29/37 | −0.02 pp [−0.1, +0.1] | +0.05 [+0.03, +0.07] | −0.0009 / −0.11° |
| G1_OOF − G1_IN | −0.04 [−0.17, +0.09] | −0.18 pp [−0.5, +0.1] | −0.02 [−0.08, +0.04] | −0.0002 / **−0.65° [−1.20, −0.18]** |
| G0_OOF − B1 | **−1.21 [−1.55, −0.92]**, better in 34/37 | **−1.04 pp [−1.7, −0.4]** | −0.59 [−0.75, −0.45] | −0.010 / −3.0° |
| G1_OOF − B1 | −0.20 [−0.72, +0.30] | −0.35 pp [−1.2, +0.6] | −0.10 [−0.35, +0.15] | −0.000 / −1.5° |
| G1_OOF − G0_OOF | **+1.01 [+0.60, +1.42]**, worse in 32/37 | +0.69 pp [+0.0, +1.5] | +0.49 [+0.29, +0.69] | +0.010 / +1.5° |

**OOF conditioning leaves temporal behavior essentially where it was.**
- G1_OOF's learned side changes slightly less per frame than G1_IN's (−0.65°/pair), but its arm residuals do not change.
- G0_OOF is marginally less steady than G0_IN (+0.10°/pair) while keeping the same advantage over B1.
- **G0_OOF repeats the Worklog 78 trade-off,** not a success: steadier than B1 with no static elbow gain.

## Reach vs bend-side attribution (elbow change vs B1, cm; E1 direction throughout)

| owner | learned reach + H0 side | H0 reach + learned side | learned reach + learned side | interaction |
|---|---:|---:|---:|---:|
| G0_IN (Worklog 78) | +0.69 | −0.22 | +0.39 | −0.08 |
| **G0_OOF** | **+0.65** | −0.15 | +0.42 | −0.09 |
| G1_IN (Worklog 78) | +0.42 | −0.10 | +0.21 | −0.10 |
| **G1_OOF** | **+0.57** | −0.06 | +0.39 | −0.13 |

**OOF changes neither owner:**
- **Reach.** Reach generalization is unchanged: OOF − IN reach MAE is −0.0003 for G0 and +0.003 for G1. The learned-reach elbow regression persists and grows for G1.
- **Side.** The side-specific elbow gain shrinks slightly.

**Temporal reach and side fluctuations** stay at their Worklog 78 levels: G1 reach change 0.039 vs H0 0.039 and GT 0.023.

## Train / validation / test input-distribution accounting

The E1 conditioning seen by the owner:

| population | chain-f MAE | stable sign | magnitude mean / p50 / p95 |
|---|---:|---:|---|
| 3DPW TRAIN, in-sample (Worklog 78 training input) | 0.081 | 98.1% | 0.280 / 0.191 / 0.815 |
| **3DPW TRAIN, OOF (Worklog 79 training input)** | **0.171** | **89.1%** | 0.243 / 0.176 / 0.721 |
| 3DPW validation (inference) | 0.167 | 88.1% | 0.246 / 0.172 / 0.735 |
| 3DPW test (inference) | 0.200 | 86.1% | 0.247 / 0.174 / 0.719 |
| BEDLAM TRAIN, in-sample | 0.172 | 89.3% | 0.265 mean |
| **BEDLAM TRAIN, OOF** | **0.207** | **85.4%** | 0.267 mean |
| BEDLAM validation (inference) | 0.217 | 81.6% | 0.252 mean |

**Gap to the held-out inference input:**

| population | in-sample gap | OOF gap |
|---|---:|---:|
| 3DPW vs test | 0.119 | 0.029 |
| 3DPW vs validation | 0.086 | 0.004 |
| BEDLAM vs validation | 0.045 | 0.010 |

- The 3DPW OOF-to-test MAE ratio is 0.85, and OOF-to-validation is 1.02.
- In every |f_GT| regime, the OOF MAE lies much closer to the held-out MAE than the in-sample MAE does. 3DPW near-plane / intermediate / out-of-plane:

  | input | near-plane | intermediate | out-of-plane |
  |---|---:|---:|---:|
  | in-sample | 0.064 | 0.087 | 0.101 |
  | OOF | 0.123 | 0.175 | 0.265 |
  | test | 0.139 | 0.203 | 0.284 |

**OOF conditioning did make the owner's training input substantially more representative of its inference input.** The causal test is therefore interpretable, within the stated limits:
- each final teacher sees ~80% of TRAIN;
- epochs are selected internally and then refit;
- BEDLAM subjects repeat across sequence folds;
- OOF does not address Qwen-feature overfitting of the owner itself, which persists (G1_OOF train 6.3° vs validation 17.9° side error).

## Regimes (elbow cm; sequences improved vs B1)

| regime | rows | B1 | G0_IN → G0_OOF | G1_IN → G1_OOF | G1_OOF − B1 (sequences improved) | G1_OOF elbow from reach / side | upper-arm residual B1 → G0_OOF / G1_OOF |
|---|---:|---:|---|---|---|---|---|
| near-straight | 931 | 5.30 | 6.79 → 6.82 | 5.37 → 5.58 | +0.28 (12/37) | +0.46 / −0.06 | 6.17 → 6.21 / 6.19 |
| moderate | 2,716 | 6.75 | 7.40 → 7.37 | 7.30 → 7.51 | +0.76 (15/37) | +0.68 / +0.14 | 10.15 → 8.74 / 9.65 |
| strong | 1,606 | 8.80 | 8.76 → 8.88 | 9.24 → 9.60 | +0.80 (15/37) | +0.82 / +0.18 | 11.22 → 9.92 / 12.32 |
| near-folded | 750 | 11.28 | 10.26 → 10.31 | 9.96 → **9.59** | **−1.68 (19/31)** | −0.21 / **−1.24** | 14.69 → 12.47 / 13.29 |
| near-plane | 2,117 | 5.65 | 6.58 → 6.65 | 5.89 → 6.13 | +0.47 (11/37) | +0.66 / −0.03 | 8.09 → 7.22 / 8.34 |
| intermediate | 2,453 | 7.13 | 7.28 → 7.33 | 7.24 → 7.39 | +0.26 (14/37) | +0.50 / −0.06 | 9.90 → 8.77 / 10.00 |
| out-of-plane | 1,433 | 11.45 | 11.45 → 11.37 | 11.78 → 11.93 | +0.49 (17/36) | +0.56 / −0.10 | 14.23 → 12.51 / 13.38 |

**The only regime where a learned owner helps the elbow is still near-folded, and the gain is still owned by the side.**
- G1_OOF elbow is −1.68 cm vs B1 there (G1_IN gave −1.32).
- On the 188 Worklog 76 near-folded conflict rows, G1_OOF elbow is 8.96 cm (G1_IN 9.79, B1 15.03, B0 9.46, C111 5.58). It is better than B1's in 76% of those rows, with wrist 10.08 cm (B1 12.29).
- Every other regime gets worse at the elbow, through the learned reach.

## Per-sequence

**G1_OOF − G1_IN elbow** improves in only 9/37 sequences.

| | sequences (elbow change, cm) |
|---|---|
| largest gains | `downtown_weeklyMarket_00` a0 −1.27, `downtown_bus_00` a0 −0.56, `downtown_car_00` a0 −0.53 |
| largest regressions | `downtown_bus_00` a1 +0.94, `downtown_warmWelcome_00` a0 +0.91, `downtown_sitOnStairs_00` a0 +0.63 |

**G0_OOF − G0_IN elbow** improves in 15/37 sequences.

**Vs B1, the OOF owners keep their Worklog 78 profiles:**
- elbow regresses in 28/37 (G0_OOF) and 23/37 (G1_OOF) sequences;
- `downtown_bus_00` a1 is the worst regression for both;
- G0_OOF's upper-arm residual improves in 34/37 sequences.

Per-sequence tables for all six candidates are in `comparison/report.json → per_sequence`.

## Owner frames (wrist cm / elbow cm / side error °)

| row (learned reach G0_OOF / G1_OOF) | B1 | G0_IN | G1_IN | G0_OOF | G1_OOF |
|---|---|---|---|---|---|
| `downtown_bar_00` a0 #514 (0.91 / 0.70; GT 0.53) | 26.1 / 29.6 / 59 | 21.6 / 20.7 / 18 | 13.9 / 13.3 / 1 | 22.9 / 21.7 / 13 | **12.5 / 12.2 / 1** |
| `downtown_bar_00` a0 #775 (0.93 / 0.67; GT 0.97) | 8.3 / 4.5 / 105 | 8.7 / 7.7 / 18 | 15.5 / 17.1 / 3 | 8.3 / 6.5 / 13 | 18.0 / **18.7** / 7 |
| `downtown_sitOnStairs_00` a0 #325 | 67.7 / 42.4 / 7 | 68.4 / 38.3 / 27 | 69.7 / 37.9 / 46 | 69.3 / 39.4 / 21 | 69.4 / 37.7 / 45 |
| `flat_packBags_00` a0 #446 | 30.2 / 25.4 / 11 | 29.6 / 25.0 / 10 | 32.1 / 29.6 / 24 | 29.1 / 25.8 / 15 | 32.8 / 28.1 / 15 |
| `downtown_runForBus_01` a0 #185 | 9.1 / 7.1 / 17 | 9.2 / 8.4 / 2 | 8.1 / 8.6 / 23 | 9.3 / 8.3 / 1 | 8.2 / 9.4 / 22 |

The owner-frame pattern is the same as in Worklog 78:
- **#514:** a near-exact side and a closer reach help.
- **#775:** the G1 under-reach breaks a frame B1 solves, now slightly worse.
- **#325 and #446:** these are E1 direction failures that no reach/side owner can repair.

## Clean validation (selection split; separate)

| arm | elbow (cm) | reach MAE | side error |
|---|---:|---:|---:|
| B1 | 6.51 | 0.044 | 21.6° |
| G0_OOF | 6.79 | 0.050 | 18.6° |
| G1_OOF | 6.06 | 0.040 | 17.9° |

These 3,056 rows chose the owner epochs. G1_OOF − B1 elbow is −0.45 cm there, as G1_IN's was (−0.54). As in Worklog 78, test does not confirm it.

## Real-scene review exports

**One correction to the inherited export.** The previous attempt rendered with a Linux Blender on LabServer63. Its FBX import differs from the authoritative rest snapshot by float32 rounding of bone matrices (≤ 1.8e-7), so the strict Worklog 77 binding gate refused it. The inherited export then added an optional "platform rest" for visual binding.

That bypass and its diagnostic script were removed. The export now always binds to the authoritative rest (digest `8519d243…`), and it was rendered with the macOS Blender 5.1.2 used for Worklogs 76–78. That import reproduces the snapshot exactly, so the binding gate applies unchanged. Blender FK end vs export: 9.5e-7 cm.

- **Drawn arms:**
  - B0 blue, B1 magenta;
  - G0_OOF orange, G1_OOF cyan;
  - GT green, E_GT yellow.
- **Captions:** the frozen E1 and GT f, GT reach, OOF and in-sample elbow/wrist errors, learned reach and side error. Strip captions also show the next pair's upper-arm residual, in-sample → OOF.

| export (`review/`) | row(s) |
|---|---|
| `owner_downtown_bar_00_514` (also the near-folded conflict pick), `owner_downtown_bar_00_775`, `owner_downtown_sitOnStairs_00_325`, `owner_flat_packBags_00_446`, `owner_downtown_runForBus_01_185` | owner frames |
| `largest_g0_oof_elbow_improvement` / `_regression` (OOF vs in-sample) | `downtown_car_00` a0 #789 / `downtown_upstairs_00` a0 #107 |
| `largest_g1_oof_elbow_improvement` / `_regression` | `downtown_enterShop_00` a0 #1191 / `outdoors_fencing_01` a0 #235 |
| `largest_g0_oof_temporal_{improvement,regression}_strip` | `downtown_upstairs_00` a0 #102–107 / `downtown_cafe_00` a1 #253–258 |
| `largest_g1_oof_temporal_{improvement,regression}_strip` | `outdoors_fencing_01` a0 #409–424 / #225–240 |

- Strips contain the available consecutive common-valid frames (2–4).
- **Server copy:** `review/`, with cases in `review/review_cases.json` (`4478f0c4…`) and the render report `a1597c9e…`.
- **QE images-only copy:** `~/animcv-output/79_oof_arm_geometry_review/` (48 PNGs).
- The superseded platform-rest `review_cases.json` at the output root is left untouched and unused.

## Architecture classification — CASE C

| CASE C requirement | result |
|---|---|
| OOF prediction validity | ✔ every one of 32,566 TRAIN rows predicted once by a teacher that never fit or selected on its capture/sequence; reloads exact |
| distribution accounting | ✔ OOF training input moves to within 0.004–0.029 MAE of held-out inference (from 0.045–0.119) |
| owner architecture and budget fixed | ✔ identical graph, init digest, steps, draws and per-source draws as Worklog 78 |
| OOF owners do not materially improve held-out results | ✔ G0: +0.03 cm elbow, not significant; G1: +0.18 cm elbow (worse); neither beats B1; no temporal change |

**Why not CASE A or B:**
- **Not A or B.** OOF improves neither reach (MAE unchanged) nor side (G0 −0.09°, not significant; G1 +0.25°) generalization, and neither static elbow nor temporal stability vs B1.

**Why not CASE D:**
- **Not D.** The cross-fitting contract and historical parity pass, and the OOF distribution is close to the held-out inference distribution. The teacher-coverage and subject-repetition limits are real but do not make the comparison uninterpretable: the OOF input is, if anything, about as hard as the held-out one.

**Conclusion.**
- The in-sample E1 conditioning mismatch is not the dominant cause of Worklog 78's failure under this controlled experiment.
- The principal failing owner remains reach, which is no more accurate than H0's and degrades the elbow outside near-folded arms.
- Bend side carries partial evidence; Qwen content adds frame-to-frame noise, not held-out accuracy.
- Per the completion condition, the sequential "independently predicted E1 chain direction → reach/bend regression" architecture is closed without further heuristic tuning.
- No joint geometry model, temporal model, fusion, gate or production step is started; the next architecture needs a new bounded decision.

## Focused tests

- **`tests/test_framepose_oof_conditioning.py` (7 tests):**
  - label-blind, deterministic, capture-isolated fold assignment;
  - refusal of held-out rows;
  - rejection of row and capture leakage and of unauthorized teacher rows;
  - exactly one prediction per TRAIN row with exact sample/feature order;
  - no production imports;
  - the comparison merge refuses changed frozen inference and keeps pair order;
  - the sequence-cluster bootstrap keeps frame- and sequence-weighting distinct.
- **Real-data checks:** `contract_verification.json` covers the real-data counterparts: teacher reloads, optimizer populations, frozen held-out inputs, targets and owner reloads.
- **Server run (`animcv-framepose:cuda118`):** together with the Worklog 61/62/73/74/75/76/77/78 focused tests, **73 passed** after the takeover changes.
- **Full regression:** not run, because production behavior is unchanged.

## Artifact identities (`~/animcv-output/framepose_arm_geometry_oof/`)

| file | sha256 |
|---|---|
| `audit/audit.json` | `accd1261…` |
| `crossfit/fold_manifest.json` | `e0550d28…` |
| `crossfit/oof_predictions.npz` | `b976c02b…` |
| `crossfit/oof_distribution.json` | `7205911f…` |
| `crossfit/owners/G0_GEOMETRY_ONLY.pt` | `d78bff95…` |
| `crossfit/owners/G1_GEOMETRY_PLUS_VISION.pt` | `c68f4789…` |
| `crossfit/owners/predictions_3dpw.npz` | `afe55ebd…` |
| `crossfit/owners/training_report.json` | `647346b5…` |
| `historical_eval/report.json` | `8f37d6cd…` (identical to Worklog 78) |
| `oof_eval/report.json` | `337cbef3…` |
| `comparison/report.json` | `6d9ef090…` |
| `contract_verification.json` | `b462e488…` |

Historical inputs are pinned in the scripts: E1 `ee002f6a…`, Worklog 73 report/rows, Worklog 76/77 reports and rows, and the Worklog 78 owners `fdbd83b2…` / `7dd6f0e9…`.
