# Worklog — Near-Plane False-Depth Attribution (2026-10-17)

**Question.** What makes the RGB-conditioned connected-limb model (G1) less accurate than the matched 2D-only model (G0) on near-plane geometry? Is there enough evidence to justify, and then validate, one targeted single-frame accuracy mechanism?

**Answer: no intervention is justified.** The result is **Diagnostic C (observation / population dominated) by the predeclared rule, so the batch STOPS at the diagnostic gate.** No candidate was trained and no historical artifact was modified.

What the diagnosis established, on frozen models whose Worklog 86 outputs were reproduced exactly:

1. **On 3DPW TEST the near-plane penalty is real, and mostly a camera-depth error.**
   - Rows with both segments near the image plane (regime A) show a significant G1 − G0 distal squared-error penalty for all three chains:

     | chain | penalty ((%L)²) | 95% interval | depth-axis share |
     |---|---:|---|---:|
     | left arm | +136 | [+66, +217] | 84% |
     | right arm | +89 | [+15, +191] | 56% |
     | right leg | +93 | [+3, +198] | 91% |

   - The same holds for the chain-near-plane-by-cancellation rows (regime B): left arm +139 (83%), right arm +124 (89%).
2. **The penalty does not reproduce on the selection-free validation sources.** Regime A, G1 − G0:

   | chain | 3DPW VALIDATION | BEDLAM VALIDATION |
   |---|---|---|
   | left arm | +7 [−68, +62] | −92 [−238, +39] |
   | right leg | −38 [−84, +10] | **−54 [−86, −26]**: G1 is better |
   | right arm | **+94 [+19, +144]** | +84 [−85, +255] |

   The penalty is a property of the evaluation population, not a consistent model behaviour.
3. **For two of three chains, the 3DPW TEST penalty sits in the rows where the detector disagrees more with GT 2D.**
   - In the higher-discrepancy half the penalty is significant: right arm +136 [+38, +328], right leg +121 [+6, +257].
   - In the lower half it is not: +43 [−24, +111] and +65 [−10, +133]. Both lower-half means are still positive, so this partly reflects lower power from halving the rows.
   - For the left arm the opposite holds (higher half +112, lower half +159). The left arm's false depth also occurs on clean observations (see review).
4. **Training support is not the cause.** Regime A is 6–11% of the arm training rows and 22% of BEDLAM leg rows. The leg, the best-supported chain, is exactly where G1 is better on BEDLAM VALIDATION.
5. **The temporal penalty is a separate, global effect.**
   - On 3DPW (1/6 s), G1 adds depth-coordinate motion residual uniformly across every regime: left arm +3.1 %L per pair [+2.5, +3.8], 37/37 sequences, about +3.3 to +4.4 in each regime.
   - It moves depth 1.4× more than GT (10.6 vs 7.6 %L per pair).
   - Its correlation with the near-plane static false depth is ρ = 0.08.
   - The static near-plane penalty and the temporal penalty do not share an identifiable cause.

Because the near-plane penalty is depth-owned on one reused test population but not consistent across validation sources, and is observation-dependent for two of three chains, no runtime-legitimate, architecture-level mechanism is identified. Training one would be speculative.

## Starting state and files

- **Start:** `arch/single_frame_first` at `18e6d0e` (Worklog 86). Local, origin and LabServer63 were equal; the worktree was clean. git was run via `/Library/Developer/CommandLineTools/usr/bin/git`.
- **Commits:**
  - `6e52dd7`: diagnostics and the predeclared `DIAGNOSIS` rules, before any diagnostic output;
  - `6700c58`: factor-hybrid scaling fix (see §6);
  - the commit adding the review renderer and this worklog.
- **Added (diagnostic only; no candidate files, no `src/` coupling):**
  - `src/framepose/near_plane_attribution.py`: regimes, exact decompositions, evaluation-only factor hybrids;
  - `scripts/compute_limb_detector_discrepancy.py`
  - `scripts/predict_near_plane_diagnostics.py`
  - `scripts/run_near_plane_attribution.py`
  - `scripts/review_near_plane_attribution.py`
  - `tests/test_framepose_near_plane_attribution.py`
  - this worklog
- **Preserved:** every historical checkpoint, cache, bank and report; production FramePose/FK/IK/Blender; H0, E1, Root Orientation and BaseRig.

## 1. Historical parity (passed before any diagnostic)

| frozen candidate (chain) | checkpoint sha256 |
|---|---|
| left arm G1 = J1X (Worklog 85) | `82fd6bcb…` |
| left arm G0X (Worklog 86) | `a7aecc6e…` |
| right arm G0 / G1 (Worklog 86) | `ae1ccea9…` / `e7f220d6…` |
| right leg G0 / G1 (Worklog 86) | `5186a471…` / `25b660c4…` |

- **Reproduction.** Re-running the Worklog 86 prediction path from the frozen checkpoints, the Worklog 68/86 readouts and the detector inputs reproduces the stored Worklog 86 predictions and canonical targets **exactly (max difference 0.0)** for all chains on 3DPW and 3DHP. Readout caches and banks are hash-verified.
- **Additional source.** The same frozen models were also run on BEDLAM VALIDATION.
- **Artifacts:** `predictions/prediction_report.json` `c1eb2332…`.

## 2. Populations and regimes (declared in `near_plane_attribution.REGIMES` before diagnostics)

- **Populations:** chain-valid rows (GT valid and detector-visible P, M, D) of:
  - 3DPW TEST: 37 sequences, primary;
  - 3DPW VALIDATION: 16 sequences;
  - BEDLAM VALIDATION: 1,121 / 1,115 / 993 rows;
  - MPI-INF-3DHP TEST: 6 clusters.
- **Depth components.** f = the camera-depth (Y) component of the unit proximal segment, distal segment and P→D chain direction, computed from GT.

| regime | definition |
|---|---|
| **A_true_near** | abs(f_proximal) < sin 10° and abs(f_distal) < sin 10°: both segments near the image plane |
| **B_cancellation** | abs(f_chain) < sin 10° but not A: the chain is near-plane only through segment cancellation, or one segment leaves the plane |
| **I_intermediate** | sin 10° ≤ abs(f_chain) < 0.5 |
| **C_out_of_plane** | abs(f_chain) ≥ 0.5 |
| **D_degenerate** | GT reach below 0.25: P→D direction and depth sign unreliable (at most 6 rows per population) |

| 3DPW TEST rows | A | B | I | C | D |
|---|---:|---:|---:|---:|---:|
| left arm (6,003) | 672 | 1,449 | 2,449 | 1,432 | 1 |
| right arm (6,076) | 522 | 1,661 | 2,616 | 1,276 | 1 |
| right leg (6,519) | 733 | 1,577 | 3,701 | 508 | 0 |

The Worklog 86 "near-plane" bin, abs(f_chain) < sin 10°, mixes A with the larger B. B is as penalized as A for both arms, and for the leg B is about 70% as large and not significant.

## 3. Camera-axis error accounting (distal joint, 3DPW TEST, (%L)²)

The decomposition |e|² = eX² + eY² + eZ² is exact (tested). Y is camera depth.

| regime | chain | G0 mean sq. | G1 mean sq. | G1 − G0 total | depth part | image-plane part | depth share | over-depth G1 − G0 (%L) |
|---|---|---:|---:|---|---|---|---:|---|
| A | left arm | 171 | 306 | **+136 [+66, +217]** | +115 | +21 | 0.84 | **+3.0 [+1.5, +4.3]** |
| A | right arm | 211 | 301 | **+89 [+15, +191]** | +50 | +39 | 0.56 | +0.4 [−1.4, +2.1] |
| A | right leg | 172 | 265 | **+93 [+3, +198]** | +84 | +9 | 0.91 | +2.6 [−0.0, +5.0] |
| B | left arm | 261 | 400 | **+139** | +116 | +23 | 0.83 | **+3.4** |
| B | right arm | 282 | 406 | **+124** | +110 | +14 | 0.89 | **+2.4** |
| C | left arm | 1,892 | 1,172 | **−720** | −682 | −39 | 0.95 | +10.6 (G1 adds needed depth) |

Over-depth = abs(predicted depth) − abs(GT depth).

- **Depth magnitude.** Both models over-predict depth magnitude near the plane: left arm regime A mean abs depth is G0 9.6 and G1 12.6 %L, against GT 5.1. G1 predicts more depth in every regime: I 15 → 20 and C 24 → 35, against GT 28 and 54. Out-of-plane, that is exactly the needed correction. Near-plane, it adds false depth.
- **Wrong-sign behaviour.** In non-degenerate regimes G1 is not worse than G0. On left-arm I rows it is 16% vs G0 22%; on C rows 5.5% vs 8.6%. False depth near-plane is about magnitude, not sign.
- **Middle joint:** A+B penalty on the left arm +35 (%L)², 77% depth.
- **Exact distal split** (|D_err|² = |M_err|² + |S_err|² + 2 M·S): left arm A, G1 vs G0:
  - middle part +43, segment part +17, cross term +75 (−61 → +14).
  - G1 loses G0's error cancellation between the elbow error and the forearm-vector error.

## 4. Connected-geometry factor attribution (evaluation-only 2³ hybrids)

Each hybrid takes the proximal direction u, the distal direction v and the ratio ρ from G0 or G1. The hybrids are valid unit chains used for accounting, never deployable.

Single-switch effect on the regime-A distal penalty (3DPW TEST, (%L)²):

| chain | total | u | v | ρ | interaction |
|---|---:|---:|---:|---:|---:|
| left arm | +136 | +56 | +55 | +1 | +24 |
| right arm | +89 | −9 | **+76** | +1 | +21 |
| right leg | +93 | +41 | **+61** | −1 | −8 |

- **Ratio ρ never owns error** (ρ error 0.01 for both models).
- **No single factor owns the left-arm penalty.** u and v contribute equally, with a positive interaction.
- **For the right arm and leg the distal direction v owns 85% and 66%,** consistent with Worklog 81: static gains and penalties need not separate cleanly by factor.
- **Regime B on the left arm is u-dominated** (+107 of +139), the opposite of the right arm (v +68).
- **There is no consistent, chain-independent geometry owner.**

## 5. Observation and supervision attribution

**Detector vs GT 2D discrepancy** (`discrepancy/discrepancy.npz` `62229379…`). GT 2D comes from three sources: 3DPW world joints projected through the official camera, 3DHP `annot2` and BEDLAM `gtkps`. Median discrepancy as a fraction of the GT limb-box diagonal:

| population | wrists / elbows | knees | ankles | hips |
|---|---|---|---|---|
| 3DPW | 1.0–1.2% | 1.3% | 0.9% | 3.1–3.4% (joint-definition offset) |
| 3DHP | 2.2–3.1% | 3.0–3.4% | 2.3–2.6% | 5.0–5.7% |
| BEDLAM VALIDATION | 1.9–3.3% | 2.1–2.4% | 2.4–2.8% | 5.2–5.7% |

These figures measure detector localization and annotation-definition differences together.

**Regime-A penalty split at the median discrepancy** (3DPW TEST; Δ = G1 − G0 distal squared error):

| chain | higher-discrepancy half | lower half | `observation_dominated` |
|---|---|---|---|
| left arm | +112 [+36, +186] | +159 [+77, +260] | no |
| right arm | +136 [+38, +328] | +43 [−24, +111] | **yes** |
| right leg | +121 [+6, +257] | +65 [−10, +133] | **yes** |

- Splitting on detector confidence gives no consistent pattern: right arm higher-confidence half +52 (not significant), lower half +126 (significant); right leg +100 and +86.
- **What this shows:** for two chains, the significant part of the penalty requires a less accurate observation.
- **What it does not show:** that the observation is the sole cause. The lower halves are positive but not significant, and the left arm is the counter-example.

**Supervision support** (regime shares of chain-valid rows; A / B / I / C):

| chain | 3DPW TRAIN | BEDLAM historical | BEDLAM Worklog 85 new | 3DPW TEST | 3DHP |
|---|---|---|---|---|---|
| left arm | .11 / .30 / .40 / .19 | .07 / .29 / .40 / .24 | .06 / .26 / .37 / .32 | .11 / .24 / .41 / .24 | .12 / .23 / .36 / .29 |
| right arm | .06 / .28 / .47 / .18 | .07 / .27 / .40 / .26 | .06 / .23 / .39 / .32 | .09 / .27 / .43 / .21 | .07 / .22 / .42 / .29 |
| right leg | .06 / .20 / .69 / .05 | .22 / .25 / .44 / .09 | .22 / .30 / .42 / .06 | .11 / .24 / .57 / .08 |.12 / .22 / .52 / .14 |

Near-plane configurations are not rare in training. The 3DPW TEST regime mix is close to TRAIN's.

**The decisive evidence is cross-population:**

| regime-A G1 − G0, distal sq. | 3DPW TEST | 3DPW VALIDATION | BEDLAM VALIDATION | 3DHP |
|---|---|---|---|---|
| left arm | **+136** | +7 (n.s.) | −92 (n.s.) | **+89 [+26, +165]** 5/6 |
| right arm | **+89** | **+94** | +84 (n.s.) | −66 (n.s.) |
| right leg | **+93** | −38 (n.s.) | **−54 (G1 better)** | +31 (n.s.) |

The near-plane penalty is consistent only for the right arm. For the left arm it appears on the two reused test sets but not on either validation source. For the leg it reverses on BEDLAM VALIDATION.

## 6. Temporal attribution

| 3DPW 1/6 s (5 frames) | left arm | right arm | right leg |
|---|---|---|---|
| depth-coordinate motion residual, G0 → G1 (%L per pair) | 7.72 → 10.85 | 7.83 → 10.89 | 6.62 → 8.02 |
| G1 − G0, cluster CI | +3.13 [+2.52, +3.76], 37/37 | +3.06 [+2.43, +3.78] | +1.41 [+1.02, +1.76] |
| depth motion amplitude G0 / G1 / GT | 5.1 / 10.6 / 7.6 | 5.7 / 10.7 / 7.2 | 5.2 / 7.8 / 7.0 |
| G1 − G0 depth residual within regimes A / B / I / C (same-regime pairs) | +3.5 / +3.3 / +4.3 / +4.4 | +3.1 / +3.4 / +4.0 / +4.7 | +2.1 / +1.8 / +1.5 / +2.4 |
| false depth-sign flips (stable GT sign, abs(f_chain) ≥ sin 10° in both frames), G0 → G1 | 4.1% → 5.4% | 4.6% → 7.0% | 1.2% → 2.4% |
| Spearman (near-plane static over-depth change, temporal residual change) | 0.08 | 0.07 | 0.08 |

**3DPW (1/6 s):**
- The RGB temporal penalty is about the same size in every regime and is not concentrated at near-plane frames or regime transitions.
- G1 over-moves in depth against GT, while G0 under-moves.
- False sign flips are rare and only slightly more frequent with G1.
- The static near-plane false depth and the temporal penalty are essentially uncorrelated (ρ ≈ 0.08).

**3DHP, 0.4 s (kept separate):**
- G1 has lower depth residual for both arms: left −4.2, right −3.5.
- At that interval the residual tracks static accuracy.
- The leg's change is not significant (−0.2).

**Implementation fix before interpretation.** The first attribution run built the factor hybrids from %L-scaled coordinates, which broke the unit-chain assumption of `factors()`. It was fixed in `6700c58` and rerun (`attribution_v2`). The classification rule's inputs other than the factor shares were unaffected, and the classification is the same (C). The buggy output is kept as `attribution/` (root-owned, unmodified); the valid report is `attribution_v2/report.json` `29598129…`.

## 7. Diagnostic classification — C (observation / population dominated)

| predeclared check (`DIAGNOSIS`, commit 6e52dd7) | left arm | right arm | right leg |
|---|---|---|---|
| penalty in A or B on 3DPW TEST | yes | yes | yes |
| `observation_dominated` | no | **yes** | **yes** |
| penalty(A), depth share ≥ 0.6, over-depth, and 3DPW VALIDATION A mean > 0 | **yes** (all four) | no (share 0.56, no over-depth) | no (no over-depth; validation reverses) |
| single coherent factor (≥ 60%, interaction < 25%) | no (u 41%) | yes (v 85%) | yes (v 66%) |

How the rule resolves:

1. Three chains have a penalty, so not D.
2. Two chains are observation-dominated, so **C**.
3. Diagnostic A could not have held regardless: depth over-correction is established only for the left arm. Even there, its 3DPW VALIDATION mean is near zero (+7) and BEDLAM VALIDATION is negative.
4. A B-type single-factor story (v-owned) holds for the right arm and leg. But the factor owner differs from the left arm's, and the leg's penalty itself reverses on BEDLAM VALIDATION.

**Why no intervention is justified.** A depth-specific or v-specific mechanism would target an effect that:
- is consistent on the selection-free validation sources for only one chain (the right arm);
- depends on observation quality for two chains;
- is unrelated to the temporal penalty.

Any candidate would be judged mainly on the two reused test populations where the effect appears. That is exactly the speculative-patch situation the directive excludes. There is also no runtime-legitimate signal that separates "near-plane" from "out-of-plane" other than the model's own depth estimate; a GT regime classifier or depth threshold is forbidden.

## 8. Real-scene review (diagnostic only, 3DPW TEST, frozen G0 / G1)

**Panel layout:**
- the source image, with the detector chain (runtime input) in white;
- front and side views: GT green, G0 orange, G1 cyan;
- regime, the three GT depth components, detector confidence and discrepancy, and per-model distal, middle and depth values.

**Location:**
- server: `~/animcv-output/framepose_near_plane_review/`;
- local QE copy (images only): `~/animcv-output/87_framepose_near_plane/`.

Rules are fixed in the script docstring and use frozen baselines only.

| file | case |
|---|---|
| `truly_near_plane_arm.png` / `truly_near_plane_leg.png` | median G1 regime-A rows |
| `chain_near_plane_cancellation.png` | median left-arm regime-B row |
| `strongly_out_of_plane_arm.png` / `_leg.png` | median regime-C rows |
| `large_historical_false_depth.png` | downtown_bus_00 actor1 #1957. A clean observation (confidence 0.85, discrepancy 0.015) with an arm hanging in the image plane (GT distal depth −0.1). G0 puts the wrist at +6.9 %L depth, G1 at +27.0: distal 7.6 → 27.4 %L. This is the left arm's observation-independent false depth. |
| `correct_historical_visual_depth.png` | downtown_cafe_00 actor1 #1101 (regime C, G1 better) |
| `severe_regression.png` | downtown_stairs_00 actor0 #915 (99th-percentile G1 − G0) |
| `temporal_depth_sign_instability.png` | downtown_arguing_00 actor0 #100 → #105 (G1 distal depth sign flips; GT stable) |
| `high_confidence_observation.png` / `low_confidence_observation.png` | top / bottom confidence-decile medians |

## 9. Focused tests

`tests/test_framepose_near_plane_attribution.py` (6 tests) covers:
- regime classification, including true near-plane vs cancellation;
- the exact axis decomposition, with depth = camera Y;
- the exact distal split identity;
- factor hybrids as valid unit chains, with effects summing to the total;
- runner contracts: rules declared before `main`, a parity STOP, no torch or training;
- hybrids built from unit chains (the scaling fix);
- no `src/` coupling.

Results:
- `pytest tests/test_framepose_*.py`: **210 passed, 14 skipped** locally (the skips need torch);
- **224 passed** on LabServer63 `animcv-framepose:cuda118`.

## 10. Artifact identities (LabServer63 `~/animcv-output/framepose_near_plane/`)

| artifact | sha256 |
|---|---|
| `predictions/prediction_report.json` | `c1eb2332…` |
| `predictions/predictions_{3dpw,3dhp,bedlam_validation}.npz` | `356f8cee…` / `cb9dc51e…` / `1b4d1e6a…` |
| `discrepancy/discrepancy.npz` / report | `62229379…` / `f01eb00a…` |
| `attribution_v2/report.json` / `rows.json` | `29598129…` / `58c9ced4…` |
| `attribution/report.json` (superseded: factor-scaling bug) | `1e0969b0…` |

## Architecture conclusion and recommended next bounded question

**What explains the near-plane penalty, and what does not:**
- It is explained by observation quality and population for two of three chains: it depends on detector-vs-GT disagreement and does not reproduce on the validation sources.
- It is not explained by training coverage or by a single geometry factor.
- For the left arm there is genuine, observation-independent depth inflation on 3DPW TEST and 3DHP. It does not appear on 3DPW VALIDATION or BEDLAM VALIDATION.
- A targeted single-frame mechanism is **not supported**. The batch stops without a candidate.

**Narrower questions the evidence points to (not started; each needs separate scoping):**

1. **Population-specific depth inflation (left arm).** Which measurable property of the 3DPW TEST / 3DHP sequences makes G1 inflate near-plane depth there but not on 3DPW VALIDATION / BEDLAM VALIDATION? Candidates: camera height or pitch, subject distance and crop scale, motion blur. This requires describing the sequences, not training.
2. **Global short-interval depth over-motion.** On 3DPW, G1 moves depth about 1.4× more than GT in every regime. Is that frame-to-frame variation of the frozen Qwen readout itself, measured as readout change between adjacent frames vs GT motion? That isolates a visual-feature stability question from the near-plane accuracy question.

Neither question authorizes a smoothing, gating, blending or depth-threshold mechanism.

## Limitations

- **Reused test sets.** 3DPW TEST and 3DHP are reused developmental test sets. The validation sources are smaller: 14–16 sequences for 3DPW VALIDATION, and BEDLAM VALIDATION clusters of only about 1.5 rows each.
- **3DHP has 6 clusters.**
- **The observation split halves the rows,** so the lower-half non-significance is partly a power effect.
- **Discrepancy mixes two sources:** detector error and joint-definition differences.
- **Factor hybrids are accounting devices,** not models.
