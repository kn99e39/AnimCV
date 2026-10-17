# Worklog 88 — Frozen Visual Feature Dynamics and Single-Frame Depth Motion Attribution

Date: 2026-10-17
Branch: `arch/single_frame_first`
Starting HEAD: `106b19e` (Worklog 87 doc, Diagnostic C)
Type: diagnostic only. No model was trained, tuned, smoothed, gated or blended. No checkpoints changed, no production
code changed, and no data was downloaded.

## 0. Answer in one paragraph

**Diagnostic CASE A, scoped.** On 3DPW TEST at 1/6 s, frozen G1 has more distal camera-depth motion error than G0
on all three chains: left arm +3.13 %L, right arm +3.06, right leg +1.41. All intervals are sequence-cluster 95%
intervals and exclude 0.

Inside frozen G1, an exact 4-player Shapley split assigns the excess to the **frozen Qwen token content** that the
joint readouts sample. It does not go to the detector geometry pathway, to the sampling positions, or to the global
readout.

The excess is a **low/mid-GT-motion phenomenon**: G1 moves when the limb barely moves. It replicates on 3DPW
VALIDATION (both cadences) and in 3DHP's low/mid strata.

The same token-content pathway also carries G1's real advantage on high-motion pairs. There G1 tracks true depth
motion about twice as steeply as G0, and removing the visual contribution makes it worse. So the visual depth response
is **not selective**: it is not noise that could simply be removed.

Two parts of the predeclared CASE A rule are weak. The 0.2 s VALIDATION corroboration is met only vacuously, and the
stratum analysis is post hoc. Both are stated in §10 and §13.

## 1. Added files

All files are new; no historical module was modified.

| file | role |
|---|---|
| `src/framepose/visual_depth_motion_attribution.py` | exact temporal terms (`motion_terms`), GT-motion strata, exact Shapley, exact u/v/ρ split of ΔD_y |
| `scripts/run_visual_depth_motion_attribution.py` | predeclared runner: parity gate, 16-coalition frozen G1 evaluation, statistics, CASE rule |
| `scripts/summarize_visual_depth_motion_strata.py` | **post hoc**: removal effects by stratum, chain kinematics (u_y, v_y, ρ, bend, side), severe tails |
| `scripts/review_visual_depth_motion_attribution.py` | fixed-rule consecutive-frame review renderer |
| `tests/test_framepose_visual_depth_motion_attribution.py` | 7 focused tests |

Commits in this batch:

- `9bba2ee`: declaration of `DIAGNOSIS` and the CASE rules before any output.
- `a6214b5`: fixed the swapped 3DPW/3DHP prediction-hash labels. They were swapped in the Worklog 87 doc and in this
  batch's pins (glob order).
- `a23a7cc`: coalition-identity tolerance changed from exact to 1e-3 %L. The first run stopped at 1.6e-4 %L, which is
  float32 batch-composition round-off. The tolerance is 1e-5 in normalized units and still catches any wrong input
  construction.
- `6f7a3c5`: validation cadence. The first full run had only 5–7 five-frame pairs per chain on 3DPW VALIDATION (see §2),
  so that run's corroboration was discarded and re-declared before any valid validation result existed.
- The final commit adds the review, the post-hoc summary and this document.

## 2. Historical identities, reproduction gate and pair accounting

**Gate (all passed; otherwise STOP)**

- **Checkpoints, SHA-256:**
  - J1X `82fd6bcb…` (left-arm G1)
  - W86 left_arm_g0 `a7aecc6e…`
  - right_arm g0 `ae1ccea9…` / g1 `e7f220d6…`
  - right_leg g0 `5186a471…` / g1 `25b660c4…`

  All match the Worklog 87 `prediction_report` identities.
- **Worklog 87 frozen predictions:** `predictions_3dpw.npz` `cb9dc51e…`, `predictions_3dhp.npz` `356f8cee…`, and
  `attribution_v2/report.json` `295981b4…`, all pinned.
- **Frozen Qwen readouts:** for every evaluated frame, the readout recomputed from the cached W68/W83 token grid
  (`token_grid_fp16.npy`) at the detector positions in the stored 448 px crop (`crop_box_xy_side`), plus the stored
  global slot, is **bit-identical** to the historical G1 input (`readout.npz` slots for J1X, the W86 limb readouts
  otherwise).
- **Model identity:** the all-a and all-b coalitions reproduce the frozen historical G1 outputs (max 1.6e-4 %L).
- **Worklog 87 depth-motion residuals reproduced on 3DPW TEST** (R0 / R1, %L):

  | chain | reproduced | max \|diff\| |
  |---|---|---|
  | left arm | 7.717 / 10.850 | ≤ 1.8e-15 |
  | right arm | 7.833 / 10.888 | ≤ 1.8e-15 |
  | right leg | 6.616 / 8.024 | ≤ 1.8e-15 |
- **Conventions:** canonical axes X right, Y depth, Z up. Unit chain P=0, M=ρu, D=M+(1−ρ)v, normalized by GT chain
  length L. Results are reported in %L.
- **Pairs:** chronological pairs come from the FrameBank `sequence_id` / `frame_index`.
  - The step is checked per pair, and the reported `pair_frames` equals the declared step in every population.
  - Every pair needs a valid chain target in both frames.
  - No pair was dropped after construction: every pair produced all 16 coalitions, and 0 pairs had off-grid sampling
    positions.

**Populations**

The 3DPW FrameBank samples TRAIN every 2 frames, VALIDATION every 3 and TEST every 5. 5-frame validation pairs
therefore barely exist.

| population | pair spacing | role | valid frames (L / R arm / R leg) | pairs (L / R arm / R leg) | seqs |
|---|---|---|---|---|---|
| 3DPW TEST | 5 fr = 0.167 s | primary | 6003 / 6076 / 6519 of 7076 | 5504 / 5576 / 6158 | 37 |
| 3DPW VALIDATION | 6 fr = 0.2 s | declared corroboration | 3056 / 2770 / 3195 of 3407 | 2857 / 2469 / 2970 | 16 |
| 3DPW VALIDATION 0.1 s | 3 fr = 0.1 s | cadence sensitivity | same frames | 2922 / 2567 / 3038 | 16 |
| 3DHP TEST | 0.4 s (`stride_for`) | secondary, separate | — | 2307 / 2027 / 2363 | 6 |
| 3DHP TS5–TS6 outdoor | 0.16 s (`SHORT_PAIR_FRAMES`) | secondary, separate | — | 500 / 470 / 487 | 2 |

BEDLAM VALIDATION is excluded because it has no chronological pairs.

The superseded outputs are kept:

- `framepose_depth_motion/attribution/`: the first run, with 5-frame validation.
- The run stopped by coalition identity.

## 3. Exact depth-motion definitions

For a pair (a, b) and distal joint D:

- **Predicted displacement** dP = 100·(D̂_y(b) − D̂_y(a)) in %L.
- **GT displacement** dG = 100·(D_y(b) − D_y(a)).
- **Signed residual** dP − dG. **Absolute residual** |dP − dG|.
- **Amplitudes** |dP| and |dG|.
- **Over-response** |dP| − |dG|.
- **Static depth error** |100·D̂_y(a) − 100·D_y(a)|.
- **R** = mean absolute residual. **E = R1 − R0** (G1 vs G0 on identical pairs).
- The middle joint gets the same terms.

**Joint vs segment vs chain.** The distal-depth change is split exactly into three terms, using midpoint averages:

- proximal direction: ρ̄·Δu_y
- distal direction: (1−ρ̄)·Δv_y
- ratio: Δρ·(ū_y − v̄_y)

The post-hoc summary adds per-pair residuals of Δu_y, Δv_y, Δρ, middle depth, distal depth, full-chain direction depth
c_y = (D/|D|)_y and the bend angle, plus the bend-side flip rate.

**GT-motion strata.** These were declared in the module before running and are used only for evaluation grouping, never
at runtime. They use the 3D distal displacement ‖ΔD‖ (all axes):

- low < 4 %L
- mid 4–15
- high ≥ 15

False motion is defined as |dP| ≥ 5 %L in a low pair.

## 4. Frozen feature / readout dynamics (§5)

**Measure.** The per-slot relative change is ‖r_b − r_a‖ / mean(‖r_a‖, ‖r_b‖). It is computed on the actual fp16 G1
inputs, which are 4096-d Qwen readouts: P, M, D and global.

**3DPW TEST medians**

| | joint P / M / D slots | global slot |
|---|---|---|
| 3DPW TEST | 0.57 / 0.62 / 0.56 | 0.12 |
| 3DHP 0.4 s | 0.62 / 0.74 / 0.71 | 0.18 |

The joint readouts change about 5× more than the global readout.

**By GT-motion stratum** (3DPW TEST, left arm; joint-slot mean change):

| stratum | joint-slot change | detector displacement (median) |
|---|---|---|
| low | 0.475 | 12 px |
| mid | 0.611 | 28 px |
| high | 0.693 | 49 px |

The other chains are within ±0.05.

So the readouts do change less when the limb is still. Even then, about half of the vector changes over 1/6 s.

**Correlates** (Spearman, 3DPW TEST, left / right arm / leg):

| pair | ρ |
|---|---|
| joint-slot change vs detector 2D displacement | 0.42 / 0.41 / 0.39 |
| joint-slot change vs GT motion | 0.39 / 0.37 / 0.43 |
| joint-slot change vs \|visual Shapley\| | 0.26 / 0.28 / 0.35 |

**Within the low stratum**, pairs above the median feature change have larger G1 output motion than those below:

| chain | mean \|dP1\| above / below median |
|---|---|
| left arm | 9.1 / 5.9 |
| right arm | 9.0 / 5.7 |
| right leg | 5.4 / 3.2 |

All of this is **correlation**. Feature distance alone does not show geometric instability (§11).

## 5. Detector and crop dynamics (§6)

**Measures**

- Detector jitter = max over P/M/D of |Δdetector − ΔGT 2D|, divided by the GT limb-box diagonal. GT 2D comes from the
  W87 `discrepancy.npz`.
- Crop change = |log side ratio| plus the centre shift divided by the side.

**3DPW TEST, left arm**

- Joint displacement median 27.6 px, p95 110 px.
- Jitter median 0.014, p95 0.046.

**Pre-declared C test (observation-dominated).** The top jitter quartile would have to carry ≥ 60% of the excess, and
the rest would have to show no excess.

| chain | top quartile share of excess | remaining 75% of pairs |
|---|---|---|
| left arm | 14% | +3.57 [2.98, 4.18], 37/37 |
| right arm | 12% | +3.59 [3.00, 4.33], 37/37 |
| right leg | 12% | +1.65 [1.31, 1.96], 33/37 |

**C fails on every chain.**

Spearman correlations with the excess are ≈ 0 for both jitter (−0.09) and crop change (−0.03). Crop change correlates
with |position Shapley| (0.39), which is expected because moving the crop moves the sampling positions. That pathway
contributes ≈ 0 to the excess, though (§6).

## 6. Frozen G1 sensitivity: 4-player exact Shapley (§7)

**Players**, each switched from frame a to frame b:

- **geometry**: W69 pair features of the chain's detector joints
- **content**: the frozen token grid the readouts are sampled from
- **position**: the detector sampling positions in that frame's own crop, plus validity
- **global**: the stored global readout

**What was evaluated.** All 16 coalitions were run through the **frozen G1** for every pair. 14 of the 16 are
**EVALUATION-ONLY hybrids**, and their off-manifold status is recorded. They measure G1's sensitivity; they are not
observations.

**Removal effect.** For input group X, removal_X = R1 − mean|dP1 − φ_X − dG|. This is how much G1's own residual drops
when that group's within-model contribution is subtracted. It is a within-model quantity.

E is a between-model contrast (G1 vs G0, separately trained weights). The removal share = removal_X / E, which mixes the
two scales, so the removal effects with their own intervals are the primary numbers.

**3DPW TEST** (%L; frame-weighted mean, sequence-cluster 95% interval, sequences > 0):

| chain | E | visual removal | content | position | global | geometry |
|---|---|---|---|---|---|---|
| left arm | +3.13 [2.52, 3.76] 37/37 | +3.67 [2.89, 4.42] | +2.05 [1.28, 2.79] | +0.07 [−0.10, 0.22] | +0.08 [0.03, 0.12] | −0.12 [−0.26, 0.01] |
| right arm | +3.06 [2.43, 3.78] 35/37 | +3.88 [3.03, 4.79] | +2.70 [2.00, 3.40] | −0.03 [−0.20, 0.17] | +0.13 [0.06, 0.21] | −0.01 [−0.11, 0.12] |
| right leg | +1.41 [1.02, 1.76] 30/37 | +1.77 [1.27, 2.27] | +1.30 [0.90, 1.70] | +0.07 [−0.10, 0.22] | +0.13 [0.09, 0.17] | −0.34 [−0.51, −0.19] |

**Mean \|φ\|** (left arm): geometry 2.2, content 9.7, position 4.1, global 1.4.

- The visual removal exceeds E on every chain. This means that G1 with its visual change subtracted would have a smaller
  depth-motion residual than G0.
- G1's geometry pathway alone (mean |φ_geometry − dG|) is 7.18 / 7.01 / 6.25. That is below R0 on all three chains, so
  the G1 geometry mapping is not over-sensitive. The B condition (geometry-path excess ≥ 0.5) fails; the values are
  −0.17 / −0.27 / −0.26.

**Interactions.** The visual removal is larger than content + position + global separately. Content and position
interact, and Shapley splits each interaction equally between the players involved.

Example: the review case `high_feature_small_output` has content +7.2 and position −8.8, which nearly cancel.

When the subject moves in 2D, a content swap at fixed positions samples a different body location. Content and
position are therefore **not cleanly separable** for large-2D-motion pairs.

In the low stratum, where the excess is largest, the median detector displacement is only 12 px. There the "content"
attribution is mostly a change of tokens at about the same body location.

## 7. Geometry-direction and ratio accounting (§8)

**Exact split of |ΔD_y|** (3DPW TEST, mean absolute term, %L; G0 → G1):

| chain | proximal direction | distal direction | ratio |
|---|---|---|---|
| left arm | 3.00 → 5.95 | 4.19 → 7.25 | 0.06 → 0.15 |
| right arm | 3.23 → 5.82 | 4.33 → 7.42 | 0.10 → 0.13 |
| right leg | 3.20 → 4.75 | 2.56 → 6.08 | 0.03 → 0.08 |

**Post-hoc kinematic residuals** (|Δpred − ΔGT|×100, G0 → G1, 3DPW TEST):

| chain | u_y | v_y | ρ | middle | distal | chain dir c_y | bend (deg) |
|---|---|---|---|---|---|---|---|
| left arm | 8.30 → 12.52 | 10.81 → 14.10 | 0.11 → 0.28 | 4.09 → 6.25 | 7.72 → 10.85 | 8.91 → 12.70 | 6.44 → 8.65 |
| right arm | 8.35 → 12.14 | 10.81 → 14.68 | 0.18 → 0.23 | 4.13 → 6.07 | 7.83 → 10.89 | 9.23 → 12.83 | 6.35 → 8.97 |
| right leg | 9.57 → 10.68 | 9.01 → 10.98 | 0.16 → 0.22 | 4.57 → 5.11 | 6.62 → 8.02 | 6.96 → 8.56 | 6.84 → 7.88 |

The distal column re-reproduces Worklog 87 independently.

**Bend-side flips** (both frames bent ≥ 20°) are rare: GT 0.000–0.001, G0 0.000–0.003, G1 0.001–0.003.

**What this shows**

- The excess is in the **depth components of both segment directions** (u and v), and it propagates into the middle and
  distal joints and into the full-chain direction.
- ρ is a negligible term, and the excess is not caused by bend-side flips.
- This split is mathematics on G1's outputs. It does not license a factor-level intervention (§8 DO-NOT).

## 8. Low- vs high-GT-motion (§9)

**3DPW TEST**

| chain | stratum | pairs | R0 → R1 | E [CI] seq+ | amplitude GT / G0 / G1 | false-motion G0 / G1 | tracking slope G0 / G1 |
|---|---|---|---|---|---|---|---|
| left arm | low | 1574 | 2.72 → 7.50 | +4.78 [4.13, 5.38] 37/37 | 1.1 / 2.4 / 7.5 | 0.12 / 0.53 | — |
| left arm | mid | 2293 | 6.23 → 9.73 | +3.50 [3.01, 4.08] 37/37 | 5.3 / 4.3 / 9.3 | — | — |
| left arm | high | 1637 | 14.61 → 15.63 | +1.03 [0.09, 2.06] 27/37 | 17.2 / 8.8 / 15.2 | — | 0.28 / 0.51 |
| right arm | low | 1615 | 2.86 → 7.46 | +4.60 [3.86, 5.38] 36/37 | 1.1 / 2.7 / 7.3 | 0.14 / 0.51 | — |
| right arm | high | 1560 | 14.94 → 14.79 | −0.15 [−1.26, 1.23] | 16.6 / 9.7 / 14.8 | — | 0.29 / 0.50 |
| right leg | low | 2434 | 2.43 → 4.33 | +1.90 [1.46, 2.28] 35/37 | 1.0 / 2.4 / 4.3 | 0.11 / 0.30 | — |
| right leg | high | 1683 | 13.37 → 13.31 | −0.06 [−0.94, 0.81] | 17.5 / 9.3 / 13.2 | — | 0.31 / 0.46 |

**Low motion.** G1 creates false depth displacement in about half the arm pairs, about 4× G0's rate.

**High motion.** G0 under-responds badly (slope 0.28 vs 1.0 ideal; amplitude about half of GT). G1 tracks real motion
better (slope 0.5). The high-motion error is dominated by both models' under-response, not by G1 over-response.

**Post-hoc removal by stratum** (3DPW TEST; content removal [CI]):

| chain | low | mid | high |
|---|---|---|---|
| left arm | +4.47 [3.93, 4.90] | +2.74 [2.18, 3.35] | −1.23 [−2.39, 0.06] |
| right arm | +4.56 [3.84, 5.39] | +3.97 [3.37, 4.74] | −1.19 [−2.52, 0.24] |
| right leg | +2.33 [2.01, 2.64] | +2.08 [1.75, 2.46] | −1.15 [−1.91, −0.31] |

In the high stratum, subtracting the content contribution makes G1 *worse*. The same pathway carries both the false
motion and the genuine tracking.

**Severe tails** (post hoc, 3DPW TEST; G0 → G1):

| chain | p95 | p99 | rate > 20 %L |
|---|---|---|---|
| left arm | 24.4 → 30.4 | 38.0 → 46.3 | 0.085 → 0.148 |
| right arm | 24.6 → 31.2 | 41.4 → 45.2 | 0.084 → 0.155 |
| right leg | 22.4 → 25.0 | 38.1 → 41.8 | 0.065 → 0.084 |

**Static depth error** (G0 → G1): 19.6 → 16.7, 18.4 → 16.0, 11.3 → 11.6. G1 is still better per frame on the arms, as
in Worklogs 86/87.

## 9. 3DPW VALIDATION vs TEST (§10)

**Aggregate**

| chain | VAL 0.2 s E | VAL 0.2 s visual removal | VAL 0.1 s E | VAL 0.1 s visual removal |
|---|---|---|---|---|
| left arm | +0.52 [−1.69, 2.13] 10/16 | +1.18 [−1.05, 2.88] | +2.49 [1.12, 3.58] 13/16 | +3.58 [2.11, 4.83] 15/16 |
| right arm | +0.78 [−0.58, 2.04] 9/16 | +1.76 [0.11, 3.27] | +2.69 [1.74, 3.55] 16/16 | +4.10 [2.77, 5.35] 16/16 |
| right leg | +0.01 [−0.53, 0.43] 9/16 | +0.60 [0.01, 1.07] | +0.82 [0.32, 1.23] 13/16 | +1.98 [1.47, 2.41] 16/16 |

Content removal at 0.1 s: +1.90 / +2.38 / +1.26, all intervals excluding 0.

**At 0.2 s the aggregate excess is null.** It is not null within strata:

- Low/mid strata show significant excess with content removal excluding 0. For example, left arm low +3.27
  [2.40, 4.15], content removal +3.10 [2.33, 3.86].
- The high stratum, which is larger at 0.2 s (1397 of 2857 left-arm pairs), favours G1. Left arm E −2.35; right arm
  −1.86 [−3.58, −0.46].
- The aggregate is a cancellation between the two, not an absence of the phenomenon.

**Consistent across 3DPW populations:**

- low/mid false motion
- content ownership
- G1's better high-motion slope (VAL 0.60 / 0.55 / 0.60 vs G0 0.22 / 0.26 / 0.35)

**Sensitive to spacing:** the aggregate excess, because longer intervals contain more real motion.

**Sequence concentration** (3DPW TEST, left arm): the most-positive sequence `downtown_bar_00` holds 8.8% of the net
excess. The worst five sequences have mean excess 4.8–6.9 %L. The excess is positive in 37 of 37 sequences, so it is not
concentrated in one subject.

## 10. 3DHP temporal evidence, kept separate (§10)

**0.4 s (6 sequences).** G1 is better in aggregate (left arm E −4.20 [−6.34, −2.68]; right arm −3.48; leg −0.21 n.s.)
because 75% of pairs are high-motion (left arm high E −6.81).

This is not a short-interval jitter measurement. Even so, the **low/mid strata** show the same signature:

| | low E | low content removal | mid E |
|---|---|---|---|
| left arm | +3.76 | +2.27 [0.64, 3.09] | +3.69 |
| right arm | +4.73 | +3.42 [0.72, 3.91] | +4.55 |

False-motion rates in the low stratum are 0.51 / 0.45 (G1) vs 0.17 / 0.16 (G0).

**0.16 s outdoor (TS5–TS6, 2 sequences).**

- Aggregate: left arm E +0.64 n.s., right arm +1.10 n.s., leg +1.37 [1.08, 2.17].
- Mid stratum E is +2.7 / +3.1 / +1.7.
- With 2 sequences and only 5–43 low-stratum pairs, this population cannot corroborate or refute anything. It is
  reported, not used.

## 11. Real-scene review (§13)

**Output.** `/output/framepose_depth_motion/review/` (12 PNGs, `review_index.json`), copied to
`~/animcv-output/88_framepose_depth_motion/`.

**Panel layout.** Each case is a strip of **four consecutive TEST rows** (previous, a, b, next) with:

- the original RGB and the detector chain (runtime input, white)
- side views (depth × up) of GT, G0 and G1 at frames a and b
- the ±12-row distal-depth trajectory of GT, G0 and G1 (GT masked where invalid)
- per-pair joint-readout change bars
- a caption with the exact sequence, actor and frames; dG, dP0, dP1, both residuals; the four Shapley values; detector
  displacement, jitter, Δconf, crop scale/shift; and the readout change

Selection rules are fixed in the script docstring.

| case | pick | what it shows |
|---|---|---|
| large_error_low_gt_motion | cafe_00 actor1, 1131→1136 | dG −1.0, G1 +20.3. Shapley spread over geometry, position and content (detector moved 109 px, a turning walker) |
| large_legitimate_gt_motion | flat_packBags_00 330→335 | dG +25.8; G0 +15.8, G1 +10.1: both under-respond on this pair (G1 worse), though G1's trace follows the swing over the window |
| correct_g1_depth_tracking | runForBus_00 actor1, 563→568 | dG −42.8; G1 −22.5 vs G0 −15.4 (content −9.4 helps) |
| g1_false_depth_oscillation | arguing_00 actor0, 130→ | near-static arm, G1 trace zig-zags ±10–20 %L while GT is flat; content +9.1 dominates |
| detector_instability | stairs_00 725→730 | jitter 0.089, crop scale 0.25, dG +2.2: **G0** overshoots (+30.1). G1 −16.2 is driven by content (−21.4) against geometry (+10.8). The observation instability hurts both models, differently |
| crop_readout_change | walking_00 360→365 | crop scale 0.29, dG −0.8: G1 +4.0 beats G0 −8.7. Content +15.3 is offset by geometry −5.9 and position −5.7. A large crop change does not by itself produce G1 false motion |
| high_feature_small_output | walkBridge_01 1141 | readout change 0.92 but dP1 −2.2: content +7.2 and position −8.8 cancel |
| small_feature_large_output | walking_00 1313→1318 | readout change only 0.28 (bottom decile), still limb (dG −1.6), yet G1 −11.7 from content −10.7 (G0 −1.7). High model sensitivity to a small content change: the B-like mechanism exists in individual pairs |
| similar_error_both_arms | rampAndStairs_00 519 | both arms move the wrong way (left dP1 +15.2 vs dG −12.9; content +9.3) |
| right_leg_contrast | enterShop_00 170 | left arm excess ≥ 5 while the leg's G0 ≈ G1 (12.5 / 12.0) |
| contradicts_visual_hypothesis | bus_00 actor1, 751→756 | excess 8.2 %L with geometry +16.1 > \|visual\| 9.1: the geometry pathway overshoots |

**What the review shows**

- The false-motion pattern is visible as G1 zig-zag on still limbs.
- The review also contains clear counter-examples: geometry-driven overshoot (bus_00), and cases where visual content
  improves tracking (runForBus_00).
- In walking sequences (walkBridge_01), the GT arm-swing depth oscillates ±30 %L. G0 is nearly flat and G1 follows it
  only partially.
- A small readout change can produce a large G1 output change (walking_00 1313). A large crop or readout change can
  leave G1 correct (walking_00 360). Readout distance and output error are therefore only loosely coupled at the level
  of single pairs, consistent with the Spearman ρ of 0.26–0.35.

## 12. Diagnostic CASE

**Predeclared rule output** (`classify()`): **A**.

- observation_dominated: none
- visual share ≥ 0.5 on TEST: all three chains (1.17 / 1.27 / 1.26)
- visual share ≥ 0.3 on VALIDATION: all three
- sub-owner: **token content** for all three

**Weakness of the rule, stated rather than hidden.** The VALIDATION condition does not require E to be significant. At
0.2 s, E is ≈ 0 and n.s., so the shares (2.29 / 2.25 / 55.4) are ratios with near-zero denominators. That condition is
satisfied **vacuously**, and I do not rely on it.

**What does corroborate CASE A beyond 3DPW TEST:**

1. VALIDATION at 0.1 s (a declared sensitivity population): E and the visual and content removals all exclude 0 on all
   three chains, in 13–16 of 16 sequences.
2. Post hoc, the low/mid strata of 0.2 s VALIDATION and 3DHP 0.4 s show the same content-owned excess.

**Final: CASE A, scoped.**

- Frozen visual token-content dynamics are a material contributor to G1's short-interval depth motion that is not
  explained by legitimate geometry change.
- This holds in low/mid-GT-motion pairs, on both arms, and less strongly on the right leg.
- Detector/crop dynamics (C) and G1 geometry-path sensitivity (B) are not supported.

**Not CASE D?** The ownership itself is not population-dependent: content in every population. Only the aggregate sign
depends on the motion mix.

**Limitations of the ownership claim**

- Content and position interact (§6).
- The same pathway also delivers G1's genuine high-motion gain.

## 13. Supported and unsupported causal claims (§11)

**Supported** (by declared measurement type)

- *Exact mathematical attribution.* Within frozen G1, the visual Shapley component accounts for more than the G1−G0
  excess on 3DPW TEST, and so does the content component in the low/mid strata. Shapley efficiency and the
  coalition identities were checked per pair.
- *Frozen-model sensitivity.*
  - G1's distal depth responds about 4× more to the content player than to the geometry player (mean |φ| 9.7 vs 2.2,
    left arm).
  - G1's geometry pathway alone is not worse than G0.
- *Correlation.*
  - Readout change is larger when the limb moves.
  - Readout change also tracks 2D detector displacement.
  - Within still-limb pairs, larger readout change goes with larger G1 output motion.
- *Not observation-dominated:* jitter and crop change do not concentrate the excess.

**Not supported / not claimed**

- That Qwen features are "noisy" or "unstable" in a geometric sense. About half of the readout vector changes over
  1/6 s even for still limbs, but feature distance is not geometric error (§11).
- That the swapped-input hybrids occur naturally. They are off-manifold, and only their Shapley aggregation is
  interpreted.
- That removing or stabilizing the visual change would improve G1 overall. In high-motion pairs it makes G1 worse (§8),
  and no intervention was run.
- That the G1−G0 gap is caused *only* by current feature values. The models have different weights. Removal effects are
  within-G1; E is between models.
- A demonstrated causal explanation. No controlled intervention on natural inputs was performed, by design.
- The stratum-wise removal effects, tails and chain kinematics are **post hoc** and do not enter the classification.

## 14. Remaining uncertainty

- **Attribution basis.** The attribution rests on 14 off-manifold coalitions per pair. A different decomposition
  (e.g. integrated gradients) could split content/position differently. The visual-vs-geometry split is robust across
  populations; the content-vs-position split is not cleanly identifiable when 2D motion is large.
- **Reuse.** 3DPW TEST has been used repeatedly across worklogs. VALIDATION corroborates only at 0.1 s and stratum-wise
  at 0.2 s.
- **3DHP short-interval evidence** rests on 2 outdoor sequences.
- **Stratum boundaries** (4 / 15 %L) were declared but are arbitrary. The low/high contrast is large enough that nearby
  boundaries should not reverse it; this was not swept (sweeps prohibited).
- **Feature-change measure.** Relative L2 on fp16 readouts. Other measures (cosine, per-token) were not computed.
- **3DHP valid-frame counts** are reported only as pair counts.

## 15. Recommended next architecture question (not implemented)

**Can G1's depth response to visual token content be made selective?** In other words: can it keep the high-motion
tracking gain while suppressing response to token-content change that does not come with limb motion, *without* any
temporal smoothing, gating on GT-derived categories, or G0/G1 blending?

A controlled intervention would be justified only if it:

1. is predeclared against both the low-motion false-motion rate and the high-motion tracking slope as co-primary
   endpoints;
2. is evaluated on 3DPW VALIDATION at matched cadence before TEST;
3. reports the static per-frame depth accuracy, which G1 currently wins on the arms, so that it is not traded away.

## 16. Tests

`tests/test_framepose_visual_depth_motion_attribution.py`, 7 tests:

- Shapley exactness, efficiency and interaction split
- exactness of the u/v/ρ split
- motion-term and stratum definitions
- runner contracts: STOP strings, EVALUATION-ONLY flags, no optimizer/smoothing/backward, no production imports
- validation-cadence declaration
- fixed review selection rules, including the oscillation rule
- post-hoc kinematics consistency (D_y = M_y + (1−ρ)v_y) and bend-side flip detection

Results: passed locally (7/7) and on LabServer63.
