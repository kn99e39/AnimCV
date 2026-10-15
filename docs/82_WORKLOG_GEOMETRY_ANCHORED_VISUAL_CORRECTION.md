# Worklog — Geometry-Anchored Visual Correction for Connected Arm Reconstruction (2026-10-15)

**Question.** Can a frozen, stable 2D-derived connected arm (Worklog 80 J0) serve as a geometric anchor for a learned RGB-conditioned correction that keeps the arm coherent and improves the trade-off between static depth accuracy and temporal stability?

**Answer.** It improves the frontier, but the trade-off with J0 remains. **CASE C** by the predeclared rule.

**R1 (Qwen-conditioned correction of the frozen J0 arm) against each reference:**

| comparison | wrist | elbow | forearm | upper-arm motion residual |
|---|---|---|---|---|
| vs J1 | −0.53 cm [−0.89, −0.13] | −0.33 cm [−0.65, −0.001] | −0.44° (not significant) | **−0.84°/pair [−1.23, −0.43]**, 30/37 |
| vs B1 | **−1.12 cm** [−1.77, −0.53], 28/37 | **−0.86 cm** [−1.48, −0.24], 27/37 | **−2.47°** [−4.06, −0.99] | **−1.84°/pair** [−2.50, −1.26], 34/37 |
| vs J0 | **−1.34 cm** (out-of-plane −4.57 cm) | −0.55 cm [−1.07, −0.05] | **−2.55°** [−4.08, −1.13], 27/37 | +2.51°/pair [2.20, 2.82], worse in 36/37 |

- **R1 is better than J1 on every static and temporal summary.** Chain direction also improves (−0.78°), and the proximal discontinuity rate drops 1.78% → 1.42%.
- **R1 is the first learned arm to beat B1 on static elbow under the Worklog 80 criterion.** That criterion needs a sequence-cluster interval excluding 0 and ≥ 2/3 of sequences improving.
- **R1 is still much less stable than J0.** It closes only 25% of the J1 − J0 temporal gap, below the 50% the predeclared temporal criterion requires. That shortfall alone makes it CASE C.
- **The gains are RGB-specific.** R1 − R0: out-of-plane wrist −3.33 cm [−6.30, −0.31], forearm −2.45° [−3.90, −1.04], elbow −0.51 cm [−1.05, −0.02].
- **R0, the 2D-only correction of the same architecture, barely changes J0.** It gives wrist −0.41 cm (28/37 sequences, interval excludes 0) and chain −0.45°, with no elbow or forearm change and +0.30°/pair temporal cost.
- **The temporal cost comes from the correction itself.** Frame to frame, R1's correction changes by 9.7° (upper arm) and 9.9° (forearm) per pair, against R0's 2.5° and 3.9°. The anchor is steady; the RGB correction is what jitters.

**Evaluation caveat.** The 3DPW test split has guided architecture decisions since Worklog 73. It was never used for gradients or checkpoint selection, but these are controlled comparisons with the historical results, not independently validated generalization.

## Starting state and files

- **Start:** `arch/single_frame_first` at `ad6af49beb09b431f8b47040ab917fea22baa849`. Local, origin and LabServer63 were equal, and the worktree was clean.
- **Commits:**
  - `bd712b0`: correction module, training, evaluation, review export, tests
  - the commit adding this worklog
- **Files added (diagnostic only, not imported by production):**
  - `src/framepose/anchored_arm_correction.py`
  - `scripts/train_framepose_anchored_arm_correction.py`
  - `scripts/run_framepose_anchored_arm_correction_eval.py`
  - `scripts/export_framepose_anchored_arm_cases.py`
  - `tests/test_framepose_anchored_arm_correction.py`
  - this worklog
- **Review renders** use the unchanged Worklog 77 renderer.
- **Unchanged:**
  - every historical module, checkpoint and report;
  - J0/J1 (read-only; J0 is loaded frozen), E1, Qwen and caches, H0;
  - the Worklog 61 IK, the Worklog 62 reference, FrameBank and splits, BaseRig and Root Orientation;
  - production FramePose/FK/IK/Blender writer.

## Frozen anchor contract

- **J0 is the exact Worklog 80 checkpoint** (`5332b691…`, checked), run with its historical zero-visual input and never shown RGB.
- **Its forward pass reproduces the stored Worklog 80 J0 outputs** to 3.0e-8 (tolerance 1e-6).
- **Its parameters are frozen** (`requires_grad_(False)`). The state digest `94967abe…` is identical before and after training both corrections.
- **The anchor features are J0's own decoded arm:** u0, v0, ρ0, E0, W0.

## Correction architecture (declared in `REPRESENTATION` before training)

**Correction.** One network predicts c = (w_u, w_v, δ) ∈ R⁷. It is applied to the anchor arm:

    u = Rodrigues(w_u) u0,   v = Rodrigues(w_v) v0,   ρ = sigmoid(logit ρ0 + δ)
    S = 0,   E = ρ u,   W = E + (1 − ρ) v

**Coherence by construction.** The output is again one connected unit-length arm with one shoulder anchor. Chain direction, reach (|W| ≤ 1), bend side and segment directions are all derived from it through the unchanged Worklog 62 `measure_chain` → Worklog 76 `full_arm`, exactly like J0/J1. Rotations of unit vectors are defined everywhere, so the correction has no degenerate state.
- Near-straight corrected arms take the historical rest-plane side fallback: 4 R1 rows, 1 J0 row.
- There is no cap, gate, blend coefficient or correction-size threshold.

**Zero-correction identity.** The head is zero-initialized, so at initialization c = 0 and the corrected arm equals J0. Measured on every row: max 1.9e-7, float32.

**Inputs:**
- detector pair geometry (39);
- the frozen anchor (13);
- for R1, frozen Qwen readouts (4 × 4096 → shared Linear → 256).
- No E1, H0, oracle sign, GT, temporal input or previous prediction.

**Network** (319,175 parameters; R0 = R1): Linear(308 → 128) → GELU → Linear(128 → 128) → GELU → Linear(128 → 7, zero-init).

**Loss.** The Worklog 80 objective: normalized elbow + wrist Euclidean error (relative to the GT shoulder, divided by the GT total arm length), with weights 1 and 1.

**Why this is not the Worklog 78/79 owner.** The correction acts on one whole arm: both segment directions and the proportion together, under one joint-position loss. It never takes or commits a separate chain direction, reach or side, and no E1/H0 quantity enters.

**Training contract** (Worklog 73/78/80, unchanged):
- 3DPW TRAIN + BEDLAM TRAIN, exactly 50/50 per epoch;
- 11,334 draws/epoch, 200 epochs, 9,000 steps, 1,133,400 draws per source;
- AdamW 3e-4 → 1e-5 cosine, weight decay 1e-4, batch 256, seed 1337;
- selection on the lowest 3DPW validation objective.

**R0/R1 control identity (enforced):** same initialization digest, parameters, steps, draws, per-source draws and frozen anchor. The only difference is R0's exact-zero visual tensors.

## Anchor distribution audit (criterion declared before any R0/R1 result)

**Criterion.** A severe stacking mismatch exists if J0's in-sample TRAIN error is below half its validation error within one source.

| J0 normalized elbow + wrist error | TRAIN | validation | test | TRAIN / validation |
|---|---:|---:|---:|---:|
| 3DPW | 0.293 | 0.336 | 0.351 | 0.87 |
| BEDLAM | 0.431 | 0.458 | — | 0.94 |

- **No severe mismatch.** The J0 anchor seen in training is close to the held-out anchor; it is a low-capacity, 2D-only model.
- **The source-domain gap is larger than the in-sample gap.** BEDLAM is harder in both splits.

## Training curves and selection

| epoch | R0 train / 3DPW val / BEDLAM val | R1 train / 3DPW val / BEDLAM val |
|---:|---|---|
| init (= J0) | 0.380 / 0.336 / 0.458 | 0.380 / 0.336 / 0.458 |
| 9 | 0.378 / 0.339 / 0.457 | 0.314 / 0.306 / 0.427 |
| 59 | 0.372 / 0.333 / 0.458 | **0.228 / 0.273 / 0.402** (selected) |
| 89 | **0.368 / 0.324 / 0.448** (selected) | 0.204 / 0.275 / 0.401 |
| 199 | 0.365 / 0.325 / 0.452 | 0.172 / 0.277 / 0.408 |

- The selected R1 validation objective (0.273) is essentially J1's (0.275). R0 moves J0 only slightly (0.336 → 0.324).
- R1 again fits TRAIN much better than validation, as every Qwen-conditioned head in Worklogs 78–81 did.

## Gates

**Gate 1 — historical parity.** Passes.
- Population: 6,003 rows set-equal to Worklog 80, 37 sequences, 5,504 pairs identical.
- B0, B1, J0 and J1 arms: joints differ by 0. Status, side status, clamp, reach, direction and side are identical, rotations are within 1e-9, and every static and temporal metric equals Worklog 80.
- The reference arm is identical, and the Worklog 76 thresholds are unchanged.
- The stored R anchor equals Worklog 80 J0 (3.0e-8).

**Gate 2 — corrected geometry.** Passes.
- R0/R1 are known on all rows, with 0 clamps.
- The unit-arm deviation is 4.4e-16, and rig residuals ≤ 1.0e-13.
- The anchor is confirmed frozen, and the initial identity holds.

## Static geometry (6,003 rows; mean / p50 / p95 / max)

| candidate | wrist cm | elbow cm | upper ° | forearm ° | chain ° | reach err | side err ° | \|bend err\| ° | proximal / distal rot vs ref ° |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| B0 | 14.88 / 11.46 / 37.08 / 90.6 | 8.25 / 6.09 / 21.54 / 49.9 | 17.42 | 22.57 | 18.21 | 0.05 | 23.4 | 14.3 | 17.8 / 24.6 |
| B1 | 11.89 / 9.63 / 29.96 / 76.3 | 7.64 / 6.11 / 18.98 / 44.2 | 15.95 | 18.33 | 13.78 | 0.05 | 23.2 | 14.3 | 16.3 / 24.3 |
| J0 | 12.11 / 9.43 / 30.18 / 77.3 | 7.32 / 5.75 / 18.42 / 45.6 | 15.28 | 18.41 | 14.19 | 0.06 | 18.4 | 15.6 | 15.6 / 24.6 |
| J1 | 11.30 / 9.17 / 28.40 / 72.9 | 7.11 / 5.80 / 17.35 / 52.9 | 14.85 | 16.30 | 13.36 | 0.05 | 18.8 | 14.1 | 15.2 / 22.2 |
| R0 | 11.70 / 9.07 / 29.90 / 64.1 | 7.29 / 5.88 / 17.56 / 51.0 | 15.20 | 18.31 | 13.74 | 0.06 | 19.1 | 15.4 | 15.6 / 24.6 |
| **R1** | **10.77 / 8.75 / 27.34** / 67.9 | **6.78 / 5.33 / 16.75** / 46.3 | **14.14** | **15.86** | **12.58** | 0.05 | **18.1** | 14.1 | **14.5 / 21.8** |
| C111 oracle | 11.16 | 6.12 | 12.74 | 12.84 | 13.78 | 0 | 0 | 0 | 13.1 / 6.4 |
| C_GT_F oracle | 4.53 | 5.75 | 11.95 | 12.40 | 3.45 | 0.05 | 23.7 | 14.3 | 12.3 / 22.9 |

**Out-of-plane rows** (|f_GT| ≥ 0.5, 1,433 rows):

| arm | wrist (cm) | forearm (°) | elbow (cm) |
|---|---:|---:|---:|
| B1 | 17.94 | 27.68 | 11.45 |
| J0 | 20.74 | 28.10 | 12.03 |
| J1 | 16.74 | 22.32 | 10.30 |
| R0 | 19.50 | 27.46 | 11.75 |
| **R1** | **16.17** | 22.74 | **9.80** |

**Paired comparisons:**
- Intervals are frame-weighted 95%: SC = sequence-cluster bootstrap; MB10 = moving-block bootstrap with block 10. All use 2,000 replicates, seed 1337.
- The sequence-weighted mean follows the frame mean in parentheses.
- "Sequences" counts sequences that improve / get worse.

| comparison · metric | mean (seq-weighted) | SC | MB10 | sequences |
|---|---|---|---|---|
| **R1 − R0 wrist, out-of-plane** | **−3.33** | [−6.30, −0.31] | | 22 / 14 |
| R1 − R0 forearm (°) | **−2.45** (−2.16) | [−3.90, −1.04] | [−3.07, −2.01] | 25 / 12 |
| R1 − R0 elbow | **−0.51** (−0.46) | [−1.05, −0.02] | [−0.77, −0.33] | 23 / 14 |
| R1 − R0 wrist | −0.93 | [−1.93, +0.03] | [−1.29, −0.63] | 19 / 18 |
| **R1 − J0 wrist, out-of-plane** | **−4.57** | [−7.40, −1.79] | | 26 / 10 |
| R1 − J0 forearm | **−2.55** (−2.29) | [−4.08, −1.13] | [−3.16, −2.07] | 27 / 10 |
| R1 − J0 elbow | −0.55 (−0.46) | [−1.07, −0.05] | [−0.79, −0.36] | 23 / 14 |
| **R1 − J1 wrist** | −0.53 (−0.50) | [−0.89, −0.13] | [−0.76, −0.26] | 28 / 9 |
| R1 − J1 elbow | −0.33 (−0.31) | [−0.65, −0.001] | [−0.53, −0.15] | 24 / 13 |
| R1 − J1 chain (°) | −0.78 | [−1.31, −0.25] | | 28 / 9 |
| R1 − J1 forearm | −0.44 | [−1.25, +0.44] | | 20 / 17 |
| R1 − J1 wrist, out-of-plane | −0.56 | [−1.77, +0.69] | | 20 / 16 |
| **R1 − B1 elbow** | **−0.86** (−0.82) | [−1.48, −0.24] | [−1.16, −0.60] | **27 / 10** |
| R1 − B1 wrist | **−1.12** (−0.98) | [−1.77, −0.53] | | 28 / 9 |
| R1 − B1 forearm | **−2.47** | [−4.06, −0.99] | | 27 / 10 |
| R1 − B1 side error (°) | −5.10 | [−7.65, −2.57] | | 29 / 8 |
| R1 − B0 wrist / elbow | −4.11 / −1.48 | both exclude 0 | | 34/3, 26/11 |
| R0 − J0 wrist | **−0.41** | [−0.66, −0.18] | | 28 / 9 |
| R0 − J0 chain | −0.45 | [−0.90, −0.05] | | 26 / 11 |
| R0 − J0 elbow / forearm | −0.03 / −0.09 | both include 0 | | — |

**The p95 errors also improve on B1:** wrist 27.34 vs 29.96 cm, elbow 16.75 vs 18.98 cm.

**R1 − B1 elbow by sequence:**
- largest gains: `flat_guitar_01` a0 (−5.4 cm), `downtown_sitOnStairs_00` a1 (−4.1) and `flat_packBags_00` a0 (−4.1);
- worst regressions: `downtown_bar_00` a0 (+2.9), `downtown_crossStreets_00` a0 (+1.9) and `downtown_downstairs_00` a0 (+1.3).

## Temporal stability (5,504 pairs; Worklog 76–81 contract and thresholds)

| per pair | B0 | B1 | J0 | J1 | R0 | **R1** | C111 |
|---|---:|---:|---:|---:|---:|---:|---:|
| upper-arm residual (°) | 8.81 | 10.97 | **6.63** | 9.98 | 6.93 | 9.14 | 8.72 |
| proximal rotation residual (°) | 9.01 | 11.23 | 6.76 | 10.24 | 7.07 | 9.34 | 8.93 |
| forearm residual (°) | 12.95 | 12.16 | 9.45 | 12.22 | 9.67 | 11.12 | 9.11 |
| chain residual (°) | 9.72 | 10.94 | 7.88 | 11.36 | 8.25 | 10.54 | 10.94 |
| elbow motion residual (cm) | 4.27 | 5.32 | 3.21 | 4.84 | 3.36 | 4.43 | 4.23 |
| wrist motion residual (cm) | 7.77 | 8.08 | 5.79 | 8.17 | 5.93 | 7.44 | 7.75 |
| reach change (GT 0.023) | 0.039 | 0.039 | 0.035 | 0.042 | 0.036 | 0.039 | 0.023 |
| side change (°) | 16.9 | 17.6 | 13.1 | 16.8 | 13.3 | 16.5 | 13.3 |
| proximal discontinuity rate | 1.29% | 3.12% | 0.58% | 1.78% | 0.73% | 1.42% | 1.78% |
| elbow discontinuity rate | 1.34% | 3.22% | 0.60% | 1.82% | 0.76% | 1.36% | 1.93% |

| comparison | upper-arm residual (°/pair) | proximal discontinuity | elbow residual (cm) | wrist residual (cm) |
|---|---|---|---|---|
| **R1 − J1** | **−0.84 [−1.23, −0.43]**, 30/37 | −0.4 pp [−0.9, +0.3] | −0.41 [−0.60, −0.21] | −0.73, 33/37 |
| **R1 − B1** | **−1.84 [−2.50, −1.26]**, 34/37 | −1.7 pp [−2.7, −0.8] | −0.89 | −0.64 |
| R1 − B0 | +0.33 [−0.21, +0.88] (not significant) | +0.1 pp (not significant) | +0.16 (not significant) | −0.32 (not significant) |
| **R1 − J0** | **+2.51 [+2.20, +2.82]**, worse in 36/37 | +0.8 pp [+0.4, +1.3] | +1.22 | +1.65 |
| R1 − R0 | +2.21 [+1.89, +2.51], worse in 37/37 | +0.7 pp | +1.07 | +1.51 |
| R0 − J0 | +0.30 [+0.19, +0.45], worse in 32/37 | +0.1 pp | +0.15 | +0.14 |

**R1 sits between J1 and J0 on temporal error.**
- It beats J1 and B1 decisively, and is as steady as H0 (B0). It closes 25% of the J1 → J0 gap.
- Wherever it is evaluated, the RGB correction adds about +2.2°/pair of upper-arm motion error on top of the anchor.

## Correction magnitude and geometry accounting

| correction (normalized arm units) | R0 mean / p50 / p95 | R1 mean / p50 / p95 |
|---|---|---|
| upper-arm rotation (°) | 5.7 / 4.3 / 14.3 | 13.4 / 11.2 / 30.9 |
| forearm rotation (°) | 12.4 / 10.6 / 25.0 | 14.3 / 12.5 / 31.9 |
| upper-share change | −0.002 / −0.001 / +0.008 | 0.000 / +0.001 / +0.010 |
| wrist displacement | 0.048 / 0.037 / 0.118 | 0.143 / 0.113 / 0.371 |
| elbow displacement | 0.029 / 0.023 / 0.075 | 0.091 / 0.069 / 0.238 |
| cos(wrist displacement, needed displacement) | 0.20 mean, 0.43 median | 0.34 mean, **0.69 median** |
| normalized wrist error change | −0.007 | −0.024 |
| normalized elbow error change | −0.001 | −0.010 |
| share of rows where the correction reduces wrist / elbow error | 57.6% / 55.7% | 55.6% / 54.4% |
| **adjacent-frame change of the correction:** upper / forearm rotation (°/pair) | 2.5 / 3.9 | **9.7 / 9.9** |

**How the correction behaves:**
- **R1 corrects mostly through the upper-arm direction:** 13.4° on average, against 5.7° for R0. The proportion change is negligible, so the learned correction almost never changes source proportions.
- **The correction points the right way in most rows** (median cosine with the needed wrist displacement 0.69). It helps on average, but it worsens the arm in about 45% of rows.
- **It is largest where depth evidence matters.** In out-of-plane rows it rotates the upper arm 18.2° and moves the wrist 0.20, reducing normalized wrist error by 0.083. In near-plane rows it rotates 11.2° and slightly increases wrist error (+0.015).
- **The correction itself jumps 9.7–9.9° between adjacent frames.** That is the measured source of R1's extra temporal error over the steady anchor.

## Regimes (sequences improved)

| regime | elbow B1 / J0 / J1 / R0 / R1 | R1 − B1 elbow | R1 − J0 wrist | upper-arm residual J0 / J1 / R1 | proximal discontinuities J0 / J1 / R1 |
|---|---|---|---|---|---|
| near-straight | 5.30 / 4.99 / 5.45 / **4.85** / 5.32 | +0.03 (20/37) | −0.17 | 4.30 / 6.11 / 6.12 | 0.0 / 0.2 / 0.4% |
| moderate | **6.75** / 7.62 / 7.17 / 7.30 / 7.10 | **+0.35 (17/37)** | −1.17 | 6.28 / 9.08 / 8.65 | 0.2 / 0.7 / 1.2% |
| strong | 8.80 / 8.48 / 7.69 / 8.62 / **6.97** | **−1.82 (24/37)** | **−3.19** | 6.43 / 10.45 / 9.68 | 1.1 / 2.3 / 2.2% |
| near-folded | 11.28 / **6.67** / 7.72 / 7.42 / 6.99 | **−4.28 (26/31)** | +0.52 | 6.09 / 12.59 / 9.50 | 0.3 / 3.3 / **0.9%** |
| near-plane | 5.65 / 5.09 / 5.62 / 5.09 / 5.34 | −0.31 (24/37) | **+0.84 (11/37 improved)** | 5.05 / 7.95 / 7.50 | 0.0 / 0.5 / 0.1% |
| intermediate | 7.13 / 6.49 / 6.54 / 6.58 / 6.25 | −0.88 (23/37) | −1.34 | 5.64 / 9.42 / 8.50 | 0.4 / 1.5 / 1.2% |
| out-of-plane | 11.45 / 12.03 / 10.30 / 11.75 / **9.80** | **−1.65 (28/36)** | **−4.57** | 7.28 / 11.24 / 10.29 | 1.9 / 2.6 / 3.4% |

**Regime by regime:**
- **Near-folded.** J0's strength largely survives: R1 6.99 cm against J0's 6.67 and J1's 7.72, with discontinuities at 0.9% against J1's 3.3%.
- **Out-of-plane.** J1's advantage is kept and extended: R1's elbow is the best of all, 9.80 cm.
- **Near-plane.** J1's false-depth regression is reduced but not removed: R1 wrist +0.84 cm vs J0, 11/37 sequences improved.
- **Moderate bend.** The weakness shared by J0 and J1 persists: R1 elbow 7.10 vs B1 6.75, 17/37 sequences improved.

**Temporal failure sequences.** The Worklog 81 temporal-failure sequence `downtown_arguing_00` a0 is R1's worst vs J1 (+2.96°/pair), followed by `downtown_runForBus_00` a0 (+1.19).

## Owner frames (wrist / elbow cm)

| row | B1 | J0 | J1 | R0 | R1 | R1 correction (upper / forearm) |
|---|---|---|---|---|---|---|
| `downtown_bar_00` a0 #514 | 26.1 / 29.6 | 9.1 / 11.3 | 12.6 / 9.2 | 13.8 / 15.5 | 11.1 / **6.7** | 9° / 24° |
| `downtown_bar_00` a0 #775 | 8.3 / 4.5 | 25.0 / 26.2 | 24.1 / 21.8 | 20.0 / 21.5 | 27.0 / 26.5 | 14° / 6° |
| `downtown_runForBus_01` a0 #185 | 9.1 / 7.1 | 27.0 / 15.8 | 7.9 / 6.2 | 24.7 / 15.5 | 11.4 / 9.9 | 13° / 32° |
| `downtown_sitOnStairs_00` a0 #325 | 67.7 / 42.4 | 31.9 / 14.6 | 65.8 / 36.6 | 29.7 / 15.9 | 55.6 / 28.5 | 31° / 27° |
| `flat_packBags_00` a0 #446 | 30.2 / 25.4 | 5.3 / 11.3 | 11.2 / 9.8 | 10.3 / 10.0 | **5.0** / 12.5 | 9° / 10° |

- **#185.** R1 pulls J0's wrong direction most of the way back. **#325.** R1 inherits part of J1's false depth (28.5 cm, against J0's 14.6). **#775.** R1 does not fix J0's direction failure.
- **Strongly out-of-plane review example** (`outdoors_fencing_01` a0 #175). R1 rotates J0's upper arm by 90° and fixes a depth-sign error: wrist 45.3 → 8.1 cm.

## Validation (selection split; separate)

| arm | elbow (cm) | wrist (cm) | forearm (°) |
|---|---:|---:|---:|
| B1 | 6.51 | 9.96 | 15.7 |
| J0 | 6.91 | 11.54 | 17.9 |
| J1 | 5.43 | 9.67 | 14.7 |
| R0 | 6.76 | 11.08 | 17.3 |
| R1 | 5.56 | 9.46 | 14.6 |

On validation, R1 ≈ J1 on the elbow (+0.13). On test, R1 beats J1 (−0.33), so for R1 the test result is not weaker than validation; this differs from Worklogs 78–80.

## Real-scene Blender review exports

- **Renders:** authoritative BaseRig rest, Worklog 76–81 cameras, macOS Blender 5.1.2. Blender FK end vs export: 9.5e-7 cm.
- **Drawn arms:**
  - J0 orange, J1 magenta;
  - R0 blue, R1 cyan;
  - GT green, E_GT yellow.
- **Captions:**
  - sample ID, GT bend and GT f;
  - B1/J0/J1/R0/R1 elbow, wrist and forearm errors;
  - R0/R1 correction rotations;
  - the next pair's upper-arm residual.

| export | row(s) |
|---|---|
| `owner_downtown_bar_00_514`, `_775`, `owner_downtown_sitOnStairs_00_325`, `owner_flat_packBags_00_446`, `owner_downtown_runForBus_01_185` | owner frames |
| `largest_r1_static_improvement` / `_regression` (R1 vs J0, elbow + wrist) | `outdoors_fencing_01` a0 #130 / #514 |
| `largest_r1_temporal_improvement_strip` / `_regression_strip` (R1 vs J1) | `downtown_runForBus_01` a1 #578… / `downtown_arguing_00` a0 #260… |
| `strongly_out_of_plane` | `outdoors_fencing_01` a0 #175 |
| `near_plane_failure` | `downtown_crossStreets_00` a1 #115 |
| `moderate_bend_failure` | `downtown_upstairs_00` a0 #477 |
| `near_folded` | `downtown_bus_00` a1 #800 |

- **QE images-only copy:** `~/animcv-output/82_anchored_arm_correction_review/` (40 PNGs).
- **Server copy:** `~/animcv-output/framepose_anchored_arm_correction_review/`. The evaluation directory is root-owned, so the renders are kept beside it.
- The renders illustrate only.

## Architecture classification — CASE C (static/temporal trade-off persists, with an improved frontier)

| predeclared criterion (`decision_inputs`) | result |
|---|---|
| static distal: R1 − J0 out-of-plane wrist and forearm < 0, intervals exclude 0 | ✔ (−4.57 cm; −2.55°) |
| RGB-specific: R1 − R0 out-of-plane wrist or forearm < 0, interval excludes 0 | ✔ (−3.33 cm; −2.45°) |
| temporal vs J1: R1 − J1 upper-arm residual < 0 (interval) AND ≥ 50% of the J1 − J0 gap closed | ✘ (−0.84°/pair is significant, but only 25% of the gap is closed) |
| elbow vs B1: mean ≤ 0 | ✔ (−0.86 cm) |
| p95 (wrist and elbow ≤ B1 + 1.0 cm) | ✔ (both better than B1) |
| distributed (R1 − J0 forearm improves in ≥ 2/3 of sequences) | ✔ (27/37) |
| validity | ✔ |
| R0 useful | ✔ (wrist −0.41 cm), but RGB-specific also holds, so not CASE B |

**Mechanical case: C.** Static depth improves, and RGB is genuinely responsible. R1 materially reduces J1's temporal error, but stays far less stable than J0.

**What the result supports:**
- **The anchored correction improves the frontier.** R1 is better than J1 on static and temporal error, and better than B1 on elbow, wrist, forearm and temporal stability. It is the first learned arm in this lineage to clear B1 on static elbow with a sequence-robust interval. On temporal stability it is equivalent to H0's own arm (B0).
- **What has not been reached is J0's stability.** The RGB-conditioned correction varies 9.7–9.9° per frame. The failing quantity is the temporal reliability of the learned RGB correction, not the anchor (R0 adds only +0.30°/pair) and not the coherence contract.

**Answer to the completion question.** A frozen 2D-derived connected arm is a useful anchor: geometry stays coherent and RGB corrections are useful. The frontier moves (R1 dominates J1 and B1). The static-depth vs temporal-stability trade-off is not resolved against J0, and the remaining limitation is frame-to-frame unreliability of the RGB-conditioned correction.

Geometry-anchored visual correction is preserved as an architecture candidate needing independent evaluation, not as a solution. Nothing is integrated. No smoothing, temporal model, gate or tuning follows automatically.

## Focused tests

- **`tests/test_framepose_anchored_arm_correction.py` (6 tests):**
  - zero correction equals the anchor;
  - any correction gives one connected unit-length arm;
  - Rodrigues is a rotation and numpy matches torch, including small angles;
  - a perfect correction reproduces the Worklog 62 reference arm through IK/FK, with fixed bone lengths and quaternion norms;
  - R0 zeroes only visual input; R0/R1 have identical initialization and parameter count;
  - the zero-initialized head gives exact identity yet is trainable;
  - static contracts: J0 hash and reproduction, frozen J0, audit criterion declared before training, the anchor never receives RGB, no E1/H0, no production imports, evaluation gate order and decisions declared before `main`.
- **Server run (`animcv-framepose:cuda118`):** with the Worklog 61/62/73–81 focused tests, **91 passed**.
- **Full regression:** not run, because production behavior is unchanged.

## Artifact identities

**Training (`~/animcv-output/framepose_anchored_arm_correction/`):**

| file | sha256 |
|---|---|
| `training_report.json` | `4bbc2d4d…` |
| `predictions_3dpw.npz` | `a8bfbaea…` |
| `R0_GEOMETRY_CORRECTION.pt` | `2f015d67…` |
| `R1_VISUAL_CORRECTION.pt` | `87377773…` |

**Evaluation (`~/animcv-output/framepose_anchored_arm_correction_eval/`):**

| file | sha256 |
|---|---|
| `report.json` | `f8530345…` |
| `rows_test.json` | `eb401098…` |
| `pairs_test.json` | `f69db7c5…` |
| `review_cases.json` | `6ffa49f8…` |
| render `report.json` | `4bd2b51d…` |

**Pinned inputs:**

| input | sha256 |
|---|---|
| J0 checkpoint | `5332b691…` |
| Worklog 80 predictions | `eaf3f57c…` |
| Worklog 80 evaluation report | `a27eeab2…` |
| Worklog 80 test rows | `8835e79a…` |
| Worklog 80 test pairs | `b25e7c97…` |
| Worklog 81 report | `ff1564d3…` |
| Worklog 76 report | `dad46288…` |

## Limitation — repeated 3DPW test use

- **No choice was made from Worklog 82 test results.** The test split has informed architecture decisions since Worklog 73. Representation, loss, audit and decision rules were committed in code before training and evaluation, and no checkpoint or hyperparameter was chosen on test.
- **What the evidence supports.** This is controlled comparison evidence against historical baselines. It is not independent generalization; an untouched evaluation source is required before claiming that.
