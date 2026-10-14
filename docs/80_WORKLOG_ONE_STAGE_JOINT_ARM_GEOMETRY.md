# Worklog — One-Stage Joint 3D Arm Geometry Viability (2026-10-14)

**Question.** Can one single-frame model that predicts one connected left arm beat the historical E1-direction + H0-reach/bend arm (B1)? It must win on both elbow accuracy and upper-arm temporal stability, without giving up the wrist and forearm advantages.

**Answer.** Not on both under the predeclared criteria. **CASE D** (the predeclared "otherwise" bucket): viability is not established. This is not a "both fail" result, though. The one-stage joint representation is the first learned arm in Worklogs 76–80 that is temporally steadier than B1 and better than every earlier learned owner.

**J1 (2D + Qwen) vs B1:**

| quantity | change | interval / sequences |
|---|---|---|
| upper-arm motion residual | **−0.99°/pair** | sequence-cluster [−1.68, −0.36], 26/37 sequences |
| proximal discontinuities | **3.12% → 1.78%** | interval excludes 0 |
| forearm direction | **−2.0°** | [−3.2, −0.8], 27/37 |
| near-folded elbow | **−3.6 cm** | 23/31 sequences |
| static elbow | −0.53 cm | sequence-cluster [−1.26, +0.19] includes 0; MB10 [−0.82, −0.26] excludes 0; only 20/37 sequences |
| wrist | −0.59 cm | not significant |

- **J1 fails only the static-elbow criterion.** That criterion needs the sequence-cluster interval to exclude 0 and improvement in ≥ 2/3 of sequences.
- **J0 (2D only)** is the steadiest arm ever measured here: upper-arm residual 6.63°/pair, against B0 8.81 and B1 10.97, better than B1 in 37/37 sequences, with proximal discontinuities of 0.58%. Its static elbow change vs B1 is −0.32 cm (not significant), and its wrist is +0.22 cm.
- **Static elbow gains are regime-split.**
  - Large in near-folded (J0 6.67 cm, J1 7.72 cm, against B1 11.28 and the oracle C111 7.03) and in strong bends (J1 −1.1 cm).
  - Worse in moderate bends: J1 +0.42, J0 +0.86 cm.
- **The limiting factor is chain direction.** The evaluation-only GT decomposition shows that replacing J1's direction with GT removes 2.2 cm of its 7.1 cm elbow error. Replacing reach removes 0.9 cm, side 0.2 cm. The one-stage model's chain direction is only E1-level: 13.4° for J1 vs E1's 13.8°.
- **RGB.** Frozen Qwen helps static forearm (J1 − J0 −2.1°, interval excludes 0), wrist and out-of-plane arms. It does not help the elbow (−0.21 cm, not significant), and it costs temporal stability heavily (+3.4°/pair, worse in 37/37 sequences). The predeclared "vision contributes" test (elbow or upper-arm residual) is not met.

**Evaluation caveat.** The 3DPW test split has guided architecture decisions since Worklog 73. It was never used for gradients or checkpoint selection, but it is no longer an untouched final-selection set. These are controlled comparisons with the historical results, not independently validated generalization.

## Starting state and files

- **Start:** `arch/single_frame_first` at `9b4d2456a9c1f10efcd423cf127e18cfe3f5992f`. Local, origin and LabServer63 were equal, and the worktree was clean.
- **Commits:**
  - `1d3db38`: model, training, evaluation, review export, tests
  - the commit adding this worklog
- **Files added (diagnostic only, not imported by production):**
  - `src/framepose/joint_arm_geometry.py`
  - `scripts/train_framepose_joint_arm_geometry.py`
  - `scripts/run_framepose_joint_arm_geometry_eval.py`
  - `scripts/export_framepose_joint_arm_cases.py`
  - `tests/test_framepose_joint_arm_geometry.py`
  - this worklog
- **Review renders** use the unchanged Worklog 77 Blender renderer.
- **Unchanged:**
  - every historical module, checkpoint and report (Worklogs 60–79), including E1, the Worklog 78/79 owners and the Worklog 79 folds;
  - Qwen and its caches, frozen H0, Worklog 61 IK, the Worklog 62 reference, FrameBank and splits;
  - BaseRig and Root Orientation;
  - production FramePose, FK, IK and the Blender writer.
- **The sequential E1 → reach/bend owner is not touched.**

## Joint representation and rationale (declared in `framepose.joint_arm_geometry.REPRESENTATION` before training)

**Prediction.** One network predicts three things in the canonical camera frame (+X right, +Y depth, +Z up):
- u, a free 3-vector normalized to the upper-arm unit direction;
- v, likewise for the forearm;
- ρ = sigmoid(z), the source upper-arm share of total arm length.

**One connected arm, built by construction:**

    S = 0,  E = ρ u,  W = ρ u + (1 − ρ) v     (single shoulder anchor, total source length 1)

**Derived quantities, all from that one configuration:**
- chain direction = W/|W|;
- reach = |W|, which is ≤ 1 by the triangle inequality, so it is always valid;
- bend side = the Worklog 61 `observed_bend_side` of (E, W − E);
- upper/forearm directions = u, v.

**At evaluation** the three points go through exactly the path H0's joints take: the Worklog 62 `measure_chain(S, E, W, yaw_H0, A)`, then the Worklog 76 `full_arm(measure, measure.direction)`. That means the same BaseRig root, L1/L2, alignment, Worklog 61 solver, clamps and zero-twist rotations. Source proportions (ρ) never touch the BaseRig lengths.

**Degeneracy.**
- Near-straight arms (bend sine < sin 1°) keep direction and reach and take the historical rest-plane side fallback, like any degenerate source.
- Only |W| = 0 would be unavailable. It did not occur.

**Why this is not the sequential contract.** Direction, reach and side are neither predicted separately nor conditioned on a committed direction. No E1 prediction is computed or consumed. Everything comes from one predicted connected configuration trained with one joint-position objective.

**Inputs (runtime only):**
- the Worklog 68/69 detector pair geometry in the person crop (3 × 13);
- for J1, the frozen Qwen3-VL readouts (S, E, W, global tokens × 4096) through a shared Linear(4096 → 64);
- no E1, H0, oracle torso sign, GT or temporal input.

**Network** (317,511 parameters, J0 = J1): concat(39 + 256) → Linear(295 → 128) → GELU → Linear(128 → 128) → GELU → Linear(128 → 7).

**Loss.** The mean over GT-valid rows of |E_pred − E_GT| + |W_pred − W_GT|, with weights 1 and 1.
- GT positions are taken relative to the GT shoulder and divided by the GT total arm length, so the loss is scale-free.
- A correct chain direction with a misplaced elbow still costs elbow error; a test checks this.

**Validity.** A row trains when its three GT arm joints are valid and finite and both GT segments are longer than 1e-8. Near-straight and folded arms are kept, since positions are always defined.

**Training contract** (the Worklog 73/78 contract, minus E1):
- 3DPW TRAIN (10,356 eligible of 11,334) + BEDLAM TRAIN (17,451 of 21,232), exactly 50/50 per epoch;
- 11,334 draws/epoch, 200 epochs, 9,000 steps, 1,133,400 draws per source;
- AdamW 3e-4 → 1e-5 per-step cosine, weight decay 1e-4, batch 256, seed 1337;
- evaluation every 10 epochs;
- selection on the lowest 3DPW validation connected-arm objective (3,056 rows).
- The training script never touches the test split.

**J0/J1 control identity (enforced, else STOP):** same initialization digest, parameters, steps, draws and per-source draws. The only difference is J0's exact-zero visual tensors.

## Training and validation curves

Objective = normalized elbow + wrist error, in units of total arm length:

| epoch | J0 train / 3DPW val / BEDLAM val | J1 train / 3DPW val / BEDLAM val |
|---:|---|---|
| 9 | 0.426 / 0.390 / 0.503 | 0.324 / 0.307 / 0.447 |
| 49 | 0.390 / 0.354 / 0.477 | 0.235 / 0.285 / 0.405 |
| 109 | 0.381 / 0.344 / 0.459 | **0.185 / 0.275 / 0.399** (selected) |
| 139 | **0.380 / 0.336 / 0.458** (selected) | 0.167 / 0.278 / 0.408 |
| 199 | 0.378 / 0.339 / 0.460 | 0.155 / 0.277 / 0.409 |

**Selected 3DPW validation elbow / wrist / reach MAE:**

| model | elbow | wrist | reach MAE |
|---|---:|---:|---:|
| J0 | 0.126 | 0.210 | 0.051 |
| J1 | 0.099 | 0.176 | 0.038 |

**Learned ρ is nearly constant:** J0 0.479–0.527, J1 0.481–0.525. The model barely varies source proportions; the BaseRig lengths are fixed anyway.

J1 again fits TRAIN far better than validation (0.155 vs 0.277), as G1 did in Worklogs 78/79.

## Gates

**Gate 1 — data identity.** Passes.
- Frozen Qwen caches are bound to the bank digests, sample order and Worklog 68 fingerprint.
- There are no duplicate sample IDs.
- There is no TRAIN/held-out sequence overlap (3DPW, BEDLAM sequence and subject).
- On every test row, J's FrameBank training target (normalized elbow and wrist) equals the arm built from the 3DPW raw GT used by the Worklog 62 reference. Max difference 9.7e-8 on test and 8.6e-8 on validation, against a tolerance of 1e-5 (float32 vs float64).

**Gate 2 — historical parity.** Passes.
- Population: 6,003 rows set-equal to Worklog 77, 37 sequences, 5,504 pairs identical.
- B0 = C000 and B1 = C100: joints, rotations, upper/forearm errors, reference and yaw differ by at most 1.4e-14.
- Status, side status, clamp and reach are equal.
- Temporal residual means and all discontinuity rates equal Worklog 77, using Worklog 76's thresholds unchanged.

**Gate 3 — J geometry and IK contract.** Passes.
- J0, J1 and all 16 decomposition arms are known on every row, with 0 clamps.
- Rig residuals ≤ 1e-9, nothing non-finite.
- One J0 row is near-straight (rest-plane fallback); J1 has none.

## Test static geometry (6,003 rows)

| candidate | wrist cm mean / p50 / p95 / max | elbow cm mean / p50 / p95 / max | upper ° | forearm ° | chain ° | reach err | side err ° | \|bend err\| ° | proximal / distal rot vs ref ° |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| B0 (H0) | 14.88 / 11.46 / 37.08 / 90.6 | 8.25 / 6.09 / 21.54 / 49.9 | 17.42 | 22.57 | 18.21 | 0.05 | 23.4 | 14.3 | 17.8 / 24.6 |
| B1 (E1 + H0 geometry) | 11.89 / 9.63 / 29.96 / 76.3 | 7.64 / 6.11 / 18.98 / 44.2 | 15.95 | 18.33 | 13.78 | 0.05 | 23.2 | 14.3 | 16.3 / 24.3 |
| **J0 (2D)** | 12.11 / 9.43 / 30.18 / 77.3 | 7.32 / 5.75 / 18.42 / 45.6 | 15.28 | 18.41 | 14.19 | 0.06 | 18.4 | 15.6 | 15.6 / 24.6 |
| **J1 (2D + Qwen)** | **11.30** / 9.17 / **28.40** / 72.9 | **7.11** / 5.80 / **17.35** / 52.9 | 14.85 | **16.30** | 13.36 | 0.05 | 18.8 | 14.1 | 15.2 / 22.2 |
| C111 oracle (E1 + GT reach/side) | 11.16 | 6.12 | 12.74 | 12.84 | 13.78 | 0 | 0 | 0 | 13.1 / 6.4 |
| C_GT_F oracle | 4.53 | 5.75 | 11.95 | 12.40 | 3.45 | 0.05 | 23.7 | 14.3 | 12.3 / 22.9 |

**Paired comparisons:**
- Intervals are frame-weighted 95%: SC = sequence-cluster bootstrap; MB10 = moving-block bootstrap with block 10. All use 2,000 replicates, seed 1337.
- The sequence-weighted mean follows the frame mean in parentheses.
- "Sequences" counts sequences that improve / get worse.

| comparison · metric | mean (seq-weighted) | SC | MB10 | sequences |
|---|---|---|---|---|
| **J1 − B1 elbow (cm)** | −0.53 (−0.51) | [−1.26, +0.19] | [−0.82, −0.26] | 20 / 17 |
| J1 − B1 wrist | −0.59 (−0.48) | [−1.24, +0.05] | [−0.82, −0.31] | 21 / 16 |
| J1 − B1 forearm (°) | **−2.04** (−1.66) | [−3.24, −0.84] | [−2.63, −1.56] | 27 / 10 |
| J1 − B1 upper (°) | −1.10 | [−2.69, +0.46] | [−1.71, −0.52] | 20 / 17 |
| J1 − B1 chain (°) | −0.42 | [−1.27, +0.40] | | 22 / 15 |
| J1 − B1 side error (°) | **−4.41** | [−7.05, −1.98] | | 27 / 10 |
| J1 − B1 reach MAE | −0.002 | [−0.008, +0.004] | | 19 / 18 |
| **J1 − B0 elbow** | **−1.14** (−1.07) | [−2.07, −0.31] | [−1.51, −0.89] | 24 / 13 |
| J1 − B0 wrist | **−3.58** | [−5.26, −2.21] | | 30 / 7 |
| **J0 − B1 elbow** | −0.32 (−0.36) | [−1.05, +0.43] | [−0.60, +0.00] | 23 / 14 |
| J0 − B1 wrist | +0.22 | [−1.05, +1.58] | | 21 / 16 |
| J0 − B1 side error | −4.85 | [−7.32, −2.60] | | 29 / 8 |
| J0 − B0 elbow | **−0.93** | [−1.88, −0.11] | | 26 / 11 |
| **J1 − J0 elbow** | −0.21 (−0.16) | [−0.75, +0.32] | [−0.47, −0.00] | 19 / 18 |
| J1 − J0 forearm | **−2.11** | [−3.60, −0.63] | | 25 / 12 |
| J1 − J0 wrist | −0.81 | [−1.93, +0.27] | [−1.27, −0.52] | 21 / 16 |

- **J1 − B1 elbow sequences:** largest gains in `flat_guitar_01` a0 (−5.5 cm), `flat_packBags_00` a0 (−4.4) and `downtown_sitOnStairs_00` a1 (−3.7); worst regressions in `downtown_bar_00` a0 (+3.4), `downtown_bus_00` a1 (+2.3) and `downtown_crossStreets_00` a0 (+2.1).
- **J1 keeps the E1 wrist/forearm advantage over B0**, and improves the B1 p95 for both wrist (29.96 → 28.40) and elbow (18.98 → 17.35).

## Test temporal stability (5,504 pairs; Worklog 76–79 contract and thresholds)

| per pair | B0 | B1 | **J0** | **J1** | C111 | C_GT_F |
|---|---:|---:|---:|---:|---:|---:|
| upper-arm direction residual (°) | 8.81 | 10.97 | **6.63** | 9.98 | 8.72 | 7.42 |
| proximal rotation residual (°) | 9.01 | 11.23 | 6.76 | 10.24 | 8.93 | 7.59 |
| distal rotation residual (°) | 15.18 | 15.69 | 11.38 | 15.14 | 4.55 | 14.46 |
| forearm direction residual (°) | 12.95 | 12.16 | 9.45 | 12.22 | 9.11 | 8.04 |
| chain direction residual (°) | 9.72 | 10.94 | 7.88 | 11.36 | 10.94 | 8.23 |
| elbow motion residual (cm) | 4.27 | 5.32 | 3.21 | 4.84 | 4.23 | 3.60 |
| wrist motion residual (cm) | 7.77 | 8.08 | 5.79 | 8.17 | 7.75 | 2.89 |
| reach change (GT 0.023) | 0.039 | 0.039 | 0.035 | 0.042 | 0.023 | 0.039 |
| side change (°) | 16.9 | 17.6 | 13.1 | 16.8 | 13.3 | 16.5 |
| proximal discontinuity rate | 1.29% | 3.12% | **0.58%** | **1.78%** | 1.78% | 1.64% |
| elbow discontinuity rate | 1.34% | 3.22% | 0.60% | 1.82% | 1.93% | 1.65% |

| comparison | upper-arm residual (°/pair) | proximal discontinuity | elbow residual (cm) | wrist residual (cm) |
|---|---|---|---|---|
| **J1 − B1** | **−0.99 [−1.68, −0.36]**, 26/37 | **−1.34 pp [−2.2, −0.5]** | −0.48 [−0.81, −0.18] | +0.08 [−0.32, +0.47] |
| **J0 − B1** | **−4.35 [−4.97, −3.77]**, 37/37 | −2.54 pp [−3.4, −1.7] | −2.11, 37/37 | −2.29, 37/37 |
| J0 − B0 | −2.18 [−2.60, −1.77], 35/37 | −0.71 pp | −1.06 | −1.97, 37/37 |
| J1 − B0 | +1.17 [+0.67, +1.62] | +0.49 pp (interval includes 0) | +0.57 | +0.40 |
| **J1 − J0** | **+3.36 [+2.90, +3.80]**, worse in 37/37 | +1.20 pp | +1.63 | +2.37 |

- **The joint representation removes the historical mixed-owner instability.** J1 is steadier than B1. J0 is steadier than B0 and even than the depth oracle C_GT_F: its residuals are about 25% below B0's, and its framewise reach and side changes are the smallest of any non-oracle arm.
- **The Qwen-conditioned output is again the jittery one.** J1's chain-direction residual (11.4°) is higher than B1's (10.9°).

## Direction / reach / bend attribution (evaluation-only GT decomposition)

Each J-derived component is replaced by its GT value. The corners run from J (all J) to the GT reference (all GT, 0 error).

| J1 elbow error (cm) | value |
|---|---:|
| J1 (direction J, reach J, side J) | 7.11 |
| replace direction only | 4.93 (**−2.18**) |
| replace reach only | 6.17 (−0.94) |
| replace side only | 6.88 (−0.24) |
| GT reach + GT side, J direction | 5.79 |
| GT direction + GT reach, J side | 2.66 |
| J1 direction + H0 reach/side | 7.46 |

| quantity | replace direction | replace reach | replace side |
|---|---:|---:|---:|
| J1 wrist (cm) | −8.41 | −0.49 | 0 |
| J1 forearm direction (°) | −5.81 | −1.47 | −2.49 |
| J1 upper direction (°) | −4.63 | −1.99 | −0.52 |
| J0 elbow (cm) | −1.90 | −1.06 | +0.13 |

**What the decomposition shows:**
- **Coherence helps.** Reach and side are jointly better than H0's for J's own direction: J1's direction with H0 reach/side is 7.46 cm, against 7.11 cm for J1.
- **Direction is the dominant remaining error owner.** J1's chain direction (13.4°) is only marginally better than E1's (13.8°).
- **Reach comes second.** J1's reach MAE (0.05) still equals H0's.
- **Side is closest to solved,** at 18.8° error against H0's 23°.

**Against the Worklog 78/79 learned owners:**

| arm | elbow (cm) | upper-arm residual (°/pair) |
|---|---:|---:|
| J1 | 7.11 | 9.98 |
| J0 | 7.32 | 6.63 |
| G1 in-sample / OOF | 7.85 / 8.03 | 10.81 / 10.77 |
| G0 in-sample / OOF | 8.03 / 8.06 | 9.66 / 9.76 |

The one-stage joint arm is better on both axes than the closed sequential design. It still does not clear the predeclared bar against B1.

## RGB vs 2D

**Frozen Qwen helps static distal and out-of-plane geometry:**
- J1 − J0: forearm −2.1°, |bend error| −1.5°, reach MAE −0.007 (all intervals exclude 0);
- out-of-plane wrist: 16.7 cm (J1) vs 20.7 cm (J0), and B1 17.9 cm.

**It does not help the elbow,** and it degrades every temporal metric in 36–37/37 sequences. Under the predeclared test (elbow or upper-arm residual), RGB does not contribute.

**The 2D-only J0 is the more stable owner, while J1 is the more accurate distal/out-of-plane owner.** That is a trade-off, not a dominance.

## Regimes

| regime | rows / pairs | elbow B1 → J0 / J1 (C111) | J1 − B1 elbow (sequences improved) | J0 − B1 elbow | wrist B1 → J0 / J1 | upper-arm residual B1 → J0 / J1 | proximal discontinuities B1 → J0 / J1 |
|---|---|---|---|---:|---|---|---|
| near-straight GT < 15° | 931 / 557 | 5.30 → 4.99 / 5.45 (4.33) | +0.16 (16/37) | −0.30 | 8.68 → 7.78 / 7.98 | 6.17 → 4.30 / 6.11 | 0.5% → 0.0 / 0.2% |
| moderate 15–45° | 2,716 / 2,039 | 6.75 → 7.62 / 7.17 (5.87) | **+0.42 (17/37)** | **+0.86** | 11.57 → 12.03 / 11.24 | 10.15 → 6.28 / 9.08 | 1.9% → 0.2 / 0.7% |
| strong 45–90° | 1,606 / 1,202 | 8.80 → 8.48 / 7.69 (7.16) | **−1.10 (22/37)** | −0.32 | 13.81 → 15.50 / 13.17 | 11.22 → 6.43 / 10.45 | 3.6% → 1.1 / 2.3% |
| near-folded ≥ 90° | 750 / 579 | 11.28 → **6.67 / 7.72** (7.03) | **−3.56 (23/31)** | **−4.61** | 12.90 → 10.53 / 11.62 | 14.69 → 6.09 / 12.59 | 8.1% → 0.3 / 3.3% |
| near-plane \|f_GT\| < sin 10° | 2,117 / 1,525 | 5.65 → 5.09 / 5.62 (4.00) | −0.03 (20/37) | −0.56 | 8.01 → 7.05 / 8.22 | 8.09 → 5.05 / 7.95 | 1.2% → 0.0 / 0.5% |
| intermediate | 2,453 / 1,645 | 7.13 → 6.49 / 6.54 (5.93) | −0.59 (17/37) | −0.64 | 11.70 → 11.43 / 10.78 | 9.90 → 5.64 / 9.42 | 2.5% → 0.4 / 1.5% |
| out-of-plane \|f_GT\| ≥ 0.5 | 1,433 / 1,101 | 11.45 → 12.03 / 10.30 (9.58) | −1.15 (21/36) | +0.59 | 17.94 → **20.74** / 16.74 | 14.23 → 7.28 / 11.24 | 6.0% → 1.9 / 2.6% |

**Near-folded.** The historical failure is repaired. Both J models beat B1 and B0 there, and J0 even beats the GT reach/side oracle C111. On the 188 Worklog 76 near-folded conflict rows:

| arm | elbow (cm) | wrist (cm) |
|---|---:|---:|
| B0 | 9.46 | 21.74 |
| B1 | 15.03 | 12.29 |
| J0 | **6.71** | 9.56 |
| J1 | 7.21 | 10.54 |
| C111 | 5.58 | 7.16 |

J1's elbow is better than B1's in 82% of these rows.

**Moderate bends.** These are the most common rows, and both J models lose elbow accuracy there. `downtown_bar_00` a0 is the largest single regression (+6.5 cm/row for J1).

**Out-of-plane.** J0 lacks depth evidence (wrist 20.7 cm). J1 recovers it (16.7 cm).

## Owner frames (wrist cm / elbow cm / chain error ° / side error °)

| row (GT reach) | B0 | B1 | J0 | J1 | J1 with GT direction | J1 with GT reach + side |
|---|---|---|---|---|---|---|
| `downtown_bar_00` a0 #514 (0.53), GT bend 116° | 44.7 / 5.9 / 57 / 21 | 26.1 / 29.6 / 15 / 59 | 9.1 / 11.3 / 13 / 17 | 12.6 / **9.2** / 10 / 7 | 11.1 / 7.5 | 4.9 / 3.1 |
| `downtown_bar_00` a0 #775 (0.97) | 81.0 / 38.8 / 97 / 166 | 8.3 / 4.5 / 9 / 105 | 25.0 / 26.2 / 26 / 31 | 24.1 / **21.8** / 27 / 22 | 7.9 / 10.7 | 24.7 / 12.8 |
| `downtown_sitOnStairs_00` a0 #325 (0.81) | 4.9 / 5.2 / 6 / 11 | 67.7 / 42.4 / 99 / 7 | 31.9 / 14.6 / 45 / 7 | 65.8 / 36.6 / 88 / 12 | 6.0 / 7.6 | 61.5 / 35.3 |
| `flat_packBags_00` a0 #446 (0.55) | 8.1 / 3.5 / 9 / 6 | 30.2 / 25.4 / 53 / 11 | 5.3 / 11.3 / 6 / 30 | 11.2 / 9.8 / 2 / 15 | 11.1 / 9.0 | 1.0 / 0.9 |
| `downtown_runForBus_01` a0 #185 (0.87) | 28.7 / 16.1 / 37 / 8 | 9.1 / 7.1 / 10 / 17 | 27.0 / 15.8 / 35 / 2 | **7.9 / 6.2** / 9 / 4 | 3.5 / 3.3 | 7.3 / 3.7 |

- **#514 and #446** are largely repaired by J1. In #446 the E1 direction failure disappears: chain error 53° → 2°.
- **#325** is still a direction failure, a sign error J1 shares with E1 (88°). The 2D-only J0 halves it.
- **#775** is a new direction failure for both J models (26–27°), on a frame B1 had solved.

## Clean validation (selection split; separate)

| arm | elbow (cm) | wrist (cm) |
|---|---:|---:|
| B1 | 6.51 | 9.96 |
| J0 | 6.91 | 11.54 |
| J1 | 5.43 | 9.67 |

These 3,056 rows chose the J epochs, so they are selection-contaminated. J1 − B1 elbow is −1.08 cm there, twice the test effect, which is the same optimism pattern as Worklogs 78/79.

## Real-scene Blender review exports

- **Renders:** authoritative BaseRig rest (digest `8519d243…`), the Worklog 76–79 camera and object-rotation contract, macOS Blender 5.1.2. Blender FK end vs export: 9.5e-7 cm.
- **Drawn arms:**
  - B0 blue, B1 magenta;
  - J0 orange, J1 cyan;
  - GT green, E_GT yellow.
- **Captions:**
  - sequence/frame/bank and the exact sample ID;
  - GT/H0/J0/J1 reach and J's ρ;
  - B0/B1/J0/J1/C111 elbow and wrist errors;
  - chain and side errors;
  - the next pair's upper-arm residual.

| export (`review/`) | row(s) |
|---|---|
| `owner_downtown_bar_00_514` (also the near-folded conflict pick), `_775`, `owner_downtown_sitOnStairs_00_325`, `owner_flat_packBags_00_446`, `owner_downtown_runForBus_01_185` | owner frames |
| `largest_j1_elbow_improvement` / `_regression` | `downtown_walking_00` a0 #370 / `downtown_bar_00` a0 #369 |
| `largest_j1_temporal_{improvement,regression}_strip` | `downtown_stairs_00` a0 #110… / `downtown_runForBus_01` a1 #578… |
| `j1_stable_sequence_flat_guitar_01_strip` / `j1_unstable_sequence_outdoors_fencing_01_strip` | J1 − B1 sequence residual −5.40° / +1.87° per pair |

- **QE images-only copy:** `~/animcv-output/80_joint_arm_geometry_review/` (52 PNGs).
- **Server copy:** `~/animcv-output/framepose_joint_arm_geometry_eval/review/`.
- The renders show connected arms; the numbers above are the evidence.

## Architecture classification — CASE D (predeclared residual bucket)

| predeclared criterion (`decision_inputs`) | J1 | J0 |
|---|---|---|
| static elbow vs B1 (mean < 0, sequence-cluster interval excludes 0, ≥ 2/3 sequences) | ✘ (−0.53; [−1.26, +0.19]; 20/37) | ✘ (−0.32; 23/37) |
| temporal vs B1 (upper-arm residual and proximal discontinuity lower, intervals exclude 0) | ✔ | ✔ |
| wrist/forearm preserved (means and p95) | ✔ | ✔ |
| near-folded elbow ≤ B1 | ✔ | ✔ |
| coverage / rig invariants | ✔ | ✔ |
| RGB contributes (J1 − J0 elbow or upper-arm residual, interval excludes 0) | ✘ | — |

**Mechanical case: D.**
- **Not A.** The static-elbow criterion fails.
- **Not B.** J0 also fails static elbow.
- **Not C.** No candidate improves static geometry at a temporal cost. Neither buys temporal stability with a significant elbow or wrist regression. J1 improves both, but its elbow gain is not sequence-robust.

**The literal CASE D wording ("fail to improve full-arm accuracy and temporal behavior") overstates this result.** The temporal criterion is met, and the elbow moves in the right direction. The honest statement is:
- this bounded one-stage joint-geometry architecture and training contract does not establish viability under the declared bar;
- it is a partial positive: robust temporal gain, near-folded repair, distal/forearm gains with RGB;
- its static-elbow gain is regime-split and not sequence-robust;
- it is not a claim that joint 3D representations are impossible.

**Answer to the completion question: no**, not in both elbow accuracy and upper-arm stability under the declared criteria.

**The limiting factor is chain direction:**
- the evaluation-only GT decomposition assigns J1's largest elbow component to it (−2.2 of 7.1 cm);
- the one-stage model's direction is only E1-level.

**Reach is second.** It is no better than H0's (0.05 MAE).

**Single-frame information also limits it.** Moderate-bend regressions and #325/#775-type direction failures persist with or without RGB, and RGB adds per-frame jitter.

The E1 direction result and the Worklog 77 oracle attribution are preserved. No temporal model, representation tuning, fusion, gating or production step is started; any next architecture needs a new bounded decision.

## Focused tests

- **`tests/test_framepose_joint_arm_geometry.py` (8 tests):**
  - decoding gives one connected unit-length arm whose reach, direction and side equal the Worklog 61 derivations, for straight, bent and folded arms, positive and negative depth, varied bend planes and proportions;
  - any raw output is a valid arm;
  - targets are scale-free and rotation-equivariant; invalid, zero-length and NaN joints are masked;
  - a perfect predicted arm reproduces the Worklog 62 reference arm under any yaw, with IK/FK, fixed bone lengths and quaternion norms;
  - a straight arm takes the rest-plane fallback;
  - BaseRig lengths do not inherit source proportions;
  - J0 zeroes only visual inputs;
  - identical J0/J1 initialization and parameter count;
  - the loss penalizes a misplaced elbow even with a correct chain;
  - exact decomposition corners;
  - static contracts: no E1/H0/target/temporal input, no test in training, gate order, Worklog 76 thresholds, decisions declared before `main`, no production imports.
- **Server run (`animcv-framepose:cuda118`):** with the Worklog 61/62/73/74/75/76/77/78/79 focused tests, **81 passed**.
- **Full regression:** not run, because production behavior is unchanged.

## Artifact identities

**Training (`~/animcv-output/framepose_joint_arm_geometry/`):**

| file | sha256 |
|---|---|
| `training_report.json` | `5d6e974a…` |
| `predictions_3dpw.npz` | `eaf3f57c…` |
| `J0_GEOMETRY_ONLY.pt` | `5332b691…` |
| `J1_GEOMETRY_PLUS_VISION.pt` | `25f0c60f…` |

**Evaluation (`~/animcv-output/framepose_joint_arm_geometry_eval/`):**

| file | sha256 |
|---|---|
| `report.json` | `a27eeab2…` |
| `rows_test.json` | `8835e79a…` |
| `pairs_test.json` | `b25e7c97…` |
| `review_cases.json` | `0d2b90dc…` |
| `review/report.json` | `ae6a579c…` |

**Pinned inputs:**

| input | sha256 |
|---|---|
| Worklog 73 rows | `4e17bd77…` |
| Worklog 76 report | `dad46288…` |
| Worklog 77 report | `e86b7de6…` |
| Worklog 77 rows | `c39a834c…` |
| Worklog 78 evaluation report | `8f37d6cd…` |
| Worklog 79 comparison report | `6d9ef090…` |

Also checked: H0 test predictions (equal to Worklog 76's), the Qwen cache identity and the FrameBank digest.

## Limitation — repeated 3DPW test use

- **How the test split has been used.** It has informed every architecture decision since Worklog 73. It was never used for gradients or checkpoint selection, and no Worklog 80 choice was made after seeing a Worklog 80 test number: representation, loss, criteria and decision rules were committed in code before training and evaluation.
- **What that does and does not support.** Its repeated use means a positive result here would not be independently validated generalization. A genuinely untouched evaluation source would be needed for that claim. None was added in this batch.

## Reproduction

    D=/output/framepose_external_visual_supervision
    docker run --rm --gpus all -v $HOME/AnimCV:/workspace:ro -v $HOME/animcv-output:/output \
      -e PYTHONPATH=/workspace/src:/workspace/scripts -w /workspace animcv-framepose:cuda118 \
      scripts/train_framepose_joint_arm_geometry.py --bank /output/framepose/bank_3dpw_paired_v2.json \
      --cache-dir /output/framepose_qwen3vl_visual_cache --external-train $D/bedlam_train_final \
      --external-train-cache $D/qwen_train --external-val $D/bedlam_validation_final \
      --external-val-cache $D/qwen_validation --out-dir /output/framepose_joint_arm_geometry
    docker run ... -v $HOME/animcv-data/datasets/3dpw:/data/3dpw_raw:ro animcv-framepose:cuda118 \
      scripts/run_framepose_joint_arm_geometry_eval.py --rig examples/BaseRig.fbx \
      --rest examples/e2e_demo/baserig_blender_rest.json --calibration examples/e2e_demo/framepose_fk_calibration.json \
      --chains examples/e2e_demo/framepose_ik_chains.json --bank /output/framepose/bank_3dpw_paired_v2.json \
      --h0-dir /output/framepose/context_h0_v2 --threedpw-raw /data/3dpw_raw --w73-dir $D/transfer \
      --w76-dir /output/framepose_visual_full_arm_ik --w77-dir /output/framepose_visual_arm_ownership \
      --w78-eval-dir /output/framepose_arm_geometry_owner/eval --w79-comparison-dir /output/framepose_arm_geometry_oof/comparison \
      --joint-dir /output/framepose_joint_arm_geometry --out-dir /output/framepose_joint_arm_geometry_eval
    scripts/export_framepose_joint_arm_cases.py ... --eval-dir /output/framepose_joint_arm_geometry_eval \
      --out /output/framepose_joint_arm_geometry_eval/review_cases.json
    Blender -b -P scripts/review_framepose_visual_arm_ownership_blender.py -- --rig examples/BaseRig.fbx \
      --rest examples/e2e_demo/baserig_blender_rest.json --cases review_cases.json --out-dir review --caption-python python3
