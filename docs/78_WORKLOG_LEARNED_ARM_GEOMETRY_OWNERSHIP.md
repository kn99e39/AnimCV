# Worklog — Learnable Visual-Conditioned Arm Geometry Ownership (2026-10-13)

**Question.** Hold the Worklog 73 E1 chain direction fixed. Can one learned owner, using the same runtime RGB and 2D observations, estimate reach and bend side well enough to replace H0's geometry? The goal is a more accurate and temporally stable connected left arm than B1 (E1 direction + H0 reach/side).

**Answer.** No, not under this single-frame contract. **CASE C.** The two owners fail in different ways:

- **G1 (2D + Qwen vision)** does not beat B1 on the test split.
  - Elbow change is +0.21 cm, interval [−0.36, +0.79] includes 0, and 16/37 sequences improve.
  - Upper-arm motion residual change is −0.16°/pair, interval [−0.72, +0.41] includes 0.
  - The proximal discontinuity rate is unchanged.
- **G0 (2D only)** is a trade-off, not a success.
  - It makes the arm temporally steadier than B1: upper-arm residual −1.31°/pair [−1.65, −1.02], better in 36/37 sequences, and proximal discontinuities 3.12% → 2.11%.
  - It does not improve the elbow: +0.39 cm [−0.08, +0.82], with 28/37 sequences worse.
- **The principal failure is reach.**
  - The learned reach is not more accurate than H0's (test reach MAE 0.05 for G1, G0 and H0 alike).
  - Swapped in alone, it worsens the elbow: +0.42 cm (G1 reach), +0.69 cm (G0 reach).
- **Bend side is partly learnable.**
  - The learned side lowers side error by 3.9° (G1) and 4.9° (G0), intervals excluding 0.
  - With H0 reach it slightly improves the elbow: −0.10 cm (G1 side), −0.22 cm (G0 side).
  - In near-folded arms the G1 side drives a −1.24 cm elbow gain.
- **Qwen features add per-frame noise and overfit.**
  - G1 vs G0: upper-arm residual +1.15°/pair [+0.67, +1.63], worse in 33/37 sequences.
  - Learned reach change per pair +0.009, side change +2.0°.
  - In-sample side error 5.4° rises to 19.3° on test, against 14.7° → 18.4° for G0.
  - RGB does not give a reliable static advantage on test: elbow −0.18 cm, interval includes 0.
- **The validation split looked better than test.** Validation chose the owner epochs, and there G1 − B1 elbow was −0.54 cm. Test does not confirm that.

## Starting state and files

- **Start:** `arch/single_frame_first` at `8e695526e0395990f58af7d49a632373fa485fac` (local = remote), with a clean worktree.
- **Commits:**
  - `64bc766`: owner module, training script, evaluation runner, tests
  - `9f28cdc`: review export
  - the commit adding this worklog
- **Files added (diagnostic only, not imported by production):**
  - `src/framepose/arm_geometry_owner.py`
  - `scripts/train_framepose_arm_geometry_owner.py`
  - `scripts/run_framepose_arm_geometry_owner_eval.py`
  - `scripts/export_framepose_arm_geometry_owner_cases.py`
  - `tests/test_framepose_arm_geometry_owner.py`
  - this worklog
- **Review renders** use the unchanged Worklog 77 `review_framepose_visual_arm_ownership_blender.py`.
- **Unchanged:**
  - every historical module, checkpoint, cache and report;
  - the E1 checkpoint and Qwen weights/cache;
  - frozen H0, Worklog 61 IK, the Worklog 62 reference and FrameBank splits;
  - BaseRig/calibration and Root Orientation;
  - production FramePose, FK, IK and the Blender writer.

## Frozen artifact identities

**Pinned and checked:**

| artifact | sha256 |
|---|---|
| E1 checkpoint `E1_3DPW_PLUS_EXTERNAL.pt` | `ee002f6a…` |
| Worklog 73 `report.json` | `1c05d50a…` |
| Worklog 73 `rows_heldout.json` | `4e17bd77…` |
| Worklog 76 `report.json` | `dad46288…` |
| Worklog 77 `report.json` | `e86b7de6…` |
| Worklog 77 `rows_test.json` | `c39a834c…` |

**Also checked:**
- Qwen cache identity: the Worklog 68 fingerprint and digest `1fce1ab2…`, equal to the E1 checkpoint's cache identity.
- External bank digests.
- The FrameBank digest.
- H0 test predictions, which must equal Worklog 76's.

**Outputs (`~/animcv-output/framepose_arm_geometry_owner/`):**

| file | sha256 |
|---|---|
| `training_report.json` | `cc233be4…` |
| `predictions_3dpw.npz` | `b79d892a…` |
| `G0_GEOMETRY_ONLY.pt` | `fdbd83b2…` |
| `G1_GEOMETRY_PLUS_VISION.pt` | `7dd6f0e9…` |
| `eval/report.json` | `8f37d6cd…` |
| `eval/rows_test.json` | `a47c762c…` |
| `eval/pairs_test.json` | `c7e42bc7…` |
| `eval/review_cases.json` | `7d62132f…` |
| `eval/review/report.json` | `91c75719…` |

## Architecture and output representation (declared in code before training)

Declared in `framepose.arm_geometry_owner.ARCHITECTURE` and committed with the code; none of it was changed after any result.

**Frozen E1 direction:**
- The E1 forward pass gives f_E1. The camera-frame direction is d_E1 = `reconstruct_direction(f_E1, observed_plane(detector 2D))`.
- E1 parameters are frozen (`requires_grad_(False)`) and receive no target input.

**Inputs (runtime only):**
- the Worklog 68/69 pair geometry (3 × 13 detector features in the person crop);
- f_E1 and d_E1 (4 values);
- the frozen Qwen3-VL readouts (S, E, W, global tokens × 4096) through a shared Linear(4096 → 64).
- No H0, no torso signs, no temporal input.

**Network** (317,636 parameters; G0 and G1 identical):
- concat(39 + 4 + 256) → Linear(299 → 128) → GELU → Linear(128 → 128) → GELU;
- reach head Linear(128 → 1);
- side head Linear(128 → 3).

**Outputs and how they enter the Worklog 61 IK:**
- **Reach.** r = sigmoid(z) ∈ (0, 1), the Worklog 61 dimensionless reach fraction. The Worklog 61 `solve_two_bone` clamp is the only clamp.
- **Bend side.** A free camera-frame 3-vector s, moved to rig space with the same A·Rz(−yaw_H0) as every direction. The Worklog 62 `_runtime_side` rule makes it perpendicular to the E1 rig direction. If that perpendicular part vanishes, the arm is unavailable, with no fallback.
- **IK.** The arm is the unchanged Worklog 76 `full_arm`.

**Targets** (labels only, from GT 3D in the canonical camera frame):
- **reach:** the Worklog 61 `source_chain_geometry` fraction;
- **side:** the Worklog 61 `observed_bend_side` of the GT arm, re-perpendicularized to d_E1 with the Worklog 62 rule. So a perfect prediction reproduces the Worklog 77 C111 arm, which a test confirms to 1e-9.

**Degeneracy contract (predeclared):**
- Rows whose GT arm is near-straight (bend sine < sin 1°, the Worklog 61 `DEGENERATE_BEND_SINE`) train reach only.
- Rows without a detector chain plane are not trainable.
- Nothing is removed from evaluation, and there is no GT-dependent inference fallback.

**Loss.** Masked mean |r − r_GT| plus masked mean (1 − cos(s_perp, s_GT_perp)), with weights 1 and 1.

**Training contract** (the Worklog 73 E1 contract, unchanged):
- 3DPW TRAIN + BEDLAM TRAIN, sampled exactly 50/50 per epoch over owner-eligible rows (10,356 / 17,451 rows; side-valid 10,353 / 17,449);
- 11,334 draws/epoch, 200 epochs, 9,000 steps, 2,266,800 draws (1,133,400 per source);
- AdamW 3e-4 → 1e-5 per-step cosine, weight decay 1e-4, batch 256, seed 1337;
- evaluation every 10 epochs;
- selection on the lowest 3DPW validation objective (3,056 rows).
- The training script never touches the test split.

**G0/G1 control identity (enforced, else STOP):**
- same initialization digest, parameter count, steps, draws and per-source draws;
- the only difference is that G0's visual tensor is exact zeros.

**E1 reproduction.** Re-run forward on all rows, E1 reproduces the Worklog 73 held-out chain f exactly (max diff 0, tolerance 1e-5). E1 predictions on TRAIN rows are in-sample, since E1 was trained on them; validation and test are held out.

## Training and validation curves

Objective = reach MAE + (1 − cos) side term:

| epoch | G0 train / 3DPW val / BEDLAM val | G1 train / 3DPW val / BEDLAM val |
|---:|---|---|
| 9 | 0.191 / 0.192 / 0.259 | 0.091 / 0.156 / 0.208 |
| 49 | 0.133 / 0.152 / 0.223 | 0.047 / 0.127 / 0.175 |
| 109 | 0.118 / 0.148 / 0.212 | **0.031 / 0.116 / 0.178** (selected) |
| 189 | **0.108 / 0.140 / 0.215** (selected) | 0.023 / 0.121 / 0.181 |

**Selected-epoch learnability:**

| owner | split | reach MAE | side error |
|---|---|---:|---:|
| G0 | train (in-sample) | 0.050 | 14.7° |
| G0 | 3DPW validation | 0.051 | 18.6° |
| G1 | train (in-sample) | 0.023 | 5.4° |
| G1 | 3DPW validation | 0.041 | 16.7° |

G1 fits the training rows much more tightly. Its train-to-validation gap is large, which signals overfitting.

## Historical parity (Gate 1) and output geometry (Gate 2)

**Gate 1.**
- Population: 6,003 rows set-equal to Worklog 77, 37 sequences, 5,504 adjacent pairs identical to Worklog 77.
- B0 = C000 and B1 = C100: joints and rotations differ by 0, and status, side status, clamp and reach are equal.
- Reference arm, yaw_H0 and the oracle elbows (C111, C110, C101, C_GT_F) differ by 0.
- B0/B1 discontinuity rates equal Worklog 77 exactly, using Worklog 76's jump thresholds unchanged.
- The E1 input to the owner equals the Worklog 73 chain f (diff 0).
- **Passes.**

**Gate 2.**
- Every learned and attribution arm is known on all 6,003 rows: 0 unavailable, 0 clamped, rig residuals ≤ 1e-9, nothing non-finite.
- Learned reach ranges: G0 [0.137, 0.999], G1 [0.252, 0.997].
- **Passes.**

## Test static geometry (6,003 rows)

| candidate | wrist cm mean / p50 / p95 / max | elbow cm mean / p50 / p95 / max | upper ° | forearm ° | reach err | side err ° | \|bend err\| ° | proximal / distal rot vs ref ° |
|---|---|---|---:|---:|---:|---:|---:|---|
| B0 (H0) | 14.88 / 11.46 / 37.08 / 90.6 | 8.25 / 6.09 / 21.54 / 49.9 | 17.42 | 22.57 | 0.05 | 23.4 | 14.3 | 17.8 / 24.6 |
| B1 (E1 + H0 geometry) | 11.89 / 9.63 / 29.96 / 76.3 | 7.64 / 6.11 / 18.98 / 44.2 | 15.95 | 18.33 | 0.05 | 23.2 | 14.3 | 16.3 / 24.3 |
| G0 (E1 + learned, 2D) | 11.93 / 9.54 / 29.76 / 73.6 | 8.03 / 6.58 / 19.61 / 42.1 | 16.75 | 17.97 | 0.06 | **18.4** | 16.5 | 17.2 / 24.7 |
| G1 (E1 + learned, 2D + Qwen) | 11.79 / 9.48 / 29.60 / 73.2 | 7.85 / 6.41 / 19.32 / 50.1 | 16.40 | 17.17 | 0.05 | 19.3 | 14.2 | 16.8 / 22.9 |
| C111 oracle (E1 + GT reach/side) | 11.16 | 6.12 | 12.74 | 12.84 | 0 | 0 | 0 | 13.1 / 6.4 |
| C_GT_F oracle (GT f + H0 geometry) | 4.53 | 5.75 | 11.95 | 12.40 | 0.05 | 23.7 | 14.3 | 12.3 / 22.9 |

**Paired comparisons:**
- Intervals are frame-weighted 95%: SC = sequence-cluster bootstrap; MB10 / MB25 = moving-block bootstrap with block 10 / 25.
- The sequence-weighted mean follows the frame mean in parentheses.
- "Sequences" counts sequences that improve / get worse.

| comparison · metric | mean (seq-weighted) | SC | MB10 | MB25 | sequences |
|---|---|---|---|---|---|
| **G1 − B1 elbow (cm)** | +0.21 (+0.14) | [−0.36, +0.79] | [+0.00, +0.47] | [−0.05, +0.52] | 16 / 21 |
| G1 − B1 wrist | −0.10 (−0.09) | [−0.29, +0.14] | [−0.19, −0.02] | [−0.22, −0.01] | 23 / 14 |
| G1 − B1 forearm (°) | −1.16 (−1.05) | [−2.55, +0.12] | [−1.82, −0.80] | [−2.15, −0.79] | 23 / 14 |
| G1 − B1 side error (°) | **−3.88** (−3.56) | [−6.60, −1.52] | [−4.78, −2.96] | [−5.32, −3.06] | 27 / 10 |
| **G0 − B1 elbow** | +0.39 (+0.38) | [−0.08, +0.82] | [+0.11, +0.54] | [−0.01, +0.53] | 9 / 28 |
| G0 − B1 side error | **−4.85** | [−7.36, −2.62] | | | 28 / 9 |
| **G1 − G0 elbow** | −0.18 (−0.24) | [−0.54, +0.17] | [−0.24, +0.06] | [−0.20, +0.16] | 23 / 14 |
| G1 − G0 forearm | −0.80 | [−1.48, −0.17] | | | 28 / 9 |
| G1 − B0 elbow | −0.40 (−0.42) | [−1.35, +0.46] | [−0.74, −0.12] | [−0.85, −0.07] | 18 / 19 |
| G1 − B0 wrist | −3.09 | [−4.95, −1.57] | | | 32 / 5 |

- **The E1 wrist and forearm advantages over B0 are preserved.** G1 − B1 wrist −0.10 cm; forearm −1.16°.
- **No learned owner beats B1 at the elbow.** The sequence-cluster interval always includes 0.
- **G0 regresses the elbow** in 28/37 sequences, and MB10 excludes 0 on the regression side.

## Reach vs side attribution (static, E1 direction throughout; B1 = H0 reach + H0 side)

Elbow change relative to B1 (cm) when each input is replaced:

| owner | learned reach only | learned side only | both (= G) | interaction |
|---|---:|---:|---:|---:|
| G1 | **+0.42** | −0.10 | +0.21 | −0.10 |
| G0 | **+0.69** | −0.22 | +0.39 | −0.08 |

- **Upper-arm direction:** G1 reach +0.88° and side −0.24°; G0 reach +1.45° and side −0.50°.
- **Forearm direction:** G1 reach −0.05° and side −0.49°.
- **The learned reach is the failing owner.**
  - Its reach error equals H0's (mean 0.05), but it moves the elbow the wrong way.
  - G0's |bend-angle error| grows from 14.3° to 16.5°, and near-straight GT arms get worse (G0 elbow 6.79 vs B1 5.30 cm).
- **The learned side is the partly working owner.** "H0 reach + learned side" is the best runtime arm on static elbow:

  | arm | elbow (cm) | elbow p95 (cm) |
  |---|---:|---:|
  | H0 reach + G0 side | 7.42 | 17.77 |
  | H0 reach + G1 side | 7.54 | 18.17 |
  | B1 | 7.64 | 18.98 |

  It is still far from the GT-side oracle C101 (7.12 cm) and from C111 (6.12 cm).

## Test temporal stability (5,504 pairs; Worklog 76/77 contract and thresholds)

**Mean per adjacent pair** (motion residual = candidate motion minus GT-reference motion):

| metric | B0 | B1 | G0 | G1 | C111 | C_GT_F |
|---|---:|---:|---:|---:|---:|---:|
| upper-arm direction residual (°) | 8.81 | 10.97 | **9.66** | 10.81 | 8.72 | 7.42 |
| proximal rotation residual (°) | 9.01 | 11.23 | 9.86 | 11.07 | 8.93 | 7.59 |
| distal rotation residual (°) | 15.18 | 15.69 | 12.60 | 15.69 | 4.55 | 14.46 |
| forearm direction residual (°) | 12.95 | 12.16 | 11.48 | 12.72 | 9.10 | 8.04 |
| elbow motion residual (cm) | 4.27 | 5.32 | 4.68 | 5.24 | 4.23 | 3.60 |
| wrist motion residual (cm) | 7.77 | 8.08 | 7.97 | 8.08 | 7.75 | 2.89 |
| reach change per pair (GT 0.023) | 0.039 | 0.039 | **0.030** | 0.039 | 0.023 | 0.039 |
| bend-side change per pair (°) | 16.9 | 17.6 | **14.7** | 16.7 | 13.3 | 16.5 |
| proximal discontinuity rate | 1.29% | 3.12% | **2.11%** | 2.96% | 1.78% | 1.64% |
| elbow discontinuity rate | 1.34% | 3.22% | 2.22% | 2.93% | 1.93% | 1.65% |

| comparison | upper-arm residual (°/pair) | proximal discontinuity | elbow residual (cm) | learned reach / side change |
|---|---|---|---|---|
| G1 − B1 | −0.16 [−0.72, +0.41], 23/37 sequences (82% of the net gain from `downtown_bar_00` a0) | −0.16 pp [−1.1, +0.8] | −0.08 [−0.35, +0.20] | −0.000 / −0.87° (interval includes 0) |
| **G0 − B1** | **−1.31 [−1.65, −1.02]**, 36/37 (MB10 [−1.60, −1.20]) | **−1.02 pp [−1.7, −0.4]** | **−0.64 [−0.80, −0.50]**, 36/37 | −0.009 / −2.9°, both intervals exclude 0 |
| **G1 − G0** | **+1.15 [+0.67, +1.63]**, worse in 33/37 | +0.85 pp [+0.1, +1.7] | +0.56 [+0.33, +0.79] | +0.009 / +2.0° |
| G1 − B0 | +2.00 [+1.35, +2.63] | +1.67 pp | +0.97 | — |

- **G0 is temporally steadier.** It closes 58% of the B1 → C111 upper-arm-residual gap and 76% of the discontinuity gap. Its own reach and side predictions change less per frame than H0's, though still more than GT (0.030 vs GT 0.023).
- **G1's extra RGB input makes its outputs jumpier.** Its reach changes as much per frame as H0's (0.039), and its side changes more than G0's.
- **The learned predictor itself is a source of frame-to-frame instability.** That is the predicted risk, and it is realized in G1.

## Oracle headroom

| metric | B1 | G0 | G1 | C111 (E1 + GT reach/side) | C_GT_F | share of the B1 → C111 gap closed by G0 / G1 |
|---|---:|---:|---:|---:|---:|---|
| elbow (cm) | 7.64 | 8.03 | 7.85 | 6.12 | 5.75 | −26% / −14% (worse than B1) |
| wrist (cm) | 11.89 | 11.93 | 11.79 | 11.16 | 4.53 | −6% / +13% |
| upper-arm residual (°/pair) | 10.97 | 9.66 | 10.81 | 8.72 | 7.42 | **58%** / 7% |
| proximal discontinuity rate | 3.12% | 2.11% | 2.96% | 1.78% | 1.64% | **76%** / 12% |

The GT reference arm has 0 error and 0 residual by construction.

**What stays open:**
- The whole static elbow headroom: 1.5 cm to C111, and 1.9 cm to the depth oracle.
- Most of the side error: the learned side is at 18–19°, against 0 for the oracle.
- All of the reach headroom.

## Regimes (G1 − B1 unless noted; sequences improved)

| regime | rows / pairs | elbow B1 → G0 / G1 (C111) | G1 − B1 elbow (sequences improved) | G1 − B1 wrist | G1 elbow from reach / side | upper-arm residual B1 → G0 / G1 |
|---|---|---|---|---:|---|---|
| near-straight GT < 15° | 931 / 557 | 5.30 → 6.79 / 5.37 (4.33) | +0.07 (18/37) | −0.02 | +0.21 / −0.02 | 6.17 → 6.12 / 6.31 |
| moderate 15–45° | 2,716 / 2,039 | 6.75 → 7.40 / 7.30 (5.87) | **+0.54 (17/37)** | −0.10 | +0.45 / +0.15 | 10.15 → 8.62 / 9.58 |
| strong 45–90° | 1,606 / 1,202 | 8.80 → 8.76 / 9.24 (7.16) | +0.44 (16/37) | +0.21 | +0.67 / −0.05 | 11.22 → 9.77 / 12.33 |
| near-folded ≥ 90° | 750 / 579 | 11.28 → 10.26 / **9.96** (7.03) | **−1.32 (18/31)** | −0.84 | +0.02 / **−1.24** | 14.69 → 12.56 / 13.62 |
| near-plane \|f_GT\| < sin 10° | 2,117 / 1,525 | 5.65 → 6.58 / 5.89 (4.00) | +0.24 (17/37) | +0.10 | +0.44 / −0.08 | 8.09 → 7.07 / 8.27 |
| intermediate | 2,453 / 1,645 | 7.13 → 7.28 / 7.24 (5.93) | +0.11 (18/37) | −0.05 | +0.38 / −0.10 | 9.90 → 8.65 / 10.13 |
| out-of-plane \|f_GT\| ≥ 0.5 | 1,433 / 1,101 | 11.45 → 11.45 / 11.78 (9.58) | +0.34 (16/36) | −0.47 | +0.43 / −0.14 | 14.23 → 12.52 / 13.50 |

**Near-folded is the only regime where the learned owner helps the elbow.**
- The gain comes from the learned side, not the reach.
- `downtown_walking_00` a0 supplies 59% of the net near-folded gain.
- The near-folded proximal discontinuity rate falls from 8.1% (B1) to 4.8% (G0) and 4.1% (G1).

**The Worklog 76 near-folded conflict rows** are the 188 rows where B1 improves the wrist but worsens the elbow vs B0:

| arm | elbow (cm) |
|---|---:|
| B0 | 9.46 |
| B1 | 15.03 |
| G0 | 10.35 |
| G1 | **9.79** |
| C111 | 5.58 |

- G1 also keeps the wrist gain: 10.30 cm, against B1's 12.29.
- G1's elbow is better than B1's in 70% of these rows, and no worse than B0's in 43%.
- The historical conflict is largely repaired here, but this is a small subset, and the regime-wide elbow in moderate and strong bends gets worse.

## Per-sequence

- **G1 − B1 elbow** improves in 16/37 sequences; the worst regression is `downtown_bus_00` a1.
- **G1 − B1 upper-arm residual** improves in 23/37, with `downtown_bar_00` a0 owning 82% of the net gain.
- **G0 − B1 upper-arm residual** improves in 36/37. The largest single owner is 13% (`downtown_bar_00` a0), so the gain is broad.
- **G1 − G0 upper-arm residual** is worse in 33/37.
- Per-sequence tables are in `eval/report.json → per_sequence`.

## Owner cases (wrist cm / elbow cm / side error °)

| row (GT reach / H0 / G0 / G1) | B0 | B1 | G0 | G1 | C111 |
|---|---|---|---|---|---|
| `downtown_bar_00` a0 #514 (0.53 / 0.97 / 0.88 / 0.73), GT bend 116° | 44.7 / 5.9 / 21 | 26.1 / 29.6 / 59 | 21.6 / 20.7 / 18 | **13.9 / 13.3 / 1** | 7.5 / 6.7 / 0 |
| `downtown_bar_00` a0 #775 (0.97 / 0.99 / 0.91 / 0.72) | 81.0 / 38.8 / 166 | 8.3 / 4.5 / 105 | 8.7 / 7.7 / 18 | 15.5 / **17.1** / 3 | 8.2 / 4.1 / 0 |
| `downtown_sitOnStairs_00` a0 #325 | 4.9 / 5.2 / 11 | 67.7 / 42.4 / 7 | 68.4 / 38.3 / 27 | 69.7 / 37.9 / 46 | 67.3 / 41.1 / 0 |
| `flat_packBags_00` a0 #446 | 8.1 / 3.5 / 6 | 30.2 / 25.4 / 11 | 29.6 / 25.0 / 10 | 32.1 / 29.6 / 24 | 27.0 / 20.4 / 0 |
| `downtown_runForBus_01` a0 #185 | 28.7 / 16.1 / 8 | 9.1 / 7.1 / 17 | 9.2 / 8.4 / 2 | 8.1 / 8.6 / 23 | 8.1 / 4.4 / 0 |

- **#514.** G1 corrects most of the conflict: its side is nearly exact (1°), and its reach (0.73) is closer to GT than H0's (0.97). It is still 6.6 cm from C111.
- **#775.** G1 gets the side right (3°), but it under-reaches (0.72 vs GT 0.97) and breaks a frame B1 had solved: elbow 4.5 → 17.1 cm.
- **#325 and #446** are E1 direction failures that no reach/side owner can fix, as Worklog 77 found.

## Clean validation (selection split; separate)

| arm | elbow (cm) | wrist (cm) | reach MAE | side error |
|---|---:|---:|---:|---:|
| B1 | 6.51 | 9.96 | 0.044 | 21.6° |
| G0 | 6.81 | 10.07 | — | — |
| G1 | 5.97 | 9.89 | 0.041 | 16.7° |

The 3,056 validation rows chose both the E1 and the owner epochs, so they are selection-contaminated. G1 − B1 elbow is −0.54 cm there and G1 − G0 is −0.84 cm. The test split does not confirm either result, which is consistent with the overfitting visible in G1's training gap.

## Real-scene Blender review exports

- **Renders:** Worklog 76/77 rig, camera and object-rotation contract, Blender 5.1.2. Blender FK end vs export: 9.5e-7 cm.
- **Drawn arms:**
  - B0 blue, B1 magenta;
  - G0 orange, G1 cyan;
  - GT reference green, E_GT yellow.
- **Captions:** exact IDs, GT/H0/G0/G1 reach, E1/GT f, and the elbow and wrist errors of B0/B1/G0/G1/C111, with side error. Strips also show the next pair's upper-arm residual.

| export (`eval/review/`) | row(s) |
|---|---|
| `w76_owner_downtown_bar_00_514` (also the near-folded conflict pick), `_775`, `w76_owner_downtown_sitOnStairs_00_325`, `w76_owner_flat_packBags_00_446` | owner frames |
| `largest_g1_elbow_improvement` / `largest_g1_elbow_regression` | `downtown_walking_00` a0 #580 / `downtown_cafe_00` a1 #668 |
| `largest_g1_temporal_improvement_strip` / `largest_g1_temporal_regression_strip` | `downtown_walking_00` a0 #365… / `flat_packBags_00` a0 #1106… |
| `g1_stable_sequence_downtown_bar_00_strip` / `g1_unstable_sequence_outdoors_fencing_01_strip` | G1 − B1 sequence residual −3.33° / +4.12° per pair |

- **Server copy:** `eval/review/`.
- **QE images-only copy:** `~/animcv-output/78_learned_arm_geometry_review/` (50 PNGs).
- **Renders illustrate only.** The #514 render looks like a clear win, but the aggregate test numbers above do not support one.

## Architecture classification — CASE C

**Predeclared criteria** (`eval/report.json → decision_inputs`):

| criterion | G1 | G0 |
|---|---|---|
| static elbow vs B1 (mean < 0, sequence-cluster interval excludes 0, ≥ 2/3 sequences) | ✘ (+0.21) | ✘ (+0.39) |
| temporal vs B1 (upper-arm residual and proximal discontinuity both lower, intervals exclude 0) | ✘ | ✔ |
| wrist ≤ +0.5 cm and forearm ≤ +1.0° vs B1 | ✔ | ✔ |
| near-folded elbow below B1 | ✔ | ✔ |
| coverage and rig invariants | ✔ | ✔ |
| RGB contributes (G1 − G0 static elbow or temporal residual < 0 with interval excluding 0) | ✘: static −0.18 not significant; temporal +1.15 (worse) | — |

**Mechanical case: C.** G1 meets neither the static nor the temporal requirement.

**Not CASE B.** B requires G0 to improve B1 materially on both static and temporal. G0 improves only temporally while the elbow regresses, which is a CASE D-type static/temporal trade-off for the 2D-only owner.

**Answer to the completion question: no.** Under this fixed single-frame representation and training contract, a reach/bend-side owner learned from runtime RGB and 2D joints does not replace H0's geometry with a more accurate and stable connected arm.

**Principal failure — reach:**
- the learned reach is no more accurate than H0's (MAE 0.05);
- inserted alone it degrades the elbow in every regime except near-folded;
- G1's reach is as jumpy as H0's.

**Partial success — bend side:**
- side error falls by 4–5°, an elbow gain with H0 reach, and a near-folded elbow repair driven by the side;
- the remaining side error (18–19°) is still large against the oracle.

**Temporal consistency:**
- the 2D-only owner is steadier than H0;
- adding Qwen features makes the owner's per-frame outputs jumpier and overfit, so RGB content as consumed here does not help.

Worklog 77's oracle diagnosis stands. No smoothing, threshold, gate, H0 fallback, selector or tuning was added, and nothing was integrated. Any next step needs a new bounded decision.

## Focused tests

- **`tests/test_framepose_arm_geometry_owner.py` (8 tests):**
  - Worklog 61/62 target definitions over straight/bent/folded arms, positive/negative depth and rotated coordinates (rotation equivariance);
  - invalid GT joints, zero-length segments and degenerate detector planes are masked;
  - a perfect reach/side prediction reproduces the Worklog 77 C111 arm (≤ 1e-9);
  - an unprojected side gives the same arm, a parallel side is unavailable, a tiny reach is clamped by the solver;
  - IK/FK and bone-length invariants;
  - G0 zeroes only the visual tensor;
  - identical G0/G1 initialization and parameter count;
  - the loss ignores side-invalid rows;
  - learned-arm assembly and side error;
  - training and evaluation static contracts: E1 hash and reproduction before training, frozen E1, no test in training, gate order, Worklog 76 thresholds, decisions declared before `main`, no production imports.
- **Server run (`animcv-framepose:cuda118`):** with the Worklog 61/62/73/74/75/76/77 focused tests, **66 passed**.
- **Full regression:** not run, because production behavior is unchanged.

## Reproduction

    D=/output/framepose_external_visual_supervision
    docker run --rm --gpus all -v $HOME/AnimCV:/workspace:ro -v $HOME/animcv-output:/output \
      -e PYTHONPATH=/workspace/src:/workspace/scripts -w /workspace animcv-framepose:cuda118 \
      scripts/train_framepose_arm_geometry_owner.py --bank /output/framepose/bank_3dpw_paired_v2.json \
      --cache-dir /output/framepose_qwen3vl_visual_cache --external-train $D/bedlam_train_final \
      --external-train-cache $D/qwen_train --external-val $D/bedlam_validation_final \
      --external-val-cache $D/qwen_validation --w73-dir $D/transfer --out-dir /output/framepose_arm_geometry_owner
    docker run ... -v $HOME/animcv-data/datasets/3dpw:/data/3dpw_raw:ro animcv-framepose:cuda118 \
      scripts/run_framepose_arm_geometry_owner_eval.py --rig examples/BaseRig.fbx \
      --rest examples/e2e_demo/baserig_blender_rest.json --calibration examples/e2e_demo/framepose_fk_calibration.json \
      --chains examples/e2e_demo/framepose_ik_chains.json --bank /output/framepose/bank_3dpw_paired_v2.json \
      --h0-dir /output/framepose/context_h0_v2 --threedpw-raw /data/3dpw_raw --w73-dir $D/transfer \
      --w76-dir /output/framepose_visual_full_arm_ik --w77-dir /output/framepose_visual_arm_ownership \
      --owner-dir /output/framepose_arm_geometry_owner --out-dir /output/framepose_arm_geometry_owner/eval
    scripts/export_framepose_arm_geometry_owner_cases.py ... --eval-dir /output/framepose_arm_geometry_owner/eval \
      --out /output/framepose_arm_geometry_owner/eval/review_cases.json
    Blender -b -P scripts/review_framepose_visual_arm_ownership_blender.py -- --rig examples/BaseRig.fbx \
      --rest examples/e2e_demo/baserig_blender_rest.json --cases review_cases.json --out-dir review --caption-python python3
