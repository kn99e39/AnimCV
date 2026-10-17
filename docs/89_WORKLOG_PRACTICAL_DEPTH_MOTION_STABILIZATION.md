# Worklog 89 — Practical RGB Depth-Motion Stabilization

Date: 2026-10-17
Branch: `arch/single_frame_first`
Starting HEAD: `54fa981` (Worklog 88). Local, origin and LabServer63 were in sync and the worktree was clean.
Type: one bounded, optional postprocessing candidate plus a real-scene evaluation.
- No model was trained.
- No checkpoint, production FramePose, IK or Blender code changed.
- No data was downloaded.

## 0. Answer

**Engineering CASE B: useful but limited.** The candidate is an offline Kalman/RTS smoother of each segment's depth tilt
(image-plane heading untouched). It runs on the frozen G1/J1X outputs.

**What it does well**

On 3DPW TEST, each frame-to-frame step is 1/6 s. Never tuned on this set:

- **Depth-motion residual drops to G0's level while keeping G1's static accuracy.** Left arm 10.85 → 7.78 %L (G0 7.72),
  right arm 10.89 → 7.65 (G0 7.83), right leg 8.02 → 6.55 (G0 6.62).
- **Fewer false motions on near-still limbs.** The false-motion rate on low-GT-motion pairs falls 0.53 → 0.32 (left arm)
  and 0.51 → 0.32 (right arm).
- **Frame-to-frame jerk falls to GT level** (left 16.5 → 8.3; GT 7.8).
- **Static per-frame accuracy improves on every chain** (distal −0.3 to −0.6 %L).
- **No lag** (best shift 0 rows).
- **Genuine motion is mostly kept.** On high-motion pairs the residual improves (−3.2 / −2.9 / −1.6 %L) and sign agreement
  rises (0.71 → 0.78).

**Why it is B, not A**

- **Slope retention misses the threshold.** It keeps only 67–71% of G1's high-motion slope gain over G0 on TEST (and
  73% for the right arm on VALIDATION 0.1 s), below the predeclared 80%. Real large motions are attenuated by about 13%
  in slope.
- **Too many severe regressions on VALIDATION.** 1.7–1.9% of arm frames get a > 10 %L static regression there, above the
  predeclared 1%.
- **Slow drift survives on still limbs.** The clips show that the slow (~1 s) false depth drift of G1 on still limbs is
  *not* removed, only its frame-to-frame spikes. Low-motion false motion remains about 3× G0's on TEST.

The candidate is kept as an experimental, switchable, offline component. It is not promoted.

## 1. Historical baselines and parity

- **Inputs:** frozen Worklog 87 predictions, the only model outputs used.
  - `predictions_3dpw.npz` `cb9dc51e…` and `predictions_3dhp.npz` `356f8cee…` are hash-pinned.
  - The checkpoint identities behind them were verified in Worklogs 87/88 (J1X `82fd6bcb…`; W86 G0/G1).
- **Worklog 88 depth-motion residuals reproduced exactly** (diff 0.0):

  | population | left arm | right arm | right leg |
  |---|---|---|---|
  | TEST | 7.72 / 10.85 | 7.83 / 10.89 | 6.62 / 8.02 |
  | VALIDATION 0.1 s | 7.10 / 9.59 | 6.91 / 9.60 | 5.13 / 5.95 |
  | VALIDATION 0.2 s | 11.64 / 12.16 | 10.81 / 11.59 | 7.36 / 7.37 |

  Each cell is G0 / G1.
- **Disabled stabilizer = exact identity** (checked on every population; STOP otherwise).
- **Historical artifacts untouched.** All outputs are new, under `~/animcv-output/framepose_depth_stabilization/`.

## 2. Existing temporal components (audit)

- **C1** (Worklog 48/49, `src/framepose/context_refiner.py:306` `interpolate_gated_h0`)
  - What it is: gated two-sided linear interpolation of individual full-body H0 joints in absolute XYZ, on about 3.6% of
    joint rows, gated by stable 2D and a high H0 interpolation residual.
  - Contract: sparse **outlier repair**. It has no chain contract, used future rows (±2), improved MPJPE and worsened
    PA-MPJPE.
  - It is **not comparable** to dense stabilization of connected-limb camera depth, so it was not rerun and its numbers are
    not reused.
- **C2/C3:** learned residual refiners on the same gate. Both negative (−0.75 and −0.23 mm on gate-on rows).
- **Production FramePose:** has no smoother. `src/framepose/evaluate.py:9` states that temporal smoothness is
  deliberately absent; IK is per-frame.
- **Older MotionGraph path** (`src/retarget/temporal_filter.py:10`, `src/pose/root_motion.py:197`): only median filters of
  landmarks and root yaw. These are a different contract.

**Task separation.** This batch addresses *stabilizing valid but inconsistent RGB depth estimates*. It does not address:

- missing-joint repair
- interpolation of unavailable observations
- detector jitter in 2D
- reconstruction of rapid motion

## 3. Chosen mechanism

`src/framepose/depth_motion_stabilizer.py` (new, experimental, imported by no production module).

**What changes.** For each chain:

- The unit chain (P=0, M=ρu, D=M+(1−ρ)v) is decomposed.
- Only the **depth tilts** θ_u = asin(u_y) and θ_v = asin(v_y) are changed.
- They are smoothed per track with a constant-velocity Kalman model in physical time: white-noise acceleration q
  (rad²/s³) and white measurement noise σ (rad).
- In offline mode, a Rauch–Tung–Striebel backward pass follows.

**What is kept.** The raw image-plane heading (u_x, u_z)/|·| of each segment and the raw ρ. The result is re-assembled as
d' = (cos θ'·h_x, sin θ', cos θ'·h_z).

**Why this should work, and its limits**

- Worklog 88 put the excess in the depth components of both segment directions (u_y, v_y), with ρ negligible. The excess
  is high-frequency within-model visual response.
- What the detector observes (the 2D heading) is not touched, and connectivity, segment lengths and P are preserved by
  construction.
- Under a CV prior, real limb motion is temporally correlated and the false motion is largely frame-independent.
- Expected limitation: correlated (slow) false drift cannot be separated from real slow motion by any smoother.

**Runtime inputs:**

- frame timestamps (s)
- the track key (sequence/actor)
- the predicted M and D
- detector chain-joint validity

There is no GT, no motion class and no sequence identity.

**Causality**

- `offline` (the candidate): uses the whole track, with zero phase lag. It is for offline animation generation, not a
  causal solution.
- `causal`: forward filter only. Reported for the latency trade-off; not part of the CASE rule.

**Missing data and resets**

- Invalid observations get no measurement update; the state is propagated.
- A new track key or a gap > 1.0 s restarts the state.
- Degenerate image-plane headings keep the raw direction.
- Tilts are clipped at ±(90° − 1e-6 rad), so the heading is never lost. See the defect note below.

**Parameters**

- σ and q were calibrated **once per chain** on 3DPW VALIDATION (0.1 s rows), with a fixed moment estimator declared in
  code before any stabilized output existed. The calibration code is offline-only and uses GT.
- **σ** = 1.4826·MAD(Δ²(θ̂ − θ))/√6. A white error contributes 6σ² to the second-difference variance; the MAD keeps
  detector failures from inflating the noise.
- **q** = 1.5·mean((Δ²θ_GT)²)/dt³. White-noise acceleration gives (2/3)·q·dt³; the mean of squares keeps the energy of
  real motion.
- Both choices err towards less smoothing.

| chain | σ (rad) | q (rad²/s³) |
|---|---|---|
| left arm | 0.0803 | 21.19 |
| right arm | 0.0744 | 16.81 |
| right leg | 0.0507 | 19.29 |

- Fixed: reset gap 1.0 s; initial velocity spread π rad/s.
- No sweep. `config.json` is hash-frozen, and the replay refuses a changed config.

**Why G1's TRAIN split was not used for calibration:** G1 was trained on 3DPW TRAIN, so its train residuals would
understate the noise.

**Caveat:** VALIDATION results are therefore in-sample for two scalars per chain. TEST and 3DHP are held out.

**Anticipated failure cases** (declared beforehand):

- attenuation of fast, short depth excursions
- inability to remove slow correlated drift
- start-of-track transients
- near-±90° tilts

**Rejection condition** (declared): validation slope retention < 0.5 on both arms, which would mean systematic loss of
genuine motion.

**Implementation defect fixed and rerun (same candidate).** The first validation run stopped at the geometry check.
Tilts clipped to exactly ±90° lost the image-plane heading. The clip was changed to ±(90° − 1e-6 rad) and the run
repeated. The deterministic calibration regenerated a byte-identical config; the runner enforces this.

## 4. Changed files

| file | role |
|---|---|
| `src/framepose/depth_motion_stabilizer.py` | the candidate (switchable: `cfg=None` → identity) and the offline calibration estimator |
| `scripts/run_depth_motion_stabilization.py` | predeclared `DECISION`; validation phase (calibrate, freeze, evaluate) and replay phase (frozen) |
| `scripts/review_depth_motion_stabilization.py` | fixed-rule A/B video clips |
| `tests/test_framepose_depth_motion_stabilizer.py` | 11 focused tests |

Nothing existing was modified.

## 5. Synthetic contracts

All pass locally and on LabServer63; together with the Worklog 88 suite, 18 tests.

- Disabled stabilizer gives an exact identity.
- Connected geometry: unit u and v, unchanged ρ and segment lengths, unchanged image-plane heading, P=0, including near-±90°
  segments.
- Input-order independence and per-track state isolation.
- Gap reset; invalid observations unused.
- Determinism.
- Causal mode has no future dependency; offline mode does, as declared.
- A constant-velocity ramp is preserved, with no lag or attenuation, at 0.1 s, 0.4 s and irregular spacing. Only a
  ≤ 0.15° track-start transient remains, from the velocity prior.
- Physical parameters smooth less at coarser sampling.
- The moment calibration recovers the generating σ and q of a continuous-time process.
- The runtime interface has no GT argument and no production coupling.

## 6. Results

A = raw G1/J1X, B = G0/G0X, C = stabilized G1 (offline). %L, distal joint, camera depth. Intervals are the Worklog 75/76
sequence-cluster 95% interval of C − A.

### 6.1 Low-GT-motion false motion (co-primary A)

Low = GT 3D distal motion below 4 %L. False motion = |dP| > 5 %L. Each cell is A / B / C.

| population | chain | false-motion rate | low-pair residual | low residual C − A | seqs improved |
|---|---|---|---|---|---|
| VAL 0.1 s | left arm | 0.379 / 0.096 / 0.132 | 5.73 / 2.36 / 2.49 | −3.25 [−4.13, −2.41] | 16/16 |
| VAL 0.1 s | right arm | 0.303 / 0.128 / 0.102 | 5.05 / 2.46 / 2.21 | −2.84 [−3.88, −2.11] | 16/16 |
| VAL 0.1 s | right leg | 0.183 / 0.095 / 0.046 | 3.07 / 2.17 / 1.65 | −1.42 [−1.67, −1.22] | 16/16 |
| VAL 0.2 s | left arm | 0.351 / 0.066 / 0.204 | 5.37 / 2.10 / 3.47 | −1.91 [−2.42, −1.42] | 16/16 |
| VAL 0.2 s | right arm | 0.237 / 0.098 / 0.136 | 4.32 / 2.50 / 2.88 | −1.44 [−1.77, −1.15] | 15/16 |
| **TEST** | left arm | 0.525 / 0.116 / 0.320 | 7.50 / 2.72 / 4.52 | −2.98 [−3.23, −2.75] | 37/37 |
| **TEST** | right arm | 0.513 / 0.144 / 0.321 | 7.46 / 2.86 / 4.49 | −2.96 [−3.38, −2.55] | 37/37 |
| **TEST** | right leg | 0.298 / 0.114 / 0.191 | 4.33 / 2.43 / 3.19 | −1.14 [−1.29, −0.98] | 37/37 |

**Visible oscillation** (TEST, consecutive pairs):

| chain | mean \|Δ dP\| A → C (GT, G0) | sign-alternating ≥ 5 %L steps on still limbs, A → C (G0) |
|---|---|---|
| left arm | 16.5 → 8.3 (GT 7.8, G0 7.3) | 0.22 → 0.04 (0.02) |
| right arm | 16.6 → 8.0 (GT 7.2) | 0.20 → 0.04 |

**Low-pair tails** (TEST): left arm p99 32.8 → 20.9.

On VALIDATION 0.1 s the candidate reaches G0's still-limb behaviour. On TEST it removes about 40% of the false motion
and all of the zig-zag, but the slow drift stays (§8).

### 6.2 Genuine high-motion preservation (co-primary B)

High = GT 3D distal motion ≥ 15 %L.

- **Slope** = OLS slope of dP on dG. It is unbiased by frame-independent noise in dP, so a drop in slope is real
  attenuation.
- **Retention** = (slope_C − slope_B)/(slope_A − slope_B).

| population | chain | slope A / B / C | retention | amplitude ratio A / B / C | signed error along GT, A → C | high residual C − A | sign agreement A → C | false hold A → C |
|---|---|---|---|---|---|---|---|---|
| VAL 0.1 s | left arm | 0.588 / 0.216 / 0.552 | 0.90 | 1.04 / 0.58 / 0.69 | −5.9 → −6.4 | −5.10 [−6.21, −4.01] | 0.72 → 0.82 | 0.10 → 0.13 |
| VAL 0.1 s | right arm | 0.561 / 0.261 / 0.479 | **0.73** | 1.11 / 0.67 / 0.65 | −5.6 → −6.7 | −5.73 | 0.72 → 0.82 | 0.09 → 0.12 |
| VAL 0.1 s | right leg | 0.635 / 0.369 / 0.558 | 0.71 | 0.97 / 0.80 / 0.67 | −4.6 → −5.5 | −3.29 | 0.75 → 0.84 | 0.13 → 0.14 |
| VAL 0.2 s | left / right arm | — | 0.90 / 0.85 | — | — | −3.08 / −3.41 | — | — |
| **TEST** | left arm | 0.506 / 0.282 / 0.439 | **0.70** | 0.89 / 0.51 / 0.65 | −7.9 → −9.0 | −3.17 [−3.60, −2.78] | 0.71 → 0.78 | 0.09 → 0.12 |
| **TEST** | right arm | 0.503 / 0.291 / 0.434 | **0.67** | 0.89 / 0.58 / 0.63 | −7.6 → −8.9 | −2.86 [−3.59, −2.26] | 0.72 → 0.78 | 0.09 → 0.13 |
| **TEST** | right leg | 0.465 / 0.310 / 0.419 | **0.71** | 0.76 / 0.53 / 0.60 | −9.0 → −9.7 | −1.55 [−1.92, −1.24] | 0.75 → 0.79 | 0.10 → 0.11 |

**How to read this**

- The high-motion **error** improves because the noise is removed.
- The high-motion **response** loses about 13% of slope, roughly 30% of G1's advantage over G0, and about 1 %L more
  under-response per pair.
- G0's under-response remains far worse. The candidate stays much closer to G1 than to G0, but misses the predeclared 80%
  retention.

**Observability counterexample (§5).** Pairs with large GT depth motion (|dG| ≥ 15) but detector 2D displacement below
the median (TEST, n ≈ 210 per arm). The candidate does not use 2D motion, and keeps about 95% of G1's slope:

| chain | slope A → C (G0) |
|---|---|
| left arm | 0.398 → 0.381 (0.227) |
| right arm | 0.400 → 0.382 (0.170) |
| right leg | 0.330 → 0.315 (0.114) |

### 6.3 Static 3D accuracy

Per frame, GT-valid rows. Each cell is A / B / C.

| population | chain | distal error | distal camera depth | distal p95 | middle | u / v angle (°) | bend (°) | reach |
|---|---|---|---|---|---|---|---|---|
| VAL | left arm | 16.49 / 20.66 / 15.65 | 13.65 / 17.84 / 12.69 | 43.3 / 51.3 / 40.3 | 9.37 / 12.68 / 8.90 | 10.6 / 14.0 → 10.1 / 13.4 | 9.4 → 9.0 | 3.56 → 3.43 |
| VAL | right arm | 17.62 / 21.24 / 16.81 | 14.52 / 18.24 / 13.61 | 43.0 / 53.8 / 39.8 | 9.75 / 11.98 / 9.32 | 11.3 / 14.4 → 10.7 / 13.8 | 11.1 → 10.7 | 4.11 → 3.97 |
| VAL | right leg | 10.99 / 14.05 / 10.55 | 9.02 / 11.60 / 8.57 | 28.8 / 36.0 / 27.0 | — | — | 7.2 → 6.8 | — |
| TEST | left arm | 20.21 / 22.23 / 19.63 | 16.83 / 19.46 / 16.25 | 51.0 / 55.6 / 49.0 | 12.71 / 13.44 / 12.40 | 14.8 / 15.5 → 14.4 / 15.1 | 13.9 → 13.7 | 5.22 → 5.12 |
| TEST | right arm | 19.48 / 21.06 / 18.94 | 16.20 / 18.49 / 15.69 | 47.8 / 52.6 / 46.1 | 10.67 / 11.80 / 10.36 | — | 12.6 → 12.3 | 5.10 → 4.98 |
| TEST | right leg | 13.92 / 14.02 / 13.62 | 11.73 / 11.37 / 11.48 | 33.6 / 33.6 / 33.0 | 8.44 / 9.36 / 8.27 | — | 9.2 → 9.0 | 2.57 → 2.49 |

**Distal C − A**

- TEST: −0.59 [−0.66, −0.51], −0.54, −0.30.
- VAL: −0.83, −0.81, −0.44.

The RGB depth advantage over G0 is kept fully (retention 1.18–1.25).

**Severe static regressions** (C distal error > A + 10 %L):

| population | rate | vs 1% criterion |
|---|---|---|
| VAL arms | 1.73% / 1.91% | **fails** |
| VAL leg | 0.09% | passes |
| TEST | 0.25% / 0.54% / 0.08% | passes |
| 3DHP | 1.2–1.6% | above |

These frames are mostly peaks of fast motion that the smoother rounds off (§8).

### 6.4 Overall temporal residual and tails (TEST; A / B / C)

| chain | residual | p95 | p99 |
|---|---|---|---|
| left arm | 10.85 / 7.72 / 7.78 | 30.4 / 24.4 / 22.5 | 46.3 / 38.0 / 36.9 |
| right arm | 10.89 / 7.83 / 7.65 | 31.2 / 24.6 / 21.8 | 45.2 / 41.4 / 34.6 |
| right leg | 8.02 / 6.62 / 6.55 | 25.0 / 22.4 / 20.4 | 41.8 / 38.1 / 33.7 |

C's p95 and p99 are the lowest of the three models. The middle-joint residual moves the same way (6.25 → 4.44, left arm).

### 6.5 3DHP, kept separate

**0.4 s pairs** (TS1–TS4 rows are themselves 0.4 s apart):

- The physical-time model predicts large uncertainty over 0.4 s, so the candidate is close to pass-through, by design.
- Residual C − A: left −0.91, right −1.28, leg −0.31.
- False-motion reduction 12% / −2% / 5%: no suppression.
- Slope retention 0.95 / 0.97 / 0.89. Static change −0.28 / −0.47 / −0.12.
- Right-arm p95 static +1.0 %L, which marks its static criterion as failed. Severe 1.2–1.6%.
- Not harmful, but not useful at this cadence.

**TS5–TS6 outdoor** (rows at about 50 fps, pairs 0.16 s, 2 sequences):

- Residual 20.2 → 15.5 (left arm) and 20.8 → 15.1 (right arm).
- High slope kept (retention 1.05 / 1.12).
- Only 5–43 low-motion pairs, so descriptive only.

### 6.6 Time-step consistency

The parameters are in seconds and radians.

**Effect shrinks with interval.** Fed every second VALIDATION row (0.2 s input cadence), the same frozen candidate gives
(on the same 0.2 s pairs) about 80% of the full-rate improvement:

| chain | residual C − A, 0.2 s input | full-rate input |
|---|---|---|
| left arm | −2.35 | −2.75 |
| right arm | −2.46 | −3.22 |
| right leg | −0.96 | −1.27 |

with the same retention pattern. At 0.4 s it is nearly pass-through, and at 50 fps (TS5–TS6) it smooths strongly.

**Frame-rate independent:** a parameter's physical meaning does not change with fps (tested).

**Sampling-dependent by design:** coarser sampling gives less smoothing.

### 6.7 Causal mode (same parameters, forward filter only; not part of the CASE rule)

| | TEST residual (left / right / leg) | low false-motion rate (left arm) | slope retention | static |
|---|---|---|---|---|
| A (raw) | 10.85 / 10.89 / 8.02 | 0.525 | — | — |
| causal | 10.11 / 10.08 / 7.74 | 0.493 | ≈ 1.0 | −0.12 |

A causal filter with these noise levels has little to work with: it cannot see the next frame that reveals a spike as
noise. **The useful effect requires lookahead.** A fixed-lag version (a few frames) was not evaluated, since that would be
a second candidate.

## 7. Runtime cost

| item | value |
|---|---|
| Overhead (Python/NumPy, per chain) | offline ≈ 70 µs/frame; causal ≈ 41 µs/frame. TEST, three chains: about 0.2 ms per frame in total. Negligible next to Qwen/G1 inference |
| State per segment, causal | a 2-vector and a 2×2 covariance |
| State, offline | stores the forward pass of the whole track: about 14 floats per frame per segment |
| Lookahead | offline: the whole track (or until a > 1 s gap); causal: none |
| Output latency | offline: end of track; causal: 0 |
| Irregular spacing | supported (dt per step; ramp test with random 0.02–0.4 s spacing) |
| Missing observations | measurement skipped; > 1 s gap resets |
| Track changes | state reset per track key; isolation tested |
| Deterministic replay | yes (tested) |
| Batch video processing | yes (per-track, independent) |

## 8. Real-video A/B review

13 clips (MP4, 6 s each, real time, 6 fps rows) plus a summary PNG each. They are at
`~/animcv-output/89_framepose_depth_stabilization/` (server: `framepose_depth_stabilization/review/`).

**Panels**

- RGB with the detector chain
- side view of GT / G0 / raw G1 / stabilized G1
- distal-depth trajectories with a cursor
- per-pair dY and |residual| with FALSE-MOTION / STALE flags

Selection rules are fixed in the script docstring.

| clip | what it shows |
|---|---|
| stationary_limb (bus_00 actor1) | GT is flat. Raw G1 zig-zags ±10 %L **and** drifts by about 30 %L over ~2 s (C: 18 → −5 → 23 %L). C removes the zig-zag; **the drift remains** |
| historical_false_motion (walking_00 540) | Raw spike of +77 %L removed. Slow ±30 %L excursions remain in C |
| walking_arm_swing (downstairs_00) | Swing tracked. C follows G1's shape without its spikes |
| toward_away_camera (rampAndStairs_00 139) | Periodic ±30 %L depth swing with small 2D motion. C keeps the phase; peaks are about 10–20% lower |
| fast_arm_movement (fencing_01 582) | Large lunges. C ≈ G1 (both miss part of the GT amplitude) |
| depth_sign_change, strong_out_of_plane, occlusion, confidence_degradation | C ≈ G1 shape with fewer spikes. No new failure mode visible |
| historical_correct_g1 / worst_static_regression (fencing_01 90) | Where G1 tracked a fast excursion, C rounds the peak: the source of the severe static regressions |
| worst_regression_left_arm (fencing_01 732) | Both miss a −70 %L GT excursion. C is marginally smoother than G1 |
| worst_regression_right_leg (runForBus_00 120) | Gait peaks slightly lower in C |

**Owner-level judgement from the clips**

- The animation becomes visibly calmer: no single-frame depth pops.
- It does not fix the slow false depth wandering of still arms. That drift is what remains of G1's low-motion problem
  after stabilization.
- Fast real excursions are slightly flattened.

## 9. Engineering CASE

Predeclared rule (`DECISION`, `classify()`): **B**.

**On VALIDATION 0.1 s**

| chain | suppression | retention | static | lag | tails |
|---|---|---|---|---|---|
| left arm | ✓ | ✓ (0.90) | ✓ | ✓ | ✗ (severe 1.7%) |
| right arm | ✓ | ✗ (0.73) | ✓ | ✓ | ✗ (1.9%) |

**On TEST, both arms:** suppression ✓ (false motion −39% / −37%, CI below 0, 37/37 sequences); retention ✗
(0.70 / 0.67); static ✓; lag ✓; tails ✓.

**Right leg:** not regressing. **3DHP 0.4 s:** arm residual not worse.

**Supported deployment:**

- offline animation generation from 6–10 Hz (or denser) frame sampling
- scenes dominated by slow or moderate motion, standing and walking, where single-frame depth pops are the main visual
  defect

**Limitations:**

- about 13% slope attenuation of real fast depth motion, with peak rounding
- slow false drift not removed
- no benefit at 0.4 s sampling
- a causal variant gives little

## 10. Production-integration implications (not done)

- **Where it would go:** after the FramePose limb predictions and before IK, as an opt-in, per-actor flag.
- **What it would need:**
  - a stable actor track key
  - real video timestamps
  - the per-chain config shipped with the model version (σ and q are tied to the G1/J1X checkpoints; recalibrate if a
    checkpoint changes)
- **Mode:** offline only, unless a fixed-lag variant is separately validated.
- **Not verified:** the effect after IK and retargeting (bone roll and joint rotations) and in Blender. Worklog 88/89
  metrics are limb-position metrics.

## 11. Recommended next bounded task

An **opt-in end-to-end replay**:

- run the frozen stabilizer, unchanged and defaults off, through FramePose → two-bone IK → Blender export on 2–3 3DPW
  TEST/VALIDATION clips;
- carry out an owner A/B review of the rendered rig animation, plus rotation-level jitter metrics (elbow and knee bend
  and pole angle per frame).

This decides whether the measured calmer limbs translate into visibly better animation before any integration decision.

The remaining slow-drift problem should be investigated separately. It is a model-level, not a filtering, issue, and
should not be stacked onto this candidate.
