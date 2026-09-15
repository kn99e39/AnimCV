# Worklog — Historical Recovery and Real Pose Reconciliation Replay (2026-09-15)

Branch: `arch/single_frame_first`

Starting HEAD: `42337d8364ad61c994e2b694f0bd7969672be4f7`

## 1. Recovery gate: PASS

The documented historical store was inspected read-only at
`LabServer63:/home/nd/animcv-output/framepose`, including the corrected
`sign_attr_v2/O_BILATERAL` lineage, `hinge_ownership_v3`, the docs/39-era
camera report, and the invalid v1 attribution directory. Original paths, sizes,
mtimes, and SHA-256 values were inventoried before copying. No historical file
was moved or changed. The exact recovery table is
[`44_HISTORICAL_ARTIFACT_RECOVERY.json`](44_HISTORICAL_ARTIFACT_RECOVERY.json).

Recovered and cross-bound:

- FrameBank content digest
  `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536`,
  21,817 samples, split 11,334 / 3,407 / 7,076, benchmark-detector regime.
- Corrected `sign_attr_v2/O_BILATERAL` prediction SHA-256
  `6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5` and
  evaluation SHA-256
  `a5881850b3769ca2329271d327ee895dc646f236451f05c3ba2e0ff917f580a3`.
  The source verifier passed for the expected candidate and 7,076 frames.
- `hinge_ownership_v3/hinge_evidence_ownership.json` matched the prediction,
  evaluation, and bank identities; its historical `minimum_norm` policy,
  residual total 279, and depth-correct count 83 all verified.
- All 24 local test camera pickles matched the recovered docs/39 manifest by
  byte size and SHA-256: no missing files, extras, or mismatches. Camera
  geometry covered all 7,076 test frames. Current reconstruction preflight
  max error was `4.12e-5 mm`; docs/39 recorded `2.98e-5 mm`; both are far below
  its fixed 1.0 mm refusal boundary. All 26,818 eligible target middle-joint
  projections succeeded.

One recorded-version discrepancy is explicit: the recovered original bank
serializes as `animcv_frame_pose_bank_v2`, whereas the direction's identity
contract names v1. Its index and NPZ hashes match the contemporaneous
experiment matrix, and its path-independent content digest, sample/split
counts, regime, and provenance fingerprint match. The v2 provenance addition
does not change the v1 content-digest domain. This exact historical bank pair
was used; no reconstruction or substitute baseline was made. The v1
`sign_attr_v1_INVALID_mask_discarded` prediction/evaluation were separately
hashed, rejected, and not used.

## 2. Replay and verification

The fixed command used the recovered bank, prediction, evaluation, and exact
local camera pickles. It ran the existing replay once, with no training,
inference regeneration, tuning, or solver change. The only source edit in this
batch is non-behavioral reporting: the wrong-sign whole-pose evaluation is
explicitly labeled and restricted to the union of opposite-request C frames.
The solver, cohort definitions, cohort semantics, historical policies, and
SignState are unchanged.

```bash
PYTHONPATH=.:src .venv/bin/python scripts/replay_pose_reconciliation.py \\
  --bank /private/tmp/animcv-pose-recovery.FyI66d/bank_3dpw_paired_v2.json \\
  --source O_BILATERAL=O_BILATERAL_oracle_forward_depth_only:/private/tmp/animcv-pose-recovery.FyI66d/prediction_test.npy:/private/tmp/animcv-pose-recovery.FyI66d/evaluation_test.json \\
  --raw-root /Users/nadan/Projects/Lab/DATASET_Motion/sequenceFiles \\
  --out /private/tmp/animcv-pose-replay.lKDzaA
```

Focused pre-run tests: `14 passed`. Replay report:

```text
/private/tmp/animcv-pose-replay.lKDzaA/pose_reconciliation_replay.json
SHA-256 61fde30e7e6d8d9769a0273450819478fc6c457af3841eae3d0333c1f8e16e8c
```

Post-run checks passed for source/bank identity, test shape/count, 24-file
camera provenance and full camera coverage, all requested-sign coverage
identities, C same-row OBS/oracle comparison, and wrong-sign C-union frame
count. Reconciled endpoints were exactly unchanged. Maximum field-level
adjacent-bone change for both swivel candidates was `2.23e-13 mm`, below the
existing solver contract's `16 * machine-epsilon` bound after unit conversion.

## 3. Cohort populations

Counts below are union frames; field counts are exact chain-frame populations.
B, C, and E happen to have identical rows in this recovered run. D remains the
secondary common-corrected cohort.

| Cohort | Union | Left elbow | Right elbow | Left knee | Right knee |
|---|---:|---:|---:|---:|---:|
| A — all test | 7,076 | 7,076 | 7,076 | 7,076 | 7,076 |
| B — requested conflict | 2,244 | 759 | 591 | 592 | 724 |
| C — observation eligible | 2,244 | 759 | 591 | 592 | 724 |
| D — all methods corrected | 1,102 | 335 | 244 | 228 | 370 |
| E — target-projection eligible | 2,244 | 759 | 591 | 592 | 724 |

## 4. Whole-pose matched results

All-test values use the canonical evaluator over all 7,076 frames. C is the
primary operational comparison; E is numerically identical here because its
field rows equal C. D is secondary and conditioned on common correction.

| Candidate | A MPJPE / PA (mm) | A flip / bend MAE | C MPJPE / PA (mm) | C flip / bend MAE |
|---|---:|---:|---:|---:|
| H0 | 79.229 / 56.425 | 1.892% / 24.099° | 77.497 / 55.026 | 3.996% / 29.605° |
| DEPTH_ONLY | 79.366 / 56.694 | 1.158% / 22.129° | 77.928 / 55.876 | 1.686% / 23.396° |
| MINIMUM_NORM | 79.282 / 56.543 | 1.158% / 22.129° | 77.663 / 55.398 | 1.686% / 23.396° |
| R_SWIVEL_OBS | 79.235 / 56.497 | 1.440% / 22.894° | 77.512 / 55.255 | 2.574% / 25.808° |
| R_SWIVEL_ORACLE_2D | 79.214 / 56.499 | 1.380% / 22.787° | 77.448 / 55.261 | 2.384% / 25.472° |

On D's 1,102 common-corrected union frames, MPJPE / PA-MPJPE / flip / bend
MAE were: DEPTH_ONLY `81.030 / 58.484 / 1.301% / 22.569°`, MINIMUM_NORM
`80.606 / 57.733 / 1.301% / 22.569°`, R_SWIVEL_OBS
`80.428 / 57.617 / 2.329% / 25.726°`, and R_SWIVEL_ORACLE_2D
`80.311 / 57.610 / 2.080% / 25.346°`. D is descriptive only; it does not
replace unresolved-inclusive C.

## 5. Exact C field-level hinge results

For each cell below, `corrected/unresolved` uses the same known-request
population `n`; the next table reports canonical evaluator hinge flip,
bend-direction MAE, and middle-joint 3D error on those exact rows.

| Field (n) | DEPTH_ONLY | MINIMUM_NORM | R_SWIVEL_OBS | R_SWIVEL_ORACLE_2D |
|---|---:|---:|---:|---:|
| Left elbow (759) | 406 / 353 | 406 / 353 | 335 / 424 | 342 / 417 |
| Right elbow (591) | 306 / 285 | 306 / 285 | 244 / 347 | 241 / 350 |
| Left knee (592) | 268 / 324 | 268 / 324 | 228 / 364 | 243 / 349 |
| Right knee (724) | 410 / 314 | 410 / 314 | 370 / 354 | 369 / 355 |

| Field | H0 flip / MAE / M3D | DEPTH_ONLY | MINIMUM_NORM | R_SWIVEL_OBS | R_SWIVEL_ORACLE_2D |
|---|---:|---:|---:|---:|---:|
| Left elbow | 12.52% / 50.36° / 109.69 mm | 3.43% / 32.58° / 124.87 mm | 3.43% / 32.58° / 119.18 mm | 6.32% / 37.59° / 116.47 mm | 5.93% / 37.25° / 116.82 mm |
| Right elbow | 7.61% / 39.48° / 91.48 mm | 3.05% / 26.05° / 116.66 mm | 3.05% / 26.05° / 107.77 mm | 4.91% / 31.54° / 102.49 mm | 5.25% / 31.31° / 101.23 mm |
| Left knee | 8.61% / 41.82° / 95.47 mm | 3.04% / 27.50° / 75.93 mm | 3.04% / 27.50° / 78.10 mm | 7.09% / 35.65° / 81.64 mm | 5.41% / 32.95° / 79.75 mm |
| Right knee | 6.35% / 38.25° / 69.60 mm | 0.97% / 22.10° / 70.39 mm | 0.97% / 22.10° / 68.60 mm | 2.35% / 28.72° / 66.09 mm | 2.07% / 28.23° / 64.94 mm |

Values in the second table are `flip rate / bend MAE / middle-joint 3D error`.
Sign requests were known throughout C, UNKNOWN was excluded, and for every
field/candidate `requested = already_satisfied + corrected + unresolved`.
R_SWIVEL_OBS retains every unresolved C row in these metrics.

## 6. C image-space and bone ownership

Image values are mean pixels, formatted `target-projection error /
observation-consistency error`; counts equal the field's C row count for every
entry. This includes the historical candidates as well as the OBS/oracle pair.

| Field | H0 | DEPTH_ONLY | MINIMUM_NORM | R_SWIVEL_OBS | R_SWIVEL_ORACLE_2D |
|---|---:|---:|---:|---:|---:|
| Left elbow | 26.10 / 25.19 | 25.45 / 25.05 | 28.47 / 27.72 | 24.75 / 23.37 | 24.13 / 24.25 |
| Right elbow | 26.56 / 26.93 | 28.51 / 29.09 | 29.51 / 30.36 | 26.76 / 26.72 | 25.72 / 27.67 |
| Left knee | 17.69 / 23.20 | 17.61 / 22.77 | 18.51 / 24.28 | 17.41 / 22.22 | 16.39 / 23.12 |
| Right knee | 22.60 / 24.84 | 21.45 / 23.69 | 21.02 / 22.93 | 19.15 / 20.02 | 18.23 / 20.91 |

Field-level absolute adjacent-bone length changes are in mm. Each vector is
`mean / p50 / p90 / p99 / max`, first for P–M, then M–D. These are the exact
field C rows; this preserves the required granularity rather than aggregating
untouched chains into the union.

| Candidate | Field | P–M change quantiles | M–D change quantiles |
|---|---|---:|---:|
| DEPTH_ONLY | Left elbow | 19.435 / 1.143 / 37.668 / 226.881 / 2952.192 | 14.196 / 0.992 / 31.489 / 109.755 / 2488.883 |
| DEPTH_ONLY | Right elbow | 20.347 / 0.845 / 24.485 / 327.360 / 1581.100 | 17.932 / 0.833 / 23.442 / 238.780 / 1775.046 |
| DEPTH_ONLY | Left knee | 6.258 / 0 / 21.125 / 59.571 / 131.910 | 8.114 / 0 / 29.325 / 86.637 / 186.296 |
| DEPTH_ONLY | Right knee | 6.655 / 1.390 / 19.719 / 44.103 / 117.080 | 6.625 / 1.395 / 19.734 / 43.538 / 117.634 |
| MINIMUM_NORM | Left elbow | 6.07e-15 / 0 / 2.78e-14 / 5.55e-14 / 1.11e-13 | 6.25e-15 / 0 / 2.78e-14 / 5.55e-14 / 2.78e-13 |
| MINIMUM_NORM | Right elbow | 5.49e-15 / 0 / 2.78e-14 / 5.55e-14 / 5.55e-14 | 4.74e-15 / 0 / 2.78e-14 / 5.55e-14 / 1.11e-13 |
| MINIMUM_NORM | Left knee | 5.49e-15 / 0 / 2.50e-14 / 5.55e-14 / 5.55e-14 | 5.44e-15 / 0 / 0 / 5.55e-14 / 1.11e-13 |
| MINIMUM_NORM | Right knee | 8.43e-15 / 0 / 5.55e-14 / 5.55e-14 / 1.11e-13 | 8.28e-15 / 0 / 5.55e-14 / 5.55e-14 / 1.11e-13 |
| R_SWIVEL_OBS | Left elbow | 6.29e-15 / 0 / 2.78e-14 / 5.55e-14 / 8.33e-14 | 1.28e-14 / 0 / 5.55e-14 / 1.11e-13 / 1.39e-13 |
| R_SWIVEL_OBS | Right elbow | 6.11e-15 / 0 / 2.78e-14 / 5.55e-14 / 5.55e-14 | 1.18e-14 / 0 / 5.55e-14 / 8.33e-14 / 1.11e-13 |
| R_SWIVEL_OBS | Left knee | 8.63e-15 / 0 / 5.55e-14 / 1.11e-13 / 1.11e-13 | 1.54e-14 / 0 / 5.55e-14 / 1.11e-13 / 2.22e-13 |
| R_SWIVEL_OBS | Right knee | 1.02e-14 / 0 / 5.55e-14 / 5.55e-14 / 1.11e-13 | 2.25e-14 / 0 / 5.55e-14 / 1.11e-13 / 2.22e-13 |
| R_SWIVEL_ORACLE_2D | Left elbow | 7.00e-15 / 0 / 2.78e-14 / 5.55e-14 / 8.33e-14 | 1.23e-14 / 0 / 2.78e-14 / 8.33e-14 / 1.39e-13 |
| R_SWIVEL_ORACLE_2D | Right elbow | 6.39e-15 / 0 / 2.78e-14 / 5.55e-14 / 5.55e-14 | 1.20e-14 / 0 / 5.55e-14 / 8.33e-14 / 1.11e-13 |
| R_SWIVEL_ORACLE_2D | Left knee | 8.67e-15 / 0 / 5.55e-14 / 1.11e-13 / 1.11e-13 | 1.65e-14 / 0 / 1.11e-13 / 1.67e-13 / 2.22e-13 |
| R_SWIVEL_ORACLE_2D | Right knee | 1.04e-14 / 0 / 5.55e-14 / 5.55e-14 / 1.11e-13 | 2.22e-14 / 0 / 1.11e-13 / 1.11e-13 / 1.67e-13 |

Thus the swivel has exact endpoint preservation and numerical-zero bones, but
MINIMUM_NORM is also at numerical zero on every C field. DEPTH_ONLY exposes
large field-local bone violations (up to 2,952 mm P–M and 2,489 mm M–D).

## 7. Same-C observation attribution, E and D

On the exact same C rows, R_SWIVEL_ORACLE_2D versus R_SWIVEL_OBS corrected
`+7 / -3 / +15 / -1` additional/fewer chains for left elbow, right elbow,
left knee, and right knee respectively. Their per-field hinge flip rates were
`5.93 / 5.25 / 5.41 / 2.07%` (oracle) versus
`6.32 / 4.91 / 7.09 / 2.35%` (OBS); bend MAEs were
`37.25 / 31.31 / 32.95 / 28.23°` versus
`37.59 / 31.54 / 35.65 / 28.72°`. The oracle is only modestly different and
does not close the gap to MINIMUM_NORM. Target-projection mean errors are
`24.13 / 25.72 / 16.39 / 18.23 px` for the oracle and
`24.75 / 26.76 / 17.41 / 19.15 px` for OBS; observation-consistency means
are `24.25 / 27.67 / 23.12 / 20.91 px` versus
`23.37 / 26.72 / 22.22 / 20.02 px`.

E's per-field counts and metrics equal C exactly in this run: every real
observation-eligible chain also had a valid target projection. All E image
consistency metrics have a finite count equal to their field frame count.

D field rows are 335 / 244 / 228 / 370. By construction all three primary
methods corrected those D requests. On those exact rows, OBS flip / bend MAE /
middle-3D error were:

| Field | MINIMUM_NORM | R_SWIVEL_OBS | R_SWIVEL_ORACLE_2D (corrected / unresolved) |
|---|---:|---:|---:|
| Left elbow | 0.90% / 28.94° / 136.24 mm | 3.58% / 35.04° / 133.93 mm | 330 / 5; 2.99% / 35.41° / 134.34 mm |
| Right elbow | 1.23% / 22.07° / 123.18 mm | 2.87% / 28.50° / 116.79 mm | 230 / 14; 4.51% / 29.23° / 113.59 mm |
| Left knee | 1.32% / 21.57° / 85.33 mm | 7.89% / 36.43° / 91.21 mm | 226 / 2; 4.82% / 30.73° / 87.35 mm |
| Right knee | 0% / 21.00° / 72.78 mm | 2.43% / 31.20° / 68.80 mm | 357 / 13; 1.62% / 30.47° / 66.80 mm |

In this table each method's first three values are `flip / bend MAE / M3D`;
the oracle column prefixes its corrected/unresolved counts. D confirms the
same trend as C and is not used to drop unresolved rows from the primary.

## 8. Opposite-request endpoint and architecture verdict

The wrong-sign whole-pose endpoint is evaluated on the 6,966-frame union of
opposite-request C, not all test frames. Per-field corrected/unresolved counts
are:

| Field | DEPTH_ONLY / MINIMUM_NORM | R_SWIVEL_OBS |
|---|---:|---:|
| Left elbow (5,022) | 4,669 / 353 | 3,964 / 1,058 |
| Right elbow (5,255) | 4,970 / 285 | 4,374 / 881 |
| Left knee (5,781) | 5,457 / 324 | 4,737 / 1,044 |
| Right knee (5,635) | 5,321 / 314 | 4,580 / 1,055 |

On that same wrong/C union, whole-pose MPJPE / PA-MPJPE / flip / bend MAE:
DEPTH_ONLY `93.771 / 75.462 mm / 54.973% / 94.531°`, MINIMUM_NORM
`92.286 / 72.876 mm / 54.973% / 94.531°`, and R_SWIVEL_OBS
`87.651 / 67.908 mm / 41.706% / 79.738°`.

**Verdict: C — SWIVEL ABSTRACTION INSUFFICIENT for the stated ownership-frontier
test.** The swivel satisfies the endpoint/bone contract, but on every one of
the four same-C chain populations it has fewer corrected requests, more
unresolved outcomes, higher hinge flip rate, and larger bend-direction error
than MINIMUM_NORM. Crucially, MINIMUM_NORM also preserves both adjacent bone
lengths to numerical precision here, so the swivel does not buy a stronger
measured ownership constraint over that baseline. Oracle 2D changes the result
only modestly and remains behind MINIMUM_NORM on the primary hinge measures;
this is not merely an OBS-only limitation. Field outcomes are directionally
consistent, so the frozen mixed/chain-dependent category is not the best fit.

No wrist/ankle orientation preservation is inferred from XYZ positions; the
downstream rig orientation-lock contract remains as documented. No training,
tuning, solver redesign, cohort change, architecture promotion, or historical
artifact substitution occurred.
