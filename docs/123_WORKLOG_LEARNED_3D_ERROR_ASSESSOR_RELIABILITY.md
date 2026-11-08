# Worklog — Learned Per-Joint 3D Error Assessor Around Frozen S0: Calibration, Triage Utility and Transfer (2026-11-08)

Batch 123. An evidence-only scratch experiment on `arch/single_frame_first`, run unattended on Main: local RTX 5080,
Python `C:\Projects\AnimCV\.venv_build_full`. `docs/next_batch.md` was read only. No LabServer63/69 connection was made.

Every number, test and artifact below is **AGENT-RUN** on Main. The Reviewer has not re-executed any of it. Visual
statements about review panels are the agent's own **AGENT REVIEW**, not user or animator validation.

## 1. Official CASE and Central Intent

**Verdict: CASE B — PARTIAL SUPPORT.** It comes from the frozen rule in `results/decision/decision.json`
(SHA-256 `edb6f391cb6c0776…`).

**Central Intent tested.** Can a separate learned assessor predict the frame/joint-specific 3D error of the frozen S0
FramePose model? It may use only runtime-available observations and the frozen model's own prediction. The questions
were:
- Is it calibrated, and does it transfer beyond raw detector scores and simple unlearned baselines?
- Does it improve a predeclared GT-free-at-decision triage task on unseen videos?

| Gate | Result | Short reason |
|---|---|---|
| V1–V10 validity | **all pass** | §2, §3, §10 |
| P1 Spearman beats every baseline (video CI > 0) | **pass** | §5 |
| P2 AP(material error) beats every baseline | **pass** | §5 |
| P3 calibration (marginal, per-quintile, invalid-input stratum) | **FAIL** | invalid-input q90 coverage 0.737 < 0.80 |
| P4 triage recall@10% beats every baseline | **pass** | §7 |
| P5 false reassurance ≤ 0.25 (overall, arm-invalid) | **pass** | 0.101 overall, 0.0 arm-invalid |
| P6 independent-subject (MPI) within-joint ranking beats C and B2 | **FAIL** | beats C and A, **loses to B/B2** |

- **Why B and not A.** P3 and P6 fail.
- **Why B and not C.** P1, P2 and P4 all pass.

**Historical CASEs are unchanged:** W113 D, W114 D, W115 D, W116 B, W117 C, W118 D, W119 D, W120 B, W121 D, W122 B.
Nothing here reinterprets them.

**What CASE B means here.** On held-out 3DPW videos, under the operational `real_animcv_observation` sensor, a small
learned assessor ranks and triages S0 joint errors better than:
- a constant prior;
- the raw detector score plus validity;
- the same plus a missing-joint count;
- an unlearned reprojection-consistency heuristic.

Its q90 is marginally calibrated (0.909).

It is **not** calibrated on missing-input joints. It loses calibration completely on an independent-subject,
oracle-2D domain. On that domain its within-joint ranking is worse than a simple validity/missing-count table. It also
shows concentrated false reassurance:
- in shoulders, neck and head;
- in one held-out video of seated people (`downtown_runForBus_01`).

On the real GT-free clips it does **not** single out the W122 S1 f251 event. Instead it marks most of S1 as highly
uncertain.

## 2. Starting state, revisions and environment

| Item | Value |
|---|---|
| Instruction blob | `e2b0eb727d2bebfe0fd721adcbf92bc893e02f93`, verified first |
| Starting commit | `b98c329b7e9c7c6b9f8c486eea12ad1852462e96`. It equalled HEAD and `origin/arch/single_frame_first` after fetch. |
| Pre-outcome instrument commit | `a630946`: scripts and tests, committed after the preflight and declaration and **before** any assessor training, calibration or held-out evaluation |
| Final source revision | The commit adding this Worklog. The COMPLETE marker is a separate, final commit. |

The code blobs committed in `a630946` equal the blobs bound in the declaration:

| File | Blob |
|---|---|
| base | `ba1f061f…` |
| stage script | `4bbddcbb…` |
| tests | `cdb6a025…` |

No code changed after the declaration.

**Environment.**
- Windows 11, RTX 5080, Python 3.11.9, Torch 2.13.0+cu130, numpy 1.26.4, scipy 1.17.1. No packages were installed.
- S0 inference uses W102's call pattern: CUDA fp32, batch 512, AMP off, neutral signs.
- Assessor training ran on CPU: deterministic algorithms, 1 thread.
- Scratch workspace: `C:\Users\dna10\animcv-w123`, outside the repository and outside every existing model or checkpoint
  location.

## 3. Sources, GT alignment, provenance and model exposure (Required Evidence 1)

**Frozen predictor.**
- S0 neutral-sign FramePose, checkpoint `6c12fea9…`. It is a geometry-only single-frame 2D→3D model with no image
  tokens.
- The checkpoint payload records:
  - training bank `bank_3dpw_paired_v2`, digest `75519e63…`, TRAIN split;
  - selection on VALIDATION MPJPE, epoch 99, `test_ground_truth_used: false`.
- The checkpoint SHA was checked before preflight, after evaluation and after the MPI stage. It is unchanged.
- No weights were modified or fine-tuned.

**Exposure.**

| Population | Exposure to S0 / the project |
|---|---|
| 3DPW TRAIN | Trained S0. **Not used** by W123: in-sample errors would be biased low. |
| 3DPW VAL | Used for S0 checkpoint selection. Repeatedly design-informing. |
| 3DPW TEST | Never trained or selected on. Repeatedly design-informing (W91–W105). **Not a fresh untouched holdout.** |
| MPI-INF-3DHP train-set annotations | Absent from the S0 bank. Earlier AnimCV temporal-lifter work used this dataset. |

**3DPW target and alignment** (`results/preflight/preflight.json`, `ab577c74…`).
- The W102 observation handoff `7fc15fc8…` and its sidecar `d0e3a879…` verify.
- The bank digest verifies.
- **S0 reproduction:** a fresh Main forward of B_Main and M_Main on VAL and TEST is **byte-identical** to the W102
  stored predictions (max |diff| 0). The four file SHAs equal W102's `s0_report.json`.
- **Raw GT row contract:** the W102 `rows.json` raw-derived GT `G` equals bank `target_3d` exactly on all 10,483
  actor-frames (max |diff| 0.0 m), and all targets are finite.
- **Formula check:** recomputing W102's per-row MPJPE as the 17-joint mean, pelvis-rooted, reproduces it to ≤ 8e-6 mm.
  This confirms units (m), rooting and the camera_root_relative frame.

**Target support.**
- **Primary regime M** (`real_animcv_observation`, W102 M1 RTMPose/RTMDet): only the 6,988 uniquely associated
  same-person rows. On other, unresolved or ambiguous rows the observed subject is another or unknown person, so the
  actor's GT is not a valid target for that observation.
- **Regime B** (`benchmark_detector_observation`): all 10,483 rows.
- No invalid target was zero-filled. Missing-input joints are **not** excluded; they form a reported stratum.

**Splits** (declared before outcomes; group = video name without its numeric suffix):

| Partition | Rule | Videos (groups) | M same-person rows | B actor-frames |
|---|---|---|---:|---:|
| TRAIN | all 3DPW VAL videos | 12 (11) | 2,745 | 3,407 |
| CALIB | TEST groups with sha256(`W123-split-v1:`+group) rank % 3 == 0 | 8 (8) | 1,327 | 2,049 |
| FINAL | remaining TEST groups | 16 (15; 15 videos with M rows) | 2,916 | 5,027 |

- Videos and groups are disjoint across partitions.
- No image SHA crosses a partition.
- `downtown_runForBus_00` and `_01` are one group.
- **Limit:** 3DPW performers recur across sequences without identity labels, and downtown scenes recur across VAL and
  TEST. Subject-level disjointness therefore **cannot be proven** within 3DPW.

**Independent GT domain: MPI-INF-3DHP train-set annotations** (in-repo `datasets/mpi_inf_3dhp`).
- All 16 reassembled `annot.mat` files match the intake-manifest SHA-256.
- Population: 8 subjects × 2 sequences × the 8 'vnect' cameras, every 100th frame. That gives **10,592 rows**:
  - 9,741 with all 12 limb joints in the image;
  - 267 with fewer than 3 valid joints.
- **Regime: `oracle_geometry`.**
  - 2D = annotated projection. A GT audit shows annot2 equals the pinhole projection of annot3 (median 0.004 px).
  - Score is a constant 1.0; validity = inside the 2048 px image.
  - Tokens are derived as in the 3DPW bank, except head = 3DHP head because there is no nose.
- Metrics use the 12 limb joints only. Hip-midpoint rooting is primary; pelvis rooting is the sensitivity variant.
- **Skeleton-convention mismatch** (preflight GT audit, before any outcome):

  | Median length | 3DHP | 3DPW SMPL joints |
  |---|---|---|
  | hip width | 0.25 m | 0.12 m |
  | thigh | 0.48 m | 0.38 m |
  | shoulder width | 0.28 m | 0.375 m |

  Metric error on MPI therefore contains a convention offset that is **not** frozen-model error. This finding is why
  the P6 gate was declared on within-joint ranking, with MPI calibration reported descriptively. The change was made
  before the declaration and before any MPI S0 prediction existed.

**Real GT-free clips.** W121/W122 S1 (kettlebell), S2 (fire juggling) and S3 (bent-over row).
- The arrays come from W122's re-executed perception. They are **byte-equal** to W121's stored arrays.
- Their 3D GT is UNKNOWN. They are not new sources relative to W121 or W122.

## 4. Declared target, features, baselines, training, calibration and gates (Required Evidence 2)

Declaration: `results/declare/declaration.json`, SHA-256 **`14b94d9768c59360f27a15abb1876eb818043c988447459f1574bcdf0b08c843`**,
`outcomes_inspected: false`. It binds:
- the preflight SHA;
- the three code blobs;
- target, regimes, splits, feature allowlist, assessor configuration, baselines and calibration;
- the decision task, metrics, the V1–V10 and P1–P6 gates and the CASE rule;
- the real-video inspection and the review-selection rules.

**Primary target.** The per-joint 3D Euclidean error `e` of the frozen S0 prediction against raw GT. Both are
pelvis-rooted, in camera_root_relative metres, over the 16 non-root joints.

**Output semantics.**
- `q90` is a per-joint error level (m) with declared 0.90 coverage after split-conformal log-space calibration on
  CALIB videos.
- `q50` is the same at 0.50.
- It is **not** a probability of correctness, **not** detector confidence, and not an epistemic/aleatoric decomposition.
- Material event: `e > 150 mm` (primary); `e > 200 mm` reported.

**Feature allowlist** (157 dims; single frame; online-compatible).
- `geometry` (17×4): crop x and y, the **raw detector score unclipped**, validity.
- S0 root-relative prediction.
- Orthographic reprojection residual and scale.
- Predicted bone lengths.
- Invalid-limb count.
- Valid score mean, max and min.

The builder refuses any other key. GT, identities, timestamps, splits, benchmark strata, crop location and hidden
tokens are excluded. No temporal context is used.

**Assessor.**
- An ensemble of 5 MLPs (157→256→256→32, GELU, dropout 0.1).
- Pinball loss at τ = 0.5 and 0.9 on log(e + 1 mm).
- AdamW, lr 1e-3, weight decay 1e-4, 80 epochs, batch 256, seeds 1231–1235.
- No early stopping, no model selection, no sweep.

**Baselines.** Every baseline is fit on TRAIN only and receives the identical CALIB conformal shift.
- **A** — constant per-joint quantiles.
- **B** — table on (joint, own validity, own raw-score TRAIN decile). Scores above 1 sit in the open top bin and are
  never clipped.
- **B2** — B plus the frame invalid-limb count.
- **C** — unlearned reprojection-consistency residual decile.
- A temporal heuristic is **UNAVAILABLE**: S0 is single-frame, and the 3DPW bank cadence (10 Hz, sparse) differs from
  the 30 fps clips.

**CASE rule.**
- D if any V gate fails.
- A if P1–P6 all pass.
- B if not A and at least one of P1, P2 or P4 passes.
- Otherwise C.

## 5. Training, calibration and held-out results (Required Evidence 3)

**Training** (`train_r1`, `train_r2`: two independent processes).
- Assessor weights `368d3a38…` are **identical** in both runs.
- Calibration file `c1599c2e…` is identical, and the FINAL output fingerprint (computed without GT) is identical.
- Mean pinball loss went from 0.77 in epoch 1 to 0.107–0.109 at epoch 80 (5 seeds). Training took 15 s.
- In-sample TRAIN q90 coverage was 0.996, so the assessor over-fits.
- The CALIB conformal shift was +0.326 in log space (×1.39) for q90 and +0.035 for q50.
- After the shift, CALIB coverage is exactly 0.900 / 0.500 for every predictor, by construction.
- CALIB Spearman: learned 0.704, baselines 0.590–0.617 (development data, shown for transparency).

**FINAL — primary M regime** (`results/evaluate/evaluate.json`, `5c2da899…`).
- 2,916 rows, 46,656 joint-cells, 15 videos.
- Errors: mean 91.4, median 59.2, p90 187.3, max 1,161 mm.
- 6,914 material cells (14.8%).
- Raw score domain 0.300–1.125, with 454 valid cells above 1.

| Predictor | Spearman(q50,e) | within-joint Spearman | AP(e>150) | q90 cov | q50 cov | triage recall@10% | excess capture | false reassurance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LEARNED** | **0.681** | **0.339** | **0.591** | 0.909 | 0.536 | **0.384** | **0.601** | 0.101 |
| A constant | 0.596 | 0.000 | 0.332 | 0.925 | 0.591 | 0.274 | 0.292 | 0.005 |
| B score+validity | 0.607 | 0.158 | 0.442 | 0.931 | 0.587 | 0.322 | 0.426 | 0.052 |
| B2 + missing count | 0.606 | 0.157 | 0.443 | 0.931 | 0.585 | 0.320 | 0.424 | 0.045 |
| C reprojection | 0.630 | 0.180 | 0.460 | 0.930 | 0.585 | 0.309 | 0.399 | 0.038 |

The AP base rate is 0.148.

Learned minus each baseline. Video-cluster bootstrap over 15 videos, 2,000 replicates, seed 1337; the within-joint
column uses 500 replicates.

| vs | Spearman | AP(material) | triage recall | within-joint Spearman |
|---|---|---|---|---|
| A | +0.085 [+0.047, +0.119] | +0.259 [+0.179, +0.328] | +0.110 [+0.053, +0.166] | +0.339 [+0.266, +0.392] |
| B | +0.074 [+0.049, +0.098] | +0.149 [+0.105, +0.181] | +0.062 [+0.030, +0.091] | +0.181 [+0.117, +0.223] |
| B2 | +0.075 [+0.049, +0.099] | +0.149 [+0.107, +0.176] | +0.064 [+0.034, +0.091] | +0.183 [+0.117, +0.226] |
| C | +0.051 [+0.020, +0.078] | +0.131 [+0.082, +0.170] | +0.074 [+0.043, +0.108] | +0.160 [+0.083, +0.215] |

These are paired comparisons on identical observations and identical target support.

**Calibration by FINAL q90 quintile (learned).**

| Quintile | q90 range (mm) | q90 coverage | q50 coverage | median q50 / median e (mm) | mean e | p90 e | material rate |
|---|---|---:|---:|---|---:|---:|---:|
| 0 | 17–69 | 0.868 | 0.451 | 22 / 23 | 28 | 51 | 0.5% |
| 1 | 69–121 | 0.892 | 0.479 | 46 / 48 | 58 | 102 | 3.2% |
| 2 | 121–166 | 0.924 | 0.556 | 67 / 62 | 73 | 130 | 6.2% |
| 3 | 166–248 | 0.931 | 0.588 | 95 / 83 | 97 | 178 | 15.9% |
| 4 | 248–1,863 | 0.933 | 0.606 | 162 / 145 | 201 | 447 | 48.2% |

The low quintiles slightly under-cover. The high quintiles over-cover. Mean error rises monotonically, from 28 to 201 mm.

**Strata (learned unless stated).**

| Stratum | n cells | mean e (mm) | material | q90 cov | Spearman | false-reassurance n |
|---|---:|---:|---:|---:|---:|---:|
| input **invalid** | 1,062 | 589 | 1,061 | **0.737** (B2/C 0.446, A 0.108) | 0.268 | 0 |
| input valid | 45,594 | 80 | 5,853 | 0.913 | 0.659 | 700 |
| arm with invalid input | 477 | 548 | 476 | 0.732 | −0.044 | 0 |
| benchmark-invisible proxy | 2,975 | 278 | 1,646 | 0.854 | 0.802 | 47 |
| raw score > 1 | 454 | 76 | 15 | 0.960 | 0.336 | 6 / 15 |
| near-straight hinge (<15°) | 4,316 | 74 | 359 | 0.935 | 0.317 | 61 |
| high GT speed (≥ 0.474 m/s, TRAIN q75) | 5,909 | 133 | 1,626 | 0.913 | 0.437 | 93 |
| torso / head joints | 11,664 | 57 | 506 | 0.916 | 0.556 | 201 (rate 0.397) |
| arms | 17,496 | 114 | 3,726 | 0.911 | 0.525 | 337 (rate 0.090) |
| legs | 17,496 | 92 | 2,682 | 0.903 | 0.758 | 162 (rate 0.060) |

Facing strata (q90 coverage):

| Facing | Coverage |
|---|---:|
| frontal | 0.877, the worst |
| profile | 0.921 |
| near-frontal | 0.921 |
| back | 0.928 |

**How marginal calibration hides failure.** The marginal q90 coverage of 0.909 looks good. Underneath it:
- **Invalid inputs under-cover.** Coverage there is 0.737, against mean errors of ~0.59 m, but these joints are only
  2.3% of cells. The pooled conformal shift is dominated by the valid majority.
- **Joint coverage varies:** from 0.866 (right hip) and 0.879 (left knee) up to 0.948 (right elbow).
- **Video coverage varies:** from 0.744 (`downtown_runForBus_01`) up to 0.962 (`flat_packBags_00`).

An occluded arm or a new motion domain can therefore be badly mis-calibrated while the pooled number passes.

**Cross-sensor B regime** (descriptive; the assessor was not retrained). 5,027 rows, 80,432 cells, 15,091 material.

| Predictor | Spearman | AP | triage recall | q90 cov (M calibration) | false reassurance |
|---|---:|---:|---:|---:|---:|
| Learned | 0.647 | 0.606 | 0.360 | 0.871 | 0.109 |
| C | 0.636 | 0.546 | 0.309 | 0.905 | — |
| B2 | 0.586 | 0.516 | 0.306 | 0.888 | — |
| A | 0.596 | 0.384 | 0.248 | 0.889 | — |

- With M calibration the learned assessor under-covers (0.871).
- Recalibrating on B CALIB gives 0.880. It still under-covers, so the calibration does **not** transfer across sensors
  without loss.

## 6. High-confidence wrong, occlusion and domain-shift failures

Selections follow the declared rules. Panels: `results/review/review_panels.jpg`. Manifest `4c8ed397…`; every image SHA
matches W102's manifest. These are AGENT REVIEW observations.

**Well-observed correct.** Hips on upright walkers have e of 1–2 mm and q90 of 29–37 mm. Correctly reassuring.

**High-confidence wrong (false reassurance).**

| Case | e | q90 | Score | 2D | What the panel shows |
|---|---:|---:|---|---|---|
| `downtown_runForBus_01` actor0 #165/#170/#175, head | 467–469 mm | 116–119 mm | 0.92–0.96 | complete, valid | A seated person beside a second seated person. Agent inference: a whole-body seated-pose reconstruction failure that the clean 2D does not reveal. |
| `office_phoneCall_00` #609, right elbow | 481 mm | 146 mm | 0.64 | valid | The subject is truncated at the image edge. |

This video is the worst calibrated one (q90 coverage 0.744; false-reassurance rate 0.452).

False reassurance is concentrated in shoulders (rates 0.36 and 0.52), neck (0.42) and head (0.31). These are mostly
moderate exceedances just above 150 mm.

**Missing or occluded.** Ankles occluded behind the store display in `downtown_enterShop_00`, and the left wrist in a
hug in `downtown_warmWelcome_00`:
- e of 0.60–0.95 m;
- q90 of 1.74–1.86 m;
- correctly flagged.

**Fast but valid motion.** Running wrists in `runForBus` have e of 31–84 mm and q90 of 253–382 mm. These are
conservative flags on valid motion.

**Score > 1.** Heads and ankles with scores 1.10–1.12 have e of 23–64 mm and q90 of 139–264 mm. The score is retained
raw. Within the score > 1 stratum, 6 of 15 material errors are falsely reassured.

**False positives.** Dark `downtown_bar_00` and phone-holding `downtown_crossStreets_00` wrists have e of 16–47 mm and
q90 of 615–668 mm.

**Domain-transfer failure (MPI).** Right wrists in S5/S1/S8 have e of 1.09–1.22 m with q90 of 95–123 mm. That is
high-confidence wrong on an independent subject and view.

## 7. Decision / triage utility, separate from correlation (Required Evidence 5)

The declared task is **TRIAGE UTILITY**: per video, flag the top ⌈10%⌉ of joint-cells by q90, with a label-free
blake2b tie-break. GT is used only for scoring, after all decisions were frozen.

**FINAL M.**
- Learned flags 4,671 cells. 2,652 are material, from 6,914 material cells in total.
- That is recall **0.384**, precision 0.568 and excess-error capture 0.601 (false-negative rate 0.616).
- Best baseline recall is 0.322 (B).
- The best achievable recall at this budget is about 0.67, because there are more material cells than budget slots.

**Per video.**
- Learned recall beats the best baseline in 11 of 15 videos. AP beats it in 13 of 15, and Spearman in 11 of 15.
- Losses:
  - `flat_guitar_01`: 0.184 vs 0.227;
  - `office_phoneCall_00`: 0.491 vs 0.541;
  - `downtown_sitOnStairs_00` and `downtown_warmWelcome_00`: small losses.

**Secondary selective acceptance.** Accepting the 80% of cells with the lowest q90 gives mean error 64.1 mm (all cells:
91.4) and a material rate of 6.5% (all: 14.8%). The best baseline gives 68.8 mm and 8.5%.

**Negative and no-op controls.**
- A constant risk flags exactly the budget, chosen by the tie-break alone (unit test).
- The constant baseline's triage recall of 0.274 is the measured no-information reference.

**Non-claims.**
- This is triage of suspect joint-cells for review.
- It is **not** Generator decision utility, not paired-candidate selection (no second frozen candidate was used) and not
  animation improvement.
- The synthetic oracle upper bound was not used as a result.

## 8. Independent-subject transfer (MPI-INF-3DHP, `oracle_geometry`)

`results/mpi/mpi.json`, `df4eaa3f…`. S0 ran on oracle 2D twice; the outputs are byte-identical.

**Hip-midpoint rooted (primary).** 127,104 limb cells. Errors: mean 216, median 171, p90 431 mm. 56% of cells are
material. Part of this is the skeleton-convention offset (§3).

| Predictor | Spearman | within-joint | AP (base 0.561) | q90 cov | false reassurance |
|---|---:|---:|---:|---:|---:|
| Learned | **0.535** | 0.180 | **0.794** | **0.421** | 0.260 |
| A | 0.465 | 0.000 | 0.718 | 0.535 | 0.005 |
| B | 0.484 | **0.287** | 0.727 | 0.311 | 0.415 |
| B2 | 0.497 | 0.244 | 0.728 | 0.320 | 0.415 |
| C | 0.502 | 0.064 | 0.768 | 0.541 | 0.044 |

Learned minus baseline. Subject-cluster bootstrap over 8 subjects, 2,000 replicates.

| vs | within-joint Spearman | pooled Spearman | AP |
|---|---|---|---|
| A | +0.180 [+0.158, +0.207] | | |
| C | +0.116 [+0.091, +0.151] | +0.033 [+0.021, +0.050] | +0.026 [+0.020, +0.033] |
| **B2** | **−0.064 [−0.086, −0.038]** | +0.039 | +0.066 |
| **B** | **−0.106 [−0.131, −0.078]** | | |

**P6 fails.** On MPI the score is a constant 1.0, so B's within-joint ranking comes only from in-image validity. That
validity information beats the learned assessor within joints. The learned pooled ranking and AP are still better,
largely through the joint prior.

**Calibration collapses.** Learned q90 coverage is 0.421:
- 0.016 in the lowest-q90 quintile, where median predicted q50 is 11 mm against a median error of 80 mm;
- 0.36–0.47 per subject.

The pelvis-rooted variant is similar (coverage 0.416). This is reported descriptively, because the skeleton convention
confounds it. But the high-confidence >1 m wrist errors (§6) are not explained by a convention offset.

**Separated calibration summary (learned q90 coverage).**

| Domain | Coverage |
|---|---:|
| TRAIN in-sample (3DPW VAL) | 0.996 |
| CALIB | 0.900, by construction |
| FINAL held-out 3DPW TEST (M) | 0.909 |
| Cross-sensor B | 0.871 |
| Independent MPI | 0.421 |

## 9. GT-free real-clip stress test (Required Evidence 4)

`results/realvideo/realvideo.json`, `41ba3a32…`. The frozen W122 arrays were used and no inference was rerun.

| Clip | Frames | Invalid-input cells | Score > 1 cells | q90 median / p90 / max (mm) | cells with q90 ≥ 150 mm |
|---|---:|---:|---:|---|---:|
| S1 kettlebell (side view) | 1,631 | 723 | 0 | 252 / 769 / 2,050 | **73.4%** |
| S2 fire juggling | 895 | 154 | 0 | 133 / 305 / 1,422 | 38.1% |
| S3 bent-over row | 154 | 0 | 0 | 116 / 220 / 391 | 32.8% |

For comparison, the per-clip flag rates of the constant baseline are about 75%, and those of B are 50–80%. On S2 and S3
the learned assessor flags less than every baseline.

**S1 f251 event** (predeclared frames 241–261, left arm plus right wrist).
- **Before the event** (f241–f250): every inspected joint is input-valid with a moderate score. Left-wrist scores are
  0.46–0.50; right-wrist scores 0.34–0.40. Left-wrist q90 is already **793–988 mm** (clip percentile 0.53–0.87).
- **At f251:** the S0 root-relative left wrist moves **368 mm** in one frame, and the left elbow 362 mm. But left-wrist
  q90 *decreases* to **756 mm** (clip percentile **0.48**). Left-elbow q90 is 421 mm (percentile 0.39). The right
  wrist is at q90 718 mm.
- **f252–f261:** left-wrist q90 is 559–677 mm.

The assessor therefore labels the whole window as uncertain but does **not** single out f251. It has no temporal input,
so it cannot see a one-frame discontinuity. The same holds for the raw-score baselines: B2 gives a constant 555 mm for
the left wrist in f241–f253.

**Predeclared unstable frames** (the largest wrist steps per clip). All are flagged on S1, S2 and S3. But:
- On S1 and S2 every other frame is flagged too.
- On S3, 63% of other frames are flagged under the absolute rule and 34% under the clip top-10% rule.
- The Spearman correlation between wrist step size and wrist q90 is −0.20 on S1, +0.24 on S2 and +0.03 on S3.

**Interpretation (agent inference).**
- On S1 the flags are indiscriminate. 723 cells have invalid input, and the side view with self-occluded arms is out of
  support. High uncertainty everywhere gives no ranking of normal versus suspect motion.
- **False-reassurance examples cannot be identified without GT.**
- A high q90 at f251 does not show that S0 is wrong there. The 3D error on S1–S3 is UNKNOWN.
- Video-person identity stays `UNKNOWN` in every sidecar. W122 verified person continuity at window level only.

## 10. Scratch sidecar, GT isolation and hard-lock invariants (Required Evidence 6)

Sidecars `sidecar_S1/S2/S3.json` (`bd3059b6…`, `83f0bdca…`, `05c5528f…`) bind:
- source clip SHA, observation-npz SHA, frame index and timestamp (30 fps);
- the 16 semantic joint names and the coordinate frame;
- S0 checkpoint SHA, pose-hypothesis array SHA and observation array SHA;
- reliability type, units, quantiles, assessor ID `368d3a38…` and calibration ID `c1599c2e…`;
- support codes (`SUPPORTED_OBSERVED` / `SUPPORTED_INPUT_MISSING` / `UNKNOWN`);
- the raw detector score, in a separately named field;
- `subject_identity: UNKNOWN`.

The validator reports zero violations on all three clips.

The sidecar keeps every source frame, so nothing is collapsed to keyframes. It contains no GT. Diagnostic GT errors live
only in the evaluation artifacts (`final_*_cells.npz`, `evaluate.json`, `mpi.json`), which no runtime function reads.

**Synthetic contract tests** (all pass) prove:
- the feature allowlist refuses `target`, `gt_3d`, `error` and `sample_id`;
- inference outputs are independent of GT;
- scores above 1 are preserved unclipped in features and sidecar;
- non-binary validity is refused;
- missing joints carry `SUPPORTED_INPUT_MISSING`;
- reordered frames are rejected at build and at validation;
- the validator rejects all of:
  - a stale checkpoint;
  - a stale assessor or calibration;
  - a changed joint order;
  - a GT field;
  - a clipped raw score;
  - a negative reliability value;
  - wrong support flags;
  - a modified pose hypothesis.

**Soft vs hard.** `illustrative_future_consumer` is labelled `ILLUSTRATIVE_NOT_POLICY`. It shows how a future consumer
could keep the two separate:
- Uncertainty may only scale a soft reference weight.
- User hard pins are copied unchanged: they stay byte-identical whether uncertainty is 0.01 m or 5 m, and they are never
  removed.

No Generator was implemented and no trust mapping was chosen. Production MotionGraph, AnimationClip, FBX and keyframe
schemas are untouched.

## 11. Negative findings, alternatives and failure mechanisms

- **Missing-input joints** carry the largest errors (~0.59 m mean) and are under-covered (0.737). Within the arm-invalid
  stratum the learned ranking has no signal (Spearman −0.04).
- **Clean 2D does not reveal some whole-body failures.** The seated people in `runForBus_01` show this (§6).
- **Torso, head and shoulders** account for most false reassurance.
- **Sensor transfer:** calibration degrades from M to B.
- **Domain transfer:** calibration collapses on MPI. Within joints, a simple validity table ranks better than the
  learned assessor there.
- **No temporal awareness:** the S1 f251 jump is not singled out.
- **S1 out of support:** flags are indiscriminate.
- **Over-fitting:** the assessor over-fits TRAIN (in-sample coverage 0.996). The conformal shift corrects this only
  marginally, in-domain.
- **Alternative explanation for the in-domain gain.** The 3DPW TRAIN, CALIB and FINAL partitions share performers and
  locations. Part of the advantage may be performer or scene familiarity rather than generalizable error modelling.
  MPI's weaker within-joint result is consistent with that.

## 12. Source preservation, changed files, tests and reproduction

**Changed tracked files (all new).**
- `scripts/w123_reliability_base.py`
- `scripts/w123_reliability_assessor.py`
- `tests/test_w123_reliability_assessor.py`
- this Worklog

No `src/`, model, checkpoint, dataset, manifest, rig, W102–W122 artifact, historical Worklog, `AGENTS.md` or
`docs/next_batch.md` was changed.

**Tests executed.**
- `python -X utf8 -m pytest tests/test_w123_reliability_assessor.py tests/test_canonical_pose_parity.py -q -p no:cacheprovider --junitxml=…`
- Result: 35 passed (28 W123 + 7 canonical-pose parity), 0 failed, 0 skipped. JUnit `7ff24cd2…`; summary `8eec1426…`.
- The W123 suite includes a GPU frozen-S0 regression: the first 512 VAL rows are byte-identical to W102, and the
  checkpoint SHA is unchanged.
- **NOT RUN:** the full Pytest suite; W102–W122 regression suites; any Blender, rig or render test (no rig output was
  produced); any LabServer job.

**Reproduction** (Main):

```text
P=C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe
$P -X utf8 scripts/w123_reliability_assessor.py <stage>
```

Stage order:
1. `preflight`
2. `declare`
3. `train --tag r1`
4. `train --tag r2`
5. `evaluate`
6. `mpi`
7. `realvideo`
8. `review`
9. `tests`
10. `decide`

Outputs go under `%USERPROFILE%\animcv-w123\results`. Stages refuse to overwrite existing outputs. Inputs are read from
`animcv-w101/data`, `animcv-w102`, `animcv-w121`, `animcv-w122` and the in-repo MPI annotations.

**Evidence.**
- Manifest: `results/evidence_manifest.json`, SHA-256 `f4254fcd…`, 35 files, 22 MB, Main-local. The MPI reassembled
  `.mat` files are outside it, under `data/`.
- Key hashes: preflight `ab577c74…`, declaration `14b94d97…`, evaluate `5c2da899…`, MPI `df4eaa3f…`, realvideo
  `41ba3a32…`, decision `edb6f391…`, assessor file `e92d0b6c…`.

**Process disclosures.**
1. A **smoke run** (`results_smoke`) exercised every stage before the declaration:
   - 2-epoch, 1-seed assessor;
   - "FINAL" relabelled to TRAIN rows only;
   - synthetic MPI targets.

   It also ran the GT-free real-clip stage with that throwaway model. No held-out 3DPW or MPI error was computed, and no
   declared parameter changed afterwards.
2. A **first declaration** was superseded before any outcome. It is archived as
   `results_smoke/declare_superseded_before_outcomes_crlf_hash`. The only change was the JSON writer, which wrote CRLF
   so that printed hashes differed from file hashes.
3. The preflight GT audit revealed the MPI skeleton-convention mismatch **before** the declaration. P6 was therefore
   declared on within-joint ranking with hip-midpoint rooting. No MPI S0 prediction existed at that time.

## 13. Unresolved uncertainty and architecture hypothesis status

**Resource and environment limits.**
- Single Main machine.
- Assessor training on CPU.
- No independent GT-supported real-sensor dataset other than 3DPW is available on Main. The MPI test set and BEDLAM are
  not present.
- MPI provides only an oracle regime with a different skeleton.

**Unresolved.**
- Whether the in-domain advantage survives truly new performers and sensors.
- Whether the missing-input under-coverage can be fixed without a separate calibration per support class. This was not
  tried: no post-hoc tuning was done.
- Whether any temporal or offline signal would flag one-frame source jumps. Out of scope here.
- The true 3D error on S1–S3.
- Statistical intervals rest on 15 FINAL videos and 8 MPI subjects. Frames within a video are correlated, and
  conformal exchangeability across videos is assumed, not established.

**Architecture hypothesis status.**
- A model-specific per-joint 3D error signal **can** be learned without GT contamination. It is decision-useful for
  in-domain triage beyond raw scores and simple heuristics (reference-supported, §5 and §7).
- It is **not** demonstrated to be calibrated on missing joints, across sensors or across domains. It does **not**
  detect temporal source jumps.
- Generator use of such a signal is UNPROVEN and was not tested.
- The sidecar is a defensible scratch contract that keeps soft reliability separate from hard pins and from the raw
  detector score. It is not a Production schema.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 123
