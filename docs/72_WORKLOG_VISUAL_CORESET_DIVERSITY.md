# Worklog — Visual Coreset Diversity / Within-Sequence De-duplication (2026-10-06)

## Starting state and files

- **Start:** fetched `origin/arch/single_frame_first` and confirmed local = remote = `3784ed62f5bd2350a7451ed559ba3254049af978`, with a clean worktree.
- **Commits:**
  - `17b6734`: coreset module, script, tests
  - the commit adding this worklog
- **Files added:**
  - `src/framepose/visual_coreset.py`
  - `scripts/run_visual_coreset_diversity.py`
  - `tests/test_framepose_visual_coreset.py`
  - this worklog
- **Unchanged:** every existing file and checkpoint, including the Worklog 68 cache, the Worklog 69 head and C1 checkpoint, Worklog 70, the Worklog 71 S0/S1 checkpoints and outputs, FrameBank splits, frozen H0, Worklog 65 D1, production FramePose, FK and IK.

**Run note.** The first attempt (2026-10-04, via the server-side wait loop) started as soon as the GPU looked free, but an unrelated project re-occupied the GPU at the same moment. The run died with CUDA OOM before training and wrote no outputs. LabServer63 was then unreachable for about two days (SSH timeouts). The run was relaunched unchanged on `17b6734` on 2026-10-06, with the GPU idle, and completed. No other job was touched.

## Fixed contract

- **Primary baseline:** Worklog 71 **S1_SEQUENCE_BALANCED**, reused as-is (checkpoint, training record and per-row held-out predictions). Its sampler is replayed only for accounting, and the replay must reproduce the per-sequence draw counts Worklog 71 recorded. It did.
- **What S2 changes:** S2 uses the identical `EpochSampler` code, seed and sequence draws. Only each sequence's candidate pool changes, from all eligible frames to that sequence's fixed visual coreset.
- **Training:** S2 is trained by the Worklog 71 `train()` function, imported unchanged.
- **Architecture:** the fixed Worklog 69 C1 head (13 pair-geometry features + endpoint A/B/global readouts, shared `Linear(4096→64)`, per-segment `Linear(205→64)→GELU→Linear(64→1)→tanh`, 301,955 parameters).
- **Optimization:** seed 1337 and identical initialization; AdamW, learning rate 3e-4 cosine to 1e-5, weight decay 1e-4; batch 256, 200 epochs, masked MAE; selection by validation mean \|f\|.
- **Runtime identity checks (each aborts the run on mismatch):**
  - Worklog 68 cache: bank digest, sample-ID order, Qwen3-VL fingerprint `a0d72ded…`, `readout.npz` hash
  - Worklog 69 C1 cache identity and Worklog 71 S1 config/sampler
  - eligible TRAIN population equal to Worklog 71 (34 sequences, 10,897 rows)
  - S1/S2 sequence draw streams identical
  - S1 replay = Worklog 71 recorded contribution
  - S2 never draws outside its declared coreset
  - S2 draws and steps equal to S1's
- **Not used:** Worklog 70 attention, Depth Anything, H0 as input, oracle signs, temporal context, external data.

## Eligible TRAIN composition and K

- **Eligible TRAIN pool:** 34 actor sequences, 10,897 eligible rows (11,334 TRAIN rows). Eligible means at least one GT-valid arm segment. These are the same population and masks as Worklog 71.
- **Sequence sizes:** median 284 eligible frames, maximum 628 (`courtyard_backpack_00` actor0), minimum **127** (`courtyard_dancing_01` actor0).
- **K = 127,** the minimum eligible TRAIN frames over sequences, derived from composition only.
- **Coreset size:** every sequence gets exactly 127 unique frames, 4,318 frames in total (39.6% of the pool). `courtyard_dancing_01` actor0 keeps all its frames. Coreset-positions SHA-256: `847776fb…`.
- **Feature:** L2-normalized concatenation of the frozen Worklog 68 shoulder, elbow and wrist readouts (global token excluded).
- **Selection:** cosine distance; start at the frame closest to the sequence centroid; farthest-point additions; ties go to the smallest FrameBank position. No GT, target, H0, Depth Anything, error, threshold or randomness.

## Does the coreset reduce visual redundancy? Yes

| measure | full eligible pool | coreset |
|---|---|---|
| pooled nearest-neighbour cosine, mean (p05 / p50 / p95) | **.919** (.825 / .928 / .982) | **.843** (.765 / .845 / .926) |
| per-sequence mean pairwise cosine, mean over 34 sequences (p05 / p95) | .662 (.613 / .724) | .629 (.568 / .693) |

- **Every sequence that loses frames gets less redundant:** both measures drop in 33 of 34 sequences (all except `courtyard_dancing_01` actor0, where the coreset is the whole sequence).
- **Biggest reductions:**
  - pairwise: `outdoors_climbing_00` .766 → .695, `bodyScannerMotions_00` .715 → .644, `outdoors_climbing_02` .617 → .549
  - nearest-neighbour: `courtyard_backpack_00` .928 → .812, `jacket_00` .920 → .798

The near-duplicate structure measured in Worklog 71 is substantially removed: the nearest-neighbour cosine p95 falls from .98 to .93. The manipulation is therefore valid.

**Post-hoc target composition (evaluation-only diagnostic, computed after the run; it played no role in selection).** The fraction of strongly out-of-plane targets (\|f\| ≥ 0.5) is *higher* in the coreset than in the full pool:

| segment | full pool | coreset |
|---|---|---|
| forearm | 27.0% | 29.6% |
| chain | 18.9% | 21.7% |
| upper | 21.6% | 27.2% |

Visual farthest-point selection does not starve the out-of-plane regime.

## S1 / S2 contribution accounting

| sampler | total draws | optimizer steps | per-sequence share (largest / smallest) | max deviation from uniform | unique frames visited | repeats per visited frame, mean (p05 / p50 / p95) |
|---|---:|---:|---|---:|---:|---|
| S1 (all eligible frames) | 2,266,800 | 9,000 | .0296 / .0291 | 0.0003 | 10,897 | 208 (100 / 179 / 409) |
| S2 (coreset) | 2,266,800 | 9,000 | .0296 / .0291 | 0.0003 | 4,318 | 525 (488 / 525 / 563) |

- **Sequence draws are identical draw-for-draw**, not just within sampling noise, because both samplers consume the same generator stream for the sequence choice.
- **The budget is identical,** and all S2 draws are eligible.
- **The only difference is the within-sequence pool.** Each S2 frame is seen about 2.5× more often, and S2 repeats are far more uniform across frames than S1's, which depend on sequence size.

## Training curves

Validation MAE is the selection metric. Train MAE is always evaluated on the **full** eligible TRAIN pool, including frames that are not in the S2 coreset.

| epoch | S1 train | S1 validation | S2 train | S2 validation |
|---:|---:|---:|---:|---:|
| 9 | .1247 | .2050 | .1300 | **.2076** ← S2 selected |
| 19 | .1018 | .2061 | .1220 | .2083 |
| 29 | .0879 | .2025 | .0985 | .2078 |
| 49 | .0686 | .2000 | .0874 | .2142 |
| 69 | .0596 | **.1970** ← S1 selected | .0841 | .2189 |
| 99 | .0488 | .2014 | .0766 | .2140 |
| 149 | .0390 | .2032 | .0721 | .2159 |
| 199 | .0359 | .2044 | .0706 | .2162 |

| sampler | selected epoch | train | validation | gap | ratio | final ratio |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 69 | .0596 | **.1970** | .137 | 3.30 | 5.69 |
| S2 | 9 | .1300 | **.2076** | .078 | 1.60 | 3.06 |

- **S2 is worse on validation at every evaluated epoch.** Its best validation (epoch 9) is 0.0106 worse than S1's, and it degrades from there.
- **S2's smaller gap and ratio come entirely from a worse training fit** (the head fits the 60% of TRAIN frames it never sees less well). Held-out error is not better, so this is not a generalization gain.

## Held-out results (S2 − S1)

**Test** (6,003 rows). Cells give mean / p50 / p95. "Improved share" is the fraction of rows where S2's error is lower. S0 = Worklog 69 C1.

| segment | metric | S1 | S2 | S2 − S1 mean (p50 / p95) | improved share | S0 | H0 | D1 |
|---|---|---|---|---|---:|---:|---:|---:|
| **forearm** | \|f\| | .242/.185/.672 | .264/.195/.765 | **+0.022** (+.014 / +.331) | 46% | .243 | .283 | .265 |
| | elevation (°) | 16.1/12.3/43.9 | 17.6/13.0/50.9 | **+1.46°** | 46% | 16.2 | 18.7 | 17.6 |
| | full (°) | 17.9/13.9/46.8 | 19.4/14.5/54.4 | **+1.46°** | 46% | 18.1 | 20.4 | 19.4 |
| **chain** | \|f\| | .249/.191/.704 | .252/.184/.751 | +0.003 (−.003 / +.274) | 51% | .251 | .271 | .224 |
| | elevation (°) | 15.7/11.7/45.0 | 15.8/11.4/47.4 | +0.16° | 51% | 15.8 | 16.9 | 14.1 |
| | full (°) | 16.7/12.6/46.6 | 16.9/12.4/49.1 | +0.17° | 51% | 16.8 | 17.9 | 15.2 |
| upper (secondary) | \|f\| | .274/.213/.746 | .263/.200/.754 | −0.011 (−.015 / +.258) | 54% | .272 | .237 | .209 |
| | full (°) | 17.8/13.8/46.4 | 17.2/13.0/48.0 | −0.67° | 54% | 17.7 | 15.7 | 13.9 |

**Validation** (3,056 rows):

| segment | \|f\| S1 → S2 (S2 − S1) | improved share | full (°) S1 → S2 | p95 \|f\| S1 → S2 |
|---|---|---:|---|---|
| **forearm** | .220 → .238 (**+0.018**) | 46% | 15.7 → 16.8 | .692 → .772 |
| **chain** | .179 → .194 (**+0.014**) | 46% | 12.1 → 12.9 | .510 → .539 |
| upper | .185 → .186 (+0.002) | 49% | 13.1 → 13.3 | .477 → .504 |

**Sign vs magnitude (stable rows, sin 10° evaluation threshold).** S2 is worse on both the sign and the magnitude for the primary segments.

| scope / segment | stable sign accuracy S1 → S2 | magnitude error when both sign-correct, S1 → S2 |
|---|---|---|
| test forearm | .834 → .814 | .197 → .222 |
| test chain | .793 → .779 | .198 → .200 |
| validation forearm | .846 → .818 | .196 → .226 |
| validation chain | .873 → .847 | .165 → .176 |

## Depth regimes (test \|f\|, S1 → S2; evaluation-only bins)

| segment | near plane (\|f\| < .174) | intermediate | strongly out of plane (≥ .5) |
|---|---|---|---|
| forearm | .181 → .182 (n 1,802) | .246 → .256 (n 2,002) | .289 → **.340** (n 2,199) |
| chain | .179 → **.169** (n 2,117) | .238 → .230 (n 2,453) | .371 → **.412** (n 1,433) |
| upper | .261 → .240 | .233 → .227 | .452 → .450 |

- **Near-plane false depth:** a small chain improvement (−0.010) and no forearm change.
- **Out-of-plane accuracy:** clearly worse for both primary segments (forearm +0.051, chain +0.041), even though the coreset contains a *larger* share of out-of-plane TRAIN targets.
- **Validation shows the same pattern:** out-of-plane forearm .352 → .418, chain .258 → .304; near-plane roughly unchanged.

## Clean / crosscountry / other (\|f\|, S1 → S2)

| scope | forearm | chain | upper |
|---|---|---|---|
| clean dancing + hug (validation, 425) | .205 → .206 (50% improved) | .175 → .176 (50%) | .185 → .163 |
| crosscountry (validation, 133) | .222 → **.251** | .217 → .223 | .216 → .219 |
| all other held-out (8,501) | .236 → **.258** (46%) | .228 → .235 (49%) | .247 → .241 |

- **The clean subset is preserved** for forearm and chain. Upper improves there, but upper is secondary.
- **Crosscountry and the large "other" scope regress** for the primary segments.

## Per-sequence held-out accounting

Gain is the S1 − S2 sum of \|f\| error per sequence, so positive means S2 is better.

| scope / segment | sequences improved / regressed | net gain (sum \|f\|) | largest share of positive gain | worst regressions (sum \|f\| increase) |
|---|---|---:|---|---|
| test forearm (37) | 13 / 24 | **−132.4** | 19% (`downtown_downstairs_00` a0) | `downtown_bar_00` a0 (+57.3), `downtown_warmWelcome_00` a0 (+14.7), `downtown_arguing_00` a1 (+14.3) |
| test chain (37) | 20 / 17 | **−16.3** | 14% (`downtown_walking_00` a1) | `downtown_stairs_00` a0 (+18.2), `downtown_bus_00` a1 (+14.6), `downtown_bar_00` a0 (+13.4) |
| held-out forearm (53) | 18 / 35 | **−187.2** | 16% | `downtown_bar_00` a0 (+57.3), `courtyard_drinking_00` a1 (+18.0), `downtown_warmWelcome_00` a0 (+14.7) |
| held-out chain (53) | 24 / 29 | **−59.6** | 12% | `downtown_stairs_00` a0 (+18.2), `courtyard_drinking_00` a1 (+16.1), `downtown_bus_00` a1 (+14.6) |
| test upper (secondary) | 24 / 13 | +66.8 | 32% of net (`downtown_sitOnStairs_00` a1) | `outdoors_fencing_01` (+32.0) |

- **The forearm regression is broad:** two-thirds of held-out sequences get worse, and no single scene owns it. `downtown_bar_00` actor0 is the largest single contributor, at 43% of the test net loss and 31% of the held-out net loss. Even without it, the forearm is net worse in both scopes.
- **The chain is a net loss with a roughly even split** of improving and regressing sequences.

## Architecture classification

**CASE B: current 3DPW visual diversity is insufficient.**

S2 is net-negative for the primary segments. It is not merely "near S1".

- **The manipulation was valid and isolated:**
  - the coreset measurably reduces within-sequence redundancy (pooled nearest-neighbour cosine .919 → .843; pairwise .662 → .629; 33/34 sequences);
  - sequence draws are identical draw-for-draw;
  - the budget (2,266,800 draws, 9,000 steps), initialization, architecture, target and evaluation are identical;
  - the S1 baseline replay reproduces Worklog 71 exactly.
- **Held-out error did not improve; it got worse:**
  - validation .1970 → .2076;
  - test forearm \|f\| +0.022 (46% of rows improved), chain +0.003;
  - forearm p95 .672 → .765.
- **The train/validation gap remains:** S2's lower gap ratio (1.60 vs 3.30 at selection) comes from a worse training fit, not better held-out error.
- **Not CASE A:** none of the A requirements beyond the redundancy reduction hold. There is no validation/test gain for forearm/chain, gains are not distributed (the forearm regresses in 24/37 test sequences), and p95 regresses materially.
- **Not CASE C:** the regressing regime is the *out-of-plane* one, not clean or ordinary poses. The only improvements are:
  - a small near-plane chain gain (−0.010);
  - the secondary upper arm on test (−0.011, about a third owned by one sequence; flat on validation).

  These do not form a trade-off worth preserving. The clean subset is unchanged rather than traded off.

**Interpretation (diagnostic, not a new claim).** The frames that visual de-duplication removed were not wasted supervision.
- Discarding about 60% of the near-duplicate frames removes useful target signal, even with an equal draw budget and a coreset richer in out-of-plane targets. Neighbouring frames differ slightly in pose and depth, and that local density is what the head uses to interpolate depth.
- S2's best validation comes at epoch 9 and degrades from there. It reaches the overfit regime faster on its 4,318 unique frames.
- This is the opposite of the "redundant supervision wastes budget" hypothesis. Within 3DPW, more distinct paired frames help and fewer hurt.

**Answer to the completion question.** No. After sequence balancing, removing within-sequence visual redundancy does not improve Qwen visual-depth generalization; it degrades it. Combined with Worklog 71:
- sequence re-weighting changed nothing;
- within-sequence de-duplication makes things worse.

Neither manipulation of the current supervision closes the generalization ceiling. The remaining limit is the amount and diversity of the paired RGB↔3D supervision itself.

**Per the decision logic:**
- current-data sampling/readout work is closed;
- the visual-coreset sampling contract is not preserved;
- the next justified architecture step is **substantially more diverse paired RGB↔3D supervision**;
- no external dataset was downloaded or integrated in this batch, and nothing was tuned (one coreset method, K from composition, no threshold).

## Outputs and tests

**Outputs:** LabServer63 `~/animcv-output/framepose_visual_coreset_diversity/`:
- `report.json` (SHA-256 `afda0d1c9583ffb8653c22e3c811f5acefd0fdd1e51d8495f9d0cf54961249da`)
- `S2_VISUAL_CORESET_BALANCED.pt`
- `rows_heldout.json` (the Worklog 71 rows plus S2 predictions)

These are machine-readable and stay on the server. No renders were made, because this batch changes only training sampling.

**Tests:** `tests/test_framepose_visual_coreset.py` has 7 tests covering:
- deterministic arm-feature construction, global token excluded, L2 normalization
- K = minimum eligible sequence size
- centroid-nearest initialization, farthest-point selection (one frame per cluster, order-independent)
- smallest-position tie-break
- measured redundancy reduction
- S1/S2 identical sequence draws and draw counts, with S2 frames always inside the coreset
- no target/GT/error/randomness terms in the coreset module or the script's coreset section; presence of the cache/population/draw/replay/coreset/budget identity checks; W71 `train` imported unchanged; no production import

The cache identity, Worklog 69 head identity, eligible-population identity, unchanged validation/test rows (the Worklog 71 held-out rows are reused as-is) and budget are enforced at runtime as listed under Fixed contract.

**Training determinism** is inherited from the Worklog 71 `train()` function, whose exact S0 = Worklog 69 reproduction was established in Worklog 71. This batch did not run S2 a second time.

**Results:**
- 33 passed in the training image, together with the unchanged Worklog 68–71 tests.
- Locally: 6 passed, 1 skipped (torch absent).

No production file changed, so the full regression was not run.
