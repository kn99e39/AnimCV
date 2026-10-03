# Worklog — Skeleton-Geometry-Biased Attention over Frozen Qwen3-VL Tokens (2026-10-04)

## Starting state and files

- **Start:** fetched `origin/arch/single_frame_first` and confirmed local = remote = `d860c734eec3a7a6b053af844141fa1f7d5a3c36`, with a clean worktree.
- **Commits:**
  - `7b3b02e`: module, script, tests
  - `ae0ece9`: attention review renderer
  - the commit adding this worklog
- **Files added:**
  - `src/framepose/skeleton_attention_depth.py`
  - `scripts/run_skeleton_attention_depth.py`
  - `scripts/render_skeleton_attention_review.py`
  - `tests/test_framepose_skeleton_attention_depth.py`
  - this worklog
- **Unchanged:** every existing file and checkpoint, including Qwen3-VL, the Worklog 68 cache and sensor, Worklog 69 C0/C1, frozen H0, Worklog 65 D1, historical F0/F1/F2, FramePoseEstimator, AnimationSemantics, FK and IK.

**First launch.** It failed with CUDA out-of-memory, because an unrelated project's container (`vismodeling-planner`) was occupying 11 GB of the shared RTX 3080 Ti. That container was not touched. The identical command was rerun once the GPU was free. All results come from that run.

## Full-grid cache identity

- **Grid:** the unchanged Worklog 68 `token_grid_fp16.npy`, (21,817, 196, 4096) float16.
  - SHA-256, computed and recorded in this run: `d83f9b1d00d1a88cf264f4beebfa55d5861ed8cd607dc3a79ac09476383bc230`.
  - Read via memmap, with background-prefetched batches.
- **Checks on load:** the Worklog 68 manifest's bank digest, sample-ID order hash and Qwen3-VL fingerprint (`a0d72ded…`), and `readout.npz` (`f5d0eb9c…`). Nothing was regenerated, and no GT enters feature extraction or the bias.

## Attention architecture and geometry bias

**Inputs and target are exactly Worklog 69's:**
- the 13 runtime pair-geometry features;
- the endpoint A, endpoint B and global readouts;
- the Worklog 64 target `f = unit(segment).Y` with its GT validity masks and FrameBank splits.

No H0, D1, Depth Anything, oracle sign, GT geometry or temporal input is used.

**Graph:** one fixed architecture, with separate per-segment weights for upper arm, forearm and chain.
- shared `visual = Linear(4096 → 64)`, applied to all 196 tokens (keys = values) and to the 4 readouts;
- per segment, query `q = Linear(13 → 64)(pair geometry)`;
- `score_i = q·k_i / √64 + bias_i`, then `context = Σ softmax(score)_i · v_i`, with a single head;
- head: `Linear(13 + 4·64 → 64) → GELU → Linear(64 → 1) → tanh`.

**Geometry bias:** deterministic and parameter-free.
- The two detector endpoints of the pair are mapped into the pinned Worklog 68 token-grid coordinates (token centres at integers, row-major 14×14). The mapping is identical to the Worklog 68 `token_coordinates`, and a test checks it.
- **A1:** `bias_i = −d_i`, where `d_i` is the Euclidean distance in token-grid units from token centre `i` to the closed 2D segment between the endpoints. There is no coefficient, threshold, radius, mask, Gaussian width or sweep.
- **A0:** `bias_i = 0`.
- **Invalid endpoint:** the pair is masked exactly as in Worklog 69, and its bias is 0 rather than computed from a substitute point.

**Interpretation of the bias.** It is built from the same 2D geometry the model already receives. A1 > A0 would mean that encoding known geometry as a spatial attention prior is useful. It would not mean that new geometric information was added.

**Not EGformer.** There are no equirectangular or spherical coordinates, no ERPE, no panorama windows and no cyclicity. The only principle carried over is "known geometry → explicit attention bias", and here the known geometry is the observed 2D skeleton.

**Parameter identity.** A0 = A1 = **316,931** parameters, with identical initialization (seed 1337). The script verifies that A0's bias is exactly zero and that A1's equals −distance.

**Training (the Worklog 69 contract, unchanged):**
- AdamW, learning rate 3e-4 per-step cosine to 1e-5, weight decay 1e-4
- batch 256, 200 epochs, seed 1337, fp32
- masked MAE
- selection on validation mean \|f\| error every 10 epochs; test never used

## Training curves (overfit accounting)

| model | selected epoch | train MAE at selection | validation MAE at selection | final train | final validation |
|---|---:|---:|---:|---:|---:|
| Worklog 69 C1 (reference) | 69 | 0.056 | 0.197 | 0.033 | 0.204 |
| A0_UNBIASED_ATTENTION | 69 | 0.056 | 0.192 | 0.029 | 0.198 |
| A1_SKELETON_BIASED_ATTENTION | 59 | 0.054 | **0.189** | 0.027 | 0.192 |

- The train/validation gap is essentially **unchanged** from Worklog 69, at about 3.5× at selection and about 7× at the end.
- Neither the full-grid readout nor the skeleton prior reduces overfitting. They only shift validation error by about 0.005–0.008.

## A0 vs A1, and vs Worklog 69 C1 (held-out test, 6,003 rows; never used for selection)

Each cell is mean / p50 / p95.

| segment | metric | C1 (Worklog 69) | A0 | **A1** | A1, shuffled vision | H0 | D1 |
|---|---|---|---|---|---|---|---|
| forearm | \|f\| | .243/.185/.666 | **.241**/.178/.678 | .252/.181/.712 | .473 | .283 | .265 |
| | elevation (°) | 16.2/12.3/44.5 | **16.1**/12.2/43.5 | 16.7/12.2/46.9 | 30.5 | 18.7 | 17.6 |
| | full (°) | 18.1/13.7/47.3 | **17.9**/13.7/46.4 | 18.5/13.8/49.8 | 31.9 | 20.4 | 19.4 |
| chain | \|f\| | .251/.192/.707 | **.237**/.179/.683 | .240/.177/.701 | .392 | .271 | **.224** |
| | elevation (°) | 15.8/11.8/44.6 | **15.0**/11.0/43.4 | 15.1/11.0/44.3 | 24.1 | 16.9 | 14.1 |
| | full (°) | 16.8/12.7/46.6 | **16.0**/12.0/44.7 | 16.2/11.9/46.0 | 24.9 | 17.9 | 15.2 |
| upper (classified separately) | \|f\| | .272/.213/.740 | .253/.197/.700 | **.234/.178/.653** | .367 | .237 | .209 |
| | elevation (°) | 16.7/13.0/45.2 | 15.5/12.0/43.6 | **14.4/10.7/40.8** | 22.4 | 14.5 | 12.7 |
| | full (°) | 17.7/13.9/46.2 | 16.6/12.8/44.8 | **15.5/11.7/42.0** | 23.3 | 15.7 | 13.9 |

**Paired comparisons on test** (\|f\| mean; share of rows improved):

| comparison | forearm | chain | upper |
|---|---|---|---|
| A1 − A0 | **+0.010** (48%) | +0.003 (50%) | **−0.018** (57%) |
| A1 − C1 | +0.008 (47%) | −0.011 (54%) | **−0.038** (60%) |
| A0 − C1 | −0.002 (50%) | **−0.014** (54%) | −0.020 (55%) |

**Validation** (3,056 rows; used for selection), C1 / A0 / A1:

| segment | C1 / A0 / A1 |
|---|---|
| forearm | .223 / .217 / .219 |
| chain | .180 / .178 / **.170** |
| upper | .180 / .175 / .173 |

On validation every pair difference is ≤ 0.010.

## Sign vs magnitude (test, stable rows)

| segment | stable sign accuracy (C1 / A0 / A1) | magnitude when A0 and A1 are both sign-correct (A0 → A1; rows) | magnitude when A1 and C1 are both sign-correct (C1 → A1; rows) |
|---|---|---|---|
| forearm | .837 / .819 / .814 | .187 → .196 (3,216) | .205 → .200 (3,219) |
| chain | .786 / .810 / .808 | .194 → .190 (2,929) | .206 → **.186** (2,838) |
| upper | .753 / .781 / .791 | .171 → **.161** (2,711) | .174 → **.163** (2,623) |

**Primary segments:**
- **Forearm:** the bias improves neither branch nor magnitude (A1 vs A0).
- **Chain:** the full-grid context (A0 and A1 alike) improves both sign (+2.4 points) and magnitude over C1. The bias itself adds nothing.

**Upper arm:** A1 improves both sign and magnitude over A0 and C1.

## Shuffled-vision control

A1 was given the deterministic Worklog 68 different-sequence donor grid and readouts. The script verifies donor identity against Worklog 68. The original geometry and original bias were kept, and there was no retraining.

| test segment | A1 real | A1 shuffled |
|---|---:|---:|
| forearm | .252 | .473 |
| chain | .240 | .392 |
| upper | .234 | .367 |

On the clean subset, shuffled vision is also far worse (forearm .630). A1's output depends on the frame's own visual content.

## GT-depth bins (test \|f\|; C1 / A0 / A1)

| segment | near plane (< sin 10°) | intermediate | strongly out of plane (≥ 0.5) |
|---|---|---|---|
| forearm | .166 / .177 / .183 | .241 / .254 / .258 | .309 / **.283** / .303 |
| chain | .179 / .175 / .180 | .239 / .228 / .228 | .377 / **.343** / .349 |
| upper | .250 / .233 / **.203** | .237 / .221 / **.204** | .451 / .416 / .424 |

- **Out-of-plane gain:** both attention readouts preserve, and slightly extend, Worklog 69's out-of-plane gain on the forearm and chain, with A0 best.
- **Near-plane over-prediction:**
  - forearm: A1 does **not** reduce it; it is slightly worse than C1;
  - chain: A1 leaves it unchanged;
  - upper: A1 **reduces** it (.250 → .203), which is the main source of its upper-arm gain.

## Clean / crosscountry / other (\|f\|: C1 / A0 / A1, with H0 for reference)

| scope | forearm | chain | upper |
|---|---|---|---|
| clean dancing + hug (425) | .195 / .187 / **.183** (H0 .200) | .183 / **.166** / .175 (H0 .152) | .185 / **.159** / .173 (H0 .151) |
| crosscountry (133) | **.224** / .227 / .237 | .221 / .225 / **.207** | .214 / .229 / .224 |
| all other held-out (8,501) | .239 / **.236** / .244 | .229 / .220 / **.218** | .245 / .230 / **.216** |

**Clean subset.** No new regression relative to C1 for any model. A1 is worse than A0 on clean for the chain and the upper arm.

## Per-sequence ownership (53 held-out actor sequences)

| segment | A1 vs A0: improved / net gain / largest share | A1 vs C1: improved / net gain / largest share |
|---|---|---|
| forearm | 23 / **net negative** (−67.0) / — | 27 / **net negative** (−39.5) / — |
| chain | 28 / +9.8 (net about 0; largest single-sequence share 161% of the tiny net, 14% of the positive gains) | 33 / +97.9 / 16.5% |
| upper | 33 / +115.4 / **14.3%** | 39 / +249.3 / **12.9%** |

The upper-arm A1 gain is, for the first time, **broadly distributed** on test, not sequence-owned. However, it does not reproduce on validation (A1 vs C1 −0.007, A1 vs A0 −0.002), and A1 is worse than A0 on the clean upper arm.

## Owner cases (f: GT / C1 / A0 / A1 / A1-shuffled / H0 / D1; full angle C1 / A0 / A1)

Attention review panels are copied locally to `~/animcv-output/70_framepose_skeleton_attention_depth/`. Each has 3 pairs × [crop with the 2D segment, A0 attention, A1 attention], with an f/error table. They are qualitative only.

Across cases, A0's attention is diffuse and often lands on background and scenery. A1's concentrates on the arm segment and adjacent torso.

**dynamic** (crosscountry #462)

| segment | f (GT / C1 / A0 / A1 / A1-shuffled / H0 / D1) | full angle C1 / A0 / A1 |
|---|---|---|
| upper | .72 / .74 / .67 / .68 / .31 / .34 / .46 | 7.9° / 8.8° / 8.5° |
| forearm | .32 / .13 / −.03 / .10 / .00 / −.11 / .07 | 14.8° / 22.5° / 16.3° |
| chain | .54 / .37 / .38 / .42 / .22 / .11 / .28 | 14.4° / 13.7° / **11.7°** |

**tracking_loss** (hug #312): forearm **−.61** / −.33 / −.36 / **−.51**, full angle 18.8° / 17.2° / **8.8°** (A1 is the closest of all, better than H0 at 16.2°). Chain 29.1° / 25.0° / 19.0°.

**strongly_bent** (crosscountry #173): chain 1.00 / .19 / .28 / **.46**, full angle 77.8° / 72.6° / **61.9°** (A1 now beats D1's 65.0°).

**largest_ratio_error** (hug #397): upper −.89 / −.66 / −.60 / **−.73**, full angle 27.8° / 31.9° / **23.0°**. The forearm sign error persists in all models (about 128°).

**largest_direction_error** (dancing #176): A1 is worse than A0 and C1 on the forearm (104.5°) and the chain (99.6°).

**turning** (crosscountry #211): all visual models keep the wrong sign (82°–120°).

**known_good** (dancing #424) and **near_straight** (crosscountry #525): about equal across models. The near_straight upper arm is worse than C1 (21.5° vs 15.3°).

**Excluded:** #202 and #406 are not eligible. No negative case was removed.

## Architecture classification

**Primary segments (forearm, chain): the skeleton bias is not justified.** This is CASE B for the chain and CASE C for the forearm.

| CASE A requirement | forearm | chain |
|---|---|---|
| A1 materially beats A0 | **no** (test +0.010 worse; net negative over sequences) | **no** (test +0.003; validation −0.008; net about 0 over sequences) |
| A1 competitive with C1 | slightly worse | better (−0.011) |
| real beats shuffled | yes | yes |
| magnitude gain vs A0 when signs correct | no | marginal |
| generalization not worse | gap unchanged | gap unchanged |
| clean preserved | yes | yes vs C1; worse than A0 |

- **Chain, CASE B:** full visual context helps, but the fixed skeleton-distance bias does not. A0 improves Worklog 69 C1 on test (−0.014 \|f\|, −0.8° elevation, p95 .707 → .683, sign +2.4 points, both-correct magnitude better), and A1 ≈ A0.
- **Forearm, CASE C:** the simple readout is already the useful limit. Neither attention model improves C1 materially (A0 −0.002, A1 +0.008).
- **Upper arm (classified separately; not a primary-decision criterion):**
  - On test, A1 gives a broadly distributed improvement over both A0 and C1: −0.018 and −0.038 \|f\|, 33 and 39 of 53 sequences, largest share 13–14%. Mostly it reduces near-plane over-prediction, and it improves both sign and magnitude.
  - It does not reproduce on validation and is worse than A0 on the clean subset.
  - It is recorded as a candidate signal for a future upper-arm study, not as an established result.
- **Data diversity:** neither readout narrows the about 3.5× train/validation gap, so data diversity (CASE D aspect) remains the dominant limit. Readout structure changes held-out error only at the ±0.01–0.02 level.

**Answer to the completion question:** No. Using observed skeleton geometry as a fixed visual-attention prior does not extract continuous arm depth more robustly than unbiased attention for the distal arm, and it does not beat the simple Worklog 69 readout for the forearm.
- **Forearm:** keep the Worklog 69 endpoint/global readout.
- **Chain:** broader full-grid context (unbiased attention) gives a modest gain worth carrying as an option.
- **Upper arm:** the bias shows a promising but unconfirmed effect.
- **Primary limit:** 3DPW visual supervision diversity, not readout structure.

No production integration was done, and the bias was not tuned.

## Outputs and tests

**Outputs:** LabServer63 `~/animcv-output/framepose_skeleton_attention_depth/`:
- `report.json` (SHA-256 `eca641342418ae4648778664dd43dde19e03c46a15176c2f9cdd79f3d83f0c04`)
- `A0_UNBIASED_ATTENTION.pt`, `A1_SKELETON_BIASED_ATTENTION.pt`
- `rows_heldout.json`
- `owner_attention.npz`
- `attention_review/`

Only the review PNGs are copied locally.

**Tests:** `tests/test_framepose_skeleton_attention_depth.py` has 7 tests covering:
- constants equal to Worklog 68, and the arm joint indices
- row-major token-grid geometry
- detector → token mapping equal to Worklog 68
- point-to-segment distance, including the degenerate segment
- the deterministic bias (A0 exactly zero, A1 exactly −distance, invalid pairs zero)
- identical graph and initialization, attention normalization, the bias changing only the attention, shuffled vision changing only the visual input, and deterministic inference
- no GT in the bias, no H0/D1/Depth Anything/VLM in training, Worklog 69 target/mask/split identity, cache and donor identity checks, and no production import

Results:
- 21 passed in the training image, together with the unchanged Worklog 68 and Worklog 69 tests.
- Locally: 18 passed, 3 skipped (torch absent).

No production file changed, so the full regression was not run.
