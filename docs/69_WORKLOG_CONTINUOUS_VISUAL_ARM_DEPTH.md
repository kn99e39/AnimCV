# Worklog — Continuous Arm Depth from Frozen Qwen3-VL Visual Tokens (2026-10-03)

## Starting state and files

- **Start:** fetched `origin/arch/single_frame_first` and confirmed local = remote = `25601be6696c3f922b6ddb43c8ebe2b7aeedc1a2`, with a clean worktree.
- **Commits:**
  - `5e3eee0`: module, script, tests
  - `94960fa`: the new module now restates the four Worklog 68 head constants instead of importing them, so that Worklog 68's unchanged no-importer test keeps passing. The values are identical and a test pins the equality. The run started from `5e3eee0`, and this change cannot affect it.
  - `6a5273b`: owner-case review renderer
  - the commit adding this worklog
- **Files added:**
  - `src/framepose/continuous_vision_depth.py`
  - `scripts/run_continuous_visual_arm_depth.py`
  - `scripts/render_continuous_vision_owner_review.py`
  - `tests/test_framepose_continuous_vision_depth.py`
  - this worklog
- **Unchanged:** every existing file, including Qwen3-VL, the Worklog 68 cache, G0/G1, Worklogs 64–67, frozen H0, historical F0/F1/F2, FramePoseEstimator, FK and IK.

## Cache identity and continuous model contract

**Cache.** The unchanged Worklog 68 cache, `readout.npz` SHA-256 `f5d0eb9c34a7eb0dbae24ace7b0ee0c6a87951dcd8c847e291e44471414ec3b5`. Its bank digest and sample-ID hash are verified on load. It derives from:
- Qwen3-VL-8B-Instruct @ `0c351dd…`, fingerprint `a0d72ded…`
- the patch-merger 14×14×4096 grid
- bilinear readout at the detector joints

**Target and splits.**
- **Target:** the Worklog 64 `f = unit(segment).Y` for upper arm, forearm and shoulder→wrist chain, from bank `target_3d`.
- **Mask:** the Worklog 64 GT validity mask.
- **Splits:** the exact FrameBank splits.

**Inputs.** Exactly the Worklog 68 inputs, and nothing else:
- the 13 pair-geometry features;
- the endpoint A, endpoint B and global frozen readouts.

There is no H0, D1, Depth Anything, oracle sign, GT geometry, temporal context or Worklog 68 class input.

**Graph** (the Worklog 68 structure; only the output changes):
- shared `Linear(4096→64)` visual projection;
- per segment, `Linear(13+3·64 → 64) → GELU → Linear(64 → 1) → tanh`.

| candidate | visual features |
|---|---|
| C0_ZERO_VISION | every feature exactly 0 |
| C1_QWEN_VISION | the cache |

- Both have the same graph, **301,955** parameters, and identical initialization from seed 1337.
- The script verifies that C0's input is all zero and that C1's equals the cache.

**Training:**
- masked MAE, unweighted
- AdamW, learning rate 3e-4 per-step cosine to 1e-5, weight decay 1e-4
- 200 epochs, batch 256, seed 1337, fp32
- selection on validation mean \|f\| error, evaluated every 10 epochs; test never used for selection

## Training curves (overfit monitoring)

| candidate | selected epoch | train MAE at selection | validation MAE at selection | final train MAE | final validation MAE |
|---|---:|---:|---:|---:|---:|
| C0_ZERO_VISION | 99 | 0.254 | 0.286 | 0.251 | 0.287 |
| C1_QWEN_VISION | 69 | **0.056** | **0.197** | 0.033 | 0.204 |

- **C1 overfits strongly:** train error falls to about 1/6 of validation error. Validation flattens from about epoch 19 (0.201) onward, so most generalizable learning happens early.
- **C0 does not overfit,** but it plateaus high.
- **Reading the gap:** this overfitting is part of the evidence. C1 memorizes 3DPW training-scene appearance far beyond what transfers.
- **Context:** C0 is a weak control. It sees only three arm joints, with no torso geometry and no oracle signs, whereas frozen H0 sees all 17 joints plus two oracle signs. C1 − C0 therefore measures what vision adds over *minimal* arm geometry. The H0, D0 and D1 references below put that gain in context.

## Held-out results

Each cell is mean / p50 / p95. "Full" is the 3D direction on the observed 2D plane.

### Test (6,003 rows; never used for selection)

| segment | metric | C0 | **C1** | C1, shuffled vision | H0 | W64 | D0 | D1 |
|---|---|---|---|---|---|---|---|---|
| upper | \|f\| | .243/.191/.698 | **.272**/.213/.740 | .389/.307/1.032 | .237/.168/.661 | .227 | .225 | .209 |
| | elevation (°) | 14.7/11.3/43.6 | 16.7/13.0/45.2 | 23.9/18.3/65.5 | 14.5/10.3/39.5 | 13.8 | 13.7 | 12.7 |
| | full (°) | 15.8/12.2/44.4 | 17.7/13.9/46.2 | 24.7 | 15.7 | 15.0 | 14.9 | 13.9 |
| forearm | \|f\| | .302/.256/.719 | **.243/.185/.666** | .473/.413/1.085 | .283/.203/.848 | .302 | .300 | .265 |
| | elevation (°) | 20.2/15.9/50.9 | **16.2/12.3/44.5** | 30.7 | 18.7/13.6/56.7 | 20.0 | 19.9 | 17.6 |
| | full (°) | 21.9/17.1/53.4 | **18.1/13.7/47.3** | 32.1 | 20.4/14.8/59.7 | 21.8 | 21.6 | 19.4 |
| chain | \|f\| | .274/.207/.727 | **.251/.192/.707** | .410/.340/1.016 | .271/.192/.810 | .276 | .267 | .224 |
| | elevation (°) | 17.1/12.3/46.5 | **15.8/11.8/44.6** | 25.4 | 16.9/11.7/50.9 | 17.2 | 16.6 | 14.1 |
| | full (°) | 18.1/12.9/47.2 | **16.8/12.7/46.6** | 26.2 | 17.9/12.6/52.1 | 18.2 | 17.6 | 15.2 |

**Paired C1 − C0 on test** (\|f\|, share of rows improved):

| segment | C1 − C0 \|f\| | improved share | elevation |
|---|---:|---:|---:|
| upper | **+0.029** | 44% | +2.0° |
| forearm | **−0.059** | 58% | −3.95° |
| chain | **−0.023** | 52% | −1.33° |

**On test, forearm C1 is the best of all candidates and references.**

| forearm metric (test) | C1 | D1 | H0 | W64 | D0 |
|---|---:|---:|---:|---:|---:|
| \|f\| | **0.243** | 0.265 | 0.283 | 0.302 | 0.300 |
| elevation | **16.2°** | 17.6° | 18.7° | — | — |
| p95 full angle | **47.3°** | 53.2° | 59.7° | — | — |

Chain C1 beats H0 (.251 vs .271; p95 46.6° vs 52.1°) but not D1 (.224).

### Validation (3,056 rows; used for epoch selection, so slightly optimistic)

| segment | C0 → C1 \|f\| | improved share | C1 vs H0 | C1 vs D1 |
|---|---|---:|---|---|
| upper | .275 → **.180** | 63% | below H0 .196 | about D1 .172 |
| forearm | .318 → **.223** | 65% | below H0 .248 | below D1 .241 |
| chain | .266 → **.180** | 62% | below H0 .201 | below D1 .208 |

On validation, C1 is the best model on every segment except upper vs D1. The upper-arm result contradicts test; see the per-sequence section.

## Sign vs magnitude attribution (stable rows, \|GT f\| > sin 10°)

| scope / segment | stable sign accuracy C0 → C1 (H0, D1) | rows where only C0 / only C1 has the right sign | magnitude error \|\|pred\|−\|gt\|\| when **both** C0 and C1 are sign-correct, C0 → C1 (rows; C1 better on) |
|---|---|---|---|
| test upper | .729 → .753 (.779, .797) | 328 / 416 | .191 → **.170** (2,394; 54%) |
| test forearm | .791 → **.837** (.757, .787) | 352 / 546 | .313 → **.215** (2,969; 64%) |
| test chain | .739 → **.786** (.751, .836) | 431 / 615 | .254 → **.203** (2,441; 57%) |
| validation forearm | .730 → .830 | 171 / 363 | .318 → **.222** (1,219; 67%) |
| validation chain | .708 → .875 | 124 / 427 | .253 → **.174** (1,158; 67%) |
| clean forearm | .799 → .852 | 15 / 29 | .355 → **.212** (196; 77%) |
| clean chain | .918 → .957 | 7 / 20 | .287 → **.200** (295; 76%) |

**Vision improves both branch and magnitude.** It fixes more signs than it breaks. On rows where both models already have the right sign, it reduces magnitude error by 16–31% for the forearm and chain, and by 11% even for the upper arm on test.

**Against H0 and D1** (rows where both are sign-correct, magnitude error):

| test segment | C1 | H0 | D1 | note |
|---|---:|---:|---:|---|
| forearm | .212 | .208 | — | about H0 |
| forearm | .215 | — | .228 | better than D1 |
| chain | .205 | .208 | .211 (vs C1 .214 on that subset) | parity |

C1's forearm advantage over H0 therefore comes mainly from **branch correction**: it is the only correct sign on 692 rows, against H0's 357. Its magnitude is at parity.

## GT-depth bins (test \|f\| error; C1 improved-row share vs C0)

| segment | near plane (< sin 10°) | intermediate | strongly out of plane (≥ 0.5) |
|---|---|---|---|
| upper | C0 .166 → C1 **.250** (34%) | .217 → .237 (47%) | .546 → **.451** (65%); D1 .308, H0 .391 |
| forearm | .162 → .166 (49%) | .279 → **.241** (58%) | .439 → **.309** (66%); D1 .342, H0 .363 |
| chain | .155 → .179 (44%) | .231 → .239 (48%) | .524 → **.377** (70%); D1 .359, H0 .470 |

- **Where vision helps:** the gain is concentrated in **strongly out-of-plane** segments, where forearm C1 is the best of all candidates.
- **Where it hurts:** near the image plane C1 is slightly worse than C0 on the upper arm and chain. It predicts out-of-plane depth where there is little.

## Shuffled-vision control

The trained C1 was given the Worklog 68 deterministic different-sequence visual donor. The script verifies donor identity against Worklog 68. Geometry is original, and there was no retraining.

| test segment | C1 real | C1 shuffled | C0 |
|---|---:|---:|---:|
| upper | .272 | .389 | — |
| forearm | .243 | .473 | — |
| chain | .251 | .410 | — |
| stable sign accuracy | — | .50–.56, about chance | .73–.79 |

Shuffled vision is far worse than both real C1 and C0. C1's output depends heavily on the frame's own image, so the gains are visual, not geometry or head priors.

## Clean / crosscountry / other

| scope | segment | C0 | C1 | H0 | D1 |
|---|---|---:|---:|---:|---:|
| clean dancing + hug (425) | upper | .278 | .185 | **.151** | .174 |
| | forearm | .282 | **.195** | .200 | .262 |
| | chain | .258 | .183 | **.152** | .233 |
| crosscountry (133) | forearm | .328 | **.224** | .324 | .283 |
| | chain | .310 | **.221** | .284 | .255 |
| all other held-out (8,501) | forearm | .309 | **.239** | .274 | .256 |
| | chain | .271 | .229 | .251 | **.218** |
| | upper | .253 | .245 | .227 | **.197** |

**Clean subset.**
- C1 strongly improves on C0 (73% / 72% / 69% of rows improve), with no clean sacrifice relative to its control.
- Against frozen H0, C1 is at parity on the forearm but worse on the chain and upper arm.
- C1 beats D1 on clean for forearm and chain (D1 degrades clean data, as in Worklog 65).

## Per-sequence ownership (53 held-out actor sequences, C1 vs C0 \|f\|)

| segment | sequences improved | largest single-sequence share of net gain |
|---|---:|---|
| forearm | **42** | **11.5%** (`outdoors_parcours_01`) |
| chain | **36** | **17.1%** (`outdoors_parcours_00`) |
| upper | 26 | **95.3%** of the net gain (`outdoors_parcours_00`; 29% of the positive gains) |

As in Worklog 68, the upper-arm result is owned by one sequence and is **not generalized**.

## Owner cases (f: GT / C0 / C1 / C1-shuffled / H0 / D1; full angle C0 / C1 / H0 / D1)

Review panels are copied locally to `~/animcv-output/69_framepose_continuous_visual_arm_depth/`. Each shows the encoder crop with review-only S/E/W dots and a table of f values and errors.

**dynamic** (crosscountry #462): C1 is the best everywhere.

| segment | f (GT / C0 / C1 / C1-shuffled / H0 / D1) | full angle C0 / C1 / H0 / D1 |
|---|---|---|
| upper | .72 / .21 / **.74** / .24 / .34 / .46 | 35.1° / **7.9°** / 27.7° / 20.7° |
| forearm | .32 / −.16 / .13 / −.13 / −.11 / .07 | 29.4° / **14.8°** / 26.9° / 17.8° |
| chain | .54 / .04 / .37 / .23 / .11 / .28 | 31.8° / **14.4°** / 27.9° / 19.1° |

**known_good** (dancing #424): upper .50 / .16 / .36 / … / .36 / .24, full angle 21.4° / 9.3° / 9.6° / 16.3°. C1 is about H0 on all three segments.

**largest_ratio_error** (hug #397): mixed.
- Upper: C1 −.66 is closest to GT −.89 (27.8° vs H0 39.6° and D1 95.2°).
- Forearm: C1 gives −.94 against GT +.56, a sign error (127.9°).
- Chain: C1 overshoots to −.93 against GT −.24 (54.9°).

**largest_direction_error** (dancing #176): C1 is worse than C0 on the upper arm (−.01 vs GT .64; 42.1°) and chain (88.5°). D1 is best.

**strongly_bent** (crosscountry #173): C1 is worse than C0 and D1 on the chain (77.8°) and forearm. On the upper arm, both C0 and C1 sit near 0 against GT .82.

**near_straight** (crosscountry #525):
- Forearm and chain: C1 is better than H0 and D1 (9.4°, 11.2°).
- Upper arm: C1 is worse than C0 (15.3° vs 3.5°).

**tracking_loss** (hug #312):
- Forearm: C1 −.33 improves on C0 (18.8° vs 29.3°) but is still behind H0 (16.2°).
- Chain: C1 is about 0, against GT −.47 (29.1°).

**turning** (crosscountry #211): C1 has the wrong sign on every segment (89.5° / 101.5° / 112.2°); H0 is best.

**Excluded:** #202 and #406 are not eligible.

No negative case was removed. The C1-shuffled values differ markedly from C1 in most cases.

## Architecture classification

**CASE A for the forearm and the shoulder→wrist chain; not established for the upper arm.**

| CASE A requirement | forearm | chain | upper |
|---|---|---|---|
| C1 materially beats C0 on held-out continuous f | **yes** (test −0.059 / −3.95°; validation −0.095) | **yes** (test −0.023 / −1.33°; validation −0.085) | **no on test** (+0.029); yes on validation |
| real beats shuffled | yes | yes | yes |
| beyond sign correction | yes | yes | — |
| magnitude improves when both signs are correct | yes (.313 → .215) | yes (.254 → .203) | yes (.191 → .170) |
| not sequence-owned | **yes** (42/53, 11.5%) | **yes** (36/53, 17%) | **no** (95% from one sequence) |
| clean preserved relative to the control | yes | yes | yes |
| p95 does not regress | improves (.719 → .666) | improves (.727 → .707) | regresses (.698 → .740) |

**Answer to the completion question:**
- **Yes, for the forearm and the chain.** Qwen3-VL's frozen visual representation encodes more than which side of the image plane the arm lies on. It carries continuous depth magnitude: with the sign already correct, magnitude error falls 16–31%.
- **Relative to current baselines:** on held-out test the forearm C1 beats frozen H0, the Worklog 64 probe and Worklog 65 D1 on \|f\|, elevation and p95. The chain C1 beats H0 but not D1.
- **The upper arm is not established,** because of the test regression and single-sequence ownership.

**Qualifications:**
- Heavy overfitting: train MAE is 0.056 against 0.197 on validation, so the signal is real but data-limited by 3DPW scene diversity.
- Validation gains exceed test gains.
- Near the image plane C1 slightly over-predicts depth.
- Some owner cases fail badly (#211, #176, #173).

**Conclusion.** Learned RGB/vision is **justified as a first-class continuous FramePose depth modality for the distal arm (forearm, shoulder→wrist chain)**. That is the basis for the next production-oriented multimodal FramePose design, scoped per segment, with the upper arm left open. No production integration was done, and nothing proceeds to yaw, temporal context or retargeting. The scale and diversity of the paired visual supervision is the obvious limiting factor for the next design.

## Outputs and tests

**Outputs:** LabServer63 `~/animcv-output/framepose_continuous_visual_arm_depth/`:
- `report.json` (SHA-256 `984f027642e5057ce270d32c553ee33bbdd71c607b9d713efdf319b4947f2027`)
- `C0_ZERO_VISION.pt`, `C1_QWEN_VISION.pt`
- `rows_heldout.json`
- `owner_review/`

Only the review PNGs are copied locally.

**Tests:** `tests/test_framepose_continuous_vision_depth.py` has 5 tests covering:
- Worklog 68 cache identity and target/split identity in the script, and pair order equal to the Worklog 64 segments, with constants equal to Worklog 68
- C0 exact zero and C1 exact cache
- identical architecture and initialization, the tanh output range, deterministic inference, and a shuffled donor replacing only the visual input
- the sign-vs-magnitude decomposition on a hand-computed example
- no H0, D1, Depth Anything or VLM in the training call, the shuffled-donor identity check, and no production import

Results:
- 22 passed in the training image, together with the unchanged Worklog 67 and Worklog 68 tests.
- Locally: 20 passed, 2 skipped (torch absent).

No production file changed, so the full regression was not run.
