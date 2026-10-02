# Worklog — Depth Anything V2 as FramePose Arm-Depth Evidence (2026-10-02)

## Starting state and files

- **Starting HEAD:** fetched `origin/arch/single_frame_first` and confirmed local = remote = `aa8fff6790f450d1f46a34365f47a1148aae204f`, with a clean worktree. This is the handoff HEAD.
- **Not implemented:** the earlier geometry-only "Worklog 65 ambiguity" proposal.
- **Commits:**
  - `2b5e44d`: evidence provider and cache builder
  - `5885bf2`: D0/D1 A/B module, script and tests
  - the commit adding this worklog
- **Files added:**
  - `src/framepose/depth_evidence.py`
  - `src/framepose/depth_evidence_probe.py`
  - `scripts/build_framepose_depth_evidence.py`
  - `scripts/run_framepose_depth_evidence_ab.py`
  - `tests/test_framepose_depth_evidence.py`
  - this worklog
- **Unchanged:** every existing file, including frozen H0, the Worklog 64 probe (`framepose/arm_depth_probe.py` is imported, never edited), `FramePoseEstimator`, the training/loss code, the Geometry Observation Layer, the legacy `pose/depth_estimator.py` and `pose/depth_sampling.py`, AnimationSemantics, Root Orientation, FK and IK.
- **Server setup:** on LabServer63 the pinned submodule `third_party/Depth-Anything-V2` was initialised at `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` (the commit already recorded in `.gitmodules` and the repo index). This changed no tracked content.

## Depth Anything model provenance

| item | value |
|---|---|
| backend | Depth Anything V2 **ViT-S (`vits`)**, the single model; no size sweep |
| checkpoint | `~/animcv-output/models/depth_anything_v2_vits/depth_anything_v2_vits.pth` (LabServer63) |
| checkpoint SHA-256 | `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378` (equals the official HF LFS oid; the builder refuses any other file) |
| upstream | `huggingface.co/depth-anything/Depth-Anything-V2-Small`, revision `03876f8651c73a60fe4c2c48294e09fcb6838fcf`, Apache-2.0 |
| code | `third_party/Depth-Anything-V2` @ `a561b849…` |
| inference | unchanged legacy adapter `DepthEstimator(vits, device=auto, input_size=518)`; torch 2.1.2+cu118, CUDA (RTX 3080 Ti); about 0.094 s per frame; 1,743 s in total |

The checkpoint was not in any project or server cache: the HF cache held only Qwen2-VL and two timm ViTs. It was downloaded from the official repository at a pinned revision and its SHA verified, so no substitute model was involved.

## Image identity and evidence cache

- **Image identity:** every one of the 18,645 unique `3dpw_images` references behind the 21,817 bank samples was read once as bytes and hashed with SHA-256. It was decoded from those same bytes (BGR, as upstream `run.py` does) and inferred once. Each sample of that image was then sampled at its own benchmark 2D joints.
- **Checks:** image size equals `sample.image_size` for every sample. The sample-ID list hash (`d28c9506…`), the bank digest (`75519e63…`) and the input-2D digest are recorded.
- **Cache identity:** the cache digest is `ff2c347f…`. It covers provider, encoder, input size, checkpoint SHA, code revision, sampling, normalization, orientation, bank and input digest.
- **Per-sample record:** each sample stores its image SHA-256, and `FramePoseDepthEvidence.digest` covers the image SHA, checkpoint SHA, policies, raw values and availability. Nothing is keyed by filename alone.
- **Location:** `~/animcv-output/framepose_depth_evidence_vits/` (`evidence.npz` SHA-256 `f6d37876…`, `manifest.json`).
- **Coverage:**
  - frames with available evidence: 21,817 of 21,817
  - sampled joints: 353,069 of 353,457 input-valid joints; the other 388 fell out of frame and are unavailable
- **Separation from legacy:** the evidence lives in a separate FramePose contract (`FramePoseDepthEvidence`) and never touches `PoseLandmark.z`. It is labelled as relative monocular depth, which is not metres, not canonical Y and not GT.

## Evidence contracts (fixed before results)

- **Sampling:** the nearest depth-map pixel at the benchmark 2D joint, `int(round(x·W)), int(round(y·H))`, exactly as the historical sampler does. An invalid or out-of-frame joint is unavailable, and no neighbouring pixels are searched.
- **Normalization:** per frame over the available joints, `d_rel = (d − mean) / std`. If std < 1e-6 or fewer than 2 joints are available, the whole frame is unavailable. There is no dataset, sequence or GT calibration.
- **Orientation:** in Depth Anything a larger value means closer, while canonical +Y means farther, so `forward_depth_evidence = −d_rel`. This is pinned analytically in a test, not chosen from GT.

## 1. Raw evidence quality (before any training)

Pairwise evidence is `Δ_DA = evidence[distal] − evidence[proximal]`. It is compared with the GT forward fraction f on eligible rows (as in Worklog 64: GT arm valid, input arm valid, non-degenerate 2D).

| scope | segment | sign agreement | stable-sign agreement (\|f_gt\| > sin 10°) | Pearson / Spearman vs f_gt | H0 stable-sign / Pearson (context) |
|---|---|---:|---:|---|---|
| test | upper | 0.597 | 0.625 | 0.23 / 0.29 | 0.78 / 0.46 |
| test | forearm | 0.672 | 0.724 | 0.31 / 0.40 | 0.76 / 0.64 |
| test | chain | 0.657 | 0.741 | 0.44 / 0.48 | 0.75 / 0.48 |
| validation, clean | upper / forearm / chain | 0.76 / 0.62 / 0.73 | 0.77 / 0.71 / 0.76 | 0.48 / 0.39 / 0.59 | 0.92 / 0.83 / 0.97 stable; Pearson 0.85 / 0.75 / 0.87 |
| validation, crosscountry | upper / forearm / chain | 0.76 / 0.50 / 0.62 | 0.76 / 0.57 / 0.66 | 0.12 / 0.09 / 0.26 | 0.82 / 0.72 / 0.73 |

Correlation against GT ΔY in metres is nearly identical.

**By \|GT f\| (test):**

| segment | sign agreement, \|f\| < 0.17 (near image plane) | sign agreement, \|f\| ≥ 0.5 | Pearson in the \|f\| ≥ 0.5 bin |
|---|---:|---:|---:|
| chain | 0.50 | 0.81 (H0 0.78) | 0.58 |
| forearm | 0.55 | 0.76 (H0 0.85) | — |
| upper | 0.55 | 0.61 | — |

The raw Depth Anything evidence therefore carries **real but moderate** continuous arm-depth signal. It is weaker than H0's own estimate overall, close to chance for near-image-plane segments, and strongest for strongly out-of-plane chains. On the clean subset it is much weaker than H0. No mapping coefficient was fitted from these numbers.

## 2. D0/D1 model identity and training

- **Graph:** one graph for both candidates. It is the unchanged `FramePoseEstimator` (O_BILATERAL ModelConfig: width 256, 8 heads, depth 2, 7 sign fields) with its 4-channel geometry projection replaced by a 5-channel `Linear(5, 256)`. Everything after the projection runs the production estimator's own code.
- **Parameters:** D0 = D1 = **1,657,859** (H0 and the Worklog 64 probe: 1,657,603; the difference is the extra 256 input weights).
- **Initialization:** identical, from the same seed.
- **D0_ZERO_DEPTH:** channels 1–4 are the H0 geometry tensor and channel 5 is exactly 0.
- **D1_DEPTH_ANYTHING:** channels 1–4 are identical and channel 5 is the forward evidence, 0 where unavailable. Joint-valid semantics are unchanged.
- **Input check:** the script verifies that the first four channels are identical and that D0's channel 5 is all zero.
- **Inputs shared with H0:** both candidates receive H0's geometry tensor and the identical O_BILATERAL oracle sign array. The inputs reproduce frozen H0 (≤ 1e-4). This remains an optimistic control.
- **Training contract (identical to Worklog 64):**
  - target f = unit(v).Y for upper, forearm and chain from bank `target_3d`
  - masked MAE on tanh-bounded f, read from the same head tokens
  - AdamW, learning rate 3e-4 cosine to 1e-5, weight decay 1e-4
  - 200 epochs, batch 256, seed 1337, AMP
  - validation every 10 epochs; selection on validation mean \|f\| error; test never used
- **Selection:**

| candidate | selected epoch | validation mean \|f\| error |
|---|---:|---:|
| D0 | 99 | 0.2342 |
| D1 | 149 | 0.2080 |

D1's validation curve sits below D0's at every evaluation from epoch 9 onward.

## 3. Held-out test results (6,003 rows; evidence available on every row)

Each cell is mean / p50 / p95. "Raw plane" means the observed 2D owns the image plane.

| segment | metric | H0 | Worklog 64 | D0 | **D1** |
|---|---|---|---|---|---|
| upper | \|f error\| | .237/.168/.661 | .227/.162/.629 | .225/.162/.615 | **.209/.159/.560** |
| | \|elevation\| (°) | 14.5/10.3/39.5 | 13.8/9.8/38.3 | 13.7/9.8/37.2 | **12.7/9.7/34.5** |
| | full, raw plane (°) | 15.7/11.4/40.1 | 15.0/10.9/39.0 | 14.9/10.9/37.9 | **13.9/10.8/35.5** |
| forearm | \|f error\| | .283/.203/.848 | .302/.221/.873 | .300/.219/.818 | **.265/.204/.781** |
| | \|elevation\| (°) | 18.7/13.6/56.7 | 20.0/14.2/57.1 | 19.9/14.2/55.7 | **17.6/13.0/51.2** |
| | full, raw plane (°) | 20.4/14.8/59.7 | 21.8/15.8/59.3 | 21.6/15.7/58.9 | **19.4/14.7/53.2** |
| shoulder→wrist | \|f error\| | .271/.192/.810 | .276/.197/.772 | .267/.196/.743 | **.224/.173/.612** |
| | \|elevation\| (°) | 16.9/11.7/50.9 | 17.2/11.9/48.3 | 16.6/11.8/46.9 | **14.1/10.5/38.6** |
| | full, raw plane (°) | 17.9/12.6/52.1 | 18.2/13.0/49.1 | 17.6/12.8/48.0 | **15.2/11.6/39.8** |

- With oracle canonical XZ (evaluation-only depth isolation), the full-angle errors equal the elevation errors shown.
- H0's own XYZ full angle is 16.2° / 21.0° / 18.2°.

**Paired D1 − D0 on test** (mean / p50 / p95; share of rows where D1 is better):

| segment | \|f error\| | \|elevation\| (°) | full, raw plane (°) | D1 better on |
|---|---|---|---|---:|
| upper | −.016 / −.009 / +.267 | −0.94 / −0.54 / +16.2 | −0.94 / −0.38 / +15.2 | 53% |
| forearm | −.034 / −.020 / +.329 | −2.38 / −1.28 / +20.4 | −2.28 / −0.99 / +18.8 | 55% |
| chain | −.043 / −.020 / +.276 | −2.52 / −1.18 / +17.1 | −2.41 / −0.94 / +16.1 | 55% |

**Size relative to run noise:** D0 and the Worklog 64 probe are both geometry-only and differ only in input width. On test they land within 0.01 \|f\| of each other (.225/.300/.267 vs .227/.302/.276). The D1 gains of 0.016 to 0.043 exceed that spread, most clearly on forearm and chain.

**Paired D1 − D0 \|f\| on test, by \|GT f\|:**

| \|GT f\| bin | upper | forearm | chain |
|---|---:|---:|---:|
| < 0.17 (near image plane) | −0.020 | +0.003 | 0.000 |
| 0.17–0.5 | +0.004 | −0.011 | −0.037 |
| ≥ 0.5 | −0.076 | −0.087 | −0.115 |

The gain is concentrated in **strongly out-of-plane arm segments**, and near image-plane rows are unchanged.

## 4. Validation, clean subset, crosscountry, per-sequence

**D1 − D0 by scope** (mean full angle on the raw plane, and share of rows D1 improves):

| scope | n | upper | forearm | chain |
|---|---:|---|---|---|
| validation, all | 3,056 | −1.28° (55%) | −2.06° (54%) | −1.18° (55%) |
| **validation, clean dancing + hug** | 425 | −0.58° (47%), median +0.23° | −0.01° (47%), median +0.24° | **+0.25° (52%)** |
| validation, crosscountry (camera caveat) | 133 | −1.34° (62%) | −3.41° (65%) | −1.67° (57%) |

- **Clean subset absolute values:** H0 is still best on every segment. Chain \|f\| is H0 0.152, Worklog 64 0.203, D0 0.230 and D1 0.233. Elevation is H0 9.5°, D0 14.0° and D1 14.3°.
- **Per sequence, val + test (53 actor sequences):** D1 has lower mean \|f\| error than D0 on 35 (upper), 34 (forearm) and 38 (chain).
  - Largest chain gains: `downtown_bar_00` −0.19, `downtown_sitOnStairs_00` −0.16/−0.14, `downtown_walking_00` −0.16, `downtown_runForBus_01` −0.15.
  - Largest losses: `downtown_bus_00` +0.06/+0.04, `courtyard_dancing_00` +0.06, `downtown_crossStreets_00` +0.04, `downtown_walking_00` (another actor) +0.04.
- **Not a single-sequence effect:** the result is not driven by one sequence, and the 53-sequence spread favours D1. The gains come mostly from the downtown test sequences, which make up most of the test split.

## 5. Owner cases

Each cell lists f as **GT / H0 / W64 / D0 / D1**, then the full angle on the raw plane as **D0 → D1**. The Depth Anything evidence for shoulder, elbow and wrist is given with each case.

**known_good** (dancing #424; DA .53 / .89 / 1.84)

| segment | f (GT / H0 / W64 / D0 / D1) | full angle D0 → D1 |
|---|---|---|
| upper | .50 / .36 / .31 / .29 / .24 | 13.3° → 16.3° |
| forearm | .12 / .04 / .03 / .04 / .05 | 4.7° → 4.2° |
| chain | .32 / .21 / .18 / .17 / .13 | 9.4° → 11.5° |

**tracking_loss** (hug #312; DA −.13 / −.33 / −.55)

| segment | f (GT / H0 / W64 / D0 / D1) | full angle D0 → D1 |
|---|---|---|
| upper | −.23 / −.17 / −.06 / −.10 / −.11 | 8.1° → 7.4° |
| forearm | −.61 / −.38 / −.24 / −.22 / .03 | 25.1° → 39.8° |
| chain | −.47 / −.31 / −.07 / −.18 / −.02 | 18.2° → 27.4° |

**largest_ratio_error** (hug #397; DA −.30 / .59 / 2.95)

| segment | f (GT / H0 / W64 / D0 / D1) | full angle D0 → D1 |
|---|---|---|
| upper | −.89 / −.47 / .02 / .43 / .49 | 91.4° → 95.2° |
| forearm | .56 / −.26 / .06 / .39 / .62 | 90.8° → 81.2° |
| chain | −.24 / −.49 / −.22 / .44 / .36 | 41.8° → 37.1° |

**largest_direction_error = largest_2d_direction_error** (dancing #176; DA −.28 / .57 / 3.30)

| segment | f (GT / H0 / W64 / D0 / D1) | full angle D0 → D1 |
|---|---|---|
| upper | .64 / −.79 / .61 / .57 / .87 | 11.9° → 22.5° |
| forearm | .59 / .32 / .71 / .81 / .94 | 70.8° → 62.4° |
| chain | .87 / −.41 / .80 / .82 / .86 | 57.4° → 54.0° |

**dynamic** (crosscountry #462; DA .52 / 1.18 / 1.03)

| segment | f (GT / H0 / W64 / D0 / D1) | full angle D0 → D1 |
|---|---|---|
| upper | .72 / .34 / .49 / .26 / .46 | 32.4° → 20.7° |
| forearm | .32 / −.11 / −.06 / −.44 / .07 | **45.8° → 17.8°** |
| chain | .54 / .11 / .07 / −.10 / .28 | **39.6° → 19.1°** |

**turning_review** (crosscountry #211; DA −1.69 / −1.61 / .02)

| segment | f (GT / H0 / W64 / D0 / D1) | full angle D0 → D1 |
|---|---|---|
| upper | −.70 / −.46 / −.12 / .05 / −.10 | 71.8° → 65.3° |
| forearm | −.99 / −.66 / −.03 / −.01 / .14 | 81.3° → 89.6° |
| chain | −.91 / −.62 / −.09 / .05 / .21 | 82.6° → 90.8° |

**near_straight** (crosscountry #525; DA −.74 / .46 / .51)

| segment | f (GT / H0 / W64 / D0 / D1) | full angle D0 → D1 |
|---|---|---|
| upper | .03 / .31 / .24 / .37 / .33 | 20.4° → 18.1° |
| forearm | −.10 / .35 / .15 / .26 / .19 | 20.8° → 17.1° |
| chain | −.03 / .33 / .21 / .26 / .33 | 17.2° → 21.1° |

**strongly_bent = largest_reach = largest_depth_component** (crosscountry #173; DA .27 / 1.22 / .89)

| segment | f (GT / H0 / W64 / D0 / D1) | full angle D0 → D1 |
|---|---|---|
| upper | .82 / −.61 / −.04 / .17 / .38 | 45.2° → 32.9° |
| forearm | .81 / .90 / .46 / .44 / .83 | 34.1° → 15.5° |
| chain | 1.00 / −.14 / .60 / .59 / .41 | 52.9° → 65.1° |

**turning_invalid** (#202) and **invalid_endpoint** (#406) are not eligible: there is no valid arm input.

**Mixed outcomes:**
- Clear gains: dynamic #462 and the forearm on #173.
- Losses: tracking_loss #312, known_good #424 upper/chain, turning #211.
- No negative case was replaced.

**Tracking_loss #312 is a fusion miss.** Δ_DA there points the right way (the wrist is closer than the elbow), yet D1 still predicts a forearm f of about 0.

## Classification and failure attribution

**CASE C: Depth Anything helps some regimes only.** It does **not** meet the declared promotion criterion.

**Criteria met:**
- On held-out test (6,003 rows, never used for selection), D1 improves D0 consistently on upper arm, forearm and chain in every continuous depth metric (chain mean \|f\| −16%, elevation −2.5°, p95 −8°).
- The improvement is larger than the geometry-only run spread.
- It holds for 34–38 of 53 sequences.
- D1 also beats frozen H0 on test on all three segments.

**Criterion not met:** the improvement is **absent on the clean dancing + hug subset**. There D1 ≈ D0, both are worse than H0, and the chain is even +0.25° worse. Promoting Depth Anything requires clean-subset improvement, so it is not promoted.

**Where it works:** the gain is concentrated where the GT arm segment is strongly out of the image plane (\|f\| ≥ 0.5: −0.08 to −0.12 \|f\|). This matches the raw-evidence profile, where Depth Anything sign agreement is about 0.81 for such chains but at chance near the image plane. Crosscountry also shows large gains, but it is not used as a deciding vote.

**Attribution:**
- It is **not A** (no signal): the raw evidence correlates with GT depth (chain Pearson 0.44 test, 0.59 clean), and fusion does convert it into held-out gains.
- It is **partly B** (fusion): on the clean subset the raw signal exists (0.59 chain Pearson) but D1 ≈ D0. #312 shows the same miss.
- **Overall C:** nearest-pixel monocular relative depth is useful for strongly out-of-plane arm poses and neutral for near-image-plane poses. On the two clean courtyard sequences, where frozen H0 is already much better than any of the probes, the evidence adds nothing measurable. That subset is small (425 rows), with few large-\|f\| rows.

**Answer to the completion question:** Depth Anything V2 vits contains genuine RGB-derived continuous arm-depth evidence. In an otherwise identical control it materially reduces held-out arm-depth error, but only in a regime-limited way. It does **not yet** clear the bar of being a justified first-class FramePose evidence provider, because the clean-subset criterion fails.

Per the direction:
- no tuning of patch, normalization, model size, loss or fusion was done;
- production FramePose, H0, AnimationSemantics, FK and IK were not touched;
- VLM, temporal and production work were not started.

The next decision is whether the regime-limited gain justifies a better fusion or evidence design, or whether another evidence family should be tried. This batch does not make that decision.

## Outputs and tests

**Outputs** (LabServer63):
- `~/animcv-output/framepose_depth_evidence_ab/`:
  - `report.json` (SHA-256 `e67a637fa0d8eca9cf7c09d98abeddc082103d06b43256197583045b632f439b`)
  - `D0_ZERO_DEPTH.pt`, `D1_DEPTH_ANYTHING.pt`
  - per-row validation and test JSON
- the evidence cache listed above

These are machine-readable outputs and stay on the server. No renders were produced.

**Tests:** `tests/test_framepose_depth_evidence.py` has 10 tests covering:
- exact nearest-pixel sampling, and invalid/out-of-frame joints without search
- per-frame z-score normalization with scale/offset invariance
- the sign convention (larger Depth Anything = closer = smaller forward evidence)
- degenerate-map and too-few-joint handling
- evidence identity (image bytes, not filename) and cache key
- checkpoint provenance pinned in the builder
- D0 channel 5 exactly zero, and D1 channel 5 being only the evidence
- same target, objective, split and signs, with no target at inference
- identical D0/D1 graph, parameter count and initialization, deterministic evaluation, and channel 5 reaching the output
- legacy depth modules and the Worklog 64 probe untouched

Results:
- With the unchanged Worklog 64 tests: 20/20 passed in the LabServer63 training image.
- Locally: 18 passed, 2 skipped (torch absent).

No shared production module changed, so the full regression was not run.
