# Worklog — Observability-Aware Near-Static Camera Reconstruction: Background Rotation and Motion-Regime Estimation on Synthetic A1–A4, TUM fr3 walking_static / walking_xyz and Authentic S1: Pilot PARTIALLY SUPPORTED (2026-11-13)

Batch 142 was an experimental, research-only Pilot on the local branch `Dual_Stage_Generator` (no branch operation, nothing pushed). Setup:

- local Windows workstation, CPU only;
- the AnimCV venv (OpenCV 4.11 SIFT, numpy, scipy; torch for the frozen W141 person segmentation);
- the W140 scratch venv for the frozen COLMAP controls;
- Blender 5.1 for synthetic renders;
- scratch storage `C:/Users/dna10/animcv-w142/` (never committed).

There was no network use: no new dataset, model or download. `docs/next_batch.md` and `scripts/batch_automation.py` were not touched, and the automation stays paused.

All numbers are **AGENT-RUN**. The agent viewed the figures as stills. **No human has reviewed** the figures or videos.

## 1. Official classification

**PARTIALLY SUPPORTED** (frozen rule; `results/evaluate/evaluation.json` `a38215714d05c2cc0fdd27465efb388c3bc1700c4efcb77802165933daff2f80`).

| Gate (declared before output) | Result | Status |
|---|---|---|
| **Synthetic**, primary RGB person mask | A1 stationary: max aligned false rotation **0.0002°** (< 0.5). A2 rotation: median **0.0023°** (< 1). False translation claims on A1 and A2: **0 / 777** edges each. Regimes A1 NEAR_STATIC, A2 ROTATING (correct). A4 translation segment 20–69: **100 %** TRANSLATING. 0 numeric translations. 0 keypoints on excluded pixels | **PASS** |
| **walking_static**, primary | Orientation coverage **52.7 %** (< 80 %). Median aligned orientation error **0.62°** (< 3° and below the constant-camera **2.07°**). 1 s relative error **0.25°** against a baseline of 1.23°. Spike fraction (> 2°) **0** | **FAIL (coverage only)** |
| **walking_xyz control**, primary | Sequence regime TRANSLATING; 0 % NEAR_STATIC frames; 0 numeric translations; all 300 frames routed to the frozen W141 6DoF SfM | **PASS** |

**Why the result is PARTIALLY SUPPORTED:**
- SUPPORTED needs all three gates.
- NOT_YET_SUPPORTED requires walking_static coverage < 50 % or no win over the constant-camera baseline. Neither holds.

**Answers to the completion question.**

1. **Observed.** Static-background correspondences recovered the walking_static camera **orientation** where incremental SfM failed, accurately: 0.62° median on the 158 frames the frozen rule covered.
   - The orientation follows the GT's 10° pitch drift that a constant-camera assumption misses.
   - Coverage was limited by the estimator's own conservative observability rule, not by missing evidence. A POST-HOC diagnostic (§9) that forces the rotation-only model on every edge reaches 100 % coverage at 0.17°. It is not an eligible estimator and does not change the classification.
2. **Not reliably observed.** Whether the walking_static camera *translated*.
   - The classifier reported "translation present (planar-ambiguous)" for 81 % of edges.
   - That label is only weakly related to the GT translation: 72 % at 0–2 mm against 93 % at 10–20 mm (§9).
   - On real images the rotation-only model leaves about 0.4 px of systematic residual that a homography absorbs. Likely causes are calibration, lens or rolling-shutter misfit, or real millimetre parallax; W142 cannot separate them.
   - So the regime label TRANSLATING on walking_static is **not trustworthy evidence of translation**. No translation value was fabricated: values are always null.
3. **Unobservable.** Translation magnitude, metric scale and absolute position on every dataset; translation itself at millimetre scale on walking_static.
4. **Moving-camera capability is intact.**
   - walking_xyz is classified TRANSLATING everywhere and routed to the W141 6DoF SfM, with no false stationary frame.
   - W142's own rotation is **less accurate** than SfM there: 0.70° against 0.26° under the same alignment.
5. **Authentic S1.** The camera is consistent with being fixed: background flow median 0.05 px; NO_DETECTABLE_MOTION on every decided edge; the K-free homography wins on every edge at every span.
   - The evidence is thin: 45 background keypoints per frame, from a sand strip and the horizon.
   - Intrinsics are uncalibrated.
   - Without the person mask, the person's motion produced 89 false "translation observed" edges.

## 2. Revisions, ownership, preserved baselines

| Item | Value |
|---|---|
| Branch | `Dual_Stage_Generator`; at start the working tree was clean apart from the pre-existing untracked `third_party/mmpose/`. `origin/Dual_Stage_Generator` already equalled HEAD; W142 pushed nothing |
| Starting revision | `06ebae7` (W141 Worklog) |
| Instrument commit | `864b29c` (before any W142 output) |
| Declaration commit | `61b4900`: `examples/experimental_camera_pose/w142_declaration.json`, `fb0a4d282f06b915f0547d7351517df77dab47e2a1fcf8f95e05e2caae113817` |
| Ending revision | the commit adding this Worklog |

**Changed files** (all new):
- `src/app/experimental_background_rotation.py`;
- `scripts/w142_background_rotation_estimate.py`, `scripts/w142_harness.py`, `scripts/w142_visualize.py`;
- `tests/test_w142_background_rotation.py`;
- `examples/experimental_camera_pose/w142_declaration.json`;
- this Worklog.

**Preserved and verified in the preflight:**
- W140: evaluation `3bb32caa…` and evidence manifest `28888c27…`.
- W141: evaluation `c370522e…`, estimate_result `ebce5a7d…`, evidence manifest `df19ba54…`, amendment `2d7f8c39…` and declaration `6dd621bb…`.
- All W141 instruments (amendment hashes) and all 11 W141 estimates (frozen SHA-256).
- The W141 inputs and masks for walking_static, walking_xyz and syn_dynamic are byte-identical to the W141 declaration. They are reused in place, not copied.
- No W140 or W141 file, estimate or metric was modified.
- W140/W141 SfM estimates are only **re-scored** under the W142 orientation alignment, a new metric on frozen outputs (§6–7).

## 3. Estimator and input contract

**Estimator.** `app.experimental_background_rotation` together with `scripts/w142_background_rotation_estimate.py`. All parameters were frozen in code before any output.

1. **Features.** OpenCV SIFT (≤ 4,000 features) on keep-mask pixels only. Keypoints are re-checked against the mask, with 0 found on excluded pixels in every run.
2. **Pairs.** Mutual ratio-test (0.8) matches for frame pairs at spans 1, 2, 4, 8, 16 and 32.
3. **Models**, fitted to the *same* matches:
   - rotation-only: 2-point bearing RANSAC + Kabsch, 1.5 px;
   - general homography;
   - essential matrix: OpenCV RANSAC + robust Sampson refinement, recoverPose cheirality;
   - fundamental matrix (K-free).
4. **Model selection** by Torr's GRIC with σ = 1 px. Each edge then gets a state (§4).
5. **Rotation averaging.** Robust chordal averaging (Cauchy IRLS) over rotation-valid edges gives orientation R_WC in the reference camera's optical frame. The reference is the first frame of the largest pair-graph component.
6. **Regimes.** Per-frame regime from the edges inside ±0.5 s; sequence regime by majority.
7. **Translation.** The **value is always null**, with an explicit status:
   - `UNRESOLVED_BELOW_PARALLAX_DETECTABILITY`, with a p90-residual / f bound;
   - `TRANSLATION_DIRECTION_OBSERVED_SCALE_UNOBSERVABLE`;
   - `INSUFFICIENT_EVIDENCE`.

**Input contract.** The estimator consumes only RGB frames, their original timestamps, the approved intrinsics and RGB-derived keep masks. The unchanged W140 leak guard (input directory) and the unchanged W141 mask contract (separate mask directory) are enforced, and tests check that a planted `groundtruth.txt` and a bad K are refused. The estimator never reads GT, depth, IMU, SfM output or oracle masks; the oracle masks appear only in the labelled synthetic diagnostic.

**Routing.** After estimation and before GT, frames whose window is TRANSLATING take the frozen 6DoF SfM pose where the SfM registered them. Other frames keep orientation-only state. Nothing is discarded (`routing.json` `b2c94ddf…`).

**Conventions.** W140 conventions are unchanged: optical OpenCV axes, `T_WC`, scalar-last quaternions. The new contract adds:
- orientation status: ESTIMATED / UNSUPPORTED / OTHER_COMPONENT / INSUFFICIENT_EVIDENCE;
- rotation observability;
- translation status;
- a per-edge state;
- the regime window;
- evidence provenance.

No Production schema was touched.

**Pre-declaration design changes.** Both came from focused unit tests on synthetic correspondences, with no data output:
- Robust Sampson refinement of the essential (R, t) was added, because unrefined OpenCV RANSAC E gave 0.84° rotation error at 0.3 px noise.
- The coherent-foreground test exposed that a rigidly moving foreground group can masquerade as parallax (§10). The test was split into a robustness test and a documented-confound test. **The estimator was not changed for it.**
- Dev smoke runs on 40 W140 clean frames measured runtime only, with no GT score. All of this is listed in the declaration.

## 4. Observability definitions (per edge, frozen)

| State | Condition | Meaning |
|---|---|---|
| NO_DETECTABLE_MOTION | rotation-only model wins GRIC and its angle < max(0.02°, 3σ/(f√n)) | no motion above the evidence's detectability; **not** a claim of zero translation |
| ROTATION_ONLY_CONSISTENT | rotation-only model wins | rotation observed; translation not resolvable at this span |
| TRANSLATION_PRESENT_PLANAR_AMBIGUOUS | homography wins, rotation-only loses | warp is not a pure rotation; direction ambiguous (planar scene + translation, or other model misfit). Rotation is used only if the essential decomposition passes cheirality (≥ 50 % and ≥ 30 points) |
| TRANSLATION_OBSERVED | essential wins | rotation and translation **direction** observed; scale never |
| INSUFFICIENT_EVIDENCE | < 30 matches or inliers | none |

**Regime per frame** (edges within ±0.5 s; ≥ 3 valid edges, otherwise INSUFFICIENT_EVIDENCE):
- TRANSLATING if ≥ 50 % of the edges show translation present or observed;
- otherwise ROTATING_TRANSLATION_UNRESOLVED if the median rotation rate is ≥ 2°/s;
- otherwise NEAR_STATIC_TRANSLATION_UNRESOLVED.

**Planar ambiguity is tested explicitly.**
- A translating camera over a plane is fitted by a homography but is never classified as rotation-only.
- A pure rotation over a plane is classified rotation-only.

## 5. Synthetic evidence (controlled)

**Scenes.** A1–A3 are new renders by the **unchanged** W141 dynamic-scene script, pointed at a W140-format directory: the W140 `scene.blend`, intrinsics and timestamps, plus a new camera GT.
- The static environment and the two walking humanoids are identical to W141.
- A1's and A3's frame 0 matches the W140 render (mean absolute difference 0.004).
- A4 is the W141 synthetic dynamic scene (W140 trajectory).
- Each scene has 140 frames at 30 fps. The GT stays outside the estimator inputs.

**Trajectories:**

| Scene | Camera motion |
|---|---|
| A1 | exactly stationary |
| A2 | fixed centre; yaw ±12°, pitch ±5°, roll ±4° sinusoids |
| A3 | 4.3 cm smooth translation plus a ±2° yaw |
| A4 | W140 segments: stationary / translation / rotation / mixed |

**RGB person masks** (frozen W141 segmentation) against the oracle on A1–A3: IoU median 0.80–0.82; 9–18 frames with foreground but no detection.

**Results** (RGB mask, primary; oracle and no-mask agree within 0.01°):

| Scene | Coverage | Aligned orientation error, median / p90 / max | Constant baseline | 1 s relative error (baseline) | Edge states | Regime (GT label) | False translation | Frozen SfM (same masks) |
|---|---|---|---|---|---|---|---|---|
| A1 stationary | 100 % | 0.0000 / 0.0000 / **0.0002°** | 0.0° | 0.000° (0.000) | 777 NO_DETECTABLE | NEAR_STATIC (NEAR_STATIC ✓) | **0/777** | **0 % registered** |
| A2 rotation | 100 % | **0.002** / 0.003 / 0.004° | 9.39° | 0.003° (13.07) | 777 ROTATION_ONLY | ROTATING (ROTATING ✓) | **0/777** | **0 %** |
| A3 small translation | 100 % | 0.147 / 0.204 / 0.207° | 1.40° | 0.109° (1.46) | 37 NO_DET, 553 ROT_ONLY, 187 PLANAR (spans 16/32) | NEAR_STATIC (GT: 32 NEAR_STATIC, 108 ROTATING) | n/a | **0 %** |
| A4 W140 trajectory | 100 % | 0.007 / 0.033 / 0.097° | 0.87° | 0.013° (8.56) | 83 / 261 / 276 / 157 TRANSLATION_OBSERVED | TRANSLATING; translation segment 100 % | 0/323 | W141 SfM 100 %, 0.026° |

**A3, small translation, is honest but imperfect.**
- At spans 1–8, the 4.3 cm motion is indistinguishable from rotation (ROTATION_ONLY), so translation is reported UNRESOLVED and never as zero.
- At spans 16 and 32 the classifier detects non-rotational warp (79/124 and 108/108 edges).
- The rotation-only model absorbs some translation into rotation: 0.15° error against 0.002° on A2. This is the known small-translation bias.
- 43 GT-ROTATING frames were labelled NEAR_STATIC; their GT rotation is just above the 1° label threshold.

**A4 regime misses.** Of GT-TRANSLATING frames, 27 % were classified NEAR_STATIC or ROTATING, in the mixed and boundary windows. The frozen gate counted only frames 20–69, which were 100 % correct.

**Frozen SfM.** COLMAP (W141 masked wrapper) registered **0 frames** on A1, A2 and A3: no initial pair without parallax. This reproduces W141's walking_static failure mechanism under control.

## 6. walking_static (primary real evaluation)

**Inputs.** The exact W141 300-frame selection, fr3 calibration, timestamps, GT association (with duplicate-timestamp removal) and frozen W141 person masks.

**GT camera motion** over the selected frames is real: up to 6.9 cm from the mean, an 11.0° rotation range, a 0.44 m interpolated path.

**Results:**

| Condition | Orientation coverage | Aligned median / p90 / max | Constant baseline (same frames) | Consecutive relative error (baseline) | 1 s relative error (baseline) | Spikes > 2° | Span-1 inliers on person mask |
|---|---|---|---|---|---|---|---|
| **W142, RGB person mask (primary)** | **52.7 %** (158; 142 OTHER_COMPONENT, 16 components) | **0.62 / 0.90 / 1.30°** | 2.07° | 0.144° (0.291) | **0.245° (1.234)** | 0 | 0 % |
| W142, no mask | 31.3 % | 0.16 / 0.47 / 0.67° | 1.35° | 0.134° (0.261) | 0.185° (0.912) | 0 | 10.5 % |
| W142, W141 area control | 31.7 % | 0.17 / 0.34 / 0.91° | 1.31° | 0.188° (0.196) | 0.280° (0.850) | 0 | 15.7 % |
| W141 SfM A, unmasked (re-scored) | 25.0 % | 2.79° | 1.85° | — | — | — | — |
| W141 SfM B, person-masked (re-scored) | 7.7 % | 1.25° | 1.63° | — | 2.41° (2.33) | — | — |
| W141 SfM C (re-scored) | 34.3 % | 2.03° | 2.59° | — | — | — | — |

The ablations cover different and smaller frame subsets, so their medians are not directly comparable with the primary; each is compared with its own baseline.

**Mechanism behind the coverage loss.**
- Of 1,738 edges, 1,401 were TRANSLATION_PRESENT_PLANAR_AMBIGUOUS, including 77 % of span-1 edges, where the GT step is about 1–2 mm.
- For those edges the essential decomposition rarely passes cheirality (tiny baseline), so they contribute no rotation. Only 263 edges are rotation-valid and the pair graph fragments.
- On the edges it did use, the estimator was accurate:

  | Edge type | n | Relative rotation error, median | GT rotation, median |
  |---|---|---|---|
  | rotation-only | 321 | 0.16° | 0.55° |
  | planar edges with a valid essential rotation | — | 0.49° | — |
  | TRANSLATION_OBSERVED | 10 | — | — |

  The 10 TRANSLATION_OBSERVED edges have an unreliable direction (57° median error).

**Temporal stability.**
- No consecutive error above 2°. Reported spikes (> 1°) are at frames 216→218, 223→231 and 233→293 (component boundaries).
- Drift slope −0.02°/s.
- The figure shows the estimate following the GT's 0→10° pitch drift over 0–19 s, while the constant baseline error grows to about 6°.

**Regime.**
- TRANSLATING for 274 frames and NEAR_STATIC for 17.
- The GT labels (≥ 5 cm in ±0.5 s means TRANSLATING) give no TRANSLATING frame: 242 ROTATING, 55 NEAR_STATIC.
- Routing therefore sent 277 frames toward SfM, which has only 23 of them. 254 frames are `ORIENTATION_ONLY_SFM_MISSING`.

## 7. walking_xyz (moving-camera control)

**Results** (primary; no-mask in parentheses):
- coverage 99.3 % (99.7 %);
- aligned median 0.70° (0.53°), p90 1.45°;
- baseline 5.74°;
- 1 s relative error 0.70° against 5.19°;
- spike fraction 2 %: frames 201→204 and 204→207, 3.8° and 3.4°.

**Edges:**
- 1,145 TRANSLATION_OBSERVED, with a **direction error median of 2.7°** against GT;
- 483 planar-ambiguous;
- 52 rotation-only;
- 57 INSUFFICIENT (long spans with little overlap).

**Regime.** TRANSLATING in all 300 frames, against GT labels of 285 TRANSLATING and 14 ROTATING. There was **no false stationary frame**.

**Routing.** All 300 frames are routed to the frozen W141 person-masked 6DoF SfM, which under the same orientation alignment scores **0.26°** (1 s: 0.37°).

So on a moving camera the W142 rotation path is about 2.7× less accurate than SfM. It complements SfM: it does not replace it, and the selector correctly keeps SfM.

## 8. Authentic S1 (observational, no GT)

**Source.** The AnimCV S1 clip: Wikimedia "Double Kettlebell Strict Press (side view)" (author Taco fleur, CC BY-SA 4.0).
- SHA-256 `de8feb9d…` verified; segment of source frames 79–318 (240 frames, 30 fps), located through the W125 action sidecar.
- Not redistributed.
- No trusted intrinsics exist, so a **nominal** f = 640 px (image width) is used, labelled `NOMINAL_UNCALIBRATED`. Calibrated rotation accuracy is **not claimed**; the K-free and image-space diagnostics are primary.

**Evidence and masks.**
- The scene is mostly sky with slow clouds, a sand strip and a horizon.
- Black letterbox rows (26.7 % of the image) were excluded by a predeclared rule.
- RGB person occupancy is 3.1 %.
- Background keypoints: median **45 per frame**; every frame has < 100.

| | Person-masked (primary) | No mask |
|---|---|---|
| Span-1 inliers, median | 39.5 | — |
| Background flow, median (span 1) | **0.054 px** | 0.125 px |
| Edges | 1,177 NO_DETECTABLE_MOTION, 200 INSUFFICIENT, **0** rotation / translation | 89 TRANSLATION_OBSERVED, 28 planar |
| K-free model | homography on every edge at every span | fundamental on 61 of 208 edges at span 32 |
| Inliers on the person, among translation-claiming edges | — | median **35 %** (POST-HOC) |
| Regime | NEAR_STATIC for all frames | NEAR_STATIC overall |

The orientation output is 238 frames ESTIMATED (nominal K), all ≈ identity.

**Reading:**
- Within the detectability of sparse sand and horizon evidence (nominal limit ≈ 0.04°), **no camera motion is observed**: consistent with a fixed camera.
- Translation is unobservable: distant, sparse evidence and no metric reference.
- The no-mask run shows the person's motion masquerading as camera translation in an authentic AnimCV source.
- **This does not identify the source of S1 Root Jitter.** Nothing was compared with root motion and no animation was touched.

## 9. POST-HOC diagnostics (after the GT evaluation; never gating)

File: `results/posthoc/posthoc.json` `c83ec3ed…`, script in the same folder.

**1. State against GT translation** (walking_static primary): fraction of edges labelled translation-present (planar), by GT translation of the pair.

| GT translation | Fraction labelled translation-present |
|---|---|
| 0–2 mm | 72 % |
| 2–5 mm | 76 % |
| 5–10 mm | 84 % |
| 10–20 mm | 93 % |

The no-mask run gives 75 → 96 %. The decision depends **weakly** on real translation and is dominated by a systematic misfit that a homography absorbs.

**2. Forced rotation-only** (the GRIC decision ignored; NOT an eligible estimator), walking_static primary, same features and spans:
- coverage **100 %**;
- aligned median **0.17°** (p90 0.30°), against a 2.44° baseline;
- 1 s relative error 0.18° against 1.35°;
- 0 consecutive errors above 2°.

The rotation information was present in the background evidence. The 52.7 % coverage is a consequence of the declared observability rule.

**3. S1 no-mask:** see §8.

**Incident.** The post-hoc script crashed once with a native segmentation fault and completed on an identical rerun. No official stage was affected (all 22 official runs returned rc 0).

## 10. False motion and unsupported translation

**Values.** No numeric translation or position was output anywhere: 0 frames across all 19 runs. ATE was not computed for W142, and no pseudo-position was created.

**False translation.** 0 of 777 edges on A1 and on A2, where GT translation is exactly zero, in every mask condition.

**False rotation.** A1 maximum 0.0002°. S1 shows no detectable motion.

**False stationarity.**
- No NO_DETECTABLE_MOTION edge has a GT rotation above 0.5° (A1–A4 and walking_static; the single walking_static case had 0.2°).
- walking_xyz has 0 NEAR_STATIC frames.
- A4 had 8 GT-TRANSLATING frames labelled NEAR_STATIC (mixed / boundary windows).

**Over-claimed translation evidence.** walking_static is labelled TRANSLATING where GT moved only millimetres to centimetres, as §9 shows. No value was produced, but the regime label and routing are wrong for that sequence.

**Coherent foreground confound** (unit test, then authentic S1 no-mask): a rigidly moving foreground group is explained by the essential model as parallax. Person exclusion removes it: S1 primary has 0 translation edges.

## 11. Background-feature and mask accounting

**Keypoints:** 0 keypoints on excluded pixels in every masked run, plus the preflight extraction check on walking_static, A1 and S1.

| Dataset | Background keypoints per frame (median) | Span-1 inliers (median) |
|---|---|---|
| walking_static | 2,289 | 1,110 |
| walking_xyz | 1,659 | 555 |
| synthetic | 3,700–3,750 | 2,360–3,580 |
| S1 | 45 | 39.5 |

**Masking cost and benefit.**
- Masking removed about 17 % of walking_static keypoints (2,744 → 2,289) and did not starve any frame (minimum 1,000).
- Without masks, 10.5 % of walking_static span-1 inliers lie on people. RANSAC tolerated this there, but on S1 people produced spurious translation claims.
- The area control (people kept, static area removed) raised the person share of inliers to 15.7 %.

## 12. Positive and negative mechanisms

**Positive:**
1. Background rotation is observable without any triangulation in near-static and low-parallax video:
   - synthetic 0.002°;
   - walking_static 0.16–0.62° on covered frames (0.17° on all frames in the forced POST-HOC diagnostic);
   - all far below the constant-camera error.
2. The estimator never fabricates translation or position. It abstains explicitly.
3. Pure rotation, no motion and planar translation are separated correctly on synthetic data, including the planar trap.
4. The moving-camera regime is recognised (walking_xyz), and the SfM 6DoF path is preserved by routing.
5. Person exclusion prevents person motion from masquerading as camera translation (S1).

**Negative:**
1. On real near-static video the rotation-only model is rejected in favour of a homography on most edges, for reasons only weakly tied to translation. The conservative rule then discards their rotation, and coverage falls to 52.7 % (gate failed).
2. The regime label TRANSLATING and the resulting routing are unreliable on walking_static: routed to SfM that cannot register.
3. On a moving camera the rotation path is 2.7× less accurate than SfM.
4. Small real translation biases rotation-only estimates: synthetic A3 0.15°.
5. S1 provides very little static evidence and no calibration.

## 13. Alternative explanations and uncertainty

**Homography preference on walking_static.** Candidate causes, not separated here: real millimetre parallax of near objects; residual lens distortion or intrinsic error in the factory-undistorted fr3 images; rolling shutter; unmasked non-person movers (chairs, monitors); and a GRIC σ (1 px) that does not match real feature noise. The no-mask and area-control runs behave the same, which argues against person contamination as the cause. W142 did not test calibration refinement, rolling-shutter models or alternative σ (not authorised after outcomes).

**Cheirality failures.** Expected for tiny baselines. A different policy, such as taking the rotation-only rotation on planar-ambiguous edges, is what the POST-HOC diagnostic approximates. It ignores the translation ambiguity the rule was designed to respect.

**Small-sample caveats:**
- The walking_static primary median is on the 52.7 % covered frames (0–19 s, the largest component).
- The ablation coverages differ.
- Each run was executed once. The pipeline is deterministic (seeded RANSAC); multi-threaded OpenCV was not checked for bit-reproducibility.

**S1.** "No detectable motion" is limited by 45 keypoints per frame and the nominal focal; very slow or sub-0.05 px motion cannot be excluded. Slow cloud drift lies in the excluded or textureless sky and was not separately masked.

**Not tested:**
- other estimators (homography-based rotation extraction, joint calibration, rolling-shutter-aware or learned methods);
- other near-static sequences;
- longer windows;
- human review.

## 14. Numerical and qualitative artifacts (not reviewed by a human)

Files are in `results/visualize/` (`visualize.json` `9158fcc3…`) and copied to `user_eval/W142_near_static_camera/` (Korean guide inside).
- `{syn_A1..A4, walking_static, walking_xyz}_orientation_regimes.png`:
  - GT against W142 orientation (rotation vectors relative to the first GT frame, evaluation-only chordal alignment);
  - error against the constant baseline and the re-scored W141 SfM;
  - timelines of W142 regime, per-edge state and GT label.
- `walking_static_flow_compensation.png` (frames 10→18, 120→128, 200→208, 280→288), `walking_xyz_…`, `s1_…`, `syn_A1/A2_…`: background flow before and after rotation compensation (×5), with the person mask.
- `s1_background_diagnostics.png`: S1 flow and residual, K-free model per span, regime.
- `synthetic_masks_features.png`: RGB against oracle masks, with kept background keypoints.
- `walking_static_playback_realtime.mp4` (25.0 s for a 24.9 s span) and `s1_playback_realtime.mp4` (8.0 s for an 8.0 s span): RGB with exclusion and background keypoints, plus per-frame orientation, state, regime and translation status. Real-time pacing: selected frames are held for their true interval.

No 3D-viewport camera video was produced: W142 has no camera positions to place.

## 15. Resources and reproducibility

**Runtime:**

| Item | Time |
|---|---|
| W142 estimation, 19 runs, 3 in parallel | **17.9 min** wall (sum of per-run walls 52.8 min, measured under concurrent load) |
| Frozen SfM controls | 4.4 min |
| Segmentation, 3 synthetic scenes + S1 | about 6.3 min |
| Renders | 23 s |
| Total primary estimation | about 22 min (≤ 2 h declared) |

**Storage.** Scratch 1.4 GB (≤ 15 GB declared).

**Evidence manifest.** `results/evidence_manifest.json` (3,669 files), `bba4ee3eaed233a57e8d1fa31af1f0b218a270186fab673b0e4f79132115da84`.

| File | SHA-256 |
|---|---|
| preflight | `af318cf1…` |
| declaration | `fb0a4d28…` |
| estimate_result | `0f438f5b…` |
| routing | `b2c94ddf…` |
| evaluation | `a3821571…` |
| visualize | `9158fcc3…` |
| post-hoc | `c83ec3ed…` |

**Reproduction** (Git Bash, at `61b4900`):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w142/results; H=scripts/w142_harness.py
$PY -m pytest -q tests/test_w142_background_rotation.py tests/test_w141_dynamic_camera_pose.py tests/test_experimental_camera_pose.py
$PY -X utf8 $H preflight --out-dir $R/preflight            # renders A1-A3 with the unchanged W141 script, segments, builds S1 input
$PY -X utf8 $H declare   --out-dir $R/declare --preflight $R/preflight/preflight.json
D="--declaration $R/declare/declaration.json --declaration-sha <sha>"
$PY -X utf8 $H estimate  --out-dir $R/estimate --preflight $R/preflight/preflight.json $D
$PY -X utf8 $H evaluate  --out-dir $R/evaluate --preflight $R/preflight/preflight.json $D --estimate-dir $R/estimate
$PY -X utf8 $H visualize --out-dir $R/visualize --preflight $R/preflight/preflight.json $D --estimate-dir $R/estimate --eval-dir $R/evaluate
```

## 16. Tests and scope

**Tests:**

| Item | Status |
|---|---|
| `tests/test_w142_background_rotation.py`: 24 | **24 PASS** |
| W141 tests (30) and W140 tests (22), reused | **52 PASS** |
| Extraction-only mask contract (walking_static, A1, S1) | **PASS** |
| 0 keypoints on excluded pixels, all masked runs | **PASS** |
| W140 / W141 identity verification | **PASS** |
| 19 W142 runs and 3 SfM controls | **RUN** (rc 0 ×22) |
| Production regression | **NOT RUN** (Production untouched) |
| Human review | **NOT RUN** |

The 24 W142 tests cover:
- Kabsch and the `M = R_CjCi` convention; optical-flow direction for a rightward pan;
- robustness to incoherent outliers, and the documented coherent-foreground confound;
- constant camera → NO_DETECTABLE; small translation → UNRESOLVED (never zero); large translation → observed direction;
- the planar-translation trap, and pure rotation over a plane;
- negative controls: wrong focal, transposed rotation; degenerate input → INSUFFICIENT; GRIC ordering;
- rotation averaging with an outlier edge, and components (other gauge); regimes and GT labels;
- chordal alignment and the constant baseline;
- masked features never on excluded pixels; the rounding convention;
- end-to-end on warped real-texture images: rotation within 0.1°, frame IDs and timestamps preserved, translation null;
- GT leak and bad calibration refused; an invalid frame kept in the denominator;
- the declared runs and trajectories.

**Separation of evidence:**

| Aspect | Evidence |
|---|---|
| Numeric and synthetic correctness | Shown, including the planar trap and abstention |
| Real measured capability | Accurate background orientation in the near-static, human-containing walking_static where SfM fails, on 52.7 % of frames under the frozen rule. The motion-state and translation-observability classifier is **not** reliable on that real sequence |
| Moving-camera reconstruction | Preserved by routing to SfM |
| Authentic S1 | Consistent with a fixed camera, on very thin, uncalibrated evidence. It establishes nothing about root jitter |
| Production readiness | None; nothing is integrated |

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 142
