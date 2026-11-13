# Worklog — Decoupled Camera Orientation and Translation Observability: Validated Background Rotation on Known walking_static and Held-Out sitting_static, Moving-Camera Control and S1: Pilot PARTIALLY SUPPORTED (2026-11-13)

Batch 143 was an experimental, research-only Pilot on the local branch `Dual_Stage_Generator`. There was no branch operation and nothing was pushed.

Environment:

- local Windows CPU;
- AnimCV venv (OpenCV 4.11, numpy, scipy; torch for the frozen W141 segmentation);
- W140 scratch venv for the frozen COLMAP SfM;
- scratch `C:/Users/dna10/animcv-w143/`, never committed.

Network use was limited to one download: the TUM `rgbd_dataset_freiburg3_sitting_static` archive. `docs/next_batch.md` and `scripts/batch_automation.py` were not touched, and the automation stays paused.

All numbers are **AGENT-RUN**. The agent viewed the figures as stills; **no human has reviewed** the figures or videos.

## 1. Official classification

**PARTIALLY SUPPORTED** (frozen rule; `results/evaluate/evaluation.json` `d0b04f99ed8b18b2104e9b347fca10a357a0c2a4d805d0287f153fb96921c146`).

| Gate (frozen before output) | Result | Status |
|---|---|---|
| **Known walking_static** | Coverage **96.0 %** (W142: 52.7 %). Aligned median **0.20°** against a constant baseline of 2.24°. 1 s relative error 0.19° against 1.33°. 0 spikes. On the 156 frames common with W142: **W143 0.17° vs W142 0.62°**. No numeric translation | **PASS** |
| **Held-out sitting_static** | Coverage **97.7 %** (frozen W142 on the same data: 61.7 %). Aligned median **0.50°** against 1.93°. 1 s relative error 0.23° against 1.43°. 0 spikes. The GT rotation range (10.8°) makes the baseline test meaningful. 0 translation-direction frames (GT window displacement median 9 mm). No numeric translation | **PASS** |
| **Moving-camera walking_xyz** | **100 % FULL_6DOF_AVAILABLE** from the frozen W141 SfM (model valid: 8,872 landmarks, 0.93 px). 0 SfM-valid frames downgraded; 0 near-static frames | **PASS** (see §10 caveat) |
| **Synthetic A1–A4** | A1 max false rotation 0.0002°. A2 0.002°. A3 0.15°. A1/A2 false translation directions 0/777. **A4 orientation coverage 47 % (< 80 %). A4 translation-segment DIRECTION_ONLY frames 0 % (< 50 %)** | **FAIL** |

SUPPORTED requires all four gates. NOT_YET is excluded because both real near-static gates pass.

**Answer to the completion question:**

- **What W143 measures.** Camera **orientation** was measured reliably from static background, independently of translation observability, on two near-static human-containing sequences.
  - On the held-out one, sitting_static (never used for design; GT read only after freezing), coverage is 97.7 % and accuracy 0.50° against a 1.93° constant-camera error.
  - Nothing beyond W140–W142 is claimed about camera position: no translation value exists unless a valid SfM reconstruction supplies it, and then only with scale ambiguous.
- **The moving-camera 6DoF path is not damaged.** walking_xyz frames all keep the valid SfM pose. The near-static sequences, whose SfM is invalid (walking_static) or failed (sitting_static), are no longer routed to SfM.
- **Translation-observability aggregation is not usable.** The frame-level rule reported 0 % TRANSLATION_DIRECTION_ONLY even on translating cameras (A4, walking_xyz), although the edge-level direction estimates were accurate (2.0° on walking_xyz). This is why the synthetic gate failed.
- **The quality gain on held-out data is coverage, not accuracy.** On the 182 frames both cover, W142 (0.24°) is more accurate than W143 (0.38°).

## 2. Revisions, ownership, preserved baselines

| Item | Value |
|---|---|
| Branch | `Dual_Stage_Generator`. At start the working tree was clean apart from the pre-existing untracked `third_party/mmpose/`. `origin` already equalled HEAD `a11a312`; W143 pushed nothing |
| Starting revision | `a11a312` (W142 Worklog) |
| Instrument commit | `e67b88e` |
| Declaration commit | `02bbddb`: `examples/experimental_camera_pose/w143_declaration.json` `1f0de7faa669682665a9faee75672be836af7a3eea2024fe4db77285f472f15a` |
| Ending revision | the commit adding this Worklog |

**Changed files** (all new):

- `src/app/experimental_decoupled_camera_state.py`;
- `scripts/w143_decoupled_estimate.py`, `scripts/w143_harness.py`, `scripts/w143_mechanism.py`, `scripts/w143_visualize.py`;
- `tests/test_w143_decoupled_camera_state.py`;
- `examples/experimental_camera_pose/w143_declaration.json`;
- this Worklog.

**Preserved and verified in the preflight:**

- the W142 declaration `fb0a4d28…`;
- W142 evaluation `a3821571…`, estimate_result `0f438f5b…`, routing `b2c94ddf…`, manifest `bba4ee3e…` and post-hoc `c83ec3ed…`;
- all 22 frozen W142 estimates;
- every W142 instrument;
- W141 evaluation `c370522e…` and W140 evaluation `3bb32caa…`;
- all reused inputs and masks, byte-identical to the W141 and W142 declarations.

No historical estimate, metric or Worklog was modified. W142 functions are imported unchanged; W143 adds only new modules.

## 3. The architectural separation

W142 decided four questions with one GRIC model choice per frame pair. W143 answers them separately (`app.experimental_decoupled_camera_state`):

| Question | W142 | W143 |
|---|---|---|
| 1. Geometric model fit | GRIC (R / H / E) decides everything | GRIC kept only as a **descriptive** label |
| 2. Is the rotation a valid orientation measurement? | only if R wins GRIC, or E decomposes with cheirality | rotation-only estimate **validated by image evidence**, regardless of the GRIC winner (§5) |
| 3. Is translation observable? | "H or E wins" is enough to call translation present | **direction only**, and only if E wins, cheirality holds, rotation-compensated parallax ≥ 2 px, and another span from the same frame agrees within 20°. **H preference never implies translation** |
| 4. Is 6DoF available? | regime label TRANSLATING routes to SfM | **only** from a frozen SfM reconstruction's own evidence (§6). Labels never upgrade a frame |

**Camera states per frame:**

| Field | Values |
|---|---|
| Camera | `FULL_6DOF_AVAILABLE` / `ORIENTATION_ONLY_AVAILABLE` / `CAMERA_MOTION_UNRESOLVED` |
| Orientation | `ORIENTATION_ESTIMATED` / `ORIENTATION_UNSUPPORTED`, with evidence: accepted edges, cycle-verified edges, confidence `CYCLE_VERIFIED` / `UNVERIFIED`, residual-based bias indicator |
| Translation | `TRANSLATION_ESTIMATED_SCALE_AMBIGUOUS` (SfM only, in the SfM gauge) / `TRANSLATION_DIRECTION_ONLY` / `TRANSLATION_UNRESOLVED` / `TRANSLATION_INSUFFICIENT_EVIDENCE` |

Each frame also carries its reference frame, SfM model and frame status, timestamp and provenance. The value is null unless SfM supplies it, and the W143 orientation gauge and the SfM gauge are never mixed. No Production schema was touched.

## 4. Why W142 discarded rotation evidence (Phase 1, pre-declaration)

Source: `results/mechanism/mechanism.json` `3aa08801…`. It combines synthetic correspondences with fr3 intrinsics and GT-free historical W142 statistics; no held-out data was used.

**1. Homography preference does not make a rotation measurement unusable.**

| Condition | Model GRIC prefers | Rotation-only error |
|---|---|---|
| Residual radial distortion k1 = 0.05 | H | **0.04°** |
| Rolling shutter (0.3°/frame) | H | 0.20° |
| Planar translation | H | — |
| Intrinsic errors of 1–2 % or 5 px | **R** (no flip) | — |

Small non-rotational image effects make a homography win even when rotation is well measured. On real fr3 images the residual source was not separated (§13).

**2. The essential decomposition is genuinely ill-conditioned at millimetre baselines.**

- Only 2–6 points pass cheirality at 5–10 mm baselines, and the essential rotation errs by 0.16–0.23°.
- The hypothesis that OpenCV's `recoverPose` distance cut-off caused the failures was tested and **rejected**.
- W142 was therefore right to distrust the essential rotation, but it also threw away the better-conditioned rotation-only estimate on the same edges.

**3. Translation-induced rotation bias.**

- 2 mm of translation at 1–5 m depth gives a 0.04° error and is accepted.
- 10–30 mm gives 0.16–0.42° and is rejected by the support and residual criteria.
- A planar 30 mm translation is rejected (residual criterion), but its bias indicator (0.17°) under-states its 0.52° error. The indicator is **not** a guaranteed bound.

**4. GT-free W142 statistics on walking_static:** 71 % of the W142 "planar-ambiguous" edges and 92 % of its rotation-only edges pass the W143 support and residual criteria.

The known W142 outcomes, including its post-hoc forced rotation-only result, were known when the thresholds were set. That is disclosed in the declaration. The thresholds were derived as follows, not by searching walking_static GT:
- the RANSAC threshold (1.5 px);
- a majority rule (support 0.8);
- synthetic controls (agreement 0.5°, cycle 0.5°, parallax 2 px).

## 5. Candidate rotation-validation mechanism

All W142 front-end functions are reused unchanged: SIFT ≤ 4,000 on keep-mask pixels, mutual ratio 0.8, spans 1–32, rotation-only 2-point RANSAC with Kabsch refinement, homography, essential with refinement, and GRIC.

**Per-edge acceptance.** A rotation measurement needs all four:

1. ≥ 30 rotation inliers.
2. Rotation inliers ≥ **80 %** of the best model's inliers.
3. p90 rotational residual over the best-model consensus set ≤ **1.5 px**.
4. Rotation-only estimate within **0.5°** of the rotation nearest to K⁻¹HK.

**Then cycle consistency.**
- Every (s, s, 2s) triangle of accepted edges is checked.
- The edge with the worst median triangle error is removed greedily while that error exceeds **0.5°**.
- A single corrupted edge therefore does not reject its innocent partners; this was found and fixed by a pre-declaration unit test.

**Aggregation.** Accepted edges feed the unchanged W142 robust chordal rotation averaging. Orientation comes from the largest component; a frame is supported if at least one accepted edge stays within the 5° outlier limit.

**Bias indicator.** p90 residual / f is reported per edge and frame. It is **not** a bound; §12 tests its calibration.

**Relation to the W142 post-hoc diagnostic.** The W142 POST-HOC forced rotation-only result (100 % coverage, 0.17°) accepted every edge with ≥ 30 rotation inliers, with no validation. It is **not** the W143 candidate and was not used as a method. W143 rejected 26–28 % of the real near-static edges.

## 6. Translation observability and the routing contract

**Edge level.** An edge reaches **DIRECTION_OBSERVED** only if all of these hold:
- E wins GRIC;
- recoverPose cheirality passes (≥ 50 % and ≥ 30 points);
- rotation-compensated median parallax is ≥ 2 px;
- a different span from the same start frame agrees in direction within 20°.

Otherwise the edge is UNRESOLVED or UNRESOLVED_UNCONFIRMED (or INSUFFICIENT).

**Frame level (± 0.5 s window).** TRANSLATION_DIRECTION_ONLY requires ≥ 50 % of the valid window edges to be DIRECTION_OBSERVED; otherwise TRANSLATION_UNRESOLVED, or TRANSLATION_INSUFFICIENT_EVIDENCE with fewer than 3 edges.

**6DoF.** FULL_6DOF_AVAILABLE requires both levels:

| Level | Requirement |
|---|---|
| SfM model | registers ≥ 50 % of the frames, has ≥ 1,000 landmarks and a mean reprojection ≤ 2 px |
| Frame | ESTIMATED in the primary model with ≥ 100 observed landmarks |

The pose is then the SfM pose in its own gauge, scale ambiguous. No translation label can upgrade a frame.

**SfM sources** (all frozen estimator outputs, never GT):

| Dataset | SfM source | Validity |
|---|---|---|
| A1–A3 | W142 masked SfM | 0 frames, invalid |
| A4 | W141 syn_rgbmask | valid |
| walking_static | W141 B | 23 frames, 18 landmarks: **invalid** |
| walking_xyz | W141 B | **valid** |
| sitting_static | new W141-wrapper SfM | **0 frames, invalid** |
| S1 | new run | 0 frames, invalid |

## 7. Synthetic A1–A4 (W142 renders and masks; W141 A4)

| Scene | Coverage | Median / p90 / max | Baseline | 1 s relative (baseline) | Accepted edges | Accepted-edge error, median | Translation claims | Camera state |
|---|---|---|---|---|---|---|---|---|
| A1 stationary | 100 % | 0.0000 / 0.0000 / **0.0002°** | 0° | 0.000 (0) | 777/777 | 0.0001° | **0** (false-direction 0/777) | 140 ORIENTATION_ONLY |
| A2 rotation | 100 % | **0.0023** / 0.003 / 0.004° | 9.39° | 0.003 (13.07) | 777/777 | 0.001° | **0/777** | 140 ORIENTATION_ONLY |
| A3 small translation | 100 % | 0.146 / 0.204 / 0.209° | 1.40° | 0.108 (1.46) | 777/777, incl. all 187 W142-planar (0.080°) | 0.015° | 0 (unresolved, never zero) | 140 ORIENTATION_ONLY |
| A4 W140 trajectory | **47.1 %** | 0.100 / 0.262 / 1.32° | 12.8° | 0.013 (18.1) | 402/777 | 0.002° (rejected: **1.36°**) | 90 edges DIRECTION_OBSERVED, direction error **0.10°**; frames DIRECTION_ONLY **0** | **140 FULL_6DOF (valid SfM)** |

**Reading:**

- **A1 and A2.** Stationary and pure-rotation behaviour is exact, with no false translation. A1 and A2 match W142.
- **A3.** W143 accepts the edges W142 called planar-ambiguous. Its orientation (0.146°) equals W142's (0.147°): the small-translation bias in the chained orientation is neither removed nor flagged, because every edge passes the support and residual criteria. That orientation is useful but carries an unflagged 0.15° bias, and translation stays unresolved.
- **A4.**
  - Validation discriminates sharply on synthetic data: accepted 0.002° against rejected 1.36°. Rotation-only measurements under real translation are rejected, so translating frames lose orientation (47 % coverage); their 6DoF comes from the valid SfM.
  - The edge-level direction is accurate (0.10°), but the frame window never reaches 50 %, so the translation segment shows 0 % DIRECTION_ONLY (§10).
  - The 80 % A4 coverage gate was a frozen expectation that conflicts with the decoupled design: orientation-only is not meant for translating frames. It is reported as a failure, not reinterpreted.

The frozen SfM registered 0 frames on A1–A3, as in W142.

## 8. Known walking_static (exact W141/W142 inputs and masks)

| | W142 (historical) | **W143** |
|---|---|---|
| Orientation coverage | 52.7 % (16 components) | **96.0 %** (2 components; 288 ORIENTATION_ONLY + 12 UNRESOLVED) |
| Aligned median / p90 / max | 0.62 / 0.90 / 1.30° | **0.20 / 0.41 / 0.81°** |
| Constant baseline (own frames) | 2.07° | 2.24° |
| 1 s relative error (baseline) | 0.245° (1.234) | **0.191° (1.334)** |
| Consecutive relative error (baseline) | 0.144° (0.291) | 0.142° (0.306) |
| On the 156 common frames | 0.62° | **0.17°** |
| Regime / translation | TRANSLATING on 274 frames, routed to SfM | **TRANSLATION_UNRESOLVED on all 300**; no direction claims; GT window displacement median 9.5 mm |
| 6DoF | 23 frames claimed via routing, 254 SFM_MISSING | **0 FULL_6DOF** (W141 B SfM invalid) |

**Where the new coverage came from:**

| W142 edge state | Accepted by W143 | Relative rotation error vs GT (accepted) |
|---|---|---|
| planar-ambiguous | **989 of 1,401** | 0.169° |
| rotation-only | 294 of 323 | 0.162° |
| translation-observed | 0 of 12 | — |

**Accepted against rejected edges:**
- Accepted edges: 0.167° median (p90 0.32°).
- Rejected edges: 0.226°. Real-data validation discriminates only weakly: rejected edges are not much worse.
- Rejection reasons, overlapping: residual 402, support 314, homography agreement 93, cycle 0.

**Bias indicator.** Only 46 % of accepted edges have error ≤ indicator + 0.05°.

## 9. Held-out sitting_static

**Data.** Two people sit at a desk, talking and gesturing; the camera is held in place manually.
- Official TUM archive (HTTPS, size equals Content-Length): SHA-256 `6239d4b7f6edcab719b51c76434acdc238d03dad666758b51e6046992a7c3914`, 443,506,009 B. Last-Modified 2012-08-07. CC BY 4.0, attribution as W141.
- Only `rgb/`, `rgb.txt` and `groundtruth.txt` were extracted. Depth was never extracted. GT was hashed in the preflight (`13b03e25…`) and read only by evaluate.
- Freiburg 3 calibration row verified; no undistortion.
- 300 of 707 frames by the frozen `round(linspace)` rule.
- Frozen W141 segmentation: person occupancy median 25.8 % (people in every frame).
- GT motion: up to 8.0 cm from the mean, 10.8° rotation range, 0.43 m path.

| | Frozen W142 (same data) | **W143** |
|---|---|---|
| Orientation coverage | 61.7 % | **97.7 %** (293; 2 components) |
| Aligned median / p90 / max | 0.25° on its 185 frames | **0.50 / 0.81 / 1.10°** |
| Constant baseline | 1.73° | 1.93° |
| 1 s relative error (baseline) | 0.274° (1.458) | **0.233° (1.430)** |
| Consecutive relative error (baseline) | — | 0.104° (0.271) |
| On the 182 common frames | **0.24°** | 0.38° |
| Regime / translation | **TRANSLATING on 254 frames**: the W142 failure reproduced on held-out data | TRANSLATION_UNRESOLVED on all frames. 2 DIRECTION_OBSERVED edges (GT 4 cm, direction error 110°: wrong, but never aggregated into a frame claim) |
| 6DoF | — | 0 (SfM registered 0 of 300 frames) |
| Edge accounting | — | 850 of 1,322 W142-planar edges accepted (0.166°), 395 of 396 rotation-only (0.112°); accepted 0.153° vs rejected 0.234°; bias-indicator calibration 50 % |

**Generalisation.** The coverage mechanism and the removal of false translation routing reproduce on held-out data. The accuracy comparison with W142 does not: W143 trades about 0.14° of accuracy on the common frames for 36 percentage points of coverage, while remaining about 4× better than a constant camera.

The figure shows the W143 orientation following GT over all 23 s, with the largest errors (about 1°) in the first second.

## 10. Moving-camera regression (walking_xyz)

**6DoF path.** The W141 person-masked SfM passes the validity rule: 300/300 frames, 8,872 landmarks, 0.93 px. All 300 frames are FULL_6DOF_AVAILABLE, with the SfM pose in its own gauge and scale ambiguous. 0 valid frames were downgraded. The historical result is available unchanged; W141 scored it at 12.9 mm / 0.64° under its Sim(3), and W142 at 0.26° orientation.

**Orientation-only path.** It covered only 5.3 % (16 frames, 0.17°), correctly:
- 1,583 edges failed rotation support under real translation;
- accepted edges 0.23° against rejected 0.50°.

It never overwrote the SfM pose.

**Caveat on translation observability.** At edge level it works: 302 DIRECTION_OBSERVED edges with a **2.0°** median direction error against GT. At frame level, short-span edges with sub-2 px parallax keep the window fraction below 50 %, so **0 frames** are TRANSLATION_DIRECTION_ONLY, and the per-frame regime label reads ROTATING (149) or INSUFFICIENT (136) for GT-TRANSLATING frames.
- The frozen gate passed because routing depends on SfM evidence, not on that label.
- The frame-level translation-observability report is nonetheless **wrong for a translating camera**, though conservative: unresolved, never zero. W143 does **not** provide reliable frame-level translation observability.

## 11. Authentic S1 (GT-free, NOMINAL_UNCALIBRATED)

| Item | Value |
|---|---|
| Edges | 1,176 of 1,377 accepted |
| Rejections | 93 too few matches, 107 rotation inliers, 24 residual |
| Orientation | 238 / 240 ORIENTATION_ESTIMATED (nominal focal) |
| Translation | UNRESOLVED on every frame |
| SfM | 0 registered → **no 6DoF** |
| Background flow (median) | 0.054 px |
| Accepted nominal rotation (median) | 0.0055° |

**Reading:**
- W142's fixed-camera reading remains **consistent** with the evidence: no detectable motion at about 45 background keypoints per frame. This is not proof of a stationary camera.
- Calibrated orientation, metric position and root motion are not claimed.

## 12. Coverage, accuracy and invalid-state accounting

**Numeric translation without valid SfM:** 0 frames in every run.

**False translation-direction frames:** 0. A1/A2 false-direction edges: 0/777 each.

**Unresolved and invalid frames:**

| Dataset | ORIENTATION_UNSUPPORTED / CAMERA_MOTION_UNRESOLVED |
|---|---|
| walking_static | 12 (2 components) |
| sitting_static | 7 (first 0.5 s, and 1 frame at 16 s) |
| A4 | 74 translating frames, all covered by valid SfM |
| walking_xyz | 284, all covered by valid SfM |
| S1 | 2 |

**Cycle check.** It rejected **0** edges in every run: the earlier criteria removed everything inconsistent beyond 0.5°, or 0.5° is lax. Its contribution is unproven.

**Bias indicator.** It is calibrated only on synthetic data:
- A1–A3: 100 %;
- A4: 94 %;
- walking_static: 46 %;
- sitting_static: 50 %;
- walking_xyz: 34 %.

On real images it under-states error and is model-internal, not verified accuracy.

**Keypoints.** 0 on excluded pixels in all runs.

## 13. Positive and negative findings

**Positive:**

1. Decoupling rotation validity from model choice recovers previously discarded, accurate rotation evidence: 989 + 850 W142-planar edges at about 0.17°.
2. Near-static orientation coverage rises from 52.7 % / 61.7 % to 96.0 % / 97.7 %, both far better than a constant camera, **including on held-out data**.
3. False 6DoF routing of near-static video is gone. The SfM path is used only when its own reconstruction evidence is valid; the moving camera keeps 100 % 6DoF.
4. No position or translation value is fabricated. Stationary and rotation-only synthetic controls are exact.
5. Edge-level translation direction is accurate where truly observable: 0.10° (A4), 2.0° (walking_xyz).

**Negative:**

1. Frame-level translation observability is non-functional for translating cameras (0 % on A4 and walking_xyz). This is why the synthetic gate failed and the regime labels are wrong there.
2. On held-out data, W143 is less accurate than W142 on common frames: 0.38° against 0.24°.
3. On real data the validation criteria discriminate weakly (accepted 0.15–0.17° against rejected 0.23°). The bias indicator is not a bound there (46–50 %). The cycle check never fired.
4. A3's small-translation orientation bias (0.15°) is accepted and unflagged.
5. A4 orientation coverage is 47 % (frozen gate failed); the orientation path does not cover translating frames by design.
6. Two wrong direction-observed edges on sitting_static (110°) show that the edge-level direction test can still be fooled. Neither became a frame claim.

## 14. Alternative explanations and uncertainty

**Why homographies win on real near-static frames.** The synthetic controls point to residual distortion or rolling shutter, which leave rotation usable. Real millimetre parallax remains possible. These were **not** separated on TUM data.

**Accuracy gain on walking_static** (0.62° → 0.17° on common frames). It may partly reflect better graph connectivity (more redundant edges averaging errors) rather than better individual edges: accepted edge errors are similar to W142's.

**Held-out accuracy loss on common frames.** W143 includes more planar-ambiguous edges, whose errors are slightly larger (0.166° against 0.112° for rotation-only edges). The trade-off is coverage against accuracy.

**Thresholds.** They were chosen with the W142 walking_static outcomes known (disclosed). The held-out sequence is the only clean generalisation evidence. It is one sequence and one run, with a similar scene type: TUM office, Kinect, fr3 calibration.

**Not tested:**
- other camera types or lenses;
- rolling-shutter or distortion modelling;
- longer sequences;
- outdoor scenes;
- a frame-level translation rule weighting spans by expected parallax (not authorised after outcomes);
- human review.

## 15. Representative visual evidence (not reviewed by a human)

Location: `results/visualize/` (`visualize.json` `744c6846…`), copied to `user_eval/W143_decoupled_camera/` (Korean guide inside).

- **`{syn_A1..A4, walking_static, walking_xyz, sitting_static}_w143_orientation_state.png`:**
  - GT orientation against W143 and W142 (evaluation-only chordal alignment);
  - error curves against the constant baseline;
  - camera-state, translation-observability and GT-label timelines;
  - a per-span edge acceptance map (accepted, or rejected by reason) with oriented frames (pair-graph connectivity).
- **`{walking_static, sitting_static, s1}_w143_correspondences.png`:** two accepted and two rejected span-8 edges, raw flow (×5) beside the residual after rotation (×10), with support, agreement, residual, bias, translation and cycle annotations.
- **`s1_w143_state.png`:** S1 edge validation and camera state.
- **Videos** (real-time pacing; selected frames held for their true interval):

  | File | Length |
  |---|---|
  | `sitting_static_w143_playback_realtime.mp4` | 23.8 s for a 23.7 s span |
  | `walking_static_w143_playback_realtime.mp4` | 25.0 s for a 24.9 s span |

  Each shows the RGB frame with the person exclusion and background keypoints, plus the per-frame camera, orientation and translation states.

There is no 3D viewport video: the orientation-only path has no camera centre to place.

## 16. Tests

| Item | Status |
|---|---|
| `tests/test_w143_decoupled_camera_state.py` (16) | **16 PASS** |
| W142 (24), W141 (30), W140 (22) tests, reused | **76 PASS** |
| Held-out extraction-only mask contract (5 frames, 0 keypoints on excluded pixels) | **PASS** |
| 11 official executions | **RUN** (rc 0 ×11) |
| Production regression | **NOT RUN** (Production untouched) |
| Human review | **NOT RUN** |

The 16 W143 tests cover:
- homography-nearest rotation, pure against planar;
- a pure-rotation edge accepted;
- **H-preferred but rotation valid** (decoupling);
- planar translation rejected and not translation-observed;
- small-translation bias, accepted small and rejected large;
- general translation giving a direction candidate;
- insufficient matches;
- greedy cycle rejection isolating a single corrupted edge;
- cross-span translation confirmation;
- window semantics;
- SfM validity from reconstruction evidence only;
- camera state never upgraded by labels;
- end-to-end on warped real texture:
  - orientation within 0.1°, null translation, frame IDs and timestamps preserved, 0 keypoints on the masked box;
  - FULL_6DOF only from a valid SfM;
  - GT-leak and GT-as-SfM refused;
- harness held-out isolation.

## 17. Resources and reproducibility

| Item | Value |
|---|---|
| Official executions (11) | wall **9.0 min** (2 SfM sequentially, then 9 estimator runs 3 in parallel; per-run walls 5–272 s) |
| Held-out segmentation | 3.1 min |
| Primary-estimation budget | ≤ 2 h |
| Scratch | **1.3 GB** (≤ 10 GB) |
| Evidence manifest | `results/evidence_manifest.json` (1,680 files) `d6b1a1bb529e7fb2eb3add0059ed7f76b189d06c357858557aa17c9908c6c3f8` |

**SHA-256 of the key results:**

| File | SHA-256 |
|---|---|
| mechanism | `3aa08801…` |
| preflight | `81010231…` |
| declaration | `1f0de7fa…` |
| estimate_result | `eacade06…` |
| evaluation | `d0b04f99…` |
| visualize | `744c6846…` |

**Reproduction** (Git Bash, at `02bbddb`):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w143/results; H=scripts/w143_harness.py
$PY -m pytest -q tests/test_w143_decoupled_camera_state.py tests/test_w142_background_rotation.py tests/test_w141_dynamic_camera_pose.py tests/test_experimental_camera_pose.py
$PY -X utf8 scripts/w143_mechanism.py --out $R/mechanism
$PY -X utf8 $H preflight --out-dir $R/preflight
$PY -X utf8 $H declare   --out-dir $R/declare --preflight $R/preflight/preflight.json
D="--declaration $R/declare/declaration.json --declaration-sha <sha>"
$PY -X utf8 $H estimate  --out-dir $R/estimate --preflight $R/preflight/preflight.json $D
$PY -X utf8 $H evaluate  --out-dir $R/evaluate --preflight $R/preflight/preflight.json $D --estimate-dir $R/estimate
$PY -X utf8 $H visualize --out-dir $R/visualize --preflight $R/preflight/preflight.json $D --estimate-dir $R/estimate --eval-dir $R/evaluate
```

## 18. Official judgment and scope

**PARTIALLY SUPPORTED.**

| Aspect | Status |
|---|---|
| **Numeric and synthetic correctness** | Stationary and rotation controls are exact. The planar trap and decoupling are verified in unit tests. The synthetic gate failed on A4 frame-level translation observability and orientation coverage of translating frames |
| **Real measured orientation capability** | Validated background orientation is accurate and high-coverage on two real near-static human-containing sequences, one held out: 96.0 % at 0.20° and 97.7 % at 0.50°, against constant-camera errors of about 2°. On held-out common frames it is less accurate than W142 |
| **Camera-state validity** | Orientation, translation and 6DoF claims are separately qualified. No fabricated translation. 6DoF routing is trustworthy on the tested sequences because it rests on SfM reconstruction evidence. Frame-level translation observability is not trustworthy: it under-reports observed translation, and the bias indicator is uncalibrated on real data |
| **Position information** | Unavailable without a valid SfM: never estimated, never zero, scale always ambiguous |
| **S1** | Consistent with a fixed camera under thin, uncalibrated evidence. Nothing about root motion |
| **Production readiness** | None. Nothing is integrated |

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 143
