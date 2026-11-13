# Worklog — Static-Scene RGB Camera Pose Reconstruction (COLMAP SfM) on Synthetic, TUM fr1_xyz and fr1_rpy: Pilot SUPPORTED (2026-11-13)

Batch 140 was an experimental, research-only Pilot. It ran on Main with this setup:

- local machine: 32 CPU threads, RTX 5080 (not used by the estimator), Windows 11;
- local branch `Dual_Stage_Generator`, with no branch operation of any kind;
- estimator environment: scratch venv `C:/Users/dna10/animcv-w140/env/pc311` (CPython 3.11.9, pycolmap 4.2.1, numpy 2.4.6, opencv-python-headless 5.0.0);
- evaluation and plots: the unmodified AnimCV venv (numpy, scipy 1.17.1, matplotlib 3.11.0, cv2 4.11.0);
- Blender 5.1.0 for the synthetic scene.

Other conditions:

- Network use was limited to the two selected TUM archives, the four official TUM documentation pages, and the pip installation of the pinned estimator into the scratch venv.
- No LLM or other external API was called, apart from one read-only audit subagent working inside this session.
- `docs/next_batch.md` was not touched. Nothing was pushed. The automation is still paused.

Every number below is **AGENT-RUN**. Plots and videos were produced but **not reviewed by a human**. The agent viewed the trajectory/error plots and one failure frame as still images.

## 1. Official classification

**SUPPORTED** (Pilot utility only, not Production readiness).
- Source: `results/evaluate/evaluation.json`, `3bb32caa90e76059e19a2cd8f95eb9ccb994e059754b07963a56f30d7e6fb953`.
- The frozen rule requires the synthetic gate, the fr1_xyz gate and the fr1_rpy gate to pass. All three pass, by wide margins:

| Gate (declared, frozen) | Threshold | Result |
|---|---|---|
| Synthetic contracts (oracle projection) | max ≤ 1.5 px, negatives worse | median 0.13 px, max 0.30 px; all 5 negatives clearly worse |
| Synthetic stationary interval | centre spread ≤ 1 % of extent, rotation ≤ 0.5° | 0.0024 % of extent, 0.0002° |
| Synthetic recovery | ATE ≤ 2 % of extent, median rotation ≤ 1°, coverage ≥ 90 %, time lag 0 | ATE 0.033 % (0.56 mm / 1.71 m), 0.048°, 100 %, lag 0 |
| fr1_xyz | coverage ≥ 70 %, Sim(3) ATE RMSE < 0.20 m, better than the stationary baseline | coverage **100 %**, ATE **9.1 mm**, baseline 184.6 mm |
| fr1_rpy | coverage ≥ 70 %, median rotation error < 5°, better than the stationary baseline | coverage **99.7 %**, median **1.85°** (p90 3.0°), baseline 27.8° |

**What this establishes.** A pinned, established RGB-only estimator can recover a geometrically consistent camera trajectory from static-scene images:
- the declared coordinate contracts are verified end to end, and the estimator inputs contain no GT, depth or IMU;
- after an evaluation-only Sim(3) alignment, the recovered trajectory agrees with independently measured motion-capture GT to about 1 cm (translation-dominant scene) and about 2° (rotation-dominant scene).

**What it does not establish:**
- absolute metric scale, an absolute world frame or position (RGB-only monocular, §9);
- online or causal tracking (COLMAP is offline global SfM with bundle adjustment);
- robustness beyond two short, textured office sequences;
- anything about S1, human root motion or the AnimCV animation pipeline.

## 2. Revisions and ownership

| Item | Value |
|---|---|
| Branch | `Dual_Stage_Generator`. At the start the working tree was clean apart from the pre-existing untracked `third_party/mmpose/` |
| Starting revision | `4279a20` (after the user-requested merge `592f56a` of `arch/single_frame_first` 22d1771 = W138 COMPLETE) |
| Instrument commit | `a2aba8a`: `src/app/experimental_camera_pose.py`, `scripts/w140_colmap_estimate.py`, `scripts/w140_synthetic_scene.py`, `scripts/w140_camera_harness.py`, `tests/test_experimental_camera_pose.py` |
| Instrument fix (pre-estimate) | `af9cfe1`: synthetic scene v3 and strict oracle marker detection (§6) |
| Declaration commit | `ff47f9f`: `examples/experimental_camera_pose/w140_declaration.json`, scratch SHA-256 `4a0a118c2012760c2fd8fb9a768681dbf9c17f3b53d379e64770409bbe0e9fbe` |
| Ending revision | the commit adding this Worklog and `docs/AUDIT_W136_W140_2026_11_13.md` |

**Branch history note.** W138 (`arch/single_frame_first`) and W139 (`Dual_Stage_Generator`) are both reachable from HEAD, because of the merge the user requested before this Batch. W140 performed no merge, rebase, cherry-pick or other branch operation. It consulted W138 only through its existing documents.

## 3. Datasets, sources, rights and identities

**A: synthetic.**
- Deterministic Blender scene (`scripts/w140_synthetic_scene.py`, v3):
  - static textured boxes and walls, flat Workbench shading, 640×480;
  - lens 24 mm / sensor 36 mm, so fx = fy = 426.667 px, principal point (319.5, 239.5) in the OpenCV convention;
  - 140 frames at nominal 30 fps.
- Segments:
  - **stationary** 0–19;
  - **translation-dominant** 20–69 (1.64 m, fixed orientation);
  - **rotation-dominant** 70–119 (fixed centre, yaw ±22°, pitch ±8°);
  - **mixed** 120–139.
- GT (`gt/camera_gt.txt`, `T_WC` in TUM format) and the oracle markers are stored **outside** the estimator input.
- Scene `.blend` and GT SHA-256 are recorded in the preflight (`ddc76702…`).

**B / C: TUM RGB-D fr1_xyz and fr1_rpy.**
- Official source `https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_{xyz,rpy}.tgz`, downloaded 2026-11-13. Only these two archives were downloaded.
- Archive SHA-256:
  - xyz `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98` (448,204,271 B);
  - rpy `78103722a25873dbbb4de027eaa8c810a6382100691f02b6cf95c3adc91c4ac1` (410,268,381 B).
- Contents: 798 / 723 RGB frames. `groundtruth.txt` comes from motion capture, per the official page: "the position of the optical center of the color camera with respect to the world origin as defined by the motion capture system".
- Licence: "licensed under a Creative Commons 4.0 Attribution License (CC BY 4.0)" (dataset page, clause checked in the preflight).
  - Attribution: J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012.
  - Archives and extracted data stay in scratch `C:/Users/dna10/animcv-w140/data` and were never committed.
- Calibration, from the official "Freiburg 1 RGB" row: fx 517.3, fy 516.5, cx 318.6, cy 255.3, d0..d4 = 0.2624 −0.9531 −0.0054 0.0026 1.1633. The page notes that only fr3 images are pre-undistorted.

**Frame selection** (frozen before inference, without GT): `round(linspace(0, N−1, 300))` over rgb.txt order.
- Real timestamps are kept (median Δt 0.100 s for xyz, 0.068 s for rpy). No frame was unreadable.
- The ID lists are in the declaration.

## 4. Input contract and independent GT

**Estimator inputs, and only these:**
- RGB PNGs (TUM frames undistorted with `cv2.undistort` using the published K and d0..d4, new matrix = K);
- `frames.json` (frame ID, original timestamp, image name);
- `camera.json` (PINHOLE intrinsics, OpenCV pixel convention).

**Never given to the estimator:** `groundtruth.txt` or the synthetic GT, motion capture, depth, accelerometer, human pose, or any GT-derived pose, scale or alignment.

**Leak guard.** The runner refuses any input entry other than `images/*.png`, `frames.json`, `camera.json` and `input_manifest.json` (tested in `test_estimator_refuses_gt_depth_or_extra_inputs`). Depth exists in the archives but was never read.

**GT access.** GT was read only by `evaluate`, after `estimate` had frozen every `estimate.json` by SHA-256:
- synthetic `d766d161…`;
- xyz `8f62e7c5…`;
- rpy `5be72c64…`.

**Association.** GT at each estimated frame's timestamp is interpolated (linear position + SLERP rotation) between the two bracketing GT samples, and only if their gap is ≤ 0.05 s. There is no index matching.

| | Associated / estimated | Nearest-GT \|Δt\| median / p95 / max |
|---|---|---|
| xyz | 298 / 300 | 3.6 / 4.6 / 7.6 ms |
| rpy | 296 / 299 | 2.5 / 4.9 / 45 ms |
| synthetic | 138 / 140 | GT sampled exactly at frame times |

- Unassociated frames lie outside a bracket. For the synthetic data this is the first and last frame, an artefact of the interpolation rule at the sequence ends. It is reported, not repaired.
- Coverage always uses all selected frames as the denominator.

## 5. Estimator

**Method.** COLMAP incremental Structure-from-Motion through pycolmap 4.2.1 (CPU), `scripts/w140_colmap_estimate.py`.

**Configuration** (frozen in the declaration):
- SIFT with up to 8,192 features;
- sequential pairing, overlap 10 (quadratic), no loop detection;
- a single shared PINHOLE camera with the given intrinsics, **not refined** (COLMAP cx/cy = OpenCV + 0.5);
- incremental-mapper defaults, including COLMAP's own initialisation relaxation;
- random_seed 0, 8 threads;
- multiple models allowed, with primary = the largest model.

**Status categories:**
- ESTIMATED: registered in the primary model;
- INVALID: secondary model or unreadable input;
- LOST: never registered;
- PREDICTED_INTERPOLATED: never produced, since there is **no interpolation**;
- UNOBSERVABLE: not used.

**Output.** An isolated experimental format (`animcv_exp_w140_camera_pose_v1`):
- timestamp, frame ID, status, `T_WC` (arbitrary gauge and scale), model ID;
- diagnostics: keypoints, observed landmarks, track error;
- the primary model's landmarks.

The format is not integrated anywhere.

## 6. Coordinate and transform definitions, and contract evidence

**Definitions:**
- `T_WC(t)` maps OpenCV optical camera axes (+X right, +Y down, +Z forward) to world, so X_W = R_WC X_C + c, where c is the **camera centre**.
- `T_CW = T_WC⁻¹`. Its translation `−R_WCᵀ c` is the extrinsic translation and is **not** the centre.
- Quaternions are scalar-last `[qx qy qz qw]` (TUM order).
- Blender camera axes convert as R_WC(optical) = R_W,blendercam·diag(1, −1, −1).
- TUM world (motion capture, metres) ≠ COLMAP world (arbitrary) ≠ AnimCV world. These worlds are related only by the evaluation-only Sim(3).

All conventions are in the module docstring, copied verbatim into the declaration. None was changed after results.

**Control C (oracle contract, preflight, synthetic GT + marker landmarks, never an estimator input).** 108 marker observations:

| Projection variant | Median / max error |
|---|---|
| declared convention | **0.13 / 0.30 px** |
| inverted transform | ≥ 1.5e4 px (behind the camera) |
| quaternion read as wxyz | behind the camera |
| focal × 1.1 | 13.5 / 31.8 px |
| y-axis flip | behind the camera |
| +1-frame time shift | 3.7 / 33.5 px |

**Disclosed pre-estimate fix.**
- The first official preflight's oracle contract **FAILED** (median 174 px). Saturated colours in the random texture rectangles contaminated the marker-centroid detector; the projections themselves landed on the rendered markers (e.g. yellow predicted (548.6, 218.9) vs visible ≈ (548, 218)).
- The scene was re-rendered as v3, with texture colours limited to [0.15, 0.85] and strict marker thresholds (`af9cfe1`). The preflight was then re-run; the failed one is kept at `results/preflight_v2scene/`.
- An earlier v1 → v2 change removed ~12× texture tiling on the walls.
- Dev smoke runs on the first 25 and first 70 v2 frames exercised the code paths. 25 frames: no initial pair (expected degeneracy). 70 frames: 70/70 registered. No GT score was computed.
- All of this is disclosed in the declaration.

## 7. Synthetic results (Dataset A)

140/140 frames ESTIMATED in one model:
- mean landmark error 0.19 px;
- 13,942 landmarks, mean track length 51;
- consecutive-pair inlier ratio 0.9998.

**After the evaluation-only Sim(3)** (scale 0.164 because the estimate's units are arbitrary; alignment rotation 94.2°):
- ATE RMSE **0.56 mm** over a 1.71 m trajectory;
- median rotation error **0.048°**;
- RPE over 1 s: 1.0 mm / 0.005°;
- best time lag 0.

**Per segment** (aligned estimate vs GT):

| Segment | ATE | Median rotation error | Estimated vs GT motion |
|---|---|---|---|
| Stationary (0–19) | 0.2 mm | 0.049° | net translation 0.0 vs 0.0; rotation range 0.0002° vs 0 → **no invented motion** |
| Translation (20–69) | 0.3 mm | 0.048° | path 1.6378 vs 1.6378 m; spurious rotation range 0.012° |
| Rotation (70–119) | 0.8 mm | 0.044° | rotation range 43.99° vs 43.99°; **spurious translation path 16.5 mm** (net 0.6 mm) vs 0 |
| Mixed (120–139) | 0.5 mm | 0.047° | 0.9576 vs 0.9578 m; 11.91° vs 11.92° |

During pure rotation the camera centre is not observable from that interval alone. SfM pins it through landmarks seen from translated views, yet about 1 % of the extent still appears as jitter.

**Declared spike metric (> 5 × median step): not meaningful here.** The stationary and pure-rotation segments set the median translation step close to zero, so every moving step is flagged. This is reported as a property of the metric on piecewise-motion data, not as an estimator spike.

**Limitation.** Synthetic PASS reflects an idealised scene: flat shading, rich unique texture, no noise, blur or distortion. It verifies contracts and pipeline wiring, not real-world quality.

## 8. Real results

### 8a. fr1_xyz: translation along the principal axes, limited rotation (GT extent 0.92 m, rotation range 34°)

**Registration:**
- 300/300 ESTIMATED in one model; LOST 0, INVALID 0, INTERPOLATED 0;
- 21,204 landmarks, mean track length 18;
- median 2,165 keypoints and 1,312 observed landmarks per frame;
- consecutive-pair inlier ratio median 0.985 (min 102 inliers);
- mean landmark error 0.79 px;
- parallax proxy (baseline / median depth) median 0.029, p10 0.009.

**After full Sim(3)** (scale 0.0465, i.e. one estimate unit ≈ 21.5 m in GT, a pure gauge choice):

| Metric | Value |
|---|---|
| ATE RMSE / median / max | **9.1 / 6.6 / 29 mm** |
| Rotation error, median / p90 | 1.55° / 2.04° |
| RPE, consecutive (≈ 0.1 s) | 6.7 mm RMSE / 0.33° |
| RPE, 1 s | 15.5 mm RMSE / 0.76° |
| Best time lag | 0 |
| Temporal spikes | none (estimated step median 28.3 mm vs GT 27.9 mm) |
| Stationary baseline | ATE 184.6 mm, rotation 5.26° |

The estimator beats the baseline by 20× in translation and 3× in rotation.

### 8b. fr1_rpy: roll/pitch/yaw-dominant, small translation (GT extent 0.30 m, rotation range 173°)

**Registration:**
- 299/300 ESTIMATED, 1 LOST, INVALID 0, INTERPOLATED 0;
- the LOST frame is ID 113 (t = 1305031233.294): severe motion blur during a fast rotation (see `rpy_failures.png`);
- 11,235 landmarks, mean track length 16.6;
- median 1,446 keypoints and 597 observed landmarks per frame;
- consecutive-pair inlier ratio median 0.978, but the minimum is **22 inliers** (a weak pair);
- mean landmark error 1.05 px;
- parallax median 0.0105, p10 0.0041 (low).

**After full Sim(3)** (scale 0.0158):

| Metric | Value |
|---|---|
| Rotation error, median / p90 / max | **1.85° / 3.0° / 4.46°** |
| ATE RMSE / median | 18.7 / 13.2 mm |
| RPE, consecutive | 9.3 mm / 0.47° |
| RPE, 1 s | 30.2 mm / 2.2° |
| Best time lag | **−1 frame** (RMSE 18.0 vs 18.7 mm at lag 0) |
| Temporal spikes | one translation spike at index 46: estimated step 75 mm vs GT 30 mm, near the low-inlier pairs |
| Estimated rotation step | median 5.0° per selected frame, max 19.1° |
| Stationary baseline | rotation 27.8°, ATE 63.6 mm |

The estimator beats the baseline 15× in rotation and 3.4× in translation.

The −1 lag is marginal: 0.7 mm in RMSE, about 0.08 s. It may reflect a camera-to-motion-capture time offset or noise. It was not investigated or corrected.

## 9. Alignment and scale (observability)

**Scale.**
- Monocular RGB gives no metric scale. Every estimated trajectory is `SCALE_AMBIGUOUS`.
- The Sim(3) scales (0.164, 0.0465, 0.0158) are COLMAP gauge choices, **not** recovered metric scales. No metric scale was supplied externally.
- ATE is reported in GT metres **only after** that evaluation-time scale fit. It is therefore evidence of trajectory **shape** fidelity, not of metric reconstruction.
- An SE(3)-only (metric) evaluation is invalid here and was not computed (§15 tests the distinction).

**World frame.**
- The alignment rotation and translation (e.g. 153.6° for xyz) are arbitrary gauge.
- The estimator does **not** recover the motion-capture world origin or orientation. A static scene defines only a *local* world: COLMAP's first registered pair.

**Early-segment alignment (declared second protocol).** Sim(3) fitted on the first 20 % of frames by time, then applied unchanged to the rest:

| | Rest ATE | Rest median rotation error |
|---|---|---|
| synthetic | 228 mm | 136° |
| xyz | 29 mm | 8.2° |
| rpy | 35 mm | 18.2° |

**These numbers measure alignment conditioning, not drift.** The early-segment GT position spread is close to rank 1:

| | Singular values (m) |
|---|---|
| synthetic | [0.027, 0, 0] (stationary plus a straight translation) |
| xyz | [0.155, 0.012, 0.006] |
| rpy | [0.033, 0.016, 0.006] |

A position-only Sim(3) therefore cannot fix the rotation about the motion axis, and the rotation error stays a **flat offset over time** (plots), never a growing one.

**POST-HOC diagnostic** (written after the GT evaluation, labelled as such, never used for a gate: `results/posthoc/posthoc_diagnostics.json` `9064a94e…`, script in the same folder). An early alignment whose rotation comes from the early segment's orientations gives these remaining errors:

| | Rest ATE | Rest median rotation error | Rotation-error slope | Position-error slope |
|---|---|---|---|---|
| xyz | 9.9 mm | 0.48° | −0.015°/s | −0.3 mm/s |
| rpy | 28 mm | 1.21° | +0.001°/s | +0.4 mm/s |
| synthetic | 3.1 mm | 0.006° | — | — |

There is **no measurable drift** over these 24–27 s sequences, which is consistent with globally bundle-adjusted SfM.

**Residual rotation offset (unresolved).** The full position-based Sim(3) rotation differs from the orientation-optimal alignment by 1.5° (xyz) and 1.4° (rpy). Under the orientation-only alignment the median rotation error drops to 0.48° and 1.32°. Possible causes, not separated here:
- the motion-capture marker-to-optical-centre calibration in the GT;
- a residual rectification or intrinsic bias;
- time offset;
- position noise in the alignment.

## 10. Summary metrics

| | Coverage | ATE RMSE (Sim3) | Rotation median / p90 | RPE 1 s, translation / rotation | Drift (post-hoc slope) | Spikes | Lost / interpolated |
|---|---|---|---|---|---|---|---|
| Synthetic | 100 % | 0.56 mm (0.03 % of extent) | 0.048° / — | 1.0 mm / 0.005° | ≈ 0 | metric not meaningful (§7) | 0 / 0 |
| fr1_xyz | 100 % | **9.1 mm** | **1.55° / 2.04°** | 15.5 mm / 0.76° | ≈ 0 | 0 | 0 / 0 |
| fr1_rpy | 99.7 % | 18.7 mm | **1.85° / 3.0°** | 30.2 mm / 2.2° | ≈ 0 | 1 translation spike | 1 (blur) / 0 |

## 11. Recovered and failed intervals

- **Recovered:** every selected frame except fr1_rpy frame 113.
- **Failed:** frame 113 (a single LOST frame; blur from a fast roll). Its neighbours were registered.
- **Weak interval:** around rpy index 46. The minimum consecutive-pair inlier count is 22, and the scaled estimate makes one 75 mm jump there against a 30 mm GT step, a local translation error that rotation does not show.
- **Low parallax:** rpy (median baseline/depth 0.0105) has translation determined mostly through scene structure. Its ATE (18.7 mm) is twice xyz's despite a 3× smaller extent; relative to extent, translation is weaker in the rotation-dominant scene.
- **No reinitialisation:** a single model everywhere, so there are no reinitialisation jumps.

## 12. Baselines and controls

| Control | Result |
|---|---|
| A, stationary best-constant pose (most favourable: GT mean centre + chordal-mean GT rotation) | xyz ATE 184.6 mm / 5.26°; rpy 63.6 mm / 27.8°; synthetic 669 mm / 0.79°. The estimator beats it on every dimension of every sequence. |
| B, RGB-only estimator | §7–8 |
| C, synthetic oracle projection | §6: the convention is verified to 0.30 px max, and all five negatives are detected |

## 13. Qualitative evidence

In `C:/Users/dna10/animcv-w140/results/visualize/`, copied to `user_eval/W140_camera_pose/`:
- `{synthetic,xyz,rpy}_trajectory_errors.png`: 3D and planar GT vs ALIGNED ESTIMATE, position and rotation error over time (full and early-segment alignment), frame-to-frame steps;
- `*_tracks.png`: model-observed feature overlays on four frames;
- `rpy_failures.png`: the TRACKING LOST frame;
- `*_synchronized_30fps.mp4`: each selected RGB frame with its model-observed features next to a top-view of GT (black) and ALIGNED ESTIMATE (red), with the current frame's error. The selected frames are played at 30 fps for inspection, not at their original timing.

**Not reviewed by a human.** The agent viewed the xyz and rpy error plots and the failure frame as stills:
- xyz: the aligned estimate overlays the GT "cross" closely;
- rpy: the aligned estimate follows the small, convoluted GT path with visible local jitter.

## 14. Positive mechanisms and limitations

**Positive:**
- Contract-correct geometry, verified on rendered pixels.
- Strictly RGB-only inputs, enforced.
- Full registration on all real sequences except one blurred frame.
- About 1 cm / 2° agreement with independent motion capture.
- No drift over about 25 s.
- No time-shift or axis error on synthetic and xyz.
- The stationary segment does not invent motion.

**Limitations:**
- Offline global SfM, not online tracking.
- Scale and world frame unobservable.
- Two short, textured, static office scenes.
- Weak translation under rotation-dominant, low-parallax motion: spurious translation in the synthetic rotation segment, the rpy spike and its 2× ATE.
- Motion blur defeats registration.
- An unexplained 1.4–1.5° alignment rotation offset and a marginal −1-frame lag on rpy.
- Runs are not bit-reproducible (multi-threaded mapping). Each dataset was run exactly once.
- No dynamic objects or humans are in view. Whether the method survives a moving person in frame, as in S1, is **untested**.

## 15. Test status (RUN / PASS / FAIL / NOT RUN)

| Item | Kind | Status |
|---|---|---|
| `tests/test_experimental_camera_pose.py` (22) | synthetic, independent references (scipy Rotation, `cv2.projectPoints`) | **22 PASS** |
| Oracle projection contract + 5 negative variants | real render, synthetic GT | **PASS** (failed once before the v3 fix, disclosed) |
| Estimator on synthetic / xyz / rpy | real | **RUN** (rc 0 ×3) |
| Pilot gates | declared | **PASS ×3** |
| Production regression | — | **NOT RUN** (Production untouched, §16) |
| Human review of plots and videos; depth or IMU diagnostics; other datasets; parameter variation | — | **NOT RUN** (not authorised) |

The 22 tests cover:
- SE(3) inverse and composition; camera centre vs extrinsic translation;
- quaternion order and normalisation; wxyz misread detection; Blender → optical axes;
- projection vs `cv2.projectPoints`, including wrong focal and axis sign;
- invalid calibration (4 cases);
- malformed trajectories (column count, non-monotonic, quaternion norm, NaN);
- interpolation and gap/span refusal; deterministic selection;
- exact Sim(3) recovery of gauge and scale; metric vs scale-ambiguous (SE(3) cannot absorb scale); known noise → known ATE;
- inverted transforms and the one-frame time shift detected;
- insufficient and degenerate estimates;
- baseline and gates; invented stationary motion detected;
- the judgment mapping;
- GT/depth/IMU/extra-input leak refusal.

## 16. Runtime, storage, reproducibility, isolation

**Runtime.** Estimator total ≈ 11.1 min, well inside the 2 h budget:

| Dataset | Total | Extract | Match | Map |
|---|---|---|---|---|
| synthetic | 244 s | 5.5 s | 71 s | 165 s |
| xyz | 180 s | — | — | — |
| rpy | 244 s | — | — | — |

Evaluation and visualisation take under 2 min. The GPU was not used.

**Storage:** 3.5 GB within the 8 GB budget.
- `data/` 2.0 GB (archives 0.86 GB + extracted + synthetic v1–v3);
- `results/` 1.3 GB;
- `dev/` 0.23 GB;
- `env/` 0.26 GB, excluded from the budget as installed tools.

**Evidence manifest.** `results/evidence_manifest.json` (1,558 files) `28888c2790842a45fde363028e2dd12e2275dc314317337b791c98726fcd7ab6`.
- preflight `ddc76702…`, declaration `4a0a118c…`, estimate_result `305cb1d8…`, evaluation `3bb32caa…`, visualize `6c5f60a8…`, post-hoc `9064a94e…`.

**Isolation.**
- `git diff 4279a20..HEAD` touches only the six W140 files plus this Worklog and the audit.
- No non-experimental `src` file changed, and no Production module imports `experimental_camera_pose`.
- `tests/test_camera_calibration.py` and `tests/test_camera_self_calibration.py` are unchanged.
- No FramePose, MotionGraph, S1 calibration, root, spine or bone code was touched.
- No AnimCV Action or animation output was produced or modified.

**Reproduction** (Git Bash, at `ff47f9f`; data and environment as in §3/§5):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w140/results; H=scripts/w140_camera_harness.py
$PY -m pytest -q tests/test_experimental_camera_pose.py
"C:/Program Files/Blender Foundation/Blender 5.1/blender.exe" -b --factory-startup --python-exit-code 1 -P scripts/w140_synthetic_scene.py -- --out C:/Users/dna10/animcv-w140/data/synthetic_v3
$PY -X utf8 $H preflight --out-dir $R/preflight
$PY -X utf8 $H declare   --out-dir $R/declare --preflight $R/preflight/preflight.json      # commit the copy under examples/
D="--declaration $R/declare/declaration.json --declaration-sha <sha>"
$PY -X utf8 $H estimate  --out-dir $R/estimate --preflight $R/preflight/preflight.json $D
$PY -X utf8 $H evaluate  --out-dir $R/evaluate --preflight $R/preflight/preflight.json $D --estimate-dir $R/estimate
$PY -X utf8 $H visualize --out-dir $R/visualize --preflight $R/preflight/preflight.json $D --estimate-dir $R/estimate --eval-dir $R/evaluate
```

COLMAP's multi-threaded mapping is seeded but not guaranteed bit-identical across reruns.

## 17. Evidence scope and unresolved observability questions

**Scope.** Two TUM fr1 office sequences (24–27 s each, 300 selected frames), one synthetic scene, one estimator configuration, one run each.

**Unobservable without external information:**
- metric scale (needs a metric reference: a known object size, depth, stereo, IMU, or a known camera height);
- the absolute world origin and orientation (needs a known map, markers or an external positioning system);
- gravity direction (RGB-only; an IMU or a known floor/vertical prior would supply it);
- camera translation during pure rotation without previously triangulated structure.

**Unresolved:**
- the 1.4–1.5° residual rotation offset;
- the marginal −1-frame lag on rpy;
- behaviour with a moving human occupying the view;
- online/causal operation;
- long sequences (no loop closure was configured);
- the reproducibility spread of COLMAP reruns.

## 18. Cumulative W136–W140 audit

Separate record: `docs/AUDIT_W136_W140_2026_11_13.md`. It was drafted by a read-only audit subagent, reviewed, and completed with the W140 isolation facts above.

**Summary:**
- No HIGH finding.
- One MEDIUM: a `scripts/batch_automation.py` edit (`0d82883`) inside the W138 window, which the W138 Worklog both names and contradicts.
- LOW items:
  - W139's preflight used an unfetched origin ref;
  - the W139 instruction is not bound in the repository;
  - one W136 frame transcription error;
  - open rights questions;
  - drifting living documents.
- All four W136–W139 evidence manifests (16,937 files) re-hashed with no mismatch.
- Every declared instrument is unchanged at HEAD.
- Production is untouched across W136–W140.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 140
