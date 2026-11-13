# Worklog — Dynamic-Scene Camera Pose Robustness: Frozen W140 COLMAP SfM with RGB Person Exclusion on TUM fr3 walking_static / walking_xyz and a Synthetic Moving-Occluder Scene: Pilot PARTIALLY SUPPORTED (2026-11-13)

Batch 141 was an experimental, research-only Pilot on the local branch `Dual_Stage_Generator` (no branch operation, nothing pushed). Setup:

- local machine: 32 CPU threads, RTX 5080 (unused), Windows 11;
- estimator: the **frozen W140** COLMAP incremental SfM (`scripts/w140_colmap_estimate.py`, pycolmap 4.2.1, CPU), in the W140 scratch venv `C:/Users/dna10/animcv-w140/env/pc311`;
- segmentation, evaluation and plots: the unmodified AnimCV venv (torch 2.13.0, torchvision 0.28.0+cpu, numpy, scipy, cv2 4.11, matplotlib);
- Blender 5.1.0 for the synthetic scene and the 3D viewport renders;
- scratch storage: `C:/Users/dna10/animcv-w141/` (data, model weights, results; never committed).

Network use was limited to:

- the two selected TUM archives;
- three official TUM documentation pages;
- the one torchvision Mask R-CNN checkpoint.

`docs/next_batch.md` and `scripts/batch_automation.py` were not touched. The automation stays paused.

Every number below is **AGENT-RUN**. The agent viewed plots and frame sheets as still images. **No human has reviewed** the videos or figures.

## 1. Official classification

**PARTIALLY SUPPORTED** (Pilot utility only, not Production readiness).
- Source: `results/evaluate/evaluation.json`, `c370522e244bd0d0af7081becdeeb811c27b8c9620e40003ed2edb6767d294e2`.
- Rule (frozen): SUPPORTED requires that an RGB-only configuration (A or B) pass the configuration gate on **both** real sequences.
- Here every configuration passes on walking_xyz, and every configuration fails on walking_static.

| Sequence | Condition | Coverage (of 300) | Sim(3) ATE RMSE | Median rotation error | Stationary baseline (ATE / rotation) | Gate |
|---|---|---|---|---|---|---|
| walking_xyz | A unmasked | **100 %** | **14.9 mm** | **0.90°** | 300.7 mm / 5.75° | **PASS** |
| walking_xyz | B RGB person exclusion | **100 %** | **12.9 mm** | **0.64°** | 300.7 mm / 5.75° | **PASS** |
| walking_xyz | C area control | 100 % | 15.1 mm | 0.90° | 300.7 mm / 5.75° | PASS |
| walking_static | A unmasked | 25.0 % | 19.4 mm (Sim(3) collapsed, §11) | 58.3° (alignment ill-conditioned, §13) | 21.3 mm / 1.85° | fail |
| walking_static | B RGB person exclusion | 7.7 % | 13.8 mm (collapsed) | 173.9° (ill-conditioned) | 14.1 mm / 1.63° | fail |
| walking_static | C area control | 34.3 % (+36 INVALID) | 25.2 mm (collapsed) | 150.1° (ill-conditioned) | 25.4 mm / 2.59° | fail |

The gate requires all of the following (§9):
- coverage ≥ 80 % of the 300 selected frames;
- ATE ≤ 0.05 m (walking_static) or ≤ 0.15 m (walking_xyz);
- median rotation error ≤ 5°;
- ATE **and** median rotation error both below the stationary best-constant-pose baseline.

**Masking-benefit verdicts** (frozen rule, §9):
- walking_xyz: **UNNECESSARY**. A already passes. B's gain (−2.0 mm, −0.26°) is consistent but below the declared materiality margin.
- walking_static: **HARMFUL** in the declared sense. B's coverage fell 17 pp below A's. Neither A nor B is a valid camera trajectory, though (§11).

**Answers to the completion question:**

1. **What the estimator recovered.**
   - On walking_xyz, a moving camera with two walking or sitting people, the camera trajectory was recovered with 100 % coverage and about 13–15 mm / 0.6–0.9° agreement with motion capture.
   - On walking_static, a nearly stationary camera (GT RMS spread 1.4–2.1 cm) with people moving through 18 % of the image on median, no condition recovered a usable camera trajectory.
2. **Whether human motion contaminated camera geometry.**
   - Yes, decisively on walking_static: 89 % of the unmasked model's observations and 88 % of its landmarks lie on people.
   - Mildly and locally on walking_xyz: an end-of-sequence excursion to 46 mm that B removes (§12).
   - Causally in the synthetic scene: 17× ATE increase, fully removed by oracle exclusion and not by the area control.
3. **Whether masking helped, hurt or was unnecessary.**
   - Unnecessary for passing on walking_xyz (small consistent gain).
   - It did not rescue walking_static.
   - In the synthetic scene, exact foreground exclusion fully prevents the disturbance. The RGB detector's poor recall on synthetic humanoids leaves most of it.
4. **Failures caused by insufficient static evidence.** walking_static under B: after the people are removed, the static background offers no usable parallax to a near-stationary camera. Keypoints remain plentiful (median 2,889 per frame); geometry, not feature count, is what is missing.
5. **What remains unobservable.**
   - metric scale and the GT world frame (as in W140);
   - camera translation and scale from a near-stationary camera;
   - on walking_static, even orientation: aligned by orientation, no condition beats the constant-pose baseline (§13).

## 2. Revisions and ownership

| Item | Value |
|---|---|
| Branch | `Dual_Stage_Generator` (local). The working tree at start was clean apart from pre-existing untracked `third_party/mmpose/` and the two W140 viewport scripts |
| Starting revision | `248beeb` (W140 Worklog + W136–W140 audit) |
| Concurrent commit (not W141) | `716a85b` "viz: W140 post-hoc third-person 3D viewport videos" (13:37). Another session committed the two previously untracked W140 viewport scripts into this checkout during W141. W141 reuses `scripts/w140_viewport_blender.py` unchanged for its viewport videos and did not author that commit |
| W141 instrument commit | `3d75d2c` (before any estimator output) |
| Disclosure commit | `6c4eda3`: the declaration lists the crashed first preflight (§6) |
| Declaration commit | `ebaeddd`: `examples/experimental_camera_pose/w141_declaration.json`, SHA-256 `6dd621bbb9c9fb4a442220271edd9824def4985abdd54d8ea15fb5402585ffbc` |
| Amendment 1 instruments | `c992226` |
| Amendment 1 record | `5e69612`: `examples/experimental_camera_pose/w141_declaration_amendment1.json`, `2d7f8c396376806441de5868605601581cd2ed163910fc644fc72ce52ec242f9`. Estimates frozen, **no GT score seen**; §6 |
| Ending revision | the commit adding this Worklog |

**Changed files** (`git diff 716a85b..HEAD`), all new:
- `src/app/experimental_dynamic_camera_pose.py`;
- `scripts/w141_dynamic_camera_harness.py`, `scripts/w141_colmap_masked_estimate.py`, `scripts/w141_segment_people.py`, `scripts/w141_synthetic_dynamic_scene.py`, `scripts/w141_visualize.py`;
- `tests/test_w141_dynamic_camera_pose.py`;
- `examples/experimental_camera_pose/w141_declaration.json`, `examples/experimental_camera_pose/w141_declaration_amendment1.json`;
- this Worklog.

No existing file was modified.

## 3. Frozen W140 identities and preserved baselines

Verified in the preflight (`w140_baseline.ok = true`) and unchanged at HEAD:

**W140 instruments.** All five W140 instruments match the LF-normalised SHA-256 in `w140_declaration.json`. `scripts/w140_colmap_estimate.py`, `src/app/experimental_camera_pose.py` and `tests/test_experimental_camera_pose.py` have 0 diff lines since `248beeb`.

**W140 results** (unchanged):

| File | SHA-256 |
|---|---|
| evaluation | `3bb32caa…` |
| evidence manifest | `28888c27…` |
| declaration | `4a0a118c…` |
| estimate_result | `305cb1d8…` |

**Reproduction check.** The W140 clean synthetic frames, re-estimated in W141 (`syn_clean`), reproduce the W140 synthetic result: 140/140 frames, ATE 0.55 mm, median rotation 0.048°, all W140 synthetic gates pass. W140's recorded figures were 0.56 mm and 0.048°. COLMAP's multi-threaded mapping is not bit-reproducible.

**Unchanged:** coordinate conventions (`T_WC` optical OpenCV axes, scalar-last quaternions, SCALE_AMBIGUOUS, evaluation-only alignment). No historical result was rewritten.

**W140 audit discrepancy.** The `scripts/batch_automation.py` discrepancy found in the W140 audit is unaffected: the file was neither modified nor run.

## 4. Real datasets: provenance, hashes, licence, calibration

**Source:** `https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_{walking_static,walking_xyz}.tgz`, downloaded 2026-11-13. Only these two archives were downloaded.

**Authentication.** TUM publishes no checksum. Each archive came from the official HTTPS URL, its byte size equals the server's Content-Length, and the server Last-Modified dates are recorded. The downloaded SHA-256 is the identity:

| Sequence | Archive SHA-256 | Size |
|---|---|---|
| walking_static | `8aca832bf01746aa95fcc0b15bf42d78ad3bb8edef672ba4cc94a404b194f6c1` | 474,074,105 B |
| walking_xyz | `1459e9488ac0e61a2ec80dfbc35cfb77942f6d8eabded1c8d26a70be650d0e1d` | 527,550,055 B |

**Extraction.** Only `rgb/`, `rgb.txt` and `groundtruth.txt` were extracted. Depth and accelerometer files were **never extracted** (the preflight checks this). The sequences have 743 and 859 RGB frames. The rgb.txt and groundtruth.txt SHA-256 are in the preflight.

**Official descriptions** (clauses verified in the saved download page):
- walking_static: "Two persons walk through an office scene. The Asus Xtion sensor has been kept in place manually." Ground-truth trajectory length 0.282 m, dimensions 0.10 × 0.05 × 0.05 m.
- walking_xyz: "…manually been moved along three directions (xyz) while keeping the same orientation." Length 5.791 m.

**Camera motion is real on walking_static.** Over the 297 associated selected frames, the GT camera centre moves up to 6.9 cm from its mean, the orientation range is 11.0°, and the consecutive-frame step median is 1.3 mm / 0.31° (POST-HOC descriptive, `posthoc_descriptive.json`). The camera is therefore **not stationary**, and W141 never treats it as identity.

The selected-frame interpolated path (0.44 m) differs from the published 0.282 m. TUM's path computation was not investigated; likely contributors are motion-capture jitter and sampling.

**Licence and rights:**
- "Unless stated otherwise, all data in the TUM RGB-D benchmark is licensed under a Creative Commons 4.0 Attribution License (CC BY 4.0)", verified in the saved page.
- Attribution: J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012.
- Data stays in scratch and is never committed or redistributed.

**Calibration** (official "Freiburg 3 RGB" row, verified verbatim): fx 535.4, fy 539.2, cx 320.1, cy 247.6, d0..d4 = 0.
- Clause: "Note that both the color and IR images of the Freiburg 3 sequences have already been undistorted, therefore the distortion parameters are all zero."
- Selected frames are therefore **byte copies with no undistortion**, unlike W140's fr1 path.
- Freiburg 1 values are not used anywhere; a test checks this.

**GT semantics:** optical-centre position and orientation of the colour camera in the motion-capture world. This is the same clause as W140, verified.

**Frame selection.** The W140 rule `round(linspace(0, N−1, 300))` over rgb.txt order was frozen without GT. It selects exactly **300 frames per sequence**. Original timestamps are kept, no frame was unreadable, and the ID lists are in the declaration.

**GT anomaly (amendment 1).** The official walking_static groundtruth.txt contains one duplicated timestamp, 1341846251.4587: two samples about 1 mm apart.

## 5. Synthetic dynamic scene (Dataset C)

Script: `scripts/w141_synthetic_dynamic_scene.py`, rendered in Blender, output in `data/synthetic_dynamic`.

**What is reused from W140, unchanged:**
- the W140 `scene.blend` (static geometry, textures, Workbench FLAT shading, intrinsics; SHA-256 recorded);
- the W140 GT camera trajectory (renderer only);
- byte copies of the W140 `camera.json`, `frames.json`, `camera_gt.txt` and `segments.json`. The preflight verifies these are byte-identical.

**What is added:** two articulated, human-shaped occluders, about 1.75 m tall.
- Built from pelvis, torso, neck, head, two-segment legs with feet, and two-segment arms.
- Clothing-like seeded textures, so they produce SIFT features.
- Each walks on a straight path with leg and arm swing, independent of the camera:
  - h1: 1.4 m from the camera start, 0.85 m/s, +x;
  - h2: about 3.3 m away, 0.80 m/s, −x.
- Exact foreground masks come from a second Workbench pass (object colour, no anti-aliasing): `gt/fg_masks`, the oracle.

**Checks:**
- **Background identity.** Frames re-rendered with the occluders hidden match the W140 frames: mean absolute difference ≤ 0.005 grey levels, and < 0.1 % of pixels differ by more than 8. The only differences are anti-aliasing samples.
- **Occupancy.** Oracle foreground covers 17 % of the image on average after dilation; the RGB person mask covers 5.3 % raw.

**Stationary-interval test.** The occluders are already walking in frames 0–19, where the W140 camera trajectory is stationary.

## 6. Input contract, leakage and declared changes

**Estimator inputs** are those of W140, and only those:
- `images/*.png`;
- `frames.json` with the original timestamps;
- `camera.json` (fr3 PINHOLE);
- for B, C, D and oracle_areactrl: a separate keep-mask directory derived from the same RGB frames (or, for D, from the oracle).

The W140 leak guard on the input directory is executed unchanged.

**Mask directory contract.** The masked wrapper refuses a mask directory unless it contains exactly one `<image>.png` per frame, single-channel uint8, of the image's size, with values in {0, 255}, and nothing else. It is tested with a planted `groundtruth.txt`, wrong shape, non-binary values, a colour image and a missing file.

**GT access.** GT, depth, IMU and oracle masks never entered the A, B or C estimation paths. GT was read only by `evaluate`, after all 11 `estimate.json` files were verified against the SHA-256 frozen in `estimate_result.json` (`ebce5a7d…`).

**Disclosed deviations and incidents** (all before any GT score):
1. **Preflight crash.** The first preflight crashed because Blender writes oracle masks as 3-channel grey PNGs, which the mask contract refused. The reader now accepts grey 3-channel input and refuses non-grey. The preflight was re-run from scratch; the crashed run is kept at `results/preflight_crash1_oracle_mask_channels`.
2. **Synthetic prototype.** The synthetic scene and Mask R-CNN were prototyped on 4 frames (`dev/syn_proto`; person IoU 0.29–0.86 against the oracle). Afterwards the nearer occluder was moved closer to raise occupancy. The occluders were never adjusted to improve detection.
3. **COLMAP mask check.** `mask_path` was dev-checked on two W140 frames (`dev/masktest`).
4. **Amendment 1.** The first `evaluate` stopped in the W140 GT reader on the duplicated walking_static timestamp before computing any score. Its output directory was empty and is kept as `results/evaluate_crash1_gt_duplicate_timestamp`.
   - The declared amendment removes **every** sample of a duplicated timestamp, with no arbitrary pick.
   - The remaining neighbours are 30 ms apart, inside W140's unchanged 50 ms bracket rule.
   - Estimates, masks, thresholds, metrics and gates were unchanged. The harness accepts an amendment only if it binds the declaration hash and states `gt_scores_seen: false`.
5. **Orphan notice.** After a session handover, the background estimate task was reported as stopped. The process tree was still running and finished the 11th run normally (`walking_xyz_C`, rc 0). No run was repeated.

## 7. Mask generation and the feature-exclusion mechanism

**Segmentation model.** One bounded checkpoint, with no model search and no training:
- `torchvision maskrcnn_resnet50_fpn_v2`, weights `COCO_V1`;
- `https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth`, SHA-256 `73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e`, verified before and after load;
- **Rights:** torchvision code is BSD-3-Clause. The weights are published by PyTorch and trained on COCO 2017 (annotations CC BY 4.0, images under Flickr terms). They were used for local inference only and are not redistributed.

**Inference.** CPU float32 with the model's own resize. Keep label `person` with score ≥ 0.5, threshold each soft mask at ≥ 0.5, and take the union over instances. Runtime: 185 s per 300 real frames, 79 s for 140 synthetic frames.

**Exclusion masks** (all parameters frozen in code before any mask existed):

| Condition | Exclusion mask |
|---|---|
| B | raw person mask, disk-dilated by 15 px (boundary error, blur, SIFT support) |
| C | deterministic area control from B (below) |
| D | oracle foreground, dilated by the same 15 px |
| oracle_areactrl | area control of D |

**Area-control construction (C):**
1. Shift B by a declared offset list, wrapping around; take the offset with the least overlap with B.
2. Remove the overlap with B.
3. Top up with non-B pixels nearest to C until the area matches.

C is always disjoint from B. It is exactly area-matched except in 4 walking_static frames with more than 50 % exclusion, where the background ran out (shortfall reported). It matches **pixel area, not keypoint count**: inside A's keypoints, C covered 23 % (walking_static) and 22 % (walking_xyz) fewer keypoints than B. People are more textured than average background.

**Mechanism.**
- COLMAP's own extractor drops keypoints that fall on zero-valued pixels of `ImageReaderOptions.mask_path`.
- The wrapper `scripts/w141_colmap_masked_estimate.py` sets that one option and then executes the unmodified W140 estimator code: same SIFT, matching and mapper configuration, same leak guard, same output schema.
- Images are **never painted**, so no artificial mask boundaries can become features.
- Condition A runs the W140 script directly.

**Validation.**
- *Extraction-only, before declaration* (8 frames per input): 0 keypoints on excluded pixels. The masked keypoint set is **exactly** the unmasked set minus the excluded ones, every frame, for walking_static, walking_xyz and synthetic. Example: walking_static frame 0 has 3,386 unmasked keypoints; 1,585 lie on people and 1,801 remain.
- *Actual runs, from each COLMAP database:* every masked run has **0** keypoints on its own excluded pixels (`kp_on_own_excluded_total = 0` for all B, C, D and oracle_areactrl runs). The camera intrinsics, timestamps and frame IDs in every `estimate.json` equal the input's.

**Mask quality:**

| | Real (walking_static / walking_xyz) | Synthetic |
|---|---|---|
| Frames with a person | 283 / 274 of 300 | — |
| Raw person occupancy, median (p90, max) | 18 % (29 %, 44 %) / 19 % (29 %, 31 %) | 5.3 % raw on average |
| Frames with < 200 keypoints | 0 | 0 |
| IoU against the oracle | not measurable (no GT masks) | median **0.19**; 46 frames with foreground but no detection |

- **Real data.** Visual inspection (agent, stills) shows masks covering the people well, including the plaid shirts that produce many features. Non-person movers, such as chairs pushed by people, are **not** masked.
- **Synthetic data.** The detector largely fails on the textured synthetic humanoids, a domain gap. Synthetic B is therefore mostly a test of *segmentation failure*, not of the exclusion mechanism; D isolates the mechanism.

## 8. Matched conditions

There are 11 declared runs, each run exactly once, in a fixed order. Within each dataset the same frames, calibration and estimator configuration are used; only the exclusion factor differs.

| Dataset | Runs |
|---|---|
| Synthetic | `syn_clean` (W140 clean frames, reference); `syn_unmasked` (A); `syn_rgbmask` (B); `syn_oracle` (D, ORACLE diagnostic, never a real-world claim); `syn_oracle_areactrl` (oracle-derived area control) |
| walking_static | A, B, C |
| walking_xyz | A, B, C |

Oracle masks were never used on real data; a test checks this. Estimator wall time totalled **47.2 min**, against a declared ceiling of 3 h.

## 9. Gates and evaluation policies

Declared before any output and unchanged after results.

**Configuration gate** (per real sequence and condition):
- coverage ≥ 80 %, with all 300 frames as the denominator;
- Sim(3) ATE ≤ 0.05 m (walking_static) or ≤ 0.15 m (walking_xyz);
- median rotation error ≤ 5°;
- ATE **and** median rotation error both below the stationary best-constant-pose baseline. This was added prospectively because walking_static's GT extent makes the ATE target non-discriminating.

**Masking-benefit rule:**
- **B improves on A** if either:
  - B reaches the coverage floor where A does not; or
  - on the A∩B∩C common registered frames (n ≥ 30, each condition re-aligned), ATE or median rotation is lower by ≥ 20 % **and** by ≥ 5 mm or ≥ 0.5°.
- **No coverage loss:** B coverage ≥ A − 0.10.
- **C distinct:** C does not improve on A by the same rule, or B beats C by it.
- **Verdicts:** DEMONSTRATED / NOT_ATTRIBUTABLE_TO_HUMAN_EXCLUSION / HARMFUL / UNNECESSARY (A passes) / NOT_DEMONSTRATED.

**Synthetic DISTURBED rule.** A condition is DISTURBED if any of these holds:
- ATE_rel > 2× clean **and** > clean + 0.2 % of extent;
- median rotation > clean + 0.5°;
- coverage < clean − 0.10;
- the W140 stationary-interval test fails.

**Alignment and observability:**
- **Benchmark alignment:** full-trajectory Umeyama Sim(3), evaluation only.
- **Early-reference alignment:** first 20 % of frames by time. It is OBSERVABLE only if n ≥ 10, the GT RMS spread σ1 ≥ 0.05 m and the scale > 0. Rotation then comes from orientations (chordal) and scale/translation from positions. Otherwise it is ROTATION_ONLY: orientation-based rotation only, with scale and translation marked UNOBSERVABLE and never repaired with later GT.
- W140's position-only early protocol is computed for continuity and **not** interpreted as drift.
- **Spikes:** consecutive relative pose error > max(5 × median, 2 cm / 2°).

## 10. Synthetic results (controlled causal test)

GT extent 1.71 m; 140 frames; 138 associated. Every run registered 140/140 in a single model.

| Run | ATE RMSE | Median rotation | Stationary 0–19 spurious path | Rotation-segment spurious translation | Mixed-segment ATE | Model observations on movers (oracle) | Landmarks majority-on-movers | Landmarks within 2 cm / beyond 10 cm of static boxes | DISTURBED |
|---|---|---|---|---|---|---|---|---|---|
| clean (W140 frames) | **0.55 mm** | 0.048° | 0.1 mm | 16.5 mm | 0.4 mm | — | — | 93 % / 1.8 % | — |
| A unmasked | **9.6 mm** (17×) | 0.160° | 13.6 mm | 64.5 mm | 23.1 mm | 3.2 % | 21 % | 75 % / 22 % | **yes** |
| B RGB mask | 6.9 mm | 0.234° | 0.8 mm | 101 mm | 11.4 mm | 2.7 % | 20 % | 45 % / 22 % | **yes** |
| D ORACLE mask | **0.64 mm** | 0.073° | 1.5 mm | 17.8 mm | 0.6 mm | 0.0 % | 0 % | 93 % / 1.6 % | **no** |
| oracle area control | **14.9 mm** | **0.997°** | 0.0 mm | 31.1 mm | 23.4 mm | 9.6 % | 37 % | 49 % / 38 % | **yes** |

**Findings:**
- **Moving foreground causally disturbs the frozen estimator.** It is the only factor changed between clean and A.
  - ATE rises 17×: still only 0.56 % of the extent, but 23 mm in the final mixed segment.
  - The model incorporates mover features: 21 % of its landmarks are anchored on the occluders.
  - The stationary interval acquires 13.6 mm of spurious camera path while the people walk. It stays within W140's stationary gate (0.25 % of extent against a 1 % limit), so this is real but small.
- **Exact exclusion of the movers prevents the disturbance** (D ≈ clean on every metric).
- **Removing a comparable static area makes it worse** (oracle area control: 14.9 mm, 1.0°, mover share of observations up to 9.6 %).
  - The area control removed *fewer* keypoints than D (17 % vs 20.5 % of all keypoints, POST-HOC).
  - So the D benefit comes from removing the movers, not from removing area or features. Removing static evidence raises the movers' relative weight.
- **RGB masks (B) reduce but do not prevent the disturbance.** This tracks the segmentation recall failure on synthetic humanoids (§7).
  - B's stationary interval is clean (0.8 mm), but residual mover observations remain.
  - The rotation-segment spurious translation is the worst of all runs (101 mm).
- **Landmark-to-static-box distance is alignment-sensitive** and is reported as a diagnostic only.

## 11. walking_static results

**Overview** (300 selected frames):

| | A unmasked | B RGB exclusion | C area control |
|---|---|---|---|
| ESTIMATED / LOST / INVALID | 75 / 225 / 0 | 23 / 277 / 0 | 103 / 161 / 36 |
| Models (registered) | 1 (75) | 1 (23) | 3 (7, **103**, 36) |
| Landmarks / mean track length | 377 / 10.6 | **18 / 2.6** | 256 / 12.0 |
| Model observations on RGB person mask | **88.9 %** | 0 % | **81.2 %** |
| Landmarks majority-on-person | **87.5 %** | 0 % | 78.5 % |
| Sim(3) scale (evaluation-only) | 2.4e-5 | 1.5e-4 | 6.4e-5 |
| Keypoints / frame, median | 3,570 | 2,889 | 2,964 |

**Registered intervals:**
- A: frames 0–181 with gaps, plus isolated frames 385 and 702;
- B: frames 35–139, fragmented;
- C: frames 0–323.

Lost intervals with timestamps are in `evaluation.json`.

**What happened:**

- **A did not reconstruct the office. It reconstructed the people.**
  - Almost all of its geometry is anchored on moving humans.
  - Because the camera barely moves, the only strong parallax in the images comes from the walking people. Incremental SfM initialised on them, and the resulting "camera" poses are relative to the people.
  - This is the camera-from-human-motion contamination the batch was designed to detect.
- **B removed that contamination** (0 % person observations), and SfM then had almost nothing to work with:
  - 18 landmarks with tracks of 2.6 views, and a degenerate reprojection error of 3e-6 px;
  - registration fell to 7.7 %, even though 2,889 keypoints per frame remain;
  - the near-stationary camera gives too little baseline for triangulating the static background: an **observability limit of SfM with a near-stationary camera**, not a feature-count problem.
- **C** removed static area while keeping the people. It registered more frames (34 %, plus 36 frames in a secondary model, INVALID), but 81 % of its observations are again on people.

**Sim(3) and baselines.**
- Full-trajectory Sim(3) "ATE" values of 14–25 mm are **not evidence of accuracy**. The scale collapses toward zero (≈1e-5 against 0.06 on walking_xyz), so every estimate is mapped to roughly the GT mean, and ATE ≈ the stationary baseline (21.3 / 14.1 / 25.4 mm).
- No condition beats the baseline on both ATE and rotation.

**The camera's actual small motion is not recovered.** In `walking_static_trajectories_errors.png`, GT moves by up to 7 cm, while the aligned estimates stay flat or follow unrelated paths.

## 12. walking_xyz results

**Overview** (300 frames, all ESTIMATED in every condition):

| | A unmasked | B RGB exclusion | C area control |
|---|---|---|---|
| ATE RMSE / p90 | 14.9 / 22.6 mm | **12.9 / 19.1 mm** | 15.1 / 21.9 mm |
| Rotation median / p90 | 0.90° / 1.15° | **0.64° / 0.91°** | 0.90° / 1.14° |
| RPE 1 s, translation RMSE / rotation median | 20.2 mm / 0.38° | **17.5 mm** / 0.37° | 20.4 mm / 0.39° |
| Early-reference (OBSERVABLE, pose-aware), rest ATE / rotation | 17.1 mm / 0.29° | **14.4 mm** / 0.29° | 16.8 mm / 0.30° |
| Drift slope (full Sim(3)), position / rotation | −0.05 mm/s / −0.006°/s | −0.35 mm/s / −0.006°/s | −0.29 mm/s / −0.005°/s |
| GT-referenced spikes | 1 (frames 528→531, t = …331.49, 45 mm) | 1 (same pair) | 2 (430→433, 528→531) |
| Model observations on person / landmarks majority-on-person | 4.4 % / 20.7 % | 0 / 0 | 7.5 % / 27.2 % |
| Landmarks; mean track length; consecutive inliers median (min) | 11,598; 27.8; 1,061 (126) | 8,872; 33.1; 902 (123) | 9,415; 20.7; 744 (57) |
| Sim(3) scale | 0.0631 | 0.0615 | 0.0605 |
| Path length, estimated (scaled) / GT | 7.22 / 6.52 m | 7.40 / 6.52 m | 7.42 / 6.52 m |

- Stationary baseline: 300.7 mm / 5.75°. All three conditions beat it by about 20× in translation and 6–9× in rotation.
- The common intersection (299 frames) gives the same picture.
- **No drift:** slopes ≈ 0 over 29 s, consistent with global bundle adjustment.

**Masking effect** (common frames):
- B lowers ATE by 2.0 mm (−14 %) and median rotation by 0.26° (−28 %). C changes nothing (+0.2 mm, −0.003°).
- B's direction is therefore attributable to removing people, not area. Its size fails the declared materiality margin (20 % **and** 5 mm, or 20 % **and** 0.5°), so it is **not counted as a demonstrated benefit**.
- **Occupancy dependence** (declared Spearman test): per-frame position-error improvement (A − B) correlates with person occupancy, ρ = +0.32, p = 2e-8, n = 299.
- POST-HOC quartile split:

  | Occupancy quartile | Position error A / B / C | Median rotation A / B |
  |---|---|---|
  | top | 13.0 / 8.3 / 13.3 mm | 0.80° / 0.53° |
  | bottom | 14.6 / 14.5 / 14.8 mm | 0.91° / 0.65° |

- So the translation benefit is concentrated where people occupy the view. The rotation benefit is uniform, which suggests a global structure effect rather than per-frame contamination.

**Observed contamination event.** The last 1.5 s of the sequence, frames 844–858, t = 1341846341.98–342.45, about 24 % occupancy:

| Frames | A | B | C |
|---|---|---|---|
| 844–858 | 26 → **46 mm** | 3–5 mm | 8–10 mm |
| 800–858 (21 frames), RMS | 25.5 mm | 4.5 mm | 7.9 mm |

- About 7 % of A's observations there lie on people.
- Errors are per condition under its own Sim(3), so part of the difference may be alignment.

## 13. Alignment and observability

**walking_xyz.**
- The early interval (first 20 %) has GT RMS spread σ = [0.23, 0.011, 0.006] m: close to rank 1, as in W140.
- The orientation-based early-reference alignment is OBSERVABLE and gives rest errors of 14–17 mm / 0.29°, close to the full-Sim(3) figures. Drift ≈ 0.
- The W140 position-only early protocol is not used for interpretation.

**walking_static.**
- Early-interval GT spread σ1 = 3.7–13.5 mm (< 50 mm), so every condition is **ROTATION_ONLY**: scale and translation UNOBSERVABLE in the early interval, by declaration.
- The full-Sim(3) rotation errors (58°, 174°, 150°) come from **alignment ill-conditioning**. A position-only Sim(3) on a near-stationary, scale-collapsed trajectory leaves the rotation essentially arbitrary. They must not be read as camera orientation errors.
- With the orientation-based early-reference rotation instead (rest frames):

  | | Rest median rotation error | Slope | Stationary baseline (rotation) |
  |---|---|---|---|
  | A | 5.81° | 0.43°/s | 1.85° |
  | B | 1.78° | — | 1.63° |
  | C | 1.11° | — | 2.59° |

  Even under orientation-aware evaluation, **no condition recovers walking_static orientation better than a constant pose** (C is marginal and covers 34 % of frames, mostly person-anchored).

**Not recovered in any sequence:**
- metric scale (all scales are evaluation-only gauge fits);
- the GT world origin and orientation;
- gravity.

The 1.4–1.5° residual rotation offset seen in W140 was not re-investigated.

## 14. Positive and negative mechanisms

**Positive:**
1. The frozen W140 estimator reconstructs a moving camera accurately with two people in view, occupying ~19 % of the image on median and up to 31 %: 100 % coverage, about 1.5 cm / 0.9°, no drift.
2. COLMAP's mask mechanism excludes person pixels exactly, without painting, and is verified on every frame.
3. In the synthetic test, exact foreground exclusion fully removes a causally induced disturbance, and the area control shows the effect is not area or feature reduction.
4. On walking_xyz, B is better than A on every reported accuracy metric, in an occupancy-dependent way that C does not reproduce. The gain is small.

**Negative:**
1. With a near-stationary camera and moving people, unmasked SfM anchors on the people (88 % of landmarks) and loses 75 % of frames.
2. Excluding people does not help there: the static background then lacks parallax and SfM collapses (7.7 % coverage).
3. RGB segmentation can fail out of domain: synthetic recall 0.19. A partial mask leaves partial contamination and can make some intervals worse (B's 101 mm rotation-segment spurious translation).
4. Non-person movers are not masked.

## 15. Causal interpretation and alternative explanations

**walking_static failure, A.** Caused by **human-anchored geometry**, supported by:
- 89 % of observations and 88 % of landmarks on person masks;
- the synthetic analogue (movers inject spurious camera motion).

Alternatives considered:
- *Calibration error:* unlikely. fr3 intrinsics were verified, and walking_xyz with the same calibration succeeds.
- *Motion blur:* not dominant. Keypoint counts stay above 2,800 per frame, and lost frames are not limited to high-occupancy ones. Registered frames had a *higher* median occupancy (23 %) than lost ones (17 %, POST-HOC), consistent with registrations that depend on people.

**walking_static failure, B.** Caused by **insufficient static parallax**:
- the near-stationary camera has a 1.3 mm median step and 1–2 cm RMS spread against office depths of metres;
- incremental SfM's initialisation and triangulation need baseline.

Alternatives considered:
- *Over-masking leaves too few features:* refuted, since 2,889 keypoints per frame and 0 low-feature frames remain.
- *Mask errors leaving residual person features:* not observed, with 0 person observations in the model.

What is untested: whether a rotation-only or homography-based estimator, or any other estimator, could recover the camera's small rotation here. W141 tested only the frozen SfM.

**walking_xyz small B gain.** Attributed to removing person features because:
- C ≈ A;
- the gain correlates with occupancy;
- a contamination event is localised at frames 844–858.

Alternative: run-to-run SfM variability. COLMAP is not bit-reproducible and each run was executed once, so a 2 mm difference is within plausible rerun noise. This is one more reason it is not counted as demonstrated.

**Synthetic.** The causal chain (movers → spurious motion; exact exclusion → none; area removal → worse) is controlled: the background, the camera and everything except the exclusion factor are identical. Limits:
- idealised rendering (flat shading, no blur or noise);
- blocky humanoids, which is why the RGB detector fails and synthetic B says little about real masks.

## 16. Representative visual evidence (not reviewed by a human)

Files are in `results/visualize/` and copied to `user_eval/W141_dynamic_camera_pose/` (Korean viewing guide inside).

**Figures:**
- `walking_{static,xyz}_masks_features.png`: RGB with B masks (red) and C masks (blue). A's keypoints are shown kept (green) or excluded, with frame ID, timestamp, occupancy and A/B/C status. Rows are 4 uniform frames plus 2 highest-occupancy frames.
- `walking_*_high_occlusion_model_features.png`: the 4 highest-occupancy frames, with A and B model-observed features (yellow = on a person). On walking_static these frames (424–454, 40–45 % occupancy) are LOST in both A and B.
- `walking_*_trajectories_errors.png`: GT against aligned A, B and C in top and side views; position and rotation error against time, with occupancy shading and spike markers; distance from the mean (walking_static shows the GT's real 0–7 cm motion against flat estimates).
- `synthetic_comparison.png`, `synthetic_masks_features.png`: clean, dynamic, masked and oracle comparisons; oracle against RGB masks; mover keypoints.

**Videos:**
- `walking_{static,xyz}_playback_realtime.mp4`: RGB with person mask and B features beside a top view of GT, A, B and C, with per-condition ESTIMATED / TRACKING LOST. **Real-time pacing**: each selected frame is held for its true inter-frame interval (25.0 s output for a 24.9 s sequence, 29.0 s for 28.9 s). Not 30 fps original pacing.
- `walking_{static,xyz}_{A,B}_viewport3d_realtime.mp4` and `_stills.png`: an external third-person 3D perspective viewport with a horizontal grid, GT (white) and aligned estimate (orange) frustums and paths, landmarks, and the RGB inset, rendered by the unchanged `scripts/w140_viewport_blender.py`.

**Noteworthy frames:**
- walking_xyz frames 844–858 (t ≈ 1341846341.98–342.45): A contamination excursion to 46 mm, not present in B.
- walking_xyz frames 528→531 (t = 1341846331.49): a 45 mm spike in all conditions, so it is not person-related.
- walking_static frames 181→385: A's registration jump (12.7° spike).
- walking_static: B registers only frames 35–139.

## 17. Runtime, storage, reproduction

| Item | Time |
|---|---|
| Estimator, 11 runs | 47.2 min (declared ceiling 3 h) |
| `syn_unmasked` (longest) | 553 s |
| `walking_static_B` (shortest) | 90 s |
| Segmentation | 7.6 min for 740 frames, CPU (ceiling 1 h) |
| Synthetic render | a few minutes |
| Visualize stage, including four 3D viewport renders | about 3 min |

**Storage.** Scratch total 6.2 GB, against a declared ceiling of 12 GB:
- data 1.9 GB;
- results 4.1 GB;
- model 0.18 GB.

**Evidence manifest.** `results/evidence_manifest.json` (7,967 files), `df19ba5480786ad95553a1af8849e5e9791da1ff550efca33955da22a19178a5`.

| File | SHA-256 |
|---|---|
| preflight | `9f98acec…` |
| declaration | `6dd621bb…` |
| amendment 1 | `2d7f8c39…` |
| estimate_result | `ebce5a7d…` |
| evaluation | `c370522e…` |
| visualize | `62cff7d2…` |
| posthoc | `4fa82592…` (POST-HOC descriptive, never gating) |

**Reproduction** (Git Bash, at `5e69612`, with data, model and environments as in §4 and §7):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w141/results; H=scripts/w141_dynamic_camera_harness.py
$PY -m pytest -q tests/test_w141_dynamic_camera_pose.py tests/test_experimental_camera_pose.py
$PY -X utf8 $H preflight --out-dir $R/preflight          # renders data/synthetic_dynamic if absent; runs segmentation + extraction-only checks
$PY -X utf8 $H declare   --out-dir $R/declare --preflight $R/preflight/preflight.json
D="--declaration $R/declare/declaration.json --declaration-sha <sha>"
$PY -X utf8 $H estimate  --out-dir $R/estimate --preflight $R/preflight/preflight.json $D
$PY -X utf8 $H amend     --out-dir $R/amend1 $D
A="--amendment $R/amend1/amendment1.json --amendment-sha <sha>"
$PY -X utf8 $H evaluate  --out-dir $R/evaluate --preflight $R/preflight/preflight.json $D $A --estimate-dir $R/estimate
$PY -X utf8 $H visualize --out-dir $R/visualize --preflight $R/preflight/preflight.json $D $A --estimate-dir $R/estimate --eval-dir $R/evaluate
```

Multi-threaded COLMAP mapping is seeded but not bit-identical across reruns.

## 18. Isolation

- `git diff 248beeb..HEAD` adds only the W141 files listed in §2 (plus the concurrent W140 viewport commit `716a85b`).
- No existing file was modified. 0 diff lines in the W140 estimator, module and tests, `scripts/batch_automation.py` and `docs/next_batch.md`.
- No Production module imports either experimental camera module.
- Not touched: FramePose, Root Motion, MotionGraph, S1 calibration, Blender Actions and pins, and the Dual-Stage Generator.
- No AnimCV animation output was produced.

## 19. Tests (RUN / PASS / FAIL / NOT RUN)

| Item | Status |
|---|---|
| `tests/test_w141_dynamic_camera_pose.py`: 30 focused tests | **30 PASS** |
| `tests/test_experimental_camera_pose.py` (W140 geometry, reused unchanged): 22 | **22 PASS** |
| Extraction-only feature-exclusion contract, 3 inputs × 8 frames, real pycolmap | **PASS** |
| Per-run keypoints on own excluded pixels (all 6 masked runs) | **0, PASS** |
| Synthetic background identity against W140 | **PASS** |
| W140 baseline identities | **PASS** |
| 11 estimator runs | **RUN** (rc 0 ×11) |
| Configuration gates | walking_xyz A/B/C **PASS**; walking_static A/B/C **FAIL** |
| Production regression | **NOT RUN** (Production untouched) |
| Human review of videos and figures | **NOT RUN** |
| Real-data mask IoU against GT masks | **NOT RUN** (no GT person masks in TUM) |
| Rerun variability; other estimators; other sequences | **NOT RUN** (not authorised) |

The 30 W141 tests cover:
- the COLMAP corner-origin mask coordinate convention; disk dilation radius; the keep-mask convention;
- area control: disjointness, area match, determinism, empty and heavy-occlusion cases with shortfall; mask quality;
- simulated feature exclusion and static preservation; invalid masks refused (missing, extra GT file, wrong shape, non-binary, colour);
- the W140 leak guard still applied;
- COLMAP database and `images.txt` readers;
- fr3 calibration identity and the absence of fr1; reproducible selection; declared runs matched (A/B/C share inputs, no oracle on real data);
- exact recovery and the coverage denominator with LOST frames; lost intervals; spike detection; path length;
- the early-reference ROTATION_ONLY case; pose-aware alignment; common-subset evaluation; scale ambiguity;
- GT duplicate-timestamp removal (and out-of-order data still refused);
- configuration gate, baseline requirement, every masking-benefit verdict, synthetic disturbance and judgment mapping;
- a two-view epipolar toy check that moving-object features violate static geometry.

## 20. Distinctions and remaining uncertainty

**Camera reconstruction capability.**
- Demonstrated, offline, for a moving camera with independently moving people: walking_xyz, about 1.5 cm / 0.9°, 100 % coverage, under evaluation-only Sim(3).
- **Not** demonstrated for a near-stationary camera with moving people (walking_static): every tested condition fails.

**Improvement caused by human exclusion.**
- Synthetic: causally shown with exact masks.
- Real (walking_xyz): small, consistent and occupancy-linked, below the declared materiality threshold, from a single run per condition.
- Real (walking_static): no rescue.
- Synthetic with RGB masks: limited by detector recall.

**Offline evaluation only.**
- COLMAP is global SfM, not causal tracking.
- Scale and world frame are unobservable.
- Every accuracy figure is after evaluation-only alignment.

**Future AnimCV applicability.** AnimCV videos may resemble either sequence: a moving camera, or a near-static or tripod camera with moving people. W141 measured both regimes and found them to behave very differently. It makes no claim about authentic AnimCV footage.

**Production readiness.** None. Nothing is integrated.

**Untested:**
- non-person dynamic objects;
- rerun variability;
- other estimators, including rotation-only methods for static cameras;
- longer sequences;
- human perceptual review.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 141
