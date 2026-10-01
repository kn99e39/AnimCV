# Worklog — FramePose Chain-Direction Attribution (2026-10-01)

## Starting state

- **Start HEAD:** fetched `origin/arch/single_frame_first` before editing. Local and remote were both `9e1ed77`, the last Worklog 62 doc commit.
- **Earlier commit `bec7eaf`:** added the Worklog 62 Blender review render scripts (`export_framepose_ik_gt_cases.py`, `review_framepose_ik_gt_attribution_blender.py`). This was separate, already-accepted review work.
- **This batch:**
  - `4f86454`: evaluation module, replay script and tests.
  - `616a409`: marks rows that have no legitimate oracle 2D, and adds the GT camera check.
- **LabServer63** was fast-forwarded to each commit before running.

This batch is diagnostic only. The following are unchanged:
- frozen H0
- Worklog 57 AnimationSemantics
- Worklog 59/60 FK
- Worklog 61 IK
- Worklog 62 attribution
- Root Orientation, Blender rest, BaseRig
- every FramePose training or inference module

## Labels, population, controls

- **H0 / runtime observable:** frozen H0. These are the predictions in `context_h0_v2`, produced from `sign_attr_v2/O_BILATERAL/checkpoint.pt` on bank `bank_3dpw_paired_v2` (digest `75519e63…`). Their split SHAs match the docs/57 lineage. The input is `benchmark_detector_observation`, i.e. 3DPW's shipped OpenPose `poses2d`. Note that H0 is not a pure 2D→3D model: as historically trained and replayed, it also receives the two oracle bilateral sign fields (`shoulder_forward_depth`, `hip_forward_depth`). This batch keeps them unchanged.
- **GT / evaluation oracle:** 3DPW `jointPositions`, root-relative in the canonical camera frame. These are the same values used in Worklog 62.
- **`oracle_geometry` 2D:** the same joints projected with the official `cam_poses` and `cam_intrinsics`, then mapped onto the bank's 17-joint convention:
  - direct SMPL shoulders, elbows, wrists, hips, knees and ankles
  - head = SMPL head (15); the detector uses the nose, which SMPL lacks
  - neck, pelvis and spine as midpoints, and thorax = neck
- **Leakage:** nothing is written back to the bank, H0, semantics, FK or IK. `src/pose/framepose_chain_direction_attribution.py` is labelled `EVALUATION ONLY`, and a test checks that no FramePose, bridge, semantics or retarget production module imports it.

**Population.** The 395 Worklog 62 matched left-arm rows: dancing actor0 79, crosscountry actor0 133, hug actor1 183.

**Controls passed:**
- Re-running the frozen checkpoint on the detector input reproduced the stored H0 to a maximum of 7.7e-7 m.
- The camera-frame H0-vs-GT shoulder→wrist angle reproduced Worklog 62's endpoint direction error to a maximum of 5.9e-5°.

**GT camera limitation (new finding).** On every matched row of `outdoors_crosscountry_00`, the official 3DPW extrinsics put all GT joints *behind* the camera, so they cannot be projected.

| check against the detector (px, median / p95) | dancing | hug | crosscountry |
|---|---|---|---|
| official projection | 6.6 / 16.4 | 8.7 / 15.6 | impossible |
| official rotation + least-squares translation | 6.4 / 15.5 | 8.3 / 14.6 | 21.4 / 56.5 |

So:
- crosscountry has **no legitimate oracle 2D**, and its 133 rows are excluded from the 2D and sensor-substitution analyses rather than given a constructed substitute;
- its camera-frame GT rotation is also less consistent, which probably inflates some of its larger Worklog 62 errors.

Crosscountry stays in the 3D-only analyses below. Pooled results are given both with and without it.

## 1. Detector-vs-GT 2D error (262 rows with oracle 2D: dancing + hug)

| quantity | p25 | p50 | p95 | max |
|---|---:|---:|---:|---:|
| shoulder 2D error (px) | 7.6 | 10.5 | 21.0 | 50.2 |
| elbow 2D error (px) | 4.9 | 7.7 | 18.1 | 118.3 |
| wrist 2D error (px) | 3.2 | 4.8 | 18.2 | 153.7 |
| **shoulder→wrist 2D direction error** | 0.30° | **0.74°** | **3.86°** | 115.6° |
| shoulder→elbow 2D direction error | 1.25° | 2.44° | 9.67° | 74.4° |
| elbow→wrist 2D direction error | 1.80° | 3.36° | 9.88° | 113.6° |
| shoulder→wrist 2D endpoint error / chain length | 0.028 | 0.044 | 0.118 | 0.595 |

- The median chain length is 238 px.
- Part of the shoulder offset is definitional: an OpenPose shoulder is not the SMPL shoulder joint.
- Projecting GT 3D gives a perspective floor between 3D XZ and the 2D direction of 2.31° median.
- **The detector 2D shoulder→wrist direction is already accurate.** The 3D error is about 10°, an order of magnitude larger. The exception is a small failure tail, for example dancing #176, where the detector is off by 116°.

## 2. Observation-regime control (same frozen H0; only the 2D xy is swapped; 262 rows)

| input (identical validity, confidence, crop rule, signs, weights) | 3D shoulder→wrist error: mean / p50 / p95 | XZ angle p50 | \|depth elevation error\| p50 |
|---|---|---:|---:|
| benchmark detector (= H0) | 11.33 / 9.07 / 25.10° | 1.66° | 8.60° |
| oracle 2D, all joints | 10.68 / 7.77 / 25.78° | 2.47° | 6.93° |
| oracle 2D, left arm only | 11.33 / 9.31 / 23.38° | 1.90° | 8.70° |

- **Substitution closes almost nothing.** Perfect 2D for all joints leaves 94% of the mean error. Perfect left-arm 2D leaves 100%.
- **Per sequence:** all-joint oracle 2D leaves 81% of the mean error on dancing, mostly by fixing the #176 detector failure. On hug it leaves 99%.
- **The torso-convention mismatch raises the image-plane error slightly** (dancing XZ p50 2.05° → 4.07°). Any substitution gain is therefore an upper bound on what better 2D can buy.

## 3. H0 3D chain direction: image plane vs depth

| shoulder→wrist (all 395 rows) | p25 | p50 | p95 | mean |
|---|---:|---:|---:|---:|
| full 3D angle | 6.2° | 10.3° | 38.3° | 14.2° |
| XZ image-plane angle | 1.0° | 2.1° | 13.9° | 4.7° |
| H0 XZ vs **detector input** 2D direction | 1.0° | 1.9° | 10.2° | 3.1° |
| \|depth elevation error\| | 5.3° | 9.8° | 33.4° | 12.8° |
| forward-fraction error (H0 − GT) | −0.108 | +0.079 | +0.412 | +0.047 |

- **Correlation with the full 3D angle:** \|depth elevation error\| r = 0.957, XZ angle r = 0.707, detector 2D direction error r = 0.732.
- **Depth sign:** the depth sign disagrees on only 25 of 395 rows (the forward component flips with \|GT\| > sin 10°). This is not a sign problem. The error is continuous depth error with a forward bias (median forward-fraction error +0.08).
- **H0's image-plane direction follows its 2D input** (1.9° median, close to the 2.3° perspective floor).

**Component hybrids** (raw displacement components swapped before normalization; angle vs GT):

| variant | mean | p50 | p95 | mean error remaining |
|---|---:|---:|---:|---:|
| A. H0 full (X, Y, Z) | 14.19° | 10.34° | 38.31° | 100% |
| B. H0 image plane + **GT depth** (H0 X, GT Y, H0 Z) | **4.47°** | **2.29°** | 14.95° | **31.5%** |
| C. **GT image plane** + H0 depth (GT X, H0 Y, GT Z) | 12.54° | 9.13° | 32.21° | 88.4% |
| D. GT full | 0 | 0 | 0 | 0% |

Per sequence, the share of mean error remaining:

| sequence | GT depth (B) | GT image plane (C) |
|---|---:|---:|
| dancing | 46% | 82% |
| hug | 19% | 94% |
| crosscountry | 37% | 86% |
| excluding crosscountry | 27% | 91% |

## 4. Upper arm vs forearm (segment directions weighted by fixed GT lengths)

| variant (shoulder→wrist rebuilt from two directions) | mean | p50 | p95 | mean error remaining |
|---|---:|---:|---:|---:|
| A. H0 upper + H0 lower | 14.42° | 10.69° | 37.87° | 101.6% |
| B. **GT upper** + H0 lower | 9.19° | 7.52° | 25.35° | 64.8% |
| C. H0 upper + **GT lower** | 8.24° | 5.71° | 21.15° | 58.1% |
| D. GT upper + GT lower | 0 | 0 | 0 | 0% |

**Segment-wise:**

| segment | 3D direction error, median / p95 | of which XZ, median |
|---|---|---:|
| upper arm | 10.8° / 39.0° | 4.2° |
| forearm | 14.3° / 49.7° | 3.5° |

**Neither segment dominates.** Fixing either one alone leaves about 60% of the error, with the forearm slightly larger. Both segments carry mostly depth error.

**Length weights:** using GT length weights instead of H0's own lengths does not reduce the direction error (A ≈ H0 full). The Worklog 62 ratio error is therefore not what drives the direction error.

## 5. Joint coordinate accounting (pelvis-relative \|H0 − GT\|, cm; supporting evidence only)

| joint | X median / p95 | **Y depth median / p95** | Z median / p95 | 3D median |
|---|---|---|---|---:|
| left shoulder | 1.6 / 6.2 | **4.2 / 14.0** | 2.5 / 6.6 | 6.6 |
| left elbow | 2.5 / 9.1 | **6.3 / 20.2** | 2.7 / 8.3 | 8.0 |
| left wrist | 2.5 / 10.5 | **9.6 / 31.0** | 2.2 / 11.9 | 11.5 |

Depth is the largest component for every joint, and it grows along the chain. This is consistent with the hybrids, but the attribution rests on the hybrid controls, not on this table.

## 6. Owner cases (Worklog 62 cases kept; two fixed-rule cases added)

| case | row | 2D s→w error | 3D full | XZ | depth elevation error | H0 plane + GT depth | GT plane + H0 depth | GT upper / GT lower | oracle 2D all / arm |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| known_good | dancing #424 | 0.2° | 6.9° | 1.7° | −6.7° | 1.7° | 6.3° | 2.1 / 4.9° | 6.8 / 6.9° |
| tracking_loss | hug #312 | 0.4° | 10.6° | 2.5° | +10.3° | 2.2° | 10.3° | 8.6 / 2.1° | 10.7 / 11.1° |
| largest_ratio_error | hug #397 | 9.7° | 17.0° | 7.9° | −15.3° | 7.8° | 12.9° | 53.3 / 18.5° | 17.0 / 18.3° |
| largest_direction_error = largest_2d_shoulder_wrist_direction_error | dancing #176 | **115.6°** | 133.2° | 136.2° | −84.3° | 65.7° | 94.7° | 67.0 / 118.3° | **55.7** / 128.6° |
| dynamic | crosscountry #462 | n/a (no oracle 2D) | 28.4° | 11.3° | −26.4° | 9.8° | 25.8° | 12.9 / 14.7° | n/a |
| turning_review | crosscountry #211 | n/a | 51.1° | **78.4°** | +26.5° | 44.3° | 6.9° | 21.5 / 28.3° | n/a |
| near_straight | crosscountry #525 | n/a | 21.3° | 1.6° | +21.2° | 1.6° | 21.6° | 12.7 / 8.9° | n/a |
| strongly_bent = largest_reach_error = largest_depth_component_error | crosscountry #173 | n/a | 97.0° | 11.4° | **−97.0°** (sign flip) | 22.1° | 163.1° | 6.0 / 57.8° | n/a |
| turning_invalid / invalid_endpoint | crosscountry #202 / hug #406 | not matched (Worklog 61 IK UNAVAILABLE) | | | | | | | |

**Overlaps:**
- dancing #176 is both the largest 3D and the largest 2D direction error.
- crosscountry #173 is `strongly_bent`, `largest_reach_error` and `largest_depth_component_error`.

**Exceptions to the depth pattern:**
- **Dancing #176** is a genuine **observation** failure. Oracle 2D for all joints reduces the 3D error from 133° to 56°, while arm-only oracle 2D does not, so the failure involves torso keypoints.
- **Crosscountry #211** is an image-plane case (XZ 78°). It lies in the sequence whose GT camera is suspect, so it is not used as evidence of an image-plane reconstruction failure.
- **#173** is the one large depth-*sign* flip (the forearm folded the wrong way in depth).

## Architecture classification

**CASE B — RECONSTRUCTION / DEPTH DOMINATED**, not segment-specific, with a small observation-failure tail.

- **The 2D observation is accurate:** detector shoulder→wrist direction p50 0.74°, p95 3.9°.
- **H0's image-plane output follows its input:** XZ vs detector 1.9° median.
- **Better 2D barely helps:** perfect 2D leaves 94–100% of the 3D error.
- **GT depth fixes most of it:** substituting the GT depth component leaves 31.5% of the mean error (median 10.3° → 2.3°). GT image-plane components leave 88%.
- **Both arm segments contribute comparably**, and each segment's error is mostly depth.
- **Sign ambiguity is not the main mechanism:** 6% of rows show sign disagreement, and the bulk is continuous depth/elevation error with a forward bias.

**Where the dominant shoulder→wrist 3D direction error enters the pipeline:** in **FramePose's 2D→3D depth reconstruction** of the arm chain, not in the Geometry Observation Layer. The next architecture owner is therefore **FramePose depth reconstruction**, at chain level and spanning both upper arm and forearm.

The secondary items are:
- the detector-failure tail (Geometry Observation Layer, rare);
- the 3DPW crosscountry GT camera inconsistency (an evaluation-data limitation that should be resolved or excluded before that sequence is used to judge a FramePose change).

No FramePose fix is proposed or implemented here. Loss, model, depth-prior or temporal changes are left for the next decision.

## Outputs, files, tests

**Outputs:** LabServer63 `~/animcv-output/framepose_chain_direction_attribution/`:
- `report.json` (SHA-256 `15d4187d97d98a80e7bbf81dc28d453b7784679441e8520a1ed95a80d80c7d57`)
- `rows.json`, with every metric per row

These are machine-readable and stay on the server. No renders were produced in this batch.

**Files added:**
- `src/pose/framepose_chain_direction_attribution.py`
- `scripts/run_framepose_chain_direction_attribution.py`
- `tests/test_framepose_chain_direction_attribution.py`
- this worklog

No existing file was modified.

**Tests:** the new tests cover:
- detector/oracle 2D joint and direction measurement
- oracle 2D in the bank convention, with confidence and arm-only substitution kept
- projection conventions
- XZ/depth decomposition
- component hybrids substituting before normalization
- GT-length segment hybrids
- identical H0/GT giving zero error
- GT masking of non-finite or degenerate chains
- no production import leakage

Results: 9 tests. Together with the unchanged Worklog 59–62 focused tests, 42 passed. Production behaviour did not change, so the full regression was not run.
