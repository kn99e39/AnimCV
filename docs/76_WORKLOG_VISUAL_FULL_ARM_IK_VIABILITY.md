# Worklog — Full-Arm IK Pose Viability of Visual Chain Depth (2026-10-12)

**Question.** Does replacing frozen H0's shoulder→wrist direction with BEDLAM-supervised E1 improve the connected left-arm Worklog 61 IK pose (wrist, elbow, bone orientations), or does it improve the wrist at the cost of elbow placement, limb geometry or rotational stability?

**Answer.** The static pose improves; the motion gets noisier. The wrist gain from Worklog 75 survives unchanged in the full IK, and the forearm direction improves robustly. The elbow / upper-arm gain is small and not broadly shared. On adjacent frames, the upper arm becomes measurably jumpier than under H0. **CASE B.**

**Test results (6,003 rows, 37 sequences, F2 = E1 vs F0 = H0):**

| quantity | F0 → F2 | rows improved | sequences improved | all predeclared 95% intervals exclude 0? |
|---|---|---|---|---|
| wrist error (cm) | 14.88 → 11.89 (−2.99) | 60% | 32/37 | yes |
| forearm direction error (°) | 22.57 → 18.33 (−4.24) | 61% | 32/37 | yes |
| elbow error (cm) | 8.25 → 7.64 (−0.61) | 53% | 20/37 | no (sequence cluster includes 0) |
| upper-arm direction error (°) | 17.42 → 15.95 (−1.47) | 53% | 20/37 | no (sequence cluster includes 0) |
| upper-arm rotation change per adjacent frame (°) | 9.23 → 11.77 (+2.54) | — | — | yes, it gets worse |
| candidate-specific upper-arm discontinuities | 1.29% → 3.13% of adjacent pairs | — | — | — |

- **Elbow gain is concentrated.** `downtown_bar_00` actor0 supplies 59% of the net elbow gain. Without it the gain is −0.26 cm.
- **Elbow p95 improves** (21.54 → 18.98 cm), so there is no material p95 regression.
- **Nothing became unavailable or clamped.** Coverage and solve validity are identical for F0, F1 and F2 (6,003/6,003 known, 0 clamps).
- **F2 beats F1 (E0) on every static metric**, in 33–35/37 sequences, and is less jumpy than F1. The BEDLAM supervision effect is real at full-arm level.
- **The temporal regression comes from changing the direction source,** not from E1 specifically: F1 shows the same pattern, slightly worse.

## Starting state and files

- **Start:** `arch/single_frame_first` at `10ae8f2377483520ca44fa709667db36e0bdaac8` (local = remote), with a clean worktree.
- **Commits:**
  - `8f57d87`: evaluation module, runner, tests
  - `474ea8c`: Blender review export + render
  - the commit adding this worklog
- **Files added (evaluation only, not imported by production):**
  - `src/retarget/framepose_visual_full_arm_ik.py`
  - `scripts/run_framepose_visual_full_arm_ik.py`
  - `scripts/export_framepose_visual_full_arm_cases.py`
  - `scripts/review_framepose_visual_full_arm_blender.py`
  - `tests/test_framepose_visual_full_arm_ik.py`
  - this worklog
- **Unchanged:**
  - every historical module, checkpoint and artifact;
  - the Worklog 60 Root Orientation, Worklog 61 IK, Worklog 62 evaluation, Worklog 73 E0/E1, and Worklog 74/75 results;
  - frozen H0 and its oracle torso-sign contract;
  - the Blender rest snapshot, calibration and bone lengths;
  - production FramePose, FK, IK and the Blender writer.

The new module builds on the Worklog 61/62 primitives (`solve_two_bone`, `_runtime_side`, `local_rotations_for_directions`, `chain_forward_kinematics`). It does not import the Worklog 74/75 evaluation modules, because their tests forbid other `src/` files from referencing them. Its equivalence to the Worklog 74 `solve_with_direction` is proven by Gate 1 and by a unit test.

## Frozen artifact identities

**Pinned and checked at runtime:**

| artifact | sha256 |
|---|---|
| Worklog 62 `report.json` | `f7a100cd…` |
| Worklog 73 `report.json` | `1c05d50a…` |
| Worklog 73 `rows_heldout.json` | `4e17bd77…` |
| Worklog 74 `report.json` | `8158e8b2…` |
| Worklog 74 `rows.json` | `d3183e65…` |
| Worklog 75 `report.json` | `89218bab…` |
| Worklog 75 `rows_test.json` | `dc39b22b…` |

**Also checked:**
- Frozen H0 validation predictions `88a007fc…` (prefix pinned since Worklog 52).
- H0 test predictions `6df04c20…`, which must equal the hash Worklog 75 recorded.
- The H0 materialization was built from this FrameBank (`75519e63…`).
- The rest snapshot matches the current `BaseRig.fbx` bytes.

**Recorded:** Worklog 61 IK report `b06f3b82…`.

**Outputs (`~/animcv-output/framepose_visual_full_arm_ik/` on LabServer63):**

| file | sha256 |
|---|---|
| `report.json` | `dad46288…` |
| `rows_test.json` | `d6b339dc…` |
| `rows_validation.json` | `7bf236c5…` |
| `review_cases.json` | `7aef29f8…` |
| `review/report.json` | `64df34af…` |

## F0/F1/F2 full-arm contracts

All three candidates use the same left-arm solve:

    direction d   F0: H0 chain direction (rig space)
                  F1/F2: reconstruct_direction(E0/E1 chain f, detector observed plane), then A·Rz(−yaw_H0)
    reach         H0 reach fraction r_H0 (unchanged)
    bend side     H0 observed side, re-perpendicularized to d (Worklog 62 _runtime_side)
    IK            solve_two_bone(root, d, r_H0·L, L1, L2, side)   (Worklog 61 closed form, clamps)
    rotations     local_rotations_for_directions (zero-twist), FK check by chain_forward_kinematics

- **Only `d` differs between candidates.** The upper/forearm forward fractions are never used, and no skeleton is stitched together.
- **GT reference.** It is the Worklog 62 oracle arm: GT direction, GT reach and GT side, on the same root, rest lengths, alignment and yaw_H0, built by the same function.
- **Scoring.** The wrist is scored against `E_GT = root + r_GT·L·d_GT` (the Worklog 74/75 endpoint contract). Everything else is scored against the reference arm.
- **No GT in candidates.** GT never enters F0/F1/F2.

## Gate 1 — historical full-IK parity (395 Worklog 74 rows)

Gate 1 ran through the Worklog 74 path: AnimationSemantics v2, the persisted Worklog 61 IK, the FrameBank plane, 3DPW GT and Worklog 73 f. The tolerance is the historical contract: 1e-9 cm, and 1e-9 per quaternion component after sign alignment.

| check (max over 395 rows) | F0 | F1 | F2 |
|---|---:|---:|---:|
| wrist vs Worklog 74 solved endpoint (cm) | 0 | 0 | 0 |
| elbow vs Worklog 74 solved mid (cm) | 0 | 0 | 0 |
| reach / direction / side vs Worklog 74 | 0 | 0 | 0 |
| wrist error vs Worklog 74 score (cm) | 0 | 0 | 0 |
| elbow error vs Worklog 74 score (cm) | 8.9e-15 | 2.0e-14 | 9.8e-15 |
| local rotations vs rotations implied by Worklog 74 joints | 0 | 0 | 0 |
| IK→FK residual (cm) | 4.5e-14 | 3.8e-14 | 4.5e-14 |

**F0 against the persisted Worklog 61 runtime sample:**
- wrist 4.5e-14 cm, elbow 3.1e-14 cm, reach 0;
- `proximal_local` 5.6e-16 and `distal_local` 7.8e-16 (component, sign-aligned).

**Worklog 62 reference:**
- E_GT: 0 difference;
- oracle elbow: 2.9e-14 cm;
- reference wrist vs E_GT: 2.9e-14 cm.

**Status agreement.** Status, side status (all `observed`), clamp (all `none`), side-vs-GT agreement and GT side status match on every row. Observed plane and yaw_H0 match exactly.

**Test path on the same rows.** The same 395 rows were re-run through the path used for test: `assemble_h0` → `build_h0_lifted_sequence` → `current_root_orientation`. Joints, rotations and the reference were identical (max diff 0) with no status mismatch.

**Gate 1 passes.**

## Gate 2 — Worklog 75 endpoint parity (6,003 test rows)

| check (max over 6,003 rows) | F0 vs T0 | F1 vs T1 | F2 vs T2 |
|---|---:|---:|---:|
| wrist error vs Worklog 75 endpoint error (cm) | 4.3e-14 | 3.6e-14 | 4.3e-14 |
| chain direction error (°) | 3.1e-12 | 1.8e-12 | 2.5e-12 |
| camera-frame visual direction | — | 0 | 0 |

**Identical inputs.** The same 6,003 identities (set-equal, same sample ids and positions), with H0 reach to 4.4e-16, GT reach to 5.6e-16 and the same detector confidence.

**No new failures.** Clamps are identical (all `none`) and no candidate became unavailable.

**Gate 2 passes.** The full-pose implementation does not change the established endpoint result.

## Test coverage

| | rows |
|---|---:|
| FrameBank test rows | 7,076 |
| excluded: no Worklog 73 prediction | 1,073 |
| invalid H0 source (flags, all within the 1,073) | shoulder 87, elbow 633, wrist 1,059 |
| unknown root orientation / invalid GT / degenerate detector plane | 0 / 0 / 0 |
| common-valid (= Worklog 75 population) | 6,003 |
| F0 / F1 / F2 / reference known | 6,003 / 6,003 / 6,003 / 6,003 |

- All 37 test actor sequences are present; per-sequence counts are in `report.json → test.by_sequence`.
- Every H0 bend side was `observed` (no rest-plane fallback rows), so the observed-vs-fallback split has a single non-empty cell.

## Wrist, elbow and segment-direction metrics (test, common-valid)

| metric | F0 H0 mean / p50 / p95 | F1 E0 mean / p50 / p95 | F2 E1 mean / p50 / p95 |
|---|---|---|---|
| wrist error (cm) | 14.88 / 11.46 / 37.08 | 14.10 / 11.44 / 36.20 | **11.89 / 9.63 / 29.96** |
| elbow error (cm) | 8.25 / 6.09 / 21.54 | 8.72 / 7.10 / 21.59 | **7.64** / 6.11 / **18.98** |
| upper-arm direction (°) | 17.42 / 12.59 / 45.63 | 18.26 / 14.69 / 45.75 | **15.95** / 12.62 / **39.96** |
| forearm direction (°) | 22.57 / 16.44 / 67.59 | 20.31 / 15.50 / 55.40 | **18.33 / 14.27 / 48.98** |
| chain direction (°) | 18.21 / 12.81 / 52.78 | 16.78 / 12.75 / 46.57 | **13.78 / 10.57 / 37.48** |
| \|bend-angle error\| (°) | 14.26 / 10.53 / 38.55 | identical | identical |
| bend-plane error (°) | 26.68 / 19.52 / 74.60 | 27.23 / 20.41 / 73.02 | 26.56 / 19.70 / 73.93 |
| proximal local rotation vs reference (°) | 17.83 | 18.69 | 16.32 |
| distal local rotation vs reference (°) | 24.64 | 24.84 | 24.26 |

**Max values:**
- wrist: F0 90.6, F1 69.7, F2 76.3 cm;
- elbow: 49.9, 48.7, 44.2 cm.

**Bend-angle error is owned by H0 reach.** With fixed bone lengths, the elbow angle is a function of the reach fraction alone, so it is the same for all candidates by construction. A better direction cannot fix a wrong bend. H0 bends 5.5° more than GT on average, with |error| 14.3°.

**Elbow and upper-arm results move together.** Both bones share the root and L1 is fixed, so elbow error = 2·L1·sin(upper-arm angle / 2).

**Rig integrity is clean for every candidate on all 6,003 rows:**
- bone-length residual ≤ 3.9e-14 cm;
- endpoint reconstruction residual 0;
- IK→FK ≤ 7.1e-14 cm;
- quaternion norm error ≤ 2.2e-16;
- 0 non-finite, 0 unavailable, 0 clamped.

## Paired causal accounting (test, 6,003 rows, 37 sequences)

Interval columns are frame-weighted 95%:
- **SC:** sequence-cluster bootstrap;
- **MB10 / MB25:** moving-block bootstrap with block 10 / 25 frames.

All use 2,000 replicates, seed 1337, as in Worklog 75.

| pair · metric | mean | p50 | p95 | improved rows | seq improved / regressed | SC | MB10 | MB25 | largest sequence share of net |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| **F2−F0 wrist (cm)** | −2.99 | −1.78 | +12.67 | 60% | 32 / 5 | [−4.76, −1.50] | [−3.71, −2.63] | [−4.10, −2.63] | 26% (bar_00 a0) |
| **F2−F0 elbow (cm)** | −0.61 | −0.32 | +8.71 | 53% | 20 / 17 | [−1.56, +0.18] | [−0.97, −0.33] | [−1.10, −0.26] | 59% (bar_00 a0) |
| **F2−F0 upper-arm dir (°)** | −1.47 | −0.67 | +18.48 | 53% | 20 / 17 | [−3.79, +0.38] | [−2.31, −0.83] | [−2.62, −0.67] | 63% (bar_00 a0) |
| **F2−F0 forearm dir (°)** | −4.24 | −2.59 | +14.60 | 61% | 32 / 5 | [−6.50, −2.40] | [−5.18, −3.85] | [−5.67, −3.96] | 24% (bar_00 a0) |
| F2−F1 wrist (cm) | −2.21 | −1.40 | +7.90 | 62% | 33 / 4 | [−2.90, −1.59] | [−2.58, −1.98] | [−2.71, −1.94] | 15% |
| F2−F1 elbow (cm) | −1.08 | −0.80 | +4.85 | 62% | 35 / 2 | [−1.39, −0.78] | [−1.28, −0.93] | [−1.34, −0.90] | 12% |
| F2−F1 upper-arm dir (°) | −2.32 | −1.68 | +10.22 | 62% | 35 / 2 | [−2.99, −1.66] | [−2.74, −1.98] | [−2.88, −1.92] | 13% |
| F2−F1 forearm dir (°) | −1.98 | −1.07 | +10.42 | 58% | 30 / 7 | [−2.80, −1.23] | [−2.48, −1.76] | [−2.66, −1.76] | 18% |
| F1−F0 wrist (cm) | −0.78 | −0.35 | +15.28 | 52% | 25 / 12 | [−2.19, +0.55] | [−1.36, −0.39] | [−1.64, −0.45] | 59% |
| F1−F0 elbow (cm) | +0.47 | +0.50 | +10.08 | 45% | 9 / 28 | [−0.37, +1.17] | [+0.17, +0.73] | [+0.09, +0.78] | — (net regression) |
| F1−F0 upper-arm dir (°) | +0.84 | +1.04 | +21.43 | 45% | 9 / 28 | [−1.18, +2.48] | [+0.17, +1.42] | [−0.05, +1.55] | — |
| F1−F0 forearm dir (°) | −2.26 | −1.35 | +18.01 | 56% | 25 / 12 | [−4.07, −0.65] | [−3.06, −1.74] | [−3.39, −1.79] | 28% |

**Sequence-weighted means (F2−F0):** wrist −2.79, elbow −0.56, upper −1.31°, forearm −3.91°. Their sequence-cluster intervals for elbow and upper-arm also include 0: [−1.33, +0.10] and [−3.11, +0.18].

**Without the largest contributor (`downtown_bar_00` a0), F2−F0:**

| metric | change |
|---|---:|
| wrist | −2.30 cm |
| forearm | −3.38° |
| elbow | −0.26 cm |
| upper-arm | −0.56° |

**Worst F2−F0 sequence regressions (mean per row):**
- elbow: `flat_packBags_00` a0 +2.76 cm, `downtown_stairs_00` a0 +2.65, `downtown_walking_00` a0 +1.55;
- wrist: `downtown_stairs_00` a0 +5.25 cm, `flat_packBags_00` a0 +2.33, `downtown_upstairs_00` a0 +1.34.

**Worst single rows:**
- `downtown_sitOnStairs_00` a0 #325: wrist +62.8 cm, elbow +37.1 cm, upper-arm +88.6°;
- `downtown_bar_00` a0 #319: elbow +33.2 cm.

**Interpretation:**
- **F2 vs F1 (BEDLAM effect).** Every joint and segment improves, in 30–35/37 sequences, with every interval excluding zero.
- **F2 vs F0 (practical replacement of H0).** Wrist and forearm improve broadly. Elbow and upper-arm improve only on average: 17/37 sequences regress, and the sequence-cluster intervals include zero.
- **F1 vs F0 (E0 visual direction without BEDLAM).** The elbow worsens, regressing in 28/37 sequences.

## Endpoint–elbow coupling

| F2 vs F0 | rows | share | mean Δwrist (cm) | mean Δelbow (cm) |
|---|---:|---:|---:|---:|
| wrist improves / elbow improves | 2,644 | 44.0% | −9.94 | −5.18 |
| wrist improves / elbow worsens | 984 | 16.4% | −6.45 | +3.56 |
| wrist worsens / elbow improves | 554 | 9.2% | +3.81 | −2.28 |
| wrist worsens / elbow worsens | 1,821 | 30.3% | +6.90 | +4.27 |

| F2 vs F1 | rows | share | mean Δwrist | mean Δelbow |
|---|---:|---:|---:|---:|
| both improve | 3,067 | 51.1% | −6.35 | −3.50 |
| wrist improves / elbow worsens | 660 | 11.0% | −3.86 | +2.41 |
| wrist worsens / elbow improves | 654 | 10.9% | +3.27 | −1.91 |
| both worsen | 1,622 | 27.0% | +4.08 | +2.41 |

**Joints mostly move together.** The row correlation of Δwrist and Δelbow is 0.72 (F2−F0) and 0.70 (F2−F1). 74% of F2−F0 rows fall on the diagonal, so there is no systematic wrist-for-elbow trade-off.

**The off-diagonal is asymmetric.** "Wrist improves but elbow worsens" (16.4%) is about twice "wrist worsens but elbow improves" (9.2%). That is the population the review case `wrist_improves_elbow_regresses` illustrates.

**By GT bend regime (F2 vs F0 coupling counts):**

| regime | improve / improve | wrist improves / elbow worsens | wrist worsens / elbow improves | worsen / worsen |
|---|---:|---:|---:|---:|
| near-straight | 336 | 136 | 160 | 299 |
| moderate | 1,291 | 397 | 232 | 796 |
| strong | 716 | 263 | 135 | 492 |
| near-folded | 301 | **188** | 27 | 234 |

**Near-folded is where the trade-off lives:** 188 rows where the wrist gains and the elbow loses, against 27 the other way. The per-sequence F2−F0 coupling counts are in `report.json → test.by_sequence`.

## Bend-regime accounting

GT bend bins follow Worklogs 74/75: near-straight [0, 15)°, moderate [15, 45)°, strong [45, 90)°, near-folded ≥ 90°.

| GT regime (n, sequences) | F2−F0 wrist | F2−F0 elbow | F2−F0 upper | F2−F0 forearm | F2 vs F0 bend-plane change | side flips F2 vs F0 |
|---|---|---|---|---|---:|---:|
| near-straight (931, 37) | −0.37 (24/37) | −0.12 (21/37) | −0.27° | −0.03° | 4.2° | 0 |
| moderate (2,716, 37) | −3.40 (31/37) | **−1.19 (26/37)** | −2.86° | −4.26° | 7.8° | 0 |
| strong (1,606, 37) | −2.90 (30/37) | −0.44 (20/37) | −0.96° | −3.79° | 12.3° | 0 |
| near-folded (750, 31) | −4.96 (23/31) | **+0.47 (11/31)** | +0.97° | **−10.36°** | 19.6° | 0 |

- **Bend-plane orientation.** E1 rotates the bend plane more as the arm bends, by up to 19.6° against F0 in near-folded rows. It never flips the elbow side relative to F0: the side stays H0-owned, re-perpendicularized.
- **Agreement with GT barely moves.** Bend-plane error vs GT is 26.68° (F0) vs 26.56° (F2), and side agreement is 5,802 (F0) vs 5,843 (F2) of 6,003. F2 has fewer flips than F0: 160 vs 201.
- **Near-folded.** The forearm direction gains strongly (−10.4°) while the elbow regresses (+0.47 cm, 11/31 sequences improved).
  - With a near-folded arm and H0's reach, the elbow position depends mostly on the bend side and reach, which E1 does not own.
  - Pointing the chain at the right wrist then swings the upper arm into a wrong place.
- **Near-straight.** Almost nothing changes, since the chain direction already defines both bones.
- **By H0 bend (instead of GT):** the same pattern. Near-folded H0 rows show elbow +0.27 cm and forearm −6.92°.
- **By GT out-of-plane-ness \|f_GT\|:** gains concentrate in out-of-plane rows (\|f\| ≥ 0.5), with wrist −7.07 cm, elbow −2.78 cm and upper-arm −6.61°. Near-plane rows (\|f\| < sin 10°) have a slight elbow regression (+0.24 cm).

## Rotation and temporal stability

**Contract (predeclared in the runner before any test candidate statistic; independent of F2):**
- **Adjacent pair:** consecutive FrameBank samples of one sequence exactly one bank stride apart (5 source frames = 1/6 s), both common-valid. This gives 5,504 pairs.
- **Jump threshold:** per metric, the 99th percentile of the GT-anchored reference arm's own change on the same pairs:

| metric | threshold |
|---|---:|
| proximal rotation | 34.6° |
| distal rotation | 49.1° |
| wrist | 31.9 cm |
| elbow | 16.0 cm |

- **Candidate-specific discontinuity:** the candidate exceeds the threshold while the reference does not. A pair where the reference also exceeds it is "true rapid motion" and is never counted as a candidate error.
- **Distance measure:** quaternion distance is the Worklog 61 sign-invariant angle.

*Implementation note.* The first run counted consecutive bank samples across bank holes (171 pairs with source-frame gaps of 6–363) as adjacent. That contradicted the declared adjacency text, so the code was corrected to enforce the stated one-stride rule before the final run. The dropped pairs are listed in the report. Conclusions are unchanged (first-run F2/F0 proximal discontinuity rate 3.07% / 1.36%; final 3.13% / 1.29%).

**Mean change per adjacent pair** (reference = true motion of the GT-anchored arm):

| metric | reference | F0 | F1 | F2 | F2−F0 [SC / MB10 / MB25] | F2−F1 |
|---|---:|---:|---:|---:|---|---:|
| proximal (upper-arm) local rotation (°) | 7.04 | 9.23 | 12.39 | 11.77 | **+2.54** [2.10, 3.00] / [2.31, 2.87] / [2.40, 2.95] | −0.61 |
| distal (forearm) local rotation (°) | 10.14 | 16.16 | 16.91 | 16.72 | +0.56 [0.26, 0.93] | −0.19 |
| elbow displacement (cm) | 3.32 | 4.35 | 5.82 | 5.55 | **+1.19** [0.99, 1.41] | −0.28 |
| wrist displacement (cm) | 7.01 | 8.35 | 9.89 | 9.30 | +0.95 [0.62, 1.31] | −0.59 |

**Discontinuity and jump rates per adjacent pair:**

| metric | F0 | F1 | F2 |
|---|---:|---:|---:|
| proximal rotation | 1.29% | 3.36% | **3.13%** |
| elbow displacement | 1.34% | 3.42% | **3.22%** |
| wrist displacement | 0.91% | 1.56% | 1.44% |
| distal rotation | 3.32% | 3.56% | 3.60% |

- True rapid-motion pairs (reference above threshold) are about 56 per metric for every candidate, and are excluded from these rates.
- The extra motion is error, not better motion tracking. The mean absolute difference from the reference's own change grows from F0 to F2: proximal 4.98° → 6.76°, elbow 2.33 → 3.17 cm.

**The extra jitter is in the upper arm (attribution metrics):**

| per adjacent pair (°) | reference | F0 | F1 | F2 | F2−F0 [SC] |
|---|---:|---:|---:|---:|---|
| chain direction change | 8.02 | 9.72 | 11.72 | 10.94 | +1.23 [0.76, 1.71] |
| upper-arm direction change | 6.88 | 9.03 | 12.12 | 11.54 | **+2.50** [2.07, 2.97] |
| forearm direction change | 10.66 | 14.13 | 14.73 | 14.11 | −0.01 [−0.37, 0.35] |
| bend-side change | 11.67 | 16.88 | 17.97 | 17.61 | +0.72 [0.39, 1.07] |

**The forearm is not noisier at all.** All of F2's extra frame-to-frame motion sits in the upper arm and elbow.

**Why.** The per-frame visual chain f changes more between frames than H0's: mean |Δf| 0.133 for E1 and 0.147 for E0, against 0.109 for H0 and 0.086 for GT. That alone adds +1.2° of chain-direction motion per step. The elbow then mixes two sources:
- the E1 chain direction, which moves on its own per-frame noise;
- the H0 reach and H0 bend side, which follow H0's per-frame noise.

These are independent, so their frame-to-frame errors add rather than cancel. In F0, direction, side and reach come from one H0 skeleton and stay mutually consistent. Within each candidate, excess chain-direction motion correlates with excess upper-arm rotation (F0 0.56, F2 0.63).

**By GT bend regime (F0 → F2):**
- proximal change grows in every regime;
- near-folded is worst: 11.83° → 15.56°, with proximal discontinuities 2.9% → 8.1%;
- near-straight is mildest: 5.05° → 6.23°, discontinuities 0.2% → 0.5%.

**The F2 proximal discontinuities cluster** in `downtown_stairs_00` a0 (18 of 185 pairs), `flat_guitar_01` a0 (16/137), `flat_packBags_00` a0 (16/243), `downtown_bar_00` a0 (13/221) and `outdoors_fencing_01` a0 (11/173).

## Per-sequence transfer (F2 vs F0, mean per row)

**Elbow improves in 20/37 sequences.**

Largest gains (wrist; elbow):

| sequence | wrist (cm) | elbow (cm) |
|---|---|---|
| `downtown_bar_00` a0 | 41.3 → 21.5 | 21.9 → 12.7 |
| `flat_guitar_01` a0 | 24.5 → 12.6 | 18.2 → 11.7 |
| `downtown_rampAndStairs_00` a1 | 15.3 → 9.0 | 9.4 → 6.3 |
| `downtown_runForBus_00` a1 | 17.4 → 11.1 | 9.0 → 6.6 |

Largest regressions:

| sequence | wrist (cm) | elbow (cm) |
|---|---|---|
| `flat_packBags_00` a0 | 12.4 → 14.8 | 6.3 → 9.1 |
| `downtown_stairs_00` a0 | 12.0 → 17.2 | 7.2 → 9.8 |
| `downtown_walking_00` a0 | 11.6 → 8.9 | 8.0 → 9.5 |
| `downtown_bar_00` a1 | 13.6 → 11.9 | 7.1 → 8.6 |

**Wrist up, elbow down within one sequence.** In `downtown_walking_00` a0 and `downtown_bar_00` a1 the wrist improves while the elbow regresses: the same trade-off at sequence level.

**The forearm improves in 32/37 sequences** (it gets worse only in `downtown_stairs_00` a0, `downtown_cafe_00` a1, `downtown_runForBus_00` a0, `downtown_crossStreets_00` a0 and `downtown_upstairs_00` a0).

**Worklog 75 owner rows (wrist cm / elbow cm / upper ° / forearm °):**

| row | F0 | F1 | F2 |
|---|---|---|---|
| `downtown_bar_00` a0 #775 | 81.0 / 38.8 / 88.6 / 106.0 | 30.9 / 11.2 / 23.2 / 45.3 | **8.3 / 4.5 / 9.2 / 24.7** |
| `downtown_runForBus_01` a0 #185 | 28.7 / 16.1 / 33.7 / 34.8 | 42.0 / 22.7 / 48.2 / 50.5 | **9.1 / 7.1 / 14.7 / 16.6** |
| `downtown_sitOnStairs_00` a0 #325 | **4.9 / 5.2 / 10.8 / 4.7** | 63.8 / 39.7 / 91.1 / 54.5 | 67.7 / 42.4 / 99.4 / 56.7 |
| `flat_packBags_00` a0 #446 | **8.1 / 3.5 / 7.2 / 16.1** | 16.7 / 14.3 / 29.8 / 6.7 | 30.2 / 25.4 / 54.5 / 17.7 |

**When E1 gets the direction right, the whole arm follows; when it is wrong, the whole arm follows too.** In the two failing rows the E1 chain f is far off:
- #325: E1 −0.93 vs GT +0.49 (wrong sign; H0 +0.58);
- #446: E1 −0.88 vs GT −0.16 (H0 −0.14).

## Worklog 74 validation population (separate; selection-contaminated split)

| population | rows / seq | F2−F0 wrist | F2−F0 elbow | F2−F0 forearm | F2−F1 elbow |
|---|---|---|---|---|---|
| Worklog 74 rows (dancing a0, crosscountry a0, hug a1) | 395 / 3 | −0.58 (1/3) | +0.04 (1/3) | −0.50° (2/3) | −0.09 (2/3) |
| all eligible validation rows | 3,056 / 16 | −1.91 (13/16) | −0.26 (9/16) | −2.04° (14/16) | −0.30 (12/16) |

- **The Worklog 74 population still shows almost no full-arm change,** consistent with Worklog 74.
- **On all validation rows, the test pattern repeats in smaller form:** a robust wrist and forearm gain, and an elbow gain in only about half the sequences.
- **Validation cannot support temporal claims.** Worklog 73 validation predictions are temporally sparse (3 and 5 adjacent pairs), so no validation temporal claim is made.
- **Validation is never merged with test.**

## Real-scene Blender review exports

- **Renders:** BaseRig, Blender 5.1.2 Workbench, 960 × 960, with the Worklog 62/74 cameras (side = depth × up; front).
- **Colors:**
  - F0 blue, F1 orange, F2 magenta;
  - GT-anchored reference green;
  - E_GT yellow wire;
  - gray = BaseRig rest pose under the Worklog 60 object rotation.
- **Object rotation for test sequences.** Test sequences have no Worklog 60 replay files, so the object rotation is the Worklog 60 formula A·Rz(yaw_H0)·A⁻¹. The export first proves it against the persisted Worklog 60 rig rotation on all 395 Worklog 74 rows (max component diff 0).
- **Captions.** Every caption carries the exact sequence, frame, bank position, GT bend, detector confidence, and the wrist, elbow, upper, forearm and chain errors with side-vs-GT for F0/F1/F2.
- **Accuracy.** Blender FK end vs export: 9.5e-7 cm.

| export (`review/<label>_{side,front}.png`) | row | note |
|---|---|---|
| `largest_e1_wrist_improvement+w75_owner_downtown_bar_00_775` | bar_00 a0 #775 | F0 81.0 → F2 8.3 cm wrist, 38.8 → 4.5 cm elbow |
| `largest_e1_wrist_regression+largest_e1_elbow_regression+w75_owner_downtown_sitOnStairs_00_325` | sitOnStairs_00 a0 #325 | E1 f wrong sign; whole arm points away |
| `largest_e1_elbow_improvement` | bar_00 a0 #399 | elbow 49.5 → 6.2 cm; F0 and F2 both flip side vs GT |
| `wrist_improves_elbow_regresses` | bar_00 a0 #514 | wrist 44.7 → 26.1 cm but elbow 5.9 → 29.6 cm; GT bend 116° (reach 0.53), but H0 reach 0.97 (bend 34°) |
| `near_straight` / `strongly_bent` | median-F2 rows of GT bend < 15° / ≥ 90° | — |
| `strongly_out_of_plane` | median-F0 row with \|f_GT\| ≥ 0.5 | — |
| `low_detector_confidence` | lowest min arm confidence | — |
| `w75_owner_downtown_runForBus_01_185`, `w75_owner_flat_packBags_00_446` | Worklog 75 owners | — |

**What the review shows.** It illustrates the mechanism above. Where E1's chain direction is right, the wrist and forearm move toward GT. Where the true bend is strong but H0's reach is far too long (#514: 0.97 vs GT 0.53), the nearly straight arm is laid along the chain and the elbow lands far from GT. The renders are illustrations only; the numbers above are the evidence.

- **Server copies:** `review/` under the output directory.
- **QE images-only copy:** `~/animcv-output/76_visual_full_arm_ik_review/` (20 PNGs).

## Architecture classification — CASE B

The predeclared criteria were mechanically evaluated in `report.json → test.decision_inputs`:

| CASE A requirement | result |
|---|---|
| Worklog 74 full-IK parity | **pass** (≤ 4.5e-14 cm, quaternion ≤ 7.8e-16) |
| Worklog 75 endpoint parity | **pass** (≤ 4.3e-14 cm on 6,003 rows) |
| F2 preserves the wrist advantage | **yes**: −2.99 cm, all intervals, 32/37 |
| meaningful elbow and/or segment improvement | **forearm yes** (−4.24°, all intervals, 32/37). Elbow/upper-arm no: sequence-cluster intervals include 0, 20/37 sequences, 59–63% from one sequence |
| no material elbow p95 regression (> 1.0 cm) | **yes**: p95 improves by 2.56 cm |
| no systematic new rotational instability | **no**: upper-arm rotation change +2.54°/pair (all intervals exclude 0), and candidate-specific upper-arm and elbow discontinuities 2.4× F0's (predeclared flag: > 1.25×) |
| no increase in invalid/unavailable solves | **yes**: 0 for all candidates |
| broadly distributed improvement | wrist/forearm yes; elbow no |

**Not CASE D:** both parities pass. **Not CASE C:** the Worklog 75 endpoint advantage reproduces exactly, invalidity does not increase, and the wrist/forearm gains are broad.

**It is CASE B:**
- the wrist improves;
- the connected arm's rotational stability deteriorates materially, concentrated in the upper arm and elbow;
- the static elbow gain is too concentrated to count as a full-arm improvement.

**The failing geometry owner is the H0 reach/bend-side ownership under a substituted chain direction.** The elbow is placed from three sources:
1. the chain direction (E1);
2. the reach fraction (H0), which alone fixes the elbow angle (identical 14.3° \|bend error\| for every candidate);
3. the bend side (H0, re-perpendicularized to the E1 direction).

When only (1) is replaced:
- **Static.** The wrist and forearm improve. The elbow cannot benefit where (2) is wrong, most visibly in near-folded arms (elbow +0.47 cm while forearm −10.4°).
- **Temporal.** The per-frame noise of the visual chain f (|Δf| 0.133 vs H0's 0.109) is no longer correlated with the H0 reach/side noise, so the upper arm swings more between frames (+2.5°/pair) while the forearm does not.

Per the direction, no IK tuning, smoothing, gating or new bend heuristic was added.

**Answer to the completion question.** No, not yet. The BEDLAM-supervised chain direction improves the wrist and forearm of the connected left-arm IK pose, and improves on E0 everywhere. But under the frozen H0 reach and bend-side ownership it does not deliver a broad elbow gain, and it introduces a systematic increase in upper-arm frame-to-frame rotation and elbow discontinuities. The handoff is therefore not promoted as a production-integration candidate in this form. No fusion, gating or production step is taken.

## Focused tests

- **`tests/test_framepose_visual_full_arm_ik.py` (6 tests):**
  - full arm = Worklog 74 `solve_with_direction` = direct `solve_two_bone`, with local rotations and FK ≤ 1e-9 and fixed bone lengths;
  - deterministic reconstruction;
  - wrist error = the Worklog 75 analytic endpoint for any yaw/direction;
  - reference = the Worklog 62 oracle;
  - fixed H0 reach and bend ownership;
  - no GT leakage (the candidate is independent of the GT arm);
  - unavailability and clamp propagation, with the same reasons as Worklog 74 and no substitution;
  - temporal metrics on synthetic rigid motion and the jump classes;
  - static runner contracts: both gates before any test metric, identity checks, no training/smoothing, no production imports.
- **Server run, `animcv-framepose:cuda118`:** the new tests together with the Worklog 61/62/74/75 focused tests (`test_framepose_two_bone_ik.py`, `test_framepose_ik_gt_attribution.py`, `test_framepose_visual_ik_handoff.py`, `test_framepose_visual_ik_endpoint.py`) give **37 passed**.
- **Full regression:** not run, because production behavior is unchanged.

## Reproduction

    # LabServer63, repo at ~/AnimCV, 3DPW raw mounted directly (the root contains sequenceFiles/)
    docker run --rm -v $HOME/AnimCV:/workspace:ro -v $HOME/animcv-output:/output \
      -v $HOME/animcv-data/datasets/3dpw:/data/3dpw_raw:ro \
      -e PYTHONPATH=/workspace/src:/workspace/scripts -w /workspace animcv-framepose:cuda118 \
      scripts/run_framepose_visual_full_arm_ik.py --rig examples/BaseRig.fbx \
      --rest examples/e2e_demo/baserig_blender_rest.json --calibration examples/e2e_demo/framepose_fk_calibration.json \
      --chains examples/e2e_demo/framepose_ik_chains.json --bank /output/framepose/bank_3dpw_paired_v2.json \
      --h0-dir /output/framepose/context_h0_v2 --threedpw-raw /data/3dpw_raw \
      --semantics-dir /output/animation_semantics_v2_valid_yaw_replay --ik61-dir /output/framepose_two_bone_ik_replay \
      --w62-dir /output/framepose_ik_gt_attribution --w73-dir /output/framepose_external_visual_supervision/transfer \
      --w74-dir /output/framepose_visual_ik_handoff --w75-dir /output/framepose_visual_ik_endpoint_test \
      --out-dir /output/framepose_visual_full_arm_ik
    # review export (same container), then Blender locally:
    scripts/export_framepose_visual_full_arm_cases.py ... --fk60-dir /output/framepose_object_owner_replay \
      --full-arm-dir /output/framepose_visual_full_arm_ik --w74-dir /output/framepose_visual_ik_handoff \
      --out /output/framepose_visual_full_arm_ik/review_cases.json
    Blender -b -P scripts/review_framepose_visual_full_arm_blender.py -- --rig examples/BaseRig.fbx \
      --rest examples/e2e_demo/baserig_blender_rest.json --cases review_cases.json --out-dir review --caption-python python3
