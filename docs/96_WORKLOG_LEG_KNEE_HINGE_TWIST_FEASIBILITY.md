# Worklog — Leg Knee-Hinge and Thigh-Roll Retargeting Feasibility (2026-10-24)

## Starting state and environment

**Branch and HEAD.** `arch/single_frame_first`, starting HEAD `4e863b4` (Worklog 95). Local, origin and LabServer63
were in sync and the worktree was clean. I read PRODUCT_VISION.md and Worklogs 93–95.

**Where it ran.**
- Locally at low priority: the replay (≈ 3 min) and Blender 5.1.2.
- LabServer63 was busy with other projects (load ≈ 33). It was used only to copy 36 RGB frames and for one short
  docker test run.

**What was not changed.**
- No checkpoint, S0, J1X, W91–W95 module, production IK/FK, BaseRig hierarchy, rest pose, schema or production default.
- Nothing was trained or tuned.
- No smoothing, pole vector, foot lock, root translation, gravity or camera heuristic was added.

## 0. Answer

**CASE C: zero-twist is not the dominant cause of W95's knee-hinge score, but a genuine deep-flexion limitation
exists, and AnimCV's existing observations cannot supply its fix reliably.**

**W95's hinge violations are mostly a metric effect.**
- 60–65% of W95's knee-hinge violations (GT legs, SMPL pelvis) come from knees flexed only 20–45°.
- There the zero-twist thigh is already within 5–7° of the independent SMPL thigh roll.
- The violation comes from how the metric is built:
  - the angle between plane normals blows up small out-of-plane offsets at low flexion;
  - the rig hinge convention is uncertain by 7–8°;
  - SMPL's own knee is ≈ 5° off a pure hinge.
- The hinge-aware candidate T (which rolls the thigh about its own axis) **worsens** the SMPL roll agreement in that
  range (5.2 → 11.5°).

**In deep flexion the limitation is real.**
- At ≥ 45° knee flexion, zero-twist bends the rig's calf 9–16° out of its hinge plane on average (sideways
  knee/shin/foot in the skinned rig), and its thigh roll is 8–19° from SMPL.
- With GT geometry, T makes the knee a pure hinge and brings the roll to 2.6–3.4° of SMPL.
- In the Blender close-ups, T visually matches the SMPL-roll oracle.

**But the predeclared independent rule fails even with GT geometry.**
- Flexion ≥ 20°: −0.8° [−4.4, 2.5] on TEST and −1.4° [−2.9, 0.8] on VAL. Neither interval excludes 0.
- Flexion < 20°: +9.5° / +8.9° (the rule allows ≤ 2°).

**With the frozen S0 legs, T is worse and discontinuous.**
- The S0 knee plane is 16° (3DPW) / 26° (3DHP) off the GT plane.
- T's roll error vs SMPL rises by +9.8° (TEST, flexion ≥ 20).
- Severe roll steps rise from 0 to 27 (3DPW TEST) and from 0 to 80 (3DHP).
- T does not fix S0's wrong knee branches.

## 1. Historical identities and reproduction

**Frozen inputs** (hash-pinned; the script STOPs on change):
- J1X `cb9dc51e…` / `356f8cee…`.
- S0 `402d70a2…` / `4357c7bd…` / `a78adeb0…`.

**W95 reproduction.** Rerunning `scripts/run_pelvis_frame_leg_articulation.py` reproduced `report.json` and both case
files byte-identically before any W96 contrast.

**Reuse.** W96 imports W95's row builder, leg poses and windows unchanged. Its `Z` is W95's zero-twist pose, and its
W95 any-violation rate under root B reproduces W95: 0.294 TEST, 0.300 VAL, 24.1% upright, 62.2% waist bend.

## 2. Knee-hinge mathematics and metric audit

**Definitions** (for H = hip, K = knee, A = ankle):
- u = H→K and c = K→A.
- The observed knee-plane normal is n_obs = unit(u × c).
- The armature-space thigh delta is δ_p = B_t L_p B_tᵀ, where B_t is the IK's rest rotation
  (`quaternion_from_matrix(matrix_local)`).
- The posed rig hinge is h = δ_p n_rest, where n_rest = unit(t₀ × c₀) is the rest bend normal that the existing IK
  contract already uses in `rest_plane_side`.

**What zero twist does.**
- δ_p is the shortest arc t₀ → u in the Object frame, so its thigh roll is "none relative to rest".
- The calf delta is the shortest arc c₀ → c, also measured from rest and not relative to the thigh.
- World u and c are exact.
- But the calf generally leaves the rig's hinge plane, so the knee rotates about a non-hinge axis.

**W95's metric** is angle(h, n_obs) for flexion ≥ 20°.
- The physically meaningful defect is the **off-hinge angle**, asin(c·h): how far the calf is bent out of the rig's
  hinge plane (rig varus/valgus).
- For a roll mismatch θ at flexion f, off-hinge ≈ asin(sin f · sin θ). At f = 25°, a 20° normal angle is only an 8°
  off-hinge bend. The 15° bound in W95 therefore over-penalises low flexion.

**Rest axes.**
- The rig rest knee bend is only 5.0°.
- n_rest is 7.8° (left) / 6.5° (right) from the thigh bone-local Z axis, which is the designer's lateral axis.
- The right calf's bone roll differs from the left. Bone roll does not change deformation, but it means bone axes
  cannot be read as anatomical axes.
- So the hinge convention is uncertain by about 7–8°. A candidate aligned to bone-local Z (`T_boneZ`) is reported only
  as a sensitivity check. Its SMPL roll error is measured against n_rest and so carries the constant offset by
  construction.

**SMPL reference** (3DPW, evaluation only).
- The thigh frame is G = R_go·R_hip through `cam_poses`. Its template +x axis is the anatomical lateral/hinge axis.
- SMPL's knee rotation axis is within 5.2° of that axis (median; p95 13.5°) over 2944 flexed knees, so it is a usable
  hinge reference.
- The rest offset to the BaseRig hinge is computed analytically from the identity pose (+7.8° left, −7.5° right),
  with no data fitting.

**Independence caveat.** 3DPW joint positions are themselves SMPL-generated. For bent knees, T's alignment to the
joint plane and SMPL's hinge are therefore not independent. The independent part is the roll at low flexion (from the
IMU-informed SMPL fit) and the visual review.

**Observability.**
- The three-joint plane is undefined for straight knees.
- Its roll accuracy falls with flexion: roll error ∝ (off-plane noise) / sin f.
- Hip internal/external rotation is not identifiable from three joint centres when the knee is straight.

## 3. Attribution (W95 GT legs, SMPL pelvis root B)

**3DPW TEST, per leg** (Z = zero twist):

| Knee flexion | Legs | W95 hinge > 15° (share of violations) | Z off-hinge mean (> 10°) | Z calf twist | Z roll vs SMPL | T roll vs SMPL |
|---|---|---|---|---|---|---|
| 0–5° | 231 | — | 2.5 (0%) | 0.4 | 4.1 | 4.1 (= Z) |
| 5–20° | 7465 | — | 6.5 (11%) | 0.7 | 4.3 | **14.1** |
| 20–45° | 3154 | 1431 (60%) | 6.8 (20%) | 1.8 | 5.2 | **11.5** |
| 45–90° | 2581 | 671 (28%) | 8.9 (37%) | 4.0 | 8.4 | **3.4** |
| ≥ 90° | 721 | 290 (12%) | **15.9 (65%)** | 8.1 | **18.9** | **2.6** |

VALIDATION shows the same pattern: 65 / 24 / 11% of violations; roll Z → T is 6.7 → 8.4, 7.5 → 3.7 and 10.1 → 2.6.

**By posture.**
- Upright legs: Z off-hinge 6.4°.
- Seated/crouched "other": 13.4° (60% > 10°).
- Lying: 11.9°.

**Root propagation.** Under zero twist the roll follows R_o: W95 A/U policies give larger off-hinge on seated rows
(U 19.0°, A 22.6° vs B 13.4°).

**Attribution summary:**

| Source | Evidence |
|---|---|
| Metric construction | The majority of W95 violations (low flexion, small off-hinge, Z roll close to SMPL) |
| Rest-axis convention | 7–8° uncertainty floor |
| Genuine zero-twist roll omission | Deep flexion: off-hinge 9–16°, roll error 8–19°, visible sideways shin |
| Root-frame propagation | Seated rows under A/U |
| Calf local rotation | Adds up to 8° calf twist relative to the thigh at ≥ 90° flexion |
| Near-straight geometry | Plane undefined |

**Zero twist can be visibly wrong while world directions are exact.** In the seated window (flat_guitar_01, left
knee ≈ 100°), Z keeps exact thigh/shin directions but bends the calf 35–47° off the hinge, and the left shin/foot visibly
turn. T and the oracle O do not.

## 4. Candidate T (declared before any candidate result was read)

**Construction (`PREDECLARED["candidate_T"]`):**
- **Roll.** φ is the signed angle about u from perp_u(δ_Z n_rest) to perp_u(s·n_obs). The sign s = ±1 is chosen so
  that |φ| ≤ 90°: no 180° flips, and a bend opposite to the rig hinge stays a visible hyperextension.
- **Degeneracy.** The weight is w = clip((f − 5°)/15°, 0, 1). Straight knees keep zero twist.
- **Calf.** Shortest arc from the rolled rest calf direction to c, which is a pure rotation about h when w = 1.
- **Invariants.** World thigh/shin directions, lengths and the parent Object are unchanged. FK parity is ≤ 2e-13; the
  only exceptions are 1–2 legs per population that carry a 0.05–0.06 rig-unit residual inherited from W95's own IK pose,
  identical in Z, T and O.
- **Mirroring.** Left and right use their own n_rest, which share the lateral sense.

**Why this canonical choice.** It is the minimal roll that makes the knee a rig hinge. Using the existing contract's
hinge axis avoids inventing a new axis.

**What remains unidentified.**
- Hip axial rotation when the knee is straight.
- Tibial rotation and varus/valgus, which are forced to 0 when w = 1.
- The true anatomical hinge, within the 7–8° convention gap.

**Oracle O (evaluation only).** Same construction, with the roll chosen to null the calibrated SMPL roll error.

## 5. Controlled GT-geometry Z/T comparison

Setup: root B, identical GT legs and J1X arm.

| | Z | T | O (SMPL roll) |
|---|---|---|---|
| World parity | exact | exact | exact |
| Off-hinge > 10° (TEST, all legs) | 20–65% by flexion bin | **0%** | 0–9% (SMPL knees are not pure hinges vs the rig axis) |
| Calf twist | 0.4–8.1° | 0° | 0° |
| Hip axial roll > 45° (TEST) | 0 | 0.1–3.5% (by bin) | 0–4.6% |
| W95 any-violation (TEST / VAL) | 0.294 / 0.300 | 0.005 / 0.015 (**circular**: T enforces the metric) | 0.129 / 0.081 |
| W95 ROM-only | 0.005 / 0.015 | unchanged | unchanged |
| Severe thigh jumps / roll steps (TEST, VAL) | 0 / 0 | 0 / 0 | 0 / 0 |

**Predeclared independent rule** (|roll error vs SMPL|, T − Z, sequence-cluster 95% interval):
- **Flexion ≥ 20°:** TEST −0.79° [−4.42, 2.51]; VAL −1.35° [−2.87, 0.75]. Not below 0, so **not met**.
- **Flexion < 20°:** TEST +9.49° [8.35, 10.56]; VAL +8.90°. Above the 2° limit, so **failed**.

The aggregate hides opposite effects: better at ≥ 45°, worse at 5–45° (§3 table).

**Sensitivity.**
- `T_boneZ` shifts the roll by the 7–8° convention gap. Its numbers include that constant offset.
- The hinge-axis choice alone moves the roll by as much as T's low-flexion error.

## 6. Independent qualitative review (skinned mannequin, matched views)

**Clips.** Full-body clips and leg close-ups (`--camera-centre-height 0.45 --ortho-scale 1.3`).
- Leg FK ≤ 7.7e-5; arm ≤ 1.02e-4.
- A re-render is pixel-identical (§10).

| Clip | What is visible |
|---|---|
| 3dpw_seated_or_crouched, 3dpw_worst_hinge (flat_guitar_01; fixed rule: largest Z hinge sum outside the W95 windows) | Z: the left shin/foot visibly turn away from the knee direction (off-hinge −35 to −47°); T ≈ O (roll error −1/−2°); the knee reads as a hinge and the thigh does not look over-twisted (−36 to −48°, the same roll the SMPL oracle needs) |
| 3dpw_walking_upright | Z ≈ O (roll error ≤ 10°); T rolls the thigh 10–20° at low flexion (right knee 10°: −15° from SMPL). Visible only as a slight thigh/knee rotation, but in the wrong direction |
| 3dpw_pelvis_stable_bend, 3dpw_turn_180 | small differences; same low-flexion pattern |
| 3dhp_ts2_supine / ts4 (root A vs L) | T removes the policy dependence of the knee orientation |
| 3dpw_bent_or_seated_s0 | S0 legs: T's thigh roll swings −15 → +20° within the window (right thigh −40° vs SMPL); Z is steady |
| 3dhp_s0_backward_knees | S0 prone knees stay wrong (knees up) under both; T adds thigh rolls of −80 to +60° following the S0 plane |

**Reading.**
- T improves **visible knee behaviour in deep flexion** without an unnatural hip twist there.
- It introduces a **wrong thigh rotation at low flexion**, because the joint plane does not carry the roll there.
- With S0 it **shifts S0's plane error into hip twist** and creates discontinuities.

## 7. Root-policy interaction

The same frozen T was applied to W95's U, A and L root policies.

**What disappears.** The knee-orientation part of W95's policy sensitivity. Off-hinge is ≈ 0 for every root, and on
seated rows the W95 any-violation drops:

| Root | Z | T |
|---|---|---|
| U | 0.791 | 0.096 |
| A | 0.799 | 0.035 |
| B | 0.545 | 0.016 |

**What persists.** The ROM differences from pelvis and spine ownership (W95): U's spine extension; on 3DHP lying,
U 0.841 any-violation, unchanged by T.

**What moves.** The difference moves into the **hip axial rotation** relative to the Object:

| | Thigh roll mean (> 45° rate) |
|---|---|
| 3DPW TEST seated, A | 25.1° (9.1%) |
| 3DPW TEST seated, B | 15.1° (1.4%) |
| 3DHP lying, U | 35.6° (34%) |
| 3DHP lying, A | 15.3° (8.5%) |
| 3DHP lying, L | 12.6° (3.4%) |

**Pelvis mesh orientation.** It still follows R_o (W94/W95). T does not change it.

**Agreement in world knee orientation across roots is not agreement in pelvis anatomy.** No root policy becomes
acceptable by this batch.

## 8. Frozen S0 legs (GT only for scoring)

| | 3DPW VAL | 3DPW TEST | 3DHP |
|---|---|---|---|
| S0 knee plane vs GT plane (both ≥ 20°), mean / median | 15.9 / 13.0° | 16.0 / 13.7° | 26.1 / 21.2° |
| Legs compared / excluded (straight) | 3104 / 3114 | 4816 / 7862 | 3270 / 1794 |
| S0 backward knees (plane > 90° from GT) | 0.3% | 0.1% | 1.4% |
| T − Z roll vs SMPL, flexion ≥ 20 (root B_S0) | **+7.1° [4.9, 9.3]** | **+9.8° [5.9, 13.4]** | n/a |
| Severe roll steps Z → T (root B_S0 / A_S0) | 0 → 9 / 0 → 20 | 0 → 27 / 0 → 70 | A_S0 0 → 80; U_S0 0 → 80 |
| Severe thigh jumps Z → T (A_S0) | 0 → 24 | 9 → 82 | 29 → 120 |
| T thigh roll > 45° (TEST B_S0, by flexion bin) | — | 5–15% | lying 36–38% |

**What is fixed and what is not:**
- Retargeting-induced off-hinge becomes 0 under T for S0 legs too.
- The S0 thigh/shin direction errors (W95: 9–32°) are unchanged.
- The wrong S0 knee planes and backward branches are **not** fixed. T only rolls the hip to follow them.
- S0 frame flips and fallback transitions (W93/W94) remain, and T adds roll steps on top.

## 9. Coverage and uncertainty

**Coverage.** W95 rows unchanged:
- 3DPW VAL 3407 / TEST 7076 rows, GT phase all rows.
- S0 phase 3109 / 6339 (S0 2D leg joints invalid elsewhere).
- 3DHP 2532 / 2728 (GT leg joints invalid elsewhere).

**3DHP.** No SMPL reference, so only rig-internal and GT-plane measures.

**Uncertainties:**
- SMPL thigh roll is a model fit; its low-flexion accuracy is not independently verified.
- The 7–8° hinge-axis convention gap.
- 3DPW has 3 gravity-lying rows.
- The off-hinge 10° and hip-axial 45° bounds are approximate anatomical values (declared before results).

## 10. Changed files, tests and reproducibility

**New files:**
- `scripts/run_leg_knee_hinge_twist.py`: diagnostics, candidate T, oracle O, cases and figure.
- `tests/test_framepose_leg_knee_hinge_twist.py`.
- This worklog.

**No other file changed.** The renderer and composer are used as committed in W95.

**Tests:** `tests/test_framepose_leg_knee_hinge_twist.py` (7 tests):
- rest frames: right-handed, bone Y = bone direction, mirrored lateral hinge, the declared 5–9° convention gap;
- zero-twist round trip, and T preserving FK mid/end with normalized quaternions and 0 off-hinge;
- straight-leg degeneracy and the backward-knee sign rule (no 180° roll);
- the partial weight between 5° and 20°;
- crouch, supine and prone with root-frame invariance of T's world hinge;
- the SMPL oracle nulling the reference roll;
- the predeclared constants, a candidate free of GT/SMPL inputs, frozen hashes and no production import.

**Test runs.** Together with the W91–W95 suites: 51 passed locally and 51 passed on LabServer63 (docker
`animcv-external:cu124`). Broad regression was not run (no shared production contract changed).

**Reproducibility:**
- The W96 replay rerun is byte-identical (`report.json`, cases, figure).
- A Blender close-up re-render is pixel-identical.

**Command:**
```
PYTHONPATH=src:scripts python3 scripts/run_leg_knee_hinge_twist.py <W95 arguments> --w95-cases <w95>/replay --out-dir <w96>/replay
```
Rendering uses the W95 Blender command; close-ups add `--camera-centre-height 0.45 --ortho-scale 1.3`.

**Outputs:**
- Clips, close-ups and the distribution figure: `~/animcv-output/96_framepose_leg_knee_hinge_twist/` (images and
  videos only).
- Reports, cases and .blend files: LabServer63 at `~/animcv-output/framepose_leg_knee_hinge_twist_archive/`.

## 11. CASE and evidence limits

**CASE C.**

**W95's dominant knee-hinge deviations are mainly a metric/convention effect at low knee flexion.** There the
zero-twist roll already agrees with SMPL within 5–7°, and the hinge-aware correction makes the roll worse.

**A meaningful missing thigh-roll representation does exist for strongly flexed knees** (≥ 45°; most visible in
seated/crouched poses):
- zero twist bends the rig knee off its hinge;
- with GT joint geometry, a geometry-derived roll reproduces the SMPL thigh orientation within about 3° and looks
  right.

**But the declared candidate fails its predeclared independent rule,** because it trusts the joint plane at low
flexion.

**With the existing S0 observations the bend plane is too noisy (16–26°) to support it.** The roll gets worse and
discontinuous, and wrong S0 knee branches stay wrong.

**Unresolved:**
- the anatomical hinge axis within the 7–8° convention gap;
- hip axial rotation for straight knees, which three joint centres cannot identify;
- the accuracy of the SMPL low-flexion roll;
- 3DHP anatomy, which has no reference.
