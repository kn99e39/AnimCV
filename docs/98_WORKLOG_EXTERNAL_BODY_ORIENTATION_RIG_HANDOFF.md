# Worklog — External Body Orientation to Rig Ownership Handoff (2026-10-26)

## Starting state and environment

**Branch and HEAD.** `arch/single_frame_first`, starting HEAD `b669173` (Worklog 97, CASE A with regressions). Local,
origin and LabServer63 were in sync. The worktree was clean apart from this batch's two new files. I read
PRODUCT_VISION.md and Worklogs 93–97.

**Prior decisions.**
- W97 CASE A is accepted within its declared scope. It is not read as approval to replace a production model.
- The superseded interim W97 CASE D record stays as it is.
- W96 candidate T was neither used nor tuned.

**Machines.**
- Local M1 Pro: all experiment computation and Blender 5.1.2 rendering.
- LabServer63: the test suite only, in docker `animcv-external:cu124`. No GPU job ran in this batch.

**What was not changed:**
- no model weight, checkpoint, preprocessing, prediction, S0/J1X artifact or W97 output;
- no IK/FK parameter, W93/W94 threshold, rig, schema or production default;
- no smoothing, fallback, root translation, contact or gravity heuristic.

All work is in new files.

## 0. Answer

**CASE B — the representation is valid, but source limitations remain.**

HMR 2.0b's own pelvis and thorax can be put through AnimCV's existing Root/Pelvis/Thorax ownership contracts with no
GT and no fitted transform. The contracts hold exactly:
- world limb directions do not depend on the Object frame (≤ 1.3e-3°);
- parent/local composition reproduces the thorax;
- the oracle reference reproduces W97 to the last digit;
- Blender FK matches the export.

Taking the thorax from the same source (T) makes the trunk more coherent than any condition that mixes sources.

The decisive limit is **source information, not coordinate alignment or transform ownership.**
- The predicted pelvis is 13.4° from the 3DPW SMPL pelvis on TEST and 13.8° on VALIDATION.
  - The tilt part is 7.2° and 8.6°.
  - This error varies from frame to frame. It is not a constant basis offset (§4).
- In bends, seated poses and VALIDATION, that tilt error moves 3–5° of bending from the spine into the hips.
- It also adds Object rotation the subject does not make: two 180° spins on TEST (O had none), and twice as much
  rotation while the reference pelvis is still.
- W97's upright-shin regression comes from HMR 2.0b's own knee flexion and depth. It is upstream of every frame and
  ownership choice tested here (§8).

**Controlled comparison** (3DPW TEST, 7076 identical rows, sequence-cluster 95% intervals; "spine rel" is the
pelvis-relative thorax rotation vs the GT decomposition):

| Change | What changes | Pelvis vs SMPL | Spine rel vs GT | Hip flexion error | Spine twist error |
|---|---|---|---|---|---|
| O → P | oracle → HMR pelvis | +13.41 [12.57, 14.22] | +0.68 [−0.45, 1.79] | +0.27 [−0.41, 1.04] | +0.64 [−0.55, 1.71] |
| P → T | S0 → HMR thorax | 0 | **−5.82 [−6.80, −4.93]** | 0 | **−4.90 [−5.60, −4.27]** |
| O → T (combined, not attributed to one cause) | both | +13.41 | −5.14 [−6.64, −3.81] | +0.27 | −4.26 |

| Regime | P − O hip flexion error | T − P spine rel | T − O hip flexion error |
|---|---|---|---|
| TEST upright (5908) | −0.39 [−0.82, 0.08] | −4.84 | −0.39 (unchanged) |
| TEST waist bend (241) | **+4.46 [0.21, 6.43]** | −11.09 | **+4.46** |
| TEST seated/crouched/other (924) | **+3.37 [1.07, 6.10]** | −10.72 | **+3.37** |
| VALIDATION all (3407) | **+3.50 [1.40, 5.70]** | −5.45 | **+3.50** |
| VALIDATION upright (2206) | **+2.96 [0.71, 5.38]** | −5.36 | **+2.96** |

**Upright TEST frames are roughly at parity:** T keeps O's hip decomposition and improves the spine. That does not hold
in VALIDATION, in bends or in seated poses. No outcome here authorises replacing production FramePose.

## 1. Frozen inputs and identities

| Input | sha256 |
|---|---|
| W97 HMR 2.0b predictions `pred_full.npz` | `fc5d3305…7003` (pinned in code; STOP if changed) |
| W97 population `population.json` | `533895c9…cdb9` |
| S0 TEST / VALIDATION / 3DHP predictions | `4357c7bd…`, `402d70a2…`, `a78adeb0…` (the W93 `S0_SHA` pins) |
| J1X (W87) 3DPW / 3DHP arm predictions | `cb9dc51e…`, `356f8cee…` (the `W87_PREDICTIONS` pins) |

The HMR 2.0b checkpoint, upstream commit, configuration and SMPL v1 body model are as recorded in W97 §1:
- 4D-Humans `efe18def…`;
- `epoch=35-step=1000000.ckpt` `c9ea09c8…`;
- SMPL v1 neutral `98e65c74…` (non-commercial research licence).

Nothing was re-inferred. Every condition reads the same frozen arrays.

## 2. Source representation and coordinate audit

**Representation.** Each W97 prediction stores the following, all from one single-frame crop forward pass:
- `global_orient` (3×3);
- `body_pose` (23×3×3 local rotations);
- `betas`;
- `joints24` (the `smplx.SMPLLayer` forward of those parameters).

| Quantity | Frame | Meaning |
|---|---|---|
| `global_orient` G | OpenCV axes of the **crop** camera (x right, y down, z forward) | SMPL root (pelvis) rotation; template → camera |
| `body_pose[k]` | parent joint frame | SMPL local rotation of joint k+1 |
| `joints24` | crop-camera OpenCV axes, metres, root-relative after subtraction | posed SMPL joints |
| SMPL template | x = subject's **left**, y = up, z = forward (facing +z) | rest pose = T-pose, identity rotations |

**Conversions.** Both are constant, documented and analytic. Neither is fitted to any data.
- `CV_TO_CANON` = [[1,0,0],[0,0,1],[0,−1,0]]: OpenCV → AnimCV camera axes (X right, Y depth, Z up), det +1.
- `SMPL_BODY` (W94): SMPL template axes → AnimCV body axes (x right, y forward, z up), det +1. It flips the template's
  subject-left x to the body's right and maps the template's up and forward onto body z and y.
- GT-free pelvis: **R_p = CV_TO_CANON · G · SMPL_BODY**. This is the same construction the W94 evaluation reference
  applies to the 3DPW SMPL pose (with the world→camera extrinsic in place of the crop camera). The only difference
  between O and P is where G comes from.

**Handedness and rest.** Both matrices are proper rotations, so no reflection enters. Fixtures pass:
- an upright subject facing the camera maps to yaw 180° with up = +Z;
- the identity SMPL orientation maps to upside-down (up = −Z), as it should in OpenCV axes.

**Distinctions kept apart:**
- **Anatomical pelvis vs Object:** the SMPL pelvis frame is *not* the BaseRig Armature Object frame. `SMPL_BODY` is
  not the identity (a test asserts this). The Object takes R_p through the W94 rig alignment (`conj`), not directly.
- **Thorax:** the W93 `torso_frames` thorax frame of HMR 2.0b's own 17 canonical joints.
- **Pelvis-relative spine:** spine_01 = R_pᵀ R_t (W94 ownership).
- **Hips and legs:** W95 two-bone IK with zero twist (Z), from the 17 joints in the Object frame.
- **Camera space vs gravity:** every rotation is camera-relative. Gravity enters only the evaluation-only strata and the
  3DHP tilt diagnostic.
- **Crop camera vs full camera:** HMR 2.0b predicts G for a virtual camera looking through the crop centre. The
  candidate does **not** correct this. The size of the effect is measured as a diagnostic in §4.

**Diagnostic, not a candidate:** the SMPL kinematic-chain thorax (G · spine1 · spine2 · spine3 locals) differs from the
joint-derived thorax by 9.7° mean (median 7.8°, p95 25.5°) on TEST. The two thorax definitions are not the same frame;
T uses the joint-derived one, as in W93/W94.

## 3. GT-free execution contract and first gate

**Adapter.** `handoff_frames(global_orient, joints24) -> (R_p, R_t, joints17)`.
- Its inputs are one frozen HMR prediction only. Its body may not reference SMPL GT, GT joints, extrinsics,
  intrinsics or strata; a source-text test enforces this.
- The evaluation loaders (`Smpl3dpw`, GT joints, intrinsics) are separate and only score.

**Unavailable frames.** A degenerate HMR torso makes T/T_S unavailable for that row. It is counted and never replaced
by GT or by another source. On 3DPW and 3DHP no HMR thorax was degenerate. The renders label unavailable rows
explicitly.

**Calibration provenance.**
- R_p uses only the two constant matrices above.
- The rig side uses the existing `derive_rig_alignment` from `baserig_blender_rest.json` and
  `framepose_fk_calibration.json`.
- The spine basis comes from the rest `spine_01` matrix.
- No transform was estimated from TEST or VALIDATION rotations.

**Gate checks (tests and run):**
- upright, rotated (30°, ±90° about the body's left–right axis) and yawed fixtures give the exact tilt and yaw;
- parent/local composition: conj(R_p) · Q(spine_01) = conj(R_t) to 1e-9;
- O reproduces W97 E+Z exactly on 6339 rows: thigh 9.2771°, shin 14.1042°, roll 6.1835°;
- world thigh directions differ by ≤ 7.8e-4° (TEST) and 1.3e-3° (VALIDATION) across O/P/T. World upper-arm directions
  differ by ≤ 8e-6°. The Object frame re-expresses limbs; it does not move them.

## 4. Pelvis orientation: accuracy and error structure

| 3DPW | Rows | Geodesic mean / median / p95 | Up-axis | Forward-axis | > 30° |
|---|---|---|---|---|---|
| TEST | 7076 | 13.41 / 12.71 / 25.8 | 7.18 | 12.69 | 1.8% |
| VALIDATION | 3407 | 13.76 / 12.94 / 23.7 | 8.65 | 12.90 | 1.3% |

**Is the error a coordinate offset or source error?** This was checked with
`run_external_orientation_handoff.py pelvis-structure`, an evaluation-only diagnostic.

| | TEST | VALIDATION |
|---|---|---|
| Mean body-side offset (chordal mean of R_smplᵀ R_p) | 2.37° | 5.51° |
| Mean camera-side offset (chordal mean of R_p R_smplᵀ) | 2.79° | 3.86° |
| Geodesic after removing the VALIDATION body offset | 14.31 (worse) | 12.61 |
| Geodesic after removing the VALIDATION camera offset | 13.70 (worse) | 13.25 |
| Geodesic after crop-ray re-expression (dataset intrinsics) | 11.55 (−1.86) | 12.91 (−0.85) |
| Up-axis after crop-ray re-expression | 7.15 (−0.03) | 8.44 (−0.21) |

**Reading:**
- The mean offset is small and not stable across splits. Removing VALIDATION's offset makes TEST worse, so there is no
  constant basis or calibration error to fix.
- The uncorrected crop camera accounts for 0.9–1.9° of the error, almost all of it about the vertical axis. It does not
  explain the tilt error.
- The pelvis-error vs crop-ray Spearman correlation is 0.33 on TEST and 0.08 on VALIDATION.
- About 11.5–13° of per-frame error, about 7–8.5° of it tilt, belongs to the predicted `global_orient` itself.

## 5. O / P / T controlled comparison (3DPW)

All conditions share identical rows, HMR legs, J1X arm, W95 Z IK, bone lengths and rig.

**TEST, all rows (7076):**

| | O | P | T |
|---|---|---|---|
| Pelvis vs SMPL | 0 (oracle) | 13.41 | 13.41 |
| Thorax vs GT | 14.63 | 14.63 | **11.44** |
| Spine rel vs GT | 14.63 | 15.31 | **9.49** |
| Spine flexion / twist error | 6.32 / 10.46 | 6.77 / 11.10 | **5.19 / 6.21** |
| Hip flexion / abduction error | 7.78 / 4.25 | 8.04 / 3.91 | 8.04 / 3.91 |
| Spine ROM violation rate | 0.01 | 0.01 | 0.00 |
| Rig thigh / shin direction | 9.57 / 15.45 | same | same |
| Rig knee-plane error | 13.18 | same | same |
| Off-hinge / roll vs SMPL | 10.62 / 6.06 | 7.78 / 12.24 | 7.78 / 12.24 |

**What the table shows:**
- **Legs are Object-invariant.** Thigh, shin and knee plane are identical across O/P/T, so the leg quality is entirely
  the source's (W97).
- **Roll vs SMPL is not invariant: +6.19 [5.24, 7.18] from O to P.** The W95 zero-twist convention defines twist
  relative to the Object frame, so a wrong pelvis puts a wrong roll on the zero-twist legs. Off-hinge falls by 2.84°
  for the same reason: a frame-defined measure moves with the frame. This is a real ownership consequence of the
  existing Z convention (W96), visible only because the pelvis is no longer oracle.
- **S0 thorax under a GT pelvis (O) is inconsistent.** O's spine rel error (14.6°) is larger than T's (9.5°). O pairs
  an oracle pelvis with a different source's thorax, and the mismatch becomes spine articulation (§9 walking clip:
  O spine flexion 15–33° vs T 5–9°).

**VALIDATION, all rows (3407):**
- P − O: spine rel +3.78 [2.33, 5.05]; hip flexion error +3.50 [1.40, 5.70]; spine flexion error +2.69.
- T − P: spine rel −5.45 [−6.32, −4.35].
- T − O: spine rel −1.67 [−3.34, −0.03]; hip flexion error +3.50.

**Strata and challenge subsets (TEST, T − O unless stated):**

| Subset | Rows | Pelvis error | Up | Hip flexion error | Spine rel | Note |
|---|---|---|---|---|---|---|
| Upright | 5908 | 13.53 | 6.68 | −0.39 | −4.40 | parity on the hip, better spine |
| Waist bend | 241 | 13.06 | **11.27** | **+4.46** | −6.32 | P alone: spine flexion error +5.40; T restores the spine (−7.11) but not the hip |
| Seated/crouched/other | 924 | 12.73 | 9.28 | **+3.37** | −9.58 | |
| GT knee flexion ≥ 45° | 2549 | 13.60 | 7.76 | +1.44 [0.38, 2.84] | −7.07 | |
| S0 wrong knee plane | 490 | **17.54** | 7.86 | +0.35 | −8.26 | occluded/bar scenes: worst pelvis |
| Out-of-plane limbs | 3090 | 13.19 | 7.57 | +1.47 [0.58, 2.59] | −5.25 | |
| Gravity-lying | 3 | 10.11 | 7.86 | −0.82 | −8.08 | three rows; no claim |

**Mechanism.** In bends the pelvis tilt error roughly doubles (11.3° vs 6.7° upright). A wrongly tilted Object forces
the rest of the bend into the hips. The HMR thorax fixes the spine part of the decomposition but cannot fix the hip
part, because the hips hang from the Object.

## 6. Optional source-isolation control (T_H = T vs T_S)

T_S keeps the HMR pelvis and HMR thorax and replaces the legs with frozen S0 legs. This is a **mixed-owner control**,
not a coherent S0 output. W97's historical S0 comparison used the GT pelvis and is not relabelled here.

**TEST, T_S − T (6339 rows):**
- shin −3.00 [−4.81, −1.14] (upright −4.61);
- knee plane +4.43 [2.94, 5.86];
- hip flexion error +1.81.

| Subset | Thigh | Shin | Knee plane |
|---|---|---|---|
| Seated/crouched/other | **+11.50** | **+6.47** | **+8.39** |
| Knee flexion ≥ 45° | **+6.01** | +2.65 | **+6.02** |
| S0 wrong knee plane (91) | +7.44 | **+13.00** | **+33.17** |
| Gravity-lying (3) | +11.76 | +9.34 | +13.17 |

VALIDATION gives the same pattern: shin −1.68 (n.s.), knee plane +3.66.

Under a fixed GT-free parent frame, the W97 leg trade-off survives unchanged:
- S0 legs are better on straight upright shins;
- HMR legs are much better on flexed, seated and wrong-branch knees.

## 7. MPI-INF-3DHP (no SMPL pelvis reference)

**Status.** HMR 2.0b is in-domain on 3DHP: it trained on 3DHP train. There is no SMPL pelvis reference, so the pelvis
error and the spine/hip decomposition errors cannot be scored. The thorax is scored against the GT W93 thorax of the
3DHP joints.

**Coverage.** 2532 of 2728 rows for P, T and T_S alike. The 196 missing rows lack the GT leg reference the evaluator
needs. Equal counts across P/T/T_S mean no HMR thorax was degenerate.

| T − P thorax vs GT | Rows | Δ |
|---|---|---|
| All | 2532 | −8.34 [−13.12, −5.14] |
| Upright | 1458 | −5.34 |
| Waist bend | 241 | −6.56 |
| Lying | 88 | **−51.59** (P 67.9° → T 16.4°) |
| Other | 745 | −9.68 |

**Lying.** P's 67.9° is the S0 thorax failing on lying bodies, as in W93/W94.
- The HMR pelvis tilts 93.9° from gravity-up on average, where the GT trunk tilts 80.4° (118 lying-trunk rows). The
  pelvis captures lying but over-tilts by about 13°.
- T_S on lying rows: thigh +18.3°, shin +20.2°, 9 severe hip jumps (T has 0). These are S0 leg failures under a
  correct parent.

## 8. Negative-case attribution: HMR 2.0b upright-walking shins

W97 found that HMR 2.0b worsens shins on typical upright walking. This is preserved and attributed. The legs are
Object-invariant (§5), so no pelvis or thorax choice in this batch can change them.

**TEST legs** (11,816 legs; S = S0, E = HMR 2.0b):

| | S0 | HMR 2.0b |
|---|---|---|
| Shin direction error | 19.16 | 15.54 |
| **Knee flexion error** (rotation-invariant) | 17.38 | **16.27** (front-facing **21.14**) |
| Shin error after the best rigid rotation of the whole skeleton | 18.10 | **13.96** |
| Shin error in the image plane only | 14.11 | **3.98** |
| Spearman(shin error, crop-ray angle) | 0.12 | 0.21 |

**VALIDATION:**
- HMR 2.0b shin error 11.52 vs S0 9.39.
- HMR knee flexion error 13.04 vs S0 7.91 (front-facing 17.81 vs 6.93).
- Image-plane share 2.78.

**Attribution (by the invariant the directive names):**
- **Not a coordinate or crop-frame error.** The best rigid rotation of the whole skeleton leaves 13.96° of HMR shin
  error (10.85° on VALIDATION). Crop-ray correlation is weak (0.21 / 0.03).
- **Not pelvis/torso ownership.** World leg directions are identical under O, P and T.
- **Predicted articulation.** HMR 2.0b's hip–knee–ankle flexion angle is wrong by 16° (TEST) and 13° (VALIDATION),
  21° and 18° on front-facing subjects. Its image-plane shin error is only 3–4°, so the wrong flexion lies along the
  camera depth axis.

This is an **upstream source limitation** of HMR 2.0b's predicted joint geometry: knee flexion is ambiguous along the
depth axis. It is recorded, not corrected.

## 9. Temporal accounting (single-frame, no smoothing)

| TEST (7039 adjacent steps) | O | P | T | T_S |
|---|---|---|---|---|
| Object spins (> 90° step while the SMPL step < 45°) | 0 | **2** | **2** | 0 |
| Spine jumps (W95 rule) | 11 | 13 | **0** | 0 |
| Severe hip jumps | 0 | 0 | 0 | 0 |
| Mean Object step while the SMPL pelvis is still (< 2°; 1726 steps) | 1.09° | **2.05°** | **2.05°** | 1.84° |

VALIDATION gives the same picture:
- Object spins 0 / 3 / 3 / 1;
- spine jumps 1 / 3 / 0 / 0;
- still-interval Object motion 0.98 / 1.53 / 1.53 / 1.36°.

On 3DHP, P and T have no spins, spine jumps or severe hip jumps. T_S has 9 severe hip jumps.

**Reading:** the HMR thorax removes spine jumps entirely. The HMR pelvis adds rare Object flips and roughly doubles
small rotation during still intervals, because each frame is independent.

## 10. Blender qualitative review

**Settings.**
- Rig: BaseRig skinned mannequin (`BaseRig_Skinned.fbx`, equivalent-rig tolerance 1e-3).
- Camera: orthographic scale 3.2, centre height 0.6.
- Front and side (depth × up) panels.
- Identical RGB frames, timestamps, sampling, J1X arm and Z legs in every panel.
- Thin green: GT legs/arm, evaluation only.

**Slots.**
- 3DPW: FIXED = O, RUNTIME = P, GT = T.
- 3DHP: FIXED = P, RUNTIME = T, GT = T_S.
- The side-view captions show the composer's slot names (FIXED/RUNTIME/GT). The header labels name the conditions.

**Selection (fixed rules, provenance):**

| Case | Source rule | Sequence / start |
|---|---|---|
| walking_upright, turn_180, pelvis_stable_bend, seated_or_crouched | W96 fixed-rule windows (W93/W94 lineage) | walkBridge_01 a0 /333; cafe_00 a1 /1021; windowShopping_00 a0 /140; flat_guitar_01 a0 /500 |
| s0_wrong_knee_plane | W97 fixed-rule window | downtown_bar_00 a1 /367 |
| worst_hmr_pelvis | new: maximum summed P pelvis error over a window, excluding frames already used | runForBus_01 a0 /451 |
| 3dhp ts2_supine, ts4_lying | W96 windows | TS2 /500; TS4 /340 |

**Parity.**
- FK end vs export: ≤ 8.4e-5 m (3DPW), 1.1e-4 m (3DHP).
- FK points vs export: ≤ 4.1e-5 m.
- Re-rendering 3DHP gives 432 identical PNGs (maximum pixel difference 0).

**Observations:**
- **Upright walking** (back-facing, walkBridge): the HMR pelvis is 7–24° off the SMPL pelvis. Body posture looks the
  same in all three panels; only the trunk decomposition differs. O shows 15–33° spurious spine flexion from the
  oracle-pelvis / S0-thorax mix. T holds a steady 5–9°.
- **Turn 180°:** the pelvis error rises from 1–6° to about 30° for a few frames in mid-turn. The Object visibly lags or
  over-rotates, and the arms and legs keep their world directions.
- **Pelvis-stable bend** (windowShopping): the pelvis error is 3–14°. T's spine flexion follows O's, with less
  oscillation than P.
- **Seated** (flat_guitar): the HMR pelvis is 12–21° off. P and T put 60–66° on the hips against O's 41–46°, the bend
  moving into the hips (§5). T's spine is flat at about 41° while P oscillates 17–57°.
- **S0 wrong-knee-plane bar scene:** pelvis error 18–39° with occluded legs. Leg poses are identical across panels and
  wrong in all three (the W97 occlusion failure). T removes P's 25° spine twist.
- **Worst HMR pelvis** (runForBus): 25–45° pelvis error for most of the window. The mannequin leans and turns
  differently from O. T's spine stays at 10–15° while O's swings 0–65°. This is the clearest case where the predicted
  pelvis is visibly worse than the oracle.
- **3DHP supine (TS2):** HMR pelvis tilt is 65–130° (above 110° while the legs are raised on the sofa). Under the S0 thorax (P) the
  trunk folds through the floor in the side view; T and T_S lie flat. Gaps in the plots are explicitly unavailable
  rows: the mannequin holds its last pose and the caption reads "condition unavailable", with no fallback.
- **3DHP lying/bending (TS4):** P's lateral spine bend (−27°) disappears under T (−8°). T_S's S0 legs lose knee
  flexion.

These are root-relative articulation only. Root translation, ground contact and foot locking are outside this batch.

## 11. Coverage, uncertainty and alternative explanations

**Coverage.**
- 3DPW TEST: O/P/T 7076 of 7076; T_S 6339 (S0 legs available).
- VALIDATION: 3407; T_S 3109.
- 3DHP: 2532 of 2728 (the GT leg reference is missing on the rest).
- HMR thorax unavailable: 0.

**Intervals.** Sequence-cluster bootstrap, 95%. TEST has 37 sequence-actors and VALIDATION 16, so subset intervals (waist bend 241 rows,
lying 3 rows) are wide.

**Alternative explanations considered:**
1. *Reference mismatch.* The SMPL pelvis reference and HMR's `global_orient` share the SMPL definition. The error is
   therefore not a definitional offset. §4 confirms this: the mean offset is small and unstable.
2. *Crop camera.* It explains 0.9–1.9° of yaw and almost none of the tilt (§4).
3. *Thorax definition.* T uses joint-derived thorax frames on both the prediction and the GT side. The SMPL chain
   thorax differs by about 10°, so T's advantage partly reflects matched definitions. Even so, T beats O, whose S0
   thorax uses the same definition.
4. *Hip error from the GT decomposition.* The reference decomposition itself uses the SMPL pelvis. A predicted pelvis
   that is closer to the true anatomical pelvis than the SMPL fit would read as error. This cannot be separated here.
5. *3DHP in-domain.* The 3DHP thorax gains may be optimistic.

**Unresolved evidence limits:**
- Only three 3DPW gravity-lying rows have SMPL references. Lying claims rest on 3DHP thorax coherence and tilt
  (118 rows), without a pelvis reference.
- No gravity/world-frame test: every rotation is camera-relative.
- Single-frame only. The temporal numbers describe unsmoothed HMR output.
- The zero-twist roll penalty (+6.2°) depends on the existing W95 convention.

## 12. Tests and files

**New files:**
- `scripts/run_external_orientation_handoff.py`: the GT-free adapter, O/P/T/T_S replay, metrics, temporal and shin
  attribution, Blender cases, and the `pelvis-structure` diagnostic.
- `tests/test_external_orientation_handoff.py` (7 tests).
- This worklog.

No existing file changed. The renderer and composer were used unchanged.

**Tests (7):**
- upright facing-camera fixture (yaw 180°, handedness, det +1);
- lying, rotated and yawed fixtures;
- SMPL basis ≠ armature basis (identity SMPL orientation → upside-down);
- SMPL chain thorax uses the spine locals;
- parent/local composition reproduces the thorax;
- GT-free adapter isolation, frozen hashes, and that only the stated source changes between O/P/T/T_S;
- the error-structure helpers: ray rotation, chordal mean, recovery and removal of a constant offset.

**Test runs.**
- W93–W98 suites (50 tests): 50 passed locally and 50 passed on LabServer63 in docker `animcv-external:cu124`.
- The first server run failed one W97 access-gate test with `ENOSPC`: the server's `/var/lib` (docker storage) is 100%
  full, with 145 MB free. The rerun mounted the container's `/tmp` on `/home`. Nothing on the server was deleted.
- Broad regression was not run: no shared contract changed.

**Reproduction.**
- O reproduces W97 exactly (§3).
- Blender re-render is pixel-identical (§10).
- A replay re-run is byte-identical (§13).

**Outputs.**
- Videos and contact sheets: `~/animcv-output/98_framepose_external_orientation_handoff/`.
- Report, pelvis-structure JSON, cases and .blend files: archived on LabServer63 at
  `~/animcv-output/framepose_external_orientation_handoff_archive/`.

## 13. Final checks

- A full replay re-run into a separate directory is byte-identical: `report.json`, `cases_3dpw.json`, `cases_3dhp.json`.
- The `pelvis-structure` diagnostic covers exactly the same 7076 TEST and 3407 VALIDATION rows as the replay.
- The W93–W98 suites pass locally and on LabServer63 (50/50).

## 14. CASE and evidence limits

**CASE B.** The coordinate and rig-ownership contracts work with HMR 2.0b's own pelvis and thorax, without GT:
- analytic conversion;
- exact parity;
- Object-invariant limbs;
- W97 reproduced;
- a coherent same-source trunk that beats the oracle/mixed trunk on spine articulation and removes spine jumps.

Practical animation is limited by the source:
- a per-frame pelvis tilt/orientation error of about 7–9° tilt, 13–14° in total, which moves 3–5° of bending into the
  hips in bends, seated poses and VALIDATION, adds rare Object flips, and doubles motion while still;
- HMR 2.0b's depth-ambiguous knee flexion on upright walking.

Upright 3DPW TEST is approximately at parity with the oracle pelvis. That is not robust across splits and does not
extend to bends or seated poses, so CASE A is not claimed.

It is not CASE C: no mismatch between representation and rig was found. Errors entering the rig equal the source
errors, except for the documented zero-twist roll coupling.

It is not CASE D: every required identity was established.

**Decisive blocker: source information** (the predicted pelvis orientation and knee depth). It is not coordinate
alignment or transform ownership.
