# Worklog — Root, Pelvis and Thorax Orientation Ownership (2026-10-23)

## Starting state and environment

- **Branch and HEAD:** `arch/single_frame_first`, starting HEAD `188629d` (Worklog 93). Local, origin and LabServer63
  were in sync and the worktree was clean. I read PRODUCT_VISION.md and Worklogs 91–93.
- **Where it ran:**
  - Locally at low priority: the replay (≈ 55 s) and Blender 5.1.2.
  - LabServer63: only the copies of the needed RGB frames and the server tests. No GPU was used.

**What was not changed.** No checkpoint, S0, J1X, W91 provider, W92 advisor, W93 frame construction or replay,
FK/IK, BaseRig hierarchy, rest pose, `RootOrientation` / AnimationSemantics schema or production default was changed.
Nothing was trained or tuned.

**Historical decisions stand.** W93 stays Engineering CASE B. Its reported numbers are reproduced byte-for-byte (§10).

## 0. Answer

**CASE B: the ownership separation is representable and useful under a GT-controlled pelvis, but AnimCV's runtime
evidence cannot determine the pelvis/root state.**

**1. W93 cannot express a waist bend.**
- Its pelvis frame is built around the same trunk axis as the thorax frame. That makes `spine_01` a pure axial-twist
  joint: its measured swing is exactly 0.
- All trunk flexion lands on the Armature Object, which also carries both top-level thighs.
- On 3DPW TEST standing waist bends, the Object tilts 50.4° from gravity while the independent SMPL pelvis tilts
  19.5°. The rest legs swing with it: rest-thigh error is 61.5° (A, W93 ownership) against 29.5° (B, independent
  pelvis).

**2. An independent pelvis on the Object fixes that within the existing rig contracts.**
- With the full pelvis→thorax rotation on `spine_01`, the thorax, shoulders and J1X arm stay bit-identical (≤ 7e-14).
- No hierarchy change is needed: BaseRig has no pelvis bone, so the Object already plays that role.

**3. It is not a complete animation improvement.**
- With REST legs, a correct pelvis still leaves the legs wrong: 29.5° on waist bends. A real hip would flex to keep
  the legs vertical.
- A hip-yaw-only root (U) looks better on standing bends but stands a lying body up (rest-thigh error 120° on 3DHP
  lying).

**4. The required pelvis state is not observable at runtime:**
- 3DPW's SMPL fit provides a pelvis reference; 3DHP provides none.
- The best joint proxy (hips + lower-trunk joint) matches the SMPL pelvis within about 5° on GT joints. That is a
  property of SMPL-generated joints: the spine1 joint rides rigidly on the SMPL pelvis.
- The same proxy on the frozen S0 joints is 18° off on average (p95 37°).

**5. World alignment is a separate missing input.** All frames are camera-relative; gravity is available only from GT
camera data.

## 1. Inputs, checkpoints and baselines

**Frozen inputs** (hash-pinned; the script STOPs on change):
- J1X: W87 predictions `cb9dc51e…` (3DPW) and `356f8cee…` (3DHP).
- S0: `402d70a2…` (VAL), `4357c7bd…` (TEST) and `a78adeb0…` (3DHP).

**Baselines, unchanged** (W93 definitions and row filtering, so the W93 fixed-rule windows are reproduced exactly):
- Y_GT = C.
- W93 F_GT = A.
- W93 F_S0 = A_S0.

**Populations:** 3DPW VALIDATION (3407 rows), 3DPW TEST (7076) and MPI-INF-3DHP TEST (2728). Timestamps and camera
conventions are unchanged.

## 2. Current representation audit (W93)

**Frames.**
- u = normalize(thorax − pelvis).
- R_t = [perp_u(R_sh − L_sh), u × x, u] and R_p = [perp_u(R_hip − L_hip), u × x, u].
- Both have z = u, so R_pᵀR_t fixes the z axis. It is a rotation **about the trunk axis only**: axial twist, 1 DOF.

**Rig mapping.**
- Object = A·R_p·A⁻¹.
- `spine_01` local = Bᵀ·A·(R_pᵀR_t)·A⁻¹·B.
- Arm = A·R_tᵀ·v through the W76 IK.

**BaseRig.**
- Top-level bones: `center_of_mass`, `interaction`, `spine_01`, `thigh_l`, `thigh_r`. There is no `pelvis` bone.
- `clavicle_l` hangs from `spine_05`.

So the Object rotation is the only transform that reaches both thigh chains (world thigh = A·R_o·d_rest). Nothing
animates the thighs.

**DOF accounting:**

| DOF | W93 owner | Representable? |
|---|---|---|
| whole-character orientation | Object (R_p) | yes, but merged with pelvis and trunk tilt |
| anatomical pelvis orientation | none (R_p is the trunk axis plus the hip line) | no: pelvis pitch about the hip axis is replaced by the trunk inclination |
| pelvis→thorax flexion / extension | none (`spine_01` swing ≡ 0) | **no**: goes to the Object |
| lateral bending | none | **no**: goes to the Object |
| axial twist | `spine_01` | yes |
| shoulder/arm articulation | W76 IK in the thorax frame | yes (unchanged) |

**Controlled fixtures** (`FIXTURES` in the script; true pelvis and thorax known by construction; rest legs; legs
along the fixture's thighs):

| Fixture | A W93: Object tilt / Object vs true pelvis / rest-thigh err | U hip-yaw root | B true pelvis |
|---|---|---|---|
| erect | 2.5 / 2.5 / 3.2 | 0 / 0 / 3.7 | 0 / 0 / 3.7 |
| upright pelvis, waist flexion 60° | **45.8 / 45.8 / 47.9**; trunk-relative rotation lost (60° off) | 0 / 0 / 3.7 | 0 / 0 / 3.7 |
| whole body lying (supine) | 92.5 / 2.5 / 3.2 | **0 / 90 / 92.0** | 90 / 0 / 3.7 |
| pelvis tilted 20°, torso compensating | 2.0 / **18.0** / 5.1 | 0 / **20.0** / 3.7 | 20 / 0 / **22.2** (rest legs) |
| axial twist 40°, no flexion | 2.5 / 2.5 / 3.2 (twist on `spine_01`) | 0 / 0 / 3.7 | 0 / 0 / 3.7 |

The fixture's 2.5° comes from its lumbar offset. Its 14° thorax error under waist flexion is the W93 definition: the
thorax z axis is the pelvis→thorax chord, not the upper-trunk segment.

**Reading the fixtures:**
- W93 cannot tell a waist bend from a whole-body tilt.
- A hip-yaw root cannot represent lying.
- Neither sees a compensated pelvis tilt.
- Even the true pelvis gives wrong legs while the legs stay at rest.

## 3. Independent pelvis reference and observability

**Sources.**

| Source | Pelvis orientation | Gravity (evaluation only) | Evaluable |
|---|---|---|---|
| 3DPW | **SMPL `poses[:3]` (global orient) through `cam_poses`**, mapped to AnimCV body axes (x right, y forward, z up) | world +Y via `cam_poses` | yes |
| MPI-INF-3DHP test | none (17 joints only; its `spine` joint belongs to the 3DHP skeleton, not a pelvis bone) | `test_util/camera_calibration` up/right vectors (TS1–4 `cam_8`: 12.5° tilt; TS5–6: 5.7°), assuming world +Y up | gravity yes, pelvis **no** |
| S0 / runtime | 17 predicted joints | none | — |

**Validity of the SMPL reference:**
- It is a body-model fit, not a direct pelvis measurement.
- Its x axis sits a constant 3–6° from the hip-joint axis (template).
- Under an upright camera, its zero pose maps exactly to facing the camera (tested).

**Validity of the gravity references:**
- The 3DPW world frame is only approximately gravity-aligned. Pelvis, trunk and camera up average ≈ +Y, but some
  sequences, such as downstairs, share a common 12° offset.
- The 3DHP studio assumption is consistent with the data: upright trunks are 9.8° from gravity against 16.0° from the
  camera vertical.

**Joint-geometry identifiability.**
- Hips plus the pelvis midpoint fix only the hip line. Pelvis pitch about that line is unobservable without a third
  rigid pelvis point.
- Leg joints do not fix it either: the hip is a 3-DOF joint.
- The only other candidate is a lower-trunk joint. Measured against the SMPL pelvis on 3DPW TEST (constant offset
  fitted on VALIDATION upright rows):

| Proxy (3DPW TEST, deg) | upright | waist bend | other | invalid |
|---|---|---|---|---|
| W93 trunk axis (GT joints) | 8.0 (p95 14.8) | **23.6** (36.1) | 22.3 | 0 |
| hips + lower trunk, pelvis→spine1 (GT joints) | **5.1** (6.5) | **5.4** (6.5) | 5.5 | 0 |
| thigh-aligned | 7.6 | 27.7 | 51.1 | 0 |
| hip-yaw upright root | 11.6 | 23.1 | 23.6 | 0 |
| W93 trunk axis (S0 joints) | 17.0 (35.1) | 25.1 | 31.0 | 175 |
| hips + lower trunk (**S0 joints**) | **18.3** (36.8) | **16.8** (42.6) | 19.3 | 176 |

**What the proxy result means.** The GT lower-trunk result shows the pelvis is identifiable from **SMPL-generated**
joints, because SMPL places spine1 rigidly on the pelvis bone (offset about 12°). It does not show the pelvis is
observable from real anatomy, from 3DHP's skeleton or from S0.

**S0 cannot recover it.**
- The spine1 lever is about 10 cm, so a few centimetres of joint error is tens of degrees of pitch.
- The S0 proxy is worse than assuming an upright pelvis on upright rows (18.3° vs 11.6°) and only modestly better on waist bends (16.8° vs 23.1°).

**GT decomposition** (3DPW, gravity strata): on standing waist bends the trunk tilts 47.1°, of which the SMPL pelvis
contributes 17.1° and pelvis→thorax flexion 32.5°. Upright: 7.3 / 6.8 / 6.8.

**Strata** (evaluation only, gravity-relative):
- upright: trunk < 30°, thighs < 30° from down;
- waist_bend: trunk ≥ 30°, thighs < 30°;
- lying: trunk ≥ 60°, thighs ≥ 60°;
- other: everything else (sitting, crouching, kneeling).

## 4. Controlled ownership experiment

**Design.** Every ownership uses the same GT thorax frame R_t, the same frozen J1X arm and the same IK. Only the
Object frame R_o changes, and `spine_01` carries R_oᵀR_t.

| Owner | Object frame | Notes |
|---|---|---|
| A | W93 R_p | frozen F_GT |
| B | SMPL pelvis | evaluation only; 3DPW only |
| L | hips around pelvis→spine joint | proxy, not a reference |
| U | Rz(hip-axis yaw) | trunk fully on `spine_01` |
| C | Y_GT yaw-only control | arm in the yaw frame |

GT-free counterparts A_S0 / L_S0 / U_S0 use the S0 joints with the W93 fallback.

**Legs.** The rig has no leg animation, so thigh results are a **rigid rest-leg diagnostic**:
- angle between A·R_o·d_rest and the GT hip→knee direction;
- GT legs appear only as thin green evaluation overlays.

## 5. Measurements (whole populations; sequence-cluster 95% intervals)

**3DPW TEST** (7076 rows; SMPL reference on all of them):

| Stratum (rows) | Object tilt from gravity: A / **B** / L / U / C | rest-thigh error: A / **B** / L / U / C | A − B thigh | A pelvis error vs SMPL |
|---|---|---|---|---|
| upright (5908) | 6.6 / 5.7 / 20.0 / 11.1 / 11.1 | 14.0 / **12.0** / 16.6 / 15.8 / 15.9 | +2.0 [1.7, 2.5] | 6.1 |
| waist_bend (241) | **50.4** / **19.5** / 12.1 / 13.5 / 13.5 | **61.5** / **29.5** / 17.0 / 19.5 / 19.7 | **+32.0 [22.3, 36.0]** | 33.2 |
| other (924) | 18.4 / 28.2 / 41.6 / 9.0 / 9.0 | 83.3 / 55.6 / 43.0 / 71.7 / 71.8 | +27.7 [19.3, 33.8] | 31.5 |
| lying (3) | 69.2 / 22.6 / … | 132.7 / 88.1 / … | (3 rows; not interpretable) | 48.1 |

**3DPW VALIDATION.**
- Waist bend (198): Object tilt A 43.0 vs B 14.2. Rest-thigh A 59.9 vs B 30.6 (+29.3 [24.2, 32.0]).
- Upright: A − B thigh +4.7 [2.0, 7.5].

**MPI-INF-3DHP** (no pelvis reference; 1473 upright / 312 waist_bend / 118 lying / 825 other):

| Stratum | Object tilt: A / L / U | rest-thigh: A / L / U |
|---|---|---|
| upright | 9.8 / 7.9 / 10.5 | 19.3 / 16.1 / 16.7 |
| waist_bend | **72.0** / 53.0 / 9.1 | **79.7** / 61.0 / **18.7** |
| lying | 80.4 / 88.7 / 12.2 | 76.7 / 58.3 / **120.4** |

**Upper body.** Across A/B/L/U, the world upper-arm direction and the pelvis-relative shoulder position are identical
(max difference 5e-14 / 7e-14 rig units). The arm local rotations are the W93 F_GT ones. Upper-body accuracy is
therefore untouched by the ownership choice; only the pelvis, `spine_01` and the legs change.

**`spine_01` swing:**
- A: exactly 0 everywhere.
- B: carries the GT flexion (33.0° on TEST waist bends).
- U: carries the whole trunk tilt (50.1°).

**Object discontinuities** (> 90° step while the reference steps < 45°):

| | GT owners | A_S0 | L_S0 | U_S0 |
|---|---|---|---|---|
| 3DPW TEST | 0 | 15 (5 at fallback) | 15 (3) | 13 (3) |
| 3DHP | 0 | 22 (13) | 31 (9) | 12 (2) |

The spins come from S0 observation (chest-sign flips and fallback switching), not from ownership. U has fewer because
its Object carries only the hip yaw.

## 6. Real-scene Blender review

**Setup.**
- Skinned mannequin `examples/skinned_mannequin/BaseRig_Skinned.fbx`, bound by numeric equivalence (6.1e-5).
- Identical cameras, timestamps, CONSTANT interpolation and J1X arm in every slot.
- Rest legs, with GT legs and GT arm as thin green overlays (evaluation only).
- Blender FK vs export ≤ 1.1e-4.
- Clips: `~/animcv-output/94_framepose_pelvis_torso_ownership/` (9 mp4 + 9 png).

**Window means** (Object tilt from gravity / rest-thigh error, degrees):

| Clip (window) | Slots | Means | What is visible |
|---|---|---|---|
| 3dpw_bent_or_seated (packBags 1006) | A / B / C | tilt 55.1 / 15.7 / 11.8 (SMPL pelvis 15.7); thigh 65.8 / 23.2 / 11.7 | A tilts the whole body like a plank with the legs off the floor; B bends at the waist with the legs near vertical; C stands upright with the arm sticking out |
| 3dpw_pelvis_stable_bend (windowShopping 140, fixed rule: most waist-bend rows outside the W93 windows) | A / B / U | tilt 48.6 / 24.7 / 18.9; thigh 62.6 / 38.9 / 31.3 | same pattern; B's legs lean by the true pelvis tilt because they are rest legs |
| 3dpw_walking_upright | A / B / C | tilt 4.3 / 4.3 / 15.3; thigh 12.6 / 12.6 / 22.8 | A ≈ B; the camera-relative lean (§7) is visible in all camera-relative slots |
| 3dpw_walking_gravity | A / B / B levelled | B levelled by the GT camera tilt (17.3° at the shown frame) stands vertical | camera tilt, not anatomy, causes the lean |
| 3dpw_turn_180 | A / B / C | thigh 9.2 / 8.2 / 10.8 | turning is unaffected |
| 3dhp_ts2_supine | A / L / U | tilt 82.0 / 100.5 / 12.5; thigh 75.2 / 56.7 / 144.6 | A lies correctly; U stands the body up with the trunk folded horizontal; L over-tilts (3DHP spine joint offset) |
| 3dhp_ts4_lying (W93 name) | A / L / U | tilt 54.0 / 60.6 / 12.5 | the shown frame is a **standing waist bend**: A lays the whole body horizontal, U matches the image; the window also contains crouch/floor poses (U thigh mean 105) |
| 3dhp_upright_lying_transition | A / L / U | — | identical while upright; A's legs rotate with the trunk once the subject bends down |
| 3dhp_s0_spins (TS2 760, fixed rule: most A_S0 spins) | A_S0 / U_S0 / A | — | A_S0 alternates between lying and the yaw fallback; U_S0 has a steadier Object but stands a prone subject up |

**Answers to the review questions:**
- **Does the body lie correctly?** Under A and B yes; under U no.
- **Does a waist bend move the trunk independently of the legs?** Only under B, U and L. Under A the legs rotate with
  the trunk.
- **Are shoulders and arms correctly attached?** Yes, identical in every owner.
- **Are legs rotated by torso motion?** Yes under A.
- **Does an upright walker lean?** Yes in every camera-relative slot; no once gravity-levelled.
- **Do transitions create Object discontinuities?** Only in the GT-free (S0) owners.

## 7. Camera vs gravity

| | camera tilt vs gravity, mean (p95) | upright trunk: camera-relative | upright trunk: gravity-relative |
|---|---|---|---|
| 3DPW TEST | 10.9° (21.4°) | 11.2° | 6.6° |
| 3DPW VAL | 10.4° (16.5°) | 11.6° | 9.3° |
| 3DHP | TS1–4 12.5°, TS5–6 5.7° | 16.0° | 9.8° |

**Upright lean.** About 10–11° of camera tilt is rendered as lean in every camera-relative frame (A, B, L). The
yaw-only and U Objects are vertical *in the camera*, which means 11° off gravity.

**Camera-relative strata mislabel postures.** W93's camera-relative "lying" stratum (inclination ≥ 60°) contains:

| | gravity lying | waist bends | other |
|---|---|---|---|
| 3DHP (417 rows) | 111 | 175 | 131 |
| 3DPW TEST (76 rows) | 3 | 61 | 12 |

W93's lying-row arm results remain valid as thorax-relative arm results. But its visual reading of TS4 as a lying
trunk was a waist bend rendered as whole-body rotation by the W93 ownership. W93's document is left unchanged as
history.

**Runtime.** The runtime has no gravity or camera extrinsics. Gravity-levelled results here use GT camera data and are
evaluation only. World alignment is a distinct product-level input, separate from the pelvis question.

## 8. Attribution

| Effect | Isolated by | Size |
|---|---|---|
| Ownership (trunk flexion on the Object) | A vs B, same thorax and arm | TEST waist bend: Object tilt +30.9°, rest-thigh +32.0° |
| Incorrect pelvis frame | A/L/U vs B pelvis error | A 33° on bends; S0 proxy 17–19° everywhere |
| Rest legs (no hip articulation) | B rest-thigh error | 12° upright, 29.5° waist bend, 55.6° other |
| Camera pitch | camera- vs gravity-relative inclination | 10–12° |
| Frozen J1X limb geometry | unchanged from W93 (arm identical across owners) | — |
| Chest-sign mirroring / invalid-frame fallback | S0 owners' Object spins | 3DHP A_S0 22 (13 at fallback) |

## 9. Contract blockers and alternative explanations

**No rig contract blocker.** With the Object as pelvis and `spine_01` taking the full relative rotation, the existing
hierarchy carries B. The `spine_01` local is verified as Bᵀ·Q·B with world arm and shoulder invariance.

**Blockers to a complete result** (outside this batch):
1. Leg articulation: the rest legs make even the true pelvis show wrong legs.
2. A runtime-observable pelvis state.
3. A runtime gravity reference.

**Alternative explanations and caveats:**
- B is SMPL-defined. A different pelvis definition (e.g. ASIS-based) would shift its offsets.
- The proxy's 5° agreement is a property of SMPL-generated joints.
- 3DPW has only 3 gravity-lying rows, so B vs A on true lying is supported only by the fixtures and by A's correct
  3DHP lying renders.
- Strata thresholds are evaluation choices.

## 10. Changed files, tests and reproduction

**New files:**
- `scripts/run_pelvis_torso_ownership.py`: the experiment, observability, fixtures and case export.
- `tests/test_framepose_pelvis_torso_ownership.py`.
- This worklog.

**Changed (optional and backward compatible):**
- `scripts/render_runtime_heading_rig_blender.py`:
  - per-frame `reference_legs_cam` GT leg overlay (optionally per slot);
  - per-slot `GT_REF_<slot>`.
- `scripts/compose_runtime_heading_rig_video.py`: optional `plot_series` and `header_note`.

No production module, schema, checkpoint or W93 module was changed.

**Tests:** `tests/test_framepose_pelvis_torso_ownership.py` (8 tests):
- right-handed candidate frames;
- shared W93 trunk axis, so `spine_01` is pure twist;
- the five fixtures (waist flexion vs whole-body tilt, lying, compensated pelvis tilt, axial twist, upright);
- Object→thigh propagation and the BaseRig top-level hierarchy;
- spine/arm parent-frame consistency across owners;
- camera vs gravity, covering the SMPL→camera mapping, levelling and the 3DHP calibration tilts;
- GT isolation of the S0 owners, frozen-input hashes and no production import;
- optional renderer/composer extensions.

Together with the W91/W91A/W92/W93 suites: 36 passed locally and 36 passed on LabServer63 (docker
`animcv-external:cu124`).

**Not run:** broad regression, because no shared production contract changed.

**Reproduction checks:**
- The W93 replay rerun is byte-identical (`report.json`, `cases.json`).
- A W93 clip re-rendered with the extended renderer is pixel-identical (216 frames).
- The W94 replay rerun is byte-identical.
- A W94 clip re-render is pixel-identical (216 frames).

**Command:**
```
PYTHONPATH=src:scripts python3 scripts/run_pelvis_torso_ownership.py --bank <w91>/bank_3dpw_paired_v2.json --s0-dir <w91>/pred/S0 \
  --raw-3dpw <w91>/3dpw --w87-3dpw <w91>/pred/W87/predictions_3dpw.npz --hp-bank <w91>/3dhp/bank/mpi3dhp_test \
  --hp-s0 <w91>/3dhp/pred/prediction_S0.npy --w87-3dhp <w92>/pred/predictions_3dhp.npz --w93-cases <w93>/replay/cases.json \
  --rest examples/e2e_demo/baserig_blender_rest.json --calibration examples/e2e_demo/framepose_fk_calibration.json \
  --chains examples/e2e_demo/framepose_ik_chains.json --out-dir <w94>/replay
```
Rendering uses the W93 Blender command with `--cases <w94>/replay/cases_{3dpw,3dhp}.json`.

**Archive:** reports, cases, the log and the .blend files are on LabServer63 at
`~/animcv-output/framepose_pelvis_torso_ownership_archive/`.

## 11. CASE

**CASE B: representation viable, runtime observation limited.**

**The representation works under GT control.**
- An independent pelvis on the Armature Object, with the full pelvis→thorax rotation on `spine_01`, is representable
  in the existing rig contracts.
- It removes W93's conversion of waist flexion into whole-body rotation, with upper-body output bit-identical.

**The required pelvis state is not runtime-observable:**
- S0 joints give a 17–19° pelvis error on every stratum (p95 ≈ 37–43°). The hip-yaw upright prior gives 11.6° on upright rows and 23° on bends.
- 3DHP has no pelvis reference.
- The rig's rest legs keep the lower body wrong even with the true pelvis.
- World (gravity) alignment is unavailable at runtime and is a separate prerequisite.

No experimental adapter was promoted.
