# Worklog — Full Torso Frame and Rig Animation Feasibility (2026-10-23)

Branch: `arch/single_frame_first`. Starting HEAD: `5490afc` (Worklog 92). Local, origin and LabServer63 were in sync and
the worktree was clean.

Nothing was trained, and no VLM was tuned. The following are all unchanged:
- checkpoints, S0, the J1X predictions and the W91 provider;
- `RootOrientation` / AnimationSemantics, FK/IK and BaseRig;
- production defaults.

No root translation, gravity or contact was added, and GT is used only as an evaluation reference.

**Where it ran:** locally at low priority (replay ≈ 29 s, Blender ≈ 5 s per clip), plus the server test run. No server
GPU was needed.

## 0. Answer

**CASE B: the representation is useful, but the GT-free estimate is limited.**

**A correct 3D torso frame does materially improve the rig in exactly the poses yaw-only cannot express:**
- With the same J1X arm and the same IK, switching from GT yaw-only to the GT full torso frame cuts the upper-arm local
  rotation error by 66° on 3DHP lying rows, 34–40° on bent rows, and 5.6–7.5° on upright rows.
- Every interval is below 0.

**The existing GT-free geometry recovers most of that gain on average, but not reliably.** The frame built from the
frozen S0 joints (F_S0) keeps 80–90% of the GT gain on lying/bent rows. However:
- It flips the chest sign on lying bodies.
- It produces more root spins than yaw-only, mostly at its own fallback switches.
- Like every camera-relative frame, it tilts upright walkers by the camera pitch.

It is preserved as an unwired experimental candidate. It is not ready for runtime.

## 1. Motivation (W91 / W92)

**W91.** The GT-free runtime heading is a single yaw ψ. On 3DHP TS2 (lying) its errors came from a mirrored front/back
sign, and the reliability state could not tell.

**W92.** A frozen VLM answers "chest up or down" correctly on lying torsos, but correcting ψ did not make a useful
animation change: on a lying body no yaw can put the chest up. The missing piece was a **root/torso orientation
representation**, not visual evidence. W93 tests that representation directly.

## 2. Orientation ownership audit (existing contracts)

| Owner | What it carries | Where |
|---|---|---|
| `RootOrientation` | `known`, `yaw_radians`, confidence, held; **no pitch/roll** | `src/motion/animation_semantics.py:168` |
| Armature Object (W60) | A·Rz(ψ)·A⁻¹ | `scripts/run_runtime_heading_rig_replay.py:73` |
| Arm chain (W61/W62, W76 IK) | heading-relative chain A·Rz(−ψ)·v → two-bone IK local rotations | `measure_chain`, `retarget.framepose_visual_full_arm_ik.full_arm` |
| BaseRig hierarchy | top-level bones `center_of_mass`, `interaction`, `spine_01`, `thigh_l`, `thigh_r` (no root/pelvis bone); arms hang from spine_05 → clavicle | `examples/e2e_demo/baserig_blender_rest.json` |
| Spine bones | **not driven by any FramePose path** (only a UI mapping hint mentions them) | — |

Nothing in the current pipeline owns trunk pitch or roll. The world arm direction is ψ-invariant by construction, so a
wrong ψ shows up as a wrong torso and wrong arm *local* rotations, not as a wrong world arm.

### Torso frame vs character root orientation

These are two different quantities:
- **Torso frame:** the anatomical frame of the trunk (pelvis→thorax spine axis plus the shoulder or hip bilateral axis).
- **Character root orientation:** what the rig's root or Object should carry.

In W93 the pelvis frame drives the Object, and the thorax-relative twist drives `spine_01`. Because the thighs are
top-level, the legs follow only the Object (pelvis frame), as they already did under yaw-only.

The W93 frames are **camera-relative, not gravity-aligned**. AnimCV has no gravity or ground reference (none was added,
per the directive), so a pitched camera tilts every frame. That includes the GT frame:
- The mean GT camera-relative trunk inclination on *upright* rows is 12.6° on 3DPW TEST, 14.6° on 3DPW VAL and 18.4° on
  3DHP. This is the yaw-only "up-axis error" in §5.
- In the 3DPW walking clip the B (GT frame) mannequin visibly leans by about 25°.

## 3. Definitions (`src/motion/torso_frame.py`, experimental)

**Thorax frame.**
- u = normalize(thorax − pelvis).
- R_t has columns x = normalize(perp_u(R_shoulder − L_shoulder)), y = u × x (chest normal) and z = u.

**Pelvis frame.** R_p is built the same way from the hips.

**Upright case.** For an upright torso R = Rz(ψ) exactly, with ψ as in the W60 convention (facing the camera = 180°).

**Invalid frames.** A frame is invalid (None) when:
- the trunk is shorter than 1e-3;
- the bilateral axis is within 20° of the spine;
- any of the six joints is non-finite.

**Rig mapping.**
- Object = A·R_p·A⁻¹.
- `spine_01` local = Bᵀ·(A·R_pᵀR_t·A⁻¹)·B, where B is the spine_01 rest basis.
- Arm chain = A·R_tᵀ·v through the unchanged W76 IK.

So the world arm is A·v in every mode: the measured maximum difference across modes is 5.5e-13.

**Modes** (identical J1X left arm and identical IK; only the frame changes):
- `Y_S0`: yaw-only with the W91 GT-free heading. This is **A**, the current runtime.
- `Y_GT`: yaw-only with the GT heading (evaluation only).
- `F_GT`: GT full frame (evaluation only). This is **B**.
- `F_S0`: full frame from the frozen S0 joints, falling back to Y_S0 when the frame is invalid or the detector misses a
  hip or shoulder. This is **C**, GT-free.

**Reference.** The GT left-arm chain in the GT thorax frame gives the anatomical local rotations (evaluation only).

**Posture strata.** Strata use the GT trunk inclination: upright < 45°, bent 45–60°, lying ≥ 60°. The GT validity
limit is the same camera-relative caveat as above: the "anatomical" reference is anatomical with respect to the
trunk, not to gravity.

**Inputs.**
- J1X = the W87 predictions `cb9dc51e…` (3DPW) and `356f8cee…` (3DHP).
- S0 = `402d70a2…` (VAL), `4357c7bd…` (TEST) and `a78adeb0…` (3DHP).
- All are hash-pinned; the script STOPs if any changes.

## 4. Gate: the GT full frame (declared before the run)

**Contract.** F_GT − Y_GT upper-arm local error must be ≤ −10° on lying and bent rows, with the sequence-cluster
interval below 0, and upright rows must not get worse by more than +1°.

| Population | lying | bent | upright |
|---|---|---|---|
| 3DHP TEST | **−66.3** [−85.1, −45.5] (n 370) | **−33.8** [−40.2, −30.0] (179) | −7.5 [−9.2, −5.5] (2021) |
| 3DPW TEST | **−50.9** [−58.9, −48.6] (60) | **−40.2** [−45.5, −26.3] (130) | −5.6 [−7.1, −4.1] (5813) |
| 3DPW VAL | **−57.9** [−72.6, −48.0] (89) | **−35.9** [−40.4, −31.0] (180) | −6.9 [−8.8, −4.8] (2787) |

**Gate passed on every population.** The upright gain is real but small, and it is mostly the camera-pitch tilt
(§2) that yaw-only cannot carry.

**Arm > 90° rate on 3DHP lying rows:** 0.42 (Y_GT) → 0.008 (F_GT).

## 5. GT-free candidate C (F_S0): quantitative comparison

**Upper-arm local error vs GT reference** (mean, degrees; forearm in brackets):

| Population / posture | A Y_S0 | Y_GT | B F_GT | C F_S0 | C − A | C − B |
|---|---|---|---|---|---|---|
| 3DHP lying | 89.7 (54.9) | 89.1 | 22.8 (37.0) | 35.7 (40.6) | **−54.1** [−67.5, −37.3] | +12.9 [8.9, 16.5] |
| 3DHP bent | 55.9 | 54.8 | 21.0 | 28.1 | −27.8 [−36.3, −21.6] | +7.1 [3.7, 9.2] |
| 3DHP upright | 28.3 | 26.3 | 18.7 | 22.6 | −5.7 [−7.6, −3.4] | +3.8 [1.9, 6.2] |
| 3DPW TEST lying | 68.2 | 67.7 | 16.8 | 22.9 | −45.4 [−51.5, −40.0] | +6.1 [4.2, 15.1] |
| 3DPW TEST bent | 58.3 | 61.6 | 21.4 | 22.3 | −36.0 [−40.5, −24.2] | +0.9 [−6.4, 4.5] |
| 3DPW TEST upright | 20.3 | 20.6 | 15.1 | 15.8 | −4.5 [−5.4, −3.6] | +0.8 [−0.3, 1.8] |
| 3DPW VAL lying / bent / upright | 76.1 / 50.3 / 17.7 | — | 17.3 / 14.4 / 10.6 | 24.9 / 19.8 / 12.6 | −51.2 / −30.6 / −5.2 | +7.6 / +5.4 / +2.0 |

**Orientation error accounting.**

| | thorax geodesic, mean / p95 / > 135° (A → C) | up axis | forward axis | inclination error |
|---|---|---|---|---|
| 3DHP lying | 93.3 / 137.8 / 5.1% → **37.3 / 111.1 / 1.9%** | 86.5 → 27.6 | 85.8 → 27.2 | 86.5 → 15.3 |
| 3DHP upright | 24.3 → 19.4 | 18.4 → 12.0 | 21.4 → 17.6 | 18.4 → 7.2 |
| 3DPW TEST bent | 59.7 → 21.2 (p95 56.4) | 53.7 → 11.7 | 56.3 → 17.2 | 53.7 → 5.6 |
| 3DPW TEST upright | 18.3 → 14.3 | 12.6 → 7.9 | 16.1 → 13.3 | 12.6 → 5.4 |

**Reading the tables:**
- C's inclination is good: 5–15° error.
- Its remaining error is concentrated in the **chest sign and forward axis on lying bodies**. The 3DHP lying p95 is
  still 111°, and 1.9% of rows are > 135° (fully mirrored).
- On the W92 advisor population (3DHP lying/side-on rows), the S0 frame's chest-up/down label matches GT on only 43
  of 79 answered lying rows.

**Integrity:**
- The world arm direction is identical across modes (≤ 5.5e-13).
- IK residual ≤ 1.4e-13.
- Blender FK vs export ≤ 8.4e-5.

## 6. Most important regressions

**1. Root spins** (pelvis-frame step > 90° while GT < 45°):

| | Y_S0 (A) | F_S0 (C) | of which at C's fallback switch |
|---|---|---|---|
| 3DHP | 9 | **22** (16 lying, 6 upright) | 13 |
| 3DPW TEST | 9 | **15** (11 upright) | 5 |
| 3DPW VAL | 2 | 1 | 0 |

- The C fallback (invalid frame → yaw-only) switches between two different representations, and a lying body snaps
  upright for that frame. 13 of the 22 3DHP spins are at these switches.
- The remaining spins are S0 chest-sign flips.
- Y_GT and F_GT have 0 spins.

**2. Camera-pitch lean.** Every camera-relative frame (B included) tilts upright subjects by the camera pitch. In the
3DPW walking clip the B mannequin leans about 25° and C about 7°. Yaw-only stays vertical, which happens to look right
there.

**3. Not uniformly better on upright clips.** In the 3DPW walking window the upper-arm error is A 8.8°, C 10.1° and B
17.2°. There the J1X arm error happens to align with the yaw-only frame. The aggregate upright contrast is still
negative (−4.5° on TEST).

**4. Invalid frames.** C falls back on 60 (VAL), 175 (TEST) and 29 (3DHP) rows.

## 7. Real-scene Blender A/B

**Exports.** `scripts/run_torso_frame_rig_replay.py` exports fixed-rule windows in the W91 renderer format, with these
slots:
- FIXED = A (Y_S0);
- RUNTIME = C (F_S0);
- GT = B (F_GT).

Each frame carries the Object rotation, the `spine_01` local and the IK arm locals.

**Rig.** The rig is the skinned mannequin `examples/skinned_mannequin/BaseRig_Skinned.fbx`. Its rest bones equal the
BaseRig snapshot within 6.1e-5, and it is bound by a numeric equivalence check (`--equivalent-rig-tolerance 1e-3`)
because the snapshot digest hashes the FBX provenance.

**Rendering.** Constant interpolation, real time, side view in the second row. Re-rendering a clip is pixel-identical
(max diff 0 over 216 frames).

| Clip | A | C | B | Observation |
|---|---|---|---|---|
| 3dhp_ts2_supine | 116.4 | 91.4 | 51.1 | B lies on its back; C lies but flips chest up/down between rows (4 fallback rows); A stands |
| 3dhp_ts4_lying | 45.3 | 23.8 | 22.9 | C ≈ B: subject bent over at the waist (GT trunk ≥ 60°), so the mannequin's trunk is horizontal with the arm hanging; A stands upright |
| 3dhp_w92_introduced | 53.0 | 31.7 | 22.3 | the W92 window with the advisor's introduced error |
| 3dhp_upright_lying_transition | 47.4 | 19.1 | 14.3 | C stays at about 10–25° through the window; A's error grows to about 120° in its second half |
| 3dpw_bent_or_seated | 56.1 | 18.1 | 8.7 | C/B bend forward over the bag; A stays upright with the arm sticking out |
| 3dpw_turn_180 | 23.6 | 20.8 | 19.5 | turning is preserved |
| 3dpw_walking_upright | 8.8 | 10.1 | 17.2 | B leans about 25° (camera pitch); see §6 |

(Mean upper-arm local error vs the GT reference over each window, in degrees.)

**Outputs.**
- Clips and PNGs: `~/animcv-output/93_framepose_torso_frame/` (7 mp4 + 7 png).
- Reports, cases and .blend files: archived on LabServer63 in `~/animcv-output/framepose_torso_frame_archive/`.

## 8. Runtime cost and confidence behaviour

**Cost.** `torso_frames` takes about 44 µs per frame (CPU, numpy), on top of the S0 joints the runtime already has.
No model and no extra inference are needed.

**Confidence behaviour.** The only policy is geometric validity plus detector visibility of the hips and shoulders,
with a yaw-only fallback. There is **no chest-sign reliability signal**:
- a mirrored lying frame is reported as valid;
- the fallback itself creates spins (§6).

## 9. W92 advisor: a legitimate integration point (diagnostic only, not integrated)

**Diagnostic** (3DHP rows where the advisor was invoked and the S0 frame is valid):
- The VLM answered 87 rows, abstaining on 406 where the prompt order disagreed or the answer was UNCLEAR.
- It matches the GT chest-up/down label in **82/87** rows. The S0 frame matches in **51/87**.
- The two disagree in 41 rows, and the VLM is right in 36 of them.

**Interface.** The advisor's BACK/STOMACH answer is exactly the missing chest-sign evidence: a frame-level constraint
on the sign of y_t·Z. Mirroring the frame about the spine axis is the full-frame counterpart of the W92 ψ mirror, and
unlike ψ it can put the chest up.

That would be a new mechanism (sign policy, abstention and temporal consistency), and it is outside this batch. It was
not implemented.

## 10. Tests and files

**Tests:**
- `tests/test_framepose_torso_frame.py` (10 tests):
  - orthonormal right-handed frames;
  - upright frame = yaw contract (facing the camera = 180°);
  - supine/prone fixtures with inclination and chest sign, and yaw-only unable to express them;
  - degenerate → None;
  - quaternion round trip on every Shepperd branch;
  - world arm-direction invariance, and full-frame IK with R = Rz(ψ) identical to the W91 IK;
  - `spine_01` conjugation and the yaw-only baseline (W60 Object rotation, identity trunk twist);
  - GT-free F_S0 input and the fallback, with no production import;
  - hash-pinned frozen inputs and sequence-cluster statistics;
  - renderer/composer defaults unchanged.
- Together with the W91/W91A/W92 suites: 28 passed locally and 28 passed on LabServer63 (docker `animcv-external:cu124`).

**New files:**
- `src/motion/torso_frame.py` (experimental, unwired).
- `scripts/run_torso_frame_rig_replay.py`.
- `tests/test_framepose_torso_frame.py`.
- This worklog.

**Changed (backward compatible, defaults unchanged):**
- `scripts/render_runtime_heading_rig_blender.py`:
  - new options `--equivalent-rig-tolerance`, `--show-mesh`, `--camera-centre-height` and `--ortho-scale` (default 2.2);
  - keys optional `extra_bone_locals_xyzw` (`spine_01`) and the frame-level `object_rotation_xyzw` / `forward_cam`.
- `scripts/compose_runtime_heading_rig_video.py`:
  - optional per-frame `image` and `mode_lines`;
  - the header names the skinned mannequin when the Blender report says `show_mesh`.

## 11. CASE and recommendation for the next architecture decision

**CASE B.**
- The representation's effect on the actual rig is large and consistent on all three populations.
- The GT-free S0 frame gets most of it on average.
- It is not reliable in the conditions that motivated the work:
  - chest-sign flips on lying bodies (p95 111°);
  - more root spins than yaw-only, mostly from its fallback switching;
  - camera-pitch lean with no gravity reference.

Runtime readiness is not claimed.

**Missing evidence:**
1. A chest-sign (BACK/STOMACH) reliability signal for the frame. The W92 advisor provides exactly this on the rows it
   answers.
2. A continuity policy that does not alternate between full-frame and yaw-only representations.
3. A gravity or camera-pitch reference if upright subjects must stay vertical.

**Recommendation for the next architecture decision** (required by the directive): the decision to make is whether
AnimCV's root orientation contract should grow from yaw-only into a torso frame owned by the Object (pelvis) and
`spine_01` (twist), as replayed here. If it should, the open prerequisite is the reliability evidence listed above,
not the frame geometry, which is verified. If it should not, yaw-only remains the contract, and lying/bent animation
quality stays bounded by the gate numbers in §4.
