# Worklog 90 — End-to-End Rig Animation Replay of RGB Depth Stabilization

Date: 2026-10-17
Branch: `arch/single_frame_first`
Starting HEAD: `1c7084a` (Worklog 89). Local, origin and LabServer63 were in sync and the worktree was clean.

**Type:** bounded, opt-in experimental integration plus a real Blender replay. Nothing below was changed:
- no model was trained;
- the W89 stabilizer code and its config are untouched;
- no production FramePose, IK, BaseRig or Blender-writer code changed;
- no data was downloaded.

## 0. Answer

**Engineering CASE B: limited rig-level benefit.**

The W89 offline stabilizer was inserted between frozen J1X and the existing W61/W76 left-arm IK and played on the
actual BaseRig in Blender.

**What it does well** (3DPW TEST; VALIDATION 0.1 s agrees):
- The rig moves more coherently:
  - Frame-to-frame local-rotation error against the GT reference falls by 2.1° (upper arm) and 2.7° (forearm) per
    step.
  - Spurious rotation discontinuities fall by 74% (upper arm) and 50% (forearm).
  - Upper-arm angular acceleration falls 567 → 384 deg/s² (GT 272).
- Static wrist and elbow accuracy improve slightly (wrist −0.32 cm, elbow −0.17 cm).
- No IK, pole or lag regression.

**Why it is not CASE A:**
- **Remaining jitter.** Still-arm rotation jitter is reduced by only about a quarter: upper arm 44 → 32 deg/s, against
  GT 13 and G0X 22.
- **Peak rounding.** Fast rotations lose 5–7% of amplitude.
- **Severe regressions.** On VALIDATION, 1.9% of frames get a > 5 cm worse wrist, above the predeclared 1%.

**What the clips show:**
- The visible defects of these replays are dominated by **frozen J1X reconstruction errors**, which the stabilizer does
  not touch:
  - an upper arm swung behind the torso;
  - multi-frame depth excursions;
  - slow drift.
- The coarse 6–10 Hz sampling also limits what can be judged.
- Fewer single-step jumps are visible, but at normal playback speed the improvement is modest, not transformative.

Keep the stabilizer as an opt-in offline option, default off. Do not adopt it as a production default.

## 1. Existing pipeline (first gate)

Audited from code and docs 58–61 and 74–76. Route used here (all existing, unchanged):

- **Source:** J1X unit chain (P=0, M=ρu, D=M+(1−ρ)v; canonical camera axes X right, Y depth, Z up).
- **Measurement:** `retarget.framepose_ik_gt_attribution.measure_chain((0,0,0), M, D, ψ, A)`.
  - This is the W61 `heading_relative` transform A·Rz(−ψ)·v. A comes from `derive_rig_alignment`, using the BaseRig rest
    landmarks.
  - It gives reach r = |D| (dimensionless), the endpoint direction and the observed bend side.
- **Solve:** `retarget.framepose_visual_full_arm_ik.full_arm` (W76), which applies the W61 closed-form two-bone solve:
  - anchored at the BaseRig `upperarm_l` rest head;
  - rig rest lengths L1 = 27.771 and L2 = 27.251 armature units (cm);
  - requested endpoint = root + r·(L1+L2)·direction, reach clamped to [|L1−L2|, L1+L2];
  - the runtime bend side, re-perpendicularized; near-straight arms fall back to the rest plane.
  - **No GT and no H0 length enters.** This is the same call path W80 used for J1X.
- **Rotations:** W61 zero-twist, parent-relative local quaternions for `upperarm_l` and `lowerarm_l`. FK closes the loop.
- **Root:** W60 object ownership, Armature Object rotation = rest·A·Rz(ψ)·A⁻¹. Translation is never written.
- **Supported region:** only the left arm is wired in `examples/e2e_demo/framepose_ik_chains.json`.
  - The right arm would need a new chain entry, so it was not added.
  - No leg IK exists, and none was built.

**Contract issues found**

1. **The historical runtime yaw is not GT-free.**
   - ψ comes from H0 (`current_root_orientation`), and H0's materialization uses two **oracle** torso sign fields
     (`shoulder_forward_depth` and `hip_forward_depth`, `sign_source: oracle`; docs/75 l.39 and l.84).
   - Using it would leak GT into the "runtime" replay.
   - **Resolution:** one constant ψ, identical for all modes. Primary ψ = 180°, so the rest torso faces the camera and
     the video-camera view of the rig is directly comparable with the RGB. Sensitivity run at ψ = 0°.
   - For any constant ψ the arm's world (camera-frame) direction is identical (tested). ψ only changes the torso frame
     the local rotations are expressed against.
   - **Consequence:** the torso does not turn with the subject. This is an **arm-only replay against a static rest
     torso**, not a whole-body reconstruction.
2. **No FramePose animation writer exists.**
   - The legacy `keyframe_writer` takes world-delta `AnimationClip`s (docs/58 l.18), which is a different contract.
   - Earlier FramePose Blender runs were isolated-pose PNGs.
   - An experimental Action writer was added inside the review script only: keyframes on the two arm bones at the
     exact bank frame indices.
3. **No skinned mesh.** `examples/BaseRig.fbx` contains one armature (51 bones) and no mesh (checked in Blender 5.1.2).
   The render shows the posed BaseRig armature through bone proxies. That is the "rigged character movement" this rig
   supports.

None of these blocks a faithful left-arm replay, so the gate passed. (1) and (2) remain integration blockers for
production (§11).

## 2. Replay inputs, changes and preserved artifacts

**Inputs** (hash-checked; STOP on mismatch):
- Frozen W87 `predictions_3dpw.npz` (`cb9dc51e…`): J1X = left-arm `g1`, G0X = left-arm `g0`.
- W89 `config.json` (hash equal to the W89 validation report). The W89 stabilizer code is unchanged.
- BaseRig rest snapshot. Its `source_fbx_sha256` must match `examples/BaseRig.fbx`.

**Parity:** the stabilized chains reproduce W89's distal static error exactly (diff 0.0 on TEST and VALIDATION).

**Modes:**

| mode | input chain | then |
|---|---|---|
| RAW | J1X | IK |
| STABILIZED | J1X → W89 stabilizer | IK |
| G0X | G0X | IK (reference) |
| GT_REF | GT chain | IK; **evaluation only**, rotational reference and green overlay |

- All four use identical rig, anchor, lengths, ψ, alignment and frames.
- Frames whose detector arm joints are invalid are **UNAVAILABLE in every mode**.
- The stabilizer is switchable: `cfg=None` reproduces the RAW rig poses exactly (tested).

**New files** (no existing file modified):

| file | role |
|---|---|
| `scripts/run_stabilized_rig_replay.py` | numeric replay: modes → IK, rig-level metrics, predeclared `DECISION`, case export |
| `scripts/render_stabilized_rig_replay_blender.py` | Blender: imports BaseRig, keys one Action per mode, renders video-camera and side views, saves `.blend` |
| `scripts/compose_stabilized_rig_replay_video.py` | side-by-side MP4 review package |
| `tests/test_framepose_stabilized_rig_replay.py` | 7 focused tests |

**Outputs:**
- Server: `framepose_rig_replay_facing/` (primary, ψ=180), `framepose_rig_replay_yaw0/` (sensitivity),
  `framepose_rig_replay_blender/` (13 `.blend` files with three Actions each, `blender_report.json`, videos).
- Review package (MP4 + PNG): `~/animcv-output/90_framepose_rig_replay/`.

## 3. Scene selection and frame accounting

Rules were declared in `DECISION` before any rig output existed.

**TEST** (7 cases): the exact W89 review windows, 36 rows = 6 s at 1/6 s.
- stationary_limb (bus_00 a1 1692)
- historical_false_motion (walking_00 540)
- walking_arm_swing (downstairs_00 90)
- toward_away_camera (rampAndStairs_00 139)
- fast_arm_movement (fencing_01 582)
- worst_static_regression (fencing_01 90)
- worst_regression_left_arm (fencing_01 732)

**VALIDATION** (6 cases): the W89 rules applied to 60-row windows (6 s at 0.1 s).
- stationary_limb (drinking_00 a1 508)
- historical_false_motion (drinking_00 a1 292)
- walking_arm_swing (drinking_00 a0 504)
- toward_away_camera (parcours_01 1146)
- fast_arm_movement (rangeOfMotions_01 108)
- worst_regression_left_arm (drinking_00 a1 490)

Two VALIDATION windows hold 23 and 27 UNAVAILABLE rows, because the inherited "≥ 30 GT-valid rows" rule is loose for
60-row windows. They are kept as selected and flagged in the clips.

**Population metrics** use every row, not only the windows.

| population | rig frames scored | pairs | sequences | UNAVAILABLE rows (detector) |
|---|---|---|---|---|
| TEST | 6,003 | 5,504 | 37 | 1,073 |
| VALIDATION 0.1 s | 3,056 | 2,922 | 16 | 351 |

**Timing:**
- Keys sit at the exact bank frame indices: Blender frame = frame_index − first + 1, scene at 30 fps.
- Interpolation is CONSTANT, so nothing is interpolated between model samples.
- Clips play only the sampled rows, in real time: 6 fps for TEST, 10 fps for VALIDATION. No 30 fps model output is
  claimed.
- UNAVAILABLE rows hold the previous pose in every mode.
- VALIDATION windows are J1X checkpoint-selection data (W85); TEST rows are the clean set.

## 4. IK and rig compatibility

| check | result |
|---|---|
| IK endpoint reconstruction residual | 0 |
| IK→FK residual | ≤ 3.5e-13 |
| Bone length residual | 0 |
| Unit quaternions | yes |
| Blender FK of the **keyed Action** vs the exported IK endpoint | ≤ 4.2e-5 cm (all 13 cases, 3 modes) |
| Re-rendering a case | pixel-identical (max diff 0) |
| Unmeasurable or degenerate rows | none among available rows |

- Spurious pole flips are rare and not increased: TEST 0.29% → 0.16% (G0X 0.25%).
- Near-straight arms (GT bend < 15°) behave better under STABILIZED: upper-arm residual 7.87 → 6.39° (G0X 6.36).

## 5. Rotation-level results

Primary run, ψ = 180°. Each cell is RAW / STABILIZED / G0X.

- **Relative-rotation residual** = |rv_pred − rv_GT|, where rv is the rotation vector of each bone's local rotation
  from one sampled frame to the next. It is in degrees per step, against the GT-reference arm through the same IK.
- **Intervals** are the W75/76 sequence-cluster bootstrap of STABILIZED − RAW.

| metric | TEST | VALIDATION 0.1 s |
|---|---|---|
| upper-arm rotation residual | 9.89 / 7.77 / 6.90 (Δ −2.11 [−2.26, −1.98]) | 8.54 / 6.01 / 7.27 (Δ −2.53 [−2.95, −2.00]) |
| forearm rotation residual | 17.00 / 14.29 / 13.68 (Δ −2.71 [−2.99, −2.45]) | 14.30 / 10.60 / 12.53 (Δ −3.69 [−4.72, −2.51]) |
| p95 upper / forearm | 25.4 / 19.7 / 18.3 ; 44.0 / 36.3 / 37.5 | 23.5 / 16.0 / 20.1 ; 38.2 / 28.9 / 35.9 |
| upper-arm angular speed, deg/s | 64 / 50 / 43 (GT 42) | 99 / 74 / 80 (GT 69) |
| still-arm (low-GT-motion) upper speed | 44 / 32 / 22 (GT 13) | 51 / 32 / 32 (GT 17) |
| still-arm forearm speed | 76 / 59 / 44 (GT 22) | 82 / 55 / 60 (GT 28) |
| upper-arm angular acceleration, deg/s² | 567 / 384 / 342 (GT 272) | 1355 / 780 / 975 (GT 542) |
| rotation discontinuities, upper (> 30° while GT < 10°) | 1.05% / 0.27% / 0.25% | 1.20% / 0.14% / 0.62% |
| rotation discontinuities, forearm | 3.71% / 1.87% / 1.60% | 2.60% / 0.82% / 2.12% |
| elbow-flexion change residual, deg | 8.65 / 7.07 / 6.44 | 7.53 / 5.83 / 6.42 |
| spurious pole flips | 0.29% / 0.16% / 0.25% | 0.21% / 0.03% / 0.27% |

**Sensitivity, ψ = 0°** (torso facing away): same direction and size.
- TEST: upper Δ −2.30, forearm Δ −3.11; discontinuities 1.40 → 0.35% (upper) and 4.76 → 2.05% (forearm).
- Same CASE (B).

**Interpretation:**
- At rig level the stabilizer gives coherent, significant gains, and on VALIDATION it beats G0X's rotation stability.
- On TEST, G0X's local rotations are still steadier (6.90 vs 7.77 upper), but G0X's poses are less accurate (§6).
- Still-arm rotation speed stays 2.5× GT on TEST. That is the rig-level face of W89's unresolved slow drift.

## 6. Depth-motion and static accuracy (rig cm)

Each cell is RAW / STABILIZED / G0X.

| metric | TEST | VALIDATION 0.1 s |
|---|---|---|
| wrist camera-depth motion residual / step | 5.97 / 4.28 / 4.25 | 5.28 / 2.91 / 3.91 |
| … on still arms | 4.13 / 2.49 / 1.50 | 3.15 / 1.37 / 1.30 |
| wrist-depth tracking slope, high-motion pairs | 0.506 / 0.439 / 0.282 | 0.588 / 0.552 / 0.216 |
| wrist error | 11.12 / 10.80 / 12.23 (Δ −0.32 [−0.37, −0.28]) | 9.07 / 8.61 / 11.37 (Δ −0.46) |
| wrist p95 | 28.1 / 27.0 / 30.6 | 23.8 / 22.2 / 28.2 |
| elbow error | 7.15 / 6.97 / 7.52 (Δ −0.17) | 5.13 / 4.88 / 6.92 (Δ −0.25) |
| upper-arm / forearm direction error, deg | 14.92 / 14.55 / 15.71 ; 15.49 / 15.13 / 18.24 | 10.66 / 10.12 / 14.47 ; 13.95 / 13.31 / 17.63 |
| bend error, deg | 13.90 / 13.70 / 16.15 | 9.38 / 8.96 / 13.97 |
| severe wrist regressions (> 5 cm worse than RAW) | 0.40% | **1.93%** (criterion 1%) |
| best wrist-depth lag, rows | 0 / 0 / 0 | 0 / 0 / 0 |

The RGB static advantage over G0X carries through IK and is fully kept.

## 7. Peak attenuation and false holds

- **Fast pairs:** GT-reference relative-rotation speed ≥ 120 deg/s.
- **Projected amplitude:** the component of the predicted rotation along the GT rotation axis, per unit GT. Isotropic
  noise does not inflate it.

| | TEST upper (n=250) | TEST forearm (n=878) | VAL upper (n=521) | VAL forearm (n=790) |
|---|---|---|---|---|
| projected amplitude RAW / S / G0X | 0.611 / 0.578 / 0.579 | 0.671 / 0.627 / 0.664 | 0.744 / 0.718 / 0.636 | 0.676 / 0.669 / 0.712 |
| S as share of RAW | 95% | 93% | 97% | 99% |
| false holds RAW / S / G0X | 0.216 / 0.192 / 0.200 | 0.174 / 0.163 / 0.170 | 0.157 / 0.098 / 0.186 | 0.197 / 0.146 / 0.203 |

**The predeclared "motion" criterion fails, and part of that failure is a metric defect.**

- **What the criterion measures:** retention (S − G0X)/(RAW − G0X) ≥ 0.8.
- **Why it is degenerate on TEST.** J1X has almost **no rotation-level fast-motion advantage over G0X** (0.611 vs 0.579;
  0.671 vs 0.664). The ratio's denominator is near zero, so it is meaningless: −0.02 and −5.4.
- **On VALIDATION it measures something real:** upper arm 0.76 (fail); forearm denominator negative.
- **Directly:**
  - STABILIZED keeps 93–99% of RAW's fast rotation amplitude.
  - False holds *decrease* (fewer noisy counter-movements).
  - There is no lag.
- **Wrist depth:** the fast-motion slope falls 13% on TEST (0.506 → 0.439), as in W89.

**Peak rounding is real but small at rig level.**

## 8. Blender replay and review package

Video files are at `~/animcv-output/90_framepose_rig_replay/<split>_<case>.mp4`, 13 clips plus a summary PNG of each
last frame. Each frame of a clip shows:

- **Top row:** input RGB (frame, time, detector arm confidence, UNAVAILABLE flag), then RAW J1X, STABILIZED J1X and G0X
  rendered by Blender from the video-camera direction.
- **Bottom row:** wrist-depth and upper-arm rotation-change trajectories with a cursor, then the three modes in the side
  view (depth × up).
- **On every rig tile:** the per-step upper-arm and forearm rotation change and the wrist-depth change for that mode and
  for GT. The thin green arm is the GT reference (evaluation only).
- **Render settings:** identical rig, cameras (orthographic, fixed per case), Workbench studio lighting, 480 px tiles and
  timing for all modes.
- **`.blend` files** with the three Actions are on the server (`framepose_rig_replay_blender/blend/`).

**Observations from the clips.** The origin of each effect is given in brackets.

| clip | what is seen |
|---|---|
| test_stationary_limb | Hand on hip. RAW twitches: upper-arm steps up to 18° per 1/6 s (GT about 1°). STABILIZED halves most spikes, but the wrist depth still wanders 13 → −3 → 13 cm over 2 s **[reconstruction drift; stabilizer cannot remove it]**. G0X is calmest but sits about 13 cm too shallow **[G0X static depth]** |
| test_historical_false_motion | In side view, RAW and STABILIZED both swing the upper arm **behind the torso** (GT hangs forward). One multi-frame excursion with the upper arm near vertical appears in both **[reconstruction]**. Hardly any visible difference between RAW and STABILIZED in this window |
| test_walking_arm_swing, test_toward_away_camera | STABILIZED follows the swing phase. Depth peaks about 1–2 cm lower. Fewer single-step jerks. No new elbow or pole artifact |
| test_fast_arm_movement, test_worst_regression_left_arm | Fencing lunges. Rotation peaks (about 95° per step) are kept in both modes. Both miss part of the GT depth excursion **[reconstruction]** |
| test_worst_static_regression | Final lunge: GT wrist depth +30 cm, RAW +10, STABILIZED +8. A small extra peak loss **[stabilizer]** on top of a large reconstruction miss |
| validation_stationary_limb, validation_worst_regression_left_arm | The first 2.3 s are UNAVAILABLE (detector) and held **[observation; window rule]**. After that, a near-straight arm is stable in all modes |
| validation_walking/fast/toward_away | Same pattern as TEST. STABILIZED is the steadiest of the three in rotation, with peaks kept |

**What limits these clips as an animation judgement:**
- 6–10 Hz sample playback without interpolation looks stepwise in every mode.
- Only the left arm moves; the torso is the rest pose; there is no mesh.

The stabilizer's benefit shows as fewer and smaller jumps between steps. It does not turn these clips into smooth
animation.

**No rig artifact attributable to the stabilizer was found:**
- no pole flips introduced;
- no elbow inversions;
- no IK failures.

## 9. Practical cost

| item | cost |
|---|---|
| Integration | about 40 lines of glue in front of the existing IK call (stabilize the chain, then the unchanged `measure_chain`/`full_arm`) |
| Stabilizer | about 70 µs per frame per chain (W89). IK cost is unchanged and identical across modes |
| Lookahead | offline; the whole track (until a > 1 s gap or a track change) |
| Memory | W89 forward-pass storage, about 14 floats per frame per segment |
| Blender export and render | keyframes on 2 bones per frame. About 5 s per 6 s case for 3 modes × 2 views (216 Workbench renders, 480 px, including Blender start-up) on the local Mac; LabServer63 has no Blender |
| Version coupling | W89 σ/q are tied to the J1X checkpoint (recalibrate if it changes); the rig is tied to the BaseRig rest snapshot digest |
| Failure and fallback | `cfg=None` gives the exact RAW output. Invalid detector frames get no measurement update in the stabilizer and are held in the replay |
| Reproducibility | deterministic numeric output and pixel-identical renders |

## 10. CASE

Predeclared rule (`DECISION`, `classify()`): **B**.

| population | rotation_stability | artifacts | motion | position |
|---|---|---|---|---|
| TEST | ✓ (both bones, intervals below 0) | ✓ (rates lower) | ✗ (degenerate ratio, §7) | ✓ |
| VALIDATION | ✓ | ✓ | ✗ (upper retention 0.76) | ✗ (severe 1.93%) |

The sensitivity run at ψ = 0° gives the same.

**Problem origin, as asked in §8 of the direction:**

| problem | origin |
|---|---|
| Large static and multi-frame depth errors, upper arm behind the torso, slow drift | frozen reconstruction (J1X) |
| Peak rounding of 5–7%; about 2% severe wrist regressions on VALIDATION | stabilizer |
| Held poses | detector observation (runtime unavailability) |
| Stepwise look | 6–10 Hz sampled bank; evaluation and rendering limitation |
| Static torso | constant-yaw contract; evaluation limitation |
| IK or retargeting artifacts | none found |

## 11. Remaining integration blockers

1. **No GT-free runtime root yaw.** H0's yaw depends on oracle torso sign fields. Any whole-body rig replay needs a
   runtime yaw source first.
2. **No production FramePose animation writer.** The Action writer here is experimental and review-only.
3. **BaseRig has no skinned mesh**, so character-level appearance cannot be judged.
4. **Only the left arm is wired.** The right arm needs a chain entry; legs need an IK path.
5. **Native cadence is untested.** The replay ran at the 3DPW bank cadence (6–10 Hz). A production video would be
   reconstructed every frame (25–30 fps), and the stabilizer smooths more at denser sampling (W89: much stronger
   effect at 50 fps on 3DHP TS5–TS6). Behaviour at native cadence on 3DPW is untested.
6. **Real runtime inputs are needed:** track keys and timestamps from a real tracker and video.

## 12. Production adoption recommendation

**Do not enable by default and do not integrate into production now.**

**Keep as an opt-in, offline-only option.**
- Suited: animation of standing, walking and moderate arm motion, where single-frame depth pops and rotation jumps are
  the visible defect. It reduces jumps and slightly improves pose accuracy.
- Unsuitable:
  - athletic or fast motions where peak depth matters;
  - still arms with slow drift, which it does not fix;
  - causal or real-time use.

**The next useful bounded task would be a native-cadence replay** (decision left to the owner; not started). Reconstruct
every frame of 1–2 of these clips (frozen Qwen + J1X inference only, no training), then repeat this RAW vs STABILIZED rig
A/B at 30 fps. This would show whether the effect becomes clearly visible at real playback cadence.

The GT-free root-yaw contract (blocker 1) is the larger prerequisite for any whole-body use.
