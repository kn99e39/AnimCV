# Worklog — Licensed CondMDI Partial-Joint Completion of the S1 Left Arm on the Real BaseRig Action (2026-11-09)

Batch 131. An experimental, quality-first batch on `arch/single_frame_first`. It ran unattended on Main:

- local RTX 5080 (16 GB), 64 GB RAM, Windows 11 Pro 10.0.26200;
- AnimCV Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9), unmodified;
- a separate scratch CondMDI runtime outside the repository: uv-managed CPython 3.12.16, torch 2.9.1+cu128, numpy 2.2.6,
  safetensors 0.8.0, einops 0.8.2, mmengine 0.10.7, OpenAI CLIP package at `d05afc4`;
- Blender 5.1.0.

`docs/next_batch.md` was read only. No LabServer and no other machine. Network use was limited to:

- `git fetch/push`;
- code-only clones of the original CondMDI and Motius repositories;
- Hugging Face metadata plus the pinned checkpoint artifact (1.3 GB);
- Google-Drive range reads of the three official CondMDI release zips (`args.json` only);
- one full download of the official `condmdi_random_joints.zip` (1.7 GB), used for tensor identity comparison only;
- the scratch `uv pip` installs.

Every number, render and verdict below is **AGENT-RUN** on Main. The visual judgement is the agent's own blinded review
of still contact sheets. It is not human or animator validation, and native-rate (30 fps) perceptual review was
**NOT RUN**.

## 1. Official CASE and scope

**Verdict: CASE B under the frozen rule.** Source: `results/assess/assessment.json`, SHA-256
`9633a3621065058f3cd0d23af1a0f6fed2baf507b578c5c00c510ed6307404ac`.

**What CASE B means here:**

- a licensed, identity-verified real pretrained model ran;
- it was conditioned on honest partial source evidence: UNKNOWN features were provably masked, and the leak control
  was bitwise;
- it produced a safe, separate, editable Blender Action that preserves every out-of-domain track and the hard pin;
- but it showed **no quality superiority**.

The CondMDI candidate C is **worse** than the simple slerp B on every quantitative source-fidelity and continuity
criterion, and markedly worse at the interior entry. A negative quality result.

**Gates** G0–G7: all PASS.

| QA | Criterion (declared before output) | C | Limit / reference | Result |
|---|---|---|---|---|
| QA1 | 0 arm-local reversal flicks | 4 | 0 | FAIL |
| QA2 | arm-local entry/exit C0 step ≤ 2×B + 10 mm | 429 / 153 mm | 58.2 / 60.0 mm | FAIL |
| QA3 | no near-straight, no bend flip, no hinge-opposite frame | none | — | PASS |
| QA4 | mean world wrist deviation from A ≤ 1.25×B + 25 mm | 292 mm | 209 mm (B 147.5) | FAIL |
| QA5 | press intent: no dip > 50 mm below the endpoint wrist-above-shoulder height | rises (dip −183 mm) | 50 mm | PASS |
| QA6 | blinded agent review favours C | C named for a flick, ranked least faithful, not read as a press | — | FAIL |

**Scope.** One opt-in scratch experiment on one source (S1).

- The candidate `.blend` files are **SCRATCH PROPOSALS**. They were never committed, accepted or placed in a lineage.
- No Production, CLI, GUI, exporter, FramePose, MotionGraph, FK/IK, schema, dependency-lock or training change.
- Historical CASEs are untouched: W121 D, W122 B, W123 B, W124 B, W125 A (transaction capability only), W126 C, W127 B,
  W128 DATA BLOCKED, W129 B, W130 DATA BLOCKED_LICENSE.
- No historical Worklog was edited.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `a06baf169f296195561dedcf123de506ffe5853b`, verified first |
| Starting commit | `d2e68f1a3e7c2148ad81d3d8256b6b48c67ed697`. Equal to `origin/arch/single_frame_first` at the first fetch. W130 carries the COMPLETE marker, and `docs/AUDIT_W126_W130_2026_11_09.md` is present. |
| Preflight commit | `f34cb3d` (`scripts/w131_condmdi_preflight.py`), committed **before any weight download** |
| Instrument commit | `fb29aa9`: adapter, runner, harness, tests (pre-outcome) |
| Declaration commit | `6ef2726` (`examples/experimental_condmdi/w131_declaration.json`, byte copy of `results/declare/declaration.json`, SHA-256 `dca4a891d21725f813b8cbb1867abaab8f237f59e6f36080086b2267355107e4`; `outcomes_inspected: false`, `primary_model_output_exists: false`; binds the LF SHA-256 of all eight instrument files) |
| Final source revision | `6ef2726`. The Worklog commit is documentation only. The COMPLETE marker is a separate, final commit. |
| Scratch root | `C:/Users/dna10/animcv-w131`, outside the repository |

**Pre-outcome instrument changes (disclosed).** The first `controls` run (`results/controls_attempt1_control_design_error`,
`85259824…`) failed G1 on **two control-design errors** of mine. No S1 model output existed at that point.

1. The "asymmetric" control rotated `upperarm_l` about its local Y axis. That is the bone axis, so the rotation is a pure
   twist, and the elbow correctly does not move. Fixed: a local-X swing.
2. The "parent-child" control required the legs' HML263 positions to be unchanged by a spine bend. But the official
   HumanML3D root heading uses the shoulders, so a 20° spine bend legitimately turns the heading by 5.4°. Fixed: the legs
   are checked in rig-world positions, and the heading change is reported as information.

A third, invalid control ("proper-rotation handedness") was removed from the tests and the harness. A proper
up-preserving rotation is only a yaw, and canonicalisation removes it. Only a mirror can flip handedness, and mirrors
are refused.

## 3. Rights, source identity and runtime (G0)

**Rights preflight.** Source: `results/preflight/preflight.json` (`a2ceb54a…`; status `OK_PENDING_IDENTITY`). The rule was
declared in the script header before any weight request, and every predicate was true:

- (a) the checkpoint card `license: mit` (API `cardData` and front matter);
- (b) the card's provenance is the "official randomly sampled frames-and-joints checkpoint" of the original CondMDI, with
  **no** HY-Motion, Hunyuan or Tencent mention;
- (c) the original CondMDI code is MIT at `0c58473`, and Motius `d04a38c` retains the CondMDI MIT licence and the
  OpenAI-CLIP MIT licence (`motius/licenses/condmdi/`);
- (d) no applicable licence text contains a territorial exclusion;
- (e) no SMPL/SMPL-X body model is needed.

**Recorded but not judged:**

- The training data is HumanML3D (repository licence MIT), derived from AMASS, whose terms are non-commercial research.
- The declared use is local, non-commercial research evaluation. No dataset, AMASS or SMPL asset was acquired.
- The execution locale is again Korea (GeoId 134). That is not a blocker for MIT terms.

This is the agent's documentary reading, not legal advice.

**Checkpoint identity, resolved before any inference.** `ZeyuLing/motius-condmdi-humanml3d` @
`1e84d9ebf9a2562a2e3aa01926b9fb3606bd23de`.

| File | Bytes | LFS SHA-256 (verified after download) |
|---|---|---|
| `model.safetensors` | 940,512,684 | `3d9e299f6096405c028b3b8b9f8a663794a1e938bfae8ff27102caaa0e1c0e29` |
| `clip.safetensors` | 353,470,708 | `431adb635d7a35aaaf5b70acf8738712c6f9ae30d08e17698c6f505c9a686fb8` |
| `Mean_abs_3d.npy` / `Std_abs_3d.npy` | 1,180 each | `1bb2633e…` / `9f5f12ca…` |

- **Metadata discrepancy found.** The HF `condmdi_config.json` says `keyframe_selection_scheme: random_frames` and
  `save_dir: save/xpnzs9fw`.
  - Range reads of the three official release zips show that `xpnzs9fw` is the **unconditional** release's run.
  - The random-joints release is `snjua2bq`; random-frames is `nm4d9951`.
  - The Motius exporter (`tools/export_condmdi_hf.py`) hard-codes `keyframe_selection_scheme: random_frames`.
  - The HF safetensors header nevertheless shows a keyframe-conditioned network (first conv 526 = 2×263 input channels).
- **Tensor identity** (`results/identity/identity.json`, `d0fd115b…`): the official `condmdi_random_joints.zip`
  `model000750000.pt` was downloaded and compared tensor by tensor. HF `model.safetensors` is **bitwise identical
  (196/196 tensors)** to its `model_avg` (EMA) weights. Against the non-EMA `model` weights, 0/196 are equal (max diff
  0.064).
- **Conclusion.** The weights are the official CondMDI trained on random **frames and joints**
  (`keyframe_selection_scheme: random_joints`). The HF config's scheme and `save_dir` fields are an exporter labelling
  artefact; network hyper-parameters are identical across the three official `args.json` files. The extracted official
  checkpoint was deleted after the comparison. The zip remains in scratch.
- **Normalisation statistics** come only from the HF artifact; the official zips do not contain them.

**Independent smoke** (`results/smoke/result.json`, `8f710d2c…`; **not AnimCV quality evidence**):

- Text-to-motion with the model-card example: 120 frames, finite, pelvis 0.93 m, 1.89 m travel, forearm 0.254–0.274 m.
- Mask contract on that smoke output: observing lower-body positions on 60 frames, an exact rerun was bitwise equal.
- Replacing every masked input feature with noise left the effective conditioning and the output **bitwise
  unchanged**.
- Observed features were imputed to 2.4e-7.

## 4. Representation, mask and clock (declared; G1, G2)

**Time map** (rational, duration-preserving, no retiming):

- Action frame *f* sits at t = (f−1)/30 s on the half-open axis [0, 8.0) s. Model sample *k* sits at t = k/20 s.
  240 frames ↔ 160 samples.
- Even *k* is the exact Action frame 1+1.5k. Odd *k* lies exactly half-way between two frames: every keyed channel
  (object + 7 bones) is SO(3)-slerped at 0.5 and labelled an INTERPOLATED hypothesis (80 exact, 80 interpolated samples).
- The edit interior Action 172–199 (= SOURCE 250–277) corresponds to samples **114–132**, the samples strictly between
  t(171) and t(200).
  - The nearest observed left-arm samples are 113 (Action 170.5) and 133 (Action 200.5), both outside the interior.
- **20→30 reconstruction** on the same 240 timestamps: frame *f* ← model position s = 2(f−1)/3.
  - When *s* is integral, the exact sample is used; otherwise the per-sample left-arm basis is slerped between its
    neighbours (weights 1/3, 2/3).
  - Action 172 = sample 114 and Action 199 = sample 132.
  - Action 240 (s = 159.33) lies beyond the last sample and is never reconstructed; it is also outside the interior.
- 30 fps, the frames and the key times are unchanged in every branch (G2).

**Mapping.**

- Axes: Blender (Z up, faces −Y) → HumanML3D (Y up, faces +Z) by (x, y, z) → (x, z, −y), det +1.
- Each of the 22 HML joints is a declared BaseRig point (§ `RIG_POINT` in `src/app/experimental_condmdi.py`). For
  example: pelvis = thigh-head midpoint; spine1/2/3 = `spine_02`/`spine_04`/`spine_05` heads; shoulder/elbow/wrist =
  `upperarm`/`lowerarm`/`hand` heads.
- The encoder is the official HumanML3D recipe:
  - the uniform skeleton with the 000021 bone lengths;
  - the canonical frame 0;
  - the gaussian-smoothed official root heading;
  - root-relative positions.
- **One declared deviation: the floor.** The official floor is the minimum over all joints, which includes the UNKNOWN
  feet. W131 uses the minimum over the ankles minus c_ankle = 0.0566 m. That constant is the checkpoint's own mean
  ankle height minus its mean foot-joint height. On S1 this floor sits 3.4 cm above the official one.

**Coverage and mask** (`run/S1/conditioning.json`).

| Class | Joints / features | Passed |
|---|---|---|
| OBSERVED_HYPOTHESIS (driven channels; W102 M1+S0 via W105; GT-free) | knees, ankles; left elbow + wrist | knees/ankles at all 160 samples; left elbow/wrist only at samples 0–113 and 133–159 (114 + 27), **0 inside 114–132** |
| PROXY_PELVIS (Object rotation, W93) | root yaw, root height, hips | all samples |
| PROXY_RIGID_TORSO | spine1/2/3, neck, both collars, both shoulders | all samples |
| UNKNOWN | feet, head, right elbow + wrist; root xz travel; **all** 6D rotations, joint velocities, foot contacts | never (masked) |

The PROXY_RIGID_TORSO joints are the rest `spine_02..05` / `neck_01` / clavicles, carried rigidly by the driven
`spine_01`.

Observed features: 7,886 of 263×160. Interpolated samples are passed and labelled.

**Declared coupling.** The coupling is disclosed; it is not hidden.

- Every HML263 arm position is pelvis-relative. So the left-arm conditioning is anchored on the rig's rest upper torso,
  which is carried rigidly by the driven `spine_01`.
- The upper-spine and clavicle bending is UNKNOWN and is the rig's rest.
- The official heading also depends on the shoulders: a 20° spine bend turns it 5.4°.
- These PROXY features were passed with that label. Without them, no left-arm feature in HML263 could be observed at
  all. No UNKNOWN feature was promoted.

**Representation controls** (`results/controls/controls.json`, `d6d234ef…`; real Blender + real rig). **G1 PASS:**

- numpy FK vs Blender-evaluated pose matrices, all 240 frames: 1.27e-6 m;
- S1 arm-direction encode→decode round trip, 160 samples: ≤ 1.5e-6°;
- projection identity (A's own directions with A as twist): 8.6e-8;
- B through the full 30→20→30 + encode/decode + projection path vs B: 6.6e-8 (quaternion);
- official Motius parity on the same S1 joints:
  - encoder positions and root ≤ 2.4e-6;
  - `relative_to_absolute` 4.5e-8;
  - `recover_from_absolute` 7.0e-8;
  - target bone lengths 4.9e-11;
- synthetic controls:
  - static: left = +X;
  - 30° Object yaw: positions invariant to 1e-16;
  - a left-arm swing moves only the left elbow and wrist;
  - a spine bend moves the arm with the torso (arm relative to chest unchanged to 3.7e-15) and leaves the legs fixed;
- **UNKNOWN-rig perturbation.** Random rotations of `upperarm_r`, `lowerarm_r`, `hand_l`, `hand_r`, `foot_l`, `foot_r`,
  `neck_01` and `neck_02` move the masked joints by up to 294 mm. Every observed absolute feature stays **bitwise equal**.

**Negative controls, all fail closed (G6):**

- mirrored axis → `AXIS_MAPPING_IMPROPER`;
- identity axis → `AXIS_UP_MISMATCH`;
- swapped left/right arm joints → `LEFT_RIGHT_MISMATCH`;
- missing `lowerarm_l` → `BASERIG_MISSING_BONES`;
- the mask observing a right elbow, a rotation, root travel, or the left wrist inside the interior →
  `MASK_OBSERVES_UNKNOWN` / `MASK_OBSERVES_EDIT_DOMAIN`;
- bad mask shape;
- non-monotone time map;
- duration-changing time map;
- an unrepresentable 239-frame window;
- an interior overlapping the pin → `PROTECTED_OVERLAP`;
- Blender builds of `spine_01` or of frame 210 → `REFUSED_*`, and no file is written.

## 5. Real pretrained inference (G3)

**Configuration (declared).**

| Setting | Value |
|---|---|
| Pipeline | official `CondMDIPipeline._sample`, respacing `ddim100` (100 steps), CFG 2.5 |
| Imputation | on (`stop_imputation_at` 0) |
| Seed | 42 |
| Length | 160 |
| Text | "a person stands in place and presses both arms straight up overhead" |
| Precision | fp32 network |
| CUDA | cudnn deterministic, TF32 off |

**Runs** (`run/S1/model_R1/model_io.npz`, `2ec93555…`, holds the raw masked input, the absolute input, the mask, the
effective normalised conditioning, and the absolute/relative output and joints):

| Run | Result |
|---|---|
| R1 primary | sample 3.24 s; process 9.5 s including 3.3 s load; peak VRAM 1.76 GB allocated (1.81 GB reserved); RSS 1.7 → 2.7 GB. Observed features imputed to 3.0e-8. |
| R2 exact rerun (new process; determinism only) | **bitwise equal** |
| R3 leak control | UNKNOWN rig channels rotated (seed 131) and every masked input feature replaced by noise. The input differs, yet the effective conditioning and the output are **bitwise equal**. |

Only one primary sample exists. Nothing was searched.

**What the model generated** (model space, template skeleton, root-relative; `projection.json`):

- The left wrist is held **above the head from the very first generated sample**:
  - height 1.85 m at sample 114, against 1.66 m at the observed sample 113;
  - a peak of 2.16 m at samples 124–126;
  - 2.11 m at 132, against the observed 1.96 m at 133.
- So the generated interior **jumps 299 mm** from the last imputed sample (113) to the first generated one (114), and
  **166 mm** back down to the observed exit (132→133). Between those boundaries it is smooth.
- This is in the model output itself, before any projection.
- The model also generated 0.69 m of root travel. Root xz was masked, so this was unconstrained and is irrelevant to the
  left-arm projection (directions only).

## 6. Real bounded Blender Action (G4, G5)

**Files.** Each candidate was built from a byte copy of the frozen baseline with the W129 Blender stage, unchanged
(`build_variant`).

| Branch | `.blend` SHA-256 | Action |
|---|---|---|
| C | `8f49a93a…` | `AnimCV_EXP W131 C S1` |
| B_rs | `993fd23a…` | — |
| Cpin | `392646ba…` | — |

`build_variant` writes a provenance string from the W129 schema name `animcv_exp_w129_variant_v1`, with branch `C` or
`Brs`. This is inherited, not a W129 artefact.

**Fresh-process readback** (`experimental_action_stage.py readback`, unchanged). For C and B_rs:

- 1 Action, 32 CONSTANT F-curves keyed 1–240, 30 fps, scene 1–240;
- **224 changed keys, all `upperarm_l`/`lowerarm_l`, frames 172–199**;
- every non-left-arm bone world point and the Object matrix are bit-equal to A on all 240 integer frames and all
  subframes;
- the left arm is bit-equal outside 172–199.

**Untouched originals:**

- A has the same SHA and the same value digest.
- The W126 `refs.json`, the baseline `.blend`, the W126V-accepted `.blend` and the W129 B `.blend` are byte-unchanged.
- No accept or commit was called, and no lineage was touched.

**G5 hard pin.** C composed onto the W126V-accepted `.blend` (Cpin):

- the residuals at Action 132–140 equal W126V's exactly, max 6.36e-7 m (tolerance 1 mm);
- the pin frames are bit-equal: C vs A, and Cpin vs W126V.

## 7. Matched A/B/C quality on S1

**Video.** `run/S1/abc_S1_30fps.mp4` (`e27aefe5…`): 240 frames written and 240 decoded, 30.000 fps. It shows:

- the source crop with SOURCE frame IDs;
- A, B, C and the B_rs diagnostic, each as a front body view plus an oblique left-arm close-up;
- an interior/pin bar.

**Sheets:** entry (170–175), midpoint (183–188), exit (196–201) and overview (every 5th frame). Blinded X/Y/Z copies are
in `blind/`.

**Wrist kinematics, Action window 164–207.** Steps run 171→172 … 199→200.

*Arm-local* (the `clavicle_l` frame; attributable to the edited channels):

| Metric | A original | B slerp | C CondMDI | B_rs |
|---|---|---|---|---|
| Entry / exit C0 step (mm) | 93 / 3 | 24 / 25 | **429 / 153** | 24 / 25 |
| Peak step (mm) @ frame | 203 @173 | 25 @200 | **429 @172** | 25 @200 |
| Reversal flicks | 6 | 0 | 4 | 0 |
| RMS jerk (m/s³) | 6,485 | 2,564 | 7,338 | 2,564 |
| Path length (mm) | 1,947 | 701 | 1,456 | 701 |
| Entry / exit Δv (m/s) | 9.2 / 2.0 | 7.2 / 1.7 | 17.9 / 6.3 | 7.2 / 1.7 |

*World space* (includes the out-of-domain torso/Object motion, which is identical in every branch):

| Metric | A original | B slerp | C CondMDI | B_rs |
|---|---|---|---|---|
| Peak step (mm) | 400 | 242 | 375 | 242 |
| Reversal flicks | 8 | 9 | 15 | 9 |
| Mean / max deviation from A (mm) | 0 / 0 | 148 / 297 | **292 / 499** | 148 / 297 |
| Max distance off the chord (mm) | 280 | 204 | 394 | 204 |

*Articulation:*

| Metric | A original | B slerp | C CondMDI | B_rs |
|---|---|---|---|---|
| Elbow flex range (°) | 70–129 | 93–112 | 87–142 | 93–112 |
| Max dip of wrist-above-shoulder below the endpoints (mm) | 61 | 9.5 | −183 (stays above) | 9.5 |

- No branch has a near-straight frame, a bend flip or a hinge-opposite frame. Limb length is unchanged by construction.
- **B_rs vs B:** max wrist difference 0.00025 mm. The 30→20→30 conversion, HML263 encode/decode and projection are
  effectively lossless on B. C's deviations are therefore attributable to the model output, not to resampling or
  representation.
- **Historical reference (frozen W129, not rerun):** MotionBricks C had mean deviation 408 mm and arm-local entry 90 mm.
  W131's C is closer to A on average (292 vs 408 mm) but has a far larger entry discontinuity (429 vs 90 mm).

**Blinded agent review** (`review/visual_review_S1.json`, `3fcd53ce…`). It was written before the key was read and
before any metric was viewed.

- **The blinding held this time.** The agent had never seen A or B renders in this session, and its identity guesses
  were wrong. It believed Z (the panel with the one-frame jump at 171/172) was A; the key says Z = C, Y = A, X = B.
- **Unblinded answers:**
  - V1 flick: C and A;
  - V2 most natural: B;
  - V3 wrong elbow or twist look: A (an apparent hand/forearm twist at 196–198);
  - V4 most → least faithful: B, A, C (least: C);
  - V5 boundary: C at 171/172, B mild at 199/200;
  - V6 upward press: B yes, A yes, **C no**.
- **What C looked like.** The arm teleports from the chest rack at 171 to a near-overhead pose at 172 and holds high.
  Against the source, which is still near the rack at SOURCE 250–255, this reads as wrong timing, not as a press.
- These are agent stills only. The judgement is not human, and native-rate playback was not perceived.

## 8. Tests and run status (RUN / PASS / FAIL / NOT RUN)

| Item | Kind | Result |
|---|---|---|
| `tests/test_experimental_condmdi.py` (11) | numpy; REAL BaseRig rest snapshot with SYNTHETIC poses; no model | **11 PASS** (`results/pytest/focused.xml`) |
| Regression: `test_experimental_motionbricks.py` + `test_experimental_video_action.py` + `test_cli.py` | existing | **75 PASS** (`regression.xml`) |
| Rights/identity preflight | real metadata | **RUN**: OK_PENDING_IDENTITY |
| Tensor identity vs the official release | real weights | **RUN, PASS** (196/196 bitwise) |
| Smoke + mask contract | **real pretrained model**, non-S1 input | **RUN, PASS** |
| Representation controls (attempt 1) | real Blender + Motius parity | **RUN, FAIL** (control-design errors, §2) |
| Representation controls (attempt 2) | real Blender + Motius parity | **RUN, PASS** |
| Negative controls (15) | adapter + real Blender | **PASS** |
| S1 R1 / R2 / R3 | **real pretrained model** | **RUN**: G3 PASS |
| C / B_rs / Cpin build, fresh readback, G4/G5 | real Blender | **RUN, PASS** |
| Video and sheets (G7) | real Blender renders | **RUN, PASS** |
| Assessment | frozen rule | **CASE B** |

**Failed:** QA1, QA2, QA4 and QA6 (quality criteria, not tests).

**Not run:**

- a human visual review;
- native-rate perceptual review;
- independent re-execution;
- the full repository suite;
- any other seed, text, mask, window, guidance, sampler or model, and S3 — all excluded by rule;
- a no-imputation or a non-EMA run — not run;
- training.

**Post-outcome changes:** none to instruments. After the outcome only this Worklog and the scratch evidence manifest
were written.

## 9. Negative findings

- **CondMDI introduced a larger discontinuity than the source.** Its first generated frame is a 429 mm arm-local step
  (A 93 mm, B 24 mm), and the exit has a 153 mm step. In model space the jump is already 299 mm between the last
  imputed and the first generated sample. With replacement imputation, the observed context did not make the
  generated motion continuous at the mask boundary.
- **Intent and timing loss.** The model lifted the arm to overhead immediately and held it high. The source, and B,
  press progressively from the rack. The no-dip criterion passes only because the arm never comes down; the timing is
  wrong.
- **Source fidelity is worse than slerp:** mean world wrist deviation 292 vs 148 mm.
- **Partial-joint conditioning reduced, but did not remove, source-intent loss** compared with W129's MotionBricks
  (292 vs 408 mm mean deviation, no down-and-up swing). It still loses to a no-model slerp on every continuity and
  fidelity measure.
- **The model ignored "stands in place"** for the masked root travel (0.69 m). That is harmless here, but it shows the
  unconditioned features follow the learned prior rather than the text.
- **The HF config misreports the training scheme and run directory.** The tensors are the random-joints EMA release.
- **Two of my controls were wrong in attempt 1**, and one control was invalid by construction. All three are disclosed
  in §2.

## 10. Unresolved uncertainties

- **How a human sees it at 30 fps.** Whether a human viewing the 30 fps video would judge C's held overhead arm as a
  plausible alternative press timing, or as wrong. Only agent stills were judged.
- **The cause of the boundary jump.** It may be replacement imputation itself, the out-of-distribution feature-level
  mask (positions only; the model was trained with position+rotation+velocity groups and an always-observed full root),
  the text cue, or the jittery source context. The source itself has a 170 mm model-space step at 112→113, right before
  the interior. No ablation was run, by rule.
- **Whether the rigid-rest upper-torso PROXY anchor or the declared floor/ankle constant biased the generated arm.** It
  is uncontrolled.
- **The legal reading.** Recorded only; there is no human licensing decision. The AMASS-derived training-data terms
  were not re-fetched.
- **Whether the S1 source arm (A) is itself correct.** Both A's interior and the endpoints are GT-free hypotheses, and
  the S1 left arm is the self-occluded far arm.
- **The runner's `motius_head` field.** It recorded the symbolic ref `ref: refs/heads/main`, not a SHA. The checkout's
  HEAD (`d04a38c`) was verified by the preflight.

## 11. Reproduction

Git Bash, repository root at `6ef2726`. Every stage refuses an existing output directory.

Scratch runtime setup (once):

```
git clone https://github.com/setarehc/diffusion-motion-inbetweening.git C:/Users/dna10/animcv-w131/ext/condmdi   # HEAD 0c58473
git clone https://github.com/ZeyuLing/Motius.git C:/Users/dna10/animcv-w131/ext/motius                           # HEAD d04a38c (GIT_LFS_SKIP_SMUDGE=1)
uv venv --python 3.12 C:/Users/dna10/animcv-w131/env/cm312
uv pip install --index-url https://download.pytorch.org/whl/cu128 --no-deps torch==2.9.1+cu128 torchvision==0.24.1+cu128
uv pip install "numpy<2.3" safetensors einops ftfy regex tqdm packaging mmengine scipy huggingface_hub psutil
uv pip install --no-deps "clip @ git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
curl -L https://huggingface.co/ZeyuLing/motius-condmdi-humanml3d/resolve/1e84d9ebf9a2562a2e3aa01926b9fb3606bd23de/<file>  -> hf/1e84d9e.../
official condmdi_random_joints.zip (Drive id 1aP-z1JxSCTcUHhMqqdL2wbwQJUZWHT2j) -> official/random_joints/ (identity only)
```

Harness stages:

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; CM=C:/Users/dna10/animcv-w131/env/cm312/Scripts/python.exe
export PYTHONPATH=src; R=C:/Users/dna10/animcv-w131/results
$PY -X utf8 scripts/w131_condmdi_preflight.py --out-dir $R/preflight
$CM -I scripts/w131_condmdi_infer.py $R/identity_req/req_identity.json     # mode identity
$CM -I scripts/w131_condmdi_infer.py $R/identity_req/req_smoke.json        # mode smoke
$PY -X utf8 scripts/w131_condmdi_harness.py controls --out-dir $R/controls --preflight $R/preflight/preflight.json --identity $R/identity/identity.json
$PY -X utf8 scripts/w131_condmdi_harness.py declare  --out-dir $R/declare  --preflight ... --identity ... --controls $R/controls/controls.json
D="--declaration $R/declare/declaration.json --declaration-sha dca4a891d21725f813b8cbb1867abaab8f237f59e6f36080086b2267355107e4"
$PY -X utf8 scripts/w131_condmdi_harness.py run    --out-dir $R/run --controls $R/controls/controls.json $D
$PY -X utf8 scripts/w131_condmdi_harness.py assess --out-dir $R/assess --preflight ... --identity ... --controls ... \
    --run-dir $R/run --visual-review $R/review/visual_review_S1.json $D
$PY -m pytest -q tests/test_experimental_condmdi.py
```

A re-run of `declare` produces a new timestamp and therefore a new SHA, so pass that SHA to the later stages.

## 12. Evidence and changed files

**Evidence**, under `C:/Users/dna10/animcv-w131/results`; local, not published, 235 MB. `evidence_manifest.json` lists
2,036 files, SHA-256 `98505f804945b6a772dd2e63aa469ae5e207a2a9d17101dde6f2775ce73d7cc0`. It was written before this
Worklog.

| File | SHA-256 |
|---|---|
| `preflight/preflight.json` | `a2ceb54a…` |
| `identity/identity.json` | `d0fd115b…` |
| `smoke/result.json` | `8f710d2c…` |
| `controls/controls.json` | `d6d234ef…` |
| `declare/declaration.json` | `dca4a891…` |
| `run/S1/run_result.json` | `f5fbdb8f…` |
| `run/S1/model_R1/model_io.npz` | `2ec93555…` |
| `run/S1/abc_S1_30fps.mp4` | `e27aefe5…` |
| `review/visual_review_S1.json` | `3fcd53ce…` |
| `assess/assessment.json` | `9633a362…` |

**Population:** one source episode (S1, 240 frames), one model sample (plus a determinism rerun and a leak-control run),
four branches.

**Changed files (repository):**

- New:
  - `scripts/w131_condmdi_preflight.py`
  - `src/app/experimental_condmdi.py`
  - `scripts/w131_condmdi_infer.py`
  - `scripts/w131_condmdi_harness.py`
  - `tests/test_experimental_condmdi.py`
  - `examples/experimental_condmdi/w131_declaration.json`
  - this Worklog
- Unchanged:
  - `docs/next_batch.md` and every historical Worklog;
  - the W129 adapter and Blender stage, and `experimental_action_stage.py` (called, not edited);
  - CLI, GUI, the exporter, FramePose, MotionGraph and FK/IK;
  - the rig, models and global dependencies;
  - automation scripts;
  - every W126/W129 file.

No weights, videos or renders are committed.

## 13. Resource and environment constraints

| Resource | Cost |
|---|---|
| Scratch disk | about 14 GB: CondMDI venv 4.6 GB, uv cache 5.1 GB, HF artifact 1.2 GB, official zip 1.7 GB, code clones 1.4 GB, results 0.23 GB |
| Inference | 1.76 GB peak allocated VRAM, about 3.2 s per 100-step 160-frame sample on the RTX 5080 |
| Full `run` stage (3 inferences, 3 builds, 6 readbacks, 4 × 480 renders, 2 videos) | about 2 minutes |

- CondMDI needs torch with sm_120 support, the OpenAI CLIP package and mmengine. These live in a separate scratch
  Python 3.12 environment; the AnimCV venv (3.11) was not modified.
- The first uv install resolved a CPU-only torch wheel. It was replaced by the explicit `+cu128` build before any model
  run.
