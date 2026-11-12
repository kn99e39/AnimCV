# Worklog — Official UniMate `--inbetween` on the Authentic S1 Rig: Raw Generator Unusable, Not Better Than W136 (2026-11-12)

Batch 137 was an experimental, quality-first, research-only batch on `arch/single_frame_first`. It ran unattended on Main with this setup:

- local RTX 5080 (16 GB, driver 596.49), Windows 11 Pro 10.0.26200;
- the AnimCV Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9), **unmodified**;
- the existing W136 scratch UniMate runtime `C:/Users/dna10/animcv-w136/env/um310` (CPython 3.10.22, torch 2.7.1+cu128), unmodified;
- Blender 5.1.0 for every AnimCV-side build, dump, readback and render.

Other conditions:

- `docs/next_batch.md` was only read.
- No LabServer and no other machine was used.
- Network use was limited to `git fetch/push` and the public, ungated Hugging Face metadata and licence pages for the pinned UniMate and flan-t5-base revisions. The weights were already on disk and hash-verified, so nothing was downloaded.
- No LLM or other external API was called.

Every number, render and verdict below is **AGENT-RUN** on Main. The reviewer has not independently rerun the GPU or Blender steps. Every visual judgement is the agent's own **unblinded still-frame** review. No human or animator reviewed anything, and the agent did **not** perceive native-rate playback. The rights reading is documentary evidence. It is not legal advice and not a human licensing decision.

## 1. Official verdict and scope

**Verdict: CASE C.** There is no STOP code: every pre-inference gate passed and one real official inference ran.

- Source: `results/assess/assessment.json`, SHA-256 `0d582a0ca98503430ae1192cc1a59eddcef9f96aaaa8f6b99adb4d7b767f086e`.
- The frozen rule gives C through three independent triggers:
  1. **projected unusable continuity:** T's arm-local wrist entry step is **300 mm** and its elbow entry step **234 mm**, against the 100 mm limit;
  2. **raw full-body continuity unusable:** the raw wrist step exceeds PROXY_B's step at the same transition by 624 mm, and the right arm, legs and fingers exceed it at the boundaries by 187–499 mm;
  3. **frozen M4 = FALSE:** this comes from an instrument log-parsing defect, explained in §6. The conditioning itself is independently shown to be exact.

**What the run settles:**

1. **The official frame-only mode runs, unmodified, on the own rig.**
   - The official `build_keep_mask` returned a `(1,1,1,60)` mask, True exactly at window frames 0–15 and 44–59.
   - The official log reports `valid_length=60` and those same 32 indices.
   - Every kept frame of the sample equals PROXY_B (max |Δ| **0.0**) in every joint group.
   - The interior is not a copy: exactly equal entries make up 0.17 % of the interior, all of them in the root group's constant channels.
   - The mask contract and the `--inbetween`/`--motion_edit` exclusivity were proven without the model.
   - W136's custom joint × time hook was **not** used. No official function was replaced; `generate_samples` was wrapped only by a read-only recorder.
2. **The official mode does not produce usable animation on this clip.**
   - The imported left arm (T) repeats W136's lateral-raise detour: it leaves the rack in one frame (wrist step 300 mm at 171→172), holds a near-horizontal sideways arm through the middle, and rises late to meet the frozen frame 200.
   - It has no reversal flicks (U had 7), but its entry jump is larger than U's.
   - Under the declared W136 anchor it is **NOT_IMPROVED**: the wrist maximum step is lower (300 vs 480 mm), but the elbow maximum is 233.6 vs 232.0 mm, so it is not lower.
3. **The failure is attributed to the RAW GENERATOR, not to frozen-parent projection.**
   - Arm-local motion is identical in the raw and the projected views by construction.
   - The raw full-body output is itself discontinuous and implausible. The body folds mid-gap: the neck is 249 mm above the pelvis at Action 178 against 636 mm in PROXY_B. The root drifts 366 mm by Action 200. The raw positions (RIC) disagree with the raw rotations (FK) by up to 284 mm.
   - The raw world wrist peak step is 681 mm, against 359 mm projected. Projecting onto the frozen parent did not create the failure; it removed part of it.

**Scope:**

- Scratch experiment only. T, Tc, Tpin and Trev are **SCRATCH PROPOSALS**. They were never committed or accepted and are in no lineage.
- No Production, CLI, GUI, exporter, FramePose, MotionGraph, schema, Unity or FBX change. `scripts/batch_automation.py` is untouched.
- Private, local, **non-commercial** technical evaluation only. No weights or model outputs were redistributed.
- Exactly one inference. No training or finetuning. No seed, prompt, checkpoint, cfg, step or window variation, and no post-hoc repair.

**Historical CASEs and artefacts are untouched:** W129 B, W135 B, W136 C. No historical Worklog, audit or W136 file was edited. Every W136 artefact that was used was verified against W136's own evidence manifest, before and after the run.

**Original task scope (short).** Run exactly one official UniMate v3 `--inbetween` inference on the frozen W136 own-rig PROXY_B clip (window Action 156–215), with the same checkpoint, prompt, seed, cfg and sampler as W136. Keep every whole-body frame outside Action 172–199 and generate everything inside. Import only `upperarm_l`/`lowerarm_l` rotations at 172–199 into a reversible scratch Action. Compare the result with B, W136 U and A/C/D/E, a matched no-model control, the actual 30-fps rig render and an explicit raw-versus-projected parent-context attribution.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `71205cd79aa3ddfab6c54142dd4fb41c6853cbcf` (verified first) |
| Starting commit | `3b8ec215a0e877955014643cfd3feca0be256eb0`, equal to `origin/arch/single_frame_first` at the first fetch. W136 has `AUTOMATION_BATCH_STATUS: COMPLETE` / `BATCH_NUMBER: 136`. No `docs/137_*`–`139_*` existed. |
| Instrument commit (pre-outcome) | `eceda19`: module, UniMate-side runner, harness, 37 synthetic tests |
| Declaration commit (pre-outcome) | `a6776b9`: `examples/experimental_unimate/w137_declaration.json`, a copy of `results/declare/declaration.json` with SHA-256 `9dc6d74eda54bc4cad2ac279456fd33fcaaa2a802314681262fc9f6f93b4a0d7` (scratch-file SHA) |
| Final source revision | `a6776b9` for code and declaration. This Worklog is documentation only; the COMPLETE marker is a separate, final commit. |
| Scratch root | `C:/Users/dna10/animcv-w137`, outside the repository, 419 MB. It re-uses the W136 env, checkpoint and T5 snapshot in place. |

**Order of stages:**

1. input-only exploration (contract mode, and FK/RIC recovery of PROXY_B; no model);
2. instruments, then the instrument commit;
3. preflight;
4. declare, then the declaration commit;
5. controls;
6. run (the only inference);
7. visual review;
8. assess.

No instrument, threshold or rule changed after any model output existed. `declare` refuses to write if any `raw_model_io.npz` or `T_S1.blend` exists under the W137 results (`model_output_exists: false`).

## 3. Identity, rights, access and resource gate (M0)

Source: `results/preflight/preflight.json`, SHA-256 `e331be8d7d940f5bbb788303ed47c060a2cbbada4ca874c864e282c2caf082f4`. Status OK.

**Code.**

- `Friedrich-M/UniMate` `b78c7807df7cf4231d4e67f8d0ece29c5e11025b`, clean tracked tree.
- The ten key files (`sample.py`, `motion_inbetweening.py`, `generate.py`, `motion_utils.py`, …) are byte-equal to W136's recorded SHA-256.
- Code licence: MIT.

**Model.**

- `Linzhan/UniMate` revision `174fc755ec248d0c98f27759b7a95648e7f05905`, not gated.
- Checkpoint `unimate_uniml3d_f60_v3/checkpoints/checkpoint_step_150000.pt`: `f991ddd87262c251f4681d18297a6163ab4f139daa28c86d5ef0a66771df7bc6` (local = published LFS).
- Stats `0ff2b9ef3ea9a15b024ab0670e55f659478251501e5596f9011d5bca4cf67773`.
- Config `1ffbd379d3667de7d701ed6f3418f1c9d1a66f35d2653776255008c07c051aaa`: flow, 60 frames, 12-D features, `sampling.cfg_scale` 3.0, `sampling.seed` 10.

**Text encoder.**

- `google/flan-t5-base` `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`, not gated, Apache-2.0.
- Local `model.safetensors`: `1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8`.
- Used offline.

**Rights determination: PERMITTED_NONCOMMERCIAL_RESEARCH_EVALUATION.**

- The model `LICENSE`, the model card and the T5 card were re-fetched at the pinned revisions. All three are **byte-identical to W136's copies**, and every required clause is present (CC BY-NC 4.0 weights "released for non-commercial research use"; MIT code; Apache-2.0 T5).
- The declared purpose is a private, local, non-commercial technical research evaluation with no redistribution.

**Unresolved, not granted:**

- Commercial use is not granted.
- The determination depends on the declared non-commercial purpose, which is a human-owned judgement.
- The licensor itself marks training-data provenance as unclear. All five provenance clauses (Truebones allegations, Mixamo terms, non-commercial Objaverse objects, no warranty on generated motions) are still on the card.

**Runtime and resources.**

- torch 2.7.1+cu128 with `sm_120`.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set for the inference subprocess only (as W136, declared).
- `env/freeze.txt` is unchanged; GPU 1.6 GB used before the run; 390 GB disk free.

**Frozen evidence — 55 identities, all OK before and after the run (M9):**

- the S1 clip `de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c`;
- A `0803ab2b…`, B `8899f433…`, C `e04efb35…`, Cpin, W135 D/E, and the W126V accepted pin `7e2ab165…`;
- the W126 refs/proposal;
- the W129/W135 records;
- W136's evidence manifest `b25805a4…`, and through it W136 U `e0fc3fe3…`, Bc, Upin, Urev, R0, the W136 declaration/assessment/run/preflight/controls/review/video/raw I/O/saved npy, and the own-rig asset files (`cond.npy` `b674dea8…`, clip `8d62b42a…`, `summary.json`, canonical GLB, annotation).

Fresh Blender readbacks of byte copies of A, B, C, D, E, the W126V pin and W136 U are **world-bit-equal to W136's readbacks** on all 240 frames and 717 subframes.

## 4. Representation and no-model controls (M1)

The W136 own-rig asset was **byte-copied** (all five files equal to W136's manifest) and not re-preprocessed.

| No-model check | Result |
|---|---|
| Official features of the copied asset vs W136's | bit-equal |
| Decoded PROXY_B vs B `upperarm_l`/`lowerarm_l` keys, shift 0 | 2.5e-7 / 2.7e-7 |
| Same at shift ±1 (time identity) | 0.221 / 0.221 |
| Own vs official decode | ≤ 4.4e-16 |
| Non-left-arm context drift | ≤ 8e-7 |
| Joint order / labels / side / hierarchy / facing (cos) / uniform scale | 50 joints, BaseRig hierarchy + pelvis; PASS; facing 1.000 |
| Official temporal contract (`parse_keep_frames` + `build_keep_mask`, declared literal string) | shape (1,1,1,60), resolved = 0..15, 44..59, equals the module mask; 1-frame-shifted list refused |
| Official `_validate_cfg_inputs` | `--inbetween`+`--motion_edit` refused ("mutually exclusive"); `--inbetween`+`--motion_expand` refused; `--inbetween` at cfg 1.0 refused; the declared arguments accepted |
| **World-coordinate map:** official FK (`recover_unimate_joint_pos_from_rot` with cond `tpos_offsets`) of PROXY_B vs B's Blender world bone heads (21 bones × 60 frames), one similarity | RMS **2.6 µm**, max 5.6 µm; ±1 frame shift → 71 mm (time-separable) |
| Torso-only affine camera on A (Action 156–215), recorded before the declaration | RMS **23.3 px > 20 px → INVALID** |

**Disclosed pre-declaration choice (input-only finding).** The official FK with the cond field `offsets` (bone-local rest offsets) disagrees with RIC by 1.44 model units on PROXY_B. With `tpos_offsets`, which is what the official `animate_motion` uses, FK equals RIC to 2.8e-7. The FK input was therefore fixed to `tpos_offsets`, and the world-map gate was set to 1e-4 m after that measurement. A control proves that the `offsets` variant is refused.

## 5. Prospective declaration (summary; the committed JSON is the full record)

| Item | Frozen value |
|---|---|
| Official mode | `InferenceArgs(inbetween=True, motion_edit=False, keep_frames="0,1,…,15,44,…,59", gt_start_frame=0, num_repetitions=1, batch_size=1, only_save_motion=True)` via `unimate.inference.sample.main` |
| Kept / free | all 50 joints × 12 channels at window 0–15 and 44–59 (32 frames, Action 156–171 and 200–215) / window 16–43 (28 frames, Action 172–199), every joint and channel, incl. root, torso, clavicles, right arm, legs |
| Prompt / seed / cfg / sampler | "An object presses its left arm overhead." / 10 / 3.0 / official 50-step Euler replacement ODE, EMA weights — identical to W136 |
| Import | only `upperarm_l`/`lowerarm_l` at 172–199 via the unchanged W129 `build_variant`. The raw output of every other joint is analysed, never imported. |
| Matched control | Tc = the PROXY_B tensor saved by the same official run, through the identical decode/import |
| Gates | M0–M12 (M12 = the frozen-parent hybrid FK equals T's Blender readback) |
| Quality | Q1–Q8 (Q3 now covers both wrist and elbow peaks); Q5 declared invalid while the camera exceeds 20 px; Q8 declared NOT REVIEWED in advance, so **CASE A was unattainable** |
| W136 anchor | T vs U: IMPROVED iff T's wrist AND elbow max(entry, exit, peak) are lower than U's and T's flicks ≤ U's |
| Raw rule | raw unusable iff a raw world elbow/wrist step exceeds PROXY_B's at the same transition by > 100 mm, or a joint group's raw entry/exit step does |
| Attribution rule | arm-local motion is identical raw vs projected → an arm-local failure is attributed to the raw generator; world differences are parent context; FROZEN_PARENT_PROJECTION only if the projection fails while the raw output is usable; otherwise UNKNOWN |
| Precedence | DATA_BLOCKED > D > C > B > A |

## 6. The single inference and conditioning (M4)

Sources:

- `results/run/run_result.json`, SHA-256 `e0b83668ddbfb06ba963c5432142543a037aaac15be8cdf5447c8ddd3eaec28b`;
- `run/model/infer_record.json` `38eeeb83…`.

**Inference:**

- exactly 1 `generate_samples` call;
- sampler 4.71 s, official `main` 6.53 s, peak VRAM **0.45 GB** (446,739,968 B);
- raw I/O `run/model/raw_model_io.npz` `68e636c0…`;
- saved sample `.npy` `0ff62614…`;
- saved PROXY_B (gt) `.npy` `fe50b40e…`, **byte-identical to W136's saved PROXY_B**.

**Conditioning:**

- The recorder validated the received mask (shape (1,1,1,60), 32 declared frames) **before** sampling.
- Kept frames: max |Δ| vs PROXY_B = **0.0** in every group (root, torso, head, clavicle, left arm, right arm, legs, fingers).
- Interior: max |Δ| 1.32 feature units, exact-equal fraction 0.0017 (left arm 0.0; root 0.083, which are the constant RIFKE root channels).

**Frozen M4 = FALSE — an instrument defect, recorded truthfully.**

- The M4 rule also required parsing the official log line. The rich logger wraps that line, and the harness regex read only its first physical line, so `official_log_ok` evaluated FALSE.
- A whitespace-joined re-parse of the same log gives `valid_length=60, keep_indices=[0..15, 44..59]`, equal to the declaration. This is in `results/posthoc/posthoc_notes.json` (`11dcaeb7…`), labelled POST-HOC.
- The frozen record and the CASE were **not** changed. CASE C holds independently of M4.

## 7. Blender Actions and gates

| File | SHA-256 | Action |
|---|---|---|
| T `T_S1.blend` | `95f25850fd5bef24b031a99bfba351a8285133cd8ba82dfe22384ff1ec2f0ab6` | `AnimCV_EXP W137 T UniMate inbetween S1` |
| Tc `Tc_S1.blend` | `494d1d796aa4736de996de2e7e124b22e52f542abd24154d6c992f0cf804033d` | `AnimCV_EXP W137 Tc PROXY_B control S1` |
| Tpin (T on W126V) | `33a9eaa27fcfcf6427e3cac1417a4dd6e586fd4b4dccb691b780927e102af467` | |
| Trev (T with A's keys restored) | `77fb58ab06f5cc169cc1e493e9bf37ca5a8ab1123b0b2cceb647be2da99950d6` | |

| Gate | Result |
|---|---|
| M0 identity/rights/access/resources | PASS |
| M1 representation + in-run PROXY_B time identity (2.5e-7 / 2.7e-7; ±1 → 0.22) | PASS |
| M2 one named Action, 32 CONSTANT F-curves keyed 1–240, 30 fps (T, Tc) | PASS |
| M3 footprint: 224 changed keys, only `upperarm_l`/`lowerarm_l` at 172–199; Action 171 and 200 bit-equal to A; every non-left-arm world point and object matrix bit-equal on 240 frames + 717 subframes | PASS |
| M4 conditioning | **FALSE (frozen; log-parse defect, §6)**; mask, kept exactness and copy detector PASS |
| M5 limb lengths: upper/lower range 0.19 / 0.22 µm; unit, finite quaternions | PASS |
| M6 hard pin: Tpin residuals at Action 132–140 equal W126V's (max 6.36e-7 m ≤ 1 mm target tolerance); pin frames bit-equal | PASS |
| M7 reversibility: Trev world-bit-equal to A on all frames and subframes | PASS |
| M8 matched control: Tc vs B max world difference 0.48 µm | PASS |
| M9 integrity (55 identities after the run) / M10 controls (62/62) / M11 video (240 frames decoded at 30.000 fps) | PASS |
| M12 hybrid FK (PROXY_B parents + generated arm rotations) vs T Blender readback | 5.7 µm, PASS |
| GENERATED_EFFECT (T vs Tc) | max **736 mm** (wrist, Action 177); mean wrist 460 mm, elbow 253 mm |

## 8. Real animation quality: A / B / C / D / E / U / T

**Arm-local** (clavicle_l frame; steps 171→172 … 199→200). Wrist: W129 `armlocal_quality`; elbow: W135 `elbow_metrics`. These are fresh readbacks. U reproduces W136's numbers exactly.

| Metric | A | B | C | D | E | U (W136) | **T (W137)** |
|---|---|---|---|---|---|---|---|
| Wrist entry / exit (mm) | 93.2 / 2.9 | 24.1 / 25.0 | 89.6 / 9.6 | 24.1 / 25.0 | 24.1 / 25.0 | 146.2 / 177.2 | **300.4 / 159.3** |
| Wrist peak step (mm) @ frame | 203.1 @173 | 25.0 | 156.6 @174 | 25.0 | 25.0 | 480.5 @175 | **300.4 @172** |
| Wrist flicks | 6 | 0 | 0 | 0 | 0 | 4 | **0** |
| Elbow entry / exit (mm) | 64.4 / 8.2 | 17.8 / 17.8 | 63.3 / 3.3 | 52.3 / 13.7 | 17.8 / 17.8 | 149.5 / 121.5 | **233.6 / 112.3** |
| Elbow peak (mm) | 214.5 | 17.8 | 84.3 | 61.5 | 17.8 | 232.0 | **233.6** |
| Elbow flicks | 6 | 0 | 0 | 0 | 0 | 3 | **0** |
| Flex range (°) | 70–129 | 93–112 | — | — | — | 65–162 | **57–153** |
| World wrist net rise 172→199 (mm) | 318 | 284 | 341 | 284 | 284 | 115 | **231** |
| World wrist dev. from B, mean / max (mm) | 148 / 297 | 0 / 0 | 347 / 579 | 0 / 0 | 0 / 0 | 512 / 847 | **460 / 736** |
| 2D left-wrist mean (px; camera INVALID) | 225 | 206 | 99 | 206 | 206 | 81 | 86 |

**Quality criteria for T:**

- Q1 FAIL, Q2 PASS, Q3 FAIL, Q4 PASS (no near-straight or hinge-opposite frame), Q6 PASS (231 ≥ 142 mm).
- Q5 FAIL: the camera is invalid, so source-2D fidelity is UNKNOWN. T's lower pixel error is **not** a fidelity gain; as in W136, the source-derived A scores worst.
- Q7 FAIL (agent review). Q8 NOT REVIEWED.
- The unusable-continuity diagnostic fires: wrist 300 mm and elbow 234 mm are both > 100 mm.

**Shared, source-derived parent popping.** The world shoulder step reaches up to 244 mm/frame in every branch (A–E, U, T). This comes from the preserved object/spine tracks, and no branch solves it.

## 9. Raw full-body motion and raw-vs-projected attribution

Raw = official FK (`tpos_offsets`, the model's own integrated root) of the raw sample, mapped to rig world by the no-model similarity. Projected = the same FK with PROXY_B features except the two generated arm rotation slots; this equals T's Blender readback to 5.7 µm. Data: `run/raw_world.npz` `349fc1a5…`.

**Continuity, Action 171→200 (mm):**

| View | Shoulder entry / exit / peak | Elbow entry / exit / peak | Wrist entry / exit / peak |
|---|---|---|---|
| PROXY_B (= B) | 41 / 23 / 244 | 40 / 18 / 211 | 51 / 16 / 242 |
| Projected (T) | 41 / 23 / 244 | 181 / 110 / 268 | 232 / 141 / 359 |
| **Raw model** | **171** / 27 / 171 | **219** / 105 / **348** | **249** / 109 / **681** |

**Raw full-body by joint group (Action 172–199):**

| Group | Interior deviation from PROXY_B, max (mm) | Raw / PROXY_B entry step (mm) | Raw / PROXY_B exit step (mm) | Raw / PROXY_B peak (mm) |
|---|---|---|---|---|
| root (pelvis) | 350 | 57 / 35 | 20 / 15 | 64 / 153 |
| torso | 784 | 82 / 47 | 19 / 22 | 193 / 238 |
| head | 1019 | 117 / 51 | 22 / 25 | 265 / 268 |
| left clavicle | 844 | 101 / 47 | 18 / 23 | 207 / 247 |
| left arm | 943 | 249 / 51 | 109 / 23 | 681 / 244 |
| right arm | 1372 | **465** / 72 | 166 / 23 | 962 / 273 |
| legs | 527 | 203 / 64 | 197 / 16 | 203 / 174 |

**Further raw-output facts:**

- **Root drift:** 0 mm before the gap, 339 mm at Action 199, 366 mm at 200 and after. The generated root velocities integrate into a sideways body translation that the kept frames cannot undo.
- **Raw positions vs raw rotations:** RIC vs FK differ by up to **284 mm** in the interior (0.2 µm on kept frames). The model's own position and rotation channels disagree in the generated frames.
- **Body collapse (post-hoc check, `posthoc_notes.json`):** neck_02 height above the pelvis at Action 171/178/185/192/199 is 629/636/633/636/642 mm in PROXY_B, 629/**249**/**405**/624/641 mm in raw FK and 629/318/459/623/637 mm in raw RIC. Both recoveries show the torso folding mid-gap, and RIC also changes limb lengths (right upper arm 196–291 mm vs a constant 278).
- **Raw vs projected elbow/wrist gap:**
  - wrist: max 628 mm, mean 383 mm;
  - elbow: max 459 mm, mean 265 mm;
  - the shoulder-translation part alone reaches 623 mm;
  - the parent-orientation part reaches 883 mm on the wrist. The two partly cancel.

**Attribution (declared rule): RAW_GENERATOR.**

- Arm-local continuity, where T fails, is identical in the raw and projected views (same local rotations).
- The raw world motion itself fails the raw rule:
  - wrist excess 624 mm and elbow excess 306 mm (both at 173→174);
  - boundary excess: left arm 198, right arm 393, legs 187, left fingers 216, right fingers 499 mm.
- Frozen-parent projection is **not** the cause: the projected world peak step (359 mm) is lower than the raw peak (681 mm).
- Whether a frozen parent fits a *good* generated arm cannot be decided here, because no good generated arm exists to test it with.

## 10. Visual review and perceptual boundary

Review file: `results/review/visual_review_S1.json`, `360a2beb…`. It was written before `assess`.

**Actually viewed by the agent (still frames, unblinded):**

- `sheet_arm_entry`, `sheet_arm_middle` and `sheet_arm_exit`, all of each;
- the D/E/U/T half of `sheet_body_overview`;
- the post-run diagnostic `posthoc/raw_vs_proxyB_sticks.png` (`8048b8fc…`).

**Not viewed:**

- `sheet_arm_overview`;
- the body entry/middle/exit sheets;
- the source/A/B/C half of the body overview;
- the 30-fps video `run/abcdeut_S1_30fps.mp4` (`992be516…`: source crop + A/B/C/D/E/U/T body and arm close-ups with an interior bar). It was encoded and machine-decoded (240 frames, 30.000 fps) but **not perceived at native rate**.

No human or animator reviewed anything.

| Branch / frames | Observation |
|---|---|
| T 171–172 | Leaves the rack pose in one frame; elbow and hand swing outward and forward |
| T 173–175 | Near-horizontal, nearly straight lateral raise by 173–174 (one frame earlier than U) |
| T 183–188 | Arm stays extended sideways at shoulder height with a slight bend; the source shows both kettlebells rising beside and above the head |
| T 196–198 | Forearm up with a low, lateral elbow (as U) |
| T 199–200 | Rises to a near-overhead pose that joins the frozen 200 more closely than U |
| T, judgement | Not more natural and not more press-faithful than B; named for a detour and an abrupt entry. No visible wobble, unlike U. |
| Raw full body (sticks) | Torso folds and legs splay mid-gap (178–192); by 198 the body re-forms translated sideways relative to PROXY_B |
| B (= E, = Tc) | Monotonic rack-to-overhead rise |

**Separated conclusions:**

1. synthetic contract correctness: PASS;
2. official model execution: real, one run, unmodified mode;
3. model contribution: GENERATED_EFFECT;
4. Blender preservation, pin and reversibility: complete;
5. source fidelity: UNKNOWN (invalid camera; the occluded far-arm 3D is UNKNOWN);
6. naturalness and continuity: negative;
7. intentional style: not evaluated;
8. Production viability: not supported (NC weights, unresolved provenance, one clip, one seed, no human or native-rate review).

## 11. Tests and controls (RUN / PASS / FAIL / NOT RUN)

| Item | Kind | Result |
|---|---|---|
| `tests/test_experimental_unimate_inbetween.py` (37): keep-set/time map, mask exactness and 8+6 mask refusals, keep-list refusals, mode flags, conditioning fidelity per joint group, interior-copy detection, shape/non-finite, hybrid touches only the two arm slots, degenerate 6D, similarity fit + reflection, world map time/joint identity, continuity/flicks, attribution split, raw rule, unauthorised channel/frame and pin overlap | synthetic | **37 PASS** |
| Re-used `test_experimental_unimate.py` (19), `test_experimental_elbow_prior.py` (14), `test_experimental_motionbricks.py` (15) | synthetic | **48 PASS** (85 total, 4.1 s) |
| W136's 31 real-data controls, re-executed unchanged | real data + real Blender refusals | **31/31 fail closed**; output `controls.json` byte-identical to W136's (`44be7c0f…`) |
| W137 temporal-mode controls (31): mask shift ±1, missing 15/44, extra 16/43, all/none kept, joint- or channel-axis mask, int dtype; keep lists `0,-1`, freeing 171 or 200, keeping 172, shifted, negative, unsorted; `motion_edit`/both/`motion_expand` flags; kept root/right-arm/torso/left-arm perturbation; PROXY_B-copy interior; 59-frame features; world map with a +1 shift, mirror, hand_l/hand_r swap, or cond `offsets` | real PROXY_B features + real B readback (no model) | **31/31 fail closed** (`controls/controls.json` `665d5046…`) |
| Official contract (mask + exclusivity), features roundtrip, world map, one inference, 4 builds, 4 readbacks, 2 dumps, 7 × 480 renders, video, 8 sheets | real | **RUN** |
| Assessment | frozen rule | **CASE C** |

**Not run:**

- the full repository suite (Production untouched);
- human, animator, blinded or native-rate review;
- any second inference, seed/prompt/cfg/step/window sweep, other model or other clip (forbidden);
- the official mesh-driving and skeleton mp4 renders (`--only_save_motion`);
- re-runs of the W129/W135/W136 inferences.

**Failed:**

- Q1, Q3, Q5 (invalid camera) and Q7 (quality criteria, not tests);
- frozen M4 (log-parse defect, §6).

**Weak controls (disclosed):**

- the W136 corrupt-file controls check file identity only;
- the fps controls call the gate on substituted values;
- the W137 world-map controls perturb the input positions, not the recovery code.

## 12. Negative findings

- **The model-native temporal mode did not fix W136's failure.** Replacing the custom joint × time mask with the official frame-only mask removed the reversal flicks but produced the same lateral-detour arm, with a larger entry jump.
- **The official in-betweening generated an implausible full body for a 28-frame gap** under this prompt and seed:
  - torso fold, splayed legs, a 465 mm right-arm entry step;
  - 366 mm root drift that persists after the gap;
  - position/rotation channels disagreeing by 284 mm.
  The published feature is not evidence of successful 28-frame completion on this rig.
- **Replacement clamping is exact but does not enforce continuity.** All 32 kept frames are bit-exact, yet the generated frames jump away at 171→172 and back at 199→200, as in W136.
- **The prompt was not followed** ("presses … overhead" → sideways raise). Phrasing, cfg and seed were not varied, by rule.
- **Own-rig FK needs `tpos_offsets`.** The cond `offsets` field gives wrong positions (1.44 units) with the official rotation recovery; `animate_motion` uses `tpos_offsets`. Whether any official visualisation path uses `offsets` for own-rig assets was not checked.
- **The instrument defect in M4** (a wrapped log line) is the agent's own and is recorded rather than repaired in the frozen evidence.

## 13. Unresolved uncertainties

- Whether a human viewing the 30-fps video would judge T as the stills and metrics do. Not reviewed.
- Domain gap vs conditioning. The context carries source-derived parent popping (244 mm/frame shoulder steps) and an unkeyed right arm at rig rest. This may push the full-body generator out of distribution. One clip and one seed cannot separate that from a general limitation.
- Whether a frozen parent is compatible with a *good* generated arm. Untestable here, because the raw arm is itself unusable.
- Source fidelity of the occluded far arm: UNKNOWN; no valid camera exists.
- Rights: whether AnimCV's purpose stays non-commercial (human-owned), and the training-data provenance that the licensor marks unclear.
- Time sensitivity: bound to code `b78c780`, model `174fc755`, T5 `7bcac572` and the licence texts fetched on 2026-11-12 (byte-equal to 2026-11-11).

## 14. Reproduction

Git Bash, repository root at `a6776b9`. The W136 scratch tree (env, model, hf, results) must exist with its manifest hash `b25805a4…`. Every stage refuses an existing output directory.

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w137/results
H=scripts/w137_unimate_inbetween_harness.py
$PY -m pytest -q tests/test_experimental_unimate_inbetween.py tests/test_experimental_unimate.py tests/test_experimental_elbow_prior.py tests/test_experimental_motionbricks.py
$PY -X utf8 $H preflight --out-dir $R/preflight
$PY -X utf8 $H declare   --out-dir $R/declare --preflight $R/preflight/preflight.json
D="--declaration $R/declare/declaration.json --declaration-sha <sha of the new declaration>"
$PY -X utf8 $H controls  --out-dir $R/controls --preflight $R/preflight/preflight.json $D
$PY -X utf8 $H run       --out-dir $R/run --preflight $R/preflight/preflight.json --controls $R/controls/controls.json $D
# write $R/review/visual_review_S1.json, then:
$PY -X utf8 $H assess    --out-dir $R/assess --preflight $R/preflight/preflight.json --controls $R/controls/controls.json \
    --run-dir $R/run --visual-review $R/review/visual_review_S1.json $D
```

A re-run of `declare` gets a new timestamp and therefore a new SHA; the official SHA is `9dc6d74e…`. The inference is deterministic only for the same seed, GPU, driver and torch build, and it was not re-run.

## 15. Evidence and changed files

**Evidence**, under `C:/Users/dna10/animcv-w137/results`; local, not published. `evidence_manifest.json` lists 3,517 files and was written before this Worklog; SHA-256 `8561d239f6a95aa849da92555d35c9d1e38ae79768ac0a997f6bd53c9a5b7a19`.

| File | SHA-256 |
|---|---|
| `preflight/preflight.json` / `preflight/contract/contract.json` | `e331be8d…` / `300820db…` |
| `declare/declaration.json` | `9dc6d74e…` |
| `controls/controls.json` | `665d5046…` |
| `run/model/raw_model_io.npz` / `infer_record.json` | `68e636c0…` / `38eeeb83…` |
| `run/recover/recovered.npz` / `run/raw_world.npz` | `0fd7cf61…` / `349fc1a5…` |
| `run/import_T.json` | `83c50f96…` |
| `run/run_result.json` | `e0b83668…` |
| `run/abcdeut_S1_30fps.mp4` | `992be516…` |
| `review/visual_review_S1.json` | `360a2beb…` |
| `assess/assessment.json` | `0d582a0c…` |
| `posthoc/posthoc_notes.json` / `posthoc/raw_vs_proxyB_sticks.png` (POST-HOC, not used for the CASE) | `11dcaeb7…` / `8048b8fc…` |

**Changed files (repository):**

- New:
  - `src/app/experimental_unimate_inbetween.py`
  - `scripts/w137_unimate_inbetween_infer.py`
  - `scripts/w137_unimate_inbetween_harness.py`
  - `tests/test_experimental_unimate_inbetween.py`
  - `examples/experimental_unimate/w137_declaration.json`
  - this Worklog
- Unchanged:
  - `docs/next_batch.md`, every historical Worklog and audit, and the living status/architecture documents (no milestone claimed);
  - the W129/W135/W136 modules, Blender stages, runners and harnesses (imported/called, not edited);
  - CLI, GUI, exporter/importer, FramePose, MotionGraph, Production retargeting, schemas;
  - `scripts/batch_automation.py`;
  - the AnimCV venv and the W136 UniMate env;
  - every W126/W129/W135/W136 file.

## 16. Resource and environment constraints

| Resource | Cost |
|---|---|
| Wall time | preflight 28 s; controls 7 s; run 70 s (inference subprocess incl. model load 11.6 s, of which sampler 4.7 s; no-model recovery 1.6 s; 4 builds, 4 readbacks, 2 dumps, 7 × 480 renders, video, 8 sheets); assess < 5 s |
| GPU | RTX 5080, peak 0.45 GB allocated by torch during inference |
| Disk | 419 MB new scratch (results); the W136 env (Python + uv cache), checkpoint (1.19 GB) and T5 snapshot were re-used in place, not copied |

- The UniMate runtime stays separate from the AnimCV venv (Python 3.10 / bpy 4.0 vs 3.11).
- torch 2.7.1+cu128 is needed for `sm_120` (as W136).
