# Worklog — Pretrained Generator Qualification and One Kimodo No-Text Real-Rig Trial on S1: Kimodo Selected, Soft Keyframes Miss the Frozen Boundary, CASE C (2026-11-13)

Batch 138 was an experimental, research-only batch on `arch/single_frame_first`. It ran unattended on Main:

- local RTX 5080 (16 GB, driver 596.49), Windows 11 Pro 10.0.26200;
- the AnimCV Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9), **unmodified**;
- a new scratch Kimodo runtime `C:/Users/dna10/animcv-w138/env/km311`: CPython 3.11.9, torch 2.7.1+cu128 (`sm_120`), transformers 5.1.0, built with uv. The pinned Kimodo clone is put on `sys.path`; it is never pip-installed or modified;
- Blender 5.1.0 for every build, dump, readback and render.

Other conditions:

- `docs/next_batch.md` was only read.
- No LabServer and no other machine was used.
- No LLM or other external API was called.
- Network use was limited to:
  - `git fetch` and `git push`;
  - one anonymous download of the ungated, pinned Kimodo checkpoint;
  - the public metadata, licence and documentation records fetched by the qualification instrument in the interrupted first attempt.

Every number, render and verdict below is **AGENT-RUN** on Main. No reviewer has rerun the GPU or Blender steps. The visual judgement is the agent's own **unblinded still-frame** review. No human or animator reviewed anything, and the agent did **not** perceive native-rate playback. The rights reading is documentary evidence: not legal advice, and not a human licensing decision.

## 0. Resumption

This run **resumed** an interrupted Batch 138.

- Before this run, `git log --grep="^Automation-Batch: 138$"` found two commits from an earlier attempt that a usage limit cut off:
  - `69d2215`: qualification instrument plus frozen selection-rule tests;
  - `4b64557`: frozen selection matrix and decision rule.
- Both commits came before any weight download or model output. They are treated as binding and immutable, and were not redone, amended or re-decided.
- Hash verification:
  - The committed `examples/experimental_kimodo/w138_selection_matrix.json` has SHA-256 `4860bc9f9ea5d39c426e00d1d4cc73a24bc9c85c276f03cdf29bff4ec792c9dd`. It is byte-identical to the evidence copy `C:/Users/dna10/animcv-w138/results/qualification/selection_matrix.json`.
  - The qualification record reports live state OK, all frozen identities OK, and `model_output_exists: false` / `weights_downloaded: false`.
  - A sibling directory `qualification_try1` is an earlier, identical-outcome run of the same instrument at `df9eddc`, made before the instrument commit. It is kept and not used.
- No qualification step was re-run.
- This run continued from the first uncommitted step: experiment declaration, then representation checks, the single inference, Blender and quality, and the Worklog.
- The worker's own commit `0d82883` ("resume an interrupted Batch") is the starting commit of this run. It is not a Batch-138 step.

## 1. Official verdict and scope

**Verdict: CASE C.** There is no STOP code. Qualification selected one eligible model, every pre-inference gate passed, and exactly one real inference ran.

- Source: `results/assess/assessment.json`, SHA-256 `306d2af6a6b3dca61bfcafe12501c33b492fff7f820964a3e1b0d61088a42b20`.
- The frozen rule gives C through two independent triggers:
  1. **M4 conditioning = FALSE.** The raw output's mean keyframe joint-position error is **58.2 mm**, above the declared 50 mm bound. At the boundary keyframe Action 171 it is 127.7 mm.
  2. **Projected unusable continuity.** K's arm-local elbow entry step (171→172) is **109.8 mm**, above the 100 mm limit.
- Raw full-body continuity is **usable** under the declared raw rule. The largest raw excess over PROXY_B is 43.5 mm (wrist) and 85.9 mm (left-arm group boundary), both under 100 mm.
- Under the declared rule the failure is attributed to **FROZEN_PARENT_PROJECTION**. §9 explains exactly what this means here.

**What the run settles:**

1. **One independently licensed, accessible, source-conditionable pretrained generator exists and runs on this rig.** Kimodo-SOMA-RP-v1.2 has Apache-2.0 code and NVIDIA Open Model License weights, and is ungated. It ran in its documented no-text, constraints-only mode with vendor full-body keyframes. No gated Llama 3 component was requested.
2. **Its interior arm is a qualitatively different and less broken proposal than UniMate's.**
   - No lateral straight-arm detour, no body collapse, and steps several times smaller than U/T.
   - Arm-local wrist max step 88.8 mm (U 480.5, T 300.4); elbow 109.8 mm (U 232.0, T 233.6).
   - The world wrist rises 203 mm across the interior (B 284).
   - It still fails the declared continuity and conditioning gates, and is not better than the no-model slerp B.
3. **The dominant failure is the soft keyframe/boundary mismatch, not raw-motion breakage.**
   - Kimodo's constraints are soft (post-processing was declared off).
   - The raw clip is internally smooth, but it misses the frozen context. Its root sits 111 mm off PROXY_B at Action 171, and the boundary keyframe error is 128 mm.
   - Importing only interior rotations against a hard-frozen frame 171 turns that miss into an arm-local entry jump.

**Scope:**

- Scratch experiment only. K, Kc, Kpin and Krev are **SCRATCH PROPOSALS**. They were never accepted and are in no lineage.
- No Production, CLI, GUI, exporter, FramePose, MotionGraph, schema, Unity or FBX change. `scripts/batch_automation.py` is untouched.
- Private, local, non-commercial technical evaluation only. No weights or outputs were redistributed.
- Exactly one inference. No training, finetuning, sweep or post-hoc repair.

**Historical CASEs and artefacts are untouched:** W129 B, W135 B, W136 C, W137 C. No historical Worklog, audit, or W129/W135/W136/W137 file was edited.

**Original task scope (short).** Qualify alternative pretrained motion generators against their real code, checkpoints, permissions and I/O contracts. Select at most one by a frozen rule. If it is compatible with the authentic S1 partial-observation task, run one frozen, source-bound inference. Import only `upperarm_l`/`lowerarm_l` rotations at Action 172–199 into a reversible scratch Action. Compare against B, W136 U and W137 T, with source, temporal, constraint and real-rig checks. Otherwise stop with QUALIFICATION_BLOCKED.

## 2. Revisions

| Item | Value |
|---|---|
| Batch issue commit | `df9eddc` (instruction blob `8129b0b504cfdca11c313ffcd6c3a53d75e2bd32`) |
| Starting commit of this run | `0d82883a063205ed0597ab7d7d9e5b1d01da31e0` (blob verified first, equal to the bound blob) |
| Pre-outcome commits, earlier attempt | `69d2215` (qualification instrument + 33 rule tests), `4b64557` (selection matrix) |
| Instrument commit, this run (pre-outcome) | `26796b0`: adapter, Kimodo-side runner, harness, 29 synthetic tests |
| Declaration commit (pre-outcome) | `b836660`: `examples/experimental_kimodo/w138_declaration.json`, byte copy of `results/declare/declaration.json`, SHA-256 `b9adcdf5757ac32218da38ab98ac62bcda5ae96722f48b5fe3303450df85475c`; pushed before the inference |
| Final source revision | `b836660` for code and declaration. This Worklog is documentation only; the COMPLETE marker is the last commit. |
| Remote | `origin/arch/single_frame_first` had not advanced at either fetch |
| Scratch root | `C:/Users/dna10/animcv-w138`, outside the repository |

**Order of stages:**

1. qualification and selection (earlier attempt);
2. env build and checkpoint download;
3. input-only trial preflight, trial controls and a no-model code-path test of the run stage;
4. instrument commit;
5. official preflight;
6. declaration, then the declaration commit and push;
7. controls;
8. run (the only inference);
9. visual review;
10. assess.

`declare` and `run` refuse to proceed if any `raw_model_io.npz` or `K_S1.blend` exists under the W138 results.

## 3. Candidate qualification (frozen, earlier attempt)

Source: `results/qualification/qualification.json` and the committed matrix (dated 2026-11-12T21:59:50Z).

- Predicates are checked as normalised substrings of primary sources read at pinned revisions: licence files, READMEs, docs, code lines and HF card records.
- A predicate counts as true only if VERIFIED_TRUE.

| Candidate | Code (pinned) / licence | Weights (pinned) / licence | Status | Blocking reason |
|---|---|---|---|---|
| **Kimodo** | [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) `e1c219c` / Apache-2.0 | `nvidia/Kimodo-SOMA-RP-v1.2` `0e3c890`, ungated / NVIDIA Open Model License; `model.safetensors` `fdcef624…` (1.13 GB) | **ELIGIBLE, SELECTED** | — (text mode ACCESS_BLOCKED; no-text constraints-only mode verified by source) |
| ARDY | [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy) `e706e27` / Apache-2.0 | `nvidia/ARDY-SOMA-RP-30FPS-Horizon60` `2442150`, ungated / NVIDIA Open Model Agreement | ELIGIBLE, not selected | Loses P2: autoregressive, 4 frames per token |
| MotionLab | [Diouo/MotionLab](https://github.com/Diouo/MotionLab) `8b2f7b3` / no licence | Google Drive, no terms; HF `nielsr/motionlab` third-party, unlicensed | RIGHTS_UNRESOLVED | E1c, E1w |
| ProjFlow | [Akihisa-Watanabe/ProjFlow](https://github.com/Akihisa-Watanabe/ProjFlow) `9550501` / no licence | `Akihisa-Watanabe/ProjFlow` `ffaba8d` / MIT (author) | RIGHTS_UNRESOLVED | E1c (code); positions only |
| GEM-X | [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) `4aef145` / Apache-2.0 | `nvidia/GEM-X` `b111f2d` / NVIDIA Open Model License | CONDITIONING_BLOCKED | Video pose estimator, no motion/keyframe input |

**Selection rule (frozen).** A candidate is eligible iff all six predicates are VERIFIED_TRUE:

- E1c: code rights;
- E1w: weight rights;
- E2: access, with no gated artefact requested;
- E3: resources;
- E4: native partial-motion conditioning that leaves the interior free;
- E5: per-frame rotations on a skeleton with an existing closed-form BaseRig mapping.

Among eligible candidates, selection is the unique maximum of (P1 vendor-benchmarked partial-motion conditioning in the declared text mode, P2 per-frame non-autoregressive whole window, P3 native 30 fps). The result: Kimodo (T, T, T) beats ARDY (T, F, T).

**Gated text encoder.** The user's existing HF credential gets HTTP 403 `GatedRepo` on `meta-llama/Meta-Llama-3-8B-Instruct`. **Text-conditioned Kimodo is ACCESS_BLOCKED and was not used.**

E2 holds only for the no-text mode, and rests on two facts:

- the vendor `_generate` zeroes an empty prompt's feature and length;
- `load_model(text_encoder=…)` is a documented parameter.

This run re-verified that reading against the pinned source:

- LLM2Vec returns `(B,1,4096)` with lengths 1;
- `use_text_mask: false`;
- the separated-CFG text term is identically zero for an empty prompt.

**Rights caveats (recorded, not resolved; human-owned).**

- NVIDIA OML §2.1 guardrail clause: the gated repository is Meta's Llama base, not a mechanism contained in the Kimodo checkpoint, and it is never requested. This is the agent's documentary reading.
- §2.3 Trustworthy-AI terms were fetched and hashed but not analysed clause by clause.
- Training data is the licensor's proprietary Bones Rigplay 1 mocap.
- Commercial use was not evaluated.

**Previously evaluated (not re-qualified):**

| Model | Batch | Outcome |
|---|---|---|
| MotionBricks | W129 | B |
| AnimateCanvas | W130 | licence-blocked |
| CondMDI | W131/W132 | B/C |
| OmniControl | W133 | licence-blocked |
| ARDY text path | W134 | ACCESS_BLOCKED |
| MotionBricks elbow prior | W135 | B |
| UniMate | W136/W137 | C/C |

## 4. Identity, access, resources (M0)

Source: `results/preflight/preflight.json`, `aedc2beb34f471098f9246deb4357f435e791610c36f82f2fd38fb44bd5f3f30`. Status OK.

- **Selection gate:**
  - prior commits and both start commits are ancestors of HEAD;
  - matrix SHA equals the committed SHA and the evidence copy;
  - selected = Kimodo, mode `no_text_constraints_only`.
- **Code:** clone HEAD `e1c219c`, clean tracked tree, no untracked non-cache files. SHA-256 recorded for 16 key files.
- **Checkpoint:**
  - downloaded anonymously from `resolve/0e3c890…` after the selection;
  - all 7 LFS files equal the published LFS SHA-256 (`model.safetensors` `fdcef624…`, six `stats/motion/*` files);
  - `LICENSE`, `README.md` and `config.yaml` are byte-identical to the qualification copies;
  - loaded via `CHECKPOINT_DIR` with HF offline, an empty `HF_HOME`, no token in the environment and an unroutable text-encoder URL.
- **Frozen identities: 63, all OK before and after the run (M9):**
  - W137's 55, covering the S1 clip `de8feb9d…`, A `0803ab2b…`, B `8899f433…`, C, Cpin, W135 D/E, the W126V accepted pin `7e2ab165…`, W126 refs/proposal, W129/W135 records, and the W136 manifest with U `e0fc3fe3…`;
  - plus the W137 manifest `8561d239…` and W137 T `95f25850…`, Tc, readback/dump/run/declaration/assessment.
- Fresh Blender readbacks of byte copies of A, B, C, D, E, U, T and the W126V pin are **world-bit-equal to W137's** on all 240 frames and 717 subframes.
- **Runtime:** `env/freeze.txt` `49b1c764…`; GPU 2.2 GB used before the run; 374 GB disk free.

## 5. Representation and no-model controls (M1)

| Check (no model) | Result |
|---|---|
| Kimodo SOMA77 skeleton | Names, parents and neutral joints from the pinned assets. Identity rest. Hip height 1.0052 m above the lowest joint. W129 hip scale 1.07457. |
| numpy W129 FK on Kimodo SOMA77 vs vendor FK (random poses) | 4.5e-7 m |
| Vendor 30←77 slice | Equal to the 30 W129 SOMA model joints; registry folder `Kimodo-SOMA-RP-v1.2`; vendor `sanitize("")` = `""` |
| W129 Mapping on Kimodo SOMA77 | Built from B's rest; no-mirroring check OK at 5 frames |
| A == B (world, bit-equal) at all 18 keyframes | True |
| PROXY_B direct mapping decode vs B keys | shift 0: 8.3e-8 / 1.1e-7; ±1: 0.221 / 0.221 (time-separable) |
| Vendor feature roundtrip (encode, normalise, unnormalise, inverse, 30→77) vs B keys | shift 0: 1.1e-7 / 1.5e-7; ±1: 0.221 |
| Vendor FK of the conditioning vs numpy FK | 2.0e-7 m |
| Vendor `FullBodyConstraintSet` + `create_conditions_from_constraints_batched` | Mask constrains exactly rows 0..8 and 37..45, 95 channels each; no interior row. Frame-0 heading 2.5687 rad. Canonical shift (+0.184, 0, −0.030) m. |
| Torso-only affine camera on A (156–215) | RMS **23.3 px > 20 px → INVALID** (as in W136/W137) |

**Input-only measurements before the declaration (disclosed):**

- the feature roundtrip error (1.5e-7);
- the numpy-vs-vendor FK error.

The feature tolerance 1e-5 and the M8 tolerance 1e-5 m (both equal to W137's) were confirmed after these measurements.

A no-model code-path test of the run stage was done in the scratchpad, using the PROXY_B roundtrip as a stand-in output. It found one representation detail before the instrument commit: the vendor 77-joint output inserts relaxed-hand finger rotations, which move only the four hand-end joints. Raw and adherence analysis therefore use the vendor's own 30-joint posed joints. This is the representation the constraint was built in. A numpy cross-check covers the other 26 joints and is part of M4.

## 6. Prospective declaration (summary; the committed JSON is the full record)

| Item | Frozen value |
|---|---|
| Window / time | Action 163..208 (SOURCE 241..286), 46 frames at 30 fps; index i ↔ Action 163+i; Action = SOURCE − 78 |
| Keyframes | Vendor `fullbody` (all 30 SOMA30 joint positions, default smooth-root 2D, root height and heading) at Action 163..171 and 200..208, 18 frames, under the vendor's sparse limit of 20. Content: PROXY_B (== A, bit-equal) mapped to Kimodo SOMA. |
| Interior | Action 172..199 (indices 9..36, 28 frames): **no constraint of any kind** (UNKNOWN; no fabricated 3D) |
| Inference | `load_model('Kimodo-SOMA-RP-v1.2', text_encoder=<null>)`; one `Kimodo.__call__('', 46, num_denoising_steps=100, constraint_lst=[fullbody], cfg_weight=[2.0,2.0], cfg_type='separated', first_heading_angle=<frame-0 heading>, post_processing=False)`; `seed_everything(10)` |
| Choices made before output | Seed 10 matches W136/W137 (not tuned). CFG and steps are vendor defaults. Post-processing OFF: it needs a C++ extension and would alter the raw output. Adherence bound 0.05 m. |
| Recorder | `Kimodo._generate` wrapped read-only. It validates the received mask before sampling and refuses a second call. |
| Import | Only `upperarm_l`/`lowerarm_l` at 172..199, via the unchanged W129 `build_variant`. Generated keyframe frames and every other joint are never imported. |
| Matched control | Kc = PROXY_B through the vendor feature roundtrip and the identical decode/import path |
| Gates | M0–M12, where M12 is import fidelity. Q1–Q8 as W137 with K in place of T. Q8 declared NOT REVIEWED, so **CASE A was unattainable**. |
| Raw rule / attribution / CASE rule | As W137, but evaluated in Kimodo's canonical model space. STOP > D > C > B > A. |

## 7. The single inference

Sources:

- `run/model/infer_record.json`, `2bd40546…`;
- `run/model/raw_model_io.npz`, `3fd19ad8a33b38b3af4fef8616c3d22b996e3a068b0ecb4d0b8e2d5c90d898f9`.

**Inference:**

- exactly 1 `_generate` call;
- the null encoder was called once, with `['']`;
- model load 0.61 s; sampler 1.40 s (100 DDIM steps × 3-way CFG batch); `__call__` 1.51 s; total 2.27 s;
- **peak VRAM 1.20 GB** (1,200,727,040 B); TF32 off;
- output shapes: `local_rot_mats` (46,77,3,3), `root_positions` (46,3).

**Conditioning (M4):**

- the received mask equals the declared 18 rows, validated before sampling;
- the received `observed_motion` equals the no-model encode-stage tensor to 3.0e-7;
- the numpy-vs-vendor FK cross-check passed;
- the interior is not a copy (exact-equal fraction 0.0);
- **keyframe adherence failed:** mean 58.2 mm, max 248 mm.

Keyframe adherence by joint group (mean mm):

| Group | Mean error (mm) |
|---|---|
| root | 48 |
| torso | 49 |
| head | 53 |
| clavicles | 53–55 |
| legs | 54 |
| right arm | 68 |
| left arm | 72 |

Keyframe adherence at the boundary keyframes: Action 171 = 127.7 mm, Action 200 = 28.3 mm. The raw root sits 111 mm off PROXY_B at Action 171 (and at frames 163–171) and 21 mm off at 200.

## 8. Blender Actions and gates

| File | SHA-256 | Action |
|---|---|---|
| K `K_S1.blend` | `31538b3a869f1f2d57a8f4bd4378b664a386281b2ac987a7619e3923cc548daa` | `AnimCV_EXP W138 K Kimodo notext S1` |
| Kc `Kc_S1.blend` | `d22894f12e0b72367212fda7558ac4a11118d77432b7998136a48c83c613f343` | `AnimCV_EXP W138 Kc PROXY_B control S1` |
| Kpin (K on W126V) | `1aed46aa09e2e4026cb4d88687cdd28e86994260ca0ca8a25f46d3d07d95c268` | |
| Krev (K with A's keys restored) | `0fb9ee94449a1fb47a256bc10bbf7031fef684b8560ee69f85a20a80f353dd2d` | |

| Gate | Result |
|---|---|
| M0 selection/identity/rights/access/resources | PASS |
| M1 representation (§5) + in-run Kc feature time identity | PASS |
| M2 one named Action, 32 CONSTANT F-curves keyed 1–240, 30 fps (K, Kc) | PASS |
| M3 footprint: 224 changed keys, only `upperarm_l`/`lowerarm_l` at 172–199; Action 171 and 200 bit-equal to A; every non-left-arm world point and object matrix bit-equal on 240 frames + 717 subframes | PASS |
| **M4 conditioning** | **FAIL** (adherence 58.2 mm > 50 mm); mask, observed-motion equality, FK cross-check and copy detector PASS |
| M5 limb lengths constant (≤ 0.1 mm), unit finite quaternions | PASS |
| M6 hard pin: Kpin residuals at 132–140 equal W126V's (max 6.36e-7 m ≤ 1 mm target tolerance); pin frames bit-equal | PASS |
| M7 reversibility: Krev world-bit-equal to A on all frames and subframes | PASS |
| M8 matched control: Kc vs B max world difference 2.4e-7 m | PASS |
| M9 integrity (63 identities + checkpoint after the run) / M10 controls (41/41 fail closed) / M11 video (240 frames decoded at 30.000 fps) | PASS |
| M12 import fidelity: stored keys vs imported 3.0e-8; evaluated W129 basis vs imported 7.2e-8 | PASS |
| GENERATED_EFFECT (K vs Kc) | max **189 mm**; mean wrist 131 mm, elbow 88 mm |

## 9. Real animation quality and raw-vs-projected attribution

**Arm-local** (clavicle_l frame, steps 171→172 … 199→200), from fresh readbacks. A–E, U and T reproduce the W137 table exactly.

| Metric | A | B | C | D | E | U | T | **K** |
|---|---|---|---|---|---|---|---|---|
| Wrist entry / exit (mm) | 93.2 / 2.9 | 24.1 / 25.0 | 89.6 / 9.6 | 24.1 / 25.0 | 24.1 / 25.0 | 146.2 / 177.2 | 300.4 / 159.3 | **88.8 / 85.8** |
| Wrist peak step (mm) @ frame | 203.1 @173 | 25.0 | 156.6 @174 | 25.0 | 25.0 | 480.5 @175 | 300.4 @172 | **88.8 @172** |
| Wrist flicks | 6 | 0 | 0 | 0 | 0 | 4 | 0 | **1** (172-173-174, cos −0.05) |
| Elbow entry / exit (mm) | 64.4 / 8.2 | 17.8 / 17.8 | 63.3 / 3.3 | 52.3 / 13.7 | 17.8 / 17.8 | 149.5 / 121.5 | 233.6 / 112.3 | **109.8 / 40.3** |
| Elbow peak (mm) | 214.5 | 17.8 | 84.3 | 61.5 | 17.8 | 232.0 | 233.6 | **109.8** |
| Elbow flicks | 6 | 0 | 0 | 0 | 0 | 3 | 0 | **0** |
| Flex range (°) | 70–129 | 93–112 | 100–144 | 93–112 | 93–112 | 65–162 | 57–153 | **101–126** |
| World wrist net rise 172→199 (mm) | 318 | 284 | 341 | 284 | 284 | 115 | 231 | **203** |
| World wrist dev. from B, mean / max (mm) | 148 / 297 | 0 / 0 | 347 / 579 | 0 / 0 | 0 / 0 | 512 / 847 | 460 / 736 | **131 / 189** |
| 2D left-wrist mean (px; camera INVALID) | — | — | — | — | — | — | — | 231 |

**Quality criteria for K:**

| Criterion | Result |
|---|---|
| Q1 | FAIL |
| Q2 | FAIL (one wrist flick) |
| Q3 | FAIL |
| Q4 | PASS (no near-straight or hinge-opposite frame) |
| Q5 | FAIL: invalid camera, source fidelity UNKNOWN |
| Q6 | PASS (203 ≥ 142 mm) |
| Q7 | FAIL (agent review, §10) |
| Q8 | NOT REVIEWED |

The unusable-continuity diagnostic fires on the elbow entry, 109.8 > 100 mm.

**Anchors (declared findings):**

- K vs U: **IMPROVED**. Both of K's max steps are lower, and K has 1 flick against U's 7.
- K vs T: **NOT_IMPROVED**. Both of K's max steps are lower (88.8/109.8 vs 300.4/233.6), but K has 1 flick against T's 0, which the frozen rule makes decisive.

**Shared parent popping.** The world shoulder step is 244 mm in every branch, A–E, U, T and K. It comes from the preserved source-derived object/spine tracks; no branch solves it, and it is separate from the arm-local discontinuity above.

**Raw full body** (Kimodo canonical model space, Kimodo SOMA proportions; `run/raw_model_space.npz` `f1738f20…`).

Continuity, Action 171→200 (mm):

| View | Shoulder entry / exit / peak / flicks | Elbow entry / exit / peak / flicks | Wrist entry / exit / peak / flicks |
|---|---|---|---|
| PROXY_B | 46 / 23 / 239 / 12 | 46 / 18 / 206 / 8 | 57 / 16 / 244 / 10 |
| Projected (PROXY_B parents + K arm rotations) | 46 / 23 / 239 / 12 | 91 / 17 / 233 / 9 | 110 / 11 / 275 / 10 |
| **Raw model** | 34 / 13 / **54** / 0 | 71 / 13 / **88** / 0 | 100 / 28 / **109** / 0 |

**Further raw-output facts:**

- **Raw is smoother than its own conditioning.** It does not reproduce the source-derived parent popping. Raw peak steps per group are 15–76 mm, against PROXY_B's 164–324 mm.
- **No collapse.** Neck2 height above Hips is 521–537 mm raw against 522–533 mm in PROXY_B.
- **Root drift** against PROXY_B: 111 mm through the leading keyframes (163–171), 142 mm maximum in the interior, 21 mm at 200.
- **Raw positions block vs rotation path:** up to 66 mm in the interior and 80 mm at keyframes. The model's two output channels disagree.
- **Interior deviation from PROXY_B** reaches 673 mm for the left arm and 778 mm for the right arm. The right arm in PROXY_B is the unkeyed rig rest, so the raw right arm moves freely.

**Raw vs projected gap.** The gap reaches 424 mm on the elbow and 508 mm on the wrist. Shoulder translation contributes up to 264 mm and parent orientation up to 442 mm.

**Attribution (declared rule): FROZEN_PARENT_PROJECTION.** The rule fires because:

- the projected arm-local continuity fails (Q1–Q3, elbow entry > 100 mm);
- the raw continuity is usable (no excess > 100 mm);
- raw-minus-projected exceeds 100 mm.

**Interpretation (agent, not part of the frozen record).** Arm-local motion is identical in the raw and projected views, so this is not a projection error inside the interior. The projected entry jump arises because:

- Action 171 is the hard-frozen source pose;
- the model's own frame 171 is 128 mm away from it (soft keyframes);
- its body is shifted about 11 cm.

In the model's own frame, the arm leaves its own 171 smoothly. "Frozen-parent projection" here therefore means the mismatch between soft model keyframes and the hard-frozen boundary and parents. It does not mean the model's arm motion is broken.

## 10. Visual review and perceptual boundary

Review file: `results/review/visual_review_S1.json`, `c3ccb260…`. It was written before `assess`.

**Viewed by the agent (still frames, unblinded):**

- every row of `sheet_arm_entry`, `sheet_arm_middle` and `sheet_arm_exit`. Each row is a synchronized source crop plus the A, B, C, D, E, U, T and K arm close-ups.

**Not viewed:**

- `sheet_arm_overview`;
- all body sheets;
- the 30-fps video `run/abcdeutk_S1_30fps.mp4` (`58e150ac…`: source crop plus A/B/C/D/E/U/T/K body and arm, with an interior bar). It was encoded and machine-decoded (240 frames, 30.000 fps) but **not perceived at native rate**;
- the raw full-body Kimodo output: not rendered, analysed numerically only.

No human or animator reviewed anything.

| Branch / frames | Observation |
|---|---|
| K 170–175 | From the frozen 171 rack (hand by the face), the forearm is vertical at 172 and the hand is above shoulder height with a lateral bent elbow at 173–175. The entry change is visibly larger than B's. **No lateral straight-arm detour** (unlike T/U). |
| K 183–188 | Upper arm raised laterally, forearm near vertical, hand above head: a bent-elbow press posture, comparable to B. The source shows the kettlebells rising beside/above the head. Which is closer to the occluded far arm is undecidable (UNKNOWN 3D, invalid camera). |
| K 196–201 | High, bent, lateral elbow, near-overhead at 199 joining frozen 200; close to B |
| T / U | Lateral near-horizontal detour, as recorded in W136/W137 |
| K, judgement | More natural than B: not demonstrated (abrupt entry). More press-faithful than B: not demonstrated. No flick, wrong elbow or detour named from the stills. The metric records one near-orthogonal wrist reversal. Q7 FAIL. |

**Separated conclusions:**

1. synthetic contract correctness: PASS;
2. official model execution: real, one run, documented no-text mode;
3. model contribution: GENERATED_EFFECT;
4. Blender preservation, pin and reversibility: complete;
5. source fidelity: UNKNOWN;
6. naturalness/continuity: improved over the UniMate failure mode, but failing the declared gates and not better than B;
7. intentional style: not evaluated;
8. authoring utility: not shown;
9. deployment rights: research reading only; commercial use not evaluated; human decision pending;
10. Production readiness: not supported (one clip, one seed, no human or native-rate review).

## 11. Tests and controls (RUN / PASS / FAIL / NOT RUN)

| Item | Kind | Result |
|---|---|---|
| `tests/test_experimental_kimodo.py` (29): window/keyframe contract and 12 keyframe refusals; window domain; fps/length; mask contract and refusals; null-text refusal; SOMA build, identity-rest FK, rotation roundtrip, skeleton refusals; canonical shift; time identity and shift refusal; import domain, sign and refusals; conditioning report and adherence/copy/shape/NaN detectors; continuity, flicks, raw rule, attribution split; hybrid touches only the two arm rotations; neck height | synthetic | **29 PASS** |
| `test_w138_generator_qualification.py` (33) plus re-used `test_experimental_motionbricks` / `unimate_inbetween` / `unimate` / `elbow_prior` | synthetic | **147 PASS** in total, run after the instrument commit (1.2 s) |
| W138 negative controls (41). Keyframes: shifted, include 172/199, missing 171/200, 20 frames, unsorted, duplicate, out of window, dense. Mask: interior row, missing 171/200, shift, all, none, 45 rows, float. Text prompt (2). fps 25; 45 frames; non-canonical root; window domain. Decode: +1 time shift, left/right swap, parent/child swap. Mirrored axis; improper and non-finite rotations; missing SOMA joint. Import frames 171/200, clavicle channel, pin frames, zero quaternion. Ignored keyframes; interior copy; 45-frame positions. Real-Blender `build_variant` refusal of a clavicle channel and of frame 171. | real PROXY_B + real Blender (no model) | **41/41 fail closed** (`controls/controls.json` `b22dc5dd…`) |
| Preflight (vendor skeleton, encode, constraint/mask contract), one inference, 4 builds, 4 readbacks, 2 dumps, 8 × 480 renders, video, 8 sheets | real | **RUN** |
| Assessment | frozen rule | **CASE C** |

**Not run:**

- the full repository suite (Production untouched);
- human, animator, blinded or native-rate review;
- ARDY, which was eligible but not selected, by rule;
- any second inference or seed/CFG/step/keyframe/window sweep;
- vendor post-processing (declared off);
- text-conditioned Kimodo (ACCESS_BLOCKED);
- re-runs of W129, W135, W136 or W137.

**Failed:**

- M4 (adherence) — a declared gate;
- Q1, Q2, Q3, Q5 and Q7 — quality criteria, not tests.

**Weak controls (disclosed):**

- the fps/length controls call the gate on substituted values;
- the decode swap controls perturb the inputs, not the decoder code.

## 12. Negative findings

- **No-text Kimodo does not hold full-body keyframes to the bound this task needs.**
  - 18 vendor keyframes gave a mean joint error of 58 mm and a 111 mm root offset at the leading context.
  - The soft constraint is not "native hard preservation", and the boundary frame 171 is where it is worst.
- **Smooth raw output ≠ usable bounded import.** The raw clip is continuous and plausible, but grafting its interior arm rotations onto the frozen source parents and the frozen frame 171 produces a 110 mm elbow entry jump.
- **The model removed, rather than reproduced, the source-derived parent popping.** This shows the raw body is not tracking the conditioning parents. It deviates up to 160 mm in the torso and 142 mm at the root inside the interior.
- **The output position and rotation channels disagree by up to 80 mm**, as UniMate's did (W137: 284 mm), but much less.
- **One generator run is not a generator comparison.** ARDY stayed unexercised by rule, and so did the text path.

## 13. Unresolved uncertainties

- Whether a human viewing the 30-fps video would judge K as the stills and metrics do. Not reviewed.
- Whether vendor post-processing (exact constraint hitting) would remove the boundary mismatch. Declared off, not run.
- Whether the domain gap causes the poor adherence: the source-derived context has popping, the right arm is at rest and the right arm/neck are unkeyed. One clip and one seed cannot separate this from a general limitation of sparse no-text keyframes.
- The rights items listed in §3: the OML guardrail reading, Trustworthy-AI terms not analysed, commercial use, and training-data provenance under the licensor's own terms. All human-owned.
- Source fidelity of the occluded far arm: UNKNOWN; no valid camera.
- Time sensitivity: bound to code `e1c219c`, checkpoint `0e3c890`, torch 2.7.1+cu128 on this GPU and driver. The inference was not re-run.

## 14. Reproduction

Git Bash, repository root at `b836660`. Requirements:

- the W136/W137 scratch trees with manifests `b25805a4…` / `8561d239…`;
- the W138 env, Kimodo clone and checkpoint as recorded.

Every stage refuses an existing output directory.

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w138/results
H=scripts/w138_kimodo_harness.py
$PY -m pytest -q tests/test_experimental_kimodo.py tests/test_w138_generator_qualification.py tests/test_experimental_motionbricks.py \
    tests/test_experimental_unimate_inbetween.py tests/test_experimental_unimate.py tests/test_experimental_elbow_prior.py
$PY -X utf8 $H preflight --out-dir $R/preflight
$PY -X utf8 $H declare   --out-dir $R/declare --preflight $R/preflight/preflight.json
D="--declaration $R/declare/declaration.json --declaration-sha <sha of the new declaration>"
$PY -X utf8 $H controls  --out-dir $R/controls --preflight $R/preflight/preflight.json $D
$PY -X utf8 $H run       --out-dir $R/run --preflight $R/preflight/preflight.json --controls $R/controls/controls.json $D
# write $R/review/visual_review_S1.json, then:
$PY -X utf8 $H assess    --out-dir $R/assess --preflight $R/preflight/preflight.json --controls $R/controls/controls.json \
    --run-dir $R/run --visual-review $R/review/visual_review_S1.json $D
```

Env rebuild:

```
uv venv --python <CPython 3.11.9> env/km311
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128
uv pip install hydra-core omegaconf "numpy<2.4" scipy transformers==5.1.0 peft einops tqdm packaging pydantic filelock safetensors huggingface_hub
```

The full pin list is `env/freeze.txt`. The checkpoint files come from `https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2/resolve/0e3c89024d896064b6851201373f7834784a386b/<file>`.

A re-run of `declare` gets a new timestamp and therefore a new SHA; the official SHA is `b9adcdf5…`.

## 15. Evidence and changed files

**Evidence**, under `C:/Users/dna10/animcv-w138/results`; local, not published.

- `evidence_manifest.json` lists 4,102 files and was written before this Worklog; SHA-256 `0e91b1712130eb77c21142bfdf23c14298a69cd9b3a5d95073889a04261a38d1`.
- `*_try1` directories are pre-declaration trial runs with no model output.

| File | SHA-256 |
|---|---|
| `qualification/selection_matrix.json` (= committed matrix) | `4860bc9f…` |
| `preflight/preflight.json` | `aedc2beb…` |
| `declare/declaration.json` (= committed copy) | `b9adcdf5…` |
| `controls/controls.json` | `b22dc5dd…` |
| `run/model/raw_model_io.npz` / `infer_record.json` | `3fd19ad8…` / `2bd40546…` |
| `run/raw_model_space.npz` / `run/import_K.json` | `f1738f20…` / `8b707375…` |
| `run/readback_K.json` / `run/dump_K.json` | `4c616a23…` / `cc703a1c…` |
| `run/run_result.json` | `aa93a2ff…` |
| `run/abcdeutk_S1_30fps.mp4` | `58e150ac…` |
| `review/visual_review_S1.json` | `c3ccb260…` |
| `assess/assessment.json` | `306d2af6…` |

**Changed files (repository, this Batch):**

- New, earlier attempt:
  - `scripts/w138_generator_qualification.py`
  - `tests/test_w138_generator_qualification.py`
  - `examples/experimental_kimodo/w138_selection_matrix.json`
- New, this run:
  - `src/app/experimental_kimodo.py`
  - `scripts/w138_kimodo_infer.py`
  - `scripts/w138_kimodo_harness.py`
  - `tests/test_experimental_kimodo.py`
  - `examples/experimental_kimodo/w138_declaration.json`
  - this Worklog
- Unchanged:
  - `docs/next_batch.md`, every historical Worklog and audit, and the living status/architecture documents (no milestone claimed);
  - the W129/W135/W136/W137 modules, Blender stages and harnesses (imported, not edited);
  - CLI, GUI, exporter, FramePose, MotionGraph, Production, schemas;
  - `scripts/batch_automation.py`;
  - the AnimCV venv;
  - every W126/W129/W135/W136/W137 file.

## 16. Resource and environment constraints

| Resource | Cost |
|---|---|
| Wall time | env build ~1 min (torch wheel 39 s); checkpoint download ~1 min; preflight 27 s; controls ~20 s; run 69 s (inference subprocess 2.3 s incl. 0.6 s model load and 1.4 s sampling; 4 builds, 4 readbacks, 2 dumps, 8 × 480 renders, video, 8 sheets); assess < 5 s |
| GPU | RTX 5080, peak 1.20 GB allocated by torch during inference |
| Disk | 5.9 GB env (incl. uv cache), 1.1 GB checkpoint, 336 MB code clones, 493 MB results |

- The Kimodo runtime stays separate from the AnimCV venv.
- Only the declared no-text path's dependencies were installed (no gradio, viser, SOMA-X or C++ `motion_correction`).
- The gated Llama 3 base and the LLM2Vec adapters were never downloaded or requested.
