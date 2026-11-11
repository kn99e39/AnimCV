# Worklog — UniMate v3 Source-Bound Left-Arm Generation on the Authentic S1 Rig: Real Inference, Unusable Continuity (2026-11-11)

Batch 136, an experimental, quality-first, research-only batch on `arch/single_frame_first`. It ran unattended on Main:

- local RTX 5080 (16 GB, driver 596.49), Windows 11 Pro 10.0.26200;
- AnimCV Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9), **unmodified**;
- a separate scratch UniMate runtime `C:/Users/dna10/animcv-w136/env/um310` (uv-managed CPython 3.10.22);
- Blender 5.1.0 for every AnimCV-side build, dump, readback and render.

`docs/next_batch.md` was read only. No LabServer and no other machine. Network use was limited to `git fetch/push`, a
code clone of the official UniMate repository, public ungated Hugging Face downloads (UniMate metadata, card, licence,
v3 config/stats/checkpoint; `google/flan-t5-base` metadata, card and weights) and the scratch `uv pip` installs. No
LLM or other external API was called; the default UniMate LLM labeller was not used.

Every number, render and verdict below is **AGENT-RUN** on Main. Every visual judgement is the agent's own **unblinded
still-frame** review. No human or animator reviewed anything, and the agent did **not** perceive native-rate playback.
The rights reading is documentary evidence, not legal advice and not a human licensing decision.

## 1. Official verdict and scope

**Verdict: CASE C** under the frozen rule (no STOP code: every pre-inference gate passed and one real inference ran).
Source: `results/assess/assessment.json`, SHA-256 `ed158590c9a9252a13533d62a4b2c530fc9cc0465bd0699c3b089fb183deb5df`.

The CASE comes from three separate findings:

1. **The source-bound conditioning path works mechanically.** The official own-rig route represents the S1 BaseRig after
   a declared, world-pose-preserving export adapter; a no-model roundtrip of B through UniMate and back into Blender
   reproduces B to 0.48 µm; the official sampler holds the declared context exactly; the generated keys import into a
   separate, reversible Action that keeps the hard pin. Mechanical gates M0–M11 **all PASS**.
2. **A genuine model contribution exists: GENERATED_EFFECT.** U (model) differs from the matched no-generator control
   Bc (= B through the identical path) by up to **847 mm** (wrist, Action 178); mean 512 mm wrist, 268 mm elbow.
3. **The generated motion is not usable animation.** It fails the declared continuity threshold and every applicable
   quality criterion:
   - arm-local wrist entry/exit steps **146 / 177 mm** (B 24 / 25 mm), arm-local wrist peak step **480 mm** at 175;
     elbow entry/exit 150 / 122 mm (B 18 / 18 mm). The declared "unusable continuity" limit is any step > 100 mm.
   - 4 wrist and 3 elbow reversal flicks (B: 0).
   - wrist net rise over 172→199 only **115 mm** (B 284 mm; Q6 needed ≥ 142 mm).
   - still review: the arm swings out sideways into a near-horizontal lateral raise (174–175), stays extended
     outward/downward through the middle (183–188) instead of pressing, and snaps overhead at 199 to meet the frozen 200.

Under the declared rule "C = real inference executed but … unusable continuity (any U arm-local wrist or elbow step in
171→172 … 199→200 > 100 mm)", the result is **C**. Even without that clause it could not exceed B: Q1–Q3 and Q5–Q8 fail.

**Scope:**

- Scratch experiment only. U, Bc, Upin, Urev and R0 are **SCRATCH PROPOSALS**: never committed, accepted or placed in any lineage.
- No Production, CLI, GUI, exporter, FramePose, MotionGraph, schema, Unity or FBX change; `scripts/batch_automation.py` untouched.
- Private, local, **non-commercial** technical evaluation only. No weights or model outputs were redistributed.
- No training, finetuning, seed/prompt/checkpoint/cfg/step sweep or post-hoc repair.

**Historical CASEs and identities are untouched:** W129 B, W130 DATA BLOCKED_LICENSE, W131 B, W132 C, W133 DATA
BLOCKED_LICENSE, W134 DATA BLOCKED / ACCESS_BLOCKED, W135 B. No historical Worklog or audit was edited; W129's
MotionBricks inference and W135's solve were not re-run.

**Original task scope (short).** Use exactly one pinned official UniMate v3 checkpoint and one 60-frame / 30-fps window
containing Action 172–199 to produce one model-inferred left-arm candidate on the S1 BaseRig, conditioned on the frozen
W129 B Action as a GT-free saved-clip proxy ("PROXY_B"). Import only `upperarm_l`/`lowerarm_l` rotations at 172–199
into a new opt-in Action. Compare with A, B, C and W135 D/E, a matched no-generator control, the source video and
native-rate evidence. Stop on any rights, access, resource, representation or conditioning blocker.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `b7fd915af1d8d9d52d51454b1d27cf969df5b1c9` (verified first) |
| Starting commit | `b641905b475ebed63c49fa6c59fe6b40b82d44f0`, equal to `origin/arch/single_frame_first` at the first fetch. W135 carries the COMPLETE marker; no W136 Worklog existed; `docs/AUDIT_W131_W135_2026_11_10.md` is the preceding five-batch audit. |
| Instrument commit (pre-outcome) | `2f3f97f`: module, Blender export stage, UniMate-side runner, harness, synthetic tests |
| Declaration commit (pre-outcome) | `c1bb8e8`: `examples/experimental_unimate/w136_declaration.json`, a copy of `results/declare/declaration.json`, SHA-256 `47259fbfd27a2e3fd762f8ebe362e2c92f616acde5b81cad5ea6de16bf616196` (scratch-file SHA; git normalises the copy to LF) |
| Final source revision | `c1bb8e8` for code and declaration. The Worklog commit is documentation only; the COMPLETE marker is a separate, final commit. |
| Scratch root | `C:/Users/dna10/animcv-w136`, outside the repository, 17.7 GB (env + uv cache, checkpoint, T5, results) |

**Order of stages:** exploration on input-only B data → instruments → preflight (three attempts, §7) → instrument
commit → declare → declaration commit → controls → run (the only inference) → visual review → assess. No instrument,
threshold or rule changed after any model output existed; the declaration refused to write if any model output existed
(`model_output_exists: false`).

## 3. Source, rights, access and resource gate (M0)

Source: `results/preflight/preflight.json`, SHA-256 `67ab4959dc1656cd2554c5087a420efa0951a2530827f982bf938cbed08c6f9f`. Status OK.

**Code.** `https://github.com/Friedrich-M/UniMate` at `b78c7807df7cf4231d4e67f8d0ece29c5e11025b` (2026-10-07, the
remote HEAD; it contains the 2026-10-05 `rig_preprocess` release). Clean tracked tree. `LICENSE`: MIT, "Copyright (c)
2026 Linzhan Mou". Key-file SHA-256s are in the preflight record.

**Model.** `Linzhan/UniMate` revision `174fc755ec248d0c98f27759b7a95648e7f05905`, not gated, card licence `cc-by-nc-4.0`.

| File | SHA-256 |
|---|---|
| `unimate_uniml3d_f60_v3/checkpoints/checkpoint_step_150000.pt` (1,185,836,040 B) | `f991ddd87262c251f4681d18297a6163ab4f139daa28c86d5ef0a66771df7bc6` (local = published LFS) |
| `unimate_uniml3d_f60_v3/dataset_stats.npy` | `0ff2b9ef3ea9a15b024ab0670e55f659478251501e5596f9011d5bca4cf67773` (local = LFS) |
| `unimate_uniml3d_f60_v3/config.json` | `1ffbd379d3667de7d701ed6f3418f1c9d1a66f35d2653776255008c07c051aaa` |

Config: flow matching, graph attention + AdaLN text, 10 layers, 60 frames, 12-D features, max 71 joints, text encoder
`google/flan-t5-base`, published `sampling.cfg_scale` 3.0 and `sampling.seed` 10. Card: 74.1 M denoiser parameters,
150k steps, UniML3D `d0f19d9`, "Recommended".

**Text encoder.** `google/flan-t5-base` revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`, not gated, `apache-2.0`;
`model.safetensors` SHA-256 `1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8` (local = LFS). The
local snapshot resolved to exactly the pinned commit; inference ran with `HF_HUB_OFFLINE=1`. No gated access was needed.

**Rights determination: PERMITTED_NONCOMMERCIAL_RESEARCH_EVALUATION.** The fetched weight `LICENSE` (at the pinned
revision) states "The model weights and other files in this repository are licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)", "They are released for non-commercial research
use", and grants (§2(a)(1)) the right to "reproduce and Share the Licensed Material, in whole or in part, for
NonCommercial purposes only". All required clauses were found mechanically. The intended use was declared as a private,
local, non-commercial technical research evaluation with no redistribution.

**Unresolved and not granted:**

- **Commercial use is not granted** (CC BY-NC 4.0). Nothing here approves product or commercial use.
- **The determination depends on the declared non-commercial purpose.** If AnimCV work is primarily intended for
  commercial advantage, this licence would not cover it; that judgement is human-owned.
- **Training-data provenance is explicitly unresolved by the licensor.** The card says "This covers only our rights in
  the weights"; "Third parties allege that some of its animal assets come from commercial games … treat the provenance
  of the Truebones-trained models as unclear"; Mixamo is under Adobe's terms; some Objaverse-XL objects are
  non-commercial; and "We make no warranty that the checkpoints, or motions generated with them, are free of
  third-party rights." The v3 model is trained on Truebones ZOO + Mixamo + Objaverse-XL.

**Runtime and resources.** The pinned `torch==2.5.1+cu124` has no `sm_120` kernels, so the scratch env uses
`torch 2.7.1+cu128` (+ the pinned versions of everything else: numpy 2.2.6, transformers 5.16.1, bpy 4.0.0, the `Motion`
package at `ac23625…`, …; `env/freeze.txt` SHA-256 `a8a54c3b…`). torch 2.7 defaults `torch.load(weights_only=True)`;
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` was set for the inference subprocess only, restoring the official code's torch-2.5
semantics for the hash-verified checkpoint (declared). Disk free 392 GB; GPU 16 GB with 1.2 GB used before the run.

**Frozen evidence.** Every frozen SHA-256 (S1 clip `de8feb9d…`, A `0803ab2b…`, B `8899f433…`, C `e04efb35…`, Cpin,
W126V-accepted `7e2ab165…`, W126 refs/proposal, W129 declaration/assessment/run/preflight/engine/raw/projection/manifest,
W135 D `2a5e441f…`, E `ac53a771…`, declaration, assessment, run record, evidence manifest) was equal before and after
the run (M9). Fresh readbacks of byte copies of A, B and C are world-bit-equal to W129's.

## 4. Representation and the no-model roundtrip (M1)

**What the official route would have done to the raw BaseRig (found on input-only roundtrips before declaration):**

- The BaseRig has five parentless bones (`spine_01`, `thigh_l`, `thigh_r`, `interaction`, `center_of_mass`). With a
  skinned mesh, UniMate's exporter keeps only the heaviest root subtree, so **both legs would be dropped**.
- The source-derived root orientation lives on the **armature object**'s rotation keys. The exporter samples
  `matrix_world` at the current frame for the rest pose, so the "T-pose" came out **tilted** by the frame-1 source rotation.

**Declared export adapter** (`src/blender/experimental_unimate_export_stage.py`; byte copy of B, never saved):

1. one static, unskinned `pelvis` bone at the thigh-head midpoint becomes the parent of `spine_01`, `thigh_l`,
   `thigh_r` (keep offset);
2. the object rotation keys are moved onto `pelvis` (Basis = Rest⁻¹·R_obj(f)·Rest, rotation + location about the
   armature origin, all 240 frames) and the object rotation set to identity (scale 0.01 kept);
3. GLB export over Action 156–216 only.

Every anatomical bone's world matrix is unchanged on all 240 frames: max |Δ| **9.5e-7** (refuse > 1e-5). The two
zero-weight helper roots were left alone and merged/pruned by the official exporter.

**Official own-rig preprocessing** (`python -m data_process.rig_preprocess run … --annotate rule --save_clips`, review
stop, then `--profile general --annotation <unchanged> --save_clips`):

- 50 joints = 49 anatomical bones + `pelvis`, BaseRig hierarchy.
- Offline deterministic rule labels; **0 of 50 flagged**. `upperarm_l`→Left Upper Arm, `lowerarm_l`→Left Forearm,
  `hand_l`→Left Hand, `pelvis`→Hips, all sides correct.
- Facing pair `thigh_r`/`thigh_l`. The agent reviewed `annotation_preview.png` and the canonical `preview.png`: upright
  A-pose facing +Z, right side on the image's left in the front view. Mechanical facing cos = 1.000.
- One clip, 61 poses at 30 fps (60 feature frames), not filtered.
- Uniform scale: bone-length ratio spread 1.7e-4.

**Decode (independent numpy implementation).** UniMate's HML layout stores a joint's T-pose-rebased rotation in its
child's 6D slot, so `upperarm_l` = slot of `lowerarm_l` and `lowerarm_l` = slot of `hand_l`; the BaseRig basis is
Basis = tq⁻¹·(cum_p⁻¹·new·cum_p·tq). The glTF/bpy-4.0 roundtrip keeps BaseRig bone frames (exported local rotation =
Rest_rel·Basis_B to ~1e-6). A degenerate or non-finite 6D vector is **refused**, never silently replaced by the
identity as the official decoder does.

| No-model check | Result |
|---|---|
| Own decoder vs official `hml_rotations_to_bvh_quaternions` decode | 3.3e-16 |
| Decoded PROXY_B vs B's `upperarm_l` / `lowerarm_l` keys at shift 0 | 2.5e-7 / 2.7e-7 (quaternion components) |
| Same at shift ±1 frame (time-identity proof) | 0.221 / 0.221 |
| Non-left-arm context drift (`spine_01`, thighs, calves) | ≤ 8.0e-7 |
| Decoded PROXY_B imported into a Blender Action (R0) vs B, all frames | max world difference **0.48 µm**; nothing outside the footprint changed |

**2D observations that actually exist.** MMPose COCO-17 detections of the authentic S1 video (W126
`observations_predictions.npz`). For the left elbow/wrist at SOURCE 250–277, the detections are flagged valid with
confidence ~0.4–0.6, but this is the **self-occluded far arm**. They were recorded as **detector hypotheses**, never used
as model input. The true 3D motion of the left arm is **UNKNOWN**. The right arm is unkeyed (rig rest) in A and B.

## 5. Prospective declaration (summary; the committed JSON is the full record)

| Item | Frozen value |
|---|---|
| Window | Action 156–215 (SOURCE 234–293), feature frame i ↔ Action 156+i, native 30 fps; footprint 172–199 = window frames 16–43 |
| Model input | PROXY_B everywhere except the generated support. B's interior left arm is **not** visible to the model, so the conditioning content is A/B-identical context outside the footprint. |
| Generated support | strict descendants of `upperarm_l` (`lowerarm_l`, `hand_l`, 14 left finger joints) × all 12 channels × Action 172–199, plus their 171→172 velocity channels: 5,424 entries |
| Kept | every other joint, including `upperarm_l`/`clavicle_l`, at every frame |
| Mask mechanism | the official `--motion_edit` path; one scratch hook refines the official (B,J,1,1) joint mask into a (B,J,12,T) joint × channel × time mask (the official sampler docstring allows "any combination as long as the mask broadcasts to (B, J, D, T)"). The hook refuses unless the official joint mask equals the complement of the generated joints. |
| Prompt | "An object presses its left arm overhead." |
| Sampler | seed 10, cfg 3.0 (both the published config defaults); official fixed-step Euler replacement ODE, 50 steps, t ∈ [0, 1]; EMA weights |
| Import | `upperarm_l`/`lowerarm_l` at 172–199 only, via the unchanged W129 `build_variant` on a byte copy of A |
| Matched control | Bc = the official PROXY_B tensor of the same run through the identical decode/import path |

**Thresholds and rule.**

- Mechanical gates M0–M11 (§6).
- GENERATED_EFFECT: max U–Bc elbow or wrist difference ≥ 10 mm.
- Quality criteria Q1–Q8:
  - Q1: arm-local wrist and elbow entry/exit ≤ 2×B + 10 mm;
  - Q2: no flicks;
  - Q3: wrist peak step ≤ 2×B + 10 mm;
  - Q4: no near-straight or hinge-opposite frame;
  - Q5: source-2D with a torso-only affine camera of RMS ≤ 20 px, U's mean wrist error ≤ B's;
  - Q6: wrist net rise ≥ 0.5×B;
  - Q7: agent still review;
  - Q8: native-rate review, declared NOT REVIEWED in advance, so CASE A was unattainable in this run.
- Unusable continuity: any arm-local wrist or elbow step > 100 mm.
- Precedence: DATA_BLOCKED > D > C > B > A.

**Disclosed pre-declaration choices (input-only data).**

- The export adapter (§4).
- The scale-spread tolerance was set to 1e-3 after the real offsets showed a float32 spread of 1.7e-4 on 3–4 cm finger bones.
- The window was restricted at export because the official stage 4 keeps only the first 200 frames of a clip.

## 6. Real inference and the actual-rig candidate

Source: `results/run/run_result.json`, SHA-256 `348f5b889f27efbad5181f6b106a1e4d1f03b324f02f1c53ab34625ba49e8dcc`.

**Inference record** (`run/model/infer_record.json`, `bbc63ec3…`):

- exactly 1 `generate_samples` call on the RTX 5080;
- sampler 4.56 s, whole official `main` 6.0 s;
- peak VRAM **0.45 GB**;
- kept entries of the raw sample equal PROXY_B with **max |Δ| = 0.0** (denormalised);
- generated region max |Δ| vs PROXY_B 1.23 (feature units);
- own decoder vs official decode 4.4e-16.

Raw tensors (input, mask, normalised output, mean/std): `run/model/raw_model_io.npz` `14e80250…`. Official saved
sample and PROXY_B `.npy`: `e68d14f5…` / `fe50b40e…`.

| File | SHA-256 | Action |
|---|---|---|
| U `U_S1.blend` | `e0fc3fe3a0485f5de97c84f9f3363fad0e0675dc09f9b7effd5eef69aaa432d3` | `AnimCV_EXP W136 U UniMate S1` |
| Bc `Bc_S1.blend` | `12e00d64a96f609ad9e6e6f7818bdf7afb39fd9d574b479763d9c2fea336eab3` | `AnimCV_EXP W136 Bc PROXY_B control S1` |
| Upin (U on W126V) | `5809d1c84b8ecebd6ea25480e7a76aa1310b50371efef22710f65fede518246e` | |
| Urev (U with A's keys restored) | `8b0b7cbabdac7b378fa8936e361c5a3132a45579daa1ccff4fade89dc97451ba` | |
| R0 (preflight decoded-B roundtrip) | `523544bcb339cafe0c5032f792b3adbea979ba888b6cf6de289e37dbdf679117` | |

| Gate | Result |
|---|---|
| M0 identity/rights/access/resources | PASS |
| M1 representation (preflight roundtrip + in-run PROXY_B time identity) | PASS |
| M2 one named Action, 32 CONSTANT F-curves keyed 1–240, 30 fps (U, Bc) | PASS |
| M3 footprint: 224 changed keys, only `upperarm_l`/`lowerarm_l` at 172–199; non-left-arm world points and object matrix bit-equal to A on 240 frames + 717 subframes | PASS |
| M4 conditioning: one call, kept entries exact, free set = declared support | PASS |
| M5 limb lengths: U upper/lower range 0.17 / 0.27 µm; unit, finite quaternions | PASS |
| M6 hard pin: Upin residuals at Action 132–140 equal W126V's (max 6.36e-7 m), pin frames bit-equal | PASS |
| M7 reversibility: Urev world-bit-equal to A, all frames and subframes | PASS |
| M8 matched control: Bc vs B max world difference 0.48 µm | PASS |
| M9 integrity, M10 controls (31/31), M11 video (240 frames decoded at 30.000 fps) | PASS |

## 7. Tests and controls (RUN / PASS / FAIL / NOT RUN)

| Item | Kind | Result |
|---|---|---|
| `tests/test_experimental_unimate.py` (19) | synthetic: time map, quaternion/6D, decode = inverse of the official rebase + HML order, degenerate-6D refusal, time-shift detection (incl. motionless clip), mask support and 7 mask controls, labels/side/hierarchy/facing pair, facing/handedness, uniform scale, domain/import/pin, affine camera | **19 PASS** |
| `tests/test_experimental_elbow_prior.py` (14), `tests/test_experimental_motionbricks.py` (15) | reused instruments | **29 PASS** (`results/pytest/focused.xml`, `9c3e7668…`) |
| Real-data controls (`results/controls/controls.json`, `44be7c0f…`) | real S1 features/dumps + real Blender refusals; no model | **31/31 fail closed** |
| Preflight no-model roundtrip, real inference, builds, fresh readbacks, gates | real | **RUN, all mechanical PASS** |
| Native-rate video + 8 contact sheets | real renders | **RUN** |
| Assessment | frozen rule | **CASE C** |

**The 31 controls:**

- labels / side: forearm label swap;
- missing left forearm; wrong facing pair; mirrored T-pose (handedness); backwards facing; joint-order permutation;
- frames shifted +1 and −1; non-uniform scaling (5%); rest/armature mismatch (5° T-pose rotation);
- non-left-arm drift (perturbed spine slot);
- corrupt checkpoint copy; corrupt S1 clip copy; 24 fps; 20 fps;
- mask freeing `spine_01`, the right forearm, frame 170, frame 200 or the `upperarm_l` joint; mask keeping a support
  entry; mask generating nothing;
- channel `spine_01`; frames 171/200; pin-overlap request; degenerate 6D; non-finite features;
- real Blender `build_variant` refusals (`spine_01`, 171, 200);
- real Blender export refusal of a non-B source.

**Preflight attempts (instrument bugs, before any declaration or model output; kept as evidence):**

1. `preflight_attempt1_instrument_bugs`: my README identity clause omitted the markdown link syntax of the card's
   table row, so the mechanical rule reported RIGHTS_UNRESOLVED, and the code check counted untracked `__pycache__` as
   a dirty tree. The checkpoint was downloaded here (21.8 s). The later attempts re-used it only after re-verifying its
   SHA-256.
2. `preflight_attempt2_instrument_bug`: the official tool requires continuing a review with the in-directory
   `annotation.json`; the run passed a copy and stopped.
3. Final preflight: OK.

**Not run:**

- the full repository suite: Production is untouched;
- human, animator, blinded or native-rate review;
- any second inference, seed/prompt/checkpoint/cfg sweep, other model, longer sequence or other clip (forbidden);
- the official `run_animate_motion.sh` mesh driving and the official skeleton mp4 renders (`--only_save_motion`);
- re-runs of W129/W135.

**Failed (quality criteria, not tests):** Q1, Q2, Q3, Q5 (camera invalid), Q6, Q7. Q8 not reviewed. Q4 PASS.

**Weak controls (disclosed).**

- The corrupt-file controls check file identity; they do not exercise a separate code path.
- The 24/20 fps controls call the frame-rate gate on substituted values; no 24-fps clip was exported.

## 8. Comparative quality: A / B / C / D / E / U

**Arm-local** (clavicle_l frame; window 164–207; steps 171→172 … 199→200). Wrist: W129 `armlocal_quality`; elbow:
W135 `elbow_metrics`.

| Metric | A | B | C | D | E | U |
|---|---|---|---|---|---|---|
| Wrist entry / exit step (mm) | 93.2 / 2.9 | 24.1 / 25.0 | 89.6 / 9.6 | 24.1 / 25.0 | 24.1 / 25.0 | **146.2 / 177.2** |
| Wrist peak step (mm) @ frame | 203.1 @173 | 25.0 | 156.6 @174 | 25.0 | 25.0 | **480.5 @175** |
| Wrist reversal flicks | 6 | 0 | 0 | 0 | 0 | **4** |
| Elbow entry / exit step (mm) | 64.4 / 8.2 | 17.8 / 17.8 | 63.3 / 3.3 | 52.3 / 13.7 | 17.8 / 17.8 | **149.5 / 121.5** |
| Elbow peak step (mm) | 214.5 | 17.8 | 84.3 | 61.5 | 17.8 | **232.0** |
| Elbow reversal flicks | 6 | 0 | 0 | 0 | 0 | **3** |

**World space and source 2D:**

| Metric | A | B | C | D | E | U |
|---|---|---|---|---|---|---|
| Flex range, interior (°; 180 = straight) | 70–129 | 93–112 | 100–144 | 93–112 | 93–112 | 65–162 |
| World wrist net rise 172→199 (mm) | 318 | 284 | 341 | 284 | 284 | **115** |
| World wrist mean deviation from A (mm) | 0 | 148 | 408 | 148 | 148 | **539** |
| World wrist max deviation from B (mm) | 297 | 0 | 579 | 0 | 0 | **847** |
| 2D left-wrist mean error (px; camera INVALID) | 225 | 206 | 99 | 206 | 206 | 81 |

U's interior elbow points posteriorly (mean anterior component −0.53; B +0.28, A +0.34). Swivel step max 43°. The world
shoulder moves up to 244 mm/frame in **every** branch: that motion comes from the preserved source-derived object/spine
tracks, is shared, and is not solved by any branch.

**Source-2D fidelity: UNKNOWN.** The torso-only affine camera (both shoulders + hips of A, Action 156–215) fits with
RMS **23.3 px > 20 px**, so Q5 cannot pass by the declared rule. The diagnostic itself is implausible as a fidelity
measure: A, the source-derived reconstruction, scores worst (225 px), and U "scores" best only because its sideways
swing lands nearer the low-confidence far-arm detections. **No fidelity claim is made from it.**

**Visual review** (`results/review/visual_review_S1.json`, `bea298b2…`, written before `assess`).

- **Viewed (unblinded stills):** `sheet_arm_entry`, `sheet_arm_middle`, `sheet_arm_exit`, `sheet_body_overview`.
- **Not opened:** `sheet_arm_overview` and the body entry/middle/exit sheets. The 30-fps video
  (`run/abcdeu_S1_30fps.mp4`, `e9d78b6b…`, source crop + A/B/C/D/E/U body and arm close-ups, interior bar) was produced
  and machine-decoded but **not perceived at native rate**.

| Branch | Observation |
|---|---|
| U, 171–172 | Leaves the A/B rack pose |
| U, 173–175 | The elbow swings out, and the whole arm becomes a near-horizontal lateral raise |
| U, 183–188 | Stays extended outward/downward with a nearly straight elbow while the source shows both kettlebells rising beside the head |
| U, 196–198 | The forearm comes back up with a low, lateral elbow |
| U, 199 | Jumps to a near-overhead pose that meets the frozen frame 200 |
| U, judgement | Not more natural and not more press-faithful than B; named for a detour and abrupt entry/exit |
| B (= E) | Monotonic, even rise from rack to overhead (as in W135) |
| D, C | Unchanged from W135 / W129 |

## 9. Causal and product interpretation (separated)

1. **Synthetic contract tests:** 48 PASS. They say nothing about animation quality.
2. **Real checkpoint execution:** one real inference of the pinned official v3 checkpoint ran on Main through the
   official motion-editing code with only the disclosed mask refinement.
3. **What conditioned the model:** PROXY_B context, i.e. the A/B-identical body and left arm outside 172–199, plus the
   prompt. B's interior was masked, so U's interior motion is model-generated, not copied or interpolated.
   GENERATED_EFFECT is attributable to inference: Bc, through the identical path, equals B to 0.48 µm. The context is
   itself a GT-free reconstruction that carries the source-derived parent popping and an unkeyed right arm. This is
   "source-bound" only through that reconstruction, not through observed 3D data.
4. **Actual Blender preservation:** complete. Footprint, pin, reversibility and integrity hold exactly.
5. **Quality:** negative on this clip. The model did not produce an overhead press for the far arm under this
   conditioning. It produced a lateral detour with entry/exit discontinuities up to 177 mm and an interior peak of 480 mm.
6. **Commercial deployment viability:** not supported. NC-only weights, unresolved training-data provenance, one clip,
   one seed, no native-rate or human review.

One clip and one seed do not establish anything general about UniMate. They show only that this configuration on this
source failed.

## 10. Negative findings

- **The official own-rig route does not accept the AnimCV BaseRig as is.** Without an adapter it silently drops both
  legs (heaviest-root pruning) and bakes a frame-dependent object rotation into the "rest" pose. Both were silent, not
  errors; they were found only by roundtrip checks.
- **Replacement conditioning held the context exactly** (max |Δ| 0.0), yet the generated interior was not coherent with
  it. Boundary steps of 146/177 mm show that clamping frames 171 and 200 by replacement did not make the model join them
  smoothly.
- **The prompt was not followed.** "An object presses its left arm overhead." gave a sideways raise/reach. Whether a
  different phrasing, cfg or seed would differ was **not tested**, by rule.
- **The 60-frame window needed a range-restricted export.** The default stage 4 keeps only the first 200 frames of a clip.
- **The official 6D decoder maps degenerate vectors to the identity silently.** The scratch decoder refuses instead; no
  refusal occurred here.
- **A torso-only affine camera does not fit this side-view clip within 20 px.** Source-2D fidelity of the occluded arm
  could not be assessed on this evidence.

## 11. Unresolved uncertainties

- Whether a human viewing the 30-fps video would judge U as badly as the stills suggest. Not reviewed; the stills and
  the metrics are unambiguous about the discontinuities.
- Whether the failure comes from the domain gap or from the conditioning. The model was trained on rigs whose
  T-pose-relative rotations come from rest poses like this A-pose rig; the context includes the source-derived
  parent popping (244 mm/frame shoulder steps) and a right arm frozen at rest. Not separable here.
- Whether the joint × time mask is in-distribution for this checkpoint. The official sampler supports it mechanically,
  but the published modes are joint-only editing and frame-only in-betweening.
- Rights:
  - whether AnimCV's eventual purpose remains non-commercial (human-owned);
  - the training-data provenance of the v3 checkpoint, which the licensor itself marks as unclear.
- Time sensitivity: bound to code `b78c780`, model revision `174fc755`, T5 revision `7bcac572` and the licence texts
  fetched on 2026-11-11 (hashes in the preflight record).

## 12. Reproduction

Git Bash, repository root at `c1bb8e8`. Every stage refuses an existing output directory. The frozen W126/W129/W135
scratch files must exist at the recorded paths with the recorded hashes.

```
# scratch UniMate runtime (once)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Friedrich-M/UniMate.git C:/Users/dna10/animcv-w136/ext/UniMate
git -C C:/Users/dna10/animcv-w136/ext/UniMate checkout b78c7807df7cf4231d4e67f8d0ece29c5e11025b
UV=C:/Users/dna10/animcv-w129/env/uvboot/Scripts/uv.exe; export UV_CACHE_DIR=C:/Users/dna10/animcv-w136/env/uvcache
UV_PYTHON_INSTALL_DIR=C:/Users/dna10/animcv-w136/env/pythons $UV python install 3.10 && $UV venv --python 3.10 env/um310
$UV pip install --python env/um310/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128
$UV pip install --python env/um310/Scripts/python.exe "setuptools<81" wheel
$UV pip install --python env/um310/Scripts/python.exe --no-build-isolation --extra-index-url https://download.blender.org/pypi/ \
    --index-strategy unsafe-best-match <every pin of UniMate requirements.txt except torch/torchvision/clip/openai/google-genai>
# experiment
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w136/results
H=scripts/w136_unimate_harness.py
$PY -m pytest -q tests/test_experimental_unimate.py tests/test_experimental_elbow_prior.py tests/test_experimental_motionbricks.py
$PY -X utf8 $H preflight --out-dir $R/preflight          # downloads + verifies the checkpoint and T5, no-model roundtrip
$PY -X utf8 $H declare   --out-dir $R/declare --preflight $R/preflight/preflight.json
D="--declaration $R/declare/declaration.json --declaration-sha <sha of the new declaration>"
$PY -X utf8 $H controls  --out-dir $R/controls --preflight $R/preflight/preflight.json $D
$PY -X utf8 $H run       --out-dir $R/run --preflight $R/preflight/preflight.json --controls $R/controls/controls.json $D
# write $R/review/visual_review_S1.json, then:
$PY -X utf8 $H assess    --out-dir $R/assess --preflight $R/preflight/preflight.json --controls $R/controls/controls.json \
    --run-dir $R/run --visual-review $R/review/visual_review_S1.json $D
```

A re-run of `declare` gets a new timestamp, and therefore a new SHA; the official SHA is `47259fbf…`. The single
inference is deterministic only given the same seed, GPU, driver and torch build; this was not re-run.

## 13. Evidence and changed files

**Evidence**, under `C:/Users/dna10/animcv-w136/results`; local, not published. `evidence_manifest.json` lists 3,090
files and was written before this Worklog. SHA-256 `b25805a4f5b07bf71e85f671d9fde289e5f7a50513ab8701922be2f7b827a91f`.

| File | SHA-256 |
|---|---|
| `preflight/preflight.json` | `67ab4959…` |
| `preflight/export/S1_PROXY_B.glb` | `0ae006e3…` |
| `preflight/annotation_reviewed.json` | `2a29ba9b…` |
| `preflight/rig/cond.npy` / `rig/motions/S1_PROXY_B-Animation-000.npz` | `b674dea8…` / `8d62b42a…` |
| `declare/declaration.json` | `47259fbf…` |
| `controls/controls.json` | `44be7c0f…` |
| `run/model/raw_model_io.npz` / `keep_mask.npy` / `infer_record.json` | `14e80250…` / `95e75201…` / `bbc63ec3…` |
| `run/import_U.json` | `86a710ad…` |
| `run/run_result.json` | `348f5b88…` |
| `run/abcdeu_S1_30fps.mp4` | `e9d78b6b…` |
| `review/visual_review_S1.json` | `bea298b2…` |
| `assess/assessment.json` | `ed158590…` |

**Changed files (repository):**

- New:
  - `src/app/experimental_unimate.py`
  - `src/blender/experimental_unimate_export_stage.py`
  - `scripts/w136_unimate_infer.py`
  - `scripts/w136_unimate_harness.py`
  - `tests/test_experimental_unimate.py`
  - `examples/experimental_unimate/w136_declaration.json`
  - this Worklog
- Unchanged:
  - `docs/next_batch.md`, every historical Worklog and audit, the living status/architecture documents (no milestone claimed);
  - the W129/W135 modules, Blender stages and harnesses (imported/called, not edited);
  - CLI, GUI, exporter/importer, FramePose, MotionGraph, Production retargeting;
  - `scripts/batch_automation.py`;
  - the AnimCV venv;
  - every W126/W129/W135 file.

## 14. Resource and environment constraints

| Resource | Cost |
|---|---|
| Wall time | final preflight about 30 s (checkpoint already verified on disk; first download 21.8 s); controls a few seconds; run 58 s (inference subprocess 10.9 s, of which sampler 4.6 s; 4 builds; 6 readbacks/dumps; 6 × 480 renders; video and 8 sheets) |
| GPU | RTX 5080, peak 0.45 GB allocated by torch during inference |
| Disk | scratch 17.7 GB (Python env + uv cache, 1.19 GB checkpoint, 1 GB T5 snapshot, code clone, results) |

- The UniMate runtime is separate because the pinned Python 3.10 / bpy 4.0 stack differs from AnimCV's 3.11 venv.
- `torch 2.5.1+cu124` cannot run on `sm_120`, which is why the scratch env uses torch 2.7.1+cu128 (§3).

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 136
