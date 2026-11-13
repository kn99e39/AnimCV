# Worklog — Two-Stage MotionBricks → Bridge → UniMate (Scheduled Replacement) on the Authentic S1 Rig: Learned Information Path Shown, No Final-Quality Gain (2026-11-13)

Batch 139 was an experimental, research-only batch run on Main with this setup:

- local RTX 5080 (16 GB), Windows 11 Pro 10.0.26200;
- AnimCV venv `C:\Projects\AnimCV\.venv_build_full` (Python 3.11.9), unmodified;
- the W136 scratch UniMate runtime `C:/Users/dna10/animcv-w136/env/um310` (Python 3.10.22, torch 2.7.1+cu128), unmodified;
- Blender 5.1.0 for every build, dump, readback and render.

**Branch.** By explicit human decision (2026-11-13) the batch ran on the local branch `Dual_Stage_Generator`, in parallel with W138, which continues on `arch/single_frame_first`. That decision supersedes the spec's branch line ("work only on arch/single_frame_first") and its W138-COMPLETE gate. No W138 Worklog existed on `origin/arch/single_frame_first` at preflight; this is recorded in the preflight, and W138 evidence could not be incorporated. All commits are local; nothing was pushed, because pushing would create a remote branch.

**Numbering.** This file is numbered by Batch (139). `docs/138_*` belongs to the parallel W138 line and does not exist on this branch.

**Conditions.**
- `docs/next_batch.md` was not touched.
- No LabServer or other machine was used.
- Network use was limited to public, ungated Hugging Face metadata and licence pages at the pinned revisions. Nothing was downloaded.
- No LLM or other external API was called.

Every number, render and verdict below is **AGENT-RUN**. Visual judgements are the agent's own **nominally blinded still-frame** review. No human or animator reviewed anything, and native-rate playback was **not perceived**. Rights statements are a documentary reading, not legal advice.

## 1. Official judgment

**PARTIALLY SUPPORTED.**
- Source: `results/assess/assessment.json`, `91ff30ad82dbdde4f94d6a7c626729f6b6e01fbdecf2aa6371edd1be2cb3e637`.
- Sub-label: *learned information path shown; final quality superiority unproven/absent.*
- All 11 prerequisites pass, the K0 mechanism control passes, and I1, I2 and I3 all pass.
- The superiority rule is false against every control: S2A, S2B, D and plain slerp B.
- SUPPORTED was unattainable by declaration, because no human native-rate review was planned.

**What the evidence settles:**

1. **Stage 1 information reaches Stage 2 through a path that Stage 2 actually uses**, beyond passive carry-over.
   - The network's own clean-data estimate at the first non-discarded step keeps **97 %** of the MotionBricks-vs-slerp difference: γ = 0.971 (passive null 0.5, context-only regeneration ≈ 0).
   - The network's own contribution to the output points the same way as the Stage-1 difference (cos 0.977). Its size is 0.98 × the carry-through term, which is 262 mm when decoded to the wrist/elbow.
2. **The information path behaves as near-faithful reconstruction, not refinement.**
   - The output keeps β = 0.977 of the Stage-1 difference. Stage 2 moves each input by about 65 mm (mean wrist) and then reproduces it.
   - The learned two-stage result S2C keeps Stage 1's defects, and under blinding it was nearly indistinguishable from Stage 1 alone:
     - the arm swings sideways and drops to the hip mid-interior;
     - the wrist is below the shoulder on 64 % of interior frames.
3. **No final-animation benefit.**
   - None of S2A, S2B or S2C passes the declared acceptability rule. Plain slerp B (no model) is still the only acceptable branch.
   - Compared with what it was given, Stage 2 made the boundary continuity worse:
     - S2B exit step 73 mm vs B 25 mm;
     - S2C wrist peak 173 mm vs D 157 mm.
4. **Stage 2's native capability is weak even on its own coherent motion** (model-native control, INFORMATIVE):
   - completion error on a hidden left arm is 466 mm (W136-style) and 527 mm (official `--inbetween`, mean wrist);
   - refining a slerp start made it 18 mm worse.

**Scope.**
- These conclusions cover only frozen W129 MotionBricks → this bridge → UniMate v3 through the non-official scheduled-replacement sampler at **t_s = 0.5** (no sweep), on one clip (S1) with one seed.
- They do not apply to UniMate's official modes or to the general Two-Stage architecture (§13).

## 2. Implementation and changed files

**Commits** (local `Dual_Stage_Generator`, all before any pretrained Stage-2 output):

| Commit | Content |
|---|---|
| `a0785f4` | Instruments: `src/app/experimental_two_stage.py` (pure numpy rules and metrics), `src/app/experimental_two_stage_sampler.py` (scheduled sampler), `scripts/w139_two_stage_infer.py` (UniMate-side runner), `scripts/w139_two_stage_harness.py` (preflight/declare/controls/run/assess), `tests/test_experimental_two_stage.py` |
| `65ae489` | Declaration **v1** `39f21e28…`, **superseded** (see below) |
| `91b1153` | Fixes the declaration check for `core.autocrlf`, but writes real CR/LF characters into a bytes literal (syntax error, caught by the synthetic tests before any run) |
| `96fe811` | Repairs that literal and LF-normalises the harness |
| `fa75fde` | Declaration **v2** `1fe996680f45db82a1e591b0d3fcca4be93a866de1754e73094bbcb0704fae1c` (`examples/experimental_two_stage/w139_declaration.json`) |
| this commit | this Worklog |

**Why v1 was superseded.**
- The first official run stage refused declaration v1 **before any inference**. Its own committed-blob check compared CRLF working-tree bytes with the LF git blob.
- At that point no model output, run directory or outcome existed.
- The check was fixed and v2 declared. Rules, thresholds, runs and masks are unchanged, and v2 discloses the supersession.
- Git history was not rewritten; the broken intermediate commit is followed by a repair commit.

**Unchanged:**
- Production, CLI, GUI, exporter/importer, FramePose, MotionGraph and schemas;
- every W126–W137 module, runner, harness, Blender stage and Worklog (imported or called, never edited);
- `docs/next_batch.md`, the normative and living documents, and `scripts/batch_automation.py`;
- both Python environments and all model weights.

**Before the declaration** (all disclosed inside it):
- input-only exploration with no model;
- an adversarial design review (3 reviewers) and an adversarial code review (3 reviewers), which found one blocker: a float32/float64 normalisation mismatch that would have forced BLOCKED regardless of the model;
- weight-free `--dev-stub` dry runs of the whole pipeline under `C:/Users/dna10/animcv-w139/dev`. These computed metrics only for the frozen historical branches.

## 3. Frozen identities

**Stage 1** (no re-run):
- NVlabs MotionBricks `54fc563`, `mmm_v6_soma_fp32.onnx` `856419dc…`;
- W129 R1 `engine_output.npz` `8a528ef1…`;
- BaseRig left-arm projection = W129 `C_S1.blend` `e04efb35…` (condition **D**);
- weights under the NVIDIA Open Model Agreement (W135: PERMITTED; local, no redistribution). The version in force at the W129 download was not archived.

**Stage 2:**
- UniMate code `Friedrich-M/UniMate b78c780` (MIT); per-file SHA-256 equal to W137's;
- weights `Linzhan/UniMate 174fc755` `checkpoint_step_150000.pt` `f991ddd8…` (CC BY-NC 4.0), stats `0ff2b9ef…`, config `1ffbd379…`;
- text encoder `flan-t5-base 7bcac572` (Apache-2.0, offline);
- prompt "An object presses its left arm overhead.", seed 10, cfg 3.0, EMA, 50 Euler steps, as in W136/W137.

**Rights.** PERMITTED_NONCOMMERCIAL_RESEARCH_EVALUATION. The chained result inherits UniMate's non-commercial terms. Commercial use is not granted (a human-owned decision).

**Data:**
- S1 clip `de8feb9d…` (Action = SOURCE − 78, 30 fps);
- A `0803ab2b…`, B `8899f433…`, C/D `e04efb35…`, W126V-accepted `7e2ab165…`;
- W136 U and W137 T are bound through their evidence manifests (`b25805a4…`, `8561d239…`);
- all frozen identities were OK before and after the run, and fresh readbacks of A/B/C/Pin/U/T are world-bit-equal to W137's.

**Evidence:**
- root `C:/Users/dna10/animcv-w139/results`;
- `evidence_manifest.json` `75ef48bc76c118eb17cebb7a6ddc739b986a3f9b690ef92fcf75de9accffb400` (6,228 files, 997 MB);
- preflight `76ac3acb…`, controls v2 `436e4390…`, `run_result_final.json` `79703d00…`, visual review `e5488bf8…`, assessment `91ff30ad…`.

## 4. Architecture and information flow

```
 S1 video ─(W102/W105/W126, GT-free)→ A: original Animation Base (left arm = source-derived hypothesis; far arm 3D UNKNOWN)
                                  │
   Stage 1 (frozen W129)          │ context 168-171, target 200-203
   MotionBricks SOMA ─────────────┤→ C = D: left arm 172-199 (learned output + deterministic endpoint-correction ramp)
   slerp (no model) ──────────────┤→ B: left arm 172-199
                                  ▼
   Bridge (identical for A/B/C): Blender Action → W136 world-pose-preserving GLB export (static pelvis)
        → official rig_preprocess → official 12-D features → official normalisation → x1 (1,71,12,60)
                                  ▼
   Stage 2 (frozen UniMate v3): official sample.main --motion_edit + W136 joint×time keep mask
        + SCHEDULED REPLACEMENT on G (the 5,424 left-arm free entries): states 0..25 pinned to (1-t)eps + t·x1,
          free integration from t_s = 0.5 to 1 (flow SDEdit); body/torso/right arm kept at every step
                                  ▼
   decode upperarm_l / lowerarm_l (172-199) → W129 build_variant on a byte copy of A → scratch Action S2A / S2B / S2C
        (+ Xpin on W126V-accepted, Xrev restoring A) → fresh-process Blender readback
```

**Why this path.**
- No official UniMate mode lets a Stage-1 interior be refined:
  - `--inbetween` over 172–199, and the W136 mask, both **discard** it;
  - keeping the Stage-1 arm through `--motion_edit` **copies** it, leaving zero Stage-2 effect.
- The scheduled replacement is a **non-official sampler procedure**. It is a verbatim copy of the official `inbetween_sample_ode` with one change: the mask is chosen per produced state.
  - With k_s = 0 it is bit-identical to the official function (synthetic test). On the real model it reproduces W136 U bit-exactly (K0, §5).
  - The state at t_s equals the linear-path training marginal (flow form of SDEdit, Meng et al. 2022).
  - t_s = 0.5 is the mode of the config's logit-normal(0,1) training-time distribution. It was fixed a priori.
- The fully official sparse-keyframe `--inbetween` path was rejected before any output: it copies Stage 1 at kept keyframes and regenerates the whole body in between (W137's failure mode).

## 5. Representation, bridge and model-visible conditioning

**Bridge (no model, preflight).** For A, B and C:

| Check | Result |
|---|---|
| Export world-matrix change | ≤ 1e-5 |
| Annotation | identical labels, 0 flagged |
| Time identity, shift 0 | ≈ 2.6e-7 |
| Time identity, shift ±1 | 0.22–0.36 |
| Own vs official decode | ≤ 1e-9 |
| Blender roundtrip of decoded keys | ≤ 4.8e-7 m vs each condition's readback |
| cond numeric fields | identical across the three assets |
| Rebuilt PROXY_B features | **bit-equal** to W136's |
| Per-condition world maps | RMS 2.6 µm; spread s 1e-7, t 9e-8 m |
| Grounding joint | foot_r in all three |

**Information content.** The normalised x1 tensors of A, B and C are bit-equal on every kept entry and differ only inside G, on about 4,630 of 5,424 entries.

**Pre-output power check.** ‖t_s Δx1‖ / ‖(1 − t_s) eps‖ on G is 0.97 for C vs B and 1.08 for A vs B.

**Stage-1 decomposition (no model).** MotionBricks' endpoint correction, `q_corr = q_raw · slerp(1, ΔS, 1 − w) · slerp(1, ΔE, w)`, was inverted from the frozen deltas:
- start/end deltas: LeftArm 33.7° / 38.4°, LeftForeArm 31.4° / 53.5°;
- endpoint identities and mapping hold to 1e-5°;
- the correction ramp dominates Action 172–176 and 189–199 (upperarm_l) and 172–179 and 186–199 (lowerarm_l);
- only **180–185** are learned-dominated for both channels.

**Model-visible conditioning (I1).** Measured only on evaluations whose output is used:
- the network input at Euler step 25 equals produced state 25 in both runs;
- the cross-run state difference equals t_s·(x1_C − x1_B) on G within 4.8e-7, and is 0 elsewhere;
- the network's input token embeddings at step 25, both CFG branches:
  - they differ on every G (joint, frame) cell, max 2.53;
  - they are **exactly equal** on all other cells and on the T-pose prefix tokens.

Steps 0–24 were computed and discarded (velocity difference up to 7.6), and they are not used as evidence.

**Mechanism controls:**
- **K0** (k_s = 0): PROXY_B and PROXY_C give **bit-identical** outputs (0.0). Stage-1 content has no other path in.
- **K0B** reproduces the frozen **W136 U** raw output **bit-exactly**: 0.0 normalised, 0.0 mm decoded, with an identical x1. This independently validates the copied sampler, the mask hook and the bridge on the real model.
- **Replicate** S2B / S2B_rep: bit-identical, so the noise floor is 0.
- **Seeds:** every seed-10 run had the same eps SHA. The native reference used seed 11 (|corr| 0.002).
- **Sampler calls** per run: generate_samples 1, scheduled sampler 1, network forwards 100.

## 6. Information transfer (C vs B, feature space on G)

| Quantity | Passive null | Measured | Meaning |
|---|---|---|---|
| γ at first free step (25) | 0.5 | **0.971** | the network's clean estimate keeps 97 % of the Stage-1 difference |
| γ at steps 30 / 40 / 49 | 0.5 | 0.975 / 0.976 / 0.977 | stable through integration |
| ‖network term‖ / ‖carry-through‖ | 0 | **0.976** | the network's own contribution is as large as the carry-over |
| cos(network term, Δx1) | — | **0.977** | AMPLIFIES_STAGE1 |
| β = output share of Δx1 | 0.5 | **0.977** | the output reproduces about 98 % of the input difference |
| decoded network term, max wrist/elbow | 0 | **262 mm** | ≥ 10 mm and ≥ 10 × floor (floor 0) |
| decoded S2C − S2B, max | — | 561 mm | |

Decomposition sanity checks: carry-through 4.8e-7; Σ dt·Δv vs N 3.5e-6.

**I1, I2 and I3 all pass, so learned_path = true.**

**Correction attribution (decoded):**

| Frames | Mean S2C − S2B | Mean network term |
|---|---|---|
| correction-dominated | 221 mm | 113 mm |
| learned-dominated (180–185) | 490 mm | 229 mm |

The largest transfer happens where MotionBricks' learned content dominates, so Stage 1's learned information does reach the output. Near the boundaries, much of what is transferred is the deterministic correction ramp.

**A vs B (single-stage reference information):** β 0.974, γ 0.957, ratio 0.976, cos 0.971, network term 192 mm. Stage 2 treats the original reference the same way.

## 7. Model-native capability control

**Status INFORMATIVE.** Admissibility checks on the reference:
- NREF (seed 11, unconstrained) FK re-encode: RIC = FK to 1e-15;
- no fold (neck-over-pelvis ratio ≥ 0.99);
- hidden-interval wrist path 1,201 mm;
- the NaN-poisoned start build is identical, so no hidden value is read.

The hidden set is the W136 free set G of N_ref*: left-arm descendants, window 16–43, plus velocity at 15.

| Run | Mean / max wrist error (mm) | Mean / max elbow error (mm) | Max angle error, upper / lower (°) | Wrist entry / peak step (mm), ref 69 / 113 |
|---|---|---|---|---|
| N0: completion from noise, k_s = 0 (W136 mask) | 466 / 777 | 227 / 386 | 92 / 85 | 116 / 397 (1 flick) |
| NB init: slerp start, no model | 257 / 403 | 162 / 261 | 67 / 34 | 40 / 41 |
| NL: Stage 2 from the slerp start, k_s = 25 | 274 / 425 | 178 / 280 | 67 / 33 | 51 / 51 |
| NIB: official `--inbetween`¹ | 527 / 777 | 186 / 338 | 82 / 77 | **339** / 515 |

¹ Kept frame 15 includes velocity channels that encode the hidden frame 16 (disclosed leak), so NIB is not a leak-free completion.

- Even with its own coherent motion as context, Stage 2 does not recover a hidden arm.
- Refining the deterministic start made it **18 mm worse** (refinement gain −17.6 mm).
- Official in-betweening reproduces W137's entry jump.

This is a Stage-2 capability limitation under native conditions, recorded separately. It is a self-generated reference, not real motion data.

## 8. Real S1 Blender output (A / B / D / S2A / S2B / S2C; U and T are NON-EQUIVALENT references)

**Mechanical gates.** All pass for S2A, S2B and S2C:
- exactly 224 changed keys, only `upperarm_l`/`lowerarm_l` at Action 172–199;
- one Action with 32 CONSTANT F-curves at 30 fps;
- Action 171/200 and every non-arm world point are bit-equal to A on 240 frames and 717 subframes;
- limb-length range ≤ 0.4 µm;
- the W126V pin residual equals W126V's (6.4e-7 m ≤ 1e-6 m), with pin frames bit-equal;
- Xrev is world-bit-equal to A;
- the bridge roundtrip holds;
- hybrid FK matches the Blender readback within 5 µm.

**Raw vs projected.** Body context is kept, so root drift is 0 and the raw full-body rule is usable for all three. The generated arm's own positions and rotations disagree by 116–142 mm (RIC vs FK, interior), the model-internal inconsistency already seen in W137.

**Arm-local metrics** (clavicle_l frame; steps 171→172 … 199→200):

| | A | B | D | S2A | S2B | **S2C** | U | T |
|---|---|---|---|---|---|---|---|---|
| Wrist entry / exit / peak (mm) | 93 / 3 / 203 | 24 / 25 / 25 | 90 / 10 / 157 | 127 / 44 / 228 | 56 / 73 / 73 | **107 / 70 / 173** | 146 / 177 / 481 | 300 / 159 / 300 |
| Elbow entry / exit / peak (mm) | 64 / 8 / 215 | 18 / 18 / 18 | 63 / 3 / 84 | 87 / 28 / 212 | 41 / 33 / 41 | **80 / 18 / 113** | 150 / 122 / 232 | 234 / 112 / 234 |
| Flicks, wrist / elbow | 6 / 6 | 0 / 0 | 0 / 0 | 6 / 4 | 0 / 0 | **1 / 0** | 4 / 3 | 0 / 0 |
| P1: wrist above shoulder, fraction of frames | 1.00 | 1.00 | 0.36 | 1.00 | 1.00 | **0.36** | 0.36 | 0.39 |
| P2: min interior wrist z − z(171) ≥ −100 mm | pass | pass | fail | fail | pass | **fail** | fail | fail |
| Mean wrist deviation from A (mm) | 0 | 148 | 408 | 65 | 181 | **420** | 539 | 476 |
| Acceptable (floor ∧ intent ∧ non-inferior to B) | no | **yes** | no | no | no | **no** | no | no |

How much Stage 2 changes its own input (mean / max over 172–199):

| | Wrist (mm) | Elbow (mm) |
|---|---|---|
| S2A − A | 65 / 109 | 44 / 74 |
| S2B − B | 69 / 103 | 43 / 62 |
| S2C − D | 64 / 104 | 44 / 90 |

**Other measurements:**
- **Superiority** (S2C over S2A, S2B, D, B): false for all four, because S2C is not acceptable.
- **Source 2D:** the torso camera is INVALID (23.3 px > 20), so source fidelity is UNKNOWN. "Deviation from A" is agreement with a GT-free hypothesis, not fidelity.
- **Contact:** not supportable.
- **Inherited defects:** shoulder/torso popping (up to 244 mm per frame in W137) and the rest-pose right arm are bit-identical in every candidate; no candidate is authorised to repair them.

## 9. Qualitative review (agent, still frames, nominally blinded)

**Protocol.**
- Letters were assigned by an `os.urandom` key sealed in `run/SEALED_blind_key/`.
- The review JSON (`e5488bf8…`) was written from the four blind arm sheets before the key was opened.
- The key turned out to be P = S2A, Q = D, R = S2C, S = S2B.

**Ranking (natural and press-faithful):** S2B > S2A > S2C ≈ D.

| Branch | Observation |
|---|---|
| S2B | Stays near the rack, then rises progressively to overhead and joins 200 smoothly. |
| S2A | Wide lateral abduction (elbow at or above the shoulder, to the side) before going overhead. |
| S2C, D | Abrupt sideways swing at entry; the arm drops beside the hip through 180–188; late lateral rise. **Nearly identical**: Stage 2 did not correct Stage 1's detour. |

The review is reported only and never decisive. **Perceptual acceptance: PENDING.**

## 10. Positive mechanisms, negative mechanisms, uncertainties

**Positive:**
- A legitimate, model-visible, non-discarded Stage-1 → Stage-2 path exists and is shown token by token.
- The network reads Stage-1 content (γ 0.97) instead of ignoring it.
- The bridge is lossless.
- Edit authority, the hard pin and reversibility hold.
- The copied sampler reproduces the frozen W136 run bit-exactly.

**Negative:**
- At t_s = 0.5 Stage 2 reconstructs its input almost faithfully, defects included:
  - S2A keeps A's flicks;
  - S2C keeps MotionBricks' detour and correction ramps;
  - S2B's slerp gains boundary discontinuity.
- The learned Stage-1 information is itself unfaithful to the press: D fails intent.
- Stage 2 adds no plausibility gain and loses continuity against its inputs.
- The native control shows Stage 2 does not complete or refine a hidden limb well even in-distribution.
- Plain slerp B stays the bar.

**Uncertainties:**
- Whether another, still-declared t_s would let Stage 2 overwrite Stage-1 errors. Not tested: no sweep.
- Whether a human at native rate would rank S2B over B or S2C differently from D. Not reviewed.
- Domain gap: own rig, a rest-pose right arm and source-derived torso popping as hard context.
- One clip and one seed.
- Training-data provenance of the UniMate weights is flagged unclear by the licensor.

**Causal attribution** (no single-mechanism claim):

| Factor | Effect |
|---|---|
| Stage 1 learned content | sets S2C's shape (largest transfer at the learned-dominated frames 180–185) |
| Deterministic correction ramp | carried near the boundaries |
| Bridge | no change |
| Stage 2 | about 65 mm reshaping plus worse boundary steps, through the non-official sampler at t_s = 0.5 |
| Source perception | A's flicks |
| Frozen parent/root | no drift; arm-local identical raw vs projected |
| Distribution mismatch | consistent with the native-control failure |

## 11. Runtime, storage, cost

| Item | Value |
|---|---|
| Model processes | 10 (6 primary + 4 native), no retries |
| Per process | sampler ≈ 5.3 s, total ≈ 6.9 s |
| Peak VRAM | 0.45 GB |
| Model wall time | 1.2 min |
| Whole run stage | including 9 Blender builds, 9 readbacks and 3 dumps, renders of 8 + 4 branches × 240 frames × 2 views, and videos (a few minutes) |
| Disk | results 968 MB, dev dry runs 1.6 GB |

Everything is within the declared budget (≤ 12 processes, ≤ 2 GB VRAM, ≤ 60 min, ≤ 5 GB). No training, finetuning, download or re-run of MotionBricks took place.

## 12. Reproduction and test status

Git Bash, repository at `fa75fde` (instruments + declaration v2):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w139/results; H=scripts/w139_two_stage_harness.py
$PY -m pytest -q tests/test_experimental_two_stage.py tests/test_experimental_unimate_inbetween.py tests/test_experimental_unimate.py tests/test_experimental_elbow_prior.py tests/test_experimental_motionbricks.py
$PY -X utf8 $H preflight --out-dir $R/preflight
$PY -X utf8 $H declare   --out-dir $R/declare_v2 --preflight $R/preflight/preflight.json     # then commit the copy under examples/
D="--declaration $R/declare_v2/declaration.json --declaration-sha <sha>"
$PY -X utf8 $H controls  --out-dir $R/controls_v2 --preflight $R/preflight/preflight.json $D
$PY -X utf8 $H run       --out-dir $R/run --preflight $R/preflight/preflight.json $D --controls $R/controls_v2/controls.json
# write $R/review/visual_review_S1.json from run/blind/* BEFORE opening run/SEALED_blind_key/, then:
$PY -X utf8 $H assess    --out-dir $R/assess --preflight $R/preflight/preflight.json $D --controls $R/controls_v2/controls.json --run-dir $R/run --visual-review $R/review/visual_review_S1.json
```

The inference is bitwise deterministic on this GPU, driver and torch build (replicate run and K0 vs W136). A re-run of `declare` gets a new timestamp and therefore a new SHA.

| Item | Kind | Status |
|---|---|---|
| `tests/test_experimental_two_stage.py` (30), covering the items below | synthetic + real-data no-model | **30 PASS** |
| Reused `test_experimental_unimate_inbetween` / `_unimate` / `_elbow_prior` / `_motionbricks` (85) | synthetic | **85 PASS** |
| Controls v2: W136's 31 re-executed + 20 W139 | real data, no model | **51/51 fail closed** |
| Preflight, bridge, 10 real inferences, builds, readbacks, renders, videos | real | **RUN** |
| Prerequisites (11), K0, I1–I3, mechanical gates | real gates | **PASS** |
| Superiority, acceptability of S2A/S2B/S2C | declared quality criteria | **FAIL** |
| Human / animator / native-rate review; full repository suite (Production untouched); t_s / seed / prompt / clip variation (forbidden) | — | **NOT RUN** |

The 30 W139 synthetic tests cover:
- k_s = 0 bit-identical to the official sampler (frame and joint×time masks);
- full-schedule copy; training-marginal state; discarded steps have no effect; out-of-schedule x1 never read; padding;
- invalid k_s; seed and eps identity;
- passive, denoiser and regenerator decomposition nulls;
- intent, acceptability and judgment mapping;
- endpoint-correction inverse, slerp and poison checks, blind key;
- time map and import-domain refusals;
- the offline normaliser reproducing W136's official x1 bit-exactly.

Spec testing items covered **only** by real-run gates or the re-executed W136 controls, not by W139 synthetic tests: rights/identity gates, Blender Action preservation, the hard pin, reversibility and fresh-process readback.

## 13. Scope of demonstrated capability

**Demonstrated:**
- A frozen Stage-1 generator's completion can be bridged losslessly into a frozen Stage-2 generator's own-rig representation.
- With a scheduled-replacement (flow-SDEdit) sampler, the Stage-1 content becomes model-visible, is read by the network, and is propagated into the final editable Blender Action under full edit-authority, pin and reversibility contracts.

**Not demonstrated:**
- that Stage 2 improves on Stage 1 or on a deterministic completion;
- that the learned two-stage result is more natural or more source-faithful than any matched control;
- that Stage 2 can complete or refine a hidden limb under its own native conditions.

**Not tested:**
- other t_s values; UniMate's official modes as a Stage-2 path for Stage-1 interiors (structurally excluded, §4);
- a different Stage 1 or Stage 2 generator, full-body Stage-1 channels, other clips or seeds;
- human perception.

The negative final-quality result applies to this configuration. It does not make the general Two-Stage architecture impossible.

## 14. Official hypothesis judgment and its limitations

**PARTIALLY SUPPORTED.**

**What counts in favour:**
- Introducing the learned MotionBricks stage before frozen UniMate provides information that is absent from the slerp baseline.
- Stage 2 demonstrably consumes that information through a learned path: γ 0.97 vs passive 0.5, an aligned network term of 262 mm, token-level visibility, K0 isolation, and bit-exact reproducibility.

**What counts against:**
- In this configuration Stage 2 consumes Stage 1 information by reproducing it, its errors included (β 0.98).
- The final editable Blender animation is not better than the matched single-stage (S2A), deterministic-completion (S2B) or Stage-1-direct (D) alternatives, and it does not reach plain slerp B.

**Limitations:**
- This judgment is distinct from the test PASS results.
- It rests on agent-run metrics and an agent still-frame review. Human native-rate perceptual acceptance is **PENDING**.
- It is scoped to t_s = 0.5, one clip, one seed and the non-official sampler.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 139
