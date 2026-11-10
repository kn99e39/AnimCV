# Worklog — CondMDI Imputation vs Learned-Conditioning Causal Gate on the Real S1 Animation (2026-11-10)

Batch 132. A bounded diagnostic batch on `arch/single_frame_first`. It ran unattended on Main:

- local RTX 5080 (16 GB), Windows 11 Pro 10.0.26200;
- AnimCV Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9), unmodified;
- the existing W131 scratch CondMDI runtime, reused unmodified: CPython 3.12.16, torch 2.9.1+cu128, CUDA 12.8,
  cuDNN 91002, numpy 2.2.6, safetensors 0.8.0, einops 0.8.2, mmengine 0.10.7, OpenAI CLIP at `d05afc4`;
- Blender 5.1.0 (used read-only, to dump the frozen baseline).

`docs/next_batch.md` was read only. No LabServer and no other machine. Network use was limited to:

- `git fetch/push`;
- the Hugging Face model API record and `README.md` of the pinned checkpoint revision (rights recheck, metadata only).

No weights, datasets or new models were downloaded. No training or tuning.

Every number below is **AGENT-RUN** on Main. No human or animator reviewed anything. Native-rate (30 fps) perceptual review
was **NOT RUN**.

## 1. Official CASE and scope

**Verdict: CASE C under the frozen rule.** Source: `results/assess/assessment.json`, SHA-256
`35919277d4c1a166d1ff8b844cf970b226de35d92615765be7ad239e91eb8d93`.

**Why C.** The real model ran on the exact W131 inputs, and every integrity guarantee held:

- the W131 result replays bit for bit;
- turning imputation OFF provably changes only the diffusion-step replacement;
- the masked-input leak control is bitwise equal;
- every negative control fails closed.

But the imputation-OFF output (R1) **does not preserve the observed source context**. Its left wrist lies 255 mm away from
the observed wrist at the boundary-window samples; the declared limit was 50 mm. The declared consequence is that R1 cannot be
accepted as an edit, so no C_OFF Action was projected or built. That is CASE C: real inference, but OFF cannot preserve the
observed context.

**Mechanism class (declared rule): `OFF_CONTINUOUS_ONLY_BY_DEPARTING_FROM_CONTEXT`.**

- R1's own internal wrist steps at the boundaries are 82 / 42 mm.
- R0's steps are 299 / 166 mm.
- R1 is continuous only because it does not follow the observed frames.

**Central finding.** It is exact, not statistical:

- With imputation ON and OFF, **all 34,194 generated (masked) output features are bitwise identical** on S1.
- Only the 7,886 observed features differ.
- The same holds on both reference-control pairs.

So diffusion-step replacement imputation does **not** cause the W131 boundary jump. The generated interior comes entirely
from the learned conditioning path. Replacement only decides what the observed frames show in the output: the source values
(ON) or the model's own reconstruction of them (OFF). §6 gives the code-level reason.

**Scope.**

- One opt-in scratch experiment, on one source (S1), with one seed.
- No Blender Action, `.blend`, video or render was produced, because candidate construction was stopped by the gate.
- No Production, CLI, GUI, exporter, FramePose, MotionGraph, FK/IK, schema, dependency or training change.
- W131 code, outputs and **CASE B are unchanged and frozen**.
- Historical CASEs are untouched: W121 D, W122 B, W123 B, W124 B, W125 A, W126 C, W127 B, W128 DATA BLOCKED, W129 B,
  W130 DATA BLOCKED_LICENSE, W131 B.
- No historical Worklog was edited.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `5abe31a8cd1e272478ca56aa2290f102fe26cc45`, verified first |
| Starting commit | `2358a73adac6fa72bbbb03677e7e2769244ff9d5` = `origin/arch/single_frame_first` at the first fetch. W131 carries the COMPLETE marker, and `docs/AUDIT_W126_W130_2026_11_09.md` is present. The remote did not advance during the batch. |
| Instrument commit | `0bd3677`: runner, harness, module, tests (pre-outcome) |
| Declaration commit | `7017aa0` (`examples/experimental_condmdi/w132_declaration.json`, byte copy of `results/declare/declaration.json`, SHA-256 `841822b0eaf3e718614f7c2a0d6f7e9804490e21eed92b17b297546b6fe76273`). It records `outcomes_inspected: false` and `new_model_output_exists: false`, and binds the LF SHA-256 of 12 instrument files (4 new + the 8 W131 instruments). Pushed before any S1 or reference-control sampling. |
| Final source revision | `7017aa0`. The Worklog commit is documentation only. The COMPLETE marker is a separate, final commit. |
| Scratch root | `C:/Users/dna10/animcv-w132`, outside the repository. W131 scratch was read only. |

**Pre-declaration instrument self-test** (disclosed; not evidence).

- Before the instrument commit, the W132 runner was exercised once ON and once OFF on a **synthetic** 40-sample input in
  `animcv-w132/selftest` (zero motion, root height and 10 frames of positions observed, text "a person stands still", seed 1).
- The purpose was to check that the hooks and the isolation rule work on the real runtime. It used no S1 data and no
  reference-control data.
- Result: isolation PASS, and the first divergent network call was 2.
- No instrument was changed after it, and none after the declaration.

## 3. Rights, identity and frozen inputs (preflight)

Source: `results/preflight/preflight.json`, `508a788b…`, status **OK**.

- **Checkpoint.** `ZeyuLing/motius-condmdi-humanml3d` @ `1e84d9eb…`.
  - Local LFS SHA-256 recomputed and equal for `model.safetensors` (`3d9e299f…`), `clip.safetensors`, `Mean_abs_3d.npy` and
    `Std_abs_3d.npy`. The runner re-checks them on every inference.
  - The W131 tensor identity (official random-joints EMA, 196/196 bitwise) was not repeated; there was no new download.
- **Terms rechecked at the pinned revision.**
  - API `sha` = pinned revision; `cardData.license: mit`; front-matter `license: mit`; no HY-Motion, Hunyuan or Tencent
    mention.
  - The card `README.md` is byte-identical to the copy W131 fetched.
  - The CondMDI `LICENSE` and the Motius `licenses/condmdi/*` files are byte-equal to W131's.
- **Code pins.** Original CondMDI `0c58473066ef582b9464a66d5542e0d1b6ae66c8` and Motius
  `d04a38c9bc9dab3f0e9e2d843f8a6b21eaa13abc`. Both checkouts are at the pin with clean worktrees.
  - This also closes W131's open item: the runner now records the Motius HEAD as a SHA rather than a symbolic ref.
- **Rights reading** (unchanged from W131).
  - Card MIT, code MIT, CLIP MIT.
  - Training data HumanML3D ← AMASS, which is non-commercial research.
  - The declared use is local non-commercial research evaluation.
  - The code MIT licence is **not** read as a commercial deployment grant for the checkpoint.
  - This is the agent's documentary reading, not legal advice.
- **Frozen W131 evidence**, all byte-equal to the W131 manifest:
  - `conditioning.npz` `b6e46ccf…`, `conditioning_leak.npz` `b07a8d3b…`;
  - `model_R1/model_io.npz` `2ec93555…`, `projection.json` `133a2ebf…`;
  - `C_S1.blend` `8f49a93a…`, `run_result.json` `f5fbdb8f…`, `assessment.json` `9633a362…`;
  - the declaration `dca4a891…`, the smoke `smoke_t2m.npz` `c60b4b35…`, and the evidence manifest `98505f80…`.
  - All 8 W131 instrument files are LF-hash-equal to the W131 declaration.
- **Other frozen inputs.**
  - S1 baseline `0803ab2b…`, W126V-accepted `7e2ab165…` and W129 B `8899f433…`.
  - The W126 lineage head and value digest are unchanged.

## 4. The S1 causal pair (R0 ON / R1 OFF)

Fixed W131 conditions, all unchanged:

| Setting | Value |
|---|---|
| Seed | 42 |
| Text | "a person stands in place and presses both arms straight up overhead" |
| Guidance | CFG 2.5 |
| Sampler | official `CondMDIPipeline._sample` → `p_sample_loop` → `p_sample` → `p_mean_variance` (100 ancestral steps on the `ddim100`-respaced schedule) |
| Length | 160 samples at 20 fps |
| Precision / CUDA | fp32; deterministic CUDA, no TF32 |
| Conditioning | the frozen W131 `conditioning.npz` bytes |
| Edit interior | samples 114–132 = Action 172–199 = SOURCE 250–277 |
| Mask | 7,886 observed features; the left arm is never observed inside the interior |

**Preprocessing identity.** The S1 conditioning was re-derived from a fresh Blender dump of the frozen baseline. The relative
HML263 tensor, the mask and the frame-0 facing r0 are all **bitwise equal** to the frozen W131 file.

**Runs.** Source: `results/causal/causal_result.json`, `1d37d6f7…`.

| Run | Runner | impute | Input | Result |
|---|---|---|---|---|
| R0_w131code | **unchanged** W131 runner | on | frozen W131 conditioning | `model_io.npz` **byte-identical** to the frozen W131 `model_R1` (`2ec93555…`): an exact cross-session replay |
| R0 | W132 instrumented runner | on | same | `model_io.npz` byte-identical to W131 R1, so the instrumentation is non-invasive |
| R1 | W132 instrumented runner | **off** | same | `cdd9fcd8…` |
| R1_leak | W132 instrumented runner | off | frozen W131 leak input (UNKNOWN rig channels rotated, seed 131) + every masked input feature replaced by seeded noise (seed 131) | input differs; effective conditioning and output **bitwise equal** to R1. This doubles as R1's determinism rerun. |

A negative control ran first: the runner, given a wrong input SHA-256, exits 3 with `stop.json` and loads no model.

**Flag isolation**, taken from runtime instrumentation, not assumed: **PASS**, every declared check true.

- **Identical inputs.** The input, absolute input, mask and effective-conditioning arrays are bitwise equal, as are the
  seed, text, length, guidance and respacing.
- **Same learned-conditioning inputs on every call.** There are 200 network calls in each run (100 steps × CFG
  conditional/unconditional). A forward pre-hook saw exactly **one** `obs_x0` hash and **one** `obs_mask` hash, identical
  in ON and OFF, so the learned conditioning path receives the same observation in both runs. The text tokens are identical.
- **Imputation keys are the only difference.** The diffusion `y` option keys differ exactly by `{imputate,
  stop_imputation_at, replacement_distribution, inpainted_motion, inpainting_mask}`.
- **Replacement on vs off.** `requires_imputation` returned True on all 100 steps in ON and False on all 100 in OFF.
  Reconstruction guidance never ran.
- **Divergence point.** The first-step network inputs and outputs are identical. The first divergent network call is
  call 2, the first call after the first replacement.

**Costs per run.**

| Item | Cost |
|---|---|
| Sampling | about 3.9–4.1 s (W131-code replay 3.4 s) |
| Model load | 2.8–2.9 s |
| Process wall time | about 9.5 s |
| Peak VRAM | 1.84 GB allocated, 1.90 GB reserved |
| RSS | 1.7 → 2.7 GB |
| Whole causal stage (4 inferences, 1 negative, 1 Blender dump) | 45 s |

## 5. Model-space evidence (before any projection)

Primary space: `ric`, the root-relative heading-frame position that the mask actually conditions. World-canonical numbers
are dominated by the **unobserved** root xz travel (0.69 m generated) and are not used for the context judgement; for
example, world wrist context errors are about 1 m in both runs.

**Left wrist, `ric`, mm:**

| | R0 (ON) = W131 R1 | R1 (OFF) |
|---|---|---|
| Internal step 113→114 / 132→133 | **299 / 166** | 82 / 42 |
| Effective step against the observed 113 / 133 | 299 / 166 | **299 / 166** (identical) |
| Context error at observed sample 113 / 133 | 0 / 0 | **255 / 127** |
| Source's own step 112→113 / 133→134 | 170 / 4 | 170 / 4 |

**Left elbow, `ric`, mm:**

| | R0 (ON) | R1 (OFF) |
|---|---|---|
| Internal step 113→114 / 132→133 | 236 / 166 | 79 / 32 |
| Effective step against the observed 113 / 133 | 236 / 166 | 236 / 166 |
| Context error at 113 / 133 | 0 / 0 | 173 / 135 |

**Observed-context residuals of R1** (frozen gate in brackets):

| Residual | R1 value | Gate |
|---|---|---|
| Left elbow/wrist error in the windows 103–113 and 133–143 | **255 mm** max | [≤ 50 mm] **FAIL** |
| Mean error over all 2,522 observed joint-samples | **64.7 mm** | [≤ 50 mm] **FAIL** |
| Root height error | **72.6 mm** max | [≤ 50 mm] **FAIL** |
| Root yaw error | **0.174 rad (9.95°)** max | [≤ 5°] **FAIL** |
| Any observed joint | 466 mm max | — |

- Per-joint mean errors range from 28 mm (spine3) to 114 mm (right shoulder, a PROXY).
- R0's observed residuals are ≤ 1.5e-8 by construction.

**Raw trajectory and timing.** These are identical in R0 and R1, because the interior is bitwise equal:

- the generated left wrist is already at 1.85 m at sample 114;
- it peaks at sample 125 (2.15 m at 123);
- it is 2.11 m at 132;
- the source hypothesis peaks at sample 127.

**Root.** The generated root travels 0.69 m net (1.24 m path) in both runs. The root yaw range is 0.25 rad in R0 and
0.36 rad in R1; yaw is observed, so the OFF reconstruction drifts.

**R1 vs R0 overall:** the maximum feature difference is 0.445, and every difference lies on observed features.

## 6. Why the generated interior is identical ON vs OFF (code-level explanation of the measured identity)

The identity was measured first. This explanation was then read from the pinned source; it is not an additional experiment.

- **The network pastes the observation over its own input.** Motius `models/condmdi/network/unet.py:776`: on every call,
  the keyframe-conditioned network replaces its own input at observed entries, `x = obs_x0 * obs_mask + x * (~obs_mask)`,
  then concatenates `obs_mask`. Observed entries of the diffusion state x_t therefore never reach the network.
- **The update is elementwise.** `p_mean_variance` with `replacement_distribution: conditional` overwrites the predicted x0
  at observed entries only. The ancestral update to x_{t-1} is elementwise per feature. The noise stream is the same
  (`fork_rng`, seed 42, and an identical number of draws).
- **Consequence.** Replacement can change only the observed entries of the trajectory. That is exactly what was measured
  on S1 and on both reference pairs.

**What this establishes for this checkpoint and runtime.**

- Turning imputation off cannot repair the W131 interior.
- The 299 mm entry jump is a mismatch between:
  - the interior that the **learned conditioning** produces from this observation (the arm already overhead at 114); and
  - the observed context itself.
- With ON, the output shows the observed context and the jump is visible. With OFF, the output shows the model's own
  reconstruction of the context, which is smooth but up to 255 mm off the source.

The flag-isolation result confirms that this is a property of the learned conditioning path, not of an altered conditioning
input.

## 7. Reference mask control (four cells)

**Evidence tier: MODEL-GENERATED CLOSED-LOOP SYNTHETIC.**

- This is neither held-out data nor proof of on-distribution generalisation, and it is not real-video quality evidence.
- **Why no authentic reference.** A HumanML3D "official tiny sample" (`004822`, hml263 199×263) exists in the Motius test
  assets. It was **not used**: it is AMASS-derived motion redistributed as a test fixture without a data-licence grant, so
  lawful availability was not established. CondMDI's `dataset/000021.npy` has the same status and is raw 52-joint
  positions.
- **The reference** is the frozen W131 smoke output (`c60b4b35…`): CondMDI text-to-motion with "a person walks forward and
  waves with the right hand", 120 samples.
  - It was conditioned through `relative_to_absolute` of its frozen relative tensor. That differs from the stored absolute
    by up to 0.116, which is the pipeline's round trip; scores use the conditioned tensor.
- **Fixed before sampling:**
  - seed **7**: seed 42 generated the reference itself and would be a degenerate loop, and 7 is the existing W131 smoke
    mask-contract seed;
  - the hidden interval: official `build_observation_mask(mode="middle", transition_length=19)` on all 22 joints, giving
    hidden samples **50–68** (19);
  - CFG 2.5, ddim100.

**Mask parity.** The runner's official masks equal the numpy replica bitwise in all four cells.

- pos ⊂ pos_rot_vel, and they differ only by `root_yaw`, `rot` and `vel`.
- **Root semantics, official and explicit:**
  - root xz + height are observed in all cells;
  - root yaw only in the pos_rot_vel cells;
  - the 4 foot-contact features are observed in all cells, because the official mapping always adds them;
  - all values are the reference's own.
- The pairs share masks. Nothing is observed in the hidden interval.
- Negative control: a wrong reference SHA exits 3, and no model is loaded.

**Results.** Source: `results/refctl/refctl_result.json`, `8c310fd0…`. The reference's own largest joint step at the
boundaries is 47 / 50 mm (`ric`).

| Cell | Features | impute | Observed feats | Hidden MPJPE ric / world (mm) | Endpoint MPJPE first / last hidden, ric (mm) | Boundary excess ric, entry / exit (mm) | Observed-frame MPJPE ric / world (mm) | Max observed residual |
|---|---|---|---|---|---|---|---|---|
| i | pos_rot_vel | ON | 26,563 | 16.4 / 58.2 | 4.9 / 5.6 | 7.7 / 7.6 | 0 / 0 | 1.2e-7 |
| ii | pos | ON | 7,070 | **60.6 / 311.1** | 44.7 / 37.9 | **133.7 / 89.1** | 0 / 143.7 (yaw unobserved) | 1.2e-7 |
| iii | pos_rot_vel | OFF | 26,563 | 16.4 / 58.2 | 4.9 / 5.6 | 1.2 / 3.9 | 2.6 / 29.8 | 0.085 (root xz) |
| iv | pos | OFF | 7,070 | 60.6 / 311.1 | 44.7 / 37.9 | 4.4 / 7.0 | **26.7 / 336.8** | **0.600** (root xz; contact 0.36, pos 0.11) |

- **Hidden features are bitwise identical** for i vs iii and for ii vs iv. Replacement again changes only the observed
  entries.
- **Declared readings** (discontinuity = ric excess > 50 mm):
  - discontinuity is present only in **cell ii**;
  - replacement effect: none for pos_rot_vel; "ON>OFF" for pos;
  - feature-group effect: "pos > pos_rot_vel" with imputation ON; no class change with OFF.
- **Interpretation, kept within the tier.** On this self-generated reference:
  - dropping the rotation, velocity and root-yaw groups (position-only conditioning, the S1 situation) raises the
    hidden-interval error 3.7× (ric) or 5.3× (world);
  - it makes the generated interior inconsistent with the observed frames.
  - Replacement ON then exposes the inconsistency as a 134 mm boundary jump; OFF hides it by moving the observed frames
    (residual up to 0.60).
  - This reproduces the S1 pattern qualitatively. It does not show the S1 causes are the same, and it says nothing about
    real-video quality.

Costs: about 3.2–4.1 s sampling and 1.84 GB peak VRAM per cell; 39 s for the stage.

## 8. Blender candidate and native-time quality

**NOT BUILT, by the declared gate.** Source: `results/candidate/S1/candidate_result.json`, `1f3f83d2…`, `built: false`.

- No projection was forced. No `.blend`, Action, render, video, contact sheet or blinded review was produced.
- The C_OFF G4/G5 footprint/pin gates, the A/B/C_ON/C_OFF native-rate comparison and the agent visual review are therefore
  **NOT RUN**. Human native-rate review is **NOT RUN**.
- **User pins and footprint.**
  - No Blender file was authored in W132. Blender was used only to dump the frozen baseline, read-only. The only `.blend`
    files in W132 scratch are hash-verified byte copies of the four frozen inputs in `inputs/`, which are excluded from the
    manifest.
  - After all stages, the W126 baseline, the W126V-accepted `.blend`, the W129 B `.blend`, the W131 C `.blend`, the W126
    lineage head and value digest, every frozen W131 artifact and all W131 instruments were re-hashed. All are unchanged.
- **Reference numbers, not rerun.** The frozen W131 S1 quality numbers (C_ON arm-local entry/exit 429 / 153 mm vs B 24 / 25
  mm; mean world wrist deviation 292 vs 148 mm) stay as published.
- **A projected C_OFF would not have been a different interior.**
  - Its left-arm `ric` positions at samples 114–132 are bitwise those of C_ON.
  - Only the OFF reconstruction of observed PROXY/root features would have differed: shoulder, yaw.
  - The arm-local entry discontinuity against the untouched source context would remain, because the effective step is
    identical (§5).
  - This is an inference from §5/§6, not a measured rig result.

## 9. Tests and run status

| Item | Kind | Result |
|---|---|---|
| `tests/test_experimental_condmdi_causal.py` (10) | numpy, SYNTHETIC tensors: official-mask replica, metrics, flag-isolation rule incl. 6 negative mutations, context gate, mechanism classes, QA/CASE precedence, declaration/instrument tamper refusal | **10 PASS** |
| `tests/test_experimental_condmdi.py` (11, W131, unchanged) | time map, mask, axes, encode/decode, projection | **11 PASS** (`results/pytest/focused.xml`) |
| `tests/test_experimental_motionbricks.py` (15, unchanged) | the W129 helpers the harness imports | **15 PASS** (`regression.xml`) |
| Preflight: rights/terms, hashes, pins, frozen inputs | real metadata + files | **RUN, OK** |
| Preprocessing identity (fresh Blender dump → S1 conditioning) | real Blender + adapter | **RUN, PASS** (bitwise) |
| R0_w131code exact replay | **real pretrained model**, unchanged W131 runner | **RUN, PASS** (byte-identical `model_io.npz`) |
| R0 instrumented replay | real model | **RUN, PASS** (byte-identical) |
| Flag isolation (ON vs OFF) | real model, runtime hooks | **RUN, PASS** |
| R1_leak masked-input leak / determinism | real model | **RUN, PASS** (bitwise) |
| Runner negative controls (wrong input / reference SHA) | real runner | **RUN, PASS** (exit 3, no model load) |
| R1 observed-context gate | frozen thresholds | **RUN, FAIL** (255 mm / 64.7 mm / 72.6 mm / 9.95°) |
| Reference control, four cells + official-mask parity | real model | **RUN** (evidence tier: model-generated closed-loop synthetic) |
| C_OFF build, readbacks, G4/G5, Blender negative builds, renders, video, sheets, blinded review | real Blender | **NOT RUN** (gate) |
| Assessment | frozen rule | **CASE C** |

Also not run:

- human review and native-rate perception;
- the full repository suite;
- any other seed, text, mask, window, guidance, sampler or model;
- a fifth reference cell or an authentic reference;
- independent re-execution elsewhere;
- training.

**Deviations from the declaration:** none.

- The synthetic self-test before the instrument commit is disclosed in §2.
- §6 (code-level explanation) and the per-feature identity counts in §5 and §7 were computed after the outcome, as analysis
  of the required output identity/difference record. They are not new conditions and did not change any verdict.

## 10. Negative findings

- **Imputation OFF does not fix W131.** The generated arm is bitwise the same. The boundary jump against the real context is
  unchanged at 299 / 166 mm.
- **The learned conditioning path does not honour the S1 partial observation.** With replacement removed, the model's own
  reconstruction of the observed context departs from the source by:
  - 255 mm at the left wrist at sample 113;
  - 65 mm on average over all observed joint-samples;
  - 73 mm in root height;
  - 10° in root yaw.
- **The model fills the interior from its prior and the text, not from the boundary context.** The arm is already overhead
  at the first generated sample in both runs; the press timing loss reported in W131 is a property of the conditioned
  generation itself.
- **On the self-generated reference, position-only conditioning (the only honest S1 option) is far worse** than the
  official pos_rot_vel joint groups: hidden error 311 vs 58 mm world. With imputation ON, it is also the only configuration
  that produces a boundary discontinuity.
- **No candidate exists.** There is no evidence of a quality-improving learned candidate for this checkpoint on S1.

## 11. Causal alternatives: supported, refuted, confounded

| Alternative | Status | Basis |
|---|---|---|
| Diffusion-step replacement imputation causes the interior / boundary jump | **Refuted** for this checkpoint and runtime | exact bitwise identity of every generated feature, ON vs OFF (S1 and both reference pairs); measured, and explained by input pasting (§6) |
| The learned partial-joint conditioning path produces an interior inconsistent with the S1 observed context | **Supported** (one source, one seed) | R1 departs from the observed context by 255 mm at the boundary while being internally continuous |
| Position-only / missing rot-vel-yaw feature groups degrade conditioning fidelity | **Supported on a model-generated reference only** | cells ii/iv vs i/iii. On S1 these groups are UNKNOWN and cannot be supplied honestly, so this is not testable on S1 without fabricating features. |
| Noisy/incomplete source context (170 mm source step at 112→113; GT-free hypothesis) | **Confounded**, not separated | the source jitter sits right at the boundary; no control varies it |
| Rigid-rest upper-torso PROXY, declared floor constant, text semantics, out-of-distribution pose | **Confounded**, not separated | unchanged from W131; the matrix does not identify them |
| Training-distribution coverage (random-joints training always observes the root; S1 masks root xz) | **Not identified** | the reference cells observe root xz in all cells, so this factor is not varied |

## 12. Unresolved uncertainties

- **Single source and seed.** Whether the learned-conditioning departure holds for other S1-like sources, seeds or texts.
  By rule only one was run, so this is not a distribution claim.
- **The reference may favour the model.** The reference is self-generated, so it may be easier for the model than real
  motion. The 5× position-only degradation is therefore probably not an upper bound, but that is unverified.
- **How much of the context departure is due to the source itself.** Whether it is the jittery, possibly wrong S1 left-arm
  hypothesis (self-occluded far arm), and not the mask representation. A and the endpoints are GT-free.
- **The legal reading** is still documentary only, with no human licensing decision. The AMASS terms were not re-fetched.
- **How a human would judge W131's C_ON at 30 fps** is still unknown. No new visual material was produced.

## 13. Reproduction

Git Bash, repository root at `7017aa0`. The W131 scratch runtime, clones, checkpoint and results must be present, as
listed in the W131 Worklog §11. Every stage refuses an existing output directory.

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w132/results
$PY -X utf8 scripts/w132_condmdi_causal_harness.py preflight --out-dir $R/preflight
$PY -X utf8 scripts/w132_condmdi_causal_harness.py declare   --out-dir $R/declare --preflight $R/preflight/preflight.json
D="--declaration $R/declare/declaration.json --declaration-sha <sha printed by declare>"   # this run: 841822b0…
$PY -X utf8 scripts/w132_condmdi_causal_harness.py causal    --out-dir $R/causal $D
$PY -X utf8 scripts/w132_condmdi_causal_harness.py candidate --out-dir $R/candidate --causal-dir $R/causal $D
$PY -X utf8 scripts/w132_condmdi_causal_harness.py refctl    --out-dir $R/refctl $D
$PY -X utf8 scripts/w132_condmdi_causal_harness.py assess    --out-dir $R/assess --preflight $R/preflight/preflight.json \
    --causal-dir $R/causal --refctl-dir $R/refctl --candidate-dir $R/candidate $D
$PY -m pytest -q tests/test_experimental_condmdi_causal.py tests/test_experimental_condmdi.py
```

The runner can be called directly in the scratch env:
`C:/Users/dna10/animcv-w131/env/cm312/Scripts/python.exe -I scripts/w132_condmdi_causal_infer.py <request.json>`. The
request files are saved as `req_*.json` in each stage directory.

A re-run of `declare` produces a new timestamp and therefore a new SHA; pass that SHA to the later stages.

## 14. Evidence and changed files

**Evidence**, under `C:/Users/dna10/animcv-w132`; local, not published, 13 MB. `results/evidence_manifest.json` lists 58
files, SHA-256 `4da3c13b42a35fc562e09259883b44854343743efdf8f7f921cd765155edf06c`. It was written before this Worklog.

| File | SHA-256 |
|---|---|
| `preflight/preflight.json` | `508a788b…` |
| `declare/declaration.json` | `841822b0…` |
| `causal/causal_result.json` | `1d37d6f7…` |
| `causal/model_R0/model_io.npz` = `model_R0_w131code/model_io.npz` = W131 R1 | `2ec93555…` |
| `causal/model_R1/model_io.npz` | `cdd9fcd8…` |
| `causal/model_R1_leak/model_io.npz` | `38e93e14…` |
| `refctl/refctl_result.json` | `8c310fd0…` |
| `candidate/S1/candidate_result.json` | `1f3f83d2…` |
| `assess/assessment.json` | `35919277…` |

**Population:**

- one source episode (S1);
- one seed;
- the S1 causal pair plus a replay through the original runner and a leak/determinism run;
- one model-generated reference with four cells;
- zero Blender candidates.

**Changed files (repository):**

- New:
  - `src/app/experimental_condmdi_causal.py`
  - `scripts/w132_condmdi_causal_infer.py`
  - `scripts/w132_condmdi_causal_harness.py`
  - `tests/test_experimental_condmdi_causal.py`
  - `examples/experimental_condmdi/w132_declaration.json`
  - this Worklog
- Unchanged:
  - `docs/next_batch.md` and every historical Worklog;
  - all W131 instruments (`experimental_condmdi.py`, `w131_condmdi_infer.py`, `w131_condmdi_harness.py`,
    `w131_condmdi_preflight.py`, the W131 tests), loaded by path and LF-hash-verified;
  - the W129 adapter and Blender stages;
  - CLI, GUI, exporters, FramePose, MotionGraph and FK/IK;
  - canonical documents, global dependencies and automation scripts.

No weights, videos, renders or large artifacts are committed.

## 15. Resource and environment constraints

| Resource | Cost |
|---|---|
| GPU | 1.84 GB peak allocated VRAM; about 4 s per 100-step, 160-sample S1 sample and about 3.2 s per 120-sample reference cell |
| Wall time | under 2 minutes for all model stages; about 1.5 s to re-hash the 1.3 GB checkpoint |
| Scratch disk | 13 MB new (no new runtime) |

- The CondMDI runtime needs torch with sm_120 support, the CLIP package and mmengine. It is the existing W131 scratch
  Python 3.12 environment, unchanged. The AnimCV venv was not modified.
- Network was needed only for the metadata rights recheck. Without it, the preflight records `network: false` and relies on
  the local W131 copies.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 132
