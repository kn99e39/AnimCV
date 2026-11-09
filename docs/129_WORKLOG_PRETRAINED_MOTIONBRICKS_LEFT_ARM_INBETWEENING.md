# Worklog — Pretrained MotionBricks Left-Arm Inbetweening on a Real-Video BaseRig Action (2026-11-09)

Batch 129. An experimental, quality-first batch on `arch/single_frame_first`. It ran unattended on Main:

- local RTX 5080, driver 596.49;
- AnimCV Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9);
- a separate scratch MotionBricks runtime, outside the repository: Python 3.12.16, TensorRT 10.8.0.43, warp 1.19.0,
  numpy 2.4.3, onnxruntime 1.31.1;
- Blender 5.1.0, Windows 11 Pro 10.0.26200.

`docs/next_batch.md` was read only. No LabServer, no training, and no network use beyond `git fetch/push`, the MotionBricks
`git clone` + LFS download, and the scratch `pip`/`uv` installs.

Every number, render and verdict below is **AGENT-RUN** on Main. Every visual judgement is the agent's own review, made
from still contact sheets. It is not user, animator or professional validation.

## 1. Official CASE and scope

**Verdict: CASE B under the frozen rule.** Source: `results/assess/assessment.json`, SHA-256
`df6693d22d58cb95f4e130e16e757ab9519c84fe359323560cf530da8b42791c`.

**Gates.** G0–G7 all PASS. The candidate run was:

- the actual pinned pretrained model;
- real TensorRT inference with the exact declared native-time alignment (9 tokens, 30 samples);
- a separate, editable Blender Action;
- every preservation and pin gate met;
- negative controls that fail closed;
- a native-rate 30 fps A/B/C video.

**Quality criteria.** QA1 and QA3 PASS. **QA2, QA4 and QA5 FAIL.** The model candidate is *not* better than the simple
slerp control, and it is markedly less faithful to the source.

| QA | Criterion | C | B (control) | Result |
|---|---|---|---|---|
| QA2 | Arm-local entry step | 89.6 mm | 24.1 mm (limit 58.2) | FAIL |
| QA4 | Mean wrist deviation from A | 408 mm | 147.5 mm (limit 209) | FAIL |
| QA5 | Visual review | C was judged the least source-faithful; the slerp control was judged most natural | — | FAIL |

**What this result is.** A complete, authentic **negative quality result** for this generator family *as applied here*.
The model ran correctly, and the candidate is safe to inspect. But in the left-arm-only projection, the generated
transition made an arm swing down to the hip and back up. The source shows a press upward from the chest rack.

**Scope:**

- Opt-in scratch experiment only. One source (S1) is primary; S3 is a secondary transfer check.
- The candidate `.blend` is a **SCRATCH PROPOSAL**. It was never committed, accepted or placed in any lineage.
- No Production, CLI, GUI, exporter, FramePose, MotionGraph, FK/IK, schema or training change.

**Historical CASEs are untouched:** W121 D, W122 B, W123 B, W124 B, W125 A (transaction capability only), W126 C,
W127 B, W128 DATA BLOCKED. No historical Worklog was edited.

## 2. Starting state, revisions, environment

| Item | Value |
|---|---|
| Instruction blob | `390668973f13c2e952fe9c8db8a686609662ff07`, verified first |
| Starting commit | `cd6db54fee0a354875c4e30df6eabd958be1ae1a`; equal to `origin/arch/single_frame_first` at the first fetch. W128 carries the COMPLETE marker. |
| Pre-outcome instrument commit | `0c9a55c`: adapter, Blender stage, runner, harness, tests |
| Declaration v1 | `bacef7d` (`examples/experimental_motionbricks/w129_declaration.json`, SHA-256 `c9196162…`) |
| Harness fix | `939081a`, before any model output on S1/S3 (§8) |
| Declaration v2 (official) | `234bbb4` (`w129_declaration_v2.json`, byte copy of `results/declare_v2/declaration.json`, SHA-256 `0c34aab0755c24a7e30a43b11a356c9f5dc0c54e5adda39a5b54b8d6e9f0b6f8`). It records `outcomes_inspected: false` and `primary_model_output_exists: false`, and it binds the LF SHA-256 of all six instrument files. Every later stage refuses to run if any of them changed. |
| Final source revision | `234bbb4` for code. The commit adding this Worklog changes documentation only. The COMPLETE marker is a separate, final commit. |
| Scratch root | `C:/Users/dna10/animcv-w129`, outside the repository |

`C:/Users/dna10/animcv-w129` holds:

- `ext/motionbricks`, the checkout with weights, 4.7 GB;
- `env/`, the uv-managed Python 3.12 and the venv, 4.6 GB;
- `trt-cache/`, 0.7 GB;
- `inputs/`, byte copies of the frozen `.blend` files;
- `results/`, 309 MB.

## 3. Authentic data, model and runtime provenance (G0)

Source: `results/preflight/preflight.json`, SHA-256 `46119b9c…`.

**S1 (primary)** — CC BY-SA 4.0 kettlebell side view:

- Clip SHA-256 `de8feb9d…` (= W121).
- The frozen W126 `S1_replay` baseline `.blend` is `0803ab2b…`, value digest `25a09802…`.
- The authoritative `lineage/refs.json` head is `73dcd3dd…`, op REVERT, with a value digest equal to the baseline's.
- The head therefore has **no active pin**.
- Segment: source 79–318 maps to Action 1–240. Action = source − 78.

**S3 (secondary)** — the same checks:

- Clip `f6b90f2f…`, baseline `e1cfbae4…`, head `be9a1f09…` (REVERT), value digest `b9058745…`.
- Segment: source 0–153 maps to Action 1–154.

**Protected pin domain (authoritative W126 run).** The pin is `edits/W126V`, accepted in W126 and then reverted:

| Source | Pin source frames | Pin Action frames | Target | Tolerance |
|---|---|---|---|---|
| S1 | 210–218 | 132–140 | (−0.0927, 0.2199, 1.4506) m | 1 mm |
| S3 | 41–49 | 42–50 | — | — |

- Readback confirms that the W126V-changed S1 frames are Action 133–140.
- W124's wording "Blender frames 211–218" refers to **source** indices, not Action frames. This Worklog does not rewrite it.
- The pin domain does not overlap either declared interior.
- The Batch's approximate "source 241–249" (a W122 scratch request) is not in the W126 run. It would also end at the last
  context frame, outside the interior.

**Model and runtime:**

- **MotionBricks.** The checkout is `NVlabs/motionbricks` at `54fc563aeaaf819a753d72cc9fdc6713989cdd43`, equal to tag
  `v0.2.0`, with a clean worktree.
- **Weights.** `mmm_v6_soma_fp32.onnx` is 757,864,273 bytes, SHA-256 `856419dc72ae…` = its Git LFS OID. It is real
  protobuf, not a pointer. The graph is opset 17, 6,993 nodes, and **has no random operators** (ArgMax/TopK decoding), so
  no seed exists.
- **Manifest.** `soma_n1.json` (`330c04f0…`): 30 SOMA joints, 4 context and 4 target slots, 6–16 tokens.
- **Licences, recorded and not judged.**
  - Model: NVIDIA Open Model Agreement (`motionbricks/assets/onnx/LICENSE.md`).
  - Code: Apache-2.0 (docs and assets CC-BY-4.0).
  - The bundled BVH clips: NVIDIA Sample Data Evaluation License. They were used only for the runtime smoke test.
  - No weights, tokens, video or render outputs are committed.
- **Runtime.** The official `MotionBricksSOMAEngine` builds a TensorRT 10.8 plan on the RTX 5080 (sm_120). The plan build
  takes about 42 s once, and the plan is cached.

**Smoke test.** The bundled walk clip goes through one generation: 9 tokens, 30 valid frames. This only isolates
installation faults; it is **not** AnimCV quality evidence.

**Runtime negative controls** (all PASS):

| Control | Result |
|---|---|
| Missing ONNX | `DATA_BLOCKED_MISSING_MODEL` |
| FP16 bytes given under the FP32 hash | `BLOCKED_MODEL_IDENTITY` |
| Git LFS pointer file | `DATA_BLOCKED_LFS_POINTER_NOT_WEIGHTS` |
| Skeleton missing `LeftForeArm` | the engine refuses |

None produces a learned result.

## 4. Source ↔ Action ↔ SOMA representation (declared; G1)

**Time rule (verified in the engine source).**

1. The engine trims 3 boundary frames per end, and `correct_generated_diff` corrects the endpoints.
2. So with allowed tokens = {9} there are 36 raw frames and 30 valid samples. Sample *k* = SOURCE 249 + *k*.
3. Sample 0 is the last context pose (source 249 / Action 171). Sample 29 is the first target pose (source 278 / Action 200).

| | Source frames | Action frames |
|---|---|---|
| Context | 246–249 | 168–171 |
| Target (4 poses) | 278–281 | 200–203 |
| Written interior | 250–277 | 172–199 |

There is no retiming and no resampling.

**Spatial mapping (closed form, no IK).**

- Blender world (Z up, faces −Y) maps to SOMA model space (Y up, faces +Z) by the proper rotation (x, y, z) → (x, z, −y),
  det +1.
- Each mapped joint: R_soma = A · R_base · D_j. The constant D_j is fixed by a per-joint rest-frame alignment Q_j:
  - primary bone direction plus forward;
  - for feet: plus up;
  - for hands: plus thumb.
  - The alignment absorbs the BaseRig A-pose versus the SOMA T-pose (e.g. LeftArm 54.8°).
- The parent-relative channels then map exactly both ways: L_soma = D_pᵀ L_base D_j.
- Only `upperarm_l` (from SOMA LeftArm relative to LeftShoulder ↔ `clavicle_l`) and `lowerarm_l` (from LeftForeArm) are
  read back.

**Coverage of the 30 SOMA model joints.**

| Class | Joints | What they carry |
|---|---|---|
| DRIVEN (frozen W102 M1+S0 3D hypothesis, GT-free, not observed truth) | Spine1, LeftArm, LeftForeArm, LeftLeg/Shin, RightLeg/Shin | the Action's driven channels |
| OBJECT_PROXY | Hips | Armature Object rotation; the BaseRig has no pelvis bone. Position = thigh-head midpoint × 1.082 hip scale. **No root travel exists**, so travel is SYNTHETIC zero. |
| RIG_REST (undriven, UNKNOWN in the video, SYNTHETIC rest) | Spine2, Chest, Neck1/2, both clavicles, the whole right arm, both hands, both feet | rig rest |
| PROXY | Head ← neck_02 (no head bone); ToeBase ← foot (no toe bone) | rigid stand-ins |
| SYNTHETIC_SOMA_REFERENCE | Jaw, eyes, thumb/middle-finger ends | SOMA reference pose |

**What 17-joint FramePose and the W105 partial rig cannot determine:**

- the right arm, clavicles and spine above `spine_01`;
- head and neck orientation;
- hands, fingers, feet and toes;
- root translation and global travel;
- ground contact.

**The model never received fully observed source motion.** It saw a body whose right arm hangs at rig rest throughout.

**Mapping controls** (`results/controls/controls.json`, `f5d6accf…`). These use the real BaseRig evaluated by Blender,
before any S1 inference. G1 PASS.

| Control | Result |
|---|---|
| Real S1, all 240 frames: left-arm channel round trip | max 1.04e-7 (quaternion component) |
| Real S1, all 240 frames: no-mirroring | ok on every frame |
| Five posed controls (static / rotated + yawed object / left arm only / spine only / both) | primary-direction residual ≤ 1.5e-5°; round trip ≤ 3.5e-8; no-mirroring ok |
| Left-right | raising the left arm moves the SOMA LeftHand 0.605 m and the RightHand 0.0 m |
| Parent-child | rotating `spine_01` moves the chest by 74 mm and the hand follows; the left-arm local rotation is unchanged (1.2e-8); the legs are unchanged |

**Adapter negative controls (all fail closed):**

- a mirrored axis → `AXIS_MAPPING_IMPROPER`;
- a proper but wrong axis → the heading/toe check fails;
- a BaseRig missing `upperarm_l`;
- a SOMA skeleton missing `LeftForeArm`;
- an interior overlapping a protected frame;
- a non-30-sample window;
- a build request for `spine_01`;
- a build request for frame 210.

The last two are refused by the Blender stage before any file exists.

## 5. Real pretrained inference (G2, G3)

Source: `results/run/S1/model_R1/result.json`.

**Configuration (declared):**

- precision FP32; allowed tokens {9};
- 4 context + 4 target poses;
- heading alignment (1, 0); endpoint correction on;
- `model_to_runtime` identity.

**The run:**

| Item | Result |
|---|---|
| Tokens chosen | `pred_num_tokens` = 9 |
| Frames | 36 raw → 30 valid |
| Endpoint-corrected LeftArm / LeftForeArm at samples 0 and 29 vs the conditioning | ≤ 1.4e-7 rad |
| Projected basis at Action 171 / 200 vs the A keys | ≤ 2.4e-7 rad |
| GPU stage times | encode 875 ms (first-launch kernel/CUDA-graph setup), inference 3.7 ms, decode 0.3 ms, post-process 0.25 ms |
| Generate wall time | 0.89 s |
| Engine load (cached plan) | 0.72 s |
| VRAM, whole-GPU `nvidia-smi` | 1,728 → 2,488 MiB peak |

**Saved separately:**

- raw ONNX inputs and outputs in `raw_model_io.npz` (`1ad837e4…`);
- the decoded, corrected engine output in `engine_output.npz` (`8a528ef1…`);
- the projection log in `projection.json`.

The projection does no filtering, no retiming and no post-processing.

**Determinism.** R2, a new process with the identical request, is labelled a determinism rerun only. Its raw tensors and
engine output are **bitwise identical**. Only one primary sample exists; nothing was searched.

**ONNX Runtime CPU parity** (diagnostic). On the identical encoded inputs it chooses the same 9 tokens. The raw-output max
abs difference over the valid frames is 0.027 (S3: 0.12). TF32/TensorRT rounding is the likely cause; this was not
investigated further.

**Endpoint-correction magnitude.** This is how far the raw model's ends sat from the conditioning poses before the engine
blended them in.

| Joint | Start | End |
|---|---|---|
| LeftArm | 33.7° | 38.4° |
| LeftForeArm | 31.4° | 53.5° |
| LeftShoulder | 14.8° | 20.2° |
| Spine1 | 32.4° | 6.6° |

The raw model therefore did not reproduce the supplied context and target arm poses. A large share of the endpoint
matching is the engine's linear correction, not learned motion.

## 6. Source-preserving Blender candidate (G4, G5)

| Candidate | `.blend` SHA-256 | Action |
|---|---|---|
| B | `8899f433…` | `AnimCV_EXP W129 B S1` |
| C | `e04efb35…` | `AnimCV_EXP W129 C S1` |

**How each was built.** Each is a new file built from a byte copy of the frozen baseline:

1. the Action is copied under a new name;
2. the original Action datablock is removed from the new file;
3. only the `upperarm_l` / `lowerarm_l` keys at Action 172–199 are replaced;
4. provenance is set to `EXPERIMENTAL SCRATCH PROPOSAL`.

**Fresh-process readback** (the unchanged `experimental_action_stage.py readback`) of each file:

- 1 Action, 32 CONSTANT F-curves keyed 1–240, scene 30 fps, frames 1–240;
- 224 changed keys, all on `lowerarm_l` / `upperarm_l` at frames 172–199;
- every non-left-arm bone world point and the object matrix are **bit-equal** to A on all 240 integer frames and all 717
  subframes;
- the left arm is bit-equal outside 172–199;
- no subframe changes outside [172, 200).

**Untouched originals:**

- A is the untouched baseline: same SHA, same value digest.
- The W126 `refs.json` (`c0f1157d…`), the baseline `.blend` files and the source clip have the same hashes after the run.
- No `commit` or `accept` was called, and no lineage was touched.

**G5 hard pin.**

- C composed onto the W126V-accepted `.blend` (`Cpin_S1.blend`, `3116d2ea…`) has pin residuals identical to W126V on
  Action 132–140 (max 6.36e-7 m ≤ 1e-6).
- The pin frames are bit-equal: C vs A, and Cpin vs W126V.
- The composition changes nothing outside the left arm or the interior.

## 7. Matched A/B/C quality (G7, QA)

**Video.** `results/run/S1/abc_S1_30fps.mp4` (`f35420de…`): 240 frames written, 240 decoded, 30.000 fps, 8.000 s. It shows:

- the source crop with SOURCE frame IDs;
- A, B and C, each as a front body view plus an oblique left-arm close-up;
- an interior bar (Action 172–199 = source 250–277).

**Contact sheets:** `sheet_event_S1.png` (Action 170–175 = source 248–253), `sheet_midpoint_S1.png` (183–188),
`sheet_target_S1.png` (196–201) and `sheet_overview_S1.png` (every 5th frame). Blinded copies are in `blind/`.

**Wrist kinematics, Action window 164–207, interior steps 171→172 … 199→200.**

*World space (visible on screen):*

| Metric | A original | B slerp | C MotionBricks |
|---|---|---|---|
| Peak wrist step (mm) / frame | 400 / 173 | 242 / 177 | 309 / 173 |
| RMS jerk (m/s³) | 11,550 | 8,448 | 7,256 |
| Reversal flicks | 8 | 9 | 6 |
| Mean / max deviation from A (mm) | 0 / 0 | 148 / 297 | **408 / 682** |
| Max distance off the chord (mm) | 280 | 204 | **613** |

*Arm-local (wrist in the `clavicle_l` frame):*

| Metric | A original | B slerp | C MotionBricks |
|---|---|---|---|
| Peak step (mm) | 203 | 25 | 157 |
| Reversal flicks | 6 | 0 | 0 |
| Entry / exit C0 step (mm) | 93 / 3 | 24 / 25 | **90** / 10 |
| RMS jerk | 6,485 | 2,564 | 2,916 |
| Path length (mm) | 1,947 | 701 | 2,005 |

*Articulation:*

| Metric | A original | B slerp | C MotionBricks |
|---|---|---|---|
| Elbow flex range (°; 180 = straight) | 70–129 | 93–112 | 100–144 |

No branch has a near-straight frame, a bend flip, an opposite-hinge frame or a limb-length change; limb length is
0.0 mm by construction.

**Why two spaces.** The pre-declaration A/B dry run showed that the world-space jump around Action 171–177 is largely
carried by the **shoulder itself**: 154–244 mm per frame from the source-derived object rotation and `spine_01` tracks.
Those tracks are out of the edit domain, PRESERVED, and identical in A, B and C. QA1/QA2 were therefore declared on
arm-local kinematics before any model output existed. In the arm channels, A's largest jump is 170→171 (220 mm). That
step lies inside the frozen context, which no branch may edit.

**Visual review** (`results/review/visual_review_S1.json`, `81416a0a…`). It was written before reading the key.

- **Blinding was COMPROMISED, and the review must be read as effectively unblinded.** The pre-declaration dry run showed
  the agent renders of A and B at the same frames, so A and B were recognisable and C could be inferred by elimination.
  The key (`blind_key_SEALED.json`: X = C, Y = B, Z = A) confirmed that inference.
- **C.** The arm swings out sideways over 172–176 to a near-horizontal straight arm. It then **drops to hang at hip
  level** (181–186) and rises back past horizontal to overhead (191–199). Frame to frame in the stills it is smooth, but
  this is a large down-and-up detour that the source (a press upward from the chest rack) does not show. It reads as a
  stylised swing or lateral raise. This is inadvertent stylisation and a loss of source intent.
- **B.** A monotonic, even rise from the rack to overhead. No visible flick. Most press-like, with a uniform rate.
- **A.** The visible one-frame forearm jump at 172/173.
- **Answers.** V1 flick: A. V2 most natural: B. V3 wrong elbow: none. V4 least source-faithful: C. V5 boundary
  discontinuity: none distinct.
- **Not perceived:** native-rate playback. The agent judged stills only. No human reviewed anything.

**Secondary S3 (bent-over row, transfer check only)** (`results/run/S3/run_result.json`, `5a69dbb6…`). It used the
identical frozen configuration. Interior Action 111–138 (source 110–137); the pin at Action 42–50 is outside it.

| Check | Result |
|---|---|
| Tokens / valid frames | 9 / 30 |
| G4, G5 | pass |
| Determinism | bitwise |
| Video | 154 frames at 30 fps |

What C did on S3:

- It **introduced** a new 235 mm arm-local entry step (A 18 mm, B 9 mm).
- The engine's LeftArm start correction was **126°**.
- The mean deviation from A was 266 mm (B 91 mm).
- Arm-local RMS jerk was 2,256, against B's 61.
- In the stills, C's left arm moves behind the torso while the source pulls the bar up.

S3 does not change the S1 CASE.

## 8. Tests and run status (RUN / PASS / FAIL / NOT RUN)

| Item | Kind | Result |
|---|---|---|
| `tests/test_experimental_motionbricks.py` (15) | unit; SYNTHETIC SOMA-like skeleton + the real BaseRig rest JSON; MOCK numpy armature FK; 1 real-Blender refusal | **15 PASS** (`results/pytest/focused.xml`) |
| Light regression: `test_experimental_video_action.py` + `test_cli.py` | existing | **60 PASS** (`regression.xml`) |
| Preflight, smoke, 4 runtime negative controls | real TensorRT | **RUN, PASS** |
| Mapping controls (G1) + 8 adapter negative controls | real Blender | **RUN, PASS** |
| S1 primary inference + R2 rerun + ORT parity | **real pretrained model** | **RUN**: G2/G3 PASS |
| S1 B/C/Cpin build, fresh readback, G4/G5 | real Blender | **RUN, PASS** |
| S1 native-rate video and sheets (G7) | real Blender renders | **RUN, PASS** |
| S3 secondary | real | **RUN** (gates pass, quality negative) |
| Assessment | frozen rule | **CASE B** |

**Failed:** QA2, QA4 and QA5 (quality criteria, not tests).

**Not run:**

- the full repository suite;
- FP16 or any other precision, token count, window or seed — not run by rule;
- a human visual review;
- native-rate perception;
- independent re-execution;
- the four alternative generators (Kimodo, ARDY, MotionLab, ProjFlow), as the Batch requires.

**Post-outcome changes:** none.

**Pre-outcome instrument changes (disclosed):**

1. **Arm-local metrics.** After an A/B-only dry run on a scratch copy (no model output), arm-local metrics were added and
   QA1/QA2 were redefined on them. The rationale is recorded in the declaration's `pre_declaration_observations`.
2. **Harness bug.** The first `controls` attempt crashed on a dict-key type bug before any check ran
   (`results/controls_attempt1_keyerror`). The fix is `939081a`, and the declaration was re-issued as v2. The rules are
   unchanged.

## 9. Negative findings

- **The candidate loses source intent.** C replaces the press with a down-and-up arm swing: mean wrist deviation from A
  is 408 mm, and it reaches 613 mm off the chord. A lower world jerk and 0 arm-local flicks do not make it better
  animation.
- **The raw model did not honour the conditioning poses.** Its own endpoints missed the supplied left-arm context and
  target by 31–54° (126° on S3). The engine's linear endpoint correction then pulls the motion onto them, so the arm
  channels near the boundaries are partly a blend artefact. On S1 the C entry step (90 mm) is close to A's (93 mm). On S3
  it creates a new 235 mm jump.
- **Partial-body conditioning is out of distribution.** The rest-pose right arm, the fixed root, the proxy pelvis and head,
  and the jittery source torso were all fed to a model trained on full-body mocap.
- **The visible S1 discontinuity is not left-arm-only.** Much of it comes from the out-of-domain object and `spine_01`
  tracks, which every branch preserves, and from the arm jump at 170→171 inside the frozen context.
- **The visual blinding was compromised** by the agent's own pre-declaration dry run.
- **TensorRT and ORT CPU raw outputs differ by up to 0.027 (S3: 0.12)**, with the same token count.

## 10. Unresolved uncertainties

- Whether a human viewing the 30 fps video would judge C's swing as acceptable stylisation or as wrong. Only stills were
  judged, by the agent.
- How much of C's detour is caused by the synthetic and rest fill (right arm, root, head), how much by the source torso
  jitter, and how much by the endpoint correction. No ablation was run, by rule.
- Whether the source-derived A arm path or the source perception is itself correct. Both are GT-free hypotheses, and the
  S1 left arm is the self-occluded far arm. The 400 mm per-frame A jump is a likely perception error, but this is **not
  proven**.
- FP16 or TensorRT-versus-ORT differences in the token choice for other inputs.

## 11. Reproduction

Git Bash, repository root at `234bbb4`. Every stage refuses an existing output directory. The scratch runtime is set up as
follows:

```
# scratch runtime (once): clone + LFS, uv-managed Python 3.12 venv
git clone https://github.com/NVlabs/motionbricks.git C:/Users/dna10/animcv-w129/ext/motionbricks   # HEAD 54fc563 = v0.2.0
py -3.14 -m venv env/uvboot && env/uvboot/Scripts/pip install uv
UV_PYTHON_INSTALL_DIR=env/pythons uv python install 3.12 && uv venv --python 3.12 env/mb312
uv pip install --python env/mb312/Scripts/python.exe numpy==2.4.3 "warp-lang>=1.17.0" scipy==1.17.1 \
   tensorrt-cu12==10.8.0.43 tensorrt-cu12-bindings==10.8.0.43 tensorrt-cu12-libs==10.8.0.43 onnxruntime onnx usd-core==26.3
cp <W126 runs>/S{1,3}_replay/baseline/baseline.blend and accepted_03_*.blend -> C:/Users/dna10/animcv-w129/inputs/ (hashes in §3)
```

Then run the harness stages:

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w129/results
H=scripts/w129_motionbricks_harness.py
D="--declaration $R/declare_v2/declaration.json --declaration-sha 0c34aab0755c24a7e30a43b11a356c9f5dc0c54e5adda39a5b54b8d6e9f0b6f8"
$PY -X utf8 $H preflight --out-dir $R/preflight
$PY -X utf8 $H declare   --out-dir $R/declare_v2 --preflight $R/preflight/preflight.json
$PY -X utf8 $H controls  --out-dir $R/controls --preflight $R/preflight/preflight.json $D
$PY -X utf8 $H run --source S1 --out-dir $R/run --preflight $R/preflight/preflight.json --controls $R/controls/controls.json $D
$PY -X utf8 $H run --source S3 --out-dir $R/run --preflight $R/preflight/preflight.json --controls $R/controls/controls.json $D
$PY -X utf8 $H assess --out-dir $R/assess --preflight $R/preflight/preflight.json --controls $R/controls/controls.json \
    --run-dir $R/run --visual-review $R/review/visual_review_S1.json $D
```

A re-run of `declare` produces a new timestamp and therefore a new SHA, so pass that SHA to the later stages.

**Usage notes:**

- The adapter (`src/app/experimental_motionbricks.py`) is numpy-only.
- `src/blender/experimental_motionbricks_stage.py` has two modes:
  - `dump`: read-only evaluation; control overrides stay in memory only;
  - `build_variant`: always writes a new file, and edits only `upperarm_l`/`lowerarm_l` inside the declared interior.
- `scripts/w129_motionbricks_infer.py` runs only in the scratch Python 3.12 env with `-I`.
- Nothing is reachable from the default CLI or GUI.

## 12. Evidence and changed files

**Evidence**, under `C:/Users/dna10/animcv-w129/results`; local, not published.
`evidence_manifest.json` lists 2,533 files and was written before this Worklog. SHA-256
`b07154ed1b20da67a247fa4d9d5a28222ecb1087cc39d172bbe0af8eca113c02`.

Key files:

| File | SHA-256 |
|---|---|
| `preflight/preflight.json` | `46119b9c…` |
| `declare_v2/declaration.json` | `0c34aab0…` |
| `controls/controls.json` | `f5d6accf…` |
| `run/S1/run_result.json` | `9c889615…` |
| `run/S3/run_result.json` | `5a69dbb6…` |
| `review/visual_review_S1.json` | `81416a0a…` |
| `assess/assessment.json` | `df6693d2…` |
| `run/S1/blind/blind_S1_30fps.mp4` | `342a46a8…` |
| `run/S1/abc_S1_30fps.mp4` | `f35420de…` |
| S3 video | `0a84a5cf…` |

**Changed files (repository):**

- New:
  - `src/app/experimental_motionbricks.py`
  - `src/blender/experimental_motionbricks_stage.py`
  - `scripts/w129_motionbricks_infer.py`
  - `scripts/w129_motionbricks_harness.py`
  - `tests/test_experimental_motionbricks.py`
  - `examples/experimental_motionbricks/w129_declaration.json` (v1)
  - `examples/experimental_motionbricks/w129_declaration_v2.json` (official)
  - this Worklog
- Unchanged:
  - `docs/next_batch.md` and every historical Worklog;
  - `src/app/cli.py` and `gui.py`; the exporter and importer; FramePose, MotionGraph and FK/IK;
  - `experimental_video_action.py` and `experimental_action_stage.py` (the latter is called, not edited);
  - the rig, models, checkpoints, global dependencies and automation scripts;
  - every W124/W126 run file.

## 13. Resource and environment constraints

| Resource | Cost |
|---|---|
| Disk (scratch) | about 10.3 GB: checkout with LFS assets 4.7 GB, Python + venv 4.6 GB, TensorRT plan cache 0.7 GB, results 0.3 GB |
| One-time TensorRT plan build | about 42 s |
| Per-source run (dump, two inferences + ORT parity, 3 builds, 5 readbacks, 3 × 480 renders, 2 videos) | about 40 s |

- The MotionBricks runtime requires Python ≥ 3.12 and pinned TensorRT 10.8, so it lives in a separate scratch environment
  (uv-managed CPython 3.12.16). The AnimCV venv (3.11) was not modified.
- The documented test platform is an RTX 4090. Running on the RTX 5080 (Blackwell, sm_120) worked with TensorRT 10.8.0.43.
