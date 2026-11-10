# Worklog — Source-Constrained MotionBricks Elbow-Swivel Prior on the Real S1 Blender Action (2026-11-10)

Batch 135, an experimental, quality-first batch on `arch/single_frame_first`. It ran unattended on Main:

- local RTX 5080; no GPU inference was needed;
- AnimCV Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9);
- Blender 5.1.0, Windows 11 Pro 10.0.26200.

`docs/next_batch.md` was read only. No LabServer, no training and no model inference. The network was used only for
`git fetch/push` and one HTTP fetch of the NVIDIA Open Model Agreement text, for the rights review.

Every number, render and verdict below is **AGENT-RUN** on Main. Every visual judgement is the agent's own **unblinded
still-frame** review. No human or animator reviewed anything, and the agent did **not** perceive native-rate playback.

## 1. Official verdict and scope

**Verdict: CASE B under the frozen rule.** Source: `results/assess/assessment.json`, SHA-256
`b2336b8f4e35b3e8ef4abdffbe660f051e9d69bd483add7f5c3bf6665e399177`.

The CASE comes from three separate findings:

1. **Mechanical gates M0–M11 all PASS.** D and E are safe, separate actual-rig Actions. Each:
   - follows B's wrist to within 0.24 µm;
   - keeps its bone lengths;
   - keeps the hard pin;
   - is reversible.
2. **A learned contribution is established: LEARNED_EFFECT.** D differs from the matched neutral control E only in the
   learned prior.
   - The D-vs-E elbow difference reaches 117 mm (at Action 180) and averages 41 mm over the interior.
   - The swivel difference reaches 36.2° and averages 13.0°.
3. **The learned contribution is not a quality improvement.** The quality criteria do not all pass:
   - Q1 FAIL: D's arm-local elbow entry step is 52.3 mm, against a limit of 2 × B's 17.8 mm + 10 = 45.6 mm.
   - Q4 FAIL: the agent did not judge D more natural or more press-faithful than B or E.
   - Q5 NOT REVIEWED: native-rate perception was declared in advance as impossible for this agent, so Q5 could not pass.
   - Q2 and Q3 PASS.

**Interpretation.** The pretrained MotionBricks output does carry a usable, separable elbow-swivel signal. When it is
transplanted onto B's wrist path:

- it changes the elbow measurably;
- it stays geometrically valid;
- it adds a back-and-forth elbow swivel while the wrist rises monotonically;
- it gives no visible gain in naturalness or overhead-press fidelity over the deterministic slerp B.

This is a **safe negative** for "learned articulation adds visible value" on this clip.

**Scope:**

- Scratch experiment only.
- D, E, Dpin, Epin and Drev are **SCRATCH PROPOSALS**. None was committed, accepted or placed in any lineage.
- No Production, CLI, GUI, exporter, FramePose, MotionGraph, schema, Unity or FBX change.
- No change to `scripts/batch_automation.py`.

**Historical CASEs and identities are untouched:** W129 B, W130 DATA BLOCKED_LICENSE, W131 B, W132 C, W133 DATA
BLOCKED_LICENSE, W134 DATA BLOCKED / ACCESS_BLOCKED. No historical Worklog was edited, and W129's MotionBricks GPU
inference was **not** re-run.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `38de1e2e155e39c7f168373ca9b053418497ab0c` (verified first) |
| Starting commit | `5b113ed930ab31a2cee713cd9e8a0ae1da3994bb`. Equal to `origin/arch/single_frame_first` at the first fetch. W134 carries the COMPLETE marker. |
| Instrument commit (pre-outcome) | `4d4213f`: solver module, harness, synthetic tests |
| Declaration commit (pre-outcome) | `546e706`: `examples/experimental_elbow_prior/w135_declaration.json`, a copy of `results/declare/declaration.json`, SHA-256 `0c69dde0e13fb909687be8d1d0cfd07953fa51f6f645c531ab788a0a41f767fe` |
| Final source revision | `546e706` for code and declaration. The Worklog commit is documentation only. The COMPLETE marker is a separate, final commit. |
| Scratch root | `C:/Users/dna10/animcv-w135`, outside the repository, 276 MB |

**About the declaration copy.** Git normalised the committed copy to LF, so the SHA-256 is that of the scratch file.

**What the declaration records:**

- `outcomes_inspected: false` and `D_or_E_exists: false`;
- the LF SHA-256 of all 7 instrument files.

Every later stage refuses to run if the declaration or any instrument changed.

**Order of stages:** preflight → instruments commit → declare → declaration commit → controls → run → visual review →
assess. No instrument or rule changed after any D/E output existed.

## 3. Identity and rights gate (M0)

Source: `results/preflight/preflight.json`, SHA-256 `3a1391e3…`. Status OK.

**Frozen evidence.** Full SHA-256 values come from the official W129/W126 records. All are equal before and after the run
(M9).

| Artefact | SHA-256 |
|---|---|
| S1 clip (CC BY-SA, side-view double-kettlebell press) | `de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c` |
| A: W126 S1 baseline `.blend` (W129 input copy and W126 original) | `0803ab2bf31210fbb729bc81b76931729fdc3e12cce27b9507a5b8caf1d12402` |
| B: W129 endpoint-slerp | `8899f43341ac1a7a565229a9551b1f8a79943cc038ad519e5544af89e772a33f` |
| C: W129 MotionBricks projection | `e04efb3552f9808f8ae89520284b0d94e860a986c1e6ec475ed6b07683f9ef04` |
| W129 Cpin | `3116d2ea3340d8356beb6adbf5f60f33c026ca92f47f7216137b4a00f6e31b02` |
| W126V-accepted `.blend` (pin) | `7e2ab16511f4cbe111171df9af0bc2e1f7d8c09a57bd5a4feff284d34e908f40` |
| W126 `lineage/refs.json` | `c0f1157dc7f067736059f754e23070de8ab5e425ec858e7415e2d7df17389b1b` |
| W126V `proposal.json` | `582174bfcd3a284f9037409c945e2440b182a0c2593e618511b9effb280a9c84` |
| W129 declaration v2 | `0c34aab0755c24a7e30a43b11a356c9f5dc0c54e5adda39a5b54b8d6e9f0b6f8` |
| W129 assessment | `df6693d22d58cb95f4e130e16e757ab9519c84fe359323560cf530da8b42791c` |
| W129 `run_result.json` (records the B/C SHAs) | `9c88961523fa41881fd87e58d366de23138b24baca316dc33403c40f76da9a83` |
| W129 `engine_output.npz` | `8a528ef1bf7976f22bd6212eddf7c5b525ee5f30e27ee01f501d565baac6a100` |
| W129 `raw_model_io.npz` | `1ad837e417b69531b35aa3108fccd6c99e05b4f924741b56d58ceec44accfb6f` |
| W129 evidence manifest | `b07154ed1b20da67a247fa4d9d5a28222ecb1087cc39d172bbe0af8eca113c02` |

**C is the real pretrained result.** It is the W129 `.blend` whose SHA is recorded in W129's official run record, built
from the bitwise-deterministic TensorRT output. Model: `mmm_v6_soma_fp32.onnx`, SHA `856419dc72ae…`, from
`NVlabs/motionbricks` at `54fc563`. Nothing was regenerated or replaced.

**Fresh, independent recompute.** Byte copies of A, B, C and the W126V-accepted file were dumped and read back in new
Blender processes. The world points are **bit-equal** to W129's readbacks for A, B and C. The scene is 30 fps, frames
1–240.

**Rights determination: PERMITTED.**

- The local `LICENSE.md` only names the "NVIDIA Open Model Agreement", so the agreement itself was fetched:
  - stored as `rights/nvidia_open_model_agreement.html`, SHA-256 `95ff88bc…`;
  - "Release Date: April 2, 2026"; "(v. March 9, 2026)".
- It states:
  - "Works are commercially usable";
  - "NVIDIA does not claim ownership to any outputs generated using the Works or Derivative Works";
  - a "perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable license".
- This run evaluates an existing model **output** locally and redistributes neither weights nor outputs.

**Remaining rights uncertainty.** The fetched text is the version current on 2026-11-10. The version in force at the W129
download (2026-11-09) was not separately archived.

## 4. Prospective declaration (summary; the full record is the committed JSON)

**Time and domain.**

- Native 30 fps; Action = SOURCE − 78.
- The only edited footprint is `upperarm_l`/`lowerarm_l` rotation at Action 172–199 (SOURCE 250–277).
- Frames 171 and 200 are frozen.
- The pin (Action 132–140) is disjoint from the footprint.
- Every prior value is read at the **same Action frame id**.

**Convention.**

- Armature-space pose matrices come from fresh Blender dumps.
- The arm-local frame is the evaluated `clavicle_l` pose matrix (the parent of `upperarm_l`; unkeyed, so it follows the
  spine and object motion).
- S = `upperarm_l` head, E = `lowerarm_l` head, W = `hand_l` head.
- The numpy FK (`P_b = P_parent·Rest_parent⁻¹·Rest_b·Basis_b`) reproduces Blender's pose matrices on all 240 frames of A,
  B and C to 3.8e-5 armature units (float32 key precision).

**Swivel.** ψ = atan2(a·(r_p×u), r_p·u), where:

- a is the unit vector from S to W;
- u is the elbow offset with its a-component removed;
- r_p is a fixed reference axis with its a-component removed.

**Solver: closed form; no optimisation, smoothing, blending, weights or endpoint correction.** For each interior frame:

1. Rotate B's whole left arm rigidly about B's own shoulder→wrist axis by wrap(ψ_target − ψ_B). This sets
   `upperarm_l` basis′ = Mᵀ G M basis_B.
2. Leave the `lowerarm_l` key equal to B's.

The rotation fixes S and W exactly. It therefore keeps B's wrist, both bone lengths, the elbow flexion angle and the
elbow-local hinge relation. Its only freedom is the swivel. The branch is unique (wrap to (−π, π]).

| Branch | Target swivel |
|---|---|
| D | ψ_target = ψ_C: the swivel of C's own elbow about C's own shoulder–wrist axis, in C's own clavicle frame |
| E | ψ_target = ψ_B: neutral; C is withheld. E was declared in advance to equal B numerically (M8, solver neutrality). |

**What the prior contains, and what it excludes.**

- The prior is 28 scalars. It is hash-bound to the C file and re-derived and verified before use.
- C's wrist position or path is never a target.
- C's root, object and spine are factored out by the clavicle frame; C equals A there anyway.
- C's direction from shoulder to wrist is used only to *define* its swivel. The control runs show that a parent or root
  motion change leaves the prior unchanged (unit test).

**Singularity handling (all refusals, never repairs).** The solver refuses when:

- the arm axis is within 15° of the reference;
- the elbow offset is below sin 10° (near-straight arm);
- the swivel changes by more than 90° between consecutive frames (FLIP);
- the wrist is unreachable;
- the post-solve FK error exceeds 0.01 mm or 1e-6 rad.

**Disclosed pre-declaration instrument revision (input-only).**

- The a-priori swivel reference was **anterior**.
- An input-only margin inspection of B and C on frames 171–200 found that reference near-singular for C (minimum sin
  0.165 < sin 15° = 0.259). The inspection used axis directions only: no swivel values, no D or E.
- The declared outcome-blind rule therefore chooses, among the anterior, lateral and superior rig axes, the one with the
  largest minimum margin over B and C.
- It selected **lateral** (minimum sin 0.818; superior 0.525).
- This was frozen in the declaration before any D or E existed.

**Thresholds.**

- Mechanical gates M0–M11 (§5).
- LEARNED_EFFECT: the D-vs-E elbow difference is at least 10 mm **and** the swivel difference at least 3°.
- Quality criteria Q1–Q5:
  - Q1: elbow entry/exit step ≤ 2 × B + 10 mm;
  - Q2: 0 elbow reversal flicks;
  - Q3: swivel step ≤ 30° per frame;
  - Q4: agent still review;
  - Q5: native-rate review.
- CASE precedence: DATA_BLOCKED > D > C > B > A. A mechanical PASS never yields A.

## 5. Real Blender Actions and independent preservation (mechanical, not quality)

Source: `results/run/run_result.json`, SHA-256 `4934fd071ebd53ce542388eb57b342afc001313a2d6b3f1147a689658ccbd7e0`.

| File | SHA-256 | Action |
|---|---|---|
| D `D_S1.blend` | `2a5e441f75eeead09fc06242b93b595124b3da3a4e69c054e33e42f18e54cebe` | `AnimCV_EXP W135 D S1` |
| E `E_S1.blend` | `ac53a771ef5d3dfa80bee7df460fe0b1bd49873ff0950784bbc60a44bb1dc19f` | `AnimCV_EXP W135 E S1` |
| Dpin (D on W126V) | `e2aa9104eff87570ff9c4923a463fa412efb1c2c7a4bd65feaedf95ad63bc3a8` | |
| Epin (E on W126V) | `b95dcc4bc4c513b251e8181ab23a8c94bcfd604e613c1bd60f646ab64080edbb` | |
| Drev (D with A's keys restored) | `eb0a728d1664dfeeae64cb4c1d6e40e95b6e7362d16bdb991c950228fdf62040` | |

Each file was built from a byte copy of its parent by the unchanged W129 `build_variant` stage. The provenance schema name
is therefore `animcv_exp_w129_variant_v1`, with branch `W135 D`/`W135 E`. Each file was then read back in a **fresh
Blender process** by the unchanged `experimental_action_stage.py readback`.

| Gate | D | E |
|---|---|---|
| M2: 1 named Action, 32 CONSTANT F-curves keyed 1–240, 30 fps, scene 1–240 | PASS | PASS |
| M3: only `upperarm_l`/`lowerarm_l` keys at 172–199 changed vs A (224 values); non-left-arm world points and the object matrix bit-equal on 240 frames + 717 subframes; nothing outside [172, 200) | PASS | PASS |
| M4: interior wrist vs B (max) / entry and exit step difference vs B | 2.4e-7 m / 1e-8, 1.6e-7 m | 0 / 0 / 0 |
| M5: limb-length range; `lowerarm_l` key = B's; no near-straight frame | 0.21 / 0.41 µm; yes; none | same |

The remaining gates:

| Gate | Result |
|---|---|
| M6, pin | Dpin and Epin pin residuals at Action 132–140 equal W126V's (max 6.36e-7 m ≤ 1e-6). Pin frames are bit-equal, and the composed footprint is clean. PASS. |
| M7, reversibility | Drev is bit-equal to A in world points and object matrix on every frame and subframe. PASS. |
| M8, solver neutrality | E vs B maximum world difference is **0.0 m**: E is bitwise B. PASS. |
| M9, originals unchanged | PASS |
| M10, negative controls | PASS (§7) |
| M11, video | 240 frames written and decoded at 30.000 fps. PASS. |

## 6. Learned-prior causal effect, and quality: A / B / C / D / E

D and E share the solver, inputs, wrist, timing and lowerarm keys. They differ **only** in ψ_target. E = B bitwise, so
every D-vs-B difference below is caused by the learned prior.

**Learned effect.** Per-frame elbow difference D−E, in mm:

| Frames | Difference (mm) |
|---|---|
| 172–175 | 37–63 |
| 176 | 10 |
| 179–181 | 93–117 |
| 184 | 0 (a crossing) |
| 185–199 | 12–37 |

Max 117.4 mm, mean 41.1 mm; swivel max 36.2°, mean 13.0°. Classification: **LEARNED_EFFECT**.

**Arm-local elbow metrics** (clavicle frame; window 164–207; interior steps 171→172 … 199→200):

| Metric | A | B | C | D | E |
|---|---|---|---|---|---|
| Elbow entry / exit step (mm) | 64.4 / 8.2 | 17.8 / 17.8 | 63.3 / 3.3 | **52.3** / 13.7 | 17.8 / 17.8 |
| Elbow peak step (mm) @ frame | 214.5 @173 | 17.8 | 84.3 @174 | 61.5 @183 | 17.8 |
| Elbow path (mm) | 1,556 | 516 | 1,131 | 694 | 516 |
| Elbow reversal flicks | 6 | 0 | 0 | 0 | 0 |
| Elbow RMS jerk (m/s³) | 5,174 | 1,916 | 2,217 | 2,023 | 1,916 |
| Bend-plane step max / entry / exit (°) | 33.6 / 10.6 / 3.7 | 3.8 / 3.2 / 2.3 | 15.1 / 9.6 / 3.3 | 15.6 / 13.2 / 4.6 | 3.8 / 3.2 / 2.3 |
| Swivel step max / entry / exit (°) | 26.1 / 5.6 / 3.5 | 2.7 / 2.2 / 2.0 | 15.3 / 13.7 / 2.6 | 15.3 / 13.7 / 2.6 | 2.7 / 2.2 / 2.0 |
| Swivel peak velocity (°/s) / RMS acceleration (°/s²) | 783 / 12,264 | 81 / 1,396 | 459 / 4,664 | 459 / 4,664 | 81 / 1,396 |
| Flex range, interior (°; 180 = straight) | 70–129 | 93–112 | 100–144 | 93–112 | 93–112 |
| Elbow direction, interior mean: anterior / lateral / superior | 0.34 / 0.52 / −0.69 | 0.28 / 0.48 / −0.80 | −0.32 / 0.48 / −0.64 | 0.23 / 0.46 / −0.76 | 0.28 / 0.48 / −0.80 |

**What D's swivel does.** D's swivel series equals C's exactly (by construction): −86° → −61° (174) → −106° (180) → −46°
(188) → −22° (199). B's rises steadily from −84° to −27°. The learned prior therefore imports an elbow that swings one
way, then the other, while B's wrist rises monotonically.

**Wrist, world-space and parent-induced motion.** These are identical in B, D and E by construction:

- arm-local wrist entry/exit 24.1 / 25.0 mm;
- world wrist peak step 242 mm;
- mean deviation from A 147.5 mm, against C's 408 mm.

The world shoulder moves by up to **244 mm per frame** in every branch. That motion comes from the source-derived object
and `spine_01` tracks, which are out of the domain. No left-arm edit, D included, removes this parent popping.

**Quality criteria:**

| Criterion | Result |
|---|---|
| Q1 | FAIL: D entry 52.3 mm > 45.6 mm; exit 13.7 mm passes |
| Q2 | PASS |
| Q3 | PASS: 15.3° maximum |
| Q4 | FAIL |
| Q5 | NOT REVIEWED |

**Visual review** (`results/review/visual_review_S1.json`, `2784eb1a…`, written before `assess`).

| Branch | Observation |
|---|---|
| D vs E | Nearly indistinguishable in the declared arm close-up (about 4 mm/px). The D/E overlay strips show only a thin fringe, largest around 180–183. No flip, hyperextension or limb-length artefact in either. |
| D's swivel oscillation | Not perceptible as a distinct detour in stills. Whether it reads as a wobble at native rate is NOT REVIEWED. |
| D vs B / E | Not judged more natural or more press-faithful than either. Its wrist path, timing and flexion are B's. Its elbow-direction change moves *away* from A's anterior component (0.23 vs 0.28; A 0.34). |
| C | Still the W129 sideways swing and hip drop. |

**Limitations of the review:**

- It was unblinded.
- `sheet_arm_exit_S1.png` was produced but not opened.
- The D/E overlay strips (`diag_DE_strip_*.png`) are a post-run diagnostic visualisation of the declared renders; they
  add no rule.

**Video and sheets.**

- Video: `results/run/abcde_S1_30fps.mp4`, `d17caf66…`, native 30 fps. It shows the source crop with SOURCE IDs, then A,
  B, C, D and E (body view + left-arm close-up), plus an interior bar.
- Fixed-time contact sheets: `sheet_{arm,body}_{entry,middle,exit,overview}_S1.png` (entry 170–175, middle 183–188, exit
  196–201).

## 7. Tests and controls (RUN / PASS / FAIL / NOT RUN)

| Item | Kind | Result |
|---|---|---|
| `tests/test_experimental_elbow_prior.py` (14) | synthetic two-bone rig, no Blender. Covers quaternion sign/axes, time mapping, swivel geometry, parent/root invariance, solver constraints, neutral identity, prior verification, domain/pin/channel, reachability, flip, fps, reference singularity, reference rule | **14 PASS** |
| `tests/test_experimental_motionbricks.py` (15) | existing W129 instruments (reused) | **15 PASS** (`results/pytest/focused.xml`, `58a95a1d…`) |
| Real-data controls | real dumps + real Blender refusals (`results/controls/controls.json`, `9df5d285…`) | **18/18 fail closed**; FK reproduction 3.8e-5 |
| D/E real build, fresh readback, M-gates | real Blender | **RUN, all PASS** |
| Native-rate video + sheets | real renders | **RUN** |
| Assessment | frozen rule | **CASE B** |

**The 18 real-data controls and their refusals:**

| Group | Control → refusal |
|---|---|
| Source file | missing C → DATA_BLOCKED_MISSING_C; corrupt C (one byte flipped) → hash mismatch; corrupt C dump identity → SOURCE_IDENTITY |
| Prior provenance | mirrored prior (negated swivel) → PRIOR_PROVENANCE_MISMATCH; time-shifted prior (+1 frame ids) → PRIOR_FRAME_IDENTITY; time-shifted values under the same ids → PRIOR_PROVENANCE_MISMATCH; side-swapped prior → PRIOR_SIDE_MISMATCH; C prior verified against B → SOURCE_IDENTITY |
| Geometry and time | mirrored geometry (reflected dump) → IMPROPER_ROTATION; 24 fps scene → NOT_30_FPS |
| Domain | pin-overlap request → PIN_OVERLAP; frames 171–200 → DOMAIN_FRAMES; channel `spine_01` → DOMAIN_CHANNEL |
| Solver | wrist at 2× reach → WRIST_UNREACHABLE; π swivel jump → ELBOW_FLIP |
| Blender builds | `spine_01` → REFUSED_CHANNEL_OUTSIDE_DOMAIN; frame 171 / frame 200 → REFUSED_FRAME_OUTSIDE_INTERIOR, with no file written |

**Not run:**

- the full repository suite: Production is untouched, so only the affected experimental tests were run;
- the existing `test_experimental_video_action.py` / `test_cli.py` regression, because no shared source was modified;
- native-rate perceptual review; human or animator review; blinded review;
- any re-run of the W129 MotionBricks inference: not claimed and not done;
- S3 or any other clip;
- any alternative reference, threshold, prior or smoothing after the outcome (forbidden).

**Failed:** Q1 and Q4 (quality criteria, not tests). Q5 was not reviewed.

**Weak controls (disclosed).**

- The "missing C" and "corrupt C" controls check file and hash identity. They do not exercise a separate code path, but
  preflight would mark DATA_BLOCKED on either.
- The 24 fps control edits the real dump in memory; no 24 fps `.blend` was built.

## 8. Comparative interpretation (separated)

1. **Synthetic unit/contract.** 29 PASS. These say nothing about animation quality.
2. **Real Blender Action and preservation.** D and E are valid, separate, opt-in 30 fps Actions within the authorised
   footprint. They keep the wrist, lengths, pin and reversibility.
3. **Learned-prior causal effect.** Present: up to 117 mm and 36° of elbow difference, attributable to C's swivel alone,
   because E = B bitwise.
4. **Native-rate visual usefulness.**
   - In stills the change is barely visible and not an improvement.
   - It quantitatively adds a non-monotonic elbow swivel: path +35%, a 52 mm entry elbow step, a bend-plane step of 15.6°
     against B's 3.8°.
   - Native-rate perception was NOT REVIEWED.
5. **Production feasibility.** Not assessed. Nothing is integrated, and no future Generator-ownership trial is endorsed or
   rejected by this Worklog beyond the evidence above.

**Attribution:**

| Quantity | Owned by |
|---|---|
| Wrist path, timing, flexion | B: the deterministic slerp of A's endpoints, GT-free |
| Elbow swivel in D | C: MODEL-DERIVED, MotionBricks |
| Shoulder and world popping | the source-derived parent tracks, preserved |

D is a deliberate authored variation. It is **not** recovered hidden motion: no hidden ground truth exists, and the S1 left
arm is the self-occluded far arm.

## 9. Negative findings

- The learned swivel is not "elbow information about this press" in any validated sense. It is the swivel of a model
  trajectory whose arm was swinging down to the hip; its elbow direction is posterior (anterior component −0.32).
  Transplanted onto B, it makes D's elbow less anterior than both B and the source-derived A.
- The transplanted swivel oscillates (about 45° down then 60° up within 15 frames) while the wrist rises steadily. A
  learned prior with no wrist authority still carries motion that is incoherent with the constrained wrist.
- D's entry elbow step (52 mm) is about 3× B's. C's own swivel step at 171→172 (13.7°) is imported unchanged, because the
  declared solver applies no endpoint correction.
- The a-priori anterior swivel reference was near-singular for C. The swivel decomposition is sensitive to the reference
  axis whenever the B and C limb axes differ. Lateral was chosen by a declared input-only rule; a different valid
  reference would give a different D. This sensitivity was not explored, by rule.

## 10. Unresolved uncertainties

- Whether a human viewing the 30 fps video perceives D's swivel oscillation as a wobble, as natural variation, or not at
  all.
- Whether any reference-free swivel transfer (e.g. a rotation-minimising transport) or a different learned source would
  yield a useful prior. Not tested.
- Whether C's swivel is dominated by the endpoint-correction blend reported in W129 (31–54° LeftArm/LeftForeArm
  corrections) rather than by learned motion. Not separable here.
- Whether the source-derived A arm path, and so B's wrist target, is itself correct. Both are GT-free.
- Rights: the agreement version in force at the W129 download was not separately archived (§3).

## 11. Reproduction

Git Bash, repository root at `546e706`. Every stage refuses an existing output directory. The frozen W129 scratch files
must exist at the recorded paths with the recorded hashes. The agreement HTML must be at
`C:/Users/dna10/animcv-w135/rights/nvidia_open_model_agreement.html`; it was fetched with
`curl -sL -A "Mozilla/5.0" <agreement URL>`.

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src; R=C:/Users/dna10/animcv-w135/results
H=scripts/w135_elbow_prior_harness.py
$PY -m pytest -q tests/test_experimental_elbow_prior.py tests/test_experimental_motionbricks.py
$PY -X utf8 $H preflight --out-dir $R/preflight
$PY -X utf8 $H declare   --out-dir $R/declare --preflight $R/preflight/preflight.json
D="--declaration $R/declare/declaration.json --declaration-sha <sha of the new declaration>"
$PY -X utf8 $H controls  --out-dir $R/controls --preflight $R/preflight/preflight.json $D
$PY -X utf8 $H run       --out-dir $R/run --preflight $R/preflight/preflight.json --controls $R/controls/controls.json $D
# write $R/review/visual_review_S1.json, then:
$PY -X utf8 $H assess    --out-dir $R/assess --preflight $R/preflight/preflight.json --controls $R/controls/controls.json \
    --run-dir $R/run --visual-review $R/review/visual_review_S1.json $D
```

A re-run of `declare` gets a new timestamp, and therefore a new SHA. The official SHA is `0c69dde0…`.

## 12. Evidence and changed files

**Evidence**, under `C:/Users/dna10/animcv-w135/results`; local, not published. `evidence_manifest.json` lists 2,492
files and was written before this Worklog. SHA-256 `8543b5ecb95567da4a8cc204720254bc09c44004f1813fb5070aa7b4ba307c31`.

Key files:

| File | SHA-256 |
|---|---|
| `preflight/preflight.json` | `3a1391e3…` |
| `declare/declaration.json` | `0c69dde0…` |
| `controls/controls.json` | `9df5d285…` |
| `run/prior_C.json` | `86351597…` |
| `run/prior_E_neutral.json` | `9d9f4eb4…` |
| `run/solve_D.json` | `90c63e72…` |
| `run/run_result.json` | `4934fd07…` |
| `review/visual_review_S1.json` | `2784eb1a…` |
| `assess/assessment.json` | `b2336b8f…` |
| `run/abcde_S1_30fps.mp4` | `d17caf66…` |

**Changed files (repository):**

- New:
  - `src/app/experimental_elbow_prior.py`
  - `scripts/w135_elbow_prior_harness.py`
  - `tests/test_experimental_elbow_prior.py`
  - `examples/experimental_elbow_prior/w135_declaration.json`
  - this Worklog
- Unchanged:
  - `docs/next_batch.md`, every historical Worklog, and the living status/architecture documents (no milestone claimed);
  - the W129 adapter, Blender stages and harness (imported/called, not edited);
  - `experimental_action_stage.py`;
  - CLI, GUI, exporter/importer, FramePose, MotionGraph and Production retargeting;
  - `scripts/batch_automation.py`;
  - every W126/W129 run file.

## 13. Resource and environment constraints

| Resource | Cost |
|---|---|
| Wall time | preflight about 6 s (4 dumps/readbacks); controls a few seconds (3 Blender refusals; not timed); run 41 s (5 builds, 7 readbacks, 2 dumps, 5 × 480 renders, video + 8 sheets) |
| Disk | 276 MB scratch |
| GPU | not used |

- The prior is closed-form numpy, so the experiment needs only the AnimCV venv and Blender.
- The MotionBricks TensorRT environment from W129 was not invoked.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 135
