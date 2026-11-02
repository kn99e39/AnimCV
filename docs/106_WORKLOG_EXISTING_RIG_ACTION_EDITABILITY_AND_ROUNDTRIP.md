# Worklog — Existing Rig Action Editability and Round-Trip Contract (2026-11-03)

## Scope, starting state and principle

Task scope (summarized, not copied): an evidence-only artifact-contract diagnostic. It asks whether an **already
exported** BaseRig Action supports one bounded local motion edit and one local timing edit, with:
- the intended visible rig change;
- declared non-target animation preserved;
- exact inverses;
- Blender save/reopen;
- optionally, an FBX round trip.

It changes no reconstruction, rig, exporter or production code. All edits are scratch-only evaluation interventions on
copied Actions. This is not an editor, retiming method or authoring system.

- Branch `arch/single_frame_first`, starting HEAD `ee804f6974af34bed780e9c1e492221cef477a2f`; instruction blob
  `c1d8f1112f5d6260aa0dff595cd1fe9c5aaba6c7`. Both were verified before any work.
- `git fetch` showed no divergence from `origin/arch/single_frame_first`, and the working tree was clean.
- The worker lock (`state/active_batch.json`) was Batch 106 / RUNNING for this executor. No other job was visible.
- Run unattended by the scheduled automation worker, entirely on **Main**:
  - Python 3.11, `C:\Projects\AnimCV\.venv_build_full`;
  - Blender 5.1.0 (file version 5.1.29), `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`,
    always launched with `--factory-startup`.
- No LabServer job, no package install, no network data access, no model inference.

Regimes, kept apart throughout:

| Action | Regime |
|---|---|
| 3DPW windows | `benchmark_detector_observation`, frozen W102 mixed-source contract: HMR2 pelvis object, J1X left arm, S0 legs, 10 Hz bank |
| Real clip | `real_animcv_observation`, GT-free, W105 S0-only configuration |
| Synthetic | Minimal armature built in Blender; contract evidence only |
| Legacy smoke | Committed `examples/e2e_demo/animated_rig.blend`; never pooled |

No GT, oracle or benchmark value enters any edit. Only the M_Main slot (`FIXED`) is used; R11 (`RUNTIME`) is not.

## Answer

**CASE A — bounded Action editability supported, Blender-only.** The same edits did not pass a separate FBX gate.

1. **Edits on all three declared real Actions pass every predeclared Blender gate.** The Actions are 3DPW
   `complete_control`, 3DPW `arm_natural_most` (moving) and the historical real clip.
   - **N0:** a null save/reopen is bitwise identical.
   - **E1:** a 20° local `upperarm_l` swing on 5 keys moves the elbow by exactly 2·L·sin 10° = 0.0964 m.
     - The motion appears only inside the CONSTANT hold interval. It renders as 400–1,770 changed pixels per frame.
     - Every other F-curve, non-target local channel and non-descendant world transform is bitwise unchanged.
     - Rendered frames outside the hold interval are pixel-identical.
   - **E2:** ± key-time shifts of 3 keys match the CONSTANT step model exactly. Other tracks are untouched.
   - **E3:** the E2 inverse is bitwise. The E1 inverse is within 1.5e-8 per quaternion component, with no sign flips
     and ≤ 2.4e-7 m world error, and renders pixel-identical. It is not bitwise.
   - **E4:** every edit survives save/reopen bitwise.
   - **Determinism:** a repeat run on fresh copies is bitwise identical (57 conditions).
2. **FBX round trip (unchanged `src/blender/export.py`, default importer): failed as a separate gate on every real
   Action.**
   - Edit effects survive (≤ 0.42 mm vs Blender), and non-target bones are identical between edited and baseline
     imports.
   - But the *unedited* round trip drifts by 1.3 / 6.1 / 7.5 mm at integer frames, above the declared 1 mm.
   - The round trip also loses CONSTANT interpolation (imports LINEAR, so half-frame deviations reach 0.96 m) and the
     scene frame range (`frame_end` becomes 250).
   - It exports every Action stored in the file as a take (16 for `complete_control`, including R11 and other cases'
     Actions).
   - The synthetic FBX round trip passes (drift about 1.0e-6 m).
3. **The timebase finding stands and is preserved, not fixed.**
   - The real clip's keys sit at native 50 fps frame indices in a 30 fps scene: 3.96 s of motion plays in 6.6 s
     (0.6×). Save/reopen and FBX both keep that mismatch.
   - The real clip has a key on every frame, so the declared ±0.5-frame timing edit behaves asymmetrically:
     - +0.5 frame shows the group one frame late, and one keyed pose (key 101) is never displayed at integer frames;
     - −0.5 frame changes nothing at integer frames.
   - Dense native-rate keys leave no headroom for an order-preserving integer timing edit.

The CASE proves **editability of an existing Blender Action** under scratch copies. It does **not** prove an FBX
round trip, native-time playback, user-facing editing, style transfer, root/contact reconstruction or production
animation quality.

## 1. Preflight and identities (`verify/verify_report.json`, SHA `93610a8e…`)

W105 and audit:
- The W105 Worklog ends with its COMPLETE marker.
- `docs/AUDIT_W101_W105_2026_11_02.md` is present.
- W105 `evidence_manifest.json` SHA `cb0eca93…` equals the value W105 recorded.

All 15 used W105 files match their manifest entries:
- the 8 per-case `.blend`;
- `real_clip.blend`;
- `cases.json` and `video_cases.json`;
- both Action audits and both Blender reports.

Byte-identical scratch copies were made in `results/inputs/`. Only copies were ever opened.

Action identity:
- For all 9 `FIXED` Actions, the channel value digest re-derived from the copies equals the W105 audit digest.
- Examples: `complete_control` `d3f05754…`, `arm_natural_most` `ee1ead92…`, `real_clip` `4d168d95…`.

Recorded rig/environment SHAs:
- Skinned FBX `c2947ba7…`; rest snapshot `1dcdc826…`.
- `src/blender/export.py`, `src/blender/executor.py`, the W91 renderer and the W105 Blender wrapper are recorded in
  `verify_report.json`.

Inventory (`inventory.json`, SHA `678503c7…`), every W105 file:
- one armature `root` at object scale 0.01;
- 32-F-curve `FIXED` Actions with a single `OBroot` slot, CONSTANT interpolation, unit quaternions
  (|q| ∈ [0.99999997, 1.00000005]);
- scene 30 fps, `frame_start` 1;
- the skinned mesh `AnimCV_Mannequin_Skin` parented to the armature.

**The saved active Action in every W105 `.blend` is `RUNTIME` (R11), not M_Main.** Opening a W105 file shows R11 by
default. The historical renderer also saved every Action created so far into each file: `complete_control.blend`
holds 16 Actions.

Timebase association, both checked against the W105 rows:
- **3DPW:** `time_s == frame_index/30` for every row; Blender key frame = `frame_index − f0 + 1`.
  - So 3DPW Actions play at native 30 fps timing; keys are only at the 10 Hz bank samples (every 3 or 5 frames) and
    hold in between.
- **Real clip:** `time_s == frame_index/50`; key frame = `frame_index + 1` (199 keys, frames 1–199) in a 30 fps scene.

Legacy smoke (`animated_rig.blend`, SHA `6c074f2c…`):
- written by a newer Blender build (5.1.30), loaded with a version warning;
- scene 8 fps, frames 1–250;
- one Action `Generated_Motion` with 8 F-curves (`upperarm_l`, `lowerarm_l`);
- only **2 BEZIER keys, at frames 0 and 7**: the first key lies before `frame_start`; about 1.6° of motion.

It is not eligible for the declared 5-key window. It is recorded as inventory only; the rule was not widened.

## 2. Declaration (frozen before any edit was executed)

`declare/declaration.json`, SHA-256 **`a72f6dc57fc305e808b4fb92a5a449fddacb96c96b643ac6d4170fb1f516714d`**, with
`edit_outcomes_read: false`. Probe and analysis refuse to run without this hash.

**Selection** (from inventory metadata only):

| Case | Rule |
|---|---|
| `complete_control` | The W105 complete-control window (basketball_01 a0, frames #66–135, VAL) |
| Moving window | First sorted case id ≠ complete_control with all rows `same_person`, the target keyed at every sample, and an original target range ≥ 15° |
| `real_clip` | The only historical real-clip M_Main Action |

For the moving window:
- `arm_natural_most` qualified first (windowShopping_00 a0, #110–225, TEST; `upperarm_l` range 146.8°).
- `high_motion` and `identity_caveat` were excluded for `unresolved`/`other_person` rows.
- Four later windows were eligible but not first.

**Edits** (target bone `upperarm_l`; it is animated in all three Actions):

| Edit | Definition |
|---|---|
| E1 | `q' = q · rot(local X, +20°)` on the central 5 keys (n//2−2 … n//2+2), handles moved with the value. Post-multiplication rotates the bone direction by exactly 20°. |
| E2 | The central 3 keys of the 4-component track shifted by ±1 frame if both boundary gaps ≥ 2 frames, else ±0.5 frame. Values and interpolation are unchanged. Reorder, merge or out-of-domain shifts are refused (`ContractViolation`), never repaired. |
| E3 | The exact inverse (`q'·rot(X, −20°)`; shift back), applied to a reopened edited copy, then saved and reopened. |
| E4 | Every edit is saved and reopened. FBX: N0/E1/E2+ exported with unchanged `export_fbx` (`use_selection=False`, `object_types={'ARMATURE'}`, `bake_anim=True`, other options at Blender defaults), then imported with default importer settings into `read_factory_settings(use_empty=True)`. |
| N0 | The unedited copy with the active-Action pointer set to the declared M_Main Action (scratch-only, required to evaluate it), saved and reopened. |

Windows per case:

| Case | Keys | E1 key frames | CONSTANT hold | E2 group | E2 offset | Native times of the E2 group |
|---|---:|---|---|---|---|---|
| complete_control | 24 | 31–43 (step 3) | [31, 46) | 34, 37, 40 | ±1 | 3.30/3.40/3.50 s → +33.3 ms |
| arm_natural_most | 24 | 51–71 (step 5) | [51, 76) | 56, 61, 66 | ±1 | 5.50/5.67/5.83 s → +33.3 ms |
| real_clip | 199 | 98–102 | [98, 103) | 99, 100, 101 | ±0.5 | 1.96/1.98/2.00 s → +10 ms native (+16.7 ms Blender playback) |

**Sampling.** Every integer and half frame of the scene range:
- 139 samples (complete_control), 231 (arm_natural_most), 397 (real_clip);
- per sample: all 51 bones' local quaternion/location/scale and world head/tail/rotation, plus the object matrix.

**Tolerances** (fixed a priori from float32/FK precision; the declaration holds the full list):

| Check | Tolerance |
|---|---|
| Untouched keys | bitwise |
| Untouched local channels | ≤ 1e-7 |
| Non-descendant world | ≤ 1e-6 m and ≤ 1e-6 rotation elements |
| E1 elbow displacement | within 1e-4 m of 2·L·sin 10°, and ≥ 3 cm |
| E1 render | ≥ 100 px changed (8-bit difference > 16) per hold frame in ≥ 1 view; 0 px outside the hold |
| E3 key components | ≤ 1e-6 after sign canonicalisation |
| E3 world | ≤ 1e-5 m |
| E2 vs step model | ≤ 1e-7 |
| FBX unedited round-trip parity | ≤ 1 mm at integer frames |
| FBX non-target | ≤ 1e-5 m |
| FBX edit effect | ≤ 1 mm |

Descendants of the target (16 bones: `lowerarm_l`, `hand_l` and fingers) are *expected* to move in world space and
are not counted as harm. Their local channels must be unchanged.

**CASE mapping** (`w106_action_contract.decide_case`):

| CASE | Condition |
|---|---|
| D | Preflight/identity failure, N0 failure, non-determinism, or no real Action testable |
| B | Any real E1–E4 failure; the first failing layer is reported (Action representation → rig hierarchy → timebase → serialization) |
| C | Synthetic failure, or only a subset of real Actions/gates testable |
| A | Every declared real Action passes every Blender gate, and the synthetic probe passes |

FBX is reported separately and never changes the Blender CASE.

## 3. Paired results — real Actions (`analysis/analysis_report.json`, SHA `806de4c8…`)

| Gate | complete_control | arm_natural_most | real_clip |
|---|---|---|---|
| N0: keys / local / world / timing / null render | bitwise / 0 / 0 / equal / 0 px | same | same |
| N0: CONSTANT step model vs Blender, all tracks | 0 | 0 | 0 |
| E1 key error vs float32(q·offset) | 0 | 0 | 0 |
| E1 other F-curves changed | none | none | none |
| E1 non-target local / non-descendant world | 0 / 0 m | 0 / 0 m | 0 / 0 m |
| E1 target angle inside hold | 20.0000° (±1.1e-6) | 20.0000° | 20.0000° |
| E1 target and subtree outside hold | 0 | 0 | 0 |
| E1 elbow displacement in hold (expected 0.09645 m) | 0.096448–0.096448 | 0.096448–0.096448 | 0.096448–0.096448 |
| E1 max descendant world change (hand/fingers) | 0.241 m | 0.176 m | 0.176 m |
| E1 rendered hold frames / changed px (video, side) | 15 / 543–1430, 851–1771 | 25 / 400–507, 1652–1753 | 5 / 998–1163, 1215–1439 |
| E1 rendered frames outside hold | 10, 0 px | 16, 0 px | 4, 0 px |
| E2+ / E2− key frames exact, values bitwise, step-model error | yes / yes / 0 | yes / yes / 0 | yes / yes / 0 |
| E2+ changed integer frames | 34, 37, 40 | 56, 61, 66 | 99, 100, 101 |
| E2− changed integer frames | 33, 36, 39 | 55, 60, 65 | **none** (only 98.5, 99.5, 100.5) |
| E2 other tracks / non-descendant world | untouched / 0 | untouched / 0 | untouched / 0 |
| E3(E1) max key component error; sign flips; world | 2.5e-9; 0; 6.0e-8 m | 1.5e-8; 0; 2.4e-7 m | 7.5e-9; 0; 1.2e-7 m |
| E3(E1) digest equal to N0 | **no** (float rounding) | no | no |
| E3(E1) render vs N0 | 0 px | 0 px | 0 px |
| E3(E2+) | bitwise = N0 | bitwise = N0 | bitwise = N0 |
| E4 (all reopened edits = in-memory, bitwise) | yes | yes | yes |
| **Gates N0/E1/E2/E3/E4** | **all pass** | **all pass** | **all pass** |

How the E2 effects follow the step model:
- **3DPW (±1 frame, spacing 3 or 5):** each moved key's display interval slides one frame. Exactly one integer frame
  per key changes; for −1 it is the frame before each key.
- **Real clip +0.5:** the frames 99/100/101 show the previous key, so the group appears 1 frame (20 ms native, 33 ms
  Blender) late. Key 101's pose (now at 101.5) is never displayed, because key 102 at 102 supersedes it.
- **Real clip −0.5:** invisible at integer playback.

The step-model agreement (error 0) shows that Blender evaluates the edited keys exactly as declared. The asymmetry
is a property of CONSTANT keys on every frame, not an evaluation error.

Visual evidence:
- Location: `review/{case}_e1.jpg` and `review/{case}_e2.jpg`: rig-only renders with the W91 video and side cameras,
  and the skinned mesh. No source pixels are used.
- In each E1 sheet the left upper arm and forearm swing forward by a visible but modest amount.
- The |E1−N0| column lights only the left arm in hold frames and is black outside.
- The inverse column matches N0.

## 4. Synthetic contract probe (built fresh in each run)

Armature `base → upperarm_l → lowerarm_l → hand_l` plus `base → sibling`:
- 9 CONSTANT keys at frames 1, 4, …, 25;
- `sibling` has a missing key at frame 13.

Results:
- **Same gates as real: all pass.** Elbow displacement 0.104189 m vs 2·0.3·sin 10° = 0.104189. E3(E1) error
  6.0e-8; E3(E2) bitwise.
- **Missing key:** `sibling` at frame 13 evaluates to its frame-10 key (CONSTANT hold across the gap).
- **Quaternion sign:** negating one `sibling` key changes channel values by up to 1.50 but world transforms by 0. It
  is classified channel-inequal, rotation-equivalent. This is why E3 compares sign-canonical values. No real inverse
  produced a sign flip.
- **Start/end windows:**
  - an E1 on keys 0–4 also changes frames before the first key, because CONSTANT extrapolation holds the first key;
  - one on keys 4–8 holds to the end of the domain;
  - both are 20.000° inside the hold and ≤ 2.4e-6° outside; non-descendants are 0.
- **Negative-key case:**
  - the contract refuses a −1 shift of the first group ("would leave the frame domain [1, 25]");
  - forced anyway on a copy, Blender accepts a key at frame 0, outside the scene range, without complaint.
  - The same refusal was recorded for every real Action's start group.
- **Collision case:**
  - the contract refuses a +3 shift onto the next key;
  - forced, Blender keeps two keys at frame 19, and its evaluation differs from the last-key step model by up to
    0.124. A merged key time is ambiguous, which is why the contract refuses it.
- **FBX:** passes. Round-trip drift is 1.01e-6 m (Euclidean max) at integer frames, and the edit effect error is
  ≤ 3.6e-7 m.

## 5. FBX round trip (separate gate; Blender-only conclusion otherwise)

Each FBX imported cleanly (FBX 7400): one unique matching Action `root|root|<name>`, no missing bones, plus leaf
`*_end` bones that the exporter adds by default.

| Measure (integer frames unless stated) | complete_control | arm_natural_most | real_clip |
|---|---|---|---|
| Unedited round trip vs Blender N0, max abs coordinate | **1.29 mm** | **6.10 mm** | **7.51 mm** |
| Frames with Euclidean drift > 1 mm | 2 of 70 | 32 of 116 | 80 of 199 |
| Object rotation drift, max (median) | 0.076° (≈0) | 0.271° (≈0) | 0.314° (≈0) |
| Armature-space bone drift, max (median) | 0.50 mm (0.07) | 1.58 mm (0.19) | 2.96 mm (0.75) |
| Edited vs baseline import, non-descendant bones | 0 | 0 | 0 |
| E1 effect, FBX vs Blender | 0.13 mm | 0.04 mm | 0.42 mm |
| E2+ effect, FBX vs Blender | 0.17 mm | 0.12 mm | 0.05 mm |
| Half-frame drift (CONSTANT → LINEAR) | 0.42 m | 0.96 m | 0.35 m |
| Imported interpolation; target keys | LINEAR; 47 (70-frame bake, reduced) | LINEAR; 47 | LINEAR; 199 |
| fps / frame_start / frame_end | 30 / 1 / **250** (was 70) | 30 / 1 / **250** (was 116) | 30 / 1 / **250** (was 199) |
| Takes in the FBX | 16 Actions | 4 | 2 (`FIXED` + R11 `RUNTIME`) |

FBX verdict:
- **Failed the predeclared round-trip parity gate on all three real Actions.**
- Passed the edit-effect and non-target sub-gates.
- The edit representation itself survives export; the baseline round trip is what is lossy.

Post-hoc attribution (`fbx_attribution/fbx_attribution.json`, SHA `7c58c231…`; descriptive only; no exporter or
importer option was varied):
- The drift is sporadic: the median is about 0.
- It appears both in the Armature Object rotation (up to 0.31°, which moves bones up to about 8 mm at ~1.5 m) and in
  armature-space bone channels (up to 3 mm).
- It is absent in the synthetic armature, which has an unanimated object at scale 1.
- This fits lossy key handling in the default bake (the default key simplification and/or rotation-order
  conversion) acting on the scale-0.01, object-rotation-animated BaseRig. The cause was not isolated, because isolating
  it would require changing exporter flags.

## 6. Timebase and animation semantics

| | 3DPW (both windows) | Real clip |
|---|---|---|
| Native rate | 30 fps video, 10 Hz bank samples (every 3 or 5 frames) | 50 fps, 199 frames |
| Blender scene | 30 fps; keys at native frame offsets | 30 fps; keys at native frame indices |
| Native span vs Blender playback | 2.30 s vs 2.30 s; 3.83 s vs 3.83 s (×1.0) | 3.96 s vs 6.60 s (**×0.6**) |
| After N0 / E4 save-reopen | unchanged | unchanged (mismatch preserved) |
| After FBX | fps kept; range reset to 250 | fps 30 kept (mismatch preserved); range reset to 250 |

Further timing properties:
- **No in-between motion exists.** All tracks are CONSTANT, and every half frame equals the previous key (step-model
  error 0 on all tracks). In 3DPW Actions each sampled pose is held for 3–5 frames, and edits act on whole hold
  intervals. Interpolation was not changed.
- The optional scratch-only 50 fps scene sensitivity was **not executed**. The mismatch is quantified from metadata
  and preserved as a known negative finding.

Not covered:
- the W105 R11 flicker;
- missing root translation and contact semantics;
- the 10 Hz cadence as temporal reconstruction.

## 7. Negative findings and limits

- **FBX round trip is not established.** Unedited drift reaches 1.3–7.5 mm, CONSTANT becomes LINEAR, the frame range
  is reset, and all stored Actions are exported. Only an FBX-preserved *edit delta* is shown.
- **The W105 `.blend` is not a clean handoff artifact.**
  - Its saved active Action is R11 (`RUNTIME`); the M_Main Action is only reachable by name.
  - It stores other cases' Actions.
  - An FBX export with existing settings ships all of them.
- **Dense native-rate keys (real clip) leave no integer timing headroom.**
  - An order-preserving shift is only possible at sub-frame scale.
  - That shift is asymmetric at integer playback: +0.5 drops one keyed pose; −0.5 is invisible.
- **The E1 inverse is not bitwise** (float32 rounding, ≤ 1.5e-8). Its digest differs from N0, but it is
  render-identical. The E2 inverse is bitwise.
- **The real clip's 0.6× playback** persists through every round trip.
- **Legacy smoke is not editable under the declared rule:**
  - only 2 BEZIER keys, one at frame 0 before the scene start;
  - 8 fps scene;
  - written by a newer Blender build.
- Blender silently accepts out-of-domain and colliding key times. Only the evaluation contract refused them.
- **Coverage is narrow:**
  - one bone (`upperarm_l`), one rotation axis, one magnitude (20°) and one window per Action;
  - three real Actions from reused, non-holdout data;
  - legs, the root object, `spine_01` and other axes were not edited;
  - "visible" means measured elbow displacement plus a pixel count, not a user judgement.

## 8. Unresolved uncertainties

- The precise mechanism of the FBX baseline drift: simplification vs rotation conversion vs the animated object at
  scale 0.01. It was not isolated, because isolating it would change exporter flags.
- Whether a consumer engine would interpret the FBX's 16 takes and LINEAR baked keys as intended.
- Whether edits on root, leg or `spine_01` tracks behave like the arm edit (expected from the FK checks, but not
  measured).
- Whether the R11-saved active Action or the 0.6× timebase confuses a human editor opening W105 files. This was not
  user-tested.

## 9. Tests

`tests/test_w106_action_editability.py`, final run **14 passed** (`results/w106_tests_final.xml`):
- quaternion offset/exact inverse, the exact-20° bone-direction change, sign equivalence and canonical form;
- E1/E2 windows and offsets;
- shift refusals (merge, negative key, past end);
- the CONSTANT step model, hold intervals including ±∞ ends, and the half-frame asymmetry;
- deterministic metadata-only selection;
- the CASE mapping;
- digest convention equal to the W105 audit;
- static check: no GT/torch/in-place save, and the exporter is the only production import;
- stored: declaration hash and `edit_outcomes_read: false`, analysis↔declaration consistency, originals unchanged.

Before the declaration: 11 passed, 3 stored-output tests skipped (`w106_tests_predeclare.xml`).

Regression of the existing Action/export/rig tests: `test_blender_executor.py`, `test_w105_rig_utility.py`,
`test_retarget_solver.py` — **40 passed** (`regression_action_export_rig.xml`).

Execution failures, retained:
- **analyze, attempt 1, crashed.** It looked up a predeclared Action name for the synthetic case, whose Action is
  created at probe time.
  - The lookup was fixed to use the probe's Action name. This is a construction bug, not a contract change.
  - The log is kept as `analysis_attempt1_failed.log` and the empty directory as `analysis_attempt1_failed/`.
- No probe, render or Blender stage failed.

Post-result code changes, all in `scripts/w106_action_editability.py`:
- the fix above;
- the new `fbx-attribution` stage (descriptive, labelled post-hoc, no gate).

`w106_action_contract.py` and `w106_action_blender.py` were not changed after the declaration.

Not executed:
- broad production regression (no shared contract changed);
- scratch 50 fps scene sensitivity;
- exporter-flag sensitivity (out of scope by DO NOT TUNE);
- edits on non-arm tracks;
- unseen footage;
- user study;
- LabServer jobs.

## 10. Reproduction (Git Bash on Main, repo root)

```
PY=/c/Projects/AnimCV/.venv_build_full/Scripts/python.exe; BL="/c/Program Files/Blender Foundation/Blender 5.1/blender.exe"
O=C:/Users/dna10/animcv-w106/results; W=C:/Users/dna10/animcv-w105; S=a72f6dc57fc305e808b4fb92a5a449fddacb96c96b643ac6d4170fb1f516714d
B=("$BL" -b --factory-startup -t 2 --python-exit-code 1 -P scripts/w106_action_blender.py --)
$PY -X utf8 scripts/w106_action_editability.py verify --w105 $W --out-dir $O/verify
"${B[@]}" inventory --spec $O/verify/inventory_spec.json --out $O/inventory.json
$PY -X utf8 scripts/w106_action_editability.py declare --w105 $W --verify $O/verify/verify_report.json --inventory $O/inventory.json --starting-head ee804f6974af34bed780e9c1e492221cef477a2f --out-dir $O/declare
"${B[@]}" probe  --declaration $O/declare/declaration.json --declaration-sha $S --out-dir $O/probe
"${B[@]}" render --declaration $O/declare/declaration.json --probe-dir $O/probe --out-dir $O/render
"${B[@]}" probe  --declaration $O/declare/declaration.json --declaration-sha $S --out-dir $O/probe_repeat
$PY -X utf8 scripts/w106_action_editability.py analyze --declaration $O/declare/declaration.json --declaration-sha $S --probe-dir $O/probe --repeat-dir $O/probe_repeat --render-dir $O/render --out-dir $O/analysis
$PY -X utf8 scripts/w106_action_editability.py fbx-attribution --declaration $O/declare/declaration.json --probe-dir $O/probe --out-dir $O/fbx_attribution
$PY -X utf8 scripts/w106_action_editability.py compose --declaration $O/declare/declaration.json --render-dir $O/render --out-dir $O/review
$PY -X utf8 scripts/w106_action_editability.py manifest --base C:/Users/dna10/animcv-w106 --verify $O/verify/verify_report.json --w105 $W --starting-head ee804f6974af34bed780e9c1e492221cef477a2f --declaration-sha $S
$PY -X utf8 -m pytest tests/test_w106_action_editability.py -q
```

Each output-writing stage refuses an existing directory. Re-running into a fresh base reproduces the run.

Measured runtimes:

| Stage | Runtime |
|---|---|
| verify / declare / analyze | a few seconds each |
| inventory | < 2 s |
| probe | 17 s (repeat 16 s) |
| render | 26 s (1,050 PNGs) |
| focused tests | < 1 s |
| regression | 4.3 s |

Outputs are local in `C:\Users\dna10\animcv-w106\` (263 MB) and are not committed. They include scratch `.blend`/FBX
copies, per-condition `.npz`/JSON, renders, contact sheets and logs.
- `evidence_manifest.json`, SHA `f3003fe54e68c6cf182313c0b784468275b7ed039f50f574ad10e64eb8fb97db`.
- It covers 1,424 files.
- It records `all_originals_unchanged: true`: every used W105 file and the legacy `animated_rig.blend` have the same
  hashes after the run.

## 11. Changed files, resources and environment

New files only:
- `scripts/w106_action_contract.py`: the pure contract (edit math, step model, shift refusal, selection, tolerances,
  CASE mapping).
- `scripts/w106_action_blender.py`: the Blender inventory / probe / render stages. It imports only the W91
  `_fcurves` helper and the unchanged `src/blender/export.py`.
- `scripts/w106_action_editability.py`: verify / declare / analyze / fbx-attribution / compose / manifest.
- `tests/test_w106_action_editability.py`.
- This Worklog.

Script SHAs at commit:

| File | SHA |
|---|---|
| contract | `d7ff860f…` |
| Blender | `8d33879c…` |
| host | `c932e613…` |
| tests | `f0dcb063…` |

Nothing else changed:
- no `src/`, rig asset, renderer, exporter setting, W102–W105 artifact or Worklog;
- not `docs/next_batch.md`;
- no automation file.

Resources:
- Main only; CPU plus Blender Workbench GPU rendering for about 30 s; no model inference; no package change.
- The private real-clip source frames were not read or copied. Contact sheets contain rig renders only.

Final source revision: the commit adding this Worklog on `arch/single_frame_first`, the next commit after `ee804f6`.
The COMPLETE marker is added in a separate final commit.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 106
