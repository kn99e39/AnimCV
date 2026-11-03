# Worklog — FBX Subframe Motion Semantics and Interpolation Ownership (2026-11-03)

## Human Assessment

- **SUCCESS:** The half-frame failure reported in W107 is reproduced exactly. Its first evidenced cause is the FBX
  serialization of interpolation. Evidence chain:
  - **Source.** All three real source Actions are CONSTANT. Blender evaluation confirms they are held: every
    subframe sample equals its integer frame bitwise.
  - **FBX file.** The exporter writes every AnimationCurve of every take with `KeyAttrFlags = 0x6104`, which decodes
    to LINEAR. Keys sit only on integer frames (±1 tick of rounding). Two independent parsers agree.
  - **Import.** The default importer produces LINEAR curves.
  - **P2 intervention.** P2 sets *only* the interpolation property of a copied, imported V1 Action to CONSTANT.
    Subframe error falls from 0.594 / 1.071 / 0.433 m to 0.95 / 1.31 / 1.66 µm (complete_control /
    arm_natural_most / real_clip). Integer-frame motion and every other key field are bitwise unchanged.

  The predeclared synthetic contrast holds:
  - The **step** control falls from 0.246 m to 0.77 µm under P2.
  - The **smooth** LINEAR control is *harmed* by P2, rising from 0.51 mm to 86 mm (168×).

  A fresh repeat probe is bitwise identical. **CASE A**, under the frozen rule.
- **FAILURE / NOT FIXED:** Nothing was repaired.
  - The production path (V0) still loses the hold. So does the W107 V1 diagnostic: both write LINEAR-flagged curves.
  - P2 is a scratch diagnostic of representational sufficiency, not a fix. Applied blindly it destroys continuous
    motion (the smooth control).
  - It reproduces the hold here only because two conditions both hold:
    - every source key lies on an integer frame;
    - V1 keeps a key on every frame.

    P2 was not tried on V0's simplified keys.
  - Every other W107 defect persists, unchanged and unresolved:
    - `frame_end` 250;
    - 16 / 4 / 2 takes;
    - the real clip's 50 fps keys in a 30 fps scene;
    - V1's 2.2–2.8× file size and 6–13× imported key cost.
- **UNCERTAIN:**
  - **Importer.** Whether Blender's importer would honour a CONSTANT flag was *not* tested behaviourally. That
    would need a hand-edited FBX, a second intervention. Static inspection of the declared add-on source shows
    `import_fbx.py` never reads `KeyAttrFlags` and always writes LINEAR. So by code evidence the importer would
    also discard it, but that is not behavioural evidence.
  - **Other consumers.** Non-Blender FBX consumers were not tested.
  - **Object vs pose channels.** P2 changed all F-curves at once, so the share carried by object channels versus
    pose channels is not separated.
  - **Euler interpolation.** The smooth control keeps a 0.51 mm residual under matching LINEAR semantics. That is
    probably Euler-vs-quaternion interpolation between baked frames, but it was not isolated.
- **ALTERNATIVE EXPLANATIONS (status):**

  | Explanation | Status |
  |---|---|
  | Sampling phase | Excluded. Identical exact times; P2 gives µm error at every phase .25 / .50 / .75. |
  | Action/slot mismatch | Excluded. Same name-matched Action; the P2 copy uses the same slot `OBSlot`; channel digest identical. |
  | Key-boundary / time-mapping error | Excluded. Imported key times are exact integers (max deviation 0.0). FBX KeyTimes deviate by ≤ 1 tick (6.5e-10 frame). |
  | Quaternion/Euler conversion as the half-frame cause | Not supported. Under P2 the Euler curves give µm error. Its LINEAR-interpolation share is the sub-mm smooth-control residual. |
  | Take selection | Excluded. Every take in every file encodes LINEAR; selection is a name match. |
  | Synthetic vs real rig | The same signature appears on both. They are reported separately and never pooled. |
  | Object vs pose animation | Not separated. Both carry loss under V1: object rotation reaches 12–18° at subframes, ≤ 6e-5° under P2. |

**CASE A — mechanism supported (interpolation semantics lost at FBX export serialization).** The proven boundary is
export-side: the hold is already absent from the encoded FBX curves. The importer-side counterpart is static
inspection only, so ownership of a CONSTANT-preserving round trip remains partly unresolved. CASE A authorizes no
production change, and no production file was modified.

## Scope and starting state

Task scope (summarized): an evidence-only trace of continuous-time semantics through the path
original Blender Action → frozen FBX export → FBX curve data → default Blender import. It covers the three fixed W106
M_Main Actions, plus one predeclared scratch-only interpolation intervention (P2) and synthetic step/smooth controls.
- No exporter, importer, Action original, rig, FPS, timebase, retargeting, checkpoint or historical Worklog was changed.
- There was no training, inference, rendering, GPU use, network access or LabServer work.

Starting state:
- Branch `arch/single_frame_first`; starting HEAD `730a5dfd15d93ec008fb8017448a89424b88f99e`; instruction blob
  `4c3255731952d1a43d8898dc44892a7b09512948`. Both were verified before any work.
- `git fetch` showed `origin` equal to HEAD, and the working tree was clean.
- The worker lock was Batch 108 / RUNNING. Its executor pid is this session, under poller 9120.
- No other Blender process was running.
- Run entirely on **Main**:
  - Python 3.11 at `C:\Projects\AnimCV\.venv_build_full`;
  - Blender **5.1.0** (`adfe2921d5f3`), always `-b --factory-startup -t 2`;
  - FBX add-on **5.15.0**. Its source SHAs are equal to W107's, and `options.json` is byte-identical to W107's
    (SHA `7d1a5472…`).

Regimes:

| Action | Regime |
|---|---|
| `complete_control`, `arm_natural_most` | `benchmark_detector_observation` (3DPW, W102 mixed-source contract) |
| `real_clip` | `real_animcv_observation` (GT-free historical clip) |
| `syn_step`, `syn_smooth` | Synthetic controls; never pooled with the real Actions |

These are reused, non-holdout data.

## 1. Preflight and immutable identities (`results/verify/verify_report.json`, SHA `5bd92c1a…`)

- **W107.**
  - The Worklog ends with its COMPLETE marker.
  - `evidence_manifest.json` SHA is `20fcb1b9…`, and declaration SHA is `aeb9ce71…`, both as published.
  - All 183 W107 manifest files are present and match.
- **W106.** The manifest SHA is `f3003fe5…`, and all 1,424 W106 manifest files match.
- **Known exception (in both manifests).** Each manifest's own stage log (`manifest.log` / `results/manifest.log`)
  was recorded with the empty-file hash. Stdout was still being redirected into it at hash time. These are the only
  differences, they are recorded explicitly, and they are not artifacts used here.
- **Exporter.** `src/blender/export.py` SHA is `521a63d9…`. Its AST-parsed arguments (`use_selection=False`,
  `object_types={'ARMATURE'}`, `bake_anim=True`) equal W107's.
- **Reused scripts.** The W106/W107 script SHAs equal those in the W107 declaration:
  - `w107_fbx_contract` `9af0a328…`;
  - `w107_fbx_blender` `985fc4bc…`;
  - `w106_action_blender` `8d33879c…`;
  - `w106_action_contract` `d7ff860f…`.

  `w107_fbx_ownership` equals its published final `a8fdebc5…`.
- **Inputs.** These are byte copies of the four W107 scratch inputs (the W106 N0 `.blend` files). Each SHA equals
  both the W107 manifest and the W107 declaration.
- **Identity.** In the probe, each real source has:
  - the declared `W91 <case> FIXED` Action, never R11 RUNTIME;
  - slot `OBroot`;
  - Action digest equal to W107's and to W106's declared digest;
  - 32 CONSTANT F-curves (768 / 768 / 6,368 keys), with integer key frames equal to the declaration;
  - 30 fps, frames 1–70 / 1–116 / 1–199;
  - 16 / 4 / 2 stored Actions.

## 2. Declaration (frozen before any W108 export, import or sample)

`results/declare/declaration.json`, SHA-256 **`0509a38c882386b35a2d895a8963570a8a3ebb875f2f475818297096b343112d`**,
`outcomes_read: false`. The probe and analysis refuse to run without this hash.

It binds:
- the script SHAs (`w108_fbx_subframe_contract` `231f61f4…`, `w108_fbx_subframe_blender` `001c4358…`,
  `w108_fbx_subframe` `b0ce3e21…`, plus the reused W106/W107 scripts and the exporter). All are unchanged at the end
  of the run.
- the add-on SHAs;
- the inputs;
- the explicit sample-time lists.

**Samples.** Every integer frame, plus +0.25 / +0.50 / +0.75 of every frame except the last:

| Case | Samples |
|---|---:|
| complete_control | 277 |
| arm_natural_most | 461 |
| real_clip | 793 |
| each synthetic control | 97 |

The W107 integer+half grid is a subset and is used for P0.

**Interval classes**, derived from source key times only. `[f, f+1)` is a *transition* if a source key lies at
`f+1`, and *quiet* otherwise.

| Case | Transition | Quiet |
|---|---:|---:|
| complete_control | 23 | 46 |
| arm_natural_most | 23 | 92 |
| real_clip | 198 | 0 |
| synthetic | 8 | 16 |

**Exports:**
- **V0:** the unchanged production `export_fbx`.
- **V1:** W107's isolated diagnostic, `bake_anim_simplify_factor=0.0`. It is not a production fix.

**Importer:** defaults, into `read_factory_settings(use_empty=True)`.

**Metrics:**
- **Primary:** the W106/W107 gate form. Max |coordinate| difference of world bone heads and tails over all source
  bones, reported:
  - by phase;
  - over all subframes;
  - over transition and quiet subframes separately.
- **Diagnostics:**
  - Euclidean max / p95 / median;
  - samples > 1 mm;
  - head-only error;
  - world bone rotation angle;
  - object rotation and translation;
  - armature-space tail error;
  - per-bone and per-time outliers.

**P1 layers:**
- source curves, plus Blender evaluation;
- the selected FBX stack's KeyTime grid and `KeyAttrFlags` interpolation bits, decoded by two parsers:
  - an **independent binary-FBX reader** written in the contract (no add-on code);
  - `io_scene_fbx.parse_fbx`.

  Flags count as reliable only if both parsers agree and `KeyAttrRefCount` covers every key.
- the imported Action;
- static inspection of the importer source.

**P2 (single intervention):** on a fresh default import of the V1 FBX:
1. `Action.copy()` of the selected imported Action only.
2. `keyframe_points.foreach_set('interpolation', CONSTANT)` on every key of every F-curve of the copy. No
   `fcurve.update()`, and no key insert, remove or move.
3. Assign the copy using the slot whose identifier equals the original's.

The intervention is NOT TESTABLE if any key field other than interpolation differs, or if the original changes.

**P3 (synthetic controls):** built on a scratch copy of the W106 synthetic N0 file (5 rotated bones, keys at frames
1, 4, …, 25). `SYN baseline` is duplicated with identical key times, values and slot. Only the duplicate's
interpolation is set:
- CONSTANT for `syn_step`;
- LINEAR for `syn_smooth`.

The duplicate is assigned and the file saved as a scratch copy. Each control then follows the same V0/V1/P2 path.

Validity requires:
- the construction differs from `SYN baseline` only in interpolation;
- the step source is held bitwise;
- the smooth source changes in every interval (> 1e-6 m).

If a control's V1 integer gate exceeds 1e-5 m, its gate is marked confounded and the contrast is evaluated on the
isolated head/rotation metric (CASE ≤ C).

Tolerances:

| Check | Tolerance |
|---|---|
| Determinism, P0, P2-vs-V1 integer frames | ≤ 1e-9 m |
| Source fidelity | bitwise |
| W107 metric reproduction | ≤ 1e-9 m |
| **Material** | P2 subframe max ≤ 0.5 × V1 (W107's "substantial" convention) |
| **Smooth harm** | P2 ≥ 2 × V1 |
| **Sufficiency** (reported, not gating) | P2 subframe ≤ 1 mm |
| Imported key time vs integer | exact |

**CASE rule** (`w108_fbx_subframe_contract.decide_case`):

| CASE | Condition |
|---|---|
| D | Preflight, source identity, P0, determinism, originals or synthetic-validity failure |
| A | On every real case: an evidenced boundary in {EXPORT_SERIALIZATION, IMPORT}, P2 testable with unchanged key data and integer motion, and P2 material. Plus both synthetic contrasts, unconfounded. |
| B | P2 testable on all three real cases but material on none |
| C | Otherwise |

**Boundary rule:**

| Evidence | Boundary |
|---|---|
| Source held, FBX non-CONSTANT | EXPORT_SERIALIZATION |
| FBX CONSTANT, import non-CONSTANT | IMPORT |
| Parsers disagree | UNRESOLVED |

## 3. P0 — W107 replay (`results/analysis/analysis_report.json`, SHA `81a59d3e…`)

| Check | complete_control | arm_natural_most | real_clip |
|---|---|---|---|
| Source samples vs W107 (common 139 / 231 / 397 times) | 0 (bitwise) | 0 | 0 |
| V0 / V1 imported samples vs W107 | 0 / 0 | 0 / 0 | 0 / 0 |
| V0 integer gate (W107) | 1.2898 mm (= W107) | 6.1038 mm (=) | 7.5123 mm (=) |
| V1 integer gate | 0.95 µm (=) | 1.31 µm (=) | 1.66 µm (=) |
| V0 = V1 half-frame max | 0.4173 m (=) | 0.9603 m (=) | 0.3542 m (=) |
| W107 npz SHAs unchanged | yes | yes | yes |

Every difference from W107 is exactly 0. V0's simplification loss (integer frames) and V1's interpolation loss
(subframes) stay separate. In quiet intervals, V0's subframe error is of the same size as its integer-frame
simplification error (1.13 / 5.93 mm quiet vs 1.29 / 6.10 mm integer), and V1's equals its µm integer error.

## 4. P1 — boundary tracing

| Layer | complete_control | arm_natural_most | real_clip | Evidence type |
|---|---|---|---|---|
| Source F-curves | CONSTANT, 24 keys/curve, integer frames | CONSTANT, 24 | CONSTANT, 199 | Action data |
| Source evaluated between keys | held: subframe = integer frame, max diff 0 | held, 0 | held, 0 | Blender evaluation |
| V1 FBX selected stack | 468 curves, 32,760 keys | 468, 54,288 | 468, 93,132 | Both parsers, identical summaries |
| … `KeyAttrFlags` | `0x6104` → **LINEAR** | `0x6104` → **LINEAR** | `0x6104` → **LINEAR** | Bit decode (CONSTANT 0x2 / LINEAR 0x4 / CUBIC 0x8) |
| … RefCount covers keys, values = times | yes | yes | yes | |
| … key grid | 70 frames (FBX 0–69), integer ±1 tick | 116 | 199 | |
| V0 FBX and all other takes | `0x6104` LINEAR | LINEAR | LINEAR | Both parsers agree on all 16 / 4 / 2 stacks |
| Default import | LINEAR, 70 keys/curve, key times exact integers | LINEAR, 116 | LINEAR, 199 | Action data |
| Importer source (`import_fbx.py` `e4e55c2e…`) | Reads only `KeyTime`/`KeyValueFloat`; never `KeyAttrFlags`; fills `interpolation` with LINEAR | same | same | Static inspection, not behavioural |
| Exporter source (`export_fbx_bin.py` `8fd324e4…`) | One literal `keyattr_flags = (1<<2 \| 1<<8 \| 1<<13 \| 1<<14,)` = 0x6104 for every curve, independent of the source | same | same | Static inspection |
| **Boundary (rule)** | **EXPORT_SERIALIZATION** | **EXPORT_SERIALIZATION** | **EXPORT_SERIALIZATION** | |

So the hold is encoded nowhere in the FBX. Each source step becomes a one-frame LINEAR ramp between two integer-frame
keys.

Subframe errors against the source (max abs coordinate of world heads/tails):

| Case | Path | Integer | +0.25 | +0.50 | +0.75 | Transition subframes | Quiet subframes | Subframe samples > 1 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| complete_control | V0 | 1.290 mm | 216.7 mm | 417.3 mm | 594.2 mm | 594.2 mm | 1.128 mm | 71 / 207 |
| | V1 | 0.95 µm | 216.7 mm | 417.3 mm | 594.2 mm | 594.2 mm | 0.95 µm | 69 / 207 |
| arm_natural_most | V0 | 6.104 mm | 533.5 mm | 960.3 mm | 1071.0 mm | 1071.0 mm | 5.931 mm | 126 / 345 |
| | V1 | 1.31 µm | 533.5 mm | 960.3 mm | 1071.0 mm | 1071.0 mm | 1.31 µm | 69 / 345 |
| real_clip | V0 | 7.512 mm | 187.1 mm | 354.2 mm | 432.6 mm | 432.6 mm | — (no quiet) | 594 / 594 |
| | V1 | 1.66 µm | 187.1 mm | 354.2 mm | 432.6 mm | 432.6 mm | — | 594 / 594 |

S2 signature (predeclared, descriptive):
- **V1 error lies entirely in transition intervals.** In the 3DPW windows, exactly the 69 = 23 × 3 transition
  subframes exceed 1 mm. Quiet subframes stay at µm.
- **The error grows monotonically with phase** (+0.75 > +0.50 > +0.25, as the ramp moves further toward the next
  pose). That is the signature of a one-frame linear ramp replacing a step at `f+1`.

Other V1 subframe diagnostics:

| Measure | complete_control | arm_natural_most | real_clip |
|---|---|---|---|
| World bone rotation max (p95) | 56.8° (12.7°) | 175.7° (6.6°) | 88.0° (4.4°) |
| Armature-object rotation max | 12.3° | 17.0° | 18.1° |
| Object translation | 0 | 0 | 0 |
| Armature-space tail max | 0.69 m | 1.35 m | 0.60 m |
| Worst bones | foot_r / foot_l | index_02_l / middle_02_l | middle_02_l / ring_02_l |

The real clip keys every frame, so every interval is a transition. Its median per-sample V1 subframe error is
19.9 mm and its p95 is 75 mm.

## 5. P2 — single scratch intervention (post-import CONSTANT on a copied V1 Action)

| Check | complete_control | arm_natural_most | real_clip |
|---|---|---|---|
| Status | TESTED | TESTED | TESTED |
| Copy vs original, before change | identical (all fields) | identical | identical |
| Fields changed by P2 | `interpolation` only | `interpolation` only | `interpolation` only |
| Unchanged | co, handles, handle types, easing, back, amplitude, period, key type, extrapolation, modifiers, group, key counts | same | same |
| Copy channel digest = original | yes | yes | yes |
| Original imported Action unchanged | yes | yes | yes |
| Slot | `OBSlot` (same identifier) | `OBSlot` | `OBSlot` |
| Imported key times vs integers | max dev 0.0 | 0.0 | 0.0 |
| Fresh import (P2_pre) vs V1 samples | 0 | 0 | 0 |
| **P2 integer samples vs V1** (all arrays) | **0** | **0** | **0** |
| **Subframe max: V1 → P2** | **0.594 m → 0.95 µm** | **1.071 m → 1.31 µm** | **0.433 m → 1.66 µm** |
| Ratio P2 / V1 | 1.6e-6 | 1.2e-6 | 3.8e-6 |
| P2 at +0.25 / +0.50 / +0.75 | 0.95 µm each | 1.31 µm each | 1.66 µm each |
| P2 rotation max / object rotation max | 2.2e-4° / 3.1e-5° | 1.0e-3° / 2.7e-5° | 7.3e-4° / 6.2e-5° |
| P2 armature-space tail max | 0.94 µm | 1.41 µm | 1.42 µm |
| Material (≤ 0.5 × V1) / sufficient (≤ 1 mm) | yes / yes | yes / yes | yes / yes |
| Fresh repeat (P2 and P2_pre) | 0 | 0 | 0 |

The residual equals the V1 integer-frame float32 error at every phase. Two conclusions follow:
- the existing V1 imported key values are sufficient to reproduce the source's continuous-time held motion once read
  with hold semantics;
- the subframe loss comes from interpolation semantics, not from missing pose data.

## 6. P3 — synthetic controls (separately labelled; W106 synthetic rig, 30 fps, frames 1–25)

| Control | Source | Construction valid | V0 / V1 integer gate | V1 subframe max | P2 subframe max | P2 / V1 | Prediction | Result |
|---|---|---|---|---:|---:|---:|---|---|
| `syn_step` | CONSTANT (held, bitwise) | yes | 0.80 / 0.77 µm | 245.6 mm (24/72 > 1 mm) | 0.77 µm | 3.2e-6 | P2 ≤ 0.5 × V1 | **holds** |
| `syn_smooth` | LINEAR (min interval change 56 mm) | yes | 0.91 / 0.80 µm | 0.512 mm (0/72 > 1 mm) | 86.0 mm (72/72 > 1 mm) | 168 | P2 ≥ 2 × V1 | **holds (harm)** |

Further control details:
- **No confound.** Both V1 integer gates are < 1e-5 m, so the W107 V3 rest-geometry confound did not occur. The
  isolated head/rotation metrics agree:

  | Control | Head error V1 → P2 | Rotation V1 → P2 |
  |---|---|---|
  | step | 0.193 m → 0.77 µm | 71° → 1.7e-4° |
  | smooth | 76 µm → 65 mm | 0.13° → 16.3° |

- **Smooth control, phase growth.** The P2 harm grows with phase: 28.8 / 57.6 / 86.0 mm at +0.25 / +0.50 / +0.75.
  Forcing CONSTANT throws away real in-between motion.
- **Smooth control, LINEAR residual.** The smooth control's V1 residual (0.51 mm, worst bone `sibling` at frame
  10.5) is the LINEAR-to-LINEAR round-trip error. It is probably the Euler-space interpolation of baked frames
  versus the source's quaternion-component interpolation, but that was not isolated.
- **Boundary.** `syn_step` shows the same EXPORT_SERIALIZATION boundary (0x6104 LINEAR). For `syn_smooth` the rule
  returns NOT_APPLICABLE, as declared, because its source is not held.

## 7. CASE and its limits

**CASE A** (`decide_case`): boundary evidenced on all three real cases, P2 material with unchanged integer motion and
key data on all three, both synthetic contrasts hold unconfounded, P0 bitwise, repeat bitwise, originals unchanged.

What is proven:
- **Export side.** The default Blender FBX export path, and W107's V1 diagnostic, serialize held (CONSTANT) source
  motion as LINEAR-flagged per-frame keys. The hold is absent from the FBX file itself. This is direct encoding
  evidence from two parsers, and the exporter's flag is a source-independent literal.
- **Key data.** The imported V1 key data is sufficient; only its interpolation semantics are wrong for these
  Actions.

What remains ambiguous:
- **Importer.** The importer ignores `KeyAttrFlags` and hard-codes LINEAR per its source, but this was not exercised
  with a CONSTANT-flagged file. Whether restoring the flag at export alone would round-trip in Blender is therefore
  untested, and the static evidence says it would not.
- **Other consumers.** How other FBX consumers interpret 0x6104 versus a CONSTANT flag was not tested.
- **V0 scope.** P2's sufficiency is shown only for V1 (key on every frame) and integer-frame source keys. With V0's
  simplified keys, a CONSTANT reading would hold wrong values across dropped frames. That was not run.

Residual error under P2: 0.95 / 1.31 / 1.66 µm, at float32 forward-kinematics level.

## 8. Negative findings and unresolved items (explicitly not addressed)

- **FBX file bytes are not reproducible** run to run: the SHA always differs and the size sometimes does, as in
  W107. All imported samples are bitwise reproducible.
- **FBX KeyTimes are not exact tick multiples** (post-hoc, `results/posthoc/posthoc.json`, SHA `25601e20…`):
  - 2,808 / 4,212 / 7,956 of the V1 selected-stack keys are exactly −1 tick (6.5e-10 frame) off the 30 fps grid. All
    others are exact.
  - The FBX time origin is the scene start: frame 1 maps to time 0.
  - Import maps every key back to an exact integer frame, so this has no measured effect.
- **Unchanged and unresolved:**
  - `frame_end` 250 after import;
  - multi-take export (16 / 4 / 2 stacks);
  - the real clip's native 50 fps keys in a 30 fps scene;
  - V1's file-size and key cost;
  - export policy for non-Blender consumers;
  - user-facing authoring of hold versus continuous motion.

  P2 forced CONSTANT on all channels. A mixed hold/continuous Action would need per-curve semantics, which this
  evidence does not address.

## 9. Tests and execution record

| Run | Result | Output |
|---|---|---|
| Focused tests before the declaration | 11 passed, 2 stored-output tests skipped | `w108_tests_predeclare.xml` |
| Focused tests, final | **13 passed** | `w108_tests_final.xml` |
| Regression (W106/W107 Action/FBX contracts + Blender executor) | **38 passed** | `regression_w106_w107_blender_executor.xml` |

The focused tests (`tests/test_w108_fbx_subframe.py`) cover:
- the quarter-frame grid and its W107 subset;
- the transition/quiet classes;
- `KeyAttrFlags` decoding, including the exporter's literal;
- the independent FBX reader, on a synthetic binary FBX written in the test and on a real W107 FBX;
- the boundary rule;
- phase and class metrics;
- the CASE mapping, including D precedence, B, C on mixed/unreliable/confounded/integer-broken inputs, and A for an
  IMPORT boundary;
- code guards: no exporter patch, no key insert/remove/update, a single `foreach_set('interpolation')`, no in-place
  save, no torch;
- the stored declaration/analysis hash consistency.

The regression files were `tests/test_w107_fbx_ownership.py`, `tests/test_w106_action_editability.py` and
`tests/test_blender_executor.py`. Broad production regression was not run: no shared or tracked file changed
(`shared_tracked_changes` = []).

Execution record:
- **Pre-declaration smoke checks** (no export, import or W108 sampling):
  - the key-data, interpolation and slot helpers were run on a throw-away copy of the W106 synthetic input;
  - the add-on-parser adapter was run on W107's `complete_control/fbx/V1.fbx`, printing only parse status and parser
    agreement. Its flag values were not printed.

  The throw-away directory was deleted. The expected LINEAR flag in the declared S1 signature was informed by reading
  the add-on source before the declaration, and the declaration records that the importer layer is static
  inspection.
- **Execution failures: none.** Both probes exited 0 with no FAILED or NOT_TESTABLE record. The wall-clock timer
  failed (`bc` missing in Git Bash), so runtimes come from output timestamps.
- **Post-declaration code.** Only the new `scripts/w108_fbx_subframe_posthoc.py` (`a9aedf93…`), which is labelled
  post-hoc and descriptive. Every declared script SHA is unchanged.

Not executed:
- an importer behavioural flag probe (a second intervention; static inspection only);
- P2 on V0;
- per-channel (object vs pose) P2 split;
- Euler-vs-quaternion isolation;
- renders (the numbers were unambiguous);
- non-Blender consumers;
- other rigs or Actions;
- LabServer.

## 10. Reproduction (Git Bash on Main, repo root)

```
PY=/c/Projects/AnimCV/.venv_build_full/Scripts/python.exe; BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
O=C:/Users/dna10/animcv-w108; R=$O/results; W6=C:/Users/dna10/animcv-w106; W7=C:/Users/dna10/animcv-w107
H=730a5dfd15d93ec008fb8017448a89424b88f99e; S=0509a38c882386b35a2d895a8963570a8a3ebb875f2f475818297096b343112d
B=("$BL" -b --factory-startup -t 2 --python-exit-code 1 -P scripts/w108_fbx_subframe_blender.py --)
mkdir -p $R
$PY -X utf8 scripts/w108_fbx_subframe.py verify --base $O --w106 $W6 --w107 $W7 --lock C:/Users/dna10/animcv-automation/state/active_batch.json --blender "$BL" --starting-head $H --branch arch/single_frame_first
"${B[@]}" options --out $R/options.json
$PY -X utf8 scripts/w108_fbx_subframe.py declare --base $O --w106 $W6 --w107 $W7 --verify $R/verify/verify_report.json --options $R/options.json --blender "$BL" --starting-head $H
"${B[@]}" probe --declaration $R/declare/declaration.json --declaration-sha $S --out-dir $R/probe
"${B[@]}" probe --declaration $R/declare/declaration.json --declaration-sha $S --out-dir $R/probe_repeat
$PY -X utf8 scripts/w108_fbx_subframe.py analyze --declaration $R/declare/declaration.json --declaration-sha $S --probe-dir $R/probe --repeat-dir $R/probe_repeat --w106 $W6 --w107 $W7 --verify $R/verify/verify_report.json --out-dir $R/analysis
$PY -X utf8 scripts/w108_fbx_subframe_posthoc.py --probe-dir $R/probe --declaration $R/declare/declaration.json --addon-dir "C:/Program Files/Blender Foundation/Blender 5.1/5.1/scripts/addons_core/io_scene_fbx" --out-dir $R/posthoc
$PY -X utf8 scripts/w108_fbx_subframe.py manifest --base $O --w106 $W6 --w107 $W7 --starting-head $H --declaration-sha $S
$PY -X utf8 -m pytest tests/test_w108_fbx_subframe.py -q
```

- Each output-writing stage refuses an existing directory.
- On a fresh base, the declaration hash changes only through recorded paths and SHAs.
- The verify stage requires a Batch-108 RUNNING lock and a clean tree.

Runtimes:

| Stage | Runtime |
|---|---|
| options | < 1 s |
| probe | ≈ 18.4 s |
| repeat probe | ≈ 18.0 s |
| verify / declare / analyze / post-hoc / manifest | a few seconds each |
| tests | 0.5 s |
| regression | 0.2 s |

Outputs are local in `C:\Users\dna10\animcv-w108\` (81 MB, 138 files) and are not committed. They include scratch
`.blend`/FBX files, npz samples, meta JSON, reports and logs.
- `evidence_manifest.json`, SHA **`bffcd9ca551744d4ebd4e5087f42ea99a0ad6c8e29395c0d0895ddd8c8768375`**.
- It records 137 files and `all_originals_unchanged: true` (W106 and W107 manifests and every file).
- Like W106/W107, it hashed its own stdout capture `manifest_stdout.txt` as empty.

Other hashes:

| File | SHA |
|---|---|
| `probe/probe_report.json` | `44892952…` |
| `probe_repeat/probe_report.json` | `925e17be…` |

## 11. Changed files, resources and environment

New files only:

| File | Role | SHA |
|---|---|---|
| `scripts/w108_fbx_subframe_contract.py` | Pure contract: cases/controls, sample grid, interval classes, metrics, independent FBX reader and flag decoder, P2 definition, boundary rule, tolerances, CASE mapping | `231f61f4…` |
| `scripts/w108_fbx_subframe_blender.py` | Blender stages `options` (W107's, reused) and `probe` (source / V0 / V1 / FBX decode / P2 / P3 build) | `001c4358…` |
| `scripts/w108_fbx_subframe.py` | Host stages verify / declare / analyze / manifest | `b0ce3e21…` |
| `scripts/w108_fbx_subframe_posthoc.py` | Post-hoc descriptive tick deviation and add-on static inspection | `a9aedf93…` |
| `tests/test_w108_fbx_subframe.py` | Focused tests | `8d86bf97…` |
| this Worklog | | |

Nothing else changed:
- no `src/` file, exporter or importer setting, rig, Action original, W106/W107 artifact or historical Worklog;
- not `docs/next_batch.md`;
- no automation file.

Resources: Main only and CPU only. No rendering, GPU, inference, package change or network. The private real-clip
frames were not read.

Final source revision: the commit adding this Worklog on `arch/single_frame_first`, directly after `730a5df`. The
COMPLETE marker is added in a separate final commit.

## 12. Limits

- **Scope.** Three reused real Actions on one BaseRig, plus two synthetic controls on the W106 synthetic rig.
  Blender 5.1.0 / FBX add-on 5.15.0, round-tripped only through Blender's own importer.
- **Sampling.** Quarter frames only; between-quarter behaviour is inferred from the step/ramp structure, not sampled.
- **What "boundary" means.** It is direct encoding evidence of where the hold first disappears, not a claim that the
  importer is correct. The importer's flag handling is static-inspection evidence only.
- **What P2 shows.** P2 shows only that the V1 imported keys are representationally sufficient for these
  integer-keyed, fully held Actions. It is not an exporter or importer fix, not evidence of FBX interoperability, and
  not a policy. The smooth control shows the harm of applying it to continuous motion.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 108
