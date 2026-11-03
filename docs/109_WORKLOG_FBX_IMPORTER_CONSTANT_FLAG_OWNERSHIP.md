# Worklog — FBX Importer CONSTANT-Flag Behavioural Ownership (2026-11-03)

## Human Assessment

- **SUCCESS:** The importer-side question W108 left open was tested behaviourally, using valid, independently
  verified, CONSTANT-flagged scratch FBX files.
  - **Inputs.** Each real case uses the frozen W107 V1 FBX; the W108 synthetic held-step control uses its W108 V1 FBX.
  - **The edit.** For each input, two predeclared scratch variants were written. Each is a byte-exact in-place rewrite
    of *only* the `KeyAttrFlags` value of the selected take's AnimationCurves:
    - `CONSTANT-standard`: `0x6104 → 0x6002`;
    - `CONSTANT-next`: `0x6104 → 0x6102`.

    File length was unchanged. Every differing byte lies inside the targeted 4-byte flag values. With the targets
    masked, the canonical node trees are identical to the control. Two parsers agree digest-for-digest: this repo's
    independent reader and `io_scene_fbx.parse_fbx`.
  - **The result.** The unchanged default Blender 5.1 importer produced **identical output from the variants and from
    the unchanged control**:
    - every imported key field (time, value, interpolation, handles, …) is bitwise equal;
    - the F-curve interpolation census is still 100 % LINEAR (object and pose channels);
    - every other imported Action is unchanged;
    - all sampled arrays differ from the control by exactly **0.0** at every integer and quarter frame.

    So the subframe error is unchanged at 0.594 / 1.071 / 0.433 m on the three real Actions and 0.246 m on the step
    control. Neither variant follows its declared CONSTANT function on any case.
  - **Controls.** The smooth LINEAR control was left LINEAR and stays smooth (0.51 mm). P0 replays W107 and W108
    bitwise. The fresh repeat produced identical FBX bytes, trees, key data and samples.

  **CASE A — importer discard supported.**
- **FAILURE / NOT FIXED:** Nothing was repaired, and nothing becomes deployable from this result.
  - The round trip still loses the hold. Restoring a valid CONSTANT flag in the FBX file alone does not restore it
    through this Blender importer.
  - W108's export-side loss (the exporter writes `0x6104` LINEAR for every curve) remains separate and unrepaired.
  - Every other W107/W108 defect persists unchanged:
    - `frame_end` 250;
    - multi-take export;
    - the 50 → 30 fps real-clip timebase;
    - V1's file-size and key cost.
- **UNCERTAINTY:**
  - **Format validity.** "Valid CONSTANT encoding" means valid under the public FBX SDK bit enumerations, as decoded
    by two parsers. No Autodesk FBX SDK or other third-party consumer was available on Main to confirm that it reads
    `0x6002` / `0x6102` as held keys.
  - **Retained tangent bits.** The two variants keep the exporter's tangent bits `0x6000`. These are generic
    time-independent / clamp-progressive bits, inert for non-cubic keys per the SDK enumeration. That inertness is not
    checked against an SDK reader.
  - **Scope.** Only Blender 5.1.0 / add-on 5.15.0 was tested. Other Blender versions or consumers may differ.
  - **Mechanism.** Behaviourally, the flag value has no measurable effect on any imported datum. The *mechanism*
    attribution (`import_fbx.py` never reads `KeyAttrFlags` and fills LINEAR) is still code inspection. It is fully
    consistent with the behaviour, but was not instrumented.
- **ALTERNATIVE EXPLANATIONS (status):**

  | Explanation | Status |
  |---|---|
  | Wrong CONSTANT-standard vs CONSTANT-next encoding | Both encodings tested; both give the identical null effect. |
  | Malformed scratch FBX | Excluded as far as two parsers can show. Byte diff is confined to the target flag values; masked trees are identical; the importer accepted every file (`FINISHED`). |
  | Altered key times / values / takes | Excluded. Masked trees are identical, selected-stack key counts and RefCount are unchanged, and imported key data is bitwise equal. |
  | Mismatched Action / slot | Excluded. The name-matched imported Action and `OBSlot` are identical in every import; rest pose and Action inventory are equal to the control. |
  | Importer-side resampling | Not separable from flag discard, and not needed: the output is bitwise identical to the control. |
  | Timebase / scene setting | Unchanged in every import; samples bitwise equal. |
  | Synthetic vs real semantics differ | Same null effect on all three real Actions and on the step control. They are reported separately and never pooled. |

## Direction / Purpose / Central Intent

Probe Blender's FBX importer with an independently validated, correctly CONSTANT-flagged scratch FBX. Animation keys
and source rig semantics stay unchanged. The aim is to separate export-serialization loss (W108) from importer
flag-handling failure.

Central question: *when the same W107 V1 per-frame key data is encoded as FBX CONSTANT instead of LINEAR, does the
unchanged Blender 5.1 importer reconstruct the held subframe motion?*

**H109:** the importer still imports LINEAR, with essentially the same subframe error. If the importer instead
preserved the hold, H109 would be falsified.

The batch was evidence only:
- no exporter, importer, Action, rig, FPS, timebase or retargeting change;
- no post-import correction;
- no training, inference, rendering, GPU, network or LabServer use.

## Scope and starting state

- **Repository.**
  - Branch `arch/single_frame_first`.
  - Starting HEAD `b5b46c3d90071f932d22bfca8ced9abd8877d5cb`.
  - Instruction blob `96970f4e23e3fe88b9c07a5dc74eab04d7f507ab`.
  - Both were verified before any work. After `git fetch`, `origin` equalled HEAD and the tree was clean.
- **Worker.** The lock was Batch 109 / RUNNING, with `start_commit` = HEAD (poller 30556, executor 34748). No other
  Blender process was running.
- **Environment.** Everything ran on **Main**:
  - Python 3.11, `C:\Projects\AnimCV\.venv_build_full`;
  - Blender **5.1.0**, build `adfe2921d5f3`, `blender.exe` SHA `f9d2e744…`, always `-b --factory-startup -t 2`;
  - FBX add-on **5.15.0**. All ten source SHAs are equal to the W108 declaration (`import_fbx.py` `e4e55c2e…`,
    `parse_fbx.py` `96fe3f11…`). `options.json` is byte-identical to W108's (`7d1a5472…`).

Regimes (reused, non-holdout data):

| Case | Regime | Role |
|---|---|---|
| `complete_control`, `arm_natural_most` | `benchmark_detector_observation` (3DPW) | real, mutated |
| `real_clip` | `real_animcv_observation` (GT-free historical clip) | real, mutated |
| `syn_step` | synthetic (W108 held-step control, W106 synthetic rig) | mutated; never pooled with real |
| `syn_smooth` | synthetic (W108 smooth LINEAR control) | negative control; **never mutated** |

## 1. Preflight and provenance (`results/verify/verify_report.json`, SHA `e979cf4b…`) — AGENT-RUN EVIDENCE

- **W108.**
  - The Worklog ends with its COMPLETE marker.
  - `evidence_manifest.json` = `bffcd9ca…8768375` and `declaration.json` = `0509a38c…343112d`, both as published.
  - All 137 W108 manifest files match. The only allowed exception is its self-hashed `manifest_stdout.txt`, recorded
    in W108.
- **W107 and W106.**
  - W107 manifest `20fcb1b9…`, 183 files intact.
  - W106 manifest `f3003fe5…`, 1,424 files intact.
  - These use the same recorded self-log exceptions as W108.
- **Exporter and reused scripts.**
  - `src/blender/export.py` SHA is `521a63d9…`, unchanged.
  - The reused W106/W107/W108 scripts are equal to the W108 declaration's SHAs.
- **Inputs.** Byte copies were made into `results/inputs/`, and each SHA equals its W107/W108 manifest entry:
  - real cases: W107 `results/probe/<case>/fbx/V1.fbx` (the frozen W107 V1) and W107 `results/inputs/<case>_n0.blend`;
  - synthetic controls: W108 `results/probe/<ctrl>/fbx/V1.fbx` and `input.blend`.

## 2. Declaration (frozen before any W109 mutation, import or sample)

`results/declare/declaration.json`, SHA-256 **`b6bf24049e4be1270c3b3d64ff2d119188b87a8c89b79ac8ac951eac2b986f73`**,
`outcomes_read: false`. The encode, probe and analysis stages refuse to run without this hash.

**Script SHAs.** The declaration binds the W109 script SHAs, and all are unchanged at the end of the run:
- contract `b3793c48…`;
- Blender stage `acd26dca…`;
- host `0949e9b2…`.

It also binds the reused scripts, the exporter, the environment and the inputs.

**Selected-stack identifiers.** These were read from the *unmodified* inputs only:

| Case | Selected stack | Target curves / all curves | Stacks in file | Selected-stack keys | Original flags |
|---|---|---:|---:|---:|---|
| complete_control | `root\|W91 complete_control FIXED` | 468 / 7,488 | 16 | 32,760 | all `0x6104` |
| arm_natural_most | `root\|W91 arm_natural_most FIXED` | 468 / 1,872 | 4 | 54,288 | all `0x6104` |
| real_clip | `root\|W91 real_clip FIXED` | 468 / 936 | 2 | 93,132 | all `0x6104` |
| syn_step | `root\|W108 SYN step CONSTANT` | 54 / 108 | 2 | 1,350 | all `0x6104` |
| syn_smooth | `root\|W108 SYN smooth LINEAR` | (not mutated) / 108 | 2 | 1,350 | all `0x6104` |

- **Curve selection.** Target curves are those connected curve → curve node → layer → stack to the uniquely
  name-matched stack. The declaration stores their uid list and its SHA.
- **Flag storage.** Every target `KeyAttrFlags` is an uncompressed 1-element int32 array. The exporter compresses
  only arrays over 128 bytes, so an in-place byte edit is possible.

**Intervention.** Exactly two encodings were tested, not a sweep. The derivation rule:
`new = (0x6104 & ~(0xE | 0x100)) | 0x2 [| 0x100 for next]`.

| Variant | Flag | Decoded | Declared expected function |
|---|---|---|---|
| CONSTANT-standard | `0x6002` | CONSTANT, standard; other bits `0x6000` | value at t = key at or before t (`floor`) |
| CONSTANT-next | `0x6102` | CONSTANT, `eConstantNext`; other bits `0x6000` | value at t ∈ (k, k+1) = key at k+1 (`ceil`) |

**Bit aliasing.** Bit `0x100` is `eTangentAuto` under LINEAR/CUBIC and `eConstantNext` under CONSTANT. The exporter
literal `0x6104` sets it as "tangent auto". A naive edit that changes only the interpolation field of `0x6104`
therefore yields `0x6102`, which is CONSTANT-*next*, not standard. CONSTANT-standard must clear the bit. A focused test
pins this.

**Declared expected functions.** These are built from the *control import's* integer samples, which are the FBX key
values as Blender evaluates them. So they isolate interpolation semantics from key data.

**Classification** (`classify`):

| Outcome | Condition |
|---|---|
| `LINEAR_UNCHANGED` | Import interpolation {LINEAR}, every key field equal to the control, every sample within 1e-9 m of the control |
| `FOLLOWS_DECLARED` | Key data (except interpolation) and integer motion unchanged, and every subframe within 1 mm of the declared function |
| `NOT_DISCRIMINABLE` | Control within 1 cm of the declared function |
| `OTHER` | Anything else |

**CASE rule** (`decide_case`, frozen):

| CASE | Condition |
|---|---|
| D | Preflight, encoding or unrelated-semantics preservation, source identity, matching, P0, determinism or originals failure |
| A | Neither variant follows its declared function on any of the 3 real cases + step, with key data and integer motion unchanged and a valid smooth control |
| B | CONSTANT-standard follows the held function on all 3 real cases + step, with a valid smooth control |
| C | Otherwise |

**Tolerances:**

| Check | Tolerance |
|---|---|
| Determinism / P0 / integer frames vs control | ≤ 1e-9 m |
| Source fidelity | bitwise |
| W108 metric reproduction | ≤ 1e-9 m |
| Follows declared function | ≤ 1 mm |
| Discriminable | > 1 cm |
| Smooth sufficiency | ≤ 1 mm |

**Samples.** W108's explicit lists: integer frames plus +0.25 / +0.50 / +0.75.

| Case | Samples |
|---|---:|
| complete_control | 277 |
| arm_natural_most | 461 |
| real_clip | 793 |
| each synthetic control | 97 |

Interval classes are W108's. Resource cap: options + 2 probes, ≤ 900 s each, ≤ 2 GB, no GPU or network.

## 3. Encoding and independent validity (`results/encode/encode_report.json`, SHA `ce4c7550…`) — AGENT-RUN EVIDENCE

Every case has `control.fbx` byte-equal to its input. The **no-op identity** holds: the patcher run with the source
value reproduces the control bytes exactly.

| Case | Variant | Curves patched | Bytes differing (all inside target value spans) | Length | Target flags after | Non-target flags | Selected-stack interp (W108 summary) / other stacks |
|---|---|---:|---:|---|---|---|---|
| complete_control | standard | 468 | 936 | 9,509,612 (=) | `0x6002` | `0x6104` | CONSTANT / LINEAR |
| | next | 468 | 468 | = | `0x6102` | `0x6104` | CONSTANT / LINEAR |
| arm_natural_most | standard / next | 468 / 468 | 936 / 468 | 2,487,276 (=) | `0x6002` / `0x6102` | `0x6104` | CONSTANT / LINEAR |
| real_clip | standard / next | 468 / 468 | 936 / 468 | 1,809,564 (=) | `0x6002` / `0x6102` | `0x6104` | CONSTANT / LINEAR |
| syn_step | standard / next | 54 / 54 | 108 / 54 | 102,380 (=) | `0x6002` / `0x6102` | `0x6104` | CONSTANT / LINEAR |

Two bytes per curve change for the standard variant (`0x04→0x02`, `0x61→0x60`) and one for next (`0x04→0x02`).

Further validity checks:
- **Masked tree digest.** For each variant, the canonical tree digest with only the target flags masked equals the
  control's. It covers every node, property type and value, and child order, including all key times, values,
  KeyAttrDataFloat, RefCount, takes, objects and connections.
- **Full tree digest.** The full digest differs from the control's, as expected.
- **Key counts.** Selected-stack key counts are unchanged, and RefCount covers every key.

**Parser agreement.** Checked inside each probe, AGENT-RUN:
- `io_scene_fbx.parse_fbx` gives the same full digest, masked digest, selected stack, target-uid SHA, curve counts and
  decoded flag sets as the independent reader, for all 13 files per run: 5 controls plus 8 variants (2 on each of
  the four target cases). The no-op patch is checked in memory only; no `noop` file is written.
- The add-on masked digest of every variant equals that of its control.
- A pre-declaration smoke check confirmed the same agreement on one unmodified W107 V1 file (see §8).

## 4. P0 — baseline identity (`results/analysis/analysis_report.json`, SHA `44417feb…`)

| Check | complete_control | arm_natural_most | real_clip | syn_step | syn_smooth |
|---|---|---|---|---|---|
| Fresh source samples vs W108 `source.npz` | 0 (bitwise) | 0 | 0 | 0 | 0 |
| Source Action digest / slot / key frames / interp | = declared (`OBroot`, CONSTANT) | = | = | = (CONSTANT) | = (LINEAR) |
| Source held by evaluation | yes | yes | yes | yes | no (min interval change 56 mm) |
| Control import vs W108 `V1.npz` (all quarter times) | 0 | 0 | 0 | 0 | 0 |
| Control import vs W107 `V1.npz` (integer + half) | 0 | 0 | 0 | — | — |
| Control subframe max − W108 value | 0 | 0 | 0 | 0 | 0 |
| Control integer max − W108 value | 0 | 0 | 0 | 0 | 0 |
| Control import interpolation | LINEAR | LINEAR | LINEAR | LINEAR | LINEAR |
| Imported Action / slot | `root\|root\|W91 … FIXED` / `OBSlot` | = | = | `root\|root\|W108 SYN step CONSTANT` / `OBSlot` | = |

## 5. P2 — behavioural import (default importer, no post-import correction) — AGENT-RUN EVIDENCE

### 5.1 Imported F-curve census and key data (identical for control, CONSTANT-standard and CONSTANT-next)

| Case | Object F-curves | Pose F-curves | Keys | Interpolation | Other imported Actions (all LINEAR, key data = control) |
|---|---|---|---:|---|---:|
| complete_control | 9 LINEAR | 510 LINEAR | 36,330 | LINEAR only | 15 |
| arm_natural_most | 9 LINEAR | 510 LINEAR | 60,204 | LINEAR only | 3 |
| real_clip | 9 LINEAR | 510 LINEAR | 103,281 | LINEAR only | 1 |
| syn_step | 9 LINEAR | 50 LINEAR | 1,475 | LINEAR only | 1 |

The key data of every variant import was compared with the control import.
- **Differing fields: none.** Compared fields: `co` (times and values), handles, interpolation, handle types,
  easing, key type, back / amplitude / period, extrapolation, modifiers, group and key counts.
- Imported key times are exact integers.
- Rest pose and Action inventory are identical.

**Flag read:** no. The importer's F-curves carry no trace of the flag change, in object or pose channels.

### 5.2 Evaluated motion

The variant-vs-control delta is the max abs difference over every sampled array: local quat/loc/scale, world
head/tail/rot and object matrix.

| Case | Variant | Δ vs control (all samples) | Integer vs control | Subframe max vs source (+0.25 / +0.50 / +0.75) | Transition / quiet | Subframes > 1 mm | Declared-function error | Outcome |
|---|---|---:|---:|---|---|---|---:|---|
| complete_control | standard | **0.0** | 0.0 | 0.594 m (0.217 / 0.417 / 0.594) | 0.594 m / 0.95 µm | 69 / 207 | 0.792 m (floor) | LINEAR_UNCHANGED |
| | next | **0.0** | 0.0 | 0.594 m (same) | same | 69 / 207 | 0.726 m (ceil) | LINEAR_UNCHANGED |
| arm_natural_most | standard | **0.0** | 0.0 | 1.071 m (0.533 / 0.960 / 1.071) | 1.071 m / 1.31 µm | 69 / 345 | 1.158 m | LINEAR_UNCHANGED |
| | next | **0.0** | 0.0 | 1.071 m | same | 69 / 345 | 1.261 m | LINEAR_UNCHANGED |
| real_clip | standard | **0.0** | 0.0 | 0.433 m (0.187 / 0.354 / 0.433) | 0.433 m / — (no quiet) | 594 / 594 | 0.498 m | LINEAR_UNCHANGED |
| | next | **0.0** | 0.0 | 0.433 m | same | 594 / 594 | 0.474 m | LINEAR_UNCHANGED |
| syn_step | standard | **0.0** | 0.0 | 0.246 m (0.081 / 0.159 / 0.246) | 0.246 m / 0.77 µm | 24 / 72 | 0.412 m | LINEAR_UNCHANGED |
| | next | **0.0** | 0.0 | 0.246 m | same | 24 / 72 | 0.482 m | LINEAR_UNCHANGED |

Notes on the table:
- **Object vs pose.** Armature-object rotation error at subframes is unchanged from the control: 12.3° / 17.0° / 18.1°
  on the real cases and 9e-6° on the synthetic rig, whose object is static. Armature-space tail error is also
  unchanged: 0.69 / 1.35 / 0.60 / 0.28 m.

  Neither the object channels nor the pose channels respond to the flag. Both still carry the W108 LINEAR-ramp loss.
- **Why the declared-function error exceeds the source-referenced error.** The default import adds importer-generated
  `_end` leaf bones: 66 bones vs 51 source bones (real), 7 vs 5 (synthetic). The declared-function metric compares the
  variant import with the control import over *all imported* bones. The source-referenced W108 metric covers source
  bones only.

  This is a metric-scope difference, recorded here as observed. It does not affect any outcome: every cell was decided
  by Δ vs control = 0.0, and every declared-function error is ≥ 0.41 m against a 1 mm threshold.
- **Discriminability.** The control differs from both declared functions by ≥ 0.41 m on every target case, far above
  1 cm. A held import would therefore have been unmistakable.

## 6. P3 — synthetic discriminators (separately labelled, never pooled)

- **`syn_step` (CONSTANT source, held bitwise).**
  - Expected: CONSTANT-standard ⇒ held previous key; CONSTANT-next ⇒ next key.
  - Observed: both variants import as LINEAR, bitwise equal to the control. Subframe max 0.246 m, the same as W108 V1.
  - This matches the real Actions: no synthetic/real contradiction.
- **`syn_smooth` negative control (never mutated).**
  - Its LINEAR flags were left in place, and it imports as LINEAR.
  - It is bitwise equal to W108 V1; subframe max vs source is 0.512 mm (≤ 1 mm), and its source changes in every
    interval (min 56 mm).
  - It remains smooth, so the null effect above is not a broken import pipeline.

  This control does *not* show that universal CONSTANT is safe. W108 already showed that forcing CONSTANT harms it.
- **Optional mixed single synthetic Action** (one held + one smooth curve): **not built.**

## 7. P4 — repeat, regressions and negative controls

- **Repeat.** `encode_repeat` and `probe_repeat` were written to fresh directories.
  - The encode reports are byte-identical (`ce4c7550…` both).
  - Variant and control FBX SHAs are equal across runs.
  - Add-on tree records, key-data digests and censuses are equal.
  - Every sampled array differs by 0.0, and outcomes are equal.
  - The two `probe_report.json` files are byte-identical (`f4ffc43d…`).
- **Negative controls:**

  | Control | Result |
  |---|---|
  | NC1 no-op identity | holds |
  | NC2 P0 replay | holds |
  | NC3 smooth control | holds |
  | NC4 non-target stacks keep `0x6104`; their imported Actions are identical | holds |
  | NC5 repeat | holds |
  | NC6 originals | holds. W106/W107/W108 manifests and all their files are unchanged after the run; every source `.blend` scratch copy is unchanged after sampling. |

- **Tests:**

  | Run | Result | Output |
  |---|---|---|
  | Focused, before the run | 11 passed, 1 stored-output test skipped | `w109_tests_predeclare_run.xml` |
  | Focused, final | **12 passed** | `w109_tests_final.xml` |
  | Regression | **51 passed** | `regression_w106_w107_w108_blender_executor.xml` |

  - The focused-before-the-run row was executed after the declaration and before encoding, and wrote
    `w109_tests_predeclare_run.xml`. The earlier development runs are listed in §8.
  - The regression files were `tests/test_w108_fbx_subframe.py`, `tests/test_w107_fbx_ownership.py`,
    `tests/test_w106_action_editability.py` and `tests/test_blender_executor.py`.
  - The focused tests (`tests/test_w109_fbx_constant_import.py`) cover:
    - the variant flags and the `0x100` aliasing;
    - stack-scoped target selection;
    - byte-exact patching: no-op identity, byte diff confined to targets, masked-tree equality, non-target change
      detection;
    - patch refusal on a non-`0x6104` source, a compressed array or a missing uid;
    - agreement between the offset index and the independent reader;
    - parser-neutral digests (add-on-style bool/array types);
    - in-place patchability of a real W107 V1 file;
    - declared floor/ceil functions;
    - classification;
    - the CASE mapping: A, B, C on mixed / not-discriminable / altered key data / invalid smooth, and D precedence;
    - code guards: no importer/exporter internals, no `foreach_set`/save/export, a single default import call, no
      torch;
    - stored declaration/analysis consistency.
  - A broad production suite was not run: no shared or tracked file changed (`shared_tracked_changes` = []).

## 8. Execution record, failed attempts and unexecuted work

- **Pre-declaration checks** (no mutation outcome, import or sampling of W109 files):
  - Development test runs, all on synthetic in-test FBX bytes plus the unmodified W107 `complete_control` V1:
    - the first run passed, but the code-guard test needed narrowing. The guard had flagged the repo's own
      `sys.path.insert` and a docstring mention of `import_fbx.py`, so it was narrowed to the actual
      importer/exporter-internal patterns. Then 11 passed, 1 skipped.
    - the `test_real_v1_fbx_targets_are_patchable_in_place` test ran a no-op patch in memory on the unmodified W107
      file.
  - A throw-away Blender smoke script (deleted afterwards) confirmed that the add-on parser and the independent
    reader give identical canonical digests on the unmodified W107 `complete_control` V1 FBX.
  - Before the declaration, the analysis code gained a guard for a missing control import, so that an absent variant
    record yields `NOT_IMPORTED` instead of an exception.
  - No script changed after the declaration hash. All declared SHAs are unchanged at the end.
- **Execution failures: none.**
  - verify, options, declare, encode ×2, probe ×2, analyze and manifest all exited 0.
  - No FAILED / NOT_TESTABLE record.
  - All 13 imports per run (26 in total) reported `FINISHED`, and every add-on parse succeeded.
- **Not executed:**
  - the optional mixed hold/smooth single synthetic Action;
  - any importer patch or post-import correction (out of scope);
  - mutation of V0 FBX, which is kept as a historical negative baseline;
  - FBX SDK or non-Blender consumer validation (not available);
  - other Blender versions;
  - instrumentation of `import_fbx.py` (the mechanism remains static inspection);
  - Euler-vs-quaternion isolation;
  - renders (the numbers are unambiguous);
  - LabServer.

## 9. First evidenced boundary and competing causes

Each layer has its own evidence type:

| Layer | Behaviour | Evidence type |
|---|---|---|
| Source Action | CONSTANT, held by evaluation | Blender evaluation (W108, re-verified bitwise) |
| Exporter | writes `0x6104` LINEAR for every curve | W108: encoding by two parsers, plus static inspection of the source-independent literal. This remains the **first** loss. |
| FBX encoding | a CONSTANT (standard or next) flag on the selected curves is representable byte-exactly, without touching anything else | W109: two parsers, byte diff, masked trees |
| Importer F-curves | identical to the LINEAR control in every field, including interpolation | W109: the flag value is **not reflected** in import output |
| Evaluated motion | bitwise identical to the LINEAR control; the subframe error is unchanged | W109: the flag has **no evaluated effect** |

- **Conclusion.** For this Blender-to-Blender path there are two independent hold-losing boundaries:
  1. export serialization (W108);
  2. importer flag handling (W109).

  Correcting either one alone does not restore the round trip, as far as this test can show. The importer is
  behaviourally shown to discard the flag. That it does so because it never reads `KeyAttrFlags` is consistent code
  evidence, not instrumentation.
- **Not resolved and not addressed:**
  - whether other FBX consumers honour `0x6002` / `0x6102`;
  - multi-take export (16 / 4 / 2 stacks);
  - `frame_end` 250;
  - the real clip's 50 fps keys in a 30 fps scene;
  - V1 storage and key cost;
  - per-curve hold/continuous semantics for mixed Actions;
  - any user-owned export/import policy.

## 10. Reproduction (Git Bash on Main, repo root)

```
PY=/c/Projects/AnimCV/.venv_build_full/Scripts/python.exe; BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
AD="C:/Program Files/Blender Foundation/Blender 5.1/5.1/scripts/addons_core/io_scene_fbx"
O=C:/Users/dna10/animcv-w109; R=$O/results; W6=C:/Users/dna10/animcv-w106; W7=C:/Users/dna10/animcv-w107; W8=C:/Users/dna10/animcv-w108
H=b5b46c3d90071f932d22bfca8ced9abd8877d5cb; S=b6bf24049e4be1270c3b3d64ff2d119188b87a8c89b79ac8ac951eac2b986f73
ES=ce4c755021b07ece79c57d5be9da05dc2f09a26fedfa81ada88d824584258b1d
B=("$BL" -b --factory-startup -t 2 --python-exit-code 1)
mkdir -p $R
$PY -X utf8 scripts/w109_fbx_constant_import.py verify --base $O --w106 $W6 --w107 $W7 --w108 $W8 --lock C:/Users/dna10/animcv-automation/state/active_batch.json --blender "$BL" --addon-dir "$AD" --starting-head $H --branch arch/single_frame_first
"${B[@]}" -P scripts/w108_fbx_subframe_blender.py -- options --out $R/options.json
$PY -X utf8 scripts/w109_fbx_constant_import.py declare --base $O --w107 $W7 --w108 $W8 --verify $R/verify/verify_report.json --options $R/options.json --blender "$BL" --starting-head $H
$PY -X utf8 scripts/w109_fbx_constant_import.py encode --declaration $R/declare/declaration.json --declaration-sha $S --out-dir $R/encode
$PY -X utf8 scripts/w109_fbx_constant_import.py encode --declaration $R/declare/declaration.json --declaration-sha $S --out-dir $R/encode_repeat
"${B[@]}" -P scripts/w109_fbx_constant_import_blender.py -- probe --declaration $R/declare/declaration.json --declaration-sha $S --encode-report $R/encode/encode_report.json --encode-sha $ES --out-dir $R/probe
"${B[@]}" -P scripts/w109_fbx_constant_import_blender.py -- probe --declaration $R/declare/declaration.json --declaration-sha $S --encode-report $R/encode_repeat/encode_report.json --encode-sha $ES --out-dir $R/probe_repeat
$PY -X utf8 scripts/w109_fbx_constant_import.py analyze --declaration $R/declare/declaration.json --declaration-sha $S --encode-dir $R/encode --encode-repeat-dir $R/encode_repeat --probe-dir $R/probe --repeat-dir $R/probe_repeat --w106 $W6 --w107 $W7 --w108 $W8 --verify $R/verify/verify_report.json --out-dir $R/analysis
$PY -X utf8 -m pytest tests/test_w109_fbx_constant_import.py -q
$PY -X utf8 scripts/w109_fbx_constant_import.py manifest --base $O --w106 $W6 --w107 $W7 --w108 $W8 --starting-head $H --declaration-sha $S
```

- Every output-writing stage refuses an existing directory.
- The verify stage requires a Batch-109 RUNNING lock and a clean tree, apart from the new W109 files.
- On a fresh base the declaration hash changes only through recorded paths and SHAs.

Runtimes:

| Stage | Runtime |
|---|---|
| each probe | ≈ 20 s |
| encode | a few seconds each |
| verify / declare / analyze / manifest | a few seconds each |
| focused tests | 0.9 s |
| regression | 0.6 s |

Outputs are local in `C:\Users\dna10\animcv-w109\` (129 MB, 130 files) and are not committed. They include input
copies, scratch FBX, imported samples (npz), meta and report JSON, and logs.
- `evidence_manifest.json`, SHA **`e62e8c657267345dcc40d6f41f099269255a120b61bd931d3599d2cb6ababdd1`**.
- It covers 128 files with `all_originals_unchanged: true`. It excludes itself and its own stdout capture
  `manifest_stdout.txt`, to avoid the W106–W108 self-hash artefact.

Other hashes:

| File | SHA |
|---|---|
| `verify_report.json` | `e979cf4b…` |
| `encode_report.json` | `ce4c7550…` |
| `probe_report.json` (both runs) | `f4ffc43d…` |
| `analysis_report.json` | `44417feb…` |

## 11. Changed files, resources and environment

New files only:

| File | Role | SHA |
|---|---|---|
| `scripts/w109_fbx_constant_import_contract.py` | Pure contract: cases/roles, flag semantics and variant derivation, offset-aware byte patcher, stack-scoped target selection, parser-neutral canonical tree digest, declared floor/ceil functions, classification, CASE mapping | `b3793c48…` |
| `scripts/w109_fbx_constant_import_blender.py` | Blender `probe`: source sampling, add-on parse, default import, census / key data / matching, sampling | `acd26dca…` |
| `scripts/w109_fbx_constant_import.py` | Host verify / declare / encode / analyze / manifest | `0949e9b2…` |
| `tests/test_w109_fbx_constant_import.py` | Focused tests | `a8bcde4f…` |
| this Worklog | | |

**Line-ending note (post-commit check).** The SHAs above, and those in the declaration, are of the bytes executed
on Main. Three of the files are CRLF in the working copy: host, contract and test. They were last edited by a Python
`write_text` on Windows. The Blender stage file is LF. Git (`core.autocrlf=true`) stores all four LF-normalized, so
the committed blob SHAs are:
- host `5451d32b…`;
- contract `2134a58a…`;
- Blender `acd26dca…`;
- test `e07e9ab5…`.

The content is identical apart from line endings. Converting the committed LF blob to CRLF reproduces the declared
`0949e9b2…` / `b3793c48…` / `a8bcde4f…` exactly.

Reused, unchanged and read-only: the W108 contract (reader, metrics), the W108 Blender key-data helpers and `options`
stage, the W108 host manifest/git helpers, and the W107/W106 Blender helpers.

Nothing else changed:
- no `src/` file, exporter or importer setting, add-on file, rig, Action original, W106/W107/W108 artifact or
  historical Worklog;
- not `docs/next_batch.md`;
- no automation file.

Resources: Main only, CPU only. No rendering, GPU, inference, package change or network. The private real-clip frames
were not read.

Final source revision: the commit adding this Worklog on `arch/single_frame_first`, directly after `b5b46c3`. The
COMPLETE marker is added in a separate final commit.

## 12. Limits

- **Coverage.** One importer (Blender 5.1.0 / io_scene_fbx 5.15.0, default options) and three reused real Actions on
  one BaseRig, plus synthetic step/smooth controls. Two predeclared CONSTANT encodings, applied to every curve of the
  selected take.
- **What was not shown.** It does not show:
  - that the variants are valid for the Autodesk SDK or for other consumers;
  - what per-curve mixed flags would do;
  - behaviour on V0 (simplified keys).
- **Sampling.** Quarter frames only.
- **What CASE A covers.** A null behavioural effect of the encoded flag on this importer. It is not a claim about the
  correct FBX policy, and it is not an exporter or importer fix. It authorizes no production change.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 109
