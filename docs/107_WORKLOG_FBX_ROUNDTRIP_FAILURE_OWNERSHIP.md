# Worklog — FBX Action Round-Trip Failure Ownership (2026-11-03)

## Human Assessment

- **SUCCESS:** The W106 real-Action FBX baseline drift was reproduced exactly (V0). It is removed by one predeclared,
  isolated exporter argument. Setting `bake_anim_simplify_factor=0.0` (V1), with the default importer and every other
  setting unchanged, lowers the integer-frame world drift on all three real Actions:

  | Action | V0 drift | V1 drift |
  |---|---:|---:|
  | complete_control | 1.29 mm | 0.95 µm |
  | arm_natural_most | 6.10 mm | 1.31 µm |
  | real_clip | 7.51 mm | 1.66 µm |

  **All three pass the original 1 mm gate.** V1 causes no collateral bone regression, and a fresh repeat is bitwise
  identical. Every identity check and negative control passed.
- **FAILURE / NOT FIXED:** V1 does not address the other W106 FBX defects. All of them persist unchanged under V1:
  - CONSTANT→LINEAR interpolation;
  - half-frame deviations of 0.42 / 0.96 / 0.35 m;
  - `frame_end` 250;
  - multi-take leakage (16 / 4 / 2 takes);
  - the real clip's 50→30 fps timebase.

  V1 also costs file size: the FBX grows 2.2–2.8× and the imported selected Action has 6–13× more keys.

  V2 (`bake_anim_use_all_actions=False` as the single toggle) exported **no animation at all**: 0 stacks and 0
  curves. Its geometric effect could therefore not be measured.
- **UNCERTAIN:**
  - **Importer.** No importer-side probe was run. The default importer reconstructs within float32 precision once it
    receives unsimplified keys, but its behaviour on other inputs is untested.
  - **Rotation conversion.** Euler conversion is not independently manipulable. Its residual is bounded at ≤ 1.7 µm in
    these cases.
  - **V3 synthetic 2×2.** Its gate metric is dominated by a constant, construction-specific 50 mm bone-tail
    reconstruction offset on the synthetic `base` bone in every cell. So the gate-level V3 contrasts carry no
    information.
  - **Why the W106 synthetic shows no simplification loss** while the real BaseRig does. This was not established.
- **ALTERNATIVE EXPLANATIONS (status):**

  | Explanation | Status |
  |---|---|
  | Sampling-time mismatch | Excluded at integer frames. V1 uses identical times and reaches µm parity. |
  | Action/slot selection | Excluded. The same uniquely name-matched Action is used in V0 and V1. |
  | 0.01 object scale | Not supported. No effect in V3 (≤ 3e-8 m), and V1 fixes the real rig without changing scale. |
  | Take contamination as the geometric cause | Not supported. V1 keeps 16/4/2 takes and still reaches µm parity. |
  | Interaction of simplification with rig properties | Not excluded. |

**CASE A — causal factor supported (default key simplification), under the frozen rule.** This is not approval to
change the exporter. No production file was modified.

## Scope and starting state

Task scope (summarized): an evidence-only, one-factor-at-a-time attribution of the W106 FBX baseline drift on
immutable scratch copies of the three W106 M_Main Actions plus the W106 synthetic control. The round trip under test
is the unchanged Blender → `src/blender/export.py` → default FBX importer path.
- No production exporter, importer, retarget, rig asset, Action original, scene FPS or interpolation was changed.
- There was no training, inference, rendering, network access or LabServer work.

Starting state:
- Branch `arch/single_frame_first`; starting HEAD `38f2274690d34de5a933c4c4cf0b78ea7ecfe8db`; instruction blob
  `a98fe18b412c76fab0a312287cfc4a5c2ce00f28`. Both were verified before any work.
- `git fetch` showed no divergence, and the working tree was clean.
- The worker lock was Batch 107 / RUNNING for this executor.
- Run entirely on **Main**:
  - Python 3.11 at `C:\Projects\AnimCV\.venv_build_full`;
  - Blender **5.1.0** (hash `adfe2921d5f3`), always `-b --factory-startup -t 2`;
  - FBX add-on `io_scene_fbx` **5.15.0**. Its source SHAs are recorded in `options.json`.

Regimes, labelled as in W106:

| Action | Regime |
|---|---|
| `complete_control`, `arm_natural_most` | `benchmark_detector_observation` (3DPW, W102 mixed-source contract) |
| `real_clip` | `real_animcv_observation` (GT-free historical clip) |
| `w106_synthetic` and V3 | Synthetic contract evidence; never pooled with the real Actions |

The 3DPW windows and the real clip are reused, non-holdout data.

## 1. Preflight and immutable identities (`results/verify/verify_report.json`, SHA `027df221…`)

- The W106 Worklog ends with its COMPLETE marker.
- The W106 `evidence_manifest.json` SHA is `f3003fe5…`, as published. It records `all_originals_unchanged: true` and
  declaration `a72f6dc5…`.
- All 32 reused W106 files match their W106 manifest entries:
  - per case: `n0.blend`, `n0_reopen.npz/json`, `fbx_n0.npz/json/meta`, `fbx/n0.fbx`;
  - plus the W106 declaration, analysis, verify and fbx_attribution reports.
- `src/blender/export.py` SHA is `521a63d9…`, equal to W106's recorded value. Its V0 arguments were parsed from source
  (AST): `use_selection=False`, `object_types={'ARMATURE'}`, `bake_anim=True`. Everything else is at Blender defaults.
- Inputs are byte-identical scratch copies of the W106 N0 `.blend` files in `results/inputs/`. In each, the active
  Action is already the declared M_Main `FIXED` Action, never R11 `RUNTIME`. Only these copies were opened.
- Action identity (probe):
  - The source Action digest of every copy equals the W106 N0 digest.
  - For the real cases it also equals the W106 declared digest (`d3f05754…`, `ee1ead92…`, `4d168d95…`).
  - It is unchanged before and after every export.

## 2. Declaration (frozen before any W107 export/import ran)

`results/declare/declaration.json`, SHA-256 **`aeb9ce710d01878e8d55a0b26d1d117e9ddfc9868ae397c6d88d1567975f5a76`**,
`outcomes_read: false`. The probe refuses to run without this hash and checks the Blender version against it.

Runtime operator properties were recorded first (`options.json`, SHA `7d1a5472…`). The relevant defaults are:
- `bake_anim_simplify_factor` = 1.0 (range 0–100);
- `bake_anim_use_all_actions` = True;
- `bake_anim_use_nla_strips` = True;
- `bake_anim_step` = 1.0;
- `bake_anim_force_startend_keying` = True;
- `add_leaf_bones` = True;
- `apply_scale_options` = `FBX_SCALE_NONE`;
- axes `-Z` / `Y`.

Variants (order V0, V0d, V1, V2, then V3):

| Variant | Exporter call | Single change vs V0 | Purpose |
|---|---|---|---|
| V0 | real `blender.export.export_fbx(path)` | — | baseline; must reproduce W106 |
| V0d | `bpy.ops.export_scene.fbx` with V0's parsed arguments | none | negative control: diagnostic path = production path |
| V1 | same as V0d | `bake_anim_simplify_factor=0.0` (documented no-simplification value) | simplification isolation |
| V2 | same as V0d | `bake_anim_use_all_actions=False` | take/Action-enumeration isolation |
| V3 | V0 production call on 4 synthetic cells | — (design factor is the rig, not the exporter) | animated object × 0.01 scale mechanism probe |

Further declared items:
- **Importer:** always `bpy.ops.import_scene.fbx` defaults into `read_factory_settings(use_empty=True)`.
- **Importer probe:** declared NOT RUN. No single importer property could be predeclared with a drift-specific
  mechanism.
- **Rotation conversion:** declared NOT INDEPENDENTLY MANIPULABLE. The exporter has no option that keeps quaternions.
- **No combinations or sweeps.** V3 is not crossed with V1 or V2.

**V3 design** (49 dense CONSTANT keys at frames 1–49, 8 bones):
- A rigid global rotation G(t) = Rx(90°)·(yaw sweep ±35° with sub-degree tilts) is applied in one of two ways:
  - *static* cells key the `base` bone with Rrest⁻¹·G·Rrest (`base` head at the origin);
  - *animated* cells key the object rotation with G and leave `base` at rest.
- The scale-0.01 cells multiply the bone geometry by 100.
- Child local rotations are identical in all cells.
- Invariant: every cell's Blender-evaluated world heads/tails equal the reference cell (`static_s1`) within 1e-6 m
  before export. Otherwise V3 is NOT TESTABLE.

Metrics, at the exact source integer and half-frame times (139 / 231 / 397 / 49 samples):
- **Gate metric:** W106's metric, unchanged: max |coordinate| difference of world heads and tails, over every source
  bone, at integer frames, with a 1 mm gate.
- **Diagnostics:**
  - Euclidean max / p95 / median, and frames > 1 mm;
  - half-frame max;
  - armature-object rotation / translation / scale;
  - armature-space tail error;
  - local quaternion angle (only for bones with a comparable rest rotation);
  - per-bone and per-frame outliers;
  - imported Action count and selection, interpolation, key counts, frame range and FBX size;
  - FBX-level stacks / curves / keys, parsed with `io_scene_fbx.parse_fbx`.

Tolerances:

| Check | Tolerance |
|---|---|
| Determinism, V0-vs-W106, V0d-vs-V0 | ≤ 1e-9 |
| Source fidelity | bitwise |
| Substantial reduction | V_F ≤ 0.5·V0 per case |
| Collateral | no bone's gate metric may grow by > 0.1 mm |
| W106 synthetic control | ≤ 1e-5 m |
| V3 equivalence | ≤ 1e-6 m |

**Factor rule:** a factor is *supported* iff, on every real case:
- it is substantial;
- it has no collateral bone;
- the repeat agrees;
- the selected Action is the declared one;

and at least one real case reaches ≤ 1 mm.

**CASE rule** (`w107_fbx_contract.decide_case`):

| CASE | Condition |
|---|---|
| D | Any identity, source-fidelity, V0-reproduction, determinism, V0d=V0, Action-mutation or originals failure |
| A | A supported factor, and the synthetic control passes, and V3 was tested |
| C | A mixed factor, an untestable declared variant, V3 not testable, or a failed synthetic control |
| B | Otherwise |

## 3. Negative controls and V0 reproduction (`results/analysis/analysis_report.json`, SHA `dcafa26b…`)

| Control | complete_control | arm_natural_most | real_clip | w106_synthetic |
|---|---|---|---|---|
| NC1 W107 source samples vs W106 N0 (bitwise) | 0 | 0 | 0 | 0 |
| NC2 V0 imported samples vs W106 `fbx_n0` | 0 | 0 | 0 | 0 |
| NC2 V0 gate metric − W106 gate metric | 0 | 0 | 0 | 0 |
| NC3 V0d vs V0 samples | 0 | 0 | 0 | 0 |
| NC4 every variant vs repeat probe | 0 | 0 | 0 | 0 |
| NC5 Action digest before/after export; work file unchanged | yes / yes | yes / yes | yes / yes | yes / yes |
| NC6 W106 synthetic V0 gate (≤ 1e-5 m) | — | — | — | 7.7e-7 m |
| NC7 W106 originals unchanged (32 files) | yes | | | |

**V0 reproduces W106 bitwise:** 1.2898 / 6.1038 / 7.5123 mm, with 2 / 32 / 80 frames > 1 mm.

The FBX *file bytes* are not reproducible run to run: SHAs always differ, and sizes sometimes do, presumably from the
FBX header timestamp. The imported samples are bitwise reproducible.

## 4. Per-case results — real Actions (integer frames unless stated)

| Measure | complete_control V0 → V1 | arm_natural_most V0 → V1 | real_clip V0 → V1 |
|---|---|---|---|
| **Gate: max abs coordinate** | **1.290 mm → 0.95 µm** | **6.104 mm → 1.31 µm** | **7.512 mm → 1.66 µm** |
| Reduction | 99.93 % | 99.98 % | 99.98 % |
| Passes 1 mm | no → **yes** | no → **yes** | no → **yes** |
| Euclidean p95 / median (bone-frames) | 0.52 mm / 0.5 µm → 0.85 / 0.43 µm | 3.24 mm / 0.9 µm → 1.09 / 0.60 µm | 2.04 / 0.28 mm → 0.96 / 0.51 µm |
| Frames > 1 mm | 2 / 70 → 0 | 32 / 116 → 0 | 80 / 199 → 0 |
| Object rotation max | 0.076° → 3.1e-5° | 0.271° → 2.7e-5° | 0.314° → 6.2e-5° |
| Object translation / scale | 0 / 2e-9 → 0 / 2e-9 | 0 / 2e-9 → same | 0 / 6e-9 → 0 / 2e-9 |
| Armature-space tail max (median) | 0.50 mm (0.07) → 0.94 µm | 1.58 mm (0.19) → 1.41 µm | 2.96 mm (0.75) → 1.42 µm |
| Collateral bones (> +0.1 mm) | — → none | — → none | — → none |
| Worst bones V0 | ring_01_l / ring_02_l | ring_01_l / ring_02_l | neck_02 / neck_01 |
| Worst frames V0 | 10, 9 | 105, 104 | 104, 173 |
| Half-frame max (CONSTANT → LINEAR) | 0.417 m → 0.417 m | 0.960 m → 0.960 m | 0.354 m → 0.354 m |
| Imported interpolation | LINEAR → LINEAR | LINEAR → LINEAR | LINEAR → LINEAR |
| Selected-Action keys (all F-curves) | 4,593 → 36,330 | 4,593 → 60,204 | 16,581 → 103,281 |
| Target / object keys per curve | 47 / 47 → 70 / 70 | 47 / 47 → 116 / 116 | 199 / 199 → 199 / 199 |
| Source Action keys (CONSTANT) | 768 | 768 | 6,368 |
| Takes (imported Actions = FBX stacks) | 16 → 16 | 4 → 4 | 2 → 2 |
| FBX curve keys, all takes | 32,637 → 876,096 | 7,861 → 232,128 | 10,219 → 186,264 |
| FBX size | 4.25 MB → 9.51 MB | 1.11 MB → 2.49 MB | 0.65 MB → 1.81 MB |
| fps / frame_start / frame_end after import | 30 / 1 / 250 → same | 30 / 1 / 250 → same | 30 / 1 / 250 → same |
| Action selection | name_match → name_match | name_match → name_match | name_match → name_match |

Local channels:
- Only 2 of 51 source bones (`interaction`, `center_of_mass`, both unanimated) keep a comparable rest rotation after
  import. Their local-quaternion error is 0.
- For every other bone the importer changes the rest orientation. So armature-space and world metrics are used, not
  raw local channels.

Outlier attribution (V0):
- The worst bones are distal: fingertips on the 3DPW windows, and neck on the real clip. The worst frames are
  isolated, and the medians are about 0.
- That pattern fits simplified keys being linearly re-interpolated between the keys that survive. In
  `complete_control` the target track keeps 47 of 70 keys.
- V1 keeps a key on every baked frame and removes all of it.

Interpretation:
- **Supported (one-factor, repeated).** The integer-frame geometric loss on these three real Actions is owned by the
  exporter's default bake key simplification (`bake_anim_simplify_factor=1.0`). The importer, scale, object
  animation, takes and Action selection were held fixed.
- **Not shown.** That the factor is the only one on other rigs or Actions. That the importer is lossless in general.
  That V1 fixes anything except integer-frame geometry.
- The residual 0.95–1.66 µm is at float32 FK level. It bounds the combined Euler-conversion and importer-reconstruction
  error for these inputs.

## 5. W106 synthetic control (separately labelled)

| | V0 | V1 |
|---|---:|---:|
| Gate | 0.77 µm | 0.86 µm |
| Imported keys | 848 | 1,475 |
| Target keys | 17 | 25 |

- Half-frame max is 0.159 m in both.
- Interpolation is LINEAR, with 1 take and frame_end 250.

The default simplification drops keys here too (17 of 25 target keys survive) without measurable geometric loss. Why
the real BaseRig loses millimetres while this armature does not is **not established**.

## 6. V2 — take-ownership isolation

- `bake_anim_use_all_actions=False` alone, with all other properties at defaults including
  `bake_anim_use_nla_strips=True` and no NLA tracks, exported an FBX with **0 AnimationStacks, 0 curves and 0 takes**.
  This held for every case, including the synthetic one.
- Import produced 0 Actions, so the selected Action could not be resolved (`NO_UNIQUE_ACTION`).
- Under the frozen rule, V2 is therefore "not supported" (no measurable reduction). Its effect on geometric parity is
  **not measured**.
- What it does establish about take ownership: with these defaults, multi-take leakage cannot be removed by this one
  toggle without losing the animation entirely.
- A second toggle (NLA strips) would be a two-factor combination. It was not run (no opportunistic combinations).
- Take contamination is separately shown **not** to own the geometric drift: V1 keeps all 16 / 4 / 2 takes and still
  reaches µm parity.

## 7. V3 — synthetic world-equivalent 2×2 (V0 settings only; descriptive)

Equivalence held before export:
- Maximum world head/tail difference vs `static_s1` was 0 / 1.2e-7 / 3.0e-7 / 3.0e-7 m.
- Each cell imported one name-matched Action, and a repeat was bitwise identical. The status is therefore TESTED.

| Cell | Gate (all bones) | Max excluding `base` tail (post-hoc) | `base` tail error (min–max) |
|---|---:|---:|---:|
| static, scale 1 | 50.001 mm | 4.879 mm | 50.000–50.011 mm |
| static, scale 0.01 | 50.001 mm | 4.879 mm | 50.000–50.011 mm |
| animated, scale 1 | 50.000 mm | 3.055 mm | 50.000–50.000 mm |
| animated, scale 0.01 | 50.000 mm | 3.055 mm | 50.000–50.000 mm |

**Gate-level contrasts:**

| Contrast | Value |
|---|---:|
| Scale | ≤ 3e-8 m |
| Object animation | −6.7e-7 m |
| Interaction | 1.5e-8 m |

**Confound (found after reading the outcome):**
- The gate is dominated by a frame-constant ≈ 50 mm error on the synthetic `base` bone's *tail*. The `base` head
  error is 0.
- `base` has two unconnected children at different positions (`spine`, `thigh_l`), and the exporter adds `*_end` leaf
  bones only to childless bones. The likely mechanism is that the importer infers this bone's tail from its children,
  differently from the source. That mechanism was inferred, not verified.
- Either way, it behaves as a rest-geometry artifact of the V3 construction: it is frame-constant, identical in all
  four cells, and the `base` head error is 0. It is not animation loss.
- So the predeclared V3 gate metric **cannot discriminate** the factors.

The post-hoc decomposition (`results/v3_posthoc/v3_posthoc.json`, SHA `3b5308b1…`) is labelled post-hoc and is never
a gate. It shows:
- **Scale 0.01 has no effect** (≤ 3.4e-8 m) in either object condition.
- Putting the global motion on the object instead of the `base` bone lowers the arm chain error from 1.1–4.1 mm to
  ≤ 0.012 mm. The leg chain keeps 1.5–3.1 mm (`thigh_l`, `calf_l` tails).
- The cells were not crossed with V1. So whether these synthetic mm errors are also simplification losses is
  **unresolved**, and no V3 cell is evidence about the real rig.

## 8. Timing, take and interpolation status (preserved, not repaired)

None of these changes under V1. All remain separate defects:
- **CONSTANT → LINEAR on import**, with half-frame deviations of 0.42 / 0.96 / 0.35 m.
- **`frame_end` reset to 250.**
- **Multi-take export of every stored Action** (16 / 4 / 2).
- **The real clip's native 50 fps keys in a 30 fps scene (0.6× playback).** fps 30 is kept through import.

No interpolation, scene FPS, frame range or key time was changed anywhere.

## 9. Tests and execution record

| Run | Result | Output |
|---|---|---|
| Focused tests before the declaration | 11 passed, 2 stored-output tests skipped | `w107_tests_predeclare.xml` |
| Focused tests, final | **13 passed** | `w107_tests_final.xml` |
| Relevant regression | **54 passed** | `regression_w106_action_export_rig.xml` |

The focused tests (`tests/test_w107_fbx_ownership.py`) cover:
- V0 arguments AST-parsed equal to the frozen exporter;
- each causal variant changes exactly one property, from its runtime default;
- variants cannot override a production argument;
- no exporter monkey-patch, in-place save or torch import;
- V3 world-equivalence algebra and the fixed 2×2 design;
- the gate metric equal to W106's definition (integer frames only, extra `_end` bones ignored);
- the collateral rule;
- the factor-support rule and CASE mapping, including D precedence;
- the stored declaration hash, `outcomes_read: false`, and analysis↔declaration consistency.

The regression suite was `test_w106_action_editability.py`, `test_blender_executor.py`, `test_w105_rig_utility.py`
and `test_retarget_solver.py`.

The metric was checked once against W106 historical data before the declaration. That reproduced W106's published
1.2898 / 6.1038 / 7.5123 mm and the synthetic 7.7e-7 m.

Execution record:
- **Execution failures: none.** No stage crashed or was retried. V2's `NO_UNIQUE_ACTION` is an outcome, not a failure.
- **Post-declaration code change.** `scripts/w107_fbx_ownership.py` gained the `v3-posthoc` stage (SHA at declaration
  `21579bed…` → final `a8fdebc5…`). The contract (`9af0a328…`) and Blender (`985fc4bc…`) scripts are unchanged since
  the declaration.

Not executed:
- importer-side probe (declared NOT RUN);
- rotation-conversion isolation (no operator option);
- V3 × V1 crossing and any other combination;
- edited-Action (E1/E2) FBX deltas, which were not needed for the baseline question;
- renders (the numbers were unambiguous);
- broad production regression (no shared file changed: `shared_tracked_changes` = []);
- other rigs, Actions or engines;
- LabServer jobs.

## 10. Reproduction (Git Bash on Main, repo root)

```
PY=/c/Projects/AnimCV/.venv_build_full/Scripts/python.exe; BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
O=C:/Users/dna10/animcv-w107; R=$O/results; W=C:/Users/dna10/animcv-w106; H=38f2274690d34de5a933c4c4cf0b78ea7ecfe8db
S=aeb9ce710d01878e8d55a0b26d1d117e9ddfc9868ae397c6d88d1567975f5a76
B=("$BL" -b --factory-startup -t 2 --python-exit-code 1 -P scripts/w107_fbx_blender.py --)
$PY -X utf8 scripts/w107_fbx_ownership.py verify --base $O --w106 $W --lock C:/Users/dna10/animcv-automation/state/active_batch.json --blender "$BL" --starting-head $H
"${B[@]}" options --out $R/options.json
$PY -X utf8 scripts/w107_fbx_ownership.py declare --base $O --w106 $W --verify $R/verify/verify_report.json --options $R/options.json --blender "$BL" --starting-head $H
"${B[@]}" probe --declaration $R/declare/declaration.json --declaration-sha $S --out-dir $R/probe
"${B[@]}" probe --declaration $R/declare/declaration.json --declaration-sha $S --out-dir $R/probe_repeat
$PY -X utf8 scripts/w107_fbx_ownership.py analyze --declaration $R/declare/declaration.json --declaration-sha $S --probe-dir $R/probe --repeat-dir $R/probe_repeat --w106 $W --verify $R/verify/verify_report.json --out-dir $R/analysis
$PY -X utf8 scripts/w107_fbx_ownership.py v3-posthoc --probe-dir $R/probe --out-dir $R/v3_posthoc
$PY -X utf8 scripts/w107_fbx_ownership.py manifest --base $O --w106 $W --verify $R/verify/verify_report.json --starting-head $H --declaration-sha $S
$PY -X utf8 -m pytest tests/test_w107_fbx_ownership.py -q
```

Each output-writing stage refuses an existing directory. On a fresh base the declaration hash changes only through
its recorded paths and SHAs.

Runtimes:

| Stage | Runtime |
|---|---|
| verify / declare / analyze / post-hoc | a few seconds each |
| options | 7 s |
| probe | 17.3 s |
| repeat probe | 15.4 s |
| tests | < 1 s |
| regression | 3.2 s |

Outputs are local in `C:\Users\dna10\animcv-w107\` (76 MB, 183 files) and are not committed. They include scratch
`.blend`/FBX files, per-variant `.npz`/meta JSON, reports and logs.
- `evidence_manifest.json`, SHA **`20fcb1b9f6ec8879247ac91ae3a21ca1185e995340976079bc80936316f57278`**.
- It records `all_originals_unchanged: true`, which covers the 32 W106 files and the W106 manifest.

## 11. Changed files, resources and environment

New files only:

| File | Role | SHA |
|---|---|---|
| `scripts/w107_fbx_contract.py` | Pure contract: variants, V3 design, metrics, tolerances, factor rule, CASE mapping | `9af0a328…` |
| `scripts/w107_fbx_blender.py` | Blender stages `options` / `probe`. Imports only the unchanged exporter and W106 measurement helpers | `985fc4bc…` |
| `scripts/w107_fbx_ownership.py` | Host stages verify / declare / analyze / v3-posthoc / manifest | `a8fdebc5…` |
| `tests/test_w107_fbx_ownership.py` | Focused tests | `34f38c0b…` |
| this Worklog | | |

Nothing else changed:
- no `src/` file, exporter or importer setting, rig asset, Action original, W105/W106 artifact or historical Worklog;
- not `docs/next_batch.md`;
- no automation file.

Resources:
- Main only, CPU only: no rendering, no GPU job, no inference, no package change, no network.
- The private real-clip source frames were not read.

Final source revision: the commit adding this Worklog on `arch/single_frame_first`, directly after `38f2274`. The
COMPLETE marker is added in a separate final commit.

## 12. Limits and the decision this evidence informs

Limits:
- Three reused real Actions on one BaseRig, Blender 5.1.0 / FBX add-on 5.15.0, round-tripped only through Blender's
  own importer. No game engine or DCC consumer was tested.
- "Causal" here means a predeclared one-factor intervention with every other setting fixed, repeated once on fresh
  outputs. It is not a sweep, and it does not show the factor is unique on other inputs.
- Cost was measured only as file size and key count, not consumer performance.

The evidence would inform a later **human-owned** decision: whether an AnimCV FBX export contract should state the
bake key-simplification setting explicitly, weighed against the measured 2.2–2.8× file size. Interpolation, frame
range, take enumeration and timebase would remain separate, unresolved contract questions. No exporter change is
authorized or made by this Worklog.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 107
