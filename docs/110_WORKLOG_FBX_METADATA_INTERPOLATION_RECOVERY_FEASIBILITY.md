# Worklog — FBX-Embedded Interpolation Metadata Recovery Feasibility (2026-11-04)

## Human Assessment

**Verdict: CASE D — INVALID / STOPPED under the frozen rule. No viability conclusion is drawn.**

The frozen CASE rule checks validity inputs before any outcome gate. One of them, `deterministic` (fresh repeat),
failed on the new mixed synthetic case. Everything below is recorded as evidence, not as support for H110.

- **SUCCESS / SUPPORTED (as observations only, not a verdict):**
  - **Producer edit.** A byte-exact producer edit wrote W109's CONSTANT-standard flag (`0x6104 → 0x6002`) only on
    selected-take FBX curves whose driving source channel is uniformly CONSTANT:

    | Case(s) | CONSTANT curves | Remaining curves (stay `0x6104`) |
    |---|---:|---|
    | Each real Action | 24 / 468 | undriven or LINEAR |
    | syn_step | 15 / 54 | undriven or LINEAR |
    | Mixed | 12 / 54 | undriven or LINEAR |

    The validity checks were the same as W109's: byte diff only inside the target spans, masked trees identical,
    the FBX-side map unchanged.
  - **Consumer adapter.** A separately invocable adapter received only the FBX file, the take name and the
    freshly imported scene. Its file access was audit-logged: it opened only the FBX. It mapped **100 %** of the
    selected-take FBX curves (468/468 real, 54/54 synthetic) and **100 %** of the imported target-Action F-curves
    (519/519, 59/59). There were 0 unresolved, ambiguous or doubly claimed curves. It changed only keyframe
    `interpolation`, on 31 / 20 / 16 F-curves (real / syn_step / mixed). Everything else stayed bitwise unchanged:
    the other key fields, the 15 / 3 / 1 other Actions, rest pose, objects, pose rotation modes and timebase.
  - **Motion.** On those runs the corrected import matched the source at every integer, quarter-frame and
    near-boundary (`f+0.99`) sample:

    | Case | Max error |
    |---|---|
    | complete_control / arm_natural_most / real_clip | 1.07 / 1.43 / 1.66 µm |
    | syn_step | 0.77 µm |
    | syn_smooth (left LINEAR) | 0.51 mm |
    | Mixed, held component | 0.77 µm |
    | Mixed, smooth component | 0.51 mm |

  - **Controls.** Every negative and attribution control behaved as declared:
    - The untouched import harms the held cases by 0.33–1.22 m.
    - Blanket CONSTANT harms the smooth components by 113 mm (syn_smooth) and 60 mm (mixed).
    - The adapter run on the unmodified LINEAR FBX changes nothing.
    - The metadata FBX without the adapter imports bitwise identically to the control, so producer and consumer
      are each required.
- **FAILURE / NOT FIXED:**
  - **The repeat gate failed.** The declared repeat compared the two fresh mixed-case V1 exports by canonical tree
    (only `FBXHeaderExtension/CreationTimeStamp` skipped) and the adapter plans verbatim. Both differed.
  - **The cause is exporter-side.** The stock exporter assigns object UIDs from Python `hash(key)`
    (`fbx_utils.py:834`), which is randomized per process. It also embeds the absolute source `.blend` path in
    `SceneInfo` (`Original|ApplicationNativeFile`). Every fresh export therefore carries new UIDs (and a different,
    UID-induced element order) and a different path string.
  - **The other repeat components were identical.** All sampled motion (max difference 0.0), changed-curve lists,
    census, producer directives and plan accounting were equal. The frozen repeat criterion did not anticipate the
    UID randomization, and it was not relaxed after the fact.
  - **Nothing in production changed.** No exporter, importer, default or production path was changed. The stock
    Blender→FBX→Blender path still loses holds unless both the scratch producer edit and the scratch consumer
    adapter are applied. Every earlier open defect persists:
    - `frame_end` 250;
    - multi-take export;
    - the 50 → 30 fps real-clip timebase;
    - V1 key and file cost.
- **UNCERTAINTY:**
  - **Correspondence granularity.** The finest deterministic correspondence available through this importer is
    *(item, Lcl property) → (object / pose bone, location | rotation | scale)*, not per axis. The importer bakes
    each item's curves through one matrix per key and decomposes it (code inspection). Per-axis 1:1 FBX-curve →
    F-curve correspondences: **0** by construction.
  - **Within-property semantics are not tested.** Different semantics *within* one Lcl property would be counted
    unresolved. No tested case contains such a mixture.
  - **Property-level separability is assumed.** Separability rests on zero pivots/offsets/geometric transforms,
    which the adapter checks from the FBX: 52/52 and 6/6 models pass. It also assumes no time-varying non-uniform
    Lcl Scaling, which is not checked. Every tested channel was rotation-only on the source side.
  - **Scope.** Only Blender 5.1.0 / io_scene_fbx 5.15.0 was tested. Main-local artifacts were produced by this
    Agent and are not independently replayed.
  - **The repeat failure itself.** Whether a fixed hash seed would make the mixed export reproducible is
    untested. It was deliberately not tried after the outcome.
- **ALTERNATIVE EXPLANATIONS (status):**

  | Explanation | Status |
  |---|---|
  | Adapter secretly uses source knowledge | Excluded as far as tested: static import/token guard (test), source-free consumer input (structural whitelist), runtime audit hook (only the FBX opened, 3 events per run), consume stage never given a source path. |
  | Recovery from blanket conversion, not metadata | Excluded: control_A (adapter on LINEAR-only FBX) changes 0 curves and equals control_U; blanket CONSTANT harms both smooth components; adapter changes exactly the CONSTANT-directed groups (8 / 5 / 4). |
  | Importer reads the flags after all | Excluded again: meta_U imports bitwise equal to control_U on all six cases (W109 result reproduced). |
  | Altered key times / values / rig | Excluded: all imported key fields except interpolation bitwise equal; rest, objects, timebase, other Actions unchanged. |
  | Name-based mapping hitting a wrong bone | Mapping requires exact unique names *and* FBX-parent == imported-bone-parent for every LimbNode; 0 parent mismatches. |
  | Repeat failure reflects non-deterministic adapter / producer | Not supported by the post-hoc check (UID-relabelled plans and directives equal; samples identical) — but this is post-hoc and does not change CASE D. |

## Original task scope (summary)

Batch 110 asked one bounded scratch question. Can per-curve interpolation metadata embedded in the FBX file be
read back by a source-independent consumer step, after the unchanged Blender 5.1 import? Specifically, can it
restore held motion while preserving smooth LINEAR channels? The test set was:
- three held real Actions;
- the W108 step and smooth synthetic controls;
- one new predeclared mixed held/smooth synthetic Action.

The source may be used only by the producer and the evaluator, never by the consumer adapter. The Batch authorizes
no production exporter/importer change, policy, add-on patch, default, retiming, simplification change or
downstream feature.

## Starting state

- **Repository.**
  - Branch `arch/single_frame_first`.
  - HEAD `186ed882f5d63e82d41682a295ed029ad638d513`.
  - `docs/next_batch.md` blob `7e05180414884791d4c389a4a6d3f717ad7fc364`.
  - Both were verified before any work. After `git fetch`, `origin` equalled HEAD and the tree was clean.
- **Worker.** The lock was `batch 110, RUNNING, start_commit = HEAD` (poller 36848, executor 4468). No other
  Blender process was running.
- **Environment.** Main, CPU only, under Python 3.11 (`C:\Projects\AnimCV\.venv_build_full`):
  - Blender **5.1.0** (`adfe2921d5f3`), exe SHA equal to W109, launched `-b --factory-startup -t 2`;
  - FBX add-on **5.15.0**, all ten source SHAs equal to W109;
  - `options.json` `7d1a5472…`, byte-identical to W108/W109.

Regimes (reused, non-holdout data; never pooled):

| Case | Regime | Role |
|---|---|---|
| complete_control, arm_natural_most | `benchmark_detector_observation` (3DPW) | real, held source |
| real_clip | `real_animcv_observation` (GT-free) | real, held source |
| syn_step | synthetic (W108) | held control |
| syn_smooth | synthetic (W108) | smooth LINEAR control |
| syn_mixed | synthetic (new, W110) | 4 held bones + 1 smooth bone, one Action |

## 1. Preflight and provenance — AGENT-RUN (`results/verify/verify_report.json`, `365e09db…`)

- **W109 history.**
  - The W109 Worklog ends with its COMPLETE marker.
  - W109 manifest `e62e8c65…` (128 files) and W109 declaration `b6bf2404…`.
- **W108 / W107 / W106 history.**
  - W108 manifest `bffcd9ca…` (137 files) and W108 declaration `0509a38c…`.
  - W107 manifest `20fcb1b9…` (183 files).
  - W106 manifest `f3003fe5…` (1,424 files).
  - Every listed file matches, with the recorded self-log exceptions.
- **Code identity.**
  - Exporter `src/blender/export.py` is `521a63d9…`.
  - The reused W106–W109 scripts equal the W109 declaration.
  - `render_runtime_heading_rig_blender.py` (`ea7d4565…`) is not in the W109 declaration and is bound here.
- **Input copies.** Byte copies were made into `results/inputs/`, each equal to its manifest SHA:
  - W107 V1 FBX and source `.blend` per real case;
  - W108 V1 FBX and `input.blend` per synthetic control;
  - the W108 copy of the W106 synthetic N0 file, used for the mixed build.
- **Re-check at the end.** The analysis and the manifest stage confirmed that every W106–W109 manifest file is
  still unchanged.

## 2. Frozen declaration — `results/declare/declaration.json`, SHA **`223b8c81ab345507ee5e60cbceca70fdad2e03bf97236afe87d94932a3ec049d`**

The declaration has `outcomes_read: false`. It was written before the mixed Action of this run was built, and
before any W110 import or adapter run.

**Aborted attempt 1.** This is v2. Declaration v1 (`673e66f8…`) is kept under `attempt1_aborted/` with a README.
- **How far it got.** Attempt 1 completed verify, options, declare, produce (run 1) and encode (run 1).
- **Why it stopped.** Encode then refused its own output. The consumer-input guard banned the bare substring
  `w108`, and the legitimate take names `root|W108 SYN step CONSTANT` / `root|W108 SYN smooth LINEAR` contain it.
  This was a false positive: the file held only take names, control/meta FBX paths and sample grids.
- **What was seen before v2.** No import or adapter had run. Only producer-side information had been seen (the
  source census, construction validity and producer counts).
- **The fix.** The guard was narrowed to a structural whitelist: declared keys only; files must be
  `control.fbx`/`meta.fbx` under the encode directory; no `.blend` / census / declaration / source / `.npz`
  reference.
- **What changed between v1 and v2.** Only the host-script SHA (`fa65c949…` → `1b08af97…`) and the regenerated
  verify/options file SHAs. The options content is byte-identical. Every rule, gate, case, grid and the other
  three script SHAs are equal; this was checked programmatically. Attempt-1 outputs are covered by the manifest.

**Bound script SHAs** (all unchanged at the end):

| Script | SHA |
|---|---|
| contract | `fe02b9fe…` |
| consumer adapter | `a6c1aa82…` |
| Blender stage | `67818230…` |
| host | `1b08af97…` |

The declaration also binds the ten reused scripts and the exporter.

**Producer rule** (source-informed, producer only):
- Each selected-take FBX curve (Model M, Lcl property P) maps to the source F-curves of the same item and transform
  group (`location` / `rotation_*` / `scale`).
- The flag is set from their uniform interpolation:
  - CONSTANT → `0x6002`;
  - LINEAR → keep `0x6104`;
  - no source F-curve (undriven) → keep `0x6104`;
  - anything else → refuse.
- CONSTANT-next `0x6102` is never used.

**Consumer mapping rule** (`w110_fbx_consumer_adapter.py`):
- **FBX side.** AnimationCurve -OP `d|X/Y/Z`→ AnimationCurveNode -OP `Lcl Translation/Rotation/Scaling`→ Model.
  The curve node must be OO-connected to the single layer of the named stack.
- **Blender side.**
  - A `Null` Model maps to the object of the same name.
  - A `LimbNode` maps to the same-named pose bone in exactly one armature, whose parent must equal the FBX parent
    Model (none if that parent is the armature Null).
  - The Action is `"<id>|<stack>"`, with exactly one slot.
- **Group directive.** Each group's directive is the consensus of its curves' decoded flags. The groups are
  location ← T, rotation ← R and scale ← S (separability checked from the FBX Model properties). An item is
  *coupled* if it has non-default pivots, offsets or geometric transforms; its groups then need the item's
  consensus.
- **Unresolved, never changed.** The following are counted as unresolved and left unchanged:
  - disagreements;
  - CONSTANT-next, CUBIC or unknown flags;
  - per-key mixtures;
  - duplicate channels (the importer would use only the first curve);
  - missing or ambiguous objects, bones, Actions, slots or F-curves;
  - F-curves claimed twice.

**Conditions** (each a fresh factory-empty scene plus the default import):

| Condition | Definition | Role |
|---|---|---|
| control_U | Unmodified V1, untouched | Baseline (P0 vs W108/W107) |
| control_A | Unmodified V1 + adapter | Consumer alone |
| meta_U | Metadata FBX, untouched | Producer alone |
| meta_A | Metadata FBX + adapter | The tested route |
| meta_K | Metadata FBX + blanket CONSTANT on the selected Action | Harness negative control, not the adapter |

**Grids:**
- the W108 integer + quarter-frame grid: 277 / 461 / 793 / 97 / 97 / 97 samples;
- plus a predeclared near-boundary probe `t = f + 0.99` for every transition interval: 23 / 23 / 198 / 8 / 8 / 8.

**Gates** (historical, not tuned):

| Gate | Threshold | Origin |
|---|---|---|
| Recovered and preserved | ≤ 1 mm world error at all samples | W108 |
| Integer | ≤ 1 mm | W107 |
| Repeat / identity | ≤ 1e-9 m | W109 |
| Harm (negative controls) | > 1 cm | W109 discriminable |
| Source fidelity | bitwise | — |

**Mixed construction** (frozen in `MIXED_SPEC`):
- A duplicate of the W106 `SYN baseline` Action: key times, values and slot identical.
- `base`, `upperarm_l`, `lowerarm_l` and `hand_l` are CONSTANT (held); `sibling` is LINEAR (smooth, with its
  existing missing key at 13).
- It is exported once per produce run through W107's V1 call.
- Component points:
  - **held** = heads and tails of the four held bones plus sibling's head;
  - **smooth** = sibling's tail. Base is exact under both the corrected and the blanket conditions, so the error
    there is attributable to sibling.

**CASE rule** (`decide_case`, frozen; D-checks come first):
- **D** if any of these fails: preflight, encoding, isolation, adapter integrity, sources, repeat, originals or P0.
- **A** requires all of: complete coverage on every case; held recovered on the 3 real cases and syn_step; smooth
  preserved; mixed held and smooth within 1 mm; controls discriminating; attribution.
- **B** if no held case is recovered, or no case is fully mapped, while the controls are valid.
- **C** otherwise.

## 3. Producer — metadata encoding — AGENT-RUN (`encode/encode_report.json` `f494896d…`, repeat `3750dff8…`)

| Case | Selected stack | Curves | CONSTANT / LINEAR (driven) / undriven | Bytes differing (all in target spans) | File bytes (unchanged) | Selected-stack keys |
|---|---|---:|---|---:|---:|---:|
| complete_control | `root\|W91 complete_control FIXED` | 468 | 24 / 0 / 444 | 48 | 9,509,612 | 32,760 |
| arm_natural_most | `root\|W91 arm_natural_most FIXED` | 468 | 24 / 0 / 444 | 48 | 2,487,276 | 54,288 |
| real_clip | `root\|W91 real_clip FIXED` | 468 | 24 / 0 / 444 | 48 | 1,809,564 | 93,132 |
| syn_step | `root\|W108 SYN step CONSTANT` | 54 | 15 / 0 / 39 | 30 | 102,380 | 1,350 |
| syn_smooth | `root\|W108 SYN smooth LINEAR` | 54 | 0 / 15 / 39 | 0 (meta = control bytes) | 102,396 | 1,350 |
| syn_mixed | `root\|W110 SYN mixed` | 54 | 12 / 3 / 39 (= declared expectation) | 24 | 102,364 | 1,350 |

What the producer was working from:
- **Real sources.** The real source Actions animate only the armature object's rotation and 7 bones' rotations. That
  is 8 items × 3 Lcl Rotation curves = 24 CONSTANT curves. The other 444 curves are undriven (constant values) and
  stay `0x6104`.
- **Unsupported source paths.** None in any case.
- **Validity checks.** Every meta file satisfies all of the following:
  - same length;
  - differences only inside the target value spans;
  - masked canonical tree equal to the control;
  - FBX-side map (without flags) unchanged;
  - targets decode as CONSTANT standard;
  - every other curve's flags unchanged;
  - selected-stack key counts and RefCount unchanged;
  - other stacks unchanged.
- **No-op identity.** The patcher run with the source value reproduces the input bytes.

**Producer key identity (diagnostic).** The mixed V1 selected-take key arrays were compared with the W108 V1 files:
- 45/54 curves are exactly equal to their expected reference: syn_step for held bones and root, syn_smooth for
  sibling.
- sibling's 9 curves differ from syn_smooth's by at most 3.05e-4 (Lcl Rotation, degrees) and ≤ 1.2e-7
  (translation/scale). This is float noise from the exporter's local-matrix computation, because sibling's parent
  `base` is held in the mixed file but LINEAR in syn_smooth.
- Key times are equal.

## 4. Correspondence and coverage — AGENT-RUN (adapter plan accounting, identical in both runs)

| Case | FBX curves mapped / total | Groups (CONSTANT / LINEAR) | Items (coupled) | Imported target F-curves mapped / total | Unresolved | Claimed twice | Per-axis 1:1 | F-curves changed (meta_A) | Keys changed |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| complete_control | 468 / 468 | 156 (8 / 148) | 52 (0) | 519 / 519 | 0 | 0 | 0 | 31 | 2,170 |
| arm_natural_most | 468 / 468 | 156 (8 / 148) | 52 (0) | 519 / 519 | 0 | 0 | 0 | 31 | 3,596 |
| real_clip | 468 / 468 | 156 (8 / 148) | 52 (0) | 519 / 519 | 0 | 0 | 0 | 31 | 6,169 |
| syn_step | 54 / 54 | 18 (5 / 13) | 6 (0) | 59 / 59 | 0 | 0 | 0 | 20 | 500 |
| syn_smooth | 54 / 54 | 18 (0 / 18) | 6 (0) | 59 / 59 | 0 | 0 | 0 | 0 | 0 |
| syn_mixed | 54 / 54 | 18 (4 / 14) | 6 (0) | 59 / 59 | 0 | 0 | 0 | 16 | 400 |

How to read the table:
- **Real cases.** The 31 changed F-curves are the root object's `rotation_euler` (3) plus `rotation_quaternion`
  (4) of each of the 7 driven bones.
- **CONSTANT set.** The adapter's CONSTANT curve set equals the producer's CONSTANT set on every case. On
  control_A it is empty for all cases (all 156/18 groups LINEAR, 0 changed).
- **Parser agreement.** The adapter (add-on parser) and the host (independent W108 reader) computed identical
  flag-free map digests for every file.
- **Per-axis correspondence.** Per-axis 1:1 is 0 by construction: Euler XYZ → quaternion WXYZ, plus rest-frame
  mixing. The supported unit is (item, transform property).

## 5. Consumer isolation — AGENT-RUN + CODE INSPECTION

- **Static (focused test).** The adapter imports only `__future__`, `hashlib`, `json`, `pathlib`, `bpy`,
  `bpy_extras`, `numpy` and `io_scene_fbx`, with no repository module. Its body contains no `.blend`, census,
  declaration or source token, and no key, handle or curve creation/removal. It has exactly one `foreach_set`,
  and its target is `"interpolation"`. `recover()` has the signature `(fbx_path, stack_name)`.

  The consume-stage functions contain no declaration, census, source, `.blend` or `open_mainfile` reference.
- **Input.** `consumer_input.json` (`4e7ce9d9…`; repeat `3dc140c8…`) passed the structural whitelist.
- **Runtime.** A Python audit hook was armed only around `recover()`. On all 24 adapter runs it recorded exactly
  3 `open` events, all of the FBX file (two SHA reads and one parse), and no other file access. The FBX was
  unchanged across the read, there was no escape-identifier mismatch, and the resolved Action equals the selected
  imported Action.
- **Integrity** (before/after snapshots, all 24 runs):
  - selected-Action key fields other than interpolation are unchanged;
  - the changed-curve count equals the adapter's count;
  - every other Action is bitwise unchanged (15 / 3 / 1 / 1 / 1 / 1);
  - the Action/slot inventory, objects (type, rotation mode, `matrix_basis`, parent), rest matrices, bone counts,
    pose rotation modes, fps/range, filepath and libraries are all unchanged.

## 6. Evaluated motion — AGENT-RUN (`analysis/analysis_report.json`, `fa317f76…`)

Max |coordinate| world head/tail error vs the source, on 51 / 5 common source bones. The importer's `_end` leaf
bones are excluded (W108 convention).

**P0 (all bitwise / 0.0).**
- Fresh source samples equal W108 `source.npz`.
- control_U equals W108 `V1.npz`, and equals W107 `V1.npz` at the common times.
- control_U subframe and integer maxima equal the W108 values.
- Source digest, slot, timing, key frames and interpolation are as declared.

### 6.1 Real Actions (reported separately; benchmark_detector_observation ×2, real_animcv_observation ×1)

| Case | Condition | All samples | Integer | Subframe (+.25 / +.50 / +.75) | Transition / quiet | Near-boundary f+0.99 | Subframes > 1 mm | Object rot. (subframe) |
|---|---|---:|---:|---|---|---:|---|---:|
| complete_control | control_U = control_A = meta_U | 0.764 m | 0.95 µm | 0.217 / 0.417 / 0.594 m | 0.594 m / 0.95 µm | 0.764 m | 69 / 207 | 12.3° |
| | **meta_A** | **1.07 µm** | 0.95 µm | 0.95 µm | 0.95 µm / 0.95 µm | 1.07 µm | 0 / 207 | 3e-5° |
| | meta_K | 0.95 µm | 0.95 µm | 0.95 µm | | 0.95 µm | 0 | 3e-5° |
| arm_natural_most | control_U = control_A = meta_U | 1.224 m | 1.31 µm | 0.533 / 0.960 / 1.071 m | 1.071 m / 1.31 µm | 1.224 m | 69 / 345 | 17.0° |
| | **meta_A** | **1.43 µm** | 1.31 µm | 1.43 µm | | 1.25 µm | 0 / 345 | 3e-5° |
| | meta_K | 1.31 µm | | | | | 0 | |
| real_clip | control_U = control_A = meta_U | 0.489 m | 1.66 µm | 0.187 / 0.354 / 0.433 m | 0.433 m / — (no quiet) | 0.489 m | 594 / 594 | 18.1° |
| | **meta_A** | **1.66 µm** | 1.66 µm | 1.65 µm | | 1.61 µm | 0 / 594 | 6e-5° |
| | meta_K | 1.66 µm | | | | | 0 | |

- **What the near-boundary probe adds.** It shows the full size of the LINEAR-ramp loss just before each held
  change: 0.76 / 1.22 / 0.49 m, versus the quarter-frame maxima 0.59 / 1.07 / 0.43 m.
- **Why blanket CONSTANT is harmless here.** On the real Actions it is not harmful, because every animated source
  channel is held. W108 already showed this. These cases therefore cannot discriminate metadata-driven from blanket
  correction; only the smooth and mixed cases do.

### 6.2 Synthetic controls (separately labelled; never pooled with real)

| Case | Condition | All samples | Integer | Subframe (+.25 / +.50 / +.75) | Near-boundary | Subframes > 1 mm |
|---|---|---:|---:|---|---:|---|
| syn_step | control_U = control_A = meta_U | 0.327 m | 0.77 µm | 0.081 / 0.159 / 0.246 m | 0.327 m | 24 / 72 |
| | **meta_A** | **0.77 µm** | 0.77 µm | 0.77 µm | 0.72 µm | 0 / 72 |
| | meta_K | 0.77 µm | | | | 0 |
| syn_smooth | control_U = control_A = meta_U = **meta_A** | **0.512 mm** | 0.80 µm | 0.377 / 0.512 / 0.388 mm | 18 µm | 0 / 72 |
| | meta_K (blanket CONSTANT) | **112.9 mm** | 0.80 µm | 28.8 / 57.6 / 86.0 mm | 112.9 mm | 72 / 72 |

The syn_smooth meta file is byte-identical to the control: its source is LINEAR, so nothing is encoded. The adapter
changes 0 curves on it.

## 7. Mixed held/smooth Action and negative controls — AGENT-RUN

- **Construction.** The construction is valid:
  - key data equals `SYN baseline` except interpolation;
  - per-bone interpolation is as declared;
  - key frames are `1, 4, …, 25`, with sibling missing 13;
  - the base input is unchanged;
  - the export left the Action and the scratch file unchanged.
- **Source semantics by evaluation.** The held component is exactly held (subframe minus floor-integer = 0.0). The
  smooth component changes in every interval (minimum 11.2 mm).

| Condition | Held component (incl. f+0.99) | Smooth component | All samples | Integer | Channel-function error (diagnostic) |
|---|---:|---:|---:|---:|---|
| control_U = control_A = meta_U (all LINEAR) | **327 mm** (harmed) | 129 mm | 327 mm | 0.77 µm | held bones vs floor 0.045–0.223; sibling vs lerp 3e-8 |
| **meta_A (FBX-metadata adapter)** | **0.77 µm** | **0.509 mm** | 0.509 mm | 0.77 µm | held bones 0.0; sibling 3e-8 |
| meta_K (blanket CONSTANT) | 0.77 µm | **60.5 mm** (harmed) | 60.5 mm | 0.77 µm | held 0.0; sibling vs lerp 0.098 |

- **Discrimination.** In both directions, against the declared 1 cm:
  - all-LINEAR harms the held component (327 mm);
  - blanket CONSTANT harms the smooth component (60.5 mm);
  - the metadata route keeps both within 1 mm.
- **Residual on the smooth component.** The 0.51 mm residual is the same order as W108's smooth control. It is the
  representational difference between source per-component quaternion interpolation across sparse keys and
  linear interpolation between baked integer-frame samples. Interpolation semantics are reproduced exactly at the
  channel level (lerp error 3e-8).

## 8. Qualitative review — AGENT-RUN (`results/figures/`, presentation only)

- **`mixed_curve_example.png`** (frames 8.5–17.5):
  - the held hand_l tail is a step function in the source and the adapter;
  - the untouched import ramps through each hold;
  - on sibling's tail, the adapter follows the source's smooth descent, including the small jumps where held
    `base` changes;
  - blanket CONSTANT turns sibling into a staircase.
- **`real_pose_transition.png`** (complete_control, `t = 21.75`, the worst untouched main-grid sample, 0.594 m):
  - the untouched import's legs and pelvis are visibly displaced toward the next held pose;
  - the adapter pose overlays the source (error 0.8 µm).

Only these two views were inspected. This is not a claim of visual product readiness.

## 9. Repeat — **FAILED (frozen gate)**, and a post-hoc characterisation

The repeat ran produce, encode and consume twice into fresh directories:

| Check | Result |
|---|---|
| All sampled arrays, all cases, all conditions (incl. near-boundary) | max difference **0.0** |
| Historical cases (5): control/meta FBX bytes, plans, census, directives | identical; per-case repeat OK |
| syn_mixed: changed-curve lists, census, directives | equal |
| syn_mixed: canonical FBX tree (timestamp skipped) and verbatim adapter plans | **differ** → `deterministic = false` → **CASE D** |

**Post-hoc, NOT a gate** (`scripts/w110_fbx_metadata_recovery_posthoc.py`, `posthoc/posthoc_report.json`
`14d35a56…`). It was written after the frozen analysis and does not change the CASE.
- **Positional differences between the two control files: 806.** They fall into these classes:
  - every Objects UID (Models, NodeAttributes, stacks, layers, curve nodes, curves);
  - `Documents/Document` UID;
  - the Connections that reference those UIDs;
  - curve-node / curve default-value positions, shifted by the UID-dependent element order;
  - `CreationTimeStamp`;
  - `SceneInfo` `Original|ApplicationNativeFile` (…`\produce\syn_mixed\input.blend` vs
    …`\produce_repeat\syn_mixed\input.blend`).
- **Keyed by identity, everything is equal.** Each object was keyed by name / connection-derived identity, curves
  by (take, model, property, channel). On that basis all 164 objects and all 198 connections are equal, and every
  curve key array is equal. The meta files show the same result.
- **Relabelled records are equal.** UID-relabelled adapter plans (control_A, meta_A) and producer directives are
  equal across runs.
- **Mechanism (CODE INSPECTION).** `io_scene_fbx/fbx_utils.py` `_key_to_uuid` uses `hash(key)` for non-integer keys.
- **Why the five historical cases were unaffected.** They reuse frozen FBX files.

## 10. CASE / verdict

**CASE D — INVALID / STOPPED** (`decide_case`, reason "fresh repeat disagrees"). The only false validity input is
`deterministic`.

For transparency, every other recorded decision input evaluated true:
- preflight, encoding, isolation, adapter integrity, sources, originals;
- P0 (6/6) and coverage (6/6);
- recovered (3 real cases + syn_step);
- smooth preserved, mixed held / mixed ok;
- controls discriminate;
- attribution.

Per the predeclared meaning of D, this Worklog draws **no viability conclusion**. In particular, it does not state
that FBX-embedded metadata is sufficient for a consumer-only recovery, even on Blender 5.1. The observations in
§3–§9 are recorded as evidence only.

## 11. Negative findings (preserved)

- The repeat criterion fails for a freshly exported FBX from the stock Blender exporter. Its UIDs and `SceneInfo`
  source path are not reproducible across processes, so a UID-keyed mapping record of a freshly exported file is
  not repeatable as written.
- The importer ignores the flags again (meta_U ≡ control_U on all six cases), and the stock round trip still loses
  holds by 0.33–1.22 m (near-boundary) without the scratch adapter.
- No per-axis FBX-curve → F-curve correspondence exists through this importer (0 of 468 / 54). Semantics that differ
  between axes of one Lcl property, or that combine within a coupled item, would be unresolved by construction.
- The mixed smooth component keeps a 0.51 mm residual. This is a baked-integer representation effect, not an
  interpolation-flag effect.
- Blanket CONSTANT does not harm the real Actions, because all their animated channels are held. The real cases
  alone cannot distinguish metadata-driven correction from blanket correction.
- Attempt 1 was aborted by an over-broad lexical guard in this Batch's own code (§2).

## 12. Unresolved uncertainties

- **Reproducibility of the mixed export.** Whether a fixed Python hash seed or a UID-independent repeat criterion
  would yield a reproducible mixed export. Not tested, deliberately, after the outcome.
- **Other consumers.** Behaviour on other consumers (Autodesk SDK, game engines), other Blender versions, CUBIC or
  CONSTANT-next sources, per-key mixed flags, and non-rotation driven channels (location / scale, non-uniform scale).
- **Coupled items.** Coupled items: none occurred, so that path is exercised only by synthetic in-test trees.
- **Remaining defects.** Multi-take export, `frame_end` 250, the 50→30 fps real-clip timebase, and V1 key/file
  cost: unchanged and unaddressed.
- **Mechanism of importer flag discard.** Still static inspection (W109), not instrumentation.
- **No independent replay.** No reviewer replayed these Main-local artifacts. Everything in §1 and §3–§9 is
  AGENT-RUN; the mechanism statements are CODE INSPECTION.

## 13. Tests

| Run | Result | Output |
|---|---|---|
| Focused, after declaration v2, before produce | 19 passed, 1 skipped (stored-output test) | `results/w110_tests_predeclare_run.xml` |
| Focused, final | **20 passed** | `results/w110_tests_final.xml` |
| Regression: `test_w109_fbx_constant_import.py`, `test_w108_fbx_subframe.py`, `test_w107_fbx_ownership.py`, `test_w106_action_editability.py`, `test_blender_executor.py` | **63 passed** | `results/regression_w106_w109_blender_executor.xml` |

`tests/test_w110_fbx_metadata_recovery.py` covers the following:

| Area | What is tested |
|---|---|
| Flags and producer | Flag classes (`0x6102` = CONSTANT-next); producer flag |
| FBX map | Connection graph, stack/layer refusal, duplicate channels, non-transform properties, separability |
| Planning | Property-level planning; coupled-item consensus; unresolved paths (CONSTANT-next, CUBIC, missing bone, parent mismatch, missing/extra F-curve, within-group disagreement) |
| Producer rule | Source path parsing; directives (CONSTANT / LINEAR / UNDRIVEN / UNSUPPORTED / UNMATCHED) |
| Digests and grids | Timestamp-skipping digest; near-boundary grid |
| Metrics | Channel-function and component metrics |
| CASE mapping | A / B / C / D precedence |
| Isolation | Adapter isolation (static); consume-stage source-freedom |
| Consumer input | Whitelist, including historical take names |
| Real data | Real W107 V1 map completeness and in-place patchability |
| Stored outputs | Declaration / analysis consistency |

Not run:
- The broad production / training suite. No tracked or shared file changed (`shared_tracked_changes = []`).
- Any LabServer test.

## 14. Execution record, failed attempts and unexecuted work

**Pre-declaration development.** All runs below were outside the evidence base (`C:\Users\dna10\animcv-w110-explore`,
throwaway, not committed).
- **Source-structure inspection.** The unmodified W109 input copies of complete_control, real_clip and syn_step
  were inspected for F-curve data paths and rotation modes. This established that only object and bone rotation
  are animated.
- **FBX-side map check.** The FBX-side map was computed on the five unmodified historical V1 files (468 / 54 curves,
  all OK, all models separable).
- **Plumbing smoke on a non-declared 2-bone rig.** Exported, patched, imported, adapter run: 29/29 F-curves mapped,
  4 changed.
- **Full dry run on another non-declared 2-bone rig.** produce (build path), encode, consume and the analysis case
  function, through a fake declaration. Its smooth bone showed an 11 mm residual from that rig's deliberately
  large, non-unit quaternion keys. It was not a declared case, and nothing was changed in response.
- **Code changes before declaration v1.**
  - The mixed-case helpers were generalized from the literal `sibling` to `smooth_bones` / `missing_key`.
  - The figures moved to a separate presentation-only script.
  - The mixed key-identity diagnostic gained numeric differences.

**Attempt 1.** Aborted at encode (§2).

**v2 runs.** verify, options, declare, produce ×2, encode ×2, consume ×2 and analyze all exited 0. Every import
returned `FINISHED`.

**After the frozen analysis.**
- The post-hoc script was added; its first draft keyed non-selected-take curves by UID. That draft output was
  deleted and regenerated after the keying was fixed.
- The figures script was run.
- No declared script changed after declaration v2; all 15 bound SHAs are equal at the end.

**Not executed:**
- any production / importer / exporter / add-on change;
- V0 FBX;
- CONSTANT-next encoding;
- a fixed-hash-seed repeat;
- non-Blender consumers;
- other Blender versions;
- renders (matplotlib stick and trajectory views were used instead);
- LabServer;
- the private real-clip frames (not read).

## 15. Storage, cost and resources

- **File size and keys.** FBX file size and key counts are unchanged by the producer edit: 2 bytes per targeted
  curve; meta = control length. The scratch route adds no keys.
- **Runtime per run:**

  | Stage | Time |
  |---|---|
  | produce | ≈ 1–2 s |
  | encode | ≈ 2–3 s |
  | consume (30 imports) | ≈ 23–24 s |
  | verify / declare / analyze / manifest | a few seconds each |
  | adapter per real import | 0.09–0.58 s |
  | adapter per synthetic import | ≈ 0.01 s |

- **Disk.** Outputs are 179 MB in `C:\Users\dna10\animcv-w110\`, not committed.
- **Compute.** Main only, CPU only; no GPU, network or package change. This was within the declared cap.

## 16. Reproduction (Git Bash on Main, repo root)

```
PY=/c/Projects/AnimCV/.venv_build_full/Scripts/python.exe; BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
AD="C:/Program Files/Blender Foundation/Blender 5.1/5.1/scripts/addons_core/io_scene_fbx"
O=C:/Users/dna10/animcv-w110; R=$O/results; H=186ed882f5d63e82d41682a295ed029ad638d513
S=223b8c81ab345507ee5e60cbceca70fdad2e03bf97236afe87d94932a3ec049d
W="--w106 C:/Users/dna10/animcv-w106 --w107 C:/Users/dna10/animcv-w107 --w108 C:/Users/dna10/animcv-w108 --w109 C:/Users/dna10/animcv-w109"
B=("$BL" -b --factory-startup -t 2 --python-exit-code 1)
$PY -X utf8 scripts/w110_fbx_metadata_recovery.py verify --base $O $W --lock C:/Users/dna10/animcv-automation/state/active_batch.json --blender "$BL" --addon-dir "$AD" --starting-head $H --branch arch/single_frame_first
"${B[@]}" -P scripts/w108_fbx_subframe_blender.py -- options --out $R/options.json
$PY -X utf8 scripts/w110_fbx_metadata_recovery.py declare --base $O --w108 C:/Users/dna10/animcv-w108 --w109 C:/Users/dna10/animcv-w109 --verify $R/verify/verify_report.json --options $R/options.json --blender "$BL" --starting-head $H
for X in "" _repeat; do
  "${B[@]}" -P scripts/w110_fbx_metadata_recovery_blender.py -- produce --declaration $R/declare/declaration.json --declaration-sha $S --out-dir $R/produce$X
  $PY -X utf8 scripts/w110_fbx_metadata_recovery.py encode --declaration $R/declare/declaration.json --declaration-sha $S --produce-report $R/produce$X/produce_report.json --out-dir $R/encode$X
  CS=$(sha256sum $R/encode$X/consumer_input.json | cut -d' ' -f1)
  "${B[@]}" -P scripts/w110_fbx_metadata_recovery_blender.py -- consume --consumer-input $R/encode$X/consumer_input.json --consumer-input-sha $CS --out-dir $R/consume$X
done
$PY -X utf8 scripts/w110_fbx_metadata_recovery.py analyze --declaration $R/declare/declaration.json --declaration-sha $S --produce-dir $R/produce --encode-dir $R/encode --consume-dir $R/consume --produce-repeat-dir $R/produce_repeat --encode-repeat-dir $R/encode_repeat --consume-repeat-dir $R/consume_repeat $W --verify $R/verify/verify_report.json --out-dir $R/analysis
$PY -X utf8 scripts/w110_fbx_metadata_recovery_posthoc.py --results $R --out-dir $R/posthoc
$PY -X utf8 scripts/w110_fbx_metadata_recovery_figures.py --declaration $R/declare/declaration.json --produce-dir $R/produce --consume-dir $R/consume --out-dir $R/figures
$PY -X utf8 -m pytest tests/test_w110_fbx_metadata_recovery.py -q
$PY -X utf8 scripts/w110_fbx_metadata_recovery.py manifest --base $O $W --starting-head $H --declaration-sha $S
```

How the stages behave on a re-run:
- Every output stage refuses an existing directory.
- verify requires a Batch-110 RUNNING lock and only new W110 files in the tree.
- On a fresh base, the declaration hash changes only through recorded paths and SHAs.
- A fresh mixed export will again carry new UIDs (§9).

**Evidence manifest.** `C:\Users\dna10\animcv-w110\evidence_manifest.json`, SHA
**`e46329f5f2e2d11b016f06ffd93f3a3beb5ffe39579f5de9ba26e318f8215fd4`**:
- 337 files, including `attempt1_aborted/`;
- `all_originals_unchanged: true`;
- it excludes itself and `manifest_stdout.txt`.

Other hashes:

| File | SHA |
|---|---|
| `verify_report` | `365e09db…` |
| `produce_report` | `cbab266f…` |
| `encode_report` | `f494896d…` |
| `consume_report` | `7bf172bd…` (repeat `ab574f6e…`) |
| `analysis_report` | `fa317f76…` |
| `posthoc_report` | `14d35a56…` |
| `figures.json` | `3a3c85cd…` |

## 17. Changed files, final source revision, environment

New files only:

| File | Role | SHA (LF bytes, as executed) |
|---|---|---|
| `scripts/w110_fbx_metadata_recovery_contract.py` | Cases, mixed spec, producer rule, timestamp-skipping digest, grids, metrics, tolerances, CASE rule | `fe02b9fe…` |
| `scripts/w110_fbx_consumer_adapter.py` | Consumer-only adapter (FBX map, plan, apply, `recover`) | `a6c1aa82…` |
| `scripts/w110_fbx_metadata_recovery_blender.py` | Blender `produce` (producer / ground truth) and `consume` (isolated consumer harness) | `67818230…` |
| `scripts/w110_fbx_metadata_recovery.py` | Host verify / declare / encode / analyze / manifest | `1b08af97…` |
| `scripts/w110_fbx_metadata_recovery_posthoc.py` | Post-hoc repeat characterisation (not a gate) | `c5dd8fda…` |
| `scripts/w110_fbx_metadata_recovery_figures.py` | Presentation-only figures | `de71f965…` |
| `tests/test_w110_fbx_metadata_recovery.py` | Focused tests | `0a7dfaea…` |
| this Worklog | | |

All W110 files were written with LF line endings.

Nothing else changed:
- no `src/` file, exporter, importer, add-on, rig, source Action, scene setting, W106–W109 artifact or historical
  Worklog;
- not `docs/next_batch.md`;
- no automation file.

Final source revision: the commit adding this Worklog on `arch/single_frame_first`, directly after `186ed88`. The
COMPLETE marker is added in a separate final commit.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 110
