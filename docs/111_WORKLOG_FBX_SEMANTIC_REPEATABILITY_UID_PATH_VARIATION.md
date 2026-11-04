# Worklog — FBX Semantic Repeatability Across Exporter UID and Source-Path Variation (2026-11-04)

## Human Assessment

**Verdict: CASE A — BOUNDED SUPPORT, under the frozen rule.** This is a Blender 5.1.0 / io_scene_fbx 5.15.0
diagnostic feasibility statement only. It is not architecture success, production readiness or interoperability.
**W110 stays CASE D**; this Batch does not reinterpret it.

The question was whether the W110 fresh-repeat failure is only a serialization-identity artifact, or a real barrier to
a source-free, metadata-driven consumer contract. On the tested workflow it behaved as a serialization-identity
artifact:

- **SUCCESS / SUPPORTED (AGENT-RUN, prospective, frozen before any W111 export).**
  - **The variation really occurred.** Five producer processes per run were launched: 2 Python hash seeds × 2
    absolute source paths, plus an independent same-cell repeat. Two full independent runs were made. The cell's hash
    seed changed **every** element UID (10,150 / 10,150 on complete_control; 164 / 164 on the synthetic files) and
    reordered most Objects and Connections. The path changed only the embedded source path string. All 60 control
    FBX files are byte-distinct.
  - **Every variant was semantically identical.** A strict canonicalizer drops only the predeclared volatile fields
    (UID references, `CreationTimeStamp`, the source-path string). Under it, every variant of every case (6 cases ×
    10 variants, control and metadata FBX) equals the reference with **0 differences** on full-content comparison.
  - **Consumer side unchanged across variants.** The unmodified W110 adapter produced UID-normalized plans identical
    across all variants, and identical on a second fresh import (A2). Imported scenes were digest-identical:
    selected-Action keys with and without interpolation, the other Actions, Action inventory, rig and timebase.
  - **Motion did not move.** Every sampled array of every condition differs from the reference variant by
    **0.0 m**. The W110 numbers reproduce exactly (§6).
  - **The canonicalizer discriminates.** All 14 negative and invariance controls on 2 cases × 2 runs were judged as
    declared. A post-hoc check on real data shows it reports exactly the producer-flagged curves between each control
    FBX and its metadata FBX (§7).
- **FAILURE / NOT FIXED.**
  - **The stock path still loses holds** (0.33–1.22 m) unless both the scratch producer edit and the scratch adapter
    are applied.
  - **The exporter is still non-deterministic at byte level:** UIDs, element order, timestamp and path. W111 shows
    that this is semantically inert for the tested files; it does not make the exporter reproducible.
  - **Other defects persist:** `frame_end` 250, multi-take export, the 50→30 fps real-clip timebase, and V1 key/file
    cost.
- **UNCERTAIN.** The canonicalizer accepts only the tested object classes: armature-only exports with Model,
  NodeAttribute, Stack, Layer, CurveNode and Curve. Meshes, skins, poses, materials and other exporters would be
  **rejected**, not handled (§11).
- **ALTERNATIVE EXPLANATIONS (status).**

  | Explanation | Status |
  |---|---|
  | Permissive canonicalizer hides real changes | Excluded as far as tested. Each of these was localized to exactly the expected record or explicitly rejected: flag, key value, key time, non-target-take curve, take swap, parent change, duplicate name, unresolved connection, UID leak. The control-vs-meta comparison (post-hoc) reports exactly the 24 / 15 / 12 / 0 flagged curves. |
  | Variants identical only because nothing varied | Excluded. All UIDs changed between seeds, and path strings changed between paths. The hash probes confirm that Blender saw each declared seed. |
  | Seed and path effects confounded | Separated. Path-only pairs had 0 UID / order changes. Seed-only pairs had identical paths. Same-cell repeats differ only in 3–4 timestamp bytes (§4). |
  | Consumer used source knowledge | Excluded as in W110. Consumer input is a structural whitelist; the audit shows 3 `open` events, all of the FBX, on every adapter run. The adapter is the unchanged W110 file. |
  | Consumer outcome depends on the consumer's own hash seed | Not observed. Consumers ran with W110's per-process random seed in two runs, and plans, scenes and motion were identical. This is a sample of two consumer processes only. |

## Original task scope (summary)

Batch 111 asked whether the W110 FBX-metadata recovery route survives controlled fresh-export variation. It also
asked whether a strict, prospective semantic identity can distinguish exporter-volatile identity from real animation
or rig changes, without patching the exporter and without converting W110's CASE D.

The prescribed design:
- a 2×2 hash-seed × source-path matrix, with a same-cell repeat, on the W110 mixed case and at least one real Action;
- replay of the other W110 cases;
- strict negative controls;
- the unchanged W110 consumer;
- an independent full rerun.

Only isolated `scripts/w111_*`, `tests/test_w111_*`, this Worklog and scratch outputs were allowed.

## Starting state and audited provenance — AGENT-RUN (`results/verify/verify_report.json`, `eaed125f…`)

- **Repository.**
  - Branch `arch/single_frame_first`, HEAD `5f9ef0b87c678b710a5c5d06aa2d69aa019af5a2`.
  - `docs/next_batch.md` blob `329fc5d4c8526b339d66c63c93b27b72bbb701a0`; both verified before any work.
  - After `git fetch`, `origin` equalled HEAD and the tree was clean.
  - Worker lock: batch 111, RUNNING, `start_commit` = HEAD. No other Blender process was running.
- **W110.**
  - The W110 Worklog ends with its COMPLETE marker.
  - Declaration SHA `223b8c81…ec049d` and evidence manifest SHA `e46329f5…8215fd4` both match the immutable references.
  - All 337 manifest files match. The manifest excludes only itself and `manifest_stdout.txt`; no other exception
    was needed.
- **W106–W109.** All manifests match: 1,424 / 183 / 137 / 128 files, with their recorded self-log exceptions.
- **Code and environment.** The following are equal to the W110 declaration:
  - the W110 scripts, the ten reused W106–W109 scripts and the exporter `src/blender/export.py` (`521a63d9…`);
  - the FBX add-on sources and the Blender exe SHA.
  - `scripts/w107_fbx_ownership.py` is additionally bound (imported read-only).
- **Inputs.** Byte copies of the six W110 source `.blend` files were written to the two declared scratch path roots.
  Each copy equals its W110 manifest SHA, and all were unchanged at the end.
  - complete_control, arm_natural_most, real_clip, syn_step and syn_smooth come from W110 `results/inputs/*_source.blend`.
  - syn_mixed comes from W110 run-1 `results/produce/syn_mixed/input.blend`.

  The two path roots:
  - `C:\Users\dna10\animcv-w111\srcpaths\A\<case>\source.blend`
  - `C:\Users\dna10\animcv-w111\srcpaths\B_alternate_location\<case>\source.blend`
- **Regimes.** Reused, non-holdout data, never pooled:
  - complete_control and arm_natural_most: `benchmark_detector_observation` (3DPW), real held source;
  - real_clip: GT-free `real_animcv_observation`, real held source;
  - syn_step, syn_smooth and syn_mixed: synthetic, separately labelled.

## Pre-declaration work (outside the evidence base)

- **Hash-seed plumbing probe.** Six throwaway Blender launches printing `hash('w111-probe')`.
  - Without `--python-use-system-env`, two `PYTHONHASHSEED=1` launches printed different hashes: Blender ignores
    the variable by default.
  - With the flag, equal seeds gave equal hashes and different seeds gave different hashes.
  - This fixed the producer launch below. No FBX was involved.
- **Canonicalizer development on single historical files.** Development used one file at a time:
  - the W110 run-1 mixed `control.fbx` and `meta.fbx`, and the W110 copy of W107 complete_control `V1.fbx`;
  - each was canonicalized and mutated against itself (controls).
  - No two files were compared under the W111 canonicalizer before the declaration.
  - This inventory showed that the files contain only the six object classes listed in §2.
- **Full plumbing dry run on a non-declared Action.** Run in `C:\Users\dna10\animcv-w111-explore\dev` (throwaway, not
  evidence, not committed) on W106 `SYN baseline` with a hand-edited development declaration.
  - It exercised verify, declare, cells, encode, consume, controls and analyze.
  - Only stage exit codes and plumbing fields were inspected (bindings, completeness, isolation, integrity,
    coverage, environment); the semantic and pair outcomes of that Action were not.
  - It found and fixed three bugs: an `int` in the digest walker, a role fall-through for an undeclared case, and a
    crash in `controls` when a file is missing.
  - Its take-dependent controls (NC4 / NC9 / NC10) errored, as expected, because that file has one take.

## 2. Frozen declaration — `results/declare/declaration.json`, SHA **`0ffa9d3cb51e45d31316ce1e429f0bb0b8d02951e7281601cd7c721bf4c29d89`**

The declaration has `outcomes_read: false`. It was written before any W111 export, encode, import, adapter run or
control. The focused tests ran after it and before the cells: 38 passed, 1 skipped (the stored-output test). All 19
bound script SHAs were unchanged at the end.

- **Hypothesis H111.** The W110 FBX interpolation metadata, plus the unchanged source-free W110 consumer mapping, is
  invariant to hash-seed UIDs, UID-induced order and the embedded source path. The canonicalizer still detects key,
  flag, take and hierarchy changes.
- **Matrix.** All six cases run the full matrix; the Batch's minimum was syn_mixed + complete_control.
  - Seeds: s1 = 1110111, s2 = 2220222.
  - Paths: pA and pB.
  - Cells, in execution order: s1_pA, s2_pB, s1_pB, s2_pA, s1_pA_r (the independent same-cell repeat).
  - Two runs, so 10 variants per case. The reference variant is run1 / s1_pA.
- **Producer launch.** `-b --factory-startup -t 2 --python-use-system-env --python-exit-code 1`.
  - The environment has every `PYTHON*` variable removed, then `PYTHONHASHSEED=<seed>` and `PYTHONNOUSERSITE=1`.
  - Export uses W110 `source_record` and `export_v1` (W107 V1 kwargs, recorded and checked per variant) on the
    source at the cell's path.
- **Consumer launch.** The W110 launch: no system environment, so a per-process random seed.
- **Producer edit.** The W110 rule and checks, unchanged: census-driven `0x6104 → 0x6002` on the CONSTANT-driven
  curves of the selected take. Producer counts must equal W110's per case.
- **Conditions** (factory-empty scene + default import):

  | Condition | Definition |
  |---|---|
  | U | untouched |
  | A | unmodified W110 adapter |
  | A2 | second fresh import + adapter |
  | K | blanket CONSTANT (harness negative control) |

  After each step a digest of the whole scene is taken.
- **Canonicalization (`w111_fbx_semantic_identity.py`).**
  - **Excluded fields (only these):**
    - `Objects/*` property 0, `Connections/C` properties 1–2 (parent 0 is kept as `ROOT`), and
      `Documents/Document` property 0 — all replaced by typed identities or a fixed token;
    - the whole `FBXHeaderExtension/CreationTimeStamp` element, which must occur exactly once;
    - the value of `SceneInfo` `Original|ApplicationNativeFile`, which must occur once and be an absolute `*.blend`
      path.
  - **Typed identity:**
    - named classes (Model, NodeAttribute, AnimationStack, AnimationLayer) are identified by (class, subtype, raw
      name bytes); a duplicate is **rejected**;
    - anonymous classes (AnimationCurveNode, AnimationCurve) are identified by (class, subtype, raw name, sorted
      outgoing typed connections to resolved identities), resolved to a fixpoint; a collision, orphan or unresolved
      element is **rejected**.
  - **Relations are checked:** one OO parent per Model, NodeAttribute→Model, Layer→Stack, CurveNode→(one Layer, one
    Model via OP), Curve→(one CurveNode via OP).
  - **Other rejections:** any other object class; dangling connections; any int64 scalar, or int64 array other than
    `KeyTime`, that carries an element/Document UID outside the declared positions (UID leak).
  - **Order:**
    - Objects are keyed by identity;
    - Connections form a sorted multiset;
    - every other node keeps file order;
    - every retained property value is compared in full bytes.
- **Gates.**
  - The W110 tolerances are unchanged: 1 mm recovered / integer, 1e-9 m identity, 1 cm harm, bitwise source.
  - The gate definitions are listed in the declaration under `gates`.
- **CASE rule** (`decide_case`).
  - **D** if any of these fails: preflight, environment fixed, completeness, encoding, consumer input clean, isolation,
    adapter integrity, sources, canonical injectivity, negative controls, repeat (the per-run aggregate decision inputs
    must be equal), originals.
  - **B** if any case shows a semantic, plan or scene difference, or a motion difference > 1 mm.
  - **A** if everything is identical and stable (≤ 1e-9 m), within the W110 gates with complete coverage, controls
    discriminate, the variation is realized and attribution is clean.
  - **C** otherwise.
- **Resource cap.**
  - Blender runs: options + 2 × (5 cells + 1 consume).
  - At most 1,200 s per run and 5 GB disk.
  - No GPU and no network.

## 3. Condition matrix and execution — AGENT-RUN

| Run | Cells (exit 0) | Wall time per cell | Encode | Consume (120 imports + sampling) | Controls |
|---|---|---|---|---|---|
| run1 | 5 / 5 | 3.30–3.36 s | valid, 30 variants | 113.6 s, exit 0 | ok |
| run2 | 5 / 5 | 3.29–3.46 s | valid, 30 variants | 113.8 s, exit 0 | ok |

Every variant was exported, encoded, imported and sampled under all four conditions (60 variants × 4 = 240 sampled
imports). Each run took about 2.7 min wall time, within the 1,200 s cap.

## 4. Variation realized and attribution (P1 / P5) — AGENT-RUN (`analysis_report.json` `99b30807…`, `pairs`)

**Hash probes.**
- Every cell's Blender saw its declared `PYTHONHASHSEED`, with no user site-packages.
- `hash('w111-probe')` was equal across all s1 cells (−3561638689648412293) and across all s2 cells, in both runs,
  and differed between s1 and s2.

The table gives raw differences between control FBX of the same case. Rows are identical in run1 and run2.

| Pair kind | UIDs changed (cc / anm / clip / each synthetic) | Object order positions differing | Connection order positions differing | Document UID | Timestamp | Source path |
|---|---|---|---|---|---|---|
| seed only (s1↔s2, same path) ×2 | 10,150 / 2,638 / 1,386 / 164 (= all) | 9,600 / 2,400 / 1,200 / 144 | 12,000 / 3,000 / 1,500 / 180 | changed | differs | same |
| path only (pA↔pB, same seed) ×2 | 0 | 0 | 0 | same | differs | **differs** |
| same cell (s1_pA ↔ s1_pA_r) | 0 | 0 | 0 | same | differs | same |
| seed and path ×2 | all | as seed only | as seed only | changed | differs | differs |
| cross-run, same cell (×5 cells) | 0 | 0 | 0 | same | differs | same |

- **Attribution.**
  - The seed accounts for every UID change, the Document UID and the UID-induced order.
  - The path accounts only for the `ApplicationNativeFile` string, plus 16 bytes of file length (pB's path is
    longer).
  - No interaction was observed: the seed-and-path pairs equal the union of the two effects.
- **Same-cell byte localization (post-hoc).** Same-cell control files have equal length. They differ in 4 bytes
  (run1) or 3 bytes (run2) per case, all inside the `CreationTimeStamp` byte span.
- **Distinct bytes.** All 10 control files per case are byte-distinct, so every comparison below is between
  different files.

## 5. Semantic identity (P2) — AGENT-RUN

- **Full canonical comparison** against run1 / s1_pA, for control and meta FBX of all 60 variants: **0 differences**.
  - No header section, Objects record or connection differed.
  - 0 rejections. Canonical injectivity held on all 120 files.
- **Accounting per file** (identical across variants):

  | Case | Objects (curves) | Connections | UID values excluded (object / connection / document) | Timestamp values excluded | Path values excluded | Object property values retained | Selected-take keys |
  |---|---|---|---|---|---|---|---|
  | complete_control | 10,150 (7,488) | 12,630 | 10,150 / 25,259 / 1 | 8 | 1 | 30,450 | 32,760 |
  | arm_natural_most | 2,638 (1,872) | 3,258 | 2,638 / 6,515 / 1 | 8 | 1 | 7,914 | 54,288 |
  | real_clip | 1,386 (936) | 1,696 | 1,386 / 3,391 / 1 | 8 | 1 | 4,158 | 93,132 |
  | syn_step / syn_smooth / syn_mixed | 164 (108) | 198 | 164 / 395 / 1 | 8 | 1 | 492 | 1,350 |

  - Every curve has a unique identity: 0 collisions.
  - Every element's retained subtree is compared in full: keys, flags, Properties70, defaults.
  - All other header sections are retained verbatim: GlobalSettings, Definitions, Takes, Documents apart from its
    UID, FileId, CreationTime and Creator.
- **Records.** Reference records (per-identity digests) are in `results/analysis/records/`. No non-reference record
  was written, because none differed.

## 6. Source-free consumer and motion A/B (P4) — AGENT-RUN

**Identical in every variant** (10 per case):
- adapter coverage of 468/468 FBX curves and 519/519 imported F-curves (real), or 54/54 and 59/59 (synthetic);
- group directives: real 8 CONSTANT / 148 LINEAR; syn_step 5 / 13; syn_smooth 0 / 18; syn_mixed 4 / 14;
- 0 unresolved, 0 claimed twice, 0 unmapped UIDs after normalization;
- F-curves changed: 31 / 20 / 0 / 16;
- adapter CONSTANT set equal to the producer CONSTANT set: 24 / 15 / 0 / 12 curves;
- A2 plan equal to the A plan.

**Isolation and integrity.**
- Every adapter run (120) recorded exactly 3 `open` events, all of the FBX.
- Integrity snapshots were unchanged apart from directed interpolation.
- Key data excluding interpolation of A, A2 and K equals that of U.

**Cross-variant stability.**
- Scene digests and final key-data digests are equal to the reference for every condition.
- The maximum difference of every sampled array from the reference variant is **0.0 m** for U, A, A2 and K, on the
  main and near-boundary grids.

### 6.1 Real Actions (reported separately; identical in all 10 variants)

| Case (regime) | U all samples | U near-boundary | A / A2 all samples | A integer | K |
|---|---:|---:|---:|---:|---:|
| complete_control (benchmark_detector_observation) | 0.764 m | 0.764 m | 1.07 µm | 0.95 µm | 0.95 µm |
| arm_natural_most (benchmark_detector_observation) | 1.224 m | 1.224 m | 1.43 µm | 1.31 µm | 1.31 µm |
| real_clip (real_animcv_observation, GT-free) | 0.489 m | 0.489 m | 1.66 µm | 1.66 µm | 1.66 µm |

### 6.2 Synthetic controls (separately labelled)

| Case | U all | A / A2 all | A integer | K all | Mixed held component (U / A / K) | Mixed smooth component (U / A / K) |
|---|---:|---:|---:|---:|---|---|
| syn_step | 0.327 m | 0.77 µm | 0.77 µm | 0.77 µm | — | — |
| syn_smooth | 0.512 mm | **0.512 mm** | 0.80 µm | 112.9 mm | — | — |
| syn_mixed | 0.327 m | **0.509 mm** | 0.77 µm | 60.5 mm | 327 mm / 0.77 µm / 0.77 µm | 129 mm / **0.509 mm** / 60.5 mm |

- **Reproduction of W110.** These values reproduce W110's exactly. The source samples of every variant equal the
  W110 `source.npz` / `source_nb.npz` bitwise (difference 0.0).
- **Smooth residual.** The 0.51 mm smooth residual is reported, not hidden. It is W110's baked-integer representation
  effect, below the historical 1 mm gate. Its worst sample is t = 10.5 (`figures.json`).
- **Real cases.** Blanket CONSTANT is harmless on the real Actions, because every animated real channel is held. Only
  the synthetic smooth and mixed cases discriminate metadata-driven from blanket correction, as in W110.

## 7. Negative and invariance controls (P3) — AGENT-RUN (`run1/controls` `e181e033…`, `run2/controls` `7202184c…`)

These controls ran on scratch copies of run1 and run2 s1_pA `meta.fbx`, for syn_mixed and complete_control. Every
mutated copy was written by the W111 writer and re-read; none was imported. Outcomes were identical in both runs and
in both cases.

| Control | Declared expectation | Outcome |
|---|---|---|
| NC0 re-serialize unchanged | EQUAL | EQUAL |
| INV1 bijective relabel of all UIDs + Document UID | EQUAL | EQUAL |
| INV2 reverse Objects and Connections order | EQUAL | EQUAL |
| INV3 change timestamp values and source path | EQUAL | EQUAL |
| NC1 one selected-take CONSTANT flag 0x6002→0x6104 | detected, that curve only | DETECTED_LOCAL |
| NC2 one key value +1e-3 | detected, that curve only | DETECTED_LOCAL |
| NC3 one key time +1 FBX tick | detected, that curve only | DETECTED_LOCAL |
| NC4 one curve node re-pointed to another take's layer | detected-local or rejected (collision) | REJECTED: NON_INJECTIVE_IDENTITY (the other take animates the same item/property) |
| NC5 one LimbNode parent → grandparent | that connection only | DETECTED_LOCAL |
| NC6 LimbNode renamed to another LimbNode's name | REJECTED duplicate | REJECTED: DUPLICATE_NAMED_IDENTITY |
| NC7 parent connection → non-existent UID | REJECTED unresolved | REJECTED: UNRESOLVED_CONNECTION |
| NC8 Document RootNode → armature Model UID | REJECTED UID leak | REJECTED: UNEXPECTED_UID_REFERENCE |
| NC9 one key value of a NON-selected take | detected, that curve only | DETECTED_LOCAL |
| NC10 swap take membership of two curve nodes | detected, only those nodes / curves | DETECTED_LOCAL (8 records) |

**Real-data discrimination (post-hoc, not a gate; `posthoc_report.json` `dc912039…`).** For each case, the reference
control FBX was compared with its metadata FBX under the canonicalizer. It reports exactly the producer-flagged curves
and nothing else: 24 / 24 / 24 / 15 / 0 / 12 differing records, equal to the producer CONSTANT set, with 0 header or
connection differences.

## 8. Historical replay diagnostics (declared as diagnostics, not gates)

- **Against W107 / W108.** The fresh reference control FBX of every historical case is canonically identical (0
  differences) to the W107 / W108 V1 FBX that W110 reused. Those files were exported earlier by other processes
  (uncontrolled seeds), from other paths.
- **Against W110 mixed.** The fresh mixed reference is identical to both W110 mixed controls.
- **W110 run vs repeat.** Compared under the W111 canonicalizer, the W110 mixed run and repeat are identical.
- **Status.** These are prospective-canonicalizer statements about historical files. They **do not** change W110's
  frozen repeat gate or its CASE D.

## 9. Qualitative review — AGENT-RUN (`results/figures/`, presentation only, `figures.json` `95a6ebda…`)

- **`mixed_variants.png`** (frames 8.5–17.5).
  - The adapter results of all 10 variants coincide; their spread is 0.0 m.
  - The held hand_l tail is a step function, matching the source.
  - The untouched import ramps through holds.
  - Blanket CONSTANT turns the smooth sibling tail into a staircase.
- **`real_pose_variants.png`** (complete_control at t = 21.75, the worst untouched held-transition sample).
  - The untouched pose is displaced by 0.594 m.
  - The adapter poses from run1/s1_pA and run2/s2_pB (different seed, path and run) overlay the source at 0.80 µm
    and are identical to each other.
- **`mixed_error_time.png`** (boundary / failure example, log scale, including the f+0.99 probes).
  - The untouched import fails the held component up to 0.33 m just before each held change.
  - Blanket CONSTANT fails the smooth component up to 60 mm.
  - The adapter's smooth residual peaks at 0.51 mm (t = 10.5), below the 1 mm line.

Only these three views were inspected. Matplotlib stick and trajectory plots were used; there were no renders. This is
no claim of visual product quality or game-engine interoperability.

## 10. CASE / verdict

**CASE A — BOUNDED SUPPORT** (`decide_case`).

| Inputs | Result |
|---|---|
| All 12 D-checks | true |
| Per case (all six): `semantic_identical`, `plans_identical`, `scene_identical`, `motion_stable`, `motion_gate`, `coverage` | true |
| `motion_material` | false |
| `controls_discriminate`, `variation_realized`, `attribution_clean` | true |

`repeat_ok` holds: the per-run aggregate decision inputs of run2 equal run1's.

**Scope of the statement.** On Blender 5.1.0 / io_scene_fbx 5.15.0, with the W107 V1 armature-only export, the six
W110 cases and this matrix, producer UID / order / path nondeterminism did not change:
- the semantic content of the exported FBX;
- the W110 producer metadata;
- the source-free W110 consumer mapping;
- the imported scene;
- the evaluated held/smooth motion.

The W110 repeat failure is therefore attributable here to serialization identity, which a strict identity-keyed
canonicalizer can separate from semantic content.

**What this does not establish:**
- a reproducible exporter;
- a production contract;
- consumer-side behavior on other DCC tools or engines;
- that W110's CASE D is reversed. It is not; W110's frozen gate stands.

## 11. Negative findings (preserved)

- The stock Blender exporter's bytes are not reproducible across processes: UIDs and order come from Python `hash()`
  (CODE INSPECTION: `fbx_utils._key_to_uuid`), and the timestamp and path are embedded. Same-seed processes reproduce
  everything but the timestamp.
- Blender ignores `PYTHONHASHSEED` unless launched with `--python-use-system-env` (plumbing probe). So W110's
  producer and consumer had uncontrolled seeds.
- NC4 could only be checked as a rejection. In both control files every take animates the same item/property set, so
  re-pointing one curve node into another take always collides. The detected-and-localized form of take-ownership
  change is covered by NC10.
- The stock round trip still loses holds (U: 0.33–1.22 m). Both producer edit and adapter remain scratch-only.
- The mixed / smooth 0.51 mm residual persists.
- Blanket CONSTANT cannot be distinguished from metadata-driven correction on the real Actions.

## 12. Unresolved uncertainties

- **Scope of the canonicalizer.** It is restricted to the observed armature-only class set. Files with meshes, skins,
  bind poses, materials, constraints or blend shapes are rejected, not canonicalized. Name uniqueness is a hard
  requirement, so rigs with duplicate bone or object names would be rejected.
- **Consumer seed.** Consumers ran with uncontrolled seeds: two consumer processes, each with identical results. A
  consumer-seed dependence below that sample size is not excluded.
- **Untested axes.** Only two seeds, two paths and one machine were tested. Not tested:
  - other Blender versions or add-on versions;
  - other exporters or consumers;
  - location / scale animation, CUBIC or CONSTANT-next sources, per-key mixed flags, coupled items.

  All real animated channels were held and rotation-only (as in W110).
- **No independent replay.** No reviewer independently replayed these Main-local artifacts.
  - §3–§9 are AGENT-RUN.
  - The exporter mechanism is CODE INSPECTION.
  - §7 post-hoc and §8 are diagnostics.

## 13. Tests

| Run | Result | Output |
|---|---|---|
| Focused, after declaration, before cells | 38 passed, 1 skipped (stored-output test) | `results/w111_tests_predeclare_run.xml` |
| Focused, final | **39 passed** | `results/w111_tests_final.xml` |
| Regression: `test_w110_fbx_metadata_recovery.py`, `test_w109_fbx_constant_import.py`, `test_w108_fbx_subframe.py`, `test_w107_fbx_ownership.py`, `test_w106_action_editability.py`, `test_blender_executor.py` | **83 passed** | `results/regression_w106_w110_blender_executor.xml` |

`tests/test_w111_fbx_semantic_repeatability.py` covers the following:

| Area | What is tested |
|---|---|
| Canonicalizer | Acceptance and accounting; writer round trip; invariance to UID relabel, order, timestamp and path |
| Discrimination | Localized detection of value / header changes; all 14 controls on a synthetic two-take tree; MASKED / NOT_LOCAL judging |
| Rejections | 10 rejection classes; `KeyTime` exemption from the UID-leak guard |
| Normalization and contract | Plan normalization across UID offsets; matrix structure; producer environment sanitization |
| CASE mapping | A / B / C / D precedence |
| Consumer isolation | Consumer-input whitelist; consume stage source-free and using the unchanged W110 adapter; no production imports |
| Data | Historical real file (when present); stored declaration / analysis consistency |

Not run:
- the broad production / training suite (no tracked or shared file changed; `shared_tracked_changes = []`);
- any LabServer test;
- any GPU job.

## 14. Execution record, deviations

- **Verify output.** The verify stage's stdout was not saved because of a `tee` path typo. `verify_report.json` is
  the record (ok, no stop).
- **No reruns.** No stage failed, and none was rerun. No declared script changed after the declaration (all bound
  SHAs equal at analysis).
- **Script SHAs.** They are of the bytes as executed. All W111 files were written with LF line endings, and the
  committed blobs are byte-identical to them (e.g. `git show HEAD:scripts/w111_fbx_semantic_identity.py` hashes to
  `7ba33abe…`).
- **Producer launch deviation.** The producer launch adds `--python-use-system-env` relative to W110. This is
  required to control the seed, and is declared. The environment was sanitized: no `PYTHON*` variables except the
  seed and `PYTHONNOUSERSITE`.
- **After the analysis.**
  - The post-hoc diagnostic script and the presentation figures script were written and run after the frozen
    analysis.
  - Neither is bound by the declaration, and neither changes the CASE.
- **Not executed:**
  - any exporter / importer / add-on patch;
  - a fixed-seed consumer;
  - other Blender versions;
  - non-Blender consumers;
  - V0 FBX;
  - CONSTANT-next encoding;
  - renders;
  - LabServer;
  - private real-clip frames (not read).

## 15. Cost, storage, resources

- **FBX file size and keys.** The producer edit adds no bytes or keys (2 bytes changed per flagged curve). File size
  differs only by path length (+16 bytes for pB).

  | Case | Bytes (pA / pB) |
  |---|---|
  | complete_control | 9,509,612 / 9,509,628 |
  | arm_natural_most | 2,487,276 / 2,487,292 |
  | real_clip | 1,809,564 / 1,809,580 |
  | synthetic | ≈ 102 KB |

- **Runtime per variant** (seconds):

  | Stage | complete_control | Synthetic |
  |---|---|---|
  | Export | 1.34–1.36 | ≈ 0.03 |
  | Encode | 1.37–1.50 | — |
  | Canonicalize | 0.42–0.55 | ≈ 0.006 |
  | Import | 0.74–0.94 | — |
  | Adapter | 0.50–0.67 | ≈ 0.008 |

  The full analysis of 120 FBX files with comparisons took 29.5 s.
- **Disk.** 1.1 GB in `C:\Users\dna10\animcv-w111\`, not committed:
  - each run 512 MB, of which controls 195 MB, encode 136 MB, consume 106 MB and cells 77 MB;
  - analysis 9.4 MB.
- **Compute.** 13 Blender launches: options + 2 × 6. Main only, CPU only, no GPU and no network. Within the declared
  cap.

## 16. Reproduction (Git Bash on Main, repo root)

```
PY=/c/Projects/AnimCV/.venv_build_full/Scripts/python.exe; BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
AD="C:/Program Files/Blender Foundation/Blender 5.1/5.1/scripts/addons_core/io_scene_fbx"
O=C:/Users/dna10/animcv-w111; R=$O/results; H=5f9ef0b87c678b710a5c5d06aa2d69aa019af5a2
W="--w106 C:/Users/dna10/animcv-w106 --w107 C:/Users/dna10/animcv-w107 --w108 C:/Users/dna10/animcv-w108 --w109 C:/Users/dna10/animcv-w109 --w110 C:/Users/dna10/animcv-w110"
M=scripts/w111_fbx_semantic_repeatability.py
$PY -X utf8 $M verify --base $O $W --lock C:/Users/dna10/animcv-automation/state/active_batch.json --blender "$BL" --addon-dir "$AD" --starting-head $H --branch arch/single_frame_first
"$BL" -b --factory-startup -t 2 --python-exit-code 1 -P scripts/w108_fbx_subframe_blender.py -- options --out $R/options.json
$PY -X utf8 $M declare --base $O --w110 C:/Users/dna10/animcv-w110 --verify $R/verify/verify_report.json --options $R/options.json --blender "$BL" --starting-head $H
S=$(cat $R/declare/declaration.sha256); D=$R/declare/declaration.json
for X in run1 run2; do
  $PY -X utf8 $M cells    --declaration $D --declaration-sha $S --run-dir $R/$X --blender "$BL"
  $PY -X utf8 $M encode   --declaration $D --declaration-sha $S --run-dir $R/$X
  $PY -X utf8 $M consume  --declaration $D --declaration-sha $S --run-dir $R/$X --blender "$BL"
  $PY -X utf8 $M controls --declaration $D --declaration-sha $S --run-dir $R/$X
done
$PY -X utf8 $M analyze --declaration $D --declaration-sha $S --base $O $W --verify $R/verify/verify_report.json --out-dir $R/analysis
$PY -X utf8 scripts/w111_fbx_semantic_repeatability_posthoc.py --results $R --out-dir $R/posthoc
$PY -X utf8 scripts/w111_fbx_semantic_repeatability_figures.py --results $R --out-dir $R/figures
$PY -X utf8 -m pytest tests/test_w111_fbx_semantic_repeatability.py -q
$PY -X utf8 $M manifest --base $O $W --starting-head $H --declaration-sha $S
```

How the stages behave on a re-run:
- Every output stage refuses an existing directory.
- verify requires a Batch-111 RUNNING lock and only new W111 files in the tree.
- On a fresh base, the declaration hash changes through recorded paths and SHAs.
- Fresh exports carry new timestamps (and, per seed, the same UIDs).

**Evidence manifest.** `C:\Users\dna10\animcv-w111\evidence_manifest.json`, SHA
**`11ea4de969429bb4dcf2ab3fb1a013942d7b345123a6be9f66363fa5b5927a8f`**:
- 1,160 files;
- `all_originals_unchanged: true` for W106–W110;
- it excludes itself and `manifest_stdout.txt`.

Other hashes:

| File | SHA |
|---|---|
| `verify_report` | `eaed125f…` |
| encode reports | `10c0bbc6…` / `8ade0165…` |
| consumer inputs | `9dbbceaf…` / `90efe5d2…` |
| consume reports | `e49ef753…` / `a70a1bff…` |
| `analysis_report` | `99b30807…` |
| `posthoc_report` | `dc912039…` |
| `figures.json` | `95a6ebda…` |

## 17. Changed files, final source revision, environment

New files only. SHAs are of the LF bytes as executed:

| File | Role | SHA |
|---|---|---|
| `scripts/w111_fbx_semantic_identity.py` | Canonicalizer, comparison, raw variation, plan normalization, scratch FBX writer, controls | `7ba33abe…` |
| `scripts/w111_fbx_semantic_repeatability_contract.py` | Cases, matrix, launches / environment, tolerances, CASE rule | `28aef3d3…` |
| `scripts/w111_fbx_semantic_repeatability_blender.py` | Blender `export` (producer, per cell) and `consume` (W110 adapter, U/A/A2/K, scene digests) | `8e3d7c71…` |
| `scripts/w111_fbx_semantic_repeatability.py` | Host verify / declare / cells / encode / consume / controls / analyze / manifest | `8266f8ef…` |
| `scripts/w111_fbx_semantic_repeatability_posthoc.py` | Post-hoc diagnostics (not a gate) | `89780c4f…` |
| `scripts/w111_fbx_semantic_repeatability_figures.py` | Presentation-only figures | `16557e8b…` |
| `tests/test_w111_fbx_semantic_repeatability.py` | Focused tests | `cc2ce4fb…` |
| this Worklog | | |

Nothing else changed:
- no `src/` file, exporter, importer, add-on, rig, source Action or scene setting;
- no W106–W110 artifact or historical Worklog;
- not `docs/next_batch.md`;
- no automation file.

**Environment.**
- Main, Windows 11, CPU only.
- Python 3.11 (`C:\Projects\AnimCV\.venv_build_full`).
- Blender 5.1.0, exe SHA equal to W110.
- io_scene_fbx 5.15.0, sources equal to W110.

**Final source revision.** The commit adding this Worklog on `arch/single_frame_first`, directly after `5f9ef0b`. The
COMPLETE marker is added in a separate final commit.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 111
