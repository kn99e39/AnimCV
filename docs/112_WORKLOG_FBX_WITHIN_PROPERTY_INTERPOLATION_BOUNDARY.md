# Worklog — FBX Within-Property Interpolation Recoverability Boundary (2026-11-04)

## Human Assessment

**Verdict: CASE D — INVALID under the frozen rule. No viability conclusion is drawn.** W110 stays CASE D and W111
stays CASE A (bounded); this Batch reinterprets neither.

The frozen CASE rule checks validity inputs before any outcome input. Two of them were false:

1. **`instrument_valid` — false.** The declared instrument check required every uniform-held family member to pass the
   FBX-level bound within 1 mm. `syn_L_uniC` (location family, uniform held) misses by **16.8 mm**. The miss is
   already present **at integer key frames, in every L-family case and condition**, including the untouched import
   (`control_U`) and blanket CONSTANT (`control_K`).
   - The whole 16.8 mm sits on **one point, the `base` bone's tail**. Every other head and tail matches within 1 µm.
   - Likely mechanism (post-hoc, §9): the armature-only V1 export writes no bind pose, so the importer rebuilds rest
     geometry from the exported Lcl values. Once the child bone `upperarm_l` carries location animation, this moves
     its parent's (`base`) rest tail. The declared point set (all heads and tails) therefore contains a
     round-trip rest-geometry artifact unrelated to interpolation.
2. **`repeat_ok` — false.** The two independent runs disagree on 15 report fields. All 15 are list *positions*:
   - `obstructed_curves[*].channel` order in five cases;
   - the candidate's `perkey[0].group_index` in `rr_int`.

   My report code ordered both lists by exporter UID, and UIDs change in every process (W110/W111 mechanism). After
   sorting those lists post-hoc, the two runs have **0 differences**. Every numeric value is equal, and both FBX
   kinds are W111-canonically equal across runs for all 15 cases. This is the same failure class as W110: a frozen
   repeat criterion that did not anticipate UID-dependent ordering. **It was not relaxed.**

All other D-inputs evaluated true:
- preflight, declaration binding, sources, exports, encoding integrity;
- isolation, integrity and originals;
- controls: all 12 positive / negative control checks.

Everything below is evidence only. It is not support or refutation of H112.

- **OBSERVED (AGENT-RUN; frozen point set; repeated identically in two runs).**
  - **Interval-mixed, axis-uniform rotation is expressible and was recovered source-free.** This covers `syn_R_int`
    (synthetic) and `rr_int` (synthetic intervention on the real complete_control rig).
    - The producer wrote per-key KeyAttrFlags runs. Keys, times and values were untouched; the run-length encoding
      split the three target curves into 9 runs (syn) or 24 runs (rr).
    - The unchanged W110 adapter leaves that group UNRESOLVED (per-key mixture), giving 79.7 mm / 296 mm.
    - Both privileged bounds are feasible:

      | Case | Import-level bound | FBX-level bound |
      |---|---:|---:|
      | syn_R_int | 16.9 µm | 0.18 mm |
      | rr_int | 70.9 µm | 0.56 mm |

    - The separate source-free per-key candidate recovered both cases at the import-level bound:

      | Case | Candidate, all probes | Held intervals |
      |---|---:|---:|
      | syn_R_int | 16.9 µm | 0.78 µm |
      | rr_int | 70.9 µm | 1.0 µm |

      It had 0 unresolved groups, 0 unmapped F-curves, three FBX opens only, and clean integrity.
  - **Axis-mixed rotation is not reached by any declared interpolation-only assignment, at either layer.** On
    `syn_R_axis` / `syn_R_int1`:

    | Bound | Max error | Intervals > 1 mm |
    |---|---:|---:|
    | Import-level | 23.8 mm | 8 of 24 |
    | FBX-level | 16.0 mm | 8 of 24 |

    The same-family uniform-LINEAR control passes both layers (17.0 µm import-level; 0.18 mm FBX-level). On `rr_axis`:

    | Bound | Max error | Intervals > 1 mm | Same-family uniform-LINEAR control |
    |---|---:|---:|---|
    | Import-level | 3.14 mm | 8 of 69 | 0.21 mm (passes) |
    | FBX-level | 35.4 mm | 19 of 69 | 1.04 mm (fails: no attribution) |

  - **The frozen producer rule obstructs every axis-mixed case** at the encoding layer (the source axes disagree).
    No metadata FBX was written for those cases, and they stopped at P2.
- **OBSERVED, POST-HOC (L family with only the `base` tail removed; §9).**
  - The FBX-level bound for axis-mixed location (`syn_L_axis`) is **0.60 µm**, via per-FBX-axis flags `LLC`
    (synthetic rest frame axis-aligned with the parent). So the FBX representation can express that case, and the
    obstruction there is the producer **rule**, not the format.
  - The import-level bound on the same case is **9.84 mm**: the imported location F-curves cannot be given matching
    CONSTANT/LINEAR choices.
  - `syn_L_int` passes both layers (0.61 µm). Uniform L controls pass (≤ 0.63 µm).
- **NOT ESTABLISHED.** Any CASE A/B/C statement. That covers:
  - whether the interpolation-only abstraction is viable for within-property mixtures;
  - which layer limits it.

  The observations above suggest a split by mixture type (interval-mixed recoverable; axis-mixed limited at the
  import layer, and for rotation also at the FBX layer). But that split is not a verdict of this Batch.
- **For transparency, the outcome inputs recorded by the frozen analysis were:**
  - `imp_ok` false on syn_R_axis, syn_R_int1 and rr_axis, and on all five L cases (the base-tail artifact);
  - `fbx_ok` false on syn_R_axis, syn_R_int1, rr_axis and all L cases;
  - `encoded` true only on syn_R_int, syn_L_int and rr_int;
  - `sf_ok` true only on syn_R_int and rr_int.

  These inputs were never evaluated past the D gate, and no B or C is claimed.
- **ALTERNATIVE EXPLANATIONS (status).**

  | Explanation | Status |
  |---|---|
  | Axis-mixed failure is just "missing FBX flags" | Not supported for synthetic rotation. The FBX-level bound fails even with an ideal flag-honouring evaluator over all 8 per-axis assignments. For location the FBX layer *can* express it (post-hoc), so there the gap is rule- and import-side. |
  | Failure is the uniform smooth residual, not the mixture | Excluded for syn_R and for rr at the import layer (the uniform-LINEAR control passes). Not excludable for rr at the FBX layer (rr_uniL is 1.04 mm). |
  | Import-level failure is quaternion normalisation only | Not separable. The imported bone frames differ from the source frames (sign-aligned local quaternion difference up to 0.84 at keys even on syn_step, where world error is 0.77 µm). Location, which has no normalisation, also fails at the import layer post-hoc. Normalisation coupling and frame mixing both remain possible. |
  | Candidate used source truth | Excluded as far as tested. It received a structural-whitelist input, ran under an audit hook (3 `open` events, all of the FBX), and passed a static import / token guard. It imports only the unchanged W110 adapter. |
  | Per-key recovery is a blanket or uniform effect | Excluded. Blanket CONSTANT harms the smooth intervals (27 mm syn, 75 mm rr). W110 uniform leaves the group unresolved. NC1 shows that one flipped key changes only that interval. |

## Original task scope (summary)

Batch 112 asked whether interpolation-only post-import recovery remains possible when FBX AnimationCurves of the SAME
transform property carry different interpolation semantics:
- across axes (X/Y/Z), or
- across successive key intervals.

It also asked which layer owns any limitation: FBX metadata encoding, Blender import decomposition, imported-Action
expressiveness, or source-free correspondence.

Required elements:
- frozen synthetic contracts and an unmodified real Action replay;
- separately labelled real-rig-derived mixed cases;
- source → FBX → import → Action → motion accounting;
- a privileged source-known feasibility bound distinct from a source-free consumer;
- controls;
- an independent repeat.

Only isolated `scripts/w112_*`, `tests/test_w112_*`, this Worklog and scratch outputs were allowed. No production,
importer, add-on, key, timebase or tolerance change was allowed.

## 1. Starting state and provenance — AGENT-RUN (`results/verify/verify_report.json`, `754c64ed…`)

- **Repository.**
  - Branch `arch/single_frame_first`, HEAD `cf93a7425722435efb8905ffec673d4c396235a7`.
  - `docs/next_batch.md` blob `e3de6bfcba3f8fe986976633978f629cb935eefa`. Both were verified before any work.
  - After `git fetch`, `origin` equalled HEAD and the tree was clean.
  - Worker lock: batch 112, RUNNING, `start_commit` = HEAD. No other Blender process was running.
- **History.**
  - The W111 Worklog ends with its COMPLETE marker.
  - W111 declaration `0ffa9d3c…` and manifest `11ea4de9…` match the immutable references, and all 1,160 W111 files
    match.
  - W110 declaration `223b8c81…` and manifest `e46329f5…` match, and all W110 files match.
  - Every W106–W109 manifest file matches.
- **Code and environment.** All 20 scripts bound by the W111 declaration are unchanged, including the exporter
  `src/blender/export.py`. FBX add-on sources and the Blender exe SHA equal W111 (Blender 5.1.0 `adfe2921d5f3`,
  io_scene_fbx 5.15.0). The exporter options file is byte-identical to W110's.
- **Inputs.** Seven byte copies of W110 inputs were made into `results/inputs/`, each equal to its W110 manifest
  SHA: syn_step / syn_smooth / complete_control source `.blend` and V1 FBX, plus the W106 synthetic N0 file. All were
  unchanged at the end.
- **Regimes (never pooled).**
  - complete_control: `benchmark_detector_observation` (3DPW), unmodified real Action (replay).
  - rr_*: synthetic interventions on a copied real Action (complete_control rig), **not** real-world mixed evidence.
  - syn_*: synthetic.

## Pre-declaration work (outside the evidence base)

All of this ran in `C:\Users\dna10\animcv-w112-explore` (throwaway, not committed):
- **Structural inspection.** The complete_control source was inspected for channels, rotation modes and rest. The
  object and 7 bones carry quaternion rotation keys every 3 frames (1–70), all CONSTANT; the object's scale is 0.01.
  The V1 KeyAttr layout was also inspected (one run `0x6104`).
- **Two full plumbing dry runs on a NON-declared target bone (`lowerarm_l`)** through a hand-edited development
  declaration. Only exit codes and plumbing fields were inspected (statuses, key counts, isolation, integrity,
  harness, instrument key coverage, NC behaviour); no bound or motion value of the dry runs was read.
- **Changes made because of the dry runs:**
  - **Key merge.** Blender merges imported keys closer than 0.01 frame. The `f+0.99` evaluation keys collided with
    the `f+1` keys (118 of 121 survived), so the instrument is split into two files per assignment (part a: phases
    0/.25/.5/.75 + end; part b: .99 only).
  - **Build target.** The Blender build reads the target from the declaration.
  - **NC4 judgement.** Tightened so it cannot pass vacuously (the twin `sibling` groups must be unresolved).
  - **Diagnostic.** A target-local quaternion / location diagnostic was added.
- **Declarations.** Only one declaration was ever made. The bound scripts did not change after it.

## 2. Frozen declaration — `results/declare/declaration.json`, SHA **`b403d53edf5fe65df0c08eeebd85f128bfcf0d490bae81dadbcbcfd1a231054d`**

`outcomes_read: false`. It was written before any W112 build, export, encode, import, adapter, oracle or candidate
run. The focused tests ran after it and before run 1: 19 passed, 1 skipped (the stored-output test). The declaration
binds the 4 W112 scripts plus the 20 W111-bound scripts; all were unchanged at the end (`scripts_bound: true`).

- **Hypothesis H112.** For within-property axis- or interval-mixed interpolation on one item / Lcl property, the
  unchanged Blender 5.1 import followed by CONSTANT/LINEAR interpolation-only edits informed by FBX metadata recovers
  the evaluated source motion within 1 mm.
- **Case matrix** (15 cases). The target is `upperarm_l` everywhere. Every other channel stays as in its source
  (all CONSTANT).
  - **Historical controls, reused byte-for-byte:**
    - syn_step: uniform held (R family uniform-held member);
    - syn_smooth: uniform LINEAR, all bones;
    - complete_control: real replay (RR uniform-held member).
  - **R family** (synthetic, source quaternion W/X/Y/Z): `syn_R_uniL`, `syn_R_axis`, `syn_R_int`, `syn_R_int1`.
  - **L family** (synthetic, location). Its 27 location keys are a **separately declared key construction**: keys at
    1+3k with declared values, identical key-set SHA `06d5e103…` in every L case. Cases: `syn_L_uniC`, `syn_L_uniL`,
    `syn_L_axis`, `syn_L_int`, `syn_L_int1`.
  - **RR family** (copied real Action, quaternion): `rr_uniL`, `rr_axis`, `rr_int`.
  - **Specs** (key index k; held axis = quaternion X, or location X):

    | Spec | Rule |
    |---|---|
    | UNI_C | all CONSTANT |
    | UNI_L | all LINEAR |
    | AXIS | held axis CONSTANT, others LINEAR |
    | INT | all axes CONSTANT on even k, LINEAR on odd k |
    | INT1 | held axis CONSTANT on even k / LINEAR on odd k; others LINEAR |

  - **Within-property cases:** the 8 AXIS / INT / INT1 cases.
- **Producer rule (source-informed, producer only).**
  - **Per-key class.** Each selected-take FBX curve maps to the source (item, transform group). FBX key j receives
    the interpolation in force on (f_j, f_j+dt) in **every** source curve of that group.
  - **Outcomes.** If all agree, the key gets that class (CONSTANT `0x6002` / LINEAR `0x6104`). If they disagree, the
    key is OBSTRUCTED and the case STOPS. With no source curve the curve is UNDRIVEN and stays untouched.
  - **Encoding.** Run-length KeyAttrFlags / DataFloat / RefCount, re-serialised with the W111 writer.
  - **Validity.** The masked tree equals the control, decoded per-key flags equal the intended flags, RefCount sums
    match, and the W111 canonical difference equals exactly the re-encoded curves.
- **Privileged bounds (SOURCE-KNOWN, never a consumer).**
  - **Import-level.** Untouched control import. Non-target groups are set to their source interpolation. Every
    combination in {C, L}^n of the target's imported F-curves is sampled (n = 4 quaternion, 3 location).
  - **FBX-level instrument.** Per assignment in {C, L}^3 of the target's FBX curves, a scratch evaluation FBX is
    written: every selected-take curve is re-keyed at probe phases with the FBX-level hold / lerp value of the
    *existing* keys, imported unchanged, and sampled at its key times. Keys are added only in these evaluation files.
  - **Exact minimum.** A probe inside (k_j, k_j+1) depends only on key j's choice, so the product-space minimum
    equals the per-interval minimum. This is tested against brute force.
  - **Not in the declared spaces:** CUBIC / tangents, CONSTANT-next, BEZIER handles, key or value changes.
- **Consumers.**
  - `control_U` / `control_A` / `control_K` and `meta_U` / `meta_A`, using the unchanged W110 adapter and harness.
  - **Conditional source-free candidate** `w112_fbx_perkey_candidate.py`. It reuses the W110 correspondence. A group
    left unresolved *only* by per-key flag variation is resolved per key when all of these hold; otherwise nothing
    changes:
    - identical KeyTime on all axes;
    - the same class on every axis at every key;
    - imported key count and spacing matching.
  - **Eligibility.** The candidate runs only if the encoding is VALID and the import-level bound is ≤ 1 mm.
- **Grid.**
  - Every interval gets `f + {0, .25, .5, .75, .99}`, plus `frame_end`: 121 synthetic and 346 real samples.
  - Metric: max |coordinate| world head/tail error over common bones (W108 form).
- **Tolerances** (historical, unchanged): 1 mm recovered / preserved, 1e-9 m identity, 1 cm harm.
- **Negative controls** on the `syn_R_int` metadata FBX:
  - **NC1:** one held-transition key flipped on all target curves (expected: localized);
  - **NC1b:** the same key flipped on one curve only (expected: unresolved);
  - **NC2:** a key time + 1 tick (expected: unresolved);
  - **NC3:** a key value + 1.0 (expected: plan unchanged, motion error > 1 mm);
  - **NC4:** target Model renamed to `sibling` (expected: unresolved).
- **CASE rule.**
  - **D-checks first** (11): preflight, declaration, sources, export, encoding integrity, instrument, isolation,
    integrity, controls, repeat, originals.
  - **B** if some within case fails a bound layer that its family's uniform-LINEAR control passes.
  - **A** if every within case passes both bounds, is encoded and is recovered source-free.
  - **C** otherwise.
- **Repeat.** Two full independent runs. Every decision input must be equal and every leaf within 1e-9. Built FBX are
  compared with the W111 canonicaliser.
- **Resource cap.** 1,800 s per run, 6 GB disk; no GPU, no network.

## 3. Execution — AGENT-RUN

| Run | produce | encode | consume (65 imports) | oracle | gate | candidate (7 cases + 6 NC × 2) | Wall time |
|---|---|---|---|---|---|---|---|
| run1 | exit 0 | exit 0 | exit 0 | exit 0 | exit 0 | exit 0 | 3.6 min |
| run2 | exit 0 | exit 0 | exit 0 | exit 0 | exit 0 | exit 0 | 3.5 min |

- **Builds.** All 12 built cases had a valid construction: key data excluding interpolation equalled the base,
  non-target interpolation was unchanged, and the base input was unchanged. Every V1 export returned `FINISHED`
  with the Action and work file unchanged.
- **Controls.** Historical sources equal their W110 digests.
- **Analysis.** It ran once over both runs: `analysis_report.json` `18dfd735…`, CASE D.

## 4. Source → FBX → import accounting — AGENT-RUN (run1; run2 identical)

| Case | Encoding | Re-encoded curves | W110 adapter on meta (or control): FBX curves mapped / imported F-curves mapped | Directives |
|---|---|---:|---|---|
| syn_step | VALID | 15 | 54/54 · 59/59 | 5 C / 13 L |
| syn_smooth | VALID | 0 | 54/54 · 59/59 | 18 L |
| complete_control | VALID | 24 | 468/468 · 519/519 | 8 C / 148 L |
| syn_R_uniL | VALID | 12 | 54/54 · 59/59 | 4 C / 14 L |
| syn_R_axis | **OBSTRUCTED** (3 curves × 25 keys) | — | 54/54 · 59/59 (control) | 18 L |
| syn_R_int | VALID | 15 (3 per-key) | **51/54** · 59/59 | 4 C / 13 L / **1 UNRESOLVED** |
| syn_R_int1 | **OBSTRUCTED** (13 keys) | — | control | — |
| syn_L_uniC / uniL | VALID | 18 / 15 | 54/54 · 59/59 | — |
| syn_L_axis / int1 | **OBSTRUCTED** | — | control | — |
| syn_L_int | VALID | 18 (3 per-key) | 51/54 · 59/59 | 1 UNRESOLVED |
| rr_uniL | VALID | 21 | 468/468 · 519/519 | 7 C / 149 L |
| rr_axis | **OBSTRUCTED** (3 × 70) | — | control | 156 L |
| rr_int | VALID | 24 (3 per-key) | 465/468 · 519/519 | 1 UNRESOLVED |

- **Validity of every VALID encoding.**
  - Masked trees are equal, and per-key flags decode as intended.
  - The W111 canonical difference is exactly the re-encoded curves, with 0 header or connection differences.
  - Only the KeyAttr arrays change semantically. The W111 writer stores arrays uncompressed, so every meta file is
    larger than its control (106.5 kB vs 102.4 kB synthetic; 14.4 MB vs 9.5 MB complete_control). The per-key cases
    add 571 bytes (synthetic) or 1,651 bytes (rr) over their uniform counterparts.
- **Per-key runs.** In `syn_R_int` the three target curves carry `C3 L3 C3 L3 C3 L3 C3 L3 C1`.
- **Importer behaviour.**
  - `meta_U` equals `control_U` (≤ 1e-9 m) on every encoded case: the importer ignores flags, as in W109.
  - `control_A` equals `control_U` and changes 0 F-curves on every case.
- **Correspondence granularity (unchanged from W110).** It is (item, Lcl property) → (pose bone, property). Imported
  bone frames are not the source frames: the sign-aligned target local quaternion differs by up to 0.84 at keys
  (syn_step), while world error stays at 0.77 µm.

## 5. Bounds, conditions and candidate — AGENT-RUN (frozen point set; maxima over all probes; both runs equal)

**R family (synthetic, separately labelled):**

| Case | control_U | meta_A (W110) | control_K | Import-level bound | (group-uniform) | FBX-level bound | (group-uniform) | Candidate S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| syn_step | 327 mm | 0.77 µm | 0.77 µm | 0.77 µm | 0.77 µm | 0.77 µm | 0.77 µm | 0.77 µm |
| syn_R_uniL | 291 mm | 17.0 µm | 27.3 mm | 17.0 µm | 17.0 µm | 0.18 mm | 0.18 mm | 17.0 µm |
| **syn_R_axis** | 291 mm | — | 32.4 mm | **23.8 mm** | 32.4 mm | **16.0 mm** | 32.4 mm | not eligible |
| **syn_R_int** | 327 mm | **79.7 mm** | 27.3 mm | 16.9 µm | 16.9 µm | 0.18 mm | 0.18 mm | **16.9 µm** |
| **syn_R_int1** | 291 mm | — | 32.4 mm | **23.8 mm** | 32.4 mm | **16.0 mm** | 32.4 mm | not eligible |

**RR family (synthetic interventions on the real complete_control rig; complete_control itself is the unmodified
replay, benchmark_detector_observation):**

| Case | control_U | meta_A | control_K | Import-level bound | (group-uniform) | FBX-level bound | (group-uniform) | Candidate S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| complete_control | 764 mm | 1.07 µm | 0.95 µm | 1.07 µm | 1.07 µm | 1.07 µm | 1.07 µm | 1.07 µm |
| rr_uniL | 764 mm | 0.21 mm | 106 mm | 0.21 mm | 0.21 mm | **1.04 mm** | 1.04 mm | 0.21 mm |
| **rr_axis** | 764 mm | — | 111 mm | **3.14 mm** | 90.4 mm | **35.4 mm** | 90.4 mm | not eligible |
| **rr_int** | 764 mm | **296 mm** | 74.6 mm | 70.9 µm | 70.9 µm | 0.56 mm | 0.56 mm | **70.9 µm** |

**syn_smooth** (all bones LINEAR; single source assignment): import-level 0.51 mm (W110's residual reproduced
exactly), FBX-level **1.50 mm** (5 intervals > 1 mm), control_K 113 mm.

**L family (frozen point set):** every case and condition carries the 16.8 mm base-tail error at integer frames:

| Bound / condition | Result |
|---|---|
| Import-level and FBX-level bounds | 16.8 mm on all five cases |
| control_U | 321–365 mm |
| meta_A | 16.8 mm (uniform) / 85.5 mm (syn_L_int) |

For the version with only that point removed, see §9.

- **Per-interval structure (axis-mixed).**
  - **syn_R_axis.** The 8 failing intervals are the last integer interval before each source key (3, 6, 9, …), where
    the held component jumps while the others ramp. The best import-level assignments are mostly `LLLL`; mixed
    per-axis labels (`LCCC`, `CLLL`, `CLLC`) win on 7 intervals. Group-uniform is worse (32.4 mm) than per-axis (23.8 mm).
  - **rr_axis.** Per-axis imported choices reduce the group-uniform 90.4 mm to 3.14 mm, but 8 intervals stay above
    1 mm.
  - **Per-axis versus source-aligned.** Applying the source's per-axis classes index-for-index to the imported
    quaternion gives 92.1 mm (syn) and 26.1 mm (rr), far above the per-axis optimum, because the imported bone frame
    is not the source frame.
- **Candidate (source-free; eligible cases only).**
  - **Inputs.** It received only a whitelisted `candidate_input.json` (`04d8c185…`) listing 7 eligible cases.
  - **Isolation.** Every run recorded 3 `open` events, all of the FBX.
  - **Integrity.** Integrity was clean: only directed interpolation changed; other Actions, rig and timebase were
    unchanged.
  - **On uniform cases** it equals `meta_A` bitwise and changes the same F-curves.
  - **On syn_R_int / rr_int** it resolved the target group per key and changed 20 / 31 F-curves (452 / 2,034 keys).
    The near-boundary error at transitions was 1.2 µm / 2.4 µm.

## 6. Controls and attribution — AGENT-RUN (run1; run2 identical)

| Check | Result |
|---|---|
| syn_step / complete_control: control_U harms, meta_A recovers | 327 mm / 764 mm → 0.77 / 1.07 µm |
| syn_smooth: control_K harms, meta_A preserves | 113 mm → 0.51 mm |
| Importer ignores flags (meta_U ≡ control_U) | all encoded cases ≤ 1e-9 m |
| W110 adapter no-op on LINEAR-only control | 0 F-curves, ≡ control_U |
| Historical replay: control_U vs W110 consume control_U | 0.0 m at 105 / 105 / 300 common times |
| Candidate reduces to W110 on uniform cases | bitwise equal, same F-curves |
| NC0 base (candidate on unmodified meta) | isolated, integrity ok |
| NC1 one held-transition key (frame 3) flipped on all target curves | per-key plan differs at key 2 only; motion identical outside [3, 4) (0.0 m); 79.7 mm inside → **as declared** |
| NC1b same flip on one curve | `UNSUPPORTED_PER_KEY_AXIS_DISAGREEMENT`, 0 target F-curves changed → **as declared** |
| NC2 key time + 1 tick (frame 13) | `KEYTIME_MISMATCH_ACROSS_AXES`, 0 changed → **as declared** |
| NC3 key value + 1.0 (frame 13) | plan identical to base; 5.18 mm error at that key → **as declared** |
| NC4 target Model renamed `sibling` | import succeeds; all 6 `sibling` groups UNRESOLVED; 0 changed (candidate and W110) → **as declared** |
| W110 adapter on every NC file | target group never resolved, 0 target changes |

**Attribution notes:**
- **Blanket versus metadata.** On the interval-mixed cases, blanket CONSTANT harms only the smooth intervals (27.3 mm
  syn, 74.6 mm rr). The candidate keeps them within 71 µm.
- **Diagnostic integrity.** The diagnostic did not pass by dropping curves or by touching unrelated channels:
  coverage is complete on all VALID cases, and unresolved groups are counted.

## 7. Repeat — **FAILED (frozen gate)**, post-hoc characterised

- **Frozen comparison.** It found 15 leaf differences:
  - in `obstructed_curves[*].channel` of syn_R_axis, syn_R_int1, syn_L_axis, syn_L_int1 and rr_axis;
  - in `candidate.perkey[0].group_index` of rr_int.
- **Cause.** Both lists follow exporter-UID order (`fbx_curve_map` iterates curves by UID; plan groups follow model
  UID), and the UIDs change per process.
- **Everything else matched.** Every numeric value and every decision input was equal. All 15 cases' control and meta
  FBX were W111-canonically equal across runs.
- **Post-hoc** (`scripts/w112_fbx_within_property_posthoc.py`, `posthoc_report.json` `63b01a35…`, not a gate).
  Sorting those two lists by (model, property, channel) and dropping the plan index leaves **0** differences. The
  frozen `repeat_ok = false` and CASE D stand.

## 8. Qualitative review — AGENT-RUN (`results/figures/`, presentation only, `figures.json` `74302481…`)

- **`bounds_summary.png`.** A per-case dot plot of control_U, both bounds and S against the 1 mm line. It shows
  three things:
  - interval-mixed cases sit with the uniform controls;
  - axis-mixed cases are 3–36 mm at both layers;
  - the L family sits at 16.8 mm.
- **`held_to_smooth.png`** (syn_R_int, frames 1–13). The source alternates holds and ramps. The candidate overlays it,
  including the jumps at 4, 7, 10 and 13. The untouched import ramps through each hold. The W110 adapter leaves the
  target LINEAR, so it ramps through the last frame of the held intervals [3, 4) and [9, 10).
- **`axis_mixed.png`.** Per-interval best error at both layers:
  - syn_R_axis: periodic spikes at the pre-key intervals;
  - syn_L_axis: flat at the base-tail floor;
  - rr_axis: irregular spikes, the FBX layer worse than the import layer.
- **`real_rig_pose.png`** (rr_axis, t = 33.99, its worst import-level interval). The import-level best overlays the
  source except a slight arm offset (3.1 mm). The untouched import displaces legs, pelvis and the left arm visibly.

Matplotlib views only; no renders. This is not a product-quality claim.

## 9. Post-hoc L family with only the `base` tail removed (NOT a gate; `posthoc_report.json`)

| Case | control_U | control_K | meta_A | Import-level bound (gu / source-aligned) | FBX-level bound (gu) | Best FBX labels |
|---|---:|---:|---:|---|---|---|
| syn_L_uniC | 365 mm | 0.57 µm | 0.63 µm | 0.63 µm | 0.60 µm | CCC ×24 |
| syn_L_uniL | 324 mm | 28.5 mm | 0.61 µm | 0.61 µm | 0.61 µm | LLL ×24 |
| syn_L_axis | 321 mm | 17.8 mm | — | **9.84 mm** (17.8 / 44.3 mm) | **0.60 µm** (17.8 mm) | LLC ×22, LLL ×2 |
| syn_L_int | 365 mm | 21.2 mm | 85.5 mm (W110 unresolved) | 0.61 µm | 0.61 µm | CCC ×12, LLL ×12 |
| syn_L_int1 | 321 mm | 21.2 mm | — | **9.84 mm** | 0.61 µm | LLC ×11, LLL ×13 |

- **The removed error is not constant in time.** It is 12.2–16.8 mm under every motion-recovering condition (K, A,
  oracle) and up to 117 mm under control_U. This fits a relocated rest tail that rotates with `base`'s pose, but the
  mechanism (no bind pose; rest rebuilt from the posed Lcl values) is inferred, not instrumented.
- **What it indicates if the base-tail point is set aside:**
  - For axis-mixed **location**, the FBX format *can* carry the semantics. Source bone-local X maps to the parent
    frame's FBX Z, and `LLC` reproduces it.
  - The frozen producer rule obstructs it, and the unchanged import yields F-curves that no CONSTANT/LINEAR choice
    can match (9.84 mm).
  - This is the opposite layer pattern to rotation.
- **Status.** Post-hoc and diagnostic only. It does not enter the CASE.

## 10. CASE / verdict

**CASE D — INVALID** (`decide_case`, reason "the FBX-level evaluation instrument or the oracle harness is invalid";
`repeat_ok` is also false).

| D-input | Value |
|---|---|
| preflight_ok, declaration_bound, sources_ok, export_ok, encoding_integrity, isolation_ok, integrity_ok, controls_ok, originals_unchanged | true |
| instrument_valid | **false** (syn_L_uniC FBX-level bound 16.8 mm) |
| repeat_ok | **false** (UID-ordered list positions) |

Under the frozen rule this Worklog draws **no viability conclusion**. It does not state that the interpolation-only
abstraction is or is not viable for within-property mixtures. It does not assign the limitation to a layer as a
verdict either.

## 11. Negative findings (preserved)

- **Location channels.**
  - The armature-only V1 round trip relocates a parent bone's rest tail once a child carries location animation
    (16.8 mm here).
  - The declared all-heads-and-tails point set cannot separate that from interpolation. My instrument check therefore
    invalidated the run.
- **Repeat ordering.** UID-ordered report lists broke the frozen repeat. This is the W110 failure mode, reintroduced
  in this Batch's own report code.
- **Per-key mixtures in W110.** The unchanged W110 adapter cannot use per-key flags at all: the group stays
  unresolved (79.7 / 85.5 / 296 mm on the INT cases).
- **Rule obstruction.** The frozen producer rule obstructs every axis-mixed case, including the location case whose
  FBX-level bound is feasible post-hoc. Group consensus is conservative there.
- **FBX-level LINEAR semantics on Euler curves.** They miss the 1 mm gate on the uniform smooth controls: syn_smooth
  1.50 mm and rr_uniL 1.04 mm. The import level stays within it (0.51 / 0.21 mm).
- **Imported frames.** Imported bone frames differ from the source frames. Source-axis-aligned imported assignments
  are therefore meaningless (up to 92 mm).
- **Inherited defects.** The stock round trip still loses holds (0.29–0.76 m), and every W110/W111 defect persists.

## 12. Unresolved uncertainties

- **Validity.** Whether a valid rerun (base-tail point handled by declaration; UID-free list ordering) would select
  B, C or A. Not tested, deliberately, after the outcome.
- **Import-level mechanism for axis-mixed rotation.** Quaternion normalisation coupling versus imported-frame axis
  mixing was not separated.
- **Location import-level failure mechanism.** Inferred only, and post-hoc.
- **Space limits.** CUBIC / tangent encodings, CONSTANT-next, BEZIER handles, Euler-mode or object-level targets,
  scale, coupled items and other bones or rigs are outside the declared spaces.
- **Mixture coverage.** Only one target bone, one held axis and one interval pattern were tested.
- **Real data.** Only one real rig, and its "mixed" cases are synthetic interventions. No real-world within-property
  mixed source was available.
- **Scope.** Blender 5.1.0 / io_scene_fbx 5.15.0 only. Main-local artifacts are AGENT-RUN and were not independently
  replayed.

## 13. Tests

| Run | Result | Output |
|---|---|---|
| Focused, after declaration, before run1 | 19 passed, 1 skipped (stored-output) | `results/w112_tests_predeclare_run.xml` |
| Focused, final | **20 passed** | `results/w112_tests_final.xml` |
| Regression: W111, W110, W109, W108, W107, W106 focused tests + `test_blender_executor.py` | **122 passed** | `results/regression_w106_w111_blender_executor.xml` |

`tests/test_w112_fbx_within_property.py` covers:
- declared specs and the case matrix;
- source in-force semantics;
- the producer rule (consensus / obstruction / undriven / unsupported / bad grid);
- run-length round trip;
- instrument densification and part coverage (no keys within 0.01 frame);
- the exact-bound equals brute force over the product space;
- CASE precedence, including the attribution guard;
- candidate static isolation;
- candidate per-key rules (axis disagreement, KeyTime mismatch, imported count/time mismatch, refcount, coupled /
  other-reason groups ignored);
- run decoding;
- meta encoding and masked equality via the writer / reader;
- evaluation-file construction;
- consumer / candidate whitelists;
- run comparison;
- NC judgement (including non-vacuous NC4);
- consume / candidate stage source-freedom;
- real V1 target mappability;
- stored declaration / analysis consistency.

**Not run:**
- the broad production / training suite (no tracked or shared file changed);
- any LabServer test.

## 14. Reproduction (Git Bash on Main)

The driver `C:\Users\dna10\animcv-w112\run_w112.sh` (scratch, not committed) runs these commands from the repo root:

```
PY=/c/Projects/AnimCV/.venv_build_full/Scripts/python.exe; BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
AD="C:/Program Files/Blender Foundation/Blender 5.1/5.1/scripts/addons_core/io_scene_fbx"
O=C:/Users/dna10/animcv-w112; R=$O/results; H=cf93a7425722435efb8905ffec673d4c396235a7
W="--w106 C:/Users/dna10/animcv-w106 --w107 C:/Users/dna10/animcv-w107 --w108 C:/Users/dna10/animcv-w108 --w109 C:/Users/dna10/animcv-w109 --w110 C:/Users/dna10/animcv-w110 --w111 C:/Users/dna10/animcv-w111"
B=("$BL" -b --factory-startup -t 2 --python-exit-code 1); DD=$R/declare/declaration.json
$PY -X utf8 scripts/w112_fbx_within_property.py verify --base $O $W --lock C:/Users/dna10/animcv-automation/state/active_batch.json --blender "$BL" --addon-dir "$AD" --starting-head $H --branch arch/single_frame_first
"${B[@]}" -P scripts/w108_fbx_subframe_blender.py -- options --out $R/options.json
$PY -X utf8 scripts/w112_fbx_within_property.py declare --base $O --w110 C:/Users/dna10/animcv-w110 --w111 C:/Users/dna10/animcv-w111 --verify $R/verify/verify_report.json --options $R/options.json --blender "$BL" --starting-head $H
DS=$(sha256sum $DD | cut -d' ' -f1)
for X in run1 run2; do RD=$R/$X
  "${B[@]}" -P scripts/w112_fbx_within_property_blender.py -- produce --declaration $DD --declaration-sha $DS --out-dir $RD/produce
  $PY -X utf8 scripts/w112_fbx_within_property.py encode --declaration $DD --declaration-sha $DS --produce-dir $RD/produce --out-dir $RD/encode
  CS=$(sha256sum $RD/encode/consumer_input.json | cut -d' ' -f1); OS=$(sha256sum $RD/encode/oracle_input.json | cut -d' ' -f1)
  "${B[@]}" -P scripts/w112_fbx_within_property_blender.py -- consume --consumer-input $RD/encode/consumer_input.json --consumer-input-sha $CS --out-dir $RD/consume
  "${B[@]}" -P scripts/w112_fbx_within_property_blender.py -- oracle --oracle-input $RD/encode/oracle_input.json --oracle-input-sha $OS --out-dir $RD/oracle
  $PY -X utf8 scripts/w112_fbx_within_property.py gate --declaration $DD --declaration-sha $DS --encode-dir $RD/encode --produce-dir $RD/produce --oracle-dir $RD/oracle --out-dir $RD/gate
  GS=$(sha256sum $RD/gate/candidate_input.json | cut -d' ' -f1)
  "${B[@]}" -P scripts/w112_fbx_within_property_blender.py -- candidate --candidate-input $RD/gate/candidate_input.json --candidate-input-sha $GS --out-dir $RD/candidate
done
$PY -X utf8 scripts/w112_fbx_within_property.py analyze --declaration $DD --declaration-sha $DS --runs $R/run1 $R/run2 --verify $R/verify/verify_report.json $W --out-dir $R/analysis
$PY -X utf8 scripts/w112_fbx_within_property_figures.py --analysis $R/analysis/analysis_report.json --run $R/run1 --out-dir $R/figures
$PY -X utf8 scripts/w112_fbx_within_property_posthoc.py --results $R --out-dir $R/posthoc
$PY -X utf8 -m pytest tests/test_w112_fbx_within_property.py -q
$PY -X utf8 scripts/w112_fbx_within_property.py manifest --base $O $W --starting-head $H --declaration-sha $DS
```

How the stages behave on a re-run:
- Every output stage refuses an existing directory.
- verify requires a Batch-112 RUNNING lock and only new W112 files in the tree.
- Fresh exports carry new UIDs; the frozen repeat fails for the ordering reason in §7.

**Evidence manifest.** `C:\Users\dna10\animcv-w112\evidence_manifest.json`, SHA
**`7ada91836e11775a22d1f1a4d5ae4460f7c6bb6715f0568dd137ea78f1a943f1`**:
- 2,238 files;
- `all_originals_unchanged: true` (W106–W111);
- it excludes itself and `manifest_stdout.txt`.

Other hashes:

| File | SHA |
|---|---|
| verify | `754c64ed…` |
| run1 produce / encode / consume / oracle / gate / candidate | `67f3a298…` / `e8eda5cf…` / `d24f26cb…` / `9ae6f702…` / `b38afc63…` / `686e0386…` |
| consumer input (run1) | `a67702a3…` |
| analysis | `18dfd735…` |
| posthoc | `63b01a35…` |
| figures | `74302481…` |

## 15. Changed files, final source revision, environment, costs

New files only:

| File | Role | SHA (LF bytes, as executed) |
|---|---|---|
| `scripts/w112_fbx_within_property_contract.py` | Cases, specs, producer rule, run-length, instrument, exact bound, metrics, CASE rule (bound) | `304693b8…` |
| `scripts/w112_fbx_perkey_candidate.py` | Conditional source-free per-key candidate (bound) | `2e7999ff…` |
| `scripts/w112_fbx_within_property_blender.py` | Blender produce / consume / oracle / candidate (bound) | `7e0eedaa…` |
| `scripts/w112_fbx_within_property.py` | Host verify / declare / encode / gate / analyze / manifest (bound) | `664e7f89…` |
| `scripts/w112_fbx_within_property_figures.py` | Presentation-only figures (not bound; legend moved after first render) | `974b5141…` |
| `scripts/w112_fbx_within_property_posthoc.py` | Post-hoc characterisation (not a gate; written after the analysis) | `38d8a101…` |
| `tests/test_w112_fbx_within_property.py` | Focused tests | `ab3f4bf1…` |
| this Worklog | | |

Nothing else changed:
- no `src/` file, exporter, importer, add-on, rig, original Action or FBX, scene setting, historical artifact or
  Worklog;
- not `docs/next_batch.md`;
- no automation file.

Final source revision: the commit adding this Worklog on `arch/single_frame_first`, directly after `cf93a74`. The
COMPLETE marker is added in a separate final commit.

**Environment and costs.**
- Main, CPU only, Python 3.11 (`C:\Projects\AnimCV\.venv_build_full`). Blender 5.1.0 was launched
  `-b --factory-startup -t 2`.
- 3.5–3.6 min per run. The oracle took about 27–29 s per real-rig case and about 1 s per synthetic case. The
  candidate took 0.005 s (synthetic) or 0.37 s (real) per call.
- 2.4 GB of scratch disk, mostly the real-rig evaluation FBX: 16 per case at about 15 MB each.
- No GPU, network, LabServer or package change. This was within the declared cap.
