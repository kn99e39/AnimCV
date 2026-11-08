# Worklog — Opt-in FBX Continuous-Time Interchange for the Verified Blender Action (2026-11-09)

Batch 127, run unattended on `arch/single_frame_first` on Main:

- local RTX 5080 (no GPU work was needed);
- Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9);
- Blender 5.1.0 with `io_scene_fbx` 5.15.0;
- Windows 11 Pro 10.0.26200, NTFS.

`docs/next_batch.md` was read only.

Every number, test result and hash below is **AGENT-RUN** on Main. No Reviewer has re-executed any of it. Where something
was established by reading code rather than running it, it is marked CODE-INSPECTED.

## 1. Official verdict and scope

**Official verdict: CASE B under the frozen CASE rule.** Source: `results/assess/assessment.json` (SHA-256 `5e2df325…`).

| Gate | Result | Why |
|---|---|---|
| G1 user path + default isolation | **FAIL** | test C7 failed on a **test defect** (§8) before its stock run executed |
| G2 mechanism controls (C1, C2, C3, C5a) | PASS | |
| G3 negative controls, safety | PASS | no package and no recovered `.blend` in any refused control |
| G3b negative controls, classification | PASS | every declared refusal code observed |
| G4 repeatability | **FAIL** | test C8 failed on a **test defect** (§8); the real S1 repeat passed |
| G5 isolation / concurrency | PASS | |
| G6 real S1/S3 | **FAIL** | the declared raw-path mismatch list was too narrow (below) |
| G7 regression | PASS | W124 35/35, W125 58/58, W126 40/41 (only the historical N12), regression 343/343 |
| G8 visual | PASS | as declared; but the stock panel is blank (§7) |

The frozen rule gives B: not D, and the opt-in path passes on all 6 real states, with G2, G3 (safety) and G5 true.

**What failed in G6.** The opt-in interchange passed every real check. What failed is my declared expectation for the
*raw* A/B (the package FBX through the stock importer alone). I declared its mismatch list as exactly
`[SUBFRAME_WORLD_DEVIATION_OVER_1MM, INTERPOLATION_NOT_RECOVERED, SCENE_FRAME_RANGE_NOT_CARRIED]`. The real runs also show:

- `OBJECT_TRANSFORM_DIFFERS` on all 6 states. The armature object rotation varies in the real Actions; it was static in the
  synthetic fixture I developed on.
- `PIN_RESIDUAL_OVER_1MM` on S1 W126V: 3.87 mm at subframes inside the pin interval.

The raw path did fail, and failed more than predicted. The prediction was wrong, not the instrument. Nothing was relaxed.

**What this evidence supports at most.** A bounded, opt-in Blender-5.1-to-Blender-5.1 FBX interchange for the declared
known-BaseRig Action regime:

- every property's keys are all CONSTANT or all LINEAR;
- integer key times;
- one assigned Action;
- 30 fps scene frames 1..N.

Recovery depends on a slice-private package manifest and an opt-in consumer adapter.

**What it does not support:**

- raw-FBX losslessness;
- Unity / Unreal / third-party importer interoperability (not tested);
- other rigs, mixed interpolation inside a property, BEZIER;
- other Blender versions;
- animation quality;
- Production readiness or a default-path change.

**Historical verdicts:** W126 C, W125 A (transactions only), W124 B, and W107 A / W108 A / W109 A / W110 D / W111 A /
W112 D are untouched. No historical Worklog, run file or W107–W112 script was edited.

## 2. Starting state, revisions, environment

| Item | Value |
|---|---|
| Instruction blob | `bd553b9f0f81bfb13f2bc59ab2fa0e6b16c93206`, verified first |
| Starting commit | `4229fe7eca6ce79bbb39a34678815d04cb45e177`. The remote had not advanced at start, at declaration or before docs. |
| W126 completion head (provenance) | `d92ddfdc4338c2ebd61ab7e32309b6d516fc10f7`, an ancestor of the starting commit |
| Declaration | `C:/Users/dna10/animcv-w127/results/declare/declaration.json`, SHA-256 `2186401d6b33e72d703a5807fefeba751ef5d6be08528dda2a74e65e4912863b`, written 2026-11-08T20:30:56Z, `outcomes_inspected: false` |
| Pre-outcome instrument commit | `cefc8c1`: implementation, focused tests, control helper, harness. The declaration SHA is in its message. |
| Post-outcome commit | `cc5fcc4`: the C7/C8 test fixes and the supplementary visual script (§8) |
| Final source revision | `cc5fcc478879854af8add83a50c1acb37bb145f8` for code, tests and scripts. The commit adding this Worklog changes documentation only; the COMPLETE marker is a separate final commit. |
| Machine | Main only. No LabServer, no training, no network beyond `git fetch/push`. |

**Instrument hashes at declaration (LF SHA-256).** `assess` verified them unchanged.

| File | SHA-256 |
|---|---|
| `src/app/experimental_video_action.py` | `45f1b965…` |
| `src/blender/experimental_action_stage.py` | `c977da7d…` |
| `src/blender/experimental_fbx_interchange.py` | `23587d88…` |
| `tests/test_experimental_fbx_interchange.py` (frozen version) | `344e6b5a…` |
| `tests/w127_fbx_controls_blender.py` | `e3b7cd9a…` |
| `scripts/w127_fbx_interchange_harness.py` | `285a82d6…` |

**Files byte-identical to the starting commit** (checked by `assess` with `git diff`):

- `src/blender/export.py`, `src/blender/executor.py`;
- `src/app/experimental_action_txn.py`, `src/app/cli.py`;
- `scripts/w110_fbx_consumer_adapter.py`;
- the W124 / W125 / W126 test files;
- `docs/next_batch.md`.

**What I saw before the declaration.** Development used only the W125/W126 8-frame **synthetic-value** BaseRig fixture.

- **C1 user path:** I saw its result before declaring: INTERCHANGE_PASS at about 1e-6 m, and its raw A/B FAIL (subframe
  0.34 m).
- **Two development findings that shaped the exporter arguments:**
  1. An armature-only FBX carries no BindPose, so the stock importer rebuilds every bone rest from the posed frame. On the
     fixture the world rest moved up to 0.92 m; exporting the bound mesh too brought it to ≤ 1 µm.
  2. A GLOBAL scene take writes absolute time (frame/fps). The importer's default `anim_offset` of 1 then re-imports every
     key one frame late (2..N+1). A per-Action take starts at time 0 and maps back to frame 1.
- **Known from W126:** the stock S1/S3 figures.
- **Not seen before the declaration:** any W127 S1/S3 interchange outcome, and every other control outcome.

**W126 / W125 limits carried forward.** W126 CASE C (the N12 classification prediction) and the W125 transaction scope are
unchanged. Their seals are unkeyed SHA consistency checks, not authentication. W127 adds no semantic edit, never accepts,
reverts or reclassifies anything on S1/S3, and never weakens the W126 commit-time gate.

## 3. Implementation and trust boundary

**New file `src/blender/experimental_fbx_interchange.py`.** A maintained adaptation of W107–W111. The parity of its curve
map with the W110 scratch adapter is a focused test. It contains:

- a binary FBX reader that fails closed (`FBX_PARSE_ERROR` on truncation);
- the W109 byte-exact `KeyAttrFlags` patcher;
- the W110 connection-graph curve map and plan/apply;
- a UID-free group table (the W111 lesson);
- the producer, the consumer and the manifest seal.

**Additions to `src/app/experimental_video_action.py`.** Three opt-in subcommands; `fbx` and every earlier command are
unchanged:

| Command | Role |
|---|---|
| `fbx-interchange` | producer |
| `fbx-consume` | consumer |
| `fbx-verify` | evaluator |

**Additions to `experimental_action_stage.py`.** Backward-compatible changes only:

- modes `interchange_export`, `interchange_consume`, `fbx_import_save`;
- an optional `--frozen-audit 0` for readback;
- an optional per-job `subframe` for render.

The frozen `export_fbx` / `fbx_readback` functions are byte-identical.

**Producer** (reads the run; writes only a NEW `--out` directory).

1. It resolves `head`, `baseline` or an explicit id. The state must be in the published lineage of the current head and
   verify: the content-addressed state, the registry `.blend` SHA, a readback bound to that `.blend`, plus `verify_head`
   for the head.
2. It refuses while any transaction is pending (`RECOVERY_REQUIRED`).
3. It hashes every lineage/baseline/accepted/reverted/edits/manifest file before and after the export. Any change, or a
   head move during a `head` export, is a STOP.
4. A Blender process opens the `.blend` and **never saves it** (the SHA is checked before and after). It checks:
   - one armature with the known BaseRig topology;
   - meshes bound to the armature only;
   - no NLA, drivers or constraints;
   - every F-curve maps to (object | bone, location | rotation | scale);
   - no modifiers;
   - integer key times;
   - each property's keys all CONSTANT or all LINEAR;
   - fps_base 1 and keys spanning scene frames 1..N.
5. In memory only, it removes the other Actions and names the take. It exports with:
   - `bake_anim_simplify_factor=0`;
   - `bake_anim_use_all_actions=True`, `bake_anim_use_nla_strips=False`;
   - `object_types={ARMATURE, MESH}`;
   - `add_leaf_bones=False`.
6. It rewrites `0x6104 → 0x6002` only on FBX curves of CONSTANT-driven groups, and verifies the result:
   - the byte diff lies only inside those flag values;
   - the map is unchanged apart from the flags;
   - the decoded classes are as directed;
   - the take is unchanged.

   An UNDRIVEN group must be constant-valued.
7. It writes the sealed `package_manifest.json` **last** (atomic). A directory without it is never a package.
   `STATUS.json` records IN_PROGRESS / FAILED / COMPLETE.

**Consumer** (`fbx-consume --package DIR`). The run is not an argument.

- A fresh factory-empty Blender reads `package_manifest.json` and `animation.fbx`, with a Python audit hook (`open` events)
  installed before anything else.
- It checks, in declared precedence (`declaration.json` `consumer_refusal_precedence`):
  - seal, schema, Blender/importer version (5.1 / 5.15.0);
  - FBX SHA;
  - a single take with the declared name;
  - take range and FBX frame rate against the manifest;
  - known-BaseRig declaration;
  - FBX group table against the manifest.
- Then it runs the stock importer with default flags, unchanged and unpatched, and checks: rig and objects equal the
  manifest, exactly one Action `<arm>|<take>`, scene fps.
- It sets only keyframe `interpolation` on uniquely mapped groups from the FBX flags; any unresolved group is a STOP.
- It requires the result to equal the manifest's directives, sets the scene range from the manifest, verifies the key
  range and the post-apply census, and saves a NEW `.blend`.

`--mode raw` saves the stock import only, labelled RAW_FBX_STOCK_IMPORT.

**Evaluator** (`fbx-verify` / `evaluate_pair`). Separate code from producer and consumer.

- It re-verifies the run state and binds the package to it (`PACKAGE_NOT_OF_THIS_STATE` otherwise).
- It opens the original `.blend` and the consumed `.blend` in **two fresh Blender processes**.
- It compares, on one frozen grid:
  - every non-finger bone's scene-world head and tail at every integer frame and at +0.25/+0.5/+0.75/+0.99;
  - object matrices;
  - rest (armature space × scale) and armature scale;
  - per-group interpolation;
  - Action inventory, key range, scene fps and range;
  - the pin residual at the declared pin domain.

**Trust boundary.**

| Role | May read | Must not read |
|---|---|---|
| Consumer | the package manifest + FBX | anything else |
| Original `.blend` | only the evaluator, after import | — |

The audit shows that **every** consumer run on S1/S3 opened exactly the 2 declared inputs, plus:

- Blender/Python installation code;
- its own output;
- `…/Blender/5.1/extensions/.cache/compat.dat`, Blender's per-user extension cache. It is disclosed in its own category,
  not an animation input.

UNDECLARED reads: none.

Limit: the audit sees Python-level opens only. Blender C-level reads are not visible. The consumer calls no
`open_mainfile` / append, and records `libraries_linked = 0` and an empty `bpy.data.filepath` before save.

**Distinctions kept separate in every result:**

1. raw FBX alone (stock importer; `--mode raw`);
2. FBX plus the package manifest;
3. FBX + manifest + the opt-in Blender adapter (the only path that recovers);
4. the unchanged stock `fbx` command.

## 4. Real S1/S3 results (SAVED-ARTIFACT REAL-VIDEO REPLAY)

**Inputs.** Copies of the W126 final `S1_replay` / `S3_replay`, verified file by file against the W126 evidence manifest
`f9139cd6…`: 3212 and 2089 files, 0 differences. Both clips match their declared SHA-256.

This is not fresh perception. S1/S3 are the correlated W121–W126 engineering sources.

**States, per source:**

| State | S1 | S3 | Label |
|---|---|---|---|
| baseline | `1284d6ce…` | `07667f65…` | NOT_APPLICABLE (unedited) |
| W126V, the W126 accepted hold edit | `674ba278…` | `fd249fbb…` | **AUTHORIZED_AT_COMMIT** |
| current head (exact revert) | `73dcd3dd…` | `be9a1f09…` | NOT_APPLICABLE (revert) |

The head is value-equal to the baseline, and its numbers are identical to the baseline's.

| S1 (240 frames) | Stock `fbx` (unchanged) | Stock FBX on the W127 grid | Raw: package FBX + stock importer | **Opt-in package + adapter** |
|---|---|---|---|---|
| Integer world max (all 3 states) | 3.90 mm | 3.90 mm | 6.3 µm | **6.3 µm** (median 0.84 µm) |
| Subframe world max | 0.69 m (W124 grid) | 0.90 m | 0.90 m | **6.5 µm** (+0.25/.5/.75/.99 each ≤ 6.5 µm) |
| Object matrix max abs diff | — | 1.2e-2 | 1.2e-2 | **3.4e-8** |
| Rest max | — | 0.80 m | 4.6e-7 m | **4.6e-7 m** |
| Interpolation | 519 LINEAR | not recovered | not recovered | **source groups recovered** (31 CONSTANT + 488 undriven LINEAR, constant-valued) |
| Bones / takes / key range / scene range | — | +5 leaf bones; range not carried | 1 Action; range not carried | **equal; 1 Action; 1..240; 30 fps; range 1..240** |
| W126V pin residual (integer / inside-pin subframes) | — | 3.5 / 3.9 mm | 1.2 µm / 3.9 mm | **1.2 / 1.2 µm** (source 0.64 µm) |
| Verdict | FAIL | FAIL | FAIL | **INTERCHANGE_PASS ×3** |

| S3 (154 frames) | Stock `fbx` | Stock on the W127 grid | Raw | **Opt-in** |
|---|---|---|---|---|
| Integer world max | 6.77 mm | 6.77 mm | 1.3 µm | **1.3 µm** |
| Subframe world max | 0.21 m | 0.28 m | 0.28 m | **1.3 µm** |
| Object matrix | — | 1.4e-3 | 1.4e-3 | **6.5e-9** |
| Rest max | — | 0.62 m | 4.6e-7 m | **4.6e-7 m** |
| W126V pin residual | — | 4.4 / 4.4 mm | 1.1 µm / 0.19 mm | **1.1 / 1.3 µm** (source 0.56 µm) |
| Verdict | FAIL | FAIL | FAIL | **INTERCHANGE_PASS ×3** |

The stock `fbx` command reproduces the W126 figures exactly on the W126V states: S1 3.90 mm / 0.69 m and S3 6.77 mm /
0.21 m, with the mismatches INTEGER, SUBFRAME, INTERPOLATION_CHANGED and SCENE_FRAME_RANGE_NOT_CARRIED.

**Exact affected frames and bones.**

- Opt-in: no integer frame, subframe sample or bone over 1 mm on any state.
- Stock, worst sample: S1 frame 22 +0.99 (`hand_l`); S3 frame 59 +0.99 (`neck_02`).
- Per-bone and per-frame lists: `real/<src>/<state>/stock_eval/eval_report.json`.

**Authoring safety** (whole-run file snapshot plus `status` stdout, before and after all 3 states):

- S1 and S3: 0 files changed, 0 removed, 0 modified under `fbx/`. The only new files are 18 under `fbx/`, from the
  unchanged stock command.
- `status` output is byte-identical.
- Every source `.blend` SHA is equal before and after.
- Unchanged: head, generation, journals, sealed refs, lineage log, W126 accepted copies, footprint/proposal/commit files,
  and the W126 exact-undo reports.
- No semantic edit was created, accepted or reverted on S1/S3.

**Repeat with exporter UID and source-path variation.** S1 W126V was exported again from a second verified copy at another
path, in another process.

- Exporter bytes differ.
- Take, timebase, rig, interpolation and group table are equal.
- The recovered integer and subframe world arrays are **bit-equal** to the first export, and the census is equal.

## 5. Synthetic / controlled matrix (focused tests: real Blender BaseRig, synthetic values)

All controls run on the 8-frame fixture, after P1 (a nonzero left-wrist pin) was accepted under the W126 gate. Results
are from the frozen test file (`pytest/focused.xml`).

| ID | Control | Expected (frozen) | Result |
|---|---|---|---|
| C1 | All-CONSTANT head with nonzero pin, via the user CLI | INTERCHANGE_PASS; pin ≤ 1 mm; raw = the 3 declared mismatches; baseline package PASS; run files and `status` identical; exact revert after export; package refused for another state | PASS |
| C2 | Smooth LINEAR control | 0 patched curves, PASS; **blanket CONSTANT on the recovered copy → SUBFRAME FAIL** (integer ≤ 1 mm); raw → only SCENE_FRAME_RANGE | PASS |
| C3 | Per-property mixed (upperarm_l LINEAR, the others CONSTANT) | 21 patched curves, PASS; blanket CONSTANT → SUBFRAME FAIL; raw = the 3 declared | PASS |
| C4 | Per-key mixture / axis mixture / BEZIER / wrong rig | `UNSUPPORTED_INTERPOLATION_MIXTURE` ×2, `UNSUPPORTED_INTERPOLATION_BEZIER`, `UNSUPPORTED_RIG`; STATUS FAILED; no manifest; source unchanged | PASS (4/4) |
| C5a | Extra unassigned Action in the source | excluded and listed; one take; PASS | PASS |
| C5/C6 | Package negatives: missing manifest, edit without reseal, FBX byte flip, stale FBX of another package, 2-take FBX resealed, take renamed, fps 25, unknown-rig declaration, wrong-rig FBX consistently resealed, producer version 4.2, directive edit, one flag reverted, truncated FBX, consumer crash after import | PACKAGE_INCOMPLETE, MANIFEST_SEAL_INVALID, FBX_HASH_MISMATCH ×2, TAKE_NOT_UNIQUE, TAKE_MISMATCH, TIMEBASE_MISMATCH, UNSUPPORTED_RIG, RIG_MISMATCH, UNSUPPORTED_PRODUCER_BLENDER_VERSION, METADATA_FBX_DISAGREE ×2, FBX_PARSE_ERROR, CONSUMER_STAGE_FAILED; no `.blend`, no report, package bytes untouched | PASS (14/14) |
| C6 | Producer failures | stage fault → PRODUCER_STAGE_FAILED; killed before manifest → exit 86, FBX present without manifest; both consume → PACKAGE_INCOMPLETE; existing output refused; run unchanged | PASS |
| C7 | Stock `fbx` unchanged and still FAIL | the function source equals the starting commit; FAIL | **FAIL: test defect** (§8) |
| C8 | Fresh repeat with UID + path variation | bytes differ; semantics and recovered readbacks equal | **FAIL: test defect** (§8) |
| C9 | Concurrency | two exports to one `--out` → exactly one COMPLETE, the other REFUSED_EXISTING_OUTPUT; an export during an in-flight commit (W125 hang) → RECOVERY_REQUIRED, no output; the commit then completes | PASS |

The unit tests (seal, path grouping, directive rule, access classification), default CLI isolation and W110 curve-map
parity all PASS.

**Why the controls discriminate.** Recovery is not "blanket CONSTANT":

- Blanket CONSTANT damages the smooth (C2) and mixed (C3) controls at subframes.
- The opt-in path passes both.
- The raw import of the same package FBX fails, so the importer alone does not explain the result.
- The flags-only FBX without the adapter changes nothing. This matches W109: the stock importer ignores `KeyAttrFlags`.

## 6. Unsupported external consumers

**Unity.** Editors 2022.3.8f1 / 6000.x are installed on Main. The declared optional, non-gating Unity probe was **NOT
RUN**, to keep the batch bounded.

**Not available on Main:** Unreal Engine, the Autodesk FBX SDK, other DCCs.

So, **NOT TESTED**:

- whether any non-Blender consumer honours the CONSTANT `KeyAttrFlags`;
- whether it honours the per-Action take time origin;
- how it handles the 3× key density.

The only demonstrated consumer is Blender 5.1 / io_scene_fbx 5.15.0 together with the W127 adapter. The raw FBX alone in
that same Blender loses held motion (0.90 m / 0.28 m) and the scene range.

## 7. Visual review (30 fps)

**4-panel videos** (source | original `.blend` | stock FBX import | opt-in recovered), W126V:

| Source | Video | SHA-256 | Frames written / decoded | Duration |
|---|---|---|---|---|
| S1 | `ab4_S1_w126v_30fps.mp4` | `1c649d73…` | 240 / 240 | 8.000 s |
| S3 | `ab4_S3_w126v_30fps.mp4` | `78771c30…` | 154 / 154 | 5.133 s |

**Quarter-frame review (4× slow):** 28 samples per source.

| Source | Window (centred on the stock worst sample) | SHA-256 |
|---|---|---|
| S1 | frames 19–25 | `c4b97a9d…` |
| S3 | frames 56–62 | `4e7124491…` |

**Instrument limitation found after the run.** The Workbench render draws the rig's skinned mesh, and the stock exporter's
FBX is armature-only. The **stock panel is therefore blank** apart from the pin marker. G8 (frames written and decoded)
passes as declared, but the declared visual gives no stock-versus-opt-in visual comparison.

**Supplementary skeleton visual** (post-declaration; not a gate). `scripts/w127_supplementary_skeleton_visual.py` draws the
bone skeletons from the evaluator's existing readbacks, with no new evaluation:

| Source | Windows | `supplementary_visual.json` |
|---|---|---|
| S1 | worst-stock window `640dd7d4…`, pin window `cfcc509e…` | `67e5018a…` |
| S3 | worst-stock window `fded5a3a…`, pin window `6792d7a7…` | (same file) |

**Agent rating, not a human real-time rating.**

- On the S1 sheet, the stock import coincides with the original at integer frames (0.9 mm). Between frames it swings
  visibly away: 106 / 219 / 333 mm at +0.25 / +0.5 / +0.75, mostly the left arm and legs.
- The opt-in skeleton lies on the original at every sample (0.0 mm at display precision).
- In the 4-panel video, the opt-in render is indistinguishable from the original.

**Unchanged quality negatives.** An FBX fix does not repair any of these:

- partial rig: right arm, upper spine, neck, hands and feet at rest;
- no root translation, no contact;
- left/right assignment unknown;
- the S1 left arm visibly disagrees with the source: the subject holds a ball at the chest while the rig's left arm is
  raised sideways;
- the S3 wrist is occluded;
- 20–22 mm one-frame pin-boundary jumps;
- the S3 elbow is near straight;
- GT and subject identity UNKNOWN.

## 8. Tests (RUN / PASS / FAIL / NOT RUN) and post-outcome changes

| Set | Kind | Result |
|---|---|---|
| `tests/test_experimental_fbx_interchange.py` (31, frozen) | 3 unit, 1 isolation, 27 real-Blender synthetic-value (CLI processes and the packaging core) | **29 PASS, 2 FAIL** (C7, C8: test defects) |
| `tests/test_experimental_video_action.py` (W124, unmodified) | unit, MOCK FK, real Blender | **35/35 PASS** |
| `tests/test_experimental_action_txn.py` (W125, unmodified) | real Blender, process | **58/58 PASS** |
| `tests/test_experimental_gate_integrity.py` (W126, unmodified) | real Blender | **40/41**. The only failure is the historical N12 classification prediction; not modified, not claimed 41/41. |
| Bounded regression (the W126 list of 16 + W107–W112 FBX tests = 22 files) | existing | **343/343 PASS** |
| Real S1/S3 interchange, stock A/B, repeat | real Blender, saved artifacts | **RUN**: opt-in 6/6 INTERCHANGE_PASS; raw and stock FAIL |
| Visual | render + compose | **RUN** (stock panel blank, §7) |

**Official files** (outside the repo):

| File | SHA-256 |
|---|---|
| `results/pytest/focused.xml` | `d1b82065…` |
| `results/pytest/regression.xml` | `2f9b5f78…` |

**Post-outcome changes, all disclosed (commit `cc5fcc4`).** The CASE was computed before them and is unchanged.

1. **C7 (test defect).** The helper extracted `cmd_fbx` as the text up to the next `def`. In the new file that span also
   contains the W127 section comment inserted *after* the function, so the comparison failed. The test stopped there,
   before its stock run. Fix: compare the function's own AST source segment. The function itself is byte-identical.
2. **C8 (test defect).** Test C1 runs `revert` on the session-shared fixture, so C8 copied a run whose head was a
   different (reverted) state. That is not a repeat. Fix: a pristine snapshot taken at fixture creation.

   The real-data repeat in the harness passed independently (§4).

**Supplementary run** of the corrected file: **31/31 PASS** (`results/pytest_supplementary_postfix/w127_postfix.xml`,
`0d808d31…`). It includes the stock `fbx` FAIL on the fixture and a bit-equal recovered repeat. It is evidence about the
test defects only; it is not the official result.

**Not run:**

- the full repository suite;
- the Unity probe and any other external consumer;
- other Blender versions;
- fresh live perception;
- power loss / OS crash;
- independent Reviewer execution;
- a human visual rating.

## 9. Negative findings

- **Raw FBX is still lossy.** Even the opt-in export, imported without the adapter, loses held motion (S1 0.90 m, S3
  0.28 m), object-rotation holds and the scene range. It also breaks the hard pin between frames (S1 3.9 mm).
- **The prediction miss** (§1) came from developing only on a fixture with a static object rotation. The fixture did not
  exercise object-level held motion; the real data did.
- **The declared visual was weak for stock.** It renders meshes, and the stock FBX has none. Discovered only after
  rendering.
- **Size and key cost.** The package FBX is 3.1× (S1: 1.15 MB vs 0.37 MB) and 2.6× (S3: 0.90 vs 0.35 MB) the stock FBX,
  with 112,320 / 72,072 keys. The FBX adds 468 baked curves where the source has 32 F-curves; the recovered Action has
  519 F-curves (488 constant-valued undriven LINEAR).
- **Export changes the package's rig content.** The opt-in export includes the skinned mesh (needed for the BindPose) and
  omits leaf bones. A package is therefore not a drop-in replacement for the stock armature-only file.
- **Take name and time origin.** The take is named `root|AnimCV_EXP_<id>`; the recovered Action is `root|root|…`. The
  per-Action take convention (FBX time 0 = Action frame 1) requires scene frames 1..N; other origins are refused.
- **Supported regime only.** Any per-key or per-axis interpolation mixture, BEZIER, NLA, constraints, drivers or another
  rig STOPs; nothing is approximated.
- **The seal is an unkeyed SHA.** A consistent forger who rewrites the FBX, group table, directives and seal is caught
  only by the rig / take / timebase / plan checks, not by authentication. The C6 controls cover this.
- **Unity is installed but untested.**

## 10. Unresolved uncertainties

- Whether SDK-based engines (Unity, Unreal) read `0x6002` as held keys, respect the per-Action time origin, and keep the
  BindPose rest. All NOT TESTED.
- Behaviour on Blender versions other than 5.1.0, or `io_scene_fbx` other than 5.15.0. The consumer refuses them; nothing
  was tested.
- Whether bit-equal recovered readbacks across processes hold on another machine. Shown on Main only, 2 processes.
- Python-level audit only; C-level file access by Blender is not observable.
- C-level importer internals (CODE-INSPECTED through W109, not re-instrumented here).

## 11. Cost, maintenance, rollback

**Per state** (end-to-end CLI wall time, Main):

| | Export | Consume | Verify |
|---|---|---|---|
| S1 | 1.0 s | 0.9–1.0 s | 2.5–2.6 s |
| S3 | 0.9 s | 0.9 s | 2.2 s |

The stock `fbx` command takes 2.3 s. Visual rendering: 15 s (S1) and 10 s (S3) for 3 `.blend` files × 2 views.

**Storage** (outside the repository, `C:/Users/dna10/animcv-w127`):

| Directory | Size |
|---|---|
| `results` | 367 MB |
| `runs` | 704 MB |
| `runs_pathvar` | 423 MB |
| `dev` | 9 MB (development fixture only) |

**Maintenance.** One new opt-in helper module (~1,100 lines), three subcommands and three stage modes. The manifest schema
`animcv_exp_fbx_interchange_package_v1` is slice-private, not a Production schema. The adapter is tied to the stock
importer's naming (Blender 5.1 / io_scene_fbx 5.15.0) and refuses other versions.

**Rollback.** Revert commit `cefc8c1` (plus `cc5fcc4` for the tests). Nothing else depends on the new commands, and run
files are never written by them.

## 12. Evidence, reproduction, changed files

**Evidence.** `C:/Users/dna10/animcv-w127/results/evidence_manifest.json` lists 11,518 files and was written before this
Worklog. SHA-256 `8d70fa406d395a111aaf9e035405ad38f7c1c4d09515cd84724eacbf0493a4c2`.

| File | SHA-256 |
|---|---|
| `preflight/preflight.json` | `e7c45e39…` |
| `declare/declaration.json` | `2186401d…` |
| `real/run_index.json` | `8af92991…` |
| `real/S1/w126v/verify/fbx_verify_report.json` | `6155d222…` |
| `real/S3/w126v/verify/fbx_verify_report.json` | `30e8689f…` |
| `visual/visual.json` | `8326cef0…` |
| `assess/assessment.json` | `5e2df325…` |

S1 W126V package:

- FBX `f8b842e7…`; raw exporter output `6de9c7b5…`;
- manifest digest `075136224d02…`;
- recovered `.blend` `9d0c17fe…`.

**Harness reproduction** (Git Bash, repository root at `cefc8c1`; every stage refuses an existing output and, after
`declare`, a changed instrument):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; R=C:/Users/dna10/animcv-w127/results; export PYTHONPATH=src
D="--declaration $R/declare/declaration.json --declaration-sha 2186401d6b33e72d703a5807fefeba751ef5d6be08528dda2a74e65e4912863b"
$PY -X utf8 scripts/w127_fbx_interchange_harness.py preflight --out-dir $R/preflight
$PY -X utf8 scripts/w127_fbx_interchange_harness.py declare   --out-dir $R/declare --preflight $R/preflight/preflight.json
$PY -X utf8 scripts/w127_fbx_interchange_harness.py real      --out-dir $R/real $D
$PY -X utf8 scripts/w127_fbx_interchange_harness.py visual    --out-dir $R/visual --run-dir $R/real $D
$PY -X utf8 scripts/w127_fbx_interchange_harness.py pytest    --out-dir $R/pytest $D
$PY -X utf8 scripts/w127_fbx_interchange_harness.py assess    --out-dir $R/assess --run-dir $R/real --visual-dir $R/visual --pytest-dir $R/pytest $D
$PY scripts/w127_supplementary_skeleton_visual.py --real-dir $R/real --visual-dir $R/visual --out-dir $R/visual_supplementary
```

**One user path** (S1 copy; inspect PASS/FAIL; revert stays available):

```
export ANIMCV_BLENDER="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
RUN=C:/Users/dna10/animcv-w127/runs/S1_replay
$PY -m app.cli experimental-video-action fbx-interchange --run $RUN --state head --out out/pkg
$PY -m app.cli experimental-video-action fbx-consume     --package out/pkg/package --out out/consumed
$PY -m app.cli experimental-video-action fbx-verify      --run $RUN --consumed out/consumed --out out/verify
#   prints "interchange_verdict": "INTERCHANGE_PASS" | "FAIL" + mismatches; details: out/verify/fbx_verify_report.json
$PY -m app.cli experimental-video-action fbx             --run $RUN --state head   # stock path: still FAIL (unchanged)
$PY -m app.cli experimental-video-action revert          --run $RUN                # revert the head's edit, as before
```

**Changed files (repository):**

- New:
  - `src/blender/experimental_fbx_interchange.py`;
  - `tests/test_experimental_fbx_interchange.py`, `tests/w127_fbx_controls_blender.py`;
  - `scripts/w127_fbx_interchange_harness.py`, `scripts/w127_supplementary_skeleton_visual.py`;
  - this Worklog.
- Modified:
  - `src/app/experimental_video_action.py`: the opt-in `fbx-interchange` / `fbx-consume` / `fbx-verify`, the packaging
    core, state verification, the evaluator; `cmd_fbx` unchanged;
  - `src/blender/experimental_action_stage.py`: three modes and two backward-compatible options;
  - `docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md`: the W127 section and the FBX limitation note;
  - `docs/README.md`, `docs/00_마스터_플랜.md`, `docs/01_프레임워크_현황.md`: dated 2026-11-09 W127 notes (experimental,
    Agent-run, CASE B).
- Unchanged:
  - `docs/next_batch.md` and every historical Worklog;
  - `src/blender/export.py`, `src/blender/executor.py`, `src/app/cli.py`, `src/app/experimental_action_txn.py`;
  - the W107–W112 scripts and tests, and the W124–W126 tests;
  - the GUI, models and the rig assets;
  - the normative architecture documents and the automation scripts.
