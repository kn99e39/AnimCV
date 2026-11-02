# BATCH 107 — FBX Action Round-Trip Failure Ownership (Evidence Only)

## CURRENT ASSESSMENT — SUCCESS / FAILURE / UNCERTAINTY

**Human Assessment**
- **SUCCESS:** W106 is COMPLETE (CASE A, Blender-only). Across three existing M_Main BaseRig Actions, scratch-only local upper-arm rotation and timing edits preserved non-target channels, produced measurable visible changes, survived .blend save/reopen, and were reversible within the declared numerical tolerances. A fresh synthetic armature also passed the FBX round-trip gate.
- **FAILURE:** Every real BaseRig Action failed the independent, predeclared FBX baseline parity gate: max integer-frame world-coordinate drift 1.29 / 6.10 / 7.51 mm (>1 mm). CONSTANT becomes LINEAR after import, half-frame differences reach 0.96 m, scene frame_end becomes 250, and all stored Actions may be exported as takes (16 in one file). W106 found the native 50 fps real clip stored in a 30 fps scene (0.6× playback), which is a separate, pre-existing timebase defect. No production correction was authorized.
- **UNCERTAIN:** Which exporter/importer stage or rig property causes the real-Action integer-frame drift; whether default key simplification, animated armature-object transforms, 0.01 object scale, rotation conversion, or an interaction owns the error. W106 post-hoc attribution was descriptive only. W106's 263 MB Main-local binary evidence and Blender runs were not independently replayed by the ChatGPT reviewer.
- **ALTERNATIVE EXPLANATIONS:** Sampling-time mismatch, Action/slot selection, importer representation, FBX coordinate conversion, Euler/quaternion bake or take contamination may account for some apparent drift. Synthetic success alone does not establish scale as the cause. Neither improved pose metrics nor an FBX edit-delta pass proves baseline fidelity.

**Why another bounded Batch matters:** W106 established a usable *Blender-local* Action boundary but identified a concrete, reproducible handoff failure. Before any exporter patch, retarget redesign, timing fix or authoring architecture decision, isolate whether the existing FBX loss is caused by one identifiable serialization contract and distinguish independent failure modes. This is a falsifiable diagnostic, not permission to ship a fix.

## HUMAN-READABLE SUMMARY

**DIRECTION:** Perform controlled, one-factor-at-a-time causal attribution of W106's FBX baseline drift and export semantics on immutable copies of its existing Actions.

**PURPOSE:** Identify the first loss-producing boundary in the unchanged Blender → FBX → Blender path, quantify whether a single factor explains the real-rig failure, and document any residual interactions without changing production.

**CENTRAL INTENT:** Is the W106 real-Action FBX round-trip error primarily caused by a specific exporter/importer transformation or animated-rig property, rather than an inherent inability to edit the Blender Action? Can a controlled scratch-only ablation reproduce and remove that error while preserving the original motion, timing and non-target bones?

**DO:** Audit original and re-imported channels, object transforms, key times and frame samples; predeclare a minimal controlled ablation matrix; run synthetic and existing real-Action comparisons; quantify causal effect, collateral regressions, unresolved alternatives and repeatability.

**DO NOT:** Modify production exporter/importer, retarget, S0, observation policy, checkpoint, rig assets, Action originals, default playback, scene FPS, or any shared pipeline. No architecture implementation, tuning search, new editor/style model, inference, training, new media, branch operation, or downstream production adoption.

**DECISION ENABLED:** Whether there is sufficient causal evidence to justify a later **human-owned** decision about an FBX export contract. A CASE A is not approval to change the exporter.

## DIRECTION / PURPOSE / CENTRAL INTENT — AGENT CONTRACT

Repository: `kn99e39/AnimCV`. Use only existing branch `arch/single_frame_first`. Read `AGENTS.md`, `PRODUCT_VISION.md`, `Architecture_v3_FramePose.md`, `Architecture_v2.md` (Blender/export and Motion Graph contracts), `COORDINATE_CONVENTIONS.md`, the W98 cumulative audit, `docs/AUDIT_W101_W105_2026_11_02.md`, W99A/W100, W102, W105 and **W106**, `src/blender/export.py`, and the current automation worker rules. W106 CASE A is Blender-only; its FBX sub-gate FAILED. Never relabel it as full end-to-end success.

**Architecture hypothesis H107:** A concrete and independently manipulable export/import property accounts for a material share of the W106 real-rig baseline drift. If that property is controlled without altering source Action values or world motion, the paired integer-frame drift decreases without non-target regression. If no single factor is sufficient, preserve that falsification and characterize interactions; do not invent a fix.

The central hypothesis concerns **FBX transport fidelity**, not pose accuracy or an animation editor. Treat default-key simplification, animated armature-object transform/scale, rotation representation, importer behavior, Action/take enumeration and timebase as *separate candidate mechanisms*, not synonyms.

## HISTORICAL BASELINE — PRESERVE

1. Remote branch and all historical Worklogs, W106 `docs/next_batch.md` input identity, W105/W106 manifests and original .blend/.fbx/NPZ/video artifacts remain immutable. No edits to `src/`, historical scripts, tests, rig assets, datasets, checkpoints or old reports.
2. Freeze the W106 baseline exporter `src/blender/export.py` exactly, including its default Blender operator options. Reproduce its V0 export/import from the W106 scratch copies (or verified W105 originals) before any variant; verify hashes and Action/slot identity. No GT or privileged reference is used.
3. Preserve the three W106 M_Main Action cases: `complete_control`, `arm_natural_most`, `real_clip`; keep the W106 synthetic minimal armature separately labelled. Reused 3DPW and historical real clip are not unseen data. Do not substitute R11 or silently choose the saved active RUNTIME Action.
4. Retain W106 negative findings: 1.29/6.10/7.51 mm default baseline drift; CONSTANT→LINEAR, frame_end=250, multi-take leakage, and real-clip 50→30 fps mismatch. Never repair or suppress them for the baseline.

## CHANGE ONLY / DO NOT TUNE

**CHANGE ONLY:** New isolated W107 diagnostic scripts, focused tests, a W107 Worklog, and scratch-only copies/FBX exports/imports/JSON/NPZ/rendered evidence under a fresh output directory. All ablations must be declared before reading their outcome metrics. Changes to exporter operator *arguments* are allowed only in an isolated diagnostic call on scratch files; never modify or monkey-patch `src/blender/export.py`. Preserve the V0 call through the real production function.

**DO NOT TUNE:** No parameter sweep or optimization against a success metric. One predeclared control value per causal factor, with rationale and Blender operator availability recorded before running. Do not change interpolation, scene FPS, key times, object scale or animation values in original Actions. A scratch synthetic scale/transform intervention is allowed only with mathematically verified equivalence of world-space reference motion; stop that arm if equivalence fails.

## BOUNDED EXPERIMENT AND CONTROLS

**Preflight/identity gate:** Verify W106 COMPLETE marker and W106 manifest, input SHA, original Action/slot/key digests, Blender version, exporter source SHA, sample times, rest/rig identity, Main working tree and active-job/worker lock. Stop on missing originals, dirty/conflicting state, running prior Batch, or mismatch. Do not download new datasets or modify the Main environment.

**Declaration gate (hash before outcome inspection):** Freeze cases, variant order, exact Blender operator properties and their runtime-supported names, metrics, sample set, tolerances, negative controls, hypothesis-to-test mapping, CASE decision rule, stop conditions and resource cap. Keep a copy of the frozen declaration hash in every output.

**Paired variants, minimal and independent:**
- **V0 unchanged baseline:** invoke the actual existing `export_fbx` and default importer; reproduce W106 results within declared deterministic tolerance. If V0 cannot reproduce, stop and report CASE D, not a new success.
- **V1 simplification isolation:** identical source Action, importer and all other exporter settings; only set the documented Blender FBX bake key-simplification factor to its zero-simplification control value. Record key counts, integer/half-frame world parity, object and local-bone errors. Do not modify the production function.
- **V2 take-ownership isolation:** identical to V0 except a single diagnostic export-all-Actions toggle (if supported). Quantify number/identity of takes, active-Action selection, and whether the selected Action's motion changes. Treat take contamination separately from geometric parity.
- **V3 transform/scale mechanism probe:** synthetic, controlled world-equivalent armature conditions (static vs animated armature object, scale 1 vs 0.01), crossed only if the equivalence invariant and a small fixed 2×2 design are feasible. Match evaluated world-space motion and Action interpolation before export. Use the same V0 settings. If no world-equivalent construction can be verified, mark NOT TESTABLE; do not normalize the real BaseRig or alter its rest pose to force a result.
- **Optional single importer-side probe:** only if a specific importer property is discoverable and can be predeclared independently of all exporter variants; otherwise record importer ownership as unresolved. No opportunistic combinations or adaptive search.

All variants are run on scratch copies, with separate original, exported and imported SHA/digests. Execute the fixed real cases plus the synthetic control; preserve case-level and variant-level failures. A new synthetic variant must not be pooled with the three real Actions.

**Measurement:** At the exact original integer and half-frame sample times, report max, p95 and median world-space head/tail position error (m), armature-object rotation/translation error, armature-local bone errors, per-bone/per-frame outliers, baseline vs edit-delta where applicable, Action/slot identity, imported interpolation, key counts, take count, FPS/frame range and file size. Separate source .blend fidelity, FBX export semantics, importer reconstruction and timebase. Compare untouched bones and Action data; do not classify descendants of a deliberately edited bone as harm. Re-render only a few predetermined outlier/negative-control frames if numerical evidence alone is ambiguous; do not expand to subjective animation quality.

**Causal decision:** A factor is *supported* only if the predeclared one-factor intervention reduces the paired real-rig baseline error in the predicted direction, with negative controls and original world-motion equivalence intact, and its effect repeats. Report per-case effect sizes and residual error. A factor that fixes only take count, key count, interpolation metadata, or a synthetic condition is not proven to fix integer-frame geometric parity. A partial reduction is not a full parity pass.

**Regression and reproducibility:** Focused W107 synthetic/identity/CASE tests; W106 Action contract tests and only relevant Blender exporter/retarget tests. No blanket production regression unless a shared file unexpectedly changes (then STOP). Run a deterministic repeat of V0 and any causal candidate on fresh scratch outputs; retain both positive and negative runs, version/configuration, logs, hashes, and reproducible commands. No training, inference, LabServer work or network acquisition.

## PREDECLARED CASE / STOP CONDITIONS

- **CASE A — CAUSAL FACTOR SUPPORTED:** V0 reproduces, all identity/synthetic/negative-control gates pass, and at least one independent predeclared factor explains a substantial real-Action geometric drift reduction on all three cases, with at least one restored to ≤1 mm; repeated results agree. State explicitly whether **all three** pass the original 1 mm gate. Other failures (CONSTANT→LINEAR, timebase, multi-take) remain separate.
- **CASE B — HYPOTHESIS FALSIFIED:** V0 and controls reproduce, but none of the tested independent factors yields the predicted material real-Action improvement; preserve the failures without tuning or proposing a patch.
- **CASE C — PARTIAL / CONFOUNDED:** Some variants are untestable, results are mixed, or an apparent benefit depends on unisolated interactions. Record the uncertainty and no production conclusion.
- **CASE D — INVALID:** Input/manifest/Action mismatch, V0 non-reproduction, declaration breach, synthetic world-equivalence failure used as evidence, untracked original modification, or irreproducible comparison. Stop; preserve partial logs.

**STOP immediately** on any unapproved source/rig/Action mutation, missing or mismatched baseline, nondeterministic V0, unexpected version/option incompatibility, excessive resource/environment cost, active concurrent Agent or Git conflict. Never bypass a STOP by changing thresholds, sampling, rig, FPS, exporter defaults or Action selection. Do not ask the coding Agent to recommend W108.

## FINAL EVIDENCE-ONLY REPORT

Write `docs/107_WORKLOG_FBX_ROUNDTRIP_FAILURE_OWNERSHIP.md` with: Human Assessment (success/failure/uncertainty/alternative explanations), predeclared hypothesis and immutable identities, all tested and untested variants, measured V0 reproduction, per-case quantitative tables and outlier attribution, causal vs correlational interpretation, negative controls, timing/take/interpolation status, test/command/log/artifact SHA provenance, all execution failures and omissions, limits on real-world generalization, resource cost, explicit CASE A/B/C/D and what human decision the evidence would inform. Do not include a next-Batch recommendation. Commit only new W107 scripts/tests/Worklog, using the existing worker protocol; never edit `docs/next_batch.md` or other historical files.

## COMPLETION CONDITION

The Batch is COMPLETE only after: (1) remote/head and original identities remain valid; (2) the hashed predeclared V0 + bounded variants are executed or explicitly marked untestable; (3) real and synthetic measurements, negative cases, reproducibility, focused regressions and evidence manifest are retained; (4) the Worklog states an honest CASE with scope/limits and **AUTOMATION_BATCH_STATUS: COMPLETE** plus **BATCH_NUMBER: 107**; and (5) the final Worklog and its completion marker are committed on the same existing branch. A failure or CASE D can be a completed diagnostic, never silently upgraded to success. Do not start W108.
