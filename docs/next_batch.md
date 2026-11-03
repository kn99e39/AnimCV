# BATCH 108 — FBX Subframe Motion Semantics and Interpolation Ownership (Evidence Only)

## CURRENT ASSESSMENT — SUCCESS / FAILURE / UNCERTAINTY

**Human Assessment**
- **SUCCESS:** W107 is COMPLETE (CASE A). On three W106 BaseRig Actions, default FBX integer-frame drift 1.29 / 6.10 / 7.51 mm is reproduced and a single isolated `bake_anim_simplify_factor=0.0` intervention reduces it to 0.95 / 1.31 / 1.66 µm. W106 Blender-local Action edits and .blend save/reopen pass their bounded gates.
- **FAILURE:** W107's diagnostic does **not** preserve subframe motion. Source CONSTANT interpolation imports as LINEAR, with half-frame errors 0.417 / 0.960 / 0.354 m. Frame_end=250, take leakage 16/4/2 and the real clip's 50 fps keys in a 30 fps scene are separate unresolved defects. W107's V2 take toggle exported zero Actions.
- **UNCERTAINTY:** Is held-pose interpolation lost in the FBX export/bake, encoded in FBX but discarded by Blender's importer, or confounded by sample timing/representation? Can the existing V1 imported key values express the original continuous-time motion under a controlled scratch-only interpretation change? The W107 binary outputs/tests were reported by the Agent and have not been independently replayed by the reviewer.
- **ALTERNATIVE EXPLANATIONS:** Sampling phase, Action/slot mismatch, object vs pose animation, quaternion/Euler conversion, take selection, key boundary semantics, and differences between synthetic and real rigs. Do not infer unique ownership from a single scratch intervention.

**Why this Batch matters:** Integer-frame pose parity is not sufficient for usable animation: the reported subframe error reaches nearly a metre. W107 removed one failure mechanism without fixing temporal semantics. A bounded, falsifiable interpolation audit can reduce uncertainty before any **human-owned** export/timebase policy change.

## HUMAN-READABLE SUMMARY

**DIRECTION:** Trace continuous-time semantics across original Blender Action → frozen FBX export → FBX curve data → default Blender import; test one predeclared scratch-only interpolation intervention.
**PURPOSE:** Locate the first evidenced loss of held-pose semantics and determine whether existing imported keys suffice to reproduce the source between sampled frames.
**CENTRAL INTENT:** Does the 0.35–0.96 m half-frame FBX error arise from a specific interpolation-semantics boundary rather than missing integer-frame pose data, and can the mechanism be isolated without altering any production exporter/importer, source Action, rig or timebase?
**DO:** Freeze inputs and metrics, inspect FBX curve interpolation attributes where reliable, compare integer/quarter/half/three-quarter frames, run a single scratch-only post-import CONSTANT intervention, include step and smooth synthetic controls, repeat and report.
**DO NOT:** Modify source pipeline, exporter defaults, original Action, rig, FPS, retargeting, production timebase, checkpoints or historical Worklogs; no optimization, heuristics, model inference, training, new editor/style system, new media, engine integration or implementation.
**DECISION ENABLED:** Is the temporal loss serialization-side, importer-side, or still unresolved? What evidence would justify a later **human-owned** FBX animation contract decision? CASE A does not authorize any production change.

## DIRECTION / PURPOSE / CENTRAL INTENT — AGENT CONTRACT

Work only on the existing `kn99e39/AnimCV` branch `arch/single_frame_first`. Read `AGENTS.md`, `PRODUCT_VISION.md`, `Architecture_v3_FramePose.md`, `Architecture_v2.md` (Motion Graph, Blender, keyframe policy), `COORDINATE_CONVENTIONS.md`, both cumulative audits, W99A/W100 and W105–W107, `src/blender/export.py`, W106/W107 diagnostic scripts and tests. W107's CASE A supports **integer-frame** FBX geometry only.

**H108:** The source's held-pose temporal function is not preserved at a discrete FBX export/import interpolation boundary even though W107 V1 integer-frame keys are adequate. If so, source, FBX and imported curve semantics should differ reproducibly; an isolated post-import scratch interpolation change should materially reduce subframe error without altering any integer-frame key value. If false, retain the failure and competing explanations. Never tune to make H108 true.

## HISTORICAL BASELINE — PRESERVE

1. W107 COMPLETE; W107 manifest SHA256 `20fcb1b9f6ec8879247ac91ae3a21ca1185e995340976079bc80936316f57278`; declaration SHA256 `aeb9ce710d01878e8d55a0b26d1d117e9ddfc9868ae397c6d88d1567975f5a76`; W106 manifest SHA256 `f3003fe54e68c6cf182313c0b784468275b7ed039f50f574ad10e64eb8fb97db`. Verify actual Main-local files/hashes, originals and complete records before execution; stop if absent or mismatched. No silent reconstruction or replacement.
2. Fixed cases: `complete_control`, `arm_natural_most` (reused 3DPW benchmark-detector observation) and `real_clip` (historical GT-free real AnimCV observation). Select precisely the declared `FIXED` M_Main Actions, never the saved active `RUNTIME` R11. Synthetic controls separately labelled; no pooled inference. Original source curves are CONSTANT, scene 30 fps, real clip native 50 fps indices. Do not alter.
3. Freeze V0 original `src/blender/export.py` default path, W107 V1 isolated no-simplification diagnostic, exact source/rig/Action/slot/timing hashes, key counts, values and interpolation. Retain the published integer and half-frame failures, frame_end=250, 16/4/2 takes, real-clip 50→30 fps mismatch, and V1 2.2–2.8× FBX size and 6–13× imported key costs. Do not call V1 a production fix.

## CHANGE ONLY / DO NOT TUNE

**CHANGE ONLY:** New W108 isolated diagnostic scripts, focused tests, W108 Worklog and scratch-only copies/FBX/NPZ/JSON/logs in a fresh W108 output directory. Read-only reuse of W106/W107 artifacts/scripts. One declared intervention may change the **interpolation property only** on a *copied, imported* V1 Action's F-curves to CONSTANT; preserve all key times, values, counts, Action assignments, rig and scene. If not possible without other mutation, mark NOT TESTABLE.

**DO NOT TUNE:** No parameter sweep, arbitrary thresholds, key insertion/removal, hand-built epsilon step keys, bake-step changes, scene FPS changes, interpolation selection by bone/frame, source Action rewrite, source rest/rig transform edits, re-targeting, exporter/importer patch, adaptive sample selection or post-hoc gate changes. No production code changes.

## BOUNDED EXPERIMENT AND CONTROLS

**Preflight:** Fetch remote and confirm matching branch, clean Main working tree, no running previous Batch, matching worker lock, Blender 5.1.0 / FBX add-on 5.15.0, intact W106/W107 artifacts and digests, frozen `src/blender/export.py`, exact Action/slot/rest/timing identities. STOP on conflicts, missing binaries, dirty originals, unexpected environment or resource cost. Main CPU/Blender only; no LabServer, GPU, training, network download or inference.

**Predeclaration:** Before any W108 outcome inspection, hash a declaration containing cases, source/FBX/import identities, sample windows, exact integer and quarter/half/three-quarter frame times, world-space and armature-space error definitions, thresholds, synthetic controls, the one scratch intervention, negative controls, expected signatures, case mapping, resource cap and stop conditions. The declaration must bind all outputs.

**P0 — reproduce:** On immutable scratch copies, replay or verify W107 V0/V1 source/imported samples and hashes. Reproduce reported integer-frame and half-frame results within predeclared deterministic tolerances. If P0 fails, CASE D. Confirm unique named Action and all source curve/scene timing metadata. Keep V0's simplification failure separate from V1's interpolation failure.

**P1 — boundary tracing:** Sample the original Blender Action, W107 V1 FBX animation curve representation, and default-imported Action at the **same exact times**, including integer, +0.25, +0.50, +0.75 frames across fixed transition and quiet windows. Inspect actual FBX key times/values and interpolation attributes/flags if independently decodable; state if the FBX parser cannot reliably identify interpolation semantics. Quantify world head/tail max/p95/median and per-bone/per-frame errors, object transform and local/armature-space residuals. Distinguish source hold semantics from FBX curve interpolation and importer defaults; do not assert export-side vs importer-side ownership without direct encoding evidence.

**P2 — single scratch intervention:** From a fresh V1 import, copy only the selected imported Action and set the predeclared F-curve interpolation properties to CONSTANT. No key time/value/count, rig, Action slot, FPS or source changes. Verify integer-frame parity and byte-equivalent key data before/after the intervention, then measure exactly the P1 subframe samples. If the imported representation is incompatible, label NOT TESTABLE. This is a diagnostic of representational sufficiency, not an exporter/importer fix or proof of FBX interoperability.

**P3 — synthetic negative controls:** A minimal held-pose/step Action and a separate genuinely smooth LINEAR Action, each with known continuous-time reference, fixed integer samples and identical rig/timing setup. Apply the same V0/V1 import path and P2 diagnostic. Step control should reveal hold→linear loss; smooth control should show the harm from forcing CONSTANT when motion should be continuous. Do not pool synthetic with the three real Actions. If rest-geometry errors confound a synthetic bone-tail metric (as in W107 V3), isolate time-varying head/rotation error and mark the original gate confounded.

**Evidence and regressions:** Deterministic fresh-copy repeat of P0 and P2; focused W108 tests and relevant W106/W107 Action/FBX contract regressions only. Preserve logs, declaration/manifest SHA256, source and scratch digests, Action/slot names, Blender/add-on versions, FBX curve metadata, key counts, time samples, negative results, unexecuted conditions, per-case tables, command reproduction and runtime/cost. Only a few preselected rendered frames if numerical ambiguity requires them. No blanket production regression absent unexpected shared-file changes (then STOP).

## PREDECLARED CASE / STOP CONDITIONS

- **CASE A — MECHANISM SUPPORTED:** P0/identities/controls reproduce; actual source/FBX/import evidence identifies a temporal-semantics boundary; P2 reduces subframe error materially on **all three** real Actions without changing integer-frame world motion or key data; synthetic step and smooth controls show the predicted contrast; fresh repeat agrees. Explicitly distinguish a proven boundary from any remaining exporter-vs-importer ambiguity and report residual error.
- **CASE B — FALSIFIED:** Valid reproducible P0 and controls, but the interpolation hypothesis fails to explain the error or P2 does not improve subframe fidelity. Preserve negative evidence.
- **CASE C — PARTIAL/CONFOUNDED:** Mixed real results, unreliable FBX interpolation flags, untestable intervention, synthetic rest confound, or insufficient boundary ownership. No unique causal claim.
- **CASE D — INVALID:** P0 mismatch, identity/timebase/Action/rig failure, declaration breach, altered originals, nondeterminism, concurrent Agent or invalid controls. STOP and report, never relabel.
**STOP** on any original mutation, GT leakage, historical artifact overwrite, unauthorized code change, Git divergence, running Agent, unexpected environment/cost, sample-time mismatch or attempts to tune thresholds/keys/FPS to pass. Do not ask the coding Agent to recommend W109.

## FINAL EVIDENCE-ONLY REPORT

Create `docs/108_WORKLOG_FBX_SUBFRAME_INTERPOLATION_OWNERSHIP.md` documenting Human Assessment (success/failure/uncertainty/alternative explanations), immutable identities, declaration and evidence hashes, all tested/not-tested probes, real and synthetic quantitative subframe and integer results, actual first observed loss boundary or explicit uncertainty, negative controls, smooth-motion harm, tests/regressions, execution failures, resource cost, reproducible commands and limitations. Keep frame_end, take ownership, real-clip 50→30 fps, export policy, non-Blender consumers and user-facing authoring explicitly unresolved. Do not modify production.

## COMPLETION CONDITION

Commit the bounded W108 evidence and honest CASE A/B/C/D on **the same branch**. Append `AUTOMATION_BATCH_STATUS: COMPLETE` and `BATCH_NUMBER: 108` as the Worklog's final lines **only after** all test/report/manifest evidence is finalized and committed. On blocked/invalid work retain partial failures without premature COMPLETE. Never modify `docs/next_batch.md`, historical Worklogs or branches. The coding Agent must not design or publish the next Batch.
