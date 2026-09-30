# Worklog — Valid-only Root Orientation contract repair (2026-09-30)

## Scope and starting state

Started from clean `arch/single_frame_first` at `f642d93` (docs/56), fetched origin, and confirmed no commits ahead. Code commit: `88b6757`. This is one bounded v2 Root Orientation producer correction before IK or retargeting. Regime for the frozen H0 signal: `benchmark_detector_observation`; 3DPW oracle torso geometry is evaluation only.

docs/56 used `pose.root_orientation_diagnostic.observe_yaw`. That diagnostic intentionally lets invalid bilateral pairs contribute to yaw, then reports reliability separately. The v2 bridge previously checked only that *some* valid pair existed, so an invalid second pair could still alter a KNOWN heading. The repair makes the production calculation itself filter invalid pairs. The diagnostic is unchanged.

## Contract and compatibility

The production policy is now `framepose_current_valid_bilateral_v1` (before: `framepose_current_bilateral_v1`). For each current frame, it considers the same canonical shoulder and hip pairs, angle convention, geometric length and confidence weights, and circular fusion as docs/55. A pair contributes only if both endpoints are `observation_valid` and its geometry/weight is defined. Shoulder-only and hip-only evidence can each give KNOWN yaw; with no contributing pair, yaw is UNKNOWN. There is no previous-frame carry, zero substitute, hold, release, smoothing, or oracle runtime input.

The `animcv_animation_semantics_v2` schema did **not** change: its Root Orientation fields and serialization still mean KNOWN/UNKNOWN plus yaw. The policy identifier in provenance distinguishes the two v2 producer meanings. The default v2 loader rejects docs/56's old policy. `load_animation_semantics_v2(..., allow_legacy_policy=True)` reads those sidecars explicitly and preserves their serialized yaw and digest. The review renderer has the matching explicit `--allow-legacy-v2-policy` option. v1 and its historical held-yaw meaning remain separate.

`src/pose/root_motion.py`, `src/pose/root_orientation_diagnostic.py`, all Contact modules and calibration logic, frozen H0, FrameBank, docs/51-56, v1 and old v2 artifacts were not rewritten. The new replay copies docs/56's calibration file byte-for-byte; its SHA-256 is `22140d162dd65d874313c112173d8e968b85dade0577849462f6a29b8f94cf41` in both directories. It checks all articulation, Contact, reliability, unavailable fields, frame indexes, timestamps, and non-orientation provenance for exact equality between old and corrected sidecars.

## Frozen control replay

LabServer63, `animcv-framepose:cuda118`, same 16 validation sequences and 3,407 rows as docs/55-56. Bank content digest `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536`; H0 TRAIN SHA-256 `43a5d53bee99eed6159b2a83107235632add91912cb985a9eae0900db65c6af6`; H0 validation SHA-256 `88a007fca42eee08833ec8ece453aa3160a26232738be01cd849cf5016524937`.

| Structurally contributing sources | Rows | Old-to-new angular delta | Oracle error before → after | Corrected UNKNOWN |
|---|---:|---:|---:|---:|
| Both shoulder and hip | 3,347 | **exactly 0° on every row** | mean 5.894° → 5.894° | 0 |
| Exactly one | 60 | mean 25.024°, median 20.209°, p95 59.954°, max 160.059° | mean **32.657° → 17.346°**, median 28.572° → 13.473°, p95 66.892° → 42.929° | 0 |
| Neither | 0 | — | — | 0 measured |

All 60 partially valid rows had **hips only**; the frozen subset contains no shoulder-only or neither-valid row. Unit tests exercise both missing categories and require UNKNOWN when neither pair can contribute. Both-valid rows are bitwise equal to the docs/56 current yaw. The old diagnostic yaw stored in each docs/56 sidecar was verified against `observe_yaw` before comparison. The new policy still has 3,407 KNOWN and 0 UNKNOWN on this frozen validation subset; no coverage was invented for invalid-only input.

The report records every partial-valid row's old diagnostic-style yaw, corrected yaw, angular change, and oracle errors. In those 60 rows, 41 improve, 9 worsen, and 10 are unchanged. The largest degradation is `courtyard_rangeOfMotions_01:actor0#295`, oracle error `9.52° → 41.79°` (+32.27°): an invalid shoulder contribution happened to pull the old fused result closer to oracle. Conversely, `courtyard_rangeOfMotions_01:actor0#298` changes yaw by 160.06° and reduces oracle error `101.07° → 58.99°`. The corrected policy is contractually preferred despite the worsened rows; invalid evidence cannot be retained just because it sometimes helps by chance. Across all rows, mean oracle error is about `6.37° → 6.10°`, but the category-specific accounting is the decision evidence.

No Contact state, calibration, physical-time rule, percentile, or horizon changed. Root translation and ground height remain UNAVAILABLE. No Pose Completion, Root Motion, IK, or retargeting work began.

## Artifacts, files, and tests

New separate server artifact: `~/animcv-output/animation_semantics_v2_valid_yaw_replay/` (`report.json` SHA-256 `19f997456175380a635524bb1e2599a284a0b3f38679b3879abae1461c8fb285`, 16 corrected v2 sidecars, byte-identical contact calibration). The docs/56 `~/animcv-output/animation_semantics_v2_replay/` directory remains intact.

Changed files: `src/motion/animation_semantics_v2_bridge.py`, `src/motion/animation_semantics_v2.py`, `scripts/run_root_orientation_validity_replay.py`, `scripts/render_animation_semantics_review.py`, `tests/test_animation_semantics_v2.py`, and this worklog. The final focused suite covered AnimationSemantics v1/v2, Root Orientation, Contact, FramePose bridge, and H0 replay contracts: **79 passed** after the final code change. A full repository regression was not run; historical/shared producers and Contact were unchanged, and the frozen-data replay checked the exact bridge outputs and serialization on all validation rows.

**Verdict:** Every KNOWN v2 Root Orientation now depends only on structurally valid current-frame bilateral torso evidence. AnimationSemantics producer correctness is closed for its supported quantities. This does not assert physical contact, world root motion, IK, or retargeting correctness.
