# Worklog — FramePose Retargeting Boundary and Canonical 3D Direction Control (2026-09-30)

## Starting state and scope

Started from clean `arch/single_frame_first` at `aca373cc9b7f04f4f0fb7418c22065eeb527edbd`, after fetching origin and confirming it had no commits ahead. `docs/57_WORKLOG_VALID_ONLY_ROOT_ORIENTATION_CONTRACT.md` existed; this batch did not edit, rename, replace, or append to it. `docs/58_WORKLOG_FRAMEPOSE_RETARGETING_BOUNDARY.md` did not exist. Implementation commit: `2b3b343`.

AnimationSemantics v2 producer correctness was treated as closed. The only implementation is a separate direction-only contract candidate consuming `AnimationSemanticsV2` directly. It neither constructs MotionGraph nor invokes the historical RetargetSolver. `src/retarget/solver.py`, `fk_solver.py`, and `ik_solver.py` are unchanged historical baselines. Root Orientation, Contact, Reliability, FramePose, Pose Reconciliation, IK, and Blender production behavior were not modified.

## Contract audit and architecture decision

| Contract | Classification for FramePose | Finding |
|---|---|---|
| `RigProfile` / `BoneInfo` | KEEP WITH EXTENSION | Hierarchy and rest-local matrices are usable. An explicit `rest_world_matrix` takes precedence; otherwise the candidate composes parent rest-local matrices. A `local_axis_hint["primary"]` explicitly defines a bone-local length axis; otherwise a **unique child** rest-local translation defines it. Multiple children or missing/degenerate matrices are a contract gap. RigProfile does not declare canonical-camera-to-rig rest alignment, and equivalence between Assimp node bases and Blender imported pose-bone bases is unverified. |
| `BoneMappingProfile` / `BoneMappingEntry` | KEEP WITH EXTENSION | `mapping_mode="direction"` and its two source names bind the correct canonical joints. `axis_hint` historically names a 2D rotation axis, not the target bone's longitudinal rest axis, so the candidate never reinterprets it. A session-level canonical-to-rig coordinate alignment is still missing. |
| `AnimationClip` / `BoneTransformSample` | LEGACY for this path; CONTRACT GAP for direct reuse | `rotation` is non-nullable, `confidence=0` still carries a quaternion, and timestamps/source validity are absent. The Blender writer keys such samples and expects world-frame rotation deltas. A separate candidate sample with `source_status` and `rotation_status` preserves UNKNOWN source evidence versus UNAVAILABLE target transform, without changing this shared historical schema. |
| MotionGraph `RetargetSolver` | LEGACY | It uses MotionGraph 2D directions, optional depth, first-visible-frame reference, and last-valid holds. It is not called by the new candidate. |
| `IKChainEntry` | LEGACY for this batch | Its historical IK lengths come from the first valid observation. No FramePose IK contract or solver was added. |
| Blender executor/writer | KEEP as historical isolation boundary; CONTRACT GAP for new candidate | Blender accepts `AnimationClip` and converts world deltas using the imported bone's `matrix_local`. It has no path for candidate target-local quaternions or UNAVAILABLE rows. This batch did not alter the boundary. |

The historical **first-frame delta** sets the first usable video direction to identity even when the person begins with a bent or rotated limb. The canonical-to-rig candidate instead aligns each current 3D direction to a declared target rest direction. In the synthetic control, the first direction was already 90° from rest: historical first-frame delta was identity, while canonical-to-rest produced a 90° local quaternion at frame 0 with the same source frame, rig, and mapping. A video frame therefore cannot silently define the source rest pose.

The candidate requires an explicit `canonical_to_rig_rest` quaternion for the session. `None` is not replaced by identity, a first-frame estimate, or a BaseRig-specific axis. The candidate uses a deterministic shortest arc with zero twist; a single direction cannot measure anatomical twist. Parent-mapped rotations are converted into child target-local rotations relative to the mapped parent's world delta. Root Orientation stays a separate owner and is never applied independently to limbs. The current contracts do **not** provide the required session alignment or prove Assimp-to-Blender rest-basis equivalence, so this is a contract control, not a production retarget solver.

## Synthetic controls

Focused tests cover: source equals rest → identity; source rotated +90° → expected local quaternion; arbitrary explicit primary axis; rotated target rest matrix; finite deterministic normalized 180° reversal; invalid source endpoint → no quaternion; missing alignment or ambiguous target axis → explicit UNAVAILABLE; already-rotated first frame → non-identity; parent/child target-local conversion; no root-yaw application to limb; frame/timestamp preservation and new output round-trip. The candidate does not hold, interpolate, or substitute identity on missing evidence.

## BaseRig contract control

`examples/BaseRig.fbx` SHA-256 `71b33779793912c1f8a3b4a88aadc7ab369dbe226eaaeab5d5b92aa589c18eea`. The repository's `RigParser` parsed it into a 53-node `RigProfile` (`rig_id=BaseRig`, root `root`; profile digest `1082ae2b1925a5dfb2ec92e5f3b4852a454d8df72da5bb26cbca92b5c7462e20`). The LabServer63 container lacks `pyassimp`, so this exact RigParser output was transferred as JSON and read there through `load_rig_profile`; the server replay also hashed the unchanged FBX bytes. This provenance is recorded rather than claiming server-side FBX parsing.

`examples/demo_mapping_profile.json` names `upper_arm.L`, `forearm.L`, and `head`; **all three target names are absent** from BaseRig. Its `+Y` hints do not repair the names and are not a declared rest axis. For an existing BaseRig mapping control, `examples/e2e_demo/mapping.json` names `upperarm_l` and `lowerarm_l` with source pairs shoulder→elbow and elbow→wrist. Both target bones exist. Hierarchy is `clavicle_l → upperarm_l → lowerarm_l → hand_l`; rest-local matrices are present, rest-world matrices and explicit local-axis hints are absent. Each arm bone has one child, so its child offset gives an unambiguous bone-local rest direction approximately +Y. That target rest information is available **inside the RigProfile coordinate system**. Absolute canonical-camera-to-rig alignment remains undeclared; Assimp-vs-Blender pose-bone basis correspondence remains unverified.

## Current-policy AnimationSemantics replay

LabServer63 used persisted `framepose_current_valid_bilateral_v1` sidecars from docs/57, regime `benchmark_detector_observation`. Their bank digest is `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536`; H0 validation SHA-256 is `88a007fca42eee08833ec8ece453aa3160a26232738be01cd849cf5016524937`. Four representative cases plus an actual invalid mapped-endpoint frame were replayed. Every mapped row is in the saved candidate files with frame index, timestamp, source pair, structural source status, normalized canonical direction (when known), target-local quaternion or UNAVAILABLE reason, and provenance.

| Case frame | `upperarm_l` shoulder→elbow | `lowerarm_l` elbow→wrist | Target rotation |
|---|---|---|---|
| Known-good standing, dancing actor0 #424 | KNOWN `(0.008, 0.358, -0.934)` | KNOWN `(0.078, 0.041, -0.996)` | both UNAVAILABLE: alignment missing |
| Turning, crosscountry actor0 #202 | KNOWN `(0.433, -0.496, -0.753)` | UNKNOWN: endpoint invalid | upper UNAVAILABLE: alignment missing; lower UNAVAILABLE: source invalid |
| Dynamic, crosscountry actor0 #462 | KNOWN `(0.104, 0.338, -0.935)` | KNOWN `(0.040, -0.109, -0.993)` | both UNAVAILABLE: alignment missing |
| Tracking-loss review, hug actor1 #312 | KNOWN `(-0.026, -0.168, -0.986)` | KNOWN `(-0.745, -0.377, -0.550)` | both UNAVAILABLE: alignment missing |
| Actual invalid endpoint, hug actor1 #406 | UNKNOWN | UNKNOWN | both UNAVAILABLE: source invalid |

The full-sequence replay counts valid source directions: dancing actor0 `93/115` upper and `79/115` lower; crosscountry actor0 `147/162` upper and `133/162` lower; hug actor1 `185/187` upper and `183/187` lower. There are **zero known target-local quaternions** on these real sequences because the alignment is absent. Quaternion continuity is therefore **not evaluable**, not a claim of zero discontinuities. Source loss does not trigger a previous-rotation hold. No Blender visual export was made because the target basis contract is unresolved; no axes were manually adjusted for BaseRig.

## Artifacts, tests, and verdict

Server artifact: `~/animcv-output/framepose_retarget_boundary/` (`report.json` SHA-256 `25a5e0fff33e076f39674e27e7959c0db6abb7487b1646126128e25d8d5c3954`, five full-sequence candidate files). Exact implementation files: `src/retarget/framepose_direction_candidate.py`, `scripts/run_framepose_retarget_boundary.py`, `tests/test_framepose_direction_candidate.py`; this new worklog is the only documentation file in the batch. Focused candidate plus historical retarget/rig tests: **65 passed**. No shared `RigProfile`, `BoneMappingProfile`, `AnimationClip`, or Blender writer code changed, so a full repository regression was not required or run.

**Verdict: NO for the current BaseRig/mapping contracts.** Canonical metric 3D articulation can drive deterministic target-local direction rotations in synthetic controls when an explicit coordinate alignment and target rest basis are supplied. The current real path lacks a declared canonical-camera-to-rig rest alignment, the demo mapping names no BaseRig bones, and Assimp rest bases are not verified against Blender's imported pose-bone bases. The next architecture work must close those exact contracts before a production FramePose FK/IK retarget solver or Blender export can be claimed. No IK or downstream animation implementation began here.
