# Worklog — Whole-Body Root Orientation at the Armature Object (2026-09-30)

## Starting state and publication check

Fetched `origin/arch/single_frame_first` before editing. Local and remote HEAD were both `7f6a9095b717b82910899af43dc9f0c0afb852cd`, the worktree was clean, and implementation `1deb01a` plus Worklog 59 `7f6a909` were already committed and published. This batch did not recreate or edit Worklogs 57–59. Implementation commit: `24c7144`.

The Worklog 59 Blender rest snapshot, fixed BaseRig calibration, direction mappings, heading-relative source articulation, rest head→tail directions, shortest-arc/zero-twist limb FK mathematics, and explicit UNAVAILABLE semantics were preserved. No FramePose, Root Orientation estimator, Contact, Reliability, BaseRig, IK, or legacy MotionGraph retarget module changed.

## Root Orientation ownership contract

Root Orientation now has one explicit whole-rig owner: the imported **Blender Armature Object**. The existing `framepose_fk_calibration.json` field `root_orientation_target_bone=spine_01` remains a fixed **calibration anchor** for deriving the rig basis; in this path it is not a pose-bone yaw owner. The separate object-owned FK result stores one `RigRotationSample` per frame (frame index, timestamp, known/unknown source status, nullable object-space rotation, availability reason) alongside the unchanged pose-bone FK samples and provenance. It does not encode object yaw as a legacy `BoneTransformSample` or a pose-bone sample.

Let `A` be Worklog 59's fixed canonical-heading-zero → imported armature-rest rotation. For the existing AnimationSemantics yaw `ψ`, the armature-space delta is `A · Rz(ψ) · A⁻¹`. The Blender adapter composes the imported Armature Object rest rotation with that delta (`Robject,rest · A · Rz(ψ) · A⁻¹`), and writes it to the **object** rotation channel. No input video frame is used as a rest or yaw calibration frame. It writes the same Worklog 59 heading-relative local FK quaternions to mapped pose bones. It leaves object location and scale unchanged; root translation and ground placement remain UNAVAILABLE. The adapter requires the actual imported rest snapshot to match, and currently requires an unparented armature with positive uniform object scale. An UNKNOWN yaw produces no object quaternion, no hold/zero substitute, and the bounded Blender adapter refuses that frame before mutation.

The only change to `src/retarget/framepose_fk.py` is an explicit `root_orientation_owner="armature_object"` mode: it permits independently rooted mapped branches and **omits** the old `spine_01` yaw pose-bone sample. The direction extraction, yaw removal, shortest-arc swing, parent-relative local conversion, and rest-basis conjugation are untouched. The default bone-owned mode and Worklog 59 output remain available for controlled comparison.

## Synthetic multi-root contract

A synthetic armature with independent `body → upper` and `thigh → calf` branches used the same local articulation at body yaw 0°, 45°, and 90°. One object rotation changed the world orientation inherited by **both** roots, while upper-arm and calf local FK quaternions remained invariant. With yaw fixed, changing real arm and leg articulation changed their respective local FK quaternions. No root pose-bone yaw sample was emitted. Invalid leg evidence remained UNAVAILABLE without holding a previous quaternion; UNKNOWN Root Orientation stayed explicit and the review adapter rejected it. Serialization and unit quaternion requirements passed.

## BaseRig hierarchy and real replay

The actual Blender-imported BaseRig has 51 bones under one Armature Object and five independent top-level bones: `spine_01`, `thigh_l`, `thigh_r`, `interaction`, `center_of_mass`. Under Worklog 59's `spine_01` pose-bone owner, both leg roots bypass yaw. Under object ownership, Blender ID-pointer and evaluated pose-matrix checks confirmed that **all five** belong to the same Armature Object and inherit its one object rotation. No synthetic root, re-parenting, or per-branch yaw was used.

LabServer63 replayed the same docs/57 current-policy AnimationSemantics v2 lineage, regime **`benchmark_detector_observation`**, with the same BaseRig FBX, Blender rest snapshot, calibration, mapping, and source frames. Results: `~/animcv-output/framepose_object_owner_replay/` (`report.json` SHA-256 `9a0bc5de7af028625509afb01f96a88345ad8a72915f4151c51938bbbc0d9fd0`; full per-frame object/limb samples for three sequences). The new pose-bone FK samples matched Worklog 59 **exactly for all 928 mapped limb rows** (230 dancing, 324 crosscountry, 374 hug), including status and UNAVAILABLE reasons.

| Current-policy sequence | Object yaw known / unknown | `upperarm_l` known / unavailable | `lowerarm_l` known / unavailable |
|---|---:|---:|---:|
| dancing actor0 | 115 / 0 | 93 / 22 | 79 / 36 |
| crosscountry actor0 | 162 / 0 | 147 / 15 | 133 / 29 |
| hug actor1 | 187 / 0 | 185 / 2 | 183 / 4 |

The replay includes known-good dancing #424, dynamic crosscountry #462, turning #202 and #211, tracking-loss review hug #312, and actual invalid mapped endpoints hug #406. Root yaw was known for all evaluated real rows; UNKNOWN behavior is covered synthetically. At turning #202, the forearm remains UNAVAILABLE, so the fully-known #211 was used for Blender A/B. No continuity smoothing or missing-state conversion was added.

## Blender turning A/B and visual review

The bounded Blender 5.1.2 review used the identical BaseRig, snapshot, calibration, current-policy frame #211, and exact same upper-arm/forearm local FK samples for both ownership variants. In A (`spine_01` pose-bone yaw), upper body turned while leg branches stayed at rest. In B (Armature Object yaw), upper body and both leg branches turned together, while `spine_01` and the other top-level pose-bone yaw channels stayed identity. Blender reported all five top-level roots on the same object and a maximum inherited-object-rotation discrepancy below **0.05°** from evaluated float matrices. Object location and scale were unchanged. The mapped source-to-rendered bone direction error in the three object-owned real poses was at most **0.025°** at report precision.

The review rendered rest, known-good #424, dynamic #462, turning #211, and the Worklog 59 bone-owned turning #211 control as isolated fully-known poses, without keyframe interpolation. Its report, five PNGs, and five `.blend` files are in `~/animcv-output/framepose_object_owner_blender_review/` (`report.json` SHA-256 `4a99b06f2ccc89799c5914cc715cb7f9d45a86099c44ac2c3003030cbffab591`). Blender extraction/render ran with the macOS installed Blender because LabServer63 has no Blender binary; numeric replay ran on LabServer63. Bone proxies were sufficient; no character mesh or final FBX acceptance was claimed.

## Files, tests, and verdict

Implementation commit `24c7144` changed exactly `src/retarget/framepose_fk.py` and added `src/retarget/framepose_object_owner.py`, `src/blender/framepose_object_owner_adapter.py`, `scripts/run_framepose_object_owner_replay.py`, `scripts/review_framepose_object_owner_blender.py`, and `tests/test_framepose_object_owner.py`. This new `docs/60_WORKLOG_WHOLE_BODY_ROOT_ORIENTATION.md` is the only worklog added in the batch. New multi-root, Worklog 59 FK, docs/58 candidate, Blender adapter/keyframe/script, rig/mapping, and historical retarget focused tests: **96 passed**. No shared legacy Blender or serialization contract changed, so full repository regression was not required or run.

**Verdict: YES for the initial BaseRig whole-body orientation + mapped local FK contract.** One Armature Object rotation reaches all independently rooted skeletal branches, while the Worklog 59 local limb rotations remain byte-for-byte unchanged and missing evidence stays explicit. This does not establish root translation, foot placement, IK, visual motion quality across long sequences, or arbitrary parented/non-uniform-scale armatures. Stop before IK; the next architecture batch may address end-effector constraints separately.
