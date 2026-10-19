# Worklog — Skinned BaseRig Mannequin (2026-10-18)

Minor worklog attached to Worklog 90. It merges two separate drafts, the first build of the mannequin (formerly `91`)
and its low-poly revision (formerly `92`), into a single minor entry. Both sections below keep the substance of the
original records.

## 1. First build: an anatomical mannequin

### Purpose and work

A human mannequin was modelled and skinned onto `examples/BaseRig.fbx`, which contains only an armature. At the user's
request it included shapes to tell front from back:

- nose, jaw and ears
- chest and back
- buttocks
- toes

**Outputs** (in `examples/skinned_mannequin/`):

- FBX
- editable blend
- previews
- validation JSON

The existing BaseRig and the existing examples were not modified.

**How it was built** (local Blender 5.1.2):

- procedural mesh generation
- surface joining and clean-up
- normalized weights with at most four bones per vertex

The original rig has no separate head or pelvis bones, so skinning was done inside the existing hierarchy.

**Scripts:** `scripts/build_skinned_mannequin.py` regenerates the asset and `scripts/verify_skinned_mannequin.py`
validates it. Usage and the lasting asset description are in the asset folder's README.

### Checks and limitations

**Checks**

- The FBX was re-imported into a fresh scene and verified for:
  - the names and parent relations of all 51 original bones
  - the world rest matrices, within tolerance
  - the Armature modifier
  - weight normalization on every vertex
- Individual rotations were applied to the arm, calf, spine, neck and index finger, confirming finite vertex
  displacement.
- Front, back and three-quarter renders, plus a visual check of a bent pose, were added.
- The exact figures and hashes are in `examples/skinned_mannequin/validation.json`.

**Limitations**

- This is asset-authoring validation. No perception evaluation was run in any regime (`oracle_geometry`,
  `benchmark_detector_observation` or `real_animcv_observation`).
- No model quality gate is claimed, and no pass through the full AnimCV pipeline.
- The surface is procedural, so extreme joint poses may need extra weight painting or corrective shapes.
- Rendering used the CPU only. No training or GPU experiment was run.

## 2. Revision: a low-poly mannequin (replaces the first build)

**Why:** the user did not want the anatomical detail of the first mannequin, so it was rebuilt as a simple low-poly
figure. The record of the first build is kept above, but the current asset is this revision.

**What changed**

- Removed:
  - rounded muscles
  - the detailed chest and buttock volumes
  - ears and jaw
- Simplified to an angular torso and limbs with mitten-shaped hands.
- Only a small nose and toe tips remain, to tell front from back.
- The material is slate-blue.

**Updated files:** `examples/skinned_mannequin/BaseRig_Skinned.fbx`, the blend, the previews and the validation JSON.

**Mesh**

- 698 vertices and 842 polygons.
- Each body part is an overlapping low-poly surface inside a single mesh object, not a connected organic skin surface.

**Checks passed**

- The original 51 bones are kept, with normalized skinning.
- FBX re-import passed.
- The per-bone deformation checks passed.

**Not verified**

- The finger bones are kept, but no per-finger mesh is modelled, so individual finger deformation is not verified.
- This is asset authoring plus a CPU render check. It is not a perception evaluation regime, a model quality gate or a
  full-pipeline evaluation.
- Collisions and gaps in extreme poses are not guaranteed against.

Usage is in the asset folder README and the figures are in `validation.json`.

## 3. Relation to Worklog 90 (added when the drafts were merged)

**Bare rig in W90.** The Worklog 90 rig replay used the bare `examples/BaseRig.fbx` and showed the posed armature
through bone proxies. This asset did not exist yet when that replay was produced.

**Same rest pose as W90.** On 2026-10-18 the re-imported `BaseRig_Skinned.fbx` was compared with the authoritative rest
snapshot `examples/e2e_demo/baserig_blender_rest.json` (Blender 5.1.2):

- the bone set and parent relations are identical;
- the armature world matrix is identical (scale 0.01; difference 1e-9);
- the largest rest-matrix and head/tail difference is 6e-5 armature units (cm), which is FBX re-export round-off.

So the local quaternions of the W61/W76 IK can drive this mesh unchanged.

**One caveat when loading it.** The existing `bind_imported_armature` compares rest-pose digests, and that digest
includes the source FBX hash and file name. It therefore rejects this file even though the bones match. Using the
mannequin in a replay needs a bone-data comparison with a tolerance instead.

## Repository state

- **Before the work:** the drafts recorded a fetch with GitHub Desktop's bundled Git, confirming that
  `arch/single_frame_first` matched the remote. macOS's default Git would not run because the Xcode licence had not
  been accepted.
- **During the work:** no branch was created or switched and nothing was committed or pushed, so the server branch is
  unchanged.
- **Doc merge:** this file replaces the uncommitted drafts `docs/91_WORKLOG_SKINNED_BASERIG_MANNEQUIN.md` and
  `docs/92_WORKLOG_LOW_POLY_MANNEQUIN_REVISION.md`.
