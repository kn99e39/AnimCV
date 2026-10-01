# Worklog — FramePose Two-Bone IK / End-Effector Constraint (2026-10-01)

## Starting state

Fetched `origin/arch/single_frame_first` before editing. Local and remote HEAD were both `f87df9d` (Worklog 60), and the worktree was clean. Implementation commits: `f14af46` (IK module, chain contract, replay, tests) and `86586dd` (Blender A/B review script). LabServer63 was fast-forwarded to `f14af46` before the replay.

No existing file was modified. Worklog 59/60 FK (`framepose_fk.py`, `framepose_object_owner.py`), the Blender rest snapshot, rig calibration, the BaseRig FBX, the direction mapping, AnimationSemantics v2, and the historical MotionGraph `retarget/ik_solver.py` were untouched. The new module imports FK's private rest helpers (`_rest_direction`, `_rest_rotation`, `_q_unit`, `derive_rig_alignment`) read-only, the same way `framepose_object_owner.py` already does.

## IK chain contract

`examples/e2e_demo/framepose_ik_chains.json` (schema `animcv_framepose_two_bone_ik_chains_v1`) is a separate, rig-specific contract. The historical `ik_chains` / `IKChainEntry` field was not reused or reinterpreted:

| field | BaseRig `left_arm` |
|---|---|
| source root / mid / end | `left_shoulder` / `left_elbow` / `left_wrist` |
| target proximal / distal | `upperarm_l` / `lowerarm_l` |
| target end anchor | `hand_l` |

When the contract is loaded, the Blender rest snapshot is checked for the chain's shape: the distal bone must be the proximal bone's child, the proximal tail must equal the distal head (connected, relative tolerance 1e-6), and the end anchor must be the distal bone's child, sitting at the distal tail. No source frame is involved, so one contract serves every video.

## Dimensionless endpoint target

The source articulation is made heading-relative exactly as in FK. Root yaw is removed with `Rz(−ψ)`, then the vectors are mapped through the fixed rig alignment `A`:

```
s_u = A·Rz(−ψ)·(mid − root)        s_l = A·Rz(−ψ)·(end − mid)
reach_fraction = |s_u + s_l| / (|s_u| + |s_l|)
e              = (s_u + s_l) / |s_u + s_l|
requested_endpoint = proximal_rest_head + reach_fraction · (L1 + L2) · e
```

The DIRECTION text wrote `normalize(root−end)`. The implementation uses the root→end direction, `end − root`, which is what that text intends. Only the ratio and direction of the source vectors are used, so source body size, target video calibration and frame 0 do not enter. The formula is also written into every result's provenance (`endpoint_formula`).

The chain root is the proximal rest head. That holds because no ancestor of `upperarm_l` is posed in the current mapping (clavicle stays at rest).

## Target Blender rest lengths (BaseRig)

All values below come from `baserig_blender_rest.json` (Blender 5.1.2 import, FBX SHA `71b33779…`), in armature units. The armature's object scale is 0.01.

| quantity | value |
|---|---:|
| `upperarm_l` L1 | 27.771168 |
| `lowerarm_l` L2 | 27.251084 |
| L1 / L2 | 1.0191 |
| reach interval `[|L1−L2|, L1+L2]` | [0.520084, 55.022252] |
| rest elbow angle (between upper and lower directions) | 38.98° |

The topology checks passed: the `lowerarm_l` head equals the `upperarm_l` tail, and the `hand_l` head equals the `lowerarm_l` tail. The Assimp RigProfile was not used.

## Reach and clamp semantics

`solve_two_bone` clamps deterministically to `[|L1−L2|, L1+L2]` and records the result as `none`, `clamped_to_max_reach` or `clamped_to_min_reach`. L1 and L2 are never changed. By the triangle inequality `reach_fraction ≤ 1`, so the source contract cannot request more than the maximum reach (only float rounding is clipped to 1). The max-reach clamp is therefore exercised through the solver directly. A below-minimum request occurs for real when a strongly folded source meets a target with unequal segments; this is tested on the full pipeline. The endpoint error is reported against the **unclamped** requested endpoint, so a clamp shows up as nonzero IK error.

## Bend-plane ownership

The bend side is taken only from the current frame's mid joint. `side = normalize(s_u − (s_u·e)e)` is the source elbow's offset from the root→end line, and the elbow is placed at `root + L1(cos α·e + sin α·side)` using the law of cosines. There is no fixed elbow-up, no pole from a previous frame, no temporal hold and no user pole. Local quaternions use the FK convention: world shortest-arc swing per bone, zero axial twist, parent-relative, conjugated by the Blender rest basis.

**Degenerate bend.** A chain is degenerate when `|û × l̂| < sin(1°)`, i.e. straight or folded within 1°. This fixed geometric definition was declared before any replay and not fitted to data. Such frames get `bend_plane_status = degenerate_rest_plane_fallback`. The side is then `± e × n_rest`, where `n_rest` is the target rest bend normal and the sign reproduces the rest elbow side. This is a rig fact and is never labelled observed. If `n_rest` is undefined, or parallel to `e`, the frame is UNAVAILABLE with `bend_plane_undefined`. Observed and fallback frames are reported separately.

**Root orientation.** It stays with the Blender Armature Object (`A·Rz(ψ)·A⁻¹`, Worklog 60). IK works only on heading-relative vectors. No yaw is added to `upperarm_l`/`lowerarm_l`, the object location is not written, and root translation and ground placement stay UNAVAILABLE.

**Per-frame output.** Each `TwoBoneIkSample` records:
- frame index and timestamp
- source status and yaw status
- reach fraction and bend angle
- bend-plane status
- requested, reachable and solved endpoints, plus the solved mid
- endpoint error and clamp status
- proximal and distal local quaternion with their status
- reason

Provenance holds the rest, calibration, contract and semantics digests. UNKNOWN or invalid evidence yields UNAVAILABLE for both bones together, with no quaternion or pose and no hold from a previous frame.

## Synthetic controls (all passed)

The synthetic rig uses identity alignment and a 90°-bent rest arm. The FK baseline is the unchanged `solve_object_owned_fk`, and both FK and IK endpoints are evaluated with the same forward kinematics over the rest geometry (`chain_forward_kinematics`).

- **Matched proportions** (source 0.6/0.4, target 1.2/0.8; elbow at 20°, 75° and 130°):
  - FK endpoint = IK endpoint, and both errors ≤ 1e-9.
  - FK and IK quaternions differ by ≤ 1e-5°.
  - IK invents no new pose when FK already meets the contract.
- **Proportion mismatch** (source 0.6/0.4, target 0.4/0.6, 80° bend):
  - FK endpoint error > 0.05 (on a chain of length 1).
  - IK error ≤ 1e-9, with no clamp and an observed bend.
  - Forward-kinematics lengths are exactly 0.4 and 0.6 (1e-12).
  - The target elbow is on the same side of the root→end line as the source elbow.
- **Body yaw 0°/45°/90°:** the Armature Object rotation changes. IK local quaternions and requested endpoints are identical to 1e-12, so no yaw leaks.
- **Articulation sensitivity at fixed yaw:** moving the wrist, flipping the elbow side, or straightening the arm (higher reach fraction) each changes the IK locals. A flipped source bend flips the target elbow side.
- **Scale invariance:** multiplying source coordinates by 0.5 or by 3 gives identical reach fraction, endpoint and quaternions (1e-12).
- **Boundaries:**
  - Exactly `L1+L2` and exactly `|L1−L2|` report `none`.
  - A request of 1.5 over a max reach of 1.0 gives `clamped_to_max_reach`.
  - A request of 0.05 under a min reach of 0.2 gives `clamped_to_min_reach`.
  - In the pipeline, an equal-segment source folded to 170° against a 0.4/0.6 target gives `clamped_to_min_reach`, with endpoint error equal to `0.2 − fraction`.
  - All outputs are finite and lengths stay exact.
- **Degenerate or invalid:**
  - Bends of 0.5° and 0° use the declared fallback, never `observed`. A fully straight source gives fraction 1 and no clamp.
  - On a straight-rest rig the fallback is `bend_plane_undefined`, and the frame is UNAVAILABLE.
  - An invalid root, mid or end gives `source_{root,mid,end}_invalid`; a zero segment gives `source_segment_degenerate`; UNKNOWN yaw gives `root_orientation_unknown` (with the source still `known`). No NaN appears in any of these.
  - The first valid frame after a gap is solved only from its own evidence, with no hold.
- **Determinism, serialization, quaternions:** repeated runs are byte-identical, the result survives a round trip, and quaternions have unit norm with `w ≥ 0`. Running the IK does not change the FK output, and object-owned FK limbs still equal FK in `armature_object` mode.

## Real replay: FK vs IK (`benchmark_detector_observation`)

LabServer63 replayed the docs/57 current-policy AnimationSemantics v2 lineage with the same BaseRig, snapshot, calibration and mapping. The replay:
- checked each persisted semantics digest;
- recomputed object-owned FK and required it to equal the stored Worklog 60 files byte-for-byte (all three sequences matched);
- solved IK and compared the two.

Output: `~/animcv-output/framepose_two_bone_ik_replay/`, with `report.json` SHA-256 `b06f3b82a7f895afb0283f8bb1e47a9902190c80bfc30b9eeab3c866a74bf0f6`, plus per-sequence IK and FK-vs-IK files. There were no clamps and no degenerate-bend frames (the smallest real bend was 3.14°). IK availability matches Worklog 60's `lowerarm_l` availability exactly.

| sequence | frames | IK known | UNAVAILABLE (root / mid / end invalid) | FK endpoint error cm: mean / p50 / p95 / max | IK endpoint error cm (max) |
|---|---:|---:|---|---|---:|
| dancing actor0 | 115 | 79 | 36 (4 / 18 / 14) | 1.65 / 0.44 / 6.65 / 18.33 | 4.6e-14 |
| crosscountry actor0 | 162 | 133 | 29 (3 / 12 / 14) | 1.78 / 0.89 / 5.44 / 16.02 | 4.5e-14 |
| hug actor1 | 187 | 183 | 4 (0 / 2 / 2) | 1.72 / 0.98 / 5.11 / 20.95 | 4.5e-14 |
| **pooled** | 464 | 395 | 69 | 1.72 / 0.78 / 5.90 / 20.95 | 4.6e-14 |

Pooled distributions over the 395 IK-known frames:

| distribution | p05 | p50 | p95 | extremes |
|---|---:|---:|---:|---|
| FK error as a fraction of chain length L1+L2 | 0.001 | 0.014 | 0.107 | max 0.381 |
| source reach fraction | 0.700 | 0.936 | 0.993 | min 0.354, max 0.9996 |
| source bend angle | 13.3° | 41.8° | 93.0° | min 3.1°, max 151.3° |
| FK-vs-IK proximal rotation delta | — | 0.93° | 7.87° | max 39.3° |
| FK-vs-IK distal rotation delta | — | 0.20° | 3.22° | max 44.0° |

- The IK clamp rate is 0.
- The bend plane was observed on all 395 frames. On every frame the IK elbow is on the same side as the FK/source elbow, and in the identical plane (minimum cosine between the two perpendicular elbow offsets: 1.000).

**Attribution (negative-leaning finding).** On real data, the gap between FK and IK is explained almost entirely by per-frame source proportion. FK error correlates with `|log(source ratio / target ratio)| · sin(bend)` at **r = 0.951**.

| subset | n | FK error mean (cm) | FK error max (cm) |
|---|---:|---:|---:|
| source upper/lower ratio within 10% of BaseRig's 1.019 | 200 | 0.44 | 2.16 |
| ratio beyond 10% | 195 | 3.04 | 20.95 |
| ratio beyond 30% | 43 | 7.22 | — |
| bend < 30° | 150 | 0.37 | — |
| bend ≥ 60° | 119 | 3.62 | — |

That mismatch, however, is not a stable difference between human and rig proportions. The median source ratio (0.978) is close to BaseRig's (1.019). What varies is the per-frame FramePose segment length: the coefficient of variation is 0.10–0.14 for the upper arm and 0.11–0.18 for the forearm. The source ratio therefore ranges from 0.35 to 4.52 (p05–p95: 0.76–1.31).

The largest FK/IK divergences sit on frames with implausible ratios:
- hug #397: ratio 4.52, FK error 20.95 cm, distal delta 44°
- dancing #205: ratio 0.35, FK error 18.3 cm
- crosscountry #170: ratio 2.11
- dancing #190 and #212: ratios 0.49 and 0.44
- hug #412: ratio 0.58, proximal delta 39°

On those frames the reach-fraction target inherits the same mid-joint error that makes FK and IK disagree. How far the contract endpoint is from the true wrist was **not** measured.

## Owner cases

The Worklog 60 cases are unchanged. `strongly_bent` and `near_straight` were chosen by a fixed rule: the maximum and minimum source bend over IK-known observed-bend frames of all three sequences. No case was replaced.

| case | sequence #frame | IK status | bend | FK error (cm) | proximal / distal delta |
|---|---|---|---:|---:|---|
| known_good | dancing #424 | known, observed | 19.0° | 0.28 | 0.30° / 0.03° |
| dynamic | crosscountry #462 | known, observed | 26.3° | 0.35 | 0.39° / 0.06° |
| turning (review) | crosscountry #211 | known, observed | 40.9° | 4.60 | 5.70° / 1.26° |
| turning (invalid) | crosscountry #202 | **UNAVAILABLE** `source_end_invalid` | — | — | — |
| strongly_bent | crosscountry #173 | known, observed | 151.3° | 13.43 | 38.8° / 29.6° |
| near_straight | crosscountry #525 | known, observed (3.1° > 1°) | 3.1° | 0.025 | 0.03° / 0.01° |
| tracking_loss review | hug #312 | known, observed | 51.4° | 1.62 | 1.95° / 0.55° |
| invalid_endpoint | hug #406 | **UNAVAILABLE** `source_mid_invalid` | — | — | — |

In every case the IK error was ≤ 3.2e-14 cm.

## Blender A/B

The synthetic contracts passed, so a bounded Blender 5.1.2 review was run on macOS (LabServer63 has no Blender). It covers the seven fully-known cases above plus the largest-divergence frame, hug #397, included as a stress case. Each case is an isolated pose with no keyframes or interpolation:
- **A** is `apply_object_owned_frame` (Worklog 60 object rotation + FK).
- **B** has the identical object rotation and pose, with only `upperarm_l`/`lowerarm_l` set to the IK locals.

The renders show:
- the body as bone proxies;
- the FK arm in orange and the IK arm in blue;
- the FK and IK endpoint spheres;
- the requested endpoint as a yellow wire sphere.

Each case has a front view and a side view (depth × up), both aligned to the character heading. Blender refused the two UNAVAILABLE cases (#202, #406) and recorded them without substituting anything.

Blender's evaluated wrists match the Python FK and IK endpoints to ≤ 1.4e-5 cm (float32). The IK lengths in Blender are 27.7712 / 27.2511, and object location and scale were unchanged. The endpoint errors seen in Blender reproduce the replay: FK 0.025–20.95 cm, IK ≤ 1.4e-5 cm.

Output is in `~/animcv-output/framepose_two_bone_ik_blender_review/` (`report.json` SHA-256 `7ead5311551e99e3f95b837adc2aa641651a357b119d94a7384ac1544c9263e3`, 14 PNGs, 7 `.blend` files), with the review PNGs copied locally to `~/animcv-output/61_framepose_two_bone_ik_blender_review/`. This is a pose-geometry review, not a claim about visual motion quality.

## Files and tests

Added:
- `src/retarget/framepose_two_bone_ik.py`
- `examples/e2e_demo/framepose_ik_chains.json`
- `scripts/run_framepose_two_bone_ik_replay.py`
- `scripts/review_framepose_two_bone_ik_blender.py`
- `tests/test_framepose_two_bone_ik.py` (12 tests covering every §22 item)
- this worklog

Nothing existing was modified.

Tests: the new IK tests plus the FramePose FK, object-owner, docs/58 direction-candidate, historical IK/FK/retarget, Blender executor/render, bridge, rig/mapping and keyframe suites gave **148 passed, 1 skipped**. The skip is `test_scale_restored_direction` (needs torch, absent locally). No shared legacy contract changed, so the full regression was not run.

## Verdict

**Against the defined contract: YES.** In the controlled proportion mismatch, target-length-aware two-bone IK hits the dimensionless reach-fraction endpoint exactly while direction-only FK misses. Throughout:
- BaseRig's Blender rest lengths are preserved exactly;
- the source bend side is taken from the current elbow;
- degeneracy is explicit;
- UNAVAILABLE is honoured without hold;
- the Worklog 60 FK baseline and Armature Object yaw ownership are byte-identical.

The real replay shows the same contract behaviour, with 0 clamps and 0 degenerate frames. This closes the initial arm IK **geometry contract**.

**Not established:** that IK is more accurate than FK in the product sense. BaseRig's arm ratio (1.019) is close to the median source ratio (0.978). On real frames FK and IK diverge mainly where FramePose's per-frame segment lengths are inconsistent (r = 0.951 with proportion mismatch × bend), which is a FramePose geometry-quality effect that the reach-fraction target also inherits. Before extending this to legs, ankles or Contact, a ground-truth-anchored control is recommended: compare the FK and IK endpoint directions and reach fractions against 3DPW GT arm geometry, on the same frames and in the same dimensionless terms. Stopped here, before legs and foot locking.
