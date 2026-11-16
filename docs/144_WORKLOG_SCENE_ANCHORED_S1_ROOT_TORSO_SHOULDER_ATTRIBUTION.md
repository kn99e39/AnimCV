# Worklog — Scene-Anchored S1 Root/Torso/Shoulder Attribution, No Justified Reconstruction Candidate (2026-11-16)

Batch 144. **Official hypothesis judgment: NOT YET SUPPORTED.** Diagnostic parent-transform attribution is supported; a source-consistent Camera-Anchored reconstruction improvement is not established. No correction candidate was produced. These are separate conclusions.

All results are **AGENT-RUN**, Windows CPU geometry and Blender 5.1.0. Authentic S1 is `real_animcv_observation` (frozen RTMDet/RTMPose M1 plus learned S0, GT-free replay). Synthetic controls are `oracle_geometry`. Neither regime is treated as 3D ground truth for S1. No training, GPU experiment, Generator inference, Production change, branch operation, commit, push, publication or automation activation occurred.

## 1. Outcome and its practical meaning

The frozen Action's Armature Object **does not translate at all**. Its rotation and `spine_01` rotation nevertheless produce large evaluated pelvis/torso/shoulder motion through the real hierarchy and its nonzero anatomical lever arms. The historical approximately 244 mm shoulder step is reproduced at **Action 176→177 / source 254→255**, inside 163–208. Over the entire 240-frame Action, the largest left-shoulder step is **549.84 mm at 26→27**. This larger value is a different time denominator, not a changed baseline.

The source-derived 2D observations already contain abrupt shoulder landmark relocations. The cached learned 3D pose additionally carries unstable hip/shoulder depth and trunk hypotheses. Those hypotheses generate the Object/pelvis frame and relative spine frame. The frozen adapter reproduces every historical channel on all 240 frames, and independent FK matches fresh Blender evaluation. This locates the observed final motion upstream of Action evaluation; it does **not** identify every erroneous 2D joint, prove a particular learned-model defect, or provide the correct hidden 3D motion.

The existing S1 reconstruction uses **no estimated camera transform**. W143's near-static evidence therefore cannot remove a camera track that was never applied. It also does not supply missing human depth, scale, placement or anatomical pivot ownership. Holding transforms is useful for diagnosis but cannot be offered as a reconstruction improvement.

## 2. Revisions, ownership and scope

| Item | Evidence |
|---|---|
| Starting local revision | `e001952bc42e6711b4582ae87e1ae0f010336a9e` (latest W143 Worklog) |
| Ending local revision | Same revision; all W144 additions remain uncommitted |
| Branch | Existing `Dual_Stage_Generator`, verified before writes |
| Remote | `git fetch origin` before work and before final reporting; `HEAD...origin/Dual_Stage_Generator = 0 0` both times |
| Pre-existing concurrent work | Untracked `third_party/mmpose/`, preserved without changes |
| Historical/Production tracked files | No modifications; `git diff --check` clean |
| Prospective declaration | [w144_declaration.json](../examples/experimental_scene_anchored/w144_declaration.json), SHA-256 `b21091a9e39debc0dd15f14d10a277352a2334415229219a6efe007a482e8c24` |

Read the product vision, v3 perception and coordinate contracts, opt-in reconstruction documentation, relevant W125/W126/W129 and W136–W143 records, W143 declaration and the actual S1-producing source. The Product Vision remains an animation creation/authoring north star; this diagnostic introduces no new system architecture or implemented style-authoring claim.

## 3. Source, rig and historical identities

Authoritative source: Taco fleur, *Double Kettlebell Strict Press (side view)*, Wikimedia Commons, CC BY-SA 4.0. Source record and original URL remain in the W126 input manifest. Review derivatives are local; no source upload was made.

| Artifact | Verified SHA-256 / identity |
|---|---|
| Original S1 video | `de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c` |
| W126 original Animation Base A | `0803ab2bf31210fbb729bc81b76931729fdc3e12cce27b9507a5b8caf1d12402` |
| A assigned Action | `AnimCV_EXP baseline 1284d6ce91af` |
| A Action value digest | `25a09802ffa210df56f0e930295468ef4cdb675e2945cb95491c080b5101c8a9` |
| Cached M1/S0 observations/predictions | `7988bf4da320d4def763c81b9d9210905f70b7854e54d71974482392c2510f51` |
| W126V historical accepted copy | `7e2ab16511f4cbe111171df9af0bc2e1f7d8c09a57bd5a4feff284d34e908f40` |
| W143 S1 estimate | `7851b2f2bf23b491c8c9fb0e2bda97124728697173cbf20378627129969c10f0` |
| W143 evidence manifest | `d6b1a1bb529e7fb2eb3add0059ed7f76b189d06c357858557aa17c9908c6c3f8` |
| BaseRig rest snapshot digest | `8519d243fc823a54b442c8f4487f2655a4c3e1b56f0cf7b07c73bfffb319e10c` |

`result/w144/preflight_verified.json` contains **127 identity checks** and an inventory of **3,226 original files**. Every inventory hash and every fixed expected hash was rechecked at the end: **0 differences**. Scope includes the entire original W126 S1 run, source video, accepted/reverted/pin/lineage evidence, selected W129/W136–W139 `.blend` outputs against their manifests, W125/W129/W136–W143 evidence manifests, W143 S1 output/declaration and frozen producing code. This is not a fresh full hash of every historical Generator checkpoint or every historical dataset. No historical file was written.

## 4. Exact time correspondence

The mapping is verified against the input manifest, Action sidecar, decoded source image bytes and observation timestamps, not filenames:

- Source frames **79–318 inclusive** → Action **1–240**; `Action = source − 78`.
- Source nominal time `source/30`; Action playback time `(Action−1)/30`; duration **8.000 s**.
- All 240 decoded BGR frame hashes equal the frozen source hash array slice `[79:319]`.
- Cached observation `frame_index` and `timestamp` equal the original 1,631-frame index and index/30 arrays.
- Presentation timestamps equal the frozen container record, including its millisecond quantization; fresh maximum difference is **0 ms**. Historical deviation from ideal cadence was at most 0.667 ms.
- W126V's protected source pin 210–218 maps to Action 132–140; its historically changed frames were 133–140. No W144 modification intersects it because no Action is modified.

Subject identity remains the historical **UNKNOWN / tracked continuity unverified** contract. The reviewed footage shows one principal performer; no temporal person switch was confirmed. This does not turn anatomical left/right keypoint identities into verified labels.

## 5. Actual implemented source-to-Action path

```text
Original immutable S1 RGB / source timestamps
  -> frozen M1: RTMDet tiny person selection + RTMPose tiny 2D
  -> cached canonical_v1 landmarks (16 x xy/score/visible)
  -> canonical 17-joint normalized 2D + validity
  -> deterministic valid-joint crop geometry (17 x 4)
  -> frozen S0 per-frame learned geometry-only 3D hypothesis (17 x 3)
  -> fixed_slot: subtract predicted pelvis; torso_frames(j) -> Rt, Rp
  -> W105/W121 Object/spine/legs + left-arm IK/local rotations
  -> W116 state channels -> W124 opt-in Action build
  -> CONSTANT quaternion keys -> fresh evaluated Blender matrices/FK
```

The actual path is `experimental_video_action.source_to_rig` → `w121_independent_transfer.fixed_slot` → `w105_rig_utility.rig_measure` and the existing arm exporter, then the frozen Action build. **No temporal lifter, camera estimator, global root position estimator or contact solver is in this path.** Existing Camera Geometry is a separate experimental evidence stream.

## 6. Representations and transforms

| Stage | Coordinates / units / parent | Provenance, confidence and missing information |
|---|---|---|
| Raw observed joints | Image X right / Y down, pixels | Detector estimates, score and visible flag; pelvis/neck/spine are derived midpoints; head is nose proxy |
| Canonical 2D | x/width, y/height, score, separate validity | Virtual thorax and neck reuse the shoulder midpoint; invalid coordinates are zero-filled model inputs, not stationary observations |
| Geometry tokens | Person crop `[-1,1]`, confidence, validity | Recomputed bytes equal cached geometry; detector bbox, silhouette, explicit contact and occlusion labels were not cached |
| S0 3D | Canonical camera X right / Y forward-depth / Z up, learned body units interpreted as metres | Hypothesis, not calibrated scene geometry; no per-joint calibrated 3D confidence. Adapter subtracts joint 0; no absolute pelvis depth |
| Pelvis/thorax frames | Active rotations, columns anatomical axes in canonical camera | `Rp` uses projected hip bilateral axis and shared trunk up; `Rt` uses shoulders and shared trunk up; no gravity/scene observation |
| Retarget local values | Project quaternion xyzw, rest-basis-conjugated local bone rotations | Fixed alignment A, existing IK and rig proportions. Only spine_01, thighs/calves, left upper/lower arm driven |
| Armature Object | Blender quaternion wxyz; imported rest rotation times `A Rp A^-1` | Object named `root`; location unkeyed (0,0,0), uniform scale approximately 0.01; no parent |
| Spine | Local rest-basis conjugate of `A (Rp^T Rt) A^-1` | Rotation about spine_01 rest head, not a new scene-root placement |
| Blender evaluation | Rig rest lengths in armature centimetre-like units, object scale produces world metres | `WorldBone = Object @ PoseBone`; top-level `PoseBone = RestBone @ Basis`; child uses `ParentPose @ inverse(ParentRest) @ RestBone @ Basis` |

Top-level bones are `center_of_mass`, `interaction`, `spine_01`, `thigh_l`, `thigh_r`. There is **no anatomical pelvis/root bone** driving the hierarchy. `center_of_mass` is a sibling marker, not an inferred pelvis trajectory. W144 reports Object translation/rotation separately and explicitly uses the **midpoint of evaluated thigh heads** as a pelvis proxy. Torso = spine_05 head; shoulders = upper-arm heads; elbows/wrists/feet are their actual bone heads. No missing intermediate was invented.

The Object rotates about its origin while pelvis/spine heads are offset. Thus zero Object travel does not imply zero anatomical pelvis/shoulder movement. This is an implemented placement/pivot limitation; an alternative pivot would impose a different trajectory and was not silently treated as a verified bug fix.

## 7. Frozen measurement and interval policy

Before official trajectory extraction, the declaration fixed all 239 edges, Euclidean step and second differences, speed = step×30, acceleration = second difference×900, SO(3) increments, valid confidence 0.3, 2D jump flag >20 px, FK tolerance 1e-5 m and subframes +0.25/+0.5/+0.75/+0.999. A high velocity or flag is not an error verdict.

Fixed review intervals: 1–12, 163–208, 168–175, 196–203, 229–240. Baseline/source-only rules selected:

- Largest five left-shoulder destinations: **27, 23, 26, 22, 177**, each ±3 frames.
- Quiet baseline window: **135–146**.
- Largest available confident wrist image path: **11–22** (both wrist paths, missing edges excluded).
- Most missing input joints: **206–217**.

These selections precede any candidate; none exists. The whole Action, not only these windows, is measured and included in the native-rate file.

## 8. Source-visible 2D evidence

| Observation | Valid frames | Median step px | Maximum step px |
|---|---:|---:|---:|
| Pelvis midpoint | 240 | 0.602 | 11.387 |
| Thorax/neck midpoint | 240 | 0.673 | 18.664 |
| Left shoulder | 240 | 0.748 | 30.252 |
| Right shoulder | 240 | 0.856 | 37.150 |
| Head/nose | 240 | 1.196 | 45.085 |
| Left/right hip | 240 / 240 | 0.660 / 0.768 | 5.333 / 23.704 |
| Left/right knee | 240 / 240 | 0.639 / 0.689 | 3.833 / 18.423 |
| Left/right ankle | 240 / 240 | 0.545 / 0.519 | 2.346 / 6.411 |
| Left/right elbow | 240 / 202 | 1.106 / 1.091 | 14.662 / 39.477 |
| Left/right wrist | 240 / 214 | 1.532 / 1.473 | 22.031 / 31.136 |

Every invalid frame is enumerated in `analysis.json`; invalid-adjacent edges are excluded rather than scored as zero. Left shoulder flags occur at destinations 22/23; right shoulder at 21/26/27. The torso midpoint can remain under the 20 px flag while one constituent shoulder moves much farther. A stable keypoint envelope is not proof of anatomical stability.

**Supplementary source-texture control:** after the primary baseline results, a separately recorded protocol fixed 21px pyramidal LK, 3 levels, 30 iterations, epsilon .01, minimum eigenvalue 1e-4 and forward/back error ≤1px. This is disclosed post-baseline evidence, **not** a prospective correction gate or anatomical GT.

| Destination Action | Landmark | Local texture flow px | Detector-vs-texture residual px |
|---|---|---:|---:|
| 22 | Left shoulder | 1.22 | 29.29 |
| 23 | Left shoulder | 0.48 | 29.34 |
| 26 | Right shoulder | 1.09 | 37.54 |
| 27 | Right shoulder | 0.87 | 36.31 |
| 177 | Pelvis midpoint | 0.23 | 6.77 |

The original frames show broadly maintained side-on torso facing during 24/26/27, while detected shoulder locations and reconstructed facing alternate. This supports existing observation instability; texture tracking is not a proof of which occluded anatomical shoulder is correct. At 11–22 there is genuine clean/standing motion; later frames include real overhead press articulation. No claim that every motion peak is jitter, and no confirmed person or left/right identity-switch repair, is made.

## 9. Camera reference, separate from human inference

The frozen W143 S1 result is authenticated. Adjacent-edge median background flow is **0.05437 px**, with **45 median background keypoints/frame**. The all-span median is 0.10824 px; it must not be substituted for the adjacent-flow statistic.

W143 reports 238/240 nominal orientation frames, 0 registered SfM frames and unresolved translation throughout. No trusted intrinsics, metric camera centre or absolute physical scene origin exists. Two frames stay unavailable; they are not filled with identity rotations.

On the 235 available adjacent pairs, W143 nominal camera orientation increments have median **0.00297°**, maximum **0.01134°**. Explicit optical→canonical conversion and orientation-only composition with the inferred pelvis frame leave a root-rotation maximum of **76.419°**, versus the approximately 76.421° baseline peak. This is sensitivity of an **uncalibrated orientation hypothesis**, not independent calibrated accuracy or a rig correction. No World-position or reprojection gate is derived from it.

The constant-camera reference is a stated convention/hypothesis. In the actual Action path there is no camera track to hold. W140–W143's positive moving-scene relative 6DoF and high-coverage near-static orientation results remain preserved; this Batch's negative human-correction result does not invalidate them.

## 10. Measured Root, pelvis, torso and shoulder motion

All positional values below are **evaluated rig World motion**, not error against physical GT.

| Trajectory, 239 edges | Median mm/frame | p95 mm/frame | Maximum mm/frame |
|---|---:|---:|---:|
| Object translation | 0 | 0 | 0 |
| Pelvis proxy | 14.55 | 111.89 | 246.76 |
| Torso/spine_05 | 21.81 | 166.85 | 409.83 |
| Left shoulder | 22.58 | 173.53 | 549.84 |
| Right shoulder | 22.65 | 184.74 | 483.01 |
| Neck_02 | 24.60 | 185.77 | 469.18 |

Object rotation increment: median **2.838°**, p95 **26.956°**, maximum **76.421° at 25→26**. Spine_01 local increment: median **1.689°**, p95 **13.308°**, maximum **66.178° at 170→171**. Unkeyed clavicle/other spine locals do not introduce new animation curves. Complete distributions, accelerations, XYZ ranges, elbow/wrist/foot trajectories, all local rotations and synchronized per-edge data are in `analysis.json`, `edge_trajectories.json`, `trajectories.npz` and `blender_trace.json`.

The learned bilateral hip vector length ranges **13.9–130.4 mm**; shoulder vector **142.2–430.1 mm**; trunk length **246.7–317.8 mm**. These are learned hypotheses, not actual anatomy. They show why frame construction can be sensitive. Both hip/shoulder image X and inferred forward-depth components vary; depth instability cannot be uniquely separated from detector error by cached outputs alone. No S0 forward run or new model was used.

## 11. Discontinuities and actual source correspondence

| Action edge | Source edge | Left shoulder mm | Pelvis proxy mm | Object / spine local increment deg |
|---|---|---:|---:|---:|
| 21→22 | 99→100 | 315.32 | 133.97 | 10.965 / 30.867 |
| 22→23 | 100→101 | 515.53 | 240.48 | 52.486 / 4.586 |
| 25→26 | 103→104 | 473.27 | 191.17 | 76.421 / 4.140 |
| 26→27 | 104→105 | 549.84 | 246.76 | 76.324 / 4.469 |
| 170→171 | 248→249 | 189.72 | — | 64.430 / 66.178 |
| 171→172 | 249→250 | 40.64 | 35.31 | 10.625 / 5.882 |
| 176→177 | 254→255 | **244.01** | 152.74 | 19.193 / 16.440 |
| 199→200 | 277→278 | 22.75 | 15.01 | 1.370 / 1.062 |

At the historical 244 mm event, observed pelvis midpoint moves 6.98px and thorax midpoint 1.18px; the predicted thorax changes 58.0mm. At the 549.84mm event, observed pelvis moves 1.23px, thorax 18.66px, and predicted thorax changes 95.4mm. No pixel-to-metre error conversion is claimed. Source images and texture control support mislocated detector points in several of these events, while the reconstructed pose itself contains unsupported 3D alternatives.

## 12. Controlled causal attribution

For **every** edge, a fresh Blender process detaches the Action **in memory only**, restores the complete captured local state and changes the declared factor. Every dependent transform is reevaluated through the real hierarchy. Both directions are measured: hold a current factor at its previous value (removal), and inject its current value into the previous state (reproduction). No diagnostic Action is saved.

| Edge | Original shoulder mm | Hold Object rotation mm | Hold spine_01 mm | Hold both mm |
|---|---:|---:|---:|---:|
| 21→22 | 315.32 | 102.42 | 230.95 | <0.001 |
| 22→23 | 515.53 | 15.40 | 528.86 | <0.001 |
| 25→26 | 473.27 | 13.90 | 459.71 | <0.001 |
| 26→27 | 549.84 | 15.01 | 548.32 | <0.001 |
| 171→172 | 40.64 | 19.75 | 40.69 | <0.001 |
| 176→177 | 244.01 | 55.03 | 289.27 | <0.001 |
| 199→200 | 22.75 | 3.57 | 19.58 | <0.001 |

Across all edges, holding Object+spine leaves at most **0.487 µm** of left-shoulder motion; injecting that pair reproduces the original steps to numerical precision. Changing only Object translation, clavicles, arm locals or thigh locals cannot explain shoulder-head displacement. This is a necessary/sufficient **parent-state set in this fixed rig**, not a uniquely necessary scalar cause and not proof the 3D parents are correct.

Interactions matter. At 176→177, the conditional spine displacement after changing Object is **55.03mm**; the conditional Object displacement after changing spine is **280.74mm**. The interaction-vector norm is **16.91mm**. At 170→171, holding Object alone gives 210.12mm, holding spine alone 399.64mm, while original is 189.72mm; interaction norm **223.98mm**. No unique causal percentages are assigned.

Additional upstream diagnostic replacement of one **3D joint group**, rerun through the actual frozen `fixed_slot`, changes all dependent adapter outputs and uses independently validated analytic FK. At 176→177, previous thorax joint reduces the shoulder step to 46.17mm, previous hip pair gives 256.45mm, and previous shoulder pair gives 246.28mm. At 26→27, those values are 286.86 / 602.88 / 427.99mm. Thus no single joint-group freeze removes all mechanisms. These are analytic FK diagnostics, not fresh Blender candidate readbacks or accepted repairs.

## 13. FK, interpolation, axis and evaluation controls

- Recomputing frozen `fixed_slot` from every cached prediction gives **0 mismatching frames/channels** against the stored cases.
- New Blender readback equals the historical readback **exactly** at all 240 integer frames; Action value digest is identical.
- Independent rest-relative FK: maximum head residual **0.404 µm**, tail **0.447 µm**, below the frozen **10 µm** tolerance.
- **956** subframe samples (+.25/.5/.75/.999) equal their left integer frame exactly. CONSTANT is genuinely stepped, with no measured intermediate overshoot. It does not explain where the integer-frame orientation changes originated.
- Bone-length range is below **0.45 µm**. Object scale is preserved. No constraints, NLA tracks or drivers are involved.
- Existing retarget alignment conjugation, quaternion layout/sign, parent-child application and camera inverse semantics pass focused tests. No double application or Blender matrix composition defect is demonstrated against the intended contract.

## 14. Synthetic mechanism controls (`oracle_geometry`)

Known transforms test constant camera/stable Root, moving camera/stationary human via inverse camera transforms, fixed camera/real Root movement, stable Root/articulated torso, an upstream orientation discontinuity and an extra parent/local transform defect.

Actual BaseRig Blender controls recover a 0.100m injected Object translation to **0.100000024m** shoulder displacement; a known 20° Object rotation produces **0.363593m** shoulder displacement with **zero Object translation**; the known torso perturbation produces **0.075920m** shoulder displacement. All independent FK head errors are ≤**0.202 µm**. This verifies layer accounting and that a large shoulder step need not be Root travel.

**W144 analysis wording correction:** `analysis.json` initially described the coupled parent control as a duplicated application. The actual instrument injects world-Object Z and local-spine Y rotations, not the same rotation twice. `source_controls.json` explicitly corrects this label: against the known parent-only oracle the extra local perturbation produces **75.920mm** shoulder error. It is a known parent/articulation defect control, not evidence of that defect in S1. Upstream-pose and Root-orientation injections can have identical endpoint states; transform endpoints alone cannot distinguish their provenance. Synthetic correctness does not establish real-source reconstruction accuracy.

## 15. Supported and rejected explanations

**Supported:** unstable observed landmark locations; unstable source-derived 3D parent hypotheses; Object/spine rotations causing anatomical World motion through measured FK and offset pivots; whole-body placement/contact absent; CONSTANT causing held continuous-time poses.

**Not supported by current evidence:** camera-motion misinterpretation as the cause of this Action's jumps; actual Root translation discontinuity; a frame-map error; Action corruption; retarget axis inversion; parent double application; Blender constraints, NLA or interpolation overshoot. These alternatives were checked against the actual producing path, not rejected just because the rig looked noisy.

**Unresolved:** exact anatomical positions of self-occluded joints, relative contributions of detector/crop/model ambiguity to every S0 jump, metric human depth/scale, physically correct root placement, gravity-aligned world frame and ground contact. LK and detector confidence do not calibrate 3D trust.

## 16. Conditional candidate decision and real-rig A/B status

**Skip candidate production under the frozen rule.** No exact identity/frame/axis/FK implementation defect is found against the intended contract. Source texture does not independently determine occluded anatomical shoulder positions or the correct 3D pose. A constant-camera assumption cannot supply metric depth or placement. Changing the rotation pivot, freezing pelvis, fitting root curves or adding foot locks would impose a new unsupported trajectory.

Consequently candidate build, A/B improvement gates and candidate reversibility are **NOT APPLICABLE**, not PASS. W126V remains an authenticated historical reference, not a correction target. Diagnostic holds/injections are not presented as better animation. No smoothing, strength sweep, post-result gate relaxation or search for a pleasing pose occurred.

## 17. Review artifacts and limits of qualitative inspection

Review directory: `user_eval/W144_scene_anchored/` (local, ignored by Git).

- **`S1_source_baseline_30fps.mp4`:** original source plus actual Blender A mesh renders from two fixed diagnostic World views. Full 240 frames, 30.000 fps, 8.000 seconds, every frame decoded; source/action IDs, source time, Object/spine increments and shoulder/pelvis steps overlaid. No candidate panel exists. These orthographic views are not calibrated source-camera projections.
- **`sheets_aspect_preserved/`:** authoritative comparison sheets for the frozen representative windows, including source-visible motion, quiet, occluded, boundary and maximum-step cases. Use these; the initial root-level sheets stretched the crop panels. Initial files are retained with a layout-correction record; the movie's original-source panel always retained its aspect ratio.
- **`trajectory_attribution.png`:** full 240-frame synchronized 2D, actual rig, rotation, parent-intervention and supplemental texture evidence.

Agent still-frame review observed a sharp rig-facing alternation at 24/26/27 absent from the source's broad torso facing; quiet 135–146 was broadly stable but retained posture/arm mismatch; 11–22 contains genuine clean motion; 174–180 shows inherited lean/facing changes; 206–217 contains real arm movement with self-occlusion and a permanently rest-pose rig right arm. No scalar source-fidelity score or hidden 3D accuracy is manufactured.

**Native-rate perceptual playback: BLOCKED, not reviewed.** `node_repl` failed initialization with Windows `0xC0000142`; reset/retry failed, and the separate `cua_repl` surface inventory failed the same way before UI access. Shell-based CPU/Blender execution succeeded through approved elevated execution. Screenshots/contact sheets were inspected; the native-rate file was machine-verified, but it was **not perceived at native speed**. No human acceptance is claimed. This is a review-stage limitation, not a prerequisite preventing the quantitative diagnosis.

## 18. Tests and execution disclosures

**145 PASS in 9.92s**, Python `-X utf8`: W144 (21), W140–W143 camera tests (92), existing torso/pelvis ownership/Object ownership/FK tests (32).

The first identical regression invocation omitted UTF-8 mode and gave 143 pass / 2 historical source-scanning `cp949` decode errors. Re-running the same suite with `-X utf8` passed without changing historical tests or Production code. No unexecuted test is called PASS. Full Production regression, new neural inference and live perception were not run.

Other transparent harness issues: initial preflight compared 240 selected decoded hashes with all 1,631 stored hashes; the failed `preflight.json` remains, and the corrected `[79:319]` check passed **before declaration**. A first test-launch wrapper had an unresolved relative executable path; direct current-interpreter invocation ran the suite. Finalizer initially assumed UTF-16 for a UTF-8 PowerShell log; corrected decoding before sealing. None changed experiment thresholds, observations, official trajectory extraction or verdict policy.

## 19. Resources and reproduction

New scratch/review evidence before report: **85.44 MB**, versus 8 GiB ceiling. Measured trace execution **5.48s**, 480 Blender renders **6.52s**, analysis **2.27s**; these sum only the named stages, not all authoring/preflight/test time. No stage approached the 2h primary-processing ceiling. GPU experiments/training/Generator inference/correction candidates: **0**.

Run from a checkout containing these additions with the existing authenticated historical directories available. Every official output refuses overwrite; the recorded commands are not instructions to erase existing evidence. Use a fresh output location/check-out copy when independently reproducing; this harness binds `result/w144` relative to its checkout and historical inputs by their exact paths.

```powershell
$env:PYTHONPATH='src'
.venv_build_full/Scripts/python.exe -X utf8 scripts/w144_harness.py preflight
.venv_build_full/Scripts/python.exe -X utf8 scripts/w144_harness.py declare
.venv_build_full/Scripts/python.exe -X utf8 scripts/w144_harness.py trace
.venv_build_full/Scripts/python.exe -X utf8 scripts/w144_analyze.py
.venv_build_full/Scripts/python.exe -X utf8 scripts/w144_source_controls.py
# render_jobs.json and render_execution.json contain exact frozen render jobs/argv
.venv_build_full/Scripts/python.exe -X utf8 scripts/w144_review_artifacts.py
# tests_utf8.txt and this Worklog section 18 identify the exact focused suite
.venv_build_full/Scripts/python.exe -X utf8 scripts/w144_finalize.py
```

Machine-readable evidence is in `result/w144/`; no wholesale JSON is pasted into this narrative. Key SHA-256 identities:

| File | SHA-256 |
|---|---|
| `preflight_verified.json` | `702af1e75ee8043338788d6160a7aa9620726590c4377109375aa8c5c123f0a4` |
| `blender_trace.json` | `3826e7dce846e96370ebc6b4d871b812b841cb48c75c9c03bd79358a1523f9e7` |
| `analysis.json` | `c466a5c754ee489d63f9b840ad7c931e59e8310391cc1376b2d4c8418e547be9` |
| `source_controls.json` | `b14ae424ae307900f4783f2a70e508f11b0b451ebe7c8b67f1dff6e1659cfd20` |
| `final_verdict.json` | `ef3d7fd43cf20cf479be17494dcf8df2cd1ce902b3e19fc48a326cdc66d27dc4` |

## 20. Exact additions and official architectural judgment

Only these ten repository files are added:

1. `src/app/experimental_scene_anchored_trace.py`
2. `scripts/w144_harness.py`
3. `scripts/w144_blender_trace.py`
4. `scripts/w144_analyze.py`
5. `scripts/w144_source_controls.py`
6. `scripts/w144_review_artifacts.py`
7. `scripts/w144_finalize.py`
8. `tests/test_w144_scene_anchored_trace.py`
9. `examples/experimental_scene_anchored/w144_declaration.json`
10. This W144 Worklog.

Ignored machine/review artifacts are confined to `result/w144/` and `user_eval/W144_scene_anchored/`. No old Worklog, normative schema, Production reconstruction/exporter, Generator, original Action or automation file is changed. The existing untracked MMPose checkout remains outside the W144 additions.

**NOT YET SUPPORTED:** the responsible evaluated parent-state mechanism is located and controlled, and upstream observation/3D-hypothesis instability is evidenced, but the available source and Camera Geometry do not establish a useful source-consistent 3D reconstruction correction. Diagnostic attribution is stronger than correction evidence. Camera capabilities remain intact; source fidelity, native-rate perceptual review and unobservable human geometry remain qualified as above. This report makes no subsequent-Batch, architecture-change or Production-adoption recommendation.
