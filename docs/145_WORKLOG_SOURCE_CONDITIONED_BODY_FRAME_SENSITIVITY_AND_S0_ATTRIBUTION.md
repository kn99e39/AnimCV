# Worklog — Source-Conditioned Body Frame Sensitivity and Frozen S0 Attribution, Conditional Repair Not Eligible (2026-11-16)

Batch 145. **PARTIALLY SUPPORTED — conditioning and causal-attribution evidence only.** No source-consistent corrected S1 Action or generally reliable temporal orientation estimator is established. The conditional real-rig repair phase was not eligible under the frozen synthetic gates.

Evidence is **AGENT-RUN**, not independent reviewer replication or human visual acceptance. S1 measurements are `real_animcv_observation`: frozen M1 detector observations and S0 learned hypotheses. Synthetic orientation controls are `oracle_geometry`, exact pose tokens with simulated image evidence, not rendered/perception accuracy. No new training, GPU experiment, dataset download, Generator change or Production integration occurred.

## 1. Principal findings

1. **The frame decoder has measurable scale sensitivity, but near-parallel axes are not the main S1 problem.** Projected hip-axis length falls to 11.49mm. Analytic Rp sensitivity reaches 7.056°/mm. However hip/trunk sine stays ≥0.828 and shoulder/trunk sine ≥0.950: the vectors are not near the historical 20° degeneracy boundary. All baseline frames and all 25,920 tested 3D perturbations remain valid.
2. **The actual learned input-to-orientation path is sensitive.** Frozen S0 CPU reproduction agrees with cached GPU output within 1.267µm and repeats exactly. Across 7,680 matched 1px interventions, Rp changes by up to **20.451°**. This is measured model response followed by unchanged frame decoding, not inferred solely from correlation.
3. **The causal observation group differs by event.** At Action 25→26, previous shoulder observations reduce the inferred Rp step from 76.421° to 9.257°, whereas previous hips leave 68.749°. At 22→23, previous hips reduce 52.486° to 8.886°, whereas previous shoulders leave 51.470°. These are controlled input replacements, not anatomical GT or accepted corrections.
4. **An assumption in frame semantics is independently exposed.** Both Rp and Rt use the same trunk-derived up vector. In a synthetic body with a truly fixed pelvis and a bending torso, the decoder assigns up to 29.995° of false pelvis tilt. It correctly separates pure torso twist. This identifies a representational coupling, not a Blender or quaternion implementation defect.
5. **The declared source/conditioning qualifier is insufficient for repair.** It preserves genuine synthetic turns and abstains on weak geometry, but accepts the interior of a persistent wrong branch. Only 2/11 injected false-pose frames are rejected. It leaves 68/240 S1 frames without both orientations and does not provide replacement states. No gaps were filled by holds, smoothing or interpolation.

The result materially explains how observed/learned geometry becomes inappropriate full-body motion. It does not establish which hidden 3D pelvis orientation should replace each S1 frame. No unique causal percentages, metric source reprojection scores or animation-quality gains are claimed.

## 2. Branch, ownership and local commits

| State | Revision / evidence |
|---|---|
| Session start | `e001952bc42e6711b4582ae87e1ae0f010336a9e`, existing `Dual_Stage_Generator` |
| Initial remote comparison | After fetch, local/remote difference `0 0` |
| W144 preservation | `53361a9c6de737910817a97d8ffbc159c8203723`: exactly the ten W144 additions, unchanged |
| W145 prospective core/declarations | `b07e83e41ab389caaee337fe953dd1b50e6ac261` |
| W145 official execution instruments | `fb70318`: committed before official conditioning/model/synthetic outcomes |
| W145 assessment/review reporter | `38899101cc486a224844e194d5281c1be62d1d5d` |
| Source revision at report writing | `3889910`; this Worklog and the cumulative audit are recorded in the closing documentation commit; exact final local revision is recorded in `result/w145/completion.json` |

The user's instruction to commit completed work was applied at each unit above. Nothing was pushed. No branch/worktree was created, switched, merged, rebased, reset, renamed or deleted. Pre-existing untracked `third_party/mmpose/` was excluded from all commits. Automation remains paused; `docs/next_batch.md` and `scripts/batch_automation.py` are unchanged.

## 3. W144 preservation and source authentication

Before new W145 files, all **539 W144 manifest entries** matched SHA-256, and all **3,226 original preservation-inventory entries** matched. The W144 evidence manifest is `94288096c4d7b99dd9d17bfb016bae22bcd73c34bd1e2189eab9fc8e60b4fffd`. Its ten source files were committed without rewriting the Worklog, declaration, code or result. W144 remains **NOT YET SUPPORTED for correction**, with supported downstream attribution.

Authenticated identities:

| Asset | SHA-256 |
|---|---|
| Original S1 source | `de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c` |
| Original Action A | `0803ab2bf31210fbb729bc81b76931729fdc3e12cce27b9507a5b8caf1d12402` |
| M1/S0 observation/prediction cache | `7988bf4da320d4def763c81b9d9210905f70b7854e54d71974482392c2510f51` |
| Frozen S0 checkpoint | `6c12fea9f84ea0493fada8b9acc96cd5c70fdd5d9114603e954f9412f4247b44` |
| W126V historical accepted file | `7e2ab16511f4cbe111171df9af0bc2e1f7d8c09a57bd5a4feff284d34e908f40` |

Source frames 79–318 map exactly to Action 1–240, `source = Action+78`, 30fps. W144's authenticated input manifest, decoded-frame identities, sidecar and timestamps remain intact. W145 uses these original arrays without resampling. A new Blender process again reproduces every original integer-frame head/tail exactly. Pin/lineage, W129/W136–W139 comparisons and W140–W143 camera evidence are preserved, not treated as new quality GT.

## 4. Prospective policy and disclosures

[Declaration v2](../examples/experimental_body_frame/w145_declaration_v2.json) SHA-256: `a56771110ae15cc86c8cecf95f9a2e2a28835a80ea41e36f7e7f07c516618293`. It supersedes but retains the [initial declaration](../examples/experimental_body_frame/w145_declaration.json), `fdf06c4c…`, and changes no algorithm, threshold, synthetic gate or perturbation magnitude.

The initial unit run had 15 PASS / 1 FAIL: a genuine 30° turn from a near-frontal view moved projected hip points less than the declared 2px support cutoff and correctly abstained under that rule, contradicting the unit fixture's unconditional acceptance expectation. Before official sequence/model outcomes, the clear-turn fixture was made oblique, and a separate test explicitly preserves the ambiguous frontal-turn abstention. The issue, old declaration hash and unchanged thresholds are disclosed in v2. This is not post-outcome tuning of a quality rule.

Frozen policies:

- Full 240 frames; review windows 1–12, 11–22, 21–27, 135–146, 168–180, 196–203, 206–217, 229–240.
- 3D: each Cartesian coordinate of joints 0/1/4/8/11/14, both signs, **1/2/5mm**, all frames: **25,920** perturbations.
- Analytic derivative versus central difference **epsilon 1e-6m**.
- 2D: direct hips/shoulders 1/4/11/14, x/y, both signs, **1px**, all frames. Match fixed cached crop versus recomputed crop; rederive virtual points. **3,840 per crop policy, 7,680 total**.
- Selected model actions: 16, 22, 23, 26, 27, 140, 171, 172, 177, 200, 211, 240. Three interventions each: previous hip xy, previous shoulder xy, forward/back-valid source-texture endpoints.
- Observation validity inherited from frozen 0.3 M1 score policy. Texture: frozen W144 LK, FB≤1px; disagreement threshold 3px. Texture is not anatomical GT.
- Qualification: 2mm bounded-perturbation sensitivity indicator ≤5°; unavailable/weak observations explicit. Raw rotation innovation >10° with source motion <2px is **unconfirmed**, not proven false. No angular-speed cap or previous-accepted-pose propagation.
- Synthetic: ten 90-frame cases; clear coverage ≥95%, stationary error ≤0.1°, turn p95 error ≤1°, false-flip rejection ≥90%, degenerate abstention ≥95%, recovery ≤2 frames, pelvis/torso separation ≤1°.
- Budget: 2h primary processing, 8GiB new evidence, no GPU/training/download, at most one conditional real candidate.

The official execution file hashes were separately frozen and committed before their first outcomes in `w145_execution_manifest.json`. No frozen instrument changed after that run. The post-result reporter implements aggregation/visual presentation; it does not change the declared experimental comparisons.

## 5. Actual M1 → S0 → Body Frame contract

The normative Product Vision, v3 perception/coordinate contracts and standing opt-in rig contract were read. The actual S1 path remains:

```text
RGB -> frozen RTMDet/RTMPose M1 xy/score/visible
    -> canonical 17 normalized 2D + validity
    -> deterministic person crop geometry (17x4)
    -> frozen S0 geometry-only network, neutral sign inputs
    -> per-frame 17x3 learned pose
    -> Rp/Rt -> Object + spine_01 -> real BaseRig FK
```

S0 is width 256, 8 heads, 2 fusion layers, seven neutral sign fields, **no RGB visual tokens**. It is not a temporal lifter. The learned output is a body-relative hypothesis in canonical X-right/Y-depth/Z-up coordinates, not calibrated World position. Derived pelvis is the hip midpoint; thorax/neck reuse the shoulder midpoint; spine is their midpoint. A diagnostic direct-joint intervention rederives those points with minimum confidence/conjoined validity.

The frozen frame formula is unchanged:

```text
u = normalize(thorax - pelvis)
v_p = right_hip - left_hip;       v_t = right_shoulder - left_shoulder
x = normalize((I-u*u^T) v);      y = cross(u,x)
Rp/Rt = [x, y, u]
Object delta = A Rp A^-1
spine_01 local = rest-basis conjugate of A (Rp^T Rt) A^-1
```

Trunk must exceed 1mm; side/trunk angle must meet the historical 20° floor. The denominator's **absolute projected side length** also controls sensitivity but has no equivalent historical absolute-length gate. Both frames share `u`: this is the tested ownership assumption. The rig has no anatomical pelvis bone; root is the Armature Object. Object translation remains zero. Anatomically offset pelvis/shoulder points move under rotation as established by W144.

## 6. Conditioning and analytic sensitivity

For a perturbation, `du=(I−uu^T) dt/|t|`; the projected-side differential is `(I−uu^T)dv−(du*u^T+u*du^T)v`; normalization adds division by projected-side length. The angular tangent comes from the skew part of `R^T dR`. The reported spectral norm is degrees per millimetre of the stacked 18-coordinate perturbation norm; it is not calibrated model error probability. Individual-coordinate finite perturbations are reported separately.

| Quantity, 240 frames | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Projected hip axis, mm | 11.49 | 87.46 | 130.40 |
| Projected shoulder axis, mm | 142.24 | 320.12 | 430.01 |
| Hip/trunk angle sine | 0.828 | 0.997 | ~1 |
| Shoulder/trunk angle sine | 0.950 | 0.997 | ~1 |
| Rp operator sensitivity, deg/mm | 0.621 | 0.927 | 7.056 |
| Rt operator sensitivity, deg/mm | 0.263 | 0.280 | 0.570 |

The directed hip/trunk angle spans 75.42–124.16°; shoulder/trunk angle spans 71.84–99.38°. These ranges confirm that small absolute axis length, rather than a near-parallel configuration, is the important conditioning factor here.

All frames are right-handed, orthogonal to numerical precision; determinants and orthogonality residuals are stored per frame. Analytic/central-difference relative error: median **2.04e-11**, maximum **2.56e-9** across 480 Jacobians.

| Single-coordinate perturbation | Median of per-frame worst Rp change | Global maximum Rp change | Global maximum Rt change |
|---|---:|---:|---:|
| 1mm | 0.617° | 4.385° | 0.376° |
| 2mm | 1.235° | 9.073° | 0.752° |
| 5mm | 3.100° | 24.491° | 1.889° |

No perturbed frame crossed the original validity threshold. Rp's 2mm operator indicator exceeds 5° on **16 frames**. This supports meaningful short-axis amplification, but the tested 1–5mm perturbations do not alone reproduce every observed 52–76° event. Large changes in learned bilateral direction and trunk geometry are also necessary. Near-parallel-axis degeneracy is not a sufficient explanation of these S1 failures.

At Action 25→26 the hip depth difference changes +70.56mm→−1.03mm; at 26→27 it returns to +80.23mm. At 176→177 it shrinks +117.92mm→+18.83mm without changing sign. Thus an assumed discrete sign-flip-only explanation is also inadequate. Per-edge X/depth ordering and angles are in `conditioning.json`.

## 7. Frozen S0 inference and 2D versus 3D interventions

CPU execution uses the authenticated checkpoint, four CPU threads, float32, AMP off, batch 512, neutral signs. No weights or training state changed.

| Reproduction check | Result | Frozen tolerance |
|---|---:|---:|
| Cached GPU vs CPU, max absolute component | 1.267µm | 10µm |
| CPU exact repeat, max component | 0 | 0.1µm |
| Cached vs CPU maximum Rp angle | 0.000645° | 0.05° |
| Cached vs CPU maximum Rt angle | 0.000156° | 0.05° |
| Derived-midpoint rounding/no-op drift | 0.507µm | Accounted separately |

This is compatible numerical reproduction, **not bitwise GPU/CPU equivalence**. Interventions compare against the CPU baseline to avoid attributing that drift to input changes.

| 1px direct-joint intervention | Median Rp response | p95 Rp response | Maximum Rp response | Maximum Rt response |
|---|---:|---:|---:|---:|
| Fixed cached crop | 0.461° | 3.725° | 20.451° | 4.398° |
| Recomputed crop | 0.459° | 3.654° | 20.451° | 4.398° |

The worst case is **Action 170/source 248**, right hip image x −1px: max 3D component change **13.579mm**, Rp **20.451°**, Rt **0.770°**, relative frame **21.111°**. This dense-model case is not among the fixed 72 Blender probe states; its orientation response is directly measured from S0, not a claimed fresh Blender render.

The largest paired difference between crop-policy Rp responses is **1.314°**. Thus normalization can influence response, but this matched test does not support blaming crop recomputation for all large orientation jumps. Learned response, short-axis frame amplification and observation errors interact. No additive error percentages are assigned.

## 8. Selected source-conditioned causal comparisons

Every intervention changes only its declared direct-observation group, rederives dependent virtual points/crop, reruns frozen S0 and decodes the same Rp/Rt formula. Previous points are diagnostic replacements, not stationary-body assumptions. Texture endpoints are image evidence, not corrected anatomy.

| Destination Action | Original Rp step | Previous hips Rp step | Previous shoulders Rp step | Texture endpoints Rp step |
|---|---:|---:|---:|---:|
| 22 | 10.965° | 9.683° | 6.220° | 4.896° |
| 23 | 52.486° | 8.886° | 51.470° | 3.253° |
| 26 | 76.421° | 68.749° | 9.257° | 3.463° |
| 27 | 76.324° | 72.594° | 39.296° | 37.421° |
| 171 | 64.430° | 12.621° | 64.123° | 12.728° |
| 172 | 10.625° | 8.144° | 9.949° | 7.397° |
| 177 | 19.193° | 24.625° | 16.064° | 26.476° |
| 200 | 1.370° | 0.892° | 1.361° | 0.912° |

At Action 26, previous shoulder input also reduces Rt's 80.459° step to 2.093°. Shoulder observations can therefore change **pelvis** orientation through the learned model; this would be missed by attributing each output frame only to its named anatomical input pair. At 23, hip observations dominate the global step instead. At 177, texture-based substitution worsens Rp. These controlled effects distinguish model response from decoder-only sensitivity, while the missing anatomical 2D/3D GT prevents a unique decomposition into M1 error versus S0 estimation error.

## 9. Actual Blender propagation and invariant accounting

A new Blender 5.1 process reads A without saving it. All 240 original-frame heads/tails match W144 **exactly**. It then detaches the Action in memory and evaluates **72** specified diagnostic states: 36 selected worst-Rp 3D perturbations (12 frames ×3 magnitudes), and 36 selected S0 observation interventions.

Only Object and spine_01 matrices are replaced; all other captured local bone matrices and Object translation/scale are retained. Independent full rest-relative FK and actual Blender agree to **0.504µm maximum head error**, below the declared 10µm tolerance. These are actual hierarchical readbacks, not sums of component displacements.

Illustrative left-shoulder steps against the previous original frame, with body-parent-only propagation:

- Action 26: original **473.27mm**; previous shoulders **46.49mm**; texture endpoints **51.55mm**.
- Action 27: original **549.84mm**; previous shoulders **67.73mm**; texture endpoints **55.40mm**, despite remaining Rp change 37.42° — local/global interactions matter.
- Action 177: original **244.01mm**; previous hips **268.60mm**; previous shoulders **208.68mm**; texture endpoints **252.14mm**.

These values demonstrate causal propagation and regressions in diagnostics. They are **not** evidence of improved source fidelity or full re-retargeted model candidates: child-local channels deliberately remain original to isolate parent effects.

No protected Action, pin or edit-domain channel is modified. The historical W126V accepted copy remains authentic. Candidate world-pin, edit-footprint and reversal gates are **NOT APPLICABLE**, not PASS, because there is no saved candidate. The source files are unchanged at the end.

## 10. Synthetic known-orientation controls

Ten 90-frame sequences cover stationary body, articulated arms, gradual 180° turn, genuine one-frame 90° turn, isolated/persistent false 3D orientation, 2D observation jump, missing/LR-swapped observations, depth ambiguity/near-degeneracy, fixed pelvis with torso twist, and fixed pelvis with torso bend.

GT is a known synthetic pelvis orientation plus independently articulated torso orientation. The diagnostic orthographic projection is explicit, with simulated texture residuals; these do **not** establish rendered M1 performance. Actual-image/M1 synthetic sequences were not run. Exact geometry controls isolate the frame decision; actual S1 CPU model response and real rig propagation are evaluated separately. No synthetic GT enters S1.

| Frozen gate | Result |
|---|---|
| Stationary, arms move without body turn | PASS: 100% coverage, 0° orientation error |
| Gradual genuine 180° turn | PASS: 100% coverage, numerical-zero error |
| Genuine one-frame 90° turn | PASS: 100% coverage, 0° error; no angular-speed clamp |
| False 3D flip rejection ≥90% | **FAIL: 18.18% (2/11 affected frames)** |
| Weak/degenerate pelvis abstention ≥95% | PASS: 100% |
| Recovery after missing observations ≤2 frames | PASS: first qualified pelvis one frame after restoration |
| Fixed pelvis, actual torso twist ≤1° error | PASS: correct separation |
| Fixed pelvis, actual torso bend ≤1° pelvis error | **FAIL: 29.995° maximum false pelvis tilt** |

The qualifier rejects a one-frame jump and the entrance to a 10-frame wrong branch, but subsequently accepts its wrong interior: accepted-frame p95 error **90°**. Swapped-identity plateaus can similarly retain **180°** error. Mere temporal continuity and good local conditioning do not establish the correct orientation branch.

The shared trunk axis assigns real torso bend to Rp even with exact synthetic hip/torso positions. This is distinct from the numerical short-axis issue and remains wrong under otherwise smooth, well-conditioned geometry. Pure torso twist passes. Synthetic GT makes that distinction possible; S1 does not supply equivalent pelvis orientation GT.

## 11. Source-conditioned qualification and candidate eligibility

Same-input qualification is the only tested temporal mechanism here. It emits original Rp/Rt when its declared checks permit, otherwise **null**, with a reason. It never creates an alternative 3D rotation or fills an Action gap.

| S1 result | Pelvis | Thorax |
|---|---:|---:|
| Qualified frames | 196/240 | 195/240 |
| Point/texture disagreement flags | 20 | 41 |
| Weak-geometry flags | 16 | 0 |
| Unconfirmed depth-turn flags | 16 | 1 |
| Texture unavailable flags | 0 | 3 |

Reasons overlap. Both orientations are available on **172/240** frames. Action 27's pelvis alone remains qualified despite the large source-inconsistent event; the thorax is flagged. Thus “QUALIFIED_MEASUREMENT” is demonstrably not an accuracy certificate.

**Candidate eligibility: FAIL.** Persistent false branches and pelvis/torso bend separation fail frozen controls, and source/conditioning evidence does not supply complete replacements. No real S1 B Action was constructed. Candidate A/B improvement and reversibility are NOT APPLICABLE. No parameter sweep, source-wide side-facing lock, velocity cap, generic rotation smoothing or unsupported hidden-pose filling was introduced.

## 12. Source fidelity and qualitative evidence

`user_eval/W145_body_frame/` contains:

- `S1_source_body_frame_diagnostics_30fps.mp4`: source plus unchanged A from the same two W144 fixed diagnostic views, with new conditioning/qualification overlays. **240 decoded frames, 30.000fps, 8.000s**. No B panel or corrected animation exists.
- `conditioning_and_attribution.png` and `synthetic_successes_and_failures.png`: full sequence and clear positive/negative controls.
- Aspect-preserved source/A sheets for early-turn, real-movement, quiet, parent-change, boundary, occlusion and late windows; Korean README.

Agent still-frame review confirmed the source remains broadly side-facing through 21–27 while A changes facing; at 168–180 A has inherited lean/parent changes; overhead/late source motion is real and A retains posture/arm mismatch. Quiet stability is not proof of correct absolute pose. The rig's rest-pose right arm and source self-occlusion remain unresolved. These are observations of the **baseline**, not a correction evaluation.

No trusted S1 intrinsics or calibrated diagnostic-view-to-source projection exists. Source pixel evidence is evaluated separately from rig World coordinates. Camera Geometry remains frozen and absent from this Action's orientation-producing path.

**Native-rate perceptual review: BLOCKED.** A new `cua_repl` inventory attempt failed before UI access with `0xC0000142`. Still sheets and figures were inspected, and the movie's complete frame count/fps were verified, but native-speed perception and human acceptance are not claimed. This does not prevent the controlled numerical diagnosis.

## 13. Tests, resources and reproducibility

**162 PASS in 9.66s**, Python `-X utf8`: W145 (17), W144 (21), W140–W143 camera (92), existing torso/pelvis/Object/FK contracts (32). The initial pre-official 15/1 unit-fixture discrepancy and v2 resolution are retained in §4. Synthetic scientific gate failures in §10 are separate from software-contract test results.

Resource snapshot: **18.77MB** new scratch/review evidence before final documents/manifests, under 8GiB. Conditioning plus model interventions: **10.63s**; dense 7,680 S0 forwards: **2.015s included in that time**. Cumulative historical hashing: **24.02s**. These are named-stage times, not the whole authoring wall time. No stage approached 2h. Training/GPU/downloads/Generator inference/real correction candidates: **0**.

Reproduction uses authenticated local inputs and the project CPU Python. Evidence writers refuse existing outputs; preserve existing results rather than overwriting them. The harness's output root is checkout-relative `result/w145`, while frozen historical inputs retain their exact recorded paths.

```powershell
$env:PYTHONPATH='src'
# declaration_v2 and execution_manifest are already frozen; verify before runs
.venv_build_full/Scripts/python.exe -X utf8 scripts/w145_measure.py
.venv_build_full/Scripts/python.exe -X utf8 scripts/w145_synthetic.py
# blender_execution.json records exact fresh-process argv/spec/readback
.venv_build_full/Scripts/python.exe -X utf8 scripts/w145_report.py
```

The focused suite is named in `result/w145/tests.txt`/this section. `blender_execution.json` contains the executable, input, spec and output arguments. Source-to-Action conversion, confidence rules, actual S0 response, all perturbation records and original/qualified frames remain machine-readable.

| Result file | SHA-256 |
|---|---|
| `assessment.json` | `422a0dbf3aa092fcc6e1249fc722a7149d8f37577f68a9233ba51cd8c195bc37` |
| `conditioning.json` | `580a762bc174ddb2ec3d1188599e50a0dad65f48556ddf3cde6f9649630b60b4` |
| `model_identity.json` | `a438e812e78aa67e377f26af2fb096af94520abdd361600d874f14fa2237a03f` |
| `synthetic.json` | `cfe2f7a1a7174928c7e46bbae38abc83b9d67254bd467f96cc9364b8c36f843b` |
| `blender_readback.json` | `152c19d36fa94a67905beef2f8b6246b74c076f69b3f6d43b576e55372bc297e` |

## 14. Cumulative audit and exact changes

See [W141–W145 cumulative audit](AUDIT_W141_W145_2026_11_16.md). W141–W144 manifests contain **13,855** files; every file was re-hashed with zero missing/mismatched entries. W141 amended instrument hashes 8/8, W142 10/10, W143 13/13 and W144 declaration-bound files remain unchanged. W145 core/official execution hashes match. Production/Generator contracts and old Worklogs are untouched.

W145 additions: `scripts/w145_body_frame_core.py`, `w145_harness.py`, `w145_measure.py`, `w145_synthetic.py`, `w145_blender_probe.py`, `w145_report.py`; `tests/test_w145_body_frame_attribution.py`; initial/v2 declarations and execution manifest under `examples/experimental_body_frame/`; this Worklog; the cumulative audit. The separate W144 preservation commit contains only its original ten files. No third-party files were committed. Scratch JSON/NPZ/media remain ignored/local.

## 15. Official judgment and unresolved information

**PARTIALLY SUPPORTED:** source input changes, S0 response, frame conditioning and actual rig propagation are separated with controlled evidence; a shared-up-axis ownership assumption is demonstrated on independent synthetic articulation. The tested source-conditioned qualifier preserves clear genuine turns and expresses some ambiguity, but does not reliably reject persistent wrong branches or infer independent pelvis tilt. No source-consistent real Action improvement is shown.

M1 point reliability, learned S0 geometry, mathematical conditioning, synthetic numerical correctness and actual animation utility are distinct findings. Unsupported metric depth/placement, anatomical identity under occlusion, hidden pelvis orientation and calibrated uncertainty remain unresolved. A poor qualifier does not establish that all temporal reconstruction is impossible. W144's judgment and W140–W143's positive camera capabilities are preserved. No next Batch, new architecture/training direction or Production adoption is recommended.
