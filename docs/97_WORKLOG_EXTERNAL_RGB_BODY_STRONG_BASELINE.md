# Worklog — External RGB Full-Body Strong Baseline (2026-10-25)

## Starting state and environment

**Branch and HEAD.** `arch/single_frame_first`, starting HEAD `98811d9` (Worklog 96). Local, origin and LabServer63 were
in sync and the worktree was clean. I read PRODUCT_VISION.md and Worklogs 93–96.

**Prior decisions.** W96 stays CASE C. Candidate T was neither used nor tuned.

**History of this worklog.**
1. A first version (commit `39f80b4`) recorded CASE D: the official SMPL v1 body model was absent, and the only GPU was
   busy.
2. The licensee then registered at smplify.is.tue.mpg.de and downloaded the SMPLify package.
3. A detached server watcher started the inference once the GPU had been idle (< 1 GiB allocated) for 10 consecutive
   minutes.

This version supersedes the interim CASE D report. No other worklog was changed.

**Machines.**
- LabServer63: RTX 3080 Ti (12 GB) for inference.
- Local M1 Pro: evaluation and Blender 5.1.2.

**What was not changed.** No AnimCV checkpoint, S0/J1X prediction, IK/FK, rig, schema or production default. Nothing was
trained, fine-tuned or optimised at test time. W96 T was not used.

## 0. Answer

**CASE A, with stated regressions: HMR 2.0b is a materially stronger 3D source than frozen S0, and the advantage
survives the unchanged W95 rig path in the failure regimes that limited AnimCV.**

**Reconstruction** (3DPW TEST, 7076 rows, identical frames, sequence-cluster 95% intervals):

| Metric | S0 | HMR 2.0b | E − S |
|---|---|---|---|
| MPJPE | 105.1 mm | **71.0 mm** | −34.2 [−38.1, −30.3] |
| Knee-plane error | 23.6° | **13.3°** | −10.3 [−13.8, −7.5] |
| Wrong knee branches | 5% | 0% | — |
| Thigh direction | 12.5° | **9.8°** | — |
| Shin direction | 18.8° | **14.9°** | — |
| Pelvis vs SMPL | 27.6° (joint proxy) | **13.4°** (from predicted SMPL global orient) | — |
| Adjacent-frame limb residual | 9.7° | **4.0°** | — |
| Knee-plane flips | 139 | 0 | — |

The predeclared material rule is met on TEST and on the GT knee flexion ≥ 45° subset. VALIDATION (−26 mm MPJPE) and
MPI-INF-3DHP (−49 mm) agree. 3DHP is in-domain for HMR 2.0: it trained on 3DHP train.

**Common rig** (same W95 zero-twist IK, root, thorax and arm; only the leg joints change). Rig segment errors track the
source errors one-for-one, so the IK passes source quality through.

| Regime (TEST) | Thigh S → E | Shin S → E | Rig knee plane S → E |
|---|---|---|---|
| Seated/crouched | 19.2 → **7.7** | 18.1 → **11.6** | 17.1 → **8.8** |
| Knee flexion ≥ 45° | 14.4 → **8.4** | — | 16.1 → **10.0** |
| S0 wrong-plane rows | — | 25.5 → **12.5** | 47.6 → **14.5** |

**Regressions and limits:**
- **Upright walking shins are worse with HMR 2.0b:** rig shin 9.9 → 14.5° on 5228 rows. HMR 2.0b over-bends some
  straight or back-facing knees.
- **Occluded legs fail for both** (downtown_bar_00).
- **Thigh roll vs SMPL is unchanged** (6.3 → 6.2°). That is the zero-twist convention, not the source.

## 1. Model provenance, licences and access

| Item | Value |
|---|---|
| Code | `https://github.com/shubham-goel/4D-Humans` at `efe18deff163b29dff87ddbd575fa29b716a356c`, MIT |
| Checkpoint | official `hmr2_data.tar.gz` (sha256 `0fdf9e66…`; a plain tar despite the name) → `epoch=35-step=1000000.ckpt` (HMR2.0b, upstream default; sha256 `c9ea09c8…`), `model_config.yaml` (`2ec161e1…`), `smpl_mean_params.npz` (`6fd6dd68…`), `SMPL_to_J19.pkl` (`a66e3404…`) |
| Archive contents | `data/smpl/` is empty (no body model is redistributed) |
| Body model | SMPL v1 neutral from the licensee's own smplify.is.tue.mpg.de download (`smplify_public.zip`, sha256 `a948f73d…`) → `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` (`98e65c74…`; 24 joints, 10 betas, 207 pose blend shapes; non-commercial research licence) |
| Body-model location | `~/animcv-data/body_models/smpl_v1/` (+ `PROVENANCE.txt`) |
| Body-model conversion | upstream `check_smpl_exists` → `convert_pkl` → `SMPL_NEUTRAL.pkl` (`94082a25…`), array-identical to the source (v_template, shapedirs, posedirs, J_regressor, weights, kintree, faces) |
| Gate | `scripts/check_external_hmr2_access.py` rejects the server's renamed SMPL+H files and accepts the SMPL v1 file. The upstream check tests only the filename, so the structural gate stays necessary |
| Training data (released `model_config.yaml`) | H36M, MPII, COCO 2014 (+ ViTPose replicate), MPI-INF-3DHP train (weight 0.02), AVA, AI Challenger, InstaVariety; **no 3DPW** |

**Credentials.** The download script prompts with a hidden field, passes values through mode-600 temp files and stores
nothing. The account and licence acceptance were the user's. The final file was downloaded in the browser.

## 2. Environment and reproduction

**Image.** `animcv-w97-hmr2:cu124` (`scripts/w97_hmr2.Dockerfile`) is built on top of `animcv-external:cu124` and leaves
it unchanged.
- Python 3.10.12, torch 2.4.1+cu124.
- numpy 1.23.5, needed by the upstream chumpy-based pickle conversion.
- pinned: pytorch-lightning 2.2.5, smplx 0.1.28, chumpy @580566ea (installed without build isolation), hydra-core
  1.3.2 / omegaconf 2.3.0 (the checkpoint's saved config), pyrender + EGL/X11 libraries (import only).

**Runtime state** under `~/animcv-data/external/w97/`: cache, build context, population, predictions and logs.

**Commands:**
- `scripts/run_external_hmr2_baseline.py population` (AnimCV env).
- `… infer` (in the image, GPU; `gpu_wait_and_infer.sh` started it after 10 idle minutes).
- `scripts/eval_external_hmr2_baseline.py` (AnimCV env).

**Inference performance.**
- 13,211 crops in 2835 s on the RTX 3080 Ti: 0.21 s per crop, batch 1, including the upstream full-image Gaussian
  anti-alias step on the CPU.
- Peak GPU allocation 2.67 GB.
- CPU: 1.6–1.7 s per crop.

**Determinism.**
- A CPU re-run is bit-identical.
- CPU vs GPU differ by at most 4.5e-7 m on the 24 joints.
- An evaluation re-run is byte-identical (§10).

## 3. Population, crops and GT isolation

**Population** (declared in `PREDECLARED` before any prediction existed). Every W95 row, with no sampling:
- 3DPW TEST 7076 (main);
- VALIDATION 3407;
- MPI-INF-3DHP 2728. 1926 of these have all 17 GT joints valid and enter the reconstruction metrics.

Coverage: 13,211 of 13,211 rows have a box and a prediction.

**Challenge subsets** are TEST rows defined by evaluation-only labels and reported separately:
- waist bend, seated/crouched;
- GT knee flexion ≥ 45°, near-straight;
- S0 knee plane > 45° from GT;
- out-of-plane limbs.

**Person crop.**
- A tight box of the valid **dataset-provided 2D detections in the bank**, which are the observations S0 consumed,
  enlarged 1.2× and passed to the official `ViTDetDataset` (which expands it to 192:256).
- No GT and no temporal information. Identity is per actor by construction.
- This is an end-to-end source comparison: S0 sees 2D detections only, HMR 2.0b sees the RGB crop at the same box. It
  is not a clean causal isolation of RGB information.
- The demo's ViTDet detector was not used. The fixed box follows upstream `eval.py`-style dataset boxes.

**Canonical mapping.**
- The 24 SMPL joints of the predicted parameters (`smplx.SMPLLayer` forward) are in OpenCV crop-camera axes. AnimCV
  canonical is (x, z, −y).
- 17 joints use the 3DPW GT adapter's index map (tested identical). Everything is pelvis-relative and metric, with no
  scale or Procrustes alignment in the primary metrics.
- Parity on fixtures and real rows: inclination and yaw agree with GT within a few degrees.

**Caveats.**
- The crop-camera rotation for off-centre people is not corrected.
- 3DPW GT and HMR 2.0b share the SMPL joint definition, which favours E on joint placement (pelvis and spine joints in
  particular). 3DHP joints differ from SMPL joints, so limb metrics are the meaningful ones there.

## 4. Reconstruction results

**3DPW TEST** (mean; E − S with sequence-cluster 95% interval; "E better" = share of rows):

| Metric | S0 | HMR 2.0b | E − S | E better |
|---|---|---|---|---|
| MPJPE (mm) | 105.1 | 71.0 | −34.2 [−38.1, −30.3] | 78% |
| PA-MPJPE (secondary) | 89.8 | 51.5 | −38.3 | 74% |
| Limb depth error (mm) | 116.7 | 82.2 | −34.5 [−40.0, −29.1] | 71% |
| Upper arm / forearm (°) | 18.5 / 29.1 | 11.6 / 17.9 | −6.9 / −11.1 | 75 / 67% |
| Thigh (°) | 12.5 (med 9.5) | 9.8 (med 7.7) | −2.7 [−4.1, −1.4] | 59% |
| **Shin (°)** | 18.8 (**med 9.6**, p95 81) | 14.9 (**med 12.5**, p95 36) | −3.9 [−7.8, −1.0] | **43%** |
| Wrist / ankle position (mm) | 202 / 181 | 120 / 137 | −82 / −44 | 73 / 57% |
| Knee-plane error (both ≥ 20°) | 23.6 | 13.3 | −10.3 [−13.8, −7.5] | 69% |
| Wrong knee branch | 5% | 0% | −5 pp | — |
| Torso up / forward (°) | 9.9 / 15.1 | 6.7 / 10.7 | −3.2 / −4.3 | 60 / 56% |
| Pelvis vs SMPL: E rotation vs S joints | 27.6 | **13.4** | −14.2 [−15.6, −12.7] | 88% |
| Pelvis vs SMPL: joints vs joints | 27.6 | 19.6 | −7.9 | 72% |

**Reading the shin row.** E wins on the mean and the tail but loses on the median: in most typical rows S0's shin is
closer.

**VALIDATION:** MPJPE 90.5 → 64.5 (−26.0 [−34.9, −14.6]); knee plane 21.3 → 12.5; thigh and shin not significant.

**MPI-INF-3DHP** (1926 rows; in-domain for E):
- MPJPE 155.1 → 105.6 (−49.5 [−62.6, −29.4]).
- Thigh / shin 18.3 / 15.5 → 12.8 / 9.9.
- Knee plane 25.0 → 17.0.

**Challenge subsets (TEST):**

| Subset | Rows | MPJPE S → E | Thigh | Shin | Knee plane | Wrong branch |
|---|---|---|---|---|---|---|
| Seated/crouched | 924 | 116.5 → 74.3 | 18.2 → 8.5 | 17.4 → 11.0 | 18.7 → 9.0 | — |
| GT knee flexion ≥ 45° | 2549 | 112.3 → 70.4 | 15.5 → 8.9 | 23.2 → 13.7 | 23.4 → 12.2 | — |
| S0 wrong knee plane | 490 | 174 → 83 | 29.5 → 12.5 | 81 → 24 | 86 → 26 | 45% → 0% |
| Waist bend | 241 | 107 → 76 | not significant | not significant | 19.5 → 11.6 | — |
| Out-of-plane limbs | 3090 | 117 → 75 | — | — | 23.4 → 12.7 | — |
| Near-straight | 3 | — | — | E worse (11 → 40) | — | — |

**Upright stratum** (5908 rows): MPJPE −33 mm, but shin median 8.8 → 13.0.

**Material rule** (CI below 0 and ≥ 10% relative, for MPJPE, thigh, shin and knee plane): met on TEST and on the
flexion ≥ 45° subset.

## 5. Pelvis and knee information hierarchy

**Pelvis.**
- E's SMPL global orient gives a pelvis orientation 13.4° from the 3DPW SMPL reference, against 19.6° for E's own joints
  through the W94 lower-trunk proxy and 27.6° for S0's joints.
- So the predicted rotation carries information beyond the joint positions.
- But the reference is SMPL too: same body-model family and fitting convention. This is not evidence of BaseRig pelvis
  correctness.

**Knee.** E's knee planes are better and never backward. In the rig this does not change thigh roll vs SMPL (§6),
because zero twist sets the roll. Predicted knee-plane accuracy is not anatomical thigh-twist evidence.

## 6. Common-rig replay

**Setup.** W95 legs with zero twist Z. Root = W95 B_S0 control: SMPL pelvis Object (evaluation-only), S0 thorax, J1X arm.
Only the leg joint source changes. Paired rows are those where S0's 2D leg joints are valid.

| Population (paired rows) | Rig thigh | Rig shin | Rig knee plane | Off-hinge | Roll vs SMPL | W95 ROM-only |
|---|---|---|---|---|---|---|
| TEST all (6339) | 10.4 → 9.3 (n.s.) | **11.1 → 14.1** (+3.0 [1.1, 4.8]) | 15.7 → 11.3 (−4.4 [−5.9, −2.9]) | 9.7 → 9.9 | 6.35 → 6.18 | 2.1 → 1.4% |
| TEST upright (5228) | 9.0 → 9.5 | **9.9 → 14.5** | 15.1 → 12.2 | 8.0 → 9.2 | = | = |
| TEST seated/crouched (881) | 19.2 → **7.7** | 18.1 → **11.6** | 17.1 → **8.8** | 19.5 → 14.4 | 16.2 → 15.4 | 8 → 3% |
| TEST knee flexion ≥ 45° (2243) | 14.4 → **8.4** | 15.3 → **12.6** | 16.1 → **10.0** | 13.3 → 11.3 | 9.4 → 9.1 | 4 → 2% |
| TEST S0 wrong plane (91) | 19.0 → **11.6** | 25.5 → **12.5** | 47.6 → **14.5** | 20.7 → 9.9 | 6.4 → 5.8 | — |
| VAL all (3109) | 9.3 → 9.1 | 9.9 → 11.6 (n.s.) | 15.9 → 12.2 | = | 6.4 → 6.1 | — |

**Attribution.** On the same rows, rig thigh/shin errors equal the source errors within about 1°:
- TEST all, source shin 10.7 → 13.6;
- seated, source thigh 17.2 → 8.1.

So the IK neither blocks nor creates the difference.

**Coverage.** On the 737 TEST rows where S0's 2D leg joints are invalid, S0 has no legs and E+Z still produces them:
thigh 12.1°, shin 27.0°, knee plane 27.2°. These are hard, mostly occluded rows.

**Integrity.**
- Severe thigh-local jumps: 0 for both sources.
- Blender FK vs export: legs ≤ 3.8e-5, arm ≤ 9.3e-5.

## 7. Temporal accounting (matched adjacent rows, no smoothing, HMR 2.0b single-frame)

| 3DPW TEST | S0 | HMR 2.0b |
|---|---|---|
| Limb step residual vs GT (mean / p95) | 9.7 / 43.3° | 4.0 / 14.3° |
| False motion when GT limb still (< 2°) | 6.9° | 2.3° |
| Limb depth-motion residual | 69.8 mm | 37.2 mm |
| Torso frame flips (> 90° while GT < 45°) | 179 | 3 |
| Knee-plane flips | 139 | 0 |
| Pelvis-rotation spins (E only) | — | 2 / 7039 |

VALIDATION shows the same pattern: 8.5 → 3.1°, flips 54 → 3, 125 → 0.

## 8. Real-scene review

**Clips** (skinned mannequin; full body and leg close-ups; slots S0 legs + Z | HMR 2.0b legs + Z | GT legs + Z
evaluation reference; same root, thorax, arm, camera and timestamps):

| Clip | What is visible |
|---|---|
| 3dpw_seated_or_crouched (flat_guitar_01) | E's knees and shins follow the GT overlay; S0's right shin is 67° off and its knee plane 16/26° vs E 5/11° |
| 3dpw_worst_hinge | as above |
| 3dpw_walking_upright (walkBridge, seen from behind) | **E worse**: one knee over-bent, shin swinging 50–90° off at times; S0 closer |
| 3dpw_s0_wrong_knee_plane (downtown_bar_00 actor1; fixed rule: most S0 wrong-plane rows outside W96 windows) | legs occluded by crates: S0 legs unavailable (held), E legs also wrong (shin 41°, plane 55°) |
| 3dpw_pelvis_stable_bend | small differences |

**Static review figures** (`review_*.png`; GT green, S0 orange, E blue; side and front views with the RGB crop; fixed
rules):

| Figure | What it shows |
|---|---|
| E much better | S0 collapses (MPJPE 380–430 mm) on partial or narrow detections |
| Typical | — |
| E much worse | frontal standing people on stairs/slopes: E folds the knees in depth (MPJPE 122–170 vs S0 50–96 mm) |
| Deep knee flexion, near-straight, S0 backward plane | — |

**Not generated.** No E-native full-body preview.

## 9. Attribution summary

| Source of difference | Evidence |
|---|---|
| Source pose / depth quality | dominant: E better on MPJPE, depth, arms, knee planes, temporal stability, robustness tails |
| Source per-row precision on upright legs | S0 better: shin median 8.8 vs 13.0° upright |
| IK / zero twist | transmits source directions (rig ≈ source); thigh roll unchanged by either source |
| Pelvis | E's rotation is closest to the SMPL reference (same model family) |
| Occlusion | both fail |
| Camera pitch / crop camera | E's frontal knee folding on pitched views is consistent with an uncorrected crop camera; not separately tested |

## 10. Tests and files

**New files:**
- `scripts/check_external_hmr2_access.py` (gate).
- `scripts/fetch_smpl_v1.sh` (credentialed SMPLify download, interactive).
- `scripts/w97_hmr2.Dockerfile`.
- `scripts/run_external_hmr2_baseline.py` (population, infer).
- `scripts/eval_external_hmr2_baseline.py` (evaluation, rig replay, Blender cases, review figures).
- `tests/test_external_hmr2_access_gate.py` and `tests/test_external_hmr2_baseline.py`.
- This worklog.

No existing file changed.

**Tests (10):**
- checkpoint/asset hash pins and the Dockerfile base;
- box from valid detections only, the fixed factor and minimum joints;
- canonical axes, handedness, and the joint map identical to the 3DPW GT adapter;
- inference and population free of GT;
- metrics, Procrustes and the material rule; pelvis from global orient;
- frozen-input hashes; no production import;
- the gate tests.

**Test runs.** With the W93–W96 suites: 43 passed locally and 43 passed on LabServer63 (docker
`animcv-external:cu124`). Broad regression was not run (no shared contract changed).

**Reproduction.**
- The evaluation re-run is byte-identical (`report.json`, cases, review rows).
- CPU inference is deterministic.
- Blender FK checks are as in §6.

**Outputs.**
- Images and videos: `~/animcv-output/97_framepose_external_hmr2_baseline/`.
- Population, predictions, report, cases and .blend files: archived on LabServer63 at
  `~/animcv-output/framepose_external_hmr2_baseline_archive/`.

## 11. CASE and evidence limits

**CASE A.**
- A fixed, officially released RGB estimator (HMR 2.0b) supplies materially better 3D geometry than frozen S0 on the
  relevant failure regimes: deep knee flexion, seated/crouched, S0 wrong knee planes, out-of-plane limbs, collapses and
  temporal flips.
- The improvement survives the unchanged W95 rig path there.

**Not universal.**
- On typical upright walking rows S0's shins are more precise, and E's rig shins are worse (+4.6°).
- Occluded legs fail for both.
- Thigh roll is set by the zero-twist convention, not the source.

**Evidence limits:**
- 3DPW GT shares E's SMPL joint and body-model conventions.
- The comparison is end-to-end: RGB crop vs 2D detections.
- 3DHP is in-domain for E.
- The B_S0 root is evaluation-only.
- 3DPW has only 3 gravity-lying rows.

**Scope.** An external-model win does not by itself authorise replacing S0. Nothing was integrated.
