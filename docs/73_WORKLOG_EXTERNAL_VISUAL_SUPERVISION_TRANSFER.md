# Worklog — External Visual Supervision Transfer: BEDLAM (2026-10-06 → 10-12)

**Question.** With the Worklog 69 head and the frozen Worklog 68 Qwen3-VL representation unchanged, does adding one substantially more diverse paired RGB↔3D source improve continuous left-arm depth (f = unit(segment).Y) on the unchanged 3DPW held-out benchmark?

**Answer.** Yes. **CASE A.**
- **E1 (3DPW + BEDLAM, 50/50, same budget) vs E0 (3DPW only), 3DPW held-out:**
  - test forearm \|f\| 0.243 → 0.215 (−11.7%) and chain 0.251 → 0.200 (−20%);
  - validation forearm 0.223 → 0.210 and chain 0.180 → 0.167.
- **The other acceptance conditions also hold:**
  - error p95 drops on test and validation;
  - the clean subset improves in mean (its forearm tail rises slightly);
  - sign accuracy and sign-correct magnitude both improve;
  - the gain is spread across 31/37 (forearm) and 33/37 (chain) test sequences.
- **Validation gains are smaller than test gains, in the same direction:** about half for the forearm, about a quarter for the chain.
- **Seed robustness:** two further seeds keep every E1 − E0 sign.
- **E0 reproduces Worklog 69 C1 exactly.**
- **Limits:** one synthetic source and one fixed ratio. Diversity is a material blocker but not the whole ceiling: E1 closes only about 8–10% of E0's train/validation gap.

## Starting state and files

- **Start:** fetched `origin/arch/single_frame_first` and confirmed local = remote = `cc05335612289fc22b8f0d73a63a15b338252cd2`, with a clean worktree.
- **Files added (diagnostic only):**
  - `containers/external_supervision/Dockerfile`
  - `src/framepose/openpose_coco18.py`
  - `src/framepose/bedlam_supervision.py`
  - `scripts/run_openpose_provider_parity.py`
  - `scripts/download_bedlam_external_supervision.sh`, `scripts/download_bedlam_validation_png.sh`, `scripts/download_bedlam_files.sh`
  - `scripts/build_bedlam_external_bank.py`
  - `scripts/generate_bedlam_validation_labels.py`
  - `scripts/finalize_bedlam_subject_split.py`
  - `scripts/extract_external_qwen3vl_features.py`
  - `scripts/run_external_visual_supervision_transfer.py`
  - `scripts/run_external_supervision_seed_check.py` (secondary seed robustness)
  - `scripts/audit_external_supervision_report.py` (post-hoc review audits)
  - `scripts/render_external_supervision_review.py`
  - `tests/test_framepose_external_visual_supervision.py`
  - this worklog
- **Unchanged:** every existing file and checkpoint, including the Worklog 68 cache, the Worklog 69 head and C1, Worklogs 70–72, FrameBank and its 3DPW validation/test, frozen H0, Worklog 65 D1, production FramePose, FK and IK.

## Candidate dataset audit (before any training)

Audited for: RGB, recoverable camera-space 3D, left shoulder/elbow/wrist, image↔annotation identity, appearance/scene/subject/pose diversity, perspective relevance, and adapter cost.

| candidate | RGB + camera-space 3D; left S/E/W | image↔annotation identity and camera correspondence | scene / appearance diversity | subjects | pose / articulation diversity | perspective relevance | adapter cost | access on 2026-10-06 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **BEDLAM v1** (synthetic, Unreal) | 1280×720 renders; per-body SMPL-X, so left S/E/W = SMPL-X joints 16/18/20 | per-frame `cam_int`/`cam_ext`; labels reproject exactly onto their own `gtkps` | 30 scene folders (HDRI, suburb, gym, office, archviz, stadium); per-instance skin/clothing/hair textures | 271 bodies (198 people) | thousands of per-body mocap clips, several bodies per sequence | true perspective; static, orbit, zoom, close-up and pitched cameras | moderate: SMPL-X forward pass, close-up rotation, 30fps PNG for validation | registration + login | **selected** |
| MPI-INF-3DHP (real) | 14-camera studio; 28-joint camera-space 3D with left S/E/W listed | calibrated per camera; adapter already exists for 2D/3D | training set is one green-screen studio | 8 subjects × 2 outfits | 8 activity sets | studio cameras, some on the ceiling | low (RGB videos still to fetch) | open download | rejected: scene diversity below 3DPW |
| AGORA (synthetic) | SMPL-X fits | per-image camera | high | many scans | mostly static scan poses | perspective renders | moderate | registration | not chosen: ~14k training images; BEDLAM is larger and more diverse, from the same lab |
| Human3.6M (real) | 4-camera lab; 3D joints with left S/E/W | calibrated | one lab | 11 subjects | 15 actions | lab cameras | an adapter would have to be written | academic approval | not chosen: lowest scene diversity |

The non-selected candidates fell on scene diversity (MPI-INF-3DHP, Human3.6M) or on scale (AGORA), so adapter cost never became the deciding criterion.

**Why BEDLAM** (independent of any model result): it is the only candidate that has all of the following at once:
- large scene, appearance and subject diversity;
- broad human pose diversity (many mocap motions per body);
- exact perspective camera geometry;
- a documented SMPL-X convention that yields the three Worklog 69 segments directly;
- a scale far beyond the 34 3DPW TRAIN sequences.

## Provenance

- **Source:** BEDLAM v1, official MPI-IS download service (`download.is.tue.mpg.de`, domain `bedlam`), fetched with the owner's login on 2026-10-06 → 10-11.
- **Training labels:** `bedlam_labels/all_npz_12_training.zip` (5,670,393,178 B, sha256 `5a36249c…`).
  - Contents: 26 BEDLAM scene files + 2 AGORA files. The AGORA files are not used.
  - The four official validation scenes are absent from this release.
- **Images:** 26 training-scene tars from `bedlam_images_train` (23 at 6fps; the three "hair" scenes exist only at 30fps). `closeup_suburb_d` arrived truncated and was resumed to 18,531,399,680 B (sha256 `484e7691…`, 10,044 members).
  - Every other tar reads end-to-end. The official repo has no checksums for these tars.
- **Validation scenes:** the official BEDLAM repo (`pixelite1201/BEDLAM@f1828fd`, `train/core/config.py` `DATASET_FILES[0]`) validates on `orbit-stadium-bmi`, `orbit-archviz-objocc`, `zoom-suburbb-frameocc`, `static-hdri-frameocc`.
  - Their images are not distributed as 6fps tars (the service answers "Error: File not found."). They come from the `b0` 30fps PNG release: 20 archives, 213.9 GB, **all 20 match the official xxh128 checksums**.
  - Their labels were produced with the official processing module (below).
- **Per-scene ground truth and motion:** `b0/<scene>/ground_truth/<scene>_gt.tar.gz` for all 30 scenes and `bedlam_labels/neutral_ground_truth.zip` (2,069,198,865 B, sha256 `3142bb43…`; 16,600 motion files).
- **Body model:** SMPL-X v1.1 (`models_smplx_v1_1.zip`, 870,108,517 B, sha256 `cb593838…`), neutral, 11 betas, flat hand mean, no PCA (the BEDLAM training convention).

## External adapter contract

**3D target (GT, target-only).**
- **Joint computation:** per selected label row, neutral SMPL-X joints from `pose_cam` (165), `shape` (11 betas) and `trans_cam`, plus `cam_ext[:3, 3]`. This gives OpenCV camera coordinates (+x right, +y down, +z forward, metres).
- **Correspondence check:** these joints reproject onto the label's `gtkps` through `cam_int` with a maximum error of **0.0022 px** (`20221019_1_250_highbmihand_closeup_suburb_c`; at most 0.0008 px in every other scene). The builder aborts above 0.5 px.
- **Image size check:** every image must match `2 × principal point`, otherwise the build aborts.

**Validation-scene labels.** The official processing module (`df_full_body.py`) runs unchanged.
- Only its `__main__` orchestration is restated. Its renderer import is stubbed, because no image is opened (`visualize='none'`). The frame list comes from the PNG archive listing.
- **Consistency check:** the same module, run on a training scene (`closeup_suburb_a`), reproduces the official training release.
  - Rows present in both: 8,881 of the official 8,951. Values are identical (`pose_cam` max diff 0, `trans_cam` 1.2e-7 m, `cam_ext` 1e-16, `gtkps` 0.001 px). The remaining 70 official rows are not missing either: their image names are among the module's rows, but the (image, betas) key misses for them (sequence `seq_000105`, see the betas drift below).
  - The module also emits 69 frames that the release omits. They exist in the image tar, so the release applied an extra row filter of its own.

**Coordinate conversion.**
- **Canonical axes:** AnimCV (X, Y, Z) = OpenCV (x, z, −y), pelvis subtracted. This is identical to `three_dpw_adapter._world_to_animcv_camera`.
- **Synthetic tests:** cover left/right, up/down, nearer/farther and behind-camera validity.
- **Closeup rotation:** "closeup" scenes store landscape PNGs for portrait labels and are rotated 90° clockwise. The direction was decided by detector agreement before any use: clockwise gives 12/12, 12/12 and 9/12 correct limb joints on three probe images; counter-clockwise or unrotated detect no person, or match 0 joints.

**Joint mapping.** SMPL-X body joints 0–21 share the SMPL-24 body order, so the 3DPW adapter's table is reused (test asserts equality).
- left_shoulder = 16, left_elbow = 18, left_wrist = 20 (the subject's anatomical left).
- Only upper (S→E), forearm (E→W) and chain (S→W) enter the target: `f = unit(segment).Y`.

## Benchmark-detector observation contract

- **What the 3DPW input is:** the FrameBank 2D input for 3DPW is the OpenPose COCO-18 detection shipped with 3DPW (`benchmark_detector_observation`). AnimCV never ran that detector.
- **What runs on BEDLAM:** the same detector family is re-run:
  - CMU OpenPose COCO body model (PyTorch port: network definition `Hzzone/pytorch-openpose@5ee71dc`, weights sha256 `25a948c1…`);
  - OpenPose default inference (single scale, net height 368, part threshold 0.1, PAF threshold 0.05);
  - the exact 3DPW conversion rules (`three_dpw_adapter._canonical_2d` plus the `build_dataset` thorax→neck alias).
- **Parity with the shipped 3DPW detections**, measured on 1,134 3DPW TRAIN rows (every 10th):

| joint | shipped-valid rows also detected | shipped-invalid rows detected by the re-run | median / p95 error (fraction of body diagonal) | mean confidence, re-run / shipped | confidence correlation |
|---|---:|---:|---|---|---:|
| left shoulder | 99.9% | 86% (6/7) | 0.006 / 0.014 | .872 / .844 | 0.93 |
| left elbow | 98.9% | 50% (26/52) | 0.007 / 0.020 | .863 / .837 | 0.92 |
| left wrist | 98.5% | 47% (40/85) | 0.007 / 0.020 | .874 / .843 | 0.92 |

- **Overall match:** all 1,134 rows matched a detected person, with a median distance of 0.7% of the body diagonal.
- **Same rules, slightly different behaviour:** the re-run detector is somewhat more permissive (it recovers about half of the elbows and wrists the shipped detections miss) and about 0.03 more confident. It follows the 3DPW provider's rules closely but not identically.

- **GT is used only for identity:** the detected person whose limb joints best agree with the GT projection is taken as the labelled subject. The row is kept only if at least 6 limb joints lie within 0.1 of the GT limb-box diagonal, mirroring 3DPW's "only provided if at least 6 joints were detected correctly".
- **The detector's values are the input:** they feed the 13 pair-geometry features, the crop and the readout locations.
- **Masks follow 3DPW:** `target_valid` = detector-visible AND GT-valid, and rows without a valid pelvis are dropped.

## Qwen cache provenance

- **Same extractor:** external caches are written by the unchanged Worklog 68 extractor (`extract_qwen3vl_visual_features.main`), with only its bank loader pointed at the Worklog 73 bank. Everything else is the same: pinned snapshot `0c351dd…` / fingerprint `a0d72ded…`, 4-bit NF4, 448 px crop contract, processor, merger feature point and readout.
- **Parity:** the wrapper re-extracted the first 32 FrameBank samples in image-path order. They reproduce the Worklog 68 cache **bit-for-bit**:
  - readout, joint validity, crop hash, crop box and image hash are all identical, with a readout max difference of 0.0;
  - the committed comparison is `audit_external_supervision_report.py` and its output `report_audit.json`.
- **Runtime:** the extension image adds only scipy/smplx/pandas/matplotlib/tqdm. torch, transformers, numpy, opencv, bitsandbytes and pillow are identical to the Worklog 67/68 runtime.
- **Runtime identity checks:** the experiment requires each external manifest to share every representation field of the Worklog 68 identity, and to match its bank digest and sample-ID hash.

## Split contract

- **Official BEDLAM split:** the four validation scenes, no frames shared.
- **Subjects are not separated by it:** counted from the official per-scene `be_seq.csv` files, all 271 bodies (198 people) appear in both training and validation scenes, and 3,514 of the 3,577 validation body+motion clips are also used in training scenes (the same person and motion re-rendered in another scene).
- **Added rule, fixed before training:**
  - **Subject key:** RenderPeople person name (`rp_<name>`, all scans) or the synthetic body id.
  - **Validation subjects:** sha256(key) mod 5 == 0.
  - **External TRAIN:** training scenes, non-validation subjects, 3 evenly spaced rows per sequence (label-blind).
  - **External validation:** validation scenes, validation subjects only, 3 evenly spaced rows per (sequence, person).
- **Identity source:**
  - **Validation:** the official `sub` field.
  - **TRAIN:** the official per-sequence body list in `be_seq.csv`, disambiguated by the label's betas.
  - **Betas drift:** label betas do not always equal the official motion-file betas.
    - 205 recovered training-scene rows (counted before the subject filter) differ, by at most max |Δ| 0.046, just under the 0.05 acceptance cap. One example is `rp_christine_posed_012_1075` (0.011; the only candidate in its sequence; the motion behind the consistency-check key misses).
    - 13 rows had no candidate within 0.05 and 17 were ambiguous (margin < 0.1). These 30 rows were dropped.
- **Verified at runtime:** no shared image, sequence or subject between external TRAIN and validation.

## Fixed model and training contract

- **Head:** the Worklog 69 C1 head, unchanged: 13 pair-geometry features + endpoint A/B/global readouts, shared `Linear(4096→64)`, per-segment `Linear(205→64)→GELU→Linear(64→1)→tanh`, 301,955 parameters.
- **Not used:** attention, skeleton bias, Depth Anything, H0, oracle signs, temporal context.
- **Training:** the Worklog 71 `train()` function, imported unchanged.
  - seed 1337, with identical initialization for E0 and E1;
  - AdamW, learning rate 3e-4 cosine to 1e-5, weight decay 1e-4;
  - batch 256, 200 epochs, masked MAE;
  - selection by 3DPW validation mean \|f\|.
- **E0_3DPW_ONLY:** the Worklog 69 epoch exactly, i.e. the S0 permutation of the 11,334 3DPW TRAIN positions.
- **E1_3DPW_PLUS_EXTERNAL:** every epoch draws exactly 5,667 rows uniformly from the eligible 3DPW TRAIN rows and 5,667 uniformly from the eligible BEDLAM TRAIN rows (with replacement), then shuffles them.
- **Budget:** identical for both: 2,266,800 draws and 9,000 optimizer steps.
- **Supervised (eligible) draws differ:** E0 has 2,179,400 and E1 has 2,266,800 (+4.0%).
  - The reason: like Worklog 69 C1 and Worklog 71 S0, E0's S0 permutation also visits the 437 all-masked 3DPW TRAIN rows every epoch. Exact C1 reproduction forces this.
  - Worklog 71's eligible-only S1 had the same +4.0% and moved test chain only from .251 to .249.
- **Realized E1 source draws:** 3DPW 1,133,400 and BEDLAM 1,133,400.
- **External rows used:**
  - TRAIN: 21,232 rows (19,261 eligible) from 8,821 sequences, 26 scenes, 159 subjects / 220 bodies;
  - validation: 1,330 rows (1,230 eligible) from 393 sequences, 4 scenes, 39 subjects / 51 bodies.
- **Bank yield:**
  - TRAIN: of 28,348 label-blind selected rows, 25,538 survived. The drops were association with fewer than 6 correct limb joints (2,570), no detection (112) and invalid pelvis (128). The subject split then removed 4,276 rows of validation subjects and 30 rows whose person was not uniquely recoverable.
  - validation: 1,330 of 1,914 selected rows survived. The heavily frame-occluded `zoom_suburb_b` kept 43 of 117.
- **Cache provenance:**
  - 25 of the 26 TRAIN scene parts were reused from per-scene caches written by the previous revision of the builder (before multi-part-tar support); only `closeup_suburb_d` was processed by the committed revision.
  - Two cached scenes rebuilt from scratch with the committed builder (`closeup_suburb_a` with rotation, multi-person `orbit_bigOffice`) reproduce the cached parts exactly: samples, metadata and all arrays are identical (`report_audit.json`).
- **Detection sharing:** no two labelled people in one image were associated with the same detection. TRAIN images carry one selected row each; 259 validation images carry several rows, and none share a detector skeleton.

## E0 reproduction

E0 reproduces Worklog 69 C1 **exactly**: training-curve max difference 0.0, selected epoch 69 = 69, and held-out prediction max difference vs C1 0.0.

**Run note.** Exact reproduction requires the Worklog 69–72 training image (`animcv-framepose:cuda118`).
- A dry run of the same script in the Qwen extraction runtime (torch 2.4.1+cu124) did not reproduce: curve difference 0.0045, selected epoch 59, held-out difference 0.27. The CUDA/torch builds are numerically different.
- The experiment therefore trains in `animcv-framepose:cuda118`, as Worklogs 69–72 did, and only the extraction runs in the Qwen runtime.
- The script refuses to continue unless E0 reproduces.

## Training curves (3DPW TRAIN / 3DPW validation MAE)

| epoch | E0 train | E0 validation | E1 train | E1 validation |
|---:|---:|---:|---:|---:|
| 9 | .1225 | .2115 | .1419 | .2099 |
| 19 | .1007 | .2011 | .1177 | .1918 |
| 29 | .0887 | .2058 | .1078 | .1908 |
| 49 | .0688 | .2002 | .0925 | .1863 |
| 69 | .0563 | **.1969** ← selected | .0820 | **.1831** ← selected |
| 99 | .0460 | .1996 | .0696 | .1880 |
| 149 | .0364 | .2030 | .0595 | .1875 |
| 199 | .0329 | .2041 | .0556 | .1867 |

| candidate | selected epoch | 3DPW train | validation | gap | ratio | final ratio |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 69 | .0563 | .1969 | .141 | 3.50 | 6.20 |
| E1 | 69 | .0820 | .1831 | .101 | 2.23 | 3.36 |

- **E1 is better on validation at every evaluated epoch.** Its validation error stops rising after the optimum: final .1867 vs .2041 for E0.
- **Interpreting the train column:** "train" is 3DPW TRAIN only. E1 fits 3DPW TRAIN less tightly because half of its draws are BEDLAM.
  - Unlike Worklogs 71–72, the smaller gap here comes with **lower** held-out error.
  - About two thirds of the narrower gap (0.026 of 0.040) is the worse 3DPW training fit; the lower held-out error accounts for the remaining 0.014.
  - Measured against E0's gap at selection, E1 closes about 10% (seeds 1338/1339: 8%). At seed 1339 the ratio does not narrow (2.91 → 3.07).

## External-validation diagnostics (secondary; BEDLAM validation scenes × validation subjects)

| segment | rows | \|f\| E0 → E1 (mean, p95) | stable sign accuracy E0 → E1 | magnitude, both sign-correct (rows) |
|---|---:|---|---|---|
| forearm | 1,121 | .359 → **.239** (.976 → .681) | .698 → .860 | .247 → .171 (559) |
| chain | 1,121 | .300 → **.217** (.868 → .633) | .685 → .816 | .246 → .197 (391) |
| upper | 1,230 | .369 → **.231** (1.048 → .641) | .663 → .835 | .236 → .195 (494) |

- **BEDLAM is learnable** for unseen subjects in the four official held-out scene renders.
  - The environments of those renders also appear in TRAIN: `orbit_archVizUI3_time15` matches a TRAIN scene in environment, lighting and camera mode, and suburb and stadium appear with other camera setups. So the backgrounds are not fully unseen.
  - E1's fit on its own TRAIN rows is forearm .186 and chain .172.
- **E0 transfers poorly** to BEDLAM in the other direction (forearm .359).

## 3DPW held-out results (unchanged Worklog 71 rows; primary)

**Test** (6,003 rows). Cells give mean / p50 / p95. "Improved" is the share of rows where E1's error is lower than E0's.

| segment | metric | E0 (= Worklog 69 C1) | E1 | E1 − E0 mean | improved | H0 | D1 |
|---|---|---|---|---:|---:|---:|---:|
| **forearm** | \|f\| | .243/.185/.666 | **.215/.162/.588** | **−0.028** | 55% | .283 | .265 |
| | elevation (°) | 16.2/12.3/44.5 | 14.4/11.3/38.0 | −1.88 | 55% | 18.7 | 17.6 |
| | full direction (°) | 18.1/13.7/47.3 | 16.2/13.0/40.9 | −1.84 | 55% | 20.4 | 19.4 |
| **chain** | \|f\| | .251/.192/.707 | **.200/.152/.558** | **−0.051** | 62% | .271 | .224 |
| | elevation (°) | 15.8/11.8/44.6 | 12.6/9.5/36.1 | −3.13 | 62% | 16.9 | 14.1 |
| | full direction (°) | 16.8/12.7/46.6 | 13.8/10.6/37.5 | −3.00 | 62% | 17.9 | 15.2 |
| upper (secondary) | \|f\| | .272/.213/.740 | .228/.178/.598 | −0.045 | 60% | .237 | .209 |
| | full direction (°) | 17.7/13.9/46.2 | 15.0/11.9/38.3 | −2.72 | 60% | 15.7 | 13.9 |

**Validation** (3,056 rows):

| segment | metric | E0 | E1 | E1 − E0 mean | improved | H0 | D1 |
|---|---|---|---|---:|---:|---:|---:|
| **forearm** | \|f\| | .223/.149/.718 | **.210/.139/.664** | **−0.013** | 53% | .248 | .241 |
| | elevation (°) | 14.4/9.2/48.8 | 13.6/8.9/44.0 | −0.75 | 53% | 15.8 | 15.4 |
| | full direction (°) | 15.8/10.6/51.0 | 15.1/10.2/46.2 | −0.71 | 53% | 17.2 | 16.8 |
| **chain** | \|f\| | .180/.131/.509 | **.167/.118/.491** | **−0.014** | 57% | .201 | .208 |
| | elevation (°) | 11.2/8.0/32.1 | 10.4/7.4/30.7 | −0.82 | 56% | 12.5 | 12.9 |
| | full direction (°) | 12.1/8.7/33.2 | 11.4/8.3/31.3 | −0.75 | 57% | 13.3 | 13.8 |
| upper (secondary) | \|f\| | .180/.142/.480 | .170/.133/.444 | −0.011 | 53% | .196 | .172 |
| | full direction (°) | 12.8/10.1/32.6 | 12.2/9.7/31.9 | −0.60 | 53% | 13.8 | 12.5 |

- **Relative to the other references:**
  - E1 is the first visual candidate in this lineage whose **chain** beats Worklog 65 D1 on test (.200 vs .224). Worklog 70's best was .237. On validation the chain already beat D1 with E0 (.180), and E1 widens that margin (.167 vs D1 .208, H0 .201).
  - Its forearm beats both H0 and D1 on both splits.
  - Upper improves strongly, but on test it still trails D1 (.228 vs .209).
- **Row-level tail:** error p95 drops for every segment and metric on test and on validation. On the clean subset the forearm tail rises slightly (\|f\| p95 .482 → .492, elevation p95 29.3° → 31.1°), and on crosscountry its full-direction p95 rises (40.4° → 44.9°).
  - Improvement is also not row-uniform: the p95 of the per-row difference is +0.21 (forearm) and +0.18 (chain) on test, so about 5% of rows get worse by more than that.
- **Validation vs test:** validation gains have the same sign but are smaller.
  - Forearm: about half the test gain (−0.013 vs −0.028; −5.9% vs −11.7% relative).
  - Chain: about a quarter (−0.014 vs −0.051; −7.7% vs −20.4% relative).
  - Upper: about a quarter as well (−0.011 vs −0.045).

## Sign vs magnitude

| scope / segment | stable sign accuracy E0 → E1 | sign flips (E0-only / E1-only correct) | magnitude error when both sign-correct E0 → E1 (rows; E1 better on) |
|---|---|---|---|
| test forearm | .837 → .861 | 211 / 315 | .214 → **.173** (3,304; 59%) |
| test chain | .786 → .861 | 162 / 450 | .213 → **.177** (2,894; 58%) |
| test upper | .753 → .801 | 147 / 325 | .175 → .151 (2,663) |
| validation forearm | .830 → .854 | 90 / 134 | .208 → .190 (1,492) |
| validation chain | .875 → .881 | 70 / 82 | .169 → .153 (1,515) |

The new supervision improves **both** near/far branch selection (net +104 forearm and +288 chain correct-sign test rows) **and** the continuous magnitude on rows where both candidates already had the right sign.

## Depth regimes (evaluation-only bins on \|GT f\|; \|f\| error E0 → E1)

| scope / segment | near image plane (< .174) | intermediate | strongly out of plane (≥ .5) |
|---|---|---|---|
| test forearm | .166 → .174 (+.008; n 1,802) | .241 → .230 (2,002) | **.309 → .235** (2,199) |
| test chain | .179 → **.139** (2,117) | .239 → .203 (2,453) | **.377 → .284** (1,433) |
| test upper | .250 → .209 | .237 → .201 | .451 → .371 |
| validation forearm | .124 → .122 | .211 → .207 | .372 → .332 |
| validation chain | .130 → .120 | .194 → .189 | .266 → .223 |

- **The historical weakness improves most:** strongly out-of-plane depth.
- **Near-plane false depth:** the chain improves, while the test forearm shows a small near-plane cost (+0.008).

## Clean / crosscountry / other (\|f\|, E0 → E1)

| scope | forearm | chain | upper |
|---|---|---|---|
| clean dancing + hug (validation, 425) | .195 → **.178** (59% improved; p95 .482 → .492) | .183 → **.162** (63%; p95 .486 → .429) | .185 → .166 |
| crosscountry (validation, 133) | .224 → .235 (+.011; p95 .557 → .530) | .221 → .212 | .214 → .208 |
| all other held-out (8,501) | .239 → .215 (p95 .694 → .620) | .229 → .189 (p95 .665 → .542) | .245 → .210 |

- **The clean subset is preserved, and its mean improves** in all three forearm depth bins (.113→.089, .224→.207, .262→.251). Its forearm tail is slightly worse (\|f\| p95 .482 → .492, elevation p95 29.3° → 31.1°).
- **The only mean regression is the crosscountry forearm** (133 rows, which also carry the known GT-camera caveat). Its full-direction p95 also rises (40.4° → 44.9°).

## Per-sequence transfer (E0 − E1 summed \|f\| error; positive = E1 better)

| scope / segment | improved / regressed | net gain | largest single-sequence share of net | worst regressions (sum increase) |
|---|---|---:|---|---|
| test forearm (37) | **31 / 6** | +170.4 | 20.5% (`downtown_bar_00` a0) | `downtown_warmWelcome_00` a0 (+9.5), `downtown_enterShop_00` a0 (+9.3), `downtown_warmWelcome_00` a1 (+1.8) |
| test chain (37) | **33 / 4** | +307.7 | 13.5% (`downtown_bar_00` a0) | `downtown_warmWelcome_00` a0 (+2.8), `downtown_cafe_00` a1 (+1.4), `downtown_weeklyMarket_00` a0 (+0.5) |
| validation forearm (16) | 12 / 4 | +39.8 | 32.8% (`courtyard_drinking_00` a1) | `courtyard_jumpBench_01` (+2.1), `outdoors_crosscountry_00` (+1.5), `courtyard_basketball_01` (+1.4) |
| validation chain (16) | 14 / 2 | +42.1 | 22.8% (`downtown_walkDownhill_00` a0) | `courtyard_drinking_00` a1 (+2.2), `courtyard_jumpBench_01` (+0.5) |
| held-out forearm (53) | 43 / 10 | +210.2 | 16.6% | as above |
| held-out chain (53) | 47 / 6 | +349.9 | 11.9% | as above |
| test upper (37) | 31 / 6 | +268.0 | 10.7% (`downtown_sitOnStairs_00` a0) | `downtown_walking_00` a1 (+5.0), `downtown_downstairs_00` a0 (+4.8), `downtown_walking_00` a0 (+2.7) |
| validation upper (16) | 9 / 7 | +32.7 | 38.5% (`courtyard_drinking_00` a1) | `outdoors_parcours_01` a0 (+3.4), `courtyard_basketball_01` a0 (+1.7), `courtyard_rangeOfMotions_01` a1 (+1.7) |

**No scene owns the primary gains.**
- **On test** they are broadly distributed (largest share 20.5% / 13.5%), and regressions are few and small: the largest is +9.5 against a +170 net.
- **On validation** they are more concentrated. One sequence (`courtyard_drinking_00` a1) supplies 32.8% of the forearm net gain and 38.5% of the secondary upper-arm gain, and the upper arm regresses in 7 of 16 sequences.

## Domain diagnostics (descriptive; nothing reweighted)

| quantity | 3DPW TRAIN (10,897 eligible) | BEDLAM TRAIN (19,261 eligible) |
|---|---|---|
| ineligible TRAIN rows (no segment with both endpoints detector-visible and GT-valid) | 437 / 11,334 (3.9%) | 1,971 / 21,232 (9.3%) |
| detector valid share on eligible rows (shoulder / elbow / wrist) | 1.000 / .995 / .955 | 1.000 / 1.000 / .906 |
| arm detector confidence mean (p05 / p50) | .840 (.647 / .863) | .796 (.290 / .864) |
| valid detector joints per row (mean; p05) | 16.4; 14 | 16.2; 14 |
| crop side / image height (mean; p05–p95) | .63; .39–.85 | .60; .29–.93 |
| crop side in pixels (median; p05–p95) | 1,206; 734–1,596 | 453; 207–1,041 |
| image resolution | 1080×1920 (10,655), 1920×1080 (242) | 1280×720 (15,104), 720×1280 (4,157) |
| target \|f\| mean (forearm / chain / upper) | .333 / .291 / .329 | .414 / .326 / .328 |
| strongly out-of-plane share (forearm / chain / upper) | 27.0% / 18.9% / 21.6% | 36.2% / 23.9% / 23.1% |
| elbow bend, degrees (p05 / p50 / p95) | 73.7 / 140.2 / 168.7 | 54.2 / 132.0 / 153.7 |
| scene families | courtyard 8,812; outdoors 2,085 | suburb 5,951; HDRI 5,093; gym 3,659; archviz 2,001; office 1,471; stadium 1,086 |

| arm-endpoint Qwen readout cosine | mean (p05 / p50 / p95) |
|---|---|
| random pairs within 3DPW TRAIN | .573 (.395 / .576 / .738) |
| random pairs within BEDLAM TRAIN | .533 (.363 / .539 / .677) |
| random cross-source pairs | .519 (.352 / .529 / .652) |
| 3DPW TRAIN row → nearest 3DPW TRAIN row of another sequence | .858 (.756 / .866 / .938) |
| 3DPW validation row → nearest 3DPW TRAIN row | .860 (.752 / .870 / .935) |
| 3DPW validation row → nearest BEDLAM TRAIN row | .755 (.673 / .758 / .830) |
| BEDLAM TRAIN row → nearest 3DPW TRAIN row | .731 (.632 / .737 / .810) |
| BEDLAM validation row → nearest BEDLAM TRAIN row | .805 (.737 / .808 / .863) |

The nearest-row baselines exclude same-sequence neighbours (computed post hoc in `report_audit.json`). Including them, a 3DPW row's nearest neighbour is almost always a temporally adjacent frame (.921), which measures near-duplication, not domain.

- **The domain is reasonably aligned:**
  - crop size relative to image height is similar (.63 vs .60);
  - median arm confidence is similar;
  - cross-source readout similarity (.519) is close to within-source (.533–.573).
- **It is still clearly distinct:**
  - **Readout space:** a BEDLAM row's nearest 3DPW TRAIN neighbour has cosine .73. Across sequences, 3DPW rows sit at .86 from 3DPW TRAIN.
  - **Coverage of 3DPW validation:** 3DPW validation rows are closer to 3DPW TRAIN (.86) than to BEDLAM TRAIN (.76), so BEDLAM does not simply cover the held-out appearance better.
  - **Person scale:** people occupy crops with about 2.3× (mean) to 2.7× (median) shorter sides, i.e. roughly 5–7× fewer source pixels.
  - **Pose:** BEDLAM adds more flexed elbows and more out-of-plane forearms.
- **Detector caveat:** validity and confidence come from different providers (shipped for 3DPW, re-run for BEDLAM). The re-run is more permissive and about 0.03 more confident, and eligibility already filters on arm detection (so the elbow share of 1.000 is structural). Like-for-like, BEDLAM's arm-detection deficit (ineligible 9.3% vs 3.9%, wrist .906 vs .955, confidence p05 .29 vs .65) would be larger, not smaller.
- **Interpretation (descriptive, not causal):** the gain is not explained by BEDLAM resembling the 3DPW held-out images. It is consistent with new visual and pose diversity, although the design cannot separate diversity from the amount of new distinct rows.

## Seed robustness (secondary; not used for the decision)

Two further fixed seeds (1338, 1339) repeat the E0/E1 pair with only the seed changed (initialization and both samplers). Cells give \|f\| E0 → E1, with E1 − E0 and the improved-row share.

| seed | validation MAE at selection E0 → E1 | test forearm | test chain | validation forearm | validation chain |
|---|---|---|---|---|---|
| 1337 (contract run) | .1969 → .1831 | .243 → .215 (−.028; 55%) | .251 → .200 (−.051; 62%) | .223 → .210 (−.013; 53%) | .180 → .167 (−.014; 57%) |
| 1338 | .1966 → .1863 | .243 → .217 (−.027; 56%) | .244 → .206 (−.038; 60%) | .225 → .215 (−.009; 52%) | .181 → .170 (−.011; 55%) |
| 1339 | .1941 → .1840 | .252 → .219 (−.033; 57%) | .240 → .204 (−.035; 60%) | .225 → .209 (−.016; 55%) | .178 → .173 (−.004; 51%) |

- **E1 − E0 is negative for every seed and every segment.**
  - The selection-metric gain (.010–.014) is 3–5× E0's own seed spread (.1941–.1969).
  - The test gains are stable: forearm −.027 to −.033, chain −.035 to −.051.
- **The contract seed is at the favourable end for the chain.** The three-seed means are:
  - test chain −.041 and forearm −.030;
  - validation forearm −.013 and chain −.010.
- **The validation chain gain is the least stable** (−.004 at seed 1339).

## Architecture classification

**CASE A: supervision diversity was a material blocker.**

| CASE A requirement | evidence | met |
|---|---|---|
| E1 materially improves E0 on 3DPW held-out forearm and/or chain | test −0.028 (−11.7%) forearm, −0.051 (−20%) chain; 55% / 62% of rows improved | yes |
| validation improves consistently with test | −0.013 forearm, −0.014 chain; better at every evaluated epoch; same sign for all three seeds | yes (forearm about half the test size, chain about a quarter) |
| p95 does not materially regress | error p95 .666 → .588 / .707 → .558 (test), .718 → .664 / .509 → .491 (validation) | yes |
| clean data preserved | clean forearm .195 → .178, chain .183 → .162 (forearm p95 .482 → .492) | yes (mean improves; forearm tail slightly worse) |
| gains distributed across sequences | 31/37 and 33/37 test sequences improved; largest share 20.5% / 13.5% | yes |
| improvement includes continuous magnitude, not only sign | sign-correct magnitude .214 → .173 / .213 → .177 | yes |

**Not CASE B or C.**
- BEDLAM is learnable (external validation forearm .359 → .239), *and* transfer improves.
- The domain diagnostics show a distinct but not prohibitive gap.
- Transfer is not "near E0 or worse".

**Limits (reported, not repaired):**
- **Small costs:**
  - a small test near-plane forearm cost (+0.008);
  - a crosscountry forearm regression (+0.011, 133 rows);
  - a slightly worse clean forearm tail (p95 +0.010).
- **Row-level tail:** about 5% of rows regress by more than 0.2 in \|f\|.
- **Smaller validation gain:** about half the test gain for the forearm, and about a quarter for the chain and upper arm.
- **The ceiling is not removed:** E1 closes only about 8–10% of E0's train/validation gap at selection.
- **Scope of the evidence:**
  - one external source, which is synthetic (Unreal renders with exact SMPL-X labels), and one fixed 50/50 ratio;
  - real-image diverse supervision is untested;
  - diversity cannot be separated from the amount of new distinct rows.
- **Seed sensitivity:** across three seeds the gain keeps its sign everywhere, but its size varies. The validation chain gain ranges from −.004 to −.014.

**Answer to the completion question.** Yes: with the architecture frozen, adding one substantially more diverse paired RGB↔3D source (BEDLAM) materially improves transfer of continuous visual arm depth to AnimCV's unchanged 3DPW benchmark. Paired-supervision diversity was a material blocker of the Worklog 69–72 ceiling, but it does not remove the ceiling:
- across seeds, E1 lowers the selection MAE by 5–7% and test forearm/chain \|f\| by 11–20%;
- it closes only about 8–10% of E0's train/validation gap.

**Per the decision logic:**
- preserve external paired supervision (this BEDLAM adapter, subject split and 50/50 source-balanced contract) as part of the next multimodal FramePose training contract;
- production integration is not claimed or started, and no further dataset, ratio sweep or architecture change was made.

## Outputs, QE images and tests

**Outputs:** LabServer63 `~/animcv-output/framepose_external_visual_supervision/`:
- **banks:** `bedlam_{train,validation}` and `bedlam_{train,validation}_final` (`.json` + `.npz`), with the build, subject-split and failure reports;
- **images:** the extracted, rotated images used;
- **Qwen caches** (`qwen_train/`, `qwen_validation/`):
  - readout sha256 `8f0e5259…` and `63ae45b7…`;
  - determinism check 0.0;
  - extraction time 2,751 s and 146 s;
- **experiment** (`transfer/`):
  - `report.json` (sha256 `1c05d50a5e6577329f0573786c6927b07268612ec150fbd079a1aa2bc4b03934`);
  - `E0_3DPW_ONLY.pt`, `E1_3DPW_PLUS_EXTERNAL.pt`;
  - `rows_heldout.json`, `seed_check.json`;
- **parity, consistency and review audits:** `openpose_provider_parity_3dpw.json`, `label_consistency/` and `report_audit.json`.
  - `report_audit.json` covers Qwen parity, cross-sequence nearest-neighbour baselines, detection sharing and cache provenance.

**Images for qualitative review:** copied locally to `~/animcv-output/73_external_visual_supervision/`:
- `A_bedlam_train_samples.png` and `A_bedlam_validation_samples.png`: the detector S/E/W input on the 448 px crop, next to a side view of the GT arm;
- `B_3dpw_heldout_e1_improved.png` and `B_3dpw_heldout_e1_regressed.png`: 3DPW rows where E1 most improved or regressed vs E0, with the GT, E0 and E1 forearm/chain directions in side view.

**Tests:** `tests/test_framepose_external_visual_supervision.py` has 14 tests. They cover:
- external bank identity and tamper detection;
- projection correspondence;
- left/right, up/down and near/far canonical conversion, and behind-camera validity;
- the exact left-arm mapping (equal to the 3DPW adapter table);
- the detector owning the input under the 3DPW rules, and GT used only for identity (association);
- deterministic, label-blind row selection;
- scene naming, closeup handling and the official validation scenes;
- the subject split (RenderPeople grouping, about 1/5) and per-person selection;
- sequence-restricted identity recovery (scene-name variants, refit drift, foreign person rejected);
- the exact 50/50 source-balanced sampler and its budget;
- no production import;
- script-level contracts, **checked only by source presence (not behaviourally)**: reprojection abort, rotation, unchanged Worklog 68 extractor, cache / head / overlap / E0-reproduction / budget checks, unchanged 3DPW rows.

**Evidence for the DIRECTION §24 items without a behavioural W73 test.** Each is enforced at runtime and recorded:
- **E0 historical reproduction:** `report.json` `e0_reproduction` gives curve difference 0.0, epoch 69 and prediction difference 0.0.
- **Unchanged 3DPW validation/test:** the Worklog 71 `rows_heldout.json` rows and the FrameBank digest.
- **Qwen model/cache identity and crop-contract parity:** the manifest identity checks and the 32-sample bit-for-bit re-extraction.
- **E0/E1 identical graph and budget:** the `SystemExit` checks; both have 301,955 parameters.
- **Split non-overlap:** the `SystemExit` check, and the split report's `subject_overlap` = 0.
- **Identical initialization and deterministic evaluation:** covered by the unchanged Worklog 69 test `test_identical_architecture_initialization_tanh_range_and_deterministic_inference`. `train()` seeds before building the model.

**Results:**
- 47 passed in the training image, with the unchanged Worklog 68–72 tests.
- Locally: 13 passed, 1 skipped (torch absent).

No production file changed, so the full regression was not run.
