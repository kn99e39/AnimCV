# Worklog — Existing-Dataset Inventory for Domain-Diverse Arm Supervision (2026-10-15)

**Question.** Can an already downloaded, previously unused paired RGB↔3D dataset extend the J1/R1 training supervision, so that adding the same new data to both architectures separates *training-domain coverage* (Hypothesis A) from the *frozen-anchor limitation* (Hypothesis B)?

**Answer.** No. **CASE D: data contract unavailable. STOP after the inventory.**

- **The inventory of LabServer63 finds no eligible source.** Every human-pose asset on the server falls into one of these groups:
  - already used for exactly this training purpose (3DPW, BEDLAM);
  - reserved for evaluation (the MPI-INF-3DHP official test);
  - annotation-only, with no on-disk RGB (MPI-INF-3DHP training set S1–S8, AMASS, AGORA labels);
  - absent (Human3.6M: empty placeholder directories).
- **Nothing was trained or changed:** no training population was built, no checkpoint trained and no historical artifact modified.
- **The question stays open:** the Hypothesis A vs B question remains unanswered. Answering it needs a dataset download, which this batch was not authorized to make. The candidates are listed at the end.

## Starting state and files

- **Start:** `arch/single_frame_first` at `68b99a35e2de56fa79773f69462b677b499b3bb4` (the Worklog 83 docs). Local, origin and LabServer63 were equal, and the local worktree was clean. The server's pre-existing untracked `.DS_Store`, `.animcv_sync_stage/` and `docker/` predate this batch and were not touched.
- **Files added (diagnostic only, not imported by production):**
  - `scripts/inventory_existing_pose_datasets.py`: an offline, read-only, standard-library-only inventory with a declared dataset table and the classification rule.
  - `tests/test_framepose_dataset_inventory.py`
  - this worklog
- **Inventory artifact (LabServer63):** `~/animcv-output/dataset_inventory_w84/inventory.json`, sha256 `3ed680c8…`. It was produced with the host `python3 -I`, with no network access and no download.
- **Historical artifacts were re-hashed and are unchanged:**
  - J0 `5332b691…`, J1 `25f0c60f…`, R0 `2f015d67…`, R1 `87377773…`;
  - Worklog 80 predictions `eaf3f57c…` and Worklog 82 predictions `a8bfbaea…`;
  - Worklog 83 report `466ba37e…` and 3DHP predictions `48a0c845…`.

## How the inventory was done

- **Breadth of the search:**
  - every dataset directory under `~/animcv-data`;
  - the repository's `datasets/` directory and `~/animcv-output/data`;
  - a bounded search of `/home/nd` by name, for 3dhp, h36m/human36, annot.mat, agora, emdb, rich, humman, aist, fit3d, mpii, coco, totalcapture, panoptic, hi4d, egobody, mupots and 3doh;
  - every video or archive larger than 50 MB under `/home/nd`.
- **Other mounts:** `/data`, `/mnt` and `/srv` do not exist or are empty, and `/media` holds only `root`. The only other filesystems are system volumes (`/`, `/usr`, `/var/lib`, `/boot`).
- **Classification rule.** `classify` applies these rules in order, and the first match wins:
  1. absent;
  2. evaluation-reserved;
  3. already used for the same purpose;
  4. annotation-only (no on-disk RGB);
  5. RGB without camera-space 3D;
  6. eligible.
- **Counts come from disk.** Images, videos, archives, 3D label files and calibration files are counted by walking the disk, not taken from documentation.

## Inventory

| dataset (version) | location | modalities on disk | camera-space 3D / calibration | identities, ordering, splits | prior use | classification |
|---|---|---|---|---|---|---|
| **3DPW** (official release; `sequenceFiles` train / validation / test) | `~/animcv-data/datasets/3dpw` (`3dpw/raw` links here) | RGB: 53,796 JPEG frames in 61 sequence directories; SMPL-24 3D and camera poses in 60 `.pkl` files | yes / yes (per-sequence intrinsics and per-frame extrinsics in the pkl) | official splits: train 34 seq / 22,646 frames, validation 16 / 10,206, test 37 / 35,310; video-ordered at 30 fps | **FrameBank since Worklog 69** (21,817 rows: train 11,334, validation 3,407, test 7,076). Train trains J1/R1; validation selects checkpoints; test has been the development evaluation since Worklog 73 | already used, same purpose |
| **BEDLAM** v1 (`all_npz_12_training` processed labels, neutral SMPL-X GT, 6/30 fps image tars) | `~/animcv-data/bedlam` (709 GB) | RGB: 27 training-scene image tars (24 at 6 fps, 3 at 30 fps; 3 validation-scene placeholders of 22 bytes) and 20 validation PNG tars (4 scenes × 5 parts); 26 training label npz; 30 per-scene GT tars | yes / yes (`cam_int`, `cam_ext`; reprojection verified in Worklog 73) | 26 training scenes, 951,066 label rows; 4 official validation scenes, 100,124 rows; body and RenderPeople identities; per-sequence frame order | **Worklog 73 external banks:** train 21,232 rows from all 26 training scenes; validation 1,330 rows from the 4 validation scenes and validation subjects. Used by Worklogs 73, 78–82 | already used, same purpose |
| **AGORA** labels (`agora-bfh.npz`, `agora-body.npz`, bundled with the BEDLAM labels) | `~/animcv-data/bedlam/labels/all_npz_12_training/` | 2 label files; **no AGORA images** | labels only | — | never used (the Worklog 73 builder skips `agora*`) | annotation-only |
| **MPI-INF-3DHP official TEST** (`mpi_inf_3dhp_test_set.zip`, 2021-07-11, sha256 `eab3d09f…`) | `~/animcv-data/mpi_inf_3dhp_test/` | RGB: 24,891 JPEG frames (TS1–TS6); 6 `annot_data.mat`; 2 calibration files | yes / yes (pinhole verified per sequence in Worklog 83) | 6 subjects / sequences; 2,875 official valid frames; video-ordered | **Worklog 83 independent evaluation.** It has now informed an architecture decision. | evaluation-reserved |
| **MPI-INF-3DHP TRAIN** S1–S8 × Seq1–2 (annotations fetched 2026-08-04) | `AnimCV/datasets/mpi_inf_3dhp` (git part files, reassembled) and `~/animcv-output/data/mpi_inf_3dhp` | 16 `annot.mat` (28 joints, 14 cameras, 2D + 3D) and 16 `camera.calibration`; **no `imageSequence` videos** — the intake manifest states that "only annot.mat + camera.calibration were fetched" | yes / yes | 8 subjects × 2 sequences, multi-camera studio, frame-ordered | legacy annotation-only temporal lifters (docs 04, 06, 09, 10; S7/S8 held out) | **annotation-only** |
| **AMASS** (Hugging Face mirror; 15 sub-datasets, e.g. ACCAD, BMLmovi, CMU, KIT, TotalCapture, HumanEva) | `~/animcv-data/amass/raw/amass_hf/raw` (18 GB) | 10,820 SMPL-H motion npz; no RGB, no camera | no / no | per-subject motion clips | legacy virtual-camera lifter training (docs 09, 10) | annotation-only |
| **Human3.6M** | `~/animcv-data/human36m/{raw,prepared}` | empty directories | — | — | never obtained (doc 08: licence blocked) | absent |
| SMPL / SMPL-H / SMPL-X body models | `~/animcv-data/body_models`, `~/animcv-data/bedlam/body_models` | model files | — | — | target construction only | not a dataset |

**Other project directories** on the server (NeuralShaderDemo, OSN-GS, VisModeling, W-VEG, TransformerLens) contain no human-pose data. The only image or video archives larger than 50 MB are BEDLAM's, the 3DHP test zip and graphics scenes.

**Potentially eligible observations for NEW paired RGB↔3D arm supervision: 0.**

## Selection outcome and rejected candidates

No source meets the predeclared criteria: paired RGB with camera-space 3D, a recoverable anatomical left arm, reliable RGB/annotation identity, compatibility with the detector and the Qwen contract, and no contamination of evaluation subjects.

| candidate | why rejected |
|---|---|
| MPI-INF-3DHP TRAIN (S1–S8) | **No RGB on disk.** Only `annot.mat` and calibration were ever fetched. Fetching the videos is a new download (official `get_dataset.sh`, about 25 GB for the VNect cameras), which this batch was not authorized to make. |
| More BEDLAM frames | All 26 training scenes and 4 validation scenes are already in use; only 3 rows per sequence were sampled, so about 97% of labelled frames are unused. Adding them would be the **same source, bodies, scenes and renderer, for the exact same training purpose**: more samples, not domain diversity. The directive forbids that substitution. Rows of BEDLAM validation subjects inside training scenes would also contaminate BEDLAM validation. |
| 3DPW (any further rows) | Every official split is already used: train for training, validation for selection, test for development evaluation. |
| MPI-INF-3DHP official TEST | Evaluation-only by directive and by the Worklog 83 contract. |
| AMASS, AGORA labels | No RGB, so they cannot provide paired RGB↔3D supervision. Rendering AMASS would be a new synthetic RGB source and a different intervention. |
| Human3.6M | Not on the server; its licence requires owner review (doc 08). |

## What was not done (by the stop rule)

- **No split manifests, training populations, source quotas or checkpoints were created.** J1_EXPANDED_DATA and R1_EXPANDED_DATA do not exist.
- **The historical reproduction gate was not rerun.** Worklog 83 last ran it on the same artifacts, and their hashes are unchanged.
- **No 3DPW or 3DHP evaluation was rerun,** and there are no new review exports.
- **No production, IK, Root Orientation or BaseRig change was made.**

## Architecture classification — CASE D (data contract unavailable)

- **Blocker.** LabServer63 holds no unused paired RGB↔3D source. This was verified by the on-disk inventory and by the intake manifest of the only unused dataset family with camera-space 3D (MPI-INF-3DHP TRAIN, annotations only).
- **Remaining architectural uncertainty.** Worklog 83's transfer failure is still compatible with both hypotheses.
  - **Hypothesis A (coverage).** R1's correction head saw only 3DPW and BEDLAM appearance, viewpoints and poses, while J1 had the same data. The data alone cannot separate them, because both architectures were equally exposed.
  - **Hypothesis B (anchor).** Worklog 83 supports it indirectly:
    - J0's error grows much more under the shift than J1's: J1 − J0 wrist −4.83 %L on 3DHP vs −1.48 on 3DPW.
    - R1's repair of gross anchor errors stays the same size: −12.6 vs −12.8 %L.
  - These facts fit an anchor-bound limit, but they do not prove the anchor is the sole cause.

**Candidates for an authorized follow-up** (the user decides; nothing was downloaded):

1. **MPI-INF-3DHP TRAIN RGB videos** (S1–S8; official `get_dataset.sh`; existing adapter, calibration and the Worklog 83 camera contract).
   - It is the most technically ready option.
   - It is the same capture family as the evaluation source: the same studio as TS1–TS4, and subject overlap with TS1–TS6 cannot be established from the shipped files.
   - Training on it would turn the Worklog 83 test into in-family evaluation, so a generalization claim would need another evaluation source.
2. **A source from a new family with real RGB and camera-space 3D** (for example Human3.6M, RICH or EMDB). Each needs a licence review and a download.
3. **Rendered AMASS.** This is a synthetic RGB intervention, which is a different experiment from paired real supervision.

Proceeding further requires an explicit authorization.

## Focused tests

`tests/test_framepose_dataset_inventory.py` (4 tests) covers:
- the order and strictness of the classification rules (annotation-only is never paired; evaluation reservation and prior use win);
- declared dataset roles matching the worklog history;
- a synthetic-tree walk with an end-to-end report;
- the static guarantee of no network, no subprocess, no credential access and no deletion.

Results:
- `pytest tests/test_framepose_*.py`: **193 passed, 14 skipped** locally (the skips need torch);
- the same set on LabServer63 in `animcv-framepose:cuda118`: **207 passed**.

## Generalization limitations

- **Evaluation sources are unchanged.** The 3DPW test remains research-development evidence. The MPI-INF-3DHP official test has informed the Worklog 83 decision and can no longer be called untouched for future architectural selection. No unseen evaluation partition exists on the server.
- **Missing assets are not judged.** The inventory covers what is on LabServer63 today. It does not assess the suitability of datasets that are not downloaded beyond the notes above.
