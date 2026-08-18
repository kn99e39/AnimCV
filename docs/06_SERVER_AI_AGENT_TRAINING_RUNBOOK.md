# Server AI Agent Training Runbook

## Purpose

Operate AnimCV's web-dataset ingestion, training, and evaluation workflow.
The source of training data is a web dataset, not a new physical camera-capture
project. Record the source's access conditions, version, and provenance.

Read [08_RESEARCH_DATASET_ASSESSMENT](08_RESEARCH_DATASET_ASSESSMENT.md), then
[07_WEB_DATASET_TRAINING_PLAN](07_WEB_DATASET_TRAINING_PLAN.md), before executing commands.

## Rules

- Use official downloads where available and record version, checksum, and access conditions.
- Keep each source dataset's official split intact unless its terms explicitly
  allow a documented alternative. Never randomly split frames from a sequence.
- Do not commit or publish licensed source data, credentials, raw imagery, or
  checkpoints without explicit authorization.
- Do not present a checkpoint as quality-validated unless every quality gate
  passes on an independent holdout source/sequence.

## Current Implementation Boundary

The current code has an executable MPI-INF-3DHP ingestion path, invalid-joint
loss/metric masking, clip-safe temporal windows, GPU-resident vectorized
batching, CUDA AMP, two-GPU DDP training, train/holdout split checks, and
source-neutral MPJPE/PA-MPJPE/yaw evaluation. 3DPW has an executable
official-label importer and AMASS has an executable SMPL+H virtual-camera
adapter plus a restartable bounded-corpus preparer. Human3.6M is intentionally
outside this execution path.

The current executable research-source ingestion path is MPI-INF-3DHP. It
imports official paired 2D/3D annotations directly into a supervised clip:

```bash
motion-tool import-mpi3dhp-supervised-dataset \
  --annotation /data/mpi3dhp/S1/Seq1/annot.mat --camera-index 0 \
  --image-width 2048 --image-height 2048 \
  --sequence-id mpi_s1_seq1_cam0_train_a --split train \
  --start-frame 0 --end-frame 999 --out /data/animcv/train_a.json
motion-tool combine-supervised-3d-datasets \
  --datasets /data/animcv/train_a.json,/data/animcv/train_b.json \
  --expected-split train --out /data/animcv/train_combined.json
```

Build holdout clips in separate files with `--split holdout`, combine them
with `--expected-split holdout`, and never place them in the train command.

### AMASS raw motion acquisition

AMASS is the synthetic-motion branch. `src/pose/amass_adapter.py` evaluates
raw SMPL+H parameters, converts the first 24 SMPL joints to AnimCV's canonical
17-joint contract, and projects deterministic GT 2D through a virtual camera.
The repository includes a restart-safe downloader for the public AMASS mirror. It
uses only the Python standard library, keeps the canonical `raw/<subset>/...`
layout, downloads only `.npz` motion parameters, and atomically verifies each
file size. It does not download videos, renders, or SMPL-X variants.

```bash
python scripts/download_amass_hf.py \
  --out /home/nd/animcv-data/amass/raw/amass_hf \
  --workers 24 --retries 3 \
  --subsets ACCAD,BMLmovi,BMLhandball,CMU,EKUT,Eyes_Japan_Dataset,KIT,TCD_handMocap,TotalCapture
```

Run it again with `MPI_HDM05,SFU,MPI_mosh,HumanEva` for validation and
`SSM_synced,Transitions_mocap` for an AMASS-internal holdout. The public
mirror currently has no `BMLrub` or `PosePrior` directory; do not pass those
names to the downloader. Re-running any command skips
already verified files and resumes the remaining subset files.

Raw AMASS NPZ files alone are not trainable samples: producing paired 2D/3D
supervision also requires a compatible SMPL+H body-model file. The server body-model cache uses
`/data/body_models/smplh/SMPLH_{MALE,FEMALE,NEUTRAL}.pkl`; keep the raw files under
`/data/amass/raw/amass_hf` and write any generated samples only under
`/data/amass/prepared`; do not mix them with MPI or 3DPW splits.

After raw acquisition completes, create a bounded initial synthetic corpus:

```bash
docker run --rm --gpus all --entrypoint python3 -w /workspace -e PYTHONPATH=/workspace/src \
  -v /home/nd/AnimCV:/workspace:ro -v /home/nd/animcv-data:/data \
  animcv-train:cuda118 scripts/prepare_amass.py \
  --raw /data/amass/raw/amass_hf --out /data/amass/prepared --body-model-root /data/body_models \
  --max-frames-per-clip 120 --train-clips 1000 --validation-clips 100 --holdout-clips 100 \
  --camera-views '0,0,4.5,1500;-45,10,5.0,1300;45,-8,4.0,1750' \
  --device cuda
```

This emits up to 144,000 synthetic 30-FPS frames in separate sequence-safe
artifacts. AMASS uses `HumanEva`/`SFU` for validation and
`SSM_synced`/`Transitions_mocap` for holdout; all other installed subsets feed
the initial synthetic training split. The preparer rejects auxiliary archives
such as per-subject `shape.npz`, records exclusion counts, uses root-relative
source paths for collision-free sequence IDs, and repairs IDs in older cached
clips without recomputing SMPL+H joints.

`--camera-views` is an explicit semicolon-separated list of
`yaw_degrees,pitch_degrees,distance_meters,focal_length` records. It is not a
Cartesian product: every selected source gets exactly one clip per listed
view, preserving the source-motion split while varying camera geometry.

### Reproducible source-mixing experiment matrix

`scripts/run_lifter_experiments.py` materializes each combined dataset and
then evaluates five candidates on the same validation and independent
holdouts: MPI-only, MPI+3DPW, direct mix, AMASS-only pretraining, and
AMASS-pretrain→MPI+3DPW fine-tuning. Its input augmentation is on-the-fly,
deterministic per epoch, and affects only 2D input observations: normalized
coordinate jitter, observed-joint dropout, and confidence jitter. 3D targets
and holdout inputs are never augmented.

```bash
docker run --rm --gpus all --entrypoint python3 -w /workspace -e PYTHONPATH=/workspace/src \
  -v /home/nd/AnimCV:/workspace:ro -v /home/nd/animcv-data:/data:ro \
  -v /home/nd/animcv-output:/output animcv-train:cuda118 \
  scripts/run_lifter_experiments.py \
  --mpi-train /output/data/animcv/train_combined.json \
  --three-dpw-train /data/3dpw/prepared/train.json \
  --amass-train /data/amass/prepared_aug_v1/train.json \
  --validation /data/3dpw/prepared/validation.json,/data/amass/prepared_aug_v1/validation.json \
  --three-dpw-holdout /data/3dpw/prepared/holdout.json \
  --amass-holdout /data/amass/prepared_aug_v1/holdout.json \
  --out /output/experiments/lifter_matrix_aug_v1 --epochs 30 \
  --input-jitter-std 0.015 --input-dropout-probability 0.05 --confidence-jitter-std 0.08
```

### 3DPW official-label preparation

3DPW supplies synchronized 2D detections, SMPL-24 joint positions, camera
extrinsics, and per-frame camera validity. The importer converts those labels
to AnimCV's canonical 17 joints, applies world-to-camera extrinsics, converts
OpenCV camera axes to AnimCV axes, and root-relativizes in metres. It is a
paired-label source, so no SMPL body model is required for this conversion.

Prepare every official split without changing the raw archive:

```bash
docker run --rm --entrypoint python3 -w /workspace -e PYTHONPATH=/workspace/src \
  -v /home/nd/AnimCV:/workspace:ro -v /home/nd/animcv-data:/data \
  animcv-train:cuda118 scripts/prepare_3dpw.py \
  --raw /data/3dpw/raw/DATASET_Motion --out /data/3dpw/prepared
```

The script writes restartable per-source JSON files plus `train.json`,
`validation.json`, and `holdout.json`. The official 3DPW `test` split maps to
AnimCV's `holdout` split and must not be combined into training data.

### Detector-input evaluation runtime

The training image intentionally excludes MMPose/MMDetection. Build the
separate `Dockerfile.pose` image before evaluating real 3DPW RGB frames; it
pins the prebuilt CUDA 11.8 / PyTorch 2.1-compatible MMCV wheel together with
MMPose 1.3.2 and MMDetection 3.3.0. This keeps detector dependencies out of
the training image and makes the detector-input gate reproducible.

```bash
docker build -f Dockerfile.pose -t animcv-pose:cuda118 .
docker run --rm --gpus all --entrypoint python3 -w /workspace \
  -e PYTHONPATH=/workspace/src \
  -v /home/nd/AnimCV:/workspace:ro -v /home/nd/animcv-data:/data:ro \
  -v /home/nd/animcv-output:/output animcv-pose:cuda118 \
  -m app.cli estimate-pose --frames /data/3dpw/imageFiles/<sequence> \
  --out /output/detector_eval/<sequence>_pose.json --device cuda
```

The first command retrieves package wheels; the first inference retrieves the
official RTMDet-tiny and RTMPose-tiny checkpoints into the container user's
model cache. Do not pass `--evaluation-ground-truth` to this production-input
gate: that option supplies official-label boxes and is diagnostic only.

### Verified three-source server corpus (2026-08-16)

The LabServer63 run completed and fully loaded these artifacts:

| Artifact | Sequences | Frames | Integrity |
| --- | ---: | ---: | --- |
| MPI-INF-3DHP train | 12 | 106,512 | schema v2 |
| 3DPW train | 34 | 22,646 | official train split |
| AMASS train | 1,000 | 112,898 | 0 duplicate IDs, 0 non-finite frames |
| Three-source train | 1,046 | 242,056 | 0 duplicate IDs |
| 3DPW + AMASS validation | 88 | 18,612 | 0 duplicate IDs |
| 3DPW test holdout | 37 | 35,310 | never used for training |
| AMASS internal holdout | 100 | 10,792 | never used for training |

AMASS raw intake contained 10,820 NPZ files across 15 subsets. Exactly 10,718
were motion archives and 102 were auxiliary `shape.npz` files. The preparation
report records this distinction. CUDA preflight passed on an RTX 3080 Ti with
PyTorch 2.1.2+cu118. A full-corpus one-epoch execution with window 81, channels
256, batch 128, and AMP processed 242,056 samples in 17.58 seconds at 13,766
samples/s with 638.9 MiB peak allocated VRAM. These numbers prove operational
readiness, not final model quality; run the planned multi-epoch ablations and
independent holdout gates before selecting a checkpoint.

## Verified Server Execution Chain

### Docker setup

The repository includes `Dockerfile.train` and `compose.train.yaml`. On the
server, select host paths for source data and writable outputs, then build and
run CUDA preflight. It reuses the server's CUDA 11.8 PyTorch base image. The
container mounts source data read-only at `/data` and writes artifacts to
`/output`.

```bash
export ANIMCV_DATA_ROOT=/path/to/research-datasets
export ANIMCV_OUTPUT_ROOT=/path/to/animcv-output
mkdir -p "$ANIMCV_OUTPUT_ROOT"
# LabServer's Docker build DNS is unreliable on its bridge network, so use
# host networking only for the image build.
docker build --network host -f Dockerfile.train -t animcv-train:cuda118 .
docker compose -f compose.train.yaml run --rm train \
  preflight-training --device cuda --out /output/preflight.json
```

Point the following commands at an installed MPI-INF-3DHP copy. Use separate
source sequences or subjects for train and holdout when available; the short
frame ranges below are only smoke-test examples.

```bash
docker compose -f compose.train.yaml run --rm train train-supervised-3d-lifter \
  --dataset /data/animcv/train_combined.json \
  --out /data/animcv/models/mpi_baseline.pth \
  --window 81 --channels 256 --epochs 30 --batch-size 128 \
  --learning-rate 0.001 --device cuda \
  --report-out /data/animcv/models/mpi_baseline_train.json

# Build the holdout pose/GT pair with the existing importer, then infer.
motion-tool import-mpi3dhp-ground-truth \
  --annotation /data/mpi3dhp/S1/Seq1/annot.mat \
  --calibration /data/mpi3dhp/S1/Seq1/camera.calibration --camera-index 0 \
  --pose-out /data/animcv/holdout_pose.json \
  --lifted-out /data/animcv/holdout_gt.json \
  --calibration-out /data/animcv/holdout_camera.json
motion-tool lift-supervised-3d \
  --pose /data/animcv/holdout_pose.json --checkpoint /data/animcv/models/mpi_baseline.pth \
  --image-width 2048 --image-height 2048 --device cuda \
  --out /data/animcv/holdout_prediction.json
motion-tool estimate-root-motion --lifted-pose /data/animcv/holdout_prediction.json \
  --out /data/animcv/holdout_prediction_root.json
motion-tool estimate-root-motion --lifted-pose /data/animcv/holdout_gt.json \
  --out /data/animcv/holdout_gt_root.json
motion-tool audit-supervised-3d \
  --predicted /data/animcv/holdout_prediction.json --ground-truth /data/animcv/holdout_gt.json \
  --predicted-root /data/animcv/holdout_prediction_root.json \
  --ground-truth-root /data/animcv/holdout_gt_root.json \
  --out /data/animcv/models/mpi_baseline_holdout.json
```

## Agent Procedure Once C1–C6 Are Complete

1. Validate the source intake manifest: URL, version, SHA-256, access conditions,
   source coordinate system, joint schema, units,
   and official train/validation/test split.
2. Run the approved source adapter to create canonical 2D/3D paired artifacts.
   Reject any coordinate, scale, camera-frame, or joint-mapping ambiguity.
3. Build train and holdout datasets by official source split, sequence, subject,
   and action-clip boundaries. Record every included sequence ID and frame count.
4. The current server is a single RTX 3080 Ti with 12 GB VRAM. Train with the
   documented single-process command and `--device cuda`; it still uses
   GPU-resident vectorized temporal windows and CUDA AMP. Start with window
   81, channels 256, epochs 30, batch size 128, learning rate 0.001. If CUDA
   reports out-of-memory, retry with batch size 64; do not silently alter the
   model or evaluation split. The training report records `parallelism`,
   elapsed time, global samples/sec, and peak GPU memory.

   DDP remains an optional future path for a multi-GPU server: invoke
   `torchrun --standalone --nproc_per_node=2 -m app.cli` and add
   `--distributed`; halve the per-GPU batch size to preserve the global batch.
5. Preserve git SHA, hardware, PyTorch version, random seed, config, source
   manifests, dataset manifests, checkpoint, training report, and holdout report.
6. Evaluate holdout quality. Product promotion requires tracker success ≥95%,
   PA-MPJPE ≤80 mm, root yaw MAE ≤15°, yaw P95 ≤30°, and zero unambiguous
   knee/elbow flips, followed by Blender visual acceptance.

## Required Agent Report

For every material phase, return:

| Section | Required content |
| --- | --- |
| Phase | Command or transformation run |
| Inputs | Source/version/license state and non-sensitive artifact paths |
| Metrics | Frame/sequence counts and all available quality metrics |
| Verdict | PASS / FAIL / BLOCKED with exact gate comparison |
| Diagnosis | Measured cause, clearly separated from hypotheses |
| Next action | One safe concrete next task |
