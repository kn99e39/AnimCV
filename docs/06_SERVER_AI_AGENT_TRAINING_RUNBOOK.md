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
loss/metric masking, clip-safe temporal windows, train/holdout split checks,
and source-neutral MPJPE/PA-MPJPE/yaw evaluation. Human3.6M, AMASS, and 3DPW
adapters remain future extensions; they are not required for the first server
training run.

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
4. Train on the RTX 3090 using `--device cuda`; start with window 81, channels
   256, epochs 30, batch size 128, learning rate 0.001. The RTX 3080 Ti may run
   independent evaluation or data preparation.
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
