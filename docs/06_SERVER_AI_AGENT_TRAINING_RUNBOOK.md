# Server AI Agent Training Runbook

## Objective

Run the licensed own-data 2D→3D training pipeline autonomously, but never
promote a model to the FBX production path before all stated quality gates pass.
The detailed Korean policy is [05_자체_학습_전환_계획](05_자체_학습_전환_계획.md).

## Preconditions

- Run from the repository root. Install training dependencies with
  `pip install -e '.[training]'`.
- On a CUDA server, install the matching CUDA PyTorch build and use `--device cuda`.
  Confirm with `python -c "import torch; print(torch.cuda.is_available())"`.
- Keep source videos, calibration files, checkpoints, and reports in protected
  external storage. Do not commit raw footage or personal data.
- Use only data, models, and generated artifacts with documented commercial rights.

## Required Input Per Take

```text
/data/animcv/take_001/
  calibration.json
  front_pose.json
  side_pose.json
  rear_pose.json                 # optional; three cameras are preferred
  manifests/source_manifest.json # rights, performer, action, take, sync metadata
```

Each pose file must be an AnimCV canonical `PoseSequence` for the same tracked
person. Camera pose sequences must share `frame_index` values and be time-synchronised.
Use a separate performer, recording session, or take for holdout; never split
train and holdout randomly by frame.

`calibration.json` must use `animcv_multiview_calibration_v1`, metric
`world_units: "metres"`, camera intrinsics, and a 4×4 **world→OpenCV-camera**
`world_to_camera` matrix. OpenCV axes are +X right, +Y down, +Z forward.

```json
{
  "schema": "animcv_multiview_calibration_v1",
  "world_units": "metres",
  "cameras": {
    "front": {
      "intrinsics": {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0},
      "world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
  }
}
```

## Procedure

### 1. Validate and triangulate GT

Confirm every camera name in `--observations` exists in calibration. Confirm
calibration was performed after the final camera placement and record any sync
offset. Then run:

```bash
motion-tool triangulate-supervised-3d-ground-truth \
  --observations front=/data/animcv/take_001/front_pose.json,side=/data/animcv/take_001/side_pose.json,rear=/data/animcv/take_001/rear_pose.json \
  --calibration /data/animcv/take_001/calibration.json --reference-camera front \
  --out /data/animcv/take_001/front_root_relative_gt.json \
  --report-out /data/animcv/take_001/triangulation_report.json \
  --min-confidence 0.3 --max-reprojection-error-pixels 10
```

Only accept a take when its report has `passed: true`, `coverage >= 0.95`, and
`p95_reprojection_error_pixels <= 10`. Otherwise quarantine the take and report
the failure. Do not train on it. Low coverage commonly indicates occlusion or
tracking/sync failure; high reprojection error commonly indicates calibration,
lens-distortion, or alignment error.

### 2. Build per-take datasets

Use the reference camera's 2D pose and its triangulated GT.

```bash
motion-tool build-supervised-3d-dataset \
  --pose /data/animcv/take_001/front_pose.json \
  --ground-truth /data/animcv/take_001/front_root_relative_gt.json \
  --image-width 1920 --image-height 1080 --sequence-id take_001 \
  --out /data/animcv/derived/train/take_001_dataset.json
```

Before training, combine only accepted train-take JSON datasets. Validate the
same schema, joint order, and image size; write a dataset manifest containing
take IDs, frame counts, source hashes, and triangulation reports. Build holdout
takes separately and never include them in the training combination.

### 3. Train

Start from the reproducible baseline below. If memory is insufficient, reduce
only batch size; do not weaken validation or holdout rules.

```bash
motion-tool train-supervised-3d-lifter \
  --dataset /data/animcv/derived/train/combined_dataset.json \
  --out /data/animcv/models/lifter_run_001.pth \
  --window 81 --channels 256 --epochs 30 --batch-size 128 \
  --learning-rate 0.001 --device cuda \
  --report-out /data/animcv/models/lifter_run_001_training_report.json
```

Record git SHA, timestamp, hostname/GPU, PyTorch version, random seed, config,
train/holdout take lists, dataset-manifest hash, checkpoint, and reports.

### 4. Evaluate holdout

```bash
motion-tool evaluate-supervised-3d-lifter \
  --dataset /data/animcv/derived/holdout/combined_dataset.json \
  --checkpoint /data/animcv/models/lifter_run_001.pth --device cuda \
  --out /data/animcv/models/lifter_run_001_holdout_report.json
```

This output's MPJPE and P95 are informational baseline metrics. Product
promotion additionally requires independent pose/root audit and Blender visual
acceptance with all of the following gates:

| Gate | Requirement |
| --- | ---: |
| Tracker success | ≥95% |
| PA-MPJPE | ≤80 mm |
| Root yaw MAE | ≤15° |
| Yaw P95 | ≤30° |
| Unambiguous knee/elbow flip | 0 |

## Non-negotiable Rules

- Never automatically adopt a public research pretrained checkpoint for commercial output.
- Never use a failed triangulation take or a failed checkpoint for production FBX output.
- Never guess coordinate axes, metre scale, calibration direction, or reference camera.
- On any gate failure, preserve the report and diagnose the affected take/action/joint;
  improve data or model, then rerun the complete holdout evaluation.
