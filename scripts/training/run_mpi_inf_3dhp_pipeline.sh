#!/bin/bash
# AnimCV MPI-INF-3DHP supervised lifter training pipeline (docs/06 runbook).
# Train: S1-S6, camera 0, both sequences. Holdout: S7-S8, camera 0, both sequences.
#
# Raw data lives in datasets/mpi_inf_3dhp/ as <100MB annot.mat.part-* chunks
# (committed as plain git objects, no Git LFS) and is reassembled on demand.
# Large, fully regenerable intermediates (per-clip import JSON, the combined
# training set, and holdout pose/prediction/root-motion JSON) are written
# under training_cache/ (gitignored, not committed) so re-running this script
# never needs to touch git. Only the checkpoint, training report, holdout
# gate report, and small per-clip audit reports land in runs/ and are
# committed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="$REPO_ROOT/datasets/mpi_inf_3dhp"
bash "$REPO_ROOT/scripts/training/reassemble_mpi_inf_3dhp.sh"
WORK="$REPO_ROOT/training_cache/mpi_s1s6_cam0"
RUN_OUT="$REPO_ROOT/runs/mpi_s1s6_cam0"
CAM=0
mkdir -p "$WORK/train_clips" "$WORK/holdout" "$RUN_OUT/holdout"

image_size() {
  # Prints "<width> <height>" for the given calibration file's camera $CAM.
  python3 - "$1" "$CAM" <<'PY'
import re, sys
path, cam = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
blocks = re.split(r"(?=^name\s+\d+\s*$)", text, flags=re.MULTILINE)
block = next(b for b in blocks if re.search(rf"^name\s+{cam}\s*$", b, re.MULTILINE))
size = re.search(r"^\s*size\s+(.+)$", block, re.MULTILINE).group(1).split()
print(int(float(size[0])), int(float(size[1])))
PY
}

TRAIN_SUBJECTS=(1 2 3 4 5 6)
HOLDOUT_SUBJECTS=(7 8)

echo "=== [1/5] import train clips ==="
TRAIN_JSONS=()
for s in "${TRAIN_SUBJECTS[@]}"; do
  for seq in 1 2; do
    ann="$DATA_ROOT/S${s}/Seq${seq}/annot.mat"
    cal="$DATA_ROOT/S${s}/Seq${seq}/camera.calibration"
    [ -f "$ann" ] || { echo "MISSING $ann" >&2; exit 1; }
    read -r W H < <(image_size "$cal")
    sid="mpi_s${s}_seq${seq}_cam${CAM}_train"
    out="$WORK/train_clips/${sid}.json"
    echo "-- $sid (${W}x${H})"
    motion-tool import-mpi3dhp-supervised-dataset \
      --annotation "$ann" --camera-index "$CAM" \
      --image-width "$W" --image-height "$H" \
      --sequence-id "$sid" --split train \
      --out "$out"
    TRAIN_JSONS+=("$out")
  done
done

echo "=== [2/5] combine train clips ==="
IFS=,; TRAIN_LIST="${TRAIN_JSONS[*]}"; unset IFS
motion-tool combine-supervised-3d-datasets \
  --datasets "$TRAIN_LIST" --expected-split train \
  --out "$WORK/train_combined.json"

echo "=== [3/5] train lifter (cuda) ==="
motion-tool train-supervised-3d-lifter \
  --dataset "$WORK/train_combined.json" \
  --out "$RUN_OUT/checkpoint.pth" \
  --window 81 --channels 256 --epochs 30 --batch-size 128 \
  --learning-rate 0.001 --device cuda \
  --report-out "$RUN_OUT/train_report.json"

echo "=== [4/5] build + lift + audit holdout clips ==="
for s in "${HOLDOUT_SUBJECTS[@]}"; do
  for seq in 1 2; do
    ann="$DATA_ROOT/S${s}/Seq${seq}/annot.mat"
    cal="$DATA_ROOT/S${s}/Seq${seq}/camera.calibration"
    read -r W H < <(image_size "$cal")
    hid="s${s}_seq${seq}_cam${CAM}"
    hdir="$WORK/holdout/$hid"
    mkdir -p "$hdir" "$RUN_OUT/holdout/$hid"
    echo "-- holdout $hid (${W}x${H})"
    motion-tool import-mpi3dhp-ground-truth \
      --annotation "$ann" --calibration "$cal" --camera-index "$CAM" \
      --pose-out "$hdir/pose.json" --lifted-out "$hdir/gt.json" \
      --calibration-out "$hdir/camera.json"
    motion-tool lift-supervised-3d \
      --pose "$hdir/pose.json" --checkpoint "$RUN_OUT/checkpoint.pth" \
      --image-width "$W" --image-height "$H" --device cuda \
      --out "$hdir/prediction.json"
    motion-tool estimate-root-motion --lifted-pose "$hdir/prediction.json" \
      --out "$hdir/prediction_root.json"
    motion-tool estimate-root-motion --lifted-pose "$hdir/gt.json" \
      --out "$hdir/gt_root.json"
    motion-tool audit-supervised-3d \
      --predicted "$hdir/prediction.json" --ground-truth "$hdir/gt.json" \
      --predicted-root "$hdir/prediction_root.json" --ground-truth-root "$hdir/gt_root.json" \
      --out "$RUN_OUT/holdout/$hid/audit.json"
  done
done

echo "=== [5/5] done ==="
