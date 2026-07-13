#!/usr/bin/env bash
# Train all six model variants (yolov8n/s, yolo11n/s, yolo26n/s) back to back.
#
# Usage:
#   train_all_models.sh <data.yaml>              holdout mode (default)
#   train_all_models.sh <kfold_root> kfold        kfold mode (all folds per model)
set -euo pipefail

DATA=${1:?usage: train_all_models.sh <data.yaml|kfold_root> [holdout|kfold]}
MODE=${2:-holdout}

for MODEL in yolov8n yolov8s yolo11n yolo11s yolo26n yolo26s; do
  echo "=== Training ${MODEL} (${MODE}) ==="
  if [ "$MODE" = "kfold" ]; then
    python -m training.train --model "$MODEL" --mode kfold --kfold-root "$DATA"
  else
    python -m training.train --model "$MODEL" --mode holdout --data "$DATA"
  fi
done
