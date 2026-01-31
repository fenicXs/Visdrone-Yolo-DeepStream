#!/usr/bin/env bash
set -euo pipefail

GT="data/VisDrone/VisDrone2019-VID-test-dev/annotations/uav0000073_00600_v.txt"
IMG_DIR="data/VisDrone/VisDrone2019-VID-test-dev/sequences/uav0000073_00600_v"

echo "=== Balanced eval ==="
python tools/eval_visdrone_clip_fixed.py \
  --gt "$GT" \
  --pred_dir "/scratch/pkrish52/Visdrone-Yolo-DeepStream/runs/detect/uav73_y8s960_balanced_conf050_iou055/labels" \
  --img_dir "$IMG_DIR" \
  --iou 0.5 \
  --min_conf 0.50 | tail -n 1

echo "=== Clean eval ==="
python tools/eval_visdrone_clip_fixed.py \
  --gt "$GT" \
  --pred_dir "/scratch/pkrish52/Visdrone-Yolo-DeepStream/runs/detect/uav73_y8s960_clean_conf060_iou050/labels" \
  --img_dir "$IMG_DIR" \
  --iou 0.5 \
  --min_conf 0.60 | tail -n 1
