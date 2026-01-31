#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="runs/visdrone/yolov8s_960_2/weights/best.pt"
VIDEO="assets/demo_inputs/uav0000073_00600_v.mp4"
PROJECT="/scratch/pkrish52/Visdrone-Yolo-DeepStream/runs/detect"

echo "=== Balanced preset (best F1) ==="
yolo predict \
  model="$WEIGHTS" source="$VIDEO" imgsz=960 \
  conf=0.50 iou=0.55 max_det=300 device=0 \
  save=True save_txt=True save_conf=True \
  project="$PROJECT" name="uav73_y8s960_balanced_conf050_iou055" exist_ok=True

echo "=== Clean demo preset (high precision) ==="
yolo predict \
  model="$WEIGHTS" source="$VIDEO" imgsz=960 \
  conf=0.60 iou=0.50 max_det=300 device=0 \
  save=True save_txt=True save_conf=True \
  project="$PROJECT" name="uav73_y8s960_clean_conf060_iou050" exist_ok=True
