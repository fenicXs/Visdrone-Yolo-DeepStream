# VisDrone → YOLOv8 → (ONNX/TensorRT) → DeepStream (Jetson) Pipeline

This repo is intentionally structured like a *deployable* project, not a notebook dump.
Phase 1 is **dataset sanity → training → evaluation**. Phase 2 is **export + Jetson/DeepStream**.

## Repo layout
See `configs/` for dataset/train/export/deepstream configs and `scripts/` for runnable steps.

## Quickstart (Phase 1: Train + Eval)

> Run commands from the repo root.

### 1) Create env + install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Dataset location
This repo expects VisDrone (YOLO-format) under:
```
data/VisDrone/
  images/train
  images/val
  labels/train
  labels/val
```
**Do not commit `data/` to git.**

If your `configs/dataset/visdrone.yaml` contains a `download:` section (Ultralytics style), you can try:
```bash
python scripts/01_data_download.py
```

### 3) Smoke test (2 epochs) — do this first
```bash
python scripts/02_train.py --config configs/train/yolo8n_640.yaml --epochs 2 --batch 4 --name smoke_yolov8n_640
```

### 4) Full training
```bash
python scripts/02_train.py --config configs/train/yolo8n_640.yaml
```

### 5) Evaluate (validation split)
After training, your best weights will be at something like:
```
runs/visdrone/yolov8n_640/weights/best.pt
```

Run:
```bash
python scripts/03_eval.py --weights runs/visdrone/yolov8n_640/weights/best.pt --train-config configs/train/yolo8n_640.yaml
```

The eval script prints metrics and also writes a JSON summary to:
```
assets/benchmark_tables/
```

## What to report back (so we can move to Phase 2)
Paste:
- mAP50-95, mAP50, precision, recall
- Your GPU/CPU details (what device you trained on)
- The path to your run dir (e.g., `runs/visdrone/yolov8n_640/`)

## Notes (don’t ignore these)
- VisDrone has lots of *small objects*. Higher `imgsz` (e.g., 960) can improve accuracy but costs FPS and memory.
- If your goal is Jetson deployment, **YOLOv8n or YOLOv8s** is the sane range.
- If dataset sanity check reports lots of missing label files, your conversion is broken. Fix that before wasting compute.
