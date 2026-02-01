# VisDrone YOLOv8 Deployment with ONNX, TensorRT, and DeepStream

This repository provides a reproducible, end-to-end pipeline for training a YOLOv8 detector on the VisDrone VID dataset and deploying it with NVIDIA DeepStream (TensorRT) for GPU inference.

## Demo preview
![Result preview](assets/demo_outputs/uav0000297_00000_v_ds.gif)

Re-generate the GIF (cropping black bars, 4 seconds @ 8 FPS):
```bash
python scripts/preview_uav0000297_gif.py
```

## Metrics (confidence presets)
| Preset     | conf | NMS IoU |  Prec |   Rec |    F1 | mAP50 | mAP50-95 |
| ---------- | ---: | ------: | ----: | ----: | ----: | ----: | -------: |
| Balanced   | 0.50 |    0.55 | 0.643 | 0.643 | 0.643 | 0.482 |    0.285 |
| Clean demo | 0.60 |    0.50 | 0.739 | 0.563 | 0.639 | 0.482 |    0.285 |

## Requirements (tested)
- Windows 11 + WSL2 (Ubuntu 24.04) or native Linux
- NVIDIA GPU with recent driver and WSL CUDA support
- Docker Desktop with WSL integration and NVIDIA Container Toolkit
- Python 3.10+ for training/export (Ultralytics)
- Optional: `imageio-ffmpeg` for building MP4/GIF previews

## Repository layout
- `configs/` dataset, training, export, and DeepStream configs
- `scripts/` runnable steps
- `assets/` demo inputs/outputs (large videos are gitignored)

## Reproducible workflow

Unless noted, run commands from the repo root in WSL bash.

### 1) Dataset layout
Place VisDrone in YOLO format under:
```
data/VisDrone/
  images/train
  images/val
  labels/train
  labels/val
```
For VID sequences and annotations:
```
data/VisDrone/VisDrone2019-VID-test-dev/
  sequences/<clip_name>/
  annotations/<clip_name>.txt
```

### 2) Train (optional)
```bash
python scripts/02_train.py --config configs/train/yolo8n_640.yaml
```
Your best weights will land at:
```
runs/visdrone/<run_name>/weights/best.pt
```

### 3) Export ONNX for DeepStream
This uses the DeepStream-Yolo exporter (clone once):
```bash
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git third_party/DeepStream-Yolo
python third_party/DeepStream-Yolo/utils/export_yoloV8.py -w runs/visdrone/yolov8s_960_2/weights/best.pt -s 960 960 --opset 17 --simplify --batch 1
cp runs/visdrone/yolov8s_960_2/weights/best.onnx models/best_ds.onnx
```

### 4) Build the DeepStream custom parser/plugin
```bash
bash scripts/08_deepstream_build.sh
```

### 5) Run DeepStream (GPU, file-in -> file-out)
```bash
bash scripts/09_deepstream_run.sh
```
Outputs are written to:
```
assets/demo_outputs/
  uav0000073_00600_v_ds.mp4
  uav0000073_00600_v_ds_debug.mp4
  uav0000073_00600_v_ds_noosd.mp4
```

### 6) Run DeepStream on a specific VID sequence
Build an MP4 from a sequence and run it with a matching config.
Example (uav0000297_00000_v):
```bash
python - <<'PY'
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

seq_dir = Path("data/VisDrone/VisDrone2019-VID-test-dev/sequences/uav0000297_00000_v")
out_mp4 = Path("assets/demo_inputs/uav0000297_00000_v.mp4")
out_mp4.parent.mkdir(parents=True, exist_ok=True)
frames = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])
writer = imageio.get_writer(str(out_mp4), fps=30, macro_block_size=None)
for p in frames:
    frame = imageio.imread(p)
    h, w = frame.shape[:2]
    if (h % 2) or (w % 2):
        frame = np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
    writer.append_data(frame)
writer.close()
print("Wrote", out_mp4)
PY

bash scripts/09_deepstream_run.sh configs/deepstream/deepstream_app_config_visdrone_960_filesink_uav0000297.txt
```

## Notes
- Large artifacts (videos, weights, logs) are gitignored; only a small GIF is tracked for the README preview.
- If CUDA is not available, Ultralytics inference will fall back to CPU (slower).
