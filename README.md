# VisDrone -> YOLOv8 -> ONNX/TensorRT -> DeepStream Pipeline

This repo is structured as a **deployable object detection pipeline** (not a notebook dump):
1) Train YOLOv8 on VisDrone
2) Export to ONNX
3) Build TensorRT engine
4) Run DeepStream inference with overlays

## Current status (what works)
- YOLOv8 model trained on VisDrone (960 input)
- ONNX exported for DeepStream
- TensorRT engine built (FP32)
- DeepStream app runs and writes annotated output video

## Result preview (short GIF)
![Result preview](assets/demo_outputs/preview.gif)

Generate it locally from the annotated MP4:
```bash
ffmpeg -y -i assets/demo_outputs/uav0000073_00600_v_ds.mp4 -t 4 -vf "fps=8,scale=960:-1:flags=lanczos" assets/demo_outputs/preview.gif
```

## Confidence presets (from `confidence_compare.txt`)
| Preset     | conf | NMS IoU |  Prec |   Rec |    F1 |
| ---------- | ---: | ------: | ----: | ----: | ----: |
| Balanced   | 0.50 |    0.55 | 0.643 | 0.643 | 0.643 |
| Clean demo | 0.60 |    0.50 | 0.739 | 0.563 | 0.639 |

## Repo layout
- `configs/` dataset/train/export/deepstream configs
- `scripts/` runnable steps
- `assets/` demo inputs/outputs (not committed to git)

## Deployment quickstart (DeepStream)
> Run commands from the repo root inside your WSL + Docker setup.

1) Build DeepStream-Yolo custom parser/plugin
```bash
bash scripts/08_deepstream_build.sh
```

2) Run DeepStream on the demo video
```bash
bash scripts/09_deepstream_run.sh
```

Output videos are written to:
```
assets/demo_outputs/
  uav0000073_00600_v_ds.mp4
  uav0000073_00600_v_ds_debug.mp4
  uav0000073_00600_v_ds_noosd.mp4
```

## Notes
- This repo keeps large artifacts out of git (weights, videos, logs).
- If you want realtime input or RTSP output, update the DeepStream app config.
