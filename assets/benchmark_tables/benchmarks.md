# Benchmarks (re-measured 2026-11-10)

All numbers below were re-run from the committed weights, not copied from earlier runs.

Hardware and software: NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, driver 617.42), Ultralytics 8.4.7, PyTorch 2.10.0+cu130, DeepStream 8.0 (`nvcr.io/nvidia/deepstream:8.0-triton-multiarch`) in Docker on WSL2 Ubuntu 24.04.

## Detection accuracy (Ultralytics val, `best.pt` of each run, 80 epochs)

Dataset: VisDrone2019-DET in YOLO format. Val = 548 images. Test-dev = 1,610 images. Checkpoints were selected on val, so test-dev is the held out number.

| Model | Input | Split | Precision | Recall | mAP50 | mAP50-95 | Inference (PyTorch, ms/img) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 640 | val | 0.419 | 0.316 | 0.309 | 0.174 | 1.2 |
| YOLOv8n | 640 | test-dev | 0.370 | 0.280 | 0.254 | 0.141 | 1.0 |
| YOLOv8n | 960 | val | 0.508 | 0.387 | 0.401 | 0.241 | 2.2 |
| YOLOv8n | 960 | test-dev | 0.438 | 0.352 | 0.328 | 0.191 | 2.3 |
| YOLOv8s | 960 | val | 0.564 | 0.463 | 0.478 | 0.293 | 5.7 |
| YOLOv8s | 960 | test-dev | 0.522 | 0.397 | 0.389 | 0.231 | 6.0 |

Weights: `runs/visdrone/yolov8n_640`, `yolov8n_6402` (trained at imgsz 960), `yolov8s_960_2`. There is no YOLOv8s at 640 run.

## Single clip detection quality (DeepStream demo clip)

Clip `uav0000073_00600_v` from VisDrone2019-VID-test-dev (328 frames, 14,312 ground truth boxes), YOLOv8s at 960, IoU 0.5, scored with `tools/eval_visdrone_clip_fixed.py`. This is one clip, not a dataset level metric.

| Preset | conf | NMS IoU | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced | 0.50 | 0.55 | 0.643 | 0.643 | 0.643 |
| Clean demo | 0.60 | 0.50 | 0.739 | 0.563 | 0.639 |

## DeepStream throughput

YOLOv8s ONNX (opset 17, 960x960) through `deepstream-app` with the DeepStream-Yolo parser. One 1920x1080 H.264 stream (the demo clip looped to 9,840 frames), batch size 1, no OSD, fakesink with `sync=0`, perf interval 2 s. Reported value is the running average at the end of the run.

| Precision | `network-mode` | FPS (avg) |
| --- | ---: | ---: |
| FP32 | 0 | 131 |
| FP16 | 2 | 302 |

INT8 was not benchmarked (it needs a calibration cache).
