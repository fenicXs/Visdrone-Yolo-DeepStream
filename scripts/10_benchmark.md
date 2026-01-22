# Benchmark checklist (Phase 2)

Capture these for each precision mode (FP32/FP16/INT8) on each device (Jetson/Qualcomm):

- Model: yolov8n vs yolov8s, imgsz
- Precision: FP32 / FP16 / INT8
- Input: video resolution, FPS, codec
- Throughput: FPS (DeepStream perf output)
- Latency: end-to-end (if available)
- Memory: peak GPU memory
- Power: device power draw (if you can)

Put results into:
- `assets/benchmark_tables/benchmarks.md` or a CSV/JSON file.
