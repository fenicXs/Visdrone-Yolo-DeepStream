#!/usr/bin/env bash
set -euo pipefail

# Phase 2: run DeepStream app (local install or Docker).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/deepstream/deepstream_app_config_visdrone_960_filesink.txt}"
IMAGE="${DS_IMAGE:-nvcr.io/nvidia/deepstream:8.0-triton-multiarch}"

mkdir -p "$ROOT/assets/demo_outputs" "$ROOT/models"

# If the ONNX isn't in models yet, try the known training output path.
DEFAULT_ONNX="$ROOT/runs/visdrone/yolov8s_960_2/weights/best.onnx"
if [[ ! -f "$ROOT/models/best_ds.onnx" && -f "$DEFAULT_ONNX" ]]; then
  cp "$DEFAULT_ONNX" "$ROOT/models/best_ds.onnx"
fi

if command -v deepstream-app >/dev/null 2>&1; then
  deepstream-app -c "$ROOT/$CONFIG"
  exit $?
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERR] deepstream-app not found and docker is missing." >&2
  exit 1
fi

TTY_FLAG="-i"
if [ -t 1 ]; then
  TTY_FLAG="-it"
fi

  docker run --rm $TTY_FLAG \
  --gpus all \
  --net=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v "$ROOT":/workspace \
  --name deepstream \
  --entrypoint /bin/bash \
  "$IMAGE" \
  bash -lc "deepstream-app -c /workspace/$CONFIG"
