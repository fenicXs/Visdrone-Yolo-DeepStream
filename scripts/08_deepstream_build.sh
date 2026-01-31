#!/usr/bin/env bash
set -euo pipefail

# Phase 2: build DeepStream-Yolo custom plugin (local or Docker).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS_YOLO_DIR="$ROOT/third_party/DeepStream-Yolo"
IMAGE="${DS_IMAGE:-nvcr.io/nvidia/deepstream:8.0-triton-multiarch}"
CUDA_VER="${CUDA_VER:-12.8}"

if [[ ! -d "$DS_YOLO_DIR" ]]; then
  echo "[ERR] Missing DeepStream-Yolo at: $DS_YOLO_DIR" >&2
  echo "Clone it first (example):" >&2
  echo "  git clone https://github.com/marcoslucianops/DeepStream-Yolo.git \"$DS_YOLO_DIR\"" >&2
  exit 1
fi

if command -v deepstream-app >/dev/null 2>&1; then
  make -C "$DS_YOLO_DIR/nvdsinfer_custom_impl_Yolo" CUDA_VER="$CUDA_VER"
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
  --name deepstream-build \
  "$IMAGE" \
  bash -lc "make -C /workspace/third_party/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo CUDA_VER=$CUDA_VER"
