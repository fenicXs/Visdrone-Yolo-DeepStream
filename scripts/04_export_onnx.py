"""04_export_onnx.py (Phase 2)

Export a trained YOLO model to ONNX using Ultralytics export.

Example:
  python scripts/04_export_onnx.py --weights runs/visdrone/yolov8n_640/weights/best.pt --config configs/export/onnx.yaml
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.append(str(ROOT))


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def filter_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(func)
    except Exception:
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default="configs/export/onnx.yaml")
    ap.add_argument("--out-dir", default="models/onnx")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERR] Weights not found: {weights}")
        return 2

    cfg = load_yaml(args.config)

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERR] ultralytics not importable.")
        print(e)
        return 1

    model = YOLO(str(weights))

    export_kwargs: Dict[str, Any] = {
        "format": "onnx",
        "imgsz": int(cfg.get("imgsz", 640)),
        "opset": int(cfg.get("opset", 12)),
        "dynamic": bool(cfg.get("dynamic", True)),
        "simplify": bool(cfg.get("simplify", True)),
    }
    export_kwargs = filter_kwargs(model.export, export_kwargs)

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Export kwargs:", export_kwargs)
    out = model.export(**export_kwargs)
    print("[OK] Export done:", out)
    print(f"[INFO] Put/copy the ONNX into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
