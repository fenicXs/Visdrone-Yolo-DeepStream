"""03_eval.py

Evaluate a trained YOLO weights file on a split (default: val).

Examples:
  python scripts/03_eval.py --weights runs/visdrone/yolov8n_640/weights/best.pt --train-config configs/train/yolo8n_640.yaml
  python scripts/03_eval.py --weights path/to/best.pt --data configs/dataset/visdrone.yaml --imgsz 640

Outputs:
  - prints key metrics
  - writes a JSON summary into assets/benchmark_tables/
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.append(str(ROOT))


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping/dict. Got: {type(data)}")
    return data


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
    ap.add_argument("--weights", required=True, help="Path to weights (best.pt)")
    ap.add_argument("--train-config", default=None, help="Optional: load data/imgsz/device from this train config")

    ap.add_argument("--data", default=None, help="Dataset YAML (overrides train-config)")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None, help='e.g. "0" or "cpu"')
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])

    # Optional output grouping (Ultralytics may or may not use these keys; script filters)
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)

    return ap.parse_args()


def extract_metrics(metrics_obj: Any) -> Dict[str, Any]:
    """Try to extract commonly used metrics across Ultralytics versions."""
    out: Dict[str, Any] = {}

    # Newer Ultralytics: metrics_obj.box.map, map50, mp, mr, etc.
    box = getattr(metrics_obj, "box", None)
    if box is not None:
        for k in ["map", "map50", "mp", "mr"]:
            if hasattr(box, k):
                try:
                    out[k] = float(getattr(box, k))
                except Exception:
                    out[k] = getattr(box, k)

    # Speed dict often present
    speed = getattr(metrics_obj, "speed", None)
    if speed is not None:
        out["speed"] = speed

    # As a fallback, stringify
    if not out:
        out["raw"] = str(metrics_obj)
    return out


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERR] Weights not found: {weights}")
        return 2

    cfg = {}
    if args.train_config:
        cfg = load_yaml(args.train_config)

    data = args.data or cfg.get("data")
    imgsz = args.imgsz or cfg.get("imgsz", 640)
    device = args.device or cfg.get("device", None)

    if not data:
        print("[ERR] You must provide --data or --train-config containing `data:`")
        return 2

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERR] ultralytics not importable. Did you install requirements?")
        print(e)
        return 1

    model = YOLO(str(weights))

    val_kwargs: Dict[str, Any] = {
        "data": data,
        "imgsz": imgsz,
        "split": args.split,
    }
    if device is not None:
        val_kwargs["device"] = device
    if args.project is not None:
        val_kwargs["project"] = args.project
    if args.name is not None:
        val_kwargs["name"] = args.name

    val_kwargs = filter_kwargs(model.val, val_kwargs)

    print("[INFO] Eval args:")
    print(json.dumps(val_kwargs, indent=2, default=str))

    metrics = model.val(**val_kwargs)
    summary = extract_metrics(metrics)

    print("[OK] Eval summary:")
    print(json.dumps(summary, indent=2))

    out_dir = ROOT / "assets/benchmark_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"val_metrics_{weights.stem}_{stamp}.json"
    out_path.write_text(json.dumps({
        "weights": str(weights),
        "data": str(data),
        "imgsz": imgsz,
        "split": args.split,
        "summary": summary,
    }, indent=2), encoding="utf-8")

    print(f"[OK] Wrote metrics JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
