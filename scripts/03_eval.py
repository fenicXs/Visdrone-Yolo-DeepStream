"""03_eval.py

Evaluate a trained YOLO weights file on a split (default: val).

What this version fixes (based on common VisDrone mistakes):
- If your train config says imgsz=640 but you actually trained with --imgsz 960,
  the old script would silently evaluate at 640 and you would think your model got worse.
- This script will auto-load Ultralytics 'args.yaml' from the run directory next to the weights
  (runs/.../<exp_name>/args.yaml) to recover the *actual* training/eval defaults.

Examples:
  # Evaluate using run defaults (recommended; avoids config mismatch)
  python scripts/03_eval.py --weights runs/visdrone/yolov8s_960/weights/best.pt

  # Override resolution explicitly
  python scripts/03_eval.py --weights runs/visdrone/yolov8s_960/weights/best.pt --imgsz 960

  # Evaluate multiple resolutions in one shot (comma-separated)
  python scripts/03_eval.py --weights runs/visdrone/yolov8s_960/weights/best.pt --imgsz-list 640,960

  # Override dataset YAML + device
  python scripts/03_eval.py --weights runs/visdrone/yolov8s_960/weights/best.pt --data configs/dataset/visdrone.yaml --device 0

Outputs:
  - prints key metrics (mAP50-95, mAP50, precision, recall if available)
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
from typing import Any, Dict, List, Optional

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
    """Filter kwargs based on function signature, unless it accepts **kwargs."""
    try:
        sig = inspect.signature(func)
    except Exception:
        return kwargs

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def parse_imgsz_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            raise ValueError(f"Invalid --imgsz-list entry: {p!r} (expected ints like 640,960)")
    return out or None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to weights (best.pt)")
    ap.add_argument("--train-config", default=None, help="Optional: train config YAML (fallback only)")
    ap.add_argument("--data", default=None, help="Dataset YAML override")
    ap.add_argument("--imgsz", type=int, default=None, help="Single eval resolution override")
    ap.add_argument("--imgsz-list", default=None, help="Comma-separated list of eval resolutions (e.g. 640,960)")
    ap.add_argument("--device", default=None, help='e.g. "0" or "cpu"')
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])

    # Optional val args (Ultralytics will ignore if unsupported; we filter anyway)
    ap.add_argument("--conf", type=float, default=None, help="NMS confidence threshold (val default ~0.001)")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU threshold (val default ~0.7)")
    ap.add_argument("--max-det", type=int, default=None, help="Max detections per image (default 300)")
    ap.add_argument("--half", action="store_true", help="Use FP16 if supported")

    # Optional output grouping (Ultralytics may or may not use these keys; script filters)
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)

    return ap.parse_args()


def find_run_args(weights: Path) -> Dict[str, Any]:
    """Try to recover Ultralytics args.yaml from the run directory next to weights."""
    # Typical: runs/.../exp_name/weights/best.pt -> run_dir = .../exp_name
    run_dir = weights.parent.parent
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        try:
            return load_yaml(args_yaml)
        except Exception:
            return {}
    return {}


def extract_metrics(metrics_obj: Any) -> Dict[str, Any]:
    """Extract commonly used metrics across Ultralytics versions."""
    out: Dict[str, Any] = {}

    box = getattr(metrics_obj, "box", None)
    if box is not None:
        for k in ["map", "map50", "mp", "mr"]:
            if hasattr(box, k):
                try:
                    out[k] = float(getattr(box, k))
                except Exception:
                    out[k] = getattr(box, k)

    speed = getattr(metrics_obj, "speed", None)
    if speed is not None:
        out["speed"] = speed

    if not out:
        out["raw"] = str(metrics_obj)
    return out


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERR] Weights not found: {weights}")
        return 2

    imgsz_list = parse_imgsz_list(args.imgsz_list)
    if args.imgsz is not None and imgsz_list is not None:
        print("[ERR] Use either --imgsz or --imgsz-list, not both.")
        return 2

    # Priority order:
    # 1) Run args.yaml next to weights (MOST TRUSTWORTHY; matches the run)
    # 2) Train-config (fallback)
    # 3) CLI overrides (always win)
    run_args = find_run_args(weights)

    cfg = {}
    if args.train_config:
        try:
            cfg = load_yaml(args.train_config)
        except Exception as e:
            print(f"[WARN] Failed to read --train-config ({args.train_config}): {e}")
            cfg = {}

    # base defaults from run args if present, else from train-config
    base_data = run_args.get("data") or cfg.get("data")
    base_imgsz = run_args.get("imgsz") or cfg.get("imgsz") or 640
    base_device = run_args.get("device") if run_args.get("device") is not None else cfg.get("device")

    # CLI overrides
    data = args.data or base_data
    device = args.device if args.device is not None else base_device

    # Final imgsz(s)
    if imgsz_list is not None:
        imgszs = imgsz_list
    else:
        imgszs = [args.imgsz or int(base_imgsz)]

    # Warn if train-config disagrees with run args (this is the exact trap you hit)
    if args.train_config and run_args:
        cfg_imgsz = cfg.get("imgsz")
        if cfg_imgsz is not None and int(cfg_imgsz) != int(base_imgsz):
            print(
                f"[WARN] train-config imgsz={cfg_imgsz} but run args.yaml imgsz={base_imgsz}. "
                f"Using run args.yaml by default. (Override with --imgsz if you *really* mean it.)"
            )

    if not data:
        print("[ERR] Could not infer dataset YAML. Provide --data or --train-config, or ensure args.yaml exists.")
        return 2

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERR] ultralytics not importable. Did you install requirements?")
        print(e)
        return 1

    model = YOLO(str(weights))

    out_dir = ROOT / "assets/benchmark_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    all_runs: List[Dict[str, Any]] = []
    for imgsz in imgszs:
        val_kwargs: Dict[str, Any] = {
            "data": data,
            "imgsz": int(imgsz),
            "split": args.split,
        }
        if device is not None:
            val_kwargs["device"] = device
        if args.project is not None:
            val_kwargs["project"] = args.project
        if args.name is not None:
            val_kwargs["name"] = args.name

        # Optional val args
        if args.conf is not None:
            val_kwargs["conf"] = float(args.conf)
        if args.iou is not None:
            val_kwargs["iou"] = float(args.iou)
        if args.max_det is not None:
            val_kwargs["max_det"] = int(args.max_det)
        if args.half:
            val_kwargs["half"] = True

        val_kwargs = filter_kwargs(model.val, val_kwargs)

        print("\n[INFO] Eval args:")
        print(json.dumps(val_kwargs, indent=2, default=str))

        metrics = model.val(**val_kwargs)
        summary = extract_metrics(metrics)

        print("[OK] Eval summary:")
        print(json.dumps(summary, indent=2))

        run_rec = {
            "weights": str(weights),
            "data": str(data),
            "imgsz": int(imgsz),
            "split": args.split,
            "summary": summary,
        }
        all_runs.append(run_rec)

        out_path = out_dir / f"val_metrics_{weights.stem}_img{int(imgsz)}_{stamp}.json"
        out_path.write_text(json.dumps(run_rec, indent=2), encoding="utf-8")
        print(f"[OK] Wrote metrics JSON: {out_path}")

    # Convenience: one combined file
    combined = out_dir / f"val_metrics_{weights.stem}_combined_{stamp}.json"
    combined.write_text(json.dumps({"runs": all_runs}, indent=2), encoding="utf-8")
    print(f"[OK] Wrote combined metrics JSON: {combined}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
