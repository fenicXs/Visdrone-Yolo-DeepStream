"""02_train.py

Train Ultralytics YOLO using a YAML config file.

Examples:
  python scripts/02_train.py --config configs/train/yolo8n_640.yaml
  python scripts/02_train.py --config configs/train/yolo8n_640.yaml --epochs 2 --batch 4 --name smoke

Notes:
  - This script does a quick dataset sanity check by default.
  - It filters config keys based on your installed Ultralytics version to reduce breakage.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.append(str(ROOT))


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping/dict. Got: {type(data)}")
    return data


def resolve_dataset_paths(data_yaml: str | Path) -> Tuple[Path, Path, Path, Path]:
    """Return (train_images_dir, val_images_dir, train_labels_dir, val_labels_dir)."""
    cfg = load_yaml(data_yaml)
    base = Path(cfg.get("path", "."))
    if not base.is_absolute():
        base = (ROOT / base).resolve()

    train_rel = cfg.get("train", "images/train")
    val_rel = cfg.get("val", "images/val")

    # train/val can be strings or lists in Ultralytics; handle common case (string).
    if isinstance(train_rel, list) or isinstance(val_rel, list):
        raise ValueError("This script expects `train:` and `val:` to be strings (single directory).")

    train_images = (base / train_rel).resolve()
    val_images = (base / val_rel).resolve()

    # Standard YOLO layout: labels mirror images directories.
    train_labels = Path(str(train_images).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))
    val_labels = Path(str(val_images).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))

    return train_images, val_images, train_labels, val_labels


def count_images(folder: Path, max_scan: int | None = None) -> int:
    if not folder.exists():
        return 0
    n = 0
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            n += 1
            if max_scan and n >= max_scan:
                break
    return n


def sample_missing_labels(images_dir: Path, labels_dir: Path, samples: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Sample a subset of images and check if corresponding label files exist."""
    if not images_dir.exists():
        return {"error": f"Images dir missing: {images_dir}"}
    if not labels_dir.exists():
        return {"error": f"Labels dir missing: {labels_dir}"}

    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return {"error": f"No images found in: {images_dir}"}

    rng = random.Random(seed)
    rng.shuffle(imgs)
    imgs = imgs[: min(samples, len(imgs))]

    missing = 0
    bad_format = 0
    checked = 0

    for img in imgs:
        label = labels_dir / (img.stem + ".txt")
        checked += 1
        if not label.exists():
            missing += 1
            continue

        # Basic format check: each line should have 5 numbers: cls xc yc w h
        try:
            text = label.read_text(encoding="utf-8").strip()
            if not text:
                # empty labels are allowed (image with no objects), do not count as bad
                continue
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    bad_format += 1
                    break
                # Try casting
                _ = float(parts[0])
                _ = float(parts[1]); _ = float(parts[2]); _ = float(parts[3]); _ = float(parts[4])
        except Exception:
            bad_format += 1

    return {
        "checked": checked,
        "missing_labels": missing,
        "missing_ratio": (missing / checked) if checked else None,
        "bad_format_labels": bad_format,
    }


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/yolo8n_640.yaml", help="Train config YAML")

    # Common overrides
    ap.add_argument("--model", default=None, help="Override model (e.g., yolov8n.pt)")
    ap.add_argument("--data", default=None, help="Override dataset YAML")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None, help='e.g. "0", "0,1", or "cpu"')
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)

    ap.add_argument("--resume", action="store_true", help="Resume last run (Ultralytics behavior)")
    ap.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run dir")

    # Sanity check controls
    ap.add_argument("--no-sanity", action="store_true", help="Skip dataset sanity checks")
    ap.add_argument("--sanity-samples", type=int, default=200, help="How many images to sample for label checks")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    # Apply overrides if provided
    if args.model is not None:
        cfg["model"] = args.model
    if args.data is not None:
        cfg["data"] = args.data
    for k in ["epochs", "batch", "imgsz", "device", "workers", "project", "name", "seed", "patience"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    # Required keys
    if "model" not in cfg:
        print("[ERR] Missing `model` in config.")
        return 2
    if "data" not in cfg:
        print("[ERR] Missing `data` in config.")
        return 2

    data_yaml = cfg["data"]

    # Sanity checks
    if not args.no_sanity:
        try:
            train_img, val_img, train_lbl, val_lbl = resolve_dataset_paths(data_yaml)
            print(f"[INFO] Train images: {train_img}")
            print(f"[INFO] Val images:   {val_img}")
            print(f"[INFO] Train labels: {train_lbl}")
            print(f"[INFO] Val labels:   {val_lbl}")

            n_train = count_images(train_img, max_scan=None)
            n_val = count_images(val_img, max_scan=None)
            print(f"[INFO] Image count: train={n_train} val={n_val}")
            if n_train == 0 or n_val == 0:
                print("[ERR] No images found. Your dataset path or conversion is broken.")
                return 3

            stats_train = sample_missing_labels(train_img, train_lbl, samples=args.sanity_samples, seed=int(cfg.get('seed', 42)))
            stats_val = sample_missing_labels(val_img, val_lbl, samples=args.sanity_samples, seed=int(cfg.get('seed', 42)))
            print("[INFO] Label sanity (train sample):", json.dumps(stats_train, indent=2))
            print("[INFO] Label sanity (val sample):  ", json.dumps(stats_val, indent=2))

            if isinstance(stats_train, dict) and stats_train.get("missing_ratio") is not None and stats_train["missing_ratio"] > 0.05:
                print("[WARN] >5% missing label files in sampled TRAIN set. Fix conversion before wasting time.")
            if isinstance(stats_val, dict) and stats_val.get("missing_ratio") is not None and stats_val["missing_ratio"] > 0.05:
                print("[WARN] >5% missing label files in sampled VAL set. Fix conversion before wasting time.")
        except Exception as e:
            print("[WARN] Sanity check failed (not fatal).")
            print(e)

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERR] ultralytics not importable. Did you install requirements?")
        print(e)
        return 1

    model_id = cfg.pop("model")
    print(f"[INFO] Loading model: {model_id}")
    model = YOLO(model_id)

    # Ultralytics sometimes expects exist_ok key as exist_ok (snake-case)
    if args.exist_ok:
        cfg["exist_ok"] = True
    if args.resume:
        cfg["resume"] = True

    # Filter kwargs for compatibility
    train_kwargs = filter_kwargs(model.train, cfg)

    # Ensure project path is relative to repo root unless absolute
    if "project" in train_kwargs:
        proj = Path(str(train_kwargs["project"]))
        if not proj.is_absolute():
            train_kwargs["project"] = str((ROOT / proj).resolve())

    print("[INFO] Training args:")
    print(json.dumps(train_kwargs, indent=2, default=str))

    results = model.train(**train_kwargs)

    # Best-effort summary
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        # Ultralytics usually uses project/name
        project = train_kwargs.get("project", "runs")
        name = train_kwargs.get("name", "exp")
        save_dir = Path(project) / name

    print(f"[OK] Training complete. Run directory: {save_dir}")
    print("Next: evaluate with scripts/03_eval.py using weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
