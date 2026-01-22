"""07_make_calib_list.py (Phase 2 helper)

Create a text file listing calibration image paths (one per line).
Useful if a downstream TensorRT/DeepStream pipeline wants a list file.

Example:
  python scripts/07_make_calib_list.py --data configs/dataset/visdrone.yaml --out models/calib/calibration.txt --n 1000
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.append(str(ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset YAML")
    ap.add_argument("--out", default="models/calib/calibration.txt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.data)
    base = Path(cfg.get("path", "."))
    if not base.is_absolute():
        base = (ROOT / base).resolve()

    train_rel = cfg.get("train", "images/train")
    if isinstance(train_rel, list):
        raise ValueError("This script expects `train:` to be a single directory string.")
    train_images = (base / train_rel).resolve()

    imgs = [p for p in train_images.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        print(f"[ERR] No images found in: {train_images}")
        return 2

    random.Random(args.seed).shuffle(imgs)
    imgs = imgs[: min(args.n, len(imgs))]

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p) for p in imgs) + "\n", encoding="utf-8")

    print(f"[OK] Wrote {len(imgs)} paths to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
