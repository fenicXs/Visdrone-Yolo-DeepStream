from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return repository root path (…/visdrone-yolo-deepstream)."""
    # src/utils/paths.py -> utils -> src -> repo root
    return Path(__file__).resolve().parents[2]


def abs_path(*parts: str) -> Path:
    """Convenience: repo_root()/parts..."""
    return repo_root().joinpath(*parts)
