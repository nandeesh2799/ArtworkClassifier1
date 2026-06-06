from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow might not be installed in some environments (e.g., app-only runs).
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def raw_dir(self) -> Path:
        return self.dataset_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.dataset_dir / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.dataset_dir / "splits"

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def exports_dir(self) -> Path:
        return self.model_dir / "exports"


def project_paths() -> Paths:
    # Resolve relative to this file (stable even when called from other cwd).
    root = Path(__file__).resolve().parents[1]
    return Paths(root=root)


def iter_image_files(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

