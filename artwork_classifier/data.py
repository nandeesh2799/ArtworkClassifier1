from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils import ensure_dir, iter_image_files, write_json


DATASET_SLUG = "thedownhill/art-images-drawings-painting-sculpture-engraving"


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    train_dir: Path
    val_dir: Path
    class_names: List[str]


def download_primary_dataset() -> Path:
    """
    Mandatory dataset source:

    import kagglehub
    path = kagglehub.dataset_download("thedownhill/art-images-drawings-painting-sculpture-engraving")
    """
    import kagglehub  # local import so app can run without kagglehub

    return Path(kagglehub.dataset_download(DATASET_SLUG))


def detect_layout(download_path: Path) -> DatasetLayout:
    """
    KaggleHub dataset v2 structure used in your notebook:
      <path>/musemart/dataset_updated/training_set/<class>/*.jpg
      <path>/musemart/dataset_updated/validation_set/<class>/*.jpg
    """
    root = download_path / "musemart" / "dataset_updated"
    train_dir = root / "training_set"
    val_dir = root / "validation_set"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Could not find expected dataset folders. "
            f"Expected: {train_dir} and {val_dir}. Got: {download_path}"
        )

    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise RuntimeError(f"No class folders found under {train_dir}")

    return DatasetLayout(root=root, train_dir=train_dir, val_dir=val_dir, class_names=class_names)


def _is_image_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def clean_corrupted_images(input_dirs: List[Path], report_path: Path) -> pd.DataFrame:
    """
    Scans the dataset folders and deletes files that PIL can't verify.
    This prevents TensorFlow decode errors at training time.
    """
    rows = []
    bad = 0

    for d in input_dirs:
        for p in tqdm(list(iter_image_files(d)), desc=f"Scanning {d.name}"):
            ok = _is_image_readable(p)
            if not ok:
                bad += 1
                rows.append({"path": str(p), "status": "deleted_corrupt"})
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    rows[-1]["status"] = "failed_to_delete"

    df = pd.DataFrame(rows)
    ensure_dir(report_path.parent)
    df.to_csv(report_path, index=False)
    print(f"Corrupted images removed: {bad}. Report: {report_path}")
    return df


def build_manifest(layout: DatasetLayout) -> pd.DataFrame:
    """
    Build a single manifest from the dataset's provided train/val folders.
    We merge them and then create our own stratified train/val/test split.
    """
    items: List[Dict[str, str]] = []
    for split_name, split_dir in [("provided_train", layout.train_dir), ("provided_val", layout.val_dir)]:
        for class_name in layout.class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for p in iter_image_files(class_dir):
                items.append(
                    {
                        "filepath": str(p.resolve()),
                        "label": class_name,
                        "source_split": split_name,
                    }
                )

    df = pd.DataFrame(items)
    if df.empty:
        raise RuntimeError("Manifest is empty after scanning dataset directories.")
    return df


def stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.10,
    val_size: float = 0.10,
) -> pd.DataFrame:
    """
    Create stratified train/val/test splits from a combined dataset.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    # val fraction relative to remaining train set
    val_frac_of_train = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_frac_of_train,
        random_state=seed,
        stratify=train_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return out


def materialize_splits(
    df: pd.DataFrame,
    out_dir: Path,
    class_names: List[str],
    link_mode: str = "symlink",
) -> None:
    """
    Create:
      dataset/processed/<split>/<class>/*  (symlinks or copies)
    """
    out_dir = ensure_dir(out_dir)
    for split in ["train", "val", "test"]:
        for c in class_names:
            ensure_dir(out_dir / split / c)

    def _link(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return
        try:
            if link_mode == "copy":
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
        except Exception:
            # Fall back to copying when symlinks are not allowed.
            shutil.copy2(src, dst)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Materializing splits"):
        src = Path(row["filepath"])
        split = row["split"]
        label = row["label"]
        dst = out_dir / split / label / src.name
        _link(src, dst)


def save_class_names(class_names: List[str], out_path: Path) -> None:
    write_json(out_path, class_names)

