import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import pytest

from artwork_classifier.utils import ensure_dir, write_json, read_json, iter_image_files, seed_everything
from artwork_classifier.data import _is_image_readable, clean_corrupted_images, stratified_split


def test_seed_everything_deterministic():
    """Verify seed_everything sets deterministic seeds across numpy and random."""
    seed_everything(1234)
    val1 = np.random.rand()
    seed_everything(1234)
    val2 = np.random.rand()
    assert val1 == val2


def test_json_io_roundtrip(tmp_path):
    """Verify write_json and read_json roundtrip data with UTF-8."""
    target = tmp_path / "subdir" / "metadata.json"
    data = {"project": "ArtworkClassifier1", "num_classes": 5, "active": True}
    write_json(target, data)

    loaded = read_json(target)
    assert loaded == data


def test_iter_image_files(tmp_path):
    """Verify iter_image_files discovers valid image extensions and ignores non-images."""
    (tmp_path / "img1.jpg").write_text("dummy")
    (tmp_path / "img2.PNG").write_text("dummy")
    (tmp_path / "notes.txt").write_text("dummy")
    (tmp_path / "model.h5").write_text("dummy")

    found = list(iter_image_files(tmp_path))
    filenames = [p.name for p in found]
    assert len(found) == 2
    assert "img1.jpg" in filenames
    assert "img2.PNG" in filenames
    assert "notes.txt" not in filenames


def test_image_readability_verification(tmp_path):
    """Verify _is_image_readable correctly differentiates valid images from corrupt files."""
    valid_path = tmp_path / "valid.png"
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(valid_path, format="PNG")

    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"corrupted raw data not an image")

    assert _is_image_readable(valid_path) is True
    assert _is_image_readable(corrupt_path) is False


def test_stratified_split_proportions():
    """Verify stratified_split creates proper train, val, and test splits with expected columns."""
    records = []
    classes = ["painting", "sculpture", "drawings", "engraving", "iconography"]
    for c in classes:
        for i in range(20):
            records.append({"filepath": f"/fake/path/{c}_{i}.jpg", "label": c})

    df = pd.DataFrame(records)
    split_df = stratified_split(df, seed=42, test_size=0.10, val_size=0.10)

    assert "split" in split_df.columns
    assert set(split_df["split"].unique()) == {"train", "val", "test"}

    # Verify all classes are present in train, val, and test splits
    for s in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == s]
        for c in classes:
            assert (sub["label"] == c).sum() > 0
