from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from artwork_classifier.data import (
    build_manifest,
    clean_corrupted_images,
    detect_layout,
    download_primary_dataset,
    materialize_splits,
    save_class_names,
    stratified_split,
)
from artwork_classifier.utils import ensure_dir, project_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download, clean and split dataset.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = project_paths()
    ensure_dir(paths.splits_dir)
    ensure_dir(paths.processed_dir)

    download_path = download_primary_dataset()
    layout = detect_layout(download_path)

    # Remove corrupted images first (this is what caused decode errors in the notebook).
    clean_corrupted_images(
        input_dirs=[layout.train_dir, layout.val_dir],
        report_path=paths.splits_dir / "corrupt_images.csv",
    )

    # Build manifest and create our own stratified splits.
    manifest = build_manifest(layout)
    manifest = stratified_split(
        manifest,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    out_manifest = paths.splits_dir / "manifest.csv"
    manifest.to_csv(out_manifest, index=False)
    print("Saved manifest:", out_manifest)

    # Materialize split folders (for quick inspection and for optional future use).
    materialize_splits(
        df=manifest,
        out_dir=paths.processed_dir,
        class_names=layout.class_names,
        link_mode=args.link_mode,
    )

    save_class_names(layout.class_names, paths.model_dir / "class_names.json")
    print("Saved class names:", paths.model_dir / "class_names.json")

    # Quick stats
    stats = manifest.groupby(["split", "label"]).size().reset_index(name="count")
    stats_path = paths.splits_dir / "split_stats.csv"
    stats.to_csv(stats_path, index=False)
    print("Saved split stats:", stats_path)


if __name__ == "__main__":
    main()

