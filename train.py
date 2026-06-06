from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf
import json

from artwork_classifier.evaluation import (
    evaluate_on_generator,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from artwork_classifier.modeling import ModelConfig, build_model
from artwork_classifier.training import TrainConfig, compute_class_weights, make_generators, train_two_phase
from artwork_classifier.utils import project_paths, read_json, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Artwork Classifier (transfer learning).")
    p.add_argument("--backbone", choices=["efficientnetb0", "resnet50"], default="efficientnetb0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs1", type=int, default=12)
    p.add_argument("--epochs2", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    paths = project_paths()

    manifest_path = paths.splits_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run: python3 prepare_dataset.py"
        )

    class_names_path = paths.model_dir / "class_names.json"
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Missing {class_names_path}. Run: python3 prepare_dataset.py"
        )

    manifest = pd.read_csv(manifest_path)
    class_names = read_json(class_names_path)

    tcfg = TrainConfig(
        batch_size=args.batch_size,
        seed=args.seed,
        backbone=args.backbone,
        epochs_phase1=args.epochs1,
        epochs_phase2=args.epochs2,
    )

    gens = make_generators(manifest=manifest, class_names=class_names, cfg=tcfg)
    class_weights = compute_class_weights(manifest, class_names)

    mcfg = ModelConfig(num_classes=len(class_names), backbone=args.backbone)
    model, base = build_model(mcfg)

    paths.exports_dir.mkdir(parents=True, exist_ok=True)

    h1, h2 = train_two_phase(
        model=model,
        base_model=base,
        gens=gens,
        class_weights=class_weights,
        cfg=tcfg,
        exports_dir=paths.exports_dir,
    )

    # Evaluate on test split
    metrics = evaluate_on_generator(model, gens["test"], class_names)
    save_metrics(metrics, paths.exports_dir / "test_metrics.json")

    cm = tf.convert_to_tensor(metrics["confusion_matrix"]).numpy()
    plot_confusion_matrix(cm, class_names, paths.exports_dir / "confusion_matrix.png")

    history = json.loads((paths.exports_dir / "history.json").read_text(encoding="utf-8"))
    plot_training_curves(history, paths.exports_dir / "training_curves.png")

    # Save final model in the required format/name
    out_model = paths.model_dir / "art_classifier.h5"
    model.save(out_model)
    print("Saved model:", out_model)
    print("Test metrics saved:", paths.exports_dir / "test_metrics.json")


if __name__ == "__main__":
    main()

