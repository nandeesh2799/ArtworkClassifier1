from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from .utils import ensure_dir, write_json


def evaluate_on_generator(model, gen, class_names: List[str]) -> Dict:
    gen.reset()
    probs = model.predict(gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = gen.classes

    acc = float(np.mean(y_pred == y_true))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_training_curves(history_json: Dict, out_path: Path) -> None:
    ensure_dir(out_path.parent)

    h1 = history_json.get("phase1", {})
    h2 = history_json.get("phase2", {})

    def _cat(k: str):
        return list(h1.get(k, [])) + list(h2.get(k, []))

    acc = _cat("accuracy")
    val_acc = _cat("val_accuracy")
    loss = _cat("loss")
    val_loss = _cat("val_loss")

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_metrics(metrics: Dict, out_path: Path) -> None:
    write_json(out_path, metrics)

