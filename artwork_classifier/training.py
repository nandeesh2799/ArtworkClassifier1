from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .modeling import ModelConfig, fine_tune, get_preprocess_fn
from .utils import ensure_dir, write_json


@dataclass(frozen=True)
class TrainConfig:
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    seed: int = 42
    backbone: str = "efficientnetb0"
    initial_lr: float = 1e-3
    fine_tune_lr: float = 1e-4
    epochs_phase1: int = 12
    epochs_phase2: int = 10
    fine_tune_last_n: int = 30


def make_generators(
    manifest: pd.DataFrame,
    class_names: List[str],
    cfg: TrainConfig,
) -> Dict[str, tf.keras.utils.Sequence]:
    preprocess_fn = get_preprocess_fn(cfg.backbone)  # includes normalization

    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
    )

    test_aug = ImageDataGenerator(preprocessing_function=preprocess_fn)

    def _flow(split: str, shuffle: bool):
        df = manifest[manifest["split"] == split].copy()
        return (train_aug if split == "train" else test_aug).flow_from_dataframe(
            dataframe=df,
            x_col="filepath",
            y_col="label",
            target_size=cfg.image_size,
            color_mode="rgb",
            classes=class_names,
            class_mode="categorical",
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            seed=cfg.seed,
        )

    return {
        "train": _flow("train", shuffle=True),
        "val": _flow("val", shuffle=False),
        "test": _flow("test", shuffle=False),
    }


def compute_class_weights(manifest: pd.DataFrame, class_names: List[str]) -> Dict[int, float]:
    train = manifest[manifest["split"] == "train"]
    y = train["label"].map({c: i for i, c in enumerate(class_names)}).to_numpy()
    classes = np.arange(len(class_names))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {i: float(w) for i, w in enumerate(weights)}


def make_callbacks(exports_dir: Path) -> List[tf.keras.callbacks.Callback]:
    exports_dir = ensure_dir(exports_dir)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(exports_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]


def train_two_phase(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    gens: Dict[str, tf.keras.utils.Sequence],
    class_weights: Dict[int, float],
    cfg: TrainConfig,
    exports_dir: Path,
) -> Tuple[tf.keras.callbacks.History, tf.keras.callbacks.History]:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.initial_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    h1 = model.fit(
        gens["train"],
        validation_data=gens["val"],
        epochs=cfg.epochs_phase1,
        callbacks=make_callbacks(exports_dir),
        class_weight=class_weights,
        verbose=1,
    )

    fine_tune(base_model, unfreeze_last_n=cfg.fine_tune_last_n)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    h2 = model.fit(
        gens["train"],
        validation_data=gens["val"],
        epochs=cfg.epochs_phase1 + cfg.epochs_phase2,
        initial_epoch=len(h1.history["loss"]),
        callbacks=make_callbacks(exports_dir),
        class_weight=class_weights,
        verbose=1,
    )

    # Save training curves data
    hist = {"phase1": h1.history, "phase2": h2.history}
    write_json(exports_dir / "history.json", hist)
    return h1, h2

