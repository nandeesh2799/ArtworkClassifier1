from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import tensorflow as tf


BackboneName = Literal["efficientnetb0", "resnet50"]


@dataclass(frozen=True)
class ModelConfig:
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 5
    backbone: BackboneName = "efficientnetb0"
    dropout1: float = 0.30
    dropout2: float = 0.20
    dense_units: int = 256


def get_preprocess_fn(backbone: BackboneName):
    if backbone == "resnet50":
        return tf.keras.applications.resnet.preprocess_input
    return tf.keras.applications.efficientnet.preprocess_input


def build_model(cfg: ModelConfig):
    input_shape = (*cfg.image_size, 3)

    if cfg.backbone == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    else:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = inputs

    # Note: we apply the backbone's preprocess_input in the generator/pipeline.
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(cfg.dropout1, name="dropout_1")(x)
    x = tf.keras.layers.Dense(cfg.dense_units, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(cfg.dropout2, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(cfg.num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="art_classifier")
    return model, base


def fine_tune(base_model: tf.keras.Model, unfreeze_last_n: int = 30) -> None:
    base_model.trainable = True
    if unfreeze_last_n <= 0:
        return
    for layer in base_model.layers[:-unfreeze_last_n]:
        layer.trainable = False

