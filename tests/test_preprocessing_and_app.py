import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from app.app import preprocess, load_artifacts, DemoModel, CLASSES_PATH


def test_preprocess_generates_standard_tensor_shape():
    """Verify input PIL image is resized to 224x224x3 with batch dimension."""
    test_img = Image.new("RGBA", (450, 300), color=(255, 128, 0, 255))
    tensor = preprocess(test_img)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_preprocess_handles_grayscale_images():
    """Verify grayscale images are converted to 3 RGB channels cleanly."""
    gray_img = Image.new("L", (100, 100), color=128)
    tensor = preprocess(gray_img)

    assert tensor.shape == (1, 224, 224, 3)


def test_demo_model_produces_valid_probability_distribution():
    """Verify fallback predictor outputs proper normalized probabilities summing to 1.0."""
    classes = ["drawings", "engraving", "iconography", "painting", "sculpture"]
    model = DemoModel(classes)
    dummy_input = np.ones((1, 224, 224, 3), dtype=np.float32)

    probs = model.predict(dummy_input)
    assert probs.shape == (1, 5)
    assert np.isclose(np.sum(probs[0]), 1.0, atol=1e-5)
    assert np.all(probs[0] >= 0.0)
    assert np.all(probs[0] <= 1.0)


def test_load_artifacts_contract():
    """Verify load_artifacts returns model, classes list, and is_demo boolean."""
    model, classes, is_demo = load_artifacts()
    assert hasattr(model, "predict")
    assert isinstance(classes, list)
    assert len(classes) == 5
    assert "painting" in classes
    assert isinstance(is_demo, bool)


def test_class_names_json_schema():
    """Verify class_names.json contains expected artwork classes."""
    assert CLASSES_PATH.exists()
    data = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    expected_classes = ["drawings", "engraving", "iconography", "painting", "sculpture"]
    for cls in expected_classes:
        assert cls in data
