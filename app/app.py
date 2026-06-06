from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "art_classifier.h5"
CLASSES_PATH = ROOT / "model" / "class_names.json"


def inject_css() -> None:
    st.markdown(
        """
<style>
/* Hide Streamlit default UI elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Try to hide top-right toolbar/menu (best effort; may change across versions) */
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {visibility: hidden;}
[data-testid="stStatusWidget"] {visibility: hidden;}

.app-title {
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: 0.2px;
  margin-bottom: 0.25rem;
}
.app-subtitle {
  color: #6b7280;
  margin-bottom: 1.25rem;
}
.card {
  background: rgba(255,255,255,0.9);
  border: 1px solid rgba(17,24,39,0.08);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(17,24,39,0.08);
}
.pill {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
  color: white;
  font-weight: 700;
  font-size: 0.9rem;
}
.muted {
  color: #6b7280;
}
.footer-custom {
  margin-top: 2.5rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(17,24,39,0.08);
  color: #6b7280;
  text-align: center;
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}. Train first (python3 train.py).")
    model = tf.keras.models.load_model(MODEL_PATH)

    if CLASSES_PATH.exists():
        class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    else:
        class_names = ["drawings", "engraving", "iconography", "painting", "sculpture"]
    return model, class_names


def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)  # normalization
    arr = np.expand_dims(arr, axis=0)
    return arr


def main() -> None:
    st.set_page_config(page_title="🎨 Artwork Classifier", page_icon="🎨", layout="centered")
    inject_css()

    st.markdown('<div class="app-title">🎨 Artwork Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Upload an artwork image and get the predicted class with confidence.</div>',
        unsafe_allow_html=True,
    )

    model, class_names = load_artifacts()

    with st.sidebar:
        st.markdown("### About")
        st.write("Transfer Learning model (EfficientNet/ResNet) trained on an art dataset.")
        st.markdown("### Classes")
        st.write(", ".join(class_names))

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded is None:
        st.markdown('<div class="card muted">Tip: use clear, centered artwork images for best results.</div>', unsafe_allow_html=True)
        st.markdown('<div class="footer-custom">Final Year Project</div>', unsafe_allow_html=True)
        return

    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    x = preprocess(image)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    pred = class_names[idx] if idx < len(class_names) else str(idx)
    conf = float(probs[idx])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'Prediction: <span class="pill">{pred}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="muted">Confidence: <b>{conf:.2%}</b></div>', unsafe_allow_html=True)
    st.progress(int(conf * 100))

    st.markdown("#### All class probabilities")
    for c, p in sorted(zip(class_names, probs), key=lambda t: t[1], reverse=True):
        st.write(f"{c}: {float(p):.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="footer-custom">Final Year Project</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

