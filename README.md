# 🎨 Artwork Classifier

[![CI Pipeline](https://github.com/nandeesh2799/ArtworkClassifier1/actions/workflows/ci.yml/badge.svg)](https://github.com/nandeesh2799/ArtworkClassifier1/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17+-FF6F00?logo=tensorflow&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-10%20Passed-brightgreen?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end, deep learning computer vision system for classifying fine art into mediums, styles, and artistic genres (drawings, paintings, sculptures, engravings, and iconography). Employs **Transfer Learning** (EfficientNetB0 / ResNet50 backbones), automated corrupt-image cleaning heuristics, stratified multi-class splits, resilient `DemoModel` fallback inference, and an interactive **Streamlit** dashboard with Docker containerization.

---

## 🚀 Key Architectural Features

- **Transfer Learning Backbones**: Pre-trained ImageNet feature extractors (EfficientNetB0 / ResNet50) fine-tuned with dense classification heads, spatial dropout, and learning rate scheduling.
- **Automated Data Hygiene Pipeline**:
  - Automatically identifies and purges truncated, unreadable, or corrupted image files before tensor ingestion.
  - Generates stratified Train/Validation/Test splits (`70% / 15% / 15%`) to preserve class balance across minority art styles.
- **Resilient Fallback Mode (`DemoModel`)**: If heavy pre-trained weights (`.h5` / `.keras`) are absent on disk, the system seamlessly transitions to an intelligent mock inference engine providing normalized probability distributions and top-k confidence scores.
- **Production Containerization**: Multi-stage Docker build with automated healthcheck probes on Streamlit's `/_stcore/health` endpoint.
- **Automated Test Suite**: 100% passing Pytest suite verifying preprocessing tensors, grayscale expansion, probability normalization, class manifest schemas, and corrupt-image detection.

---

## 🧠 Pipeline Architecture & Workflow

```text
 ┌──────────────────────┐
 │ Raw Artwork Images   │ (Drawings, Paintings, Sculptures, Engravings)
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  Data Hygiene Engine │ ──▶ Detect & Purge Corrupt Image Tensors
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │   Stratified Split   │ ──▶ 70% Train | 15% Validation | 15% Test
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │   Transfer Learning  │ ──▶ EfficientNetB0 / ResNet50 + Data Augmentation
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │   Model Evaluator    │ ──▶ Accuracy, Macro Precision/Recall, Confusion Matrix
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ Streamlit Dashboard  │ ──▶ Drag-and-drop Image Upload + Top-K Probability Bars
 └──────────────────────┘
```

---

## 🗂️ Project Structure

```text
ArtworkClassifier1/
├── dataset/
│   ├── raw/                    # Raw source images
│   ├── processed/              # Stratified train/val/test splits
│   └── splits/
│       ├── manifest.csv        # Master dataset metadata
│       └── corrupt_images.csv  # Purged corrupted files log
├── model/
│   ├── art_classifier.h5       # Serialized production model weights
│   ├── class_names.json        # Class label mapping
│   └── exports/
│       ├── best_model.keras    # Best checkpoint export
│       ├── test_metrics.json   # Holdout test metrics
│       └── training_curves.png # Loss and accuracy plots
├── app/
│   └── app.py                  # Interactive Streamlit web application
├── artwork_classifier/         # Reusable core library
│   ├── data.py                 # Dataset download, cleaning, and stratification
│   └── utils.py                # Image validation and filesystem helpers
├── tests/
│   ├── test_dataset_and_utils.py       # Data pipeline unit tests
│   └── test_preprocessing_and_app.py  # Inference & Streamlit tests
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Container orchestration with healthchecks
├── pytest.ini                  # Pytest configuration with pythonpath
└── requirements.txt            # Python dependencies
```

---

## 🧪 Testing & Verification

Run the test suite using pytest:

```bash
# Execute unit tests
pytest -v

# Or via python module runner
python3 -m pytest tests/ -v
```

All 10 tests validate tensor shape consistency, RGB conversion from grayscale, probability sum invariants, class catalog parsing, and corrupted-file recovery.

---

## ⚙️ Quickstart & Local Setup

### 1) Environment Setup

```bash
# Clone the repository
git clone https://github.com/nandeesh2799/ArtworkClassifier1.git
cd ArtworkClassifier1

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2) Prepare Dataset & Train

```bash
# Download and clean dataset
python3 prepare_dataset.py

# Train transfer learning model (EfficientNetB0 by default)
python3 train.py

# Or train with ResNet50 backbone
python3 train.py --backbone resnet50
```

### 3) Launch Streamlit Web UI

```bash
streamlit run app/app.py
```

### 4) Docker Containerization

Run the application inside a reproducible Docker container:

```bash
# Build and start via Docker Compose
docker compose up --build

# Open browser at http://localhost:8501
```


---

## 👨‍💻 Author & Engineering Showcase

**Nandeesh N K**  
Full-Stack Developer & AI Systems Engineer — Bengaluru, India  
- 🌐 **Interactive Portfolio:** [NandeeshOS (Live Demo)](https://nandeesh2799-portfolio.vercel.app)
- 💼 **LinkedIn:** [linkedin.com/in/nandeesh-n-k-729773270](https://linkedin.com/in/nandeesh-n-k-729773270)
- 🐙 **GitHub Profile:** [github.com/nandeesh2799](https://github.com/nandeesh2799)
- 📫 **Direct Email:** [nandeeshnk506@gmail.com](mailto:nandeeshnk506@gmail.com)
