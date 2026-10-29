# 🎨 Artwork Classifier

[![CI Pipeline](https://github.com/nandeesh2799/ArtworkClassifier1/actions/workflows/ci.yml/badge.svg)](https://github.com/nandeesh2799/ArtworkClassifier1/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17+-FF6F00?logo=tensorflow&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-10%20Passed-brightgreen?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)

An end-to-end artwork type classifier using **transfer learning** (EfficientNetB0 / ResNet50), a **clean dataset pipeline** (download → corrupt-image cleaning → stratified train/val/test splits), full **evaluation metrics**, and a polished **Streamlit** web application with automated containerization and offline evaluation capabilities.

## 🗂️ Project Structure

```
artworkclassifier/
├── dataset/
│   ├── raw/                # optional (links/copies)
│   ├── processed/          # stratified train/val/test folder view (links/copies)
│   └── splits/
│       ├── manifest.csv
│       ├── split_stats.csv
│       └── corrupt_images.csv
├── model/
│   ├── art_classifier.h5   # FINAL model (required output name)
│   ├── class_names.json
│   └── exports/
│       ├── best_model.keras
│       ├── test_metrics.json
│       ├── training_curves.png
│       └── confusion_matrix.png
├── app/
│   └── app.py              # Streamlit app
├── notebooks/
│   ├── art_classifier_improved.ipynb
│   └── ppt/
│       └── generate_ppt.py
├── artwork_classifier/     # reusable python modules
├── download_dataset.py
├── prepare_dataset.py
├── train.py
└── requirements.txt
```

## ✅ Mandatory Dataset Source

This project uses KaggleHub as the primary dataset source:

```python
import kagglehub
path = kagglehub.dataset_download("thedownhill/art-images-drawings-painting-sculpture-engraving")
print("Path:", path)
```

The pipeline then:
- deletes unreadable/corrupted images
- builds a unified manifest
- creates stratified train/val/test splits
- trains a transfer-learning model with ImageDataGenerator augmentation

## 🧪 Terminal Commands (Exact)

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download dataset (optional; prepare step also downloads)

```bash
python3 download_dataset.py
```

### 3) Clean + split dataset

```bash
python3 prepare_dataset.py
```

### 4) Train model (saves `model/art_classifier.h5`)

```bash
python3 train.py
```

Optional backbone:

```bash
python3 train.py --backbone resnet50
```

### 5) Run the Streamlit app

```bash
streamlit run app/app.py
```

### 6) Run the Test Suite

```bash
python3 -m pytest tests/ -v
```

### 7) Docker Containerization

Run the application inside a reproducible Docker container:

```bash
# Build and run with Docker Compose
docker compose up --build

# Access Streamlit dashboard at http://localhost:8501
```

### 8) Generate Presentation PPT

```bash
python3 notebooks/ppt/generate_ppt.py
```


---

## 👨‍💻 Author & Engineering Showcase

**Nandeesh N K**  
Full-Stack Developer & AI Systems Engineer — Bengaluru, India  
- 🌐 **Interactive Portfolio:** [NandeeshOS (Live Demo)](https://nandeesh2799-portfolio.vercel.app)
- 💼 **LinkedIn:** [linkedin.com/in/nandeesh-n-k-729773270](https://linkedin.com/in/nandeesh-n-k-729773270)
- 🐙 **GitHub Profile:** [github.com/nandeesh2799](https://github.com/nandeesh2799)
- 📫 **Direct Email:** [nandeeshnk506@gmail.com](mailto:nandeeshnk506@gmail.com)
