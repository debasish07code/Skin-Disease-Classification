# Skin Disease Classification — DermAI

> **Reproduction & Extension of:**
> *"Enhanced Deep Learning Approach for Accurate Eczema and Psoriasis Skin Detection"*
> Sensors, 2023

This project reproduces the CNN-based classification model from the paper and extends it with an improved **Hybrid CNN + ResNet50** transfer learning pipeline, along with a full **Flask web application (DermAI)** for real-time skin disease prediction.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline 1 — Original CNN](#pipeline-1--original-cnn)
- [Pipeline 2 — Hybrid CNN--ResNet50](#pipeline-2--hybrid-cnn--resnet50)
- [DermAI Web Application](#dermai-web-application)
- [Setup](#setup)
- [Usage](#usage)
- [Outputs](#outputs)

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle Skin Disease Dataset](https://www.kaggle.com/) |
| **Total dataset images** | ~27,153 (10 disease classes) |
| **Classes used (original paper)** | Eczema (1,677) · Psoriasis (2,055) |
| **Classes used (hybrid model)** | Eczema · Psoriasis · Melanoma · Basal Cell Carcinoma · Benign Keratosis |
| **Original images** | 3,732 |
| **After augmentation** | ~6,286 |
| **Split** | 80% Training / 20% Testing |

---

## Project Structure

```
SkinDiseaseClassification/
├── app/
│   ├── app.py                     ← Flask web app (DermAI)
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/main.js
│   └── templates/
│       └── index.html
├── dataset/
│   └── IMG_CLASSES/
│       ├── 1. Eczema 1677/
│       └── 7. Psoriasis pictures .../
├── models/
│   ├── best_model.keras           ← Original CNN checkpoint
│   ├── final_model.keras
│   ├── best_model_hybrid.keras    ← Hybrid model checkpoint
│   ├── final_model_hybrid.keras
│   └── history_phase1.json        ← Phase 1 training history
├── outputs/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   ├── accuracy_hybrid.png
│   ├── loss_hybrid.png
│   └── confusion_matrix_hybrid.png
├── src/
│   ├── config.py                  ← Shared config and TARGET_CLASSES
│   ├── load_dataset.py
│   ├── preprocess.py              ← Original preprocessing (180x180)
│   ├── preprocess_hybrid.py       ← Hybrid preprocessing (224x224, ResNet50)
│   ├── model.py                   ← Original CNN architecture
│   ├── model_hybrid.py            ← Hybrid CNN + ResNet50 architecture
│   ├── train.py                   ← Original training script
│   ├── train_hybrid.py            ← Two-phase transfer learning script
│   ├── evaluate.py
│   ├── evaluate_hybrid.py
│   ├── predict.py
│   └── predict_hybrid.py
├── requirements.txt
└── README.md
```

---

## Pipeline 1 — Original CNN

Reproduction of the paper's Sequential CNN for **binary classification** (Eczema vs Psoriasis).

### Architecture

| Layer | Details |
|---|---|
| Conv2D Block 1 | 32 filters, 3×3, ReLU + MaxPooling |
| Conv2D Block 2 | 64 filters, 3×3, ReLU + MaxPooling |
| Conv2D Block 3 | 128 filters, 3×3, ReLU + MaxPooling |
| Conv2D Block 4 | 256 filters, 3×3, ReLU + MaxPooling |
| Conv2D Block 5 | 256 filters, 3×3, ReLU + MaxPooling |
| Flatten | — |
| Dense | 256 units, ReLU |
| Dropout | 0.5 |
| Output | 2 units, Softmax |

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 30 |
| Batch Size | 32 |
| Input Size | 180 × 180 × 3 |

### Augmentation (offline, before training)

- Horizontal flip
- Rotation
- Cropping
- Gaussian blur

---

## Pipeline 2 — Hybrid CNN + ResNet50

An improved model extending the original to **5-class** skin disease classification using transfer learning.

### Key Improvements

| Aspect | Original | Hybrid |
|---|---|---|
| Backbone | Custom CNN from scratch | ResNet50 (ImageNet pretrained) |
| Input resolution | 180 × 180 | **224 × 224** (ResNet50 native) |
| Classes | 2 (Eczema, Psoriasis) | **5** (+Melanoma, BCC, Benign Keratosis) |
| Training strategy | Single phase, frozen backbone | **Two-phase transfer learning** |
| Domain adaptation | None | Fine-tunes top 30 ResNet50 layers |

### Architecture

```
ResNet50 (pretrained on ImageNet, 175 layers)
    └── GlobalAveragePooling2D          → 2048-D feature vector
    └── Dense(512, ReLU)
    └── BatchNormalization
    └── Dropout(0.4)
    └── Dense(256, ReLU)
    └── BatchNormalization
    └── Dropout(0.3)
    └── Dense(5, Softmax)               → 5 disease classes
```

### Two-Phase Training

**Phase 1 — Head Warm-Up (15 epochs)**
- ResNet50 backbone fully **frozen**
- Only the classifier head is trained
- Learning Rate: `1e-3`

**Phase 2 — Fine-Tuning (20 epochs)**
- Top **30 ResNet50 layers** unfrozen
- Very small LR (`1e-5`) prevents catastrophic forgetting
- Adapts ImageNet features → dermatology domain

### Callbacks (both phases)

- `ModelCheckpoint` — saves best `val_accuracy` checkpoint
- `EarlyStopping` — stops if `val_loss` stagnates (patience=5)
- `ReduceLROnPlateau` — halves LR on plateau (patience=3, min_lr=1e-7)

---

## DermAI Web Application

A Flask-based web app that lets users upload a skin image and get an **instant AI-powered diagnosis**.

### Features

- 🖼️ **Drag-and-drop / click-to-upload** image interface
- 🔬 **Real-time prediction** using the Hybrid CNN + ResNet50 model
- 📊 **Confidence score** with probability breakdown for all 5 classes
- 💊 **Disease info cards** — description, symptoms, and severity for each condition
- ⚕️ **Medical disclaimer** for responsible AI use

### Supported Conditions

| Disease | Severity |
|---|---|
| Eczema | Chronic |
| Psoriasis | Chronic |
| Melanoma | Critical — Consult a doctor immediately |
| Basal Cell Carcinoma | High — Medical attention required |
| Benign Keratosis | Benign — Non-cancerous |

### Supported Upload Formats

JPG · JPEG · PNG · BMP · WEBP (max 10 MB)

---

## Setup

**1. Clone the repository:**
```bash
git clone https://github.com/debasish07code/Skin-Disease-Classification.git
cd Skin-Disease-Classification
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Usage

All commands should be run from the **project root** `SkinDiseaseClassification/`.

### Original CNN Pipeline

**Train:**
```bash
python src/train.py
```

**Evaluate:**
```bash
python src/evaluate.py
```

**Predict on a single image:**
```bash
python src/predict.py "C:/path/to/your/image.jpg"
```

---

### Hybrid CNN + ResNet50 Pipeline

**Train (two-phase):**
```bash
python src/train_hybrid.py
```

> To skip Phase 1 on re-runs (weights already saved), set `SKIP_PHASE1 = True` in `src/train_hybrid.py`.

**Evaluate:**
```bash
python src/evaluate_hybrid.py
```

**Predict on a single image:**
```bash
python src/predict_hybrid.py "C:/path/to/your/image.jpg"
```

---

### DermAI Web Application

**Run the app:**
```bash
python app/app.py
```

Then open your browser and visit:
```
http://127.0.0.1:5000
```

---

## Outputs

### Original CNN

| File | Description |
|---|---|
| `models/best_model.keras` | Best checkpoint (by val_accuracy) |
| `models/final_model.keras` | Model after last epoch |
| `outputs/accuracy.png` | Training vs validation accuracy |
| `outputs/loss.png` | Training vs validation loss |
| `outputs/confusion_matrix.png` | Confusion matrix on test set |

### Hybrid Model

| File | Description |
|---|---|
| `models/best_model_hybrid.keras` | Best hybrid checkpoint |
| `models/final_model_hybrid.keras` | Final hybrid model |
| `models/history_phase1.json` | Phase 1 training history (for resumable runs) |
| `outputs/accuracy_hybrid.png` | Phase 1 + Phase 2 accuracy plot |
| `outputs/loss_hybrid.png` | Phase 1 + Phase 2 loss plot |
| `outputs/confusion_matrix_hybrid.png` | 5-class confusion matrix |
