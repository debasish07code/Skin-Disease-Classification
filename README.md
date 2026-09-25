# Skin Disease Classification — DermAI

> **Reproduction & Extension of:**
> *"Enhanced Deep Learning Approach for Accurate Eczema and Psoriasis Skin Detection"*
> Sensors, 2023

This project reproduces and extends the CNN-based classification model from the paper.
The original paper addressed binary classification (Eczema vs Psoriasis), but **both pipelines in
this implementation perform 5-class multiclass classification** across a broader set of skin diseases.
A **Flask web application (DermAI)** is also included for real-time skin disease prediction.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline 1 — Custom CNN (Multiclass)](#pipeline-1--custom-cnn-multiclass)
- [Pipeline 2 — Hybrid CNN + ResNet50 (Multiclass)](#pipeline-2--hybrid-cnn--resnet50-multiclass)
- [DermAI Web Application](#dermai-web-application)
- [Setup](#setup)
- [Usage](#usage)
- [Outputs](#outputs)

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle Skin Disease Dataset](https://www.kaggle.com/) |
| **Total dataset images** | ~27,153 (10 disease classes available) |
| **Classes selected** | 5 (see below) |
| **Split** | 80% Training / 20% Testing |

### Classes Used

| # | Class | Original Images |
|---|---|---|
| 0 | Eczema | 1,677 |
| 1 | Psoriasis | 2,055 |
| 2 | Melanoma | — |
| 3 | Basal Cell Carcinoma | — |
| 4 | Benign Keratosis | — |

### Augmentation (offline, before training)

Applied to maintain the same ~1.685× augmentation ratio as the paper (3,732 → ~6,286 images):

| Technique | Details |
|---|---|
| Horizontal flip | Left-right mirror |
| Rotation | Random angle in range [−20°, +20°] |
| Random crop | 10–20 px from each side, resized back |
| Gaussian blur | Kernel size 3×3 |

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
│       ├── 7. Psoriasis pictures .../
│       └── ...
├── models/
│   ├── best_model.keras           ← Custom CNN best checkpoint
│   ├── final_model.keras
│   ├── best_model_hybrid.keras    ← Hybrid model best checkpoint
│   ├── final_model_hybrid.keras
│   └── history_phase1.json        ← Phase 1 training history (hybrid)
├── outputs/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   ├── accuracy_hybrid.png
│   ├── loss_hybrid.png
│   └── confusion_matrix_hybrid.png
├── src/
│   ├── config.py                  ← Shared config and TARGET_CLASSES (5 classes)
│   ├── load_dataset.py
│   ├── preprocess.py              ← Custom CNN preprocessing (180×180, normalized)
│   ├── preprocess_hybrid.py       ← Hybrid preprocessing (224×224, ResNet50 scaled)
│   ├── model.py                   ← Custom CNN architecture
│   ├── model_hybrid.py            ← Hybrid CNN + ResNet50 architecture
│   ├── train.py                   ← Custom CNN training script
│   ├── train_hybrid.py            ← Two-phase transfer learning script
│   ├── evaluate.py
│   ├── evaluate_hybrid.py
│   ├── predict.py
│   └── predict_hybrid.py
├── requirements.txt
└── README.md
```

---

## Pipeline 1 — Custom CNN (Multiclass)

A from-scratch Sequential CNN based on the paper's architecture, **extended to 5-class multiclass classification**.

> The original paper used this architecture for binary (Eczema vs Psoriasis) detection.
> In this implementation, the same architecture is applied across all **5 disease classes**
> using `len(TARGET_CLASSES)` as the output size, making it a true multiclass classifier.

### Architecture

| Layer | Details |
|---|---|
| Conv2D Block 1 | 32 filters, 3×3, ReLU, padding=same + MaxPooling 2×2 |
| Conv2D Block 2 | 64 filters, 3×3, ReLU, padding=same + MaxPooling 2×2 |
| Conv2D Block 3 | 128 filters, 3×3, ReLU, padding=same + MaxPooling 2×2 |
| Conv2D Block 4 | 256 filters, 3×3, ReLU, padding=same + MaxPooling 2×2 |
| Conv2D Block 5 | 256 filters, 3×3, ReLU, padding=same + MaxPooling 2×2 |
| Flatten | — |
| Dense | 256 units, ReLU |
| Dropout | 0.5 |
| Output | **5 units, Softmax** (Eczema / Psoriasis / Melanoma / BCC / Benign Keratosis) |

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 30 |
| Batch Size | 32 |
| Input Size | 180 × 180 × 3 |
| Normalization | Pixel values scaled to [0, 1] |

---

## Pipeline 2 — Hybrid CNN + ResNet50 (Multiclass)

An improved pipeline using ResNet50 as a pretrained backbone with a two-phase transfer learning strategy,
also performing **5-class multiclass classification**.

### Key Differences vs Pipeline 1

| Aspect | Pipeline 1 (Custom CNN) | Pipeline 2 (Hybrid) |
|---|---|---|
| Backbone | Custom CNN (from scratch) | ResNet50 (ImageNet pretrained) |
| Input resolution | 180 × 180 | **224 × 224** (ResNet50 native) |
| Preprocessing | Normalize [0, 1] | ResNet50 `preprocess_input` |
| Classifier head | Dense(256) → Dropout(0.5) | Dense(512) → BN → Dropout(0.4) → Dense(256) → BN → Dropout(0.3) |
| Training strategy | Single phase | **Two-phase transfer learning** |
| Domain adaptation | None | Fine-tunes top 30 ResNet50 layers |
| Output | 5-class Softmax | 5-class Softmax |

### Architecture

```
ResNet50 (pretrained on ImageNet, backbone frozen in Phase 1)
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

- ResNet50 backbone fully **frozen** (175 layers locked)
- Only the custom classifier head is trained
- Learning Rate: `1e-3`
- Safe to use a high LR since pretrained weights are untouched

**Phase 2 — Fine-Tuning (20 epochs)**

- Top **30 ResNet50 layers** unfrozen
- Very small LR (`1e-5`) to prevent catastrophic forgetting
- Adapts ImageNet features → dermatology domain

### Callbacks (both phases)

| Callback | Monitor | Config |
|---|---|---|
| `ModelCheckpoint` | `val_accuracy` | Saves best checkpoint |
| `EarlyStopping` | `val_loss` | patience=5, restores best weights |
| `ReduceLROnPlateau` | `val_loss` | factor=0.5, patience=3, min_lr=1e-7 |

---

## DermAI Web Application

A Flask-based web app that lets users upload a skin image and get an **instant AI-powered diagnosis** using the Hybrid CNN + ResNet50 model.

### Features

- 🖼️ **Drag-and-drop / click-to-upload** image interface
- 🔬 **Real-time prediction** across all 5 disease classes
- 📊 **Confidence score** with full probability breakdown
- 💊 **Disease info cards** — description, symptoms, and severity per class
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

JPG · JPEG · PNG · BMP · WEBP &nbsp;(max 10 MB)

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

### Pipeline 1 — Custom CNN

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

### Pipeline 2 — Hybrid CNN + ResNet50

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

```bash
python app/app.py
```

Then open your browser and visit:
```
http://127.0.0.1:5000
```

---

## Outputs

### Pipeline 1 — Custom CNN

| File | Description |
|---|---|
| `models/best_model.keras` | Best checkpoint saved during training |
| `models/final_model.keras` | Model after the last epoch |
| `outputs/accuracy.png` | Training vs validation accuracy |
| `outputs/loss.png` | Training vs validation loss |
| `outputs/confusion_matrix.png` | 5-class confusion matrix on test set |

### Pipeline 2 — Hybrid Model

| File | Description |
|---|---|
| `models/best_model_hybrid.keras` | Best hybrid checkpoint |
| `models/final_model_hybrid.keras` | Final hybrid model |
| `models/history_phase1.json` | Phase 1 training history (for resumable runs) |
| `outputs/accuracy_hybrid.png` | Phase 1 + Phase 2 combined accuracy plot |
| `outputs/loss_hybrid.png` | Phase 1 + Phase 2 combined loss plot |
| `outputs/confusion_matrix_hybrid.png` | 5-class confusion matrix on test set |
