# Skin Disease Classification — Eczema vs Psoriasis

Reproduction of the research paper:
> **"Enhanced Deep Learning Approach for Accurate Eczema and Psoriasis Skin Detection"**
> *Sensors, 2023*

This project reproduces the CNN-based binary classification model proposed in the paper using TensorFlow/Keras.

---

## Dataset

- **Source:** [Kaggle Skin Disease Dataset](https://www.kaggle.com/)
- **Total original images in dataset:** ~27,153 (10 disease classes)
- **Classes used:** Eczema (1,677 images) and Psoriasis (2,055 images)
- **Original images used:** 3,732
- **After augmentation:** ~6,286 images
- **Split:** 80% Training / 20% Testing

---

## Model Architecture

Sequential CNN with 5 convolutional blocks:

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

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 30
- **Batch Size:** 32
- **Input Size:** 180 × 180 × 3

---

## Augmentation

Applied offline before training to expand dataset from 3,732 → 6,286 images:
- Horizontal flip
- Rotation
- Cropping
- Gaussian blur (filtering)

---

## Project Structure

```
SkinDiseaseClassification/
├── dataset/
│   └── IMG_CLASSES/
│       ├── 1. Eczema 1677/
│       └── 7. Psoriasis pictures .../
├── models/
│   ├── best_model.keras
│   └── final_model.keras
├── outputs/
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
├── src/
│   ├── config.py
│   ├── load_dataset.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## Setup

1. Clone the repository and navigate to the project folder:
```
cd "c:/Major project/SkinDiseaseClassification"
```

2. Install dependencies:
```
pip install -r requirements.txt
```

---

## Usage

All commands should be run from the project root `SkinDiseaseClassification/`.

**Train the model:**
```
python src/train.py
```

**Evaluate the model:**
```
python src/evaluate.py
```

**Predict on a single image:**
```
python src/predict.py "C:/path/to/your/image.jpg"
```

---

## Output

After training, the following files are saved:

- `models/best_model.keras` — best model checkpoint saved during training
- `models/final_model.keras` — model saved after last epoch
- `outputs/accuracy.png` — training vs validation accuracy plot
- `outputs/loss.png` — training vs validation loss plot
- `outputs/confusion_matrix.png` — confusion matrix on test set
