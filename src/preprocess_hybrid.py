"""
preprocess_hybrid.py
--------------------
Hybrid-model-specific preprocessing pipeline.

Key differences from preprocess.py:
  - Images resized to 224×224 (ResNet50 native input resolution)
  - Pixel values scaled using tf.keras.applications.resnet50.preprocess_input
    (channel-wise mean subtraction using ImageNet statistics) instead of /255.0
  - Augmentation logic and train/test split strategy are identical to preprocess.py

This file is completely independent of preprocess.py.
Original preprocess.py is NOT modified.
"""

import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input

from config import *


# ================================================================
# ResNet50 native input resolution
# ================================================================

HYBRID_IMAGE_HEIGHT = 224
HYBRID_IMAGE_WIDTH  = 224

# Folder keyword → integer label mapping (same 5 classes as main pipeline)
CLASS_MAP = {
    "Eczema":               0,
    "Psoriasis":            1,
    "Melanoma":             2,
    "Basal Cell Carcinoma": 3,
    "Benign Keratosis":     4,
}

# Same augmentation ratio as the paper (6286 / 3732 ≈ 1.685×)
AUGMENTATION_RATIO = 6286 / 3732


# ================================================================
# Image loading (224×224)
# ================================================================

def load_images_hybrid():
    """Load and resize images to 224×224 for all 5 disease classes."""

    X = []
    y = []

    for folder_name in os.listdir(DATASET_PATH):

        folder_path = os.path.join(DATASET_PATH, folder_name)

        if not os.path.isdir(folder_path):
            continue

        label = None
        for keyword, lbl in CLASS_MAP.items():
            if keyword in folder_name:
                label = lbl
                break

        if label is None:
            continue

        for image_name in os.listdir(folder_path):

            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (HYBRID_IMAGE_WIDTH, HYBRID_IMAGE_HEIGHT))

            X.append(image)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


# ================================================================
# Augmentation (identical strategy to preprocess.py)
# ================================================================

def augment_image_hybrid(image):
    """
    Apply a single random augmentation.
    Paper mentions: rotation, flipping, cropping, filtering.
    Exact parameters are NOT specified in the paper.
    """

    rng    = np.random.default_rng()
    choice = rng.integers(0, 4)

    if choice == 0:
        # Horizontal flip
        return cv2.flip(image, 1)

    elif choice == 1:
        # Random rotation (-20° to +20°)
        angle = rng.uniform(-20, 20)
        h, w  = image.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    elif choice == 2:
        # Random crop and resize back
        h, w  = image.shape[:2]
        crop  = rng.integers(10, 20)
        cropped = image[crop:h - crop, crop:w - crop]
        return cv2.resize(cropped, (HYBRID_IMAGE_WIDTH, HYBRID_IMAGE_HEIGHT))

    else:
        # Gaussian blur
        return cv2.GaussianBlur(image, (3, 3), 0)


# ================================================================
# Full preprocessing pipeline
# ================================================================

def preprocess_data_hybrid():
    """
    Load, augment, apply ResNet50 preprocessing, and split the dataset.

    Returns
    -------
    X_train, X_test : np.ndarray  (float32, ResNet50 preprocessed)
    y_train, y_test : np.ndarray  (int labels)
    """

    X, y = load_images_hybrid()

    original_count = len(X)

    # Apply same augmentation ratio as paper (≈ 1.685×)
    target_total      = int(original_count * AUGMENTATION_RATIO)
    augmented_needed  = target_total - original_count

    np.random.seed(SEED)
    indices = np.random.choice(original_count, size=augmented_needed, replace=True)

    X_aug = np.array(
        [augment_image_hybrid(X[i]) for i in indices],
        dtype=np.float32
    )
    y_aug = y[indices]

    X_full = np.concatenate([X, X_aug],   axis=0)
    y_full = np.concatenate([y, y_aug],   axis=0)

    # ── ResNet50-specific preprocessing ──────────────────────────────────────
    # preprocess_input applies channel-wise mean subtraction using ImageNet
    # statistics. This is REQUIRED for ResNet50 pretrained weights to work
    # correctly. Using /255.0 instead causes a major distribution mismatch.
    X_full = preprocess_input(X_full)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=0.20,
        random_state=SEED,
        stratify=y_full,
        shuffle=True
    )

    print("=" * 55)
    print("Hybrid Dataset Loaded (224×224, ResNet50 preprocessing)")
    print("=" * 55)
    print(f"Original Images   : {original_count}")
    print(f"Augmented Added   : {augmented_needed}")
    print(f"Total Images      : {len(X_full)}")
    print(f"Training Images   : {len(X_train)}")
    print(f"Testing Images    : {len(X_test)}")
    print("\nClass Distribution (full dataset)")
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"  {cls:25s}: {np.sum(y_full == i)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_data_hybrid()
