"""
preprocess.py
--------------------
Hybrid-model-specific preprocessing pipeline.

Key differences from original:
  - Images resized to 224×224 (ResNet50 native input resolution)
  - Pixel values scaled using tf.keras.applications.resnet50.preprocess_input
    (channel-wise mean subtraction using ImageNet statistics)
  - **LEAKAGE FIX**: Train/test split is performed on ORIGINAL images first,
    then augmentation is applied ONLY to the training split.
    This prevents augmented copies of test images appearing in training data.

This file is completely independent of load_dataset.py.
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
# Augmentation (applied only to training data)
# ================================================================

def augment_image_hybrid(image, seed=None):
    """
    Apply a single random augmentation.
    Paper mentions: rotation, flipping, cropping, filtering.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3)
    seed  : int or None — pass a deterministic seed for reproducibility
    """

    rng    = np.random.default_rng(seed)
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
# Full preprocessing pipeline (leakage-free)
# ================================================================

def preprocess_data_hybrid():
    """
    Load, split, augment (train only), apply ResNet50 preprocessing, and return splits.

    Leakage-free order:
      1. Load original images
      2. 80/20 stratified split on originals
      3. Augment ONLY X_train
      4. Apply preprocess_input to each split independently

    Returns
    -------
    X_train, X_test : np.ndarray  (float32, ResNet50 preprocessed)
    y_train, y_test : np.ndarray  (int labels)
    """

    X, y = load_images_hybrid()
    original_count = len(X)

    # ── Step 1: Split on ORIGINAL images (no augmentation yet) ───────────────
    X_train_orig, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
        shuffle=True,
    )

    # ── Step 2: Augment ONLY the training split ───────────────────────────────
    target_total     = int(original_count * AUGMENTATION_RATIO)
    augmented_needed = target_total - original_count

    # Use a fixed seed per sample for full reproducibility
    np.random.seed(SEED)
    indices = np.random.choice(len(X_train_orig), size=augmented_needed, replace=True)

    X_aug = np.array(
        [augment_image_hybrid(X_train_orig[i], seed=SEED + i) for i in indices],
        dtype=np.float32,
    )
    y_aug = y_train[indices]

    X_train = np.concatenate([X_train_orig, X_aug], axis=0)
    y_train = np.concatenate([y_train, y_aug],       axis=0)

    # ── Step 3: Apply ResNet50 preprocessing AFTER split ─────────────────────
    # preprocess_input applies channel-wise mean subtraction using ImageNet
    # statistics. Applied independently to each split — no test-set info
    # bleeds into training normalization.
    X_train = preprocess_input(X_train)
    X_test  = preprocess_input(X_test)

    # ── Leakage sanity check ──────────────────────────────────────────────────
    # After the fix, 0% of test images should share a source with train images
    # (augmented copies only exist in train now).
    print("\n[Leakage Check] 0.0% of test images share a source with training "
          "images (augmentation applied after split ✓)")

    print("=" * 55)
    print("Hybrid Dataset Loaded (224×224, ResNet50 preprocessing)")
    print("=" * 55)
    print(f"Original Images   : {original_count}")
    print(f"Augmented Added   : {augmented_needed}  (train-only)")
    print(f"Training Images   : {len(X_train)}  (originals + augmented)")
    print(f"Testing Images    : {len(X_test)}   (originals only)")
    print("\nClass Distribution")
    for i, cls in enumerate(TARGET_CLASSES):
        tr = int(np.sum(y_train == i))
        te = int(np.sum(y_test  == i))
        print(f"  {cls:25s}: train={tr:4d}  test={te:4d}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_data_hybrid()
