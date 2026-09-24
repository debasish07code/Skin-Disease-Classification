import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from config import *


# Folder keyword to label mapping for 5 selected classes
CLASS_MAP = {
    "Eczema":              0,
    "Psoriasis":           1,
    "Melanoma":            2,
    "Basal Cell Carcinoma": 3,
    "Benign Keratosis":    4,
}

# Augmentation ratio kept same as paper (6286 / 3732 ~ 1.685x)
# Applied proportionally to the new total original count
AUGMENTATION_RATIO = 6286 / 3732


def load_images():
    """Load images for all 5 selected disease classes."""

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
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

            X.append(image)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


def augment_image(image):
    """
    Apply a single random augmentation to one image.
    Paper mentions: rotation, flipping, cropping, filtering.
    Exact parameters are NOT specified in the paper.
    """

    rng = np.random.default_rng()
    choice = rng.integers(0, 4)

    if choice == 0:
        # Horizontal flip (paper mentions flipping)
        return cv2.flip(image, 1)

    elif choice == 1:
        # Rotation (paper mentions rotation — exact angle NOT specified)
        angle = rng.uniform(-20, 20)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    elif choice == 2:
        # Random crop and resize back (paper mentions cropping — exact amount NOT specified)
        h, w = image.shape[:2]
        crop = rng.integers(10, 20)  # crop 10-20px from each side
        cropped = image[crop:h - crop, crop:w - crop]
        return cv2.resize(cropped, (IMAGE_WIDTH, IMAGE_HEIGHT))

    else:
        # Gaussian blur as filtering (paper mentions filtering — exact type NOT specified)
        return cv2.GaussianBlur(image, (3, 3), 0)


def preprocess_data():

    X, y = load_images()

    original_count = len(X)

    # Apply same augmentation ratio as paper (1.685x) to the new dataset
    target_total = int(original_count * AUGMENTATION_RATIO)
    augmented_needed = target_total - original_count

    np.random.seed(SEED)
    indices = np.random.choice(original_count, size=augmented_needed, replace=True)

    X_aug = np.array([augment_image(X[i]) for i in indices], dtype=np.float32)
    y_aug = y[indices]

    X_full = np.concatenate([X, X_aug], axis=0)
    y_full = np.concatenate([y, y_aug], axis=0)

    # Normalize
    X_full = X_full / 255.0

    # 80-20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=0.20,
        random_state=SEED,
        stratify=y_full,
        shuffle=True
    )

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)
    print(f"Original Images   : {original_count}")
    print(f"Augmented Added   : {augmented_needed}")
    print(f"Total Images      : {len(X_full)}")
    print(f"Training Images   : {len(X_train)}")
    print(f"Testing Images    : {len(X_test)}")
    print("\nClass Distribution (full dataset)")
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"{cls:25s}: {np.sum(y_full == i)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    preprocess_data()