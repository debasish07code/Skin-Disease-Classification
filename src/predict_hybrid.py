"""
predict_hybrid.py
-----------------
Single-image inference script for the improved Hybrid CNN + ResNet50 model.

Key differences from predict.py:
  - Loads best_model_hybrid.keras  (ResNet50-based, 224×224)
  - Resizes image to 224×224       (ResNet50 native resolution)
  - Applies resnet50.preprocess_input  (channel-wise mean subtraction using
    ImageNet statistics) instead of /255.0 normalisation

Original predict.py is NOT modified.

Usage
-----
    python predict_hybrid.py <image_path>
    python predict_hybrid.py C:/path/to/skin_image.jpg

Returns
-------
    Predicted class name + confidence printed to console.
    Full probability breakdown for all 5 classes.
"""

import sys

import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

from config import *


# ================================================================
# ResNet50 native input resolution (must match preprocess_hybrid.py)
# ================================================================

HYBRID_IMAGE_HEIGHT = 224
HYBRID_IMAGE_WIDTH  = 224


def predict_image_hybrid(image_path: str):
    """
    Run inference on a single image using the best hybrid model.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the input image.

    Returns
    -------
    predicted_class : str   — name of the predicted disease class
    confidence      : float — confidence percentage (0–100)
    probabilities   : dict  — {class_name: probability_%} for all 5 classes
    Returns None on error.
    """

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / "best_model_hybrid.keras"

    if not model_path.exists():
        print(f"Error: Hybrid model not found at '{model_path}'")
        print("       Run train_hybrid.py first to generate the model.")
        return None

    model = load_model(model_path)

    # ── Load image ────────────────────────────────────────────────────────────
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        print("       Check the file path and ensure it is a valid image.")
        return None

    # ── Preprocess (must exactly match preprocess_hybrid.py pipeline) ─────────
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (HYBRID_IMAGE_WIDTH, HYBRID_IMAGE_HEIGHT))
    image = image.astype(np.float32)

    # ResNet50-specific preprocessing:
    # channel-wise mean subtraction using ImageNet statistics.
    # Using /255.0 here would cause a major distribution mismatch!
    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)   # (1, 224, 224, 3)

    # ── Predict ───────────────────────────────────────────────────────────────
    predictions     = model.predict(image, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence      = predictions[0][predicted_index] * 100
    predicted_class = TARGET_CLASSES[predicted_index]

    probabilities = {
        cls: float(predictions[0][i] * 100)
        for i, cls in enumerate(TARGET_CLASSES)
    }

    # ── Print results ─────────────────────────────────────────────────────────
    print("=" * 50)
    print("  Hybrid CNN + ResNet50 — Prediction Result")
    print("=" * 50)
    print(f"  Image           : {image_path}")
    print(f"  Predicted Class : {predicted_class}")
    print(f"  Confidence      : {confidence:.2f}%")
    print("-" * 50)
    print("  Class Probabilities:")
    for cls, prob in probabilities.items():
        bar = "█" * int(prob / 5)   # simple visual bar (max 20 chars)
        print(f"    {cls:25s}: {prob:6.2f}%  {bar}")
    print("=" * 50)

    return predicted_class, confidence, probabilities


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage   : python predict_hybrid.py <image_path>")
        print("Example : python predict_hybrid.py C:/path/to/image.jpg")
        sys.exit(1)

    result = predict_image_hybrid(sys.argv[1])

    if result is None:
        sys.exit(1)
