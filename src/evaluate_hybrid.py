"""
evaluate_hybrid.py
------------------
Evaluation script for the improved hybrid CNN + ResNet50 model.

Uses preprocess_data_hybrid() (224×224 + ResNet50 preprocessing) to ensure
the evaluation data matches the training preprocessing pipeline exactly.

Outputs:
    - Test loss and accuracy printed to console
    - Full classification report (per-class precision, recall, F1)
    - Confusion matrix saved → outputs/confusion_matrix_hybrid.png

Original evaluate.py is NOT modified.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from preprocess_hybrid import preprocess_data_hybrid
from config import *


# ================================================================
# Load best hybrid model
# ================================================================

print("\nLoading best hybrid model...")

model = load_model(MODEL_DIR / "best_model_hybrid.keras")

print(f"  Input shape  : {model.input_shape}")
print(f"  Output shape : {model.output_shape}")


# ================================================================
# Load dataset (224×224, ResNet50 preprocessing)
# ================================================================

print("\nLoading dataset (224×224, ResNet50 preprocessing)...")

X_train, X_test, y_train, y_test = preprocess_data_hybrid()


# ================================================================
# Evaluation
# ================================================================

print("\nEvaluating hybrid model on test set...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"\n{'=' * 40}")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"{'=' * 40}")


# ================================================================
# Classification report
# ================================================================

print("\nGenerating predictions...")

y_pred_prob = model.predict(X_test, verbose=1)
y_pred      = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report\n" + "-" * 60)
print(
    classification_report(
        y_test,
        y_pred,
        target_names=TARGET_CLASSES,
        digits=4,
    )
)


# ================================================================
# Confusion matrix
# ================================================================

cm = confusion_matrix(y_test, y_pred)

# Normalise to percentages for readability
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: raw counts
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=TARGET_CLASSES,
    yticklabels=TARGET_CLASSES,
    ax=axes[0],
    linewidths=0.5,
)
axes[0].set_title("Confusion Matrix — Hybrid Model (Counts)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("Actual Label")
axes[0].tick_params(axis="x", rotation=30)

# Right: normalised (%)
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=TARGET_CLASSES,
    yticklabels=TARGET_CLASSES,
    ax=axes[1],
    linewidths=0.5,
)
axes[1].set_title("Confusion Matrix — Hybrid Model (%)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("Actual Label")
axes[1].tick_params(axis="x", rotation=30)

plt.suptitle(
    f"Hybrid CNN + ResNet50  |  Test Accuracy: {test_accuracy * 100:.2f}%",
    fontsize=14, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_hybrid.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nConfusion matrix saved → outputs/confusion_matrix_hybrid.png")
print("\nEvaluation complete.")
