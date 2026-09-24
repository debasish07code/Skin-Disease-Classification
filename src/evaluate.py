from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_data
from config import *

import matplotlib.pyplot as plt
import numpy as np


print("Loading trained model...")

model = load_model(MODEL_DIR / "best_model.keras")


print("Loading dataset...")

X_train, X_test, y_train, y_test = preprocess_data()


print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\nTest Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy*100:.2f}%")


print("\nGenerating predictions...")

y_pred_prob = model.predict(X_test)

y_pred = np.argmax(y_pred_prob, axis=1)


print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=TARGET_CLASSES
    )
)


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(TARGET_CLASSES))

plt.xticks(tick_marks, TARGET_CLASSES, rotation=45)

plt.yticks(tick_marks, TARGET_CLASSES)

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.ylabel("Actual Label")

plt.xlabel("Predicted Label")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved successfully.")