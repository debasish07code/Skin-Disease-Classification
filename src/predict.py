import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import *


def predict_image(image_path):

    # Load model
    model = load_model(MODEL_DIR / "best_model.keras")

    # Load and preprocess image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(image, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100

    predicted_class = TARGET_CLASSES[predicted_index]

    print("=" * 40)
    print(f"Image            : {image_path}")
    print(f"Predicted Class  : {predicted_class}")
    print(f"Confidence       : {confidence:.2f}%")
    print("-" * 40)
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"{cls:25s}: {predictions[0][i] * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py C:/path/to/image.jpg")
        sys.exit(1)

    predict_image(sys.argv[1])
