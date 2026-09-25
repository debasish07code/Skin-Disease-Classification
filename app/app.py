"""
app.py
------
Flask web application for Skin Disease Classification
using the Hybrid CNN + ResNet50 model.

Run from project root:
    python app/app.py
    Visit: http://127.0.0.1:5000
"""

import sys
import os

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

from config import MODEL_DIR, TARGET_CLASSES

# ── Disease info cards ────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Eczema": {
        "description": "A chronic inflammatory skin condition causing itchy, red, and cracked patches. Often triggered by allergens or irritants.",
        "symptoms": ["Itchy skin", "Red or brownish-gray patches", "Thickened or cracked skin", "Raw or swollen skin"],
        "icon": "🔴",
        "color": "#FF6B6B",
        "severity": "Chronic",
    },
    "Psoriasis": {
        "description": "An autoimmune condition that causes rapid skin cell buildup, resulting in scaling, inflammation, and red patches.",
        "symptoms": ["Red patches covered with silvery scales", "Dry cracked skin", "Itching or burning", "Thickened nails"],
        "icon": "🟠",
        "color": "#FF9F43",
        "severity": "Chronic",
    },
    "Melanoma": {
        "description": "The most serious type of skin cancer, developing in melanocytes. Early detection is critical for successful treatment.",
        "symptoms": ["Unusual mole or growth", "Changes in existing mole", "Dark lesion on skin", "Irregular borders"],
        "icon": "⚫",
        "color": "#A29BFE",
        "severity": "Critical — Consult a doctor immediately",
    },
    "Basal Cell Carcinoma": {
        "description": "The most common form of skin cancer. Rarely spreads but can cause significant local damage if left untreated.",
        "symptoms": ["Pearly or waxy bump", "Flat flesh-colored lesion", "Bleeding or scabbing sore", "Pink growth"],
        "icon": "🟣",
        "color": "#FD79A8",
        "severity": "High — Medical attention required",
    },
    "Benign Keratosis": {
        "description": "A non-cancerous skin growth that appears as a waxy, scaly, slightly raised patch. Very common in older adults.",
        "symptoms": ["Waxy, scaly patch", "Varies from white to black", "Slightly raised surface", "Round or oval shape"],
        "icon": "🟡",
        "color": "#FDCB6E",
        "severity": "Benign — Non-cancerous",
    },
}

# ── Constants ─────────────────────────────────────────────────────────────────
HYBRID_IMAGE_HEIGHT = 224
HYBRID_IMAGE_WIDTH  = 224
MAX_FILE_SIZE_MB    = 10

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

# ── Load model once at startup ────────────────────────────────────────────────
print("\nLoading Hybrid CNN + ResNet50 model...")
model_path = MODEL_DIR / "best_model_hybrid.keras"
MODEL = load_model(str(model_path))
print(f"  Model loaded  : {model_path.name}")
print(f"  Input shape   : {MODEL.input_shape}")
print(f"  Output classes: {len(TARGET_CLASSES)}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Decode, resize, and apply ResNet50 preprocessing to uploaded image bytes."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG file.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (HYBRID_IMAGE_WIDTH, HYBRID_IMAGE_HEIGHT))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)


def image_to_base64(file_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Convert raw image bytes to a base64 data-URI for display."""
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"Unsupported file type '{ext}'. Use JPG, PNG, BMP, or WEBP."}), 400

    try:
        file_bytes = file.read()

        # Preprocess & predict
        img_tensor  = preprocess_image(file_bytes)
        predictions = MODEL.predict(img_tensor, verbose=0)[0]
        pred_index  = int(np.argmax(predictions))
        pred_class  = TARGET_CLASSES[pred_index]
        confidence  = float(predictions[pred_index] * 100)

        # Build sorted probability list
        probabilities = [
            {"class": TARGET_CLASSES[i], "prob": round(float(predictions[i] * 100), 2)}
            for i in range(len(TARGET_CLASSES))
        ]
        probabilities.sort(key=lambda x: x["prob"], reverse=True)

        # MIME type for base64
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".bmp": "image/bmp", ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/jpeg")

        return jsonify({
            "predicted_class": pred_class,
            "confidence":      round(confidence, 2),
            "probabilities":   probabilities,
            "image_data":      image_to_base64(file_bytes, mime),
            "disease_info":    DISEASE_INFO.get(pred_class, {}),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  DermAI — Skin Disease Classification")
    print("  Visit: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=5000)
