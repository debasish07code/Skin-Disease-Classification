"""
model_hybrid.py
---------------
Improved Hybrid CNN + ResNet50 architecture for 5-class skin disease
classification.

Key improvements over the original hybrid model:
  1. Input size changed to 224×224 — ResNet50's native resolution.
     The original 180×180 caused significant feature-map under-extraction.
  2. Deeper classifier head with BatchNormalization for stable training.
  3. Two-stage transfer learning:
       - Stage 1 : ResNet50 fully frozen → train head only (warm-up)
       - Stage 2 : Top 30 ResNet50 layers unfrozen → fine-tune at low LR
     The original model kept the backbone frozen throughout, which prevented
     any domain adaptation from ImageNet → dermatology.
  4. Model is compiled here for Stage 1; fine_tune_model() recompiles for
     Stage 2 with a very small learning rate to avoid catastrophic forgetting.

Original model.py is NOT modified.
"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import *

# ================================================================
# ResNet50 native input size (must match preprocess_hybrid.py)
# ================================================================

HYBRID_IMAGE_HEIGHT = 224
HYBRID_IMAGE_WIDTH  = 224


def create_hybrid_model():
    """
    Build the improved hybrid model (Stage 1 — head-only training).

    Architecture
    ------------
    ResNet50 (pretrained, frozen)  →  2048-D feature vector
    GlobalAveragePooling2D
    Dense(512, relu) → BN → Dropout(0.4)
    Dense(256, relu) → BN → Dropout(0.3)
    Dense(5, softmax)

    Returns
    -------
    model : tf.keras.Model  (compiled, ready for Stage 1 training)
    """

    # ── Backbone ──────────────────────────────────────────────────────────────
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(HYBRID_IMAGE_HEIGHT, HYBRID_IMAGE_WIDTH, CHANNELS)
    )

    # Freeze ALL backbone layers for Stage 1 (head warm-up)
    base_model.trainable = False

    # ── Classifier head ───────────────────────────────────────────────────────
    x = base_model.output
    x = GlobalAveragePooling2D()(x)              # 2048-D vector

    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    output = Dense(len(TARGET_CLASSES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    # ── Compile for Stage 1 ───────────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def fine_tune_model(model, unfreeze_last_n_layers: int = 30):
    """
    Prepare the model for Stage 2: fine-tuning.

    Because create_hybrid_model() connects ResNet50 via base_model.input /
    base_model.output (not by calling base_model as a layer), Keras inlines
    all 175 ResNet50 layers directly into model.layers — there is no nested
    sub-model to retrieve.

    Strategy
    --------
    1. Locate the GlobalAveragePooling2D layer — it is the first custom head
       layer, so everything before it belongs to the ResNet50 backbone.
    2. Freeze the entire model.
    3. Unfreeze the last `unfreeze_last_n_layers` backbone layers.
    4. Always keep all head layers trainable.
    5. Recompile at 1e-5 to avoid catastrophic forgetting.

    Parameters
    ----------
    model               : tf.keras.Model returned by create_hybrid_model()
    unfreeze_last_n_layers : int  (default 30 — last residual block)

    Returns
    -------
    model : tf.keras.Model  (recompiled for Stage 2)
    """

    # ── Find the backbone / head boundary ────────────────────────────────────
    # The custom head starts at GlobalAveragePooling2D.
    gap_idx = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, GlobalAveragePooling2D):
            gap_idx = i
            break

    if gap_idx is None:
        raise ValueError(
            "Could not locate GlobalAveragePooling2D in model.layers. "
            "Ensure create_hybrid_model() includes a GlobalAveragePooling2D layer."
        )

    # backbone = everything from InputLayer up to (not including) GAP
    # head     = GAP + Dense + BN + Dropout + Dense + BN + Dropout + Dense
    backbone_layers = model.layers[1:gap_idx]   # skip InputLayer
    head_layers     = model.layers[gap_idx:]

    # ── Freeze everything first ───────────────────────────────────────────────
    for layer in model.layers:
        layer.trainable = False

    # ── Unfreeze last N backbone layers ───────────────────────────────────────
    for layer in backbone_layers[-unfreeze_last_n_layers:]:
        layer.trainable = True

    # ── Always keep head trainable ────────────────────────────────────────────
    for layer in head_layers:
        layer.trainable = True

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"\n[Fine-tune] Trainable layers : {trainable_count}  "
          f"(last {unfreeze_last_n_layers} backbone + {len(head_layers)} head)")

    # ── Recompile at a very low LR ────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    m = create_hybrid_model()
    m.summary()
    print(f"\nTotal params        : {m.count_params():,}")
    print(f"Trainable params    : {sum(w.numpy().size for w in m.trainable_weights):,}")
