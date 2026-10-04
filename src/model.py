"""
model.py
--------
Hybrid CNN + ResNet50 architecture for 5-class skin disease classification.

Architecture matches the project flow diagram:

  Input (224×224×3)
      │
      ├──► ResNet50 Branch (pre-trained, ImageNet)
      │         └── GlobalAveragePooling2D → 2048-D features
      │
      └──► Custom CNN Branch
                Conv Block 1 (64  filters) → BN → ReLU → MaxPool
                Conv Block 2 (128 filters) → BN → ReLU → MaxPool
                Conv Block 3 (256 filters) → BN → ReLU → MaxPool
                Conv Block 4 (512 filters) → BN → ReLU → MaxPool
                GlobalAveragePooling2D     → 512-D features
                │
  Concatenate (2048 + 512 = 2560-D)
      │
  Classification Head
      Dense(512, relu) → BatchNorm → Dropout(0.4)
      Dense(256, relu) → Dropout(0.3)
      Dense(N, softmax)

Two-phase transfer learning:
  Phase 1 — ResNet50 fully frozen, only head + CNN branch trained (LR=1e-3)
  Phase 2 — Top 30 ResNet50 layers unfrozen, fine-tuned at LR=1e-5
"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import *


# ================================================================
# ResNet50 native input size (must match preprocess.py)
# ================================================================

HYBRID_IMAGE_HEIGHT = 224
HYBRID_IMAGE_WIDTH  = 224


# ================================================================
# Custom CNN branch helper
# ================================================================

def _cnn_branch(inputs):
    """
    4-block custom CNN branch.

    Conv Block 1 :  64 filters  → BN → ReLU → MaxPool
    Conv Block 2 : 128 filters  → BN → ReLU → MaxPool
    Conv Block 3 : 256 filters  → BN → ReLU → MaxPool
    Conv Block 4 : 512 filters  → BN → ReLU → MaxPool
    GlobalAveragePooling2D      → 512-D feature vector
    """

    def conv_block(x, filters):
        x = Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2))(x)
        return x

    x = conv_block(inputs,  64)   # Conv Block 1
    x = conv_block(x,      128)   # Conv Block 2
    x = conv_block(x,      256)   # Conv Block 3
    x = conv_block(x,      512)   # Conv Block 4

    x = GlobalAveragePooling2D()(x)   # 512-D features
    return x


# ================================================================
# Full hybrid model
# ================================================================

def create_hybrid_model():
    """
    Build the hybrid model (Phase 1 — head-only + CNN branch training).

    Returns
    -------
    model : tf.keras.Model  (compiled, ready for Phase 1 training)
    """

    inputs = Input(shape=(HYBRID_IMAGE_HEIGHT, HYBRID_IMAGE_WIDTH, CHANNELS),
                   name="input_image")

    # ── Branch A: ResNet50 (pre-trained, frozen in Phase 1) ──────────────────
    resnet = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    resnet.trainable = False   # Freeze ALL ResNet50 layers for Phase 1

    resnet_features = GlobalAveragePooling2D(name="resnet_gap")(resnet.output)
    # resnet_features shape: (batch, 2048)

    # ── Branch B: Custom CNN (always trainable) ───────────────────────────────
    cnn_features = _cnn_branch(inputs)
    # cnn_features shape: (batch, 512)

    # ── Feature Fusion: Concatenate (2048 + 512 = 2560-D) ────────────────────
    fused = Concatenate(name="feature_fusion")([resnet_features, cnn_features])

    # ── Classification Head ───────────────────────────────────────────────────
    x = Dense(512, activation="relu", name="head_dense_512")(fused)
    x = BatchNormalization(name="head_bn_512")(x)
    x = Dropout(0.4, name="head_drop_512")(x)

    x = Dense(256, activation="relu", name="head_dense_256")(x)
    x = Dropout(0.3, name="head_drop_256")(x)

    output = Dense(len(TARGET_CLASSES), activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=output, name="HybridCNN_ResNet50")

    # ── Compile for Phase 1 ───────────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ================================================================
# Phase 2 fine-tuning
# ================================================================

def fine_tune_model(model, unfreeze_last_n_layers: int = 30):
    """
    Prepare the model for Phase 2: fine-tuning top ResNet50 layers.

    Strategy
    --------
    1. Freeze everything.
    2. Unfreeze the last `unfreeze_last_n_layers` ResNet50 layers.
    3. Always keep the custom CNN branch and head trainable.
    4. Recompile at 1e-5 to avoid catastrophic forgetting.

    Parameters
    ----------
    model                  : tf.keras.Model from create_hybrid_model()
    unfreeze_last_n_layers : int (default 30 — last residual block)

    Returns
    -------
    model : tf.keras.Model  (recompiled for Phase 2)
    """

    # Collect ResNet50 layers by name (they are the non-head, non-CNN layers)
    resnet_layers = [
        l for l in model.layers
        if not l.name.startswith(("head_", "output", "feature_fusion",
                                  "re_lu", "max_pooling2d", "batch_normalization_",
                                  "conv2d_", "resnet_gap", "input_image"))
        and "conv2d" not in l.name.lower()
        and "max_pooling" not in l.name.lower()
        and "re_lu" not in l.name.lower()
    ]

    # Simpler and more robust: freeze ALL, then selectively unfreeze
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze last N ResNet50 layers — identify them by the resnet50 model
    # All ResNet50 layers have names that DON'T start with our custom prefixes
    custom_prefixes = ("head_", "output", "feature_fusion",
                       "resnet_gap", "input_image",
                       # CNN branch layers (added by _cnn_branch):
                       "conv2d", "batch_normalization", "re_lu", "max_pooling2d")

    backbone_layers = [
        l for l in model.layers
        if not any(l.name.startswith(p) for p in custom_prefixes)
    ]

    # Unfreeze last N backbone layers
    for layer in backbone_layers[-unfreeze_last_n_layers:]:
        layer.trainable = True

    # Always keep custom CNN branch + head trainable
    for layer in model.layers:
        if any(layer.name.startswith(p) for p in
               ("head_", "output", "feature_fusion", "resnet_gap",
                "conv2d", "batch_normalization", "re_lu", "max_pooling2d")):
            layer.trainable = True

    trainable = sum(1 for l in model.layers if l.trainable)
    total     = len(model.layers)
    print(f"\n[Fine-tune] Trainable layers: {trainable} / {total}  "
          f"(last {unfreeze_last_n_layers} ResNet50 + full CNN branch + head)")

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    m = create_hybrid_model()
    m.summary()
    print(f"\nTotal params     : {m.count_params():,}")
    print(f"Trainable params : {sum(w.numpy().size for w in m.trainable_weights):,}")
