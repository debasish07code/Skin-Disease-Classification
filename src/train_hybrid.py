"""
train_hybrid.py
---------------
Two-phase transfer learning training script for the improved hybrid model.

Phase 1 — Head Warm-Up (EPOCHS_PHASE1 epochs)
    ResNet50 backbone fully frozen.
    Only the new classifier head is trained.
    Higher learning rate (1e-3) is safe because pretrained weights are frozen.

Phase 2 — Fine-Tuning  (EPOCHS_PHASE2 epochs)
    Top 30 ResNet50 layers unfrozen.
    Very small learning rate (1e-5) prevents catastrophic forgetting.
    The model adapts ImageNet features → skin-disease domain.

Callbacks used in both phases:
    - ModelCheckpoint  : save the best val_accuracy checkpoint
    - EarlyStopping    : stop early if val_loss stops improving (patience=5)
    - ReduceLROnPlateau: halve LR on plateau (patience=3, min_lr=1e-7)

Outputs (all saved separately from the original pipeline):
    models/best_model_hybrid.keras
    models/final_model_hybrid.keras
    models/history_phase1.json   ← Phase 1 history saved here after Phase 1 runs
    outputs/accuracy_hybrid.png
    outputs/loss_hybrid.png

Original train.py is NOT modified.
"""

import json
import time

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import load_model

from preprocess_hybrid import preprocess_data_hybrid
from model_hybrid import create_hybrid_model, fine_tune_model
from config import *


# ================================================================
# Hyper-parameters
# ================================================================

EPOCHS_PHASE1 = 15   # head warm-up
EPOCHS_PHASE2 = 20   # fine-tuning

# ── Set to True to skip Phase 1 and load the saved Phase 1 checkpoint ──────────
# Phase 1 already completed → set SKIP_PHASE1 = True to go straight to Phase 2
SKIP_PHASE1 = False


# ================================================================
# Load dataset (224×224, ResNet50 preprocessed)
# ================================================================

print("\nLoading Dataset (224×224, ResNet50 preprocessing)...\n")

X_train, X_test, y_train, y_test = preprocess_data_hybrid()


# ================================================================
# Build model
# ================================================================

model = create_hybrid_model()

print("\nModel created successfully.")
print(f"  Input shape  : {model.input_shape}")
print(f"  Output shape : {model.output_shape}")


# ================================================================
# Shared callbacks helper
# ================================================================

def make_callbacks(phase: int):
    """Return a list of callbacks appropriate for the given training phase."""

    checkpoint = ModelCheckpoint(
        filepath=MODEL_DIR / "best_model_hybrid.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )

    return [checkpoint, early_stop, reduce_lr]


# ================================================================
# Phase 1 — Head-only training  (skippable)
# ================================================================

checkpoint_path = MODEL_DIR / "best_model_hybrid.keras"

history_path = MODEL_DIR / "history_phase1.json"

if SKIP_PHASE1 and checkpoint_path.exists():
    print("\n" + "=" * 60)
    print("PHASE 1 — SKIPPED (restoring weights from checkpoint)")
    print(f"  Checkpoint : {checkpoint_path}")
    print("=" * 60 + "\n")

    # Rebuild the model with the same architecture so fine_tune_model()
    # can always find the ResNet50 sub-model via isinstance() check.
    # load_model() flattens the layer structure on disk which breaks
    # the sub-model lookup; load_weights() restores only the weight values.
    model = create_hybrid_model()
    model.load_weights(checkpoint_path)
    print("  Weights restored successfully.\n")
    phase1_time = 0.0

    # ── Load real Phase 1 history from JSON (if available) ───────────────────
    # This allows the full Phase 1 + Phase 2 graph to be plotted correctly
    # even when Phase 1 is skipped on subsequent runs.
    if history_path.exists():
        print(f"  Loading Phase 1 history from {history_path.name}...")
        with open(history_path, "r") as f:
            saved = json.load(f)
        class _LoadedHistory:
            def __init__(self, d):
                self.history = d
        history_phase1 = _LoadedHistory(saved)
        print(f"  Phase 1 epochs in history : {len(saved['accuracy'])}\n")
    else:
        # Fallback: dummy history if JSON not found (first SKIP_PHASE1 run)
        print("  WARNING: history_phase1.json not found — graph will show Phase 2 only.")
        print("  Run once with SKIP_PHASE1=False to generate full graph.\n")
        class _DummyHistory:
            def __init__(self, n):
                self.history = {
                    "accuracy":     [None] * n,
                    "val_accuracy": [None] * n,
                    "loss":         [None] * n,
                    "val_loss":     [None] * n,
                }
        history_phase1 = _DummyHistory(EPOCHS_PHASE1)

else:
    print("\n" + "=" * 60)
    print("PHASE 1 — Head Warm-Up (backbone frozen)")
    print(f"Epochs : {EPOCHS_PHASE1}  |  LR : 1e-3")
    print("=" * 60 + "\n")

    start_time = time.time()

    history_phase1 = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS_PHASE1,
        batch_size=BATCH_SIZE,
        callbacks=make_callbacks(phase=1),
        verbose=1,
    )

    phase1_time = time.time() - start_time
    print(f"\nPhase 1 completed in {phase1_time / 60:.2f} minutes")

    # ── Save Phase 1 history to JSON for future SKIP_PHASE1=True runs ────────
    history_dict = {
        key: [float(v) for v in vals]
        for key, vals in history_phase1.history.items()
    }
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"  Phase 1 history saved → models/history_phase1.json")


# ================================================================
# Phase 2 — Fine-tuning (top 30 ResNet50 layers unfrozen)
# ================================================================

print("\n" + "=" * 60)
print("PHASE 2 — Fine-Tuning (top 30 ResNet50 layers unfrozen)")
print(f"Epochs : {EPOCHS_PHASE2}  |  LR : 1e-5")
print("=" * 60)

model = fine_tune_model(model, unfreeze_last_n_layers=30)

start_time = time.time()

history_phase2 = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS_PHASE2,
    batch_size=BATCH_SIZE,
    callbacks=make_callbacks(phase=2),
    verbose=1,
)

phase2_time = time.time() - start_time
print(f"\nPhase 2 completed in {phase2_time / 60:.2f} minutes")

total_time = phase1_time + phase2_time
print(f"Total Training Time : {total_time / 60:.2f} minutes")


# ================================================================
# Save final model
# ================================================================

model.save(MODEL_DIR / "final_model_hybrid.keras")
print("\nFinal hybrid model saved → models/final_model_hybrid.keras")


# ================================================================
# Combine histories for plotting
# ================================================================

def combine(key, h1, h2):
    """Concatenate Phase 1 and Phase 2 metric lists, skipping None placeholders."""
    p1 = [v for v in h1.history[key] if v is not None]
    p2 = h2.history[key]
    return p1 + p2


train_acc  = combine("accuracy",     history_phase1, history_phase2)
val_acc    = combine("val_accuracy", history_phase1, history_phase2)
train_loss = combine("loss",         history_phase1, history_phase2)
val_loss   = combine("val_loss",     history_phase1, history_phase2)

total_epochs = len(train_acc)
phase1_end   = len([v for v in history_phase1.history["accuracy"] if v is not None])


# ── Accuracy plot ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))

plt.plot(range(1, total_epochs + 1), train_acc, label="Training Accuracy",   color="#2196F3")
plt.plot(range(1, total_epochs + 1), val_acc,   label="Validation Accuracy", color="#FF5722")

# Mark the boundary between Phase 1 and Phase 2
plt.axvline(x=phase1_end, color="gray", linestyle="--", linewidth=1.2,
            label=f"Fine-tune starts (epoch {phase1_end + 1})")

plt.title("Hybrid Model Accuracy — Phase 1 + Phase 2", fontsize=13)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_hybrid.png", dpi=150)
plt.close()

print("Accuracy plot saved → outputs/accuracy_hybrid.png")


# ── Loss plot ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))

plt.plot(range(1, total_epochs + 1), train_loss, label="Training Loss",   color="#2196F3")
plt.plot(range(1, total_epochs + 1), val_loss,   label="Validation Loss", color="#FF5722")

plt.axvline(x=phase1_end, color="gray", linestyle="--", linewidth=1.2,
            label=f"Fine-tune starts (epoch {phase1_end + 1})")

plt.title("Hybrid Model Loss — Phase 1 + Phase 2", fontsize=13)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_hybrid.png", dpi=150)
plt.close()

print("Loss plot saved → outputs/loss_hybrid.png")

print("\nTraining pipeline complete.")
