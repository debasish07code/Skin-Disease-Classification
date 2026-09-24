import time

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

from preprocess import preprocess_data
from model import create_model
from config import *


print("\nLoading Dataset...\n")

X_train, X_test, y_train, y_test = preprocess_data()


model = create_model()


checkpoint = ModelCheckpoint(
    filepath=MODEL_DIR / "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


print("\nTraining Started...\n")

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
    verbose=1
)

end_time = time.time()

print("\nTraining Completed!")

print(f"\nTraining Time : {(end_time-start_time)/60:.2f} minutes")


model.save(MODEL_DIR / "final_model.keras")

print("\nFinal model saved successfully.")


plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Testing Accuracy")

plt.title("Model Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.savefig(OUTPUT_DIR / "accuracy.png")

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Testing Loss")

plt.title("Model Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig(OUTPUT_DIR / "loss.png")

plt.show()