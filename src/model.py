from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)

from config import *


def create_model():

    model = Sequential()

    # ==========================================
    # CNN Block 1
    # ==========================================

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
        )
    )
    model.add(MaxPooling2D((2, 2)))

    # ==========================================
    # CNN Block 2
    # ==========================================

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))

    # ==========================================
    # CNN Block 3
    # ==========================================

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))

    # ==========================================
    # CNN Block 4
    # ==========================================

    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))

    # ==========================================
    # CNN Block 5
    # ==========================================

    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))

    # ==========================================
    # Classifier
    # ==========================================

    model.add(Flatten())

    model.add(Dense(256, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(len(TARGET_CLASSES), activation="softmax"))

    # ==========================================
    # Compile
    # ==========================================

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":

    cnn_model = create_model()

    cnn_model.summary()