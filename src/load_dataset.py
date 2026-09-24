import tensorflow as tf
import matplotlib.pyplot as plt

from config import *


dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


print("\n==============================")
print("Dataset Loaded Successfully")
print("==============================")

print(f"\nDataset Path : {DATASET_PATH}")

print("\nDetected Classes:")

for index, class_name in enumerate(dataset.class_names):
    print(f"{index + 1}. {class_name}")


total_images = 0

for images, labels in dataset:
    total_images += images.shape[0]

print(f"\nTotal Images Loaded : {total_images}")


plt.figure(figsize=(10, 10))

for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(dataset.class_names[labels[i]])

        plt.axis("off")

plt.tight_layout()
plt.show()