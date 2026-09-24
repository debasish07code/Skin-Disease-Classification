from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset
DATASET_PATH = PROJECT_ROOT / "dataset" / "IMG_CLASSES"

# Output folders
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create output folders if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

CHANNELS = 3


# ==================================
# Training Parameters
# ==================================

BATCH_SIZE = 32  # Not specified in the paper

EPOCHS = 30

SEED = 42  # Not specified in the paper


TARGET_CLASSES = [
    "Eczema",
    "Psoriasis",
    "Melanoma",
    "Basal Cell Carcinoma",
    "Benign Keratosis"
]