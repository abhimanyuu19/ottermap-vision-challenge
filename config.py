"""
Project configuration.

Contains dataset paths, training hyperparameters,
model configuration and output directories.

Author: Abhimanyu Upadhyay
"""
from pathlib import Path
import torch

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).parent

TRAIN_IMAGES = PROJECT_ROOT / "data/train/images"
TRAIN_MASKS = PROJECT_ROOT / "data/train/masks"

VAL_IMAGES = PROJECT_ROOT / "data/val/images"
VAL_MASKS = PROJECT_ROOT / "data/val/masks"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Training Parameters
# -----------------------------
IMAGE_SIZE = 512

BATCH_SIZE = 2

NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

NUM_CLASSES = 2

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42