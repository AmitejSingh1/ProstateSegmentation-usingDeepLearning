"""
Central configuration for the Prostate Segmentation project.
All paths and hyperparameters are defined here.
Override via command-line arguments or environment variables.
"""

import os
import argparse

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BASE_DIR, "files")
MODEL_PATH = os.path.join(FILES_DIR, "model.keras")
CSV_PATH = os.path.join(FILES_DIR, "data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Data directories (override with environment variables or CLI args)
DATA_DIR = os.environ.get("PROSTATE_DATA_DIR", os.path.join(BASE_DIR, "data"))
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train_png", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train_png", "masks")
VAL_IMAGE_DIR = os.path.join(DATA_DIR, "val_png", "images")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val_png", "masks")

# Raw dataset paths (for preprocessing)
RAW_DATASET_DIR = os.environ.get(
    "PROSTATE_RAW_DIR",
    os.path.join(BASE_DIR, "Micro_Ultrasound_Prostate_Segmentation_Dataset"),
)

# ---------- Image Settings ----------
H = 512
W = 512
INPUT_SHAPE = (H, W, 3)

# ---------- Training Hyperparameters ----------
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
SEED = 42


def get_train_args():
    """Parse command-line arguments to override training config."""
    parser = argparse.ArgumentParser(description="Train VGG16-UNet for Prostate Segmentation")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Root directory for processed data")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to save/load model")
    return parser.parse_args()
