"""
PyTorch Evaluation script for VGG16-UNet prostate segmentation model.

Usage:
    python evaluate.py
    python evaluate.py --data-dir /path/to/data --model-path /path/to/model.pth
"""

import os
import argparse
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, jaccard_score, precision_score, recall_score,
)

import torch
from model import build_vgg16_unet
from config import H, W, SEED, MODEL_PATH, RESULTS_DIR, VAL_IMAGE_DIR, VAL_MASK_DIR, FILES_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch prostate segmentation model")
    # Change default to .pth for PyTorch
    default_pth = MODEL_PATH.replace('.keras', '.pth')
    parser.add_argument("--model-path", type=str, default=default_pth, help="Path to trained model")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, help="Directory to save results")
    return parser.parse_args()


def save_results(image, mask, y_pred, save_image_path):
    """Save side-by-side comparison: Original | Ground Truth | Prediction."""
    line = np.ones((H, 10, 3)) * 128

    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    args = parse_args()

    # Determine image/mask directories
    if args.data_dir:
        image_dir = os.path.join(args.data_dir, "val_png", "images")
        mask_dir = os.path.join(args.data_dir, "val_png", "masks")
    else:
        image_dir = VAL_IMAGE_DIR
        mask_dir = VAL_MASK_DIR

    os.makedirs(args.results_dir, exist_ok=True)

    # Load model
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        print("Train a model first with: python train.py")
        exit(1)

    model = build_vgg16_unet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    test_x = sorted(glob(os.path.join(image_dir, "*.png")))
    test_y = sorted(glob(os.path.join(mask_dir, "*.png")))
    print(f"Evaluating on {len(test_x)} images")

    if len(test_x) == 0:
        print(f"ERROR: No images found in {image_dir}")
        exit(1)

    # Evaluate
    SCORE = []
    
    with torch.no_grad():
        for x_path, y_path in tqdm(zip(test_x, test_y), total=len(test_x)):
            name = os.path.basename(x_path).split(".")[0]

            # Read image for saving
            image_raw = cv2.imread(x_path, cv2.IMREAD_COLOR)
            image_resized = cv2.resize(image_raw, (W, H))
            
            # Prepare tensor for PyTorch
            x = image_resized / 255.0
            x = x.astype(np.float32)
            x_tensor = torch.tensor(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(device)

            # Read mask
            mask_raw = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask_raw, (W, H))
            y = mask_resized / 255.0
            y = (y > 0.5).astype(np.int32)

            # Predict
            y_pred_tensor = model(x_tensor)
            y_pred = y_pred_tensor.squeeze().cpu().numpy()
            y_pred_bin = (y_pred > 0.5).astype(np.int32)

            # Save visual result
            save_image_path = os.path.join(args.results_dir, f"{name}.png")
            save_results(image_resized, y, y_pred_bin, save_image_path)

            # Compute metrics
            y_flat = y.flatten()
            y_pred_flat = y_pred_bin.flatten()

            acc = accuracy_score(y_flat, y_pred_flat)
            f1 = f1_score(y_flat, y_pred_flat, labels=[0, 1], average="binary", zero_division=1)
            jac = jaccard_score(y_flat, y_pred_flat, labels=[0, 1], average="binary", zero_division=1)
            rec = recall_score(y_flat, y_pred_flat, labels=[0, 1], average="binary", zero_division=1)
            prec = precision_score(y_flat, y_pred_flat, labels=[0, 1], average="binary", zero_division=1)

            SCORE.append([name, acc, f1, jac, rec, prec])

    # Print summary
    score = np.mean([s[1:] for s in SCORE], axis=0)
    print(f"\n{'='*40}")
    print(f"  Accuracy:  {score[0]:.5f}")
    print(f"  F1 Score:  {score[1]:.5f}")
    print(f"  Jaccard:   {score[2]:.5f}")
    print(f"  Recall:    {score[3]:.5f}")
    print(f"  Precision: {score[4]:.5f}")
    print(f"{'='*40}")

    # Save to CSV
    csv_path = os.path.join(FILES_DIR, "score.csv")
    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    print(f"Visual results saved to: {args.results_dir}")
