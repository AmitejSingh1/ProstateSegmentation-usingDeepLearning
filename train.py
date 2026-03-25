"""
PyTorch Training script for VGG16-UNet prostate segmentation model.

Usage:
    python train.py
    python train.py --epochs 100 --batch-size 4 --lr 1e-4
    python train.py --data-dir /path/to/data --model-path /path/to/model.pth
"""

import os
import argparse
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import build_vgg16_unet
from metrics import dice_loss, dice_coef
from config import H, W, SEED, FILES_DIR, get_train_args


def create_dir(path):
    os.makedirs(path, exist_ok=True)


class ProstateDataset(Dataset):
    """Custom PyTorch Dataset for loading images and masks."""
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image
        x_path = self.image_paths[idx]
        x = cv2.imread(x_path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)
        # HWC to CHW format for PyTorch
        x = np.transpose(x, (2, 0, 1))

        # Read mask
        y_path = self.mask_paths[idx]
        y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        y = cv2.resize(y, (W, H))
        y = y / 255.0
        y = (y > 0.5).astype(np.float32)
        # Add channel dimension: (1, H, W)
        y = np.expand_dims(y, axis=0)

        return torch.tensor(x), torch.tensor(y)


def load_data(image_dir, mask_dir):
    x = sorted(glob(os.path.join(image_dir, "*.png")))
    y = sorted(glob(os.path.join(mask_dir, "*.png")))
    return x, y


def train_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = dice_loss(y, y_pred)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)


def evaluate_epoch(model, loader, device):
    model.eval()
    epoch_loss = 0.0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating", leave=False):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = dice_loss(y, y_pred)
            
            epoch_loss += loss.item()
            
    return epoch_loss / len(loader)


if __name__ == "__main__":
    # ---- Seed & Hardware ----
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Parse CLI args ----
    args = get_train_args()
    
    # In PyTorch, weights are usually saved as .pth instead of .keras
    model_path = args.model_path.replace('.keras', '.pth')

    create_dir(FILES_DIR)
    train_image_dir = os.path.join(args.data_dir, "train_png", "images")
    train_mask_dir = os.path.join(args.data_dir, "train_png", "masks")
    val_image_dir = os.path.join(args.data_dir, "val_png", "images")
    val_mask_dir = os.path.join(args.data_dir, "val_png", "masks")

    # ---- Load data ----
    train_x, train_y = load_data(train_image_dir, train_mask_dir)
    valid_x, valid_y = load_data(val_image_dir, val_mask_dir)

    print(f"Train: {len(train_x)} images | Valid: {len(valid_x)} images")

    if len(train_x) == 0:
        print("ERROR: No training images found. Check --data-dir path.")
        exit(1)

    # ---- Datasets & DataLoaders ----
    train_dataset = ProstateDataset(train_x, train_y)
    valid_dataset = ProstateDataset(valid_x, valid_y)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )

    # ---- Build model ----
    model = build_vgg16_unet()
    model = model.to(device)

    # ---- Optimizer & Scheduler ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7, verbose=True)

    # ---- Training Loop ----
    print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 15
    csv_log = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_epoch(model, valid_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        csv_log.append([epoch, train_loss, val_loss])
        
        # Save Best Model
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model.")
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                break

    # Save training logs
    df = pd.DataFrame(csv_log, columns=['epoch', 'train_loss', 'val_loss'])
    df.to_csv(os.path.join(FILES_DIR, "data.csv"), index=False)
    
    print("\nTraining complete!")
    print(f"Best model saved to: {model_path}")
