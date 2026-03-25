"""
Data preprocessing: converts NIfTI 3D volumes to 2D PNG slices.

Reads the Micro-Ultrasound Prostate Segmentation Dataset and generates
512x512 PNG slices for training and validation.

Usage:
    python preprocess.py
    python preprocess.py --raw-dir /path/to/dataset --output-dir /path/to/output
"""

import os
import argparse
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

try:
    import SimpleITK as sitk
except ImportError:
    print("SimpleITK is required for preprocessing. Install it with:")
    print("  pip install SimpleITK")
    exit(1)

from config import H, W, RAW_DATASET_DIR, DATA_DIR, SEED


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess NIfTI volumes to PNG slices")
    parser.add_argument("--raw-dir", type=str, default=RAW_DATASET_DIR, help="Path to raw dataset")
    parser.add_argument("--output-dir", type=str, default=DATA_DIR, help="Output directory for PNGs")
    parser.add_argument("--test-split", type=float, default=0.2, help="Validation split ratio")
    return parser.parse_args()


def process_volume(img_path, mask_path, out_image_dir, out_mask_dir, width, height, st_path=None, out_st_dir=None):
    """Convert a single NIfTI volume to 2D PNG slices."""
    img = sitk.ReadImage(img_path)
    gt = sitk.ReadImage(mask_path)
    image_array = sitk.GetArrayFromImage(img)
    seg_array = sitk.GetArrayFromImage(gt)

    # Normalize image (data volumes are normalized to 0-254 beforehand)
    image_array = 255 * (image_array - 0) / 254

    # Optionally process non-expert annotations
    student_seg_array = None
    if st_path and out_st_dir:
        st = sitk.ReadImage(st_path)
        student_seg_array = sitk.GetArrayFromImage(st)

    number_of_slices = image_array.shape[0]
    max_slices = number_of_slices
    if student_seg_array is not None:
        max_slices = min(number_of_slices, student_seg_array.shape[0])

    # Extract subject name and index from filename
    basename = os.path.basename(img_path)
    parts = basename.replace(".nii.gz", "").split("_")
    sub_name = parts[0] if parts else "subj"
    idx = parts[-1] if len(parts) > 1 else "0"

    count = 0
    for z in range(max_slices):
        image_2d = image_array[z]

        if len(seg_array.shape) == 3:
            seg_2d = seg_array[z]
        else:
            continue

        image_2d_resized = cv2.resize(image_2d, (width, height))
        seg_2d_resized = 255 * (cv2.resize(seg_2d, (width, height)) > 0)

        output_image = os.path.join(out_image_dir, f"{sub_name}_{idx}_img_slice_{z}.png")
        output_seg = os.path.join(out_mask_dir, f"{sub_name}_{idx}_gt_slice_{z}.png")

        cv2.imwrite(output_image, image_2d_resized)
        cv2.imwrite(output_seg, seg_2d_resized)

        if student_seg_array is not None and out_st_dir:
            student_seg_2d = student_seg_array[z]
            student_seg_2d_resized = 255 * (cv2.resize(student_seg_2d, (width, height)) > 0)
            output_st = os.path.join(out_st_dir, f"{sub_name}_{idx}_st_slice_{z}.png")
            cv2.imwrite(output_st, student_seg_2d_resized)

        count += 1

    return count


if __name__ == "__main__":
    args = parse_args()

    # ---- Input paths ----
    image_path = os.path.join(args.raw_dir, "train", "micro_ultrasound_scans")
    mask_path = os.path.join(args.raw_dir, "train", "expert_annotations")
    non_exp_path = os.path.join(args.raw_dir, "train", "non_expert_annotations")

    # ---- Validate input ----
    if not os.path.exists(image_path):
        print(f"ERROR: Dataset not found at {args.raw_dir}")
        print(f"Expected directory structure:")
        print(f"  {args.raw_dir}/train/micro_ultrasound_scans/*.nii.gz")
        print(f"  {args.raw_dir}/train/expert_annotations/*.nii.gz")
        exit(1)

    # ---- Output paths ----
    train_out_image = os.path.join(args.output_dir, "train_png", "images")
    train_out_mask = os.path.join(args.output_dir, "train_png", "masks")
    val_out_image = os.path.join(args.output_dir, "val_png", "images")
    val_out_mask = os.path.join(args.output_dir, "val_png", "masks")

    for d in [train_out_image, train_out_mask, val_out_image, val_out_mask]:
        os.makedirs(d, exist_ok=True)

    # ---- Load file lists ----
    list_of_image = sorted(glob.glob(os.path.join(image_path, "*.nii.gz")))
    list_of_mask = sorted(glob.glob(os.path.join(mask_path, "*.nii.gz")))
    list_of_st = sorted(glob.glob(os.path.join(non_exp_path, "*.nii.gz")))

    print(f"Found {len(list_of_image)} images, {len(list_of_mask)} masks")

    if len(list_of_image) == 0:
        print("ERROR: No .nii.gz files found. Check your dataset path.")
        exit(1)

    assert len(list_of_image) == len(list_of_mask), \
        f"Mismatch: {len(list_of_image)} images vs {len(list_of_mask)} masks"

    # ---- Train/Val split ----
    # Split image and mask lists together using the same indices
    indices = list(range(len(list_of_image)))
    train_idx, val_idx = train_test_split(indices, test_size=args.test_split, random_state=SEED)

    train_images = [list_of_image[i] for i in train_idx]
    train_masks = [list_of_mask[i] for i in train_idx]
    val_images = [list_of_image[i] for i in val_idx]
    val_masks = [list_of_mask[i] for i in val_idx]

    # Use the SAME indices for non-expert annotations (fixes the original bug)
    train_st = [list_of_st[i] for i in train_idx] if list_of_st else []
    val_st = [list_of_st[i] for i in val_idx] if list_of_st else []

    print(f"Train: {len(train_images)} volumes | Val: {len(val_images)} volumes")
    print(f"Output: {args.output_dir}")

    # ---- Process training set ----
    print("\nProcessing training set...")
    total_train = 0
    for i in tqdm(range(len(train_images))):
        st_path = train_st[i] if train_st else None
        total_train += process_volume(
            train_images[i], train_masks[i],
            train_out_image, train_out_mask,
            W, H, st_path=st_path,
        )

    # ---- Process validation set ----
    print("Processing validation set...")
    total_val = 0
    for i in tqdm(range(len(val_images))):
        st_path = val_st[i] if val_st else None
        total_val += process_volume(
            val_images[i], val_masks[i],
            val_out_image, val_out_mask,
            W, H, st_path=st_path,
        )

    print(f"\nPreprocessing complete!")
    print(f"  Training slices: {total_train}")
    print(f"  Validation slices: {total_val}")
    print(f"  Output directory: {args.output_dir}")
