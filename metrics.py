"""
PyTorch implementation of Dice Metrics and Loss for Segmentation.
"""

import torch

def iou(y_true, y_pred, smooth=1e-15):
    """
    Intersection over Union calculating function.
    y_true: Ground truth mask (Tensor)
    y_pred: Predicted mask (Tensor, usually thresholded)
    """
    if y_pred.requires_grad:
        y_pred = y_pred.detach()
    if y_true.requires_grad:
        y_true = y_true.detach()
        
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    
    val = (intersection + smooth) / (union + smooth)
    return val.item()

def dice_coef(y_true, y_pred, smooth=1e-15):
    """
    Differentiable Sørensen-Dice coefficient.
    """
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice loss (1 - Dice Coefficient) for training.
    """
    return 1.0 - dice_coef(y_true, y_pred)