import torch
import numpy as np


def intersection_over_union(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    return intersection / union


def batch_iout(pred_masks: torch.Tensor, true_masks: torch.Tensor, th: float = 0.0):
    # masks shape is (batch_size, 1, height, width)
    pred_masks = (pred_masks > th).float()
    intersection = (pred_masks * true_masks).sum(dim=(1, 2, 3))
    union = pred_masks.sum(dim=(1, 2, 3)) + true_masks.sum(dim=(1, 2, 3))
    iou = intersection / (union - intersection)
    iou[iou != iou] = 1  # trick to set all NaN's to 1 (if we have None, we have correctly predicted empty mask)

    precision = (((iou - 0.5 + 1e-6) * 20).ceil() / 10).clamp(0, 1)

    return precision.tolist()


def iout_numpy(pred_masks: np.ndarray, true_masks: np.ndarray, th: float = 0.5):
    # masks shape is (batch_size, 1, height, width)
    pred_masks = (pred_masks > th).astype(float)
    intersection = (pred_masks * true_masks).sum(axis=(1, 2))
    union = pred_masks.sum(axis=(1, 2)) + true_masks.sum(axis=(1, 2))
    iou = intersection / (union - intersection)
    iou[iou != iou] = 1  # trick to set all NaN's to 1 (if we have None, we have correctly predicted empty mask)

    precision = (np.ceil((iou - 0.5 + 1e-6) * 20) / 10).clip(0, 1)

    return precision.mean()


def batch_iou(pred_masks: torch.Tensor, true_masks: torch.Tensor, th: float = 0.0):
    # masks shape is (batch_size, 1, height, width)
    pred_masks = (pred_masks > th).float()
    intersection = (pred_masks * true_masks).sum(dim=(1, 2, 3))
    union = pred_masks.sum(dim=(1, 2, 3)) + true_masks.sum(dim=(1, 2, 3))
    iou = intersection / (union - intersection)
    iou[iou != iou] = 1  # trick to set all NaN's to 1 (if we have None, we have correctly predicted empty mask)
    return iou.tolist()
