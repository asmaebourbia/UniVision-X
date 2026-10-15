"""IoU/J, Dice and boundary F proxy (not official MeViS evaluation)."""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import torch
import torch.nn.functional as F


def segmentation_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dims = tuple(range(2, p.ndim))
    dice = 1 - ((2 * (p * target).sum(dim=dims) + 1) /
                (p.sum(dim=dims) + target.sum(dim=dims) + 1)).mean()
    return bce + dice


def mask_iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    uni = np.logical_or(pred, gt).sum()
    return 1.0 if uni == 0 else float(np.logical_and(pred, gt).sum() / uni)


def mask_dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    size = pred.sum() + gt.sum()
    return 1.0 if size == 0 else float(2 * np.logical_and(pred, gt).sum() / size)


def boundary_f(pred, gt, tolerance=2):
    """Simple fixed-pixel tolerance proxy; use official script for benchmark J&F."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if pred.any() != gt.any():
        return 0.0
    pb = np.logical_xor(pred, binary_erosion(pred))
    gb = np.logical_xor(gt, binary_erosion(gt))
    if not pb.any() or not gb.any():
        return 0.0
    pb_d = binary_dilation(pb, iterations=tolerance)
    gb_d = binary_dilation(gb, iterations=tolerance)
    prec = np.logical_and(pb, gb_d).sum() / pb.sum()
    recall = np.logical_and(gb, pb_d).sum() / gb.sum()
    return float(2 * prec * recall / (prec + recall + 1e-8))


def uncertainty_mean(logits):
    p = logits.sigmoid().float().clamp(1e-6, 1-1e-6)
    entropy = -(p * torch.log(p) + (1-p) * torch.log(1-p)) / np.log(2)
    return entropy.mean().item()
