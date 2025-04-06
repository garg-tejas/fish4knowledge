import torch

def iou_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = ((preds + targets) - (preds * targets)).sum(dim=(1,2,3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    dice = (2. * intersection + 1e-6) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + 1e-6)
    return dice.mean()
