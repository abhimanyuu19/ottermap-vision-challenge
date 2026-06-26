"""
Evaluation metrics for semantic segmentation.

Includes:
- Pixel Accuracy
- Intersection over Union (IoU)
- Dice Score

Author: Abhimanyu Upadhyay
"""
import torch


def pixel_accuracy(preds, labels):
    """
    Pixel Accuracy
    """

    preds = torch.argmax(preds, dim=1)

    correct = (preds == labels).float().sum()

    total = labels.numel()

    return (correct / total).item()


def intersection_over_union(preds, labels):
    """
    Compute Intersection over Union (IoU) for binary segmentation.
    """

    preds = torch.argmax(preds, dim=1)

    intersection = ((preds == 1) & (labels == 1)).sum().float()

    union = ((preds == 1) | (labels == 1)).sum().float()

    if union == 0:
        return 1.0

    return (intersection / union).item()


def dice_score(preds, labels):
    """
    Compute Dice Score for binary segmentation.
    """

    preds = torch.argmax(preds, dim=1)

    intersection = ((preds == 1) & (labels == 1)).sum().float()

    total = preds.sum() + labels.sum()

    if total == 0:
        return 1.0

    return (2 * intersection / total).item()


def evaluate_metrics(preds, labels):
    """
    Compute all evaluation metrics and return them as a dictionary.
    """

    return {
        "pixel_accuracy": pixel_accuracy(preds, labels),
        "iou": intersection_over_union(preds, labels),
        "dice": dice_score(preds, labels)
    }