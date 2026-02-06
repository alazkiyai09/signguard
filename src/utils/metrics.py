"""
Metrics Module

Utility functions for computing evaluation metrics.
"""

import torch
import numpy as np
from typing import List


def compute_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Compute model accuracy on a dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation data
        device: Device to run on

    Returns:
        Accuracy as a percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return 100.0 * correct / total if total > 0 else 0.0


def compute_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
) -> float:
    """
    Compute average loss on a dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run on

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Confusion matrix
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        conf_matrix[int(t), int(p)] += 1

    return conf_matrix
