"""common/metrics.py — Shared evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score


def accuracy(preds: np.ndarray | torch.Tensor,
             labels: np.ndarray | torch.Tensor) -> float:
    """Top-1 accuracy as a float in [0, 1]."""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    return float(np.mean(preds == labels))


def macro_f1(preds: np.ndarray | torch.Tensor,
             labels: np.ndarray | torch.Tensor) -> float:
    """Macro-averaged F1 score."""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def mean_max_confidence(probs: np.ndarray | torch.Tensor) -> float:
    """Mean of max softmax probability across all samples.

    Args:
        probs: (N, C) probability tensor or array.
    Returns:
        Scalar mean-max-confidence value.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    return float(np.mean(np.max(probs, axis=1)))


def prediction_consistency(preds_clean: np.ndarray | torch.Tensor,
                            preds_transformed: np.ndarray | torch.Tensor) -> float:
    """Fraction of samples where prediction does NOT change after a transform."""
    if isinstance(preds_clean, torch.Tensor):
        preds_clean = preds_clean.cpu().numpy()
    if isinstance(preds_transformed, torch.Tensor):
        preds_transformed = preds_transformed.cpu().numpy()
    return float(np.mean(preds_clean == preds_transformed))


# ── Aliases used by Task 2 and Task 3 ─────────────────────────────────────────

def calculate_accuracy(preds, labels) -> float:
    """Alias for accuracy(); used by task2/task3 methods."""
    return accuracy(preds, labels)


def calculate_macro_f1(preds, labels) -> float:
    """Alias for macro_f1(); used by task2/task3 methods."""
    return macro_f1(preds, labels)


def calculate_metrics(labels, preds) -> dict:
    """Return dict with accuracy and macro_f1 keys.

    Note: argument order is (labels, preds) matching usage in task2/task3.
    Both tensors and numpy arrays are accepted.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    return {
        "accuracy": float(np.mean(preds == labels)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
