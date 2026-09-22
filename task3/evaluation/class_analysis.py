"""task3/evaluation/class_analysis.py — Per-class accuracy, confusion analysis, and plots for Task 3.

Required by the PDF:
  • Per-class Sketch changes and selected failures that support the conclusions
  • Row-normalised confusion matrices
  • Comparison with Task 2 results
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from common.plotting import savefig

PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def per_class_accuracy(targets: np.ndarray, preds: np.ndarray, n_classes: int = 7) -> np.ndarray:
    """Return per-class accuracy as a (n_classes,) float array."""
    accs = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        mask = (targets == c)
        if mask.sum() > 0:
            accs[c] = float((preds[mask] == c).mean())
    return accs


def top_confusions(targets: np.ndarray, preds: np.ndarray, class_names: list[str], top_k: int = 5) -> list[dict]:
    """Return the top-k most common (true_class → predicted_class) misclassification pairs.

    Returns a list of dicts with keys: 'true', 'pred', 'count'.
    """
    n = len(class_names)
    cm = confusion_matrix(targets, preds, labels=list(range(n)))
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # ignore correct predictions
    result = []
    for _ in range(top_k):
        idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
        if cm_copy[idx] == 0:
            break
        result.append({
            "true": class_names[idx[0]],
            "pred": class_names[idx[1]],
            "count": int(cm_copy[idx]),
        })
        cm_copy[idx] = 0
    return result


def plot_per_class_delta(
    baseline_acc: np.ndarray,
    method_accs: dict[str, np.ndarray],
    class_names: list[str],
    out_path: str,
    baseline_name: str = "ERM",
) -> None:
    """Grouped bar chart: ΔAcc = method_acc − baseline_acc per class."""
    n_classes = len(class_names)
    methods = list(method_accs.keys())
    n_methods = len(methods)
    if n_methods == 0:
        return
    x = np.arange(n_classes)
    bar_w = 0.8 / max(n_methods, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_methods, 1)))

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 1.6), 5))
    for i, (method, acc) in enumerate(method_accs.items()):
        delta = acc - baseline_acc
        offset = (i - n_methods / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, delta, bar_w, label=method,
                      color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, delta):
            ypos = bar.get_height() + (0.008 if val >= 0 else -0.022)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.2f}", ha="center", va="bottom",
                    fontsize=7, color="black")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(f"ΔAcc vs {baseline_name}")
    ax.set_title(f"Task 3 — Per-Class Sketch Accuracy Change vs {baseline_name}")
    ax.legend(loc="upper right")
    plt.tight_layout()

    savefig(out_path, fig)


def plot_confusion_matrix(
    targets: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    method_name: str,
    out_path: str,
) -> None:
    """Row-normalised confusion matrix heatmap for Sketch target set."""
    n = len(class_names)
    cm = confusion_matrix(targets, preds, labels=list(range(n)), normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"{method_name} — Target (Sketch) Confusion Matrix", fontsize=11)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if cm[i, j] > 0.5 else "black")

    plt.tight_layout()
    savefig(out_path, fig)
