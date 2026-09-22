"""common/plotting.py — Shared matplotlib/seaborn styling."""
from __future__ import annotations

import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None


# ── Global defaults ────────────────────────────────────────────────────────────
PALETTE = "tab10"
FIGURE_DPI = 150
FONT_SIZE = 11

_STYLE_APPLIED = False


def apply_style() -> None:
    """Apply consistent style across all tasks. Call once per script."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    if sns is not None:
        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
    else:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "axes.titlesize": FONT_SIZE + 2,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.constrained_layout.use": True,
    })
    _STYLE_APPLIED = True


def savefig(path: str, fig: plt.Figure | None = None) -> None:
    """Save figure (or current figure) to *path*, creating parent dirs."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    target = fig or plt.gcf()
    target.savefig(path)
    plt.close(target)
    print(f"  [saved] {path}")


def model_colors() -> dict[str, str]:
    """Consistent color mapping for the three Task-1 models."""
    return {
        "ResNet-50": "#1f77b4",
        "ViT-B/16":  "#ff7f0e",
        "CLIP":      "#2ca02c",
    }
