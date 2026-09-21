"""task3/models/classifier_head.py — Linear classification head for Task 3."""
from __future__ import annotations

import torch.nn as nn


class ClassifierHead(nn.Module):
    """Single linear classifier head.

    Accepts both in_dim/n_classes and in_features/num_classes kwargs for
    compatibility with different call sites.
    """

    def __init__(self, in_dim: int = 512, n_classes: int = 7,
                 # legacy aliases
                 in_features: int | None = None, num_classes: int | None = None) -> None:
        super().__init__()
        in_dim = in_features if in_features is not None else in_dim
        n_classes = num_classes if num_classes is not None else n_classes
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)
