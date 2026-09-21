"""task3/evaluation/source_domain_separability.py — 3-class domain separability.

Evaluates how separable the 3 source domains are in feature space.
Chance = 33.3% (3-class random baseline).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from common.seed import set_all_seeds


def compute_3class_separability(
    backbone: torch.nn.Module,
    source_val_loaders: dict[str, DataLoader],
    device: torch.device | None = None,
    seed: int = 6304,
) -> float:
    """Compute held-out 3-class accuracy for source domain identification.

    Args:
        backbone:           Feature extractor (frozen).
        source_val_loaders: Dict mapping domain name → DataLoader.
        device:             If None, inferred from backbone parameters.
        seed:               Random state for split + logistic regression.

    Returns:
        Held-out 3-class accuracy (float in [0,1]). Chance = 1/3 ≈ 0.333.
    """
    set_all_seeds(seed)
    backbone.eval()

    if device is None:
        device = next(backbone.parameters()).device

    domain_map = {"photo": 0, "art_painting": 1, "cartoon": 2}
    domain_features: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for name, loader in source_val_loaders.items():
            domain_idx = domain_map.get(name)
            if domain_idx is None:
                continue
            for x, _ in loader:
                x = x.to(device)
                feats = backbone(x).cpu().numpy()
                domain_features[domain_idx].append(feats)
                counts[domain_idx] += feats.shape[0]

    # Balance across domains
    min_count = min(counts[i] for i in range(3) if counts[i] > 0)
    if min_count == 0:
        return 0.0

    feat_list, label_list = [], []
    for i in range(3):
        if not domain_features[i]:
            continue
        f = np.concatenate(domain_features[i], axis=0)[:min_count]
        feat_list.append(f)
        label_list.append(np.full(min_count, i, dtype=np.int64))

    X = np.concatenate(feat_list, axis=0)
    y = np.concatenate(label_list, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )

    clf = LogisticRegression(
        C=1.0, multi_class="multinomial", max_iter=1000, random_state=seed
    )
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))
