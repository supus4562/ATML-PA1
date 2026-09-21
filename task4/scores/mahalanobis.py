"""task4/scores/mahalanobis.py — Mahalanobis distance OOD score (vectorized)."""
from __future__ import annotations

import numpy as np


class MahalanobisScore:
    """Minimum Mahalanobis distance to class centroids.

    Uses per-class means and a shared diagonal covariance estimated from
    UNAUGMENTED CIFAR-10 training features.

    Higher score → more likely to be out-of-distribution.
    """

    def __init__(self) -> None:
        self.mu_c: np.ndarray | None = None       # (C, D)
        self.sigma_inv: np.ndarray | None = None  # (D,) diagonal precision

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
        """Estimate per-class means and shared diagonal covariance.

        Args:
            train_features: (N, D) float array of unaugmented training features.
            train_labels:   (N,)   integer class labels.
        """
        classes = np.unique(train_labels)
        n_classes = len(classes)
        D = train_features.shape[1]

        self.mu_c = np.zeros((n_classes, D), dtype=np.float64)
        pooled_var = np.zeros(D, dtype=np.float64)
        N_total = len(train_labels)

        for c in classes:
            mask = train_labels == c
            feats_c = train_features[mask].astype(np.float64)
            self.mu_c[c] = feats_c.mean(axis=0)
            # Accumulate per-class sum of squared deviations
            pooled_var += ((feats_c - self.mu_c[c]) ** 2).sum(axis=0)

        # Shared diagonal covariance: average over all training examples
        pooled_var /= N_total
        # Add 1e-6 to every diagonal entry for numerical stability
        self.sigma_inv = 1.0 / (pooled_var + 1e-6)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Compute minimum Mahalanobis distance for each sample.

        Args:
            features: (N, D) float array.
        Returns:
            (N,) array where higher value = more likely OOD.
        """
        if self.mu_c is None or self.sigma_inv is None:
            raise RuntimeError("Call fit() before score().")

        features = features.astype(np.float64)
        N = features.shape[0]
        C = self.mu_c.shape[0]

        # Vectorized: for each class c, compute Mahalanobis distance for all N samples
        # diff[c]: (N, D)
        # dist[c]: (N,) = sum_d sigma_inv[d] * diff[c, :, d]^2
        dists = np.empty((C, N), dtype=np.float64)
        for c in range(C):
            diff = features - self.mu_c[c]  # (N, D)
            dists[c] = (diff ** 2 * self.sigma_inv).sum(axis=1)

        # Minimum distance over classes → per-sample score
        return dists.min(axis=0)  # (N,)
