"""task3/evaluation/sharpness.py — Standardized local sharpness proxy calculation."""
from __future__ import annotations

import torch
import torch.nn as nn
from common.seed import set_all_seeds


def compute_sharpness(
    backbone: nn.Module,
    classifier: nn.Module,
    source_val_loaders: dict,
    device: torch.device,
    rho: float = 0.05,
    n_per_domain: int = 32,
    seed: int = 6304,
) -> float:
    """Compute local sharpness proxy Δ_sharp = L_val(θ + ϵ) - L_val(θ).

    Standard standardized diagnostic:
      Fixed validation batch with n_per_domain examples from each source domain.
      Normalized gradient ascent step with radius rho=0.05.
    """
    set_all_seeds(seed)

    loaders = list(source_val_loaders.values()) if isinstance(source_val_loaders, dict) else source_val_loaders
    X_list, y_list = [], []
    for loader in loaders:
        n_collected = 0
        for x, y in loader:
            take = min(n_per_domain - n_collected, x.size(0))
            X_list.append(x[:take])
            y_list.append(y[:take])
            n_collected += take
            if n_collected >= n_per_domain:
                break

    X = torch.cat(X_list, dim=0).to(device)
    y = torch.cat(y_list, dim=0).to(device)

    backbone.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss()

    params = [p for p in list(backbone.parameters()) + list(classifier.parameters()) if p.requires_grad]

    with torch.enable_grad():
        logits = classifier(backbone(X))
        loss = criterion(logits, y)
        loss.backward()

    # GPU-vectorized gradient norm calculation
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        grad_norm = torch.norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        scale = rho / (grad_norm + 1e-12)
    else:
        scale = 0.0

    eps_list = []
    for p in params:
        if p.grad is not None:
            e = p.grad * scale
            eps_list.append(e)
            p.data.add_(e)
        else:
            eps_list.append(None)

    with torch.no_grad():
        logits2 = classifier(backbone(X))
        loss2 = criterion(logits2, y)

    for p, e in zip(params, eps_list):
        if e is not None:
            p.data.sub_(e)

    # Zero grads to leave model clean
    backbone.zero_grad()
    classifier.zero_grad()

    return float((loss2 - loss).item())

