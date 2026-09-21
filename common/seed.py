"""common/seed.py — Centralised seeding utility."""
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 6304) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may slow training; acceptable for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
