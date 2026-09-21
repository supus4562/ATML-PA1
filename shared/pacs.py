"""shared/pacs.py — PACS dataset loader.

Expected directory layout (standard PACS):
  <root>/
    art_painting/
      dog/  elephant/  giraffe/  guitar/  horse/  house/  person/
    cartoon/   (same 7 classes)
    photo/     (same 7 classes)
    sketch/    (same 7 classes)

Usage:
  dataset = PACSDataset(root="/path/to/pacs", domain="photo", transform=t)
  loader  = DataLoader(dataset, batch_size=32, shuffle=True)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset

# Canonical class names and integer labels (alphabetical = PACS convention)
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
CLASS_TO_IDX = {c: i for i, c in enumerate(PACS_CLASSES)}

# Canonical domain folder names
DOMAIN_FOLDERS = {
    "art":      "art_painting",
    "art_painting": "art_painting",
    "cartoon":  "cartoon",
    "photo":    "photo",
    "sketch":   "sketch",
}


class PACSDataset(Dataset):
    """PyTorch Dataset for a single PACS domain.

    Args:
        root:    Path to the PACS root directory (contains domain sub-folders).
        domain:  One of 'art' | 'art_painting' | 'cartoon' | 'photo' | 'sketch'.
        transform: Optional torchvision transform applied to PIL images.
        indices: Optional list of indices to use (for train/val splits).
    """

    def __init__(
        self,
        root: str | Path,
        domain: str,
        transform: Optional[Callable] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.domain = domain
        self.transform = transform

        folder_name = DOMAIN_FOLDERS.get(domain)
        if folder_name is None:
            raise ValueError(
                f"Unknown PACS domain '{domain}'. "
                f"Choose from: {list(DOMAIN_FOLDERS.keys())}"
            )
        domain_dir = self.root / folder_name
        if not domain_dir.exists():
            raise FileNotFoundError(
                f"PACS domain directory not found: {domain_dir}\n"
                f"Please download PACS and point --pacs_root to the folder containing "
                f"art_painting/, cartoon/, photo/, sketch/."
            )

        self._samples: list[tuple[str, int]] = []  # (abs_path, label)
        for cls_name in PACS_CLASSES:
            cls_dir = domain_dir / cls_name
            if not cls_dir.exists():
                continue
            label = CLASS_TO_IDX[cls_name]
            for fname in sorted(cls_dir.iterdir()):
                if fname.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
                    self._samples.append((str(fname), label))

        if len(self._samples) == 0:
            raise RuntimeError(
                f"No images found in {domain_dir}. "
                "Check folder structure (expected: <domain>/<class>/<image.*>)."
            )

        if indices is not None:
            self._samples = [self._samples[i] for i in indices]

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        path, label = self._samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # ── Helpers ────────────────────────────────────────────────────────────────

    @property
    def targets(self) -> list[int]:
        """List of integer labels (for stratified splitting)."""
        return [s[1] for s in self._samples]

    @property
    def num_classes(self) -> int:
        return len(PACS_CLASSES)
