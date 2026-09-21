"""shared/pacs_protocol.py — PACS train/val split generation.

Creates a stratified 80/20 split for each source domain (Photo, Art, Cartoon)
and saves them to shared/splits/pacs_sketch_seed6304.json.

Usage:
  python shared/pacs_protocol.py --pacs_root /path/to/pacs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Allow running from repo root
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.seed import set_all_seeds
from shared.pacs import PACSDataset

SEED = 6304
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
SPLITS_DIR = Path(__file__).parent / "splits"


def make_splits(pacs_root: str, seed: int = SEED) -> dict:
    """Generate stratified 80/20 splits for each source domain.

    Returns a dict  {domain: {"train": [idx, ...], "val": [idx, ...]}}
    """
    set_all_seeds(seed)
    splits = {}

    for domain in SOURCE_DOMAINS:
        dataset = PACSDataset(root=pacs_root, domain=domain)
        targets = np.array(dataset.targets)
        n = len(targets)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(n), targets))

        splits[domain] = {
            "train": train_idx.tolist(),
            "val":   val_idx.tolist(),
            "n_train": int(len(train_idx)),
            "n_val":   int(len(val_idx)),
        }
        print(
            f"  {domain:14s}: {len(train_idx)} train / {len(val_idx)} val "
            f"(total {n})"
        )

    return splits


def load_or_create_splits(pacs_root: str, seed: int = SEED) -> dict:
    """Load cached splits if they exist, otherwise generate and save them."""
    out_path = SPLITS_DIR / "pacs_sketch_seed6304.json"

    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    print(f"[pacs_protocol] Generating stratified splits (seed={seed}) …")
    splits = make_splits(pacs_root, seed=seed)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[pacs_protocol] Saved to {out_path}")
    return splits


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PACS train/val splits.")
    parser.add_argument("--pacs_root", required=True, help="Path to PACS root directory.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SPLITS_DIR / "pacs_sketch_seed6304.json"

    print(f"[pacs_protocol] Generating stratified splits for seed={args.seed} …")
    splits = make_splits(args.pacs_root, seed=args.seed)

    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[pacs_protocol] Written to {out_path}")


if __name__ == "__main__":
    main()
