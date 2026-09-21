"""task3/train.py — Training dispatch for Task 3 (Domain Generalization).

Usage:
  # ERM: uses task2 checkpoint, no training
  python task3/train.py --config task3/configs/erm.yaml    --pacs_root /path/to/pacs

  # DAN-DG
  python task3/train.py --config task3/configs/dan_dg.yaml --pacs_root /path/to/pacs

  # SAM (main comparison rho=0.05)
  python task3/train.py --config task3/configs/sam.yaml    --pacs_root /path/to/pacs

  # SAM design study
  python task3/train.py --config task3/configs/sam.yaml    --pacs_root /path/to/pacs --rho 0.01
  python task3/train.py --config task3/configs/sam.yaml    --pacs_root /path/to/pacs --rho 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from common.seed import set_all_seeds
from shared.pacs import PACSDataset
from shared.pacs_protocol import load_or_create_splits
from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead
from task3.methods.dan_dg import DanDGTrainer
from task3.methods.sam import SAMTrainer

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 — Domain Generalization training")
    parser.add_argument("--config", required=True, help="Path to method YAML config")
    parser.add_argument("--pacs_root", default=None, help="Path to PACS root directory")
    parser.add_argument("--rho", type=float, default=None,
                        help="Override SAM rho (for design study)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.pacs_root:
        config["pacs_root"] = args.pacs_root
    if args.rho is not None:
        config["rho"] = args.rho

    if not config.get("pacs_root"):
        raise ValueError("pacs_root must be set via config or --pacs_root")

    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)

    method = config["method"]
    print(f"[task3/train] method={method}  device={device}")

    if method == "erm":
        ckpt = config.get("erm_checkpoint", "task2/results/source_only_checkpoint.pth")
        if not os.path.exists(ckpt):
            print(f"[task3/train] WARNING: ERM checkpoint not found at {ckpt}.")
            print("  Run task2/train.py --config task2/configs/source_only.yaml first.")
        else:
            print(f"[task3/train] ERM checkpoint found at {ckpt}. No retraining needed.")
        return

    # ── Splits ────────────────────────────────────────────────────────────────
    splits = load_or_create_splits(config["pacs_root"], seed=config["seed"])

    # ── Source-only loaders (NO target data!) ─────────────────────────────────
    source_train_loaders = {}
    source_val_loaders = {}
    for domain in config["source_domains"]:
        ds_train = PACSDataset(config["pacs_root"], domain=domain,
                               transform=TRAIN_TRANSFORM,
                               indices=splits[domain]["train"])
        ds_val = PACSDataset(config["pacs_root"], domain=domain,
                             transform=VAL_TRANSFORM,
                             indices=splits[domain]["val"])
        source_train_loaders[domain] = DataLoader(
            ds_train, batch_size=config["batch_size_per_domain"],
            shuffle=True, drop_last=True, num_workers=2, pin_memory=True,
        )
        source_val_loaders[domain] = DataLoader(
            ds_val, batch_size=64, shuffle=False, num_workers=2, pin_memory=True,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone = ResNet18Backbone(pretrained=True).to(device)
    classifier = ClassifierHead(in_dim=512, n_classes=config["n_classes"]).to(device)

    # ── Train ─────────────────────────────────────────────────────────────────
    if method == "dan_dg":
        trainer = DanDGTrainer(backbone, classifier, config, device)
    elif method == "sam":
        trainer = SAMTrainer(backbone, classifier, config, device)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: erm, dan_dg, sam")

    history = trainer.train(source_train_loaders, source_val_loaders)

    curves_path = os.path.join(config["output_dir"], f"{method}_training_history.json")
    with open(curves_path, "w") as f:
        json.dump(history, f, indent=2, default=float)
    print(f"[task3/train] Training history saved to {curves_path}")


if __name__ == "__main__":
    main()
