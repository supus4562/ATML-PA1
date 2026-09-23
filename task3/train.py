"""task3/train.py — Training dispatch for Task 3 (Domain Generalization).

Usage:
  # ERM: uses task2 checkpoint, no training
  python task3/train.py --config task3/configs/erm.yaml    --pacs_root /path/to/pacs

  # DAN-DG (main comparison lambda_dg=1.0)
  python task3/train.py --config task3/configs/dan_dg.yaml --pacs_root /path/to/pacs

  # DAN-DG design study
  python task3/train.py --config task3/configs/dan_dg.yaml --pacs_root /path/to/pacs --lambda_dg 0.1
  python task3/train.py --config task3/configs/dan_dg.yaml --pacs_root /path/to/pacs --lambda_dg 10.0

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
from tqdm import tqdm

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
    parser.add_argument("--lambda_dg", type=float, default=None,
                        help="Override lambda_dg (for DAN-DG design study)")
    parser.add_argument("--batch_size_per_domain", type=int, default=None,
                        help="Override batch_size_per_domain (default: 8)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.pacs_root:
        config["pacs_root"] = args.pacs_root
    if args.rho is not None:
        config["rho"] = args.rho
    if args.lambda_dg is not None:
        config["lambda_dg"] = args.lambda_dg
    if args.batch_size_per_domain is not None:
        config["batch_size_per_domain"] = args.batch_size_per_domain

    if not config.get("pacs_root"):
        raise ValueError("pacs_root must be set via config or --pacs_root")

    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        tqdm.write("[train] Enabled TF32 + cuDNN benchmark for Ampere (A100) optimization")

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "figures"), exist_ok=True)

    method = config["method"]
    tqdm.write(f"[task3/train] method={method}  device={device}")

    if method == "erm":
        ckpt = config.get("erm_checkpoint", "task2/results/source_only_checkpoint.pth")
        if not os.path.exists(ckpt):
            tqdm.write(f"[task3/train] WARNING: ERM checkpoint not found at {ckpt}.")
            tqdm.write("  Run task2/train.py --config task2/configs/source_only.yaml first.")
        else:
            tqdm.write(f"[task3/train] ERM checkpoint found at {ckpt}. No retraining needed.")
        return

    # Checkpoint path handling for parameter sweeps
    if method == "dan_dg":
        if args.lambda_dg is not None:
            config["checkpoint_path"] = os.path.join(
                config["output_dir"], f"dan_dg_checkpoint_ldg{args.lambda_dg}.pth"
            )
        elif "checkpoint_path" not in config:
            config["checkpoint_path"] = os.path.join(config["output_dir"], "dan_dg_checkpoint.pth")
    elif method == "sam":
        rho_val = config.get("rho", 0.05)
        config["checkpoint_path"] = os.path.join(
            config["output_dir"], f"sam_rho{rho_val}_checkpoint.pth"
        )

    num_workers = config.get("num_workers", 8)
    pin_memory = device.type == "cuda"

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
            shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory,
        )
        source_val_loaders[domain] = DataLoader(
            ds_val, batch_size=config.get("val_batch_size", 64), shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
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

    # Save training curves / history JSON
    tag = method
    if method == "dan_dg" and args.lambda_dg is not None:
        tag = f"dan_dg_ldg{args.lambda_dg}"
    elif method == "sam":
        tag = f"sam_rho{config.get('rho', 0.05)}"

    curves_path = os.path.join(config["output_dir"], f"{tag}_training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(history, f, indent=2, default=float)
    tqdm.write(f"[task3/train] Training curves saved to {curves_path}")
    tqdm.write(f"[task3/train] Done. Checkpoint saved to: {config['checkpoint_path']}")


if __name__ == "__main__":
    main()

