"""task2/train.py — Main training dispatch for Task 2 (UDA).

Usage:
  python task2/train.py --config task2/configs/source_only.yaml --pacs_root /path/to/pacs
  python task2/train.py --config task2/configs/dan.yaml         --pacs_root /path/to/pacs
  python task2/train.py --config task2/configs/dann.yaml        --pacs_root /path/to/pacs
  python task2/train.py --config task2/configs/cdan.yaml        --pacs_root /path/to/pacs
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import yaml

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Allow imports from repo root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from common.seed import set_all_seeds
from shared.pacs import PACSDataset
from shared.pacs_protocol import load_or_create_splits
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead
from task2.methods.source_only import SourceOnlyTrainer
from task2.methods.dan import DANTrainer
from task2.methods.dann import DANNTrainer
from task2.methods.cdan import CDANTrainer


# ── Transforms ────────────────────────────────────────────────────────────────

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


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML config, merging from base.yaml if 'defaults' key present."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        base_path = os.path.join(os.path.dirname(config_path), "base.yaml")
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if k != "defaults":
                base_cfg[k] = v
        return base_cfg
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 — Unsupervised Domain Adaptation")
    parser.add_argument("--config", required=True, help="Path to method config YAML")
    parser.add_argument("--pacs_root", default=None, help="Override PACS root directory")
    parser.add_argument("--lambda_mmd", type=float, default=None,
                        help="Override lambda_mmd (DAN controlled study)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.pacs_root:
        config["pacs_root"] = args.pacs_root
    if args.lambda_mmd is not None and "lambda_mmd" in config:
        config["lambda_mmd"] = args.lambda_mmd

    if not config.get("pacs_root"):
        raise ValueError("pacs_root must be set via config or --pacs_root")

    set_all_seeds(config["seed"])
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "figures"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        tqdm.write("[train] Enabled TF32 + cuDNN benchmark for Ampere (A100) optimization")
    tqdm.write(f"[train] device={device}  method={config['method']}")

    num_workers = config.get("num_workers", 8)
    pin_memory  = device.type == "cuda"

    # ── Build splits ──────────────────────────────────────────────────────────
    splits = load_or_create_splits(config["pacs_root"], seed=config["seed"])

    # ── Source loaders ────────────────────────────────────────────────────────
    source_train_loaders = []
    source_val_loaders = []
    for domain in config["source_domains"]:
        train_indices = splits[domain]["train"]
        val_indices = splits[domain]["val"]

        ds_train = PACSDataset(config["pacs_root"], domain=domain,
                               transform=TRAIN_TRANSFORM, indices=train_indices)
        ds_val = PACSDataset(config["pacs_root"], domain=domain,
                             transform=VAL_TRANSFORM, indices=val_indices)

        source_train_loaders.append(
            DataLoader(ds_train, batch_size=config["batch_size_per_domain"],
                       shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
        )
        source_val_loaders.append(
            DataLoader(ds_val, batch_size=64, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)
        )

    # ── Target (Sketch) loader — NO LABELS used during training ──────────────
    target_ds = PACSDataset(config["pacs_root"], domain=config["target_domain"],
                            transform=TRAIN_TRANSFORM)
    target_loader = DataLoader(target_ds, batch_size=config["target_batch_size"],
                               shuffle=True, drop_last=True,
                               num_workers=num_workers, pin_memory=pin_memory)

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone = ResNet18Backbone(pretrained=True).to(device)
    classifier = ClassifierHead(
        in_dim=config["feature_dim"], n_classes=config["n_classes"]
    ).to(device)

    # ── Trainer ───────────────────────────────────────────────────────────────
    method = config["method"]
    # Build checkpoint path (handle controlled study lambda variants)
    if method == "dan" and args.lambda_mmd is not None:
        lmbd_tag = f"_lmmd{args.lambda_mmd}"
        base_ckpt = config.get("checkpoint_path", f"task2/results/dan_checkpoint.pth")
        base, ext = os.path.splitext(base_ckpt)
        config["checkpoint_path"] = f"{base}{lmbd_tag}{ext}"

    os.makedirs(os.path.dirname(os.path.abspath(
        config.get("checkpoint_path", f"task2/results/{method}_checkpoint.pth")
    )), exist_ok=True)
    if "checkpoint_path" not in config:
        config["checkpoint_path"] = f"task2/results/{method}_checkpoint.pth"

    if method == "source_only":
        trainer = SourceOnlyTrainer(backbone, classifier, config, device)
        history = trainer.train(source_train_loaders, target_loader, source_val_loaders)
    elif method == "dan":
        trainer = DANTrainer(backbone, classifier, config, device)
        history = trainer.train(source_train_loaders, target_loader, source_val_loaders)
    elif method == "dann":
        trainer = DANNTrainer(backbone, classifier, config, device)
        history = trainer.train(source_train_loaders, target_loader, source_val_loaders)
    elif method == "cdan":
        trainer = CDANTrainer(backbone, classifier, config, device)
        history = trainer.train(source_train_loaders, target_loader, source_val_loaders)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Save training curves ──────────────────────────────────────────────────
    curves_path = os.path.join(config["output_dir"], f"{method}_training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(history, f, indent=2)
    tqdm.write(f"[train] Training curves saved to {curves_path}")
    tqdm.write(f"[train] Done. Best checkpoint: {config['checkpoint_path']}")


if __name__ == "__main__":
    main()
