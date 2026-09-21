"""task2/evaluate_final.py — Final evaluation for all Task 2 methods.

Loads every saved checkpoint and reports:
  - Per-source-domain val accuracy + macro-F1 (Photo, Art, Cartoon)
  - Mean source val accuracy + macro-F1
  - Target (Sketch) accuracy + macro-F1
  - Target Δ relative to Source-only
  - Domain separability (binary logistic regression, source vs. target features)
  - DAN controlled study (λ_MMD ∈ {0.1, 1, 10})

Usage:
  python task2/evaluate_final.py --pacs_root /path/to/pacs --output_dir task2/results
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from common.seed import set_all_seeds
from common.metrics import calculate_metrics
from common.plotting import apply_style, savefig
from shared.pacs import PACSDataset
from shared.pacs_protocol import load_or_create_splits
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead
from task2.evaluation.domain_separability import compute_domain_separability

SEED = 6304
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def evaluate_loader(backbone, classifier, loader, device) -> dict:
    backbone.eval()
    classifier.eval()
    preds_list, targets_list = [], []
    feats_list = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feat = backbone(x)
            logits = classifier(feat)
            feats_list.append(feat.cpu().numpy())
            preds_list.append(logits.argmax(dim=1).cpu())
            targets_list.append(y.cpu())
    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)
    feats = np.concatenate(feats_list)
    m = calculate_metrics(targets, preds)
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "features": feats}


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load backbone + classifier from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    feature_dim = config.get("feature_dim", 512)
    n_classes = config.get("n_classes", 7)
    method = config.get("method", os.path.basename(ckpt_path).replace("_checkpoint.pth", ""))

    bb = ResNet18Backbone(pretrained=False).to(device)
    cls = ClassifierHead(in_dim=feature_dim, n_classes=n_classes).to(device)
    bb.load_state_dict(ckpt["backbone_state_dict"])
    cls.load_state_dict(ckpt["head_state_dict"])
    return bb, cls, method, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 — Final evaluation across all methods")
    parser.add_argument("--pacs_root", required=True)
    parser.add_argument("--output_dir", default="task2/results")
    args = parser.parse_args()

    apply_style()
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    # ── Build loaders ─────────────────────────────────────────────────────────
    splits = load_or_create_splits(args.pacs_root, seed=SEED)

    source_val_loaders: dict[str, DataLoader] = {}
    for domain in SOURCE_DOMAINS:
        ds = PACSDataset(args.pacs_root, domain=domain,
                         transform=VAL_TRANSFORM, indices=splits[domain]["val"])
        source_val_loaders[domain] = DataLoader(ds, batch_size=64, shuffle=False,
                                                 num_workers=2, pin_memory=True)

    # Target (Sketch) — full test set
    sketch_ds = PACSDataset(args.pacs_root, domain="sketch", transform=VAL_TRANSFORM)
    sketch_loader = DataLoader(sketch_ds, batch_size=64, shuffle=False,
                               num_workers=2, pin_memory=True)

    # ── Find all checkpoints ──────────────────────────────────────────────────
    ckpt_paths = sorted(glob.glob(os.path.join(args.output_dir, "*_checkpoint.pth")))
    if not ckpt_paths:
        print(f"[evaluate_final] No checkpoints found in {args.output_dir}. Run train.py first.")
        sys.exit(1)

    all_results: dict[str, dict] = {}
    source_only_target_acc: float | None = None

    for ckpt_path in ckpt_paths:
        try:
            bb, cls, method, config = load_checkpoint(ckpt_path, device)
        except Exception as e:
            print(f"[evaluate_final] Could not load {ckpt_path}: {e}")
            continue

        print(f"\n[evaluate_final] {method} ({os.path.basename(ckpt_path)})")

        method_res: dict = {}
        source_feats_all: list[np.ndarray] = []

        # Source val
        source_accs, source_f1s = [], []
        for domain in SOURCE_DOMAINS:
            res = evaluate_loader(bb, cls, source_val_loaders[domain], device)
            method_res[f"{domain}_val_acc"] = res["accuracy"]
            method_res[f"{domain}_val_f1"] = res["macro_f1"]
            source_accs.append(res["accuracy"])
            source_f1s.append(res["macro_f1"])
            source_feats_all.append(res["features"])
            print(f"  {domain}: acc={res['accuracy']:.4f}  f1={res['macro_f1']:.4f}")

        method_res["mean_source_val_acc"] = float(np.mean(source_accs))
        method_res["mean_source_val_f1"] = float(np.mean(source_f1s))

        # Target (Sketch)
        sketch_res = evaluate_loader(bb, cls, sketch_loader, device)
        method_res["target_acc"] = sketch_res["accuracy"]
        method_res["target_f1"] = sketch_res["macro_f1"]
        print(f"  Sketch: acc={sketch_res['accuracy']:.4f}  f1={sketch_res['macro_f1']:.4f}")

        if method == "source_only":
            source_only_target_acc = sketch_res["accuracy"]

        # Domain separability (binary: source=0, target=1)
        src_feats = np.concatenate(source_feats_all)
        tgt_feats = sketch_res["features"]
        sep = compute_domain_separability(src_feats, tgt_feats, seed=SEED)
        method_res["domain_separability"] = float(sep)
        print(f"  Domain separability: {sep:.4f}")

        all_results[method] = method_res

        # ── Training curve plot ───────────────────────────────────────────────
        curves_file = os.path.join(args.output_dir, f"{method}_training_curves.json")
        if os.path.exists(curves_file):
            with open(curves_file) as f:
                history = json.load(f)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(history.get("train_loss", []), label="Cls Loss")
            if "align_loss" in history and history["align_loss"]:
                axes[0].plot(history["align_loss"], label="Align Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].set_title(f"{method} — Training Loss")
            axes[1].plot(history.get("val_macro_f1", []), label="Val Macro-F1", color="green")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Macro-F1")
            axes[1].legend()
            axes[1].set_title(f"{method} — Validation Macro-F1")
            savefig(os.path.join(args.output_dir, "figures", f"{method}_curves.png"), fig)

    # Compute Δ vs source-only
    if source_only_target_acc is not None:
        for m, res in all_results.items():
            res["target_delta_acc"] = res["target_acc"] - source_only_target_acc

    # ── Save results ──────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "final_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[evaluate_final] Results saved to {json_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    if all_results:
        all_keys = sorted({k for res in all_results.values() for k in res if k != "features"})
        csv_path = os.path.join(args.output_dir, "final_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method"] + all_keys,
                                    extrasaction="ignore")
            writer.writeheader()
            for m, res in all_results.items():
                row = {k: res.get(k, "") for k in all_keys}
                row["method"] = m
                writer.writerow(row)
        print(f"[evaluate_final] CSV saved to {csv_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Method':<15} {'Src Acc':<10} {'Sketch Acc':<12} {'ΔAcc':<10} {'Sep':<8}")
    print("-" * 70)
    for m, res in all_results.items():
        print(f"{m:<15} {res.get('mean_source_val_acc', 0):<10.4f} "
              f"{res.get('target_acc', 0):<12.4f} "
              f"{res.get('target_delta_acc', 0):+<10.4f} "
              f"{res.get('domain_separability', 0):<8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
