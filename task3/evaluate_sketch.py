"""task3/evaluate_sketch.py — Final evaluation on Sketch (ONLY file that loads Sketch).

CRITICAL: This is the ONLY place where Sketch images are loaded in Task 3.
No training script, method, or selection module may import or load Sketch data.

Usage:
  python task3/evaluate_sketch.py --pacs_root /path/to/pacs --output_dir task3/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from common.seed import set_all_seeds
from common.metrics import calculate_accuracy, calculate_macro_f1
from common.plotting import apply_style, savefig
from shared.pacs import PACSDataset
from shared.pacs_protocol import load_or_create_splits
from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead
from task3.evaluation.source_domain_separability import compute_3class_separability
from task3.evaluation.sharpness import compute_sharpness
from task3.evaluation.domain_metrics import get_mean_metrics, get_worst_metrics

SEED = 6304

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
ERM_CHECKPOINT = "task2/results/source_only_checkpoint.pth"


def load_model(backbone_path: str | None, head_path: str | None,
               device: torch.device, n_classes: int = 7) -> tuple:
    """Load backbone + head from separate state_dict keys in a single checkpoint file."""
    bb = ResNet18Backbone(pretrained=False).to(device)
    cls = ClassifierHead(in_dim=512, n_classes=n_classes).to(device)
    return bb, cls


def load_checkpoint(checkpoint_path: str, device: torch.device, n_classes: int = 7):
    """Load backbone + classifier from a checkpoint dict."""
    bb = ResNet18Backbone(pretrained=False).to(device)
    cls = ClassifierHead(in_dim=512, n_classes=n_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bb.load_state_dict(ckpt["backbone_state_dict"])
    cls.load_state_dict(ckpt["head_state_dict"])
    return bb, cls


def evaluate_on_loader(backbone, classifier, loader, device) -> dict:
    backbone.eval()
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = classifier(backbone(x))
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "accuracy": float(calculate_accuracy(preds, labels)),
        "macro_f1": float(calculate_macro_f1(preds, labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 — Sketch evaluation (Sketch loaded here only)")
    parser.add_argument("--pacs_root", required=True)
    parser.add_argument("--output_dir", default="task3/results")
    args = parser.parse_args()

    apply_style()
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    # ── Build source val loaders ───────────────────────────────────────────────
    splits = load_or_create_splits(args.pacs_root, seed=SEED)
    source_val_loaders: dict[str, DataLoader] = {}
    for domain in SOURCE_DOMAINS:
        ds = PACSDataset(args.pacs_root, domain=domain,
                         transform=VAL_TRANSFORM, indices=splits[domain]["val"])
        source_val_loaders[domain] = DataLoader(ds, batch_size=64, shuffle=False,
                                                 num_workers=2, pin_memory=True)

    # ── *** ONLY SKETCH LOAD HERE *** ─────────────────────────────────────────
    sketch_ds = PACSDataset(args.pacs_root, domain="sketch", transform=VAL_TRANSFORM)
    sketch_loader = DataLoader(sketch_ds, batch_size=64, shuffle=False,
                               num_workers=2, pin_memory=True)

    # ── Collect checkpoints ───────────────────────────────────────────────────
    checkpoints = {}
    if os.path.exists(ERM_CHECKPOINT):
        checkpoints["ERM"] = ERM_CHECKPOINT
    else:
        print(f"[WARNING] ERM checkpoint not found at {ERM_CHECKPOINT}")

    dan_path = os.path.join(args.output_dir, "dan_dg_checkpoint.pth")
    if os.path.exists(dan_path):
        checkpoints["DAN-DG"] = dan_path

    for rho in [0.01, 0.05, 0.1]:
        sam_path = os.path.join(args.output_dir, f"sam_rho{rho}_checkpoint.pth")
        if os.path.exists(sam_path):
            checkpoints[f"SAM (ρ={rho})"] = sam_path

    if not checkpoints:
        print("[ERROR] No checkpoints found. Run train.py first.")
        sys.exit(1)

    # ── Evaluate each method ──────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    erm_sketch_acc: float | None = None

    for method_name, ckpt_path in checkpoints.items():
        print(f"\n[evaluate_sketch] Evaluating {method_name} ...")
        bb, cls = load_checkpoint(ckpt_path, device)

        # Source val per-domain
        source_res: dict[str, dict] = {}
        for dname, loader in source_val_loaders.items():
            source_res[dname] = evaluate_on_loader(bb, cls, loader, device)
            print(f"  {dname}: acc={source_res[dname]['accuracy']:.4f}  f1={source_res[dname]['macro_f1']:.4f}")

        mean_src = get_mean_metrics(source_res)
        worst_src = get_worst_metrics(source_res)

        # Sketch evaluation
        sketch_res = evaluate_on_loader(bb, cls, sketch_loader, device)
        print(f"  Sketch: acc={sketch_res['accuracy']:.4f}  f1={sketch_res['macro_f1']:.4f}")

        if method_name == "ERM":
            erm_sketch_acc = sketch_res["accuracy"]

        # Domain separability (3-class, source only — no sketch)
        sep = compute_3class_separability(bb, source_val_loaders, device=device, seed=SEED)
        print(f"  Source 3-class separability: {sep:.4f}")

        # Sharpness
        sharp = compute_sharpness(bb, cls, source_val_loaders, device=device, rho=0.05,
                                  n_per_domain=32, seed=SEED)
        print(f"  Sharpness proxy Δ_sharp: {sharp:.6f}")

        all_results[method_name] = {
            "source_per_domain": source_res,
            "mean_source": mean_src,
            "worst_source": worst_src,
            "sketch": sketch_res,
            "source_separability_3class": float(sep),
            "sharpness_proxy": float(sharp),
        }

    # Compute sketch Δ vs ERM
    if erm_sketch_acc is not None:
        for name, res in all_results.items():
            res["sketch_delta_acc"] = res["sketch"]["accuracy"] - erm_sketch_acc

    # ── Save results ─────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "final_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[evaluate_sketch] Results saved to {json_path}")

    # ── Build CSV table ───────────────────────────────────────────────────────
    rows = []
    for name, res in all_results.items():
        rows.append({
            "Method": name,
            "Sketch Acc": f"{res['sketch']['accuracy']:.4f}",
            "Sketch F1": f"{res['sketch']['macro_f1']:.4f}",
            "Sketch ΔAcc": f"{res.get('sketch_delta_acc', 0):+.4f}",
            "Mean Src Acc": f"{res['mean_source']['accuracy']:.4f}",
            "Worst Src Acc": f"{res['worst_source']['accuracy']:.4f}",
            "Separability": f"{res['source_separability_3class']:.4f}",
            "Sharpness": f"{res['sharpness_proxy']:.6f}",
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "final_results.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))

    # ── Table figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) + 1)))
    ax.axis("tight")
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    fig.suptitle("Task 3 — Domain Generalization Results", fontsize=12, fontweight="bold")
    savefig(os.path.join(args.output_dir, "figures", "comparison_table.png"), fig)
    print(f"[evaluate_sketch] Figure saved.")


if __name__ == "__main__":
    main()
