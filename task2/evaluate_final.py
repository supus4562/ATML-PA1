"""task2/evaluate_final.py — Complete final evaluation for all Task 2 methods.

Produces all PA-required deliverables:
  Required Evidence (PA §5, §6):
  ✓ Table: Source-only, DAN, DANN, CDAN on each source val domain, mean source
    acc/F1, target acc/F1, target Δ, domain separability
  ✓ Training curve plots (cls + align loss, val macro-F1) for every method
  ✓ Per-class target accuracy changes vs Source-only
  ✓ Top confusions / failure cases per method
  ✓ Controlled DAN design study: λ_MMD ∈ {0.1, 1, 10} — table + plot

Output files:
  task2/results/
    final_results.json          — full results dict (all methods incl. DAN variants)
    final_results.csv           — scalar metrics, one row per method
    per_class_results.csv       — per-class Sketch accuracy per method
    top_confusions.csv          — top-5 confusion pairs per method
    figures/
      {method}_curves.png       — training curves for each method
      {method}_confusion.png    — row-normalised confusion matrix on Sketch
      per_class_delta.png       — grouped bar: ΔAcc vs Source-only
      controlled_study_lambda_mmd.png

Usage:
  python task2/evaluate_final.py --pacs_root /path/to/pacs
  python task2/evaluate_final.py --pacs_root /path/to/pacs --output_dir task2/results
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

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
from task2.evaluation.class_analysis import (
    PACS_CLASSES,
    per_class_accuracy,
    top_confusions,
    plot_per_class_delta,
    plot_confusion_matrix,
)

SEED = 6304
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Method display names ──────────────────────────────────────────────────────
METHOD_LABELS = {
    "source_only":       "Source-only (ERM)",
    "dan":               "DAN (λ=1)",
    "dan_lmmd0.1":       "DAN (λ=0.1)",
    "dan_lmmd1.0":       "DAN (λ=1.0)",
    "dan_lmmd10.0":      "DAN (λ=10)",
    "dann":              "DANN",
    "cdan":              "CDAN",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _method_key_from_path(ckpt_path: str) -> str:
    """Derive a unique result key from the checkpoint filename.

    Examples:
        source_only_checkpoint.pth  → "source_only"
        dan_checkpoint.pth          → "dan"
        dan_checkpoint_lmmd0.1.pth  → "dan_lmmd0.1"
        dann_checkpoint.pth         → "dann"
        cdan_checkpoint.pth         → "cdan"
    """
    name = os.path.basename(ckpt_path)           # e.g. "dan_checkpoint_lmmd0.1.pth"
    name = name.replace("_checkpoint", "")       # "dan_lmmd0.1.pth"
    name = name.replace(".pth", "")              # "dan_lmmd0.1"
    return name


def evaluate_loader(
    backbone: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate a single loader. Returns accuracy, macro_f1, features, preds, targets."""
    backbone.eval()
    classifier.eval()
    preds_list, targets_list, feats_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feat   = backbone(x)
            logits = classifier(feat)
            feats_list.append(feat.cpu().numpy())
            preds_list.append(logits.argmax(dim=1).cpu().numpy())
            targets_list.append(y.cpu().numpy())
    preds   = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)
    feats   = np.concatenate(feats_list)
    m = calculate_metrics(
        torch.from_numpy(targets),
        torch.from_numpy(preds),
    )
    return {
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "features": feats,
        "preds":    preds,
        "targets":  targets,
    }


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load backbone + classifier, returning (backbone, classifier, config_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    feature_dim = config.get("feature_dim", 512)
    n_classes   = config.get("n_classes", 7)

    bb  = ResNet18Backbone(pretrained=False).to(device)
    cls = ClassifierHead(in_dim=feature_dim, n_classes=n_classes).to(device)

    # Handle both Task 2 and Task 3 backbone key prefixes
    bb_sd = ckpt["backbone_state_dict"]
    if any(k.startswith("features.conv1.") for k in bb_sd):
        mapping = {
            "features.conv1.": "features.0.",
            "features.bn1.":   "features.1.",
            "features.layer1.": "features.4.",
            "features.layer2.": "features.5.",
            "features.layer3.": "features.6.",
            "features.layer4.": "features.7.",
        }
        new_sd = {}
        for k, v in bb_sd.items():
            new_k = k
            for old_p, new_p in mapping.items():
                if k.startswith(old_p):
                    new_k = new_p + k[len(old_p):]
                    break
            new_sd[new_k] = v
        bb_sd = new_sd

    bb.load_state_dict(bb_sd)
    cls.load_state_dict(ckpt["head_state_dict"])
    return bb, cls, config


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(method_key: str, output_dir: str) -> None:
    """Plot cls + align loss and val macro-F1 from saved JSON curves."""
    # Try to find the curve file by method key pattern
    curves_candidates = [
        os.path.join(output_dir, f"{method_key}_training_curves.json"),
        # dan_lmmd0.1 → may be stored as "dan_training_curves.json" if run was before fix
    ]
    curves_file = next((f for f in curves_candidates if os.path.exists(f)), None)
    if curves_file is None:
        return

    with open(curves_file) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Loss panel
    axes[0].plot(history.get("train_loss", []), label="Cls Loss", color="steelblue")
    if history.get("align_loss"):
        axes[0].plot(history["align_loss"], label="Align/Dom Loss", color="tomato", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title(f"{METHOD_LABELS.get(method_key, method_key)} — Training Loss")

    # F1 panel
    axes[1].plot(history.get("val_macro_f1", []), label="Val Macro-F1", color="seagreen")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].set_title(f"{METHOD_LABELS.get(method_key, method_key)} — Validation Macro-F1")

    out = os.path.join(output_dir, "figures", f"{method_key}_curves.png")
    savefig(out, fig)


def plot_controlled_study(all_results: dict, output_dir: str) -> None:
    """Plot DAN controlled study: target acc + domain separability vs λ_MMD."""
    # Collect the three DAN λ variants
    study_data = {}
    for lmmd, key in [(0.1, "dan_lmmd0.1"), (1.0, "dan"), (10.0, "dan_lmmd10.0")]:
        # Accept both "dan" and "dan_lmmd1.0" keys for the λ=1 main run
        actual_key = key if key in all_results else ("dan_lmmd1.0" if lmmd == 1.0 and "dan_lmmd1.0" in all_results else None)
        if actual_key and actual_key in all_results:
            study_data[lmmd] = all_results[actual_key]

    if len(study_data) < 2:
        tqdm.write("[controlled_study] Not enough DAN λ variants found — skipping plot.")
        return

    lambdas     = sorted(study_data.keys())
    target_accs = [study_data[l]["target_acc"]         for l in lambdas]
    dom_seps    = [study_data[l]["domain_separability"] for l in lambdas]
    src_accs    = [study_data[l]["mean_source_val_acc"] for l in lambdas]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    x_ticks = list(range(len(lambdas)))
    xlabels = [f"λ={l}" for l in lambdas]

    def _bar(ax, values, ylabel, title, color):
        bars = ax.bar(x_ticks, values, color=color, alpha=0.8, edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)

    _bar(ax1, target_accs, "Target (Sketch) Accuracy",
         "DAN: Target Acc vs λ_MMD", "steelblue")
    _bar(ax2, dom_seps,    "Domain Separability",
         "DAN: Domain Sep vs λ_MMD", "tomato")
    _bar(ax3, src_accs,    "Mean Source Val Accuracy",
         "DAN: Source Acc vs λ_MMD", "seagreen")

    fig.suptitle(
        "Controlled DAN Design Study — λ_MMD ∈ {0.1, 1, 10}\n"
        "(Main comparison uses λ=1; target results for analysis only)",
        fontsize=11,
    )
    out = os.path.join(output_dir, "figures", "controlled_study_lambda_mmd.png")
    savefig(out, fig)
    tqdm.write(f"  [saved] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 — Final evaluation (all methods)")
    parser.add_argument("--pacs_root",  required=True)
    parser.add_argument("--output_dir", default="task2/results")
    args = parser.parse_args()

    apply_style()
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"[evaluate_final] device={device}  output_dir={args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    # ── Build data loaders ────────────────────────────────────────────────────
    splits = load_or_create_splits(args.pacs_root, seed=SEED)

    source_val_loaders: dict[str, DataLoader] = {}
    for domain in SOURCE_DOMAINS:
        ds = PACSDataset(
            args.pacs_root, domain=domain,
            transform=VAL_TRANSFORM, indices=splits[domain]["val"],
        )
        source_val_loaders[domain] = DataLoader(
            ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
        )

    # Target: full Sketch set — labels used ONLY at final analysis
    sketch_ds = PACSDataset(args.pacs_root, domain="sketch", transform=VAL_TRANSFORM)
    sketch_loader = DataLoader(
        sketch_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Discover checkpoints ──────────────────────────────────────────────────
    ckpt_paths = sorted(glob.glob(os.path.join(args.output_dir, "*_checkpoint*.pth")))
    if not ckpt_paths:
        tqdm.write(f"[evaluate_final] No checkpoints found in {args.output_dir}. Run train.py first.")
        sys.exit(1)

    tqdm.write(f"[evaluate_final] Found {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        tqdm.write(f"  {os.path.basename(p)}")

    # ── Per-checkpoint evaluation ─────────────────────────────────────────────
    all_results: dict[str, dict]    = {}
    sketch_preds_by_method: dict[str, np.ndarray] = {}
    source_only_target_acc: Optional[float] = None

    for ckpt_path in tqdm(ckpt_paths, desc="checkpoints"):
        method_key = _method_key_from_path(ckpt_path)
        try:
            bb, cls, config = load_checkpoint(ckpt_path, device)
        except Exception as e:
            tqdm.write(f"  [WARN] Could not load {ckpt_path}: {e}")
            continue

        label = METHOD_LABELS.get(method_key, method_key)
        tqdm.write(f"\n[evaluate_final] {label}  ({os.path.basename(ckpt_path)})")

        res: dict = {}
        source_feats_all: list[np.ndarray] = []

        # Source validation
        source_accs, source_f1s = [], []
        for domain in SOURCE_DOMAINS:
            ev = evaluate_loader(bb, cls, source_val_loaders[domain], device)
            res[f"{domain}_val_acc"] = float(ev["accuracy"])
            res[f"{domain}_val_f1"]  = float(ev["macro_f1"])
            source_accs.append(ev["accuracy"])
            source_f1s.append(ev["macro_f1"])
            source_feats_all.append(ev["features"])
            tqdm.write(f"  {domain}: acc={ev['accuracy']:.4f}  f1={ev['macro_f1']:.4f}")

        res["mean_source_val_acc"] = float(np.mean(source_accs))
        res["worst_source_val_acc"] = float(np.min(source_accs))
        res["mean_source_val_f1"]  = float(np.mean(source_f1s))
        res["worst_source_val_f1"] = float(np.min(source_f1s))

        # Target (Sketch) — labels used for final analysis
        sketch_ev = evaluate_loader(bb, cls, sketch_loader, device)
        res["target_acc"] = float(sketch_ev["accuracy"])
        res["target_f1"]  = float(sketch_ev["macro_f1"])
        tqdm.write(f"  Sketch: acc={sketch_ev['accuracy']:.4f}  f1={sketch_ev['macro_f1']:.4f}")

        if method_key == "source_only":
            source_only_target_acc = sketch_ev["accuracy"]

        # Domain separability: binary logistic regression (source vs target)
        # PA spec: 70/30 split, C=1, seed 6304
        src_feats = np.concatenate(source_feats_all)
        tgt_feats = sketch_ev["features"]
        sep = compute_domain_separability(src_feats, tgt_feats, seed=SEED)
        res["domain_separability"] = float(sep)
        tqdm.write(f"  Domain separability: {sep:.4f}  (0.50=chance, 1.0=fully separable)")

        # Per-class accuracy on Sketch
        pc_acc = per_class_accuracy(sketch_ev["targets"], sketch_ev["preds"], n_classes=7)
        res["per_class_target_acc"] = pc_acc.tolist()

        # Top-5 confusions
        conf = top_confusions(sketch_ev["targets"], sketch_ev["preds"], PACS_CLASSES, top_k=5)
        res["top_confusions"] = conf

        # Confusion matrix figure
        cm_path = os.path.join(args.output_dir, "figures", f"{method_key}_confusion.png")
        plot_confusion_matrix(
            sketch_ev["targets"], sketch_ev["preds"],
            PACS_CLASSES, label, cm_path,
        )

        # Training curves figure
        plot_training_curves(method_key, args.output_dir)

        all_results[method_key]          = res
        sketch_preds_by_method[method_key] = sketch_ev["preds"]

    # ── Compute Δ vs source-only ──────────────────────────────────────────────
    if source_only_target_acc is not None:
        for res in all_results.values():
            res["target_delta_acc"] = float(res["target_acc"] - source_only_target_acc)

    # ── Per-class delta figure (all main methods) ─────────────────────────────
    MAIN_METHODS = ["dan", "dann", "cdan"]
    if "source_only" in all_results:
        src_pc = np.array(all_results["source_only"]["per_class_target_acc"])
        method_pc = {
            METHOD_LABELS.get(m, m): np.array(all_results[m]["per_class_target_acc"])
            for m in MAIN_METHODS if m in all_results
        }
        if method_pc:
            plot_per_class_delta(
                src_pc, method_pc, PACS_CLASSES,
                os.path.join(args.output_dir, "figures", "per_class_delta.png"),
            )

    # ── Controlled DAN study figure ───────────────────────────────────────────
    plot_controlled_study(all_results, args.output_dir)

    # ── Save final_results.json ───────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "final_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    tqdm.write(f"\n[evaluate_final] Saved {json_path}")

    # ── Save final_results.csv (scalar metrics only) ─────────────────────────
    SCALAR_KEYS = [
        "photo_val_acc", "photo_val_f1",
        "art_painting_val_acc", "art_painting_val_f1",
        "cartoon_val_acc", "cartoon_val_f1",
        "mean_source_val_acc", "mean_source_val_f1",
        "worst_source_val_acc", "worst_source_val_f1",
        "target_acc", "target_f1",
        "target_delta_acc",
        "domain_separability",
    ]
    csv_path = os.path.join(args.output_dir, "final_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + SCALAR_KEYS, extrasaction="ignore")
        writer.writeheader()
        for m_key, res in all_results.items():
            row = {"method": METHOD_LABELS.get(m_key, m_key)}
            for k in SCALAR_KEYS:
                row[k] = res.get(k, "")
            writer.writerow(row)
    tqdm.write(f"[evaluate_final] Saved {csv_path}")

    # ── Save per_class_results.csv ────────────────────────────────────────────
    pc_csv = os.path.join(args.output_dir, "per_class_results.csv")
    with open(pc_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + PACS_CLASSES)
        writer.writeheader()
        for m_key, res in all_results.items():
            if "per_class_target_acc" not in res:
                continue
            row = {"method": METHOD_LABELS.get(m_key, m_key)}
            for cls_name, acc in zip(PACS_CLASSES, res["per_class_target_acc"]):
                row[cls_name] = f"{acc:.4f}"
            writer.writerow(row)
    tqdm.write(f"[evaluate_final] Saved {pc_csv}")

    # ── Save top_confusions.csv ───────────────────────────────────────────────
    conf_csv = os.path.join(args.output_dir, "top_confusions.csv")
    with open(conf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "rank", "true_class", "pred_class", "count"])
        writer.writeheader()
        for m_key, res in all_results.items():
            for rank, entry in enumerate(res.get("top_confusions", []), start=1):
                writer.writerow({
                    "method":     METHOD_LABELS.get(m_key, m_key),
                    "rank":       rank,
                    "true_class": entry["true"],
                    "pred_class": entry["pred"],
                    "count":      entry["count"],
                })
    tqdm.write(f"[evaluate_final] Saved {conf_csv}")

    # ── Summary table ─────────────────────────────────────────────────────────
    tqdm.write("\n" + "=" * 82)
    tqdm.write(
        f"{'Method':<22} {'Src Acc':<10} {'Worst':<8} {'Sketch Acc':<13}"
        f"{'ΔAcc':<10} {'Sep':<8}"
    )
    tqdm.write("-" * 82)
    # Print main methods first, then DAN variants
    ordered = ["source_only", "dan", "dann", "cdan", "dan_lmmd0.1", "dan_lmmd1.0", "dan_lmmd10.0"]
    for m_key in ordered + [k for k in all_results if k not in ordered]:
        if m_key not in all_results:
            continue
        res = all_results[m_key]
        label = METHOD_LABELS.get(m_key, m_key)
        tqdm.write(
            f"{label:<22} {res.get('mean_source_val_acc', 0):<10.4f}"
            f"{res.get('worst_source_val_acc', 0):<8.4f}"
            f"{res.get('target_acc', 0):<13.4f}"
            f"{res.get('target_delta_acc', 0):+<10.4f}"
            f"{res.get('domain_separability', 0):<8.4f}"
        )
    tqdm.write("=" * 82)

    # Per-class breakdown for main methods
    tqdm.write("\n── Per-Class Sketch Accuracy ──")
    cls_header = f"{'Method':<22}" + "".join(f"{c:>10}" for c in PACS_CLASSES)
    tqdm.write(cls_header)
    tqdm.write("-" * (22 + 10 * len(PACS_CLASSES)))
    for m_key in ["source_only", "dan", "dann", "cdan"]:
        if m_key not in all_results or "per_class_target_acc" not in all_results[m_key]:
            continue
        label = METHOD_LABELS.get(m_key, m_key)
        row_str = f"{label:<22}" + "".join(
            f"{v:>10.3f}" for v in all_results[m_key]["per_class_target_acc"]
        )
        tqdm.write(row_str)

    tqdm.write(f"\n[evaluate_final] Done. All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
