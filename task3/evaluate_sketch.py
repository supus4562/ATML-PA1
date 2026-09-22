
"""task3/evaluate_sketch.py — Final evaluation on Sketch (ONLY file that loads Sketch).

CRITICAL: This is the ONLY place where Sketch images are loaded in Task 3.
No training script, method, or selection module may import or load Sketch data.

Usage:
  python task3/evaluate_sketch.py --pacs_root /path/to/pacs --output_dir task3/results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
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
from common.metrics import calculate_accuracy, calculate_macro_f1
from common.plotting import apply_style, savefig
from shared.pacs import PACSDataset
from shared.pacs_protocol import load_or_create_splits
from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead
from task3.evaluation.source_domain_separability import compute_3class_separability
from task3.evaluation.sharpness import compute_sharpness
from task3.evaluation.domain_metrics import get_mean_metrics, get_worst_metrics
from task3.evaluation.class_analysis import (
    PACS_CLASSES,
    per_class_accuracy,
    top_confusions,
    plot_per_class_delta,
    plot_confusion_matrix,
)

SEED = 6304

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
ERM_CHECKPOINT = "task2/results/source_only_checkpoint.pth"


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
    all_feats = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = backbone(x)
            logits = classifier(feats)
            preds = torch.argmax(logits, dim=1)
            all_feats.append(feats.cpu().numpy())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    features = np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 512))
    return {
        "accuracy": float(calculate_accuracy(preds, labels)),
        "macro_f1": float(calculate_macro_f1(preds, labels)),
        "preds": preds,
        "labels": labels,
        "features": features,
    }


def plot_training_curves(output_dir: str, method_tag: str) -> None:
    """Plot training classification loss, align loss (if available), and val macro-F1."""
    possible_files = [
        os.path.join(output_dir, f"{method_tag}_training_curves.json"),
        os.path.join(output_dir, f"{method_tag}_training_history.json"),
    ]
    curves_file = None
    for f in possible_files:
        if os.path.exists(f):
            curves_file = f
            break
    if not curves_file:
        return

    try:
        with open(curves_file) as f:
            history = json.load(f)

        train_loss = history.get("train_loss", [])
        align_loss = history.get("align_loss", [])
        val_f1 = history.get("val_macro_f1", [])

        # If history was saved as list of dicts:
        if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
            train_loss = [h.get("train_loss", 0.0) for h in history]
            align_loss = [h.get("align_loss", 0.0) for h in history if "align_loss" in h]
            val_f1 = [h.get("mean_f1", h.get("val_macro_f1", 0.0)) for h in history]

        if not train_loss:
            return

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs = list(range(1, len(train_loss) + 1))
        axes[0].plot(epochs, train_loss, label="Cls Loss", color="#1f77b4", marker="o", markersize=3)
        if align_loss and any(a > 0 for a in align_loss):
            axes[0].plot(epochs[:len(align_loss)], align_loss, label="MMD Penalty", color="#ff7f0e", marker="s", markersize=3)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{method_tag} — Training Loss")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        if val_f1:
            axes[1].plot(list(range(1, len(val_f1) + 1)), val_f1, label="Val Macro-F1", color="#2ca02c", marker="^", markersize=3)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Macro-F1")
            axes[1].set_title(f"{method_tag} — Source Val Macro-F1")
            axes[1].legend()
            axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        savefig(os.path.join(output_dir, "figures", f"{method_tag}_curves.png"), fig)
    except Exception as e:
        tqdm.write(f"  [WARNING] Could not plot training curves for {method_tag}: {e}")


def plot_sam_study(output_dir: str, sam_results: dict[float, dict]) -> None:
    """Plot SAM study varying rho in {0.01, 0.05, 0.1}."""
    if len(sam_results) < 2:
        return
    sorted_rhos = sorted(sam_results.keys())
    sketch_accs = [sam_results[r]["sketch"]["accuracy"] for r in sorted_rhos]
    mean_src_accs = [sam_results[r]["mean_source"]["accuracy"] for r in sorted_rhos]
    sharpness = [sam_results[r]["sharpness_proxy"] for r in sorted_rhos]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(sorted_rhos, sketch_accs, marker="o", color="#1f77b4", label="Sketch Acc", linewidth=2)
    l2 = ax1.plot(sorted_rhos, mean_src_accs, marker="s", color="#2ca02c", linestyle="--", label="Mean Src Acc", linewidth=2)
    l3 = ax2.plot(sorted_rhos, sharpness, marker="^", color="#d62728", linestyle=":", label="Sharpness Proxy Δ_sharp", linewidth=2)

    ax1.set_xlabel("SAM Perturbation Radius ρ")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Sharpness Proxy Δ_sharp", color="#d62728")
    ax1.set_xscale("log")
    ax1.set_xticks(sorted_rhos)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("SAM Controlled Study (ρ ∈ {0.01, 0.05, 0.1})", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    savefig(os.path.join(output_dir, "figures", "controlled_study_sam_rho.png"), fig)


def plot_dan_study(output_dir: str, dan_results: dict[float, dict]) -> None:
    """Plot DAN-DG study varying lambda_dg in {0.1, 1.0, 10.0}."""
    if len(dan_results) < 2:
        return
    sorted_lambdas = sorted(dan_results.keys())
    sketch_accs = [dan_results[l]["sketch"]["accuracy"] for l in sorted_lambdas]
    mean_src_accs = [dan_results[l]["mean_source"]["accuracy"] for l in sorted_lambdas]
    separability = [dan_results[l]["source_separability_3class"] for l in sorted_lambdas]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(sorted_lambdas, sketch_accs, marker="o", color="#1f77b4", label="Sketch Acc", linewidth=2)
    l2 = ax1.plot(sorted_lambdas, mean_src_accs, marker="s", color="#2ca02c", linestyle="--", label="Mean Src Acc", linewidth=2)
    l3 = ax2.plot(sorted_lambdas, separability, marker="^", color="#9467bd", linestyle=":", label="Source 3-Class Sep", linewidth=2)

    ax1.set_xlabel("DAN-DG Alignment Weight λ_DG")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Source 3-Class Separability", color="#9467bd")
    ax1.set_xscale("log")
    ax1.set_xticks(sorted_lambdas)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("DAN-DG Controlled Study (λ_DG ∈ {0.1, 1, 10})", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    savefig(os.path.join(output_dir, "figures", "controlled_study_lambda_dg.png"), fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 — Sketch evaluation (Sketch loaded here only)")
    parser.add_argument("--pacs_root", required=True)
    parser.add_argument("--output_dir", default="task3/results")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    args = parser.parse_args()

    apply_style()
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        tqdm.write("[evaluate_sketch] Enabled TF32 + cuDNN benchmark for Ampere (A100) optimization")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    num_workers = args.num_workers
    pin_memory = device.type == "cuda"
    batch_size = args.eval_batch_size

    # ── Build source val loaders ───────────────────────────────────────────────
    splits = load_or_create_splits(args.pacs_root, seed=SEED)
    source_val_loaders: dict[str, DataLoader] = {}
    for domain in SOURCE_DOMAINS:
        ds = PACSDataset(args.pacs_root, domain=domain,
                         transform=VAL_TRANSFORM, indices=splits[domain]["val"])
        source_val_loaders[domain] = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    # ── *** ONLY SKETCH LOAD HERE *** ─────────────────────────────────────────
    sketch_ds = PACSDataset(args.pacs_root, domain="sketch", transform=VAL_TRANSFORM)
    sketch_loader = DataLoader(
        sketch_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # ── Collect checkpoints ───────────────────────────────────────────────────
    checkpoints = {}
    if os.path.exists(ERM_CHECKPOINT):
        checkpoints["ERM"] = ERM_CHECKPOINT
    else:
        tqdm.write(f"[WARNING] ERM checkpoint not found at {ERM_CHECKPOINT}")

    # Standard DAN-DG
    dan_path = os.path.join(args.output_dir, "dan_dg_checkpoint.pth")
    if os.path.exists(dan_path):
        checkpoints["DAN-DG"] = dan_path

    # DAN-DG study sweeps
    for ldg in [0.1, 1.0, 10.0]:
        ldg_path = os.path.join(args.output_dir, f"dan_dg_checkpoint_ldg{ldg}.pth")
        if os.path.exists(ldg_path):
            checkpoints[f"DAN-DG (λ={ldg})"] = ldg_path

    # Standard SAM or sweeps
    sam_std_path = os.path.join(args.output_dir, "sam_checkpoint.pth")
    if os.path.exists(sam_std_path) and "SAM (ρ=0.05)" not in checkpoints:
        checkpoints["SAM"] = sam_std_path

    for rho in [0.01, 0.05, 0.1]:
        sam_path = os.path.join(args.output_dir, f"sam_rho{rho}_checkpoint.pth")
        if os.path.exists(sam_path):
            checkpoints[f"SAM (ρ={rho})"] = sam_path

    # Any other custom checkpoint matching *_checkpoint*.pth in output_dir
    for ckpt_f in sorted(glob.glob(os.path.join(args.output_dir, "*_checkpoint*.pth"))):
        base_name = os.path.basename(ckpt_f).replace("_checkpoint", "").replace(".pth", "")
        if ckpt_f not in checkpoints.values():
            checkpoints[base_name] = ckpt_f

    if not checkpoints:
        tqdm.write("[ERROR] No checkpoints found. Run train.py first.")
        sys.exit(1)

    tqdm.write(f"\n[evaluate_sketch] Found {len(checkpoints)} checkpoint(s) to evaluate: {list(checkpoints.keys())}")

    # ── Evaluate each method ──────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    sketch_per_class_accs: dict[str, np.ndarray] = {}
    erm_sketch_acc: float | None = None
    erm_per_class_acc: np.ndarray | None = None

    sam_study_results: dict[float, dict] = {}
    dan_study_results: dict[float, dict] = {}

    for method_name, ckpt_path in tqdm(checkpoints.items(), desc="Evaluating checkpoints"):
        tqdm.write(f"\n[evaluate_sketch] Evaluating {method_name} ({os.path.basename(ckpt_path)}) ...")
        bb, cls = load_checkpoint(ckpt_path, device)

        # Source val per-domain
        source_res: dict[str, dict] = {}
        for dname, loader in source_val_loaders.items():
            res = evaluate_on_loader(bb, cls, loader, device)
            source_res[dname] = {
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
            }
            tqdm.write(f"  {dname}: acc={res['accuracy']:.4f}  f1={res['macro_f1']:.4f}")

        mean_src = get_mean_metrics(source_res)
        worst_src = get_worst_metrics(source_res)

        # Sketch evaluation
        sketch_res = evaluate_on_loader(bb, cls, sketch_loader, device)
        tqdm.write(f"  Sketch: acc={sketch_res['accuracy']:.4f}  f1={sketch_res['macro_f1']:.4f}")

        # Per-class accuracy & top confusions on Sketch
        cls_accs = per_class_accuracy(sketch_res["labels"], sketch_res["preds"], n_classes=len(PACS_CLASSES))
        sketch_per_class_accs[method_name] = cls_accs
        confs = top_confusions(sketch_res["labels"], sketch_res["preds"], PACS_CLASSES, top_k=5)

        clean_tag = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("ρ", "rho").replace("λ", "lambda")
        plot_confusion_matrix(
            sketch_res["labels"], sketch_res["preds"],
            PACS_CLASSES, method_name,
            os.path.join(args.output_dir, "figures", f"{clean_tag}_confusion.png"),
        )

        if method_name == "ERM":
            erm_sketch_acc = sketch_res["accuracy"]
            erm_per_class_acc = cls_accs

        # Domain separability (3-class, source only — no sketch)
        sep = compute_3class_separability(bb, source_val_loaders, device=device, seed=SEED)
        tqdm.write(f"  Source 3-class separability: {sep:.4f}")

        # Sharpness proxy Δ_sharp
        sharp = compute_sharpness(bb, cls, source_val_loaders, device=device, rho=0.05,
                                  n_per_domain=32, seed=SEED)
        tqdm.write(f"  Sharpness proxy Δ_sharp: {sharp:.6f}")

        # Plot training curves if available
        plot_training_curves(args.output_dir, clean_tag)

        eval_summary = {
            "source_per_domain": source_res,
            "mean_source": mean_src,
            "worst_source": worst_src,
            "sketch": {
                "accuracy": sketch_res["accuracy"],
                "macro_f1": sketch_res["macro_f1"],
            },
            "per_class_sketch_acc": {cls_name: float(acc) for cls_name, acc in zip(PACS_CLASSES, cls_accs)},
            "top_confusions": confs,
            "source_separability_3class": float(sep),
            "sharpness_proxy": float(sharp),
        }
        all_results[method_name] = eval_summary

        # Group into controlled studies if applicable
        if "SAM (ρ=" in method_name:
            try:
                rho_val = float(method_name.split("=")[1].replace(")", ""))
                sam_study_results[rho_val] = eval_summary
            except Exception:
                pass
        if "DAN-DG (λ=" in method_name:
            try:
                ldg_val = float(method_name.split("=")[1].replace(")", ""))
                dan_study_results[ldg_val] = eval_summary
            except Exception:
                pass

    # Compute sketch Δ vs ERM
    if erm_sketch_acc is not None:
        for name, res in all_results.items():
            res["sketch_delta_acc"] = res["sketch"]["accuracy"] - erm_sketch_acc

    # Per-class delta plot vs ERM
    if erm_per_class_acc is not None and len(sketch_per_class_accs) > 1:
        non_erm_accs = {k: v for k, v in sketch_per_class_accs.items() if k != "ERM"}
        plot_per_class_delta(
            erm_per_class_acc, non_erm_accs, PACS_CLASSES,
            os.path.join(args.output_dir, "figures", "per_class_delta.png"),
            baseline_name="ERM",
        )

    # Plot controlled studies if sweeps exist
    plot_sam_study(args.output_dir, sam_study_results)
    plot_dan_study(args.output_dir, dan_study_results)

    # ── Save results ─────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "final_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    tqdm.write(f"\n[evaluate_sketch] Results saved to {json_path}")

    # ── Build CSV table ───────────────────────────────────────────────────────
    rows = []
    for name, res in all_results.items():
        rows.append({
            "Method": name,
            "Sketch Acc": f"{res['sketch']['accuracy']:.4f}",
            "Sketch F1": f"{res['sketch']['macro_f1']:.4f}",
            "Sketch ΔAcc": f"{res.get('sketch_delta_acc', 0.0):+.4f}",
            "Mean Src Acc": f"{res['mean_source']['accuracy']:.4f}",
            "Worst Src Acc": f"{res['worst_source']['accuracy']:.4f}",
            "Separability": f"{res['source_separability_3class']:.4f}",
            "Sharpness": f"{res['sharpness_proxy']:.6f}",
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "final_results.csv")
    df.to_csv(csv_path, index=False)
    tqdm.write(f"[evaluate_sketch] CSV saved to {csv_path}")
    tqdm.write("\n" + df.to_string(index=False))

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
    tqdm.write(f"[evaluate_sketch] Comparison table figure saved to figures/comparison_table.png")

    # ── Task 2 vs Task 3 comparison (Target-Aware vs Target-Free) ─────────────
    task2_json = os.path.join("task2", "results", "final_results.json")
    if os.path.exists(task2_json):
        try:
            with open(task2_json) as f:
                t2_res = json.load(f)
            t2_dan_acc = t2_res.get("dan", {}).get("target_acc")
            t2_erm_acc = t2_res.get("source_only", {}).get("target_acc")
            t3_dan_acc = all_results.get("DAN-DG", {}).get("sketch", {}).get("accuracy")

            comp_data = {
                "shared_erm_sketch_acc": t2_erm_acc,
                "task2_dan_target_aware_acc": t2_dan_acc,
                "task2_dan_target_aware_delta": (t2_dan_acc - t2_erm_acc) if (t2_dan_acc is not None and t2_erm_acc is not None) else None,
                "task3_dan_dg_target_free_acc": t3_dan_acc,
                "task3_dan_dg_target_free_delta": (t3_dan_acc - t2_erm_acc) if (t3_dan_acc is not None and t2_erm_acc is not None) else None,
            }
            comp_path = os.path.join(args.output_dir, "task2_vs_task3_dan_comparison.json")
            with open(comp_path, "w") as f:
                json.dump(comp_data, f, indent=2)

            tqdm.write("\n" + "=" * 70)
            tqdm.write("Cross-Task Comparison: Target-Aware DAN (Task 2) vs Target-Free DAN-DG (Task 3)")
            tqdm.write("-" * 70)
            if t2_erm_acc is not None:
                tqdm.write(f"Shared ERM (Source-Only) Sketch Acc: {t2_erm_acc:.4f}")
            if t2_dan_acc is not None and t2_erm_acc is not None:
                tqdm.write(f"Task 2 DAN (sees unlabeled Sketch):  {t2_dan_acc:.4f} (Δ = {t2_dan_acc - t2_erm_acc:+.4f})")
            if t3_dan_acc is not None and t2_erm_acc is not None:
                tqdm.write(f"Task 3 DAN-DG (aligned sources only): {t3_dan_acc:.4f} (Δ = {t3_dan_acc - t2_erm_acc:+.4f})")
            tqdm.write("=" * 70)
            tqdm.write(f"[evaluate_sketch] Task 2 vs 3 comparison saved to {comp_path}")
        except Exception as e:
            tqdm.write(f"[evaluate_sketch] Note: Could not compute Task 2 vs Task 3 comparison: {e}")


if __name__ == "__main__":
    main()

