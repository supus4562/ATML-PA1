"""task1/scripts/run_task1.py — Main entry point for Task 1.

Runs all 6 steps and saves every required plot and metric automatically.

Usage:
  python task1/scripts/run_task1.py --data_root ./data/oxford-iiit-pet --output_dir task1/results
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from common.seed import set_all_seeds
from common.plotting import apply_style, savefig, model_colors
from task1.data.make_subset import make_balanced_subset
from task1.models.backbones import BackboneExtractor, CLIPZeroShot, MODEL_NAMES
from task1.analysis.evaluate_bias import (
    evaluate_clean, evaluate_color_bias, evaluate_cue_conflicts,
    evaluate_translation, evaluate_patch_shuffle,
)
from task1.analysis.feature_similarity import compute_all_stabilities
from task1.analysis.representation import run_tsne
from task1.data.make_cue_conflicts import generate_cue_conflicts
from task1.data.transforms import apply_grayscale, apply_translation, apply_patch_shuffle


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


class PILDataset(torch.utils.data.Dataset):
    def __init__(self, pils, labels, transform):
        self.pils = pils
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.pils)
    def __getitem__(self, i): return self.transform(self.pils[i]), self.labels[i]


class IndexDataset(torch.utils.data.Dataset):
    """Wraps a torchvision dataset and applies a transform."""
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, label = self.ds[i]
        return self.transform(img), label


class SubsetWithTransform(torch.utils.data.Dataset):
    """Wraps a subset of a dataset by indices and applies a transform."""
    def __init__(self, ds, indices, transform):
        self.ds = ds
        self.indices = indices
        self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, label = self.ds[self.indices[i]]
        return self.transform(img), label


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_clean_baseline(metrics: dict, out_dir: str) -> None:
    """Bar chart: clean accuracy + macro-F1 per model."""
    names = list(metrics.keys())
    accs = [metrics[m]["accuracy"] for m in names]
    f1s  = [metrics[m]["macro_f1"] for m in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy")
    bars2 = ax.bar(x + w/2, f1s,  w, label="Macro-F1")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Step 1 — Clean Baseline")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
    plt.tight_layout()
    savefig(os.path.join(out_dir, "figures", "step1_clean_baseline.png"), fig)


def plot_color_bias(metrics: dict, out_dir: str) -> None:
    """Grouped bars: accuracy and consistency for grayscale & hue rotation."""
    names = list(metrics.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, title in [
        (axes[0], "gray",  "Grayscale"),
        (axes[1], "hue",   "Hue Rotation (+120°)"),
    ]:
        accs  = [metrics[m][f"{key}scale_acc" if key == "gray" else f"{key}_acc"] for m in names]
        cons  = [metrics[m][f"{key}scale_consistency" if key == "gray" else f"{key}_consistency"] for m in names]
        x = np.arange(len(names)); w = 0.35
        b1 = ax.bar(x - w/2, accs, w, label="Accuracy")
        b2 = ax.bar(x + w/2, cons, w, label="Consistency")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title(f"Step 2 — {title}")
        ax.legend(); ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=7)
        ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=7)

    plt.tight_layout()
    savefig(os.path.join(out_dir, "figures", "step2_color_bias.png"), fig)


def plot_cue_conflicts(metrics: dict, out_dir: str) -> None:
    """Horizontal bar: shape bias % per model. Also plots coverage."""
    names = list(metrics.keys())
    shape_biases = [metrics[m]["shape_bias"] for m in names]
    coverages    = [metrics[m]["coverage"]    for m in names]
    colors = [model_colors().get(m, "#999") for m in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(names, shape_biases, color=colors)
    axes[0].axvline(50, ls="--", color="gray", label="No bias (50%)")
    axes[0].set_xlim(0, 100); axes[0].set_xlabel("Shape Bias (%)")
    axes[0].set_title("Step 3 — Shape Bias"); axes[0].legend()
    for i, v in enumerate(shape_biases):
        axes[0].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

    axes[1].barh(names, coverages, color=colors)
    axes[1].set_xlim(0, 100); axes[1].set_xlabel("Coverage (%)")
    axes[1].set_title("Step 3 — Cue Conflict Coverage")
    for i, v in enumerate(coverages):
        axes[1].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    savefig(os.path.join(out_dir, "figures", "step3_shape_bias.png"), fig)


def plot_translation(metrics: dict, out_dir: str) -> None:
    """Line plots: accuracy and consistency vs. displacement (4 lines, one per model)."""
    colors = model_colors()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for model_name, m_res in metrics.items():
        disps = sorted([int(d) for d in m_res.keys()])
        accs  = [m_res[str(d)]["acc"]         for d in disps]
        cons  = [m_res[str(d)]["consistency"] for d in disps]
        c = colors.get(model_name, "#999")
        axes[0].plot(disps, accs, marker="o", label=model_name, color=c)
        axes[1].plot(disps, cons, marker="o", label=model_name, color=c)

    for ax, ylabel, title in [
        (axes[0], "Accuracy",    "Step 4 — Translation: Accuracy vs Displacement"),
        (axes[1], "Consistency", "Step 4 — Translation: Consistency vs Displacement"),
    ]:
        ax.set_xlabel("Displacement δ (pixels)")
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
        ax.set_ylim(0, 1); ax.set_xticks([0, 8, 16, 32])

    plt.tight_layout()
    savefig(os.path.join(out_dir, "figures", "step4_translation.png"), fig)


def plot_patch_shuffle(metrics: dict, out_dir: str) -> None:
    """Bar chart: patch-shuffle accuracy drop and consistency."""
    names = list(metrics.keys())
    accs = [metrics[m]["acc"]         for m in names]
    cons = [metrics[m]["consistency"] for m in names]

    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy")
    b2 = ax.bar(x + w/2, cons, w, label="Consistency")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Step 5 — Patch Shuffle"); ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    plt.tight_layout()
    savefig(os.path.join(out_dir, "figures", "step5_patch_shuffle.png"), fig)


def plot_stability(metrics: dict, out_dir: str) -> None:
    """Grouped bars: cosine stability for each transform × model."""
    names = list(metrics.keys())
    transforms_order = ["gray", "patch", "translation_32", "cue_conflict"]
    labels_map = {
        "gray":           "Grayscale",
        "patch":          "Patch Shuffle",
        "translation_32": "Translation δ=32",
        "cue_conflict":   "Cue Conflict",
    }
    colors = model_colors()

    n_transforms = len(transforms_order)
    x = np.arange(n_transforms)
    w = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, name in enumerate(names):
        vals = [metrics[name].get(t, 0.0) for t in transforms_order]
        offset = (i - len(names)/2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=name, color=colors.get(name, "#999"))

    ax.set_xticks(x)
    ax.set_xticklabels([labels_map[t] for t in transforms_order], fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("Cosine Stability")
    ax.set_title("Step 6 — Representation Stability (cosine similarity)")
    ax.legend()
    plt.tight_layout()
    savefig(os.path.join(out_dir, "figures", "step6_stability.png"), fig)


def plot_mean_max_conf(metrics: dict, out_dir: str) -> None:
    """Bar chart: mean max confidence per model."""
    names = list(metrics.keys())
    confs = [metrics[m]["mean_max_conf"] for m in names]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(names, confs, color=[model_colors().get(m, "#999") for m in names])
    ax.set_ylim(0, 1); ax.set_ylabel("Mean Max Confidence")
    ax.set_title("Step 1 — Mean Max Confidence")
    for i, v in enumerate(confs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    savefig(os.path.join(out_dir, "figures", "step1_confidence.png"), fig)



def plot_cka_heatmap(features_dict: dict, title: str, out_path: str) -> None:
    from task1.analysis.feature_similarity import linear_cka
    names = list(features_dict.keys())
    n = len(names)
    cka_matrix = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i <= j:
                v = linear_cka(features_dict[n1], features_dict[n2])
                cka_matrix[i, j] = v
                cka_matrix[j, i] = v
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cka_matrix, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_title(title)
    
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cka_matrix[i, j]:.2f}", ha="center", va="center", color="black" if cka_matrix[i,j] < 0.5 else "white")
            
    savefig(out_path, fig)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 — Inductive Biases")
    parser.add_argument("--data_root",  default="./data/oxford-iiit-pet")
    parser.add_argument("--output_dir", default="task1/results")
    parser.add_argument("--config",     default="task1/configs/task1.yaml")
    parser.add_argument("--seed", type=int, default=6304)
    args = parser.parse_args()

    config = load_config(args.config)
    config["seed"]       = args.seed
    config["data_root"]  = args.data_root
    config["output_dir"] = args.output_dir

    apply_style()
    set_all_seeds(config["seed"])
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "figures"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # Re-apply A100 optimizations AFTER set_all_seeds (which sets deterministic=True)
        # TF32 must be explicitly re-enabled; benchmark=True is OK for inference-only pipeline
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # fast kernel selection; safe since input size is fixed
        print("[task1] Enabled TF32 for Ampere (A100) optimization")
    print(f"[task1] device={device}")

    # Batch / dataloader defaults
    train_bs = config["training"].get("batch_size", config.get("batch_size", 128))
    eval_bs = config.get("batch_size", train_bs)
    num_workers = config.get("num_workers", 4)
    train_workers = min(4, int(num_workers))
    eval_workers = 0  # In-memory PIL evaluation: 0 workers eliminates Docker IPC deadlock and queue hangs
    pin_memory = True if device.type == 'cuda' else False

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("[task1] Downloading / loading Oxford-IIIT Pet ...")
    trainval_dataset = OxfordIIITPet(root=config["data_root"], split="trainval", download=True)
    test_dataset     = OxfordIIITPet(root=config["data_root"], split="test",     download=True)

    config["classes"] = trainval_dataset.classes
    config["training"]["classes"] = trainval_dataset.classes

    # Stratified 80/20 train/val split from official training partition using seed 6304
    from sklearn.model_selection import train_test_split
    targets_trainval = np.array(
        trainval_dataset._labels if hasattr(trainval_dataset, '_labels') else trainval_dataset.targets
    )
    train_indices, val_indices = train_test_split(
        np.arange(len(targets_trainval)),
        test_size=0.20,
        random_state=config["seed"],
        stratify=targets_trainval
    )
    print(f"[task1] Created stratified 80/20 split: {len(train_indices)} train, {len(val_indices)} val")

    # Select class-balanced subset of 500 official test images using seed 6304
    subset_indices = make_balanced_subset(
        test_dataset,
        total_samples=config["subset"].get("total_test_images", 500),
        n_per_class=config["subset"].get("n_per_class", None),
        seed=config["seed"]
    )

    # Build shared 500 PIL images at 224×224 BEFORE any model normalization
    print(f"[task1] Building evaluation subset ({len(subset_indices)} images) ...")
    clean_pil_images: list[Image.Image] = []
    labels: list[int] = []
    for idx in subset_indices:
        img, label = test_dataset[idx]
        img = img.resize((224, 224), Image.BILINEAR)
        clean_pil_images.append(img)
        labels.append(label)
    labels_arr = np.array(labels)

    # Pre-generate per-image patch-shuffled images (non-identity per image, seed 6304)
    print("[task1] Pre-generating per-image patch permutations (seed 6304) ...")
    patch_shuffled_images: list[Image.Image] = []
    for idx, img in enumerate(clean_pil_images):
        rng = np.random.RandomState(config["seed"] + idx)
        perm = rng.permutation(16).tolist()
        while perm == list(range(16)):
            perm = rng.permutation(16).tolist()
        patch_shuffled_images.append(apply_patch_shuffle(img, perm))

    # ── Step 3 prep: cue conflicts (done once for all models) ───────────────────
    print("[task1] Generating AdaIN cue conflicts ...")
    pairs = [
        # Distinct Cat Texture/Pattern Pairs
        ("Abyssinian", "Bengal"),
        ("Birman", "Ragdoll"),
        ("Persian", "Siamese"),
        ("British_Shorthair", "Egyptian_Mau"),
        # Distinct Dog Fur/Shape Pairs
        ("Beagle", "Boxer"),
        ("Chihuahua", "Pug"),
        ("Samoyed", "Keeshond"),
        ("Saint_Bernard", "Newfoundland"),
    ]
    conflicts = generate_cue_conflicts(
        test_dataset, subset_indices, pairs, config["cue_conflicts"], device
    )
    print(f"  → {len(conflicts)} accepted cue conflicts")
    
    # Save the generated images to disk so the user can use them in the report
    cc_img_dir = os.path.join(config["output_dir"], "cue_conflict_images")
    report_img_dir = os.path.join(config["output_dir"], "report_images")
    os.makedirs(cc_img_dir, exist_ok=True)
    os.makedirs(report_img_dir, exist_ok=True)
    for i, c in enumerate(conflicts):
        img_name = f"{c['content_class']}_shape_{c['style_class']}_texture_{i}.jpg"
        c['stylized_pil'].save(os.path.join(cc_img_dir, img_name))
        if i < 10:
            c['stylized_pil'].save(os.path.join(report_img_dir, f"cue_conflict_{img_name}"))
            c_orig, _ = test_dataset[c['content_idx']]
            c_orig.resize((224, 224), Image.BILINEAR).save(
                os.path.join(report_img_dir, f"cue_conflict_content_{c['content_class']}_{i}.jpg")
            )
            s_orig, _ = test_dataset[c['style_idx']]
            s_orig.resize((224, 224), Image.BILINEAR).save(
                os.path.join(report_img_dir, f"cue_conflict_style_{c['style_class']}_{i}.jpg")
            )

    # ── Metrics storage ────────────────────────────────────────────────────────
    metrics: dict = {
        "step1_clean":         {},
        "step1_clip_zeroshot": {},
        "step2_color":         {},
        "step3_cue_conflicts": {},
        "step4_translation":   {},
        "step5_patch":         {},
        "step6_stability":     {},
    }
    features_for_cka = {"clean": {}, "gray": {}, "patch": {}, "translation": {}, "cue_conflict": {}}

    # ── Per-model evaluation ──────────────────────────────────────────────────
    for model_name in MODEL_NAMES:
        print(f"\n[task1] === {model_name} ===")
        model = BackboneExtractor(model_name, device)
        transform = model.transform  # model's own normalization transform

        # Linear head training
        train_transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=28),
            T.RandomHorizontalFlip(),
            transform,
        ])
        val_transform = T.Compose([T.Resize((224, 224)), T.CenterCrop(224), transform])

        train_ds = SubsetWithTransform(trainval_dataset, train_indices, train_transform)
        val_ds   = SubsetWithTransform(trainval_dataset, val_indices,   val_transform)
        train_loader = DataLoader(train_ds, batch_size=train_bs,
                      shuffle=True, num_workers=train_workers, pin_memory=pin_memory)
        val_loader   = DataLoader(val_ds,   batch_size=train_bs,
                      shuffle=False, num_workers=train_workers, pin_memory=pin_memory)

        model.train_linear_head(train_loader, val_loader, config["training"])

        # CLIP zero-shot (in addition to linear probe)
        if model_name == "CLIP":
            print("[task1]   → CLIP zero-shot ...")
            zs = CLIPZeroShot(device)
            PET_CLASSES = trainval_dataset.classes
            clean_ds_zs = PILDataset(clean_pil_images, labels_arr, transform)
            zs_loader   = DataLoader(clean_ds_zs, batch_size=eval_bs, shuffle=False, num_workers=eval_workers, pin_memory=pin_memory)
            zs_preds, zs_probs, zs_lbls = zs.predict(zs_loader, PET_CLASSES)
            from sklearn.metrics import accuracy_score, f1_score
            metrics["step1_clip_zeroshot"] = {
                "accuracy":      float(accuracy_score(zs_lbls, zs_preds)),
                "macro_f1":      float(f1_score(zs_lbls, zs_preds, average="macro",
                                                zero_division=0)),
                "mean_max_conf": float(np.mean(np.max(zs_probs, axis=1))),
            }

        # Build shared clean loader using this model's transform
        clean_ds     = PILDataset(clean_pil_images, labels_arr, transform)
        clean_loader = DataLoader(clean_ds, batch_size=eval_bs, shuffle=False, num_workers=eval_workers, pin_memory=pin_memory)

        # Step 1: clean baseline
        print("[task1]   Step 1: clean baseline ...")
        metrics["step1_clean"][model_name] = evaluate_clean(model, clean_loader, config)

        # Step 2: color bias
        print("[task1]   Step 2: color bias ...")
        metrics["step2_color"][model_name] = evaluate_color_bias(
            model, clean_pil_images, labels_arr, config
        )

        # Step 3: cue conflicts
        print("[task1]   Step 3: cue conflicts ...")
        metrics["step3_cue_conflicts"][model_name] = evaluate_cue_conflicts(
            model, conflicts, config
        )

        # Step 4: translation
        print("[task1]   Step 4: translation ...")
        metrics["step4_translation"][model_name] = evaluate_translation(
            model, clean_pil_images, labels_arr, config
        )

        # Step 5: patch shuffle
        print("[task1]   Step 5: patch shuffle ...")
        metrics["step5_patch"][model_name] = evaluate_patch_shuffle(
            model, clean_pil_images, labels_arr, config, shuffled_pil_images=patch_shuffled_images
        )

        # Step 6: cosine stability for 4 transforms
        print("[task1]   Step 6: representation stability ...")

        # Grayscale
        gray_imgs = [apply_grayscale(img) for img in clean_pil_images]
        gray_ds   = PILDataset(gray_imgs, labels_arr, transform)
        gray_loader = DataLoader(gray_ds, batch_size=eval_bs, shuffle=False, num_workers=eval_workers, pin_memory=pin_memory)
        f_clean, _ = model.extract_features(clean_loader)
        f_gray, _  = model.extract_features(gray_loader)
        features_for_cka["clean"][model_name] = f_clean
        features_for_cka["gray"][model_name] = f_gray

        from task1.analysis.feature_similarity import cosine_stability
        s_gray = cosine_stability(f_clean, f_gray)

        # Patch shuffle (reusing the pre-generated patch-shuffled images)
        patch_ds   = PILDataset(patch_shuffled_images, labels_arr, transform)
        patch_loader = DataLoader(patch_ds, batch_size=eval_bs, shuffle=False, num_workers=eval_workers, pin_memory=pin_memory)
        f_patch, _ = model.extract_features(patch_loader)
        features_for_cka["patch"][model_name] = f_patch
        s_patch = cosine_stability(f_clean, f_patch)

        # Translation δ=32 (average over 4 directions)
        s_trans_list = []
        f_trans_list = []
        for direction in ["up", "down", "left", "right"]:
            trans_imgs = [apply_translation(img, 32, direction) for img in clean_pil_images]
            trans_ds   = PILDataset(trans_imgs, labels_arr, transform)
            trans_loader = DataLoader(trans_ds, batch_size=eval_bs, shuffle=False, num_workers=eval_workers, pin_memory=pin_memory)
            f_trans, _ = model.extract_features(trans_loader)
            f_trans_list.append(f_trans)
            s_trans_list.append(cosine_stability(f_clean, f_trans))
        s_trans = float(np.mean(s_trans_list))
        # Aggregate translation features across directions (mean per-sample)
        if len(f_trans_list) > 0:
            features_for_cka["translation"][model_name] = np.mean(np.stack(f_trans_list, axis=0), axis=0)

        # Cue conflict
        content_idxs = []
        if conflicts:
            cc_imgs = [c["stylized_pil"] for c in conflicts]
            cc_labs = [0] * len(cc_imgs)  # dummy labels, not used for stability
            cc_ds   = PILDataset(cc_imgs, cc_labs, transform)
            cc_loader = DataLoader(cc_ds, batch_size=eval_bs, shuffle=False, num_workers=eval_workers, pin_memory=pin_memory)
            f_cc, _ = model.extract_features(cc_loader)
            subset_idx_map = {v: i for i, v in enumerate(subset_indices)}
            content_idxs = [subset_idx_map[c["content_idx"]]
                            for c in conflicts if c["content_idx"] in subset_idx_map]
            if content_idxs:
                f_clean_cc = f_clean[content_idxs[:len(f_cc)]]
                s_cc = cosine_stability(f_clean_cc, f_cc[:len(f_clean_cc)])
                features_for_cka["cue_conflict"][model_name] = f_cc[:len(f_clean_cc)]
            else:
                s_cc = 0.0
        else:
            s_cc = 0.0

        metrics["step6_stability"][model_name] = {
            "gray":           float(s_gray),
            "patch":          float(s_patch),
            "translation_32": float(s_trans),
            "cue_conflict":   float(s_cc),
        }

        # ── t-SNE visualizations per backbone ─────────────────────────────────
        print("[task1]   t-SNE ...")
        run_tsne(f_clean, f_gray, labels_arr, model_name, config["output_dir"], config, transform_name="Grayscale")
        run_tsne(f_clean, f_patch, labels_arr, model_name, config["output_dir"], config, transform_name="Patch Shuffle")
        if conflicts and len(content_idxs) > 0:
            cc_subset_labels = labels_arr[content_idxs[:len(f_cc)]]
            run_tsne(f_clean_cc, f_cc[:len(f_clean_cc)], cc_subset_labels, model_name, config["output_dir"], config, transform_name="Cue Conflict")

    # ── Save all metrics to JSON ──────────────────────────────────────────────
    metrics_path = os.path.join(config["output_dir"], "all_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\n[task1] Metrics saved to {metrics_path}")

    # ── Generate all plots ────────────────────────────────────────────────────
    print("[task1] Generating plots ...")

    plot_clean_baseline(metrics["step1_clean"],         config["output_dir"])
    plot_mean_max_conf(metrics["step1_clean"],          config["output_dir"])
    plot_color_bias(metrics["step2_color"],             config["output_dir"])
    plot_cue_conflicts(metrics["step3_cue_conflicts"],  config["output_dir"])
    plot_translation(metrics["step4_translation"],      config["output_dir"])
    plot_patch_shuffle(metrics["step5_patch"],          config["output_dir"])
    plot_stability(metrics["step6_stability"],          config["output_dir"])

    print("[task1] Generating CKA heatmaps ...")
    plot_cka_heatmap(features_for_cka["clean"], "CKA - Clean Baseline", os.path.join(config["output_dir"], "figures", "cka_clean.png"))
    plot_cka_heatmap(features_for_cka["gray"], "CKA - Color Bias (Grayscale)", os.path.join(config["output_dir"], "figures", "cka_gray.png"))
    plot_cka_heatmap(features_for_cka["patch"], "CKA - Patch Shuffle", os.path.join(config["output_dir"], "figures", "cka_patch.png"))
    plot_cka_heatmap(features_for_cka["translation"], "CKA - Translation (32px right)", os.path.join(config["output_dir"], "figures", "cka_translation.png"))
    if features_for_cka["cue_conflict"]:
        plot_cka_heatmap(features_for_cka["cue_conflict"], "CKA - Shape vs Texture", os.path.join(config["output_dir"], "figures", "cka_cue_conflict.png"))

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_rows = []
    for m in MODEL_NAMES:
        trans = metrics["step4_translation"][m]
        disps = sorted(int(d) for d in trans)
        summary_rows.append({
            "Model":              m,
            "Clean Acc":          metrics["step1_clean"][m]["accuracy"],
            "Clean Macro-F1":     metrics["step1_clean"][m]["macro_f1"],
            "Mean Max Conf":      metrics["step1_clean"][m]["mean_max_conf"],
            "Gray Acc":           metrics["step2_color"][m]["grayscale_acc"],
            "Gray Consistency":   metrics["step2_color"][m]["grayscale_consistency"],
            "Hue Acc":            metrics["step2_color"][m]["hue_acc"],
            "Hue Consistency":    metrics["step2_color"][m]["hue_consistency"],
            "Shape Bias %":       metrics["step3_cue_conflicts"][m]["shape_bias"],
            "CC Coverage %":      metrics["step3_cue_conflicts"][m]["coverage"],
            "Trans32 Acc":        trans[str(max(disps))]["acc"],
            "Trans32 Consistency":trans[str(max(disps))]["consistency"],
            "Patch Acc":          metrics["step5_patch"][m]["acc"],
            "Patch Consistency":  metrics["step5_patch"][m]["consistency"],
            "Stability Gray":     metrics["step6_stability"][m]["gray"],
            "Stability Patch":    metrics["step6_stability"][m]["patch"],
            "Stability Trans32":  metrics["step6_stability"][m]["translation_32"],
            "Stability CC":       metrics["step6_stability"][m]["cue_conflict"],
        })

    if "step1_clip_zeroshot" in metrics and metrics["step1_clip_zeroshot"]:
        zs = metrics["step1_clip_zeroshot"]
        summary_rows.append({
            "Model":              "CLIP (Zero-Shot)",
            "Clean Acc":          zs.get("accuracy", np.nan),
            "Clean Macro-F1":     zs.get("macro_f1", np.nan),
            "Mean Max Conf":      zs.get("mean_max_conf", np.nan),
            "Gray Acc":           np.nan,
            "Gray Consistency":   np.nan,
            "Hue Acc":            np.nan,
            "Hue Consistency":    np.nan,
            "Shape Bias %":       np.nan,
            "CC Coverage %":      np.nan,
            "Trans32 Acc":        np.nan,
            "Trans32 Consistency":np.nan,
            "Patch Acc":          np.nan,
            "Patch Consistency":  np.nan,
            "Stability Gray":     np.nan,
            "Stability Patch":    np.nan,
            "Stability Trans32":  np.nan,
            "Stability CC":       np.nan,
        })

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(config["output_dir"], "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[task1] Summary CSV saved to {csv_path}")
    print("\n" + df.to_string(index=False))
    print("\n[task1] All done! Figures in:", os.path.join(config["output_dir"], "figures"))


if __name__ == "__main__":
    main()
