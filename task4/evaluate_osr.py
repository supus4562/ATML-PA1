"""task4/evaluate_osr.py — Evaluation dispatch for Task 4 (Open-Set Recognition).

Computes all required metrics, AUROCs, validation-calibrated rejection rates,
evaluation losses, per-class diagnostics, and exports results to CSV and JSON.

Usage:
  python task4/evaluate_osr.py --data_root ./data/cifar --output_dir task4/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from task4.scores.msp import msp_score
from task4.scores.mls import mls_score
from task4.scores.energy import energy_score
from task4.scores.mahalanobis import MahalanobisScore
from task4.evaluation.metrics import compute_auroc
from task4.evaluation.thresholds import calibrate_threshold
from task4.evaluation.failure_analysis import find_failures, compute_per_class_rejection
from task4.data.cifar100_unknowns import CIFAR100Unknowns, NEAR_CLASSES, FAR_CLASSES


def compute_eval_loss(logits_10: np.ndarray, labels: np.ndarray) -> float:
    """Compute cross-entropy evaluation loss on 10 known classes."""
    exp_z = np.exp(logits_10 - np.max(logits_10, axis=1, keepdims=True))
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    n = len(labels)
    loss = -np.mean(np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1.0)))
    return float(loss)


def evaluate_method(known_val_scores: np.ndarray, 
                    known_test_scores: np.ndarray, 
                    near_scores: np.ndarray, 
                    far_scores: np.ndarray) -> dict:
    """Evaluate open-set recognition metrics following ATML PA-1 PDF protocol.
    
    Threshold tau is calibrated as 95th percentile of unknownness on known validation set:
    accept x when u(x) <= tau.
    """
    tau = float(calibrate_threshold(known_val_scores, 95))
    
    # 1. AUROC across all thresholds
    all_unknown_scores = np.concatenate([near_scores, far_scores])
    near_auroc = float(compute_auroc(known_test_scores, near_scores))
    far_auroc = float(compute_auroc(known_test_scores, far_scores))
    all_auroc = float(compute_auroc(known_test_scores, all_unknown_scores))
    
    # 2. FPR@95TPR: fraction of unknowns incorrectly accepted (u(x) <= tau)
    near_fpr = float(np.mean(near_scores <= tau))
    far_fpr = float(np.mean(far_scores <= tau))
    all_fpr = float(np.mean(all_unknown_scores <= tau))
    
    # 3. Rejection rates: 1 - FPR
    near_rej = float(1.0 - near_fpr)
    far_rej = float(1.0 - far_fpr)
    all_rej = float(1.0 - all_fpr)
    
    # 4. Achieved CIFAR-10 test acceptance rate
    test_acc_rate = float(np.mean(known_test_scores <= tau))
    
    return {
        'Near AUROC': round(near_auroc, 4),
        'Far AUROC': round(far_auroc, 4),
        'All AUROC': round(all_auroc, 4),
        'Near FPR@95': round(near_fpr, 4),
        'Far FPR@95': round(far_fpr, 4),
        'All FPR@95': round(all_fpr, 4),
        'Near Rejection Rate': round(near_rej, 4),
        'Far Rejection Rate': round(far_rej, 4),
        'All Rejection Rate': round(all_rej, 4),
        'CIFAR10 Test Acceptance Rate': round(test_acc_rate, 4),
        'Threshold': round(tau, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="Task 4 — OSR Evaluation and Reporting")
    parser.add_argument('--data_root', required=True, help="Path to CIFAR data root")
    parser.add_argument('--output_dir', default='task4/results', help="Results output directory")
    parser.add_argument('--cache_dir', default='task4/cache', help="Cached features directory")
    args = parser.parse_args()
    
    c_dir = args.cache_dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/figures", exist_ok=True)
    
    print(f"[task4/evaluate] Loading cached representations from {c_dir}...")
    val_logits = np.load(f"{c_dir}/cifar10_val_logits.npy")
    test_logits = np.load(f"{c_dir}/cifar10_test_logits.npy")
    near_logits = np.load(f"{c_dir}/near_logits.npy")
    far_logits = np.load(f"{c_dir}/far_logits.npy")
    test_labels = np.load(f"{c_dir}/cifar10_test_labels.npy")
    
    train_f_unaug = np.load(f"{c_dir}/cifar10_train_unaugmented_features.npy")
    train_l_unaug = np.load(f"{c_dir}/cifar10_train_unaugmented_labels.npy")
    val_f = np.load(f"{c_dir}/cifar10_val_features.npy")
    test_f = np.load(f"{c_dir}/cifar10_test_features.npy")
    near_f = np.load(f"{c_dir}/near_features.npy")
    far_f = np.load(f"{c_dir}/far_features.npy")
    
    # Fit Mahalanobis on unaugmented training features
    print("[task4/evaluate] Fitting Mahalanobis score on unaugmented training features...")
    mah = MahalanobisScore()
    mah.fit(train_f_unaug, train_l_unaug)
    
    scores_dict = {
        'MSP': (msp_score(val_logits), msp_score(test_logits), msp_score(near_logits), msp_score(far_logits)),
        'MLS': (mls_score(val_logits), mls_score(test_logits), mls_score(near_logits), mls_score(far_logits)),
        'Energy': (energy_score(val_logits), energy_score(test_logits), energy_score(near_logits), energy_score(far_logits)),
        'Mahalanobis': (mah.score(val_f), mah.score(test_f), mah.score(near_f), mah.score(far_f))
    }
    
    # ── Table 1: Post-hoc Novelty Scores on Vanilla ───────────────────────────
    print("\n" + "=" * 70)
    print("Table 1: Post-hoc Novelty Scores on Frozen Vanilla ResNet-18")
    print("=" * 70)
    
    table1 = []
    vanilla_eval_loss = compute_eval_loss(test_logits[:, :10], test_labels)
    vanilla_csa = float(np.mean(test_logits[:, :10].argmax(axis=1) == test_labels))
    
    for method_name, (v_scores, t_scores, n_scores, f_scores) in scores_dict.items():
        res = evaluate_method(v_scores, t_scores, n_scores, f_scores)
        res['Method'] = 'Vanilla'
        res['Score'] = method_name
        res['Eval Loss'] = round(vanilla_eval_loss, 4)
        table1.append(res)
        
    df1 = pd.DataFrame(table1)[[
        'Method', 'Score', 'Eval Loss',
        'Near AUROC', 'Far AUROC', 'All AUROC',
        'Near FPR@95', 'Far FPR@95', 'All FPR@95',
        'Near Rejection Rate', 'Far Rejection Rate', 'All Rejection Rate',
        'CIFAR10 Test Acceptance Rate', 'Threshold'
    ]]
    print(df1.to_string(index=False))
    
    # Export Table 1 CSVs
    t1_path = f"{args.output_dir}/table1_posthoc_scores.csv"
    df1.to_csv(t1_path, index=False)
    df1.to_csv(f"{args.output_dir}/table1.csv", index=False)
    print(f"[task4/evaluate] Saved Table 1 to {t1_path} and {args.output_dir}/table1.csv")

    # ── Table 2: Model Comparison (Vanilla vs GCSC vs PROSER) ─────────────────
    print("\n" + "=" * 70)
    print("Table 2: Comparison of Vanilla, GCSC, and PROSER")
    print("=" * 70)
    
    table2 = []
    models = ['vanilla', 'gcsc', 'proser']
    
    for m in models:
        try:
            if m == 'vanilla':
                t_log, v_log, n_log, f_log = test_logits, val_logits, near_logits, far_logits
            else:
                t_log = np.load(f"{c_dir}/{m}_cifar10_test_logits.npy")
                v_log = np.load(f"{c_dir}/{m}_cifar10_val_logits.npy")
                n_log = np.load(f"{c_dir}/{m}_near_logits.npy")
                f_log = np.load(f"{c_dir}/{m}_far_logits.npy")
                
            # Closed-Set Accuracy on 10 known classes
            preds = t_log[:, :10].argmax(axis=1)
            csa = float(np.mean(preds == test_labels))
            m_eval_loss = compute_eval_loss(t_log[:, :10], test_labels)
            
            # Common score: MLS on the 10 known-class logits
            res = evaluate_method(
                mls_score(v_log[:, :10]), mls_score(t_log[:, :10]), 
                mls_score(n_log[:, :10]), mls_score(f_log[:, :10])
            )
            res['Method'] = m.upper()
            res['Score'] = 'MLS'
            res['CSA'] = round(csa, 4)
            res['Eval Loss'] = round(m_eval_loss, 4)
            table2.append(res)
            
            # PROSER placeholder detection score: max(dummy) - max(known)
            if m == 'proser':
                def proser_score(l):
                    return np.max(l[:, 10:], axis=1) - np.max(l[:, :10], axis=1)
                    
                res_p = evaluate_method(
                    proser_score(v_log), proser_score(t_log), 
                    proser_score(n_log), proser_score(f_log)
                )
                res_p['Method'] = 'PROSER'
                res_p['Score'] = 'Placeholder-Score'
                res_p['CSA'] = round(csa, 4)
                res_p['Eval Loss'] = round(m_eval_loss, 4)
                table2.append(res_p)
                
        except FileNotFoundError as e:
            print(f"[task4/evaluate] Note: Checkpoint outputs for '{m}' not found in cache ({e}). Skipping.")

    if table2:
        df2 = pd.DataFrame(table2)[[
            'Method', 'Score', 'CSA', 'Eval Loss',
            'Near AUROC', 'Far AUROC', 'All AUROC',
            'Near FPR@95', 'Far FPR@95', 'All FPR@95',
            'Near Rejection Rate', 'Far Rejection Rate', 'All Rejection Rate',
            'CIFAR10 Test Acceptance Rate', 'Threshold'
        ]]
        print(df2.to_string(index=False))
        
        # Export Table 2 CSVs
        t2_path = f"{args.output_dir}/table2_methods_comparison.csv"
        df2.to_csv(t2_path, index=False)
        df2.to_csv(f"{args.output_dir}/table2.csv", index=False)
        print(f"[task4/evaluate] Saved Table 2 to {t2_path} and {args.output_dir}/table2.csv")

    # ── Final Results Consolidated JSON & CSV ─────────────────────────────────
    final_results = {
        'table1_posthoc': table1,
        'table2_methods': table2,
        'vanilla_csa': vanilla_csa,
        'vanilla_eval_loss': vanilla_eval_loss
    }
    with open(f"{args.output_dir}/final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
        
    all_summary_rows = table1 + [r for r in table2 if r['Method'] != 'VANILLA']
    df_final = pd.DataFrame(all_summary_rows)
    df_final.to_csv(f"{args.output_dir}/final_results.csv", index=False)
    print(f"[task4/evaluate] Consolidated final_results.csv and final_results.json saved.")

    # ── Score Distributions & ROC Curves Figures ─────────────────────────────
    print("\n[task4/evaluate] Generating publication figures...")
    plot_scores = ['MSP', 'MLS', 'Energy', 'Mahalanobis']
    
    # 1. Score Distributions Figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for i, p in enumerate(plot_scores):
        v, t, n, f = scores_dict[p]
        for data, label, col in [
            (t, 'Known (CIFAR-10 Test)', 'tab:blue'),
            (n, 'Near Unknowns', 'tab:orange'),
            (f, 'Far Unknowns', 'tab:green')
        ]:
            if len(data) > 1 and np.var(data) > 1e-12:
                kde = gaussian_kde(data)
                xs = np.linspace(data.min(), data.max(), 200)
                ys = kde(xs)
                axes[i].plot(xs, ys, label=label, color=col, lw=2)
                axes[i].fill_between(xs, ys, color=col, alpha=0.3)
            else:
                axes[i].hist(data, bins=30, density=True, alpha=0.4, label=label, color=col)
                
        axes[i].set_title(f'{p} Unknownness Score', fontweight='bold')
        axes[i].set_xlabel('Score u(x)')
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='upper left', fontsize=8)
    fig_dist = f"{args.output_dir}/figures/score_distributions.png"
    plt.savefig(fig_dist, dpi=200)
    plt.close()
    print(f"[task4/evaluate] Saved score distributions plot to {fig_dist}")

    # 2. ROC Curves Figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for i, p in enumerate(plot_scores):
        v, t, n, f = scores_dict[p]
        for name, scores, col in [('Near Unknowns', n, 'tab:orange'), ('Far Unknowns', f, 'tab:green')]:
            y_true = np.concatenate([np.zeros(len(t)), np.ones(len(scores))])
            y_scores = np.concatenate([t, scores])
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            axes[i].plot(fpr, tpr, label=name, color=col, lw=2)
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        axes[i].set_title(f'{p} ROC Curve', fontweight='bold')
        axes[i].set_xlabel('False Positive Rate (FPR)')
        axes[i].set_ylabel('True Positive Rate (TPR)')
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='lower right', fontsize=8)
    fig_roc = f"{args.output_dir}/figures/roc_curves.png"
    plt.savefig(fig_roc, dpi=200)
    plt.close()
    print(f"[task4/evaluate] Saved ROC curves plot to {fig_roc}")

    # 3. Model Comparison Bar Chart (if table2 populated)
    if len(table2) >= 2:
        df_comp = pd.DataFrame(table2)
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        x = np.arange(len(df_comp))
        width = 0.25
        labels = [f"{row['Method']}\n({row['Score']})" for _, row in df_comp.iterrows()]
        
        ax.bar(x - width, df_comp['CSA'] * 100, width, label='Closed-Set Acc (%)', color='tab:blue')
        ax.bar(x, df_comp['Near AUROC'] * 100, width, label='Near AUROC (%)', color='tab:orange')
        ax.bar(x + width, df_comp['Far AUROC'] * 100, width, label='Far AUROC (%)', color='tab:green')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight='bold')
        ax.set_ylabel('Score / Accuracy (%)')
        ax.set_title('Task 4: Model Comparison across CSA, Near AUROC, and Far AUROC', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, axis='y', alpha=0.3)
        fig_bar = f"{args.output_dir}/figures/osr_comparison_bars.png"
        plt.savefig(fig_bar, dpi=200)
        plt.close()
        print(f"[task4/evaluate] Saved model comparison bar chart to {fig_bar}")

    # ── Per-Class Unknown Rejection Analysis ───────────────────────────────────
    unknowns = CIFAR100Unknowns(args.data_root)
    _, near_names = unknowns.get_near_loader(batch_size=1024)
    _, far_names = unknowns.get_far_loader(batch_size=1024)
    
    # Validation threshold for Vanilla MLS
    vanilla_mls_tau = float(calibrate_threshold(mls_score(val_logits), 95))
    near_preds = near_logits[:, :10].argmax(axis=1)
    far_preds = far_logits[:, :10].argmax(axis=1)
    
    near_class_stats = compute_per_class_rejection(
        mls_score(near_logits), near_names, near_preds, vanilla_mls_tau, split='near'
    )
    far_class_stats = compute_per_class_rejection(
        mls_score(far_logits), far_names, far_preds, vanilla_mls_tau, split='far'
    )
    all_class_stats = near_class_stats + far_class_stats
    df_class_stats = pd.DataFrame(all_class_stats)
    
    class_stats_csv = f"{args.output_dir}/per_class_unknown_rejection.csv"
    df_class_stats.to_csv(class_stats_csv, index=False)
    with open(f"{args.output_dir}/per_class_unknown_rejection.json", "w") as f:
        json.dump(all_class_stats, f, indent=2)
    print(f"[task4/evaluate] Saved per-class unknown rejection stats to {class_stats_csv}")

    # Plot per-class rejection bar chart
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    df_class_stats_sorted = df_class_stats.sort_values(by=['split', 'rejection_rate'], ascending=[False, True])
    colors = ['tab:orange' if s == 'near' else 'tab:green' for s in df_class_stats_sorted['split']]
    bars = ax.barh(df_class_stats_sorted['unknown_class'], df_class_stats_sorted['rejection_rate'], color=colors)
    ax.set_xlabel('Rejection Rate (%) at 95% Known Validation Acceptance')
    ax.set_title('Per-Class Unknown Rejection Rates (Vanilla MLS Threshold)', fontweight='bold')
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    
    # Custom legend for splits
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:orange', label='Near Unknowns'),
        Patch(facecolor='tab:green', label='Far Unknowns')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    fig_cls = f"{args.output_dir}/figures/per_class_rejection_bars.png"
    plt.savefig(fig_cls, dpi=200)
    plt.close()
    print(f"[task4/evaluate] Saved per-class rejection plot to {fig_cls}")

    # ── Failure Analysis (Vanilla MLS Threshold) ──────────────────────────────
    print("\n[task4/evaluate] Extracting failure modes under Vanilla MLS threshold...")
    near_failures = find_failures(mls_score(near_logits), None, near_names, near_preds, vanilla_mls_tau, 'near')
    far_failures = find_failures(mls_score(far_logits), None, far_names, far_preds, vanilla_mls_tau, 'far')
    all_failures = near_failures + far_failures
    
    # Export failures to CSV and JSON
    df_failures = pd.DataFrame(all_failures)
    df_failures.to_csv(f"{args.output_dir}/failure_analysis.csv", index=False)
    
    with open(f"{args.output_dir}/failure_analysis.json", "w") as f:
        json.dump({
            'threshold': vanilla_mls_tau,
            'near_failures_count': len(near_failures),
            'far_failures_count': len(far_failures),
            'top_near_failures': near_failures[:15],
            'top_far_failures': far_failures[:15]
        }, f, indent=4)
        
    print(f"[task4/evaluate] Total Near Failures (erroneously accepted): {len(near_failures)} / {len(near_names)}")
    print(f"[task4/evaluate] Total Far Failures (erroneously accepted): {len(far_failures)} / {len(far_names)}")
    print(f"[task4/evaluate] Saved failure_analysis.csv and failure_analysis.json")
    print(f"\n[task4/evaluate] All Task 4 evaluations and exports successfully completed!")


if __name__ == '__main__':
    main()
