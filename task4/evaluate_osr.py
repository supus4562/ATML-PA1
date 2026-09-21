import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from task4.scores.msp import msp_score
from task4.scores.mls import mls_score
from task4.scores.energy import energy_score
from task4.scores.mahalanobis import MahalanobisScore
from task4.evaluation.metrics import compute_auroc, compute_fpr_at_tpr
from task4.evaluation.thresholds import calibrate_threshold
from task4.evaluation.failure_analysis import find_failures
from task4.data.cifar100_unknowns import CIFAR100Unknowns
import argparse

def evaluate_method(known_val_scores, known_test_scores, near_scores, far_scores):
    tau = calibrate_threshold(known_val_scores, 95)
    near_auroc = compute_auroc(known_test_scores, near_scores)
    far_auroc = compute_auroc(known_test_scores, far_scores)
    all_auroc = compute_auroc(known_test_scores, np.concatenate([near_scores, far_scores]))
    near_fpr = np.mean(near_scores < tau)
    far_fpr = np.mean(far_scores < tau)
    
    return {
        'Near AUROC': near_auroc, 'Far AUROC': far_auroc, 'All AUROC': all_auroc,
        'Near FPR@95': near_fpr, 'Far FPR@95': far_fpr, 'Threshold': tau
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    c_dir = 'task4/cache'
    
    val_logits = np.load(f"{c_dir}/cifar10_val_logits.npy")
    test_logits = np.load(f"{c_dir}/cifar10_test_logits.npy")
    near_logits = np.load(f"{c_dir}/near_logits.npy")
    far_logits = np.load(f"{c_dir}/far_logits.npy")
    
    train_f_unaug = np.load(f"{c_dir}/cifar10_train_unaugmented_features.npy")
    train_l_unaug = np.load(f"{c_dir}/cifar10_train_unaugmented_labels.npy")
    val_f = np.load(f"{c_dir}/cifar10_val_features.npy")
    test_f = np.load(f"{c_dir}/cifar10_test_features.npy")
    near_f = np.load(f"{c_dir}/near_features.npy")
    far_f = np.load(f"{c_dir}/far_features.npy")
    
    mah = MahalanobisScore()
    mah.fit(train_f_unaug, train_l_unaug)
    
    scores_dict = {
        'MSP': (msp_score(val_logits), msp_score(test_logits), msp_score(near_logits), msp_score(far_logits)),
        'MLS': (mls_score(val_logits), mls_score(test_logits), mls_score(near_logits), mls_score(far_logits)),
        'Energy': (energy_score(val_logits), energy_score(test_logits), energy_score(near_logits), energy_score(far_logits)),
        'Mahalanobis': (mah.score(val_f), mah.score(test_f), mah.score(near_f), mah.score(far_f))
    }
    
    table1 = []
    for method_name, (v_scores, t_scores, n_scores, f_scores) in scores_dict.items():
        res = evaluate_method(v_scores, t_scores, n_scores, f_scores)
        res['Method'] = method_name
        table1.append(res)
        
    os.makedirs(args.output_dir, exist_ok=True)
    df1 = pd.DataFrame(table1)[['Method', 'Near AUROC', 'Far AUROC', 'All AUROC', 'Near FPR@95', 'Far FPR@95']]
    df1.to_csv(f"{args.output_dir}/table1.csv", index=False)
    
    table2 = []
    methods = ['vanilla', 'gcsc', 'proser']
    
    for m in methods:
        try:
            if m == 'vanilla':
                t_log, v_log, n_log, f_log = test_logits, val_logits, near_logits, far_logits
            else:
                t_log = np.load(f"{c_dir}/{m}_cifar10_test_logits.npy")
                v_log = np.load(f"{c_dir}/{m}_cifar10_val_logits.npy")
                n_log = np.load(f"{c_dir}/{m}_near_logits.npy")
                f_log = np.load(f"{c_dir}/{m}_far_logits.npy")
                
            test_labels = np.load(f"{c_dir}/cifar10_test_labels.npy")
            preds = t_log[:, :10].argmax(axis=1)
            csa = np.mean(preds == test_labels)
            
            res = evaluate_method(mls_score(v_log[:, :10]), mls_score(t_log[:, :10]), 
                                  mls_score(n_log[:, :10]), mls_score(f_log[:, :10]))
            res['Method'] = m.upper()
            res['Score'] = 'MLS'
            res['CSA'] = csa
            table2.append(res)
            
            if m == 'proser':
                def proser_score(l): return np.max(l[:, 10:], axis=1) - np.max(l[:, :10], axis=1)
                res_p = evaluate_method(proser_score(v_log), proser_score(t_log), 
                                        proser_score(n_log), proser_score(f_log))
                res_p['Method'] = 'PROSER'
                res_p['Score'] = 'PROSER'
                res_p['CSA'] = csa
                table2.append(res_p)
        except FileNotFoundError:
            pass

    if table2:
        df2 = pd.DataFrame(table2)[['Method', 'Score', 'CSA', 'Near AUROC', 'Far AUROC', 'Near FPR@95', 'Far FPR@95']]
        df2.to_csv(f"{args.output_dir}/table2.csv", index=False)

    os.makedirs(f"{args.output_dir}/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_scores = ['MSP', 'MLS', 'Mahalanobis']
    for i, p in enumerate(plot_scores):
        v, t, n, f = scores_dict[p]
        sns.kdeplot(t, fill=True, label='Known', ax=axes[i])
        sns.kdeplot(n, fill=True, label='Near', ax=axes[i])
        sns.kdeplot(f, fill=True, label='Far', ax=axes[i])
        axes[i].set_title(f'{p} Distribution')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/figures/score_distributions.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, p in enumerate(plot_scores):
        v, t, n, f = scores_dict[p]
        for name, scores in [('Near', n), ('Far', f)]:
            y_true = np.concatenate([np.zeros(len(t)), np.ones(len(scores))])
            y_scores = np.concatenate([t, scores])
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            axes[i].plot(fpr, tpr, label=name)
        axes[i].set_title(f'{p} ROC')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/figures/roc_curves.png")

    unknowns = CIFAR100Unknowns(args.data_root)
    _, near_names = unknowns.get_near_loader()
    _, far_names = unknowns.get_far_loader()
    
    tau = calibrate_threshold(mls_score(val_logits), 95)
    near_preds = near_logits[:, :10].argmax(axis=1)
    far_preds = far_logits[:, :10].argmax(axis=1)
    
    near_failures = find_failures(mls_score(near_logits), None, near_names, near_preds, tau, 'near')
    far_failures = find_failures(mls_score(far_logits), None, far_names, far_preds, tau, 'far')
    
    with open(f"{args.output_dir}/failure_analysis.json", "w") as f:
        json.dump({'near_failures': near_failures[:10], 'far_failures': far_failures[:10]}, f, indent=4)

if __name__ == '__main__':
    main()
