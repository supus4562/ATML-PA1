import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auroc(known_scores, unknown_scores):
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_scores = np.concatenate([known_scores, unknown_scores])
    return roc_auc_score(y_true, y_scores)

def compute_fpr_at_tpr(known_scores, unknown_scores, tpr=0.95):
    # TPR = 0.95 means threshold is at 95th percentile of known_scores
    tau = np.percentile(known_scores, tpr * 100)
    # FPR is fraction of unknowns < tau (incorrectly accepted)
    fpr = np.mean(unknown_scores < tau)
    return fpr
