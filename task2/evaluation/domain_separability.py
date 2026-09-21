import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def compute_domain_separability(features_source, features_target, seed=6304):
    min_len = min(len(features_source), len(features_target))
    X_s = features_source[:min_len]
    X_t = features_target[:min_len]
    X = np.concatenate([X_s, X_t], axis=0)
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)], axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    
    clf = LogisticRegression(C=1.0, random_state=seed, max_iter=1000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return acc
