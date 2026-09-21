import numpy as np

def mls_score(logits):
    return -np.max(logits, axis=1)
