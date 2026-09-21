import numpy as np
from scipy.special import softmax

def msp_score(logits):
    probs = softmax(logits, axis=1)
    return 1 - np.max(probs, axis=1)
