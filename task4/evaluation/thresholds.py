import numpy as np

def calibrate_threshold(val_scores, percentile=95):
    return np.percentile(val_scores, percentile)
