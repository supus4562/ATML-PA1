import os
import json
import argparse
import numpy as np

def make_balanced_subset(dataset, total_samples=500, n_per_class=None, seed=6304):
    rng = np.random.RandomState(seed)
    try:
        targets = np.array(dataset.targets)
    except AttributeError:
        try:
            targets = np.array(dataset.labels)
        except AttributeError:
            targets = np.array(dataset._labels)
            
    indices = []
    num_classes = len(dataset.classes) if hasattr(dataset, 'classes') else len(np.unique(targets))
    
    # If total_samples is specified (assignment specifies 500 official test images):
    if total_samples is not None:
        base_n = total_samples // num_classes
        rem = total_samples % num_classes
        per_class_counts = [base_n + (1 if i < rem else 0) for i in range(num_classes)]
    else:
        per_class_counts = [n_per_class or 50] * num_classes

    for c in range(num_classes):
        c_indices = np.where(targets == c)[0]
        if len(c_indices) == 0:
            continue
        req_count = per_class_counts[c]
        replace = len(c_indices) < req_count
        chosen = rng.choice(c_indices, req_count, replace=replace)
        indices.extend(chosen.tolist())
    
    os.makedirs('task1/data', exist_ok=True)
    with open('task1/data/subset_indices.json', 'w') as f:
        json.dump(indices, f)
    return indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data/oxford-iiit-pet')
    parser.add_argument('--total_samples', type=int, default=500)
    args = parser.parse_args()
    
    from torchvision.datasets import OxfordIIITPet
    dataset = OxfordIIITPet(root=args.data_root, split='test', download=True)
    make_balanced_subset(dataset, total_samples=args.total_samples)
