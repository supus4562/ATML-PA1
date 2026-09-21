import os
import json
import argparse
import numpy as np

def make_balanced_subset(dataset, n_per_class=50, seed=6304):
    np.random.seed(seed)
    try:
        targets = np.array(dataset.targets)
    except AttributeError:
        try:
            targets = np.array(dataset.labels)
        except AttributeError:
            targets = np.array(dataset._labels)
            
    indices = []
    num_classes = len(dataset.classes) if hasattr(dataset, 'classes') else len(np.unique(targets))
    for c in range(num_classes):
        c_indices = np.where(targets == c)[0]
        chosen = np.random.choice(c_indices, n_per_class, replace=False)
        indices.extend(chosen.tolist())
    
    os.makedirs('task1/data', exist_ok=True)
    with open('task1/data/subset_indices.json', 'w') as f:
        json.dump(indices, f)
    return indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data/oxford-iiit-pet')
    parser.add_argument('--n_per_class', type=int, default=50)
    args = parser.parse_args()
    
    from torchvision.datasets import OxfordIIITPet
    dataset = OxfordIIITPet(root=args.data_root, split='test', download=True)
    make_balanced_subset(dataset, args.n_per_class)
