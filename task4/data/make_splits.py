import os
import json
import numpy as np
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from common.seed import set_all_seeds

# Use high-speed verified CDN mirror to bypass slow Toronto server throttling
datasets.CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

def make_cifar10_splits(data_root, val_fraction=0.1, seed=6304):
    set_all_seeds(seed)
    cache_path = 'task4/cache/cifar10_split_indices.json'
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True)
    labels = np.array(dataset.targets)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    splits = {
        'train': train_idx.tolist(),
        'val': val_idx.tolist()
    }
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(splits, f)
        
    return splits
