"""task4/extract_outputs.py — Feature and Logit Extraction for Task 4.

Usage:
  python task4/extract_outputs.py --data_root ./data/cifar
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from task4.models.resnet_cifar import ResNetCIFAR
from task4.data.cifar10 import get_train_val_loaders, get_test_loader, get_train_loader_unaugmented
from task4.data.cifar100_unknowns import CIFAR100Unknowns
from task4.data.make_splits import make_cifar10_splits


def extract_features(model, loader, device, desc="Extracting"):
    model.eval()
    all_logits = []
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            features = model.forward_features(x)
            logits = model.fc(features)
            
            all_logits.append(logits.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())
            
    return np.concatenate(all_logits), np.concatenate(all_features), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Task 4 — Extract features and logits")
    parser.add_argument('--data_root', required=True, help="Path to CIFAR data")
    parser.add_argument('--batch_size', type=int, default=1024, help="Inference batch size (optimized for 80GB VRAM)")
    parser.add_argument('--num_workers', type=int, default=8, help="Dataloader workers")
    parser.add_argument('--vanilla_ckpt', default='task4/results/vanilla_checkpoint.pth')
    parser.add_argument('--gcsc_ckpt', default='task4/results/gcsc_checkpoint.pth')
    parser.add_argument('--proser_ckpt', default='task4/results/proser_checkpoint.pth')
    parser.add_argument('--cache_dir', default='task4/cache')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        tqdm.write("[task4/extract] Enabled TF32 + cuDNN benchmark for Ampere (A100 80GB)")
        
    os.makedirs(args.cache_dir, exist_ok=True)
    pin_memory = (device.type == "cuda")
    
    tqdm.write(f"[task4/extract] Building data loaders with batch_size={args.batch_size}, num_workers={args.num_workers}...")
    train_loader, val_loader = get_train_val_loaders(
        args.data_root, randaugment=False, batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = get_test_loader(
        args.data_root, batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    
    unknowns = CIFAR100Unknowns(args.data_root)
    near_loader, _ = unknowns.get_near_loader(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory
    )
    far_loader, _ = unknowns.get_far_loader(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory
    )
    
    splits = make_cifar10_splits(args.data_root)
    unaug_train_loader = get_train_loader_unaugmented(
        args.data_root, splits['train'], batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    
    datasets = {
        'cifar10_train': train_loader,
        'cifar10_val': val_loader,
        'cifar10_test': test_loader,
        'near': near_loader,
        'far': far_loader
    }

    # 1. Vanilla model feature & logit extraction
    if os.path.exists(args.vanilla_ckpt):
        tqdm.write(f"[task4/extract] Extracting Vanilla outputs from {args.vanilla_ckpt}...")
        model = ResNetCIFAR(num_classes=10).to(device)
        model.load_state_dict(torch.load(args.vanilla_ckpt, map_location=device))
        
        for name, loader in datasets.items():
            logits, features, labels = extract_features(model, loader, device, desc=f"Vanilla {name}")
            np.save(f"{args.cache_dir}/{name}_logits.npy", logits)
            np.save(f"{args.cache_dir}/{name}_features.npy", features)
            np.save(f"{args.cache_dir}/{name}_labels.npy", labels)
            
        _, features, labels = extract_features(model, unaug_train_loader, device, desc="Vanilla unaug_train")
        np.save(f"{args.cache_dir}/cifar10_train_unaugmented_features.npy", features)
        np.save(f"{args.cache_dir}/cifar10_train_unaugmented_labels.npy", labels)
    else:
        tqdm.write(f"[task4/extract] WARNING: Vanilla checkpoint not found at {args.vanilla_ckpt}")

    # 2. GCSC model logit extraction
    if os.path.exists(args.gcsc_ckpt):
        tqdm.write(f"[task4/extract] Extracting GCSC outputs from {args.gcsc_ckpt}...")
        model = ResNetCIFAR(num_classes=10).to(device)
        model.load_state_dict(torch.load(args.gcsc_ckpt, map_location=device))
        for name, loader in datasets.items():
            logits, features, _ = extract_features(model, loader, device, desc=f"GCSC {name}")
            np.save(f"{args.cache_dir}/gcsc_{name}_logits.npy", logits)
            np.save(f"{args.cache_dir}/gcsc_{name}_features.npy", features)
    else:
        tqdm.write(f"[task4/extract] Note: GCSC checkpoint not found at {args.gcsc_ckpt}")
            
    # 3. PROSER model logit extraction (15 classes: 10 known + 5 dummy)
    if os.path.exists(args.proser_ckpt):
        tqdm.write(f"[task4/extract] Extracting PROSER outputs from {args.proser_ckpt}...")
        model = ResNetCIFAR(num_classes=10).to(device)
        model.fc = torch.nn.Linear(512, 15).to(device)
        model.load_state_dict(torch.load(args.proser_ckpt, map_location=device))
        for name, loader in datasets.items():
            logits, features, _ = extract_features(model, loader, device, desc=f"PROSER {name}")
            np.save(f"{args.cache_dir}/proser_{name}_logits.npy", logits)
            np.save(f"{args.cache_dir}/proser_{name}_features.npy", features)
    else:
        tqdm.write(f"[task4/extract] Note: PROSER checkpoint not found at {args.proser_ckpt}")

    tqdm.write(f"[task4/extract] Feature extraction complete. Cache saved to: {args.cache_dir}")


if __name__ == '__main__':
    main()
