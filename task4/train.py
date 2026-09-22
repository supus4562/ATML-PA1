"""task4/train.py — Main training dispatch for Task 4 (Open-Set Recognition).

Usage:
  # Vanilla baseline (ResNet-18 on CIFAR-10)
  python task4/train.py --config task4/configs/vanilla.yaml --data_root ./data/cifar

  # GCSC (Strong closed-set classifier with RandAugment)
  python task4/train.py --config task4/configs/gcsc.yaml    --data_root ./data/cifar

  # PROSER (Classifier and data placeholders initialized from Vanilla)
  python task4/train.py --config task4/configs/proser.yaml  --data_root ./data/cifar
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from tqdm import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from common.seed import set_all_seeds
from task4.data.cifar10 import get_train_val_loaders
from task4.methods.vanilla import VanillaTrainer
from task4.methods.gcsc import GCSCTrainer
from task4.methods.proser import PROSERTrainer


def main():
    parser = argparse.ArgumentParser(description="Task 4 — Open-Set Recognition Training")
    parser.add_argument('--config', required=True, help="Path to YAML config")
    parser.add_argument('--data_root', required=True, help="Path to CIFAR data root")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size")
    parser.add_argument('--epochs', type=int, default=None, help="Override number of epochs")
    parser.add_argument('--lr', type=float, default=None, help="Override learning rate")
    parser.add_argument('--num_workers', type=int, default=None, help="Override num workers")
    parser.add_argument('--output_dir', default=None, help="Override output directory")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.data_root:
        config['data_root'] = args.data_root
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['n_epochs'] = args.epochs
    if args.lr is not None:
        config['lr'] = args.lr
        config['scale_lr'] = False  # Explicit override
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir

    # Linear learning rate scaling if batch size was increased above base 128
    if config.get('scale_lr', False) and args.lr is None:
        base_bs = config.get('base_batch_size', 128)
        current_bs = config.get('batch_size', 128)
        if current_bs != base_bs:
            scaling_factor = current_bs / base_bs
            config['lr'] = config['lr'] * scaling_factor
            print(f"[task4/train] Scaled LR to {config['lr']:.5f} (factor {scaling_factor:.2f} for batch_size={current_bs})")
        
    set_all_seeds(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        tqdm.write("[task4/train] Enabled TF32 + cuDNN benchmark for Ampere (A100 80GB) optimization")
        
    randaugment = config.get('randaugment_ops', 0) > 0
    num_workers = config.get('num_workers', 8)
    pin_memory = (device.type == "cuda")
    
    train_loader, val_loader = get_train_val_loaders(
        args.data_root, val_fraction=config['val_fraction'], seed=config['seed'], 
        batch_size=config['batch_size'], randaugment=randaugment, 
        ra_ops=config.get('randaugment_ops', 2), ra_mag=config.get('randaugment_mag', 9),
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    method = config['method']
    tqdm.write(f"[task4/train] Starting {method.upper()} on {device} (batch_size={config['batch_size']}, workers={num_workers})")
    
    if method == 'vanilla':
        trainer = VanillaTrainer(config, device)
    elif method == 'gcsc':
        trainer = GCSCTrainer(config, device)
    elif method == 'proser':
        trainer = PROSERTrainer(config, device)
    else:
        raise ValueError(f"Unknown method '{method}'")
        
    history = trainer.train(train_loader, val_loader)
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    
    # 1. Save JSON curves
    json_path = f"{output_dir}/{method}_training_curves.json"
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    tqdm.write(f"[task4/train] Saved JSON training history to {json_path}")
    
    # 2. Save CSV curves (user requirement: show all data, evaluation losses in .csv)
    df_curves = pd.DataFrame(history)
    csv_path = f"{output_dir}/{method}_training_curves.csv"
    df_curves.to_csv(csv_path, index=False)
    tqdm.write(f"[task4/train] Saved CSV training history to {csv_path}")
        
    # 3. Plot multi-panel curves (Train Loss vs Val Loss, Val Accuracy)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    epochs = history['epoch'] if 'epoch' in history else list(range(1, len(history['train_loss']) + 1))
    
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='tab:blue', lw=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], label='Val Loss (Eval)', color='tab:orange', lw=2, linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title(f'{method.upper()} Loss Curves')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(epochs, [v * 100 for v in history['val_acc']], label='Val Acc (%)', color='tab:green', lw=2)
    if 'train_acc' in history:
        axes[1].plot(epochs, [t * 100 for t in history['train_acc']], label='Train Acc (%)', color='tab:purple', lw=2, linestyle=':')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{method.upper()} Accuracy Curves')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    fig_path = f"{output_dir}/figures/{method}_curves.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    tqdm.write(f"[task4/train] Saved training curves plot to {fig_path}")


if __name__ == '__main__':
    main()
