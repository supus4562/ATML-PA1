import argparse
import yaml
import torch
import json
import os
from common.seed import set_all_seeds
from task4.data.cifar10 import get_train_val_loaders
from task4.methods.vanilla import VanillaTrainer
from task4.methods.gcsc import GCSCTrainer
from task4.methods.proser import PROSERTrainer
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_root', required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    set_all_seeds(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    randaugment = config.get('randaugment_ops', 0) > 0
    train_loader, val_loader = get_train_val_loaders(
        args.data_root, val_fraction=config['val_fraction'], seed=config['seed'], 
        batch_size=config['batch_size'], randaugment=randaugment, 
        ra_ops=config.get('randaugment_ops', 2), ra_mag=config.get('randaugment_mag', 9)
    )
    
    method = config['method']
    if method == 'vanilla':
        trainer = VanillaTrainer(config, device)
    elif method == 'gcsc':
        trainer = GCSCTrainer(config, device)
    elif method == 'proser':
        trainer = PROSERTrainer(config, device)
        
    history = trainer.train(train_loader, val_loader)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(f"{config['output_dir']}/{method}_curves.json", "w") as f:
        json.dump(history, f)
        
    # Plot curves
    os.makedirs(f"{config['output_dir']}/figures", exist_ok=True)
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.savefig(f"{config['output_dir']}/figures/{method}_curves.png")
    plt.close()

if __name__ == '__main__':
    main()
