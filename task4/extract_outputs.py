import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from task4.models.resnet_cifar import ResNetCIFAR
from task4.data.cifar10 import get_train_val_loaders, get_test_loader, get_train_loader_unaugmented
from task4.data.cifar100_unknowns import CIFAR100Unknowns
from task4.data.make_splits import make_cifar10_splits

def extract_features(model, loader, device, is_proser=False):
    model.eval()
    all_logits = []
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting"):
            x = x.to(device)
            features = model.forward_features(x)
            logits = model.fc(features)
            
            all_logits.append(logits.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())
            
    return np.concatenate(all_logits), np.concatenate(all_features), np.concatenate(all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--vanilla_ckpt', default='task4/results/vanilla_checkpoint.pth')
    parser.add_argument('--gcsc_ckpt', default='task4/results/gcsc_checkpoint.pth')
    parser.add_argument('--proser_ckpt', default='task4/results/proser_checkpoint.pth')
    parser.add_argument('--cache_dir', default='task4/cache')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.cache_dir, exist_ok=True)
    
    train_loader, val_loader = get_train_val_loaders(args.data_root, randaugment=False)
    test_loader = get_test_loader(args.data_root)
    
    unknowns = CIFAR100Unknowns(args.data_root)
    near_loader, _ = unknowns.get_near_loader()
    far_loader, _ = unknowns.get_far_loader()
    
    splits = make_cifar10_splits(args.data_root)
    unaug_train_loader = get_train_loader_unaugmented(args.data_root, splits['train'])
    
    datasets = {
        'cifar10_train': train_loader,
        'cifar10_val': val_loader,
        'cifar10_test': test_loader,
        'near': near_loader,
        'far': far_loader
    }

    model = ResNetCIFAR(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.vanilla_ckpt))
    
    for name, loader in datasets.items():
        logits, features, labels = extract_features(model, loader, device)
        np.save(f"{args.cache_dir}/{name}_logits.npy", logits)
        np.save(f"{args.cache_dir}/{name}_features.npy", features)
        np.save(f"{args.cache_dir}/{name}_labels.npy", labels)
        
    _, features, labels = extract_features(model, unaug_train_loader, device)
    np.save(f"{args.cache_dir}/cifar10_train_unaugmented_features.npy", features)
    np.save(f"{args.cache_dir}/cifar10_train_unaugmented_labels.npy", labels)

    if os.path.exists(args.gcsc_ckpt):
        model.load_state_dict(torch.load(args.gcsc_ckpt))
        for name, loader in datasets.items():
            logits, _, _ = extract_features(model, loader, device)
            np.save(f"{args.cache_dir}/gcsc_{name}_logits.npy", logits)
            
    if os.path.exists(args.proser_ckpt):
        model.fc = torch.nn.Linear(512, 15).to(device)
        model.load_state_dict(torch.load(args.proser_ckpt))
        for name, loader in datasets.items():
            logits, _, _ = extract_features(model, loader, device, is_proser=True)
            np.save(f"{args.cache_dir}/proser_{name}_logits.npy", logits)

if __name__ == '__main__':
    main()
