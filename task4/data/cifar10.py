import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from task4.data.make_splits import make_cifar10_splits

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

def get_train_val_loaders(data_root, val_fraction=0.1, seed=6304, batch_size=128, randaugment=False, ra_ops=2, ra_mag=9):
    splits = make_cifar10_splits(data_root, val_fraction, seed)
    
    train_transform_list = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip()
    ]
    if randaugment:
        train_transform_list.append(T.RandAugment(num_ops=ra_ops, magnitude=ra_mag))
    
    train_transform_list.extend([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    
    train_transform = T.Compose(train_transform_list)
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    
    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    val_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=val_transform)
    
    train_subset = Subset(train_ds, splits['train'])
    val_subset = Subset(val_ds, splits['val'])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def get_test_loader(data_root, batch_size=256):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def get_train_loader_unaugmented(data_root, indices, batch_size=256):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    subset = Subset(ds, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
