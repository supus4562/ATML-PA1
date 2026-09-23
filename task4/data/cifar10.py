import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from task4.data.make_splits import make_cifar10_splits

# Use high-speed verified CDN mirror to bypass slow Toronto server throttling (~50 KB/s)
datasets.CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

def get_train_val_loaders(data_root, val_fraction=0.1, seed=6304, batch_size=2048, 
                           eval_batch_size=4096, randaugment=False, ra_ops=2, ra_mag=9, 
                           num_workers=8, pin_memory=None):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if eval_batch_size is None:
        eval_batch_size = batch_size
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
    
    persistent = (num_workers > 0)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_subset, batch_size=eval_batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent
    )
    
    return train_loader, val_loader

def get_test_loader(data_root, batch_size=4096, num_workers=8, pin_memory=None):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    persistent = (num_workers > 0)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent
    )

def get_train_loader_unaugmented(data_root, indices, batch_size=4096, num_workers=8, pin_memory=None):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    subset = Subset(ds, indices)
    persistent = (num_workers > 0)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent
    )
