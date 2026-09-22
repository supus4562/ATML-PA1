import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from task4.data.cifar10 import CIFAR_MEAN, CIFAR_STD

NEAR_CLASSES = {
    'bus': 19, 'pickup_truck': 58, 'motorcycle': 48, 'tractor': 89,
    'wolf': 97, 'fox': 34, 'leopard': 42, 'camel': 15
}
FAR_CLASSES = {
    'bottle': 9, 'bowl': 10, 'chair': 20, 'clock': 22,
    'keyboard': 39, 'mushroom': 51, 'sunflower': 84, 'wardrobe': 94
}

class CIFAR100Unknowns:
    def __init__(self, data_root):
        self.data_root = data_root
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
        self.ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=self.transform)
        
    def _filter_classes(self, class_dict, n_per_class):
        indices = []
        class_names = []
        
        target_indices = set(class_dict.values())
        counts = {idx: 0 for idx in target_indices}
        inv_class_dict = {v: k for k, v in class_dict.items()}
        
        # Fast targets indexing without evaluating image transforms
        targets = self.ds.targets
        for i, target in enumerate(targets):
            if target in target_indices and counts[target] < n_per_class:
                indices.append(i)
                counts[target] += 1
                class_names.append(inv_class_dict[target])
        
        subset = Subset(self.ds, indices)
        return subset, class_names

    def get_near_loader(self, batch_size=1024, n_per_class=100, num_workers=8, pin_memory=None):
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        subset, class_names = self._filter_classes(NEAR_CLASSES, n_per_class)
        persistent = (num_workers > 0)
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent
        )
        return loader, class_names

    def get_far_loader(self, batch_size=1024, n_per_class=100, num_workers=8, pin_memory=None):
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        subset, class_names = self._filter_classes(FAR_CLASSES, n_per_class)
        persistent = (num_workers > 0)
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent
        )
        return loader, class_names

    def get_all_unknowns_loader(self, batch_size=1024, n_per_class=100, num_workers=8, pin_memory=None):
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        all_classes = {**NEAR_CLASSES, **FAR_CLASSES}
        subset, class_names = self._filter_classes(all_classes, n_per_class)
        persistent = (num_workers > 0)
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent
        )
        return loader, class_names
