import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
import torch

class PILDataset(Dataset):
    def __init__(self, pil_images, transform):
        self.pil_images = pil_images
        self.transform = transform
    def __len__(self):
        return len(self.pil_images)
    def __getitem__(self, idx):
        return self.transform(self.pil_images[idx]), 0

def cosine_stability(features_clean, features_transformed):
    # dot product for normalized, but let's be safe
    norms_clean = np.linalg.norm(features_clean, axis=1, keepdims=True)
    norms_trans = np.linalg.norm(features_transformed, axis=1, keepdims=True)
    
    clean_normed = features_clean / (norms_clean + 1e-8)
    trans_normed = features_transformed / (norms_trans + 1e-8)
    
    return np.mean(np.sum(clean_normed * trans_normed, axis=1))

def compute_all_stabilities(backbone, clean_pil_images, transform_fn, batch_size=64):
    transform = getattr(backbone, 'get_transform', lambda: backbone.transform)()
    
    clean_ds = PILDataset(clean_pil_images, transform)
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False)
    features_clean, _ = backbone.extract_features(clean_loader)
    
    trans_imgs = [transform_fn(img) for img in clean_pil_images]
    trans_ds = PILDataset(trans_imgs, transform)
    trans_loader = DataLoader(trans_ds, batch_size=batch_size, shuffle=False)
    features_trans, _ = backbone.extract_features(trans_loader)
    
    return cosine_stability(features_clean, features_trans), features_clean, features_trans
