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

def compute_all_stabilities(backbone, clean_pil_images, transform_fn, batch_size=64, num_workers=None, pin_memory=None):
    transform = getattr(backbone, 'get_transform', lambda: backbone.transform)()

    if num_workers is None:
        num_workers = 8
    if pin_memory is None:
        pin_memory = True if torch.cuda.is_available() else False

    clean_ds = PILDataset(clean_pil_images, transform)
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    features_clean, _ = backbone.extract_features(clean_loader)

    trans_imgs = [transform_fn(img) for img in clean_pil_images]
    trans_ds = PILDataset(trans_imgs, transform)
    trans_loader = DataLoader(trans_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    features_trans, _ = backbone.extract_features(trans_loader)
    
    return cosine_stability(features_clean, features_trans), features_clean, features_trans

def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    num = np.linalg.norm(X.T @ Y, ord='fro') ** 2
    den = np.linalg.norm(X.T @ X, ord='fro') * np.linalg.norm(Y.T @ Y, ord='fro')
    return float(num / den) if den > 0 else 0.0
