import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from task1.data.transforms import apply_grayscale, apply_hue_rotation, apply_translation, apply_patch_shuffle
from torch.utils.data import DataLoader, Dataset
import torch

class PILDataset(Dataset):
    def __init__(self, pil_images, labels, transform):
        self.pil_images = pil_images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.pil_images)
    def __getitem__(self, idx):
        return self.transform(self.pil_images[idx]), self.labels[idx]

def evaluate_clean(model, dataloader, config):
    preds, probs, labels = model.predict(dataloader)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    mean_max_conf = np.mean(np.max(probs, axis=1))
    return {'accuracy': acc, 'macro_f1': macro_f1, 'mean_max_conf': float(mean_max_conf)}

def evaluate_color_bias(model, clean_pil_images, labels, config):
    transform = getattr(model, 'get_transform', lambda: model.transform)()
    
    gray_imgs = [apply_grayscale(img) for img in clean_pil_images]
    gray_ds = PILDataset(gray_imgs, labels, transform)
    gray_loader = DataLoader(gray_ds, batch_size=config.get("batch_size", 1024), shuffle=False, num_workers=config.get("num_workers", 8), pin_memory=True)
    gray_preds, _, _ = model.predict(gray_loader)
    
    hue_imgs = [apply_hue_rotation(img) for img in clean_pil_images]
    hue_ds = PILDataset(hue_imgs, labels, transform)
    hue_loader = DataLoader(hue_ds, batch_size=config.get("batch_size", 1024), shuffle=False, num_workers=config.get("num_workers", 8), pin_memory=True)
    hue_preds, _, _ = model.predict(hue_loader)
    
    # get clean preds for consistency
    clean_ds = PILDataset(clean_pil_images, labels, transform)
    clean_loader = DataLoader(clean_ds, batch_size=config.get("batch_size", 1024), shuffle=False, num_workers=config.get("num_workers", 8), pin_memory=True)
    clean_preds, _, _ = model.predict(clean_loader)
    
    return {
        'grayscale_acc': accuracy_score(labels, gray_preds),
        'grayscale_consistency': np.mean(clean_preds == gray_preds),
        'hue_acc': accuracy_score(labels, hue_preds),
        'hue_consistency': np.mean(clean_preds == hue_preds)
    }

def evaluate_cue_conflicts(model, conflicts, config):
    if not conflicts:
        return {'shape_bias': 0, 'coverage': 0}
        
    transform = getattr(model, 'get_transform', lambda: model.transform)()
    # We assume 'classes' is passed via config or we extract it from model predictions if available.
    # To prevent breakage, we'll pass classes dynamically if needed, or rely on model class names.
    classes = config.get('classes', None)
    if classes is None:
        # Fallback to model's default classes if available
        classes = getattr(model, 'classes', None)
        # Or just use the string predictions directly if model.predict returns strings

    
    imgs = [c['stylized_pil'] for c in conflicts]
    ds = PILDataset(imgs, [0]*len(imgs), transform)
    loader = DataLoader(ds, batch_size=config.get("batch_size", 1024), shuffle=False, num_workers=config.get("num_workers", 8), pin_memory=True)
    
    preds, _, _ = model.predict(loader)
    
    n_shape = 0
    n_texture = 0
    
    for i, p in enumerate(preds):
        pred_class = classes[p]
        if pred_class == conflicts[i]['content_class']:
            n_shape += 1
        elif pred_class == conflicts[i]['style_class']:
            n_texture += 1
            
    shape_bias = n_shape / (n_shape + n_texture) * 100 if (n_shape + n_texture) > 0 else 0
    coverage = (n_shape + n_texture) / len(conflicts) * 100
    
    return {'shape_bias': shape_bias, 'coverage': coverage}

def evaluate_translation(model, clean_pil_images, labels, config):
    transform = getattr(model, 'get_transform', lambda: model.transform)()
    
    clean_ds = PILDataset(clean_pil_images, labels, transform)
    clean_loader = DataLoader(clean_ds, batch_size=config.get("batch_size", 1024), shuffle=False, num_workers=config.get("num_workers", 8), pin_memory=True)
    clean_preds, _, _ = model.predict(clean_loader)
    
    displacements = config['translation']['displacements']
    directions = config['translation']['directions']
    
    results = {}
    for d in displacements:
        if d == 0:
            results[str(d)] = {'acc': accuracy_score(labels, clean_preds), 'consistency': 1.0}
            continue
            
        all_d_preds = []
        for direction in directions:
            imgs = [apply_translation(img, d, direction) for img in clean_pil_images]
            ds = PILDataset(imgs, labels, transform)
            loader = DataLoader(ds, batch_size=config.get("batch_size", 1024), shuffle=False, num_workers=config.get("num_workers", 8), pin_memory=True)
            preds, _, _ = model.predict(loader)
            all_d_preds.append(preds)
            
        # average consistency and acc across directions
        consistencies = [np.mean(clean_preds == p) for p in all_d_preds]
        accs = [accuracy_score(labels, p) for p in all_d_preds]
        
        results[str(d)] = {'acc': np.mean(accs), 'consistency': np.mean(consistencies)}
        
    return results

def evaluate_patch_shuffle(model, clean_pil_images, labels, config):
    transform = getattr(model, 'get_transform', lambda: model.transform)()
    
    np.random.seed(config.get('seed', 6304))
    perm = np.random.permutation(16).tolist()
    
    clean_ds = PILDataset(clean_pil_images, labels, transform)
    clean_loader = DataLoader(clean_ds, batch_size=config.get("batch_size", 1024), shuffle=False, num_workers=config.get("num_workers", 8), pin_memory=True)
    clean_preds, _, _ = model.predict(clean_loader)
    
    shuffled = [apply_patch_shuffle(img, perm) for img in clean_pil_images]
    ds = PILDataset(shuffled, labels, transform)
    loader = DataLoader(ds, batch_size=config.get("batch_size", 1024), shuffle=False, num_workers=config.get("num_workers", 8), pin_memory=True)
    preds, _, _ = model.predict(loader)
    
    return {
        'acc': accuracy_score(labels, preds),
        'consistency': np.mean(clean_preds == preds)
    }
