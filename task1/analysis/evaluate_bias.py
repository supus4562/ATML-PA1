import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from task1.data.transforms import apply_grayscale, apply_hue_rotation, apply_translation, apply_patch_shuffle
from torch.utils.data import DataLoader, Dataset
import torch
import math

class PILDataset(Dataset):
    def __init__(self, pil_images, labels, transform):
        self.pil_images = pil_images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.pil_images)
    def __getitem__(self, idx):
        return self.transform(self.pil_images[idx]), self.labels[idx]

def _dl_kwargs(config):
    """Build DataLoader kwargs from the top-level config dict."""
    return {
        'batch_size':  int(config.get('batch_size', 128)),
        'num_workers': int(config.get('num_workers', 8)),
        'pin_memory':  torch.cuda.is_available(),
        'shuffle':     False,
    }

def evaluate_clean(model, dataloader, config):
    preds, probs, labels = model.predict(dataloader)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    mean_max_conf = np.mean(np.max(probs, axis=1))
    return {'accuracy': float(acc), 'macro_f1': float(macro_f1), 'mean_max_conf': float(mean_max_conf)}

def evaluate_color_bias(model, clean_pil_images, labels, config):
    transform = getattr(model, 'get_transform', lambda: model.transform)()
    kw = _dl_kwargs(config)

    out_dir = os.path.join(config.get("output_dir", "task1/results"), "report_images")
    os.makedirs(out_dir, exist_ok=True)

    # --- Grayscale ---
    gray_imgs = [apply_grayscale(img) for img in clean_pil_images]
    for i in range(min(5, len(gray_imgs))):
        gray_imgs[i].save(os.path.join(out_dir, f"grayscale_{i}.jpg"))
        clean_pil_images[i].save(os.path.join(out_dir, f"original_clean_{i}.jpg"))
    gray_preds, _, _ = model.predict(DataLoader(PILDataset(gray_imgs, labels, transform), **kw))

    # --- Hue ---
    hue_imgs = [apply_hue_rotation(img) for img in clean_pil_images]
    for i in range(min(5, len(hue_imgs))):
        hue_imgs[i].save(os.path.join(out_dir, f"hue_{i}.jpg"))
    hue_preds, _, _ = model.predict(DataLoader(PILDataset(hue_imgs, labels, transform), **kw))

    # --- Clean (for consistency) ---
    clean_preds, _, _ = model.predict(DataLoader(PILDataset(clean_pil_images, labels, transform), **kw))

    return {
        'grayscale_acc':         float(accuracy_score(labels, gray_preds)),
        'grayscale_consistency': float(np.mean(clean_preds == gray_preds)),
        'hue_acc':               float(accuracy_score(labels, hue_preds)),
        'hue_consistency':       float(np.mean(clean_preds == hue_preds)),
    }

def evaluate_cue_conflicts(model, conflicts, config):
    if not conflicts:
        return {'shape_bias': 0.0, 'coverage': 0.0, 'n_shape': 0, 'n_texture': 0, 'n_other': 0, 'n_total': 0}

    transform = getattr(model, 'get_transform', lambda: model.transform)()
    kw = _dl_kwargs(config)

    classes = config.get('classes', None)
    if classes is None:
        classes = getattr(model, 'classes', None)

    imgs   = [c['stylized_pil'] for c in conflicts]
    preds, _, _ = model.predict(DataLoader(PILDataset(imgs, [0]*len(imgs), transform), **kw))

    n_shape = n_texture = n_other = 0
    for i, p in enumerate(preds):
        pred_class = classes[p] if classes is not None and p < len(classes) else str(p)
        if pred_class == conflicts[i]['content_class']:
            n_shape += 1
        elif pred_class == conflicts[i]['style_class']:
            n_texture += 1
        else:
            n_other += 1

    total = n_shape + n_texture
    shape_bias = (n_shape / total * 100) if total > 0 else 0.0
    coverage   = (total / len(conflicts) * 100)
    return {
        'shape_bias': float(shape_bias),
        'coverage':   float(coverage),
        'n_shape':    int(n_shape),
        'n_texture':  int(n_texture),
        'n_other':    int(n_other),
        'n_total':    len(conflicts),
    }

def evaluate_translation(model, clean_pil_images, labels, config):
    transform = getattr(model, 'get_transform', lambda: model.transform)()
    kw = _dl_kwargs(config)

    out_dir = os.path.join(config.get("output_dir", "task1/results"), "report_images")
    os.makedirs(out_dir, exist_ok=True)

    clean_preds, _, _ = model.predict(DataLoader(PILDataset(clean_pil_images, labels, transform), **kw))

    displacements = config['translation']['displacements']
    directions    = config['translation']['directions']

    results = {}
    for d in displacements:
        if d == 0:
            results[str(d)] = {'acc': float(accuracy_score(labels, clean_preds)), 'consistency': 1.0}
            continue

        all_d_preds = []
        for direction in directions:
            imgs = [apply_translation(img, d, direction) for img in clean_pil_images]
            # Save a couple of examples for the report (only once per displacement/direction)
            for i in range(min(2, len(imgs))):
                imgs[i].save(os.path.join(out_dir, f"translation_d{d}_{direction}_{i}.jpg"))
            preds, _, _ = model.predict(DataLoader(PILDataset(imgs, labels, transform), **kw))
            all_d_preds.append(preds)

        consistencies = [float(np.mean(clean_preds == p)) for p in all_d_preds]
        accs          = [float(accuracy_score(labels, p)) for p in all_d_preds]
        results[str(d)] = {'acc': float(np.mean(accs)), 'consistency': float(np.mean(consistencies))}

    return results

def evaluate_patch_shuffle(model, clean_pil_images, labels, config, shuffled_pil_images=None):
    transform = getattr(model, 'get_transform', lambda: model.transform)()
    kw = _dl_kwargs(config)

    clean_preds, _, _ = model.predict(DataLoader(PILDataset(clean_pil_images, labels, transform), **kw))

    if shuffled_pil_images is None:
        seed = config.get('seed', 6304)
        shuffled_pil_images = []
        for idx, img in enumerate(clean_pil_images):
            rng = np.random.RandomState(seed + idx)
            perm = rng.permutation(16).tolist()
            while perm == list(range(16)):
                perm = rng.permutation(16).tolist()
            shuffled_pil_images.append(apply_patch_shuffle(img, perm))

    out_dir = os.path.join(config.get("output_dir", "task1/results"), "report_images")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(5, len(shuffled_pil_images))):
        shuffled_pil_images[i].save(os.path.join(out_dir, f"patch_shuffle_{i}.jpg"))

    preds, _, _ = model.predict(DataLoader(PILDataset(shuffled_pil_images, labels, transform), **kw))

    return {
        'acc':         float(accuracy_score(labels, preds)),
        'consistency': float(np.mean(clean_preds == preds)),
    }
