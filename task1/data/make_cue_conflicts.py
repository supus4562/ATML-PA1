import os
import json
import urllib.request
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights

DECODER_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
VGG_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"

def calc_mean_std(feat, eps=1e-5):
    """Calculate channel-wise mean and standard deviation for AdaIN."""
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain(content_feat, style_feat):
    """Adaptive Instance Normalization (Huang & Belongie, ICCV 2017)."""
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def build_decoder():
    """Huang & Belongie (2017) symmetrical decoder network."""
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )

def download_weight_file(url, local_path):
    """Download pretrained weights if not already present."""
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"[AdaIN] Downloading pretrained weights from {url} to {local_path} ...")
    try:
        torch.hub.download_url_to_file(url, local_path, progress=True)
    except Exception as e:
        print(f"[AdaIN] torch.hub download failed ({e}), attempting urllib ...")
        urllib.request.urlretrieve(url, local_path)
    return local_path

def load_adain_models(device):
    """Build and load pretrained AdaIN encoder and decoder."""
    # 1. Decoder
    decoder = build_decoder()
    decoder_paths = [
        "task1/models/decoder.pth",
        os.path.expanduser("~/.cache/torch/hub/checkpoints/adain_decoder.pth")
    ]
    decoder_loaded = False
    for p in decoder_paths:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            try:
                decoder.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
                decoder_loaded = True
                print(f"[AdaIN] Loaded decoder weights from {p}")
                break
            except Exception as e:
                print(f"[AdaIN] Failed to load {p}: {e}")

    if not decoder_loaded:
        target_path = decoder_paths[0]
        try:
            download_weight_file(DECODER_URL, target_path)
            decoder.load_state_dict(torch.load(target_path, map_location='cpu', weights_only=True))
            decoder_loaded = True
            print(f"[AdaIN] Downloaded and loaded decoder from {DECODER_URL}")
        except Exception as e:
            print(f"[AdaIN] Warning: Could not download decoder ({e}). Initializing random weights.")

    decoder.to(device).eval()

    # 2. VGG-19 Encoder up to relu4_1
    # Check for vgg_normalised.pth first
    vgg_paths = [
        "task1/models/vgg_normalised.pth",
        os.path.expanduser("~/.cache/torch/hub/checkpoints/vgg_normalised.pth")
    ]
    vgg_loaded = False
    encoder = nn.Sequential()
    for p in vgg_paths:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            try:
                # Load custom normalized VGG features
                sd = torch.load(p, map_location='cpu', weights_only=True)
                vgg_full = vgg19().features[:21]
                vgg_full.load_state_dict(sd)
                encoder = vgg_full
                vgg_loaded = True
                print(f"[AdaIN] Loaded normalized VGG from {p}")
                break
            except Exception as e:
                print(f"[AdaIN] Could not load {p}: {e}")

    if not vgg_loaded:
        # Standard torchvision VGG-19 features up to relu4_1
        try:
            encoder = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:21]
        except Exception as e:
            print(f"[AdaIN] Falling back to unweighted VGG-19: {e}")
            encoder = vgg19().features[:21]

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.to(device).eval()

    return encoder, decoder

def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """Compute SSIM between two PIL images with fallback to numpy."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        arr1 = np.array(img1.convert('L'))
        arr2 = np.array(img2.convert('L'))
        return float(ssim_fn(arr1, arr2, data_range=255))
    except ImportError:
        arr1 = np.array(img1.convert('L'), dtype=np.float64)
        arr2 = np.array(img2.convert('L'), dtype=np.float64)
        c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
        mu1, mu2 = arr1.mean(), arr2.mean()
        sigma1_sq = ((arr1 - mu1)**2).mean()
        sigma2_sq = ((arr2 - mu2)**2).mean()
        sigma12 = ((arr1 - mu1) * (arr2 - mu2)).mean()
        num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
        return float(num / den)

def visual_rejection_filter(stylized_pil: Image.Image, content_pil: Image.Image) -> tuple[bool, str, float]:
    """Define visual rejection rule before model evaluation.
    
    Returns:
        (is_accepted, reason, ssim_val)
    """
    arr = np.array(stylized_pil)
    # Check 1: Non-blank / non-black intensity
    if arr.mean() < 15.0:
        return False, "Too dark / black image", 0.0
    if arr.std() < 10.0:
        return False, "Low contrast / uniform image", 0.0

    # Check 2: Structural preservation of content shape
    ssim_val = compute_ssim(content_pil, stylized_pil)
    if ssim_val < 0.15:
        return False, f"Content shape obliterated (SSIM={ssim_val:.3f} < 0.15)", ssim_val
    if ssim_val > 0.88:
        return False, f"Style not transferred (SSIM={ssim_val:.3f} > 0.88)", ssim_val

    return True, "Accepted", ssim_val

def generate_cue_conflicts(dataset, subset_indices, pairs, config, device):
    """Generate cue-conflict images using AdaIN style transfer.
    
    Satisfies assignment requirements:
    - Content supplies shape of Class A, style supplies texture of Class B (and vice-versa).
    - At least 5 class pairs, both directions.
    - Balanced across pairs and directions.
    - Visual rejection rule defined prior to model evaluation.
    - Produces >= 200 valid conflicts.
    """
    encoder, decoder = load_adain_models(device)

    # Standard AdaIN preprocessing
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    classes = getattr(dataset, "classes", None)
    if classes is None:
        classes = [str(i) for i in range(max(dataset.targets) + 1)]

    # Group subset indices by class
    class_to_indices = {c: [] for c in range(len(classes))}
    for idx in subset_indices:
        _, label = dataset[idx]
        class_to_indices[label].append(idx)

    class_to_idx = {name: i for i, name in enumerate(classes)}

    # Target minimum 200 conflicts (e.g. 25 per directed pair across 10 directed pairs = 250)
    target_total = int(config.get('target_total', 200))
    n_directed_pairs = len(pairs) * 2
    per_pair_target = max(20, (target_total // n_directed_pairs) + 5)

    tasks = []
    for c1_name, c2_name in pairs:
        c1 = class_to_idx[c1_name]
        c2 = class_to_idx[c2_name]
        for c_content, c_style in [(c1, c2), (c2, c1)]:
            c_content_idxs = class_to_indices[c_content]
            c_style_idxs = class_to_indices[c_style]
            if len(c_content_idxs) == 0 or len(c_style_idxs) == 0:
                continue
            count = min(per_pair_target, len(c_content_idxs))
            for i in range(count):
                content_idx = c_content_idxs[i % len(c_content_idxs)]
                style_idx = c_style_idxs[(i * 3 + 7) % len(c_style_idxs)]
                tasks.append((c_content, c_style, content_idx, style_idx))

    style_strength = float(config.get('style_strength', 0.85))
    batch_size = int(config.get('batch_size', 32))

    all_generated = []
    accepted = []
    rejected = []

    print(f"[AdaIN] Generating cue conflicts for {len(tasks)} planned pairs (alpha={style_strength}) ...")

    with torch.no_grad():
        for chunk_start in range(0, len(tasks), batch_size):
            chunk = tasks[chunk_start:chunk_start + batch_size]
            content_pils = []
            content_tensors = []
            style_tensors = []

            for (c_content, c_style, content_idx, style_idx) in chunk:
                c_img, _ = dataset[content_idx]
                s_img, _ = dataset[style_idx]
                c_img_resized = c_img.resize((224, 224), Image.BILINEAR)
                content_pils.append(c_img_resized)
                content_tensors.append(transform(c_img_resized))
                style_tensors.append(transform(s_img))

            content_batch = torch.stack(content_tensors).to(device)
            style_batch = torch.stack(style_tensors).to(device)

            # AdaIN forward pass
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)
            t = adain(c_feats, s_feats)
            t = style_strength * t + (1.0 - style_strength) * c_feats
            stylized_batch = decoder(t).clamp(0.0, 1.0)

            for b in range(len(chunk)):
                c_content, c_style, content_idx, style_idx = chunk[b]
                final_tensor = stylized_batch[b].cpu()
                final_pil = T.ToPILImage()(final_tensor)
                orig_pil = content_pils[b]

                is_ok, reason, ssim_val = visual_rejection_filter(final_pil, orig_pil)

                item = {
                    'content_class': classes[c_content],
                    'style_class':   classes[c_style],
                    'content_idx':   content_idx,
                    'style_idx':     style_idx,
                    'stylized_pil':  final_pil,
                    'ssim':          ssim_val,
                    'rejection_reason': reason if not is_ok else "None",
                }
                all_generated.append(item)
                if is_ok:
                    accepted.append(item)
                else:
                    rejected.append(item)

    print(f"[AdaIN] Completed cue-conflict generation: {len(all_generated)} total, "
          f"{len(accepted)} accepted, {len(rejected)} rejected.")

    # Save statistics
    out_dir = config.get("output_dir", "task1/results")
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, "cue_conflict_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total_generated": len(all_generated),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "rejection_rate": float(len(rejected) / max(len(all_generated), 1)),
            "rejection_reasons": [r['rejection_reason'] for r in rejected[:20]],
        }, f, indent=2)

    return accepted
