import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights
import numpy as np
from PIL import Image
import json
import os
from skimage.metrics import structural_similarity as ssim_fn

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(2): self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7): self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12): self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21): self.slice4.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h
        return [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1]

def calc_style_loss(input_features, target_features):
    loss = 0
    for inp, tgt in zip(input_features, target_features):
        b, c, h, w = inp.shape
        inp_gram = torch.bmm(inp.view(b, c, -1), inp.view(b, c, -1).transpose(1, 2)) / (c * h * w)
        tgt_gram = torch.bmm(tgt.view(b, c, -1), tgt.view(b, c, -1).transpose(1, 2)) / (c * h * w)
        loss += nn.MSELoss()(inp_gram, tgt_gram)
    return loss

def calc_content_loss(input_feat, target_feat):
    return nn.MSELoss()(input_feat[-1], target_feat[-1])

def generate_cue_conflicts(dataset, subset_indices, pairs, config, device):
    vgg = VGGEncoder().to(device).eval()
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    denorm = T.Compose([
        T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                    std=[1/0.229, 1/0.224, 1/0.225])
    ])
    
    classes = getattr(dataset, "classes", None)
    if classes is None:
        classes = [str(i) for i in range(max(dataset.targets)+1)]
    class_to_indices = {c: [] for c in range(len(classes))}
    for idx in subset_indices:
        img, label = dataset[idx]
        class_to_indices[label].append(idx)
        
    results = []
    
    # Map class name to label
    
    tasks = []
    class_to_idx = {name: i for i, name in enumerate(classes)}
    for c1_name, c2_name in pairs:
        c1 = class_to_idx[c1_name]
        c2 = class_to_idx[c2_name]
        for c_content, c_style in [(c1, c2), (c2, c1)]:
            for i in range(len(class_to_indices[c_content])):
                content_idx = class_to_indices[c_content][i]
                style_idx = class_to_indices[c_style][(i * 3 + 7) % len(class_to_indices[c_style])]
                tasks.append((c_content, c_style, content_idx, style_idx))
    
    batch_size = int(config.get('batch_size', 32))
    for chunk_start in range(0, len(tasks), batch_size):
        chunk = tasks[chunk_start:chunk_start+batch_size]
        
        content_tensors = []
        style_tensors = []
        for (c_content, c_style, content_idx, style_idx) in chunk:
            content_img, _ = dataset[content_idx]
            style_img, _ = dataset[style_idx]
            content_tensors.append(transform(content_img))
            style_tensors.append(transform(style_img))
            
        content_batch = torch.stack(content_tensors).to(device)
        style_batch = torch.stack(style_tensors).to(device)
        
        with torch.no_grad():
            # Use AMP for forward passes to reduce memory/compute on CUDA
            use_amp = (device.type == 'cuda')
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    content_feats = vgg(content_batch)
                    style_feats = vgg(style_batch)
            else:
                content_feats = vgg(content_batch)
                style_feats = vgg(style_batch)
            
        opt_batch = content_batch.clone().requires_grad_(True)
        optimizer = optim.Adam([opt_batch], lr=config['lr'])
        
        for step in range(int(config.get('n_steps', 200))):
            optimizer.zero_grad()
            # AMP for optimization forward as well
            if device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    opt_feats = vgg(opt_batch)
            else:
                opt_feats = vgg(opt_batch)
            c_loss = calc_content_loss(opt_feats, content_feats)
            s_loss = calc_style_loss(opt_feats, style_feats)
            loss = float(config['content_weight']) * c_loss + float(config['style_weight']) * s_loss
            loss.backward()
            optimizer.step()
            
        # Post-process
        final_batch = denorm(opt_batch.detach().cpu()).clamp(0, 1)
        orig_batch = denorm(content_batch.cpu()).clamp(0, 1)
        # Replace any NaN/Inf values before converting to PIL to avoid invalid casts
        final_batch = torch.nan_to_num(final_batch, nan=0.0, posinf=1.0, neginf=0.0)
        orig_batch = torch.nan_to_num(orig_batch, nan=0.0, posinf=1.0, neginf=0.0)
        
        for b in range(len(chunk)):
            c_content, c_style, content_idx, style_idx = chunk[b]
            
            final_img_tensor = final_batch[b]
            orig_img_tensor = orig_batch[b]
            
            final_pil = T.ToPILImage()(final_img_tensor)
            orig_pil = T.ToPILImage()(orig_img_tensor)
            
            # SSIM
            img1_np = np.array(final_pil)
            img2_np = np.array(orig_pil)
            ssim_val = ssim_fn(img1_np, img2_np, channel_axis=2, data_range=255)
            
            # Individual Style Loss
            with torch.no_grad():
                f_feats = vgg(transform(final_pil).unsqueeze(0).to(device))
                s_feats = vgg(style_tensors[b].unsqueeze(0).to(device))
                final_s_loss = calc_style_loss(f_feats, s_feats).item()
                
            results.append({
                'content_class': classes[c_content],
                'style_class': classes[c_style],
                'content_idx': content_idx,
                'style_idx': style_idx,
                'stylized_pil': final_pil,
                'style_loss': final_s_loss,
                'ssim': ssim_val
            })
            
        if len(results) >= config.get('max_generate_total', 1000):
            break
                    
    # Rejection
    all_s_loss = [r['style_loss'] for r in results]
    median_s_loss = np.median(all_s_loss)
    accepted = []
    for r in results:
        if r['style_loss'] <= 3 * median_s_loss and r['ssim'] >= 0.05:
            accepted.append(r)
            
    os.makedirs('task1/results', exist_ok=True)
    with open('task1/results/cue_conflict_stats.json', 'w') as f:
        json.dump({
            'total_generated': len(results),
            'accepted': len(accepted),
            'rejected': len(results) - len(accepted)
        }, f)
        
    return accepted
