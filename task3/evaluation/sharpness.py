import torch
import torch.nn as nn
import math
from common.seed import set_all_seeds

def compute_sharpness(backbone, classifier, source_val_loaders, device, rho=0.05, n_per_domain=32, seed=6304):
    set_all_seeds(seed)
    
    X_list, y_list = [], []
    for loader in source_val_loaders.values():
        n_collected = 0
        for x, y in loader:
            take = min(n_per_domain - n_collected, x.size(0))
            X_list.append(x[:take])
            y_list.append(y[:take])
            n_collected += take
            if n_collected >= n_per_domain:
                break
                
    X = torch.cat(X_list).to(device)
    y = torch.cat(y_list).to(device)
    
    backbone.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    
    with torch.enable_grad():
        logits = classifier(backbone(X))
        loss = criterion(logits, y)
        loss.backward()
        
    global_norm = 0.0
    for param in list(backbone.parameters()) + list(classifier.parameters()):
        if param.grad is not None:
            global_norm += param.grad.norm(2).item() ** 2
    global_norm = math.sqrt(global_norm)
    
    scale = rho / (global_norm + 1e-12)
    
    eps = {}
    for name, param in backbone.named_parameters():
        if param.grad is not None:
            e = param.grad * scale
            eps[f"backbone_{name}"] = e
            param.data.add_(e)
    for name, param in classifier.named_parameters():
        if param.grad is not None:
            e = param.grad * scale
            eps[f"classifier_{name}"] = e
            param.data.add_(e)
            
    with torch.no_grad():
        logits2 = classifier(backbone(X))
        loss2 = criterion(logits2, y)
        
    for name, param in backbone.named_parameters():
        if param.grad is not None:
            param.data.sub_(eps[f"backbone_{name}"])
    for name, param in classifier.named_parameters():
        if param.grad is not None:
            param.data.sub_(eps[f"classifier_{name}"])
            
    return (loss2 - loss).item()
