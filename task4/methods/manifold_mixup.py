import torch
from torch.distributions.beta import Beta

def manifold_mixup_forward(model, x_i, x_j, alpha=2.0):
    lam = Beta(alpha, alpha).sample().item()
    
    h_i = model.conv1(x_i)
    h_i = model.bn1(h_i)
    h_i = model.relu(h_i)
    h_i = model.layer1(h_i)
    h_i = model.layer2(h_i)
    
    h_j = model.conv1(x_j)
    h_j = model.bn1(h_j)
    h_j = model.relu(h_j)
    h_j = model.layer1(h_j)
    h_j = model.layer2(h_j)
    
    h_mixed = lam * h_i + (1 - lam) * h_j
    
    logits = model.layer3(h_mixed)
    logits = model.layer4(logits)
    logits = model.avgpool(logits)
    logits = torch.flatten(logits, 1)
    logits = model.fc(logits)
    
    return logits, lam
