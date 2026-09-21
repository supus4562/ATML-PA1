import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        self.feature_dim = 512
        # Replace fc with Identity
        model.fc = nn.Identity()
        self.features = model

    def forward(self, x):
        x = self.features(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
